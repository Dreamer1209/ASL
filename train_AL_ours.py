import argparse
import logging
import os
import random
import shutil
import sys
import time
os.environ['CUDA_VISIBLE_DEVICES'] = '5,6'
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn import BCEWithLogitsLoss
from torch.nn.modules.loss import CrossEntropyLoss, MSELoss
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm
# from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from dataloaders import utils
from dataloaders.brats2019 import (BraTS2019, CenterCrop, RandomCrop,
                                   RandomRotFlip, ToTensor,
                                   TwoStreamBatchSampler)
from dataloaders.LA_AL import LeftArium
from dataloaders.FeTA_AL import FeTA
from dataloaders.pancreas import Pancreas
from dataloaders.dataset import WeakStrongAugment
from networks.net_factory_3d import net_factory_3d
from utils import losses, metrics, ramps, ActiveLearning, update, Pseudo_Correction
from utils.GMM import GaussianMixture
from val_3D import test_all_case
from dataloaders import utils
from utils.PoolEnhance import poolenhance
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='/data/kw/fetah5/', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='test', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='voxresnet', help='model_name')
parser.add_argument('--max_iterations', type=int,
                    default=5000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=4,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list,  default=[112, 112, 80],
                    help='patch size of network input')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--applybank', type=bool,  default=True, help='whether use active bank')

# label and unlabel
parser.add_argument('--labeled_bs', type=int, default=2,
                    help='labeled_batch_size per gpu')
parser.add_argument('--labeled_num', type=int, default=6,
                    help='labeled data')
parser.add_argument('--initial_labeled_num', type=int, default=4,
                    help='labeled data')
# costs
parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str,
                    default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float,
                    default=0.1, help='consistencypseudoLabels')
parser.add_argument('--consistency_rampup', type=float,
                    default=200.0, help='consistency_rampup')

# new add
parser.add_argument('--gpu', type=str, default='1,2,3', help='GPU to use')
parser.add_argument('--dataset', type=str, default='FeTA', help='Dataset to use')
parser.add_argument('--pseudoLabels', action="store_true")
parser.add_argument('--threshold', type=float, default=0.9, help='Threshold for denoising')
parser.add_argument('--k', type=int, default=1, help='Select Top k as labeled')



args = parser.parse_args()


if args.dataset != 'LA':
    args.patch_size = [96, 96, 96]

if args.dataset == 'LA':
    args.initial_labeled_num = 4
elif args.dataset == 'FeTA':
    args.initial_labeled_num = 2

def iou(pred, target, n_classes = 37):
#n_classes ：the number of classes in your dataset
  ious = []
  pred = pred.view(-1)
  target = target.view(-1)

  # Ignore IoU for background class ("0")
  for cls in range(0, n_classes):  # This goes from 1:n_classes-1 -> class "0" is ignored
    pred_inds = pred == cls
    target_inds = target == cls
    # intersection = (pred_inds[target_inds]).long().sum().data.cpu()[0]  # Cast to long to prevent overflows
    # union = pred_inds.long().sum().data.cpu()[0] + target_inds.long().sum().data.cpu()[0] - intersection
    TP = pred_inds * target_inds
    FP = pred_inds.sum() - TP.sum()
    FN = target_inds.sum() - TP.sum()
    if (TP.sum() + FP + FN) == 0:
        continue
    ans = TP.sum() / (TP.sum() + FP + FN)
    ious.append(float(ans))
    # print("cls",cls,"ans", ans)
  return np.array(ious)

def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def train(args, snapshot_path):
    os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3'
    base_lr = args.base_lr
    train_data_path = args.root_path
    batch_size = args.batch_size
    max_iterations = args.max_iterations
    if args.dataset == 'FeTA':
        num_classes = 8
    else:
        num_classes = 2

    print(snapshot_path)

    file = open(snapshot_path + 'train_results.txt', "a+")
    file.write(str(args) + '\n')
    file.close()

    def create_model(ema=False):
        # Network definition
        net = net_factory_3d(net_type=args.model, in_chns=1, class_num=num_classes)
        model = torch.nn.DataParallel(net)      # multi GPU
        model = model.cuda()
        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    model = create_model()
    ema_model = create_model(ema=True)

    # dataset

    train_path = args.root_path + '/train.txt'
    print(train_path)
    with open(train_path, 'r') as f:
        image_list = f.readlines()
    train_list = [item.replace('\n', '') for item in image_list]
    print(len(train_list))

    labeled_list = train_list[0:args.initial_labeled_num]
    unlabeled_list = train_list[args.initial_labeled_num:]
    print(len(labeled_list), len(unlabeled_list))

    file = open(snapshot_path + 'train_list_vary.txt', "a+")
    file.write(str(args) + '\n')
    file.write('labeled_list' + '\n')
    for i in range(len(labeled_list)):
        file.write(labeled_list[i] + '    ')
    file.write('\n'+ 'unlabeled_list' + '\n')
    for i in range(len(unlabeled_list)):
        file.write(unlabeled_list[i] + '   ')
    file.write('\n')
    file.close()

    if args.dataset == 'BraTS':
        db_train = BraTS2019(base_dir=train_data_path,
                             split='train',
                             num=None,
                             transform=transforms.Compose([
                                 RandomRotFlip(),
                                 RandomCrop(args.patch_size),
                                 ToTensor(),
                             ]))
    elif args.dataset == 'LA':
        train_labeled_dataset = LeftArium(base_dir=train_data_path,
                                          split='train',
                                          num=None,
                                          transform=WeakStrongAugment(args.patch_size),
                                          train_list=labeled_list)
        train_unlabeled_dataset = LeftArium(base_dir=train_data_path,
                                            split='train',
                                            num=None,
                                            transform=WeakStrongAugment(args.patch_size),
                                            train_list=unlabeled_list)
    elif args.dataset == 'FeTA':
        train_labeled_dataset = FeTA(base_dir=train_data_path,
                                          split='train',
                                          num=None,
                                          transform=WeakStrongAugment(args.patch_size),
                                          train_list=labeled_list)
        train_unlabeled_dataset = FeTA(base_dir=train_data_path,
                                            split='train',
                                            num=None,
                                            transform=WeakStrongAugment(args.patch_size),
                                            train_list=unlabeled_list)
    elif args.dataset == 'PA':
        train_labeled_dataset = Pancreas(base_dir=train_data_path,
                                     split='train',
                                     num=None,
                                     transform=WeakStrongAugment(args.patch_size),
                                     train_list=labeled_list)
        train_unlabeled_dataset = Pancreas(base_dir=train_data_path,
                                       split='train',
                                       num=None,
                                       transform=WeakStrongAugment(args.patch_size),
                                       train_list=unlabeled_list)

    labeled_trainloader = DataLoader(train_labeled_dataset, batch_size=args.labeled_bs, shuffle=True,
                             num_workers=2, drop_last=True)
    unlabeled_trainloader = DataLoader(train_unlabeled_dataset, batch_size=args.batch_size-args.labeled_bs, shuffle=True,
                                     num_workers=2, drop_last=True)
    print(len(unlabeled_trainloader))

    model = model.cuda()
    ema_model = ema_model.cuda()
    model.train()
    ema_model.train()

    if args.model == 'unet_3D' or args.model == 'attention_unet':
        optimizer = optim.Adam(model.parameters(), lr=base_lr)
    else:
        optimizer = optim.SGD(model.parameters(), lr=base_lr,
                              momentum=0.9, weight_decay=0.0001)
    # optimizer = optim.SGD(model.parameters(), lr=base_lr,
    #                       momentum=0.9, weight_decay=0.0001)
    ce_loss = CrossEntropyLoss()
    sce_loss = losses.SCELoss(alpha=0.5, beta=1, num_classes=num_classes)
    bace_loss = losses.BaCELoss(alpha=0.1, beta=1, num_classes=num_classes, reduction = "mean")
    norm_bace_loss = losses.BaCELoss(alpha=0.1, beta=1, num_classes=num_classes, reduction = "mean")
    gmm = GaussianMixture(n_components=2, n_features=1)
    pool = poolenhance(num_class = num_classes, pool_size = (args.patch_size[0], args.patch_size[1],args.patch_size[2]))
    gmm = gmm.cuda()
    sum_ce_loss = CrossEntropyLoss(reduction='sum')
    sum_mse_loss = MSELoss(reduction='sum')

    if args.dataset == 'FeTA':
        dice_loss = losses.DiceLoss(8)
    else:
        dice_loss = losses.DiceLoss(2)

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(unlabeled_trainloader)))

    iter_num = 0
    max_epoch = max_iterations // (len(unlabeled_trainloader) + len(labeled_trainloader)) + 1
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    lr_ = base_lr
    score_memory = dict()
    predict_history = dict()
    if args.dataset == 'LA':
        score_function = ActiveLearning.FloatingRegionScore(in_channels=2, padding_mode='zeros', size=1)
    if args.dataset == 'FeTA':
        score_function = ActiveLearning.FloatingRegionScore(in_channels=8, padding_mode='zeros', size=1)
    # 记录计算score的epoch
    count = 0
    current_epoch_iou = 0
    start_epoch_now = 1
    final_update = False
    total_iou_record = []

    for epoch_num in iterator:
        labeled_train_iter = iter(labeled_trainloader)
        unlabeled_train_iter = iter(unlabeled_trainloader)

        for i_batch in range(len(unlabeled_trainloader)):
            # print(i_batch)
            # labeled data
            try:
                sampled_batch = labeled_train_iter.next()
            except:
                labeled_train_iter = iter(labeled_trainloader)
                sampled_batch = labeled_train_iter.next()
            # unlabeled data
            try:
                unlabeled_sampled_batch = unlabeled_train_iter.next()
            except:
                unlabeled_train_iter = iter(unlabeled_trainloader)
                unlabeled_sampled_batch = unlabeled_train_iter.next()

            volume_batch_image, volume_batch_weak, volume_batch_strong, label_batch = \
                sampled_batch['image'], sampled_batch['image_weak'], sampled_batch['image_strong'], sampled_batch['label']
            volume_batch_image, volume_batch_weak, volume_batch_strong, label_batch = \
                volume_batch_image.cuda(), volume_batch_weak.cuda(), volume_batch_strong.cuda(), label_batch.cuda()
            unlabeled_volume_batch_image,unlabeled_volume_batch_weak,unlabeled_volume_batch_strong, unlabeled_label_batch = \
                unlabeled_sampled_batch['image'], unlabeled_sampled_batch['image_weak'], unlabeled_sampled_batch['image_strong'], unlabeled_sampled_batch['label']
            unlabeled_volume_batch_image, unlabeled_volume_batch_weak, unlabeled_volume_batch_strong, unlabeled_label_batch = \
                unlabeled_volume_batch_image.cuda(),unlabeled_volume_batch_weak.cuda(),unlabeled_volume_batch_strong.cuda(), unlabeled_label_batch.cuda()

            volume_batch_image_name = sampled_batch['name']
            unlabeled_volume_batch_image_name = unlabeled_sampled_batch['name']

            noise = torch.clamp(torch.randn_like(
                unlabeled_volume_batch_weak) * 0.1, -0.2, 0.2)
            ema_inputs = unlabeled_volume_batch_weak + noise

            # labeled data fed into student model
            labeled_outputs = model(volume_batch_strong)
            labeled_outputs_soft = torch.softmax(labeled_outputs, dim=1)

            # unlabeled data fed into student model
            unlabeled_outputs = model(unlabeled_volume_batch_strong)
            unlabeled_outputs_soft = torch.softmax(unlabeled_outputs, dim=1)


            # unlabeled data fed into teacher model
            with torch.no_grad():
                ema_output = ema_model(ema_inputs)
                #correction
                ema_output = pool(ema_output)

                # if iter_num % 20 == 0:
                #     predict = torch.softmax(ema_output, dim=1).long()
                    # xxx = predict[0, :, :, 20:61:10].permute(0,3,1,2).cpu().numpy()
                    # for i in range(5):
                    #     image = xxx[i,:,:]
                    #     print(image.shape)
                    #     image = cv2.applyColorMap(image, cv2.COLORMAP_JET)
                    #     grid_image = make_grid(image, 5, normalize=True)
                    #     writer.add_image('train/before'+i, grid_image, iter_num)

                ema_output_soft = torch.softmax(ema_output, dim=1)
                if args.pseudoLabels:
                    ema_score = torch.max(ema_output, dim=1).values
                    pseudo_labels = torch.argmax(ema_output_soft, dim=1).long()
                    pseudo_labels[ema_score < args.threshold] = -1 # labeling noise pseudo labels as -1
                else:
                    pseudo_labels = torch.argmax(ema_output_soft, dim=1).long() # pseudo labels for unlabeled data
                error = (pseudo_labels != unlabeled_label_batch) # noise pseudo labels
                error = error.long()

                supervised_pred = torch.argmax(labeled_outputs_soft, axis=1)
                supervised_iou = iou(supervised_pred, label_batch, num_classes).mean()
                unsupervised_pred = torch.argmax(unlabeled_outputs_soft, axis=1)
                unsupervised_iou = iou(unsupervised_pred, pseudo_labels, num_classes).mean()
                current_epoch_iou = (supervised_iou + unsupervised_iou) / 2
                writer.add_scalar('iou/supervised_iou', supervised_iou, iter_num)
                writer.add_scalar('iou/unsupervised_iou', unsupervised_iou, iter_num)
                writer.add_scalar('iou/current_epoch_iou', current_epoch_iou, iter_num)
                # iou calculaiton
                if len(labeled_list) <= args.labeled_num and final_update == False:
                    print("su", supervised_iou, "unsu", unsupervised_iou, "curr", current_epoch_iou)
                    total_iou_record.append(current_epoch_iou)



            # Calculate score
            # print("111",len(labeled_list))
            # print("222",args.labeled_num)
            if epoch_num >= 0  and len(labeled_list) < args.labeled_num and args.applybank: # 开始计算每个epoch中，每个样本的不确定性度量
                for i in range(ema_output.shape[0]):
                    scorei, _, _ = score_function(ema_output[i])
                    if unlabeled_volume_batch_image_name[i] in score_memory:
                        pre = score_memory[unlabeled_volume_batch_image_name[i]]
                        score_memory[unlabeled_volume_batch_image_name[i]] = torch.cat((pre, scorei.unsqueeze(0)), dim=0)
                    else:
                        score_memory[unlabeled_volume_batch_image_name[i]] = scorei.unsqueeze(0)
                    # print('size', score_memory[unlabeled_volume_batch_image_name[i]].shape)
            loss_ce = ce_loss(labeled_outputs_soft,label_batch).mean()

            target = torch.argmax(unlabeled_outputs_soft, axis=1)
            # print(target.shape)
            # print(label_batch.shape)
            # iou_arr = iou(target,unlabeled_label_batch,8)
            # writer.add_scalar('label/w1', iou_arr[0], iter_num)
            # writer.add_scalar('label/w2', iou_arr[1], iter_num)
            # writer.add_scalar('label/w3', iou_arr[2], iter_num)
            # writer.add_scalar('label/w4', iou_arr[3], iter_num)
            # writer.add_scalar('label/w5', iou_arr[4], iter_num)
            # writer.add_scalar('label/w6', iou_arr[5], iter_num)
            # writer.add_scalar('label/w7', iou_arr[6], iter_num)
            # writer.add_scalar('label/w8', iou_arr[7], iter_num)
            #
            # iou_arr2 = iou(target, pseudo_labels, 8)
            # writer.add_scalar('plabel/w1', iou_arr2[0], iter_num)
            # writer.add_scalar('plabel/w2', iou_arr2[1], iter_num)
            # writer.add_scalar('plabel/w3', iou_arr2[2], iter_num)
            # writer.add_scalar('plabel/w4', iou_arr2[3], iter_num)
            # writer.add_scalar('plabel/w5', iou_arr2[4], iter_num)
            # writer.add_scalar('plabel/w6', iou_arr2[5], iter_num)
            # writer.add_scalar('plabel/w7', iou_arr2[6], iter_num)
            # writer.add_scalar('plabel/w8', iou_arr2[7], iter_num)

            loss_dice = dice_loss(labeled_outputs_soft, label_batch.unsqueeze(1))
            supervised_loss = 0.5 * (loss_dice + loss_ce)

            consistency_weight = get_current_consistency_weight(iter_num // 150)
            # consistency_loss = norm_bace_loss(unlabeled_outputs_soft, ema_output_soft)
            pseudo_labels = pseudo_labels.long()
            consistency_loss = ce_loss(unlabeled_outputs_soft, pseudo_labels)
            unsupervised_loss = consistency_weight * consistency_loss
            loss = supervised_loss + unsupervised_loss

            # loss = supervised_loss + unsupervised_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            update_ema_variables(model, ema_model, args.ema_decay, iter_num)
            lr_ = max(base_lr * (1.0 - iter_num / max_iterations) ** 0.9, 1e-8)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_
            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            writer.add_scalar('info/loss_dice', loss_dice, iter_num)
            # writer.add_scalar('info/consistency_loss',
            #                   consistency_loss, iter_num)
            # writer.add_scalar('info/consistency_weight',
            #                   consistency_weight, iter_num)
            writer.add_scalar('info/error',
                              torch.sum(error) / error.numel(), iter_num)

            logging.info(
                'iteration %d : loss : %f, loss_ce: %f, loss_dice: %f, error: %f lr: %f' %
                (iter_num, loss.item(), loss_ce.item(), loss_dice.item(), torch.sum(error) / error.numel(), lr_))
            writer.add_scalar('loss/loss', loss, iter_num)

            # if iter_num % 20 == 0:
            #     image = volume_batch_image[0, 0:1, :, :, 20:61:10].permute(
            #         3, 0, 1, 2).repeat(1, 3, 1, 1)
            #     grid_image = make_grid(image, 5, normalize=True)
            #     writer.add_image('train/Labeled_Image', grid_image, iter_num)
            #
            #     image = volume_batch_strong[0, 0:1, :, :, 20:61:10].permute(
            #         3, 0, 1, 2).repeat(1, 3, 1, 1)
            #     grid_image = make_grid(image, 5, normalize=True)
            #     writer.add_image('train/Labeled_Image_Strong', grid_image, iter_num)
            #
            #     image = labeled_outputs_soft[0, 1:2, :, :, 20:61:10].permute(
            #         3, 0, 1, 2).repeat(1, 3, 1, 1)
            #     grid_image = make_grid(image, 5, normalize=False)
            #     writer.add_image('train/Predicted_label',
            #                      grid_image, iter_num)
            #
            #     image = label_batch[0, :, :, 20:61:10].permute(2,0,1).data.cpu().numpy()
            #     image = utils.decode_seg_map_sequence(image)
            #     grid_image = make_grid(image, 5, normalize=False)
            #     writer.add_image('train/Groundtruth_label',
            #                      grid_image, iter_num)
            #
            #     image = unlabeled_volume_batch_image[0, 0:1, :, :, 20:61:10].permute(
            #         3, 0, 1, 2).repeat(1, 3, 1, 1)
            #     grid_image = make_grid(image, 5, normalize=True)
            #     writer.add_image('train/Unlabeld_Image', grid_image, iter_num)
            #
            #     image = unlabeled_volume_batch_weak[0, 0:1, :, :, 20:61:10].permute(
            #         3, 0, 1, 2).repeat(1, 3, 1, 1)
            #     grid_image = make_grid(image, 5, normalize=True)
            #     writer.add_image('train/Unlabeld_Image_Weak', grid_image, iter_num)
            #
            #     image = unlabeled_volume_batch_strong[0, 0:1, :, :, 20:61:10].permute(
            #         3, 0, 1, 2).repeat(1, 3, 1, 1)
            #     grid_image = make_grid(image, 5, normalize=True)
            #     writer.add_image('train/Unlabeld_Image_Strong', grid_image, iter_num)
            #
            #     image = ema_output_soft[0, 1:2, :, :, 20:61:10].permute(
            #         3, 0, 1, 2).repeat(1, 3, 1, 1)
            #     grid_image = make_grid(image, 5, normalize=False)
            #     writer.add_image('train/Unlabeled_Predicted_label_teacher',
            #                      grid_image, iter_num)
            #
            #     image = unlabeled_outputs_soft[0, 1:2, :, :, 20:61:10].permute(
            #         3, 0, 1, 2).repeat(1, 3, 1, 1)
            #     grid_image = make_grid(image, 5, normalize=False)
            #     writer.add_image('train/Unlabeled_Predicted_label_student',
            #                      grid_image, iter_num)
            #
            #     # image = unlabeled_label_batch[0, :, :, 20:61:10].unsqueeze(
            #     #     0).permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
            #     # image = utils.decode_seg_map_sequence(image)
            #     # grid_image = make_grid(image, 5, normalize=False)
            #     # writer.add_image('train/Unlabeled_Groundtruth_label',
            #     #                  grid_image, iter_num)
            #     image = unlabeled_label_batch[0, :, :, 20:61:10].permute(2, 0, 1).data.cpu().numpy()
            #     image = utils.decode_seg_map_sequence(image)
            #     grid_image = make_grid(image, 5, normalize=False)
            #     writer.add_image('train/Unlabeled_Groundtruth_label',
            #                      grid_image, iter_num)
            #
            #     image = error[0, :, :, 20:61:10].unsqueeze(
            #         0).permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
            #     grid_image = make_grid(image, 5, normalize=False)
            #     writer.add_image('train/Noise_pseudo_label',
            #                      grid_image, iter_num)

            if iter_num > 0 and iter_num % 1000 == 0:
                model.eval()
                if args.dataset == 'LA':
                    stride_xy = 18
                    stride_z = 4
                else:
                    stride_xy = 16
                    stride_z = 16
                if args.dataset == 'FeTA':
                    avg_metric = test_all_case(
                        model, args.root_path, test_list="val.txt", num_classes=8, patch_size=args.patch_size,
                        stride_xy=stride_xy, stride_z=stride_z)
                    for i in range(len(avg_metric)):
                        print(i+1, 'class->', 'dice:', avg_metric[i][0], '   hd95:', avg_metric[i][1])
                else:
                    avg_metric = test_all_case(
                        model, args.root_path, test_list="val.txt", num_classes=2, patch_size=args.patch_size,
                        stride_xy=stride_xy, stride_z=stride_z)
                if avg_metric[:, 0].mean() > best_performance:
                    best_performance = avg_metric[:, 0].mean()
                    save_mode_path = os.path.join(snapshot_path,
                                                  'iter_{}_dice_{}.pth'.format(
                                                      iter_num, round(best_performance, 4)))
                    save_best = os.path.join(snapshot_path,
                                             '{}_best_model.pth'.format(args.model))
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best)
                    if args.dataset == 'FeTA':
                        file = open(snapshot_path + 'train_results.txt', "a+")
                        file.write(
                            'Metrics---DICE:' + str(format(best_performance, ".4f")) + ', HD:' + str(
                                format(avg_metric[:, 1].mean(), ".4f")) + ', iterations:' + str(iter_num) + '\n'
                            + 'class1->Dice:' + str(format(avg_metric[0][0], ".4f")) + '   HD:' + str(
                                format(avg_metric[0][1], ".4f")) + '\n'
                            + 'class2->Dice:' + str(format(avg_metric[1][0], ".4f")) + '   HD:' + str(
                                format(avg_metric[1][1], ".4f")) + '\n'
                            + 'class3->Dice:' + str(format(avg_metric[2][0], ".4f")) + '   HD:' + str(
                                format(avg_metric[2][1], ".4f")) + '\n'
                            + 'class4->Dice:' + str(format(avg_metric[3][0], ".4f")) + '   HD:' + str(
                                format(avg_metric[3][1], ".4f")) + '\n'
                            + 'class5->Dice:' + str(format(avg_metric[4][0], ".4f")) + '   HD:' + str(
                                format(avg_metric[4][1], ".4f")) + '\n'
                            + 'class6->Dice:' + str(format(avg_metric[5][0], ".4f")) + '   HD:' + str(
                                format(avg_metric[5][1], ".4f")) + '\n'
                            + 'class7->Dice:' + str(format(avg_metric[6][0], ".4f")) + '   HD:' + str(
                                format(avg_metric[6][1], ".4f")) + '\n')
                        file.close()
                    else:
                        file = open(snapshot_path + 'train_results.txt', "a+")
                        file.write(
                            'Metrics---DICE:' + str(format(best_performance, ".4f")) + ', HD:' + str(
                                format(avg_metric[:, 1].mean(), ".4f")) + ', iterations:' + str(iter_num) + '\n')
                        file.close()

                writer.add_scalar('info/val_dice_score',
                                  avg_metric[0, 0], iter_num)
                writer.add_scalar('info/val_hd95',
                                  avg_metric[0, 1], iter_num)
                logging.info(
                    'iteration %d : dice_score : %f hd95 : %f' % (iter_num, avg_metric[:, 0].mean(), avg_metric[:, 1].mean()))
                model.train()

            # if iter_num % 2500 == 0:
            #     lr_ = base_lr * 0.1 ** (iter_num // 2500)
            #     for param_group in optimizer.param_groups:
            #         param_group['lr'] = lr_

            if iter_num % 3000 == 0:
                save_mode_path = os.path.join(
                    snapshot_path, 'iter_' + str(iter_num) + '.pth')
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

            if iter_num >= max_iterations:
                break
        # whether use active
        ifupdate = False
        if epoch_num >= 0 and len(labeled_list) < args.labeled_num and epoch_num - start_epoch_now + 1>= 10 and epoch_num>=10:
            ifupdate = update.if_update(iou_value=total_iou_record, current_epoch=epoch_num+1, threshold=0.9,
                                        start_epoch=start_epoch_now, per_epoch = len(unlabeled_trainloader))
        elif len(labeled_list) == args.labeled_num and epoch_num - start_epoch_now + 1>=3 and final_update== False:
            final_update = update.if_update(iou_value=total_iou_record, current_epoch=epoch_num + 1, threshold=0.9,
                                        start_epoch=start_epoch_now, per_epoch=len(unlabeled_trainloader))


        # If score-memory meets the conditions for selecting samples for labeling, and the labor force has not been exhausted, the samples will be selected for labeling at this time.
        # if len(score_memory) > 0:
        #     count += 1
        print('count:', count, '   epoch_num:', epoch_num,"len",len(labeled_list),"score",len(score_memory) )
        # if len(labeled_list) < args.labeled_num and epoch_num>0 and epoch_num%10==0:
        if len(labeled_list) < args.labeled_num and ifupdate:
            total_var_score = []
            name = []
            for key in score_memory: # Calculate the difference in change in uncertainty for each sample
                value = score_memory[key]
                per_var_scorei = torch.var(value, dim=0)
                total_var_scorei = torch.mean(per_var_scorei)
                total_var_score.append(total_var_scorei)
                name.append(key)
            print('11111', len(name), len(total_var_score))
            # Filter the samples with top-k scores and label them.
            sort_score = total_var_score
            sort_score.sort()
            file = open(snapshot_path + 'train_list_vary.txt', "a+")
            file.write('\n' + 'Score Bank:' + '\n')
            for i in range(len(total_var_score)):
                value = total_var_score[i]
                file.write(str(value) + '   ')
            file.write('\n' + 'New Adding' + '\n')
            for i in range(args.k):
                top_k = sort_score[i]
                name_k = name[total_var_score.index(top_k)]
                # Update labeled_list, and unlabeled_list
                labeled_list.append(name_k)
                unlabeled_list.remove(name_k)
                file.write(name_k + '   ')
                if len(labeled_list) == args.labeled_num:
                    break
            file.close()

            # Update trainloader
            if args.dataset == 'LA':
                train_labeled_dataset = LeftArium(base_dir=train_data_path,
                                                  split='train',
                                                  num=None,
                                                  transform=WeakStrongAugment(args.patch_size),
                                                  train_list=labeled_list)
                train_unlabeled_dataset = LeftArium(base_dir=train_data_path,
                                                    split='train',
                                                    num=None,
                                                    transform=WeakStrongAugment(args.patch_size),
                                                    train_list=unlabeled_list)
            elif args.dataset == 'FeTA':
                train_labeled_dataset = FeTA(base_dir=train_data_path,
                                             split='train',
                                             num=None,
                                             transform=WeakStrongAugment(args.patch_size),
                                             train_list=labeled_list)
                train_unlabeled_dataset = FeTA(base_dir=train_data_path,
                                               split='train',
                                               num=None,
                                               transform=WeakStrongAugment(args.patch_size),
                                               train_list=unlabeled_list)


            labeled_trainloader = DataLoader(train_labeled_dataset, batch_size=args.labeled_bs, shuffle=True,
                                             num_workers=2, drop_last=True)
            unlabeled_trainloader = DataLoader(train_unlabeled_dataset, batch_size=args.batch_size - args.labeled_bs,
                                               shuffle=True,
                                               num_workers=2, drop_last=True)
            # Clear score-memory and retrain.
            # Clear cache and counters
            count = 0
            start_epoch_now = epoch_num + 1
            score_memory.clear()
            total_iou_record.clear()


        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()
    return "Training Finished!"


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    snapshot_path = "/data/kw/model/{}_{}/{}".format(
        args.exp, args.labeled_num, args.model)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    # if os.path.exists(snapshot_path + '/code'):
    #     shutil.rmtree(snapshot_path + '/code')
    # shutil.copytree('.', snapshot_path + '/code',
    #                 shutil.ignore_patterns(['.git', '__pycache__']))

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, snapshot_path)


