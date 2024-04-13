import math
from glob import glob

import h5py
import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F
from medpy import metric
import os
from tqdm import tqdm


def test_single_case(net, image, stride_xy, stride_z, patch_size, num_classes=1):
    w, h, d = image.shape

    # if the size of image is less than patch_size, then padding it
    add_pad = False
    if w < patch_size[0]:
        w_pad = patch_size[0]-w
        add_pad = True
    else:
        w_pad = 0
    if h < patch_size[1]:
        h_pad = patch_size[1]-h
        add_pad = True
    else:
        h_pad = 0
    if d < patch_size[2]:
        d_pad = patch_size[2]-d
        add_pad = True
    else:
        d_pad = 0
    wl_pad, wr_pad = w_pad//2, w_pad-w_pad//2
    hl_pad, hr_pad = h_pad//2, h_pad-h_pad//2
    dl_pad, dr_pad = d_pad//2, d_pad-d_pad//2
    if add_pad:
        image = np.pad(image, [(wl_pad, wr_pad), (hl_pad, hr_pad),
                               (dl_pad, dr_pad)], mode='constant', constant_values=0)
    ww, hh, dd = image.shape

    sx = math.ceil((ww - patch_size[0]) / stride_xy) + 1
    sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
    sz = math.ceil((dd - patch_size[2]) / stride_z) + 1
    # print("{}, {}, {}".format(sx, sy, sz))
    score_map = np.zeros((num_classes, ) + image.shape).astype(np.float32)
    cnt = np.zeros(image.shape).astype(np.float32)

    for x in range(0, sx):
        xs = min(stride_xy * x, ww-patch_size[0])
        for y in range(0, sy):
            ys = min(stride_xy * y, hh-patch_size[1])
            for z in range(0, sz):
                zs = min(stride_z * z, dd-patch_size[2])
                test_patch = image[xs:xs+patch_size[0],
                                   ys:ys+patch_size[1], zs:zs+patch_size[2]]
                test_patch = np.expand_dims(np.expand_dims(
                    test_patch, axis=0), axis=0).astype(np.float32)
                test_patch = torch.from_numpy(test_patch).cuda()

                with torch.no_grad():
                    y1 = net(test_patch)
                    # ensemble
                    y = torch.softmax(y1, dim=1)
                y = y.cpu().data.numpy()
                y = y[0, :, :, :, :]
                score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                    = score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + y
                cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                    = cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + 1
    score_map = score_map/np.expand_dims(cnt, axis=0)
    label_map = np.argmax(score_map, axis=0)

    if add_pad:
        label_map = label_map[wl_pad:wl_pad+w,
                              hl_pad:hl_pad+h, dl_pad:dl_pad+d]
        score_map = score_map[:, wl_pad:wl_pad +
                              w, hl_pad:hl_pad+h, dl_pad:dl_pad+d]
    return label_map


def cal_metric(gt, pred):
    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return np.array([dice, hd95])
    else:
        return np.zeros(2)


def test_all_case(net, base_dir, test_list="full_test.list", num_classes=4, patch_size=(48, 160, 160), stride_xy=32, stride_z=24):
    with open(base_dir + '/{}'.format(test_list), 'r') as f:
        image_list = f.readlines()
    image_list = [base_dir + "/{}".format(
        item.replace('\n', '').split(",")[0]) for item in image_list]
    total_metric = np.zeros((num_classes-1, 2))
    print("Validation begin")
    for image_path in tqdm(image_list):
        h5f = h5py.File(image_path, 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        prediction = test_single_case(
            net, image, stride_xy, stride_z, patch_size, num_classes=num_classes)
        # print(prediction.shape)
        # print(prediction.max())
        img = sitk.GetImageFromArray(prediction)
        sitk.WriteImage(img, "/data/kw/"+'save.nii.gz')
        for i in range(1, num_classes):
            total_metric[i-1, :] += cal_metric(label == i, prediction == i)
    print("Validation end")
    return total_metric / len(image_list)

def predict_all_case(net, base_dir, num_classes=8, patch_size=(64, 64, 64), stride_xy=32, stride_z=24):
    image_list = os.listdir(base_dir)
    print("Preditc begin")
    store_path = "/data/kw/predict/"
    for image_path in tqdm(image_list):
        # h5f = h5py.File(image_path, 'r')
        image_obj = nib.load(base_dir+image_path)
        image = image_obj.get_fdata()
        # print(image.shape)
        prediction = test_single_case(
            net, image, stride_xy, stride_z, patch_size, num_classes=num_classes)
        # print(image_path)
        # print(prediction.shape)
        total_metric = np.zeros((num_classes - 1, 2))
        # for i in range(1, 16):
        #     total_metric[i - 1, :] += cal_metric(label == i, prediction == i)
        # print(prediction.shape)
        prediction =np.array(prediction).astype(np.int16)
        new_image = nib.Nifti1Image(prediction, image_obj.affine)
        nib.save(new_image, store_path + image_path)
        # img = sitk.GetImageFromArray(prediction)
        # sitk.WriteImage(img, "/data/kw/"+'save.nii.gz')
    print("Validation end")
    # return total_metric / len(image_list)

if __name__ == "__main__":
    root_path = ""
    model_dir = os.path.join(root_path, 'Weights')
    model_list = os.listdir(model_dir)
    model_list.sort()
    model_path = os.path.join(model_dir, model_list[-1])
    state = torch.load(str(model_path), map_location=torch.device('cpu'))
    print(model_path)
    epoch = state['epoch'] + 1
    step = state['step']
    model = SwinUNet()
    model = nn.DataParallel(model, device_ids=None)
    model.load_state_dict(state['model'])
