import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class FloatingRegionScore(nn.Module):

    def __init__(self, in_channels=8, padding_mode='zeros', size=3):
        """
        purity_conv: size*size
        entropy_conv: size*size
        """
        super(FloatingRegionScore, self).__init__()
        self.in_channels = in_channels
        assert size % 2 == 1, "error size"
        self.purity_conv = nn.Conv3d(in_channels=in_channels, out_channels=in_channels, kernel_size=size,
                                     stride=1, padding=int(size / 2), bias=False,
                                     padding_mode=padding_mode, groups=in_channels)
        weight = torch.ones((size, size,size), dtype=torch.float32).cuda()  # 区域大小
        weight = weight.unsqueeze(dim=0).unsqueeze(dim=0)
        weight = weight.repeat([in_channels, 1, 1, 1, 1])
        weight = nn.Parameter(weight)
        self.purity_conv.weight = weight
        self.purity_conv.requires_grad_(False)

        self.entropy_conv = nn.Conv3d(in_channels=1, out_channels=1, kernel_size=size,
                                      stride=1, padding=int(size / 2), bias=False,
                                      padding_mode=padding_mode)
        weight = torch.ones((size, size,size), dtype=torch.float32).cuda()
        weight = weight.unsqueeze(dim=0).unsqueeze(dim=0)
        weight = nn.Parameter(weight)
        self.entropy_conv.weight = weight
        self.entropy_conv.requires_grad_(False)

    def forward(self, logit):
        """
        return:
            score, purity, entropy
        """
        logit = logit.squeeze(dim=0)  # [c, h ,w]，输出的概率结果
        p = torch.softmax(logit, dim=0)  # [c, h, w]， 输出值归一化

        pixel_entropy = torch.sum(-p * torch.log(p + 1e-6), dim=0).unsqueeze(dim=0).unsqueeze(dim=0) / math.log(
            self.in_channels)  # [1, 1, h, w]， 计算每个像素点的熵
        region_sum_entropy = self.entropy_conv(pixel_entropy)  # [1, 1, h, w]，计算一个区域的熵的总和
        # print('1',region_sum_entropy.shape)

        predict = torch.argmax(p, dim=0)  # [h, w]，预测的硬标签形式
        one_hot = F.one_hot(predict, num_classes=self.in_channels).float() # h,w,c
        # print('2',one_hot.shape)
        one_hot = one_hot.permute((3, 0, 1, 2)).unsqueeze(dim=0)  # [1, c, h, w]， 转换为one-hot形式
        summary = self.purity_conv(one_hot)  # [1, c, h, w]， 计算纯度
        # print('3',summary.shape)
        count = torch.sum(summary, dim=1, keepdim=True)  # [1, 1, h, w]
        # print('4',count.shape)
        dist = summary / count  # [1, c, h, w]
        # print('5',dist.shape)
        region_impurity = torch.sum(-dist * torch.log(dist + 1e-6), dim=1, keepdim=True) / math.log(self.in_channels)  # [1, 1, h, w]，计算区域不纯性
        # print('6',region_impurity.shape)
        prediction_uncertainty = region_sum_entropy / count  # [1, 1, h, w]，计算预测不确定性
        # print('7',prediction_uncertainty.shape)
        score = region_impurity * prediction_uncertainty # [1, 1, h, w], 计算最后得分

        return score.squeeze(dim=0).squeeze(dim=0), region_impurity.squeeze(dim=0).squeeze(
            dim=0), prediction_uncertainty.squeeze(dim=0).squeeze(dim=0)

if __name__ == '__main__':
    input = torch.rand(8,36,36,36)
    score = FloatingRegionScore(size=1)
    res,_,_ = score(input)
    print(score.shape)

