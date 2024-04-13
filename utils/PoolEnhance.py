import os
import copy
import torch
import torch.nn as nn


class poolenhance(nn.Module):

    def __init__(self, num_class = 8, pool_size = 1):
        super().__init__()
        self.sig = nn.Sigmoid()
        self.num_class = num_class
        # self.pool = nn.AvgPool3d(pool_size, stride = 1, padding = pool_size // 2, count_include_pad = False)
        self.pool = nn.AdaptiveAvgPool3d(pool_size)
    def forward(self, x):

        b, c, h, w, z = x.shape
        attn = x * self.pool(x)

        # attn = attn.sum(dim=1, keepdim=True)  # bs*g,1,h,w
        attn = attn.view(b*c, -1)
        mean = attn.mean(dim=1, keepdim=True)
        std = attn.std(dim=1, keepdim=True)
        attn = (attn - mean) / (std + 1e-5)
        attn = attn.view(b, c, h, w, z)

        x = x * self.sig(attn)

        return x

x = torch.rand((2,2,3,3,3))
pool = poolenhance(2)
print(pool(x).shape)


