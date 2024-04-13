import torch.nn as nn
import math
from torch.nn.modules.batchnorm import _BatchNorm
import torch.nn.functional as F

class External_attention(nn.Module):
    '''
    Arguments:
        c (int): The input and output channel number.
    '''
    def __init__(self, c):
        super(External_attention, self).__init__()

        self.conv1 = nn.Conv3d(c, c, 3)
        self.k = 64
        self.linear_0 = nn.Conv1d(c, self.k, 1, bias=False)

        self.linear_1 = nn.Conv1d(self.k, c, 1, bias=False)
        self.linear_1.weight.data = self.linear_0.weight.data.permute(1, 0, 2)

        self.conv2 = nn.Sequential(
            nn.Conv3d(c, c, 3, bias=False))

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, _BatchNorm):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()
    #x = b*c*n
    def forward(self, x):

        # x = x.permute(0, 2, 1)
        # b, c, l = x.size()
        # h = w = (int)(l**0.5)
        # x = x.view(b, c, h, w)

        idn = x
        x = self.conv1(x)

        b, c, h, w, z = x.size()
        x = x.view(b, c, h * w *z)  # b * c * n

        attn = self.linear_0(x)  # b, k, n
        attn = F.softmax(attn, dim=-1)  # b, k, n
        attn = attn / (1e-9 + attn.sum(dim=1, keepdim=True))  # # b, k, n
        x = self.linear_1(attn)  # b, c, n

        x = x.view(b, c, h, w, z)
        x = self.conv2(x)
        x = x + idn
        x = F.relu(x)

        return x
