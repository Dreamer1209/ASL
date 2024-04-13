import torch
import torch.nn as nn
import numpy as np

kernl = np.asarray([[[1/27, 1/27, 1/27], [1/27, 1/27, 1/27], [1/27, 1/27, 1/27]], [[1/27, 1/27, 1/27], [1/27, 0, 1/27], [1/27, 1/27, 1/27]],
                     [[1/27, 1/27, 1/27], [1/27, 1/27, 1/27], [1/27, 1/27, 1/27]]])
filter = nn.Conv3d(in_channels=1, out_channels=1, stride=1, kernel_size=3,
                                        padding=1, bias=False)
filter.weight.data = torch.from_numpy(kernl.astype(np.float32)).reshape(1, 1, 3, 3, 3)
filter = filter.cuda()

def student_correction(b, x, y, z, ema_output_soft, bound):
    x_lower_bound = max(0, x-bound)
    x_upper_bound = min(ema_output_soft.shape[2], x + bound)
    y_lower_bound = max(0, y-bound)
    y_upper_bound = min(ema_output_soft.shape[3], y + bound)
    z_lower_bound = max(0, z - bound)
    z_upper_bound = min(ema_output_soft.shape[4], z + bound)
    count = 0
    value = torch.zeros(8).cuda()
    for bi in range(x_lower_bound, x_upper_bound):
        for bj in range(y_lower_bound, y_upper_bound):
            for bk in range(z_lower_bound, z_upper_bound):
                count += 1
                value += ema_output_soft[b,:,bi,bj,bk]
    return value/count



def pseudo_correction(dirty_labels_ones, ema_output_soft, bound, alpha, beta):
    local_suggestion = torch.zeros(ema_output_soft.shape).cuda()
    # b, x, y, z = dirty_labels_ones.shape
    # for bb in range(b):
    #     for ii in range(x):
    #         for jj in range(y):
    #             for kk in range(z):
    #
    #                 if dirty_labels_ones[bb,ii,jj,kk] == 1:
    #                     local_suggestion[bb,:,ii,jj,kk] += best_neighbour(bb,ii,jj,kk, ema_output_soft,bound)
    b, c, x, y, z = ema_output_soft.shape
    for i in range(8):
        ema_output_class = ema_output_soft[:,i,:,:,:].reshape(b,1,x,y,z)
        ema_output_class = filter(ema_output_class)
        local_suggestion[:,i:i+1,:,:,:] = ema_output_class * dirty_labels_ones
    ema_output_soft_after = alpha * ema_output_soft + (1- alpha) * local_suggestion
    return ema_output_soft_after


