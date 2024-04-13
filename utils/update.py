from scipy.optimize import curve_fit
import numpy as np

def curve_func(x, a, b, c, d):
    return a * (1 - np.exp(-1 / c * (x - d+1e-4) ** b))

def fit(func, x, y):
    popt, pcov = curve_fit(func, x, y, method='trf',
                           absolute_sigma=True, bounds=([0, 0, 0, 0], [1, 1, np.inf, np.inf]))
    return tuple(popt)


def derivation(x, a, b, c, d):
    x = x + 1e-6  # numerical robustness
    return a * b * 1 / c * np.exp(-1 / c * (x - d+1e-4) ** b) * ((x - d+1e-4) ** (b - 1))


def label_update_epoch(ydata_fit, threshold=0.9, start_epoch=0, per_epoch=0):
    xdata_fit = np.arange(start_epoch, start_epoch + len(ydata_fit)/per_epoch, 1/per_epoch)
    print(xdata_fit)
    a, b, c, d = fit(curve_func, xdata_fit, ydata_fit)
    print("abcd is ",a,b,c, d)
    epoch = np.arange(1, 16)
    # y_hat = curve_func(epoch, a, b, c)
    relative_change = abs(abs(derivation(epoch, a, b, c, d)) - abs(derivation(1, a, b, c, d))) / abs(derivation(1, a, b, c, d))
    relative_change[relative_change > 1] = 0
    update_epoch = np.sum(relative_change <= threshold) + 1
    return update_epoch  # , a, b, c


def if_update(iou_value, current_epoch, threshold=0.90, start_epoch=0, per_epoch=0):
    update_epoch = label_update_epoch(iou_value, threshold=threshold, start_epoch=start_epoch, per_epoch=per_epoch)
    print("update_epoch is ", update_epoch)
    return current_epoch >= update_epoch
