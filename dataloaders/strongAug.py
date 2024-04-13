# import augmentation from monai
# website:https://docs.monai.io/en/latest/transforms.html#fourier

import monai.transforms as t
import h5py
import random
import numpy as np
import torch
# RandGibbsNoise: Add Gaussian noise to image.
# shiftIntensity: Shift intensity uniformly for the entire image with specified offset.
# RandShiftIntensity: Randomly shift intensity with randomly picked offset.
# StdShiftIntensity: Shift intensity for the image with a factor and the standard deviation of the image by: v = v + factor * std(v). This transform can focus on only non-zero values or the entire image, and can also calculate the std on each channel separately.
# RandStdShiftIntensit: Shift intensity for the image with a factor and the standard deviation of the image by: v = v + factor * std(v) where the factor is randomly picked.
# RandBiasField: Random bias field augmentation for MR images. The bias field is considered as a linear combination of smoothly varying basis (polynomial) functions, as described in Automated Model-Based Tissue Classification of MR Images of the Brain. This implementation adapted from NiftyNet. Referred to Longitudinal segmentation of age-related white matter hyperintensities.
# ScaleIntensity: Scale the intensity of input image to the given value range (minv, maxv). If minv and maxv not provided, use factor to scale image by v = v * (1 + factor).
# RandScaleIntensity: Randomly scale the intensity of input image by v = v * (1 + factor) where the factor is randomly picked.
# NormalizeIntensity: Normalize input based on provided args, using calculated mean and std if not provided. This transform can normalize only non-zero values or entire image, and can also calculate mean and std on each channel separately. When channel_wise is True, the first dimension of subtrahend and divisor should be the number of image channels if they are not None.
# ThresholdIntensity: Filter the intensity values of whole image to below threshold or above threshold. And fill the remaining parts of the image to the cval value.
# ScaleIntensityRange: Apply specific intensity scaling to the whole numpy array. Scaling from [a_min, a_max] to [b_min, b_max] with clip option.
# ScaleIntensityRangePercentiles: Apply range scaling to a numpy array based on the intensity distribution of the input.
# AdjustContrast: Changes image intensity by gamma. Each pixel/voxel intensity is updated as:x = ((x - min) / intensity_range) ^ gamma * intensity_range + min
# RandAdjustContrast: Randomly changes image intensity by gamma. Each pixel/voxel intensity is updated as:
# MaskIntensity: Mask the intensity values of input image with the specified mask data. Mask data must have the same spatial size as the input image, and all the intensity values of input image corresponding to 0 in the mask data will be set to 0, others will keep the original value.
# SavitzkyGolaySmooth: Smooth the input data along the given axis using a Savitzky-Golay filter.
# GaussianSmooth: Apply Gaussian smooth to the input data based on specified sigma parameter. A default value sigma=1.0 is provided for reference.
# RandGaussianSmooth: Apply Gaussian smooth to the input data based on randomly selected sigma parameters.
# GaussianSharpen: Sharpen images using the Gaussian Blur filter.
# RandGaussianSharpen: Sharpen images using the Gaussian Blur filter based on randomly selected sigma1, sigma2 and alpha.
# RandHistogramShift: Apply random nonlinear transform to the image’s intensity histogram.
# GibbsNoise: The transform applies Gibbs noise to 2D/3D MRI images. Gibbs artifacts are one of the common type of type artifacts appearing in MRI scans.
# RandGibbsNoised: Naturalistic image augmentation via Gibbs artifacts. The transform randomly applies Gibbs noise to 2D/3D MRI images.
# KSpaceSpikeNoise: Apply localized spikes in k-space at the given locations and intensities. Spike (Herringbone) artifact is a type of data acquisition artifact which may occur during MRI scans.
# RandKSpaceSpikeNoise: Naturalistic data augmentation via spike artifacts. The transform applies localized spikes in k-space,


# define augmentation pool

PARAMETER_MAX = 10

def _float_parameter(v, max_v):
    return float(v) * max_v / PARAMETER_MAX

def randGibbsNoise(img, **kwargs):
    T = t.RandGibbsNoise(prob=0.5)
    if torch.is_tensor(img):
        img = img.numpy()
    out = T(img)
    return out

def shiftIntensity(img, v, max_v):
    v = _float_parameter(v, max_v)
    T = t.ShiftIntensity(v)
    if torch.is_tensor(img):
        img = img.numpy().astype(np.float32)
    return T(img)

def randShiftIntensity(img, v, max_v):
    v = _float_parameter(v, max_v)
    T = t.RandShiftIntensity(v)
    if torch.is_tensor(img):
        img = img.numpy()
    return T(img)

def stdShiftIntensity(img, v, max_v):
    v = _float_parameter(v, max_v)
    T = t.StdShiftIntensity(v)
    if torch.is_tensor(img):
        img = img.numpy()
    return T(img)

def randStdShiftIntensity(img, v, max_v):
    v = _float_parameter(v, max_v)
    T = t.RandStdShiftIntensity(v)
    if torch.is_tensor(img):
        img = img.numpy()
    return T(img)

def randBiasField(img, **kwargs):
    T = t.RandBiasField()
    if torch.is_tensor(img):
        img = img.numpy()
    return T(img)

def scaleIntensity(img, **kwargs):
    # v = _float_parameter(v, max_v)
    T = t.ScaleIntensity()
    if torch.is_tensor(img):
        img = img.numpy()
    return T(img)

def randScaleIntensity(img, v, max_v):
    v = _float_parameter(v, max_v)
    T = t.RandScaleIntensity(v)
    if torch.is_tensor(img):
        img = img.numpy()
    return T(img)

def normalizeIntensity(img, **kwargs):
    T = t.NormalizeIntensity()
    if torch.is_tensor(img):
        img = img.numpy()
    return T(img)

def thresholdIntensity(img, v, max_v):
    v = _float_parameter(v,max_v)
    T = t.ThresholdIntensity(threshold=v, above=False)
    if torch.is_tensor(img):
        img = img.numpy()
    return T(img)

def adjustContrast(img, v, max_v):
    v = _float_parameter(v, max_v)
    T = t.AdjustContrast(v)
    if torch.is_tensor(img):
        img = img.numpy()
    return T(img)

def randAdjustContrast(img, v, max_v):
    v = _float_parameter(v, max_v)
    T = t.RandAdjustContrast(prob=v)
    if torch.is_tensor(img):
        img = img.numpy()
    return T(img)

#def MaskIntensity(img, **kwargs): before strong

def savitzkyGolaySmooth(img, v, max_v):
    v = _float_parameter(v, max_v)
    T = t.SavitzkyGolaySmooth(window_length=v, order=v-1)
    if torch.is_tensor(img):
        img = img.numpy()
    return T(img)

def GaussianSmooth(img, **kwargs):
    T = t.GaussianSmooth()
    if torch.is_tensor(img):
        img = img.numpy()
    return T(img)

def randGaussianSmooth(img, **kwargs):
    T = t.RandGaussianSmooth()
    if torch.is_tensor(img):
        img = img.numpy()
    return T(img)

def gaussianSharpen(img, **kwargs):
    T = t.GaussianSharpen()
    if torch.is_tensor(img):
        img = img.numpy()
    return T(img)

def randGaussianSharpen(img, **kwargs):
    T = t.RandGaussianSharpen()
    if torch.is_tensor(img):
        img = img.numpy()
    return T(img)

def randHistogramShift(img, **kwargs):
    T = t.RandHistogramShift(prob=0.5)
    if torch.is_tensor(img):
        img = img.numpy().astype(np.float32)
    return T(img)

def gibbsNoise(img, **kwargs):
    T = t.GibbsNoise()
    if torch.is_tensor(img):
        img = img.numpy()
    return T(img)

def randKSpaceSpikeNoise(img, **kwargs):
    T = t.RandKSpaceSpikeNoise(prob=0.5)
    if torch.is_tensor(img):
        img = img.numpy().astype(np.float32)
    return T(img)

def augment_fool_3D():
    augs = [
        (randGibbsNoise, None, None),
        (shiftIntensity, 5, 128),
        (randShiftIntensity, 3, 10),
        # (stdShiftIntensity, 5, 10),
        # (randStdShiftIntensity, 5, 10),
        (randBiasField, None, None),
        (scaleIntensity, None, None),
        (randScaleIntensity, 5, 1),
        (thresholdIntensity, 3, 128),
        (adjustContrast, 3, 5),
        (randAdjustContrast, 5, 1),
        (randGaussianSmooth, None, None),
        (gaussianSharpen, None, None),
        (randGaussianSharpen, None, None),
        (randHistogramShift, None, None),
        (gibbsNoise, None, None),
        # (randKSpaceSpikeNoise, None, None)
    ]
    return augs

class RandAugment(object):
    def __init__(self, n, m):
        assert n >= 1
        assert 1 <= m <= 10
        self.n = n
        self.m = m
        self.augment_pool = augment_fool_3D()

    def __call__(self, img):
        ops = random.choices(self.augment_pool, k=self.n)
        for op, v, max_v in ops:
            prob = np.random.uniform(0.2, 0.8)
            v = np.random.randint(1, self.m)
            if random.random() + prob >= 1:
                img = op(img, v=v, max_v=max_v)
        return img



if __name__ == '__main__':
    image_name = "06SR5RBREL16DQ6M8LWS"
    h5f = h5py.File("/data/lcx/LA/2018LA_Seg_Training Set/" + image_name + "/mri_norm2.h5", 'r')
    image = h5f['image'][:]
    print(image.shape)
    img_1 = randGibbsNoise(image)
    print(img_1.shape)
    img_2 = shiftIntensity(image, 5, 128)
    print(img_2.shape)
    img_3 = randShiftIntensity(image, 3, 10)
    print(img_3.shape)
    img_4 = stdShiftIntensity(image, 5, 10)
    print(img_4.shape)
    img_5 = randStdShiftIntensity(image, 5, 10)
    print(img_5.shape)
    randBiasField = t.RandBiasField()
    img_6 = randBiasField(image)
    print(img_6.shape)
    img_7 = scaleIntensity(image)
    print(img_7.shape)
    img_8 = randScaleIntensity(image, 5, 1)
    print(img_8.shape)
    img_9 = normalizeIntensity(image)
    print(img_9.shape)
    img_10 = thresholdIntensity(image, 3, 128)
    print(img_10.shape)
    img_11 = adjustContrast(image, 3, 5)
    print(img_11.shape)
    img_12 = randAdjustContrast(image, 5, 1)
    print(img_12.shape)
    img_13 = scaleIntensity(image)
    print(img_13.shape)
    img_14 = randGaussianSmooth(image)
    print(img_14.shape)
    img_15 = gaussianSharpen(image)
    print(img_15.shape)
    img_16 = randGaussianSharpen(image)
    print(img_16.shape)
    img_17 = randHistogramShift(image)
    print(img_17.shape)
    img_18 = gibbsNoise(image)
    print(img_18.shape)
    img_19 = randGibbsNoise(image)
    print(img_19.shape)
    img_20 = randKSpaceSpikeNoise(image)
    print(img_20.shape)
    print(type(img_1),type(img_2),type(img_3),type(img_4),type(img_5),type(img_6),type(img_7),type(img_8),type(img_9),type(img_10),type(img_11),type(img_12),type(img_13),type(img_14),type(img_15),type(img_16),type(img_17),type(img_18),type(img_19),type(img_20),)


    strong = RandAugment(n=3, m=10)
    img = strong(image)
    print(img.shape)



