import os
import torch
import numpy as np
from glob import glob
from torch.utils.data import Dataset
import h5py
import itertools
from torch.utils.data.sampler import Sampler


class FeTA(Dataset):
    """ FeTA Dataset """

    def __init__(self, base_dir=None, split='train', num=None, transform=None, train_list = None):
        self._base_dir = base_dir
        self.transform = transform
        self.sample_list = []

        train_path = self._base_dir+'/train.txt'
        test_path = self._base_dir+'/test.txt'

        if split == 'train':
            if train_list is not None:
                self.image_list = train_list
            else:
                with open(train_path, 'r') as f:
                    self.image_list = f.readlines()
        elif split == 'test':
            with open(test_path, 'r') as f:
                self.image_list = f.readlines()

        self.image_list = [item.replace('\n', '').split(",")[0] for item in self.image_list]
        if num is not None:
            self.image_list = self.image_list[:num]
        print("total {} samples".format(len(self.image_list)))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        # h5f = h5py.File(self._base_dir + "/h5f2/{}.h5".format(image_name), 'r')
        h5f = h5py.File(self._base_dir + "/{}.h5".format(image_name), 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        sample = {'name': image_name, 'image': image, 'label': label.astype(np.uint8)}
        if self.transform:
            sample = self.transform(sample)
        return sample


