import torch
from torch.utils.data import Dataset

from skimage import io
import numpy as np
from utils import load_pfm

import cv2

class StereoDataset(Dataset):
    def __init__(self, img_fns_lft, img_fns_rgt, disp_fns, transform=None):
        """
        img_fns_lft : list of left image filenames
        img_fns_rgt : list of right image filenames
        disp_fns: list of disparity filenames
        transform: transform methods
        """
        self.img_fns_lft = img_fns_lft
        self.img_fns_rgt = img_fns_rgt
        self.disp_fns = disp_fns
        assert len(self.img_fns_lft) == len(self.img_fns_rgt), 'left image number must be equal to right image number'
        assert len(self.img_fns_lft) == len(self.disp_fns), 'image number must be equal to disparity number'
        self.transform = transform

    def __len__(self):
        return len(self.img_fns_lft)

    def __getitem__(self, idx):
        img_lft = io.imread(self.img_fns_lft[idx])  # HWC; BGR
        img_rgt = io.imread(self.img_fns_rgt[idx])  # HWC; BGR
        if self.disp_fns[idx].endswith('.pfm'):
            disp = load_pfm(self.disp_fns[idx])[::-1]  # HW
        elif self.disp_fns[idx].endswith('.png'):
            disp = io.imread(self.disp_fns[idx]) / 256.0
        sample = {'left': img_lft, 'right': img_rgt, 'disparity': disp}

        if self.transform:
            sample = self.transform(sample)

        return sample


class ToTensor():
    def __call__(self, sample):
        img_lft = sample['left'].astype(np.float32) / 255.0 * 2 - 1
        img_rgt = sample['right'].astype(np.float32) / 255.0 * 2 - 1
        disp = sample['disparity'].astype(np.float32)

        img_lft = img_lft.transpose([2, 0, 1])  # from HWC to CHW
        img_rgt = img_rgt.transpose([2, 0, 1])
        return {'left': torch.from_numpy(img_lft), 'right': torch.from_numpy(img_rgt), 'disparity': torch.from_numpy(disp)}


class CenterCrop():
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        img_lft = sample['left']
        img_rgt = sample['right']
        disp = sample['disparity']
        
        h, _, _ = img_lft.shape
        h_, w_ = disp.shape
        
        new_h, new_w = self.output_size
        top = (h_ - new_h) // 2
        left = (w_ - new_w) // 2

        img_lft = img_lft[top + h - h_: top + h - h_ + new_h, left: left + new_w]
        img_rgt = img_rgt[top + h - h_: top + h - h_ + new_h, left: left + new_w]
        disp = disp[top: top + new_h, left: left + new_w]

        return {'left': img_lft, 'right': img_rgt, 'disparity': disp}


class BottomLeftCrop():
    def __call__(self, sample):
        img_lft = sample['left']
        img_rgt = sample['right']
        disp = sample['disparity']
        
        h, _, _ = img_lft.shape
        h_, w_ = disp.shape
        
        top = h - h_

        img_lft = img_lft[top: top + h_, 0: w_]
        img_rgt = img_rgt[top: top + h_, 0: w_]

        return {'left': img_lft, 'right': img_rgt, 'disparity': disp}