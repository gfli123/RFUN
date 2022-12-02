import os
import glob
import random
import torch
import numpy as np
from img_preprocess import imread, img_trans, modcropHR
import torch.nn.functional as F
from option import opt
from torch.utils import data
from skimage import io
from torchvision import transforms
from Guassian import Guassian_downsample


class Train_data(data.Dataset):
    def __init__(self):
        self.train_data = open(opt.train_data, 'rt').read().splitlines()
        self.scale = opt.scale
        self.num_frames = opt.num_frames
        self.trans_tensor = transforms.ToTensor()

    def __getitem__(self, idx):

        img_path = sorted(glob.glob(os.path.join('./data81/sequences', self.train_data[idx], '*.png')))

        HR_all = []
        for i in range(self.num_frames):
            # HR
            img = imread(img_path[i])
            HR_all.append(img)

        HR_all = img_trans(HR_all)
        HR_all = [modcropHR(HR) for HR in HR_all]
        HR_all = [self.trans_tensor(HR).float() for HR in HR_all]
        HR_all = torch.stack(HR_all, dim=1)
        LR = Guassian_downsample(HR_all, self.scale)
        return LR, HR_all

    def __len__(self):
        return len(self.train_data)


