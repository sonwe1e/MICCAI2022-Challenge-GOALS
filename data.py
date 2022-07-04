#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: data.py
@Time: 2018/10/13 6:21 PM
"""

import os
import cv2
import random
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class Dataset(Dataset):
    def __init__(self, transform=0, tvt='train'):
        self.transform = transform
        self.tvt = tvt
        self.train_path = '/home/gdut403/sonwe1e/MICCAI/data/Train/'
        self.gt_path = self.train_path + 'Layer_Masks/'
        self.img_path = self.train_path + 'Image/'
        self.img_list = os.listdir(self.img_path)
        random.shuffle(self.img_list)
        if self.tvt == 'train':
            self.img_list = self.img_list[:90]
        elif self.tvt == 'val':
            self.img_list = self.img_list[90:]
        else:
            self.img_path = '/home/gdut403/sonwe1e/MICCAI/data/GOALS2022-Validation/Image/'
            self.img_list = sorted(os.listdir(self.img_path))

    def __getitem__(self, item):
        img_name = self.img_list[item]
        img = cv2.imread(self.img_path + img_name)
        if self.tvt != 'test':
            gt = cv2.imread(self.gt_path + img_name)
            gt[gt == 80] = 1
            gt[gt == 160] = 2
            gt[gt == 255] = 3
            if self.transform:
                transformed = self.transform(image=img, mask=gt)
                img = transformed['image']
                gt = transformed['mask']
            return img, gt[..., 0].long()
        if self.transform:
            img = self.transform(image=img)['image']
        return img

    def __len__(self):
        return len(self.img_list)


print('=========Testing Dataset========')
train_transform = A.Compose([
    A.Resize(height=768, width=1024, interpolation=cv2.INTER_CUBIC),
    A.ShiftScaleRotate(shift_limit=0.2, rotate_limit=30, scale_limit=0.2, p=0.8),
    A.Normalize(),
    ToTensorV2()
])
test_transform = A.Compose([
    A.Resize(height=768, width=1024, interpolation=cv2.INTER_CUBIC),
    A.Normalize(),
    ToTensorV2()
])
train = DataLoader(Dataset(transform=train_transform, tvt='train'))
val = DataLoader(Dataset(transform=test_transform, tvt='val'))
for i in train:
    img, gt = i
    print(img.shape, gt.shape)
    break

for i in val:
    img, gt = i
    print(img.shape, gt.shape)
    break

test_loader = DataLoader(Dataset(tvt='test', transform=test_transform), num_workers=16, pin_memory=True, batch_size=1, shuffle=False)
next(iter(test_loader))
