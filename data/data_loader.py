#!/user/bin/python
# -*- encoding: utf-8 -*-

from torch.utils import data
import os
from os.path import join, basename
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
import cv2
from .transform import *


def prepare_image_PIL(im, mean_bgr):
    im = im[:, :, ::-1] - np.zeros_like(im)  # rgb to bgr
    im -= np.array(mean_bgr)
    im = np.transpose(im, (2, 0, 1))  # (H x W x C) to (C x H x W)
    return im


class MyDataLoader(data.Dataset):
    """
    Dataloader
    """

    def __init__(self, cfg, args, split='train', transform=True):
        self.args = args
        self.root = cfg.dataset
        self.split = split
        self.mean = cfg.mean
        self.transform = transform
        if self.split == 'train':
            self.filelist = join(self.root, cfg.train_list)
        elif self.split == 'test':
            self.filelist = join(self.root, cfg.test_list)
        else:
            raise ValueError("Invalid split type!")
        with open(self.filelist, 'r') as f:
            self.filelist = f.readlines()

        # pre-processing
        if self.transform:
            self.trans = Compose([
                ColorJitter(
                    brightness=0.5,
                    contrast=0.5,
                    saturation=0.5),
            ])

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, index):
        if self.split == "train":
            img_file, lb_file, KDlb_file = self.filelist[index].split()

            label = Image.open(join(self.root, lb_file))
            KDlabel = Image.open(join(self.root, KDlb_file))
            img = Image.open(join(self.root, img_file))

            if self.transform:
                im_lb = dict(im=img, lb=label)
                im_lb = self.trans(im_lb)
                img, label = im_lb['im'], im_lb['lb']
            img = np.array(img, dtype=np.float32)
            img = prepare_image_PIL(img, self.mean)

            label = np.array(label, dtype=np.float32)
            KDlabel = np.array(KDlabel, dtype=np.float32)

            if label.ndim == 3:
                label = np.squeeze(label[:, :, 0])
            if KDlabel.ndim == 3:
                KDlabel = np.squeeze(KDlabel[:, :, 0])

            assert label.ndim == 2 and KDlabel.ndim == 2

            KDlabel = KDlabel[np.newaxis, :, :]/255

            label = label[np.newaxis, :, :]
            yita = 100
            label[label == 0] = 0
            label[np.logical_and(label > 0, label < yita)] = 2
            label[label >= yita] = 1

            return img, label, KDlabel, basename(img_file).split('.')[0]
        else:
            img_file = self.filelist[index].rstrip()
            img = np.array(Image.open(join(self.root, img_file)), dtype=np.float32)
            img = prepare_image_PIL(img, self.mean)
            return img, basename(img_file).split('.')[0]
