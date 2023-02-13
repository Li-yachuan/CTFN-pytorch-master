#!/user/bin/python
# -*- encoding: utf-8 -*-

import os
import torch
import torchvision
from PIL import Image
from os.path import join, isdir
import numpy as np
from tqdm import tqdm
import cv2
import scipy.io as sio
import time


def test_side_edge(model, test_loader, save_dir, epoch="test-designed"):
    save_mat_dir = []
    save_png_dir = []

    for i in ["1", "2", "3", "4", "5", "fuse"]:
        save_mat_dir.append(join(save_dir, i, 'mat'))
        save_png_dir.append(join(save_dir, i, 'png'))
        os.makedirs(join(save_dir, i, 'mat'), exist_ok=True)
        os.makedirs(join(save_dir, i, 'png'), exist_ok=True)
    # scale = [0.5, 1, 1.5]
    scale = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]

    model.eval()
    dl = tqdm(test_loader)

    for image, pth in dl:
        dl.set_description("Multi-scale test")
        image = image[0]
        image_in = image.numpy().transpose((1, 2, 0))
        _, H, W = image.shape
        multi_fuse = np.zeros((6,H, W), np.float32)

        for k in range(0, len(scale)):
            im_ = cv2.resize(image_in, None, fx=scale[k], fy=scale[k], interpolation=cv2.INTER_LINEAR)
            im_ = im_.transpose((2, 0, 1))
            inputs = torch.unsqueeze(torch.from_numpy(im_).cuda(), 0)
            results = model(inputs)
            for index, result in enumerate(results):
                result = torch.squeeze(result.detach()).cpu().numpy()
                result = cv2.resize(result, (W, H), interpolation=cv2.INTER_LINEAR)
                multi_fuse[index] += result
        # multi_fuse = [mf / len(scale) for mf in multi_fuse]
        multi_fuse =[(mf-mf.min()) / (mf.max()-mf.min()) for mf in multi_fuse]

        filename = pth[0]
        for index, multi_r in enumerate(multi_fuse):
            Image.fromarray((multi_r * 255).astype(np.uint8)).\
                save(join(save_png_dir[index], "%s.png" % filename))
            sio.savemat(join(save_mat_dir[index], '%s.mat' % filename), {'result': multi_r})



def test(model, test_loader, save_dir, epoch=None):
    if epoch is None:
        save_mat_dir = join(save_dir, 'mat')
        save_png_dir = join(save_dir, 'png')
    else:
        save_mat_dir = join(save_dir, epoch, 'mat')
        save_png_dir = join(save_dir, epoch, 'png')
    os.makedirs(save_mat_dir, exist_ok=True)
    os.makedirs(save_png_dir, exist_ok=True)
    model.eval()
    dl = tqdm(test_loader)

    for image, pth in dl:
        dl.set_description("Single-scale test")
        image = image.cuda()
        _, _, H, W = image.shape
        filename = pth[0]
        results = model(image)

        result = torch.squeeze(results[-1].detach()).cpu().numpy()

        result = (result - result.min()) / (result.max() - result.min())

        Image.fromarray((result * 255).astype(np.uint8)).save(join(save_png_dir, "%s.png" % filename))
        sio.savemat(join(save_mat_dir, '%s.mat' % filename), {'result': result})


def multiscale_test(model, test_loader, save_dir, epoch):
    model.eval()
    dl = tqdm(test_loader)
    if epoch != "":
        save_mat_dir = join(save_dir, epoch, 'mat')
        save_png_dir = join(save_dir, epoch, 'png')
    else:
        save_mat_dir = join(save_dir, 'mat')
        save_png_dir = join(save_dir, 'png')
    os.makedirs(save_mat_dir, exist_ok=True)
    os.makedirs(save_png_dir, exist_ok=True)

    # scale = [0.5, 1, 1.5]
    scale = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
    for image, pth in dl:
        dl.set_description("Multi-scale test")
        image = image[0]
        image_in = image.numpy().transpose((1, 2, 0))
        _, H, W = image.shape
        multi_fuse = np.zeros((H, W), np.float32)

        for k in range(0, len(scale)):
            im_ = cv2.resize(image_in, None, fx=scale[k], fy=scale[k], interpolation=cv2.INTER_LINEAR)
            im_ = im_.transpose((2, 0, 1))
            inputs = torch.unsqueeze(torch.from_numpy(im_).cuda(), 0)
            results = model(inputs)

            result = torch.squeeze(results[-1].detach()).cpu().numpy()
            fuse = cv2.resize(result, (W, H), interpolation=cv2.INTER_LINEAR)
            multi_fuse += fuse
        multi_fuse = multi_fuse / len(scale)
        ### rescale trick suggested by jiangjiang
        multi_fuse = (multi_fuse - multi_fuse.min()) / (multi_fuse.max() - multi_fuse.min())
        filename = pth[0]
        # result_out = Image.fromarray(((1-multi_fuse) * 255).astype(np.uint8))
        # result_out.save(join(save_dir, "%s.jpg" % filename))
        Image.fromarray((multi_fuse * 255).astype(np.uint8)).save(join(save_png_dir, "%s.png" % filename))
        sio.savemat(join(save_mat_dir, '%s.mat' % filename), {'result': multi_fuse})
