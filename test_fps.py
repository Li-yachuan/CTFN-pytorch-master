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
from thop import profile
from torchvision.utils import save_image

def test(model, test_loader):
    model.eval()

    flops_lst = []
    params = None
    for image, pth in test_loader:
        with torch.no_grad():
            image = image.cuda()
            flops, params = profile(model, (image,))
            flops_lst.append(flops)

    flops = torch.tensor(flops_lst).mean() / 1e9
    print('flops:', flops.item(), "G")
    print('params:', params / 1e6, "M")


    use_time = 0
    for image, pth in test_loader:
        with torch.no_grad():
            image = image.cuda()
            start_time = time.perf_counter()
            output = model(image)
            torch.cuda.synchronize()
            use_time += (time.perf_counter() - start_time)

    print('single scale fps: %f' % (len(test_loader) / use_time))

    use_time = 0
    scale = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
    for image, pth in test_loader:
        for k in range(0, len(scale)):
            image_in = image[0].numpy().transpose((1, 2, 0))
            im_ = cv2.resize(image_in, None, fx=scale[k], fy=scale[k], interpolation=cv2.INTER_LINEAR)
            im_ = im_.transpose((2, 0, 1))
            inputs = torch.unsqueeze(torch.from_numpy(im_).cuda(), 0)
            with torch.no_grad():
                inputs = inputs.cuda()
                start_time = time.perf_counter()
                output = model(inputs)
                torch.cuda.synchronize()
                use_time += (time.perf_counter() - start_time)

    print('multi scale fps: %f' % (len(test_loader) / use_time))
