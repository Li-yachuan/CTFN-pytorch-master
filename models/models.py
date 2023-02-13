#!/user/bin/python
# -*- encoding: utf-8 -*-

import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from os.path import isfile, join
from .cofusion import CoFusion, DefaultConv, PPW
from .FPN import NoFPN, StFPN
from .backbone import VGG16


class Network(nn.Module):
    def __init__(self, cfg, args):
        super(Network, self).__init__()
        self.cfg = cfg
        self.args = args
        self.uni_dim = 21

        self.backbone = VGG16(self.cfg.pretrained)

        self.conv1_1_down = nn.Conv2d(64, self.uni_dim, 1)
        self.conv1_2_down = nn.Conv2d(64, self.uni_dim, 1)

        self.conv2_1_down = nn.Conv2d(128, self.uni_dim, 1)
        self.conv2_2_down = nn.Conv2d(128, self.uni_dim, 1)

        self.conv3_1_down = nn.Conv2d(256, self.uni_dim, 1)
        self.conv3_2_down = nn.Conv2d(256, self.uni_dim, 1)
        self.conv3_3_down = nn.Conv2d(256, self.uni_dim, 1)

        self.conv4_1_down = nn.Conv2d(512, self.uni_dim, 1)
        self.conv4_2_down = nn.Conv2d(512, self.uni_dim, 1)
        self.conv4_3_down = nn.Conv2d(512, self.uni_dim, 1)

        self.conv5_1_down = nn.Conv2d(512, self.uni_dim, 1)
        self.conv5_2_down = nn.Conv2d(512, self.uni_dim, 1)
        self.conv5_3_down = nn.Conv2d(512, self.uni_dim, 1)

        self.score_dsn1 = nn.Conv2d(self.uni_dim, 1, 1)
        self.score_dsn2 = nn.Conv2d(self.uni_dim, 1, 1)
        self.score_dsn3 = nn.Conv2d(self.uni_dim, 1, 1)
        self.score_dsn4 = nn.Conv2d(self.uni_dim, 1, 1)
        self.score_dsn5 = nn.Conv2d(self.uni_dim, 1, 1)

        self.weight_deconv2 = self.make_bilinear_weights(4, 1).cuda()
        self.weight_deconv3 = self.make_bilinear_weights(8, 1).cuda()
        self.weight_deconv4 = self.make_bilinear_weights(16, 1).cuda()
        self.weight_deconv5 = self.make_bilinear_weights(16, 1).cuda()

        self.fpn_dict = {
            "StFPN": StFPN(args, self.uni_dim),
            "NoFPN": NoFPN(args)
        }
        self.fpn = self.fpn_dict[args.fpn]

        self.cofuse_dict = {
            "CoFusion": CoFusion(),
            "Default": DefaultConv(),
            "PPW": PPW(),
        }
        self.attention = self.cofuse_dict[args.atten]

        self.init_weight() if args.resume is None else self.load_checkpoint(args.resume)

    def init_weight(self):
        print("=> Initialization by normal_(0, 0.01)")

        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                ly.weight.data.normal_(0, 0.01)
                if not ly.bias is None: ly.bias.data.zero_()

        if hasattr(self.args.fpn, 'init_weight'):
            self.fpn.init_weight()
        if hasattr(self.attention, 'init_weight'):
            self.attention.init_weight()

    def load_checkpoint(self, resume):
        if isfile(resume):
            checkpoint = torch.load(resume)
            self.load_state_dict(checkpoint['state_dict'])
            print("=> Loaded checkpoint '{}'".format(resume))
        else:
            raise Exception("No checkpoint found at '{}'".format(resume))

    def forward(self, x):
        img_H, img_W = x.shape[2], x.shape[3]
        [conv1_1, conv1_2,
         conv2_1, conv2_2,
         conv3_1, conv3_2, conv3_3,
         conv4_1, conv4_2, conv4_3,
         conv5_1, conv5_2, conv5_3] = self.backbone(x)

        conv1_down = self.conv1_1_down(conv1_1) + self.conv1_2_down(conv1_2)
        conv2_down = self.conv2_1_down(conv2_1) + self.conv2_2_down(conv2_2)
        conv3_down = self.conv3_1_down(conv3_1) + self.conv3_2_down(conv3_2) + self.conv3_3_down(conv3_3)
        conv4_down = self.conv4_1_down(conv4_1) + self.conv4_2_down(conv4_2) + self.conv4_3_down(conv4_3)
        conv5_down = self.conv5_1_down(conv5_1) + self.conv5_2_down(conv5_2) + self.conv5_3_down(conv5_3)

        conv1_down, conv2_down, conv3_down, conv4_down, conv5_down = \
            self.fpn((conv1_down, conv2_down, conv3_down, conv4_down, conv5_down))

        so1 = self.score_dsn1(conv1_down)
        so2 = self.score_dsn2(conv2_down)
        so3 = self.score_dsn3(conv3_down)
        so4 = self.score_dsn4(conv4_down)
        so5 = self.score_dsn5(conv5_down)

        so2 = F.conv_transpose2d(so2, self.weight_deconv2, stride=2)
        so3 = F.conv_transpose2d(so3, self.weight_deconv3, stride=4)
        so4 = F.conv_transpose2d(so4, self.weight_deconv4, stride=8)
        so5 = F.conv_transpose2d(so5, self.weight_deconv5, stride=8)

        results = [self.resize(r, img_H, img_W) for r in [so1, so2, so3, so4, so5]]
        fuse = self.attention(torch.cat(results, dim=1))
        results.append(fuse)

        return [torch.sigmoid(r) for r in results]

    @staticmethod
    def make_bilinear_weights(size, num_channels):
        factor = (size + 1) // 2
        if size % 2 == 1:
            center = factor - 1
        else:
            center = factor - 0.5
        og = np.ogrid[:size, :size]
        filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
        # print(filt)
        filt = torch.from_numpy(filt)
        w = torch.zeros(num_channels, num_channels, size, size)
        w.requires_grad = False
        for i in range(num_channels):
            for j in range(num_channels):
                if i == j:
                    w[i, j] = filt
        return w

    @staticmethod
    def resize(data, h, w):
        return torch.nn.functional.interpolate(data, (h, w))
