#!/user/bin/python
# -*- encoding: utf-8 -*-
import torch.nn as nn
import torch.nn.functional as F


class DefaultConv(nn.Module):
    def __init__(self, in_ch=5, out_ch=1):
        super(DefaultConv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)
        self.init_weight()

    def forward(self, x):
        out = self.conv(x)
        return out

    def init_weight(self):
        print("=> Initialization by Gaussian(0, 0.01)")

        for ly in self.modules():
            if isinstance(ly, nn.Conv2d):
                ly.weight.data.normal_(0, 0.01)
                if not ly.bias is None: ly.bias.data.zero_()


class PPW(nn.Module):
    def __init__(self, in_ch=5, out_ch=1):
        super(PPW, self).__init__()
        self.att = nn.Sequential(
            nn.Conv2d(in_ch, 16, 3, padding=1),
            nn.GroupNorm(4, 16),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.GroupNorm(4, 16),
            nn.ReLU(),
            nn.Conv2d(16, 1, 3, padding=1),
            nn.Sigmoid()
        )

        self.conv = nn.Conv2d(in_ch, out_ch, 1)
        self.init_weight()

    def forward(self, x):

        atten = self.att(x)
        out = self.conv(x) * (1 + atten)
        return out

    def init_weight(self):
        print("=> Initialization by Gaussian(0, 0.01)")

        for ly in self.modules():
            if isinstance(ly, nn.Conv2d):
                ly.weight.data.normal_(0, 0.01)
                if not ly.bias is None: ly.bias.data.zero_()


class CoFusion(nn.Module):

    def __init__(self, in_ch=5, out_ch=5):
        super(CoFusion, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, 64, kernel_size=3,
                               stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3,
                               stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, out_ch, kernel_size=3,
                               stride=1, padding=1)
        self.relu = nn.ReLU()

        self.norm_layer1 = nn.GroupNorm(4, 64)
        self.norm_layer2 = nn.GroupNorm(4, 64)

        self.init_weight()

    def init_weight(self):
        print("=> Initialization CoFusion by Gaussian(0, 0.01)")

        for ly in self.modules():
            if isinstance(ly, nn.Conv2d):
                ly.weight.data.normal_(0, 0.01)
                if not ly.bias is None: ly.bias.data.zero_()

    def forward(self, x):
        attn = self.relu(self.norm_layer1(self.conv1(x)))
        attn = self.relu(self.norm_layer2(self.conv2(attn)))
        attn = F.softmax(self.conv3(attn), dim=1)

        return ((x * attn).sum(1)).unsqueeze(1)
