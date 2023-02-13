# coding=utf-8
from torch import nn
import torch.nn.functional as F

class NoFPN(nn.Module):
    def __init__(self, args):
        super(NoFPN, self).__init__()

        self.gn_layer = nn.ModuleList([nn.GroupNorm(3, 21) for i in range(5)])

        # self.gn_layer = nn.ModuleList([nn.Identity() for i in range(5)])

    def forward(self, x):
        return [m(f) for m, f in zip(self.gn_layer, x)]



class StFPN(nn.Module):
    """
    固定的FPN，没有可学习参数
    """

    def __init__(self, args, uni_dim, group=3):
        super(StFPN, self).__init__()

        self.gn_layer = nn.ModuleList([nn.GroupNorm(group, uni_dim) for i in range(5)])

        self.resize = self.interpolate

    def forward(self, x):
        [so1, so2, so3, so4, so5] = [m(f) for m, f in zip(self.gn_layer, x)]

        so4 = (so4 + self.resize((so4, so5))) / 2
        so3 = (so3 + self.resize((so3, so4))) / 2
        so2 = (so2 + self.resize((so2, so3))) / 2
        so1 = (so1 + self.resize((so1, so2))) / 2

        return [so1, so2, so3, so4, so5]

    @staticmethod
    def interpolate(x):
        b, a = x
        return F.interpolate(a, size=(b.size(2), b.size(3)), mode='bilinear')
