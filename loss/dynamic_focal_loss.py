# coding=utf-8
import numpy as np
import torch
import random
import math
from torchvision.utils import save_image


def dy_focal_loss(inputs, targets, epoch, args):
    """
    之所以称为动态，是指focal loss中的alpha不是手动设置的超参，而是通过计算正负样本的比例得到的
    :param epoch:
    :param inputs: inputs is a 4 dimensional data nx1xhxw
    :param targets: targets is a 4 dimensional data nx1xhxw
    :return:
    """
    assert args.mu >= 0
    prob_wce = args.mu / (epoch + args.mu + 1e-7)
    prob_fl = 1- prob_wce

    weights = torch.zeros_like(inputs).cuda()
    pos = (targets == 1).sum()
    neg = (targets == 0).sum()
    valid = neg + pos
    weights[targets == 1] = neg * 1. / valid
    weights[targets == 0] = pos * args.balance / valid

    bce_loss = torch.nn.BCELoss(reduction='none')(inputs, torch.clamp_max(targets, 1))
    with torch.no_grad():
        pt = torch.exp(-bce_loss)
    F_loss = (prob_fl * (1 - pt) ** args.gamma + prob_wce) * weights * bce_loss

    return torch.sum(F_loss)


