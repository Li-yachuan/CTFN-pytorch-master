#!/user/bin/python
# -*- encoding: utf-8 -*-

import os, sys
import torch
import torch.nn as nn
import numpy as np
from os.path import join
import random


class Logger(object):
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()


class Averagvalue(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, filename='checkpoint.pth'):
    torch.save(state, filename)


def load_pretrained(model, fname, optimizer=None):
    """
    resume training from previous checkpoint
    :param fname: filename(with path) of checkpoint file
    :return: model, optimizer, checkpoint epoch
    """
    if os.path.isfile(fname):
        print("=> loading checkpoint '{}'".format(fname))
        checkpoint = torch.load(fname)
        model.load_state_dict(checkpoint['state_dict'])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
            return model, optimizer, checkpoint['epoch']
        else:
            return model, checkpoint['epoch']
    else:
        print("=> no checkpoint found at '{}'".format(fname))


def setup_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = False  # 这个跟空洞卷积不兼容
    # 该函数的作用是通过预先统计特征图的大小，然后在之后的卷积过程中选用合适的矩阵计算来代替卷积，从而达到加速的目的，
    # 然而空洞卷积是一个动态的过程，因此需要每次都重新统计，反而使计算速度极大的减慢
    torch.backends.cudnn.benchmark = False

    torch.cuda.manual_seed_all(seed)





def log_lr(optimizer, nm=None, onlyone=True):
    if nm is not None:
        for n, p in zip(nm, optimizer.state_dict()['param_groups']):
            print(n, 'lr:', p['lr'])
    elif onlyone:
        print('Current lr(only show the 1st):', optimizer.state_dict()['param_groups'][0]["lr"])
    else:
        for p in optimizer.state_dict()['param_groups']:
            print('lr:', p['lr'])


def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print("=======================Model Param=========================")
    print('Total:' ,total_num/1e6,"(M)", 'Trainable:', trainable_num/1e6,"(M)")
    return {'Total': total_num, 'Trainable': trainable_num}
