# coding=utf-8
import torch


def sample_balance_loss(inputs, softlabel, args, epoch):
    """
    mu=10
    """
    with torch.no_grad():

        weight = args.mu**softlabel*torch.abs(inputs.detach() - softlabel) ** args.gamma
        weight /= weight.max()

    bce_loss = torch.nn.BCELoss(weight=weight, reduction='sum')(inputs, softlabel)
    return bce_loss
