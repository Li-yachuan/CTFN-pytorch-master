import torch
from torch.optim import lr_scheduler


class Optimizer():
    def __init__(self, cfg, args):
        self.cfg = cfg
        self.args = args

    def __call__(self, net):

        net_parameters_id = {}
        net_parameters_id['backbone1_4.weight'] = []
        net_parameters_id['backbone1_4.bias'] = []
        net_parameters_id['backbone5.weight'] = []
        net_parameters_id['backbone5.bias'] = []
        net_parameters_id['conv_down.weight'] = []
        net_parameters_id['conv_down.bias'] = []
        net_parameters_id['score_dsn.weight'] = []
        net_parameters_id['score_dsn.bias'] = []
        net_parameters_id['attn.weight'] = []
        net_parameters_id['attn.bias'] = []
        net_parameters_id['fpn.weight'] = []
        net_parameters_id['fpn.bias'] = []

        for pname, p in net.named_parameters():

            if "backbone" in pname:
                if "5" in pname:
                    if 'weight' in pname:
                        net_parameters_id['backbone5.weight'].append(p)
                    else:
                        net_parameters_id['backbone5.bias'].append(p)
                else:
                    if 'weight' in pname:
                        net_parameters_id['backbone1_4.weight'].append(p)
                    else:
                        net_parameters_id['backbone1_4.bias'].append(p)

            elif 'down' in pname:
                if 'weight' in pname:
                    net_parameters_id['conv_down.weight'].append(p)
                elif 'bias' in pname:
                    net_parameters_id['conv_down.bias'].append(p)

            elif 'score_dsn' in pname:
                if 'weight' in pname:
                    net_parameters_id['score_dsn.weight'].append(p)
                else:
                    net_parameters_id['score_dsn.bias'].append(p)

            elif 'attention' in pname:
                if 'weight' in pname:
                    net_parameters_id['attn.weight'].append(p)
                else:
                    net_parameters_id['attn.bias'].append(p)

            elif 'fpn' in pname:
                if 'weight' in pname:
                    net_parameters_id['fpn.weight'].append(p)
                else:
                    net_parameters_id['fpn.bias'].append(p)

            else:
                raise Exception("{} is not grouped".format(pname))

        SGD_params = []

        SGD_params.append(
            {'params': net_parameters_id['backbone1_4.weight'], 'lr': self.cfg.lr * 0.01, 'weight_decay': self.cfg.wd})
        SGD_params.append(
            {'params': net_parameters_id['backbone1_4.bias'], 'lr': self.cfg.lr * 0.02, 'weight_decay': 0.})
        SGD_params.append(
            {'params': net_parameters_id['backbone5.weight'], 'lr': self.cfg.lr * 1, 'weight_decay': self.cfg.wd})
        SGD_params.append(
            {'params': net_parameters_id['backbone5.bias'], 'lr': self.cfg.lr * 2, 'weight_decay': 0.})

        SGD_params.append(
            {'params': net_parameters_id['conv_down.weight'], 'lr': self.cfg.lr * 0.1, 'weight_decay': self.cfg.wd})
        SGD_params.append(
            {'params': net_parameters_id['conv_down.bias'], 'lr': self.cfg.lr * 0.2, 'weight_decay': 0.})

        SGD_params.append(
            {'params': net_parameters_id['score_dsn.weight'], 'lr': self.cfg.lr * 0.01, 'weight_decay': self.cfg.wd})
        SGD_params.append(
            {'params': net_parameters_id['score_dsn.bias'], 'lr': self.cfg.lr * 0.02, 'weight_decay': 0.})
        SGD_params.append(
            {'params': net_parameters_id['attn.weight'], 'lr': self.cfg.lr * 1., 'weight_decay': self.cfg.wd})
        SGD_params.append(
            {'params': net_parameters_id['attn.bias'], 'lr': self.cfg.lr * 2., 'weight_decay': 0.})
        SGD_params.append(
            {'params': net_parameters_id['fpn.weight'], 'lr': self.cfg.lr * 1, 'weight_decay': self.cfg.wd})
        SGD_params.append(
            {'params': net_parameters_id['fpn.bias'], 'lr': self.cfg.lr * 2, 'weight_decay': 0.})

        optim = torch.optim.SGD(SGD_params, lr=self.cfg.lr, momentum=self.cfg.momentum, weight_decay=self.cfg.wd)

        if self.args.scheduler == "StepLR":
            scheduler = lr_scheduler.StepLR(optim, step_size=self.cfg.stepsize[0], gamma=self.cfg.gamma)
        elif self.args.scheduler == "MultiStepLR":
            scheduler = lr_scheduler.MultiStepLR(optim, milestones=self.cfg.stepsize, gamma=self.cfg.gamma)
        elif self.args.scheduler == "CosineAnnealingLR":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=self.cfg.stepsize[0], eta_min=self.cfg.lr * 0.1)
        else:
            raise Exception("incorrect scheduler")
        return optim, scheduler
