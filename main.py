#!/user/bin/python
# -*- encoding: utf-8 -*-

import argparse
import os
import sys
from os.path import join, isfile, abspath, dirname
from torch.utils.data import DataLoader
from configs import Config
from data.data_loader import MyDataLoader
from models import Network
from models import Optimizer
from test import multiscale_test, test, test_side_edge
from test_fps import test as test_fps
from train import train
from utils import Logger, get_parameter_number, setup_seed
import time

parser = argparse.ArgumentParser(description='Mode Selection')
parser.add_argument('--mode', default='train', type=str, choices={"train", "test", "fps", "sidedge"},
                    help="Setting models for training or testing"
                         "fps is used to test the speed of model"
                         "sidedge is a short of `side edge`,to deside whither output the result of side edge,"
                         "fps and sidedge is only used in test model")
parser.add_argument('--resume', default=None, help='model path for loading trained models')
parser.add_argument('-r', '--randseed', type=int, default=None,
                    help="rand seed: 3 for bsds and biped, 2 for nyud")


parser.add_argument('-e', '--max_epoch', type=int, default=20, help="")

parser.add_argument('--scheduler', default="StepLR", choices=["StepLR", "MultiStepLR","CosineAnnealingLR"])
parser.add_argument('--stepsize', default=None,help='Period of learning rate decay.  `10-16`')


parser.add_argument('-b', '--balance', type=float, default=None,
                    help='balance edge and noedge,it is used at edge:1.1 for bsds and biped,'
                         '1.2 for nyud')
parser.add_argument('-l', '--loss', default="WCE", choices=["DFL", "WCE","SBL"])
parser.add_argument('-a', '--atten', default="PPW",
                    choices=["CoFusion", "Default", "PPW"],
                    help="choose a kind of attention")

parser.add_argument('--lr', default=None, type=float)
parser.add_argument('--wd', default=None, type=float)

parser.add_argument('--gamma', default=1., type=float,
                    help="hy-para of focal loss,is different from cfg.gamma,used to balence hard/easy sample")
parser.add_argument('--mu', default=0.5, type=float, help="used to trans WCE 2 Focal Loss")

parser.add_argument('--gpu', default='0')
parser.add_argument('-f', '--fpn', default="StFPN", choices=["NoFPN", "StFPN"])

parser.add_argument('-s', '--save_pth', required=True)

parser.add_argument('-n', '--note', required=True, help="record the change of every exp")
parser.add_argument('-d', '--dataset', required=True,
                    help="bsds bsds-pascal nyud-rgb nyud-hha nyudv2-rgb nyudv2-hha biped")

parser.add_argument('--start_epoch', type=int, default=0, required=False)
parser.add_argument('--ss', action="store_true",
                    help="only test on single scale,due to img in BIPED is too big, "
                         "multi scale is not support")

args = parser.parse_args()
# torch.cuda.current_device()
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

cfg = Config(args)

if 'biped' in args.dataset.lower():
    assert args.ss
    if args.randseed is None:
        args.randseed = 3
    if args.balance is None:
        args.balance = 1.1
elif 'bsds' in args.dataset.lower():
    if args.randseed is None:
        args.randseed = 3
    if args.balance is None:
        args.balance = 1.1
elif 'pascal' in args.dataset.lower():
    if args.randseed is None:
        args.randseed = 3
    if args.balance is None:
        args.balance = 1.1

elif 'nyud' in args.dataset.lower():
    if args.randseed is None:
        args.randseed = 2
    if args.balance is None:
        args.balance = 1.2

else:
    raise Exception("not exist dataset")

setup_seed(args.randseed)

if args.lr is None:
    args.lr = cfg.lr

THIS_DIR = abspath(dirname(__file__))
TMP_DIR = join(THIS_DIR, "output2", args.save_pth)

os.makedirs(TMP_DIR, exist_ok=True)


def main():
    # log
    log = Logger(join(TMP_DIR, "log-{}.txt".format(time.time())))
    sys.stdout = log

    print('============== paramter cfg =============================')
    for (key, value) in cfg.__dict__.items():
        print('{0:15} | {1}'.format(key, value))
    print('============== paramter args =============================')
    for (key, value) in vars(args).items():
        print('{0:15} | {1}'.format(key, value))

    model = Network(cfg, args)

    get_parameter_number(model)

    print('=> Load model')

    model.cuda()

    test_dataset = MyDataLoader(args=args, cfg=cfg, split="test")

    test_loader = DataLoader(test_dataset, batch_size=1, drop_last=True, shuffle=False)

    if args.mode == "fps":
        print("test FPS mode ...")
        # assert isfile(args.resume), "No checkpoint is found at '{}'".format(args.resume)
        #
        # model.load_checkpoint(args.resume)

        test_fps(model, test_loader)

    elif args.mode == "test":
        print("test mode ...")
        assert isfile(args.resume), "No checkpoint is found at '{}'".format(args.resume)

        model.load_checkpoint(args.resume)
        if not args.ss:
            multiscale_test(model,
                            test_loader,
                            join(TMP_DIR, "multi-sacle-test"),
                            args.resume.split('/')[-1].split('.')[0])
        else:
            test(model, test_loader, save_dir=join(TMP_DIR, "test", "single-scale-test"))

    elif args.mode == "sidedge":
        test_side_edge(model, test_loader, save_dir=join(TMP_DIR, "side-edge-test", "mutli_scale_test"))

    else:
        print("train mode ...")
        train_dataset = MyDataLoader(args=args, cfg=cfg, split="train", transform=cfg.aug)

        train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, drop_last=True, shuffle=True)

        if args.resume is not None:
            pth_nm = os.path.basename(args.resume).split("-")
            # epoch-0-checkpoint.pth
            print(pth_nm)
            assert len(pth_nm) == 3
            assert pth_nm[0] == "epoch"
            assert pth_nm[1].isdecimal()
            assert pth_nm[2] == "checkpoint.pth"
            args.start_epoch = int(pth_nm[1])

        model.train()

        # optimizer

        optim, scheduler = Optimizer(cfg,args)(model)

        from utils import log_lr
        log_lr(optim)

        train_loss = []
        train_loss_detail = []

        for epoch in range(args.start_epoch, args.start_epoch + args.max_epoch):
            tr_avg_loss, tr_detail_loss = train(cfg, args, train_loader, model, optim, scheduler, epoch,
                                                save_dir=join(TMP_DIR, "train", "epoch-%d-training-record" % epoch))
            if not args.ss:
                multiscale_test(model, test_loader, join(TMP_DIR, "multi-test"),
                                "epoch-{:0>2d}-multiscale_test".format(epoch))

            test(model, test_loader, join(TMP_DIR, "single-test"),
                 "epoch-{:0>2d}-singlescale_test".format(epoch))

            log.flush()

            train_loss.append(tr_avg_loss)
            train_loss_detail += tr_detail_loss


if __name__ == '__main__':
    main()
