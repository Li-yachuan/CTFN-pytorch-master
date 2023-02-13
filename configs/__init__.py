#!/user/bin/python
# -*- encoding: utf-8 -*-

from os.path import join


class Config(object):
    def __init__(self, args):
        self.aug =True
        self.batch_size = 1
        self.itersize = 10
        self.msg_iter = 1000
        self.pretrained = "./models/pretrained/vgg16.pth"

        # =============== optimizer
        self.lr = args.lr if args.lr else 1e-4
        self.wd = args.wd if args.wd else 2e-4
        self.momentum = 0.9

        # Period of learning rate decay.
        if args.stepsize is None:
            self.stepsize= [1000]
        else:
            self.stepsize = [int(i) for i in args.stepsize.split("-")]
            for i in range(len(self.stepsize)-1):
                assert self.stepsize[i+1]>self.stepsize[i]

        self.gamma = 0.1

        # ================ dataset
        self.data = args.dataset
        # self.resume = "./pretrained/{}.pth".format(self.data)

        if self.data.lower() == "bsds":
            self.dataset = "../00Dataset/HED-BSDS"
            self.train_list = 'train_pair_KD.lst'
            self.test_list = 'test.lst'
            self.mean = (104.00698793, 116.66876762, 122.67891434)
        elif self.data.lower() == "bsds-pascal":
            self.dataset = "../00Dataset"
            self.train_list = 'HED-BSDS/bsds_pascal_train_pair_KD.lst'
            self.test_list = 'HED-BSDS/bsds_pascal_test.lst'
            self.mean = (104.00698793, 116.66876762, 122.67891434)

        elif self.data.lower() == "pascal":
            self.dataset = "../00Dataset"
            self.train_list = 'PASCAL/train_pair_KD.lst'
            self.test_list = 'PASCAL/bsds_test.lst'
            self.mean = (104.00698793, 116.66876762, 122.67891434)
        # elif self.data.lower() == "nyud-rgb":
        #     self.dataset = "../00Dataset/NYUD"
        #     self.train_list = 'image-train.lst'
        #     self.test_list = 'image-test.lst'
        #     self.mean = (104.00698793, 116.66876762, 122.67891434)
        # elif self.data.lower() == "nyud-hha":
        #     self.dataset = "../00Dataset/NYUD"
        #     self.train_list = 'hha-train.lst'
        #     self.test_list = 'hha-test.lst'
        #     self.mean = (109.92, 88.24, 127.42)
        elif self.data.lower() == "nyud-rgb":
            self.dataset = "../00Dataset/NYUDv2"
            self.train_list = 'image-train-KD.lst'
            self.test_list = 'image-test.lst'
            self.mean = (104.00698793, 116.66876762, 122.67891434)
        # elif self.data.lower() == "biped":
        #     self.dataset = "../00Dataset/BIPEDv2"
        #     self.train_list = 'train_pair.lst'
        #     self.test_list = 'test.lst'
        #     self.mean = (103.939, 116.779, 123.68)
        else:
            raise Exception("incorrect dataset!")
