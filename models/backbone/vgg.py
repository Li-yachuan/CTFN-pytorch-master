import torch.nn as nn
import torch
from os.path import isfile

class VGG16(nn.Module):

    def __init__(self, resume=None):
        super(VGG16, self).__init__()
        self.resume = resume

        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)

        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)

        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)

        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3,
                                 stride=1, padding=2, dilation=2)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3,
                                 stride=1, padding=2, dilation=2)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3,
                                 stride=1, padding=2, dilation=2)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.maxpool4 = nn.MaxPool2d(2, stride=1, ceil_mode=True)

        if self.resume:
            self.init_weight()

    def init_weight(self):
        assert isfile(self.resume), "No pretrained model found at {}".format(self.resume)

        print("=> Initialize VGG16 backbone")

        state_dict = torch.load(self.resume, map_location=torch.device("cpu"))
        if "state_dict" in state_dict.keys():
            state_dict = state_dict["state_dict"]
        self_state_dict = self.state_dict()
        for k, v in self_state_dict.items():
            if k in state_dict.keys():
                self_state_dict.update({k: state_dict[k]})
            else:
                raise Exception("In VGG backbone, {} is uninited".format(k))

        self.load_state_dict(self_state_dict)

        print("=> Pretrained Loaded")

    def forward(self, x):
        # VGG:extract feature

        conv1_1 = self.relu(self.conv1_1(x))
        conv1_2 = self.relu(self.conv1_2(conv1_1))
        pool1 = self.maxpool(conv1_2)
        conv2_1 = self.relu(self.conv2_1(pool1))
        conv2_2 = self.relu(self.conv2_2(conv2_1))
        pool2 = self.maxpool(conv2_2)

        conv3_1 = self.relu(self.conv3_1(pool2))
        conv3_2 = self.relu(self.conv3_2(conv3_1))
        conv3_3 = self.relu(self.conv3_3(conv3_2))
        pool3 = self.maxpool(conv3_3)

        conv4_1 = self.relu(self.conv4_1(pool3))
        conv4_2 = self.relu(self.conv4_2(conv4_1))
        conv4_3 = self.relu(self.conv4_3(conv4_2))
        pool4 = self.maxpool4(conv4_3)

        conv5_1 = self.relu(self.conv5_1(pool4))
        conv5_2 = self.relu(self.conv5_2(conv5_1))
        conv5_3 = self.relu(self.conv5_3(conv5_2))

        return [conv1_1, conv1_2,
                conv2_1, conv2_2,
                conv3_1, conv3_2, conv3_3,
                conv4_1, conv4_2, conv4_3,
                conv5_1, conv5_2, conv5_3]
