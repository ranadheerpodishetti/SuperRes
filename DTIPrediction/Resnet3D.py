#!/usr/bin/env python

"""
Original file Resnet2Dv2b14 of NCC1701
"""

import torch.nn as nn
import torch.nn.functional as F
import torch
import pixel_shuffle_new


# from utils.TorchAct.pelu import PELU_oneparam as PELU

__author__ = "Soumick Chatterjee"
__copyright__ = "Copyright 2018, Soumick Chatterjee & OvGU:ESF:MEMoRIAL"
__credits__ = ["Soumick Chatterjee"]

__license__ = "GPL"
__version__ = "1.0.0"
__email__ = "soumick.chatterjee@ovgu.de"
__status__ = "Under Testing"


class ResidualBlock(nn.Module):
    def __init__(self, in_features, PReLU, norm):
        super(ResidualBlock, self).__init__()

        conv_block = [nn.ReplicationPad3d(1),
                      nn.Conv3d(in_features, in_features, in_features, 3),
                      norm(in_features),
                      PReLU(),
                      nn.Dropout3d(p=0.2),
                      nn.ReplicationPad3d(1),
                      nn.Conv3d(in_features, in_features, 3),
                      norm(in_features)]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)


class ResNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, res_blocks=14, starting_n_features=64, updown_blocks=2,
                 is_PReLU_leaky=True, final_out_sigmoid=True,
                 do_batchnorm=True):  # should use 14 as that gives number of trainable parameters close to number of possible pixel values in a image 256x256
        super(ResNet, self).__init__()

        if is_PReLU_leaky:
            PReLU = nn.PReLU
        else:
            PReLU = nn.ReLU
        if do_batchnorm:
            norm = nn.BatchNorm3d
        else:
            norm = nn.InstanceNorm3d

        # Initial convolution block
        model = [nn.ReplicationPad3d(3),
                 nn.Conv3d(in_channels, starting_n_features, 7),
                 norm(starting_n_features),
                 PReLU()]

        # Downsampling
        in_features = starting_n_features
        out_features = in_features * 2
        for _ in range(updown_blocks):
            model += [nn.Conv3d(in_features, out_features, 3, stride=2, padding=1),
                      norm(out_features),
                      PReLU()]
            in_features = out_features
            out_features = in_features * 2

        # Residual blocks
        for _ in range(res_blocks):
            model += [ResidualBlock(in_features, PReLU, norm)]

        # Upsampling
        out_features = in_features // 2
        for _ in range(updown_blocks):
            model += [pixel_shuffle_new.pixel_shuffle_new(in_features, out_features,1,1),
                      norm(out_features),
                      PReLU()]
            in_features = out_features
            out_features = in_features // 2

        # Output layer
        model += [nn.ReplicationPad3d(3),
                  nn.Conv3d(starting_n_features, out_channels, 7)]

        # final activation
        if final_out_sigmoid:
            model += [nn.Tanh(), ]  # [ nn.Sigmoid(), ]
        else:
            model += [PReLU(), ]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)


if __name__ == "__main__":
    tensor = torch.rand((1, 4, 8, 8, 8)).cuda()
    model = ResNet(4,8,10).cuda()
    print(model)
    k = model(tensor)
    print(k.size())
