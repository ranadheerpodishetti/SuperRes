import torch
import torch.nn as nn
import numpy as np
import math
import dyn_conv
import unet_dynamic_convolution
import SRDenseNet
import SRCNN


class Ensemble(nn.module) :
    def __init__(self) :
        super(Ensemble, self).__init__()
        self.dense = SRDenseNet.SRDenseNet()
        self.srcnn = SRCNN.SRCNN3Dv2(scale_factor=(2, 4, 3))
        self.dyn1 = unet_dynamic_convolution.dynamic(2 , 1, 1, 3, 5, 1)

    def forward(self, x) :
        denseout = self.dense(x)
        srcnnout = self.srcnn(x)
        # concat along the channels, dim 1
        #dynamic conv code:
        out = torch.cat([denseout, srcnnout], dim=1)
        out = self.dyn1(out)

        return denseout, srcnnout, out


if __name__ == "__main__" :
    tensor = torch.rand((2, 1, 24, 16, 16)).cuda()
    model = Ensemble().cuda()
    k = model(tensor)
    print(k.size())
