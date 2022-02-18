import dyn_conv
from dyn_conv import Dynamic_conv3d
import torch
import torch.nn as nn
import numpy as np



class double_conv(nn.Module):
    """
    Convolution Block
    """

    def __init__(self, in_channels, out_channels, k_size, stride, bias=True):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=k_size,
                      stride=stride, padding=k_size//2, bias=bias),
            nn.BatchNorm3d(num_features=out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=k_size,
                      stride=stride, padding=k_size//2, bias=bias),
            nn.BatchNorm3d(num_features=out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class dynamic_conv(nn.Module):
    """
    Convolution Block
    """

    def __init__(self, in_channels, out_channels, k_size, stride, bias=True):
        super(dynamic_conv, self).__init__()
        self.dynconv = Dynamic_conv3d(in_planes=in_channels, out_planes=out_channels, kernel_size=k_size, ratio=0.25,
                                      padding=k_size // 2, stride=stride, bias=True)
        self.dynconv.update_temperature()

    def forward(self, x):
        x = self.dynconv(x)

        return x


##  maxpooling can be replaced with pixel_unshuffle
## TODO
class concatenated(nn.Module):
    """
    Convolution Block
    """

    def __init__(self, in_channels, out_channels, k_1, k_3, k_5, stride, bias=True):
        super(concatenated, self).__init__()
        self.conv_1 = dynamic_conv(in_channels, out_channels, k_1, stride)
        self.conv_3 = dynamic_conv(in_channels, out_channels, k_3, stride)
        self.conv_5 = dynamic_conv(in_channels, out_channels, k_5, stride)
        self.max_pool = nn.MaxPool3d(k_3, stride, k_3 // 2)
        self.relu = nn.ReLU(inplace=True)
        self.conv_out = nn.Conv3d(out_channels*2, out_channels, k_3, 1, k_3 // 2)

    def forward(self, x):
        x_1 = (self.conv_1(x))
        x_3 = (self.conv_3(x))

        x_5 = (self.conv_5(x))
        # maxpool = self.max_pool(x)
        # maxpool = self.conv(maxpool)
        x = torch.cat([x_1, x_3], dim=1)
        k1 =  x.detach().cpu().numpy()
        k1 = np.sum(k1)
        if np.isnan(k1) == True:
            print('True')
        x = self.conv_out(x)
        x = self.relu(x)
        return x

class dynamic(nn.Module):
    """
    Convolution Block
    """

    def __init__(self, in_channels, out_channels, k_1, k_3, k_5, stride, bias=True):
        super(dynamic, self).__init__()
        self.dyn = concatenated(in_channels, out_channels,k_1, k_3, k_5, stride,bias)
        #self.doubleconv = double_conv(out_channels*3, out_channels, k_3, 1)


    def forward(self, x):
        x = self.dyn(x)
        #print(x.size())
        #x = self.doubleconv(x)
        return x


if __name__ == '__main__':
    p = torch.randn(2, 1, 20, 20, 20)
    # model = double_conv(1, 64, 3, 1)
    # model1 = double_conv(1, 64, 5, 1).to('cuda')
    # model2 = double_conv(1,64,1,1).to('cuda')
    # p = p.to('cuda:0')
    # model.to('cuda')
    # inpu = model(p)
    # inps = model1(p)
    # inp1 = model2(p)
    # p = torch.cat([inpu, inps, inp1], dim = 1)
    # print(p.size())
    model = dynamic(1, 4, 1, 3, 5, 1)
    l = model(p)
    print(l.size())
