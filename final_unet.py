import torch
import torch.nn as nn
import torch.nn.functional as F
import dyn_conv
import unet_dynamic_convolution
import pixel_unshuffle
import pixel_unshuffle_new
import pixel_shuffle_icnr
import pixel_shuffle
import pixel_shuffle_new


# -------------------------------------------------------------------------------------------------------------------------------------------------##


# -------------------------------------------------------------------------------------------------------------------------------------------------##

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


class up_conv(nn.Module):
    """
    Up Convolution Block
    """

    # def __init__(self, in_ch, out_ch):
    def __init__(self, in_c, out_c, kernel, stride, bias=True):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv3d(in_channels=in_c, out_channels=out_c, kernel_size=kernel,
                      stride=stride, padding=kernel // 2, bias=bias),
            nn.BatchNorm3d(num_features=out_c),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.up(x)
        return x


# num_features 4 --kernel_size_1 3 --stride_1 1 --kernel_size_2 2 --stride_2 2 --kernel_size_3 1
class UNet(nn.Module):
    def __init__(self, num_features, kernel_size_1, stride_1, kernel_size_2, stride_2, kernel_size_3):
        super(UNet, self).__init__()


        self.pixel_unshuffle_1 = pixel_unshuffle_new.pixel_unshuffle_new(num_features, num_features, 3, 1)
        self.pixel_unshuffle_2 = pixel_unshuffle_new.pixel_unshuffle_new(num_features * 2, num_features * 2, 3, 1)
        self.pixel_unshuffle_3 = pixel_unshuffle_new.pixel_unshuffle_new(num_features * 4, num_features * 4, 3, 1)
        self.pixel_unshuffle_4 = pixel_unshuffle_new.pixel_unshuffle_new(num_features * 8, num_features * 8, 3, 1)

        self.pixel_1 = pixel_shuffle_new.pixel_shuffle_new(num_features * 16, num_features * 16 * (2 ** 2),
                                                           kernel_size_1, stride_1)
        self.pixel_2 = pixel_shuffle_new.pixel_shuffle_new(num_features * 8, num_features * 8 * (2 ** 2),
                                                           kernel_size_1, stride_1)
        self.pixel_3 = pixel_shuffle_new.pixel_shuffle_new(num_features * 4, num_features * 4 * (2 ** 2),
                                                           kernel_size_1, stride_1)
        self.pixel_4 = pixel_shuffle_new.pixel_shuffle_new(num_features * 2, num_features * 2 * (2 ** 2),
                                                           kernel_size_1, stride_1)

        self.dyn_1 = unet_dynamic_convolution.dynamic(1, num_features, 1, 3, 5, 1)
        self.dyn_2 = unet_dynamic_convolution.dynamic(num_features, num_features * 2, 1, 3, 5, 1)
        self.dyn_3 = unet_dynamic_convolution.dynamic(num_features * 2, num_features * 4, 1, 3, 5, 1)
        self.dyn_4 = unet_dynamic_convolution.dynamic(num_features * 4, num_features * 8, 1, 3, 5, 1)
        self.dyn_5 = unet_dynamic_convolution.dynamic(num_features * 8, num_features * 16, 1, 3, 5, 1)

        self.up_dyn_5 = unet_dynamic_convolution.dynamic(num_features * 16, num_features * 8, 1, 3, 5, 1)
        self.up_dyn_4 = unet_dynamic_convolution.dynamic(num_features * 8, num_features * 4, 1, 3, 5, 1)
        self.up_dyn_3 = unet_dynamic_convolution.dynamic(num_features * 4, num_features * 2, 1, 3, 5, 1)
        self.up_dyn_2 = unet_dynamic_convolution.dynamic(num_features * 2, num_features * 1, 1, 3, 5, 1)
        self.up_dyn_1 = unet_dynamic_convolution.dynamic(num_features * 1, 1, 1, 3, 5, 1)

        self.Conv_d3 = double_conv(num_features * 2, 1, kernel_size_1, stride_1)
        self.Conv_d4 = double_conv(num_features * 4, 1, kernel_size_1, stride_1)
        # self.up_trans_1 = nn.Conv3d(in_channels= num_features*16, out_channels= num_features*8, kernel_size= kernel_size_3)

        self.Conv = nn.Conv3d(in_channels=1, out_channels=1, kernel_size=kernel_size_3)

    def forward(self, image):
        # encoder
        print(image.size())
        # x1 = self.Conv1(image)
        x1 = self.dyn_1(image)
        print(x1.size())
        x2 = self.pixel_unshuffle_1(x1)
        print(x2.size())
        x3 = self.dyn_2(x2)
        x4 = self.pixel_unshuffle_2(x3)
        print(x4.size())
        x5 = self.dyn_3(x4)
        x6 = self.pixel_unshuffle_3(x5)
        print(x6.size())
        x7 = self.dyn_4(x6)
        x8 = self.pixel_unshuffle_4(x7)
        print(x8.size())
        x9 = self.dyn_5(x8)
        print(x9.size())

        # decoder
        x = self.pixel_1(x9)
        # x = F.interpolate(x, scale_factor= 2, mode= 'trilinear')
        print(x.size())
        x = self.up_dyn_5(torch.cat([x, x7], 1))
        print('final', x.size())

        x = self.pixel_2(x)
        # x = F.interpolate(x, scale_factor= 2, mode= 'trilinear')
        x = self.up_dyn_4(torch.cat([x, x5], 1))
        x_3 = (self.Conv_d4(x))
        x_3_out = F.interpolate(x_3, size=(32, 32, 32), mode='trilinear')
        print('x_3', x_3.size())
        print(x.size())
        x = self.pixel_3(x)
        # x = nn.functional.interpolate(x, scale_factor= 2, mode= 'trilinear')
        x = self.up_dyn_3(torch.cat([x, x3], 1))
        x_4 = (self.Conv_d3(x))
        x_4_out = F.interpolate(x_4, size=(32, 32, 32), mode='trilinear')

        print('x_4', x_4.size())
        print(x.size())
        x = self.pixel_4(x)
        # x = nn.functional.interpolate(x, scale_factor= 2, mode= 'trilinear')
        x = self.up_dyn_2(torch.cat([x, x1], 1))

        # print(x.size())
        # x = self.Up_conv2(x)
        print('x2', x.size())
        x = self.up_dyn_1(x)
        print(x.size())
        x = self.Conv(x)
        print(x.size())

        return x, x_3_out, x_4_out


if __name__ == "__main__":
    tensor = torch.rand((1, 1, 32, 32, 32))
    model = UNet(4, 3, 1, 2, 2, 1)
    model(tensor)
