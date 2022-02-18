import torch
import torch.nn as nn
import torch.nn.functional as F
import pixel_shuffle_icnr
import pixel_shuffle

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


class pixel_shuffle_new(nn.Module):
    """
    Up Convolution Block
    """

    # def __init__(self, in_ch, out_ch):
    def __init__(self, in_c, out_c, kernel, stride, bias=True):
        super(pixel_shuffle_new, self).__init__()
        self.conv = nn.Conv3d(in_c, out_c, kernel, stride, bias)
        self.icnr_weights = pixel_shuffle_icnr.ICNR(self.conv.weight, 2)
        self.conv.weight.data.copy_(self.icnr_weights)

    def forward(self, x):
        x = self.conv(x)
        x = pixel_shuffle.pixel_shuffle_generic(x, 2)
        print(x.size())
        return x


if __name__ == "__main__":
    tensor = torch.rand((1, 4, 32, 32, 32))
    model = pixel_shuffle_new(4, 16, 1, 1)
    k = model(tensor)
    print(k.size())