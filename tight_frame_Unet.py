import torch
import torch.nn as nn
import torch.nn.functional as F
import pixel_unshuffle
import pixel_unshuffle_new
import pixel_shuffle_icnr
import pixel_shuffle
# from torchsummary import summary
import pixel_shuffle_new
import dyn_conv
import unet_dynamic_convolution


# -------------------------------------------------------------------------------------------------------------------------------------------------##

class Attention_block(nn.Module):
    """
    Attention Block
    """

    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()

        self.W_g = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        out = x * psi
        return out


class double_conv(nn.Module):
    """
    Convolution Block
    """

    def __init__(self, in_channels, out_channels, k_size, stride, bias=True):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=k_size,
                      stride=stride, padding=k_size // 2, bias=bias),
            nn.BatchNorm3d(num_features=out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=k_size,
                      stride=stride, padding=k_size // 2, bias=bias),
            nn.BatchNorm3d(num_features=out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class wave_conv_block(nn.Module):
    """
    Convolution Block
    """

    def __init__(self, in_channels, out_channels, k_size, stride, bias=True):
        super(wave_conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=k_size,
                      stride=stride, padding=k_size // 2, bias=bias),
            nn.BatchNorm3d(num_features=out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class wave_decomposition(nn.Module):
    """
    Convolution Block
    """

    def __init__(self, in_channels, out_channels, k_size, stride, bias=True):
        super(wave_decomposition, self).__init__()
        self.conv1 = wave_conv_block(in_channels=in_channels, out_channels=out_channels, k_size=k_size, stride=stride)
        self.conv2 = wave_conv_block(in_channels=in_channels, out_channels=out_channels, k_size=k_size, stride=stride)
        self.conv3 = wave_conv_block(in_channels=in_channels, out_channels=out_channels, k_size=k_size, stride=stride)
        self.conv4 = wave_conv_block(in_channels=in_channels, out_channels=out_channels, k_size=k_size, stride=stride)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        return x1, x2, x3, x4


class concat(nn.Module):
    """
    Convolution Block
    """

    def __init__(self):
        super(concat, self).__init__()

    def forward(self, e1, e2, e3, e4, d1, d2, d3, d4):
        self.X1 = e1 + d1
        self.X2 = e2 + d2
        self.X3 = e3 + d3
        self.X4 = e4 + d4
        x = torch.cat([self.X1, self.X2, self.X3, self.X4], dim=1)

        return x


class up_conv(nn.Module):
    """
    Up Convolution Block
    """

    # def __init__(self, num_features, out_ch):
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


class Recurrent_block(nn.Module):
    """
    Recurrent Block for R2Unet_CNN
    """

    def __init__(self, out_ch, t=2):
        super(Recurrent_block, self).__init__()

        self.t = t
        self.out_ch = out_ch
        self.conv = nn.Sequential(
            nn.Conv3d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        for i in range(self.t):
            if i == 0:
                x = self.conv(x)
            out = self.conv(x + x)
        return out


class RRCNN_block(nn.Module):
    """
    Recurrent Residual Convolutional Neural Network Block
    """

    def __init__(self, num_features, out_ch, t=2):
        super(RRCNN_block, self).__init__()

        self.RCNN = nn.Sequential(
            Recurrent_block(out_ch, t=t),
            Recurrent_block(out_ch, t=t)
        )
        self.Conv = unet_dynamic_convolution.dynamic(num_features, out_ch, 1, 3, 5, 1)

    def forward(self, x):
        x1 = self.Conv(x)
        x2 = self.RCNN(x1)
        out = x1 + x2
        return out


class wavenet(nn.Module):
    """
    Residual Recuurent Block with attention Unet
    Implementation : https://github.com/LeeJunHyun/Image_Segmentation
    """

    def __init__(self, num_features, out_ch, kernel_size_1, stride_1, kernel_size_2, stride_2):
        super(wavenet, self).__init__()

        num_features = num_features
        print(num_features)
        filters = [num_features, num_features * 2, num_features * 4, num_features * 8, num_features * 16]

        self.Maxpool_1 = nn.MaxPool3d(kernel_size=kernel_size_2, stride=stride_2)
        self.Maxpool_2 = nn.MaxPool3d(kernel_size=kernel_size_2, stride=stride_2)
        self.Maxpool_3 = nn.MaxPool3d(kernel_size=kernel_size_2, stride=stride_2)
        self.Maxpool_4 = nn.MaxPool3d(kernel_size=kernel_size_2, stride=stride_2)

        self.Convtrans_1 = nn.ConvTranspose3d(num_features * 16, num_features * 8, kernel_size_2, stride_2)
        self.Convtrans_2 = nn.ConvTranspose3d(num_features * 8, num_features * 4, kernel_size_2, stride_2)
        self.Convtrans_3 = nn.ConvTranspose3d(num_features * 4, num_features * 2, kernel_size_2, stride_2)
        self.Convtrans_4 = nn.ConvTranspose3d(num_features * 2, num_features, kernel_size_2, stride_2)

        self.conv_inp = double_conv(1, filters[0], kernel_size_1, stride_1)
        self.wave_1_down = wave_decomposition(filters[0], filters[0], kernel_size_1, stride_1)
        self.conv_enc_1 = double_conv(filters[0], filters[1], kernel_size_1, stride_1)
        self.wave_2_down = wave_decomposition(filters[1], filters[1], kernel_size_1, stride_1)
        self.conv_enc_2 = double_conv(filters[1], filters[2], kernel_size_1, stride_1)
        self.wave_3_down = wave_decomposition(filters[2], filters[2], kernel_size_1, stride_1)
        self.conv_enc_3 = double_conv(filters[2], filters[3], kernel_size_1, stride_1)
        self.wave_4_down = wave_decomposition(filters[3], filters[3], kernel_size_1, stride_1)
        self.conv_enc_4 = double_conv(filters[3], filters[4], kernel_size_1, stride_1)

        self.conv_dec_4 = double_conv(filters[4], filters[3], kernel_size_1, stride_1)
        self.wave_4_up = wave_decomposition(filters[3], filters[3], kernel_size_1, stride_1)
        self.conv_dec_3 = double_conv(filters[3], filters[2], kernel_size_1, stride_1)
        self.wave_3_up = wave_decomposition(filters[2], filters[2], kernel_size_1, stride_1)
        self.conv_dec_2 = double_conv(filters[2], filters[1], kernel_size_1, stride_1)
        self.wave_2_up = wave_decomposition(filters[1], filters[1], kernel_size_1, stride_1)
        self.conv_dec_1 = double_conv(filters[1], 1, kernel_size_1, stride_1)
        self.wave_1_up = wave_decomposition(filters[0], filters[0], kernel_size_1, stride_1)
        self.cat = concat()
        self.convup_4 = double_conv(filters[3] * 5, filters[3], kernel_size_1, stride_1)
        self.convup_3 = double_conv(filters[2] * 5, filters[2], kernel_size_1, stride_1)
        self.convup_2 = double_conv(filters[1] * 5, filters[1], kernel_size_1, stride_1)
        self.convup_1 = double_conv(filters[0] * 5, filters[0], kernel_size_1, stride_1)
        self.out = nn.Conv3d(filters[0], out_ch, kernel_size=1, stride=stride_1, padding=0, bias=True)

    def forward(self, x):
        e1 = self.conv_inp(x)
        wave_e1_1, wave_e1_2, wave_e1_3, wave_e1_4 = self.wave_1_down(e1)
        e2 = self.Maxpool_1(wave_e1_4)

        e3 = self.conv_enc_1(e2)
        wave_e2_1, wave_e2_2, wave_e2_3, wave_e2_4 = self.wave_2_down(e3)
        e4 = self.Maxpool_2(wave_e2_4)

        e5 = self.conv_enc_2(e4)
        wave_e3_1, wave_e3_2, wave_e3_3, wave_e3_4 = self.wave_3_down(e5)
        e6 = self.Maxpool_3(wave_e3_4)

        e7 = self.conv_enc_3(e6)
        wave_e4_1, wave_e4_2, wave_e4_3, wave_e4_4 = self.wave_4_down(e7)
        e8 = self.Maxpool_4(wave_e4_4)

        e9 = self.conv_enc_4(e8)

        d4 = self.Convtrans_1(e9)
        print(d4.shape)
        wave_d4_1, wave_d4_2, wave_d4_3, wave_d4_4 = self.wave_4_up(d4)
        print(wave_d4_1.shape)
        cat_4 = self.cat(wave_e4_1, wave_e4_2, wave_e4_3, wave_e4_4, wave_d4_1, wave_d4_2, wave_d4_3, wave_d4_4)
        d4 = self.convup_4(torch.cat([e7, cat_4], dim=1))

        d3 = self.Convtrans_2(d4)
        wave_d3_1, wave_d3_2, wave_d3_3, wave_d3_4 = self.wave_3_up(d3)
        cat_3 = self.cat(wave_e3_1, wave_e3_2, wave_e3_3, wave_e3_4, wave_d3_1, wave_d3_2, wave_d3_3, wave_d3_4)
        d3 = self.convup_3(torch.cat([e5, cat_3], dim=1))

        d2 = self.Convtrans_3(d3)
        wave_d2_1, wave_d2_2, wave_d2_3, wave_d2_4 = self.wave_2_up(d2)
        cat_2 = self.cat(wave_e2_1, wave_e2_2, wave_e2_3, wave_e2_4, wave_d2_1, wave_d2_2, wave_d2_3, wave_d2_4)
        d2 = self.convup_2(torch.cat([e3, cat_2], dim=1))

        d1 = self.Convtrans_4(d2)
        wave_d1_1, wave_d1_2, wave_d1_3, wave_d1_4 = self.wave_1_up(d1)
        cat_1 = self.cat(wave_e1_1, wave_e1_2, wave_e1_3, wave_e1_4, wave_d1_1, wave_d1_2, wave_d1_3, wave_d1_4)
        d1 = self.convup_1(torch.cat([e1, cat_1], dim=1))
        print(d1.shape)
        out = self.out(d1)
        print(out.shape)
        return out, wave_d1_1, wave_d1_2


# self.wave_e1_1, self.wave_e1_2, self.wave_e1_3, self.wave_e1_4
if __name__ == '__main__':
    tensor = torch.rand(1, 1, 32, 32, 32)
    model = wavenet(4, 1, 3, 1, 2,2)
    model(tensor)
    # summary(model, (1, 48,48,48))
