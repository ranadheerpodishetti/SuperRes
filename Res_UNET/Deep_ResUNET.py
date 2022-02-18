import torch
import torch.nn as nn

import torch.nn.functional as F

class ResidualConv(nn.Module):
    def __init__(self, input_dim, output_dim, stride, padding):
        super(ResidualConv, self).__init__()

        self.conv_block = nn.Sequential(
            nn.BatchNorm3d(input_dim),
            nn.ReLU(),
            nn.Conv3d(
                input_dim, output_dim, kernel_size=3, stride=stride, padding=padding
            ),
            nn.BatchNorm3d(output_dim),
            nn.ReLU(),
            nn.Conv3d(output_dim, output_dim, kernel_size=3, padding=1),
        )
        self.conv_skip = nn.Sequential(
            nn.Conv3d(input_dim, output_dim, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm3d(output_dim),
        )

    def forward(self, x):

        return self.conv_block(x) + self.conv_skip(x)


class Upsample(nn.Module):
    def __init__(self, scale=2):
        super(Upsample, self).__init__()

        self.upsample = nn.Upsample(mode="trilinear", scale_factor=scale)

    def forward(self, x):
        return self.upsample(x)

class ResUnet(nn.Module):
    def __init__(self, channel, filters=[64, 128, 256, 512]):
        super(ResUnet, self).__init__()

        self.input_layer = nn.Sequential(
            nn.Conv3d(channel, filters[0], kernel_size=3, padding=1),
            nn.BatchNorm3d(filters[0]),
            nn.ReLU(),
            nn.Conv3d(filters[0], filters[0], kernel_size=3, padding=1),
        )
        self.input_skip = nn.Sequential(
            nn.Conv3d(channel, filters[0], kernel_size=3, padding=1)
        )

        self.residual_conv_1 = ResidualConv(filters[0], filters[1], 2, 1)
        self.residual_conv_2 = ResidualConv(filters[1], filters[2], 2, 1)

        self.bridge = ResidualConv(filters[2], filters[3], 2, 1)

        self.upsample_1 = Upsample(scale =2)
        self.up_residual_conv1 = ResidualConv(filters[3] + filters[2], filters[2], 1, 1)

        self.upsample_2 = Upsample( scale=2)
        self.up_residual_conv2 = ResidualConv(filters[2] + filters[1], filters[1], 1, 1)

        self.upsample_3 = Upsample(scale=2)
        self.up_residual_conv3 = ResidualConv(filters[1] + filters[0], filters[0], 1, 1)

        self.output_layer = nn.Sequential(
            nn.Conv3d(filters[0], 1, 1, 1),

        )
        self.conv_d3 = nn.Conv3d(filters[0]*3, channel, kernel_size = 1, stride =1, padding = 0 )
        self.conv_d2 = nn.Conv3d(filters[0] , channel, kernel_size=1, stride=1, padding=0)
    def forward(self, x):
        # Encode
        x1 = self.input_layer(x) + self.input_skip(x)
        x2 = self.residual_conv_1(x1)
        x3 = self.residual_conv_2(x2)
        # Bridge
        x4 = self.bridge(x3)
        # Decode
        x4 = self.upsample_1(x4)
        x5 = torch.cat([x4, x3], dim=1)

        x6 = self.up_residual_conv1(x5)

        x6 = self.upsample_2(x6)
        x7 = torch.cat([x6, x2], dim=1)

        x8 = self.up_residual_conv2(x7)

        x8 = self.upsample_3(x8)
        x9 = torch.cat([x8, x1], dim=1)
        d3_out = F.interpolate(self.conv_d3(x9), mode='trilinear', size=x.size()[2:])
        print(d3_out.size())
        x10 = self.up_residual_conv3(x9)
        d2_out = F.interpolate(self.conv_d2(x10), mode='trilinear', size=x.size()[2:])
        print(d2_out.size())
        output = self.output_layer(x10)
        print(output.size())
        return output, d3_out, d2_out


if __name__ == "__main__":
    tensor = torch.rand((1,1,32,32,32))
    model = ResUnet(1)
    result = model(tensor)