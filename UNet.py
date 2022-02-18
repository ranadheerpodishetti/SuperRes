import torch
import torch.nn as nn








# -------------------------------------------------------------------------------------------------------------------------------------------------##

def double_conv(in_c, out_c, kernel):
    conv = nn.Sequential(
        nn.Conv3d(in_c, out_c, kernel_size=kernel, padding=kernel // 2),
        nn.ReLU(inplace=True),
        nn.Conv3d(out_c, out_c, kernel_size=kernel, padding=kernel // 2),
        nn.ReLU(inplace=True)
    )
    return conv


class UNet(nn.Module):
    def __init__(self,num_features,kernel_size_1,stride_1,kernel_size_2,stride_2,kernel_size_3):
        super(UNet, self).__init__()

        # self.max_pool_3x3 = nn.MaxPool3d(kernel_size=2, stride=2)
        # self.down_conv_1 = double_conv(1, 4, 3)
        # self.down_conv_2 = double_conv(4, 8, 3)
        # self.down_conv_3 = double_conv(8, 16, 3)
        # self.down_conv_4 = double_conv(16, 32, 3)
        # self.down_conv_5 = double_conv(32, 64, 3)

        self.max_pool_3x3 = nn.MaxPool3d(kernel_size=kernel_size_2, stride=stride_2)
        self.down_conv_1 = double_conv(1, num_features, kernel_size_1)
        self.down_conv_2 = double_conv(num_features, num_features*2, kernel_size_1)
        self.down_conv_3 = double_conv(num_features*2, num_features*4, kernel_size_1)
        self.down_conv_4 = double_conv(num_features*4, num_features*8, kernel_size_1)
        self.down_conv_5 = double_conv(num_features*8, num_features*16, kernel_size_1)

        # self.up_trans_1 = nn.ConvTranspose3d(
        #     in_channels=64,
        #     out_channels=32,
        #     kernel_size=2,
        #     stride=2)

        self.up_trans_1 = nn.ConvTranspose3d(
            in_channels=num_features*16,
            out_channels=num_features*8,
            kernel_size=kernel_size_2,
            stride=stride_2)

        #self.up_conv_1 = double_conv(64, 32, 3)
        self.up_conv_1 = double_conv(num_features*16, num_features*8, kernel_size_1)

        # self.up_trans_2 = nn.ConvTranspose3d(
        #     in_channels=32,
        #     out_channels=16,
        #     kernel_size=2,
        #     stride=2)

        self.up_trans_2 = nn.ConvTranspose3d(
            in_channels=num_features*8,
            out_channels=num_features*4,
            kernel_size=kernel_size_2,
            stride=stride_2)

        #self.up_conv_2 = double_conv(32, 16, 3)
        self.up_conv_2 = double_conv(num_features*8, num_features*4, kernel_size_1)

        # self.up_trans_3 = nn.ConvTranspose3d(
        #     in_channels=16,
        #     out_channels=8,
        #     kernel_size=2,
        #     stride=2)

        self.up_trans_3 = nn.ConvTranspose3d(
            in_channels=num_features*4,
            out_channels=num_features*2,
            kernel_size=kernel_size_2,
            stride=stride_2)

        #self.up_conv_3 = double_conv(16, 8, 3)


        self.up_conv_3 = double_conv(num_features*4, num_features*2, kernel_size_1)

        # self.up_trans_4 = nn.ConvTranspose3d(
        #     in_channels=8,
        #     out_channels=4,
        #     kernel_size=2,
        #     stride=2)

        self.up_trans_4 = nn.ConvTranspose3d(
            in_channels=num_features*2,
            out_channels=num_features,
            kernel_size=kernel_size_2,
            stride=stride_2)

        #self.up_conv_4 = double_conv(8, 4, 3)

        self.up_conv_4 = double_conv(num_features*2, num_features, kernel_size_1)

        #self.out = nn.Conv3d(in_channels=4, out_channels=1, kernel_size=1)

        self.out = nn.Conv3d(in_channels=num_features, out_channels=1, kernel_size=kernel_size_3)

    def forward(self, image):
        # encoder
        #print(image.size())
        x1 = self.down_conv_1(image)
        print(x1.size())
        x2 = self.max_pool_3x3(x1)
        print(x2.size())
        x3 = self.down_conv_2(x2)
        x4 = self.max_pool_3x3(x3)
        print(x4.size())
        x5 = self.down_conv_3(x4)
        x6 = self.max_pool_3x3(x5)
        print(x6.size())
        x7 = self.down_conv_4(x6)
        x8 = self.max_pool_3x3(x7)
        print(x8.size())
        x9 = self.down_conv_5(x8)
        print(x9.size())

        # decoder
        x = self.up_trans_1(x9)
        print(x.size())
        x = self.up_conv_1(torch.cat([x, x7], 1))
        print(x.size())

        x = self.up_trans_2(x)
        print(x.size())
        x = self.up_conv_2(torch.cat([x, x5], 1))
        print(x.size())

        x = self.up_trans_3(x)
        print(x.size())
        x = self.up_conv_3(torch.cat([x, x3], 1))
        print(x.size())

        x = self.up_trans_4(x)
        print(x.size())
        x = torch.cat([x, x1], 1)

        print(x.size())
        x = self.up_conv_4(x)
        print(x.size())

        x = self.out(x)
        print(x.size())

        return x


if __name__ == '__main__':
    tensor = torch.rand(1, 1, 32, 32, 32)
    model = UNet(4, 1, 3, 1, 2, 2)
    model(tensor)

