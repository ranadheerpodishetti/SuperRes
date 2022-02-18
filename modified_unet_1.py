import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------------------------------------------------------------------------------------------------------------------------------##

class double_conv(nn.Module):
    """
    Convolution Block
    """

    def __init__(self, in_channels, out_channels, k_size=3, stride=1, padding=1, bias=True):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=k_size,
                      stride=stride, padding=padding, bias=bias),
            nn.BatchNorm3d(num_features=out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=k_size,
                      stride=stride, padding=padding, bias=bias),
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
    def __init__(self, in_c, out_c, kernel, stride , bias = True):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv3d(in_channels=in_c, out_channels=out_c, kernel_size=kernel,
                      stride=stride, padding=kernel//2, bias=bias),
            nn.BatchNorm3d(num_features=out_c),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.up(x)
        return x

#num_features 4 --kernel_size_1 3 --stride_1 1 --kernel_size_2 2 --stride_2 2 --kernel_size_3 1
class UNet(nn.Module):
    def __init__(self,num_features,kernel_size_1,stride_1,kernel_size_2,stride_2,kernel_size_3):
        super(UNet, self).__init__()

        #self.max_pool_3x3 = nn.MaxPool3d(kernel_size=2, stride=2)
        #self.down_conv_1 = double_conv(1, 4, 3)
        #self.down_conv_2 = double_conv(4, 8, 3)
        #self.down_conv_3 = double_conv(8, 16, 3)
        #self.down_conv_4 = double_conv(16, 32, 3)
        #self.down_conv_5 = double_conv(32, 64, 3)
        self.max_pool_3x3 = nn.MaxPool3d(kernel_size=kernel_size_2, stride=stride_2)
        self.down_conv_1 = double_conv(1, num_features, kernel_size_1, stride_1)
        self.down_conv_2 = double_conv(num_features, num_features*2, kernel_size_1, stride_1)
        self.down_conv_3 = double_conv(num_features*2, num_features*4, kernel_size_1, stride_1)
        self.down_conv_4 = double_conv(num_features*4, num_features*8, kernel_size_1, stride_1)
        self.down_conv_5 = double_conv(num_features*8, num_features*16, kernel_size_1, stride_1)
        self.conv2 = double_conv(num_features*4, 1, kernel_size_1, stride_1)
        self.conv3 = double_conv(num_features * 2, 1, kernel_size_1, stride_1)
        #self.up_trans_1 = nn.Conv3d(in_channels= num_features*16, out_channels= num_features*8, kernel_size= kernel_size_3)
        self.up_trans_1 = up_conv(num_features * 16, num_features * 8,
                                    kernel_size_3, stride_1) # use concat after this
        self.up_conv_1 = double_conv(num_features*16, num_features*8, kernel_size_1, stride_1 )

        self.up_trans_2 = up_conv(num_features*8, num_features*4, kernel_size_3, stride_1)

        self.up_conv_2 = double_conv(num_features*8, num_features*4, kernel_size_1, stride_1)

        self.up_trans_3 = up_conv(num_features*4, num_features*2,kernel_size_3, stride_1)

        self.up_conv_3 = double_conv(num_features*4, num_features*2, kernel_size_1, stride_1)

        self.up_trans_4 = up_conv(num_features*2, num_features, kernel_size_3, stride_1)

        self.up_conv_4 = double_conv(num_features*2, num_features, kernel_size_1, stride_1)

        self.out = nn.Conv3d(in_channels=num_features, out_channels=1, kernel_size=1, stride= stride_1)

    def forward(self, image):
        # encoder
        print(image.size())
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
        #x = F.interpolate(x, scale_factor= 2, mode= 'trilinear')
        print(x.size())
        x = self.up_conv_1(torch.cat([x, x7], 1))
        print('final',x.size())

        x = self.up_trans_2(x)
        #x = F.interpolate(x, scale_factor= 2, mode= 'trilinear')
        x = self.up_conv_2(torch.cat([x, x5], 1))
        x_3 = F.interpolate(self.conv2(x), (image.size()[2:] ), mode='trilinear')
        print('x_3', x_3.size())
        print(x.size())
        x = self.up_trans_3(x)
        #x = nn.functional.interpolate(x, scale_factor= 2, mode= 'trilinear')
        x = self.up_conv_3(torch.cat([x, x3], 1))
        x_4 = F.interpolate(self.conv3(x), (image.size()[2:]), mode='trilinear')
        print('x_4', x_4.size())
        print(x.size())
        x = self.up_trans_4(x)
        #x = nn.functional.interpolate(x, scale_factor= 2, mode= 'trilinear')
        x = torch.cat([x, x1], 1)

        #print(x.size())
        x = self.up_conv_4(x)
        print(x.size())

        x = self.out(x)
        print(x.size())

        return x, x_3, x_4



if __name__ == "__main__":
    tensor = torch.rand((32,1,32, 32,32))
    model = UNet(64, 3,1, 2, 2, 1)
    model(tensor)