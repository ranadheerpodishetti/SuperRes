import torch
import torch.nn as nn
import torch.nn.functional as F
import pixel_unshuffle
import pixel_unshuffle_new
#from torchsummary import summary

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
        self.conv1 = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=k_size,
                      stride=stride, padding=k_size // 2, bias=bias)
        self.conv2 = nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=k_size,
                      stride=stride, padding=k_size // 2, bias=bias)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        Norm = nn.LayerNorm(x.size()[1:])
        x = self.relu(Norm(x))
        x = self.conv2(x)
        Norm = nn.LayerNorm(x.size()[1:])
        x = self.relu(Norm(x))

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
            #nn.BatchNorm3d(num_features=out_c),
           )
        self.relu =  nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.up(x)
        Norm = nn.LayerNorm(x.size()[1:])
        x = self.relu(Norm(x))
        return x

class Recurrent_block(nn.Module):
    """
    Recurrent Block for R2Unet_CNN
    """

    def __init__(self, out_ch, t=2):
        super(Recurrent_block, self).__init__()

        self.t = t
        self.out_ch = out_ch
        self.conv = nn.Conv3d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True)
            #nn.BatchNorm3d(out_ch),
        self.relu =    nn.ReLU(inplace=True)


    def forward(self, x):
        for i in range(self.t):
            if i == 0:
                x = self.conv(x)
                Norm = nn.LayerNorm(x.size()[1:])
                x = self.relu(Norm(x))
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
        self.Conv = nn.Conv3d(num_features, out_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x1 = self.Conv(x)
        x2 = self.RCNN(x1)
        out = x1 + x2
        return out


class R2AttU_Net(nn.Module):
    """
    Residual Recuurent Block with attention Unet
    Implementation : https://github.com/LeeJunHyun/Image_Segmentation
    """

    def __init__(self, num_features,out_ch, kernel_size_1, stride_1, kernel_size_2, stride_2, kernel_size_3, t=2):
        super(R2AttU_Net, self).__init__()

        num_features = num_features
        filters = [num_features, num_features * 2, num_features * 4, num_features * 8, num_features * 16]

        self.Maxpool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.pixel_unshuffle_1 = pixel_unshuffle_new.pixel_unshuffle_new(num_features, num_features, 3, 1)
        self.pixel_unshuffle_2 = pixel_unshuffle_new.pixel_unshuffle_new(num_features * 2, num_features * 2, 3, 1)
        self.pixel_unshuffle_3 = pixel_unshuffle_new.pixel_unshuffle_new(num_features * 4, num_features * 4, 3, 1)
        self.pixel_unshuffle_4 = pixel_unshuffle_new.pixel_unshuffle_new(num_features * 8, num_features * 8, 3, 1)

        self.RRCNN1 = RRCNN_block(1, num_features, t=t)
        self.RRCNN2 = RRCNN_block(num_features, num_features*2, t=t)
        self.RRCNN3 = RRCNN_block(num_features*2, num_features*4, t=t)
        self.RRCNN4 = RRCNN_block(num_features*4, num_features*8, t=t)
        self.RRCNN5 = RRCNN_block(num_features*8, num_features*16, t=t)

        self.Up5 = up_conv(filters[4], filters[3], kernel_size_1, stride_1)
        self.Att5 = Attention_block(F_g=filters[3], F_l=filters[3], F_int=filters[2])
        self.Up_RRCNN5 = RRCNN_block(filters[4], filters[3], t=t)

        self.Up4 = up_conv(filters[3], filters[2], kernel_size_1, stride_1)
        self.Att4 = Attention_block(F_g=filters[2], F_l=filters[2], F_int=filters[1])
        self.Up_RRCNN4 = RRCNN_block(filters[3], filters[2], t=t)

        self.Up3 = up_conv(filters[2], filters[1], kernel_size_1, stride_1)
        self.Att3 = Attention_block(F_g=filters[1], F_l=filters[1], F_int=filters[0])
        self.Up_RRCNN3 = RRCNN_block(filters[2], filters[1], t=t)

        self.Up2 = up_conv(filters[1], filters[0], kernel_size_1, stride_1)
        self.Att2 = Attention_block(F_g=filters[0], F_l=filters[0], F_int=32)
        self.Up_RRCNN2 = RRCNN_block(filters[1], filters[0], t=t)

        self.Conv = nn.Conv3d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)
        self.conv_d3 = nn.Conv3d(filters[0] * 2, out_ch, kernel_size=1, stride=1, padding=0)
        self.conv_d2 = nn.Conv3d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                # weight initialization
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                # weight = nn.Parameter(
                #     torch.randn( 128, 1 , m.kernel_size[0], m.kernel_size[0], m.kernel_size[0]),
                #     requires_grad=True)
                weight = nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                m.weight.data.copy_(weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    # self.active = torch.nn.Sigmoid()

    def forward(self, x):

        e1 = self.RRCNN1(x)

        e2 = self.pixel_unshuffle_1(e1)
        e2 = self.RRCNN2(e2)

        e3 = self.pixel_unshuffle_2(e2)
        e3 = self.RRCNN3(e3)

        e4 = self.pixel_unshuffle_3(e3)
        e4 = self.RRCNN4(e4)

        e5 = self.pixel_unshuffle_4(e4)
        e5 = self.RRCNN5(e5)

        d5 = self.Up5(e5)
        e4 = self.Att5(g=d5, x=e4)
        d5 = torch.cat((e4, d5), dim=1)
        d5 = self.Up_RRCNN5(d5)

        d4 = self.Up4(d5)
        e3 = self.Att4(g=d4, x=e3)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_RRCNN4(d4)

        d3 = self.Up3(d4)
        e2 = self.Att3(g=d3, x=e2)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_RRCNN3(d3)
        d3_out = F.interpolate(self.conv_d3(d3), mode='trilinear', size=x.size()[2:])
        print(d3_out.size())
        d2 = self.Up2(d3)
        e1 = self.Att2(g=d2, x=e1)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_RRCNN2(d2)
        d2_out = F.interpolate(self.conv_d2(d2), mode='trilinear', size=x.size()[2:])
        print(d2_out.size())
        out = self.Conv(d2)
        print(out.size())
        #  out = self.active(out)

        return out, d3_out, d2_out


if __name__ == '__main__':
    tensor = torch.rand(1, 1, 32, 32, 32)
    model = R2AttU_Net(4,1, 3, 1, 2, 2, 1).cuda()
    #model(tensor)
    #summary(model, (1,32,32,32))


