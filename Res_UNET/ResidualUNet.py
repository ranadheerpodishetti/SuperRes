import torch
import torch.nn as nn
import torch.nn.functional as F


class conv_block(nn.Module):
    """
    Convolution Block
    """

    def __init__(self, in_channels, out_channels,stride):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.BatchNorm3d(num_features=out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=3,
                      padding=1,stride=stride)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class stem(nn.Module):
    def __init__(self, in_channels,out_channels):
        super(stem,self).__init__()

        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1, padding=1, stride=1),
            conv_block(out_channels,out_channels,stride=1),
        )
        self.stem_conv = nn.Conv3d( out_channels,out_channels,kernel_size=1, padding=1, stride=1)
        self.stem_Batch_norm = nn.BatchNorm3d(num_features=out_channels)
    def forward(self,x):
        x1 = self.stem(x)
        y = self.stem_conv(x1)
        y = self.stem_Batch_norm(y)
        output = torch.add([x1,y])
        return output

class residual_block(nn.Module):
    def __init__(self,in_channels,out_channels,stride):
        super(residual_block,self).__init__()
        self.res = nn.Sequential(
            conv_block(in_channels,out_channels,stride),
            conv_block(out_channels, out_channels,stride)
        )
        self.shortcut = nn.Conv3d(out_channels,out_channels,kernel_size=3, padding=1, stride=stride)
        self.Residual_Batch_norm = nn.BatchNorm3d(num_features=out_channels)





    def forward(self,x):
        x1= self.res(x)
        y = self.shortcut(x1)
        y = self.Residual_Batch_norm(y)

        output = torch.add([x1, y])
        return output

class upsample_concat_block(nn.Module):
    def __init__(self):
        super(upsample_concat_block,self).__init__()
        self.upsample = nn.Upsample((2))

    def forward(self ,x,xskip):
        u = self.upsample(x)
        c = torch.cat([u, xskip],dim=1)
        return c

class ResUNet(nn.Module):
    def __init__(self, in_channels,out_channels):
        super(ResUNet,self).__init__()
        #encoder
        n = 16
        f = [n, n*2, n*4, n*8, n*16]
        self.stem_0 = stem(in_channels,out_channels)
        self.residual_block_1 = residual_block(f[0], f[1], stride=2)
        self.residual_block_2 = residual_block(f[1], f[2], stride=2)
        self.residual_block_3 = residual_block(f[2], f[3], stride=2)
        self.residual_block_4 = residual_block(f[3], f[4], stride=2)
        #Bridge
        self.Bridge_0 = conv_block(f[4], f[4], stride=1)
        self.Bridge_1 = conv_block(f[4], f[4], stride=1)

        #Decoder
        self.upsample = upsample_concat_block()
        self.Decoder_1 = residual_block(f[4], f[4],stride=1)
        self.Decoder_2 = residual_block(f[4], f[3],stride=1)
        self.Decoder_3 = residual_block(f[3], f[2],stride=1)
        self.Decoder_4 = residual_block(f[2], f[1],stride=1)

    def forward(self,image):
        e1 = self.stem_0(image)
        print(e1.size())
        e2 = self.residual_block_1(e1)
        print(e2.size())
        e3 = self.residual_block_2(e2)
        print(e3.size())

        e4 = self.residual_block_3(e3)
        print(e4.size())

        e5 = self.residual_block_4(e4)
        print(e5.size())


        b0 = self.Bridge_0(e5)
        print(b0.size())

        b1 = self.Bridge_1(b0)
        print(b1.size())


        u1 = self.upsample(b1,e4)
        print(u1.size())

        d1 = self.Decoder_1(u1)
        u2 = self.upsample(d1, e3)
        print(u2.size())

        d2 = self.Decoder_2(u2)
        u3 = self.upsample(d2,u2)
        print(u3.size())

        d3 = self.Decoder_3(u3)
        u4 = self.upsample(d3,e1)
        print(u4.size())

        d4 = self.Decoder_4(u4)

        output = nn.Conv3D(1, 1, padding=1)(d4)
        print(output.size())

        return output







if __name__ == "__main__":
    tensor = torch.rand((1,4,4,4,4))
    model = ResUNet(1 , 1)
    model(tensor)

    #model(tensor)