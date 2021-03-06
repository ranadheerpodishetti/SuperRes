import torch
import torch.nn as nn
import numpy as np
import math
import sys
#import torchsummary

#This ensemble model combines the

#Reference: https://github.com/twtygqyy/pytorch-SRDenseNet/blob/master/srdensenet.py
def get_upsample_filter(size) :
    """Make a 3d bilinear kernel suitable for upsampling"""
    # weight initialization
    factor = (size + 1) // 2
    if size % 2 == 1 :
        center = factor - 1
    else :
        center = factor - 0.5
    og = np.ogrid[:size, :size, :size]
    filter = (1 - abs(og[0] - center) / factor) * \
             (1 - abs(og[1] - center) / factor) * \
             (1 - abs(og[2] - center) / factor)
    print(filter.shape)
    return torch.from_numpy(filter).float()


class _Dense_Block(nn.Module) :
    def __init__(self, channel_in) :
        super(_Dense_Block, self).__init__()

        self.relu = nn.ReLU()
        self.conv1 = nn.Conv3d(in_channels=channel_in, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv3d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv3d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv3d(in_channels=48, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv3d(in_channels=64, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv3d(in_channels=80, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv3d(in_channels=96, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv8 = nn.Conv3d(in_channels=112, out_channels=16, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        conv1 = self.relu(self.conv1(x))

        conv2 = self.relu(self.conv2(conv1))
        cout2_dense = self.relu(torch.cat([conv1, conv2], 1))

        conv3 = self.relu(self.conv3(cout2_dense))
        cout3_dense = self.relu(torch.cat([conv1, conv2, conv3], 1))

        conv4 = self.relu(self.conv4(cout3_dense))
        cout4_dense = self.relu(torch.cat([conv1, conv2, conv3, conv4], 1))

        conv5 = self.relu(self.conv5(cout4_dense))
        cout5_dense = self.relu(torch.cat([conv1, conv2, conv3, conv4, conv5], 1))

        conv6 = self.relu(self.conv6(cout5_dense))
        cout6_dense = self.relu(torch.cat([conv1, conv2, conv3, conv4, conv5, conv6], 1))

        conv7 = self.relu(self.conv7(cout6_dense))
        cout7_dense = self.relu(torch.cat([conv1, conv2, conv3, conv4, conv5, conv6, conv7], 1))

        conv8 = self.relu(self.conv8(cout7_dense))
        cout8_dense = self.relu(torch.cat([conv1, conv2, conv3, conv4, conv5, conv6, conv7, conv8], 1))

        return cout8_dense


class SRDenseNet(nn.Module) :
    def __init__(self) :
        super(SRDenseNet, self).__init__()
        filt = [1, 128, 256, 1152]
        self.relu = nn.ReLU()
        self.lowlevel = nn.Conv3d(in_channels=1, out_channels=filt[1], kernel_size=3, stride=1, padding=1)
        self.bottleneck = nn.Conv3d(in_channels=filt[-1], out_channels=filt[-2], kernel_size=1, stride=1, padding=0, bias=False)
        self.reconstruction = nn.Conv3d(in_channels=filt[-2], out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
        self.denseblock1 = self.make_layer(_Dense_Block, 128)
        self.denseblock2 = self.make_layer(_Dense_Block, 256)
        self.denseblock3 = self.make_layer(_Dense_Block, 384)
        self.denseblock4 = self.make_layer(_Dense_Block, 512)
        self.denseblock5 = self.make_layer(_Dense_Block, 640)
        self.denseblock6 = self.make_layer(_Dense_Block, 768)
        self.denseblock7 = self.make_layer(_Dense_Block, 896)
        self.denseblock8 = self.make_layer(_Dense_Block, 1024)
        self.deconv = nn.Sequential(
            nn.ConvTranspose3d(in_channels=256, out_channels=256, kernel_size=2, stride=2, padding=0, bias=False),
            nn.ReLU()
        )

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                #weight initialization
                #m.weight.data.normal_(0, math.sqrt(2. / n))
                # weight = nn.Parameter(
                #     torch.randn( 128, 1 , m.kernel_size[0], m.kernel_size[0], m.kernel_size[0]),
                #     requires_grad=True)
                weight = nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                m.weight.data.copy_(weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.ConvTranspose3d):
                #adding 3rd dimension
                c1, c2, h, w, d= m.weight.data.size()
                weight = get_upsample_filter(h)
                #to retrieve weights
                m.weight.data = weight.view(1, 1, h, w, d).repeat(c1, c2, 1, 1, 1)
                if m.bias is not None :
                    m.bias.data.zero_()

    def make_layer(self, block, channel_in):
        layers = []
        layers.append(block(channel_in))
        return nn.Sequential(*layers)

    def forward(self, x):
        residual = self.relu(self.lowlevel(x))

        out = self.denseblock1(residual)
        concat = torch.cat([residual, out], 1)

        out = self.denseblock2(concat)
        concat = torch.cat([concat, out], 1)

        out = self.denseblock3(concat)
        concat = torch.cat([concat, out], 1)

        out = self.denseblock4(concat)
        concat = torch.cat([concat, out], 1)

        out = self.denseblock5(concat)
        concat = torch.cat([concat, out], 1)

        out = self.denseblock6(concat)
        concat = torch.cat([concat, out], 1)

        out = self.denseblock7(concat)
        concat = torch.cat([concat, out], 1)

        out = self.denseblock8(concat)
        out = torch.cat([concat, out], 1)

        out = self.bottleneck(out)

        out = self.deconv(out)

        out = self.reconstruction(out)

        return out



if __name__ == "__main__":
    tensor = torch.rand((1, 1, 8, 8, 8))
    model = SRDenseNet()

    p = model(tensor)
    x = p.detach().cpu().numpy()
    if np.isnan(np.sum(x)):
        sys.exit()

    print('yo yo')