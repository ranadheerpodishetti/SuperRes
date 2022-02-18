

import numpy as np
import torch
import torch.nn as nn


class SRCNN3D(nn.Module):
    def __init__(self,n_channels=1,scale_factor=2,num_features=32,kernel_size=3,stride=1):
        super(SRCNN3D, self).__init__()
        if type(scale_factor) is int:
            self.scale_factor=(scale_factor,scale_factor,scale_factor)
        else:
            self.scale_factor=scale_factor
        # n_dim_upscale = 0
        # for f in self.scale_factor:
        #     if f > 1:
        #         n_dim_upscale += 1 #This will only work for scale factor of 2 in any num of dims TODO
        # activation_maps = 2 ** n_dim_upscale
        self.n_channels=n_channels
        activation_maps = np.prod(self.scale_factor)

        self.conv_1 = nn.Sequential(nn.Conv3d(n_channels, num_features, kernel_size, stride, padding=kernel_size // 2),
                                    nn.BatchNorm3d(num_features=num_features), nn.ReLU(inplace=True))
        self.conv_2 = nn.Sequential(nn.Conv3d(num_features, num_features, kernel_size, stride, padding=kernel_size // 2), 
                                    nn.BatchNorm3d(num_features=num_features), nn.ReLU(inplace=True))
        self.conv_3 = nn.Sequential(nn.Conv3d(num_features, num_features, kernel_size, stride, padding=kernel_size // 2), 
                                    nn.BatchNorm3d(num_features=num_features), nn.ReLU(inplace=True))
        self.conv_4 = nn.Sequential(nn.Conv3d(num_features, num_features, kernel_size, stride, padding=kernel_size // 2), 
                                    nn.BatchNorm3d(num_features=num_features), nn.ReLU(inplace=True))
        self.conv_5 = nn.Sequential(nn.Conv3d(num_features, num_features, kernel_size, stride, padding=kernel_size // 2), 
                                    nn.BatchNorm3d(num_features=num_features), nn.ReLU(inplace=True))
        self.conv_6 = nn.Sequential(nn.Conv3d(num_features, activation_maps, kernel_size, stride, padding=kernel_size // 2), 
                                    nn.BatchNorm3d(num_features=activation_maps), nn.ReLU(inplace=True))
        self.conv_7 = nn.Sequential(nn.Conv3d(n_channels, num_features, kernel_size, padding=kernel_size // 2),
                                    nn.BatchNorm3d(num_features=num_features), nn.ReLU(inplace=True))
        self.conv_8 = nn.Sequential(nn.Conv3d(num_features, num_features, kernel_size, padding=kernel_size // 2),
                                    nn.BatchNorm3d(num_features=num_features), nn.ReLU(inplace=True))
        self.conv_9 = nn.Sequential(nn.Conv3d(num_features, num_features, kernel_size, padding=kernel_size // 2),
                                    nn.BatchNorm3d(num_features=num_features), nn.ReLU(inplace=True))
        self.conv_10 = nn.Sequential(nn.Conv3d(num_features, n_channels, kernel_size, padding=kernel_size // 2),
                                     nn.Sigmoid())
        for m in self.modules():
            if isinstance(m, nn.Conv3d) :
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

    def forward(self, image):
        output_1 = self.conv_1(image)
        output_2 = self.conv_2(output_1)
        output_3a = self.conv_3(output_2)
        output_3 = torch.add(output_1, output_3a) #torch.mul(output_1, 1) = output_1 #Note for Geetha
        output_4 = self.conv_4(output_3)
        output_5a = self.conv_5(output_4)
        output_5 = torch.add(output_1, output_5a)
        output_6 = self.conv_6(output_5)
        suffled_size = tuple(np.multiply(output_6.shape[2:], self.scale_factor))
        output_7 = output_6.view(output_6.shape[0], self.n_channels, *suffled_size)
        output_8 = self.conv_7(output_7)
        output_9 = self.conv_8(output_8)
        output_10a = self.conv_9(output_9)
        output_10 = torch.add(output_8, output_10a)
        output = self.conv_10(output_10)  # Final Loss
        return output_7, output


if __name__ == "__main__":
    tensor = torch.rand((32, 1, 8, 8, 8))
    model = SRCNN3D()
    model(tensor)

    # model = SRCNN3D(1,num_features=64,scale_factor=(2,1,1)).cuda()
    # from torchsummary import summary
    # summary(model, input_size=(1, 32, 32, 32))
