#Author: Geetha Doddapaneni Gopinath

import torch
import torch.nn as nn
import params
import pixel_shuffle


class SRCNN_late_upscaling(nn.Module):
    def __init__(self):
        super(SRCNN_late_upscaling, self).__init__()
        self.conv_1 = nn.Sequential(nn.Conv3d(1, params.num_features, params.kernel, params.stride,padding=params.kernel//2),nn.ReLU(inplace=True))
        self.conv_2 = nn.Sequential(nn.Conv3d(params.num_features, params.num_features, params.kernel, params.stride, padding=params.kernel//2),nn.ReLU(inplace=True))
        self.conv_3 = nn.Sequential(nn.Conv3d(params.num_features, params.num_features, params.kernel, params.stride, padding=params.kernel//2),nn.ReLU(inplace=True))
        self.conv_4 = nn.Sequential(nn.Conv3d(params.num_features, params.num_features, params.kernel, params.stride, padding=params.kernel//2),nn.ReLU(inplace=True))
        self.conv_5 = nn.Sequential(nn.Conv3d(params.num_features, params.num_features, params.kernel, params.stride, padding=params.kernel//2),nn.ReLU(inplace=True))
        self.conv_6 = nn.Sequential(nn.Conv3d(params.num_features, params.activation_maps, params.kernel, params.stride, padding=params.kernel//2),nn.ReLU(inplace=True))
        self.conv_7 = nn.Sequential(nn.Conv3d(1, params.num_features, params.kernel, padding=params.kernel//2),nn.ReLU(inplace=True))
        self.conv_8 = nn.Sequential(nn.Conv3d(params.num_features, params.num_features, params.kernel, padding=params.kernel//2),nn.ReLU(inplace=True))
        self.conv_9 = nn.Sequential(nn.Conv3d(params.num_features, params.num_features, params.kernel, padding=params.kernel//2),nn.ReLU(inplace=True))
        self.conv_10 = nn.Sequential(nn.Conv3d(params.num_features, 1, params.kernel, padding=params.kernel//2),nn.ReLU(inplace=True))

    def forward(self, image):
        output_1 = self.conv_1(image)
        output_2 = self.conv_2(output_1)
        output_3a = self.conv_3(output_2)
        output_3 = torch.add(torch.mul(output_1,1),output_3a)
        output_4 = self.conv_4(output_3)
        output_5a = self.conv_5(output_4)
        output_5 = torch.add(torch.mul(output_1,1),output_5a)
        output_6 = self.conv_6(output_5)
        output_7 = pixel_shuffle(output_6,params.scale_factor) #Interm loss
        output_8 = self.conv_7(output_7)
        output_9 = self.conv_8(output_8)
        output_10a = self.conv_9(output_9)
        output_10 = torch.add(torch.mul(output_8,1),output_10a)
        output = self.conv_10(output_10) #Final Loss
        return output_7,output
