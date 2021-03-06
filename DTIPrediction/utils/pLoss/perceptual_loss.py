import math
import torch
import torch.nn as nn
import torchvision
# from utils.utils import *
from pytorch_msssim import SSIM
# from utils.pLoss.Resnet2D import ResNet
# from utils.pLoss.simpleunet import UNet 
# from utils.pLoss.VesselSeg_UNet3d_DeepSup import U_Net_DeepSup

class PerceptualLoss(torch.nn.Module): #currently configured for 1 channel only, with datarange as 1 for SSIM
    def __init__(self, device="cuda:0", loss_model="resnext1012D", n_level=math.inf, resize=None, loss_type="L1"):
        super(PerceptualLoss, self).__init__()
        blocks = []

        if loss_model == "resnet2D": #TODO: not finished
            model = ResNet(in_channels=1, out_channels=1).to(device)
            chk = torch.load(r"./utils/pLoss/ResNet14_IXIT2_Base_d1p75_t0_n10_dir01_5depth_L1Loss_best.pth.tar", map_location=device)
            model.load_state_dict(chk['state_dict'])
        elif loss_model == "unet2D": 
            model = UNet(in_channels=1, out_channels=1, depth=5, wf=6, padding=True,
                            batch_norm=False, up_mode='upsample', droprate=0.0, is3D=False, 
                            returnBlocks=False, downPath=True, upPath=True).to(device)
            chk = torch.load(r"./utils/pLoss/SimpleU_IXIT2_Base_d1p75_t0_n10_dir01_5depth_L1Loss_best.pth.tar", map_location=device)
            model.load_state_dict(chk['state_dict'])
            blocks.append(model.down_path[0].block.eval())
            if n_level >= 2:
                blocks.append(
                    nn.Sequential(
                        nn.AvgPool2d(2),
                        model.down_path[1].block.eval()
                    )
                )
            if n_level >= 3:
                blocks.append(
                    nn.Sequential(
                        nn.AvgPool2d(2),
                        model.down_path[2].block.eval()
                    )
                )
            if n_level >= 4:
                blocks.append(
                    nn.Sequential(
                        nn.AvgPool2d(2),
                        model.down_path[3].block.eval()
                    )
                )
        elif loss_model == "unet3Dds":
            model = U_Net_DeepSup().to(device)
            chk = torch.load(r"./utils/pLoss/VesselSeg_UNet3d_DeepSup.pth", map_location=device)
            model.load_state_dict(chk['state_dict'])
            blocks.append(model.Conv1.conv.eval())
            if n_level >= 2:
                blocks.append(
                    nn.Sequential(
                        model.Maxpool1.eval(),
                        model.Conv2.conv.eval()
                    )
                )
            if n_level >= 3:
                blocks.append(
                    nn.Sequential(
                        model.Maxpool2.eval(),
                        model.Conv3.conv.eval()
                    )
                )
            if n_level >= 4:
                blocks.append(
                    nn.Sequential(
                        model.Maxpool3.eval(),
                        model.Conv4.conv.eval()
                    )
                )
            if n_level >= 5:
                blocks.append(
                    nn.Sequential(
                        model.Maxpool4.eval(),
                        model.Conv5.conv.eval()
                    )
                )
        elif loss_model == "resnext1012D": 
            model = torchvision.models.resnext101_32x8d()
            model.conv1 = nn.Conv2d(1, model.conv1.out_channels, kernel_size=model.conv1.kernel_size, 
                                    stride=model.conv1.stride, padding=model.conv1.padding, bias=False if model.conv1.bias is None else True)
            model.fc = nn.Linear(in_features=model.fc.in_features, out_features=33, bias=False if model.fc.bias is None else True)
            chk = torch.load(r"./utils/pLoss/ResNeXt-3-class-best-latest.pth", map_location=device)  # ResNet14_IXIT2_Base_d1p75_t0_n10_dir01_5depth_L1Loss_best.pth.tar
            model.load_state_dict(chk)
            model = model.to(device)
            blocks.append(
                    nn.Sequential(
                        model.conv1.eval(),
                        model.bn1.eval(),
                        model.relu.eval(),
                    )
                )
            if n_level >= 2:
                blocks.append(
                    nn.Sequential(
                        model.maxpool.eval(),
                        model.layer1.eval()
                    )
                )
            if n_level >= 3:
                blocks.append(model.layer2.eval())
            if n_level >= 4:
                blocks.append(model.layer3.eval())
            if n_level >= 5:
                blocks.append(model.layer4.eval())
        elif loss_model == "densenet161": 
            model = torchvision.models.densenet161()
            model.features.conv0 = nn.Conv2d(1, model.features.conv0.out_channels, kernel_size=model.features.conv0.kernel_size, 
                                            stride=model.features.conv0.stride, padding=model.features.conv0.padding, 
                                            bias=False if model.features.conv0.bias is None else True)
            model.classifier = nn.Linear(in_features=model.classifier.in_features, out_features=33, bias=False if model.classifier.bias is None else True)
            # chk = torch.load(r"./utils/pLoss/ResNet14_IXIT2_Base_d1p75_t0_n10_dir01_5depth_L1Loss_best.pth.tar", map_location=device)
            # model.load_state_dict(chk['state_dict'])
            model = model.features.to(device)
            blocks.append(
                    nn.Sequential(
                        model.conv0.eval(),
                        model.norm0.eval(),
                        model.relu0.eval(),
                    )
                )
            if n_level >= 2:
                blocks.append(
                    nn.Sequential(
                        model.pool0.eval(),
                        model.denseblock1.eval()
                    )
                )
            if n_level >= 3:
                blocks.append(model.denseblock2.eval())
            if n_level >= 4:
                blocks.append(model.denseblock3.eval())
            if n_level >= 5:
                blocks.append(model.denseblock4.eval())
            
            for bl in blocks:
                for p in bl:
                    if type(p) is str:
                        b = bl[p]
                        for _p in b.__dict__["_modules"]:
                            try:
                                b.__dict__["_modules"][_p].requires_grad = False
                            except:
                                pass
                    else:
                        p.requires_grad = False

        if loss_model != "densenet161": 
            for bl in blocks:
                for b in bl:
                    for params in b.parameters():
                        params.requires_grad = False 
                
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        # self.mean = torch.nn.Parameter(torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        # self.std = torch.nn.Parameter(torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))
        self.resize = resize

        if loss_type == "L1":
            self.loss_func = torch.nn.functional.l1_loss
        elif loss_type == "MultiSSIM":
            self.loss_func = MultiSSIM(reduction='mean').to(device)
        elif loss_type == "SSIM3D":
            self.loss_func = SSIM(data_range=1, size_average=True, channel=1, spatial_dims=3).to(device)
        elif loss_type == "SSIM2D":
            self.loss_func = SSIM(data_range=1, size_average=True, channel=1, spatial_dims=2).to(device)

    def forward(self, input, target):
        # input = (input-self.mean) / self.std
        # target = (target-self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='trilinear' if len(input.shape) == 5 else 'bilinear', size=self.resize, align_corners=False)
            target = self.transform(target, mode='trilinear' if len(input.shape) == 5 else 'bilinear', size=self.resize, align_corners=False)
        loss = 0.0
        x = input
        y = target
        # loss += self.loss_func(x, y)
        for block in self.blocks:
           x = block(x)
           y = block(y)
           loss += self.loss_func(x, y)
        return loss

if __name__ == '__main__':
    x = PerceptualLoss(resize=None).cuda()
    a = torch.rand(2,1,24,24).cuda()
    b = torch.rand(2,1,24,24).cuda()
    l = x(a,b)
    sdsd
