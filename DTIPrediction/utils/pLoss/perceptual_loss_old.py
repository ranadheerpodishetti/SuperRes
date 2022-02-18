import math
import torch
import torch.nn as nn
import torchvision
from utils import *
# from utils.pytorch_ssim_3D import SSIM3D
import pytorch_ssim
from utils.pLoss.Resnet2D import ResNet
from utils.pLoss.simpleunet import UNet 
from utils.pLoss.VesselSeg_UNet3d_DeepSup import U_Net_DeepSup

class PerceptualLoss(torch.nn.Module):
    def __init__(self, device="cuda:0", loss_model="unet2D", n_level=math.inf, resize=None, loss_type="L1"):
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
                
        for bl in blocks:
            for p in bl:
                p.requires_grad = False
                
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
            self.loss_func = SSIM3D().to(device)
        elif loss_type == "SSIM2D":
            self.loss_func = pytorch_ssim.SSIM().to(device)

    def forward(self, input, target):
        # input = (input-self.mean) / self.std
        # target = (target-self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='trilinear' if len(input.shape) == 5 else 'bilinear', size=self.resize, align_corners=False)
            target = self.transform(target, mode='trilinear' if len(input.shape) == 5 else 'bilinear', size=self.resize, align_corners=False)
        loss = 0.0
        x = input
        y = target
        for block in self.blocks:
            x = block(x)
            y = block(y)
            loss += self.loss_func(x, y)
        return loss

if __name__ == '__main__':
    x = PerceptualLoss(resize=None)
    a = torch.rand(2,1,24,24).cuda()
    b = torch.rand(2,1,24,24).cuda()
    l = x(a,b)
