import torch
from torch.nn import Module
import os
import nibabel as nib
import numpy as np


def pixel_unshuffle(input, upscale_factor):
    r"""Rearranges elements in a Tensor of shape :math:(C, rH, rW) to a
    tensor of shape :math:(*, r^2C, H, W).
    written by: Zhaoyi Yan, https://github.com/Zhaoyi-Yan
    and Kai Zhang, https://github.com/cszn/FFDNet
    01/01/2019
    """
    batch_size, channels, depth, in_height, in_width = input.size()

    depth_final = depth // upscale_factor
    out_height = in_height // upscale_factor
    out_width = in_width // upscale_factor

    input_view = input.contiguous().view(
        batch_size, channels, depth_final, upscale_factor, out_height, upscale_factor,
        out_width, upscale_factor)

    channels *= upscale_factor ** 3
    unshuffle_out = input_view.permute(0, 1, 3, 5, 7, 2, 4, 6).contiguous()
    return unshuffle_out.view(batch_size, channels, depth_final, out_height, out_width)


class PixelUnShuffle(Module):
    # r"""Rearranges elements in a Tensor of shape :math:(C, rH, rW) to a
    # tensor of shape :math:(*, r^2C, H, W).
    # written by: Zhaoyi Yan, https://github.com/Zhaoyi-Yan
    # and Kai Zhang, https://github.com/cszn/FFDNet
    # 01/01/2019
    # """

    def __init__(self, upscale_factor):
        super(PixelUnShuffle, self).__init__()
        self.upscale_factor = upscale_factor

    def forward(self, input):
        return pixel_unshuffle(input, self.upscale_factor)

    def extra_repr(self):
        return 'upscale_factor={}'.format(self.upscale_factor)


# ------------------------------------------- testing ------------------------------------------------------

def listFiles(root):
    allFiles = []
    for folder, folders, files in os.walk(root):
        for file in files: allFiles += [folder.replace("\\", "/") + "/" + file]
    return allFiles


if __name__ == "__main__":
    tensor = torch.rand((32, 1, 32, 64, 64))
    model = pixel_unshuffle(tensor, 2)
    print(model.size())
    k = listFiles('E:/master/sem2/superres/execution/original_normalized')
    images_original = []
    img_original1 = nib.load(k[0])
    images_original.append([np.array(img_original1.dataobj, dtype=np.float32)])
    img_original = np.array(images_original, dtype=np.float32)
    # print(img_original.shape)
    img_original = torch.from_numpy(img_original)
    print(img_original.shape)
    # p = ICNR(tensor, 2)
    upscale = 2
    out = pixel_unshuffle(img_original, upscale)
    op = out.squeeze()
    print(op.size())
    for op in op:
       op = op.detach().numpy()
       nii_input = nib.Nifti1Image(op, img_original1.affine)
       nib.save(nii_input, os.path.join('E:/master/sem2/superres/IXI555/', 'hello.nii.gz'))
    print(out.shape)


# ----------------------------------------------------------------------------------------------------------
