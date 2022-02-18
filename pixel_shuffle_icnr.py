import torch
import torchio
import nibabel  as nib
import numpy as np
import os
from scipy import signal
import scipy.fft as fft
import h5py
import pixel_shuffle

# images_original = []
# img_1 = nib.load('E:/master/sem2/supermudi/cdmri0011/MB_Re_t_moco_registered_applytopup_anisotropic_voxcor_cdmri0011_0.nii.gz')
# img1 = np.array(img_1.dataobj)
# #print(type(img1))
# img = torch.from_numpy(img1)
# # images_original.append([np.array(img.dataobj, dtype=np.float32)])
# # img_original = np.array(images_original, dtype=np.float32)
# img = torch.unsqueeze(img , dim = 0)
# img = torch.unsqueeze(img , dim = 0)
# #print(img.size())
# #img_original = torch.from_numpy(img_original)
#
# img_original = torch.nn.functional.interpolate(img, size=(76, 92, 56),
#                                           mode='nearest')
# img_original = torch.squeeze(img_original)
# nii_input = nib.Nifti1Image(img_original.numpy(), np.eye(4))
# #img_original = torch.tensor(nii_input.dataobj)
# #nib.save(nii_input, os.path.join('E:/master/sem2', 'mudi.nii.gz'))
# f = signal.resample(img1, 96, axis = 0)
# f = signal.resample(f, 96, axis = 1)
# f = signal.resample(f, 64, axis = 2)
#
# nii_input1 = nib.Nifti1Image(f, img_1.affine)
# #img_original = torch.tensor(nii_input.dataobj)
# #nib.save(nii_input1, os.path.join('E:/master/sem2', 'mudi.nii.gz'))
# source_size = img1.shape
# target_size = source_size[:-1] + (source_size[-1]*2,)
#
# print(f.shape,target_size)
import torch
import torch.nn as nn


def ICNR(tensor, upscale_factor=2, inizializer=nn.init.kaiming_normal_):
    new_shape = [int(tensor.shape[0] / (upscale_factor ** 2))] + list(tensor.shape[1:])
    subkernel = torch.zeros(new_shape)
    # print(subkernel)
    subkernel = inizializer(subkernel)
    # print(subkernel)
    subkernel = subkernel.transpose(0, 1)

    subkernel = subkernel.contiguous().view(subkernel.shape[0],
                                            subkernel.shape[1], -1)
    #print(subkernel.shape)

    kernel = subkernel.repeat(1, 1, upscale_factor ** 2)

    #print(kernel.shape)
    transposed_shape = [tensor.shape[1]] + [tensor.shape[0]] + list(tensor.shape[2:])
    kernel = kernel.contiguous().view(transposed_shape)
    #print(kernel.shape)

    kernel = kernel.transpose(0, 1)
    #print(kernel.shape)

    return kernel


def listFiles(root):
    allFiles = []
    for folder, folders, files in os.walk(root):
        for file in files: allFiles += [folder.replace("\\", "/") + "/" + file]
    return allFiles

#-------------------------------------------------------testing the code ----------------------------------#
# k = listFiles('E:/master/sem2/superres/IXI555')
# images_original = []
# img_original1 = nib.load(k[1])
# images_original.append([np.array(img_original1.dataobj, dtype=np.float32)])
# img_original = np.array(images_original, dtype=np.float32)
# # print(img_original.shape)
# img_original = torch.from_numpy(img_original)
# # p = ICNR(tensor, 2)
# upscale = 2
# # num_classes = 1
# # previous_layer_features = torch.rand((32, 1, 8, 8, 8))
# conv_shuffle = nn.Conv3d(1, 2 * (upscale ** 2), 3, padding=1)
# ip = conv_shuffle(img_original)
# # print('initial weights 2 ip and 4 op{}'.format(conv_shuffle.weight))
# # #ps = nn.PixelShuffle(upscale)
# kernel = ICNR(conv_shuffle.weight, upscale)
# conv_shuffle.weight.data.copy_(kernel)
# #print('\n ICNR weights {}'.format(conv_shuffle.weight))
# output = pixel_shuffle.pixel_shuffle_generic(conv_shuffle(img_original), 2)
# # for img in img_original:
# #     o = nn.PixelShuffle(2)
# #     o1 = o(img).detach().numpy().squeeze()
# #     print(o1.shape)
# #     nii_input = nib.Nifti1Image(o1, img_original1.affine)
# #     # img_original = torch.tensor(nii_input.dataobj)
# #     nib.save(nii_input, os.path.join('E:/master/sem2/superres/IXI555/', 'hello.nii.gz'))
#
# op = output.squeeze()
# op = op.detach().numpy()
# nii_input = nib.Nifti1Image(op, img_original1.affine)
# # img_original = torch.tensor(nii_input.dataobj)
# #nib.save(nii_input, os.path.join('E:/master/sem2/superres/IXI555/', 'hello.nii.gz'))
# print(output.shape)
#----------------------------------------------------------------------------------------------------------#