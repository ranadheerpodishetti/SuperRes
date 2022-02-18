import torch
import torch.nn as nn
import torch.nn.functional as F
import nibabel as nib
import os
import numpy as np


def listFiles3(root):
    allFiles = []
    for folder, folders, files in os.walk(root):
        for file in files: allFiles += [folder.replace("\\", "/") + "/" + file]
    return allFiles


groundtruth = listFiles3('D:/wavenet_results/downsampled_original')
gt_list = os.listdir('D:/wavenet_results/downsampled_original')
print(gt_list)
for i in range(len(groundtruth)):
    images_gt = []
    name = gt_list[i].split('.')
    trilinear_name = name[0] + '_trilinear_interpolated.nii.gz'
    print(trilinear_name)
    gt = nib.load(groundtruth[i])
    images_gt.append([np.array(gt.dataobj, dtype=np.float32)])
    img_gt = np.array(images_gt, dtype=np.float32)
    # print(img_original.shape)
    img_ground = torch.from_numpy(img_gt)

    interp = F.upsample(img_ground, scale_factor=2, mode='trilinear')
    interp = interp/torch.max(interp)
    if torch.max(interp) != 1:
        print(torch.max(interp))

    trilinear_interp = interp.squeeze()
    tril = np.array(trilinear_interp)
    print(tril.shape)
    nii = nib.Nifti1Image(tril, np.eye(4))
    nib.save(nii, os.path.join('D:/wavenet_results/outputs/trilinear_interpolation', trilinear_name))
