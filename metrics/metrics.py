import torch
import nibabel as nib
import torch.nn.functional as F
import numpy as np
from torchio import DATA
import pandas as pd
import os
import shutil
import math
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM



def listFiles3(root):
    allFiles = []
    for folder, folders, files in os.walk(root):
        for file in files: allFiles += [folder.replace("\\", "/") + "/" + file]
    return allFiles

def psnr(img1, img2):
    error = torch.nn.MSELoss(reduction= 'mean')
    mse = error(img1, img2)
    if mse == 0:
        return 100
    PIXEL_MAX = torch.max(img1)
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


dict = dict()
excel =[]
groundtruth = listFiles3('E:/master/sem2/superres/file zilla/result256/results/groundtruth')
test_list = listFiles3('E:/master/sem2/superres/file zilla/result256/results/test_images')
org_img = os.listdir('E:/master/sem2/superres/file zilla/result256/results/groundtruth')
test_img = os.listdir('E:/master/sem2/superres/file zilla/result256/results/test_images')

for i in range(len(groundtruth)):
    images_gt = []
    gt = nib.load(groundtruth[i])
    images_gt.append([np.array(gt.dataobj, dtype=np.float32)])
    img_gt = np.array(images_gt, dtype=np.float32)
    # print(img_original.shape)
    img_ground = torch.from_numpy(img_gt)

    images_test = []
    test = nib.load(test_list[i])
    images_test.append([np.array(test.dataobj, dtype=np.float32)])
    img_test = np.array(images_test, dtype=np.float32)
    # print(img_original.shape)
    img_test = torch.from_numpy(img_test)

    Psnr = psnr(img_ground, img_test)
    ssim_val = ssim(img_ground, img_test, data_range=1, size_average=False)
    print('PSNR in dB = ', Psnr)
    print('SSIM = ', ssim_val.item())
    dict.update({'groundtruth':org_img[i], 'test': test_img[i], 'PSNR in dB': Psnr, 'SSIM': ssim_val.item() })
    dict1 = dict.copy()
    excel.append(dict1)

print(excel)
df = pd.DataFrame(excel, index=range(len(groundtruth)))
df.to_csv(os.path.join('E:/master/sem2/superres/execution', 'metrics.csv'), index=False)
print(df)
