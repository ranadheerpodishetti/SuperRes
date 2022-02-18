import torch
import nibabel as nib
import torch.nn.modules.loss as nn
import numpy as np
from torchio import DATA
import pandas as pd
import os
import shutil
import math
from metrics_config import Config
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM

configs = Config()
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
groundtruth = listFiles3('D:/wavenet_results/outpts/gt')
test_list = listFiles3('D:/wavenet_results/outputs/baseline_unet_output_1')
org_img = os.listdir('D:/wavenet_results/outpts/gt')
test_img = os.listdir('D:/wavenet_results/outputs/baseline_unet_output_1')
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
    img_test = img_test/torch.max(img_test)
    Psnr = psnr(img_ground, img_test)
    Rmse = nn.MSELoss(reduction= 'mean')
    Rmse = math.sqrt(Rmse(img_ground, img_test).item())
    ssim_val = ssim(img_ground, img_test, data_range=1, size_average=True, win_sigma=1.5, win_size=3)
    ms_ssim_val = ms_ssim(img_ground, img_test, data_range=1, size_average=True, win_size= 3)
    print('Rmse = ',Rmse)
    print('PSNR in dB = ', Psnr)
    print('SSIM = ', ssim_val.item())
    print('MS-SSIM = ',ms_ssim_val.item())
    dict.update({'groundtruth':org_img[i], 'test': test_img[i], 'PSNR in dB': Psnr, 'SSIM': ssim_val.item(), 'MS-SSIM': ms_ssim_val.item(), 'RMSE': Rmse})
    dict1 = dict.copy()
    excel.append(dict1)

df = pd.DataFrame(excel, index=range(len(groundtruth)))
df.to_csv(os.path.join('D:/wavenet_results', 'unet_comp_new.csv'), index=False)
print(df.mean(), df.std())
