import numpy as np
import dipy
import nibabel as nib
import matplotlib.pyplot as plt
import dipy.denoise.nlmeans
from skimage import data, img_as_float
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage.metrics import peak_signal_noise_ratio
from skimage.util import random_noise
import time


def listFiles(root):
    allFiles = []
    for folder, folders, files in os.walk(root):
        for file in files: allFiles += [folder.replace("\\", "/") + "/" + file]
    return allFiles


def denoise(img):
    sigma = estimate_sigma(img, average_sigmas=True, multichannel=False)
    img = dipy.denoise.nlmeans.nlmeans(img, sigma, mask=None, patch_radius=1, block_radius=5, rician=True,
                                       num_threads=None)
    print(sigma)
    return img, sigma


def signaltonoise_dB(a, sd, axis=-1, ddof=0):
    m = a.mean(axis)
    SNR = 20 * np.log10(np.mean(m / sd))
    print('signal to noise ratio =', SNR)
    # sd = a.std(axis=axis, ddof=ddof)
    #print('end',time.perf_counter())
    return SNR

groundtruth_path = listFiles('D:/MASTERS/12cp/original/training_data')
shuffleunet_path = listFiles('D:/MASTERS/12cp/wavenet')

gt_SNR = []
shuffle_SNR = []

for i in groundtruth_path:
    gt_img = nib.load(i).get_fdata()
    denoise_op, sigma = denoise(img)
    snr_gt = signaltonoise_dB(denoise_op,sigma)
    gt_img_name = i.split('/')
    gt_SNR.append([gt_img_name[-1],snr_gt])

import csv

gt_df = pd.DataFrame(gt_SNR)
gr_df.to_csv('denoised_groundtruth_SNR.csv', index=False, header=True)


for j in shuffleunet_path:
    shuffle_img = nib.load(i).get_fdata()
    sigma = estimate_sigma(shuffle_img, average_sigmas=True, multichannel=False)
    snr_shf = signaltonoise_dB(shuffle_img,sigma)
    shf_img_name = j.split('/')
    shuffle_SNR.append([shf_img_name[-1],snr_shf])

shf_df =pd.Dataframe(shuffle_SNR)
shf_df.to_csv('shuffle_output_SNR.csv',index = False,header = True )

# print(denoise(img))

signaltonoise_dB(denoise_op, sigma, axis=-1, ddof=0)
