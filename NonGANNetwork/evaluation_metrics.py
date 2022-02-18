#Author: Geetha Doddapaneni Gopinath

import skimage.measure as measure
import torch
import numpy as np


def ssim(img_gt, img_test,Norm_factor):
    img_gt = img_gt.float()/Norm_factor
    img_gt = img_gt.numpy()
    
    img_test = img_test.numpy()
    
    ssim=[]
    for i in range(img_true.shape[0]):
        ssim = np.append(ssim, measure.compare_ssim(img_true[i], img_test[i]))
    return ssim

def psnr(img_gt, img_test,Norm_factor):
    img_gt = img_gt.float()/Norm_factor  
    img_gt = img_gt.numpy()
    
    img_test = img_test.numpy()
    psnr=[]
    for i in range(img_gt.shape[0]):
        psnr = np.append(psnr, measure.compare_psnr(img_gt[i], img_test[i]))
    return psnr

def nrmse(img_gt, img_test,Norm_factor):
    img_gt = img_gt.float()/Norm_factor
    img_gt = img_gt.numpy()
    
    img_test = img_test.numpy()
    nrmse=[]
    for i in range(img_gt.shape[0]):
        nrmse = np.append(nrmse, measure.compare_nrmse(img_gt[i], img_test[i]))
    return nrmse


