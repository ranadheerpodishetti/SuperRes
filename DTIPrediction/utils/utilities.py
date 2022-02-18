import os
import numpy as np
import nibabel as nib
import glob

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import torch.nn.functional as F
import torchvision.utils as vutils

import warnings
warnings.filterwarnings("ignore")

##########################################################################
##########################################################################
##########################################################################
def norm_for_visual(input_):
  img = torch.abs(input_)/torch.abs(input_).max()
  return img

def tensorboard_images(writer, inputs, outputs, targets,epoch, section='train'):
    fact = 10.0
    writer.add_image('{}/input'.format(section),
                     vutils.make_grid(inputs[0, 0, ...],
                                      normalize=True,
                                      scale_each=True),
                     epoch)
    # ----------------------------------------
    writer.add_image('{}/output'.format(section),
                     vutils.make_grid(torch.cat((outputs[0, 0, ...], 
                                                 norm_for_visual(outputs[0, 1, ...]),
                                                 norm_for_visual(outputs[0, 2, ...]),
                                                 norm_for_visual(outputs[0, 3, ...]), 
                                                 norm_for_visual(outputs[0, 4, ...]),
                                                 norm_for_visual(outputs[0, 5, ...]),
                                                 norm_for_visual(outputs[0, 6, ...])), axis=1),
                                      normalize=True,
                                      scale_each=True),
                     epoch)
    
    # ----------------------------------------
    writer.add_image('{}/target'.format(section),
                     vutils.make_grid(torch.cat((targets[0, 0, ...],
                                                 norm_for_visual(targets[0, 1, ...]),
                                                norm_for_visual(targets[0, 2, ...]),
                                                norm_for_visual(targets[0, 3, ...]),
                                                 norm_for_visual(targets[0, 4, ...]),
                                                 norm_for_visual(targets[0, 5, ...]),
                                                 norm_for_visual(targets[0, 6, ...])), axis=1),
                                      normalize=True,
                                      scale_each=True),
                     epoch)
##########################################################################
# aux functions  #########################################################
##########################################################################

def adjust_learning_rate(optimizer, epoch, lrInitial, lrDecayNEpoch, lrDecayRate):
    """Sets the learning rate to the initial LR decayed by lrDecayRate every lrDecayNEpoch epochs"""

    lr = lrInitial * (lrDecayRate ** (epoch // lrDecayNEpoch))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

##########################################################################
##########################################################################
##########################################################################

def normalize_slice_by_slice(input_vol_):
    vol_ = np.zeros((input_vol_.shape))
    for ii in range(0,vol_.shape[2]):
        tmp_ = input_vol_[:,:,ii]/input_vol_[:,:,ii].max()
        where_are_nan = np.isnan(tmp_)
        tmp_[where_are_nan] = 0
        vol_[:,:,ii] = tmp_
    
    return vol_ 
###############################################################################
###############################################################################
###############################################################################

class DatasetB0toTensor():
    """DatasetB0toTensor"""

    def __init__(self, input_, target_, transform=None):
        
        self.files_in_ = input_
        self.files_out_ = target_
        
        self.transform = transform

    def __len__(self):
        return (self.files_in_.shape[2])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        input_ =  self.files_in_[:,:,idx] 
        target_ = self.files_out_[:,:,idx,:]
        
        input_ = np.swapaxes(input_, 1,0)
        input_ = np.expand_dims(input_, 0)
        target_ = np.swapaxes(target_, 0,2)

        # imga = nib.Nifti1Image(input_, np.eye(4))
        # imgb = nib.Nifti1Image(input_, np.eye(4))
        # nib.save(img, 'test_corrupted_FLsag_.nii.gz')

        if self.transform:
            input_ = self.transform(input_)
            target_ = self.transform(target_)
            

        return input_, target_

###############################################################################
###############################################################################
###############################################################################

class DatasetUNetMax():
    """DatasetB0toTensor"""

    def __init__(self, input_, target_, transform=None):
        
        self.files_in_ = input_
        self.files_out_ = target_
        
        self.transform = transform

    def __len__(self):
        return (self.files_in_.shape[2])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        input_ =  self.files_in_[:,:,idx] 
        target_ = self.files_out_[:,:,idx,:]
        
        # input_ = np.swapaxes(input_, 1,0)
        input_ = np.expand_dims(input_, 2)
        input_ = np.repeat(input_, 7, axis=2)
        input_ = np.swapaxes(input_, 0, 2)
        # input_ = np.swap
        ## input_ = np.repeat(input_, 6)
        ## input_ = np.expand_dims(input_, 0)
        target_ = np.swapaxes(target_, 0,2)
        # print(input_.shape, target_.shape)
        # imga = nib.Nifti1Image(input_, np.eye(4))
        # imgb = nib.Nifti1Image(input_, np.eye(4))
        # nib.save(img, 'test_corrupted_FLsag_.nii.gz')

        if self.transform:
            input_ = self.transform(input_)
            target_ = self.transform(target_)
            

        return input_, target_
        
###############################################################################
###############################################################################
###############################################################################

class DatasetB0toTensorMask():
    """DatasetB0toTensorMask"""

    def __init__(self, input_, mask_, target_, transform=None):
        
        self.files_in_ = input_
        self.files_mask_ = mask_
        self.files_out_ = target_
        
        self.transform = transform

    def __len__(self):
        return (self.files_in_.shape[2])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        input_ =  self.files_in_[:,:,idx] 
        mask_ = self.files_mask_[:,:,idx]
        target_ = self.files_out_[:,:,idx,:]

        input_ = input_ * mask_
        for ii in range(0,target_.shape[2]):
            target_[:,:,ii] = mask_ * target_[:,:,ii]
        ## target_ = target_ * mask_ 
        
        input_ = np.swapaxes(input_, 1,0)
        input_ = np.expand_dims(input_, 0)
        target_ = np.swapaxes(target_, 0,2)

        # imga = nib.Nifti1Image(input_, np.eye(4))
        # imgb = nib.Nifti1Image(input_, np.eye(4))
        # nib.save(img, 'test_corrupted_FLsag_.nii.gz')

        if self.transform:
            input_ = self.transform(input_)
            target_ = self.transform(target_)
            

        return input_, target_       
 