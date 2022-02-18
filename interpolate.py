import torch
import nibabel as nib
import torch.nn.functional as F
import numpy as np
from torchio import DATA
import pandas as pd
import os
import shutil
import argparse

#This file interpolates unevenly sized depth images and saves to the folders provided. 
#Example : if in a dataset we have images with depths 64 and 56, then all the images are interpolated to the depth 64
#Also another task is normalsation of files. 

prsr = argparse.ArgumentParser()
prsr.add_argument('--GroundTruthPath', required=True) #/scratch/gdoddapa/data/dataset/groundTruth
prsr.add_argument('--UnderSampledPath', required=True) #/scratch/gdoddapa/data/dataset/underSampled
prsr.add_argument('--GroundTruthInterpolatedPath', required= True)#/scratch/gdoddapa/data/dataset/groundTruthInterpolated
prsr.add_argument('--UnderSampledInterpolatedPathUnderSize', required=True) #/scratch/gdoddapa/data/dataset/underSampledInterpolatedSRCNN
prsr.add_argument('--UnderSampledInterpolatedPathGTSize', required= True) #/scratch/gdoddapa/data/dataset/underSampledInterpolatedUNET
prsr.add_argument('--GTImageSize',  required=True, nargs='+', type=int)
args = prsr.parse_args()


#This method would generate the gt_img_list or under_img_list needed for the methods original and down respectively 
def listFiles3(root):
    allFiles = []
    for folder, folders, files in os.walk(root):
        for file in files: allFiles += [folder.replace("\\", "/") + "/" + file]
    return allFiles

#This method generated the gt_img_name_list or under_img_name_list needed for the method original or down
def imgNameList(root):
    img = os.listdir(root)
    img_name_list = []
    for i in img:
        i = (i.split('.', 1))
        img_name_list.append(i[0]+ '.' + i[-1])
    #print(img_name_list)
    return img_name_list 
    
    


#This method is used for interpolation of ground truth images
def original(depth, height, width, gt_img_list, gt_img_name_list, gt_img_interpolated_path):
    for i in range(len(gt_img_list)):
        images_original = []
        img_original1 = nib.load(gt_img_list[i])
        images_original.append([np.array(img_original1.dataobj, dtype=np.float32)])
        img_original = np.array(images_original, dtype=np.float32)
        # print(img_original.shape)
        img_original = torch.from_numpy(img_original)
        if img_original.size()[-1] <= depth/2 and(img_original.size()[3] >= width/2 or img_original.size()[2] >= height/2):
            img_original = F.interpolate(img_original, size=(img_original.size()[2], img_original.size()[3], img_original.size()[-1]*2),
                                         mode='nearest')

        elif img_original.size()[-1] <= depth/2 and(img_original.size()[3] <= width/2 or img_original.size()[2] <= height/2):
            img_original = F.interpolate(img_original, size=(
            img_original.size()[2]*2, img_original.size()[3]*2, img_original.size()[-1] * 2),
                                         mode='nearest')
            #print('in loop', 38)
            #print(img_original.size())

        img_original = F.interpolate(img_original, size=(height, width, depth),
                                         mode='nearest')
        #print(img_original.size())
        img_original = torch.squeeze(img_original)
        img_original = img_original / torch.max(img_original)
        #Normalisation Check
        if(torch.max(img_original) != 1):print(gt_img_name_list[i] + ' - Normalisation not done')
        nii_input = nib.Nifti1Image(img_original.numpy(), img_original1.affine)
        nib.save(nii_input, os.path.join(gt_img_interpolated_path, gt_img_name_list[i]))
        #Shape of the image check
        gt_test = (height,width,depth)
        if(gt_test!=nii_input.shape): print(gt_img_name_list[i] + " - Shape error")






#Down method is for undersampled images interpolation, it interpolates to undersampled size(Ex: Needed for SRCNN) and also to ground truth size(Ex: needed for UNET) 
def down( under_depth, gt_depth,  gt_height, gt_width, under_img_list, under_img_name_list, under_interpolation_path_undersize, under_interpolation_path_gtsize):
   for i in range(len(under_img_list)):
       images_down = []
       img_down = nib.load(under_img_list[i])
       images_down.append([np.array(img_down.dataobj, dtype=np.float32)])
       img_down = np.array(images_down, dtype=np.float32)
       #print(img_down.shape)
       img_down = torch.from_numpy(img_down)
       if img_down.size()[-1] != gt_depth:
           if img_down.size()[-1] != gt_depth / 2:
               #print('28',under_img_name_list[i])
               #print(gt_width//2 + gt_depth//2, under_depth)
               img_down = F.interpolate(img_down, size=(gt_width // 2, gt_height // 2, under_depth),
                                       mode='nearest')
               img_down1 = torch.squeeze(img_down)
               img_down1 = img_down1 / torch.max(img_down1)
               if (torch.max(img_down1) != 1): print(under_img_name_list[i],' - Normalisation error for under size image')
               nii_input1 = nib.Nifti1Image(img_down1.numpy(), np.eye(4))
               img_original = torch.tensor(nii_input1.dataobj)
               nib.save(nii_input1, os.path.join(under_interpolation_path_undersize, under_img_name_list[i]))
               img_down = F.interpolate(img_down, size=(gt_width, gt_height, gt_depth),
                                       mode='nearest')


               img_down = torch.squeeze(img_down)
               img_down = img_down/torch.max(img_down)
               if (torch.max(img_down) != 1): print(under_img_name_list[i],' - Normalisation error for gt size image')
               nii_input = nib.Nifti1Image(img_down.numpy(), np.eye(4))
               img_original = torch.tensor(nii_input.dataobj)
               nib.save(nii_input, os.path.join(under_interpolation_path_gtsize, under_img_name_list[i]))
    #save image here
           else:
    #save image here

               img_down1 = torch.squeeze(img_down)
               img_down1 = img_down1 / torch.max(img_down1)
               if (torch.max(img_down1) != 1): print(under_img_name_list[i],' - Normalisation error for under size image')
               nii_input = nib.Nifti1Image(img_down1.numpy(), np.eye(4))
               nib.save(nii_input, os.path.join(under_interpolation_path_undersize, under_img_name_list[i]))
               img_down = F.interpolate(img_down, size=(gt_width, gt_height, gt_depth),
                                       mode='nearest')
               #print('name',under_img_name_list[i])
               img_down = torch.squeeze(img_down)
               img_down = img_down / torch.max(img_down)
               if (torch.max(img_down) != 1): print(under_img_name_list[i],' - Normalisation error for gt size image')
               nii_input = nib.Nifti1Image(img_down.numpy(), np.eye(4))
               nib.save(nii_input, os.path.join(under_interpolation_path_gtsize, under_img_name_list[i]))
#save image here


gt_img_list = listFiles3(args.GroundTruthPath)
gt_img_name_list = imgNameList(args.GroundTruthPath)
gt_image_size = tuple(args.GTImageSize)

under_img_list = listFiles3(args.UnderSampledPath)
under_img_name_list = imgNameList(args.UnderSampledPath)

original(gt_image_size[2], gt_image_size[1], gt_image_size[0], gt_img_list, gt_img_name_list, args.GroundTruthInterpolatedPath)
down((gt_image_size[2]//2), gt_image_size[2], gt_image_size[1], gt_image_size[0], under_img_list, under_img_name_list, args.UnderSampledInterpolatedPathUnderSize, args.UnderSampledInterpolatedPathGTSize)

#down(32, 64, 128, k, name, path, path1)
#original(64, 128, 128, k, name, path)
