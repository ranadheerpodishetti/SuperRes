
#Class for reading LR and HR images 

import torch
import numpy as np
import math
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import subprocess
import pandas as pd
from torch.utils.data.dataset import Dataset
import scipy.io
import nibabel as nib

class CustomDatasetFromCSV(Dataset):
    def __init__(self, csv_path, data_path):
        self.data = pd.read_csv(csv_path)
        self.data_path= data_path
        self.is_transform = True
        
    def transform(self, image):
        image_torch = torch.ShortTensor(image)
        return image_torch
      
    def __getitem__(self, idx):
        img_id = str(self.data.iloc[idx, 0]) #Can be changed 
        hr_path = self.data_path + img_id + 'GT.nii.gz'
        lr_path = self.data_path + img_id + 'Under.nii.gz'
        # read nii file, change it to numpy and do a transpose to (z,x,y), Old approach not relevant
        nii_hr = nib.load(hr_path)
        image_hr = np.array(nii_hr.dataobj)#np.array(np.transpose(nii.dataobj,(2,0,1))) 
        # read nii file, change it to numpy and do a transpose to (z,x,y) , Old approach not relevant
        nii_lr = nib.load(lr_path)
        image_lr = np.array(nii_lr.dataobj) #np.array(np.transpose(nii_lr.dataobj,(2,0,1))) 
        if(self.is_transform):
            sample_lr = self.transform(image_lr)
            sample_hr = self.transform(image_hr)
            
        return sample_lr, sample_hr
    def __len__(self):
        return len(self.data)
