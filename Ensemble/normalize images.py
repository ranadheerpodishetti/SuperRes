import os
#import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import nibabel as nb
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import random


random.seed(10)
#path = 'P:\OVGU_projects\Artefact detection and segmentation\Data_Modified\stackednii\cross'

path = 'P:\OVGU_projects\SuperResolution\downsampled'

#os.chdir(path)
nifti_norm = 'P:\OVGU_projects\SuperResolution\downsampled_normalized'

for file in os.listdir(path):
    #if file.endswith('.gz'):
    path_file = path + '\\' + file
    print('path_file',path_file)
        # saving normalized images to avoid lossy conversion since image is converted from uint8 to int16 as eg.
    #img_name = path_file.split('\\')
    #img_name = path_file.split('\\')
    #img_name1 = img_name[-1].split('.', 1)
        #img_name1 = img_name[-1].split('.')
        #img_name2 = img_name1[0]
        #img_nii = img_name[-2] + '--' + img_name2
        # print(img_nii)
    img = nb.load(path_file)

    img = img.dataobj[:,:,:]
    print(img.shape)
    img_norm = img / np.max(img)
    print(img_norm.shape)
    print(np.max(img_norm))
    print(np.min(img_norm))
    #img_norm = np.flip(img_norm, axis=1)
    save_niftiname = file
    print('save_niftiname', save_niftiname)
    #save_niftiname = path.split('\\')[-1]

    nii_input = nb.Nifti1Image(img_norm, np.eye(4))
    nb.save(nii_input, os.path.join(nifti_norm, save_niftiname))





