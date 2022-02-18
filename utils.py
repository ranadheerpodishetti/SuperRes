import os
import random
import torch
import numpy as np
import pandas as pd

# def listFiles(root):
#     allFiles = []
#     for folder, folders, files in os.walk(root):
#         for file in files: allFiles += [folder.replace("\\", "/") + "/" + file]
#     return allFiles


#l = listFiles('E:/master/sem2/superres/execution/down_unet_interpolated')

#p = listFiles('E:/master/sem2/superres/execution/original_interpolated')
#org_img = os.listdir('E:/master/sem2/superres/execution/original_interpolated')
#down_img = os.listdir('E:/master/sem2/superres/execution/down_unet_interpolated')
#print(p)
def list(groundtruth_path, csv_path):

    groundtruth_name_list = []
    data = pd.read_csv(csv_path, sep =',', header = None)
    data = data[0].tolist()
    #print(type(data[0]))
    for k in range(len(data)):
        for i in os.listdir(groundtruth_path):
            if os.path.isfile(os.path.join(groundtruth_path, i)) and data[k] in i:
                groundtruth_name_list.append(i)
    #print(len(files))
    print(groundtruth_name_list[:17])
    downsampled_name_list = []
    for i in groundtruth_name_list:
        i = i.split('.', 1)

        downsampled_name_list.append(i[0] + '_down.' + i[1])
    print(downsampled_name_list[:17])
    return groundtruth_name_list, downsampled_name_list

def list_mudi(anisotropic_path, csv_path):

    anisotropic_interpolated = []
    data = pd.read_csv(csv_path, sep =',', header = None)
    data = data[0].tolist()
    lists = os.listdir(anisotropic_path)
    #print(lists)
    ## re run interpolate function with -interpolated added at the end for both anisotropic and resized
    for k in data:
        for i in lists:
            #print( i.split('-', 1)[0])
            if k.split('.', 1)[0] == i.split('-', 1)[0]:
                #print(k.split('.', 1)[0], i.split('-', 1)[0])
                anisotropic_interpolated.append(k.split('.', 1)[0]+'-interpolated.' +k.split('.', 1)[-1])
    #print(len(files))
    print(anisotropic_interpolated)
    resized_interpolated = []
    for i in anisotropic_interpolated:
        i = i.split('-', 1)
        #print(i)
        resized_interpolated.append(i[0] +'_resized-' + i[-1] )
    print(resized_interpolated[:17])
    return  anisotropic_interpolated, resized_interpolated

#list_mudi('E:/master/sem2/supermudi/anisotropic_interpolated', 'E:/master/sem2/supermudi/supermudi.csv')


#list('E:/master/sem2/superres/execution/original_interpolated', 'E:/master/sem2/superres/execution/info-data.csv')
# print('org_img',org_img)
#print(org_img)

# name = []
# names= []
# for i, j in zip(org_img, down_img):
#     i = i.split('-')
#     j = j.split('-')
#     #print(i, j)
#     name.append(i[0] + '-' + i[1] + '-' + i[2] + '-' + i[3])
#     names.append(j[0] + '-' + j[1] + '-' + j[2] + '-' + j[3])
#
# for k in range(len(name)):
#     if(name[k] != names[k]):
#         print(name[k], names[k])
#     #else:
#     #    print('no mismatch')
#
# images_down = []
# tensor = torch.rand((128, 128, 64))
# images_down.append([np.array(tensor, dtype=np.float32)])
# img_down = np.array(images_down, dtype=np.float32)
# print(img_down.shape)
# tensor = torch.unsqueeze(tensor, dim = 0)
# tensor = torch.unsqueeze(tensor, dim = 0)
# print(tensor.size())


#
# for k in range(len(data)):
#     for i in os.listdir(path2):
#         if os.path.isfile(os.path.join(path2,i)) and data[k] in i:
#             file.append(i)
# print(len(file))
# print(file)
