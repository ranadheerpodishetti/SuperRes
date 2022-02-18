##This code does not use grid aggregator


############## other file --- main_w/Grid.py
import torch
import nibabel as nib
import dipy
from dipy.io.image import load_nifti
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchio
from torchio import ImagesDataset, Image, Subject
import torch.nn.functional as F
from skimage import io, transform
import numpy as np
from torchvision.utils import make_grid, save_image
from torchio.transforms import RandomAffine
from torchio.transforms.interpolation import Interpolation
from torchvision import transforms
import os
import torch.optim as optim
from numba import cuda
import torchio
from torchio import AFFINE, DATA, PATH, TYPE, STEM
import random
import SimpleITK as sitk
import matplotlib.pyplot as plt
from scipy import stats
from torchvision.utils import make_grid, save_image
from nilearn import plotting
import pandas as pd
import argparse
import torch.cuda.amp
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
import os.path
import time, datetime
import SRCNN
import modified_unet_1
#import interpolate
import patch_loading
import checkpoints
import subject_loading
import srcnn_subject_loading
import train
import test
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
import perceptual_loss
import config_rec_res_with_att_l1
from config_rec_unet_with_pus_ps_shallow import Config
import dynamic_unet
import Rec_UNet_with_pus_ps_shallow

configs = Config()

manualSeed = 999
#manualSeed = random.randint(1, 10000) # use if you want new results
#print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
torch.set_default_dtype(torch.float32)






now = datetime.datetime.today()

time = datetime.datetime.now()

current_time = time.strftime("-%H-%M")
print("Current Time =", current_time)

nTime = now.strftime("-%d-%m-%Y") + current_time
source = configs.result
dest = os.path.join(source+nTime)
if not os.path.exists(dest):
    os.makedirs(dest)
    #os.makedirs(dest + '/valid_logs')
    #os.makedirs(dest + '/train_logs')


writer_train = SummaryWriter(log_dir=os.path.join(dest, configs.trainlogs))
writer_valid = SummaryWriter(log_dir=os.path.join(dest, configs.validlogs))
writer_images = SummaryWriter(log_dir=os.path.join(dest, configs.images))
######## tar file extraction ------------------------------------------------------------------------------------------------------------------#####
# my_tar = tarfile.open('E:/master/sem2/superres/IXI-DTI.tar')
# my_tar.extractall('E:/master/sem2/superres/dataset') # specify which folder to extract to
# my_tar.close()


# -------------------------------------------------------------------------------------------------------------------------------------------------#s


# --------------------------------------------------------------------------------------------------------------------------------------------------##


# The UNet class has 4 layers of double convolutional 3d blocks each followed by MaxPool layer for downsampling.
# #The UNet class has 4 layers of convolutional transpose 3d blocks each followed by double convolutional layer for upsampling.

device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'

if __name__ == "__main__":

    training_batch_size = configs.training_batch_size
    validation_batch_size = configs.validation_batch_size

    patch_size = tuple(configs.patchsize)  # 16,16,16
    patch_size_orig = tuple(configs.patchsize_original)
    samples_per_volume = configs.samples
    # 5

    max_queue_length = configs.maxqueue  # 300

    device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'

    ##refer to patch_loading.py
    # display.Image(image_path)
    fname = configs.checkpointpath
    valid_loss_min = np.inf
    if configs.loss == 'L1':
        criterion = nn.L1Loss(reduction='mean')
    elif configs.loss == 'L2':
        criterion = nn.MSELoss(reduction='mean')
    elif configs.loss == 'SSIM':
        criterion = SSIM(data_range=1, size_average=True, spatial_dims=3, channel=1)
    elif configs.loss == 'Perceptual':
        criterion = perceptual_loss.PerceptualLoss(configs.num_features, configs.kernel_size_1, configs.stride_1,
                                                   configs.kernel_size_2, configs.stride_2, configs.kernel_size_3,
                                                   resize=None)
    elif configs.loss == 'NOVEL':
        criterion = nn.L1Loss(reduction='none')
    else:
        criterion = nn.L1Loss(reduction='none')
    criterion_perc = nn.L1Loss(reduction='mean')
    criterion_novel = MS_SSIM(data_range=1,win_size=1, size_average=True, spatial_dims=3, channel=1)
    if configs.model == 'UNET':
        model = Rec_UNet_with_pus_ps_shallow.R2AttU_Net(configs.num_features, configs.kernel_size_1, configs.stride_1, configs.kernel_size_2, configs.stride_2, configs.kernel_size_3)
        model.to(device)

        optimizer = optim.Adam(model.parameters(), lr=configs.learningrate)
        scaler = GradScaler()
        # opt_level = 'O1'
        # model, optimizer = amp.initialize(model, optimizer, opt_level=opt_level)
        training_set, validation_set = subject_loading.subject_loading(configs.data, configs.groundtruth,
                                                                                 configs.downsampled, configs.inputsize,
                                                                                 os.path.join(dest, configs.original_path),
                                                                                 os.path.join(dest,configs.down_path),
                                                                                 configs.training_split_ratio,
                                                                                 configs.validation_split_ratio, configs.dataset)
        tensor_visual = validation_set[0]
        orig = tensor_visual['original'][DATA].squeeze()
        #orig = orig.unsqueeze(dim= 0)
        orig = orig.permute(2, 0 , 1)
        orig_visualization = orig[32: 33, :, :]
        #print(orig.size())
        down = tensor_visual['down'][DATA].squeeze()
        #down = down.unsqueeze(dim = 0)
        down = down.permute(2, 0, 1)
        down_visualization = down[32: 33, :, :]
        #img_grid = make_grid([orig_visualization, down_visualization])
        #writer_images.add_image('images', img_grid)
        down_valid = torch.unsqueeze(down, dim =0 )
        down_valid = torch.unsqueeze(down_valid, dim=0).to(device)
        print(down.size())


        training_loader, validation_loader = patch_loading.patch_Loading_unet(patch_size,
                                                                                           samples_per_volume,
                                                                                           max_queue_length,
                                                                                           training_set,
                                                                                           validation_set,
                                                                                           configs.patch_check,
                                                                                           configs.training_batch_size,
                                                                                           configs.validation_batch_size)

        print('UNET')

        try:
            if os.path.exists(fname):
                print('file found')
                model, optimizer, start_epoch, valid_loss_min, scaler = checkpoints.load_ckp(configs.checkpointpath, model,
                                                                                          optimizer, scaler)
                print(start_epoch)
                _, valid, train = train.train(configs.epochs - start_epoch, training_loader, validation_loader, valid_loss_min, optimizer,
                            model,
                            writer_train, writer_valid,writer_images, criterion ,criterion_perc,criterion_novel ,configs.checkpointpath, configs.bestmodelckp, configs.model, configs.mode, configs.loss, scaler)
            else:
                _, valid, train = train.train(configs.epochs, training_loader, validation_loader, valid_loss_min, optimizer, model,
                            writer_train,
                            writer_valid,writer_images, criterion , criterion_perc, criterion_novel ,configs.checkpointpath, configs.bestmodelckp, configs.model, configs.mode, configs.loss, scaler)
                #tensor_vis1 = unet(down_valid)
                #tensor_vis = tensor_vis1.squeeze().cpu()
                #tensor_vis = tensor_vis[32:33, :, :]


        except MemoryError as error:
            print('memory error')

        model, optimizer, start_epoch, valid_loss_min = checkpoints.load_ckp(configs.bestmodelckp, model,
                                                                             optimizer)


        #tensor_vis = tensor_vis[32:33, :, :]

    elif configs.model == 'SRCNN':
        activation_maps = configs.scale_factor ** configs.num_dimensions
        model = SRCNN.SRCNN_late_upscaling(configs.num_features, configs.kernel_size_1, configs.stride_1,activation_maps)
        model.to(device)
        criterion = nn.L1Loss(reduction='mean')
        optimizer = optim.Adam(model.parameters(), lr=configs.learningrate)
        opt_level = 'O1'
        model, optimizer = amp.initialize(model, optimizer, opt_level=opt_level)
        training_set_down, validation_set_down, test_set_down, training_set_original, validation_set_original, test_set_original = srcnn_subject_loading.subject_loading(
            configs.data, configs.groundtruth,
            configs.downsampled, configs.inputsize,
            configs.training_split_ratio,configs.validation_split_ratio, configs.dataset)
        test_set = zip(test_set_down, test_set_original)
        ##refer to subject_loading.py
        tensor_visual = validation_set_original[0]
        orig = tensor_visual['original'][DATA].squeeze()
        # orig = orig.unsqueeze(dim= 0)
        orig = orig.permute(2, 0, 1)
        orig_visualization = orig[32: 33, :, :]
        # print(orig.size())
        down = validation_set_down[0]['down'][DATA].squeeze()
        # down = down.unsqueeze(dim = 0)
        down = down.permute(2, 0, 1)
        down_visualization = down[15: 16, :, :]
        img_grid = make_grid([down_visualization])
        writer_images.add_image('images', img_grid)
        down_valid = torch.unsqueeze(down, dim=0)
        down_valid = torch.unsqueeze(down_valid, dim=0).to(device)
        print(down.size())

        training_loader_down, validation_loader_down, test_loader_down = patch_loading.patch_Loading_unet(patch_size,
                                                                                                          samples_per_volume,
                                                                                                          max_queue_length,
                                                                                                          training_set_down,
                                                                                                          validation_set_down,
                                                                                                          test_set_down,
                                                                                                          configs.patch_check,
                                                                                                          configs.training_batch_size,
                                                                                                          configs.validation_batch_size,
                                                                                                          configs.test_batch_size, 'Under')
        training_loader_original, validation_loader_original, test_loader_original = patch_loading.patch_Loading_unet(
            patch_size_orig, samples_per_volume,
            max_queue_length, training_set_original,
            validation_set_original, test_set_original,
             configs.patch_check,
             configs.training_batch_size,
             configs.validation_batch_size,
             configs.test_batch_size, 'go') #Doubt on max_queue_length = 200 and samples_per_volume=8
        training_loader = zip(training_loader_down, training_loader_original)
        training_loader = list(training_loader)

        validation_loader = zip(validation_loader_down, validation_loader_original)
        validation_loader = list(validation_loader)
        test_loader = zip(test_loader_down, test_loader_original)
        test_loader = list(test_loader)
        print('SRCNN')
        try:
            if os.path.exists(fname):
                print('file found')
                model, optimizer, start_epoch, valid_loss_min = checkpoints.load_ckp(configs.checkpointpath, model,
                                                                                           optimizer)
                print(start_epoch)
                _, valid, train = train.train(configs.epochs - start_epoch, training_loader_down, validation_loader_down, valid_loss_min,
                            optimizer, model,
                            writer_train, writer_valid,writer_images, criterion, configs.checkpointpath, configs.bestmodelckp, configs.model, down_valid,  orig_visualization,down_visualization)
            else:
                _, valid, train = train.train(configs.epochs, training_loader, validation_loader, valid_loss_min, optimizer, model,
                            writer_train,
                            writer_valid, writer_images,criterion, configs.checkpointpath, configs.bestmodelckp, configs.model, down_valid,orig_visualization, down_visualization)
        except MemoryError as error:
            print('memory error')

        model, optimizer, start_epoch, valid_loss_min = checkpoints.load_ckp(configs.bestmodelckp, model,
                                                                             optimizer)
    plt.plot(valid, 'g', label='valid')

    plt.plot(train, 'r', label='train')
    plt.legend()
    plt.savefig('curves.png')
