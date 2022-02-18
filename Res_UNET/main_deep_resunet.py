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
import config_deep_res_l1
from config_deep_res_l1 import Config
import dynamic_unet
import Deep_ResUNET

configs = Config()

manualSeed = 999
#manualSeed = random.randint(1, 10000) # use if you want new results
#print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
torch.set_default_dtype(torch.float32)


# prsr = argparse.ArgumentParser()
# prsr.add_argument('--dataset', required=True, type=str) ## configs.data
# prsr.add_argument('--model', required=True, type=str) ## configs.model
# prsr.add_argument('--mode', required= True, type= str) ## configs.mode
# prsr.add_argument('--loss', required= True, type = str)## configs.loss
# prsr.add_argument('--result', required= True)
# prsr.add_argument('--trainlogs', required=True)  # tensorboard logs directory configs.logs
# prsr.add_argument('--validlogs', required=True)  # tensorboard logs directory configs.logs
# prsr.add_argument('--images', required=True)  # tensorboard logs directory configs.images
# prsr.add_argument('--data', required=True)  # csv path configs.data
# prsr.add_argument('--original_path', required=True)  # visualizing configs.original_path
# prsr.add_argument('--down_path', required=True)  # visualizing configs.down_path
# prsr.add_argument('--groundtruth', required=True)  # groundtruth img path configs.groundtruth
# prsr.add_argument('--downsampled', required=True)  # downsampled  img path configs.downsampled
# prsr.add_argument('--inputsize', required=True)  # no of input images according to ram capacity configs.inputsize
# prsr.add_argument('--samples', required=True, type=int)  # samples per epoch configs.samples
# prsr.add_argument('--patchsize', required=True, nargs='+', type=int)  # patch_size configs.patchsize
# prsr.add_argument('--patchsize_original', required=True, nargs='+', type=int)
# prsr.add_argument('--maxqueue', required=True, type=int)  # max queue length configs.maxqueue
# prsr.add_argument('--patch_check', required=True)  # patches images path configs.patch_check
# prsr.add_argument('--learningrate', required=True, type=float)  # learning rate configs.learningrate
# prsr.add_argument('--epochs', required=True, type=int)  # no.of epochs configs.epochs
# prsr.add_argument('--checkpointpath', required=True)  # chkpt path configs.checkpointpath
# prsr.add_argument('--bestmodelckp', required=True)  # best model path configs.bestmodelckp
# prsr.add_argument('--testimage', required=True)  # test image superres path configs.testimage
# prsr.add_argument('--test_superres_path', required=True)  # test image superres path configs.test_superres_path
# prsr.add_argument('--test_gt_path', required=True)  # test image gt path configs.test_gt_path
# prsr.add_argument('--test_inp_path', required=True)  # test image input path configs.test_inp_path
# prsr.add_argument('--training_batch_size',required=True, type= int)  #Training Batch Size
# prsr.add_argument('--validation_batch_size',required=True, type= int) #Validation Batch Size
# prsr.add_argument('--test_batch_size',required=True, type= int) #Test Batch Size
# prsr.add_argument('--training_split_ratio',required=True, type= float)
# prsr.add_argument('--validation_split_ratio',required=True, type= float)
# prsr.add_argument('--num_features',required=True, type= int)
# prsr.add_argument('--kernel_size_1',required=True, type= int) #3 for SRCNN
# prsr.add_argument('--stride_1',required=True, type= int)
# prsr.add_argument('--kernel_size_2',required=True, type= int) #2
# prsr.add_argument('--stride_2',required=True, type= int) #2
# prsr.add_argument('--kernel_size_3',required=True, type= int) #1
# prsr.add_argument('--scale_factor',required=True, type= int) #super resolve by 2
# prsr.add_argument('--num_dimensions',required=True, type= int) #3d Image
# configs = prsr.parse_args()




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
    criterion_perc = perceptual_loss.PerceptualLoss(configs.num_features, configs.kernel_size_1, configs.stride_1,
                                                    configs.kernel_size_2, configs.stride_2, configs.kernel_size_3,
                                                    resize=None)
    criterion_novel = MS_SSIM(data_range=1,win_size=1, size_average=True, spatial_dims=3, channel=1)
    if configs.model == 'UNET':
        model = Deep_ResUNET.ResUnet(1)
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



    # forward pass
    #model_unet = UNet.UNet()
    #model_unet.to(device)
    #model_srcnn = SRCNN.SRCNN_late_upscaling()
    #model_srcnn.to(device)

    #criterion_unet = nn.MSELoss(reduction='mean')
    #criterion_SRCNN = nn.L1Loss(reduction='mean')

    #optimizer_unet = optim.Adam(model_unet.parameters(), lr=configs.learningrate)
    #opt_level = 'O1'
    #model_unet, optimizer_unet = amp.initialize(model_unet, optimizer_unet, opt_level=opt_level)
    # valid_loss_min = np.inf
    #optimizer_srcnn = optim.Adam(model_srcnn.parameters(), lr=configs.learningrate)
    #opt_level = 'O1'
    #model_srcnn, optimizer_srcnn = amp.initialize(model_srcnn, optimizer_srcnn, opt_level=opt_level)
    # if configs.model == 'UNET':
    #     training_set, validation_set, test_set = subject_loading.subject_loading(configs.data, configs.groundtruth,
    #                                                                              configs.downsampled, configs.inputsize, os.path.join(dest, configs.original_path), os.path.join(dest,configs.down_path))
    #     ##refer to subject_loading.py
    #
    #     training_loader, validation_loader, test_loader = patch_loading.patch_Loading_unet(patch_size,
    #                                                                                        samples_per_volume,
    #                                                                                        max_queue_length,
    #                                                                                        training_set,
    #                                                                                        validation_set, test_set,
    #                                                                                        configs.patch_check)
    #     # print((training_loader))
    #     print('UNET')
    #     try:
    #         if os.path.exists(fname):
    #             print('file found')
    #             model_unet, optimizer, start_epoch, valid_loss_min = interpolate.load_ckp(configs.checkpointpath, model_unet,
    #                                                                                       optimizer_unet)
    #             print(start_epoch)
    #             train.train(configs.epochs - start_epoch, training_loader, validation_loader, valid_loss_min, optimizer,
    #                         model_unet,
    #                         writer_train, writer_valid, criterion_unet, configs.checkpointpath, configs.bestmodelckp, configs.model)
    #         else:
    #             train.train(configs.epochs, training_loader, validation_loader, valid_loss_min, optimizer_unet, model_unet,
    #                         writer_train,
    #                         writer_valid, criterion_unet, configs.checkpointpath, configs.bestmodelckp, configs.model)
    #     except MemoryError as error:
    #         print('memory error')
    # else:
    #     training_set_down, validation_set_down, test_set_down, training_set_original, validation_set_original, test_set_original = srcnn_subject_loading.subject_loading(
    #         configs.data, configs.groundtruth,
    #         configs.downsampled, configs.inputsize)
    #     test_set = zip(test_set_down, test_set_original)
    #     ##refer to subject_loading.py
    #
    #     training_loader_down, validation_loader_down, test_loader_down = patch_loading.patch_Loading_unet(patch_size,
    #                                                                                                       samples_per_volume,
    #                                                                                                       max_queue_length,
    #                                                                                                       training_set_down,
    #                                                                                                       validation_set_down,
    #                                                                                                       test_set_down,
    #                                                                                                       configs.patch_check)
    #     training_loader_original, validation_loader_original, test_loader_original = patch_loading.patch_Loading_unet(
    #         patch_size_orig, 8,
    #         200, training_set_original,
    #         validation_set_original, test_set_original,
    #         configs.patch_check)
    #     training_loader = zip(training_loader_down, training_loader_original)
    #     training_loader = list(training_loader)
    #
    #     validation_loader = zip(validation_loader_down, validation_loader_original)
    #     validation_loader = list(validation_loader)
    #     test_loader = zip(test_loader_down, test_loader_original)
    #     test_loader = list(test_loader)
    #     print('SRCNN')
    #     try:
    #         if os.path.exists(fname):
    #             print('file found')
    #             model_srcnn, optimizer, start_epoch, valid_loss_min = interpolate.load_ckp(configs.checkpointpath, model_srcnn,
    #                                                                                        optimizer_srcnn)
    #             print(start_epoch)
    #             train.train(configs.epochs - start_epoch, training_loader_down, validation_loader_down, valid_loss_min,
    #                         optimizer, model_srcnn,
    #                         writer_train, writer_valid, criterion_SRCNN, configs.checkpointpath, configs.bestmodelckp, configs.model)
    #         else:
    #             train.train(configs.epochs, training_loader, validation_loader, valid_loss_min, optimizer_srcnn, model_srcnn,
    #                         writer_train,
    #                         writer_valid, criterion_SRCNN, configs.checkpointpath, configs.bestmodelckp, configs.model)
    #     except MemoryError as error:
    #         print('memory error')



    # if configs.model == 'SRCNN':
    #     model, optimizer, start_epoch, valid_loss_min = interpolate.load_ckp(configs.bestmodelckp, model_srcnn,
    #                                                                          optimizer_srcnn)
    # else:
    #     model, optimizer, start_epoch, valid_loss_min = interpolate.load_ckp(configs.bestmodelckp, model_unet,
    #                                                                          optimizer_unet)

    model.eval()
    with torch.no_grad():
        output, inputs,labels, loss, loss_list = test.test(test_loader, configs.model, model, criterion, configs.mode) #Ask Raghava
        N = 1

        # Indices of N largest elements in list
        # using sorted() + lambda + list slicing
        res = sorted(range(len(loss_list)), key=lambda sub: loss_list[sub])[:N]
        print(res)
        batch = str(res[0])
        # evaluation
        image_list = []
        counter = 0
        for image, gt, input in zip(output[res[0]], labels[res[0]], inputs[res[0]]):
            print(image.size())
            image = torch.squeeze(image, dim=0)
            #image_test.append(image)
            ground_truth = torch.squeeze(gt, dim= 0)
            input = torch.squeeze(input, dim= 0)
            img = nib.Nifti1Image(image.cpu().numpy(), np.eye(4))
            ground_truth_img = nib.Nifti1Image(ground_truth.cpu().numpy(), np.eye(4))
            inp_img = nib.Nifti1Image(input.cpu().numpy(), np.eye(4))

            print(img.shape)


                #dest = source + 'test' + batch + '-' + str(counter)
            nib.save(img, os.path.join(dest, 'test' + batch + '-' + str(counter)+ '.nii.gz' ))
            #if os.path.exists(configs.test_gt_path):
            nib.save(ground_truth_img, os.path.join(dest, 'gt' + batch + '-' + str(counter)+'.nii.gz'))
            #if os.path.exists(configs.test_inp_path):

            nib.save(inp_img,os.path.join(dest, 'inp' + batch + '-' + str(counter) +'.nii.gz' ))
            plotting.plot_img(img, cmap='gray')
            plotting.show()
            counter += 1


    #         image = torchio.Image(tensor=image)
    #         subject_dict = {'test': image}
    #         subject = torchio.Subject(subject_dict)
    #         image_list.append(subject)
    #         counter += 1
    #     grid_sampler = torchio.inference.GridSampler(
    #         image_list[-1],
    #         patch_size,
    #         patch_overlap=(2, 2, 2)
    #     )
    #     patch_loader = torch.utils.data.DataLoader(
    #         grid_sampler, batch_size=validation_batch_size)
    #     aggregator = torchio.inference.GridAggregator(grid_sampler)
    #     for patches_batch in patch_loader:
    #         inputs = patches_batch['test'][DATA].to(device)
    #         locations = patches_batch[torchio.LOCATION]
    #         if configs.model == 'UNET':
    #             output = model(inputs)
    #         else:
    #             output1, output = model(inputs)
    #         # labels = logits.argmax(dim=CHANNELS_DIMENSION, keepdim=True)
    #         aggregator.add_batch(output, locations)
    #
    #     # outputs = output.cpu().numpy()
    #     # outputs1 = np.asarray(outputs)
    #     # labels = logits.argmax(dim=CHANNELS_DIMENSION, keepdim=True)
    #     aggregator.add_batch(output, locations)
    #
    # aggregator = aggregator.get_output_tensor()
    # aggregator = torch.squeeze(aggregator)
    #
    # nii_super_res = nib.Nifti1Image(aggregator.numpy(), np.eye(4))
    # #nii_input = nib.Nifti1Image(input_tensor.numpy(), np.eye(4))
    # #nii_ground_truth = nib.Nifti1Image(ground_truth.numpy(), np.eye(4))
    # #nib.save(nii_super_res, configs.test_superres_path)
    # #nib.save(nii_ground_truth, configs.test_gt_path)
    # #nib.save(nii_input, configs.test_inp_path)
    # # nib.save(nii_super_res, os.path.join(configs.test_superres_path, 'super_res.nii.gz'))
    # plotting.plot_img(nii_super_res, cmap='gray')
    # plotting.show()

