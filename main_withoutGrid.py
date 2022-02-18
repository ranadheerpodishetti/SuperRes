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
import matplotlib as plt
from scipy import stats
from torchvision.utils import make_grid, save_image
from nilearn import plotting
import pandas as pd
import argparse
import apex.amp as amp
from torch.utils.tensorboard import SummaryWriter
import os.path
import time, datetime
import SRCNN
import UNet
import interpolate
import patch_loading
import subject_loading
import srcnn_subject_loading
import train
import test

manualSeed = 999
#manualSeed = random.randint(1, 10000) # use if you want new results
#print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)



prsr = argparse.ArgumentParser()
prsr.add_argument('--model', required=True, type=str)
prsr.add_argument('--result', required= True)
prsr.add_argument('--trainlogs', required=True)  # tensorboard logs directory args.logs
prsr.add_argument('--validlogs', required=True)  # tensorboard logs directory args.logs
prsr.add_argument('--data', required=True)  # csv path args.data
prsr.add_argument('--original_path', required=True)  # visualizing args.original_path
prsr.add_argument('--down_path', required=True)  # visualizing args.down_path
prsr.add_argument('--groundtruth', required=True)  # groundtruth img path args.groundtruth
prsr.add_argument('--downsampled', required=True)  # downsampled  img path args.downsampled
prsr.add_argument('--inputsize', required=True)  # no of input images according to ram capacity args.inputsize
prsr.add_argument('--samples', required=True, type=int)  # samples per epoch args.samples
prsr.add_argument('--patchsize', required=True, nargs='+', type=int)  # patch_size args.patchsize
prsr.add_argument('--patchsize_original', required=True, nargs='+', type=int)
prsr.add_argument('--maxqueue', required=True, type=int)  # max queue length args.maxqueue
prsr.add_argument('--patch_check', required=True)  # patches images path args.patch_check
prsr.add_argument('--learningrate', required=True, type=float)  # learning rate args.learningrate
prsr.add_argument('--epochs', required=True, type=int)  # no.of epochs args.epochs
prsr.add_argument('--checkpointpath', required=True)  # chkpt path args.checkpointpath
prsr.add_argument('--bestmodelckp', required=True)  # best model path args.bestmodelckp
prsr.add_argument('--testimage', required=True)  # test image superres path args.testimage
prsr.add_argument('--test_superres_path', required=True)  # test image superres path args.test_superres_path
prsr.add_argument('--test_gt_path', required=True)  # test image gt path args.test_gt_path
prsr.add_argument('--test_inp_path', required=True)  # test image input path args.test_inp_path
args = prsr.parse_args()




now = datetime.datetime.today()

time = datetime.datetime.now()

current_time = time.strftime("-%H-%M")
print("Current Time =", current_time)

nTime = now.strftime("-%d-%m-%Y") + current_time
source = args.result
dest = os.path.join(source+nTime)
if not os.path.exists(dest):
    os.makedirs(dest)
    os.makedirs(dest + '/valid_logs')
    os.makedirs(dest + '/train_logs')


writer_train = SummaryWriter(log_dir=args.trainlogs)
writer_valid = SummaryWriter(log_dir=args.validlogs)
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

    training_batch_size = 32
    validation_batch_size = 16

    patch_size = tuple(args.patchsize)  # 32, 64, 64
    patch_size_orig = tuple(args.patchsize_original)
    samples_per_volume = args.samples
    # 5

    max_queue_length = args.maxqueue  # 300

    device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'

    ##refer to patch_loading.py
    # display.Image(image_path)
    fname = args.checkpointpath

    # forward pass
    model_unet = UNet.UNet()
    model_unet.to('cuda')
    model_srcnn = SRCNN.SRCNN_late_upscaling()
    model_srcnn.to('cuda')

    criterion_unet = nn.MSELoss(reduction='mean')
    criterion_SRCNN = nn.L1Loss(reduction='mean')

    optimizer_unet = optim.Adam(model_unet.parameters(), lr=args.learningrate)
    opt_level = 'O1'
    model_unet, optimizer_unet = amp.initialize(model_unet, optimizer_unet, opt_level=opt_level)
    valid_loss_min = np.inf
    optimizer_srcnn = optim.Adam(model_srcnn.parameters(), lr=args.learningrate)
    opt_level = 'O1'
    model_srcnn, optimizer_srcnn = amp.initialize(model_srcnn, optimizer_srcnn, opt_level=opt_level)
    if args.model == 'UNET':
        training_set, validation_set, test_set = subject_loading.subject_loading(args.data, args.groundtruth,
                                                                                 args.downsampled, args.inputsize, args.original_path, args.down_path)
        ##refer to subject_loading.py

        training_loader, validation_loader, test_loader = patch_loading.patch_Loading_unet(patch_size,
                                                                                           samples_per_volume,
                                                                                           max_queue_length,
                                                                                           training_set,
                                                                                           validation_set, test_set,
                                                                                           args.patch_check)
        # print((training_loader))
        print('UNET')
        try:
            if os.path.exists(fname):
                print('file found')
                model_unet, optimizer, start_epoch, valid_loss_min = interpolate.load_ckp(args.checkpointpath, model_unet,
                                                                                          optimizer_unet)
                print(start_epoch)
                train.train(args.epochs - start_epoch, training_loader, validation_loader, valid_loss_min, optimizer,
                            model_unet,
                            writer_train, writer_valid, criterion_unet, args.checkpointpath, args.bestmodelckp, args.model)
            else:
                train.train(args.epochs, training_loader, validation_loader, valid_loss_min, optimizer_unet, model_unet,
                            writer_train,
                            writer_valid, criterion_unet, args.checkpointpath, args.bestmodelckp, args.model)
        except MemoryError as error:
            print('memory error')
    else:
        training_set_down, validation_set_down, test_set_down, training_set_original, validation_set_original, test_set_original = srcnn_subject_loading.subject_loading(
            args.data, args.groundtruth,
            args.downsampled, args.inputsize)
        test_set = zip(test_set_down, test_set_original)
        ##refer to subject_loading.py

        training_loader_down, validation_loader_down, test_loader_down = patch_loading.patch_Loading_unet(patch_size,
                                                                                                          samples_per_volume,
                                                                                                          max_queue_length,
                                                                                                          training_set_down,
                                                                                                          validation_set_down,
                                                                                                          test_set_down,
                                                                                                          args.patch_check)
        training_loader_original, validation_loader_original, test_loader_original = patch_loading.patch_Loading_unet(
            patch_size_orig, 8,
            200, training_set_original,
            validation_set_original, test_set_original,
            args.patch_check)
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
                model_srcnn, optimizer, start_epoch, valid_loss_min = interpolate.load_ckp(args.checkpointpath, model_srcnn,
                                                                                           optimizer_srcnn)
                print(start_epoch)
                train.train(args.epochs - start_epoch, training_loader_down, validation_loader_down, valid_loss_min,
                            optimizer, model_srcnn,
                            writer_train, writer_valid, criterion_SRCNN, args.checkpointpath, args.bestmodelckp, args.model)
            else:
                train.train(args.epochs, training_loader, validation_loader, valid_loss_min, optimizer_srcnn, model_srcnn,
                            writer_train,
                            writer_valid, criterion_SRCNN, args.checkpointpath, args.bestmodelckp, args.model)
        except MemoryError as error:
            print('memory error')

    if args.model == 'SRCNN':
        model, optimizer, start_epoch, valid_loss_min = interpolate.load_ckp(args.bestmodelckp, model_srcnn,
                                                                             optimizer_srcnn)
    else:
        model, optimizer, start_epoch, valid_loss_min = interpolate.load_ckp(args.bestmodelckp, model_unet,
                                                                             optimizer_unet)

    model.eval()
    with torch.no_grad():
        output, labels, inputs, loss, loss_list = test.test(test_loader, args.model, model, criterion_SRCNN)
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


            #    counter += 1
                #dest = source + 'test' + batch + '-' + str(counter)
            nib.save(img, os.path.join(dest, 'test' + batch + '-' + str(counter)+ '.nii.gz' ))
            #if os.path.exists(args.test_gt_path):
            #    counter += 1
            nib.save(ground_truth_img, os.path.join(dest, 'gt' + batch + '-' + str(counter)+'.nii.gz'))
            #if os.path.exists(args.test_inp_path):
            #    counter += 1
            nib.save(inp_img,os.path.join(dest, 'inp' + batch + '-' + str(counter) +'.nii.gz' ))
            plotting.plot_img(img, cmap='gray')
            plotting.show()
            counter += 1


