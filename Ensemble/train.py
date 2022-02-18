import torch
import nibabel as nib
import dipy
# import visdom
# from visdom import Visdom
from dipy.io.image import load_nifti
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchio
from torchio import ImagesDataset, Image, Subject
import torch.nn.functional as F
import torchio
from skimage import io, transform
import numpy as np
from torchvision.utils import make_grid, save_image
from torchio.transforms import RandomAffine
from torchio.transforms.interpolation import Interpolation
from torchvision import transforms, models
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

import modified_unet
# import interpolate
import patch_loading
import subject_loading
import checkpoints
import SRCNN
import SRDenseNet

torch.set_default_dtype(torch.float32)



device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'

loss_ratio = [0.5, 0.33, 0.17]
def train(epochs, training_loader, validation_loader, valid_loss_min, optimizer, model, writer_train, writer_valid,
          writer_images, criterion,criterion_perc,criterion_novel, checkpointpath, bestmodelckp, Model, mode, loss_func, scaler):
    train_loss = []
    valid_loss = []

    for epoch in range(epochs):

        running_loss = 0.0
        batch_size = 0

        for i, data in enumerate(training_loader, 0):
            batch_size += 1
            with autocast():
                if Model == 'UNET':
                    inputs = data['down'][DATA].to(device)
                    labels = data['original'][DATA].to(device)

                    if mode != 'DS':
                        print('normal')
                        output, _, _ = model(inputs)
                        if loss_func == 'L1':
                            print('l1')
                            loss = criterion(output, labels)
                        elif loss_func == 'SSIM':
                            print('ssim')
                            loss = 1 - criterion(output.type(torch.cuda.FloatTensor), labels)
                        elif loss_func  == 'Perceptual':
                            loss = criterion(output, labels)
                        else:
                            loss_l1 = criterion(output, labels)
                            loss_perc = criterion_perc(output, labels)
                            loss_msssim = 1 - criterion_novel(output.type(torch.cuda.FloatTensor), labels)
                            print('loss_size = ', loss_l1.size())
                            gaussian = torch.normal(1, 0.5, loss_l1.shape).to(device)
                            loss_l1 = torch.mean(torch.mul(loss_l1, gaussian)).to(device)
                            loss = (loss_l1*0.16 + loss_msssim*0.64 + loss_perc*0.2)/3.0
                            print(type(loss))
                    else:
                        print('DS')
                        output, x_3, x_4 = model(inputs)
                        # x_3 = F.interpolate(x_3, size = (32, 32, 32), mode= 'trilinear')
                        # x_4 = F.interpolate(x_4, size = (32, 32, 32), mode= 'trilinear')
                        print(output.size(), x_4.size(), x_3.size())
                        if loss_func == 'L1':
                            loss_out = criterion(output, labels)
                            loss_x3 = criterion(x_3, labels)
                            loss_x4 = criterion(x_4, labels)
                        elif loss_func == 'SSIM':
                            print((output.type()))
                            print((labels.size()))
                            loss_out = 1.0 - criterion(output.type(torch.cuda.FloatTensor), labels)
                            loss_x3 = 1.0 - criterion(x_3.type(torch.cuda.FloatTensor), labels)
                            loss_x4 = 1.0 - criterion(x_4.type(torch.cuda.FloatTensor), labels)
                        else:
                            print(1)
                            loss_out = criterion(output, labels)
                            loss_x3 = criterion(x_3, labels)
                            loss_x4 = criterion(x_4, labels)
                        loss = loss_ratio[0]*loss_out+loss_ratio[-1]*loss_x3+loss_ratio[1]*loss_x4
                        print(type(loss))
                    input_tensor = inputs[0]
                    ground_tensor = labels[0]
                    tensor_vis = output[0]

                elif Model == "SRDENSENET":
                    #from dictionary read inputs and GT
                    inputs = data[0]['down'][DATA].to(device)
                    labels = data[1]['original'][DATA].to(device)
                    # TB visualization with 1st img
                    input_tensor = inputs[0]
                    ground_tensor = labels[0]
                    #op returned after concat and dynconv
                    #the model name is specified in main.py
                    output  = model(input)
                    lossl1_dense = criterion(output1, labels)
                    loss_msssim_dense = criterion_novel(output1, labels)
                    loss = lossl1_dense + loss_msssim_dense

## passed new arguments model1 and model2 please change it in main file as well
#model1 is srdensenet
#model2 is srcnn
#model is final ENSEMBLE model

                else:
                    inputs = data[0]['down'][DATA].to(device)
                    labels = data[1]['original'][DATA].to(device)
                    input_tensor = inputs[0]
                    ground_tensor = labels[0]
                    output1, output2 = model(inputs)
                    tensor_vis = output2[0]
                    loss1 = criterion(output1, labels)
                    loss2 = criterion(output2, labels)
                    loss = loss2 + loss1


            # 64x64x28 and 64x64x28

            optimizer.zero_grad()

            # comapring 128 label with 64 patch

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            #optimizer.step()

            running_loss += loss.item()

            print('[%d,%5d] loss: %3f' % (epoch + 1, i + 1, running_loss / batch_size))

            writer_train.add_scalar('training loss',
                                    running_loss / batch_size,
                                    epoch * len(
                                        training_loader) + i)  # tensorboard --logdir D:\^flirt\logs\  --host localhost  --port=6006 use this in pycharm terminal.....
            #writer_train.close()
            train_loss.append(running_loss / batch_size)
            # plt.plot(train_loss, 'r', label = 'train')
            # plt.legend()

            running_loss = 0.0

        model.eval()
        with torch.no_grad():
            losses = 0.0
            val_batch = 0

            for l, data in enumerate(validation_loader, 0):
                val_batch += 1

                if Model == 'UNET':
                    input_val = data['down'][DATA].to(device)
                    labels_val = data['original'][DATA].to(device)

                    if mode != 'DS':
                        print('normal')
                        output, _, _ = model(input_val)
                        if loss_func == 'L1':
                            print('l1')
                            loss = criterion(output, labels_val)
                        elif loss_func == 'SSIM':
                            print('ssim')
                            loss = 1 - criterion(output.type(torch.cuda.FloatTensor), labels_val)
                        elif loss_func == 'Perceptual':
                            loss = criterion(output, labels_val)
                        else:
                            loss = criterion(output, labels_val)
                            print('loss_size = ', loss.size())
                            gaussian = torch.normal(1, 0.5, loss.shape).to(device)
                            loss = torch.mean(torch.mul(loss, gaussian)).to(device)
                            print(type(loss))
                    else:
                        print('DS')
                        output, x_3, x_4 = model(input_val)
                        # x_3 = F.interpolate(x_3, size=(32, 32, 32), mode='trilinear')
                        # x_4 = F.interpolate(x_4, size=(32, 32, 32), mode='trilinear')
                        if loss_func == 'L1':
                            loss_out = criterion(output, labels_val)
                            loss_x3 = criterion(x_3, labels_val)
                            loss_x4 = criterion(x_4, labels_val)
                            loss = loss_ratio[0] * loss_out + loss_ratio[-1] * loss_x3 + loss_ratio[1] * loss_x4
                        elif loss_func == 'SSIM':
                            loss_out = 1 - criterion(output.type(torch.cuda.FloatTensor), labels_val)
                            loss_x3 = 1 - criterion(x_3.type(torch.cuda.FloatTensor), labels_val)
                            loss_x4 = 1 - criterion(x_4.type(torch.cuda.FloatTensor), labels_val)
                            loss = loss_ratio[0] * loss_out + loss_ratio[-1] * loss_x3 + loss_ratio[1] * loss_x4
                        elif loss_func == 'Perceptual' :
                            loss_out = criterion(output, labels_val)
                            loss_x3 = criterion(x_3, labels_val)
                            loss_x4 = criterion(x_4, labels_val)
                            loss = loss_ratio[0] * loss_out + loss_ratio[-1] * loss_x3 + loss_ratio[1] * loss_x4
                        else:
                            loss_out = criterion(output, labels_val)
                            loss_x3 = criterion(x_3, labels_val)
                            loss_x4 = criterion(x_4, labels_val)
                            loss = loss_ratio[0] * loss_out + loss_ratio[-1] * loss_x3 + loss_ratio[1] * loss_x4
                            gaussian = torch.normal(1, 0.5, (loss.size())).to(device)
                            loss = torch.mean(torch.mul(loss, gaussian)).to(device)

                else:
                    input_val = data[0]['down'][DATA].to(device)
                    labels_val = data[1]['original'][DATA].to(device)
                    output1, output2 = model(input_val)
                    #_, tensor_vis = model(tensor)

                    loss1 = criterion(output1, labels_val)
                    loss2 = criterion(output2, labels_val)
                    loss = loss2 + loss1

                losses += loss.item()

                print(losses / val_batch)
                writer_train.add_scalar('validation loss',
                                        losses / batch_size,
                                        epoch * len(
                                            validation_loader) + l)  # tensorboard --logdir D:\^flirt\logs\  --host localhost  --port=6006 use this in pycharm terminal.....

                valid_loss.append(losses / val_batch)
            tensor_vis = tensor_vis.squeeze().cpu()
            tensor_vis = tensor_vis[15:16, :, :]
            input_tensor = input_tensor.squeeze().cpu()
            ground_tensor = ground_tensor.squeeze().cpu()
            if Model == 'UNET':
                img_grid = make_grid([input_tensor[15:16,:,:], ground_tensor[15:16,:,:], tensor_vis])
                writer_images.add_image('images', img_grid, global_step = epoch)
            else:
                img_grid = make_grid([ground_tensor, tensor_vis])
                writer_images.add_image('images', img_grid, global_step = epoch)

        checkpoint = {
            'epoch': epoch + 1,
            'valid_loss_min': losses,
            'model': model,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'AMPScaler': scaler.state_dict()
        }

        # save checkpoint
        checkpoints.save_ckp(checkpoint, False, checkpointpath, bestmodelckp)

        if losses <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min, losses))
            # save checkpoint as best model
            checkpoints.save_ckp(checkpoint, True, checkpointpath, bestmodelckp)
            valid_loss_min = losses
    writer_train.close()

    writer_valid.close()
    return model, valid_loss, train_loss
