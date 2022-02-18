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
import torch.cuda.amp
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
import os.path

device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'


def test(test_loader,  Model, model, criterion):
    global inputs, labels, output2, loss
    loss_list = []
    output = []
    input =[]
    label = []
    for i, data in enumerate(test_loader, 0):

        if Model == 'UNET':
            inputs = data['down'][DATA].to(device)
            labels = data['original'][DATA].to(device)
            print('label_size', labels.size())
            output2 = model(inputs)
            output.append(output2)
            input.append(inputs)
            label.append(labels)
            loss = criterion(output2, labels)
            loss_list.append(loss)
            print('test loss = ', loss_list)
        else:
            inputs = data[0]['down'][DATA].to(device)
            labels = data[1]['original'][DATA].to(device)
            output1, output2 = model(inputs)
            output.append(output2)
            input.append(inputs)
            label.append(labels)
            loss1 = criterion(output1, labels)
            loss2 = criterion(output2, labels)
            loss = loss2 + loss1
            loss_list.append(loss)

    return output, input, label, loss.item(), loss_list

