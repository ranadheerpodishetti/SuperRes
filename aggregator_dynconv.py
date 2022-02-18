import torch
import nibabel as nib
import dipy
from dipy.io.image import load_nifti
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchio
from torch.autograd import Variable
from torchio import SubjectsDataset, Image, Subject
import torch.nn.functional as F
from skimage import io, transform
import numpy as np
from torchvision.utils import make_grid, save_image
from torchio.transforms import RandomAffine
from torchio.transforms.interpolation import Interpolation
from torchvision import transforms
import os
import torch.optim as optim
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
import os.path
#import subject_loading
import checkpoints
import utils
import test_config_dynconv
from test_config_dynconv import Config

configs = Config()
device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
data = configs.dataset
data_path = configs.data
gt = configs.groundtruth
down = configs.downsampled
model = configs.model.UNet(64, 3, 1, 2, 2, 1)
model.to(device)

#opt_level = 'O1'
optimizer = optim.Adam(model.parameters(), lr= 0.0001)
#model, optimizer = amp.initialize(model, optimizer, opt_level=opt_level)
scaler = GradScaler()

# k = torch.load('E:/master/sem2/superres/execution/best_unet_l1.pt')
# model.load_state_dict(k['state_dict'])
if data == 'IXI':
    print('IXI')
    path_original, path_downsampled = utils.list(gt, data_path)
    print(path_original)
    print(path_downsampled)
    dataset = pd.read_csv(data_path, sep=',', header=None)
else:
    path_downsampled, path_original = utils.list_mudi(downsampled_path, data_path)
    dataset = pd.read_csv(data_path, sep=',', header=None)
test_name = []
for i in range(len(path_original)):
    test_names = path_original[i].split('.', 1)
    test_name.append(test_names[0]+'_test.'+test_names[-1])
print(test_name)
print(dataset.head())
# path_original = listFiles(groundtruth_path)  # pass the path for groundtruth images
# print(path_original)

# path_downsampled = listFiles(downsampled_path)  # pass the path for downsampled images

subjects = []
for image in range(len(path_original)):  ## put int(inputsize) while executing on local machine
    # and len(path_original) while executing on cluster
    # img_original = torchio.Image(tensor=interpolate.original(path_original[image], y, dimensions))##pass the path to original interpolated

    # img_down = torchio.Image(tensor=interpolate.down(path_downsampled[image],  x, y, dimensions))## pass the path to downsampled interpolated
    # subject_dict = {'original': img_original, 'down': img_down}
    subject_dict = {
        'original': torchio.Image(os.path.join(gt, path_original[image]), type=torchio.LABEL),
        'down': torchio.Image(os.path.join(down, path_downsampled[image]), type=torchio.INTENSITY)
    }
    subject = torchio.Subject(subject_dict)
    subjects.append(subject)

test_set = torchio.SubjectsDataset(subjects)
valid_loss_min = np.inf
k = 0
def grid_aggregator(path, test_set, patch_size, overlap, model, optimizer, scaler, num_workers, test_path, gt_path, inp_path, k):
    model, optimizer, start_epoch, valid_loss_min, scaler = checkpoints.load_ckp(path, model, optimizer, scaler)
    #summary(model, (1, 32, 32, 32))

    for i, test_set in enumerate(test_set, 0):

        grid_sampler = torchio.inference.GridSampler(test_set, patch_size, overlap)
        patch_loader = torch.utils.data.DataLoader(grid_sampler, batch_size= 1, shuffle= False, num_workers= num_workers)
        input = test_set['down'][DATA].to(device).squeeze()
        gt = test_set['original'][DATA].to(device).squeeze()
        aggregator = torchio.inference.GridAggregator(grid_sampler)
        model.eval()
        with torch.no_grad():
            for p, patches_batch in enumerate(patch_loader):
                #images = Variable(patches_batch['down'][DATA]).to(device).permute(0, 1, 4, 2, 3)
                inputs = patches_batch['down'][DATA].to(device)
                print(type(inputs))
                locations = patches_batch[torchio.LOCATION]
                output, _ , _ = model(inputs)
                print(len(output))
                aggregator.add_batch(output, locations)

    # outputs = output.cpu().numpy()
    # outputs1 = np.asarray(outputs)
    # labels = logits.argmax(dim=CHANNELS_DIMENSION, keepdim=True)

        aggregator = aggregator.get_output_tensor()
        aggregator = torch.squeeze(aggregator)

        nii_super_res = nib.Nifti1Image(aggregator.numpy(), np.eye(4))
        nii_input = nib.Nifti1Image(input.detach().cpu().numpy(), np.eye(4))
        nii_ground_truth = nib.Nifti1Image(gt.detach().cpu().numpy(), np.eye(4))
        nib.save(nii_super_res, os.path.join(configs.path, test_path[k]))
        nib.save(nii_ground_truth, os.path.join(configs.gt, gt_path[k]))
        nib.save(nii_input, os.path.join(configs.inp, inp_path[k]))
        # nib.save(nii_super_res, os.path.join(args.test_superres_path, 'super_res.nii.gz'))
        plotting.plot_img(nii_super_res, cmap='gray')
        plotting.show()
        k +=1

#
#
# model = modified_unet_1.UNet(4, 3, 1, 2, 2, 1)
# model.to(device)
#
# criterion = nn.MSELoss(reduction='mean')
#
# optimizer = optim.Adam(model.parameters(), lr= 0.0001)
# opt_level = 'O1'
# model, optimizer = amp.initialize(model, optimizer, opt_level=opt_level)

if os.path.exists(configs.path):
    grid_aggregator(configs.checkpointpath, test_set, configs.patchsize, configs.patch_overlap, model, optimizer, scaler, configs.num_workers, test_name, path_original, path_downsampled, k)
else:
    os.mkdir(configs.path)
    grid_aggregator(configs.checkpointpath, test_set, configs.patchsize, configs.patch_overlap, model, optimizer, scaler,
                    configs.num_workers, test_name, path_original, path_downsampled, k)

