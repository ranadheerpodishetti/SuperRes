
import torch
import torchio
import torchvision
from torch.utils.data import Dataset, DataLoader
#import interpolate
from torchvision.utils import make_grid, save_image
from torchio import DATA

import nibabel as nib
import numpy as np

def patch_Loading_unet(patchsize, samples, maxqueue, training_set, validation_set, patch_check_path, main_training_batch_size, main_validation_batch_size):
    training_batch_size = main_training_batch_size
    validation_batch_size = main_validation_batch_size
    #test_batch_size = main_test_batch_size
    patch_size = tuple(patchsize)  # Use Patch Size same for all the models (use 8 16 16)
    samples_per_volume = samples  # 5
    max_queue_length = maxqueue  # 300

    patches_training_set = torchio.Queue(
        subjects_dataset=training_set,
        max_length=max_queue_length,
        samples_per_volume=samples_per_volume,
        sampler=torchio.sampler.UniformSampler(patch_size),
        shuffle_subjects=False,
        shuffle_patches=False,
        num_workers = 0
    )

    patches_validation_set = torchio.Queue(
        subjects_dataset=validation_set,
        max_length=max_queue_length,
        samples_per_volume=samples_per_volume,
        sampler=torchio.sampler.UniformSampler(patch_size),
        shuffle_subjects=False,
        shuffle_patches=False,
        num_workers= 0
    )

    training_loader = torch.utils.data.DataLoader(
        patches_training_set, batch_size=training_batch_size, num_workers=0)

    validation_loader = torch.utils.data.DataLoader(
        patches_validation_set, batch_size=validation_batch_size, num_workers=0)
    # test_loader = torch.utils.data.DataLoader(
    #     test_set, batch_size= test_batch_size)
    #one_batch = next(iter(training_loader))
    #k = 5
    #batch_mri = one_batch['down'][DATA][..., k]
    #batch_label = one_batch['original'][DATA][..., k]
    #slices = torch.cat((batch_mri, batch_label))
    #image_path = patch_check_path
    #save_image(slices, image_path, nrow=training_batch_size)
    # one_batch = training_loader.dataset[0]
    # k = 15
    # if variable=='Under':
    #     batch_mri = one_batch['down'][DATA].squeeze()
    #     print(batch_mri.size())
    #     batch_mri1 = batch_mri[15:16, :,: ]
    #     print(batch_mri1.size())
    #     image_path1 = 'E:/master/sem2/superres/execution/patch_down.jpg'
    #     slices = batch_mri1
    #     save_image(slices, image_path1)
    #     slices = batch_mri.numpy()
    #     down = nib.Nifti1Image(slices, np.eye(4))
    #     nib.save(down,'E:/master/sem2/superres/execution/patch_down.nii.gz')
    #
    # else :
    #     batch_label = one_batch['original'][DATA].squeeze()
    #     batch_label1 = batch_label[30:31, :, :]
    #     slices=batch_label1
    # #slices = torch.cat((batch_mri, batch_label))
    #     image_path = patch_check_path
    #     save_image(slices, image_path, nrow=training_batch_size)
    #     slices = batch_label.numpy()
    #     down = nib.Nifti1Image(slices, np.eye(4))
    #     nib.save(down,'E:/master/sem2/superres/execution/patch_orig.nii.gz')
    return training_loader, validation_loader
    # display.Image(image_path)
