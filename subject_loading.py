import torch
import torchio
from torchio import ImagesDataset, Image, Subject
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
import random
import nibabel as nib
import utils

manualSeed = 999
random.seed(manualSeed)


def listFiles(root):
    allFiles = []
    for folder, folders, files in os.walk(root):
        for file in files: allFiles += [folder.replace("\\", "/") + "/" + file]
    return allFiles


def subject_loading(data_path, groundtruth_path, downsampled_path, inputsize, original_path, down_path,
                    training_split_ratio, validation_split_ratio, data):
    if data == 'IXI':
        print('yes')
        path_original, path_downsampled = utils.list(groundtruth_path, data_path)
        dataset = pd.read_csv(data_path, sep=',', header=None)
    else:
        path_downsampled, path_original = utils.list_mudi(downsampled_path, data_path)
        dataset = pd.read_csv(data_path, sep=',', header=None)

    print(dataset.head())

    subjects = []
    for image in range(len(path_original)):  ## put int(inputsize) while executing on local machine
        # and len(path_original) while executing on cluster
        subject_dict = {
            'original': torchio.Image(os.path.join(groundtruth_path, path_original[image]), type=torchio.LABEL),
            'down': torchio.Image(os.path.join(downsampled_path, path_downsampled[image]), type=torchio.INTENSITY)
        }
        subject = torchio.Subject(subject_dict)
        # transform = torchio.CropOrPad((128, 128, 64))
        # transformed_subject = transform(subject)
        # subjects.append(transformed_subject)
        subjects.append(subject)

        print(image)
    dataset = torchio.ImagesDataset(subjects)
    sample = dataset[0]
    original = sample['original'].data.squeeze().numpy()
    down = sample['down'].data.squeeze().numpy()
    original = nib.Nifti1Image(original, np.eye(4))
    down = nib.Nifti1Image(down, np.eye(4))
    nib.save(original, original_path)
    nib.save(down, down_path)
    print(type(original), type(down))
    print('Dataset size:', len(dataset), 'subjects')
    num_subjects = len(dataset)
    num_training_subjects = 5097  ## 300 subjects (3 subjects has 16 directions)
    num_validation_subjects = 5930

    training_subjects = subjects[:num_training_subjects]
    training_subjects = random.sample(training_subjects, len(training_subjects))
    training_set = torchio.ImagesDataset(training_subjects)
    print('Training set:', len(training_set), 'subjects')
    validation_subjects = subjects[num_training_subjects:]
    validation_subjects = random.sample(validation_subjects, len(validation_subjects))

    validation_set = torchio.ImagesDataset(validation_subjects)
    print('Validation set:', len(validation_set), 'subjects')
    return training_set, validation_set
