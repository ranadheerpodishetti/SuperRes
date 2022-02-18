import torch
import torchio
from torchio import ImagesDataset, Image, Subject
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
# import interpolate
import os
import random
import utils

manualSeed = 999
random.seed(manualSeed)


# training_split_ratio = 0.75
# validation_split_ratio = 0.15


def listFiles(root):
    allFiles = []
    for folder, folders, files in os.walk(root):
        for file in files: allFiles += [folder.replace("\\", "/") + "/" + file]
    return allFiles


def subject_loading(data_path, groundtruth_path, downsampled_path, inputsize, main_training_split_ratio,
                    main_validation_split_ratio, data):
    training_split_ratio = main_training_split_ratio  # 0.75
    validation_split_ratio = main_validation_split_ratio  # 0.15
    if data == 'IXI':
        path_original, path_downsampled = utils.list(groundtruth_path, data_path)
        dataset = pd.read_csv(data_path, sep=',', header=None)
    else:
        path_original, path_downsampled = utils.list_mudi(groundtruth_path, data_path)
        dataset = pd.read_csv(data_path, sep=',', header=None)
    print(dataset.head())
    # path_original = listFiles(groundtruth_path)  # pass the path for groundtruth images

    # path_downsampled = listFiles(downsampled_path)  # pass the path for downsampled images

    subjects_down = []
    subjects_original = []
    for image in range(len(path_original)):  ## put int(inputsize) while executing on local machine
        # and len(path_original) while executing on cluster
        ##img_original = torchio.Image(tensor=interpolate.original(path_original[image], y, dimensions_original))
        ##img_down = torchio.Image(tensor=interpolate.original(path_downsampled[image], k, dimensions_down))
        ##subject_down = {'down': img_down}
        ##subject_original = {'original': img_original}
        subject_down = {
            'down': torchio.Image(os.path.join(downsampled_path, path_downsampled[image]), type=torchio.INTENSITY)}
        subject_original = {
            'original': torchio.Image(os.path.join(groundtruth_path, path_original[image]), type=torchio.LABEL)}

        subject_down = torchio.Subject(subject_down)
        subject_original = torchio.Subject(subject_original)
        subjects_down.append(subject_down)
        subjects_original.append(subject_original)
        shuffle = list(zip(subjects_original, subjects_down))
        shuffled_data = random.sample(shuffle, len(shuffle))
        subjects_original, subjects_down = zip(*shuffled_data)
        subjects_original = list(subjects_original)
        subjects_down = list(subjects_down)
    dataset_down = torchio.ImagesDataset(subjects_down)
    dataset_original = torchio.ImagesDataset(subjects_original)
    print('Dataset size:', len(dataset_down), 'subjects')

    num_subjects = len(dataset_down)
    num_training_subjects = int(training_split_ratio * num_subjects)
    num_validation_subjects = int((training_split_ratio + validation_split_ratio) * num_subjects)

    training_subjects_down = subjects_down[:num_training_subjects]
    training_set_down = torchio.ImagesDataset(training_subjects_down)
    # print('Training set:', len(training_set), 'subjects')
    validation_subjects_down = subjects_down[num_training_subjects: ]
    #test_subjects_down = subjects_down[num_validation_subjects:]
    # training_set = torchio.ImagesDataset(training_subjects)

    validation_set_down = torchio.ImagesDataset(validation_subjects_down)
    #test_set_down = torchio.ImagesDataset(test_subjects_down)

    training_subjects_original = subjects_original[:num_training_subjects]
    training_set_original = torchio.ImagesDataset(training_subjects_original)
    # print('Training set:', len(training_set), 'subjects')
    validation_subjects_original = subjects_original[num_training_subjects: num_validation_subjects]
    #test_subjects_original = subjects_original[num_validation_subjects:]
    # training_set = torchio.ImagesDataset(training_subjects)

    validation_set_original = torchio.ImagesDataset(validation_subjects_original)
    #test_set_original = torchio.ImagesDataset(test_subjects_original)

    # print('Training set:', len(training_set), 'subjects')
    # print('Validation set:', len(validation_set), 'subjects')
    # print('test set:', len(test_set), 'subjects')
    return training_set_down, validation_set_down, training_set_original, validation_set_original
