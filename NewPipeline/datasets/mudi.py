import h5py as h5
import numpy as np

import torch
from torch.utils.data import Dataset
from torchio.data.subject import Subject

from datasets.torchiowrap import H5DSImage

class MUDISet(Dataset):
    def __init__(self, subset='Train', task=1, n_scans=1344, interp=True, cropini_even=True, data_path='MOOD_train.h5', torchiosub=True, lazypatch=True, preload=False, transform=None, returnWOtrans=False):
        self.h5 = h5.File(data_path, 'r', swmr=True)
        
        target_key = subset + '-data-target'
        if task == 1:
            taskstr = 'one'
            source_key = subset + '-data-source-task-one'
            data_ext = '-source-ani'
            self.crop_downfact = (1,1,2)
        else:
            taskstr = 'two'
            source_key = subset + '-data-source-task-two'
            data_ext = '-source-iso'
            self.crop_downfact = (2,2,2)
        if interp:
            self.crop_downfact = (1,1,1)

        if subset == 'ChallangeTest':
            if interp:
                source_key = 'source-task-' + taskstr + '-interp'
            else:
                source_key = 'source-task-' + taskstr
            self.samples=[self.h5[source_key][key] for key in self.h5[source_key].keys()]
            self.subs = list(self.h5[source_key].keys())
        else:            
            self.samples=[(self.h5[source_key][key.split('-')[0]+data_ext], self.h5[target_key][key]) for key in self.h5[target_key].keys()]
            self.subs = [key.split('-')[0] for key in self.h5[target_key].keys()]

        if preload:
            print('Preloading MUDISet: '+subset)
            for i in range(len(self.samples)):
                if type(self.samples[i]) is tuple:
                    self.samples[i] = (self.samples[i][0][:], self.samples[i][1][:])
                else:
                    self.samples[i] = self.samples[i][:]

        self.subset = subset
        self.n_scans = n_scans
        self.cropini_even = cropini_even
        self.torchiosub = torchiosub
        self.lazypatch = lazypatch
        self.transform = transform #currently only applicable to torchio Subjects
        self.returnWOtrans = returnWOtrans

    def __len__(self):
        return len(self.samples) * self.n_scans

    def minmax(self, data):
        return (data - data.min()) / (data.max() - data.min())

    def __getitem__(self, idx):
        if idx >= len(self): raise IndexError

        item, scan = np.unravel_index(idx, (len(self.samples), self.n_scans))
        # print(self.subs[item])
        if type(self.samples[item]) is tuple:
            im = self.minmax(self.samples[item][0][...,scan])
            gt = self.minmax(self.samples[item][1][...,scan])
            if self.torchiosub:
                sub = Subject({'img':H5DSImage(im, lazypatch=self.lazypatch, cropini_even=self.cropini_even, 
                                                crop_downfact=self.crop_downfact),
                                'gt':H5DSImage(gt, lazypatch=self.lazypatch, cropini_even=self.cropini_even)})
                if self.transform:
                    if self.returnWOtrans:
                        return self.transform(sub), sub #Tansform is causing re-initialization of the Image class. Not good. TODO
                    else:
                        return self.transform(sub) #Tansform is causing re-initialization of the Image class. Not good. TODO
                else:
                    return sub
            else:
                return (torch.from_numpy(im).unsqueeze(0), torch.from_numpy(gt).unsqueeze(0))
        else:
            im = self.minmax(self.samples[item][...,scan])
            if self.torchiosub:
                sub = Subject({'img':H5DSImage(im, lazypatch=self.lazypatch, cropini_even=self.cropini_even, 
                                                crop_downfact=self.crop_downfact)})
                if self.transform:
                    if self.returnWOtrans:
                        return self.transform(sub), sub 
                    else:
                        return self.transform(sub) 
                else:
                    return sub
            else:
                return torch.from_numpy(im).unsqueeze(0)

