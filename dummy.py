import torch.utils.data.dataloader as td
import torch
import torchio
import numpy as np
import random
from torchio import DATA

rand_list = random.sample(range(0, 6780), 5930)
print(len(rand_list[:5097]))
print(len(rand_list[5097:]))
print(6780 - 850)
subjects = []
tensor = torch.randn((20,20,20))
for i in range(10):
    sub = {'org': torchio.Image(tensor=tensor)}
    subj = torchio.Subject(sub)
    subjects.append(subj)
subjs = torchio.ImagesDataset(subjects)
data = td.DataLoader(subjs)
for i,datas in enumerate(data):
    print(type(datas['org'][DATA][0]))
    print(i)

