import torch
import nibabel as nib
import torch.nn as nn
import torch.nn.functional
from torch.nn.modules.utils import _ntuple,_quadruple
from typing import  Tuple
from torch.nn.common_types  import _size_2_t, _size_4_t, _size_6_t

"""
train_in = nib.load('./data/train-4D-input.nii.gz').get_fdata()
print(train_in.shape)
train_mask = nib.load('./data/train-4D-mask.nii.gz').get_fdata()
train_in = train_in * train_mask
print('Hi')

"""



class ReflectionPad3d(torch.nn.modules.padding._ReflectionPadNd):
  padding: Tuple[int, int, int, int, int, int]

  def __init__(self, padding: _size_6_t) -> None:
    super(ReflectionPad3d, self).__init__()
    self.padding = _ntuple(6)(padding)

m = nn.ReflectionPad2d(2)
n = nn.ReplicationPad2d(2)
p = ReflectionPad3d(2)
input_Rep = torch.arange(9, dtype=torch.float).reshape(1, 1, 3, 3)
#input_Ref = torch.arange(9, dtype=torch.float).reshape(1, 1, 3, 3)
#input = torch.randn(1, 1, 3, 3, 3)
Custom = torch.nn.functional.pad(input_Rep, (0,1,0,1), 'reflect')
function_1 = m(input_Rep)
#print(input_Ref)
#print(input_Rep)
#print(m(input_Ref))
#print(n(input_Rep))
print(Custom)
print(function_1)
print('Debug Mode')








