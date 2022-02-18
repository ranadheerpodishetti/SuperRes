from silx.io.convert import write_to_h5, convert
import tarfile
import h5py
import torch
import torchio
import nibabel  as nib
import numpy as np
import os
from scipy import signal
import scipy.fft as fft
import h5py


def listFiles3(root):
    allFiles = []
    for folder, folders, files in os.walk(root):
        for file in files: allFiles += [folder.replace("\\", "/") + "/" + file]
    return allFiles


k = listFiles3('E:/master/sem2/superres/downsampled')
name = os.listdir('E:/master/sem2/superres/downsampled')
my_tar = tarfile.open('E:/master/sem2/superres/dataset/IXI-DTI.tar', 'w')
comp_kwargs = {'compression': 'gzip', 'compression_opts': 1}
images_original = []


def interp_slc(img, target_size):
    ## from 2.5x2.5x5.0 to 2.5x2.5x2.5
    ## https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.resample.html
    ## resample uses FFT transformations, which can be very slow if the number of input or output samples is large and prime; see scipy.fft.fft.
    interp_ = np.zeros((img.shape[0], img.shape[1], target_size[2], img.shape[3]))

    for vv in range(0, img.shape[3]):
        for ii in range(0, img.shape[0]):
            for jj in range(0, img.shape[1]):
                interp_[ii, jj, :, vv] = signal.resample(img[ii, jj, :, vv], target_size[2])




def interp_3dim(img, target_size):
    ## from 5.0x5.0x5.0 to 2.5x2.5x2.5
    interp3_ = np.zeros((img.shape[0], img.shape[1], target_size[2]))
    interp2_ = np.zeros((target_size[0], img.shape[1], target_size[2]))
    interp1_ = np.zeros((target_size[0], target_size[1], target_size[2]))

        ## xy
    for ii in range(0, interp3_.shape[0]):
        for jj in range(0, interp3_.shape[1]):
            interp3_[ii, jj, :] = signal.resample(img[ii, jj, :], target_size[2])

    ## yz
    for ii in range(0, interp2_.shape[1]):
        for jj in range(0, interp2_.shape[2]):
            interp2_[:, ii, jj] = signal.resample(interp3_[:, ii, jj], target_size[0])

    ## xz
    for ii in range(0, interp1_.shape[0]):
        for jj in range(0, interp1_.shape[2]):
            interp1_[ii, :, jj] = signal.resample(interp2_[ii, :, jj], target_size[1])

    return interp1_


for i in range(len(k)):
    img_1 = nib.load(k[i])
    img1 = img_1.dataobj
    if os.path.exists(os.path.join('E:/master/sem2/superres/downsampled_interpolated', name[i])):
        print('there')
    else:
        img_interp = interp_3dim(img1, (img1.shape[0]*2,img1.shape[1]*2, img1.shape[-1]*2))
        img_new_interp = img_interp
        # if img_interp.shape[-1] != 64:
        #     img_new_interp = np.pad(img_interp, ((0,0), (0,0), (0,8)), mode = 'constant')
        print(img_interp.shape)
        img_new_interp = img_new_interp/np.max(img_new_interp)
        print(np.max(img_new_interp))
        nii = nib.Nifti1Image(img_new_interp, np.eye(4))
        nib.save(nii, os.path.join('E:/master/sem2/superres/downsampled_interpolated', name[i]))
    #img1 = signal.resample(img_1.dataobj, 32, axis=-1)

#     images_original.append(img1)
#
# res = np.concatenate([arr[np.newaxis] for arr in images_original])
# res = np.transpose(res, (1, 2, 3, 0))
# print(res.shape)
# images_original1 = np.array(images_original, dtype= np.float32)
# img_1 = nib.load('E:/master/sem2/supermudi/cdmri0011/MB_Re_t_moco_registered_applytopup_anisotropic_voxcor_cdmri0011_0.nii.gz')
# with h5py.File('E:/master/sem2/superres/train_filename.h5', 'w') as f:
#     for i, list in enumerate(images_original):
#         f.create_dataset(str(i), data=res)
#     # f.create_dataset('data', data=images_original)
# # f.create_dataset('d', data = np.expand_dims(img_1.dataobj, axis=-1))
#
# f = h5py.File('E:/master/sem2/superres/train_filename.h5', 'r')
# # print(list(f.keys()))
# print(type(f['0'][:]))
#
#
# def inter(f):
#     f1 = signal.resample(f[str(i)][:, :, :], 96, axis=0)
#     print(f1.shape)
#     f = signal.resample(f1, 96, axis=1)
#     print(f.shape)
#     f = signal.resample(f, 64, axis=2)
#     print(f.shape)
#     return f
#
#
# # print((np.expand_dims(f['1'][:], axis= -1)).shape)
# for i in range(10):
#     print(i)
#     k = inter(f)

# convert(my_tar,'E:/master/sem2/superres/my.h5')
