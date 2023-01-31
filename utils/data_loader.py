"""
data loader function

Editor: Marshall Xu
Last Edited: 12/31/2022
"""

import nibabel as nib
import numpy as np
from patchify import patchify
import torch
from torch.utils.data import Dataset
import os

from utils.unet_utils import aug_utils

def standardiser(x):
    # only campatible with dtype = numpy array
    return (x - np.mean(x)) / np.std(x)

def _data_loader(raw_img, seg_img, patch_size, step):

    """
    :param raw_img: str, path of the raw file
    :param seg_img: str, path of the label file
    :param patch_size: tuple, e.g. (64,64,64)
    :param step: if step >= patch_size then no overlap between patches 
    """
    
    raw_arr = nib.load(raw_img).get_fdata() # (1090*1280*52)
    seg_arr = nib.load(seg_img).get_fdata()

    # # Cropping (using Saskia's data as template), make em the same size
    # raw_arr = raw_arr[64:1024, 64:1216, :]
    # seg_arr = seg_arr[64:1024, 64:1216, :]

    # # Standardisation
    # raw_arr = standardiser(raw_arr)

    # Patchify the loaded data
    raw_patches = patchify(raw_arr, patch_size, step)
    seg_patches = patchify(seg_arr, patch_size, step)

    assert (raw_patches.shape[0] == seg_patches.shape[0]), "Patches dimension not equal, 0"
    assert (raw_patches.shape[1] == seg_patches.shape[1]), "Patches dimension not equal, 1"
    assert (raw_patches.shape[2] == seg_patches.shape[2]), "Patches dimension not equal, 2"

    input_imgs = np.reshape(raw_patches, (-1, raw_patches.shape[3], raw_patches.shape[4], raw_patches.shape[5]))
    input_msks = np.reshape(seg_patches, (-1, seg_patches.shape[3], seg_patches.shape[4], seg_patches.shape[5]))

    return input_imgs, input_msks 

def data_concatenator(raw_path, seg_path, patch_size, step):
    """
    :param raw_path: str, "../data/raw/"
    :param seg_img: str, "../data/seg/"
    :param patch_size: tuple, e.g. (128,128,52)
    :param step: if step >= patch_size then no overlap between patches 
    :param out_size: tuple, e.g. (64,64,64)
    :param batch_size: int

    """    
    raw_file_list = os.listdir(raw_path)
    seg_file_list = os.listdir(seg_path)

    if raw_path == "./data/train/" or raw_path == "./data/train_4/":
        raw_file_list.sort(key=lambda x:int(x[:-7]))
        seg_file_list.sort(key=lambda x:int(x[:-4]))
    else:
        raw_file_list.sort(key=lambda x:int(x[:-7]))
        seg_file_list.sort(key=lambda x:int(x[4:-7]))

    assert (len(raw_file_list) == len(seg_file_list)), "Number of images and correspinding segs not matched!"
    file_num = len(raw_file_list)
    raw_img_0 = raw_path+raw_file_list[0]
    seg_img_0 = seg_path+seg_file_list[0]

    img_0, msk_0 = _data_loader(raw_img_0, seg_img_0, patch_size, step)

    for idx in range(1, file_num):
        next_raw_img = raw_path+raw_file_list[idx]
        next_seg_img = seg_path+seg_file_list[idx]

        next_img, next_msk = _data_loader(next_raw_img, next_seg_img, patch_size, step)

        img_0 = np.concatenate((img_0, next_img), axis=0)
        msk_0 = np.concatenate((msk_0, next_msk), axis=0)
    
    return standardiser(img_0), msk_0

class data_loader(Dataset):
    """
    :param raw_path: str, "../data/raw/"
    :param seg_img: str, "../data/seg/"
    :param patch_size: tuple, e.g. (128,128,52)
    :param step: if step >= patch_size then no overlap between patches 
    :param out_size: tuple, e.g. (64,64,64)
    :param batch_size: int

    """
    def __init__(self, raw_path, seg_path, patch_size, out_size, step):

        self.raw_path = raw_path
        self.seg_path = seg_path
        self.patch_size = patch_size
        self.step = step
        self.out_size = out_size

        self.con_img, self.con_msk = data_concatenator(self.raw_path, self.seg_path, self.patch_size, self.step)


    def __len__(self):
        assert (self.con_img.shape[0] == self.con_msk.shape[0]), "Image data and label data size not matched!"
        return self.con_img.shape[0]

    def __getitem__(self, idx):

        img = self.con_img[idx, :, :, :]
        msk = self.con_msk[idx, :, :, :]

        # aug_item = aug_utils(self.out_size)

        # if len(img.shape) == 3:
        #     aug_img, aug_msk = aug_item(img, msk)

        #     tensor_img = torch.from_numpy(aug_img.copy()).to(torch.float32)
        #     tensor_msk = torch.from_numpy(aug_msk.copy()).to(torch.float32)
        
        #     return tensor_img.unsqueeze(0), tensor_msk.unsqueeze(0)

        # elif len(img.shape) == 4:
        #     aug_img = np.zeros((img.shape[0], self.out_size[0], self.out_size[1], self.out_size[2]))
        #     aug_msk = np.zeros((msk.shape[0], self.out_size[0], self.out_size[1], self.out_size[2]))
        #     for i in range(img.shape[0]):
        #         aug_img[i, :, :, :], aug_msk[i, :, :, :] = aug_item(img[i, :, :, :], msk[i, :, :, :])

        #     tensor_img = torch.from_numpy(aug_img[:,None,:,:,:].copy()).to(torch.float32)
        #     tensor_msk = torch.from_numpy(aug_msk[:,None,:,:,:].copy()).to(torch.float32)
            
        return img, msk





    