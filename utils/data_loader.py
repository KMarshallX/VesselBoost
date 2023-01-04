"""
data loader function

Editor: Marshall Xu
Last Edited: 12/31/2022
"""

import nibabel as nib
import numpy as np
from patchify import patchify
import torch
import os

from utils.unet_utils import aug_utils

# def aug(img, msk, thickness):
#     """
#     :params img: input 3D image
#     :params msk: imput 3D mask
#     :params thickness: expected augmented thickness of the data (in z-direction)
#     """

#     diff = thickness - img.shape[2]
#     return np.concatenate((img, img[:,:,0:diff]), axis=2), np.concatenate((msk, msk[:,:,0:diff]), axis=2)

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

    # Cropping (using Saskia's data as template), make em the same size
    raw_arr = raw_arr[64:1024, 64:1216, :]
    seg_arr = seg_arr[64:1024, 64:1216, :]

    # Standardisation
    raw_arr = standardiser(raw_arr)

    # Patchify the loaded data
    raw_patches = patchify(raw_arr, patch_size, step)
    seg_patches = patchify(seg_arr, patch_size, step)

    assert (raw_patches.shape[0] == seg_patches.shape[0]), "Patches dimension not equal, 0"
    assert (raw_patches.shape[1] == seg_patches.shape[1]), "Patches dimension not equal, 1"
    assert (raw_patches.shape[2] == seg_patches.shape[2]), "Patches dimension not equal, 2"

    input_imgs = np.reshape(raw_patches, (-1, raw_patches.shape[3], raw_patches.shape[4], raw_patches.shape[5]))
    input_msks = np.reshape(seg_patches, (-1, seg_patches.shape[3], seg_patches.shape[4], seg_patches.shape[5]))

    return input_imgs, input_msks 

    
class data_loader:
    """
    :param raw_path: str, "../data/raw/"
    :param seg_img: str, "../data/seg/"
    :param patch_size: tuple, e.g. (128,128,52)
    :param step: if step >= patch_size then no overlap between patches 
    :param out_size: tuple, e.g. (64,64,64)
    """
    def __init__(self, raw_path, seg_path, patch_size, out_size, step):

        self.raw_path = raw_path
        self.seg_path = seg_path
        self.patch_size = patch_size
        self.step = step
        self.out_size = out_size

    def __iter__(self):

        raw_file_list = os.listdir(self.raw_path)
        seg_file_list = os.listdir(self.seg_path)

        assert (len(raw_file_list) == len(seg_file_list)), "Number of images and correspinding segs not matched!"
        file_num = len(raw_file_list)

        for idx in range(file_num):
            raw_img = self.raw_path+raw_file_list[idx]
            seg_img = self.seg_path+seg_file_list[idx]

            input_imgs, input_msks = _data_loader(raw_img, seg_img, self.patch_size, self.step)
            patches_num = input_imgs.shape[0]

            for i in range(patches_num):
                # filter out pure background
                if len(np.unique(input_imgs[i])) != 1:

                    aug_item = aug_utils(self.out_size)
                    img, msk = aug_item(input_imgs[i], input_msks[i])

                    img = torch.from_numpy(img.copy()).to(torch.float32)
                    msk = torch.from_numpy(msk.copy()).to(torch.float32)
                else:
                    continue

                yield img.unsqueeze(0), msk.unsqueeze(0)







    