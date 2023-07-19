"""
new data loader functions

Editor: Marshall Xu
Last Edited: 07/07/2023
"""

import nibabel as nib
import os

from .unet_utils import RandomCrop3D, standardiser

class single_channel_loader:
    def __init__(self, raw_img, seg_img, patch_size, step):
        """
        :param raw_img: str, path of the raw file
        :param seg_img: str, path of the label file
        :param patch_size: tuple, e.g. (64,64,64)
        :param step: how many (integer) patches will be cropped from the raw input image 
        """
        raw_nifti = nib.load(raw_img)
        raw_numpy = raw_nifti.get_fdata()
        self.raw_arr = standardiser(raw_numpy) # (1090*1280*52), (480, 640, 163)
        self.seg_arr = nib.load(seg_img).get_fdata()

        self.raw_img = raw_img
        self.seg_img = seg_img
        
        self.patch_size = patch_size
        self.step = step

    def __repr__(self):
        return f"Processing image {self.raw_img} and its segmentation {self.seg_img}\n"

    def __len__(self):
        return self.step
        
    def __iter__(self):
        # Save the raw image size
        raw_size = self.raw_arr.shape
        seg_size = self.seg_arr.shape
        assert (raw_size[0] == seg_size[0]), "Input image and segmentation dimension not matched, 0"
        assert (raw_size[1] == seg_size[1]), "Input image and segmentation dimension not matched, 1"
        assert (raw_size[2] == seg_size[2]), "Input image and segmentation dimension not matched, 2"

        for i in range(self.step):
            cropper = RandomCrop3D(raw_size, self.patch_size)
            img_crop, seg_crop = cropper(self.raw_arr, self.seg_arr)

            yield img_crop, seg_crop
