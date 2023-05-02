"""
new data loader functions

Editor: Marshall Xu
Last Edited: 03/01/2023
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
        
        self.patch_size = patch_size
        self.step = step

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

class new_data_loader:
    def __init__(self, raw_path, seg_path, patch_size, step):
        """
        :param raw_path: str, "../data/raw/"
        :param seg_img: str, "../data/seg/"
        :param patch_size: tuple, e.g. (64,64,64)
        :param step: how many (integer) patches will be cropped from the raw input image 
        """
        self.raw_path = raw_path
        self.seg_path = seg_path

        self.raw_file_list = os.listdir(self.raw_path)
        self.seg_file_list = os.listdir(self.seg_path)
        assert (len(self.raw_file_list) == len(self.seg_file_list)), "Number of images and correspinding segs not matched!"
        self.file_num = len(self.raw_file_list)

        self.patch_size = patch_size
        self.step = step

    def __len__(self):
        return self.step*self.file_num

    def __iter__(self):
        
        if self.raw_path == "./data/train/" or self.raw_path == "./data/train_4/":
            self.raw_file_list.sort(key=lambda x:int(x[:-7]))
            self.seg_file_list.sort(key=lambda x:int(x[:-4]))
        elif self.raw_path == "./data/train_bfc/":
            self.raw_file_list.sort(key=lambda x:int(x[4:-11]))
            self.seg_file_list.sort(key=lambda x:int(x[:-4]))
        else:
            self.raw_file_list.sort(key=lambda x:int(x[:-7]))
            self.seg_file_list.sort(key=lambda x:int(x[4:-7]))

        for idx in range(self.file_num):
            raw_arr_name = self.raw_path+self.raw_file_list[idx]
            seg_arr_name = self.seg_path+self.seg_file_list[idx]
            raw_arr = standardiser(nib.load(raw_arr_name).get_fdata()) # (1090*1280*52), (480, 640, 163)
            seg_arr = nib.load(seg_arr_name).get_fdata()
            
            # Save the raw image size
            raw_size = raw_arr.shape
            seg_size = seg_arr.shape
            assert (raw_size[0] == seg_size[0]), "Input image and segmentation dimension not matched, 0"
            assert (raw_size[1] == seg_size[1]), "Input image and segmentation dimension not matched, 1"
            assert (raw_size[2] == seg_size[2]), "Input image and segmentation dimension not matched, 2"

            for i in range(self.step):
                cropper = RandomCrop3D(raw_size, self.patch_size)
                img_crop, seg_crop = cropper(raw_arr, seg_arr)
                yield img_crop, seg_crop



