"""
new data loader functions

Editor: Marshall Xu
Last Edited: 07/07/2023
"""

import nibabel as nib
import os

from .unet_utils import RandomCrop3D, standardiser

class single_channel_loader:
    def __init__(self, raw_img, seg_img, patch_size, step, test_mode=False):
        """
        :param raw_img: str, path of the raw file
        :param seg_img: str, path of the label file
        :param patch_size: tuple, e.g. (64,64,64)
        :param step: how many (integer) patches will be cropped from the raw input image 
        """
        raw_nifti = nib.load(raw_img)
        raw_numpy = raw_nifti.get_fdata()
        # self.raw_arr = normaliser(raw_numpy) # (1090*1280*52), (480, 640, 163)
        self.raw_arr = standardiser(raw_numpy) # (1090*1280*52), (480, 640, 163)
        self.seg_arr = nib.load(seg_img).get_fdata()

        self.raw_img = raw_img
        self.seg_img = seg_img
        
        self.patch_size = patch_size
        self.step = step

        self.test_mode = test_mode

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
            cropper = RandomCrop3D(raw_size, self.patch_size, self.test_mode)
            img_crop, seg_crop = cropper(self.raw_arr, self.seg_arr)

            yield img_crop, seg_crop

def multi_channel_loader(ps_path, seg_path, patch_size, step, test_mode=False):
    """
    Loads multiple images and their corresponding segmentation masks from the given paths.
    Args:
        ps_path (str): Path to the folder containing the processed images.
        seg_path (str): Path to the folder containing the label images.
        patch_size (tuple): Size of the patches to be extracted from the images.
        step (int): Step size for the sliding window approach to extract patches.
    Returns:
        dict: A dictionary containing the initialized single_channel_loaders for each image.
    """
    # make sure the image path and seg path contains equal number of files
    raw_file_list = os.listdir(ps_path)
    seg_file_list = os.listdir(seg_path)
    assert (len(raw_file_list) == len(seg_file_list)), "Number of images and correspinding segs not matched!"
    file_num = len(raw_file_list)

    # initialize single_channel_loaders for each image
    # and store the initialized loaders in a linked hashmaps
    loaders_dict = dict()
    for i in range(file_num):
        # joined path to the current image file 
        raw_img_name = os.path.join(ps_path, raw_file_list[i])
        # find the corresponding seg file in the seg_folder
        seg_img_name = None
        for j in range(file_num):
            if seg_file_list[j].find(raw_file_list[i].split('.')[0]) != -1:
                seg_img_name = os.path.join(seg_path, seg_file_list[j])
                break
        assert (seg_img_name != None), f"There is no corresponding label to {raw_file_list[i]}!"
        # a linked hashmap to store the provoked data loaders
        loaders_dict.__setitem__(i, single_channel_loader(raw_img_name, seg_img_name, patch_size, step, test_mode))
    
    return loaders_dict