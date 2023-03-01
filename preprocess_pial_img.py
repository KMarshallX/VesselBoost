"""
Reshaping the label image of pial, and denoise the bfc pial images
Last edited: 03/01/2023 
"""

import numpy as np
import ants
import nibabel as nib
import os
from tqdm import tqdm
import scipy.ndimage as scind

if __name__ == "__main__":
    
    raw_path = "./data/pial_bfc/"
    seg_path = "./data/pial_label/"
    
    raw_file_list = os.listdir(raw_path)
    seg_file_list = os.listdir(seg_path)
    assert (len(raw_file_list) == len(seg_file_list)), "Number of images and correspinding segs not matched!"
    file_num = len(raw_file_list)
    raw_file_list.sort(key=lambda x:int(x[:2]))
    seg_file_list.sort(key=lambda x:int(x[:2]))

    for i in tqdm(range(file_num)):

        test_data_path = raw_path+raw_file_list[i]
        test_label_path = seg_path+seg_file_list[i]

        test_img = nib.load(test_data_path)
        header = test_img.header
        affine = test_img.affine
        test_data = test_img.get_fdata()

        test_label_nif = nib.load(test_label_path)
        test_label = test_label_nif.get_fdata()
        label_header = test_label_nif.header

        # Reshape the label file to make sure the label and the image are of smae dimensions
        if (test_data.shape[0] == test_label.shape[0]) and (test_data.shape[1] == test_label.shape[1]) and (test_data.shape[2] == test_label.shape[2]):
            continue
        else:
            test_label = scind.zoom(test_label, (test_data.shape[0]/test_label.shape[0],test_data.shape[1]/test_label.shape[1],test_data.shape[2]/test_label.shape[2]), order=0, mode='nearest')


        ant_img = ants.utils.convert_nibabel.from_nibabel(test_img)
        ant_msk = ants.utils.get_mask(ant_img, low_thresh=ant_img.min(), high_thresh=ant_img.max())

        ant_img_denoised = ants.utils.denoise_image(image=ant_img, mask=ant_msk)

        bfc_denoised_arr = ant_img_denoised.numpy()
        bfc_denoised_nifti = nib.Nifti1Image(bfc_denoised_arr, affine, header)

        reshaped_label_nifti = nib.Nifti1Image(test_label, affine, label_header)

        file_name = raw_path+raw_file_list[i]
        file_name_label = seg_path+seg_file_list[i]

        nib.save(bfc_denoised_nifti, filename=file_name)
        print(f"Image {raw_file_list[i]} successfully saved!")

        nib.save(reshaped_label_nifti, filename=file_name_label)
        print(f"Label Image {seg_file_list[i]} successfully saved!")