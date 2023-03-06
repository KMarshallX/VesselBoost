"""
Preprocess the input image slabs (bias field correction and denoise(noise_model='Rician'))

Marshall @ 03/01/2023
"""

import numpy as np
import ants
import nibabel as nib
import os
from tqdm import tqdm

if __name__ == "__main__":
    
    raw_path = "./data/validate/"
    processed_data_path = "./data/validate_bfcdn/"
    if os.path.exists(processed_data_path)==False:
        os.mkdir(processed_data_path)
    
    raw_file_list = os.listdir(raw_path)
    file_num = len(raw_file_list)
    raw_file_list.sort(key=lambda x:int(x[:2]))

    for i in tqdm(range(file_num)):

        test_data_path = raw_path+raw_file_list[i]

        test_img = nib.load(test_data_path)
        header = test_img.header
        affine = test_img.affine
        test_data = test_img.get_fdata()

        ant_img = ants.utils.convert_nibabel.from_nibabel(test_img)
        ant_msk = ants.utils.get_mask(ant_img, low_thresh=ant_img.min(), high_thresh=ant_img.max())

        ant_img_bfc = ants.utils.n4_bias_field_correction(image=ant_img, mask=ant_msk)
        ant_img_denoised = ants.utils.denoise_image(image=ant_img_bfc, mask=ant_msk)

        bfc_denoised_arr = ant_img_denoised.numpy()
        bfc_denoised_nifti = nib.Nifti1Image(bfc_denoised_arr, affine, header)

        file_name = processed_data_path+raw_file_list[i]
        nib.save(bfc_denoised_nifti, filename=file_name)
        print(f"Image {raw_file_list[i]} successfully saved!")