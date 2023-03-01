"""
sigmoid image post processing

Editor: Marshall Xu
Last Edited: 03/01/2023
"""

import numpy as np
import nibabel as nib
import cc3d
import matplotlib.pyplot as plt

import config

def thresholding(arr,thresh):
    arr[arr<thresh] = 0
    arr[arr>thresh] = 1
    return arr

def post_processing_pipeline(arr, thresh, connect_threshold):
    # thresholding
    arr = thresholding(arr, thresh)
    # morphologies (currently disabled)
    # arr = scind.binary_dilation(arr, structure)
    # arr = scind.binary_erosion(arr, structure).astype(np.int8)
    # connected components
    return cc3d.dust(arr, connect_threshold, connectivity=26, in_place=False)

if __name__ == "__main__":
    args = config.args
    image_path = args.outim_path
    image_name = args.outim
    sav_img_name = args.img_name
    path = image_path+image_name

    sig_img = nib.load(path)
    header = sig_img.header
    affine = sig_img.affine

    sig_arr = sig_img.get_fdata()
    processed_img = post_processing_pipeline(sig_arr, thresh=0.3, connect_threshold=120)
    mip = np.max(processed_img, axis=2) # create maximum intensity projection
    nifimg = nib.Nifti1Image(processed_img, affine, header)
    # saved the processed image as nifti file
    # save the MIP image
    sav_img_path = image_path+sav_img_name+".nii.gz"
    sav_mip_img_path = image_path+sav_img_name+".png"
    plt.imsave(sav_mip_img_path, mip, cmap='gray')
    print("Output MIP image is successfully saved!\n")
    # save the nii image
    nib.save(nifimg, sav_img_path)
    print("Output Neuroimage is successfully saved!\n")

    # command line:
    # python postprocessing.py --outim_path "./saved_image/week11/" --outim "finetuning_unet_ep5000_bce_15.nii.gz" --img_name "postpro_15_2"
    # Note: the thresholding value 'thresh', and connceted components threshold value 'connect_thresh' needs to be manually tuned
    # Note2: currently it is hard thresholding, could be updated later
