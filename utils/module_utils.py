"""
Provides all the utilities used in eval.py

Last edited: 07/04/2023

"""

import os
import numpy as np
import ants
import torch
import nibabel as nib
from tqdm import tqdm
import scipy.ndimage as scind
import matplotlib.pyplot as plt
from patchify import patchify, unpatchify
import cc3d

from utils.unet_utils import *
from models.unet_3d import Unet

class preprocess:
    """
    This object takes an input path and an output path to initialize
    """

    def __init__(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path

    def __call__(self, prep_mode):
        
        if prep_mode != 4:
            print("The preprocessing procedure is starting!\n")
            
            # bias field correction and denoising procedure
            if os.path.exists(self.output_path)==False:
                os.mkdir(self.output_path)

            raw_file_list = os.listdir(self.input_path)
            file_num = len(raw_file_list)
            for i in tqdm(range(file_num)):

                test_data_path = os.path.join(self.input_path, raw_file_list[i])

                test_img = nib.load(test_data_path)
                header = test_img.header
                affine = test_img.affine

                ant_img = ants.utils.convert_nibabel.from_nibabel(test_img)
                ant_msk = ants.utils.get_mask(ant_img, low_thresh=ant_img.min(), high_thresh=ant_img.max())

                if prep_mode == 1:
                    # bias field correction only
                    ant_img = ants.utils.n4_bias_field_correction(image=ant_img, mask=ant_msk)
                elif prep_mode == 2:
                    # non-local denoising only
                    ant_img = ants.utils.denoise_image(image=ant_img, mask=ant_msk)
                else:
                    # bfc + denoising
                    ant_img = ants.utils.n4_bias_field_correction(image=ant_img, mask=ant_msk)
                    ant_img = ants.utils.denoise_image(image=ant_img, mask=ant_msk)

                bfc_denoised_arr = ant_img.numpy()
                bfc_denoised_nifti = nib.Nifti1Image(bfc_denoised_arr, affine, header)

                file_name = os.path.join(self.output_path, raw_file_list[i])
                nib.save(bfc_denoised_nifti, filename=file_name)
            
            print("All processed images are successfully saved!")
        
        elif prep_mode == 4:
            print("Aborting the preprocessing procedure!\n")


class testAndPostprocess:
    """
    This opbject takes only one image and process only one image
    """
    def __init__(self, model_name, input_channel, output_channel, filter_number, input_path, output_path):
        
        self.mo = model_name
        self.ic = input_channel
        self.oc = output_channel
        self.fil = filter_number

        self.input_path = input_path # preprocessed data
        self.output_path = output_path # output proxy / final seg

    def standardiser(self, x):
        # only campatible with dtype = numpy array
        return (x - np.mean(x)) / np.std(x)
    
    def sigmoid(self, z):
        return 1/(1+np.exp(-z))
    
    def make_prediction(self, test_patches, load_model, ori_size):
        print("Prediction procedure starts!")
        # Predict each 3D patch  
        for i in tqdm(range(test_patches.shape[0])):
            for j in range(test_patches.shape[1]):
                for k in range(test_patches.shape[2]):

                    single_patch = test_patches[i,j,k, :,:,:]
                    single_patch_input = single_patch[None, :]
                    single_patch_input = torch.from_numpy(single_patch_input).type(torch.FloatTensor).unsqueeze(0)

                    single_patch_prediction = load_model(single_patch_input)

                    single_patch_prediction_out = single_patch_prediction.detach().numpy()[0,0,:,:,:]

                    test_patches[i,j,k, :,:,:] = single_patch_prediction_out

        test_output = unpatchify(test_patches, (ori_size[0], ori_size[1], ori_size[2]))
        test_output_sigmoid = self.sigmoid(test_output)
        print("Prediction procedure ends! Please wait for the post processing!")
        return test_output_sigmoid
    
    def post_processing_pipeline(self, arr, percent, connect_threshold):
        """
        thresh: thresholding value converting the probability to 0 and 1, anything below thresh are 0s, above are 1s.
        connected_threshold
        connect_threshold: any component smaller than this value (voxel) will be wiped out.
        """
        def thresholding(arr, thresh):
            arr[arr<thresh] = 0
            arr[arr>thresh] = 1
            return arr.astype(int)
        # thresholding
        arr = thresholding(arr, percent)
        # connected components
        return cc3d.dust(arr, connect_threshold, connectivity=26, in_place=False)
    
    def one_img_process(self, img_name, load_model, thresh, connect_thresh, mip_flag):
        # Load data
        raw_img_path = os.path.join(self.input_path, img_name) # should be full path

        raw_img = nib.load(raw_img_path)
        header = raw_img.header
        affine = raw_img.affine
        raw_arr = raw_img.get_fdata() # (1080*1280*52), (480, 640, 163)

        ori_size = raw_arr.shape    # record the original size of the input image slab
        # resize the input image, to make sure it can be cropped into small patches with size of (64,64,64)
        if (ori_size[0] // 64 != 0) and (ori_size[0] > 64):
            w = int(np.ceil(ori_size[0]/64)) * 64 # new width (x)
        else:
            w = ori_size[0]
        if (ori_size[1] // 64 != 0) and (ori_size[1] > 64):
            h = int(np.ceil(ori_size[1]/64)) * 64 # new height (y)
        else:
            h = ori_size[1]
        if (ori_size[2] // 64 != 0) and (ori_size[2] > 64):
            t = int(np.ceil(ori_size[2]/64)) * 64 # new thickness (z)
        elif ori_size[2] < 64:
            t = 64
        else:
            t = ori_size[2]
        new_raw = scind.zoom(raw_arr, (w/ori_size[0], h/ori_size[1], t/ori_size[2]), order=0, mode='nearest')
        
        # Standardization
        new_raw = self.standardiser(new_raw)
        new_size = new_raw.shape       # new size of the reshaped input image

        # pachify
        test_patches = patchify(new_raw, (64,64,64), 64)
        test_output_sigmoid = self.make_prediction(test_patches, load_model, new_size)

        # reshape to original shape
        test_output_sigmoid = scind.zoom(test_output_sigmoid, (ori_size[0]/new_size[0], ori_size[1]/new_size[1], ori_size[2]/new_size[2]), order=0, mode="nearest")

        # thresholding
        postprocessed_output = self.post_processing_pipeline(test_output_sigmoid, thresh, connect_thresh)
        nifimg_post = nib.Nifti1Image(postprocessed_output, affine, header)
        save_img_path_post = os.path.join(self.output_path, img_name) #img_name with extension

        # save the nifti file
        nib.save(nifimg_post, save_img_path_post)
        print(f"Output processed {img_name} is successfully saved!\n")

        # If mip_flag is true, then
        # save the maximum intensity projection as jpg file
        if mip_flag == True:
            mip = np.max(postprocessed_output, axis=2)
            save_mip_path_post = os.path.join(self.output_path, img_name.split('.')[0], ".jpg")
            plt.imsave(save_mip_path_post, mip, cmap='gray')
            print(f"Output MIP image {img_name} is successfully saved!\n")
    
    def __call__(self, thresh, connect_thresh, test_model_name, test_img_name, mip_flag):

        # model configuration
        load_model = Unet(self.ic, self.oc, self.fil)
        model_path = test_model_name # this should be the path to the model
        if torch.cuda.is_available() == True:
            print("Running with GPU")
            load_model.load_state_dict(torch.load(model_path))
        else:
            print("Running with CPU")
            load_model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        load_model.eval()

        self.one_img_process(test_img_name, load_model, thresh, connect_thresh, mip_flag)
        print("Prediction and thresholding procedure end!\n")



