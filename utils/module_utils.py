"""
Provides all the utilities used in the three modules
(train.py, prediction.py, test_time_adaptation.py)

Last edited: 19/10/2023

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

from .unet_utils import *
from models import Unet

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
                affine = test_img.affine # type: ignore

                ant_img = ants.utils.convert_nibabel.from_nibabel(test_img)
                ant_msk = ants.utils.get_mask(ant_img, low_thresh=ant_img.min(), high_thresh=ant_img.max()) # type: ignore

                if prep_mode == 1:
                    # bias field correction only
                    ant_img = ants.utils.n4_bias_field_correction(image=ant_img, mask=ant_msk)
                elif prep_mode == 2:
                    # non-local denoising only
                    ant_img = ants.utils.denoise_image(image=ant_img, mask=ant_msk) # type: ignore
                else:
                    # bfc + denoising
                    ant_img = ants.utils.n4_bias_field_correction(image=ant_img, mask=ant_msk)
                    ant_img = ants.utils.denoise_image(image=ant_img, mask=ant_msk) # type: ignore

                bfc_denoised_arr = ant_img.numpy()
                bfc_denoised_nifti = nib.Nifti1Image(bfc_denoised_arr, affine, header)

                file_name = os.path.join(self.output_path, raw_file_list[i])
                nib.save(bfc_denoised_nifti, filename=file_name)
            
            print("All processed images are successfully saved!")
        
        elif prep_mode == 4:
            print("Aborting the preprocessing procedure!\n")


class prediction_and_postprocess:
    """
    A class that contains methods for standardizing, normalizing, and making predictions on 3D image patches using a given model.
    It also includes a post-processing pipeline for thresholding and connected component analysis.
    
    Attributes:
    - model_name (str): the type of the model used for prediction
    - input_channel (int): the number of input channels for the model
    - output_channel (int): the number of output channels for the model
    - filter_number (int): the number of filters used in the model
    - input_path (str): the path to the preprocessed data
    - output_path (str): the path to save the output proxy/final segmentation
    
    Methods:
    - standardiser(x): standardizes the input numpy array
    - normaliser(x): normalizes the input numpy array
    - sigmoid(z): applies the sigmoid function to the input numpy array
    - make_prediction(test_patches, load_model, ori_size): makes predictions on 3D image patches using the given model
    - post_processing_pipeline(arr, percent, connect_threshold): applies thresholding and connected component analysis to the input numpy array
    - one_img_process(img_name, load_model, thresh, connect_thresh, mip_flag): processes a single image using the given model and post-processing pipeline
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
    
    def normaliser(self, x):
        # only campatible with dtype = numpy array
        return (x - np.amin(x)) / (np.amax(x) - np.amin(x))
    
    def sigmoid(self, z):
        return 1/(1+np.exp(-z))
    
    def nn_sigmoid(self, z):
        sigmoid_fn = torch.nn.Sigmoid()
        return sigmoid_fn(z)
    
    def inference(self, test_patches, load_model, ori_size):
        print("Prediction procedure starts!")
        # Predict each 3D patch  
        for i in tqdm(range(test_patches.shape[0])):
            for j in range(test_patches.shape[1]):
                for k in range(test_patches.shape[2]):

                    single_patch = test_patches[i,j,k, :,:,:]
                    single_patch_input = single_patch[None, :]
                    single_patch_input = torch.from_numpy(single_patch_input).type(torch.FloatTensor).unsqueeze(0)

                    single_patch_prediction = self.nn_sigmoid(load_model(single_patch_input)) 

                    single_patch_prediction_out = single_patch_prediction.detach().numpy()[0,0,:,:,:]

                    test_patches[i,j,k, :,:,:] = single_patch_prediction_out

        test_output = unpatchify(test_patches, (ori_size[0], ori_size[1], ori_size[2]))

        print("Prediction procedure ends! Please wait for the post processing!")
        return test_output
    
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
        affine = raw_img.affine # type: ignore
        raw_arr = raw_img.get_fdata() # type: ignore # (1080*1280*52), (480, 640, 163)

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
        test_output_sigmoid = self.inference(test_patches, load_model, new_size)

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
            save_mip_path_post = os.path.join(self.output_path, img_name.split('.')[0]) + ".jpg"
            # save_mip_path_post = self.output_path + img_name.split('.')[0] + ".jpg"
            #rotate the mip 90 degrees, counterclockwise
            mip = np.rot90(mip, axes=(0, 1))
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

def preprocess_procedure(ds_path, ps_path, prep_mode):
    """
    Preprocesses data

    Args:
        ds_path (str): The path to the dataset.
        ps_path (str): The path to the preprocessed data storage location.
        prep_mode (int): The preprocessing mode to use.

    Returns:
        None
    """
    # initialize the preprocessing method with input/output paths
    preprocessing = preprocess(ds_path, ps_path)
    # start or abort preprocessing 
    preprocessing(prep_mode)

def make_prediction(model_name, input_channel, output_channel, 
                    filter_number, input_path, output_path, 
                    thresh, connect_thresh, test_model_name, 
                    mip_flag):
    """
    Makes a prediction

    Args:
        model_name (str): The name of the model.
        input_channel (int): The number of input channels.
        output_channel (int): The number of output channels.
        filter_number (int): The number of filters.
        input_path (str): The path to the input data (normally the path to processed data).
        output_path (str): The path to the output data.
        thresh (float): The threshold value.
        connect_thresh (int): The connected threshold value.
        test_model_name (str): The path to the pre-trained model.
        test_img_name (str): The name of the test image.
        mip_flag (bool): The MIP flag.

    Returns:
        None
    """
    # initialize the prediction method with model configuration and input/output paths
    prediction_postpo = prediction_and_postprocess(model_name, input_channel, output_channel, filter_number, input_path, output_path)
    # take each processed image for prediction
    processed_data_list = os.listdir(input_path)
    for i in range(len(processed_data_list)):
        # generate inferred segmentation fot the current image
        prediction_postpo(thresh, connect_thresh, test_model_name, processed_data_list[i], mip_flag)

