"""
Provides all the utilities used in eval.py

Last edited: 03/27/2023

"""

import os
import numpy as np
import ants
import torch
import nibabel as nib
from tqdm import tqdm
import scipy.ndimage as scind
from patchify import patchify, unpatchify
import cc3d

from utils.unet_utils import *
from utils.new_data_loader import single_channel_loader
from models.unet_3d import Unet
from models.siyu import CustomSegmentationNetwork
from models.asppcnn import ASPPCNN
from models.ra_unet import MainArchitecture

def model_chosen(model_name, in_chan, out_chan, filter_num):
    if model_name == "unet3d":
        return Unet(in_chan, out_chan, filter_num)
    elif model_name == "aspp":
        return ASPPCNN(in_chan, out_chan, [1,2,3,5,7])
    elif model_name == "test":
        return CustomSegmentationNetwork()
    elif model_name == "atrous":
        return MainArchitecture()
    else:
        print("Insert a valid model name.") 

def optim_chosen(optim_name, model_params, lr):
    if optim_name == 'sgd':
        return torch.optim.SGD(model_params, lr)
    elif optim_name == 'adam':
        return torch.optim.Adam(model_params, lr)
    else:
        print("Insert a valid optimizer name.")


def loss_metric(metric_name):
    """
    :params metric_name: string, choose from the following: bce->binary cross entropy, dice->dice score 
    """
    # loss metric could be updated later -> split into 2 parts
    if metric_name == "bce":
        # binary cross entropy
        return BCELoss()
    elif metric_name == "dice":
        # dice loss
        return DiceLoss()
    elif metric_name == "tver":
        # tversky loss
        return TverskyLoss()
    else:
        print("Enter a valid loss metric.")


class preprocess:
    """
    This object takes an input path and an output path to initialize
    """

    def __init__(self, input_path, output_path) -> None:
        self.input_path = input_path
        self.output_path = output_path

    def __call__(self, prep_bool):
        
        if prep_bool == True:
            print("The preprocessing procedure is starting!\n")
            
            # bias field correction and denoising procedure
            if os.path.exists(self.output_path)==False:
                os.mkdir(self.output_path)

            raw_file_list = os.listdir(self.input_path)
            file_num = len(raw_file_list)
            for i in tqdm(range(file_num)):

                test_data_path = self.input_path + raw_file_list[i]

                test_img = nib.load(test_data_path)
                header = test_img.header
                affine = test_img.affine

                ant_img = ants.utils.convert_nibabel.from_nibabel(test_img)
                ant_msk = ants.utils.get_mask(ant_img, low_thresh=ant_img.min(), high_thresh=ant_img.max())

                ant_img_bfc = ants.utils.n4_bias_field_correction(image=ant_img, mask=ant_msk)
                ant_img_denoised = ants.utils.denoise_image(image=ant_img_bfc, mask=ant_msk)

                bfc_denoised_arr = ant_img_denoised.numpy()
                bfc_denoised_nifti = nib.Nifti1Image(bfc_denoised_arr, affine, header)

                file_name = self.output_path + raw_file_list[i]
                nib.save(bfc_denoised_nifti, filename=file_name)
            
            print("All processed images are successfully saved!")
        
        elif prep_bool == False:
            print("Aborting the preprocessing procedure!\n")


class testAndPostprocess:
    """
    This opbject takes only one image and process only one image
    """
    def __init__(self, model_name, input_channel, output_channel, filter_number, input_path, output_path) -> None:
        
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
                    single_patch_input = single_patch_input

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
    
    def one_img_process(self, img_name, load_model, thresh, connect_thresh):
        # Load data
        raw_img_path = self.input_path + img_name # should be full path

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

        # save as nifti image
        # reshape to original shape
        test_output_sigmoid = scind.zoom(test_output_sigmoid, (ori_size[0]/new_size[0], ori_size[1]/new_size[1], ori_size[2]/new_size[2]), order=0, mode="nearest")
        postprocessed_output = self.post_processing_pipeline(test_output_sigmoid, thresh, connect_thresh)
        nifimg = nib.Nifti1Image(postprocessed_output, affine, header)

        save_img_path = self.output_path + img_name
        nib.save(nifimg, save_img_path)
        print(f"Output processed {img_name} is successfully saved!\n")
    
    def __call__(self, thresh, connect_thresh, test_model_name, test_img_name):

        # model configuration
        load_model = model_chosen(self.mo, self.ic, self.oc, self.fil)
        model_path = "./saved_models/" + test_model_name
        load_model.load_state_dict(torch.load(model_path))
        load_model.eval()

        self.one_img_process(test_img_name, load_model, thresh, connect_thresh)
        print("Prediction and thresholding procedure end!\n")

class finetune:
    def __init__(self, data_path, proxy_path, model_type, input_channel, output_channel, filter_number, init_model_name) -> None:
        self.ds_path = data_path # processed data
        self.px_path = proxy_path # proxy seg
        self.out_mo_path = "./saved_models/"

        self.mo = model_type
        self.ic = input_channel
        self.oc = output_channel
        self.fil = filter_number
        self.init_mo_path = "./saved_models/" + init_model_name
        # initialize loss metric & optimizer
        self.loss_metric = loss_metric("tver")
        # initialize augmentation object
        self.aug_item = aug_utils((64,64,64), "mode1")
        
    def __call__(self, learning_rate, optim_gamma, optim_patience, epoch_num):
        print("Finetuing process starts!")
        # initialize pre-trained model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        load_model = model_chosen(self.mo, self.ic, self.oc, self.fil).to(device)
        load_model.load_state_dict(torch.load(self.init_mo_path))
        load_model = load_model.eval()
        # initialize optimizer & scheduler
        optimizer = optim_chosen('adam', load_model.parameters(), learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=optim_gamma, patience=optim_patience)
        
        processed_img_list = os.listdir(self.ds_path)
        file_num = len(processed_img_list)

        for idx in tqdm(range(file_num)):
            test_ds_path = self.ds_path + processed_img_list[idx]
            # find the corresponding proxy
            assert (processed_img_list[idx] in self.px_path), "No such proxy file!"
            test_px_path = self.px_path + processed_img_list[idx]
            file_name = processed_img_list[idx].split('.')[0]
            #initialize the data loader
            data_loader = single_channel_loader(test_ds_path, test_px_path, (64,64,64), epoch_num)

            # training loop
            for epoch in range(epoch_num):
                image, label = next(iter(data_loader))
                image_batch, label_batch = self.aug_item(image, label)
                image_batch, label_batch = image_batch.to(device), label_batch.to(device)

                optimizer.zero_grad()
            
                # Forward pass
                output = load_model(image_batch)
                loss = self.loss_metric(output, label_batch)

                # Backward and optimize
                loss.backward()
                optimizer.step()

                loss_mean = loss_mean + loss.item()
                # test feature, delete this before submission
                current_lr = optimizer.param_groups[0]['lr']

                # Learning rate shceduler
                scheduler.step(loss)
                print(f'Epoch: [{epoch+1}/{epoch_num}], Loss: {loss.item(): .4f}, Current learning rate: {current_lr: .6f}\n')
            
            out_mo_name = self.out_mo_path + file_name
            torch.save(load_model.state_dict(), out_mo_name)
            print(f"Training finished! The finetuning model of {file_name} successfully saved!")
        print("All finetuned models are saved!")

            
            




        


        

