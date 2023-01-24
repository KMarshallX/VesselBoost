"""
helper functions for 3D-Unet

Editor: Marshall Xu
Last Edited: 01/07/2022
"""

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import zoom
import nibabel as nib
from tqdm import tqdm
from patchify import patchify, unpatchify
import os

class DiceLoss(nn.Module):
    def __init__(self, smooth = 1e-4):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):

        batch_size = target.size(0)

        pred = torch.sigmoid(pred)
        # flatten pred and target tensors
        pred = pred.view(batch_size, -1).type(torch.FloatTensor)
        target = target.view(batch_size, -1).type(torch.FloatTensor)

        intersection =  (pred * target).sum(-1)
        dice = (2.*intersection + self.smooth) / ((pred * pred).sum(-1) + (target * target).sum(-1) + self.smooth)

        return torch.mean(1-dice)
    
class FocalLoss(nn.Module):
    def __init__(self, alpha = 0.8, gamma = 0, smooth = 1e-6):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)

        # Flatten
        pred = pred.reshape(-1)
        target = target.reshape(-1)

        # compute the binary cross-entropy
        bce = F.binary_cross_entropy(pred, target, reduction='mean')
        bce_exp = torch.exp(-bce)
        focal_loss = self.alpha * (1-bce_exp) ** self.gamma * bce

        return focal_loss

class BCELoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, pred, target):
        batch_size = target.size(0)

        # flatten pred and target tensors
        pred = pred.view(batch_size, -1)
        target = target.view(batch_size, -1)
        loss = nn.BCEWithLogitsLoss()
        return loss(pred, target)

class aug_utils:
    def __init__(self, size, mode):
        super().__init__()
        # size: expected size for the resampled data, tuple, e.g. (64,64,64)
        # mode: "off"-> augmentation is off; 
        #       "test"->test mode, no augmentation, but send one meaningful patch along with 5 other empty blocks;
        #       "on" -> augmentation is on.
        self.size = size
        self.mode = mode
    
    def rot(self, inp, k):
        # k: interger, Number of times the array is rotated by 90 degrees.

        return np.rot90(inp, k, axes=(0,1))

    def flip_hr(self, inp, k):
        # Filp horizontally, axis x -> axis z
        # k: interger, Number of times the array is rotated by 90 degrees.

        return np.rot90(inp, k, axes=(0,2))

    def flip_vt(self, inp, k):
        # Filp vertically, axis y -> axis z
        # k: interger, Number of times the array is rotated by 90 degrees.

        return np.rot90(inp, k, axes=(1,2))

    def zooming(self, inp):
        
        size = self.size
        assert (len(inp.shape)==3), "Only 3D data is accepted"
        w = inp.shape[0] # width
        h = inp.shape[1] # height
        d = inp.shape[2] # depth

        return zoom(inp, (size[0]/w, size[1]/h, size[2]/d), order=0, mode='nearest')

    def __call__(self, input, segin):
        input = self.zooming(input)
        segin = self.zooming(segin)

        if self.mode == "on":
            input_batch = np.stack((input, self.rot(input, 1), self.rot(input, 2), self.rot(input, 3), 
            self.flip_hr(input, 1), self.flip_vt(input, 1)), axis=0)
            segin_batch = np.stack((segin, self.rot(segin, 1), self.rot(segin, 2), self.rot(segin, 3), 
            self.flip_hr(segin, 1), self.flip_vt(segin, 1)), axis=0)
        elif self.mode == "off":
            input_batch = np.stack((input, input, input, input, input, input), axis=0)
            segin_batch = np.stack((segin, segin, segin, segin, segin, segin), axis=0)
        elif self.mode == "test":
            dummy = np.zeros((self.size))
            input_batch = np.stack((input, dummy, dummy, dummy, dummy, dummy), axis=0)
            segin_batch = np.stack((segin, dummy, dummy, dummy, dummy, dummy), axis=0)
        input_batch = input_batch[:,None,:,:,:]
        segin_batch = segin_batch[:,None,:,:,:]

        return torch.from_numpy(input_batch.copy()).to(torch.float32), torch.from_numpy(segin_batch.copy()).to(torch.float32)


def sigmoid(z):
    return 1/(1+np.exp(-z))

def standardiser(x):
    # only campatible with dtype = numpy array
    return (x - np.mean(x)) / np.std(x)

def aug(img, msk, thickness):
    """
    :params img: input 3D image
    :params msk: imput 3D mask
    :params thickness: expected augmented thickness of the data (in z-direction)
    """

    diff = thickness - img.shape[2]
    return np.concatenate((img, img[:,:,0:diff]), axis=2), np.concatenate((msk, msk[:,:,0:diff]), axis=2)

def make_prediction(test_patches, load_model, ori_size):

    # hardware config
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Predict each 3D patch  
    for i in tqdm(range(test_patches.shape[0])):
        for j in range(test_patches.shape[1]):
            for k in range(test_patches.shape[2]):
            #print(i,j,k)
                single_patch = test_patches[i,j,k, :,:,:]
                single_patch_input = single_patch[None, :]
                single_patch_input = torch.from_numpy(single_patch_input).type(torch.FloatTensor).unsqueeze(0)
                single_patch_input = single_patch_input

                single_patch_prediction = load_model(single_patch_input)

                # single_patch_prediction_sigmoid = torch.sigmoid(single_patch_prediction)
                # single_patch_prediction_sigmoid = single_patch_prediction_sigmoid.detach().numpy()[0,0,:,:,:]

                single_patch_prediction_out = single_patch_prediction.detach().numpy()[0,0,:,:,:]
                
                # test_patches[i,j,k, :,:,:] = single_patch_prediction_sigmoid
                test_patches[i,j,k, :,:,:] = single_patch_prediction_out

    test_output = unpatchify(test_patches, (ori_size[0], ori_size[1], ori_size[2]))
    test_output_sigmoid = sigmoid(test_output)

    return test_output, test_output_sigmoid

def verification(traw_path, idx, load_model, sav_img_path, mode):
    """
    idx: inde of the test img/seg
    mode: str, decide which output to save => ['sigmoid', 'normal']
    """

    # Load data
    raw_file_list = os.listdir(traw_path)
    # seg_file_list = os.listdir(tseg_path)

    raw_img = traw_path+raw_file_list[idx]
    # seg_img = tseg_path+seg_file_list[idx]
    print(f"Test image file: {raw_img}")

    raw_arr = nib.load(raw_img).get_fdata() # (1080*1280*52)
    # seg_arr = nib.load(seg_img).get_fdata()

    raw_arr = raw_arr[64:1024, 64:1216, :]
    # seg_arr = seg_arr[64:1024, 64:1216, :]

    ori_thickness = raw_arr.shape[2]

    new_raw = zoom(raw_arr, (1,1,64/raw_arr.shape[2]), order=0, mode='nearest')
    # new_seg = zoom(seg_arr, (1,1,64/seg_arr.shape[2]), order=0, mode='nearest')
    
    # Standardization
    new_raw = standardiser(new_raw)
    ori_size = new_raw.shape

    # pachify
    test_patches = patchify(new_raw, (64,64,64), 64)

    test_output, test_output_sigmoid = make_prediction(test_patches, load_model, ori_size)

    # save as nifti image
    if mode == 'sigmoid':
        # reshape to original thickness
        test_output_sigmoid = zoom(test_output_sigmoid, (1,1,ori_thickness/64), order=0, mode="nearest")
        nifimg = nib.Nifti1Image(test_output_sigmoid, np.eye(4))

    elif mode == 'normal':
        # reshape to original thickness
        test_output = zoom(test_output, (1,1,ori_thickness/64), order=0, mode="nearest")
        nifimg = nib.Nifti1Image(test_output, np.eye(4))

    
    nib.save(nifimg, sav_img_path)
    print("Output Neuroimage is successfully saved!")