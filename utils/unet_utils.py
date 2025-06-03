"""
helper functions library

Editor: Marshall Xu
Last Edited: 07/09/2023
"""

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import scipy.ndimage as scind

def sigmoid(z):
    return 1/(1+np.exp(-z))

def normaliser(x):
    # only campatible with dtype = numpy array
    return (x - np.amin(x)) / (np.amax(x) - np.amin(x))

def standardiser(x):
    # only campatible with dtype = numpy array
    return (x - np.mean(x)) / np.std(x)

def thresholding(arr,thresh):
    arr[arr<thresh] = 0
    arr[arr>thresh] = 1
    return arr

class DiceLoss(nn.Module):
    def __init__(self, smooth = 1e-4):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):


        pred = torch.sigmoid(pred)
        # flatten pred and target tensors
        pred = pred.view(-1)
        target = target.view(-1)

        intersection =  (pred * target).sum(-1)
        dice = (2.*intersection + self.smooth) / (pred.sum(-1) + target.sum(-1) + self.smooth)

        return 1 - dice
    
class DiceCoeff(nn.Module):
    def __init__(self, delta=0.5, smooth = 1e-4):
        super().__init__()
        self.delta = delta
        self.smooth = smooth
        """
        The Dice similarity coefficient, also known as the Sørensen–Dice index or simply Dice coefficient, is a statistical tool which measures the similarity between two sets of data.
        
        Parameters
        ----------
        delta : float, optional
            controls weight given to false positive and false negatives, by default 0.5
        smooth : float, optional
            smoothing constant to prevent division by zero errors, by default 0.000001
        """
    def forward(self, pred, target):
        
        pred = torch.sigmoid(pred)
        
        # Flatten
        pred = pred.reshape(-1)
        target = target.reshape(-1)
        
        # cardinalities
        tp = (pred * target).sum()
        fp = ((1-target) * pred).sum()
        fn = (target * (1-pred)).sum()

        dice_score = (tp + self.smooth) / (tp + self.delta*fn + (1-self.delta)*fp + self.smooth)

        return dice_score.mean()

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
    
class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, smooth=1e-3):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        # Flatten
        pred = pred.reshape(-1)
        target = target.reshape(-1)

        # cardinalities
        tp = (pred * target).sum()
        fp = ((1-target) * pred).sum()
        fn = (target * (1-pred)).sum()

        tversky_score = (tp + self.smooth) / (tp + self.alpha*fp + self.beta*fn + self.smooth)

        return 1 - tversky_score
    
class ComboLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        """
        Combo Loss: Handling Input and Output Imbalance in Multi-Organ Segmentation
        Link: https://arxiv.org/abs/1805.02798
        
        Parameters
        ----------
        alpha : float, optional
            controls weighting of dice and cross-entropy loss., by default 0.5
        beta : float, optional
        beta > 0.5 penalises false negatives more than false positives., by default 0.5
        """
        
    def forward(self, pred, target):
        dice = DiceCoeff()(pred, target)
        # Clip values to prevent division by zero error
        epsilon = 1e-9
        pred = torch.clamp(pred, epsilon, 1. - epsilon)
        bce = BCELoss()(pred, target)

        if self.beta is not None:
            device = torch.device("cuda:0" if bce.is_cuda else "cpu")
            beta_weight = torch.tensor([self.beta, 1-self.beta], device=device)
            bce = beta_weight * bce
        bce = bce.sum().mean()

        if self.alpha is not None:
            combo_loss = (self.alpha * bce) - ((1 - self.alpha) * dice)
        else:
            combo_loss = bce - dice
        return combo_loss
    
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

        if size[0] == w and size[1] == h and size[2] == d:
            return inp
        else:
            return scind.zoom(inp, (size[0]/w, size[1]/h, size[2]/d), order=0, mode='nearest')

    def filter(self, inp, sigma):
        # apply gaussian filter to the patch
        return scind.gaussian_filter(inp, sigma)
    
    def __call__(self, input, segin):
        input = self.zooming(input)
        segin = self.zooming(segin)

        if self.mode == "on":
            # print("Aug mode on, rotation/flip only")
            input_batch = np.stack((input, self.rot(input, 1), self.rot(input, 2), self.rot(input, 3), 
            self.flip_hr(input, 1), self.flip_vt(input, 1)), axis=0)
            segin_batch = np.stack((segin, self.rot(segin, 1), self.rot(segin, 2), self.rot(segin, 3), 
            self.flip_hr(segin, 1), self.flip_vt(segin, 1)), axis=0)
        elif self.mode == "repeat":
            # print("Aug mode repeat, repeat the same patch 6 times")
            input_batch = np.stack((input, input, input, input, input, input), axis=0)
            segin_batch = np.stack((segin, segin, segin, segin, segin, segin), axis=0)
        elif self.mode == "mode1":
            # print("Aug mode 1, rotation & blurring")
            input_batch = np.stack((input, self.rot(input, 1), self.rot(input, 2), self.rot(input, 3), self.filter(input, 2), self.filter(input, 3)), axis=0)
            segin_batch = np.stack((segin, self.rot(segin, 1), self.rot(segin, 2), self.rot(segin, 3), segin, segin), axis=0)
        elif self.mode == "mode2":
            # print("Aug mode 2, only one patch, with blurring effect")
            input_batch = np.expand_dims(self.filter(input, 2), axis=0)
            segin_batch = np.expand_dims(self.filter(segin, 2), axis=0)
        elif self.mode == "mode3":
            ind = np.random.randint(0, 2)
            if ind == 0:
                k = np.random.randint(1, 4)
                # print("Aug mode 3, rotate")
                input_batch = np.expand_dims(self.rot(input, k), axis=0)
                segin_batch = np.expand_dims(self.rot(segin, k), axis=0)
            elif ind == 1:
                # print("Aug mode 3, blur")
                input_batch = np.expand_dims(self.filter(input, 2), axis=0)
                segin_batch = np.expand_dims(self.filter(segin, 2), axis=0)
        elif self.mode == "off":
            # print("Aug mode off")
            input_batch = np.expand_dims(input, axis=0)
            segin_batch = np.expand_dims(segin, axis=0)
            
        input_batch = input_batch[:,None,:,:,:] # type: ignore
        segin_batch = segin_batch[:,None,:,:,:] # type: ignore

        return torch.from_numpy(input_batch.copy()).to(torch.float32), torch.from_numpy(segin_batch.copy()).to(torch.float32)

class RandomCrop3D():
    """
    Resample the input image slab by randomly cropping a 3D volume, and reshape to a fixed size e.g.(64,64,64)
    """
    def __init__(self, img_sz, exp_sz, test_mode=False, lower_thresh=128):
        h, w, d = img_sz
        self.test_mode = test_mode
        
        if not test_mode:
            crop_h = torch.randint(128, 256, (1,)).item()
            crop_w = torch.randint(128, 256, (1,)).item()
            crop_d = torch.randint(128, 256, (1,)).item()
            assert (h, w, d) > (crop_h, crop_w, crop_d)
            self.crop_sz = tuple((crop_h, crop_w, crop_d))
        else:
            #NOTE: modified @ 26/05/2025
            crop_h = torch.randint(lower_thresh, h, (1,)).item()
            crop_w = torch.randint(lower_thresh, w, (1,)).item()
            crop_d = torch.randint(lower_thresh, d, (1,)).item()
            assert (h, w, d) > (crop_h, crop_w, crop_d)
            self.crop_sz = tuple((crop_h, crop_w, crop_d))
        
        self.img_sz  = tuple((h, w, d))
        self.exp_sz = exp_sz
        
    def __call__(self, img, lab):
        print("*"*50+'/n')
        slice_hwd = [self._get_slice(i, k) for i, k in zip(self.img_sz, self.crop_sz)]
        print(f"Random crop size: {self.crop_sz}")
        print("*"*50+'/n')
        return scind.zoom(self._crop(img, *slice_hwd),(self.exp_sz[0]/self.crop_sz[0], self.exp_sz[1]/self.crop_sz[1], self.exp_sz[2]/self.crop_sz[2]), order=0, mode='nearest'), scind.zoom(self._crop(lab, *slice_hwd),(self.exp_sz[0]/self.crop_sz[0], self.exp_sz[1]/self.crop_sz[1], self.exp_sz[2]/self.crop_sz[2]), order=0, mode='nearest')

    @staticmethod
    def _get_slice(sz, crop_sz):
        try : 
            lower_bound = torch.randint(sz-crop_sz, (1,)).item()
            print(f"Origin location of the crop: {lower_bound}")
            return lower_bound, lower_bound + crop_sz
        except: 
            return (None, None)
    
    @staticmethod
    def _crop(x, slice_h, slice_w, slice_d):
        return x[slice_h[0]:slice_h[1], slice_w[0]:slice_w[1], slice_d[0]:slice_d[1]]
    