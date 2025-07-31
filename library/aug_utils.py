"""
Data augmentation utilities for 3D medical image segmentation

This module provides:
- AugmentationUtils: Comprehensive augmentation techniques
- RandomCrop3D: Random 3D cropping for memory efficiency

Editor: Marshall Xu
Last Edited: 30/07/2025
"""

import torch
import numpy as np
import scipy.ndimage as scind
from typing import Tuple


class AugmentationUtils:
    """
    Data augmentation utilities for 3D medical image segmentation.
    
    Provides various augmentation techniques including rotation, flipping,
    zooming, and filtering for training data enhancement.
    """
    
    def __init__(self, size: Tuple[int, int, int], mode: str):
        """
        Initialize augmentation utilities.
        
        Args:
            size: Expected size for resampled data (width, height, depth)
            mode: Augmentation mode:
                - "off": No augmentation, single patch
                - "on": Full augmentation with rotations and flips
                - "repeat": Repeat same patch 6 times
                - "mode1": Rotation and blurring
                - "mode2": Single patch with blurring
                - "mode3": Random rotation or blurring
        """
        self.size = size
        self.mode = mode
        
        # Validate mode
        valid_modes = {"off", "on", "repeat", "mode1", "mode2", "mode3"}
        if mode not in valid_modes:
            raise ValueError(f"Invalid mode '{mode}'. Valid modes: {valid_modes}")
    
    def rot(self, inp: np.ndarray, k: int) -> np.ndarray:
        """
        Rotate array k times by 90 degrees in the xy-plane.
        
        Args:
            inp: Input 3D array
            k: Number of 90-degree rotations
            
        Returns:
            Rotated array
        """
        return np.rot90(inp, k, axes=(0, 1))

    def flip_hr(self, inp: np.ndarray, k: int) -> np.ndarray:
        """
        Rotate array horizontally (x-z plane).
        
        Args:
            inp: Input 3D array
            k: Number of 90-degree rotations
            
        Returns:
            Rotated array
        """
        return np.rot90(inp, k, axes=(0, 2))

    def flip_vt(self, inp: np.ndarray, k: int) -> np.ndarray:
        """
        Rotate array vertically (y-z plane).
        
        Args:
            inp: Input 3D array
            k: Number of 90-degree rotations
            
        Returns:
            Rotated array
        """
        return np.rot90(inp, k, axes=(1, 2))

    def zooming(self, inp: np.ndarray) -> np.ndarray:
        """
        Resize input array to target size using nearest neighbor interpolation.
        
        Args:
            inp: Input 3D array
            
        Returns:
            Resized array
            
        Raises:
            ValueError: If input is not 3D
        """
        if len(inp.shape) != 3:
            raise ValueError("Only 3D data is accepted")
            
        w, h, d = inp.shape
        target_w, target_h, target_d = self.size

        if (target_w, target_h, target_d) == (w, h, d):
            return inp
        
        zoom_factors = (target_w / w, target_h / h, target_d / d)
        return scind.zoom(inp, zoom_factors, order=0, mode='nearest')

    def filter(self, inp: np.ndarray, sigma: float) -> np.ndarray:
        """
        Apply Gaussian filter to the input array.
        
        Args:
            inp: Input array
            sigma: Standard deviation for Gaussian kernel
            
        Returns:
            Filtered array
        """
        return scind.gaussian_filter(inp, sigma)
    
    def __call__(self, input_img: np.ndarray, seg_img: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply augmentation to input and segmentation images.
        
        Args:
            input_img: Input image array
            seg_img: Segmentation mask array
            
        Returns:
            Tuple of (augmented_input_batch, augmented_seg_batch) as tensors
        """
        # First resize both images to target size
        input_img = self.zooming(input_img)
        seg_img = self.zooming(seg_img)

        if self.mode == "on":
            # Full augmentation: rotation and flipping
            input_batch = np.stack([
                input_img,
                self.rot(input_img, 1),
                self.rot(input_img, 2), 
                self.rot(input_img, 3),
                self.flip_hr(input_img, 1),
                self.flip_vt(input_img, 1)
            ], axis=0)
            
            seg_batch = np.stack([
                seg_img,
                self.rot(seg_img, 1),
                self.rot(seg_img, 2),
                self.rot(seg_img, 3), 
                self.flip_hr(seg_img, 1),
                self.flip_vt(seg_img, 1)
            ], axis=0)
            
        elif self.mode == "repeat":
            # Repeat same patch 6 times
            input_batch = np.stack([input_img] * 6, axis=0)
            seg_batch = np.stack([seg_img] * 6, axis=0)
            
        elif self.mode == "mode1":
            # Rotation and blurring
            input_batch = np.stack([
                input_img,
                self.rot(input_img, 1),
                self.rot(input_img, 2),
                self.rot(input_img, 3),
                self.filter(input_img, 2),
                self.filter(input_img, 3)
            ], axis=0)
            
            seg_batch = np.stack([
                seg_img,
                self.rot(seg_img, 1), 
                self.rot(seg_img, 2),
                self.rot(seg_img, 3),
                seg_img,  # No filtering for segmentation
                seg_img
            ], axis=0)
            
        elif self.mode == "mode2":
            # Single patch with blurring
            input_batch = np.expand_dims(self.filter(input_img, 2), axis=0)
            seg_batch = np.expand_dims(seg_img, axis=0)  # No filtering for segmentation
            
        elif self.mode == "mode3":
            # Random rotation or blurring
            if np.random.rand() < 0.5:
                # Random rotation
                k = np.random.randint(1, 4)
                input_batch = np.expand_dims(self.rot(input_img, k), axis=0)
                seg_batch = np.expand_dims(self.rot(seg_img, k), axis=0)
            else:
                # Random blurring
                input_batch = np.expand_dims(self.filter(input_img, 2), axis=0)
                seg_batch = np.expand_dims(seg_img, axis=0)
                
        elif self.mode == "off":
            # No augmentation
            input_batch = np.expand_dims(input_img, axis=0)
            seg_batch = np.expand_dims(seg_img, axis=0)
            
        # Add channel dimension: (batch, channel, depth, height, width)
        input_batch = input_batch[:, None, :, :, :]
        seg_batch = seg_batch[:, None, :, :, :]

        # Convert to PyTorch tensors
        input_tensor = torch.from_numpy(input_batch.copy()).to(torch.float32)
        seg_tensor = torch.from_numpy(seg_batch.copy()).to(torch.float32)

        return input_tensor, seg_tensor


class RandomCrop3D:
    """
    Random 3D cropping utility for medical image segmentation.
    
    Randomly crops a 3D volume from the input image and resizes it to a fixed size.
    Useful for data augmentation and memory-efficient training on large volumes.
    """
    
    def __init__(
        self, 
        img_sz: Tuple[int, int, int], 
        exp_sz: Tuple[int, int, int], 
        test_mode: bool = False, 
        crop_low_thresh: int = 128
    ):
        """
        Initialize Random Crop 3D.
        
        Args:
            img_sz: Original image size (height, width, depth)
            exp_sz: Expected output size after cropping and resizing
            test_mode: If True, use deterministic cropping for testing
            crop_low_thresh: Minimum crop size for each dimension
            
        Raises:
            ValueError: If image size is smaller than crop_low_thresh
        """
        h, w, d = img_sz
        self.test_mode = test_mode
        self.img_sz = img_sz
        self.exp_sz = exp_sz
        
        # Validate input dimensions
        if any(dim < crop_low_thresh for dim in img_sz):
            raise ValueError(f"Image size {img_sz} must be >= {crop_low_thresh} in all dimensions")
        
        # Determine crop size
        if not test_mode:
            # Training mode: random crop size
            crop_h = torch.randint(crop_low_thresh, h + 1, (1,)).item()
            crop_w = torch.randint(crop_low_thresh, w + 1, (1,)).item()
            crop_d = torch.randint(crop_low_thresh, d + 1, (1,)).item()
        else:
            # Test mode: use larger crop for more context
            crop_h = torch.randint(crop_low_thresh, h + 1, (1,)).item()
            crop_w = torch.randint(crop_low_thresh, w + 1, (1,)).item()
            crop_d = torch.randint(crop_low_thresh, d + 1, (1,)).item()
        
        self.crop_sz = (crop_h, crop_w, crop_d)
        
        # Validate crop size
        if any(crop >= orig for crop, orig in zip(self.crop_sz, img_sz)):
            self.crop_sz = tuple(min(crop, orig - 1) for crop, orig in zip(self.crop_sz, img_sz))
        
    def __call__(
        self, 
        img: np.ndarray, 
        lab: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply random cropping and resizing to image and label.
        
        Args:
            img: Input image array
            lab: Input label array
            
        Returns:
            Tuple of (cropped_resized_image, cropped_resized_label)
            
        Raises:
            ValueError: If image and label shapes don't match
        """
        if img.shape != lab.shape:
            raise ValueError(f"Image shape {img.shape} != label shape {lab.shape}")
        
        # Generate random crop coordinates
        slice_coords = [
            self._get_slice(int(orig_size), int(crop_size)) 
            for orig_size, crop_size in zip(self.img_sz, self.crop_sz)
        ]
        
        # Crop both image and label
        cropped_img = self._crop(img, *slice_coords)
        cropped_lab = self._crop(lab, *slice_coords)
        
        # Resize to expected size
        zoom_factors = tuple(
            exp / crop for exp, crop in zip(self.exp_sz, self.crop_sz)
        )
        
        resized_img = scind.zoom(cropped_img, zoom_factors, order=1, mode='nearest')
        resized_lab = scind.zoom(cropped_lab, zoom_factors, order=0, mode='nearest')
        
        return resized_img, resized_lab

    @staticmethod
    def _get_slice(sz: int, crop_sz: int) -> Tuple[int, int]:
        """
        Get random slice coordinates for cropping.
        
        Args:
            sz: Original dimension size
            crop_sz: Desired crop size
            
        Returns:
            Tuple of (start_idx, end_idx)
        """
        if sz <= crop_sz:
            return 0, sz
        
        max_start = sz - crop_sz
        lower_bound = torch.randint(0, max_start + 1, (1,)).item()
        return int(lower_bound), int(lower_bound + crop_sz)
    
    @staticmethod
    def _crop(
        x: np.ndarray, 
        slice_h: Tuple[int, int], 
        slice_w: Tuple[int, int], 
        slice_d: Tuple[int, int]
    ) -> np.ndarray:
        """
        Crop array using slice coordinates.
        
        Args:
            x: Input array
            slice_h: Height slice coordinates (start, end)
            slice_w: Width slice coordinates (start, end) 
            slice_d: Depth slice coordinates (start, end)
            
        Returns:
            Cropped array
        """
        return x[slice_h[0]:slice_h[1], slice_w[0]:slice_w[1], slice_d[0]:slice_d[1]]


