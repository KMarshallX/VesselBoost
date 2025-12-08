"""
Data augmentation utilities for 3D medical image segmentation

This module provides:
- AugmentationUtils: Comprehensive augmentation techniques
- RandomCrop3D: Random 3D cropping for memory efficiency

Editor: Marshall Xu
Last Edited: 26/11/2025
"""

import torch
import torchio as tio
import numpy as np
import scipy.ndimage as scind
from skimage.filters import threshold_otsu
from typing import Optional, Tuple, Union

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
            
        if inp.shape == self.size:
            return inp
            
        w, h, d = inp.shape
        target_w, target_h, target_d = self.size
        
        # OPTIMIZED: Use precomputed zoom factors
        zoom_factors = (target_w / w, target_h / h, target_d / d)
        
        # OPTIMIZED: Use order=0 (nearest neighbor) for speed, preserve memory layout
        return scind.zoom(inp, zoom_factors, order=0, mode='nearest', prefilter=False)

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
        # First resize both images to target size - OPTIMIZE: Only resize if needed
        if input_img.shape != self.size:
            input_img = self.zooming(input_img)
        if seg_img.shape != self.size:
            seg_img = self.zooming(seg_img)

        if self.mode == "on":
            # OPTIMIZED: Pre-allocate arrays to avoid repeated memory allocation
            batch_size = 6
            input_batch = np.empty((batch_size,) + input_img.shape, dtype=input_img.dtype)
            seg_batch = np.empty((batch_size,) + seg_img.shape, dtype=seg_img.dtype)
            
            # Original + rotations + flips
            input_batch[0] = input_img
            seg_batch[0] = seg_img
            
            # Rotations (reuse computation)
            for i in range(1, 4):
                input_batch[i] = self.rot(input_img, i)
                seg_batch[i] = self.rot(seg_img, i)
            
            # Flips
            input_batch[4] = self.flip_hr(input_img, 1)
            seg_batch[4] = self.flip_hr(seg_img, 1)
            input_batch[5] = self.flip_vt(input_img, 1)
            seg_batch[5] = self.flip_vt(seg_img, 1)
            
        elif self.mode == "repeat":
            # OPTIMIZED: Use np.tile instead of list multiplication
            input_batch = np.tile(input_img[None, ...], (6, 1, 1, 1))
            seg_batch = np.tile(seg_img[None, ...], (6, 1, 1, 1))
            
        elif self.mode == "mode1":
            # Pre-allocate for efficiency
            input_batch = np.empty((6,) + input_img.shape, dtype=input_img.dtype)
            seg_batch = np.empty((6,) + seg_img.shape, dtype=seg_img.dtype)
            
            input_batch[0] = input_img
            seg_batch[0] = seg_img
            
            # Rotations
            for i in range(1, 4):
                input_batch[i] = self.rot(input_img, i)
                seg_batch[i] = self.rot(seg_img, i)
            
            # Blurring (only for input)
            input_batch[4] = self.filter(input_img, 2)
            input_batch[5] = self.filter(input_img, 3)
            seg_batch[4] = seg_img  # No filtering for segmentation
            seg_batch[5] = seg_img
            
        elif self.mode == "mode2":
            # Single patch with blurring
            input_batch = self.filter(input_img, 2)[None, ...]
            seg_batch = seg_img[None, ...]  # No filtering for segmentation
            
        elif self.mode == "mode3":
            # Random rotation or blurring
            if np.random.rand() < 0.5:
                # Random rotation
                k = np.random.randint(1, 4)
                input_batch = self.rot(input_img, k)[None, ...]
                seg_batch = self.rot(seg_img, k)[None, ...]
            else:
                # Random blurring
                input_batch = self.filter(input_img, 2)[None, ...]
                seg_batch = seg_img[None, ...]
                
        elif self.mode == "off":
            # No augmentation - most efficient
            input_batch = input_img[None, ...]
            seg_batch = seg_img[None, ...]
            
        # OPTIMIZED: Add channel dimension more efficiently
        input_batch = input_batch[:, None, ...]  # (batch, 1, depth, height, width)
        seg_batch = seg_batch[:, None, ...]

        # OPTIMIZED: Convert to tensors without unnecessary copy
        input_tensor = torch.from_numpy(input_batch).to(torch.float32)
        seg_tensor = torch.from_numpy(seg_batch).to(torch.float32)

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


class TorchIOAugmentationUtils:

    def __init__(self, mode: str = 'spatial'):
        """
        Initialize TorchIO augmentation utilities.
        Args:
            mode: Augmentation mode:
                - 'all': Apply all augmentations
                - 'random': Randomly apply a subset of augmentations
                - 'spatial': Spatial transformations only (flips, elastic deformations)
                - 'intensity': Intensity transformations only (blurring, bias, noise)
                - 'flip': Legacy flipping only
                - 'off': No augmentation, return original subject
        Raises:
            ValueError: For unsupported modes
        """

        self.mode = mode

    def _blur(self, p: float = 1, std: float = 0.85) -> tio.RandomBlur:
        return tio.RandomBlur(std=std, p=p)

    def _bias(self, p: float = 1, coefficients: float = 0.15, order: int = 3) -> tio.RandomBiasField:
        return tio.RandomBiasField(coefficients=coefficients, order=order, p=p)

    def _noise(self, p: float = 1, mean: float = 0, std: float = 0.008) -> tio.RandomNoise:
        return tio.RandomNoise(mean=mean, std=std, p=p)

    def _flip(self, p: float = 1, axes: Union[Tuple[int, ...], int] = (0, 1, 2), probability: float = 1.0) -> tio.RandomFlip:
        # Randomly choose a single axis from the available axes
        if isinstance(axes, tuple):
            # If fewer than 2 options are provided, fall back to single-axis choice
            if len(axes) < 2:
                random_axis = int(np.random.choice(axes))
            else:
                # Choose either a single axis or a pair of distinct axes at random
                if np.random.rand() < 0.2:
                    random_axis = int(np.random.choice(axes))
                else:
                    chosen = np.random.choice(axes, size=2, replace=False)
                    # ensure returned as a tuple of ints
                    random_axis = (int(chosen[0]), int(chosen[1]))
        else:
            random_axis = axes

        return tio.RandomFlip(axes=random_axis, flip_probability=probability, p=p)

    def _elastic_deform(self, p: float = 1, num_control_points: int = 9, 
                        max_displacement: int = 2, 
                        locked_borders: int = 2) -> tio.RandomElasticDeformation:
        return tio.RandomElasticDeformation(num_control_points=num_control_points, 
                                            max_displacement=max_displacement, 
                                            locked_borders=locked_borders, p=p)

    def __call__(self, image_batch: torch.Tensor, seg_batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply TorchIO augmentations to a batch pair
        
        Args:
            image_batch: Input image batch tensor (shape: [batch_size, depth, height, width])
            seg_batch: Input segmentation label batch tensor (shape: [batch_size, depth, height, width])

        Returns:
            Tuple of torch.Tensor (augmented_image_batch, augmented_label_batch)
        """
        # Create TorchIO Subject
        subject_batch = tio.Subject(
            image=tio.ScalarImage(tensor=image_batch),
            label=tio.LabelMap(tensor=seg_batch)
        )
        # Define augmentation transforms based on mode
        if self.mode == 'all':
            transforms = tio.Compose([
                self._blur(),
                self._bias(),
                self._noise(),
                self._flip(axes=(0, 1, 2)),
                self._elastic_deform()
            ])
        elif self.mode == 'random':
            transforms = tio.OneOf({
                self._blur() : 0.1,
                self._bias() : 0.1,
                self._noise() : 0.1,
                self._flip(axes=(0, 1, 2)) : 0.35,
                self._elastic_deform() : 0.35
            })
        elif self.mode == 'spatial':
            transforms = tio.Compose([
                self._flip(axes=(0, 1, 2)),
                self._elastic_deform()
            ])
        elif self.mode == 'intensity':
            transforms = tio.OneOf([
                self._blur(),
                self._bias(),
                self._noise()
            ])
        elif self.mode == 'flip': # legacy
            transforms = self._flip(axes=(0, 1, 2))
        elif self.mode == 'off':
            # No augmentation, return original subject
            return subject_batch['image'].data.unsqueeze(1), subject_batch['label'].data.unsqueeze(1) # type: ignore
        else:
            raise ValueError(f"Unsupported mode '{self.mode}' for TorchIO augmentations")
        
        # Apply the transform to the subject batch
        transformed_subject = transforms(subject_batch)
        
        # # Track which transform was applied (for OneOf modes)
        # # TESTING BLOCK, UNCOMMENT THIS FOR DEBUGGING
        # if self.mode in ['random', 'intensity']:
        #     applied_transforms = [str(transform) for transform in transformed_subject.history]
        #     print(f"Applied transforms in {self.mode} mode: {applied_transforms[-1] if applied_transforms else 'None'}")

        # Extract image and label tensors
        image_tensor = transformed_subject['image'].data.unsqueeze(1) # type: ignore
        label_tensor = transformed_subject['label'].data.unsqueeze(1) # type: ignore
        
        return image_tensor, label_tensor
    
class Crop3D:
        
    def __init__(self, mode: str = 'random', 
                output_size: Union[Tuple[int, int, int], None] = (64, 64, 64), 
                resize: bool = False, 
                mean: int = 128):
        """
        Initialize Crop3D cropping configuration.
        Args:
            mode: 'random' or 'fixed'.
            output_size: Output size after cropping/resizing (tuple of ints).
            resize: If True, resize cropped patch to output_size.
            mean: Mean crop size (used as the center of a normal distribution) for random cropping.
        Raises:
            ValueError: For invalid mode or missing output_size.
        Example:
            1. Random cropping with resizing:
                cropper = Crop3D(mode='random', output_size=(64, 64, 64), resize=True, mean=128)
            2. Random cropping without resizing:
                cropper = Crop3D(mode='random', output_size=None, resize=False, mean=128)
            3. Fixed cropping without resizing:
                cropper = Crop3D(mode='fixed', output_size=(64, 64, 64), resize=False)
        """
        if mode not in ('random', 'fixed'):
            raise ValueError(f"Invalid mode '{mode}'. Valid modes: ['random', 'fixed']")
        if mode == 'fixed' and output_size is None:
            raise ValueError("Output size must be specified for 'fixed' mode")
        if resize == True and output_size is None:
            raise ValueError("Output size must be specified when resizing is enabled")
        
        self.mode = mode
        self.output_size = output_size
        self.resize = resize
        # Backwards-compat: if caller provided the old keyword, prefer it
        self.mean = int(mean)

    @staticmethod
    def _crop_size(mode: str, shape: Tuple[int, int, int], output_size: Union[Tuple[int, int, int], None], mean: int) -> Tuple[int, int, int]:
        """
        Compute crop size based on mode and shape.
        """
        if mode == 'random':
            # crop_h = int(torch.normal(mean, 5, ()).item())
            # crop_w = int(torch.normal(mean, 5, ()).item())
            # crop_d = int(torch.normal(mean, 5, ()).item())
            crop_h = int(torch.randint(32, shape[0], ()).item())
            crop_w = int(torch.randint(32, shape[1], ()).item())
            crop_d = int(torch.randint(32, shape[2], ()).item())
        else:
            if not isinstance(output_size, tuple) or len(output_size) != 3:
                raise ValueError("Output size must be a tuple of 3 integers for 'fixed' mode")
            crop_h, crop_w, crop_d = (int(output_size[0]), int(output_size[1]), int(output_size[2]))
        crop_h = min(crop_h, int(shape[0]) - 1)
        crop_w = min(crop_w, int(shape[1]) - 1)
        crop_d = min(crop_d, int(shape[2]) - 1)
        return int(crop_h), int(crop_w), int(crop_d)

    @staticmethod
    def _get_slice(sz: int, crop_sz: int) -> Tuple[int, int]:
        """
        Get random slice coordinates for cropping.
        """
        if sz <= crop_sz:
            return 0, int(sz)
        start = int(torch.randint(0, int(sz) - int(crop_sz) + 1, (1,)).item())
        return int(start), int(start + crop_sz)
    
    @staticmethod
    def _crop(x: np.ndarray, slice_h: Tuple[int, int], slice_w: Tuple[int, int], slice_d: Tuple[int, int]) -> np.ndarray:
        """Crop array using slice coordinates."""
        return x[slice_h[0]:slice_h[1], slice_w[0]:slice_w[1], slice_d[0]:slice_d[1]]

    def __call__(self, image_array: np.ndarray, 
                label_array: np.ndarray, 
                mask: Union[np.ndarray, str, None] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Crop and optionally resize a 3D image and label.

        Cropping strategies:
            - If mask is None: random or fixed cropping.
            - If mask is a np.ndarray: crop center chosen from ROI voxels (nonzero mask).
            - If mask == 'otsu': Otsu thresholding is applied to image to generate mask, then crop center chosen from ROI.
            - If mask == 'lazy': crop center randomly chosen within 20%-80% of each dimension.

        Args:
            image_array: Input image (3D numpy array).
            label_array: Input label (3D numpy array, same shape as image).
            mask: np.ndarray (ROI mask), 'otsu', 'lazy', or None.
        Returns:
            Cropped (and optionally resized) image and label as numpy arrays.
        Raises:
            ValueError: For shape mismatch, invalid mask, or no ROI voxels.
        """
        if image_array.shape != label_array.shape:
            raise ValueError(f"Image shape {image_array.shape} != label shape {label_array.shape}")
        shape = tuple(int(dim) for dim in image_array.shape[:3])
        if len(shape) != 3:
            raise ValueError(f"Input image must be 3D, got shape {image_array.shape}")
        if self.mode == 'random' and any(dim < self.mean for dim in shape):
            raise ValueError(f"Image size {shape} must be >= {self.mean} in all dimensions")
        crop_size = self._crop_size(self.mode, shape, self.output_size, self.mean)

        # If mask is provided, select crop center from ROI or use 'lazy' option
        if mask is not None:
            if isinstance(mask, str):
                if mask == 'otsu':
                    mask = (image_array > threshold_otsu(image_array)).astype(np.uint8)
                    # Now treat as np.ndarray mask
                elif mask == 'lazy':
                    # 'lazy' option: center falls within 20%-80% of each dimension
                    crop_half = [s // 2 for s in crop_size]
                    valid_min = [int(0.2 * shape[i]) for i in range(3)]
                    valid_max = [int(0.8 * shape[i]) for i in range(3)]
                    center_idx = [np.random.randint(valid_min[i], valid_max[i]) for i in range(3)]
                    start = [max(0, int(center_idx[i]) - crop_half[i]) for i in range(3)]
                    end = [min(shape[i], start[i] + crop_size[i]) for i in range(3)]
                    for i in range(3):
                        if end[i] - start[i] < crop_size[i]:
                            start[i] = max(0, shape[i] - crop_size[i])
                            end[i] = start[i] + crop_size[i]
                    slice_coords = [(start[i], end[i]) for i in range(3)]
                else:
                    raise ValueError(f"Invalid mask type: {mask}. Expected np.ndarray, 'otsu', or 'lazy'.")
            if isinstance(mask, np.ndarray):
                if mask.shape != shape:
                    raise ValueError(f"Mask shape {mask.shape} must match image shape {shape}")
                roi_flat_indices = np.flatnonzero(mask)
                if roi_flat_indices.size == 0:
                    raise ValueError("Mask contains no ROI voxels.")
                chosen_flat = np.random.choice(roi_flat_indices)
                center_idx = np.unravel_index(chosen_flat, mask.shape)
                crop_half = [s // 2 for s in crop_size]
                start = [max(0, int(center_idx[i]) - crop_half[i]) for i in range(3)]
                end = [min(shape[i], start[i] + crop_size[i]) for i in range(3)]
                for i in range(3):
                    if end[i] - start[i] < crop_size[i]:
                        start[i] = max(0, shape[i] - crop_size[i])
                        end[i] = start[i] + crop_size[i]
                slice_coords = [(start[i], end[i]) for i in range(3)]
        else:
            slice_coords = [self._get_slice(int(orig_dim), int(crop_dim)) for orig_dim, crop_dim in zip(shape, crop_size)]

        cropped_img = self._crop(image_array, *slice_coords)
        cropped_lab = self._crop(label_array, *slice_coords)
        if not self.resize:
            return cropped_img, cropped_lab
        if self.output_size is None:
            raise ValueError("output_size must be specified for resizing.")
        zoom_factors = tuple(float(out_dim) / float(crop_dim) for out_dim, crop_dim in zip(self.output_size, crop_size))
        resized_img = scind.zoom(cropped_img, zoom_factors, order=0, mode='nearest')
        resized_lab = scind.zoom(cropped_lab, zoom_factors, order=0, mode='nearest')
        return resized_img, resized_lab


