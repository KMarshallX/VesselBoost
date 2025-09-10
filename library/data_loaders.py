"""
Improved data loader functions

Editor: Marshall Xu
Last Edited: 05/08/2025
"""

import time
import nibabel as nib  # type: ignore
import numpy as np
import logging
import scipy.ndimage as scind  
from pathlib import Path
from typing import Tuple, List, Dict, Optional, Union

import torch

from .aug_utils import RandomCrop3D, Crop3D
from .loss_func import standardiser, normaliser

# Set up logging
logger = logging.getLogger(__name__)

class SingleChannelLoader:
    
    def __init__(self, img_path: Union[str, Path], seg_path: Union[str, Path], 
                patch_size: Tuple[int, int, int], step: int, 
                normalization: str = 'normalize',
                crop_low_thresh: int = 128,
                batch_multiplier: int = 5):
        """
        Initialize the data loader for a single image/segmentation pair.
        
        Args:
            img_path (str or Path): Path to the raw image file (NIfTI format)
            seg_path (str or Path): Path to the segmentation file (NIfTI format)
            patch_size (Tuple[int, int, int]): Size of patches to extract (height, width, depth)
            step (int): Number of batches (iterations) to generate per epoch
            normalization (str): Type of normalization to apply ('standardize', 'normalize', or 'none')
            crop_low_thresh (int): Minimum crop size for random cropping in each dimension
            batch_multiplier (int): Number of fixed-size patches to generate from each large crop (default: 5)
        
        Raises:
            FileNotFoundError: If image or segmentation files do not exist
            ValueError: If image and segmentation dimensions or affines do not match
        """
        self.img_path = Path(img_path)
        self.seg_path = Path(seg_path)

        # Validate file existence
        if not self.img_path.exists():
            raise FileNotFoundError(f"Image not found: {self.img_path}")
        if not self.seg_path.exists():
            raise FileNotFoundError(f"Segmentation image not found: {self.seg_path}")

        self.patch_size = patch_size
        self.step = step
        self.normalization = normalization
        self.crop_low_thresh = crop_low_thresh
        self.batch_multiplier = batch_multiplier

        # Load images & check compatibility
        # Cache image shapes
        self.img_arr, self.seg_arr = self._load_and_preprocess_images()

        logger.info(f"Initialized loader for {self.img_path.name}")

    def _load_and_preprocess_images(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load and preprocess images with specified normalization."""
        try:
            # Load images
            img_nifti = nib.load(str(self.img_path))  # type: ignore
            img_numpy = img_nifti.get_fdata()  # type: ignore
            img_shape = img_numpy.shape
            img_affine = img_nifti.affine  # type: ignore
            seg_nifti = nib.load(str(self.seg_path))  # type: ignore
            seg_numpy = seg_nifti.get_fdata()  # type: ignore
            seg_shape = seg_numpy.shape
            seg_affine = seg_nifti.affine  # type: ignore
            if img_shape != seg_shape:
                raise ValueError(
                    f"Image shape mismatch: img {img_shape} vs seg {seg_shape}"
                )
            self._image_shape = img_shape
            # Apply normalization
            if self.normalization == 'standardize':
                img_numpy = standardiser(img_numpy)
            elif self.normalization == 'normalize':
                img_numpy = normaliser(img_numpy)
            # 'none' means no normalization
            return img_numpy, seg_numpy

        except Exception as e:
            raise RuntimeError(f"Error loading images: {e}")

    def _initialize_random_cropper(self) -> Crop3D:
        """Initialize the random cropper"""
        return Crop3D('random', None, False, self.crop_low_thresh)
    
    def _initialize_fixed_cropper(self) -> Crop3D:
        """Initialize the fixed cropper"""
        return Crop3D('fixed', self.patch_size, False)
    
    def _zooming(self, img_crop: np.ndarray, seg_crop: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Resize the cropped image and segmentation to the patch size.
        
        Args:
            img_crop: Cropped image array
            seg_crop: Cropped segmentation array
        
        Returns:
            Tuple of resized image and segmentation arrays
        """
        if img_crop.shape != self.patch_size:
            zoom_factors = tuple(float(out_dim) / float(crop_dim) for out_dim, crop_dim in zip(self.patch_size, img_crop.shape))
            img_crop = scind.zoom(img_crop, zoom_factors, order=0, mode='nearest')
            seg_crop = scind.zoom(seg_crop, zoom_factors, order=0, mode='nearest')
        return img_crop.astype(np.float32), seg_crop.astype(np.int8)

    def __repr__(self) -> str:
        return (f"SingleChannelLoader(\n"
                f"  img: {self.img_path.name}\n"
                f"  seg_img: {self.seg_path.name}\n"
                f"  patch_size: {self.patch_size}\n"
                f"  normalization: {self.normalization}\n"
                f"  crop_low_thresh: {self.crop_low_thresh}\n"
                f"  number of batches: {self.step}\n"
                f"  num. of patches per batch: {self.batch_multiplier+1}\n"
                f"  total num. of patches: {self.step * (self.batch_multiplier + 1)}\n"
                f")")

    def __len__(self) -> int:
        return self.step
        
    def __iter__(self):
        """
        What this loader does:
        1. Crop a large patch from the image and segmentation arrays preserving global features
        2. Generate multiple fixed-sized smaller patches from the large patch
        3. Resize the large patch to match the smaller patch size
        4. Stack all patches as a training batch
        Yields:
            Tuple of (image_batch, segmentation_batch) as torch.Tensor
        """
        # Load images only when iteration begins
        for i in range(self.step):
            try:
                
                # cropper = RandomCrop3D(self._image_shape, self.patch_size, False, self.crop_low_thresh)
                random_cropper = self._initialize_random_cropper()
                fixed_cropper = self._initialize_fixed_cropper()

                large_img_crop, large_seg_crop = random_cropper(self.img_arr, self.seg_arr, mask='lazy')
                img_batch = np.empty((self.batch_multiplier + 1, *self.patch_size), dtype=self.img_arr.dtype)
                seg_batch = np.empty((self.batch_multiplier + 1, *self.patch_size), dtype=self.seg_arr.dtype)
                for j in range(self.batch_multiplier):
                    # Generate fixed-size patches from the large crop
                    img_crop, seg_crop = fixed_cropper(large_img_crop, large_seg_crop)
                    img_batch[j] = img_crop
                    seg_batch[j] = seg_crop
                # Resize the large crop to match the patch size
                large_img_crop, large_seg_crop = self._zooming(large_img_crop, large_seg_crop)
                img_batch[self.batch_multiplier] = large_img_crop
                seg_batch[self.batch_multiplier] = large_seg_crop  
                # Yield the cropped and resized patches as a batch of pytorch tensors
                yield torch.from_numpy(img_batch).float(), torch.from_numpy(seg_batch).ceil().int()

            except Exception as e:
                logger.error(f"Error generating patch {i}: {e}")
                raise

    @property
    def image_shape(self) -> Tuple[int, int, int]:
        """Get the shape of the loaded images."""
        return self._image_shape

class MultiChannelLoader:
    """
    Efficient loader for multiple 3D medical image/segmentation pairs.
    
    Provides organization and memory management for handling multiple image pairs,
    with patch extraction and normalization options matching SingleChannelLoader.
    """
    
    def __init__(self, img_dir: Union[str, Path], seg_dir: Union[str, Path], 
                patch_size: Tuple[int, int, int], step: int, 
                normalization: str = 'normalize',
                crop_low_thresh: int = 128,
                batch_multiplier: int = 5):
        """
        Initialize the multi-channel loader for a directory of image/segmentation pairs.
        
        Args:
            img_dir (str or Path): Path to folder containing processed images
            seg_dir (str or Path): Path to folder containing label images
            patch_size (Tuple[int, int, int]): Size of patches to extract (height, width, depth)
            step (int): Number of batches (iterations) to generate per image
            normalization (str): Type of normalization to apply ('standardize', 'normalize', or 'none')
            crop_low_thresh (int): Minimum crop size for random cropping in each dimension
            batch_multiplier (int): Number of fixed-size patches to generate from each large crop (default: 5)
        """
        self.img_dir = Path(img_dir)
        self.seg_dir = Path(seg_dir)
        self.patch_size = patch_size
        self.step = step
        self.normalization = normalization
        self.crop_low_thresh = crop_low_thresh
        self.batch_multiplier = batch_multiplier

        # Find and match image pairs
        self.image_pairs = self._find_image_pairs()
        
        logger.info(f"Found {len(self.image_pairs)} image pairs")

    def _find_image_pairs(self) -> List[Tuple[Path, Path]]:
        """
        Efficiently find matching image and segmentation pairs.
        
        Returns:
            List of (raw_image_path, seg_image_path) tuples
        """
        if not self.img_dir.exists() or not self.seg_dir.exists():
            raise FileNotFoundError("Image or segmentation directory not found")
        
        # Get sorted file lists
        img_files = sorted([f for f in self.img_dir.iterdir() if f.is_file()])
        seg_files = sorted([f for f in self.seg_dir.iterdir() if f.is_file()])

        if len(img_files) != len(seg_files):
            raise ValueError(f"Mismatch: {len(img_files)} raw images vs {len(seg_files)} segmentations")

        # Create efficient lookup for segmentation files
        seg_lookup = {f.stem.split('.')[0]: f for f in seg_files}
        
        image_pairs = []
        for img_file in img_files:
            img_stem = img_file.stem.split('.')[0]

            # Find matching segmentation file
            seg_file = None
            for seg_key, seg_path in seg_lookup.items():
                if img_stem in seg_key or seg_key in img_stem:
                    seg_file = seg_path
                    break
            
            if seg_file is None:
                raise ValueError(f"No matching segmentation found for {img_file.name}")

            image_pairs.append((img_file, seg_file))

        return image_pairs

    def get_loader(self, index: int) -> SingleChannelLoader:
        """Get a SingleChannelLoader for a specific image pair."""
        if index >= len(self.image_pairs):
            raise IndexError(f"Index {index} out of range for {len(self.image_pairs)} images")

        img_path, seg_path = self.image_pairs[index]
        return SingleChannelLoader(
            img_path, seg_path, self.patch_size, self.step,
            self.normalization, self.crop_low_thresh, self.batch_multiplier
        )

    def get_all_loaders(self) -> Dict[int, SingleChannelLoader]:
        """Get all loaders as a dictionary (for backward compatibility)."""
        return {i: self.get_loader(i) for i in range(len(self.image_pairs))}

    def get_cv_loaders(self, train_indices: List[int]) -> Dict[int, SingleChannelLoader]:
        """
        Get loaders for cross-validation training subset.
        
        Args:
            train_indices: List of indices to include in training set
            
        Returns:
            Dictionary of loaders for specified indices
        """
        loaders = {}
        for i, idx in enumerate(train_indices):
            if idx >= len(self.image_pairs):
                raise IndexError(f"Index {idx} out of range")
            loaders[i] = self.get_loader(idx)
        
        return loaders

    def __len__(self) -> int:
        """
        Get the number of image pairs in the dataset.
        Not the total number of patches.
        """
        return len(self.image_pairs)

    def __getitem__(self, index: int) -> SingleChannelLoader:
        return self.get_loader(index)

    def __repr__(self) -> str:
        return (f"MultiChannelLoader(\n"
                f"  img_dir: {self.img_dir}\n"
                f"  seg_dir: {self.seg_dir}\n"
                f"  num_images: {len(self.image_pairs)}\n"
                f"  patch_size: {self.patch_size}\n"
                f"  normalization: {self.normalization}\n"
                f"  crop_low_thresh: {self.crop_low_thresh}\n"
                f"  total number of batches: {self.step * len(self.image_pairs)}\n"
                f"  total number of patches: {self.step * len(self.image_pairs) * (self.batch_multiplier + 1)}\n"
                f")")


