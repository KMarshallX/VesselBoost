"""
Improved data loader functions

Editor: Marshall Xu
Last Edited: 30/07/2025
"""

import os
import nibabel as nib  # type: ignore
import numpy as np
from typing import Tuple, List, Dict, Optional, Union
from pathlib import Path
import logging

from .aug_utils import RandomCrop3D
from .loss_func import standardiser, normaliser

# Set up logging
logger = logging.getLogger(__name__)

class SingleChannelLoader:
    """
    Efficient data loader for single channel 3D medical images with lazy loading.
    
    This class loads images on-demand rather than storing them in memory,
    improving memory efficiency for large datasets.
    """
    
    def __init__(self, raw_img_path: Union[str, Path], seg_img_path: Union[str, Path], 
                patch_size: Tuple[int, int, int], step: int, 
                test_mode: bool = False, normalization: str = 'standardize',
                crop_low_thresh: int = 128):
        """
        Initialize the data loader.
        
        Args:
            raw_img_path: Path to the raw image file
            seg_img_path: Path to the segmentation file
            patch_size: Tuple of (height, width, depth) for patches
            step: Number of patches to generate per iteration
            test_mode: Whether to use test mode for cropping
            normalization: Type of normalization ('standardize', 'normalize', or 'none')
            crop_low_thresh: Minimum crop size for RandomCrop3D
        
        Raises:
            FileNotFoundError: If image files don't exist
            ValueError: If image dimensions don't match
        """
        self.raw_img_path = Path(raw_img_path)
        self.seg_img_path = Path(seg_img_path)
        
        # Validate file existence
        if not self.raw_img_path.exists():
            raise FileNotFoundError(f"Raw image not found: {self.raw_img_path}")
        if not self.seg_img_path.exists():
            raise FileNotFoundError(f"Segmentation image not found: {self.seg_img_path}")
        
        self.patch_size = patch_size
        self.step = step
        self.test_mode = test_mode
        self.normalization = normalization
        self.crop_low_thresh = crop_low_thresh
        
        # Cache image shapes for validation (lightweight operation)
        self._validate_image_compatibility()
        
        logger.info(f"Initialized loader for {self.raw_img_path.name}")

    def _validate_image_compatibility(self) -> None:
        """Validate that raw and segmentation images have compatible dimension and affine."""
        try:
            raw_nifti = nib.load(str(self.raw_img_path))  # type: ignore
            seg_nifti = nib.load(str(self.seg_img_path))  # type: ignore
            
            raw_shape = raw_nifti.shape  # type: ignore
            seg_shape = seg_nifti.shape  # type: ignore

            raw_affine = raw_nifti.affine  # type: ignore
            seg_affine = seg_nifti.affine  # type: ignore
            
            if raw_shape != seg_shape:
                raise ValueError(
                    f"Image shape mismatch: raw {raw_shape} vs seg {seg_shape}"
                )
            
            if not np.allclose(raw_affine, seg_affine, atol=1e-7):
                raise ValueError(
                    f"Image affine mismatch: raw {raw_affine} vs seg {seg_affine}"
                )
            
            self._image_shape = raw_shape
            
        except Exception as e:
            raise ValueError(f"Error validating images: {e}")

    def _load_and_preprocess_images(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load and preprocess images with specified normalization."""
        try:
            # Load images
            raw_nifti = nib.load(str(self.raw_img_path))  # type: ignore
            raw_numpy = raw_nifti.get_fdata()  # type: ignore
            seg_numpy = nib.load(str(self.seg_img_path)).get_fdata()  # type: ignore
            
            # Apply normalization
            if self.normalization == 'standardize':
                raw_numpy = standardiser(raw_numpy)
            elif self.normalization == 'normalize':
                raw_numpy = normaliser(raw_numpy)
            # 'none' means no normalization
            
            return raw_numpy, seg_numpy
            
        except Exception as e:
            raise RuntimeError(f"Error loading images: {e}")

    def __repr__(self) -> str:
        return (f"SingleChannelLoader(\n"
                f"  raw_img: {self.raw_img_path.name}\n"
                f"  seg_img: {self.seg_img_path.name}\n"
                f"  patch_size: {self.patch_size}\n"
                f"  step: {self.step}\n"
                f"  test_mode: {self.test_mode}\n"
                f")")

    def __len__(self) -> int:
        return self.step
        
    def __iter__(self):
        """Generate patches on-demand for memory efficiency."""
        # Load images only when iteration begins
        raw_arr, seg_arr = self._load_and_preprocess_images()
        
        for i in range(self.step):
            try:
                cropper = RandomCrop3D(self._image_shape, self.patch_size, self.test_mode, self.crop_low_thresh)
                img_crop, seg_crop = cropper(raw_arr, seg_arr)
                yield img_crop, seg_crop
            except Exception as e:
                logger.error(f"Error generating patch {i}: {e}")
                raise

    @property
    def image_shape(self) -> Tuple[int, int, int]:
        """Get the shape of the loaded images."""
        return self._image_shape

class MultiChannelDataset:
    """
    Efficient dataset manager for multiple 3D medical images.
    
    This class provides better organization and memory management
    for handling multiple image pairs.
    """
    
    def __init__(self, ps_path: Union[str, Path], seg_path: Union[str, Path], 
                patch_size: Tuple[int, int, int], step: int, 
                test_mode: bool = False, normalization: str = 'standardize',
                crop_low_thresh: int = 128):
        """
        Initialize the multi-channel dataset.
        
        Args:
            ps_path: Path to folder containing processed images
            seg_path: Path to folder containing label images
            patch_size: Tuple of (height, width, depth) for patches
            step: Number of patches to generate per image
            test_mode: Whether to use test mode for cropping
            normalization: Type of normalization to apply
            crop_low_thresh: Minimum crop size for RandomCrop3D
        """
        self.ps_path = Path(ps_path)
        self.seg_path = Path(seg_path)
        self.patch_size = patch_size
        self.step = step
        self.test_mode = test_mode
        self.normalization = normalization
        self.crop_low_thresh = crop_low_thresh
        
        # Find and match image pairs
        self.image_pairs = self._find_image_pairs()
        
        logger.info(f"Found {len(self.image_pairs)} image pairs")

    def _find_image_pairs(self) -> List[Tuple[Path, Path]]:
        """
        Efficiently find matching image and segmentation pairs.
        
        Returns:
            List of (raw_image_path, seg_image_path) tuples
        """
        if not self.ps_path.exists() or not self.seg_path.exists():
            raise FileNotFoundError("Image or segmentation directory not found")
        
        # Get sorted file lists
        raw_files = sorted([f for f in self.ps_path.iterdir() if f.is_file()])
        seg_files = sorted([f for f in self.seg_path.iterdir() if f.is_file()])
        
        if len(raw_files) != len(seg_files):
            raise ValueError(f"Mismatch: {len(raw_files)} raw images vs {len(seg_files)} segmentations")
        
        # Create efficient lookup for segmentation files
        seg_lookup = {f.stem.split('.')[0]: f for f in seg_files}
        
        image_pairs = []
        for raw_file in raw_files:
            raw_stem = raw_file.stem.split('.')[0]
            
            # Find matching segmentation file
            seg_file = None
            for seg_key, seg_path in seg_lookup.items():
                if raw_stem in seg_key or seg_key in raw_stem:
                    seg_file = seg_path
                    break
            
            if seg_file is None:
                raise ValueError(f"No matching segmentation found for {raw_file.name}")
            
            image_pairs.append((raw_file, seg_file))
        
        return image_pairs

    def get_loader(self, index: int) -> SingleChannelLoader:
        """Get a SingleChannelLoader for a specific image pair."""
        if index >= len(self.image_pairs):
            raise IndexError(f"Index {index} out of range for {len(self.image_pairs)} images")
        
        raw_path, seg_path = self.image_pairs[index]
        return SingleChannelLoader(
            raw_path, seg_path, self.patch_size, self.step, 
            self.test_mode, self.normalization, self.crop_low_thresh
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
        return len(self.image_pairs)

    def __getitem__(self, index: int) -> SingleChannelLoader:
        return self.get_loader(index)

    def __repr__(self) -> str:
        return (f"MultiChannelDataset(\n"
                f"  ps_path: {self.ps_path}\n"
                f"  seg_path: {self.seg_path}\n"
                f"  num_images: {len(self.image_pairs)}\n"
                f"  patch_size: {self.patch_size}\n"
                f"  step: {self.step}\n"
                f")")


