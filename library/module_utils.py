"""
Image processing utilities for preprocessing, prediction, and postprocessing.

This module provides:
- Image preprocessing with bias field correction and denoising
- Neural network inference on 3D images
- Postprocessing with thresholding and connected component analysis

Editor: Marshall Xu
Last edited: 31/07/2025
"""

import numpy as np
import ants
import torch
import nibabel as nib  # type: ignore
from tqdm import tqdm
import scipy.ndimage as scind
import matplotlib.pyplot as plt
from patchify import patchify, unpatchify
import cc3d
from typing import Tuple, Optional, Union, List, Any
from pathlib import Path
import logging

from .loss_func import choose_DL_model, normaliser, standardiser
from models import *

# Set up logging
logger = logging.getLogger(__name__)

class ImagePreprocessor:
    """
    Image preprocessing with bias field correction and denoising.

    This class handles preprocessing of multiple images using ANTs library
    for bias field correction and non-local means denoising.
    
    Args:
        input_path: Path to directory containing raw images
        output_path: Path to directory for saving processed images
        
    Raises:
        FileNotFoundError: If input_path doesn't exist
        ValueError: If paths are invalid
    """

    def __init__(self, input_path: Union[str, Path], output_path: Union[str, Path]):
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        
        # Validate input path
        if not self.input_path.exists():
            raise FileNotFoundError(f"Input path does not exist: {self.input_path}")
        
        # Create output directory if it doesn't exist
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Preprocessor initialized: {self.input_path} -> {self.output_path}")

    def _get_image_files(self) -> List[Path]:
        """Get list of image files from input directory."""
        # Common medical image extensions
        extensions = {'.nii', '.nii.gz', '.mgz', '.mgh'}
        files = [f for f in self.input_path.iterdir() 
                if f.is_file() and f.suffix in extensions]
        
        if not files:
            logger.warning(f"No medical image files found in {self.input_path}")
        
        return sorted(files)

    def _process_single_image(self, image_path: Path, mode: int) -> None:
        """
        Process a single image with specified preprocessing mode.
        
        Args:
            image_path: Path to input image
            mode: Preprocessing mode (1=BFC only, 2=denoising only, 3=both, 4=abort)
        """
        try:
            # Load image
            test_img = nib.load(str(image_path))  # type: ignore
            header = test_img.header
            affine = test_img.affine  # type: ignore

            # Convert to ANTs format
            ant_img = ants.utils.convert_nibabel.from_nibabel(test_img)
            ant_mask = ants.utils.get_mask(
                ant_img, 
                low_thresh=ant_img.min(), 
                high_thresh=ant_img.max()
            )  # type: ignore

            # Apply preprocessing based on mode
            if mode == 1:
                # Bias field correction only
                ant_img = ants.utils.n4_bias_field_correction(image=ant_img, mask=ant_mask)
            elif mode == 2:
                # Non-local denoising only
                ant_img = ants.utils.denoise_image(image=ant_img, mask=ant_mask)  # type: ignore
            elif mode == 3:
                # Both bias field correction and denoising
                ant_img = ants.utils.n4_bias_field_correction(image=ant_img, mask=ant_mask)
                ant_img = ants.utils.denoise_image(image=ant_img, mask=ant_mask)  # type: ignore
            else:
                raise ValueError(f"Invalid preprocessing mode: {mode}")

            # Convert back to numpy and save
            processed_array = ant_img.numpy()
            processed_nifti = nib.Nifti1Image(processed_array, affine, header)  # type: ignore

            output_path = self.output_path / image_path.name
            nib.save(processed_nifti, str(output_path))  # type: ignore
            
        except Exception as e:
            logger.error(f"Failed to process {image_path.name}: {e}")
            raise

    def process_images(self, mode: int) -> None:
        """
        Process all images in the input directory.
        
        Args:
            mode: Preprocessing mode
                1: Bias field correction only
                2: Non-local denoising only  
                3: Bias field correction + denoising
                4: Abort processing
                
        Raises:
            ValueError: If mode is invalid
        """
        if mode == 4:
            logger.info("Preprocessing aborted by user")
            return
        
        if mode not in [1, 2, 3]:
            raise ValueError(f"Invalid preprocessing mode: {mode}. Valid modes: 1, 2, 3, 4")
        
        logger.info("Starting preprocessing procedure")
        
        image_files = self._get_image_files()
        if not image_files:
            logger.warning("No image files found to process")
            return
        
        # Process each image with progress bar
        for image_path in tqdm(image_files, desc="Processing images"):
            self._process_single_image(image_path, mode)
        
        logger.info(f"Successfully processed {len(image_files)} images")

    def __call__(self, mode: int) -> None:
        """Make the class callable for backward compatibility."""
        self.process_images(mode)


class ImagePredictor:
    """
    Neural network prediction and postprocessing.
    
    This class handles:
    - Loading and preprocessing of 3D images
    - Patch-based neural network inference
    - Postprocessing with thresholding and connected component analysis
    - Saving results in various formats
    
    Args:
        model_name: Type of neural network model to use
        input_channel: Number of input channels for the model
        output_channel: Number of output channels for the model  
        filter_number: Number of filters in the model
        input_path: Path to directory containing preprocessed images
        output_path: Path to directory for saving predictions
        
    Raises:
        FileNotFoundError: If input_path doesn't exist
        ValueError: If model parameters are invalid
    """
    
    def __init__(
        self, 
        model_name: str, 
        input_channel: int, 
        output_channel: int, 
        filter_number: int,
        input_path: Union[str, Path], 
        output_path: Union[str, Path]
    ):
        # Validate parameters
        if input_channel <= 0 or output_channel <= 0 or filter_number <= 0:
            raise ValueError("Channel and filter numbers must be positive")
        
        self.model_name = model_name
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.filter_number = filter_number
        
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        
        # Validate paths
        if not self.input_path.exists():
            raise FileNotFoundError(f"Input path does not exist: {self.input_path}")
        
        # Create output directory
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Device selection
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

    def _calculate_patch_dimensions(self, original_size: Tuple[int, int, int], patch_size: int = 64) -> Tuple[int, int, int]:
        """
        Calculate optimal dimensions for patch-based processing.
        
        Args:
            original_size: Original image dimensions (w, h, d)
            patch_size: Size of each patch (default 64)
            
        Returns:
            Tuple of new dimensions that are divisible by patch_size
        """
        new_dims = []
        for dim in original_size:
            if dim > patch_size and dim % patch_size != 0:
                new_dim = int(np.ceil(dim / patch_size)) * patch_size
            elif dim < patch_size:
                new_dim = patch_size
            else:
                new_dim = dim
            new_dims.append(new_dim)
        
        return tuple(new_dims)

    def _resize_image(self, image: np.ndarray, target_size: Tuple[int, int, int]) -> np.ndarray:
        """
        Resize image to target dimensions using scipy zoom.
        
        Args:
            image: Input image array
            target_size: Target dimensions
            
        Returns:
            Resized image array
        """
        original_size = image.shape
        zoom_factors = tuple(target / original for target, original in zip(target_size, original_size))
        return scind.zoom(image, zoom_factors, order=0, mode='nearest')

    def _run_inference(self, patches: np.ndarray, model: torch.nn.Module, original_size: Tuple[int, int, int]) -> np.ndarray:
        """
        Run neural network inference on image patches.
        
        Args:
            patches: Patchified image array
            model: Loaded neural network model
            original_size: Original image dimensions for unpatchifying
            
        Returns:
            Predicted probability map
        """
        logger.info("Starting prediction procedure")
        
        model.eval()
        sigmoid_fn = torch.nn.Sigmoid()
        
        # Process each patch
        with torch.no_grad():
            for i in tqdm(range(patches.shape[0]), desc="Processing patches"):
                for j in range(patches.shape[1]):
                    for k in range(patches.shape[2]):
                        # Extract and prepare single patch
                        single_patch = patches[i, j, k, :, :, :]
                        
                        # Convert to tensor with proper shape: (batch, channel, depth, height, width)
                        patch_tensor = torch.from_numpy(single_patch).float().unsqueeze(0).unsqueeze(0)
                        patch_tensor = patch_tensor.to(self.device)
                        
                        # Run inference
                        prediction = model(patch_tensor)
                        prediction = sigmoid_fn(prediction)
                        
                        # Convert back to numpy and store
                        prediction_np = prediction.cpu().numpy()[0, 0, :, :, :]
                        patches[i, j, k, :, :, :] = prediction_np

        # Reconstruct full image
        prediction_output = unpatchify(patches, original_size)
        logger.info("Prediction procedure completed")
        
        return prediction_output

    def _apply_postprocessing(self, probability_map: np.ndarray, threshold: float, connect_threshold: int) -> np.ndarray:
        """
        Apply thresholding and connected component analysis.
        
        Args:
            probability_map: Model prediction probabilities
            threshold: Probability threshold for binarization
            connect_threshold: Minimum component size to keep (in voxels)
            
        Returns:
            Binary segmentation mask
        """
        # Threshold probabilities to binary
        binary_mask = (probability_map >= threshold).astype(np.int32)
        
        # Remove small connected components in-place for memory efficiency
        cc3d.dust(binary_mask, connect_threshold, connectivity=26, in_place=True)
        return binary_mask

    def _save_results(
        self, 
        image_name: str, 
        probability_map: np.ndarray, 
        binary_mask: np.ndarray,
        affine: np.ndarray, 
        header: Any,
        save_probability: bool = False,
        save_mip: bool = False
    ) -> None:
        """
        Save prediction results in various formats.
        
        Args:
            image_name: Name of the processed image
            probability_map: Continuous probability predictions
            binary_mask: Binary segmentation mask
            affine: NIfTI affine transformation matrix
            header: NIfTI header information
            save_probability: Whether to save probability map
            save_mip: Whether to save maximum intensity projection
        """
        # Save probability map if requested
        if save_probability:
            prob_image = nib.Nifti1Image(probability_map, affine, header)  # type: ignore
            prob_path = self.output_path / f"SIGMOID_{image_name}"
            nib.save(prob_image, str(prob_path))  # type: ignore
            logger.info(f"Saved probability map: {prob_path}")

        # Save binary segmentation
        binary_image = nib.Nifti1Image(binary_mask, affine, header)  # type: ignore
        binary_path = self.output_path / image_name
        nib.save(binary_image, str(binary_path))  # type: ignore
        logger.info(f"Saved binary segmentation: {binary_path}")

        # Save MIP if requested
        if save_mip:
            mip = np.max(binary_mask, axis=2)
            mip = np.rot90(mip, axes=(0, 1))  # Rotate 90 degrees counterclockwise
            
            mip_path = self.output_path / f"{Path(image_name).stem}.jpg"
            plt.imsave(str(mip_path), mip, cmap='gray')
            logger.info(f"Saved MIP image: {mip_path}")

    def process_single_image(
        self, 
        image_name: str, 
        model: torch.nn.Module, 
        threshold: float,
        connect_threshold: int, 
        save_mip: bool = False, 
        save_probability: bool = False
    ) -> None:
        """
        Process a single image through the complete pipeline.
        
        Args:
            image_name: Name of the image file to process
            model: Loaded neural network model
            threshold: Probability threshold for binarization
            connect_threshold: Minimum component size to keep
            save_mip: Whether to save maximum intensity projection
            save_probability: Whether to save probability map
            
        Raises:
            FileNotFoundError: If image file doesn't exist
            RuntimeError: If processing fails
        """
        image_path = self.input_path / image_name
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        try:
            # Load image
            raw_image = nib.load(str(image_path))  # type: ignore
            header = raw_image.header
            affine = raw_image.affine  # type: ignore
            image_array = raw_image.get_fdata()  # type: ignore

            original_size = image_array.shape
            
            # Calculate optimal patch dimensions and resize
            target_size = self._calculate_patch_dimensions(original_size)
            resized_image = self._resize_image(image_array, target_size)
            
            # Standardize image
            standardized_image = standardiser(resized_image)
            
            # Create patches
            patches = patchify(standardized_image, (64, 64, 64), 64)
            
            # Run inference
            prediction_map = self._run_inference(patches, model, target_size)
            
            # Resize back to original dimensions
            prediction_map = self._resize_image(prediction_map, original_size)
            
            # Apply postprocessing
            binary_mask = self._apply_postprocessing(prediction_map, threshold, connect_threshold)
            
            # Save results
            self._save_results(
                image_name, prediction_map, binary_mask, affine, header,
                save_probability, save_mip
            )
            
        except Exception as e:
            logger.error(f"Failed to process {image_name}: {e}")
            raise RuntimeError(f"Processing failed for {image_name}: {e}")

    def predict_all_images(
        self,
        model_path: Union[str, Path],
        threshold: float,
        connect_threshold: int,
        save_mip: bool = False,
        save_probability: bool = False
    ) -> None:
        """
        Process all images in the input directory.
        
        Args:
            model_path: Path to the trained model file
            threshold: Probability threshold for binarization
            connect_threshold: Minimum component size to keep
            save_mip: Whether to save maximum intensity projections
            save_probability: Whether to save probability maps
        """
        # Load model
        model = choose_DL_model(
            self.model_name, self.input_channel, 
            self.output_channel, self.filter_number
        ).to(self.device)
        
        # Load model weights
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        if self.device.type == "cuda":
            model.load_state_dict(torch.load(str(model_path)))  # type: ignore
        else:
            model.load_state_dict(torch.load(str(model_path), map_location=self.device))  # type: ignore
        
        logger.info(f"Loaded model from: {model_path}")
        
        # Get list of images to process
        image_files = [f for f in self.input_path.iterdir() if f.is_file() and f.suffix in {'.nii', '.nii.gz'}]
        
        if not image_files:
            logger.warning(f"No image files found in {self.input_path}")
            return
        
        # Process each image
        for image_file in image_files:
            try:
                self.process_single_image(
                    image_file.name, model, threshold, connect_threshold,
                    save_mip, save_probability
                )
            except Exception as e:
                logger.error(f"Skipping {image_file.name} due to error: {e}")
                continue
        
        logger.info(f"Completed processing {len(image_files)} images")

    def __call__(
        self, 
        threshold: float, 
        connect_threshold: int, 
        model_path: str, 
        image_name: str,
        save_mip: bool, 
        save_probability: bool = False
    ) -> None:
        """Make the class callable for backward compatibility."""
        # Load model
        model = choose_DL_model(
            self.model_name, self.input_channel, 
            self.output_channel, self.filter_number
        ).to(self.device)
        
        if self.device.type == "cuda":
            logger.info("Running with GPU")
            model.load_state_dict(torch.load(model_path))  # type: ignore
        else:
            logger.info("Running with CPU")
            model.load_state_dict(torch.load(model_path, map_location=self.device))  # type: ignore
        
        self.process_single_image(
            image_name, model, threshold, connect_threshold, 
            save_mip, save_probability
        )
        logger.info("Prediction and thresholding procedure completed")

def preprocess_procedure(ds_path: Union[str, Path], ps_path: Union[str, Path], prep_mode: int) -> None:
    """
    Preprocesses medical images with bias field correction and/or denoising.

    Args:
        ds_path: Path to the input dataset directory
        ps_path: Path to the preprocessed data storage directory
        prep_mode: Preprocessing mode (1=BFC, 2=denoising, 3=both, 4=abort)

    Raises:
        FileNotFoundError: If dataset path doesn't exist
        ValueError: If preprocessing mode is invalid
    """
    try:
        # Initialize preprocessing with input/output paths
        preprocessor = ImagePreprocessor(ds_path, ps_path)
        # Start or abort preprocessing
        preprocessor.process_images(prep_mode)
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        raise


def make_prediction(
    model_name: str, 
    input_channel: int, 
    output_channel: int,
    filter_number: int, 
    input_path: Union[str, Path], 
    output_path: Union[str, Path],
    thresh: float, 
    connect_thresh: int, 
    test_model_name: Union[str, Path],
    mip_flag: bool
) -> None:
    """
    Performs neural network prediction on all images in a directory.

    Args:
        model_name: Name of the neural network model
        input_channel: Number of input channels
        output_channel: Number of output channels
        filter_number: Number of filters in the model
        input_path: Path to preprocessed input images
        output_path: Path for saving predictions
        thresh: Probability threshold for binarization
        connect_thresh: Minimum connected component size
        test_model_name: Path to the trained model file
        mip_flag: Whether to save maximum intensity projections

    Raises:
        FileNotFoundError: If input path or model file doesn't exist
        ValueError: If model parameters are invalid
    """
    try:
        # Initialize predictor with model configuration and paths
        predictor = ImagePredictor(
            model_name, input_channel, output_channel, 
            filter_number, input_path, output_path
        )
        
        # Process all images in the input directory
        predictor.predict_all_images(
            model_path=test_model_name,
            threshold=thresh,
            connect_threshold=connect_thresh,
            save_mip=mip_flag
        )
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise




