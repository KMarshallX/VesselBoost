"""
Training utilities.

This module provides:
- TTA training implementation
- Cross-validation training support
- Model initialization and optimization utilities
- Resource management for training workflows

Editor: Marshall Xu
Last edited: 31/07/2025
"""

import time
import shutil
import torch
import torchio as tio
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
import logging

from .loss_func import choose_DL_model, choose_optimizer, choose_loss_metric
from .aug_utils import AugmentationUtils, TorchIOAugmentationUtils
from .eval_utils import CrossValidationHelper
from .data_loaders import SingleChannelLoader, MultiChannelLoader
from .module_utils import ImagePredictor

# Set up logging
logger = logging.getLogger(__name__)

class Trainer:
    """
    Instance for training models and TTA process.
    
    This class implements training workflows for DL segmentation models
    with support for test-time adaptation, cross-validation, and resource optimization.
    
    Args:
        loss_name: Name of the loss function to use
        model_name: Name of the neural network architecture
        input_channels: Number of input channels
        output_channels: Number of output channels
        filter_count: Number of filters in the model
        optimizer_name: Name of the optimizer ('adam', 'sgd')
        learning_rate: Learning rate for training
        optimizer_gamma: Learning rate decay factor
        num_epochs: Number of training epochs
        batch_multiplier: Batch size multiplier for data loading
        patch_size: Size of image patches for training
        augmentation_mode: Augmentation strategy ('on', 'off', etc.)
        pretrained_model_path: Path to pretrained model (optional)
        threshold: Probability threshold for binarization (optional)
        connect_threshold: Minimum connected component size (optional)
        test_mode: Whether to run in test mode (disables some augmentations)
        crop_low_thresh: Minimum crop size for RandomCrop3D (optional)
        
    Raises:
        ValueError: If required parameters are invalid
        FileNotFoundError: If pretrained model path doesn't exist
    """
    
    def __init__(
        self,
        loss_name: str,
        model_name: str,
        input_channels: int,
        output_channels: int,
        filter_count: int,
        optimizer_name: str,
        learning_rate: float,
        optimizer_gamma: float,
        num_epochs: int,
        batch_multiplier: int,
        patch_size: Tuple[int, int, int],
        augmentation_mode: str,
        pretrained_model_path: Optional[Union[str, Path]] = None,
        threshold: Optional[float] = None,
        connect_threshold: Optional[int] = None,
        test_mode: bool = False,
        crop_low_thresh: int = 128
    ):
        # Validate inputs
        if input_channels <= 0 or output_channels <= 0 or filter_count <= 0:
            raise ValueError("Channel and filter counts must be positive")
        if learning_rate <= 0 or num_epochs <= 0:
            raise ValueError("Learning rate and epochs must be positive")
        if batch_multiplier <= 0:
            raise ValueError("Batch multiplier must be positive")
        
        # Core model configuration
        self.loss_name = loss_name
        self.model_name = model_name
        self.model_config = [input_channels, output_channels, filter_count]
        
        # Training configuration
        self.optimizer_name = optimizer_name
        self.learning_rate = learning_rate
        self.optimizer_gamma = optimizer_gamma
        self.num_epochs = num_epochs
        self.batch_multiplier = batch_multiplier
        
        # Data configuration
        self.patch_size = patch_size if isinstance(patch_size, tuple) else (patch_size, patch_size, patch_size)
        self.augmentation_mode = augmentation_mode
        self.test_mode = test_mode
        
        # Postprocessing configuration
        self.threshold = threshold
        self.connect_threshold = connect_threshold
        
        # RandomCrop3D configuration
        self.crop_low_thresh = crop_low_thresh
        
        # Model path validation
        if pretrained_model_path is not None:
            self.pretrained_model_path = Path(pretrained_model_path)
            if not self.pretrained_model_path.exists():
                raise FileNotFoundError(f"Pretrained model not found: {self.pretrained_model_path}")
        else:
            self.pretrained_model_path = None
        
        # Hardware configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Trainer initialized on device: {self.device}")

    def _initialize_loss(self) -> torch.nn.Module:
        """Initialize and return the loss function."""
        return choose_loss_metric(self.loss_name)

    def _initialize_model(self) -> torch.nn.Module:
        """Initialize and return the model."""
        return choose_DL_model(
            self.model_name, 
            self.model_config[0], 
            self.model_config[1], 
            self.model_config[2]
        ).to(self.device)

    def _initialize_scheduler(self, optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler.ReduceLROnPlateau: # type: ignore
        """Initialize learning rate scheduler with patience based on epoch count."""
        patience = int(np.ceil(self.num_epochs * 0.2))
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            factor=self.optimizer_gamma, 
            patience=patience
        )

    def _initialize_augmentation(self) -> TorchIOAugmentationUtils:
        """Initialize augmentation utilities based on test mode."""
        return TorchIOAugmentationUtils(self.augmentation_mode)

    def _load_pretrained_model(self) -> torch.nn.Module:
        """Load and return a pretrained model."""
        if self.pretrained_model_path is None:
            raise ValueError("Running TTA, but no pretrained model path specified")

        model = self._initialize_model()
        
        try:
            if self.device.type == "cuda":
                logger.info("Loading pretrained model on GPU")
                model.load_state_dict(torch.load(str(self.pretrained_model_path)))
            else:
                logger.info("Loading pretrained model on CPU")
                model.load_state_dict(
                    torch.load(str(self.pretrained_model_path), map_location=self.device)
                )
            
            model.eval()
            logger.info(f"Loaded pretrained model: {self.pretrained_model_path}")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load pretrained model: {e}")
            raise RuntimeError(f"Could not load pretrained model: {e}")

    def _train_epoch(
        self, 
        data_loaders: Dict[int, Any], 
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer, # type: ignore
        scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
        loss_function: torch.nn.Module,
        augmentation: TorchIOAugmentationUtils
    ) -> Tuple[float, float]:
        """
        Train model for one epoch.
        
        Returns:
            Tuple of (average_loss, average_learning_rate)
        """
        model.train()
        total_loss = 0.0
        total_lr = 0.0
        num_batches = 0

        data_loading_time = 0.0 #TEST
        model_training_time = 0.0 #TEST
        
        iterators = {idx: iter(loader) for idx, loader in data_loaders.items()}
        
        for file_idx in range(len(data_loaders)):
            start_data = time.time()  # TEST
            
            # OPTIMIZED: Use pre-created iterator
            image_batch, label_batch = next(iterators[file_idx])
            # Apply augmentations
            image_batch, label_batch = augmentation(image_batch, label_batch)
            # Move to device (OPTIMIZED: non_blocking transfer for speed)
            image_batch = image_batch.to(self.device, non_blocking=True)
            label_batch = label_batch.to(self.device, non_blocking=True)
            data_loading_time += time.time() - start_data  # TEST
            
            # Forward pass
            start_model = time.time()  # TEST
            optimizer.zero_grad()
            output = model(image_batch)
            loss = loss_function(output, label_batch)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Update scheduler
            scheduler.step(loss)
            model_training_time += time.time() - start_model  # TEST

            # Track metrics
            total_loss += loss.item()
            total_lr += optimizer.param_groups[0]['lr']
            num_batches += 1

        logger.info(f"\nData loading time: {data_loading_time:.2f}s, Model training time: {model_training_time:.2f}s")  # TEST
        return total_loss / num_batches, total_lr / num_batches

    def train_model(
        self, 
        data_loaders: Dict[int, Any], 
        model: torch.nn.Module, 
        save_path: Union[str, Path]
    ) -> None:
        """
        Execute the complete training loop.
        
        Args:
            data_loaders: Dictionary of data loaders
            model: Model to train
            save_path: Path to save the trained model
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize training components
        optimizer = choose_optimizer(self.optimizer_name, model.parameters(), self.learning_rate)
        scheduler = self._initialize_scheduler(optimizer)
        loss_function = self._initialize_loss()
        augmentation = self._initialize_augmentation()
        
        logger.info(f"Starting training for {self.num_epochs} epochs")
        
        # Training loop
        for epoch in tqdm(range(self.num_epochs), desc="Training"):
            avg_loss, avg_lr = self._train_epoch(
                data_loaders, model, optimizer, scheduler, loss_function, augmentation
            )
            
            # Log progress
            tqdm.write(
                f'Epoch [{epoch+1}/{self.num_epochs}], '
                f'Loss: {avg_loss:.8f}, LR: {avg_lr:.8f}'
            )
        
        # Save model
        torch.save(model.state_dict(), str(save_path))
        logger.info(f"Model saved to: {save_path}")

    def train(
        self, 
        processed_path: Union[str, Path], 
        segmentation_path: Union[str, Path], 
        output_model_path: Union[str, Path]
    ) -> None:
        """
        Train a model from scratch.
        
        Args:
            processed_path: Path to preprocessed images
            segmentation_path: Path to segmentation masks
            output_model_path: Path to save the trained model
        """
        # Initialize dataset and loaders
        dataset = MultiChannelLoader(
            processed_path, segmentation_path, 
            self.patch_size, self.num_epochs, 
            crop_low_thresh=self.crop_low_thresh,
            batch_multiplier=self.batch_multiplier
        )
        data_loaders = dataset.get_all_loaders()
        
        # Initialize model
        model = self._initialize_model()

        logger.info(f"Training with effective batch size: {self.batch_multiplier + 1}")

        # Execute training
        self.train_model(data_loaders, model, output_model_path)

    def cross_validation_train(
        self, 
        processed_path: Union[str, Path], 
        segmentation_path: Union[str, Path], 
        model_output_dir: Union[str, Path]
    ) -> None:
        """
        Perform cross-validation training.
        
        Args:
            processed_path: Path to preprocessed images
            segmentation_path: Path to segmentation masks
            model_output_dir: Directory to save CV models
        """
        model_output_dir = Path(model_output_dir)
        model_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate cross-validation splits
        cv_helper = CrossValidationHelper(processed_path)
        cv_splits = cv_helper.generate_cv_splits()
        
        logger.info(f"Performing {len(cv_splits)}-fold cross-validation")
        
        for fold_idx, (test_file, train_files) in enumerate(cv_splits.items(), 1):
            logger.info(f"CV Fold {fold_idx}: Test file = {test_file}")
            
            # Initialize dataset
            dataset = MultiChannelLoader(
                processed_path, segmentation_path, 
                self.patch_size, self.num_epochs,
                crop_low_thresh=self.crop_low_thresh,
                batch_multiplier=self.batch_multiplier
            )
            
            # Get training indices for this fold
            train_indices = []
            for i, (raw_path, _) in enumerate(dataset.image_pairs):
                if raw_path.name in train_files:
                    train_indices.append(i)
            
            data_loaders = dataset.get_cv_loaders(train_indices)
            
            # Initialize fresh model for this fold
            model = self._initialize_model()
            
            # Define output path for this fold
            test_name = Path(test_file).stem
            output_path = model_output_dir / f"cv_{fold_idx}_{test_name}"
            
            # Train model for this fold
            self.train_model(data_loaders, model, output_path)

    def test_time_adaptation(
        self,
        processed_path: Union[str, Path],
        proxy_path: Union[str, Path],
        output_path: Union[str, Path],
        model_output_dir: Union[str, Path],
        resource_optimization: int = 0
    ) -> None:
        """
        Perform test-time adaptation on processed images.
        
        Args:
            processed_path: Path to preprocessed images
            proxy_path: Path to store proxy segmentations
            output_path: Path for final predictions
            model_output_dir: Directory for adapted models
            resource_optimization: 0=keep files, 1=clean up intermediate files
        """
        if self.pretrained_model_path is None:
            raise ValueError("Pretrained model required for test-time adaptation")
        
        if self.threshold is None or self.connect_threshold is None:
            raise ValueError("Threshold values missing!")
        
        processed_path = Path(processed_path)
        proxy_path = Path(proxy_path)
        output_path = Path(output_path)
        model_output_dir = Path(model_output_dir)
        
        # Create output directories
        for path in [proxy_path, output_path, model_output_dir]:
            path.mkdir(parents=True, exist_ok=True)
        
        processed_files = [f for f in processed_path.iterdir() if f.is_file()]
        logger.info(f"Processing {len(processed_files)} images for test-time adaptation")
        
        for processed_file in processed_files:
            file_stem = processed_file.name.split(".")[0]
            logger.info(f"Processing: {processed_file.name}")
            
            # Generate proxy if needed
            proxy_files = list(proxy_path.glob("*"))
            if len(proxy_files) != len(processed_files):
                logger.info("Generating proxy segmentations...")
                predictor = ImagePredictor(
                    self.model_name, 
                    self.model_config[0], 
                    self.model_config[1], 
                    self.model_config[2], 
                    processed_path, 
                    proxy_path
                )
                predictor(
                    self.threshold, self.connect_threshold, 
                    str(self.pretrained_model_path), processed_file.name, 
                    save_mip=False
                )
            
            # Find corresponding proxy file
            proxy_file = self._find_corresponding_proxy(processed_file, proxy_path)
            if proxy_file is None:
                logger.error(f"No proxy found for {processed_file.name}")
                continue
            
            logger.info("Found proxy file, starting fine-tuning...")
            
            # Initialize single-image data loader for adaptation
            data_loader = {0: SingleChannelLoader(
                str(processed_file), str(proxy_file), 
                self.patch_size, step=self.num_epochs,
                crop_low_thresh=self.crop_low_thresh,
                batch_multiplier=self.batch_multiplier
            )}
            
            # Load pretrained model
            model = self._load_pretrained_model()
            
            # Fine-tune model
            adapted_model_path = model_output_dir / file_stem
            self.train_model(data_loader, model, adapted_model_path)
            
            # Generate final prediction
            logger.info(f"Generating final prediction for {file_stem}")
            predictor = ImagePredictor(
                self.model_name, 
                self.model_config[0], 
                self.model_config[1], 
                self.model_config[2], 
                processed_path, 
                output_path
            )
            predictor.predict_all_images(
                model_path=adapted_model_path, 
                threshold=self.threshold,
                connect_threshold=self.connect_threshold, 
                save_mip=True
            )
        
        logger.info("Test-time adaptation completed")
        
        # Handle resource optimization
        if resource_optimization == 1:
            logger.info("Cleaning up intermediate files...")
            shutil.rmtree(proxy_path)
            shutil.rmtree(model_output_dir)
            logger.info("Intermediate files cleaned up")
        else:
            logger.info(f"Intermediate files preserved:")
            logger.info(f"  Adapted models: {model_output_dir}")
            logger.info(f"  Proxy segmentations: {proxy_path}")

    def _find_corresponding_proxy(self, processed_file: Path, proxy_path: Path) -> Optional[Path]:
        """Find the proxy file corresponding to a processed image."""
        file_stem = processed_file.stem
        proxy_files = list(proxy_path.iterdir())
        
        for proxy_file in proxy_files:
            if file_stem in proxy_file.name:
                return proxy_file
        
        return None
