"""
Loss functions library for neural network training

This module provides:
- Loss functions for segmentation tasks
- Model factory functions
- Optimizer factory functions  
- Preprocessing utilities

Editor: Marshall Xu
Last Edited: 30/07/2025
"""

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim
import numpy as np
from typing import Optional, Any
from models import Unet, ASPPCNN, CustomSegmentationNetwork, MainArchitecture


def sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Compute sigmoid activation function.
    
    Args:
        z: Input array
        
    Returns:
        Sigmoid activated array

    Raises:
        ValueError: If input is not a numpy array or contains invalid values
    """
    if not isinstance(z, np.ndarray):
        raise ValueError("Input must be a numpy array")

    return 1 / (1 + np.exp(-z))


def normaliser(x: np.ndarray) -> np.ndarray:
    """
    Normalize array to range [0, 1].
    
    Args:
        x: Input numpy array
        
    Returns:
        Normalized array
        
    Raises:
        ValueError: If input is not a numpy array or contains invalid values
    """
    if not isinstance(x, np.ndarray):
        raise ValueError("Input must be a numpy array")
    
    x_min, x_max = np.amin(x), np.amax(x)
    if x_max == x_min:
        return np.zeros_like(x)
    
    return (x - x_min) / (x_max - x_min)


def standardiser(x: np.ndarray) -> np.ndarray:
    """
    Standardize array using z-score normalization (mean=0, std=1).
    
    Args:
        x: Input numpy array
        
    Returns:
        Standardized array
        
    Raises:
        ValueError: If input is not a numpy array or has zero std
    """
    if not isinstance(x, np.ndarray):
        raise ValueError("Input must be a numpy array")
    
    mean_x, std_x = np.mean(x), np.std(x)
    if std_x == 0:
        return np.zeros_like(x)
    
    return (x - mean_x) / std_x


def thresholding(arr: np.ndarray, thresh: float) -> np.ndarray:
    """
    Apply binary thresholding to array with memory-efficient options.
    
    Args:
        arr: Input array
        thresh: Threshold value
        inplace: If True, modify array in-place to save memory. If False, create copy.
        
    Returns:
        Binary thresholded array 

    """
    if not isinstance(arr, np.ndarray):
        raise ValueError("Input must be a numpy array")
    
    return (arr >= thresh).astype(arr.dtype)

def choose_DL_model(model_name: str, in_chan: int, out_chan: int, filter_num: int) -> nn.Module:
    """
    Factory function to create neural network models.
    
    Args:
        model_name: Name of the model ('unet3d', 'aspp', 'test', 'atrous')
        in_chan: Number of input channels
        out_chan: Number of output channels
        filter_num: Number of filters in the model
        
    Returns:
        Initialized model instance
        
    Raises:
        ValueError: If model_name is not supported
    """
    model_registry = {
        "unet3d": lambda: Unet(in_chan, out_chan, filter_num),
        "aspp": lambda: ASPPCNN(in_chan, out_chan, [1, 2, 3, 5, 7]),
        "test": lambda: CustomSegmentationNetwork(),
        "atrous": lambda: MainArchitecture()
    }
    
    if model_name not in model_registry:
        raise ValueError(f"Invalid model name '{model_name}'. "
                        f"Supported models: {list(model_registry.keys())}")
    
    return model_registry[model_name]()


def choose_optimizer(optim_name: str, model_params: Any, lr: float) -> Any:
    """
    Factory function to create optimizers.
    
    Args:
        optim_name: Name of the optimizer ('sgd', 'adam')
        model_params: Model parameters to optimize
        lr: Learning rate
        
    Returns:
        Initialized optimizer instance
        
    Raises:
        ValueError: If optimizer name is not supported
    """
    optimizer_registry = {
        'sgd': lambda: torch.optim.SGD(model_params, lr),
        'adam': lambda: torch.optim.Adam(model_params, lr),
        'adamw': lambda: torch.optim.AdamW(model_params, lr)
    }
    
    if optim_name not in optimizer_registry:
        raise ValueError(f"Invalid optimizer name '{optim_name}'. "
                        f"Supported optimizers: {list(optimizer_registry.keys())}")
    
    return optimizer_registry[optim_name]()


def choose_loss_metric(metric_name: str) -> nn.Module:
    """
    Factory function to create loss functions.
    
    Args:
        metric_name: Name of the loss metric ('bce', 'dice', 'tver', 'combo')
        
    Returns:
        Initialized loss function instance
        
    Raises:
        ValueError: If loss metric name is not supported
    """
    loss_registry = {
        "bce": BCELoss,
        "dice": DiceLoss,
        "tver": TverskyLoss,
        "combo": ComboLoss
    }
    
    if metric_name not in loss_registry:
        raise ValueError(f"Invalid loss metric '{metric_name}'. "
                        f"Supported metrics: {list(loss_registry.keys())}")
    
    return loss_registry[metric_name]()

class DiceLoss(nn.Module):
    """
    Dice Loss for binary segmentation.
    
    The Dice loss is computed as 1 - Dice coefficient, where Dice coefficient
    measures the overlap between predicted and target segmentations.
    """
    
    def __init__(self, smooth: float = 1e-4):
        """
        Initialize Dice Loss.
        
        Args:
            smooth: Smoothing constant to prevent division by zero
        """
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute Dice loss.
        
        Args:
            pred: Predicted segmentation logits
            target: Ground truth binary segmentation
            
        Returns:
            Dice loss value
        """
        # Apply sigmoid to convert logits to probabilities
        pred = torch.sigmoid(pred)
        
        # Flatten tensors for computation
        pred = pred.view(-1)
        target = target.view(-1)

        # Compute Dice coefficient
        intersection = (pred * target).sum()
        dice = (2.0 * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)

        return 1 - dice
    
class DiceCoeff(nn.Module):
    """
    Dice Similarity Coefficient (also known as Sørensen–Dice index).
    
    The Dice similarity coefficient is a statistical tool which measures 
    the similarity between two sets of data.
    """
    
    def __init__(self, delta: float = 0.5, smooth: float = 1e-4):
        """
        Initialize Dice Coefficient.
        
        Args:
            delta: Controls weight given to false positive and false negatives
            smooth: Smoothing constant to prevent division by zero errors
        """
        super().__init__()
        self.delta = delta
        self.smooth = smooth
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute Dice coefficient.
        
        Args:
            pred: Predicted segmentation logits
            target: Ground truth binary segmentation
            
        Returns:
            Dice coefficient value
        """
        pred = torch.sigmoid(pred)
        
        # Flatten tensors
        pred = pred.reshape(-1)
        target = target.reshape(-1)
        
        # Compute confusion matrix components
        tp = (pred * target).sum()
        fp = ((1 - target) * pred).sum()
        fn = (target * (1 - pred)).sum()

        # Compute generalized Dice coefficient
        dice_score = (tp + self.smooth) / (
            tp + self.delta * fn + (1 - self.delta) * fp + self.smooth
        )

        return dice_score

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance in binary classification.
    
    Reference: Lin, T. Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017).
    Focal loss for dense object detection.
    """
    
    def __init__(self, alpha: float = 0.8, gamma: float = 2.0, smooth: float = 1e-6):
        """
        Initialize Focal Loss.
        
        Args:
            alpha: Weighting factor for rare class (between 0 and 1)
            gamma: Focusing parameter (higher gamma puts more focus on hard examples)
            smooth: Smoothing constant
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute Focal Loss.
        
        Args:
            pred: Predicted segmentation logits
            target: Ground truth binary segmentation
            
        Returns:
            Focal loss value
        """
        pred = torch.sigmoid(pred)

        # Flatten tensors
        pred = pred.reshape(-1)
        target = target.reshape(-1)

        # Compute binary cross-entropy
        bce = F.binary_cross_entropy(pred, target, reduction='none')
        
        # Compute focal loss components
        pt = torch.where(target == 1, pred, 1 - pred)
        focal_weight = (1 - pt) ** self.gamma
        
        # Apply alpha weighting
        alpha_weight = torch.where(target == 1, self.alpha, 1 - self.alpha)
        
        focal_loss = alpha_weight * focal_weight * bce

        return focal_loss.mean()

class BCELoss(nn.Module):
    """
    Binary Cross-Entropy Loss with logits for numerical stability.
    
    This implementation uses BCEWithLogitsLoss internally for better
    numerical stability compared to applying sigmoid + BCE separately.
    """
    
    def __init__(self, pos_weight: Optional[torch.Tensor] = None):
        """
        Initialize BCE Loss.
        
        Args:
            pos_weight: Weight for positive examples (useful for class imbalance)
        """
        super().__init__()
        self.pos_weight = pos_weight
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute Binary Cross-Entropy Loss.
        
        Args:
            pred: Predicted segmentation logits (before sigmoid)
            target: Ground truth binary segmentation
            
        Returns:
            BCE loss value
        """
        batch_size = target.size(0)

        # Flatten pred and target tensors
        pred = pred.view(batch_size, -1)
        target = target.view(batch_size, -1)
        
        # Use BCEWithLogitsLoss for numerical stability
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight, reduction='mean')
        return loss_fn(pred, target)
    
class TverskyLoss(nn.Module):
    """
    Tversky Loss for binary segmentation.
    
    The Tversky index is a generalization of the Dice coefficient that allows
    for different weighting of false positives and false negatives.
    
    Reference: Tversky, A. (1977). Features of similarity.
    """
    
    def __init__(self, alpha: float = 0.3, beta: float = 0.7, smooth: float = 1e-3):
        """
        Initialize Tversky Loss.
        
        Args:
            alpha: Weight for false positives (typically < 0.5 to focus on recall)
            beta: Weight for false negatives (typically > 0.5 to focus on recall)
            smooth: Smoothing constant to prevent division by zero

        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute Tversky Loss.
        
        Args:
            pred: Predicted segmentation logits
            target: Ground truth binary segmentation
            
        Returns:
            Tversky loss value
        """
        pred = torch.sigmoid(pred)
        
        # Flatten tensors
        pred = pred.reshape(-1)
        target = target.reshape(-1)

        # Compute confusion matrix components
        tp = (pred * target).sum()
        fp = ((1 - target) * pred).sum()
        fn = (target * (1 - pred)).sum()

        # Compute Tversky index
        tversky_score = (tp + self.smooth) / (
            tp + self.alpha * fp + self.beta * fn + self.smooth
        )

        return 1 - tversky_score
    
class ComboLoss(nn.Module):
    """
    Combo Loss: Handling Input and Output Imbalance in Multi-Organ Segmentation.
    
    Combines Dice loss and Binary Cross-Entropy loss for better handling of
    class imbalance in segmentation tasks.
    
    Reference: Taghanaki, S. A., et al. (2019). Combo loss: Handling input 
    and output imbalance in multi-organ segmentation. Computerized Medical 
    Imaging and Graphics, 75, 24-33.
    """
    
    def __init__(self, alpha: float = 0.5, beta: Optional[float] = None):
        """
        Initialize Combo Loss.
        
        Args:
            alpha: Controls weighting of dice and cross-entropy loss (0.0 to 1.0)
            beta: Controls penalty for false negatives vs false positives.
                If None, standard BCE is used. If provided, weighted BCE is used.
                beta > 0.5 penalizes false negatives more than false positives.
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute Combo Loss.
        
        Args:
            pred: Predicted segmentation logits
            target: Ground truth binary segmentation
            
        Returns:
            Combo loss value
        """
        # Compute Dice coefficient (higher is better)
        dice = DiceCoeff()(pred, target)
        
        # Compute BCE loss
        if self.beta is not None:
            # Use weighted BCE
            pos_weight = torch.tensor([self.beta / (1 - self.beta)], 
                                    device=pred.device, dtype=pred.dtype)
            bce_loss = BCELoss(pos_weight=pos_weight)(pred, target)
        else:
            # Use standard BCE
            bce_loss = BCELoss()(pred, target)

        # Combine losses: BCE (minimize) and negative Dice (maximize Dice)
        combo_loss = self.alpha * bce_loss - (1 - self.alpha) * dice
        
        return combo_loss
    