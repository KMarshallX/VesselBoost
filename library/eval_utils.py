"""
Helper functions library for evaluation

Editor: Marshall Xu
Last Edited: 30/07/2025
"""

import re
import warnings
from typing import List, Dict, Tuple, Union, Optional
from pathlib import Path
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import logging

# Set up logging
logger = logging.getLogger(__name__)

class LossFileAnalyzer:
    """
    Enhanced class for reading and analyzing loss values from log files.
    
    This class provides better organization and additional functionality
    compared to the original out_file_reader function.
    """
    
    def __init__(self, file_path: Union[str, Path], pattern: str = r'Loss:\s*(\d+\.?\d*),'):
        """
        Initialize the loss file analyzer.
        
        Args:
            file_path: Path to the log file
            pattern: Regex pattern to extract loss values
        """
        self.file_path = Path(file_path)
        self.pattern = re.compile(pattern)
        self.loss_values: List[float] = []
        
        if not self.file_path.exists():
            raise FileNotFoundError(f"Log file not found: {self.file_path}")
    
    def extract_loss_values(self) -> List[float]:
        """
        Extract loss values from the log file.
        
        Returns:
            List of extracted loss values
        """
        self.loss_values = []
        
        try:
            with open(self.file_path, 'r', encoding='utf-8') as file:
                for line_num, line in enumerate(file, 1):
                    try:
                        match = self.pattern.search(line)
                        if match:
                            loss_value = float(match.group(1))
                            self.loss_values.append(loss_value)
                    except (ValueError, IndexError) as e:
                        logger.warning(f"Could not parse line {line_num}: {line.strip()}, Error: {e}")
            
            logger.info(f"Extracted {len(self.loss_values)} loss values from {self.file_path.name}")
            return self.loss_values
            
        except Exception as e:
            raise RuntimeError(f"Error reading file {self.file_path}: {e}")
    
    def plot_loss_curve(self, save_path: Optional[Union[str, Path]] = None, 
                    title: str = "Loss Values Over Iterations",
                    figsize: Tuple[int, int] = (10, 6)) -> None:
        """
        Create and display/save a loss curve plot.
        
        Args:
            save_path: Optional path to save the plot
            title: Plot title
            figsize: Figure size (width, height)
        """
        if not self.loss_values:
            self.extract_loss_values()
        
        if not self.loss_values:
            raise ValueError("No loss values found to plot")
        
        plt.figure(figsize=figsize)
        plt.plot(range(1, len(self.loss_values) + 1), self.loss_values, 
                linestyle='-', linewidth=2, marker='o', markersize=3)
        plt.xlabel('Iterations/Batches')
        plt.ylabel('Loss')
        plt.title(title)
        plt.grid(True, alpha=0.3)
        
        # Add some statistical information
        min_loss = min(self.loss_values)
        min_idx = self.loss_values.index(min_loss) + 1
        plt.axhline(y=min_loss, color='r', linestyle='--', alpha=0.7, 
                label=f'Min Loss: {min_loss:.4f} at iteration {min_idx}')
        plt.legend()
        
        if save_path:
            save_path = Path(save_path)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        
        plt.show()
    
    def get_statistics(self) -> Dict[str, float]:
        """
        Get statistical summary of loss values.
        
        Returns:
            Dictionary with statistical measures
        """
        if not self.loss_values:
            self.extract_loss_values()
        
        if not self.loss_values:
            return {}
        
        loss_array = np.array(self.loss_values)
        return {
            'min': float(np.min(loss_array)),
            'max': float(np.max(loss_array)),
            'mean': float(np.mean(loss_array)),
            'std': float(np.std(loss_array)),
            'final': float(loss_array[-1]),
            'count': len(self.loss_values)
        }

class CrossValidationHelper:
    """
    Enhanced cross-validation helper with better organization and validation.
    
    This class provides improved functionality compared to the original cv_helper function.
    """
    
    def __init__(self, data_path: Union[str, Path]):
        """
        Initialize the cross-validation helper.
        
        Args:
            data_path: Path to the directory containing data files
        """
        self.data_path = Path(data_path)
        
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_path}")
        
        if not self.data_path.is_dir():
            raise ValueError(f"Path is not a directory: {self.data_path}")
        
        # Check for subdirectories
        subdirs = [entry for entry in self.data_path.iterdir() if entry.is_dir()]
        if subdirs:
            raise ValueError(f"Directory contains subdirectories: {[d.name for d in subdirs]}")
        
        self.file_list = self._get_file_list()
        logger.info(f"Found {len(self.file_list)} files for cross-validation")
    
    def _get_file_list(self) -> List[str]:
        """Get sorted list of files in the directory."""
        files = [f.name for f in self.data_path.iterdir() if f.is_file()]
        return sorted(files)  # Sort for consistent ordering
    
    def generate_cv_splits(self) -> Dict[str, List[str]]:
        """
        Generate leave-one-out cross-validation splits.
        
        Returns:
            Dictionary where each key is a test file and value is list of training files
        """
        cv_dict = {}
        
        for i, test_file in enumerate(self.file_list):
            # Create training set by excluding the current test file
            train_files = self.file_list[:i] + self.file_list[i + 1:]
            cv_dict[test_file] = train_files
        
        return cv_dict
    
    def generate_k_fold_splits(self, k: int = 5, shuffle: bool = True, 
                            random_state: Optional[int] = None) -> Dict[int, Dict[str, List[str]]]:
        """
        Generate k-fold cross-validation splits.
        
        Args:
            k: Number of folds
            shuffle: Whether to shuffle files before splitting
            random_state: Random seed for reproducibility
            
        Returns:
            Dictionary with fold indices and their train/test splits
        """
        if k <= 1 or k > len(self.file_list):
            raise ValueError(f"k must be between 2 and {len(self.file_list)}, got {k}")
        
        files = self.file_list.copy()
        
        if shuffle:
            if random_state is not None:
                np.random.seed(random_state)
            np.random.shuffle(files)
        
        fold_size = len(files) // k
        remainder = len(files) % k
        
        folds = {}
        start_idx = 0
        
        for fold in range(k):
            # Handle remainder by adding one extra item to first 'remainder' folds
            current_fold_size = fold_size + (1 if fold < remainder else 0)
            end_idx = start_idx + current_fold_size
            
            test_files = files[start_idx:end_idx]
            train_files = files[:start_idx] + files[end_idx:]
            
            folds[fold] = {
                'train': train_files,
                'test': test_files
            }
            
            start_idx = end_idx
        
        return folds
    
    def validate_splits(self, cv_splits: Dict) -> bool:
        """
        Validate that CV splits cover all files exactly once.
        
        Args:
            cv_splits: Cross-validation splits to validate
            
        Returns:
            True if valid, raises ValueError if not
        """
        all_test_files = set()
        
        for test_files in cv_splits.values():
            if isinstance(test_files, list):
                all_test_files.update(test_files)
            else:
                all_test_files.add(test_files)
        
        expected_files = set(self.file_list)
        
        if all_test_files != expected_files:
            missing = expected_files - all_test_files
            extra = all_test_files - expected_files
            raise ValueError(f"Invalid CV splits. Missing: {missing}, Extra: {extra}")
        
        return True
    
    def get_statistics(self) -> Dict[str, int]:
        """Get statistics about the dataset."""
        return {
            'total_files': len(self.file_list),
            'loo_folds': len(self.file_list),  # Leave-one-out folds
            'max_k_fold': len(self.file_list)
        }

class ImageMaskProcessor:
    """
    Enhanced image masking and MIP generation with better organization and validation.
    
    This class provides improved functionality compared to the original mra_deskull function.
    """
    
    def __init__(self, img_path: Union[str, Path], mask_path: Union[str, Path]):
        """
        Initialize the image mask processor.
        
        Args:
            img_path: Path to the input NIfTI image
            mask_path: Path to the mask NIfTI image
        """
        self.img_path = Path(img_path)
        self.mask_path = Path(mask_path)
        
        # Validate file existence
        if not self.img_path.exists():
            raise FileNotFoundError(f"Image file not found: {self.img_path}")
        if not self.mask_path.exists():
            raise FileNotFoundError(f"Mask file not found: {self.mask_path}")
        
        # Validate file extensions
        valid_extensions = {'.nii', '.nii.gz'}
        if self.img_path.suffix not in valid_extensions and not str(self.img_path).endswith('.nii.gz'):
            raise ValueError(f"Unsupported image format: {self.img_path.suffix}")
        if self.mask_path.suffix not in valid_extensions and not str(self.mask_path).endswith('.nii.gz'):
            raise ValueError(f"Unsupported mask format: {self.mask_path.suffix}")
    
    def _generate_output_filename(self, suffix: str = "_MASKED") -> str:
        """Generate output filename with suffix."""
        if str(self.img_path).endswith(".nii.gz"):
            return str(self.img_path).replace(".nii.gz", f"{suffix}.nii.gz")
        elif str(self.img_path).endswith(".nii"):
            return str(self.img_path).replace(".nii", f"{suffix}.nii")
        else:
            raise ValueError(f"Unsupported file format: {self.img_path}")
    
    def apply_mask(self, output_path: Optional[Union[str, Path]] = None) -> Path:
        """
        Apply mask to the image and save the result.
        
        Args:
            output_path: Optional custom output path
            
        Returns:
            Path to the saved masked image
        """
        try:
            # Load images
            img_nifti = nib.load(str(self.img_path))    # type: ignore
            mask_nifti = nib.load(str(self.mask_path))  # type: ignore
            
            # Get data arrays
            img_data = img_nifti.get_fdata()    # type: ignore
            mask_data = mask_nifti.get_fdata()  # type: ignore

            # Validate dimensions
            if img_data.shape != mask_data.shape:
                raise ValueError(f"Shape mismatch: image {img_data.shape} vs mask {mask_data.shape}")
            
            # Apply mask (element-wise multiplication)
            masked_data = img_data * mask_data
            
            # Create new NIfTI image
            masked_nifti = nib.Nifti1Image(masked_data, img_nifti.affine, img_nifti.header) # type: ignore
            
            # Determine output path
            if output_path is None:
                output_path = Path(self._generate_output_filename())
            else:
                output_path = Path(output_path)
            
            # Save masked image
            nib.save(masked_nifti, str(output_path)) # type: ignore
            logger.info(f"Masked image saved to: {output_path}")
            
            return output_path
            
        except Exception as e:
            raise RuntimeError(f"Error applying mask: {e}")
    
    def generate_mip(self, masked_data: Optional[np.ndarray] = None, 
                    output_path: Optional[Union[str, Path]] = None,
                    axis: int = 2, colormap: str = 'gray') -> Path:
        """
        Generate Maximum Intensity Projection (MIP) image.
        
        Args:
            masked_data: Optional pre-computed masked data
            output_path: Optional custom output path
            axis: Axis along which to compute maximum projection
            colormap: Colormap for the output image
            
        Returns:
            Path to the saved MIP image
        """
        try:
            # Load masked data if not provided
            if masked_data is None:
                masked_path = self.apply_mask()
                masked_nifti = nib.load(str(masked_path)) # type: ignore
                masked_data = masked_nifti.get_fdata()  # type: ignore
            
            # Generate MIP
            mip_data = np.max(masked_data, axis=axis) # type: ignore
            
            # Rotate for proper orientation (if needed)
            mip_data = np.rot90(mip_data, axes=(0, 1))
            
            # Determine output path
            if output_path is None:
                base_name = self._generate_output_filename().split('.')[0]
                output_path = Path(f"{base_name}_MIP.jpg")
            else:
                output_path = Path(output_path)
            
            # Save MIP image
            plt.imsave(str(output_path), mip_data, cmap=colormap)
            logger.info(f"MIP image saved to: {output_path}")
            
            return output_path
            
        except Exception as e:
            raise RuntimeError(f"Error generating MIP: {e}")
    
    def process(self, generate_mip: bool = True, 
            masked_output_path: Optional[Union[str, Path]] = None,
            mip_output_path: Optional[Union[str, Path]] = None) -> Dict[str, Path]:
        """
        Complete processing: apply mask and optionally generate MIP.
        
        Args:
            generate_mip: Whether to generate MIP image
            masked_output_path: Optional custom path for masked image
            mip_output_path: Optional custom path for MIP image
            
        Returns:
            Dictionary with paths to generated files
        """
        results = {}
        
        # Apply mask
        masked_path = self.apply_mask(masked_output_path)
        results['masked'] = masked_path
        
        # Generate MIP if requested
        if generate_mip:
            # Load the masked data for MIP generation
            masked_nifti = nib.load(str(masked_path))   # type: ignore
            masked_data = masked_nifti.get_fdata()  # type: ignore
            
            mip_path = self.generate_mip(masked_data, mip_output_path)
            results['mip'] = mip_path
        
        return results

class SegmentationMetrics:
    """
    Calculating comprehensive segmentation evaluation metrics.
    
    """
    
    def __init__(self, prediction: np.ndarray, ground_truth: np.ndarray, 
                epsilon: float = 1e-7):
        """
        Initialize the segmentation metrics calculator.
        
        Args:
            prediction: Predicted segmentation (binary or probability)
            ground_truth: Ground truth segmentation (binary)
            epsilon: Small value to prevent division by zero
        """
        # Input validation
        if prediction.shape != ground_truth.shape:
            raise ValueError(f"Shape mismatch: prediction {prediction.shape} vs ground_truth {ground_truth.shape}")
        
        # Convert to binary if needed and flatten
        self.pred = (prediction > 0.5).astype(np.float32).flatten()
        self.gt = (ground_truth > 0.5).astype(np.float32).flatten()
        self.epsilon = epsilon
        
        # Calculate confusion matrix components
        self.tp, self.fp, self.fn, self.tn = self._compute_confusion_matrix()
        self.n = len(self.pred)
        
        logger.debug(f"Confusion matrix: TP={self.tp}, FP={self.fp}, FN={self.fn}, TN={self.tn}")
    
    def _compute_confusion_matrix(self) -> Tuple[int, int, int, int]:
        """
        Efficiently compute confusion matrix components.
        
        Returns:
            Tuple of (true_positive, false_positive, false_negative, true_negative)
        """
        # Vectorized computation is much faster than element-wise operations
        tp = int(np.sum(self.pred * self.gt))
        fp = int(np.sum(self.pred * (1 - self.gt)))
        fn = int(np.sum((1 - self.pred) * self.gt))
        tn = int(np.sum((1 - self.pred) * (1 - self.gt)))
        
        return tp, fp, fn, tn
    
    def dice_coefficient(self) -> float:
        """
        Calculate Dice Similarity Coefficient (DSC).
        
        Returns:
            Dice coefficient (0-1, higher is better)
        """
        denominator = 2 * self.tp + self.fp + self.fn
        if denominator == 0:
            return 1.0 if self.tp == 0 else 0.0
        return (2 * self.tp) / denominator
    
    def jaccard_index(self) -> float:
        """
        Calculate Jaccard Index (IoU - Intersection over Union).
        
        Returns:
            Jaccard index (0-1, higher is better)
        """
        denominator = self.tp + self.fp + self.fn
        if denominator == 0:
            return 1.0 if self.tp == 0 else 0.0
        return self.tp / denominator
    
    def sensitivity(self) -> float:
        """
        Calculate Sensitivity (Recall, True Positive Rate).
        
        Returns:
            Sensitivity (0-1, higher is better)
        """
        denominator = self.tp + self.fn
        if denominator == 0:
            return 1.0
        return self.tp / denominator
    
    def specificity(self) -> float:
        """
        Calculate Specificity (True Negative Rate).
        
        Returns:
            Specificity (0-1, higher is better)
        """
        denominator = self.tn + self.fp
        if denominator == 0:
            return 1.0
        return self.tn / denominator
    
    def precision(self) -> float:
        """
        Calculate Precision (Positive Predictive Value).
        
        Returns:
            Precision (0-1, higher is better)
        """
        denominator = self.tp + self.fp
        if denominator == 0:
            return 1.0 if self.tp == 0 else 0.0
        return self.tp / denominator
    
    def volume_similarity(self) -> float:
        """
        Calculate Volume Similarity Index.
        
        Returns:
            Volume similarity (0-1, higher is better)
        """
        denominator = 2 * self.tp + self.fp + self.fn
        if denominator == 0:
            return 1.0
        return 1 - abs(self.fn - self.fp) / denominator
    
    def mutual_information(self) -> float:
        """
        Calculate Normalized Mutual Information.
        
        Returns:
            Normalized mutual information (0-1, higher is better)
        """
        # Calculate probabilities
        p_seg_1 = (self.tp + self.fn) / self.n
        p_seg_0 = (self.tn + self.fp) / self.n
        p_gt_1 = (self.tp + self.fp) / self.n
        p_gt_0 = (self.tn + self.fn) / self.n
        
        # Handle edge cases
        if p_seg_1 == 0 or p_seg_0 == 0 or p_gt_1 == 0 or p_gt_0 == 0:
            return 0.0
        
        # Calculate entropies
        h_seg = -(p_seg_1 * np.log(p_seg_1 + self.epsilon) + 
                  p_seg_0 * np.log(p_seg_0 + self.epsilon))
        h_gt = -(p_gt_1 * np.log(p_gt_1 + self.epsilon) + 
                 p_gt_0 * np.log(p_gt_0 + self.epsilon))
        
        # Calculate joint entropy
        p_joint = np.array([self.tp, self.fn, self.fp, self.tn]) / self.n
        p_joint = p_joint[p_joint > 0]  # Remove zero probabilities
        h_joint = -np.sum(p_joint * np.log(p_joint + self.epsilon))
        
        # Calculate mutual information
        mi = h_seg + h_gt - h_joint
        
        # Normalize
        if h_seg + h_gt == 0:
            return 0.0
        return 2 * mi / (h_seg + h_gt)
    
    def adjusted_rand_index(self) -> float:
        """
        Calculate Adjusted Rand Index.
        
        Returns:
            Adjusted Rand Index (-1 to 1, higher is better)
        """
        # Calculate index components
        a = (self.tp * (self.tp - 1) + self.fp * (self.fp - 1) + 
             self.tn * (self.tn - 1) + self.fn * (self.fn - 1)) / 2
        
        b = ((self.tp + self.fn) ** 2 + (self.tn + self.fp) ** 2 - 
             (self.tp ** 2 + self.tn ** 2 + self.fp ** 2 + self.fn ** 2)) / 2
        
        c = ((self.tp + self.fp) ** 2 + (self.tn + self.fn) ** 2 - 
             (self.tp ** 2 + self.tn ** 2 + self.fp ** 2 + self.fn ** 2)) / 2
        
        d = self.n * (self.n - 1) / 2 - (a + b + c)
        
        # Calculate ARI
        numerator = 2 * (a * d - b * c)
        denominator = c ** 2 + b ** 2 + 2 * a * d + (a + d) * (c + b)
        
        if denominator == 0:
            return 0.0
        
        return numerator / denominator
    
    def get_all_metrics(self) -> Dict[str, float]:
        """
        Calculate all available metrics.
        
        Returns:
            Dictionary with all metric values
        """
        return {
            'dice': self.dice_coefficient(),
            'jaccard': self.jaccard_index(),
            'sensitivity': self.sensitivity(),
            'specificity': self.specificity(),
            'precision': self.precision(),
            'volume_similarity': self.volume_similarity(),
            'mutual_information': self.mutual_information(),
            'adjusted_rand_index': self.adjusted_rand_index(),
            'true_positive': self.tp,
            'false_positive': self.fp,
            'false_negative': self.fn,
            'true_negative': self.tn
        }
    