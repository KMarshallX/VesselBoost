"""
nnU-Net architecture wrapper for VesselBoost

This module provides a wrapper for the nnU-Net architecture that is compatible
with VesselBoost's training pipeline. It uses the PlainConvUNet from the
dynamic_network_architectures package while maintaining VesselBoost's interface.

Editor: Marshall Xu
Last Edited: 05/12/2025

References:
    Isensee, F., et al. (2021). "nnU-Net: a self-configuring method for deep learning-based 
    biomedical image segmentation." Nature Methods, 18(2), 203-211.
"""

import torch
from torch import nn
import numpy as np

# Import nnU-Net components
try:
    from dynamic_network_architectures.architectures.unet import PlainConvUNet
    NNUNET_AVAILABLE = True
except ImportError:
    NNUNET_AVAILABLE = False
    print("Warning: nnU-Net architecture not installed. Install with: pip install nnunetv2")


class nnUNetWrapper(nn.Module):
    """
    Wrapper for nnU-Net architecture compatible with VesselBoost interface.
    
    This class adapts nnU-Net's dynamic UNet architecture to work within
    VesselBoost's training pipeline while maintaining VesselBoost's standard
    interface for model initialization and forward passes.
    
    The architecture uses:
    - Instance normalization (nnU-Net standard) instead of batch normalization
    - LeakyReLU activation functions
    - Residual connections and deep architecture
    - Convolutional upsampling and downsampling
    
    Args:
        in_chan (int): Number of input channels (typically 1 for vessel segmentation)
        out_chan (int): Number of output channels (typically 1 for binary segmentation)
        filter_num (int): Base number of filters (controls model size).
                          Recommended: 32 (vs. 16 for standard UNet3D)
    
    Example:
        >>> model = nnUNetWrapper(in_chan=1, out_chan=1, filter_num=32)
        >>> x = torch.randn(1, 1, 64, 64, 64)
        >>> output = model(x)
        >>> print(output.shape)  # torch.Size([1, 1, 64, 64, 64])
    
    Note:
        This wrapper disables deep supervision for compatibility with VesselBoost's
        training pipeline, which expects a single output tensor.
    """
    
    def __init__(self, in_chan: int = 1, out_chan: int = 1, filter_num: int = 32):
        super().__init__()
        
        if not NNUNET_AVAILABLE:
            raise ImportError(
                "nnU-Net architecture is not installed. "
                "Please install with: pip install nnunetv2"
            )
        
        # Store configuration
        self.input_channels = in_chan
        self.output_channels = out_chan
        self.base_num_features = filter_num
        
        # Define network topology
        # 5 stages (0-4) with 4 downsampling operations = 5-level encoder-decoder
        self.n_stages = 5
        
        # Configure features per stage: [base, base*2, base*4, base*8, base*16]
        self.features_per_stage = [filter_num * (2 ** i) for i in range(self.n_stages)]
        
        # Configure strides: first stage no stride, then 2x2x2 for downsampling
        self.strides = [1] + [2] * (self.n_stages - 1)
        
        # Configure kernel sizes per level (all 3x3x3 for 3D)
        self.kernel_sizes = [3] * self.n_stages
        
        # Number of convolutions per stage (encoder and decoder)
        self.n_conv_per_stage = 2
        self.n_conv_per_stage_decoder = 2
        
        # Initialize nnU-Net architecture with VesselBoost-compatible settings
        self.model = PlainConvUNet(
            input_channels=self.input_channels,
            n_stages=self.n_stages,
            features_per_stage=self.features_per_stage,
            conv_op=nn.Conv3d,
            kernel_sizes=self.kernel_sizes,
            strides=self.strides,
            n_conv_per_stage=self.n_conv_per_stage,
            num_classes=self.output_channels,
            n_conv_per_stage_decoder=self.n_conv_per_stage_decoder,
            conv_bias=True,
            norm_op=nn.InstanceNorm3d,  # nnU-Net uses InstanceNorm (better for small batches)
            norm_op_kwargs={'eps': 1e-5, 'affine': True},
            dropout_op=None,  # No dropout by default
            dropout_op_kwargs=None, # type: ignore
            nonlin=nn.LeakyReLU,
            nonlin_kwargs={'inplace': True},
            deep_supervision=False,  # Disabled for VesselBoost compatibility
            nonlin_first=False,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through nnU-Net architecture.
        
        Args:
            x (torch.Tensor): Input tensor with shape (batch, channel, depth, height, width)
            
        Returns:
            torch.Tensor: Output tensor with same spatial dimensions as input
                         Shape: (batch, output_channels, depth, height, width)
        
        Note:
            The output is the raw logits (before sigmoid). VesselBoost's training
            pipeline applies sigmoid activation during loss computation.
        """
        return self.model(x)
    
    def get_model_info(self) -> dict:
        """
        Get information about the model architecture.
        
        Returns:
            dict: Dictionary containing model configuration details
        """
        return {
            'model_type': 'nnUNet',
            'input_channels': self.input_channels,
            'output_channels': self.output_channels,
            'base_num_features': self.base_num_features,
            'n_stages': self.n_stages,
            'features_per_stage': self.features_per_stage,
            'normalization': 'InstanceNorm3d',
            'activation': 'LeakyReLU',
            'deep_supervision': False,
        }


if __name__ == "__main__":
    """
    Test the nnU-Net wrapper with dummy data.
    
    This test verifies:
    1. Model instantiation
    2. Forward pass with 64^3 patches (VesselBoost standard)
    3. Output shape matches input shape
    4. CUDA compatibility (if available)
    """
    print("=" * 70)
    print("Testing nnU-Net Wrapper for VesselBoost")
    print("=" * 70)
    
    # Test configuration
    batch_size = 1
    in_channels = 1
    out_channels = 1
    filter_num = 32
    patch_size = (64, 64, 64)
    
    # Create test input
    test_input = torch.randn(batch_size, in_channels, *patch_size)
    print(f"\n✓ Created test input tensor")
    print(f"  Shape: {test_input.shape}")
    print(f"  Device: {test_input.device}")
    
    # Initialize model
    print(f"\n✓ Initializing nnU-Net wrapper...")
    print(f"  Input channels: {in_channels}")
    print(f"  Output channels: {out_channels}")
    print(f"  Base filters: {filter_num}")
    
    model = nnUNetWrapper(in_chan=in_channels, out_chan=out_channels, filter_num=filter_num)
    print(f"  Model created successfully")
    
    # Display model info
    model_info = model.get_model_info()
    print(f"\n✓ Model configuration:")
    for key, value in model_info.items():
        print(f"  {key}: {value}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n✓ Model parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    
    # Test forward pass (CPU)
    print(f"\n✓ Testing forward pass (CPU)...")
    model.eval()
    with torch.no_grad():
        output = model(test_input)
    print(f"  Output shape: {output.shape}")
    print(f"  Output range: [{output.min():.4f}, {output.max():.4f}]")
    
    # Verify output shape
    assert output.shape == test_input.shape, \
        f"Output shape {output.shape} doesn't match input shape {test_input.shape}"
    print(f"  ✓ Output shape matches input shape")
    
    # Test with CUDA if available
    if torch.cuda.is_available():
        print(f"\n✓ Testing with CUDA...")
        device = torch.device('cuda')
        model_cuda = model.to(device)
        test_input_cuda = test_input.to(device)
        
        with torch.no_grad():
            output_cuda = model_cuda(test_input_cuda)
        
        print(f"  Device: {output_cuda.device}")
        print(f"  Output shape: {output_cuda.shape}")
        print(f"  ✓ CUDA test passed")
        
        # Memory info
        memory_allocated = torch.cuda.memory_allocated(device) / 1024**2  # MB
        memory_reserved = torch.cuda.memory_reserved(device) / 1024**2  # MB
        print(f"\n✓ GPU Memory usage:")
        print(f"  Allocated: {memory_allocated:.2f} MB")
        print(f"  Reserved: {memory_reserved:.2f} MB")
    else:
        print(f"\n⚠ CUDA not available, skipping GPU test")
    
    print("\n" + "=" * 70)
    print("✓ All tests passed! nnU-Net wrapper is ready for VesselBoost.")
    print("=" * 70)
