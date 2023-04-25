"""
argparse configuration for test_time_adaptation.py

Editor: Marshall Xu
Last Edited: 04/24/2023 MM/DD/YYYY
"""

import argparse

parser = argparse.ArgumentParser(description="VesselBoost test time adaptation arguments")

# paths
parser.add_argument('--ds_path', type=str, default = "/ds_path/", help="path of the original data")
parser.add_argument('--ps_path', type=str, default = "/preprocessed_path/", help="path of the preprocessed data")
parser.add_argument('--out_path', type=str, default = "/out_path/", help="path of the output segmentation")
parser.add_argument('--px_path', type=str, default = None, help="path of the proxy segmentations")

# preprocessing mode
parser.add_argument('--prep_mode', type=int, default=4, help="Preprocessing mode options. prep_mode=1 : bias field correction only | prep_mode=2 : denoising only | prep_mode=3 : bfc + denoising | prep_mode=4 : no preprocessing applied")

# model configuration
# model name
parser.add_argument('--mo', type=str, default="unet3d", help=argparse.SUPPRESS)
# input channel for unet 3d
parser.add_argument('--ic', type=int, default=1, help=argparse.SUPPRESS)
# output channel for unet 3d
parser.add_argument('--oc', type=int, default=1, help=argparse.SUPPRESS)
# number of filters for each layer in unet 3d
parser.add_argument('--fil', type=int, default=16, help=argparse.SUPPRESS)

# postprocessing / threshold
# hard thresholding value
parser.add_argument('--thresh', type=int, default=0.1, help=argparse.SUPPRESS)
# connected components analysis threshold value (denoising)
parser.add_argument('--cc', type=int, default=10, help=argparse.SUPPRESS)

parser.add_argument('--pretrained', type=str, default = "/pretrained_model_path/", help="path of the prertrained model")

# Training configurations
parser.add_argument('--lr', type=float, default=1e-3, help="learning rate, dtype: float, default=1e-3")
parser.add_argument('--ep', type=int, default=5000, help="epoch number (times of iteration), dtype: int, default=5000")
# expected size after zooming
parser.add_argument('--osz', type=tuple, default=(64,64,64), help=argparse.SUPPRESS)

# Optimizer tuning
# Decays the learning rate of each parameter group by this ratio, dtype: float
parser.add_argument('--optim_gamma', type=float, default=0.8, help=argparse.SUPPRESS)
# Number of steps with no improvement after which learning rate will be reduced. For example, if patience = 2, then we will ignore the first 2 steps with no improvement, and will only decrease the LR after the 3rd step if the loss still hasn’t improved then. Default: 10.
parser.add_argument('--optim_patience', type=int, default=100, help=argparse.SUPPRESS)

# Augmentation mode, available : [on, off, test, mode1]
parser.add_argument('--aug_mode', type=str, default="mode1", help=argparse.SUPPRESS)

args = parser.parse_args()