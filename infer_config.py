"""
argparse configuration for inference.py

Editor: Marshall Xu
Last Edited: 04/24/2023 MM/DD/YYYY
"""

import argparse

parser = argparse.ArgumentParser(description="VesselBoost inference arguments")

# paths
parser.add_argument('--ds_path', type=str, default = "/ds_path/", help="path of the original data")
parser.add_argument('--ps_path', type=str, default = "/preprocessed_path/", help="path of the preprocessed data")
parser.add_argument('--out_path', type=str, default = "/out_path/", help="path of the output segmentation")

# preprocessing mode
parser.add_argument('--prep_mode', type=int, default=3, help="Preprocessing mode options. prep_mode=1 : bias field correction only | prep_mode=2 : denoising only | prep_mode=3 : bfc + denoising | prep_mode=4 : no preprocessing applied")

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

args = parser.parse_args()