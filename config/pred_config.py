"""
argparse configuration for prediction.py

Editor: Marshall Xu
Last Edited: 31/07/2025
"""

import argparse

pred_parser = argparse.ArgumentParser(description="VesselBoost prediction arguments")

# paths
pred_parser.add_argument('--ds_path', type=str, default = "/ds_path/", help="path of the original data")
pred_parser.add_argument('--ps_path', type=str, default = "/preprocessed_path/", help="path of the preprocessed data")
pred_parser.add_argument('--out_path', type=str, default = "/out_path/", help="path of the output segmentation")

# preprocessing mode
pred_parser.add_argument('--prep_mode', type=int, default=4, help="Preprocessing mode options. prep_mode=1 : bias field correction only | prep_mode=2 : denoising only | prep_mode=3 : bfc + denoising | prep_mode=4 : no preprocessing applied")

# model configuration
# model name
pred_parser.add_argument('--mo', type=str, default="unet3d", help=argparse.SUPPRESS)
# input channel for unet 3d
pred_parser.add_argument('--ic', type=int, default=1, help=argparse.SUPPRESS)
# output channel for unet 3d
pred_parser.add_argument('--oc', type=int, default=1, help=argparse.SUPPRESS)
# number of filters for each layer in unet 3d
pred_parser.add_argument('--fil', type=int, default=16, help=argparse.SUPPRESS)

# postprocessing / threshold
# hard thresholding value
pred_parser.add_argument('--thresh', type=float, default=0.1, help="binary threshold for the probability map after prediction, default=0.1")
# connected components analysis threshold value (denoising)
pred_parser.add_argument('--cc', type=int, default=10, help="connected components analysis threshold value (denoising), default=10")

pred_parser.add_argument('--pretrained', type=str, default = "/pretrained_model_path/", help="path of the prertrained model")

args = pred_parser.parse_args()