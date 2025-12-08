"""
argparse configuration for prediction.py

Editor: Marshall Xu
Last Edited: 06/08/2025
"""

import argparse

pred_parser = argparse.ArgumentParser(description="VesselBoost prediction arguments")

# input /output
pred_parser.add_argument('--image_path', type=str, default = "/ds_path/", help="path of the original data")
pred_parser.add_argument('--preprocessed_path', type=str, default = "/preprocessed_path/", help="path of the preprocessed data")
pred_parser.add_argument('--output_path', type=str, default = "/out_path/", help="path of the output segmentation")
pred_parser.add_argument('--prep_mode', type=int, default=4, help="Preprocessing mode options. prep_mode=1 : bias field correction only | prep_mode=2 : denoising only | prep_mode=3 : bfc + denoising | prep_mode=4 : no preprocessing applied")

# model configuration
pred_parser.add_argument('--model', type=str, default="unet3d", help="available: [unet3d, aspp, atrous, nnunet]")
pred_parser.add_argument('--input_channel', type=int, default=1, help=argparse.SUPPRESS)
pred_parser.add_argument('--output_channel', type=int, default=1, help=argparse.SUPPRESS)
pred_parser.add_argument('--filters', type=int, default=16, help=argparse.SUPPRESS)

# postprocessing / threshold
# hard thresholding value
pred_parser.add_argument('--thresh', type=float, default=0.1, help="binary threshold for the probability map after prediction, default=0.1")
# connected components analysis threshold value (denoising)
pred_parser.add_argument('--cc', type=int, default=10, help="connected components analysis threshold value (denoising), default=10")

# experimental features
pred_parser.add_argument('--use_blending', action='store_true', help="EXPERIMENTAL: Use Gaussian blending to reduce patch boundary artifacts. Default: False (original method)")
pred_parser.add_argument('--overlap_ratio', type=float, default=0.5, help="Overlap ratio for Gaussian blending (0-1). Only used with --use_blending. Default: 0.5 (50%% overlap)")

pred_parser.add_argument('--pretrained', type=str, default = "/pretrained_model_path/", help="path of the pretrained model")

args = pred_parser.parse_args()