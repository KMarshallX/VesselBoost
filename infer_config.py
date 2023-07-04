"""
argparse configuration for inference.py

Editor: Marshall Xu
Last Edited: 07/04/2023 MM/DD/YYYY
"""

import argparse

infer_parser = argparse.ArgumentParser(description="VesselBoost inference arguments")

# paths
infer_parser.add_argument('--ds_path', type=str, default = "/ds_path/", help="path of the original data")
infer_parser.add_argument('--ps_path', type=str, default = "/preprocessed_path/", help="path of the preprocessed data")
infer_parser.add_argument('--out_path', type=str, default = "/out_path/", help="path of the output segmentation")

# preprocessing mode
infer_parser.add_argument('--prep_mode', type=int, default=4, help="Preprocessing mode options. prep_mode=1 : bias field correction only | prep_mode=2 : denoising only | prep_mode=3 : bfc + denoising | prep_mode=4 : no preprocessing applied")

# model configuration
# model name
infer_parser.add_argument('--mo', type=str, default="unet3d", help=argparse.SUPPRESS)
# input channel for unet 3d
infer_parser.add_argument('--ic', type=int, default=1, help=argparse.SUPPRESS)
# output channel for unet 3d
infer_parser.add_argument('--oc', type=int, default=1, help=argparse.SUPPRESS)
# number of filters for each layer in unet 3d
infer_parser.add_argument('--fil', type=int, default=16, help=argparse.SUPPRESS)

# postprocessing / threshold
# hard thresholding value
infer_parser.add_argument('--thresh', type=int, default=0.1, help=argparse.SUPPRESS)
# connected components analysis threshold value (denoising)
infer_parser.add_argument('--cc', type=int, default=10, help=argparse.SUPPRESS)

infer_parser.add_argument('--pretrained', type=str, default = "/pretrained_model_path/", help="path of the prertrained model")

args = infer_parser.parse_args()