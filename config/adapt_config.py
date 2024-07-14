"""
argparse configuration for test_time_adaptation.py

Editor: Marshall Xu
Last Edited: 07/04/2023 MM/DD/YYYY
"""

import argparse

adapt_parser = argparse.ArgumentParser(description="VesselBoost test time adaptation arguments")

# paths
adapt_parser.add_argument('--ds_path', type=str, default = "/ds_path/", help="path of the original data")
adapt_parser.add_argument('--ps_path', type=str, default = "/preprocessed_path/", help="path of the preprocessed data")
adapt_parser.add_argument('--out_path', type=str, default = "/out_path/", help="path of the output segmentation")
adapt_parser.add_argument('--px_path', type=str, default = None, help="path of the proxy segmentations")

# preprocessing mode
adapt_parser.add_argument('--prep_mode', type=int, default=4, help="Preprocessing mode options. prep_mode=1 : bias field correction only | prep_mode=2 : denoising only | prep_mode=3 : bfc + denoising | prep_mode=4 : no preprocessing applied")

# model configuration
# model name
adapt_parser.add_argument('--mo', type=str, default="unet3d", help=argparse.SUPPRESS)
# input channel for unet 3d
adapt_parser.add_argument('--ic', type=int, default=1, help=argparse.SUPPRESS)
# output channel for unet 3d
adapt_parser.add_argument('--oc', type=int, default=1, help=argparse.SUPPRESS)
# number of filters for each layer in unet 3d
adapt_parser.add_argument('--fil', type=int, default=16, help=argparse.SUPPRESS)

# postprocessing / threshold
# hard thresholding value
adapt_parser.add_argument('--thresh', type=float, default=0.1, help="binary threshold for the probability map after prediction, default=0.1")
# connected components analysis threshold value (denoising)
adapt_parser.add_argument('--cc', type=int, default=10, help="connected components analysis threshold value (denoising), default=10")

adapt_parser.add_argument('--pretrained', type=str, default = "/pretrained_model_path/", help="path of the prertrained model")

# Training configurations
adapt_parser.add_argument('--lr', type=float, default=1e-3, help="learning rate, dtype: float, default=1e-3")
adapt_parser.add_argument('--ep', type=int, default=5000, help="epoch number (times of iteration), dtype: int, default=5000")
# expected size after zooming
adapt_parser.add_argument('--osz', type=tuple, default=(64,64,64), help=argparse.SUPPRESS)
# optimizer type, available: [sgd, adam]
adapt_parser.add_argument('--op', type=str, default="adam", help=argparse.SUPPRESS)
# loss metric type, available: [bce, dice, tver]
adapt_parser.add_argument('--loss_m', type=str, default="tver", help=argparse.SUPPRESS)

# Optimizer tuning
# Decays the learning rate of each parameter group by this ratio, dtype: float
adapt_parser.add_argument('--optim_gamma', type=float, default=0.95, help=argparse.SUPPRESS)
# Number of steps with no improvement after which learning rate will be reduced. For example, if patience = 2, then we will ignore the first 2 steps with no improvement, and will only decrease the LR after the 3rd step if the loss still hasn’t improved then. Default: 10.
# Discarded feature (06/20/2023)
# adapt_parser.add_argument('--optim_patience', type=int, default=10000, help=argparse.SUPPRESS)

# Augmentation mode, available : [on, off, test, mode1]
adapt_parser.add_argument('--aug_mode', type=str, default="mode1", help=argparse.SUPPRESS)

# Resource optimization flag. 0: intermediate files are saved, 1: intermediate files are deleted
adapt_parser.add_argument('--resource', type=int, default=0, help=argparse.SUPPRESS)

# batch size multiplier
adapt_parser.add_argument('--batch_mul', type=int, default=4, help=argparse.SUPPRESS)

args = adapt_parser.parse_args()