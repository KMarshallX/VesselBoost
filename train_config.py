"""
argparse configuration for train.py

Editor: Marshall Xu
Last Edited: 04/25/2023 MM/DD/YYYY
"""

import argparse

parser = argparse.ArgumentParser(description="VesselBoost training arguments")

# Train.py
# input /output (train.py)
parser.add_argument('--inimg', default = "./data/raw/", help="input image path")
parser.add_argument('--inlab', default = "./data/seg/", help="input ground truth path")
parser.add_argument('--ps_path', type=str, default = "/preprocessed_path/", help="path of the preprocessed data")

# preprocessing mode
parser.add_argument('--prep_mode', type=int, default=4, help="Preprocessing mode options. prep_mode=1 : bias field correction only | prep_mode=2 : denoising only | prep_mode=3 : bfc + denoising | prep_mode=4 : no preprocessing applied")

# The following needs to be changed manually (for now)
parser.add_argument('--outmo', default = "./saved_models/model", help="output model path, e.g. ./saved_models/xxxxx")
# model configuration
# input channel for unet 3d
parser.add_argument('--ic', type=int, default=1, help=argparse.SUPPRESS)
# output channel for unet 3d
parser.add_argument('--oc', type=int, default=1, help=argparse.SUPPRESS)
# number of filters for each layer in unet 3d
parser.add_argument('--fil', type=int, default=16, help=argparse.SUPPRESS)

# Training configurations
parser.add_argument('--lr', type=float, default=1e-3, help="learning rate, dtype: float, default=1e-3,when training set is small, use higher learning rate, vice versa")
parser.add_argument('--ep', type=int, default=5000, help="epoch number (times of iteration), dtype: int, default=16")
# model name, available: [unet3d, aspp, atrous]
parser.add_argument('--mo', type=str, default="unet3d", help=argparse.SUPPRESS)
# expected size after zooming
parser.add_argument('--osz', type=tuple, default=(64,64,64), help=argparse.SUPPRESS)
# optimizer type, available: [sgd, adam]
parser.add_argument('--op', type=str, default="adam", help=argparse.SUPPRESS)
# loss metric type, available: [bce, dice, tver]
parser.add_argument('--loss_m', type=str, default="tver", help=argparse.SUPPRESS)

# Optimizer tuning
# Decays the learning rate of each parameter group by this ratio, dtype: float
parser.add_argument('--optim_gamma', type=float, default=0.8, help=argparse.SUPPRESS)
# Number of steps with no improvement after which learning rate will be reduced. For example, if patience = 2, then we will ignore the first 2 steps with no improvement, and will only decrease the LR after the 3rd step if the loss still hasn’t improved then. Default: 10.
parser.add_argument('--optim_patience', type=int, default=10, help=argparse.SUPPRESS)

# Augmentation mode, available : [on, off, test, mode1]
parser.add_argument('--aug_mode', type=str, default="mode1", help=argparse.SUPPRESS)

args = parser.parse_args()