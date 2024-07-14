"""
argparse configuration for boost.py

Editor: Marshall Xu
Last Edited: 23/10/2023
"""
#@TODO: remember change the optim_patience

import argparse

boost_parser = argparse.ArgumentParser(description="VesselBoost training arguments")

# Train.py
# input /output (train.py)
boost_parser.add_argument('--ds_path', default = "/ds_path/", help="input image path")
boost_parser.add_argument('--lb_path', default = "/lb_path/", help="input label path")
boost_parser.add_argument('--ps_path', type=str, default = "/preprocessed_path/", help="path of the preprocessed data")
boost_parser.add_argument('--out_path', type=str, default = "/out_path/", help="path of the output segmentation")

# preprocessing mode
boost_parser.add_argument('--prep_mode', type=int, default=4, help="Preprocessing mode options. prep_mode=1 : bias field correction only | prep_mode=2 : denoising only | prep_mode=3 : bfc + denoising | prep_mode=4 : no preprocessing applied")

# The following needs to be changed manually (for now)
boost_parser.add_argument('--outmo', default = "./saved_models/model", help="output model path, e.g. ./saved_models/xxxxx")

# model configuration
# model name, available: [unet3d, aspp, atrous]
boost_parser.add_argument('--mo', type=str, default="unet3d", help=argparse.SUPPRESS)
# input channel for unet 3d
boost_parser.add_argument('--ic', type=int, default=1, help=argparse.SUPPRESS)
# output channel for unet 3d
boost_parser.add_argument('--oc', type=int, default=1, help=argparse.SUPPRESS)
# number of filters for each layer in unet 3d
boost_parser.add_argument('--fil', type=int, default=16, help=argparse.SUPPRESS)

# Training configurations
boost_parser.add_argument('--lr', type=float, default=1e-3, help="learning rate, dtype: float, default=1e-3")
boost_parser.add_argument('--ep', type=int, default=5000, help="epoch number (times of iteration), dtype: int, default=16")

# expected size after zooming
boost_parser.add_argument('--osz', type=tuple, default=(64,64,64), help=argparse.SUPPRESS)
# optimizer type, available: [sgd, adam]
boost_parser.add_argument('--op', type=str, default="adam", help=argparse.SUPPRESS)
# loss metric type, available: [bce, dice, tver]
boost_parser.add_argument('--loss_m', type=str, default="tver", help=argparse.SUPPRESS)

# Optimizer tuning
# Decays the learning rate of each parameter group by this ratio, dtype: float
boost_parser.add_argument('--optim_gamma', type=float, default=0.95, help=argparse.SUPPRESS)
# Number of steps with no improvement after which learning rate will be reduced. For example, if patience = 2, then we will ignore the first 2 steps with no improvement, and will only decrease the LR after the 3rd step if the loss still hasn’t improved then. Default: 10.
# Discarded feature (06/20/2023)
# boost_parser.add_argument('--optim_patience', type=int, default=100, help=argparse.SUPPRESS)

# Augmentation mode, available : [on, off, test, mode1]
boost_parser.add_argument('--aug_mode', type=str, default="mode1", help=argparse.SUPPRESS)

# batch size multiplier
boost_parser.add_argument('--batch_mul', type=int, default=4, help=argparse.SUPPRESS)

# postprocessing / threshold
# hard thresholding value
boost_parser.add_argument('--thresh', type=float, default=0.1, help="binary threshold for the probability map after prediction, default=0.1")
# connected components analysis threshold value (denoising)
boost_parser.add_argument('--cc', type=int, default=10, help="connected components analysis threshold value (denoising), default=10")

args = boost_parser.parse_args()