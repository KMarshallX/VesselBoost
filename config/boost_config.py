"""
argparse configuration for boost.py

Editor: Marshall Xu
Last Edited: 31/07/2025
"""

import argparse

boost_parser = argparse.ArgumentParser(description="VesselBoost booster arguments")

# input /output 
boost_parser.add_argument('--image_path', default = "/ds_path/", help="input image path")
boost_parser.add_argument('--label_path', default = "/lb_path/", help="input label path")
boost_parser.add_argument('--preprocessed_path', type=str, default = "/preprocessed_path/", help="path of the preprocessed data")
boost_parser.add_argument('--output_path', type=str, default = "/out_path/", help="path of the output segmentation")
boost_parser.add_argument('--prep_mode', type=int, default=4, help="Preprocessing mode options. prep_mode=1 : bias field correction only | prep_mode=2 : denoising only | prep_mode=3 : bfc + denoising | prep_mode=4 : no preprocessing applied")
boost_parser.add_argument('--output_model', default = "./saved_models/model", help="output model path, e.g. ./saved_models/xxxxx")

# model configuration
boost_parser.add_argument('--model', type=str, default="unet3d", help="available: [unet3d, aspp, atrous]")
boost_parser.add_argument('--input_channel', type=int, default=1, help=argparse.SUPPRESS)
boost_parser.add_argument('--output_channel', type=int, default=1, help=argparse.SUPPRESS)
boost_parser.add_argument('--filters', type=int, default=16, help=argparse.SUPPRESS)

# Training configurations
boost_parser.add_argument('--learning_rate', type=float, default=1e-3, help="learning rate, dtype: float, default=1e-3")
boost_parser.add_argument('--epochs', type=int, default=5000, help="epoch number (times of iteration), dtype: int, default=16")

# expected size of the training patch
boost_parser.add_argument('--patch_size', nargs=3, type=int, default=(64, 64, 64),
                        help='Expected size of the training patch (x y z)')
boost_parser.add_argument('--optimizer', type=str, default="adam", help="available: [sgd, adam, adamw]")
boost_parser.add_argument('--loss_metric', type=str, default="tver", help="available: [bce, dice, tver]")

# Optimizer tuning
boost_parser.add_argument('--optim_gamma', type=float, default=0.95, help="Decays the learning rate of each parameter group by this ratio, dtype: float")

# Augmentation configuration
boost_parser.add_argument('--augmentation_mode', type=str, default="spatial", help="available: [all, off, random, spatial, intensity, flip]")
boost_parser.add_argument('--crop_mean', type=int, default=128, help="mean value for random crop, used in augmentation")

# batch size multiplier
boost_parser.add_argument('--batch_multiplier', type=int, default=5, help=argparse.SUPPRESS)

# postprocessing / threshold
# hard thresholding value
boost_parser.add_argument('--thresh', type=float, default=0.1, help="binary threshold for the probability map after prediction, default=0.1")
# connected components analysis threshold value (denoising)
boost_parser.add_argument('--cc', type=int, default=10, help="connected components analysis threshold value (denoising), default=10")
# lower threshold for random crop minimum size

args = boost_parser.parse_args()