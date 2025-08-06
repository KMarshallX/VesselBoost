"""
argparse configuration for train.py

Editor: Marshall Xu
Last Edited: 31/07/2025
"""

import argparse

train_parser = argparse.ArgumentParser(description="VesselBoost training arguments")

# input /output
train_parser.add_argument('--image_path', default = "/ds_path/", help="input image path")
train_parser.add_argument('--label_path', default = "/lb_path/", help="input label path")
train_parser.add_argument('--preprocessed_path', type=str, default = "/preprocessed_path/", help="path of the preprocessed data")
train_parser.add_argument('--prep_mode', type=int, default=4, help="Preprocessing mode options. prep_mode=1 : bias field correction only | prep_mode=2 : denoising only | prep_mode=3 : bfc + denoising | prep_mode=4 : no preprocessing applied")
train_parser.add_argument('--output_model', default = "./saved_models/model", help="output model path, e.g. ./saved_models/xxxxx")

# model configuration
train_parser.add_argument('--input_channel', type=int, default=1, help=argparse.SUPPRESS)
train_parser.add_argument('--output_channel', type=int, default=1, help=argparse.SUPPRESS)
train_parser.add_argument('--filters', type=int, default=16, help=argparse.SUPPRESS)

# Training configurations
train_parser.add_argument('--learning_rate', type=float, default=1e-3, help="learning rate, dtype: float, default=1e-3")
train_parser.add_argument('--epochs', type=int, default=5000, help="epoch number (times of iteration), dtype: int, default=16")
# model name, available: [unet3d, aspp, atrous]
train_parser.add_argument('--model', type=str, default="unet3d", help="available: [unet3d, aspp, atrous]")
# expected patch size for training
train_parser.add_argument('--output_size', nargs=3, type=int, default=(64, 64, 64), help='Expected patch size for training (x y z)')
# optimizer type, available: [sgd, adam]
train_parser.add_argument('--optimizer', type=str, default="adam", help='available: [sgd, adam]')
# loss metric type, available: [bce, dice, tver]
train_parser.add_argument('--loss_metric', type=str, default="tver", help="available: [bce, dice, tver]")

# Optimizer tuning
# Decays the learning rate of each parameter group by this ratio, dtype: float
train_parser.add_argument('--optimizer_gamma', type=float, default=0.95, help=argparse.SUPPRESS)
# Number of steps with no improvement after which learning rate will be reduced. For example, if patience = 2, then we will ignore the first 2 steps with no improvement, and will only decrease the LR after the 3rd step if the loss still hasn’t improved then. Default: 10.
# Discarded feature (06/20/2023)
# train_parser.add_argument('--optim_patience', type=int, default=100, help=argparse.SUPPRESS)

# Augmentation mode, available : [on, off, test, mode1]
train_parser.add_argument('--augmentation_mode', type=str, default="spatial", help="available: [all, off, random, spatial, intensity]")
train_parser.add_argument('--crop_low_thresh', type=int, default=128, help=argparse.SUPPRESS)

# batch size multiplier
train_parser.add_argument('--batch_multiplier', type=int, default=5, help=argparse.SUPPRESS)

# lower threshold for random crop minimum size

args = train_parser.parse_args()