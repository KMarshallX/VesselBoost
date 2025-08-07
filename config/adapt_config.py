"""
argparse configuration for test_time_adaptation.py

Editor: Marshall Xu
Last Edited: 06/08/2025
"""

import argparse

adapt_parser = argparse.ArgumentParser(description="VesselBoost test-time-adaptation arguments")

# input /output
adapt_parser.add_argument('--image_path', type=str, default = "/ds_path/", help="path of the original data")
adapt_parser.add_argument('--preprocessed_path', type=str, default = "/preprocessed_path/", help="path of the preprocessed data")
adapt_parser.add_argument('--output_path', type=str, default = "/out_path/", help="path of the output segmentation")
adapt_parser.add_argument('--proxy_path', type=str, default = None, help="path of the proxy segmentations")
adapt_parser.add_argument('--prep_mode', type=int, default=4, help="Preprocessing mode options. prep_mode=1 : bias field correction only | prep_mode=2 : denoising only | prep_mode=3 : bfc + denoising | prep_mode=4 : no preprocessing applied")

# model configuration
adapt_parser.add_argument('--model', type=str, default="unet3d", help="available: [unet3d, aspp, atrous]")
adapt_parser.add_argument('--input_channel', type=int, default=1, help=argparse.SUPPRESS)
adapt_parser.add_argument('--output_channel', type=int, default=1, help=argparse.SUPPRESS)
adapt_parser.add_argument('--filters', type=int, default=16, help=argparse.SUPPRESS)

# postprocessing / threshold
adapt_parser.add_argument('--thresh', type=float, default=0.1, help="binary threshold for the probability map after prediction, default=0.1")
adapt_parser.add_argument('--cc', type=int, default=10, help="connected components analysis threshold value (denoising), default=10")

# pretrained model path for TTA
adapt_parser.add_argument('--pretrained', type=str, default = "/pretrained_model_path/", help="path of the pretrained model")

# Training configurations
adapt_parser.add_argument('--learning_rate', type=float, default=1e-3, help="learning rate, dtype: float, default=1e-3")
adapt_parser.add_argument('--epochs', type=int, default=5000, help="epoch number (times of iteration), dtype: int, default=5000")
# expected size for training
adapt_parser.add_argument('--patch_size', nargs=3, type=int, default=(64, 64, 64),
                        help='Expected size for training (x y z)')
# optimizer type, available: [sgd, adam]
adapt_parser.add_argument('--optimizer', type=str, default="adam", help='available: [sgd, adam]')
# loss metric type, available: [bce, dice, tver]
adapt_parser.add_argument('--loss_metric', type=str, default="tver", help="available: [bce, dice, tver]")

# Optimizer tuning
# Decays the learning rate of each parameter group by this ratio, dtype: float
adapt_parser.add_argument('--optim_gamma', type=float, default=0.95, help=argparse.SUPPRESS)

# Augmentation configuration
adapt_parser.add_argument('--augmentation_mode', type=str, default="spatial", help="available: [all, off, random, spatial, intensity]")
adapt_parser.add_argument('--crop_low_thresh', type=int, default=128, help="lower threshold for random crop, minimum size in each dimension")

# Resource optimization flag. 0: intermediate files are saved, 1: intermediate files are deleted
adapt_parser.add_argument('--resource', type=int, default=0, help=argparse.SUPPRESS)

# batch size multiplier
adapt_parser.add_argument('--batch_multiplier', type=int, default=5, help=argparse.SUPPRESS)

args = adapt_parser.parse_args()