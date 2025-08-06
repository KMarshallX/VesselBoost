"""
argparse configuration for angiboost.py (openrecon)

Editor: Marshall Xu
Last Edited: 06/08/2025
"""

import argparse

angiboost_parser = argparse.ArgumentParser(description="VesselBoost AngiBoost arguments")

# input /output
angiboost_parser.add_argument('--image_path', default = "/ds_path/", help="input image path")
angiboost_parser.add_argument('--label_path', default = "/lb_path/", help="initially generated label path")
angiboost_parser.add_argument('--preprocessed_path', type=str, default = "/preprocessed_path/", help="path of the preprocessed data")
angiboost_parser.add_argument('--output_path', type=str, default = "/out_path/", help="path of the output segmentation")
angiboost_parser.add_argument('--pretrained', type=str, default = "/pretrained_model_path/", help="path of the prertrained model")
angiboost_parser.add_argument('--prep_mode', type=int, default=4, help="Preprocessing mode options. prep_mode=1 : bias field correction only | prep_mode=2 : denoising only | prep_mode=3 : bfc + denoising | prep_mode=4 : no preprocessing applied")
angiboost_parser.add_argument('--output_model', default = "./saved_models/model", help="output model path, e.g. ./saved_models/xxxxx")

# model configuration
angiboost_parser.add_argument('--model', type=str, default="unet3d", help="available: [unet3d, aspp, atrous]")
angiboost_parser.add_argument('--input_channel', type=int, default=1, help=argparse.SUPPRESS)
angiboost_parser.add_argument('--output_channel', type=int, default=1, help=argparse.SUPPRESS)
angiboost_parser.add_argument('--filters', type=int, default=16, help=argparse.SUPPRESS)

# Training configurations
angiboost_parser.add_argument('--lr', type=float, default=1e-3, help="learning rate, dtype: float, default=1e-3")
angiboost_parser.add_argument('--ep', type=int, default=1000, help="epoch number (times of iteration), dtype: int, default=16")

# expected size of the training patch
angiboost_parser.add_argument('--osz', nargs=3, type=int, default=(64, 64, 64),
                        help='Expected size of the training patch (x y z)')
angiboost_parser.add_argument('--optimizer', type=str, default="adam", help="available: [sgd, adam]")
angiboost_parser.add_argument('--loss_metric', type=str, default="tver", help="available: [bce, dice, tver]")

# Optimizer tuning
# Decays the learning rate of each parameter group by this ratio, dtype: float
angiboost_parser.add_argument('--optim_gamma', type=float, default=0.95, help=argparse.SUPPRESS)

# Augmentation configuration
angiboost_parser.add_argument('--augmentation_mode', type=str, default="spatial", help="available: [all, off, random, spatial, intensity]")
angiboost_parser.add_argument('--crop_low_thresh', type=int, default=128, help="minimum crop size threshold for random cropping, dtype: int")

# batch size multiplier
angiboost_parser.add_argument('--batch_multiplier', type=int, default=5, help=argparse.SUPPRESS)

# postprocessing / threshold
angiboost_parser.add_argument('--thresh', type=float, default=0.1, help="binary threshold for the probability map after prediction, default=0.1")
angiboost_parser.add_argument('--cc', type=int, default=10, help="connected components analysis threshold value (denoising), default=10")

args = angiboost_parser.parse_args()