"""
argparse configuration for boost_3c.py

Editor: Marshall Xu
Last Edited: 24/03/2026
"""

import argparse

boost_3c_parser = argparse.ArgumentParser(description="VesselBoost booster arguments for 3-channel input")

# input / output
boost_3c_parser.add_argument('--image_path', default="/ds_path/", help="input 3-channel image directory")
boost_3c_parser.add_argument('--label_path', default="/lb_path/", help="input single-channel label directory")
boost_3c_parser.add_argument('--preprocessed_path', type=str, default="/preprocessed_path/", help="path of prepared 3-channel data (kept for interface parity)")
boost_3c_parser.add_argument('--output_path', type=str, default="/out_path/", help="path of the output segmentation")
boost_3c_parser.add_argument('--prep_mode', type=int, default=4, help="3-channel booster expects pre-prepared input; use prep_mode=4")
boost_3c_parser.add_argument('--output_model', default="./saved_models/model_3c", help="output model path, e.g. ./saved_models/xxxxx_3c")

# model configuration
boost_3c_parser.add_argument('--model', type=str, default="unet3d", help="available: [unet3d, aspp, atrous, nnunet]")
boost_3c_parser.add_argument('--input_channel', type=int, default=3, help=argparse.SUPPRESS)
boost_3c_parser.add_argument('--output_channel', type=int, default=1, help=argparse.SUPPRESS)
boost_3c_parser.add_argument('--filters', type=int, default=16, help=argparse.SUPPRESS)

# training configurations
boost_3c_parser.add_argument('--learning_rate', type=float, default=1e-3, help="learning rate, dtype: float, default=1e-3")
boost_3c_parser.add_argument('--epochs', type=int, default=1200, help="epoch number (times of iteration), dtype: int, default=5000")

# expected size of the training patch
boost_3c_parser.add_argument('--patch_size', nargs=5, type=int, default=(64, 64, 64),
                             help='Expected size of the training patch (x y z)')
boost_3c_parser.add_argument('--optimizer', type=str, default="adam", help="available: [sgd, adam, adamw]")
boost_3c_parser.add_argument('--loss_metric', type=str, default="tver", help="available: [bce, dice, tver]")

# optimizer tuning
boost_3c_parser.add_argument('--optim_gamma', type=float, default=0.95, help="Decays the learning rate of each parameter group by this ratio, dtype: float")

# augmentation configuration
boost_3c_parser.add_argument('--augmentation_mode', type=str, default="spatial", help="available: [all, off, random, spatial, intensity, flip]")
boost_3c_parser.add_argument('--crop_low_thresh', type=int, default=64, help="lower threshold for random crop minimum size, used in augmentation")

# batch size multiplier
boost_3c_parser.add_argument('--batch_multiplier', type=int, default=5, help=argparse.SUPPRESS)

# postprocessing / threshold
boost_3c_parser.add_argument('--thresh', type=float, default=0.1, help="binary threshold for the probability map after prediction, default=0.1")
boost_3c_parser.add_argument('--cc', type=int, default=10, help="connected components analysis threshold value (denoising), default=10")

# experimental features
boost_3c_parser.add_argument('--use_blending', action='store_true', help="Reserved for parity with boost.py; currently unsupported in boost_3c.py")
boost_3c_parser.add_argument('--overlap_ratio', type=float, default=0.5, help="Reserved for parity with boost.py")

# keep compatibility with shared CLI options
boost_3c_parser.add_argument('--enable_brain_extraction', action='store_true', help="Unused in 3-channel prepared-input workflow")

args = boost_3c_parser.parse_args()
