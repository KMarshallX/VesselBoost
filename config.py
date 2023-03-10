"""
argparse configuration

Editor: Marshall Xu
Last Edited: 03/01/2023 MM/DD/YYYY
"""


import argparse

parser = argparse.ArgumentParser(description="vessel summer hyperparameters tuning")

# input /output (train.py)
parser.add_argument('--inimg', default = "./data/raw/", help="input image path")
parser.add_argument('--inlab', default = "./data/seg/", help="input ground truth path")
# The following needs to be changed manually (for now)
parser.add_argument('--outmo', default = "./saved_models/model", help="output model path, e.g. ./saved_models/xxxxx")

# model configuration
parser.add_argument('--ic', type=int, default=1, help="input channel number, e.g. RGB -> 3 channels, grayscale -> 1 channel")
parser.add_argument('--oc', type=int, default=1, help="output channel number, e.g. binary classification -> 1 channel")
parser.add_argument('--fil', type=int, default=16, help="filter number, default 16")

# For training
parser.add_argument('--mo', type=str, default="unet3d", help="training model, choose from the following: [unet3d, aspp, atrous]")
parser.add_argument('--bsz', type=int, default=10, help="batch size, dtype: int")
parser.add_argument('--psz', type=tuple, default=(64,64,52), help="input patch size, dtype: tuple")
parser.add_argument('--pst', type=int, default=64, help="input patch step, dtype: int, when patch_step >= patch_size, patches will not be overlapped")
parser.add_argument('--osz', type=tuple, default=(64,64,64), help="output patch size, dtype: tuple")
parser.add_argument('--lr', type=float, default=1e-3, help="learning rate, dtype: float, default=1e-3,when training set is small, use higher learning rate, vice versa")
parser.add_argument('--op', type=str, default="adam", help="optimizer, please choose from following: [sgd, adam]")
parser.add_argument('--ep', type=int, default=5000, help="epoch number (times of iteration), dtype: int, default=16")
parser.add_argument('--loss_m', type=str, default="bce", help="loss metrics, choose from the following: [bce, dice, tver]")

# For optimizer 
parser.add_argument('--optim_step', type=int, default=5000, help="Decays the learning rate of each parameter group every ___ epochs, dtype: int")
parser.add_argument('--optim_gamma', type=float, default=0.5, help="Decays the learning rate of each parameter group by this ratio, dtype: float")

# Warning: this part of CLI is only for experimental tests, not for training nor testing (Marshall, 01/16/2023)
parser.add_argument('--aug_mode', type=str, default="mode1", help="experimental tests CLI, controls the function of the augmentation method, choose from the following: [on, off, test, mode1]")

# test.py
parser.add_argument('--tinimg', default = "./data/train/", help="input image path for test")
parser.add_argument('--tinlab', default = "./data/label/", help="input ground truth path for test")
parser.add_argument('--tm', type=str, help="path to the trained model, e.g. './saved_models/xxxxxxx' ")
parser.add_argument('--tic', type=int, default=1, help="input channel number, e.g. RGB -> 3 channels, grayscale -> 1 channel")
parser.add_argument('--toc', type=int, default=1, help="output channel number, e.g. binary classification -> 1 channel")
parser.add_argument('--tfil', type=int, default=64, help="filter number, default 64")
parser.add_argument('--outim', default = "test_neuro", help="output neuro image name")
parser.add_argument('--img_idx', type=int, default=0, help="index of the image file to be used for prediction, default is 0 (the first file in the given directory)")

# postprocessing.py
parser.add_argument('--outim_path', default = "./saved_image/", help="output sigmoid image path for postprocessing")
parser.add_argument('--img_name', default = "post_processed_image", help="Postprocessed image name")
parser.add_argument('--thresh_vector', nargs='+', type=float, help="Pass TWO integers to the postprocessing procedure. The first integer is the threshold percentile for hard thresholding, recommended value is 5 (5%); the second one is the minimum size of the components in the final image, any components below this size will be wiped out.")

args = parser.parse_args()


