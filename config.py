"""
argparse configuration

Editor: Marshall Xu
Last Edited: 01/03/2023 MM/DD/YYYY
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
parser.add_argument('--fil', type=int, default=64, help="filter number, default 64")


# For training
parser.add_argument('--mo', type=str, default="unet3d", help="training model, choose from the following: [test_mo, unet3d]")
parser.add_argument('--bsz', type=int, default=10, help="batch size, dtype: int")
parser.add_argument('--psz', type=tuple, default=(64,64,52), help="input patch size, dtype: tuple")
parser.add_argument('--pst', type=int, default=64, help="input patch step, dtype: int, when patch_step >= patch_size, patches will not be overlapped")
parser.add_argument('--osz', type=tuple, default=(64,64,64), help="output patch size, dtype: tuple")
parser.add_argument('--lr', type=float, default=5e-3, help="learning rate, dtype: float, default=1e-4,when training set is small, use higher learning rate, vice versa")
parser.add_argument('--op', type=str, default="adam", help="optimizer, please choose from following: [sgd, adam]")
parser.add_argument('--ep', type=int, default=50, help="epoch number (times of iteration), dtype: int, default=16")
# TODO: add loss/score metrics
parser.add_argument('--loss_m', type=str, default="bce", help="loss metrics, choose from the following: [bce, dice, fdice]")

# test.py
parser.add_argument('--tinimg', default = "./data/raw2/", help="input image path for test")
parser.add_argument('--tinlab', default = "./data/seg2/", help="input ground truth path for test")
parser.add_argument('--tm', type=str, default="test_mo", help="testing model, choose from the following: [test_mo, unet3d]")
parser.add_argument('--tic', type=int, default=1, help="input channel number, e.g. RGB -> 3 channels, grayscale -> 1 channel")
parser.add_argument('--toc', type=int, default=1, help="output channel number, e.g. binary classification -> 1 channel")
parser.add_argument('--tfil', type=int, default=64, help="filter number, default 64")
parser.add_argument('--outim', default = "test_neuro", help="output neuro image name")


args = parser.parse_args()


