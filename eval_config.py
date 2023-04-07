"""
argparse configuration for eval.py

Editor: Marshall Xu
Last Edited: 04/07/2023 
"""

import argparse

parser = argparse.ArgumentParser(description="eval.py hyperparameters tuning")

# eval.py
parser.add_argument('--ds_path', type=str, default = "/ds_path/", help="path of the original data")
parser.add_argument('--ps_path', type=str, default = "./preprocessed_data/", help="path of the preprocessed data")
parser.add_argument('--out_path', type=str, default = "/out_path/", help="path of the output segmentation")
parser.add_argument('--prep_bool', type=str, default="yes", help="Whether to proceed the processing procedure?")
parser.add_argument('--init_tm', type=str, default = "Init_ep1000_lr1e3_tver_2", help="name of the pretrained (initial) model")
parser.add_argument('--init_thresh_vector', nargs='+', type=float, help="Pass TWO float numbers to the postprocessing procedure. The first number is the threshold value for hard thresholding, recommended value is 0.01~0.02; the second one is the minimum size of the components in the final image, any components below this size will be wiped out.")
parser.add_argument('--final_thresh_vector', nargs='+', type=float, help="Pass TWO float numbers to the postprocessing procedure. The first number is the threshold value for hard thresholding, recommended value is 0.01~0.02; the second one is the minimum size of the components in the final image, any components below this size will be wiped out.")
parser.add_argument('--eval_lr', type=float, default=1e-3, help="learning rate, dtype: float, default=1e-2")
parser.add_argument('--eval_gamma', type=float, default=0.95, help="Decays the learning rate of each parameter group by this ratio, dtype: float")
parser.add_argument('--eval_patience', type=int, default=100, help="Number of epochs with no improvement after which learning rate will be reduced. For example, if patience = 2, then we will ignore the first 2 epochs with no improvement, and will only decrease the LR after the 3rd epoch if the loss still hasn’t improved then. Default: 10.")
parser.add_argument('--eval_ep', type=int, default=1000, help="epoch number (times of iteration), dtype: int, default=5000")

# model configuration
parser.add_argument('--mo', type=str, default="unet3d", help="training model, choose from the following: [unet3d, aspp, atrous]")
parser.add_argument('--ic', type=int, default=1, help="input channel number, e.g. RGB -> 3 channels, grayscale -> 1 channel")
parser.add_argument('--oc', type=int, default=1, help="output channel number, e.g. binary classification -> 1 channel")
parser.add_argument('--fil', type=int, default=16, help="filter number, default 16")

args = parser.parse_args()