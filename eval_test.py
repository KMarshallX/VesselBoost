import os
from eval_utils import testAndPostprocess, preprocess, finetune
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

input_path = "./data/eval_test/"
output_path = "./data/eval_prep/"

# pres_item = preprocess(input_path, output_path)
# pres_item(True)

prep_data = "./data/eval_prep/"
proxy = "./data/proxy/"

finetune_item = finetune(prep_data, proxy, "unet3d", 1, 1, 16, "Init_ep1000_lr1e3_tver")
finetune_item(0.01, 0.95, 10, 100)

