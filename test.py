"""
Training the chosen model 

Editor: Marshall Xu
Last Edited: 01/03/2023  MM/DD/YYYY
"""


import torch

from models.unet_3d import Unet
import config
from utils.unet_utils import verification


if __name__ == "__main__":

    args = config.args

    # load the pre-trained model
    load_model = Unet(1, 1, 64)
    trained_model_path = "./saved_models/" + "bce_500_lr_5e3_batch_1img"
    load_model.load_state_dict(torch.load(trained_model_path))
    load_model.eval()
    
    raw_path = args.tinimg
    seg_path = args.tinlab
    out_img_name = "bce_500_lr_5e3_batch_1img"

    out_img_path = "./saved_image/" + out_img_name + ".nii.gz"

    verification(raw_path, seg_path, 1, load_model, out_img_path, mode='sigmoid')
