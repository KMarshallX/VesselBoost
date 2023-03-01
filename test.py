"""
Testing the chosen model 

Editor: Marshall Xu
Last Edited: 03/01/2023 MM/DD/YYYY 
"""


import torch

from models.unet_3d import Unet
from models.siyu import CustomSegmentationNetwork
from models.asppcnn import ASPPCNN
from models.ra_unet import MainArchitecture
import config
from utils.unet_utils import verification


def model_chosen(model_name, in_chan, out_chan, filter_num):
    if model_name == "unet3d":
        return Unet(in_chan, out_chan, filter_num)
    elif model_name == "aspp":
        return ASPPCNN(in_chan, out_chan, [1,2,3,5,7])
    elif model_name == "test":
        return CustomSegmentationNetwork()
    elif model_name == "atrous":
        return MainArchitecture()
    else:
        print("Insert a valid model name.")

if __name__ == "__main__":

    args = config.args

    # model configuration
    load_model = model_chosen(args.mo, args.ic, args.oc, args.fil)
    trained_model_path = "./saved_models/" + args.tm
    # trained_model_path = "./saved_models/Unet_ep10_lr1e4_1slab_raw"
    load_model.load_state_dict(torch.load(trained_model_path))
    load_model.eval()
    
    raw_path = args.tinimg
    out_img_name = args.outim

    verification(raw_path, 1, load_model, out_img_name, mode='sigmoid')
