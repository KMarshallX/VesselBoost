"""
Testing the chosen model 

Editor: Marshall Xu
Last Edited: 01/02/2022 MM/DD/YYYY 
"""


import torch

from models.unet_3d import Unet
from models.siyu import CustomSegmentationNetwork
from models.asppcnn import ASPPCNN
import config
from utils.unet_utils import verification


if __name__ == "__main__":

    args = config.args

    # load the pre-trained model
    load_model = Unet(args.ic, args.oc, args.fil)
    # load_model = CustomSegmentationNetwork()
    trained_model_path = "./saved_models/" + args.tm
    load_model.load_state_dict(torch.load(trained_model_path))
    load_model.eval()
    
    raw_path = args.tinimg
    seg_path = args.tinlab
    out_img_name = args.outim

    out_img_path = "./saved_images/"+out_img_name+".nii.gz"

    verification(raw_path, 0, load_model, out_img_path, mode='sigmoid')
