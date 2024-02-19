#!/usr/bin/env python3

"""
Training the chosen model (new pipeline)

Editor: Marshall Xu
Last Edited: 18/10/2023
"""

import config.train_config as train_config
from utils import preprocess_procedure
from utils import TTA_Training

args = train_config.args
# input images & labels
raw_img = args.ds_path
processed_img = args.ps_path
seg_img = args.lb_path
prep_mode = args.prep_mode
out_path = args.outmo

# when the preprocess is skipped, 
# directly take the raw data for prediction
if prep_mode == 4:
    processed_img = raw_img

if __name__ == "__main__":
    print("Training session will start shortly..")
    print("Parameters Info:\n*************************************************************\n")
    print(f"Input image path: {raw_img}, Segmentation path: {seg_img}, Prep_mode: {prep_mode}\n")
    print(f"Epoch number: {args.ep}, Learning rate: {args.lr} \n")
    
    # preprocess procedure
    preprocess_procedure(raw_img, processed_img, prep_mode)
    # initialize the training process
    train_process = TTA_Training(args.loss_m, args.mo, 
                            args.ic, args.oc, args.fil,
                            args.op, args.lr, 
                            args.optim_gamma, args.ep, 
                            args.batch_mul, 
                            args.osz, args.aug_mode)

    # traning loop (this could be separate out )
    train_process.train(processed_img, seg_img, out_path)

















