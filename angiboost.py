#!/usr/bin/env python3

"""
Angio Boost module - train a model on single subject from scratch, then make prediction

Editor: Marshall Xu
Last Edited: 04/10/2024
"""

import config.angiboost_config as angiboost_config
from utils import preprocess_procedure, make_prediction
from utils import TTA_Training
import os

args = angiboost_config.args
# input images & labels
ds_path = args.ds_path # needed as input argument
ps_path = args.ps_path
seg_path = args.lb_path # needed as input argument (places to store the initial segmentation)
prep_mode = args.prep_mode # needed as input argument
outmo_path = args.outmo # needed as input argument
out_path = args.out_path # needed as input argument
pretrained = args.pretrained # needed as input argument

if os.path.exists(seg_path) == False:
    print(f"{seg_path} does not exist.")
    os.mkdir(seg_path)
    print(f"{seg_path} has been created!")

if os.path.exists(out_path) == False:
    print(f"{out_path} does not exist.")
    os.mkdir(out_path)
    print(f"{out_path} has been created!")

# when the preprocess is skipped, 
# directly take the raw data for prediction
if prep_mode == 4:
    ps_path = ds_path

if __name__ == "__main__":
    print("Boosting session will start shortly..")
    print("Parameters Info:\n*************************************************************\n")
    print(f"Input image path: {ds_path}, Segmentation path: {seg_path}, Prep_mode: {prep_mode}\n")
    print(f"Epoch number: {args.ep}, Learning rate: {args.lr} \n")
    
    # preprocess procedure
    preprocess_procedure(ds_path, ps_path, prep_mode)
    
    # genereate the initial segmentation
    make_prediction(args.mo, args.ic, args.oc, 
                    args.fil, ps_path, seg_path,
                    args.thresh, args.cc, pretrained,
                    mip_flag=False)
    
    # initialize the training process
    train_process = TTA_Training(args.loss_m, args.mo, 
                            args.ic, args.oc, args.fil,
                            args.op, args.lr, 
                            args.optim_gamma, args.ep, 
                            args.batch_mul, 
                            args.osz, args.aug_mode)

    # traning loop (this could be separate out )
    train_process.train(ps_path, seg_path, outmo_path)

    # make prediction
    make_prediction(args.mo, args.ic, args.oc, 
                    args.fil, ps_path, out_path,
                    args.thresh, args.cc, outmo_path,
                    mip_flag=True)
    
    print(f"Boosting session has been completed! Resultant segmentation has been saved to {out_path}.")

