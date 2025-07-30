#!/usr/bin/env python3

"""
Boost module - train a model on single subject from scratch, then make prediction

Editor: Marshall Xu
Last Edited: 22/10/2023
"""
import logging
import config.boost_config as boost_config
from library import preprocess_procedure, make_prediction
from library import TTA_Training
import os

# Set up logging & arguments
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
args = boost_config.args
# input images & labels
ds_path = args.ds_path
ps_path = args.ps_path
seg_path = args.lb_path
prep_mode = args.prep_mode
outmo_path = args.outmo
out_path = args.out_path
if os.path.exists(out_path) == False:
    print(f"{out_path} does not exist.")
    os.mkdir(out_path)
    print(f"{out_path} has been created!")

# when the preprocess is skipped, 
# directly take the raw data for prediction
if prep_mode == 4:
    ps_path = ds_path

if __name__ == "__main__":
    logging.info("Boosting session will start shortly..")
    logging.info("Parameters Info:\n*************************************************************\n")
    logging.info(f"Input image path: {ds_path}, Segmentation path: {seg_path}, Prep_mode: {prep_mode}\n")
    logging.info(f"Epoch number: {args.ep}, Learning rate: {args.lr} \n")
    
    # preprocess procedure
    preprocess_procedure(ds_path, ps_path, prep_mode)
    # initialize the training process
    train_process = TTA_Training(args.loss_m, args.mo, 
                            args.ic, args.oc, args.fil,
                            args.op, args.lr, 
                            args.optim_gamma, args.ep, 
                            args.batch_mul, 
                            args.osz, args.aug_mode) #NOTE: modified @ 26/05

    # traning loop (this could be separate out )
    train_process.train(ps_path, seg_path, outmo_path)

    # make prediction
    make_prediction(args.mo, args.ic, args.oc, 
                    args.fil, ps_path, out_path,
                    args.thresh, args.cc, args.outmo,
                    mip_flag=True)
    
    print(f"Boosting session has been completed! Resultant segmentation has been saved to {out_path}.")

