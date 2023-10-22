#!/usr/bin/env python3

"""
Test time adpatation module

Editor: Marshall Xu
Last Edited: 22/10/2023
"""

import os
import config.adapt_config as adapt_config
from utils.module_utils import preprocess_procedure
from utils.unet_utils import *
from utils.train_utils import TTA_Training


args = adapt_config.args

ds_path = args.ds_path # path to original data
ps_path = args.ps_path # path to preprocessed data
out_path = args.out_path # path to infered data
if os.path.exists(out_path) == False:
    print(f"{out_path} does not exist.")
    os.mkdir(out_path)
    print(f"{out_path} has been created!")

prep_mode = args.prep_mode # preprocessing mode
# when the preprocess is skipped, 
# directly take the raw data for inference
if prep_mode == 4:
    ps_path = ds_path

px_path = args.px_path # path to proxies
if px_path == None: # when the proxy segmentation is not provided
    px_path = os.path.join(out_path, "proxies", "")
    if os.path.exists(px_path) == False:
        print(f"{px_path} does not exist.")     
        os.mkdir(px_path) # create an intermediate output folder inside the output path
        print(f"{px_path} has been created!")
    assert os.path.exists(px_path) == True, "Container doesn't initialize properly, contact for maintenance: https://github.com/KMarshallX/vessel_code"

# output fintuned model path
out_mo_path = os.path.join(out_path, "finetuned", "")
if os.path.exists(out_mo_path) == False:
    print(f"{out_mo_path} does not exist.")     
    os.mkdir(out_mo_path) # create an intermediate output folder inside the output path
    print(f"{out_mo_path} has been created!")
assert os.path.exists(out_mo_path) == True, "Container doesn't initialize properly, contact for maintenance: https://github.com/KMarshallX/vessel_code"

# Resource optimization flag
resource_opt = args.resource

if __name__ == "__main__":

    print("TTA session will start shortly..")

    # preprocessing procedure
    preprocess_procedure(ds_path, ps_path, prep_mode)
    
    # initialize the tta process
    tta_process = TTA_Training(args.loss_m, args.mo,
                                args.ic, args.oc, args.fil,
                                args.op, args.lr,
                                args.optim_gamma, args.ep,
                                args.batch_mul,
                                args.osz, args.aug_mode,
                                args.pretrained,
                                args.thresh, args.cc)
    # tta procedure
    tta_process.test_time_adaptation(ps_path, px_path, out_path, out_mo_path, resource_opt)




