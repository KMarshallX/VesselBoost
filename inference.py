#!/usr/bin/env python3

"""
Inference using provided pre-trained model


Editor: Marshall Xu
Last edited: 07/04/2023
"""

import os
import infer_config
from utils.module_utils import preprocess, testAndPostprocess

args = infer_config.infer_parser.parse_args()

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

model_type = args.mo # model type
in_chan = args.ic # input channel
ou_chan = args.oc # output channel
fil_num = args.fil # number of filters

threshold_vector = [args.thresh, args.cc]
pretrained_model = args.pretrained # path to pretrained model

if __name__ == "__main__":

    print("Inference session will start shortly..")

    # initialize the preprocessing method with input/output paths
    preprocessing = preprocess(ds_path, ps_path)
    # start or abort preprocessing 
    preprocessing(prep_mode)

    # initialize the inference method
    inference_postpo = testAndPostprocess(model_type, in_chan, ou_chan, fil_num, ps_path, out_path)
    # take each processed image for inference
    processed_data_list = os.listdir(ps_path)
    for i in range(len(processed_data_list)):
        # generate inferred segmentation fot the current image
        inference_postpo(threshold_vector[0], threshold_vector[1], pretrained_model, processed_data_list[i])



