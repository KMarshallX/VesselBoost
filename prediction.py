#!/usr/bin/env python3

"""
Inference using provided pre-trained model


Editor: Marshall Xu
Last edited: 18/10/2023
"""


import os
import logging
import config.pred_config as pred_config
from library import preprocess_procedure, make_prediction

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_prediction():
    config = pred_config.args

    image_path = config.image_path
    preprocessed_path = config.preprocessed_path
    output_path = config.output_path
    prep_mode = config.prep_mode
    model_type = config.model
    in_chan = config.input_channel
    ou_chan = config.output_channel
    fil_num = config.filters
    threshold = config.thresh
    cc = config.cc
    pretrained_model = config.pretrained

    if not os.path.exists(output_path):
        logger.info(f"{output_path} does not exist.")
        os.mkdir(output_path)
        logger.info(f"{output_path} has been created!")

    # when the preprocess is skipped, directly take the raw data for inference
    if prep_mode == 4:
        preprocessed_path = image_path

    logger.info("Prediction session will start shortly..")
    logger.info("Parameters Info:")
    logger.info("*" * 61)
    logger.info(f"Input image path: {image_path}, Preprocessed path: {preprocessed_path}, Output path: {output_path}, Prep_mode: {prep_mode}")

    # preprocess procedure
    preprocess_procedure(image_path, preprocessed_path, prep_mode)

    # make prediction
    make_prediction(
        model_type, in_chan, ou_chan,
        fil_num, preprocessed_path, output_path,
        threshold, cc, pretrained_model,
        mip_flag=True
    )

    logger.info(f"Prediction session has been completed! Resultant segmentation has been saved to {output_path}.")

if __name__ == "__main__":
    run_prediction()

