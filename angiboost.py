#!/usr/bin/env python3

"""
Angio Boost module - train a model on single subject from scratch, then make prediction
ONLY USE FOR NEUROCONTAINER ON OPEN RECON

Editor: Marshall Xu
Last Edited: 04/10/2024
"""

from logging import config
import os
import logging
import config.angiboost_config as angiboost_config
from library import preprocess_procedure, make_prediction
from library import Trainer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_angiboost():
    config = angiboost_config.args

    image_path = config.image_path
    preprocessed_path = config.preprocessed_path
    label_path = config.label_path
    prep_mode = config.prep_mode
    output_model = config.output_model
    output_path = config.output_path
    pretrained = config.pretrained

    if not os.path.exists(label_path):
        logger.info(f"{label_path} does not exist.")
        os.mkdir(label_path)
        logger.info(f"{label_path} has been created!")

    if not os.path.exists(output_path):
        logger.info(f"{output_path} does not exist.")
        os.mkdir(output_path)
        logger.info(f"{output_path} has been created!")

    # when the preprocess is skipped, directly take the raw data for prediction
    if prep_mode == 4:
        preprocessed_path = image_path

    logger.info("Boosting session will start shortly..")
    logger.info("Parameters Info:")
    logger.info("*" * 61)
    logger.info(f"Input image path: {image_path}, Segmentation path: {label_path}, Prep_mode: {prep_mode}")
    logger.info(f"Epoch number: {config.ep}, Learning rate: {config.lr}")

    # preprocess procedure
    preprocess_procedure(image_path, preprocessed_path, prep_mode)

    # generate the initial segmentation
    make_prediction(
        config.model, config.input_channel, config.output_channel,
        config.filters, preprocessed_path, label_path,
        config.thresh, config.cc, pretrained,
        mip_flag=False
    )

    # initialize the training process
    train_process = Trainer(
        loss_name=config.loss_metric, model_name=config.model,
        input_channels=config.input_channel, output_channels=config.output_channel, filter_count=config.filters,
        optimizer_name=config.optimizer, learning_rate=config.lr, optimizer_gamma=config.optim_gamma, num_epochs=config.ep,
        batch_multiplier=config.batch_multiplier, patch_size=tuple(config.osz), augmentation_mode=config.augmentation_mode,
        crop_mean=config.crop_mean
    )

    # training loop
    train_process.train(preprocessed_path, label_path, output_model)

    # make prediction
    make_prediction(
        config.model, config.input_channel, config.output_channel,
        config.filters, preprocessed_path, output_path,
        config.thresh, config.cc, output_model,
        mip_flag=True
    )

    logger.info(f"Boosting session has been completed! Resultant segmentation has been saved to {output_path}.")

if __name__ == "__main__":
    run_angiboost()

