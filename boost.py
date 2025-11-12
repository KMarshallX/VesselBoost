#!/usr/bin/env python3

"""
Boost module - train a model on single subject from scratch, then make prediction

Editor: Marshall Xu
Last Edited: 06/08/2025
"""
import logging
import config.boost_config as boost_config
from library import preprocess_procedure, make_prediction
from library import Trainer
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_boost():
    config = boost_config.args

    # input images & labels
    image_path = config.image_path
    preprocessed_path = config.preprocessed_path
    label_path = config.label_path
    prep_mode = config.prep_mode
    output_model = config.output_model
    output_path = config.output_path

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
    logger.info(f"Epoch number: {config.epochs}, Learning rate: {config.learning_rate}")

    # preprocess procedure
    preprocess_procedure(image_path, preprocessed_path, prep_mode)

    # initialize the training process
    train_process = Trainer(
        loss_name=config.loss_metric, model_name=config.model,
        input_channels=config.input_channel, output_channels=config.output_channel, filter_count=config.filters,
        optimizer_name=config.optimizer, learning_rate=config.learning_rate, optimizer_gamma=config.optim_gamma, num_epochs=config.epochs,
        batch_multiplier=config.batch_multiplier, patch_size=tuple(config.patch_size), augmentation_mode=config.augmentation_mode,
        threshold=config.thresh, connect_threshold=config.cc,
        crop_mean=config.crop_mean
    )

    # training loop
    train_process.train(preprocessed_path, label_path, output_model)

    # make prediction
    make_prediction(
        config.model, config.input_channel, config.output_channel,
        config.filters, preprocessed_path, output_path,
        config.thresh, config.cc, config.output_model,
        mip_flag=True
    )

    logger.info(f"Boosting session has been completed! Resultant segmentation has been saved to {output_path}.")

if __name__ == "__main__":
    run_boost()

