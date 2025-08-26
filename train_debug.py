#!/usr/bin/env python3

"""
Training the chosen model

Editor: Marshall Xu
Last Edited: 01/08/2025
"""

import logging
import config.train_config as train_config
from library import preprocess_procedure, Trainer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_training():
    """Execute the complete training pipeline."""
    config = train_config.args

    # Determine input paths based on preprocessing mode
    input_path = config.image_path if config.prep_mode == 4 else config.preprocessed_path

    logger.info("Training session will start shortly..")
    logger.info("Parameters Info:")
    logger.info("*" * 61)
    logger.info(f"Input image path: {config.image_path}, Segmentation path: {config.label_path}, Prep_mode: {config.prep_mode}")
    logger.info(f"Epoch number: {config.epochs}, Learning rate: {config.learning_rate}")

    # Preprocessing procedure
    preprocess_procedure(config.image_path, config.preprocessed_path, config.prep_mode)

    # Initialize and run training
    trainer = Trainer(
        loss_name=config.loss_metric, model_name=config.model,
        input_channels=config.input_channel, output_channels=config.output_channel, filter_count=config.filters,
        optimizer_name=config.optimizer, learning_rate=config.learning_rate, optimizer_gamma=config.optimizer_gamma, num_epochs=config.epochs,
        batch_multiplier=config.batch_multiplier, patch_size=tuple(config.output_size), augmentation_mode=config.augmentation_mode,
        crop_low_thresh=config.crop_low_thresh
    )

    # Training loop
    trainer.train(input_path, config.label_path, config.output_model)

def run_debugging(input_path, label_path, output_model, epochs=1000, learning_rate=1e-3, augmentation_mode='off', preprocessed_path=None):
    """Execute the complete training pipeline."""
    config = train_config.args

    # Determine input paths based on preprocessing mode
    input_path = input_path if config.prep_mode == 4 else preprocessed_path

    logger.info("Training session will start shortly..")
    logger.info("Parameters Info:")
    logger.info("*" * 61)
    logger.info(f"Input image path: {input_path}, Segmentation path: {label_path}, Prep_mode: {config.prep_mode}")
    logger.info(f"Augmentation mode: {augmentation_mode}")
    logger.info(f"Epoch number: {epochs}, Learning rate: {learning_rate}")

    # Preprocessing procedure
    preprocess_procedure(input_path, preprocessed_path, config.prep_mode) #type: ignore

    # Initialize and run training
    trainer = Trainer(
        loss_name=config.loss_metric, model_name=config.model,
        input_channels=config.input_channel, output_channels=config.output_channel, filter_count=config.filters,
        optimizer_name=config.optimizer, learning_rate=learning_rate, optimizer_gamma=config.optimizer_gamma, num_epochs=epochs,
        batch_multiplier=config.batch_multiplier, patch_size=tuple(config.output_size), augmentation_mode=augmentation_mode,
        crop_low_thresh=config.crop_low_thresh
    )

    # Training loop
    trainer.train(input_path, label_path, output_model) #type: ignore



if __name__ == "__main__":
    input_path = "/scratch/project/simvascmri/data/single_data/prep_160um_skull/"
    label_path = "/scratch/project/simvascmri/data/single_data/tta_seg_160um/"
    output_model = "./result_models/DEBUG_NewVB_train_aug_off_ep1k_0826"

    run_debugging(input_path, label_path, output_model, epochs=1000, learning_rate=1e-3, augmentation_mode='off')

















