#!/usr/bin/env python3

"""
Test time adaptation module

Editor: Marshall Xu
Last Edited: 06/08/2025
"""

import os
import logging
import config.adapt_config as adapt_config
from library import preprocess_procedure, Trainer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_tta():
    config = adapt_config.args

    # Paths
    image_path = config.image_path  # path to original data
    preprocessed_path = config.preprocessed_path  # path to preprocessed data
    output_path = config.output_path  # path to inferred data

    # Ensure output directory exists
    if not os.path.exists(output_path):
        logger.info(f"{output_path} does not exist.")
        os.mkdir(output_path)
        logger.info(f"{output_path} has been created!")

    prep_mode = config.prep_mode  # preprocessing mode
    # When the preprocess is skipped, directly take the raw data for prediction
    if prep_mode == 4:
        preprocessed_path = image_path

    # Proxies path
    proxy_path = config.proxy_path  # path to proxies
    if proxy_path is None:
        proxy_path = os.path.join(output_path, "proxies", "")
        if not os.path.exists(proxy_path):
            logger.info(f"{proxy_path} does not exist.")
            os.mkdir(proxy_path)
            logger.info(f"{proxy_path} has been created!")
        assert os.path.exists(proxy_path), "Container doesn't initialize properly, contact for maintenance: https://github.com/KMarshallX/VesselBoost"

    # Output finetuned model path
    output_model_path = os.path.join(output_path, "finetuned", "")
    if not os.path.exists(output_model_path):
        logger.info(f"{output_model_path} does not exist.")
        os.mkdir(output_model_path)
        logger.info(f"{output_model_path} has been created!")
    assert os.path.exists(output_model_path), "Container doesn't initialize properly, contact for maintenance: https://github.com/KMarshallX/VesselBoost"

    # Resource optimization flag
    resource_opt = config.resource

    logger.info("TTA session will start shortly..")
    logger.info("Parameters Info:")
    logger.info("*" * 61)
    logger.info(f"Input image path: {image_path}, Preprocessed path: {preprocessed_path}, Output path: {output_path}, Prep_mode: {prep_mode}")
    logger.info(f"Proxy path: {proxy_path}, Output model path: {output_model_path}")
    logger.info(f"Epoch number: {config.epochs}, Learning rate: {config.learning_rate}")

    # Preprocessing procedure
    preprocess_procedure(image_path, preprocessed_path, prep_mode)

    # Initialize the TTA process
    tta_process = Trainer(
        loss_name=config.loss_metric, model_name=config.model,
        input_channels=config.input_channel, output_channels=config.output_channel, filter_count=config.filters,
        pretrained_model_path=config.pretrained,
        optimizer_name=config.optimizer, learning_rate=config.learning_rate, 
        optimizer_gamma=config.optim_gamma, num_epochs=config.epochs,
        batch_multiplier=config.batch_multiplier, patch_size=tuple(config.patch_size), 
        augmentation_mode=config.augmentation_mode,
        threshold=config.thresh, connect_threshold=config.cc,
        crop_low_thresh=config.crop_low_thresh
    )
    # TTA procedure
    tta_process.test_time_adaptation(
        preprocessed_path, proxy_path, output_path, output_model_path, 
        resource_opt,
        use_gaussian_blending=config.use_blending,
        overlap_ratio=config.overlap_ratio
    )

if __name__ == "__main__":
    run_tta()


