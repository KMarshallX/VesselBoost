#!/usr/bin/env python3

"""
Training the chosen model

Editor: Marshall Xu
Last Edited: 01/08/2025
"""

import torch
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
    input_path = config.ds_path if config.prep_mode == 4 else config.ps_path
    
    logger.info("Training session will start shortly..")
    logger.info("Parameters Info:")
    logger.info("*" * 61)
    logger.info(f"Input image path: {config.ds_path}, Segmentation path: {config.lb_path}, Prep_mode: {config.prep_mode}")
    logger.info(f"Epoch number: {config.ep}, Learning rate: {config.lr}")

    # Preprocessing procedure
    preprocess_procedure(config.ds_path, config.ps_path, config.prep_mode)
    
    # Initialize and run training
    trainer = Trainer(
        config.loss_m, config.mo, config.ic, config.oc, config.fil,
        config.op, config.lr, config.optim_gamma, config.ep,
        config.batch_mul, config.osz, config.aug_mode,
        crop_low_thresh=config.crop_low_thresh
    )
    
    # Training loop
    trainer.train(input_path, config.lb_path, config.outmo)


if __name__ == "__main__":
    run_training()

















