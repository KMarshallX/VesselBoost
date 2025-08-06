#!/usr/bin/env python3

"""
Test time adaptation module

Editor: Marshall Xu
Last Edited: 06/08/2025
"""

from logging import config
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
    ds_path = config.ds_path  # path to original data
    ps_path = config.ps_path  # path to preprocessed data
    out_path = config.out_path  # path to inferred data

    # Ensure output directory exists
    if not os.path.exists(out_path):
        logger.info(f"{out_path} does not exist.")
        os.mkdir(out_path)
        logger.info(f"{out_path} has been created!")

    prep_mode = config.prep_mode  # preprocessing mode
    # When the preprocess is skipped, directly take the raw data for prediction
    if prep_mode == 4:
        ps_path = ds_path

    # Proxies path
    px_path = config.px_path  # path to proxies
    if px_path is None:
        px_path = os.path.join(out_path, "proxies", "")
        if not os.path.exists(px_path):
            logger.info(f"{px_path} does not exist.")
            os.mkdir(px_path)
            logger.info(f"{px_path} has been created!")
        assert os.path.exists(px_path), "Container doesn't initialize properly, contact for maintenance: https://github.com/KMarshallX/vessel_code"

    # Output finetuned model path
    out_mo_path = os.path.join(out_path, "finetuned", "")
    if not os.path.exists(out_mo_path):
        logger.info(f"{out_mo_path} does not exist.")
        os.mkdir(out_mo_path)
        logger.info(f"{out_mo_path} has been created!")
    assert os.path.exists(out_mo_path), "Container doesn't initialize properly, contact for maintenance: https://github.com/KMarshallX/vessel_code"

    # Resource optimization flag
    args = adapt_config.args

    # Paths
    image_path = args.image_path  # path to original data
    preprocessed_path = args.preprocessed_path  # path to preprocessed data
    output_path = args.output_path  # path to inferred data

    # Ensure output directory exists
    if not os.path.exists(output_path):
        logger.info(f"{output_path} does not exist.")
        os.mkdir(output_path)
        logger.info(f"{output_path} has been created!")

    prep_mode = args.prep_mode  # preprocessing mode
    # When the preprocess is skipped, directly take the raw data for prediction
    if prep_mode == 4:
        preprocessed_path = image_path

    # Proxies path
    proxy_path = args.proxy_path  # path to proxies
    if proxy_path is None:
        proxy_path = os.path.join(output_path, "proxies", "")
        if not os.path.exists(proxy_path):
            logger.info(f"{proxy_path} does not exist.")
            os.mkdir(proxy_path)
            logger.info(f"{proxy_path} has been created!")
        assert os.path.exists(proxy_path), "Container doesn't initialize properly, contact for maintenance: https://github.com/KMarshallX/vessel_code"

    # Output finetuned model path
    out_mo_path = os.path.join(output_path, "finetuned", "")
    if not os.path.exists(out_mo_path):
        logger.info(f"{out_mo_path} does not exist.")
        os.mkdir(out_mo_path)
        logger.info(f"{out_mo_path} has been created!")
    assert os.path.exists(out_mo_path), "Container doesn't initialize properly, contact for maintenance: https://github.com/KMarshallX/vessel_code"

    # Resource optimization flag
    resource_opt = args.resource

    logger.info("TTA session will start shortly..")
    logger.info("Parameters Info:")
    logger.info("*" * 61)
    logger.info(f"Input image path: {image_path}, Preprocessed path: {preprocessed_path}, Output path: {output_path}, Prep_mode: {prep_mode}")

    # Preprocessing procedure
    preprocess_procedure(image_path, preprocessed_path, prep_mode)

    # Initialize the TTA process
    tta_process = Trainer(
        loss_name=args.loss_metric, model_name=args.model,
        input_channels=args.input_channel, output_channels=args.output_channel, filter_count=args.filters,
        optimizer_name=args.optimizer, learning_rate=args.learning_rate, optimizer_gamma=args.optim_gamma, num_epochs=args.epochs,
        batch_multiplier=args.batch_multiplier, patch_size=tuple(args.osz), augmentation_mode=args.augmentation_mode,
        crop_low_thresh=args.crop_low_thresh
    )
    # TTA procedure
    tta_process.test_time_adaptation(preprocessed_path, proxy_path, output_path, out_mo_path, resource_opt)




