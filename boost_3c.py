#!/usr/bin/env python3

"""
Boost module for 3-channel input:
train a model on prepared 3-channel subject data, then make prediction.

Editor: Marshall Xu
Last Edited: 24/03/2026
"""

import logging
import os

import config.boost_3c_config as boost_3c_config
from library.boost_3c_utils import Trainer3C, make_prediction_3c, validate_three_channel_dataset


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run_boost_3c() -> None:
    config = boost_3c_config.args

    image_path = config.image_path
    preprocessed_path = config.preprocessed_path
    label_path = config.label_path
    prep_mode = config.prep_mode
    output_model = config.output_model
    output_path = config.output_path

    if config.input_channel != 3:
        raise ValueError(f"boost_3c.py requires --input_channel 3, got {config.input_channel}")

    if not os.path.exists(output_path):
        logger.info("%s does not exist.", output_path)
        os.mkdir(output_path)
        logger.info("%s has been created!", output_path)

    if prep_mode != 4:
        raise ValueError(
            "boost_3c.py expects pre-prepared 3-channel data and only supports prep_mode=4."
        )
    preprocessed_path = image_path

    logger.info("3-channel boosting session will start shortly..")
    logger.info("Parameters Info:")
    logger.info("*" * 61)
    logger.info(
        "Input image path: %s, Segmentation path: %s, Prep_mode: %s",
        image_path,
        label_path,
        prep_mode,
    )
    logger.info("Epoch number: %s, Learning rate: %s", config.epochs, config.learning_rate)
    logger.info("Expected image layout in memory: [C, D, H, W] with C=3")
    logger.info("Labels are expected as single-channel segmentation masks [D, H, W]")

    num_pairs = validate_three_channel_dataset(preprocessed_path, label_path)
    logger.info("Validated %d image/label pair(s) for 3-channel booster training.", num_pairs)

    train_process = Trainer3C(
        loss_name=config.loss_metric,
        model_name=config.model,
        input_channels=config.input_channel,
        output_channels=config.output_channel,
        filter_count=config.filters,
        optimizer_name=config.optimizer,
        learning_rate=config.learning_rate,
        optimizer_gamma=config.optim_gamma,
        num_epochs=config.epochs,
        batch_multiplier=config.batch_multiplier,
        patch_size=tuple(config.patch_size),
        augmentation_mode=config.augmentation_mode,
        threshold=config.thresh,
        connect_threshold=config.cc,
        crop_low_thresh=config.crop_low_thresh,
    )

    train_process.train(preprocessed_path, label_path, output_model)

    if config.use_blending:
        logger.warning("--use_blending is not implemented in boost_3c.py; using standard patch prediction.")

    make_prediction_3c(
        config.model,
        config.input_channel,
        config.output_channel,
        config.filters,
        preprocessed_path,
        output_path,
        config.thresh,
        config.cc,
        config.output_model,
        mip_flag=True,
        probability_flag=True,
    )

    logger.info(
        "3-channel boosting session has been completed! Resultant segmentation has been saved to %s.",
        output_path,
    )


if __name__ == "__main__":
    run_boost_3c()
