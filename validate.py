# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import argparse
import os
import sys
import torch
import time
import numpy as np
import random

from torch.backends import cudnn

sys.path.append('.')
from config import cfg
from data import make_val_data_loader
from data.datasets import init_dataset
from engine.validator import validator
from modeling import build_model, build_camera_model
from layers import make_loss
from solver import make_optimizer, WarmupMultiStepLR

from utils.logger import setup_logger

def setup_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(1)

def test(cfg):
    logger = setup_logger("reid_baseline", cfg.OUTPUT_DIR)
    logger.info("Running with config:\n{}".format(cfg))

    # prepare dataset
    val_data_loader, num_query = make_val_data_loader(cfg)

    # prepare model
    model = build_model(cfg, num_classes=[700,500])
    logger.info('Path to the checkpoint of model:%s' %(cfg.TEST.WEIGHT))
    model.load_param(cfg.TEST.WEIGHT, 'self')
    camera_model = build_camera_model(cfg, num_classes=5)
    logger.info('Path to the checkpoint of model:%s' %(cfg.TEST.CAMERA_WEIGHT))
    camera_model.load_param(cfg.TEST.CAMERA_WEIGHT, 'self')
    validator(cfg,
            model,
            camera_model,
            val_data_loader,
            num_query
            )

def main():

    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if cfg.MODEL.DEVICE == "cuda":
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID
    cudnn.benchmark = True

    test(cfg)

if __name__ == '__main__':
    main()
