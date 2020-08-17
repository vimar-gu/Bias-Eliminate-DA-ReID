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
import random

from torch.backends import cudnn

sys.path.append('.')
from config import cfg
from data import make_camstyle_data_loader, make_val_data_loader
from engine.baseline_trainer import do_train
from modeling import build_model
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

def train(cfg):

    logger = setup_logger("reid_baseline", cfg.OUTPUT_DIR)
    logger.info("Running with config:\n{}".format(cfg))

    # prepare camstyle dataset
    train_loader, train_camstyle_loader, val_loader, num_query, num_classes = make_camstyle_data_loader(cfg)
    num_classes.append(-1)

    # prepare model
    model = build_model(cfg, num_classes)

    optimizer, _ = make_optimizer(cfg, model)
    loss_fn = make_loss(cfg, num_classes)

    # Add for using self trained model
    if cfg.MODEL.PRETRAIN_CHOICE == 'resume':
        start_epoch = eval(cfg.MODEL.PRETRAIN_PATH.split('/')[-1].split('.')[0].split('_')[-1])
        logger.info('Start epoch:%d' %start_epoch)
        path_to_optimizer = cfg.MODEL.PRETRAIN_PATH.replace('model', 'optimizer')
        logger.info('Path to the checkpoint of optimizer:%s' %path_to_optimizer)
        model.load_state_dict(torch.load(cfg.MODEL.PRETRAIN_PATH))
        optimizer.load_state_dict(torch.load(path_to_optimizer))
        scheduler = WarmupMultiStepLR(optimizer, cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA, cfg.SOLVER.WARMUP_FACTOR,
                                          cfg.SOLVER.WARMUP_ITERS, cfg.SOLVER.WARMUP_METHOD, start_epoch)
    elif cfg.MODEL.PRETRAIN_CHOICE == 'self' or cfg.MODEL.PRETRAIN_CHOICE == 'imagenet':
        start_epoch = 0
        model.load_param(cfg.MODEL.PRETRAIN_PATH,cfg.MODEL.PRETRAIN_CHOICE)
        scheduler = WarmupMultiStepLR(optimizer, cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA, cfg.SOLVER.WARMUP_FACTOR,
                                          cfg.SOLVER.WARMUP_ITERS, cfg.SOLVER.WARMUP_METHOD)
    else:
        logger.info('Only support pretrain_choice for imagenet and self, but got {}'.format(cfg.MODEL.PRETRAIN_CHOICE))

    do_train(cfg,
            model,
            train_loader,
            val_loader,
            optimizer,
            scheduler,      # modify for using self trained model
            loss_fn,
            num_query,
            start_epoch,     # add for using self trained model
            0,
            train_camstyle_loader
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

    train(cfg)

if __name__ == '__main__':
    main()
