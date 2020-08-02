# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

from .baseline import Baseline, CameraBaseline
import numpy as np

def build_model(cfg, num_classes):
    model = Baseline(num_classes, cfg.MODEL.LAST_STRIDE, cfg.MODEL.PRETRAIN_PATH, cfg.MODEL.NECK, cfg.MODEL.DROPOUT, cfg.MODEL.NAME, cfg.MODEL.PRETRAIN_CHOICE, cfg.LOSS.LOSS_TYPE, cfg.DATALOADER.SAMPLER_PROB)
    return model

def build_camera_model(cfg, num_classes):
    model = CameraBaseline(num_classes, cfg.MODEL.LAST_STRIDE, cfg.MODEL.PRETRAIN_PATH, cfg.MODEL.NECK, cfg.MODEL.DROPOUT, cfg.MODEL.CAMERA_NAME, cfg.MODEL.PRETRAIN_CHOICE, cfg.LOSS.LOSS_TYPE, cfg.DATALOADER.SAMPLER_PROB)
    return model
