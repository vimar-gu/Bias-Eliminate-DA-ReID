# encoding: utf-8
"""
@author:  liaoxingyu
@contact: liaoxingyu2@jd.com
"""

import logging
import torchvision.transforms as T

from .transforms import RandomErasing


def build_transforms(cfg, is_train=True):

    logger = logging.getLogger("reid_baseline.check")

    trans_list = []
    # flip
    if cfg.INPUT.PROB > 0 and is_train:
        trans_list += [T.RandomHorizontalFlip(p=cfg.INPUT.PROB)]
        logger.info('flip is used: %f.' %cfg.INPUT.PROB)
    # pad
    if cfg.INPUT.PADDING > 0 and is_train:
        trans_list += [T.Pad(cfg.INPUT.PADDING),T.RandomCrop(cfg.INPUT.SIZE_TRAIN)]
        logger.info('pad is used: %d.' %cfg.INPUT.PADDING)
    # norm with mean and std
    trans_list += [T.ToTensor()]
    if cfg.INPUT.PIXEL_NORM == 'yes':
        mean = cfg.INPUT.PIXEL_MEAN
        std = cfg.INPUT.PIXEL_STD
        mean = [k/255. for k in cfg.INPUT.PIXEL_MEAN]
        std = [k/255. for k in cfg.INPUT.PIXEL_STD]
        logger.info('mean and std are used. mean %s, std %s',str(mean),str(std))
        trans_list += [T.Normalize(mean=mean, std=std)]
    # random erase
    if cfg.INPUT.RE_PROB > 0 and is_train:
        value = [0.,0.,0.]
        if cfg.INPUT.RE_USING_MEAN == 'yes':
            value = cfg.INPUT.PIXEL_MEAN
        trans_list += [RandomErasing(probability=cfg.INPUT.RE_PROB, sh=cfg.INPUT.RE_SH, mean=value)]
        logger.info('random erase is used. prob=%f, sh=%f, mean=%s' %(cfg.INPUT.RE_PROB,cfg.INPUT.RE_SH,str(value)))

    transform = T.Compose(trans_list)
    return transform

