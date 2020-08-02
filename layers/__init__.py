# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch.nn.functional as F

from .triplet_loss import TripletLoss, CrossEntropyLabelSmooth, WeightedTripletLoss

import logging
import copy

def make_loss(cfg, num_classes):

    logger = logging.getLogger("reid_baseline.check")

    # whether semihnm for triplet
    if cfg.LOSS.TRP_HNM == 'yes':
        logger.info('semi-hnm is used for triplet loss.')
        semi = True
    else:
        logger.info('semi-hnm is not used for triplet loss.')
        semi = False

    # whether l2norm for triplet
    if cfg.LOSS.TRP_L2 == 'yes':
        logger.info('l2 normal is used for triplet loss.')
        use_l2 = True
    else:
        logger.info('l2 normal is not used for triplet loss.')
        use_l2 = False

    infos = [cfg.SRC_DATA,cfg.TGT_UNSUPDATA]

    # src loss
    i = 0
    src_func = []
    if 'trp' in cfg.LOSS.LOSS_TYPE[0]:
        src_triplet = WeightedTripletLoss()
        def src_trploss_func(score, feat, target):
            return src_triplet(feat, target, use_l2, semi)[0]
        src_func.append(src_trploss_func)
    if 'cls' in cfg.LOSS.LOSS_TYPE[0]:
        if cfg.LOSS.IF_LABELSMOOTH == 'on':
            def src_clsloss_func(score, feat, target):
                src_xent = CrossEntropyLabelSmooth(num_classes=num_classes[0], epsilon=cfg.LOSS.LABELSMOOTH_EPSILON)
                return src_xent(score, target)
        else:
            def src_clsloss_func(score, feat, target):
                return F.cross_entropy(score, target)
        src_func.append(src_clsloss_func)

    # tgt unsup loss
    i = 1
    tgt_unsup_func = []
    if 'trp' in cfg.LOSS.LOSS_TYPE[1]:
        tgt_unsup_triplet = WeightedTripletLoss()
        def tgt_unsup_trploss_func(score, feat, target):
            return tgt_unsup_triplet(feat, target, use_l2, semi)[0]
        tgt_unsup_func.append(tgt_unsup_trploss_func)
    if 'cls' in cfg.LOSS.LOSS_TYPE[1]:
        if cfg.LOSS.IF_LABELSMOOTH == 'on':
            def tgt_unsup_clsloss_func(score, feat, target):
                tgt_unsup_xent = CrossEntropyLabelSmooth(num_classes=num_classes[1])
                return tgt_unsup_xent(score, target)
        else:
            def tgt_unsup_clsloss_func(score, feat, target):
                return F.cross_entropy(score, target)
        tgt_unsup_func.append(tgt_unsup_clsloss_func)

    loss_funcs = [src_func, tgt_unsup_func]

    return loss_funcs

def make_camera_loss(cfg, num_classes):

    logger = logging.getLogger("reid_baseline.check")

    # pre-define loss
    if 'trp' in cfg.LOSS.LOSS_TYPE[0]:
        triplet = TripletLoss(cfg.LOSS.TRP_MARGIN)
    if 'cls' in cfg.LOSS.LOSS_TYPE[0] and cfg.LOSS.IF_LABELSMOOTH == 'on':
        xent = CrossEntropyLabelSmooth(num_classes=num_classes)
        logger.info("label smooth on, numclasses:%d" %num_classes)

    # whether semihnm for triplet
    if cfg.LOSS.TRP_HNM == 'yes':
        logger.info('semi-hnm is used for triplet loss.')
        semi = True
    else:
        logger.info('semi-hnm is not used for triplet loss.')
        semi = False

    # whether l2norm for triplet
    if cfg.LOSS.TRP_L2 == 'yes':
        logger.info('l2 normal is used for triplet loss.')
        use_l2 = True
    else:
        logger.info('l2 normal is not used for triplet loss.')
        use_l2 = False

    # build losses
    if cfg.LOSS.LOSS_TYPE[0] == 'trp':
        def loss_func(score, feat, target):
            return triplet(feat, target, use_l2, semi)[0]
    elif cfg.LOSS.LOSS_TYPE[0] == 'cls':
        if cfg.LOSS.IF_LABELSMOOTH == 'on':
            def loss_func(score, feat, target):
                return xent(score, target)
        else:
            def loss_func(score, feat, target):
                return F.cross_entropy(score, target)
    elif cfg.LOSS.LOSS_TYPE[0] == 'trp_cls':
        if cfg.LOSS.IF_LABELSMOOTH == 'on':
            def loss_func(score, feat, target):
                return xent(score, target) + triplet(feat, target, use_l2, semi)[0]
        else:
            def loss_func(score, feat, target):
                return F.cross_entropy(score, target) + triplet(feat, target, use_l2, semi)[0]
    else:
        logger.info('wrong loss type {}'.format(cfg.DATALOADER.SAMPLER))
    return loss_func

