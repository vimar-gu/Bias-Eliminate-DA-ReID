# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""
import logging

import torch
import torch.nn as nn
from ignite.engine import Engine, Events

from utils.reid_metric import R1_mAP, R1_mAP_reranking 


def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long()  # N x C x H x W
    img_flip = img.index_select(3,inv_idx)
    return img_flip


def create_supervised_evaluator(model, camera_model, metrics,
                                device=None):
    """
    Factory function for creating an evaluator for supervised models

    Args:
        model (`torch.nn.Module`): the model to validate
        camera_model (`torch.nn.Module`): the camera model to extract camera features
        metrics (dict of str - :class:`ignite.metrics.Metric`): a map of metric names to Metrics
        device (str, optional): device type specification (default: None).
            Applies to both model and batches.
    Returns:
        Engine: an evaluator engine with supervised inference function
    """
    if device:
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
            camera_model = nn.DataParallel(camera_model)
        model.to(device)
        camera_model.to(device)

    def _inference(engine, batch):
        model.eval()
        with torch.no_grad():
            data, pids, camids, _ = batch
            n, c, h, w = data.size()
            feat = torch.FloatTensor(n, 512).zero_().to(device)
            for i in range(2):
                if (i == 1):
                    data = fliplr(data)
                img = data.to(device)
                outputs = model(img)
                f = outputs
                feat = feat + f
            # norm feature
            fnorm = torch.norm(feat, p=2, dim=1, keepdim=True)
            feat = feat.div(fnorm.expand_as(feat))
            camera_score, camera_feat = camera_model(img)
            camera_score = camera_score.max(1)[1]
            return feat, camera_score, camera_feat, pids, camids

    engine = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine


def tester(
        cfg,
        model,
        camera_model,
        test_loader,
        num_query
):
    device = cfg.MODEL.DEVICE
    log_period = cfg.SOLVER.LOG_PERIOD

    logger = logging.getLogger("reid_baseline.inference")
    logger.info("Enter inferencing")
    if cfg.TEST.RE_RANKING == 'no':
        print("Create evaluator")
        evaluator = create_supervised_evaluator(model, camera_model, metrics={'r1_mAP': R1_mAP(num_query, False, True, feat_norm=cfg.TEST.FEAT_NORM)},
                                                device=device)
    elif cfg.TEST.RE_RANKING == 'yes':
        print("Create evaluator for reranking")
        evaluator = create_supervised_evaluator(model, camera_model, metrics={'r1_mAP': R1_mAP_reranking(num_query, False, True, cfg.OUTPUT_DIR, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)},
                                                device=device)
    else:
        print("Unsupported re_ranking config. Only support for no or yes, but got {}.".format(cfg.TEST.RE_RANKING))

    @evaluator.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        iter = (engine.state.iteration - 1) % len(test_loader) + 1
        if iter % log_period == 0:
            logger.info("Epoch[{}] Iter[{}/{}]"
                        .format(engine.state.epoch, iter, len(test_loader)))

    evaluator.run(test_loader)
    cmc, mAP = evaluator.state.metrics['r1_mAP']
    logger.info('Validation Results')
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10, 20, 50]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
