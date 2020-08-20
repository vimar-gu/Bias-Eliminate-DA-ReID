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
            data, pids, camids, im_paths = batch
            data = data.to(device) if torch.cuda.device_count() >= 1 else data
            feat = model(data)
            camera_score, camera_feat = camera_model(data)
            camera_score = camera_score.max(1)[1].long()
            return feat, camera_score, camera_feat, pids, camids

    engine = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine


def validator(
        cfg,
        model,
        camera_model,
        val_loader,
        num_query
):
    device = cfg.MODEL.DEVICE
    log_period = cfg.SOLVER.LOG_PERIOD

    logger = logging.getLogger("reid_baseline.inference")
    logger.info("Enter inferencing")
    if cfg.TEST.RE_RANKING == 'no':
        print("Create evaluator")
        evaluator = create_supervised_evaluator(model, camera_model, metrics={'r1_mAP': R1_mAP(num_query, True, False, feat_norm=cfg.TEST.FEAT_NORM)},
                                                device=device)
    elif cfg.TEST.RE_RANKING == 'yes':
        print("Create evaluator for reranking")
        evaluator = create_supervised_evaluator(model, camera_model, metrics={'r1_mAP': R1_mAP_reranking(num_query, True, False, cfg.OUTPUT_DIR, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)},
                                                device=device)
    else:
        print("Unsupported re_ranking config. Only support for no or yes, but got {}.".format(cfg.TEST.RE_RANKING))

    @evaluator.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        iter = (engine.state.iteration - 1) % len(val_loader) + 1
        if iter % log_period == 0:
            logger.info("Epoch[{}] Iter[{}/{}]"
                        .format(engine.state.epoch, iter, len(val_loader)))

    evaluator.run(val_loader)
    cmc, mAP = evaluator.state.metrics['r1_mAP']
    logger.info('Validation Results')
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10, 20, 50]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
