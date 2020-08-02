# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import logging
import time
import numpy as np

import torch
import torch.nn as nn
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint, Timer
from ignite.metrics import RunningAverage

from layers import make_loss
from data import make_camstyle_target_unsupdata_loader, make_camstyle_alltrain_data_loader
from utils.reid_metric import R1_mAP
from utils.reid_metric import Cluster

import torch.distributed as dist
from torch.distributed import get_rank, get_world_size

def create_supervised_trainer(model, optimizer, loss_fn, loss_weight,
                              device=None, device_id=-1):
    """
    Factory function for creating a trainer for supervised models

    Args:
        model (`torch.nn.Module`): the model to train
        optimizer (`torch.optim.Optimizer`): the optimizer to use
        loss_fn (torch.nn loss function): the loss function to use
        device (str, optional): device type specification (default: None).
            Applies to both model and batches.

    Returns:
        Engine: a trainer engine with supervised update function
    """
    if device:
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
            model.to(device)
        else:
            model.to(device)

    def _update(engine, batch):
        model.train()
        optimizer.zero_grad()
        batch = engine.state.batch
        img, target, setid = batch[0][0], batch[0][1], batch[0][2]
        img = img.to(device) if torch.cuda.device_count() >= 1 else img
        target = target.to(device) if torch.cuda.device_count() >= 1 else target
        feats = model(img)

        losses = []
        for i in range(len(loss_fn)):
            loss = torch.tensor(0.).cuda()
            if i == setid[0]:
                for j in range(len(loss_fn[i])):
                    loss += loss_fn[i][j](feats[i], feats[-1], target)
            else:
                loss += 0. * torch.sum(feats[i])
            loss += 0. * sum(p.sum() for p in model.parameters())
            losses.append(loss)

        camstyle_img, camstyle_target, camstyle_setid = batch[1][0], batch[1][1], batch[1][2]
        camstyle_img = camstyle_img.to(device) if torch.cuda.device_count() >= 1 else camstyle_img
        camstyle_target = camstyle_target.to(device) if torch.cuda.device_count() >= 1 else camstyle_target
        camstyle_feats = model(camstyle_img)

        camstyle_losses = []
        for i in range(len(loss_fn)):
            camstyle_loss = torch.tensor(0.).cuda()
            if i == camstyle_setid[0]:
                for j in range(len(loss_fn[i])):
                    camstyle_loss += loss_fn[i][j](camstyle_feats[i], camstyle_feats[-1], camstyle_target)
            else:
                camstyle_loss += 0. * torch.sum(camstyle_feats[i])
            camstyle_loss += 0. * sum(p.sum() for p in model.parameters())
            camstyle_losses.append(camstyle_loss)

        all_losses = []
        for i in range(len(loss_fn)):
            backloss = losses[i] * loss_weight[i] + camstyle_losses[i] * loss_weight[i]
            if i == len(loss_fn) - 1:
                backloss.backward()
            else:
                backloss.backward(retain_graph=True)
            
            all_losses.append(backloss)
      
        optimizer.step()
        return {'src':all_losses[0].item(), 'tgt_unsup':all_losses[1].item()}

    return Engine(_update)


def create_supervised_evaluator(model, metrics,
                                device=None, device_id=-1):
    """
    Factory function for creating an evaluator for supervised models

    Args:
        model (`torch.nn.Module`): the model to train
        metrics (dict of str - :class:`ignite.metrics.Metric`): a map of metric names to Metrics
        device (str, optional): device type specification (default: None).
            Applies to both model and batches.
    Returns:
        Engine: an evaluator engine with supervised inference function
    """

    def _inference(engine, batch):
        model.eval()
        with torch.no_grad():
            data, pids, camids, _ = batch
            data = data.to(device) if torch.cuda.device_count() >= 1 else data
            feat = model(data)
            return feat, 0, 0, pids, camids

    engine = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine


def create_psolabel_producer(model, camera_model, metrics,
                                device=None, device_id=-1):
    """
    Factory function for creating an evaluator for supervised models

    Args:
        model (`torch.nn.Module`): the model to train
        camera_model (`torch.nn.Module`): the camera model to extract camera features
        metrics (dict of str - :class:`ignite.metrics.Metric`): a map of metric names to Metrics
        device (str, optional): device type specification (default: None).
            Applies to both model and batches.
    Returns:
        Engine: an evaluator engine with supervised inference function
    """
    if device:
        if torch.cuda.device_count() > 1:
            camera_model = nn.DataParallel(camera_model)
            camera_model.to(device)
        else:
            camera_model.to(device)

    def _inference(engine, batch):
        model.eval()
        camera_model.eval()
        with torch.no_grad():
            data, pids, camids, trkids = batch
            data = data.to(device) if torch.cuda.device_count() >= 1 else data
            feat = model(data)
            _, camfeat = camera_model(data)
            return feat, camfeat, pids, camids, trkids

    engine = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine


def do_train(
        cfg,
        model,
        camera_model,
        val_data_loader,
        optimizer,
        scheduler,
        loss_fn,
        num_query,
        start_epoch,
        device_id
):

    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD
    psolabel_period = cfg.TGT_UNSUPDATA.PSOLABEL_PERIOD
    output_dir = cfg.OUTPUT_DIR
    epochs = cfg.SOLVER.MAX_EPOCHS
    device = cfg.MODEL.DEVICE

    logger = logging.getLogger("reid_baseline.train")
    logger.info("Start training")
    trainer = create_supervised_trainer(model, optimizer, loss_fn, cfg.LOSS.LOSS_WEIGHTS, device=device, device_id=device_id)
    evaluator = create_supervised_evaluator(model, metrics={'r1_mAP': R1_mAP(num_query, True, False, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)}, device=device, device_id=device_id)
    psolabel_producer = create_psolabel_producer(model, camera_model, metrics={'cluster': Cluster(topk=cfg.TGT_UNSUPDATA.CLUSTER_TOPK,dist_thrd=cfg.TGT_UNSUPDATA.CLUSTER_DIST_THRD, finetune=cfg.MODEL.FINETUNE)}, device=device, device_id=device_id)
    if device_id == 0:
        checkpointer = ModelCheckpoint(output_dir, cfg.MODEL.NAME, checkpoint_period, n_saved=10, require_empty=False)
        trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpointer, {'model': model.state_dict(),
                                                                     'optimizer': optimizer.state_dict()})

    timer = Timer(average=True)
    timer.attach(trainer, start=Events.EPOCH_STARTED, resume=Events.ITERATION_STARTED,
                 pause=Events.ITERATION_COMPLETED, step=Events.ITERATION_COMPLETED)

    # average metric to attach on trainer
    RunningAverage(output_transform=lambda x: x['src']).attach(trainer, 'src_loss')
    RunningAverage(output_transform=lambda x: x['tgt_unsup']).attach(trainer, 'tgt_unsup_loss')

    @trainer.on(Events.STARTED)
    def start_training(engine):
        engine.state.epoch = start_epoch

    @trainer.on(Events.EPOCH_STARTED)
    def adjust_learning_rate(engine):
        scheduler.step()

    def cycle(iterable):
        while True:
            for i in iterable:
                yield i

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        iter = (engine.state.iteration - 1) % len(alltrain_data_loader) + 1
        if iter % log_period == 0:
            if cfg.DATALOADER.SAMPLER_PROB[0] != 0:
                src_loss = engine.state.metrics['src_loss']/cfg.DATALOADER.SAMPLER_PROB[0]
            else:
                src_loss = 0.
            if cfg.DATALOADER.SAMPLER_PROB[1] != 0:
                tgt_unsup_loss = engine.state.metrics['tgt_unsup_loss']/cfg.DATALOADER.SAMPLER_PROB[1]
            else:
                tgt_unsup_loss = 0.
            logger.info("Epoch[{}] Iter[{}/{}] src: {:.3f}, unsup: {:.3f}, lr: {:.2e}/{:.2e}"
                        .format(engine.state.epoch, iter, len(alltrain_data_loader),
                                src_loss, tgt_unsup_loss,
                                scheduler.get_lr()[0],scheduler.get_lr()[-1]))

    # adding handlers using `trainer.on` decorator API
    @trainer.on(Events.EPOCH_COMPLETED)
    def print_times(engine):
        logger.info('Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]'
                    .format(engine.state.epoch, timer.value() * timer.step_count,
                            alltrain_data_loader.batch_size / timer.value()))
        logger.info('-' * 10)
        timer.reset()

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        if engine.state.epoch % eval_period == 0:
            evaluator.run(val_data_loader)
            cmc, mAP = evaluator.state.metrics['r1_mAP']
            logger.info("Validation Results - Epoch: {}".format(engine.state.epoch))
            logger.info("mAP: {:.1%}".format(mAP))
            for r in [1, 5, 10]:
                logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))

    @trainer.on(Events.EPOCH_COMPLETED)
    def update_psolabels(engine):
        if engine.state.epoch % psolabel_period == 0:
            camstyle_target_unsupdata_loader = make_camstyle_target_unsupdata_loader(cfg) 
            psolabel_producer.run(camstyle_target_unsupdata_loader)
            psolabels,cluster_acc,num = psolabel_producer.state.metrics['cluster']
            logger.info("Cluster Acc: {:.3f}, classes: {} imgnum: {}".format(cluster_acc,len(set(psolabels))-1,num))
            alltrain_data_loader, alltrain_camstyle_data_loader = make_camstyle_alltrain_data_loader(cfg, psolabels)

    camstyle_target_unsupdata_loader = make_camstyle_target_unsupdata_loader(cfg)
    psolabel_producer.run(camstyle_target_unsupdata_loader)
    psolabels,cluster_acc,num = psolabel_producer.state.metrics['cluster']
    logger.info("Cluster Acc: {:.3f}, classes: {} imgnum: {}".format(cluster_acc,len(set(psolabels))-1,num))
    alltrain_data_loader, alltrain_camstyle_data_loader = make_camstyle_alltrain_data_loader(cfg, psolabels)

    alltrain_loader_iter = cycle(alltrain_data_loader)
    alltrain_camstyle_loader_iter = cycle(alltrain_camstyle_data_loader)

    @trainer.on(Events.ITERATION_STARTED)
    def generate_batch(engine):
        current_iter = engine.state.iteration
        batch = next(alltrain_loader_iter)
        camstyle_batch = next(alltrain_camstyle_loader_iter)
        engine.state.batch = [batch, camstyle_batch]

    num_iters = len(alltrain_data_loader)
    data = list(range(num_iters))
    trainer.run(data, max_epochs=epochs)
