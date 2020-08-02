# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import logging
import time

import torch
import torch.nn as nn
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint, Timer
from ignite.metrics import RunningAverage

from layers import make_loss
from utils.reid_metric import R1_mAP

import torch.distributed as dist
from torch.distributed import get_rank, get_world_size

def create_supervised_trainer(model, optimizer, loss_fn,
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
        img, target = batch[0][0], batch[0][1]
        img = img.to(device) if torch.cuda.device_count() >= 1 else img
        target = target.to(device) if torch.cuda.device_count() >= 1 else target
        feats = model(img)
        loss = torch.tensor(0.).cuda()
        for i in range(len(loss_fn[0])):
            loss += loss_fn[0][i](feats[0], feats[-1], target)
        
        img_camstyle, target_camstyle = batch[1][0], batch[1][1] 
        img_camstyle = img_camstyle.to(device) if torch.cuda.device_count() >= 1 else img_camstyle
        target_camstyle = target_camstyle.to(device) if torch.cuda.device_count() >= 1 else target_camstyle
        feats_camstyle = model(img_camstyle)
        loss_camstyle = torch.tensor(0.).cuda()
        for i in range(len(loss_fn[0])):
            loss_camstyle += loss_fn[0][i](feats_camstyle[0], feats_camstyle[-1], target_camstyle)

        # compute loss
        global_loss = loss + loss_camstyle
        global_loss.backward()
        optimizer.step()
        # compute acc
        acc = (feats[0].max(1)[1] == target).float().mean()
        acc_camstyle = (feats_camstyle[0].max(1)[1] == target_camstyle).float().mean()
        global_acc = (acc + acc_camstyle) / 2.0
        # real_data/cam_data ratio
        ratio = target.shape[0] / target_camstyle.shape[0]
        return global_loss.item(), global_acc.item(), ratio
    
    return Engine(_update)

def create_supervised_evaluator(model, metrics,
                                device=None, device_id=-1, distribute=False):
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


def do_train(
        cfg,
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        loss_fn,
        num_query,
        start_epoch,
        device_id,
        train_camstyle_loader
):

    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD
    output_dir = cfg.OUTPUT_DIR
    epochs = cfg.SOLVER.MAX_EPOCHS
    device = cfg.MODEL.DEVICE

    logger = logging.getLogger("reid_baseline.train")
    logger.info("Start training")
    trainer = create_supervised_trainer(model, optimizer, loss_fn, device=device, device_id=device_id)
    evaluator = create_supervised_evaluator(model, metrics={'r1_mAP': R1_mAP(num_query, True, False, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)}, device=device, device_id=device_id)
    if device_id == 0:
        checkpointer = ModelCheckpoint(output_dir, cfg.MODEL.NAME, checkpoint_period, n_saved=10, require_empty=False)
        trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpointer, {'model': model.state_dict(),
                                                                     'optimizer': optimizer.state_dict()})

    timer = Timer(average=True)
    timer.attach(trainer, start=Events.EPOCH_STARTED, resume=Events.ITERATION_STARTED,
                 pause=Events.ITERATION_COMPLETED, step=Events.ITERATION_COMPLETED)

    # average metric to attach on trainer
    RunningAverage(output_transform=lambda x: x[0]).attach(trainer, 'avg_loss')
    RunningAverage(output_transform=lambda x: x[1]).attach(trainer, 'avg_acc')
    RunningAverage(output_transform=lambda x: x[2]).attach(trainer, 'data_ratio')

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

    train_loader_iter = cycle(train_loader)
    train_camstyle_loader_iter = cycle(train_camstyle_loader)

    @trainer.on(Events.ITERATION_STARTED)
    def generate_batch(engine):
        current_iter = engine.state.iteration
        batch = next(train_loader_iter)
        camstyle_batch = next(train_camstyle_loader_iter)
        engine.state.batch = [batch, camstyle_batch]

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        iter = (engine.state.iteration - 1) % len(train_loader) + 1
        if iter % log_period == 0:
            logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, ratio of data/cam_data: {:.3f}, Base Lr: {:.2e}"
                        .format(engine.state.epoch, iter, len(train_loader),
                                engine.state.metrics['avg_loss'], engine.state.metrics['avg_acc'], engine.state.metrics['data_ratio'],
                                scheduler.get_lr()[0]))

    # adding handlers using `trainer.on` decorator API
    @trainer.on(Events.EPOCH_COMPLETED)
    def print_times(engine):
        logger.info('Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]'
                    .format(engine.state.epoch, timer.value() * timer.step_count,
                            train_loader.batch_size / timer.value()))
        logger.info('-' * 10)
        timer.reset()

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        if engine.state.epoch % eval_period == 0:
            evaluator.run(val_loader)
            cmc, mAP = evaluator.state.metrics['r1_mAP']
            logger.info("Validation Results - Epoch: {}".format(engine.state.epoch))
            logger.info("mAP: {:.1%}".format(mAP))
            for r in [1, 5, 10]:
                logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))

    num_iters = len(train_loader)
    data = list(range(num_iters))

    trainer.run(data, max_epochs=epochs)
   

