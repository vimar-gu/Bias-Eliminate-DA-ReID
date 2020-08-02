# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import copy
from torch.utils.data import DataLoader

from .collate_batch import train_collate_fn, val_collate_fn, test_collate_fn
from .datasets import init_dataset, add_psolabels, add_camstyle_psolabels, ImageDataset, CamStyleImageDataset, CameraImageDataset
from .samplers import Sampler_All, RandomIdentitySampler_ImgUniform, RandomIdentitySampler_IdUniform
from .transforms import build_transforms
import numpy as np


def make_val_data_loader(cfg):
    val_transforms = build_transforms(cfg, is_train=False)

    dataset = init_dataset(cfg.VAL_DATA.NAMES, root_val=cfg.VAL_DATA.TRAIN_DIR)

    val_set = ImageDataset(dataset.query + dataset.gallery, val_transforms, cfg.INPUT.SIZE_TEST)
    val_loader = DataLoader(
        val_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=cfg.DATALOADER.NUM_WORKERS,
        collate_fn=test_collate_fn
    )

    return val_loader, len(dataset.query)


def make_test_data_loader(cfg):
    test_transforms = build_transforms(cfg, is_train=False)

    dataset = init_dataset(cfg.TEST_DATA.NAMES, root_val=cfg.TEST_DATA.TRAIN_DIR)

    test_set = ImageDataset(dataset.query + dataset.gallery, test_transforms, cfg.INPUT.SIZE_TEST)
    test_loader = DataLoader(
        test_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=cfg.DATALOADER.NUM_WORKERS,
        collate_fn=test_collate_fn
    )

    return test_loader, len(dataset.query)


def make_camstyle_target_unsupdata_loader(cfg):
    val_transforms = build_transforms(cfg, is_train=False)

    dataset = init_dataset(cfg.TGT_UNSUPDATA.NAMES, root_train=cfg.TGT_UNSUPDATA.TRAIN_DIR)

    target_unsupdata_set = CamStyleImageDataset(dataset, val_transforms, cfg.INPUT.SIZE_TEST)

    target_unsupdata_loader = DataLoader(
        target_unsupdata_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=cfg.DATALOADER.NUM_WORKERS,
        collate_fn=val_collate_fn
    )
    return target_unsupdata_loader


def make_camstyle_data_loader(cfg):
    train_transforms = build_transforms(cfg, is_train=True)
    val_transforms = build_transforms(cfg, is_train=False)
    num_workers = cfg.DATALOADER.NUM_WORKERS

    dataset = init_dataset(cfg.SRC_DATA.NAMES, root_train=cfg.SRC_DATA.TRAIN_DIR, root_val=cfg.VAL_DATA.TRAIN_DIR, transfered=cfg.SRC_DATA.TRANSFERED)

    num_classes = [dataset.num_train_pids]
    train_set = ImageDataset(dataset.train, train_transforms, cfg.INPUT.SIZE_TRAIN)
    train_camstyle_set = ImageDataset(dataset.train_camstyle, train_transforms, cfg.INPUT.SIZE_TRAIN)
        
    if cfg.SRC_DATA.SAMPLER_UNIFORM == 'img':
        if cfg.SRC_DATA.NUM_INSTANCE == 1:
            train_loader = DataLoader(
                train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH, shuffle=True, num_workers=num_workers,
                collate_fn=train_collate_fn
            )
            train_camstyle_loader = DataLoader(
                train_camstyle_set, batch_size=cfg.SOLVER.IMS_PER_BATCH, shuffle=True, num_workers=num_workers,
                collate_fn=train_collate_fn
            )
        else:
            src_batch = cfg.DATALOADER.IMS_PER_BATCH * 1 // 2
            camstyle_batch = cfg.DATALOADER.IMS_PER_BATCH * 1 // 2
            train_loader = DataLoader(
                train_set, batch_size=src_batch,
                sampler=RandomIdentitySampler_ImgUniform(dataset.train, src_batch, cfg.SRC_DATA.NUM_INSTANCE),
                num_workers=num_workers, collate_fn=train_collate_fn
            )
            train_camstyle_loader = DataLoader(
                train_camstyle_set, batch_size=camstyle_batch,
                sampler=RandomIdentitySampler_ImgUniform(dataset.train_camstyle, camstyle_batch, cfg.SRC_DATA.NUM_INSTANCE),
                num_workers=num_workers, collate_fn=train_collate_fn
            )

    elif cfg.DATALOADER.SAMPLER_UNIFORM == 'id':
        src_batch = cfg.DATALOADER.IMS_PER_BATCH * 1 // 2
        camstyle_batch = cfg.DATALOADER.IMS_PER_BATCH * 1 // 2
        train_loader = DataLoader(
            train_set, batch_size=src_batch,
            sampler=RandomIdentitySampler_IdUniform(dataset.train, src_batch, cfg.SRC_DATA.NUM_INSTANCE),
            num_workers=num_workers, collate_fn=train_collate_fn, drop_last=True,
        )
        train_camstyle_loader = DataLoader(
            train_camstyle_set, batch_size=camstyle_batch,
            sampler=RandomIdentitySampler_IdUniform(dataset.train_camstyle, camstyle_batch, cfg.DATALOADER.NUM_INSTANCE),
            num_workers=num_workers, collate_fn=train_collate_fn, drop_last=True,
        )
    val_set = ImageDataset(dataset.query + dataset.gallery, val_transforms, cfg.INPUT.SIZE_TRAIN)
    val_loader = DataLoader(
        val_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn
    )

    return train_loader, train_camstyle_loader, val_loader, len(dataset.query), num_classes


def make_camstyle_alltrain_data_loader(cfg, psolabels):
    train_transforms = build_transforms(cfg, is_train=True)

    infos = [cfg.SRC_DATA,cfg.TGT_UNSUPDATA]

    datasets = [(init_dataset(infos[i].NAMES, root_train=infos[i].TRAIN_DIR, setid=i, transfered=infos[i].TRANSFERED)) for i in range(len(infos))]
    datasets_train = [datasets[i].train for i in range(len(infos))]
    datasets_camstyle = [datasets[i].train_camstyle for i in range(len(infos))]

    datasets_train[1] = add_psolabels(datasets_train[1], psolabels)
    datasets_camstyle[1] = add_camstyle_psolabels(datasets[1].train, datasets[1].camstyle_dict, psolabels)

    merged_dataset = []
    merged_camstyle_dataset = []
    for train in datasets_train:
        print ('len in datasets_train: ', len(train))
        merged_dataset.extend(train)
    for train_camstyle in datasets_camstyle:
        print ('len in datasets_camstyle: ', len(train_camstyle))
        merged_camstyle_dataset.extend(train_camstyle)

    src_batch = cfg.DATALOADER.IMS_PER_BATCH * 1 // 4
    camstyle_batch = cfg.DATALOADER.IMS_PER_BATCH * 3 // 4
    
    train_set = ImageDataset(merged_dataset, train_transforms, cfg.INPUT.SIZE_TRAIN)
    train_loader = DataLoader(
        train_set, batch_size=src_batch,
        sampler=Sampler_All(datasets_train, infos, cfg.DATALOADER.SAMPLER_PROB, src_batch),
        num_workers=cfg.DATALOADER.NUM_WORKERS, collate_fn=train_collate_fn, drop_last=True,
    )

    train_camstyle_set = ImageDataset(merged_camstyle_dataset, train_transforms, cfg.INPUT.SIZE_TRAIN)
    train_camstyle_loader = DataLoader(
        train_camstyle_set, batch_size=camstyle_batch,
        sampler=Sampler_All(datasets_camstyle, infos, cfg.DATALOADER.SAMPLER_PROB, camstyle_batch),
        num_workers=cfg.DATALOADER.NUM_WORKERS, collate_fn=train_collate_fn, drop_last=True,
    )

    return train_loader, train_camstyle_loader

def make_camera_data_loader(cfg):
    train_transforms = build_transforms(cfg, is_train=True)
    val_transforms = build_transforms(cfg, is_train=False)
    num_workers = cfg.DATALOADER.NUM_WORKERS

    dataset = init_dataset(cfg.SRC_DATA.NAMES, root_train=cfg.SRC_DATA.TRAIN_DIR, root_val=cfg.VAL_DATA.TRAIN_DIR, transfered=cfg.SRC_DATA.TRANSFERED)
    camera_dataset = copy.deepcopy(dataset)
    camera_dataset.train = []
    camera_dataset.query = []
    camera_dataset.gallery = []
    camid_dict = {0: 0, 1: 1, 3: 2, 4: 3, 5: 4}
    for im_path, pid, camid, setid, trkid, pidx in dataset.train:
        camera_dataset.train.append((im_path, camid_dict[camid], camid_dict[camid], setid, trkid, pidx))
    for im_path, pid, camid, setid, trkid, pidx in dataset.query:
        camera_dataset.query.append((im_path, camid_dict[camid], camid_dict[camid], setid, trkid, pidx))
    for im_path, pid, camid, setid, trkid, pidx in dataset.gallery:
        camera_dataset.gallery.append((im_path, camid_dict[camid], camid_dict[camid], setid, trkid, pidx))

    train_set = CameraImageDataset(camera_dataset.train, train_transforms, cfg.INPUT.SIZE_TRAIN)
        
    if cfg.SRC_DATA.SAMPLER_UNIFORM == 'img':
        if cfg.SRC_DATA.NUM_INSTANCE == 1:
            train_loader = DataLoader(
                train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH, shuffle=True, num_workers=num_workers,
                collate_fn=train_collate_fn
            )
        else:
            src_batch = cfg.DATALOADER.IMS_PER_BATCH
            train_loader = DataLoader(
                train_set, batch_size=src_batch,
                sampler=RandomIdentitySampler_ImgUniform(camera_dataset.train, src_batch, cfg.SRC_DATA.NUM_INSTANCE),
                num_workers=num_workers, collate_fn=train_collate_fn
            )

    elif cfg.DATALOADER.SAMPLER_UNIFORM == 'id':
        src_batch = cfg.DATALOADER.IMS_PER_BATCH
        train_loader = DataLoader(
            train_set, batch_size=src_batch,
            sampler=RandomIdentitySampler_IdUniform(camera_dataset.train, src_batch, cfg.SRC_DATA.NUM_INSTANCE),
            num_workers=num_workers, collate_fn=train_collate_fn, drop_last=True,
        )
    val_set = ImageDataset(camera_dataset.query + camera_dataset.gallery, val_transforms, cfg.INPUT.SIZE_TRAIN)
    val_loader = DataLoader(
        val_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn
    )

    return train_loader, val_loader, len(camera_dataset.query)

