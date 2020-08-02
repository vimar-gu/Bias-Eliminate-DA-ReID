# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch


def train_collate_fn(batch):
    imgs, pids, _, _, setids, _ = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, setids

def val_collate_fn(batch):
    imgs, pids, camids, _, _, trkids = zip(*batch)
    return torch.stack(imgs, dim=0), pids, camids, trkids

def test_collate_fn(batch):
    imgs, pids, camids, impaths, _, _ = zip(*batch)
    return torch.stack(imgs, dim=0), pids, camids, impaths
