# encoding: utf-8

import copy
import random
import torch
from collections import defaultdict

import math
import numpy as np
from torch.utils.data.sampler import Sampler
from .classifier_sampler import RandomIdentitySampler_ImgUniform, RandomIdentitySampler_IdUniform


def Select_Sampler(dataset,info,batchsize):
    if info.SAMPLER_UNIFORM == 'img':
        sampler = RandomIdentitySampler_ImgUniform(dataset, batchsize, info.NUM_INSTANCE)
    elif info.SAMPLER_UNIFORM == 'id':
        sampler = RandomIdentitySampler_IdUniform(dataset, batchsize, info.NUM_INSTANCE)
    return sampler


class Sampler_All(Sampler):
    def __init__(self, datasets, infos, probs, batchsize):
        assert(len(datasets)==len(infos))
        assert(len(datasets)==len(probs))
        self.samplers = [Select_Sampler(datasets[i],infos[i],batchsize) for i in range(len(infos))]
        self.set_probs = probs
        self.set_idxs = np.arange(0,len(infos),1)
        self.offset = [len(i) for i in datasets]
        self.offset = [0] + self.offset[:-1]
        print('offset:',self.offset)

        self.length = max(len(self.samplers[0]),len(self.samplers[1]))
        print('max batches:',[len(s) for s in self.samplers])

    def __iter__(self):
        ret = []
        while len(ret) <= self.length:
            cur_set_idx = int(np.random.choice(self.set_idxs, size=1, replace=False, p=self.set_probs))
            cur_ret = next(self.samplers[cur_set_idx])
            cur_ret = [i + sum(self.offset[:cur_set_idx+1]) for i in cur_ret]
            ret.extend(cur_ret)
        return iter(ret)

    def __len__(self):
        return self.length
