# encoding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from easydict import EasyDict as edict

C = edict()
config = C
cfg = C

C.seed = 12345

"""Data Dir"""
C.dataset_path = r".\UCF-QNRF_ECCV18"

"""Criterion Parameters"""
C.sigma = 8.0
C.crop_size = 512
C.downsample_ratio = 8
C.background_ratio = 0.15
C.use_background = True
C.is_gray = False

""" Settings for network, this would be different for each kind of model"""
C.bn_eps = 1e-5
C.bn_momentum = 0.1

"""Train Config"""
C.lr = 0.01
C.momentum = 0.9
C.weight_decay = 5e-4
C.num_workers = 4
C.train_scale_array = [0.75, 1, 1.25]

""" Search Config """
C.grad_clip = 5
C.train_portion = 0.5
C.arch_learning_rate = 3e-4
C.arch_weight_decay = 0
C.layers = 16
C.branch = 2


'''预训练'''
C.pretrain = True
# C.pretrain = "search-pretrain-256x512_F12.L16_batch3-20200101-012345"
########################################
C.prun_modes = ['max', 'arch_ratio',]
C.Fch = 12
C.width_mult_list = [4./12, 6./12, 8./12, 10./12, 1.]
C.stem_head_width = [(1, 1), (8./12, 8./12),]
C.FPS_min = [0, 155.]
C.FPS_max = [0, 175.]
if C.pretrain == True:
    C.batch_size = 3
    C.niters_per_epoch = 400
    C.lr = 2e-2
    C.latency_weight = [0, 0]
    C.nepochs = 20
    C.save = "pretrain-F%d.L%d_batch%d"%(C.Fch, C.layers, C.batch_size)
else:
    C.batch_size = 2
    C.niters_per_epoch = 1800
    C.latency_weight = [0, 1e-2,]
    C.nepochs = 30
    C.save = "F%d.L%d_batch%d"%(C.Fch, C.layers, C.batch_size)
########################################
assert len(C.latency_weight) == len(C.stem_head_width) and len(C.stem_head_width) == len(C.FPS_min) and len(C.FPS_min) == len(C.FPS_max)

C.unrolled = False
