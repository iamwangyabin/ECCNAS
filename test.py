from __future__ import division
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '7,8,9'
import torch
import torch.nn as nn
import torch.utils
import numpy as np
from search.loss import Criterion
from search.config_search import config
from search.architect import Architect
from search.supernet import Network_Multi_Path as Network

def train_collate(batch):
    transposed_batch = list(zip(*batch))
    images = torch.stack(transposed_batch[0], 0)
    points = transposed_batch[1]  # the number of points is not fixed, keep it as a list of tensor
    targets = transposed_batch[2]
    st_sizes = torch.FloatTensor(transposed_batch[3])
    return images, points, targets, st_sizes


def main():
    # preparation ################
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    seed = config.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = Criterion(config.sigma, config.crop_size, config.downsample_ratio, config.background_ratio, config.use_background, device)
    model = Network(config.layers, criterion, Fch=config.Fch, width_mult_list=config.width_mult_list,
                    prun_modes=config.prun_modes, stem_head_width=config.stem_head_width)
    # model = nn.DataParallel(model)
    model.to(device)
    sample = torch.rand([1,3,512,512]).cuda()
    architect = Architect(model, config)
    for i in range(10000):
        model(sample)

if __name__ == '__main__':
    main()
