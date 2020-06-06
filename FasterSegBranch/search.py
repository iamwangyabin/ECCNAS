from __future__ import division
import os
import sys
import time
import glob
import logging
from tqdm import tqdm

os.environ['CUDA_VISIBLE_DEVICES'] = '7,8,9'

import torch
import torch.nn as nn
import torch.utils
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from tensorboardX import SummaryWriter
import numpy as np
import matplotlib
matplotlib.use('Agg')
from thop import profile
from search.config_search import config
from search.crowd_dataloader import Crowd
from tools.utils.init_func import init_weight
from search.loss import Criterion
from search.architect import Architect
from tools.utils.darts_utils import create_exp_dir, save, plot_op, plot_path_width, objective_acc_lat
from search.supernet import Network_Multi_Path as Network


def train_collate(batch):
    transposed_batch = list(zip(*batch))
    images = torch.stack(transposed_batch[0], 0)
    points = transposed_batch[1]  # the number of points is not fixed, keep it as a list of tensor
    targets = transposed_batch[2]
    st_sizes = torch.FloatTensor(transposed_batch[3])
    return images, points, targets, st_sizes

def main(pretrain=True):
    config.save = 'search-{}-{}'.format(config.save, time.strftime("%Y%m%d-%H%M%S"))
    create_exp_dir(config.save, scripts_to_save=glob.glob('*.py')+glob.glob('*.sh'))
    logger = SummaryWriter(config.save)

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(config.save, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    assert type(pretrain) == bool or type(pretrain) == str
    update_arch = True
    if pretrain == True:
        update_arch = False
    logging.info("args = %s", str(config))
    # preparation ################
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    seed = config.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # config network and criterion ################
    criterion = Criterion(config.sigma, config.crop_size, config.downsample_ratio, config.background_ratio, config.use_background, device)
    model = Network(config.layers, criterion, Fch=config.Fch, width_mult_list=config.width_mult_list, prun_modes=config.prun_modes, stem_head_width=config.stem_head_width)
    flops, params = profile(model, inputs=(torch.randn(1, 3, 512, 512),), verbose=False)
    model.to(device)
    logging.info("params = %fMB, FLOPs = %fGB", params / 1e6, flops / 1e9)

    if type(pretrain) == str:
        partial = torch.load(pretrain + "/weights.pt", map_location='cuda:0')
        state = model.state_dict()
        pretrained_dict = {k: v for k, v in partial.items() if k in state and state[k].size() == partial[k].size()}
        state.update(pretrained_dict)
        model.load_state_dict(state)
    else:
        init_weight(model, nn.init.kaiming_normal_, nn.BatchNorm2d, config.bn_eps, config.bn_momentum, mode='fan_in', nonlinearity='relu')

    architect = Architect(model, config)

    # Optimizer ###################################
    base_lr = config.lr
    parameters = []
    parameters += list(model.stem.parameters())
    parameters += list(model.cells.parameters())
    parameters += list(model.refine32.parameters())
    parameters += list(model.refine16.parameters())
    parameters += list(model.head0.parameters())
    parameters += list(model.head1.parameters())
    parameters += list(model.head2.parameters())
    parameters += list(model.head02.parameters())
    parameters += list(model.head12.parameters())
    optimizer = torch.optim.SGD(
        parameters,
        lr=base_lr,
        momentum=config.momentum,
        weight_decay=config.weight_decay)

    # lr policy ##############################
    lr_policy = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.978)
    # data loader ###########################
    device_count = torch.cuda.device_count()
    datasets = {x: Crowd(os.path.join(config.dataset_path, x),
                          config.crop_size,
                          config.downsample_ratio,
                          config.is_gray, x) for x in ['train', 'val']}
    dataloaders = {x: DataLoader(datasets[x],
                          collate_fn=(train_collate if x == 'train' else default_collate),
                          batch_size=(config.batch_size if x == 'train' else 1),
                          shuffle=(True if x == 'train' else False),
                          num_workers=config.num_workers * device_count,
                          pin_memory=(True if x == 'train' else False))
                    for x in ['train', 'val']}

    train_loader_model = dataloaders['train']
    val_loader_model = dataloaders['val']

    if update_arch:
        for idx in range(len(config.latency_weight)):
            logger.add_scalar("arch/latency_weight%d"%idx, config.latency_weight[idx], 0)
            logging.info("arch_latency_weight%d = "%idx + str(config.latency_weight[idx]))

    tbar = tqdm(range(config.nepochs), ncols=80)

    for epoch in tbar:
        logging.info(pretrain)
        logging.info(config.save)
        logging.info("lr: " + str(optimizer.param_groups[0]['lr']))
        logging.info("update arch: " + str(update_arch))

        # training
        tbar.set_description("[Epoch %d/%d][train...]" % (epoch + 1, config.nepochs))
        train(pretrain, train_loader_model, train_loader_model, model, architect, criterion, optimizer, lr_policy, logger, epoch, update_arch=update_arch)
        torch.cuda.empty_cache()
        lr_policy.step()

        # validation
        tbar.set_description("[Epoch %d/%d][validation...]" % (epoch + 1, config.nepochs))
        mse, mae = eval(model, val_loader_model, pretrain)
        logging.info("Val mse: %f mae: %f " % (mse, mae))
        torch.save(model, os.path.join(config.save, "arch_%d_%f_%f.pt" % (epoch, mse, mae)))


def train(pretrain, train_loader_model, train_loader_arch, model, architect, criterion, optimizer, lr_policy, logger, epoch, update_arch=True):
    model.train()

    bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
    pbar = tqdm(range(len(train_loader_model)), file=sys.stdout, bar_format=bar_format, ncols=80)
    dataloader_model = iter(train_loader_model)
    dataloader_arch = iter(train_loader_arch)

    for step in pbar:
        optimizer.zero_grad()
        try:
            inputs, points, targets, st_sizes = dataloader_model.next()
        except StopIteration:
            dataloader_model = iter(train_loader_model)
            inputs, points, targets, st_sizes = dataloader_model.next()

        inputs = inputs.cuda(non_blocking=True)
        st_sizes = st_sizes.cuda(non_blocking=True)
        if update_arch:
            # get a random minibatch from the search queue with replacement
            try:
                inputs_search, points_search, targets_search, st_sizes_search = dataloader_arch.next()
            except StopIteration:
                dataloader_arch = iter(train_loader_arch)
                inputs_search, points_search, targets_search, st_sizes_search = dataloader_arch.next()
            inputs_search = inputs_search.cuda(non_blocking=True)
            st_sizes_search = st_sizes_search.cuda(non_blocking=True)

            loss_arch = architect.step(inputs, points, targets, st_sizes, inputs_search, points_search, targets_search, st_sizes_search)

            pbar.set_description("[Arch Step %d/%d Loss %f]" % (step + 1, len(train_loader_model), loss_arch.item()))

            if (step+1) % 10 == 0:
                logger.add_scalar('loss_arch/train', loss_arch, epoch*len(pbar)+step)
                logger.add_scalar('arch/latency_supernet', architect.latency_supernet, epoch*len(pbar)+step)

        loss = model._loss(inputs, points, targets, st_sizes, pretrain)
        logger.add_scalar('loss/train', loss, epoch*len(pbar)+step)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
        optimizer.zero_grad()

        pbar.set_description("[Step %d/%d Loss %f]" % (step + 1, len(train_loader_model), loss.item()))
    torch.cuda.empty_cache()


def eval(model, data_loader, pretrain):
    epoch_res = []
    with torch.no_grad():
        bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
        pbar = tqdm(range(len(data_loader)), file=sys.stdout, bar_format=bar_format, ncols=80)
        dataloader_model = iter(data_loader)
        for step in pbar:
            inputs, count, name = dataloader_model.next()
            inputs = inputs.cuda(non_blocking=True)
            if pretrain is not True:
                # "random width": sampled by gambel softmax
                model.prun_mode = None
                for idx in range(len(model._arch_names)):
                    model.arch_idx = idx
                    outputs = model(inputs)
                    res = count[0].item() - torch.sum(outputs[0]).item()
                    epoch_res.append(res)
            if len(model._width_mult_list) > 1:
                model.prun_mode = "max"
                outputs = model(inputs)
                res = count[0].item() - torch.sum(outputs[0]).item()
                epoch_res.append(res)
                model.prun_mode = "min"
                outputs = model(inputs)
                res = count[0].item() - torch.sum(outputs[0]).item()
                epoch_res.append(res)
                if pretrain == True:
                    model.prun_mode = "random"
                    outputs = model(inputs)
                    res = count[0].item() - torch.sum(outputs[0]).item()
                    epoch_res.append(res)
                    model.prun_mode = "random"
                    outputs = model(inputs)
                    res = count[0].item() - torch.sum(outputs[0]).item()
                    epoch_res.append(res)
            elif pretrain == True and len(model._width_mult_list) == 1:
                model.prun_mode = "max"
                outputs = model(inputs)
                res = count[0].item() - torch.sum(outputs[0]).item()
                epoch_res.append(res)
            pbar.set_description("[Val Step %d/%d]" % (step + 1, len(data_loader)))
        epoch_res = np.array(epoch_res)
        mse = np.sqrt(np.mean(np.square(epoch_res)))
        mae = np.mean(np.abs(epoch_res))
    torch.cuda.empty_cache()
    return mse, mae


if __name__ == '__main__':
    main(pretrain=config.pretrain) 
