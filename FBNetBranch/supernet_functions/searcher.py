import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate

import pytorch_lightning as pl

import os
import numpy as np
from general_functions.crowd import Crowd
import supernet_functions.lookup_table_builder as ltb
from supernet_functions.lookup_table_builder import LookUpTable
from supernet_functions.model_supernet import FBNet_Stochastic_SuperNet, SupernetLoss

from general_functions.utils import weights_init, check_tensor_in_list

def train_collate(batch):
    transposed_batch = list(zip(*batch))
    images = torch.stack(transposed_batch[0], 0)
    points = transposed_batch[1]  # the number of points is not fixed, keep it as a list of tensor
    targets = transposed_batch[2]
    st_sizes = torch.FloatTensor(transposed_batch[3])
    return images, points, targets, st_sizes, transposed_batch[4]


class NAS(pl.LightningModule):
    def __init__(self, config):
        super(NAS, self).__init__()
        self.cfg = config
        self.temperature = self.cfg['train_settings']['init_temperature']
        self.exp_anneal_rate = self.cfg['train_settings']['exp_anneal_rate']  # apply it every epoch
        self.cnt_epochs = self.cfg['train_settings']['cnt_epochs']
        self.train_thetas_from_the_epoch = self.cfg['train_settings']['train_thetas_from_the_epoch']

        self.branch1_lookup_table = LookUpTable(search_space=ltb.BRANCH1_SEARCH_SPACE,
                                           calulate_latency=self.cfg['lookup_table']['create_from_scratch'],
                                           brancename="branch1.txt")
        self.branch2_lookup_table = LookUpTable(search_space=ltb.BRANCH2_SEARCH_SPACE,
                                           calulate_latency=self.cfg['lookup_table']['create_from_scratch'],
                                           brancename="branch2.txt")
        self.branch3_lookup_table = LookUpTable(search_space=ltb.BRANCH3_SEARCH_SPACE,
                                           calulate_latency=self.cfg['lookup_table']['create_from_scratch'],
                                           brancename="branch3.txt")

        self.model = FBNet_Stochastic_SuperNet(self.branch1_lookup_table, self.branch2_lookup_table, self.branch3_lookup_table)
        self.model = self.model.apply(weights_init)

        self.loss = SupernetLoss(self.cfg['loss']['sigma'],
                                 self.cfg['dataloading']['crop_size'],
                                 self.cfg['loss']['downsample_ratio'],
                                 self.cfg['loss']['background_ratio'],
                                 self.cfg['loss']['use_background'])

    def forward(self, x, *args, **kwargs):
        return self.model(x)

    def prepare_data(self):
        self.datasets = {x: Crowd(os.path.join(self.cfg['dataloading']['dataset_path'], x),
                             self.cfg['dataloading']['crop_size'],
                             self.cfg['dataloading']['downsample_ratio'],
                             self.cfg['dataloading']['is_gray'], x) for x in ['train', 'val']}
        self.dataloaders = {x: DataLoader(self.datasets[x],
                                     collate_fn=(train_collate if x == 'train' else default_collate),
                                     batch_size=(self.cfg['dataloading']['batch_size'] if x == 'train' else 1),
                                     shuffle=(True if x == 'train' else False),
                                     num_workers=self.cfg['dataloading']['num_workers'],
                                     pin_memory=(True if x == 'train' else False))
                       for x in ['train', 'val']}

    def train_dataloader(self):
        return self.dataloaders['train']

    def val_dataloader(self):
        return self.dataloaders['val']

    def configure_optimizers(self):
        thetas_params = [param for name, param in self.model.named_parameters() if 'thetas' in name]
        params_except_thetas = [param for param in self.model.parameters() if not check_tensor_in_list(param, thetas_params)]

        w_optimizer = torch.optim.SGD(params=params_except_thetas,
                                      lr=self.cfg['optimizer']['w_lr'],
                                      momentum=self.cfg['optimizer']['w_momentum'],
                                      weight_decay=self.cfg['optimizer']['w_weight_decay'])
        theta_optimizer = torch.optim.Adam(params=thetas_params,
                                           lr=self.cfg['optimizer']['thetas_lr'],
                                           weight_decay=self.cfg['optimizer']['thetas_weight_decay'])
        last_epoch = -1
        w_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(w_optimizer,
                                                                 T_max=self.cfg['train_settings']['cnt_epochs'],
                                                                 last_epoch=last_epoch)

        return [w_optimizer, theta_optimizer] #, [w_scheduler, None]

    def training_step(self, batch, batch_idx):
        import pdb;pdb.set_trace()
        inputs, points, targets, st_sizes, cood = batch
        latency_to_accumulate = Variable(torch.Tensor([[0.0]]), requires_grad=True)
        outs, latency_to_accumulate = self.model(inputs, self.temperature, latency_to_accumulate)
        loss = self.loss(outs, points, targets, st_sizes, cood, latency_to_accumulate)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        inputs, count, name = batch
        latency_to_accumulate = Variable(torch.Tensor([[0.0]]), requires_grad=True).to(inputs.device)
        outs, latency_to_accumulate = self.model(inputs, self.temperature, latency_to_accumulate)
        res = count[0].item() - torch.sum(outs).item()
        return {"batch_val_res": res}

    def validation_epoch_end(self, outputs):
        epoch_res = []
        for i in outputs:
            epoch_res.append(i["batch_val_res"])
        epoch_res = np.array(epoch_res)
        mse = np.sqrt(np.mean(np.square(epoch_res)))
        mae = np.mean(np.abs(epoch_res))
        return {"log": {"val_mse": mse, "val_mae": mae}}

    def optimizer_step(self, current_epoch, batch_nb, optimizer, optimizer_i, second_order_closure=None):
        # w optimization
        # if optimizer_i == 0:
        #     if current_epoch % 2 == 0:
        #         optimizer.step()
        #         optimizer.zero_grad()
        if optimizer_i == 1:
            if current_epoch <= self.cfg['train_settings']['train_thetas_from_the_epoch'] or current_epoch % 2 == 1:
                optimizer.step()
                optimizer.zero_grad()


# check_val_every_n_epoch=1