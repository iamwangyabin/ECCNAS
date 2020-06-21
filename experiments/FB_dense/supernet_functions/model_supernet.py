import torch
from torch import nn
from torch.nn import functional as F
from supernet_functions.config_for_supernet import CONFIG_SUPERNET
from supernet_functions.loss import Post_Prob, Bay_Loss

class SupernetLoss(nn.Module):
    def __init__(self, sigma, crop_size, downsample_ratio, background_ratio, use_background):
        super(SupernetLoss, self).__init__()
        self.post_prob = Post_Prob(sigma, crop_size, downsample_ratio, background_ratio, use_background)
        self.criterion = Bay_Loss(use_background)
        self.beta = CONFIG_SUPERNET["loss"]['beta']
        self.alpha = CONFIG_SUPERNET["loss"]['alpha']

    def forward(self, pred, points, target, st_sizes):
        prob_list = self.post_prob(points, st_sizes)
        ce = self.criterion(prob_list, target, pred)
        return ce
