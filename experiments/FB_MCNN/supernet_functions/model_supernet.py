import torch
from torch import nn
from torch.nn import functional as F
from fbnet_building_blocks.fbnet_builder import ConvBNRelu, Flatten
from supernet_functions.config_for_supernet import CONFIG_SUPERNET
from supernet_functions.loss import Post_Prob, Bay_Loss
import torch.utils.checkpoint as checkpoint
from torchvision import models

class MixedOperation(nn.Module):
    # Arguments:
    # proposed_operations is a dictionary {operation_name : op_constructor}
    def __init__(self, layer_parameters, proposed_operations, latency):
        super(MixedOperation, self).__init__()
        ops_names = [op_name for op_name in proposed_operations]
        self.ops = nn.ModuleList([proposed_operations[op_name](*layer_parameters)
                                  for op_name in ops_names])
        self.thetas = nn.Parameter(torch.Tensor([1.0 / len(ops_names) for i in range(len(ops_names))]))
    
    def forward(self, x, temperature, latency_to_accumulate):
        soft_mask_variables = nn.functional.gumbel_softmax(self.thetas, temperature)
        output  = sum(m * op(x) for m, op in zip(soft_mask_variables, self.ops))
        return output, latency_to_accumulate

class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, relu=True, same_padding=False, bn=False):
        super(Conv2d, self).__init__()
        padding = int((kernel_size - 1) / 2) if same_padding else 0
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class FBNet_Stochastic_SuperNet(nn.Module):
    def __init__(self):
        super(FBNet_Stochastic_SuperNet, self).__init__()
        bn = False
        self.branch1 = nn.Sequential(Conv2d(3, 16, 9, same_padding=True, bn=bn),
                                     nn.MaxPool2d(2),
                                     Conv2d(16, 32, 7, same_padding=True, bn=bn),
                                     nn.MaxPool2d(2),
                                     Conv2d(32, 16, 7, same_padding=True, bn=bn),
                                     nn.MaxPool2d(2),
                                     Conv2d(16, 8, 7, same_padding=True, bn=bn, relu=False))

        self.branch2 = nn.Sequential(Conv2d(3, 20, 7, same_padding=True, bn=bn),
                                     nn.MaxPool2d(2),
                                     Conv2d(20, 40, 5, same_padding=True, bn=bn),
                                     nn.MaxPool2d(2),
                                     Conv2d(40, 20, 5, same_padding=True, bn=bn),
                                     nn.MaxPool2d(2),
                                     Conv2d(20, 10, 5, same_padding=True, bn=bn, relu=False))

        self.branch3 = nn.Sequential(Conv2d(3, 24, 5, same_padding=True, bn=bn),
                                     nn.MaxPool2d(2),
                                     Conv2d(24, 48, 3, same_padding=True, bn=bn),
                                     nn.MaxPool2d(2),
                                     Conv2d(48, 24, 3, same_padding=True, bn=bn),
                                     nn.MaxPool2d(2),
                                     Conv2d(24, 12, 3, same_padding=True, bn=bn, relu=False))

        self.fuse = nn.Sequential(Conv2d(30, 1, 1, same_padding=True, bn=bn, relu=False))

    def forward(self, im_data, temperature):
        x1 = self.branch1(im_data)
        x2 = self.branch2(im_data)
        x3 = self.branch3(im_data)
        x = torch.cat((x1, x2, x3), 1)
        x = self.fuse(x)
        return torch.abs(x)

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