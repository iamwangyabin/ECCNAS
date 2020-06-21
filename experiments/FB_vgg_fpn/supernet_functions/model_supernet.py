import torch
from torch import nn
from torch.nn import functional as F
from fbnet_building_blocks.fbnet_builder import ConvBNRelu, Flatten
from supernet_functions.config_for_supernet import CONFIG_SUPERNET
from supernet_functions.loss import Post_Prob, Bay_Loss
import torch.utils.checkpoint as checkpoint



class MixedOperation(nn.Module):
    def __init__(self, layer_parameters, proposed_operations, latency):
        super(MixedOperation, self).__init__()
        ops_names = [op_name for op_name in proposed_operations]
        self.ops = nn.ModuleList([proposed_operations[op_name](*layer_parameters)
                                  for op_name in ops_names])
        self.thetas = nn.Parameter(torch.Tensor([1.0 / len(ops_names) for i in range(len(ops_names))]))
    
    def forward(self, x, temperature):
        soft_mask_variables = nn.functional.gumbel_softmax(self.thetas, temperature)
        output  = sum(m * op(x) for m, op in zip(soft_mask_variables, self.ops))
        return output









def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=5, padding=2)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512]
}


class FBNet_Stochastic_SuperNet(nn.Module):
    def __init__(self, branch1, branch2, branch3):
        super(FBNet_Stochastic_SuperNet, self).__init__()
        self.features = make_layers(cfg['E'])
        # import pdb;pdb.set_trace()
        self.features1 = self.features[:4]
        self.features2 = self.features[4:9]
        self.features3 = self.features[9:18]
        self.features4 = self.features[18:27]
        self.features5 = self.features[27:]

        self.reg_layer_mid = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 1, 1)
        )

        self.reg_layer = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, 1)
        )
    def forward(self, x, temperature):
        y = self.features1(x)
        y = self.features2(y)
        y = self.features3(y)
        y4 = self.features4(y)
        y5 = self.features5(y4)
        y4 = self.reg_layer_mid(y4)
        y5 = self.reg_layer_mid(y5)
        y5 = F.upsample_bilinear(y5, scale_factor=2)
        # y = self.reg_layer(y)
        return torch.abs(y4), torch.abs(y5)

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
