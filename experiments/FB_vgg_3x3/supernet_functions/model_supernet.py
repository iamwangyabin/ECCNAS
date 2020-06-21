import torch
from torch import nn
from torch.nn import functional as F
from fbnet_building_blocks.fbnet_builder import ConvBNRelu, Flatten
from supernet_functions.config_for_supernet import CONFIG_SUPERNET
from supernet_functions.loss import Post_Prob, Bay_Loss
import torch.utils.checkpoint as checkpoint

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

def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
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
        # self.first = ConvBNRelu(input_depth=3, output_depth=16, kernel=3, stride=1,
        #                         pad=1, no_bias=1, use_relu="relu", bn_type="bn")
        # self.branch1_to_search = nn.ModuleList([MixedOperation(
        #                                            branch1.layers_parameters[layer_id],
        #                                            branch1.lookup_table_operations,
        #                                            branch1.lookup_table_latency[layer_id])
        #                                        for layer_id in range(branch1.cnt_layers)])
        # self.branch2_to_search = nn.ModuleList([MixedOperation(
        #                                            branch2.layers_parameters[layer_id],
        #                                            branch2.lookup_table_operations,
        #                                            branch2.lookup_table_latency[layer_id])
        #                                        for layer_id in range(branch2.cnt_layers)])
        # self.branch3_to_search = nn.ModuleList([MixedOperation(
        #                                            branch3.layers_parameters[layer_id],
        #                                            branch3.lookup_table_operations,
        #                                            branch3.lookup_table_latency[layer_id])
        #                                        for layer_id in range(branch3.cnt_layers)])
        # self.last_stages = nn.Sequential(
        #     ConvBNRelu(input_depth=512, output_depth=64, kernel=3, stride=1,
        #                pad=1, no_bias=1, use_relu="relu", bn_type="bn"),
        #     nn.Conv2d(64,1,1)
        # )
        self.features = make_layers(cfg['E'])
        self.reg_layer = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, 1)
        )

    # def custom(self, module):
    #     def custom_forward(*inputs):
    #         output = module(inputs[0][0], inputs[0][1], inputs[0][2])
    #         return output
    #     return custom_forward

    def forward(self, x, temperature):
        # y = self.first(x)
        # for mixed_op in self.branch1_to_search:
        #     y1, latency_to_accumulate = checkpoint.checkpoint(self.custom(mixed_op),[y1, temperature, latency_to_accumulate])
        # for mixed_op in self.branch2_to_search:
        #     y2, latency_to_accumulate = mixed_op(y2, temperature, latency_to_accumulate)
        #     y2, latency_to_accumulate = checkpoint.checkpoint(self.custom(mixed_op),[y2, temperature, latency_to_accumulate])
        # for mixed_op in self.branch3_to_search:
        #     y = mixed_op(y, temperature)
        # y1 = torch.nn.Upsample(scale_factor=0.125, mode="bilinear")(y1)
        # y = torch.cat([y2,y3], dim=1)
        # y = self.last_stages(y)
        y = self.features(x)
        y = F.upsample_bilinear(y, scale_factor=2)
        y = self.reg_layer(y)
        return torch.abs(y)

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

