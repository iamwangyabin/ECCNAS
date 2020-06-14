import torch
from torch import nn
from fbnet_building_blocks.fbnet_builder import ConvBNRelu, Flatten
from supernet_functions.config_for_supernet import CONFIG_SUPERNET
from supernet_functions.loss import Post_Prob, Bay_Loss
import torch.utils.checkpoint as checkpoint

class MixedOperation(nn.Module):
    
    # Arguments:
    # proposed_operations is a dictionary {operation_name : op_constructor}
    # latency is a dictionary {operation_name : latency}
    def __init__(self, layer_parameters, proposed_operations, latency):
        super(MixedOperation, self).__init__()
        ops_names = [op_name for op_name in proposed_operations]
        
        self.ops = nn.ModuleList([proposed_operations[op_name](*layer_parameters)
                                  for op_name in ops_names])
        self.latency = [latency[op_name] for op_name in ops_names]
        self.thetas = nn.Parameter(torch.Tensor([1.0 / len(ops_names) for i in range(len(ops_names))]))
    
    def forward(self, x, temperature, latency_to_accumulate):
        soft_mask_variables = nn.functional.gumbel_softmax(self.thetas, temperature)
        output  = sum(m * op(x) for m, op in zip(soft_mask_variables, self.ops))
        latency = sum(m * lat for m, lat in zip(soft_mask_variables, self.latency))
        latency_to_accumulate = latency_to_accumulate + latency
        return output, latency_to_accumulate

class FBNet_Stochastic_SuperNet(nn.Module):
    def __init__(self, branch1, branch2, branch3):
        super(FBNet_Stochastic_SuperNet, self).__init__()
        
        # self.first identical to 'add_first' in the fbnet_building_blocks/fbnet_builder.py
        self.first = ConvBNRelu(input_depth=3, output_depth=16, kernel=3, stride=1,
                                pad=1, no_bias=1, use_relu="relu", bn_type="bn")
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

        self.branch3_to_search = nn.ModuleList([MixedOperation(
                                                   branch3.layers_parameters[layer_id],
                                                   branch3.lookup_table_operations,
                                                   branch3.lookup_table_latency[layer_id])
                                               for layer_id in range(branch3.cnt_layers)])

        self.last_stages = nn.Sequential(
            ConvBNRelu(input_depth=512, output_depth=64, kernel=3, stride=1,
                       pad=1, no_bias=1, use_relu="relu", bn_type="bn"),
            nn.Conv2d(64,1,1)
        )

    def custom(self, module):
        def custom_forward(*inputs):
            output = module(inputs[0][0], inputs[0][1], inputs[0][2])
            return output
        return custom_forward
    
    def forward(self, x, temperature, latency_to_accumulate):
        y = self.first(x)
        # for mixed_op in self.branch1_to_search:
        #     y1, latency_to_accumulate = checkpoint.checkpoint(self.custom(mixed_op),[y1, temperature, latency_to_accumulate])
        # for mixed_op in self.branch2_to_search:
        #     y2, latency_to_accumulate = mixed_op(y2, temperature, latency_to_accumulate)
        #     y2, latency_to_accumulate = checkpoint.checkpoint(self.custom(mixed_op),[y2, temperature, latency_to_accumulate])
        for mixed_op in self.branch3_to_search:
            y, latency_to_accumulate = mixed_op(y, temperature, latency_to_accumulate)

        # y1 = torch.nn.Upsample(scale_factor=0.125, mode="bilinear")(y1)
        y = torch.nn.Upsample(scale_factor=2, mode="bilinear")(y)
        # y = torch.cat([y2,y3], dim=1)
        y = self.last_stages(y)
        return torch.abs(y), latency_to_accumulate

class SupernetLoss(nn.Module):
    def __init__(self, sigma, crop_size, downsample_ratio, background_ratio, use_background):
        super(SupernetLoss, self).__init__()
        self.post_prob = Post_Prob(sigma, crop_size, downsample_ratio, background_ratio, use_background)
        self.criterion = Bay_Loss(use_background)
        self.beta = CONFIG_SUPERNET["loss"]['beta']
        self.alpha = CONFIG_SUPERNET["loss"]['alpha']

    # point, target, st_sizes: dataset outputs
    # pred: model outputs
    def forward(self, pred, points, target, st_sizes, cood, latency):
        points = [p.cuda() for p in points]
        target = [t.cuda() for t in target]
        cood = [c.cuda() for c in cood]

        prob_list = self.post_prob(cood[0], points, st_sizes)
        ce = self.criterion(prob_list, target, pred)*0.1
        lat = torch.log(latency ** self.beta)
        loss = self.alpha * ce #+ lat
        return loss
