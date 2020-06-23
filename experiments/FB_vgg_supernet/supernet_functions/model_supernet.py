import torch
from torch import nn
from fbnet_building_blocks.fbnet_builder import ConvBNRelu


class MergeOperation(nn.Module):
    def __init__(self, inChannels=[128,256,256,512], inFms=[1, 0.5, 2, 1], outChannel=256 ,proposed_operations):
        super(MergeOperation, self).__init__()
        ops_names = [op_name for op_name in proposed_operations]
        # layers_parameters are : C_in, C_out, expansion, stride
        # only use Identity
        self.mergeOps = []
        for c, s in zip(inChannels, inFms):
            self.mergeOps.append(proposed_operations['skip']([c, outChannel, s]))
        self.ops = nn.ModuleList([proposed_operations[op_name]([outChannel, outChannel, -999, 1]) for op_name in ops_names])
        self.thetas = nn.Parameter(torch.Tensor([1.0 / len(ops_names) for i in range(len(ops_names))]))
        self.betas = nn.Parameter(torch.Tensor([1.0 / len(inChannels) for i in range(len(inChannels))]))

    def forward(self, xlist, temperature):
        soft_mask_merge = nn.functional.gumbel_softmax(self.betas, temperature)
        soft_mask_stack = nn.functional.gumbel_softmax(self.thetas, temperature)
        output = 0
        for op, i, m in zip(self.mergeOps, xlist, soft_mask_merge):
            output += op(i) * m
        output  = sum(m * op(output) for m, op in zip(soft_mask_variables, self.ops))
        return output
 

class MixedOperation(nn.Module):
    def __init__(self, layer_parameters, proposed_operations):
        super(MixedOperation, self).__init__()
        ops_names = [op_name for op_name in proposed_operations]
        self.ops = nn.ModuleList([proposed_operations[op_name](*layer_parameters)
                                  for op_name in ops_names])
        self.thetas = nn.Parameter(torch.Tensor([1.0 / len(ops_names) for i in range(len(ops_names))]))

    def forward(self, x, temperature):
        soft_mask_variables = nn.functional.gumbel_softmax(self.thetas, temperature)
        output = sum(m * op(x) for m, op in zip(soft_mask_variables, self.ops))
        return output


class Supernet(nn.Module):
    def __init__(self, lookuptable):
        super(Supernet, self).__init__()
        self.first = ConvBNRelu(input_depth=3, output_depth=16, kernel=3, stride=1,
                                pad=1, no_bias=1, use_relu="relu", bn_type="bn")

        self.backbone_to_search = nn.ModuleList([MixedOperation(
            lookuptable.layers_parameters[layer_id],
            lookuptable.lookup_table_operations)
            for layer_id in range(lookuptable.cnt_layers)])

        # searchable fpn
        self.reg_layer = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.Dropout(0.15),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.Dropout(0.15),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, 1)
        )

    def forward(self, x, temperature):
        y = self.first(x)
        i = 0
        # get 4 8 12  and final y layer
        # 4: fm size(128,256,256)
        # 8: fm size(256,128,128)
        # 12: fm size(512,64,64)
        # y : fm size(512,32,32)
        scale_feature = []
        for mixed_op in self.backbone_to_search:
            y = mixed_op(y, temperature)
            if i%4==0 and i!=0 and i!=16:
                scale_feature.append(y)
            i+=1
            
            
            
            
            
        y = self.reg_layer(y)
        
        
        
        
        
        
        
        
        return torch.abs(y)



class Supernet_pretrain(nn.Module):
    def __init__(self, lookuptable):
        super(Supernet_pretrain, self).__init__()
        self.first = ConvBNRelu(input_depth=3, output_depth=16, kernel=3, stride=1,
                                pad=1, no_bias=1, use_relu="relu", bn_type="bn")

        self.backbone_to_search = nn.ModuleList([MixedOperation(
            lookuptable.layers_parameters[layer_id],
            lookuptable.lookup_table_operations)
            for layer_id in range(lookuptable.cnt_layers)])


        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 1000)
        )

    def forward(self, x, temperature):
        y = self.first(x)
        for mixed_op in self.backbone_to_search:
            y = mixed_op(y, temperature)
        y = self.maxpool(y)
        y = y.view(y.size(0), -1)
        y = self.classifier(y)
        return y
