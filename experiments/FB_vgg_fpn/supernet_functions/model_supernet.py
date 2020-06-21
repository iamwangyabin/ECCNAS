import torch
from torch import nn
from torch.nn import functional as F
from fbnet_building_blocks.fbnet_builder import ConvBNRelu, Flatten
from supernet_functions.config_for_supernet import CONFIG_SUPERNET
from supernet_functions.loss import Post_Prob, Bay_Loss
import torch.utils.checkpoint as checkpoint

class MergeOperation(nn.Module):
    def __init__(self, inChannels=[128,256,256,512], inFms=[1, 0.5, 2, 1], outChannel,proposed_operations):
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
        output  = sum(m * op(x) for m, op in zip(soft_mask_variables, self.ops))
        return output

class Supernet(nn.Module):
    def __init__(self):
        super(FPN, self).__init__()
        self.first = ConvBNRelu(input_depth=3, output_depth=16, kernel=3, stride=1,
                                pad=1, no_bias=1, use_relu="relu", bn_type="bn")
        # self.back_bone = resnet50(False)
        self.backbone_to_search = nn.ModuleList([MixedOperation(
                                                   branch3.layers_parameters[layer_id],
                                                   branch3.lookup_table_operations,
                                                   branch3.lookup_table_latency[layer_id])
                                               for layer_id in range(branch3.cnt_layers)])
        
        
        
        
        self.back_bone = VGGNet(True)
        # Top layer
        self.toplayer = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)  # Reduce channels
        # Smooth layers
        self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        # Lateral layers
        self.latlayer1 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d( 256, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d( 128, 256, kernel_size=1, stride=1, padding=0)
        # Semantic branch
        self.semantic_branch = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 1, kernel_size=1, stride=1, padding=0)
        # num_groups, num_channels
        self.gn1 = nn.GroupNorm(128, 128) 
        self.gn2 = nn.GroupNorm(256, 256)

        self.reg_layer = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, 1)
        )
    def _upsample(self, x, h, w):
        return F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)
    def _upsample_add(self, x, y):
        _,_,H,W = y.size()
        return F.interpolate(x, size=(H,W), mode='bilinear', align_corners=True) + y

    def forward(self, x, t):
        # Bottom-up using backbone
        low_level_features = self.back_bone(x)
        c1 = low_level_features['x1']
        c2 = low_level_features['x2']
        c3 = low_level_features['x3']
        c4 = low_level_features['x4']
        c5 = low_level_features['x5']
        # Top-down
        p5 = self.toplayer(c5)
        p4 = self._upsample_add(p5, self.latlayer1(c4))
        p3 = self._upsample_add(p4, self.latlayer2(c3))
        p2 = self._upsample_add(p3, self.latlayer3(c2))
        # Smooth
        p4 = self.smooth1(p4)
        p3 = F.leaky_relu(self.smooth2(p3))
        p2 = self.smooth3(p2)

        # Semantic
        # _, _, h, w = p3.size()
        # # 256->256
        # s5 = self._upsample(F.relu(self.gn2(self.conv2(p5))), h, w)
        # # 256->256
        # s5 = self._upsample(F.relu(self.gn2(self.conv2(s5))), h, w)
        # # 256->128
        # s5 = self._upsample(F.relu(self.gn1(self.semantic_branch(s5))), h, w)
        #
        # # 256->256
        # s4 = self._upsample(self.gn2(self.conv2(p4)), h, w)
        # # 256->128
        # s4 = self._upsample(self.gn1(self.semantic_branch(s4)), h, w)
        #
        # # 256->128
        # s3 = self._upsample(self.gn1(self.semantic_branch(p3)), h, w)
        s2 = self.semantic_branch(p3)
        return torch.abs(self.conv3(s2))
    

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
