__all__ = ['ConvNorm', 'BasicResidual1x', 'BasicResidual_downup_1x', 'BasicResidual2x', 'BasicResidual_downup_2x', 'FactorizedReduce', 'OPS', 'OPS_name', 'OPS_Class']

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from thop import profile
import sys
import os.path as osp
from easydict import EasyDict as edict

C = edict()

try:
    from tools.utils.darts_utils import compute_latency_ms_tensorrt as compute_latency
    print("use TensorRT for latency test")
except:
    from tools.utils.darts_utils import compute_latency_ms_pytorch as compute_latency
    print("use PyTorch for latency test")

latency_lookup_table = {}
table_file_name = "latency_lookup_table.npy"
if osp.isfile(table_file_name):
    latency_lookup_table = np.load(table_file_name).item()

BatchNorm2d = nn.BatchNorm2d

def make_divisible(v, divisor=8, min_value=1):
    """
    forked from slim:
    https://github.com/tensorflow/models/blob/\
    0344c5503ee55e24f0de7f37336a6e08f10976fd/\
    research/slim/nets/mobilenet/mobilenet.py#L62-L69
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class USConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, depthwise=False, bias=True, width_mult_list=[1.]):
        super(USConv2d, self).__init__(
            in_channels, out_channels,
            kernel_size, stride=stride, padding=padding, dilation=dilation,
            groups=groups, bias=bias)
        self.depthwise = depthwise
        self.in_channels_max = in_channels
        self.out_channels_max = out_channels
        self.width_mult_list = width_mult_list
        self.ratio = (1., 1.)
    
    def set_ratio(self, ratio):
        self.ratio = ratio

    def forward(self, input):
        assert self.ratio[0] in self.width_mult_list, str(self.ratio[0]) + " in? " + str(self.width_mult_list)
        self.in_channels = make_divisible(self.in_channels_max * self.ratio[0])
        assert self.ratio[1] in self.width_mult_list, str(self.ratio[1]) + " in? " + str(self.width_mult_list)
        self.out_channels = make_divisible(self.out_channels_max * self.ratio[1])
        self.groups = self.in_channels if self.depthwise else 1
        weight = self.weight[:self.out_channels, :self.in_channels, :, :]
        if self.bias is not None:
            bias = self.bias[:self.out_channels]
        else:
            bias = self.bias
        y = nn.functional.conv2d(input, weight, bias, self.stride, self.padding, self.dilation, self.groups)
        return y


class USBatchNorm2d(nn.BatchNorm2d):
    def __init__(self, num_features, width_mult_list=[1.]):
        super(USBatchNorm2d, self).__init__(
            num_features, affine=True, track_running_stats=False)
        self.num_features_max = num_features
        self.width_mult_list = width_mult_list
        # for tracking performance during training
        self.bn = nn.ModuleList(
            [ nn.BatchNorm2d(i, affine=True) for i in [ make_divisible(self.num_features_max * width_mult) for width_mult in width_mult_list ] ]
        )
        self.ratio = 1.
    
    def set_ratio(self, ratio):
        self.ratio = ratio

    def forward(self, input):
        assert self.ratio in self.width_mult_list
        idx = self.width_mult_list.index(self.ratio)
        y = self.bn[idx](input)
        return y

class ConvNorm(nn.Module):
    '''
    conv => norm => activation
    use native nn.Conv2d, not slimmable
    '''
    def __init__(self, C_in, C_out, kernel_size=3, stride=1, padding=None, dilation=1, groups=1, bias=False, slimmable=True, width_mult_list=[1.]):
        super(ConvNorm, self).__init__()
        self.C_in = C_in
        self.C_out = C_out
        self.kernel_size = kernel_size
        assert stride in [1, 2]
        self.stride = stride
        if padding is None:
            # assume h_out = h_in / s
            self.padding = int(np.ceil((dilation * (kernel_size - 1) + 1 - stride) / 2.))
        else:
            self.padding = padding
        self.dilation = dilation
        assert type(groups) == int
        if kernel_size == 1:
            self.groups = 1
        else:
            self.groups = groups
        self.bias = bias
        self.slimmable = slimmable
        self.width_mult_list = width_mult_list
        self.ratio = (1., 1.)

        if slimmable:
            self.conv = nn.Sequential(
                USConv2d(C_in, C_out, kernel_size, stride, padding=self.padding, dilation=dilation, groups=self.groups, bias=bias, width_mult_list=width_mult_list),
                USBatchNorm2d(C_out, width_mult_list),
                nn.ReLU(inplace=True),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(C_in, C_out, kernel_size, stride, padding=self.padding, dilation=dilation, groups=self.groups, bias=bias),
                # nn.BatchNorm2d(C_out),
                BatchNorm2d(C_out),
                nn.ReLU(inplace=True),
            )
    
    def set_ratio(self, ratio):
        assert self.slimmable
        assert len(ratio) == 2
        self.ratio = ratio
        self.conv[0].set_ratio(ratio)
        self.conv[1].set_ratio(ratio[1])

    @staticmethod
    def _flops(h, w, C_in, C_out, kernel_size=3, stride=1, padding=None, dilation=1, groups=1, bias=False):
        layer = ConvNorm(C_in, C_out, kernel_size, stride, padding, dilation, groups, bias, slimmable=False)
        flops, params = profile(layer, inputs=(torch.randn(1, C_in, h, w),), verbose=False)
        return flops
    
    @staticmethod
    def _latency(h, w, C_in, C_out, kernel_size=3, stride=1, padding=None, dilation=1, groups=1, bias=False):
        layer = ConvNorm(C_in, C_out, kernel_size, stride, padding, dilation, groups, bias, slimmable=False)
        latency = compute_latency(layer, (1, C_in, h, w))
        return latency

    def forward_latency(self, size):
        c_in, h_in, w_in = size
        if self.slimmable:
            assert c_in == int(self.C_in * self.ratio[0]), "c_in %d, self.C_in * self.ratio[0] %d"%(c_in, self.C_in * self.ratio[0])
            c_out = int(self.C_out * self.ratio[1])
        else:
            assert c_in == self.C_in, "c_in %d, self.C_in %d"%(c_in, self.C_in)
            c_out = self.C_out
        if self.stride == 1:
            h_out = h_in; w_out = w_in
        else:
            h_out = h_in // 2; w_out = w_in // 2
        name = "ConvNorm_H%d_W%d_Cin%d_Cout%d_kernel%d_stride%d"%(h_in, w_in, c_in, c_out, self.kernel_size, self.stride)
        if name in latency_lookup_table:
            latency = latency_lookup_table[name]
        else:
            print("not found in latency_lookup_table:", name)
            latency = ConvNorm._latency(h_in, w_in, c_in, c_out, self.kernel_size, self.stride, self.padding, self.dilation, self.groups, self.bias)
            latency_lookup_table[name] = latency
            np.save(table_file_name, latency_lookup_table)
        return latency, (c_out, h_out, w_out)

    def forward(self, x):
        assert x.size()[1] == self.C_in, "{} {}".format(x.size()[1], self.C_in)
        x = self.conv(x)
        return x


class BasicResidual1x(nn.Module):
    def __init__(self, C_in, C_out, kernel_size=3, stride=1, dilation=1, groups=1, slimmable=True, width_mult_list=[1.]):
        super(BasicResidual1x, self).__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.C_in = C_in
        self.C_out = C_out
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.slimmable = slimmable
        self.width_mult_list = width_mult_list
        assert stride in [1, 2]
        if self.stride == 2: self.dilation = 1
        self.ratio = (1., 1.)

        self.relu = nn.ReLU(inplace=True)
        if slimmable:
            self.conv1 = USConv2d(C_in, C_out, 3, stride, padding=dilation, dilation=dilation, groups=groups, bias=False, width_mult_list=width_mult_list)
            self.bn1 = USBatchNorm2d(C_out, width_mult_list)
        else:
            self.conv1 = nn.Conv2d(C_in, C_out, 3, stride, padding=dilation, dilation=dilation, groups=groups, bias=False)
            # self.bn1 = nn.BatchNorm2d(C_out)
            self.bn1 = BatchNorm2d(C_out)
    
    def set_ratio(self, ratio):
        assert len(ratio) == 2
        self.ratio = ratio
        self.conv1.set_ratio(ratio)
        self.bn1.set_ratio(ratio[1])

    @staticmethod
    def _flops(h, w, C_in, C_out, kernel_size=3, stride=1, dilation=1, groups=1):
        layer = BasicResidual1x(C_in, C_out, kernel_size, stride, dilation, groups, slimmable=False)
        flops, params = profile(layer, inputs=(torch.randn(1, C_in, h, w),), verbose=False)
        return flops
    
    @staticmethod
    def _latency(h, w, C_in, C_out, kernel_size=3, stride=1, dilation=1, groups=1):
        layer = BasicResidual1x(C_in, C_out, kernel_size, stride, dilation, groups, slimmable=False)
        latency = compute_latency(layer, (1, C_in, h, w))
        return latency

    def forward_latency(self, size):
        c_in, h_in, w_in = size
        if self.slimmable:
            assert c_in == int(self.C_in * self.ratio[0]), "c_in %d, int(self.C_in * self.ratio[0]) %d"%(c_in, int(self.C_in * self.ratio[0]))
            c_out = int(self.C_out * self.ratio[1])
        else:
            assert c_in == self.C_in, "c_in %d, self.C_in %d"%(c_in, self.C_in)
            c_out = self.C_out
        if self.stride == 1:
            h_out = h_in; w_out = w_in
        else:
            h_out = h_in // 2; w_out = w_in // 2
        name = "BasicResidual1x_H%d_W%d_Cin%d_Cout%d_stride%d_dilation%d"%(h_in, w_in, c_in, c_out, self.stride, self.dilation)
        if name in latency_lookup_table:
            latency = latency_lookup_table[name]
        else:
            print("not found in latency_lookup_table:", name)
            latency = BasicResidual1x._latency(h_in, w_in, c_in, c_out, self.kernel_size, self.stride, self.dilation, self.groups)
            latency_lookup_table[name] = latency
            np.save(table_file_name, latency_lookup_table)
        return latency, (c_out, h_out, w_out)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        return out


class BasicResidual_downup_1x(nn.Module):
    def __init__(self, C_in, C_out, kernel_size=3, stride=1, dilation=1, groups=1, slimmable=True, width_mult_list=[1.]):
        super(BasicResidual_downup_1x, self).__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.C_in = C_in
        self.C_out = C_out
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.slimmable = slimmable
        self.width_mult_list = width_mult_list
        assert stride in [1, 2]
        if self.stride == 2: self.dilation = 1
        self.ratio = (1., 1.)

        self.relu = nn.ReLU(inplace=True)
        if slimmable:
            self.conv1 = USConv2d(C_in, C_out, 3, 1, padding=dilation, dilation=dilation, groups=groups, bias=False, width_mult_list=width_mult_list)
            self.bn1 = USBatchNorm2d(C_out, width_mult_list)
        else:
            self.conv1 = nn.Conv2d(C_in, C_out, 3, 1, padding=dilation, dilation=dilation, groups=groups, bias=False)
            # self.bn1 = nn.BatchNorm2d(C_out)
            self.bn1 = BatchNorm2d(C_out)
    
    def set_ratio(self, ratio):
        assert len(ratio) == 2
        self.ratio = ratio
        self.conv1.set_ratio(ratio)
        self.bn1.set_ratio(ratio[1])

    @staticmethod
    def _flops(h, w, C_in, C_out, kernel_size=3, stride=1, dilation=1, groups=1):
        assert stride in [1, 2]
        layer = BasicResidual_downup_1x(C_in, C_out, kernel_size, stride, dilation, groups, slimmable=False)
        flops, params = profile(layer, inputs=(torch.randn(1, C_in, h, w),), verbose=False)
        return flops
    
    @staticmethod
    def _latency(h, w, C_in, C_out, kernel_size=3, stride=1, dilation=1, groups=1):
        assert stride in [1, 2]
        layer = BasicResidual_downup_1x(C_in, C_out, kernel_size, stride, dilation, groups, slimmable=False)
        latency = compute_latency(layer, (1, C_in, h, w))
        return latency

    def forward_latency(self, size):
        c_in, h_in, w_in = size
        if self.slimmable:
            assert c_in == int(self.C_in * self.ratio[0]), "c_in %d, int(self.C_in * self.ratio[0]) %d"%(c_in, int(self.C_in * self.ratio[0]))
            c_out = int(self.C_out * self.ratio[1])
        else:
            assert c_in == self.C_in, "c_in %d, self.C_in %d"%(c_in, self.C_in)
            c_out = self.C_out
        if self.stride == 1:
            h_out = h_in; w_out = w_in
        else:
            h_out = h_in // 2; w_out = w_in // 2
        name = "BasicResidual_downup_1x_H%d_W%d_Cin%d_Cout%d_stride%d_dilation%d"%(h_in, w_in, c_in, c_out, self.stride, self.dilation)
        if name in latency_lookup_table:
            latency = latency_lookup_table[name]
        else:
            print("not found in latency_lookup_table:", name)
            latency = BasicResidual_downup_1x._latency(h_in, w_in, c_in, c_out, self.kernel_size, self.stride, self.dilation, self.groups)
            latency_lookup_table[name] = latency
            np.save(table_file_name, latency_lookup_table)
        return latency, (c_out, h_out, w_out)

    def forward(self, x):
        out = F.interpolate(x, size=(int(x.size(2))//2, int(x.size(3))//2), mode='bilinear', align_corners=True)
        out = self.conv1(out)
        out = self.bn1(out)
        if self.stride == 1:
            out = F.interpolate(out, size=(int(x.size(2)), int(x.size(3))), mode='bilinear', align_corners=True)
        out = self.relu(out)
        return out


class BasicResidual2x(nn.Module):
    def __init__(self, C_in, C_out, kernel_size=3, stride=1, dilation=1, groups=1, slimmable=True, width_mult_list=[1.]):
        super(BasicResidual2x, self).__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.C_in = C_in
        self.C_out = C_out
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.slimmable = slimmable
        self.width_mult_list = width_mult_list
        assert stride in [1, 2]
        if self.stride == 2: self.dilation = 1
        self.ratio = (1., 1.)

        self.relu = nn.ReLU(inplace=True)
        if self.slimmable:
            self.conv1 = USConv2d(C_in, C_out, 3, stride, padding=dilation, dilation=dilation, groups=groups, bias=False, width_mult_list=width_mult_list)
            self.bn1 = USBatchNorm2d(C_out, width_mult_list)
            self.conv2 = USConv2d(C_out, C_out, 3, 1, padding=dilation, dilation=dilation, groups=groups, bias=False, width_mult_list=width_mult_list)
            self.bn2 = USBatchNorm2d(C_out, width_mult_list)
        else:
            self.conv1 = nn.Conv2d(C_in, C_out, 3, stride, padding=dilation, dilation=dilation, groups=groups, bias=False)
            # self.bn1 = nn.BatchNorm2d(C_out)
            self.bn1 = BatchNorm2d(C_out)
            self.conv2 = nn.Conv2d(C_out, C_out, 3, 1, padding=dilation, dilation=dilation, groups=groups, bias=False)
            # self.bn2 = nn.BatchNorm2d(C_out)
            self.bn2 = BatchNorm2d(C_out)
    
    def set_ratio(self, ratio):
        assert len(ratio) == 2
        self.ratio = ratio
        self.conv1.set_ratio(ratio)
        self.bn1.set_ratio(ratio[1])
        self.conv2.set_ratio((ratio[1], ratio[1]))
        self.bn2.set_ratio(ratio[1])

    @staticmethod
    def _flops(h, w, C_in, C_out, kernel_size=3, stride=1, dilation=1, groups=1):
        layer = BasicResidual2x(C_in, C_out, kernel_size, stride, dilation, groups, slimmable=False)
        flops, params = profile(layer, inputs=(torch.randn(1, C_in, h, w),), verbose=False)
        return flops
    
    @staticmethod
    def _latency(h, w, C_in, C_out, kernel_size=3, stride=1, dilation=1, groups=1):
        layer = BasicResidual2x(C_in, C_out, kernel_size, stride, dilation, groups, slimmable=False)
        latency = compute_latency(layer, (1, C_in, h, w))
        return latency

    def forward_latency(self, size):
        c_in, h_in, w_in = size
        if self.slimmable:
            assert c_in == int(self.C_in * self.ratio[0])
            c_out = int(self.C_out * self.ratio[1])
        else:
            assert c_in == self.C_in, "c_in %d, self.C_in%d"%(c_in, self.C_in)
            c_out = self.C_out
        if self.stride == 1:
            h_out = h_in; w_out = w_in
        else:
            h_out = h_in // 2; w_out = w_in // 2
        name = "BasicResidual2x_H%d_W%d_Cin%d_Cout%d_stride%d_dilation%d"%(h_in, w_in, c_in, c_out, self.stride, self.dilation)
        if name in latency_lookup_table:
            latency = latency_lookup_table[name]
        else:
            print("not found in latency_lookup_table:", name)
            latency = BasicResidual2x._latency(h_in, w_in, c_in, c_out, self.kernel_size, self.stride, self.dilation, self.groups)
            latency_lookup_table[name] = latency
            np.save(table_file_name, latency_lookup_table)
        return latency, (c_out, h_out, w_out)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        return out


class BasicResidual_downup_2x(nn.Module):
    def __init__(self, C_in, C_out, kernel_size=3, stride=1, dilation=1, groups=1, slimmable=True, width_mult_list=[1.]):
        super(BasicResidual_downup_2x, self).__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.C_in = C_in
        self.C_out = C_out
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.slimmable = slimmable
        self.width_mult_list = width_mult_list
        assert stride in [1, 2]
        if self.stride == 2: self.dilation = 1
        self.ratio = (1., 1.)

        self.relu = nn.ReLU(inplace=True)
        if self.slimmable:
            self.conv1 = USConv2d(C_in, C_out, 3, 1, padding=dilation, dilation=dilation, groups=groups, bias=False, width_mult_list=width_mult_list)
            self.bn1 = USBatchNorm2d(C_out, width_mult_list)
            self.conv2 = USConv2d(C_out, C_out, 3, 1, padding=dilation, dilation=dilation, groups=groups, bias=False, width_mult_list=width_mult_list)
            self.bn2 = USBatchNorm2d(C_out, width_mult_list)
        else:
            self.conv1 = nn.Conv2d(C_in, C_out, 3, 1, padding=dilation, dilation=dilation, groups=groups, bias=False)
            # self.bn1 = nn.BatchNorm2d(C_out)
            self.bn1 = BatchNorm2d(C_out)
            self.conv2 = nn.Conv2d(C_out, C_out, 3, 1, padding=dilation, dilation=dilation, groups=groups, bias=False)
            # self.bn2 = nn.BatchNorm2d(C_out)
            self.bn2 = BatchNorm2d(C_out)
    
    def set_ratio(self, ratio):
        assert len(ratio) == 2
        self.ratio = ratio
        self.conv1.set_ratio(ratio)
        self.bn1.set_ratio(ratio[1])
        self.conv2.set_ratio((ratio[1], ratio[1]))
        self.bn2.set_ratio(ratio[1])

    @staticmethod
    def _flops(h, w, C_in, C_out, kernel_size=3, stride=1, dilation=1, groups=1):
        assert stride in [1, 2]
        layer = BasicResidual_downup_2x(C_in, C_out, kernel_size, stride, dilation, groups, slimmable=False)
        flops, params = profile(layer, inputs=(torch.randn(1, C_in, h, w),), verbose=False)
        return flops
    
    @staticmethod
    def _latency(h, w, C_in, C_out, kernel_size=3, stride=1, dilation=1, groups=1):
        assert stride in [1, 2]
        layer = BasicResidual_downup_2x(C_in, C_out, kernel_size, stride, dilation, groups, slimmable=False)
        latency = compute_latency(layer, (1, C_in, h, w))
        return latency

    def forward_latency(self, size):
        c_in, h_in, w_in = size
        if self.slimmable:
            assert c_in == int(self.C_in * self.ratio[0])
            c_out = int(self.C_out * self.ratio[1])
        else:
            assert c_in == self.C_in, "c_in %d, self.C_in%d"%(c_in, self.C_in)
            c_out = self.C_out
        if self.stride == 1:
            h_out = h_in; w_out = w_in
        else:
            h_out = h_in // 2; w_out = w_in // 2
        name = "BasicResidual2x_H%d_W%d_Cin%d_Cout%d_stride%d_dilation%d"%(h_in, w_in, c_in, c_out, self.stride, self.dilation)
        if name in latency_lookup_table:
            latency = latency_lookup_table[name]
        else:
            print("not found in latency_lookup_table:", name)
            latency = BasicResidual2x._latency(h_in, w_in, c_in, c_out, self.kernel_size, self.stride, self.dilation, self.groups)
            latency_lookup_table[name] = latency
            np.save(table_file_name, latency_lookup_table)
        return latency, (c_out, h_out, w_out)

    def forward(self, x):
        out = F.interpolate(x, size=(int(x.size(2))//2, int(x.size(3))//2), mode='bilinear', align_corners=True)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.stride == 1:
            out = F.interpolate(out, size=(int(x.size(2)), int(x.size(3))), mode='bilinear', align_corners=True)
        out = self.relu(out)
        return out


class FactorizedReduce(nn.Module):
    def __init__(self, C_in, C_out, stride=1, slimmable=True, width_mult_list=[1.]):
        super(FactorizedReduce, self).__init__()
        assert stride in [1, 2]
        assert C_out % 2 == 0
        self.C_in = C_in
        self.C_out = C_out
        self.stride = stride
        self.slimmable = slimmable
        self.width_mult_list = width_mult_list
        self.ratio = (1., 1.)
        if stride == 1 and slimmable:
            self.conv1 = USConv2d(C_in, C_out, 1, stride=1, padding=0, bias=False, width_mult_list=width_mult_list)
            self.bn = USBatchNorm2d(C_out, width_mult_list)
            self.relu = nn.ReLU(inplace=True)
        elif stride == 2:
            self.relu = nn.ReLU(inplace=True)
            if slimmable:
                self.conv1 = USConv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False, width_mult_list=width_mult_list)
                self.conv2 = USConv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False, width_mult_list=width_mult_list)
                self.bn = USBatchNorm2d(C_out, width_mult_list)
            else:
                self.conv1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
                self.conv2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
                self.bn = BatchNorm2d(C_out)

    def set_ratio(self, ratio):
        assert len(ratio) == 2
        if self.stride == 1:
            self.ratio = ratio
            self.conv1.set_ratio(ratio)
            self.bn.set_ratio(ratio[1])
        elif self.stride == 2:
            self.ratio = ratio
            self.conv1.set_ratio(ratio)
            self.conv2.set_ratio(ratio)
            self.bn.set_ratio(ratio[1])

    @staticmethod
    def _flops(h, w, C_in, C_out, stride=1):
        layer = FactorizedReduce(C_in, C_out, stride, slimmable=False)
        flops, params = profile(layer, inputs=(torch.randn(1, C_in, h, w),), verbose=False)
        return flops

    @staticmethod
    def _latency(h, w, C_in, C_out, stride=1):
        layer = FactorizedReduce(C_in, C_out, stride, slimmable=False)
        latency = compute_latency(layer, (1, C_in, h, w))
        return latency

    def forward_latency(self, size):
        c_in, h_in, w_in = size
        if self.slimmable:
            assert c_in == int(self.C_in * self.ratio[0])
            c_out = int(self.C_out * self.ratio[1])
        else:
            assert c_in == self.C_in
            c_out = self.C_out
        if self.stride == 1:
            h_out = h_in; w_out = w_in
        else:
            h_out = h_in // 2; w_out = w_in // 2
        name = "FactorizedReduce_H%d_W%d_Cin%d_Cout%d_stride%d"%(h_in, w_in, c_in, c_out, self.stride)
        if name in latency_lookup_table:
            latency = latency_lookup_table[name]
        else:
            print("not found in latency_lookup_table:", name)
            latency = FactorizedReduce._latency(h_in, w_in, c_in, c_out, self.stride)
            latency_lookup_table[name] = latency
            np.save(table_file_name, latency_lookup_table)
        return latency, (c_out, h_out, w_out)

    def forward(self, x):
        if self.stride == 2:
            out = torch.cat([self.conv1(x), self.conv2(x[:,:,1:,1:])], dim=1)
            out = self.bn(out)
            out = self.relu(out)
            return out
        else:
            if self.slimmable:
                out = self.conv1(x)
                out = self.bn(out)
                out = self.relu(out)
                return out
            else:
                return x



from collections import OrderedDict
OPS = {
    'skip' : lambda C_in, C_out, stride, slimmable, width_mult_list: FactorizedReduce(C_in, C_out, stride, slimmable, width_mult_list),
    'conv' : lambda C_in, C_out, stride, slimmable, width_mult_list: BasicResidual1x(C_in, C_out, kernel_size=3, stride=stride, dilation=1, slimmable=slimmable, width_mult_list=width_mult_list),
    'conv_downup' : lambda C_in, C_out, stride, slimmable, width_mult_list: BasicResidual_downup_1x(C_in, C_out, kernel_size=3, stride=stride, dilation=1, slimmable=slimmable, width_mult_list=width_mult_list),
    'conv_2x' : lambda C_in, C_out, stride, slimmable, width_mult_list: BasicResidual2x(C_in, C_out, kernel_size=3, stride=stride, dilation=1, slimmable=slimmable, width_mult_list=width_mult_list),
    'conv_2x_downup' : lambda C_in, C_out, stride, slimmable, width_mult_list: BasicResidual_downup_2x(C_in, C_out, kernel_size=3, stride=stride, dilation=1, slimmable=slimmable, width_mult_list=width_mult_list),
}
OPS_name = ["FactorizedReduce", "BasicResidual1x", "BasicResidual_downup_1x", "BasicResidual2x", "BasicResidual_downup_2x"]
OPS_Class = OrderedDict()
OPS_Class['skip'] = FactorizedReduce
OPS_Class['conv'] = BasicResidual1x
OPS_Class['conv_downup'] = BasicResidual_downup_1x
OPS_Class['conv_2x'] = BasicResidual2x
OPS_Class['conv_2x_downup'] = BasicResidual_downup_2x
