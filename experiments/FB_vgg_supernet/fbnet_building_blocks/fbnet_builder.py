# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from __future__ import absolute_import, division, print_function, unicode_literals

import copy
import logging
import math
from collections import OrderedDict

import torch
import torch.nn as nn
from .layers import (BatchNorm2d, Conv2d, FrozenBatchNorm2d, interpolate, _NewEmptyTensorOp)

logger = logging.getLogger(__name__)


def _py2_round(x):
    return math.floor(x + 0.5) if x >= 0.0 else math.ceil(x - 0.5)


def _get_divisible_by(num, divisible_by, min_val):
    ret = int(num)
    if divisible_by > 0 and num % divisible_by != 0:
        ret = int((_py2_round(num / divisible_by) or min_val) * divisible_by)
    return ret


PRIMITIVES = {
    "skip": lambda C_in, C_out, expansion, stride, **kwargs: Identity(C_in, C_out, stride),
    'k3_conv': lambda C_in, C_out, expansion, stride: CommonConv(C_in, C_out, kernel_size=3, stride=stride),
    'k5_conv': lambda C_in, C_out, expansion, stride: CommonConv(C_in, C_out, kernel_size=5, stride=stride),
    'sep_conv_3x3': lambda C_in, C_out, expansion, stride: SepConv(C_in, C_out, 3, stride, 1),
    'sep_conv_5x5': lambda C_in, C_out, expansion, stride: SepConv(C_in, C_out, 5, stride, 2),
    'dil_conv_3x3': lambda C_in, C_out, expansion, stride: DilConv(C_in, C_out, 3, stride, 2, 2),
    'dil_conv_5x5': lambda C_in, C_out, expansion, stride: DilConv(C_in, C_out, 5, stride, 4, 2),

    'avg_pool_3x3': lambda C, stride, affine: nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False),
    'max_pool_3x3': lambda C, stride, affine: nn.MaxPool2d(3, stride=stride, padding=1),

    'none': lambda C, stride, affine: Zero(stride),
    "ir_k3": lambda C_in, C_out, expansion, stride, **kwargs: IRFBlock(C_in, C_out, expansion, stride, **kwargs),
    "ir_k5": lambda C_in, C_out, expansion, stride, **kwargs: IRFBlock(C_in, C_out, expansion, stride, kernel=5, **kwargs),
    "ir_k7": lambda C_in, C_out, expansion, stride, **kwargs: IRFBlock(C_in, C_out, expansion, stride, kernel=7, **kwargs),
    "ir_k1": lambda C_in, C_out, expansion, stride, **kwargs: IRFBlock(C_in, C_out, expansion, stride, kernel=1, **kwargs),
    "shuffle": lambda C_in, C_out, expansion, stride, **kwargs: IRFBlock(C_in, C_out, expansion, stride, shuffle_type="mid", pw_group=4, **kwargs),
    "basic_block": lambda C_in, C_out, expansion, stride, **kwargs: CascadeConv3x3(C_in, C_out, stride),
    "shift_5x5": lambda C_in, C_out, expansion, stride, **kwargs: ShiftBlock5x5(C_in, C_out, expansion, stride),
    # layer search 2
    "ir_k3_e1": lambda C_in, C_out, expansion, stride, **kwargs: IRFBlock(
        C_in, C_out, 1, stride, kernel=3, **kwargs
    ),
    "ir_k3_e3": lambda C_in, C_out, expansion, stride, **kwargs: IRFBlock(
        C_in, C_out, 3, stride, kernel=3, **kwargs
    ),
    "ir_k3_e6": lambda C_in, C_out, expansion, stride, **kwargs: IRFBlock(
        C_in, C_out, 6, stride, kernel=3, **kwargs
    ),
    "ir_k3_s4": lambda C_in, C_out, expansion, stride, **kwargs: IRFBlock(
        C_in, C_out, 4, stride, kernel=3, shuffle_type="mid", pw_group=4, **kwargs
    ),
    "ir_k5_e1": lambda C_in, C_out, expansion, stride, **kwargs: IRFBlock(
        C_in, C_out, 1, stride, kernel=5, **kwargs
    ),
    "ir_k5_e3": lambda C_in, C_out, expansion, stride, **kwargs: IRFBlock(
        C_in, C_out, 3, stride, kernel=5, **kwargs
    ),
    "ir_k5_e6": lambda C_in, C_out, expansion, stride, **kwargs: IRFBlock(
        C_in, C_out, 6, stride, kernel=5, **kwargs
    ),
    "ir_k5_s4": lambda C_in, C_out, expansion, stride, **kwargs: IRFBlock(
        C_in, C_out, 4, stride, kernel=5, shuffle_type="mid", pw_group=4, **kwargs
    ),
    # layer search se
    "ir_k3_e1_se": lambda C_in, C_out, expansion, stride, **kwargs: IRFBlock(
        C_in, C_out, 1, stride, kernel=3, se=True, **kwargs
    ),
    "ir_k3_e3_se": lambda C_in, C_out, expansion, stride, **kwargs: IRFBlock(
        C_in, C_out, 3, stride, kernel=3, se=True, **kwargs
    ),
    "ir_k3_e6_se": lambda C_in, C_out, expansion, stride, **kwargs: IRFBlock(
        C_in, C_out, 6, stride, kernel=3, se=True, **kwargs
    ),
    "ir_k3_s4_se": lambda C_in, C_out, expansion, stride, **kwargs: IRFBlock(
        C_in,
        C_out,
        4,
        stride,
        kernel=3,
        shuffle_type="mid",
        pw_group=4,
        se=True,
        **kwargs
    ),
    "ir_k5_e1_se": lambda C_in, C_out, expansion, stride, **kwargs: IRFBlock(
        C_in, C_out, 1, stride, kernel=5, se=True, **kwargs
    ),
    "ir_k5_e3_se": lambda C_in, C_out, expansion, stride, **kwargs: IRFBlock(
        C_in, C_out, 3, stride, kernel=5, se=True, **kwargs
    ),
    "ir_k5_e6_se": lambda C_in, C_out, expansion, stride, **kwargs: IRFBlock(
        C_in, C_out, 6, stride, kernel=5, se=True, **kwargs
    ),
    "ir_k5_s4_se": lambda C_in, C_out, expansion, stride, **kwargs: IRFBlock(
        C_in,
        C_out,
        4,
        stride,
        kernel=5,
        shuffle_type="mid",
        pw_group=4,
        se=True,
        **kwargs
    ),
    # layer search 3 (in addition to layer search 2)
    "ir_k3_s2": lambda C_in, C_out, expansion, stride, **kwargs: IRFBlock(
        C_in, C_out, 1, stride, kernel=3, shuffle_type="mid", pw_group=2, **kwargs
    ),
    "ir_k5_s2": lambda C_in, C_out, expansion, stride, **kwargs: IRFBlock(
        C_in, C_out, 1, stride, kernel=5, shuffle_type="mid", pw_group=2, **kwargs
    ),
    "ir_k3_s2_se": lambda C_in, C_out, expansion, stride, **kwargs: IRFBlock(
        C_in,
        C_out,
        1,
        stride,
        kernel=3,
        shuffle_type="mid",
        pw_group=2,
        se=True,
        **kwargs
    ),
    "ir_k5_s2_se": lambda C_in, C_out, expansion, stride, **kwargs: IRFBlock(
        C_in,
        C_out,
        1,
        stride,
        kernel=5,
        shuffle_type="mid",
        pw_group=2,
        se=True,
        **kwargs
    ),
    # layer search 4 (in addition to layer search 3)
    "ir_k3_sep": lambda C_in, C_out, expansion, stride, **kwargs: IRFBlock(
        C_in, C_out, expansion, stride, kernel=3, cdw=True, **kwargs
    ),
    "ir_k33_e1": lambda C_in, C_out, expansion, stride, **kwargs: IRFBlock(
        C_in, C_out, 1, stride, kernel=3, cdw=True, **kwargs
    ),
    "ir_k33_e3": lambda C_in, C_out, expansion, stride, **kwargs: IRFBlock(
        C_in, C_out, 3, stride, kernel=3, cdw=True, **kwargs
    ),
    "ir_k33_e6": lambda C_in, C_out, expansion, stride, **kwargs: IRFBlock(
        C_in, C_out, 6, stride, kernel=3, cdw=True, **kwargs
    ),
    # layer search 5 (in addition to layer search 4)
    "ir_k7_e1": lambda C_in, C_out, expansion, stride, **kwargs: IRFBlock(
        C_in, C_out, 1, stride, kernel=7, **kwargs
    ),
    "ir_k7_e3": lambda C_in, C_out, expansion, stride, **kwargs: IRFBlock(
        C_in, C_out, 3, stride, kernel=7, **kwargs
    ),
    "ir_k7_e6": lambda C_in, C_out, expansion, stride, **kwargs: IRFBlock(
        C_in, C_out, 6, stride, kernel=7, **kwargs
    ),
    "ir_k7_sep": lambda C_in, C_out, expansion, stride, **kwargs: IRFBlock(
        C_in, C_out, expansion, stride, kernel=7, cdw=True, **kwargs
    ),
    "ir_k7_sep_e1": lambda C_in, C_out, expansion, stride, **kwargs: IRFBlock(
        C_in, C_out, 1, stride, kernel=7, cdw=True, **kwargs
    ),
    "ir_k7_sep_e3": lambda C_in, C_out, expansion, stride, **kwargs: IRFBlock(
        C_in, C_out, 3, stride, kernel=7, cdw=True, **kwargs
    ),
    "ir_k7_sep_e6": lambda C_in, C_out, expansion, stride, **kwargs: IRFBlock(
        C_in, C_out, 6, stride, kernel=7, cdw=True, **kwargs
    ),
}


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        shape = torch.prod(torch.tensor(x.shape[1:])).item()
        return x.view(-1, shape)


class Identity(nn.Module):
    def __init__(self, C_in, C_out, stride):
        super(Identity, self).__init__()
        self.output_depth = C_out  # ANNA's code here
        self.conv = (
            ConvBNRelu(
                C_in,
                C_out,
                kernel=1,
                stride=stride,
                pad=0,
                no_bias=1,
                use_relu="relu",
                bn_type="bn",
            )
            if C_in != C_out or stride != 1
            else None
        )

    def forward(self, x):
        if self.conv:
            out = self.conv(x)
        else:
            out = x
        return out


class CascadeConv3x3(nn.Sequential):
    def __init__(self, C_in, C_out, stride):
        assert stride in [1, 2]
        ops = [
            Conv2d(C_in, C_in, 3, stride, 1, bias=False),
            BatchNorm2d(C_in),
            nn.ReLU(inplace=True),
            Conv2d(C_in, C_out, 3, 1, 1, bias=False),
            BatchNorm2d(C_out),
        ]
        super(CascadeConv3x3, self).__init__(*ops)
        self.res_connect = (stride == 1) and (C_in == C_out)

    def forward(self, x):
        y = super(CascadeConv3x3, self).forward(x)
        if self.res_connect:
            y += x
        return y


class Shift(nn.Module):
    def __init__(self, C, kernel_size, stride, padding):
        super(Shift, self).__init__()
        self.C = C
        kernel = torch.zeros((C, 1, kernel_size, kernel_size), dtype=torch.float32)
        ch_idx = 0

        assert stride in [1, 2]
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size
        self.dilation = 1

        hks = kernel_size // 2
        ksq = kernel_size ** 2

        for i in range(kernel_size):
            for j in range(kernel_size):
                if i == hks and j == hks:
                    num_ch = C // ksq + C % ksq
                else:
                    num_ch = C // ksq
                kernel[ch_idx: ch_idx + num_ch, 0, i, j] = 1
                ch_idx += num_ch

        self.register_parameter("bias", None)
        self.kernel = nn.Parameter(kernel, requires_grad=False)

    def forward(self, x):
        if x.numel() > 0:
            return nn.functional.conv2d(
                x,
                self.kernel,
                self.bias,
                (self.stride, self.stride),
                (self.padding, self.padding),
                self.dilation,
                self.C,  # groups
            )

        output_shape = [
            (i + 2 * p - (di * (k - 1) + 1)) // d + 1
            for i, p, di, k, d in zip(
                x.shape[-2:],
                (self.padding, self.dilation),
                (self.dilation, self.dilation),
                (self.kernel_size, self.kernel_size),
                (self.stride, self.stride),
            )
        ]
        output_shape = [x.shape[0], self.C] + output_shape
        return _NewEmptyTensorOp.apply(x, output_shape)


class ShiftBlock5x5(nn.Sequential):
    def __init__(self, C_in, C_out, expansion, stride):
        assert stride in [1, 2]
        self.res_connect = (stride == 1) and (C_in == C_out)

        C_mid = _get_divisible_by(C_in * expansion, 8, 8)

        ops = [
            # pw
            Conv2d(C_in, C_mid, 1, 1, 0, bias=False),
            BatchNorm2d(C_mid),
            nn.ReLU(inplace=True),
            # shift
            Shift(C_mid, 5, stride, 2),
            # pw-linear
            Conv2d(C_mid, C_out, 1, 1, 0, bias=False),
            BatchNorm2d(C_out),
        ]
        super(ShiftBlock5x5, self).__init__(*ops)

    def forward(self, x):
        y = super(ShiftBlock5x5, self).forward(x)
        if self.res_connect:
            y += x
        return y


class ChannelShuffle(nn.Module):
    def __init__(self, groups):
        super(ChannelShuffle, self).__init__()
        self.groups = groups

    def forward(self, x):
        """Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]"""
        N, C, H, W = x.size()
        g = self.groups
        assert C % g == 0, "Incompatible group size {} for input channel {}".format(
            g, C
        )
        return (
            x.view(N, g, int(C / g), H, W)
                .permute(0, 2, 1, 3, 4)
                .contiguous()
                .view(N, C, H, W)
        )


class ConvBNRelu(nn.Sequential):
    def __init__(
            self,
            input_depth,
            output_depth,
            kernel,
            stride,
            pad,
            no_bias,
            use_relu,
            bn_type,
            group=1,
            *args,
            **kwargs
    ):
        super(ConvBNRelu, self).__init__()

        assert use_relu in ["relu", None]
        if isinstance(bn_type, (list, tuple)):
            assert len(bn_type) == 2
            assert bn_type[0] == "gn"
            gn_group = bn_type[1]
            bn_type = bn_type[0]
        assert bn_type in ["bn", "af", "gn", None]
        assert stride in [1, 2, 4]

        op = Conv2d(
            input_depth,
            output_depth,
            kernel_size=kernel,
            stride=stride,
            padding=pad,
            bias=not no_bias,
            groups=group,
            *args,
            **kwargs
        )
        nn.init.kaiming_normal_(op.weight, mode="fan_out", nonlinearity="relu")
        if op.bias is not None:
            nn.init.constant_(op.bias, 0.0)
        self.add_module("conv", op)

        if bn_type == "bn":
            bn_op = BatchNorm2d(output_depth)
        elif bn_type == "gn":
            bn_op = nn.GroupNorm(num_groups=gn_group, num_channels=output_depth)
        elif bn_type == "af":
            bn_op = FrozenBatchNorm2d(output_depth)
        if bn_type is not None:
            self.add_module("bn", bn_op)

        if use_relu == "relu":
            self.add_module("relu", nn.ReLU(inplace=True))


class SEModule(nn.Module):
    reduction = 4

    def __init__(self, C):
        super(SEModule, self).__init__()
        mid = max(C // self.reduction, 8)
        conv1 = Conv2d(C, mid, 1, 1, 0)
        conv2 = Conv2d(mid, C, 1, 1, 0)

        self.op = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), conv1, nn.ReLU(inplace=True), conv2, nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.op(x)


class Upsample(nn.Module):
    def __init__(self, scale_factor, mode, align_corners=None):
        super(Upsample, self).__init__()
        self.scale = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        return interpolate(
            x, scale_factor=self.scale, mode=self.mode,
            align_corners=self.align_corners
        )


def _get_upsample_op(stride):
    assert (
            stride in [1, 2, 4]
            or stride in [-1, -2, -4]
            or (isinstance(stride, tuple) and all(x in [-1, -2, -4] for x in stride))
    )

    scales = stride
    ret = None
    if isinstance(stride, tuple) or stride < 0:
        scales = [-x for x in stride] if isinstance(stride, tuple) else -stride
        stride = 1
        ret = Upsample(scale_factor=scales, mode="nearest", align_corners=None)

    return ret, stride


class IRFBlock(nn.Module):
    def __init__(
            self,
            input_depth,
            output_depth,
            expansion,
            stride,
            bn_type="bn",
            kernel=3,
            width_divisor=1,
            shuffle_type=None,
            pw_group=1,
            se=False,
            cdw=False,
            dw_skip_bn=False,
            dw_skip_relu=False,
    ):
        super(IRFBlock, self).__init__()

        assert kernel in [1, 3, 5, 7], kernel

        self.use_res_connect = stride == 1 and input_depth == output_depth
        self.output_depth = output_depth

        mid_depth = int(input_depth * expansion)
        mid_depth = _get_divisible_by(mid_depth, width_divisor, width_divisor)

        # pw
        self.pw = ConvBNRelu(
            input_depth,
            mid_depth,
            kernel=1,
            stride=1,
            pad=0,
            no_bias=1,
            use_relu="relu",
            bn_type=bn_type,
            group=pw_group,
        )

        # negative stride to do upsampling
        self.upscale, stride = _get_upsample_op(stride)

        # dw
        if kernel == 1:
            self.dw = nn.Sequential()
        elif cdw:
            dw1 = ConvBNRelu(
                mid_depth,
                mid_depth,
                kernel=kernel,
                stride=stride,
                pad=(kernel // 2),
                group=mid_depth,
                no_bias=1,
                use_relu="relu",
                bn_type=bn_type,
            )
            dw2 = ConvBNRelu(
                mid_depth,
                mid_depth,
                kernel=kernel,
                stride=1,
                pad=(kernel // 2),
                group=mid_depth,
                no_bias=1,
                use_relu="relu" if not dw_skip_relu else None,
                bn_type=bn_type if not dw_skip_bn else None,
            )
            self.dw = nn.Sequential(OrderedDict([("dw1", dw1), ("dw2", dw2)]))
        else:
            self.dw = ConvBNRelu(
                mid_depth,
                mid_depth,
                kernel=kernel,
                stride=stride,
                pad=(kernel // 2),
                group=mid_depth,
                no_bias=1,
                use_relu="relu" if not dw_skip_relu else None,
                bn_type=bn_type if not dw_skip_bn else None,
            )

        # pw-linear
        self.pwl = ConvBNRelu(
            mid_depth,
            output_depth,
            kernel=1,
            stride=1,
            pad=0,
            no_bias=1,
            use_relu=None,
            bn_type=bn_type,
            group=pw_group,
        )

        self.shuffle_type = shuffle_type
        if shuffle_type is not None:
            self.shuffle = ChannelShuffle(pw_group)

        self.se4 = SEModule(output_depth) if se else nn.Sequential()

        self.output_depth = output_depth

    def forward(self, x):
        y = self.pw(x)
        if self.shuffle_type == "mid":
            y = self.shuffle(y)
        if self.upscale is not None:
            y = self.upscale(y)
        y = self.dw(y)
        y = self.pwl(y)
        if self.use_res_connect:
            y += x
        y = self.se4(y)
        return y


class ReLUConvBN(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(ReLUConvBN, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(C_out, affine=affine)
        )

    def forward(self, x):
        return self.op(x)


class DilConv(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
        super(DilConv, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_out, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
                      bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
        )

    def forward(self, x):
        return self.op(x)


# Regular convoluation operations
class CommonConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride):
        super(CommonConv, self).__init__()
        self.op = nn.Sequential(
            nn.Conv2d(C_in, C_out, kernel_size=kernel_size, stride=stride, padding=(kernel_size // 2), bias=False),
            nn.BatchNorm2d(C_out, affine=True),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.op(x)


class SepConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(SepConv, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
        )

    def forward(self, x):
        return self.op(x)


class Zero(nn.Module):
    def __init__(self, stride):
        super(Zero, self).__init__()
        self.stride = stride

    def forward(self, x):
        if self.stride == 1:
            return x.mul(0.)
        return x[:, :, ::self.stride, ::self.stride].mul(0.)


class FactorizedReduce(nn.Module):

    def __init__(self, C_in, C_out, affine=True):
        super(FactorizedReduce, self).__init__()
        assert C_out % 2 == 0
        self.relu = nn.ReLU(inplace=False)
        self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(C_out, affine=affine)

    def forward(self, x):
        x = self.relu(x)
        out = torch.cat([self.conv_1(x), self.conv_2(x[:, :, 1:, 1:])], dim=1)
        out = self.bn(out)
        return out


class FactorizedIncrease(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(FactorizedIncrease, self).__init__()

        self._in_channel = in_channel
        self.op = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear"),
            nn.ReLU(inplace=False),
            nn.Conv2d(self._in_channel, out_channel, 1, stride=1, padding=0),
            nn.BatchNorm2d(out_channel)
        )

    def forward(self, x):
        return self.op(x)


class LearnedGroupConv(nn.Module):
    global_progress = 0.0

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 condense_factor=None, dropout_rate=0.):
        super(LearnedGroupConv, self).__init__()
        self.norm = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout_rate = dropout_rate
        if self.dropout_rate > 0:
            self.drop = nn.Dropout(dropout_rate, inplace=False)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                              padding, dilation, groups=1, bias=False)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups
        self.condense_factor = condense_factor
        if self.condense_factor is None:
            self.condense_factor = self.groups
        ### Parameters that should be carefully used
        self.register_buffer('_count', torch.zeros(1))
        self.register_buffer('_stage', torch.zeros(1))
        self.register_buffer('_mask', torch.ones(self.conv.weight.size()))
        ### Check if arguments are valid
        assert self.in_channels % self.groups == 0, "group number can not be divided by input channels"
        assert self.in_channels % self.condense_factor == 0, "condensation factor can not be divided by input channels"
        assert self.out_channels % self.groups == 0, "group number can not be divided by output channels"

    def forward(self, x):
        self._check_drop()
        x = self.norm(x)
        x = self.relu(x)
        if self.dropout_rate > 0:
            x = self.drop(x)
        ### Masked output
        weight = self.conv.weight * self.mask
        return torch.nn.functional.conv2d(x, weight, None, self.conv.stride,
                                          self.conv.padding, self.conv.dilation, 1)

    def _check_drop(self):
        progress = LearnedGroupConv.global_progress
        delta = 0
        ### Get current stage
        for i in range(self.condense_factor - 1):
            if progress * 2 < (i + 1) / (self.condense_factor - 1):
                stage = i
                break
        else:
            stage = self.condense_factor - 1
        ### Check for dropping
        if not self._reach_stage(stage):
            self.stage = stage
            delta = self.in_channels // self.condense_factor
        if delta > 0:
            self._dropping(delta)
        return

    def _dropping(self, delta):
        weight = self.conv.weight * self.mask
        ### Sum up all kernels
        ### Assume only apply to 1x1 conv to speed up
        assert weight.size()[-1] == 1
        weight = weight.abs().squeeze()
        assert weight.size()[0] == self.out_channels
        assert weight.size()[1] == self.in_channels
        d_out = self.out_channels // self.groups
        ### Shuffle weight
        weight = weight.view(d_out, self.groups, self.in_channels)
        weight = weight.transpose(0, 1).contiguous()
        weight = weight.view(self.out_channels, self.in_channels)
        ### Sort and drop
        for i in range(self.groups):
            wi = weight[i * d_out:(i + 1) * d_out, :]
            ### Take corresponding delta index
            di = wi.sum(0).sort()[1][self.count:self.count + delta]
            for d in di.data:
                self._mask[i::self.groups, d, :, :].fill_(0)
        self.count = self.count + delta

    @property
    def count(self):
        return int(self._count[0])

    @count.setter
    def count(self, val):
        self._count.fill_(val)

    @property
    def stage(self):
        return int(self._stage[0])

    @stage.setter
    def stage(self, val):
        self._stage.fill_(val)

    @property
    def mask(self):
        return torch.autograd.Variable(self._mask)

    def _reach_stage(self, stage):
        return (self._stage >= stage).all()

    @property
    def lasso_loss(self):
        if self._reach_stage(self.groups - 1):
            return 0
        weight = self.conv.weight * self.mask
        ### Assume only apply to 1x1 conv to speed up
        assert weight.size()[-1] == 1
        weight = weight.squeeze().pow(2)
        d_out = self.out_channels // self.groups
        ### Shuffle weight
        weight = weight.view(d_out, self.groups, self.in_channels)
        weight = weight.sum(0).clamp(min=1e-6).sqrt()
        return weight.sum()
