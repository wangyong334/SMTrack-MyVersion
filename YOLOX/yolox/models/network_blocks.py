#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
import warnings

import torch
import torch.nn as nn


import torch.nn.functional as F

class SiLU(nn.Module):
    """export-friendly version of nn.SiLU()"""

    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)


def get_activation(name="silu", inplace=True):
    if name == "silu":
        module = nn.SiLU(inplace=inplace)
    elif name == "relu":
        module = nn.ReLU(inplace=inplace)
    elif name == "lrelu":
        module = nn.LeakyReLU(0.1, inplace=inplace)
    else:
        raise AttributeError("Unsupported act type: {}".format(name))
    return module


class BaseConv(nn.Module):
    """A Conv2d -> Batchnorm -> silu/leaky relu block"""

    def __init__(
        self, in_channels, out_channels, ksize, stride, groups=1, bias=False, act="silu"
    ):
        super().__init__()
        # same padding
        pad = (ksize - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=ksize,
            stride=stride,
            padding=pad,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = get_activation(act, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class DWConv(nn.Module):
    """Depthwise Conv + Conv"""

    def __init__(self, in_channels, out_channels, ksize, stride=1, act="silu"):
        super().__init__()
        self.dconv = BaseConv(
            in_channels,
            in_channels,
            ksize=ksize,
            stride=stride,
            groups=in_channels,
            act=act,
        )
        self.pconv = BaseConv(
            in_channels, out_channels, ksize=1, stride=1, groups=1, act=act
        )

    def forward(self, x):
        x = self.dconv(x)
        return self.pconv(x)


class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(
        self,
        in_channels,
        out_channels,
        shortcut=True,
        expansion=0.5,
        depthwise=False,
        act="silu",
    ):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        Conv = DWConv if depthwise else BaseConv
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv2 = Conv(hidden_channels, out_channels, 3, stride=1, act=act)
        self.use_add = shortcut and in_channels == out_channels

    def forward(self, x):
        y = self.conv2(self.conv1(x))
        if self.use_add:
            y = y + x
        return y


class ResLayer(nn.Module):
    "Residual layer with `in_channels` inputs."

    def __init__(self, in_channels: int):
        super().__init__()
        mid_channels = in_channels // 2
        self.layer1 = BaseConv(
            in_channels, mid_channels, ksize=1, stride=1, act="lrelu"
        )
        self.layer2 = BaseConv(
            mid_channels, in_channels, ksize=3, stride=1, act="lrelu"
        )

    def forward(self, x):
        out = self.layer2(self.layer1(x))
        return x + out


class SPPBottleneck(nn.Module):
    """Spatial pyramid pooling layer used in YOLOv3-SPP"""

    def __init__(
        self, in_channels, out_channels, kernel_sizes=(5, 9, 13), activation="silu"
    ):
        super().__init__()
        hidden_channels = in_channels // 2
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=activation)
        self.m = nn.ModuleList(
            [
                nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2)
                for ks in kernel_sizes
            ]
        )
        conv2_channels = hidden_channels * (len(kernel_sizes) + 1)
        self.conv2 = BaseConv(conv2_channels, out_channels, 1, stride=1, act=activation)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.cat([x] + [m(x) for m in self.m], dim=1)
        x = self.conv2(x)
        return x


class CSPLayer(nn.Module):
    """C3 in yolov5, CSP Bottleneck with 3 convolutions"""

    def __init__(
        self,
        in_channels,
        out_channels,
        n=1,
        shortcut=True,
        expansion=0.5,
        depthwise=False,
        act="silu",
    ):
        """
        Args:
            in_channels (int): input channels.
            out_channels (int): output channels.
            n (int): number of Bottlenecks. Default value: 1.
        """
        # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        hidden_channels = int(out_channels * expansion)  # hidden channels
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv2 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv3 = BaseConv(2 * hidden_channels, out_channels, 1, stride=1, act=act)
        module_list = [
            Bottleneck(
                hidden_channels, hidden_channels, shortcut, 1.0, depthwise, act=act
            )
            for _ in range(n)
        ]
        self.m = nn.Sequential(*module_list)

    def forward(self, x):
        x_1 = self.conv1(x)
        x_2 = self.conv2(x)
        x_1 = self.m(x_1)
        x = torch.cat((x_1, x_2), dim=1)
        return self.conv3(x)


class Focus(nn.Module):
    """Focus width and height information into channel space."""

    def __init__(self, in_channels, out_channels, ksize=1, stride=1, act="silu"):
        super().__init__()
        self.conv = BaseConv(in_channels * 4, out_channels, ksize, stride, act=act)

    def forward(self, x):
        # shape of x (b,c,w,h) -> y(b,4c,w/2,h/2)
        patch_top_left = x[..., ::2, ::2]
        patch_top_right = x[..., ::2, 1::2]
        patch_bot_left = x[..., 1::2, ::2]
        patch_bot_right = x[..., 1::2, 1::2]
        x = torch.cat(
            (
                patch_top_left,
                patch_bot_left,
                patch_top_right,
                patch_bot_right,
            ),
            dim=1,
        )
        return self.conv(x)


class SpikeCSPDarknet(nn.Module):
    def __init__(
        self,
        dep_mul,
        wid_mul,
        out_features=("dark3", "dark4", "dark5"),
        depthwise=False,
        act="silu",
    ):
        super().__init__()
        assert out_features, "please provide output features of Darknet"
        self.out_features = out_features
        self.width = wid_mul

        base_channels = int(wid_mul * 128)  # 64
        # base_depth = max(round(dep_mul * 3), 1)  # 3

        # stem
        self.getT = MS_GetT(T=4)
        # self.stem = Focus(3, base_channels, ksize=3, act=act)

        # dark2
        self.dark2 = nn.Sequential(
            MS_DownSampling(in_channels=3, embed_dims=base_channels,kernel_size=7,stride=4,padding=2,first_layer=True),
            MS_AllConvBlock(input_dim=base_channels,mlp_ratio=4,sep_kernel_size=7,group=True)
        )

        # dark3
        self.dark3 = nn.Sequential(
            MS_DownSampling(in_channels=base_channels, embed_dims=base_channels*2,kernel_size=3,stride=2,padding=1,first_layer=False),
            MS_AllConvBlock(input_dim=base_channels*2,mlp_ratio=4,sep_kernel_size=7,group=True)
        )

        # dark4
        self.dark4 = nn.Sequential(
            MS_DownSampling(in_channels=2*base_channels, embed_dims=base_channels*4,kernel_size=3,stride=2,padding=1,first_layer=False),
            MS_AllConvBlock(input_dim=4*base_channels,mlp_ratio=3,sep_kernel_size=7)
        )

        # dark5
        self.dark5 = nn.Sequential(
            MS_DownSampling(in_channels=4*base_channels, embed_dims=base_channels*8,kernel_size=3,stride=2,padding=1,first_layer=False),
            MS_AllConvBlock(input_dim=8*base_channels,mlp_ratio=2,sep_kernel_size=7),
            SpikeSPPF(c1=8*base_channels,c2=8*base_channels,k=5)
        )

    def forward(self, x):
        outputs = {}
        # x = self.stem(x)
        x = self.getT(x)        #0
        # outputs["stem"] = x
        x = self.dark2(x)       #2
        # outputs["dark2"] = x
        x = self.dark3(x)       #4
        outputs["dark3"] = x
        x = self.dark4(x)       #6
        outputs["dark4"] = x
        x = self.dark5(x)       #9
        outputs["dark5"] = x
        return {k: v for k, v in outputs.items() if k in self.out_features}


class SpikeDFL(nn.Module):
    """
    Integral module of Distribution Focal Loss (DFL).

    Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    """

    def __init__(self, c1=16):
        """Initialize a convolutional layer with a given number of input channels."""
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)  # [0,1,2,...,15]
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))  # 这里不是脉冲驱动的，但是是整数乘法
        self.c1 = c1  # 本质上就是个加权和。输入是每个格子的概率(小数)，权重是每个格子的位置(整数)
        self.lif = mem_update()

    def forward(self, x):
        """Applies a transformer layer on input tensor 'x' and returns a tensor."""
        b, c, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)  # 原版

class mem_update(nn.Module):
    def __init__(self, act=False):
        super(mem_update, self).__init__()
        # self.actFun= torch.nn.LeakyReLU(0.2, inplace=False)
        self.decay = 0.25  # 0.25 # decay constants

        self.act = act
        self.qtrick = MultiSpike4()  # change the max value

    def forward(self, x):

        spike = torch.zeros_like(x[0]).to(x.device)
        output = torch.zeros_like(x)
        mem_old = 0
        time_window = x.shape[0]
        for i in range(time_window):
            if i >= 1:
                mem = (mem_old - spike.detach()) * self.decay + x[i]

            else:
                mem = x[i]
            spike = self.qtrick(mem)

            mem_old = mem.clone()
            output[i] = spike
        # print(output[0][0][0][0])
        return output

class MultiSpike8(nn.Module):  # 直接调用实例化的quant6无法实现深拷贝。解决方案是像下面这样用嵌套的类

    class quant8(torch.autograd.Function):

        @staticmethod
        def forward(ctx, input):
            ctx.save_for_backward(input)
            return torch.round(torch.clamp(input, min=0, max=8))

        @staticmethod
        def backward(ctx, grad_output):
            input, = ctx.saved_tensors
            grad_input = grad_output.clone()
            #             print("grad_input:",grad_input)
            grad_input[input < 0] = 0
            grad_input[input > 8] = 0
            return grad_input

    def forward(self, x):
#         print(self.quant8.apply(x))
        return self.quant8.apply(x)

class MultiSpike4(nn.Module):

    class quant4(torch.autograd.Function):

        @staticmethod
        def forward(ctx, input):
            ctx.save_for_backward(input)
            return torch.round(torch.clamp(input, min=0, max=4))

        @staticmethod
        def backward(ctx, grad_output):
            input, = ctx.saved_tensors
            grad_input = grad_output.clone()
            #             print("grad_input:",grad_input)
            grad_input[input < 0] = 0
            grad_input[input > 4] = 0
            return grad_input

    def forward(self, x):
        return self.quant4.apply(x)

class MultiSpike2(nn.Module):  # 直接调用实例化的quant6无法实现深拷贝。解决方案是像下面这样用嵌套的类

    class quant2(torch.autograd.Function):

        @staticmethod
        def forward(ctx, input):
            ctx.save_for_backward(input)
            return torch.round(torch.clamp(input, min=0, max=2))

        @staticmethod
        def backward(ctx, grad_output):
            input, = ctx.saved_tensors
            grad_input = grad_output.clone()
            #             print("grad_input:",grad_input)
            grad_input[input < 0] = 0
            grad_input[input > 2] = 0
            return grad_input

    def forward(self, x):
        return self.quant2.apply(x)

class MultiSpike1(nn.Module):

    class quant1(torch.autograd.Function):

        @staticmethod
        def forward(ctx, input):
            ctx.save_for_backward(input)
            return torch.round(torch.clamp(input, min=0, max=1))

        @staticmethod
        def backward(ctx, grad_output):
            input, = ctx.saved_tensors
            grad_input = grad_output.clone()
            #             print("grad_input:",grad_input)
            grad_input[input < 0] = 0
            grad_input[input > 1] = 0
            return grad_input

    def forward(self, x):
        return self.quant1.apply(x)


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    # Pad to 'same' shape outputs
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p



@torch.jit.script
def jit_mul(x, y):
    return x.mul(y)

@torch.jit.script
def jit_sum(x):
    return x.sum(dim=[-1, -2], keepdim=True)

class RepConv(nn.Module):
    def __init__(
            self,
            in_channel,
            out_channel,
            kernel_size=3,
            bias=False,
            group=1
    ):
        super().__init__()
        padding = int((kernel_size - 1) / 2)
        # hidden_channel = in_channel
        conv1x1 = nn.Conv2d(in_channel, in_channel, 1, 1, 0, bias=False, groups=group)
        bn = BNAndPadLayer(pad_pixels=padding, num_features=in_channel)
        conv3x3 = nn.Sequential(
            # mem_update(), #11111
            nn.Conv2d(in_channel, in_channel, kernel_size, 1, 0, groups=in_channel, bias=False),  # 这里也是分组卷积
            # mem_update(),  #11111
            nn.Conv2d(in_channel, out_channel, 1, 1, 0, groups=group, bias=False),
            nn.BatchNorm2d(out_channel),
        )

        self.body = nn.Sequential(conv1x1, bn, conv3x3)

    def forward(self, x):
        return self.body(x)


class SepRepConv(nn.Module):
    def __init__(
            self,
            in_channel,
            out_channel,
            kernel_size=3,
            bias=False,
            group=1
    ):
        super().__init__()
        padding = int((kernel_size - 1) / 2)
        # hidden_channel = in_channel
        bn = BNAndPadLayer(pad_pixels=padding, num_features=in_channel)
        conv3x3 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1, 1, 0, groups=group, bias=False),  # 这里也是分组卷积
            # mem_update(), #11111
            nn.Conv2d(out_channel, out_channel, kernel_size, 1, 0, groups=out_channel, bias=False),
        )

        self.body = nn.Sequential(bn, conv3x3)

    def forward(self, x):
        return self.body(x)

class BNAndPadLayer(nn.Module):
    def __init__(
            self,
            pad_pixels,
            num_features,
            eps=1e-5,
            momentum=0.1,
            affine=True,
            track_running_stats=True,
    ):
        super(BNAndPadLayer, self).__init__()
        self.bn = nn.BatchNorm2d(
            num_features, eps, momentum, affine, track_running_stats
        )
        self.pad_pixels = pad_pixels

    def forward(self, input):
        output = self.bn(input)
        if self.pad_pixels > 0:
            if self.bn.affine:
                pad_values = (
                        self.bn.bias.detach()
                        - self.bn.running_mean
                        * self.bn.weight.detach()
                        / torch.sqrt(self.bn.running_var + self.bn.eps)
                )
            else:
                pad_values = -self.bn.running_mean / torch.sqrt(
                    self.bn.running_var + self.bn.eps
                )
            output = F.pad(output, [self.pad_pixels] * 4)
            pad_values = pad_values.view(1, -1, 1, 1)
            output[:, :, 0: self.pad_pixels, :] = pad_values
            output[:, :, -self.pad_pixels:, :] = pad_values
            output[:, :, :, 0: self.pad_pixels] = pad_values
            output[:, :, :, -self.pad_pixels:] = pad_values
        return output

    @property
    def weight(self):
        return self.bn.weight

    @property
    def bias(self):
        return self.bn.bias

    @property
    def running_mean(self):
        return self.bn.running_mean

    @property
    def running_var(self):
        return self.bn.running_var

    @property
    def eps(self):
        return self.bn.eps

class SepAllConv(nn.Module):
    r"""
    Inverted separable convolution from MobileNetV2: https://arxiv.org/abs/1801.04381.
    """

    def __init__(self,
                 dim,
                 expansion_ratio=2,
                 act2_layer=nn.Identity,
                 bias=False,
                 kernel_size=3,  # 7,3
                 padding=1):
        super().__init__()
        padding = int((kernel_size - 1) / 2)
        med_channels = int(expansion_ratio * dim)
        self.pwconv1 = nn.Conv2d(dim, med_channels, kernel_size=1, stride=1, bias=bias)
        self.dwconv2 = nn.Conv2d(
            med_channels, med_channels, kernel_size=kernel_size,  # 7*7
            padding=padding, groups=med_channels, bias=bias)  # depthwise conv
        #         self.pwconv3 = nn.Conv2d(med_channels, dim, kernel_size=1, stride=1, bias=bias,groups=1)
        self.pwconv3 = SepRepConv(med_channels, dim)  # 这里将sepconv最后一个卷积替换为重参数化卷积  大概提0.5个点，可以保留

        self.bn1 = nn.BatchNorm2d(med_channels)
        self.bn2 = nn.BatchNorm2d(med_channels)
        self.bn3 = nn.BatchNorm2d(dim)

        self.lif1 = mem_update()
        self.lif2 = mem_update()
        self.lif3 = mem_update()

    def forward(self, x):
        T, B, C, H, W = x.shape
        #         print("x.shape:",x.shape)
        x = self.lif1(x)  # x1_lif:0.2328  x2_lif:0.0493  这里x2的均值偏小，因此其经过bn和lif后也偏小，发放率比较低；而x1均值偏大，因此发放率也高
        x = self.bn1(self.pwconv1(x.flatten(0, 1))).reshape(T, B, -1, H, W)  # flatten：从第0维开始，展开到第一维
        x = self.lif2(x)
        x = self.bn2(self.dwconv2(x.flatten(0, 1))).reshape(T, B, -1, H, W)
        x = self.lif3(x)
        x = self.bn3(self.pwconv3(x.flatten(0, 1))).reshape(T, B, -1, H, W)
        return x


class SepConv(nn.Module):
    r"""
    Inverted separable convolution from MobileNetV2: https://arxiv.org/abs/1801.04381.
    """

    def __init__(self,
                 dim,
                 expansion_ratio=2,
                 act2_layer=nn.Identity,
                 bias=False,
                 kernel_size=3,  # 7,3
                 padding=1):
        super().__init__()
        padding = int((kernel_size - 1) / 2)
        med_channels = int(expansion_ratio * dim)
        self.pwconv1 = nn.Conv2d(dim, med_channels, kernel_size=1, stride=1, bias=bias)
        self.dwconv2 = nn.Conv2d(
            med_channels, med_channels, kernel_size=kernel_size,  # 7*7
            padding=padding, groups=med_channels, bias=bias)  # depthwise conv
        #         self.pwconv3 = nn.Conv2d(med_channels, dim, kernel_size=1, stride=1, bias=bias,groups=1)
        self.pwconv3 = SepRepConv(med_channels, dim)  # 这里将sepconv最后一个卷积替换为重参数化卷积  大概提0.5个点，可以保留

        self.bn1 = nn.BatchNorm2d(med_channels)
        self.bn2 = nn.BatchNorm2d(med_channels)
        self.bn3 = nn.BatchNorm2d(dim)

        self.lif1 = mem_update()
        self.lif2 = mem_update()
        self.lif3 = mem_update()

    def forward(self, x):
        T, B, C, H, W = x.shape
        #         print("x.shape:",x.shape)
        x = self.lif1(x)  # x1_lif:0.2328  x2_lif:0.0493  这里x2的均值偏小，因此其经过bn和lif后也偏小，发放率比较低；而x1均值偏大，因此发放率也高
        x = self.bn1(self.pwconv1(x.flatten(0, 1))).reshape(T, B, -1, H, W)  # flatten：从第0维开始，展开到第一维
        x = self.lif2(x)
        x = self.bn2(self.dwconv2(x.flatten(0, 1))).reshape(T, B, -1, H, W)
        x = self.lif3(x)
        x = self.bn3(self.pwconv3(x.flatten(0, 1))).reshape(T, B, -1, H, W)
        return x


class MS_ConvBlock(nn.Module):
    def __init__(self, input_dim, mlp_ratio=4., sep_kernel_size=7, full=False):  # in_channels(out_channels), 内部扩张比例
        super().__init__()

        self.full = full
        self.Conv = SepConv(dim=input_dim, kernel_size=sep_kernel_size)  # 内部扩张2倍
        self.mlp_ratio = mlp_ratio

        self.lif1 = mem_update()
        self.lif2 = mem_update()

        self.conv1 = RepConv(input_dim, int(input_dim * mlp_ratio))  # 137以外的模型，在第一个block不做分组

        self.bn1 = nn.BatchNorm2d(int(input_dim * mlp_ratio))  # 这里可以进行改进

        self.conv2 = RepConv(int(input_dim * mlp_ratio), input_dim)
        self.bn2 = nn.BatchNorm2d(input_dim)  # 这里可以进行改进

    # @get_local('x_feat')
    def forward(self, x):
        T, B, C, H, W = x.shape
        x = self.Conv(x) + x  # sepconv  pw+dw+pw

        x_feat = x

        x = self.bn1(self.conv1(self.lif1(x).flatten(0, 1))).reshape(T, B, int(self.mlp_ratio * C), H, W)
        # repconv，对应conv_mixer，包含1*1,3*3,1*1三个卷积，等价于一个3*3卷积
        x = self.bn2(self.conv2(self.lif2(x).flatten(0, 1))).reshape(T, B, C, H, W)
        x = x_feat + x

        return x


class MS_AllConvBlock(nn.Module):  # standard conv
    def __init__(self, input_dim, mlp_ratio=4., sep_kernel_size=7, group=False):  # in_channels(out_channels), 内部扩张比例
        super().__init__()

        self.Conv = SepConv(dim=input_dim, kernel_size=sep_kernel_size)

        self.mlp_ratio = mlp_ratio
        self.conv1 = MS_StandardConv(input_dim, int(input_dim * mlp_ratio), 3)
        self.conv2 = MS_StandardConv(int(input_dim * mlp_ratio), input_dim, 3)

    # @get_local('x_feat')
    def forward(self, x):
        T, B, C, H, W = x.shape
        x = self.Conv(x) + x  # sepconv  pw+dw+pw

        x_feat = x

        x = self.conv1(x)
        x = self.conv2(x)
        x = x_feat + x

        return x


class MS_StandardConv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1):  # in_channels(out_channels), 内部扩张比例
        super().__init__()
        self.c1 = c1
        self.c2 = c2
        self.s = s
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.lif = mem_update()

    def forward(self, x):
        T, B, C, H, W = x.shape
        x = self.bn(self.conv(self.lif(x).flatten(0, 1))).reshape(T, B, self.c2, int(H / self.s), int(W / self.s))
        return x


class MS_DownSampling(nn.Module):
    def __init__(self, in_channels=2, embed_dims=256, kernel_size=3, stride=2, padding=1, first_layer=True):
        super().__init__()

        self.encode_conv = nn.Conv2d(in_channels, embed_dims, kernel_size=kernel_size, stride=stride, padding=padding)
        self.encode_bn = nn.BatchNorm2d(embed_dims)
        if not first_layer:
            self.encode_lif = mem_update()
        # self.pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        T, B, _, _, _ = x.shape

        if hasattr(self, "encode_lif"):  # 如果不是第一层
            # x_pool = self.pool(x)
            x = self.encode_lif(x)

        x = self.encode_conv(x.flatten(0, 1))
        _, C, H, W = x.shape
        x = self.encode_bn(x).reshape(T, B, -1, H, W).contiguous()

        return x


class MS_GetT(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, T=4):
        super().__init__()
        self.T = T
        self.in_channels = in_channels

    def forward(self, x):
        x = (x.unsqueeze(0)).repeat(self.T, 1, 1, 1, 1)
        return x

class MS_HistoryGetT(nn.Module):
    def __init__(self, in_channels=1, T=4, sim_thresh=0.9850, mode='cosine'):
        super().__init__()
        self.T = T
        self.sim_thresh = sim_thresh
        self.mode = mode  # 'cosine' or 'nmi'
        self.history = []

    def forward(self, x):
        if not self.history or self._check_similarity(x) < self.sim_thresh:
            self.reset_history()
            self._update_history(x)
            return x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)

        use_history = min(len(self.history), self.T - 1)
        repeat_times = self.T - use_history

        if use_history == 0:
            output = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
        else:
            hist_part = [f.unsqueeze(0) for f in self.history[-use_history:]]
            curr_part = [x.unsqueeze(0)] * repeat_times
            output = torch.cat(hist_part + curr_part, dim=0)

        self._update_history(x)
        return output

    def _check_similarity(self, x):
        if not self.history:
            return 0.0

        last_feat = self.history[-1].flatten()
        current_feat = x.flatten()

        if last_feat.shape != current_feat.shape:
            return 0.0

        if self.mode == 'nmi':
            last_np = last_feat.detach().cpu().numpy()
            curr_np = current_feat.detach().cpu().numpy()
            return normalized_mutual_info_score(last_np, curr_np)
        else:
            return F.cosine_similarity(last_feat, current_feat, dim=0)

    def _update_history(self, feat):
        with torch.no_grad():
            self.history.append(feat.detach())
            if len(self.history) > self.T - 1:
                self.history.pop(0)

    def reset_history(self):
        self.history.clear()
#
# class MS_HistoryGetT(nn.Module):
#     def __init__(self, in_channels=1, T=4, sim_thresh=0.98):
#         super().__init__()
#         self.T = T
#         self.sim_thresh = sim_thresh
#         self.history = []  # 存储历史特征队列
#
#     def forward(self, x):
#         """输入形状: [batch, channels, H, W]"""
#         x = x.squeeze(0)  # 移除batch维度
#
#         # 初始化或相似度不足时，使用全复制策略
#         if not self.history:
#             # print("初始化")
#             self.history = [x.detach()]
#             return x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
#
#         if self._check_similarity(x) < self.sim_thresh:
#             # print("重置")
#             self.history = []  # 存储历史特征队列
#             self.history = [x.detach()]
#             return x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
#
#         # 当历史特征足够时，组合历史特征
#         use_history = min(len(self.history), self.T - 1)
#         repeat_times = self.T - use_history
#
#         # 组合历史特征和当前特征
#         hist_part = [f.unsqueeze(0) for f in self.history[-use_history:]]
#         curr_part = [x.unsqueeze(0)] * repeat_times
#         output = torch.cat(hist_part + curr_part, dim=0)
#
#         # 更新历史（保留最多T-1个）
#         self._update_history(x.detach())
#         return output
#
#     def _nmi_similarity(self, x):
#         """使用归一化互信息计算相似度"""
#         if not self.history:
#             return 0.0
#         last_feat = self.history[-1].flatten()
#         current_feat = x.flatten()
#
#         # 转换为0-255整型数组
#         # img1 = (last_feat.detach().numpy() * 255).astype(np.uint8)
#         # img2 = (current_feat.detach().numpy() * 255).astype(np.uint8)
#         f = normalized_mutual_info_score(last_feat, current_feat)
#         # print(f)
#         return f
#
#     def _check_similarity(self, x):
#         """计算与上一帧的余弦相似度"""
#         if not self.history:
#             return 0.0
#         last_feat = self.history[-1].flatten()
#         current_feat = x.flatten()
#         if last_feat.shape != current_feat.shape:
#             return 0.0
#         f = F.cosine_similarity(last_feat, current_feat, dim=0)
#         # print(f)
#         return f
#
#     def _update_history(self, feat):
#         """更新历史特征队列"""
#         self.history.append(feat)
#         if len(self.history) > self.T - 1:
#             self.history.pop(0)

# class MS_HistoryGetT(nn.Module):
#     def __init__(self, in_channels=1, T=4, sim_thresh=0.8):
#         super().__init__()
#         self.T = T
#         self.sim_thresh = sim_thresh
#         self.history = []  # 存储前T-1个历史特征
#         self.current_feat = None
#
#     def forward(self, x):
#         """输入形状: [batch, channels, H, W]"""
#         batch_size = x.shape[0]
#
#         # 初始化时或相似度不足时，使用全复制策略
#         if not self.history:
#             output = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
#             self._update_history(x.detach())
#             return output
#
#         # 计算相似度
#         last_feat = self.history[-1].flatten()
#         current_feat = x.detach().flatten()
#         sim = F.cosine_similarity(last_feat, current_feat, dim=0)
#
#         if sim < self.sim_thresh:
#             # 重置历史记录
#             self.history = []
#             output = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
#             self._update_history(x.detach())
#         else:
#             # 构建时序序列
#             time_steps = min(len(self.history), self.T - 1)
#             repeat_times = self.T - time_steps
#
#             # 组合历史特征和当前特征
#             hist_part = [f.unsqueeze(0) for f in self.history[-time_steps:]]
#             curr_part = [x.unsqueeze(0)] * repeat_times
#             output = torch.cat(hist_part + curr_part, dim=0)
#
#             # 更新历史（保留最多T-1个）
#             self._update_history(x.detach())
#
#         return output
#
#     def _update_history(self, feat):
#         """更新历史特征缓存"""
#         self.history.append(feat)
#         if len(self.history) > self.T - 1:
#             self.history.pop(0)

class MS_CancelT(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, T=2):
        super().__init__()
        self.T = T

    def forward(self, x):
        x = x.mean(0)
        return x


class SpikeConv(nn.Module):
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)
    default_act = nn.SiLU()  # default activation

    def __init__(self, in_channels, out_channels, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.lif = mem_update()
        self.bn = nn.BatchNorm2d(out_channels)
        self.s = s
        # self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        T, B, C, H, W = x.shape
        H_new = int(H / self.s)
        W_new = int(W / self.s)
        x = self.lif(x)
        x = self.bn(self.conv(x.flatten(0, 1))).reshape(T, B, -1, H_new, W_new)
        return x


class SpikeConvWithoutBN(nn.Module):
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, autopad(kernel_size, p, d), groups=g, dilation=d, bias=True)
        self.lif = mem_update()

        self.s = stride
        # self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        T, B, C, H, W = x.shape
        H_new = int(H / self.s)
        W_new = int(W / self.s)
        x = self.lif(x)
        x = self.conv(x.flatten(0, 1)).reshape(T, B, -1, H_new, W_new)
        return x


class SpikeSPPF(nn.Module):
    # Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
    def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = SpikeConv(c1, c_, 1, 1)
        self.cv2 = SpikeConv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            T, B, C, H, W = x.shape
            warnings.simplefilter('ignore')  # suppress torch 1.9.0 max_pool2d() warning
            y1 = self.m(x.flatten(0, 1)).reshape(T, B, -1, H, W)
            y2 = self.m(y1.flatten(0, 1)).reshape(T, B, -1, H, W)
            y3 = self.m(y2.flatten(0, 1)).reshape(T, B, -1, H, W)
            return self.cv2(torch.cat((x, y1, y2, y3), 2))


class MS_Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):  # 这里输入x是一个list
        for i in range(len(x)):
            if x[i].dim() == 5:
                x[i] = x[i].mean(0)
        return torch.cat(x, self.d)



class SpikeCSPLayer(nn.Module):
    def __init__(self, input_dim, mlp_ratio=4., sep_kernel_size=7, group=False):  # in_channels(out_channels), 内部扩张比例
        super().__init__()

        self.Conv = SepConv(dim=input_dim, kernel_size=sep_kernel_size)

        self.mlp_ratio = mlp_ratio
        self.conv1 = MS_StandardConv(input_dim, int(input_dim * mlp_ratio), 3)
        self.conv2 = MS_StandardConv(int(input_dim * mlp_ratio), input_dim, 3)

    # @get_local('x_feat')
    def forward(self, x):
        T, B, C, H, W = x.shape
        x = self.Conv(x) + x  # sepconv  pw+dw+pw

        x_feat = x

        x = self.conv1(x)
        x = self.conv2(x)
        x = x_feat + x

        return x