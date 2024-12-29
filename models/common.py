import json
import math
import platform
import warnings
from copy import copy
from pathlib import Path

import sys
sys.path.append('/home/algointern/project/EMS-YOLO-main/utils')

import cv2
import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
from PIL import Image
from torch import Tensor
from torch.cuda import amp
import torch.nn.functional as F

from einops import rearrange

from general import (LOGGER, check_requirements, check_suffix, colorstr, increment_path, make_divisible,
                           non_max_suppression, scale_coords, xywh2xyxy, xyxy2xywh)
from plots import Annotator, colors, save_one_box
from torch_utils import time_sync

from dataloader import exif_transpose, letterbox
# from test import RFAConv
# from test import ScConv
# from spikingjelly.activation_based import layer

# from spikingjelly.clock_driven.neuron import (
#     MultiStepParametricLIFNode,
#     MultiStepLIFNode,
# )
thresh = 0.5  # 0.5 # neuronal threshold 神经元阈值
lens = 0.5  # 0.5 # hyper-parameters of approximate function 近似函数的超参数
decay = 0.25  # 0.25 # decay constants 衰变常数
time_window = 4

'''===========1.autopad：根据输入的卷积核计算该卷积模块所需的pad值================'''


# 为same卷积或者same池化自动扩充
# 通过卷积核的大小来计算需要的padding为多少才能把tensor补成原来的形状
def autopad(k, p=None):  # kernel, padding 内核，填充
    # Pad to 'same'
    if p is None:
        # 如果k是int 则进行k//2 若不是则进行x//2
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


# 通过继承torch.autograd.Function类，并实现forward 和 backward函数
class ActFun(torch.autograd.Function):
    # torch.autograd.Function 自定义反向求导

    @staticmethod
    def forward(ctx, input):
        # ctx为上下文context，save_for_backward函数可以将对象保存起来，接收包含输入的Tensor并返回包含输出的Tensor。
        # ctx是环境变量，用于提供反向传播是需要的信息。可通过ctx.save_for_backward方法缓存数据。
        ctx.save_for_backward(input)
        return input.gt(thresh).float()

    @staticmethod
    def backward(ctx, grad_output):
        """
        在backward函数中，接收包含了损失梯度的Tensor，
        我们需要根据输入计算损失的梯度。
        """
        # backward函数的输出的参数个数需要与forward函数的输入的参数个数一致。
        # grad_output默认值为tensor([1.])，对应的forward的输出为标量。
        # ctx.saved_tensors会返回forward函数内存储的对象
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        temp = abs(input - thresh) < lens
        temp = temp / (2 * lens)  # ？
        return grad_input * temp.float()


act_fun = ActFun.apply


class mem_update(nn.Module):
    def __init__(self, act=False):
        super(mem_update, self).__init__()
        # self.actFun= torch.nn.LeakyReLU(0.2, inplace=False)
        self.actFun = nn.SiLU()
        # silu(x)=x*σ(x),where σ(x) is logistic sigmoid
        # logistic sigmoid:sig(x)=1/(1+exp(-x))
        self.act = act

    def forward(self, x):
        mem = torch.zeros_like(x[0]).to(x.device)
        # 该语句的主要作用是生成一个值全为0的、维度与输入尺寸相同的矩阵。
        # torch.zeros_like(input)相当于torch.zeros(input.size(),dtype=input.dtype,layout=input.layout, device=input.device)
        spike = torch.zeros_like(x[0]).to(x.device)
        output = torch.zeros_like(x)
        mem_old = 0
        for i in range(time_window):
            # range(1)->[0]
            if i >= 1:
                # .detach() 返回一个新的Variable，从当前计算图中分离下来的，但是仍指向原变量的存放位置,不同之处只是requires_grad为false，得到的这个Variable永远不需要计算其梯度，不具有grad。
                # 即使之后重新将它的requires_grad置为true,它也不会具有梯度grad。
                # mem = mem_old * decay * (1-spike.detach()) + x[i]
                mem = self.jit_neuronal_charge(x[i], mem_old, decay, spike)
            else:
                mem = x[i]
            if self.act:
                spike = self.actFun(mem)
            else:
                spike = act_fun(mem)

            mem_old = mem.clone()
            output[i] = spike
        # print(output[0][0][0][0])
        return output

    @staticmethod
    @torch.jit.script
    def jit_neuronal_charge(x: torch.Tensor, mem_old: torch.Tensor, decay: float, spike: torch.Tensor):
        return mem_old * decay * (1 - spike.detach()) + x






# class mem_update(nn.Module):
#     def __init__(self, act=False, ecs_tau: float = 5., alpha: float = 0.75, beta: float = 0.25, ECS=False):
#         super(mem_update, self).__init__()
#         # self.actFun= torch.nn.LeakyReLU(0.2, inplace=False)
#         self.ECS = ECS
#         self.actFun = nn.SiLU()
#         # silu(x)=x*σ(x),where σ(x) is logistic sigmoid
#         # logistic sigmoid:sig(x)=1/(1+exp(-x))
#         self.act = act
#         self.alpha = alpha
#         self.beta = beta
#         self.ecs_tau = ecs_tau
#         self.spread = None
#
#     def forward(self, x):
#         mem = torch.zeros_like(x[0]).to(x.device)
#         # 该语句的主要作用是生成一个值全为0的、维度与输入尺寸相同的矩阵。
#         # torch.zeros_like(input)相当于torch.zeros(input.size(),dtype=input.dtype,layout=input.layout, device=input.device)
#         spike = torch.zeros_like(x[0]).to(x.device)
#         output = torch.zeros_like(x)
#         mem_old = 0
#         ecs = 0.
#         fecs = 0.
#         if self.ECS:
#             if self.spread is None:
#                 self.InitEcsSpread(x[0])
#         for i in range(time_window):
#             # range(1)->(0,1)
#             if i >= 1:
#                 # .detach() 返回一个新的Variable，从当前计算图中分离下来的，但是仍指向原变量的存放位置,不同之处只是requires_grad为false，得到的这个Variable永远不需要计算其梯度，不具有grad。
#                 # 即使之后重新将它的requires_grad置为true,它也不会具有梯度grad。
#                 if self.ECS:
#                     mem = self.jit_neuronal_charge1(x[i], mem_old, decay, spike, fecs)
#                 else:
#                     mem = self.jit_neuronal_charge(x[i], mem_old, decay, spike, fecs)
#             else:
#                 if self.ECS:
#                     mem = x[i] + fecs
#                 else:
#                     mem = x[i]
#             if self.act:
#                 spike = self.actFun(mem)
#             else:
#                 spike = act_fun(mem)
#
#             if self.ECS:
#                 ecs = self.alpha * self.spread(spike) + (1. - 1. / self.ecs_tau) * ecs
#                 fecs = self.beta * torch.tanh(ecs)
#
#             mem_old = mem.clone()
#             output[i] = spike
#         # print(output[0][0][0][0])
#         return output
#
#     def InitEcsSpread(self, x: torch.Tensor):
#         if x.ndim == 4:
#             shape = x.shape
#             self.spread = nn.Sequential(
#                 # nn.Conv2d(shape[1], shape[1],
#                 #           kernel_size=3, padding=1, device=x.device, groups=shape[1]),
#                 # layer.Conv2d(shape[1], shape[1],
#                 #              kernel_size=3, padding=1, groups=shape[1]),
#                 Conv2d(shape[1], shape[1],
#                        kernel_size=3, padding=1, device=x.device, groups=shape[1]),
#                 # nn.Conv2d(shape[1], shape[1], kernel_size=1, device=x.device)
#                 # layer.Conv2d(shape[1], shape[1], kernel_size=1)
#                 Conv2d(shape[1], shape[1], kernel_size=1, device=x.device)
#             )
#         elif x.ndim == 2:
#             self.spread = layer.Linear(x.shape[1], x.shape[1])
#         else:
#             print('\nx.ndim=' + x.ndim)
#             raise NotImplementedError
#
#     @staticmethod
#     @torch.jit.script
#     def jit_neuronal_charge1(x: torch.Tensor, mem_old: torch.Tensor, decay: float, spike: torch.Tensor,
#                              fecs: torch.Tensor):
#         return mem_old * decay * (1 - spike.detach()) + x + fecs
#
#     @staticmethod
#     @torch.jit.script
#     def jit_neuronal_charge2(x: torch.Tensor, mem_old: torch.Tensor, decay: float, spike: torch.Tensor):
#         return mem_old * decay * (1 - spike.detach()) + x


# class mem_update(nn.Module):
#     def __init__(self, act=False, ecs_tau: float = 5., alpha: float = 0.75, beta: float = 0.25, ECS=False):
#         super(mem_update, self).__init__()
#         # self.actFun= torch.nn.LeakyReLU(0.2, inplace=False)
#         self.ECS = ECS
#         self.actFun = nn.SiLU()
#         # silu(x)=x*σ(x),where σ(x) is logistic sigmoid
#         # logistic sigmoid:sig(x)=1/(1+exp(-x))
#         self.act = act
#         # self.alpha = nn.Parameter(torch.tensor(alpha))
#         self.alpha = alpha
#         # self.beta = nn.Parameter(torch.tensor(beta))
#         self.beta =beta
#         self.ecs_tau = ecs_tau
#         self.spread = None
#
#     def forward(self, x):
#         mem = torch.zeros_like(x[0]).to(x.device)
#         # 该语句的主要作用是生成一个值全为0的、维度与输入尺寸相同的矩阵。
#         # torch.zeros_like(input)相当于torch.zeros(input.size(),dtype=input.dtype,layout=input.layout, device=input.device)
#         spike = torch.zeros_like(x[0]).to(x.device)
#         output = torch.zeros_like(x)
#         mem_old = 0
#         ecs = 0.
#         fecs = 0.
#         if self.spread is None:
#             self.InitEcsSpread(x[0])
#         for i in range(time_window):
#             # range(1)->[0]
#             if i >= 1:
#                 # .detach() 返回一个新的Variable，从当前计算图中分离下来的，但是仍指向原变量的存放位置,不同之处只是requires_grad为false，得到的这个Variable永远不需要计算其梯度，不具有grad。
#                 # 即使之后重新将它的requires_grad置为true,它也不会具有梯度grad。
#                 # mem = mem_old * decay * (1-spike.detach()) + x[i]
#                 mem = self.jit_neuronal_charge(x[i], mem_old, decay, spike, fecs)
#             else:
#                 mem = x[i] + fecs
#             if self.act:
#                 spike = self.actFun(mem)
#             else:
#                 spike = act_fun(mem)
#
#             ecs = self.alpha * self.spread(spike) + (1. - 1. / self.ecs_tau) * ecs
#             fecs = self.beta * torch.tanh(ecs)
#
#             mem_old = mem.clone()
#             output[i] = spike
#         # print(output[0][0][0][0])
#         return output
#
#     def InitEcsSpread(self, x: torch.Tensor):
#         if x.ndim == 4:
#             shape = x.shape
#             self.spread = nn.Sequential(
#                 # nn.Conv2d(shape[1], shape[1],
#                 #           kernel_size=3, padding=1, device=x.device, groups=shape[1]),
#                 # layer.Conv2d(shape[1], shape[1],
#                 #              kernel_size=3, padding=1, groups=shape[1]),
#                 Conv2d(shape[1], shape[1],
#                        kernel_size=3, padding=1, device=x.device, groups=shape[1]),
#                 # nn.Conv2d(shape[1], shape[1], kernel_size=1, device=x.device)
#                 # layer.Conv2d(shape[1], shape[1], kernel_size=1)
#                 Conv2d(shape[1], shape[1], kernel_size=1, device=x.device)
#             )
#         elif x.ndim == 2:
#             self.spread = layer.Linear(x.shape[1], x.shape[1])
#         else:
#             print('\nx.ndim=' + x.ndim)
#             raise NotImplementedError
#     @staticmethod
#     @torch.jit.script
#     def jit_neuronal_charge(x: torch.Tensor, mem_old: torch.Tensor, decay: float, spike: torch.Tensor,
#                              fecs: torch.Tensor):
#         return mem_old * decay * (1 - spike.detach()) + x + fecs

class DFL(nn.Module):
    # DFL module
    def __init__(self, c1=17):
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        self.conv.weight.data[:] = nn.Parameter(torch.arange(c1, dtype=torch.float).view(1, c1, 1, 1))  # / 120.0
        self.c1 = c1
        # self.bn = nn.BatchNorm2d(4)

    def forward(self, x):
        b, c, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
        # return self.conv(x.view(b, self.c1, 4, a).softmax(1)).view(b, 4, a)


class CBLinear(nn.Module):
    def __init__(self, c1, c2s, k=1, s=1, p=None, g=1):  # ch_in, ch_outs, kernel, stride, padding, groups
        super(CBLinear, self).__init__()
        self.c2s = c2s
        self.conv = nn.Sequential(
            mem_update(False),
            Snn_Conv2d(c1, sum(c2s), k, s, autopad(k, p), groups=g, bias=True))

    def forward(self, x):
        outs = self.conv(x).split(self.c2s, dim=2)
        return outs


class CBFuse(nn.Module):
    def __init__(self, idx):
        super(CBFuse, self).__init__()
        self.idx = idx

    def forward(self, xs):
        target_size = xs[-1].shape[2:]
        res = [F.interpolate(x[self.idx[i]], size=target_size, mode='nearest') for i, x in enumerate(xs[:-1])]
        out = torch.sum(torch.stack(res + xs[-1:]), dim=0)
        return out


class ConvBN(torch.nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, with_bn=True):
        super().__init__()
        self.add_module('conv', Snn_Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation, groups))
        if with_bn:
            self.add_module('bn', batch_norm_2d(out_planes))
            torch.nn.init.constant_(self.bn.bn.weight, 1)
            torch.nn.init.constant_(self.bn.bn.bias, 0)


class Conv(nn.Module):
    # Standard convolution 标准的卷积
    def __init__(self, c1, c2, k, s, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        # groups通道分组的参数，输入通道数、输出通道数必须同时满足被groups整除；
        super().__init__()
        self.conv = Snn_Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = batch_norm_2d(c2)
        self.act = mem_update(act=True)

    def forward(self, x):  # 前向传播函数
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):  # 去掉了bn层
        return self.act(self.conv(x))


class Conv_A(nn.Module):
    # Standard convolution 标准的卷积
    def __init__(self, c1, c2, k, s, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = Snn_Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = batch_norm_2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class Conv_B(nn.Module):
    # Standard convolution 标准的卷积
    def __init__(self, c1, c2, k, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        # groups通道分组的参数，输入通道数、输出通道数必须同时满足被groups整除；
        super().__init__()
        self.act = mem_update(act=False)
        self.conv = Snn_Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = batch_norm_2d(c2)

    def forward(self, x):  # 前向传播函数
        return self.bn(self.conv(self.act(x)))

    def forward_fuse(self, x):  # 去掉了bn层
        return self.conv(self.act(x))


class Conv_1(nn.Module):
    # Standard convolution 标准的卷积
    def __init__(self, c1, c2, k, s, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = Snn_Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        # self.conv = layer.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False, step_mode='m')
        # self.conv = DepthWiseConv(c1, c2, k, s, autopad(k, p), bias=False)
        self.bn = batch_norm_2d(c2)
        # self.bn = BatchNorm3d1(c2)
        # self.bn = layer.BatchNorm2d(c2, step_mode='m')
        # self.act = mem_update(act=False) if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.bn(self.conv(x))

    def forward_fuse(self, x):
        return self.conv(x)


class Conv_2(nn.Module):
    # Standard convolution 标准的卷积
    def __init__(self, c1, c2, k, s, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = Snn_Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = batch_norm_2d(c2)
        self.act = mem_update(act=False) if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.bn(self.conv(self.act(x)))

    def forward_fuse(self, x):
        return self.conv(self.act(x))


class Conv_3(nn.Module):
    # Standard convolution
    # init初始化构造函数
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups

        super().__init__()
        # 卷积层
        self.conv = Snn_Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        # self.conv = layer.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        # 归一化层
        self.bn = batch_norm_2d(c2)
        # self.bn = layer.BatchNorm2d(c2)
        # 激活函数
        self.act = mem_update(act=False) if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        # self.act = LIFNode(decay, 0) if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        # self.act = neuron.IFNode(surrogate_function=surrogate.ATan())
        # functional.set_step_mode(self, step_mode='m')

    # 正向计算，网络执行的顺序是根据forward函数来决定的
    def forward(self, x):
        # conv卷积 -> bn -> act激活
        # print("----------_________---------")
        # print(x.shape)
        # print(self.conv)
        # print(self.bn(self.conv(self.act(x))).shape)
        # print("----------_________---------")
        return self.bn(self.conv(self.act(x)))

    # 正向融合计算
    def forward_fuse(self, x):
        # 这里只有卷积和激活
        return self.conv(self.act(x))


class Conv_4(nn.Module):
    # Standard convolution
    # init初始化构造函数
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups

        super().__init__()
        # 卷积层
        self.conv = Snn_Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        # self.conv = layer.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        # 归一化层
        self.bn = batch_norm_2d1(c2)
        # self.bn = layer.BatchNorm2d(c2)
        # 激活函数
        self.act = mem_update(act=False) if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        # self.act = LIFNode(decay, 0) if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        # self.act = neuron.IFNode(surrogate_function=surrogate.ATan())
        # functional.set_step_mode(self, step_mode='m')

    # 正向计算，网络执行的顺序是根据forward函数来决定的
    def forward(self, x):
        # conv卷积 -> bn -> act激活
        # print("----------_________---------")
        # print(x.shape)
        # print(self.bn(self.conv(self.act(x))).shape)
        # print("----------_________---------")
        return self.bn(self.conv(self.act(x)))

    # 正向融合计算
    def forward_fuse(self, x):
        # 这里只有卷积和激活
        return self.conv(self.act(x))


class Conv_5(nn.Module):
    # Standard convolution
    # init初始化构造函数
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = DepthWiseConv(c1, c2, k, s, autopad(k, p), bias=False)
        self.bn = batch_norm_2d(c2)
        # self.bn = layer.BatchNorm2d(c2)
        # 激活函数
        self.act = mem_update(act=False) if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        # conv卷积 -> bn -> act激活
        # print("----------_________---------")
        # print(x.shape)
        # print(self.bn(self.conv(self.act(x))).shape)
        # print("----------_________---------")
        return self.bn(self.conv(self.act(x)))


class Conv_6(nn.Module):
    # Standard convolution
    # init初始化构造函数
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = DepthWiseConv(c1, c2, k, s, autopad(k, p), bias=False)
        self.bn = batch_norm_2d1(c2)
        # self.bn = layer.BatchNorm2d(c2)
        # 激活函数
        self.act = mem_update(act=False) if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        # conv卷积 -> bn -> act激活
        # print("----------_________---------")
        # print(x.shape)
        # print(self.bn(self.conv(self.act(x))).shape)
        # print("----------_________---------")
        return self.bn(self.conv(self.act(x)))

# class Snn_Conv2d(nn.Conv2d):
#     def __init__(self, in_channels, out_channels, kernel_size, stride=1,
#                  padding=0, dilation=1, groups=1,
#                  bias=True, padding_mode='zeros', marker='b'):
#         # in_channels参数代表输入特征矩阵的深度即channel，比如输入一张RGB彩色图像，那in_channels = 3；
#         # out_channels参数代表卷积核的个数，使用n个卷积核输出的特征矩阵深度即channel就是n；
#         # kernel_size参数代表卷积核的尺寸，输入可以是int类型如3 代表卷积核的height = width = 3，也可以是tuple类型如(3,  5)代表卷积核的height = 3，width = 5；
#         # stride参数代表卷积核的步距默认为1，和kernel_size一样输入可以是int类型，也可以是tuple类型，这里注意，若为tuple类型即第一个int用于高度尺寸，第二个int用于宽度尺寸；
#         # padding参数代表在输入特征矩阵四周补零的情况默认为0，同样输入可以为int型如1 代表上下方向各补一行0元素，左右方向各补一列0像素（即补一圈0）
#         # ，如果输入为tuple型如(2, 1) 代表在上方补两行下方补两行，左边补一列，右边补一列。可见下图，padding[0]是在H高度方向两侧填充的，padding[1]是在W宽度方向两侧填充的；
#         # dilation参数代表每个点之间有空隙的过滤器。例如，在一个维度上，一个大小为3的过滤器w会对输入的x进行如下计算：w[0] * x[0] + w[1] * x[1] + w[2] * x[2]。
#         # 若dilation = 1，过滤器会计算：w[0] * x[0] + w[1] * x[2] + w[2] * x[4]；换句话说，在不同点之间有一个1的差距。
#         super(Snn_Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups,
#                                          bias, padding_mode)
#         self.marker = marker
#
#     def forward(self, input):
#         weight = self.weight  #
#         # print(self.padding[0],'=======')
#         # 输入数据的维度应该是 (time_window,batch_size, input_channels, height, width)
#         h = (input.size()[3] - self.kernel_size[0] + 2 * self.padding[0]) // self.stride[0] + 1
#         w = (input.size()[4] - self.kernel_size[0] + 2 * self.padding[0]) // self.stride[0] + 1
#         # torch.zeros()函数,返回一个形状为size,类型为torch.dtype，里面的每一个值都是0的tensor
#         c1 = torch.zeros(time_window, input.size()[1], self.out_channels, h, w, device=input.device, dtype=input.dtype)
#         # print(weight.size(),'=====weight====')
#         for i in range(time_window):
#             c1[i] = F.conv2d(input[i], weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
#         return c1

class Snn_Conv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros', marker='b'):
        # in_channels参数代表输入特征矩阵的深度即channel，比如输入一张RGB彩色图像，那in_channels = 3；
        # out_channels参数代表卷积核的个数，使用n个卷积核输出的特征矩阵深度即channel就是n；
        # kernel_size参数代表卷积核的尺寸，输入可以是int类型如3 代表卷积核的height = width = 3，也可以是tuple类型如(3,  5)代表卷积核的height = 3，width = 5；
        # stride参数代表卷积核的步距默认为1，和kernel_size一样输入可以是int类型，也可以是tuple类型，这里注意，若为tuple类型即第一个int用于高度尺寸，第二个int用于宽度尺寸；
        # padding参数代表在输入特征矩阵四周补零的情况默认为0，同样输入可以为int型如1 代表上下方向各补一行0元素，左右方向各补一列0像素（即补一圈0）
        # ，如果输入为tuple型如(2, 1) 代表在上方补两行下方补两行，左边补一列，右边补一列。可见下图，padding[0]是在H高度方向两侧填充的，padding[1]是在W宽度方向两侧填充的；
        # dilation参数代表每个点之间有空隙的过滤器。例如，在一个维度上，一个大小为3的过滤器w会对输入的x进行如下计算：w[0] * x[0] + w[1] * x[1] + w[2] * x[2]。
        # 若dilation = 1，过滤器会计算：w[0] * x[0] + w[1] * x[2] + w[2] * x[4]；换句话说，在不同点之间有一个1的差距。
        super(Snn_Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups,
                                         bias, padding_mode)
        self.marker = marker

    def forward(self, input: Tensor) -> Tensor:
        # print(self.padding[0],'=======')
        # 输入数据的维度应该是 (time_window,batch_size, input_channels, height, width)
        # h = (input.size()[3] - self.kernel_size[0] + 2 * self.padding[0]) // self.stride[0] + 1
        # w = (input.size()[4] - self.kernel_size[1] + 2 * self.padding[1]) // self.stride[1] + 1
        h = (input.size()[3] - self.dilation[0] * (self.kernel_size[0] - 1) + 2 * self.padding[0] - 1) // self.stride[
            0] + 1
        w = (input.size()[4] - self.dilation[1] * (self.kernel_size[1] - 1) + 2 * self.padding[1] - 1) // self.stride[
            1] + 1
        # torch.zeros()函数,返回一个形状为size,类型为torch.dtype，里面的每一个值都是0的tensor
        c1 = torch.zeros(time_window, input.size()[1], self.out_channels, h, w, device=input.device, dtype=input.dtype)
        # print(weight.size(),'=====weight====')
        for i in range(time_window):
            # print(c1[i].shape)
            c1[i] = self._conv_forward(input[i], self.weight, self.bias)
        return c1


class Conv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros', marker='b', device=None):
        # in_channels参数代表输入特征矩阵的深度即channel，比如输入一张RGB彩色图像，那in_channels = 3；
        # out_channels参数代表卷积核的个数，使用n个卷积核输出的特征矩阵深度即channel就是n；
        # kernel_size参数代表卷积核的尺寸，输入可以是int类型如3 代表卷积核的height = width = 3，也可以是tuple类型如(3,  5)代表卷积核的height = 3，width = 5；
        # stride参数代表卷积核的步距默认为1，和kernel_size一样输入可以是int类型，也可以是tuple类型，这里注意，若为tuple类型即第一个int用于高度尺寸，第二个int用于宽度尺寸；
        # padding参数代表在输入特征矩阵四周补零的情况默认为0，同样输入可以为int型如1 代表上下方向各补一行0元素，左右方向各补一列0像素（即补一圈0）
        # ，如果输入为tuple型如(2, 1) 代表在上方补两行下方补两行，左边补一列，右边补一列。可见下图，padding[0]是在H高度方向两侧填充的，padding[1]是在W宽度方向两侧填充的；
        # dilation参数代表每个点之间有空隙的过滤器。例如，在一个维度上，一个大小为3的过滤器w会对输入的x进行如下计算：w[0] * x[0] + w[1] * x[1] + w[2] * x[2]。
        # 若dilation = 1，过滤器会计算：w[0] * x[0] + w[1] * x[2] + w[2] * x[4]；换句话说，在不同点之间有一个1的差距。
        super(Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups,
                                     bias, padding_mode, device)
        self.marker = marker

    def forward(self, input: Tensor) -> Tensor:
        # print(weight.size(),'=====weight====')
        return self._conv_forward(input, self.weight, self.bias)


class batch_norm_2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(batch_norm_2d, self).__init__()  # num_features=16
        self.bn = BatchNorm3d1(
            num_features)  # input (N,C,D,H,W) imension batch norm on (N,D,H,W) slice. spatio-temporal Batch Normalization

    def forward(self, input):
        y = input.transpose(0, 2).contiguous().transpose(0, 1).contiguous()
        # transpose函数对多维数组进行转置操作
        # contiguous()这个函数，把tensor变成在内存中连续分布的形式。
        y = self.bn(y)
        return y.contiguous().transpose(0, 1).contiguous().transpose(0, 2)  # 


class batch_norm_2d1(nn.Module):

    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(batch_norm_2d1, self).__init__()
        self.bn = BatchNorm3d2(num_features)

    def forward(self, input):
        y = input.transpose(0, 2).contiguous().transpose(0, 1).contiguous()
        y = self.bn(y)
        return y.contiguous().transpose(0, 1).contiguous().transpose(0, 2)


class BatchNorm3d1(torch.nn.BatchNorm3d):  # 5
    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            nn.init.constant_(self.weight, thresh)
            # torch.nn.init.constant_(tensor, val),基于输入参数（val）初始化输入张量tensor，即tensor的值均初始化为val。
            nn.init.zeros_(self.bias)
            # torch.nn.init.zeros_(tensor),tensor的值均初始化为0。

    # def forward(self, input: Tensor) -> Tensor:
    #     input = input.transpose(0, 2).contiguous().transpose(0, 1).contiguous()
    #     self._check_input_dim(input)
    #
    #     # exponential_average_factor is set to self.momentum
    #     # (when it is available) only so that it gets updated
    #     # in ONNX graph when this node is exported to ONNX.
    #     if self.momentum is None:
    #         exponential_average_factor = 0.0
    #     else:
    #         exponential_average_factor = self.momentum
    #
    #     if self.training and self.track_running_stats:
    #         # TODO: if statement only here to tell the jit to skip emitting this when it is None
    #         if self.num_batches_tracked is not None:  # type: ignore[has-type]
    #             self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    #             if self.momentum is None:  # use cumulative moving average
    #                 exponential_average_factor = 1.0 / float(self.num_batches_tracked)
    #             else:  # use exponential moving average
    #                 exponential_average_factor = self.momentum
    #
    #     r"""
    #     Decide whether the mini-batch stats should be used for normalization rather than the buffers.
    #     Mini-batch stats are used in training mode, and in eval mode when buffers are None.
    #     """
    #     if self.training:
    #         bn_training = True
    #     else:
    #         bn_training = (self.running_mean is None) and (self.running_var is None)
    #
    #     r"""
    #     Buffers are only updated if they are to be tracked and we are in training mode. Thus they only need to be
    #     passed when the update should occur (i.e. in training mode when they are tracked), or when buffer stats are
    #     used for normalization (i.e. in eval mode when buffers are not None).
    #     """
    #     return F.batch_norm(
    #         input,
    #         # If buffers are not to be tracked, ensure that they won't be updated
    #         self.running_mean
    #         if not self.training or self.track_running_stats
    #         else None,
    #         self.running_var if not self.training or self.track_running_stats else None,
    #         self.weight,
    #         self.bias,
    #         bn_training,
    #         exponential_average_factor,
    #         self.eps,
    #     ).contiguous().transpose(0, 1).contiguous().transpose(0, 2)


class BatchNorm3d2(torch.nn.BatchNorm3d):
    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            nn.init.constant_(self.weight, 0.2 * thresh)
            nn.init.zeros_(self.bias)

    # def forward(self, input: Tensor) -> Tensor:
    #     input = input.transpose(0, 2).contiguous().transpose(0, 1).contiguous()
    #     self._check_input_dim(input)
    #
    #     # exponential_average_factor is set to self.momentum
    #     # (when it is available) only so that it gets updated
    #     # in ONNX graph when this node is exported to ONNX.
    #     if self.momentum is None:
    #         exponential_average_factor = 0.0
    #     else:
    #         exponential_average_factor = self.momentum
    #
    #     if self.training and self.track_running_stats:
    #         # TODO: if statement only here to tell the jit to skip emitting this when it is None
    #         if self.num_batches_tracked is not None:  # type: ignore[has-type]
    #             self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    #             if self.momentum is None:  # use cumulative moving average
    #                 exponential_average_factor = 1.0 / float(self.num_batches_tracked)
    #             else:  # use exponential moving average
    #                 exponential_average_factor = self.momentum
    #
    #     r"""
    #     Decide whether the mini-batch stats should be used for normalization rather than the buffers.
    #     Mini-batch stats are used in training mode, and in eval mode when buffers are None.
    #     """
    #     if self.training:
    #         bn_training = True
    #     else:
    #         bn_training = (self.running_mean is None) and (self.running_var is None)
    #
    #     r"""
    #     Buffers are only updated if they are to be tracked and we are in training mode. Thus they only need to be
    #     passed when the update should occur (i.e. in training mode when they are tracked), or when buffer stats are
    #     used for normalization (i.e. in eval mode when buffers are not None).
    #     """
    #     return F.batch_norm(
    #         input,
    #         # If buffers are not to be tracked, ensure that they won't be updated
    #         self.running_mean
    #         if not self.training or self.track_running_stats
    #         else None,
    #         self.running_var if not self.training or self.track_running_stats else None,
    #         self.weight,
    #         self.bias,
    #         bn_training,
    #         exponential_average_factor,
    #         self.eps,
    #     ).contiguous().transpose(0, 1).contiguous().transpose(0, 2)


class Pools(nn.Module):
    def __init__(self, kernel_size, stride, padding=0, dilation=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.pool = nn.MaxPool2d(kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)

    def forward(self, input):
        h = int((input.size()[3] + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1) / self.stride + 1)
        w = int((input.size()[4] + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1) / self.stride + 1)
        c1 = torch.zeros(time_window, input.size()[1], input.size()[2], h, w, device=input.device)
        for i in range(time_window):
            c1[i] = self.pool(input[i])
        return c1


class zeropad(nn.Module):
    def __init__(self, padding):
        super().__init__()
        self.padding = padding
        self.pad = nn.ZeroPad2d(padding=self.padding)
        # nn.ZeroPad2d，对Tensor使用0进行边界填充

    def forward(self, input):
        h = input.size()[3] + self.padding[2] + self.padding[3]
        w = input.size()[4] + self.padding[0] + self.padding[1]
        c1 = torch.zeros(time_window, input.size()[1], input.size()[2], h, w, device=input.device)
        for i in range(time_window):
            c1[i] = self.pad(input[i])
        return c1


class Sample(nn.Module):
    def __init__(self, size=None, scale_factor=None, mode='nearset'):
        super(Sample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.size = size
        self.up = nn.Upsample(self.size, self.scale_factor, mode=self.mode)
        # size:指定输出的尺寸大小
        # scale_factor:指定输出的尺寸是输入尺寸的倍数
        # mode:上采样的算法可选 ‘nearest’, ‘linear’, ‘bilinear’, ‘bicubic’，‘trilinear’. 默认: ‘nearest’
        # align_corners和recomputer_scale_factor:几乎用不到

    def forward(self, input):
        # self.cpu()
        # temp = torch.zeros(time_window, input.size()[1], input.size()[2], input.size()[3] * self.scale_factor,
        #                    input.size()[4] * self.scale_factor, device=input.device, dtype=input.dtype)
        temp = torch.zeros(input.size()[0], input.size()[1], input.size()[2], input.size()[3] * self.scale_factor,
                           input.size()[4] * self.scale_factor, device=input.device, dtype=input.dtype)
        # print(temp.device,'-----')
        # for i in range(time_window):
        for i in range(input.size()[0]):
            temp[i] = self.up(input[i])

            # temp[i]= F.interpolate(input[i], scale_factor=self.scale_factor,mode='nearest')
        return temp


# class upSample(nn.Upsample):
#     def forward(self, input: Tensor) -> Tensor:
#         self.scale_factor = int(self.scale_factor)
#         temp = torch.zeros(input.size()[0], input.size()[1], input.size()[2], input.size()[3] * self.scale_factor,
#                            input.size()[4] * self.scale_factor, device=input.device, dtype=input.dtype)
#         for i in range(input.size()[0]):
#             temp[i] = F.interpolate(input[i], self.size, self.scale_factor, self.mode, self.align_corners,
#                                     recompute_scale_factor=self.recompute_scale_factor)
#         return temp


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel=3, stride=1, e=0.5):
        super().__init__()
        c_ = int(out_channels * e)

        self.cv1 = Conv(in_channels, c_, k=kernel, s=stride)
        self.cv2 = Conv(c_, out_channels, 3, 1)
        # self.shortcut=Conv_2(in_channels,out_channels,k=1,s=stride)
        self.shortcut = nn.Sequential(
        )
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                Snn_Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                batch_norm_2d(out_channels),
            )

    def forward(self, x):
        return (self.cv2(self.cv1(x)) + self.shortcut(x))


class Bottleneck_2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel=3, stride=1, e=4):
        super(Bottleneck_2, self).__init__()
        p = None
        if kernel == 3:
            pad = 1
        if kernel == 1:
            pad = 0
        width = int(out_channels * e)
        self.residual_function = nn.Sequential(
            mem_update(act=False),
            Snn_Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            batch_norm_2d1(out_channels),
            mem_update(act=False),
            Snn_Conv2d(out_channels, out_channels, kernel_size=kernel, stride=stride, padding=pad, bias=False),
            batch_norm_2d(out_channels),
            mem_update(act=False),
            Snn_Conv2d(out_channels, width, kernel_size=1, stride=1, bias=False),
            batch_norm_2d1(width),
        )
        # self.shortcut=Conv_2(in_channels,out_channels,k=1,s=stride)
        self.shortcut = nn.Sequential(
        )
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.MaxPool3d((1, stride, stride), stride=(1, stride, stride)),
                mem_update(act=False),
                # EcsLifNode(v_threshold=thresh, step_mode='m'),
                Snn_Conv2d(in_channels, width, kernel_size=1, stride=1, bias=False),
                batch_norm_2d(width),
            )

    def forward(self, x):
        # print(self.residual_function(x).shape)
        # print(self.shortcut(x).shape)
        return (self.residual_function(x) + self.shortcut(x))


class Bottleneck_3(nn.Module):
    def __init__(self, in_channels, out_channels, kernel=3, stride=1, e=4):
        super(Bottleneck_3, self).__init__()
        p = None
        if kernel == 3:
            pad = 1
        if kernel == 1:
            pad = 0
        width = int(in_channels * e)
        self.residual_function = nn.Sequential(
            mem_update(act=False),
            Snn_Conv2d(in_channels, width, kernel_size=1, stride=1, bias=False),
            batch_norm_2d1(width),
            mem_update(act=False),
            Snn_Conv2d(width, width, kernel_size=kernel, stride=stride, padding=pad, bias=False, groups=width),
            batch_norm_2d(width),
            mem_update(act=False),
            Snn_Conv2d(width, out_channels, kernel_size=1, stride=1, bias=False),
            batch_norm_2d1(out_channels),
        )
        # self.shortcut=Conv_2(in_channels,out_channels,k=1,s=stride)
        self.shortcut = nn.Sequential(
        )
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.MaxPool3d((1, stride, stride), stride=(1, stride, stride)),
                mem_update(act=False),
                # EcsLifNode(v_threshold=thresh, step_mode='m'),
                Snn_Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                batch_norm_2d(out_channels),
            )

    def forward(self, x):
        # print(self.residual_function(x).shape)
        # print(self.shortcut(x).shape)
        return (self.residual_function(x) + self.shortcut(x))


class BasicBlock_1(nn.Module):  #
    def __init__(self, in_channels, out_channels, stride=1, e=0.5):
        super().__init__()
        # c_ = int(out_channels * e)  # hidden channels  
        c_ = 1024
        self.residual_function = nn.Sequential(
            mem_update(act=False),
            Snn_Conv2d(in_channels, c_, kernel_size=3, stride=stride, padding=1, bias=False),
            batch_norm_2d(c_),
            mem_update(act=False),
            Snn_Conv2d(c_, out_channels, kernel_size=3, padding=1, bias=False),
            batch_norm_2d1(out_channels),
        )
        # shortcut
        self.shortcut = nn.Sequential(
        )
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.MaxPool3d((1, stride, stride), stride=(1, stride, stride)),
                mem_update(act=False),
                Snn_Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                batch_norm_2d(out_channels),
            )

    def forward(self, x):
        # print("BasicBlock_1")
        # print(x.dtype)
        # print(self.residual_function(x).dtype)
        # print("BasicBlock_1--end")
        return (self.residual_function(x) + self.shortcut(x))


class BasicBlock_1n(nn.Module):  #
    def __init__(self, in_channels, out_channels, stride=1, e=0.5):
        super().__init__()
        # c_ = int(out_channels * e)  # hidden channels
        c_ = 1024
        self.residual_function = nn.Sequential(
            # MHSA(in_channels, out_channels, num_heads=8),
            # nn.AvgPool3d(2, 2),
            batch_norm_2d(in_channels),
            mem_update(act=False),
            Snn_Conv2d(in_channels, c_, kernel_size=3, stride=stride, padding=1, bias=False),
            # batch_norm_2d(out_channels),
            # OSRAAttention(out_channels, num_heads=4),
            batch_norm_2d1(c_),
            mem_update(act=False),
            Snn_Conv2d(c_, out_channels, kernel_size=3, padding=1, bias=False),
            # batch_norm_2d1(out_channels),
        )
        # shortcut
        self.shortcut = nn.Sequential(
        )
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.MaxPool3d((1, stride, stride), stride=(1, stride, stride)),
                batch_norm_2d(in_channels),
                mem_update(act=False),
                Snn_Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                # batch_norm_2d(out_channels),
            )

    def forward(self, x):
        # print(self.residual_function(x).shape)
        return (self.residual_function(x) + self.shortcut(x))


class BasicBlock_1m(nn.Module):  #
    def __init__(self, in_channels, out_channels, stride=1, e=0.5):
        super().__init__()
        # c_ = int(out_channels * e)  # hidden channels
        c_ = 1024
        self.residual_function = nn.Sequential(
            RepConv(in_channels, c_, k=3, s=stride, p=1),
            mem_update(act=False),
            Snn_Conv2d(c_, out_channels, kernel_size=3, padding=1, bias=False),
            batch_norm_2d1(out_channels),
        )
        # shortcut
        self.shortcut = nn.Sequential(
        )
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.MaxPool3d((1, stride, stride), stride=(1, stride, stride)),
                mem_update(act=False),
                Snn_Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                batch_norm_2d(out_channels),
            )

    def forward(self, x):
        # print(self.residual_function(x).shape)
        return (self.residual_function(x) + self.shortcut(x))


class BasicBlock_1s(nn.Module):  #
    def __init__(self, in_channels, out_channels, stride=1, e=0.5):
        super().__init__()
        # c_ = int(out_channels * e)  # hidden channels
        c_ = 1024
        # self.act1 = neuron.IFNode(surrogate_function=surrogate.ATan())
        # self.act1 = mem_update(act=False)
        # self.conv1 = layer.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv1 = GSConv(in_channels, c_, k=3, s=stride)
        # self.conv1 = DualConv(in_channels, out_channels, stride=stride, g=4)
        # self.bn1 = layer.BatchNorm2d(out_channels)
        # self.bn1 = batch_norm_2d(out_channels)
        # self.act2 = mem_update(act=False)
        # self.conv2 = layer.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.conv2 = GSConv(c_, out_channels, k=3, s=1)
        # self.conv2 = DualConv(out_channels, out_channels, stride=1, g=4)
        # self.bn2 = layer.BatchNorm2d(out_channels)
        # self.bn2 = batch_norm_2d1(out_channels)

        self.shortcut = nn.Sequential(
        )

        if stride != 1 or in_channels != out_channels:
            self.shortcut = shortcut(in_channels, out_channels, stride)

        # functional.set_step_mode(self, step_mode='m')

    def forward(self, x):
        x1 = self.shortcut(x)
        # x = self.act1(x)
        x = self.conv1(x)
        # x = self.bn1(x)
        # x = self.act2(x)
        x = self.conv2(x)
        # x = self.bn2(x)
        return x + x1


class BasicBlock_2(nn.Module):
    def __init__(self, in_channels, out_channels, k_size=3, stride=1, add=True):
        super().__init__()
        p = None
        self.add = add
        if k_size == 3:
            pad = 1
        if k_size == 1:
            pad = 0
        self.residual_function = nn.Sequential(
            mem_update(act=False),
            # RFAConv(in_channels, out_channels, k_size, stride),
            Snn_Conv2d(in_channels, out_channels, kernel_size=k_size, stride=stride, padding=pad, bias=False),
            # DualConv(in_channels, out_channels, stride=stride, g=8),
            batch_norm_2d(out_channels),
            mem_update(act=False),
            # RFAConv(out_channels, out_channels, k_size, 1),
            Snn_Conv2d(out_channels, out_channels, kernel_size=k_size, padding=pad, bias=False),
            # DualConv(out_channels, out_channels, stride=1, g=8),
            batch_norm_2d1(out_channels),
        )

        self.shortcut = nn.Sequential(
        )

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.MaxPool3d((1, stride, stride), stride=(1, stride, stride)),
                mem_update(act=False),
                Snn_Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                batch_norm_2d(out_channels),
            )

    def forward(self, x):
        # print(self.residual_function(x).shape)
        # print(self.shortcut(x).shape)
        return (self.residual_function(x) + self.shortcut(x)) if self.add else self.residual_function(x)


class BasicBlock_3(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, k_size=3, stride=1):
        super().__init__()
        p = None
        if k_size == 3:
            pad = 1
        if k_size == 1:
            pad = 0
        self.residual_function = nn.Sequential(
            # MHSA(in_channels, out_channels, num_heads=8),
            # nn.AvgPool3d(2, 2),
            batch_norm_2d(in_channels),
            mem_update(act=False),
            Snn_Conv2d(in_channels, out_channels, kernel_size=k_size, stride=stride, padding=pad, bias=False),
            # batch_norm_2d(out_channels),
            # OSRAAttention(out_channels, num_heads=4),
            batch_norm_2d1(out_channels),
            mem_update(act=False),
            Snn_Conv2d(out_channels, out_channels, kernel_size=k_size, padding=pad, bias=False),
            # batch_norm_2d1(out_channels),
        )
        # self.channel = EfficientChannelAttention(out_channels)
        self.shortcut = nn.Sequential(
        )
        # self.SE = SELayer(out_channels)  # Squeeze-and-Excitation block
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.MaxPool3d((1, stride, stride), stride=(1, stride, stride)),
                batch_norm_2d(in_channels),
                mem_update(act=False),
                Snn_Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                # batch_norm_2d(out_channels),
            )

    def forward(self, x):
        # print(x.shape)
        # print(self.residual_function(x).shape)
        # print(self.shortcut(x).shape)
        # out = self.residual_function(x)
        # SE_out = self.SE(out)
        return (self.residual_function(x) + self.shortcut(x))


class BasicBlock_4(nn.Module):
    def __init__(self, in_channels, out_channels, k_size=3, stride=1):
        super().__init__()
        p = None
        if k_size == 3:
            pad = 1
        if k_size == 1:
            pad = 0
        self.residual_function = nn.Sequential(
            RepConv(in_channels, out_channels, k=k_size, s=stride, p=pad),
            mem_update(act=False),
            Snn_Conv2d(out_channels, out_channels, kernel_size=k_size, padding=pad, bias=False),
            batch_norm_2d1(out_channels),
        )

        self.shortcut = nn.Sequential(
        )

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.MaxPool3d((1, stride, stride), stride=(1, stride, stride)),
                mem_update(act=False),
                Snn_Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                batch_norm_2d(out_channels),
            )

    def forward(self, x):
        # print(x.shape)
        # print(self.residual_function(x).shape)
        # print(self.shortcut(x).shape)
        return (self.residual_function(x) + self.shortcut(x))


class BasicBlock_5(nn.Module):
    def __init__(self, in_channels, out_channels, k_size=3, stride=1, dilation=1):
        super().__init__()
        # c_ = int(in_channels * e)
        p = None
        if k_size == 3:
            pad = 1
        if k_size == 1:
            pad = 0
        self.residual_function = nn.Sequential(
            # mem_update(act=False),
            # PartialConv(in_channels, dilation=dilation),
            # Snn_Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=False),
            # batch_norm_2d(in_channels),
            mem_update(act=False),
            # PartialConv(in_channels),
            # AKConv(in_channels, out_channels, num_param=k_size, stride=stride),
            Snn_Conv2d(in_channels, out_channels, kernel_size=k_size, stride=stride, padding=pad, bias=False),
            batch_norm_2d(out_channels),
            mem_update(act=False),
            # Snn_Conv2d(in_channels, out_channels, kernel_size=k_size, stride=stride, padding=pad, bias=False),
            # AKConv(out_channels, out_channels, num_param=k_size, stride=1),
            # mem_update(act=False),
            # # Snn_Conv2d(out_channels, out_channels, 1, 1, bias=False),
            # ScConv(out_channels),
            PartialConv(out_channels, dilation=dilation),
            # Snn_Conv2d(out_channels, out_channels, 1, 1, bias=False),
            batch_norm_2d1(out_channels),
        )

        self.shortcut = nn.Sequential(
        )

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.MaxPool3d((1, stride, stride), stride=(1, stride, stride)),
                mem_update(act=False),
                Snn_Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                batch_norm_2d(out_channels),
            )

    def forward(self, x):
        # print("BasicBlock_2")
        # print(x.dtype)
        # print(self.residual_function(x).dtype)
        # print(self.shortcut(x).dtype)
        # print("BasicBlock_2--end")
        return (self.residual_function(x) + self.shortcut(x))


class BasicBlock_6(nn.Module):
    def __init__(self, in_channels, out_channels, k_size=3, stride=1, e=2):
        super().__init__()
        c_ = int(in_channels * e)
        p = None
        if k_size == 3:
            pad = 1
        if k_size == 1:
            pad = 0
        # self.act1 = neuron.IFNode(surrogate_function=surrogate.ATan())
        # self.act1 = mem_update(act=False)
        # self.conv1 = layer.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv1 = GSConv(in_channels, out_channels, k=k_size, s=stride)
        # self.bn1 = layer.BatchNorm2d(out_channels)
        # self.bn1 = batch_norm_2d(out_channels)
        # self.act2 = mem_update(act=False)
        # self.conv2 = layer.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.conv2 = GSConv(out_channels, out_channels, k=k_size)
        # self.bn2 = layer.BatchNorm2d(out_channels)
        # self.bn2 = batch_norm_2d1(out_channels)

        self.shortcut = nn.Sequential(
        )

        if stride != 1 or in_channels != out_channels:
            self.shortcut = shortcut(in_channels, out_channels, stride)

        # functional.set_step_mode(self, step_mode='m')

    def forward(self, x):
        # print("你好")
        # print(x.shape)
        x1 = self.shortcut(x)
        # x = self.act1(x)
        x = self.conv1(x)
        # x = self.bn1(x)
        # x = self.act2(x)
        x = self.conv2(x)
        # x = self.bn2(x)
        return x + x1


class BasicBlock_6(nn.Module):
    def __init__(self, in_channels, out_channels, k_size=3, stride=1, e=2):
        super().__init__()
        c_ = int(in_channels * e)
        p = None
        if k_size == 3:
            pad = 1
        if k_size == 1:
            pad = 0
        # self.act1 = neuron.IFNode(surrogate_function=surrogate.ATan())
        # self.act1 = mem_update(act=False)
        # self.conv1 = layer.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv1 = GSConv(in_channels, out_channels, k=k_size, s=stride)
        # self.bn1 = layer.BatchNorm2d(out_channels)
        # self.bn1 = batch_norm_2d(out_channels)
        # self.act2 = mem_update(act=False)
        # self.conv2 = layer.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.conv2 = GSConv(out_channels, out_channels, k=k_size)
        # self.bn2 = layer.BatchNorm2d(out_channels)
        # self.bn2 = batch_norm_2d1(out_channels)

        self.shortcut = nn.Sequential(
        )

        if stride != 1 or in_channels != out_channels:
            self.shortcut = shortcut(in_channels, out_channels, stride)

        # functional.set_step_mode(self, step_mode='m')

    def forward(self, x):
        # print("你好")
        # print(x.shape)
        x1 = self.shortcut(x)
        # x = self.act1(x)
        x = self.conv1(x)
        # x = self.bn1(x)
        # x = self.act2(x)
        x = self.conv2(x)
        # x = self.bn2(x)
        return x + x1


class shortcut(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.max = nn.MaxPool3d((1, stride, stride), stride=(1, stride, stride))
        # self.max = layer.MaxPool2d(stride, stride)
        # self.m = neuron.IFNode(surrogate_function=surrogate.ATan())
        self.m = mem_update(act=False)
        # self.conv = layer.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv = Snn_Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        # self.bn = layer.BatchNorm2d(out_channels)
        self.bn = batch_norm_2d(out_channels)

    def forward(self, x):
        x = self.max(x)
        x = self.m(x)
        x = self.conv(x)
        out = self.bn(x)
        return out


class Concat_res2(nn.Module):  #
    def __init__(self, in_channels, out_channels, k_size=3, stride=1, e=0.5):
        super().__init__()
        c_ = int(out_channels * e)  # hidden channels
        if k_size == 3:
            pad = 1
        if k_size == 1:
            pad = 0
        self.residual_function = nn.Sequential(
            mem_update(act=False),
            # RFAConv(in_channels, out_channels, k_size, stride),
            Snn_Conv2d(in_channels, out_channels, kernel_size=k_size, stride=stride, padding=pad, bias=False),
            batch_norm_2d(out_channels),
            mem_update(act=False),
            # RFAConv(out_channels, out_channels, k_size, 1),
            Snn_Conv2d(out_channels, out_channels, kernel_size=k_size, padding=pad, bias=False),
            batch_norm_2d1(out_channels),
        )
        # shortcut
        self.shortcut = nn.Sequential(
        )
        if in_channels < out_channels:
            self.shortcut = nn.Sequential(
                mem_update(act=False),
                Snn_Conv2d(in_channels, out_channels - in_channels, kernel_size=1, stride=1, bias=False),
                batch_norm_2d(out_channels - in_channels),
            )
        self.pools = nn.MaxPool3d((1, stride, stride), stride=(1, stride, stride))

    def forward(self, x):
        temp = self.shortcut(x)
        out = torch.cat((temp, x), dim=2)
        out = self.pools(out)
        return (self.residual_function(x) + out)


class Concat_res3(nn.Module):  #
    expansion = 1

    def __init__(self, in_channels, out_channels, k_size=3, stride=1, e=0.5):
        super().__init__()
        c_ = int(out_channels * e)  # hidden channels
        if k_size == 3:
            pad = 1
        if k_size == 1:
            pad = 0
        self.residual_function = nn.Sequential(
            mem_update(act=False),
            Snn_Conv2d(in_channels, out_channels, kernel_size=k_size, stride=stride, padding=pad, bias=False),
            batch_norm_2d(out_channels),
            mem_update(act=False),
            Snn_Conv2d(out_channels, out_channels, kernel_size=k_size, padding=pad, bias=False),
            batch_norm_2d1(out_channels),
        )
        # shortcut
        self.shortcut = nn.Sequential(
        )
        # self.channel = EfficientChannelAttention(out_channels)
        self.SE = SELayer(out_channels)  # Squeeze-and-Excitation block
        if in_channels < out_channels:
            self.shortcut = nn.Sequential(
                mem_update(act=False),
                Snn_Conv2d(in_channels, out_channels - in_channels, kernel_size=1, stride=1, bias=False),
                batch_norm_2d(out_channels - in_channels),
            )
        self.pools = nn.MaxPool3d((1, stride, stride), stride=(1, stride, stride))

    def forward(self, x):
        # print(x.shape)
        # print(self.residual_function(x).shape)
        temp = self.shortcut(x)
        out = torch.cat((temp, x), dim=2)
        out = self.pools(out)
        # print(self.residual_function(x).shape)
        # print(out.shape)
        out1 = self.residual_function(x)
        SE_out = self.SE(out1)
        return (out1 * SE_out + out)


class Concat_res4(nn.Module):  #
    def __init__(self, in_channels, out_channels, k_size=3, stride=1, e=0.5, resolution=(20, 20)):
        super().__init__()
        c_ = int(out_channels * e)  # hidden channels
        if k_size == 3:
            pad = 1
        if k_size == 1:
            pad = 0
        self.residual_function = nn.Sequential(
            # MHSA(in_channels, out_channels, num_heads=8),
            # nn.AvgPool3d(2, 2),
            batch_norm_2d(in_channels),
            mem_update(act=False),
            Snn_Conv2d(in_channels, out_channels, kernel_size=k_size, stride=stride, padding=pad, bias=False),
            # batch_norm_2d(out_channels),
            # OSRAAttention(out_channels, num_heads=4),
            batch_norm_2d1(out_channels),
            mem_update(act=False),
            Snn_Conv2d(out_channels, out_channels, kernel_size=k_size, padding=pad, bias=False),
            # batch_norm_2d1(out_channels),
        )
        # shortcut
        self.shortcut = nn.Sequential(
        )

        if in_channels < out_channels:
            self.shortcut = nn.Sequential(
                batch_norm_2d(in_channels),
                mem_update(act=False),
                Snn_Conv2d(in_channels, out_channels - in_channels, kernel_size=1, stride=1, bias=False),
                # batch_norm_2d(out_channels - in_channels),
            )
        self.pools = nn.MaxPool3d((1, stride, stride), stride=(1, stride, stride))

    def forward(self, x):
        # print(x.shape)
        # print(self.residual_function(x).shape)
        temp = self.shortcut(x)
        out = torch.cat((temp, x), dim=2)
        out = self.pools(out)
        # print(self.residual_function(x).shape)
        # print(out.shape)
        return (self.residual_function(x) + out)


class Concat_res5(nn.Module):  #
    def __init__(self, in_channels, out_channels, k_size=3, stride=1, e=0.5):
        super().__init__()
        c_ = int(out_channels * e)  # hidden channels
        if k_size == 3:
            pad = 1
        if k_size == 1:
            pad = 0
        self.residual_function = nn.Sequential(
            mem_update(act=False),
            # AKConv(in_channels, out_channels, num_param=k_size, stride=stride),
            Snn_Conv2d(in_channels, out_channels, kernel_size=k_size, stride=stride, padding=pad, bias=False),
            batch_norm_2d(out_channels),
            # RepConv(in_channels, out_channels, k=k_size, s=stride, p=pad),
            mem_update(act=False),
            # Snn_Conv2d(out_channels, out_channels, kernel_size=k_size, padding=pad, bias=False),
            AKConv(out_channels, out_channels, num_param=k_size, stride=1),
            # batch_norm_2d1(out_channels),
        )
        # shortcut
        self.shortcut = nn.Sequential(
        )
        if in_channels < out_channels:
            self.shortcut = nn.Sequential(
                mem_update(act=False),
                Snn_Conv2d(in_channels, out_channels - in_channels, kernel_size=1, stride=1, bias=False),
                batch_norm_2d(out_channels - in_channels),
            )
        self.pools = nn.MaxPool3d((1, stride, stride), stride=(1, stride, stride))

    def forward(self, x):
        temp = self.shortcut(x)
        out = torch.cat((temp, x), dim=2)
        out = self.pools(out)
        return (self.residual_function(x) + out)


class Concat_res6(nn.Module):  #
    def __init__(self, in_channels, out_channels, k_size=3, stride=1, e=2):
        super().__init__()
        c_ = int(in_channels * e)  # hidden channels
        if k_size == 3:
            pad = 1
        if k_size == 1:
            pad = 0
        self.residual_function = nn.Sequential(
            # mem_update(act=False),
            # Snn_Conv2d(in_channels, c_, kernel_size=1, stride=1, padding=0, bias=False),
            # batch_norm_2d(c_),
            mem_update(act=False),
            PartialConv(in_channels),
            # batch_norm_2d(in_channels),
            mem_update(act=False),
            Snn_Conv2d(in_channels, out_channels, kernel_size=k_size, stride=stride, padding=pad, bias=False),
            batch_norm_2d(out_channels),
            mem_update(act=False),
            # Snn_Conv2d(out_channels, out_channels, 1, 1, bias=False),
            ScConv(out_channels),
            batch_norm_2d1(out_channels),
        )
        # shortcut
        self.shortcut = nn.Sequential(
        )
        if in_channels < out_channels:
            self.shortcut = nn.Sequential(
                mem_update(act=False),
                Snn_Conv2d(in_channels, out_channels - in_channels, kernel_size=1, stride=1, padding=0, bias=False),
                batch_norm_2d(out_channels - in_channels),
            )
        self.pools = nn.MaxPool3d((1, stride, stride), stride=(1, stride, stride))

    def forward(self, x):
        temp = self.shortcut(x)
        out = torch.cat((temp, x), dim=2)
        out = self.pools(out)
        return (self.residual_function(x) + out)


class BasicBlock_ms(nn.Module):  # tiny3.yaml
    def __init__(self, in_channels, out_channels, k_size=3, stride=1, e=0.5):
        super().__init__()
        c_ = int(out_channels * e)  # hidden channels  
        if k_size == 3:
            pad = 1
        if k_size == 1:
            pad = 0
        self.residual_function = nn.Sequential(
            mem_update(act=False),
            Snn_Conv2d(in_channels, c_, kernel_size=k_size, stride=stride, padding=pad, bias=False),
            batch_norm_2d(c_),
            mem_update(act=False),
            Snn_Conv2d(c_, out_channels, kernel_size=k_size, padding=pad, bias=False),
            batch_norm_2d1(out_channels),

        )
        # shortcut
        self.shortcut = nn.Sequential(
        )
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.MaxPool3d((1, stride, stride), stride=(1, stride, stride)),
                Snn_Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                batch_norm_2d(out_channels),
            )

    def forward(self, x):
        # print(self.residual_function(x).shape)
        return (self.residual_function(x) + self.shortcut(x))


class ConcatBlock_ms(nn.Module):  #
    def __init__(self, in_channels, out_channels, k_size=3, stride=1, e=0.5):
        super().__init__()
        c_ = int(out_channels * e)  # hidden channels  
        if k_size == 3:
            pad = 1
        if k_size == 1:
            pad = 0
        self.residual_function = nn.Sequential(
            mem_update(act=False),
            Snn_Conv2d(in_channels, c_, kernel_size=k_size, stride=stride, padding=pad, bias=False),
            batch_norm_2d(c_),
            mem_update(act=False),
            Snn_Conv2d(c_, out_channels, kernel_size=k_size, padding=pad, bias=False),
            batch_norm_2d1(out_channels),
        )
        # shortcut
        self.shortcut = nn.Sequential(
        )

        if in_channels < out_channels:
            self.shortcut = nn.Sequential(
                mem_update(act=False),
                Snn_Conv2d(in_channels, out_channels - in_channels, kernel_size=1, stride=1, bias=False),
                batch_norm_2d(out_channels - in_channels),
            )
        self.pools = nn.MaxPool3d((1, stride, stride), stride=(1, stride, stride))

    def forward(self, x):
        # print(self.residual_function(x).shape)
        temp = self.shortcut(x)
        out = torch.cat((temp, x), dim=2)
        out = self.pools(out)
        return (self.residual_function(x) + out)


class Contract(nn.Module):
    # Contract width-height into channels, i.e. x(1,64,80,80) to x(1,256,40,40)
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        b, c, h, w = x.size()  # assert (h / s == 0) and (W / s == 0), 'Indivisible gain'
        s = self.gain
        # .view: 改变tensor的维度
        x = x.view(b, c, h // s, s, w // s, s)  # x(1,64,40,2,40,2)
        # permute: 改变tensor的维度顺序
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # x(1,2,2,64,40,40)
        return x.view(b, c * s * s, h // s, w // s)  # x(1,256,40,40)


class Expand(nn.Module):
    # Expand channels into width-height, i.e. x(1,64,80,80) to x(1,16,160,160)
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        b, c, h, w = x.size()  # assert C / s ** 2 == 0, 'Indivisible gain'
        s = self.gain
        x = x.view(b, s, s, c // s ** 2, h, w)  # x(1,2,2,16,80,80)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()  # x(1,16,80,2,80,2)
        return x.view(b, c // s ** 2, h * s, w * s)  # x(1,16,160,160)


# 作拼接的一个类
# 拼接函数，将两个tensor进行拼接
class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)


class ContextGuideFusionModule(nn.Module):
    def __init__(self, inc, dimension=1):
        super().__init__()
        self.d = dimension
        self.adjust_conv = nn.Identity()
        if inc[0] != inc[1]:
            self.adjust_conv = Snn_Conv2d(inc[0], inc[1], 1)


        # self.se = ELA(inc[1] * 2)
        # self.se = MHSA(inc[1] * 2, inc[1] * 2, 4)

    def forward(self, x):
        x0, x1 = x
        ans = torch.ones_like(x1)
        x0 = self.adjust_conv(x0)
        x_concat = torch.cat([x0, x1], dim=self.d)
        # x_concat = self.se(x_concat)
        x0_weight, x1_weight = torch.split(x_concat, [x0.size()[2], x1.size()[2]], dim=self.d)
        x0_weight = x0 * x0_weight
        x1_weight = x1 * x1_weight
        return torch.cat([x0 + x1_weight, x1 + x0_weight], dim=self.d)


class ContextGuideFusionModulev2(nn.Module):
    def __init__(self, inc, dimension=1):
        super().__init__()
        self.d = dimension
        self.adjust_conv = nn.Identity()
        if inc[0] != inc[1]:
            self.adjust_conv = Snn_Conv2d(inc[0], inc[1], 1)

        #self.convs = nn.ModuleList(
        #    [Snn_Conv2d(inc[1], inc[1], 3, 1, 1) for _ in range(2)]
        #)
        self.convs = Snn_Conv2d(inc[1], inc[1], 3, 1, 1)

        self.se = EMA(inc[1] * 2)
        # self.se = MHSA(inc[1] * 2, inc[1] * 2, 4)

    def forward(self, x):
        x0, x1 = x
        ans = torch.ones_like(x1)
        x0 = self.adjust_conv(x0)
        x_concat = torch.cat([x0, x1], dim=self.d)
        x_concat = self.se(x_concat)
        x0_weight, x1_weight = torch.split(x_concat, [x0.size()[2], x1.size()[2]], dim=self.d)
        for i, z in enumerate([x0_weight, x1_weight]):
            #ans = ans * self.convs[i](z)
            ans = ans * self.convs(z)
        x0_weight = x0 * x0_weight
        x1_weight = x1 * x1_weight
        return torch.cat([x0 + x1_weight + ans, x1 + x0_weight + ans], dim=self.d)

'''===========2.DetectMultiBackend： ================'''


class DetectMultiBackend(nn.Module):
    # YOLOv5 MultiBackend class for python inference on various backends
    def __init__(self, weights='yolov5s.pt', device=None, dnn=True, fuse=True):
        # Usage:
        #   PyTorch:      weights = *.pt
        #   TorchScript:            *.torchscript.pt
        #   CoreML:                 *.mlmodel
        #   TensorFlow:             *_saved_model
        #   TensorFlow:             *.pb
        #   TensorFlow Lite:        *.tflite
        #   ONNX Runtime:           *.onnx
        #   OpenCV DNN:             *.onnx with dnn=True
        super().__init__()
        # 判断weights是否为list，若是取出第一个值作为传入路径
        w = str(weights[0] if isinstance(weights, list) else weights)
        suffix, suffixes = Path(w).suffix.lower(), ['.pt', '.onnx', '.tflite', '.pb', '', '.mlmodel']
        check_suffix(w, suffixes)  # check weights have acceptable suffix
        pt, onnx, tflite, pb, saved_model, coreml = (suffix == x for x in suffixes)  # backend booleans
        jit = pt and 'torchscript' in w.lower()
        stride, names = 64, [f'class{i}' for i in range(1000)]  # assign defaults

        if jit:  # TorchScript
            LOGGER.info(f'Loading {w} for TorchScript inference...')
            extra_files = {'config.txt': ''}  # model metadata
            model = torch.jit.load(w, _extra_files=extra_files)
            if extra_files['config.txt']:
                d = json.loads(extra_files['config.txt'])  # extra_files dict
                stride, names = int(d['stride']), d['names']
        elif pt:  # PyTorch
            from models.experimental import attempt_load  # scoped to avoid circular import
            model = torch.jit.load(w) if 'torchscript' in w else attempt_load(weights, map_location=device, fuse=fuse)
            stride = int(model.stride.max())  # model stride
            names = model.module.names if hasattr(model, 'module') else model.names  # get class names
        elif coreml:  # CoreML *.mlmodel
            import coremltools as ct
            model = ct.models.MLModel(w)
        elif dnn:  # ONNX OpenCV DNN
            LOGGER.info(f'Loading {w} for ONNX OpenCV DNN inference...')
            check_requirements(('opencv-python>=4.5.4',))
            net = cv2.dnn.readNetFromONNX(w)
        elif onnx:  # ONNX Runtime
            LOGGER.info(f'Loading {w} for ONNX Runtime inference...')
            check_requirements(('onnx', 'onnxruntime-gpu' if torch.has_cuda else 'onnxruntime'))
            import onnxruntime
            session = onnxruntime.InferenceSession(w, None)
        else:  # TensorFlow model (TFLite, pb, saved_model)
            import tensorflow as tf
            if pb:  # https://www.tensorflow.org/guide/migrate#a_graphpb_or_graphpbtxt
                def wrap_frozen_graph(gd, inputs, outputs):
                    x = tf.compat.v1.wrap_function(lambda: tf.compat.v1.import_graph_def(gd, name=""), [])  # wrapped
                    return x.prune(tf.nest.map_structure(x.graph.as_graph_element, inputs),
                                   tf.nest.map_structure(x.graph.as_graph_element, outputs))

                LOGGER.info(f'Loading {w} for TensorFlow *.pb inference...')
                graph_def = tf.Graph().as_graph_def()
                graph_def.ParseFromString(open(w, 'rb').read())
                frozen_func = wrap_frozen_graph(gd=graph_def, inputs="x:0", outputs="Identity:0")
            elif saved_model:
                LOGGER.info(f'Loading {w} for TensorFlow saved_model inference...')
                model = tf.keras.models.load_model(w)
            elif tflite:  # https://www.tensorflow.org/lite/guide/python#install_tensorflow_lite_for_python
                if 'edgetpu' in w.lower():
                    LOGGER.info(f'Loading {w} for TensorFlow Edge TPU inference...')
                    import tflite_runtime.interpreter as tfli
                    delegate = {'Linux': 'libedgetpu.so.1',  # install https://coral.ai/software/#edgetpu-runtime
                                'Darwin': 'libedgetpu.1.dylib',
                                'Windows': 'edgetpu.dll'}[platform.system()]
                    interpreter = tfli.Interpreter(model_path=w, experimental_delegates=[tfli.load_delegate(delegate)])
                else:
                    LOGGER.info(f'Loading {w} for TensorFlow Lite inference...')
                    interpreter = tf.lite.Interpreter(model_path=w)  # load TFLite model
                interpreter.allocate_tensors()  # allocate
                input_details = interpreter.get_input_details()  # inputs
                output_details = interpreter.get_output_details()  # outputs
        self.__dict__.update(locals())  # assign all variables to self

    def forward(self, im, augment=False, visualize=False, val=False):
        # YOLOv5 MultiBackend inference
        b, ch, h, w = im.shape  # batch, channel, height, width
        if self.pt:  # PyTorch
            y = self.model(im) if self.jit else self.model(im, augment=augment, visualize=visualize)
            return y if val else y[0]
        elif self.coreml:  # CoreML *.mlmodel
            im = im.permute(0, 2, 3, 1).cpu().numpy()  # torch BCHW to numpy BHWC shape(1,320,192,3)
            im = Image.fromarray((im[0] * 255).astype('uint8'))
            # im = im.resize((192, 320), Image.ANTIALIAS)
            y = self.model.predict({'image': im})  # coordinates are xywh normalized
            box = xywh2xyxy(y['coordinates'] * [[w, h, w, h]])  # xyxy pixels
            conf, cls = y['confidence'].max(1), y['confidence'].argmax(1).astype(np.float)
            y = np.concatenate((box, conf.reshape(-1, 1), cls.reshape(-1, 1)), 1)
        elif self.onnx:  # ONNX
            im = im.cpu().numpy()  # torch to numpy
            if self.dnn:  # ONNX OpenCV DNN
                self.net.setInput(im)
                y = self.net.forward()
            else:  # ONNX Runtime
                y = self.session.run([self.session.get_outputs()[0].name], {self.session.get_inputs()[0].name: im})[0]
        else:  # TensorFlow model (TFLite, pb, saved_model)
            im = im.permute(0, 2, 3, 1).cpu().numpy()  # torch BCHW to numpy BHWC shape(1,320,192,3)
            if self.pb:
                y = self.frozen_func(x=self.tf.constant(im)).numpy()
            elif self.saved_model:
                y = self.model(im, training=False).numpy()
            elif self.tflite:
                input, output = self.input_details[0], self.output_details[0]
                int8 = input['dtype'] == np.uint8  # is TFLite quantized uint8 model
                if int8:
                    scale, zero_point = input['quantization']
                    im = (im / scale + zero_point).astype(np.uint8)  # de-scale
                self.interpreter.set_tensor(input['index'], im)
                self.interpreter.invoke()
                y = self.interpreter.get_tensor(output['index'])
                if int8:
                    scale, zero_point = output['quantization']
                    y = (y.astype(np.float32) - zero_point) * scale  # re-scale
            y[..., 0] *= w  # x
            y[..., 1] *= h  # y
            y[..., 2] *= w  # w
            y[..., 3] *= h  # h
        y = torch.tensor(y)
        return (y, []) if val else y


'''===========3.AutoShape：自动调整shape,该类基本未用================'''


class AutoShape(nn.Module):
    # YOLOv5 input-robust model wrapper for passing cv2/np/PIL/torch inputs. Includes preprocessing, inference and NMS
    conf = 0.25  # NMS confidence threshold
    iou = 0.45  # NMS IoU threshold
    classes = None  # (optional list) filter by class, i.e. = [0, 15, 16] for COCO persons, cats and dogs
    multi_label = False  # NMS multiple labels per box
    max_det = 1000  # maximum number of detections per image

    def __init__(self, model):
        super().__init__()
        self.model = model.eval()

    def autoshape(self):
        LOGGER.info('AutoShape already enabled, skipping... ')  # model already converted to model.autoshape()
        return self

    def _apply(self, fn):
        # Apply to(), cpu(), cuda(), half() to model tensors that are not parameters or registered buffers
        self = super()._apply(fn)
        m = self.model.model[-1]  # Detect()
        m.stride = fn(m.stride)
        m.grid = list(map(fn, m.grid))
        if isinstance(m.anchor_grid, list):
            m.anchor_grid = list(map(fn, m.anchor_grid))
        return self

    @torch.no_grad()
    def forward(self, imgs, size=640, augment=False, profile=False):
        # Inference from various sources. For height=640, width=1280, RGB images example inputs are:
        #   file:       imgs = 'data/images/zidane.jpg'  # str or PosixPath
        #   URI:             = 'https://ultralytics.com/images/zidane.jpg'
        #   OpenCV:          = cv2.imread('image.jpg')[:,:,::-1]  # HWC BGR to RGB x(640,1280,3)
        #   PIL:             = Image.open('image.jpg') or ImageGrab.grab()  # HWC x(640,1280,3)
        #   numpy:           = np.zeros((640,1280,3))  # HWC
        #   torch:           = torch.zeros(16,3,320,640)  # BCHW (scaled to size=640, 0-1 values)
        #   multiple:        = [Image.open('image1.jpg'), Image.open('image2.jpg'), ...]  # list of images

        t = [time_sync()]
        p = next(self.model.parameters())  # for device and type
        if isinstance(imgs, torch.Tensor):  # torch
            with amp.autocast(enabled=p.device.type != 'cpu'):
                return self.model(imgs.to(p.device).type_as(p), augment, profile)  # inference

        # Pre-process
        n, imgs = (len(imgs), imgs) if isinstance(imgs, list) else (1, [imgs])  # number of images, list of images
        shape0, shape1, files = [], [], []  # image and inference shapes, filenames
        for i, im in enumerate(imgs):
            f = f'image{i}'  # filename
            if isinstance(im, (str, Path)):  # filename or uri
                im, f = Image.open(requests.get(im, stream=True).raw if str(im).startswith('http') else im), im
                im = np.asarray(exif_transpose(im))
            elif isinstance(im, Image.Image):  # PIL Image
                im, f = np.asarray(exif_transpose(im)), getattr(im, 'filename', f) or f
            files.append(Path(f).with_suffix('.jpg').name)
            if im.shape[0] < 5:  # image in CHW
                im = im.transpose((1, 2, 0))  # reverse dataloader .transpose(2, 0, 1)
            im = im[..., :3] if im.ndim == 3 else np.tile(im[..., None], 3)  # enforce 3ch input
            s = im.shape[:2]  # HWC
            shape0.append(s)  # image shape
            g = (size / max(s))  # gain
            shape1.append([y * g for y in s])
            imgs[i] = im if im.data.contiguous else np.ascontiguousarray(im)  # update
        shape1 = [make_divisible(x, int(self.stride.max())) for x in np.stack(shape1, 0).max(0)]  # inference shape
        x = [letterbox(im, new_shape=shape1, auto=False)[0] for im in imgs]  # pad
        x = np.stack(x, 0) if n > 1 else x[0][None]  # stack
        x = np.ascontiguousarray(x.transpose((0, 3, 1, 2)))  # BHWC to BCHW
        x = torch.from_numpy(x).to(p.device).type_as(p) / 255  # uint8 to fp16/32
        t.append(time_sync())

        with amp.autocast(enabled=p.device.type != 'cpu'):
            # Inference
            y = self.model(x, augment, profile)[0]  # forward
            t.append(time_sync())

            # Post-process
            y = non_max_suppression(y, self.conf, iou_thres=self.iou, classes=self.classes,
                                    multi_label=self.multi_label, max_det=self.max_det)  # NMS
            for i in range(n):
                scale_coords(shape1, y[i][:, :4], shape0[i])

            t.append(time_sync())
            return Detections(imgs, y, files, t, self.names, x.shape)


'''===========3.Detections：对推理结果进行处理================'''


class Detections:
    #  detections class for inference results
    def __init__(self, imgs, pred, files, times=None, names=None, shape=None):
        super().__init__()
        d = pred[0].device  # device
        gn = [torch.tensor([*(im.shape[i] for i in [1, 0, 1, 0]), 1, 1], device=d) for im in imgs]  # normalizations
        self.imgs = imgs  # list of images as numpy arrays
        self.pred = pred  # list of tensors pred[0] = (xyxy, conf, cls)
        self.names = names  # class names
        self.files = files  # image filenames
        self.xyxy = pred  # xyxy pixels
        self.xywh = [xyxy2xywh(x) for x in pred]  # xywh pixels
        self.xyxyn = [x / g for x, g in zip(self.xyxy, gn)]  # xyxy normalized
        self.xywhn = [x / g for x, g in zip(self.xywh, gn)]  # xywh normalized
        self.n = len(self.pred)  # number of images (batch size)
        self.t = tuple((times[i + 1] - times[i]) * 1000 / self.n for i in range(3))  # timestamps (ms)
        self.s = shape  # inference BCHW shape

    def display(self, pprint=False, show=False, save=False, crop=False, render=False, save_dir=Path('')):
        crops = []
        for i, (im, pred) in enumerate(zip(self.imgs, self.pred)):
            s = f'image {i + 1}/{len(self.pred)}: {im.shape[0]}x{im.shape[1]} '  # string
            if pred.shape[0]:
                for c in pred[:, -1].unique():
                    n = (pred[:, -1] == c).sum()  # detections per class
                    s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string
                if show or save or render or crop:
                    annotator = Annotator(im, example=str(self.names))
                    for *box, conf, cls in reversed(pred):  # xyxy, confidence, class
                        label = f'{self.names[int(cls)]} {conf:.2f}'
                        if crop:
                            file = save_dir / 'crops' / self.names[int(cls)] / self.files[i] if save else None
                            crops.append({'box': box, 'conf': conf, 'cls': cls, 'label': label,
                                          'im': save_one_box(box, im, file=file, save=save)})
                        else:  # all others
                            annotator.box_label(box, label, color=colors(cls))
                    im = annotator.im
            else:
                s += '(no detections)'

            im = Image.fromarray(im.astype(np.uint8)) if isinstance(im, np.ndarray) else im  # from np
            if pprint:
                LOGGER.info(s.rstrip(', '))
            if show:
                im.show(self.files[i])  # show
            if save:
                f = self.files[i]
                im.save(save_dir / f)  # save
                if i == self.n - 1:
                    LOGGER.info(f"Saved {self.n} image{'s' * (self.n > 1)} to {colorstr('bold', save_dir)}")
            if render:
                self.imgs[i] = np.asarray(im)
        if crop:
            if save:
                LOGGER.info(f'Saved results to {save_dir}\n')
            return crops

    def print(self):
        self.display(pprint=True)  # print results
        LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {tuple(self.s)}' %
                    self.t)

    def show(self):
        self.display(show=True)  # show results

    def save(self, save_dir='runs/detect/exp'):
        save_dir = increment_path(save_dir, exist_ok=save_dir != 'runs/detect/exp', mkdir=True)  # increment save_dir
        self.display(save=True, save_dir=save_dir)  # save results

    def crop(self, save=True, save_dir='runs/detect/exp'):
        save_dir = increment_path(save_dir, exist_ok=save_dir != 'runs/detect/exp', mkdir=True) if save else None
        return self.display(crop=True, save=save, save_dir=save_dir)  # crop results

    def render(self):
        self.display(render=True)  # render results
        return self.imgs

    def pandas(self):
        # return detections as pandas DataFrames, i.e. print(results.pandas().xyxy[0])
        new = copy(self)  # return copy
        ca = 'xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class', 'name'  # xyxy columns
        cb = 'xcenter', 'ycenter', 'width', 'height', 'confidence', 'class', 'name'  # xywh columns
        for k, c in zip(['xyxy', 'xyxyn', 'xywh', 'xywhn'], [ca, ca, cb, cb]):
            a = [[x[:5] + [int(x[5]), self.names[int(x[5])]] for x in x.tolist()] for x in getattr(self, k)]  # update
            setattr(new, k, [pd.DataFrame(x, columns=c) for x in a])
        return new

    def tolist(self):
        # return a list of Detections objects, i.e. 'for result in results.tolist():'
        x = [Detections([self.imgs[i]], [self.pred[i]], self.names, self.s) for i in range(self.n)]
        for d in x:
            for k in ['imgs', 'pred', 'xyxy', 'xyxyn', 'xywh', 'xywhn']:
                setattr(d, k, getattr(d, k)[0])  # pop out of list
        return x

    def __len__(self):
        return self.n


class LightConv(nn.Module):
    """
    Light convolution with args(ch_in, ch_out, kernel).

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, c2, k=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv1 = Conv_3(c1, c2, 1, act=False)
        self.conv2 = DWConv(c2, c2, k, act=act)

    def forward(self, x):
        """Apply 2 convolutions to input tensor."""
        return self.conv2(self.conv1(x))


'''===========3.DWConv：深度可分离卷积================'''


class DWConv(Conv_3):
    # Depth-wise convolution class
    def __init__(self, c1, c2, k=1, s=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), act=act)


'''===========4.Bottleneck：标准的瓶颈层 由1x1conv+3x3conv+残差块组成================'''


class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv_3(c1, c_, k[0], 1)
        self.cv2 = Conv_4(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


'''===========5.BottleneckCSP：瓶颈层 由几个Bottleneck模块的堆叠+CSP结构组成================'''


class BottleneckCSP(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        """在C3模块和yolo.py的parse_model模块调用
            CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
            :params c1: 整个BottleneckCSP的输入channel
            :params c2: 整个BottleneckCSP的输出channel
            :params n: 有n个Bottleneck
            :params shortcut: bool Bottleneck中是否有shortcut，默认True
            :params g: Bottleneck中的3x3卷积类型  =1普通卷积  >1深度可分离卷积
            :params e: expansion ratio c2xe=中间其他所有层的卷积核个数/中间所有层的输入输出channel数
            c_: bottleneckCSP 结构的中间层的通道数，由膨胀率e决定
            """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        # 4个1*1卷积层的堆叠
        self.cv1 = Conv_3(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv_3(2 * c_, c2, 1, 1)
        # bn层
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        # 激活函数
        self.act = nn.SiLU()
        # m：叠加n次Bottleneck的操作
        # 操作符*可以把一个list拆开成一个个独立的元素
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        # y1相当于先做一次cv1操作然后进行m操作最后进行cv3操作，也就是BCSPn模块中的上面的分支操作
        # 输入x ->Conv模块 ->n个bottleneck模块 ->Conv模块 ->y1
        y1 = self.cv3(self.m(self.cv1(x)))
        # y2就是进行cv2操作，也就是BCSPn模块中的下面的分支操作（直接逆行conv操作的分支， Conv--nXBottleneck--conv）
        # 输入x -> Conv模块 -> 输出y2
        y2 = self.cv2(x)
        # 最后y1和y2做拼接， 接着进入bn层做归一化， 然后做act激活， 最后输出cv4
        # 输入y1,y2->按照通道数融合 ->归一化 -> 激活函数 -> Conv输出 -> 输出
        # torch.cat(y1, y2), dim=1: 这里是指定在第一个维度上进行合并，即在channel维度上合并
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), dim=2))))


'''===========6.C3：和BottleneckCSP模块类似，但是少了一个Conv模块================'''


# ===6.1 C3===
class C3(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        """在C3TR模块和yolo.py的parse_model模块调用
         CSP Bottleneck with 3 convolutions
         :params c1: 整个BottleneckCSP的输入channel
         :params c2: 整个BottleneckCSP的输出channel
         :params n: 有n个Bottleneck
         :params shortcut: bool Bottleneck中是否有shortcut，默认True
         :params g: Bottleneck中的3x3卷积类型  =1普通卷积  >1深度可分离卷积
         :params e: expansion ratio c2xe=中间其他所有层的卷积核个数/中间所有层的输入输出channel数
         """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        # 3个1*1卷积层的堆叠，比BottleneckCSP少一个
        self.cv1 = Conv_3(c1, c_, 1, 1)
        self.cv2 = Conv_3(c1, c_, 1, 1)
        self.cv3 = Conv_4(2 * c_, c2, 1)  # act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))
        # self.m = nn.Sequential(*[CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)])

    def forward(self, x):
        # 将第一个卷积层与第二个卷积层的结果拼接在一起
        # print(x.shape)
        # print(self.cv1(x).shape)
        # print(self.cv2(x).shape)
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=2))


class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv_3(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv_4((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 2))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 2))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 2))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 2))


# class C3(nn.Module):
#     # CSP Bottleneck with 3 convolutions
#     def __init__(self, c1, c2, n=1, k=1, s=1, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
#         """在C3TR模块和yolo.py的parse_model模块调用
#          CSP Bottleneck with 3 convolutions
#          :params c1: 整个BottleneckCSP的输入channel
#          :params c2: 整个BottleneckCSP的输出channel
#          :params n: 有n个Bottleneck
#          :params shortcut: bool Bottleneck中是否有shortcut，默认True
#          :params g: Bottleneck中的3x3卷积类型  =1普通卷积  >1深度可分离卷积
#          :params e: expansion ratio c2xe=中间其他所有层的卷积核个数/中间所有层的输入输出channel数
#          """
#         super().__init__()
#         c_ = int(c2 * e)  # hidden channels
#         if s == 2:
#             self.dw = nn.Sequential(
#                 mem_update(act=False),
#                 Snn_Conv2d(c1, c1, kernel_size=k, stride=s, padding=1, bias=False),
#                 batch_norm_2d(c1))
#         # 3个1*1卷积层的堆叠，比BottleneckCSP少一个
#         self.cv1 = Conv_3(c1, c_, 1, 1)
#         self.cv2 = Conv_3(c1, c_, 1, 1)
#         self.cv3 = Conv_4(2 * c_, c2, 1)  # act=FReLU(c2)
#         self.m = nn.Sequential(*(Bottleneck_2(c_, c_, g, e=4) for _ in range(n)))
#         # self.m = nn.Sequential(*[CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)])
#
#     def forward(self, x):
#         # 将第一个卷积层与第二个卷积层的结果拼接在一起
#         # print(x.shape)
#         # print(self.cv1(x).shape)
#         # print(self.cv2(x).shape)
#         if self.dw:
#             x = self.dw(x)
#         return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=2))


# ===6.2 C3SPP(C3)：继承自 C3，n 个 Bottleneck 更换为 1 个 SPP=== #
class C3SPP(C3):
    # C3 module with SPP()
    def __init__(self, c1, c2, k=(5, 9, 13), n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = SPP(c_, c_, k)


# ===6.3 C3Ghost(C3)：继承自 C3，Bottleneck 更换为 GhostBottleneck=== #
class C3Ghost(C3):
    # C3 module with GhostBottleneck()
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(GhostBottleneck(c_, c_) for _ in range(n)))


'''===========7.SPP：空间金字塔池化模块================'''


# 用在骨干网络收尾阶段，用于融合多尺度特征。
# ===7.1 SPP：空间金字塔池化=== #
class SPP(nn.Module):
    # Spatial Pyramid Pooling (SPP) layer https://arxiv.org/abs/1406.4729
    def __init__(self, c1, c2, k=(5, 9, 13)):
        """在yolo.py的parse_model模块调用
               空间金字塔池化 Spatial pyramid pooling layer used in YOLOv3-SPP
               :params c1: SPP模块的输入channel
               :params c2: SPP模块的输出channel
               :params k: 保存着三个maxpool的卷积核大小 默认是(5, 9, 13)
               """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        # 1*1卷积
        self.cv1 = Conv_3(c1, c_, 1, 1)
        #  这里+1是因为有len(k)+1个输入
        self.cv2 = Conv_3(c_ * (len(k) + 1), c2, 1, 1)
        # m先进行最大池化操作， 然后通过nn.ModuleList进行构造一个模块 在构造时对每一个k都要进行最大池化
        self.m = nn.ModuleList([nn.MaxPool3d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        # 先进行cv1的操作
        x = self.cv1(x)
        # 忽略了警告错误的输出
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # suppress torch 1.9.0 max_pool2d() warning
            # 对每一个m进行最大池化 和没有做池化的每一个输入进行叠加  然后做拼接 最后做cv2操作
            return self.cv2(torch.cat([x] + [m(x) for m in self.m], 2))


# ===7.2 SPPF：快速版的空间金字塔池化=== #
class SPPF(nn.Module):
    # Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
    def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv_3(c1, c_, 1, 1)
        self.cv2 = Conv_4(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool3d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        # print(x.shape)
        x = self.cv1(x)  # 先通过CBL进行通道数的减半
        # print(x.shape)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # suppress torch 1.9.0 max_pool2d() warning
            y1 = self.m(x)
            y2 = self.m(y1)
            # print(y1.shape)
            # print(y2.shape)
            # # 上述两次最大池化
            # print(self.cv2(torch.cat([x, y1, y2, self.m(y2)], 2)).shape)
            return self.cv2(torch.cat([x, y1, y2, self.m(y2)], 2))
            # 将原来的x,一次池化后的y1,两次池化后的y2,3次池化的self.m(y2)先进行拼接，然后再CBL


'''===========8.Focus：把宽度w和高度h的信息整合到c空间================'''


class Focus(nn.Module):
    # Focus wh information into c-space
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        """在yolo.py的parse_model函数中被调用
                理论：从高分辨率图像中，周期性的抽出像素点重构到低分辨率图像中，即将图像相邻的四个位置进行堆叠，
                    聚焦wh维度信息到c通道空，提高每个点感受野，并减少原始信息的丢失，该模块的设计主要是减少计算量加快速度。
                Focus wh information into c-space 把宽度w和高度h的信息整合到c空间中
                先做4个slice 再concat 最后再做Conv
                slice后 (b,c1,w,h) -> 分成4个slice 每个slice(b,c1,w/2,h/2)
                concat(dim=1)后 4个slice(b,c1,w/2,h/2)) -> (b,4c1,w/2,h/2)
                conv后 (b,4c1,w/2,h/2) -> (b,c2,w/2,h/2)
                :params c1: slice后的channel
                :params c2: Focus最终输出的channel
                :params k: 最后卷积的kernel
                :params s: 最后卷积的stride
                :params p: 最后卷积的padding
                :params g: 最后卷积的分组情况  =1普通卷积  >1深度可分离卷积
                :params act: bool激活函数类型  默认True:SiLU()/Swish  False:不用激活函数
                """
        super().__init__()
        # concat后的卷积（最后的卷积）
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act)
        # self.contract = Contract(gain=2)

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        # 先进行切分， 然后进行拼接， 最后再做conv操作
        return self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))
        # return self.conv(self.contract(x))


'''===========1.GhostConv：幻象卷积  轻量化网络卷积模块================'''


class GhostConv(nn.Module):
    # Ghost Convolution https://github.com/huawei-noah/ghostnet

    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):  # ch_in, ch_out, kernel, stride, groups
        super().__init__()
        c_ = c2 // 2  # hidden channels
        # 第一步卷积: 少量卷积, 一般是一半的计算量
        self.cv1 = Conv_3(c1, c_, k, s, None, g, act)
        # 第二步卷积: cheap operations 使用3x3或5x5的卷积, 并且是逐个特征图的进行卷积（Depth-wise convolutional
        self.cv2 = Conv_4(c_, c_, 5, 1, None, c_, act)

    def forward(self, x):
        y = self.cv1(x)
        return torch.cat([y, self.cv2(y)], 2)


'''===========2.GhostBottleneck：幻象瓶颈层 ================'''


class GhostBottleneck(nn.Module):
    # Ghost Bottleneck https://github.com/huawei-noah/ghostnet
    def __init__(self, c1, c2, k=3, s=1):  # ch_in, ch_out, kernel, stride
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(GhostConv(c1, c_, 1, 1),  # pw
                                  DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
                                  GhostConv(c_, c2, 1, 1, act=False))  # pw-linear
        # 注意, 源码中并不是直接Identity连接, 而是先经过一个DWConv + Conv, 再进行shortcut连接的。
        self.shortcut = nn.Sequential(DWConv(c1, c1, k, s, act=False),
                                      Conv_3(c1, c2, 1, 1, act=False)) if s == 2 else nn.Identity()

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)


'''===============================================五、模型扩展模块==================================================='''
'''===========1.C3TR(C3)：继承自 C3，n 个 Bottleneck 更换为 1 个 TransformerBlock ================'''


class C3TR(C3):
    """
        这部分是根据上面的C3结构改编而来的, 将原先的Bottleneck替换为调用TransformerBlock模块
        """

    # C3 module with TransformerBlock()
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        ''' 在C3RT模块和yolo.py的parse_model函数中被调用
                :params c1: 整个C3的输入channel
                :params c2: 整个C3的输出channel
                :params n: 有n个子模块[Bottleneck/CrossConv]
                :params shortcut: bool值，子模块[Bottlenec/CrossConv]中是否有shortcut，默认True
                :params g: 子模块[Bottlenec/CrossConv]中的3x3卷积类型，=1普通卷积，>1深度可分离卷积
                :params e: expansion ratio，e*c2=中间其它所有层的卷积核个数=中间所有层的的输入输出channel
                '''
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)


'''===========1.TransformerLayer：================'''


class TransformerLayer(nn.Module):
    # Transformer layer https://arxiv.org/abs/2010.11929 (LayerNorm layers removed for better performance)
    """
        Transformer layer https://arxiv.org/abs/2010.11929 (LayerNorm layers removed for better performance)
        这部分相当于原论文中的单个Encoder部分(只移除了两个Norm部分, 其他结构和原文中的Encoding一模一样)
       """

    def __init__(self, c, num_heads):
        super().__init__()
        self.q = nn.Linear(c, c, bias=False)
        self.k = nn.Linear(c, c, bias=False)
        self.v = nn.Linear(c, c, bias=False)
        # 输入: query、key、value
        # 输出: 0 attn_output 即通过self-attention之后，从每一个词语位置输出来的attention 和输入的query它们形状一样的
        #      1 attn_output_weights 即attention weights 每一个单词和任意另一个单词之间都会产生一个weight
        self.ma = nn.MultiheadAttention(embed_dim=c, num_heads=num_heads)
        self.fc1 = nn.Linear(c, c, bias=False)
        self.fc2 = nn.Linear(c, c, bias=False)

    def forward(self, x):
        # 多头注意力机制 + 残差(这里移除了LayerNorm for better performance)
        x = self.ma(self.q(x), self.k(x), self.v(x))[0] + x
        # feed forward 前馈神经网络 + 残差(这里移除了LayerNorm for better performance)
        x = self.fc2(self.fc1(x)) + x
        return x


'''===========2.TransformerBlock：================'''


class TransformerBlock(nn.Module):
    # Vision Transformer https://arxiv.org/abs/2010.11929
    def __init__(self, c1, c2, num_heads, num_layers):
        super().__init__()
        self.conv = None
        if c1 != c2:
            self.conv = Conv(c1, c2)
        self.linear = nn.Linear(c2, c2)  # learnable position embedding
        self.tr = nn.Sequential(*(TransformerLayer(c2, num_heads) for _ in range(num_layers)))
        self.c2 = c2

    def forward(self, x):
        if self.conv is not None:
            x = self.conv(x)
        b, _, w, h = x.shape
        p = x.flatten(2).permute(2, 0, 1)
        return self.tr(p + self.linear(p)).permute(1, 2, 0).reshape(b, self.c2, w, h)


class snn_resnet:
    def __init__(self):
        super().__init__()


class CSABlock:
    def __init__(self):
        super().__init__()


class LIAFBlock:
    def __init__(self):
        super().__init__()


class Conv_LIAF:
    def __init__(self):
        super().__init__()


# class Bottleneck_2:
#     def __init__(self):
#         super().__init__()

class TCSABlock:
    def __init__(self):
        super().__init__()


class BasicTCSA:
    def __init__(self):
        super().__init__()


class HAMBlock:
    def __init__(self):
        super().__init__()


class ConcatCSA_res2:
    def __init__(self):
        super().__init__()


class BasicBlock_ms1:
    def __init__(self):
        super().__init__()


class MHSA(nn.Module):
    def __init__(self, in_channel, out_channel, num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.0,
                 proj_drop=0.0,
                 sr_ratio=1, ):
        super(MHSA, self).__init__()

        self.num_heads = num_heads
        self.out_channel = out_channel
        self.scale = 0.125
        self.m = mem_update(act=False)
        # self.head_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")
        self.query = nn.Sequential(
            Snn_Conv2d(in_channel, out_channel, kernel_size=1, stride=1, bias=False),
            batch_norm_2d(out_channel),
        )
        self.key = nn.Sequential(
            Snn_Conv2d(in_channel, out_channel, kernel_size=1, stride=1, bias=False),
            batch_norm_2d(out_channel),
        )
        self.value = nn.Sequential(
            Snn_Conv2d(in_channel, out_channel, kernel_size=1, stride=1, bias=False),
            batch_norm_2d(out_channel),
        )
        # self.q_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")
        #
        # self.k_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")
        #
        # self.v_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")
        #
        # self.attn_lif = MultiStepLIFNode(
        #     tau=2.0, v_threshold=0.5, detach_reset=True, backend="cupy"
        # )

        self.proj_conv = nn.Sequential(
            Snn_Conv2d(out_channel, out_channel, kernel_size=1, stride=1), batch_norm_2d(out_channel),
        )

    def forward(self, x):
        T, B, C, H, W = x.size()
        N = H * W
        x = self.m(x)
        q = self.query(x)
        q = self.m(q).flatten(3)
        q = (
            q.transpose(-1, -2)
            .reshape(T, B, N, self.num_heads, self.out_channel // self.num_heads)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )
        k = self.key(x)
        k = self.m(k).flatten(3)
        k = (
            k.transpose(-1, -2)
            .reshape(T, B, N, self.num_heads, self.out_channel // self.num_heads)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )
        v = self.value(x)
        v = self.m(v).flatten(3)
        v = (
            v.transpose(-1, -2)
            .reshape(T, B, N, self.num_heads, self.out_channel // self.num_heads)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )
        x = k.transpose(-2, -1) @ v
        x = (q @ x) * self.scale

        x = x.transpose(3, 4).reshape(T, B, self.out_channel, N).contiguous()
        x = self.m(x).reshape(T, B, self.out_channel, H, W)
        x = x.reshape(T, B, self.out_channel, H, W)
        out = self.proj_conv(x).reshape(T, B, self.out_channel, H, W)

        return out


class BottleneckTransformer(nn.Module):
    # Transformer bottleneck
    # expansion = 1

    def __init__(self, c1, c2, stride=1, heads=4, mhsa=True, resolution=None, expansion=1):
        super(BottleneckTransformer, self).__init__()
        c_ = int(c2 * expansion)
        # self.cv1 = Conv(c1, c_, 1, 1)
        self.cv1 = nn.Sequential(
            mem_update(False),
            Snn_Conv2d(c1, c_, 1, 1),
            batch_norm_2d(c_)
        )
        # self.bn1 = nn.BatchNorm2d(c2)
        if not mhsa:
            self.cv2 = nn.Sequential(
                mem_update(False),
                Snn_Conv2d(c_, c2, 1, 1),
                batch_norm_2d1(c2)
            )
        else:
            self.cv2 = nn.ModuleList()
            self.cv2.append(MHSA(c2, width=int(resolution[0]), height=int(resolution[1]), heads=heads))
            if stride == 2:
                self.cv2.append(nn.AvgPool2d(2, 2))
            self.cv2 = nn.Sequential(*self.cv2)
        self.shortcut = c1 == c2
        if stride != 1 or c1 != expansion * c2:
            self.shortcut = nn.Sequential(
                nn.MaxPool3d((1, stride, stride), stride=(1, stride, stride)),
                mem_update(False),
                Snn_Conv2d(c1, expansion * c2, kernel_size=1, stride=stride),
                batch_norm_2d(expansion * c2)
            )
        self.cv3 = nn.Sequential(
            mem_update(False),
            Snn_Conv2d(c2, c_, 1, 1),
            batch_norm_2d(c_)
        )

    def forward(self, x):
        out = x + self.cv2(self.cv1(x)) if self.shortcut else self.cv2(self.cv1(x))
        # out = x + self.cv3(self.cv2(self.cv1(x))) if self.shortcut else self.cv3(self.cv2(self.cv1(x)))
        return out


class BoT3(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, e=0.5, e2=1, w=20, h=20):  # ch_in, ch_out, number, , expansion,w,h
        super(BoT3, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv1 = nn.Sequential(
            mem_update(False),
            Snn_Conv2d(c1, c_, 1, 1),
            batch_norm_2d(c_)
        )
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Sequential(
            mem_update(False),
            Snn_Conv2d(c1, c_, 1, 1),
            batch_norm_2d(c_)
        )
        # self.cv3 = Conv(2 * c_, c2, 1)  # act=FReLU(c2)
        self.cv3 = nn.Sequential(
            mem_update(False),
            Snn_Conv2d(2 * c_, c2, 1, 1),
            batch_norm_2d(c2)
        )
        self.m = nn.Sequential(
            *[BottleneckTransformer(c_, c_, stride=1, heads=4, mhsa=True, resolution=(w, h), expansion=e2) for _ in
              range(n)])
        # self.m = nn.Sequential(*[CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)])

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=2))


class RepConvN(nn.Module):
    """RepConv is a basic rep-style block, including training and deploy status
    This code is based on https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
    """
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=3, s=1, p=1, g=1, d=1, act=True, bn=False, deploy=False):
        super().__init__()
        assert k == 3 and p == 1
        self.g = g
        self.c1 = c1
        self.c2 = c2
        # self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        self.m = mem_update(act=False)
        assert k == 3
        assert autopad(k, p) == 1

        padding_11 = autopad(k, p) - k // 2

        self.bn = None
        self.conv1 = nn.Sequential(
            Snn_Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False),
            batch_norm_2d(num_features=c2),
        )
        self.conv2 = nn.Sequential(
            Snn_Conv2d(c1, c2, 1, s, padding_11, groups=g, bias=False),
            batch_norm_2d(num_features=c2),
        )

    def forward_fuse(self, x):
        """Forward process"""
        x = self.m(x)
        return self.conv(x)

    def forward(self, x):
        """Forward process"""
        id_out = 0 if self.bn is None else self.bn(x)
        x = self.m(x)
        return self.conv1(x) + self.conv2(x) + id_out

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
        kernelid, biasid = self._fuse_bn_tensor(self.bn)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _avg_to_3x3_tensor(self, avgp):
        channels = self.c1
        groups = self.g
        kernel_size = avgp.kernel_size
        input_dim = channels // groups
        k = torch.zeros((channels, input_dim, kernel_size, kernel_size))
        k[np.arange(channels), np.tile(np.arange(input_dim), groups), :, :] = 1.0 / kernel_size ** 2
        return k

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, Conv):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        elif isinstance(branch, batch_norm_2d):
            if not hasattr(self, 'id_tensor'):
                input_dim = self.c1 // self.g
                kernel_value = np.zeros((self.c1, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.c1):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def fuse_convs(self):
        if hasattr(self, 'conv'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv = nn.Sequential(
            mem_update(False),
            Snn_Conv2d(in_channels=self.conv1.conv.in_channels,
                       out_channels=self.conv1.conv.out_channels,
                       kernel_size=self.conv1.conv.kernel_size,
                       stride=self.conv1.conv.stride,
                       padding=self.conv1.conv.padding,
                       dilation=self.conv1.conv.dilation,
                       groups=self.conv1.conv.groups,
                       bias=True).requires_grad_(False))
        self.conv.weight.data = kernel
        self.conv.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('conv1')
        self.__delattr__('conv2')
        if hasattr(self, 'nm'):
            self.__delattr__('nm')
        if hasattr(self, 'bn'):
            self.__delattr__('bn')
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')


class RepNBottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):  # ch_in, ch_out, shortcut, kernels, groups, expand
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = RepConvN(c1, c_, k[0], 1)
        self.cv2 = Conv_3(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2
        # if stride != 1 or c1 != c2:
        #     self.shortcut = nn.Sequential(
        #         nn.MaxPool3d((1, stride, stride), stride=(1, stride, stride)),
        #         mem_update(act=False),
        #         Snn_Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
        #         batch_norm_2d(out_channels),
        #     )

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class RepNCSP(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv_3(c1, c_, 1, 1)
        self.cv2 = Conv_3(c1, c_, 1, 1)
        self.cv3 = nn.Sequential(
            mem_update(False),
            Snn_Conv2d(2 * c_, c2, 1),
            batch_norm_2d(c2)
        )
        self.m = nn.Sequential(*(RepNBottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 2))


class RepNCSPELAN4(nn.Module):
    # csp-elan
    def __init__(self, c1, c2, c3, c4, c5=1):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = c3 // 2
        self.cv1 = Conv_3(c1, c3, 1, 1)
        self.cv2 = nn.Sequential(RepNCSP(c3 // 2, c4, c5), Conv_3(c4, c4, 3, 1))
        self.cv3 = nn.Sequential(RepNCSP(c4, c4, c5), Conv_3(c4, c4, 3, 1))
        self.cv4 = Conv_3(c3 + (2 * c4), c2, 1, 1)

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 2))
        y.extend((m(y[-1])) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 2))

    def forward_split(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 2))
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 2))


class BasicELAN4(nn.Module):
    # csp-elan
    def __init__(self, c1, c2, c3, c4, c5=1, s=1):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        if s == 2:
            self.dw = nn.Sequential(
                mem_update(act=False),
                Snn_Conv2d(c1, c1, kernel_size=3, stride=s, padding=1, bias=False),
                batch_norm_2d(c1))
        self.c = c3 // 2
        self.cv1 = BasicBlock_2(c1, c3, 1, 1)
        self.cv2 = nn.Sequential(BasicBlock_2C3(c3 // 2, c4, c5), Conv_3(c4, c4, 3, 1))
        self.cv3 = nn.Sequential(BasicBlock_2C3(c4, c4, c5), Conv_3(c4, c4, 3, 1))
        self.cv4 = BasicBlock_2(c3 + (2 * c4), c2, 1, 1)

    def forward(self, x):
        if self.dw:
            x = self.dw(x)
        y = list(self.cv1(x).chunk(2, 2))
        y.extend((m(y[-1])) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 2))

    def forward_split(self, x):
        if self.dw:
            x = self.dw(x)
        y = list(self.cv1(x).split((self.c, self.c), 2))
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 2))


class BasicBlock_2C3(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, k=3, s=1, e=0.5):  # ch_in, ch_out, number, kernel_size, stride, expansion
        """在C3模块和yolo.py的parse_model模块调用
            CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
            :params c1: 整个BottleneckCSP的输入channel
            :params c2: 整个BottleneckCSP的输出channel
            :params n: 有n个BasicBlock_2
            :params s:
            :params g: Bottleneck中的3x3卷积类型  =1普通卷积  >1深度可分离卷积
            :params e: expansion ratio c2xe=中间其他所有层的卷积核个数/中间所有层的输入输出channel数
            c_: bottleneckCSP 结构的中间层的通道数，由膨胀率e决定
            """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.dw = None
        if s == 2:
            self.dw = nn.Sequential(
                mem_update(act=False),
                Snn_Conv2d(c1, c1, kernel_size=k, stride=s, padding=1, bias=False),
                batch_norm_2d(c1))
        # 4个1*1卷积层的堆叠
        self.cv1 = Conv_3(c1, c_, 1, 1)
        self.cv2 = Conv_3(c1, c_, 1, 1)
        self.cv3 = Conv_4(2 * c_, c2, 1)
        self.m = nn.Sequential(*(BasicBlock_2(c_, c_, k_size=k, stride=1) for _ in range(n)))

    def forward(self, x):
        if self.dw:
            x = self.dw(x)
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=2))


class BasicBlock_1C3(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, s=1, e=0.5):  # ch_in, ch_out, number, kernel_size, stride, expansion
        """在C3模块和yolo.py的parse_model模块调用
            CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
            :params c1: 整个BottleneckCSP的输入channel
            :params c2: 整个BottleneckCSP的输出channel
            :params n: 有n个BasicBlock_2
            :params s:
            :params g: Bottleneck中的3x3卷积类型  =1普通卷积  >1深度可分离卷积
            :params e: expansion ratio c2xe=中间其他所有层的卷积核个数/中间所有层的输入输出channel数
            c_: bottleneckCSP 结构的中间层的通道数，由膨胀率e决定
            """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        # 4个1*1卷积层的堆叠
        self.cv1 = Conv_3(c1, c_, 1, 1)
        self.cv2 = Conv_3(c1, c_, 1, s=s)
        self.cv3 = Conv_3(2 * c_, c2, 1)
        self.m = nn.Sequential(*(BasicBlock_1(c_, c_, stride=s) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=2))


class Concat_res2C3(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, k=3, s=1, e=0.5):  # ch_in, ch_out, number, kernel_size, stride, expansion
        """在C3模块和yolo.py的parse_model模块调用
            CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
            :params c1: 整个BottleneckCSP的输入channel
            :params c2: 整个BottleneckCSP的输出channel
            :params n: 有n个BasicBlock_2
            :params s:
            :params g: Bottleneck中的3x3卷积类型  =1普通卷积  >1深度可分离卷积
            :params e: expansion ratio c2xe=中间其他所有层的卷积核个数/中间所有层的输入输出channel数
            c_: bottleneckCSP 结构的中间层的通道数，由膨胀率e决定
            """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        cc = int(c_ * e)
        # 4个1*1卷积层的堆叠
        self.cv1 = Conv_3(c1, cc, 1, 1)
        self.cv2 = Conv_3(c1, c_, 1, s=s)
        self.cv3 = Conv_3(2 * c_, c2, 1)
        self.m = nn.Sequential(*(Concat_res2(cc, c_, k_size=k, stride=s) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=2))


class CoT(nn.Module):
    # Contextual Transformer Networks https://arxiv.org/abs/2107.12292
    def __init__(self, dim=512, kernel_size=3):
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size

        self.head_lif = mem_update(act=False)
        self.key_embed = nn.Sequential(
            Snn_Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=4, bias=False),
            batch_norm_2d(dim),
        )
        self.value_embed = nn.Sequential(
            Snn_Conv2d(dim, dim, 1, bias=False),
            batch_norm_2d(dim)
        )

        factor = 4
        self.attention_embed = nn.Sequential(
            mem_update(act=False),
            Snn_Conv2d(2 * dim, 2 * dim // factor, 1, bias=False),
            batch_norm_2d(2 * dim // factor),
            mem_update(act=False),
            Snn_Conv2d(2 * dim // factor, kernel_size * kernel_size * dim, 1)
        )

    def forward(self, x):
        T, bs, c, h, w = x.shape
        x = self.head_lif(x)
        k1 = self.key_embed(x)  # T,bs,c,h,w
        v = self.value_embed(x).view(T, bs, c, -1)  # T,bs,c,h,w

        y = torch.cat([k1, x], dim=2)  # T,bs,2c,h,w
        att = self.attention_embed(y)  # bs,c*k*k,h,w
        att = att.reshape(bs, c, self.kernel_size * self.kernel_size, h, w)
        att = att.mean(2, keepdim=False).view(T, bs, c, -1)  # bs,c,h*w
        k2 = att * v
        k2 = k2.view(T, bs, c, h, w)

        return k1 + k2


# class EMA(nn.Module):
#     def __init__(self, channels, factor=8):
#         super(EMA, self).__init__()
#         self.groups = factor
#         assert channels // self.groups > 0
#         self.softmax = nn.Softmax(-1)
#         self.m = mem_update(act=False)
#         self.agp = nn.AdaptiveAvgPool2d((1, 1))
#         self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
#         self.pool_w = nn.AdaptiveAvgPool2d((1, None))
#         self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
#         self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
#         self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)
#
#     def forward(self, x):
#         t, b, c, h, w = x.size()
#         # x = self.m(x)
#         group_x = x.reshape(t * b * self.groups, -1, h, w)  # t*b*g,c//g,h,w
#         x_h = self.pool_h(group_x)
#         x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
#         hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
#         # hw = self.m(hw)
#         x_h, x_w = torch.split(hw, [h, w], dim=2)
#         x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
#         x2 = self.conv3x3(group_x)
#         x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
#         x12 = x2.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
#         x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
#         x22 = x1.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
#         weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
#         return (group_x * weights.sigmoid()).reshape(t, b, c, h, w)


class EMA(nn.Module):
    def __init__(self, channels, factor=8):
        super(EMA, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.m = mem_update(act=False)
        self.agp = nn.AdaptiveAvgPool3d((None, 1, 1))
        self.pool_h = nn.AdaptiveAvgPool3d((None, None, 1))
        self.pool_w = nn.AdaptiveAvgPool3d((None, 1, None))
        self.gn = GN(channels // self.groups, channels // self.groups)
        self.conv1x1 = Snn_Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = Snn_Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        t, b, c, h, w = x.size()
        # x = self.m(x)
        group_x = x.reshape(t, b * self.groups, -1, h, w)  # t*b*g,c//g,h,w
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 2, 4, 3)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=3))
        # hw = self.m(hw)
        x_h, x_w = torch.split(hw, [h, w], dim=3)
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 2, 4, 3).sigmoid())
        x2 = self.conv3x3(group_x)
        x11 = self.softmax(self.agp(x1).reshape(t, b * self.groups, -1, 1).permute(0, 1, 3, 2))
        x12 = x2.reshape(t, b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        x21 = self.softmax(self.agp(x2).reshape(t, b * self.groups, -1, 1).permute(0, 1, 3, 2))
        x22 = x1.reshape(t, b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(t, b * self.groups, 1, h, w)
        return (group_x * weights.sigmoid()).reshape(t, b, c, h, w)


class GN(nn.Module):
    def __init__(self, num_groups, channels):
        super().__init__()
        self.gn = nn.GroupNorm(num_groups, channels)

    def forward(self, x):
        y = x.transpose(0, 2).contiguous().transpose(0, 1).contiguous()
        y = self.gn(y)
        return y.contiguous().transpose(0, 1).contiguous().transpose(0, 2)  #


class BasicBlock_2C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, add=True, k=3, s=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.s = s
        if self.s == 2:
            self.cv = Conv_3(c1, c1, 3, 2)
        self.cv1 = Conv_3(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv_4((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(BasicBlock_2(self.c, self.c, k_size=k, stride=1, add=add) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        if self.s == 2:
            x = self.cv(x)
        y = list(self.cv1(x).chunk(2, 2))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 2))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        if self.s == 2:
            x = self.cv(x)
        y = list(self.cv1(x).split((self.c, self.c), 2))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 2))


class BasicBlock_1C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, s=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.s = s
        if self.s == 2:
            self.cv = Conv_3(c1, c1, 3, 2)
        self.cv1 = Conv_3(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv_3((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(BasicBlock_1(self.c, self.c, stride=1) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        if self.s == 2:
            x = self.cv(x)
        y = list(self.cv1(x).chunk(2, 2))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 2))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        if self.s == 2:
            x = self.cv(x)
        y = list(self.cv1(x).split((self.c, self.c), 2))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 2))


class Concat_res2C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, k=3, s=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        if s == 2:
            self.cv = Conv_3(c1, c1, 3, 2)
        else:
            self.cv = None
        self.cv1 = Conv_3(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv_3((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Concat_res2(self.c, self.c, k_size=k, stride=s) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 2))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 2))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 2))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 2))


class Attention(nn.Module):
    def __init__(self, dim, num_heads):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.m = mem_update(act=False)

        self.norm1 = batch_norm_2d(dim)
        self.project_out = Snn_Conv2d(dim, dim, kernel_size=1)
        self.conv0_1 = Snn_Conv2d(dim, dim, (1, 7), padding=(0, 3), groups=dim)
        self.conv0_2 = Snn_Conv2d(dim, dim, (7, 1), padding=(3, 0), groups=dim)
        self.conv1_1 = Snn_Conv2d(dim, dim, (1, 11), padding=(0, 5), groups=dim)
        self.conv1_2 = Snn_Conv2d(dim, dim, (11, 1), padding=(5, 0), groups=dim)
        self.conv2_1 = Snn_Conv2d(
            dim, dim, (1, 21), padding=(0, 10), groups=dim)
        self.conv2_2 = Snn_Conv2d(
            dim, dim, (21, 1), padding=(10, 0), groups=dim)

    def forward(self, x):
        t, b, c, h, w = x.shape
        x = self.m(x)
        x1 = self.norm1(x)
        attn_00 = self.conv0_1(x1)
        attn_00 = self.m(attn_00)
        attn_01 = self.conv0_2(x1)
        attn_01 = self.m(attn_01)
        attn_10 = self.conv1_1(x1)
        attn_10 = self.m(attn_10)
        attn_11 = self.conv1_2(x1)
        attn_11 = self.m(attn_11)
        attn_20 = self.conv2_1(x1)
        attn_20 = self.m(attn_20)
        attn_21 = self.conv2_2(x1)
        attn_21 = self.m(attn_21)
        out1 = attn_00 + attn_10 + attn_20
        out2 = attn_01 + attn_11 + attn_21
        out1 = self.project_out(out1)
        out1 = self.m(out1)
        out2 = self.project_out(out2)
        out2 = self.m(out2)
        k1 = rearrange(out1, 't b (head c) h w -> t b head h (w c)', head=self.num_heads)
        v1 = rearrange(out1, 't b (head c) h w -> t b head h (w c)', head=self.num_heads)
        k2 = rearrange(out2, 't b (head c) h w -> t b head w (h c)', head=self.num_heads)
        v2 = rearrange(out2, 't b (head c) h w -> t b head w (h c)', head=self.num_heads)
        q2 = rearrange(out1, 't b (head c) h w -> t b head w (h c)', head=self.num_heads)
        q1 = rearrange(out2, 't b (head c) h w -> t b head h (w c)', head=self.num_heads)
        # q1 = torch.nn.functional.normalize(q1, dim=-1)
        # q2 = torch.nn.functional.normalize(q2, dim=-1)
        # k1 = torch.nn.functional.normalize(k1, dim=-1)
        # k2 = torch.nn.functional.normalize(k2, dim=-1)
        attn1 = (q1 @ k1.transpose(-2, -1))
        out3 = (attn1 @ v1) + q1
        attn2 = (q2 @ k2.transpose(-2, -1))
        out4 = (attn2 @ v2) + q2
        out3 = rearrange(out3, 't b head h (w c) -> t b (head c) h w', head=self.num_heads, h=h, w=w)
        out4 = rearrange(out4, 't b head w (h c) -> t b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out3) + self.project_out(out4) + x

        return out


from torch.nn.common_types import _size_2_t


class SpikingMatmul(nn.Module):
    def __init__(self, spike: str) -> None:
        super().__init__()
        assert spike == 'l' or spike == 'r' or spike == 'both'
        self.spike = spike

    def forward(self, left: torch.Tensor, right: torch.Tensor):
        return torch.matmul(left, right)


class DSSA(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        # self.lenth = lenth
        self.register_buffer('firing_rate_x', torch.zeros(1, 1, num_heads, 1, 1))
        self.register_buffer('firing_rate_attn', torch.zeros(1, 1, num_heads, 1, 1))
        self.init_firing_rate_x = False
        self.init_firing_rate_attn = False
        self.momentum = 0.999

        self.activation_in = mem_update(act=False)

        self.W = Snn_Conv2d(dim, dim * 2, 3, 1, bias=False)
        self.norm = batch_norm_2d(dim * 2)
        self.matmul1 = SpikingMatmul('r')
        self.matmul2 = SpikingMatmul('r')
        self.activation_attn = mem_update(act=False)
        self.activation_out = mem_update(act=False)

        self.Wproj = Snn_Conv2d(dim, dim, kernel_size=1, stride=1)
        self.norm_proj = batch_norm_2d(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # X: [T, B, C, H, W]
        T, B, C, H, W = x.shape
        x_feat = x.clone()
        x = self.activation_in(x)

        y = self.W(x)
        y = self.norm(y)
        y = y.reshape(T, B, self.num_heads, 2 * C // self.num_heads, -1)
        y1, y2 = y[:, :, :, :C // self.num_heads, :], y[:, :, :, C // self.num_heads:, :]
        x = x.reshape(T, B, self.num_heads, C // self.num_heads, -1)

        if self.training:
            firing_rate_x = x.detach().mean((0, 1, 3, 4), keepdim=True)
            if not self.init_firing_rate_x and torch.all(self.firing_rate_x == 0):
                self.firing_rate_x = firing_rate_x
            self.init_firing_rate_x = True
            self.firing_rate_x = self.firing_rate_x * self.momentum + firing_rate_x * (
                    1 - self.momentum)
        scale1 = 1. / torch.sqrt(self.firing_rate_x * (self.dim // self.num_heads))
        attn = self.matmul1(y1.transpose(-1, -2), x)
        attn = attn * scale1
        attn = self.activation_attn(attn)

        if self.training:
            firing_rate_attn = attn.detach().mean((0, 1, 3, 4), keepdim=True)
            if not self.init_firing_rate_attn and torch.all(self.firing_rate_attn == 0):
                self.firing_rate_attn = firing_rate_attn
            self.init_firing_rate_attn = True
            self.firing_rate_attn = self.firing_rate_attn * self.momentum + firing_rate_attn * (
                    1 - self.momentum)
        scale2 = 1. / torch.sqrt(self.firing_rate_attn)
        out = self.matmul2(y2, attn)
        out = out * scale2
        out = out.reshape(T, B, C, H, W)
        out = self.activation_out(out)

        out = self.Wproj(out)
        out = self.norm_proj(out)
        out = out + x_feat
        return out


class OSRAAttention(nn.Module):  ### OSRA
    def __init__(self, dim,
                 num_heads=1,
                 qk_scale=None,
                 attn_drop=0,
                 sr_ratio=1, ):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.sr_ratio = sr_ratio
        self.q = Snn_Conv2d(dim, dim, kernel_size=1)
        self.kv = Snn_Conv2d(dim, dim * 2, kernel_size=1)
        self.attn_drop = nn.Dropout(attn_drop)
        if sr_ratio > 1:
            self.sr = nn.Sequential(
                mem_update(act=False),
                Snn_Conv2d(dim, dim,
                           kernel_size=sr_ratio + 3,
                           stride=sr_ratio,
                           padding=(sr_ratio + 3) // 2,
                           groups=dim,
                           bias=False),
                batch_norm_2d(dim),
                mem_update(act=False),
                Snn_Conv2d(dim, dim,
                           kernel_size=1,
                           groups=dim,
                           bias=False),
                batch_norm_2d1(dim),
            )
        else:
            self.sr = nn.Identity()
        self.local_conv = Snn_Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.m = mem_update(act=False)

    def forward(self, x, relative_pos_enc=None):
        T, B, C, H, W = x.shape
        x = self.m(x)
        q = self.q(x).reshape(T, B, self.num_heads, C // self.num_heads, -1).transpose(-1, -2)
        q = self.m(q)
        kv = self.sr(x)
        kv = self.m(kv)
        kv = self.local_conv(kv) + kv
        kv = self.m(kv)
        k, v = torch.chunk(self.kv(kv), chunks=2, dim=2)
        k = k.reshape(T, B, self.num_heads, C // self.num_heads, -1)
        v = v.reshape(T, B, self.num_heads, C // self.num_heads, -1).transpose(-1, -2)
        attn = (q @ k) * self.scale
        if relative_pos_enc is not None:
            if attn.shape[3:] != relative_pos_enc.shape[3:]:
                relative_pos_enc = F.interpolate(relative_pos_enc, size=attn.shape[2:],
                                                 mode='bicubic', align_corners=False)
            attn = attn + relative_pos_enc
        attn = self.attn_drop(attn)
        attn = self.m(attn)
        x = (attn @ v).transpose(-1, -2)
        return x.reshape(T, B, C, H, W)


class DynamicConv2d(nn.Module):  ### IDConv
    def __init__(self,
                 dim,
                 kernel_size=3,
                 reduction_ratio=4,
                 num_groups=1,
                 bias=True):
        super().__init__()
        assert num_groups > 1, f"num_groups {num_groups} should > 1."

        self.num_groups = num_groups
        self.K = kernel_size
        self.bias_type = bias
        self.weight = nn.Parameter(torch.empty(num_groups, dim, kernel_size, kernel_size), requires_grad=True)
        self.pool = nn.AdaptiveAvgPool3d(output_size=(None, kernel_size, kernel_size))
        self.proj = nn.Sequential(
            mem_update(act=False),
            Snn_Conv2d(dim,
                       dim // reduction_ratio,
                       kernel_size=1),
            batch_norm_2d(dim // reduction_ratio),
            mem_update(act=False),
            Snn_Conv2d(dim // reduction_ratio, dim * num_groups, kernel_size=1))

        if bias:
            self.bias = nn.Parameter(torch.empty(num_groups, dim), requires_grad=True)
        else:
            self.bias = None

        self.reset_parameters()
        self.m = mem_update(act=False)

    def reset_parameters(self):
        nn.init.trunc_normal_(self.weight, std=0.02)
        if self.bias is not None:
            nn.init.trunc_normal_(self.bias, std=0.02)

    def forward(self, x):

        T, B, C, H, W = x.shape

        x = self.m(x)

        scale = self.proj(self.pool(x)).reshape(T * B, self.num_groups, C, self.K, self.K)
        # scale = torch.softmax(scale, dim=1)
        scale = self.m(scale)
        weight = scale * self.weight.unsqueeze(0)
        weight = torch.sum(weight, dim=1, keepdim=False)
        weight = weight.reshape(-1, 1, self.K, self.K)

        if self.bias is not None:
            # print(x.shape)
            # print(torch.mean(x, dim=[-2, -1], keepdim=True).shape)
            scale = self.proj(torch.mean(x, dim=[-2, -1], keepdim=True))
            # scale = torch.softmax(scale.reshape(B, self.num_groups, C), dim=1)
            scale = self.m(scale.reshape(T * B, self.num_groups, C))
            bias = scale * self.bias.unsqueeze(0)
            bias = torch.sum(bias, dim=1).flatten(0)
        else:
            bias = None

        x = F.conv2d(x.reshape(1, -1, H, W),
                     weight=weight,
                     padding=self.K // 2,
                     groups=B * C,
                     bias=bias)

        return x.reshape(T, B, C, H, W)


class TransxnetHybridTokenMixer(nn.Module):  ### D-Mixer
    def __init__(self,
                 dim,
                 kernel_size=3,
                 num_groups=2,
                 num_heads=1,
                 sr_ratio=1,
                 reduction_ratio=8):
        super().__init__()
        assert dim % 2 == 0, f"dim {dim} should be divided by 2."

        self.local_unit = DynamicConv2d(
            dim=dim // 2, kernel_size=kernel_size, num_groups=num_groups)
        self.global_unit = OSRAAttention(
            dim=dim // 2, num_heads=num_heads, sr_ratio=sr_ratio)

        inner_dim = max(16, dim // reduction_ratio)
        self.proj = nn.Sequential(
            mem_update(act=False),
            Snn_Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim),
            batch_norm_2d(dim),
            mem_update(act=False),
            Snn_Conv2d(dim, inner_dim, kernel_size=1),
            batch_norm_2d(inner_dim),
            mem_update(act=False),
            Snn_Conv2d(inner_dim, dim, kernel_size=1),
            batch_norm_2d1(dim), )

    def forward(self, x, relative_pos_enc=None):
        x1, x2 = torch.chunk(x, chunks=2, dim=2)
        # x = x.permute(1, 0, 2, 3)
        x1 = self.local_unit(x1)
        x2 = self.global_unit(x2, relative_pos_enc)
        x = torch.cat([x1, x2], dim=2)
        x = self.proj(x) + x  ## STE
        return x


class RepConv(nn.Module):
    """
    RepConv is a basic rep-style block, including training and deploy status.

    This module is used in RT-DETR.
    Based on https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
    """
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=3, s=1, p=1, g=1, d=1, act=True, bn=False, deploy=False):
        """Initializes Light Convolution layer with inputs, outputs & optional activation function."""
        super().__init__()
        # assert k == 3 and p == 1
        self.g = g
        self.c1 = c1
        self.c2 = c2
        self.act = mem_update(act=False) if act is True else act if isinstance(act, nn.Module) else nn.Identity()

        self.bn = batch_norm_2d(num_features=c1) if bn and c2 == c1 and s == 1 else None
        self.conv1 = Conv_2(c1, c2, k, s, p=p, g=g, act=False)
        self.conv2 = Conv_2(c1, c2, 1, s, p=(p - k // 2), g=g, act=False)

    def forward_fuse(self, x):
        """Forward process."""
        return self.conv(self.act(x))

    def forward(self, x):
        """Forward process."""
        id_out = 0 if self.bn is None else self.bn(x)
        x = self.act(x)
        return self.conv1(x) + self.conv2(x) + id_out

    def get_equivalent_kernel_bias(self):
        """Returns equivalent kernel and bias by adding 3x3 kernel, 1x1 kernel and identity kernel with their biases."""
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
        kernelid, biasid = self._fuse_bn_tensor(self.bn)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        """Pads a 1x1 tensor to a 3x3 tensor."""
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        """Generates appropriate kernels and biases for convolution by fusing branches of the neural network."""
        if branch is None:
            return 0, 0
        if isinstance(branch, Conv_2):
            kernel = branch.conv.weight
            running_mean = branch.bn.bn.running_mean
            running_var = branch.bn.bn.running_var
            gamma = branch.bn.bn.weight
            beta = branch.bn.bn.bias
            eps = branch.bn.bn.eps
        elif isinstance(branch, batch_norm_2d):
            if not hasattr(self, 'id_tensor'):
                input_dim = self.c1 // self.g
                kernel_value = np.zeros((self.c1, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.c1):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def fuse_convs(self):
        """Combines two convolution layers into a single layer and removes unused attributes from the class."""
        if hasattr(self, 'conv'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv = Snn_Conv2d(in_channels=self.conv1.conv.in_channels,
                               out_channels=self.conv1.conv.out_channels,
                               kernel_size=self.conv1.conv.kernel_size,
                               stride=self.conv1.conv.stride,
                               padding=self.conv1.conv.padding,
                               dilation=self.conv1.conv.dilation,
                               groups=self.conv1.conv.groups,
                               bias=True).requires_grad_(False)
        self.conv.weight.data = kernel
        self.conv.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('conv1')
        self.__delattr__('conv2')
        if hasattr(self, 'nm'):
            self.__delattr__('nm')
        if hasattr(self, 'bn'):
            self.__delattr__('bn')
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')


class Silence(nn.Module):
    def __init__(self):
        super(Silence, self).__init__()

    def forward(self, x):
        # print("Silence:")
        # print(x.dtype)
        return x


class ResNetBlock(nn.Module):
    """ResNet block with standard convolution layers."""

    def __init__(self, c1, c2, s=1, e=4):
        """Initialize convolution with given parameters."""
        super().__init__()
        c3 = e * c2
        self.cv1 = Conv(c1, c2, k=1, s=1, act=True)
        self.cv2 = Conv(c2, c2, k=3, s=s, p=1, act=True)
        self.cv3 = Conv(c2, c3, k=1, act=False)
        self.shortcut = nn.Sequential(Conv(c1, c3, k=1, s=s, act=False)) if s != 1 or c1 != c3 else nn.Identity()

    def forward(self, x):
        """Forward pass through the ResNet block."""
        return F.relu(self.cv3(self.cv2(self.cv1(x))) + self.shortcut(x))


class ResNetLayerBasic(nn.Module):
    """ResNet layer with multiple ResNet blocks."""

    def __init__(self, c1, c2, k=1, s=1, is_first=False, n=1):
        """Initializes the ResNetLayer given arguments."""
        super().__init__()
        self.is_first = is_first

        if self.is_first:
            # self.layer = nn.Sequential(Conv_1(c1, c2, k=3, s=2, p=1, act=True),
            #                            Conv_1(c2, c2, k=3, s=1, p=1, act=True),
            #                            Conv_1(c2, c2, k=3, s=1, p=1, act=True))
            self.layer = nn.Sequential(Conv_1(c1, c2, k=7, s=2, p=3, act=True))
        else:
            blocks = [GhostBottleneck(c1, c2, k, s)]
            blocks.extend([GhostBottleneck(c2, c2, k, 1) for _ in range(n - 1)])
            # blocks = [BasicBlock_2C3(c1, c2, n, k, s)]
            self.layer = nn.Sequential(*blocks)

    def forward(self, x):
        """Forward pass through the ResNet layer."""
        return self.layer(x)


class ResNetLayerBo(nn.Module):
    """ResNet layer with multiple ResNet blocks."""

    def __init__(self, c1, c2, k=1, s=1, is_first=False, n=1, e=4):
        """Initializes the ResNetLayer given arguments."""
        super().__init__()
        self.is_first = is_first

        if self.is_first:
            self.layer = nn.Sequential(Conv_1(c1, c2, k=7, s=2, p=3, act=True),
                                       nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        else:
            # blocks = [Bottleneck_2(c1, c2, k, s, e=e)]
            # blocks.extend([Bottleneck_2(e * c2, c2, k, 1, e=e) for _ in range(n - 1)])
            blocks = [C3(c1, c2, n, k, s)]
            self.layer = nn.Sequential(*blocks)

    def forward(self, x):
        """Forward pass through the ResNet layer."""
        return self.layer(x)


class ELA(nn.Module):
    def __init__(self, channel, kernel_size=7):
        super(ELA, self).__init__()
        self.pad = kernel_size // 2
        self.conv = nn.Conv1d(channel, channel, kernel_size=kernel_size, padding=self.pad, groups=channel, bias=False)
        self.gn = nn.GroupNorm(16, channel)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        t, b, c, h, w = x.size()

        # 处理高度维度
        x_h = torch.mean(x, dim=4, keepdim=True).view(t * b, c, h)
        x_h = self.sigmoid(self.gn(self.conv(x_h))).view(t, b, c, h, 1)

        # 处理宽度维度
        x_w = torch.mean(x, dim=3, keepdim=True).view(t * b, c, w)
        x_w = self.sigmoid(self.gn(self.conv(x_w))).view(t, b, c, 1, w)

        # print(x_h.shape, x_w.shape)
        # 在两个维度上应用注意力
        return x * x_h * x_w


class DepthWiseConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=False):
        # 这一行千万不要忘记
        super(DepthWiseConv, self).__init__()

        # 逐通道卷积
        self.depth_conv = Snn_Conv2d(in_channels=in_channel,
                                     out_channels=in_channel,
                                     kernel_size=kernel_size,
                                     stride=stride,
                                     padding=padding,
                                     groups=in_channel,
                                     bias=bias)
        # self.depth_conv = GSConv(in_channel, out_channel, k=kernel_size, s=stride, g=in_channel)
        # groups是一个数，当groups=in_channel时,表示做逐通道卷积

        # 逐点卷积
        # self.point_conv = Snn_Conv2d(in_channels=in_channel,
        #                              out_channels=out_channel,
        #                              kernel_size=1,
        #                              stride=1,
        #                              padding=0,
        #                              groups=1,
        #                              bias=bias)
        self.point_conv = GSConv(in_channel, out_channel, k=1, s=1, g=1)

    def forward(self, input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        return out


class PartialConv(nn.Module):
    def __init__(self, dim, n_div=4, kernel_size=3, dilation=1, forward='split_cat'):
        """
        PartialConv 模块
        Args:
            dim (int): 输入张量的通道数。
            n_div (int): 分割通道数的分母，用于确定部分卷积的通道数。
            forward (str): 使用的前向传播方法，可选 'slicing' 或 'split_cat'。
        """
        super().__init__()
        self.dim_conv3 = dim // n_div
        self.dim_untouched = dim - self.dim_conv3
        # self.partial_conv3 = nn.Sequential(
        #     Snn_Conv2d(self.dim_conv3, self.dim_conv3 * 4, 1),
        #     Snn_Conv2d(self.dim_conv3 * 4, self.dim_conv3 * 4, kernel_size, 1, 1, groups=self.dim_conv3, bias=False),
        #     Snn_Conv2d(self.dim_conv3 * 4, self.dim_conv3, 1)
        # )
        self.partial_conv3 = nn.Sequential(
            Snn_Conv2d(self.dim_conv3, self.dim_conv3, kernel_size, 1, padding=dilation, bias=False, dilation=dilation),
        )
        # self.partial_conv3 = nn.Sequential(
        #     RepConv(self.dim_conv3, self.dim_conv3, kernel_size, 1, 1)
        # )

        if forward == 'slicing':
            self.forward = self.forward_slicing
        elif forward == 'split_cat':
            self.forward = self.forward_split_cat
        else:
            raise NotImplementedError

    def forward_slicing(self, x):
        # only for inference
        x = x.clone()  # !!! Keep the original input intact for the residual connection later
        x[:, :, :self.dim_conv3, :, :] = self.partial_conv3(x[:, :, :self.dim_conv3, :, :])

        return x

    def forward_split_cat(self, x):
        # for training/inference
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=2)
        # print(x1.shape)
        x1 = self.partial_conv3(x1)
        x = torch.cat((x1, x2), 2)

        return x


class FasterNetBlock(nn.Module):
    def __init__(self, dim, expand_ratio=2, forward='split_cat'):
        super().__init__()
        self.pconv = PartialConv(dim, forward=forward)
        self.conv1 = Snn_Conv2d(dim, dim * expand_ratio, 1)
        self.bn = batch_norm_2d1(dim * expand_ratio)
        self.act_layer = mem_update(act=False)
        self.conv2 = Snn_Conv2d(dim * expand_ratio, dim, 1)

    def forward(self, x):
        residual = x
        x = self.pconv(x)
        x = self.conv1(x)
        x = self.bn(x)
        x = self.act_layer(x)
        x = self.conv2(x)
        return x + residual


# 自定义 GroupBatchnorm2d 类，实现分组批量归一化
class GroupBatchnorm2d(nn.Module):
    def __init__(self, c_num: int, group_num: int = 16, eps: float = 1e-10):
        super(GroupBatchnorm2d, self).__init__()  # 调用父类构造函数
        assert c_num >= group_num  # 断言 c_num 大于等于 group_num
        self.group_num = group_num  # 设置分组数量
        self.gamma = nn.Parameter(torch.randn(c_num, 1, 1))  # 创建可训练参数 gamma
        self.beta = nn.Parameter(torch.zeros(c_num, 1, 1))  # 创建可训练参数 beta
        self.eps = eps  # 设置小的常数 eps 用于稳定计算

    def forward(self, x):
        T, N, C, H, W = x.size()  # 获取输入张量的尺寸
        x = x.reshape(T * N, self.group_num, -1)  # 将输入张量重新排列为指定的形状
        mean = x.mean(dim=2, keepdim=True)  # 计算每个组的均值
        std = x.std(dim=2, keepdim=True)  # 计算每个组的标准差
        x = (x - mean) / (std + self.eps)  # 应用批量归一化
        x = x.reshape(T, N, C, H, W)  # 恢复原始形状
        return x * self.gamma + self.beta  # 返回归一化后的张量


# 自定义 SRU（Spatial and Reconstruct Unit）类
class SRU(nn.Module):
    def __init__(self,
                 oup_channels: int,  # 输出通道数
                 group_num: int = 16,  # 分组数，默认为16
                 gate_treshold: float = 0.5,  # 门控阈值，默认为0.5
                 torch_gn: bool = False  # 是否使用PyTorch内置的GroupNorm，默认为False
                 ):
        super().__init__()  # 调用父类构造函数

        # 初始化 GroupNorm 层或自定义 GroupBatchnorm2d 层
        self.gn = nn.GroupNorm(num_channels=oup_channels, num_groups=group_num) if torch_gn else GroupBatchnorm2d(
            c_num=oup_channels, group_num=group_num)
        self.gate_treshold = gate_treshold  # 设置门控阈值
        self.sigmoid = nn.Sigmoid()  # 创建 sigmoid 激活函数

    def forward(self, x):
        gn_x = self.gn(x)  # 应用分组批量归一化
        w_gamma = self.gn.gamma / sum(self.gn.gamma)  # 计算 gamma 权重
        reweights = self.sigmoid(gn_x * w_gamma)  # 计算重要性权重

        # 门控机制
        info_mask = reweights >= self.gate_treshold  # 计算信息门控掩码
        noninfo_mask = reweights < self.gate_treshold  # 计算非信息门控掩码
        x_1 = info_mask * x  # 使用信息门控掩码
        x_2 = noninfo_mask * x  # 使用非信息门控掩码
        x = self.reconstruct(x_1, x_2)  # 重构特征
        return x

    def reconstruct(self, x_1, x_2):
        x_11, x_12 = torch.split(x_1, x_1.size(2) // 2, dim=2)  # 拆分特征为两部分
        x_21, x_22 = torch.split(x_2, x_2.size(2) // 2, dim=2)  # 拆分特征为两部分
        return torch.cat([x_11 + x_22, x_12 + x_21], dim=2)  # 重构特征并连接


# 自定义 CRU（Channel Reduction Unit）类
class CRU(nn.Module):
    def __init__(self, op_channel: int, alpha: float = 1 / 2, squeeze_radio: int = 2, group_size: int = 2,
                 group_kernel_size: int = 3):
        super().__init__()  # 调用父类构造函数

        self.up_channel = up_channel = int(alpha * op_channel)  # 计算上层通道数
        self.low_channel = low_channel = op_channel - up_channel  # 计算下层通道数
        self.squeeze1 = Snn_Conv2d(up_channel, up_channel // squeeze_radio, kernel_size=1, bias=False)  # 创建卷积层
        self.squeeze2 = Snn_Conv2d(low_channel, low_channel // squeeze_radio, kernel_size=1, bias=False)  # 创建卷积层

        # 上层特征转换
        self.GWC = Snn_Conv2d(up_channel // squeeze_radio, op_channel, kernel_size=group_kernel_size, stride=1,
                              padding=group_kernel_size // 2, groups=group_size)  # 创建卷积层
        self.PWC1 = Snn_Conv2d(up_channel // squeeze_radio, op_channel, kernel_size=1, bias=False)  # 创建卷积层

        # 下层特征转换
        self.PWC2 = Snn_Conv2d(low_channel // squeeze_radio, op_channel - low_channel // squeeze_radio, kernel_size=1,
                               bias=False)  # 创建卷积层
        self.advavg = nn.AdaptiveAvgPool3d((None, 1, 1))  # 创建自适应平均池化层

    def forward(self, x):
        # 分割输入特征
        up, low = torch.split(x, [self.up_channel, self.low_channel], dim=2)
        up, low = self.squeeze1(up), self.squeeze2(low)

        # 上层特征转换
        Y1 = self.GWC(up) + self.PWC1(up)

        # 下层特征转换
        Y2 = torch.cat([self.PWC2(low), low], dim=2)

        # 特征融合
        out = torch.cat([Y1, Y2], dim=2)
        out = F.softmax(self.advavg(out), dim=2) * out
        out1, out2 = torch.split(out, out.size(2) // 2, dim=2)
        return out1 + out2


# 自定义 ScConv（Squeeze and Channel Reduction Convolution）模型
class ScConv(nn.Module):
    def __init__(self, op_channel: int, group_num: int = 16, gate_treshold: float = 0.5, alpha: float = 1 / 2,
                 squeeze_radio: int = 2, group_size: int = 2, group_kernel_size: int = 3):
        super().__init__()  # 调用父类构造函数

        self.SRU = SRU(op_channel, group_num=group_num, gate_treshold=gate_treshold)  # 创建 SRU 层
        self.CRU = CRU(op_channel, alpha=alpha, squeeze_radio=squeeze_radio, group_size=group_size,
                       group_kernel_size=group_kernel_size)  # 创建 CRU 层

    def forward(self, x):
        x = self.SRU(x)  # 应用 SRU 层
        x = self.CRU(x)  # 应用 CRU 层
        return x


class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        # Squeeze操作
        self.avg_pool = nn.AdaptiveAvgPool3d((None, 1, 1))
        # Excitation操作(FC+ReLU+FC+Sigmoid)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        t, b, c, _, _ = x.size()
        y = self.avg_pool(x)
        y = y.view(t, b, c)
        y = self.fc(y).view(t, b, c, 1, 1)  # 学习到的每一channel的权重
        return x * y


class MobileNetV3(nn.Module):
    def __init__(self, inp, oup, hidden_dim, kernel_size, stride, use_se, use_hs):
        super(MobileNetV3, self).__init__()
        assert stride in [1, 2]

        self.identity = stride == 1 and inp == oup

        # 输入通道数=扩张通道数 则不进行通道扩张
        if inp == hidden_dim:
            self.conv = nn.Sequential(
                mem_update(act=False) if use_hs else nn.ReLU(inplace=True),
                # dw
                Snn_Conv2d(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2, groups=hidden_dim,
                           bias=False),
                batch_norm_2d(hidden_dim),
                mem_update(act=False) if use_hs else nn.ReLU(inplace=True),
                # Squeeze-and-Excite
                SELayer(hidden_dim) if use_se else nn.Sequential(),
                # pw-linear
                Snn_Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                batch_norm_2d1(oup),
            )
        else:
            # 否则 先进行通道扩张
            self.conv = nn.Sequential(
                mem_update(act=False) if use_hs else nn.ReLU(inplace=True),
                # pw
                Snn_Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                batch_norm_2d(hidden_dim),
                mem_update(act=False) if use_hs else nn.ReLU(inplace=True),
                # dw
                Snn_Conv2d(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2, groups=hidden_dim,
                           bias=False),
                batch_norm_2d(hidden_dim),
                # Squeeze-and-Excite
                SELayer(hidden_dim) if use_se else nn.Sequential(),
                mem_update(act=False) if use_hs else nn.ReLU(inplace=True),
                # pw-linear
                Snn_Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                batch_norm_2d1(oup),
            )

    def forward(self, x):
        y = self.conv(x)
        if self.identity:
            return x + y
        else:
            return y


class AKConv(nn.Module):
    def __init__(self, inc, outc, num_param, stride=1, bias=None):
        super(AKConv, self).__init__()
        self.num_param = num_param
        self.stride = stride
        self.conv = nn.Sequential(mem_update(act=False),
                                  Snn_Conv2d(inc, outc, kernel_size=(num_param, 1), stride=(num_param, 1), bias=bias),
                                  batch_norm_2d(
                                      outc))  # the conv adds the BN and SiLU to compare original Conv in YOLOv5.
        self.p_conv = Snn_Conv2d(inc, 2 * num_param, kernel_size=3, padding=1, stride=stride)
        nn.init.constant_(self.p_conv.weight, 0)
        self.p_conv.register_full_backward_hook(self._set_lr)

    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
        grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))

    def forward(self, x):
        # N is num_param.
        offset = self.p_conv(x)
        dtype = offset.data.type()
        # t, b, n, h, w =offset.shape
        # offset = offset.reshape(t*b, n, h, w)
        N = offset.size(2) // 2
        # (t, b, 2N, h, w)
        p = self._get_p(offset, dtype)

        # (t, b, h, w, 2N)
        p = p.contiguous().permute(0, 1, 3, 4, 2)
        q_lt = p.detach().floor()
        q_rb = q_lt + 1

        q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(3) - 1), torch.clamp(q_lt[..., N:], 0, x.size(4) - 1)],
                         dim=-1).long()
        q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(3) - 1), torch.clamp(q_rb[..., N:], 0, x.size(4) - 1)],
                         dim=-1).long()
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)

        # clip p
        p = torch.cat([torch.clamp(p[..., :N], 0, x.size(3) - 1), torch.clamp(p[..., N:], 0, x.size(4) - 1)], dim=-1)

        # bilinear kernel (t, b, h, w, N)
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))

        # resampling the features based on the modified coordinates.
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)

        # bilinear
        x_offset = g_lt.unsqueeze(dim=2) * x_q_lt + \
                   g_rb.unsqueeze(dim=2) * x_q_rb + \
                   g_lb.unsqueeze(dim=2) * x_q_lb + \
                   g_rt.unsqueeze(dim=2) * x_q_rt

        x_offset = self._reshape_x_offset(x_offset, self.num_param)
        # x_offset = x_offset.
        out = self.conv(x_offset)

        return out

    # generating the inital sampled shapes for the AKConv with different sizes.
    def _get_p_n(self, N, dtype):
        base_int = round(math.sqrt(self.num_param))
        row_number = self.num_param // base_int
        mod_number = self.num_param % base_int
        p_n_x, p_n_y = torch.meshgrid(
            torch.arange(0, row_number),
            torch.arange(0, base_int), indexing='xy')
        p_n_x = torch.flatten(p_n_x)
        p_n_y = torch.flatten(p_n_y)
        if mod_number > 0:
            mod_p_n_x, mod_p_n_y = torch.meshgrid(
                torch.arange(row_number, row_number + 1),
                torch.arange(0, mod_number), indexing='xy')

            mod_p_n_x = torch.flatten(mod_p_n_x)
            mod_p_n_y = torch.flatten(mod_p_n_y)
            p_n_x, p_n_y = torch.cat((p_n_x, mod_p_n_x)), torch.cat((p_n_y, mod_p_n_y))
        p_n = torch.cat([p_n_x, p_n_y], 0)
        p_n = p_n.view(1, 1, 2 * N, 1, 1).type(dtype)
        return p_n

    # no zero-padding
    def _get_p_0(self, h, w, N, dtype):
        p_0_x, p_0_y = torch.meshgrid(
            torch.arange(0, h * self.stride, self.stride),
            torch.arange(0, w * self.stride, self.stride), indexing='xy')

        p_0_x = torch.flatten(p_0_x).view(1, 1, 1, h, w).repeat(1, 1, N, 1, 1)
        p_0_y = torch.flatten(p_0_y).view(1, 1, 1, h, w).repeat(1, 1, N, 1, 1)
        p_0 = torch.cat([p_0_x, p_0_y], 2).type(dtype)

        return p_0

    def _get_p(self, offset, dtype):
        N, h, w = offset.size(2) // 2, offset.size(3), offset.size(4)

        # (1, 1, 2N, 1, 1)
        p_n = self._get_p_n(N, dtype)
        # (1, 2N, h, w)
        p_0 = self._get_p_0(h, w, N, dtype)
        p = p_0 + p_n + offset
        return p

    def _get_x_q(self, x, q, N):
        t, b, h, w, _ = q.size()
        padded_w = x.size(4)
        c = x.size(2)
        # (t, b, c, h*w)
        x = x.contiguous().view(t, b, c, -1)

        # (t, b, h, w, N)
        index = q[..., :N] * padded_w + q[..., N:]  # offset_x*w + offset_y
        # (t, b, c, h*w*N)

        index = index.contiguous().unsqueeze(dim=2).expand(-1, -1, c, -1, -1, -1).contiguous().view(t, b, c, -1)

        # 根据实际情况调整
        index = index.clamp(min=0, max=x.shape[-1] - 1)

        x_offset = x.gather(dim=-1, index=index).contiguous().view(t, b, c, h, w, N)

        return x_offset

    #  Stacking resampled features in the row direction.
    @staticmethod
    def _reshape_x_offset(x_offset, num_param):
        t, b, c, h, w, n = x_offset.size()
        # using Conv3d
        # x_offset = x_offset.permute(0,1,4,2,3), then Conv3d(c,c_out, kernel_size =(num_param,1,1),stride=(num_param,1,1),bias= False)
        # using 1 × 1 Conv
        # x_offset = x_offset.permute(0,1,4,2,3), then, x_offset.view(b,c×num_param,h,w)  finally, Conv2d(c×num_param,c_out, kernel_size =1,stride=1,bias= False)
        # using the column conv as follow， then, Conv2d(inc, outc, kernel_size=(num_param, 1), stride=(num_param, 1), bias=bias)

        x_offset = rearrange(x_offset, 't b c h w n -> t b c (h n) w')
        return x_offset


class DualConv(nn.Module):

    def __init__(self, in_channels, out_channels, stride, g):
        """
        Initialize the DualConv class.
        :param input_channels: the number of input channels
        :param output_channels: the number of output channels
        :param stride: convolution stride
        :param g: the value of G used in DualConv
        """
        super(DualConv, self).__init__()
        # Group Convolution
        # self.gc = Snn_Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, groups=g, bias=False)
        self.gc = GSConv(in_channels, out_channels, k=3, s=stride)
        # self.gc = HetConv(in_channels, out_channels, s=stride, p=3)
        # Pointwise Convolution
        # self.pwc = Snn_Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
        self.pwc = GSConv(in_channels, out_channels, k=1, s=stride)

    def forward(self, input_data):
        """
        Define how DualConv processes the input images or input feature maps.
        :param input_data: input images or input feature maps
        :return: return output feature maps
        """
        return self.gc(input_data) + self.pwc(input_data)


class GSConv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        super().__init__()
        c_ = c2 // 2
        self.cv1 = Conv_3(c1, c_, k, s, None, g, act)
        self.cv2 = Conv_4(c_, c_, 5, 1, None, c_, act)

    def forward(self, x):
        x1 = self.cv1(x)
        x2 = torch.cat((x1, self.cv2(x1)), 2)
        # shuffle
        t, b, n, h, w = x2.data.size()
        b_n = b * n // 2
        y = x2.reshape(t, b_n, 2, h * w)
        y = y.permute(2, 0, 1, 3)
        y = y.reshape(2, t, -1, n // 2, h, w)
        return torch.cat((y[0], y[1]), 2)


class HGBlock(nn.Module):
    """
    HG_Block of PPHGNetV2 with 2 convolutions and LightConv.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, cm, c2, k=3, n=6, stride=1, lightconv=False, shortcut=False, act=nn.ReLU()):
        """Initializes a CSP Bottleneck with 1 convolution using specified input and output channels."""
        super().__init__()
        # block = LightConv if lightconv else BasicBlock_2
        block = LightConv if lightconv else Conv_3
        self.dw = nn.Identity()
        if stride == 2:
            self.dw = DWConv(c1, c1, k=k, s=stride)
        # self.m = nn.ModuleList(block(c1 if i == 0 else cm, cm, k_size=k, stride=1) for i in range(n))
        self.m = nn.ModuleList(block(c1 if i == 0 else cm, cm, k=k) for i in range(n))
        self.sc = Conv_3(c1 + n * cm, c2 // 2, 1, 1)  # squeeze conv
        self.ec = Conv_4(c2 // 2, c2, 1, 1)  # excitation conv
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Forward pass of a PPHGNetV2 backbone layer."""
        x = self.dw(x)
        y = [x]
        y.extend(m(y[-1]) for m in self.m)
        y = self.ec(self.sc(torch.cat(y, 2)))
        return y + x if self.add else y


class StarBlock(nn.Module):
    def __init__(self, in_channels, out_channels, k_size=3, stride=1, mlp_ratio=3, drop_path=0., add=True):
        super().__init__()
        self.dw = Conv_3(in_channels, out_channels, k=k_size, s=stride)
        self.dwconv = Conv_1(out_channels, out_channels, 7, 1, (7 - 1) // 2, g=out_channels)
        self.f1 = Conv_1(out_channels, mlp_ratio * out_channels, 1, 1)
        self.f2 = Conv_1(out_channels, mlp_ratio * out_channels, 1, 1)
        self.g = Conv_1(mlp_ratio * out_channels, out_channels, 1, 1)
        self.dwconv2 = Conv_1(out_channels, out_channels, 7, 1, (7 - 1) // 2, g=out_channels)
        self.act = nn.ReLU6()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = self.dw(x)
        input = x
        x = self.dwconv(x)
        x1, x2 = self.f1(x), self.f2(x)
        x = self.act(x1) * x2
        x = self.dwconv2(self.g(x))
        x = input + self.drop_path(x)
        return x


class StarBlock_2(nn.Module):
    def __init__(self, in_channels, out_channels, k_size=3, stride=1, mlp_ratio=3, drop_path=0., add=True):
        super().__init__()
        self.dw = nn.Sequential(
            mem_update(act=False),
            DepthWiseConv(in_channels, in_channels, kernel_size=k_size, stride=stride, padding=(k_size - 1) // 2),
            batch_norm_2d(in_channels))
        self.f1 = Conv_3(in_channels, mlp_ratio * in_channels, 1, 1)
        self.f2 = Conv_3(in_channels, mlp_ratio * in_channels, 1, 1)
        # self.g = Conv_3(mlp_ratio * in_channels, out_channels, 1, 1)
        self.g = Conv_3(mlp_ratio * in_channels, out_channels, k=k_size, s=1)
        # self.dw2 = nn.Sequential(
        #     mem_update(act=False),
        #     DepthWiseConv(out_channels, out_channels, kernel_size=k_size, stride=1, padding=(k_size - 1) // 2),
        #     batch_norm_2d1(out_channels))
        self.act = nn.ReLU6()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.shortcut = nn.Sequential(
        )
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.MaxPool3d((1, stride, stride), stride=(1, stride, stride)),
                mem_update(act=False),
                # EcsLifNode(v_threshold=thresh, step_mode='m'),
                Snn_Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                batch_norm_2d(out_channels),
            )

    def forward(self, x):
        input = x
        x = self.dw(x)
        x1, x2 = self.f1(x), self.f2(x)
        x = self.act(x1) * x2
        x = self.g(x)
        # x = self.dw2(x)
        x = self.shortcut(input) + self.drop_path(x)
        return x


class StarBlock_3(nn.Module):
    def __init__(self, in_channels, out_channels, k_size=3, stride=1, mlp_ratio=3, drop_path=0., add=True):
        super().__init__()
        # self.dw = nn.Sequential(
        #     mem_update(act=False),
        #     DepthWiseConv(in_channels, in_channels, kernel_size=k_size, stride=stride, padding=1),
        #     PartialConv(in_channels),
        #     batch_norm_2d(in_channels))
        self.dw = GSConv(in_channels, in_channels, k=k_size, s=stride)
        self.f1 = Conv_1(in_channels, mlp_ratio * in_channels, 1, 1)
        self.f2 = Conv_1(in_channels, mlp_ratio * in_channels, 1, 1)
        # self.g = nn.Sequential(
        #     mem_update(act=False),
        #     DepthWiseConv(in_channels * mlp_ratio, out_channels, kernel_size=k_size, stride=1, padding=1),
        #     PartialConv(out_channels),
        #     batch_norm_2d(out_channels))
        self.g = GSConv(in_channels * mlp_ratio, out_channels, k=k_size, s=1)
        self.act = nn.ReLU6()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.shortcut = nn.Sequential(
        )
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.MaxPool3d((1, stride, stride), stride=(1, stride, stride)),
                mem_update(act=False),
                # EcsLifNode(v_threshold=thresh, step_mode='m'),
                Snn_Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                batch_norm_2d(out_channels),
            )

    def forward(self, x):
        input = x
        # print(x.shape)
        x = self.dw(x)
        # print(x.shape)
        x1, x2 = self.f1(x), self.f2(x)
        # print(x1.shape)
        # print(x2.shape)
        x = self.act(x1) * x2
        # print(x.shape)
        x = self.g(x)
        # print(x.shape)
        # print(self.shortcut(input).shape)
        # print(self.drop_path(x).shape)
        x = self.shortcut(input) + self.drop_path(x)
        return x


class StarBlock_4(nn.Module):
    def __init__(self, in_channels, out_channels, k_size=3, stride=1, mlp_ratio=3, drop_path=0., add=True):
        super().__init__()
        if k_size == 3:
            pad = 1
        if k_size == 1:
            pad = 0
        self.f1 = nn.Sequential(
            mem_update(act=False),
            Snn_Conv2d(in_channels, out_channels, kernel_size=k_size, stride=stride, padding=pad, bias=False),
            batch_norm_2d(out_channels),
        )
        #self.f1 = GSConv(in_channels, in_channels * mlp_ratio, k=k_size, s=stride)
        #self.f1 = HetConv(in_channels, out_channels, s=stride, p=4)
        #self.f1 = DualConv(in_channels, out_channels, stride=stride, g=1)
        self.f2 = nn.Sequential(
            mem_update(act=False),
            Snn_Conv2d(in_channels, out_channels, kernel_size=k_size, stride=stride, padding=pad, bias=False),
            batch_norm_2d(out_channels),
        )
        #self.f2 = HetConv(in_channels, out_channels, s=stride, p=4)
        #self.f2 = DualConv(in_channels, out_channels, stride=stride, g=1)
        # self.f1 = Conv_1(in_channels, in_channels, 1, 1)
        # self.f1 = nn.Sequential(
        #     GSConv(in_channels, in_channels, 1, 1),
        #     batch_norm_2d(in_channels)
        # )
        # self.f2 = Conv_1(in_channels, in_channels, 1, 1)
        # self.f2 = nn.Sequential(
        #     GSConv(in_channels, in_channels, 1, 1),
        #     batch_norm_2d(in_channels)
        # )
        self.dw2 = nn.Sequential(
            mem_update(act=False),
            # DepthWiseConv(out_channels, out_channels, kernel_size=7, stride=1, padding=(7 - 1) // 2),
            Snn_Conv2d(out_channels, out_channels, kernel_size=k_size, stride=1, padding=(k_size - 1) // 2),
            batch_norm_2d(out_channels))
        # self.dw2 = GSConv(out_channels, out_channels, k=k_size, s=1)
        # self.g = Conv_3(mlp_ratio * in_channels, out_channels, k=k_size, s=1)
        self.act = nn.ReLU6()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.shortcut = nn.Sequential(
        )
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.MaxPool3d((1, stride, stride), stride=(1, stride, stride)),
                mem_update(act=False),
                # EcsLifNode(v_threshold=thresh, step_mode='m'),
                Snn_Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                batch_norm_2d(out_channels),
            )

    def forward(self, x):
        input = x
        x1, x2 = self.f1(x), self.f2(x)
        x = self.act(x1) * x2
        x = self.dw2(x)
        x = self.shortcut(input) + self.drop_path(x)
        return x


class StarBlock_5(nn.Module):
    def __init__(self, in_channels, out_channels, k_size=3, stride=1, mlp_ratio=3, drop_path=0., add=True):
        super().__init__()
        if k_size == 3:
            pad = 1
        if k_size == 1:
            pad = 0
        # self.dw = nn.Sequential(
        #     mem_update(act=False),
        #     # RFAConv(in_channels, out_channels, k_size, stride),
        #     Snn_Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
        #     # DualConv(in_channels, out_channels, stride=stride, g=8),
        #     batch_norm_2d(out_channels),
        # )
        self.f1 = nn.Sequential(
            mem_update(act=False),
            # ScConv(in_channels),
            # mem_update(act=False),
            # DepthWiseConv(in_channels, mlp_ratio * out_channels, kernel_size=k_size, stride=stride, padding=pad),
            Snn_Conv2d(in_channels, out_channels, kernel_size=k_size, stride=stride, padding=pad, bias=False, groups=1),
            # Snn_Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            batch_norm_2d(out_channels),
        )
        self.f2 = nn.Sequential(
            # mem_update(act=False),
            # ScConv(in_channels),
            mem_update(act=False),
            # DepthWiseConv(in_channels, mlp_ratio * out_channels, kernel_size=k_size, stride=stride, padding=pad),
            Snn_Conv2d(in_channels, out_channels, kernel_size=k_size, stride=stride, padding=pad, bias=False, groups=1),
            # Snn_Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            batch_norm_2d1(out_channels),
        )
        # self.f1 = Conv_1(in_channels, in_channels, 1, 1)
        # self.f1 = nn.Sequential(
        #     GSConv(in_channels, in_channels, 1, 1),
        #     batch_norm_2d(in_channels)
        # )
        # self.f2 = Conv_1(in_channels, in_channels, 1, 1)
        # self.f2 = nn.Sequential(
        #     GSConv(in_channels, in_channels, 1, 1),
        #     batch_norm_2d(in_channels)
        # )
        self.dw2 = nn.Sequential(
            mem_update(act=False),
            # RFAConv(out_channels, out_channels, k_size, 1),
            Snn_Conv2d(out_channels, out_channels, kernel_size=k_size, padding=pad, bias=False),
            # CRU(out_channels),
            # DualConv(out_channels, out_channels, stride=1, g=8),
            batch_norm_2d1(out_channels), )
        # self.dw2 = GSConv(out_channels, out_channels, k=k_size, s=1)
        # self.g = Conv_3(mlp_ratio * in_channels, out_channels, k=k_size, s=1)
        self.act = nn.ReLU6()
        # self.act = ClippedTPReLU(num_parameters=out_channels)
        # self.act = nn.GELU()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.shortcut = nn.Sequential(
        )
        if in_channels < out_channels:
            self.shortcut = nn.Sequential(
                mem_update(act=False),
                Snn_Conv2d(in_channels, out_channels - in_channels, kernel_size=1, stride=1, bias=False),
                batch_norm_2d(out_channels - in_channels),
            )
        self.pools = nn.MaxPool3d((1, stride, stride), stride=(1, stride, stride))

    def forward(self, x):
        input = x
        # x = self.dw(x)
        x1, x2 = self.f1(x), self.f2(x)
        x = self.act(x1) * x2
        # x = x1 * x2
        x = self.dw2(x)
        temp = self.shortcut(input)
        out = torch.cat((temp, input), dim=2)
        out = self.pools(out)
        x = out + self.drop_path(x)
        return x


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (x.shape[1],) + (1,) * (x.ndim - 2)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output


class StarNet(nn.Module):
    def __init__(self, in_channels, out_channels, k_size=3, stride=1, mlp_ratio=3, e=2):
        super().__init__()
        c_ = int(in_channels * e)
        p = None
        if k_size == 3:
            pad = 1
        if k_size == 1:
            pad = 0
        # self.act1 = neuron.IFNode(surrogate_function=surrogate.ATan())
        # self.act1 = mem_update(act=False)
        # self.conv1 = nn.Sequential(
        #     mem_update(act=False),
        #     Snn_Conv2d(in_channels, out_channels, kernel_size=k_size, stride=stride, padding=pad, bias=False),
        #     batch_norm_2d(out_channels),
        # )
        self.conv1 = nn.Sequential(
            # mem_update(False),
            StarBlock_4(in_channels, in_channels, k_size=k_size, stride=stride, mlp_ratio=mlp_ratio))
        # self.bn1 = layer.BatchNorm2d(out_channels)
        # self.bn1 = batch_norm_2d(out_channels)
        # self.act2 = mem_update(act=False)
        # self.conv2 = layer.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.conv2 = GSConv(in_channels, out_channels, k=k_size)
        # self.conv2 = nn.Sequential(
        #     # mem_update(False),
        #     StarBlock_2(out_channels, out_channels, k_size=k_size, stride=1, mlp_ratio=mlp_ratio))
        # self.conv2 = nn.Sequential(
        #     mem_update(act=False),
        #     Snn_Conv2d(out_channels, out_channels, kernel_size=1, stride=1, bias=False),
        #     batch_norm_2d1(out_channels),
        # )
        # self.bn2 = layer.BatchNorm2d(out_channels)
        # self.bn2 = batch_norm_2d1(out_channels)

        self.shortcut = nn.Sequential(
        )

        if stride != 1 or in_channels != out_channels:
            self.shortcut = shortcut(in_channels, out_channels, stride)

        # functional.set_step_mode(self, step_mode='m')

    def forward(self, x):
        # print("你好")
        # print(x.shape)
        x1 = self.shortcut(x)
        # x = self.act1(x)
        x = self.conv1(x)
        # x = self.bn1(x)
        # x = self.act2(x)
        x = self.conv2(x)
        # x = self.bn2(x)
        return x + x1


class StarBlock_2C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, k=3, s=1, mlp_ratio=3, drop_path=0., e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.s = s
        if self.s == 2:
            self.cv = Conv_3(c1, c1, 3, 2)
        self.cv1 = Conv_3(c1, 2 * self.c, 1, 1)
        # self.cv1 = StarBlock_2(c1, 2 * self.c)
        self.cv2 = Conv_4((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(StarBlock_2(self.c, self.c, k_size=k, stride=1, mlp_ratio=mlp_ratio) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        if self.s == 2:
            x = self.cv(x)
        y = list(self.cv1(x).chunk(2, 2))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 2))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        if self.s == 2:
            x = self.cv(x)
        y = list(self.cv1(x).split((self.c, self.c), 2))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 2))


class HetConv(nn.Module):
    def __init__(self, in_channels, out_channels, s, p):
        super(HetConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.s = s

        num_k3 = in_channels // p
        num_k1 = in_channels - num_k3
        interval = num_k1 // num_k3

        self.kernels = []
        for i in range(num_k3):
            self.kernels.append(1)
            for j in range(interval):
                self.kernels.append(0)

        self.all_filters = nn.ModuleList()

        for k in range(out_channels):
            if k == 0:
                self.all_filters.append(self.make_filter())
            else:
                temp = self.kernels.pop(-1)
                self.kernels.insert(0, temp)
                self.all_filters.append(self.make_filter())

    def make_filter(self, ):
        filters = nn.ModuleList()
        for i in range(self.in_channels):
            if self.kernels[i] == 1:
                filters.append(Snn_Conv2d(1, 1, 3, stride=self.s, padding=1))
            elif self.kernels[i] == 0:
                filters.append(Snn_Conv2d(1, 1, 1, stride=self.s, padding=0))
        return filters

    def forward(self, x):
        out = []
        for i in range(self.out_channels):
            out_ = self.all_filters[i][0](x[:, :, 0: 1, :, :])
            for j in range(1, self.in_channels):
                out_ += self.all_filters[i][j](x[:, :, j:j + 1, :, :])
            out.append(out_)
        return torch.cat(out, 2)

class TPReLU(nn.PReLU):
    def __init__(self, num_parameters: int = 1, init: float = 0.25,
                 device=None, dtype=None) -> None:
        super(TPReLU, self).__init__(num_parameters, init, device, dtype)

    def forward(self, input: Tensor) -> Tensor:
        c1 = torch.zeros(input.size()[0], input.size()[1], input.size()[2], input.size()[3], input.size()[4],
                         device=input.device, dtype=input.dtype)
        for i in range(input.size()[0]):
            # print(c1[i].shape)
            c1[i] = F.prelu(input[i], self.weight)
        return c1


class ClippedTPReLU(nn.Module):
    def __init__(self, num_parameters=1, init=0.25):
        super(ClippedTPReLU, self).__init__()
        self.tprelu = TPReLU(num_parameters=num_parameters, init=init)

    def forward(self, x):
        output = self.tprelu(x)
        return torch.clamp(output, max=6)


class ASFF3(nn.Module):
    def __init__(self, level, rfb=False, vis=False):
        super(ASFF3, self).__init__()
        self.level = level
        # 特征金字塔从上到下三层的channel数
        # 对应特征图大小(以640*640输入为例)分别为20*20, 40*40, 80*80
        self.dim = [512, 256, 128]
        self.inter_dim = self.dim[self.level]
        if level == 0:  # 特征图最小的一层，channel数512
            self.stride_level_1 = Conv_5(256, self.inter_dim, 3, 2)
            self.stride_level_2 = Conv_5(128, self.inter_dim, 3, 2)
            self.expand = Conv_6(self.inter_dim, 512, 3, 1)
        elif level == 1:  # 特征图大小适中的一层，channel数256
            self.compress_level_0 = Conv_5(512, self.inter_dim, 1, 1)
            self.stride_level_2 = Conv_5(128, self.inter_dim, 3, 2)
            self.resized_level_0 = CustomInterpolate(scale_factor=2, mode='nearest')
            self.expand = Conv_6(self.inter_dim, 256, 3, 1)
        elif level == 2:  # 特征图最大的一层，channel数128
            self.compress_level_0 = Conv_5(512, self.inter_dim, 1, 1)
            self.resized_level_0 = CustomInterpolate(scale_factor=4, mode='nearest')
            self.compress_level_1 = Conv_5(256, self.inter_dim, 1, 1)
            self.resized_level_1 = CustomInterpolate(scale_factor=2, mode='nearest')
            self.expand = Conv_6(self.inter_dim, 128, 3, 1)

        compress_c = 8 if rfb else 16  # when adding rfb, we use half number of channels to save memory

        self.weight_level_0 = Conv_5(self.inter_dim, compress_c, 1, 1)
        self.weight_level_1 = Conv_5(self.inter_dim, compress_c, 1, 1)
        self.weight_level_2 = Conv_5(self.inter_dim, compress_c, 1, 1)

        self.weight_levels = Snn_Conv2d(compress_c * 3, 3, kernel_size=1, stride=1, padding=0)
        self.vis = vis

    def forward(self, x_level_0, x_level_1, x_level_2):
        if self.level == 0:
            level_0_resized = x_level_0
            level_1_resized = self.stride_level_1(x_level_1)

            level_2_downsampled_inter = F.max_pool3d(x_level_2, kernel_size=(1, 3, 3), stride=(1, 2, 2),
                                                     padding=(0, 1, 1))
            level_2_resized = self.stride_level_2(level_2_downsampled_inter)

        elif self.level == 1:
            level_0_compressed = self.compress_level_0(x_level_0)
            level_0_resized = self.resized_level_0(level_0_compressed)
            level_1_resized = x_level_1
            level_2_resized = self.stride_level_2(x_level_2)
        elif self.level == 2:
            level_0_compressed = self.compress_level_0(x_level_0)
            level_0_resized = self.resized_level_0(level_0_compressed)
            level_1_compressed = self.compress_level_1(x_level_1)
            level_1_resized = self.resized_level_1(level_1_compressed)
            level_2_resized = x_level_2


        level_0_weight_v = self.weight_level_0(level_0_resized)
        level_1_weight_v = self.weight_level_1(level_1_resized)
        level_2_weight_v = self.weight_level_2(level_2_resized)
        levels_weight_v = torch.cat((level_0_weight_v, level_1_weight_v, level_2_weight_v), 2)
        levels_weight = self.weight_levels(levels_weight_v)
        levels_weight = F.softmax(levels_weight, dim=2)

        fused_out_reduced = (level_0_resized * levels_weight[:, :, 0:1, :, :] +
                             level_1_resized * levels_weight[:, :, 1:2, :, :] +
                             level_2_resized * levels_weight[:, :, 2:, :, :])

        out = self.expand(fused_out_reduced)

        if self.vis:
            return out, levels_weight, fused_out_reduced.sum(dim=2)
        else:
            return out


class CustomInterpolate(nn.Module):
    def __init__(self, scale_factor=None, mode='bilinear'):
        super(CustomInterpolate, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        # 将张量重塑为 (T * B, C, H, W)
        x_reshaped = x.reshape(-1, x.size(2), x.size(3), x.size(4))

        # 使用 interpolate 函数进行插值
        x_interpolated = F.interpolate(x_reshaped, scale_factor=self.scale_factor, mode=self.mode)

        # 将张量恢复为原始形状 (T, B, C, H, W)
        x_interpolated = x_interpolated.reshape(x.size(0), x.size(1), x_interpolated.size(1), x_interpolated.size(2),
                                             x_interpolated.size(3))
        return x_interpolated


class ASFF2(nn.Module):
    def __init__(self, level, rfb=False, vis=False):
        super(ASFF2, self).__init__()
        self.level = level
        # 特征金字塔从上到下三层的channel数
        # 对应特征图大小(以640*640输入为例)分别为20*20, 40*40, 80*80
        self.dim = [512, 256]
        self.inter_dim = self.dim[self.level]
        if level == 0:  # 特征图最小的一层，channel数512
            self.stride_level_1 = Conv_3(256, self.inter_dim, 3, 2)
            self.stride_level_2 = Conv_3(128, self.inter_dim, 3, 2)
            self.expand = Conv_4(self.inter_dim, 512, 3, 1)
        elif level == 1:  # 特征图大小适中的一层，channel数256
            self.compress_level_0 = Conv_3(512, self.inter_dim, 1, 1)
            self.stride_level_2 = Conv_3(128, self.inter_dim, 3, 2)
            self.resized_level_0 = CustomInterpolate(scale_factor=2, mode='nearest')
            self.expand = Conv_4(self.inter_dim, 256, 3, 1)

        compress_c = 8 if rfb else 16  # when adding rfb, we use half number of channels to save memory

        self.weight_level_0 = GSConv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_1 = GSConv(self.inter_dim, compress_c, 1, 1)

        self.weight_levels = Snn_Conv2d(compress_c * 2, 2, kernel_size=1, stride=1, padding=0)
        self.vis = vis

    def forward(self, x_level_0, x_level_1):
        if self.level == 0:
            level_0_resized = x_level_0
            level_1_resized = self.stride_level_1(x_level_1)
        elif self.level == 1:
            level_0_compressed = self.compress_level_0(x_level_0)
            level_0_resized = self.resized_level_0(level_0_compressed)
            level_1_resized = x_level_1

        level_0_weight_v = self.weight_level_0(level_0_resized)
        level_1_weight_v = self.weight_level_1(level_1_resized)
        levels_weight_v = torch.cat((level_0_weight_v, level_1_weight_v), 2)
        levels_weight = self.weight_levels(levels_weight_v)
        levels_weight = F.softmax(levels_weight, dim=2)

        fused_out_reduced = (level_0_resized * levels_weight[:, :, 0:1, :, :] +
                             level_1_resized * levels_weight[:, :, 1:, :, :])

        out = self.expand(fused_out_reduced)

        if self.vis:
            return out, levels_weight, fused_out_reduced.sum(dim=2)
        else:
            return out