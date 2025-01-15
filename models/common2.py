import json
import math
import platform
import warnings
from copy import copy
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
from PIL import Image
from torch.cuda import amp
import torch.nn.functional as F

from spikingjelly.activation_based import neuron, functional, surrogate, layer

from utils.general import (LOGGER, check_requirements, check_suffix, colorstr, increment_path, make_divisible,
                           non_max_suppression, scale_coords, xywh2xyxy, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import time_sync

from utils.datasets import exif_transpose, letterbox

thresh = 0.5  # 0.5 # neuronal threshold 神经元阈值
lens = 0.5  # 0.5 # hyper-parameters of approximate function 近似函数的超参数
decay = 0.25  # 0.25 # decay constants 衰变常数
time_window = 1

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
        return input.gt(0).float()

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
            # range(1)->(0,1)
            if i >= 1:
                # .detach() 返回一个新的Variable，从当前计算图中分离下来的，但是仍指向原变量的存放位置,不同之处只是requires_grad为false，得到的这个Variable永远不需要计算其梯度，不具有grad。
                # 即使之后重新将它的requires_grad置为true,它也不会具有梯度grad。
                mem = mem_old * decay * (1 - spike.detach()) + x[i]
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




class Conv(nn.Module):
    # Standard convolution 标准的卷积
    def __init__(self, c1, c2, k, s, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        # groups通道分组的参数，输入通道数、输出通道数必须同时满足被groups整除；
        super().__init__()
        self.conv = Snn_Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = batch_norm_2d(c2)
        # self.act = mem_update(act=True)

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


class Conv_1(nn.Module):
    # Standard convolution 标准的卷积
    def __init__(self, c1, c2, k, s, p=None, g=1, act=True,use_cupy=False):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = Snn_Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = batch_norm_2d(c2)
        # self.act = mem_update() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

        functional.set_step_mode(self, step_mode='m')

        if use_cupy:
            functional.set_backend(self, backend='cupy')

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
    # Standard convolution 标准的卷积
    def __init__(self, c1, c2, k, s=1, p=None, g=1, act=True,use_cupy=False):  # ch_in, ch_out, kernel, stride, padding, groups
        # groups通道分组的参数，输入通道数、输出通道数必须同时满足被groups整除；
        super().__init__()
        self.act = neuron.LIFNode(v_threshold=thresh,surrogate_function=surrogate.ATan())
        self.conv = Snn_Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = batch_norm_2d(c2)


        functional.set_step_mode(self, step_mode='m')

        if use_cupy:
            functional.set_backend(self, backend='cupy')

    def forward(self, x):  # 前向传播函数
        return self.bn(self.conv(self.act(x)))

    def forward_fuse(self, x):  # 去掉了bn层
        return self.conv(self.act(x))


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

    def forward(self, input):
        weight = self.weight  #
        # print(self.padding[0],'=======')
        # 输入数据的维度应该是 (time_window,batch_size, input_channels, height, width)
        h = (input.size()[3] - self.kernel_size[0] + 2 * self.padding[0]) // self.stride[0] + 1
        w = (input.size()[4] - self.kernel_size[0] + 2 * self.padding[0]) // self.stride[0] + 1
        # torch.zeros()函数,返回一个形状为size,类型为torch.dtype，里面的每一个值都是0的tensor
        c1 = torch.zeros(time_window, input.size()[1], self.out_channels, h, w, device=input.device)
        # print(weight.size(),'=====weight====')
        for i in range(time_window):
            c1[i] = F.conv2d(input[i], weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return c1


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


class BatchNorm3d2(torch.nn.BatchNorm3d):
    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            nn.init.constant_(self.weight, 0.2 * thresh)
            nn.init.zeros_(self.bias)


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
        temp = torch.zeros(time_window, input.size()[1], input.size()[2], input.size()[3] * self.scale_factor,
                           input.size()[4] * self.scale_factor, device=input.device)
        # print(temp.device,'-----')
        for i in range(time_window):
            temp[i] = self.up(input[i])

            # temp[i]= F.interpolate(input[i], scale_factor=self.scale_factor,mode='nearest')
        return temp





class RepC3(nn.Module):
    """Rep C3."""

    def __init__(self, c1, c2, n=3, e=1.0):
        """Initialize CSP Bottleneck with a single convolution using input channels, output channels, and number."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv_2(c1, c2, 1, 1)
        self.cv2 = Conv_2(c1, c2, 1, 1)
        self.m = nn.Sequential(*[RepConv(c_, c_) for _ in range(n)])
        self.cv3 = Conv_2(c_, c2, 1, 1) if c_ != c2 else nn.Identity()

    def forward(self, x):
        """Forward pass of RT-DETR neck layer."""
        return self.cv3(self.m(self.cv1(x)) + self.cv2(x))

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
        assert k == 3 and p == 1
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



class TransformerEncoderLayer(nn.Module):
    # paper: DETRs Beat YOLOs on Real-time Object Detection
    # https://arxiv.org/pdf/2304.08069.pdf
    """Defines a single layer of the transformer encoder."""

    def __init__(self, c1, cm=2048, num_heads=8, dropout=0.0, act=nn.GELU(), normalize_before=False):
        """Initialize the TransformerEncoderLayer with specified parameters."""
        super().__init__()
        self.ma = nn.MultiheadAttention(c1, num_heads, dropout=dropout, batch_first=True)
        # Implementation of Feedforward model
        self.fc1 = nn.Linear(c1, cm)
        self.fc2 = nn.Linear(cm, c1)

        self.norm1 = nn.LayerNorm(c1)
        self.norm2 = nn.LayerNorm(c1)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.act = mem_update(act=False)
        self.normalize_before = normalize_before

    @staticmethod
    def with_pos_embed(tensor, pos=None):
        """Add position embeddings to the tensor if provided."""
        return tensor if pos is None else tensor + pos

    def forward_post(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
        """Performs forward pass with post-normalization."""
        src, pos = self.act(src), self.act(pos)
        q = k = self.with_pos_embed(src, pos)
        src2 = self.ma(q, k, value=src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src2 = self.act(src2)
        src = src + self.dropout1(src2)
        src = self.act(src)
        src = self.norm1(src)
        src = self.act(src)
        src2 = self.fc2(self.dropout(self.act(self.fc1(src))))
        src2 = self.act(src2)
        src = src + self.dropout2(src2)
        return self.norm2(src)

    def forward_pre(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
        """Performs forward pass with pre-normalization."""
        src, pos = self.act(src), self.act(pos)
        src2 = self.norm1(src)
        src2 = self.act(src2)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.ma(q, k, value=src2, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src2 = self.act(src2)
        src = src + self.dropout1(src2)
        src = self.act(src)
        src2 = self.norm2(src)
        src2 = self.act(src2)
        src2 = self.fc2(self.dropout(self.act(self.fc1(src2))))
        src2 = self.act(src2)
        return src + self.dropout2(src2)

    def forward(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
        """Forward propagates the input through the encoder module."""
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class AIFI(TransformerEncoderLayer):
    ### intrascale feature interaction (AIFI)
    """Defines the AIFI transformer layer."""

    def __init__(self, c1, cm=2048, num_heads=8, dropout=0, act=nn.GELU(), normalize_before=False):
        """Initialize the AIFI instance with specified parameters."""
        super().__init__(c1, cm, num_heads, dropout, act, normalize_before)

    def forward(self, x):
        ### intrascale feature interaction (AIFI)
        """Forward pass for the AIFI transformer layer."""
        t, b, c, h, w = x.shape
        pos_embed = self.build_2d_sincos_position_embedding(w, h, c)

        x = super().forward(x.view(t*b, c, h, w).flatten(2).permute(0, 2, 1), pos=pos_embed.to(device=x.device, dtype=x.dtype))
        return x.permute(0, 2, 1).view([t, -1, c, h, w]).contiguous()

    @staticmethod
    def build_2d_sincos_position_embedding(w, h, embed_dim=256, temperature=10000.0):
        """Builds 2D sine-cosine position embedding."""
        grid_w = torch.arange(int(w), dtype=torch.float32)
        grid_h = torch.arange(int(h), dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing='ij')
        assert embed_dim % 4 == 0, \
            'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1. / (temperature ** omega)

        out_w = grid_w.flatten()[..., None] @ omega[None]
        out_h = grid_h.flatten()[..., None] @ omega[None]

        return torch.cat([torch.sin(out_w), torch.cos(out_w), torch.sin(out_h), torch.cos(out_h)], 1)[None]