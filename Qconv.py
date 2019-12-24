import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parameter as Parameter
import torch.nn.init as init
import numpy as np
from numpy.random import RandomState
from typing import List, Tuple
class _QconvBase(nn.Module):
    def __init__(self,
                 in_size,
                 out_size,
                 kernel_size,
                 rank,
                 stride,
                 init,
                 padding='valid',
                 activation=None,
                 conv=None,
                 bias=True,
                 preact=False,
                 name=""
				):
        super().__init__()
        self.stride = stride
        self.padding = padding
        self.activation = activation
        self.weight = torch.nn.Parameter(torch.Tensor(out_size*2, in_size//3, *kernel_size), requires_grad=True).cuda()
        self.conv = conv
        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(3*out_size))
        else:
            self.register_parameter('bias', None)
        if rank == 1:
	        self.f_phase = self.weight[:out_size, :, :]
	        self.f_modulus = self.weight[out_size:, :, :]
        elif rank == 2:
            self.f_phase = self.weight[:out_size, :, :, :]
            self.f_modulus = self.weight[out_size:, :, :, :]
        elif rank == 3:
            self.f_phase = self.weight[:out_size, :, :, :, :]
            self.f_modulus = self.weight[out_size:, :, :, :, :]

        self.reset_parameters(bias)

    def reset_parameters(self, bias, criterion='he'):
        rng = RandomState(1337)
        fan_in, fan_out = init._calculate_fan_in_and_fan_out(self.weight)
        if criterion == 'glorot':
            s = 1. / (fan_in + fan_out)
        elif criterion=='he':
            s = 1. / fan_in
        else:
            raise ValueError('Invalid criterion: ' + self.criterion)

        init.uniform_(self.f_phase, -np.sqrt(s)*np.sqrt(3), np.sqrt(s)*np.sqrt(3))
        init.uniform_(self.f_modulus, -np.pi/2, np.pi/2)

        if self.bias is not None:
            init.constant_(self.bias, 0)

    def forward(self,input):
        f_phase1 = torch.cos(self.f_phase)
        f_phase2 = torch.sin(self.f_phase)*(3**0.5/3)

        f1 = (torch.pow(f_phase1,2)-torch.pow(f_phase2,2))*self.f_modulus
        f2 = (2*(torch.pow(f_phase2,2)-f_phase2*f_phase1))*self.f_modulus
        f3 = (2*(torch.pow(f_phase2,2)+f_phase2*f_phase1))*self.f_modulus
        f4 = (2*(torch.pow(f_phase2,2)+f_phase2*f_phase1))*self.f_modulus
        f5 = (torch.pow(f_phase1,2)-torch.pow(f_phase2,2))*self.f_modulus
        f6 = (2*(torch.pow(f_phase2,2)-f_phase2*f_phase1))*self.f_modulus
        f7 = (2*(torch.pow(f_phase2,2)-f_phase2*f_phase1))*self.f_modulus
        f8 = (2*(torch.pow(f_phase2,2)+f_phase2*f_phase1))*self.f_modulus
        f9 = (torch.pow(f_phase1,2)-torch.pow(f_phase2,2))*self.f_modulus

        matrix1 = torch.cat([f1, f2, f3], axis=1)
        matrix2 = torch.cat([f4, f5, f6], axis=1)
        matrix3 = torch.cat([f7, f8, f9], axis=1)
        matrix = torch.cat([matrix1, matrix2, matrix3], axis=0)
        output = self.conv(input, matrix, self.bias, self.stride, self.padding)

        if self.activation is not None:
            output = self.activation(output)

        return output

class QConv1d(_QconvBase):
    def __init__(self,
            in_size: int,
            out_size: int,
            *,
            kernel_size: int = 1,
            rank: int = 1,
            stride: int = 1,
            padding: int = 0,
            activation=nn.ReLU(inplace=True),
            init=nn.init.kaiming_normal,
            bias: bool = True,
            preact: bool = False,
            name: str = ""
    ):
        super(QConv1d,self).__init__(
            in_size,
            out_size,
            tuple([kernel_size]),
            1,
            stride,
            init,
            padding,
            activation,
            conv=F.conv1d,
            bias=bias,
            preact=preact,
            name=name
        )

class QConv2d(_QconvBase):
    def __init__(
            self,
            in_size: int,
            out_size: int,
            *,
            kernel_size: Tuple[int, int] = (1, 1),
            rank: int = 2 ,
            stride:  Tuple[int, int] = (1, 1),
            padding: Tuple[int, int] = (0, 0),
            activation=nn.ReLU(inplace=True),
            init=nn.init.kaiming_normal,
            bias: bool = True,
            preact: bool = False,
            name: str = ""
    ):
        super(QConv2d,self).__init__(
            in_size,
            out_size,
            kernel_size,
            2,
            stride,
            init,
            padding,
            activation,
            conv=F.conv2d,
            bias=bias,
            preact=preact)
