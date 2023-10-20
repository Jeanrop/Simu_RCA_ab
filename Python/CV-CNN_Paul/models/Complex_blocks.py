# -*- coding: utf-8 -*-
"""
@author: Paul Xing
"""

import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.autograd import Function
from models.ComplexTorch import *


class passthrough(nn.Module):
    def __init__(self):
        super(passthrough, self).__init__()
    def forward(self, In_real, In_im, **kwargs):
        return In_real, In_im


#complex conv block crelu activation
class CReLUConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        super(CReLUConv1d, self).__init__()
        self.ComplexConv =  ComplexConv1d(in_channels, out_channels, kernel_size=kernel_size,
                                            stride=stride, padding=padding)
        Complex_weight_init(self.ComplexConv.convWr.weight,
                            self.ComplexConv.convWi.weight)
        self.ComplexBN = ComplexBatchNorm1d(out_channels)
        self.crelu = CReLU()

    def forward(self, In_real, In_im):
        Out_real, Out_im = self.crelu(*self.ComplexBN(*self.ComplexConv(In_real, In_im)))
        return Out_real, Out_im


class CReLUConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        super(CReLUConv2d, self).__init__()
        self.ComplexConv =  ComplexConv2d(in_channels, out_channels, kernel_size=kernel_size,
                                            stride=stride, padding=padding)
        Complex_weight_init(self.ComplexConv.convWr.weight,
                            self.ComplexConv.convWi.weight)
        self.ComplexBN = ComplexBatchNorm2d(out_channels)
        self.crelu = CReLU()

    def forward(self, In_real, In_im):
        Out_real, Out_im = self.crelu(*self.ComplexBN(*self.ComplexConv(In_real, In_im)))
        return Out_real, Out_im


#complex conv3d block crelu activation
class CReLUConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        super(CReLUConv3d, self).__init__()
        self.ComplexConv =  ComplexConv3d(in_channels, out_channels, kernel_size=kernel_size,
                                            stride=stride, padding=padding)
        Complex_weight_init(self.ComplexConv.convWr.weight,
                            self.ComplexConv.convWi.weight)
        self.ComplexBN = ComplexBatchNorm3d(out_channels)
        self.crelu = CReLU()

    def forward(self, In_real, In_im):
        Out_real, Out_im = self.crelu(*self.ComplexBN(*self.ComplexConv(In_real, In_im)))
        return Out_real, Out_im


class CReLUConvTranspose1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                padding=0, output_padding=0, dilation=1):
        super(CReLUConvTranspose1d, self).__init__()
        self.ComplexConv =  ComplexConvTranspose1d(in_channels, out_channels, kernel_size=kernel_size,
                                            stride=stride, padding=padding, output_padding=output_padding)
        Complex_weight_init(self.ComplexConv.convWr.weight,
                            self.ComplexConv.convWi.weight)
        self.ComplexBN = ComplexBatchNorm1d(out_channels)
        self.crelu = CReLU()

    def forward(self, In_real, In_im):
        Out_real, Out_im = self.crelu(*self.ComplexBN(*self.ComplexConv(In_real, In_im)))
        return Out_real, Out_im



#complex conv block maxout activation
class CMaxoutConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        super(CMaxoutConv1d, self).__init__()
        self.ComplexConv =  ComplexConv1d(in_channels, 4*out_channels, kernel_size=kernel_size,
                                            stride=stride, padding=padding)
        Complex_weight_init(self.ComplexConv.convWr.weight,
                            self.ComplexConv.convWi.weight)
        self.ComplexBN = ComplexBatchNorm1d(4*out_channels)
        self.maxout = ComplexMaxout()

    def forward(self, In_real, In_im):
        Out_real, Out_im = self.maxout(*self.ComplexBN(*self.ComplexConv(In_real, In_im)))
        return Out_real, Out_im


class CMaxoutConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        super(CMaxoutConv2d, self).__init__()
        self.ComplexConv =  ComplexConv2d(in_channels, 4*out_channels, kernel_size=kernel_size,
                                            stride=stride, padding=padding)
        Complex_weight_init(self.ComplexConv.convWr.weight,
                            self.ComplexConv.convWi.weight)
        self.ComplexBN = ComplexBatchNorm2d(4*out_channels)
        self.maxout = ComplexMaxout()

    def forward(self, In_real, In_im):
        Out_real, Out_im = self.maxout(*self.ComplexBN(*self.ComplexConv(In_real, In_im)))
        return Out_real, Out_im


#complex conv3d block maxout activation
class CMaxoutConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        super(CMaxoutConv3d, self).__init__()
        self.ComplexConv =  ComplexConv3d(in_channels, 4*out_channels, kernel_size=kernel_size,
                                            stride=stride, padding=padding)
        Complex_weight_init(self.ComplexConv.convWr.weight,
                            self.ComplexConv.convWi.weight)
        self.ComplexBN = ComplexBatchNorm3d(4*out_channels)
        self.maxout = ComplexMaxout()

    def forward(self, In_real, In_im):
        Out_real, Out_im = self.maxout(*self.ComplexBN(*self.ComplexConv(In_real, In_im)))
        return Out_real, Out_im



class CMaxoutConvTranspose1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                padding=0, output_padding=0, dilation=1):
        super(CMaxoutConvTranspose1d, self).__init__()
        self.ComplexConv =  ComplexConvTranspose1d(in_channels, 4*out_channels, kernel_size=kernel_size,
                                            stride=stride, padding=padding, output_padding=output_padding)
        Complex_weight_init(self.ComplexConv.convWr.weight,
                            self.ComplexConv.convWi.weight)
        self.ComplexBN = ComplexBatchNorm1d(4*out_channels)
        self.maxout = ComplexMaxout()

    def forward(self, In_real, In_im):
        Out_real, Out_im = self.maxout(*self.ComplexBN(*self.ComplexConv(In_real, In_im)))
        return Out_real, Out_im
