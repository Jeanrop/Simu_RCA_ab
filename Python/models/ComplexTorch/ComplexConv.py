# -*- coding: utf-8 -*-
"""
@author: Paul Xing
"""
import torch
from torch import nn


#implementation of complex convolution module
## see https://arxiv.org/abs/1705.09792
class ComplexConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                padding=0, dilation=1, groups=1, bias=True):
        super(ComplexConv1d, self).__init__()
        #real part convoluution kernel
        self.convWr = nn.Conv1d(in_channels, out_channels, kernel_size= kernel_size,
                stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        #im part convoluution kernel
        self.convWi= nn.Conv1d(in_channels, out_channels, kernel_size= kernel_size,
                stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)

    def forward(self, In_real, In_im):
        assert(In_real.size() == In_im.size())
        Out_real = self.convWr(In_real) - self.convWi(In_im)
        Out_im = self.convWr(In_im) + self.convWi(In_real)

        return Out_real, Out_im

class ComplexConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                padding=0, dilation=1, groups=1, bias=True):
        super(ComplexConv2d, self).__init__()
        #real part convoluution kernel
        self.convWr = nn.Conv2d(in_channels, out_channels, kernel_size= kernel_size,
                stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        #im part convoluution kernel
        self.convWi= nn.Conv2d(in_channels, out_channels, kernel_size= kernel_size,
                stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)

    def forward(self, In_real, In_im):
        assert(In_real.size() == In_im.size())
        Out_real = self.convWr(In_real) - self.convWi(In_im)
        Out_im = self.convWr(In_im) + self.convWi(In_real)

        return Out_real, Out_im


class ComplexConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                padding=0, dilation=1, groups=1, bias=True):
        super(ComplexConv3d, self).__init__()
        #real part convoluution kernel
        self.convWr = nn.Conv3d(in_channels, out_channels, kernel_size= kernel_size,
                stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        #im part convoluution kernel
        self.convWi= nn.Conv3d(in_channels, out_channels, kernel_size= kernel_size,
                stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)

    def forward(self, In_real, In_im):
        assert(In_real.size() == In_im.size())
        Out_real = self.convWr(In_real) - self.convWi(In_im)
        Out_im = self.convWr(In_im) + self.convWi(In_real)

        return Out_real, Out_im


class ComplexConvTranspose1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                padding=0, output_padding=0, dilation=1, groups=1, bias=True):
        super(ComplexConvTranspose1d, self).__init__()
        #real part convoluution kernel
        self.convWr = nn.ConvTranspose1d(in_channels, out_channels, kernel_size= kernel_size,
                stride=stride, padding=padding, output_padding=output_padding, dilation=dilation, groups=groups, bias=bias)
        #im part convoluution kernel
        self.convWi= nn.ConvTranspose1d(in_channels, out_channels, kernel_size= kernel_size,
                stride=stride, padding=padding, output_padding=output_padding, dilation=dilation, groups=groups, bias=bias)

    def forward(self, In_real, In_im):
        assert(In_real.size() == In_im.size())
        Out_real = self.convWr(In_real) - self.convWi(In_im)
        Out_im = self.convWr(In_im) + self.convWi(In_real)

        return Out_real, Out_im


class ComplexConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                padding=0, output_padding=0, dilation=1, groups=1, bias=True):
        super(ComplexConvTranspose2d, self).__init__()
        #real part convoluution kernel
        self.convWr = nn.ConvTranspose2d(in_channels, out_channels, kernel_size= kernel_size,
                stride=stride, padding=padding, output_padding=output_padding, dilation=dilation, groups=groups, bias=bias)
        #im part convoluution kernel
        self.convWi= nn.ConvTranspose2d(in_channels, out_channels, kernel_size= kernel_size,
                stride=stride, padding=padding, output_padding=output_padding, dilation=dilation, groups=groups, bias=bias)

    def forward(self, In_real, In_im):
        assert(In_real.size() == In_im.size())
        Out_real = self.convWr(In_real) - self.convWi(In_im)
        Out_im = self.convWr(In_im) + self.convWi(In_real)

        return Out_real, Out_im



class ComplexConvTranspose3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                padding=0, output_padding=0, dilation=1, groups=1, bias=True):
        super(ComplexConvTranspose3d, self).__init__()
        #real part convoluution kernel
        self.convWr = nn.ConvTranspose3d(in_channels, out_channels, kernel_size= kernel_size,
                stride=stride, padding=padding, output_padding=output_padding, dilation=dilation, groups=groups, bias=bias)
        #im part convoluution kernel
        self.convWi= nn.ConvTranspose3d(in_channels, out_channels, kernel_size= kernel_size,
                stride=stride, padding=padding, output_padding=output_padding, dilation=dilation, groups=groups, bias=bias)

    def forward(self, In_real, In_im):
        assert(In_real.size() == In_im.size())
        Out_real = self.convWr(In_real) - self.convWi(In_im)
        Out_im = self.convWr(In_im) + self.convWi(In_real)

        return Out_real, Out_im
