# -*- coding: utf-8 -*-
"""
@author: Paul Xing
"""
import torch
from torch import nn



class ComplexLinear(nn.Module):

    def __init__(self, in_features, out_features, bias = True):
        super(ComplexLinear, self).__init__()
        self.l_r = nn.Linear(in_features, out_features, bias)
        self.l_i = nn.Linear(in_features, out_features, bias)

    def forward(self, In_real, In_im):
        assert(In_real.size() == In_im.size())
        Out_real = self.l_r(In_real) - self.l_i(In_im)
        Out_im = self.l_r(In_im) + self.l_i(In_real)

        return Out_real, Out_im
