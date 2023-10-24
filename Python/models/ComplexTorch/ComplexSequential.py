# -*- coding: utf-8 -*-
"""
@author: Paul Xing
"""

import torch
from torch import nn

#todo testing
class ComplexSequential(nn.Sequential):
    def forward(self, input_r, input_i):
        for module in self._modules.values():
            input_r, input_i = module(input_r, input_i)
        return input_r, input_i
