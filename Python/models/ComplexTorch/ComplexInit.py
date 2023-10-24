# -*- coding: utf-8 -*-
"""
@author: Paul Xing
"""
import math
import torch
import numpy as np
from torch.nn import init


def Complex_weight_init(tensor_real, tensor_imag, criterion = 'glorot'):
    """
    criterion : glorot or he
    see https://arxiv.org/abs/1705.09792
    """

    assert(tensor_real.size() == tensor_imag.size())
    criterion = criterion.lower()
    valid_criterions = ['glorot', 'he']
    if criterion not in valid_criterions:
        raise ValueError("Criterion {} not supported, please use one of {}".format(criterion, valid_criterions))

    fan_in, fan_out = init._calculate_fan_in_and_fan_out(tensor_real)

    if criterion == "glorot":
        sigma = 1 / math.sqrt(fan_in + fan_out)
    else:
        sigma = 1 / math.sqrt(fan_in)

    modulus = np.random.rayleigh(scale=sigma, size=tensor_real.shape)
    phase = np.random.uniform(-np.pi, np.pi, size=tensor_real.shape)

    weight_real = torch.tensor(modulus*np.cos(phase))
    weight_imag =torch.tensor(modulus*np.sin(phase))

    with torch.no_grad():
        return tensor_real.copy_(weight_real), tensor_imag.copy_(weight_imag)
