# -*- coding: utf-8 -*-
"""
@author: Paul Xing
"""


import torch
from torch import nn
from torch.autograd import Variable
from torch.autograd import Function
import numpy as np


class ComplexMaxoutF(Function):
    """
    adapted from https://github.com/Usama113/Maxout-PyTorch/blob/master/Maxout.ipynb
    Maxout not implemented in PyTorch
    """
    # Note that both forward and backward are @staticmethods
    @staticmethod
    def forward(ctx, In_real, In_im, max_out=4):
        x = torch.sqrt(torch.pow(In_real,2) + torch.pow(In_im,2)) #amplitude
        x_r = In_real
        x_i = In_im
        kernels = x.shape[1]  # to get how many kernels/output
        feature_maps = int(kernels / max_out)

        out_shape = (x.shape[0], feature_maps, max_out, *x.shape[2:])
        #reshape before max_out
        x= x.view(out_shape)
        x_r = x_r.view(out_shape)
        x_i = x_i.view(out_shape)

        y, indices = torch.max(x[:, :, :], 2)

        ctx.save_for_backward(In_real, In_im)
        ctx.indices=indices
        ctx.max_out=max_out

        #shape after  max_out
        indices_shape = (x.shape[0], feature_maps, 1,*x.shape[3:])
        indices =  indices.view(indices_shape)

        x_r = x_r.gather(2, indices)
        x_i = x_i.gather(2, indices)
        x_r = x_r.view(y.shape)
        x_i = x_i.view(y.shape)

        return x_r, x_i


    @staticmethod
    def backward(ctx, grad_output_r, grad_output_i ):
        indices,max_out = Variable(ctx.indices), ctx.max_out
        input1_r, input1_i = ctx.saved_variables[0], ctx.saved_variables[1]
        input_r = input1_r.clone()
        input_i = input1_i.clone()

        for i in range(max_out):
            a0=indices==i #boolean
            input_r[:,i:input_r.data.shape[1]:max_out]=a0.float()*grad_output_r
            input_i[:,i:input_i.data.shape[1]:max_out]=a0.float()*grad_output_i

        return input_r, input_i, None

#adapted from, from https://github.com/Usama113/Maxout-PyTorch/blob/master/Maxout.ipynb
#Maxout not implemented in PyTorch
class MaxoutF(Function):
    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, max_out=4):
        x = input
        kernels = x.shape[1]  # to get how many kernels/output
        feature_maps = int(kernels / max_out)
        out_shape = (x.shape[0], feature_maps, max_out, x.shape[2], x.shape[3])
        x= x.view(out_shape)
        y, indices = torch.max(x[:, :, :], 2)
        ctx.save_for_backward(input)
        ctx.indices=indices
        ctx.max_out=max_out
        return y

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        input1,indices,max_out= ctx.saved_variables[0],Variable(ctx.indices),ctx.max_out
        input=input1.clone()
        for i in range(max_out):
            a0=indices==i
            input[:,i:input.data.shape[1]:max_out]=a0.float()*grad_output
        return input, None


#creates a nn.Module calling the Complex Maxout function
class ComplexMaxout(nn.Module):
    """
    custom implementation of the complex maxout unit
    maxout is applied to the modulus of complex numbers
    """
    def __init__(self, max_out=4):
        super(ComplexMaxout, self).__init__()
        self.cMaxout = ComplexMaxoutF.apply
        self.max_out = max_out

    def forward(self, In_real, In_im):
        assert(In_real.size() == In_im.size())
        Out_real, Out_im = self.cMaxout(In_real, In_im, self.max_out)

        return Out_real, Out_im


#naive implemntation of maxout (real and imag independent)
class nComplexMaxout(nn.Module):
    def __init__(self, max_out=4):
        super(nComplexMaxout, self).__init__()
        self.Maxout = MaxoutF.apply
        self.max_out = max_out

    def forward(self, In_real, In_im):
        assert(In_real.size() == In_im.size())
        Out_real = self.Maxout(In_real, self.max_out)
        Out_im = self.Maxout(In_im, self.max_out)

        return Out_real, Out_im





#**************************************************************
#**************************************************************
# Complex non-linear activation functions
#**************************************************************
#**************************************************************



#**************************************************************
# Complex RelU functions
#**************************************************************

#TODO: never tested yet...
#note: according to orignal paper: "All CReLU experiments converged and
#outperformed both modReLU and zReLU, both which variously failed to converge or
#fared substantially worse"
class CReLU(nn.Module):
    """
    implemntation of complex ReLU activation module
    see https://arxiv.org/abs/1705.09792
    """
    def __init__(self):
        super(CReLU, self).__init__()
        self.relu = nn.ReLU()

    def forward(self, In_real, In_im):
        assert(In_real.size() == In_im.size())

        return self.relu(In_real), self.relu(In_im)


#TODO: never tested yet...
class modReLU(nn.Module):
    """
    implemntation of modReLU activation module
    see https://arxiv.org/abs/1705.09792

    return ReLU(|z|+b)exp(i*phase)
    default: same bias for every features
    """
    def __init__(self, num_features=1):
        super(modReLU, self).__init__()
        self.relu = nn.ReLU()
        ## learnable parameters
        self.bias = torch.nn.Parameter(torch.rand(num_features))

    def forward(self, In_real, In_im):
        assert(In_real.size() == In_im.size())
        #modulus of complex numbers
        modulus = torch.sqrt(torch.pow(In_real,2) + torch.pow(In_im,2))
        #see documentation: - instead of + because modulus si always > 0
        activation = self.relu(1 - self.bias/modulus)

        return In_real*activation, In_im*activation


#TODO: never tested yet...
class zReLU(nn.Module):
    """
    implemntation of zReLU activation module
    see https://arxiv.org/abs/1705.09792

    return z if phase [0, pi/2], else return 0
    """

    def __init__(self):
        super(zReLU).__init__()

    def forward(self, In_real, In_im):
        assert(In_real.size() == In_im.size())

        angle = torch.atan2(In_im, In_real)
        activation = torch.where((angle> 0) & (angle < np.pi),
                        torch.ones_like(angle), torch.zeros_like(angle))

        return In_real*activation, In_im*activation



#**************************************************************
# Generalization to complex PRelU functions
#**************************************************************
#TODO: never tested yet...
class CPReLU(nn.Module):
    def __init__(self, num_parameters = 1, init = 0.25):
        super(CPReLU, self).__init__()
        self.prelu = nn.PReLU(num_parameters, init)

    def forward(self, In_real, In_im):
        assert(In_real.size() == In_im.size())

        return self.prelu(In_real), self.prelu(In_im)


#TODO: never tested yet...
class modPReLU(nn.Module):
    """
    return PReLU(|z|+b)exp(i*phase)
    default: same bias for every features
    """
    def __init__(self, num_features=1, num_parameters = 1, init = 0.25):
        super(modPReLU, self).__init__()
        self.prelu = nn.PReLU(num_parameters, init)
        ## learnable parameters
        self.bias = torch.nn.Parameter(torch.rand(num_features))

    def forward(self, In_real, In_im):
        assert(In_real.size() == In_im.size())
        #modulus of complex numbers
        modulus = torch.sqrt(torch.pow(In_real,2) + torch.pow(In_im,2))
        #see documentation: - instead of + because modulus si always > 0
        activation = self.prelu(1 - self.bias/modulus)

        return In_real*activation, In_im*activation




#**************************************************************
# Generalization to complex ElU functions
#**************************************************************

#TODO: never tested yet...
class CELU(nn.Module):
    def __init__(self, alpha=1.0):
        super(CELU, self).__init__()
        self.elu = nn.ELU(alpha)

    def forward(self, In_real, In_im):
        assert(In_real.size() == In_im.size())

        return self.elu(In_real), self.elu(In_im)


#TODO: never tested yet...
class modELU(nn.Module):
    """
    return ELU(|z|+b)exp(i*phase)
    default: same bias for every features
    """
    def __init__(self, num_features=1, alpha = 1.0):
        super(modELU, self).__init__()
        self.elu = nn.ELU(alpha)
        ## learnable parameters
        self.bias = torch.nn.Parameter(torch.rand(num_features))

    def forward(self, In_real, In_im):
        assert(In_real.size() == In_im.size())
        #modulus of complex numbers
        modulus = torch.sqrt(torch.pow(In_real,2) + torch.pow(In_im,2))
        #see documentation: - instead of + because modulus si always > 0
        activation = self.elu(1 - self.bias/modulus)

        return In_real*activation, In_im*activation
