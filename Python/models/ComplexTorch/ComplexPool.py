# -*- coding: utf-8 -*-
"""
@author: Paul Xing
"""
import torch
from torch import nn
import torch.nn.functional as F



class ComplexAvgPool1d(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0, ceil_mode = False,
                count_include_pad = True):
        super(ComplexAvgPool1d, self).__init__()
        self.avgpool = nn.AvgPool1d(kernel_size, stride, padding, ceil_mode,
                                    count_include_pad)

    def forward(self, In_real, In_im):
        assert(In_real.size() == In_im.size())

        return self.avgpool(In_real), self.avgpool(In_im)


class ComplexAvgPool2d(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0, ceil_mode = False,
                count_include_pad = True, divisor_override = None):
        super(ComplexAvgPool2d, self).__init__()
        self.avgpool = nn.AvgPool2d(kernel_size, stride, padding, ceil_mode,
                                    count_include_pad, divisor_override)

    def forward(self, In_real, In_im):
        assert(In_real.size() == In_im.size())

        return self.avgpool(In_real), self.avgpool(In_im)


class ComplexAvgPool3d(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0, ceil_mode = False,
                count_include_pad = True, divisor_override = None):
        super(ComplexAvgPool3d, self).__init__()
        self.avgpool = nn.AvgPool3d(kernel_size, stride, padding, ceil_mode,
                                    count_include_pad, divisor_override)

    def forward(self, In_real, In_im):
        assert(In_real.size() == In_im.size())

        return self.avgpool(In_real), self.avgpool(In_im)


#*****************************************************************
# implemntation of complex maxpooling with respect to the modulus
#*****************************************************************
def retrieve_elements_from_indices(tensor, indices):
    """
    this function return a tensor with elements from indices
    """
    flattened_tensor = tensor.flatten(start_dim=2)
    output = flattened_tensor.gather(dim=2, index=indices.flatten(start_dim=2)).view_as(indices)
    return output



class ComplexMaxPool1d(nn.Module):
    """
    MaxPooling with respect to the modulus of complex numbers
    """
    def __init__(self, kernel_size, stride= None, padding = 0,
                 dilation = 1, return_indices = False, ceil_mode = False):
        super( ComplexMaxPool1d, self).__init__()
        self.return_indices = return_indices
        self.maxpool = nn.MaxPool1d(kernel_size, stride, padding, dilation,
                                    return_indices = True, ceil_mode = ceil_mode)

    def forward(self, In_real, In_im):
        assert(In_real.size() == In_im.size())
        modulus =  torch.sqrt(torch.pow(In_real,2) + torch.pow(In_im,2))
        out, indices = self.maxpool(modulus)

        Out_real = In_real.gather(dim=2, index=indices)
        Out_im = In_im.gather(dim=2, index=indices)

        if self.return_indices:
            return Out_real, Out_im, indices
        else:
            return Out_real, Out_im



class ComplexMaxPool2d(nn.Module):
    """
    MaxPooling with respect to the modulus of complex numbers
    """
    def __init__(self, kernel_size, stride= None, padding = 0,
                 dilation = 1, return_indices = False, ceil_mode = False):
        super( ComplexMaxPool2d, self).__init__()
        self.return_indices = return_indices
        self.maxpool = nn.MaxPool2d(kernel_size, stride, padding, dilation,
                                    return_indices = True, ceil_mode = ceil_mode)

    def forward(self, In_real, In_im):
        assert(In_real.size() == In_im.size())
        modulus =  torch.sqrt(torch.pow(In_real,2) + torch.pow(In_im,2))
        out, indices = self.maxpool(modulus)

        Out_real = retrieve_elements_from_indices(In_real, indices)
        Out_im =  retrieve_elements_from_indices(In_im, indices)

        if self.return_indices:
            return Out_real, Out_im, indices
        else:
            return Out_real, Out_im



class ComplexMaxPool3d(nn.Module):
    """
    MaxPooling with respect to the modulus of complex numbers
    """
    def __init__(self, kernel_size, stride= None, padding = 0,
                 dilation = 1, return_indices = False, ceil_mode = False):
        super( ComplexMaxPool3d, self).__init__()
        self.return_indices = return_indices
        self.maxpool = nn.MaxPool3d(kernel_size, stride, padding, dilation,
                                    return_indices = True, ceil_mode = ceil_mode)

    def forward(self, In_real, In_im):
        assert(In_real.size() == In_im.size())
        modulus =  torch.sqrt(torch.pow(In_real,2) + torch.pow(In_im,2))
        out, indices = self.maxpool(modulus)

        Out_real = retrieve_elements_from_indices(In_real, indices)
        Out_im =  retrieve_elements_from_indices(In_im, indices)

        if self.return_indices:
            return Out_real, Out_im, indices
        else:
            return Out_real, Out_im



class ComplexAdaptativeAvgPool1d(nn.Module):
    def __init__(self, output_size):
        super(ComplexAdaptativeAvgPool2d, self).__init__()
        self.output_size=output_size

    def forward(self, In_real, In_im):
        Out_real = F.adaptive_avg_pool1d(In_real, self.output_size)
        Out_im = F.adaptive_avg_pool1d(In_im, self.output_size)

        return Out_real, Out_im


class ComplexAdaptativeAvgPool2d(nn.Module):
    def __init__(self, output_size):
        super(ComplexAdaptativeAvgPool2d, self).__init__()
        self.output_size=output_size

    def forward(self, In_real, In_im):
        Out_real = F.adaptive_avg_pool2d(In_real, self.output_size)
        Out_im = F.adaptive_avg_pool2d(In_im, self.output_size)

        return Out_real, Out_im


class ComplexAdaptativeAvgPool3d(nn.Module):
    def __init__(self, output_size):
        super(ComplexAdaptativeAvgPool3d, self).__init__()
        self.output_size=output_size

    def forward(self, In_real, In_im):
        Out_real = F.adaptive_avg_pool3d(In_real, self.output_size)
        Out_im = F.adaptive_avg_pool3d(In_im, self.output_size)

        return Out_real, Out_im





#*****************************************************************
#naive implemntation of max pooling (real and imag independently)
#*****************************************************************
#TODO testing
class nComplexMaxPool1d(nn.Module):
    """
    MaxPooling real and imag independently
    """
    def __init__(self, kernel_size, stride= None, padding = 0,
                 dilation = 1, return_indices = False, ceil_mode = False):
        super( nComplexMaxPool1d, self).__init__()
        self.return_indices = return_indices
        self.maxpool = nn.MaxPool1d(kernel_size, stride, padding, dilation,
                                    return_indices, ceil_mode)

    def forward(self, In_real, In_im):
        assert(In_real.size() == In_im.size())

        if self.return_indices:
            Out_real, indices_r = self.maxpool(In_real)
            Out_im, indices_i = self.maxpool(In_im)

            return Out_real, Out_im, indices_r, indices_i

        else:
            Out_real = self.maxpool(In_real)
            Out_im = self.maxpool(In_im)

            return Out_real, Out_im

#TODO testing
class nComplexMaxPool2d(nn.Module):
    """
    MaxPooling real and imag independently
    """
    def __init__(self, kernel_size, stride= None, padding = 0,
                 dilation = 1, return_indices = False, ceil_mode = False):
        super( nComplexMaxPool2d, self).__init__()
        self.return_indices = return_indices
        self.maxpool = nn.MaxPool2d(kernel_size, stride, padding, dilation,
                                    return_indices, ceil_mode)

    def forward(self, In_real, In_im):
        assert(In_real.size() == In_im.size())

        if self.return_indices:
            Out_real, indices_r = self.maxpool(In_real)
            Out_im, indices_i = self.maxpool(In_im)

            return Out_real, Out_im, indices_r, indices_i

        else:
            Out_real = self.maxpool(In_real)
            Out_im = self.maxpool(In_im)

            return Out_real, Out_im

#TODO testing
class nComplexMaxPool3d(nn.Module):
    """
    MaxPooling real and imag independently
    """
    def __init__(self, kernel_size, stride= None, padding = 0,
                 dilation = 1, return_indices = False, ceil_mode = False):
        super( nComplexMaxPool3d, self).__init__()
        self.return_indices = return_indices
        self.maxpool = nn.MaxPool3d(kernel_size, stride, padding, dilation,
                                    return_indices, ceil_mode)

    def forward(self, In_real, In_im):
        assert(In_real.size() == In_im.size())

        if self.return_indices:
            Out_real, indices_r = self.maxpool(In_real)
            Out_im, indices_i = self.maxpool(In_im)

            return Out_real, Out_im, indices_r, indices_i

        else:
            Out_real = self.maxpool(In_real)
            Out_im = self.maxpool(In_im)

            return Out_real, Out_im
