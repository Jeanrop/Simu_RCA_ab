# -*- coding: utf-8 -*-
"""
@author: Paul Xing
"""
import torch
from torch import nn

#TODO not tested yet..

#*****************************************************************
# implemntation of complex dropout (same dropout for real and imag)
#*****************************************************************
class ComplexDropout(nn.Module):
    def __init__(self, p=0.5):
        super(ComplexDropout, self).__init__()
        self.p = p
        self.dropout = nn.Dropout(p)

    def forward(self, In_real, In_im):
        assert(In_real.size() == In_im.size())
        Out_real = self.dropout(In_real)
        #apply same dropout to imag
        Out_im = torch.where(Out_real !=In_real/(1-self.p), \
                            torch.zeros_like(Out_real), In_im/(1-self.p))

        return Out_real, Out_im



class ComplexDropout2d(nn.Module):
    def __init__(self, p=0.5):
        super(ComplexDropout2d, self).__init__()
        self.p = p
        self.dropout = nn.Dropout2d(p)

    def forward(self, In_real, In_im):
        assert(In_real.size() == In_im.size())
        Out_real = self.dropout(In_real)
        #apply same dropout to imag
        Out_im = torch.where(Out_real !=In_real/(1-self.p), \
                            torch.zeros_like(Out_real), In_im/(1-self.p))

        return Out_real, Out_im


class ComplexDropout3d(nn.Module):
    def __init__(self, p=0.5):
        super(ComplexDropout3d, self).__init__()
        self. p = p
        self.dropout = nn.Dropout3d(p)

    def forward(self, In_real, In_im):
        assert(In_real.size() == In_im.size())
        Out_real = self.dropout(In_real)
        #apply same dropout to imag
        Out_im = torch.where(Out_real !=In_real/(1-self.p), \
                            torch.zeros_like(Out_real), In_im/(1-self.p))

        return Out_real, Out_im







#*****************************************************************
#naive implemntation of dropout (real and imag independently)
#*****************************************************************


class nComplexDropout(nn.Module):
    def __init__(self, p=0.5):
        super(nComplexDropout, self).__init__()
        self.dropout = nn.Dropout(p)

    def forward(self, In_real, In_im):
        assert(In_real.size() == In_im.size())
        Out_real = self.dropout(In_real)
        Out_im = self.dropout(In_im)

        return Out_real, Out_im


class nComplexDropout2d(nn.Module):
    def __init__(self, p=0.5):
        super(nComplexDropout2d, self).__init__()
        self.dropout = nn.Dropout2d(p)

    def forward(self, In_real, In_im):
        assert(In_real.size() == In_im.size())
        Out_real = self.dropout(In_real)
        Out_im = self.dropout(In_im)

        return Out_real, Out_im

class nComplexDropout3d(nn.Module):
    def __init__(self, p=0.5):
        super(nComplexDropout3d, self).__init__()
        self.dropout = nn.Dropout3d(p)

    def forward(self, In_real, In_im):
        assert(In_real.size() == In_im.size())
        Out_real = self.dropout(In_real)
        Out_im = self.dropout(In_im)

        return Out_real, Out_im
