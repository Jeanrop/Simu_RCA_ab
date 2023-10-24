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
from models.Complex_blocks import *




class conv_blocks3d_a(nn.Module):
    def __init__(self, in_channels, out_channels=64):
        super(conv_blocks3d_a, self).__init__()
        self.Conv3d_1 = CReLUConv3d(in_channels, 256, kernel_size=(5,5,17), padding=(2,2,8))
        #self.Conv3d_2 = CReLUConv3d(256, 128, kernel_size=3, padding = 1)
        self.Conv3d_3 = CReLUConv3d(256, 64, kernel_size=(3,3,4), stride=2, padding=(0,0,1))
        self.Conv3d_4 = CReLUConv3d(64, out_channels, kernel_size=1)


    def forward(self, In_real, In_im):
        # N x 31 x 33 x 33 x 128
        Out_real, Out_im = self.Conv3d_1(In_real, In_im)
        # N x 256 x 33 x 33 x 128
        #Out_real, Out_im = self.Conv3d_2(Out_real, Out_im)
        # N x 128 x 33 x 33 x 128
        Out_real, Out_im = self.Conv3d_3(Out_real, Out_im)
        # N x 64 x 16 x 16 x 64
        Out_real, Out_im = self.Conv3d_4(Out_real, Out_im)
        # N x 64 x 16 x 16 x 64
        return Out_real, Out_im

class conv_blocks3d_b(nn.Module):
    def __init__(self, in_channels, out_channels=16):
        super(conv_blocks3d_b, self).__init__()
        self.Conv3d_1 = CReLUConv3d(in_channels, 64, kernel_size=3, padding=1)
        self.Conv3d_2 =  CReLUConv3d(64, 32, kernel_size=4, stride=2, padding=1)
        self.Conv3d_3 =  CReLUConv3d(32, out_channels, kernel_size=1)


    def forward(self, In_real, In_im):
        # N x in x 16 x 16 x 64
        Out_real, Out_im = self.Conv3d_1(In_real, In_im)
        Out_real, Out_im = self.Conv3d_2(Out_real, Out_im)
        Out_real, Out_im = self.Conv3d_3(Out_real, Out_im)
        # N x out x 16 x 16 x 32
        return Out_real, Out_im

class up_CReLU_conv_skip(nn.Module):
    def __init__(self, in_channels=64):
        super(up_CReLU_conv_skip, self).__init__()
        self.up_conv1 = CReLUConvTranspose1d(in_channels, 32, kernel_size=3, padding=1,
                            output_padding=1, stride=2) #16
        self.up_conv2 = CReLUConvTranspose1d(32+32, 16, kernel_size=3, padding=1,
                            output_padding=1, stride=2) #32
        self.up_conv3 = CReLUConvTranspose1d(16+16, 8, kernel_size=3, padding=1,
                            output_padding=1, stride=2) #64
        self.up_conv4 = CReLUConvTranspose1d(8, 1, kernel_size=3, padding=1,
                            output_padding=1, stride=2) #128
    def forward(self, In_real, In_im, skip_r16, skip_i16, skip_r32, skip_i32):
        Out_r16, Out_i16 = self.up_conv1(In_real, In_im)
        Out_r16 = torch.cat((Out_r16, skip_r16),1)
        Out_i16 = torch.cat((Out_i16, skip_i16),1)

        Out_r32, Out_i32 = self.up_conv2(Out_r16, Out_i16)
        Out_r32 = torch.cat((Out_r32, skip_r32),1)
        Out_i32 = torch.cat((Out_i32, skip_i32),1)
        Out_r64, Out_i64 = self.up_conv3(Out_r32, Out_i32)
        Out_r128, Out_i128 = self.up_conv4(Out_r64, Out_i64)
        Out_r128 = torch.flatten(Out_r128, 1)
        Out_i128 = torch.flatten(Out_i128, 1)
        return Out_r128, Out_i128


class ComplexInceptionA3d(nn.Module):
    def __init__(self, in_channels, pool_features, conv_block = None):
        super(ComplexInceptionA3d, self).__init__()
        if conv_block is None:
            conv_block = CReLUConv3d

        self.branch1x1 = conv_block(in_channels, 64, kernel_size=1)

        self.branch5x5_1 = conv_block(in_channels, 48, kernel_size=1)
        self.branch5x5_2 = conv_block(48, 64, kernel_size=5, padding=2)

        self.branch3x3dbl_1 = conv_block(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = conv_block(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = conv_block(96, 96, kernel_size=3, padding=1)

        self.avg_pool = ComplexAvgPool3d( kernel_size=3, stride=1, padding=1)
        self.branch_pool = conv_block(in_channels, pool_features, kernel_size=1)


    def _forward(self, x_real, x_im):
        branch1x1 = self.branch1x1(x_real, x_im)

        branch5x5 = self.branch5x5_1(x_real, x_im)
        branch5x5 = self.branch5x5_2(*branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x_real, x_im)
        branch3x3dbl = self.branch3x3dbl_2(*branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(*branch3x3dbl)

        branch_pool = self.avg_pool(x_real, x_im)
        branch_pool = self.branch_pool(*branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return outputs

    def forward(self, x_real, x_im):
        outputs = self._forward(x_real, x_im)
        return torch.cat([i for i,j in outputs], 1), torch.cat([j for i,j in outputs],1)


#global pooling block
class GlobalAvgPool3dto1d(nn.Module):
    def __init__(self, in_channels,out_channels, Nele=16):
        super(GlobalAvgPool3dto1d, self).__init__()
        self.pool = ComplexAdaptativeAvgPool3d((1, 1, Nele))
        self.dropout = ComplexDropout()
        self.conv = ComplexConv1d(in_channels,out_channels,kernel_size=3,padding=1)
        #self.crelu = CReLU()


    def forward(self, In_real, In_im):
        Out_real, Out_im = self.pool(*self.dropout(In_real, In_im))
        Out_real = torch.flatten(Out_real, 2)
        Out_im = torch.flatten(Out_im, 2)
        Out_real, Out_im = self.conv(Out_real, Out_im)

        return Out_real, Out_im


class IQInception(nn.Module):
    def __init__(self, ntx=11, Nele=128):
        super(IQInception, self).__init__()
        self.conv_blocks_a = conv_blocks3d_a(ntx,32)
        self.do_a = ComplexDropout3d()
        self.inception_blocks_a = ComplexInceptionA3d(32,32)
        self.do_b = ComplexDropout3d()
        self.conv_blocks_b = conv_blocks3d_b(256,32)
        self.global_pool = GlobalAvgPool3dto1d(32, 16,32)
        self.conv1d_1 =  CReLUConv1d(16,32,kernel_size=3, padding=1, stride=2) # 16
        self.conv1d_2 =  CReLUConv1d(32,64,kernel_size=3, padding=1, stride=2) # 8
        self.up_conv_blocks = up_CReLU_conv_skip()
        self.fc1 = ComplexLinear(128,128)


    def forward(self, In_real, In_im):
        # N x 11 x 17 x 17 x 128
        Out_real, Out_im = self.do_a(*self.conv_blocks_a(In_real, In_im))
        # N x 64 x 8 x 8 x 64
        Out_real, Out_im = self.do_b(*self.inception_blocks_a(Out_real, Out_im))
        # N x 256 x 8 x 8 x 64
        Out_real, Out_im = self.conv_blocks_b(Out_real, Out_im)
        # N x 32 x 4 x 4 x 32
        Out_r32, Out_i32 = self.global_pool(Out_real, Out_im)
        # N x 16 x 32
        Out_r16, Out_i16 = self.conv1d_1(Out_r32, Out_i32)
        # N x 32 x 16
        Out_r8, Out_i8 = self.conv1d_2(Out_r16, Out_i16)
        # N x 64 x 8
        Out_real, Out_im = self.up_conv_blocks(Out_r8, Out_i8,Out_r16, Out_i16, Out_r32, Out_i32)
        # N x 128
        Out_real, Out_im = self.fc1(Out_real, Out_im)
        # N x 128
        return Out_real, Out_im
