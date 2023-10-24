from __future__ import division
import torch
from torch import Tensor
from torch.nn import Module
from torch.nn import BatchNorm1d, BatchNorm2d, BatchNorm3d
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from torch.nn import init
import numpy as np


#************************************************************************
# see https://arxiv.org/abs/1705.09792
#************************************************************************

class _ComplexNormBase(Module):
    """
    adapted from the pytorch batchnorm source code
    https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/batchnorm.py
    """
    def __init__(
        self,
        num_features,
        eps = 1e-5,
        momentum = 0.9,
        affine = True,
        track_running_stats = True
    ):
        super(_ComplexNormBase, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if self.affine:
            self.weight = Parameter(torch.Tensor(num_features, 3))
            self.bias = Parameter(torch.Tensor(num_features, 2))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features, 2))
            self.register_buffer('running_covar', torch.zeros(num_features, 3))
            #initialization of Vrr and Vii to 1/sqrt(2)
            #initialization of Vri to 0
            self.running_covar[:,0] = 1/np.sqrt(2)
            self.running_covar[:,1] = 1/np.sqrt(2)
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_covar', None)
            self.register_parameter('num_batches_tracked', None)
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_covar.zero_()
            #initialization of Vrr and Vii to 1/sqrt(2)
            #initialization of Vri to 0
            self.running_covar[:,0] = 1/np.sqrt(2)
            self.running_covar[:,1] = 1/np.sqrt(2)
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            #initialization of g_rr and g_ii to 1/sqrt(2)
            init.constant_(self.weight[:,:2], 1/np.sqrt(2))
            #initialization of g_ri, beta_r and beta_i to 0
            init.zeros_(self.weight[:,2])
            init.zeros_(self.bias)

    def _check_input_dim(self, input):
        raise NotImplementedError

    def extra_repr(self):
        return '{num_features}, eps={eps}, momentum={momentum}, affine={affine}, ' \
               'track_running_stats={track_running_stats}'.format(**self.__dict__)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        version = local_metadata.get('version', None)

        if (version is None or version < 2) and self.track_running_stats:
            # at version 2: added num_batches_tracked buffer
            #               this should have a default value of 0
            num_batches_tracked_key = prefix + 'num_batches_tracked'
            if num_batches_tracked_key not in state_dict:
                state_dict[num_batches_tracked_key] = torch.tensor(0, dtype=torch.long)

        super(_ComplexNormBase, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)


class _ComplexBatchNorm(_ComplexNormBase):
    """
    adapted from the pytorch batchnorm source code
    https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/batchnorm.py

    see https://arxiv.org/abs/1705.09792

    and original implemntation
    https://github.com/ChihebTrabelsi/deep_complex_networks/blob/master/complexnn/bn.py

    adapted with correction from
    https://github.com/wavefrontshaping/complexPyTorch/blob/master/complexLayers.py

    this implementation allows 3d complex batch normalization and corrected parameter
    initialization
    """
    def __init__(self, num_features, eps=1e-5, momentum=0.9, affine=True,
                 track_running_stats=True):
        super(_ComplexBatchNorm, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)

    def forward(self, input_r, input_i):
        assert(input_r.size() == input_i.size())
        self._check_input_dim(input_r)

        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:
                self.num_batches_tracked = self.num_batches_tracked + 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        """ Decide whether the mini-batch stats should be used for normalization rather than the buffers.
        Mini-batch stats are used in training mode, and in eval mode when buffers are None.
        """
        dims_expand=[None for x in range(2,len(input_r.shape))]
        if self.training:
            #reduction axis for average calculation
            dims = [0]+[dim for dim in range(2, len(input_r.shape))]
            #calculation of expectated value of real and imag
            mean_r = input_r.mean(dims)
            mean_i = input_i.mean(dims)
            #concactenate the means
            mean = torch.stack((mean_r,mean_i),dim=1)

            with torch.no_grad():
                self.running_mean = self.running_mean * exponential_average_factor \
                                    + mean * (1 - exponential_average_factor)

            #centering input_r and input_i : x_centering = x - E[x]
            input_r = input_r-mean_r[tuple([None, ...]+dims_expand)]
            input_i = input_i-mean_i[tuple([None, ...]+dims_expand)]

            #calculation of components of covariance matrix
            Vrr = input_r.pow(2).mean(dim=dims)+self.eps
            Vii = input_i.pow(2).mean(dim=dims)+self.eps
            # Vri contains the real and imaginary covariance for each feature map.
            Vri = (input_r.mul(input_i)).mean(dim=dims) #Vri=Vir

            #moving average update
            n = input_r.numel() / input_r.size(1)
            with torch.no_grad():
                self.running_covar[:,0] = self.running_covar[:,0] * exponential_average_factor \
                                    + n/(n-1) * Vrr * (1 - exponential_average_factor)
                self.running_covar[:,1] = self.running_covar[:,1] * exponential_average_factor \
                                    + n/(n-1) * Vii * (1 - exponential_average_factor)
                self.running_covar[:,2] = self.running_covar[:,2] * exponential_average_factor \
                                    + n/(n-1) * Vri * (1 - exponential_average_factor)

        else:
            mean = self.running_mean
            Vrr = self.running_covar[:,0]+self.eps
            Vii = self.running_covar[:,1]+self.eps
            Vri = self.running_covar[:,2]

            #centering input_r and input_i : x_centering = x - E[x]
            input_r = input_r-mean[tuple([None, ...,0] + dims_expand)]
            input_i = input_i-mean[tuple([None, ...,1] + dims_expand)]

        #calculation of the inverse square root the covariance matrix
        # see https://en.wikipedia.org/wiki/Square_root_of_a_2_by_2_matrix
        det = Vrr*Vii-Vri.pow(2) # determinant of V matrix
        s = torch.sqrt(det)
        t = torch.sqrt(Vii + Vrr + 2 * s)

        #components of the inverse square root the covariance matrix
        i_st= 1/(s*t) #inverse st
        iVrr = (Vii + s)*i_st
        iVii = (Vrr + s)*i_st
        iVri = - Vri*i_st #iVri = iVir

        #calculation of x tilde (input scaling) = iV^(1/2)*(x-E[x])
        input_r, input_i = iVrr[tuple([None, ...]+dims_expand)]*input_r+\
                            iVri[tuple([None, ...]+dims_expand)]*input_i, \
                           iVii[tuple([None, ...]+dims_expand)]*input_i+ \
                           iVri[tuple([None, ...]+dims_expand)]*input_r

        if self.affine:
            #calculation of batch norm: BN(x_tilde) = gamma * x_tilde + beta
            input_r, input_i = self.weight[tuple([None, ...,0]+dims_expand)]*input_r+\
                                self.weight[tuple([None, ...,2]+dims_expand)]*input_i+\
                               self.bias[tuple([None, ...,0]+dims_expand)], \
                               self.weight[tuple([None, ...,2]+dims_expand)]*input_r+\
                               self.weight[tuple([None, ...,1]+dims_expand)]*input_i+\
                               self.bias[tuple([None, ...,1]+dims_expand)]

        return input_r, input_i



class ComplexBatchNorm1d(_ComplexBatchNorm):
    """Complex-valued batch normalization for 3D data.
    See torch.nn.BatchNorm1d for details.
    """
    def _check_input_dim(self, input):
        if input.dim() != 3:
            raise ValueError('expected 3D input (got {}D input)'
                             .format(input.dim()))



class ComplexBatchNorm2d(_ComplexBatchNorm):
    """Complex-valued batch normalization for 4D data.
    See torch.nn.BatchNorm2d for details.
    """
    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))



class ComplexBatchNorm3d(_ComplexBatchNorm):
    """Complex-valued batch normalization for 5D data.
    See torch.nn.BatchNorm3d for details.
    """
    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError('expected 5D input (got {}D input)'
                             .format(input.dim()))



#**************************************************************************
# naive implementation of batch normalization (low computation cost)
#**************************************************************************

class nComplexBatchNorm1d(Module):
    def __init__(
            self,
            num_features,
            eps = 1e-5,
            momentum = 0.1,
            affine = True,
            track_running_stats = True):
        super(nComplexBatchNorm1d, self).__init__()
        self.BNr = BatchNorm1d(num_features, eps, momentum, affine, track_running_stats)
        self.BNi = BatchNorm1d(num_features, eps, momentum, affine, track_running_stats)

    def forward(self, In_real, In_im):
        assert(In_real.size() == In_im.size())

        return self.BNr(In_real), self.BNi(In_im)


class nComplexBatchNorm2d(Module):
    def __init__(
            self,
            num_features,
            eps = 1e-5,
            momentum = 0.1,
            affine = True,
            track_running_stats = True):
        super(nComplexBatchNorm2d, self).__init__()
        self.BNr = BatchNorm2d(num_features, eps, momentum, affine, track_running_stats)
        self.BNi = BatchNorm2d(num_features, eps, momentum, affine, track_running_stats)

    def forward(self, In_real, In_im):
        assert(In_real.size() == In_im.size())

        return self.BNr(In_real), self.BNi(In_im)


class nComplexBatchNorm3d(Module):
    def __init__(
            self,
            num_features,
            eps = 1e-5,
            momentum = 0.1,
            affine = True,
            track_running_stats = True):
        super(nComplexBatchNorm3d, self).__init__()
        self.BNr = BatchNorm3d(num_features, eps, momentum, affine, track_running_stats)
        self.BNi = BatchNorm3d(num_features, eps, momentum, affine, track_running_stats)

    def forward(self, In_real, In_im):
        assert(In_real.size() == In_im.size())

        return self.BNr(In_real), self.BNi(In_im)
