import numpy as np
import torch
import torch.utils.data
import h5py
import os



class IQrealigned_dataset(torch.utils.data.Dataset):

    def __init__(self, path2file, align=False, nsamples=None, normalization=False, transform=None,  fc = 15.6250*1e6, ntx= True):
        self.path2file = path2file
        self.align=align
        if nsamples is None:
            with h5py.File(self.path2file, 'r',  libver='latest', swmr=True) as f:
                self.nsamples = f["IQ_r"].shape[0]
                self.nangles = f["IQ_r"].shape[1]
        else:
            self.nsamples = nsamples
        self.normalization = normalization
        self.transform=transform
        self.fc=fc
        self.ntx = ntx

    def __len__(self):
        return int(self.nsamples)

        ## TODO:
    def __getitem__(self, index):

        with h5py.File(self.path2file, 'r',  libver='latest', swmr=True) as f:
            if self.ntx:
                x_r =  np.asarray(f['BFabnosum_r'][index, self.nangles//2-1:self.nangles//2+2,...]) # 31x33x33x128
                x_i =  np.asarray(f['BFabnosum_i'][index,  self.nangles//2-1:self.nangles//2+2,...])
            else:
                x_r =  np.asarray(f['BFabnosum_r'][index, ...]) # 31x33x33x128
                x_i =  np.asarray(f['BFabnosum_i'][index, ...])

            if self.normalization:
                norm = np.max(np.sqrt(x_r**2+x_i**2))
                x_r = x_r/norm
                x_i = x_i/norm

            y = np.asarray(f['AmplitudeAberration'])*np.exp(1.j*2*np.pi*self.fc*np.asarray(f['DelaysAberration']))
            if len(y.shape)==1:
                y_r = np.real(y)
                y_i = np.imag(y)
            else:
                y_r = np.real(y[index, ...])
                y_i = np.imag(y[index, ...])


            x_r = torch.Tensor(x_r)
            x_i =  torch.Tensor(x_i)
            y_r = torch.Tensor(y_r)
            y_i = torch.Tensor(y_i)

        return {'x_r': x_r, 'x_i': x_i, 'y_r': y_r, 'y_i': y_i}


class ConcatIQrealigned_dataset(torch.utils.data.ConcatDataset):
    """
    This class concate multiple datasets from the RFdataset. Uses this when
    each datasets contain batches of simulation samples
    """

    def __init__(self, split, path, normalization=False, align=False, nsamples=None, ntx=True):
        """
        Parameters
        ---------
        split : str
            specifies 'training' or 'val' datasets
        path : dict
                contains the path for the 'training' and 'val' dataset
        """

        self.path=path
        self.split=split
        self.normalization=normalization
        self.align=align
        self.nsamples = nsamples
        self.ntx= ntx

        super(ConcatIQrealigned_dataset, self).__init__([IQrealigned_dataset(os.path.join(self.path[self.split],filename),\
        align=self.align, nsamples=self.nsamples, ntx=self.ntx) for filename in os.listdir(self.path[self.split])])
