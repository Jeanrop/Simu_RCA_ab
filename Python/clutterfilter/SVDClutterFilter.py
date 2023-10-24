# -*- coding: utf-8 -*-
"""
Created on Fri May  8 08:26:33 2020

@author: jopo86
"""
import arrayfire as af
import numpy as np

def SVDClutterFilter(IQ,mask,wcut):
    
    SizIQ       = np.shape(IQ) 
    ncut        = wcut*SizIQ[len(SizIQ)-1] 
    
    IQ          = af.from_ndarray(IQ)
    IQ          = af.moddims(IQ, SizIQ[0]*SizIQ[1],SizIQ[-1]) 
    [u,s,vt]    = af.svd(af.matmul(af.conjg(IQ.T),IQ))
    eig_v       = u[:,int(ncut[0]):int(ncut[1])]
    IQf      = af.matmul(af.matmul(IQ,eig_v), af.conjg(eig_v.T))
    IQf         = af.moddims(IQf,SizIQ[0],SizIQ[1],SizIQ[2])  
    IQf         = IQf.to_ndarray()
    
    return IQf, s

def SVDClutterFilterSliding(IQ,wcut,L,d):
    
    SizIQ       = np.shape(IQ) 
    ndim        = len(SizIQ)
     
    ncut        = wcut*L
    
    IQ          = IQ.reshape( (np.prod(SizIQ[:ndim-1]), -1), order='F') 
    
    IQf         = np.zeros( IQ.shape, dtype='complex64')                            
    
    Nttot       = IQ.shape[1] 
    Nblk        = int( (Nttot - L)/d )
    
    EigVal      = np.zeros( (L,Nblk), dtype='complex64')  
    for iblk in range(Nblk):
        
        it1         = iblk*d
        it2         = it1+L 
        IQblk       = af.from_ndarray(IQ[:,it1:it2]) 
        [u,s,vt]    = af.svd(af.matmul(af.conjg(IQblk.T),IQblk))
        eig_v       = u[:,int(ncut[0]):int(ncut[1])]
        IQblk       = af.matmul(af.matmul(IQblk,eig_v), af.conjg(eig_v.T))
        IQf[:,it1:it2]      =+ IQblk.to_ndarray() 
        EigVal[:,iblk]      = s.to_ndarray()
    
    IQf = IQf.reshape(SizIQ, order='F')        
    return IQf, EigVal
