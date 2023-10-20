# -*- coding: utf-8 -*-
"""
Created on Tue May 12 10:27:25 2020

@author: jopo86
"""

import arrayfire as af
import numpy as np

# PsfCorrMap2D
def PsfCorrMap2D(IQ,psf,batchsize=0):
    # WARNING IQ must be decomposable into  n batchsize
    
    kernel      = af.cast(af.from_ndarray(psf),af.Dtype.c32)
    kernelmask  = af.constant(1,psf.shape[0],psf.shape[1])
    #kernelmask  = af.from_ndarray(np.hanning(psf.shape[1])*np.transpose(np.hanning(psf.shape[0])[np.newaxis],(1,0)))
    NormPsf     = np.sqrt( af.sum(kernel*af.conjg(kernel)*kernelmask**2) ) 
     
    SizIQ       = np.shape(IQ)
    
    if batchsize>0:
        nbatch      = int(SizIQ[len(SizIQ)-1]/batchsize)
        
        IQ          = IQ.reshape(SizIQ[0],SizIQ[1],batchsize,nbatch,order='F') 
        
        CorrMap     = np.empty((SizIQ[0],SizIQ[1],batchsize,nbatch),np.complex64,order='F') 
        for ibatch in range(nbatch):
        
            IQaf        = af.cast(af.from_ndarray(IQ[:,:,:,ibatch]),af.Dtype.c32)
            #NormIQ      = af.sqrt( af.convolve3( af.from_ndarray(IQaf*np.conj(IQaf)) , kernelmask**2)) 
            NormIQ      = af.sqrt( af.convolve3( IQaf*af.conjg(IQaf) , kernelmask))
            CorrMapaf   = af.convolve3(IQaf,af.conjg(kernel)*kernelmask)
            CorrMapaf   = CorrMapaf/NormIQ/NormPsf
        
            CorrMap[:,:,:,ibatch] = CorrMapaf.to_ndarray()  
    
    elif batchsize==0:
        IQaf        = af.cast(af.from_ndarray(IQ),af.Dtype.c32)
        NormIQ      = af.sqrt( af.convolve3( IQaf*af.conjg(IQaf) , kernelmask))
        CorrMapaf   = af.convolve3(IQaf,af.conjg(kernel)*kernelmask)
        CorrMapaf   = CorrMapaf/NormIQ/NormPsf
        CorrMap     = CorrMapaf.to_ndarray()
        
    CorrMap = CorrMap.reshape(SizIQ,order='F') 
    
    return CorrMap

# ImRegionalMax
def ImRegionalMax(I,SizKern=3):
    BW = af.dilate(I ,af.constant(1,SizKern,SizKern) ) == I    
    return BW

# InitGaussianFit
def InitGaussianFit(SizFilter=5):
    
    edge    = int(SizFilter/2) + 1    
    roi     = np.linspace(-edge,edge,2*edge+1)
    roix,roiz = np.meshgrid(roi, roi)
    roix    = af.cast(af.from_ndarray(roix),af.Dtype.f32)                              
    roiz    = af.cast(af.from_ndarray(roiz),af.Dtype.f32)
    roi     = af.cast(af.from_ndarray(roi),af.Dtype.f32)    
    
    Hann    = af.cast(af.from_ndarray(np.hanning(2*edge+1)*np.transpose(np.hanning(2*edge+1)[np.newaxis],(1,0))),af.Dtype.f32)
    Hannx   = Hann*roix
    Hannx2  = Hannx*roix
    Hannx3  = Hannx2*roix
    Hannx4  = Hannx3*roix

    Hannz   = Hann*roiz
    Hannz2  = Hannz*roiz
    Hannz3  = Hannz2*roiz
    Hannz4  = Hannz3*roiz

    AtAz    = af.constant(0,3,3,dtype=af.Dtype.f32)
    AtAz[0,0] = af.sum(Hann)
    AtAz[0,1] = af.sum(Hannz)
    AtAz[0,2] = af.sum(Hannz2)
    AtAz[1,0] = af.sum(Hannz)
    AtAz[1,1] = af.sum(Hannz2)
    AtAz[1,2] = af.sum(Hannz3)
    AtAz[2,0] = af.sum(Hannz2)
    AtAz[2,1] = af.sum(Hannz3)
    AtAz[2,2] = af.sum(Hannz4)
    
    AtAx    = af.constant(0,3,3,dtype=af.Dtype.f32)
    AtAx[0,0] = af.sum(Hann)
    AtAx[0,1] = af.sum(Hannx)
    AtAx[0,2] = af.sum(Hannx2)
    AtAx[1,0] = af.sum(Hannx)
    AtAx[1,1] = af.sum(Hannx2)
    AtAx[1,2] = af.sum(Hannx3)
    AtAx[2,0] = af.sum(Hannx2)
    AtAx[2,1] = af.sum(Hannx3)
    AtAx[2,2] = af.sum(Hannx4)
    
    AtAxm = af.inverse(AtAx)
    AtAzm = af.inverse(AtAz)
    
    return roi, Hann, Hannx, Hannx2, Hannz, Hannz2,  AtAxm, AtAzm 

## SUPERLOCFILTER
def SuperLocFilter(CorrMap, IQmap, SizFilter=5, SizEns=5, fit=0):

    edge    = int(SizFilter/2) + 1

    # To GPU    
    CorrMapaf   = af.cast(af.from_ndarray(CorrMap),af.Dtype.c32)
    EnvMapaf    = af.cast(af.from_ndarray(IQmap),af.Dtype.c32)
    
    # Ensemble
    kernelEns   = np.hanning(SizEns) 
    kernelEns   = kernelEns/np.sum(kernelEns)
    kernelEns   = af.moddims( af.from_ndarray( kernelEns ) , 1, 1, SizEns)
    
    CorrMapaf   = CorrMapaf[:,:,0:CorrMapaf.shape[2]-1]*af.conjg(CorrMapaf[:,:,1:CorrMapaf.shape[2]])
    CorrMapaf   = af.sqrt(af.abs(af.convolve3(CorrMapaf ,kernelEns))).to_ndarray()
    
    EnvMapaf    = EnvMapaf[:,:,0:EnvMapaf.shape[2]-1]*af.conjg(EnvMapaf[:,:,1:EnvMapaf.shape[2]])
    EnvMapaf    = af.sqrt(af.abs(af.convolve3(EnvMapaf ,kernelEns))).to_ndarray()
    
    [Nz,Nx,Nt]  = np.shape(CorrMap)
        
    ROI     = af.constant(0,CorrMapaf.shape[0],CorrMapaf.shape[1],CorrMapaf.shape[2])
    ROI[edge:CorrMapaf.shape[0]-edge,edge:CorrMapaf.shape[1]-edge,:] = 1
    
    if fit == 0:
        BW1     = ImRegionalMax(af.from_ndarray(CorrMapaf) , 3)
        BW2     = ImRegionalMax(af.from_ndarray(EnvMapaf) , 3)
        BW      = BW1 & af.dilate(BW2,af.constant(1,3,3))
        BW      = BW*ROI
        
    elif fit == 1:   
        BW      = ImRegionalMax(af.from_ndarray(EnvMapaf) , 3)
        BW      = BW*ROI
        
    BW      = af.moddims(BW,np.prod(BW.shape))
    
    ## 
    ix      = np.linspace(0,CorrMapaf.shape[0]-1,CorrMapaf.shape[0],dtype=int)
    iz      = np.linspace(0,CorrMapaf.shape[1]-1,CorrMapaf.shape[1],dtype=int)
    it      = np.linspace(0,CorrMapaf.shape[2]-1,CorrMapaf.shape[2],dtype=int)
    
    iX,iZ,iT   = np.meshgrid(ix, iz, it)
    
    iX      = iX.reshape(np.prod(iX.shape),order='F')
    iZ      = iZ.reshape(np.prod(iZ.shape),order='F')
    iT      = iT.reshape(np.prod(iT.shape),order='F')
    
    idex    = af.where(BW)
    NbbCount = len(idex)

    CorrMapafVect = af.moddims(af.from_ndarray(CorrMapaf),np.prod(CorrMapaf.shape))
    EnvMapafVect  = af.moddims(af.from_ndarray(EnvMapaf),np.prod(CorrMapaf.shape))
    
    if fit == 0:
        iC  = af.abs(CorrMapafVect[idex])
        [iCs,idxs] = af.sort_index(iC,dim=0,is_ascending=False)
    elif fit == 1:   
        iC  = af.abs(EnvMapafVect[idex])
        [iCs,idxs] = af.sort_index(iC,dim=0,is_ascending=False)
        
    iX = iX[idex[idxs]]
    iZ = iZ[idex[idxs]]
    iT = iT[idex[idxs]]
    
    ##     
    [roi, Hann, Hannx, Hannx2, Hannz, Hannz2,  AtAxm, AtAzm]    = InitGaussianFit(SizFilter)
#    roi     = af.cast(af.from_ndarray(GFit['roi']),af.Dtype.s16) 
#    Hann    = GFit['Hann'] 
#    Hannx   = GFit['Hannx'] 
#    Hannx2  = GFit['Hannx2'] 
#    Hannz   = GFit['Hannz'] 
#    Hannz2  = GFit['Hannz2'] 
    
    Atx     = af.constant(0,3,NbbCount)
    Atz     = af.constant(0,3,NbbCount)
    
    ## 
    for ixroi in range(len(roi)):
        for izroi in range(len(roi)):
            iXroi = iX - edge + ixroi
            iZroi = iZ - edge + izroi
            
            idexroi = af.cast(af.from_ndarray(iZroi + iXroi*Nz + iT*Nz*Nx),af.Dtype.u32)
            
            if fit == 0:
                MapRoiLog = af.log(af.abs(CorrMapafVect[idexroi]))                
            elif fit == 1:
                MapRoiLog = af.log(af.abs(EnvMapafVect[idexroi]))    
        
            Atx[0,:] += af.reorder(af.matmul(MapRoiLog,Hann[izroi,ixroi]),1,0)
            Atx[1,:] += af.reorder(af.matmul(MapRoiLog,Hannx[izroi,ixroi]),1,0)
            Atx[2,:] += af.reorder(af.matmul(MapRoiLog,Hannx2[izroi,ixroi]),1,0)
            
            Atz[0,:] += af.reorder(af.matmul(MapRoiLog,Hann[izroi,ixroi]),1,0)
            Atz[1,:] += af.reorder(af.matmul(MapRoiLog,Hannz[izroi,ixroi]),1,0)
            Atz[2,:] += af.reorder(af.matmul(MapRoiLog,Hannz2[izroi,ixroi]),1,0)
    
    
    mx = af.matmul(AtAxm,Atx)  
    mz = af.matmul(AtAzm,Atz)  
    
    subx = -.5*mx[1,:]/mx[2,:]
    subz = -.5*mz[1,:]/mz[2,:]
        
    PosLoc = af.constant(0,5,NbbCount)
    PosLoc[0,:] = af.reorder(af.from_ndarray(iX),1,0) + subx
    PosLoc[2,:] = af.reorder(af.from_ndarray(iZ),1,0) + subz
    PosLoc[3,:] = af.reorder(iCs,1,0)
    PosLoc[4,:] = af.reorder(af.from_ndarray(iT),1,0)
    
    return PosLoc.to_ndarray()

##  DensityMappingND
def DensityMappingND(pos,xbins=[0,1],ybins=[0,1],zbins=[0,1]):
    
    Sizpos  = np.shape(pos)
    pos     = np.reshape(pos,(Sizpos[0],np.prod(Sizpos[1:])),order='F')
    
    dx      = np.mean(np.diff(xbins))    
    dy      = np.mean(np.diff(ybins))    
    dz      = np.mean(np.diff(zbins))    

    xmin    = min(xbins)
    ymin    = min(ybins)
    zmin    = min(zbins)

    nxbin   = len(xbins)
    nybin   = len(ybins)
    nzbin   = len(zbins)
    
    iX      = np.cast['int16']((pos[0,:] - xmin)/dx)
    iY      = np.cast['int16']((pos[1,:] - ymin)/dy)
    iZ      = np.cast['int16']((pos[2,:] - zmin)/dz)
    
    iX      = (iX>0)*iX 
    iY      = (iY>0)*iY 
    iZ      = (iZ>0)*iZ 

    iX      = ((nxbin - iX)>0)*iX + ((nxbin - iX)<0)*(nxbin-1)
    iY      = ((nybin - iY)>0)*iY + ((nybin - iY)<0)*(nybin-1)
    iZ      = ((nzbin - iZ)>0)*iZ + ((nzbin - iZ)<0)*(nzbin-1)
       
    DensMap = np.zeros((nzbin,nxbin,nybin))
    for index in range(len(iX)):
        DensMap[iZ[index], iX[index], iY[index]]+=1
 
    DensMap = np.squeeze(DensMap)
    return DensMap


## VelocityMappingND
def VelocityMappingND(pos, vel, xbins=[0,1], ybins=[0,1], zbins=[0,1]):
    
    Sizpos  = np.shape(pos)
    Sizvel  = np.shape(vel)

    pos     = np.reshape(pos, (Sizpos[0], np.prod(Sizpos[1:])), order='F')
    vel     = np.reshape(vel, (Sizvel[0], np.prod(Sizvel[1:])), order='F')
    
    dx      = np.mean(np.diff(xbins))    
    dy      = np.mean(np.diff(ybins))    
    dz      = np.mean(np.diff(zbins))    

    xmin    = min(xbins)
    ymin    = min(ybins)
    zmin    = min(zbins)

    nxbin   = len(xbins)
    nybin   = len(ybins)
    nzbin   = len(zbins)
    
    iX      = np.cast['int16']((pos[0,:] - xmin)/dx)
    iY      = np.cast['int16']((pos[1,:] - ymin)/dy)
    iZ      = np.cast['int16']((pos[2,:] - zmin)/dz)
    
    iX      = (iX>0)*iX 
    iY      = (iY>0)*iY 
    iZ      = (iZ>0)*iZ 

    iX      = ((nxbin - iX)>0)*iX + ((nxbin - iX)<0)*(nxbin-1)
    iY      = ((nybin - iY)>0)*iY + ((nybin - iY)<0)*(nybin-1)
    iZ      = ((nzbin - iZ)>0)*iZ + ((nzbin - iZ)<0)*(nzbin-1)
       
    DensMap = np.zeros( (nzbin, nxbin, nybin))
    VelMap  = np.zeros( (3, nzbin, nxbin, nybin))
    
    for index in range(len(iX)):
        DensMap[iZ[index], iX[index], iY[index]] += 1
        VelMap[:, iZ[index], iX[index], iY[index]]  += vel[:,index]
    
    
    VelMap  = VelMap.transpose((1,2,0,3))
    
    VelMap  = np.squeeze(VelMap)/DensMap
    VelMap[np.isnan(VelMap)] = 0
    
    DensMap = np.squeeze(DensMap)
    
    return DensMap, VelMap

##  DensityMappingNDt
def DensityMappingNDt(pos, xbins=[0,1], ybins=[0,1], zbins=[0,1], tbins=[0,1]):
    
    Sizpos  = np.shape(pos)
    pos     = np.reshape(pos,(Sizpos[0],np.prod(Sizpos[1:])),order='F')
    
    dx      = np.mean(np.diff(xbins))    
    dy      = np.mean(np.diff(ybins))    
    dz      = np.mean(np.diff(zbins))
    dt      = np.mean(np.diff(tbins))        

    xmin    = min(xbins)
    ymin    = min(ybins)
    zmin    = min(zbins)
    tmin    = min(tbins)

    nxbin   = len(xbins)
    nybin   = len(ybins)
    nzbin   = len(zbins)
    ntbin   = len(tbins)
        
    iX      = np.cast['int16']((pos[0,:] - xmin)/dx)
    iY      = np.cast['int16']((pos[1,:] - ymin)/dy)
    iZ      = np.cast['int16']((pos[2,:] - zmin)/dz)
    iT      = np.cast['int16']((pos[4,:] - tmin)/dt)
    
    iX      = (iX>0)*iX 
    iY      = (iY>0)*iY 
    iZ      = (iZ>0)*iZ 
    iT      = (iT>0)*iT 

    iX      = ((nxbin - iX)>0)*iX + ((nxbin - iX)<0)*(nxbin-1)
    iY      = ((nybin - iY)>0)*iY + ((nybin - iY)<0)*(nybin-1)
    iZ      = ((nzbin - iZ)>0)*iZ + ((nzbin - iZ)<0)*(nzbin-1)
    iT      = ((ntbin - iT)>0)*iT + ((ntbin - iT)<0)*(ntbin-1)
       
    DensMap = np.zeros((nzbin, nxbin, nybin, ntbin))
    for index in range(len(iX)):
        DensMap[iZ[index], iX[index], iY[index], iT[index]] += 1
 
    DensMap = np.squeeze(DensMap)
    return DensMap

## VelocityMappingNDt
def VelocityMappingNDt(pos, vel, xbins=[0,1], ybins=[0,1], zbins=[0,1], tbins=[0,1]):
    
    Sizpos  = np.shape(pos)
    Sizvel  = np.shape(vel)
    
    pos     = np.reshape(pos, (Sizpos[0], np.prod(Sizpos[1:])), order='F')
    vel     = np.reshape(vel, (Sizvel[0], np.prod(Sizvel[1:])), order='F')
    
    dx      = np.mean(np.diff(xbins))    
    dy      = np.mean(np.diff(ybins))    
    dz      = np.mean(np.diff(zbins))
    dt      = np.mean(np.diff(tbins))        

    xmin    = min(xbins)
    ymin    = min(ybins)
    zmin    = min(zbins)
    tmin    = min(tbins)

    nxbin   = len(xbins)
    nybin   = len(ybins)
    nzbin   = len(zbins)
    ntbin   = len(tbins)
        
    iX      = np.cast['int16']((pos[0,:] - xmin)/dx)
    iY      = np.cast['int16']((pos[1,:] - ymin)/dy)
    iZ      = np.cast['int16']((pos[2,:] - zmin)/dz)
    iT      = np.cast['int16']((pos[4,:] - tmin)/dt)
    
    iX      = (iX>0)*iX 
    iY      = (iY>0)*iY 
    iZ      = (iZ>0)*iZ 
    iT      = (iT>0)*iT 

    iX      = ((nxbin - iX)>0)*iX + ((nxbin - iX)<0)*(nxbin-1)
    iY      = ((nybin - iY)>0)*iY + ((nybin - iY)<0)*(nybin-1)
    iZ      = ((nzbin - iZ)>0)*iZ + ((nzbin - iZ)<0)*(nzbin-1)
    iT      = ((ntbin - iT)>0)*iT + ((ntbin - iT)<0)*(ntbin-1)
       
    DensMap = np.zeros( (nzbin, nxbin, nybin, ntbin))
    VelMap  = np.zeros( (3, nzbin, nxbin, nybin, ntbin))
    
    for index in range(len(iX)):
        DensMap[iZ[index], iX[index], iY[index], iT[index]] += 1
        VelMap[:, iZ[index], iX[index], iY[index], iT[index]]  += vel[:,index]
    
    VelMap[0,:] = VelMap[0,:]/DensMap  
    VelMap[1,:] = VelMap[1,:]/DensMap  
    VelMap[2,:] = VelMap[2,:]/DensMap  
    
    VelMap  = VelMap.transpose((1,2,3,4,0))
    
    VelMap  = np.squeeze(VelMap)
    VelMap[np.isnan(VelMap)] = 0
    
    DensMap = np.squeeze(DensMap)
    
    return DensMap, VelMap