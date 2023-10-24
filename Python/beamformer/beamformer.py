# -*- coding: utf-8 -*-
"""
Created on Tue May 12 14:20:43 2020

@author: jopo86, paxing
"""
import arrayfire as af
import numpy as np

## RF2IQ VERASONICS
def RF2IQVerasonics(RF,sub):
    RF      = af.cast(af.from_ndarray(RF),dtype = af.Dtype.f32)
    IQ      = RF[0::sub,:]
    #idxt    = af.from_ndarray(np.linspace(+1/4, IQ.shape[0] - 1 + 1/4, IQ.shape[0], dtype="float32"))
    #IQ      += 1j*af.approx1(RF[1::sub,:], idxt, method=af.INTERP.LINEAR_COSINE, off_grid = 0.0)
    IQ      += 1j*RF[1::sub,:]
    return IQ.to_ndarray()

## afsinc
def afsinc(input_array):
    output_array = af.sin(np.pi*input_array)/(np.pi*input_array)
    output_array[input_array==0]=1
    return output_array


## nextpow2
def nextpow2(a):
    b = np.ceil(np.log(a)/np.log(2))
    return b


## BfIQFlatLinear
def BfIQFlatLinear(Raw, pmig):

    if len(np.shape(Raw)) > 3:
        SizRaw = np.shape(Raw)
        Raw = af.from_ndarray(np.reshape(Raw,(SizRaw[0],SizRaw[1],-1), order='F'))         # reshape Nt
        SizRaw = np.shape(Raw)
    else:
        Raw = af.from_ndarray(Raw)         # reshape Nt

    [NtIQ,Nchannel,Nevent] = np.shape(Raw)


    if 'c' not in pmig:
        pmig['c'] = 1540
        print('A Propagation velocity must be set ')
        print(pmig['c'])
        print(' m/s has been set as default')

    if 'fc' not in pmig:
        pmig['fc'] = 3e6
        print('A Central frequency must be set ')
        print(pmig['fc']/1e6)
        print(' MHz has been set as default')

    if 'lbd' not in pmig:
        pmig['lbd'] = pmig['c']/pmig['fc']

    if 'fs' not in pmig:
        pmig['fs'] = 4*pmig['fc']
        print('A Sampling frequency must be set ')
        print(pmig['fs'] /1e6)
        print('MHz has been set as default')

    if 'sub' not in pmig:
        pmig['sub'] = 2

    if 'fsIQ' not in pmig:
        pmig['fsIQ'] = pmig['fs']/pmig['sub']

    if 'pitch' not in pmig:
        pmig['pitch'] = pmig['lbd']

    if 'width' not in pmig:
        pmig['width'] = pmig['lbd']/2

    Aperture = (Nchannel-1)*pmig['pitch']
    X0 = Aperture/2

    if 'theta' not in pmig:
        pmig['theta'] = 0

    Ntx = len(pmig['theta'])

    if 't0' not in pmig:
        pmig['t0'] = 0

    # Grid definition
    if 'dx' not in pmig:
        pmig['dx'] = pmig['pitch']

    if 'dz' not in pmig:
        pmig['dz'] = pmig['lbd']

    if 'Nx' not in pmig:
        pmig['Nx'] = Nchannel

    if 'Nz' not in pmig:
        pmig['Nz'] = Nchannel

    Npix = pmig['Nx']*pmig['Nz']

    if 'Xmin' not in pmig:
        pmig['Xmin'] = 0 -(pmig['Nx']-1)*pmig['dx']/2

    if 'Zmin' not in pmig:
        pmig['Zmin'] = 0

    if 'fnum' not in pmig:
        pmig['fnum'] = 1

    theta = af.cast(af.from_ndarray(pmig['theta']),af.Dtype.f32)
    invc  = 1/pmig['c']

    Nframe = int(Nevent/Ntx)
    Raw = af.moddims(Raw,NtIQ,Nchannel,Ntx,Nframe)
    if Raw.is_real():
        Raw = Raw[0:NtIQ:2,:,:]+1j*Raw[1:NtIQ:2,:,:]
        NtIQ = int(NtIQ/2)

    Raw = af.cast(Raw,af.Dtype.c32)

    ## Linear Array definition
    xele = np.linspace(0, Nchannel-1, Nchannel,dtype="float32")*pmig['pitch'] - X0    # SHAPE MUST BE [1, Nchannel]
    xele = af.cast(af.from_ndarray(xele.reshape(1,1,Nchannel,order='F')),af.Dtype.f32)
    yele = af.constant(0,1,1,Nchannel)                       # SHAPE MUST BE [1, Nchannel]
    zele = af.constant(0,1,1,Nchannel)                       # SHAPE MUST BE [1, Nchannel]

    ## Cartesian Grid definition
    xp = np.linspace(0, pmig['Nx']-1, pmig['Nx'],dtype="float32")*pmig['dx'] + pmig['Xmin']
    zp = np.linspace(0, pmig['Nz']-1, pmig['Nz'],dtype="float32")*pmig['dz'] + pmig['Zmin']

    xp, zp = np.meshgrid(xp, zp)

    Zpix = af.cast(af.from_ndarray(zp.reshape(1,Npix,order='F')),af.Dtype.f32)
    Xpix = af.cast(af.from_ndarray(xp.reshape(1,Npix,order='F')),af.Dtype.f32)

    Ypix = af.constant(0, 1,Npix)

    demoddelay  = af.reorder(4*np.pi*pmig['fc']*Zpix*invc,1,0)
    phaseRot    = 2*np.pi*pmig['fc']/pmig['fsIQ']

    r_e = af.join(0, xele, yele, third=zele)                      # SHAPE MUST BE [3, Nchannel]
    r_p = af.join(0, Xpix, Ypix, third=Zpix)

    RXdelay = af.reorder(af.sqrt(af.sum( (af.tile(r_p,1,1,Nchannel) - af.tile(r_e,1,Npix) )**2,dim=0) )*invc,1,2,0)
    TXdelay = af.reorder(af.matmul(af.cos(af.abs(theta)),Zpix)*invc,1,0)
    TXdelay += af.reorder(af.matmul(af.sin(af.abs(theta))*(theta>0),(Xpix + X0))*invc,1,0)
    TXdelay += af.reorder(af.matmul(af.sin(af.abs(theta))*(theta<0),(X0 - Xpix))*invc,1,0)
    TXdelay = TXdelay + pmig['t0']
    idex = (af.tile(RXdelay,1,1,Ntx) + af.tile(af.reorder(TXdelay,0,2,1),1,Nchannel,1))*pmig['fsIQ']
    deltaDelay = phaseRot*idex - af.tile(demoddelay,1,Nchannel,Ntx)


    ## Fnumber
    dist     = af.abs(af.tile(r_p[0,:],1,1,Nchannel) - af.tile(r_e[0,:],1,Npix))

    fnummask = af.cast(af.tile(af.reorder(2*dist*pmig['fnum']<af.tile(r_p[2,:],1,1,Nchannel),1,2,0),1,1,Ntx),af.Dtype.f32)
    fnummask = fnummask/af.tile(af.sum(fnummask,1),1,Nchannel,1)
    #print(fnummask.to_ndarray().nonzero())




    IQ = np.zeros((Npix,Nframe),dtype=float,order='F')+1j*np.zeros((Npix,Nframe),dtype=float,order='F')
    for iframe in range(Nframe):

        IQ[:,iframe]  = af.sum(af.sum(af.approx1(Raw[:,:,:,iframe],idex)*af.exp(-1j*deltaDelay)*fnummask,1),2).to_ndarray()
        #print(IQ)

    IQ = np.reshape(IQ,(pmig['Nz'] ,pmig['Nx'] , Nframe),order='F')
    return IQ



def BfIQFlatLinear_rx(Raw, Law, pmig):

    if len(np.shape(Raw)) > 3:
        SizRaw = np.shape(Raw)
        Raw = af.from_ndarray(np.reshape(Raw,(SizRaw[0],SizRaw[1],-1), order='F'))         # reshape Nt
        SizRaw = np.shape(Raw)
    else:
        Raw = af.from_ndarray(Raw)         # reshape Nt


    [NtIQ,Nchannel,Nevent] = np.shape(Raw)

    phase_delay = af.cast(af.from_ndarray(np.reshape(np.unwrap(np.angle(Law)),(1,Nchannel), order= 'F'))/(2*np.pi*pmig['fc']),af.Dtype.f32)


    if 'c' not in pmig:
        pmig['c'] = 1540
        print('A Propagation velocity must be set ')
        print(pmig['c'])
        print(' m/s has been set as default')

    if 'fc' not in pmig:
        pmig['fc'] = 3e6
        print('A Central frequency must be set ')
        print(pmig['fc']/1e6)
        print(' MHz has been set as default')

    if 'lbd' not in pmig:
        pmig['lbd'] = pmig['c']/pmig['fc']

    if 'fs' not in pmig:
        pmig['fs'] = 4*pmig['fc']
        print('A Sampling frequency must be set ')
        print(pmig['fs'] /1e6)
        print('MHz has been set as default')

    if 'sub' not in pmig:
        pmig['sub'] = 2

    if 'fsIQ' not in pmig:
        pmig['fsIQ'] = pmig['fs']/pmig['sub']

    if 'pitch' not in pmig:
        pmig['pitch'] = pmig['lbd']

    if 'width' not in pmig:
        pmig['width'] = pmig['lbd']/2

    Aperture = (Nchannel-1)*pmig['pitch']
    X0 = Aperture/2

    if 'theta' not in pmig:
        pmig['theta'] = 0

    Ntx = len(pmig['theta'])

    if 't0' not in pmig:
        pmig['t0'] = 0

    # Grid definition
    if 'dx' not in pmig:
        pmig['dx'] = pmig['pitch']

    if 'dz' not in pmig:
        pmig['dz'] = pmig['lbd']

    if 'Nx' not in pmig:
        pmig['Nx'] = Nchannel

    if 'Nz' not in pmig:
        pmig['Nz'] = Nchannel

    Npix = pmig['Nx']*pmig['Nz']

    if 'Xmin' not in pmig:
        pmig['Xmin'] = 0 -(pmig['Nx']-1)*pmig['dx']/2

    if 'Zmin' not in pmig:
        pmig['Zmin'] = 0

    if 'fnum' not in pmig:
        pmig['fnum'] = 1

    #start_time = time.time()
    theta = af.cast(af.from_ndarray(pmig['theta']),af.Dtype.f32)
    invc  = 1/pmig['c']

    Nframe = int(Nevent/Ntx)
    Raw = af.moddims(Raw,NtIQ,Nchannel,Ntx,Nframe)
    if Raw.is_real():
        Raw = Raw[0:NtIQ:2,:,:]+1j*Raw[1:NtIQ:2,:,:]
        NtIQ = int(NtIQ/2)

    Raw = af.cast(Raw,af.Dtype.c32)

    ## Linear Array definition
    xele = np.linspace(0, Nchannel-1, Nchannel,dtype="float32")*pmig['pitch'] - X0    # SHAPE MUST BE [1, Nchannel]
    xele = af.cast(af.from_ndarray(xele.reshape(1,1,Nchannel,order='F')),af.Dtype.f32)
    yele = af.constant(0,1,1,Nchannel)                       # SHAPE MUST BE [1, Nchannel]
    zele = af.constant(0,1,1,Nchannel)                       # SHAPE MUST BE [1, Nchannel]

    ## Cartesian Grid definition
    xp = np.linspace(0, pmig['Nx']-1, pmig['Nx'],dtype="float32")*pmig['dx'] + pmig['Xmin']
    zp = np.linspace(0, pmig['Nz']-1, pmig['Nz'],dtype="float32")*pmig['dz'] + pmig['Zmin']

    xp, zp = np.meshgrid(xp, zp)

    Zpix = af.cast(af.from_ndarray(zp.reshape(1,Npix,order='F')),af.Dtype.f32)
    Xpix = af.cast(af.from_ndarray(xp.reshape(1,Npix,order='F')),af.Dtype.f32)

    Ypix = af.constant(0, 1,Npix)

    demoddelay  = af.reorder(4*np.pi*pmig['fc']*Zpix*invc,1,0)
    phaseRot    = 2*np.pi*pmig['fc']/pmig['fsIQ']

    r_e = af.join(0, xele, yele, third=zele)                      # SHAPE MUST BE [3, Nchannel]
    r_p = af.join(0, Xpix, Ypix, third=Zpix)

    RXdelay = af.reorder(af.sqrt(af.sum( (af.tile(r_p,1,1,Nchannel) - af.tile(r_e,1,Npix) )**2,dim=0) )*invc,1,2,0)


    RXdelay += af.tile(phase_delay,Npix,1)

    TXdelay = af.reorder(af.matmul(af.cos(af.abs(theta)),Zpix)*invc,1,0)
    TXdelay += af.reorder(af.matmul(af.sin(af.abs(theta))*(theta>0),(Xpix + X0))*invc,1,0)
    TXdelay += af.reorder(af.matmul(af.sin(af.abs(theta))*(theta<0),(X0 - Xpix))*invc,1,0)
    TXdelay = TXdelay + pmig['t0']

    idex = (af.tile(RXdelay,1,1,Ntx) + af.tile(af.reorder(TXdelay,0,2,1),1,Nchannel,1))*pmig['fsIQ']
    deltaDelay = phaseRot*idex - af.tile(demoddelay,1,Nchannel,Ntx)

    ## Fnumber
    dist     = af.abs(af.tile(r_p[0,:],1,1,Nchannel) - af.tile(r_e[0,:],1,Npix))
    fnummask = af.cast(af.tile(af.reorder(2*dist*pmig['fnum']<af.tile(r_p[2,:],1,1,Nchannel),1,2,0),1,1,Ntx),af.Dtype.f32)
    fnummask = fnummask/af.tile(af.sum(fnummask,1),1,Nchannel,1)

    IQ = np.zeros((Npix,Nframe),dtype=float,order='F')+1j*np.zeros((Npix,Nframe),dtype=float,order='F')
    for iframe in range(Nframe):
        IQ[:,iframe]  = af.sum(af.sum(af.approx1(Raw[:,:,:,iframe],idex,method=af.INTERP.LINEAR_COSINE)*af.exp(-1j*deltaDelay)*fnummask,1),2).to_ndarray()

    IQ = np.reshape(IQ,(pmig['Nz'] ,pmig['Nx'] , Nframe),order='F')
    return IQ


## BfIQFlatLinear
def BfIQFlatLinear_nosum(Raw, pmig, pos, patch_size=(17,17)):

    if len(np.shape(Raw)) > 3:
        SizRaw = np.shape(Raw)
        Raw = af.from_ndarray(np.reshape(Raw,(SizRaw[0],SizRaw[1],-1), order='F'))         # reshape Nt
        SizRaw = np.shape(Raw)
    else:
        Raw = af.from_ndarray(Raw)         # reshape Nt

    [NtIQ,Nchannel,Nevent] = np.shape(Raw)

    if 'c' not in pmig:
        pmig['c'] = 1540
        print('A Propagation velocity must be set ')
        print(pmig['c'])
        print(' m/s has been set as default')

    if 'fc' not in pmig:
        pmig['fc'] = 3e6
        print('A Central frequency must be set ')
        print(pmig['fc']/1e6)
        print(' MHz has been set as default')

    if 'lbd' not in pmig:
        pmig['lbd'] = pmig['c']/pmig['fc']

    if 'fs' not in pmig:
        pmig['fs'] = 4*pmig['fc']
        print('A Sampling frequency must be set ')
        print(pmig['fs'] /1e6)
        print('MHz has been set as default')

    if 'sub' not in pmig:
        pmig['sub'] = 2

    if 'fsIQ' not in pmig:
        pmig['fsIQ'] = pmig['fs']/pmig['sub']

    if 'pitch' not in pmig:
        pmig['pitch'] = pmig['lbd']

    if 'width' not in pmig:
        pmig['width'] = pmig['lbd']/2

    Aperture = (Nchannel-1)*pmig['pitch']
    X0 = Aperture/2

    if 'theta' not in pmig:
        pmig['theta'] = 0

    Ntx = len(pmig['theta'])

    if 't0' not in pmig:
        pmig['t0'] = 0

    # Grid definition
    if 'dx' not in pmig:
        pmig['dx'] = pmig['pitch']

    if 'dz' not in pmig:
        pmig['dz'] = pmig['lbd']

    if 'Nx' not in pmig:
        pmig['Nx'] = Nchannel

    if 'Nz' not in pmig:
        pmig['Nz'] = Nchannel

    Npix = pmig['Nx']*pmig['Nz']

    if 'Xmin' not in pmig:
        pmig['Xmin'] = 0 -(pmig['Nx']-1)*pmig['dx']/2

    if 'Zmin' not in pmig:
        pmig['Zmin'] = 0

    if 'fnum' not in pmig:
        pmig['fnum'] = 1

    #start_time = time.time()
    theta = af.cast(af.from_ndarray(pmig['theta']),af.Dtype.f32)
    invc  = 1/pmig['c']

    Nframe = int(Nevent/Ntx)
    Raw = af.moddims(Raw,NtIQ,Nchannel,Ntx,Nframe)
    if Raw.is_real():
        Raw = Raw[0:NtIQ:2,:,:]+1j*Raw[1:NtIQ:2,:,:]
        NtIQ = int(NtIQ/2)

    Raw = af.cast(Raw,af.Dtype.c32)

    ## Linear Array definition
    xele = np.linspace(0, Nchannel-1, Nchannel,dtype="float32")*pmig['pitch'] - X0    # SHAPE MUST BE [1, Nchannel]
    xele = af.cast(af.from_ndarray(xele.reshape(1,1,Nchannel,order='F')),af.Dtype.f32)
    yele = af.constant(0,1,1,Nchannel)                       # SHAPE MUST BE [1, Nchannel]
    zele = af.constant(0,1,1,Nchannel)                       # SHAPE MUST BE [1, Nchannel]

    ## Cartesian Grid definition
    xp = np.linspace(0, pmig['Nx']-1, pmig['Nx'],dtype="float32")*pmig['dx'] + pmig['Xmin']
    zp = np.linspace(0, pmig['Nz']-1, pmig['Nz'],dtype="float32")*pmig['dz'] + pmig['Zmin']

    xp, zp = np.meshgrid(xp, zp)

    Zpix = af.cast(af.from_ndarray(zp.reshape(1,Npix,order='F')),af.Dtype.f32)
    Xpix = af.cast(af.from_ndarray(xp.reshape(1,Npix,order='F')),af.Dtype.f32)

    Ypix = af.constant(0, 1,Npix)

    demoddelay  = af.reorder(4*np.pi*pmig['fc']*Zpix*invc,1,0)
    phaseRot    = 2*np.pi*pmig['fc']/pmig['fsIQ']

    r_e = af.join(0, xele, yele, third=zele)                      # SHAPE MUST BE [3, Nchannel]
    r_p = af.join(0, Xpix, Ypix, third=Zpix)

    RXdelay = af.reorder(af.sqrt(af.sum( (af.tile(r_p,1,1,Nchannel) - af.tile(r_e,1,Npix) )**2,dim=0) )*invc,1,2,0)

    TXdelay = af.reorder(af.matmul(af.cos(af.abs(theta)),Zpix)*invc,1,0)
    TXdelay += af.reorder(af.matmul(af.sin(af.abs(theta))*(theta>0),(Xpix + X0))*invc,1,0)
    TXdelay += af.reorder(af.matmul(af.sin(af.abs(theta))*(theta<0),(X0 - Xpix))*invc,1,0)
    TXdelay = TXdelay + pmig['t0']

    idex = (af.tile(RXdelay,1,1,Ntx) + af.tile(af.reorder(TXdelay,0,2,1),1,Nchannel,1))*pmig['fsIQ']
    deltaDelay = phaseRot*idex - af.tile(demoddelay,1,Nchannel,Ntx)

    ## Fnumber
    #dist     = af.abs(af.tile(r_p[0,:],1,1,Nchannel) - af.tile(r_e[0,:],1,Npix))
    #fnummask = af.cast(af.tile(af.reorder(2*dist*pmig['fnum']<af.tile(r_p[2,:],1,1,Nchannel),1,2,0),1,1,Ntx),af.Dtype.f32)
    #fnummask = fnummask/af.tile(af.sum(fnummask,1),1,Nchannel,1)

    IQ = np.zeros((Npix,Nchannel, Ntx, Nframe),dtype=complex,order='F')
    for iframe in range(Nframe):
        IQ[:,:,:,iframe]  = (af.approx1(Raw[:,:,:,iframe],idex,method=af.INTERP.LINEAR_COSINE)*af.exp(-1j*deltaDelay)).to_ndarray()

    IQ = np.reshape(IQ,(pmig['Nz'] ,pmig['Nx'] , Nchannel, Ntx, Nframe),order='F')

    idz = int((pos[2]-pmig['Zmin'])/pmig['dz'])

    patch_z = patch_size[0]-1
    patch_x = patch_size[1]-1
    if idz-patch_z//2<0:
        idz+=patch_z//2-idz
    elif idz+patch_z//2>=pmig['Nz']:
        idz-=idz+patch_z//2-pmig['Nz']+1

    idx = int(pos[0]/pmig['dx'])+pmig['Nx']//2

    if idx-patch_x//2<0:
        idx+=patch_x//2-idx
    elif idx+patch_x//2>=pmig['Nx']:
        idx-=idx+patch_x//2-pmig['Nx']+1

    return IQ[idz-patch_z//2:idz+patch_z//2+1, idx-patch_x//2:idx+patch_x//2+1,...]

def BfIQFlatLinear_rx_nosum(Raw, Law, pmig, pos, patch_size=(33,33)):

    if len(np.shape(Raw)) > 3:
        SizRaw = np.shape(Raw)
        Raw = af.from_ndarray(np.reshape(Raw,(SizRaw[0],SizRaw[1],-1), order='F'))         # reshape Nt
        SizRaw = np.shape(Raw)
    else:
        Raw = af.from_ndarray(Raw)         # reshape Nt

    [NtIQ,Nchannel,Nevent] = np.shape(Raw)

    phase_delay = af.cast(af.from_ndarray(np.reshape(np.unwrap(np.angle(Law)),(1,Nchannel), order= 'F'))/(2*np.pi*pmig['fc']),af.Dtype.f32)


    if 'c' not in pmig:
        pmig['c'] = 1540
        print('A Propagation velocity must be set ')
        print(pmig['c'])
        print(' m/s has been set as default')

    if 'fc' not in pmig:
        pmig['fc'] = 3e6
        print('A Central frequency must be set ')
        print(pmig['fc']/1e6)
        print(' MHz has been set as default')

    if 'lbd' not in pmig:
        pmig['lbd'] = pmig['c']/pmig['fc']

    if 'fs' not in pmig:
        pmig['fs'] = 4*pmig['fc']
        print('A Sampling frequency must be set ')
        print(pmig['fs'] /1e6)
        print('MHz has been set as default')

    if 'sub' not in pmig:
        pmig['sub'] = 2

    if 'fsIQ' not in pmig:
        pmig['fsIQ'] = pmig['fs']/pmig['sub']

    if 'pitch' not in pmig:
        pmig['pitch'] = pmig['lbd']

    if 'width' not in pmig:
        pmig['width'] = pmig['lbd']/2

    Aperture = (Nchannel-1)*pmig['pitch']
    X0 = Aperture/2

    if 'theta' not in pmig:
        pmig['theta'] = 0

    Ntx = len(pmig['theta'])

    if 't0' not in pmig:
        pmig['t0'] = 0

    # Grid definition
    if 'dx' not in pmig:
        pmig['dx'] = pmig['pitch']

    if 'dz' not in pmig:
        pmig['dz'] = pmig['lbd']

    if 'Nx' not in pmig:
        pmig['Nx'] = Nchannel

    if 'Nz' not in pmig:
        pmig['Nz'] = Nchannel

    Npix = pmig['Nx']*pmig['Nz']

    if 'Xmin' not in pmig:
        pmig['Xmin'] = 0 -(pmig['Nx']-1)*pmig['dx']/2

    if 'Zmin' not in pmig:
        pmig['Zmin'] = 0

    if 'fnum' not in pmig:
        pmig['fnum'] = 1

    #start_time = time.time()
    theta = af.cast(af.from_ndarray(pmig['theta']),af.Dtype.f32)
    invc  = 1/pmig['c']

    Nframe = int(Nevent/Ntx)
    Raw = af.moddims(Raw,NtIQ,Nchannel,Ntx,Nframe)
    if Raw.is_real():
        Raw = Raw[0:NtIQ:2,:,:]+1j*Raw[1:NtIQ:2,:,:]
        NtIQ = int(NtIQ/2)

    Raw = af.cast(Raw,af.Dtype.c32)

    ## Linear Array definition
    xele = np.linspace(0, Nchannel-1, Nchannel,dtype="float32")*pmig['pitch'] - X0    # SHAPE MUST BE [1, Nchannel]
    xele = af.cast(af.from_ndarray(xele.reshape(1,1,Nchannel,order='F')),af.Dtype.f32)
    yele = af.constant(0,1,1,Nchannel)                       # SHAPE MUST BE [1, Nchannel]
    zele = af.constant(0,1,1,Nchannel)                       # SHAPE MUST BE [1, Nchannel]

    ## Cartesian Grid definition
    xp = np.linspace(0, pmig['Nx']-1, pmig['Nx'],dtype="float32")*pmig['dx'] + pmig['Xmin']
    zp = np.linspace(0, pmig['Nz']-1, pmig['Nz'],dtype="float32")*pmig['dz'] + pmig['Zmin']

    xp, zp = np.meshgrid(xp, zp)

    Zpix = af.cast(af.from_ndarray(zp.reshape(1,Npix,order='F')),af.Dtype.f32)
    Xpix = af.cast(af.from_ndarray(xp.reshape(1,Npix,order='F')),af.Dtype.f32)

    Ypix = af.constant(0, 1,Npix)

    demoddelay  = af.reorder(4*np.pi*pmig['fc']*Zpix*invc,1,0)
    phaseRot    = 2*np.pi*pmig['fc']/pmig['fsIQ']

    r_e = af.join(0, xele, yele, third=zele)                      # SHAPE MUST BE [3, Nchannel]
    r_p = af.join(0, Xpix, Ypix, third=Zpix)

    RXdelay = af.reorder(af.sqrt(af.sum( (af.tile(r_p,1,1,Nchannel) - af.tile(r_e,1,Npix) )**2,dim=0) )*invc,1,2,0)
    RXdelay += af.tile(phase_delay,Npix,1)

    TXdelay = af.reorder(af.matmul(af.cos(af.abs(theta)),Zpix)*invc,1,0)
    TXdelay += af.reorder(af.matmul(af.sin(af.abs(theta))*(theta>0),(Xpix + X0))*invc,1,0)
    TXdelay += af.reorder(af.matmul(af.sin(af.abs(theta))*(theta<0),(X0 - Xpix))*invc,1,0)
    TXdelay = TXdelay + pmig['t0']

    idex = (af.tile(RXdelay,1,1,Ntx) + af.tile(af.reorder(TXdelay,0,2,1),1,Nchannel,1))*pmig['fsIQ']
    deltaDelay = phaseRot*idex - af.tile(demoddelay,1,Nchannel,Ntx)

    ## Fnumber
    #dist     = af.abs(af.tile(r_p[0,:],1,1,Nchannel) - af.tile(r_e[0,:],1,Npix))
    #fnummask = af.cast(af.tile(af.reorder(2*dist*pmig['fnum']<af.tile(r_p[2,:],1,1,Nchannel),1,2,0),1,1,Ntx),af.Dtype.f32)
    #fnummask = fnummask/af.tile(af.sum(fnummask,1),1,Nchannel,1)

    IQ = np.zeros((Npix,Nchannel, Ntx, Nframe),dtype=complex,order='F')
    for iframe in range(Nframe):
        IQ[:,:,:,iframe]  = (af.approx1(Raw[:,:,:,iframe],idex,method=af.INTERP.LINEAR_COSINE)*af.exp(-1j*deltaDelay)).to_ndarray()

    IQ = np.reshape(IQ,(pmig['Nz'] ,pmig['Nx'] , Nchannel, Ntx, Nframe),order='F')

    idz = int((pos[2]-pmig['Zmin'])/pmig['dz'])

    patch_z = patch_size[0]-1
    patch_x = patch_size[1]-1
    if idz-patch_z//2<0:
        idz+=patch_z//2-idz
    elif idz+patch_z//2>=pmig['Nz']:
        idz-=idz+patch_z//2-pmig['Nz']+1

    idx = int(pos[0]/pmig['dx'])+pmig['Nx']//2

    if idx-patch_x//2<0:
        idx+=patch_x//2-idx
    elif idx+patch_x//2>=pmig['Nx']:
        idx-=idx+patch_x//2-pmig['Nx']+1

    return IQ[idz-patch_z//2:idz+patch_z//2+1, idx-patch_x//2:idx+patch_x//2+1,...]


##
def BfIQFlatLinearGFOR(Raw, pmig, batchsize=100):

    if len(np.shape(Raw)) > 3:
        SizRaw = np.shape(Raw)
        Raw = af.from_ndarray(np.reshape(Raw,(SizRaw[0],SizRaw[1],-1), order='F'))         # reshape Nt
        SizRaw = np.shape(Raw)
    else:
        Raw = af.from_ndarray(Raw)         # reshape Nt

    [NtIQ,Nchannel,Nevent] = np.shape(Raw)

    #print(time.time() - start_time)

    if 'c' not in pmig:
        pmig['c'] = 1540
        print('A Propagation velocity must be set ')
        print(pmig['c'])
        print(' m/s has been set as default')

    if 'fc' not in pmig:
        pmig['fc'] = 3e6
        print('A Central frequency must be set ')
        print(pmig['fc']/1e6)
        print(' MHz has been set as default')

    if 'lbd' not in pmig:
        pmig['lbd'] = pmig['c']/pmig['fc']

    if 'fs' not in pmig:
        pmig['fs'] = 4*pmig['fc']
        print('A Sampling frequency must be set ')
        print(pmig['fs'] /1e6)
        print('MHz has been set as default')

    if 'sub' not in pmig:
        pmig['sub'] = 2

    if 'fsIQ' not in pmig:
        pmig['fsIQ'] = pmig['fs']/pmig['sub']

    if 'pitch' not in pmig:
        pmig['pitch'] = pmig['lbd']

    if 'width' not in pmig:
        pmig['width'] = pmig['lbd']/2

    Aperture = (Nchannel-1)*pmig['pitch']
    X0 = Aperture/2

    if 'theta' not in pmig:
        pmig['theta'] = 0

    Ntx = len(pmig['theta'])

    if 't0' not in pmig:
        pmig['t0'] = 0

    # Grid definition
    if 'dx' not in pmig:
        pmig['dx'] = pmig['pitch']

    if 'dz' not in pmig:
        pmig['dz'] = pmig['lbd']

    if 'Nx' not in pmig:
        pmig['Nx'] = Nchannel

    if 'Nz' not in pmig:
        pmig['Nz'] = Nchannel

    Npix = pmig['Nx']*pmig['Nz']

    if 'Xmin' not in pmig:
        pmig['Xmin'] = 0 -(pmig['Nx']-1)*pmig['dx']/2

    if 'Zmin' not in pmig:
        pmig['Zmin'] = 0

    if 'order' not in pmig:
        pmig['order'] = 4

    #start_time = time.time()
    theta = af.cast(af.from_ndarray(pmig['theta']),af.Dtype.f32)
    invc  = 1/pmig['c']

    Nframe = int(Nevent/Ntx)
    Raw = af.moddims(Raw,NtIQ,Nchannel,Ntx,Nframe)
    if Raw.is_real():
        Raw = Raw[0:NtIQ:2,:,:]+1j*Raw[1:NtIQ:2,:,:]
        NtIQ = int(NtIQ/2)

    Raw = af.cast(Raw,af.Dtype.c32)

    ## Linear Array definition
    xele = np.linspace(0, Nchannel-1, Nchannel,dtype="float32")*pmig['pitch'] - X0
    xele = af.cast(af.from_ndarray(xele.reshape(1,Nchannel,order='F')),af.Dtype.f32)
    yele = af.constant(0,1,Nchannel)                          # SHAPE MUST BE [1, Nchannel]
    zele = af.constant(0,1,Nchannel)                          # SHAPE MUST BE [1, Nchannel]

    r_e = af.join(0, xele, yele, third=zele)                      # SHAPE MUST BE [3, Nchannel]

    ## Cartesian Grid definition
    xp = np.linspace(0, pmig['Nx']-1, pmig['Nx'],dtype="float32")*pmig['dx'] + pmig['Xmin']
    zp = np.linspace(0, pmig['Nz']-1, pmig['Nz'],dtype="float32")*pmig['dz'] + pmig['Zmin']

    xp, zp = np.meshgrid(xp, zp)

    Zpix = af.cast(af.from_ndarray(zp.reshape(1,Npix,order='F')),af.Dtype.f32)
    Xpix = af.cast(af.from_ndarray(xp.reshape(1,Npix,order='F')),af.Dtype.f32)
    Ypix = af.constant(0, 1,Npix)

    r_p  = af.join(0, Xpix, Ypix, third=Zpix)

    demoddelay  = af.reorder(4*np.pi*pmig['fc']*Zpix*invc,1,0)
    phaseRot    = 2*np.pi*pmig['fc']/pmig['fsIQ']

    TXdelay = af.reorder(af.matmul(af.cos(af.abs(theta)),Zpix)*invc,1,0)
    TXdelay += af.reorder(af.matmul(af.sin(af.abs(theta))*(theta>0),(Xpix + X0))*invc,1,0)
    TXdelay += af.reorder(af.matmul(af.sin(af.abs(theta))*(theta<0),(X0 - Xpix))*invc,1,0)
    TXdelay = TXdelay + pmig['t0']


    IQ = np.zeros((pmig['Nz']*pmig['Nx'],batchsize,int(Nframe/batchsize)),dtype=float,order='F')+1j*np.zeros((pmig['Nz']*pmig['Nx'],batchsize,int(Nframe/batchsize)),dtype=float,order='F')

    for ibatch in range(int(Nframe/batchsize)):

        IQtmp = af.constant(0,pmig['Nz']*pmig['Nx'], batchsize, dtype=af.Dtype.c32)

        for ich in range(Nchannel):

            RXdelay = af.reorder(af.sqrt(af.sum( (r_p - af.tile(r_e[:,ich],1,Npix) )**2,dim=0) )*invc,1,0)

            ## Fnumber
            dist     = af.abs(r_p[0,:] - af.tile(r_e[0,ich],1,Npix) )
            fnummask = af.cast(af.reorder( 2*dist*pmig['fnum']<r_p[2,:] ,1,0),af.Dtype.f32)

            idex = (af.tile(RXdelay,1,Ntx) + TXdelay)*pmig['fsIQ']
            deltaDelay = phaseRot*idex - af.tile(demoddelay,1,Ntx)

            idex[idex<pmig['order']/2] = pmig['order']/2
            idex[idex>(NtIQ-pmig['order']/2)] = NtIQ - pmig['order']/2

            for ia in range(Ntx):

                for iorder in range(pmig['order']):

                    idexInt = af.cast(idex[:,ia] ,af.Dtype.u8) + iorder - pmig['order']/2

                    weight  = afsinc( (idex[:,ia] - idexInt ))
                    weight  = weight*af.exp(-1j*deltaDelay[:,ia])
                    weight  = weight*fnummask

                    IQtmp  += af.reorder(Raw[idexInt,ich,ia,ibatch*batchsize:(ibatch+1)*batchsize],0,3,1,2)*af.tile(weight,1,batchsize)

        IQ[:,:,ibatch] = IQtmp.to_ndarray()
    IQ = np.reshape(IQ,(pmig['Nz'],pmig['Nx'], Nframe),order='F')
    return IQ

## BfRFFlatLinear
def BfRFFlatLinear(Raw, pmig):

    #start_time = time.time()
    if len(np.shape(Raw)) > 3:
        SizRaw = np.shape(Raw)
        Raw = af.from_ndarray(np.reshape(Raw,(SizRaw[0],SizRaw[1],-1), order='F'))         # reshape Nt
        SizRaw = np.shape(Raw)
    else:
        Raw = af.from_ndarray(Raw)         # reshape Nt

    Raw = af.cast(Raw,dtype=af.Dtype.f32)
    [Nt,Nchannel,Nevent] = np.shape(Raw)

    #print(time.time() - start_time)

    if hasattr(pmig,'c')==0:
        pmig.c = 1540
        print('A Propagation velocity must be set ')
        print(pmig.c)
        print(' m/s has been set as default')

    if hasattr(pmig,'fc')==0:
        pmig.fc = 3e6

    if hasattr(pmig,'lbd')==0:
        pmig.lbd = pmig.c/pmig.fc

    if hasattr(pmig,'fs')==0:
        pmig.fs = 4*pmig.fc

    if hasattr(pmig,'sub')==0:
        pmig.sub = 2


    if hasattr(pmig,'pitch')==0:
        pmig.pitch = pmig.lbd

    Aperture = (Nchannel-1)*pmig.pitch
    X0 = Aperture/2

    if hasattr(pmig,'theta')==0:
        pmig.theta = 0

    Ntx = len(pmig.theta)

    if hasattr(pmig,'t0')==0:
        pmig.t0 = 0

    # Grid definition
    if hasattr(pmig,'dx')==0:
        pmig.dx = pmig.lbd

    if hasattr(pmig,'dz')==0:
        pmig.dz = pmig.lbd

    if hasattr(pmig,'Nx')==0:
        pmig.Nx = Nchannel

    if hasattr(pmig,'Nz')==0:
        pmig.Nz = Nchannel

    Npix = pmig.Nx*pmig.Nz

    if hasattr(pmig,'Xmin')==0:
        pmig.Xmin = -(pmig.Nx-1)*pmig.dx/2

    if hasattr(pmig,'Zmin')==0:
        pmig.Zmin = 0

    if hasattr(pmig,'order')==0:
        pmig.order = 1              # sinc interpolation

    #start_time = time.time()
    theta = np.array(pmig['theta'])
    theta = af.cast(af.from_ndarray(theta),af.Dtype.f32)
    invc  = 1/pmig.c

    Nframe = int(Nevent/Ntx)
    Raw = af.moddims(Raw,Nt,Nchannel,Ntx,Nframe)

    ## Linear Array definition
    xele = np.linspace(0, Nchannel-1, Nchannel,dtype="float32")*pmig.pitch - X0    # SHAPE MUST BE [1, Nchannel]
    xele = af.cast(af.from_ndarray(xele.reshape(1,1,Nchannel,order='F')),af.Dtype.f32)
    yele = af.constant(0,1,1,Nchannel)                       # SHAPE MUST BE [1, Nchannel]
    zele = af.constant(0,1,1,Nchannel)                       # SHAPE MUST BE [1, Nchannel]

    ## Cartesian Grid definition
    xp = np.linspace(0, pmig.Nx-1, pmig.Nx,dtype="float32")*pmig.dx + pmig.Xmin
    zp = np.linspace(0, pmig.Nz-1, pmig.Nz,dtype="float32")*pmig.dz + pmig.Zmin

    xp, zp = np.meshgrid(xp, zp)

    Zpix = af.cast(af.from_ndarray(zp.reshape(1,Npix,order='F')),af.Dtype.f32)
    Xpix = af.cast(af.from_ndarray(xp.reshape(1,Npix,order='F')),af.Dtype.f32)
    Ypix = af.constant(0, 1,Npix)

    r_e = af.join(0, xele, yele, third=zele)                      # SHAPE MUST BE [3, Nchannel]
    r_p = af.join(0, Xpix, Ypix, third=Zpix)

    d_rx = af.tile(r_p,1,1,Nchannel) - af.tile(r_e,1,Npix)

    RXdelay = af.reorder(af.sqrt(af.sum( d_rx**2,dim=0) )*invc,1,2,0)

    TXdelay = af.reorder(af.matmul(af.cos(af.abs(theta)),Zpix)*invc,1,0)
    TXdelay += af.reorder(af.matmul(af.sin(af.abs(theta))*(theta>0),(Xpix + X0))*invc,1,0)
    TXdelay += af.reorder(af.matmul(af.sin(af.abs(theta))*(theta<0),(X0 - Xpix))*invc,1,0)
    TXdelay = TXdelay + pmig.t0

    idex = (af.tile(RXdelay,1,1,Ntx) + af.tile(af.reorder(TXdelay,0,2,1),1,Nchannel,1))*pmig.fs

    RF = np.zeros((Npix,Nframe),dtype=float,order='F')
    for iframe in range(Nframe):
        RF[:,iframe]  = af.sum(af.sum(af.approx1(Raw[:,:,:,iframe],idex),1),2).to_ndarray()

    return RF




## BfIQFlatLinearMtx
def BfIQFlatLinearMtx(Raw, pmig):

    if len(np.shape(Raw)) > 3:
        SizRaw = np.shape(Raw)
        Raw    = np.reshape(Raw,(SizRaw[0],SizRaw[1],-1), order='F')         # reshape Nt
        SizRaw = np.shape(Raw)

    [NtIQ,Nchannel,Nevent] = np.shape(Raw)

    if 'c' not in pmig:
        pmig['c'] = 1540
        print('A Propagation velocity must be set ')
        print(pmig['c'])
        print(' m/s has been set as default')

    if 'fc' not in pmig:
        pmig['fc'] = 3e6
        print('A Central frequency must be set ')
        print(pmig['fc']/1e6)
        print(' MHz has been set as default')

    if 'lbd' not in pmig:
        pmig['lbd'] = pmig['c']/pmig['fc']

    if 'fs' not in pmig:
        pmig['fs'] = 4*pmig['fc']
        print('A Sampling frequency must be set ')
        print(pmig['fs'] /1e6)
        print('MHz has been set as default')

    if 'sub' not in pmig:
        pmig['sub'] = 2

    if 'fsIQ' not in pmig:
        pmig['fsIQ'] = pmig['fs']/pmig['sub']

    if 'pitch' not in pmig:
        pmig['pitch'] = pmig['lbd']

    if 'width' not in pmig:
        pmig['width'] = pmig['lbd']/2

    Aperture = (Nchannel-1)*pmig['pitch']
    X0 = Aperture/2

    if 'theta' not in pmig:
        pmig['theta'] = 0

    Ntx = len(pmig['theta'])

    if 't0' not in pmig:
        pmig['t0'] = 0

    # Grid definition
    if 'dx' not in pmig:
        pmig['dx'] = pmig['pitch']

    if 'dz' not in pmig:
        pmig['dz'] = pmig['lbd']

    if 'Nx' not in pmig:
        pmig['Nx'] = Nchannel

    if 'Nz' not in pmig:
        pmig['Nz'] = Nchannel

    if 'batchsize' not in pmig:
        pmig['batchsize'] = 8

    Npix            = pmig['Nx']*pmig['Nz']
    batchsize       = pmig['batchsize']
    NpixPerBatch    = int(Npix/batchsize)

    if 'Xmin' not in pmig:
        pmig['Xmin'] = 0 -(pmig['Nx']-1)*pmig['dx']/2

    if 'Zmin' not in pmig:
        pmig['Zmin'] = 0

    if 'fnum' not in pmig:
        pmig['fnum'] = 1

    if 'order' not in pmig:
        pmig['order'] = 1

    theta = af.cast(af.from_ndarray(pmig['theta']),af.Dtype.f32)
    invc  = 1/pmig['c']

    Nframe = int(Nevent/Ntx)
    if np.isreal(Raw).all():
        NtIQ = int(NtIQ/2)

    ## Linear Array definition
    xele = np.linspace(0, Nchannel-1, Nchannel,dtype="float32")*pmig['pitch'] - X0    # SHAPE MUST BE [1, Nchannel]
    xele = af.cast(af.from_ndarray(xele.reshape(1,1,Nchannel,order='F')),af.Dtype.f32)
    yele = af.constant(0,1,1,Nchannel)                       # SHAPE MUST BE [1, Nchannel]
    zele = af.constant(0,1,1,Nchannel)                       # SHAPE MUST BE [1, Nchannel]
    r_e  = af.join(0, xele, yele, third=zele)                # SHAPE MUST BE [Ndim, Nchannel]

    idxCh   = af.iota(1,Nchannel,dtype=af.Dtype.s32)
    idxTx   = af.iota(1,1,Ntx,dtype=af.Dtype.s32)

    ##
    Ninterp     = pmig['order']*2+1
    Ndata       = NtIQ * Nchannel * Ntx
    NNz         = Ninterp*NpixPerBatch*Nchannel*Ntx # Non-Zeros values

    ## Cartesian Grid definition
    xp = np.linspace(0, pmig['Nx']-1, pmig['Nx'],dtype="float32")*pmig['dx'] + pmig['Xmin']
    zp = np.linspace(0, pmig['Nz']-1, pmig['Nz'],dtype="float32")*pmig['dz'] + pmig['Zmin']
    xp, zp = np.meshgrid(xp, zp)

    Zpix = zp.reshape(1,NpixPerBatch,batchsize,order='F')
    Xpix = xp.reshape(1,NpixPerBatch,batchsize,order='F')
    Ypix = np.zeros((1,NpixPerBatch,batchsize),order='F')

    idxPix = af.iota(NpixPerBatch,dtype=af.Dtype.s32)
    idxPix = af.moddims(af.tile(idxPix,1,Nchannel*Ntx),NpixPerBatch*Nchannel*Ntx)
    idxPix = af.tile(idxPix, 1, Ninterp)

    phaseRot    = 2*np.pi*pmig['fc']/pmig['fsIQ']

    ## Memory allocation
    idxRow      = np.empty( (NNz , batchsize), dtype = int)
    idxCol      = np.empty( (NNz , batchsize), dtype = int)
    values      = np.empty( (NNz , batchsize), dtype = complex)

    for ibatch in range(batchsize):

        x_p_batch  = af.cast( af.from_ndarray( Xpix[:,:,ibatch] ), af.Dtype.f32)
        y_p_batch  = af.cast( af.from_ndarray( Ypix[:,:,ibatch] ), af.Dtype.f32)
        z_p_batch  = af.cast( af.from_ndarray( Zpix[:,:,ibatch] ), af.Dtype.f32)
        r_p_batch  = af.join(0, x_p_batch, y_p_batch, third=z_p_batch)  # SHAPE MUST BE [Ndim, NpixPerBatch]

        ## Time of flight
        demoddelay  = af.reorder(4*np.pi*pmig['fc']*r_p_batch[2,:]*invc,1,0)

        RXdelay = af.reorder(af.sqrt(af.sum( (af.tile(r_p_batch,1,1,Nchannel) - af.tile(r_e,1,NpixPerBatch) )**2,dim=0) )*invc,1,2,0)  # [NpixPerBatch,Nchannel]

        TXdelay = af.reorder(af.matmul(af.cos(af.abs(theta)),r_p_batch[2,:])*invc,1,0)
        TXdelay += af.reorder(af.matmul(af.sin(af.abs(theta))*(theta>0),(r_p_batch[1,:] + X0))*invc,1,0)
        TXdelay += af.reorder(af.matmul(af.sin(af.abs(theta))*(theta<0),(X0 - r_p_batch[1,:]))*invc,1,0)
        TXdelay = TXdelay + pmig['t0'] # [NpixPerBatch,Ntx]

        idxt = (af.tile(RXdelay,1,1,Ntx) + af.tile(af.reorder(TXdelay,0,2,1),1,Nchannel,1))*pmig['fsIQ'] # [NpixPerBatch,Nchannel,Ntx]
        deltaDelay = phaseRot*idxt - af.tile(demoddelay,1,Nchannel,Ntx)

        del demoddelay

        ## Fnumber
        ApertureF   = af.tile( af.reorder(r_p_batch[2,:]/pmig['fnum'],1,2,0) ,1,Nchannel)
        D_e         = af.reorder(af.sqrt(af.sum( (af.tile(r_p_batch[0:1,:],1,1,Nchannel) - af.tile(r_e[0:1,:],1,NpixPerBatch) )**2,dim=0) ) ,1,2,0) # [NpixPerBatch,Nchannel]
        ApodFnum    = (af.cos(np.pi * D_e/ApertureF/2)**2)  * ( D_e/ApertureF/2 < 1/2)
        ApodFnum    = ApodFnum/af.tile(1 + af.sum( ApodFnum , 1) ,1, Nchannel) /Nchannel
        ApodFnum    = af.tile(ApodFnum,1,1,Ntx) # [NpixPerBatch,Nchannel,Ntx]

        del D_e

        ## Delay To Index & Weight
        I           = (idxt<(pmig['order']+1)) + (idxt>( NtIQ - pmig['order'] - 1))
        idxt[I]     = NtIQ - pmig['order'] - 1
        idxt        = idxt + NtIQ * af.tile(idxCh,NpixPerBatch,1,Ntx) + NtIQ * Nchannel * af.tile(idxTx,NpixPerBatch,Nchannel,1)
        idxt        = af.moddims(idxt,NpixPerBatch*Nchannel*Ntx)        # [NpixPerBatch*Nchannel*Ntx]

        deltaDelay  = af.moddims(deltaDelay,NpixPerBatch*Nchannel*Ntx)  # [NpixPerBatch*Nchannel*Ntx]
        ApodFnum    = af.moddims(ApodFnum,NpixPerBatch*Nchannel*Ntx)    # [NpixPerBatch*Nchannel*Ntx]
        I           = af.moddims(I,NpixPerBatch*Nchannel*Ntx)           # [NpixPerBatch*Nchannel*Ntx]

        ## DASmtx coefficients
        idxtf       = af.tile(af.cast(idxt,dtype=af.Dtype.s32),1,Ninterp)
        idxtf       += af.tile(af.iota(1,pmig['order']*2+1,dtype=af.Dtype.s32),NpixPerBatch*Nchannel*Ntx)
        weight      = afsinc( idxtf - af.tile( idxt ,1, Ninterp ) )
        weight      = af.tile( ApodFnum * af.exp( -1j * deltaDelay ), 1, Ninterp ) * weight

        idxRow[:,ibatch] = af.moddims( idxPix , NNz ).to_ndarray()
        idxCol[:,ibatch] = af.moddims( idxtf  , NNz ).to_ndarray()
        values[:,ibatch] = af.moddims( weight , NNz ).to_ndarray()

    return idxRow, idxCol, values, pmig

def DoBf(Raw, idxRow, idxCol, values , pmig):

    Sizraw  = np.shape(Raw)
    if np.isreal(Raw).all():
        Raw = Raw[0::2,:] + 1j*Raw[1::2,:]

    Nframe  = int(np.prod(Raw.shape[2:])/len(pmig['theta']))
    Ntx     = len(pmig['theta'])
    Rawr    = af.cast(af.from_ndarray(Raw), af.Dtype.c64 )

    Rawr    = af.moddims(Rawr,np.prod(Rawr.shape[0:2])*Ntx,Nframe)

    Npix    = pmig['Nz']*pmig['Nx']
    NpixPerBatch = int(Npix/pmig['batchsize'])
    Ndata        = Rawr.shape[0]

    IQmig   = np.empty((NpixPerBatch,pmig['batchsize'],Nframe) , dtype = complex)

    for ibatch in range(pmig['batchsize']):

        valuesaf = af.cast(af.from_ndarray(values[:,ibatch]),af.Dtype.c64)
        idxRowaf = af.cast(af.from_ndarray(idxRow[:,ibatch]),af.Dtype.s32)
        idxColaf = af.cast(af.from_ndarray(idxCol[:,ibatch]),af.Dtype.s32)
        DASmtx = af.sparse.create_sparse(valuesaf,idxRowaf,idxColaf,NpixPerBatch,Ndata,storage = af.STORAGE.COO)
        DASmtx = af.sparse.convert_sparse(DASmtx,storage = af.STORAGE.CSR)
        IQmig[:,ibatch,:]  = af.matmul(DASmtx , Rawr).to_ndarray().reshape(-1,1)

    IQmig = np.reshape(IQmig,(pmig['Nz'],pmig['Nx'],Nframe),order='F')

    return IQmig



##
## BfIQFlatLinearFFT
def BfIQFlatLinearFFT(Raw, pmig):

    if len(np.shape(Raw)) > 3:
        SizRaw = np.shape(Raw)
        Raw = af.from_ndarray(np.reshape(Raw,(SizRaw[0],SizRaw[1],-1), order='F'))         # reshape Nt
        SizRaw = np.shape(Raw)
    else:
        Raw = af.from_ndarray(Raw)         # reshape Nt

    [NtIQ,Nchannel,Nevent] = np.shape(Raw)

    if 'c' not in pmig:
        pmig['c'] = 1540
        print('A Propagation velocity must be set ')
        print(pmig['c'])
        print(' m/s has been set as default')

    if 'fc' not in pmig:
        pmig['fc'] = 3e6
        print('A Central frequency must be set ')
        print(pmig['fc']/1e6)
        print(' MHz has been set as default')

    if 'lbd' not in pmig:
        pmig['lbd'] = pmig['c']/pmig['fc']

    if 'fs' not in pmig:
        pmig['fs'] = 4*pmig['fc']
        print('A Sampling frequency must be set ')
        print(pmig['fs'] /1e6)
        print('MHz has been set as default')

    if 'sub' not in pmig:
        pmig['sub'] = 2

    if 'fsIQ' not in pmig:
        pmig['fsIQ'] = pmig['fs']/pmig['sub']

    if 'pitch' not in pmig:
        pmig['pitch'] = pmig['lbd']

    if 'width' not in pmig:
        pmig['width'] = pmig['lbd']/2

    Aperture = (Nchannel-1)*pmig['pitch']
    X0 = Aperture/2

    if 'theta' not in pmig:
        pmig['theta'] = 0

    Ntx = len(pmig['theta'])

    if 't0' not in pmig:
        pmig['t0'] = 0

    # Grid definition
    if 'dx' not in pmig:
        pmig['dx'] = pmig['pitch']

    if 'dz' not in pmig:
        pmig['dz'] = pmig['lbd']

    if 'Nx' not in pmig:
        pmig['Nx'] = Nchannel

    if 'Nz' not in pmig:
        pmig['Nz'] = Nchannel

    Npix = pmig['Nx']*pmig['Nz']

    if 'Xmin' not in pmig:
        pmig['Xmin'] = 0 -(pmig['Nx']-1)*pmig['dx']/2

    if 'Zmin' not in pmig:
        pmig['Zmin'] = 0

    if 'fnum' not in pmig:
        pmig['fnum'] = 1

    #start_time = time.time()
    theta = af.cast(af.from_ndarray(pmig['theta']),af.Dtype.f32)
    invc  = 1/pmig['c']

    Nframe = int(Nevent/Ntx)
    Raw = af.moddims(Raw,NtIQ,Nchannel,Ntx,Nframe)
    if Raw.is_real():
        Raw = Raw[0:NtIQ:2,:,:]+1j*Raw[1:NtIQ:2,:,:]
        NtIQ = int(NtIQ/2)

    Nf      = int( 2**(nextpow2(NtIQ)) )
    Nch     = int( 2*Nchannel )#Nchannel

    lbd0    = pmig['c']/pmig['fc']

    Nxup    = int(Nchannel*pmig['pitch']/pmig['dx'])
    Nzup    = int(Nf*lbd0/pmig['dz']/2)

    kx      = np.linspace(-.5, .5, Nxup, dtype="float32")/pmig['dx']
    kz      = np.linspace(-.5, .5, Nzup, dtype="float32")/pmig['dz'] - 2/lbd0
    kx, kz  = np.meshgrid(kx, kz)
    kx      = af.cast( af.from_ndarray(kx) ,dtype=af.Dtype.f32)
    kz      = af.cast( af.from_ndarray(kz) ,dtype=af.Dtype.f32)

    kc      = np.linspace(-.5, .5, Nch, dtype="float32")/pmig['pitch']
    f       = np.linspace(-.5, .5, Nf, dtype="float32")*pmig['fsIQ'] - pmig['fc']

    kc, f   = np.meshgrid(kc, f)
    kc      = af.cast(af.from_ndarray(kc) ,dtype=af.Dtype.f32)
    f       = af.cast(af.from_ndarray(f) ,dtype=af.Dtype.f32)

    IQ    = np.zeros( (Nzup, Nxup, Nframe), dtype="complex64")

    for itheta in range(Ntx):

        IQfft   = af.shift(af.fft2(af.cast(Raw[:,:,itheta,:],af.Dtype.c32), dim0=int(Nf), dim1=int(Nch)), int(Nf/2), int(Nch/2))
        IQfft   = af.moddims(IQfft , Nf, Nch, Nframe)

        Norm    = af.tile( af.cos(theta[itheta]) , Nzup , Nxup) * kz
        Norm    += af.tile( af.sin(theta[itheta]) , Nzup , Nxup) * kx
        f2      = pmig['c']/2 * (kx**2+kz**2) / Norm

        IQfft   = af.approx2(IQfft,f2,kx,method=af.INTERP.BILINEAR_COSINE,off_grid=0.0,yp=kc,xp=f)
        IQ      += af.ifft2( af.shift(IQfft , int(Nzup/2), int(Nxup/2) ) ).to_ndarray()


#        IQ      += af.moddims(af.approx2(IQfft,f2,kx,method=af.INTERP.LINEAR,off_grid=0.0,yp=kc,xp=f), Nzup , Nxup, Nframe)
#
#    IQ      = af.ifft2( af.shift(IQ , int(Nzup/2), int(Nxup/2) ) ).to_ndarray()
    IQ      = IQ[0:pmig['Nz']-1,0:pmig['Nx']-1,:]
#    ## Linear Array definition
#    xele = np.linspace(0, Nchannel-1, Nchannel,dtype="float32")*pmig['pitch'] - X0    # SHAPE MUST BE [1, Nchannel]
#    xele = af.cast(af.from_ndarray(xele.reshape(1,1,Nchannel,order='F')),af.Dtype.f32)
#    yele = af.constant(0,1,1,Nchannel)                       # SHAPE MUST BE [1, Nchannel]
#    zele = af.constant(0,1,1,Nchannel)                       # SHAPE MUST BE [1, Nchannel]
#
#    ## Cartesian Grid definition
#    xp = np.linspace(0, pmig['Nx']-1, pmig['Nx'],dtype="float32")*pmig['dx'] + pmig['Xmin']
#    zp = np.linspace(0, pmig['Nz']-1, pmig['Nz'],dtype="float32")*pmig['dz'] + pmig['Zmin']
#
#    xp, zp = np.meshgrid(xp, zp)
#
#    Zpix = af.cast(af.from_ndarray(zp.reshape(1,Npix,order='F')),af.Dtype.f32)
#    Xpix = af.cast(af.from_ndarray(xp.reshape(1,Npix,order='F')),af.Dtype.f32)
#
#    Ypix = af.constant(0, 1,Npix)
#
#    demoddelay  = af.reorder(4*np.pi*pmig['fc']*Zpix*invc,1,0)
#    phaseRot    = 2*np.pi*pmig['fc']/pmig['fsIQ']
#
#    r_e = af.join(0, xele, yele, third=zele)                      # SHAPE MUST BE [3, Nchannel]
#    r_p = af.join(0, Xpix, Ypix, third=Zpix)
#
#    RXdelay = af.reorder(af.sqrt(af.sum( (af.tile(r_p,1,1,Nchannel) - af.tile(r_e,1,Npix) )**2,dim=0) )*invc,1,2,0)
#
#    TXdelay = af.reorder(af.matmul(af.cos(af.abs(theta)),Zpix)*invc,1,0)
#    TXdelay += af.reorder(af.matmul(af.sin(af.abs(theta))*(theta>0),(Xpix + X0))*invc,1,0)
#    TXdelay += af.reorder(af.matmul(af.sin(af.abs(theta))*(theta<0),(X0 - Xpix))*invc,1,0)
#    TXdelay = TXdelay + pmig['t0']
#
#    idex = (af.tile(RXdelay,1,1,Ntx) + af.tile(af.reorder(TXdelay,0,2,1),1,Nchannel,1))*pmig['fsIQ']
#    deltaDelay = phaseRot*idex - af.tile(demoddelay,1,Nchannel,Ntx)
#
#    ## Fnumber
#    dist     = af.abs(af.tile(r_p[0,:],1,1,Nchannel) - af.tile(r_e[0,:],1,Npix))
#    fnummask = af.cast(af.tile(af.reorder(2*dist*pmig['fnum']<af.tile(r_p[2,:],1,1,Nchannel),1,2,0),1,1,Ntx),af.Dtype.f32)
#    fnummask = fnummask/af.tile(af.sum(fnummask,1),1,Nchannel,1)
#
#    IQ = np.zeros((Npix,Nframe),dtype=float,order='F')+1j*np.zeros((Npix,Nframe),dtype=float,order='F')
#    for iframe in range(Nframe):
#        IQ[:,iframe]  = af.sum(af.sum(af.approx1(Raw[:,:,:,iframe],idex)*af.exp(-1j*deltaDelay)*fnummask,1),2).to_ndarray()
#
#    IQ = np.reshape(IQ,(pmig['Nz'] ,pmig['Nx'] , Nframe),order='F')
    return IQ
