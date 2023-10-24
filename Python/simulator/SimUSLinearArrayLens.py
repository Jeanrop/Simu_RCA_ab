# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 18:18:12 2020

@author: jopo86
"""

import arrayfire as af
import numpy as np

def nextpow2(b):
    a = 0
    while 2**a <= b: a += 1
    return a

def afsinc(input_array):
    output_array = af.sin(np.pi*input_array)/(np.pi*input_array)
    output_array[input_array==0]=1
    return output_array

def SimUSLinearArrayLens(r_scat, bsc, Delays, Apod,  psim):

    if 'c' not in psim:
        psim['c'] = 1540
        print('A Propagation velocity must be set ')
        print(psim['c'])
        print(' m/s has been set as default')

    if 'fc' not in psim:
        psim['fc'] = 3e6
        print('A Central frequency must be set ')
        print(psim['fc']/1e6)
        print(' MHz has been set as default')

    if 'lbd' not in psim:
        psim['lbd'] = psim['c']/psim['fc']

    if 'fs' not in psim:
        psim['fs'] = 4*psim['fc']
        print('A Sampling frequency must be set ')
        print(psim['fs'] /1e6)
        print('MHz has been set as default')

    if 'pitch' not in psim:
        psim['pitch'] = psim['lbd']

    if 'width' not in psim:
        psim['width'] = psim['lbd']/2

    if 'height' not in psim:
        psim['height'] = 5*psim['lbd']

    if 'elevfocus' not in psim:
        psim['elevfocus'] = 64*psim['lbd']

    if 'nele' not in psim:
        psim['nele'] = 128

    if 'eleBW' not in psim:
        psim['eleBW'] = 200
        print('The element bandwith must be set ')
        print(psim['eleBW'])
        print('% has been set as default')

    if 'baffle' not in psim:
        psim['baffle'] = 0
        # Details Baffle

    if 'Npulse' not in psim:
        psim['Npulse'] = 2

    if 'pulse' not in psim:
        tpulse = np.mgrid[0:psim['Npulse']/psim['fc']:1/psim['fs']]
        pulse  = np.sin(2*np.pi*psim['fc']*tpulse)
        pulse  = np.float32(pulse * np.hanning(pulse.size).T)

    if 'DepthMax' not in psim:
        psim['DepthMax'] = np.max(r_scat)

    if 'FastSim' not in psim:
        psim['FastSim'] = 0 # if = 1 simulate directivity from the central frequency

    ## PROBE DEFINITION #######################################################
    # Probe element positions
    widthy      = psim['lbd']/2
    neleHeight  = 2*int(psim['height']/widthy/2)+1

    neleTotal   = psim['nele']*neleHeight
    aperture    = (psim['nele'] - 1)*psim['pitch']

    xele    = np.linspace(0, aperture, psim['nele'], dtype="float32") - aperture/2
    yele    = np.linspace(0, psim['height'], neleHeight, dtype="float32") - psim['height']/2
    yele,xele   = np.meshgrid(yele, xele)

    yele    = af.reorder(af.cast(af.from_ndarray(yele.reshape(neleTotal,order='F')),af.Dtype.f32),1,0)
    xele    = af.reorder(af.cast(af.from_ndarray(xele.reshape(neleTotal,order='F')),af.Dtype.f32),1,0)
    zele    = psim['elevfocus'] - af.sqrt(psim['elevfocus']**2 - yele**2)
    r_e     = af.join(0, xele, yele, third=zele)               # SHAPE MUST BE [3, sim_prm.nele]

    [ntx,nele] = np.shape(Delays)

    ## PHANTOM DEFINITION #####################################################
    if len(np.shape(r_scat)) > 2:
        [ndim,nscatPerEvent,nevent] = np.shape(r_scat)
    else:
        [ndim,nscatPerEvent] = np.shape(r_scat)
        nevent = 1

    r_scat      = af.cast(af.from_ndarray(r_scat),af.Dtype.f32)
    bsc         = af.cast(af.from_ndarray(bsc),af.Dtype.f32)

    nframe      = int(nevent/ntx)

    r_scat      = af.moddims(r_scat[:,:,0:nframe*ntx],ndim,nscatPerEvent,ntx,nframe)
    bsc         = af.moddims(bsc[:,0:nframe*ntx],1,nscatPerEvent,ntx,nframe)


    PRFmax      = psim['c']/2/np.sqrt(aperture**2 + psim['DepthMax']**2)

    Nt = 8*int(np.ceil(psim['fs']/PRFmax/8))   # Nb of fast time indices

    fi = af.from_ndarray(np.linspace(0, psim['fs']/2, int(Nt/2), dtype="float32"))
    Nf = fi.shape[0]
    wi = 2*np.pi*fi

    ki = wi/psim['c']

    k0 = 2*np.pi*psim['fc']/psim['c']

    # Pulse in frequency domain
    PULSE   = np.fft.fft(pulse, Nt)
    PULSE   = PULSE[0:int(Nt/2)]
    PULSE   = af.cast(af.from_ndarray(PULSE),af.Dtype.c32)

    # Element in frequency domain
    tele    = np.mgrid[0:(200/psim['eleBW']/psim['fc']):1/psim['fs']]
    hele    = np.sin(2*np.pi*psim['fc']*tele)
    hele    = hele * np.hanning(hele.size).T
    HELE    = np.fft.fft(hele, Nt)
    HELE    = HELE[0:int(Nt/2)]
    HELE    = af.cast(af.from_ndarray(HELE),af.Dtype.c32)


    RF     = af.constant(0,Nt, nele, ntx, nframe, af.Dtype.f32)

    for index in range(ntx):
        de = af.from_ndarray(Delays[index])
        ap = af.from_ndarray(Apod[index])

        ap = af.tile(af.reorder(ap,1,0),1,neleHeight)
        de = af.tile(af.reorder(de,1,0),1,neleHeight)

        s_tx = af.tile(ap, Nf) *af.exp(1j * af.matmul(wi, de))

        for iframe in range(nframe):

            r_scat_tmp  = r_scat[:,:,index,iframe]
            bsc_tmp     = bsc[:,:,index,iframe]

            #-- Tx distance [ndim neleTotal nscatPerEvent]
            d_tx    = af.reorder(af.tile(af.reorder(r_scat_tmp,0,1,2), 1,1,neleTotal),0,2,1) - af.tile(r_e, 1, 1, nscatPerEvent)
            r_tx    = af.sqrt(af.sum(d_tx**2, dim=0))

            sin_tx_x    = d_tx[0]/r_tx[0]
            sin_tx_y    = d_tx[1]/r_tx[0]
            cos_tx      = d_tx[2]/r_tx[0]

            #-- Directivity
            if psim['FastSim']:
                #-- Directivity [k0 ele scat]
                D_tx = afsinc( psim['width']/2/np.pi * k0 * sin_tx_x  )
                D_tx = D_tx * afsinc( widthy/2/np.pi * k0 * sin_tx_y  )
                if psim['baffle'] == 1:
                    D_tx = cos_tx * D_tx
                D_tx = af.tile(D_tx, Nf)

            elif ~psim['FastSim']:
                #-- Directivity [ki ele scat]
                D_tx = afsinc( psim['width']/2/np.pi * af.matmul(ki,sin_tx_x) )
                D_tx = D_tx * afsinc( widthy/2/np.pi * af.matmul(ki,sin_tx_y)  )
                if psim['baffle'] == 1:
                    D_tx = af.tile(cos_tx, Nf) * D_tx


            #-- Green Function [ki ele scat]
            G_tx = af.exp(1j * af.matmul(ki, r_tx) )/af.tile(r_tx,Nf)

            #-- Transmit
            TX = af.sum(G_tx * D_tx * af.tile(s_tx/neleTotal, 1, 1, nscatPerEvent), dim=1)

            #-- Receive [ki ele scat]
            RX = af.tile(af.reorder(bsc_tmp,2,0,1),Nf,neleTotal) * G_tx * D_tx

            #-- Sum over scatter [ki ele scat]
            TXRX = af.sum(RX * af.tile(TX, 1,neleTotal)/neleTotal, 2)
            TXRX = af.tile(PULSE*HELE*af.conjg(HELE), 1,int(neleTotal)) * TXRX

            TXRX = af.moddims( TXRX ,  Nf, psim['nele'] , neleHeight)

            TXRX = af.flip(af.ifft(af.sum(TXRX, dim=2), Nt) )

            RF[:,:,index,iframe] =  af.real(TXRX + af.conjg(TXRX))

    RF = RF.to_ndarray()
    return RF                           ### CAREFUL, THE REORDER IS TEMPORARY
