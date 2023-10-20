#include <string>
#include <iostream>
#include <algorithm>
#include <cuComplex.h>

using namespace std;

__constant__ float  PI = (float)(3.14159265359);

__constant__ float  Ar[4] = {0.187, 0.288, 0.187, 0.288};
__constant__ float  Ai[4] = {0.275 , -1.954, -0.275, 1.954};
__constant__ float  Br[4] = {4.558, 8.598,  4.558, 8.598};
__constant__ float  Bi[4] = {-25.59, -7.924, 25.59, 7.924};


// Device Functions

// sinc fonction
__device__ float Sinc(float x)
{
    if(fabsf(x)>0)
    {
        if(fabsf(x)<(PI))
        {
            return sinf(x)/(x);
        }else{
            return 0;
        }
    }else{
        return 1;
    }
}

// Complex exponential
__device__ cuFloatComplex cuCexpf(cuFloatComplex x)
{
    float realx = cuCrealf(x);
    float imagx = cuCimagf(x);
    cuFloatComplex res = make_cuFloatComplex( expf(realx)*cosf(imagx), expf(realx)*sinf(imagx) );
    return res ;
}

// cuCsinc fonction
__device__ cuFloatComplex cuCSinc(cuFloatComplex x)
{
    if(cuCabsf(x)>0)
    {
    //    return sinf(x)/(x);
    //    return make_cuFloatComplex(sinf(PI*cuCabsf(x))/(PI*cuCabsf(x)), 0); // Warning
        if(cuCabsf(x)<(PI))
        {
        cuFloatComplex Den      = make_cuFloatComplex( 2*cuCimagf(x), -2*cuCrealf(x));
        cuFloatComplex Arg_p    = make_cuFloatComplex( cuCimagf(x), -cuCrealf(x));
        cuFloatComplex Arg_m    = make_cuFloatComplex( -cuCimagf(x),cuCrealf(x));

        cuFloatComplex cuSincf  = cuCdivf( cuCsubf( cuCexpf(Arg_p), cuCexpf(Arg_m)), Den);

        return cuSincf; //

        }else{
        return make_cuFloatComplex( 0, 0);
        }

    }else{
        return make_cuFloatComplex( 1, 0);
    }
}

// cuCSqrt
__device__ cuFloatComplex cuCsqrtf(cuFloatComplex x)
{
    float realx = cuCrealf(x);
    float imagx = cuCimagf(x);
    float ampx  = sqrtf(realx*realx + imagx*imagx);
    float aglx  = atan2f(imagx, realx);
    cuFloatComplex amp = make_cuFloatComplex( sqrtf(ampx), 0);
    cuFloatComplex agl = make_cuFloatComplex( 0, fdividef(aglx, 2.0f));
    cuFloatComplex res = cuCmulf( amp, cuCexpf( agl ) );
    return res;
}

// Green fonction
__device__ cuFloatComplex Green(float r, float kWaveNumber)
{
    float reGreen = fdividef( cosf(kWaveNumber*r), r);
    float imGreen = fdividef( sinf(kWaveNumber*r), r);
    return make_cuFloatComplex( reGreen, imGreen );
}

// cuCGreen fonction
__device__ cuFloatComplex cuCGreen(float r, cuFloatComplex kWaveNumber)
{
    cuFloatComplex rC = make_cuFloatComplex( r, 0 );
    cuFloatComplex I  = make_cuFloatComplex(0, 1.0f);
    cuFloatComplex Im = make_cuFloatComplex(0, -1.0f);

    return cuCdivf( cuCexpf( cuCmulf( cuCmulf(I,kWaveNumber), rC) ), rC); // exp(-kr)/r

}

// Directivity flat
__device__ cuFloatComplex DirFlat(float theta, float kWaveNumber, float width, float baffle)
{
    float cosTheta = cosf( theta );
    float sinTheta = sinf( theta );

    float DirFlat  = Sinc( fdividef( sinTheta, 2) * kWaveNumber * width);
    if(baffle==1)
    {
       DirFlat = DirFlat * cosTheta;
    }
    return make_cuFloatComplex( DirFlat, 0 );
}

// Directivity flat
__device__ cuFloatComplex cuCDirFlat(float theta, cuFloatComplex kWaveNumber, float width, float baffle)
{
//    float cosTheta = cosf( theta );
    float sinTheta = sinf( theta );

//     float DirFlat  = Sinc( fdividef( sinTheta, 2) * kWaveNumber * width);
    cuFloatComplex arg = make_cuFloatComplex( fdividef( sinTheta, 2) * width, 0);
    cuFloatComplex DirFlat  = cuCSinc( cuCmulf(arg, kWaveNumber) );

    if(baffle==1)
    {
       cuFloatComplex cosTheta = make_cuFloatComplex(cosf( theta ), 0);
       DirFlat = cuCmulf( DirFlat , cosTheta);
    }
    return DirFlat;
}

// Directivity Curved
__device__ cuFloatComplex DirCurv(float r, float y, float kWaveNumber, float height, float rfocus)
{

    float r_diff_norm = fdividef(r - rfocus, r * rfocus);
    cuFloatComplex height2 = make_cuFloatComplex(height*height,0);

    cuFloatComplex beta = make_cuFloatComplex(0, -kWaveNumber * fdividef(y, r) );
    cuFloatComplex gama = make_cuFloatComplex(0, kWaveNumber * fdividef(y*y, 2*r) );

    cuFloatComplex alpha = make_cuFloatComplex(0, .5*kWaveNumber*r_diff_norm);

    cuFloatComplex alpha_g;
    cuFloatComplex exp_g;
    cuFloatComplex sqrt_g;

    cuFloatComplex DirCurv = make_cuFloatComplex(0, 0);

    for(int ig = 0; ig<4; ig++)
    {
        cuFloatComplex Ag = make_cuFloatComplex(Ar[ig], Ai[ig]);
        cuFloatComplex Bg = make_cuFloatComplex(Br[ig], Bi[ig]);

        alpha_g = cuCaddf( cuCdivf(Bg, height2), alpha);

        exp_g = cuCexpf( cuCaddf( cuCdivf( cuCmulf(beta, beta),
                                           cuCmulf(alpha_g, make_cuFloatComplex(4.0f, 0) ) ), gama) );
        sqrt_g = cuCsqrtf( cuCdivf( make_cuFloatComplex(PI, 0), alpha_g) );

        DirCurv = cuCaddf( DirCurv, cuCmulf(Ag, cuCmulf(sqrt_g, exp_g) ) );
    }

    return DirCurv;
}

// Directivity Curved
__device__ cuFloatComplex cuCDirCurv(float r, float y, cuFloatComplex kWaveNumber, float height, float rfocus)
{

    cuFloatComplex r_diff_norm_2 = make_cuFloatComplex( fdividef(r - rfocus, r * rfocus) * 0.5f, 0);
    cuFloatComplex height2 = make_cuFloatComplex(height*height,0);

    cuFloatComplex y_r  = make_cuFloatComplex(fdividef(y, r), 0);
    cuFloatComplex y2_2r  = make_cuFloatComplex(fdividef(y*y, 2*r), 0);
    cuFloatComplex I    = make_cuFloatComplex(0, 1.0f);
    cuFloatComplex Im   = make_cuFloatComplex(0, -1.0f);

    cuFloatComplex beta = cuCmulf( cuCmulf(Im, kWaveNumber), y_r);
    cuFloatComplex gama = cuCmulf( cuCmulf(I, kWaveNumber), y2_2r);

    cuFloatComplex alpha = cuCmulf( cuCmulf(I, kWaveNumber), r_diff_norm_2);

    cuFloatComplex alpha_g;
    cuFloatComplex exp_g;
    cuFloatComplex sqrt_g;

    cuFloatComplex DirCurv = make_cuFloatComplex(0, 0);

    for(int ig = 0; ig<4; ig++)
    {
        cuFloatComplex Ag = make_cuFloatComplex(Ar[ig], Ai[ig]);
        cuFloatComplex Bg = make_cuFloatComplex(Br[ig], Bi[ig]);

        alpha_g = cuCaddf( cuCdivf(Bg, height2), alpha);

        exp_g = cuCexpf( cuCaddf( cuCdivf( cuCmulf(beta, beta),
                                           cuCmulf(alpha_g, make_cuFloatComplex(4.0f, 0) ) ), gama) );
        sqrt_g = cuCsqrtf( cuCdivf( make_cuFloatComplex(PI, 0), alpha_g) );

        DirCurv = cuCaddf( DirCurv, cuCmulf(Ag, cuCmulf(sqrt_g, exp_g) ) );
    }

    return DirCurv;
}

// TXfield (kWaveNumber, r_scat, event)
__global__ void TXfield(cuFloatComplex *TX,
        const float *r_scat,
        const float *r_ele,
        const cuFloatComplex *kWaveNumber,
        const cuFloatComplex *PULSE,
        const cuFloatComplex *IR,
        const float *Delay,
        const float *Apod,
        const float width,
        const float height,
        const float rfocus,
        const float baffle,
        const float lensCorrection,
        const float SpeedOfSound,
        const int NscatPerEvent,
        const int Nele,
        const int Nf,
        const int Ntx,
        const int Nsubele)
{
    int ik      = blockIdx.x * blockDim.x + threadIdx.x;   //kwaveIndex index
    int iscat	= blockIdx.y * blockDim.y + threadIdx.y;   //scatter index
    int itx     = blockIdx.z * blockDim.z + threadIdx.z;   //transmit index

    if(ik<Nf)
    {
    if(iscat<NscatPerEvent)
    {
    if(itx<Ntx)
    {

        cuFloatComplex TXtmp = make_cuFloatComplex(0, 0);

        // Output index
        int idv         = (ik + iscat*Nf + itx*Nf*NscatPerEvent);  //

        // Scatterer Position
        float Xscat     = r_scat[iscat + 3*NscatPerEvent*itx];
        float Yscat     = r_scat[iscat + NscatPerEvent + 3*NscatPerEvent*itx];
        float Zscat     = r_scat[iscat + 2*NscatPerEvent + 3*NscatPerEvent*itx];

        // Wavenumber
        cuFloatComplex k = kWaveNumber[ik];
        float w = cuCrealf(kWaveNumber[ik]) * SpeedOfSound;

        cuFloatComplex TXele, s_tx;
        float Xele, Yele, Zele, Azele, Elele, r, az, el;

        for(int iele = 0; iele < Nele; iele++)
        {
            for(int isub = 0; isub < Nsubele; isub++)
            {
                // Element Position
                Xele      = r_ele[iele + isub*5*Nele];
                Yele      = r_ele[iele + Nele + isub*5*Nele];
                Zele      = r_ele[iele + 2*Nele + isub*5*Nele];
                Azele     = r_ele[iele + 3*Nele + isub*5*Nele];
                Elele     = r_ele[iele + 4*Nele + isub*5*Nele];

                // Distance To Element
                r         = norm3df(Xscat - Xele, Yscat - Yele, Zscat - Zele);
                az        = asinf( fdividef(Xscat - Xele, r)) - Azele;
                el        = asinf( fdividef(Yscat - Yele, r)) - Elele;

                // Transmit Wavefield
                TXele     = cuCGreen(r, k);
                TXele     = cuCmulf(TXele, cuCDirFlat(az, k, width, baffle) );
                if(rfocus>10.0f)
                {
                TXele	= cuCmulf(TXele, cuCDirFlat(el, k, width, baffle) );
                }else{
                TXele	= cuCmulf(TXele, cuCDirCurv(r, r*sinf(el), k, height, rfocus) );
                }
                TXele     = cuCmulf(TXele, cuCexpf( make_cuFloatComplex(0, w*lensCorrection) ) );
                TXele     = cuCmulf(TXele, cuCmulf( cuConjf( PULSE[ik] ), cuConjf( IR[ik] ))  );

                // Transmit Wave Front
                s_tx      = cuCmulf( make_cuFloatComplex( Apod[iele + itx*Nele], 0),
                                     cuCexpf( make_cuFloatComplex(0, w*Delay[iele + itx*Nele]) ) );
                TXele     = cuCmulf( TXele, s_tx);

                // Sum the contribution of every element
                TXtmp     = cuCaddf( TXtmp, TXele);
            }
        }
        TX[idv] = TXtmp;
    }
    }
    }
}

// RXfield (kWaveNumber, r_ele, iscat)
__global__ void RXfield(cuFloatComplex *RX,
        const float *r_scat,
        const float *r_ele,
        const cuFloatComplex *kWaveNumber,
        const cuFloatComplex *IR,
        const float width,
        const float height,
        const float rfocus,
        const float baffle,
        const float lensCorrection,
        const float SpeedOfSound,
        const int Nscat,
        const int Nele,
        const int Nf,
        const int Nsubele)
{
    int ik      = blockIdx.x * blockDim.x + threadIdx.x;   //frequency index
    int iele    = blockIdx.y * blockDim.y + threadIdx.y;   //element index
    int iscat   = blockIdx.z * blockDim.z + threadIdx.z;   //scatter index

    if(iscat<Nscat)
    {
    if(iele<Nele)
    {
    if(ik<Nf)
    {
        // Allocate output
        cuFloatComplex RXout = make_cuFloatComplex(0, 0);;
        // Output index
        int idv         = (ik + iele*Nf + iscat*Nf*Nele);  //

        // Scatterer Position
        float Xscat     = r_scat[iscat];
        float Yscat     = r_scat[iscat + Nscat];
        float Zscat     = r_scat[iscat + 2*Nscat];

        // Wavenumber
        cuFloatComplex k = kWaveNumber[ik];
        float w = cuCrealf(kWaveNumber[ik]) * SpeedOfSound;

        float Xele, Yele, Zele, Azele, Elele, r, az, el;

        cuFloatComplex RXtmp;
        for(int isub = 0; isub < Nsubele; isub++)
        {
            // Element Position
            Xele      = r_ele[iele + isub*5*Nele];
            Yele      = r_ele[iele + Nele + isub*5*Nele];
            Zele      = r_ele[iele + 2*Nele + isub*5*Nele];
            Azele     = r_ele[iele + 3*Nele + isub*5*Nele];
            Elele     = r_ele[iele + 4*Nele + isub*5*Nele];

            // Distance To Element
            r         = norm3df(Xscat - Xele, Yscat - Yele, Zscat - Zele);
            az        = asinf( fdividef(Xscat - Xele, r)) - Azele;
            el        = asinf( fdividef(Yscat - Yele, r)) - Elele;

            // Transmit Wavefield
            RXtmp     = cuCGreen(r, k);
            RXtmp     = cuCmulf(RXtmp, cuCDirFlat(az, k, width, baffle) );
            if(rfocus>10.0f)
            {
            RXtmp	= cuCmulf(RXtmp, cuCDirFlat(el, k, width, baffle) );
            }else{
            RXtmp	= cuCmulf(RXtmp, cuCDirCurv(r, r*sinf(el), k, height, rfocus) );
            }
            RXtmp     = cuCmulf(RXtmp, cuCexpf( make_cuFloatComplex(0, w*lensCorrection) ) );
            RXtmp     = cuCmulf(RXtmp, IR[ik] );
            
            // Sum the contribution of every element
            RXout     = cuCaddf( RXout, RXtmp);

        }
        // Return output
        RX[idv]	  = RXout;
    }
    }
    }
}