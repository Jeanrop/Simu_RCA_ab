/* 
I/Q beamformer for flat angles, can be use for 1D/2D/3D imaging.
     
bfVirtualSourceGeneric(
          float *dataBf,
          float *mask,
    const float *data,
    const float *gridBfX,	
    const float *gridBfY,	
    const float *gridBfZ,	
    const float *posEleX,
    const float *posEleY,
    const float *posEleZ,
    const float *virtualSourceX,
    const float *virtualSourceY,
	const float *virtualSourceZ,  
    const float  apertureX,
    const float  apertureY,         
    const float  apertureZ,  
    const float  samplingFrequency,
    const float  demodulationFrequency,
    const float  speedOfSound,
    const float  startTime,
    const float  fnumber,
    const int    nPix,
    const int    nChannel,
    const int    nTx,
    const int    nFrame,
    const int    nSample,
	const bool   flagInterpolationLanczos,
	const bool   flagMaskTransmit
)


Outputs:
    [0] dataBf: beamformed data [pixels (I/Q interleaved) x transmits x frames]
	[1] mask: mask [pixels x transmits]

Parameters:
    [0-1] GPU objects for outputs
    [2]   data: raw data [samples (I/Q interleaved) x channels x transmits x frames]
    [3]   gridBfX: x-coordinates to beamform (array, m, all coordinates must match)
    [4]   gridBfY: y-coordinates to beamform (array, m, all coordinates must match)
    [5]   gridBfZ: z-coordinates to beamform (array, m, all coordinates must match)
    [6]   posEleX: x-coordinates of the elements (array, m, all coordinates must match)
    [7]   posEleY: y-coordinates of the elements (array, m, all coordinates must match)
    [8]   posEleZ: z-coordinates of the elements (array, m, all coordinates must match)
    [9]   virtualSourceX: position of the virtual sources along x-axis (array, m)
    [10]  virtualSourceY: position of the virtual sources along y-axis (array, m)
    [11]  virtualSourceZ: position of the virtual sources along z-axis (array, m)
    [13]  minDistVirtualSource: minimal distance from the virtual source(s) to the array (array, m)
    [14]  apertureX: aperture along x-axis (m)
    [15]  apertureY: aperture along y-axis (m)
	[16]  apertureZ: aperture along z-axis (m)
    [17]  samplingFrequency: sampling frequency of the I/Q data (Hz)
    [18]  demodulationFrequency: demodulation frequency (Hz)
    [19]  speedOfSound: speed-of-sound (m/s)
    [20]  startTime: start time (s)
    [21]  nPix: number of pixels to beamform (must match the beamforming grid)
    [22]  nChannel: number of channels (must match the raw data)
    [23]  nTx: number of transmits (must match the raw data)
    [24]  nFrame: number of frames (must match the raw data)
    [25]  nSample: number of samples (must match the raw data)
	[26]  flagInterpolationLanczos: flag for interpolation method (true = Lanczos, false = quadratic)
	[27]  flagMaskTransmit: flag for the mask in transmit (boolean, true = only beamform insonied points)

Vincent Perrot, polymtl, Montreal, Canada.
Feb 2, 2022.
*/      
        
     

// INTERNAL DEFINITIONS AND FUNCTIONS      
__constant__ float PI = 3.14159265358979323846; // pi


__device__ float sinc(float x) // sinc function  
{
    return fminf(fdividef(sinpi(x), PI*x), 1);
}


 __device__ float angleAperture(float ax, float ay,
								float bx, float by, 
							    float cx, float cy) // Angle between two vectors (AB, AC)
{
	// CUDA intrinsic functions are faster than standard ones here, double precision is needed for intermediate terms (to avoir precision errors)
	double abx = bx-ax;
	double aby = by-ay;
	double acx = cx-ax;
	double acy = cy-ay;
	
	return acosf(((abx*acx)+(aby*acy)) * 
			       rsqrt((abx*abx+aby*aby) * 
                         (acx*acx+acy*acy)));
}



// MAIN FUNCTION      
__global__ void bfVirtualSourceGeneric_RCA_Nosum(
              float *dataBf,
              float *mask,
        const float *data,
        const float *gridBfX,	
        const float *gridBfY,	
        const float *gridBfZ,	
        const float *posEleX,
        const float *posEleY,
        const float *posEleZ,
        const float *virtualSourceX,
        const float *virtualSourceY,
        const float *virtualSourceZ,
        const float *minDistVirtualSource,
        const bool   flagReceiveX,
		const float  apertureX,
		const float  apertureY,		
        const float  apertureZ,	
        const float  samplingFrequency,
        const float  demodulationFrequency,
        const float  speedOfSound,
        const float  startTime,
        const float  fnumber,
        const int 	 nPix,
        const int 	 nChannel,
        const int 	 nTx,
        const int 	 nFrame,
        const int 	 nSample,
		const bool 	 flagInterpolationLanczos,
		const bool 	 flagMaskTransmit
		) // Main function definition


{
    // Indices
    int idxPix   = blockIdx.x*blockDim.x + threadIdx.x; // Pixel index     
    if (idxPix<nPix) // If pixel index is valid
    {
    int idxTx    = blockIdx.y*blockDim.y + threadIdx.y; // Transmit index
    if (idxTx<nTx) // If transmit index is valid 
    {
    int idxFrame = blockIdx.z*blockDim.z + threadIdx.z; // Frame index 
    if (idxFrame<nFrame) // If frame index is valid
    {
        
        
        // Definitions
		// CUDA intrinsic functions can be slow here, do not use them
        float dataReal, dataImag; // Terms for beamforming
                
        float pixX = gridBfX[idxPix]; // X position of the pixel
        float pixY = gridBfY[idxPix]; // Y position of the pixel
        float pixZ = gridBfZ[idxPix]; // Z position of the pixel
               
        float virtualSourceXTx       = virtualSourceX[idxTx]; // Position of the virtual sources along x-axis
        float virtualSourceYTx       = virtualSourceY[idxTx]; // Position of the virtual sources along y-axis
        float virtualSourceZTx       = virtualSourceZ[idxTx]; // Position of the virtual sources along z-axis
        float minDistVirtualSourceTx = minDistVirtualSource[idxTx]; // Minimal distance from the virtual source to the array
                
        float eleX, eleY, eleZ; // Position of the elements  
                
        float invSpeedOfSound = 1 / speedOfSound; // Inverse of the speed-of-sound
        
        float delayTx, delayRx; // Delays in transmit and receive  
        float delayDelta; // Delay for phase rotation  
           
        float apod; // Apodization                  
        float aperture; // Aperture
                
        float halfApertureX = 0.5 * apertureX; // Half aperture along x-axis      
        float halfApertureY = 0.5 * apertureY; // Half aperture along y-axis 
        float apertureF     = pixZ / fnumber; // Normalized f-number aperture
        
        float angConeX, angConeY; // Angles (apertures) defining a cone where the apex if the virtual source
                
        bool maskTransmit; // Total mask in transmit
                            
        int idxData; // Index of the sample of interest 
        
        float idxDelay; // Delay in samples
		int   idxDelayFloor; // Round or floor (depending upon the interpolation type) of the delay in samples
        float idxDelayDelta; // Distance between samples
        
		float q0, q1, q2, q3, q4, q5; // Terms for interpolation
		int   ptsLowBound, ptsHighBound; // Number of points needed (bound limits) around the delay for interpolation
	
	
        // Compute mask in transmit for the pixel of interest (boolean)
		if (flagMaskTransmit) // True = beamform only if the point was insonified
		{
			angConeX = angleAperture(virtualSourceXTx, virtualSourceZTx, 
								    -halfApertureX, -apertureZ, 
								     halfApertureX, -apertureZ);
			angConeY = angleAperture(virtualSourceYTx, virtualSourceZTx, 
								    -halfApertureY, -apertureZ, 
								     halfApertureY, -apertureZ);
			
			maskTransmit = angleAperture(virtualSourceXTx, virtualSourceZTx, 
									    -halfApertureX, -apertureZ, 
									     pixX, pixZ) <= angConeX
						   &&
						   angleAperture(virtualSourceXTx, virtualSourceZTx, 
										 halfApertureX, -apertureZ, 
										 pixX, pixZ) <= angConeX
						   &&
						   angleAperture(virtualSourceYTx, virtualSourceZTx, 
								        -halfApertureY, -apertureZ, 
								         pixY, pixZ) <= angConeY
					       &&
						   angleAperture(virtualSourceYTx, virtualSourceZTx, 
								         halfApertureY, -apertureZ, 
								         pixY, pixZ) <= angConeY; // True only if the pixel was insonified
		}			 
		else // False = beamform the pixel regardless of the insonification pattern
		{
			maskTransmit = true; // Mask always true
		}
		

        // Continue if the mask is valid
		// CUDA intrinsic functions for trigonometric, abs, and sign functions are faster than standard ones 
        if (maskTransmit)
		{
			
			
			// Indices
			int idxPixTx = idxPix + idxTx*nPix; // Tx pixel index 
            int idxPixBf = 2*(idxPix + idxTx*nPix + idxFrame*nPix*nTx); // Output pixel index
            int nPixBf = 2*(nFrame*nPix*nTx);
			
			
			// Set bound limits depending upon the interpolation method
            if (flagInterpolationLanczos) // Lanczos 5-lobes
            {
                ptsLowBound  = 2; // Lower bound
                ptsHighBound = 3; // Higher bound
            }	
            else // Quadratic
            {
                ptsLowBound  = 1; // Lower bound
                ptsHighBound = 1; // Higher bound
            }	
				
				
			// Initialize output (DO NOT REMOVE, usefull if CUDA is in a loop) 
            if (idxFrame==0) // Only needed for one frame 
            {
                mask[idxPixTx] = 0; // Total mask
            }
            //dataBf[idxPixBf]     = 0; // Real part of the beamformed data
            //dataBf[idxPixBf + 1] = 0; // Imaginary part of the beamformed data	
				
				
			// Delay in transmit -  Pas de loie d'aberration en transmission ici 
            if (flagReceiveX) // receive on X channels transmit on Y channels -> virtualSourceY
            {
                delayTx = (norm3df((pixX - virtualSourceXTx)*0,  // emission on Y channels 
                                pixY - virtualSourceYTx,
                                pixZ - virtualSourceZTx) - 
                        minDistVirtualSourceTx) * invSpeedOfSound;
            }
            else // receive on Y channels transmit on X channels -> virtualSourceY
            {
                delayTx = (norm3df(pixX - virtualSourceXTx,  // emission on X channels 
                                (pixY - virtualSourceYTx)*0,
                                pixZ - virtualSourceZTx) - 
                        minDistVirtualSourceTx) * invSpeedOfSound;
            }	
						
					
			// Delay & Sum over the receive channels   
			// In a loop most CUDA intrinsic functions are faster than standard ones 
			for (int idxChannel = 0; idxChannel < nChannel; idxChannel++)
			{
            int idxPixBfCh = idxPixBf + idxChannel*nPixBf;

            dataBf[idxPixBfCh]     = 0; // Real part of the beamformed data
            dataBf[idxPixBfCh + 1] = 0; // Imaginary part of the beamformed data	
				
				// Select element and aperture
				eleX = posEleX[idxChannel]; // X position of the element of interest
				eleY = posEleY[idxChannel]; // Y position of the element of interest
				eleZ = posEleZ[idxChannel]; // Z position of the element of interest    
						
				// Aperture
				if (flagReceiveX) // receive on X channels
                {
                    aperture = abs(pixX - eleX);
                }
                else  // receive on Y channels
                {
                    aperture = abs(pixY - eleY);
                }
						
				if (aperture<apertureF) // If the aperture is valid
				{
					
					
					// Apodization
					apod = 0.5*(1+cospif(fdividef(aperture, apertureF))); // Hann window

					if (idxFrame==0) // Only needed for one frame 
						{
							mask[idxPixTx] += apod; // Total mask
						}  


					// Delays 
					delayRx  = hypotf(aperture, pixZ - eleZ) * 
									  invSpeedOfSound; // In receive
					idxDelay = (delayTx + delayRx + startTime) *
								samplingFrequency; // Total delays in samples


					// Definitions	
					idxData  = 2*idxChannel*nSample +
							   2*idxTx*nSample*nChannel +
							   2*idxFrame*nSample*nChannel*nTx; // Index of the sample of interest 
					
					idxDelayFloor = floorf(idxDelay); // Floor of the delay for all the other interpolation methods
					
                            
					// Continue only if the sample is valid (in the receive data)
					if (idxDelayFloor>=ptsLowBound && idxDelayFloor<nSample-1-ptsHighBound)
					{
                        idxDelayDelta = idxDelay-idxDelayFloor; // Distance between samples
						
                                
						// Interpolation 
						if (flagInterpolationLanczos)
						{
							q0 = sinc(idxDelayDelta+2)*sinc((idxDelayDelta+2)/2); // First term for interpolation
							q1 = sinc(idxDelayDelta+1)*sinc((idxDelayDelta+1)/2); // Second term for interpolation
							q2 = sinc(idxDelayDelta)  *sinc((idxDelayDelta)/2); // Third term for interpolation
							q3 = sinc(idxDelayDelta-1)*sinc((idxDelayDelta-1)/2); // Fourth term for interpolation
							q4 = sinc(idxDelayDelta-2)*sinc((idxDelayDelta-2)/2); // Fifth term for interpolation
							q5 = sinc(idxDelayDelta-3)*sinc((idxDelayDelta-3)/2); // Sixth term for interpolation

							dataReal = q0 * data[2*(idxDelayFloor-2) + idxData]+
									   q1 * data[2*(idxDelayFloor-1) + idxData]+
									   q2 * data[2*(idxDelayFloor)   + idxData]+
									   q3 * data[2*(idxDelayFloor+1) + idxData]+
									   q4 * data[2*(idxDelayFloor+2) + idxData]+
									   q5 * data[2*(idxDelayFloor+3) + idxData]; // Real term before phase rotation	

							dataImag = q0 * data[2*(idxDelayFloor-2) + idxData + 1]+
									   q1 * data[2*(idxDelayFloor-1) + idxData + 1]+
									   q2 * data[2*(idxDelayFloor)   + idxData + 1]+
									   q3 * data[2*(idxDelayFloor+1) + idxData + 1]+
									   q4 * data[2*(idxDelayFloor+2) + idxData + 1]+
									   q5 * data[2*(idxDelayFloor+3) + idxData + 1]; // Imag term before phase rotation
						}
						
						else // Quadratic
						{
                            q0 =  idxDelayDelta     * (idxDelayDelta-1)/2; // First term for interpolation
                            q1 = -(idxDelayDelta-1) * (idxDelayDelta+1); // Second term for interpolation
                            q2 =  idxDelayDelta     * (idxDelayDelta+1)/2; // Third term for interpolation

                            dataReal = q0 * data[2*(idxDelayFloor-1) + idxData]+
                                       q1 * data[2*(idxDelayFloor)   + idxData]+
                                       q2 * data[2*(idxDelayFloor+1) + idxData]; // Real term before phase rotation

                            dataImag = q0 * data[2*(idxDelayFloor-1) + idxData + 1]+
                                       q1 * data[2*(idxDelayFloor)   + idxData + 1]+
                                       q2 * data[2*(idxDelayFloor+1) + idxData + 1]; // Imag term before phase rotation
						}
						
						

						// Apply phase rotation and apodization
                        delayDelta = -2.0f * PI * idxDelay *
                                     fdividef(demodulationFrequency, samplingFrequency) + 
                                     4.0f * PI * demodulationFrequency * pixZ * invSpeedOfSound;

						dataBf[idxPixBfCh]     = apod *
									           (dataReal * cosf(delayDelta) +
									            dataImag * sinf(delayDelta)); // Sum real part after phase rotation
						dataBf[idxPixBfCh + 1] = apod *
									          (-dataReal * sinf(delayDelta) + 
									            dataImag * cosf(delayDelta)); // Sum imaginary part after phase rotation	
					}
				}
			}  
			
		}

    }
    }
    }

}