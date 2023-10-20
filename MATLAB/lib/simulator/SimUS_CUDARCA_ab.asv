function [Raw, Raw_ab ] = SimUS_CUDARCA_ab(Media, TransTX, TransRX, TXSI, RXSI, AbLaw, Options)

% SIMUSGENERIC MultiDimensionnal Ultrasound Simulator
% Simulate the recorded signals from a receive transducer array for
% a given transmission with a emitting transducer
% Matlab/CUDA codes based on the linear acoustic model developped in :
% [1] Garcia, D. (2022). SIMUS: An open-source simulator for medical
%     ultrasound imaging. Part I: Theory & examples.
%     Computer Methods and Programs in Biomedicine, 106726.
% DO NOT CHANGE the parameters (grid/size) for the CUDA kernel.
% If no input is speficied, returns the lines to compile the PTX.
% The PTX must be compiled once for the GPU configuration.
%      RF          = SimUSGeneric(TransTX, TransRX, Transmit, Receive, Media)
%      RF          = SimUSGeneric(TransTX, TransRX, Transmit, Receive, Media, Options)
%
%     OUTPUTS:
%         RF: real simulated received radiofrequency data [fast-time x receiveChannel x transmit-event ]
%
%     INPUTS:
%         Media: structure describing the propagation medium (struct)
%         TransTX: structure describing the emitting Transducer in SI units (struct)
%         TransRX: structure describing the receiving Transducer in SI units (struct)
%         TXSI: structure describing the transmit sequence (struct)
%         RXSI: structure describing the receive sequence (struct)
%         Options: structure for some extra optional parameters (struct)
%
%     Media: TODO
%         speedOfSound: sound speed velocity (m/s)
%         attenuation: attenuation of the medium (dB/cm/MHz)
%         scattererPos: scatter position and amplitude [x-pos y-pos z-pos bsc] (matrix)
%
%     TransTX/TransRX:
%         frequency: Center Frequency of the transducer aperture (Hz)
%         bandwidth: Lower and Upper -6 dB round trip transducer bandwidth cutoff points (Hz)
%         impulseResponse: One way Impulse response of the transducer (sampled at 250MHz) (array)
%         lensCorrection: one way propagation time through the lens (s) (scalar)
%         ElementPos: Transducer element position & orientation (azimuth &
%         elevation) [x y z az el] (m & rad) (matrix)
%         ElementHeight: height of the elements (m) (scalar)
%         ElementWidth: witdth of the element (m) (scalar)
%         ElementElevationFocus: elevation focus depth from lens (m) (scalar)
%         baffle: soft = 1 hard = 0 (scalar)
%
%     TXSI:
%         pulseShape: Shape of the transmitted pulse (sampled at 250MHz) (array)
%         Delay: Delay law applied to the element in transmit (array)
%         Apod: Apodisation law applied to the element in transmit (array)
%
%     RXSI: TODO
%         samplingFrequency: sampling frequency of the receive channel (Hz)
%         demodFrequency: Demodulation frequency (Hz)
%         startTime: time after first emission (s)
%         numelSample: number of sample per receive channel
%
%     OPTIONS: TODO
%         decreaseFactorGPU: decrease factor for the GPU memory (integer, default = 1)
%
%     NOTES: TODO
%         If a 2D/1D array is used, be sure to use singletons (permute) the
%         array/data to keep a 3D form the inputs.
%         You need to specify the transmit angle(s) OR the position of the
%         virtual source(s) but NOT a mix of both. Transmit angle(s) will
%         be converted to virtual source(s) (far behind the array).
%         The origin of the axes MUST be at the center of the array.
%         The nearest interpolation is slightly faster than quadratic or
%         linear methods. Linear and quadratic interpolations have very
%         similar computing times on most modern GPUs. Then, there is not many
%         advantages to use the nearest or linear interpolation instead of
%         the quadratic one. Lanzcos 5-lobes can be twice as long as the
%         other methods. For most applications the qudratic interpolation
%         if the best method, Lanzcos 5-lobes can be use for a slightly
%         more precised beamforming but the gain is very very low.
%         DO NOT CHANGE the parameters (grid/size) for the CUDA kernel. If
%         no input is speficied, the function returns the lines to compile
%         the PTX. The PTX must be compiled once for your GPU
%         configuration. By passing all the grids, frames, and angles to
%         the GPU, the calculation time has been reduced by half compared
%         to the previous version. The optimization of the handling of the
%         data by CUDA (grids, size, chunks) further reduce the calculation
%         time of about 20 percent for large dataset.
%
% Jonathan Poree, polymtl, Montreal, Canada.
% March 07, 2022.

%% INITIALIZATION
% %--- Number of inputs
% narginchk(5, 6); % Five or six inputs
% nargoutchk(1, 3); % Three or four inputs
%
%
% if nargin == 4 % Without options (4 args) TransTX, TransRX, Transmit, Receive, Media
%     TransTX  = varargin{1};
%     TransRX  = varargin{2};
%     Transmit    = varargin{3};
%     Receive     = varargin{4};
%     Media       = varargin{5};
%     Options     = [];
% else % With options (5 args)
%     TransTX  = varargin{1};
%     TransRX  = varargin{2};
%     Transmit    = varargin{3};
%     Receive     = varargin{4};
%     Media       = varargin{5};
%     Options     = varargin{6};
% end

%% INITIALIZATION
%--- Return line to compile PTX if no input
if nargin == 0
    fprintf(...
        ['\nFOR WINDOWS ONLY!\n', ...
        'The PTX must be compiled in the same folder that the CUDA ',...
        'file before first use, the command (Visual Studio required) ', ...
        'should be of the form (the exact path is system dependent):\n\n', ...
        'system(''nvcc -ptx SimusKernel.cu -use_fast_math -ccbin ', ...
        '"C:/Program Files (x86)/Microsoft Visual Studio/', ...
        '[YEAR]/[Professional OR Enterprise OR Community]/', ...
        'VC/Tools/MSVC/[VERSION]/bin/Hostx64/x64"'')\n\n']);
    return
end

if ~isfield(Options, 'decreaseFactorGPU') % GPU decrease factor
    Options.decreaseFactorGPU = 1; % Default (no decrease)
elseif (floor(Options.decreaseFactorGPU) ~= Options.decreaseFactorGPU) || ...
        Options.decreaseFactorGPU < 0 % If invalid value
    errStr.message = ['GPU decrease factor ', ...
        '(''Options.decreaseFactorGPU'') must be a positive integer!'];
    errStr.identifier = [mfilename, ':decreaseFactorGPUInvalid'];
    error(errStr);
end

%% BEGIN
nElementTX      = (size(TransTX.ElementPos,1));
nSubElementTX   = (size(TransTX.ElementPos,3));

if ~isempty(TransRX)
    nElementRX  = (size(TransRX.ElementPos,1));
    nSubElementRX   = (size(TransTX.ElementPos,3));
else
    nElementRX  = (size(TransTX.ElementPos,1));
    nSubElementRX   = (size(TransRX.ElementPos,3));
end
nTX         = (numel(TXSI));

nScatPerTransmit = (size(Media.scattererPos,1));
nDim        = (size(Media.scattererPos,2));
nEvent      = (size(Media.scattererPos,3));

%% Set Time Frequency Sampling
simSamplingFrequency    = (250e6);
outputSamplingFrequency = (RXSI(1).samplingFrequency);
demodFrequency          = (RXSI(1).demodFrequency); 
pulseDuration           = (numel(TXSI(1).pulseShape)/simSamplingFrequency);
receiveStartTime        = (RXSI(1).startTime - 1/TransTX.frequency);

maxDelayTX              = (max( TXSI(1).Delay(:) ));
for iTX = nTX:-1:2
    maxDelayTX          = (max( maxDelayTX, max(TXSI(iTX).Delay(:)) ));
end

%% SampleMode
if strcmp(RXSI(1).sampleMode,'NS200BW')
    sub = 2;
    fmin = 0;
    fmax = outputSamplingFrequency/2;
elseif strcmp(RXSI(1).sampleMode,'BS100BW')
    sub = 4;
    fmin = TransTX.frequency * (1 - 1/2);
    fmax = TransTX.frequency * (1 + 1/2);
elseif strcmp(RXSI(1).sampleMode,'BS50BW')
    sub = 8;
    fmin = TransTX.frequency * (1 - 1/4);
    fmax = TransTX.frequency * (1 + 1/4);
else
    warning('Sampling Mode Not implemented')
end

Tmax = (RXSI(1).numelSample/outputSamplingFrequency + pulseDuration + maxDelayTX + receiveStartTime );

nFastTimeSample         = (128*ceil(Tmax*outputSamplingFrequency/128));
nFrequencySample        = (nFastTimeSample/2);
simFrequency            = single(linspace(0,outputSamplingFrequency/2,nFrequencySample));



%idFrequency             = find(simFrequency>=fmin & simFrequency<=fmax);
idFrequency             = find(simFrequency>=0);

aglFrequency            = 2*pi*simFrequency;
kWaveNumber             = aglFrequency./Media.speedOfSound;
ikWaveNumber            = Media.attenuation/8.7*kWaveNumber*Media.speedOfSound/(2*pi*1e4);

kWaveNumber             = kWaveNumber + 1j*ikWaveNumber;

waveLength              = Media.speedOfSound/TransTX.frequency;

%% Aperture Array
if ~isempty(TransRX)
    r_ele_tx    = (single(TransTX.ElementPos(:,:,:))); %[:,x,y,z,az,el,:]
    r_ele_rx    = (single(TransRX.ElementPos(:,:,:)));
else
    r_ele_tx    = (single(TransTX.ElementPos(:,:,:))); %[:,x,y,z,az,el,:]
    r_ele_rx    = r_ele_tx;
    TransRX     = TransTX;
end

%% Pulse Spectrum
simSampling2outputSampling  = ceil(single(nFastTimeSample)*...
    simSamplingFrequency/...
    outputSamplingFrequency);
PULSE   = fft(TXSI(1).pulseShape, simSampling2outputSampling);

%% IR Spectrum
IRtx    = fft(TransTX.impulseResponse, simSampling2outputSampling);

if ~isempty(TransRX)
    IRrx    = fft(TransRX.impulseResponse, simSampling2outputSampling);
    IRrx    = single(IRrx(1:nFrequencySample));
else
    IRrx    = IRtx;
end

%%
PULSE   = single(PULSE(1:nFrequencySample));
IRtx    = single(IRtx(1:nFrequencySample));

if ~isempty(TransRX)
    IRrx    = fft(TransRX.impulseResponse, simSampling2outputSampling);
    IRrx    = single(IRrx(1:nFrequencySample));
else
    IRrx    = IRtx;
end

%% InputFilter
IPFilt = fft(RXSI(1).InputFilter(:),nFastTimeSample);
IPFilt = single(IPFilt(1:nFrequencySample));

%%
for itx = nTX:-1:1
    Delay(:,itx) = single(TXSI(itx).Delay(:));
    Apod(:,itx) = single(TXSI(itx).Apod(:));
    DelayAb = single(AbLaw.Delay(:));
    ApodAb = single(AbLaw.Apod(:));
end

%% Scatterer position
r_scat      = (single(Media.scattererPos(:,1:3,:)));
bsc_scat    = (single(Media.scattererPos(:,4,:)));

%% Start SIMULATION on GPU
g = gpuDevice();
nFrame          = (ceil(nEvent/nTX));
maxSizSim       = (nFrequencySample*nScatPerTransmit*nTX);
maxMemSim       = maxSizSim*2*32;
cudaMem         = g.AvailableMemory;
nChunks         = 1;%ceil(maxMemSim/cudaMem/4)*4;
nScatPerChunck  = nScatPerTransmit;%int32(ceil(nScatPerTransmit/nChunks));

%% Memory allocation
RF  = zeros(nFastTimeSample, ...
    nElementRX, ...
    nEvent,'single');

RF_ab  = zeros(nFastTimeSample, ...
    nElementRX, ...
    nEvent,'single');

%% Init GPU
%-- Transmit Wave Field (Nf, nscat, nTX)
TX  = gpuArray(complex(zeros(numel(idFrequency), ...
    1, ...
    nScatPerChunck, ...
    nTX,'single')));

%-- Receive Wave Field (Nf, nEleRX, nscat)
RX  = gpuArray(complex(zeros(numel(idFrequency), ...
    nElementRX, ...
    nScatPerChunck,'single')));

%-- Receive Wave Field (Nf, nElementRX, nTX)
TXRX = gpuArray(complex(zeros(nFrequencySample, ...
    nElementRX, 'single')));

%-- Init Gpu
[TXfieldKernel, RXfieldKernel] = InitGPU(TransTX, TransRX);

TXfieldKernel.GridSize  = ...
    ceil([numel(idFrequency)/TXfieldKernel.ThreadBlockSize(1) ...
    nScatPerChunck/TXfieldKernel.ThreadBlockSize(2) ...
    nTX/TXfieldKernel.ThreadBlockSize(3)] );

RXfieldKernel.GridSize  = ...
    ceil([numel(idFrequency)/RXfieldKernel.ThreadBlockSize(1) ...
    nElementRX/RXfieldKernel.ThreadBlockSize(2) ...
    nScatPerChunck/RXfieldKernel.ThreadBlockSize(3)] );

%% Begin
for iFrame = nFrame:-1:1
    %     disp(['Frame ',num2str(nFrame-iFrame+1),' out of ',num2str(nFrame)])
    ievent = (iFrame - 1)*nTX + (1:nTX);
    
    for iChunks = 1:nChunks
        %         disp(['Chunk ',num2str(iChunks),' out of ',num2str(nChunks)])
        iScat  = (iChunks-1)*nScatPerChunck + (1:nScatPerChunck);
        
        r_scat_tmp = ones(nScatPerChunck,3,nTX,'single').*1e-3;
        bsc_scat_tmp = zeros(nScatPerChunck,1,nTX,'single');
        
        if iScat(end)>nScatPerTransmit
            idx_scat = iScat(1):nScatPerTransmit;
            r_scat_tmp(1:numel(idx_scat),:,:) = r_scat(iScat(1):nScatPerTransmit,:,ievent);
            bsc_scat_tmp(1:numel(idx_scat),:,:) = bsc_scat(iScat(1):nScatPerTransmit,:,ievent);
        else
            r_scat_tmp(1:numel(iScat),:,:) = r_scat(iScat,:,ievent);
            bsc_scat_tmp(1:numel(iScat),:,:) = bsc_scat(iScat,:,ievent);
        end
        
        %% Transmit Wave Field (Nf, 1, nScatPerChunck, nTX)
        TX      = feval(TXfieldKernel,...
            TX, ...
            (r_scat_tmp(:,1:3,:)), ...
            (r_ele_tx), ... (:,:,isubtx)
            (kWaveNumber(idFrequency)), ...
            (PULSE(idFrequency)), ...
            (IRtx(idFrequency)), ...
            (Delay-receiveStartTime), ...
            (Apod), ...
            single(TransTX.ElementWidth(1)), ...
            single(TransTX.ElementHeight(1)), ...
            single(TransTX.ElementElevationFocus(1)), ...
            single(TransTX.baffle) , ...
            single(TransTX.lensCorrection) , ...
            single(Media.speedOfSound) , ...
            int32(nScatPerChunck), ...
            int32(nElementTX), ...
            int32(numel(idFrequency)) , ...
            int32(nTX),...
            int32(nSubElementTX));
        
        %% Apply BackScatterCoeff
        TX = TX.*permute(bsc_scat_tmp,[4 2 1 3]);
        
        %% Permute to match dim of RX
        TX(isnan(TX)) = 0;
        
        for itx = nTX:-1:1
            %% Receive Wave Field (Nf, Nele, nScatPerChunck)
            RX = feval(RXfieldKernel,...
                RX,...
                single(r_scat_tmp(:,1:3,itx)), ... %r_scat(iScat,:,iTx(ievent)), ...
                single(r_ele_rx),...    (:,:,isubrx)), ...
                single(kWaveNumber(idFrequency)), ...
                single(IRrx(idFrequency)), ...
                single(TransRX.ElementWidth(1)), ...
                single(TransRX.ElementHeight(1)), ...
                single(TransRX.ElementElevationFocus(1)), ...
                single(TransRX.baffle) , ...
                single(TransRX.lensCorrection) , ...
                single(Media.speedOfSound) , ...
                int32(nScatPerChunck), ...
                int32(nElementRX), ...
                int32(numel(idFrequency)), ...
                int32(nSubElementRX));
            
            %% Apodization in receive (Nf, Nele, 1 )
            % s_ab = AbLaw.Apod(:).'.*exp(1j.*aglFrequency(idFrequency).'.*AbLaw.Delay(:).');

            Law_ab    =ApodAb.'.*exp(1j.*aglFrequency.'.*DelayAb.');
            RX = RX.*RXSI(itx).Apod(:)';
            RX_ab = RX.*Law_ab;

            %% Sum over scatterer
            TXRX(idFrequency,:,:) = sum(TX(:,:,:,itx).*RX,3).*IPFilt(idFrequency);
            TXRX_ab(idFrequency,:,:) = sum(TX(:,:,:,itx).*RX_ab,3).*IPFilt(idFrequency);
            
            %-- Evaluate TXRX field in time domain
            TXRX = real(ifft(conj(TXRX),nFastTimeSample,1));
            TXRX_ab = real(ifft(conj(TXRX_ab),nFastTimeSample,1));
            
            %-- Gather output from device
            RF(:,:,ievent(itx)) = RF(:,:,ievent(itx)) + gather(TXRX);
            RF_ab(:,:,ievent(itx)) = RF_ab(:,:,ievent(itx)) + gather(TXRX_ab);
            
            %-- reset TXRX
            TXRX = TXRX*0;
            
        end
    end
end

% Reshape Output
if size(RF,1) > (RXSI(1).numelSample)
    RF = RF(1:ceil(RXSI(1).numelSample),:,:);
else
    RF = cat(1,RF,zeros(RXSI(1).numelSample-size(RF,1),nElementRX,nEvent));
end

if size(RF_ab,1) > (RXSI(1).numelSample)
    RF_ab = RF_ab(1:ceil(RXSI(1).numelSample),:,:);
else
    RF_ab = cat(1,RF_ab,zeros(RXSI(1).numelSample-size(RF_ab,1),nElementRX,nEvent));
end

%
Raw(1:2:size(RF,1)/sub*2, :, :) = RF(1:sub:end, :, :);
Raw(2:2:size(RF,1)/sub*2, :, :) = RF(2:sub:end, :, :);

Raw_ab(1:2:size(RF,1)/sub*2, :, :) = RF_ab(1:sub:end, :, :);
Raw_ab(2:2:size(RF,1)/sub*2, :, :) = RF_ab(2:sub:end, :, :);

% End SIMULATION
end

function [TXfieldKernel, RXfieldKernel] = InitGPU(TransTX, TransRX)
% Transmit
TXfieldKernel = parallel.gpu.CUDAKernel('SimUS_CUDARCA_ab.ptx', ...
    ['float2*,' ...         % Output Pfield
    'const float*,' ...     % r_grid
    'const float*,' ...     % r_ele
    'const float2*, ' ...   % kWaveNumber
    'const float2*,' ...    % PULSE
    'const float2*, ' ...   % IR
    'const float*, ' ...    % Delay
    'const float*, ' ...    % Apod
    'const float, ' ...     % width
    'const float, ' ...     % height
    'const float, ' ...     % rfocus
    'const float, ' ...     % baffle
    'const float, ' ...     % lensCorrection
    'const float, ' ...     % SpeedOfSound
    'const int, ' ...       % nScatPerEvent
    'const int, ' ...       % nElement
    'const int, ' ...       % Nf
    'const int, ' ...       % Ntx
    'const int '],...       % Nsubele
    'TXfield');
TXfieldKernel.ThreadBlockSize = [128 8 1]; % prod = 1024

% Receive
RXfieldKernel = parallel.gpu.CUDAKernel('SimUS_CUDARCA_ab.ptx', ...
    ['float2*,' ...         % Output Pfield
    'const float*,' ...     % r_scat
    'const float*,' ...     % r_ele
    'const float2*, ' ...   % kWaveNumber
    'const float2*, ' ...   % IR
    'const float, ' ...     % width
    'const float, ' ...     % height
    'const float, ' ...     % rfocus
    'const float, ' ...     % baffle
    'const float, ' ...     % lensCorrection
    'const float, ' ...     % SpeedOfSound
    'const int, ' ...       % nScatPerEvent
    'const int, ' ...       % nElement
    'const int,'...         % Nf
    'const int'],...        % Nsubele
    'RXfield');
RXfieldKernel.ThreadBlockSize = [128 8 1]; % prod = 1024

end