function [dataBf, mask] = bfVirtualSourceGeneric_RCA_ab_2(varargin)
% bfVirtualSourceGeneric_RCA 3D/2D/1D I/Q beamformer for virtual sources behind the array.
%     [dataBf, mask] = bfVirtualSourceGeneric_RCA(data, Geometry, Transmit, Param)
%      dataBf        = bfVirtualSourceGeneric_RCA(data, Geometry, Transmit, Param)
%     [dataBf, mask] = bfVirtualSourceGeneric_RCA(data, Geometry, Transmit, Param, Options)
%      dataBf        = bfVirtualSourceGeneric_RCA(data, Geometry, Transmit, Param, Options)
%                      bfVirtualSourceGeneric_RCA()
% 
%
%
%     OUTPUTS:
%         dataBf: complex beamformed data [z-axis x x-axis x y-axis x transmits x frames]
%         mask  : mask [z-axis x x-axis x y-axis x transmits]
%
%     INPUTS:
%         data    : raw data [samples x (x-channels x 1 OR 1 x y-channels) x events]
%         Geometry: structure describing geometry
%         Param   : structure describing beamforming parameters
%         Transmit: structure describing transmit parameters
%         Options : structure for some extra optional parameters
%
%
%
%     Geometry: Define the geometry
%         gridBfX  : x-grid for beamforming (m, 3D meshgrid)
%         gridBfY  : y-grid for beamforming (m, 3D meshgrid)
%         gridBfZ  : z-grid for beamforming (m, 3D meshgrid)
%         posEleX  : position of the elements along x-axis (m, vector)
%         posEleY  : position of the elements along y-axis (m, vector)
%         posEleZ  : position of the elements along z-axis (m, vector)
%         apertureX: aperture along x-axis (m, single value, default = range of posEleX)
%         apertureY: aperture along y-axis (m, single value, default = range of posEleY)
%         apertureZ: aperture along z-axis (m, single value, default = range of posEleZ)
%
%     Transmit: Transmit scheme
%         thetaX        : tilt angles along x-axis (rad, vector, length = transmits
%         thetaY        : tilt angles along y-axis (rad, vector, length = transmits
%         OR
%         virtualSourceX: position of the virtual sources along x-axis (m, vector, length = transmits)
%         virtualSourceY: position of the virtual sources along y-axis (m, vector, length = transmits)
%         virtualSourceZ: position of the virtual sources along z-axis (m, vector, length = transmits)
%         
%         flagReceiveX   : flag for receive channels (flag=1 receive on X channels, flag=0 receive on Y channels)
%
%
%     Param: Beamforming parameters
%         samplingFrequency    : sampling frequency of the INPUT data (Hz)
%         demodulationFrequency: transmit center frequency (Hz)
%         speedOfSound         : speed-of-sound (m/s, default = 1540 m/s)
%         startTime            : start time (s, default = 0 s)
%         fnumber              : f-number (default = 0)
% 
%     Options: Extra optional parameters
%         flagInterpolationLanczos: flag for interpolation method (boolean, default = false, true = Lanczos, false = quadratic) 
%         flagCompounding         : flag for the coumpounding (boolean, default = true, true = do the coumpounding)
%         flagMaskTransmit        : flag for the mask in transmit (boolean, default = true, true = only beamform insonied pixels)
%         decreaseFactorGPU       : decrease factor for the GPU memory (integer, default = 1)
%         
%
%
%     NOTES:
%         The PTX must be compiled once for your GPU configuration.
%         Without any input, the function will compile the PTX (for CUDA)
%         for you. 
%         Run "bfVirtualSourceGeneric_RCA()" before first use.
%
%         If a 2D/1D array is used, be sure to use singleton dimensions 
%         (permute) in the array/data to keep a 3D form the inputs. It is
%         also important to manually set the apertures for a linear array 
%         for the out-of-plane axis (should be equal to the  height of the 
%         elements for most cases).
% 
%         You need to specify the transmit angle(s) OR the position of the
%         virtual source(s) but NOT a mix of both. Transmit angle(s) will
%         be converted to virtual source(s) (far behind the array).
%         The origin of the axes MUST be at the center of the array.
% 
%         Lanzcos 5-lobes can be twice as long as the quadractic 
%         interpolation. For most applications the quadratic interpolation 
%         is the best method, Lanzcos 5-lobes can be use for a slightly 
%         more precised beamforming but the gain is very low.
% 
%         DO NOT CHANGE the parameters (grid/size) of the CUDA kernel.
%         By passing all the grids, frames, and angles to the GPU, the 
%         calculation time has been reduced by half compared to the 
%         previous version. The optimization of the handling of the data by 
%         CUDA (grids, size, chunks) further reduce the calculation time 
%         of about 20 percent for large dataset.
% 
%
%
%     EXAMPLE: For the Verasonics system, param should be for most applications
%         Param.samplingFrequency     = (Receive(1).decimSampleRate*1e6)/ ...
%                                       (2*Receive(1).quadDecim); % Sampling frequency of IQ signals (Hz)
%         Param.demodulationFrequency = Receive(1).demodulationFrequency*1e6; % Demodulation frequency (Hz)
%         Param.speedOfSound          = Resource.Parameters.speedOfSound; % Speed-of-sound (m/s)
%         Param.startTime             = ((-2*(Receive(1).startDepth- ...
%                                         Trans.lensCorrection))+ ...
%                                         TW(1).peak)/ ...
%                                        (Trans.frequency*1e6);  % Start time (s)
%         Param.fnumber               = 1; % F-number, YOU CAN TUNE THIS VALUE 
% 
% 
%
% Modified from bfVirtualSourceGeneric (Vincent Perrot, polymtl, Montreal, Canada)
% April 26, 2022.



%% INITIALIZATION
%--- Compile ptx if no input
if nargin == 0
    compilePTX;
    return
end


%--- Number of inputs
narginchk(4, 5); % Three or four inputs

if nargin == 4 % Without options (4 args)
    data = varargin{1};
    Geometry = varargin{2};
    Transmit = varargin{3};
    Param = varargin{4};
    Options = []; % Empty, use all default value
else % With options (5 args)
    data = varargin{1};
    Geometry = varargin{2};
    Transmit = varargin{3};
    Param = varargin{4};
    Options = varargin{5};
end


%--- data
if ~isnumeric(data) % 'data' must be a numeric array
    errStr.message = 'Input data (''data'') must be a numeric array!';
    errStr.identifier = [mfilename, ':dataNotNumeric'];
    error(errStr);
end

if isreal(data) % If 'data' are real
    wrnStr.message = ['Input data ', ...
        '(''data'') are real. I am assuming that the I and Q channels ', ...
        'are interleaved.'];
    wrnStr.identifier = [mfilename, ':dataReal'];
    warning(wrnStr.message, wrnStr.identifier);
end


%--- Options
if ~isstruct(Options) && ~isempty(Options) % 'Options' must be a structure
    errStr.message = 'Options (''Options'') must be a structure!';
    errStr.identifier = [mfilename, ':OptionsNotStruct'];
    error(errStr);
end

if ~isfield(Options, 'flagInterpolationLanczos') % Interpolation method
    Options.flagInterpolationLanczos = false; % Default (quadratic interpolation)
end

if ~islogical(Options.flagInterpolationLanczos) % Invalid flag
    errStr.message = ['Interpolation method ', ...
        '(''Options.flagInterpolationLanczos'') is invalid; it must be a ', ...
        'boolean (true = Lanczos 5-lobes, false = quadratic)!'];
    errStr.identifier = [mfilename, ':flagInterpolationLanczosInvalid'];
    error(errStr);
end

if ~isfield(Options, 'flagCompounding') % Set default value for compounding
    Options.flagCompounding = true; % Default (do compounding)
elseif ~islogical(Options.flagCompounding) % If invalid value
    errStr.message = ['Compounding flag ', ...
        '(''Options.flagCompounding'') must be a boolean!'];
    errStr.identifier = [mfilename, ':flagCompoundingInvalid'];
    error(errStr);
end
     
if ~isfield(Options, 'flagMaskTransmit') % Compounding
    Options.flagMaskTransmit = true; % Default (use transmit mask)
elseif ~islogical(Options.flagMaskTransmit) % If invalid value
    errStr.message = ['Transmit mask flag ', ...
        '(''Options.flagMaskTransmit'') must be a boolean!'];
    errStr.identifier = [mfilename, ':flagMaskTransmitInvalid'];
    error(errStr);
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


%--- Geometry
if ~isfield(Geometry, 'gridBfX') || ...
   ~isfield(Geometry, 'gridBfY') || ...
   ~isfield(Geometry, 'gridBfZ')  % Grids are required
    errStr.message = ['Beamforming grids ', ...
        '(''Geometry.gridBfX'', ''Geometry.gridBfY'', ', ...
        'and ''Geometry.gridBfX'')', ...
        'are required!'];
    errStr.identifier = [mfilename, ':gridBfMissing'];
    error(errStr);
elseif (numel(Geometry.gridBfX) ~= numel(Geometry.gridBfY)) || ...
       (numel(Geometry.gridBfY) ~= numel(Geometry.gridBfZ)) || ...
       numel(Geometry.gridBfX) < 1 || ...
       numel(Geometry.gridBfY) < 1 || ...
	   numel(Geometry.gridBfZ) < 1 || ...
       ~isnumeric(Geometry.gridBfX) || ...
       ~isnumeric(Geometry.gridBfY) || ...
       ~isnumeric(Geometry.gridBfZ) % If invalid
    errStr.message = ['Beamforming grids ', ...
        '(''Geometry.gridBfX'', ''Geometry.gridBfY'', ', ...
        'and ''Geometry.gridBfX'') ', ...
        'must must be 3D numeric meshgrids and have the same size!'];
    errStr.identifier = [mfilename, ':gridBfInvalid'];
    error(errStr);
end

if ~isfield(Geometry, 'posEleX') || ...
   ~isfield(Geometry, 'posEleY') || ...
   ~isfield(Geometry, 'posEleZ')  % Grids are required
    errStr.message = ['Element positions ', ...
        '(''Geometry.posEleX'', ''Geometry.posEleY'', ', ...
        'and ''Geometry.posEleZ'')', ...
        'are required!'];
    errStr.identifier = [mfilename, ':posEleMissing'];
    error(errStr);
elseif (numel(Geometry.posEleX) ~= numel(Geometry.posEleY)) || ...
       (numel(Geometry.posEleY) ~= numel(Geometry.posEleZ)) || ...
       numel(Geometry.posEleX) < 1 || ...
       numel(Geometry.posEleY) < 1 || ...
	   numel(Geometry.posEleZ) < 1 || ...
       ~isnumeric(Geometry.posEleX) || ...
       ~isnumeric(Geometry.posEleY) || ...
       ~isnumeric(Geometry.posEleZ) % If invalid
    errStr.message = ['Pposition of the element ', ...
        '(''Geometry.posEleX'', ''Geometry.posEleY'', ', ...
        'and ''Geometry.posEleZ'') ', ...
        'must be numeric arrays and have the same size!'];
    errStr.identifier = [mfilename, ':posEleInvalid'];
    error(errStr);
end

if ~isfield(Geometry, 'apertureX') % Aperture 
	flagApertureX = true; % Get aperture from the position of the elements
    wrnStr.message = ['Aperture along the x-axis ', ...
        '(''Param.apertureX'') is missing, the position of the elements ',... 
        'will be used to estimate it. If this is an out-of-plane axis it ', ...
        'is recommended to set that value manually based on the size of ', ...
        'the elements!'];
    wrnStr.identifier = [mfilename, ':apertureXDefault'];
    warning(wrnStr.message, wrnStr.identifier);
elseif ~isnumeric(Geometry.apertureX) || ...
        length(Geometry.apertureX) ~= 1  % If invalid value
    errStr.message = ['Aperture along the x-axis ', ...
        '(''Geometry.apertureX'') must be a single numeric value!'];
    errStr.identifier = [mfilename, ':apertureXInvalid'];
    error(errStr);
else % No error
    flagApertureX = false; % DO NOT get aperture from the position of the elements
end

if ~isfield(Geometry, 'apertureY') % Aperture 
	flagApertureY = true; % Get aperture from the position of the elements
    wrnStr.message = ['Aperture along the y-axis ', ...
        '(''Param.apertureY'') is missing, the position of the elements ',... 
        'will be used to estimate it. If this is an out-of-plane axis it ', ...
        'is recommended to set that value manually based on the size of ', ...
        'the elements!'];
    wrnStr.identifier = [mfilename, ':apertureYDefault'];
    warning(wrnStr.message, wrnStr.identifier);
elseif ~isnumeric(Geometry.apertureY) || ...
        length(Geometry.apertureY) ~= 1  % If invalid value
    errStr.message = ['Aperture along the y-axis ', ...
        '(''Geometry.apertureY'') must be a single numeric value!'];
    errStr.identifier = [mfilename, ':apertureYInvalid'];
    error(errStr);
else % No error
    flagApertureY = false; % DO NOT get aperture from the position of the elements
end

if ~isfield(Geometry, 'apertureZ') % Aperture 
	flagApertureZ = true; % Get aperture from the position of the elements
    wrnStr.message = ['Aperture along the z-axis ', ...
        '(''Param.apertureZ'') is missing, the position of the elements ',... 
        'will be used to estimate it. If this is an out-of-plane axis it ', ...
        'is recommended to set that value manually based on the size of ', ...
        'the elements!'];
    wrnStr.identifier = [mfilename, ':apertureZDefault'];
    warning(wrnStr.message, wrnStr.identifier);
elseif ~isnumeric(Geometry.apertureZ) || ...
        length(Geometry.apertureZ) ~= 1  % If invalid value
    errStr.message = ['Aperture along the z-axis ', ...
        '(''Geometry.apertureZ'') must be a single numeric value!'];
    errStr.identifier = [mfilename, ':apertureZInvalid'];
    error(errStr);
else % No error
    flagApertureZ = false; % DO NOT get aperture from the position of the elements
end


%--- Transmit
if ~isstruct(Transmit) % 'Transmit' must be a structure
    errStr.message = 'Transmit (''Transmit'') must be a structure array!';
    errStr.identifier = [mfilename, ':TransmitNotStruct'];
    error(errStr);
end

if any([isfield(Transmit, 'thetaX'), isfield(Transmit, 'thetaY')]) == ...
   any([isfield(Transmit, 'virtualSourceX'), ...
        isfield(Transmit, 'virtualSourceY'), ...
        isfield(Transmit, 'virtualSourceZ')]) % Use either the angle(s) OR the virtual source(s)
    errStr.message = ['Origin of the insonification(s) ', ...
        '(''thetaX'' and ''thetaY'' OR ', ...
        '''virtualSourceX'', ''virtualSourceY'', and ''virtualSourceZ'') ',... 
        'is required and must be unique ' ,...
        '(use either angle(s) OR virtual source(s))!'];
    errStr.identifier = [mfilename, ':ThetaORVirtualSourceRequired'];
    error(errStr);
    
elseif isfield(Transmit, 'thetaX') && isfield(Transmit, 'thetaY') % If angle
    flagTheta = true; % Flag to compute the virtual source(s)
    wrnStr.message = ['Tilt angles ', ...
        '(''Transmit.thetaX'' and ''Transmit.thetaY'') ', ...
        'will be converted to virtual source(s)'];
    wrnStr.identifier = [mfilename, ':thetaToVirtualSource'];
    warning(wrnStr.message, wrnStr.identifier);
    
    if length(Transmit.thetaX) ~= length(Transmit.thetaY) % Tilts must have the same length
        errStr.message = ['Tilts ',...
            '(''Param.thetaX'' and ''Param.thetaY'') ', ...
            ' must have the same length!'];
        errStr.identifier = [mfilename, ':tiltsNotSameLength'];
        error(errStr);
    end
 
elseif isfield(Transmit, 'virtualSourceX') && ...
       isfield(Transmit, 'virtualSourceY') && ...
       isfield(Transmit, 'virtualSourceZ') % If virtual sources     
    flagTheta = false; % Flag to NOT compute the virtual source(s)
    
    if (length(Transmit.virtualSourceX) ~= length(Transmit.virtualSourceY)) || ...
       (length(Transmit.virtualSourceX) ~= length(Transmit.virtualSourceZ))% Virtual sources must have the same length
        errStr.message = ['Virtual sources ', ...
            '(''Param.virtualSourceX'', ''Param.virtualSourceY'', ' , ...
            'and ''Param.virtualSourceZ'') ', ...
            'must have the same length!'];
        errStr.identifier = [mfilename, ':virtualSourcesNotSameLength'];
        error(errStr);
    end
    
else
    errStr.message = ['Origin of the insonification(s) ', ...
        '(''thetaX'' and ''thetaY'' OR ', ...
        '''virtualSourceX'', ''virtualSourceY'', and ''virtualSourceZ'') ', ...
        'is incorrect. Verify that all axes/positions are well defined ', ...
        'and that you are using either angle(s) OR virtual source(s))!'];
    errStr.identifier = [mfilename, ':ThetaORVirtualSourceInvalid'];
    error(errStr);
end


if ~isfield(Transmit, 'flagReceiveX') % Compounding
    Transmit.flagReceiveX = true; % Default (use X channels receive)
    wrnStr.message = ['Receive channels (X or Y) ', ...
        '(''Transmit.flagReceiveX'') is missing, Xchannels receive is chosen by default! '];
elseif ~islogical(Transmit.flagReceiveX) % If invalid value
    errStr.message = ['Transmit receive channel flag ', ...
        '(''Transmit.flagReceiveX'') must be a boolean!'];
    errStr.identifier = [mfilename, ':flagReceiveXInvalid'];
    error(errStr);
end


%--- Param
if ~isstruct(Param) % 'Param' structure must be a structure
    errStr.message = 'Parameters (''Param'') must be a structure array!';
    errStr.identifier = [mfilename, ':ParamNotStruct'];
    error(errStr);
end

if ~isfield(Param, 'samplingFrequency') % Sampling frequency
    errStr.message = ['Sampling frequency ', ...
        '(''Param.samplingFrequency'') is required!'];
    errStr.identifier = [mfilename, ':samplingFrequencyMissing'];
    error(errStr);
elseif ~isnumeric(Param.samplingFrequency) || ...
        length(Param.samplingFrequency) ~= 1  % If invalid value
    errStr.message = ['Sampling frequency ', ...
        '(''Param.samplingFrequency'') must be a single numeric value!'];
    errStr.identifier = [mfilename, ':samplingFrequencyInvalid'];
    error(errStr);
end

if ~isfield(Param, 'demodulationFrequency') % Demodulation frequency
    errStr.message = ['Demodulation frequency ', ...
        '(''Param.demodulationFrequency'') is required!'];
    errStr.identifier = [mfilename, ':demodulationFrequencyMissing'];
    error(errStr);
elseif ~isnumeric(Param.demodulationFrequency) || ...
        length(Param.demodulationFrequency) ~= 1
    errStr.message = ['Demodulation frequency ', ...
        '(''Param.demodulationFrequency'') must be a single numeric value!'];
    errStr.identifier = [mfilename, ':demodulationFrequencyInvalid'];
    error(errStr);
end

if ~isfield(Param, 'speedOfSound') % Speed of sound
    Param.speedOfSound = 1540; % Default
    wrnStr.message = ['Speed of sound (''Param.speedOfSound'') ', ...
        'is missing, default = ', num2str(Param.speedOfSound), ' m/s.'];
    wrnStr.identifier = [mfilename, ':speedOfSoundDefault'];
    warning(wrnStr.message, wrnStr.identifier);
elseif ~isnumeric(Param.speedOfSound) || ...
        length(Param.speedOfSound) ~= 1 % If invalid value
    errStr.message = ['Speed of sound ', ...
        ' (''Param.speedOfSound'') must be a single numeric value!'];
    errStr.identifier = [mfilename, ':speedOfSoundInvalid'];
    error(errStr);
end

if ~isfield(Param, 'startTime') % Start time
    Param.startTime = 0; % Default (start at 0s)
    wrnStr.message = ['Start time (''Param.startTime'') is missing, ', ... 
        'default = ', num2str(Param.startTime), '.'];
    wrnStr.identifier = [mfilename, ':startTimeDefault'];
    warning(wrnStr.message, wrnStr.identifier);
elseif ~isnumeric(Param.startTime) || ...
        length(Param.startTime) ~= 1  % If invalid value
    errStr.message = ['Start time ', ...
        '(''Param.startTime'') must be a single numeric value!'];
    errStr.identifier = [mfilename, ':startTimeInvalid'];
    error(errStr);
end

if ~isfield(Param, 'fnumber') % F-number
    Param.fnumber = 1; % Default (1 is a common value)
    wrnStr.message = ['F-number (''Param.fnumber'') is missing,', ... 
        ' default = ', num2str(Param.fnumber), '.'];
    wrnStr.identifier = [mfilename, ':fnumberDefault'];
    warning(wrnStr.message, wrnStr.identifier);
elseif ~isnumeric(Param.fnumber) || ...
        length(Param.fnumber) ~= 1 % If invalid value
    errStr.message = ['F-number ', ...
        ' (''Param.startTime'') must be a single numeric value!'];
    errStr.identifier = [mfilename, ':startTimeInvalid'];
    error(errStr);
end



%% PARAMETERS
samplingFrequency     = single(Param.samplingFrequency); % Extract sampling frequency
demodulationFrequency = single(Param.demodulationFrequency); % Extract demodulation frequency
speedOfSound          = single(Param.speedOfSound); % Extract speed of sound
startTime             = single(Param.startTime); % Extract start time
fnumber               = single(Param.fnumber); % Extract f-number



%% DATA
%--- Data size
[nSample, nColX, nColY, nEvent] = size(data); % Input data size
nChannel                        = nColX*nColY; % Number of channels


%--- If real, interleaved
if isreal(data) % If data is real
    data(2:2:end) = -data(2:2:end); % Reverse sign: IQ(n) = RF(2*n) - i*RF(2*n+1);
    data          = reshape(data, nSample, nChannel, []); % Reshape for CUDA

    
%--- If complex, already demodulated
else % If data is complex
    nSample = 2*nSample; % Double the number of real samples
    data = reshape([real(data(:)) imag(data(:))].', ...
                       nSample, nChannel, nEvent);  % Reshape for CUDA
end



%% GEOMETRY
%--- Beamforming grids
gridBfX = Geometry.gridBfX(:).'; % Extract beamforming grid along x-axis
gridBfY = Geometry.gridBfY(:).'; % Extract beamforming grid along y-axis
gridBfZ = Geometry.gridBfZ(:).'; % Extract beamforming grid along z-axis

%--- Position of the elements
posEleX = Geometry.posEleX(:).'; % Extract element positions along x-axis
posEleY = Geometry.posEleY(:).'; % Extract element positions along y-axis
posEleZ = Geometry.posEleZ(:).'; % Extract element positions along z-axis

posEleX = posEleX-mean(posEleX); % Centered element positions along x-axis 
posEleY = posEleY-mean(posEleY); % Centered element positions along y-axis
posEleZ = posEleZ-mean(posEleZ); % Centered element positions along z-axis

%--- Apertures
if flagApertureX % If true, aperture was not provided so compute it
    apertureX = max(posEleX)-min(posEleX); % Aperture along x-axis
else % If false, was provided
    apertureX = Geometry.apertureX; % Extract aperture along x-axis
end

if flagApertureY % If true, aperture was not provided so compute it
    apertureY = max(posEleY)-min(posEleY); % Aperture along y-axis
else % If false, was provided
    apertureY = Geometry.apertureY; % Extract aperture along y-axis
end

if flagApertureZ % If true, aperture was not provided so compute it
    apertureZ = max(posEleZ)-min(posEleZ); % Aperture along z-axis
else % If false, was provided
    apertureZ = Geometry.apertureZ; % Extract aperture along z-axis
end



%% VIRTUAL SOURCES
%--- Virtual sources
if flagTheta % If true, angles were used so compute virtual sources
    
    r = 10; % Distance from the virtual source to the array
    
    virtualSourceX = zeros(1, length(Transmit.thetaX)); % Pre-allocation
    virtualSourceY = virtualSourceX; % Pre-allocation   
    virtualSourceZ = virtualSourceY; % Pre-allocation    
    
    for s = 1:length(virtualSourceX)
        virtualSourceZ(s) = -r/...
            (sqrt(1+tan(Transmit.thetaX(s)).^2+ ...
            tan(Transmit.thetaY(s)).^2)); % Coordinate along z-axis
        virtualSourceX(s) = virtualSourceZ(s)*tan(Transmit.thetaX(s)); % Coordinate along x-axis
        virtualSourceY(s) = virtualSourceZ(s)*tan(Transmit.thetaY(s)); % Coordinate along y-axis
    end
    
else % If virtual source were used
    virtualSourceX = Transmit.virtualSourceX(:).'; % Extract virtual source positions along x-axis
    virtualSourceY = Transmit.virtualSourceY(:).'; % Extract virtual source positions along y-axis
    virtualSourceZ = Transmit.virtualSourceZ(:).'; % Extract virtual source positions along z-axis
end


%--- Minimal distance to the virtual source(s)
minDistVirtualSource = zeros(1, length(virtualSourceX)); % Pre-allocation

for s = 1:length(minDistVirtualSource)
    minDistVirtualSource(s) = min(sqrt( ...
                          (posEleY-virtualSourceX(s)).^2+ ...
                          (posEleX-virtualSourceY(s)).^2+ ...
                          (posEleZ-virtualSourceZ(s)).^2));
end



%% SIZES
%--- Beamforming grid (for reshaping after beamforming)
sizeGridBf = size(Geometry.gridBfX); % Size of the beamforming grid

if length(sizeGridBf) == 1 % 1D
        sizeGridBf = [sizeGridBf 1 1]; % Add singleton dimensions
elseif length(sizeGridBf) == 2 % 2D
        sizeGridBf = [sizeGridBf 1]; % Add singleton dimension
end

%--- Data
nPix   = numel(Geometry.gridBfX); % Number of pixels to beamform
nTx    = numel(virtualSourceX); % Number of transmits

data   = reshape(data, nSample, nChannel, nTx, []); % Reshape data for CUDA

nFrame = size(data, 4); % Number of frames



%% OPTIONS
flagMaskTransmit = Options.flagMaskTransmit; % Extract



%% CHECK IF CHUNKS ARE REQUIRED
maxSizCuda = 2^32/2-1; % Maximum variable size for CUDA
maxSizBf   = 2*nPix*nTx*nFrame; % Total number of pixels for beamforming

nChunks         = ceil(maxSizBf/maxSizCuda*Options.decreaseFactorGPU); % Number of chunks
chunksList      = round(linspace(0, nFrame, nChunks+1)); % List of chunks
chunksList(end) = nFrame; % Avoid rounding errors
nFrameMax       = max(diff(chunksList)); % Maximum number of frames



%% CASTING
chunksList = uint32(chunksList); % Cast

gridBfX = single(gridBfX); % Extract and cast
gridBfY = single(gridBfY); % Extract and cast
gridBfZ = single(gridBfZ); % Extract and cast

posEleX = single(posEleX); % Cast
posEleY = single(posEleY); % Cast
posEleZ = single(posEleZ); % Cast

virtualSourceX = single(virtualSourceX); % Cast
virtualSourceY = single(virtualSourceY); % Cast
virtualSourceZ = single(virtualSourceZ); % Cast

minDistVirtualSource = single(minDistVirtualSource); % Cast

apertureX = single(apertureX); % Cast
apertureY = single(apertureY); % Cast
apertureZ = single(apertureZ); % Cast

samplingFrequency = single(samplingFrequency); % Cast
demodulationFrequency    = single(demodulationFrequency); % Cast

speedOfSound      = single(speedOfSound); % Cast

startTime         = single(startTime); % Cast

fnumber           = single(fnumber); % Cast

AbLawRx           = single(Param.AbLawRx); % Ajout loi aberration reception

nPix      = uint32(nPix); % Cast
nChannel  = uint32(nChannel); % Cast
nTx       = uint32(nTx); % Cast

nSampleIQ = uint32(nSample/2); % Cast

flagInterpolationLanczos = logical(Options.flagInterpolationLanczos); % Cast
flagMaskTransmit         = logical(flagMaskTransmit); % Cast
flagReceiveX             = logical(Transmit.flagReceiveX); % Cast


%% DO BF WITH CUDA
%--- CUDA kernel
cudaKernel = parallel.gpu.CUDAKernel('bfVirtualSourceGeneric_RCA_ab_2.ptx',...
    'bfVirtualSourceGeneric_RCA_ab_2.cu',...
    'bfVirtualSourceGeneric_RCA_ab_2'); % Create CUDA object

cudaKernel.ThreadBlockSize = [1024 1 1]; % Thread (1024). DO NOT CHANGE.

cudaKernel.GridSize = ceil([...
    single(nPix)/cudaKernel.ThreadBlockSize(1) ...
    single(nTx)/cudaKernel.ThreadBlockSize(2) ...
    single(nFrame)/cudaKernel.ThreadBlockSize(3)]); % Size of the CUDA grid


%--- Pre-allocation
if Options.flagCompounding % If compounding
    dataBf = zeros(nPix, 1, nFrame, 'single'); % Beamformed data for compounding
else
    dataBf = zeros(nPix, nTx, nFrame, 'single'); % Beamformed data without compounding
end

maskGPU   = gpuArray(zeros(nPix, nTx, 'single')); % Mask  (gpu array)
dataBfGPU = gpuArray(zeros(2*nPix, nTx, nFrameMax, 'single')); % Beamformed data (gpu array)


%--- Run CUDA for each chunk
for k = 1:length(chunksList)-1
    
    framekVector = chunksList(k)+1:chunksList(k+1); % Frame to use
    nFramek = length(framekVector); % Number of frames in the chunk
    
    [dataBfGPU, maskGPU] = feval(cudaKernel, ...
        dataBfGPU, maskGPU, ...
        data(:, :, :, framekVector), ...
        gridBfX, gridBfY, gridBfZ, ...
        posEleX, posEleY, posEleZ, ...
        virtualSourceX, virtualSourceY, virtualSourceZ, ...
        minDistVirtualSource, ...
        flagReceiveX,...
        apertureX, apertureY, apertureZ, ...
        samplingFrequency, demodulationFrequency, ...
        speedOfSound, ...
        startTime, ...
        AbLawRx, ...
        fnumber, ...
        nPix, nChannel, nTx, nFramek, ...
        nSampleIQ,...
        flagInterpolationLanczos, ...
        flagMaskTransmit);
    
    
    %--- Gather data w/ coumpounding
    if Options.flagCompounding
        dataBf(:, :, framekVector) = ...
            complex( ...
            gather(sum(dataBfGPU(1:2:end, :, 1:nFramek), 2)), ...
            gather(sum(dataBfGPU(2:2:end, :, 1:nFramek), 2))); % Complex beamformed data
        if k == 1 % only for the first chunk
            mask = gather(sum(maskGPU, 2)); % Mask
        end
        
        
    %--- Gather data  w/o coumpounding
    else
        dataBf(:, :, framekVector) = ...
            complex(...
            gather(dataBfGPU(1:2:end, :, 1:nFramek)), ...
            gather(dataBfGPU(2:2:end, :, 1:nFramek))); % Complex beamformed data
        if k == 1 % only for the first chunk
            mask = gather(maskGPU); % Mask
        end
    end
    
end



%% RESHAPE OUTPUT DATA
dataBf = reshape(dataBf, ...
    sizeGridBf(1), sizeGridBf(2), sizeGridBf(3), ...
    [], nFrame);  % Beamformed data

mask = reshape(mask, ...
    sizeGridBf(1), sizeGridBf(2), sizeGridBf(3), ...
    []); % Total mask

end



%% COMPILE PTX
function compilePTX

    %--- Get initial and function paths 
    initialPath = pwd; % Initial path
    functionPath = fileparts(which(mfilename)); % Path of the current function
    

    %--- Error message and ID if needed
    errStr.message = ['Microsoft Visual Studio not found! ', ...
        'You must have installed it in the default folder.\n\n', ...
        'If you want to manually compile the PTX, it must be compiled in', ...
        'the same folder that the CUDA file before first use.\n', ...
        'The command (Visual Studio required) must be of the form', ...
        '(the exact path is system dependent):\n\n', ...
        'system(''nvcc -ptx bfVirtualSourceGeneric_RCA.cu -use_fast_math', ...
        ' -ccbin "C:/Program Files (x86)/Microsoft Visual Studio/', ...
        '[YEAR]/[Professional OR Enterprise OR Community]/', ...
        'VC/Tools/MSVC/[VERSION]/bin/Hostx64/x64"'')\n\n'];
    errStr.identifier = [mfilename, ':VisualStudioNotFound'];
    
    
    %--- Find the path to Visual Studio
    str = [getenv('ProgramFiles') ...
           filesep 'Microsoft Visual Studio'];

    if ~isfolder(str)
        str = [getenv('ProgramW6432') ...
               filesep 'Microsoft Visual Studio'];
        if ~isfolder(str)
            str = [getenv('ProgramFiles(x86)') ...
               filesep 'Microsoft Visual Studio'];
        else % No path was found
            error(errStr); % Generate an error
        end
    end


    %--- Find the different CUDA version
    cudaList = dir([str filesep '*' filesep '*' filesep 'VC' filesep, ...
                   'Tools' filesep 'MSVC' filesep '*' filesep, ...
                   'bin' filesep 'Hostx64' filesep 'x64']);
    cudaList = cudaList([cudaList(:).isdir]);
    cudaList = cudaList(~ismember({cudaList.name},{'.','..'}));
    cudaList = {cudaList.folder};

    if isempty(cudaList) % No version was found
        error(errStr); % Generate an error
    end


    %--- Ask to the user what version to use
    if length(cudaList) > 1 % Ask the user if more than one version
        fprintf('\nCUDA toolkits found:\n')
        for n = 1:length(cudaList)
            fprintf('[%u] %s\n', n, cudaList{n});
        end

    p = input('\nSelect the CUDA Toolkit you want to use:\n');
    cudaSelected = cudaList{p};

    else % Select it if only one version was found
        cudaSelected = cudaList{1};
    end


    %--- Run nvcc
    cd(functionPath); % Move to the function path to compile the PTX
    system(['nvcc -ptx bfVirtualSourceGeneric_RCA_ab_2.cu -use_fast_math -ccbin ', ...
            '"' cudaSelected '"']);
    cd(initialPath); % Move back top the initial path
end