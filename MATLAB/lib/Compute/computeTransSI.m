function TransSI = computeTransSI(Trans, Resource)

%% TODO Comment and test

%% 
SpeedOfSound                = Resource.Parameters.speedOfSound; % (m/s)
TransSI.frequency           = Trans.frequency*1e6; % (Hz)
TransSI.WaveLength          = SpeedOfSound/TransSI.frequency; % (m)

%% Array Geometry of the probe in SI units (m, s, Hz, kg)
TransSI.bandwidth        = Trans.Bandwidth*1e6; % the lower and upper -6 dB round trip transducer bandwidth cutoff points in MHz (Hz).
TransSI.impulseResponse  = Trans.IR1wy;         % One way Impulse response of the transducer (sampled at 250MHz)

if isfield(Trans, 'lensCorrection')
TransSI.lensCorrection   = Trans.lensCorrection/Trans.frequency*1e-6; %(s) one way propagation through the lens
else
TransSI.lensCorrection   = 0;     
end
%-- [x, y, z, az, el] => [m, m, m, rad, rad]
if strcmp(Trans.units,'mm')
    TransSI.ElementWidth = Trans.elementWidth.*1e-3; % Elements Width (m)
    TransSI.ElementPos   = Trans.ElementPos;
    TransSI.ElementPos(:,1:3) = Trans.ElementPos(:,1:3)*1e-3;    
else
    TransSI.ElementWidth = Trans.elementWidth.*TransSI.WaveLength; % Elements Width (m)
    TransSI.ElementPos   = Trans.ElementPos;
    TransSI.ElementPos(:,1:3) = Trans.ElementPos(:,1:3)*TransSI.WaveLength ;
end

TransSI.spacing          = Trans.spacingMm*1e-3; % spacing between elements (m)  

if isfield(Trans, 'elevationApertureMm')
    TransSI.ElementHeight    = Trans.elevationApertureMm.*1e-3; % Elements Height (m) 
else
    TransSI.ElementHeight    = TransSI.ElementWidth; % Elements Height (m) 
end

if isfield(Trans, 'elevationFocusMm')
    TransSI.ElementElevationFocus = Trans.elevationFocusMm.*1e-3; % Elements Width (m)
else
    TransSI.ElementElevationFocus = 10000;
end

TransSI.baffle          = 0;
end