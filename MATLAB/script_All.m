clc
clear all
close all

disp('load')
tic
%% Load parameters, flow data
% Working folder
addpath(genpath('/home/jeanrop/Documents/simu_ab/'))

% Verasonics parameters
load('/home/jeanrop/Documents/simu_ab/parameters.mat');

% Microbubbles flow data
load('/home/jeanrop/scratch/Pos_Steph_100/data_stack_1')

data_pos = permute(data, [2 1 3 4]);
data_pos(:,4,:,:) = 1;

toc
%% Save stack of position from Stephen simulation
% small_data = dataset.pos(:,:,1:33600);
% small_data = reshape(small_data,3,500,20,50,672);
% small = reshape(small, 3, 500, 672, 1000);
% 
% path = '/home/jeanrop/Documents/simu_ab/save_data/';
% for i = 1:10
%     data = small(:,:,:,1 + (i-1)*100:i*100);
%     file_path = [path, 'data_stack_', num2str(i)];
%     save(file_path, "data", '-mat')
% end
%% Parameters to change
disp('Parameters')
tic

NX = 80; % Window size (lambda)
NY = 80; % Window size (lambda)
NZ = 60; % Window size (lambda)

decimeRate      = 4;
NbbPerROI       = 375;     % floor(NbbPerFOV*VolumeROI/VolumeFOV);
NframePerBuffer = 16;      % Allow us to choose how many frames are contained into a buffer
coeff_law = 1;

Nlambda2  = 4;
sub2      = 4;
i_bubble  = 1;


NEvent = NframePerBuffer*P.nbAngles;

%% Media
MediaRF.speedOfSound          = Resource.Parameters.speedOfSound;  % Speed-of-sound (m/s)
MediaRF.attenuation           = 0.5;                               % Attenuation dB/cm/MHz

%% Modification frequences
TW.Parameters(1) = Receive(1).demodFrequency;
Trans.frequency  = Receive(1).demodFrequency;

%% Array Geometry of the probe in SI units (m, s, Hz, kg)
% [x, y, z, az, el] => [m, m, m, rad, rad]
% Trans.frequency = Receive(1).demodFrequency;

TransMAT = computeTransSI(Trans, Resource);

% Matrix Array 128*128 from RCA Daxsonics 
xele    = TransMAT.ElementPos(1:128,1);
yele    = TransMAT.ElementPos(128+1:end,2);
[yele,xele] = meshgrid(yele,xele);
TransMAT.ElementPos = [xele(:),yele(:),0*xele(:),zeros(numel(xele),2)];

Trans_X = TransMAT;
% [Nelement , [x y z az al], Subelement]
Trans_X.ElementPos = permute(reshape(TransMAT.ElementPos,128,128,[]),[1,3,2]);

Trans_Y = TransMAT;
% [Nelement , [x y z az al], Subelement]
Trans_Y.ElementPos = permute(reshape(TransMAT.ElementPos,128,128,[]),[2,3,1]);

%% Modification Trans

Trans_Y.ElementPos(:,1,:) = 0;
Trans_X.ElementPos(:,2,:) = 0;
%% Event matrix definition

TX_MAT   = computeTXSI(Trans, TX, TW);  % Apod
Ntx     = numel(TX_MAT);

% X Emission - Y Reception
% Transmit matrix Definition 

TX_RCA_X = TX_MAT(1:Ntx/2);

for itx = 1:Ntx/2
    TX_RCA_X(itx).Apod     = TX_MAT(itx).Apod(1:128);
    TX_RCA_X(itx).Delay    = TX_MAT(itx).Delay(1:128);
end

% Receive matrix Definition 
for itx = Ntx/2:-1:1
    RX_RCA_X(itx).demodFrequency    = Receive(1).demodFrequency*1e6;    % Demodulation frequency (Hz)
    RX_RCA_X(itx).samplingFrequency = (Receive(1).decimSampleRate*1e6); % Sampling frequency of IQ signals (Hz)
    RX_RCA_X(itx).sampleMode        = 'BS100BW';  %'NS200BW'  or  'BS50BW'  or 'BS100BW'             
    RX_RCA_X(itx).startTime         = (2*Receive(itx).startDepth-TW(1).peak)/(Trans.frequency*1e6);  %  Start time (s) : Ajuste le temps liee a la lentille du capteur
    RX_RCA_X(itx).numelSample       = 1024;                             % Nb sample along Fast Time
    RX_RCA_X(itx).Apod              = Receive(itx).Apod(128+(1:128));   % Elements Apodization in Receive   
    RX_RCA_X(itx).InputFilter       = Receive(itx).InputFilter;
end

% Y Emission - X Reception
% Transmit matrix Definition 
TX_RCA_Y = TX_MAT(Ntx/2 + (1:Ntx/2));

for itx = 1:Ntx/2
    TX_RCA_Y(itx).Apod     = TX_MAT(itx+Ntx/2).Apod(128+(1:128));
    TX_RCA_Y(itx).Delay    = TX_MAT(itx+Ntx/2).Delay(128+(1:128));
end

% Receive matrix Definition 
RX_RCA_Y = RX_RCA_X;
for itx = Ntx/2:-1:1                           % Nb sample along Fast Time
    RX_RCA_Y(itx).startTime         = (2*Receive(itx+Ntx/2).startDepth-TW(1).peak)/(Trans.frequency*1e6);
    RX_RCA_Y(itx).Apod              = Receive(itx+Ntx/2).Apod(1:128);   % Elements Apodization in Receive
    RX_RCA_Y(itx).Delay             = Receive(itx+Ntx/2).Apod(1:128);   % Elements Apodization in Receive
    RX_RCA_Y(itx).InputFilter       = Receive(itx+Ntx/2).InputFilter;
end

%% Beamforming parameters
Param.samplingFrequency     = (Receive(1).decimSampleRate*1e6)/(2*Receive(1).quadDecim);   % Sampling frequency of IQ signals (Hz)
Param.demodulationFrequency = Receive(1).demodFrequency*1e6;    % Demodulation frequency (Hz)
Param.speedOfSound          = Resource.Parameters.speedOfSound; % Speed-of-sound (m/s)
Param.startTime             = (-2*Receive(1).startDepth)/(Trans.frequency*1e6);% Start time (s) / +TW(1).peak
Param.fnumber               = 1;
Param.wavelength            = Param.speedOfSound/(Trans.frequency*1e6);%  Param.demodulationFrequency ;

%% Shift the position to center at [0 0 Zmin+rangeZ/2]
clear PosX PosY

i_simu = 1; 

Pos = data_pos(1:NbbPerROI,:,1:NframePerBuffer*P.nbAngles,i_simu);
Pos = Pos*0.001;

rangeX	   = NX*Param.wavelength;  %To modify
rangeY	   = NY*Param.wavelength;  %To modify
rangeZ	   = NZ*Param.wavelength;  %To modify

Zmin        = 25*Param.wavelength;

maxi = max(Pos, [], [1 3]); % A voir selon
mini = min(Pos, [], [1 3]);

shift = (mini + maxi)/2;

Pos(:,1,:) = Pos(:,1,:)-shift(1);
Pos(:,2,:) = Pos(:,2,:)-shift(2);
Pos(:,3,:) = Pos(:,3,:)-shift(3) + Zmin + rangeZ/2;

% 1 Bulle au milieu
% Pos = Pos_load(1,:,:);
% Pos(:,1,:) = 0;
% Pos(:,2,:) = 0;
% Pos(:,3,:) = Zmin + rangeZ/2;
% Pos(:,4,:) = 1;

% Pos = repmat(Pos(:,:,1), [1 1 42]);

for i=1:NframePerBuffer
    % No movement between angles
    Pos_frames(:,:,i) = Pos(:,:,1+(i-1)*P.nbAngles);
    Pos(:,:,1+(i-1)*P.nbAngles:i*P.nbAngles) = repmat(Pos_frames(:,:,i),1,1,P.nbAngles);  
    Pos_bubble (:,i)  = Pos_frames(i_bubble,:,i);
    % The simulator altern between VH and HV configuration
    PosX(:,:,(i-1)*21+1:i*21) = Pos(:,:,2*(i-1)*21+1:(2*i-1)*21); 
    PosY(:,:,(i-1)*21+1:i*21) = Pos(:,:,1+(2*i-1)*21:(2*i)*21);
end
%% Define reconstruction beamformer grid
clear grid grid2

grid.subx   = 1;    % number of samples per pitch along x-axis
grid.suby   = 1;    % number of samples per pitch along y-axis
grid.subz   = 1;    % number of samples per wavelength along z-axis

grid.dx     = Param.wavelength/grid.subx;   % x-pixel size (m)
grid.dy     = Param.wavelength/grid.suby;   % y-pixel size (m)
grid.dz     = Param.wavelength/grid.subz;   % z-pixel size (m)
grid.Nx     = NX*grid.subx-1;        % number of pixels along x-axis ++
grid.Ny     = NY*grid.suby-1;        % number of pixels along y-axis ++
grid.Nz     = NZ*grid.subz-1;        % number of pixels along z-axis
grid.Zmin   = (P.startDepth+15)*Param.wavelength;     % Start Depth [m]
grid.Xmin   = -(grid.Nx+1)*grid.dx/2;           % Left corner [m]
grid.Ymin   = -(grid.Ny+1)*grid.dy/2;           % Left corner [m]

grid.x      = (0:grid.Nx).*grid.dx + grid.Xmin;   % x values of the reconstruction grid [m] ++
grid.y      = (0:grid.Ny).*grid.dy + grid.Ymin;   % y values of the reconstruction grid [m] ++
grid.z      = (0:grid.Nz-1).*grid.dz + grid.Zmin; % z values of the reconstruction grid [m]

[grid.X,grid.Z,grid.Y] = meshgrid(grid.x,grid.z,grid.y); % Careful with convention image

%  Define reconstruction grid to align RF
for iFrame = 1:NframePerBuffer
    grid2(iFrame).sub     = sub2;          % To modify
    grid2(iFrame).Nlambda = Nlambda2;           % To modify

    grid2(iFrame).dx    = Param.wavelength/grid2(iFrame).sub ;  % x-pixel size (m)
    grid2(iFrame).dy    = Param.wavelength/grid2(iFrame).sub ;  % y-pixel size (m)
    grid2(iFrame).dz    = Param.wavelength/grid2(iFrame).sub ;  % z-pixel size (m)

    grid2(iFrame).Nx    = 1;
    grid2(iFrame).Ny    = 1;
    grid2(iFrame).Nz    = grid2(iFrame).sub*grid2(iFrame).Nlambda + 1;

    grid2(iFrame).Xmin  = Pos_bubble(1,iFrame);
    grid2(iFrame).Ymin  = Pos_bubble(2,iFrame);
    grid2(iFrame).Zmin  = Pos_bubble(3,iFrame) - grid2(iFrame).Nlambda*Param.wavelength/2;


    grid2(iFrame).x     = (1:grid2(iFrame).Nx).*grid2(iFrame).dx + grid2(iFrame).Xmin;
    grid2(iFrame).y     = (1:grid2(iFrame).Ny).*grid2(iFrame).dy + grid2(iFrame).Ymin;
    grid2(iFrame).z     = (1:grid2(iFrame).Nz).*grid2(iFrame).dz + grid2(iFrame).Zmin;

    [grid2(iFrame).X,grid2(iFrame).Z,grid2(iFrame).Y] = meshgrid(grid2(iFrame).x,grid2(iFrame).z,grid2(iFrame).y);
end
%% DEFINE GEOMETRY (3D)
% Y-EMISSION AND X-RECEPTION + BF

GeometryX.gridBfX     =  grid.X;         % x-grid for beamforming (m, 3D meshgrid)
GeometryX.gridBfY     =  grid.Y;         % y-grid for beamforming (m, 3D meshgrid)
GeometryX.gridBfZ     =  grid.Z;         % z-grid for beamforming (m, 3D meshgrid)

% Element position along x-axis and aperture
GeometryX.posEleX     =  Trans_X.ElementPos(:,1,1);  % position of the elements along x-axis (m, vector)
GeometryX.posEleY     =  Trans_X.ElementPos(:,2,1);  % position of the elements along y-axis (m, vector)
GeometryX.posEleZ     =  Trans_X.ElementPos(:,3,1);  % position of the elements along z-axis (m, vector)

GeometryX.apertureX   =  max(GeometryX.posEleX)-min(GeometryX.posEleX);  % aperture along x-axis (m)
GeometryX.apertureY   =  max(GeometryX.posEleY)-min(GeometryX.posEleY);  % aperture along x-axis (m)
GeometryX.apertureZ   =  max(GeometryX.posEleZ)-min(GeometryX.posEleZ);  % aperture along x-axis (m)

% Define TransmitX scheme Y-emission X-reception (angles along y P.naEl angles)
for i =1: P.naEl
    TransmitX.thetaX(i) = TX(i+P.naAz).Steer(1);   % tilt angles along x-axis
    TransmitX.thetaY(i) = TX(i+P.naAz).Steer(2);   % tilt angles along y-axis
end
TransmitX.flagReceiveX = true; % flag for receiving along x-elements

% Options
Options.flagCompounding = true;     % do the compounding if true
Options.flagMaskTransmit = false;   % BF only transmit non zero (could have displaying issu if log(0) ! )
Options.decreaseFactorGPU = 2;      % decrease factor for the GPU memory (integer, default = 1)


% X-EMISSION AND Y-RECEPTION + BF

% Element position along y-axis and aperture
GeometryY             =  GeometryX;
GeometryY.posEleX     =  Trans_Y.ElementPos(:,1,1);  % position of the elements along x-axis (m, vector)
GeometryY.posEleY     =  Trans_Y.ElementPos(:,2,1);  % position of the elements along y-axis (m, vector)
GeometryY.posEleZ     =  Trans_Y.ElementPos(:,3,1);  % position of the elements along z-axis (m, vector)

GeometryY.apertureY   =  max(GeometryY.posEleY)-min(GeometryY.posEleY);%max(GeometryY.posEleY)-min(GeometryY.posEleY);  % aperture along x-axis (m)
GeometryY.apertureX   =  max(GeometryY.posEleX)-min(GeometryY.posEleX);  % aperture along x-axis (m)
GeometryY.apertureZ   =  max(GeometryY.posEleZ)-min(GeometryY.posEleZ);  % aperture along x-axis (m)

% Define TransmitY scheme  X-emission Y-reception (angles along x P.naAz angles)
for idx =1: P.naEl
    TransmitY.thetaX(idx) = TX(idx).Steer(1);   % tilt angles along x-axis
    TransmitY.thetaY(idx) = TX(idx).Steer(2);   % tilt angles along y-axis
end
TransmitY.flagReceiveX = false; % flag for receiving along x-elements

% Geometry ROI

for iFrame = 1:NframePerBuffer
    GeometryX2(iFrame) = GeometryX;
    GeometryX2(iFrame).gridBfX = grid2(iFrame).X;
    GeometryX2(iFrame).gridBfY = grid2(iFrame).Y;
    GeometryX2(iFrame).gridBfZ = grid2(iFrame).Z;
end

GeometryY2             =  GeometryX2;
for iFrame = 1:NframePerBuffer
    GeometryY2(iFrame).posEleX     =  Trans_Y.ElementPos(:,1,1);  % position of the elements along x-axis (m, vector)
    GeometryY2(iFrame).posEleY     =  Trans_Y.ElementPos(:,2,1);  % position of the elements along y-axis (m, vector)
    GeometryY2(iFrame).posEleZ     =  Trans_Y.ElementPos(:,3,1);  % position of the elements along z-axis (m, vector)

    GeometryY2(iFrame).apertureY   =  max(GeometryY2(iFrame).posEleY)-min(GeometryY2(iFrame).posEleY);  % aperture along x-axis (m)
    GeometryY2(iFrame).apertureX   =  max(GeometryY2(iFrame).posEleX)-min(GeometryY2(iFrame).posEleX);  % aperture along x-axis (m)
    GeometryY2(iFrame).apertureZ   =  max(GeometryY2(iFrame).posEleZ)-min(GeometryY2(iFrame).posEleZ);  % aperture along x-axis (m)
end

%% Loi aberration

phase_max = 0.5;
amp_min   = 0.5;
amp_max   = 1;

N_Ele   = 10;
N_repet = 130/N_Ele;

%  Emission X / Reception Y
tau_x = unifrnd(-phase_max,phase_max,1,N_Ele);
tau_x = interp(tau_x, N_repet)./Param.samplingFrequency;

amp_x = unifrnd(amp_min,amp_max,1,N_Ele);
amp_x = interp(amp_x, N_repet);

%  Emission Y / Reception X
tau_y = unifrnd(-phase_max,phase_max,1,N_Ele);
tau_y = interp(tau_y, N_repet)./Param.samplingFrequency;

amp_y = unifrnd(amp_min,amp_max,1,N_Ele);
amp_y = interp(amp_y, N_repet);

% Application of the law
AbLawX.Delay = tau_x(1:128)*coeff_law;
AbLawX.Apod  = ones(1,128); %amp_x.'

AbLawY.Delay = tau_y(1:128)*coeff_law;
AbLawY.Apod  = ones(1,128); %amp_y.'
toc

% Amp aberration
AmpAb_x = repmat(AbLawX.Apod,128,1,P.naAz);
AmpAb_y = repmat(AbLawY.Apod.',1,128,P.naAz);
AmpAb = cat(3,AmpAb_x,AmpAb_y);

% Phase aberration
DelayAb_x = repmat(AbLawX.Delay,128,1,P.naAz);
DelayAb_y = repmat(AbLawY.Delay.',1,128,P.naAz);
DelaysAb = cat(3,DelayAb_x,DelayAb_y);

%% Simulator 
clear RF RFab

%Emission ligne, Reception colonne
Options.decreaseFactorGPU = 1;
g = gpuDevice(1);

tic
disp('Simu RCA - Emission X, Réception Y')
MediaRF.scattererPos = PosX;
[RF_X, RF_X_ab] = SimUS_CUDARCA_ab(MediaRF, Trans_X, Trans_Y, TX_RCA_X, RX_RCA_X, AbLawX, Options);

RF_X_ab = reshape(RF_X_ab, size(RF_X_ab,1), 128, Ntx/2, []);
RF_X = reshape(RF_X, size(RF_X,1), 128, Ntx/2, []);


disp('Simu RCA - Emission Y, Réception X')
MediaRF.scattererPos = PosY;
[RF_Y, RF_Y_ab] = SimUS_CUDARCA_ab(MediaRF, Trans_Y, Trans_X, TX_RCA_Y, RX_RCA_Y, AbLawY, Options);

RF_Y_ab = reshape(RF_Y_ab, size(RF_Y_ab,1), 128, Ntx/2, []);
RF_Y = reshape(RF_Y, size(RF_Y,1), 128, Ntx/2, []);

RF_X     = RF_X./max(abs(RF_X(:))).*32768;
RF_Y     = RF_Y./max(abs(RF_Y(:))).*32768;
RF = cat(2, RF_X, RF_Y);

RF_X_ab  = RF_X_ab./max(abs(RF_X_ab(:))).*32768;
RF_Y_ab  = RF_Y_ab./max(abs(RF_Y_ab(:))).*32768;
RF_ab = cat(2, RF_X_ab, RF_Y_ab);

toc

%% Beamforming
warning('off')

% Normal
disp('Beamforming - Emission X, Réception Y')
[dataBfX,maskX] = bfVirtualSourceGeneric_RCA(...
                     	reshape(RF_X,size(RF_X,1),128,1,[]),...
                     	GeometryX, TransmitX, Param, Options);
dataBfX        = squeeze(dataBfX);

disp('Beamforming - Emission Y, Réception X')
[dataBfY,maskY] = bfVirtualSourceGeneric_RCA(...
                     	reshape(RF_Y,size(RF_Y,1),128,1,[]),...
                     	GeometryY, TransmitY, Param, Options);
dataBfY        = squeeze(dataBfY);

dataBf = dataBfX + dataBfY;

% Aberre
disp('Beamforming - Emission X, Réception Y')
[dataBfXab,maskXab] = bfVirtualSourceGeneric_RCA(...
                     	reshape(RF_X_ab,size(RF_X_ab,1),128,1,[]),...
                     	GeometryX, TransmitX, Param, Options);
dataBfXab        = squeeze(dataBfXab);


disp('Beamforming - Emission Y, Réception X')
[dataBfYab,maskYab] = bfVirtualSourceGeneric_RCA(...
                     	reshape(RF_Y_ab,size(RF_Y_ab,1),128,1,[]),...
                     	GeometryY, TransmitY, Param, Options);
dataBfYab        = squeeze(dataBfYab);

dataBf_ab = dataBfXab + dataBfYab;



% Aberre Corrige
Param.AbLawRx = AbLawX.Delay;
Param.AmpAbRx = AbLawX.Apod;
disp('Beamforming - Emission X, Réception Y')
[dataBfXab_c,maskXab_c] = bfVirtualSourceGeneric_RCA_ab_1(...
                     	reshape(RF_X_ab,size(RF_X_ab,1),128,1,[]),...
                     	GeometryX, TransmitX, Param, Options);
dataBfXab_c        = squeeze(dataBfXab_c);

Param.AbLawRx = AbLawY.Delay;
Param.AmpAbRx = AbLawY.Apod;
disp('Beamforming - Emission Y, Réception X')
[dataBfYab_c,maskYab_c] = bfVirtualSourceGeneric_RCA_ab_1(...
                     	reshape(RF_Y_ab,size(RF_Y_ab,1),128,1,[]),...
                     	GeometryY, TransmitY, Param, Options);
dataBfYab_c        = squeeze(dataBfYab_c);

dataBf_ab_c = dataBfXab_c + dataBfYab_c;

%% test 

[dataBfX,maskX] = bfVirtualSourceGeneric_RCA_Nosum(...
    reshape(RF_X,size(RF_X,1),128,1,[]),...
    GeometryX2(iFrame), TransmitX, Param, Options);
dataBfX_al_1        = squeeze(dataBfX);

[dataBfY,maskY] = bfVirtualSourceGeneric_RCA_Nosum(...
    reshape(RF_Y,size(RF_Y,1),128,1,[]),...
    GeometryY2(iFrame), TransmitY, Param, Options);
dataBfY_al_1        = squeeze(dataBfY);


%% Alignement
clear dataBfX_al dataBfY_al dataBfX_al_ab dataBfY_al_ab dataBf_al_ab dataBf_al dataBfX dataBfY

for iFrame = 1:16
    iTransmit = iShot*iFrame;
    disp('Normal')
    [dataBfX,maskX] = bfVirtualSourceGeneric_RCA_Nosum(...
        reshape(RF_X(:,:,:,iFrame),size(RF_X,1),128,1,[]),...
        GeometryX2(iTransmit), TransmitX, Param, Options);
    dataBfX_al(:,:,:,iFrame)        = squeeze(dataBfX);

    [dataBfY,maskY] = bfVirtualSourceGeneric_RCA_Nosum(...
        reshape(RF_Y(:,:,:,iFrame),size(RF_Y,1),128,1,[]),...
        GeometryY2(iTransmit), TransmitY, Param, Options);
    dataBfY_al(:,:,:,iFrame)        = squeeze(dataBfY);

    dataBf_al(:,:,:,iFrame) = cat(3,dataBfX_al(:,:,:,iFrame), dataBfY_al(:,:,:,iFrame));

    disp('Aberre corrige')
    Param.AbLawRx = AbLawX.Delay;
    [dataBfXab_c,maskXab] = bfVirtualSourceGeneric_RCA_Nosum_ab(...
        reshape(RF_X_ab(:,:,:,iFrame),size(RF_X_ab,1),128,1,[]),...
        GeometryX2(iTransmit), TransmitX, Param, Options);
    dataBfX_al_ab_c(:,:,:,iFrame)        = squeeze(dataBfXab_c);

    Param.AbLawRx = AbLawY.Delay;
    [dataBfYab_c,maskYab] = bfVirtualSourceGeneric_RCA_Nosum_ab(...
        reshape(RF_Y_ab(:,:,:,iFrame),size(RF_Y_ab,1),128,1,[]),...
        GeometryY2(iTransmit), TransmitY, Param, Options);
    dataBfY_al_ab_c(:,:,:,iFrame)        = squeeze(dataBfYab_c);

    dataBf_al_ab_c(:,:,:,iFrame) = cat(3,dataBfX_al_ab_c(:,:,:,iFrame), dataBfY_al_ab_c(:,:,:,iFrame));


    disp('Aberre')
    [dataBfXab,maskXab] = bfVirtualSourceGeneric_RCA_Nosum(...
        reshape(RF_X_ab(:,:,:,iFrame),size(RF_X_ab,1),128,1,[]),...
        GeometryX2(iTransmit), TransmitX, Param, Options);
    dataBfX_al_ab(:,:,:,iFrame)        = squeeze(dataBfXab);

    [dataBfYab,maskYab] = bfVirtualSourceGeneric_RCA_Nosum(...
        reshape(RF_Y_ab(:,:,:,iFrame),size(RF_Y_ab,1),128,1,[]),...
        GeometryY2(iTransmit), TransmitY, Param, Options);
    dataBfY_al_ab(:,:,:,iFrame)        = squeeze(dataBfYab);

    dataBf_al_ab(:,:,:,iFrame) = cat(3,dataBfX_al_ab(:,:,:,iFrame), dataBfY_al_ab(:,:,:,iFrame));
end
%% Plot
% figure(1)
% plot(AbLawY.Delay*Param.samplingFrequency)
% xlim([1,128])
% ylabel('Sample Time (T)','FontSize',18)
% xlabel('Element Y','FontSize',18)
% title('Delay law','FontSize',18)

%% Visua RF
% figure(2)
sgtitle('RF Transmit X, Receive Y')

RF_X = RF_ab(:,1:128,1,1);
%subplot 211
imagesc(20*log10(rescale(abs(complex(RF_X(1:2:end,:,1), RF_X(2:2:end,:,1))))))
caxis([-40 0])
%colormap gray
ylabel('Sample Time')
xlabel('Element Y')
colorbar
title('Reference')

subplot 212
imagesc(20*log10(rescale(abs(complex(RF_X_ab(1:2:end,:,1), RF_X_ab(2:2:end,:,1))))))
caxis([-40 0])
%colormap gray
ylabel('Sample Time')
xlabel('Element Y')
colorbar
title('No correction')
drawnow

%% Visua BF

miNi = round(min(Pos_frames,[],1)/Param.wavelength);

x_plan = round(Pos_bubble(1, 1)/Param.wavelength) - miNi(1) + 1;
y_plan = round(Pos_bubble(2, 1)/Param.wavelength) - miNi(2) + 1;
z_plan = round(Pos_bubble(3, 1)/Param.wavelength) - miNi(3) + 1;

figure(3);
colormap gray
sgtitle('Along the X and Z axis')

subplot 133
dataBF = dataBf;
imagesc(grid.y*1000, grid.z*1000, 20*log10(squeeze(abs(dataBF(:,y_plan,:,1))./max(abs(dataBF(:))))));
axis image tight equal
hold on,
plot(Pos_bubble(1,1)*1000,Pos_bubble(3,1)*1000,'ro')  
title('Reference')
ylabel('Depth (mm)')
xlabel('Length (mm)')
axis([min(grid.x*1000) max(grid.x*1000) min(grid.z*1000) max(grid.z*1000)])
caxis([-40 0])

subplot 131
dataBF = dataBf_ab;
imagesc(grid.y*1000, grid.z*1000, 20*log10(squeeze(abs(dataBF(:,y_plan,:,1))./max(abs(dataBF(:))))));
axis image tight equal
hold on,
plot(Pos_bubble(1,1)*1000,Pos_bubble(3,1)*1000,'ro')    
title('No correction')
ylabel('Depth (mm)')
xlabel('Length (mm)')
axis([min(grid.x*1000) max(grid.x*1000) min(grid.z*1000) max(grid.z*1000)])
caxis([-40 0])

subplot 132
%dataBF = dataBf_ab_c;
imagesc(grid.y*1000, grid.z*1000, 20*log10(squeeze(abs(dataBF(:,y_plan,:,1))./max(abs(dataBF(:))))));
axis image tight equal
% plot(Pos_bubble(1,1)*1000,Pos_bubble(3,1)*1000,'ro')    
title('With correction')
ylabel('Depth (mm)')
xlabel('Length (mm)')
axis([min(grid.x*1000) max(grid.x*1000) min(grid.z*1000) max(grid.z*1000)])
caxis([-40 0])

figure(4);
colormap gray
sgtitle('Along the Y and Z axis ')

subplot 133
dataBF = dataBf;
imagesc(grid.y*1000, grid.z*1000, 20*log10(squeeze(abs(dataBF(:,:,x_plan,1))./max(abs(dataBF(:))))));
axis image tight equal
hold on,
plot(Pos_bubble(2,1)*1000,Pos_bubble(3,1)*1000,'ro')    
title('Reference')
ylabel('Depth (mm)')
xlabel('Length (mm)')
axis([min(grid.x*1000) max(grid.x*1000) min(grid.z*1000) max(grid.z*1000)])
caxis([-40 0])

subplot 131
dataBF = dataBf_ab;
imagesc(grid.y*1000, grid.z*1000, 20*log10(squeeze(abs(dataBF(:,:,x_plan,1))./max(abs(dataBF(:))))));
axis image tight equal
hold on,
plot(Pos_bubble(2,1)*1000,Pos_bubble(3,1)*1000,'ro')    
title('No correction')
ylabel('Depth (mm)')
xlabel('Length (mm)')
axis([min(grid.x*1000) max(grid.x*1000) min(grid.z*1000) max(grid.z*1000)])
caxis([-40 0])

subplot 132
dataBF = dataBf_ab_c;
imagesc(grid.y*1000, grid.z*1000, 20*log10(squeeze(abs(dataBF(:,:,x_plan,1))./max(abs(dataBF(:))))));
axis image tight equal
%plot(Pos_bubble(2,1)*1000,Pos_bubble(3,1)*1000,'ro')    
title('With correction')
ylabel('Depth (mm)')
xlabel('Length (mm)')
axis([min(grid.x*1000) max(grid.x*1000) min(grid.z*1000) max(grid.z*1000)])
caxis([-40 0])

figure(5);
colormap gray
sgtitle('Along the X and Y axis ')

subplot 133
dataBF = dataBf;
imagesc(grid.x*1000, grid.y*1000, 20*log10(squeeze(abs(dataBF(z_plan,:,:,1))./max(abs(dataBF(:))))));
axis image tight equal
hold on,
plot(Pos_bubble(1,1)*1000,Pos_bubble(2,1)*1000,'ro')   
title('Reference')
ylabel('Depth (mm)')
xlabel('Length (mm)')
axis([min(grid.x*1000) max(grid.x*1000) min(grid.y*1000) max(grid.y*1000)])
caxis([-40 0])

subplot 131
dataBF = dataBf_ab;
imagesc(grid.x*1000, grid.y*1000, 20*log10(squeeze(abs(dataBF(z_plan,:,:,1))./max(abs(dataBF(:))))));
axis image tight equal
hold on,
plot(Pos_bubble(1,1)*1000,Pos_bubble(2,1)*1000,'ro')  
title('No correction')
ylabel('Depth (mm)')
xlabel('Length (mm)')
axis([min(grid.x*1000) max(grid.x*1000) min(grid.y*1000) max(grid.y*1000)])
caxis([-40 0])

subplot 132
dataBF = dataBf_ab_c;
imagesc(grid.x*1000, grid.y*1000, 20*log10(squeeze(abs(dataBF(z_plan,:,:,1))./max(abs(dataBF(:))))));
axis image tight equal
%plot(Pos_bubble(1,1)*1000,Pos_bubble(2,1)*1000,'ro')    
title('With correction')
ylabel('Depth (mm)')
xlabel('Length (mm)')
axis([min(grid.x*1000) max(grid.x*1000) min(grid.y*1000) max(grid.y*1000)])
caxis([-40 0])

%% Visua RF aligne

iAngle = 1;
iFrame = 1;

figure(6)
subplot 311
dataBF_al = dataBfX_al;
imagesc((1:128), linspace(-2,2,17), ...
    20*log10(squeeze(abs(dataBF_al(:,iAngle,:,iFrame))./max(abs(dataBF_al(:))))));
ylabel('Sample Time (T)')
xlabel('Element Y')
title('Reference')
caxis([-40 0])

subplot 312
dataBF_al = dataBfX_al_ab;
imagesc((1:128), linspace(-4,2,17), ...
    20*log10(squeeze(abs(dataBF_al(:,iAngle,:,iFrame))./max(abs(dataBF_al(:))))));
ylabel('Sample Time (T)')
xlabel('Element Y')
title('No correction')
caxis([-40 0])

subplot 313
dataBF_al = dataBfX_al_ab_c;
imagesc((1:128), linspace(-4,2,17), ...
    20*log10(squeeze(abs(dataBF_al(:,iAngle,:,iFrame))./max(abs(dataBF_al(:))))));
ylabel('Sample Time (T)')
xlabel('Element Y')
title('Correction')
caxis([-40 0])

%% Test save mat

Path_save = ('/home/jeanrop/scratch/10_job_100_simu');

% data BF
save([Path_save, '/dataBf_al.mat'], 'dataBf_al', "-v7.3", "-nocompression")
save([Path_save, '/dataBf_al_ab.mat'], 'dataBf_al_ab', "-v7.3", "-nocompression" )
save([Path_save, '/dataBf_al_ab_c.mat'], 'dataBf_al_ab_c', "-v7.3", "-nocompression")

save([Path_save, '/DelaysAb.mat'], 'DelaysAb', "-v7.3", "-nocompression")
save([Path_save, '/dataBf_al_ab.mat'], 'dataBf_al_ab', "-v7.3", "-nocompression" )
save([Path_save, '/dataBf_al_ab_c.mat'], 'dataBf_al_ab_c', "-v7.3", "-nocompression")




%% Visua 3d
% g = gpuDevice(1);
% 
% volumeViewer(20*log10(squeeze(abs(dataBf_ab_c(:,:,:))./max(abs(dataBf_ab_c(:)))+eps)))