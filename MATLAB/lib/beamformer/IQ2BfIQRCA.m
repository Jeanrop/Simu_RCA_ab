function [IQbfRCA,mask]=IQ2BfIQRCA(IQ,c,pitch_x,pitch_y,fc,fsIQ,theta,t0,xp,yp,zp,Fnumber,emission)

% IQ = IQ matrix (time, channel, angle, event)x2 row and colum emission
% c = US celerity
% pitch_x = distance between elements along x axis
% pitch_y = distance between elements along y axis
% fsIQ = sampling frequency of IQ signal
% theta = transmit angles
% xp,yp,zp = reconstruction volume
% t0 = time delay adjustment
% Fnumber = z/2a
% Emission 1 or 2 = row or colum + orthogonal reception

% double = double;
[Nt,Nch,Nangle,Ntx]=size(IQ);
Npix=numel(xp);


%% Pixel Positions
rp = gpuArray(double([xp(:),yp(:),zp(:)]));

%% Elements Positions

% 1D array (eg row)
xe_x = (-floor(Nch/2):floor(Nch/2)-1)*pitch_x+pitch_x/2; 
ye_x = zeros(Nch,1); ze_x=zeros(Nch,1);
re_x = gpuArray(double([xe_x(:),ye_x(:),ze_x(:)]));

% perpendicular 1D array (eg column)
xe_y = zeros(Nch,1); ze_y=zeros(Nch,1);
ye_y = (-floor(Nch/2):floor(Nch/2)-1)*pitch_y+pitch_y/2;
re_y = gpuArray(double([xe_y(:),ye_y(:),ze_y(:)]));



%% Delay 

switch emission
    case 1 
        %% Forward Delay %EMISSION X
        TXDelay = (rp(:,3)*cosd(abs(theta))+rp(:,1)*sind(theta)+(Nch/2*pitch_x)*sind(abs(theta)))/c;        
        %Npixel*Nangle        
        %% Backward Delay % RECEPTION Y
        %-- Distance Matrix Pixels to Y-Elements [Npixel,Nelement,x/y/z]
        d_e_p = bsxfun(@minus,permute(rp,[1 3 2]),permute(re_y,[3 1 2]));
        % pixel positions Npixel*1*3 (x/y/z)
        % element positions 1*Nele*3 (x/y/z)
        % d_e_p Npixel*Nele*3 (x/y/z)
        RXDelay = sqrt(d_e_p(:,:,2).^2+d_e_p(:,:,3).^2)/c;
        %  RXDelay Npixel*Nele
        %% Aperture with Fnumber
        A=(abs(rp(:,2)-re_y(:,2)')) < (rp(:,3)./(2*Fnumber)); %aperture

        
    case 2
        %% Forward Delay %EMISSION Y
        TXDelay = (rp(:,3)*cosd(abs(theta))+rp(:,2)*sind(theta)+(Nch/2*pitch_y)*sind(abs(theta)))/c;        
        %Npixel*Nangle        
        %% Backward Delay % RECEPTION X
        %-- Distance Matrix Pixels to X-Elements [Npixel,Nelement,x/y/z]
        d_e_p = bsxfun(@minus,permute(rp,[1 3 2]),permute(re_x,[3 1 2]));
        % pixel positions Npixel*1*3 (x/y/z)
        % element positions 1*Nele*3 (x/y/z)
        % d_e_p Npixel*Nele*3 (x/y/z)
        RXDelay = sqrt(d_e_p(:,:,1).^2+d_e_p(:,:,3).^2)/c;
        %  RXDelay Npixel*Nele
        %% Aperture with Fnumber
        A=(abs(rp(:,1)-re_x(:,1)')) < (rp(:,3)./(2*Fnumber)); %aperture  Npixel*Nele
        
end

%% Upper and Lower time indexes for interpolation
id = (t0+bsxfun(@plus,permute(TXDelay,[1 3 2]),RXDelay))*fsIQ+1;%first value initial time 'zero' in Matlab 
% id Npixel*Nele*Nangle
idd = floor(id); idu = idd+1;
wd = id-idd; wu = idu-id;

%% Phase Rotation
Phi = bsxfun(@minus,2*pi*id*fc/fsIQ,2*pi*rp(:,3)*2*fc/c); % id Npixel*Nele*Nangle
% sinPhi = sin(Phi);
% cosPhi = cos(Phi);

% Deal with boundaries
idu(idu<1)=1; idu(idu>Nt)=Nt;
idd(idd<1)=1; idd(idd>Nt)=Nt;

%% Convert time/channel indexes to Data indexes
% Channel index
ide = repmat(1:Nch,Npix,1,Nangle);
ida = repmat(permute(1:numel(theta),[3 1 2]),Npix,Nch);
idusub = sub2ind([Nt,Nch,Nangle],idu,ide,ida);
iddsub = sub2ind([Nt,Nch,Nangle],idd,ide,ida);


%% Sum over channel indexes 

%Ibf = zeros(Npix,Ntx);
%Qbf = zeros(Npix,Ntx);

mask = gather(sum(A,2));

IQbfRCA = zeros(Npix,Ntx);
for itx = 1:Ntx
%     for i_angle=1:Nangle
%     I = real(IQ(:,:,:,itx));
%     Q = imag(IQ(:,:,:,itx));
% 
%     Ibf(:,itx) = sum(sum( A.*(wd.*(I(idusub).*cosPhi-Q(idusub).*sinPhi)...
%         + wu.*(I(iddsub).*cosPhi-Q(iddsub).*sinPhi)),2),3);
%     
%     Qbf(:,itx) = sum(sum( A.*(wd.*(I(idusub).*sinPhi+Q(idusub).*cosPhi)...
%         + wu.*(I(iddsub).*sinPhi+Q(iddsub).*cosPhi)),2),3);
%    
%     IQbf(:,itx) = complex(Ibf(:,itx),Qbf(:,itx));


%     IQi = double(IQ(:,:,:,itx));
%     IQbfRCA(:,itx) = gather((sum(sum(A.*(wd.*IQi(idusub).*exp(1j*Phi)+wu.*IQi(iddsub).*exp(1j*Phi)),2),3)));
% 

    IQi = gpuArray(double(IQ(:,:,:,itx)));
    IQi = A.*(wd.*IQi(idusub).*exp(1j*Phi)+wu.*IQi(iddsub).*exp(1j*Phi));
    IQi = sum(IQi,2);
    IQi = sum(IQi,3);

%     idu_gpu = gpuArray(idu(:,:,i_angle))+(0:127);
%     idd_gpu = gpuArray(idd(:,:,i_angle))+(0:127);
%     wd_gpu = gpuArray(wd(:,:,i_angle));
%     wu_gpu = gpuArray(wu(:,:,i_angle));
%     IQi = gpuArray(double(IQ(:,:,i_angle,itx)));
%     IQi = A.*(wd_gpu.*IQi(idu_gpu).*exp(1j*Phi(:,:,i_angle))+wu_gpu.*IQi(idd_gpu).*exp(1j*Phi(:,:,i_angle)));
%     IQi = sum(IQi,2);
%         
%     IQbfRCA(:,i_angle,itx) = gather(IQi);
    IQbfRCA(:,itx) = gather(IQi);

%     end
end
end
