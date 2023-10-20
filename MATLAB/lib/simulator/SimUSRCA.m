function [RF,psim]   = SimUSRCA_V2(Pos,BSC,Delays,Apod,psim)

% Simus from : Shahrokh Shahriari & Garcia Damien PMB 2018
% Meshfree simulations of ultrasound vector flow imaging using smoothed
% particle hydrodynamics

% Pos       = [x,y,z] position of scatter
% BSC       = BackScattering Coefficient
% Delays    = Delay law in transmit
% Apod      = Apodization in transmit
% psim      = struct of invariant parameters for tx and rx

% Simulation of RCA probe
% Updated 21/02/2022

if ~isfield(psim,'c');          psim.c          = 1540;                     end % propagation velocity (m/s)
if ~isfield(psim,'fc');         psim.fc         = 3e6;                      end % Transmit Frequency [Hz]
if ~isfield(psim,'fs');         psim.fs         = 4*psim.fc;                end % Sampling Frequency [Hz]
if ~isfield(psim,'fspulse');    psim.fspulse    = 10*psim.fs;               end % Sampling Frequency [Hz]

if ~isfield(psim,'lbd');        psim.lbd        = psim.c./psim.fc;          end
if ~isfield(psim,'pitch_x');    psim.pitch_x    = psim.lbd;                 end
if ~isfield(psim,'width_x');    psim.width_x    = psim.lbd/2;               end
if ~isfield(psim,'kerf_x');     psim.kerf_x     = psim.pitch_x ...
                                                    - psim.width_x;  end

if ~isfield(psim,'pitch_y');    psim.pitch_y    = psim.lbd;                 end
if ~isfield(psim,'width_y');    psim.width_y    = psim.lbd/2;               end
if ~isfield(psim,'kerf_y');     psim.kerf_y     = psim.pitch_y ...
                                                    - psim.width_y;  end

if ~isfield(psim,'Nele_x');     psim.Nele_x     = 128;                      end
if ~isfield(psim,'Nele_y');     psim.Nele_y     = 128;                      end

if ~isfield(psim,'BW');         psim.BW         = 1;                        end
if ~isfield(psim,'eleBW');      psim.eleBW      = 1;                        end
if ~isfield(psim,'baffle');     psim.baffle     = 0;                        end

if ~isfield(psim,'pulse')
    t           = (0:1/psim.fspulse:2/psim.BW/psim.fc);
    psim.pulse  = sin(2*pi*psim.fc*t).*...
                  hanning(numel(t))';
end
%[0 1 2] 0 = cpu 1 = gpuArray 2 = CUDAKernel
if ~isfield(psim,'gpu');        psim.gpu    = 0;                        end

if ~isfield(psim,'t0');         psim.t0     = 0;                        end 



Ntx     = size(Pos,3);
Pos     = single(Pos); 
BSC     = single(BSC); 
Apod    = single(Apod); 
Delays	= single(Delays); 

% Delays  = Delays + numel(psim.pulse)./psim.fspulse;
Delays  = Delays + psim.t0;

%% BEGIN
% Element position
xele    = single(0:psim.Nele_x-1)*psim.pitch_x - (psim.Nele_x-1)*psim.pitch_x/2;
yele    = single(0:psim.Nele_y-1)*psim.pitch_y - (psim.Nele_y-1)*psim.pitch_y/2;
[xele,yele] = ndgrid(xele,yele);
xele    = xele(:);
yele    = yele(:);

zele    = zeros(size(xele),'single');

dmax    = sqrt((max(xele) - min(xele)).^2 + ...
               (max(yele) - min(yele)).^2 + ...
                max(Pos(:,3)).^2);
            
DelaysMax = max(xele.*sind(30)/psim.c);
Tmax    = 2*dmax/psim.c + DelaysMax; % Maximum round trip time
Nscat   = size(Pos,1);

% Spectrum sampling
Nt      = 8*(ceil(Tmax*psim.fs/8));
t       = single(0:Nt-1)./psim.fs;
fi      = single(linspace(0,psim.fs/2,Nt/2));
df      = mean(diff(fi));
Nf      = numel(fi);

% idfmin  = round(Nf/2 + -Nf*psim.BW/2+1);
% idfmax  = round(Nf/2 + Nf*psim.BW/2);
% 
% idf     = single(max(1,idfmin):min(Nf,idfmax));
idf     = 1:Nf;

wi      = 2*pi*fi(idf);
ki      = wi./psim.c;
wc      = 2*pi*psim.fc;
kc      = wc/psim.c;

% Pulse Spectrum
PULSE   = fft(psim.pulse,ceil(Nt*psim.fspulse/psim.fs));
PULSE   = single(PULSE(1:Nt/2));

% Element Spectrum !!! Warning
if ~isfield(psim,'hele')
    psim.hele    = sin(2*pi*psim.fc*(0:1/psim.fspulse:1/psim.eleBW/psim.fc));
end
HELE    = fft(psim.hele(:),ceil(Nt*psim.fspulse/psim.fs));
HELE    = single(HELE(1:Nt/2));
 
%%
switch psim.gpu
    case 0 % 0 = cpu
        TXRX    = complex(zeros(Nf,psim.Nele_x*psim.Nele_y,'single'));
        r_e     = ([xele(:),yele(:),zele(:)]);
        RF      = zeros(Nt,psim.Nele_x+psim.Nele_y,Ntx,'single');
        
        for itx = 1:Ntx
            
            r_scat  = (Pos(:,:,itx));
            
            %-- Tx distance
            d_tx    = permute(r_scat,[1 3 2]) - ...
                        permute(r_e,[3 1 2]); % Nscatter * Nele * xyz
            r_tx    = sqrt(sum(d_tx.^2,3));  % Nscatter * Nele
            
            sin_tx_x    = d_tx(:,:,1)./r_tx; % Nscatter * Nele
            sin_tx_y    = d_tx(:,:,2)./r_tx; % Nscatter * Nele
            cos_tx      = d_tx(:,:,3)./r_tx; % Nscatter * Nele
            
            %-- Directivity [scat ele ki]
            if psim.baffle == 1
                D_tx	= cos_tx.*sinc(psim.width_x/2/pi*sin_tx_x.*permute(ki(:),[2 3 1]))...
                        .*cos_tx.*sinc(psim.width_y/2/pi*sin_tx_y.*permute(ki(:),[2 3 1]));
            else
                D_tx	= sinc(psim.width_x/2/pi*sin_tx_x.*permute(ki(:),[2 3 1]))...
                        .*sinc(psim.width_y/2/pi*sin_tx_y.*permute(ki(:),[2 3 1]));
            end
            
            %-- Green Function
            D_tx = D_tx.*exp(1j*r_tx.*permute(ki(:),[2 3 1]))./r_tx; % Nscatter * Nele * Nki
            
            %-- Transmit
            s_tx    = permute(Apod(:).*exp(1j.*Delays(:).*wi),[3 1 2]); % 1 * Nele * Nki (Nt/2)
            TX      = permute(sum(D_tx.*s_tx,2),[1 3 2]); % Sum over element  :  Nscatter * Nki 

            %-- Receive
            RX      = BSC(:,itx).*D_tx;  % Nscatter * Nele * Nki 
            
            %-- Sum over scatter
            TXRX(idf,:)     = permute(sum(RX.*permute(TX,[1 3 2]),1),[3 2 1]); % Nki * Nele  
            
            RFi(:,:,:,itx)	= reshape(flipud(imag(ifft(...
                PULSE(:).*HELE(:).*HELE(:).*TXRX,Nt))),Nt,psim.Nele_x,psim.Nele_y);
            
            RF(:,:,itx)  = [squeeze(sum(RFi,3)),squeeze(sum(RFi,2))];
        end

    case 1
        
        HELE    = gpuArray(single(HELE)); % Nki (Nt/2)
        PULSE   = gpuArray(single(PULSE)); % Nki (Nt/2)

        TXRX    = gpuArray(complex(zeros(Nf,psim.Nele_x*psim.Nele_y,'single')));
        r_e     = gpuArray([xele(:),yele(:),zele(:)]);
        RF      = zeros(Nt,psim.Nele_x+psim.Nele_y,Ntx,'single');
        
        s_tx    = permute(Apod(:).*exp(1j.*Delays(:).*wi),[3 1 2]);
        
        for itx = 1:Ntx
            r_scat  = gpuArray(Pos(:,:,itx));
            
            %-- Tx distance
            d_tx    = permute(r_scat,[1 3 2]) - ...
                        permute(r_e,[3 1 2]);
            r_tx    = sqrt(sum(d_tx.^2,3));
            
            sin_tx_x    = d_tx(:,:,1)./r_tx;
            sin_tx_y    = d_tx(:,:,2)./r_tx;
            cos_tx      = d_tx(:,:,3)./r_tx;
            
            %-- Directivity [scat ele ki]
            if psim.baffle == 1
                D_tx	= cos_tx.*sinc(psim.width_x/2/pi*sin_tx_x.*permute(ki(:),[2 3 1]))...
                        .*cos_tx.*sinc(psim.width_y/2/pi*sin_tx_y.*permute(ki(:),[2 3 1]));
            else
                D_tx	= sinc(psim.width_x/2/pi*sin_tx_x.*permute(ki(:),[2 3 1]))...
                        .*sinc(psim.width_y/2/pi*sin_tx_y.*permute(ki(:),[2 3 1]));
            end
            
            %-- Green Function
            D_tx = D_tx.*exp(1j*r_tx.*permute(ki(:),[2 3 1]))./r_tx;
            
            %-- Transmit
            TX      = permute(sum(D_tx.*s_tx,2),[1 3 2]); % Sum over element

            %-- Receive
            RX      = BSC(:,itx).*D_tx;
            
            %-- Sum over scatter
            TXRX(idf,:)     = permute(sum(RX.*permute(TX,[1 3 2]),1),[3 2 1]);
            
            RFi             = reshape(gather(flipud(imag(ifft(...
                PULSE(:).*HELE(:).*HELE(:).*TXRX,Nt)))),Nt,psim.Nele_x,psim.Nele_y);
            
            RF(:,:,itx)  = [squeeze(sum(RFi,3)),squeeze(sum(RFi,2))];
            
        end
end
