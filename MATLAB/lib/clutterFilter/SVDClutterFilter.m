function [IQf,eig_val] = SVDClutterFilter(IQ,Mask,w,type)

% w Normalized Cutoff Frequency
% Mask is the noise weight in IQ

if min(w)<0 || max(w)>1
   error('w must be is the normalized cutoff frequency [0 1]') 
end

SizIQ   = size(IQ);
NumDim  = numel(SizIQ);

Ncut    = round(w.*SizIQ(end)); 

if isempty(Mask)
    Mask = ones(SizIQ(1:end-1));
end

if nargin<4
type = 'band';
end

IQf	= bsxfun(@times,double(IQ),1./sqrt(Mask));
IQf = reshape(IQf,[],SizIQ(end)); 
[Eig_vect,Eig_val]  = svd(double(IQf'*IQf));
eig_val             = diag(Eig_val); 

switch type
    case 'low'  % LowPassSVDFilter
    Eig_vect        = Eig_vect(:,1:Ncut(1));    
    IQf             = (IQf*Eig_vect)*Eig_vect';   
        
    case 'high' % HighPassFilter    
    Eig_vect        = Eig_vect(:,Ncut(1):end);    
    IQf             = (IQf*Eig_vect)*Eig_vect';    
        
    case 'stop' % StopBandFilter
    Eig_vect        = Eig_vect(:,[1:Ncut(1) , Ncut(2):end]);    
    IQf             = (IQf*Eig_vect)*Eig_vect';    
        
    otherwise   % BandPassFilter 
    Eig_vect        = Eig_vect(:,Ncut(1):Ncut(2));    
    IQf             = (IQf*Eig_vect)*Eig_vect';  
end
IQf             = reshape(IQf,SizIQ);
end