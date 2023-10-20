function [outputArg1,outputArg2] = gaussian_noise(RF, RFab, Nscat)

noise_coeff = squeeze( sqrt(sum(RF.^2) ./ sum(abs(RF) > 0.1*max(RF)) ) );
noise = randn(RF.shape);

RF   = RF + noise*noise_coeff;
RFab = RFab + noise*noise_coeff;

end

