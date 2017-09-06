function denoise_IEICE_2010_main

%
% This is a demo program of the paper J. Tian, L. Chen and L. Ma, 
% "A wavelet-domain non-parametric statistical approach for image denoising," 
% IEICE Electronics Express, Vol. 7, Sep. 2010, pp. 1409-1415.
%
%
% Note that the PSNR values could be slightly different with that reported 
% in paper due to the random number generator used to generate noisy image.
%
% Acknowledgement: This program needs Wavelet transform toolbox, which is 
% downloaded from http://www-stat.stanford.edu/~wavelab/.

clear all; close all; clc;

% Load the ground truth image
img_truth = double(imread('babara_truth.bmp'));
[nRow, nColumn] = size(img_truth);    

% Use the ground truth image to generate the noisy image
noise_sig_truth = 10; % sigma_n used in the paper. This parameter is adjusted by the user.
noise_mu = 0;
img_noisy = img_truth + randn(size(img_truth)) .* noise_sig_truth + noise_mu;

% Setting of wavelet transform
wbase = 'Daubechies';
mom = 8;
dwt_level = 5; % wavelet decomposition level
[n,J] = func_quadlength(img_truth);
L = J-dwt_level;
win_size=2;

% Perform denoising
img_denoised = func_denoise_kde(img_noisy, wbase, mom, dwt_level, win_size);

% Write the output image
imwrite(uint8(img_denoised),'babara_denoised.bmp', 'bmp');

% Calculate the PSNR performance
fprintf('PSNR=%.2fdB\n', func_psnr_gray(img_truth, img_denoised));

%-------------------------------------------------------------------------
%------------------------------Inner Function ----------------------------
%-------------------------------------------------------------------------
% Proposed image denoising approach
function x_out= func_denoise_kde(x_in, wbase, mom, dwt_level, win_size)

[nRow, nColumn] = size(x_in);
L = log2(size(x_in,2))-dwt_level;

%%%%%%%%%%%%%%%%% estimate the noise_sigma from the noisy signal
qmf = func_MakeONFilter(wbase,mom);
[temp, coef] = func_NormNoise_2d(x_in, qmf);
noise_sigma = 1/coef;

wx  = func_FWT2_PO(x_in, L, qmf);

[n,J] = func_dyadlength(wx);
ws = wx;
for j=(J-1):-1:L
    [t1,t2] = func_dyad2HH(j);    
    ws(t1,t2) = func_subband_denoise(wx(t1,t2), noise_sigma, win_size);
    [t1,t2] = func_dyad2HL(j);
    ws(t1,t2) = func_subband_denoise(wx(t1,t2), noise_sigma, win_size);
    [t1,t2] = func_dyad2LH(j);
    ws(t1,t2) = func_subband_denoise(wx(t1,t2), noise_sigma, win_size);
end
x_out  = func_IWT2_PO(ws, L, qmf);

%-------------------------------------------------------------------------
%------------------------------Inner Function ----------------------------
%-------------------------------------------------------------------------
% Perform denoising for each subband
function result = func_subband_denoise(x_in, noise_sigma, win_size)

AA = padarray(x_in, [win_size win_size], 'replicate', 'bot');
AA = im2col(AA, [win_size*2+1 win_size*2+1],'sliding');

x_in_1d = (x_in(:))';
result_1d = zeros(size(AA));

var_gaussian = mean(AA.^2)-noise_sigma^2;
var_gaussian(var_gaussian<0)=0;
sigma_gaussian = sqrt(var_gaussian);
judge = (sigma_gaussian~=0);

BB = mean(AA);
y = x_in_1d(judge);

for i=1:size(AA,1)
    temp_x = AA(i,:);
    temp_A = temp_x(judge);
    temp_BB = BB(judge);
    temp_sigma = sigma_gaussian(judge);
    temp_result = (y.*temp_sigma.^2+temp_A.*noise_sigma.^2-temp_BB.*noise_sigma.^2) ./ (temp_sigma.^2+noise_sigma.^2);            
    result_1d_temp = zeros(size(result_1d(i,:)));
    result_1d_temp(judge) = temp_result;
    result_1d(i,:) = result_1d_temp;
end
result_1d = mean(result_1d);
result = reshape(result_1d,size(x_in,1),size(x_in,2));

%-------------------------------------------------------------------------
%------------------------------Inner Function ----------------------------
%-------------------------------------------------------------------------
% Calculate the PSNR performance to two images
function result = func_psnr_gray(f, g)

f = double(f);
g = double(g);
Q=255; MSE=0;
[M,N]=size(f);
h = f - g;
MSE = sum(sum(h.*h));
MSE=MSE/M/N;
result=10*log10(Q*Q/MSE);
