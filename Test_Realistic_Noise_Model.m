%%% Test Code for realistic noise model from 
%%% https://arxiv.org/abs/1807.04686
addpath('./utils');

%% load CRF parameters
load('201_CRF_data.mat');
load('dorfCurvesInv.mat');
I_gl = I;
B_gl = B;
I_inv_gl = invI;
B_inv_gl = invB;

%% load images
Img = im2single(imread('1.png'));

%% Realistic Noise Model:
%%% y = M^{-1}(M(f(L + n(x)))), L = f^{-1}(x) 
%%% x and y are the original clean image and the noisy image we created. 
%%% n(x) = n_s(x) + n_c, 
%%% Var(n_s(x)) = \sigma_s * x, Var(n_c) = \sigma_c

%%% model1: randomlu choose \sigma_s, \sigma_c, CRF and mosaic pattern
% noise = AddNoiseMosai(Img,I_gl,B_gl,I_inv_gl,B_inv_gl);

%%% model2: setting \sigma_s, \sigma_c, CRF and mosaic pattern
sigma_s = 0.08; % recommend 0~0.16
sigma_c = 0.03; % recommend 0~0.06
CRF_index = 5;  % 1~201
pattern = 1;    % 1: 'gbrg', 2: 'grbg', 3: 'bggr', 4: 'rggb', 5: no mosaic

noise = AddNoiseMosai(Img,I_gl,B_gl,I_inv_gl,B_inv_gl, sigma_s, ...
    sigma_c, CRF_index, pattern);
%% JPEG compression
%%% If JPEG compression is not considered, just commented out the following 
%%% Code or just set "quality" equal to 100
qality = 70; % image quality, recommend 60~100
path_temp = fullfile('./', 'jpeg.jpg');
imwrite(double(noise), path_temp, 'jpg','Quality', qality);
noise = im2double(imread(path_temp));

%% display image
imshow(cat(2, Img, noise), 'InitialMagnification', 'fit');