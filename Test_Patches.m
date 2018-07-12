format compact;
addpath(fullfile('utilities'));

folderTest  = '.\testsets\';
imageSets   = {'DND_patches','Nam_patches','NC12'}; % testing datasets

setTestCur  = imageSets{3};      % current testing dataset

showResult  = 1;
useGPU      = 1; % CPU or GPU. For single-threaded (ST) CPU computation, use "matlab -singleCompThread" to start matlab.
pauseTime   = 0;

%%% for DND dataset, using 'CBDNet.mat' model,
%%% for Nam dataset and other noisy images with JPEG format, using
%%% 'CBDNet_JPEG.mat' model for considering JPEG compression
if strcmp(setTestCur, 'DND_patches') || strcmp(setTestCur, 'NC12')
    load('.\models\CBDNet.mat');
else
    load('.\models\CBDNet_JPEG.mat');
end

net = dagnn.DagNN.loadobj(net) ;
net.removeLayer('objective1') ;
net.removeLayer('objective2') ;
net.removeLayer('objective_TV') ;
denoiseOutput = net.getVarIndex('prediction') ;
net.vars(net.getVarIndex('prediction')).precious = 1 ;

net.mode = 'test';
if useGPU
   net.move('gpu');
end       

% read images
ext         =  {'*.jpg','*.png','*.bmp'};
filePaths   =  [];
for i = 1 : length(ext)
    filePaths = cat(1,filePaths, dir(fullfile(folderTest,setTestCur,ext{i})));
end

for i = 1:length(filePaths)
    %% read images
    input = imread(fullfile(folderTest,setTestCur,filePaths(i).name));
    [w,h,~]=size(input);
    
    [~,nameCur,extCur] = fileparts(filePaths(i).name);
    input = im2double(input);
    if mod(w,4) ~= 0
        input = cat(1,input, input([w:-1:(w-(4-mod(w,4))+1)],:,:)) ;
    end
    if mod(h,4)~=0
        input = cat(2,input, input(:,[h:-1:(h-(4-mod(h,4))+1)],:)) ;
    end
    input = single(input);
    %tic;
    if useGPU
        input = gpuArray(input);
    end

    net.eval({'input',input}) ;
    output = gather(squeeze(gather(net.vars(denoiseOutput).value)));
   %%%
   if mod(w,4) ~= 0
        input = input(1:w,:,:) ;
        output = output(1:w,:,:) ;
    end
    if mod(h,4)~=0
       input = input(:,1:h,:) ;
       output = output(:,1:h,:) ;
    end
    
    if useGPU
        output = gather(output);
        input  = gather(input);
    end

    if showResult
        imshow(cat(2,im2uint8(input),im2uint8(output)),'InitialMagnification', 'fit');
        drawnow;
        pause(pauseTime)
    end
end




