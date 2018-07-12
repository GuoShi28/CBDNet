format compact;
addpath(fullfile('utilities'));

folderTest  = '.\testsets\';
imageSets   = {'DND','Nam'};     % testing datasets

setTestCur  = imageSets{1};      % current testing dataset

showResult  = 1;
useGPU      = 1; % CPU or GPU. For single-threaded (ST) CPU computation, use "matlab -singleCompThread" to start matlab.
pauseTime   = 0;
Patch_size = 512;

%%% for DND dataset, using 'CBDNet.mat' model,
%%% for Nam dataset and other noisy images with JPEG format, using
%%% 'CBDNet_JPEG.mat' model for considering JPEG compression
if strcmp(setTestCur, 'DND')
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
    input = im2double(input);
     [w,h,~]=size(input);
     denoising_image = zeros(size(input),'single');
     
    w_num = ceil(w / Patch_size);
    h_num = ceil(h / Patch_size);
    for w_index = 1: w_num
        for h_index = 1: h_num
            start_x = 1 + (w_index - 1)*Patch_size;
            end_x = w_index * Patch_size;
            if end_x > w
                end_x = w;
            end
            start_y = 1 + (h_index - 1)*Patch_size;
            end_y = h_index * Patch_size;
            if end_y > h
                end_y = h;
            end
            image_patch = input(start_x:end_x,start_y:end_y,:);
            [wp,hp,~] = size(image_patch);
            if mod(wp,4) ~= 0
                image_patch = cat(1,image_patch, image_patch([wp:-1:(wp-(4-mod(wp,4))+1)],:,:)) ;
            end
            if mod(hp,4)~=0
                image_patch = cat(2,image_patch, image_patch(:,[hp:-1:(hp-(4-mod(hp,4))+1)],:)) ;
            end
            image_patch = single(image_patch);
            %tic;
            if useGPU
                image_patch = gpuArray(image_patch);
            end
    
            %% set noise level map
            net.eval({'input',image_patch}) ;
    
            output = gather(squeeze(gather(net.vars(denoiseOutput).value)));
            if mod(wp,4) ~= 0
                output = output(1:wp,:,:) ;
            end
            if mod(hp,4)~=0
                output = output(:,1:hp,:) ;
            end
    
            if useGPU
                output = gather(output);
            end
            denoising_image(start_x:end_x,start_y:end_y,:) = output;
        end
    end

    if showResult
        imshow(cat(2,im2uint8(input),im2uint8(denoising_image)),'InitialMagnification', 'fit');
        drawnow;
        pause(pauseTime)
    end
end




