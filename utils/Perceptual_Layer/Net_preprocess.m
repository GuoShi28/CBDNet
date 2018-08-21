%%% edited by Guo Shi at 2018/8/21
function [net] = Net_preprocess()
net = load('fast-rcnn-vgg16-pascal07-dagnn.mat');
net = dagnn.DagNN.loadobj(net) ;

%%% the output of "conv3_3x" is used for perceptual loss
%%% layers after "conv3_3x" are removed
l_remove = 17;
Layer_names = cell(1,22);
for in = l_remove: 38
    Layer_names{in} = net.layers(in).name; 
end

for in = l_remove: 38
    net.removeLayer(Layer_names{in}) ;
end

% net = net;


