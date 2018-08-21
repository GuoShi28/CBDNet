%%% edited by Guo Shi at 2018/8/21
function [perce_response] = PerceptualForward(img)
global net_load;
useGPU      = 1; 
net = dagnn.DagNN.loadobj(net_load) ;
OutputVer = net.getVarIndex('conv3_3x') ;
net.vars(net.getVarIndex('conv3_3x')).precious = 1 ;

net.mode = 'test';
if useGPU
   net.move('gpu');
end       

net.eval({'data',img}) ;
output = gather(squeeze(gather(net.vars(OutputVer).value)));
perce_response = output;

if useGPU
    perce_response = gpuArray(perce_response);
end




