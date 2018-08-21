%%% edited by Guo Shi at 2018/8/21
function [perce_response_der] = PerceptualBackward(img, back_der)
global net_load;
useGPU      = 1; 
net = dagnn.DagNN.loadobj(net_load) ;

opts.gradientClipping = false; %%% set 'true' to prevent exploding gradients in the beginning.
opts.backPropDepth    = inf;

InputDer = net.getVarIndex('data') ;
net.vars(net.getVarIndex('data')).precious = 1 ;
net.conserveMemory = false;

if useGPU
   net.move('gpu');
end    


opts.derOutputs       = {'conv3_3x', back_der} ;
net.eval({'data', img}, opts.derOutputs , 'holdOn', 0) ;

in_der = gather(squeeze(gather(net.vars(InputDer).der)));
perce_response_der = in_der;




if useGPU
    perce_response_der = gpuArray(perce_response_der);
end




