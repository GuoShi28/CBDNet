%%% edited by Guo Shi at 2018/8/21
classdef Perceptual < dagnn.ElementWise
  % Perceptual DagNN perceptual layer
  %   The Perceptual layer output the vgg response of input and store the result
  %   as its only output.

  properties (Transient)
    numInputs
  end

  methods
    function outputs = forward(obj, inputs, params)
       outputs{1} = PerceptualForward(inputs{1}) ;
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      [ derInputs{1}] = PerceptualBackward(inputs{1}, derOutputs{1});
      derParams = {} ;
    end

  end
end
