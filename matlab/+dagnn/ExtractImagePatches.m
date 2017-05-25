classdef ExtractImagePatches < dagnn.Layer
  properties
    dim = 3
    stride = [1 1]
    rate = [1 1]
    pad = [0 0]
  end

  properties (Transient)
    inputSizes = {}
  end

  methods
    function outputs = forward(obj, inputs, params)
      outputs{1} = vl_nnconcat(inputs, obj.dim) ;
      obj.inputSizes = cellfun(@size, inputs, 'UniformOutput', false) ;
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      derInputs = vl_nnconcat(inputs, obj.dim, derOutputs{1}, 'inputSizes', obj.inputSizes) ;
      derParams = {} ;
    end

    function obj = ExtractImagePatches(varargin)
      obj.load(varargin{:}) ;
    end
  end
end

