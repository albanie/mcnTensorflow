function setup_mcnTensorflow()
%SETUP_MCNTENSORFLOW Sets up MCNTENSORFLOW, by adding its folders 
% to the Matlab path

  root = fileparts(mfilename('fullpath')) ;
  addpath(root, [root '/matlab']) ;
