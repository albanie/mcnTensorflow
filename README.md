### parser for yolo binary weights

weightsPath = '~/coding/libs/darknet/models/yolo-voc.weights'
modelPath = '~/coding/libs/darknet/cfg/yolo-voc.cfg'

% read weights
if 0
  fID = open(weightsPath, 'r') ;
  data = fread(fID) ;
  fclose(fID) ;
end

% read weights layout
if 0
  cfg = importdata(modelPath) ;
  rows = cfg.textdata ;
end


% layer definitions are split according to square brackets
% the first task is to group rows by layer
assert(strcmp(rows{1}, '[net]'), 'cfg should begin with a net definition') ;
current = 'net' ;
layer = {} ;
layers = {} ;
layerHeader = false ;

% darknet convention for layer definitions
template = '\[([a-zA-Z0-9]*)\]' ;

for ii = 1:numel(rows)
  row = rows{ii} ;
  if layerHeader
    layers{end+1} = layer ;
    layer = {row} ;
    layerHeader = 0 ;
  else
    [startIdx, endIdx] = regexp(row, template) ;
    if ~isempty(start)
      current = row(startIdx:endIdx) ;
    if regexp(ko
    layer{end+1} = row ;


for ii = 1:numel(rows)
  row = rows{ii} ;
  %fprintf('%s\n', rows{ii}) ;
  switch row
    case '[net]'
      fprintf('found net\n') ;
    case '[convolutional]'
      fprintf('found conv\n') ;

    otherwise
      ;
      %fprintf('%s not recognised\n', row) ;
  end
end
