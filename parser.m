weightsPath = '~/coding/libs/darknet/models/yolo-voc.weights' ;
modelPath = '~/coding/libs/darknet/cfg/yolo-voc.cfg' ;

% read weights
if 0
  fID = open(weightsPath, 'r') ;
  data = fread(fID) ;
  fclose(fID) ;
end

% read weights layout
if 1
  rows = importdata(modelPath, '\n') ;
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

for ii = 2:numel(rows)
  row = rows{ii} ;

  % skip comments
  if strcmp(row(1), '#') 
    continue 
  end
  %fprintf('number of parsed layers: %d\n', numel(layers)) ;
  fprintf('processing (%d/%d): %s\n', ii, numel(rows), row) ;

  if layerHeader
    disp(layer') ;
    layers{end+1} = layer ;
    layer = {row} ;
    layerHeader = 0 ;
  else
    [startIdx, endIdx] = regexp(row, template) ;
    if ~isempty(startIdx)
      current = row(startIdx:endIdx) ;
      layerHeader = 1 ;
      fprintf('found header: %s\n', current) ;
    else 
      % parse key-value pair
      tokens = strsplit(row, '=') ;
      key = tokens{1} ;
      values = cellfun(@str2num, strsplit(tokens{2}, ','), 'Uni', 0) ;
      layer{end+1} = {key, values} ;
    end
  end
end
    keyboard


if 0
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
end
