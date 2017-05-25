# Objects to wrap tensor flow nodes for conversion to matconvnet,
# based on the mcn-caffe importer
# # author: Samuel Albanie 

from collections import OrderedDict
from math import floor, ceil
from operator import mul
import numpy as np
from numpy import array
import pdb
import scipy
import scipy.io
import scipy.misc
import ipdb
import copy
import collections

# --------------------------------------------------------------------
#                  MatConvNet in NumPy (A.V magic from caffe importer)
# --------------------------------------------------------------------

mlayerdt = [('name',object),
            ('type',object),
            ('inputs',object),
            ('outputs',object),
            ('params',object),
            ('block',object)]

mparamdt = [('name',object),
            ('value',object)]

minputdt = [('name',object),
            ('size',object)]

def reorder(aList, order):
    return [aList[i] for i in order]

def row(x):
    return np.array(x,dtype=float).reshape(1,-1)

def rowarray(x):
    return x.reshape(1,-1)

def rowcell(x):
    return np.array(x,dtype=object).reshape(1,-1)

def dictToMatlabStruct(d):
    if not d:
        return np.zeros((0,))
    dt = []
    for x in d.keys():
        pair = (x,object)
        if isinstance(d[x], np.ndarray): pair = (x,type(d[x]))
        dt.append(pair)
    y = np.empty((1,),dtype=dt)
    for x in d.keys():
        y[x][0] = d[x]
    return y

# --------------------------------------------------------------------
#                                              Basic TF Node + helpers
# --------------------------------------------------------------------

class TFNode(object):
    def __init__(self, name, input_names, op, **kwargs):
        self.name = name
        self.input_names = input_names
        self.op = op 
        for key in kwargs:
            setattr(self, key, kwargs[key])
        self.inputs = [] # used later for graph construction

    def summary_str(self, obj_type):
        summary = """TF {} object:
        + name : {} 
        + input_names : {}
        + input_types : {}
        + op : {}"""
        return summary.format(obj_type, self.name, self.input_names, 
                              self.input_types, self.op)

class TfValue(object):
    def __init__(self, name):
        self.name = name
        self.shape = None
        self.value = np.zeros(shape=(0,0), dtype='float32')

    def toMatlab(self):
        mparam = np.empty(shape=[1,], dtype=mparamdt)
        mparam['name'][0] = self.name
        mparam['value'][0] = self.value
        return mparam

class TFModel(object):
    def __init__(self):
        self.layers = OrderedDict()
        self.vars = OrderedDict()
        self.params = OrderedDict()

    def addLayer(self, layer):
        ename = layer.name
        while ename in self.layers.keys():
            ename = ename + 'x'
        if layer.name != ename:
            print('Warning: a layer with name {} was already found, using ',
                  '{} instead'.format(layer.name, ename))
            layer.name = ename

        # add the variables and parameters associated with each layer to 
        # the model, and set their values where applicable
        for v in layer.inputs:  
            self.addVar(v)

        for v in layer.outputs: 
            self.addVar(v)

        for p in layer.params: 
            self.addParam(p, layer)

        self.layers[layer.name] = layer

    def addVar(self, name):
        if name not in self.vars.keys():
            self.vars[name] = TfValue(name)

    def addParam(self, name, layer):
        if name not in self.params.keys():
            self.params[name] = TfValue(name)
            self.params[name].value = layer.param_values[name]

class ParseException(Exception):
    pass

def parse_inputs(input_nodes, ops):
    """
    parse a list of input nodes and return them in the order 
    given by `ops`. `Any` can be specified as a special argument 
    to match any op - it must be supplied as the last argument
    """
    if len(input_nodes) != len(ops):
        raise ParseException('number of inputs did not match ops')

    wildcard = False

    if 'Any' in ops:
        if ops.index('Any') != len(ops) - 1:
            raise ParseException('`Any` must be last arg')
        wildcard = True
        ops.pop()

    out_nodes = []
    for op in ops:
        for node in input_nodes:
            if node not in out_nodes and node.op == op:
                out_nodes.append(node)

    if wildcard:
        for node in input_nodes:
            if node not in out_nodes:
                out_nodes.append(node)

    if len(out_nodes) != len(ops) + wildcard:
        raise ParseException('not enough nodes were matched')
    return out_nodes

# --------------------------------------------------------------------
#                                                             TF Graph
# --------------------------------------------------------------------

class TFGraph(object):
    def __init__(self, node_list):
        self.nodes = node_list

    def print(self):
        for node in self.nodes:
            print(self.node)

    def __repr__(self):
        return 'TensorFlow graph object with {} nodes'.format(len(self.nodes))

# --------------------------------------------------------------------
#                                                   Matconvnet objects
# --------------------------------------------------------------------

def buildMcnLayerName(op, layerNames):
    """ 
    construct matconvnet layer name
    """ 
    num_prev_ops = sum([op in x for x in layerNames])
    name = '{}_{}'.format(op, num_prev_ops + 1)
    return name

class McnNode(object):
    def __init__(self, name, value, op, input_nodes=None):
        self.name = name 
        self.value = value
        self.op = op
        self.input_nodes = input_nodes

class McnLayer(object):
    def __init__(self, name, inputs, outputs):
        self.name = name
        self.inputs = inputs
        self.outputs = outputs
        self.params = []
        self.model = None

    def toMatlab(self):
        mlayer = np.empty(shape=[1,],dtype=mlayerdt)
        mlayer['name'][0] = self.name
        mlayer['type'][0] = None
        mlayer['inputs'][0] = rowcell(self.inputs)
        mlayer['outputs'][0] = rowcell(self.outputs)
        mlayer['params'][0] = rowcell(self.params)
        mlayer['block'][0] = dictToMatlabStruct({})
        return mlayer

class McnConv(McnLayer):
    def __init__(self, name, tf_node, input_nodes, dilation=[1,1]):

        # parse the expression formed by input nodes
        assert len(input_nodes) == 2, 'conv layer expects two nodes as inputs'

        [pad_node, filter_node] = parse_inputs(input_nodes, ['Pad', 'Const'])

        # define input and output variable names
        inputs = [pad_node.name]
        outputs = [name]
        super().__init__(name, inputs, outputs)

        # determine filter dimensions
        self.kernel_size = filter_node.value.shape[:2]
        self.num_out = filter_node.value.shape[3]

        # a bias is often not used in conjuction with batch norm
        self.bias_term = 0 

        # reformat padding to match mcn
        tf_pad = pad_node.value
        param_format = tf_node.data_format
        pad_top_bottom = tf_pad[param_format[0],:]
        pad_left_right = tf_pad[param_format[1],:]
        self.pad = np.hstack((pad_top_bottom, pad_left_right))

        # reformat stride to match mcn
        stride_x = tf_node.stride[param_format[0]]
        stride_y = tf_node.stride[param_format[1]]
        self.stride = np.hstack((stride_x, stride_y))
        self.op = 'Conv2D'
        self.input_nodes = input_nodes

        # check options are correctly formatted
        assert len(self.pad) == 4, ('padding format does hvae the '
         'required number of elements for `[top bottom left right]`')
        assert len(self.stride) == 2, ('stride format does hvae the '
         'expected number of elements for `[strideY strideX]`')
        assert len(self.kernel_size) == 2, ('kernel size should contain '
          'exactly two elements')

        # set dilation 
        # TODO(sam) - handle dilated convs properly
        self.dilation = dilation

        self.filter_depth = filter_node.value.shape[2]
        self.num_output = filter_node.value.shape[3]

        # set param names and store weights on the layer - note that
        # biases may be set later
        filter_name = name + '_filter'
        self.params = [filter_name,]
        self.param_values = {filter_name: filter_node.value}

    def toMatlab(self):
        size = list(self.kernel_size) + [self.filter_depth, self.num_output]
        mlayer = super().toMatlab()
        mlayer['type'][0] = u'dagnn.Conv'
        mlayer['block'][0] = dictToMatlabStruct(
            {'hasBias': self.bias_term,
             'size': row(size),
             'pad': row(self.pad),
             'stride': row(self.stride),
             'dilate': row(self.dilation)})
        return mlayer

class McnReLU(McnLayer):
    def __init__(self, name, tf_node, input_nodes):
        assert len(input_nodes) == 2, 'relu layer expects two nodes as inputs'

        # parse the expressions formed by input nodes
        [mul_node, raw_node] = parse_inputs(input_nodes, ['Mul', 'Any'])
        [leak_node, other_node] = parse_inputs(mul_node.input_nodes, ['Const', 'Any'])

        inputs = raw_node.outputs
        outputs = [name]

        super().__init__(name, inputs, outputs)

        # check for leak
        self.leak = leak_node.value
        self.op = 'relu'

    @staticmethod
    def is_leaky_relu_expression(tf_node, input_nodes):
        """ 
        pattern match to check for leaky relu
        """
        try :
            [mul_node, raw_node] = parse_inputs(input_nodes, ['Mul', 'Any'])
        except ParseException:
            return False 

        #TODO(sam): add in more robust checks
        is_leaky_relu = id(raw_node) in [id(node) for node in mul_node.input_nodes]
        return is_leaky_relu

    def toMatlab(self):
        mlayer = super().toMatlab()
        mlayer['type'][0] = u'dagnn.ReLU'
        mlayer['block'][0] = dictToMatlabStruct({'leak': self.leak })
        return mlayer

class McnConcat(McnLayer):
    def __init__(self, name, tf_node, input_nodes):
        assert len(input_nodes) == 3, 'concat layer expects three nodes as inputs'

        inputs = [node.outputs[0] for node in input_nodes[:2]]
        assert sum([len(node.outputs) for node in input_nodes[:2]]) == 2, 'more outputs not yet supported'
        outputs = [name]
 
        super().__init__(name, inputs, outputs)

        self.axis = tf_node.axis
        self.op = 'concat'

    def toMatlab(self):
        mlayer = super().toMatlab()
        mlayer['type'][0] = u'dagnn.Concat'
        mlayer['block'][0] = dictToMatlabStruct({'dim': float(self.axis) + 1})
        return mlayer

class McnExtractImagePatches(McnLayer):
    def __init__(self, name, tf_node, input_nodes):
        assert len(input_nodes) == 1, 'extract image patches layer expects one node as input'

        # parse inputs
        src_node = input_nodes[0]
        param_format = [1,2,3,0] # this is currently the form of the TF layer, but may change

        inputs = src_node.outputs
        outputs = [name]

        super().__init__(name, inputs, outputs)

        # reformat kernel size to match mcn
        tf_stride = tf_node.stride
        stride_y = tf_stride[param_format[0]]
        stride_x = tf_stride[param_format[1]]
        self.stride = np.hstack((stride_y, stride_x))

        # reformat kernel size to match mcn
        tf_rate = tf_node.rate
        rate_y = tf_rate[param_format[0]]
        rate_x = tf_rate[param_format[1]]
        self.rate = np.hstack((rate_y, rate_x))

        #TODO(sam) fix properly on the first pass
        self.pad = [1, 1, 1, 1]
        self.pad_type = tf_node.pad_type

    def toMatlab(self):
        mlayer = super().toMatlab()
        mlayer['type'][0] = u'dagnn.ExtractImagePatches'
        mlayer['block'][0] = dictToMatlabStruct(
            {'stride': row(self.stride),
             'rate': row(self.rate),
             'pad': row(self.pad)})
        return mlayer

class McnBatchNorm(McnLayer):

    def __init__(self, name, tf_node, input_nodes, eps=1e-5):

        # parse the expressions formed by input nodes
        [bias_node, mul_node] = parse_inputs(input_nodes, ['Const', 'Mul'])
        [gain_node, div_node] = parse_inputs(mul_node.input_nodes, ['Const', 'RealDiv'])
        [var_node, sub_node] = parse_inputs(div_node.input_nodes, ['Const', 'Sub'])
        [mean_node, conv_node] = parse_inputs(sub_node.input_nodes, ['Const', 'Conv2D'])

        # parse inputs
        self.bias_term = bias_node.value
        self.scale_factor = gain_node.value
        self.variance = var_node.value
        self.mean = mean_node.value

        # define input and output variable names
        inputs = conv_node.outputs
        outputs = [name]

        super().__init__(name, inputs, outputs)

        self.eps = eps
        self.op = 'batch_norm'

        # set params
        mean_name = name + u'_mean'
        variance_name = name + u'_variance'
        scale_factor_name = name + u'_scale_factor'
        self.params = [mean_name, variance_name, scale_factor_name]
        self.param_values = {mean_name: self.mean, 
                             variance_name: self.variance, 
                             scale_factor_name: self.scale_factor}

    @staticmethod
    def is_batch_norm_expression(tf_node, input_nodes):
        """ 
        in order to extract batch normalization layers from a tensor flow
        computational graph, we need to be able to match against the set
        of operations which constitute batch norm.  This method performs
        that pattern matching
        """

        try :
            [bias_node, mul_node] = parse_inputs(input_nodes, ['Const', 'Mul'])
        except ParseException:
            return False # unexpected format from batch norm

        # check for required sequence of ops
        expected_div_ops = {'Const', 'RealDiv'}
        div_ops = set([node.op for node in mul_node.input_nodes])

        try :
            [var_node, div_node] = parse_inputs(mul_node.input_nodes, ['Const', 'RealDiv'])
        except ParseException:
            return False # unexpected format from batch norm

        expected_sub_ops = {'Const', 'Sub'}
        sub_ops = set([node.op for node in div_node.input_nodes])
        is_bn = (sub_ops == expected_sub_ops) and (div_ops == expected_div_ops)

        #TODO(sam): add in more robust checks
        return is_bn

    def toMatlab(self):
        mlayer = super().toMatlab()
        mlayer['type'][0] = u'dagnn.BatchNorm'
        mlayer['block'][0] = dictToMatlabStruct(
            {'epsilon': self.eps})
        return mlayer

class McnPooling(McnLayer):

    def __init__(self, name, tf_node, input_nodes, method):
        assert len(input_nodes) == 1, 'pooling layer takes a single input node'

        # parse inputs
        pool_node = input_nodes[0]

        # reformat kernel size to match mcn
        tf_kernel_size = tf_node.ksize
        param_format = tf_node.data_format
        kernel_size_y = tf_kernel_size[param_format[0]]
        kernel_size_x = tf_kernel_size[param_format[1]]
        self.kernel_size = np.hstack((kernel_size_y, kernel_size_x))

        # reformat kernel size to match mcn
        tf_stride = tf_node.stride
        param_format = tf_node.data_format
        stride_y = tf_stride[param_format[0]]
        stride_x = tf_stride[param_format[1]]
        self.stride = np.hstack((stride_y, stride_x))

        #TODO(sam) fix properly on the first pass
        self.pad = [1, 1, 1, 1]

        # define input and output variable names
        inputs = pool_node.outputs
        outputs = [name]

        super().__init__(name, inputs, outputs)

        self.method = method
        self.op = 'pool'

    def toMatlab(self):
        mlayer = super().toMatlab()
        mlayer['type'][0] = u'dagnn.Pooling'
        mlayer['block'][0] = dictToMatlabStruct(
            {'method': self.method,
             'poolSize': row(self.kernel_size),
             'stride': row(self.stride),
             'pad': row(self.pad)})
        return mlayer
