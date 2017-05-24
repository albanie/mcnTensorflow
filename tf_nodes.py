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
#                                                       Basic TF Nodes
# --------------------------------------------------------------------

class TFNode(object):
    def __init__(self, name, input_names, input_types):
        self.name = name
        self.input_names = input_names
        self.input_types = input_types
        self.inputs = [] # used to store node references

    def summary_str(self, obj_type):
        summary = """TF {} object:
        + name : {} 
        + input_names : {}
        + input_types : {}"""
        return summary.format(obj_type, self.name, self.input_names, self.input_types)
        

class TFPlaceHolder(TFNode):
    def __init__(self, *args):
        super().__init__(*args)

    def __repr__(self):
        return super().summary_str('PlaceHolder')

class TFPad(TFNode):
    def __init__(self, *args):
        super().__init__(*args)

    def __repr__(self):
        return super().summary_str('Pad')

class TFSub(TFNode):
    def __init__(self, *args):
        super().__init__(*args)

    def __repr__(self):
        return super().summary_str('Sub')

class TFRealDiv(TFNode):
    def __init__(self, *args):
        super().__init__(*args)

    def __repr__(self):
        return super().summary_str('RealDiv')

class TFMul(TFNode):
    def __init__(self, *args):
        super().__init__(*args)

    def __repr__(self):
        return super().summary_str('Mul')

class TFMaximum(TFNode):
    def __init__(self, *args):
        super().__init__(*args)

    def __repr__(self):
        return super().summary_str('Maximum')

class TFIdentity(TFNode):
    def __init__(self, *args):
        super().__init__(*args)

    def toMatlab(self):
        return None

    def __repr__(self):
        return super().summary_str('Identity')

class TFNoOp(TFNode):
    def __init__(self, *args):
        super().__init__(*args)

    def __repr__(self):
        return super().summary_str('NoOp')


class TFConst(TFNode):
    def __init__(self, name, input_names, input_types, shape, value):
        super().__init__(name, input_names, input_types)
        self.shape = shape
        self.value = value

    def __repr__(self):
        common = super().summary_str('Const')
        summary = """{}
        + shape : {}"""
        return summary.format(common, self.shape)

class TFConcatV2(TFNode):
    def __init__(self, name, input_names, input_types, axis):
        super().__init__(name, input_names, input_types)
        self.axis = axis

    def __repr__(self):
        common = super().summary_str('ConcatV2')
        summary = """{}
        + axis : {}"""
        return summary.format(common, self.axis)

class TFBiasAdd(TFNode):
    def __init__(self, name, input_names, input_types, data_format):
        super().__init__(name, input_names, input_types)
        self.data_format = data_format 

    def __repr__(self):
        common = super().summary_str('BiasAdd')
        summary = """{}
        + data_format : {}"""
        return summary.format(common, self.data_format)

class TFMaxPool(TFNode):
    def __init__(self, name, input_names, stride, pad_type, ksize, input_types, 
                                                                  data_format):
        super().__init__(name, input_names, input_types)
        self.stride = stride
        self.ksize = ksize
        self.pad_type = pad_type
        self.data_format = data_format

    def __repr__(self):
        common = super().summary_str('MaxPool')
        summary = """{}
        + stride : {}
        + ksize : {}
        + pad_type : {}
        + data_format : {}"""
        return summary.format(common, self.stride, self.ksize, self.pad_type, 
                                                             self.data_format)

class TFConv2D(TFNode):
    def __init__(self, name, input_names, stride, pad_type, input_types, 
                                                                 data_format):
        super().__init__(name, input_names, input_types)
        self.stride = stride
        self.pad_type = pad_type
        self.data_format = data_format

    def __repr__(self):
        common = super().summary_str('Conv2D')
        summary = """{}
        + stride : {}
        + pad_type : {}
        + data_format : {}"""
        return summary.format(common, self.stride, self.pad_type, 
                                                             self.data_format)

class TFExtractImagePatches(TFNode):
    def __init__(self, name, input_names, rate, stride, pad_type, ksize, 
                                                                  input_types):
        super().__init__(name, input_names, input_types)
        self.rate = rate
        self.stride = stride
        self.ksize = ksize
        self.pad_type = pad_type

    def __repr__(self):
        common = super().summary_str('Const')
        summary = """{}
        + rate : {}
        + stride : {}
        + ksize : {}
        + pad_type : {}"""
        return summary.format(common, self.rate, self.stride, self.ksize, 
                                                            self.pad_type)

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
class McnNode(object):

    def __init__(self, name, value, op):
        self.name = name 
        self.value = value
        self.op = None

class McnLayer(object):
    def __init__(self, name, inputs, outputs):
        self.name = name
        self.inputs = inputs
        self.outputs = outputs
        self.params = []
        self.model = None

    def reshape(self, model):
        pass

    def transpose(self, model):
        raise NotImplementedError

    def setBlob(self, model, i, blob):
        raise NotImplementedError

    def display(self):
        print('Layer \'{}\''.format(self.name))
        print('  +- type: {}'.format(self.__class__.__name__))
        print('  +- inputs: {}'.format(self.inputs,))
        print('  +- outputs: %s'.format(self.outputs,))
        print('  +- params: %s'.format(self.params,))

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
    def __init__(self, name, tf_conv_node, node_inputs, dilation=[1,1]):

        # parse the expression formed by input nodes
        assert len(node_inputs) == 2, 'conv layer expects two nodes as inputs'

        # parse inputs
        padding_expr = node_inputs[0]
        filters = node_inputs[1]

        # define input and output variable names
        inputs = padding_expr[0]
        outputs = name

        super().__init__(name, inputs, outputs)

        # determine filter dimensions
        self.kernel_size = filters.value.shape[:2]
        self.num_out = filters.value.shape[3]

        # a bias is often not used in conjuction with batch norm
        self.bias_term = 0 

        # reformat padding to match mcn
        tf_pad = padding_expr[1]
        param_format = tf_conv_node.data_format
        pad_top_bottom = tf_pad[param_format[0],:]
        pad_left_right = tf_pad[param_format[1],:]
        self.pad = np.hstack((pad_top_bottom, pad_left_right))

        # reformat stride to match mcn
        stride_x = tf_conv_node.stride[param_format[0]]
        stride_y = tf_conv_node.stride[param_format[1]]
        self.stride = np.hstack((stride_x, stride_y))

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

        # set param names and store weights on the layer - note that
        # params are not set on the shared model until all the layers
        # have been constructed
        self.params = [name + '_filter']
        if self.bias_term: self.params.append(name + '_bias')
        self.filters = filters
        self.filter_depth = None # this is set dynamically

    def display(self):
        super().display()
        print("  +- filter dimension:", self.filter_depth)
        print("  c- num_output (num filters): %s" % self.num_output)
        print("  c- bias_term: %s" % self.bias_term)
        print("  c- pad: %s" % (self.pad,))
        print("  c- kernel_size: %s" % self.kernel_size)
        print("  c- stride: %s" % (self.stride,))
        print("  c- dilation: %s" % (self.dilation,))
        print("  c- group: %s" % (self.group,))

    # def reshape(self, model):
        # varin = model.vars[self.inputs[0]]
        # varout = model.vars[self.outputs[0]]
        # if not varin.shape: return
        # varout.shape = getFilterOutputSize(varin.shape[0:2],
                                           # self.kernel_size,
                                           # self.stride,
                                           # self.pad) \
                                           # + [self.num_output, varin.shape[3]]
        # self.filter_depth = varin.shape[2] / self.group

    # def getTransforms(self, model):
        # return [[getFilterTransform(self.kernel_size, self.stride, self.pad)]]

    def setParams(self, filters, data_format):
        # TODO(sam): implement
        ipdb.set_trace()
        model.params[self.params[i]].value = blob
        model.params[self.params[i]].shape = blob.shape

    # def transpose(self, model):
        # self.kernel_size = reorder(self.kernel_size, [1,0])
        # self.stride = reorder(self.stride, [1,0])
        # self.pad = reorder(self.pad, [2,3,0,1])
        # self.dilation = reorder(self.dilation, [1,0])
        # if model.params[self.params[0]].hasValue():
            # print "Layer %s: transposing filters" % self.name
            # param = model.params[self.params[0]]
            # param.value = param.value.transpose([1,0,2,3])
            # if model.vars[self.inputs[0]].bgrInput:
                # print "Layer %s: BGR to RGB conversion" % self.name
                # param.value = param.value[:,:,: : -1,:]

    # def toMatlab(self):
        # size = self.kernel_size + [self.filter_depth, self.num_output]
        # mlayer = super(CaffeConv, self).toMatlab()
        # mlayer['type'][0] = u'dagnn.Conv'
        # mlayer['block'][0] = dictToMatlabStruct(
            # {'hasBias': self.bias_term,
             # 'size': row(size),
             # 'pad': row(self.pad),
             # 'stride': row(self.stride),
             # 'dilate': row(self.dilation)})
        # return mlayer

    # def toMatlabSimpleNN(self):
        # size = self.kernel_size + [self.filter_depth, self.num_output]
        # mlayer = super(CaffeConv, self).toMatlabSimpleNN()
        # mlayer['type'] = u'conv'
        # mlayer['weights'] = np.empty([1,len(self.params)], dtype=np.object)
        # mlayer['size'] = row(size)
        # mlayer['pad'] = row(self.pad)
        # mlayer['stride'] = row(self.stride)
        # mlayer['dilate'] = row(self.dilation)
        # for p, name in enumerate(self.params):
            # mlayer['weights'][0,p] = self.model.params[name].toMatlabSimpleNN()
        # return mlayer


class McnBatchNorm(McnLayer):

    def __init__(self, name, tf_conv_node, node_inputs, eps=1e-5):
    # def __init__(self, name, inputs, outputs, use_global_stats, moving_average_fraction, eps):
        # parse the expression formed by input nodes
        assert len(node_inputs) == 2, 'batch norm layer expects two nodes as inputs'

        # parse inputs
        bias_expr = node_inputs
        self.bias_term = node_inputs[1].value

        # parse gain expression
        gain_expr = bias_expr[0]
        assert gain_expr[2] == 'mul', 'bn expression does not multiply by gain'
        self.scale_factor = gain_expr[1].value
        
        # parse var expression
        var_expr = gain_expr[0]
        assert var_expr[2] == 'div', 'bn expression does not divide by variance'
        self.variance = var_expr[1].value

        # parse mean expression
        mean_expr = var_expr[0]
        assert mean_expr[2] == 'sub', 'bn expression does not subtract the mean'
        self.mean = mean_expr[1].value

        # parse conv expression
        conv_expr = mean_expr[0]
        assert conv_expr[1] == 'conv', 'bn expression does not follow a convolution'
        self.prev_conv = conv_expr[0]

        # define input and output variable names
        inputs = self.prev_conv.outputs
        outputs = name

        super().__init__(name, inputs, outputs)

        self.eps = eps
        self.params = [name + u'_mean',
                       name + u'_variance',
                       name + u'_scale_factor']

    @staticmethod
    def is_batch_norm_expression(tf_node, node_inputs):
        """ 
        in order to extract batch normalization layers from a tensor flow
        computational graph, we need to be able to match against the set
        of operations which constitute batch norm.  This method performs
        that pattern matching
        """
        # parse inputs
        expr = node_inputs[0]
        bias = node_inputs[1]
   
        # check for required sequence of ops
        is_bn = (expr[-1] == 'mul') and \
                (expr[0][-1] == 'div') and \
                (expr[0][0][-1] == 'sub')

        #TODO(sam): add in more robust checks
        return is_bn


    def display(self):
        super(CaffeBatchNorm, self).display()
        print("  c- use_global_stats: %s" % (self.use_global_stats,))
        print("  c- moving_average_fraction: %s" % (self.moving_average_fraction,))
        print("  c- eps: %s" % (self.eps))

    def setBlob(self, model, i, blob):
        assert(i < 3)
        model.params[self.params[i]].value = blob
        model.params[self.params[i]].shape = blob.shape

    def reshape(self, model):
        shape = model.vars[self.inputs[0]].shape
        mean = model.params[self.params[0]].value
        variance = model.params[self.params[1]].value
        scale_factor = model.params[self.params[2]].value
        for i in range(3): del model.params[self.params[i]]
        self.params = [self.name + u'_mult',
                       self.name + u'_bias',
                       self.name + u'_moments']

        model.addParam(self.params[0])
        model.addParam(self.params[1])
        model.addParam(self.params[2])

        if shape:
            mult = np.ones((shape[2],),dtype='float32')
            bias = np.zeros((shape[2],),dtype='float32')
            model.params[self.params[0]].value = mult
            model.params[self.params[0]].shape = mult.shape
            model.params[self.params[1]].value = bias
            model.params[self.params[1]].shape = bias.shape

        if mean.size:
            moments = np.concatenate(
                (mean.reshape(-1,1) / scale_factor,
                 np.sqrt(variance.reshape(-1,1) / scale_factor + self.eps)),
                axis=1)
            model.params[self.params[2]].value = moments
            model.params[self.params[2]].shape = moments.shape

        model.vars[self.outputs[0]].shape = shape

    def toMatlab(self):
        mlayer = super(CaffeBatchNorm, self).toMatlab()
        mlayer['type'][0] = u'dagnn.BatchNorm'
        mlayer['block'][0] = dictToMatlabStruct(
            {'epsilon': self.eps})
        return mlayer

# class McnRelu(McnLayer):
    # def __init__(self, name, tf_conv_node, node_inputs):
        # super().__init__(name, inputs, outputs)

    # def toMatlab(self):
        # mlayer = super(CaffeReLU, self).toMatlab()
        # mlayer['type'][0] = u'dagnn.ReLU'
        # mlayer['block'][0] = dictToMatlabStruct(
            # {'leak': float(0.0) })
        # # todo: leak factor
        # return mlayer

class McnReLU(McnLayer):
    def __init__(self, name, tf_node, node_inputs):
        assert len(node_inputs) == 2, 'relu layer expects two nodes as inputs'

        # parse inputs
        leak_expr = node_inputs[0]
        raw_expr = node_inputs[1]

        inputs = raw_expr[0].outputs
        outputs = name

        super().__init__(name, inputs, outputs)

        # check for leak
        assert leak_expr[2] == 'mul', 'leak expression does not multiply by factor'
        leak = leak_expr[0].value
        self.leak = leak

    @staticmethod
    def is_leaky_relu_expression(tf_node, node_inputs):
        """ 
        pattern match to check for leaky relu
        """
        # parse inputs
        x = node_inputs[1]
        leak_expr = node_inputs[0]

        #TODO(sam): add in more robust checks

        # check for leaky relu inputs
        match = (hash(leak_expr[0]) == hash(x)) or (hash(leak_expr[1]) == hash(x))
        is_leaky_relu = (leak_expr[-1] == 'mul') and match
        return is_leaky_relu

    def toMatlab(self):
        mlayer = super().toMatlab()
        mlayer['type'][0] = u'dagnn.ReLU'
        mlayer['block'][0] = dictToMatlabStruct({'leak': self.leak })
        return mlayer

class McnBatchNorm(McnLayer):

    def __init__(self, name, tf_conv_node, node_inputs, eps=1e-5):
    # def __init__(self, name, inputs, outputs, use_global_stats, moving_average_fraction, eps):
        # parse the expression formed by input nodes
        assert len(node_inputs) == 2, 'batch norm layer expects two nodes as inputs'

        # parse inputs
        bias_expr = node_inputs
        self.bias_term = node_inputs[1].value

        # parse gain expression
        gain_expr = bias_expr[0]
        assert gain_expr[2] == 'mul', 'bn expression does not multiply by gain'
        self.scale_factor = gain_expr[1].value
        
        # parse var expression
        var_expr = gain_expr[0]
        assert var_expr[2] == 'div', 'bn expression does not divide by variance'
        self.variance = var_expr[1].value

        # parse mean expression
        mean_expr = var_expr[0]
        assert mean_expr[2] == 'sub', 'bn expression does not subtract the mean'
        self.mean = mean_expr[1].value

        # parse conv expression
        conv_expr = mean_expr[0]
        assert conv_expr[1] == 'conv', 'bn expression does not follow a convolution'
        self.prev_conv = conv_expr[0]

        # define input and output variable names
        inputs = self.prev_conv.outputs
        outputs = name

        super().__init__(name, inputs, outputs)

        self.eps = eps
        self.params = [name + u'_mean',
                       name + u'_variance',
                       name + u'_scale_factor']

    @staticmethod
    def is_batch_norm_expression(tf_node, node_inputs):
        """ 
        in order to extract batch normalization layers from a tensor flow
        computational graph, we need to be able to match against the set
        of operations which constitute batch norm.  This method performs
        that pattern matching
        """
        # parse inputs
        expr = node_inputs[0]
        bias = node_inputs[1]
   
        # check for required sequence of ops
        is_bn = (expr[-1] == 'mul') and \
                (expr[0][-1] == 'div') and \
                (expr[0][0][-1] == 'sub')

        #TODO(sam): add in more robust checks
        return is_bn


    def display(self):
        super(CaffeBatchNorm, self).display()
        print("  c- use_global_stats: %s" % (self.use_global_stats,))
        print("  c- moving_average_fraction: %s" % (self.moving_average_fraction,))
        print("  c- eps: %s" % (self.eps))

    def setBlob(self, model, i, blob):
        assert(i < 3)
        model.params[self.params[i]].value = blob
        model.params[self.params[i]].shape = blob.shape

    def reshape(self, model):
        shape = model.vars[self.inputs[0]].shape
        mean = model.params[self.params[0]].value
        variance = model.params[self.params[1]].value
        scale_factor = model.params[self.params[2]].value
        for i in range(3): del model.params[self.params[i]]
        self.params = [self.name + u'_mult',
                       self.name + u'_bias',
                       self.name + u'_moments']

        model.addParam(self.params[0])
        model.addParam(self.params[1])
        model.addParam(self.params[2])

        if shape:
            mult = np.ones((shape[2],),dtype='float32')
            bias = np.zeros((shape[2],),dtype='float32')
            model.params[self.params[0]].value = mult
            model.params[self.params[0]].shape = mult.shape
            model.params[self.params[1]].value = bias
            model.params[self.params[1]].shape = bias.shape

        if mean.size:
            moments = np.concatenate(
                (mean.reshape(-1,1) / scale_factor,
                 np.sqrt(variance.reshape(-1,1) / scale_factor + self.eps)),
                axis=1)
            model.params[self.params[2]].value = moments
            model.params[self.params[2]].shape = moments.shape

        model.vars[self.outputs[0]].shape = shape

    def toMatlab(self):
        mlayer = super(CaffeBatchNorm, self).toMatlab()
        mlayer['type'][0] = u'dagnn.BatchNorm'
        mlayer['block'][0] = dictToMatlabStruct(
            {'epsilon': self.eps})
        return mlayer

    def toMatlabSimpleNN(self):
        mlayer = super(CaffeBatchNorm, self).toMatlabSimpleNN()
        mlayer['type'] = u'bnorm'
        mlayer['epsilon'] = self.eps
        return mlayer

class McnPooling(McnLayer):

    def __init__(self, name, tf_node, node_inputs, method):


        # parse inputs
        pool_expr = node_inputs
        assert len(pool_expr) == 1, 'pooling layer takes a single input node'

        # reformat kernel size to match mcn
        tf_kernel_size = tf_node.ksize
        param_format = tf_node.data_format
        kernel_size_y = tf_kernel_size[param_format[0]]
        kernel_size_x = tf_kernel_size[param_format[1]]
        self.kernel_size = np.hstack((kernel_size_y, kernel_size_x))

        # reformat kernel size to match mcn
        tf_stride = tf_node.ksize
        param_format = tf_node.data_format
        stride_y = tf_stride[param_format[0]]
        stride_x = tf_stride[param_format[1]]
        self.stride = np.hstack((stride_y, stride_x))

        #TODO(sam) fix properly on the first pass
        self.pad = [1, 1, 1, 1]

        # define input and output variable names
        inputs = pool_expr[0][0].outputs
        outputs = name

        super().__init__(name, inputs, outputs)

        self.method = method

    def display(self):
        super(CaffePooling, self).display()
        print("  c- method: ".format(self.method))
        print("  c- pad: {}".format(self.pad))
        print("  c- kernel_size: {}".format(self.kernel_size))
        print("  c- stride: {}".format(self.stride))

    def reshape(self, model):
        shape = model.vars[self.inputs[0]].shape
        if not shape: return
        # MatConvNet uses a slighly different definition of padding, which we think
        # is the correct one (it corresponds to the filters)
        self.pad_corrected = copy.deepcopy(self.pad)
        for i in [0, 1]:
            self.pad_corrected[1 + i*2] = min(
                self.pad[1 + i*2] + self.stride[i] - 1,
                self.kernel_size[i] - 1)
        model.vars[self.outputs[0]].shape = \
            getFilterOutputSize(shape[0:2],
                                self.kernel_size,
                                self.stride,
                                self.pad_corrected) + shape[2:5]

    def getTransforms(self, model):
        return [[getFilterTransform(self.kernel_size, self.stride, self.pad)]]

    def transpose(self, model):
        self.kernel_size = reorder(self.kernel_size, [1,0])
        self.stride = reorder(self.stride, [1,0])
        self.pad = reorder(self.pad, [2,3,0,1])
        if self.pad_corrected:
            self.pad_corrected = reorder(self.pad_corrected, [2,3,0,1])

    def toMatlab(self):
        mlayer = super(CaffePooling, self).toMatlab()
        mlayer['type'][0] = u'dagnn.Pooling'
        mlayer['block'][0] = dictToMatlabStruct(
            {'method': self.method,
             'poolSize': row(self.kernel_size),
             'stride': row(self.stride),
             'pad': row(self.pad_corrected)})
        if not self.pad_corrected:
            print(('Warning: pad correction for layer {} could not be ',
                 ('computed because the layer input shape could not be ', 
                  'determined').format(self.name)))
        return mlayer
