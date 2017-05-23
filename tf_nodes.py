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
        return summary.format(common, self.stride, self.pad_type, 
                                    self.ksize, self.data_format)

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

class McnParam(object):

    def __init__(self, name, value):
        self.name = name 
        self.value = value

class McnInput(object):

    def __init__(self, name):
        self.name = name 

class McnLayer(object):
    def __init__(self, name, inputs, outputs):
        self.name = name
        self.inputs = inputs
        self.outputs = outputs
        self.params = []
        self.model = None

    def reshape(self, model):
        pass

    # def display(self):
        # print "Layer \'{}\'".format(self.name)
        # print "  +- type: %s" % (self.__class__.__name__)
        # print "  +- inputs: %s" % (self.inputs,)
        # print "  +- outputs: %s" % (self.outputs,)
        # print "  +- params: %s" % (self.params,)

    # def getTransforms(self, model):
        # transforms = []
        # for i in enumerate(self.inputs):
            # row = []
            # for j in enumerate(self.outputs):
                # row.append(CaffeTransform([1.,1.], [1.,1.], [1.,1.]))
            # transforms.append(row)
        # return transforms

    def transpose(self, model):
        raise NotImplementedError

    def setBlob(self, model, i, blob):
        raise NotImplementedError

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
    def __init__(self, name, inputs, outputs,
                 num_output,
                 bias_term,
                 pad,
                 kernel_size,
                 stride,
                 dilation,
                 group):

        super(CaffeConv, self).__init__(name, inputs, outputs)

        if len(kernel_size) == 1 : kernel_size = kernel_size * 2
        if len(stride) == 1 : stride = stride * 2
        if len(dilation) == 1 : dilation = dilation * 2
        if len(pad) == 1 : pad = pad * 4
        elif len(pad) == 2 : pad = [pad[0], pad[0], pad[1], pad[1]]

        self.num_output = num_output
        self.bias_term = bias_term
        self.pad = pad
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.group = group

        self.params = [name + '_filter']
        if bias_term: self.params.append(name + '_bias')
        self.filter_depth = None

    # def display(self):
        # super(CaffeConv, self).display()
        # print "  +- filter dimension:", self.filter_depth
        # print "  c- num_output (num filters): %s" % self.num_output
        # print "  c- bias_term: %s" % self.bias_term
        # print "  c- pad: %s" % (self.pad,)
        # print "  c- kernel_size: %s" % self.kernel_size
        # print "  c- stride: %s" % (self.stride,)
        # print "  c- dilation: %s" % (self.dilation,)
        # print "  c- group: %s" % (self.group,)

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

    # def setBlob(self, model, i, blob):
        # assert(i < 2)
        # if i == 0:
            # assert(blob.shape[0] == self.kernel_size[0])
            # assert(blob.shape[1] == self.kernel_size[1])
            # assert(blob.shape[3] == self.num_output)
            # self.filter_depth = blob.shape[2]
        # elif i == 1:
            # assert(blob.shape[0] == self.num_output)
        # model.params[self.params[i]].value = blob
        # model.params[self.params[i]].shape = blob.shape

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

