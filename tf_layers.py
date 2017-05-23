# Objects to wrap tensor flow nodes for conversion to matconvnet,
# based on the mcn-caffe importer
#
# author: Samuel Albanie 

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
#                                                         Basic Nodes
# --------------------------------------------------------------------

class TFNode(object):
    def __init__(self, name, inputs):
        self.name = name
        self.inputs = inputs

class TFPlaceHolder(TFNode):
    def __init__(self, name, inputs):
        self.name = name
        self.inputs = inputs

    def __repr__(self):
        summary = """TF PlaceHolder object:
        + name : {} 
        + inputs : {}"""
        return summary.format(self.name, self.inputs)

class TFConst(TFNode):
    def __init__(self, name, inputs, shape, value):
        self.name = name
        self.inputs = inputs
        self.shape = shape
        self.value = value

    def __repr__(self):
        summary = """TF Const object:
        + name : {} 
        + inputs : {}
        + shape : {}
        + mean value: {}"""
        return summary.format(self.name, self.inputs, self.shape, 
                              np.mean(self.value))

class TFPad(TFNode):
    def __init__(self, name, inputs, input_types):
        self.name = name
        self.inputs = inputs
        self.input_types = input_types

    def __repr__(self):
        summary = """TF PlaceHolder object:
        + name : {} 
        + inputs : {}
        + input_types : {}"""
        return summary.format(self.name, self.inputs, self.input_types)

class TFSub(TFNode):
    def __init__(self, name, inputs, input_types):
        self.name = name
        self.inputs = inputs
        self.input_types = input_types

    def __repr__(self):
        summary = """TF Sub object:
        + name : {} 
        + inputs : {}
        + input_types : {}"""
        return summary.format(self.name, self.inputs, self.input_types)

class TFConv2D(TFNode):
    def __init__(self, name, inputs, stride, pad_type, data_format, input_types):
        self.name = name
        self.inputs = inputs
        self.stride = stride
        self.pad_type = pad_type
        self.data_format = data_format
        self.input_types = input_types

    def __repr__(self):
        summary = """TF Conv2D object:
        + name : {} 
        + inputs : {}
        + stride : {}
        + pad_type : {}
        + data_format : {}
        + input_types : {}
        """
        return summary.format(self.name, self.inputs, self.stride, 
                self.pad_type, self.data_format, self.input_types)

# --------------------------------------------------------------------
#                                                             TF Graph
# --------------------------------------------------------------------

class TFGraph(object):
    def __init__(self):
        self.nodes = OrderedDict()

    def __repr__(self):
        return 'TensorFlow graph object with {} nodes'.format(len(self.nodes))

# class TFGraph(object):
    # def __init__(self):
        # self.layers = OrderedDict()
        # self.vars = OrderedDict()
        # self.params = OrderedDict()

    # def addLayer(self, layer):
        # ename = layer.name
        # while self.layers.has_key(ename):
            # ename = ename + 'x'
        # if layer.name != ename:
            # print "Warning: a layer with name %s was already found, using %s instead" % \
                # (layer.name, ename)
            # layer.name = ename
        # for v in layer.inputs:  self.addVar(v)
        # for v in layer.outputs: self.addVar(v)
        # for p in layer.params: self.addParam(p)
        # self.layers[layer.name] = layer

    # def addVar(self, name):
        # if not self.vars.has_key(name):
            # self.vars[name] = CaffeBlob(name)

    # def addParam(self, name):
        # if not self.params.has_key(name):
            # self.params[name] = CaffeBlob(name)

    # def renameLayer(self, old, new):
        # self.layers[old].name = new
        # # reinsert layer with new name -- this mess is to preserve the order
        # layers = OrderedDict([(new,v) if k==old else (k,v)
                              # for k,v in self.layers.items()])
        # self.layers = layers

    # def renameVar(self, old, new, afterLayer=None):
        # self.vars[old].name = new
        # if afterLayer is not None:
            # start = self.layers.keys().index(afterLayer) + 1
        # else:
            # start = 0
        # # fix all references to the variable
        # for layer in self.layers.values()[start:-1]:
            # layer.inputs = [new if x==old else x for x in layer.inputs]
            # layer.outputs = [new if x==old else x for x in layer.outputs]
        # self.vars[new] = copy.deepcopy(self.vars[old])
        # # check if we can delete the old one (for afterLayet != None)
        # stillUsed = False
        # for layer in self.layers.values():
            # stillUsed = stillUsed or old in layer.inputs or old in layer.outputs
        # if not stillUsed:
            # del self.vars[old]

    # def renameParam(self, old, new):
        # self.params[old].name = new
        # # fix all references to the variable
        # for layer in self.layers.itervalues():
            # layer.params = [new if x==old else x for x in layer.params]
        # var = self.params[old]
        # del self.params[old]
        # self.params[new] = var

    # def removeParam(self, name):
        # del self.params[name]

    # def removeLayer(self, name):
        # # todo: fix this stuff for weight sharing
        # layer = self.layers[name]
        # for paramName in layer.params:
            # self.removeParam(paramName)
        # del self.layers[name]

    # def getLayersWithOutput(self, varName):
        # layerNames = []
        # for layer in self.layers.itervalues():
            # if varName in layer.outputs:
                # layerNames.append(layer.name)
        # return layerNames

    # def getLayersWithInput(self, varName):
        # layerNames = []
        # for layer in self.layers.itervalues():
            # if varName in layer.inputs:
                # layerNames.append(layer.name)
        # return layerNames

    # def reshape(self):
        # for layer in self.layers.itervalues():
            # layer.reshape(self)

    # def display(self):
        # for layer in self.layers.itervalues():
            # layer.display()
        # for var in self.vars.itervalues():
            # print 'Variable \'{}\''.format(var.name)
            # print '   + shape (computed): %s' % (var.shape,)
        # for par in self.params.itervalues():
            # print 'Parameter \'{}\''.format(par.name)
            # print '   + data found: %s' % (par.shape is not None)
            # print '   + data shape: %s' % (par.shape,)

    # def transpose(self):
        # for var in self.vars.itervalues():
            # if var.transposable: var.transpose()
        # for layer in self.layers.itervalues():
            # layer.transpose(self)

    # def getParentTransforms(self, variableName, topLayerName=None):
        # layerNames = self.layers.keys()
        # if topLayerName:
            # layerIndex = layerNames.index(topLayerName)
        # else:
            # layerIndex = len(self.layers) + 1
        # transforms = OrderedDict()
        # transforms[variableName] = CaffeTransform([1.,1.], [1.,1.], [1.,1.])
        # for layerName in reversed(layerNames[0:layerIndex]):
            # layer = self.layers[layerName]
            # layerTfs = layer.getTransforms(self)
            # for i, inputName in enumerate(layer.inputs):
                # tfs = []
                # if transforms.has_key(inputName):
                    # tfs.append(transforms[inputName])
                # for j, outputName in enumerate(layer.outputs):
                    # if layerTfs[i][j] is None: continue
                    # if transforms.has_key(outputName):
                        # composed = composeTransforms(layerTfs[i][j], transforms[outputName])
                        # tfs.append(composed)

                # if len(tfs) > 0:
                    # # should resolve conflicts, not simply pick the first tf
                    # transforms[inputName] = tfs[0]
        # return transforms
