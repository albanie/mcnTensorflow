import sys
import os
import argparse
import importlib
import code
import re
import numpy as np
from math import floor, ceil
import scipy.io
import scipy.misc
import google.protobuf.text_format
import ipdb
import tensorflow as tf
from pathlib import Path
import json
from tensorflow.python.lib.io import file_io
from tensorflow.core.framework import graph_pb2
from tensorflow.core.protobuf import meta_graph_pb2
from google.protobuf import text_format
import tf_nodes
from tf_nodes import *

# --------------------------------------------------------------------
#                                                   conversion helpers
# --------------------------------------------------------------------

# convert tf data types into numpy data types
tf2np_dtype = {
    1: np.float32, 
    3: np.int32, 
}

# convert data format enumerations into a canonical array ordering
# matching the mcn convention of (H,W,C,N)
tf2mcn_order = {
    'NHWC': [1,2,3,0], 
}

# --------------------------------------------------------------------
#                                                       Load layers 
# --------------------------------------------------------------------

# Unlike caffe, Tensorflow stores the network structure in a single file
base = Path.home() / 'coding/libs/darkflow/built_graph'
path = base / 'yolo-voc-v2.pb'
meta_path = base / 'yolo-voc-v2.meta'

# import the graph definition from TF
graph_def = graph_pb2.GraphDef() 

# parse the data
with open(str(path), "rb") as f:
  graph_def.ParseFromString(f.read())

# parse the meta info as json, since the protobuf appears to have issues
meta_raw = file_io.FileIO(str(meta_path), "rb").read().decode('utf-8')
meta = json.loads(meta_raw)

# --------------------------------------------------------------------
#                                        Read ops into TF node objects
# --------------------------------------------------------------------

node_list = []

for idx, node in enumerate(graph_def.node):

    # process each node according to its op
    op = node.op
    name = node.name 
    inputs = node.input

    print('-------------------')
    print('Node: {:3d} Added op \'{}\' ({})'.format(idx, op, name))

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if op in ['Placeholder']:
        input_types = []
        tf_node = TFPlaceHolder(name, inputs, input_types)

    elif op in ['Const']:
        input_types = []
        tensor = node.attr['value'].tensor
        shape = [x.size for x in tensor.tensor_shape.dim]
        np_dtype = tf2np_dtype[node.attr['dtype'].type]

        # handle the differnt forms of storing data
        if len(tensor.float_val) > 0:
            value = tensor.float_val
        elif len(tensor.int_val) > 0:
            value = tensor.int_val
        elif len(tensor.tensor_content) > 0:
            raw = np.fromstring(tensor.tensor_content, dtype=np_dtype)
            value = np.reshape(raw, shape)
        else:
            raise ValueError('Unrecognised tensor values')

        tf_node = TFConst(name, inputs, input_types, shape, value)

    elif op in ['Pad']:
        input_types = {k:tf2np_dtype[node.attr[k].type] for k in node.attr.keys()}
        tf_node = TFPad(name, inputs, input_types)

    elif op in ['NoOp']:
        input_types = []
        tf_node = TFNoOp(name, inputs, input_types)

    elif op in ['Sub']:
        input_types = tf2np_dtype[node.attr['T'].type]
        tf_node = TFSub(name, inputs, input_types)

    elif op in ['RealDiv']:
        input_types = tf2np_dtype[node.attr['T'].type]
        tf_node = TFRealDiv(name, inputs, input_types)

    elif op in ['Mul']:
        input_types = tf2np_dtype[node.attr['T'].type]
        tf_node = TFMul(name, inputs, input_types)

    elif op in ['Maximum']:
        input_types = tf2np_dtype[node.attr['T'].type]
        tf_node = TFMaximum(name, inputs, input_types)

    elif op in ['Identity']:
        input_types = tf2np_dtype[node.attr['T'].type]
        tf_node = TFIdentity(name, inputs, input_types)

    elif op in ['MaxPool']:
        input_types = tf2np_dtype[node.attr['T'].type]
        data_format = tf2mcn_order[node.attr['data_format'].s.decode('utf-8')]
        ksize = node.attr['ksize'].list.i
        stride = node.attr['strides'].list.i
        pad_type = node.attr['padding'].s.decode('utf-8')
        tf_node = TFMaxPool(name, inputs, stride, pad_type, ksize, input_types, 
                                                                   data_format)

    elif op in ['BiasAdd']:
        input_types = tf2np_dtype[node.attr['T'].type]
        data_format = tf2mcn_order[node.attr['data_format'].s.decode('utf-8')]
        tf_node = TFBiasAdd(name, inputs, input_types, data_format)

    elif op in ['Conv2D']:
        input_types = tf2np_dtype[node.attr['T'].type]
        data_format = tf2mcn_order[node.attr['data_format'].s.decode('utf-8')]
        stride = node.attr['strides'].list.i
        pad_type = node.attr['padding'].s
        tf_node = TFConv2D(name, inputs, stride, pad_type, input_types, data_format) 

    elif op in ['ExtractImagePatches']:
        input_types = tf2np_dtype[node.attr['T'].type]
        stride = node.attr['strides'].list.i
        ksize = node.attr['ksize'].list.i
        rate = node.attr['rates'].list.i
        pad_type = node.attr['padding'].s
        tf_node = TFExtractImagePatches(name, inputs, rate, stride, pad_type, ksize, 
                                                                    input_types) 

    elif op in ['ConcatV2']:
        axis = node.attr['N'].i
        input_types = tf2np_dtype[node.attr['T'].type]
        tf_node = TFConcatV2(name, inputs, input_types, axis) 
    else:
        raise ValueError('Unrecognised op: {}'.format(op))

    node_list.append(tf_node)


# --------------------------------------------------------------------
#                                        construct computational graph
# --------------------------------------------------------------------

# graph construction is done in reverse order, using input_names
# to set references to previous nodes in the graph
node_list = list(reversed(node_list))
node_names = [node.name for node in node_list]

for idx, node in enumerate(node_list):
    print('processing node {}/{}'.format(idx, len(node_list)))
    if len(node.input_names) == 0:
        print('Parameter node {}, skipping'.format(node.name))
    else:
        for input_name in node.input_names:
            input_node = node_list[node_names.index(input_name)]
            node.inputs.append(input_node)

tf_graph = TFGraph(node_list)

# build layers from root
head = tf_graph.nodes[node_names.index('output')]
layers = []
layerNames = [] # ensure unique names for each new layer

def tf2mcn(node, depth):
    # print('processing {}'.format(node.name))
    # print('processing inputs {}'.format(node.inputs))
    # print('current depth {}'.format(depth))

    # --------------------------------
    # Base cases - input node to graph
    # --------------------------------
    if not node.inputs:
        if isinstance(node, tf_nodes.TFPlaceHolder):
            x = tf_nodes.McnInput(node.name)
            op = 'input'
        elif isinstance(node, tf_nodes.TFConst):
            x = tf_nodes.McnParam(node.name, node.value)
            op = 'const'
        else:
            error('unrecognised input {}'.format(type(node)))
        node.mcn = (x, [], op) # store mcn expression to indicated as visited
        return node

    # post-order traversal - gather expression inputs for binary ops
    mcnIns = []
    for in_node in node.inputs:
        if not hasattr(in_node, 'mcn'):
            in_node = tf2mcn(in_node, depth+1)
        mcnIns.append(in_node.mcn)

    # resolve binary ops
    if isinstance(node, tf_nodes.TFPad):
        assert len(mcnIns) == 2, 'padding op expects two inputs'

        # check that the second input contains the padding
        padIn = mcnIns[1]
        assert(padIn[2] == 'const')

        vname = mcnIns[0][0].name
        pad = mcnIns[1][0].value
        node.mcn = (vname, pad, 'pad')
    elif isinstance(node, tf_nodes.TFConv2D):

        # construct matconvnet layer name
        num_prev_convs = sum(['conv' in x for x in layerNames])
        name = 'conv{}'.format(num_prev_convs + 1)

        # build layer
        mcnLayer = tf_nodes.McnConv(name, node, mcnIns) 
        layers.append(mcnLayer)
        layerNames.append(name)

        # store new layer as expression
        node.mcn = (mcnLayer, 'conv')

    elif isinstance(node, tf_nodes.TFSub):
        assert len(mcnIns) == 2, 'sub op expects two inputs'
        arg1 = mcnIns[0]
        arg2 = mcnIns[1]
        node.mcn = (arg1, arg2, 'sub')
    elif isinstance(node, tf_nodes.TFRealDiv):
        assert len(mcnIns) == 2, 'realdiv op expects two inputs'
        arg1 = mcnIns[0]
        arg2 = mcnIns[1]
        node.mcn = (arg1, arg2, 'div')
    elif isinstance(node, tf_nodes.TFMul):
        assert len(mcnIns) == 2, 'mul op expects two inputs'
        arg1 = mcnIns[0]
        arg2 = mcnIns[1]
        node.mcn = (arg1, arg2, 'mul')
    elif isinstance(node, tf_nodes.TFBiasAdd):
        # while the bias add op is a generic operation, we perform
        # pattern matching here to check for the presence of a batch norm layer
        is_bn_layer = McnBatchNorm.is_batch_norm_expression(node, mcnIns)

        # ----------------------------------------------------------
        if not is_bn_layer:
            print('not handled yet')
            ipdb.set_trace()

            arg1 = mcnIns[0]
            arg2 = mcnIns[1]
            node.mcn = (arg1, arg2, 'biasAdd')
        # ----------------------------------------------------------


        #TODO(sam): Note this currently assumes the standard format
        # of conv then BN, will need to make more general 

        # construct matconvnet layer name
        num_prev_convs = sum(['conv' in x for x in layerNames])
        name = 'bn{}'.format(num_prev_convs)

        # build layer
        mcnLayer = tf_nodes.McnBatchNorm(name, node, mcnIns) 
        layers.append(mcnLayer)
        layerNames.append(name)
        node.mcn = (mcnLayer, 'batchNorm')

    elif isinstance(node, tf_nodes.TFMaximum):
        # similarly as above, the elementwise max is a generic operation, but 
        # we perform pattern matching here to check for the presence of a relu
        is_leaky_relu_layer = McnReLU.is_leaky_relu_expression(node, mcnIns)

        # ----------------------------------------------------------
        if not is_leaky_relu_layer:
            print('not handled yet')
            ipdb.set_trace()

            arg1 = mcnIns[0]
            arg2 = mcnIns[1]
            node.mcn = (arg1, arg2, 'max')
        # ----------------------------------------------------------
        # construct matconvnet layer name
        num_prev_convs = sum(['conv' in x for x in layerNames])
        name = 'relu{}'.format(num_prev_convs)

        # build layer
        mcnLayer = tf_nodes.McnReLU(name, node, mcnIns) 
        layers.append(mcnLayer)
        layerNames.append(name)
        node.mcn = (mcnLayer, 'relu')
    elif isinstance(node, tf_nodes.TFMaxPool):
        # construct matconvnet layer name
        num_prev_pools = sum(['pool' in x for x in layerNames])
        name = 'pool{}'.format(num_prev_pools + 1)

        # build layer
        mcnLayer = tf_nodes.McnPooling(name, node, mcnIns, 'max') 
        layers.append(mcnLayer)
        layerNames.append(name)

        # store new layer as expression
        node.mcn = (mcnLayer, 'pool')
    else:
        ipdb.set_trace()
    print('processed: {}'.format(node.name))
    return node

    # if isinstance(node, tf_nodes.TFPlaceHolder):
        # var = tf_nodes.McnInput(node.name)
        # out = {'input', var}
        # return out
    # elif isinstance(node, tf_nodes.TFConst):
        # param = tf_nodes.McnParam(node.name, node.value)
        # out = {'param': param}
        # return out
    # elif isinstance(node, tf_nodes.TFConv2D):
        # print('here')
    # elif isinstance(node, tf_nodes.TFBiasAdd):
        # print('here2')
    # else:
        # print('here3')

depth = 0 
graph2layers(head, depth)

# --------------------------------------------------------------------
#                                            extract meta information
# --------------------------------------------------------------------

anchors = meta['anchors']
labels = meta['labels']
net_meta = meta['net']

in_size = [net_meta[x] for x in ['height', 'width', 'channels']]
out_size = meta['out_size']

