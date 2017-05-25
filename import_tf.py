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
import google.protobuf.text_format import ipdb
import tensorflow as tf
from pathlib import Path
import json
from tensorflow.python.lib.io import file_io
from tensorflow.core.framework import graph_pb2
from tensorflow.core.protobuf import meta_graph_pb2
from google.protobuf import text_format
import tf_mcn

verbose = 0 
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
out_path = Path.home() / 'coding/libs/matconvnets/contrib-matconvnet/contrib/mcnYOLO/models/yolo-voc-mcn.mat'

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
    kwargs = {}

    if verbose:
        print('-------------------')
        print('Node: {:3d} Added op \'{}\' ({})'.format(idx, op, name))

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if op in ['Placeholder', 'NoOp', 'Pad', 'Sub', 'RealDiv', 
                               'Mul', 'Maximum', 'Identity']:
        pass

    elif op in ['Const']:
        tensor = node.attr['value'].tensor
        shape = [x.size for x in tensor.tensor_shape.dim]
        np_dtype = tf2np_dtype[node.attr['dtype'].type]

        # handle differnt forms of data storage
        if len(tensor.float_val) > 0:
            value = tensor.float_val
        elif len(tensor.int_val) > 0:
            value = tensor.int_val
        elif len(tensor.tensor_content) > 0:
            raw = np.fromstring(tensor.tensor_content, dtype=np_dtype)
            value = np.reshape(raw, shape)
        else:
            raise ValueError('Unrecognised tensor values')

        kwargs['shape'] = shape
        kwargs['value'] = value


    elif op in ['MaxPool']:
        kwargs['data_format'] = tf2mcn_order[node.attr['data_format'].s.decode('utf-8')]
        kwargs['ksize'] = node.attr['ksize'].list.i
        kwargs['stride'] = node.attr['strides'].list.i
        kwargs['pad_type'] = node.attr['padding'].s.decode('utf-8')

    elif op in ['BiasAdd']:
        kwargs['data_format'] = tf2mcn_order[node.attr['data_format'].s.decode('utf-8')]

    elif op in ['Conv2D']:
        kwargs['data_format'] = tf2mcn_order[node.attr['data_format'].s.decode('utf-8')]
        kwargs['stride'] = node.attr['strides'].list.i
        kwargs['pad_type'] = node.attr['padding'].s.decode('utf-8')

    elif op in ['ExtractImagePatches']:
        kwargs['stride'] = node.attr['strides'].list.i
        kwargs['ksize'] = node.attr['ksize'].list.i
        kwargs['rate'] = node.attr['rates'].list.i
        kwargs['pad_type'] = node.attr['padding'].s.decode('utf-8')

    elif op in ['ConcatV2']:
        kwargs['axis'] = node.attr['N'].i
    else:
        raise ValueError('Unrecognised op: {}'.format(op))

    tf_node = tf_mcn.TFNode(name, inputs, op, **kwargs)
    node_list.append(tf_node)


# --------------------------------------------------------------------
#                                        construct computational graph
# --------------------------------------------------------------------

# graph construction is done in reverse order, using input_names
# to set references to previous nodes in the graph
node_list = list(reversed(node_list))
node_names = [node.name for node in node_list]

for idx, node in enumerate(node_list):
    if len(node.input_names) == 0:
        if verbose:
            print('Parameter node {}, skipping'.format(node.name))
    else:
        for input_name in node.input_names:
            input_node = node_list[node_names.index(input_name)]
            node.inputs.append(input_node)
    if verbose:
        print('processing node {}/{}'.format(idx, len(node_list)))

tf_graph = tf_mcn.TFGraph(node_list)

# build layers from root
head = tf_graph.nodes[node_names.index('output')]
layers = []
layerNames = [] # ensure unique names for each new layer

def tf2mcn(node):
    """
    Overlay a graph of mcn nodes over the tf computation graph. A TF
    node is considered to have been visited once its `mcn` attribute
    has been set.
    """

    # ----------------------------------
    # Base cases - input nodes for graph
    # ----------------------------------
    if not node.inputs:
        if node.op == 'Placeholder':
            value = []
        elif node.op == 'Const':
            value = node.value
        else:
            error('unrecognised input {}'.format(type(node)))
        node.mcn = tf_mcn.McnNode(name=node.name, value=value, op=node.op) 
        return node

    # post-order traversal - gather expression inputs for each op
    mcnIns = []
    for in_node in node.inputs:
        if not hasattr(in_node, 'mcn'):
            in_node = tf2mcn(in_node)
        mcnIns.append(in_node.mcn)

    # ----------------------------------
    # Expression resolution
    # ----------------------------------
    # Matconvnet works at the 'layer' abstraction, while TensorFlow works at
    # the `op` level of abstraction. To reconcile this difference, the graph
    # of tf nodes is converted into a set of mcn nodes by pattern matching 
    # common operations and clustering them into layers.

    node_candidates = ['Sub', 'RealDiv', 'Mul', 'Pad', 'Identity']
    layer_candidates = ['Conv2D', 'MaxPool', 'ConcatV2', 'ExtractImagePatches']
    special_cases =  ['BiasAdd', 'Maximum']

    # tf nodes -> mcn nodes
    if node.op in node_candidates:
        if node.op in ['Sub', 'RealDiv', 'Mul']:
            name = node.name
            value = []
        elif node.op == 'Identity':
            src_node, = tf_mcn.parse_inputs(mcnIns, ['Any'])
            name = src_node.name
            value = []
        elif node.op == 'Pad':
            [pad_node, src_node] = tf_mcn.parse_inputs(mcnIns, ['Const', 'Any'])
            name = src_node.name
            value = pad_node.value
        node.mcn = tf_mcn.McnNode(name=name, value=value, op=node.op, input_nodes=mcnIns) 

    # tf nodes -> mcn layers
    elif node.op in layer_candidates:
        name = tf_mcn.buildMcnLayerName(node.op, layerNames)

        if node.op == 'Conv2D':
            layerType = tf_mcn.McnConv
        elif node.op == 'MaxPool':
            layerType = lambda x,y,z: tf_mcn.McnPooling(x,y,z,'max')
        elif node.op == 'AvgPool':
            layerType = lambda x,y,z: tf_mcn.McnPooling(x,y,z,'avg')
        elif node.op == 'ConcatV2':
            layerType = tf_mcn.McnConcat
        elif node.op == 'ExtractImagePatches':
            layerType = tf_mcn.McnExtractImagePatches

        mcnLayer = layerType(name, node, mcnIns) 
        layers.append(mcnLayer)
        layerNames.append(name)
        node.mcn = mcnLayer

    elif node.op in special_cases:
        if node.op == 'BiasAdd':
            # while the bias add op is a generic operation, we perform
            # pattern matching here to check for the presence of a batch norm layer
            is_bn_layer = tf_mcn.McnBatchNorm.is_batch_norm_expression(node, mcnIns)

            # ----------------------------------------------------------
            if not is_bn_layer:

                # if batch norm is not used, then we merge the bias with
                # the preceeding convolutional layer (if one exists)
                for node in mcnIns:
                    if node.op == 'Conv2D':
                        conv_node = node
                    elif node.op == 'Const':
                        bias_node = node
                    else:
                        ipdb.set_trace()
                        raise NotImplementedError('no support for solo biases yet')

                if 'conv_node' not in locals():
                    raise NotImplementedError('no support for solo biases yet')

                conv_node.bias_term = 1 # bias is now used
                bias_name = conv_node.name + '_bias'
                conv_node.params.append(bias_name)
                conv_node.param_values[bias_name] = bias_node.value
                value = []
                node.mcn = tf_mcn.McnNode(node.name, value, node.op, mcnIns) 
            # ----------------------------------------------------------
            else:
                # construct matconvnet layer name
                name = tf_mcn.buildMcnLayerName(node.op, layerNames)
                mcnLayer = tf_mcn.McnBatchNorm(name, node, mcnIns) 
                layers.append(mcnLayer)
                layerNames.append(name)
                node.mcn = mcnLayer
            # ----------------------------------------------------------

        elif node.op == 'Maximum':
            # similarly as above, the elementwise max is a generic operation, but 
            # we perform pattern matching here to check for the presence of a relu
            is_leaky_relu_layer = tf_mcn.McnReLU.is_leaky_relu_expression(node, mcnIns)

            # ----------------------------------------------------------
            if not is_leaky_relu_layer:
                print('not handled yet')
                ipdb.set_trace()
            # ----------------------------------------------------------
            # construct matconvnet layer name
            name = tf_mcn.buildMcnLayerName(node.op, layerNames)
            mcnLayer = tf_mcn.McnReLU(name, node, mcnIns) 
            layers.append(mcnLayer)
            layerNames.append(name)
            node.mcn = mcnLayer

    else:
        raise ValueError('node op {} not recognised'.format(node.op))

    if verbose:
        print('processed: {}'.format(node.name))
    return node

# magic
tf2mcn(head)

tf_model = tf_mcn.TFModel()
for layer in layers:
    tf_model.addLayer(layer)

# --------------------------------------------------------------------
#                                            extract meta information
# --------------------------------------------------------------------

anchors = meta['anchors']
net_meta = meta['net']

in_size = [net_meta[x] for x in ['height', 'width', 'channels']]
out_size = meta['out_size']

classes = meta['labels']
mnormalization = {}
mnormalization['imageSize'] = in_size

# --------------------------------------------------------------------
#                                                    Convert to MATLAB
# --------------------------------------------------------------------

# net.meta
meta_dict = {'inputs': in_size,
             'normalization': mnormalization, 
             'classes': meta['labels'],
             'thresh': meta['thresh'],
             'anchors': meta['anchors'],
             }

mmeta = tf_mcn.dictToMatlabStruct(meta_dict)

mnet = {'layers': np.empty(shape=[0,], dtype=tf_mcn.mlayerdt),
        'params': np.empty(shape=[0,], dtype=tf_mcn.mparamdt),
        'meta': mmeta}

for layer in tf_model.layers.values():
    mnet['layers'] = np.append(mnet['layers'], layer.toMatlab(), axis=0)

for param in tf_model.params.values():
    mnet['params'] = np.append(mnet['params'], param.toMatlab(), axis=0)

# # to row
mnet['layers'] = mnet['layers'].reshape(1,-1)
mnet['params'] = mnet['params'].reshape(1,-1)

# --------------------------------------------------------------------
#                                                          Save output
# --------------------------------------------------------------------

print('Saving network to {}'.format(str(out_path)))
scipy.io.savemat(str(out_path), mnet, oned_as='column')
