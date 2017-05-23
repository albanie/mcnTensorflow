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
from tensorflow.core.framework import graph_pb2
from google.protobuf import text_format
from tf_layers import *

# --------------------------------------------------------------------
#                                                              helpers
# --------------------------------------------------------------------

# convert tf data types into numpy data types
tf2np_dtype = {
    1: np.float32, 
    3: np.int32, 
}

# convert data format enumerations into a canonical array ordering
# matching the mcn convention of (H,W,C,N)
tf2mcn_order = {
    'NHWC': [3,0,1,2], 
}


# def format2order(tf_data_format):
    # """
    # convert data format enumerations into a canonical array ordering
    # matching the mcn convention of (H,W,C,N)
    # """
    # data_format_lookup = {
       # 's: "NHWC"': [3,0,1,2], 
    # }
    # return data_format_lookup[str(tf_data_format).rstrip()]

# --------------------------------------------------------------------
#                                                          Load layers
# --------------------------------------------------------------------

# Unlike caffe, Tensorflow stores the network structure in a single file
path = '/users/albanie/coding/libs/darkflow/built_graph/tiny-yolo.pb'

# import the graph definition from TF
graph_def = graph_pb2.GraphDef() 

# parse the data
with open(path, "rb") as f:
  graph_def.ParseFromString(f.read())

# --------------------------------------------------------------------
#                                         Read layers into a TF object
# --------------------------------------------------------------------

# store the graph nodes in a list

tf_graph = TFGraph()
for node in graph_def.node[:6]:

    # process each node according to its op
    op = node.op
    name = node.name 
    inputs = node.input

    print('-------------------')
    print('Added op \'{}\' ({})'.format(op, name))

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if op in ['Placeholder']:
        tf_node = TFPlaceHolder(name, inputs)
        tf_graph.nodes[name] = tf_node

    if op in ['Const']:
        tensor = node.attr['value'].tensor
        shape = [x.size for x in tensor.tensor_shape.dim]
        np_dtype = tf2np_dtype[node.attr['dtype'].type]
        raw = np.fromstring(tensor.tensor_content, dtype=np_dtype)
        value = np.reshape(raw, shape)
        tf_node = TFConst(name, inputs, shape, value)
        tf_graph.nodes[name] = tf_node

    if op in ['Pad']:
        input_types = {k:tf2np_dtype[node.attr[k].type] for k in node.attr.keys()}
        tf_node = TFPad(name, inputs, input_types)
        tf_graph.nodes[name] = tf_node

    if op in ['Conv2D']:
        input_types = tf2np_dtype[node.attr['T'].type]
        data_format = tf2mcn_order[node.attr['data_format'].s.decode("utf-8")]
        stride = node.attr['strides'].list.i
        pad_type = node.attr['padding'].s
        tf_node = TFConv2D(name, inputs, stride, pad_type, data_format, input_types) 
        tf_graph.nodes[name] = tf_node

    if op in ['Sub']:
        input_types = tf2np_dtype[node.attr['T'].type]
        tf_node = TFSub(name, inputs, input_types)
        tf_graph.nodes[name] = tf_node
