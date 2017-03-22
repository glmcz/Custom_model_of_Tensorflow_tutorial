"""
Convert model.ckpt to model.pb
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.framework import graph_util

# create a session
sess = tf.Session()

# import best model
saver = tf.train.import_meta_graph('/home/glm/PycharmProjects/custom_model/graph/check.ckpt.meta') # graph
saver.restore(sess, '/home/glm/PycharmProjects/custom_model/graph/check.ckpt') # variables

# get graph definition
gd = sess.graph.as_graph_def()

# fix batch norm nodes
for node in gd.node:
  if node.op == 'RefSwitch':
    node.op = 'Switch'
    for index in xrange(len(node.input)):
      if 'moving_' in node.input[index]:
        node.input[index] = node.input[index] + '/read'
  elif node.op == 'AssignSub':
    node.op = 'Sub'
    if 'use_locking' in node.attr: del node.attr['use_locking']

# generate protobuf
converted_graph_def = graph_util.convert_variables_to_constants(sess, gd, ["dense2_end_W"])
tf.train.write_graph(converted_graph_def, 'graph', 'ckpt_to_pb_model.pb', as_text=False)