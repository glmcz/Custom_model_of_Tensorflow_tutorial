import tensorflow as tf
from tensorflow.python.framework import graph_util
model_path = '/home/glm/PycharmProjects/custom_model/graph/def.pb'

f = tf.python.platform.gfile.FastGFile(model_path)
gd = graph_def = tf.GraphDef()
graph_def.ParseFromString(f.read())

# # fix nodes
# for node in graph_def.node:
#   if node.op == 'RefSwitch':
#     node.op = 'Switch'
#     for index in xrange(len(node.input)):
#       if 'moving_' in node.input[index]:
#         node.input[index] = node.input[index] + '/read'
#   elif node.op == 'AssignSub':
#     node.op = 'Sub'
#     if 'use_locking' in node.attr: del node.attr['use_locking']

# import graph into session
tf.import_graph_def(graph_def, name='')