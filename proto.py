import tensorflow as tf
from tensorflow.python.framework import graph_util

import sys
# p =
# v1 = tf.Variable(none,p)
#  with tf.Session() as sess:
#     saver = tf.train.Saver()
#     saver.restore(sess, 'a.ckpt.data-00000-of-00001')
init_op = tf.global_variables_initializer()
# saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(init_op)
    # save_path = saver.save(sess, "/tmp/model.ckpt")
    # tf.train.Saver().save(sess, 'a.ckpt')
    # tf.train.write_graph(sess.graph.as_graph_def(), logdir='.', name='a.pb', as_text=False)
    # import best model
    saver = tf.train.import_meta_graph('simple.ckpt.meta')  # graph Meta
    saver.restore(sess, "/home/glm/PycharmProjects/custom_model/simple.ckpt")
    # saver.restore(sess, 'a.ckpt')  # variables

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
    # output_graph_def = graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), ["save_1/restore_all"])
    converted_graph_def = graph_util.convert_variables_to_constants(sess, gd, ["output"])
    tf.train.write_graph(converted_graph_def, '', 'proto.pb', as_text=False)
