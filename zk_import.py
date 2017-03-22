import tensorflow as tf
# N
# # Let's load a previously saved meta graph in the default graph
# # This function returns a Saver
# saver = tf.train.import_meta_graph('final.ckpt.meta')
#
# # We can now access the default graph where all our metadata has been loaded
# graph = tf.get_default_graph()
#
# # Finally we can retrieve tensors, operations, collections, etc.
# global_step_tensor = graph.get_operation_by_name('output')
# tf.import_graph_def('model.meta')

# sess = tf.Session()
# saver = tf.train.import_meta_graph('model.meta')
# saver.restore(sess, tf.train.latest_checkpoint('./'))

# tf.import_graph_def('model.meta')
tf.import_graph_def('graph/def.proto')
tf.import_graph_def('graph/defText.proto')
# tf.import_graph_def('graph/graphDef.proto')

# def save(self, filename):
#     for variable in tf.trainable_variables():
#         tensor = tf.constant(variable.eval())
#         tf.assign(variable, tensor, name="nWeights")
#
#     tf.train.write_graph(self.graph_def, 'graph/', 'graph_human', as_text=True)
