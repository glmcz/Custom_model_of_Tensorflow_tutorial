import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
import tensorflow as tf
from keras.models import model_from_json
from tensorflow.core.protobuf import saver_pb2

import freezegraph
import zk_import

img_width, img_height = 150, 150

train_data_dir = '/home/glm/bankovky/train'

validation_data_dir = '/home/glm/bankovky/validace'
nb_train_samples = 2000
nb_validation_samples = 800
nb_epoch = 20
step = 1000

def save_graph(sess,output_path,checkpoint,checkpoint_state_name,input_graph_name,output_graph_name):

    checkpoint_prefix = os.path.join(output_path,checkpoint)
    saver = tf.train.Saver(tf.all_variables())
    saver.save(sess, checkpoint_prefix, global_step=0,latest_filename=checkpoint_state_name)
    tf.train.write_graph(sess.graph.as_graph_def(),output_path,
                           input_graph_name)

    # We save out the graph to disk, and then call the const conversion
    # routine.
    input_graph_path = os.path.join(output_path, input_graph_name)
    input_saver_def_path = ""
    input_binary = False
    input_checkpoint_path = checkpoint_prefix + "-0"
    output_node_names = "out/Softmax"
    restore_op_name = "restore_op/restore_all"
    filename_tensor_name = "filename_tensor/Const:0"
    output_graph_path = os.path.join(output_path, output_graph_name)
    clear_devices = False

    freezegraph.freeze_graph(input_graph_path, input_saver_def_path,
                              input_binary, input_checkpoint_path,
                              output_node_names, restore_op_name,
                              filename_tensor_name, output_graph_path,clear_devices, "")



# # # this is the augmentation configuration we will use for training
# train_datagen = ImageDataGenerator(
#         rescale=1./255,
#         shear_range=0.2,
#         zoom_range=0.2,
#         horizontal_flip=True)
#
# # this is the augmentation configuration we will use for testing:
# # only rescaling
# test_datagen = ImageDataGenerator(rescale=1./255)
#
# train_generator = train_datagen.flow_from_directory(
#         train_data_dir,
#         target_size=(img_width, img_height),
#         batch_size=32,
#         class_mode='binary')
#
# validation_generator = test_datagen.flow_from_directory(
#         validation_data_dir,
#         target_size=(img_width, img_height),
#         batch_size=32,
#         class_mode='binary')
#
# model = Sequential()
#
# model.add(Convolution2D(32, 3, 3, input_shape=(img_width, img_height, 3)))
#
# # model.add(Convolution2D(32, 3, 3, input_shape=(3, img_width, img_height))) pro theano backend
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
#
# model.add(Convolution2D(32, 3, 3))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
#
# model.add(Convolution2D(64, 3, 3))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Flatten())
# model.add(Dense(64))
# model.add(Activation('relu'))
# # In Keras, Dropout applies to just the layer preceding it (It technically applies it to its own inputs, but its own inputs are just the outputs from the layer preceding it.)
# model.add(Dropout(0.5))
# model.add(Dense(1))
# model.add(Activation('softmax', name='output'))
#
#
#
# # When dealing with high-dimensional inputs such as images, as we saw above it is IMPRACTICAL to connect neurons to all neurons in the previous volume.
#
# model.compile(loss='binary_crossentropy',
#               optimizer='rmsprop',
#               metrics=['accuracy'])
#
# model.fit_generator(
#         train_generator,
#         samples_per_epoch=nb_train_samples,
#         nb_epoch=nb_epoch,
#         validation_data=validation_generator,
#         nb_val_samples=nb_validation_samples)
#

# model.save_weights('weight_5hdf.h5')
# model.save('model_5hdf.h5')
#
# # serialize model to JSON
# model_json = model.to_json()
# with open("model.json", "w") as json_file:
#     json_file.write(model_json)
# print("Saved model to disk")

# init_op = tf.initialize_all_variables()
# with tf.Session() as sess:
#     sess.run(init_op)
#     # # Write out the trained graph and labels with the weights stored as constants.
#     output_graph_def = graph_util.convert_variables_to_constants(
#         sess, graph.as_graph_def(), ['output'])
#     with gfile.FastGFile(FLAGS.output_graph, 'wb') as f:
#         f.write(output_graph_def.SerializeToString())
#     with gfile.FastGFile(FLAGS.output_labels, 'w') as f:
#         f.write('\n'.join(image_lists.keys()) + '\n')

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("1_weight_5hdf.h5")
print("Loaded model from disk")
# v2_saver = tf.train.Saver({"v2": v2})

# vgg_saver = tf.train.import_meta_graph('simple.ckpt.meta')
# Access the graph
# vgg_graph = tf.get_default_graph()

# Retrieve VGG inputs
# u'dense_1_W/Assign_1': <tensorflow.python.framework.ops.Operation object at 0x7f80a27e1150>, u'random_uniform/shape': <tensorflow.python.framework.ops.Operation object at 0x7f80a290f110>}
# x_plh = vgg_graph.get_tensor_by_name('^dense_2_b/Assign')
from tensorflow.python.framework import graph_util

with tf.Session() as sess:
    init_op = tf.variables_initializer(tf.global_variables(), name="nInit")
    # init_op = tf.global_variables_initializer()
    sess.run(init_op)
    zk_import.save(sess, 'def')
    tf.import_graph_def('graph/graph_human') #nejde nejspis se vyt
    # saver = tf.train.import_meta_graph('simple.ckpt.meta')  # graph Meta funguje
    # Files  architecture of TF with 5 different type of files:

    graph_def = sess.graph.as_graph_def()
    for node in graph_def.node:
        print node

    print "******************end*****************"
    constant_graf =graph_util.convert_variables_to_constants(sess, graph_def, ['Assign_9'])
    for node in constant_graf.node:
        print node
    # graph = ses.graph.get_tensor_by_name('^dense_2_b/Assign')
    # graph = ses.graph.get_operations() list of objects
    # graph = ses.graph.get_operation_by_name('^dense_2_b/Assign')
    # global_step_tensor = graph.get_tensor_by_name('^dense_2_b/Assign')
    v2_saver = tf.train.Saver({"output": global_step_tensor})
    v2_saver.save(sess, 'output.ckpt')

    tf.train.Saver().save(sess, 'simple.ckpt')
    tf.train.write_graph(sess.graph.as_graph_def(), logdir='.', name='simple_as_binary.pb', as_text=False)
#
    saver = tf.train.Saver()
    saver = tf.train.Saver(write_version=saver_pb2.SaverDef.V1)
    saver.save(sess, 'final.ckpt')
# #     save_path = saver.save(sess, "modelEras.ckpt")
# sess.close()
#
# with tf.Session() as sess:
#     saver = tf.train.import_meta_graph('final.ckpt.meta')
#     graf = tf.train.import_meta_graph('final.ckpt.meta')
#     saver.restore(sess, "/home/glm/PycharmProjects/custom_model/final.ckpt")
#
#     # Write out the trained graph and labels with the weights stored as constants.
#     output_graph_def = graph_util.convert_variables_to_constants(
#         sess, graph.as_graph_def(), [FLAGS.final_tensor_name])
#     with gfile.FastGFile(FLAGS.output_graph, 'wb') as f:
#         f.write(output_graph_def.SerializeToString())
#     with gfile.FastGFile(FLAGS.output_labels, 'w') as f:
#         f.write('\n'.join(image_lists.keys()) + '\n')

    # convert_variables_to_constants(sess,graf,)

# # evaluate loaded model on test data
# loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
# score = loaded_model.evaluate(X, Y, verbose=0)
# print "%s: %.2f%%" % (loaded_model.metrics_names[1], score[1] * 100)
# serialize model to YAML
# model_yaml = model.to_yaml()
# with open("model.yaml", "w") as yaml_file:
#     yaml_file.write(model_yaml)
# # serialize weights to HDF5
# model.save_weights("model.h5")
# print("Saved model to disk")
# # load YAML and create model
# yaml_file = open('model.yaml', 'r')
# loaded_model_yaml = yaml_file.read()
# yaml_file.close()
# loaded_model = model_from_yaml(loaded_model_yaml)
# # load weights into new model
# loaded_model.load_weights("model.h5")
# print("Loaded model from disk")
#
# # evaluate loaded model on test data
# loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
# score = loaded_model.evaluate(X, Y, verbose=0)
# print "%s: %.2f%%" % (loaded_model.metrics_names[1], score[1] * 100)
# tf.python.client.graph_util.convert_variables_to_constant

# ***********************************************Description************************************************
# TensorFlow save method saves three kinds of files because it stores the graph structure separately
# from the variable values. The .meta file describes the saved graph structure, so you need to
# import it before restoring the checkpoint (otherwise it doesn't know what variables the saved checkpoint values correspond to).
# with tf.Session() as sess:
#     saver = tf.train.import_meta_graph('/tmp/model.ckpt.meta')
#     saver.restore(sess, "/tmp/model.ckpt")
# Alternatively, you could do this:
# Recreate the EXACT SAME variables
# v1 = tf.Variable(..., name="v1")
# v2 = tf.Variable(..., name="v2")
# Now load the checkpoint variable values
# with tf.Session() as sess:
#     saver = tf.train.Saver()
#     saver.restore(sess, "/tmp/model.ckpt")
# *********************************************************************************************************



# model.load_weights('weight_5hdf.h5')