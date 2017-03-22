from keras.models import Sequential
from keras.layers import Flatten
from tensorflow.python.framework.graph_util import convert_variables_to_constants
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Convolution2D, MaxPooling2D, Dense

import tensorflow as tf
import keras
img_width, img_height = 150, 150

#     A “checkpoint” file
#     An “events” file
#     A “textual
#     protobufs” file
#     Some “chkp” files
#     Some “meta
#     chkp” files

train_data_dir = '/home/glm/bankovky/train'

validation_data_dir = '/home/glm/bankovky/validace'
nb_train_samples = 224
nb_validation_samples = 80
nb_epoch = 1
step = 1

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=32,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=32,
    class_mode='binary')

def save(self, filename):
    for variable in tf.trainable_variables():
        tensor = tf.constant(variable.eval())
        tf.assign(variable, tensor, name="nWeights")

    tf.train.write_graph(self.graph_def, 'graph/', 'graphDef.proto', as_text=True)
    tf.train.Saver().save(sess, 'check.ckpt')
    tf.train.write_graph(sess.graph.as_graph_def(), logdir='graph', name='def.pb', as_text=False)
    tf.train.write_graph(sess.graph.as_graph_def(), logdir='graph', name='defText.pb', as_text=True)


with tf.Session() as sess:
        own = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=1, write_graph=True, write_images=True)
        model = Sequential(name='zacatek')

        # model.add(Convolution2D(32, 3, 3, input_shape=(img_width, img_height, 3)))
        model.add(Convolution2D(32, 3, 3, input_shape=(150, 150, 3), name='input'))
        # model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        #
        # model.add(Convolution2D(64, 3, 3))
        # model.add(Activation('relu'))
        # model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        # model.add(Dense(64, name='dense1'))
        # model.add(Activation('relu', name='activation'))

        model.add(Dense(1, activation='softmax', name='dense2_end'))
        model.compile(loss='binary_crossentropy',
                      optimizer='rmsprop',
                      metrics=['accuracy'])
        model.fit_generator(
                train_generator,
                samples_per_epoch=nb_train_samples,
                nb_epoch=nb_epoch,
                validation_data=validation_generator,
                nb_val_samples=nb_validation_samples,
                callbacks=[own])

        model.save_weights('1_weight_5hdf.h5')
        model_json = model.to_json()

        config = model.get_config()
        weights = model.get_weights()

        from keras.models import model_from_config

        new_model = model.from_config(config)
        new_model.set_weights(weights)

        init_op = tf.variables_initializer(tf.global_variables(), name="nInit")

        export_path ="/home/glm/bankovky" # where to save the exported graph
        export_version = 123 # version number (integer)

        from tensorflow_serving.session_bundle import exporter

        saver = tf.train.Saver(sharded=True)
        model_exporter = exporter.Exporter(saver)
        signature = exporter.classification_signature(input_tensor=model.input,
                                                      scores_tensor=model.output)
        model_exporter.init(sess.graph.as_graph_def(),
                            default_graph_signature=signature)
        model_exporter.export(export_path, tf.constant(export_version), sess)

        with open("model.json", "w") as json_file:
                json_file.write(model_json)

        # init_op = tf.variables_initializer(tf.global_variables(), name="nInit")
        # meta_graph_def = tf.train.export_meta_graph(filename='tmp/modelMeta.meta')
        minimal_graph = convert_variables_to_constants(sess, sess.graph_def, ["group_deps"])
        save(sess, '')

        saver = tf.train.Saver()
        save_path = saver.save(sess, "model")

        tf.train.write_graph(minimal_graph, '.', 'minimal_graph.proto', as_text=False)
        tf.train.write_graph(minimal_graph, '.', 'minimal_graph.txt', as_text=True)

#  -------------Export Keras to Serving -----------------------------
# from keras import backend as K
#
# K.set_learning_phase(0)  # all new operations will be in test mode from now on
#
# # serialize the model and get its weights, for quick re-building
# config = previous_model.get_config()
# weights = previous_model.get_weights()
#
# # re-build a model where the learning phase is now hard-coded to 0
# from keras.models import model_from_config
# new_model = model_from_config(config)
# new_model.set_weights(weights)
#
# from tensorflow_serving.session_bundle import exporter
#
# export_path = ... # where to save the exported graph
# export_version = ... # version number (integer)
#
# saver = tf.train.Saver(sharded=True)
# model_exporter = exporter.Exporter(saver)
# signature = exporter.classification_signature(input_tensor=model.input,
#                                               scores_tensor=model.output)
# model_exporter.init(sess.graph.as_graph_def(),
#                     default_graph_signature=signature)
# model_exporter.export(export_path, tf.constant(export_version), sess)

# ----------------Working Example of graph converted to .pb ------------------------------------------

# with tf.Graph().as_default():
#     images = tf.placeholder(dtype=tf.float32, shape=[None, 299, 299, 3], name='inputs')
#     labels = tf.placeholder(dtype=tf.int32, shape=[None])
#
#     # Number of classes in the Dataset label set plus 1.
#     # Label 0 is reserved for an (unused) background class.
#     num_classes = dataset.num_classes() + 1
#
#     logits, softmax_weights, convoluted = inception.inference(images, num_classes)
#
#     final_tensor = tf.nn.softmax(logits, name='final_result')
#     prediction = tf.argmax(final_tensor, 1)
#     top_2_op = tf.nn.in_top_k(logits, labels, 2)
#     activation_map = heat_map(convoluted, softmax_weights, labels)
#
#     saver = tf.train.Saver()
#     # Build an initialization operation to run below.
#     init = tf.initialize_all_variables()
#     sess = tf.Session()
#     sess.run(init)
#
#     assert tf.gfile.Exists(FLAGS.pretrained_model_checkpoint_path)
#     saver.restore(sess, FLAGS.pretrained_model_checkpoint_path)
#     print('%s: Pre-trained model restored from %s' %
#           (datetime.now(), FLAGS.pretrained_model_checkpoint_path))
#
#     output_graph_def = graph_util.convert_variables_to_constants(
#         sess, sess.graph.as_graph_def(), ['final_result'])
#     with gfile.FastGFile(FLAGS.output_graph, 'wb') as f:
#         f.write(output_graph_def.SerializeToString())
