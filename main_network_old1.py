from keras.models import Model
from keras.layers import merge,Input
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
import tensorflow as tf
from tensorflow.python.framework.graph_util import convert_variables_to_constants
import keras
from keras.preprocessing.image import ImageDataGenerator

from keras.layers import Convolution2D, MaxPooling2D, Dense

img_width, img_height = 150, 150

train_data_dir = '/home/glm/bankovky/train'

validation_data_dir = '/home/glm/bankovky/validace'
nb_train_samples = 224
nb_validation_samples = 80
nb_epoch = 1
step = 1

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1./255)

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


# input_img = Convolution2D(32, 3, 3, input_shape=(img_width, img_height, 3))
# to je vlastni model jenz se neslucuje ze Senquencialem hazi erorr
# The added layer must  be an instance  of class Layer.
#  Reseni z_embed = Reshape((1, target_embed_size))(output)

# input_img = Input(shape=(150, 150, 3))
# tower_1 = Convolution2D(64, 1, 1, border_mode='same', activation='relu')(input_img)
# tower_1 = Convolution2D(64, 3, 3, border_mode='same', activation='relu')(tower_1)
#
# tower_2 = Convolution2D(64, 1, 1, border_mode='same', activation='relu')(input_img)
# tower_2 = Convolution2D(64, 5, 5, border_mode='same', activation='relu')(tower_2)
#
# tower_3 = MaxPooling2D((3, 3), strides=(1, 1), border_mode='same')(input_img)
# tower_3 = Convolution2D(64, 1, 1, border_mode='same', activation='relu', name="dense_one")(tower_3)
# # predictions = Dense(1, activation='sigmoid')(tower_3)
# output = merge([tower_1, tower_2, tower_3], mode='concat', concat_axis=1)
# output.get_output_shape_at("dense_one") #index ?
# model = Model(input=input_img, output=output, name='konecne')
# print a  == model.output


# print model.get_weights()
# print model.get_config()
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
        myGraph =  tf.Graph()
        my= myGraph.as_default()

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
        with open("model.json", "w") as json_file:
                json_file.write(model_json)

        init_op = tf.variables_initializer(tf.global_variables(), name="nInit")
        # meta_graph_def = tf.train.export_meta_graph(filename='tmp/modelMeta.meta')
        minimal_graph = convert_variables_to_constants(sess, sess.graph_def, ["group_deps"])
        save(sess, '')

        saver = tf.train.Saver()
        save_path = saver.save(sess, "model")

        tf.train.write_graph(minimal_graph, '.', 'minimal_graph.proto', as_text=False)
        tf.train.write_graph(minimal_graph, '.', 'minimal_graph.txt', as_text=True)


# dostaneme vysledky z vrstvy konecne nefunguje nezna .output atribute >>>> output=model.get_layer(layer_name).output)
# AttributeError: 'NoneType' object has no attribute 'output
# layer_name = 'konecne'
# intermediate_layer_model = Model(input=model.input,
#                                  output=model.get_layer(layer_name).output)
# intermediate_output = intermediate_layer_model.predict(model)

# this embedding layer will encode the input sequence
# into a sequence of dense 512-dimensional vectors.


# a LSTM will transform the vector sequence into a single vector,
# containing information about the entire sequence
# lstm_out = LSTM(32)(x)
#
# auxiliary_loss = Dense(1, activation='sigmoid', name='aux_output')(lstm_out)
#
# auxiliary_input = Input(shape=(5,), name='aux_input')
# x = merge([lstm_out, auxiliary_input], mode='concat')
#
# # we stack a deep fully-connected network on top
# x = Dense(64, activation='relu', name="dense_one")(x) # names are added here
# x = Dense(64, activation='relu', name="dense_two")(x)
# x = Dense(64, activation='relu', name="dense_three")(x)
#
# # and finally we add the main logistic regression layer
# main_loss = Dense(1, activation='sigmoid', name='main_output')(x)
# model = Model(input=[main_input, auxiliary_input], output=[main_loss, auxiliary_loss])
# model.compile(optimizer='rmsprop', loss='binary_crossentropy',
#               loss_weights=[1., 0.2])
# model.get_layer("dense_one")
