from keras.layers import Input, Embedding, LSTM, Dense, merge
from keras.models import Model


# headline input: meant to receive sequences of 100 integers, between 1 and 10000.
# note that we can name any layer by passing it a "name" argument.
main_input = Input(shape=(100,), dtype='int32', name='main_input')

# this embedding layer will encode the input sequence
# into a sequence of dense 512-dimensional vectors.
x = Embedding(output_dim=512, input_dim=10000, input_length=100)(main_input)

# a LSTM will transform the vector sequence into a single vector,
# containing information about the entire sequence
lstm_out = LSTM(32)(x)

auxiliary_loss = Dense(1, activation='sigmoid', name='aux_output')(lstm_out)

auxiliary_input = Input(shape=(5,), name='aux_input')
x = merge([lstm_out, auxiliary_input], mode='concat')

# we stack a deep fully-connected network on top
x = Dense(64, activation='relu', name="dense_one")(x) # names are added here
x = Dense(64, activation='relu', name="dense_two")(x)
x = Dense(64, activation='relu', name="dense_three")(x)

# and finally we add the main logistic regression layer
main_loss = Dense(1, activation='sigmoid', name='main_output')(x)

model = Model(input=[main_input, auxiliary_input], output=[main_loss, auxiliary_loss])

model.compile(optimizer='rmsprop', loss='binary_crossentropy',
              loss_weights=[1., 0.2])
model.get_layer("dense_one")
model.get_layer("dense_one")
