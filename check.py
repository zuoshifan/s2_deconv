# Build a simple classification NN
import numpy as np
import tensorflow as tf
import tensorflow.keras
import nnhealpix.layers

# Take NSIDE=32 maps as input, and go through NSIDE=8 maps
NSIDE_INPUT = 2
NSIDE_OUTPUT = 1

shape = (12 * 2**2, 1)
num_classes = 2


# x = np.arange(np.prod(shape)).reshape(shape)
# y = nnhealpix.layers.ConvNeighbours(NSIDE_INPUT, filters=32, kernel_size=9)(x)
# print(y.shape)


inputs =tf.keras.layers.Input(shape)

x = nnhealpix.layers.ConvNeighbours(NSIDE_INPUT, filters=32, kernel_size=9)(inputs)
x = tf.keras.layers.Activation('relu')(x)
x = nnhealpix.layers.MaxPooling(NSIDE_INPUT, NSIDE_OUTPUT)(x)
x = tf.keras.layers.Dropout(0.2)(x)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(128)(x)
x = tf.keras.layers.Activation('relu')(x)
x = tf.keras.layers.Dense(num_classes)(x)
out = tf.keras.layers.Activation('softmax')(x)

model = tf.keras.models.Model(inputs=inputs, outputs=out)
model.compile(loss=tf.keras.losses.mse, optimizer='adam', metrics=['accuracy'])
model.summary()