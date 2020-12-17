# Build a simple classification NN
import numpy as np
import tensorflow as tf
import tensorflow.keras
import nnhealpix.layers

# code to fix error: Could not create cudnn handle: CUDNN_STATUS_INTERNAL_ERROR
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.InteractiveSession(config=config)


input_shape = (1, 12, 1)
x = np.arange(np.prod(input_shape)).reshape(input_shape)
print(x)
y = nnhealpix.layers.UpSampling(1, 2)(x)
print(y)
y = nnhealpix.layers.AveragePooling(2, 1)(y)
# y = nnhealpix.layers.MaxPooling(2, 1)(y)
print(y)
