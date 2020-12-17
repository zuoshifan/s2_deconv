import os
import numpy as np
import h5py
import healpy as hp
import tensorflow as tf
import tensorflow.keras
import nnhealpix.layers

import matplotlib
matplotlib.use('Agg')
try:
    # for new version matplotlib
    import matplotlib.style as mstyle
    mstyle.use('classic')
except ImportError:
    pass
import matplotlib.pyplot as plt


input_file = './train_data/train_data.hdf5'
output_dir = './unet_result'

if not os.path.isdir(output_dir):
    os.mkdir(output_dir)

with h5py.File(input_file, 'r') as f:
    in_map = f['input'][:]
    rec_map = f['reconstruction'][:]

in_map = np.repeat(in_map, 1000, axis=0).astype(np.float32)
rec_map = np.repeat(rec_map, 1000, axis=0).astype(np.float32)
print( in_map.shape )

# read in dataset
train1 = rec_map[:600]
val1 = rec_map[600:900]
test1 = rec_map[900:]

train2 = in_map[:600]
val2 = in_map[600:900]
test2 = in_map[900:]


# unet
nsample, npix, _ = train1.shape
nside = hp.npix2nside(npix)
input_shape = (npix, 1)

inputs = tf.keras.layers.Input(input_shape)

conv1 = nnhealpix.layers.ConvNeighbours(nside, filters=32, kernel_size=9)(inputs)
relu1 = tf.keras.layers.Activation('relu')(conv1)
conv1 = nnhealpix.layers.ConvNeighbours(nside, filters=32, kernel_size=9)(relu1)
relu1 = tf.keras.layers.Activation('relu')(conv1)
pool1 = nnhealpix.layers.MaxPooling(nside, nside//2)(relu1)

conv2 = nnhealpix.layers.ConvNeighbours(nside//2, filters=64, kernel_size=9)(pool1)
relu2 = tf.keras.layers.Activation('relu')(conv2)
conv2 = nnhealpix.layers.ConvNeighbours(nside//2, filters=64, kernel_size=9)(relu2)
relu2 = tf.keras.layers.Activation('relu')(conv2)
drop2 = tf.keras.layers.Dropout(0.5)(relu2)

up3 = nnhealpix.layers.UpSampling(nside//2, nside)(drop2)
merge3 = tf.keras.layers.Concatenate()([relu1, up3])
conv3 = nnhealpix.layers.ConvNeighbours(nside, filters=32, kernel_size=9)(merge3)
relu3 = tf.keras.layers.Activation('relu')(conv3)
conv3 = nnhealpix.layers.ConvNeighbours(nside, filters=32, kernel_size=9)(relu3)
relu3 = tf.keras.layers.Activation('relu')(conv3)

conv4 = tf.keras.layers.Conv1D(1, kernel_size=1)(conv3)
# outputs = tf.keras.layers.Activation('relu')(conv4)
outputs = tf.keras.layers.Activation('linear')(conv4)


model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
model.summary()


# save model
json_string = model.to_json()
with open('unet.json', 'w') as f:
        f.write(json_string)


model.compile(loss=tf.keras.losses.mse, optimizer='adam', metrics=['accuracy'])


batch_size = 32
# epochs = 100
epochs = 1

loss = []
val_loss = []
for ii in range(50):

    # es_cb = EarlyStopping(monitor='val_loss', patience=2, verbose=1, mode='auto')
    # chkpt = saveDir + 'AutoEncoder_Cifar10_denoise_weights.{epoch:02d}-{loss:.2f}-{val_loss:.2f}.hdf5'
    # cp_cb = ModelCheckpoint(filepath = chkpt, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

    history = model.fit(train1, train2,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_data=(val1, val2),
                        # callbacks=[es_cb,],
                        # callbacks=[es_cb, cp_cb],
                        shuffle=True)

    # save weights
    model.save_weights(output_dir + '/model_weights_%04d.h5' % ii)

    predict = model.predict(train1[0].reshape(1, npix, 1))

    # plot predict
    fig = plt.figure(1, figsize=(13, 5))
    hp.mollview(predict[0, :, 0], fig=1, title='', min=0, max=50)
    hp.graticule()
    fig.savefig(output_dir + '/predict_%04d.png' % ii)
    fig.clf()

    # save loss
    loss.append(history.history['loss'][-1])
    val_loss.append(history.history['val_loss'][-1])

    # plot history for loss
    plt.figure()
    # plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    plt.plot(loss)
    plt.plot(val_loss)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['train', 'validation'], loc='upper right')
    plt.savefig(output_dir + '/loss_%04d.png' % ii)
    plt.close()


# save loss for plot
with h5py.File(output_dir + '/history_loss.hdf5', 'w') as f:
    f.create_dataset('loss', data=np.array(loss))
    f.create_dataset('val_loss', data=np.array(val_loss))
