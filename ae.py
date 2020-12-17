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


normalization = True

input_file = './train_data/train_data.hdf5'
if normalization:
    output_dir = './ae_normalization_result'
else:
    output_dir = './ae_result'

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

if normalization:
    train_mean = np.mean(train1, axis=(1, 2))
    val_mean = np.mean(val1, axis=(1, 2))
    test_mean = np.mean(test1, axis=(1, 2))
    train_std = np.std(train1, axis=(1, 2))
    val_std = np.std(val1, axis=(1, 2))
    test_std = np.std(test1, axis=(1, 2))

    train1 = (train1 - train_mean[:, np.newaxis, np.newaxis]) / train_std[:, np.newaxis, np.newaxis]
    train2 = (train2 - train_mean[:, np.newaxis, np.newaxis]) / train_std[:, np.newaxis, np.newaxis]
    val1 = (val1 - val_mean[:, np.newaxis, np.newaxis]) / val_std[:, np.newaxis, np.newaxis]
    val2 = (val2 - val_mean[:, np.newaxis, np.newaxis]) / val_std[:, np.newaxis, np.newaxis]
    test1 = (test1 - test_mean[:, np.newaxis, np.newaxis]) / test_std[:, np.newaxis, np.newaxis]
    test2 = (test2 - test_mean[:, np.newaxis, np.newaxis]) / test_std[:, np.newaxis, np.newaxis]


# autoencoder
nsample, npix, _ = train1.shape
nside = hp.npix2nside(npix)
input_shape = (npix, 1)

inputs = tf.keras.layers.Input(input_shape)

conv1 = nnhealpix.layers.ConvNeighbours(nside, filters=32, kernel_size=9)(inputs)
relu1 = tf.keras.layers.Activation('relu')(conv1)
pool1 = nnhealpix.layers.MaxPooling(nside, nside//2)(relu1)
conv2 = nnhealpix.layers.ConvNeighbours(nside//2, filters=64, kernel_size=9)(pool1)
relu2 = tf.keras.layers.Activation('relu')(conv2)
pool2 = nnhealpix.layers.MaxPooling(nside//2, nside//4)(relu2)

conv3 = nnhealpix.layers.ConvNeighbours(nside//4, filters=64, kernel_size=9)(pool2)
relu3 = tf.keras.layers.Activation('relu')(conv3)
up3 = nnhealpix.layers.UpSampling(nside//4, nside//2)(relu3)
conv4 = nnhealpix.layers.ConvNeighbours(nside//2, filters=32, kernel_size=9)(up3)
relu4 = tf.keras.layers.Activation('relu')(conv4)
up4 = nnhealpix.layers.UpSampling(nside//2, nside)(relu4)

conv5 = tf.keras.layers.Conv1D(1, kernel_size=1)(up4)
# outputs = tf.keras.layers.Activation('relu')(conv5)
outputs = tf.keras.layers.Activation('linear')(conv5)


model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
model.summary()


# save model
json_string = model.to_json()
with open('ae.json', 'w') as f:
        f.write(json_string)


def crop_mse_loss(theta1=0.0, theta2=np.radians(120.0)):
    def crop_mse(y_true, y_pred):
        nside = hp.npix2nside(y_true.shape[1])
        ipix = hp.query_strip(nside, theta1, theta2, inclusive=True, nest=False, buff=None)
        y_true_crop = tf.gather(y_true, ipix, axis=1)
        y_pred_crop = tf.gather(y_pred, ipix, axis=1)
        # print (y_true_crop.shape)
        # print (y_pred_crop.shape)
        return tf.keras.backend.mean(tf.keras.backend.square(y_pred_crop - y_true_crop))
    return crop_mse


# model.compile(loss=tf.keras.losses.mse, optimizer='adam', metrics=['accuracy'])
# model.compile(loss=tf.keras.losses.mse, optimizer='adam')
model.compile(loss=crop_mse_loss(), optimizer='adam')


batch_size = 32
# epochs = 100
epochs = 1

loss = []
val_loss = []
for ii in range(100):

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
    if normalization:
        predict = predict * test_std[0] + test_mean[0]

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
