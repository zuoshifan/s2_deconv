import numpy as np
import h5py
import healpy as hp
from nnhealpix.layers import OrderMap
from tensorflow.keras.models import model_from_json

import matplotlib
matplotlib.use('Agg')
try:
    # for new version matplotlib
    import matplotlib.style as mstyle
    mstyle.use('classic')
except ImportError:
    pass
import matplotlib.pyplot as plt


with h5py.File('./train_data/map_merge_new.hdf5', 'r') as f:
    hpmap = f['map'][0, 0, :]

print(hpmap.shape)

hpmap_nside64 = hp.ud_grade(hpmap, 64)
print(hpmap_nside64.shape)


# # plot hpmap_nside64
# fig = plt.figure(1, figsize=(13, 5))
# hp.mollview(hpmap_nside64, fig=1, title='', min=0, max=50)
# hp.graticule()
# fig.savefig('./train_data/map_merge_new.png')
# fig.clf()


epoch = 99
net = 'unet1'
model_weight_dir = './%s_result' % net

# load model
with open('%s.json' % net, 'r') as f:
    json_string = f.read()

model = model_from_json(json_string, custom_objects={'OrderMap': OrderMap})
# model.summary()

model.load_weights(model_weight_dir + '/model_weights_%04d.h5' % epoch)
predict = model.predict(hpmap_nside64.reshape(1, -1, 1))
print(predict.shape)


# plot predict
fig = plt.figure(1, figsize=(13, 5))
# hp.mollview(predict[0, :, 0], fig=1, title='', min=0, max=50)
hp.mollview(predict[0, :, 0], fig=1, title='', min=0, max=100)
hp.graticule()
fig.savefig('./train_data/map_merge_new_predict.png')
fig.clf()
