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
    in_map = f['input'][:].flatten()
    rec_map = f['reconstruction'][:].flatten()

print( in_map.shape )

nside = hp.npix2nside(in_map.shape[0])
theta1 = 0.0
theta2 = np.radians(120.0)
ipix = hp.query_strip(nside, theta1, theta2, inclusive=True, nest=False, buff=None)
print (ipix.shape)

in_map_crop = np.zeros_like(in_map)
in_map_crop[ipix] = in_map[ipix]
rec_map_crop = np.zeros_like(rec_map)
rec_map_crop[ipix] = rec_map[ipix]

# plot map
fig = plt.figure(1, figsize=(13, 5))
hp.mollview(in_map_crop, fig=1, title='', min=0, max=50)
hp.graticule()
fig.savefig('in_map_crop.png')
fig.clf()

fig = plt.figure(1, figsize=(13, 5))
hp.mollview(rec_map_crop, fig=1, title='', min=0, max=50)
hp.graticule()
fig.savefig('rec_map_crop.png')
fig.clf()