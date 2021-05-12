import numpy as np
from scipy.stats import pearsonr
import h5py
import healpy as hp

from simple_metrics import peak_signal_noise_ratio
from simple_metrics import mean_squared_error
from structural_similarity import structural_similarity


fg_file1 = '/tianlai/kfyu/machine_learning/tianlai_cyl_nn/750_maps/fg_750.0_num_200_0.hdf5'
# fg_file2 = '/tianlai/kfyu/machine_learning/tianlai_cyl_nn/750_maps/fg_750.0_num_200_1.hdf5'
# fg_file3 = '/tianlai/kfyu/machine_learning/tianlai_cyl_nn/750_maps/fg_750.0_num_200_2.hdf5'
# fg_file4 = '/tianlai/kfyu/machine_learning/tianlai_cyl_nn/750_maps/fg_750.0_num_200_3.hdf5'
# fg_file5 = '/tianlai/kfyu/machine_learning/tianlai_cyl_nn/750_maps/fg_750.0_num_200_4.hdf5'

cm_file1 = '/tianlai/kfyu/machine_learning/tianlai_cyl_nn/750_maps/21cm_750.0_num_200_0.hdf5'
# cm_file2 = '/tianlai/kfyu/machine_learning/tianlai_cyl_nn/750_maps/21cm_750.0_num_200_1.hdf5'
# cm_file3 = '/tianlai/kfyu/machine_learning/tianlai_cyl_nn/750_maps/21cm_750.0_num_200_2.hdf5'
# cm_file4 = '/tianlai/kfyu/machine_learning/tianlai_cyl_nn/750_maps/21cm_750.0_num_200_3.hdf5'
# cm_file5 = '/tianlai/kfyu/machine_learning/tianlai_cyl_nn/750_maps/21cm_750.0_num_200_4.hdf5'

with h5py.File(fg_file1, 'r') as f1, h5py.File(cm_file1, 'r') as f2:
    # input_map1 = f1['map'][:] + f2['map'][:]
    in_map = f1['map'][0, 0, :] + f2['map'][0, 0, :]

nside_in_map = hp.npix2nside(len(in_map))



# fl = './tk_deconv_map/deconv_map_iter3000_loop0.5.hdf5'
fl = './tk_deconv_map/deconv_map_iter20000_loop0.5.hdf5'
with h5py.File(fl, 'r') as f:
    clean_map = f['clean_map'][0, 0]
    res_map = f['residual_map'][0, 0]
    tk_map = clean_map + res_map

nside_tk_map = hp.npix2nside(len(tk_map))

in_map = hp.ud_grade(in_map, nside_tk_map)
# tk_map = hp.ud_grade(tk_map, nside_in_map)

print(in_map.shape, tk_map.shape)

# # plot in_map and tk_map for check
# import matplotlib
# matplotlib.use('Agg')
# try:
#     # for new version matplotlib
#     import matplotlib.style as mstyle
#     mstyle.use('classic')
# except ImportError:
#     pass
# import matplotlib.pyplot as plt


# fig = plt.figure(1, figsize=(13, 5))
# hp.mollview(in_map, fig=1, title='', min=0, max=50)
# hp.graticule()
# fig.savefig('in_map_check.png')
# fig.clf()

# fig = plt.figure(1, figsize=(13, 5))
# hp.mollview(tk_map, fig=1, title='', min=0, max=50)
# hp.graticule()
# fig.savefig('tk_map_check.png')
# fig.clf()




# crop before measure computation
nside = hp.npix2nside(in_map.shape[0])
theta1 = 0.0
theta2 = np.radians(120.0)
ipix = hp.query_strip(nside, theta1, theta2, inclusive=True, nest=False, buff=None)
in_map = in_map[ipix]
tk_map = tk_map[ipix]

# compute mse
# l = np.mean((in_map - rec_map)**2)
l = mean_squared_error(in_map, tk_map)
l /= (nside_tk_map / 64)**2 # normalize mse to compare with neural network resuts

# compute pearson r
r, p = pearsonr(in_map.flatten(), tk_map.flatten())

# compute psnr
p = peak_signal_noise_ratio(in_map, tk_map, data_range=in_map.max()-in_map.min())

# compute ssim
s = structural_similarity(in_map, tk_map)


print ( '%6s %6s %6s %6s %6s' % ('model', 'MSE', 'r', 'PSNR', 'SSIM') )
print ( '%6s %6.2f %6.2f %6.2f %6.2f' % ('tk', l, r, p, s) )