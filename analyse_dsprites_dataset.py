import numpy as np
import matplotlib.pyplot as plt

loc = 'datasets/dsprites/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz'
dataset_zip = np.load(loc, encoding='latin1')
imgs = dataset_zip['imgs']
latent_values = dataset_zip['latents_values']

fig, ax = plt.subplots(2, 2)
ax[0, 0].imshow(imgs[0, :, :])
ax[0, 1].plot(latent_values[0, :], '.r')
ax[1, 0].imshow(imgs[10000, :, :])
ax[1, 1].plot(latent_values[10000, :], '.r')
plt.show()
