"""Demonstration of the components of the HaarPSI figure of merit.

This example shows the individual steps of computing the HaarPSI FOM,
including comparison images.

Required additional packages:

    - ``scipy`` to generate the example image
    - ``matplotlib`` to display images
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.misc
import torch
from torch import autograd
from dipp import pytorch


true_image_arr = scipy.misc.ascent().astype('float32')

# Image with false structures
corrupt1_arr = true_image_arr.copy()
corrupt1_arr[100:110, 50:350] = 0  # long narrow hole
corrupt1_arr[200:204, 60:64] = 180  # small square corrupted
corrupt1_arr[350:400, 200:250] = 200  # large square corrupted

# Noisy image, using a predictable outcome
np.random.seed(123)
corrupt2_arr = true_image_arr + np.random.uniform(0, 40, true_image_arr.shape)

# Show images
fig, ax = plt.subplots(ncols=3)
ax[0].imshow(true_image_arr, cmap='gray')
ax[0].set_title('original')
ax[1].imshow(corrupt1_arr, cmap='gray')
ax[1].set_title('corrupted')
ax[2].imshow(corrupt2_arr, cmap='gray')
ax[2].set_title('noisy')
fig.suptitle('Input images for comparison')
fig.tight_layout()
fig.show()

# Create pytorch Variable objects from the images
true_image = autograd.Variable(torch.Tensor(true_image_arr))
corrupt1 = autograd.Variable(torch.Tensor(corrupt1_arr))
corrupt2 = autograd.Variable(torch.Tensor(corrupt2_arr))

# Compute local similarity images
a = 4.2
c = 100
lsim_ax0 = pytorch.modules.haarpsi.HaarPSISimilarityMap(axis=0, a=a, c=c)
lsim_ax1 = pytorch.modules.haarpsi.HaarPSISimilarityMap(axis=1, a=a, c=c)

sim1_ax0 = lsim_ax0(corrupt1, true_image)
sim1_ax1 = lsim_ax1(corrupt1, true_image)
sim2_ax0 = lsim_ax0(corrupt2, true_image)
sim2_ax1 = lsim_ax1(corrupt2, true_image)

# Show similarity images
fig, ax = plt.subplots(ncols=2)
ax[0].imshow(sim1_ax0.data.numpy())
ax[0].set_title('axis 0')
ax[1].imshow(sim1_ax1.data.numpy())
ax[1].set_title('axis 1')
fig.suptitle('Similarity maps for corrupted image')
fig.tight_layout()
fig.show()

fig, ax = plt.subplots(ncols=2)
ax[0].imshow(sim2_ax0.data.numpy())
ax[0].set_title('axis 0')
ax[1].imshow(sim2_ax1.data.numpy())
ax[1].set_title('axis 1')
fig.suptitle('Similarity maps for noisy image')
fig.tight_layout()
fig.show()

# Compute similarity weight maps
wmap_ax0 = pytorch.modules.haarpsi.HaarPSIWeightMap(axis=0)
wmap_ax1 = pytorch.modules.haarpsi.HaarPSIWeightMap(axis=1)

weight1_ax0 = wmap_ax0(corrupt1, true_image)
weight1_ax1 = wmap_ax1(corrupt1, true_image)
weight2_ax0 = wmap_ax0(corrupt2, true_image)
weight2_ax1 = wmap_ax1(corrupt2, true_image)

# Show weight maps
fig, ax = plt.subplots(ncols=2)
ax[0].imshow(weight1_ax0.data.numpy())
ax[0].set_title('axis 0')
ax[1].imshow(weight1_ax1.data.numpy())
ax[1].set_title('axis 1')
fig.suptitle('Weight maps for corrupted image')
fig.tight_layout()
fig.show()

fig, ax = plt.subplots(ncols=2)
ax[0].imshow(weight2_ax0.data.numpy())
ax[0].set_title('axis 0')
ax[1].imshow(weight2_ax1.data.numpy())
ax[1].set_title('axis 1')
fig.suptitle('Weight maps for noisy image')
fig.tight_layout()
fig.show()


# Compute HaarPSI scores
haarpsi = pytorch.HaarPSI(a, c)

score1 = haarpsi(corrupt1, true_image)
score2 = haarpsi(corrupt2, true_image)

print('Similarity score (a = {}, c = {}) for corrupted image: {}'
      ''.format(a, c, score1.data[0]))
print('Similarity score (a = {}, c = {}) for noisy image: {}'
      ''.format(a, c, score2.data[0]))
