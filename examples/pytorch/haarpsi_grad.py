"""Demonstration of the autograd support of the HaarPSI FOM.

This example shows how the HaarPSI figure of merit can be used as a
loss function with either fixed or learnable parameters.
"""

import numpy as np
import torch
from torch import autograd
from dipp import pytorch

# Make noise images with predictable randomness
np.random.seed(123)
image1_arr = np.random.uniform(0, 1, (128, 128))
image2_arr = np.random.uniform(0, 1, (128, 128))

# Create pytorch Variable objects from the images
image1 = autograd.Variable(torch.Tensor(image1_arr))
image2 = autograd.Variable(torch.Tensor(image2_arr))

# HaarPSI with fixed parameters
a = 4.2
c = 3.0
haarpsi = pytorch.HaarPSI(a, c)
print('Learnable parameters: ', len(list(haarpsi.parameters())))
score = haarpsi(image1, image2)
print('Score: ', score.data[0])

# HaarPSI with learnable parameters (`c` needs to be explicitly initialized)
haarpsi = pytorch.HaarPSI(init_c=3.0)
params = list(haarpsi.parameters())
print('Learnable parameters: ', len(params))
print('a =', params[0])  # random
print('c =', params[1])
score = haarpsi(image1, image2)
print('Score: ', score.data[0])

# Do backpropagation
score.backward()
print('After backprop:')
print('grad_a =', params[0].grad)
print('grad_c =', params[1].grad)
