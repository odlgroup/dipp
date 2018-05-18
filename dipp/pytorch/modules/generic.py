# Copyright 2017,2018 The DIPP contributors
#
# This file is part of DIPP.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Basic pytorch modules for general-purpose use."""

import torch
from torch import nn

__all__ = ('Reshape', 'Flatten', 'InvSigmoid')


class Reshape(nn.Module):

    """Module for reshaping a tensor along non-batch axes."""

    def __init__(self, shape_out):
        """Initialize a new instance.

        Parameters
        ----------
        shape_out : sequence of int
            Desired shape along the non-batch axes.

        Examples
        --------
        >>> reshape = Reshape(shape_out=(2, 3))
        >>> y = reshape(torch.ones((1, 6)))
        >>> y
        tensor([[[ 1.,  1.,  1.],
                 [ 1.,  1.,  1.]]])
        >>> y.shape
        torch.Size([1, 2, 3])
        """
        super(Reshape, self).__init__()
        self.shape_out = shape_out

    def forward(self, x):
        return x.reshape(((x.shape[0],) + self.shape_out))


class Flatten(Reshape):

    """Module for flattening along the non-batch axes."""

    def __init__(self):
        """Initialize a new instance.

        Examples
        --------
        >>> flatten = Flatten()
        >>> y = flatten(torch.ones((1, 2, 3)))
        >>> y
        tensor([[ 1.,  1.,  1.,  1.,  1.,  1.]])
        >>> y.shape
        torch.Size([1, 6])
        """
        super(Flatten, self).__init__(shape_out=(-1,))


class InvSigmoid(nn.Module):

    r"""Pointwise :math:`s^{-1}(x) = \log(x / (1-x))`."""

    def __init__(self, a=None):
        """Initialize a new instance.

        Examples
        --------
        >>> invsig = InvSigmoid()
        >>> xvals = [1 / (1 + np.exp(-1)), 1 / (1 + np.exp(-3))]
        >>> x = torch.Tensor(xvals)
        >>> invsig(x)  # should be [1, 3]
        tensor([ 1.0000,  3.0000])
        """
        super(InvSigmoid, self).__init__()

    def forward(self, x):
        return torch.log(x / (1 - x))


if __name__ == '__main__':
    from dipp.util.testutils import run_doctests
    import numpy as np
    extraglobs = {'np': np}
    run_doctests(extraglobs=extraglobs)
