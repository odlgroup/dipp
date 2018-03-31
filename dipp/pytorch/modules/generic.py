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

__all__ = ('InvSigmoid',)


class InvSigmoid(nn.Module):

    """Pointwise :math:`s^{-1}(x) = \log(x / (1-x))`."""

    def __init__(self, a=None):
        """Initialize a new instance.

        Examples
        --------
        >>> invsig = InvSigmoid()
        >>> xvals = [1 / (1 + np.exp(-1)), 1 / (1 + np.exp(-3))]
        >>> x = autograd.Variable(torch.Tensor(xvals))
        >>> invsig(x)  # should be [1, 3]
        Variable containing:
         1.0000
         3.0000
        [torch.FloatTensor of size 2]
        """
        super(InvSigmoid, self).__init__()

    def forward(self, x):
        return torch.log(x / (1 - x))


if __name__ == '__main__':
    from dipp.util.testutils import run_doctests
    from torch import autograd
    import numpy as np
    extraglobs = {'np': np, 'autograd': autograd}
    run_doctests(extraglobs=extraglobs)
