# Copyright 2014-2017 The DIPP contributors
#
# This file is part of DIPP.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Basic pytorch modules for general-purpose use."""

import torch
from torch import nn

__all__ = ('Logistic', 'InvLogistic')


class Logistic(nn.Module):

    """Pointwise :math:`l_a(x) = 1 / (1 + \exp(-ax))`."""

    def __init__(self, a=None):
        """Initialize a new instance.

        Parameters
        ----------
        a : positive float, optional
            The :math:`a` parameter in the function. If not provided, it
            is registered as a learnable `torch.nn.parameter.Parameter`.
            It is initialized randomly between 0.5 and 5.

        Examples
        --------
        >>> logi = Logistic(np.log(3))
        >>> x = autograd.Variable(torch.Tensor([0, 1]))
        >>> logi(x)  # should be [1/2, 3/4]
        Variable containing:
         0.5000
         0.7500
        [torch.FloatTensor of size 2]
        """
        super(Logistic, self).__init__()
        if a is None:
            self.a = nn.Parameter(torch.Tensor(1))
            self.a.data.uniform_(0.5, 5)
        else:
            assert a > 0
            self.a = float(a)

    def forward(self, x):
        return nn.functional.sigmoid(self.a * x)


class InvLogistic(nn.Module):

    """Pointwise :math:`l_a^{-1}(x) = (\log(x) - \log(1-x)) / a`."""

    def __init__(self, a=None):
        """Initialize a new instance.

        Parameters
        ----------
        a : positive float, optional
            The :math:`a` parameter in the function. If not provided, it
            is registered as a learnable `torch.nn.parameter.Parameter`.
            It is initialized randomly between 0.5 and 5.

        Examples
        --------
        >>> invlogi = InvLogistic(2)
        >>> xvals = [1 / (1 + np.exp(-1)), 1 / (1 + np.exp(-3))]
        >>> x = autograd.Variable(torch.Tensor(xvals))
        >>> invlogi(x)  # should be [1/2, 3/2]
        Variable containing:
         0.5000
         1.5000
        [torch.FloatTensor of size 2]
        """
        super(InvLogistic, self).__init__()
        if a is None:
            self.a = nn.Parameter(torch.Tensor(1))
            self.a.data.uniform_(0.5, 5)
        else:
            assert a > 0
            self.a = float(a)

    def forward(self, x):
        return (torch.log(x) - torch.log(1 - x)) / self.a


if __name__ == '__main__':
    from dipp.util.testutils import run_doctests
    from torch import autograd
    import numpy as np
    extraglobs = {'np': np, 'autograd': autograd}
    run_doctests(extraglobs=extraglobs)
