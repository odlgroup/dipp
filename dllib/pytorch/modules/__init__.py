# Copyright 2014-2017 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Predefined modules to be used in Neural Networks.

Modules are intended as layers in neural network architectures.
They can define learnable parameters that will automatically be updated
in the backpropagation pass by being marked as ``requires_grad=True``.

A `torch.nn.modules.module.Module` implements a ``forward()`` method for
evaluation that acts on `torch.autograd.variable.Variable` objects. Any
component used in this forward pass should support automatic differentiation,
which is always true for built-in functions in ``torch``.

If a custom function cannot be composed of such built-in functions, it
must be implemented as a `torch.autograd.function.Function` with custom
``forward()`` and ``backward()`` methods, in order to tell ``torch`` how
to perform the backward pass with this object.

See `the pytorch documentation on extending nn
<http://pytorch.org/docs/master/notes/extending.html#extending-torch-nn>`_
for further details.
"""

from __future__ import absolute_import

__all__ = ()

from .generic import *
__all__ += generic.__all__

from .haarpsi import *
__all__ += haarpsi.__all__
