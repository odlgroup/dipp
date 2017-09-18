# Copyright 2014-2017 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Predefined functions implementing custom automatic differentiation.

Functions are intended to create objects that customize their forward
evaluation (``forward()``) as well as their automatic gradient
contribution (``backward()``).

Use this class if a function cannot be composed of predefined functions that
already support automatic differentiation, e.g., built-in ``torch``
functions.

The ``forward`` method should accept arbitrary Python objects as inputs and
convert them to tensors, including unwrapping of
`torch.autograd.variable.Variable` objects. The output should
be a tensor or a tuple of tensors.

The ``backward`` method always receives `torch.autograd.variable.Variable`
objects as inputs and should return as many variables as there are inputs to
``forward``.

See the `pytorch documentation on extending autograd
<http://pytorch.org/docs/master/notes/extending.html#\
extending-torch-autograd>`_ for further details.
"""

__all__ = ()

if __name__ == '__main__':
    from odl.util.testutils import run_doctests
    run_doctests()
