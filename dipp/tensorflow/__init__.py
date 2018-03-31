# Copyright 2017,2018 The DIPP contributors
#
# This file is part of DIPP.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Custom add-ons for tensorflow."""

__all__ = ('interpolation',)

try:
    import tensorflow
except ImportError:
    raise ImportError('to use `dipp/tensorflow`, you need to install the '
                      '`tensorflow` package either '
                      'from conda: `conda install tensorflow`, '
                      'or using pip: `pip install tensorflow`')

from .interpolation import *
__all__ += interpolation.__all__
