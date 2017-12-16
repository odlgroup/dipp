# Copyright 2014-2017 The DIPP contributors
#
# This file is part of DIPP.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Custom add-ons for PyTorch.

See http://pytorch.org/.
"""

__all__ = ('functions', 'modules')

try:
    import torch
except ImportError:
    raise ImportError('to use `dipp/pytorch`, you need to install the '
                      '`pytorch` package either '
                      'from conda: `conda install pytorch`, '
                      'or from source, see http://pytorch.org/ for details')

from .functions import *
__all__ += functions.__all__

from .modules import *
__all__ += modules.__all__

from .utils import *
__all__ += utils.__all__
