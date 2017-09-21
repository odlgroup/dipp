# Copyright 2014-2017 The DIPP contributors
#
# This file is part of DIPP.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Test configuration file."""

from __future__ import print_function, division, absolute_import

import numpy as np
import os

import dipp
try:
    import torch
except ImportError:
    PYTORCH_AVAILABLE = False
else:
    PYTORCH_AVAILABLE = True

try:
    import tensorflow
except ImportError:
    TENSORFLOW_AVAILABLE = False
else:
    TENSORFLOW_AVAILABLE = True


try:
    from pytest import fixture
except ImportError:
    # Make fixture the identity decorator
    def fixture(*args, **kwargs):
        if len(args) == 1:
            return args[0]
        else:
            return fixture


# --- Add numpy and dipp to all doctests --- #


@fixture(autouse=True)
def add_doctest_modules(doctest_namespace):
    """Make some modules available by default in doctests."""
    doctest_namespace['np'] = np

    if PYTORCH_AVAILABLE:
        from torch import autograd, nn
        doctest_namespace['pytorch'] = dipp.pytorch
        doctest_namespace['torch'] = torch
        doctest_namespace['autograd'] = autograd
        doctest_namespace['nn'] = nn

    if TENSORFLOW_AVAILABLE:
        doctest_namespace['tf'] = tensorflow


# --- Ignored tests due to missing modules --- #

this_dir = os.path.dirname(__file__)
root = os.path.abspath(os.path.join(this_dir, os.pardir, os.pardir))
collect_ignore = [os.path.join(root, 'setup.py')]

if not PYTORCH_AVAILABLE:
    collect_ignore.append(os.path.join(root, 'dipp', 'pytorch'))

if not TENSORFLOW_AVAILABLE:
    collect_ignore.append(os.path.join(root, 'dipp', 'tensorflow'))


# Add example directories to `collect_ignore`
def find_example_dirs():
    dirs = []
    for dirpath, dirnames, _ in os.walk(root):
        if 'examples' in dirnames:
            dirs.append(os.path.join(dirpath, 'examples'))
    return dirs


collect_ignore.extend(find_example_dirs())


# Remove duplicates
collect_ignore = list(set(collect_ignore))
collect_ignore = [os.path.normcase(ignored) for ignored in collect_ignore]


def pytest_ignore_collect(path, config):
    normalized = os.path.normcase(str(path))
    return any(normalized.startswith(ignored) for ignored in collect_ignore)
