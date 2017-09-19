# Copyright 2014-2017 The DIPP contributors
#
# This file is part of DIPP.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Utilities for testing."""

__all__ = ('run_doctests',)


def run_doctests(skip_if=False, **kwargs):
    """Run all doctests in the current module.

    This function calls ``doctest.testmod()``, by default with the options
    ``optionflags=doctest.NORMALIZE_WHITESPACE`` and
    ``extraglobs={'np': np}``. This can be changed with keyword arguments.

    Parameters
    ----------
    skip_if : bool
        For ``True``, skip the doctests in this module.
    kwargs :
        Extra keyword arguments passed on to the ``doctest.testmod``
        function.
    """
    from doctest import testmod, NORMALIZE_WHITESPACE, SKIP
    from pkg_resources import parse_version
    import numpy as np
    import warnings

    optionflags = kwargs.pop('optionflags', NORMALIZE_WHITESPACE)
    if skip_if:
        optionflags |= SKIP

    extraglobs = kwargs.pop('extraglobs', {'np': np})

    if '__IPYTHON__' in globals():
        # Run from an IPython console
        try:
            import spyder
        except ImportError:
            pass
        else:
            if parse_version(spyder.__version__) < parse_version('3.1.4'):
                warnings.warn('A bug with IPython and Spyder < 3.1.4 '
                              'sometimes causes doctests to fail to run. '
                              'Please upgrade Spyder or use another '
                              'interpreter if the doctests do not work.',
                              RuntimeWarning)

    testmod(optionflags=optionflags, extraglobs=extraglobs, **kwargs)
