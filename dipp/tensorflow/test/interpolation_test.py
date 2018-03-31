# Copyright 2017,2018 The DIPP contributors
#
# This file is part of DIPP.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

import numpy as np
import tensorflow as tf
from dipp.tensorflow import bilinear_interpolation
import pytest


def test_sample_single_point():
    with tf.Session():
        data = np.array([[1, 2, 3],
                         [4, 5, 6],
                         [7, 8, 9]])
        img = tf.constant(data)[None, ..., None]  # add empty batch and channel

        for xi, x in enumerate([0, 0.5, 1]):
            for yi, y in enumerate([0, 0.5, 1]):
                result = bilinear_interpolation(img, [[[x]]], [[[y]]]).eval()
                assert result.shape == (1, 1, 1, 1)
                assert pytest.approx(result.squeeze()) == data[yi, xi]


def test_sample_interpolate():
    with tf.Session():
        data = np.array([[1, 2, 3],
                         [4, 5, 6],
                         [7, 8, 9]])
        img = tf.constant(data)[None, ..., None]  # add empty batch and channel

        x, y = 0.25, 0.25
        result = bilinear_interpolation(img, [[[x]]], [[[y]]]).eval()
        assert pytest.approx(result.squeeze()) == 3.0  # (1+2+4+5)/4

        x, y = 0.75, 0.25
        result = bilinear_interpolation(img, [[[x]]], [[[y]]]).eval()
        assert pytest.approx(result.squeeze()) == 4.0

        x, y = 0.25, 0.75
        result = bilinear_interpolation(img, [[[x]]], [[[y]]]).eval()
        assert pytest.approx(result.squeeze()) == 6.0

        x, y = 0.75, 0.75
        result = bilinear_interpolation(img, [[[x]]], [[[y]]]).eval()
        assert pytest.approx(result.squeeze()) == 7.0


def test_boundary_interpolate():
    """Validate that the "constant" boundary condition is working."""
    with tf.Session():
        data = np.array([[1, 2, 3],
                         [4, 5, 6],
                         [7, 8, 9]])
        img = tf.constant(data)[None, ..., None]  # add empty batch and channel

        x, y = 0.5, -0.5
        result = bilinear_interpolation(img, [[[x]]], [[[y]]], 'order0').eval()
        assert pytest.approx(result.squeeze()) == 2.0

        x, y = 0.5, -0.5
        result = bilinear_interpolation(img, [[[x]]], [[[y]]], 'order1').eval()
        assert pytest.approx(result.squeeze()) == -1.0


if __name__ == '__main__':
        pytest.main([str(__file__.replace('\\', '/')), '-v'])