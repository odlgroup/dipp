# Copyright 2014-2017 The DIPP contributors
#
# This file is part of DIPP.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

import tensorflow as tf


__all__ = ('bilinear_interpolation',)


def _pixel_value(img, x, y):
    """Sample image at coordinates.

    Parameters
    ----------
    img: `tf.Tensor` of shape (B, H, W, C)
        The image to interpolate
    x, y : `tf.Tensor` with shape (B, H2, W2)
        x and y, integer coordinates.

    Returns
    -------
    - output: `tf.Tensor` of shape (B, H2, W2, C)
    """
    B = tf.shape(img)[0]
    H2 = tf.shape(x)[1]
    W2 = tf.shape(x)[2]

    batch_idx = tf.range(0, B)
    batch_idx = tf.reshape(batch_idx, (B, 1, 1))
    b = tf.tile(batch_idx, (1, H2, W2))

    indices = tf.stack([b, y, x], 3)

    return tf.gather_nd(img, indices)


def bilinear_interpolation(img, x, y, boundary='order0'):
    """Bilinear sampling of `img`.

    Parameters
    ----------
    img: `tf.Tensor` with shape (B, H, W, C).
        The image to sample
    x: `tf.Tensor` with shape (B, H2, W2)
        x coordinate, normalized to fit in [0, 1].
    y: `tf.Tensor` with shape (B, H2, W2)
        y coordinate, normalized to fit in [0, 1].
    boundary : {'order0', 'order1'}
        Boundary behaviour. Order 0 means constant extrapolation and
        order 1 linear extrapolation.

    Returns
    -------
    bilinear_interpolation: `tf.Tensor` with shape (B, H2, W2, C).
        The resulting image, dtype is ``'float32'``.

    Examples
    --------
    Get value in upper left corner

    >>> data = np.array([[1, 2, 3],
    ...                  [4, 5, 6],
    ...                  [7, 8, 9]])
    >>> img = data[None, ..., None]  # add empty batch and channel
    >>> bilinear_interpolation(img, [[[0]]], [[[0]]]).eval()
    array([[[[ 1.]]]], dtype=float32)

    Get data back as is

    >>> x, y = np.meshgrid([0, 0.5, 1], [0, 0.5, 1])
    >>> x, y = x[None, ...], y[None, ...]  # Add empty batch
    >>> bilinear_interpolation(img, x, y).eval().squeeze()
    array([[ 1.,  2.,  3.],
           [ 4.,  5.,  6.],
           [ 7.,  8.,  9.]], dtype=float32)
    """
    H = tf.shape(img)[1]
    W = tf.shape(img)[2]

    max_y = tf.cast(H - 1, 'int32')
    max_x = tf.cast(W - 1, 'int32')

    # cast indices as float32 (for rescaling)
    x = tf.cast(x, 'float32')
    y = tf.cast(y, 'float32')
    img = tf.cast(img, 'float32')

    # rescale x and y to [0, W/H]
    x = x * tf.cast(W - 1, 'float32')
    y = y * tf.cast(H - 1, 'float32')

    # grab 4 nearest corner points for each (x_i, y_i)
    # i.e. we need a rectangle around the point of interest
    x0 = tf.cast(tf.floor(x), 'int32')
    x1 = x0 + 1
    y0 = tf.cast(tf.floor(y), 'int32')
    y1 = y0 + 1

    # clip to range [0, H/W] to not violate img boundaries
    if boundary == 'order0':
        x0_get = tf.clip_by_value(x0, 0, max_x)
        x1_get = tf.clip_by_value(x1, 0, max_x)
        y0_get = tf.clip_by_value(y0, 0, max_y)
        y1_get = tf.clip_by_value(y1, 0, max_y)
    elif boundary == 'order1':
        print(x0.eval(), x1.eval(), y0.eval(), y1.eval(),
              x.eval(), y.eval())
        x0_get = tf.clip_by_value(x0, 0, max_x - 1)
        x1_get = tf.clip_by_value(x1, 1, max_x)
        y0_get = tf.clip_by_value(y0, 0, max_y - 1)
        y1_get = tf.clip_by_value(y1, 1, max_y)
    else:
        raise ValueError('unknown `boundary`')

    # get pixel value at corner coords
    Ia = _pixel_value(img, x0_get, y0_get)
    Ib = _pixel_value(img, x0_get, y1_get)
    Ic = _pixel_value(img, x1_get, y0_get)
    Id = _pixel_value(img, x1_get, y1_get)

    # recast as float for delta calculation
    x0 = tf.cast(x0, 'float32')
    x1 = tf.cast(x1, 'float32')
    y0 = tf.cast(y0, 'float32')
    y1 = tf.cast(y1, 'float32')

    # calculate deltas
    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)

    # add dimension for addition
    wa = tf.expand_dims(wa, axis=-1)
    wb = tf.expand_dims(wb, axis=-1)
    wc = tf.expand_dims(wc, axis=-1)
    wd = tf.expand_dims(wd, axis=-1)

    # compute output
    out = tf.add_n([wa*Ia, wb*Ib, wc*Ic, wd*Id])

    return out

if __name__ == '__main__':
    from dipp.util.testutils import run_doctests
    with tf.Session():
        run_doctests()