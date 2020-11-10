# Copyright 2014-2017 The DIPP contributors
#
# This file is part of DIPP.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

import numpy as np
import tensorflow as tf
from dipp.tensorflow.activation import prelu, leaky_relu
from dipp.tensorflow.layers import (conv1d, conv2d,
                                    conv1dtransp, conv2dtransp,
                                    maxpool1d, maxpool2d)


__all__ = ('unet',)


def unet(x, nout,
         features=64,
         keep_prob=1.0,
         use_batch_norm=True,
         activation='relu',
         is_training=True,
         init='he',
         depth=4,
         name='unet'):
    """Reference implementation of the original U-net.

    All defaults are according to the reference article:

    https://arxiv.org/abs/1505.04597

    Parameters
    ----------
    x : `tf.Tensor` with shape ``(B, L, C)`` or ``(B, H, W, C)``
        The input vector/image
    nout : int
        Number of output channels.
    features : int, optional
        Number of features at the finest resultion.
    keep_prob : float in [0, 1], optional
        Used for dropout.
    use_batch_norm : bool, optional
        Wether batch normalization should be used.
    activation : {'relu', 'elu', 'leaky_relu', 'prelu'}, optional
        Activation function to use.
    is_training : bool or `tf.Tensor` with dtype bool, optional
        Flag indicating if training is currently done.
        Needed for batch normalization.
    init : {'he', 'xavier'}, optional
        Initialization scheme for the weights. Biases are initialized to zero.
    depth : positive int, optional
        Number of downsamplings that should be done.
    name : str, optional
        Name of the created layer.

    Returns
    -------
    unet : `tf.Tensor` with shape ``(B, L, nout)`` or ``(B, H, W, nout)``

    Examples
    --------
    Create 2d unet

    >>> data = np.array([[1, 2, 3],
    ...                  [4, 5, 6],
    ...                  [7, 8, 9]])
    >>> x = tf.constant(data[None, ..., None])  # add empty batch and channel
    >>> y = unet(x, 1)
    >>> y.shape
    TensorShape([Dimension(1), Dimension(3), Dimension(3), Dimension(1)])
    """
    x = tf.cast(x, 'float32')
    ndim = len(x.shape) - 2

    assert depth >= 1

    def get_weight_bias(nin, nout, transpose, size):
        if transpose:
            shape = [size] * ndim + [nout, nin]
        else:
            shape = [size] * ndim + [nin, nout]

        b_shape = [1] * (1 + ndim) + [nout]

        if init == 'xavier':
            stddev = np.sqrt(2.6 / (size ** ndim * (nin + nout)))
        elif init == 'he':
            stddev = np.sqrt(2.6 / (size ** ndim * nin))

        w = tf.Variable(tf.truncated_normal(shape, stddev=stddev))
        b = tf.Variable(tf.constant(0.0, shape=b_shape))

        return w, b

    def apply_activation(x):
        if activation == 'relu':
            return tf.nn.relu(x)
        elif activation == 'elu':
            return tf.nn.elu(x)
        elif activation == 'leaky_relu':
            return leaky_relu(x)
        elif activation == 'prelu':
            return prelu(x)
        else:
            raise RuntimeError('unknown activation')

    def apply_conv(x, nout,
                   stride=False,
                   size=3,
                   disable_batch_norm=False,
                   disable_dropout=False,
                   disable_activation=False):

        if stride:
            if ndim == 1:
                stride = 2
            elif ndim == 2:
                stride = (2, 2)
        else:
            if ndim == 1:
                stride = 1
            elif ndim == 2:
                stride = (1, 1)

        with tf.name_scope('apply_conv'):
            nin = int(x.get_shape()[-1])

            w, b = get_weight_bias(nin, nout, transpose=False, size=size)

            if ndim == 1:
                out = conv1d(x, w, stride=stride) + b
            elif ndim == 2:
                out = conv2d(x, w, stride=stride) + b

            if use_batch_norm and not disable_batch_norm:
                out = tf.contrib.layers.batch_norm(out,
                                                   is_training=is_training)
            if keep_prob != 1.0 and not disable_dropout:
                out = tf.contrib.layers.dropout(out, keep_prob=keep_prob,
                                                is_training=is_training)

            if not disable_activation:
                out = apply_activation(out)

            return out

    def apply_convtransp(x, nout,
                         stride=True, out_shape=None,
                         size=2,
                         disable_batch_norm=False,
                         disable_dropout=False,
                         disable_activation=False):

        if stride:
            if ndim == 1:
                stride = 2
            elif ndim == 2:
                stride = (2, 2)
        else:
            if ndim == 1:
                stride = 1
            elif ndim == 2:
                stride = (1, 1)

        with tf.name_scope('apply_convtransp'):
            nin = int(x.get_shape()[-1])

            w, b = get_weight_bias(nin, nout, transpose=True, size=size)

            if ndim == 1:
                out = conv1dtransp(x, w, stride=stride, out_shape=out_shape) + b
            elif ndim == 2:
                out = conv2dtransp(x, w, stride=stride, out_shape=out_shape) + b

            if use_batch_norm and not disable_batch_norm:
                out = tf.contrib.layers.batch_norm(out,
                                                   is_training=is_training)
            if keep_prob != 1.0 and not disable_dropout:
                out = tf.contrib.layers.dropout(out, keep_prob=keep_prob,
                                                is_training=is_training)

            if not disable_activation:
                out = apply_activation(out)

            return out

    def apply_maxpool(x):
        if ndim == 1:
            return maxpool1d(x)
        else:
            return maxpool2d(x)

    finals = []

    with tf.name_scope('{}_call'.format(name)):
        with tf.name_scope('in'):
            current = apply_conv(x, features)
            current = apply_conv(current, features)
            finals.append(current)

        for layer in range(depth - 1):
            with tf.name_scope('down_{}'.format(layer + 1)):
                features_layer = 2 ** (layer + 1)
                current = apply_maxpool(current)
                current = apply_conv(current, features_layer)
                current = apply_conv(current, features_layer)
                finals.append(current)

        with tf.name_scope('coarse'):
            current = apply_maxpool(current)
            current = apply_conv(current, features * 2 ** depth)
            current = apply_conv(current, features * 2 ** depth)

        for layer in reversed(range(depth - 1)):
            with tf.name_scope('up_{}'.format(layer + 1)):
                features_layer = 2 ** (layer + 1)
                skip = finals.pop()
                current = apply_convtransp(current, features_layer,
                                           out_shape=tf.shape(skip),
                                           disable_activation=True)
                current = tf.concat([current, skip], axis=-1)

                current = apply_conv(current, features_layer)
                current = apply_conv(current, features_layer)

        with tf.name_scope('out'):
            skip = finals.pop()
            current = apply_convtransp(current, features,
                                       out_shape=tf.shape(skip),
                                       disable_activation=True)
            current = tf.concat([current, skip], axis=-1)

            current = apply_conv(current, features)
            current = apply_conv(current, features)

            current = apply_conv(current, nout,
                                 size=1,
                                 disable_activation=True,
                                 disable_batch_norm=True,
                                 disable_dropout=True)

    return current


if __name__ == '__main__':
    from dipp.util.testutils import run_doctests
    with tf.Session():
        run_doctests()
