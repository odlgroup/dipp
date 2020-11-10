# Copyright 2014-2017 The DIPP contributors
#
# This file is part of DIPP.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

import tensorflow as tf


__all__ = ('leaky_relu', 'prelu')


def leaky_relu(_x, alpha=0.2, name='leaky_relu'):
    return prelu(_x, init=alpha, name=name, trainable=False)


def prelu(_x, init=0.0, name='prelu', trainable=True):
    with tf.variable_scope(name):
        alphas = tf.get_variable('alphas',
                                 shape=[int(_x.get_shape()[-1])],
                                 initializer=tf.constant_initializer(init),
                                 dtype=tf.float32,
                                 trainable=True)
        pos = tf.nn.relu(_x)
        neg = -alphas * tf.nn.relu(-_x)

        return pos + neg
