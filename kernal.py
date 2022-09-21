#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf


def hyperNet(x_dim=3, y_dim=3, ch_in=64, ch_out=64):
    xx_range = tf.range(-(x_dim - 1) / 2, (x_dim + 1) / 2, dtype='float32')
    yy_range = tf.range(-(y_dim - 1) / 2, (y_dim + 1) / 2, dtype='float32')

    xx_range = tf.tile(tf.expand_dims(xx_range, -1), [1, y_dim])
    yy_range = tf.tile(tf.expand_dims(yy_range, 0), [x_dim, 1])

    xx_range = tf.expand_dims(xx_range, -1)
    yy_range = tf.expand_dims(yy_range, -1)

    pos = tf.concat([xx_range, yy_range], -1)

    pos = tf.expand_dims(pos, 0)

    return pos
