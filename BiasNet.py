#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from keras.layers import Layer


class BiasNet(Layer):
    def __init__(self, **kwargs):
        super(BiasNet, self).__init__(**kwargs)
        self.kernel = None

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',
                                      shape=(1, 1, input_shape[3]),
                                      initializer='he_normal',
                                      trainable=True)
        super(BiasNet, self).build(input_shape)

    def call(self, x, **kwargs):
        return x + self.kernel

    def compute_output_shape(self, input_shape):
        return input_shape
