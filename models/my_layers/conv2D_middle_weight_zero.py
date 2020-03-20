from keras.engine.topology import Layer
from keras.constraints import Constraint
import tensorflow as tf
import numpy as np
import keras.backend as K


class ZeroCenteredConv(Constraint):
    def __init__(self):
        self.mask = None

    def __call__(self, w):
        if self.mask is None:
            filter_shape = w.shape
            self.mask = np.ones(shape=filter_shape, dtype=np.float32)
            self.mask[filter_shape[0] // 2, filter_shape[1] // 2, :, :] = 0

        w *= self.mask
        return w

    def get_config(self):
        return {}


class NoCentreWeightConv2D(Layer):
    def __init__(self, num_filters, kernel_size, padding, kernel_regularizer=None, **kwargs):
        super(NoCentreWeightConv2D, self).__init__(**kwargs)
        self.output_dim = num_filters
        self.filter_height, self.filter_width = kernel_size
        self.padding = padding
        self.kernel_regularizer=kernel_regularizer
        self.kernel_constraint = ZeroCenteredConv()

    def build(self, input_shape):
        self.filter_shape = [self.filter_height, self.filter_width, input_shape[-1], self.output_dim]
        initializer = tf.contrib.layers.xavier_initializer()
        self.W = self.add_weight(name='kernel', shape=self.filter_shape, initializer=initializer, trainable=True,
                                 regularizer=self.kernel_regularizer, constraint=self.kernel_constraint)

        self.B = self.add_weight(name='bias', shape=(self.output_dim,), initializer='uniform',
                                 trainable=True)

        super(NoCentreWeightConv2D, self).build(input_shape)

    def call(self, x, mask=None):
        # self.W *= self.mask
        return K.conv2d(x, self.W, strides=(1, 1), padding=self.padding) + self.B

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (self.output_dim, )

    def get_config(self):
        config = {
            'filters': self.output_dim,
            'kernel_size': (self.filter_height, self.filter_width),
            'strides': (1, 1),
            'padding': self.padding,
        }
        base_config = super(NoCentreWeightConv2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))