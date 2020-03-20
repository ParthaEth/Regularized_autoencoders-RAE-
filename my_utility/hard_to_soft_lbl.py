import numpy as np
import keras.backend as K


def get_gaussian_kernel(sigma):
    length = 20
    xs = (np.array(range(length)) - (length / 2)) / sigma
    kernel = np.exp(-np.square(xs))
    kernel /= np.sum(kernel)
    kernel = np.reshape(kernel, (length, 1, 1))
    return K.variable(kernel)


def to_soft_label(x, kernel):
    original_shape = K.shape(x)
    x = K.reshape(x, (-1, original_shape[-1], 1))
    x = K.conv1d(x=x, kernel=kernel, strides=1, padding='same')
    x /= K.sum(x, axis=1, keepdims=True)
    return K.reshape(x, original_shape)
