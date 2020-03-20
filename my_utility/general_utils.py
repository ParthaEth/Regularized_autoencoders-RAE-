import numpy as np


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / (np.sum(np.exp(x), axis=-1, keepdims=True) + 1e-11)