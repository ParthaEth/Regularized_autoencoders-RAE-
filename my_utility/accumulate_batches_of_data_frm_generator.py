import numpy as np


def get_n_batches_of_input(n, gen):
    for i in range(n):
        if i == 0:
            data = gen.next()[0]
        else:
            data = np.concatenate((data, gen.next()[0]), axis=0)

    return data