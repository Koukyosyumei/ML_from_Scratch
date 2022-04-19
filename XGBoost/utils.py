import numpy as np


def _sigmoid(x):
    sigmoid_range = 34.538776394910684

    if x <= -sigmoid_range:
        return 1e-15
    if x >= sigmoid_range:
        return 1.0 - 1e-15

    return 1.0 / (1.0 + np.exp(-x))


sigmoid = np.vectorize(_sigmoid)
