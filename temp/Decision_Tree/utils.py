import numpy as np
import scipy.stats as stats


def sq_loss(y):
    if len(y) != 0:
        y_bar = np.mean(y)
        return sum((y-y_bar)**2)
    else:
        return 0


def mis_math(y):
    y_hat = stats.mode(y)[0][0]
    return sum(y != y_hat)


def gini(y):
    n = len(y)
    if n == 0:
        return 0
    else:
        _, counts = np.unique(y, return_counts=True)
        T = 0
        for z in counts:
            T += z * (n - z)
        return T/n


def entropy(y):
    n = len(y)
    if n == 0:
        return 0
    _, counts = np.unique(y, return_counts=True)
    T = 0
    for z in counts:
        if z != 0:
            T += z * np.log(n/z)
    return T
