import numpy as np


def RZZ(theta):
    return np.diag(np.exp(-1j * theta * np.array([1, -1, -1, 1]) / 2)).reshape([2] * 4)

def RX(theta):
    cos = np.cos(theta / 2)
    misin = -1j * np.sin(theta / 2)
    return np.array([[cos, misin], [misin, cos]])

def ZZ(i, j, n):
    i = n - i - 1
    j = n - j - 1
    return np.array([-1 if (((m >> i) & 1) == 1 and ((m >> j) & 1) == 1) else 1 for m in range(1 << n)])