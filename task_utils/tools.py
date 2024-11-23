import numpy as np


class dict2Obj(object):
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            setattr(self, key, value)

def batch_softmax(z):
    """
    Apply the softmax function over the last dimension for each batch instance.
    Input:
      - z: ndarray of shape (B, N)
    Output:
      - out: ndarray of shape (B, N) after applying softmax
    """
    exp_z = np.exp(z - np.max(z, axis=-1, keepdims=True))
    out = exp_z / exp_z.sum(axis=-1, keepdims=True)
    return out