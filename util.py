
from np import *
import os
import pickle

def to_cpu(x):
    import numpy
    if type(x) == numpy.ndarray:
        return x
    return np.asnumpy(x)


def to_gpu(x):
    import cupy
    if type(x) == cupy.ndarray:
        return x
    return cupy.asarray(x)


def get_vocab_size(file_name):
    if '/' in file_name:
        file_name = file_name.replace('/', os.sep)

    if not os.path.exists(file_name):
        raise IOError('No file: ' + file_name)

    with open(file_name, 'rb') as f:
        params = pickle.load(f)

    return params[0].shape[0]


