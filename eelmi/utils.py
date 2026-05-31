import numpy as np

def is_scalar(value):
    return isinstance(value, (int, float, np.number))
