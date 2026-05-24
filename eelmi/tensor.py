import numpy as np


class Tensor:

    def __init__(self, data):
        ''' Initial implementation for 2D tensors, numpy-based
        '''
        self._shape = data.shape
        self._data = data
        self._grad = np.zeros(shape=self._shape)
        self._parents = set()
        self._backward_func = None
        
    def __add__(self, other):
        assert isinstance(other, Tensor)
        result = Tensor(self._data + other._data)
        result._parents.add(self)
        result._parents.add(other)

        def _addition_backward():
            self._grad += result._grad
            other._grad += result._grad

        self._backward_func = _addition_backward

        return result

    def __mul__(self, other):
        assert isinstance(other, Tensor)
        assert self._shape[1] == other._shape[0]
        result = Tensor(np.matmul(self._data, other._data))
        result._parents.add(self)
        result._parents.add(other)

        def _multiplication_backward():
            self._grad += np.matmul(result._grad, other._data.T)
            other._grad += np.matmul(self._data.T, result._grad)
        
        self._backward_func = _multiplication_backward

        return result

    def __pow__(self, other):
        pass

    def backward(self):
        self._grad = np.ones(shape=self._shape)