import numpy as np


class Tensor:

    def __init__(self, data):
        ''' Initial implementation for 2D tensors, numpy-based
        '''
        self._data = np.array(data)
        self._shape = self._data.shape
        self._grad = np.zeros(shape=self._shape)
        self._parents = set()
        self._backward_func = lambda: None
        
    def __add__(self, other):
        assert isinstance(other, Tensor)
        result = Tensor(self._data + other._data)
        result._parents.add(self)
        result._parents.add(other)

        def _addition_backward():
            self._grad += result._grad
            print(self)
            other._grad += result._grad
            print(other)

        result._backward_func = _addition_backward

        return result

    def __mul__(self, other):
        assert isinstance(other, Tensor)
        assert self._shape[1] == other._shape[0]
        result = Tensor(np.matmul(self._data, other._data))
        result._parents.add(self)
        result._parents.add(other)

        def _multiplication_backward():
            self._grad += np.matmul(result._grad, other._data.T)
            print(self)

            other._grad += np.matmul(self._data.T, result._grad)
            print(other)

        result._backward_func = _multiplication_backward

        return result
    
    def mean(self, dim=None):
        assert isinstance(dim, int) or dim is None
        assert

    def __pow__(self, other):
        pass

    def backward(self):
        self._grad = np.ones(shape=self._shape)
        visited = set()

        def _bckwrd(v):
            if v not in visited:
                visited.add(v)
                v._backward_func()
                for w in v._parents:
                    _bckwrd(w)

        _bckwrd(self)

    def __repr__(self):
        return f"eelmiTensor, data={self._data}, grad={self._grad}"

