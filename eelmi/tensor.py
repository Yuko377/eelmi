import numpy as np

from .utils import is_scalar

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
        if is_scalar(other):
            result = Tensor(self._data + other)

            def _addition_backward():
                self._grad += result._grad

            result._parents.add(self)
            result._backward_func = _addition_backward
            return result

        assert isinstance(other, Tensor)
        result = Tensor(self._data + other._data)
        result._parents.add(self)
        result._parents.add(other)

        def _addition_backward():
            self._grad += result._grad
            other._grad += result._grad

        result._backward_func = _addition_backward
        return result
    
    def __radd__(self, other):
        return self.__add__(other)
    
    def __mul__(self, other):
        if is_scalar(other):
            result = Tensor(self._data * other)

            def _multiplication_backward():
                self._grad += result._grad * other

            result._parents.add(self)
            result._backward_func = _multiplication_backward
            return result
        
        assert isinstance(other, Tensor)
        assert self._shape == other._shape

        result = Tensor(self._data * other._data)
        result._parents.add(self)
        result._parents.add(other)

        def _multiplication_backward():
            self._grad += result._grad * other._data
            other._grad += result._grad * self._data

        result._backward_func = _multiplication_backward
        return result

    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __truediv__(self, other):
        if is_scalar(other):
            result = Tensor(self._data / other)

            def _division_backward():
                self._grad += result._grad / other

            result._parents.add(self)
            result._backward_func = _division_backward
            return result
        
        assert isinstance(other, Tensor) and other._shape == (1, 1)

        result = Tensor(self._data / other._data)
        result._parents.add(self)
        result._parents.add(other)

        def _division_backward():
            self._grad += result._grad / other._data
            other._grad += np.sum(-1 * result._grad * self._data / (other._data ** 2))
        
        result._backward_func = _division_backward
        return result

    def __matmul__(self, other):        
        assert isinstance(other, Tensor)
        assert self._shape[1] == other._shape[0]
        result = Tensor(np.matmul(self._data, other._data))
        result._parents.add(self)
        result._parents.add(other)

        def _matmul_backward():
            self._grad += np.matmul(result._grad, other._data.T)
            other._grad += np.matmul(self._data.T, result._grad)

        result._backward_func = _matmul_backward
        return result
    
    def __rmatmul__(self, other):
        return other.__matmul__(self)
    
    def sum(self, dim=None):
        assert isinstance(dim, int) or dim is None
        if dim is None:
            result = Tensor(np.sum(self._data, keepdims=True))
            result._parents.add(self)

            def _sum_backward():
                self._grad += np.ones_like(self._data) * result._grad

            result._backward_func = _sum_backward
            return result

        assert 0 <= dim <= 1
        result = Tensor(np.sum(self._data, axis=dim, keepdims=True))
        result._parents.add(self)
        def _sum_backward():
            self._grad += np.repeat(result._grad, self._shape[dim], axis=dim)

        result._backward_func = _sum_backward
        return result
    
    def mean(self, dim=None):
        assert isinstance(dim, int) or dim is None
        if dim is None:
            return self.sum() / self._data.size

        assert 0 <= dim <= 1
        return self.sum(dim) / self._shape[dim]

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

