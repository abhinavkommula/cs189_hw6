"""
Author: Sophia Sanborn
Institution: UC Berkeley
Date: Spring 2020
Course: CS189/289A
Website: github.com/sophiaas
"""

import numpy as np
from abc import ABC, abstractmethod


class Activation(ABC):
    """Abstract class defining the common interface for all activation methods."""

    def __call__(self, Z):
        return self.forward(Z)

    @abstractmethod
    def forward(self, Z):
        pass


def initialize_activation(name: str) -> Activation:
    """Factory method to return an Activation object of the specified type."""
    if name == "linear":
        return Linear()
    elif name == "sigmoid":
        return Sigmoid()
    elif name == "tanh":
        return TanH()
    elif name == "arctan":
        return ArcTan()
    elif name == "relu":
        return ReLU()
    elif name == "softmax":
        return SoftMax()
    else:
        raise NotImplementedError("{} activation is not implemented".format(name))


class Linear(Activation):
    def __init__(self):
        super().__init__()

    def forward(self, Z: np.ndarray) -> np.ndarray:
        """Forward pass for f(z) = z.
        
        Parameters
        ----------
        Z  input pre-activations (any shape)

        Returns
        -------
        f(z) as described above applied elementwise to `Z`
        """
        return Z

    def backward(self, Z: np.ndarray, dY: np.ndarray) -> np.ndarray:
        """Backward pass for f(z) = z.
        
        Parameters
        ----------
        Z   input to `forward` method
        dY  derivative of loss w.r.t. the output of this layer
            same shape as `Z`

        Returns
        -------
        derivative of loss w.r.t. input of this layer
        """
        return dY


class Sigmoid(Activation):
    def __init__(self):
        super().__init__()

    def forward(self, Z: np.ndarray) -> np.ndarray:
        """Forward pass for sigmoid function:
        f(z) = 1 / (1 + exp(-z))
        
        Parameters
        ----------
        Z  input pre-activations (any shape)

        Returns
        -------
        f(z) as described above applied elementwise to `Z`
        """
        ### YOUR CODE HERE ###
        return ...

    def backward(self, Z: np.ndarray, dY: np.ndarray) -> np.ndarray:
        """Backward pass for sigmoid.
        
        Parameters
        ----------
        Z   input to `forward` method
        dY  derivative of loss w.r.t. the output of this layer
            same shape as `Z`

        Returns
        -------
        derivative of loss w.r.t. input of this layer
        """
        ### YOUR CODE HERE ###
        return ...


class TanH(Activation):
    def __init__(self):
        super().__init__()

    def forward(self, Z: np.ndarray) -> np.ndarray:
        """Forward pass for f(z) = tanh(z).
        
        Parameters
        ----------
        Z  input pre-activations (any shape)

        Returns
        -------
        f(z) as described above applied elementwise to `Z`
        """
        return 2 / (1 + np.exp(-2 * Z)) - 1

    def backward(self, Z: np.ndarray, dY: np.ndarray) -> np.ndarray:
        """Backward pass for f(z) = tanh(z).
        
        Parameters
        ----------
        Z   input to `forward` method
        dY  derivative of loss w.r.t. the output of this layer

        Returns
        -------
        derivative of loss w.r.t. input of this layer
        """
        fn = self.forward(Z)
        return dY * (1 - fn ** 2)


class ReLU(Activation):
    def __init__(self):
        super().__init__()

    def forward(self, Z: np.ndarray) -> np.ndarray:
        """Forward pass for relu activation:
        f(z) = z if z >= 0
               0 otherwise
        
        Parameters
        ----------
        Z  input pre-activations (any shape)

        Returns
        -------
        f(z) as described above applied elementwise to `Z`
        """
        ### YOUR CODE HERE ###    
        return np.maximum(Z, 0)

    def backward(self, Z: np.ndarray, dY: np.ndarray) -> np.ndarray:
        """Backward pass for relu activation.
        
        Parameters
        ----------
        Z   input to `forward` method
        dY  derivative of loss w.r.t. the output of this layer
            same shape as `Z`

        Returns
        -------
        derivative of loss w.r.t. input of this layer
        """
        ### YOUR CODE HERE ###
        mult = (Z >= 0).astype(int)
        res = np.multiply(dY, mult)
        
        return res
"""
ReLU OUTPUT:

test_backward (tests.test_activations.TestReLU.test_backward) .../opt/homebrew/Cellar/python@3.11/3.11.2_1/Frameworks/Python.framework/Versions/3.11/lib/python3.11/unittest/case.py:678: DeprecationWarning: It is deprecated to return a value that is not None from a test case (<bound method TestReLU.test_backward of <tests.test_activations.TestReLU testMethod=test_backward>>)
  return self.run(*args, **kwds)
ok
test_forward (tests.test_activations.TestReLU.test_forward) ... /opt/homebrew/Cellar/python@3.11/3.11.2_1/Frameworks/Python.framework/Versions/3.11/lib/python3.11/unittest/case.py:678: DeprecationWarning: It is deprecated to return a value that is not None from a test case (<bound method TestReLU.test_forward of <tests.test_activations.TestReLU testMethod=test_forward>>)
  return self.run(*args, **kwds)
ok

----------------------------------------------------------------------
Ran 2 tests in 0.016s

OK
"""


class SoftMax(Activation):
    def __init__(self):
        super().__init__()

    def forward(self, Z: np.ndarray) -> np.ndarray:
        """Forward pass for softmax activation.
        Hint: The naive implementation might not be numerically stable.
        
        Parameters
        ----------
        Z  input pre-activations (any shape)

        Returns
        -------
        f(z) as described above applied elementwise to `Z`
        """
        large = np.max(Z, axis = 1)
        res = np.exp(Z - large.reshape(-1, 1)) / np.sum(np.exp(Z - large.reshape(-1, 1)), axis=1).reshape(-1, 1)
        
        return res

    def backward(self, Z: np.ndarray, dY: np.ndarray) -> np.ndarray:
        """Backward pass for softmax activation.
        
        Parameters
        ----------
        Z   input to `forward` method
        dY  derivative of loss w.r.t. the output of this layer
            same shape as `Z`

        Returns
        -------
        derivative of loss w.r.t. input of this layer
        """
        res = np.zeros(Z.shape)
        newZ = self.forward(Z)
        
        for i in range(0, newZ.shape[0]):
            J = np.matmul(-newZ[i, :][:, None], newZ[i, :][:, None].T)
            new_ar = np.array([s * (1 - s) for s in newZ[i, :][:, None]])
 
            np.fill_diagonal(J, new_ar)
            res[i, :][:, None] = np.matmul(J, dY[i, :][:, None])
        
        return res

"""
SoftMax OUTPUT:

test_backward (tests.test_activations.TestSoftMax.test_backward) ... /opt/homebrew/Cellar/python@3.11/3.11.2_1/Frameworks/Python.framework/Versions/3.11/lib/python3.11/unittest/case.py:678: DeprecationWarning: It is deprecated to return a value that is not None from a test case (<bound method TestSoftMax.test_backward of <tests.test_activations.TestSoftMax testMethod=test_backward>>)
  return self.run(*args, **kwds)
ok
test_forward (tests.test_activations.TestSoftMax.test_forward) ... /opt/homebrew/Cellar/python@3.11/3.11.2_1/Frameworks/Python.framework/Versions/3.11/lib/python3.11/unittest/case.py:678: DeprecationWarning: It is deprecated to return a value that is not None from a test case (<bound method TestSoftMax.test_forward of <tests.test_activations.TestSoftMax testMethod=test_forward>>)
  return self.run(*args, **kwds)
ok

----------------------------------------------------------------------
Ran 2 tests in 0.297s

OK
"""
    
    
class ArcTan(Activation):
    def __init__(self):
        super().__init__()

    def forward(self, Z):
        return np.arctan(Z)

    def backward(self, Z, dY):
        return dY * 1 / (Z ** 2 + 1)
