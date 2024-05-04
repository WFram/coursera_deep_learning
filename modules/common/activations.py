from typing import Union

import numpy as np


def sigmoid(Z : Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    A = 1 / (1 + np.exp(-Z))
    return A


def ReLU(Z : Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    return np.maximum(0, Z)

def sigmoid_derivative(Z : Union[float, np.ndarray], dJdA: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    dAdZ = np.exp(-Z) / ((1 + np.exp(-Z)) ** 2)
    dJdZ = dJdA * dAdZ
    return dJdZ

def ReLU_derivative(Z : Union[float, np.ndarray], dJdA: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    A = ReLU(Z)
    ids = np.where(A > 0)
    dAdZ = np.copy(A)
    dAdZ[ids] = 1
    dJdZ = dJdA * dAdZ
    return dJdZ
    