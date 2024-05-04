import sys

import numpy as np
import matplotlib.pyplot as plt

from typing import Tuple, Dict, Union, List, Callable
from pathlib import Path
from math import ceil

import test_cases

sys.path.append(str(Path('modules').resolve()))
from common.activations import sigmoid, ReLU, sigmoid_derivative, ReLU_derivative
from common.costs import cross_entropy


def linear_forward(A: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    assert A.shape[0] == W.shape[1]
    
    Z = W @ A + b
    
    assert Z.shape == (W.shape[0], A.shape[1])
    
    return Z


def linear_activation_forward(A0: np.ndarray, W: np.ndarray, b: np.ndarray,
                              activation : Callable[[np.ndarray], np.ndarray]) -> Tuple[np.ndarray,
                                                                                        Dict[str, np.ndarray]]:
    Z = linear_forward(A0, W, b)
    A = activation(Z)

    cache = {"A0": A0, "W": W, "b": b, "Z": Z}

    return A, cache


def initialize_parameters(input_layer_dim: int, hidden_layer_dim : int, output_layer_dim: int) -> Dict[str, Union[np.ndarray, float]]:
    np.random.seed(1)
    W1 = np.random.randn(hidden_layer_dim, input_layer_dim) / np.sqrt(input_layer_dim)
    b1 = np.zeros((hidden_layer_dim, 1))
    W2 = np.random.randn(output_layer_dim, hidden_layer_dim) / np.sqrt(hidden_layer_dim)
    b2 = np.zeros((output_layer_dim, 1))

    assert W1.shape == (hidden_layer_dim, input_layer_dim)
    assert b1.shape == (hidden_layer_dim, 1)
    assert W2.shape == (output_layer_dim, hidden_layer_dim)
    assert b2.shape == (output_layer_dim, 1)

    parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}

    return parameters


def initialize_parameters_deep(layer_dims: List[int]):
    parameters : Dict[str, Union[np.ndarray, float]] = {}
    np.random.seed(1)
    n = len(layer_dims)

    for i in range(1, n):
        parameters[f"W{i}"] = np.random.randn(layer_dims[i], layer_dims[i - 1]) / np.sqrt(layer_dims[i - 1])
        parameters[f"b{i}"] = np.zeros((layer_dims[i], 1))

        assert parameters[f"W{i}"].shape == (layer_dims[i], layer_dims[i - 1])
        assert parameters[f"b{i}"].shape == (layer_dims[i], 1)

    return parameters


def model_forward(data: np.ndarray, parameters: Dict[str, np.ndarray]) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    n = len(parameters) // 2
    cache : List[Dict[str, np.ndarray]] = []
    
    A = data

    for i in range(1, n + 1):
        A, linear_forward_cache = linear_activation_forward(A, parameters[f"W{i}"], parameters[f"b{i}"], ReLU if i < n else sigmoid)
        cache.append(linear_forward_cache)

    assert A.shape == (1, data.shape[1]), f"{A.shape=}"

    return A, cache


def linear_backward(dJdZ : np.ndarray, cache: Dict[str, np.ndarray]):
    A0 = cache["A0"]
    W = cache["W"]
    b = cache["b"]

    m = A0.shape[1]

    dJdW = dJdZ @ A0.T / m
    dJdb = np.sum(dJdZ, axis=1, keepdims=True) / m
    dJdA0 = W.T @ dJdZ
    
    assert (dJdA0.shape == A0.shape)
    assert (dJdW.shape == W.shape)
    assert (dJdb.shape == b.shape)

    return dJdA0, dJdW, dJdb


def linear_activation_backward(dJdA : np.ndarray, cache: Dict[str, np.ndarray], activation : Callable[[np.ndarray], np.ndarray]) -> Tuple[np.ndarray]:
    Z = cache["Z"]
    dJdZ = activation(Z, dJdA)

    dJdA0, dJdW, dJdb = linear_backward(dJdZ, cache)

    return dJdA0, dJdW, dJdb


def model_backward(predictions : np.ndarray, labels : np.ndarray, cache : List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    gradients : Dict[str, np.ndarray] = {}
    n = len(cache)
    dJdA = -(np.divide(labels, predictions) - np.divide(1 - labels, 1 - predictions))
    dJdAprev, dJdW, dJdb = linear_activation_backward(dJdA, cache[-1], sigmoid_derivative)

    gradients[f"dJdA{n}"] = dJdA
    gradients[f"dJdA{n - 1}"] = dJdAprev
    gradients[f"dJdW{n}"] = dJdW
    gradients[f"dJdb{n}"] = dJdb

    for i in range(n - 1, 0, -1):
        dJdAprev, dJdW, dJdb = linear_activation_backward(gradients[f"dJdA{i}"], cache[i - 1], ReLU_derivative)
        gradients[f"dJdA{i - 1}"] = dJdAprev
        gradients[f"dJdW{i}"] = dJdW
        gradients[f"dJdb{i}"] = dJdb

    return gradients


def update_parameters(parameters: Dict[str, np.ndarray], gradients: Dict[str, np.ndarray], lr: float) -> Dict[str, np.ndarray]:
    assert len(gradients) % 3 == 1
    n = len(parameters) // 2

    for i in range(n, 0, -1):
        W = parameters[f"W{i}"]
        b = parameters[f"b{i}"]
        parameters[f"W{i}"] = W - lr * gradients[f"dJdW{i}"]
        parameters[f"b{i}"] = b - lr * gradients[f"dJdb{i}"]

    return parameters 


if __name__ == '__main__':
    parameters, grads = test_cases.update_parameters_test_case()
    parameters = update_parameters(parameters, grads, 0.1)

    print ("W1 = "+ str(parameters["W1"]))
    print ("b1 = "+ str(parameters["b1"]))
    print ("W2 = "+ str(parameters["W2"]))
    print ("b2 = "+ str(parameters["b2"]))