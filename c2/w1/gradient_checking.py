import sys

import numpy as np
import matplotlib.pyplot as plt

from typing import Tuple, Dict, Union, List, Callable
from pathlib import Path
from math import ceil
from copy import deepcopy

import test_cases

sys.path.append(str(Path('modules').resolve()))
from common.activations import sigmoid, ReLU, sigmoid_derivative, ReLU_derivative
from common.costs import cross_entropy


def dictionary_to_vector(parameters : Dict[str, np.ndarray]) -> Tuple[np.ndarray, List[str]]:
    keys : List[str] = []
    
    count = 0

    for key in ["W1", "b1", "W2", "b2", "W3", "b3"]:
        vector = parameters[key].reshape(-1, 1).squeeze()
        keys += [key] * vector.shape[0]
        theta = vector if count == 0 else np.vstack((theta, vector))
        count += 1

    return theta, keys


def vector_to_dictionary(theta : np.ndarray) -> Dict[str, np.ndarray]:
    parameters : Dict[str, np.ndarray] = {}
    parameters["W1"] = theta[:20].reshape((5, 4))
    parameters["b1"] = theta[20:25].reshape((5, 1))
    parameters["W2"] = theta[25:40].reshape((3, 5))
    parameters["b2"] = theta[40:43].reshape((3, 1))
    parameters["W3"] = theta[43:46].reshape((1, 3))
    parameters["b3"] = theta[46:47].reshape((1, 1))

    return parameters


def gradients_to_vector(gradients) -> np.ndarray:
    count = 0
    for key in ["dW1", "db1", "dW2", "db2", "dW3", "db3"]:
        vector = np.reshape(gradients[key], (-1, 1))
        theta = vector if count == 0 else np.concatenate((theta, vector), axis=0)
        count += 1

    return theta


def forward(x : float, theta : float) -> float:
    J = theta * x
    return J


def backward(x : float) -> float:
    dJdtheta = x
    return dJdtheta


def gradient_check(x : float, theta : float, eps : float = 1e-7) -> float:
    J_right = forward(x, theta + eps)
    J_left = forward(x, theta - eps)
    gradapprox = (J_right - J_left) / 2 / eps
    grad = backward(x)
    diff = np.linalg.norm(grad - gradapprox) / (np.linalg.norm(grad) + np.linalg.norm(gradapprox))
    
    print("Gradient is correct") if diff < 1e-7 else print("Gradient is wrong")

    return diff


def forward_propagation_n(X : np.ndarray, Y : np.ndarray, parameters : Dict[str, np.ndarray]) -> Tuple[float, Dict[str, np.ndarray]]:
    m = X.shape[1]

    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    W3 = parameters["W3"]
    b3 = parameters["b3"]

    Z1 = W1 @ X + b1
    A1 = ReLU(Z1)
    Z2 = W2 @ A1 + b2
    A2 = ReLU(Z2)
    Z3 = W3 @ A2 + b3
    A3 = sigmoid(Z3)

    logprobs = cross_entropy(A3, Y)
    cost = np.sum(logprobs) / m
    
    cache = (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3)
   
    return cost, cache


def backward_propagation_n(X : np.ndarray, Y : np.ndarray, cache : Dict[str, np.ndarray]) -> Dict[str, Union[float, np.ndarray]]:
    m = X.shape[1]

    _, A1, _, _, _, A2, W2, _, _, A3, W3, _ = cache
    
    dZ3 = A3 - Y
    dW3 = dZ3 @ A2.T / m
    db3 = np.sum(dZ3, axis=1, keepdims=True) / m
    
    dA2 = W3.T @ dZ3
    print(f"{dA2.shape=}\t{A2.shape=}")
    dZ2 = dA2 * np.int64(A2 > 0)
    dW2 = dZ2 @ A1.T / m
    db2 = np.sum(dZ2, axis=1, keepdims=True) / m
    
    dA1 = W2.T @ dZ2
    print(f"{dA1.shape=}\t{A1.shape=}")
    dZ1 = dA1 * np.int64(A1 > 0)
    dW1 = dZ1 @ X.T
    db1 = np.sum(dZ1, axis=1, keepdims=True) / m
    
    gradients = {"dZ3": dZ3, "dW3": dW3, "db3": db3,
                 "dA2": dA2, "dZ2": dZ2, "dW2": dW2, "db2": db2,
                 "dA1": dA1, "dZ1": dZ1, "dW1": dW1, "db1": db1}
    
    return gradients


def gradient_check_n(x : np.ndarray, y : np.ndarray, parameters : Dict[str, np.ndarray], gradients : Dict[str, Union[float, np.ndarray]],
                     eps : float = 1e-7) -> float:
    num_parameters = len(parameters)
    gradients_values = gradients_to_vector(gradients)
    J_left = np.zeros((num_parameters, 1))
    J_right = np.zeros((num_parameters, 1))
    gradients_approx_values = np.zeros((num_parameters, 1))

    for i, key in enumerate(["W1", "b1", "W2", "b2", "W3", "b3"]):
        upd_parameters = deepcopy(parameters)
        upd_parameters[key] += eps
        J_right[i, :], _ = forward_propagation_n(x, y, upd_parameters)
        upd_parameters[key] -= (2 * eps)
        J_left[i, :], _ = forward_propagation_n(x, y, upd_parameters)
        gradients_approx_values[i, :] = (J_right[i, :] - J_left[i, :]) / 2 / eps
    
    '''
    This method is incorrect. If parameter is a matrix, than we need to compute a gradient per element.
    Then, compare with a gradient from backprop.
    '''

    calculate_diff = lambda x, y : np.linalg.norm(x - y) / (np.linalg.norm(x) + np.linalg.norm(y))
    differences = np.array(list(map(calculate_diff, gradients_values, gradients_approx_values)))

    print("Gradient is correct") if np.all(differences) < 2e-7 else print("Gradient is wrong")

    return differences


if __name__ == '__main__':
    X, Y, parameters = test_cases.gradient_check_n_test_case()

    cost, cache = forward_propagation_n(X, Y, parameters)
    gradients = backward_propagation_n(X, Y, cache)
    difference = gradient_check_n(X, Y, parameters, gradients)