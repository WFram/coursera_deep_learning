import sys

import numpy as np
import matplotlib.pyplot as plt

from typing import Tuple, Dict, Callable, List, Union
from pathlib import Path

sys.path.append(str(Path('modules').resolve()))
from common.activations import sigmoid, ReLU


def dropout(activations : np.ndarray, keep_probability : float) -> Tuple[np.ndarray, np.ndarray]:
    mask = (np.random.rand(activations.shape[0], activations.shape[1]) < keep_probability).astype(int)
    activations = activations * mask / keep_probability
    return activations, mask


def forward_propagation(X : np.ndarray, parameters : Dict[str, np.ndarray],
                        keep_probability : float = 1.0) -> Tuple[np.ndarray, Tuple[np.ndarray]]:
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    W3 = parameters["W3"]
    b3 = parameters["b3"]
    
    # TODO: make a normal loop
    # TODO: care about precision for a condition to apply dropout
    np.random.seed(1)
    z1 = np.dot(W1, X) + b1
    a1 = ReLU(z1)
    if keep_probability < 1.0:
        a1, d1 = dropout(a1, keep_probability)
    z2 = np.dot(W2, a1) + b2
    a2 = ReLU(z2)
    if keep_probability < 1.0:
        a2, d2 = dropout(a2, keep_probability)
    z3 = np.dot(W3, a2) + b3
    a3 = sigmoid(z3)

    cache = (z1, a1, W1, b1, z2, a2, W2, b2, z3, a3, W3, b3)
    if keep_probability < 1.0:
        cache = (z1, d1, a1, W1, b1, z2, d2, a2, W2, b2, z3, a3, W3, b3)
    
    return a3, cache


def regularization_derivative(weights : np.ndarray, m : int, regularizer : float):
    return regularizer * weights / m


def backward_propagation(X : np.ndarray, Y : np.ndarray, cache : Tuple[np.ndarray],
                         regularizer : float = 0.0, keep_probability : float = 1.0) -> Dict[str, np.ndarray]:
    m = X.shape[1]
    if keep_probability < 1.0:
        _, d1, a1, W1, _, _, d2, a2, W2, _, _, a3, W3, _ = cache
    else:
        _, a1, W1, _, _, a2, W2, _, _, a3, W3, _ = cache
    
    dz3 = (a3 - Y)
    dW3 = 1./m * np.dot(dz3, a2.T)
    db3 = 1./m * np.sum(dz3, axis=1, keepdims = True)
    
    da2 = np.dot(W3.T, dz3)
    if keep_probability < 1.0:
       da2 = da2 * d2 / keep_probability     
    dz2 = np.multiply(da2, np.int64(a2 > 0))
    dW2 = 1./m * np.dot(dz2, a1.T)
    db2 = 1./m * np.sum(dz2, axis=1, keepdims = True)
    
    da1 = np.dot(W2.T, dz2)
    if keep_probability < 1.0:
       da1 = da1 * d1 / keep_probability
    dz1 = np.multiply(da1, np.int64(a1 > 0))
    dW1 = 1./m * np.dot(dz1, X.T)
    db1 = 1./m * np.sum(dz1, axis=1, keepdims = True)

    dW1 += regularization_derivative(W1, m, regularizer)
    dW2 += regularization_derivative(W2, m, regularizer)
    dW3 += regularization_derivative(W3, m, regularizer)
    
    gradients = {"dz3": dz3, "dW3": dW3, "db3": db3,
                 "da2": da2, "dz2": dz2, "dW2": dW2, "db2": db2,
                 "da1": da1, "dz1": dz1, "dW1": dW1, "db1": db1}
    
    return gradients


def update_parameters(parameters : Dict[str, np.ndarray], grads : Dict[str, np.ndarray], learning_rate : float) -> Dict[str, np.ndarray]:
    L = len(parameters) // 2 

    for k in range(L):
        parameters["W" + str(k+1)] = parameters["W" + str(k+1)] - learning_rate * grads["dW" + str(k+1)]
        parameters["b" + str(k+1)] = parameters["b" + str(k+1)] - learning_rate * grads["db" + str(k+1)]
        
    return parameters


def compute_loss(a3 : np.ndarray, Y : np.ndarray) -> float:
    m = Y.shape[1]
    logprobs = np.multiply(-np.log(a3),Y) + np.multiply(-np.log(1 - a3), 1 - Y)
    loss = 1./m * np.nansum(logprobs)
    
    return loss


def predict(X : np.ndarray, y : np.ndarray, parameters : Dict[str, np.ndarray]):
    m = X.shape[1]
    p = np.zeros((1, m), dtype = int)
    
    a3, _ = forward_propagation(X, parameters)
    
    for i in range(0, a3.shape[1]):
        if a3[0,i] > 0.5:
            p[0,i] = 1
        else:
            p[0,i] = 0

    print("Accuracy: "  + str(np.mean((p[0,:] == y[0,:]))))
    
    return p


def model(X : np.ndarray, Y : np.ndarray, initialization : Callable[[List[int]], Dict[str, Union[np.ndarray, float]]],
          learning_rate : float = 0.01, num_iterations : int = 15000, regularizer : float = 0.0,
          keep_probability : float = 1.0, print_cost : bool = True):        
    grads : Dict[str, np.ndarray] = {}
    costs : List[float] = []
    layers_dims = [X.shape[0], 20, 3, 1]
    
    parameters = initialization(layers_dims)

    for i in range(0, num_iterations):
        a3, cache = forward_propagation(X, parameters, keep_probability)
        cost = compute_loss(a3, Y)
        grads = backward_propagation(X, Y, cache, regularizer=regularizer, keep_probability=keep_probability)
        parameters = update_parameters(parameters, grads, learning_rate)
        
        if print_cost and i % 1000 == 0:
            print("Cost after iteration {}: {}".format(i, cost))
            costs.append(cost)
            
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
    return parameters