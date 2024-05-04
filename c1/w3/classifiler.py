import sys

import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegressionCV 
from typing import Tuple, Dict, Union, List, Callable
from pathlib import Path
from math import ceil

import test_cases

sys.path.append(str(Path('modules').resolve()))
from common.activations import sigmoid, ReLU
from common.costs import cross_entropy
from common.evaluation import compute_accuracy


def load_planar_dataset():
    num_examples : int = 400
    dims : int = 2
    points_per_class = int(num_examples / 2)

    X = np.zeros((num_examples, dims), dtype='float32')
    Y = np.zeros((num_examples, 1), dtype='uint8')

    np.random.seed(1)

    max_ray = 4
    for j in range(2):
        ix = range(points_per_class * j, points_per_class * (j + 1))
        theta = np.linspace(j * 3.12, (j + 1) * 3.12, points_per_class) + np.random.randn(points_per_class) * 0.2
        r = max_ray * np.sin(4 * theta) + np.random.randn(points_per_class) * 0.2
        X[ix] = np.c_[r * np.sin(theta), r * np.cos(theta)]
        Y[ix] = j
    
    X = X.T
    Y = Y.T

    return X, Y


def plot_decision_boundary(predict_functor : Callable[[np.ndarray], Union[np.ndarray, float]],
                           data : np.ndarray, labels : np.ndarray):
    x_min, x_max = data[0, :].min() - 1, data[0, :].max() + 1
    y_min, y_max = data[1, :].min() - 1, data[1, :].max() + 1
    distance = 0.01

    xx, yy = np.meshgrid(np.arange(x_min, x_max, distance), np.arange(y_min, y_max, distance))

    prediction = predict_functor(np.c_[xx.ravel(), yy.ravel()])
    prediction = prediction.reshape(xx.shape)

    plt.title('Logistic Regression')
    plt.contourf(xx, yy, prediction, cmap=plt.cm.Spectral)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(data[0, :], data[1, :], c=labels, cmap=plt.cm.Spectral)
    plt.savefig('dumped.png')


def layer_sizes(data: np.ndarray, labels: np.ndarray) -> Tuple[int, int, int]:
    assert data.shape[0] == 2
    assert labels.shape[0] == 1

    input_layer_dim = data.shape[0]
    output_layer_dim = labels.shape[0]
    hidden_layer_dim = 4
    
    return (input_layer_dim, hidden_layer_dim, output_layer_dim)


def initialize_parameters(input_layer_dim: int, hidden_layer_dim : int, output_layer_dim: int) -> Dict[str, np.ndarray]:
    np.random.seed(2)
    W1 = np.random.randn(hidden_layer_dim, input_layer_dim) * 0.01
    b1 = np.zeros((hidden_layer_dim, 1))
    W2 = np.random.randn(output_layer_dim, hidden_layer_dim) * 0.01
    b2 = np.zeros((output_layer_dim, 1))

    assert W1.shape == (hidden_layer_dim, input_layer_dim)
    assert b1.shape == (hidden_layer_dim, 1)
    assert W2.shape == (output_layer_dim, hidden_layer_dim)
    assert b2.shape == (output_layer_dim, 1)

    parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}

    return parameters


def forward_propagation(data : np.ndarray, parameters: Dict[str, np.ndarray]) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    assert data.shape[0] == 2

    Z1 = W1 @ data + b1
    A1 = np.tanh(Z1)
    Z2 = W2 @ A1 + b2
    A2 = sigmoid(Z2)

    assert A2.shape == (1, data.shape[1])

    cache = {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}

    return A2, cache


def backward_propagation(cache : Dict[str, np.ndarray],
                         parameters : Dict[str, np.ndarray],
                         data : np.ndarray,
                         labels : np.ndarray) -> Dict[str, np.ndarray]:
    m = data.shape[1]
    W2 = parameters["W2"]
    A1 = cache["A1"]
    A2 = cache["A2"]

    dJdZ2 = A2 - labels
    dJdW2 = dJdZ2 @ A1.T / m
    dJdb2 = np.sum(dJdZ2, axis=1, keepdims=True) / m
    dJdZ1 = W2.T @ dJdZ2 * (1 - np.power(A1, 2))
    dJdW1 = dJdZ1 @ data.T / m
    dJdb1 = np.sum(dJdZ1, axis=1, keepdims=True) / m

    gradients = {"dJdW1": dJdW1,
                 "dJdb1": dJdb1,
                 "dJdW2": dJdW2,
                 "dJdb2": dJdb2}

    return gradients


def update_parameters(parameters : Dict[str, np.ndarray],
                      gradients : Dict[str, Union[np.ndarray, float]],
                      lr : float) -> Dict[str, np.ndarray]:
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    W1 = W1 - lr * gradients["dJdW1"]
    b1 = b1 - lr * gradients["dJdb1"]
    W2 = W2 - lr * gradients["dJdW2"]
    b2 = b2 - lr * gradients["dJdb2"]

    parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}

    return parameters


def nn_model(data : np.ndarray, labels : np.ndarray,
             lr : float, num_iter : int = 10_000, print_cost = False) -> Dict[str, np.ndarray]:
    input_layer_size, hidden_layer_size, output_layer_size = layer_sizes(data, labels)
    parameters = initialize_parameters(input_layer_size, hidden_layer_size, output_layer_size)

    costs : List[float] = []

    for i in range(num_iter):
        predictions, cache = forward_propagation(data, parameters)
        cost = cross_entropy(predictions, labels)
        gradients = backward_propagation(cache, parameters, data, labels)
        parameters = update_parameters(parameters, gradients, lr)

        costs.append(cost)

        if print_cost and i % ceil(num_iter / 10) == 0:
            print(f"Iteration: {i:4} J: {costs[-1]:0.2e}")
        
    return parameters


def predict(data : np.ndarray, parameters : Dict[str, np.ndarray]) -> float:
    assert data.shape[0] == 2
    predictions, _ = forward_propagation(data, parameters)
    THRESHOLD : float = 0.5

    return predictions > THRESHOLD


if __name__ == '__main__':
    X, Y = load_planar_dataset()

    assert X.shape[0] == 2 and Y.shape[0] == 1

    model = LogisticRegressionCV()
    model.fit(X.T, Y.T)

    compute_accuracy(lambda x : model.predict(x.T), X, Y)

    parameters = nn_model(X, Y, lr = 1.2, num_iter = 10_000, print_cost=True)

    compute_accuracy(lambda x : predict(x, parameters), X, Y)
