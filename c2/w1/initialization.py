import numpy as np
import matplotlib.pyplot as plt
import sys

from sklearn import datasets 
from typing import Tuple, Dict, Union, List
from pathlib import Path

from model import model, predict, forward_propagation

sys.path.append(str(Path('modules').resolve()))
from common.evaluation import plot_decision_boundary, predict_dec


def load_dataset() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    np.random.seed(1)
    train_X, train_Y = datasets.make_circles(n_samples=300, noise=.05)
    np.random.seed(2)
    test_X, test_Y = datasets.make_circles(n_samples=100, noise=.05)
    
    plt.scatter(train_X[:, 0], train_X[:, 1], c=train_Y, s=40, cmap=plt.cm.Spectral);
    train_X = train_X.T
    train_Y = train_Y.reshape((1, train_Y.shape[0]))
    test_X = test_X.T
    test_Y = test_Y.reshape((1, test_Y.shape[0]))
    
    return train_X, train_Y, test_X, test_Y


def initialize_parameters_zeros(layers_dims : List[int]) -> Dict[str, Union[np.ndarray, float]]:
    parameters : Dict[str, Union[np.ndarray, float]] = {}
    np.random.seed(1)
    n = len(layers_dims)

    for i in range(1, n):
        parameters[f"W{i}"] = np.zeros((layers_dims[i], layers_dims[i - 1]))
        parameters[f"b{i}"] = np.zeros((layers_dims[i], 1))

        assert parameters[f"W{i}"].shape == (layers_dims[i], layers_dims[i - 1])
        assert parameters[f"b{i}"].shape == (layers_dims[i], 1)

    return parameters


def initialize_parameters_random(layers_dims : List[int]) -> Dict[str, Union[np.ndarray, float]]:
    parameters : Dict[str, Union[np.ndarray, float]] = {}
    np.random.seed(3)
    n = len(layers_dims)

    for i in range(1, n):
        parameters[f"W{i}"] = np.random.randn(layers_dims[i], layers_dims[i - 1]) * 10
        parameters[f"b{i}"] = np.zeros((layers_dims[i], 1))
        
        assert parameters[f"W{i}"].shape == (layers_dims[i], layers_dims[i - 1])
        assert parameters[f"b{i}"].shape == (layers_dims[i], 1)

    return parameters

def initialize_parameters_he(layers_dims : List[int]) -> Dict[str, Union[np.ndarray, float]]:
    parameters : Dict[str, Union[np.ndarray, float]] = {}
    np.random.seed(3)
    n = len(layers_dims)

    for i in range(1, n):
        parameters[f"W{i}"] = np.random.randn(layers_dims[i], layers_dims[i - 1]) * np.sqrt(2 / layers_dims[i - 1])
        parameters[f"b{i}"] = np.zeros((layers_dims[i], 1))
        
        assert parameters[f"W{i}"].shape == (layers_dims[i], layers_dims[i - 1])
        assert parameters[f"b{i}"].shape == (layers_dims[i], 1)

    return parameters


if __name__ == '__main__':
    train_X, train_Y, test_X, test_Y = load_dataset()

    parameters = model(train_X, train_Y, initialize_parameters_he)
    print ("On the train set:")
    predictions_train = predict(train_X, train_Y, parameters)
    print ("On the test set:")
    predictions_test = predict(test_X, test_Y, parameters)

    print (predictions_train)
    print (predictions_test)

    plt.title("Model with He initialization")
    axes = plt.gca()
    axes.set_xlim([-1.5,1.5])
    axes.set_ylim([-1.5,1.5])
    plot_decision_boundary(lambda x: predict_dec(parameters, x.T, forward_propagation), train_X, train_Y)