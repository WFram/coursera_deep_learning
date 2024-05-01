import numpy as np
import matplotlib.pyplot as plt

from typing import Callable, Union


def compute_accuracy(predict_functor : Callable[[np.ndarray], Union[np.ndarray, float]],
                     data : np.ndarray, labels : np.ndarray):
    assert labels.shape[0] == 1

    predictions = predict_functor(data).reshape(1, -1).T
    assert predictions.shape == labels.T.shape

    accuracy = float((labels @ predictions + (1 - labels) @ (1 - predictions)) / float(labels.size) * 100)
    print(f"Accuracy of model: {accuracy} % (percentage of correctly labelled datapoints)")


def plot_decision_boundary(model, X : np.ndarray, y : np.ndarray) -> None:
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    h : float = 0.01

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    Z = model(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(X[0, :], X[1, :], c=y, cmap=plt.cm.Spectral)
    plt.show()


def predict_dec(parameters, X, forward_propagator):
    a3, _ = forward_propagator(X, parameters)
    predictions = (a3 > 0.5)
    return predictions