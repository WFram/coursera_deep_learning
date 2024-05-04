import sys
import scipy
import numpy as np
import matplotlib.pyplot as plt

from typing import Dict
from pathlib import Path

from model import model, predict, backward_propagation, forward_propagation
from initialization import initialize_parameters_he, initialize_parameters_random, initialize_parameters_zeros

import test_cases

sys.path.append(str(Path('modules').resolve()))
from common.costs import cross_entropy, l2_regularization_cost
from common.evaluation import plot_decision_boundary, predict_dec


def load_2D_dataset():
    data = scipy.io.loadmat('datasets/data.mat')
    train_X = data['X'].T
    train_Y = data['y'].T
    test_X = data['Xval'].T
    test_Y = data['yval'].T

    plt.scatter(train_X[0, :], train_X[1, :], c=train_Y, s=40, cmap=plt.cm.Spectral)
    
    return train_X, train_Y, test_X, test_Y


def compute_cost_with_regularization(predictions : np.ndarray, labels : np.ndarray,
                                     parameters : Dict[str, np.ndarray], regularizer : float):
    return cross_entropy(predictions, labels) + l2_regularization_cost(predictions.shape[1], int(len(parameters) / 2),
                                                                       parameters, regularizer)


if __name__ == '__main__': 
    train_X, train_Y, test_X, test_Y = load_2D_dataset()

    parameters = model(train_X, train_Y, initialize_parameters_he, learning_rate=0.3, num_iterations=30000, keep_probability=0.86)
    print ("On the training set:")
    predictions_train = predict(train_X, train_Y, parameters)
    print ("On the test set:")
    predictions_test = predict(test_X, test_Y, parameters)

    plt.title("Model with dropout")
    axes = plt.gca()
    axes.set_xlim([-0.75,0.40])
    axes.set_ylim([-0.75,0.65])
    plot_decision_boundary(lambda x: predict_dec(parameters, x.T, forward_propagation), train_X, train_Y)