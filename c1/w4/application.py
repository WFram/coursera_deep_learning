import matplotlib.pyplot as plt
import sys
import argparse
import numpy as np

from pathlib import Path
from typing import List, Union, Dict

sys.path.append(str(Path('modules').resolve()))
from common.costs import cross_entropy
from datasets.datasets import load_dataset, prepare_images
from common.evaluation import compute_accuracy
from common.io import save_parameters, read_parameters


from dnn import initialize_parameters, initialize_parameters_deep, \
                model_forward, model_backward, update_parameters


def two_layer_model(data : np.ndarray, labels : np.ndarray, layer_dims : List[int],
                    lr : float = 0.0075, num_iter : int = 3000, print_cost : bool = True) -> Dict[str, Union[np.ndarray, float]]:
    parameters = initialize_parameters(layer_dims[0], layer_dims[1], layer_dims[2])
    costs : List[float] = []

    for i in range(num_iter):
        predictions, cache = model_forward(data, parameters)
        cost = cross_entropy(predictions, labels)
        costs.append(cost)
        gradients = model_backward(predictions, labels, cache)
        parameters = update_parameters(parameters, gradients, lr)

        if print_cost and i % 100 == 0:
            print(f"Iteration: {i:4} J: {costs[-1]:0.2e}")

    return parameters

def l_layer_model(data : np.ndarray, labels : np.ndarray, layer_dims : List[int],
                  lr : float = 0.0075, num_iter : int = 3000, print_cost : bool = True) -> Dict[str, Union[np.ndarray, float]]:
    np.random.seed(1)
    parameters = initialize_parameters_deep(layer_dims)
    costs : List[float] = []

    for i in range(num_iter):
        predictions, cache = model_forward(data, parameters)
        cost = cross_entropy(predictions, labels)
        costs.append(cost)
        gradients = model_backward(predictions, labels, cache)
        parameters = update_parameters(parameters, gradients, lr)

        if print_cost and i % 100 == 0:
            print(f"Iteration: {i:4} J: {costs[-1]:0.2e}")

    return parameters


def predict(data : np.ndarray, parameters) -> np.ndarray:
    predictions, _ = model_forward(data, parameters)
    ids = np.where(predictions > 0.5)
    logits = np.zeros_like(predictions)
    logits[ids] = 1

    return logits


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_dataset',
                        type=str,
                        help='path to train dataset folder')
    
    parser.add_argument('--test_dataset',
                        type=str,
                        help='path to test dataset folder')
    
    parser.add_argument('-s',
                        dest='sample',
                        type=str,
                        help='path to sample image')
    
    parser.add_argument('-p',
                        dest='parameters',
                        type=str,
                        help='path to parameters (save / load)')
    
    args = parser.parse_args()

    train_dataset = Path(args.train_dataset).resolve()
    test_dataset = Path(args.test_dataset).resolve()
    checkpoints = Path(args.parameters).resolve()

    train_x_orig, train_y, test_x_orig, test_y, classes = load_dataset(train_dataset, test_dataset)

    train_x, test_x = prepare_images(train_x_orig, test_x_orig)

    if checkpoints.exists() and checkpoints.is_file():
        print("Load pretrained model")
        parameters = read_parameters(args.parameters)
    else:
        layers_dims = [12288, 20, 7, 5, 1]
        parameters = l_layer_model(train_x, train_y, layers_dims, num_iter = 2500, print_cost=True)
        save_parameters(parameters, args.parameters)

    compute_accuracy(lambda x : predict(x, parameters), train_x, train_y)
    compute_accuracy(lambda x : predict(x, parameters), test_x, test_y)
