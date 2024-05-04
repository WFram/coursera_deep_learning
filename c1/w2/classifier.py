import numpy as np
import matplotlib.pyplot as plt
import argparse

from typing import Tuple, Union, Dict, List
from pathlib import Path

sys.path.append(str(Path('modules').resolve()))
from common.math import sigmoid
from datasets.datasets import load_dataset, prepare_images, load_image


def initialize_with_zeros(dim: int) -> Tuple[np.ndarray, np.ndarray]:
    weights : np.ndarray = np.zeros((dim , 1))
    bias : float = 0

    assert weights.shape == (dim, 1)
    assert isinstance(bias, float) or isinstance(bias, int)

    return (weights, bias)


def propagate(weights : np.ndarray,
              bias : Union[float, int],
              data : np.ndarray,
              labels : np.ndarray) -> Tuple[float, Dict[str, Union[np.ndarray, float]]]:
    predictions = sigmoid(weights.T @ data + bias)
    m = data.shape[1]
    cost = np.sum(-labels * np.log(predictions) + -(1 - labels) * np.log(1 - predictions)) / m

    # TODO: why dot prod?
    dJdW = (data @ (predictions - labels).T) / m
    dJdb = np.sum(predictions - labels) / m

    assert dJdW.shape == weights.shape
    assert dJdb.dtype == float

    cost = np.squeeze(cost)

    assert cost.shape == ()

    gradients = {"dJdW": dJdW, "dJdb": dJdb}

    return (cost, gradients)


def optimize(weights : np.ndarray,
             bias : Union[float, int],
             data : np.ndarray,
             labels : np.ndarray,
             num_iterations : int,
             learning_rate : float,
             print_cost : bool = False) -> Tuple[Dict[str, Union[np.ndarray, float]],
                                                 Dict[str, Union[np.ndarray, float]],
                                                 List[float]]:
    costs : List[float] = []
    
    for i in range(num_iterations):
        cost, gradients = propagate(weights, bias, data, labels)
        
        dJdW = gradients["dJdW"]
        dJdb = gradients["dJdb"]

        weights = weights - learning_rate * dJdW
        bias = bias - learning_rate * dJdb

        if i % 100 == 0:
            costs.append(cost)
            if print_cost:
                print(f"Cost at {i}th iteration: {cost}")
        
    parameters = {"W": weights, "b": bias}
    gradients = {"dJdW": dJdW, "dJdb": dJdb}

    return (parameters, gradients, costs)


def predict(weights : np.ndarray,
            bias : Union[float, int],
            data : np.ndarray) -> np.ndarray:
    predictions = sigmoid(weights.T @ data + bias)
    
    logits = np.zeros_like(predictions)
    indices = np.where(predictions > 0.5)
    logits[indices] = 1
    
    return logits


def model(data_train : np.ndarray,
          labels_train : np.ndarray,
          data_test : np.ndarray,
          labels_test : np.ndarray,
          num_iterations : int,
          learning_rate : float) -> dict:
    weights, bias = initialize_with_zeros(data_train.shape[0])
    parameters, _, costs = optimize(weights, bias, data_train, labels_train, \
                                    num_iterations, learning_rate)
    weights = parameters["W"]
    bias = parameters["b"]

    predictions_test = predict(weights, bias, data_test)
    predictions_train = predict(weights, bias, data_train)

    print("Train accuracy: {} %".format(100 - np.mean(np.abs(predictions_train - labels_train)) * 100))
    print("Test accuracy: {} %".format(100 - np.mean(np.abs(predictions_test - labels_test)) * 100))

    model_info = {"costs": costs,
                  "test predictions": predictions_test, 
                  "train predictions" : predictions_train, 
                  "W" : weights, 
                  "b" : bias,
                  "learning_rate" : learning_rate,
                  "num_iterations": num_iterations}
    
    return model_info


def plot_cost(costs : List[float], learning_rate : float) -> None:
    costs = np.squeeze(costs)
    plt.plot(costs)
    plt.ylabel('Cost')
    plt.xlabel('Iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()


def learning_rate_ablation():
    learning_rates : List[float] = [0.01, 0.001, 0.0001]
    models : Dict[str, dict] = {}

    for lr in learning_rates:
        print("Learning rate is: " + str(lr))
        models[str(lr)] = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations=1500, learning_rate=lr)
        print ('\n' + "-------------------------------------------------------" + '\n')

    for lr in learning_rates:
        plt.plot(np.squeeze(models[str(lr)]["costs"]), label=str(models[str(lr)]["learning_rate"]))

    plt.ylabel('Cost')
    plt.xlabel('Iterations (hundreds)')

    legend = plt.legend(loc='upper center', shadow=True)
    frame = legend.get_frame()
    frame.set_facecolor('0.90')
    plt.show()


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
    
    args = parser.parse_args()

    train_dataset = Path(args.train_dataset).resolve()
    test_dataset = Path(args.test_dataset).resolve()

    train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset(train_dataset, test_dataset)
    image_hw = train_set_x_orig.shape[1]

    train_set_x, test_set_x = prepare_images(train_set_x_orig, test_set_x_orig)

    model_info = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations=1500, learning_rate=0.005)

    image = load_image(args.sample, image_hw, image_hw)

    logits = predict(model_info["W"], model_info["b"], image)
    logit2class = lambda x : 'cat' if x == 0 else 'non-cat'
    print(f'Predicted {logit2class(np.squeeze(logits))}')