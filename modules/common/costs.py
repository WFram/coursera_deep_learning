import numpy as np

from typing import Union, Dict


def cross_entropy(predictions : np.ndarray, labels : np.ndarray) -> float:
    m = predictions.shape[1]
    cost = -np.sum(labels * np.log(predictions) + (1 - labels) * np.log(1 - predictions)) / m

    assert isinstance(cost, float)

    return cost


def l2_regularization_cost(num_samples : int, num_layers : int, parameters : Dict[str, Union[float, np.ndarray]], regularizer : float) -> float:
    cost : float = 0

    # TODO: is the end of this range always that?
    for i in range(1, num_layers + 1):
        cost += np.sum(np.square(parameters[f"W{i}"]))

    cost /= num_samples
    cost *= (0.5 * regularizer)

    return cost