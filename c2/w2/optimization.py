import numpy as np
import sys
import matplotlib.pyplot as plt
import sklearn

from typing import Tuple, Dict, List
from pathlib import Path

import test_cases

from optimizer import Optimizer, Adam

sys.path.append(str(Path('modules').resolve()))
sys.path.append(str(Path('c2/w1').resolve()))
sys.path.append(str(Path('c2').resolve()))
from w1.model import forward_propagation, backward_propagation, predict
from w1.initialization import initialize_parameters_he
from common.costs import cross_entropy
from common.evaluation import plot_decision_boundary, predict_dec


def load_dataset() -> Tuple[np.ndarray, np.ndarray]:
    np.random.seed(3)
    train_X, train_Y = sklearn.datasets.make_moons(n_samples=300, noise=.2)
    plt.scatter(train_X[:, 0], train_X[:, 1], c=train_Y, s=40, cmap=plt.cm.Spectral)
    train_X = train_X.T
    train_Y = train_Y.reshape((1, train_Y.shape[0]))
    
    return train_X, train_Y


def random_mini_batches(data : np.ndarray, labels : np.ndarray, mini_batch_size : int = 64,
                        seed : int = 0) -> List[Tuple[np.ndarray, ...]]:
    np.random.seed(seed)
    m = data.shape[1]
    mini_batches : List[Tuple[np.ndarray, ...]] = []
    num_batches = m // mini_batch_size
    if m % mini_batch_size != 0: num_batches += 1

    permutation = list(np.random.permutation(m))
    shuffled_data = data[:, permutation]
    shuffled_labels = labels[:, permutation]

    mini_batches = [(shuffled_data[:, i * mini_batch_size:(i + 1) * mini_batch_size],
                     shuffled_labels[:, i * mini_batch_size:(i + 1) * mini_batch_size])
                     for i in range(0, num_batches - 1)]
    
    mini_batches.append((shuffled_data[:, (num_batches - 1) * mini_batch_size:],
                         shuffled_labels[:, (num_batches - 1) * mini_batch_size:]))
    
    return mini_batches


def model(X : np.ndarray, Y : np.ndarray, layers_dims : List[int], optimizer : Optimizer,
          mini_batch_size : int = 64, num_epochs : int = 10000, print_cost : bool = True):          
    costs : List[float] = []
    seed = 10
    m = X.shape[1]
    print(f"The number of training examples: {m}")
    print(f"The mini-batch size: {mini_batch_size}")

    parameters = initialize_parameters_he(layers_dims)

    velocities = optimizer.initialize(parameters)
    
    for i in range(num_epochs):
        seed += 1
        minibatches = random_mini_batches(X, Y, mini_batch_size, seed)
        cost_total : float = 0
        
        for minibatch in minibatches:
            minibatch_X, minibatch_Y = minibatch

            a3, caches = forward_propagation(minibatch_X, parameters)

            cost_total += cross_entropy(a3, minibatch_Y)

            grads = backward_propagation(minibatch_X, minibatch_Y, caches)
            
            optimizer_values = (grads, *velocities) if velocities is not None else (grads, None)
            parameters, optimizer_values = optimizer.update(parameters, optimizer_values)
            velocities = optimizer_values
       
        cost_avg = cost_total / m
        
        if print_cost and i % 1000 == 0:
            print(f"Cost after epoch {i}: {cost_avg}")
        if print_cost and i % 100 == 0:
            costs.append(cost_avg)
                
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('epochs (per 100)')
    plt.title("Learning rate = " + str(optimizer.learning_rate))
    plt.show()

    return parameters


if __name__ == '__main__':
    train_X, train_Y = load_dataset()

    # train 3-layer model
    layers_dims = [train_X.shape[0], 5, 2, 1]
    parameters = model(train_X, train_Y, layers_dims, optimizer = Adam(learning_rate=0.0007))

    # Predict
    predictions = predict(train_X, train_Y, parameters)

    # Plot decision boundary
    plt.title("Model with Momentum optimization")
    axes = plt.gca()
    axes.set_xlim([-1.5,2.5])
    axes.set_ylim([-1,1.5])
    plot_decision_boundary(lambda x: predict_dec(parameters, x.T, forward_propagation), train_X, train_Y)