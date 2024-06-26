import numpy as np

from typing import Tuple, Dict, List


def linear_forward_test_case():
    np.random.seed(1)
    A = np.random.randn(3, 2)
    W = np.random.randn(1, 3)
    b = np.random.randn(1, 1)
    
    return A, W, b

def linear_activation_forward_test_case():
    np.random.seed(2)
    A_prev = np.random.randn(3,2)
    W = np.random.randn(1,3)
    b = np.random.randn(1,1)
    return A_prev, W, b

def L_model_forward_test_case_2hidden():
    np.random.seed(6)
    X = np.random.randn(5, 4)
    W1 = np.random.randn(4, 5)
    b1 = np.random.randn(4, 1)
    W2 = np.random.randn(3, 4)
    b2 = np.random.randn(3, 1)
    W3 = np.random.randn(1, 3)
    b3 = np.random.randn(1, 1)
  
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3}
    
    return X, parameters

def compute_cost_test_case() -> Tuple[np.ndarray, np.ndarray]:
    Y = np.asarray([[1, 1, 0]])
    aL = np.array([[.8,.9,0.4]])
    
    return Y, aL

def linear_backward_test_case() -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    np.random.seed(1)
    dZ = np.random.randn(3, 4)
    A0 = np.random.randn(5, 4)
    W = np.random.randn(3, 5)
    b = np.random.randn(3, 1)
    linear_cache = {"A0": A0, "W": W, "b": b}

    return dZ, linear_cache

def linear_activation_backward_test_case() -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    np.random.seed(2)
    dA = np.random.randn(1,2)
    A0 = np.random.randn(3,2)
    W = np.random.randn(1,3)
    b = np.random.randn(1,1)
    Z = np.random.randn(1,2)

    cache = {"A0": A0, "W": W, "b": b, "Z": Z}

    return dA, cache

def L_model_backward_test_case() -> Tuple[np.ndarray, np.ndarray, List[Dict[str, np.ndarray]]]:
    np.random.seed(3)
    AL = np.random.randn(1, 2)
    Y = np.array([[1, 0]])

    A1 = np.random.randn(4,2)
    W1 = np.random.randn(3,4)
    b1 = np.random.randn(3,1)
    Z1 = np.random.randn(3,2)
    linear_cache_activation_1 = {"A0": A1, "W": W1, "b": b1, "Z": Z1}

    A2 = np.random.randn(3,2)
    W2 = np.random.randn(1,3)
    b2 = np.random.randn(1,1)
    Z2 = np.random.randn(1,2)
    linear_cache_activation_2 = {"A0": A2, "W": W2, "b": b2, "Z": Z2}

    caches = [linear_cache_activation_1, linear_cache_activation_2]

    return AL, Y, caches

def update_parameters_test_case() -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    np.random.seed(2)
    W1 = np.random.randn(3,4)
    b1 = np.random.randn(3,1)
    W2 = np.random.randn(1,3)
    b2 = np.random.randn(1,1)
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    np.random.seed(3)
    dW1 = np.random.randn(3,4)
    db1 = np.random.randn(3,1)
    dW2 = np.random.randn(1,3)
    db2 = np.random.randn(1,1)
    grads = {"dJdW1": dW1,
             "dJdb1": db1,
             "dJdW2": dW2,
             "dJdb2": db2}
    
    return parameters, grads