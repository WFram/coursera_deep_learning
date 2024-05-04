import numpy as np

from typing import Dict, Tuple, Any
from abc import ABC, abstractmethod


class Optimizer(ABC):
    _learning_rate : float


    @abstractmethod
    def __init__(self, *args, **kwargs):
        pass


    @abstractmethod
    def initialize(self, *args, **kwargs) -> Tuple[Dict[str, np.ndarray], ...]:
        pass


    @abstractmethod
    def update(self, *args, **kwargs) -> Tuple[Dict[str, np.ndarray], ...]:
        pass


    @property
    def learning_rate(self):
        return self._learning_rate


class MiniBatchGradientDescent(Optimizer):
    _learning_rate : float

    def __init__(self, learning_rate : float):
        self._learning_rate = learning_rate

    
    def initialize(self, *args, **kwargs) -> None:
        pass


    def update(self, parameters : Dict[str, np.ndarray],
               optimizer_values : Tuple[Dict[str, np.ndarray], ...]) -> Tuple[Dict[str, np.ndarray], ...]:
        num_layers = len(parameters) // 2

        gradients, _ = optimizer_values

        for l in range(num_layers):
            parameters[f'W{l + 1}'] -= self._learning_rate * gradients[f'dW{l + 1}']
            parameters[f'b{l + 1}'] -= self._learning_rate * gradients[f'db{l + 1}']
        
        return (parameters, None)
    

class MiniBatchGradientDescentMomentum(Optimizer):
    _learning_rate : float
    _momentum : float


    def __init__(self, learning_rate : float, momentum : float = 0.9):
        self._learning_rate = learning_rate
        self._momentum = momentum


    def initialize(self, parameters : Dict[str, np.ndarray]) -> Tuple[Dict[str, np.ndarray], ...]:
        velocities : Dict[str, np.ndarray] = {}
        num_layers = int(len(parameters) / 2) + 1
        
        for l in range(1, num_layers):
            velocities[f'dW{l}'] = np.zeros(parameters[f'W{l}'].shape)
            velocities[f'db{l}'] = np.zeros(parameters[f'b{l}'].shape)

        return velocities,
    

    def update(self, parameters : Dict[str, np.ndarray],
               optimizer_values : Tuple[Dict[str, np.ndarray], ...]) -> Tuple[Dict[str, np.ndarray], ...]:
        num_layers = len(parameters) // 2

        gradients, velocities = optimizer_values
        
        for l in range(0, num_layers):
            velocities[f'dW{l + 1}'] = self._momentum * velocities[f'dW{l + 1}'] + (1 - self._momentum) * gradients[f'dW{l + 1}']
            parameters[f'W{l + 1}'] -= self._learning_rate * velocities[f'dW{l + 1}']
            velocities[f'db{l + 1}'] = self._momentum * velocities[f'db{l + 1}'] + (1 - self._momentum) * gradients[f'db{l + 1}']
            parameters[f'b{l + 1}'] -= self._learning_rate * velocities[f'db{l + 1}']

        output_values = (velocities,)

        return parameters, output_values
    

class Adam(Optimizer):
    _learning_rate : float
    _momentum_1 : float
    _momentum_2 : float
    
    __eps : float
    __t : int


    def __init__(self, learning_rate : float, momentum_1 : float = 0.9,
                 momentum_2 : float = 0.999, eps : float = 1e-8,
                 t : int = 0):
        self._learning_rate = learning_rate
        self._momentum_1 = momentum_1
        self._momentum_2 = momentum_2
        self.__eps = eps
        self.__t = t


    def initialize(self, parameters : Dict[str, np.ndarray]) -> Tuple[Dict[str, np.ndarray], ...]:
        num_layers = len(parameters) // 2
        v : Dict[str, np.ndarray] = {}
        s : Dict[str, np.ndarray] = {}
        for l in range(num_layers):
            v[f'dW{l + 1}'] = np.zeros(parameters[f'W{l + 1}'].shape)
            s[f'dW{l + 1}'] = np.zeros(parameters[f'W{l + 1}'].shape)
            v[f'db{l + 1}'] = np.zeros(parameters[f'b{l + 1}'].shape)
            s[f'db{l + 1}'] = np.zeros(parameters[f'b{l + 1}'].shape)

        return v, s
    

    def update(self, parameters : Dict[str, np.ndarray],
               optimizer_values : Tuple[Dict[str, np.ndarray], ...]) -> Tuple[Dict[str, np.ndarray], ...]:
        self.__t += 1

        gradients, v, s = optimizer_values
        
        num_layers = len(parameters) // 2
        v_unbiased : Dict[str, np.ndarray] = {}
        s_unbiased : Dict[str, np.ndarray] = {}
        
        for l in range(num_layers):
            v[f'dW{l + 1}'] = self._momentum_1 * v[f'dW{l + 1}'] + (1 - self._momentum_1) * gradients[f'dW{l + 1}']
            v[f'db{l + 1}'] = self._momentum_1 * v[f'db{l + 1}'] + (1 - self._momentum_1) * gradients[f'db{l + 1}']

            v_unbiased[f'dW{l + 1}'] = v[f'dW{l + 1}'] / (1 - self._momentum_1 ** self.__t)
            v_unbiased[f'db{l + 1}'] = v[f'db{l + 1}'] / (1 - self._momentum_1 ** self.__t)
            
            s[f'dW{l + 1}'] = self._momentum_2 * s[f'dW{l + 1}'] + (1 - self._momentum_2) * gradients[f'dW{l + 1}'] ** 2
            s[f'db{l + 1}'] = self._momentum_2 * s[f'db{l + 1}'] + (1 - self._momentum_2) * gradients[f'db{l + 1}'] ** 2
            
            s_unbiased[f'dW{l + 1}'] = s[f'dW{l + 1}'] / (1 - self._momentum_2 ** self.__t)
            s_unbiased[f'db{l + 1}'] = s[f'db{l + 1}'] / (1 - self._momentum_2 ** self.__t)

            parameters[f'W{l + 1}'] -= self._learning_rate * v_unbiased[f'dW{l + 1}'] / (np.sqrt(s_unbiased[f'dW{l + 1}']) + self.__eps)
            parameters[f'b{l + 1}'] -= self._learning_rate * v_unbiased[f'db{l + 1}'] / (np.sqrt(s_unbiased[f'db{l + 1}']) + self.__eps)

        output_values = (v, s)

        return parameters, output_values