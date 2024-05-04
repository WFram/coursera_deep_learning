import numpy as np

from typing import Dict


def save_parameters(parameters : Dict[str, np.ndarray], filename : str) -> None:
    np.savez(filename, **parameters)


def read_parameters(filename : str) -> Dict[str, np.ndarray]:
    file = np.load(filename)
    parameters = dict(file)
    return parameters