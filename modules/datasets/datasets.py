import h5py
import numpy as np

from PIL import Image
from pathlib import Path
from typing import Tuple


def load_dataset(train_dataset_path : Path,
                 test_dataset_path : Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_dataset = h5py.File(train_dataset_path, 'r')
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])

    test_dataset = h5py.File(test_dataset_path, "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) 
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) 

    classes = np.array(test_dataset["list_classes"][:])

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


def prepare_images(train_set_x_orig, test_set_x_orig) -> Tuple[np.ndarray, np.ndarray]:
    m_train = train_set_x_orig.shape[0]
    m_test =  test_set_x_orig.shape[0]

    train_set_x_flatten = train_set_x_orig.reshape(m_train, -1).T
    test_set_x_flatten = test_set_x_orig.reshape(m_test, -1).T

    train_set_x = train_set_x_flatten / 255
    test_set_x = test_set_x_flatten / 255

    return (train_set_x, test_set_x)


def load_image(image_path : Path, h : int, w : int):
    image = Image.open(image_path).convert('RGB')
    min_image_dim = min(image.size[0], image.size[1])
    image_size = (int(h * image.size[0] / min_image_dim), \
                  int(w * image.size[1] / min_image_dim))
    image = image.resize(image_size, Image.LANCZOS)
    image = np.array(image, dtype=np.float64)
    image = image[:h, :w].reshape(1, -1).T
    image /= 255
    return image