import tensorflow as tf
import tensorflow.compat.v1 as tf1
import numpy as np
import sys
import matplotlib.pyplot as plt
import argparse

from typing import Union, List, Tuple, Dict
from pathlib import Path

sys.path.append(str(Path('modules').resolve()))
from datasets.datasets import load_dataset, prepare_images, load_image
from common.io import save_parameters, read_parameters

sys.path.append(str(Path('c2/w2').resolve()))
sys.path.append(str(Path('c2').resolve()))
from w2.optimization import random_mini_batches


def convert_to_one_hot(labels : np.ndarray, num_classes : int):
    Y = np.eye(num_classes)[labels.reshape(-1)].T
    return Y


def linear_func() -> np.ndarray:
    np.random.seed(1)

    x = tf.constant(np.random.randn(3, 1), name = 'X')
    w = tf.constant(np.random.randn(4, 3), name = 'W')
    b = tf.constant(np.random.randn(4, 1), name = 'b')
    y = tf.constant(np.random.randn(4, 1), name = 'Y')

    operations = tf.add(tf.matmul(w, x), b)

    with tf1.Session() as session:
        y = session.run(operations)

    return y


def sigmoid_func(z : Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    placeholder = tf1.placeholder(tf.float32, name = 'z')
    operations = tf.sigmoid(placeholder)
    with tf1.Session() as session:
        return session.run(operations, feed_dict = {placeholder : z})


def cost(logits : np.ndarray, labels : np.ndarray) -> float:
    z = tf1.placeholder(tf.float32, name = 'z')
    y = tf1.placeholder(tf.float32, name = 'y')
    operations = tf.nn.sigmoid_cross_entropy_with_logits(logits = z, labels = y)
    with tf1.Session() as session:
        return session.run(operations, feed_dict = {z : logits, y : labels})


def one_hot_matrix(labels : np.ndarray, num_classes : int) -> np.ndarray:
    y = tf1.placeholder(tf.uint8, name = 'y')
    d = tf1.placeholder(tf.int32, name = 'd')
    operations = tf.one_hot(y, d, axis = 0)
    with tf1.Session() as session:
        return session.run(operations, feed_dict = {y : labels, d : num_classes})


def ones(shape : Tuple[int, int]) -> np.ndarray:
    with tf1.Session() as session:
        return session.run(tf.ones(shape))


def create_placeholders(image_size : int, num_classes : int):
    X = tf1.placeholder(np.float32, [image_size, None], name = 'X')
    Y = tf1.placeholder(np.float32, [num_classes, None], name = 'Y')
    return X, Y


def initialize_parameters(input_dim : int) -> Dict[str, np.ndarray]:
    parameters : Dict[str, np.ndarray] = {}

    output_dim = 25

    for i in range(1, 4):
        parameters[f"W{i}"] = tf1.get_variable(f"W{i}", [output_dim, input_dim], initializer = tf.initializers.GlorotUniform(seed = 1))
        parameters[f"b{i}"] = tf1.get_variable(f"b{i}", [output_dim, 1], initializer = tf.zeros_initializer())
        input_dim = output_dim
        output_dim //= 2

    return parameters


def forward_propagation(data : tf.Tensor, parameters : Dict[str, np.ndarray]) -> tf.Tensor:
    Z1 = tf.add(tf.matmul(parameters["W1"], data), parameters["b1"])
    A1 = tf.nn.relu(Z1)
    Z2 = tf.add(tf.matmul(parameters["W2"], A1), parameters["b2"])
    A2 = tf.nn.relu(Z2)
    Z3 = tf.add(tf.matmul(parameters["W3"], A2), parameters["b3"])
    
    return Z3


def compute_cost(logits : tf.Tensor, labels : tf.Tensor) -> tf.Tensor:
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = tf.transpose(logits), labels = tf.transpose(labels)))


def model(train_data : np.ndarray, train_labels : np.ndarray, learning_rate : float = 1e-4, num_epochs : int = 1500,
          minibatch_size : int = 32, print_cost : bool = True):
    tf1.reset_default_graph()
    tf1.set_random_seed(1)
    seed = 3
    image_size, _ = train_data.shape
    num_classes = train_labels.shape[0]
    costs : List[float] = []

    X, Y = create_placeholders(image_size, num_classes)
    parameters = initialize_parameters(image_size)
    Z3 = forward_propagation(X, parameters)
    cost = compute_cost(Z3, Y)

    optimizer = tf1.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)

    init = tf1.global_variables_initializer()

    with tf1.Session() as session:
        session.run(init)
        for epoch in range(0, num_epochs):
            epoch_cost : float = 0.0
            seed += 1
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)

            for batch in minibatches:
                _, J = session.run([optimizer, cost], feed_dict = {X : batch[0], Y : batch[1]})
                epoch_cost += J / len(minibatches)
            
            if print_cost == True and epoch % 100 == 0:
                print (f"Cost after epoch {epoch}: {epoch_cost}")
            if print_cost == True and epoch % 5 == 0:
                costs.append(epoch_cost)

        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per fives)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        parameters = session.run(parameters)
        print("Parameters are learned")

        is_correct = tf.equal(tf.argmax(Z3), tf.argmax(Y))

        accuracy = tf.reduce_mean(tf.cast(is_correct, "float"))

        print (f"Train Accuracy: {accuracy.eval({X: X_train, Y: Y_train})}")
        print (f"Test Accuracy: {accuracy.eval({X: X_test, Y: Y_test})}")

    return parameters


def predict(image : np.ndarray, parameters : Dict[str, np.ndarray]) -> None:
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    logits = forward_propagation(image, parameters)
    prediction = tf.argmax(logits)
    with tf1.Session() as session:
        prediction = session.run(prediction)
        print(f"Predicted class: {prediction[0]}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-p',
                        dest='parameters',
                        type=str,
                        help='path to parameters (save / load)',
                        required=True)
    
    parser.add_argument('-s',
                        dest='sample_image_path',
                        type=str,
                        help='path to test image',
                        required=False,
                        default=None)
    
    args = parser.parse_args()
    
    X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset(Path('datasets/train_signs.h5').resolve(),
                                                                                 Path('datasets/test_signs.h5').resolve())
    X_train, X_test = prepare_images(X_train_orig, X_test_orig)

    Y_train, Y_test = convert_to_one_hot(Y_train_orig, 6), convert_to_one_hot(Y_test_orig, 6)

    tf1.disable_eager_execution()

    checkpoints = Path(args.parameters).resolve()
    if checkpoints.exists() and checkpoints.is_file():
        print("Load pretrained model")
        parameters = read_parameters(checkpoints)
    else:
        parameters = model(X_train, Y_train)
        save_parameters(parameters, checkpoints)

    if args.sample_image_path:
        image_path = Path(args.sample_image_path).resolve()
        assert image_path.exists() and image_path.is_file()
        image = load_image(image_path, X_train_orig.shape[1], X_train_orig.shape[2])
        predict(image, parameters)
