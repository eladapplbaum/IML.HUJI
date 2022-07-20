import time
import numpy as np
import gzip
from typing import Tuple

from IMLearn.metrics.loss_functions import accuracy
from IMLearn.learners.neural_networks.modules import FullyConnectedLayer, ReLU, \
    CrossEntropyLoss, softmax
from IMLearn.learners.neural_networks.neural_network import NeuralNetwork
from IMLearn.desent_methods import GradientDescent, StochasticGradientDescent, \
    FixedLR
from IMLearn.utils.utils import confusion_matrix

import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

from exercises.gradient_descent_investigation import \
    get_gd_state_recorder_callback

pio.templates.default = "simple_white"


def load_mnist() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads the MNIST dataset

    Returns:
    --------
    train_X : ndarray of shape (60,000, 784)
        Design matrix of train set

    train_y : ndarray of shape (60,000,)
        Responses of training samples

    test_X : ndarray of shape (10,000, 784)
        Design matrix of test set

    test_y : ndarray of shape (10,000, )
        Responses of test samples
    """

    def load_images(path):
        with gzip.open(path) as f:
            # First 16 bytes are magic_number, n_imgs, n_rows, n_cols
            raw_data = np.frombuffer(f.read(), 'B', offset=16)
        # converting raw data to images (flattening 28x28 to 784 vector)
        return raw_data.reshape(-1, 784).astype('float32') / 255

    def load_labels(path):
        with gzip.open(path) as f:
            # First 8 bytes are magic_number, n_labels
            return np.frombuffer(f.read(), 'B', offset=8)

    return (load_images('../datasets/mnist-train-images.gz'),
            load_labels('../datasets/mnist-train-labels.gz'),
            load_images('../datasets/mnist-test-images.gz'),
            load_labels('../datasets/mnist-test-labels.gz'))


def plot_images_grid(images: np.ndarray, title: str = ""):
    """
    Plot a grid of images

    Parameters
    ----------
    images : ndarray of shape (n_images, 784)
        List of images to print in grid

    title : str, default="
        Title to add to figure

    Returns
    -------
    fig : plotly figure with grid of given images in gray scale
    """
    side = int(len(images) ** 0.5)
    subset_images = images.reshape(-1, 28, 28)

    height, width = subset_images.shape[1:]
    grid = subset_images.reshape(side, side, height, width).swapaxes(1,
                                                                     2).reshape(
        height * side, width * side)

    return px.imshow(grid, color_continuous_scale="gray") \
        .update_layout(
        title=dict(text=title, y=0.97, x=0.5, xanchor="center", yanchor="top"),
        font=dict(size=16), coloraxis_showscale=False) \
        .update_xaxes(showticklabels=False) \
        .update_yaxes(showticklabels=False)


def callback5(**kwargs):
    values, weights, grads = [], [], []

    def callback(**kwargs):
        values.append(kwargs["val"])
        grads.append(np.linalg.norm(kwargs["grad"], ord=2))
        if int(kwargs["t"]) % 100 == 0:
            weights.append(kwargs["weights"])

    return callback, values, weights, grads


def callback10(**kwargs):
    losses, times = [], []

    def callback(**kwargs):
        losses.append(kwargs["loss"])
        times.append(kwargs["time"])

    return callback, losses, times


if __name__ == '__main__':
    train_X, train_y, test_X, test_y = load_mnist()
    (n_samples, n_features), n_classes = train_X.shape, 10

    # ---------------------------------------------------------------------------------------------#
    # Question 5+6+7: Network with ReLU activations using SGD + recording convergence              #
    # ---------------------------------------------------------------------------------------------#
    # Initialize, fit and test network
    neurons = 64
    callback, values, weights, grads_norm = callback5()
    relu1, relu2, lr = ReLU(), ReLU(), FixedLR(0.1)
    loss = CrossEntropyLoss()
    layer_one = FullyConnectedLayer(input_dim=len(train_X[0]),
                                    output_dim=neurons, activation=relu1,
                                    include_intercept=True)
    hidden_one = FullyConnectedLayer(input_dim=neurons, output_dim=neurons,
                                     activation=relu2,
                                     include_intercept=True)
    hidden_two = FullyConnectedLayer(input_dim=neurons, output_dim=10,
                                     include_intercept=True)
    gradient = StochasticGradientDescent(learning_rate=lr, max_iter=10000,
                                         batch_size=256, callback=callback)
    nn = NeuralNetwork(modules=[layer_one, hidden_one, hidden_two],
                       loss_fn=loss, solver=gradient)
    nn._fit(train_X, train_y)
    print(f"Q5: accuracy over test = {accuracy(test_y, nn._predict(test_X))}")

    # Plotting convergence process
    fig = go.Figure([
        go.Scatter(x=np.arange(gradient.max_iter), y=values,
                   mode="markers", name="loss"),
        go.Scatter(x=np.arange(gradient.max_iter), y=grads_norm,
                   mode="markers", name="gradient norm")]
    )
    fig.update_layout(
        title=f"loss and gradient nomrn as function of iteration",
        xaxis_title="Iteration",
        yaxis_title="loss and gradient norm")
    fig.write_image(f'Q6.png')
    # Plotting test true- vs predicted confusion matrix
    print(confusion_matrix(test_y, nn._predict(train_y)))
    # ---------------------------------------------------------------------------------------------#
    # Question 8: Network without hidden layers using SGD                                          #
    # ---------------------------------------------------------------------------------------------#
    nn = NeuralNetwork(modules=[layer_one],
                       loss_fn=loss, solver=gradient)
    nn._fit(train_X, train_y)
    print(f"Q8: accuracy over test = {accuracy(test_y, nn._predict(test_X))}")

    # ---------------------------------------------------------------------------------------------#
    # Question 9: Most/Least confident predictions                                                 #
    # ---------------------------------------------------------------------------------------------#
    raise NotImplementedError()

    # ---------------------------------------------------------------------------------------------#
    # Question 10: GD vs GDS Running times                                                         #
    # ---------------------------------------------------------------------------------------------#

    neurons = 64
    gd_callback, gd_losses, gd_times = callback_10()
    sgd_callback, sgd_losses, sgd_times = callback_10()
    relu1, relu2, lr = ReLU(), ReLU(), FixedLR(0.1)
    loss = CrossEntropyLoss()
    layer_one = FullyConnectedLayer(input_dim=len(train_X[0]),
                                    output_dim=neurons, activation=relu1,
                                    include_intercept=True)
    hidden_one = FullyConnectedLayer(input_dim=neurons, output_dim=neurons,
                                     activation=relu2,
                                     include_intercept=True)
    hidden_two = FullyConnectedLayer(input_dim=neurons, output_dim=10,
                                     include_intercept=True)
    gd = GradientDescent(learning_rate=lr,
                         max_iter=10000,
                         callback=callback, tol=10 ** -10)
    nn_gd = NeuralNetwork(modules=[layer_one, hidden_one, hidden_two],
                          loss_fn=loss, solver=gd, callback=gd_callback)
    nn_gd._fit(train_X[:, 2500], train_y[:, 2500])
    fig = go.Figure(
        go.Scatter(x=gd_losses, y=gd_times,
                   mode="markers"),
    )
    fig.update_layout(
        title=f"gd time as function of loss",
        xaxis_title="losse",
        yaxis_title="time")
    fig.write_image(f'Q8gd.png')

    sgd = StochasticGradientDescent(learning_rate=lr, max_iter=10000,
                                    batch_size=64, callback=callback,
                                    tol=10 ** -10)
    nn_sgd = NeuralNetwork(modules=[layer_one, hidden_one, hidden_two],
                           loss_fn=loss, solver=sgd, callback=sgd_callback)
    nn_sgd._fit(train_X[:, 2500], train_y[:, 2500])
    fig = go.Figure(
        go.Scatter(x=sgd_losses, y=sgd_times,
                   mode="markers"),
    )
    fig.update_layout(
        title=f"sgd time as function of loss",
        xaxis_title="losse",
        yaxis_title="time")
    fig.write_image(f'Q8sgd.png')

    fig = go.Figure([
        go.Scatter(x=gd_losses, y=gd_times,
                   mode="markers", name=gd),
        go.Scatter(x=sgd_losses, y=sgd_times,
                   mode="markers", name=sgd)]
    )
    fig.update_layout(
        title=f"gd and sgd time as function of loss",
        xaxis_title="losse",
        yaxis_title="time")
    fig.write_image(f'Q8gd_sgd.png')
