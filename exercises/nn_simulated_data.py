import numpy as np
import pandas as pd
from typing import Tuple, List
from IMLearn.metrics.loss_functions import accuracy
from IMLearn.learners.neural_networks.modules import FullyConnectedLayer, ReLU, \
    CrossEntropyLoss
from IMLearn.learners.neural_networks.neural_network import NeuralNetwork
from IMLearn.desent_methods import GradientDescent, FixedLR
from IMLearn.utils.utils import split_train_test
from exercises.gradient_descent_investigation import \
    get_gd_state_recorder_callback
from utils import *

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

pio.templates.default = "simple_white"


def generate_nonlinear_data(
        samples_per_class: int = 100,
        n_features: int = 2,
        n_classes: int = 2,
        train_proportion: float = 0.8) -> \
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a multiclass non linearly-separable dataset. Adopted from Stanford CS231 course code.

    Parameters:
    -----------
    samples_per_class: int, default = 100
        Number of samples per class

    n_features: int, default = 2
        Data dimensionality

    n_classes: int, default = 2
        Number of classes to generate

    train_proportion: float, default=0.8
        Proportion of samples to be used for train set

    Returns:
    --------
    train_X : ndarray of shape (ceil(train_proportion * n_samples), n_features)
        Design matrix of train set

    train_y : ndarray of shape (ceil(train_proportion * n_samples), )
        Responses of training samples

    test_X : ndarray of shape (floor((1-train_proportion) * n_samples), n_features)
        Design matrix of test set

    test_y : ndarray of shape (floor((1-train_proportion) * n_samples), )
        Responses of test samples
    """
    X, y = np.zeros((samples_per_class * n_classes, n_features)), np.zeros(
        samples_per_class * n_classes, dtype='uint8')
    for j in range(n_classes):
        ix = range(samples_per_class * j, samples_per_class * (j + 1))
        r = np.linspace(0.0, 1, samples_per_class)  # radius
        t = np.linspace(j * 4, (j + 1) * 4,
                        samples_per_class) + np.random.randn(
            samples_per_class) * 0.2  # theta
        X[ix], y[ix] = np.c_[r * np.sin(t), r * np.cos(t)], j

    split = split_train_test(pd.DataFrame(X), pd.Series(y), train_proportion)
    return tuple(map(lambda x: x.values, split))


def plot_decision_boundary(nn: NeuralNetwork, lims, X: np.ndarray = None,
                           y: np.ndarray = None, title=""):
    data = [decision_surface(nn.predict, lims[0], lims[1], density=40,
                             showscale=False)]
    if X is not None:
        col = y if y is not None else "black"
        data += [go.Scatter(x=X[:, 0], y=X[:, 1], mode="markers",
                            marker=dict(color=col, colorscale=custom,
                                        line=dict(color="black", width=1)))]

    return go.Figure(data,
                     go.Layout(
                         title=rf"$\text{{Network Decision Boundaries {title}}}$",
                         xaxis=dict(title=r"$x_1$"),
                         yaxis=dict(title=r"$x_2$"),
                         width=400, height=400))


def animate_decision_boundary(nn: NeuralNetwork, weights: List[np.ndarray],
                              lims, X: np.ndarray, y: np.ndarray,
                              title="", save_name=None):
    frames = []
    for i, w in enumerate(weights):
        nn.weights = w
        frames.append(go.Frame(data=[
            decision_surface(nn.predict, lims[0], lims[1], density=40,
                             showscale=False),
            go.Scatter(x=X[:, 0], y=X[:, 1], mode="markers",
                       marker=dict(color=y, colorscale=custom,
                                   line=dict(color="black", width=1)))
            ],
                               layout=go.Layout(
                                   title=rf"$\text{{{title} Iteration {i + 1}}}$")))

    fig = go.Figure(data=frames[0]["data"], frames=frames[1:],
                    layout=go.Layout(title=frames[0]["layout"]["title"]))
    if save_name:
        animation_to_gif(fig, save_name, 200, width=400, height=400)


if __name__ == '__main__':
    np.random.seed(0)

    # Generate and visualize dataset
    n_features, n_classes = 2, 3
    train_X, train_y, test_X, test_y = generate_nonlinear_data(
        samples_per_class=500, n_features=n_features, n_classes=n_classes,
        train_proportion=0.8)
    lims = np.array([np.r_[train_X, test_X].min(axis=0),
                     np.r_[train_X, test_X].max(axis=0)]).T + np.array(
        [-.1, .1])

    go.Figure(
        data=[go.Scatter(x=train_X[:, 0], y=train_X[:, 1], mode='markers',
                         marker=dict(color=train_y, colorscale=custom,
                                     line=dict(color="black", width=1)))],
        layout=go.Layout(title=r"$\text{Train Data}$",
                         xaxis=dict(title=r"$x_1$"),
                         yaxis=dict(title=r"$x_2$"),
                         width=400, height=400)) \
        .write_image(f"../figures/nonlinear_data.png")


    def callback_func(**kwargs):
        values, weights, grads = [], [], []

        def callback(**kwargs):
            values.append(kwargs["val"])
            grads.append(np.linalg.norm(kwargs["grad"], ord=2))
            if int(kwargs["t"]) % 100 == 0:
                weights.append(kwargs["weights"])

        return callback, values, weights, grads


    # ---------------------------------------------------------------------------------------------#
    # Question 1: Fitting simple network with two hidden layers                                    #
    # ---------------------------------------------------------------------------------------------#
    # fit simple network

    def quest1_3(neurons = 16):
        callback, values, weights, grads_norm = callback_func()
        relu1, relu2 = ReLU(), ReLU()
        l1 = FullyConnectedLayer(input_dim=len(train_X[0]),
                                        output_dim=neurons, activation=relu1)
        h1 = FullyConnectedLayer(input_dim=neurons, output_dim=neurons,
                                         activation=relu2)
        h2 = FullyConnectedLayer(include_intercept=True,
                                        input_dim=neurons,
                                        output_dim=3)
        grad = GradientDescent(learning_rate=FixedLR(0.1),
                                                  max_iter=5000,
                                                  callback=callback)
        nn = NeuralNetwork(modules=[l1, h1, h2],
                           solver=grad
                           , loss_fn= CrossEntropyLoss())
        nn._fit(train_X, train_y)

        fig = plot_decision_boundary(nn, lims, train_X, train_y,
                                     title="Boundaries learned by network")
        fig.write_image(f'Q1_plot_decision_boundary{neurons}.png')
        performance = accuracy(test_y, nn._predict(test_X))
        print(f"Q1: accuracy over test = {performance}, neurons ={neurons}")
        print()
        losses, gradient_norm = values.copy(), weights.copy()

        # ---------------------------------------------------------------------------------------------#
        # Question 2: Fitting a network with no hidden layers                                          #
        # ---------------------------------------------------------------------------------------------#
        l1 = FullyConnectedLayer(input_dim=len(train_X[0]), output_dim=3,
                                        activation=relu1,
                                        include_intercept=True)
        nn = NeuralNetwork(modules=[l1], loss_fn=CrossEntropyLoss(), solver=grad)
        nn._fit(train_X, train_y)
        fig = plot_decision_boundary(nn, lims, train_X, train_y, title="Test")
        fig.write_image(f'Q2_plot_decision_boundary{neurons}.png')
        performance = accuracy(test_y, nn._predict(test_X))
        print(f"Q: 2accuracy over test = {performance}, neurons ={neurons}")
        print()

        # ---------------------------------------------------------------------------------------------#
        # Question 3+4: Plotting network convergence process                                           #
        # ---------------------------------------------------------------------------------------------#
        fig = go.Figure([
            go.Scatter(x=np.arange(grad.max_iter_), y=values,
                       mode="markers", name="loss"),
            go.Scatter(x=np.arange(grad.max_iter_), y=grads_norm,
                       mode="markers", name="gradient norm")]
        )
        fig.update_layout(
            title=f"loss and gradient nomrn as function of iteration",
            xaxis_title="Iteration",
            yaxis_title="loss and gradient norm")
        fig.write_image(f'Q3neurons={neurons}.png')


    quest1_3()
    quest1_3(neurons=6)
    print("done")
