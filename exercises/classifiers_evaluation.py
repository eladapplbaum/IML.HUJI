from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
import numpy as np
from typing import Tuple
from utils import *
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

pio.templates.default = "simple_white"
from math import atan2, pi


def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers. File is assumed to be an
    ndarray of shape (n_samples, 3) where the first 2 columns represent features and the third column the class

    Parameters
    ----------
    filename: str
        Path to .npy data file

    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used

    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class

    """
    data = np.load(filename)
    return data[:, :2], data[:, 2].astype(int)


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """
    for n, f in [("Linearly Separable", "linearly_separable.npy"),
                 ("Linearly Inseparable", "linearly_inseparable.npy")]:
        # Load dataset
        X, y = load_dataset("../datasets/" + f)

        # Fit Perceptron and record loss in each fit iteration
        losses = []
        perceptron = Perceptron(
            callback=lambda p, X, y: losses.append(p._loss(X, y))).fit(X, y)

        # Plot figure
        fig = go.Figure(
            go.Scatter(x=np.arange(perceptron.max_iter_), y=losses))
        fig.update_layout(xaxis_title='Iteration',
                          yaxis_title='Loss',
                          title='loss as function of Iteration of fitting Perceptron model for ' + n, )
        fig.write_image(f"../{n}.png")
        fig.show()




def get_ellipse(mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix

    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse

    cov: ndarray of shape (2,2)
        Covariance of Gaussian

    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """
    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines", marker_color="black")


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    for f in ["gaussian1.npy", "gaussian2.npy"]:
        # Load dataset
        X, y = load_dataset(f"../datasets/{f}")
        model = [LDA(), GaussianNaiveBayes()]
        model_name = ["LDA", "GaussianNaiveBayes"]
        symbols = {c:l for c,l in enumerate(set(y.flatten()))}
        from IMLearn.metrics import accuracy
        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=[rf"$\textbf{{{m}}}$" for m in
                                            model_name])
        for i, m in enumerate(model):
            m.fit(X, y)
            acc = accuracy(y, m.predict(X))
            fig.add_traces([
                    go.Scatter(x=X[:, 0], y=X[:, 1], mode="markers",
                               showlegend=False,
                               marker=dict(symbol=[symbols[i] for i in y.flatten().astype(int)],
                                           color=m.predict(X).astype(int)))
                    ,go.Scatter(x=m.mu_[:, 0], y=m.mu_[:, 1], mode='markers',
                               marker=dict(color="black",
                                           symbol="x"),
                               showlegend=False)
                ])
            # fig = make_subplots(rows=1, cols=2,
            #                     subplot_titles=[rf"$\textbf{{{m}}}$" for m in
            #                                     ["Gaussian Naive Bayes",
            #                                      "Linear Discriminant Analysis"]])
            # fig = go.Figure([
            #     # go.Scatter(
            #     # x=X[:, 0],
            #     # y=X[:, 1],
            #     # mode='markers',
            #     # marker=dict(symbol=symboles,
            #     #             color=gaus_pred)),
            #     go.Scatter(
            #         x=X[:, 0],
            #         y=X[:, 1],
            #         mode='markers',
            #         marker=dict(symbol=symboles,
            #                     color=lda_pred))
            # ])

            fig.write_image(f"../{f}.png")
        # Fit models and predict over training set
        raise NotImplementedError()
        # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
        # Create subplots
        from IMLearn.metrics import accuracy
        raise NotImplementedError()

        # Add traces for data-points setting symbols and colors
        raise NotImplementedError()

        # Add `X` dots specifying fitted Gaussians' means
        raise NotImplementedError()

        # Add ellipses depicting the covariances of the fitted Gaussians
        raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    run_perceptron()
    compare_gaussian_classifiers()
