from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
import numpy as np
from typing import Tuple
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

pio.templates.default = "simple_white"


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
    return np.split(np.load(filename), [-1], axis=1)


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
        from IMLearn.metrics import accuracy
        raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    # run_perceptron()
    compare_gaussian_classifiers()
