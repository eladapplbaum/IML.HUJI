import numpy as np
import pandas as pd
from typing import Tuple, List, Callable, Type

from IMLearn import BaseModule
from IMLearn.desent_methods import GradientDescent, FixedLR, ExponentialLR
from IMLearn.desent_methods.modules import L1, L2
from IMLearn.learners.classifiers.logistic_regression import LogisticRegression
from IMLearn.metrics import misclassification_error
from IMLearn.model_selection import cross_validate
from IMLearn.utils import split_train_test

import plotly.graph_objects as go


def plot_descent_path(module: Type[BaseModule],
                      descent_path: np.ndarray,
                      title: str = "",
                      xrange=(-1.5, 1.5),
                      yrange=(-1.5, 1.5)) -> go.Figure:
    """
    Plot the descent path of the gradient descent algorithm

    Parameters:
    -----------
    module: Type[BaseModule]
        Module type for which descent path is plotted

    descent_path: np.ndarray of shape (n_iterations, 2)
        Set of locations if 2D parameter space being the regularization path

    title: str, default=""
        Setting details to add to plot title

    xrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    yrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    Return:
    -------
    fig: go.Figure
        Plotly figure showing module's value in a grid of [xrange]x[yrange] over which regularization path is shown

    Example:
    --------
    fig = plot_descent_path(IMLearn.desent_methods.modules.L1, np.ndarray([[1,1],[0,0]]))
    fig.show()
    """

    def predict_(w):
        return np.array([module(weights=wi).compute_output() for wi in w])

    from utils import decision_surface
    return go.Figure([decision_surface(predict_, xrange=xrange, yrange=yrange,
                                       density=70, showscale=False),
                      go.Scatter(x=descent_path[:, 0], y=descent_path[:, 1],
                                 mode="markers+lines", marker_color="black")],
                     layout=go.Layout(xaxis=dict(range=xrange),
                                      yaxis=dict(range=yrange),
                                      title=f"GD Descent Path {title}"))


def get_gd_state_recorder_callback() -> Tuple[
    Callable[[], None], List[np.ndarray], List[np.ndarray]]:
    """
    Callback generator for the GradientDescent class, recording the objective's value and parameters at each iteration

    Return:
    -------
    callback: Callable[[], None]
        Callback function to be passed to the GradientDescent class, recoding the objective's value and parameters
        at each iteration of the algorithm

    values: List[np.ndarray]
        Recorded objective values

    weights: List[np.ndarray]
        Recorded parameters
    """
    values = []
    weights = []

    def callback(**kwargs):
        weights.append(kwargs['weights'])
        values.append(kwargs['val'])

    return callback, values, weights


def compare_fixed_learning_rates(
        init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
        etas: Tuple[float] = (1, .1, .01, .001)):
    for model, name in [[L1, "L1"], [L2, "L2"]]:
        for eta in etas:
            callback = get_gd_state_recorder_callback()
            gd = GradientDescent(FixedLR(eta),
                                 callback=callback[0])
            gd.fit(model(init.copy()), np.empty(0), np.empty(0))
            fig = plot_descent_path(model, descent_path=np.array(callback[2]),
                                    title=f'model: {name}, eta: {eta}')
            fig.write_image(f'descent_path_fixed_{name}_{eta}.png')

            fig = go.Figure(
                data=go.Scatter(x=np.arange(gd.max_iter_), y=callback[1],
                                mode="markers"))
            fig.update_layout(title=f"{name} with eta={eta} ",
                              xaxis_title="Iteration",
                              yaxis_title="norm")
            fig.write_image(f'convergence_rate_fixed_{name}_{eta}.png')
            print(f'model {name} eta: {eta} loss: {callback[1][np.argmin(callback[1])]}')

def compare_exponential_decay_rates(
        init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
        eta: float = .1,
        gammas: Tuple[float] = (.9, .95, .99, 1)):
    # Optimize the L1 objective using different decay-rate values of the exponentially decaying learning rate

    scatters = []
    for gamma in gammas:
        callback = get_gd_state_recorder_callback()
        gd = GradientDescent(ExponentialLR(eta, gamma),
                             callback=callback[0])
        gd.fit(L1(init.copy()), np.empty(0), np.empty(0))
        scatters.append(
            go.Scatter(x=np.arange(gd.max_iter_), y=callback[1],name=f'gamma={gamma}'))
        print(
            f'l1 exp eta = 0.1: gama: {gamma} loss: {callback[1][np.argmin(callback[1])]}')

    # Plot algorithm's convergence for the different values of gamma
    fig = go.Figure(data=scatters)
    fig.update_layout(title="convergence rate for all decay rates",
                      xaxis_title="iteration", yaxis_title="norm")
    fig.write_image(f'exponential_decay_rates_L1_{eta}.png')

    # Plot descent path for gamma=0.95
    callback = get_gd_state_recorder_callback()
    gd = GradientDescent(ExponentialLR(eta, 0.95),
                         callback=callback[0])
    gd.fit(L1(init.copy()), np.empty(0), np.empty(0))
    fig = plot_descent_path(L1, np.array(callback[2]),
                            title=f"L1 model with eta={eta} and gama=0.95")
    fig.write_image(f'exponential_L1_descent_path_gama={0.95}.png')


def load_data(path: str = "../datasets/SAheart.data",
              train_portion: float = .8) -> \
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Load South-Africa Heart Disease dataset and randomly split into a train- and test portion

    Parameters:
    -----------
    path: str, default= "../datasets/SAheart.data"
        Path to dataset

    train_portion: float, default=0.8
        Portion of dataset to use as a training set

    Return:
    -------
    train_X : DataFrame of shape (ceil(train_proportion * n_samples), n_features)
        Design matrix of train set

    train_y : Series of shape (ceil(train_proportion * n_samples), )
        Responses of training samples

    test_X : DataFrame of shape (floor((1-train_proportion) * n_samples), n_features)
        Design matrix of test set

    test_y : Series of shape (floor((1-train_proportion) * n_samples), )
        Responses of test samples
    """
    df = pd.read_csv(path)
    df.famhist = (df.famhist == 'Present').astype(int)
    return split_train_test(df.drop(['chd', 'row.names'], axis=1), df.chd,
                            train_portion)


def fit_logistic_regression():
    # Load and split SA Heard Disease dataset
    X_train, y_train, X_test, y_test = load_data()
    X_train, y_train, X_test, y_test = X_train.to_numpy(), y_train.to_numpy(), X_test.to_numpy(), y_test.to_numpy()
    # Plotting convergence rate of logistic regression over SA heart disease data
    from sklearn.metrics import roc_curve, auc
    gd = GradientDescent(FixedLR(1e-4), max_iter=20000)
    model = LogisticRegression(solver=gd)
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_train)
    fpr, tpr, thresholds = roc_curve(y_train, y_prob)
    fig = go.Figure(
        data=[go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                         line=dict(color="black", dash='dash'),
                         name="Random Class Assignment"),
              go.Scatter(x=fpr, y=tpr, mode='markers+lines', text=thresholds,
                         name="", showlegend=False, marker_size=5,
                         # marker_color=c[1][1],
                         hovertemplate="<b>Threshold:</b>%{text:.3f}<br>FPR: %{x:.3f}<br>TPR: %{y:.3f}")],
        layout=go.Layout(
            title=rf"$\text{{ROC Curve Of Fitted Model}}={auc(fpr, tpr):.6f}$",
            xaxis=dict(title=r"$\text{False Positive Rate (FPR)}$"),
            yaxis=dict(title=r"$\text{True Positive Rate (TPR)}$")))
    fig.write_image(f'ROC.png')
    best_alpha = thresholds[np.argmax(tpr - fpr)]
    model = LogisticRegression(solver=gd, alpha=best_alpha)
    best_alpha_loss = model.fit(X_train, y_train)._loss(X_test, y_test)
    print(f'best_alpha: {best_alpha}')
    print(f'best_alpha_loss: {best_alpha_loss}')
    # Fitting l1- and l2-regularized logistic regression models, using cross-validation to specify values
    # of regularization parameter

    gd = GradientDescent(FixedLR(1e-4), max_iter=20000)
    for reg in ["l1", "l2"]:
        val = []
        lams = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
        for lam in lams:
            model = LogisticRegression(solver=gd, penalty="l1", alpha=0.5, lam=lam)
            tr_sc, val_sc = cross_validate(estimator=model, X=X_train, y=y_train,
                           scoring=misclassification_error)
            val.append(val_sc)
        c_lam = lams[np.argmin(val)]
        model = LogisticRegression(solver=gd, penalty=reg, alpha=0.5, lam=c_lam)
        model.fit(X_train, y_train)
        print(f'for model {reg} best lamda is: {c_lam} with loss: {model.loss(X_test, y_test)}')




if __name__ == '__main__':
    np.random.seed(0)
    # compare_fixed_learning_rates()
    # compare_exponential_decay_rates()
    fit_logistic_regression()
