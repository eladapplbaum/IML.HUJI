from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn import datasets
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, \
    RidgeRegression
from sklearn.linear_model import Lasso

from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best fitting degree

    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate

    noise: float, default = 5
        Noise level to simulate in responses
    """
    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps Gaussian noise
    # and split into training- and testing portions
    X = np.linspace(-1.2, 2, n_samples)
    eps = [np.random.normal(0, noise) for i in range(n_samples)]
    y = (X + 3) * (X + 2) * (X + 1) * (X - 1) * (X - 2) + eps
    y_true = (X + 3) * (X + 2) * (X + 1) * (X - 1) * (X - 2)
    train_X, train_y, test_X, test_y = split_train_test(pd.DataFrame(X),
                                                        pd.Series(y),
                                                        2.0 / 3.0)
    train_X, train_y, test_X, test_y = train_X.to_numpy(), train_y.to_numpy(), test_X.to_numpy(), test_y.to_numpy()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=X, y=y_true, mode="markers", name="true"))
    fig.add_trace(go.Scatter(x=train_X.flatten(), y=train_y, mode="markers",
                             name="train"))
    fig.add_trace(
        go.Scatter(x=test_X.flatten(), y=test_y, mode="markers", name="test"))
    title = f"train and test sets in compare to true value with noise={noise}, n_samples={n_samples} "
    fig.update_layout(title=title,
                      xaxis_title="X",
                      yaxis_title="y")
    fig.write_image(f"1_noise{noise}_n_samples{n_samples}.png")
    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    averages_train_score = np.empty(11)
    average_validations_score = np.empty(11)
    for i in range(11):
        average, validation = cross_validate(PolynomialFitting(i), train_X,
                                             train_y, mean_square_error)
        averages_train_score[i] = average
        average_validations_score[i] = validation

    fig = go.Figure()
    fig.add_trace(go.Bar(x=np.arange(11), y=averages_train_score,
                             name="averages_train_score"))
    fig.add_trace(go.Bar(x=np.arange(11), y=average_validations_score,
                             name="average_validations_score"))
    title = f"averages_score as function of k degree with noise={noise}, n_samples={n_samples} "
    fig.update_layout(title=title,
                      xaxis_title="k degree",
                      yaxis_title="averages_score")
    fig.write_image(f"2_noise{noise}_n_samples{n_samples}.png")

    print("done")

    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    k = 4
    print(PolynomialFitting(4).fit(train_X,train_y).loss(test_y,test_X))


def select_regularization_parameter(n_samples: int = 50,
                                    n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 6 - Load diabetes dataset and split into training and testing portions
    raise NotImplementedError()

    # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    raise NotImplementedError()

    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model
    raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    select_polynomial_degree()
    select_polynomial_degree(noise=0)
    select_polynomial_degree(n_samples=1500, noise=10)

    raise NotImplementedError()
