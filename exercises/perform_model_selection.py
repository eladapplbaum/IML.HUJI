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
        train_score, validation_score = cross_validate(PolynomialFitting(i),
                                                       train_X,
                                                       train_y,
                                                       mean_square_error)
        averages_train_score[i] = train_score
        average_validations_score[i] = validation_score

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
    a = PolynomialFitting(4).fit(train_X, train_y).loss(test_X, test_y)
    print(PolynomialFitting(4).fit(train_X, train_y).loss(test_X, test_y))


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
    X, y = datasets.load_diabetes(return_X_y=True, as_frame=True)
    from sklearn.model_selection import train_test_split
    train_X, test_X, train_y, test_y = train_test_split(X, y,
                                                        train_size=n_samples)
    train_X, train_y, test_X, test_y = train_X.to_numpy(), train_y.to_numpy(), test_X.to_numpy(), test_y.to_numpy()

    # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    range = [0, 3]
    for model, model_name in [(RidgeRegression, "RidgeRegression"),
                              (Lasso, "Lasso")]:
        lams = np.empty(n_evaluations)
        model_train_score = np.empty(n_evaluations)
        model_validations_score = np.empty(n_evaluations)
        for i, lam in enumerate(
                np.linspace(range[0], range[1], n_evaluations)):
            train_score, validation_score = cross_validate(model(lam), train_X,
                                                           train_y,
                                                           mean_square_error)
            lams[i] = lam
            model_train_score[i] = train_score
            model_validations_score[i] = validation_score
        print(model_name, lams[np.argmin(model_validations_score)])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=lams, y=model_train_score,
                                 name="averages train score"))
        fig.add_trace(go.Scatter(x=lams, y=model_validations_score,
                                 name="average validations score"))
        title = f"{model_name} averages score by lam in range {range[0], range[1]} "
        fig.update_layout(title=title,
                          xaxis_title="n evaluations",
                          yaxis_title="averages score")
        fig.write_image(
            f"{model_name}n_evaluations={n_evaluations},range={range[0], range[1]}.png")

    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model
    Lasso_best_parameter = 0.30060120240480964
    RidgeRegression_best_parameter = 0.02404809619238477
    X, y = X.to_numpy(), y.to_numpy()
    for model, model_name in [(Lasso(Lasso_best_parameter), "Lasso"), (
    RidgeRegression(RidgeRegression_best_parameter), "Ridge"),
                              (LinearRegression(), "LinearRegression")]:
        model.fit(X, y)
        print(model_name, mean_square_error(y, model.predict(X)))


if __name__ == '__main__':
    np.random.seed(0)
    # select_polynomial_degree()
    # select_polynomial_degree(noise=0)
    # select_polynomial_degree(n_samples=1500, noise=10)
    select_regularization_parameter()
    raise NotImplementedError()
