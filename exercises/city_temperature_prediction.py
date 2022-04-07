import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import plotly.graph_objects as go
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"


def load_data(filename: str):
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    df = pd.read_csv(filename, parse_dates=['Date'])
    df = df.dropna()
    df = df[df["Temp"] > -60]
    df['DayOfYear'] = df['Date'].dt.dayofyear
    return df


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    df = load_data('../datasets/City_Temperature.csv')

    # # Question 2 - Exploring data for specific country

    X_israel = df[df.Country == "Israel"]
    fig = px.scatter(x=X_israel['DayOfYear'], y=X_israel['Temp'],
                     color=X_israel["Year"].astype(str))
    fig.update_layout(
        title="temperature vs day of year in israel",
        xaxis_title="day of year in israel",
        yaxis_title="temperature")
    fig.show()

    fig = px.bar(X_israel.groupby('Month').agg({'Temp': 'std'}), y="Temp")
    fig.update_layout(
        title="std temperature vs month in israel",
        xaxis_title="month in israel",
        yaxis_title="std temperature")
    fig.show()

    # # Question 3 - Exploring differences between countries

    group = df.groupby(['Month', 'Country'])['Temp'].agg(['mean', 'std']).reset_index()
    fig = px.line(group, x='Month', y='mean',
                  line_group='Country', color='Country',
                  error_y='std')
    fig.update_layout(
        title="temperature vs month in different countries",
        xaxis_title="month",
        yaxis_title="temperature")
    fig.show()

    # # Question 4 - Fitting model for different values of `k`

    X = X_israel.loc[:, ["DayOfYear"]]
    y = X_israel.loc[:, "Temp"]
    train_X, train_y, test_X, test_y = split_train_test(X, y)
    k_value = np.arange(1,11)
    losses = np.ndarray((10,))
    for k in k_value:
        pr = PolynomialFitting(k)
        pr.fit(train_X.to_numpy(),train_y.to_numpy())
        losses[k-1] = pr.loss(test_X.to_numpy(), test_y.to_numpy())
        print(k, losses[k-1])
    fig = px.bar(x=k_value, y=losses)
    fig.update_layout(
        title="test error recorded for each value of k.",
        xaxis_title="k value",
        yaxis_title="model loss error")
    fig.show()

    # # Question 5 - Evaluating fitted model on different countries
    min_k = 3
    pr = PolynomialFitting(min_k)
    pr.fit(X.to_numpy(), y.to_numpy())
    countries = ["Jordan", "South Africa", "The Netherlands"]
    countries_loss = np.ndarray((3,))
    for i,country in enumerate(countries):
        C = df[df.Country == country]
        X = C.loc[:, ["DayOfYear"]]
        y = C.loc[:, "Temp"]
        countries_loss[i] = pr.loss(X.to_numpy(), y.to_numpy())
    fig = px.bar(x=countries, y=countries_loss)
    fig.update_layout(
        title="model fitted on israel error over each of the other countries",
        xaxis_title="country",
        yaxis_title="model loss error")
    fig.show()

