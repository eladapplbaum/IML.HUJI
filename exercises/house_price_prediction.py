from scipy.stats import stats

from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"


def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    df = pd.read_csv(filename).dropna().drop_duplicates()
    for f in ["date", "price", "sqft_living", "sqft_lot", "sqft_above",
              "yr_built", "sqft_living15", "sqft_lot15"]:
        df = df[df[f] != "0"]
    for f in ["bathrooms", "floors", "sqft_basement", "yr_renovated"]:
        df = df[df[f] >= 0]
    df = df[df["sqft_lot"] < 1250000]
    df = df[df["bedrooms"] < 25]
    df = df[df["sqft_lot15"] < 500000]

    series = df.loc[:, ["price"]].squeeze()

    df['yr_sold'] = df['date'].str[:4].astype(float)
    df['is_renovated'] = np.where(df['yr_renovated'] == 0, 1, 0)
    df['house_age'] = np.where(df['yr_renovated'],
                               df['yr_sold'] - df['yr_renovated'],
                               df['yr_sold'] - df['yr_built'])

    # drop long, lot, sqft_lot15, condition for bad correlation

    df = df.loc[:,
         ['bedrooms', 'bathrooms', 'sqft_living', 'floors',
          'waterfront', 'view', 'condition', 'grade', 'sqft_above',
          'sqft_basement', 'sqft_living15', 'house_age', 'is_renovated',
          'lat', 'zipcode']]
    df = pd.get_dummies(df, columns=['zipcode'])

    return df, series


def feature_evaluation(X: pd.DataFrame, y: pd.Series,
                       output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    for feature in X.columns:
        pirson = np.cov(X[feature], y)[0, 1] / (np.std(X[feature]) * np.std(y))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=X[feature], y=y, mode="markers"))
        title = "Pearson Correlation " + str(pirson) + " of feature " + feature
        fig.update_layout(title=title,
                          xaxis_title="feature",
                          yaxis_title="response")
        fig.write_image(output_path + feature + ".png")


if __name__ == '__main__':
    np.random.seed(0)
    # for quiz
    # y_true = np.array([279000, 432000, 326000, 333000, 437400, 555950])
    # y_pred = np.array(
    #     [199000.37562541, 452589.25533196, 345267.48129011, 345856.57131275,
    #      563867.1347574, 395102.94362135])
    # print(mean_square_error(y_true, y_pred))

    # Question 1 - Load and preprocessing of housing prices dataset
    X, y = load_data('../datasets/house_prices.csv')

    # Question 2 - Feature evaluation with respect to response
    feature_evaluation(X, y, "../feature_evaluation/")

    # Question 3 - Split samples into training- and testing sets.
    train_X, train_y, test_X, test_y = split_train_test(X, y)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    lr = LinearRegression()
    x_percent = np.arange(10, 101)
    mean_loss_pred = np.ndarray((91,))
    std_pred = np.ndarray((91,))
    for p in x_percent:
        pred = np.ndarray((10,))
        for j in range(10):
            lr.fit(train_X.sample(frac=(p / 100), random_state=j).to_numpy(),
                   train_y.sample(frac=(p / 100), random_state=j).to_numpy())
            pred[j] = lr._loss(test_X.to_numpy(), test_y.to_numpy())
        mean_loss_pred[p - 10] = np.mean(pred)
        std_pred[p - 10] = np.std(pred)
    fig = go.Figure([
        go.Scatter(
            name='Mean value',
            x=x_percent,
            y=mean_loss_pred,
            mode='markers+lines',
            line=dict(color='rgb(31, 119, 180)'),
        ),
        go.Scatter(
            name='Upper Bound',
            x=x_percent,
            y=mean_loss_pred + 2 * std_pred,
            mode='lines',
            marker=dict(color="#444"),
            line=dict(width=0),
            showlegend=False
        ),
        go.Scatter(
            name='Lower Bound',
            x=x_percent,
            y=mean_loss_pred - 2 * std_pred,
            marker=dict(color="#444"),
            line=dict(width=0),
            mode='lines',
            fillcolor='rgba(68, 68, 68, 0.3)',
            fill='tonexty',
            showlegend=False
        )
    ])
    fig.update_layout(
        xaxis_title='Percent of sample',
        yaxis_title='Mean values',
        title='Mean loss as function of percent from samples',
        hovermode="x")
    fig.show()
