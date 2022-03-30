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
    df = pd.read_csv(filename)
    series = df.loc[:, ["price"]]

    df['yr_sold'] = df['date'].str[:4].astype(float)
    df['is_renovated'] = np.where(df['yr_renovated'] == 0, 1, 0)
    df['house_age'] = np.where(df['yr_renovated'],df['yr_sold'] - df['yr_renovated'],
                               df['yr_sold'] - df['yr_built'])
    zip_dum = pd.get_dummies(df['zipcode'])

    df = df.loc[:,['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
                 'waterfront', 'view', 'condition', 'grade', 'sqft_above',
                 'sqft_basement', 'sqft_living15', 'sqft_lot15', 'house_age',
                  'lat', 'long','is_renovated', 'zipcode']]
    df = pd.get_dummies(df, columns=['zipcode'], drop_first=True)
    return df, series




def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
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
        #pirson = np.cov(X[feature], y) / (np.std(X[feature]) * np.std(y))
        y = y.to_numpy().ravel()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=X[feature], y=y, mode="markers"))
        title = "Pearson Correlation "  + " of feature " + feature ##todo: add pirson
        fig.update_layout(title=title,
                      xaxis_title="feature",
                      yaxis_title="response")
        fig.write_image(output_path + feature + ".png")


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    X,y = load_data('../datasets/house_prices.csv')
    # Question 2 - Feature evaluation with respect to response
    # raise NotImplementedError()
    feature_evaluation(X,y, "../feature_evaluation/")
    # Question 3 - Split samples into training- and testing sets.
    train_X, train_y, test_X, test_y = split_train_test(X,y)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    lr = LinearRegression()
    x_axias = np.arange(10,100)
    y_axias = np.array(90)
    for p in range(10, 100):
        average = 0
        for j in range(10):
            lr.fit(train_X.sample(frac=p).to_numpy(),train_y.to_numpy())
            average += lr._loss(test_X.to_numpy(), test_y.to_numpy())
        average /= 10
        y_axias[i] = average

