from __future__ import annotations
from copy import deepcopy
from typing import Tuple, Callable
import numpy as np
from IMLearn import BaseEstimator


def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
                   scoring: Callable[[np.ndarray, np.ndarray, ...], float],
                   cv: int = 5) -> Tuple[float, float]:
    """
    Evaluate metric by cross-validation for given estimator

    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data

    X: ndarray of shape (n_samples, n_features)
       Input data to fit

    y: ndarray of shape (n_samples, )
       Responses of input data to fit to

    scoring: Callable[[np.ndarray, np.ndarray, ...], float]
        Callable to use for evaluating the performance of the cross-validated model.
        When called, the scoring function receives the true- and predicted values for each sample
        and potentially additional arguments. The function returns the score for given input.

    cv: int
        Specify the number of folds.

    Returns
    -------
    train_score: float
        Average train score over folds

    validation_score: float
        Average validation score over folds
    """
    # train_scores = []
    # validation_score = []
    # index = np.arange(X.shape[0])
    # np.random.shuffle(index)
    # index_sub = np.array_split(index, cv)
    # for i in range(cv):
    #     X_train = np.delete(X, index_sub[i], axis=0)
    #     y_train = np.delete(y, index_sub[i], axis=0)
    #     X_test =  X[index_sub[i], :]
    #     y_test =  y[index_sub[i]]
    #     estimator.fit(X_train, y_train)
    #     train_scores.append(scoring(y_train, estimator.predict(X_train)))
    #     validation_score.append(scoring(y_test,estimator.predict(X_test)))
    # return float(np.mean(train_scores)), float(np.mean(validation_score))

    ids = np.arange(X.shape[0])

    # Randomly split samples into `cv` folds
    folds = np.array_split(ids, cv)

    train_score, validation_score = .0, .0
    for fold_ids in folds:
        train_msk = ~np.isin(ids, fold_ids)
        fit = deepcopy(estimator).fit(X[train_msk], y[train_msk])

        train_score += scoring(y[train_msk], fit.predict(X[train_msk]))
        validation_score += scoring(y[fold_ids], fit.predict(X[fold_ids]))

    return train_score / cv, validation_score / cv