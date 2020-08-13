import sys

import numpy as np
import pandas as pd
from loguru import logger
from matplotlib import pyplot as plt
from mlxtend.regressor import StackingRegressor
from scipy.cluster import hierarchy
from sklearn import datasets
from sklearn.base import BaseEstimator
from sklearn.linear_model import BayesianRidge, Lasso, LinearRegression
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor

from casc import CascadingEnsemble
from support import ham_distance, hamming_distances

if __name__ == '__main__':

    logger.remove()
    logger.add(sys.stderr, level="WARNING")

    import warnings
    from sklearn.exceptions import ConvergenceWarning
    warnings.filterwarnings(action="ignore", module="sklearn",
                        message="^internal gelsd")
    warnings.filterwarnings(action="ignore", module="sklearn",
                        category=ConvergenceWarning)

    # dset = datasets.load_diabetes()
    dset = datasets.load_boston()
    X = dset.data
    y = dset.target

    print(y.min(), y.max())

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    X_train_nan = X_train.copy()

    # Replace values with nans after a certain index (selected at random)
    nan_offsets = np.random.randint(
        X_train.shape[0] - 50, size=(X_train.shape[1]))
    for i, offset in enumerate(nan_offsets):
        X_train_nan[offset:, i] = np.NaN

    # Randomly remove data points
    nan_targets = np.random.randint(6, size=(X_train.shape))
    X_train_nan[nan_targets == 1] = np.NaN

    logger.info(
        f"Fraction NaNs: {np.isnan(X_train_nan).sum() / X_train_nan.size: .3f}")

    logger.debug(
        f"{X_train.shape}, {y_train.shape}, {X_test.shape}, {y_test.shape}")

    # Visualise missing data
    import missingno as msno
    msno.dendrogram(pd.DataFrame(X_train_nan))
    plt.savefig('latest_cluster.png')

    results = {}

    # Make a version of X_train by filling NaNs with zero
    X_train_filled = np.nan_to_num(X_train_nan)

    # Make a version of X_train by filling NaNs with median
    X_train_filled_med = X_train_nan.copy()
    for i in range(X_train_nan.shape[1]):
        med = np.nanmedian(X_train_filled_med[:, i])
        X_train_filled_med = np.where(
            np.isnan(X_train_filled_med), med, X_train_filled_med)

    # Make a version of X_train (and corresp y) by dropping NaN rows
    keep_filter = np.prod(~np.isnan(X_train_nan), axis=1) == 1
    X_train_dropped = X_train_nan[keep_filter, :]
    y_train_dropped = y_train[keep_filter]

    print(X_train_nan.shape, X_train_filled.shape, X_train_dropped.shape)

    for est_class in [LinearRegression, Lasso, MLPRegressor, XGBRegressor]:
        casc = CascadingEnsemble(estimator_class=est_class)
        casc.fit(X_train_nan, y_train)
        pred = casc.predict(X_test)
        results[f"CSC ({est_class.__name__})"] = mse(y_test, pred)

    comparison_models = [LinearRegression, Lasso,
                            MLPRegressor, BayesianRidge, XGBRegressor]

    for mod in comparison_models:

        model = mod().fit(X_train, y_train)
        results[f"{mod.__name__} (full)"] = mse(y_test, model.predict(X_test))

        model = mod().fit(X_train_filled, y_train)
        results[f"{mod.__name__} (fill 0)"] = mse(
            y_test, model.predict(X_test))

        model = mod().fit(X_train_filled_med, y_train)
        results[f"{mod.__name__} (fill m)"] = mse(
            y_test, model.predict(X_test))

        model = mod().fit(X_train_dropped, y_train_dropped)
        results[f"{mod.__name__} (drop)"] = mse(y_test, model.predict(X_test))

    stregr = StackingRegressor(
        regressors=[
            CascadingEnsemble(estimator_class=LinearRegression),
            CascadingEnsemble(estimator_class=Lasso),
            CascadingEnsemble(estimator_class=MLPRegressor),
            CascadingEnsemble(estimator_class=XGBRegressor),
        ],
        meta_regressor=XGBRegressor(),
    ).fit(X_train_nan, y_train)
    results[f"Stacked CSC"] = mse(y_test, stregr.predict(X_test))

    print(pd.DataFrame.from_dict(results, orient='index').sort_values(by=0))
