import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
import pandas as pd


class CascadingEnsemble():

    def __init__(self):
        pass

    def fit(self, X, y):
        '''Organise / identify features'''

        '''Generate feature collections'''

        '''Assign feature collections to estimators'''

        '''Prioritise estimators'''

        '''Train estimators'''

        return self

    def predict(self, X):
        '''For each x in X:'''

        '''Determine feature collections for vector'''

        '''Get each estimator's prediction in sequence'''

        '''Produce single unified prediction'''

        y_hat = X[:, -1]

        return y_hat


if __name__ == '__main__':

    import warnings
    warnings.filterwarnings(action="ignore", module="sklearn",
                            message="^internal gelsd")

    boston = datasets.load_boston()
    df = pd.DataFrame(boston.data, columns=boston.feature_names)
    X_train, X_test, y_train, y_test = train_test_split(
        boston.data, boston.target)

    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    lr = LinearRegression(normalize=True)
    casc = CascadingEnsemble()
    casc.fit(X_train, y_train)
    pred = casc.predict(X_test)
    casc_score = mean_squared_error(y_test, pred)
    lr.fit(X_train, y_train)
    lr_score = mean_squared_error(y_test, lr.predict(X_test))
    print(f"LR  MSE: {lr_score: .3f}")
    print(f"CSC MSE: {casc_score: .3f}")
