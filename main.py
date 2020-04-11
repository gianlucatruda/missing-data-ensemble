import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
import pandas as pd
from loguru import logger
from support import ham_distance, hamming_distances


class FeatureCollection():
    def __init__(self, indeces):
        self.indeces = indeces

    def __add__(self, ft2):
        self.indeces.extend(ft2.indeces)
        return self

    def __repr__(self):
        return str(self.indeces)


class CascadingEnsemble():

    def __init__(self):
        logger.debug("Initialising CascadingEnsemble...")
        self.features = None

    def fit(self, X, y):
        logger.info(f"Fitting X {X.shape} and y {y.shape} ...")

        '''Organise / identify features'''
        self._encode_features(X)
        logger.debug(f"Features: {self.features}")

        '''Generate feature collections'''
        self._make_feature_collections(X)

        '''Assign feature collections to estimators'''

        '''Prioritise estimators'''

        '''Train estimators'''

        return self

    def predict(self, X):
        logger.info(f"Predicting X {X.shape} ...")

        '''For each x in X:'''

        '''Determine feature collections for vector'''

        '''Get each estimator's prediction in sequence'''

        '''Produce single unified prediction'''

        y_hat = np.random.randint(5, high=50, size=(X.shape[0]))

        return y_hat

    def _encode_features(self, X):
        logger.debug("Encoding features...")
        self.features = {i: None for i in range(X.shape[1])}

    def _make_feature_collections(self, X):
        feature_collections = []
        for i in range(X.shape[1]):
            feature_collections.append(FeatureCollection([i]))

        logger.debug(f"Initial feat. cols.:\n{feature_collections}")

        logger.debug(f"Making missing value filter...")
        _X = np.isnan(X)
        logger.debug(f"\n{_X}")

        # Split _X into vectors
        nan_vectors = np.hsplit(_X, _X.shape[1])
        logger.debug(f"nan_vectors: {len(nan_vectors)}")

        done = False
        for n in range(3):
            logger.debug(f"Feat. cols.:\n{feature_collections}")
            # new_feature_collections = []
            dists = hamming_distances(nan_vectors)
            logger.debug(f"Ham dists:\n{dists}")
            favourites = []
            for i in range(dists.shape[0]):
                vec = dists[i, :]
                vec[i] = np.inf
                favourites.append(vec.argmin())
                logger.debug(f"{i}|{vec.argmin()}")
            paired = []
            pairs = []
            for i, fav in enumerate(favourites):
                if favourites[fav] == i:
                    if i not in paired and fav not in paired:
                        pairs.append([i, fav])
                        paired.append(i)
                        paired.append(fav)
            logger.debug(f"Pairs: {pairs}")
            remaining = [i for i in range(len(favourites)) if i not in paired]
            logger.debug(f"Remaining: {remaining}")

            new_feature_collections = []
            for (a, b) in pairs:
                new_feature_collections.append(feature_collections[a] + feature_collections[b])

            for rem in remaining:
                new_feature_collections.append(feature_collections[rem])

            feature_collections = new_feature_collections

            if True:
                done = True


if __name__ == '__main__':

    import warnings
    warnings.filterwarnings(action="ignore", module="sklearn",
                            message="^internal gelsd")

    boston = datasets.load_boston()
    X = boston.data
    y = boston.target
    print(y.min(), y.max())

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    X_train_nan = X_train.copy()
    nan_targets = np.random.randint(4, size=(X_train.shape))
    X_train_nan[nan_targets == 1] = np.NaN

    # X_test_nan = X_test.copy()
    # nan_targets = np.random.randint(20, size=(X_test.shape))
    # X_test_nan[nan_targets == 19] = np.NaN

    logger.debug(
        f"{X_train.shape}, {y_train.shape}, {X_test.shape}, {y_test.shape}")

    casc = CascadingEnsemble()
    casc.fit(X_train_nan, y_train)
    pred = casc.predict(X_test)

    casc_score = mean_squared_error(y_test, pred)

    lr = LinearRegression(normalize=True)
    lr_X = np.nan_to_num(X_train_nan)
    lr.fit(lr_X, y_train)
    lr_score = mean_squared_error(y_test, lr.predict(X_test))

    logger.info(f"LR  MSE: {lr_score: .3f}")
    logger.info(f"CSC MSE: {casc_score: .3f}")
