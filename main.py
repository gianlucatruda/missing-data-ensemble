import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.metrics import mean_squared_error as mse
from sklearn.linear_model import LinearRegression, Lasso, BayesianRidge
from sklearn.neural_network import MLPRegressor
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBRegressor
import pandas as pd
from loguru import logger
from support import ham_distance, hamming_distances
import sys


class FeatureCollection():
    def __init__(self, indices):
        if isinstance(indices, int):
            self.indices = [indices]
        else:
            self.indices = indices

    def __add__(self, ft2):
        self.indices.extend(ft2.indices)
        return self

    def __repr__(self):
        return 'FC: ' + str(self.indices)


class CascadeNode():

    def __init__(self, estimator_class, feature_col, prev_node):
        self.estimator = estimator_class()
        self.feature_col = feature_col
        self.prev_node = prev_node
        self._is_fit = False

    def fit(self, X, y):
        if self._is_fit:
            raise RuntimeError("You've already fit this Node")

        _X = X[:, self.feature_col.indices]
        good_filter = np.prod(~np.isnan(_X), axis=1) == 1

        X_train = _X[good_filter, :]
        y_train = y[good_filter]

        if self.prev_node is not None:
            # We need to get the Nx3 output of the previous Node first
            y_prev = self.prev_node.predict(X)
            X_train = np.hstack((_X, y_prev))[good_filter, :]

        self.estimator = self.estimator.fit(X_train, y_train)
        self._is_fit = True

        return self

    def predict(self, X):
        if not self._is_fit:
            raise RuntimeError("You need to fit this Node before predicting")

        # New lists (2 of the 3 output features)
        preds, is_missing = [], []

        # Only the features this node is in charge of
        _X = X[:, self.feature_col.indices]

        if self.prev_node is not None:
            # Retrieve predictions from previous node
            y_prev = self.prev_node.predict(X)
            ins = list(y_prev[:, 1])  # The predictions of the previous node
            X_pred = np.hstack((_X, y_prev))
        else:
            # This is the first node, so make a list of Zeros for its input
            # TODO should this be zeros? Nans?
            ins = list(np.zeros(X.shape[0]))
            X_pred = _X

        # TODO vectorise this to make it faster
        for prev, x in zip(ins, X_pred):  # Loop through rows
            if np.isnan(x).any():  # If there are any missing features in row
                preds.append(prev)
                is_missing.append(1)
            else:  # If all features are present in a row
                preds.append(self.estimator.predict(x.reshape(1, -1))[0])
                is_missing.append(0)
        # Making sure everything is the right shape before hstacking
        ins, preds, is_missing = np.array(
            ins).reshape(-1, 1), np.array(preds).reshape(-1, 1), np.array(is_missing).reshape(-1, 1)

        return np.hstack((ins, preds, is_missing))

    def __repr__(self):
        return 'CN: ' + str(type(self.estimator))


class CascadingEnsemble():

    def __init__(self, estimator_class=LinearRegression):
        logger.debug("Initialising CascadingEnsemble...")
        self.features = None
        self.feature_collections = []
        self.estimator_class = estimator_class
        self.nodes = {}

    def fit(self, X, y):
        logger.info(f"Fitting X {X.shape} and y {y.shape} ...")

        '''Organise / identify features'''
        self._encode_features(X)
        logger.debug(f"Features: {self.features}")

        '''Generate prioritised feature collections'''
        self._make_feature_collections_simple(X)
        logger.debug(f"{self.feature_collections}")

        '''Assign feature collections to estimators'''
        prev_node = None
        for fc in self.feature_collections:
            new_node = CascadeNode(self.estimator_class, fc, prev_node)
            self.nodes[fc] = new_node
            prev_node = new_node
        logger.debug(f"Nodes: {self.nodes}")

        '''Train estimators'''
        for i, fc in enumerate(self.feature_collections):
            current_node = self.nodes[fc]
            logger.debug(f"Fitting {current_node} on {fc}")
            current_node.fit(X, y)

        return self

    def predict(self, X):
        logger.info(f"Predicting X {X.shape} ...")

        '''CascadeNodes call previous node themselves, so we just predict on last'''

        Y = self.nodes[self.feature_collections[-1]].predict(X)
        # Return only the predictions (not ins and is_missing)
        return Y[:, 1]

    def _encode_features(self, X):
        logger.debug("Encoding features...")
        self.features = {i: None for i in range(X.shape[1])}

    def _make_feature_collections_simple(self, X):

        def nan_count(vec):
            return np.sum(np.isnan(vec))

        feat_cols = []

        feat_vecs = np.hsplit(X, X.shape[1])
        feat_nans = {i: nan_count(v) for i, v in enumerate(feat_vecs)}
        logger.debug(f"feat_nans: {feat_nans}")

        ordered = {k: v for k, v in sorted(
            feat_nans.items(), key=lambda x: x[1])}
        logger.debug(f"Ordered feat_nans: {ordered}")

        ord_ind = list(ordered.keys())

        N_main = len(ord_ind) // 3 - 1
        N_addon = len(ord_ind) - 3 * N_main
        logger.debug(f"N_main: {N_main} ({3*N_main}), N_addon: {N_addon}")

        for i in range(N_main):
            feat_cols.append(
                FeatureCollection(ord_ind[i*3:(i+1)*3]))

        for i in range(N_main*3, len(ord_ind)):
            feat_cols.append(FeatureCollection(ord_ind[i]))

        self.feature_collections = feat_cols

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
                new_feature_collections.append(
                    feature_collections[a] + feature_collections[b])

            for rem in remaining:
                new_feature_collections.append(feature_collections[rem])

            feature_collections = new_feature_collections

            if True:
                done = True


if __name__ == '__main__':

    logger.remove()
    logger.add(sys.stderr, level="WARNING")

    import warnings
    from sklearn.exceptions import ConvergenceWarning
    warnings.filterwarnings(action="ignore", module="sklearn",
                            message="^internal gelsd")
    warnings.filterwarnings(action="ignore", module="sklearn",
                            category=ConvergenceWarning)

    boston = datasets.load_boston()
    X = boston.data
    y = boston.target
    print(y.min(), y.max())

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    X_train_nan = X_train.copy()
    nan_targets = np.random.randint(3, size=(X_train.shape))
    X_train_nan[nan_targets == 1] = np.NaN

    logger.debug(
        f"{X_train.shape}, {y_train.shape}, {X_test.shape}, {y_test.shape}")


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
        results[f"{mod.__name__} (fill 0)"] = mse(y_test, model.predict(X_test))

        model = mod().fit(X_train_filled_med, y_train)
        results[f"{mod.__name__} (fill m)"] = mse(y_test, model.predict(X_test))

        model = mod().fit(X_train_dropped, y_train_dropped)
        results[f"{mod.__name__} (drop)"] = mse(y_test, model.predict(X_test))

    print(pd.DataFrame.from_dict(results, orient='index').sort_values(by=0))
