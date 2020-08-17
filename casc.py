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

from support import ham_distance, hamming_distances


class FeatureCollection():
    def __init__(self, indices, nan_vect):
        self.nan_vect = nan_vect
        self.indices = [indices] if isinstance(indices, int) else indices

    def __add__(self, ft2):
        self.indices.extend(ft2.indices)
        self.nan_vect = self.nan_vect * ft2.nan_vect
        return self

    def __repr__(self):
        return 'FC: ' + str(self.indices)


class CascadeNode(BaseEstimator):

    def __init__(self, estimator_class, feature_col, prev_node):
        # TODO allow passing args and kwargs
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
                # TODO optimise this prediction line
                preds.append(self.estimator.predict(x.reshape(1, -1))[0])
                is_missing.append(0)
        # Making sure everything is the right shape before hstacking
        ins, preds, is_missing = np.array(
            ins).reshape(-1, 1), np.array(preds).reshape(-1, 1), np.array(is_missing).reshape(-1, 1)

        return np.hstack((ins, preds, is_missing))

    def get_params(self, *args, **kwargs):
        return self.estimator.get_params(*args, **kwargs)

    def __repr__(self):
        return 'CN: ' + str(type(self.estimator))


class CascadingEnsemble():

    def __init__(self, estimator_class=LinearRegression, cluster_mode='hierarchical', n_nodes=None):
        logger.debug("Initialising CascadingEnsemble...")
        self.features = None
        self.feature_collections = []
        self.estimator_class = estimator_class
        self.nodes = {}
        self.__n_nodes = n_nodes
        self.cluster_mode = cluster_mode

    def fit(self, X, y):
        logger.info(f"Fitting X {X.shape} and y {y.shape} ...")

        if self.__n_nodes is None:
            self.__n_nodes = X.shape[1] * (1 + X.shape[0] // 500) // 3
            logger.debug(f"Using {self.__n_nodes} nodes.")

        '''Organise / identify features'''
        self._encode_features(X)
        logger.debug(f"Features: {self.features}")

        '''Generate prioritised feature collections'''
        if self.cluster_mode == 'legacy':
            self._make_feature_collections(X)
        elif self.cluster_mode == 'hierarchical':
            self._make_feature_collections_hierarchically(X)
        else:
            raise ValueError(f"Unknown cluster_mode '{self.cluster_mode}'")

        logger.debug(f"{self.feature_collections}")

        '''Assign feature collections to estimators'''
        prev_node = None
        for fc in self.feature_collections:
            new_node = CascadeNode(self.estimator_class, fc, prev_node)
            self.nodes[fc] = new_node
            prev_node = new_node
        logger.debug(f"Nodes: {self.nodes}")
        self.__n_nodes = len(self.nodes.keys())

        '''Train estimators'''
        for fc in self.feature_collections:
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

    def get_params(self, *args, **kwargs):
        return {k: self.nodes[k].get_params(*args, **kwargs) for k in self.feature_collections}

    def _encode_features(self, X):
        logger.debug("Encoding features...")
        self.features = {i: None for i in range(X.shape[1])}

    def _make_feature_collections(self, X):

        logger.debug(f"Making missing value filter...")
        _X = np.isnan(X)
        logger.debug(f"\n{_X}")

        # Split _X into vectors
        nan_vectors = np.hsplit(_X, _X.shape[1])
        logger.debug(f"nan_vectors: {len(nan_vectors)}")

        feature_collections = []
        for i in range(X.shape[1]):
            feature_collections.append(FeatureCollection([i], nan_vectors[i]))

        logger.debug(f"Initial feat. cols.:\n{feature_collections}")

        done = False
        while not done:
            logger.debug(f"Feat. cols.:\n{feature_collections}")
            nan_vectors = [fc.nan_vect for fc in feature_collections]
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
                if favourites[fav] == i and i not in paired and fav not in paired:
                    pairs.append([i, fav])
                    paired.append(i)
                    paired.append(fav)
            logger.debug(f"Pairs: {pairs}")
            remaining = [i for i in range(len(favourites)) if i not in paired]
            logger.debug(f"Remaining: {remaining}")

            new_feature_collections = [
                feature_collections[a] + feature_collections[b] for (a, b) in pairs
            ]

            for rem in remaining:
                new_feature_collections.append(feature_collections[rem])

            feature_collections = new_feature_collections

            if len(feature_collections) <= self.__n_nodes:
                done = True

        self.feature_collections = feature_collections

    def _make_feature_collections_hierarchically(self, X):

        logger.debug(f"Making hierarchy...")
        _Xnull = np.transpose(pd.DataFrame(X).isnull().astype(int).values)
        Z = hierarchy.linkage(_Xnull, 'average')
        n = Z.shape[0] + 1
        rootnode, nodelist = hierarchy.to_tree(Z, rd=True)
        logger.debug(f"Hierarchy has {len(nodelist)} nodes")

        # Split _X into vectors
        nan_vectors = np.hsplit(X, X.shape[1])
        logger.debug(f"nan_vectors: {len(nan_vectors)}")

        # Form flat clusters from the hierarchical clustering defined by the given linkage matrix.
        # TODO remove hardcoding and optimise on silhouette score
        clust_dist = rootnode.dist // 1.3
        T = hierarchy.fcluster(Z, clust_dist, criterion='distance')

        # Create the feature collections from the clusters
        self.feature_collections = []
        for i in range(1, T.max()+1):
            feat_inds = list(np.arange(n)[T == i])
            fc = FeatureCollection(feat_inds, None)
            self.feature_collections.append(fc)
