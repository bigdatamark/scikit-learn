# [MODULE DOC]
# Mixin providing transform/inverse_transform for feature agglomeration: features are
# pooled per cluster label (labels_ from a fitted FeatureAgglomeration) and expanded
# back to the original feature space.
"""
Feature agglomeration. Base classes and functions for performing feature
agglomeration.
"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
from scipy.sparse import issparse

from sklearn.base import TransformerMixin
from sklearn.utils.validation import check_is_fitted, validate_data

###############################################################################
# Mixin class for feature agglomeration.


class AgglomerationTransform(TransformerMixin):
    # [CLASS DOC]
    # Mixin for estimators exposing labels_ (one cluster label per input feature) and
    # pooling_func (reduction applied along an axis, e.g. numpy mean). Provides
    # TransformerMixin's fit_transform on top.
    """
    A class for feature agglomeration via the transform interface.
    """

    def transform(self, X):
        # [FUNCTION DOC]
        # Pools the features of X per cluster. Fast path when pooling_func is numpy mean and X
        # is dense: for each sample, sum features per label with a weighted bincount and
        # divide by the cluster sizes. Otherwise, for each unique label (sorted), apply
        # pooling_func over the columns of that label along axis 1, then stack and transpose.
        # Inputs: X - array-like or sparse (n_samples, n_features), validated with reset=False.
        # Outputs: ndarray (n_samples, n_clusters), columns ordered by label value.
        # Exceptions: NotFittedError if not fitted; validation errors on feature mismatch.
        # KNOWN ISSUE: the docstring mentions 1-D input and a (n_clusters,) output, but
        # validate_data requires 2-D input, so that shape is never produced. For sparse X
        # with a mean pooling_func the result of the column reduction is a numpy matrix, so
        # the stacked output may have an extra dimension.
        """
        Transform a new matrix using the built clustering.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or \
                (n_samples, n_samples)
            An M by N array of M observations in N dimensions or a length
            M array of M one-dimensional observations.

        Returns
        -------
        Y : ndarray of shape (n_samples, n_clusters) or (n_clusters,)
            The pooled values for each feature cluster.
        """
        check_is_fitted(self)

        X = validate_data(self, X, reset=False)
        if self.pooling_func == np.mean and not issparse(X):
            size = np.bincount(self.labels_)
            n_samples = X.shape[0]
            # a fast way to compute the mean of grouped features
            nX = np.array(
                [np.bincount(self.labels_, X[i, :]) / size for i in range(n_samples)]
            )
        else:
            nX = [
                self.pooling_func(X[:, self.labels_ == l], axis=1)
                for l in np.unique(self.labels_)
            ]
            nX = np.array(nX).T
        return nX

    def inverse_transform(self, X):
        # [FUNCTION DOC]
        # Expands pooled values back: each original feature takes the value of its cluster,
        # by indexing the last axis of X with the inverse mapping from np.unique(labels_).
        # Inputs: X - array (..., n_clusters); not validated, must be a numpy array.
        # Outputs: array (..., n_features).
        # Exceptions: NotFittedError if not fitted; IndexError if X has too few clusters.
        """
        Inverse the transformation and return a vector of size `n_features`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_clusters) or (n_clusters,)
            The values to be assigned to each cluster of samples.

        Returns
        -------
        X_original : ndarray of shape (n_samples, n_features) or (n_features,)
            A vector of size `n_samples` with the values of `X` assigned to
            each of the cluster of samples.
        """
        check_is_fitted(self)

        unil, inverse = np.unique(self.labels_, return_inverse=True)
        return X[..., inverse]
