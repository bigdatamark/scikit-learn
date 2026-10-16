# [MODULE DOC]
# Thin Python wrapper (TreePredictor) around a fitted tree stored as a structured
# array of PREDICTOR_RECORD_DTYPE nodes. All heavy work is delegated to the Cython
# routines in _predictor (raw prediction, binned prediction, partial dependence).
"""
This module contains the TreePredictor class which is used for prediction.
"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np

from sklearn.ensemble._hist_gradient_boosting._predictor import (
    _compute_partial_dependence,
    _predict_from_binned_data,
    _predict_from_raw_data,
)
from sklearn.ensemble._hist_gradient_boosting.common import (
    PREDICTOR_RECORD_DTYPE,
    Y_DTYPE,
)


class TreePredictor:
    # [CLASS DOC]
    # Holds one fitted tree: the nodes record array plus categorical bitsets for
    # binned and raw prediction. Built by TreeGrower.make_predictor; stored in
    # _predictors of the HGB estimators. Supports pickling with dtype normalization.
    """Tree class used for predictions.

    Parameters
    ----------
    nodes : ndarray of PREDICTOR_RECORD_DTYPE
        The nodes of the tree.
    binned_left_cat_bitsets : ndarray of shape (n_categorical_splits, 8), dtype=uint32
        Array of bitsets for binned categories used in predict_binned when a
        split is categorical.
    raw_left_cat_bitsets : ndarray of shape (n_categorical_splits, 8), dtype=uint32
        Array of bitsets for raw categories used in predict when a split is
        categorical.
    """

    def __init__(self, nodes, binned_left_cat_bitsets, raw_left_cat_bitsets):
        # [FUNCTION DOC]
        # Stores the three arrays as attributes nodes, binned_left_cat_bitsets,
        # raw_left_cat_bitsets without validation or copying.
        self.nodes = nodes
        self.binned_left_cat_bitsets = binned_left_cat_bitsets
        self.raw_left_cat_bitsets = raw_left_cat_bitsets

    def get_n_leaf_nodes(self):
        # [FUNCTION DOC]
        # Returns the count of nodes whose is_leaf field is true, as a Python int.
        """Return number of leaves."""
        return int(self.nodes["is_leaf"].sum())

    def get_max_depth(self):
        # [FUNCTION DOC]
        # Returns the maximum value of the depth field over all nodes (leaves are the
        # deepest), as a Python int.
        """Return maximum depth among all leaves."""
        return int(self.nodes["depth"].max())

    def predict(self, X, known_cat_bitsets, f_idx_map, n_threads):
        # [FUNCTION DOC]
        # Allocates a float output vector of length n_samples (Y_DTYPE) and lets
        # _predict_from_raw_data walk the tree for each row of raw X (handling NaN via
        # missing_go_to_left and categorical splits via the raw bitsets, with unknown
        # categories checked through known_cat_bitsets/f_idx_map), filling it in parallel.
        # Outputs: ndarray (n_samples,) of leaf values. Side effects: none.
        """Predict raw values for non-binned data.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            The input samples.

        known_cat_bitsets : ndarray of shape (n_categorical_features, 8)
            Array of bitsets of known categories, for each categorical feature.

        f_idx_map : ndarray of shape (n_features,)
            Map from original feature index to the corresponding index in the
            known_cat_bitsets array.

        n_threads : int
            Number of OpenMP threads to use.

        Returns
        -------
        y : ndarray, shape (n_samples,)
            The raw predicted values.
        """
        out = np.empty(X.shape[0], dtype=Y_DTYPE)

        _predict_from_raw_data(
            self.nodes,
            X,
            self.raw_left_cat_bitsets,
            known_cat_bitsets,
            f_idx_map,
            n_threads,
            out,
        )
        return out

    def predict_binned(self, X, missing_values_bin_idx, n_threads):
        # [FUNCTION DOC]
        # Same as predict but for already binned uint8 X: _predict_from_binned_data
        # compares bin indices with bin_threshold, routes missing_values_bin_idx using
        # missing_go_to_left, and uses binned_left_cat_bitsets for categorical splits.
        # Outputs: ndarray (n_samples,) of leaf values.
        """Predict raw values for binned data.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            The input samples.
        missing_values_bin_idx : uint8
            Index of the bin that is used for missing values. This is the
            index of the last bin and is always equal to max_bins (as passed
            to the GBDT classes), or equivalently to n_bins - 1.
        n_threads : int
            Number of OpenMP threads to use.

        Returns
        -------
        y : ndarray, shape (n_samples,)
            The raw predicted values.
        """
        out = np.empty(X.shape[0], dtype=Y_DTYPE)
        _predict_from_binned_data(
            self.nodes,
            X,
            self.binned_left_cat_bitsets,
            missing_values_bin_idx,
            n_threads,
            out,
        )
        return out

    def compute_partial_dependence(self, grid, target_features, out):
        # [FUNCTION DOC]
        # Delegates to the Cython _compute_partial_dependence (weighted tree traversal
        # method) which writes into out. Side effects: mutates out in place; returns None.
        """Fast partial dependence computation.

        Parameters
        ----------
        grid : ndarray, shape (n_samples, n_target_features)
            The grid points on which the partial dependence should be
            evaluated.
        target_features : ndarray, shape (n_target_features)
            The set of target features for which the partial dependence
            should be evaluated.
        out : ndarray, shape (n_samples)
            The value of the partial dependence function on each grid
            point.
        """
        _compute_partial_dependence(self.nodes, grid, target_features, out)

    def __setstate__(self, state):
        # [FUNCTION DOC]
        # Unpickling hook: restores state via the parent __setstate__ when available,
        # otherwise by updating __dict__ directly, then casts nodes to the current
        # platform PREDICTOR_RECORD_DTYPE (intp width differs between 32/64-bit) with
        # same_kind casting. Side effects: sets self attributes; may raise numpy casting
        # errors if the stored dtype is incompatible.
        try:
            super().__setstate__(state)
        except AttributeError:
            self.__dict__.update(state)

        # The dtype of feature_idx is np.intp which is platform dependent. Here, we
        # make sure that saving and loading on different bitness systems works without
        # errors. For instance, on a 64 bit Python runtime, np.intp = np.int64,
        # while on 32 bit np.intp = np.int32.
        #
        # TODO: consider always using platform agnostic dtypes for fitted
        # estimator attributes. For this particular estimator, this would
        # mean replacing the intp field of PREDICTOR_RECORD_DTYPE by an int32
        # field. Ideally this should be done consistently throughout
        # scikit-learn along with a common test.
        if self.nodes.dtype != PREDICTOR_RECORD_DTYPE:
            self.nodes = self.nodes.astype(PREDICTOR_RECORD_DTYPE, casting="same_kind")
