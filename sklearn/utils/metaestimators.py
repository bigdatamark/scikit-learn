# [MODULE DOC]
# Utilities for meta-estimators: the _BaseComposition base class handling
# get/set_params over lists of (name, estimator) pairs, _safe_split for slicing
# data (including precomputed kernels) in cross-validation, and a re-export of
# available_if (the only name in __all__).
"""Utilities for meta-estimators."""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

from abc import ABCMeta, abstractmethod
from contextlib import suppress

import numpy as np

from sklearn.base import BaseEstimator
from sklearn.utils import _safe_indexing
from sklearn.utils._available_if import available_if
from sklearn.utils._tags import get_tags

__all__ = ["available_if"]


class _BaseComposition(BaseEstimator, metaclass=ABCMeta):
    # [CLASS DOC]
    # Abstract base for composite estimators storing named sub-estimators as a list
    # of (name, estimator) tuples under some attribute (passed as attr to methods).
    # Subclasses implement __init__ and call _get_params/_set_params from their
    # get_params/set_params.
    """Base class for estimators that are composed of named sub-estimators.

    This abstract class provides parameter management functionality for
    meta-estimators that contain collections of named estimators. It handles
    the complex logic for getting and setting parameters on nested estimators
    using the "estimator_name__parameter" syntax.

    The class is designed to work with any attribute containing a list of
    (name, estimator) tuples.
    """

    @abstractmethod
    def __init__(self):
        # [FUNCTION DOC]
        # Abstract constructor placeholder; does nothing.
        pass

    def _get_params(self, attr, deep=True):
        # [FUNCTION DOC]
        # HELPER: Returns BaseEstimator params; if deep, also adds each (name, estimator)
        # from getattr(self, attr) and, for estimators with get_params, their deep
        # params under keys "name__param".
        # Inputs: attr - attribute name of the estimator list; deep - bool.
        # Outputs: dict of parameters. If the attribute is not a valid list of pairs
        # (TypeError/ValueError on dict update), returns the shallow params only.
        out = super().get_params(deep=deep)
        if not deep:
            return out

        estimators = getattr(self, attr)
        try:
            out.update(estimators)
        except (TypeError, ValueError):
            # Ignore TypeError for cases where estimators is not a list of
            # (name, estimator) and ignore ValueError when the list is not
            # formatted correctly. This is to prevent errors when calling
            # `set_params`. `BaseEstimator.set_params` calls `get_params` which
            # can error for invalid values for `estimators`.
            return out

        for name, estimator in estimators:
            if hasattr(estimator, "get_params"):
                for key, value in estimator.get_params(deep=True).items():
                    out["%s__%s" % (name, key)] = value
        return out

    def _set_params(self, attr, **params):
        # [FUNCTION DOC]
        # HELPER: Sets params in a strict order: first replaces the whole collection if
        # attr is among params; then, for keys without "__" matching an existing
        # estimator name, replaces that estimator; finally passes the remaining params
        # to BaseEstimator.set_params (nested "name__param" keys included).
        # Inputs: attr - attribute name; params - keyword params.
        # Outputs: self.
        # Side effects: mutates self attributes and nested estimators. A malformed
        # collection (TypeError from unpacking) silently skips step 2.
        # Ensure strict ordering of parameter setting:
        # 1. Replace the entire estimators collection
        if attr in params:
            setattr(self, attr, params.pop(attr))
        # 2. Replace individual estimators by name
        items = getattr(self, attr)
        if isinstance(items, list) and items:
            # Get item names used to identify valid names in params
            # `zip` raises a TypeError when `items` does not contains
            # elements of length 2
            with suppress(TypeError):
                item_names, _ = zip(*items)
                for name in list(params.keys()):
                    if "__" not in name and name in item_names:
                        self._replace_estimator(attr, name, params.pop(name))

        # 3. Individual estimator parameters and other initialisation arguments
        super().set_params(**params)
        return self

    def _replace_estimator(self, attr, name, new_val):
        # [FUNCTION DOC]
        # HELPER: Replaces the estimator named name in getattr(self, attr) with new_val,
        # assigning a new list (first match only) back to the attribute.
        # Side effects: sets self.<attr> to a new list.
        # assumes `name` is a valid estimator name
        new_estimators = list(getattr(self, attr))
        for i, (estimator_name, _) in enumerate(new_estimators):
            if estimator_name == name:
                new_estimators[i] = (name, new_val)
                break
        setattr(self, attr, new_estimators)

    def _validate_names(self, names):
        # [FUNCTION DOC]
        # HELPER: Validates estimator names: must be unique, must not collide with
        # constructor parameter names, and must not contain "__".
        # Inputs: names - sequence of str. Outputs: None.
        # Exceptions: ValueError describing the first failed rule.
        if len(set(names)) != len(names):
            raise ValueError("Names provided are not unique: {0!r}".format(list(names)))
        invalid_names = set(names).intersection(self.get_params(deep=False))
        if invalid_names:
            raise ValueError(
                "Estimator names conflict with constructor arguments: {0!r}".format(
                    sorted(invalid_names)
                )
            )
        invalid_names = [name for name in names if "__" in name]
        if invalid_names:
            raise ValueError(
                "Estimator names must not contain __: got {0!r}".format(invalid_names)
            )

    def _check_estimators_are_instances(self, estimators):
        # [FUNCTION DOC]
        # HELPER: Ensures each element of estimators is an instance, not a class.
        # Exceptions: TypeError on the first class object found.
        for estimator in estimators:
            if isinstance(estimator, type):
                raise TypeError(
                    f"Expected an estimator instance ({estimator.__name__}()), got "
                    f"estimator class instead ({estimator.__name__})."
                )


def _safe_split(estimator, X, y, indices, train_indices=None):
    # [FUNCTION DOC]
    # HELPER: If the estimator's pairwise input tag is set, X must have a shape and be
    # square; rows are selected by indices and columns by train_indices (or indices
    # if None) via np.ix_. Otherwise rows are selected with _safe_indexing. y is
    # row-indexed with _safe_indexing unless None.
    # Outputs: (X_subset, y_subset), y_subset None when y is None.
    # Exceptions: ValueError when a pairwise X has no shape or is not square.
    # KNOWN ISSUE: docstring refers to estimator._pairwise but code uses
    # get_tags(estimator).input_tags.pairwise.
    """Create subset of dataset and properly handle kernels.

    Slice X, y according to indices for cross-validation, but take care of
    precomputed kernel-matrices or pairwise affinities / distances.

    If ``estimator._pairwise is True``, X needs to be square and
    we slice rows and columns. If ``train_indices`` is not None,
    we slice rows using ``indices`` (assumed the test set) and columns
    using ``train_indices``, indicating the training set.

    Labels y will always be indexed only along the first axis.

    Parameters
    ----------
    estimator : object
        Estimator to determine whether we should slice only rows or rows and
        columns.

    X : array-like, sparse matrix or iterable
        Data to be indexed. If ``estimator._pairwise is True``,
        this needs to be a square array-like or sparse matrix.

    y : array-like, sparse matrix or iterable
        Targets to be indexed.

    indices : array of int
        Rows to select from X and y.
        If ``estimator._pairwise is True`` and ``train_indices is None``
        then ``indices`` will also be used to slice columns.

    train_indices : array of int or None, default=None
        If ``estimator._pairwise is True`` and ``train_indices is not None``,
        then ``train_indices`` will be use to slice the columns of X.

    Returns
    -------
    X_subset : array-like, sparse matrix or list
        Indexed data.

    y_subset : array-like, sparse matrix or list
        Indexed targets.

    """
    if get_tags(estimator).input_tags.pairwise:
        if not hasattr(X, "shape"):
            raise ValueError(
                "Precomputed kernels or affinity matrices have "
                "to be passed as arrays or sparse matrices."
            )
        # X is a precomputed square kernel matrix
        if X.shape[0] != X.shape[1]:
            raise ValueError("X should be a square kernel matrix")
        if train_indices is None:
            X_subset = X[np.ix_(indices, indices)]
        else:
            X_subset = X[np.ix_(indices, train_indices)]
    else:
        X_subset = _safe_indexing(X, indices)

    if y is not None:
        y_subset = _safe_indexing(y, indices)
    else:
        y_subset = None

    return X_subset, y_subset
