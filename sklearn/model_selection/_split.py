# [MODULE DOC]
# Cross-validation splitters and helpers. Splitters yield (train, test) integer index
# arrays: K-fold variants (KFold, GroupKFold, StratifiedKFold, StratifiedGroupKFold,
# TimeSeriesSplit), exhaustive leave-out schemes (LeaveOneOut, LeavePOut,
# LeaveOneGroupOut, LeavePGroupsOut), repeated K-fold, random shuffle splits
# (ShuffleSplit, GroupShuffleSplit, StratifiedShuffleSplit) and PredefinedSplit.
# Helpers: check_cv (normalises a cv argument), train_test_split, repr builders and
# _yields_constant_splits.
# Side effects at import: sets train_test_split.__test__ = False so test collectors skip
# it.
"""
The :mod:`sklearn.model_selection._split` module includes classes and
functions to split the data based on a preset strategy.
"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import numbers
import warnings
from abc import ABCMeta, abstractmethod
from collections import defaultdict
from collections.abc import Iterable
from inspect import signature
from itertools import chain, combinations
from math import ceil, floor

import numpy as np
from scipy.special import comb

from sklearn.utils import (
    _safe_indexing,
    check_random_state,
    indexable,
    metadata_routing,
)
from sklearn.utils._array_api import (
    get_namespace,
    get_namespace_and_device,
    move_to,
)
from sklearn.utils._param_validation import Interval, RealNotInt, validate_params
from sklearn.utils.extmath import _approximate_mode
from sklearn.utils.metadata_routing import _MetadataRequester
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import _num_samples, check_array, column_or_1d

__all__ = [
    "BaseCrossValidator",
    "GroupKFold",
    "GroupShuffleSplit",
    "KFold",
    "LeaveOneGroupOut",
    "LeaveOneOut",
    "LeavePGroupsOut",
    "LeavePOut",
    "PredefinedSplit",
    "RepeatedKFold",
    "RepeatedStratifiedKFold",
    "ShuffleSplit",
    "StratifiedGroupKFold",
    "StratifiedKFold",
    "StratifiedShuffleSplit",
    "check_cv",
    "train_test_split",
]


class _UnsupportedGroupCVMixin:
    # [CLASS DOC]
    # HELPER: Mixin for splitters that ignore groups; warns when groups is supplied.
    """Mixin for splitters that do not support Groups."""

    def split(self, X, y=None, groups=None):
        # [FUNCTION DOC]
        # Emits a UserWarning if groups is not None, then delegates to the next split in
        # the MRO (groups still passed along). Outputs: that generator.
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        y : array-like of shape (n_samples,), default=None
            The target variable for supervised learning problems.

        groups : array-like of shape (n_samples,), default=None
            Always ignored, exists for API compatibility.

        Yields
        ------
        train : ndarray
            The training set indices for that split.

        test : ndarray
            The testing set indices for that split.
        """
        if groups is not None:
            warnings.warn(
                f"The groups parameter is ignored by {self.__class__.__name__}",
                UserWarning,
            )
        return super().split(X, y, groups=groups)


class GroupsConsumerMixin(_MetadataRequester):
    # [CLASS DOC]
    # Mixin setting the metadata request for split so groups is requested by default.
    """A Mixin to ``groups`` by default.

    This Mixin makes the object to request ``groups`` by default as ``True``.

    .. versionadded:: 1.3
    """

    __metadata_request__split = {"groups": True}


class BaseCrossValidator(_MetadataRequester, metaclass=ABCMeta):
    # [CLASS DOC]
    # Abstract base for index-based splitters. Subclasses implement either
    # _iter_test_masks or _iter_test_indices, plus get_n_splits. groups is marked UNUSED
    # for metadata routing unless a groups mixin overrides it.
    """Base class for all cross-validators.

    Implementations must define `_iter_test_masks` or `_iter_test_indices`.
    """

    # This indicates that by default CV splitters don't have a "groups" kwarg,
    # unless indicated by inheriting from ``GroupsConsumerMixin``.
    # This also prevents ``set_split_request`` to be generated for splitters
    # which don't support ``groups``.
    __metadata_request__split = {"groups": metadata_routing.UNUSED}

    def split(self, X, y=None, groups=None):
        # [FUNCTION DOC]
        # Makes X, y, groups indexable, then for each boolean test mask yields the
        # indices where the mask is False as train and True as test.
        # Inputs: X - array-like (n_samples, ...); y, groups - optional (n_samples,).
        # Outputs: generator of (train, test) int ndarrays.
        # Exceptions: inconsistent lengths from indexable; whatever the mask iterator
        # raises.
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        y : array-like of shape (n_samples,)
            The target variable for supervised learning problems.

        groups : array-like of shape (n_samples,), default=None
            Group labels for the samples used while splitting the dataset into
            train/test set.

        Yields
        ------
        train : ndarray
            The training set indices for that split.

        test : ndarray
            The testing set indices for that split.
        """
        X, y, groups = indexable(X, y, groups)
        indices = np.arange(_num_samples(X))
        for test_index in self._iter_test_masks(X, y, groups):
            train_index = indices[np.logical_not(test_index)]
            test_index = indices[test_index]
            yield train_index, test_index

    # Since subclasses must implement either _iter_test_masks or
    # _iter_test_indices, neither can be abstract.
    def _iter_test_masks(self, X=None, y=None, groups=None):
        # [FUNCTION DOC]
        # Default: converts each index set from _iter_test_indices into a boolean mask
        # of length n_samples. Outputs: generator of bool ndarrays.
        """Generates boolean masks corresponding to test sets.

        By default, delegates to _iter_test_indices(X, y, groups)
        """
        for test_index in self._iter_test_indices(X, y, groups):
            test_mask = np.zeros(_num_samples(X), dtype=bool)
            test_mask[test_index] = True
            yield test_mask

    def _iter_test_indices(self, X=None, y=None, groups=None):
        # [FUNCTION DOC]
        # Default raises NotImplementedError; subclasses override.
        """Generates integer indices corresponding to test sets."""
        raise NotImplementedError

    @abstractmethod
    def get_n_splits(self, X=None, y=None, groups=None):
        # [FUNCTION DOC]
        # Abstract: returns the number of splits.
        """Returns the number of splitting iterations in the cross-validator."""

    def __repr__(self):
        # [FUNCTION DOC]
        # Returns _build_repr(self), a ClassName(param=value, ...) string.
        return _build_repr(self)


class LeaveOneOut(_UnsupportedGroupCVMixin, BaseCrossValidator):
    # [CLASS DOC]
    # Each sample in turn forms a singleton test set.
    """Leave-One-Out cross-validator.

    Provides train/test indices to split data in train/test sets. Each
    sample is used once as a test set (singleton) while the remaining
    samples form the training set.

    Note: ``LeaveOneOut()`` is equivalent to ``KFold(n_splits=n)`` and
    ``LeavePOut(p=1)`` where ``n`` is the number of samples.

    Due to the high number of test sets (which is the same as the
    number of samples) this cross-validation method can be very costly.
    For large datasets one should favor :class:`KFold`, :class:`ShuffleSplit`
    or :class:`StratifiedKFold`.

    Read more in the :ref:`User Guide <leave_one_out>`.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.model_selection import LeaveOneOut
    >>> X = np.array([[1, 2], [3, 4]])
    >>> y = np.array([1, 2])
    >>> loo = LeaveOneOut()
    >>> loo.get_n_splits(X)
    2
    >>> print(loo)
    LeaveOneOut()
    >>> for i, (train_index, test_index) in enumerate(loo.split(X)):
    ...     print(f"Fold {i}:")
    ...     print(f"  Train: index={train_index}")
    ...     print(f"  Test:  index={test_index}")
    Fold 0:
      Train: index=[1]
      Test:  index=[0]
    Fold 1:
      Train: index=[0]
      Test:  index=[1]

    See Also
    --------
    LeaveOneGroupOut : For splitting the data according to explicit,
        domain-specific grouping of the dataset.
    GroupKFold : K-fold iterator variant with non-overlapping groups.
    """

    def _iter_test_indices(self, X, y=None, groups=None):
        # [FUNCTION DOC]
        # Returns range(n_samples). Raises ValueError if n_samples <= 1.
        n_samples = _num_samples(X)
        if n_samples <= 1:
            raise ValueError(
                "Cannot perform LeaveOneOut with n_samples={}.".format(n_samples)
            )
        return range(n_samples)

    def get_n_splits(self, X, y=None, groups=None):
        # [FUNCTION DOC]
        # Returns the number of samples in X. y and groups: Always ignored, exists for
        # API compatibility.
        # Exceptions: ValueError if X is None.
        """Returns the number of splitting iterations in the cross-validator.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        y : array-like of shape (n_samples,), default=None
            Always ignored, exists for API compatibility.

        groups : array-like of shape (n_samples,), default=None
            Always ignored, exists for API compatibility.

        Returns
        -------
        n_splits : int
            Returns the number of splitting iterations in the cross-validator.
        """
        if X is None:
            raise ValueError("The 'X' parameter should not be None.")
        return _num_samples(X)


class LeavePOut(_UnsupportedGroupCVMixin, BaseCrossValidator):
    # [CLASS DOC]
    # Every distinct subset of p samples forms a test set (n choose p splits).
    """Leave-P-Out cross-validator.

    Provides train/test indices to split data in train/test sets. This results
    in testing on all distinct samples of size p, while the remaining n - p
    samples form the training set in each iteration.

    Note: ``LeavePOut(p)`` is NOT equivalent to
    ``KFold(n_splits=n_samples // p)`` which creates non-overlapping test sets.

    Due to the high number of iterations which grows combinatorically with the
    number of samples this cross-validation method can be very costly. For
    large datasets one should favor :class:`KFold`, :class:`StratifiedKFold`
    or :class:`ShuffleSplit`.

    Read more in the :ref:`User Guide <leave_p_out>`.

    Parameters
    ----------
    p : int
        Size of the test sets. Must be strictly less than the number of
        samples.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.model_selection import LeavePOut
    >>> X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    >>> y = np.array([1, 2, 3, 4])
    >>> lpo = LeavePOut(2)
    >>> lpo.get_n_splits(X)
    6
    >>> print(lpo)
    LeavePOut(p=2)
    >>> for i, (train_index, test_index) in enumerate(lpo.split(X)):
    ...     print(f"Fold {i}:")
    ...     print(f"  Train: index={train_index}")
    ...     print(f"  Test:  index={test_index}")
    Fold 0:
      Train: index=[2 3]
      Test:  index=[0 1]
    Fold 1:
      Train: index=[1 3]
      Test:  index=[0 2]
    Fold 2:
      Train: index=[1 2]
      Test:  index=[0 3]
    Fold 3:
      Train: index=[0 3]
      Test:  index=[1 2]
    Fold 4:
      Train: index=[0 2]
      Test:  index=[1 3]
    Fold 5:
      Train: index=[0 1]
      Test:  index=[2 3]
    """

    def __init__(self, p):
        # [FUNCTION DOC]
        # Stores p (int test-set size) without validation.
        self.p = p

    def _iter_test_indices(self, X, y=None, groups=None):
        # [FUNCTION DOC]
        # Yields each combination of p sample indices in lexicographic order as an
        # array. Raises ValueError if n_samples <= p.
        n_samples = _num_samples(X)
        if n_samples <= self.p:
            raise ValueError(
                "p={} must be strictly less than the number of samples={}".format(
                    self.p, n_samples
                )
            )
        for combination in combinations(range(n_samples), self.p):
            yield np.array(combination)

    def get_n_splits(self, X, y=None, groups=None):
        # [FUNCTION DOC]
        # Returns the exact binomial coefficient C(n_samples, p) as int. Raises
        # ValueError if X is None.
        """Returns the number of splitting iterations in the cross-validator.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        y : array-like of shape (n_samples,), default=None
            Always ignored, exists for API compatibility.

        groups : array-like of shape (n_samples,), default=None
            Always ignored, exists for API compatibility.
        """
        if X is None:
            raise ValueError("The 'X' parameter should not be None.")
        return int(comb(_num_samples(X), self.p, exact=True))


class _BaseKFold(BaseCrossValidator, metaclass=ABCMeta):
    # [CLASS DOC]
    # Abstract base for K-fold style splitters holding n_splits, shuffle and
    # random_state, validated eagerly in the constructor.
    """Base class for K-Fold cross-validators and TimeSeriesSplit."""

    @abstractmethod
    def __init__(self, n_splits, *, shuffle, random_state):
        # [FUNCTION DOC]
        # Validates and stores the parameters.
        # Inputs: n_splits - Integral >= 2; shuffle - bool; random_state - must be None
        # when shuffle is False.
        # Exceptions: ValueError if n_splits is not Integral or <= 1, or random_state is
        # set with shuffle=False; TypeError if shuffle is not a bool.
        # Side effects: sets n_splits (as int), shuffle, random_state.
        if not isinstance(n_splits, numbers.Integral):
            raise ValueError(
                "The number of folds must be of Integral type. "
                "%s of type %s was passed." % (n_splits, type(n_splits))
            )
        n_splits = int(n_splits)

        if n_splits <= 1:
            raise ValueError(
                "k-fold cross-validation requires at least one"
                " train/test split by setting n_splits=2 or more,"
                " got n_splits={0}.".format(n_splits)
            )

        if not isinstance(shuffle, bool):
            raise TypeError("shuffle must be True or False; got {0}".format(shuffle))

        if not shuffle and random_state is not None:  # None is the default
            raise ValueError(
                (
                    "Setting a random_state has no effect since shuffle is "
                    "False. You should leave "
                    "random_state to its default (None), or set shuffle=True."
                ),
            )

        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X, y=None, groups=None):
        # [FUNCTION DOC]
        # Checks n_splits <= n_samples (ValueError otherwise) and yields from
        # BaseCrossValidator.split.
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        y : array-like of shape (n_samples,), default=None
            The target variable for supervised learning problems.

        groups : array-like of shape (n_samples,), default=None
            Group labels for the samples used while splitting the dataset into
            train/test set.

        Yields
        ------
        train : ndarray
            The training set indices for that split.

        test : ndarray
            The testing set indices for that split.
        """
        X, y, groups = indexable(X, y, groups)
        n_samples = _num_samples(X)
        if self.n_splits > n_samples:
            raise ValueError(
                (
                    "Cannot have number of splits n_splits={0} greater"
                    " than the number of samples: n_samples={1}."
                ).format(self.n_splits, n_samples)
            )

        for train, test in super().split(X, y, groups):
            yield train, test

    def get_n_splits(self, X=None, y=None, groups=None):
        # [FUNCTION DOC]
        # Returns n_splits; X, y and groups: Always ignored, exists for API
        # compatibility.
        """Returns the number of splitting iterations as set with the `n_splits` param
        when instantiating the cross-validator.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features), default=None
            Always ignored, exists for API compatibility.

        y : array-like of shape (n_samples,), default=None
            Always ignored, exists for API compatibility.

        groups : array-like of shape (n_samples,), default=None
            Always ignored, exists for API compatibility.

        Returns
        -------
        n_splits : int
            Returns the number of splitting iterations in the cross-validator.
        """
        return self.n_splits


class KFold(_UnsupportedGroupCVMixin, _BaseKFold):
    # [CLASS DOC]
    # Splits samples into n_splits consecutive folds (optionally after shuffling).
    """K-Fold cross-validator.

    Provides train/test indices to split data in train/test sets. Split
    dataset into k consecutive folds (without shuffling by default).

    Each fold is then used once as a validation while the k - 1 remaining
    folds form the training set.

    Read more in the :ref:`User Guide <k_fold>`.

    For visualisation of cross-validation behaviour and
    comparison between common scikit-learn split methods
    refer to :ref:`sphx_glr_auto_examples_model_selection_plot_cv_indices.py`

    Parameters
    ----------
    n_splits : int, default=5
        Number of folds. Must be at least 2.

        .. versionchanged:: 0.22
            ``n_splits`` default value changed from 3 to 5.

    shuffle : bool, default=False
        Whether to shuffle the data before splitting into batches.
        Note that the samples within each split will not be shuffled.

    random_state : int, RandomState instance or None, default=None
        When `shuffle` is True, `random_state` affects the ordering of the
        indices, which controls the randomness of each fold. Otherwise, this
        parameter has no effect.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.model_selection import KFold
    >>> X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
    >>> y = np.array([1, 2, 3, 4])
    >>> kf = KFold(n_splits=2)
    >>> kf.get_n_splits()
    2
    >>> print(kf)
    KFold(n_splits=2, random_state=None, shuffle=False)
    >>> for i, (train_index, test_index) in enumerate(kf.split(X)):
    ...     print(f"Fold {i}:")
    ...     print(f"  Train: index={train_index}")
    ...     print(f"  Test:  index={test_index}")
    Fold 0:
      Train: index=[2 3]
      Test:  index=[0 1]
    Fold 1:
      Train: index=[0 1]
      Test:  index=[2 3]

    Notes
    -----
    The first ``n_samples % n_splits`` folds have size
    ``n_samples // n_splits + 1``, other folds have size
    ``n_samples // n_splits``, where ``n_samples`` is the number of samples.

    Randomized CV splitters may return different results for each call of
    split. You can make the results identical by setting `random_state`
    to an integer.

    See Also
    --------
    StratifiedKFold : Takes class information into account to avoid building
        folds with imbalanced class distributions (for binary or multiclass
        classification tasks).

    GroupKFold : K-fold iterator variant with non-overlapping groups.

    RepeatedKFold : Repeats K-Fold n times.
    """

    def __init__(self, n_splits=5, *, shuffle=False, random_state=None):
        # [FUNCTION DOC]
        # Forwards to _BaseKFold.__init__ (default n_splits=5).
        super().__init__(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    def _iter_test_indices(self, X, y=None, groups=None):
        # [FUNCTION DOC]
        # Builds arange(n_samples), shuffles it in place with random_state if shuffle,
        # and yields consecutive slices: the first n_samples % n_splits folds get one
        # extra sample. Side effects: randomness when shuffle.
        n_samples = _num_samples(X)
        indices = np.arange(n_samples)
        if self.shuffle:
            check_random_state(self.random_state).shuffle(indices)

        n_splits = self.n_splits
        fold_sizes = np.full(n_splits, n_samples // n_splits, dtype=int)
        fold_sizes[: n_samples % n_splits] += 1
        current = 0
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            yield indices[start:stop]
            current = stop


class GroupKFold(GroupsConsumerMixin, _BaseKFold):
    # [CLASS DOC]
    # K-fold where each group appears in exactly one test fold.
    """K-fold iterator variant with non-overlapping groups.

    Each group will appear exactly once in the test set across all folds (the
    number of distinct groups has to be at least equal to the number of folds).

    The folds are approximately balanced in the sense that the number of
    samples is approximately the same in each test fold when `shuffle` is True.

    Read more in the :ref:`User Guide <group_k_fold>`.

    For visualisation of cross-validation behaviour and
    comparison between common scikit-learn split methods
    refer to :ref:`sphx_glr_auto_examples_model_selection_plot_cv_indices.py`

    Parameters
    ----------
    n_splits : int, default=5
        Number of folds. Must be at least 2.

        .. versionchanged:: 0.22
            ``n_splits`` default value changed from 3 to 5.

    shuffle : bool, default=False
        Whether to shuffle the groups before splitting into batches.
        Note that the samples within each split will not be shuffled.

        .. versionadded:: 1.6

    random_state : int, RandomState instance or None, default=None
        When `shuffle` is True, `random_state` affects the ordering of the
        indices, which controls the randomness of each fold. Otherwise, this
        parameter has no effect.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

        .. versionadded:: 1.6

    Notes
    -----
    Groups appear in an arbitrary order throughout the folds.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.model_selection import GroupKFold
    >>> X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])
    >>> y = np.array([1, 2, 3, 4, 5, 6])
    >>> groups = np.array([0, 0, 2, 2, 3, 3])
    >>> group_kfold = GroupKFold(n_splits=2)
    >>> group_kfold.get_n_splits()
    2
    >>> print(group_kfold)
    GroupKFold(n_splits=2, random_state=None, shuffle=False)
    >>> for i, (train_index, test_index) in enumerate(group_kfold.split(X, y, groups)):
    ...     print(f"Fold {i}:")
    ...     print(f"  Train: index={train_index}, group={groups[train_index]}")
    ...     print(f"  Test:  index={test_index}, group={groups[test_index]}")
    Fold 0:
      Train: index=[2 3], group=[2 2]
      Test:  index=[0 1 4 5], group=[0 0 3 3]
    Fold 1:
      Train: index=[0 1 4 5], group=[0 0 3 3]
      Test:  index=[2 3], group=[2 2]

    See Also
    --------
    LeaveOneGroupOut : For splitting the data according to explicit,
        domain-specific grouping of the dataset.

    StratifiedKFold : Takes class information into account to avoid building
        folds with imbalanced class proportions (for binary or multiclass
        classification tasks).
    """

    def __init__(self, n_splits=5, *, shuffle=False, random_state=None):
        # [FUNCTION DOC]
        # Forwards to _BaseKFold.__init__.
        super().__init__(n_splits, shuffle=shuffle, random_state=random_state)

    def _iter_test_indices(self, X, y, groups):
        # [FUNCTION DOC]
        # Assigns groups to folds and yields per-fold sample indices.
        # With shuffle: permute unique groups with random_state, split them into
        # n_splits chunks of near-equal group count, and yield samples whose group is in
        # each chunk. Without shuffle: greedy bin-packing - sort groups by sample count
        # descending (stable), assign each to the fold currently holding the fewest
        # samples, then yield samples per fold.
        # Inputs: groups - array-like (n_samples,), required.
        # Exceptions: ValueError if groups is None or n_splits exceeds the number of
        # groups.
        # KNOWN ISSUE: the class docstring says folds are balanced in sample count when
        # shuffle is True, but the shuffle branch balances the number of groups;
        # sample-count balancing is what the non-shuffle branch does.
        if groups is None:
            raise ValueError("The 'groups' parameter should not be None.")
        groups = check_array(groups, input_name="groups", ensure_2d=False, dtype=None)

        unique_groups, group_idx = np.unique(groups, return_inverse=True)
        n_groups = len(unique_groups)

        if self.n_splits > n_groups:
            raise ValueError(
                "Cannot have number of splits n_splits=%d greater"
                " than the number of groups: %d." % (self.n_splits, n_groups)
            )

        if self.shuffle:
            # Split and shuffle unique groups across n_splits
            rng = check_random_state(self.random_state)
            unique_groups = rng.permutation(unique_groups)
            split_groups = np.array_split(unique_groups, self.n_splits)

            for test_group_ids in split_groups:
                test_mask = np.isin(groups, test_group_ids)
                yield np.where(test_mask)[0]

        else:
            # Weight groups by their number of occurrences
            n_samples_per_group = np.bincount(group_idx)

            # Distribute the most frequent groups first
            indices = np.argsort(n_samples_per_group, kind="stable")[::-1]
            n_samples_per_group = n_samples_per_group[indices]

            # Total weight of each fold
            n_samples_per_fold = np.zeros(self.n_splits)

            # Mapping from group index to fold index
            group_to_fold = np.zeros(len(unique_groups))

            # Distribute samples by adding the largest weight to the lightest fold
            for group_index, weight in enumerate(n_samples_per_group):
                lightest_fold = np.argmin(n_samples_per_fold)
                n_samples_per_fold[lightest_fold] += weight
                group_to_fold[indices[group_index]] = lightest_fold

            indices = group_to_fold[group_idx]

            for f in range(self.n_splits):
                yield np.where(indices == f)[0]

    def split(self, X, y=None, groups=None):
        # [FUNCTION DOC]
        # Delegates to _BaseKFold.split; exists to document that groups is required.
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        y : array-like of shape (n_samples,), default=None
            The target variable for supervised learning problems.

        groups : array-like of shape (n_samples,)
            Group labels for the samples used while splitting the dataset into
            train/test set.

        Yields
        ------
        train : ndarray
            The training set indices for that split.

        test : ndarray
            The testing set indices for that split.
        """
        return super().split(X, y, groups)


class StratifiedKFold(_BaseKFold):
    # [CLASS DOC]
    # K-fold preserving class proportions of y in every test fold.
    """Class-wise stratified K-Fold cross-validator.

    Provides train/test indices to split data in train/test sets.

    This cross-validation object is a variation of KFold that returns
    stratified folds. The folds are made by preserving the percentage of
    samples for each class in `y` in a binary or multiclass classification
    setting.

    Read more in the :ref:`User Guide <stratified_k_fold>`.

    For visualisation of cross-validation behaviour and
    comparison between common scikit-learn split methods
    refer to :ref:`sphx_glr_auto_examples_model_selection_plot_cv_indices.py`

    .. note::

        Stratification on the class label solves an engineering problem rather
        than a statistical one. See :ref:`stratification` for more details.

    Parameters
    ----------
    n_splits : int, default=5
        Number of folds. Must be at least 2.

        .. versionchanged:: 0.22
            ``n_splits`` default value changed from 3 to 5.

    shuffle : bool, default=False
        Whether to shuffle each class's samples before splitting into batches.
        Note that the samples within each split will not be shuffled.

    random_state : int, RandomState instance or None, default=None
        When `shuffle` is True, `random_state` affects the ordering of the
        indices, which controls the randomness of each fold for each class.
        Otherwise, leave `random_state` as `None`.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.model_selection import StratifiedKFold
    >>> X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
    >>> y = np.array([0, 0, 1, 1])
    >>> skf = StratifiedKFold(n_splits=2)
    >>> skf.get_n_splits()
    2
    >>> print(skf)
    StratifiedKFold(n_splits=2, random_state=None, shuffle=False)
    >>> for i, (train_index, test_index) in enumerate(skf.split(X, y)):
    ...     print(f"Fold {i}:")
    ...     print(f"  Train: index={train_index}")
    ...     print(f"  Test:  index={test_index}")
    Fold 0:
      Train: index=[1 3]
      Test:  index=[0 2]
    Fold 1:
      Train: index=[0 2]
      Test:  index=[1 3]

    Notes
    -----
    The implementation is designed to:

    * Generate test sets such that all contain the same distribution of
      classes, or as close as possible.
    * Be invariant to class label: relabelling ``y = ["Happy", "Sad"]`` to
      ``y = [1, 0]`` should not change the indices generated.
    * Preserve order dependencies in the dataset ordering, when
      ``shuffle=False``: all samples from class k in some test set were
      contiguous in y, or separated in y by samples from classes other than k.
    * Generate test sets where the smallest and largest differ by at most one
      sample.

    .. versionchanged:: 0.22
        The previous implementation did not follow the last constraint.

    See Also
    --------
    RepeatedStratifiedKFold : Repeats Stratified K-Fold n times.
    """

    def __init__(self, n_splits=5, *, shuffle=False, random_state=None):
        # [FUNCTION DOC]
        # Forwards to _BaseKFold.__init__.
        super().__init__(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    def _make_test_folds(self, X, y=None):
        # [FUNCTION DOC]
        # Computes, for every sample, the index of the test fold it belongs to.
        # Steps: convert y to a NumPy 1d array (moving array API inputs to CPU); require
        # a binary or multiclass target; encode classes by order of first appearance;
        # error if every class has fewer than n_splits members and warn if the smallest
        # does; compute per-fold per-class allocations by dealing the sorted encoded
        # labels round-robin over folds; for each class, assign its samples (in data
        # order) to folds in contiguous blocks matching the allocation, shuffling those
        # fold labels when shuffle.
        # Inputs: X - unused; y - class labels (n_samples,).
        # Outputs: int ndarray (n_samples,) of fold ids.
        # Exceptions: ValueError for unsupported target types or too few class members.
        # Side effects: UserWarning when the least populated class has fewer than
        # n_splits members; randomness when shuffle.
        rng = check_random_state(self.random_state)
        # XXX: as of now, cross-validation splitters only operate in NumPy-land
        # without attempting to leverage array API namespace features. However
        # they might be fed by array API inputs, e.g. in CV-enabled estimators so
        # we need the following explicit conversion:
        xp, is_array_api = get_namespace(y)
        if is_array_api:
            y = move_to(y, xp=np, device="cpu")
        else:
            y = np.asarray(y)
        type_of_target_y = type_of_target(y)
        allowed_target_types = ("binary", "multiclass")
        if type_of_target_y not in allowed_target_types:
            raise ValueError(
                "Supported target types are: {}. Got {!r} instead.".format(
                    allowed_target_types, type_of_target_y
                )
            )

        y = column_or_1d(y)

        _, y_idx, y_inv = np.unique(y, return_index=True, return_inverse=True)
        # y_inv encodes y according to lexicographic order. We invert y_idx to
        # map the classes so that they are encoded by order of appearance:
        # 0 represents the first label appearing in y, 1 the second, etc.
        _, class_perm = np.unique(y_idx, return_inverse=True)
        y_encoded = class_perm[y_inv]

        n_classes = len(y_idx)
        y_counts = np.bincount(y_encoded)
        min_groups = np.min(y_counts)
        if np.all(self.n_splits > y_counts):
            raise ValueError(
                "n_splits=%d cannot be greater than the"
                " number of members in each class." % (self.n_splits)
            )
        if self.n_splits > min_groups:
            warnings.warn(
                "The least populated class in y has only %d"
                " members, which is less than n_splits=%d."
                % (min_groups, self.n_splits),
                UserWarning,
            )

        # Determine the optimal number of samples from each class in each fold,
        # using round robin over the sorted y. (This can be done direct from
        # counts, but that code is unreadable.)
        y_order = np.sort(y_encoded)
        allocation = np.asarray(
            [
                np.bincount(y_order[i :: self.n_splits], minlength=n_classes)
                for i in range(self.n_splits)
            ]
        )

        # To maintain the data order dependencies as best as possible within
        # the stratification constraint, we assign samples from each class in
        # blocks (and then mess that up when shuffle=True).
        test_folds = np.empty(len(y), dtype="i")
        for k in range(n_classes):
            # since the kth column of allocation stores the number of samples
            # of class k in each test set, this generates blocks of fold
            # indices corresponding to the allocation for class k.
            folds_for_class = np.arange(self.n_splits).repeat(allocation[:, k])
            if self.shuffle:
                rng.shuffle(folds_for_class)
            test_folds[y_encoded == k] = folds_for_class
        return test_folds

    def _iter_test_masks(self, X, y=None, groups=None):
        # [FUNCTION DOC]
        # Yields, for each fold i, the mask of samples whose test fold equals i.
        test_folds = self._make_test_folds(X, y)
        for i in range(self.n_splits):
            yield test_folds == i

    def split(self, X, y, groups=None):
        # [FUNCTION DOC]
        # Warns if groups is given, validates y as an array (not forced 2d, any dtype)
        # and delegates to _BaseKFold.split.
        # Exceptions: ValueError from check_array for invalid y.
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.

            Note that providing ``y`` is sufficient to generate the splits and
            hence ``np.zeros(n_samples)`` may be used as a placeholder for
            ``X`` instead of actual training data.

        y : array-like of shape (n_samples,)
            The target variable for supervised learning problems.
            Stratification is done based on the y labels.

        groups : array-like of shape (n_samples,), default=None
            Always ignored, exists for API compatibility.

        Yields
        ------
        train : ndarray
            The training set indices for that split.

        test : ndarray
            The testing set indices for that split.

        Notes
        -----
        Randomized CV splitters may return different results for each call of
        split. You can make the results identical by setting `random_state`
        to an integer.
        """
        if groups is not None:
            warnings.warn(
                f"The groups parameter is ignored by {self.__class__.__name__}",
                UserWarning,
            )
        y = check_array(y, input_name="y", ensure_2d=False, dtype=None)
        return super().split(X, y, groups)


class StratifiedGroupKFold(GroupsConsumerMixin, _BaseKFold):
    # [CLASS DOC]
    # Group-aware K-fold that greedily assigns whole groups to folds so that class
    # proportions across folds stay as even as possible.
    """Class-wise stratified K-Fold iterator variant with non-overlapping groups.

    This cross-validation object is a variation of :class:`StratifiedKFold` that
    attempts to return stratified folds with non-overlapping groups. The folds are made
    by preserving the percentage of samples for each class in `y` in a binary or
    multiclass classification setting.

    Each group will appear exactly once in the test set across all folds (the
    number of distinct groups has to be at least equal to the number of folds).

    The difference between :class:`GroupKFold`
    and `StratifiedGroupKFold` is that
    the former attempts to create balanced folds such that the number of
    distinct groups is approximately the same in each fold, whereas
    `StratifiedGroupKFold` attempts to create folds which preserve the
    percentage of samples from each class as much as possible given the
    constraint of non-overlapping groups between splits.

    Read more in the :ref:`User Guide <stratified_group_k_fold>`.

    For visualisation of cross-validation behaviour and
    comparison between common scikit-learn split methods
    refer to :ref:`sphx_glr_auto_examples_model_selection_plot_cv_indices.py`

    .. note::

        Stratification on the class label solves an engineering problem rather
        than a statistical one. See :ref:`stratification` for more details.

    Parameters
    ----------
    n_splits : int, default=5
        Number of folds. Must be at least 2.

    shuffle : bool, default=False
        Whether to shuffle each class's samples before splitting into batches.
        Note that the samples within each split will not be shuffled.
        This implementation can only shuffle groups that have approximately the
        same `y` class distribution, no global shuffle will be performed.

    random_state : int or RandomState instance, default=None
        When `shuffle` is True, `random_state` affects the ordering of the
        indices, which controls the randomness of each fold for each class.
        Otherwise, leave `random_state` as `None`.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.model_selection import StratifiedGroupKFold
    >>> X = np.ones((17, 2))
    >>> y = np.array([0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    >>> groups = np.array([1, 1, 2, 2, 3, 3, 3, 4, 5, 5, 5, 5, 6, 6, 7, 8, 8])
    >>> sgkf = StratifiedGroupKFold(n_splits=3)
    >>> sgkf.get_n_splits()
    3
    >>> print(sgkf)
    StratifiedGroupKFold(n_splits=3, random_state=None, shuffle=False)
    >>> for i, (train_index, test_index) in enumerate(sgkf.split(X, y, groups)):
    ...     print(f"Fold {i}:")
    ...     print(f"  Train: index={train_index}")
    ...     print(f"         group={groups[train_index]}")
    ...     print(f"  Test:  index={test_index}")
    ...     print(f"         group={groups[test_index]}")
    Fold 0:
      Train: index=[ 0  1  2  3  7  8  9 10 11 15 16]
             group=[1 1 2 2 4 5 5 5 5 8 8]
      Test:  index=[ 4  5  6 12 13 14]
             group=[3 3 3 6 6 7]
    Fold 1:
      Train: index=[ 4  5  6  7  8  9 10 11 12 13 14]
             group=[3 3 3 4 5 5 5 5 6 6 7]
      Test:  index=[ 0  1  2  3 15 16]
             group=[1 1 2 2 8 8]
    Fold 2:
      Train: index=[ 0  1  2  3  4  5  6 12 13 14 15 16]
             group=[1 1 2 2 3 3 3 6 6 7 8 8]
      Test:  index=[ 7  8  9 10 11]
             group=[4 5 5 5 5]

    Notes
    -----
    The implementation is designed to:

    * Mimic the behavior of :class:`StratifiedKFold` as much as possible for trivial
      groups (e.g. when each group contains only one sample).
    * Be invariant to class label: relabelling ``y = ["Happy", "Sad"]`` to
      ``y = [1, 0]`` should not change the indices generated.
    * Stratify based on samples as much as possible while keeping
      non-overlapping groups constraint. That means that in some cases when
      there is a small number of groups containing a large number of samples
      the stratification will not be possible and the behavior will be close
      to :class:`GroupKFold`.

    See also
    --------
    StratifiedKFold: Takes class information into account to build folds which
        retain class distributions (for binary or multiclass classification
        tasks).

    GroupKFold: K-fold iterator variant with non-overlapping groups.
    """

    def __init__(self, n_splits=5, shuffle=False, random_state=None):
        # [FUNCTION DOC]
        # Forwards to _BaseKFold.__init__.
        # KNOWN ISSUE: shuffle and random_state are positional-or-keyword here, unlike
        # the keyword-only signature of the other K-fold splitters.
        super().__init__(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    def _iter_test_indices(self, X, y, groups):
        # [FUNCTION DOC]
        # Assigns groups to folds and yields each fold's test sample indices.
        # Steps: require a binary/multiclass y; error if every class has fewer than
        # n_splits members, warn if the smallest does; error if n_splits exceeds the
        # number of groups; build a groups x classes count matrix; if shuffle, permute
        # group order with random_state; sort groups by descending std of their class
        # counts (stable); place each group into the fold chosen by _find_best_fold and
        # update fold class counts; finally yield, per fold, the list of sample indices
        # whose group is in that fold.
        # Inputs: y - labels (n_samples,); groups - group labels (n_samples,).
        # Outputs: generator of Python lists of int indices.
        # Exceptions: ValueError for unsupported target, too few class members, or too
        # few groups.
        # Side effects: UserWarning; randomness when shuffle.
        # KNOWN ISSUE: groups=None is not explicitly checked (np.unique(None) yields one
        # object group, leading to a less clear 'number of groups' error).
        # Implementation is based on this kaggle kernel:
        # https://www.kaggle.com/jakubwasikowski/stratified-group-k-fold-cross-validation
        # and is a subject to Apache 2.0 License. You may obtain a copy of the
        # License at http://www.apache.org/licenses/LICENSE-2.0
        # Changelist:
        # - Refactored function to a class following scikit-learn KFold
        #   interface.
        # - Added heuristic for assigning group to the least populated fold in
        #   cases when all other criteria are equal
        # - Swtch from using python ``Counter`` to ``np.unique`` to get class
        #   distribution
        # - Added scikit-learn checks for input: checking that target is binary
        #   or multiclass, checking passed random state, checking that number
        #   of splits is less than number of members in each class, checking
        #   that least populated class has more members than there are splits.
        rng = check_random_state(self.random_state)
        y = np.asarray(y)
        type_of_target_y = type_of_target(y)
        allowed_target_types = ("binary", "multiclass")
        if type_of_target_y not in allowed_target_types:
            raise ValueError(
                "Supported target types are: {}. Got {!r} instead.".format(
                    allowed_target_types, type_of_target_y
                )
            )

        y = column_or_1d(y)
        _, y_inv, y_cnt = np.unique(y, return_inverse=True, return_counts=True)
        if np.all(self.n_splits > y_cnt):
            raise ValueError(
                "n_splits=%d cannot be greater than the"
                " number of members in each class." % (self.n_splits)
            )
        n_smallest_class = np.min(y_cnt)
        if self.n_splits > n_smallest_class:
            warnings.warn(
                "The least populated class in y has only %d"
                " members, which is less than n_splits=%d."
                % (n_smallest_class, self.n_splits),
                UserWarning,
            )
        n_classes = len(y_cnt)

        _, groups_inv, groups_cnt = np.unique(
            groups, return_inverse=True, return_counts=True
        )
        n_groups = len(groups_cnt)

        if self.n_splits > n_groups:
            raise ValueError(
                f"Cannot have number of splits n_splits={self.n_splits} greater"
                f" than the number of groups: {n_groups}."
            )

        y_counts_per_group = np.zeros((len(groups_cnt), n_classes))
        for class_idx, group_idx in zip(y_inv, groups_inv):
            y_counts_per_group[group_idx, class_idx] += 1

        y_counts_per_fold = np.zeros((self.n_splits, n_classes))
        groups_per_fold = defaultdict(set)

        if self.shuffle:
            perm = np.arange(len(groups_cnt))
            rng.shuffle(perm)
            y_counts_per_group = y_counts_per_group[perm]
            inv_perm = np.empty_like(perm)
            inv_perm[perm] = np.arange(perm.size)
            groups_inv = inv_perm[groups_inv]

        # Stable sort to keep shuffled order for groups with the same
        # class distribution variance
        sorted_groups_idx = np.argsort(
            -np.std(y_counts_per_group, axis=1), kind="stable"
        )

        for group_idx in sorted_groups_idx:
            group_y_counts = y_counts_per_group[group_idx]
            best_fold = self._find_best_fold(
                y_counts_per_fold=y_counts_per_fold,
                y_cnt=y_cnt,
                group_y_counts=group_y_counts,
            )
            y_counts_per_fold[best_fold] += group_y_counts
            groups_per_fold[best_fold].add(group_idx)

        for i in range(self.n_splits):
            test_indices = [
                idx
                for idx, group_idx in enumerate(groups_inv)
                if group_idx in groups_per_fold[i]
            ]
            yield test_indices

    def _find_best_fold(self, y_counts_per_fold, y_cnt, group_y_counts):
        # [FUNCTION DOC]
        # HELPER: For each fold, tentatively adds group_y_counts and evaluates the mean
        # over classes of the std across folds of the per-class proportions (counts
        # divided by class totals). Picks the fold with the lowest value, breaking
        # near-ties (isclose) by fewest samples already in the fold.
        # Inputs: y_counts_per_fold - (n_splits, n_classes) float array; y_cnt -
        # per-class totals; group_y_counts - the group's per-class counts.
        # Outputs: int fold index.
        # Side effects: temporarily mutates y_counts_per_fold but restores it.
        best_fold = None
        min_eval = np.inf
        min_samples_in_fold = np.inf
        for i in range(self.n_splits):
            y_counts_per_fold[i] += group_y_counts
            # Summarise the distribution over classes in each proposed fold
            std_per_class = np.std(y_counts_per_fold / y_cnt.reshape(1, -1), axis=0)
            y_counts_per_fold[i] -= group_y_counts
            fold_eval = np.mean(std_per_class)
            samples_in_fold = np.sum(y_counts_per_fold[i])
            is_current_fold_better = fold_eval < min_eval or (
                np.isclose(fold_eval, min_eval)
                and samples_in_fold < min_samples_in_fold
            )
            if is_current_fold_better:
                min_eval = fold_eval
                min_samples_in_fold = samples_in_fold
                best_fold = i
        return best_fold


class TimeSeriesSplit(_BaseKFold):
    # [CLASS DOC]
    # Expanding-window splitter for time-ordered data: each test fold follows its
    # training window, with optional max_train_size, fixed test_size and gap.
    """Time Series cross-validator.

    Provides train/test indices to split time-ordered data, where other
    cross-validation methods are inappropriate, as they would lead to training
    on future data and evaluating on past data.
    To ensure comparable metrics across folds, samples must be equally spaced.
    Once this condition is met, each test set covers the same time duration,
    while the train set size accumulates data from previous splits.

    This cross-validation object is a variation of :class:`KFold`.
    In the k-th split, it returns the first k folds as the train set and the
    (k+1)-th fold as the test set.

    Note that, unlike standard cross-validation methods, successive
    training sets are supersets of those that come before them.

    Read more in the :ref:`User Guide <time_series_split>`.

    For visualisation of cross-validation behaviour and
    comparison between common scikit-learn split methods
    refer to :ref:`sphx_glr_auto_examples_model_selection_plot_cv_indices.py`

    .. versionadded:: 0.18

    Parameters
    ----------
    n_splits : int, default=5
        Number of splits. Must be at least 2.

        .. versionchanged:: 0.22
            ``n_splits`` default value changed from 3 to 5.

    max_train_size : int, default=None
        Maximum size for a single training set.

    test_size : int, default=None
        Used to limit the size of the test set. Defaults to
        ``n_samples // (n_splits + 1)``, which is the maximum allowed value
        with ``gap=0``.

        .. versionadded:: 0.24

    gap : int, default=0
        Number of samples to exclude from the end of each train set before
        the test set.

        .. versionadded:: 0.24

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.model_selection import TimeSeriesSplit
    >>> X = np.array([[1, 2], [3, 4], [1, 2], [3, 4], [1, 2], [3, 4]])
    >>> y = np.array([1, 2, 3, 4, 5, 6])
    >>> tscv = TimeSeriesSplit()
    >>> print(tscv)
    TimeSeriesSplit(gap=0, max_train_size=None, n_splits=5, test_size=None)
    >>> for i, (train_index, test_index) in enumerate(tscv.split(X)):
    ...     print(f"Fold {i}:")
    ...     print(f"  Train: index={train_index}")
    ...     print(f"  Test:  index={test_index}")
    Fold 0:
      Train: index=[0]
      Test:  index=[1]
    Fold 1:
      Train: index=[0 1]
      Test:  index=[2]
    Fold 2:
      Train: index=[0 1 2]
      Test:  index=[3]
    Fold 3:
      Train: index=[0 1 2 3]
      Test:  index=[4]
    Fold 4:
      Train: index=[0 1 2 3 4]
      Test:  index=[5]
    >>> # Fix test_size to 2 with 12 samples
    >>> X = np.random.randn(12, 2)
    >>> y = np.random.randint(0, 2, 12)
    >>> tscv = TimeSeriesSplit(n_splits=3, test_size=2)
    >>> for i, (train_index, test_index) in enumerate(tscv.split(X)):
    ...     print(f"Fold {i}:")
    ...     print(f"  Train: index={train_index}")
    ...     print(f"  Test:  index={test_index}")
    Fold 0:
      Train: index=[0 1 2 3 4 5]
      Test:  index=[6 7]
    Fold 1:
      Train: index=[0 1 2 3 4 5 6 7]
      Test:  index=[8 9]
    Fold 2:
      Train: index=[0 1 2 3 4 5 6 7 8 9]
      Test:  index=[10 11]
    >>> # Add in a 2 period gap
    >>> tscv = TimeSeriesSplit(n_splits=3, test_size=2, gap=2)
    >>> for i, (train_index, test_index) in enumerate(tscv.split(X)):
    ...     print(f"Fold {i}:")
    ...     print(f"  Train: index={train_index}")
    ...     print(f"  Test:  index={test_index}")
    Fold 0:
      Train: index=[0 1 2 3]
      Test:  index=[6 7]
    Fold 1:
      Train: index=[0 1 2 3 4 5]
      Test:  index=[8 9]
    Fold 2:
      Train: index=[0 1 2 3 4 5 6 7]
      Test:  index=[10 11]

    For a more extended example see
    :ref:`sphx_glr_auto_examples_applications_plot_cyclical_feature_engineering.py`.

    Notes
    -----
    The training set has size ``i * n_samples // (n_splits + 1)
    + n_samples % (n_splits + 1)`` in the ``i`` th split,
    with a test set of size ``n_samples//(n_splits + 1)`` by default,
    where ``n_samples`` is the number of samples. Note that this
    formula is only valid when ``test_size`` and ``max_train_size`` are
    left to their default values.
    """

    def __init__(self, n_splits=5, *, max_train_size=None, test_size=None, gap=0):
        # [FUNCTION DOC]
        # Calls _BaseKFold.__init__ with shuffle=False and stores max_train_size,
        # test_size and gap without validation.
        super().__init__(n_splits, shuffle=False, random_state=None)
        self.max_train_size = max_train_size
        self.test_size = test_size
        self.gap = gap

    def split(self, X, y=None, groups=None):
        # [FUNCTION DOC]
        # Warns if groups is given and returns _split(X).
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        y : array-like of shape (n_samples,), default=None
            Always ignored, exists for API compatibility.

        groups : array-like of shape (n_samples,), default=None
            Always ignored, exists for API compatibility.

        Yields
        ------
        train : ndarray
            The training set indices for that split.

        test : ndarray
            The testing set indices for that split.
        """
        if groups is not None:
            warnings.warn(
                f"The groups parameter is ignored by {self.__class__.__name__}",
                UserWarning,
            )
        return self._split(X)

    def _split(self, X):
        # [FUNCTION DOC]
        # HELPER: Computes test_size (default n_samples // (n_splits + 1)); test windows
        # start at n_samples - n_splits * test_size and advance by test_size. For each,
        # the train window ends gap samples before the test start and, if max_train_size
        # is set and smaller, keeps only its last max_train_size samples.
        # Outputs: generator of (train, test) index arrays.
        # Exceptions: ValueError if n_splits + 1 > n_samples, or if n_samples - gap -
        # test_size * n_splits <= 0.
        # KNOWN ISSUE: max_train_size=0 is treated like None because of a truthiness
        # test.
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        Yields
        ------
        train : ndarray
            The training set indices for that split.

        test : ndarray
            The testing set indices for that split.
        """
        (X,) = indexable(X)
        n_samples = _num_samples(X)
        n_splits = self.n_splits
        n_folds = n_splits + 1
        gap = self.gap
        test_size = (
            self.test_size if self.test_size is not None else n_samples // n_folds
        )

        # Make sure we have enough samples for the given split parameters
        if n_folds > n_samples:
            raise ValueError(
                f"Cannot have number of folds={n_folds} greater"
                f" than the number of samples={n_samples}."
            )
        if n_samples - gap - (test_size * n_splits) <= 0:
            raise ValueError(
                f"Too many splits={n_splits} for number of samples"
                f"={n_samples} with test_size={test_size} and gap={gap}."
            )

        indices = np.arange(n_samples)
        test_starts = range(n_samples - n_splits * test_size, n_samples, test_size)

        for test_start in test_starts:
            train_end = test_start - gap
            if self.max_train_size and self.max_train_size < train_end:
                yield (
                    indices[train_end - self.max_train_size : train_end],
                    indices[test_start : test_start + test_size],
                )
            else:
                yield (
                    indices[:train_end],
                    indices[test_start : test_start + test_size],
                )


class LeaveOneGroupOut(GroupsConsumerMixin, BaseCrossValidator):
    # [CLASS DOC]
    # Each unique group in turn forms the test set.
    """Leave One Group Out cross-validator.

    Provides train/test indices to split data such that each training set is
    comprised of all samples except ones belonging to one specific group.
    Arbitrary domain-specific group information is provided as an array of integers
    that encodes the group of each sample.

    For instance the groups could be the year of collection of the samples
    and thus allow for cross-validation against time-based splits.

    Read more in the :ref:`User Guide <leave_one_group_out>`.

    Notes
    -----
    Splits are ordered according to the index of the group left out. The first
    split has testing set consisting of the group whose index in `groups` is
    lowest, and so on.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.model_selection import LeaveOneGroupOut
    >>> X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    >>> y = np.array([1, 2, 1, 2])
    >>> groups = np.array([1, 1, 2, 2])
    >>> logo = LeaveOneGroupOut()
    >>> logo.get_n_splits(groups=groups)
    2
    >>> print(logo)
    LeaveOneGroupOut()
    >>> for i, (train_index, test_index) in enumerate(logo.split(X, y, groups)):
    ...     print(f"Fold {i}:")
    ...     print(f"  Train: index={train_index}, group={groups[train_index]}")
    ...     print(f"  Test:  index={test_index}, group={groups[test_index]}")
    Fold 0:
      Train: index=[2 3], group=[2 2]
      Test:  index=[0 1], group=[1 1]
    Fold 1:
      Train: index=[0 1], group=[1 1]
      Test:  index=[2 3], group=[2 2]

    See also
    --------
    GroupKFold: K-fold iterator variant with non-overlapping groups.
    """

    def _iter_test_masks(self, X, y, groups):
        # [FUNCTION DOC]
        # Yields, for each unique group in sorted order, the mask of samples in that
        # group.
        # Exceptions: ValueError if groups is None or has fewer than 2 unique values.
        # Side effects: none (groups is copied).
        if groups is None:
            raise ValueError("The 'groups' parameter should not be None.")
        # We make a copy of groups to avoid side-effects during iteration
        groups = check_array(
            groups, input_name="groups", copy=True, ensure_2d=False, dtype=None
        )
        unique_groups = np.unique(groups)
        if len(unique_groups) <= 1:
            raise ValueError(
                "The groups parameter contains fewer than 2 unique groups "
                "(%s). LeaveOneGroupOut expects at least 2." % unique_groups
            )
        for i in unique_groups:
            yield groups == i

    def get_n_splits(self, X=None, y=None, groups=None):
        # [FUNCTION DOC]
        # Returns the number of unique groups. X and y are ignored. ValueError if groups
        # is None.
        """Returns the number of splitting iterations in the cross-validator.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features), default=None
            Always ignored, exists for API compatibility.

        y : array-like of shape (n_samples,), default=None
            Always ignored, exists for API compatibility.

        groups : array-like of shape (n_samples,), default=None
            Group labels for the samples used while splitting the dataset into
            train/test set. This 'groups' parameter must always be specified to
            calculate the number of splits, though the other parameters can be
            omitted.

        Returns
        -------
        n_splits : int
            Returns the number of splitting iterations in the cross-validator.
        """
        if groups is None:
            raise ValueError("The 'groups' parameter should not be None.")
        groups = check_array(groups, input_name="groups", ensure_2d=False, dtype=None)
        return len(np.unique(groups))

    def split(self, X, y=None, groups=None):
        # [FUNCTION DOC]
        # Delegates to BaseCrossValidator.split; documents that groups is required.
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        y : array-like of shape (n_samples,), default=None
            The target variable for supervised learning problems.

        groups : array-like of shape (n_samples,)
            Group labels for the samples used while splitting the dataset into
            train/test set.

        Yields
        ------
        train : ndarray
            The training set indices for that split.

        test : ndarray
            The testing set indices for that split.
        """
        return super().split(X, y, groups)


class LeavePGroupsOut(GroupsConsumerMixin, BaseCrossValidator):
    # [CLASS DOC]
    # Every combination of n_groups unique groups forms a test set.
    """Leave P Group(s) Out cross-validator.

    Provides train/test indices to split data according to a third-party
    provided group. This group information can be used to encode arbitrary
    domain-specific groupings of the samples as integers.

    For instance the groups could be the year of collection of the samples
    and thus allow for cross-validation against time-based splits.

    The difference between LeavePGroupsOut and LeaveOneGroupOut is that
    the former builds the test sets with all the samples assigned to
    ``p`` different values of the groups while the latter uses samples
    all assigned the same groups.

    Read more in the :ref:`User Guide <leave_p_groups_out>`.

    Parameters
    ----------
    n_groups : int
        Number of groups (``p``) to leave out in the test split.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.model_selection import LeavePGroupsOut
    >>> X = np.array([[1, 2], [3, 4], [5, 6]])
    >>> y = np.array([1, 2, 1])
    >>> groups = np.array([1, 2, 3])
    >>> lpgo = LeavePGroupsOut(n_groups=2)
    >>> lpgo.get_n_splits(groups=groups)
    3
    >>> print(lpgo)
    LeavePGroupsOut(n_groups=2)
    >>> for i, (train_index, test_index) in enumerate(lpgo.split(X, y, groups)):
    ...     print(f"Fold {i}:")
    ...     print(f"  Train: index={train_index}, group={groups[train_index]}")
    ...     print(f"  Test:  index={test_index}, group={groups[test_index]}")
    Fold 0:
      Train: index=[2], group=[3]
      Test:  index=[0 1], group=[1 2]
    Fold 1:
      Train: index=[1], group=[2]
      Test:  index=[0 2], group=[1 3]
    Fold 2:
      Train: index=[0], group=[1]
      Test:  index=[1 2], group=[2 3]

    See Also
    --------
    GroupKFold : K-fold iterator variant with non-overlapping groups.
    """

    def __init__(self, n_groups):
        # [FUNCTION DOC]
        # Stores n_groups (int) without validation.
        self.n_groups = n_groups

    def _iter_test_masks(self, X, y, groups):
        # [FUNCTION DOC]
        # For each combination of n_groups indices into the sorted unique groups, yields
        # a mask selecting all samples of those groups.
        # Exceptions: ValueError if groups is None or there are not more than n_groups
        # unique groups.
        if groups is None:
            raise ValueError("The 'groups' parameter should not be None.")
        groups = check_array(
            groups, input_name="groups", copy=True, ensure_2d=False, dtype=None
        )
        unique_groups = np.unique(groups)
        if self.n_groups >= len(unique_groups):
            raise ValueError(
                "The groups parameter contains fewer than (or equal to) "
                "n_groups (%d) numbers of unique groups (%s). LeavePGroupsOut "
                "expects that at least n_groups + 1 (%d) unique groups be "
                "present" % (self.n_groups, unique_groups, self.n_groups + 1)
            )
        combi = combinations(range(len(unique_groups)), self.n_groups)
        for indices in combi:
            test_index = np.zeros(_num_samples(X), dtype=bool)
            for l in unique_groups[np.array(indices)]:
                test_index[groups == l] = True
            yield test_index

    def get_n_splits(self, X=None, y=None, groups=None):
        # [FUNCTION DOC]
        # Returns C(n_unique_groups, n_groups). ValueError if groups is None.
        """Returns the number of splitting iterations in the cross-validator.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features), default=None
            Always ignored, exists for API compatibility.

        y : array-like of shape (n_samples,), default=None
            Always ignored, exists for API compatibility.

        groups : array-like of shape (n_samples,), default=None
            Group labels for the samples used while splitting the dataset into
            train/test set. This 'groups' parameter must always be specified to
            calculate the number of splits, though the other parameters can be
            omitted.

        Returns
        -------
        n_splits : int
            Returns the number of splitting iterations in the cross-validator.
        """
        if groups is None:
            raise ValueError("The 'groups' parameter should not be None.")
        groups = check_array(groups, input_name="groups", ensure_2d=False, dtype=None)
        return int(comb(len(np.unique(groups)), self.n_groups, exact=True))

    def split(self, X, y=None, groups=None):
        # [FUNCTION DOC]
        # Delegates to BaseCrossValidator.split; documents that groups is required.
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        y : array-like of shape (n_samples,), default=None
            The target variable for supervised learning problems.

        groups : array-like of shape (n_samples,)
            Group labels for the samples used while splitting the dataset into
            train/test set.

        Yields
        ------
        train : ndarray
            The training set indices for that split.

        test : ndarray
            The testing set indices for that split.
        """
        return super().split(X, y, groups)


class _RepeatedSplits(_MetadataRequester, metaclass=ABCMeta):
    # [CLASS DOC]
    # HELPER: Repeats a shuffling splitter class n_repeats times, building a new
    # instance with shuffle=True and the shared RNG for each repetition.
    """Repeated splits for an arbitrary randomized CV splitter.

    Repeats splits for cross-validators n times with different randomization
    in each repetition.

    Parameters
    ----------
    cv : callable
        Cross-validator class.

    n_repeats : int, default=10
        Number of times cross-validator needs to be repeated.

    random_state : int, RandomState instance or None, default=None
        Passes `random_state` to the arbitrary repeating cross validator.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    **cvargs : additional params
        Constructor parameters for cv. Must not contain random_state
        and shuffle.
    """

    # This indicates that by default CV splitters don't have a "groups" kwarg,
    # unless indicated by inheriting from ``GroupsConsumerMixin``.
    # This also prevents ``set_split_request`` to be generated for splitters
    # which don't support ``groups``.
    __metadata_request__split = {"groups": metadata_routing.UNUSED}

    def __init__(self, cv, *, n_repeats=10, random_state=None, **cvargs):
        # [FUNCTION DOC]
        # Inputs: cv - splitter class; n_repeats - positive int; random_state; cvargs -
        # extra constructor kwargs.
        # Exceptions: ValueError if n_repeats is not Integral or <= 0, or cvargs
        # contains random_state or shuffle.
        # Side effects: stores cv, n_repeats, random_state, cvargs.
        if not isinstance(n_repeats, numbers.Integral):
            raise ValueError("Number of repetitions must be of Integral type.")

        if n_repeats <= 0:
            raise ValueError("Number of repetitions must be greater than 0.")

        if any(key in cvargs for key in ("random_state", "shuffle")):
            raise ValueError("cvargs must not contain random_state or shuffle.")

        self.cv = cv
        self.n_repeats = n_repeats
        self.random_state = random_state
        self.cvargs = cvargs

    def split(self, X, y=None, groups=None):
        # [FUNCTION DOC]
        # Creates one RNG from random_state and, for each repetition, instantiates cv
        # with that RNG and shuffle=True and yields all its splits.
        # Outputs: generator of (train, test).
        """Generates indices to split data into training and test set.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        y : array-like of shape (n_samples,)
            The target variable for supervised learning problems.

        groups : array-like of shape (n_samples,), default=None
            Group labels for the samples used while splitting the dataset into
            train/test set.

        Yields
        ------
        train : ndarray
            The training set indices for that split.

        test : ndarray
            The testing set indices for that split.
        """
        n_repeats = self.n_repeats
        rng = check_random_state(self.random_state)

        for idx in range(n_repeats):
            cv = self.cv(random_state=rng, shuffle=True, **self.cvargs)
            for train_index, test_index in cv.split(X, y, groups):
                yield train_index, test_index

    def get_n_splits(self, X=None, y=None, groups=None):
        # [FUNCTION DOC]
        # Instantiates cv once and returns its get_n_splits(X, y, groups) times
        # n_repeats.
        """Returns the number of splitting iterations as set with the `n_splits` param
        when instantiating the cross-validator.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features), default=None
            Always ignored, exists for API compatibility.

        y : array-like of shape (n_samples,), default=None
            Always ignored, exists for API compatibility.

        groups : array-like of shape (n_samples,), default=None
            Always ignored, exists for API compatibility.

        Returns
        -------
        n_splits : int
            Returns the number of splitting iterations in the cross-validator.
        """
        rng = check_random_state(self.random_state)
        cv = self.cv(random_state=rng, shuffle=True, **self.cvargs)
        return cv.get_n_splits(X, y, groups) * self.n_repeats

    def __repr__(self):
        # [FUNCTION DOC]
        # Returns _build_repr(self), which also looks up parameters in cvargs.
        return _build_repr(self)


class RepeatedKFold(_UnsupportedGroupCVMixin, _RepeatedSplits):
    # [CLASS DOC]
    # Repeats KFold n_repeats times with different shuffles.
    """Repeated K-Fold cross validator.

    Repeats K-Fold `n_repeats` times with different randomization in each repetition.

    Read more in the :ref:`User Guide <repeated_k_fold>`.

    Parameters
    ----------
    n_splits : int, default=5
        Number of folds. Must be at least 2.

    n_repeats : int, default=10
        Number of times cross-validator needs to be repeated.

    random_state : int, RandomState instance or None, default=None
        Controls the randomness of each repeated cross-validation instance.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.model_selection import RepeatedKFold
    >>> X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
    >>> y = np.array([0, 0, 1, 1])
    >>> rkf = RepeatedKFold(n_splits=2, n_repeats=2, random_state=2652124)
    >>> rkf.get_n_splits()
    4
    >>> print(rkf)
    RepeatedKFold(n_repeats=2, n_splits=2, random_state=2652124)
    >>> for i, (train_index, test_index) in enumerate(rkf.split(X)):
    ...     print(f"Fold {i}:")
    ...     print(f"  Train: index={train_index}")
    ...     print(f"  Test:  index={test_index}")
    ...
    Fold 0:
      Train: index=[0 1]
      Test:  index=[2 3]
    Fold 1:
      Train: index=[2 3]
      Test:  index=[0 1]
    Fold 2:
      Train: index=[1 2]
      Test:  index=[0 3]
    Fold 3:
      Train: index=[0 3]
      Test:  index=[1 2]

    Notes
    -----
    Randomized CV splitters may return different results for each call of
    split. You can make the results identical by setting `random_state`
    to an integer.

    See Also
    --------
    RepeatedStratifiedKFold : Repeats Stratified K-Fold n times.
    """

    def __init__(self, *, n_splits=5, n_repeats=10, random_state=None):
        # [FUNCTION DOC]
        # Configures _RepeatedSplits with KFold and n_splits as cvargs.
        super().__init__(
            KFold, n_repeats=n_repeats, random_state=random_state, n_splits=n_splits
        )


class RepeatedStratifiedKFold(_UnsupportedGroupCVMixin, _RepeatedSplits):
    # [CLASS DOC]
    # Repeats StratifiedKFold n_repeats times with different shuffles.
    """Repeated class-wise stratified K-Fold cross validator.

    Repeats Stratified K-Fold n times with different randomization in each
    repetition.

    Read more in the :ref:`User Guide <repeated_k_fold>`.

    .. note::

        Stratification on the class label solves an engineering problem rather
        than a statistical one. See :ref:`stratification` for more details.

    Parameters
    ----------
    n_splits : int, default=5
        Number of folds. Must be at least 2.

    n_repeats : int, default=10
        Number of times cross-validator needs to be repeated.

    random_state : int, RandomState instance or None, default=None
        Controls the generation of the random states for each repetition.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.model_selection import RepeatedStratifiedKFold
    >>> X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
    >>> y = np.array([0, 0, 1, 1])
    >>> rskf = RepeatedStratifiedKFold(n_splits=2, n_repeats=2,
    ...     random_state=36851234)
    >>> rskf.get_n_splits()
    4
    >>> print(rskf)
    RepeatedStratifiedKFold(n_repeats=2, n_splits=2, random_state=36851234)
    >>> for i, (train_index, test_index) in enumerate(rskf.split(X, y)):
    ...     print(f"Fold {i}:")
    ...     print(f"  Train: index={train_index}")
    ...     print(f"  Test:  index={test_index}")
    ...
    Fold 0:
      Train: index=[1 2]
      Test:  index=[0 3]
    Fold 1:
      Train: index=[0 3]
      Test:  index=[1 2]
    Fold 2:
      Train: index=[1 3]
      Test:  index=[0 2]
    Fold 3:
      Train: index=[0 2]
      Test:  index=[1 3]

    Notes
    -----
    Randomized CV splitters may return different results for each call of
    split. You can make the results identical by setting `random_state`
    to an integer.

    See Also
    --------
    RepeatedKFold : Repeats K-Fold n times.
    """

    def __init__(self, *, n_splits=5, n_repeats=10, random_state=None):
        # [FUNCTION DOC]
        # Configures _RepeatedSplits with StratifiedKFold and n_splits as cvargs.
        super().__init__(
            StratifiedKFold,
            n_repeats=n_repeats,
            random_state=random_state,
            n_splits=n_splits,
        )

    def split(self, X, y, groups=None):
        # [FUNCTION DOC]
        # Validates y with check_array and delegates to the mixin/_RepeatedSplits split.
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.

            Note that providing ``y`` is sufficient to generate the splits and
            hence ``np.zeros(n_samples)`` may be used as a placeholder for
            ``X`` instead of actual training data.

        y : array-like of shape (n_samples,)
            The target variable for supervised learning problems.
            Stratification is done based on the y labels.

        groups : array-like of shape (n_samples,), default=None
            Always ignored, exists for API compatibility.

        Yields
        ------
        train : ndarray
            The training set indices for that split.

        test : ndarray
            The testing set indices for that split.

        Notes
        -----
        Randomized CV splitters may return different results for each call of
        split. You can make the results identical by setting `random_state`
        to an integer.
        """
        y = check_array(y, input_name="y", ensure_2d=False, dtype=None)
        return super().split(X, y, groups=groups)


class BaseShuffleSplit(_MetadataRequester, metaclass=ABCMeta):
    # [CLASS DOC]
    # Base for random-permutation splitters producing n_splits independent random
    # train/test partitions of sizes derived from test_size/train_size.
    """Base class for *ShuffleSplit.

    Parameters
    ----------
    n_splits : int, default=10
        Number of re-shuffling & splitting iterations.

    test_size : float or int, default=None
        If float, should be between 0.0 and 1.0 and represent the proportion
        of the dataset to include in the test split. If int, represents the
        absolute number of test samples. If None, the value is set to the
        complement of the train size. If ``train_size`` is also None, it will
        be set to 0.1.

    train_size : float or int, default=None
        If float, should be between 0.0 and 1.0 and represent the
        proportion of the dataset to include in the train split. If
        int, represents the absolute number of train samples. If None,
        the value is automatically set to the complement of the test size.

    random_state : int, RandomState instance or None, default=None
        Controls the randomness of the training and testing indices produced.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.
    """

    # This indicates that by default CV splitters don't have a "groups" kwarg,
    # unless indicated by inheriting from ``GroupsConsumerMixin``.
    # This also prevents ``set_split_request`` to be generated for splitters
    # which don't support ``groups``.
    __metadata_request__split = {"groups": metadata_routing.UNUSED}

    def __init__(
        self, n_splits=10, *, test_size=None, train_size=None, random_state=None
    ):
        # [FUNCTION DOC]
        # Stores parameters without validation and sets _default_test_size to 0.1.
        self.n_splits = n_splits
        self.test_size = test_size
        self.train_size = train_size
        self.random_state = random_state
        self._default_test_size = 0.1

    def split(self, X, y=None, groups=None):
        # [FUNCTION DOC]
        # Makes inputs indexable and yields from _iter_indices.
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        y : array-like of shape (n_samples,)
            The target variable for supervised learning problems.

        groups : array-like of shape (n_samples,), default=None
            Group labels for the samples used while splitting the dataset into
            train/test set.

        Yields
        ------
        train : ndarray
            The training set indices for that split.

        test : ndarray
            The testing set indices for that split.

        Notes
        -----
        Randomized CV splitters may return different results for each call of
        split. You can make the results identical by setting `random_state`
        to an integer.
        """
        X, y, groups = indexable(X, y, groups)
        for train, test in self._iter_indices(X, y, groups):
            yield train, test

    def _iter_indices(self, X, y=None, groups=None):
        # [FUNCTION DOC]
        # Resolves (n_train, n_test) via _validate_shuffle_split, then n_splits times
        # draws a random permutation and takes the first n_test as test and the next
        # n_train as train.
        # Exceptions: ValueError from _validate_shuffle_split. Side effects: randomness.
        """Generate (train, test) indices"""
        n_samples = _num_samples(X)
        n_train, n_test = _validate_shuffle_split(
            n_samples,
            self.test_size,
            self.train_size,
            default_test_size=self._default_test_size,
        )

        rng = check_random_state(self.random_state)
        for i in range(self.n_splits):
            # random partition
            permutation = rng.permutation(n_samples)
            ind_test = permutation[:n_test]
            ind_train = permutation[n_test : (n_test + n_train)]
            yield ind_train, ind_test

    def get_n_splits(self, X=None, y=None, groups=None):
        # [FUNCTION DOC]
        # Returns n_splits; X, y and groups: Always ignored, exists for API
        # compatibility.
        """Returns the number of splitting iterations as set with the `n_splits` param
        when instantiating the cross-validator.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features), default=None
            Always ignored, exists for API compatibility.

        y : array-like of shape (n_samples,), default=None
            Always ignored, exists for API compatibility.

        groups : array-like of shape (n_samples,), default=None
            Always ignored, exists for API compatibility.

        Returns
        -------
        n_splits : int
            Returns the number of splitting iterations in the cross-validator.
        """
        return self.n_splits

    def __repr__(self):
        # [FUNCTION DOC]
        # Returns _build_repr(self).
        return _build_repr(self)


class ShuffleSplit(_UnsupportedGroupCVMixin, BaseShuffleSplit):
    # [CLASS DOC]
    # Plain random permutation splitter; warns if groups is passed.
    """Random permutation cross-validator.

    Yields indices to split data into training and test sets.

    Note: contrary to other cross-validation strategies, random splits
    do not guarantee that test sets across all folds will be mutually exclusive,
    and might include overlapping samples. However, this is still very likely for
    sizeable datasets.

    Read more in the :ref:`User Guide <ShuffleSplit>`.

    For visualisation of cross-validation behaviour and
    comparison between common scikit-learn split methods
    refer to :ref:`sphx_glr_auto_examples_model_selection_plot_cv_indices.py`

    Parameters
    ----------
    n_splits : int, default=10
        Number of re-shuffling & splitting iterations.

    test_size : float or int, default=None
        If float, should be between 0.0 and 1.0 and represent the proportion
        of the dataset to include in the test split. If int, represents the
        absolute number of test samples. If None, the value is set to the
        complement of the train size. If ``train_size`` is also None, it will
        be set to 0.1.

    train_size : float or int, default=None
        If float, should be between 0.0 and 1.0 and represent the
        proportion of the dataset to include in the train split. If
        int, represents the absolute number of train samples. If None,
        the value is automatically set to the complement of the test size.

    random_state : int, RandomState instance or None, default=None
        Controls the randomness of the training and testing indices produced.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.model_selection import ShuffleSplit
    >>> X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [3, 4], [5, 6]])
    >>> y = np.array([1, 2, 1, 2, 1, 2])
    >>> rs = ShuffleSplit(n_splits=5, test_size=.25, random_state=0)
    >>> rs.get_n_splits()
    5
    >>> print(rs)
    ShuffleSplit(n_splits=5, random_state=0, test_size=0.25, train_size=None)
    >>> for i, (train_index, test_index) in enumerate(rs.split(X)):
    ...     print(f"Fold {i}:")
    ...     print(f"  Train: index={train_index}")
    ...     print(f"  Test:  index={test_index}")
    Fold 0:
      Train: index=[1 3 0 4]
      Test:  index=[5 2]
    Fold 1:
      Train: index=[4 0 2 5]
      Test:  index=[1 3]
    Fold 2:
      Train: index=[1 2 4 0]
      Test:  index=[3 5]
    Fold 3:
      Train: index=[3 4 1 0]
      Test:  index=[5 2]
    Fold 4:
      Train: index=[3 5 1 0]
      Test:  index=[2 4]
    >>> # Specify train and test size
    >>> rs = ShuffleSplit(n_splits=5, train_size=0.5, test_size=.25,
    ...                   random_state=0)
    >>> for i, (train_index, test_index) in enumerate(rs.split(X)):
    ...     print(f"Fold {i}:")
    ...     print(f"  Train: index={train_index}")
    ...     print(f"  Test:  index={test_index}")
    Fold 0:
      Train: index=[1 3 0]
      Test:  index=[5 2]
    Fold 1:
      Train: index=[4 0 2]
      Test:  index=[1 3]
    Fold 2:
      Train: index=[1 2 4]
      Test:  index=[3 5]
    Fold 3:
      Train: index=[3 4 1]
      Test:  index=[5 2]
    Fold 4:
      Train: index=[3 5 1]
      Test:  index=[2 4]
    """

    def __init__(
        self, n_splits=10, *, test_size=None, train_size=None, random_state=None
    ):
        # [FUNCTION DOC]
        # Forwards to BaseShuffleSplit and sets _default_test_size to 0.1.
        super().__init__(
            n_splits=n_splits,
            test_size=test_size,
            train_size=train_size,
            random_state=random_state,
        )
        self._default_test_size = 0.1


class GroupShuffleSplit(GroupsConsumerMixin, BaseShuffleSplit):
    # [CLASS DOC]
    # Shuffle split over unique groups: test_size/train_size count groups, and all
    # samples of selected groups go to the corresponding side.
    """Shuffle-Group(s)-Out cross-validation iterator.

    Provides randomized train/test indices to split data according to a
    third-party provided group. This group information can be used to encode
    arbitrary domain-specific groupings of the samples as integers.

    For instance the groups could be the year of collection of the samples
    and thus allow for cross-validation against time-based splits.

    The difference between :class:`LeavePGroupsOut` and ``GroupShuffleSplit`` is that
    the former generates splits using all subsets of size ``p`` unique groups,
    whereas ``GroupShuffleSplit`` generates a user-determined number of random
    test splits, each with a user-determined fraction of unique groups.

    For example, a less computationally intensive alternative to
    ``LeavePGroupsOut(p=10)`` would be
    ``GroupShuffleSplit(test_size=10, n_splits=100)``.

    Contrary to other cross-validation strategies, the random splits
    do not guarantee that test sets across all folds will be mutually exclusive,
    and might include overlapping samples. However, this is still very likely for
    sizeable datasets.

    Note: The parameters ``test_size`` and ``train_size`` refer to groups, and
    not to samples as in :class:`ShuffleSplit`.

    Read more in the :ref:`User Guide <group_shuffle_split>`.

    For visualisation of cross-validation behaviour and
    comparison between common scikit-learn split methods
    refer to :ref:`sphx_glr_auto_examples_model_selection_plot_cv_indices.py`

    Parameters
    ----------
    n_splits : int, default=5
        Number of re-shuffling & splitting iterations.

    test_size : float, int, default=None
        If float, should be between 0.0 and 1.0 and represent the proportion
        of groups to include in the test split (rounded up). If int,
        represents the absolute number of test groups. If None, the value is
        set to the complement of the train size. If ``train_size`` is also None,
        it will be set to 0.2.

    train_size : float or int, default=None
        If float, should be between 0.0 and 1.0 and represent the
        proportion of the groups to include in the train split. If
        int, represents the absolute number of train groups. If None,
        the value is automatically set to the complement of the test size.

    random_state : int, RandomState instance or None, default=None
        Controls the randomness of the training and testing indices produced.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.model_selection import GroupShuffleSplit
    >>> X = np.ones(shape=(8, 2))
    >>> y = np.ones(shape=(8, 1))
    >>> groups = np.array([1, 1, 2, 2, 2, 3, 3, 3])
    >>> print(groups.shape)
    (8,)
    >>> gss = GroupShuffleSplit(n_splits=2, train_size=.7, random_state=42)
    >>> gss.get_n_splits()
    2
    >>> print(gss)
    GroupShuffleSplit(n_splits=2, random_state=42, test_size=None, train_size=0.7)
    >>> for i, (train_index, test_index) in enumerate(gss.split(X, y, groups)):
    ...     print(f"Fold {i}:")
    ...     print(f"  Train: index={train_index}, group={groups[train_index]}")
    ...     print(f"  Test:  index={test_index}, group={groups[test_index]}")
    Fold 0:
      Train: index=[2 3 4 5 6 7], group=[2 2 2 3 3 3]
      Test:  index=[0 1], group=[1 1]
    Fold 1:
      Train: index=[0 1 5 6 7], group=[1 1 3 3 3]
      Test:  index=[2 3 4], group=[2 2 2]

    See Also
    --------
    ShuffleSplit : Shuffles samples to create independent test/train sets.

    LeavePGroupsOut : Train set leaves out all possible subsets of `p` groups.
    """

    def __init__(
        self, n_splits=5, *, test_size=None, train_size=None, random_state=None
    ):
        # [FUNCTION DOC]
        # Forwards to BaseShuffleSplit (default n_splits=5) and sets _default_test_size
        # to 0.2.
        super().__init__(
            n_splits=n_splits,
            test_size=test_size,
            train_size=train_size,
            random_state=random_state,
        )
        self._default_test_size = 0.2

    def _iter_indices(self, X, y, groups):
        # [FUNCTION DOC]
        # Encodes groups to unique ids, runs the base shuffle split on the array of
        # unique groups, and maps the selected group ids back to sample indices.
        # Exceptions: ValueError if groups is None.
        if groups is None:
            raise ValueError("The 'groups' parameter should not be None.")
        groups = check_array(groups, input_name="groups", ensure_2d=False, dtype=None)
        classes, group_indices = np.unique(groups, return_inverse=True)
        for group_train, group_test in super()._iter_indices(X=classes):
            # these are the indices of classes in the partition
            # invert them into data indices

            train = np.flatnonzero(np.isin(group_indices, group_train))
            test = np.flatnonzero(np.isin(group_indices, group_test))

            yield train, test

    def split(self, X, y=None, groups=None):
        # [FUNCTION DOC]
        # Delegates to BaseShuffleSplit.split; documents that groups is required.
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        y : array-like of shape (n_samples,), default=None
            The target variable for supervised learning problems.

        groups : array-like of shape (n_samples,)
            Group labels for the samples used while splitting the dataset into
            train/test set.

        Yields
        ------
        train : ndarray
            The training set indices for that split.

        test : ndarray
            The testing set indices for that split.

        Notes
        -----
        Randomized CV splitters may return different results for each call of
        split. You can make the results identical by setting `random_state`
        to an integer.
        """
        return super().split(X, y, groups)


class StratifiedShuffleSplit(BaseShuffleSplit):
    # [CLASS DOC]
    # Random splits that preserve class proportions of y in train and test sets.
    """Class-wise stratified ShuffleSplit cross-validator.

    Provides train/test indices to split data in train/test sets.

    This cross-validation object is a merge of :class:`StratifiedKFold` and
    :class:`ShuffleSplit`, which returns stratified randomized folds. The folds
    are made by preserving the percentage of samples for each class in `y` in a
    binary or multiclass classification setting.

    Note: like the :class:`ShuffleSplit` strategy, stratified random splits
    do not guarantee that test sets across all folds will be mutually exclusive,
    and might include overlapping samples. However, this is still very likely for
    sizeable datasets.

    Read more in the :ref:`User Guide <stratified_shuffle_split>`.

    For visualisation of cross-validation behaviour and
    comparison between common scikit-learn split methods
    refer to :ref:`sphx_glr_auto_examples_model_selection_plot_cv_indices.py`

    .. note::

        Stratification on the class label solves an engineering problem rather
        than a statistical one. See :ref:`stratification` for more details.

    Parameters
    ----------
    n_splits : int, default=10
        Number of re-shuffling & splitting iterations.

    test_size : float or int, default=None
        If float, should be between 0.0 and 1.0 and represent the proportion
        of the dataset to include in the test split. If int, represents the
        absolute number of test samples. If None, the value is set to the
        complement of the train size. If ``train_size`` is also None, it will
        be set to 0.1.

    train_size : float or int, default=None
        If float, should be between 0.0 and 1.0 and represent the
        proportion of the dataset to include in the train split. If
        int, represents the absolute number of train samples. If None,
        the value is automatically set to the complement of the test size.

    random_state : int, RandomState instance or None, default=None
        Controls the randomness of the training and testing indices produced.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.model_selection import StratifiedShuffleSplit
    >>> X = np.array([[1, 2], [3, 4], [1, 2], [3, 4], [1, 2], [3, 4]])
    >>> y = np.array([0, 0, 0, 1, 1, 1])
    >>> sss = StratifiedShuffleSplit(n_splits=5, test_size=0.5, random_state=0)
    >>> sss.get_n_splits()
    5
    >>> print(sss)
    StratifiedShuffleSplit(n_splits=5, random_state=0, ...)
    >>> for i, (train_index, test_index) in enumerate(sss.split(X, y)):
    ...     print(f"Fold {i}:")
    ...     print(f"  Train: index={train_index}")
    ...     print(f"  Test:  index={test_index}")
    Fold 0:
      Train: index=[5 2 3]
      Test:  index=[4 1 0]
    Fold 1:
      Train: index=[5 1 4]
      Test:  index=[0 2 3]
    Fold 2:
      Train: index=[5 0 2]
      Test:  index=[4 3 1]
    Fold 3:
      Train: index=[4 1 0]
      Test:  index=[2 3 5]
    Fold 4:
      Train: index=[0 5 1]
      Test:  index=[3 4 2]
    """

    def __init__(
        self, n_splits=10, *, test_size=None, train_size=None, random_state=None
    ):
        # [FUNCTION DOC]
        # Forwards to BaseShuffleSplit and sets _default_test_size to 0.1.
        super().__init__(
            n_splits=n_splits,
            test_size=test_size,
            train_size=train_size,
            random_state=random_state,
        )
        self._default_test_size = 0.1

    def _iter_indices(self, X, y, groups=None):
        # [FUNCTION DOC]
        # Draws stratified random splits.
        # Steps: resolve (n_train, n_test); move y to NumPy; for 2d y, turn each row
        # into a space-joined string label; find classes and counts; validate counts;
        # group sample indices by class (stable). For each split: allocate per-class
        # train counts with _approximate_mode(class_counts, n_train), then test counts
        # from the remaining counts; for each class permute its indices and take the
        # train then test slices; finally permute the concatenated train and test
        # arrays.
        # Outputs: generator of (train, test) int arrays.
        # Exceptions: ValueError if any class has fewer than 2 members, or n_train or
        # n_test is smaller than the number of classes.
        # Side effects: randomness via random_state.
        n_samples = _num_samples(X)
        y = check_array(y, input_name="y", ensure_2d=False, dtype=None)
        n_train, n_test = _validate_shuffle_split(
            n_samples,
            self.test_size,
            self.train_size,
            default_test_size=self._default_test_size,
        )

        # Convert to numpy as not all operations are supported by the Array API.
        # `y` is probably never a very large array, which means that converting it
        # should be cheap
        xp, _ = get_namespace(y)
        y = move_to(y, xp=np, device="cpu")

        if y.ndim == 2:
            # for multi-label y, map each distinct row to a string repr
            # using join because str(row) uses an ellipsis if len(row) > 1000
            y = np.array([" ".join(row.astype("str")) for row in y])

        classes, y_indices, class_counts = np.unique(
            y, return_inverse=True, return_counts=True
        )
        n_classes = classes.shape[0]

        if np.min(class_counts) < 2:
            too_few_classes = classes[class_counts < 2].tolist()
            raise ValueError(
                "The least populated classes in y have only 1"
                " member, which is too few. The minimum"
                " number of groups for any class cannot"
                " be less than 2. Classes with too few"
                " members are: %s" % (too_few_classes)
            )

        if n_train < n_classes:
            raise ValueError(
                "The train_size = %d should be greater or "
                "equal to the number of classes = %d" % (n_train, n_classes)
            )
        if n_test < n_classes:
            raise ValueError(
                "The test_size = %d should be greater or "
                "equal to the number of classes = %d" % (n_test, n_classes)
            )

        # Find the sorted list of instances for each class:
        # (np.unique above performs a sort, so code is O(n logn) already)
        class_indices = np.split(
            np.argsort(y_indices, kind="stable"), np.cumsum(class_counts)[:-1]
        )

        rng = check_random_state(self.random_state)

        for _ in range(self.n_splits):
            # if there are ties in the class-counts, we want
            # to make sure to break them anew in each iteration
            n_i = _approximate_mode(class_counts, n_train, rng)
            class_counts_remaining = class_counts - n_i
            t_i = _approximate_mode(class_counts_remaining, n_test, rng)

            train = []
            test = []

            for i in range(n_classes):
                permutation = rng.permutation(class_counts[i])
                perm_indices_class_i = class_indices[i].take(permutation, mode="clip")

                train.extend(perm_indices_class_i[: n_i[i]])
                test.extend(perm_indices_class_i[n_i[i] : n_i[i] + t_i[i]])

            train = rng.permutation(train)
            test = rng.permutation(test)

            yield train, test

    def split(self, X, y, groups=None):
        # [FUNCTION DOC]
        # Warns if groups is given, validates y via check_array and delegates to
        # BaseShuffleSplit.split.
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.

            Note that providing ``y`` is sufficient to generate the splits and
            hence ``np.zeros(n_samples)`` may be used as a placeholder for
            ``X`` instead of actual training data.

        y : array-like of shape (n_samples,) or (n_samples, n_labels)
            The target variable for supervised learning problems.
            Stratification is done based on the y labels.

        groups : array-like of shape (n_samples,), default=None
            Always ignored, exists for API compatibility.

        Yields
        ------
        train : ndarray
            The training set indices for that split.

        test : ndarray
            The testing set indices for that split.

        Notes
        -----
        Randomized CV splitters may return different results for each call of
        split. You can make the results identical by setting `random_state`
        to an integer.
        """
        if groups is not None:
            warnings.warn(
                f"The groups parameter is ignored by {self.__class__.__name__}",
                UserWarning,
            )
        y = check_array(y, input_name="y", ensure_2d=False, dtype=None)
        return super().split(X, y, groups)


def _validate_shuffle_split(n_samples, test_size, train_size, default_test_size=None):
    # [FUNCTION DOC]
    # HELPER: Turns test_size/train_size into absolute (n_train, n_test) ints.
    # Rules: if both are None, test_size = default_test_size. Ints must lie in [1,
    # n_samples - 1] and floats in (0, 1); other non-None types are rejected. Two floats
    # must sum to at most 1. Float test sizes are rounded up, float train sizes down; a
    # missing side is the complement of the other.
    # Inputs: n_samples - int; test_size, train_size - float, int or None;
    # default_test_size - float or None.
    # Outputs: (n_train, n_test) ints.
    # Exceptions: ValueError for out-of-range or invalid sizes, a sum exceeding
    # n_samples, or an empty train set.
    """
    Validation helper to check if the train/test sizes are meaningful w.r.t. the
    size of the data (n_samples).
    """
    if test_size is None and train_size is None:
        test_size = default_test_size

    test_size_type = np.asarray(test_size).dtype.kind
    train_size_type = np.asarray(train_size).dtype.kind

    if (test_size_type == "i" and (test_size >= n_samples or test_size <= 0)) or (
        test_size_type == "f" and (test_size <= 0 or test_size >= 1)
    ):
        raise ValueError(
            "test_size={0} should be either positive and smaller"
            " than the number of samples {1} or a float in the "
            "(0, 1) range".format(test_size, n_samples)
        )

    if (train_size_type == "i" and (train_size >= n_samples or train_size <= 0)) or (
        train_size_type == "f" and (train_size <= 0 or train_size >= 1)
    ):
        raise ValueError(
            "train_size={0} should be either positive and smaller"
            " than the number of samples {1} or a float in the "
            "(0, 1) range".format(train_size, n_samples)
        )

    if train_size is not None and train_size_type not in ("i", "f"):
        raise ValueError("Invalid value for train_size: {}".format(train_size))
    if test_size is not None and test_size_type not in ("i", "f"):
        raise ValueError("Invalid value for test_size: {}".format(test_size))

    if train_size_type == "f" and test_size_type == "f" and train_size + test_size > 1:
        raise ValueError(
            "The sum of test_size and train_size = {}, should be in the (0, 1)"
            " range. Reduce test_size and/or train_size.".format(train_size + test_size)
        )

    if test_size_type == "f":
        n_test = ceil(test_size * n_samples)
    elif test_size_type == "i":
        n_test = float(test_size)

    if train_size_type == "f":
        n_train = floor(train_size * n_samples)
    elif train_size_type == "i":
        n_train = float(train_size)

    if train_size is None:
        n_train = n_samples - n_test
    elif test_size is None:
        n_test = n_samples - n_train

    if n_train + n_test > n_samples:
        raise ValueError(
            "The sum of train_size and test_size = %d, "
            "should be smaller than the number of "
            "samples %d. Reduce test_size and/or "
            "train_size." % (n_train + n_test, n_samples)
        )

    n_train, n_test = int(n_train), int(n_test)

    if n_train == 0:
        raise ValueError(
            "With n_samples={}, test_size={} and train_size={}, the "
            "resulting train set will be empty. Adjust any of the "
            "aforementioned parameters.".format(n_samples, test_size, train_size)
        )

    return n_train, n_test


class PredefinedSplit(BaseCrossValidator):
    # [CLASS DOC]
    # Splitter driven by a user array test_fold assigning each sample to a test fold id,
    # with -1 meaning always in train.
    """Predefined split cross-validator.

    Provides train/test indices to split data into train/test sets using a
    predefined scheme specified by the user with the ``test_fold`` parameter.

    Read more in the :ref:`User Guide <predefined_split>`.

    .. versionadded:: 0.16

    Parameters
    ----------
    test_fold : array-like of shape (n_samples,)
        The entry ``test_fold[i]`` represents the index of the test set that
        sample ``i`` belongs to. It is possible to exclude sample ``i`` from
        any test set (i.e. include sample ``i`` in every training set) by
        setting ``test_fold[i]`` equal to -1.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.model_selection import PredefinedSplit
    >>> X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
    >>> y = np.array([0, 0, 1, 1])
    >>> test_fold = [0, 1, -1, 1]
    >>> ps = PredefinedSplit(test_fold)
    >>> ps.get_n_splits()
    2
    >>> print(ps)
    PredefinedSplit(test_fold=array([ 0,  1, -1,  1]))
    >>> for i, (train_index, test_index) in enumerate(ps.split()):
    ...     print(f"Fold {i}:")
    ...     print(f"  Train: index={train_index}")
    ...     print(f"  Test:  index={test_index}")
    Fold 0:
      Train: index=[1 2 3]
      Test:  index=[0]
    Fold 1:
      Train: index=[0 2]
      Test:  index=[1 3]
    """

    def __init__(self, test_fold):
        # [FUNCTION DOC]
        # Converts test_fold to a 1d int array and stores the sorted unique fold ids
        # other than -1 in unique_folds.
        # Exceptions: ValueError from column_or_1d / int conversion.
        # Side effects: sets test_fold and unique_folds.
        # KNOWN ISSUE: performs computation in __init__ and stores a transformed
        # parameter, contrary to the usual estimator conventions (clone/get_params
        # round-trips).
        self.test_fold = np.array(test_fold, dtype=int)
        self.test_fold = column_or_1d(self.test_fold)
        self.unique_folds = np.unique(self.test_fold)
        self.unique_folds = self.unique_folds[self.unique_folds != -1]

    def split(self, X=None, y=None, groups=None):
        # [FUNCTION DOC]
        # Warns if groups is given and returns _split(); X and y are ignored.
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features), default=None
            Always ignored, exists for API compatibility.

        y : array-like of shape (n_samples,), default=None
            Always ignored, exists for API compatibility.

        groups : array-like of shape (n_samples,), default=None
            Always ignored, exists for API compatibility.

        Yields
        ------
        train : ndarray
            The training set indices for that split.

        test : ndarray
            The testing set indices for that split.
        """
        if groups is not None:
            warnings.warn(
                f"The groups parameter is ignored by {self.__class__.__name__}",
                UserWarning,
            )
        return self._split()

    def _split(self):
        # [FUNCTION DOC]
        # HELPER: For each test mask yields (indices not in the mask, indices in the
        # mask).
        """Generate indices to split data into training and test set.

        Yields
        ------
        train : ndarray
            The training set indices for that split.

        test : ndarray
            The testing set indices for that split.
        """
        ind = np.arange(len(self.test_fold))
        for test_index in self._iter_test_masks():
            train_index = ind[np.logical_not(test_index)]
            test_index = ind[test_index]
            yield train_index, test_index

    def _iter_test_masks(self):
        # [FUNCTION DOC]
        # HELPER: Yields, for each fold id in unique_folds, the mask of samples with
        # that id.
        """Generates boolean masks corresponding to test sets."""
        for f in self.unique_folds:
            test_index = np.where(self.test_fold == f)[0]
            test_mask = np.zeros(len(self.test_fold), dtype=bool)
            test_mask[test_index] = True
            yield test_mask

    def get_n_splits(self, X=None, y=None, groups=None):
        # [FUNCTION DOC]
        # Returns the number of distinct fold ids other than -1.
        """Returns the number of splitting iterations in the cross-validator.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features), default=None
            Always ignored, exists for API compatibility.

        y : array-like of shape (n_samples,), default=None
            Always ignored, exists for API compatibility.

        groups : array-like of shape (n_samples,), default=None
            Always ignored, exists for API compatibility.

        Returns
        -------
        n_splits : int
            Returns the number of splitting iterations in the cross-validator.
        """
        return len(self.unique_folds)


class _CVIterableWrapper(BaseCrossValidator):
    # [CLASS DOC]
    # HELPER: Adapts an iterable of (train, test) pairs into a splitter.
    """Wrapper class for old style cv objects and iterables."""

    def __init__(self, cv):
        # [FUNCTION DOC]
        # Materialises cv into a list so it can be iterated repeatedly.
        self.cv = list(cv)

    def get_n_splits(self, X=None, y=None, groups=None):
        # [FUNCTION DOC]
        # Returns the number of stored splits.
        """Returns the number of splitting iterations in the cross-validator.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features), default=None
            Always ignored, exists for API compatibility.

        y : array-like of shape (n_samples,), default=None
            Always ignored, exists for API compatibility.

        groups : array-like of shape (n_samples,), default=None
            Always ignored, exists for API compatibility.

        Returns
        -------
        n_splits : int
            Returns the number of splitting iterations in the cross-validator.
        """
        return len(self.cv)

    def split(self, X=None, y=None, groups=None):
        # [FUNCTION DOC]
        # Yields the stored (train, test) pairs unchanged; arguments are ignored.
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features), default=None
            Always ignored, exists for API compatibility.

        y : array-like of shape (n_samples,), default=None
            Always ignored, exists for API compatibility.

        groups : array-like of shape (n_samples,), default=None
            Always ignored, exists for API compatibility.

        Yields
        ------
        train : ndarray
            The training set indices for that split.

        test : ndarray
            The testing set indices for that split.
        """
        for train, test in self.cv:
            yield train, test


def check_cv(cv=5, y=None, *, classifier=False, shuffle=False, random_state=None):
    # [FUNCTION DOC]
    # Normalises a cv argument into a splitter object.
    # Behavior: None becomes 5. An int yields StratifiedKFold when classifier is true
    # and y is binary or multiclass, else KFold (both with the given
    # shuffle/random_state). An object without split (or a str) must be a non-string
    # iterable and is wrapped in _CVIterableWrapper. Objects with split are returned
    # unchanged.
    # Outputs: splitter instance.
    # Exceptions: ValueError for strings or non-iterables without split; errors from the
    # K-fold constructors (e.g. cv < 2).
    """Input checker utility for building a cross-validator.

    Parameters
    ----------
    cv : int, cross-validation generator, iterable or None, default=5
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
        - None, to use the default 5-fold cross validation,
        - integer, to specify the number of folds,
        - :term:`CV splitter`,
        - an iterable that generates (train, test) splits as arrays of indices.

        For integer/None inputs, if classifier is True and ``y`` is either
        binary or multiclass, :class:`StratifiedKFold` is used. In all other
        cases, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.

        .. versionchanged:: 0.22
            ``cv`` default value changed from 3-fold to 5-fold.

    y : array-like, default=None
        The target variable for supervised learning problems.

    classifier : bool, default=False
        Whether the task is a classification task. When ``True`` and `cv` is an
        integer or ``None``, :class:`StratifiedKFold` is used if ``y`` is binary
        or multiclass; otherwise :class:`KFold` is used. Ignored if `cv` is a
        cross-validator instance or iterable.

    shuffle : bool, default=False
        Whether to shuffle the data before splitting into batches. Note that the samples
        within each split will not be shuffled. Only applies if `cv` is an int or
        `None`. If `cv` is a cross-validation generator or an iterable, `shuffle` is
        ignored.

    random_state : int, RandomState instance or None, default=None
        When `shuffle` is True and `cv` is an integer or `None`, `random_state` affects
        the ordering of the indices, which controls the randomness of each fold.
        Otherwise, this parameter has no effect.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    Returns
    -------
    checked_cv : a cross-validator instance.
        The return value is a cross-validator which generates the train/test
        splits via the ``split`` method.

    Examples
    --------
    >>> from sklearn.model_selection import check_cv
    >>> check_cv(cv=5, y=None, classifier=False)
    KFold(...)
    >>> check_cv(cv=5, y=[1, 1, 0, 0, 0, 0], classifier=True)
    StratifiedKFold(...)
    """
    cv = 5 if cv is None else cv
    if isinstance(cv, numbers.Integral):
        if (
            classifier
            and (y is not None)
            and (type_of_target(y, input_name="y") in ("binary", "multiclass"))
        ):
            return StratifiedKFold(cv, shuffle=shuffle, random_state=random_state)
        else:
            return KFold(cv, shuffle=shuffle, random_state=random_state)

    if not hasattr(cv, "split") or isinstance(cv, str):
        if not isinstance(cv, Iterable) or isinstance(cv, str):
            raise ValueError(
                "Expected `cv` as an integer, a cross-validation object "
                "(from sklearn.model_selection), or an iterable yielding (train, test) "
                f"splits as arrays of indices. Got {cv}."
            )
        return _CVIterableWrapper(cv)

    return cv  # New style cv objects are passed without any modification


@validate_params(
    {
        "test_size": [
            Interval(RealNotInt, 0, 1, closed="neither"),
            Interval(numbers.Integral, 1, None, closed="left"),
            None,
        ],
        "train_size": [
            Interval(RealNotInt, 0, 1, closed="neither"),
            Interval(numbers.Integral, 1, None, closed="left"),
            None,
        ],
        "random_state": ["random_state"],
        "shuffle": ["boolean"],
        "stratify": ["array-like", None],
    },
    prefer_skip_nested_validation=True,
)
def train_test_split(
    *arrays,
    test_size=None,
    train_size=None,
    random_state=None,
    shuffle=True,
    stratify=None,
):
    # [FUNCTION DOC]
    # Splits each array into train and test subsets with shared indices.
    # Steps: require at least one array; make them indexable; compute (n_train, n_test)
    # with default test fraction 0.25. Without shuffle, train is the first n_train rows
    # and test the next n_test (stratify not allowed). With shuffle, take the first
    # split of StratifiedShuffleSplit (if stratify) or ShuffleSplit with those absolute
    # sizes. Move the indices to the namespace/device of the first array and index every
    # array.
    # Outputs: list [a1_train, a1_test, a2_train, a2_test, ...].
    # Exceptions: ValueError for no arrays, invalid sizes, stratify with shuffle=False,
    # or stratification failures; parameter validation errors from validate_params.
    # Side effects: randomness via random_state.
    """Split arrays or matrices into random train and test subsets.

    Quick utility that wraps input validation,
    ``next(ShuffleSplit().split(X, y))``, and application to input data
    into a single call for splitting (and optionally subsampling) data into a
    one-liner.

    Read more in the :ref:`User Guide <cross_validation>`.

    Parameters
    ----------
    *arrays : sequence of array-like of shape (n_samples,) or \
            (n_samples, n_outputs)
        Indexable data-structures can be arrays, lists, dataframes, scipy
        sparse matrices or pandas dataframes with consistent first dimension.

    test_size : float or int, default=None
        If float, should be between 0.0 and 1.0 and represent the proportion
        of the dataset to include in the test split. If int, represents the
        absolute number of test samples. If None, the value is set to the
        complement of the train size. If ``train_size`` is also None, it will
        be set to 0.25.

    train_size : float or int, default=None
        If float, should be between 0.0 and 1.0 and represent the
        proportion of the dataset to include in the train split. If
        int, represents the absolute number of train samples. If None,
        the value is automatically set to the complement of the test size.

    random_state : int, RandomState instance or None, default=None
        Controls the shuffling applied to the data before applying the split.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    shuffle : bool, default=True
        Whether or not to shuffle the data before splitting. If shuffle=False
        then stratify must be None.

    stratify : array-like of shape (n_samples,), default=None
        If not None, data is split in a stratified fashion, using this as
        the class labels.
        Read more in the :ref:`User Guide <stratification>`.

    Returns
    -------
    splitting : list, length=2 * len(arrays)
        List containing train-test split of inputs.

        .. versionadded:: 0.16
            If the input is sparse, the output will be a
            ``scipy.sparse.csr_matrix``. Else, output type is the same as the
            input type.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.model_selection import train_test_split
    >>> X, y = np.arange(10).reshape((5, 2)), range(5)
    >>> X
    array([[0, 1],
           [2, 3],
           [4, 5],
           [6, 7],
           [8, 9]])
    >>> list(y)
    [0, 1, 2, 3, 4]

    >>> X_train, X_test, y_train, y_test = train_test_split(
    ...     X, y, test_size=0.33, random_state=42)
    ...
    >>> X_train
    array([[4, 5],
           [0, 1],
           [6, 7]])
    >>> y_train
    [2, 0, 3]
    >>> X_test
    array([[2, 3],
           [8, 9]])
    >>> y_test
    [1, 4]

    >>> train_test_split(y, shuffle=False)
    [[0, 1, 2], [3, 4]]

    >>> from sklearn import datasets
    >>> iris = datasets.load_iris(as_frame=True)
    >>> X, y = iris['data'], iris['target']
    >>> X.head()
        sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)
    0                5.1               3.5                1.4               0.2
    1                4.9               3.0                1.4               0.2
    2                4.7               3.2                1.3               0.2
    3                4.6               3.1                1.5               0.2
    4                5.0               3.6                1.4               0.2
    >>> y.head()
    0    0
    1    0
    2    0
    3    0
    4    0
    ...

    >>> X_train, X_test, y_train, y_test = train_test_split(
    ... X, y, test_size=0.33, random_state=42)
    ...
    >>> X_train.head()
        sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)
    96                 5.7               2.9                4.2               1.3
    105                7.6               3.0                6.6               2.1
    66                 5.6               3.0                4.5               1.5
    0                  5.1               3.5                1.4               0.2
    122                7.7               2.8                6.7               2.0
    >>> y_train.head()
    96     1
    105    2
    66     1
    0      0
    122    2
    ...
    >>> X_test.head()
        sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)
    73                 6.1               2.8                4.7               1.2
    18                 5.7               3.8                1.7               0.3
    118                7.7               2.6                6.9               2.3
    78                 6.0               2.9                4.5               1.5
    76                 6.8               2.8                4.8               1.4
    >>> y_test.head()
    73     1
    18     0
    118    2
    78     1
    76     1
    ...
    """
    n_arrays = len(arrays)
    if n_arrays == 0:
        raise ValueError("At least one array required as input")

    arrays = indexable(*arrays)

    n_samples = _num_samples(arrays[0])
    n_train, n_test = _validate_shuffle_split(
        n_samples, test_size, train_size, default_test_size=0.25
    )

    if shuffle is False:
        if stratify is not None:
            raise ValueError(
                "Stratified train/test split is not implemented for shuffle=False"
            )

        train = np.arange(n_train)
        test = np.arange(n_train, n_train + n_test)

    else:
        if stratify is not None:
            CVClass = StratifiedShuffleSplit
        else:
            CVClass = ShuffleSplit

        cv = CVClass(test_size=n_test, train_size=n_train, random_state=random_state)

        train, test = next(cv.split(X=arrays[0], y=stratify))

    xp, _, device = get_namespace_and_device(arrays[0])
    train, test = move_to(train, test, xp=xp, device=device)

    return list(
        chain.from_iterable(
            (_safe_indexing(a, train), _safe_indexing(a, test)) for a in arrays
        )
    )


# Tell nose that train_test_split is not a test.
# (Needed for external libraries that may use nose.)
# Use setattr to avoid pyrefly errors when monkeypatching.
setattr(train_test_split, "__test__", False)


def _pprint(params, offset=0, printer=repr):
    # [FUNCTION DOC]
    # HELPER: Formats params as 'key=value' pairs sorted by key, floats via str and
    # others via printer, truncating reprs longer than 500 chars, wrapping lines near 75
    # chars with an indentation derived from offset, and stripping trailing spaces.
    # Inputs: params - dict; offset - int; printer - callable.
    # Outputs: str.
    # Side effects: temporarily changes NumPy print options, restored afterwards.
    # KNOWN ISSUE: print options are not restored if formatting raises (no try/finally).
    """Pretty print the dictionary 'params'

    Parameters
    ----------
    params : dict
        The dictionary to pretty print

    offset : int, default=0
        The offset in characters to add at the begin of each line.

    printer : callable, default=repr
        The function to convert entries to strings, typically
        the builtin str or repr

    """
    # Do a multi-line justified repr:
    options = np.get_printoptions()
    np.set_printoptions(precision=5, threshold=64, edgeitems=2)
    params_list = list()
    this_line_length = offset
    line_sep = ",\n" + (1 + offset // 2) * " "
    for i, (k, v) in enumerate(sorted(params.items())):
        if isinstance(v, float):
            # use str for representing floating point numbers
            # this way we get consistent representation across
            # architectures and versions.
            this_repr = "%s=%s" % (k, str(v))
        else:
            # use repr of the rest
            this_repr = "%s=%s" % (k, printer(v))
        if len(this_repr) > 500:
            this_repr = this_repr[:300] + "..." + this_repr[-100:]
        if i > 0:
            if this_line_length + len(this_repr) >= 75 or "\n" in this_repr:
                params_list.append(line_sep)
                this_line_length = len(line_sep)
            else:
                params_list.append(", ")
                this_line_length += 2
        params_list.append(this_repr)
        this_line_length += len(this_repr)

    np.set_printoptions(**options)
    lines = "".join(params_list)
    # Strip trailing space to avoid nightmare in doctests
    lines = "\n".join(l.rstrip(" ") for l in lines.split("\n"))
    return lines


def _build_repr(self):
    # [FUNCTION DOC]
    # HELPER: Builds 'ClassName(params)' from the constructor signature's named
    # parameters (excluding self and **kwargs), reading each as an attribute or, when
    # None, from cvargs; parameters whose access emits a FutureWarning are skipped.
    # Inputs: self - splitter instance. Outputs: str.
    init = self.__class__.__init__
    # Ignore varargs, kw and default values and pop self
    init_signature = signature(init)
    # Consider the constructor parameters excluding 'self'
    if init is object.__init__:
        args = []
    else:
        args = sorted(
            [
                p.name
                for p in init_signature.parameters.values()
                if p.name != "self" and p.kind != p.VAR_KEYWORD
            ]
        )
    class_name = self.__class__.__name__
    params = dict()
    for key in args:
        with warnings.catch_warnings(record=True) as w:
            # We need deprecation warnings to always be on in order to
            # catch deprecated param values.
            # This is set in utils/__init__.py but it gets overwritten
            # when running under python3 somehow.
            warnings.simplefilter("always", FutureWarning)
            value = getattr(self, key, None)
            if value is None and hasattr(self, "cvargs"):
                value = self.cvargs.get(key, None)
        if len(w) and w[0].category is FutureWarning:
            # if the parameter is deprecated, don't show it
            continue

        params[key] = value

    return "%s(%s)" % (class_name, _pprint(params, offset=len(class_name)))


def _yields_constant_splits(cv):
    # [FUNCTION DOC]
    # HELPER: Returns True when repeated cv.split calls are expected to give identical
    # splits: shuffle is False, or random_state is an int. Missing shuffle defaults to
    # True and missing random_state to 0.
    # Inputs: cv - splitter. Outputs: bool.
    # KNOWN ISSUE: heuristic only; custom splitters with other randomness sources are
    # not inspected beyond these two attributes.
    # Return True if calling cv.split() always returns the same splits
    # We assume that if a cv doesn't have a shuffle parameter, it shuffles by
    # default (e.g. ShuffleSplit). If it actually doesn't shuffle (e.g.
    # LeaveOneOut), then it won't have a random_state parameter anyway, in
    # which case it will default to 0, leading to output=True
    shuffle = getattr(cv, "shuffle", True)
    random_state = getattr(cv, "random_state", 0)
    return isinstance(random_state, numbers.Integral) or not shuffle
