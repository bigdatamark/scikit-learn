# [MODULE DOC]
# Implements IterativeImputer, a multivariate (MICE-style) imputer that models each
# feature with missing values as a regression target of the other features and refines
# the imputations over several round-robin passes. Also defines the _ImputerTriplet
# record (feat_idx, neighbor_feat_idx, fitted estimator) stored for every imputation
# step, and the _assign_where helper. The class is experimental and exposed via
# sklearn.experimental.enable_iterative_imputer.
# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import warnings
from collections import namedtuple
from numbers import Integral, Real
from time import time

import numpy as np
from scipy import stats

from sklearn.base import _fit_context, clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute._base import SimpleImputer, _BaseImputer, _check_inputs_dtype
from sklearn.preprocessing import normalize
from sklearn.utils import _safe_indexing, check_array, check_random_state
from sklearn.utils._indexing import _safe_assign
from sklearn.utils._mask import _get_mask
from sklearn.utils._missing import is_scalar_nan
from sklearn.utils._param_validation import HasMethods, Interval, StrOptions
from sklearn.utils.metadata_routing import (
    MetadataRouter,
    MethodMapping,
    _raise_for_params,
    process_routing,
)
from sklearn.utils.validation import (
    FLOAT_DTYPES,
    _check_feature_names_in,
    _num_samples,
    check_is_fitted,
    validate_data,
)

_ImputerTriplet = namedtuple(
    "_ImputerTriplet", ["feat_idx", "neighbor_feat_idx", "estimator"]
)


def _assign_where(X1, X2, cond):
    # [FUNCTION DOC]
    # HELPER: Copies entries of X2 into X1 at the positions where the boolean mask cond
    # is True; uses DataFrame.mask in place for pandas objects, boolean indexing
    # otherwise.
    # Inputs: X1 - ndarray or DataFrame (n_samples, n_features), the destination; X2 -
    # ndarray of the same shape, the source; cond - boolean array of the same shape.
    # Outputs: None.
    # Side effects: mutates X1 in place.
    """Assign X2 to X1 where cond is True.

    Parameters
    ----------
    X1 : ndarray or dataframe of shape (n_samples, n_features)
        Data.

    X2 : ndarray of shape (n_samples, n_features)
        Data to be assigned.

    cond : ndarray of shape (n_samples, n_features)
        Boolean mask to assign data.
    """
    if hasattr(X1, "mask"):  # pandas dataframes
        X1.mask(cond=cond, other=X2, inplace=True)
    else:  # ndarrays
        X1[cond] = X2[cond]


class IterativeImputer(_BaseImputer):
    # [CLASS DOC]
    # Round-robin iterative imputer. Fit: initial fill via SimpleImputer, then for up to
    # max_iter rounds, for each feature in an order chosen by imputation_order, fit a
    # clone of estimator (BayesianRidge by default) on rows where that feature is
    # observed, using the current values of neighbor features, and overwrite the missing
    # entries with the predictions (or truncated-normal draws if sample_posterior).
    # Stops early when the max absolute change between rounds falls below tol times the
    # max absolute observed value (only without sample_posterior). Transform replays the
    # stored sequence of fitted estimators. Fitted attributes: initial_imputer_,
    # imputation_sequence_, n_iter_, n_features_with_missing_, random_state_, indicator_
    # (if add_indicator), plus private _estimator, _min_value, _max_value,
    # _is_empty_feature.
    """Multivariate imputer that estimates each feature from all the others.

    A strategy for imputing missing values by modeling each feature with
    missing values as a function of other features in a round-robin fashion.

    Read more in the :ref:`User Guide <iterative_imputer>`.

    .. versionadded:: 0.21

    .. note::

      This estimator is still **experimental** for now: the predictions
      and the API might change without any deprecation cycle. To use it,
      you need to explicitly import `enable_iterative_imputer`::

        >>> # explicitly require this experimental feature
        >>> from sklearn.experimental import enable_iterative_imputer  # noqa
        >>> # now you can import normally from sklearn.impute
        >>> from sklearn.impute import IterativeImputer

    Parameters
    ----------
    estimator : estimator object, default=BayesianRidge()
        The estimator to use at each step of the round-robin imputation.
        If `sample_posterior=True`, the estimator must support
        `return_std` in its `predict` method.

    missing_values : int or np.nan, default=np.nan
        The placeholder for the missing values. All occurrences of
        `missing_values` will be imputed. For pandas' dataframes with
        nullable integer dtypes with missing values, `missing_values`
        should be set to `np.nan`, since `pd.NA` will be converted to `np.nan`.

    sample_posterior : bool, default=False
        Whether to sample from the (Gaussian) predictive posterior of the
        fitted estimator for each imputation. Estimator must support
        `return_std` in its `predict` method if set to `True`. Set to
        `True` if using `IterativeImputer` for multiple imputations.

    max_iter : int, default=10
        Maximum number of imputation rounds to perform before returning the
        imputations computed during the final round. A round is a single
        imputation of each feature with missing values. The stopping criterion
        is met once `max(abs(X_t - X_{t-1}))/max(abs(X[known_vals])) < tol`,
        where `X_t` is `X` at iteration `t`. Note that early stopping is only
        applied if `sample_posterior=False`.

    tol : float, default=1e-3
        Tolerance of the stopping condition.

    n_nearest_features : int, default=None
        Number of other features to use to estimate the missing values of
        each feature column. Nearness between features is measured using
        the absolute correlation coefficient between each feature pair (after
        initial imputation). To ensure coverage of features throughout the
        imputation process, the neighbor features are not necessarily nearest,
        but are drawn with probability proportional to correlation for each
        imputed target feature. Can provide significant speed-up when the
        number of features is huge. If `None`, all features will be used.

    initial_strategy : {'mean', 'median', 'most_frequent', 'constant'}, \
            default='mean'
        Which strategy to use to initialize the missing values. Same as the
        `strategy` parameter in :class:`~sklearn.impute.SimpleImputer`.

    fill_value : str or numerical value, default=None
        When `strategy="constant"`, `fill_value` is used to replace all
        occurrences of missing_values. For string or object data types,
        `fill_value` must be a string.
        If `None`, `fill_value` will be 0 when imputing numerical
        data and "missing_value" for strings or object data types.

        .. versionadded:: 1.3

    imputation_order : {'ascending', 'descending', 'roman', 'arabic', \
            'random'}, default='ascending'
        The order in which the features will be imputed. Possible values:

        - `'ascending'`: From features with fewest missing values to most.
        - `'descending'`: From features with most missing values to fewest.
        - `'roman'`: Left to right.
        - `'arabic'`: Right to left.
        - `'random'`: A random order for each round.

    skip_complete : bool, default=False
        If `True` then features with missing values during :meth:`transform`
        which did not have any missing values during :meth:`fit` will be
        imputed with the initial imputation method only. Set to `True` if you
        have many features with no missing values at both :meth:`fit` and
        :meth:`transform` time to save compute.

    min_value : float or array-like of shape (n_features,), default=-np.inf
        Minimum possible imputed value. Broadcast to shape `(n_features,)` if
        scalar. If array-like, expects shape `(n_features,)`, one min value for
        each feature. The default is `-np.inf`.

        .. versionchanged:: 0.23
           Added support for array-like.

    max_value : float or array-like of shape (n_features,), default=np.inf
        Maximum possible imputed value. Broadcast to shape `(n_features,)` if
        scalar. If array-like, expects shape `(n_features,)`, one max value for
        each feature. The default is `np.inf`.

        .. versionchanged:: 0.23
           Added support for array-like.

    verbose : int, default=0
        Verbosity flag, controls the debug messages that are issued
        as functions are evaluated. The higher, the more verbose. Can be 0, 1,
        or 2.

    random_state : int, RandomState instance or None, default=None
        The seed of the pseudo random number generator to use. Randomizes
        selection of estimator features if `n_nearest_features` is not `None`,
        the `imputation_order` if `random`, and the sampling from posterior if
        `sample_posterior=True`. Use an integer for determinism.
        See :term:`the Glossary <random_state>`.

    add_indicator : bool, default=False
        If `True`, a :class:`MissingIndicator` transform will stack onto output
        of the imputer's transform. This allows a predictive estimator
        to account for missingness despite imputation. If a feature has no
        missing values at fit/train time, the feature won't appear on
        the missing indicator even if there are missing values at
        transform/test time.

    keep_empty_features : bool, default=False
        If True, features that consist exclusively of missing values when
        `fit` is called are returned in results when `transform` is called.
        The imputed value is always `0` except when
        `initial_strategy="constant"` in which case `fill_value` will be
        used instead.

        .. versionadded:: 1.2

    Attributes
    ----------
    initial_imputer_ : object of type :class:`~sklearn.impute.SimpleImputer`
        Imputer used to initialize the missing values.

    imputation_sequence_ : list of tuples
        Each tuple has `(feat_idx, neighbor_feat_idx, estimator)`, where
        `feat_idx` is the current feature to be imputed,
        `neighbor_feat_idx` is the array of other features used to impute the
        current feature, and `estimator` is the trained estimator used for
        the imputation. Length is `self.n_features_with_missing_ *
        self.n_iter_`.

    n_iter_ : int
        Number of iteration rounds that occurred. Will be less than
        `self.max_iter` if early stopping criterion was reached.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    n_features_with_missing_ : int
        Number of features with missing values.

    indicator_ : :class:`~sklearn.impute.MissingIndicator`
        Indicator used to add binary indicators for missing values.
        `None` if `add_indicator=False`.

    random_state_ : RandomState instance
        RandomState instance that is generated either from a seed, the random
        number generator or by `np.random`.

    See Also
    --------
    SimpleImputer : Univariate imputer for completing missing values
        with simple strategies.
    KNNImputer : Multivariate imputer that estimates missing features using
        nearest samples.

    Notes
    -----
    To support imputation in inductive mode we store each feature's estimator
    during the :meth:`fit` phase, and predict without refitting (in order)
    during the :meth:`transform` phase.

    Features which contain all missing values at :meth:`fit` are discarded upon
    :meth:`transform`.

    Using defaults, the imputer scales in :math:`\\mathcal{O}(knp^3\\min(n,p))`
    where :math:`k` = `max_iter`, :math:`n` the number of samples and
    :math:`p` the number of features. It thus becomes prohibitively costly when
    the number of features increases. Setting
    `n_nearest_features << n_features`, `skip_complete=True` or increasing `tol`
    can help to reduce its computational cost.

    Depending on the nature of missing values, simple imputers can be
    preferable in a prediction context.

    References
    ----------
    .. [1] `Stef van Buuren, Karin Groothuis-Oudshoorn (2011). "mice:
        Multivariate Imputation by Chained Equations in R". Journal of
        Statistical Software 45: 1-67.
        <https://www.jstatsoft.org/article/view/v045i03>`_

    .. [2] `S. F. Buck, (1960). "A Method of Estimation of Missing Values in
        Multivariate Data Suitable for use with an Electronic Computer".
        Journal of the Royal Statistical Society 22(2): 302-306.
        <https://www.jstor.org/stable/2984099>`_

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.experimental import enable_iterative_imputer
    >>> from sklearn.impute import IterativeImputer
    >>> imp_mean = IterativeImputer(random_state=0)
    >>> imp_mean.fit([[7, 2, 3], [4, np.nan, 6], [10, 5, 9]])
    IterativeImputer(random_state=0)
    >>> X = [[np.nan, 2, 3], [4, np.nan, 6], [10, np.nan, 9]]
    >>> imp_mean.transform(X)
    array([[ 6.9584,  2.       ,  3.        ],
           [ 4.       ,  2.6000,  6.        ],
           [10.       ,  4.9999,  9.        ]])

    For a more detailed example see
    :ref:`sphx_glr_auto_examples_impute_plot_missing_values.py` or
    :ref:`sphx_glr_auto_examples_impute_plot_iterative_imputer_variants_comparison.py`.
    """

    _parameter_constraints: dict = {
        **_BaseImputer._parameter_constraints,
        "estimator": [None, HasMethods(["fit", "predict"])],
        "sample_posterior": ["boolean"],
        "max_iter": [Interval(Integral, 0, None, closed="left")],
        "tol": [Interval(Real, 0, None, closed="left")],
        "n_nearest_features": [None, Interval(Integral, 1, None, closed="left")],
        "initial_strategy": [
            StrOptions({"mean", "median", "most_frequent", "constant"})
        ],
        "fill_value": "no_validation",  # any object is valid
        "imputation_order": [
            StrOptions({"ascending", "descending", "roman", "arabic", "random"})
        ],
        "skip_complete": ["boolean"],
        "min_value": [None, Interval(Real, None, None, closed="both"), "array-like"],
        "max_value": [None, Interval(Real, None, None, closed="both"), "array-like"],
        "verbose": ["verbose"],
        "random_state": ["random_state"],
    }

    def __init__(
        self,
        estimator=None,
        *,
        missing_values=np.nan,
        sample_posterior=False,
        max_iter=10,
        tol=1e-3,
        n_nearest_features=None,
        initial_strategy="mean",
        fill_value=None,
        imputation_order="ascending",
        skip_complete=False,
        min_value=-np.inf,
        max_value=np.inf,
        verbose=0,
        random_state=None,
        add_indicator=False,
        keep_empty_features=False,
    ):
        # [FUNCTION DOC]
        # Stores all constructor parameters unchanged (sklearn convention, no validation
        # here); missing_values, add_indicator and keep_empty_features are forwarded to
        # _BaseImputer.
        # Outputs: None. Side effects: sets the same-named attributes on self.
        super().__init__(
            missing_values=missing_values,
            add_indicator=add_indicator,
            keep_empty_features=keep_empty_features,
        )

        self.estimator = estimator
        self.sample_posterior = sample_posterior
        self.max_iter = max_iter
        self.tol = tol
        self.n_nearest_features = n_nearest_features
        self.initial_strategy = initial_strategy
        self.fill_value = fill_value
        self.imputation_order = imputation_order
        self.skip_complete = skip_complete
        self.min_value = min_value
        self.max_value = max_value
        self.verbose = verbose
        self.random_state = random_state

    def _impute_one_feature(
        self,
        X_filled,
        mask_missing_values,
        feat_idx,
        neighbor_feat_idx,
        estimator=None,
        fit_mode=True,
        params=None,
    ):
        # [FUNCTION DOC]
        # HELPER: Performs one imputation step for column feat_idx. Steps: if fit_mode,
        # take rows where the feature is observed, use columns neighbor_feat_idx as
        # inputs and feat_idx as target and fit the estimator (a fresh clone of
        # self._estimator when estimator is None) with params. If the feature has no
        # missing rows, return immediately. Otherwise predict on missing rows: with
        # sample_posterior, get mean and std; entries with non-positive std take the
        # mean, means below/above the feature bounds take the bound, the rest are
        # sampled from a normal truncated to [min, max] using random_state_. Without it,
        # predictions are clipped to the bounds. The results are written into X_filled
        # at the missing rows of feat_idx.
        # Inputs: X_filled - 2-D array with current imputations; mask_missing_values -
        # boolean mask of the same shape; feat_idx - int column; neighbor_feat_idx - int
        # array of predictor columns; estimator - fitted or unfitted estimator or None;
        # fit_mode - bool; params - dict of fit kwargs (must be a dict when fit_mode is
        # True).
        # Outputs: tuple (X_filled, estimator).
        # Exceptions: ValueError if estimator is None while fit_mode is False.
        # Side effects: mutates X_filled in place; consumes random_state_ when sampling.
        """Impute a single feature from the others provided.

        This function predicts the missing values of one of the features using
        the current estimates of all the other features. The `estimator` must
        support `return_std=True` in its `predict` method for this function
        to work.

        Parameters
        ----------
        X_filled : ndarray
            Input data with the most recent imputations.

        mask_missing_values : ndarray
            Input data's missing indicator matrix.

        feat_idx : int
            Index of the feature currently being imputed.

        neighbor_feat_idx : ndarray
            Indices of the features to be used in imputing `feat_idx`.

        estimator : object
            The estimator to use at this step of the round-robin imputation.
            If `sample_posterior=True`, the estimator must support
            `return_std` in its `predict` method.
            If None, it will be cloned from self._estimator.

        fit_mode : boolean, default=True
            Whether to fit and predict with the estimator or just predict.

        params : dict
            Additional params routed to the individual estimator.

        Returns
        -------
        X_filled : ndarray
            Input data with `X_filled[missing_row_mask, feat_idx]` updated.

        estimator : estimator with sklearn API
            The fitted estimator used to impute
            `X_filled[missing_row_mask, feat_idx]`.
        """
        if estimator is None and fit_mode is False:
            raise ValueError(
                "If fit_mode is False, then an already-fitted "
                "estimator should be passed in."
            )

        if estimator is None:
            estimator = clone(self._estimator)

        missing_row_mask = mask_missing_values[:, feat_idx]
        if fit_mode:
            X_train = _safe_indexing(
                _safe_indexing(X_filled, neighbor_feat_idx, axis=1),
                ~missing_row_mask,
                axis=0,
            )
            y_train = _safe_indexing(
                _safe_indexing(X_filled, feat_idx, axis=1),
                ~missing_row_mask,
                axis=0,
            )
            estimator.fit(X_train, y_train, **params)

        # if no missing values, don't predict
        if np.sum(missing_row_mask) == 0:
            return X_filled, estimator

        # get posterior samples if there is at least one missing value
        X_test = _safe_indexing(
            _safe_indexing(X_filled, neighbor_feat_idx, axis=1),
            missing_row_mask,
            axis=0,
        )
        if self.sample_posterior:
            mus, sigmas = estimator.predict(X_test, return_std=True)
            imputed_values = np.zeros(mus.shape, dtype=X_filled.dtype)
            # two types of problems: (1) non-positive sigmas
            # (2) mus outside legal range of min_value and max_value
            # (results in inf sample)
            positive_sigmas = sigmas > 0
            imputed_values[~positive_sigmas] = mus[~positive_sigmas]
            mus_too_low = mus < self._min_value[feat_idx]
            imputed_values[mus_too_low] = self._min_value[feat_idx]
            mus_too_high = mus > self._max_value[feat_idx]
            imputed_values[mus_too_high] = self._max_value[feat_idx]
            # the rest can be sampled without statistical issues
            inrange_mask = positive_sigmas & ~mus_too_low & ~mus_too_high
            mus = mus[inrange_mask]
            sigmas = sigmas[inrange_mask]
            a = (self._min_value[feat_idx] - mus) / sigmas
            b = (self._max_value[feat_idx] - mus) / sigmas

            truncated_normal = stats.truncnorm(a=a, b=b, loc=mus, scale=sigmas)
            imputed_values[inrange_mask] = truncated_normal.rvs(
                random_state=self.random_state_
            )
        else:
            imputed_values = estimator.predict(X_test)
            imputed_values = np.clip(
                imputed_values, self._min_value[feat_idx], self._max_value[feat_idx]
            )

        # update the feature
        _safe_assign(
            X_filled,
            imputed_values,
            row_indexer=missing_row_mask,
            column_indexer=feat_idx,
        )
        return X_filled, estimator

    def _get_neighbor_feat_idx(self, n_features, feat_idx, abs_corr_mat):
        # [FUNCTION DOC]
        # HELPER: Chooses predictor columns for feat_idx. When n_nearest_features is set
        # and smaller than n_features, draws n_nearest_features distinct columns with
        # probabilities given by column feat_idx of abs_corr_mat (whose diagonal is
        # zero, so feat_idx itself is never drawn); otherwise returns all columns except
        # feat_idx in ascending order.
        # Inputs: n_features - int; feat_idx - int; abs_corr_mat - (n_features,
        # n_features) column-normalized array or None (only used in the sampling
        # branch).
        # Outputs: 1-D int ndarray of column indices.
        # Side effects: consumes random_state_ in the sampling branch.
        """Get a list of other features to predict `feat_idx`.

        If `self.n_nearest_features` is less than or equal to the total
        number of features, then use a probability proportional to the absolute
        correlation between `feat_idx` and each other feature to randomly
        choose a subsample of the other features (without replacement).

        Parameters
        ----------
        n_features : int
            Number of features in `X`.

        feat_idx : int
            Index of the feature currently being imputed.

        abs_corr_mat : ndarray, shape (n_features, n_features)
            Absolute correlation matrix of `X`. The diagonal has been zeroed
            out and each feature has been normalized to sum to 1. Can be None.

        Returns
        -------
        neighbor_feat_idx : array-like
            The features to use to impute `feat_idx`.
        """
        if self.n_nearest_features is not None and self.n_nearest_features < n_features:
            p = abs_corr_mat[:, feat_idx]
            neighbor_feat_idx = self.random_state_.choice(
                np.arange(n_features), self.n_nearest_features, replace=False, p=p
            )
        else:
            inds_left = np.arange(feat_idx)
            inds_right = np.arange(feat_idx + 1, n_features)
            neighbor_feat_idx = np.concatenate((inds_left, inds_right))
        return neighbor_feat_idx

    def _get_ordered_idx(self, mask_missing_values):
        # [FUNCTION DOC]
        # HELPER: Determines the order of features visited in one round. Candidate
        # features are all columns, or only columns having at least one missing value
        # when skip_complete. "roman" keeps index order, "arabic" reverses it,
        # "ascending"/"descending" stably sort by fraction of missing values (fewest /
        # most first) keeping only the candidates, and "random" shuffles the candidate
        # indices.
        # Inputs: mask_missing_values - boolean array (n_samples, n_features).
        # Outputs: 1-D int ndarray of feature indices.
        # Side effects: "random" shuffles in place using random_state_.
        # KNOWN ISSUE: docstring says the output has shape (n_features,) and that
        # complete features are always skipped; both only hold when skip_complete is
        # True / False resp.
        # KNOWN ISSUE: the ascending/descending branches assume the excluded complete
        # features are exactly those with the smallest missing fraction (zero), which
        # holds, but the argsort slicing relies on that implicitly.
        """Decide in what order we will update the features.

        As a homage to the MICE R package, we will have 4 main options of
        how to order the updates, and use a random order if anything else
        is specified.

        Also, this function skips features which have no missing values.

        Parameters
        ----------
        mask_missing_values : array-like, shape (n_samples, n_features)
            Input data's missing indicator matrix, where `n_samples` is the
            number of samples and `n_features` is the number of features.

        Returns
        -------
        ordered_idx : ndarray, shape (n_features,)
            The order in which to impute the features.
        """
        frac_of_missing_values = mask_missing_values.mean(axis=0)
        if self.skip_complete:
            missing_values_idx = np.flatnonzero(frac_of_missing_values)
        else:
            missing_values_idx = np.arange(np.shape(frac_of_missing_values)[0])
        if self.imputation_order == "roman":
            ordered_idx = missing_values_idx
        elif self.imputation_order == "arabic":
            ordered_idx = missing_values_idx[::-1]
        elif self.imputation_order == "ascending":
            n = len(frac_of_missing_values) - len(missing_values_idx)
            ordered_idx = np.argsort(frac_of_missing_values, kind="mergesort")[n:]
        elif self.imputation_order == "descending":
            n = len(frac_of_missing_values) - len(missing_values_idx)
            ordered_idx = np.argsort(frac_of_missing_values, kind="mergesort")[n:][::-1]
        elif self.imputation_order == "random":
            ordered_idx = missing_values_idx
            self.random_state_.shuffle(ordered_idx)
        return ordered_idx

    def _get_abs_corr_mat(self, X_filled, tolerance=1e-6):
        # [FUNCTION DOC]
        # HELPER: Builds the neighbor sampling matrix used when n_nearest_features
        # restricts the predictors. Returns None when n_nearest_features is None or >=
        # n_features. Otherwise computes the absolute Pearson correlation between
        # columns, replaces NaN (constant columns) by tolerance, clips values below
        # tolerance up to tolerance, zeroes the diagonal and L1-normalizes each column
        # so it sums to 1.
        # Inputs: X_filled - 2-D array (n_samples, n_features); tolerance - float,
        # default 1e-6.
        # Outputs: (n_features, n_features) ndarray or None.
        # Side effects: none (numpy invalid-value warnings suppressed locally).
        """Get absolute correlation matrix between features.

        Parameters
        ----------
        X_filled : ndarray, shape (n_samples, n_features)
            Input data with the most recent imputations.

        tolerance : float, default=1e-6
            `abs_corr_mat` can have nans, which will be replaced
            with `tolerance`.

        Returns
        -------
        abs_corr_mat : ndarray, shape (n_features, n_features)
            Absolute correlation matrix of `X` at the beginning of the
            current round. The diagonal has been zeroed out and each feature's
            absolute correlations with all others have been normalized to sum
            to 1.
        """
        n_features = X_filled.shape[1]
        if self.n_nearest_features is None or self.n_nearest_features >= n_features:
            return None
        with np.errstate(invalid="ignore"):
            # if a feature in the neighborhood has only a single value
            # (e.g., categorical feature), the std. dev. will be null and
            # np.corrcoef will raise a warning due to a division by zero
            abs_corr_mat = np.abs(np.corrcoef(X_filled.T))
        # np.corrcoef is not defined for features with zero std
        abs_corr_mat[np.isnan(abs_corr_mat)] = tolerance
        # ensures exploration, i.e. at least some probability of sampling
        np.clip(abs_corr_mat, tolerance, None, out=abs_corr_mat)
        # features are not their own neighbors
        np.fill_diagonal(abs_corr_mat, 0)
        # needs to sum to 1 for np.random.choice sampling
        abs_corr_mat = normalize(abs_corr_mat, norm="l1", axis=0, copy=False)
        return abs_corr_mat

    def _initial_imputation(self, X, in_fit=False):
        # [FUNCTION DOC]
        # HELPER: Validates X and computes the starting imputation. Steps: validate X as
        # float, Fortran order, allowing NaN only when missing_values is NaN (feature
        # counts/names reset when in_fit); check dtype compatibility; compute the
        # missing mask. If initial_imputer_ is None, create and fit_transform a
        # SimpleImputer with initial_strategy/fill_value/keep_empty_features (default
        # output container); otherwise transform with it. In fit, record
        # _is_empty_feature (all-missing columns). If keep_empty_features is False, drop
        # those columns from X and from the working mask; otherwise mark them as not
        # missing and copy their initial fill into X.
        # Inputs: X - array-like (n_samples, n_features); in_fit - bool, default False.
        # Outputs: tuple (Xt, X_filled, mask_missing_values, X_missing_mask): validated
        # data (possibly column-reduced), initial imputation, working mask aligned with
        # Xt, and the full original missing mask.
        # Exceptions: propagated from validate_data / _check_inputs_dtype (e.g. inf
        # values, feature-count mismatch, incompatible dtype).
        # Side effects: may set initial_imputer_, _is_empty_feature and validation
        # attributes (n_features_in_, feature_names_in_) on self; when
        # keep_empty_features is True writes into the validated array (which may alias
        # the caller's input if no copy was needed).
        """Perform initial imputation for input `X`.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        in_fit : bool, default=False
            Whether function is called in :meth:`fit`.

        Returns
        -------
        Xt : ndarray of shape (n_samples, n_features)
            Input data, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        X_filled : ndarray of shape (n_samples, n_features)
            Input data with the most recent imputations.

        mask_missing_values : ndarray of shape (n_samples, n_features)
            Input data's missing indicator matrix, where `n_samples` is the
            number of samples and `n_features` is the number of features,
            masked by non-missing features.

        X_missing_mask : ndarray, shape (n_samples, n_features)
            Input data's mask matrix indicating missing datapoints, where
            `n_samples` is the number of samples and `n_features` is the
            number of features.
        """
        if is_scalar_nan(self.missing_values):
            ensure_all_finite = "allow-nan"
        else:
            ensure_all_finite = True

        X = validate_data(
            self,
            X,
            dtype=FLOAT_DTYPES,
            order="F",
            reset=in_fit,
            ensure_all_finite=ensure_all_finite,
        )
        _check_inputs_dtype(X, self.missing_values)

        X_missing_mask = _get_mask(X, self.missing_values)
        mask_missing_values = X_missing_mask.copy()

        if self.initial_imputer_ is None:
            self.initial_imputer_ = SimpleImputer(
                missing_values=self.missing_values,
                strategy=self.initial_strategy,
                fill_value=self.fill_value,
                keep_empty_features=self.keep_empty_features,
            ).set_output(transform="default")

            X_filled = self.initial_imputer_.fit_transform(X)

        else:
            X_filled = self.initial_imputer_.transform(X)

        if in_fit:
            self._is_empty_feature = np.all(mask_missing_values, axis=0)

        if not self.keep_empty_features:
            # drop empty features
            Xt = X[:, ~self._is_empty_feature]
            mask_missing_values = mask_missing_values[:, ~self._is_empty_feature]

        else:
            # mark empty features as not missing and keep the original
            # imputation
            mask_missing_values[:, self._is_empty_feature] = False
            Xt = X
            Xt[:, self._is_empty_feature] = X_filled[:, self._is_empty_feature]

        return Xt, X_filled, mask_missing_values, X_missing_mask

    @staticmethod
    def _validate_limit(
        limit, limit_type, n_features, is_empty_feature, keep_empty_feature
    ):
        # [FUNCTION DOC]
        # HELPER: Normalizes min_value/max_value into a per-feature float array. None
        # becomes +inf (max) or -inf (min); a scalar is broadcast to length n_features;
        # an array must have one entry per original input feature and, when
        # keep_empty_feature is False, the entries of empty features are removed.
        # Inputs: limit - None, scalar or array-like; limit_type - "min" or "max";
        # n_features - number of features after empty-feature removal; is_empty_feature
        # - boolean array of length n_features_in_; keep_empty_feature - bool.
        # Outputs: 1-D ndarray of bounds.
        # Exceptions: ValueError when an array-like limit length differs from the number
        # of original features.
        """Validate the limits (min/max) of the feature values.

        Converts scalar min/max limits to vectors of shape `(n_features,)`.

        Parameters
        ----------
        limit: scalar or array-like
            The user-specified limit (i.e, min_value or max_value).
        limit_type: {'max', 'min'}
            Type of limit to validate.
        n_features: int
            Number of features in the dataset.
        is_empty_feature: ndarray, shape (n_features, )
            Mask array indicating empty feature imputer has seen during fit.
        keep_empty_feature: bool
            If False, remove empty-feature indices from the limit.

        Returns
        -------
        limit: ndarray, shape(n_features,)
            Array of limits, one for each feature.
        """
        n_features_in = _num_samples(is_empty_feature)
        if (
            limit is not None
            and not np.isscalar(limit)
            and _num_samples(limit) != n_features_in
        ):
            raise ValueError(
                f"'{limit_type}_value' should be of shape ({n_features_in},) when an"
                f" array-like is provided. Got {len(limit)}, instead."
            )

        limit_bound = np.inf if limit_type == "max" else -np.inf
        limit = limit_bound if limit is None else limit
        if np.isscalar(limit):
            limit = np.full(n_features, limit)
        limit = check_array(limit, ensure_all_finite=False, copy=False, ensure_2d=False)

        # Make sure to remove the empty feature elements from the bounds
        if not keep_empty_feature and len(limit) == len(is_empty_feature):
            limit = limit[~is_empty_feature]

        return limit

    @_fit_context(
        # IterativeImputer.estimator is not validated yet
        prefer_skip_nested_validation=False
    )
    def fit_transform(self, X, y=None, **params):
        # [FUNCTION DOC]
        # Fits the imputer and returns the imputed training data. Steps: validate routed
        # params; keep an existing random_state_ or create one; set _estimator
        # (BayesianRidge if estimator is None, else a clone); reset imputation_sequence_
        # and initial_imputer_; run _initial_imputation; fit and compute the missing
        # indicator. If max_iter is 0, every remaining entry is missing, or there is a
        # single feature, set n_iter_ to 0 and return the initial imputation. Otherwise
        # build _min_value/_max_value, compute the visiting order and the correlation
        # matrix, then run rounds 1..max_iter, each calling _impute_one_feature for
        # every ordered feature and appending an _ImputerTriplet. Without
        # sample_posterior, stop when the infinity norm of the change since last round
        # is below tol times the max absolute observed value. Finally restore the
        # observed entries from X and append the indicator columns.
        # Inputs: X - array-like (n_samples, n_features); y - ignored; **params -
        # metadata routed to the estimator's fit.
        # Outputs: ndarray (n_samples, n_output_features).
        # Exceptions: ValueError if any feature has min_value >= max_value or limit
        # shapes are wrong; errors from routing validation when params are passed
        # without routing enabled.
        # Side effects: sets all fitted attributes; prints progress when verbose; emits
        # ConvergenceWarning when max_iter rounds finish without meeting tol (not with
        # sample_posterior).
        # KNOWN ISSUE: random_state_ is reused if already present, so refitting
        # continues the previous random stream instead of restarting from random_state.
        """Fit the imputer on `X` and return the transformed `X`.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input data, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        y : Ignored
            Not used, present for API consistency by convention.

        **params : dict
            Parameters routed to the `fit` method of the sub-estimator via the
            metadata routing API.

            .. versionadded:: 1.5
              Only available if
              `sklearn.set_config(enable_metadata_routing=True)` is set. See
              :ref:`Metadata Routing User Guide <metadata_routing>` for more
              details.

        Returns
        -------
        Xt : array-like, shape (n_samples, n_features)
            The imputed input data.
        """
        _raise_for_params(params, self, "fit")

        routed_params = process_routing(
            self,
            "fit",
            **params,
        )

        self.random_state_ = getattr(
            self, "random_state_", check_random_state(self.random_state)
        )

        if self.estimator is None:
            from sklearn.linear_model import BayesianRidge

            self._estimator = BayesianRidge()
        else:
            self._estimator = clone(self.estimator)

        self.imputation_sequence_ = []

        self.initial_imputer_ = None

        X, Xt, mask_missing_values, complete_mask = self._initial_imputation(
            X, in_fit=True
        )

        super()._fit_indicator(complete_mask)
        X_indicator = super()._transform_indicator(complete_mask)

        if self.max_iter == 0 or np.all(mask_missing_values):
            self.n_iter_ = 0
            return super()._concatenate_indicator(Xt, X_indicator)

        # Edge case: a single feature, we return the initial imputation.
        if Xt.shape[1] == 1:
            self.n_iter_ = 0
            return super()._concatenate_indicator(Xt, X_indicator)

        self._min_value = self._validate_limit(
            self.min_value,
            "min",
            X.shape[1],
            self._is_empty_feature,
            self.keep_empty_features,
        )
        self._max_value = self._validate_limit(
            self.max_value,
            "max",
            X.shape[1],
            self._is_empty_feature,
            self.keep_empty_features,
        )

        if not np.all(np.greater(self._max_value, self._min_value)):
            raise ValueError("One (or more) features have min_value >= max_value.")

        # order in which to impute
        # note this is probably too slow for large feature data (d > 100000)
        # and a better way would be good.
        # see: https://goo.gl/KyCNwj and subsequent comments
        ordered_idx = self._get_ordered_idx(mask_missing_values)
        self.n_features_with_missing_ = len(ordered_idx)

        abs_corr_mat = self._get_abs_corr_mat(Xt)

        n_samples, n_features = Xt.shape
        if self.verbose > 0:
            print("[IterativeImputer] Completing matrix with shape %s" % (X.shape,))
        start_t = time()
        if not self.sample_posterior:
            Xt_previous = Xt.copy()
            normalized_tol = self.tol * np.max(np.abs(X[~mask_missing_values]))
        for self.n_iter_ in range(1, self.max_iter + 1):
            if self.imputation_order == "random":
                ordered_idx = self._get_ordered_idx(mask_missing_values)

            for feat_idx in ordered_idx:
                neighbor_feat_idx = self._get_neighbor_feat_idx(
                    n_features, feat_idx, abs_corr_mat
                )
                Xt, estimator = self._impute_one_feature(
                    Xt,
                    mask_missing_values,
                    feat_idx,
                    neighbor_feat_idx,
                    estimator=None,
                    fit_mode=True,
                    params=routed_params.estimator.fit,
                )
                estimator_triplet = _ImputerTriplet(
                    feat_idx, neighbor_feat_idx, estimator
                )
                self.imputation_sequence_.append(estimator_triplet)

            if self.verbose > 1:
                print(
                    "[IterativeImputer] Ending imputation round "
                    "%d/%d, elapsed time %0.2f"
                    % (self.n_iter_, self.max_iter, time() - start_t)
                )

            if not self.sample_posterior:
                inf_norm = np.linalg.norm(Xt - Xt_previous, ord=np.inf, axis=None)
                if self.verbose > 0:
                    print(
                        "[IterativeImputer] Change: {}, scaled tolerance: {} ".format(
                            inf_norm, normalized_tol
                        )
                    )
                if inf_norm < normalized_tol:
                    if self.verbose > 0:
                        print("[IterativeImputer] Early stopping criterion reached.")
                    break
                Xt_previous = Xt.copy()
        else:
            if not self.sample_posterior:
                warnings.warn(
                    "[IterativeImputer] Early stopping criterion not reached.",
                    ConvergenceWarning,
                )
        _assign_where(Xt, X, cond=~mask_missing_values)

        return super()._concatenate_indicator(Xt, X_indicator)

    def transform(self, X):
        # [FUNCTION DOC]
        # Imputes new data by replaying imputation_sequence_ with fit_mode False: after
        # the initial imputation, each stored (feat_idx, neighbor_feat_idx, estimator)
        # triplet predicts its feature in sequence; rounds are counted only for verbose
        # output. Returns the initial imputation when n_iter_ is 0 or everything is
        # missing. Observed values are restored and indicator columns appended.
        # Inputs: X - array-like (n_samples, n_features).
        # Outputs: ndarray (n_samples, n_output_features).
        # Exceptions: NotFittedError if not fitted; validation errors from
        # _initial_imputation.
        # Side effects: prints when verbose; consumes random_state_ when
        # sample_posterior.
        """Impute all missing values in `X`.

        Note that this is stochastic, and that if `random_state` is not fixed,
        repeated calls, or permuted input, results will differ.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data to complete.

        Returns
        -------
        Xt : array-like, shape (n_samples, n_features)
             The imputed input data.
        """
        check_is_fitted(self)

        X, Xt, mask_missing_values, complete_mask = self._initial_imputation(
            X, in_fit=False
        )

        X_indicator = super()._transform_indicator(complete_mask)

        if self.n_iter_ == 0 or np.all(mask_missing_values):
            return super()._concatenate_indicator(Xt, X_indicator)

        imputations_per_round = len(self.imputation_sequence_) // self.n_iter_
        i_rnd = 0
        if self.verbose > 0:
            print("[IterativeImputer] Completing matrix with shape %s" % (X.shape,))
        start_t = time()
        for it, estimator_triplet in enumerate(self.imputation_sequence_):
            Xt, _ = self._impute_one_feature(
                Xt,
                mask_missing_values,
                estimator_triplet.feat_idx,
                estimator_triplet.neighbor_feat_idx,
                estimator=estimator_triplet.estimator,
                fit_mode=False,
            )
            if not (it + 1) % imputations_per_round:
                if self.verbose > 1:
                    print(
                        "[IterativeImputer] Ending imputation round "
                        "%d/%d, elapsed time %0.2f"
                        % (i_rnd + 1, self.n_iter_, time() - start_t)
                    )
                i_rnd += 1

        _assign_where(Xt, X, cond=~mask_missing_values)

        return super()._concatenate_indicator(Xt, X_indicator)

    def fit(self, X, y=None, **fit_params):
        # [FUNCTION DOC]
        # Fits by calling fit_transform and discarding its output.
        # Inputs: X - array-like; y - ignored; **fit_params - routed metadata.
        # Outputs: self. Side effects: same as fit_transform.
        """Fit the imputer on `X` and return self.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input data, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        y : Ignored
            Not used, present for API consistency by convention.

        **fit_params : dict
            Parameters routed to the `fit` method of the sub-estimator via the
            metadata routing API.

            .. versionadded:: 1.5
              Only available if
              `sklearn.set_config(enable_metadata_routing=True)` is set. See
              :ref:`Metadata Routing User Guide <metadata_routing>` for more
              details.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        self.fit_transform(X, **fit_params)
        return self

    def get_feature_names_out(self, input_features=None):
        # [FUNCTION DOC]
        # Returns output feature names: the initial imputer's names for the input
        # features (empty features dropped unless kept) followed by missing-indicator
        # names if add_indicator.
        # Inputs: input_features - None or array-like of str.
        # Outputs: ndarray of str.
        # Exceptions: NotFittedError; ValueError if input_features mismatch
        # feature_names_in_.
        """Get output feature names for transformation.

        Parameters
        ----------
        input_features : array-like of str or None, default=None
            Input features.

            - If `input_features` is `None`, then `feature_names_in_` is
              used as feature names in. If `feature_names_in_` is not defined,
              then the following input feature names are generated:
              `["x0", "x1", ..., "x(n_features_in_ - 1)"]`.
            - If `input_features` is an array-like, then `input_features` must
              match `feature_names_in_` if `feature_names_in_` is defined.

        Returns
        -------
        feature_names_out : ndarray of str objects
            Transformed feature names.
        """
        check_is_fitted(self, "n_features_in_")
        input_features = _check_feature_names_in(self, input_features)
        names = self.initial_imputer_.get_feature_names_out(input_features)
        return self._concatenate_indicator_feature_names_out(names, input_features)

    def get_metadata_routing(self):
        # [FUNCTION DOC]
        # Builds a MetadataRouter that routes fit metadata to the sub-estimator's fit.
        # Outputs: MetadataRouter. Side effects: none.
        # KNOWN ISSUE: routes to self.estimator, which may be None (default
        # BayesianRidge is only created in fit), so no routing is declared for the
        # default estimator.
        """Get metadata routing of this object.

        Please check :ref:`User Guide <metadata_routing>` on how the routing
        mechanism works.

        .. versionadded:: 1.5

        Returns
        -------
        routing : MetadataRouter
            A :class:`~sklearn.utils.metadata_routing.MetadataRouter` encapsulating
            routing information.
        """
        router = MetadataRouter(owner=self).add(
            estimator=self.estimator,
            method_mapping=MethodMapping().add(callee="fit", caller="fit"),
        )
        return router
