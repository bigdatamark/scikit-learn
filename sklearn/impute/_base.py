# [MODULE DOC]
# Core univariate imputation module: defines _BaseImputer (shared add_indicator plumbing
# for all imputers), SimpleImputer (per-column mean/median/most-frequent/
# constant/callable imputation for dense and CSC sparse data) and MissingIndicator
# (boolean mask of missing entries, optionally restricted to features that had missing
# values at fit), plus small helpers for dtype checks and mode computation.
# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import numbers
import warnings
from collections import Counter
from functools import partial
from typing import Callable

import numpy as np
import numpy.ma as ma
from scipy import sparse as sp

from sklearn.base import BaseEstimator, TransformerMixin, _fit_context
from sklearn.utils._mask import _get_mask
from sklearn.utils._missing import is_pandas_na, is_scalar_nan
from sklearn.utils._param_validation import MissingValues, StrOptions
from sklearn.utils._sparse import _align_api_if_sparse
from sklearn.utils.fixes import SCIPY_VERSION_BELOW_1_12, _mode
from sklearn.utils.sparsefuncs import _get_median
from sklearn.utils.validation import (
    FLOAT_DTYPES,
    _check_feature_names_in,
    _check_n_features,
    check_is_fitted,
    validate_data,
)


def _check_inputs_dtype(X, missing_values):
    # [FUNCTION DOC]
    # HELPER: Checks that a numeric X is paired with a numeric missing_values
    # placeholder.
    # Inputs: X - validated array or sparse matrix with a dtype; missing_values - scalar
    # placeholder.
    # Outputs: None. Returns early (accepts) when missing_values is pandas NA.
    # Exceptions: ValueError if X has float/int/uint dtype and missing_values is not a
    # real number (e.g. a string).
    if is_pandas_na(missing_values):
        # Allow using `pd.NA` as missing values to impute numerical arrays.
        return
    if X.dtype.kind in ("f", "i", "u") and not isinstance(missing_values, numbers.Real):
        raise ValueError(
            "'X' and 'missing_values' types are expected to be"
            " both numerical. Got X.dtype={} and "
            " type(missing_values)={}.".format(X.dtype, type(missing_values))
        )


def _safe_min(items):
    # [FUNCTION DOC]
    # HELPER: Minimum of a list that may mix non-comparable types. Tries the builtin
    # min; on a "'<' not supported" TypeError, falls back to ordering by (type name
    # string, value string).
    # Inputs: items - non-empty iterable. Outputs: one element of items.
    # Exceptions: re-raises any other TypeError; ValueError from min on an empty list.
    """Compute the minimum of a list of potentially non-comparable values.

    If values cannot be directly compared due to type incompatibility, the object with
    the lowest string representation is returned.
    """
    try:
        return min(items)
    except TypeError as e:
        if "'<' not supported between" in str(e):
            return min(items, key=lambda x: (str(type(x)), str(x)))
        raise  # pragma: no cover


def _most_frequent(array, extra_value, n_repeat):
    # [FUNCTION DOC]
    # HELPER: Mode of a 1D array conceptually extended with n_repeat copies of
    # extra_value (used for sparse columns whose implicit zeros are not stored).
    # Algorithm: compute the mode of array (Counter for object dtype, ties broken by
    # _safe_min; scipy-based _mode otherwise, which returns the smallest tied value),
    # with count 0 for an empty array; then compare its count with n_repeat: return NaN
    # if both are 0, extra_value if n_repeat is larger, the array mode if larger, and
    # the smaller of the two (via _safe_min) on a tie.
    # Inputs: array - 1D ndarray of non-missing values; extra_value - value assumed
    # absent from array; n_repeat - int >= 0.
    # Outputs: scalar most frequent value, or np.nan.
    """Compute the most frequent value in a 1d array extended with
    [extra_value] * n_repeat, where extra_value is assumed to be not part
    of the array."""
    # Compute the most frequent value in array only
    if array.size > 0:
        if array.dtype == object:
            # scipy.stats.mode is slow with object dtype array.
            # Python Counter is more efficient
            counter = Counter(array)
            most_frequent_count = counter.most_common(1)[0][1]
            # tie breaking similarly to scipy.stats.mode
            most_frequent_value = _safe_min(
                [
                    value
                    for value, count in counter.items()
                    if count == most_frequent_count
                ]
            )
        else:
            mode = _mode(array)
            most_frequent_value = mode[0][0]
            most_frequent_count = mode[1][0]
    else:
        most_frequent_value = 0
        most_frequent_count = 0

    # Compare to array + [extra_value] * n_repeat
    if most_frequent_count == 0 and n_repeat == 0:
        return np.nan
    elif most_frequent_count < n_repeat:
        return extra_value
    elif most_frequent_count > n_repeat:
        return most_frequent_value
    elif most_frequent_count == n_repeat:
        # tie breaking similarly to scipy.stats.mode
        return _safe_min([most_frequent_value, extra_value])


class _BaseImputer(TransformerMixin, BaseEstimator):
    # [CLASS DOC]
    # HELPER: Base class for imputers adding the optional MissingIndicator
    # (add_indicator) and keep_empty_features handling; subclasses call _fit_indicator
    # during fit and _transform_indicator / _concatenate_indicator during transform.
    """Base class for all imputers.

    It adds automatically support for `add_indicator`.
    """

    _parameter_constraints: dict = {
        "missing_values": [MissingValues()],
        "add_indicator": ["boolean"],
        "keep_empty_features": ["boolean"],
    }

    def __init__(
        self, *, missing_values=np.nan, add_indicator=False, keep_empty_features=False
    ):
        # [FUNCTION DOC]
        # Stores missing_values (default np.nan), add_indicator (default False) and
        # keep_empty_features (default False); keyword-only.
        self.missing_values = missing_values
        self.add_indicator = add_indicator
        self.keep_empty_features = keep_empty_features

    def _fit_indicator(self, X):
        # [FUNCTION DOC]
        # HELPER: If add_indicator, creates and fits indicator_ as a MissingIndicator
        # with error_on_new=False on X, which must already be a boolean missing mask
        # (precomputed); otherwise sets indicator_ to None.
        # Inputs: X - boolean mask array or sparse matrix (n_samples, n_features).
        # Side effects: sets self.indicator_.
        # Exceptions: ValueError from MissingIndicator._fit if X is not boolean.
        """Fit a MissingIndicator."""
        if self.add_indicator:
            self.indicator_ = MissingIndicator(
                missing_values=self.missing_values, error_on_new=False
            )
            self.indicator_._fit(X, precomputed=True)
        else:
            self.indicator_ = None

    def _transform_indicator(self, X):
        # [FUNCTION DOC]
        # HELPER: Returns indicator_.transform(X) when add_indicator, else None
        # (implicitly).
        # Inputs: X - the boolean missing mask of the original (pre-imputation) data.
        # Exceptions: ValueError if add_indicator but _fit_indicator was never called.
        """Compute the indicator mask.'

        Note that X must be the original data as passed to the imputer before
        any imputation, since imputation may be done inplace in some cases.
        """
        if self.add_indicator:
            if not hasattr(self, "indicator_"):
                raise ValueError(
                    "Make sure to call _fit_indicator before _transform_indicator"
                )
            return self.indicator_.transform(X)

    def _concatenate_indicator(self, X_imputed, X_indicator):
        # [FUNCTION DOC]
        # HELPER: Horizontally stacks the imputed data and the indicator mask when
        # add_indicator, otherwise returns X_imputed unchanged. For sparse input the
        # sparse format of X_imputed is kept and the result is aligned to sparse
        # array/matrix API.
        # Inputs: X_imputed - dense or sparse (n_samples, n_features_kept); X_indicator
        # - mask.
        # Outputs: combined array (n_samples, n_features_kept + n_indicator_features).
        # Exceptions: ValueError if add_indicator and X_indicator is None.
        """Concatenate indicator mask with the imputed data."""
        if not self.add_indicator:
            return X_imputed

        if sp.issparse(X_imputed):
            # sp.hstack may result in different formats between sparse arrays and
            # matrices; specify the format to keep consistent behavior
            hstack = partial(sp.hstack, format=X_imputed.format)
        else:
            hstack = np.hstack

        if X_indicator is None:
            raise ValueError(
                "Data from the missing indicator are not provided. Call "
                "_fit_indicator and _transform_indicator in the imputer "
                "implementation."
            )

        return _align_api_if_sparse(hstack((X_imputed, X_indicator)))

    def _concatenate_indicator_feature_names_out(self, names, input_features):
        # [FUNCTION DOC]
        # HELPER: Appends the indicator's output feature names to names when
        # add_indicator.
        # Inputs: names - ndarray of output names for imputed columns; input_features -
        # full input names. Outputs: ndarray of str.
        if not self.add_indicator:
            return names

        indicator_names = self.indicator_.get_feature_names_out(input_features)
        return np.concatenate([names, indicator_names])

    def __sklearn_tags__(self):
        # [FUNCTION DOC]
        # Extends parent tags: allow_nan is True only if missing_values is a scalar NaN.
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = is_scalar_nan(self.missing_values)
        return tags


class SimpleImputer(_BaseImputer):
    # [CLASS DOC]
    # Fits one statistic per column (statistics_) and replaces missing entries with it.
    # Columns whose statistic is NaN (all missing at fit) are dropped at transform
    # unless keep_empty_features is True.
    """Univariate imputer for completing missing values with simple strategies.

    Replace missing values using a descriptive statistic (e.g. mean, median, or
    most frequent) along each column, or using a constant value.

    Read more in the :ref:`User Guide <impute>`.

    .. versionadded:: 0.20
       `SimpleImputer` replaces the previous `sklearn.preprocessing.Imputer`
       estimator which is now removed.

    Parameters
    ----------
    missing_values : int, float, str, np.nan, None or pandas.NA, default=np.nan
        The placeholder for the missing values. All occurrences of
        `missing_values` will be imputed. For pandas' dataframes with
        nullable integer dtypes with missing values, `missing_values`
        can be set to either `np.nan` or `pd.NA`.

    strategy : str or Callable, default='mean'
        The imputation strategy.

        - If "mean", then replace missing values using the mean along
          each column. Can only be used with numeric data.
        - If "median", then replace missing values using the median along
          each column. Can only be used with numeric data.
        - If "most_frequent", then replace missing using the most frequent
          value along each column. Can be used with strings or numeric data.
          If there is more than one such value, only the smallest is returned.
        - If "constant", then replace missing values with fill_value. Can be
          used with strings or numeric data.
        - If an instance of Callable, then replace missing values using the
          scalar statistic returned by running the callable over a dense 1d
          array containing non-missing values of each column.

        .. versionadded:: 0.20
           strategy="constant" for fixed value imputation.

        .. versionadded:: 1.5
           strategy=callable for custom value imputation.

    fill_value : str or numerical value, default=None
        When strategy == "constant", `fill_value` is used to replace all
        occurrences of missing_values. For string or object data types,
        `fill_value` must be a string.
        If `None`, `fill_value` will be 0 when imputing numerical
        data and "missing_value" for strings or object data types.

    copy : bool, default=True
        If True, a copy of X will be created. If False, imputation will
        be done in-place whenever possible. Note that, in the following cases,
        a new copy will always be made, even if `copy=False`:

        - If `X` is not an array of floating values;
        - If `X` is encoded as a CSR matrix;
        - If `add_indicator=True`.

    add_indicator : bool, default=False
        If True, a :class:`MissingIndicator` transform will stack onto output
        of the imputer's transform. This allows a predictive estimator
        to account for missingness despite imputation. If a feature has no
        missing values at fit/train time, the feature won't appear on
        the missing indicator even if there are missing values at
        transform/test time.

    keep_empty_features : bool, default=False
        If True, features that consist exclusively of missing values when
        `fit` is called are returned in results when `transform` is called.
        The imputed value is always `0` except when `strategy="constant"`
        in which case `fill_value` will be used instead.

        .. versionadded:: 1.2

    Attributes
    ----------
    statistics_ : array of shape (n_features,)
        The imputation fill value for each feature.
        Computing statistics can result in `np.nan` values.
        During :meth:`transform`, features corresponding to `np.nan`
        statistics will be discarded.

    indicator_ : :class:`~sklearn.impute.MissingIndicator`
        Indicator used to add binary indicators for missing values.
        `None` if `add_indicator=False`.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    IterativeImputer : Multivariate imputer that estimates values to impute for
        each feature with missing values from all the others.
    KNNImputer : Multivariate imputer that estimates missing features using
        nearest samples.

    Notes
    -----
    Columns which only contained missing values at :meth:`fit` are discarded
    upon :meth:`transform` if strategy is not `"constant"`.

    In a prediction context, simple imputation usually performs poorly when
    associated with a weak learner. However, with a powerful learner, it can
    lead to as good or better performance than complex imputation such as
    :class:`~sklearn.impute.IterativeImputer` or :class:`~sklearn.impute.KNNImputer`.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.impute import SimpleImputer
    >>> imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
    >>> imp_mean.fit([[7, 2, 3], [4, np.nan, 6], [10, 5, 9]])
    SimpleImputer()
    >>> X = [[np.nan, 2, 3], [4, np.nan, 6], [10, np.nan, 9]]
    >>> print(imp_mean.transform(X))
    [[ 7.   2.   3. ]
     [ 4.   3.5  6. ]
     [10.   3.5  9. ]]

    For a more detailed example see
    :ref:`sphx_glr_auto_examples_impute_plot_missing_values.py`.
    """

    _parameter_constraints: dict = {
        **_BaseImputer._parameter_constraints,
        "strategy": [
            StrOptions({"mean", "median", "most_frequent", "constant"}),
            callable,
        ],
        "fill_value": "no_validation",  # any object is valid
        "copy": ["boolean"],
    }

    def __init__(
        self,
        *,
        missing_values=np.nan,
        strategy="mean",
        fill_value=None,
        copy=True,
        add_indicator=False,
        keep_empty_features=False,
    ):
        # [FUNCTION DOC]
        # Stores strategy (default "mean"), fill_value (default None), copy (default
        # True) and the base-class arguments; keyword-only, no validation here.
        super().__init__(
            missing_values=missing_values,
            add_indicator=add_indicator,
            keep_empty_features=keep_empty_features,
        )
        self.strategy = strategy
        self.fill_value = fill_value
        self.copy = copy

    def _validate_input(self, X, in_fit):
        # [FUNCTION DOC]
        # HELPER: Validates X for fit (in_fit=True) or transform. Chooses dtype: object
        # when a list containing strings is given with most_frequent/constant, None
        # (keep) for those strategies otherwise, float dtypes for numeric strategies; at
        # transform, reuses the object dtype seen at fit. Accepts CSC sparse input,
        # allows NaN only if missing_values is NaN or pandas NA, copies according to
        # self.copy and forces writeable at transform. For the constant strategy, checks
        # that the fill dtype (fill_value's type at fit, the fit-time dtype at
        # transform) can be cast "same_kind" to X's dtype.
        # Inputs: X - array-like or sparse; in_fit - bool.
        # Outputs: validated ndarray or CSC sparse matrix.
        # Side effects: on fit, sets _fit_dtype and n_features_in_/feature_names_in_
        # (via validate_data reset).
        # Exceptions: ValueError for non-numeric data with numeric strategies ("Cannot
        # use ... strategy with non-numeric data"), mismatched numeric X and non-numeric
        # missing_values, unsupported dtype kinds, sparse X with missing_values == 0,
        # and uncastable fill values.
        # KNOWN ISSUE: at fit with constant strategy and fill_value None,
        # fill_value_dtype is the input dtype, so the cast check is trivially true; also
        # np.can_cast on a Python type (e.g. str) can behave differently across numpy
        # versions.
        if self.strategy in ("most_frequent", "constant"):
            # If input is a list of strings, dtype = object.
            # Otherwise ValueError is raised in SimpleImputer
            # with strategy='most_frequent' or 'constant'
            # because the list is converted to Unicode numpy array
            if isinstance(X, list) and any(
                isinstance(elem, str) for row in X for elem in row
            ):
                dtype = object
            else:
                dtype = None
        else:
            dtype = FLOAT_DTYPES

        if not in_fit and self._fit_dtype.kind == "O":
            # Use object dtype if fitted on object dtypes
            dtype = self._fit_dtype

        if is_pandas_na(self.missing_values) or is_scalar_nan(self.missing_values):
            ensure_all_finite = "allow-nan"
        else:
            ensure_all_finite = True

        try:
            X = validate_data(
                self,
                X,
                reset=in_fit,
                accept_sparse="csc",
                dtype=dtype,
                force_writeable=True if not in_fit else None,
                ensure_all_finite=ensure_all_finite,
                copy=self.copy,
            )
        except ValueError as ve:
            if "could not convert" in str(ve):
                new_ve = ValueError(
                    "Cannot use {} strategy with non-numeric data:\n{}".format(
                        self.strategy, ve
                    )
                )
                raise new_ve from None
            else:
                raise ve

        if in_fit:
            # Use the dtype seen in `fit` for non-`fit` conversion
            self._fit_dtype = X.dtype

        _check_inputs_dtype(X, self.missing_values)
        if X.dtype.kind not in ("i", "u", "f", "O"):
            raise ValueError(
                "SimpleImputer does not support data with dtype "
                "{0}. Please provide either a numeric array (with"
                " a floating point or integer dtype) or "
                "categorical data represented either as an array "
                "with integer dtype or an array of string values "
                "with an object dtype.".format(X.dtype)
            )

        if sp.issparse(X) and self.missing_values == 0:
            # missing_values = 0 not allowed with sparse data as it would
            # force densification
            raise ValueError(
                "Imputation not possible when missing_values "
                "== 0 and input is sparse. Provide a dense "
                "array instead."
            )

        if self.strategy == "constant":
            if in_fit and self.fill_value is not None:
                fill_value_dtype = type(self.fill_value)
                err_msg = (
                    f"fill_value={self.fill_value!r} (of type {fill_value_dtype!r}) "
                    f"cannot be cast to the input data that is {X.dtype!r}. "
                    "If fill_value is a Python scalar, instead pass  a numpy scalar "
                    "(e.g. fill_value=np.uint8(0) if your data is of type np.uint8). "
                    "Make sure that both dtypes are of the same kind."
                )
            elif not in_fit:
                fill_value_dtype = self._fill_dtype
                err_msg = (
                    f"The dtype of the filling value (i.e. {fill_value_dtype!r}) "
                    f"cannot be cast to the input data that is {X.dtype!r}. "
                    "Make sure that the dtypes of the input data are of the same kind "
                    "between fit and transform."
                )
            else:
                # By default, fill_value=None, and the replacement is always
                # compatible with the input data
                fill_value_dtype = X.dtype

            # Make sure we can safely cast fill_value dtype to the input data dtype
            if not np.can_cast(fill_value_dtype, X.dtype, casting="same_kind"):
                raise ValueError(err_msg)

        return X

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None):
        # [FUNCTION DOC]
        # Validates X, resolves the effective fill value (fill_value, or 0 for numeric
        # and "missing_value" for other dtypes when None), records _fill_dtype as X's
        # dtype, and computes statistics_ with _sparse_fit or _dense_fit.
        # Inputs: X - array-like or sparse (n_samples, n_features); y - ignored.
        # Outputs: self.
        # Side effects: sets statistics_, _fill_dtype, _fit_dtype, indicator_,
        # n_features_in_ (and feature_names_in_).
        """Fit the imputer on `X`.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Input data, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        X = self._validate_input(X, in_fit=True)

        # default fill_value is 0 for numerical input and "missing_value"
        # otherwise
        if self.fill_value is None:
            if X.dtype.kind in ("i", "u", "f"):
                fill_value = 0
            else:
                fill_value = "missing_value"
        else:
            fill_value = self.fill_value

        self._fill_dtype = X.dtype

        if sp.issparse(X):
            self.statistics_ = self._sparse_fit(
                X, self.strategy, self.missing_values, fill_value
            )
        else:
            self.statistics_ = self._dense_fit(
                X, self.strategy, self.missing_values, fill_value
            )

        return self

    def _sparse_fit(self, X, strategy, missing_values, fill_value):
        # [FUNCTION DOC]
        # HELPER: Computes per-column statistics for CSC X. Builds the missing mask on
        # stored values; implicit zeros per column = n_samples minus stored count. For
        # constant: every statistic is fill_value, set to NaN for columns whose stored
        # values are all missing unless keep_empty_features (different column slicing
        # depending on scipy version). Otherwise, per column: drop missing stored
        # values, separate explicit zeros and count all zeros (explicit + implicit); if
        # no nonzero values remain and keep_empty_features, statistic is 0; else mean =
        # sum / (nonzeros + zeros) (NaN if both are 0), median via _get_median with the
        # zero count, most_frequent via _most_frequent with 0 repeated n_zeros times, or
        # the callable applied to the column.
        # Inputs: X - CSC matrix; strategy; missing_values; fill_value.
        # Outputs: float ndarray (n_features,).
        # Side effects: fits indicator_ on the sparse missing mask.
        # KNOWN ISSUE: the callable strategy receives only the nonzero non-missing
        # values (zeros are excluded), unlike the dense path which passes all
        # non-missing values. With keep_empty_features, a column of only zeros (no
        # missing) also gets statistic 0 via the "empty" branch (harmless for numeric
        # strategies, but bypasses the strategy). For constant, a column with no stored
        # values (all implicit zeros) has all() of an empty mask equal to True and is
        # wrongly marked NaN/empty.
        """Fit the transformer on sparse data."""
        missing_mask = _get_mask(X, missing_values)
        mask_data = missing_mask.data
        n_implicit_zeros = X.shape[0] - np.diff(X.indptr)

        statistics = np.empty(X.shape[1])

        if strategy == "constant":
            # for constant strategy, self.statistics_ is used to store
            # fill_value in each column, or np.nan for columns to drop
            statistics.fill(fill_value)

            if not self.keep_empty_features:
                if SCIPY_VERSION_BELOW_1_12:
                    for i in range(missing_mask.shape[1]):
                        if all(missing_mask[:, [i]].data):
                            statistics[i] = np.nan
                else:
                    for i in range(missing_mask.shape[1]):
                        if all(missing_mask[:, i].data):
                            statistics[i] = np.nan

        else:
            for i in range(X.shape[1]):
                column = X.data[X.indptr[i] : X.indptr[i + 1]]
                mask_column = mask_data[X.indptr[i] : X.indptr[i + 1]]
                column = column[~mask_column]

                # combine explicit and implicit zeros
                mask_zeros = _get_mask(column, 0)
                column = column[~mask_zeros]
                n_explicit_zeros = mask_zeros.sum()
                n_zeros = n_implicit_zeros[i] + n_explicit_zeros

                if len(column) == 0 and self.keep_empty_features:
                    # in case we want to keep columns with only missing values.
                    statistics[i] = 0
                else:
                    if strategy == "mean":
                        s = column.size + n_zeros
                        statistics[i] = np.nan if s == 0 else column.sum() / s

                    elif strategy == "median":
                        statistics[i] = _get_median(column, n_zeros)

                    elif strategy == "most_frequent":
                        statistics[i] = _most_frequent(column, 0, n_zeros)

                    elif isinstance(strategy, Callable):
                        statistics[i] = self.strategy(column)

        super()._fit_indicator(missing_mask)

        return statistics

    def _dense_fit(self, X, strategy, missing_values, fill_value):
        # [FUNCTION DOC]
        # HELPER: Computes per-column statistics for dense X using a masked array of the
        # missing mask. mean/median: masked column mean/median, with fully-masked
        # columns set to 0 if keep_empty_features else NaN. most_frequent: for each
        # column, mode of non-missing values via _most_frequent (object array for object
        # dtype), or 0 for empty columns when keep_empty_features. constant: object
        # array filled with fill_value, NaN for fully missing columns unless
        # keep_empty_features. callable: the callable applied to each column's
        # non-missing values (float result array).
        # Inputs: X - ndarray; strategy; missing_values; fill_value.
        # Outputs: ndarray (n_features,).
        # Side effects: fits indicator_ on the dense missing mask.
        # KNOWN ISSUE: for the callable strategy, a fully missing column is passed as an
        # empty array and keep_empty_features is not applied.
        """Fit the transformer on dense data."""
        missing_mask = _get_mask(X, missing_values)
        masked_X = ma.masked_array(X, mask=missing_mask)

        super()._fit_indicator(missing_mask)

        # Mean
        if strategy == "mean":
            mean_masked = np.ma.mean(masked_X, axis=0)
            # Avoid the warning "Warning: converting a masked element to nan."
            mean = np.ma.getdata(mean_masked)
            mean[np.ma.getmask(mean_masked)] = 0 if self.keep_empty_features else np.nan

            return mean

        # Median
        elif strategy == "median":
            median_masked = np.ma.median(masked_X, axis=0)
            # Avoid the warning "Warning: converting a masked element to nan."
            median = np.ma.getdata(median_masked)
            median[np.ma.getmaskarray(median_masked)] = (
                0 if self.keep_empty_features else np.nan
            )

            return median

        # Most frequent
        elif strategy == "most_frequent":
            # Avoid use of scipy.stats.mstats.mode due to the required
            # additional overhead and slow benchmarking performance.
            # See Issue 14325 and PR 14399 for full discussion.

            # To be able access the elements by columns
            X = X.transpose()
            mask = missing_mask.transpose()

            if X.dtype.kind == "O":
                most_frequent = np.empty(X.shape[0], dtype=object)
            else:
                most_frequent = np.empty(X.shape[0])

            for i, (row, row_mask) in enumerate(zip(X[:], mask[:])):
                row_mask = np.logical_not(row_mask).astype(bool)
                row = row[row_mask]
                if len(row) == 0 and self.keep_empty_features:
                    most_frequent[i] = 0
                else:
                    most_frequent[i] = _most_frequent(row, np.nan, 0)

            return most_frequent

        # Constant
        elif strategy == "constant":
            # for constant strategy, self.statistics_ is used to store
            # fill_value in each column, or np.nan for columns to drop
            statistics = np.full(X.shape[1], fill_value, dtype=np.object_)

            if not self.keep_empty_features:
                for i in range(missing_mask.shape[1]):
                    if missing_mask[:, i].all():
                        statistics[i] = np.nan

            return statistics

        # Custom
        elif isinstance(strategy, Callable):
            statistics = np.empty(masked_X.shape[1])
            for i in range(masked_X.shape[1]):
                statistics[i] = self.strategy(masked_X[:, i].compressed())
            return statistics

    def transform(self, X):
        # [FUNCTION DOC]
        # Validates X with fit-time settings, computes the missing mask, and drops
        # columns with NaN statistics (with a UserWarning naming them, using feature
        # names when available) unless keep_empty_features. Statistics are cast to the
        # fit dtype. Sparse: replaces missing stored values with their column statistic
        # (column index derived from indptr), recomputing the mask if columns were
        # dropped. Dense: fills missing positions column by column with np.repeat of
        # statistics. Finally appends the indicator mask computed on the original mask.
        # Inputs: X - array-like or sparse (n_samples, n_features).
        # Outputs: imputed ndarray or sparse matrix (n_samples, n_features_out).
        # Exceptions: NotFittedError; ValueError on feature-count mismatch or sparse X
        # with missing_values == 0; validation errors from _validate_input.
        # Side effects: may modify X in place when copy=False and no conversion was
        # needed.
        # KNOWN ISSUE: the sparse missing_values == 0 check is unreachable because
        # _validate_input already raises for that case.
        """Impute all missing values in `X`.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The input data to complete.

        Returns
        -------
        X_imputed : {ndarray, sparse matrix} of shape \
                (n_samples, n_features_out)
            `X` with imputed values.
        """
        check_is_fitted(self)

        X = self._validate_input(X, in_fit=False)
        statistics = self.statistics_

        if X.shape[1] != statistics.shape[0]:
            raise ValueError(
                "X has %d features per sample, expected %d"
                % (X.shape[1], self.statistics_.shape[0])
            )

        # compute mask before eliminating invalid features
        missing_mask = _get_mask(X, self.missing_values)

        # Decide whether to keep missing features
        if self.keep_empty_features:
            valid_statistics = statistics.astype(self._fill_dtype, copy=False)
            valid_statistics_indexes = None
        else:
            # same as np.isnan but also works for object dtypes
            invalid_mask = _get_mask(statistics, np.nan)
            valid_mask = np.logical_not(invalid_mask)
            valid_statistics = statistics[valid_mask].astype(
                self._fill_dtype, copy=False
            )
            valid_statistics_indexes = np.flatnonzero(valid_mask)

            if invalid_mask.any():
                invalid_features = np.arange(X.shape[1])[invalid_mask]
                # use feature names warning if features are provided
                if hasattr(self, "feature_names_in_"):
                    invalid_features = self.feature_names_in_[invalid_features]
                warnings.warn(
                    "Skipping features without any observed values:"
                    f" {invalid_features}. At least one non-missing value is needed"
                    f" for imputation with strategy='{self.strategy}'."
                )
                X = X[:, valid_statistics_indexes]

        # Do actual imputation
        if sp.issparse(X):
            if self.missing_values == 0:
                raise ValueError(
                    "Imputation not possible when missing_values "
                    "== 0 and input is sparse. Provide a dense "
                    "array instead."
                )
            else:
                # if no invalid statistics are found, use the mask computed
                # before, else recompute mask
                if valid_statistics_indexes is None:
                    mask = missing_mask.data
                else:
                    mask = _get_mask(X.data, self.missing_values)
                indexes = np.repeat(
                    np.arange(len(X.indptr) - 1, dtype=int), np.diff(X.indptr)
                )[mask]

                X.data[mask] = valid_statistics[indexes]
        else:
            # use mask computed before eliminating invalid mask
            if valid_statistics_indexes is None:
                mask_valid_features = missing_mask
            else:
                mask_valid_features = missing_mask[:, valid_statistics_indexes]
            n_missing = np.sum(mask_valid_features, axis=0)
            values = np.repeat(valid_statistics, n_missing)
            coordinates = np.where(mask_valid_features.transpose())[::-1]

            X[coordinates] = values

        X_indicator = super()._transform_indicator(missing_mask)

        return super()._concatenate_indicator(X, X_indicator)

    def inverse_transform(self, X):
        # [FUNCTION DOC]
        # Reverses transform using the appended indicator columns: splits X into imputed
        # part and boolean mask, rebuilds an all-zero (n_samples, n_features_original)
        # float array, places the mask into the indicator_.features_ columns, then
        # copies imputed columns into original columns in order, skipping columns whose
        # mask column is all True (assumed to be features dropped at transform), and
        # finally writes missing_values into masked cells.
        # Inputs: X - ndarray (n_samples, n_kept_features + n_indicator_features).
        # Outputs: float ndarray (n_samples, n_features_original).
        # Exceptions: NotFittedError; ValueError if add_indicator is False.
        # KNOWN ISSUE: a kept feature whose values are missing in every row of X is
        # treated as a dropped feature and skipped, misaligning the following columns; X
        # is not validated and object/string data is coerced to float.
        """Convert the data back to the original representation.

        Inverts the `transform` operation performed on an array.
        This operation can only be performed after :class:`SimpleImputer` is
        instantiated with `add_indicator=True`.

        Note that `inverse_transform` can only invert the transform in
        features that have binary indicators for missing values. If a feature
        has no missing values at `fit` time, the feature won't have a binary
        indicator, and the imputation done at `transform` time won't be
        inverted.

        .. versionadded:: 0.24

        Parameters
        ----------
        X : array-like of shape \
                (n_samples, n_features + n_features_missing_indicator)
            The imputed data to be reverted to original data. It has to be
            an augmented array of imputed data and the missing indicator mask.

        Returns
        -------
        X_original : ndarray of shape (n_samples, n_features)
            The original `X` with missing values as it was prior
            to imputation.
        """
        check_is_fitted(self)

        if not self.add_indicator:
            raise ValueError(
                "'inverse_transform' works only when "
                "'SimpleImputer' is instantiated with "
                "'add_indicator=True'. "
                f"Got 'add_indicator={self.add_indicator}' "
                "instead."
            )

        n_features_missing = len(self.indicator_.features_)
        non_empty_feature_count = X.shape[1] - n_features_missing
        array_imputed = X[:, :non_empty_feature_count].copy()
        missing_mask = X[:, non_empty_feature_count:].astype(bool)

        n_features_original = len(self.statistics_)
        shape_original = (X.shape[0], n_features_original)
        X_original = np.zeros(shape_original)
        X_original[:, self.indicator_.features_] = missing_mask
        full_mask = X_original.astype(bool)

        imputed_idx, original_idx = 0, 0
        while imputed_idx < len(array_imputed.T):
            if not np.all(X_original[:, original_idx]):
                X_original[:, original_idx] = array_imputed.T[imputed_idx]
                imputed_idx += 1
                original_idx += 1
            else:
                original_idx += 1

        X_original[full_mask] = self.missing_values
        return X_original

    def __sklearn_tags__(self):
        # [FUNCTION DOC]
        # Extends tags: sparse input supported; allow_nan when missing_values is NaN or
        # pandas NA.
        tags = super().__sklearn_tags__()
        tags.input_tags.sparse = True
        tags.input_tags.allow_nan = is_pandas_na(self.missing_values) or is_scalar_nan(
            self.missing_values
        )
        return tags

    def get_feature_names_out(self, input_features=None):
        # [FUNCTION DOC]
        # Returns input feature names for columns with non-NaN statistics (columns kept
        # at transform), followed by indicator names if add_indicator.
        # Inputs: input_features - optional array-like of str.
        # Outputs: ndarray of str. Exceptions: NotFittedError; ValueError on mismatched
        # names.
        # KNOWN ISSUE: with keep_empty_features=True, transform keeps NaN-statistic
        # columns but they are omitted here (only relevant when statistics are NaN, e.g.
        # callable strategy).
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
        non_missing_mask = np.logical_not(_get_mask(self.statistics_, np.nan))
        names = input_features[non_missing_mask]
        return self._concatenate_indicator_feature_names_out(names, input_features)


class MissingIndicator(TransformerMixin, BaseEstimator):
    # [CLASS DOC]
    # Transformer outputting a boolean mask of missing entries for the features selected
    # at fit (features_: either all features or those with missing values at fit).
    """Binary indicators for missing values.

    Note that this component typically should not be used in a vanilla
    :class:`~sklearn.pipeline.Pipeline` consisting of transformers and a
    classifier, but rather could be added using a
    :class:`~sklearn.pipeline.FeatureUnion` or
    :class:`~sklearn.compose.ColumnTransformer`.

    Read more in the :ref:`User Guide <impute>`.

    .. versionadded:: 0.20

    Parameters
    ----------
    missing_values : int, float, str, np.nan or None, default=np.nan
        The placeholder for the missing values. All occurrences of
        `missing_values` will be imputed. For pandas' dataframes with
        nullable integer dtypes with missing values, `missing_values`
        should be set to `np.nan`, since `pd.NA` will be converted to `np.nan`.

    features : {'missing-only', 'all'}, default='missing-only'
        Whether the imputer mask should represent all or a subset of
        features.

        - If `'missing-only'` (default), the imputer mask will only represent
          features containing missing values during fit time.
        - If `'all'`, the imputer mask will represent all features.

    sparse : bool or 'auto', default='auto'
        Whether the imputer mask format should be sparse or dense.

        - If `'auto'` (default), the imputer mask will be of same type as
          input.
        - If `True`, the imputer mask will be a sparse matrix.
        - If `False`, the imputer mask will be a numpy array.

    error_on_new : bool, default=True
        If `True`, :meth:`transform` will raise an error when there are
        features with missing values that have no missing values in
        :meth:`fit`. This is applicable only when `features='missing-only'`.

    Attributes
    ----------
    features_ : ndarray of shape (n_missing_features,) or (n_features,)
        The features indices which will be returned when calling
        :meth:`transform`. They are computed during :meth:`fit`. If
        `features='all'`, `features_` is equal to `range(n_features)`.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    SimpleImputer : Univariate imputation of missing values.
    IterativeImputer : Multivariate imputation of missing values.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.impute import MissingIndicator
    >>> X1 = np.array([[np.nan, 1, 3],
    ...                [4, 0, np.nan],
    ...                [8, 1, 0]])
    >>> X2 = np.array([[5, 1, np.nan],
    ...                [np.nan, 2, 3],
    ...                [2, 4, 0]])
    >>> indicator = MissingIndicator()
    >>> indicator.fit(X1)
    MissingIndicator()
    >>> X2_tr = indicator.transform(X2)
    >>> X2_tr
    array([[False,  True],
           [ True, False],
           [False, False]])
    """

    _parameter_constraints: dict = {
        "missing_values": [MissingValues()],
        "features": [StrOptions({"missing-only", "all"})],
        "sparse": ["boolean", StrOptions({"auto"})],
        "error_on_new": ["boolean"],
    }

    def __init__(
        self,
        *,
        missing_values=np.nan,
        features="missing-only",
        sparse="auto",
        error_on_new=True,
    ):
        # [FUNCTION DOC]
        # Stores missing_values (default np.nan), features ("missing-only"), sparse
        # ("auto") and error_on_new (True); keyword-only.
        self.missing_values = missing_values
        self.features = features
        self.sparse = sparse
        self.error_on_new = error_on_new

    def _get_missing_features_info(self, X):
        # [FUNCTION DOC]
        # HELPER: Computes the missing mask (or uses X itself when _precomputed) and the
        # feature indices to report. Sparse: removes explicit zeros from the mask,
        # counts missing per column for "missing-only", converts to dense if sparse is
        # False or CSR to CSC. Dense: counts missing per column and converts to CSC
        # sparse if sparse is True.
        # Inputs: X - validated ndarray or sparse matrix (or boolean mask if
        # precomputed).
        # Outputs: tuple (imputer_mask, features_indices) - mask (n_samples, n_features)
        # and indices of all features or those with at least one missing value.
        # Side effects: when precomputed and sparse, eliminate_zeros mutates the
        # caller's mask.
        # KNOWN ISSUE: the dense branch recomputes imputer_mask identically to the code
        # before the branch (redundant work), and the comment "count number of True
        # values in each row" is wrong (it counts per column).
        """Compute the imputer mask and the indices of the features
        containing missing values.

        Parameters
        ----------
        X : {ndarray, sparse matrix} of shape (n_samples, n_features)
            The input data with missing values. Note that `X` has been
            checked in :meth:`fit` and :meth:`transform` before to call this
            function.

        Returns
        -------
        imputer_mask : {ndarray, sparse matrix} of shape \
        (n_samples, n_features)
            The imputer mask of the original data.

        features_with_missing : ndarray of shape (n_features_with_missing)
            The features containing missing values.
        """
        if not self._precomputed:
            imputer_mask = _get_mask(X, self.missing_values)
        else:
            imputer_mask = X

        if sp.issparse(X):
            imputer_mask.eliminate_zeros()

            if self.features == "missing-only":
                # count number of True values in each row.
                n_missing = imputer_mask.sum(axis=0)

            if self.sparse is False:
                imputer_mask = imputer_mask.toarray()
            elif imputer_mask.format == "csr":
                imputer_mask = imputer_mask.tocsc()
        else:
            if not self._precomputed:
                imputer_mask = _get_mask(X, self.missing_values)
            else:
                imputer_mask = X

            if self.features == "missing-only":
                n_missing = imputer_mask.sum(axis=0)

            if self.sparse is True:
                imputer_mask = _align_api_if_sparse(sp.csc_array(imputer_mask))

        if self.features == "all":
            features_indices = np.arange(X.shape[1])
        else:
            features_indices = np.flatnonzero(n_missing)

        return imputer_mask, features_indices

    def _validate_input(self, X, in_fit):
        # [FUNCTION DOC]
        # HELPER: Validates X (CSC/CSR sparse allowed, any dtype kept, NaN allowed only
        # when missing_values is NaN), then checks dtype compatibility.
        # Inputs: X; in_fit - bool (resets n_features_in_ when True).
        # Outputs: validated X.
        # Exceptions: ValueError for numeric X with non-numeric missing_values,
        # unsupported dtype kinds, or sparse X with missing_values == 0.
        if not is_scalar_nan(self.missing_values):
            ensure_all_finite = True
        else:
            ensure_all_finite = "allow-nan"
        X = validate_data(
            self,
            X,
            reset=in_fit,
            accept_sparse=("csc", "csr"),
            dtype=None,
            ensure_all_finite=ensure_all_finite,
        )
        _check_inputs_dtype(X, self.missing_values)
        if X.dtype.kind not in ("i", "u", "f", "O"):
            raise ValueError(
                "MissingIndicator does not support data with "
                "dtype {0}. Please provide either a numeric array"
                " (with a floating point or integer dtype) or "
                "categorical data represented either as an array "
                "with integer dtype or an array of string values "
                "with an object dtype.".format(X.dtype)
            )

        if sp.issparse(X) and self.missing_values == 0:
            # missing_values = 0 not allowed with sparse data as it would
            # force densification
            raise ValueError(
                "Sparse input with missing_values=0 is "
                "not supported. Provide a dense "
                "array instead."
            )

        return X

    def _fit(self, X, y=None, precomputed=False):
        # [FUNCTION DOC]
        # HELPER: Shared fit logic. If precomputed, X must be a boolean mask (only
        # feature count is checked); otherwise X is validated. Records _precomputed,
        # _n_features and features_.
        # Inputs: X; y - ignored; precomputed - bool, default False.
        # Outputs: the full imputer mask (before column selection).
        # Exceptions: ValueError if precomputed is True and X is not boolean; validation
        # errors.
        # Side effects: sets _precomputed, n_features_in_ (and feature_names_in_ when
        # validated), _n_features, features_.
        """Fit the transformer on `X`.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Input data, where `n_samples` is the number of samples and
            `n_features` is the number of features.
            If `precomputed=True`, then `X` is a mask of the input data.

        precomputed : bool
            Whether the input data is a mask.

        Returns
        -------
        imputer_mask : {ndarray, sparse matrix} of shape (n_samples, \
        n_features)
            The imputer mask of the original data.
        """
        if precomputed:
            if not (hasattr(X, "dtype") and X.dtype.kind == "b"):
                raise ValueError("precomputed is True but the input data is not a mask")
            self._precomputed = True
        else:
            self._precomputed = False

        # Need not validate X again as it would have already been validated
        # in the Imputer calling MissingIndicator
        if not self._precomputed:
            X = self._validate_input(X, in_fit=True)
        else:
            # only create `n_features_in_` in the precomputed case
            _check_n_features(self, X, reset=True)

        self._n_features = X.shape[1]

        missing_features_info = self._get_missing_features_info(X)
        self.features_ = missing_features_info[1]

        return missing_features_info[0]

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None):
        # [FUNCTION DOC]
        # Fits the indicator via _fit and returns self. Inputs: X; y ignored.
        """Fit the transformer on `X`.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Input data, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        self._fit(X, y)

        return self

    def transform(self, X):
        # [FUNCTION DOC]
        # Computes the mask for X (validating unless precomputed). For "missing-only",
        # raises if error_on_new and features with missing values at transform had none
        # at fit, and restricts the mask to features_ columns.
        # Inputs: X (n_samples, n_features). Outputs: boolean ndarray or sparse matrix.
        # Exceptions: NotFittedError; ValueError for new missing features or non-boolean
        # input when precomputed.
        """Generate missing values indicator for `X`.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input data to complete.

        Returns
        -------
        Xt : {ndarray, sparse matrix} of shape (n_samples, n_features) \
        or (n_samples, n_features_with_missing)
            The missing indicator for input data. The data type of `Xt`
            will be boolean.
        """
        check_is_fitted(self)

        # Need not validate X again as it would have already been validated
        # in the Imputer calling MissingIndicator
        if not self._precomputed:
            X = self._validate_input(X, in_fit=False)
        else:
            if not (hasattr(X, "dtype") and X.dtype.kind == "b"):
                raise ValueError("precomputed is True but the input data is not a mask")

        imputer_mask, features = self._get_missing_features_info(X)

        if self.features == "missing-only":
            features_diff_fit_trans = np.setdiff1d(features, self.features_)
            if self.error_on_new and features_diff_fit_trans.size > 0:
                raise ValueError(
                    "The features {} have missing values "
                    "in transform but have no missing values "
                    "in fit.".format(features_diff_fit_trans)
                )

            if self.features_.size < self._n_features:
                imputer_mask = imputer_mask[:, self.features_]

        return imputer_mask

    @_fit_context(prefer_skip_nested_validation=True)
    def fit_transform(self, X, y=None):
        # [FUNCTION DOC]
        # Fits and returns the mask in one pass, restricted to features_ when fewer than
        # all features are selected. Inputs: X; y ignored. Outputs: boolean mask.
        # Side effects: same as fit.
        """Generate missing values indicator for `X`.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input data to complete.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        Xt : {ndarray, sparse matrix} of shape (n_samples, n_features) \
        or (n_samples, n_features_with_missing)
            The missing indicator for input data. The data type of `Xt`
            will be boolean.
        """
        imputer_mask = self._fit(X, y)

        if self.features_.size < self._n_features:
            imputer_mask = imputer_mask[:, self.features_]

        return imputer_mask

    def get_feature_names_out(self, input_features=None):
        # [FUNCTION DOC]
        # Returns "missingindicator_<name>" for each selected input feature in
        # features_.
        # Inputs: input_features - optional array-like of str.
        # Outputs: object ndarray of str. Exceptions: NotFittedError; ValueError on
        # mismatch.
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
        prefix = self.__class__.__name__.lower()
        return np.asarray(
            [
                f"{prefix}_{feature_name}"
                for feature_name in input_features[self.features_]
            ],
            dtype=object,
        )

    def __sklearn_tags__(self):
        # [FUNCTION DOC]
        # Extends tags: allows NaN, string and sparse input; preserves_dtype is empty
        # (output is boolean).
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        tags.input_tags.string = True
        tags.input_tags.sparse = True
        tags.transformer_tags.preserves_dtype = []
        return tags
