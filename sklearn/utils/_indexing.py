# [MODULE DOC]
# Indexing utilities that select rows/items/columns uniformly across NumPy arrays,
# Array API arrays, scipy sparse matrices, pandas and other narwhals-compatible
# dataframes/series, pyarrow arrays and Python lists, plus resample/shuffle built on
# top of them. No import-time side effects beyond importing narwhals.
# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import numbers
import sys
import warnings
from itertools import compress, islice

import narwhals.stable.v2 as nw
import numpy as np
from scipy.sparse import issparse

from sklearn.utils._array_api import (
    _is_numpy_namespace,
    get_namespace,
    get_namespace_and_device,
    move_to,
)
from sklearn.utils._dataframe import is_pyarrow_data
from sklearn.utils._param_validation import Interval, validate_params
from sklearn.utils.extmath import _approximate_mode
from sklearn.utils.fixes import SCIPY_VERSION_BELOW_1_12
from sklearn.utils.validation import (
    _check_sample_weight,
    _is_arraylike_not_scalar,
    check_array,
    check_consistent_length,
    check_random_state,
)


def _array_indexing(array, key, key_dtype, axis):
    # [FUNCTION DOC]
    # HELPER: Indexes an array-like with a "shape" along axis 0 or 1.
    # Steps: for Array API (non-NumPy namespaces handled through the compat layer) the
    # key is moved to the array's namespace/device (ints and slices pass through);
    # integer-dtype keys use take along axis, boolean keys are converted to integer
    # positions via an arange then take. Otherwise (NumPy, sparse, or key without
    # dtype), for sparse arrays boolean keys become ndarrays and, on SciPy < 1.12, an
    # integer scalar key is wrapped into a list; tuple keys become lists; finally
    # plain getitem is used on rows (axis 0, with ellipsis) or columns (axis 1).
    # Inputs: array - ndarray, Array API array or sparse matrix; key - int, slice,
    # array-like of int/bool; key_dtype - result of _determine_key_type; axis - 0 or 1.
    # Outputs: indexed subset of the same container type.
    # Side effects: none.
    """Index an array or scipy.sparse consistently across NumPy version."""
    xp, is_array_api, device_ = get_namespace_and_device(array)
    if is_array_api:
        if hasattr(key, "shape"):
            key = move_to(key, xp=xp, device=device_)
        elif isinstance(key, (int, slice)):
            # Passthrough for valid __getitem__ inputs as noted in the array
            # API spec.
            pass
        else:
            key = xp.asarray(key, device=device_)

        if hasattr(key, "dtype"):
            if xp.isdtype(key.dtype, "integral"):
                return xp.take(array, key, axis=axis)
            elif xp.isdtype(key.dtype, "bool"):
                # Array API does not support boolean indexing for n-dim arrays
                # yet hence the need to turn to equivalent integer indexing.
                indices = xp.arange(array.shape[axis], device=device_)
                return xp.take(array, indices[key], axis=axis)

    if issparse(array):
        if key_dtype == "bool":
            key = np.asarray(key)
        elif SCIPY_VERSION_BELOW_1_12:
            if isinstance(key, numbers.Integral):
                key = [key]
    if isinstance(key, tuple):
        key = list(key)
    return array[key, ...] if axis == 0 else array[:, key]


def _narwhals_indexing(X, key, key_dtype, axis):
    # [FUNCTION DOC]
    # HELPER: Indexes a dataframe/series supported by narwhals (e.g. polars, pyarrow).
    # Keys that are not list, slice or None are converted to Python lists. For axis 1:
    # boolean keys select the columns whose flag is true; other keys use positional
    # column getitem; native object is returned. For axis 0: boolean keys filter rows,
    # others use row getitem. With a scalar key, a series input returns the scalar
    # item directly, a dataframe input returns a 1d NumPy array of the row's values;
    # otherwise the native subset is returned.
    # Inputs: X - native dataframe or series; key, key_dtype, axis as in
    # _array_indexing.
    # Outputs: native dataframe/series, scalar, or ndarray (scalar row of a frame).
    # Side effects: none.
    """Index a narwhals dataframe or series."""
    X = nw.from_native(X, allow_series=True)
    if not (isinstance(key, (list, slice)) or key is None):
        # Note that at least tuples should be converted to either list or ndarray as
        # tuples in __getitem__ are special: x[(1, 2)] is equal to x[1, 2].
        # Also, not all backends of narwhals support ndarray, but all support lists.
        key = np.asarray(key).tolist()

    if axis == 1:
        if key_dtype == "bool":
            subset = X.select(col for (col, select) in zip(X.columns, key) if select)
            return subset.to_native()
        return X[:, key].to_native()

    # From here on axis == 0:
    if key_dtype == "bool":
        X_indexed = X.filter(key)
    else:
        X_indexed = X[key]

    if np.isscalar(key):
        if len(X.shape) <= 1:
            return X_indexed
        # TODO: `X_indexed` is a DataFrame with a single row; we return a Series to be
        # consistent with pandas. Narwhals would return a dataframe which is
        # advantageous if the columns have different dtypes.
        return np.array([col.item(0) for col in X_indexed.iter_columns()])
    return X_indexed.to_native()


def _pandas_indexing(X, key, key_dtype, axis):
    # [FUNCTION DOC]
    # HELPER: Indexes a pandas DataFrame/Series. Array-like keys become ndarrays.
    # Integer array keys use take along axis (to return a real copy); scalars, slices
    # and non-integer keys use iloc for "int" keys and loc otherwise (str/bool), on
    # columns if axis is 1 else rows.
    # Inputs: X - pandas object; key, key_dtype, axis as in _array_indexing.
    # Outputs: pandas object or scalar value.
    # Side effects: none.
    """Index a pandas dataframe or a series."""
    if _is_arraylike_not_scalar(key):
        key = np.asarray(key)

    if key_dtype == "int" and not (isinstance(key, slice) or np.isscalar(key)):
        # using take() instead of iloc[] ensures the return value is a "proper"
        # copy that will not raise SettingWithCopyWarning
        return X.take(key, axis=axis)
    else:
        # check whether we should index with loc or iloc
        indexer = X.iloc if key_dtype == "int" else X.loc
        return indexer[:, key] if axis else indexer[key]


def _list_indexing(X, key, key_dtype):
    # [FUNCTION DOC]
    # HELPER: Indexes a Python list along its only axis. Scalars/slices use getitem;
    # boolean keys keep elements where the mask is true; integer array-likes gather
    # elements in order.
    # Inputs: X - list; key - scalar, slice or array-like; key_dtype - "bool"/"int"/None.
    # Outputs: element or new list.
    # Side effects: none.
    """Index a Python list."""
    if np.isscalar(key) or isinstance(key, slice):
        # key is a slice or a scalar
        return X[key]
    if key_dtype == "bool":
        # key is a boolean array-like
        return list(compress(X, key))
    # key is an integer array-like of key
    return [X[idx] for idx in key]


def _determine_key_type(key, accept_slice=True):
    # [FUNCTION DOC]
    # HELPER: Infers whether a key is integer-, string- or boolean-based.
    # Rules: None -> None. Python int/str/bool or np.bool_ scalars map by exact type
    # (subclasses not listed raise ValueError). Slices: if not accept_slice raise
    # TypeError; full slice -> None; otherwise recursively type start and stop, which
    # must agree when both set, and return whichever is set. Lists/tuples: type each
    # unique element; empty -> None; mixed types -> ValueError. Objects with dtype: for
    # non-NumPy Array API arrays bool -> "bool", integral -> "int", else ValueError; for
    # NumPy-like dtypes map kind i/u -> int, b -> bool, O/U/S -> str, else ValueError.
    # Anything else -> ValueError.
    # Inputs: key - scalar, slice, list/tuple or array; accept_slice - bool, default True.
    # Outputs: "int", "str", "bool" or None.
    # Exceptions: ValueError (unsupported/mixed key), TypeError (slice not accepted).
    # KNOWN ISSUE: docstring says accept_slice controls "whether to raise"; actually an
    # error is raised when accept_slice is False. Object-dtype arrays are always
    # reported as "str" regardless of their content.
    """Determine the data type of key.

    Parameters
    ----------
    key : scalar, slice or array-like
        The key from which we want to infer the data type.

    accept_slice : bool, default=True
        Whether or not to raise an error if the key is a slice.

    Returns
    -------
    dtype : {'int', 'str', 'bool', None}
        Returns the data type of key.
    """
    err_msg = (
        "No valid specification of the columns. Only a scalar, list or "
        "slice of all integers or all strings, or boolean mask is "
        "allowed"
    )

    dtype_to_str = {int: "int", str: "str", bool: "bool", np.bool_: "bool"}
    array_dtype_to_str = {
        "i": "int",
        "u": "int",
        "b": "bool",
        "O": "str",
        "U": "str",
        "S": "str",
    }

    if key is None:
        return None
    if isinstance(key, tuple(dtype_to_str.keys())):
        try:
            return dtype_to_str[type(key)]
        except KeyError:
            raise ValueError(err_msg)
    if isinstance(key, slice):
        if not accept_slice:
            raise TypeError(
                "Only array-like or scalar are supported. A Python slice was given."
            )
        if key.start is None and key.stop is None:
            return None
        key_start_type = _determine_key_type(key.start)
        key_stop_type = _determine_key_type(key.stop)
        if key_start_type is not None and key_stop_type is not None:
            if key_start_type != key_stop_type:
                raise ValueError(err_msg)
        if key_start_type is not None:
            return key_start_type
        return key_stop_type
    if isinstance(key, (list, tuple)):
        unique_key = set(key)
        key_type = {_determine_key_type(elt) for elt in unique_key}
        if not key_type:
            return None
        if len(key_type) != 1:
            raise ValueError(err_msg)
        return key_type.pop()
    if hasattr(key, "dtype"):
        xp, is_array_api = get_namespace(key)
        # NumPy arrays are special-cased in their own branch because the Array API
        # cannot handle object/string-based dtypes that are often used to index
        # columns of dataframes by names.
        if is_array_api and not _is_numpy_namespace(xp):
            if xp.isdtype(key.dtype, "bool"):
                return "bool"
            elif xp.isdtype(key.dtype, "integral"):
                return "int"
            else:
                raise ValueError(err_msg)
        else:
            try:
                return array_dtype_to_str[key.dtype.kind]
            except KeyError:
                raise ValueError(err_msg)
    raise ValueError(err_msg)


def _safe_indexing(X, indices, *, axis=0):
    # [FUNCTION DOC]
    # Dispatches row/column selection to the right backend helper.
    # Steps: None indices returns X unchanged. Validate axis in (0, 1); infer the key
    # type; reject string keys on axis 0, axis 1 on lists, axis 1 on non-2D inputs, and
    # string keys on axis 1 for non-dataframes. Then dispatch: objects with iloc ->
    # _pandas_indexing; narwhals-compatible frames/series -> _narwhals_indexing;
    # pyarrow arrays are wrapped into a ChunkedArray, indexed via narwhals, and a
    # ChunkedArray result is combined back into a single array; objects with shape ->
    # _array_indexing; otherwise -> _list_indexing.
    # Inputs: X - data container; indices - see docstring; axis - 0 or 1 keyword-only.
    # Outputs: subset of X (container type preserved where possible).
    # Exceptions: ValueError for invalid axis/key/X combinations, propagated from
    # _determine_key_type and backend indexing errors.
    # Side effects: none.
    """Return rows, items or columns of X using indices.

    .. warning::

        This utility is documented, but **private**. This means that
        backward compatibility might be broken without any deprecation
        cycle.

    Parameters
    ----------
    X : array-like, sparse-matrix, list, pandas.DataFrame, pandas.Series
        Data from which to sample rows, items or columns. `list` are only
        supported when `axis=0`.
    indices : bool, int, str, slice, array-like
        - If `axis=0`, boolean and integer array-like, integer slice,
          and scalar integer are supported.
        - If `axis=1`:
            - to select a single column, `indices` can be of `int` type for
              all `X` types and `str` only for dataframe. The selected subset
              will be 1D, unless `X` is a sparse matrix in which case it will
              be 2D.
            - to select multiples columns, `indices` can be one of the
              following: `list`, `array`, `slice`. The type used in
              these containers can be one of the following: `int`, 'bool' and
              `str`. However, `str` is only supported when `X` is a dataframe.
              The selected subset will be 2D.
    axis : int, default=0
        The axis along which `X` will be subsampled. `axis=0` will select
        rows while `axis=1` will select columns.

    Returns
    -------
    subset
        Subset of X on axis 0 or 1.

    Notes
    -----
    CSR, CSC, and LIL sparse matrices are supported. COO sparse matrices are
    not supported.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.utils import _safe_indexing
    >>> data = np.array([[1, 2], [3, 4], [5, 6]])
    >>> _safe_indexing(data, 0, axis=0)  # select the first row
    array([1, 2])
    >>> _safe_indexing(data, 0, axis=1)  # select the first column
    array([1, 3, 5])
    """
    if indices is None:
        return X

    if axis not in (0, 1):
        raise ValueError(
            "'axis' should be either 0 (to index rows) or 1 (to index "
            " column). Got {} instead.".format(axis)
        )

    indices_dtype = _determine_key_type(indices)

    if axis == 0 and indices_dtype == "str":
        raise ValueError(
            f"String indexing (indices={indices}) is not supported with 'axis=0'. "
            "Did you mean to use axis=1 for column selection?"
        )

    if axis == 1 and isinstance(X, list):
        raise ValueError("axis=1 is not supported for lists")

    if axis == 1 and (ndim := len(getattr(X, "shape", [0]))) != 2:
        raise ValueError(
            "'X' should be a 2D NumPy array, 2D sparse matrix or "
            "dataframe when indexing the columns (i.e. 'axis=1'). "
            f"Got {type(X)} instead with {ndim} dimension(s)."
        )

    if (
        axis == 1
        and indices_dtype == "str"
        and not (
            nw.dependencies.is_into_dataframe(X) or nw.dependencies.is_into_series(X)
        )
    ):
        raise ValueError(
            "Specifying the columns using strings is only supported for dataframes."
        )

    if hasattr(X, "iloc"):
        # TODO: we should probably use is_pandas_df_or_series(X) instead but:
        # 1) Currently, it (probably) works for dataframes compliant to pandas' API.
        # 2) Updating would require updating some tests such as
        #    test_train_test_split_mock_pandas.
        # 3) Should also work with _narwhals_indexing, but
        #    test_safe_indexing_pandas_no_settingwithcopy_warning does not pass.
        return _pandas_indexing(X, indices, indices_dtype, axis=axis)
    elif nw.dependencies.is_into_dataframe(X) or nw.dependencies.is_into_series(X):
        return _narwhals_indexing(X, indices, indices_dtype, axis=axis)
    elif is_pyarrow_data(X):
        # Narwhals Series are backed by ChunkedArray, not Array.
        # To reuse `_narwhals_indexing`, we temporarily convert to `ChunkedArray`.
        pa = sys.modules["pyarrow"]
        X = pa.chunked_array(X)
        ret = _narwhals_indexing(X, indices, indices_dtype, axis=axis)
        if isinstance(ret, pa.ChunkedArray):
            return ret.combine_chunks()
        return ret
    if hasattr(X, "shape"):
        return _array_indexing(X, indices, indices_dtype, axis=axis)
    else:
        return _list_indexing(X, indices, indices_dtype)


def _safe_assign(X, values, *, row_indexer=None, column_indexer=None):
    # [FUNCTION DOC]
    # HELPER: Assigns values into a 2D container in place. Missing row/column indexers
    # default to full slices. pandas objects use iloc assignment while silencing
    # FutureWarning; NumPy/sparse use getitem-style assignment.
    # Inputs: X - 2D ndarray, sparse matrix or DataFrame; values - array of
    # compatible shape; row_indexer, column_indexer - 1d int/bool arrays or None.
    # Outputs: None.
    # Side effects: mutates X in place; suppresses pandas FutureWarning locally.
    """Safe assignment to a numpy array, sparse matrix, or pandas dataframe.

    Parameters
    ----------
    X : {ndarray, sparse-matrix, dataframe}
        Array to be modified. It is expected to be 2-dimensional.

    values : ndarray
        The values to be assigned to `X`.

    row_indexer : array-like, dtype={int, bool}, default=None
        A 1-dimensional array to select the rows of interest. If `None`, all
        rows are selected.

    column_indexer : array-like, dtype={int, bool}, default=None
        A 1-dimensional array to select the columns of interest. If `None`, all
        columns are selected.
    """
    row_indexer = slice(None, None, None) if row_indexer is None else row_indexer
    column_indexer = (
        slice(None, None, None) if column_indexer is None else column_indexer
    )

    if hasattr(X, "iloc"):  # pandas dataframe
        with warnings.catch_warnings():
            # pandas >= 1.5 raises a warning when using iloc to set values in a column
            # that does not have the same type as the column being set. It happens
            # for instance when setting a categorical column with a string.
            # In the future the behavior won't change and the warning should disappear.
            # TODO(1.3): check if the warning is still raised or remove the filter.
            warnings.simplefilter("ignore", FutureWarning)
            X.iloc[row_indexer, column_indexer] = values
    else:  # numpy array or sparse matrix
        X[row_indexer, column_indexer] = values


def _get_column_indices_for_bool_or_int(key, n_columns):
    # [FUNCTION DOC]
    # HELPER: Converts an int/bool column key into a list of column positions by
    # indexing an arange of n_columns with the key (negative ints are kept as their
    # resolved positive positions).
    # Inputs: key - int, bool mask, slice or list of ints; n_columns - int.
    # Outputs: list of int.
    # Exceptions: ValueError (chained from IndexError) when a position is out of range.
    # Convert key into list of positive integer indexes
    try:
        idx = _safe_indexing(np.arange(n_columns), key)
    except IndexError as e:
        raise ValueError(
            f"all features must be in [0, {n_columns - 1}] or [-{n_columns}, 0]"
        ) from e
    return np.atleast_1d(idx).tolist()


def _get_column_indices(X, key):
    # [FUNCTION DOC]
    # HELPER: Resolves a column key into a list of integer column positions for X.
    # Steps: infer key type; for narwhals-compatible dataframes obtain shape and column
    # names (duplicate names raise in narwhals). Empty list/tuple -> []. int/bool keys
    # -> _get_column_indices_for_bool_or_int. Otherwise (string keys, None key, or
    # slice of strings) requires column names: string slices resolve start by name and
    # inclusive stop by name (step must be 1 or None); scalar/list of names map each
    # name to its position.
    # Inputs: X - 2D array or dataframe; key - see _safe_indexing for axis 1.
    # Outputs: list of int positions.
    # Exceptions: ValueError (strings on non-dataframes, unknown column names),
    # NotImplementedError (slice step other than 1/None), ValueError from list.index for
    # unknown slice bounds.
    # KNOWN ISSUE: a key of None or a full slice (key type None) falls into the
    # string branch: it fails on non-dataframes, and on dataframes a None key leads
    # to iterating None (TypeError) while a full slice selects all columns.
    """Get feature column indices for input data X and key.

    For accepted values of `key`, see the docstring of
    :func:`_safe_indexing`.
    """
    key_dtype = _determine_key_type(key)

    if nw.dependencies.is_into_dataframe(X):
        # Note: narwhals raises DuplicateError if column names are not unique.
        df_nw = nw.from_native(X)
        n_columns = df_nw.shape[1]
        column_names = df_nw.columns
    else:
        n_columns = X.shape[1]
        column_names = None

    if isinstance(key, (list, tuple)) and not key:
        # we get an empty list
        return []
    elif key_dtype in ("bool", "int"):
        return _get_column_indices_for_bool_or_int(key, n_columns)
    else:
        if column_names is None:
            raise ValueError(
                "Specifying the columns using strings is only supported for dataframes."
            )

        if isinstance(key, slice):
            if key.step not in [1, None]:
                raise NotImplementedError("key.step must be 1 or None")

            start, stop = key.start, key.stop
            if start is not None:
                start = column_names.index(start)

            if stop is not None:
                stop = column_names.index(stop) + 1
            else:
                stop = n_columns + 1
            return list(islice(range(n_columns), start, stop))
        else:
            selected_columns = [key] if np.isscalar(key) else key
            try:
                return [column_names.index(col) for col in selected_columns]
            except ValueError as e:
                missing = {*selected_columns} - {*column_names}
                msg = f"Some column names are not columns of the dataframe: {missing}"
                raise ValueError(msg) from e


@validate_params(
    {
        "replace": ["boolean"],
        "n_samples": [Interval(numbers.Integral, 1, None, closed="left"), None],
        "random_state": ["random_state"],
        "stratify": ["array-like", "sparse matrix", None],
        "sample_weight": ["array-like", None],
    },
    prefer_skip_nested_validation=True,
)
def resample(
    *arrays,
    replace=True,
    n_samples=None,
    random_state=None,
    stratify=None,
    sample_weight=None,
):
    # [FUNCTION DOC]
    # Draws a consistent random subset (bootstrap or permutation) of rows from all arrays.
    # Steps: return None if no arrays. n_samples defaults to the length of the first
    # array; error if it exceeds that length without replacement. Check consistent
    # lengths. sample_weight is only allowed with replace=True and stratify=None.
    # Without stratify: with replacement draw indices uniformly or with probabilities
    # proportional to validated sample_weight; without replacement shuffle all
    # positions and keep the first n_samples. With stratify: multi-column labels are
    # collapsed into one string per row; compute per-class counts and sorted member
    # indices, allocate per-class draw counts with _approximate_mode, draw from each
    # class (with or without replacement) and randomly permute the concatenation.
    # Sparse inputs are converted to CSR and every array is indexed with
    # _safe_indexing.
    # Inputs: *arrays - indexable sequences with equal first dimension; replace - bool;
    # n_samples - int >= 1 or None; random_state - seed/RandomState/None; stratify -
    # labels array or None; sample_weight - 1d array or None.
    # Outputs: single resampled array if one input, else list of resampled arrays.
    # Exceptions: ValueError (n_samples too large without replacement, inconsistent
    # lengths), NotImplementedError (sample_weight with replace=False or stratify),
    # InvalidParameterError from validate_params.
    # Side effects: consumes random_state; inputs are not modified.
    """Resample arrays or sparse matrices in a consistent way.

    The default strategy implements one step of the bootstrapping
    procedure.

    Parameters
    ----------
    *arrays : sequence of array-like of shape (n_samples,) or \
            (n_samples, n_outputs)
        Indexable data-structures can be arrays, lists, dataframes or scipy
        sparse matrices with consistent first dimension.

    replace : bool, default=True
        Implements resampling with replacement. It must be set to True
        whenever sampling with non-uniform weights: a few data points with very large
        weights are expected to be sampled several times with probability to preserve
        the distribution induced by the weights. If False, this will implement
        (sliced) random permutations.

    n_samples : int, default=None
        Number of samples to generate. If left to None this is
        automatically set to the first dimension of the arrays.
        If replace is False it should not be larger than the length of
        arrays.

    random_state : int, RandomState instance or None, default=None
        Determines random number generation for shuffling
        the data.
        Pass an int for reproducible results across multiple function calls.
        See :term:`Glossary <random_state>`.

    stratify : {array-like, sparse matrix} of shape (n_samples,) or \
            (n_samples, n_outputs), default=None
        If not None, data is split in a stratified fashion, using this as
        the class labels.

    sample_weight : array-like of shape (n_samples,), default=None
        Contains weight values to be associated with each sample. Values are
        normalized to sum to one and interpreted as probability for sampling
        each data point.

        .. versionadded:: 1.7

    Returns
    -------
    resampled_arrays : sequence of array-like of shape (n_samples,) or \
            (n_samples, n_outputs)
        Sequence of resampled copies of the collections. The original arrays
        are not impacted.

    See Also
    --------
    shuffle : Shuffle arrays or sparse matrices in a consistent way.

    Examples
    --------
    It is possible to mix sparse and dense arrays in the same run::

      >>> import numpy as np
      >>> X = np.array([[1., 0.], [2., 1.], [0., 0.]])
      >>> y = np.array([0, 1, 2])

      >>> from scipy.sparse import coo_array
      >>> X_sparse = coo_array(X)

      >>> from sklearn.utils import resample
      >>> X, X_sparse, y = resample(X, X_sparse, y, random_state=0)
      >>> X
      array([[1., 0.],
             [2., 1.],
             [1., 0.]])

      >>> X_sparse
      <Compressed Sparse Row sparse array of dtype 'float64'
          with 4 stored elements and shape (3, 2)>

      >>> X_sparse.toarray()
      array([[1., 0.],
             [2., 1.],
             [1., 0.]])

      >>> y
      array([0, 1, 0])

      >>> resample(y, n_samples=2, random_state=0)
      array([0, 1])

    Example using stratification::

      >>> y = [0, 0, 1, 1, 1, 1, 1, 1, 1]
      >>> resample(y, n_samples=5, replace=False, stratify=y,
      ...          random_state=0)
      [1, 1, 1, 0, 1]
    """
    max_n_samples = n_samples
    random_state = check_random_state(random_state)

    if len(arrays) == 0:
        return None

    first = arrays[0]
    n_samples = first.shape[0] if hasattr(first, "shape") else len(first)

    if max_n_samples is None:
        max_n_samples = n_samples
    elif (max_n_samples > n_samples) and (not replace):
        raise ValueError(
            "Cannot sample %d out of arrays with dim %d when replace is False"
            % (max_n_samples, n_samples)
        )

    check_consistent_length(*arrays)

    if sample_weight is not None and not replace:
        raise NotImplementedError(
            "Resampling with sample_weight is only implemented for replace=True."
        )
    if sample_weight is not None and stratify is not None:
        raise NotImplementedError(
            "Resampling with sample_weight is only implemented for stratify=None."
        )
    if stratify is None:
        if replace:
            if sample_weight is not None:
                sample_weight = _check_sample_weight(
                    sample_weight, first, dtype=np.float64
                )
                p = sample_weight / sample_weight.sum()
            else:
                p = None
            indices = random_state.choice(
                n_samples,
                size=max_n_samples,
                p=p,
                replace=True,
            )
        else:
            indices = np.arange(n_samples)
            random_state.shuffle(indices)
            indices = indices[:max_n_samples]
    else:
        # Code adapted from StratifiedShuffleSplit()
        y = check_array(stratify, ensure_2d=False, dtype=None)
        if y.ndim == 2:
            # for multi-label y, map each distinct row to a string repr
            # using join because str(row) uses an ellipsis if len(row) > 1000
            y = np.array([" ".join(row.astype("str")) for row in y])

        classes, y_indices = np.unique(y, return_inverse=True)
        n_classes = classes.shape[0]

        class_counts = np.bincount(y_indices)

        # Find the sorted list of instances for each class:
        # (np.unique above performs a sort, so code is O(n logn) already)
        class_indices = np.split(
            np.argsort(y_indices, kind="mergesort"), np.cumsum(class_counts)[:-1]
        )

        n_i = _approximate_mode(class_counts, max_n_samples, random_state)

        indices = []

        for i in range(n_classes):
            indices_i = random_state.choice(class_indices[i], n_i[i], replace=replace)
            indices.extend(indices_i)

        indices = random_state.permutation(indices)

    # convert sparse matrices to CSR for row-based indexing
    arrays = [a.tocsr() if issparse(a) else a for a in arrays]
    resampled_arrays = [_safe_indexing(a, indices) for a in arrays]
    if len(resampled_arrays) == 1:
        # syntactic sugar for the unit argument case
        return resampled_arrays[0]
    else:
        return resampled_arrays


def shuffle(*arrays, random_state=None, n_samples=None):
    # [FUNCTION DOC]
    # Thin wrapper returning resample(*arrays, replace=False, n_samples, random_state),
    # i.e. a consistent random permutation (optionally truncated) of all arrays.
    # Inputs/Outputs/Exceptions: as resample with replace fixed to False.
    # Side effects: consumes random_state.
    """Shuffle arrays or sparse matrices in a consistent way.

    This is a convenience alias to ``resample(*arrays, replace=False)`` to do
    random permutations of the collections.

    Parameters
    ----------
    *arrays : sequence of indexable data-structures
        Indexable data-structures can be arrays, lists, dataframes or scipy
        sparse matrices with consistent first dimension.

    random_state : int, RandomState instance or None, default=None
        Determines random number generation for shuffling
        the data.
        Pass an int for reproducible results across multiple function calls.
        See :term:`Glossary <random_state>`.

    n_samples : int, default=None
        Number of samples to generate. If left to None this is
        automatically set to the first dimension of the arrays.  It should
        not be larger than the length of arrays.

    Returns
    -------
    shuffled_arrays : sequence of indexable data-structures
        Sequence of shuffled copies of the collections. The original arrays
        are not impacted.

    See Also
    --------
    resample : Resample arrays or sparse matrices in a consistent way.

    Examples
    --------
    It is possible to mix sparse and dense arrays in the same run::

      >>> import numpy as np
      >>> X = np.array([[1., 0.], [2., 1.], [0., 0.]])
      >>> y = np.array([0, 1, 2])

      >>> from scipy.sparse import coo_array
      >>> X_sparse = coo_array(X)

      >>> from sklearn.utils import shuffle
      >>> X, X_sparse, y = shuffle(X, X_sparse, y, random_state=0)
      >>> X
      array([[0., 0.],
             [2., 1.],
             [1., 0.]])

      >>> X_sparse
      <Compressed Sparse Row sparse array of dtype 'float64'
          with 3 stored elements and shape (3, 2)>

      >>> X_sparse.toarray()
      array([[0., 0.],
             [2., 1.],
             [1., 0.]])

      >>> y
      array([2, 1, 0])

      >>> shuffle(y, n_samples=2, random_state=0)
      array([0, 1])
    """
    return resample(
        *arrays, replace=False, n_samples=n_samples, random_state=random_state
    )
