# [MODULE DOC]
# Utilities for building boolean masks of missing/target values (dense and
# sparse) and for safely applying row masks to dense or sparse data.
# No import-time side effects.
# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

from contextlib import suppress

import numpy as np
from scipy import sparse as sp

from sklearn.utils._missing import is_scalar_nan
from sklearn.utils._param_validation import validate_params
from sklearn.utils._sparse import _align_api_if_sparse
from sklearn.utils.fixes import _object_dtype_isnan


def _get_dense_mask(X, value_to_mask):
    # [FUNCTION DOC]
    # HELPER: Elementwise boolean mask of where X equals value_to_mask on a dense array.
    # Steps: if pandas is importable and value_to_mask is pandas.NA, return pandas.isna
    # of X. Otherwise if value_to_mask is a scalar NaN: float dtype uses NaN test,
    # integer/unsigned dtype gives an all-False array, any other dtype (object) uses
    # an object-aware NaN test. For non-NaN values use elementwise equality.
    # Inputs: X - ndarray (or 1d data array of a sparse matrix); value_to_mask - scalar.
    # Outputs: boolean array of X's shape (pandas output type when pandas.NA path used).
    # Exceptions: ImportError/AttributeError around pandas are suppressed.
    # Side effects: may import pandas.
    with suppress(ImportError, AttributeError):
        # We also suppress `AttributeError` because older versions of pandas do
        # not have `NA`.
        import pandas

        if value_to_mask is pandas.NA:
            return pandas.isna(X)

    if is_scalar_nan(value_to_mask):
        if X.dtype.kind == "f":
            Xt = np.isnan(X)
        elif X.dtype.kind in ("i", "u"):
            # can't have NaNs in integer array.
            Xt = np.zeros(X.shape, dtype=bool)
        else:
            # np.isnan does not work on object dtypes.
            Xt = _object_dtype_isnan(X)
    else:
        Xt = X == value_to_mask

    return Xt


def _get_mask(X, value_to_mask):
    # [FUNCTION DOC]
    # HELPER: Like _get_dense_mask but supports scipy sparse inputs. For dense input
    # delegates directly. For sparse input, computes the mask over the stored data
    # only and builds a boolean CSR array (if X is CSR) or CSC array (any other
    # format) reusing copies of X's indices/indptr and shape, then aligns the
    # sparse API (matrix vs array) with the input via _align_api_if_sparse.
    # Inputs: X - ndarray or sparse matrix/array; value_to_mask - scalar.
    # Outputs: boolean ndarray or sparse container of the same shape.
    # KNOWN ISSUE: non-CSR/CSC sparse formats (e.g. COO) are treated as CSC and
    # reference X.indices/X.indptr, which such formats do not have.
    # Note: implicit zeros are never masked, even when value_to_mask is 0.
    # Side effects: none on X (indices/indptr are copied).
    """Compute the boolean mask X == value_to_mask.

    Parameters
    ----------
    X : {ndarray, sparse matrix} of shape (n_samples, n_features)
        Input data, where ``n_samples`` is the number of samples and
        ``n_features`` is the number of features.

    value_to_mask : {int, float}
        The value which is to be masked in X.

    Returns
    -------
    X_mask : {ndarray, sparse matrix} of shape (n_samples, n_features)
        Missing mask.
    """
    if not sp.issparse(X):
        # For all cases apart of a sparse input where we need to reconstruct
        # a sparse output
        return _get_dense_mask(X, value_to_mask)

    Xt = _get_dense_mask(X.data, value_to_mask)

    sparse_constructor = sp.csr_array if X.format == "csr" else sp.csc_array
    Xt_sparse = sparse_constructor(
        (Xt, X.indices.copy(), X.indptr.copy()), shape=X.shape, dtype=bool
    )

    return _align_api_if_sparse(Xt_sparse)


@validate_params(
    {
        "X": ["array-like", "sparse matrix"],
        "mask": ["array-like"],
    },
    prefer_skip_nested_validation=True,
)
def safe_mask(X, mask):
    # [FUNCTION DOC]
    # Converts a mask into a form usable to index rows of X. Mask is converted to an
    # ndarray; signed-integer masks are returned unchanged. Otherwise, if X is sparse
    # (has toarray), the boolean mask is turned into integer positions by indexing a
    # range of the mask length; for dense X the boolean mask is returned as is.
    # Inputs: X - array-like or sparse matrix; mask - array-like of bools or ints.
    # Outputs: ndarray mask (bool or integer indices).
    # Exceptions: InvalidParameterError from validate_params on bad input types.
    # Side effects: none.
    """Return an indexing mask compatible with X.

    Parameters
    ----------
    X : {array-like, sparse matrix}
        Data on which to apply mask.

    mask : array-like
        Mask to be used on X.

    Returns
    -------
    mask : ndarray
        Array that is safe to use on X.

    Examples
    --------
    >>> from sklearn.utils import safe_mask
    >>> from scipy.sparse import csr_array
    >>> data = csr_array([[1], [2], [3], [4], [5]])
    >>> condition = [False, True, True, False, True]
    >>> mask = safe_mask(data, condition)
    >>> data[mask].toarray()
    array([[2],
           [3],
           [5]])
    """
    mask = np.asarray(mask)
    if np.issubdtype(mask.dtype, np.signedinteger):
        return mask

    if hasattr(X, "toarray"):
        ind = np.arange(mask.shape[0])
        mask = ind[mask]
    return mask


def axis0_safe_slice(X, mask, len_mask):
    # [FUNCTION DOC]
    # Returns the rows of X selected by mask (not a mask, despite the docstring), or an
    # empty float array of shape (0, n_features) when len_mask is 0.
    # Inputs: X - 2d array or sparse matrix; mask - boolean/int row mask; len_mask - int
    # count of selected rows (only compared to 0).
    # Outputs: sliced X (same container type) or zeros array of shape (0, X.shape[1]).
    # KNOWN ISSUE: docstring says it returns a mask; it returns the sliced data. The
    # empty result is always a dense float64 array even for sparse/other-dtype X.
    # Side effects: none.
    """Return a mask which is safer to use on X than safe_mask.

    This mask is safer than safe_mask since it returns an
    empty array, when a sparse matrix is sliced with a boolean mask
    with all False, instead of raising an unhelpful error in older
    versions of SciPy.

    See: https://github.com/scipy/scipy/issues/5361

    Also note that we can avoid doing the dot product by checking if
    the len_mask is not zero in _huber_loss_and_gradient but this
    is not going to be the bottleneck, since the number of outliers
    and non_outliers are typically non-zero and it makes the code
    tougher to follow.

    Parameters
    ----------
    X : {array-like, sparse matrix}
        Data on which to apply mask.

    mask : ndarray
        Mask to be used on X.

    len_mask : int
        The length of the mask.

    Returns
    -------
    mask : ndarray
        Array that is safe to use on X.
    """
    if len_mask != 0:
        return X[safe_mask(X, mask), :]
    return np.zeros(shape=(0, X.shape[1]))


def indices_to_mask(indices, mask_length):
    # [FUNCTION DOC]
    # Builds a boolean array of length mask_length set True at the given positions.
    # Inputs: indices - list-like of ints; mask_length - int, must exceed max(indices).
    # Outputs: 1d boolean ndarray of length mask_length.
    # Exceptions: ValueError if mask_length <= max(indices); max of empty indices
    # raises a numpy ValueError.
    # Side effects: none.
    """Convert list of indices to boolean mask.

    Parameters
    ----------
    indices : list-like
        List of integers treated as indices.
    mask_length : int
        Length of boolean mask to be generated.
        This parameter must be greater than max(indices).

    Returns
    -------
    mask : 1d boolean nd-array
        Boolean array that is True where indices are present, else False.

    Examples
    --------
    >>> from sklearn.utils._mask import indices_to_mask
    >>> indices = [1, 2 , 3, 4]
    >>> indices_to_mask(indices, 5)
    array([False,  True,  True,  True,  True])
    """
    if mask_length <= np.max(indices):
        raise ValueError("mask_length must be greater than max(indices)")

    mask = np.zeros(mask_length, dtype=bool)
    mask[indices] = True

    return mask
