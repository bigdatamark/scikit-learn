# [MODULE DOC]
# Cython kernels for scipy.sparse CSR/CSC matrices, mostly wrapped by
# sklearn.utils.sparsefuncs and used by preprocessing (StandardScaler,
# Normalizer), clustering and SGD code: squared row norms, weighted NaN-aware
# per-feature mean and variance (one-shot and incremental), in-place row
# L1/L2 normalisation, densification of selected CSR rows, and CSR @ CSR
# products into a dense output. Public def functions validate/convert dtypes
# (non-float data is cast to float64) and pass the raw data/indices/indptr
# arrays to private fused-type kernels (fused over float32/float64 data and
# int32/int64 index arrays). Most loops run with the GIL held; none use OpenMP.
"""Utilities to work with sparse matrices and arrays written in Cython."""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

from libc.math cimport fabs, sqrt, isnan
from libc.stdint cimport intptr_t

import numpy as np
from cython cimport floating
from sklearn.utils.fixes import _ensure_sparse_index_int32
from sklearn.utils._typedefs cimport float64_t, int32_t, int64_t, intp_t, uint64_t


ctypedef fused integral:
    int32_t
    int64_t

ctypedef fused integral2:
    int32_t
    int64_t


def csr_row_norms(X):
    # [FUNCTION DOC]
    # Behavior: cast X to float64 if it is not float32/float64, then return the
    # squared L2 norm of each row.
    # Inputs: X, CSR matrix/array of shape (n_samples, n_features).
    # Outputs: 1d array (n_samples,) of X's float dtype.
    # Exceptions: buffer errors if X.data is not contiguous. Side effects: none.
    """Squared L2 norm of each row in CSR matrix X."""
    if X.dtype not in [np.float32, np.float64]:
        X = X.astype(np.float64)
    return _sqeuclidean_row_norms_sparse(X.data, X.indptr)


def _sqeuclidean_row_norms_sparse(
    const floating[::1] X_data,
    const integral[::1] X_indptr,
):
    # [FUNCTION DOC]
    # HELPER: kernel for csr_row_norms.
    # Behavior: for each row i, sum the squares of data[indptr[i]:indptr[i+1]]
    # (accumulated in the data dtype). Runs without the GIL.
    # Inputs: X_data (contiguous floating), X_indptr (contiguous int32/int64).
    # Outputs: new array of length len(indptr) - 1. Exceptions: buffer errors.
    cdef:
        integral n_samples = X_indptr.shape[0] - 1
        integral i, j

    dtype = np.float32 if floating is float else np.float64

    cdef floating[::1] squared_row_norms = np.zeros(n_samples, dtype=dtype)

    with nogil:
        for i in range(n_samples):
            for j in range(X_indptr[i], X_indptr[i + 1]):
                squared_row_norms[i] += X_data[j] * X_data[j]

    return np.asarray(squared_row_norms)


def csr_mean_variance_axis0(X, weights=None, return_sum_weights=False):
    # [FUNCTION DOC]
    # Behavior: cast X to float64 if needed, default weights to ones of X's dtype,
    # compute per-column weighted mean and variance ignoring NaNs (implicit zeros
    # count as values) via _csr_mean_variance_axis0.
    # Inputs: X (CSR, (n_samples, n_features)), weights (n_samples,) or None,
    # return_sum_weights (bool).
    # Outputs: (means, variances) or (means, variances, sum_weights), each of
    # shape (n_features,) in X's float dtype.
    # Exceptions: buffer dtype errors (e.g. weights dtype differing from X's).
    """Compute mean and variance along axis 0 on a CSR matrix

    Uses an np.float64 accumulator.

    Parameters
    ----------
    X : CSR sparse matrix, shape (n_samples, n_features)
        Input data.

    weights : ndarray of shape (n_samples,), dtype=floating, default=None
        If it is set to None samples will be equally weighted.

        .. versionadded:: 0.24

    return_sum_weights : bool, default=False
        If True, returns the sum of weights seen for each feature.

        .. versionadded:: 0.24

    Returns
    -------
    means : float array with shape (n_features,)
        Feature-wise means

    variances : float array with shape (n_features,)
        Feature-wise variances

    sum_weights : ndarray of shape (n_features,), dtype=floating
        Returned if return_sum_weights is True.
    """
    if X.dtype not in [np.float32, np.float64]:
        X = X.astype(np.float64)

    if weights is None:
        weights = np.ones(X.shape[0], dtype=X.dtype)

    means, variances, sum_weights = _csr_mean_variance_axis0(
        X.data, X.shape[0], X.shape[1], X.indices, X.indptr, weights)

    if return_sum_weights:
        return means, variances, sum_weights
    return means, variances


def _csr_mean_variance_axis0(
    const floating[::1] X_data,
    uint64_t n_samples,
    uint64_t n_features,
    const integral[:] X_indices,
    const integral[:] X_indptr,
    const floating[:] weights,
):
    # [FUNCTION DOC]
    # HELPER: kernel for csr_mean_variance_axis0 (and the incremental variant).
    # Behavior: float64 accumulation. sum_weights starts at the total weight and
    # counts at n_rows for each feature. Pass 1 over stored entries: for non-NaN
    # values add w*x to means, w to sum_weights_nz and 1 to counts_nz; for NaN
    # subtract w from sum_weights and 1 from counts. means /= sum_weights. Pass 2:
    # for stored non-NaN values accumulate correction += w*(x - mean) and
    # variances += w*(x - mean)^2. Finally, if a feature has implicit zeros
    # (counts != counts_nz), account for them: correction -= (sum_weights -
    # sum_weights_nz) * mean and variances += (sum_weights - sum_weights_nz) *
    # mean^2; then variance = (variances - correction^2 / sum_weights) /
    # sum_weights (corrected two-pass algorithm of Chan, Golub and LeVeque).
    # Inputs: X_data, n_samples (unused), n_features, X_indices, X_indptr,
    # weights (length n_samples).
    # Outputs: (means, variances, sum_weights) as float32 copies if the data is
    # float32, else float64 arrays.
    # Exceptions: buffer errors. A feature whose entries are all NaN (zero total
    # weight) yields NaN/inf (C division, no exception).
    # Implement the function here since variables using fused types
    # cannot be declared directly and can only be passed as function arguments
    cdef:
        intp_t row_ind
        uint64_t feature_idx
        integral i, col_ind
        float64_t diff
        # means[j] contains the mean of feature j
        float64_t[::1] means = np.zeros(n_features)
        # variances[j] contains the variance of feature j
        float64_t[::1] variances = np.zeros(n_features)

        float64_t[::1] sum_weights = np.full(
            fill_value=np.sum(weights, dtype=np.float64), shape=n_features
        )
        float64_t[::1] sum_weights_nz = np.zeros(shape=n_features)
        float64_t[::1] correction = np.zeros(shape=n_features)

        uint64_t[::1] counts = np.full(
            fill_value=weights.shape[0], shape=n_features, dtype=np.uint64
        )
        uint64_t[::1] counts_nz = np.zeros(shape=n_features, dtype=np.uint64)

    for row_ind in range(len(X_indptr) - 1):
        for i in range(X_indptr[row_ind], X_indptr[row_ind + 1]):
            col_ind = X_indices[i]
            if not isnan(X_data[i]):
                means[col_ind] += <float64_t>(X_data[i]) * weights[row_ind]
                # sum of weights where X[:, col_ind] is non-zero
                sum_weights_nz[col_ind] += weights[row_ind]
                # number of non-zero elements of X[:, col_ind]
                counts_nz[col_ind] += 1
            else:
                # sum of weights where X[:, col_ind] is not nan
                sum_weights[col_ind] -= weights[row_ind]
                # number of non nan elements of X[:, col_ind]
                counts[col_ind] -= 1

    for feature_idx in range(n_features):
        means[feature_idx] /= sum_weights[feature_idx]

    for row_ind in range(len(X_indptr) - 1):
        for i in range(X_indptr[row_ind], X_indptr[row_ind + 1]):
            col_ind = X_indices[i]
            if not isnan(X_data[i]):
                diff = X_data[i] - means[col_ind]
                # correction term of the corrected 2 pass algorithm.
                # See "Algorithms for computing the sample variance: analysis
                # and recommendations", by Chan, Golub, and LeVeque.
                correction[col_ind] += diff * weights[row_ind]
                variances[col_ind] += diff * diff * weights[row_ind]

    for feature_idx in range(n_features):
        if counts[feature_idx] != counts_nz[feature_idx]:
            correction[feature_idx] -= (
                sum_weights[feature_idx] - sum_weights_nz[feature_idx]
            ) * means[feature_idx]
        correction[feature_idx] = correction[feature_idx]**2 / sum_weights[feature_idx]
        if counts[feature_idx] != counts_nz[feature_idx]:
            # only compute it when it's guaranteed to be non-zero to avoid
            # catastrophic cancellation.
            variances[feature_idx] += (
                sum_weights[feature_idx] - sum_weights_nz[feature_idx]
            ) * means[feature_idx]**2
        variances[feature_idx] = (
            (variances[feature_idx] - correction[feature_idx]) /
            sum_weights[feature_idx]
        )

    if floating is float:
        return (
            np.array(means, dtype=np.float32),
            np.array(variances, dtype=np.float32),
            np.array(sum_weights, dtype=np.float32),
        )
    else:
        return (
            np.asarray(means), np.asarray(variances), np.asarray(sum_weights)
        )


def csc_mean_variance_axis0(X, weights=None, return_sum_weights=False):
    # [FUNCTION DOC]
    # Behavior: CSC counterpart of csr_mean_variance_axis0: cast to float64 if
    # needed, default unit weights, call _csc_mean_variance_axis0.
    # Inputs: X (CSC), weights or None, return_sum_weights.
    # Outputs: (means, variances[, sum_weights]) of shape (n_features,).
    # Exceptions: buffer dtype errors.
    """Compute mean and variance along axis 0 on a CSC matrix

    Uses an np.float64 accumulator.

    Parameters
    ----------
    X : CSC sparse matrix, shape (n_samples, n_features)
        Input data.

    weights : ndarray of shape (n_samples,), dtype=floating, default=None
        If it is set to None samples will be equally weighted.

        .. versionadded:: 0.24

    return_sum_weights : bool, default=False
        If True, returns the sum of weights seen for each feature.

        .. versionadded:: 0.24

    Returns
    -------
    means : float array with shape (n_features,)
        Feature-wise means

    variances : float array with shape (n_features,)
        Feature-wise variances

    sum_weights : ndarray of shape (n_features,), dtype=floating
        Returned if return_sum_weights is True.
    """
    if X.dtype not in [np.float32, np.float64]:
        X = X.astype(np.float64)

    if weights is None:
        weights = np.ones(X.shape[0], dtype=X.dtype)

    means, variances, sum_weights = _csc_mean_variance_axis0(
        X.data, X.shape[0], X.shape[1], X.indices, X.indptr, weights)

    if return_sum_weights:
        return means, variances, sum_weights
    return means, variances


def _csc_mean_variance_axis0(
    const floating[::1] X_data,
    uint64_t n_samples,
    uint64_t n_features,
    const integral[:] X_indices,
    const integral[:] X_indptr,
    const floating[:] weights,
):
    # [FUNCTION DOC]
    # HELPER: kernel for csc_mean_variance_axis0 (and the incremental variant).
    # Behavior: identical algorithm to _csr_mean_variance_axis0 but iterating
    # column by column (indptr over columns, indices are row numbers used to look
    # up weights).
    # Inputs: X_data, n_samples (unused), n_features, X_indices, X_indptr, weights.
    # Outputs: (means, variances, sum_weights) as in the CSR kernel.
    # Exceptions: buffer errors; all-NaN columns give NaN/inf.
    # Implement the function here since variables using fused types
    # cannot be declared directly and can only be passed as function arguments
    cdef:
        integral i, row_ind
        uint64_t feature_idx, col_ind
        float64_t diff
        # means[j] contains the mean of feature j
        float64_t[::1] means = np.zeros(n_features)
        # variances[j] contains the variance of feature j
        float64_t[::1] variances = np.zeros(n_features)

        float64_t[::1] sum_weights = np.full(
            fill_value=np.sum(weights, dtype=np.float64), shape=n_features
        )
        float64_t[::1] sum_weights_nz = np.zeros(shape=n_features)
        float64_t[::1] correction = np.zeros(shape=n_features)

        uint64_t[::1] counts = np.full(
            fill_value=weights.shape[0], shape=n_features, dtype=np.uint64
        )
        uint64_t[::1] counts_nz = np.zeros(shape=n_features, dtype=np.uint64)

    for col_ind in range(n_features):
        for i in range(X_indptr[col_ind], X_indptr[col_ind + 1]):
            row_ind = X_indices[i]
            if not isnan(X_data[i]):
                means[col_ind] += <float64_t>(X_data[i]) * weights[row_ind]
                # sum of weights where X[:, col_ind] is non-zero
                sum_weights_nz[col_ind] += weights[row_ind]
                # number of non-zero elements of X[:, col_ind]
                counts_nz[col_ind] += 1
            else:
                # sum of weights where X[:, col_ind] is not nan
                sum_weights[col_ind] -= weights[row_ind]
                # number of non nan elements of X[:, col_ind]
                counts[col_ind] -= 1

    for feature_idx in range(n_features):
        means[feature_idx] /= sum_weights[feature_idx]

    for col_ind in range(n_features):
        for i in range(X_indptr[col_ind], X_indptr[col_ind + 1]):
            row_ind = X_indices[i]
            if not isnan(X_data[i]):
                diff = X_data[i] - means[col_ind]
                # correction term of the corrected 2 pass algorithm.
                # See "Algorithms for computing the sample variance: analysis
                # and recommendations", by Chan, Golub, and LeVeque.
                correction[col_ind] += diff * weights[row_ind]
                variances[col_ind] += diff * diff * weights[row_ind]

    for feature_idx in range(n_features):
        if counts[feature_idx] != counts_nz[feature_idx]:
            correction[feature_idx] -= (
                sum_weights[feature_idx] - sum_weights_nz[feature_idx]
            ) * means[feature_idx]
        correction[feature_idx] = correction[feature_idx]**2 / sum_weights[feature_idx]
        if counts[feature_idx] != counts_nz[feature_idx]:
            # only compute it when it's guaranteed to be non-zero to avoid
            # catastrophic cancellation.
            variances[feature_idx] += (
                sum_weights[feature_idx] - sum_weights_nz[feature_idx]
            ) * means[feature_idx]**2
        variances[feature_idx] = (
            (variances[feature_idx] - correction[feature_idx])
        ) / sum_weights[feature_idx]

    if floating is float:
        return (np.array(means, dtype=np.float32),
                np.array(variances, dtype=np.float32),
                np.array(sum_weights, dtype=np.float32))
    else:
        return (
            np.asarray(means), np.asarray(variances), np.asarray(sum_weights)
        )


def incr_mean_variance_axis0(X, last_mean, last_var, last_n, weights=None):
    # [FUNCTION DOC]
    # Behavior: normalise dtypes (X and weights to float, last_n to float), ensure
    # index arrays are int32 (via _ensure_sparse_index_int32, which may raise),
    # cast last statistics to X's dtype and call _incr_mean_variance_axis0.
    # Inputs: X (CSR or CSC), last_mean, last_var, last_n (n_features,), weights
    # (n_samples,) or None.
    # Outputs: (updated_mean, updated_var, updated_n), each (n_features,).
    # Exceptions: ValueError if indices do not fit in int32; buffer errors.
    # Side effects: see KNOWN ISSUE in _incr_mean_variance_axis0 (the caller's
    # last_mean / last_var may be modified when already of X's dtype).
    """Compute mean and variance along axis 0 on a CSR or CSC matrix.

    last_mean, last_var are the statistics computed at the last step by this
    function. Both must be initialized to 0.0. last_n is the
    number of samples encountered until now and is initialized at 0.

    Parameters
    ----------
    X : CSR or CSC sparse matrix, shape (n_samples, n_features)
      Input data.

    last_mean : float array with shape (n_features,)
      Array of feature-wise means to update with the new data X.

    last_var : float array with shape (n_features,)
      Array of feature-wise var to update with the new data X.

    last_n : float array with shape (n_features,)
      Sum of the weights seen so far (if weights are all set to 1
      this will be the same as number of samples seen so far, before X).

    weights : float array with shape (n_samples,) or None. If it is set
      to None samples will be equally weighted.

    Returns
    -------
    updated_mean : float array with shape (n_features,)
      Feature-wise means

    updated_variance : float array with shape (n_features,)
      Feature-wise variances

    updated_n : int array with shape (n_features,)
      Updated number of samples seen

    Notes
    -----
    NaNs are ignored during the computation.

    References
    ----------
    T. Chan, G. Golub, R. LeVeque. Algorithms for computing the sample
      variance: recommendations, The American Statistician, Vol. 37, No. 3,
      pp. 242-247

    Also, see the non-sparse implementation of this in
    `utils.extmath._batch_mean_variance_update`.

    """
    if X.dtype not in [np.float32, np.float64]:
        X = X.astype(np.float64)
    X_dtype = X.dtype
    if weights is None:
        weights = np.ones(X.shape[0], dtype=X_dtype)
    elif weights.dtype not in [np.float32, np.float64]:
        weights = weights.astype(np.float64, copy=False)
    if last_n.dtype not in [np.float32, np.float64]:
        last_n = last_n.astype(np.float64, copy=False)

    _ensure_sparse_index_int32(X)
    return _incr_mean_variance_axis0(X.data,
                                     np.sum(weights),
                                     X.shape[1],
                                     X.indices,
                                     X.indptr,
                                     X.format,
                                     last_mean.astype(X_dtype, copy=False),
                                     last_var.astype(X_dtype, copy=False),
                                     last_n.astype(X_dtype, copy=False),
                                     weights.astype(X_dtype, copy=False))


def _incr_mean_variance_axis0(
    const floating[:] X_data,
    floating n_samples,
    uint64_t n_features,
    const int[:] X_indices,
    # X_indptr might be either int32 or int64
    const integral[:] X_indptr,
    str X_format,
    floating[:] last_mean,
    floating[:] last_var,
    floating[:] last_n,
    # previous sum of the weights (ie float)
    const floating[:] weights,
):
    # [FUNCTION DOC]
    # HELPER: kernel for incr_mean_variance_axis0.
    # Behavior: compute the new batch stats (new_mean, new_var, new_n = sum of
    # non-NaN weights) with the CSR or CSC kernel according to X_format. If all
    # last_n are 0 return the new stats. Otherwise, per feature: updated_n =
    # last_n + new_n; if new_n > 0, combine with Chan et al.'s pairwise update:
    # updated_mean = (last_n*last_mean + new_n*new_mean) / updated_n and
    # updated_var = (last_n*last_var + new_n*new_var + last_n*new_n/updated_n *
    # (last_mean - new_mean)^2) / updated_n; if new_n == 0 keep the last stats.
    # Inputs: X_data, n_samples (total weight, unused by the kernels), n_features,
    # X_indices (int32), X_indptr, X_format ('csr' or anything else = CSC),
    # last_mean, last_var, last_n, weights.
    # Outputs: tuple of three new arrays of the floating dtype.
    # Exceptions: buffer errors.
    # KNOWN ISSUE: last_mean and last_var are scaled in place by last_n (to form
    # unnormalised sums) for features with new_n > 0; since the public wrapper
    # uses astype(copy=False), the caller's arrays are silently mutated when they
    # already have X's dtype.
    # Implement the function here since variables using fused types
    # cannot be declared directly and can only be passed as function arguments
    cdef:
        uint64_t i

        # last = stats until now
        # new = the current increment
        # updated = the aggregated stats
        # when arrays, they are indexed by i per-feature
        floating[::1] new_mean
        floating[::1] new_var
        floating[::1] updated_mean
        floating[::1] updated_var

    if floating is float:
        dtype = np.float32
    else:
        dtype = np.float64

    new_mean = np.zeros(n_features, dtype=dtype)
    new_var = np.zeros_like(new_mean, dtype=dtype)
    updated_mean = np.zeros_like(new_mean, dtype=dtype)
    updated_var = np.zeros_like(new_mean, dtype=dtype)

    cdef:
        floating[::1] new_n
        floating[::1] updated_n
        floating[::1] last_over_new_n

    # Obtain new stats first
    updated_n = np.zeros(shape=n_features, dtype=dtype)
    last_over_new_n = np.zeros_like(updated_n, dtype=dtype)

    # X can be a CSR or CSC matrix
    if X_format == 'csr':
        new_mean, new_var, new_n = _csr_mean_variance_axis0(
            X_data, n_samples, n_features, X_indices, X_indptr, weights)
    else:  # X_format == 'csc'
        new_mean, new_var, new_n = _csc_mean_variance_axis0(
            X_data, n_samples, n_features, X_indices, X_indptr, weights)

    # First pass
    cdef bint is_first_pass = True
    for i in range(n_features):
        if last_n[i] > 0:
            is_first_pass = False
            break

    if is_first_pass:
        return np.asarray(new_mean), np.asarray(new_var), np.asarray(new_n)

    for i in range(n_features):
        updated_n[i] = last_n[i] + new_n[i]

    # Next passes
    for i in range(n_features):
        if new_n[i] > 0:
            last_over_new_n[i] = dtype(last_n[i]) / dtype(new_n[i])
            # Unnormalized stats
            last_mean[i] *= last_n[i]
            last_var[i] *= last_n[i]
            new_mean[i] *= new_n[i]
            new_var[i] *= new_n[i]
            # Update stats
            updated_var[i] = (
                last_var[i] + new_var[i] +
                last_over_new_n[i] / updated_n[i] *
                (last_mean[i] / last_over_new_n[i] - new_mean[i])**2
            )
            updated_mean[i] = (last_mean[i] + new_mean[i]) / updated_n[i]
            updated_var[i] /= updated_n[i]
        else:
            updated_var[i] = last_var[i]
            updated_mean[i] = last_mean[i]
            updated_n[i] = last_n[i]

    return (
        np.asarray(updated_mean),
        np.asarray(updated_var),
        np.asarray(updated_n),
    )


def inplace_csr_row_normalize_l1(X):
    # [FUNCTION DOC]
    # Behavior: divide every stored value of each row of a CSR matrix by the
    # row's L1 norm (sum of absolute stored values); rows with zero norm are left
    # untouched.
    # Inputs: X, CSR matrix/array with float32/float64 data.
    # Outputs: None; X.data modified in place.
    # Exceptions: buffer errors for non-float data.
    """Normalize inplace the rows of a CSR matrix or array by their L1 norm.

    Parameters
    ----------
    X : scipy.sparse.csr_matrix and scipy.sparse.csr_array, \
            shape=(n_samples, n_features)
        The input matrix or array to be modified inplace.

    Examples
    --------
    >>> from scipy.sparse import csr_array
    >>> from sklearn.utils.sparsefuncs_fast import inplace_csr_row_normalize_l1
    >>> import numpy as np
    >>> indptr = np.array([0, 2, 3, 4])
    >>> indices = np.array([0, 1, 2, 3])
    >>> data = np.array([1.0, 2.0, 3.0, 4.0])
    >>> X = csr_array((data, indices, indptr), shape=(3, 4))
    >>> X.toarray()
    array([[1., 2., 0., 0.],
           [0., 0., 3., 0.],
           [0., 0., 0., 4.]])
    >>> inplace_csr_row_normalize_l1(X)
    >>> X.toarray()
    array([[0.33...   , 0.66...   , 0.        , 0.        ],
           [0.        , 0.        , 1.        , 0.        ],
           [0.        , 0.        , 0.        , 1.        ]])
    """
    _inplace_csr_row_normalize_l1(X.data, X.shape, X.indices, X.indptr)


def _inplace_csr_row_normalize_l1(
    floating[:] X_data,
    shape,
    const integral[:] X_indices,
    const integral[:] X_indptr,
):
    # [FUNCTION DOC]
    # HELPER: kernel for inplace_csr_row_normalize_l1.
    # Behavior: per row, accumulate the L1 norm in double, skip if 0, else divide
    # the row's stored values by it. Inputs: X_data (writable), shape (only
    # shape[0] used), X_indices (unused), X_indptr. Outputs: X_data in place.
    cdef:
        uint64_t n_samples = shape[0]

        # the column indices for row i are stored in:
        #    indices[indptr[i]:indices[i+1]]
        # and their corresponding values are stored in:
        #    data[indptr[i]:indptr[i+1]]
        uint64_t i
        integral j
        double sum_

    for i in range(n_samples):
        sum_ = 0.0

        for j in range(X_indptr[i], X_indptr[i + 1]):
            sum_ += fabs(X_data[j])

        if sum_ == 0.0:
            # do not normalize empty rows (can happen if CSR is not pruned
            # correctly)
            continue

        for j in range(X_indptr[i], X_indptr[i + 1]):
            X_data[j] /= sum_


def inplace_csr_row_normalize_l2(X):
    # [FUNCTION DOC]
    # Behavior: divide each row's stored values by the row's L2 norm; rows with
    # zero norm are left untouched.
    # Inputs: X, CSR matrix/array with float data.
    # Outputs: None; X.data modified in place. Exceptions: buffer errors.
    """Normalize inplace the rows of a CSR matrix or array by their L2 norm.

    Parameters
    ----------
    X : scipy.sparse.csr_matrix, shape=(n_samples, n_features)
        The input matrix or array to be modified inplace.

    Examples
    --------
    >>> from scipy.sparse import csr_array
    >>> from sklearn.utils.sparsefuncs_fast import inplace_csr_row_normalize_l2
    >>> import numpy as np
    >>> indptr = np.array([0, 2, 3, 4])
    >>> indices = np.array([0, 1, 2, 3])
    >>> data = np.array([1.0, 2.0, 3.0, 4.0])
    >>> X = csr_array((data, indices, indptr), shape=(3, 4))
    >>> X.toarray()
    array([[1., 2., 0., 0.],
           [0., 0., 3., 0.],
           [0., 0., 0., 4.]])
    >>> inplace_csr_row_normalize_l2(X)
    >>> X.toarray()
    array([[0.44...   , 0.89...   , 0.        , 0.        ],
           [0.        , 0.        , 1.        , 0.        ],
           [0.        , 0.        , 0.        , 1.        ]])
    """
    _inplace_csr_row_normalize_l2(X.data, X.shape, X.indices, X.indptr)


def _inplace_csr_row_normalize_l2(
    floating[:] X_data,
    shape,
    const integral[:] X_indices,
    const integral[:] X_indptr,
):
    # [FUNCTION DOC]
    # HELPER: kernel for inplace_csr_row_normalize_l2.
    # Behavior: per row, sum of squares in double, skip if 0, else divide stored
    # values by its square root. Inputs: X_data (writable), shape (shape[0]
    # used), X_indices (unused), X_indptr. Outputs: X_data in place.
    cdef:
        uint64_t n_samples = shape[0]
        uint64_t i
        integral j
        double sum_

    for i in range(n_samples):
        sum_ = 0.0

        for j in range(X_indptr[i], X_indptr[i + 1]):
            sum_ += (X_data[j] * X_data[j])

        if sum_ == 0.0:
            # do not normalize empty rows (can happen if CSR is not pruned
            # correctly)
            continue

        sum_ = sqrt(sum_)

        for j in range(X_indptr[i], X_indptr[i + 1]):
            X_data[j] /= sum_


def assign_rows_csr(
    X,
    const intptr_t[:] X_rows,
    const intptr_t[:] out_rows,
    floating[:, ::1] out,
):
    # [FUNCTION DOC]
    # Behavior: zero the rows out[out_rows], then for each i scatter the stored
    # entries of X row X_rows[i] into out[out_rows[i], column]. Equivalent to
    # out[out_rows] = X[X_rows].toarray() without temporaries. Loops run without
    # the GIL.
    # Inputs: X (CSR with int32 indices/indptr and data dtype matching out),
    # X_rows and out_rows (intp arrays of equal length), out (C-contiguous 2d,
    # n_features columns).
    # Outputs: None; out modified in place.
    # Exceptions: ValueError if X_rows and out_rows lengths differ; buffer errors
    # for int64 indices or dtype mismatch. Row indices are not bounds checked.
    """Densify selected rows of a CSR matrix into a preallocated array.

    Like out[out_rows] = X[X_rows].toarray() but without copying.
    No-copy supported for both dtype=np.float32 and dtype=np.float64.

    Parameters
    ----------
    X : scipy.sparse.csr_array, shape=(n_samples, n_features)
    X_rows : array, dtype=np.intp, shape=n_rows
    out_rows : array, dtype=np.intp, shape=n_rows
    out : array, shape=(arbitrary, n_features)
    """
    cdef:
        # intptr_t (npy_intp, np.intp in Python) is what np.where returns,
        # but int is what scipy.sparse uses.
        intp_t i, ind, j, k
        intptr_t rX
        const floating[:] data = X.data
        const int32_t[:] indices = X.indices
        const int32_t[:] indptr = X.indptr

    if X_rows.shape[0] != out_rows.shape[0]:
        raise ValueError("cannot assign %d rows to %d"
                         % (X_rows.shape[0], out_rows.shape[0]))

    with nogil:
        for k in range(out_rows.shape[0]):
            out[out_rows[k]] = 0.0

        for i in range(X_rows.shape[0]):
            rX = X_rows[i]
            for ind in range(indptr[rX], indptr[rX + 1]):
                j = indices[ind]
                out[out_rows[i], j] = data[ind]


def csr_matmul_csr_to_dense(
    const floating[:] a_data,
    const integral[:] a_indices,
    const integral[:] a_indptr,
    const floating[:] b_data,
    const integral2[:] b_indices,
    const integral2[:] b_indptr,
    floating[:, :] out,
    uint64_t n1,
    uint64_t n2,
    uint64_t n3,
):
    # [FUNCTION DOC]
    # Behavior: dense result of the CSR product a @ b using Gustavson's row-wise
    # algorithm: for each row i of a, zero out[i, :n3], then for each stored
    # a[i, k] add a[i, k] * b[k, j] to out[i, j] for each stored b[k, j].
    # Inputs: CSR arrays of a (shape (n1, n2)) and b (shape (n2, n3)), with
    # independent index dtypes, floating data of the same dtype; out, a writable
    # (n1, n3) array; n1, n2 (unused), n3.
    # Outputs: None; out overwritten in place. Exceptions: buffer errors; no
    # shape validation. Runs with the GIL held.
    """Computes a @ b for sparse csr a and b, returns dense array.

    The shape of `a` is `(n1, n2)` and the shape of `b` is `(n2, n3)`.

    See also
    Gamma: Leveraging Gustavson's Algorithm to Accelerate Sparse Matrix Multiplication
    https://dl.acm.org/doi/pdf/10.1145/3445814.3446702
    """
    cdef uint64_t i
    cdef uint64_t j
    cdef integral2 j_ind
    cdef uint64_t k
    cdef integral k_ind
    cdef floating a_value

    for i in range(n1):
        for j in range(n3):
            out[i, j] = 0
        for k_ind in range(a_indptr[i], a_indptr[i + 1]):  # n2
            k = a_indices[k_ind]
            a_value = a_data[k_ind]
            for j_ind in range(b_indptr[k], b_indptr[k + 1]):  # n3
                j = b_indices[j_ind]
                # out[i, j] += a[i, k] * b[k, j]
                out[i, j] += a_value * b_data[j_ind]
