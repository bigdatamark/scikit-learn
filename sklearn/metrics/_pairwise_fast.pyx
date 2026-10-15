# [MODULE DOC]
# Compiled kernels backing sklearn.metrics.pairwise: the dense additive chi-squared
# kernel loop (used by additive_chi2_kernel) and pairwise L1 (manhattan) distances
# between two CSR sparse matrices (used by manhattan_distances for sparse input).
# Both functions write into a caller-allocated output array and return nothing.
# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

from cython cimport floating
from cython.parallel cimport prange
from libc.math cimport fabs

from sklearn.utils._typedefs cimport intp_t

from sklearn.utils._openmp_helpers import _openmp_effective_n_threads


def _chi2_kernel_fast(floating[:, :] X,
                      floating[:, :] Y,
                      floating[:, :] result):
    # [FUNCTION DOC]
    # HELPER: inner loop of additive_chi2_kernel in sklearn.metrics.pairwise.
    # Behavior: for every pair (row i of X, row j of Y) computes
    #   negative sum over features k of (X[i,k]-Y[j,k])^2 / (X[i,k]+Y[j,k]),
    # skipping features where the denominator X[i,k]+Y[j,k] is exactly zero.
    # Accumulation is done in double precision, then cast to the output dtype.
    # Inputs: X (n_samples_X, n_features), Y (n_samples_Y, n_features), both of the
    #   same floating dtype (float32 or float64 fused type); features are expected
    #   to be non-negative (not checked here). result: (n_samples_X, n_samples_Y)
    #   output array of the same dtype.
    # Outputs: result[i, j] is overwritten; returns None.
    # Exceptions: none beyond argument type/buffer errors at call time; the loop
    #   runs without the GIL and performs no bounds checks of its own beyond the
    #   module defaults (shape mismatches between X/Y features are not validated).
    # Side effects: writes result in place; releases the GIL; single-threaded.
    cdef intp_t i, j, k
    cdef intp_t n_samples_X = X.shape[0]
    cdef intp_t n_samples_Y = Y.shape[0]
    cdef intp_t n_features = X.shape[1]
    cdef double res, nom, denom

    with nogil:
        for i in range(n_samples_X):
            for j in range(n_samples_Y):
                res = 0
                for k in range(n_features):
                    denom = (X[i, k] - Y[j, k])
                    nom = (X[i, k] + Y[j, k])
                    if nom != 0:
                        res += denom * denom / nom
                result[i, j] = -res


def _sparse_manhattan(
    const floating[::1] X_data,
    const int[:] X_indices,
    const int[:] X_indptr,
    const floating[::1] Y_data,
    const int[:] Y_indices,
    const int[:] Y_indptr,
    double[:, ::1] D,
):
    # [FUNCTION DOC]
    # HELPER: sparse branch of manhattan_distances in sklearn.metrics.pairwise.
    # Behavior: for each row px of X and row py of Y (CSR), merges the two sorted
    #   column-index lists like a merge step of merge sort: matching columns add
    #   |x - y|, a column present in only one row adds the absolute value of that
    #   entry; once one row is exhausted the remaining entries of the other row
    #   are added. Sum is accumulated in double and stored in D[px, py].
    # Inputs: X_data/X_indices/X_indptr and Y_data/Y_indices/Y_indptr are the CSR
    #   arrays of X and Y (same floating dtype); column indices within each row must
    #   be sorted and without duplicates (caller must call sort_indices /
    #   sum_duplicates), otherwise results are wrong. D: C-contiguous float64 array
    #   of shape (n_rows_X, n_rows_Y), preallocated.
    # Outputs: D overwritten in place; returns None.
    # Exceptions: none raised inside the parallel nogil loop.
    # Side effects: writes D; releases the GIL and parallelizes over rows of X with
    #   OpenMP using _openmp_effective_n_threads() threads.
    # KNOWN ISSUE: the docstring usage example calls np.zeros(X.shape[0], Y.shape[0])
    #   (shape passed as two arguments instead of a tuple), which would not create a
    #   2-D float64 array; D must be np.zeros((n_X, n_Y)).
    """Pairwise L1 distances for CSR matrices.

    Usage:
    >>> D = np.zeros(X.shape[0], Y.shape[0])
    >>> _sparse_manhattan(X.data, X.indices, X.indptr,
    ...                   Y.data, Y.indices, Y.indptr,
    ...                   D)
    """
    cdef intp_t px, py, i, j, ix, iy
    cdef double d = 0.0

    cdef int m = D.shape[0]
    cdef int n = D.shape[1]

    cdef int X_indptr_end = 0
    cdef int Y_indptr_end = 0

    cdef int num_threads = _openmp_effective_n_threads()

    # We scan the matrices row by row.
    # Given row px in X and row py in Y, we find the positions (i and j
    # respectively), in .indices where the indices for the two rows start.
    # If the indices (ix and iy) are the same, the corresponding data values
    # are processed and the cursors i and j are advanced.
    # If not, the lowest index is considered. Its associated data value is
    # processed and its cursor is advanced.
    # We proceed like this until one of the cursors hits the end for its row.
    # Then we process all remaining data values in the other row.

    # Below the avoidance of inplace operators is intentional.
    # When prange is used, the inplace operator has a special meaning, i.e. it
    # signals a "reduction"

    for px in prange(m, nogil=True, num_threads=num_threads):
        X_indptr_end = X_indptr[px + 1]
        for py in range(n):
            Y_indptr_end = Y_indptr[py + 1]
            i = X_indptr[px]
            j = Y_indptr[py]
            d = 0.0
            while i < X_indptr_end and j < Y_indptr_end:
                ix = X_indices[i]
                iy = Y_indices[j]

                if ix == iy:
                    d = d + fabs(X_data[i] - Y_data[j])
                    i = i + 1
                    j = j + 1
                elif ix < iy:
                    d = d + fabs(X_data[i])
                    i = i + 1
                else:
                    d = d + fabs(Y_data[j])
                    j = j + 1

            if i == X_indptr_end:
                while j < Y_indptr_end:
                    d = d + fabs(Y_data[j])
                    j = j + 1
            else:
                while i < X_indptr_end:
                    d = d + fabs(X_data[i])
                    i = i + 1

            D[px, py] = d
