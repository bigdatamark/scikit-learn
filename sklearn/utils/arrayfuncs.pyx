# [MODULE DOC]
# Small array helpers implemented in Cython: smallest positive value of a
# float array (used by LARS / lasso path code), a memory-light check whether
# any row of a 2d array equals a constant (used in validation), and a
# Cholesky-factor row/column deletion via Givens rotations (used by lars_path
# when variables leave the active set).
"""A small collection of auxiliary functions that operate on arrays."""

from cython cimport floating
from libc.math cimport fabs
from libc.float cimport DBL_MAX, FLT_MAX

from sklearn.utils._cython_blas cimport _copy, _rotg, _rot


ctypedef fused real_numeric:
    short
    int
    long
    long long
    float
    double


def min_pos(const floating[:] X):
    # [FUNCTION DOC]
    # Behavior: linear scan returning the smallest element strictly greater than 0;
    # if none exists, returns FLT_MAX for float32 input or DBL_MAX for float64.
    # NaNs are ignored (comparisons are false).
    # Inputs: X, 1d float32/float64 array (strided allowed).
    # Outputs: Python float. Exceptions: buffer dtype errors (e.g. integer input).
    """Find the minimum value of an array over positive values.

    Returns the maximum representable value of the input dtype if none of the
    values are positive.

    Parameters
    ----------
    X : ndarray of shape (n,)
        Input array.

    Returns
    -------
    min_val : float
        The smallest positive value in the array, or the maximum representable value
         of the input dtype if no positive values are found.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.utils.arrayfuncs import min_pos
    >>> X = np.array([0, -1, 2, 3, -4, 5])
    >>> min_pos(X)
    2.0
    """
    cdef Py_ssize_t i
    cdef floating min_val = FLT_MAX if floating is float else DBL_MAX
    for i in range(X.size):
        if 0. < X[i] < min_val:
            min_val = X[i]
    return min_val


def _all_with_any_reduction_axis_1(real_numeric[:, :] array, real_numeric value):
    # [FUNCTION DOC]
    # Behavior: for each row, compare elements to value until one differs; return
    # True as soon as a row has all elements equal, else False. An array with
    # rows but zero columns returns True; zero rows returns False.
    # Inputs: array, 2d memoryview of a numeric fused type (writable buffer
    # required since the memoryview is not const); value of the same type.
    # Outputs: bool. Exceptions: buffer dtype errors.
    """Check whether any row contains all values equal to `value`.

    It is equivalent to `np.any(np.all(X == value, axis=1))`, but it avoids to
    materialize the temporary boolean matrices in memory.

    Parameters
    ----------
    array: array-like
        The array to be checked.
    value: short, int, long, float, or double
        The value to use for the comparison.

    Returns
    -------
    any_all_equal: bool
        Whether or not any rows contains all values equal to `value`.
    """
    cdef Py_ssize_t i, j

    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            if array[i, j] != value:
                break
        else:  # no break
            return True
    return False


# General Cholesky Delete.
# Remove an element from the cholesky factorization
# m = columns
# n = rows
#
# TODO: put transpose as an option
def cholesky_delete(floating[:, :] L, int go_out):
    # [FUNCTION DOC]
    # Behavior: given the lower-triangular Cholesky factor L of a Gram matrix
    # (n x n, stored C-contiguously with row stride m elements), remove variable
    # go_out: first shift rows go_out+1..n-1 up by one (copying the i+2 leading
    # entries of each row), then restore lower-triangular form by computing, for
    # each affected row i, a Givens rotation zeroing the superdiagonal element
    # (i, i+1), flipping its sign to keep the diagonal non-negative, and applying
    # it to columns i and i+1 of the remaining rows below.
    # Inputs: L (writable 2d float memoryview, rows contiguous), go_out (index of
    # the row/column to delete, 0 <= go_out < n).
    # Outputs: None; L modified in place so its leading (n-1) x (n-1) block is the
    # updated factor; the last row is left as garbage.
    # Exceptions: buffer errors only; no bounds checks on go_out.
    cdef:
        int n = L.shape[0]
        int m = L.strides[0]
        floating c, s
        floating *L1
        int i

    if floating is float:
        m /= sizeof(float)
    else:
        m /= sizeof(double)

    # delete row go_out
    L1 = &L[0, 0] + (go_out * m)
    for i in range(go_out, n-1):
        _copy(i + 2, L1 + m, 1, L1, 1)
        L1 += m

    L1 = &L[0, 0] + (go_out * m)
    for i in range(go_out, n-1):
        _rotg(L1 + i, L1 + i + 1, &c, &s)
        if L1[i] < 0:
            # Diagonals cannot be negative
            L1[i] = fabs(L1[i])
            c = -c
            s = -s

        L1[i + 1] = 0.  # just for cleanup
        L1 += m

        _rot(n - i - 2, L1 + i, m, L1 + i + 1, m, c, s)
