# [MODULE DOC]
# Compiled degree-2/3 polynomial and interaction expansion of CSR matrices for
# sklearn.preprocessing.PolynomialFeatures (Nystrom & Hughes K-simplex method):
# computes output nnz counts and fills output CSR arrays using closed-form
# column indices for products of non-zero features only. Uses 128-bit integers
# for index arithmetic where the compiler supports them.
# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

from sklearn.utils._typedefs cimport uint8_t, int64_t, intp_t

ctypedef uint8_t FLAG_t

# We use the following verbatim block to determine whether the current
# platform's compiler supports 128-bit integer values intrinsically.
# This should work for GCC and CLANG on 64-bit architectures, but doesn't for
# MSVC on any architecture. We prefer to use 128-bit integers when possible
# because the intermediate calculations have a non-trivial risk of overflow. It
# is, however, very unlikely to come up on an average use case, hence 64-bit
# integers (i.e. `long long`) are "good enough" for most common cases. There is
# not much we can do to efficiently mitigate  the overflow risk on the Windows
# platform at this time. Consider this a "best effort" design decision that
# could be revisited later in case someone comes up with a safer option that
# does not hurt the performance of the common cases.
# See `test_sizeof_LARGEST_INT_t()`for more information on exact type expectations.
cdef extern from *:
    """
    #ifdef __SIZEOF_INT128__
        typedef __int128 LARGEST_INT_t;
    #elif (__clang__ || __EMSCRIPTEN__) && !__i386__
        typedef _BitInt(128) LARGEST_INT_t;
    #else
        typedef long long LARGEST_INT_t;
    #endif
    """
    ctypedef long long LARGEST_INT_t


# Determine the size of `LARGEST_INT_t` at runtime.
# Used in `test_sizeof_LARGEST_INT_t`.
def _get_sizeof_LARGEST_INT_t():
    # [FUNCTION DOC]
    # HELPER: used by tests to check which integer width LARGEST_INT_t has.
    # Inputs: none. Outputs: size in bytes (16 or 8). Exceptions/side effects: none.
    return sizeof(LARGEST_INT_t)


# TODO: use `{int,float}{32,64}_t` when cython#5230 is resolved:
# https://github.com/cython/cython/issues/5230
ctypedef fused DATA_t:
    float
    double
    int
    long long
# INDEX_{A,B}_t are defined to generate a proper Cartesian product
# of types through Cython fused-type expansion.
ctypedef fused INDEX_A_t:
    signed int
    signed long long
ctypedef fused INDEX_B_t:
    signed int
    signed long long

cdef inline int64_t _deg2_column(
    LARGEST_INT_t n_features,
    LARGEST_INT_t i,
    LARGEST_INT_t j,
    FLAG_t interaction_only
) nogil:
    # [FUNCTION DOC]
    # HELPER: column index of the product of features i <= j (i < j if
    # interaction_only) in the degree-2 block, via triangular-number formulas.
    # Inputs: n_features, i, j (wide ints), interaction_only flag.
    # Outputs: the 0-based column offset within the degree-2 block, as int64.
    # Exceptions: none (nogil). Side effects: none.
    """Compute the index of the column for a degree 2 expansion

    n_features is the dimensionality of the input data, i and j are the indices
    for the columns involved in the expansion.
    """
    if interaction_only:
        return n_features * i - i * (i + 3) / 2 - 1 + j
    else:
        return n_features * i - i* (i + 1) / 2 + j


cdef inline int64_t _deg3_column(
    LARGEST_INT_t n_features,
    LARGEST_INT_t i,
    LARGEST_INT_t j,
    LARGEST_INT_t k,
    FLAG_t interaction_only
) nogil:
    # [FUNCTION DOC]
    # HELPER: column index of the product i <= j <= k (strict if interaction_only)
    # in the degree-3 block via tetrahedral-number formulas.
    # Inputs: n_features, i, j, k, interaction_only.
    # Outputs: column offset within the degree-3 block, as int64.
    # Exceptions: none (nogil). Side effects: none.
    """Compute the index of the column for a degree 3 expansion

    n_features is the dimensionality of the input data, i, j and k are the indices
    for the columns involved in the expansion.
    """
    if interaction_only:
        return (
            (
                (3 * n_features) * (n_features * i - i**2)
                + i * (i**2 + 11) - (3 * j) * (j + 3)
            ) / 6 + i**2 + n_features * (j - 1 - 2 * i) + k
        )
    else:
        return (
            (
                (3 * n_features) * (n_features * i - i**2)
                + i ** 3 - i - (3 * j) * (j + 1)
            ) / 6 + n_features * j + k
        )


def py_calc_expanded_nnz_deg2(n, interaction_only):
    # [FUNCTION DOC]
    # HELPER: arbitrary-precision (Python int) fallback for _calc_expanded_nnz.
    # Behavior: n(n+1)/2 minus n if interaction_only (degree-2 terms from n nonzeros).
    # Inputs: n, interaction_only. Outputs: int. Exceptions/side effects: none.
    return n * (n + 1) // 2 - interaction_only * n


def py_calc_expanded_nnz_deg3(n, interaction_only):
    # [FUNCTION DOC]
    # HELPER: Python-int fallback for degree 3: n(n^2+3n+2)/6 minus n^2 if
    # interaction_only.
    # Inputs: n, interaction_only. Outputs: int. Exceptions/side effects: none.
    return n * (n**2 + 3 * n + 2) // 6 - interaction_only * n**2


cpdef int64_t _calc_expanded_nnz(
    LARGEST_INT_t n,
    FLAG_t interaction_only,
    LARGEST_INT_t degree
):
    # [FUNCTION DOC]
    # HELPER: number of degree-2 or degree-3 products produced by a row with n
    # non-zeros (used to size outputs).
    # Behavior: same formulas as the py_ fallbacks; uses native arithmetic when the
    # input is below the overflow threshold, otherwise the Python-int fallback.
    # Inputs: n non-zeros; interaction_only; degree (2, anything else means 3).
    # Outputs: int64 count. Exceptions: may raise when converting an oversized
    # Python result to int64. Side effects: none.
    # KNOWN ISSUE: the native fast path is taken only when LARGEST_INT_t is narrower
    # than 16 bytes; with 128-bit support (where native math is safest) it always
    # falls back to slow Python integers, the inverse of what the comment implies
    # (correct results, just slower).
    """
    Calculates the number of non-zero interaction terms generated by the
    non-zero elements of a single row.
    """
    # This is the maximum value before the intermediate computation
    # d**2 + d overflows
    # Solution to d**2 + d = maxint64
    # SymPy: solve(x**2 + x - int64_max, x)
    cdef int64_t MAX_SAFE_INDEX_CALC_DEG2 = 3037000499

    # This is the maximum value before the intermediate computation
    # d**3 + 3 * d**2 + 2*d overflows
    # Solution to d**3 + 3 * d**2 + 2*d = maxint64
    # SymPy: solve(x * (x**2 + 3 * x + 2) - int64_max, x)
    cdef int64_t MAX_SAFE_INDEX_CALC_DEG3 = 2097151

    if degree == 2:
        # Only need to check when not using 128-bit integers
        if sizeof(LARGEST_INT_t) < 16 and n <= MAX_SAFE_INDEX_CALC_DEG2:
            return n * (n + 1) / 2 - interaction_only * n
        return <int64_t> py_calc_expanded_nnz_deg2(n, interaction_only)
    else:
        # Only need to check when not using 128-bit integers
        if sizeof(LARGEST_INT_t) < 16 and n <= MAX_SAFE_INDEX_CALC_DEG3:
            return n * (n**2 + 3 * n + 2) / 6 - interaction_only * n**2
        return <int64_t> py_calc_expanded_nnz_deg3(n, interaction_only)

cpdef int64_t _calc_total_nnz(
    INDEX_A_t[:] indptr,
    FLAG_t interaction_only,
    int64_t degree,
):
    # [FUNCTION DOC]
    # HELPER: total output nnz for one degree, summing _calc_expanded_nnz over the
    # per-row non-zero counts given by indptr.
    # Inputs: indptr (int32/int64 view); interaction_only; degree.
    # Outputs: int64 total. Exceptions: as _calc_expanded_nnz. Side effects: none.
    """
    Calculates the number of non-zero interaction terms generated by the
    non-zero elements across all rows for a single degree.
    """
    cdef int64_t total_nnz=0
    cdef intp_t row_idx
    for row_idx in range(len(indptr) - 1):
        total_nnz += _calc_expanded_nnz(
            indptr[row_idx + 1] - indptr[row_idx],
            interaction_only,
            degree
        )
    return total_nnz


cpdef void _csr_polynomial_expansion(
    const DATA_t[:] data,           # IN READ-ONLY
    const INDEX_A_t[:] indices,     # IN READ-ONLY
    const INDEX_A_t[:] indptr,      # IN READ-ONLY
    INDEX_A_t n_features,
    DATA_t[:] result_data,          # OUT
    INDEX_B_t[:] result_indices,    # OUT
    INDEX_B_t[:] result_indptr,     # OUT
    FLAG_t interaction_only,
    FLAG_t degree
):
    # [FUNCTION DOC]
    # HELPER: called by PolynomialFeatures.transform for sparse input per degree.
    # Behavior: for each row, iterates over ordered pairs (or triples) of the row's
    # stored entries, with j starting at i (or i+1 when interaction_only) and k
    # likewise from j; writes the closed-form column index and product of values to
    # the outputs sequentially and sets result_indptr[row+1] from the row's count.
    # Assumes column indices within each row are sorted so i <= j <= k as features.
    # Inputs: data, indices, indptr of input CSR (read-only); n_features; output
    # result_data, result_indices, result_indptr preallocated with sizes from
    # _calc_total_nnz and n_rows+1; interaction_only (0/1); degree (2 or 3; any
    # value other than 2 is treated as 3).
    # Outputs: none returned; output arrays filled in place.
    # Exceptions: none (runs without the GIL, no bounds checking).
    # Side effects: mutates outputs; releases the GIL.
    """
    Perform a second or third degree polynomial or interaction expansion on a
    compressed sparse row (CSR) matrix. The method used only takes products of
    non-zero features. For a matrix with density :math:`d`, this results in a
    speedup on the order of :math:`(1/d)^k` where :math:`k` is the degree of
    the expansion, assuming all rows are of similar density.

    Parameters
    ----------
    data : memory view on nd-array
        The "data" attribute of the input CSR matrix.

    indices : memory view on nd-array
        The "indices" attribute of the input CSR matrix.

    indptr : memory view on nd-array
        The "indptr" attribute of the input CSR matrix.

    n_features : int
        The dimensionality of the input CSR matrix.

    result_data : nd-array
        The output CSR matrix's "data" attribute.
        It is modified by this routine.

    result_indices : nd-array
        The output CSR matrix's "indices" attribute.
        It is modified by this routine.

    result_indptr : nd-array
        The output CSR matrix's "indptr" attribute.
        It is modified by this routine.

    interaction_only : int
        0 for a polynomial expansion, 1 for an interaction expansion.

    degree : int
        The degree of the expansion. This must be either 2 or 3.

    References
    ----------
    "Leveraging Sparsity to Speed Up Polynomial Feature Expansions of CSR
    Matrices Using K-Simplex Numbers" by Andrew Nystrom and John Hughes.
    """

    # Make the arrays that will form the CSR matrix of the expansion.
    cdef INDEX_A_t row_i, row_starts, row_ends, i, j, k, i_ptr, j_ptr, k_ptr
    cdef INDEX_B_t expanded_index=0, num_cols_in_row, col
    with nogil:
        result_indptr[0] = indptr[0]
        for row_i in range(indptr.shape[0]-1):
            row_starts = indptr[row_i]
            row_ends = indptr[row_i + 1]
            num_cols_in_row = 0
            for i_ptr in range(row_starts, row_ends):
                i = indices[i_ptr]
                for j_ptr in range(i_ptr + interaction_only, row_ends):
                    j = indices[j_ptr]
                    if degree == 2:
                        col = <INDEX_B_t> _deg2_column(
                            n_features,
                            i, j,
                            interaction_only
                        )
                        result_indices[expanded_index] = col
                        result_data[expanded_index] = (
                            data[i_ptr] * data[j_ptr]
                        )
                        expanded_index += 1
                        num_cols_in_row += 1
                    else:
                        # degree == 3
                        for k_ptr in range(j_ptr + interaction_only, row_ends):
                            k = indices[k_ptr]
                            col = <INDEX_B_t> _deg3_column(
                                n_features,
                                i, j, k,
                                interaction_only
                            )
                            result_indices[expanded_index] = col
                            result_data[expanded_index] = (
                                data[i_ptr] * data[j_ptr] * data[k_ptr]
                            )
                            expanded_index += 1
                            num_cols_in_row += 1

            result_indptr[row_i+1] = result_indptr[row_i] + num_cols_in_row
    return
