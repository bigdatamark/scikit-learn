# [MODULE DOC]
# Compiled kernels for sklearn.isotonic (IsotonicRegression / isotonic_regression).
# Provides an in-place weighted pool-adjacent-violators (PAVA) solver for
# non-decreasing fits, and a routine that collapses duplicate (sorted) X values
# into weighted-average targets before fitting. Both are fused over float32/float64.
# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

# Uses the pool adjacent violators algorithm (PAVA), with the
# enhancement of searching for the longest decreasing subsequence to
# pool at each step.

import numpy as np
from cython cimport floating


def _inplace_contiguous_isotonic_regression(floating[::1] y, floating[::1] w):
    # [FUNCTION DOC]
    # HELPER: core solver used by sklearn.isotonic.isotonic_regression.
    # Behavior: weighted PAVA producing the non-decreasing sequence minimizing
    # weighted squared error. Blocks are tracked with a target array: for an active
    # block spanning i..j, target[i]=j and target[j]=i; block start i stores the
    # block's total weight in w[i] and weighted mean in y[i]. Scanning from i, if the
    # next block's value is larger, move on; otherwise absorb the whole run of
    # non-increasing following blocks (a decreasing subsequence) into one block,
    # set its mean and weight, then step back to the previous block start (if any)
    # to re-check monotonicity. This makes it single pass, O(n). Finally every
    # block's mean is copied across all its positions.
    # Inputs: y, contiguous float array of length n (targets, already ordered by X);
    # w, contiguous float array of length n of positive sample weights.
    # Outputs: none returned; y is overwritten with the fitted values.
    # Exceptions: none in the main loop (runs without the GIL); zero total weight in a
    # pooled block yields NaN/inf silently (C division semantics).
    # Side effects: mutates y and w in place (w becomes block weights at block
    # starts, garbage otherwise); allocates a temporary index array; releases the GIL.
    cdef:
        Py_ssize_t n = y.shape[0], i, k
        floating prev_y, sum_wy, sum_w
        Py_ssize_t[::1] target = np.arange(n, dtype=np.intp)

    # target describes a list of blocks.  At any time, if [i..j] (inclusive) is
    # an active block, then target[i] := j and target[j] := i.

    # For "active" indices (block starts):
    # w[i] := sum{w_orig[j], j=[i..target[i]]}
    # y[i] := sum{y_orig[j]*w_orig[j], j=[i..target[i]]} / w[i]

    with nogil:
        i = 0
        while i < n:
            k = target[i] + 1
            if k == n:
                break
            if y[i] < y[k]:
                i = k
                continue
            sum_wy = w[i] * y[i]
            sum_w = w[i]
            while True:
                # We are within a decreasing subsequence.
                prev_y = y[k]
                sum_wy += w[k] * y[k]
                sum_w += w[k]
                k = target[k] + 1
                if k == n or prev_y < y[k]:
                    # Non-singleton decreasing subsequence is finished,
                    # update first entry.
                    y[i] = sum_wy / sum_w
                    w[i] = sum_w
                    target[i] = k - 1
                    target[k - 1] = i
                    if i > 0:
                        # Backtrack if we can.  This makes the algorithm
                        # single-pass and ensures O(n) complexity.
                        i = target[i - 1]
                    # Otherwise, restart from the same point.
                    break
        # Reconstruct the solution.
        i = 0
        while i < n:
            k = target[i] + 1
            y[i + 1 : k] = y[i]
            i = k


def _make_unique(const floating[::1] X,
                 const floating[::1] y,
                 const floating[::1] sample_weights):
    # [FUNCTION DOC]
    # HELPER: preprocessing for IsotonicRegression fit, merges duplicate X values.
    # Behavior: walks sorted X; consecutive values whose difference from the current
    # group's first value is below the dtype resolution are treated as the same
    # value. For each group outputs the group's first X, the summed weight, and the
    # weight-averaged y.
    # Inputs: X, y, sample_weights: contiguous read-only float arrays of equal length
    # n >= 1, X sorted ascending.
    # Outputs: tuple of three new arrays (x_unique, y_mean, weight_sum) of the group
    # count length, same float dtype as inputs.
    # Exceptions: IndexError if X is empty.
    # Side effects: allocates output arrays; no mutation of inputs.
    # KNOWN ISSUE: output buffers are sized by exact-equality unique count, while
    # grouping uses an eps tolerance; it is safe (tolerance only merges more) but
    # a group whose total weight is zero gives NaN (C division, no error raised).
    """Average targets for duplicate X, drop duplicates.

    Aggregates duplicate X values into a single X value where
    the target y is a (sample_weighted) average of the individual
    targets.

    Assumes that X is ordered, so that all duplicates follow each other.
    """
    unique_values = len(np.unique(X))

    if floating is float:
        dtype = np.float32
    else:
        dtype = np.float64

    cdef floating[::1] y_out = np.empty(unique_values, dtype=dtype)
    cdef floating[::1] x_out = np.empty_like(y_out)
    cdef floating[::1] weights_out = np.empty_like(y_out)

    cdef floating current_x = X[0]
    cdef floating current_y = 0
    cdef floating current_weight = 0
    cdef int i = 0
    cdef int j
    cdef floating x
    cdef int n_samples = len(X)
    cdef floating eps = np.finfo(dtype).resolution

    for j in range(n_samples):
        x = X[j]
        if x - current_x >= eps:
            # next unique value
            x_out[i] = current_x
            weights_out[i] = current_weight
            y_out[i] = current_y / current_weight
            i += 1
            current_x = x
            current_weight = sample_weights[j]
            current_y = y[j] * sample_weights[j]
        else:
            current_weight += sample_weights[j]
            current_y += y[j] * sample_weights[j]

    x_out[i] = current_x
    weights_out[i] = current_weight
    y_out[i] = current_y / current_weight
    return(
        np.asarray(x_out[:i+1]),
        np.asarray(y_out[:i+1]),
        np.asarray(weights_out[:i+1]),
    )
