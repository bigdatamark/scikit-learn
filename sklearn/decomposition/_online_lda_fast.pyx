# [MODULE DOC]
# Compiled helpers for sklearn.decomposition.LatentDirichletAllocation (online
# variational Bayes): mean absolute change for convergence checks, Dirichlet
# expectations of log variables, and a fast approximate digamma (psi).
import numpy as np


from cython cimport floating
from libc.math cimport exp, fabs, log

from sklearn.utils._typedefs cimport float64_t, intp_t


def mean_change(const floating[:] arr_1, const floating[:] arr_2):
    # [FUNCTION DOC]
    # HELPER: convergence check in LDA's E-step.
    # Behavior: mean of |arr_1[i] - arr_2[i]| computed in double precision.
    # Inputs: two 1D read-only float arrays of equal length (arr_2 at least as long).
    # Outputs: the mean as a Python float.
    # Exceptions: none; empty input yields NaN (C division). Side effects: none.
    """Calculate the mean difference between two arrays.

    Equivalent to np.abs(arr_1 - arr2).mean().
    """

    cdef float64_t total, diff
    cdef intp_t i, size

    size = arr_1.shape[0]
    total = 0.0
    for i in range(size):
        diff = fabs(arr_1[i] - arr_2[i])
        total += diff

    return total / size


def _dirichlet_expectation_1d(
    floating[:] doc_topic,
    floating doc_topic_prior,
    floating[:] out
):
    # [FUNCTION DOC]
    # HELPER: per-document update in LDA's E-step.
    # Behavior: adds doc_topic_prior to each doc_topic entry in place, then writes
    # exp(psi(doc_topic[i]) - psi(sum of doc_topic)) to out.
    # Inputs: doc_topic 1D float array (unnormalized topic counts); doc_topic_prior
    # scalar; out 1D float array of the same length.
    # Outputs: none returned; out filled. Exceptions: none.
    # Side effects: mutates doc_topic and out in place.
    """Dirichlet expectation for a single sample:
        exp(E[log(theta)]) for theta ~ Dir(doc_topic)
    after adding doc_topic_prior to doc_topic, in-place.

    Equivalent to
        doc_topic += doc_topic_prior
        out[:] = np.exp(psi(doc_topic) - psi(np.sum(doc_topic)))
    """

    cdef floating dt, psi_total, total
    cdef intp_t i, size

    size = doc_topic.shape[0]

    total = 0.0
    for i in range(size):
        dt = doc_topic[i] + doc_topic_prior
        doc_topic[i] = dt
        total += dt
    psi_total = psi(total)

    for i in range(size):
        out[i] = exp(psi(doc_topic[i]) - psi_total)


def _dirichlet_expectation_2d(const floating[:, :] arr):
    # [FUNCTION DOC]
    # HELPER: used by LDA to compute E[log theta] per row (no prior, no exp).
    # Behavior: for each row, psi(arr[i,j]) - psi(row sum).
    # Inputs: arr 2D read-only float array of positive values.
    # Outputs: new ndarray with the same shape/dtype. Exceptions: none.
    # Side effects: allocates the output.
    """Dirichlet expectation for multiple samples:
    E[log(theta)] for theta ~ Dir(arr).

    Equivalent to psi(arr) - psi(np.sum(arr, axis=1))[:, np.newaxis].

    Note that unlike _dirichlet_expectation_1d, this function doesn't compute
    the exp and doesn't add in the prior.
    """
    cdef floating row_total, psi_row_total
    cdef floating[:, :] d_exp
    cdef intp_t i, j, n_rows, n_cols

    n_rows = arr.shape[0]
    n_cols = arr.shape[1]

    d_exp = np.empty_like(arr)
    for i in range(n_rows):
        row_total = 0
        for j in range(n_cols):
            row_total += arr[i, j]
        psi_row_total = psi(row_total)

        for j in range(n_cols):
            d_exp[i, j] = psi(arr[i, j]) - psi_row_total

    return d_exp.base


# Psi function for positive arguments. Optimized for speed, not accuracy.
#
# After: J. Bernardo (1976). Algorithm AS 103: Psi (Digamma) Function.
# https://www.uv.es/~bernardo/1976AppStatist.pdf
cdef floating psi(floating x) noexcept nogil:
    # [FUNCTION DOC]
    # HELPER: digamma used by the Dirichlet expectation functions.
    # Behavior: for x <= 1e-6 returns the asymptote -EulerGamma - 1/x; otherwise uses
    # the recurrence psi(x) = psi(x+1) - 1/x to shift x to >= 6, then the asymptotic
    # series log(x) - 1/(2x) - 1/(12x^2) + 1/(120x^4) - 1/(252x^6).
    # Inputs: x, float (intended positive). Outputs: approximate psi(x).
    # Exceptions: none (noexcept, nogil). Side effects: none.
    # Note: non-positive x silently falls into the small-x branch (inaccurate).
    cdef double EULER = 0.577215664901532860606512090082402431
    if x <= 1e-6:
        # psi(x) = -EULER - 1/x + O(x)
        return -EULER - 1. / x

    cdef floating r, result = 0

    # psi(x + 1) = psi(x) + 1/x
    while x < 6:
        result -= 1. / x
        x += 1

    # psi(x) = log(x) - 1/(2x) - 1/(12x**2) + 1/(120x**4) - 1/(252x**6)
    #          + O(1/x**8)
    r = 1. / x
    result += log(x) - .5 * r
    r = r * r
    result -= r * ((1./12.) - r * ((1./120.) - r * (1./252.)))
    return result
