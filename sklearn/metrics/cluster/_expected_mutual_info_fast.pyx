# [MODULE DOC]
# Compiled computation of the expected mutual information (EMI) between two
# clusterings under a hypergeometric (permutation) model of randomness, used by
# adjusted_mutual_info_score in sklearn.metrics.cluster._supervised.
# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

from libc.math cimport exp, lgamma

from sklearn.utils._typedefs cimport float64_t, int64_t

import numpy as np
from scipy.special import gammaln


def expected_mutual_information(contingency, int64_t n_samples):
    # [FUNCTION DOC]
    # HELPER: computes the EMI term used to adjust mutual information for chance in
    #   adjusted_mutual_info_score.
    # Behavior: with row sums a_i and column sums b_j of the contingency table and
    #   N = n_samples, returns the sum over i, j and over n from max(1, a_i+b_j-N)
    #   to min(a_i, b_j) of (n/N) * log(N*n/(a_i*b_j)) * P(n), where P(n) is the
    #   hypergeometric probability a_i! b_j! (N-a_i)! (N-b_j)! /
    #   (N! n! (a_i-n)! (b_j-n)! (N-a_i-b_j+n)!) evaluated in log space via log-gamma
    #   (vectorized precomputation with scipy gammaln, per-term libc lgamma) to avoid
    #   overflow. Returns 0.0 early if either labeling has a single cluster.
    # Inputs: contingency: 2-D array-like (dense or scipy sparse) of shape
    #   (n_classes, n_clusters) with non-negative integer counts summing to
    #   n_samples; n_samples: total number of samples (int64).
    # Outputs: float, the expected mutual information in nats.
    # Exceptions: none explicitly raised; empty rows/columns would give log(0) of a
    #   zero marginal (numpy warning, -inf) but such entries contribute no terms
    #   since their summation range is empty.
    # Side effects: allocates temporary numpy arrays; holds the GIL throughout.
    """Calculate the expected mutual information for two labelings."""
    cdef:
        float64_t emi = 0
        int64_t n_rows, n_cols
        float64_t term2, term3, gln
        int64_t[::1] a_view, b_view
        float64_t[::1] term1
        float64_t[::1] gln_a, gln_b, gln_Na, gln_Nb, gln_Nnij, log_Nnij
        float64_t[::1] log_a, log_b
        Py_ssize_t i, j, nij
        int64_t start, end

    n_rows, n_cols = contingency.shape
    a = np.ravel(contingency.sum(axis=1).astype(np.int64, copy=False))
    b = np.ravel(contingency.sum(axis=0).astype(np.int64, copy=False))
    a_view = a
    b_view = b

    # any labelling with zero entropy implies EMI = 0
    if a.size == 1 or b.size == 1:
        return 0.0

    # There are three major terms to the EMI equation, which are multiplied to
    # and then summed over varying nij values.
    # While nijs[0] will never be used, having it simplifies the indexing.
    nijs = np.arange(0, max(np.max(a), np.max(b)) + 1, dtype='float')
    nijs[0] = 1  # Stops divide by zero warnings. As its not used, no issue.
    # term1 is nij / N
    term1 = nijs / n_samples
    # term2 is log((N*nij) / (a * b)) == log(N * nij) - log(a * b)
    log_a = np.log(a)
    log_b = np.log(b)
    # term2 uses log(N * nij) = log(N) + log(nij)
    log_Nnij = np.log(n_samples) + np.log(nijs)
    # term3 is large, and involved many factorials. Calculate these in log
    # space to stop overflows.
    gln_a = gammaln(a + 1)
    gln_b = gammaln(b + 1)
    gln_Na = gammaln(n_samples - a + 1)
    gln_Nb = gammaln(n_samples - b + 1)
    gln_Nnij = gammaln(nijs + 1) + gammaln(n_samples + 1)

    # emi itself is a summation over the various values.
    for i in range(n_rows):
        for j in range(n_cols):
            start = max(1, a_view[i] - n_samples + b_view[j])
            end = min(a_view[i], b_view[j]) + 1
            for nij in range(start, end):
                term2 = log_Nnij[nij] - log_a[i] - log_b[j]
                # Numerators are positive, denominators are negative.
                gln = (gln_a[i] + gln_b[j] + gln_Na[i] + gln_Nb[j]
                       - gln_Nnij[nij] - lgamma(a_view[i] - nij + 1)
                       - lgamma(b_view[j] - nij + 1)
                       - lgamma(n_samples - a_view[i] - b_view[j] + nij + 1))
                term3 = exp(gln)
                emi += (term1[nij] * term2 * term3)
    return emi
