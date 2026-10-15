# [MODULE DOC]
# Compiled inner loop of the coordinate descent ("cd") solver of
# sklearn.decomposition.NMF / non_negative_factorization.
# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

from cython cimport floating
from libc.math cimport fabs


def _update_cdnmf_fast(floating[:, ::1] W, floating[:, :] HHt,
                       floating[:, :] XHt, Py_ssize_t[::1] permutation):
    # [FUNCTION DOC]
    # HELPER: one coordinate descent sweep for NMF (used for both W and H, H via
    # transposed inputs).
    # Behavior: for each component t in the given permutation order and each row i,
    # computes the gradient of 0.5*||X - W H||^2 wrt W[i,t] as (W HHt)[i,t] - XHt[i,t],
    # adds the absolute projected gradient (gradient clipped to <= 0 when W[i,t] is 0)
    # to the violation, then applies a Newton step W[i,t] = max(W[i,t] - grad/HHt[t,t], 0)
    # when HHt[t,t] != 0. Updates are immediately visible to later coordinates.
    # Inputs: W (n_samples, n_components) C-contiguous; HHt (n_components,
    # n_components) = H H^T; XHt (n_samples, n_components) = X H^T; permutation of
    # component indices.
    # Outputs: the summed projected-gradient violation (float). W updated in place.
    # Exceptions: none (runs without the GIL). Side effects: mutates W; releases GIL.
    cdef:
        floating violation = 0
        Py_ssize_t n_components = W.shape[1]
        Py_ssize_t n_samples = W.shape[0]  # n_features for H update
        floating grad, pg, hess
        Py_ssize_t i, r, s, t

    with nogil:
        for s in range(n_components):
            t = permutation[s]

            for i in range(n_samples):
                # gradient = GW[t, i] where GW = np.dot(W, HHt) - XHt
                grad = -XHt[i, t]

                for r in range(n_components):
                    grad += HHt[t, r] * W[i, r]

                # projected gradient
                pg = min(0., grad) if W[i, t] == 0 else grad
                violation += fabs(pg)

                # Hessian
                hess = HHt[t, t]

                if hess != 0:
                    W[i, t] = max(W[i, t] - grad / hess, 0.)

    return violation
