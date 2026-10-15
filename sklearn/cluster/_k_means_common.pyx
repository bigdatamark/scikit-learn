# [MODULE DOC]
# Shared low-level helpers for the k-means implementations (Lloyd, Elkan,
# minibatch): Euclidean distance kernels for dense/dense and sparse(CSR)/dense
# pairs, inertia computation, relocation of empty clusters, center averaging,
# center-shift computation, and a label-permutation equality check. Also defines
# CHUNK_SIZE, the number of samples per chunk used by the parallel loops in the
# other k-means modules. All numeric routines are fused over float32/float64.
# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
from cython cimport floating
from cython.parallel cimport prange
from libc.math cimport sqrt

from sklearn.utils.extmath import row_norms


# Number of samples per data chunk defined as a global constant.
CHUNK_SIZE = 256


cdef floating _euclidean_dense_dense(
        const floating* a,  # IN
        const floating* b,  # IN
        int n_features,
        bint squared
) noexcept nogil:
    # [FUNCTION DOC]
    # HELPER: core distance kernel used by all k-means modules.
    # Behavior: sums squared coordinate differences of a and b over n_features, with
    # the main loop manually unrolled by 4 and a remainder loop; returns the sum if
    # squared is true, otherwise its square root.
    # Inputs: a, b pointers to n_features contiguous values; n_features >= 0; squared flag.
    # Outputs: the (squared) Euclidean distance.
    # Exceptions: none (noexcept nogil). Side effects: none.
    """Euclidean distance between a dense and b dense"""
    cdef:
        int i
        int n = n_features // 4
        int rem = n_features % 4
        floating result = 0

    # We manually unroll the loop for better cache optimization.
    for _ in range(n):
        result += (
            (a[0] - b[0]) * (a[0] - b[0]) +
            (a[1] - b[1]) * (a[1] - b[1]) +
            (a[2] - b[2]) * (a[2] - b[2]) +
            (a[3] - b[3]) * (a[3] - b[3])
        )
        a += 4
        b += 4

    for i in range(rem):
        result += (a[i] - b[i]) * (a[i] - b[i])

    return result if squared else sqrt(result)


def _euclidean_dense_dense_wrapper(
    const floating[::1] a,
    const floating[::1] b,
    bint squared
):
    # [FUNCTION DOC]
    # HELPER: Python-callable test wrapper around _euclidean_dense_dense.
    # Inputs: contiguous 1d arrays a and b (b at least as long as a), squared flag.
    # Outputs: the distance as a Python float. Exceptions: IndexError if a is empty.
    # Side effects: none.
    """Wrapper of _euclidean_dense_dense for testing purpose"""
    return _euclidean_dense_dense(&a[0], &b[0], a.shape[0], squared)


cdef floating _euclidean_sparse_dense(
        const floating[::1] a_data,  # IN
        const int[::1] a_indices,    # IN
        const floating[::1] b,       # IN
        floating b_squared_norm,
        bint squared
) noexcept nogil:
    # [FUNCTION DOC]
    # HELPER: distance kernel for sparse samples.
    # Behavior: uses the identity ||a-b||^2 = ||b||^2 + sum over nonzeros of
    # ((a_i - b_i)^2 - b_i^2), so only the nonzero entries of a are visited. The
    # result is clipped at 0 to absorb floating point cancellation, then square-rooted
    # unless squared is true.
    # Inputs: a_data / a_indices the values and column indices of one CSR row;
    # b dense vector; b_squared_norm the precomputed squared norm of b; squared flag.
    # Outputs: the (squared) Euclidean distance.
    # Exceptions: none (noexcept nogil). Side effects: none.
    """Euclidean distance between a sparse and b dense"""
    cdef:
        int nnz = a_indices.shape[0]
        int i
        floating tmp, bi
        floating result = 0.0

    for i in range(nnz):
        bi = b[a_indices[i]]
        tmp = a_data[i] - bi
        result += tmp * tmp - bi * bi

    result += b_squared_norm

    if result < 0:
        result = 0.0

    return result if squared else sqrt(result)


def _euclidean_sparse_dense_wrapper(
        const floating[::1] a_data,
        const int[::1] a_indices,
        const floating[::1] b,
        floating b_squared_norm,
        bint squared
):
    # [FUNCTION DOC]
    # HELPER: Python-callable test wrapper around _euclidean_sparse_dense.
    # Inputs/outputs as for the wrapped kernel; returns a Python float.
    # Exceptions: none beyond argument conversion. Side effects: none.
    """Wrapper of _euclidean_sparse_dense for testing purpose"""
    return _euclidean_sparse_dense(
        a_data, a_indices, b, b_squared_norm, squared)


cpdef floating _inertia_dense(
        const floating[:, ::1] X,           # IN
        const floating[::1] sample_weight,  # IN
        const floating[:, ::1] centers,     # IN
        const int[::1] labels,              # IN
        int n_threads,
        int single_label=-1,
):
    # [FUNCTION DOC]
    # Behavior: sum over samples of sample_weight[i] times the squared distance between
    # X[i] and centers[labels[i]]; if single_label >= 0, only samples with that label
    # contribute. Parallel over samples with an OpenMP static schedule and a sum reduction.
    # Inputs: X (n_samples, n_features) C-contiguous; sample_weight (n_samples,);
    # centers (n_clusters, n_features); labels (n_samples,) int32 in [0, n_clusters);
    # n_threads number of OpenMP threads; single_label optional cluster filter.
    # Outputs: the inertia (float32 or float64 matching X).
    # Exceptions: none from the loop; argument conversion errors only.
    # Side effects: releases the GIL inside the parallel loop; uses OpenMP threads.
    """Compute inertia for dense input data

    Sum of squared distance between each sample and its assigned center.

    If single_label is >= 0, the inertia is computed only for that label.
    """
    cdef:
        int n_samples = X.shape[0]
        int n_features = X.shape[1]
        int i, j

        floating sq_dist = 0.0
        floating inertia = 0.0

    for i in prange(n_samples, nogil=True, num_threads=n_threads,
                    schedule='static'):
        j = labels[i]
        if single_label < 0 or single_label == j:
            sq_dist = _euclidean_dense_dense(&X[i, 0], &centers[j, 0],
                                             n_features, True)
            inertia += sq_dist * sample_weight[i]

    return inertia


cpdef floating _inertia_sparse(
        X,                                  # IN
        const floating[::1] sample_weight,  # IN
        const floating[:, ::1] centers,     # IN
        const int[::1] labels,              # IN
        int n_threads,
        int single_label=-1,
):
    # [FUNCTION DOC]
    # Behavior: same as _inertia_dense for a CSR matrix X. Precomputes squared norms of
    # the centers (via row_norms) and evaluates each sample's distance to its center
    # with _euclidean_sparse_dense. Parallel over samples with OpenMP.
    # Inputs: X scipy CSR matrix with int32 indices/indptr and data dtype matching
    # the centers; sample_weight, centers, labels, n_threads, single_label as in
    # _inertia_dense.
    # Outputs: the inertia.
    # Exceptions: ValueError/TypeError if X's buffers have unexpected dtype.
    # Side effects: allocates the centers' squared-norm array; releases the GIL in the
    # parallel loop; uses OpenMP threads.
    """Compute inertia for sparse input data

    Sum of squared distance between each sample and its assigned center.

    If single_label is >= 0, the inertia is computed only for that label.
    """
    cdef:
        floating[::1] X_data = X.data
        int[::1] X_indices = X.indices
        int[::1] X_indptr = X.indptr

        int n_samples = X.shape[0]
        int i, j

        floating sq_dist = 0.0
        floating inertia = 0.0

        floating[::1] centers_squared_norms = row_norms(centers, squared=True)

    for i in prange(n_samples, nogil=True, num_threads=n_threads,
                    schedule='static'):
        j = labels[i]
        if single_label < 0 or single_label == j:
            sq_dist = _euclidean_sparse_dense(
                X_data[X_indptr[i]: X_indptr[i + 1]],
                X_indices[X_indptr[i]: X_indptr[i + 1]],
                centers[j], centers_squared_norms[j], True)
            inertia += sq_dist * sample_weight[i]

    return inertia


cpdef void _relocate_empty_clusters_dense(
        const floating[:, ::1] X,            # IN
        const floating[::1] sample_weight,   # IN
        const floating[:, ::1] centers_old,  # IN
        floating[:, ::1] centers_new,        # INOUT
        floating[::1] weight_in_clusters,    # INOUT
        const int[::1] labels                # IN
):
    # [FUNCTION DOC]
    # HELPER: used after each k-means update step to handle empty clusters.
    # Behavior: finds clusters whose weight_in_clusters is exactly 0. If there are
    # n_empty of them, computes each sample's squared distance to its old assigned
    # center and picks the n_empty farthest samples (descending distance). If all
    # distances are 0 (more clusters than distinct samples), does nothing. Otherwise
    # each chosen sample is moved into one empty cluster: its weighted coordinates are
    # subtracted from its former cluster's accumulated sum in centers_new and become
    # the empty cluster's sum; cluster weights are transferred likewise.
    # Inputs: X (n_samples, n_features); sample_weight (n_samples,); centers_old
    # (n_clusters, n_features); labels (n_samples,) assignments to centers_old.
    # Outputs: centers_new (n_clusters, n_features) holding UNNORMALIZED weighted
    # sums (before _average_centers) and weight_in_clusters (n_clusters,), both
    # updated in place. labels is not modified.
    # Exceptions: none expected. Side effects: numpy temporaries allocated; holds the GIL.
    # KNOWN ISSUE: if a chosen far sample was the only member of its former cluster,
    # that cluster becomes empty (weight 0) and is not relocated in this call.
    """Relocate centers which have no sample assigned to them."""
    cdef:
        int[::1] empty_clusters = np.where(np.equal(weight_in_clusters, 0))[0].astype(np.int32)
        int n_empty = empty_clusters.shape[0]

    if n_empty == 0:
        return

    cdef:
        int n_features = X.shape[1]

        floating[::1] distances = ((np.asarray(X) - np.asarray(centers_old)[labels])**2).sum(axis=1)
        int[::1] far_from_centers = np.argpartition(distances, -n_empty)[:-n_empty-1:-1].astype(np.int32)

        int new_cluster_id, old_cluster_id, far_idx, idx, k
        floating weight

    if np.max(distances) == 0:
        # Happens when there are more clusters than non-duplicate samples. Relocating
        # is pointless in this case.
        return

    for idx in range(n_empty):

        new_cluster_id = empty_clusters[idx]

        far_idx = far_from_centers[idx]
        weight = sample_weight[far_idx]

        old_cluster_id = labels[far_idx]

        for k in range(n_features):
            centers_new[old_cluster_id, k] -= X[far_idx, k] * weight
            centers_new[new_cluster_id, k] = X[far_idx, k] * weight

        weight_in_clusters[new_cluster_id] = weight
        weight_in_clusters[old_cluster_id] -= weight


cpdef void _relocate_empty_clusters_sparse(
        const floating[::1] X_data,          # IN
        const int[::1] X_indices,            # IN
        const int[::1] X_indptr,             # IN
        const floating[::1] sample_weight,   # IN
        const floating[:, ::1] centers_old,  # IN
        floating[:, ::1] centers_new,        # INOUT
        floating[::1] weight_in_clusters,    # INOUT
        const int[::1] labels                # IN
):
    # [FUNCTION DOC]
    # HELPER: sparse counterpart of _relocate_empty_clusters_dense.
    # Behavior: identical algorithm; distances to old centers are computed with
    # _euclidean_sparse_dense using precomputed squared center norms, and only the
    # nonzero coordinates of each relocated sample are transferred between centers.
    # Inputs: X_data, X_indices, X_indptr the CSR arrays of X (int32 indices);
    # sample_weight, centers_old, labels as in the dense version.
    # Outputs: centers_new (weighted sums) and weight_in_clusters updated in place.
    # Exceptions: none expected. Side effects: numpy temporaries allocated; holds the GIL.
    # KNOWN ISSUE: same as the dense version (a relocated sample can leave its former
    # cluster empty without that cluster being relocated).
    """Relocate centers which have no sample assigned to them."""
    cdef:
        int[::1] empty_clusters = np.where(np.equal(weight_in_clusters, 0))[0].astype(np.int32)
        int n_empty = empty_clusters.shape[0]

    if n_empty == 0:
        return

    cdef:
        int n_samples = X_indptr.shape[0] - 1
        int i, j, k

        floating[::1] distances = np.zeros(n_samples, dtype=X_data.base.dtype)
        floating[::1] centers_squared_norms = row_norms(centers_old, squared=True)

    for i in range(n_samples):
        j = labels[i]
        distances[i] = _euclidean_sparse_dense(
            X_data[X_indptr[i]: X_indptr[i + 1]],
            X_indices[X_indptr[i]: X_indptr[i + 1]],
            centers_old[j], centers_squared_norms[j], True)

    if np.max(distances) == 0:
        # Happens when there are more clusters than non-duplicate samples. Relocating
        # is pointless in this case.
        return

    cdef:
        int[::1] far_from_centers = np.argpartition(distances, -n_empty)[:-n_empty-1:-1].astype(np.int32)

        int new_cluster_id, old_cluster_id, far_idx, idx
        floating weight

    for idx in range(n_empty):

        new_cluster_id = empty_clusters[idx]

        far_idx = far_from_centers[idx]
        weight = sample_weight[far_idx]

        old_cluster_id = labels[far_idx]

        for k in range(X_indptr[far_idx], X_indptr[far_idx + 1]):
            centers_new[old_cluster_id, X_indices[k]] -= X_data[k] * weight
            centers_new[new_cluster_id, X_indices[k]] = X_data[k] * weight

        weight_in_clusters[new_cluster_id] = weight
        weight_in_clusters[old_cluster_id] -= weight


cdef void _average_centers(
        floating[:, ::1] centers,               # INOUT
        const floating[::1] weight_in_clusters  # IN
):
    # [FUNCTION DOC]
    # HELPER: final normalization step of a k-means center update.
    # Behavior: for every cluster with positive weight, divides its accumulated
    # weighted coordinate sum by the weight to get the mean. Clusters with zero weight
    # are set to a copy of the (already normalized or not, depending on index order)
    # row of the cluster with maximum weight.
    # Inputs: centers (n_clusters, n_features) weighted sums; weight_in_clusters (n_clusters,).
    # Outputs: centers modified in place. Exceptions: none expected.
    # Side effects: mutates centers; holds the GIL (calls numpy argmax).
    # KNOWN ISSUE: an empty cluster with index greater than argmax_weight copies the
    # already-averaged center, but one with a smaller index copies the still
    # unnormalized weighted sum of the biggest cluster. Rarely matters because
    # empty clusters are normally relocated beforehand.
    """Average new centers wrt weights."""
    cdef:
        int n_clusters = centers.shape[0]
        int n_features = centers.shape[1]
        int j, k
        floating alpha
        int argmax_weight = np.argmax(weight_in_clusters)

    for j in range(n_clusters):
        if weight_in_clusters[j] > 0:
            alpha = 1.0 / weight_in_clusters[j]
            for k in range(n_features):
                centers[j, k] *= alpha
        else:
            # For convenience, we avoid setting empty clusters at the origin but place
            # them at the location of the biggest cluster.
            for k in range(n_features):
                centers[j, k] = centers[argmax_weight, k]


cdef void _center_shift(
        const floating[:, ::1] centers_old,  # IN
        const floating[:, ::1] centers_new,  # IN
        floating[::1] center_shift           # OUT
):
    # [FUNCTION DOC]
    # HELPER: convergence measure for k-means.
    # Behavior: for each cluster j, writes the (non-squared) Euclidean distance
    # between centers_new[j] and centers_old[j] into center_shift[j].
    # Inputs: centers_old, centers_new (n_clusters, n_features).
    # Outputs: center_shift (n_clusters,) written in place.
    # Exceptions: none. Side effects: mutates center_shift.
    """Compute shift between old and new centers."""
    cdef:
        int n_clusters = centers_old.shape[0]
        int n_features = centers_old.shape[1]
        int j

    for j in range(n_clusters):
        center_shift[j] = _euclidean_dense_dense(
            &centers_new[j, 0], &centers_old[j, 0], n_features, False)


def _is_same_clustering(
    const int[::1] labels1,
    const int[::1] labels2,
    n_clusters
):
    # [FUNCTION DOC]
    # Behavior: builds a mapping from labels1 values to labels2 values while scanning
    # both arrays; returns False as soon as one labels1 value would map to two
    # different labels2 values, True otherwise.
    # Inputs: labels1, labels2 int32 arrays of equal length with values in
    # [0, n_clusters); n_clusters the number of distinct labels.
    # Outputs: bool.
    # Exceptions: IndexError possible if labels exceed n_clusters (with bounds checks).
    # Side effects: allocates a small mapping array.
    # KNOWN ISSUE: only checks that the mapping is a function, not that it is
    # injective; two different labels1 values mapping to the same labels2 value
    # still return True, so this is not a strict permutation check.
    """Check if two arrays of labels are the same up to a permutation of the labels"""
    cdef int[::1] mapping = np.full(fill_value=-1, shape=(n_clusters,), dtype=np.int32)
    cdef int i

    for i in range(labels1.shape[0]):
        if mapping[labels1[i]] == -1:
            mapping[labels1[i]] = labels2[i]
        elif mapping[labels1[i]] != labels2[i]:
            return False
    return True
