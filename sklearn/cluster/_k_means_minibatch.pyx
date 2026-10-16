# [MODULE DOC]
# Center update step of MiniBatchKMeans (_kmeans.py) for dense and CSR input.
# Each center is updated as the weighted running mean of all samples ever assigned
# to it: previous center times its accumulated weight, plus the weighted new batch
# members, divided by the new accumulated weight. Clusters are processed in
# parallel with OpenMP; labels for the batch are computed elsewhere (Lloyd E-step).
from cython cimport floating
from cython.parallel cimport parallel, prange
from libc.stdlib cimport malloc, free


def _minibatch_update_dense(
        const floating[:, ::1] X,            # IN
        const floating[::1] sample_weight,   # IN
        const floating[:, ::1] centers_old,  # IN
        floating[:, ::1] centers_new,        # OUT
        floating[::1] weight_sums,           # INOUT
        const int[::1] labels,               # IN
        int n_threads):
    # [FUNCTION DOC]
    # Behavior: in a nogil OpenMP parallel region each thread allocates a scratch
    # array of n_samples ints, then clusters are distributed statically over threads
    # and each is updated with update_center_dense; scratch arrays are freed at the end.
    # Inputs: X (n_samples, n_features) batch; sample_weight (n_samples,);
    # centers_old (n_clusters, n_features); labels (n_samples,) batch assignments;
    # n_threads.
    # Outputs: centers_new written in place; weight_sums (n_clusters,) incremented in place.
    # Exceptions: none raised from the nogil region.
    # Side effects: releases the GIL; OpenMP threads; malloc/free of scratch buffers.
    # KNOWN ISSUE: the malloc result is not checked for NULL.
    """Update of the centers for dense MiniBatchKMeans.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features), dtype=floating
        The observations to cluster.

    sample_weight : ndarray of shape (n_samples,), dtype=floating
        The weights for each observation in X.

    centers_old : ndarray of shape (n_clusters, n_features), dtype=floating
        Centers before previous iteration, placeholder for the centers after
        previous iteration.

    centers_new : ndarray of shape (n_clusters, n_features), dtype=floating
        Centers after previous iteration, placeholder for the new centers
        computed during this iteration.

    weight_sums : ndarray of shape (n_clusters,), dtype=floating
        Current sums of the accumulated weights for each center.

    labels : ndarray of shape (n_samples,), dtype=int
        labels assignment.

    n_threads : int
        The number of threads to be used by openmp.
    """
    cdef:
        int n_samples = X.shape[0]
        int n_clusters = centers_old.shape[0]
        int cluster_idx

        int *indices

    with nogil, parallel(num_threads=n_threads):
        indices = <int*> malloc(n_samples * sizeof(int))

        for cluster_idx in prange(n_clusters, schedule="static"):
            update_center_dense(cluster_idx, X, sample_weight,
                                centers_old, centers_new, weight_sums, labels,
                                indices)

        free(indices)


cdef void update_center_dense(
        int cluster_idx,
        const floating[:, ::1] X,            # IN
        const floating[::1] sample_weight,   # IN
        const floating[:, ::1] centers_old,  # IN
        floating[:, ::1] centers_new,        # OUT
        floating[::1] weight_sums,           # INOUT
        const int[::1] labels,               # IN
        int *indices) noexcept nogil:        # TMP
    # [FUNCTION DOC]
    # HELPER: per-cluster worker for _minibatch_update_dense.
    # Behavior: collects into indices the batch samples labeled cluster_idx and their
    # total weight wsum. If wsum > 0: sets the new center to old center times
    # weight_sums[cluster_idx], adds weighted coordinates of those samples, increases
    # weight_sums[cluster_idx] by wsum and divides the center by the new total.
    # Otherwise copies the old center unchanged.
    # Inputs: cluster_idx; X, sample_weight, centers_old, labels as above;
    # indices scratch buffer of at least n_samples ints.
    # Outputs: row cluster_idx of centers_new; weight_sums[cluster_idx] updated.
    # Exceptions: none (noexcept nogil). Side effects: writes indices scratch; each
    # call touches only its own cluster's row, so calls are thread-safe.
    """Update of a single center for dense MinibatchKMeans"""
    cdef:
        int n_samples = sample_weight.shape[0]
        int n_features = centers_old.shape[1]
        floating alpha
        int n_indices
        int k, sample_idx, feature_idx

        floating wsum = 0

    # indices = np.where(labels == cluster_idx)[0]
    k = 0
    for sample_idx in range(n_samples):
        if labels[sample_idx] == cluster_idx:
            indices[k] = sample_idx
            wsum += sample_weight[sample_idx]
            k += 1
    n_indices = k

    if wsum > 0:
        # Undo the previous count-based scaling for this cluster center
        for feature_idx in range(n_features):
            centers_new[cluster_idx, feature_idx] = centers_old[cluster_idx, feature_idx] * weight_sums[cluster_idx]

        # Update cluster with new point members
        for k in range(n_indices):
            sample_idx = indices[k]
            for feature_idx in range(n_features):
                centers_new[cluster_idx, feature_idx] += X[sample_idx, feature_idx] * sample_weight[sample_idx]

        # Update the count statistics for this center
        weight_sums[cluster_idx] += wsum

        # Rescale to compute mean of all points (old and new)
        alpha = 1 / weight_sums[cluster_idx]
        for feature_idx in range(n_features):
            centers_new[cluster_idx, feature_idx] *= alpha
    else:
        # No sample was assigned to this cluster in this batch of data
        for feature_idx in range(n_features):
            centers_new[cluster_idx, feature_idx] = centers_old[cluster_idx, feature_idx]


def _minibatch_update_sparse(
        X,                                   # IN
        const floating[::1] sample_weight,   # IN
        const floating[:, ::1] centers_old,  # IN
        floating[:, ::1] centers_new,        # OUT
        floating[::1] weight_sums,           # INOUT
        const int[::1] labels,               # IN
        int n_threads):
    # [FUNCTION DOC]
    # Behavior: sparse counterpart of _minibatch_update_dense: extracts the CSR
    # buffers of X and runs update_center_sparse for each cluster in parallel with
    # per-thread scratch arrays.
    # Inputs: X scipy CSR matrix (int32 indices/indptr, data dtype matching centers);
    # other parameters as in the dense version.
    # Outputs: centers_new written; weight_sums incremented in place.
    # Exceptions: errors converting X buffers with unexpected dtypes.
    # Side effects: releases the GIL; OpenMP; malloc/free of scratch buffers.
    # KNOWN ISSUE: the malloc result is not checked for NULL.
    """Update of the centers for sparse MiniBatchKMeans.

    Parameters
    ----------
    X : sparse matrix of shape (n_samples, n_features), dtype=floating
        The observations to cluster. Must be in CSR format.

    sample_weight : ndarray of shape (n_samples,), dtype=floating
        The weights for each observation in X.

    centers_old : ndarray of shape (n_clusters, n_features), dtype=floating
        Centers before previous iteration, placeholder for the centers after
        previous iteration.

    centers_new : ndarray of shape (n_clusters, n_features), dtype=floating
        Centers after previous iteration, placeholder for the new centers
        computed during this iteration.

    weight_sums : ndarray of shape (n_clusters,), dtype=floating
        Current sums of the accumulated weights for each center.

    labels : ndarray of shape (n_samples,), dtype=int
        labels assignment.

    n_threads : int
        The number of threads to be used by openmp.
    """
    cdef:
        floating[::1] X_data = X.data
        int[::1] X_indices = X.indices
        int[::1] X_indptr = X.indptr
        int n_samples = X.shape[0]
        int n_clusters = centers_old.shape[0]
        int cluster_idx

        int *indices

    with nogil, parallel(num_threads=n_threads):
        indices = <int*> malloc(n_samples * sizeof(int))

        for cluster_idx in prange(n_clusters, schedule="static"):
            update_center_sparse(cluster_idx, X_data, X_indices, X_indptr,
                                 sample_weight, centers_old, centers_new,
                                 weight_sums, labels, indices)

        free(indices)


cdef void update_center_sparse(
        int cluster_idx,
        const floating[::1] X_data,          # IN
        const int[::1] X_indices,            # IN
        const int[::1] X_indptr,             # IN
        const floating[::1] sample_weight,   # IN
        const floating[:, ::1] centers_old,  # IN
        floating[:, ::1] centers_new,        # OUT
        floating[::1] weight_sums,           # INOUT
        const int[::1] labels,               # IN
        int *indices) noexcept nogil:        # TMP
    # [FUNCTION DOC]
    # HELPER: per-cluster worker for _minibatch_update_sparse.
    # Behavior: same as update_center_dense, but the new members' contributions are
    # added only over their nonzero entries (CSR row ranges from X_indptr).
    # Inputs: cluster_idx; X_data, X_indices, X_indptr the CSR arrays; sample_weight,
    # centers_old, labels; indices scratch of at least n_samples ints.
    # Outputs: row cluster_idx of centers_new; weight_sums[cluster_idx] updated.
    # Exceptions: none (noexcept nogil). Side effects: writes indices scratch only otherwise.
    """Update of a single center for sparse MinibatchKMeans"""
    cdef:
        int n_samples = sample_weight.shape[0]
        int n_features = centers_old.shape[1]
        floating alpha
        int n_indices
        int k, sample_idx, feature_idx

        floating wsum = 0

    # indices = np.where(labels == cluster_idx)[0]
    k = 0
    for sample_idx in range(n_samples):
        if labels[sample_idx] == cluster_idx:
            indices[k] = sample_idx
            wsum += sample_weight[sample_idx]
            k += 1
    n_indices = k

    if wsum > 0:
        # Undo the previous count-based scaling for this cluster center:
        for feature_idx in range(n_features):
            centers_new[cluster_idx, feature_idx] = centers_old[cluster_idx, feature_idx] * weight_sums[cluster_idx]

        # Update cluster with new point members
        for k in range(n_indices):
            sample_idx = indices[k]
            for feature_idx in range(X_indptr[sample_idx], X_indptr[sample_idx + 1]):
                centers_new[cluster_idx, X_indices[feature_idx]] += X_data[feature_idx] * sample_weight[sample_idx]

        # Update the count statistics for this center
        weight_sums[cluster_idx] += wsum

        # Rescale to compute mean of all points (old and new)
        alpha = 1 / weight_sums[cluster_idx]
        for feature_idx in range(n_features):
            centers_new[cluster_idx, feature_idx] *= alpha
    else:
        # No sample was assigned to this cluster in this batch of data
        for feature_idx in range(n_features):
            centers_new[cluster_idx, feature_idx] = centers_old[cluster_idx, feature_idx]
