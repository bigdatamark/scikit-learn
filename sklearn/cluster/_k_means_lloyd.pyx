# [MODULE DOC]
# Lloyd's algorithm for k-means (used by KMeans(algorithm="lloyd") and
# BisectingKMeans in _kmeans.py). Provides one full iteration (assignment E-step
# plus optional center update M-step) for dense and CSR sparse input. Work is split
# into chunks of CHUNK_SIZE samples processed in parallel with OpenMP; each thread
# accumulates partial center sums in private buffers that are reduced under a lock.
# Empty-cluster relocation, averaging and center-shift come from _k_means_common.
# Licence: BSD 3 clause

from cython cimport floating
from cython.parallel import prange, parallel
from libc.stdlib cimport malloc, calloc, free
from libc.string cimport memset
from libc.float cimport DBL_MAX, FLT_MAX

from sklearn.utils._openmp_helpers cimport omp_lock_t
from sklearn.utils._openmp_helpers cimport omp_init_lock
from sklearn.utils._openmp_helpers cimport omp_destroy_lock
from sklearn.utils._openmp_helpers cimport omp_set_lock
from sklearn.utils._openmp_helpers cimport omp_unset_lock
from sklearn.utils.extmath import row_norms
from sklearn.utils._cython_blas cimport _gemm
from sklearn.utils._cython_blas cimport RowMajor, Trans, NoTrans
from sklearn.cluster._k_means_common import CHUNK_SIZE
from sklearn.cluster._k_means_common cimport _relocate_empty_clusters_dense
from sklearn.cluster._k_means_common cimport _relocate_empty_clusters_sparse
from sklearn.cluster._k_means_common cimport _average_centers, _center_shift


def lloyd_iter_chunked_dense(
        const floating[:, ::1] X,            # IN
        const floating[::1] sample_weight,   # IN
        const floating[:, ::1] centers_old,  # IN
        floating[:, ::1] centers_new,        # OUT
        floating[::1] weight_in_clusters,    # OUT
        int[::1] labels,                     # OUT
        floating[::1] center_shift,          # OUT
        int n_threads,
        bint update_centers=True):
    # [FUNCTION DOC]
    # Behavior: returns immediately for 0 samples. Splits samples into n_chunks chunks
    # of CHUNK_SIZE (last chunk may be smaller) and caps n_threads at n_chunks. If
    # update_centers, zeroes centers_new and weight_in_clusters. In a nogil OpenMP
    # parallel region, each thread allocates private buffers (center sums, weights,
    # and a chunk x n_clusters distance buffer), processes its chunks statically with
    # _update_chunk_dense, then adds its private sums into the shared outputs while
    # holding an OpenMP lock, and frees its buffers. Afterwards (if update_centers),
    # relocates empty clusters, divides sums by weights to obtain new centers and
    # computes the per-center shift.
    # Inputs: X (n_samples, n_features) C-contiguous; sample_weight (n_samples,);
    # centers_old (n_clusters, n_features); n_threads; update_centers flag.
    # Outputs (in place): labels (n_samples,) nearest-center indices; if update_centers
    # also centers_new, weight_in_clusters, center_shift (n_clusters,). Returns None.
    # Exceptions: none from the nogil region (allocation failures are not detected).
    # Side effects: releases the GIL, spawns OpenMP threads, malloc/calloc/free of
    # thread-local buffers, initializes and destroys an OpenMP lock.
    # KNOWN ISSUE: results of malloc/calloc are not checked for NULL; an allocation
    # failure would lead to a segmentation fault instead of a MemoryError.
    """Single iteration of K-means lloyd algorithm with dense input.

    Update labels and centers (inplace), for one iteration, distributed
    over data chunks.

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
        computed during this iteration. `centers_new` can be `None` if
        `update_centers` is False.

    weight_in_clusters : ndarray of shape (n_clusters,), dtype=floating
        Placeholder for the sums of the weights of every observation assigned
        to each center. `weight_in_clusters` can be `None` if `update_centers`
        is False.

    labels : ndarray of shape (n_samples,), dtype=int
        labels assignment.

    center_shift : ndarray of shape (n_clusters,), dtype=floating
        Distance between old and new centers.

    n_threads : int
        The number of threads to be used by openmp.

    update_centers : bool
        - If True, the labels and the new centers will be computed, i.e. runs
          the E-step and the M-step of the algorithm.
        - If False, only the labels will be computed, i.e runs the E-step of
          the algorithm. This is useful especially when calling predict on a
          fitted model.
    """
    cdef:
        int n_samples = X.shape[0]
        int n_features = X.shape[1]
        int n_clusters = centers_old.shape[0]

    if n_samples == 0:
        # An empty array was passed, do nothing and return early (before
        # attempting to compute n_chunks). This can typically happen when
        # calling the prediction function of a bisecting k-means model with a
        # large fraction of outliers.
        return

    cdef:
        # hard-coded number of samples per chunk. Appeared to be close to
        # optimal in all situations.
        int n_samples_chunk = CHUNK_SIZE if n_samples > CHUNK_SIZE else n_samples
        int n_chunks = n_samples // n_samples_chunk
        int n_samples_rem = n_samples % n_samples_chunk
        int chunk_idx
        int start, end

        int j, k

        floating[::1] centers_squared_norms = row_norms(centers_old, squared=True)

        floating *centers_new_chunk
        floating *weight_in_clusters_chunk
        floating *pairwise_distances_chunk

        omp_lock_t lock

    # count remainder chunk in total number of chunks
    n_chunks += n_samples != n_chunks * n_samples_chunk

    # number of threads should not be bigger than number of chunks
    n_threads = min(n_threads, n_chunks)

    if update_centers:
        memset(&centers_new[0, 0], 0, n_clusters * n_features * sizeof(floating))
        memset(&weight_in_clusters[0], 0, n_clusters * sizeof(floating))
        omp_init_lock(&lock)

    with nogil, parallel(num_threads=n_threads):
        # thread local buffers
        centers_new_chunk = <floating*> calloc(n_clusters * n_features, sizeof(floating))
        weight_in_clusters_chunk = <floating*> calloc(n_clusters, sizeof(floating))
        pairwise_distances_chunk = <floating*> malloc(n_samples_chunk * n_clusters * sizeof(floating))

        for chunk_idx in prange(n_chunks, schedule='static'):
            start = chunk_idx * n_samples_chunk
            if chunk_idx == n_chunks - 1 and n_samples_rem > 0:
                end = start + n_samples_rem
            else:
                end = start + n_samples_chunk

            _update_chunk_dense(
                X[start: end],
                sample_weight[start: end],
                centers_old,
                centers_squared_norms,
                labels[start: end],
                centers_new_chunk,
                weight_in_clusters_chunk,
                pairwise_distances_chunk,
                update_centers)

        # reduction from local buffers.
        if update_centers:
            # The lock is necessary to avoid race conditions when aggregating
            # info from different thread-local buffers.
            omp_set_lock(&lock)
            for j in range(n_clusters):
                weight_in_clusters[j] += weight_in_clusters_chunk[j]
                for k in range(n_features):
                    centers_new[j, k] += centers_new_chunk[j * n_features + k]

            omp_unset_lock(&lock)

        free(centers_new_chunk)
        free(weight_in_clusters_chunk)
        free(pairwise_distances_chunk)

    if update_centers:
        omp_destroy_lock(&lock)
        _relocate_empty_clusters_dense(
            X, sample_weight, centers_old, centers_new, weight_in_clusters, labels
        )

        _average_centers(centers_new, weight_in_clusters)
        _center_shift(centers_old, centers_new, center_shift)


cdef void _update_chunk_dense(
        const floating[:, ::1] X,                   # IN
        const floating[::1] sample_weight,          # IN
        const floating[:, ::1] centers_old,         # IN
        const floating[::1] centers_squared_norms,  # IN
        int[::1] labels,                            # OUT
        floating *centers_new,                      # OUT
        floating *weight_in_clusters,               # OUT
        floating *pairwise_distances,               # OUT
        bint update_centers) noexcept nogil:
    """K-means combined EM step for one dense data chunk.

    Compute the partial contribution of a single data chunk to the labels and
    centers.
    """
    # [FUNCTION DOC]
    # HELPER: per-chunk worker for lloyd_iter_chunked_dense.
    # Behavior: fills pairwise_distances (n_samples x n_clusters, row-major) with the
    # squared center norms, then adds -2 X C^T via BLAS gemm, giving squared distances
    # minus the constant ||x||^2 (enough for argmin). Assigns each sample to the first
    # center with minimal value; if update_centers, adds sample_weight to that
    # cluster's weight and sample_weight * x to its row of centers_new.
    # Inputs: X chunk (n_samples, n_features); sample_weight chunk; centers_old;
    # centers_squared_norms (n_clusters,); update_centers flag.
    # Outputs: labels chunk written; centers_new (n_clusters*n_features) and
    # weight_in_clusters (n_clusters) thread-local buffers accumulated;
    # pairwise_distances used as scratch (overwritten).
    # Exceptions: none (noexcept nogil). Side effects: writes the buffers only.
    cdef:
        int n_samples = labels.shape[0]
        int n_clusters = centers_old.shape[0]
        int n_features = centers_old.shape[1]

        floating sq_dist, min_sq_dist
        int i, j, k, label

    # Instead of computing the full pairwise squared distances matrix,
    # ||X - C||² = ||X||² - 2 X.C^T + ||C||², we only need to store
    # the - 2 X.C^T + ||C||² term since the argmin for a given sample only
    # depends on the centers.
    # pairwise_distances = ||C||²
    for i in range(n_samples):
        for j in range(n_clusters):
            pairwise_distances[i * n_clusters + j] = centers_squared_norms[j]

    # pairwise_distances += -2 * X.dot(C.T)
    _gemm(RowMajor, NoTrans, Trans, n_samples, n_clusters, n_features,
          -2.0, &X[0, 0], n_features, &centers_old[0, 0], n_features,
          1.0, pairwise_distances, n_clusters)

    for i in range(n_samples):
        min_sq_dist = pairwise_distances[i * n_clusters]
        label = 0
        for j in range(1, n_clusters):
            sq_dist = pairwise_distances[i * n_clusters + j]
            if sq_dist < min_sq_dist:
                min_sq_dist = sq_dist
                label = j
        labels[i] = label

        if update_centers:
            weight_in_clusters[label] += sample_weight[i]
            for k in range(n_features):
                centers_new[label * n_features + k] += X[i, k] * sample_weight[i]


def lloyd_iter_chunked_sparse(
        X,                                   # IN
        const floating[::1] sample_weight,   # IN
        const floating[:, ::1] centers_old,  # IN
        floating[:, ::1] centers_new,        # OUT
        floating[::1] weight_in_clusters,    # OUT
        int[::1] labels,                     # OUT
        floating[::1] center_shift,          # OUT
        int n_threads,
        bint update_centers=True):
    # [FUNCTION DOC]
    # Behavior: sparse counterpart of lloyd_iter_chunked_dense with the same chunking,
    # thread-local buffers, locked reduction and post-processing; chunks are passed as
    # slices of the CSR data/indices and the indptr slice for the chunk (indptr values
    # remain absolute). No distance buffer is allocated.
    # Inputs: X scipy CSR matrix (int32 indices/indptr, data dtype matching centers);
    # others as in the dense version.
    # Outputs (in place): labels; if update_centers centers_new, weight_in_clusters,
    # center_shift.
    # Exceptions: errors when extracting X buffers with unexpected dtypes; none from
    # the nogil region. Side effects: as in the dense version (GIL release, OpenMP,
    # allocations, lock).
    # KNOWN ISSUE: malloc/calloc results are not checked for NULL.
    """Single iteration of K-means lloyd algorithm with sparse input.

    Update labels and centers (inplace), for one iteration, distributed
    over data chunks.

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
        computed during this iteration. `centers_new` can be `None` if
        `update_centers` is False.

    weight_in_clusters : ndarray of shape (n_clusters,), dtype=floating
        Placeholder for the sums of the weights of every observation assigned
        to each center. `weight_in_clusters` can be `None` if `update_centers`
        is False.

    labels : ndarray of shape (n_samples,), dtype=int
        labels assignment.

    center_shift : ndarray of shape (n_clusters,), dtype=floating
        Distance between old and new centers.

    n_threads : int
        The number of threads to be used by openmp.

    update_centers : bool
        - If True, the labels and the new centers will be computed, i.e. runs
          the E-step and the M-step of the algorithm.
        - If False, only the labels will be computed, i.e runs the E-step of
          the algorithm. This is useful especially when calling predict on a
          fitted model.
    """
    cdef:
        int n_samples = X.shape[0]
        int n_features = X.shape[1]
        int n_clusters = centers_old.shape[0]

    if n_samples == 0:
        # An empty array was passed, do nothing and return early (before
        # attempting to compute n_chunks). This can typically happen when
        # calling the prediction function of a bisecting k-means model with a
        # large fraction of outliers.
        return

    cdef:
        # Choose same as for dense. Does not have the same impact since with
        # sparse data the pairwise distances matrix is not precomputed.
        # However, splitting in chunks is necessary to get parallelism.
        int n_samples_chunk = CHUNK_SIZE if n_samples > CHUNK_SIZE else n_samples
        int n_chunks = n_samples // n_samples_chunk
        int n_samples_rem = n_samples % n_samples_chunk
        int chunk_idx
        int start = 0, end = 0

        int j, k

        floating[::1] X_data = X.data
        int[::1] X_indices = X.indices
        int[::1] X_indptr = X.indptr

        floating[::1] centers_squared_norms = row_norms(centers_old, squared=True)

        floating *centers_new_chunk
        floating *weight_in_clusters_chunk

        omp_lock_t lock

    # count remainder chunk in total number of chunks
    n_chunks += n_samples != n_chunks * n_samples_chunk

    # number of threads should not be bigger than number of chunks
    n_threads = min(n_threads, n_chunks)

    if update_centers:
        memset(&centers_new[0, 0], 0, n_clusters * n_features * sizeof(floating))
        memset(&weight_in_clusters[0], 0, n_clusters * sizeof(floating))
        omp_init_lock(&lock)

    with nogil, parallel(num_threads=n_threads):
        # thread local buffers
        centers_new_chunk = <floating*> calloc(n_clusters * n_features, sizeof(floating))
        weight_in_clusters_chunk = <floating*> calloc(n_clusters, sizeof(floating))

        for chunk_idx in prange(n_chunks, schedule='static'):
            start = chunk_idx * n_samples_chunk
            if chunk_idx == n_chunks - 1 and n_samples_rem > 0:
                end = start + n_samples_rem
            else:
                end = start + n_samples_chunk

            _update_chunk_sparse(
                X_data[X_indptr[start]: X_indptr[end]],
                X_indices[X_indptr[start]: X_indptr[end]],
                X_indptr[start: end+1],
                sample_weight[start: end],
                centers_old,
                centers_squared_norms,
                labels[start: end],
                centers_new_chunk,
                weight_in_clusters_chunk,
                update_centers)

        # reduction from local buffers.
        if update_centers:
            # The lock is necessary to avoid race conditions when aggregating
            # info from different thread-local buffers.
            omp_set_lock(&lock)
            for j in range(n_clusters):
                weight_in_clusters[j] += weight_in_clusters_chunk[j]
                for k in range(n_features):
                    centers_new[j, k] += centers_new_chunk[j * n_features + k]
            omp_unset_lock(&lock)

        free(centers_new_chunk)
        free(weight_in_clusters_chunk)

    if update_centers:
        omp_destroy_lock(&lock)
        _relocate_empty_clusters_sparse(
            X_data, X_indices, X_indptr, sample_weight,
            centers_old, centers_new, weight_in_clusters, labels)

        _average_centers(centers_new, weight_in_clusters)
        _center_shift(centers_old, centers_new, center_shift)


cdef void _update_chunk_sparse(
        const floating[::1] X_data,                 # IN
        const int[::1] X_indices,                   # IN
        const int[::1] X_indptr,                    # IN
        const floating[::1] sample_weight,          # IN
        const floating[:, ::1] centers_old,         # IN
        const floating[::1] centers_squared_norms,  # IN
        int[::1] labels,                            # OUT
        floating *centers_new,                      # OUT
        floating *weight_in_clusters,               # OUT
        bint update_centers) noexcept nogil:
    """K-means combined EM step for one sparse data chunk.

    Compute the partial contribution of a single data chunk to the labels and
    centers.
    """
    # [FUNCTION DOC]
    # HELPER: per-chunk worker for lloyd_iter_chunked_sparse.
    # Behavior: for each sample of the chunk, computes for each center the value
    # ||c||^2 - 2 x.c using the sample's nonzeros only (offsetting indptr by its first
    # value s so the data/indices slices are addressed locally), and picks the first
    # minimum. If update_centers, accumulates the sample weight and weighted nonzero
    # coordinates into the thread-local buffers.
    # Inputs: X_data, X_indices chunk slices; X_indptr (n_samples+1) absolute offsets;
    # sample_weight, centers_old, centers_squared_norms; update_centers flag.
    # Outputs: labels chunk; centers_new and weight_in_clusters buffers accumulated.
    # Exceptions: none (noexcept nogil). Side effects: writes buffers only.
    cdef:
        int n_samples = labels.shape[0]
        int n_clusters = centers_old.shape[0]
        int n_features = centers_old.shape[1]

        floating sq_dist, min_sq_dist
        int i, j, k, label
        floating max_floating = FLT_MAX if floating is float else DBL_MAX
        int s = X_indptr[0]

    # XXX Precompute the pairwise distances matrix is not worth for sparse
    # currently. Should be tested when BLAS (sparse x dense) matrix
    # multiplication is available.
    for i in range(n_samples):
        min_sq_dist = max_floating
        label = 0

        for j in range(n_clusters):
            sq_dist = 0.0
            for k in range(X_indptr[i] - s, X_indptr[i + 1] - s):
                sq_dist += centers_old[j, X_indices[k]] * X_data[k]

            # Instead of computing the full squared distance with each cluster,
            # ||X - C||² = ||X||² - 2 X.C^T + ||C||², we only need to compute
            # the - 2 X.C^T + ||C||² term since the argmin for a given sample
            # only depends on the centers C.
            sq_dist = centers_squared_norms[j] -2 * sq_dist
            if sq_dist < min_sq_dist:
                min_sq_dist = sq_dist
                label = j

        labels[i] = label

        if update_centers:
            weight_in_clusters[label] += sample_weight[i]
            for k in range(X_indptr[i] - s, X_indptr[i + 1] - s):
                centers_new[label * n_features + X_indices[k]] += X_data[k] * sample_weight[i]
