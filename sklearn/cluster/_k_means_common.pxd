# [MODULE DOC]
# Declarations of the shared k-means helpers implemented in _k_means_common.pyx,
# cimported by _k_means_lloyd, _k_means_elkan and _k_means_minibatch. All are
# fused over float32/float64 ("floating").
from cython cimport floating


cdef floating _euclidean_dense_dense(
    const floating*,
    const floating*,
    int,
    bint
) noexcept nogil
# [FUNCTION DOC]
# HELPER: distance kernel for k-means. Euclidean (or squared) distance between
# two dense vectors of given length; nogil, cannot raise. See the .pyx.

cdef floating _euclidean_sparse_dense(
    const floating[::1],
    const int[::1],
    const floating[::1],
    floating,
    bint
) noexcept nogil
# [FUNCTION DOC]
# HELPER: distance kernel for k-means. Euclidean (or squared) distance between a
# CSR row (data, indices) and a dense vector with its precomputed squared norm;
# nogil, cannot raise. See the .pyx.

cpdef void _relocate_empty_clusters_dense(
    const floating[:, ::1],
    const floating[::1],
    const floating[:, ::1],
    floating[:, ::1],
    floating[::1],
    const int[::1]
)
# [FUNCTION DOC]
# HELPER: moves empty clusters onto the samples farthest from their centers
# (dense input), updating centers_new and weight_in_clusters in place. See the .pyx.

cpdef void _relocate_empty_clusters_sparse(
    const floating[::1],
    const int[::1],
    const int[::1],
    const floating[::1],
    const floating[:, ::1],
    floating[:, ::1],
    floating[::1],
    const int[::1]
# [FUNCTION DOC]
# HELPER: same as _relocate_empty_clusters_dense for CSR input given as
# data/indices/indptr arrays. See the .pyx.
)

cdef void _average_centers(
    floating[:, ::1],
    const floating[::1]
# [FUNCTION DOC]
# HELPER: divides accumulated weighted center sums by cluster weights in place.
# See the .pyx.
)

cdef void _center_shift(
    const floating[:, ::1],
    const floating[:, ::1],
    floating[::1]
# [FUNCTION DOC]
# HELPER: computes per-center Euclidean shift between two center arrays into an
# output vector. See the .pyx.
)
