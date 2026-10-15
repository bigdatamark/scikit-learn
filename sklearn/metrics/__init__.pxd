# [PACKAGE DOC]
# Package: sklearn.metrics
# Cython components of sklearn.metrics:
# - _dist_metrics (generated from _dist_metrics.pyx.tp / .pxd.tp): the DistanceMetric
#   hierarchy (public as sklearn.metrics.DistanceMetric), with float64 and float32
#   specialisations of ~20 metrics. Each metric exposes GIL-free C kernels for dense
#   vectors and CSR rows plus rank-preserving surrogate distances; the .pxd lets other
#   Cython code (sklearn.neighbors trees, HDBSCAN, the reduction backend below) cimport
#   and call them directly.
# - _pairwise_fast: small kernels used by sklearn.metrics.pairwise (dense additive
#   chi-squared kernel loop, OpenMP-parallel L1 distances between CSR matrices).
# - _pairwise_distances_reduction/ (subpackage, documented separately): templated
#   Cython/OpenMP backend computing chunked pairwise distances combined with a reduction
#   (argkmin, radius neighbors, ...), built on DistanceMetric and BLAS GEMM for the
#   Euclidean case; used by pairwise_distances_argmin(_min), sklearn.neighbors, etc.
# - cluster/_expected_mutual_info_fast: expected mutual information for AMI (see the
#   cluster subpackage doc).
# (This __init__.pxd holds only documentation; it declares nothing.)
