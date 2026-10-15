# [PACKAGE DOC]
# Package: sklearn.cluster
# Cython components of sklearn.cluster:
# - _k_means_common: shared k-means kernels (dense/sparse Euclidean distances,
#   inertia, empty-cluster relocation, center averaging and shift, CHUNK_SIZE).
#   Its .pxd exposes these to the other k-means modules.
# - _k_means_lloyd: chunked, OpenMP-parallel Lloyd iteration (dense and CSR).
# - _k_means_elkan: Elkan's triangle-inequality accelerated k-means (bound
#   initialization and iteration, dense and CSR).
# - _k_means_minibatch: running-mean center updates for MiniBatchKMeans.
#   These four back KMeans, BisectingKMeans and MiniBatchKMeans in _kmeans.py and
#   _bisect_k_means.py.
# - _dbscan_inner: connected-component labeling loop of DBSCAN.
# - _hierarchical_fast: agglomerative clustering helpers (Ward costs, tree
#   cutting, IntFloatDict merge strategies, WeightedEdge, UnionFind, MST-based
#   single linkage); its .pxd exports UnionFind, reused by _hdbscan._linkage.
# - _hdbscan subpackage: mutual reachability, MST/single linkage and condensed
#   tree cluster extraction for HDBSCAN (see its __init__.py).
# (This __init__.pxd holds only documentation; it declares nothing.)
