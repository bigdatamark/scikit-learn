# [PACKAGE DOC]
# Package: sklearn.metrics._pairwise_distances_reduction
# Cython components of the pairwise distances reductions package. All are
# Tempita templates (.tp) generating float64 ("64") and float32 ("32") variants;
# float32 inputs are always reduced with float64 accumulators.
# - _datasets_pair: DatasetsPair classes wrapping (X, Y) in any dense/CSR
#   combination with a DistanceMetric, giving exact and surrogate
#   (rank-preserving) distances for one pair of rows.
# - _middle_term_computer: MiddleTermComputer classes computing, per chunk pair,
#   the -2 X_c Y_c^T term of squared euclidean distances (BLAS GEMM for dense
#   data, dedicated loops for CSR data, float32-to-float64 upcasting).
# - _base: BaseDistancesReduction, the abstract chunked OpenMP driver (parallel
#   on X or parallel on Y) with overridable hooks, and the squared row norms
#   helper.
# - _argkmin: ArgKmin (k nearest neighbors via per-row heaps) and its
#   Euclidean specialization combining row norms and the middle term.
# - _radius_neighbors: RadiusNeighbors (all neighbors within a radius, stored
#   in C++ vectors returned as numpy object arrays) and its Euclidean
#   specialization.
# - _argkmin_classmode, _radius_neighbors_classmode: subclasses that turn
#   neighbor sets directly into per-class probability estimates for the
#   neighbors classifiers; _classmode declares their WeightingStrategy enum.
# The Python _dispatcher module checks usability and dtypes and calls the
# compute class methods of the matching compiled class; dependencies flow
# _datasets_pair and _middle_term_computer into _base, then into the concrete
# reductions, then into the classmode subclasses.
# (This __init__.pxd holds only documentation; it declares nothing.)
