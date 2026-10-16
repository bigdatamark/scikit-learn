# [PACKAGE DOC]
# Package: sklearn.utils
# Compiled (Cython) components of sklearn.utils, shared by many estimators:
# - _typedefs: C aliases for NumPy dtypes (intp_t, float64_t, int32_t, ...)
#   cimported by almost every Cython module in scikit-learn.
# - _cython_blas (+ _blas_int.pxi): nogil float/double wrappers over SciPy's
#   BLAS (dot, axpy, nrm2, scal, rot, gemv, ger, gemm) with row/column-major
#   handling.
# - _openmp_helpers: OpenMP shim (no-ops without OpenMP) and the effective
#   thread-count logic used by all prange code.
# - _random: xorshift RNG (our_rand_r) for nogil code and
#   sample_without_replacement.
# - _heap / _sorting: fixed-size max-heap push and introsort over parallel
#   value/index arrays (used by neighbors and pairwise-distance reductions).
# - _vector_sentinel: zero-copy conversion of C++ vectors to NumPy arrays.
# - _fast_dict: IntFloatDict, a C++ map-backed int->float dict (hierarchical
#   clustering).
# - _bitset: 256-value bitsets for categorical splits in tree code.
# - _isfinite: GIL-free NaN/inf detection for input validation.
# - _seq_dataset / _weight_vector (Tempita templates, float32 and float64
#   variants): row-wise dataset access and scaled weight vectors for the
#   SGD/SAG stochastic solvers; _seq_dataset uses _random for shuffling,
#   _weight_vector uses _cython_blas.
# - arrayfuncs: min_pos, row-equality check and Cholesky deletion (LARS).
# - murmurhash: MurmurHash3 32-bit wrappers (feature hashing, murmurhash3_32).
# - sparsefuncs_fast: CSR/CSC kernels (row norms, NaN-aware weighted
#   mean/variance, incremental variance, row normalisation, row assignment,
#   CSR @ CSR to dense) behind sklearn.utils.sparsefuncs.
# (This __init__.pxd holds only documentation; it declares nothing.)
