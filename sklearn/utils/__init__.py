# [PACKAGE DOC]
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
"""Various utilities to help with development."""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

from sklearn.exceptions import DataConversionWarning
from sklearn.utils import metadata_routing
from sklearn.utils._bunch import Bunch
from sklearn.utils._chunking import gen_batches, gen_even_slices

# Make _safe_indexing importable from here for backward compat as this particular
# helper is considered semi-private and typically very useful for third-party
# libraries that want to comply with scikit-learn's estimator API. In particular,
# _safe_indexing was included in our public API documentation despite the leading
# `_` in its name.
from sklearn.utils._indexing import _safe_indexing, resample, shuffle
from sklearn.utils._mask import safe_mask
from sklearn.utils._repr_html.base import _HTMLDocumentationLinkMixin  # noqa: F401
from sklearn.utils._repr_html.estimator import estimator_html_repr
from sklearn.utils._sparse import _align_api_if_sparse
from sklearn.utils._tags import (
    ClassifierTags,
    InputTags,
    RegressorTags,
    Tags,
    TargetTags,
    TransformerTags,
    get_tags,
)
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
from sklearn.utils.deprecation import deprecated
from sklearn.utils.discovery import all_estimators
from sklearn.utils.extmath import safe_sqr
from sklearn.utils.murmurhash import murmurhash3_32
from sklearn.utils.validation import (
    as_float_array,
    assert_all_finite,
    check_array,
    check_consistent_length,
    check_random_state,
    check_scalar,
    check_symmetric,
    check_X_y,
    column_or_1d,
    indexable,
)

__all__ = [
    "Bunch",
    "ClassifierTags",
    "DataConversionWarning",
    "InputTags",
    "RegressorTags",
    "Tags",
    "TargetTags",
    "TransformerTags",
    "_align_api_if_sparse",
    "_safe_indexing",
    "all_estimators",
    "as_float_array",
    "assert_all_finite",
    "check_X_y",
    "check_array",
    "check_consistent_length",
    "check_random_state",
    "check_scalar",
    "check_symmetric",
    "column_or_1d",
    "compute_class_weight",
    "compute_sample_weight",
    "deprecated",
    "estimator_html_repr",
    "gen_batches",
    "gen_even_slices",
    "get_tags",
    "indexable",
    "metadata_routing",
    "murmurhash3_32",
    "resample",
    "safe_mask",
    "safe_sqr",
    "shuffle",
]
