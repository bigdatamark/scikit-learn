# [MODULE DOC]
# Array-API-compatible numpy.linalg namespace: clones numpy.linalg, re-exports matmul,
# matrix_transpose, tensordot and vecdot, binds the common._linalg wrappers (named-tuple
# results, rtol/upper keywords, svdvals, matrix_norm, last-two-axes diagonal/trace) to
# numpy, and defines NumPy-specific solve, eig and eigvals built on NumPy's private
# _umath_linalg gufuncs. vector_norm is NumPy's own when present (NumPy 2.x).
# __all__ is the sorted union of numpy.linalg's names, common._linalg.__all__ and an
# explicit list of extra standard names.
# pyright: reportAttributeAccessIssue=false
# pyright: reportUnknownArgumentType=false
# pyright: reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false

from __future__ import annotations

import numpy as np

from .._internal import clone_module, get_xp
from ..common import _linalg

__all__ = clone_module("numpy.linalg", globals())

# These functions are in both the main and linalg namespaces
from ._aliases import matmul, matrix_transpose, tensordot, vecdot  # noqa: F401
from ._typing import Array

cross = get_xp(np)(_linalg.cross)
outer = get_xp(np)(_linalg.outer)
EighResult = _linalg.EighResult
EigResult = _linalg.EigResult
QRResult = _linalg.QRResult
SlogdetResult = _linalg.SlogdetResult
SVDResult = _linalg.SVDResult
eigh = get_xp(np)(_linalg.eigh)
qr = get_xp(np)(_linalg.qr)
slogdet = get_xp(np)(_linalg.slogdet)
svd = get_xp(np)(_linalg.svd)
cholesky = get_xp(np)(_linalg.cholesky)
matrix_rank = get_xp(np)(_linalg.matrix_rank)
pinv = get_xp(np)(_linalg.pinv)
matrix_norm = get_xp(np)(_linalg.matrix_norm)
svdvals = get_xp(np)(_linalg.svdvals)
diagonal = get_xp(np)(_linalg.diagonal)
trace = get_xp(np)(_linalg.trace)

# Note: unlike np.linalg.solve, the array API solve() only accepts x2 as a
# vector when it is exactly 1-dimensional. All other cases treat x2 as a stack
# of matrices. The np.linalg.solve behavior of allowing stacks of both
# matrices and vectors is ambiguous c.f.
# https://github.com/numpy/numpy/issues/15349 and
# https://github.com/data-apis/array-api/issues/285.

# To workaround this, the below is the code from np.linalg.solve except
# only calling solve1 in the exactly 1D case.


# This code is here instead of in common because it is numpy specific. Also
# note that CuPy's solve() does not currently support broadcasting (see
# https://github.com/cupy/cupy/blob/main/cupy/cublas.py#L43).
def solve(x1: Array, x2: Array, /) -> Array:
    # [FUNCTION DOC]
    # Array-API solve: same as numpy.linalg.solve except x2 is treated as a vector only when
    # it is exactly 1-D (otherwise a stack of matrices), avoiding NumPy's ambiguous rule.
    # Imports private helpers from numpy.linalg._linalg (NumPy 2) or numpy.linalg.linalg
    # (NumPy 1); checks x1 is a stack of square matrices; determines the computation and
    # result types; calls _umath_linalg.solve1 (x2 1-D) or solve with a double or complex
    # signature under an errstate that turns invalid results into LinAlgError; casts to the
    # result type and re-wraps to the input array subclass.
    # Inputs: x1 - (..., M, M) array; x2 - (M,) or (..., M, K) array.
    # Outputs: solution array.
    # Exceptions: LinAlgError for non-square/non-2-D x1 or singular matrices.
    # KNOWN ISSUE: depends on private NumPy internals that may change between releases.
    try:
        from numpy.linalg._linalg import (  # type: ignore[attr-defined]
            _assert_stacked_2d,
            _assert_stacked_square,
            _commonType,
            _makearray,
            _raise_linalgerror_singular,
            isComplexType,
        )
    except ImportError:
        from numpy.linalg.linalg import (  # type: ignore[attr-defined]
            _assert_stacked_2d,
            _assert_stacked_square,
            _commonType,
            _makearray,
            _raise_linalgerror_singular,
            isComplexType,
        )
    from numpy.linalg import _umath_linalg

    x1, _ = _makearray(x1)
    _assert_stacked_2d(x1)
    _assert_stacked_square(x1)
    x2, wrap = _makearray(x2)
    t, result_t = _commonType(x1, x2)

    # This part is different from np.linalg.solve
    gufunc: np.ufunc
    if x2.ndim == 1:
        gufunc = _umath_linalg.solve1
    else:
        gufunc = _umath_linalg.solve

    # This does nothing currently but is left in because it will be relevant
    # when complex dtype support is added to the spec in 2022.
    signature = "DD->D" if isComplexType(t) else "dd->d"
    with np.errstate(
        call=_raise_linalgerror_singular,
        invalid="call",
        over="ignore",
        divide="ignore",
        under="ignore",
    ):
        r: Array = gufunc(x1, x2, signature=signature)

    return wrap(r.astype(result_t, copy=False))


# Unlike numpy.linalg.eig, Array API version always returns complex results

def eig(x: Array, /) -> tuple[Array, Array]:
    # [FUNCTION DOC]
    # Eigen-decomposition of a (stack of) square matrices that always returns complex
    # results (numpy.linalg.eig returns real ones when possible). Uses private NumPy helpers
    # to validate (square, finite), runs _umath_linalg.eig with a complex-output signature,
    # casts eigenvalues and eigenvectors to the complex counterpart of the result type.
    # Inputs: x - (..., M, M) array.
    # Outputs: EigResult(eigenvalues (..., M), eigenvectors (..., M, M)), complex dtype.
    # Exceptions: LinAlgError for non-square, non-finite input or non-convergence.
    # KNOWN ISSUE: the return annotation says a plain tuple but EigResult is returned; only
    # the eigenvectors are passed through the input-subclass wrap.
    try:
        from numpy.linalg._linalg import (  # type: ignore[attr-defined]
            _assert_stacked_square,
            _assert_finite,
            _commonType,
            _makearray,
            _raise_linalgerror_eigenvalues_nonconvergence,
            isComplexType,
            _complexType,
        )
    except ImportError:
        from numpy.linalg.linalg import (  # type: ignore[attr-defined]
            _assert_stacked_square,
            _assert_finite,
            _commonType,
            _makearray,
            _raise_linalgerror_eigenvalues_nonconvergence,
            isComplexType,
            _complexType,
        )
    from numpy.linalg import _umath_linalg

    x, wrap = _makearray(x)
    _assert_stacked_square(x)
    _assert_finite(x)
    t, result_t = _commonType(x)

    signature = 'D->DD' if isComplexType(t) else 'd->DD'
    with np.errstate(call=_raise_linalgerror_eigenvalues_nonconvergence,
                  invalid='call', over='ignore', divide='ignore',
                  under='ignore'):
        w, vt = _umath_linalg.eig(x, signature=signature)

    result_t = _complexType(result_t)
    vt = vt.astype(result_t, copy=False)
    return EigResult(w.astype(result_t, copy=False), wrap(vt))


def eigvals(x: Array, /) -> Array:
    # [FUNCTION DOC]
    # Eigenvalues only, always complex; same validation and errstate handling as eig, using
    # _umath_linalg.eigvals.
    # Inputs: x - (..., M, M) array.
    # Outputs: complex array (..., M).
    # Exceptions: LinAlgError for non-square, non-finite input or non-convergence.
    try:
        from numpy.linalg._linalg import (  # type: ignore[attr-defined]
            _assert_stacked_square,
            _assert_finite,
            _commonType,
            _makearray,
            _raise_linalgerror_eigenvalues_nonconvergence,
            isComplexType,
            _complexType,
        )
    except ImportError:
        from numpy.linalg.linalg import (  # type: ignore[attr-defined]
            _assert_stacked_square,
            _assert_finite,
            _commonType,
            _makearray,
            _raise_linalgerror_eigenvalues_nonconvergence,
            isComplexType,
            _complexType,
        )
    from numpy.linalg import _umath_linalg

    x, wrap = _makearray(x)
    _assert_stacked_square(x)
    _assert_finite(x)
    t, result_t = _commonType(x)

    signature = 'D->D' if isComplexType(t) else 'd->D'
    with np.errstate(call=_raise_linalgerror_eigenvalues_nonconvergence,
                  invalid='call', over='ignore', divide='ignore',
                  under='ignore'):
        w = _umath_linalg.eigvals(x, signature=signature)

    result_t = _complexType(result_t)
    return w.astype(result_t, copy=False)


# These functions are completely new here. If the library already has them
# (i.e., numpy 2.0), use the library version instead of our wrapper.
if hasattr(np.linalg, "vector_norm"):
    vector_norm = np.linalg.vector_norm
else:
    vector_norm = get_xp(np)(_linalg.vector_norm)


_all = [
    "LinAlgError",
    "cond",
    "det",
    "eig",
    "eigvals",
    "eigvalsh",
    "inv",
    "lstsq",
    "matrix_power",
    "multi_dot",
    "norm",
    "solve", 
    "tensorinv",
    "tensorsolve",
    "vector_norm",
]
__all__ = sorted(set(__all__) | set(_linalg.__all__) | set(_all))

def __dir__() -> list[str]:
    # [FUNCTION DOC]
    # HELPER: Module __dir__ hook returning __all__.
    return __all__
