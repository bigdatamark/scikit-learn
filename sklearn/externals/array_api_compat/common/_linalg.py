# [MODULE DOC]
# Namespace-generic linear algebra wrappers that adapt NumPy-like backends (NumPy, CuPy,
# Dask) to the array API standard linalg extension: named-tuple results, extra keyword
# arguments (upper, rtol), functions missing from the backend (svdvals, matrix_norm,
# vector_norm, cross, outer) and last-two-axes semantics for diagonal/trace. Every
# function takes the backend namespace as xp. At import time picks
# normalize_axis_tuple from numpy.lib.array_utils on NumPy 2.x, else numpy.core.numeric.
# KNOWN ISSUE: the NumPy version check looks only at the first character of
# np.__version__, so a hypothetical NumPy 10+ would take the 1.x import path.
from __future__ import annotations

import math
from typing import Literal, NamedTuple, cast

import numpy as np

if np.__version__[0] == "2":
    from numpy.lib.array_utils import normalize_axis_tuple
else:
    from numpy.core.numeric import normalize_axis_tuple  # type: ignore[no-redef]

from .._internal import get_xp
from ._aliases import isdtype, matmul, matrix_transpose, tensordot, vecdot
from ._typing import Array, DType, JustFloat, JustInt, Namespace


# These are in the main NumPy namespace but not in numpy.linalg
def cross(
    x1: Array,
    x2: Array,
    /,
    xp: Namespace,
    *,
    axis: int = -1,
    **kwargs: object,
) -> Array:
    # [FUNCTION DOC]
    # Cross product of x1 and x2 along axis via xp.cross.
    # Inputs: x1, x2 - broadcastable arrays whose size along axis is 3; xp - namespace;
    # axis - default -1; kwargs - forwarded to xp.cross.
    # Outputs: array of cross products.
    # Exceptions: backend errors for incompatible shapes.
    return xp.cross(x1, x2, axis=axis, **kwargs)

def outer(x1: Array, x2: Array, /, xp: Namespace, **kwargs: object) -> Array:
    # [FUNCTION DOC]
    # Outer product of two 1-D arrays via xp.outer; kwargs forwarded.
    # Outputs: 2-D array of shape (len(x1), len(x2)).
    return xp.outer(x1, x2, **kwargs)

class EighResult(NamedTuple):
    # [CLASS DOC]
    # Named-tuple result of eigh: fields eigenvalues and eigenvectors (arrays).
    eigenvalues: Array
    eigenvectors: Array

class EigResult(NamedTuple):
    # [CLASS DOC]
    # Named-tuple result of a general eigen-decomposition: eigenvalues, eigenvectors.
    # KNOWN ISSUE: defined but not used by any function here and not listed in __all__.
    eigenvalues: Array
    eigenvectors: Array

class QRResult(NamedTuple):
    # [CLASS DOC]
    # Named-tuple result of qr: fields Q (orthonormal factor) and R (upper triangular).
    Q: Array
    R: Array

class SlogdetResult(NamedTuple):
    # [CLASS DOC]
    # Named-tuple result of slogdet: sign and logabsdet arrays.
    sign: Array
    logabsdet: Array

class SVDResult(NamedTuple):
    # [CLASS DOC]
    # Named-tuple result of svd: U, S (singular values) and Vh.
    U: Array
    S: Array
    Vh: Array

# These functions are the same as their NumPy counterparts except they return
# a namedtuple.
def eigh(x: Array, /, xp: Namespace, **kwargs: object) -> EighResult:
    # [FUNCTION DOC]
    # Eigen-decomposition of a (stack of) Hermitian/symmetric matrices via xp.linalg.eigh,
    # wrapped into EighResult.
    # Inputs: x - array with shape (..., M, M); xp - namespace; kwargs forwarded.
    # Outputs: EighResult(eigenvalues, eigenvectors).
    # Exceptions: backend LinAlgError on non-convergence or bad shapes.
    return EighResult(*xp.linalg.eigh(x, **kwargs))

def qr(
    x: Array,
    /,
    xp: Namespace,
    *,
    mode: Literal["reduced", "complete"] = "reduced",
    **kwargs: object,
) -> QRResult:
    # [FUNCTION DOC]
    # QR factorization via xp.linalg.qr, wrapped in QRResult.
    # Inputs: x - (..., M, N) array; mode - "reduced" (default) or "complete"; kwargs
    # forwarded.
    # Outputs: QRResult(Q, R).
    return QRResult(*xp.linalg.qr(x, mode=mode, **kwargs))

def slogdet(x: Array, /, xp: Namespace, **kwargs: object) -> SlogdetResult:
    # [FUNCTION DOC]
    # Sign and natural log of the absolute determinant via xp.linalg.slogdet.
    # Outputs: SlogdetResult(sign, logabsdet).
    return SlogdetResult(*xp.linalg.slogdet(x, **kwargs))

def svd(
    x: Array,
    /,
    xp: Namespace,
    *,
    full_matrices: bool = True,
    **kwargs: object,
) -> SVDResult:
    # [FUNCTION DOC]
    # Singular value decomposition via xp.linalg.svd, wrapped in SVDResult.
    # Inputs: x - (..., M, N) array; full_matrices - default True; kwargs forwarded.
    # Outputs: SVDResult(U, S, Vh).
    return SVDResult(*xp.linalg.svd(x, full_matrices=full_matrices, **kwargs))

# These functions have additional keyword arguments

# The upper keyword argument is new from NumPy
def cholesky(
    x: Array,
    /,
    xp: Namespace,
    *,
    upper: bool = False,
    **kwargs: object,
) -> Array:
    # [FUNCTION DOC]
    # Cholesky factorization with the array-API "upper" option. Computes the lower factor
    # L with xp.linalg.cholesky; if upper is True, returns the matrix transpose of L,
    # additionally conjugated when the dtype is complex (i.e. the conjugate transpose).
    # Inputs: x - (..., M, M) positive-definite array; upper - default False; kwargs
    # forwarded to the backend.
    # Outputs: lower or upper triangular factor.
    # Exceptions: backend LinAlgError for non positive-definite input.
    L = xp.linalg.cholesky(x, **kwargs)
    if upper:
        U = get_xp(xp)(matrix_transpose)(L)
        if get_xp(xp)(isdtype)(U.dtype, 'complex floating'):
            U = xp.conj(U)  # pyright: ignore[reportConstantRedefinition]
        return U
    return L

# The rtol keyword argument of matrix_rank() and pinv() is new from NumPy.
# Note that it has a different semantic meaning from tol and rcond.
def matrix_rank(
    x: Array,
    /,
    xp: Namespace,
    *,
    rtol: float | Array | None = None,
    **kwargs: object,
) -> Array:
    # [FUNCTION DOC]
    # Numerical rank following array-API semantics. Computes the singular values S via
    # svdvals; the tolerance per matrix is the largest singular value times either
    # max(M, N) * machine eps of S's dtype (rtol None) or rtol (broadcast with a trailing
    # new axis); returns the count of singular values strictly greater than it.
    # Inputs: x - (..., M, N) array with ndim >= 2; rtol - float, array or None; kwargs -
    # forwarded to svdvals.
    # Outputs: integer array of shape x.shape[:-2].
    # Exceptions: xp.linalg.LinAlgError when x has fewer than 2 dimensions.
    # KNOWN ISSUE: kwargs are forwarded to svdvals, whose signature accepts no extra
    # keywords, so any extra keyword raises TypeError.
    # this is different from xp.linalg.matrix_rank, which supports 1
    # dimensional arrays.
    if x.ndim < 2:
        raise xp.linalg.LinAlgError("1-dimensional array given. Array must be at least two-dimensional")
    S: Array = get_xp(xp)(svdvals)(x, **kwargs)
    if rtol is None:
        tol = S.max(axis=-1, keepdims=True) * max(x.shape[-2:]) * xp.finfo(S.dtype).eps
    else:
        # this is different from xp.linalg.matrix_rank, which does not
        # multiply the tolerance by the largest singular value.
        tol = S.max(axis=-1, keepdims=True)*xp.asarray(rtol)[..., xp.newaxis]
    return xp.count_nonzero(S > tol, axis=-1)

def pinv(
    x: Array,
    /,
    xp: Namespace,
    *,
    rtol: float | Array | None = None,
    **kwargs: object,
) -> Array:
    # [FUNCTION DOC]
    # Moore-Penrose pseudo-inverse via xp.linalg.pinv, passing rtol as rcond. When rtol is
    # None the default is max(M, N) * eps of x's dtype (array-API default).
    # Inputs: x - (..., M, N) array; rtol - float/array/None; kwargs forwarded.
    # Outputs: (..., N, M) array.
    # Exceptions: xp.finfo raises for non-floating x when rtol is None.
    # this is different from xp.linalg.pinv, which does not multiply the
    # default tolerance by max(M, N).
    if rtol is None:
        rtol = max(x.shape[-2:]) * xp.finfo(x.dtype).eps
    return xp.linalg.pinv(x, rcond=rtol, **kwargs)

# These functions are new in the array API spec

def matrix_norm(
    x: Array,
    /,
    xp: Namespace,
    *,
    keepdims: bool = False,
    ord: Literal[1, 2, -1, -2] | JustFloat | Literal["fro", "nuc"] | None = "fro",
) -> Array:
    # [FUNCTION DOC]
    # Matrix norm over the last two axes via xp.linalg.norm.
    # Inputs: x - (..., M, N) array; keepdims - default False; ord - 1, 2, -1, -2, float,
    # "fro" (default) or "nuc".
    # Outputs: array of norms of shape x.shape[:-2] (or with size-1 dims if keepdims).
    return xp.linalg.norm(x, axis=(-2, -1), keepdims=keepdims, ord=ord)

# svdvals is not in NumPy (but it is in SciPy). It is equivalent to
# xp.linalg.svd(compute_uv=False).
def svdvals(x: Array, /, xp: Namespace) -> Array | tuple[Array, ...]:
    # [FUNCTION DOC]
    # Singular values only, computed by xp.linalg.svd with compute_uv=False.
    # Inputs: x - (..., M, N) array; xp - namespace.
    # Outputs: array of shape (..., min(M, N)).
    return xp.linalg.svd(x, compute_uv=False)

def vector_norm(
    x: Array,
    /,
    xp: Namespace,
    *,
    axis: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
    ord: JustInt | JustFloat = 2,
) -> Array:
    # [FUNCTION DOC]
    # Vector norm over arbitrary axes, avoiding xp.linalg.norm's matrix-norm behaviour.
    # Algorithm: if axis is None, flatten x and use axis 0; if axis is a tuple, normalise
    # it, move those axes to the front by transposing and collapse them into one leading
    # axis (product of their sizes), then use axis 0; otherwise use x and the int axis
    # directly. Compute xp.linalg.norm with ord. If keepdims, reshape the result to x's
    # shape with every reduced axis set to 1.
    # Inputs: x - array; axis - None, int or tuple of ints; keepdims - default False;
    # ord - int or float, default 2.
    # Outputs: array of norms.
    # Exceptions: normalize_axis_tuple raises AxisError for out-of-range/duplicate axes.
    # KNOWN ISSUE: the transpose permutation uses the raw (possibly negative) axis tuple
    # rather than normalized_axis; this works for backends accepting negative axes but is
    # inconsistent with how rest is computed.
    # xp.linalg.norm tries to do a matrix norm whenever axis is a 2-tuple or
    # when axis=None and the input is 2-D, so to force a vector norm, we make
    # it so the input is 1-D (for axis=None), or reshape so that norm is done
    # on a single dimension.
    if axis is None:
        # Note: xp.linalg.norm() doesn't handle 0-D arrays
        _x = x.ravel()
        _axis = 0
    elif isinstance(axis, tuple):
        # Note: The axis argument supports any number of axes, whereas
        # xp.linalg.norm() only supports a single axis for vector norm.
        normalized_axis = cast(
            "tuple[int, ...]",
            normalize_axis_tuple(axis, x.ndim),  # pyright: ignore[reportCallIssue]
        )
        rest = tuple(i for i in range(x.ndim) if i not in normalized_axis)
        newshape = axis + rest
        _x = xp.transpose(x, newshape).reshape(
            (math.prod([x.shape[i] for i in axis]), *[x.shape[i] for i in rest]))
        _axis = 0
    else:
        _x = x
        _axis = axis

    res = xp.linalg.norm(_x, axis=_axis, ord=ord)

    if keepdims:
        # We can't reuse xp.linalg.norm(keepdims) because of the reshape hacks
        # above to avoid matrix norm logic.
        shape = list(x.shape)
        axes = cast(
            "tuple[int, ...]",
            normalize_axis_tuple(  # pyright: ignore[reportCallIssue]
                range(x.ndim) if axis is None else axis,
                x.ndim,
            ),
        )
        for i in axes:
            shape[i] = 1
        res = xp.reshape(res, tuple(shape))

    return res

# xp.diagonal and xp.trace operate on the first two axes whereas these
# operates on the last two

def diagonal(x: Array, /, xp: Namespace, *, offset: int = 0, **kwargs: object) -> Array:
    # [FUNCTION DOC]
    # Returns the given diagonal of the last two axes via xp.diagonal with axis1=-2,
    # axis2=-1.
    # Inputs: x - (..., M, N) array; offset - default 0; kwargs forwarded.
    # Outputs: array of shape (..., K).
    return xp.diagonal(x, offset=offset, axis1=-2, axis2=-1, **kwargs)

def trace(
    x: Array,
    /,
    xp: Namespace,
    *,
    offset: int = 0,
    dtype: DType | None = None,
    **kwargs: object,
) -> Array:
    # [FUNCTION DOC]
    # Sum along the given diagonal of the last two axes via xp.trace(axis1=-2, axis2=-1),
    # wrapped in xp.asarray so a 0-D result is an array rather than a scalar.
    # Inputs: x - (..., M, N) array; offset - default 0; dtype - optional accumulator /
    # output dtype; kwargs forwarded.
    # Outputs: array of shape x.shape[:-2].
    return xp.asarray(
        xp.trace(x, offset=offset, dtype=dtype, axis1=-2, axis2=-1, **kwargs)
    )

__all__ = ['cross', 'matmul', 'outer', 'tensordot', 'EighResult',
           'QRResult', 'SlogdetResult', 'SVDResult', 'eigh', 'qr', 'slogdet',
           'svd', 'cholesky', 'matrix_rank', 'pinv', 'matrix_norm',
           'matrix_transpose', 'svdvals', 'vecdot', 'vector_norm', 'diagonal',
           'trace']


def __dir__() -> list[str]:
    # [FUNCTION DOC]
    # HELPER: Module __dir__ hook returning __all__.
    return __all__
