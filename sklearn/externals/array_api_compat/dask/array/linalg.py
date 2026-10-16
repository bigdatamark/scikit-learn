# [MODULE DOC]
# Array-API-compatible dask.array.linalg namespace: clones dask.array.linalg, re-exports
# matmul, outer, tensordot, matrix_transpose and vecdot, and adds common._linalg
# wrappers bound to dask (trace, cholesky, matrix_rank, matrix_norm, vector_norm,
# diagonal) plus Dask-specific qr, svd and svdvals. Imports dask at import time.
from __future__ import annotations

from typing import Literal

import dask.array as da

# The `matmul` and `tensordot` functions are in both the main and linalg namespaces
from dask.array import matmul, outer, tensordot

# Exports
from ..._internal import clone_module, get_xp
from ...common import _linalg
from ...common._typing import Array

__all__ = clone_module("dask.array.linalg", globals())

from ._aliases import matrix_transpose, vecdot

EighResult = _linalg.EighResult
QRResult = _linalg.QRResult
SlogdetResult = _linalg.SlogdetResult
SVDResult = _linalg.SVDResult
# TODO: use the QR wrapper once dask
# supports the mode keyword on QR
# https://github.com/dask/dask/issues/10388
#qr = get_xp(da)(_linalg.qr)
def qr(  # type: ignore[no-redef]
    x: Array,
    mode: Literal["reduced", "complete"] = "reduced",
    **kwargs: object,
) -> QRResult:
    # [FUNCTION DOC]
    # QR decomposition restricted to mode "reduced" (dask has no mode keyword), wrapped
    # in QRResult.
    # Inputs: x - 2-D dask array (tall-and-skinny); mode - must be "reduced"; kwargs to
    # dask.array.linalg.qr.
    # Outputs: QRResult(Q, R).
    # Exceptions: ValueError for any other mode.
    if mode != "reduced":
        raise ValueError("dask arrays only support using mode='reduced'")
    return QRResult(*da.linalg.qr(x, **kwargs))
trace = get_xp(da)(_linalg.trace)
cholesky = get_xp(da)(_linalg.cholesky)
matrix_rank = get_xp(da)(_linalg.matrix_rank)
matrix_norm = get_xp(da)(_linalg.matrix_norm)


# Wrap the svd functions to not pass full_matrices to dask
# when full_matrices=False (as that is the default behavior for dask),
# and dask doesn't have the full_matrices keyword
def svd(x: Array, full_matrices: bool = True, **kwargs: object) -> SVDResult:  # type: ignore[no-redef]
    # [FUNCTION DOC]
    # SVD via dask.array.linalg.svd with coerce_signs=False; only the reduced form is
    # supported.
    # Inputs: x - 2-D dask array; full_matrices - must be False; kwargs forwarded.
    # Outputs: tuple (U, S, Vh) as returned by dask.
    # Exceptions: ValueError when full_matrices is True.
    # KNOWN ISSUE: full_matrices defaults to True, so calling svd(x) with defaults always
    # raises; also the result is a plain tuple, not the annotated SVDResult.
    if full_matrices:
        raise ValueError("full_matrices=True is not supported by dask.")
    return da.linalg.svd(x, coerce_signs=False, **kwargs)

def svdvals(x: Array) -> Array:
    # [FUNCTION DOC]
    # Singular values computed through the full svd (dask cannot skip U and Vh).
    # Inputs: x - 2-D dask array. Outputs: 1-D dask array of singular values.
    # KNOWN ISSUE: it calls svd(x) with the default full_matrices=True, which this module's
    # svd rejects, so svdvals always raises ValueError.
    # TODO: can't avoid computing U or V for dask
    _, s, _ =  svd(x)
    return s

vector_norm = get_xp(da)(_linalg.vector_norm)
diagonal = get_xp(da)(_linalg.diagonal)

__all__ += ["trace", "outer", "matmul", "tensordot",
            "matrix_transpose", "vecdot", "EighResult",
            "QRResult", "SlogdetResult", "SVDResult", "qr",
            "cholesky", "matrix_rank", "matrix_norm", "svdvals",
            "vector_norm", "diagonal"]

def __dir__() -> list[str]:
    # [FUNCTION DOC]
    # HELPER: Module __dir__ hook returning __all__.
    return __all__
