# [MODULE DOC]
# Array-API compatible torch.linalg namespace. Clones torch.linalg, adds outer, matmul,
#   matrix_transpose and tensordot from the main namespace, and overrides cross, vecdot,
#   solve, trace and vector_norm to fix dtype promotion, broadcasting and argument
#   naming differences from the array API standard.
# Side effects: importing torch and the sibling _aliases module.
from __future__ import annotations

import torch
import torch.linalg

from .._internal import clone_module

__all__ = clone_module("torch.linalg", globals())

# outer is implemented in torch but aren't in the linalg namespace
from torch import outer
from ._aliases import _fix_promotion, sum
# These functions are in both the main and linalg namespaces
from ._aliases import matmul, matrix_transpose, tensordot
from ._typing import Array, DType
from ..common._typing import JustInt, JustFloat

# Note: torch.linalg.cross does not default to axis=-1 (it defaults to the
# first axis with size 3), see https://github.com/pytorch/pytorch/issues/58743

# torch.cross also does not support broadcasting when it would add new
# dimensions https://github.com/pytorch/pytorch/issues/39656
def cross(x1: Array, x2: Array, /, *, axis: int = -1) -> Array:
    # [FUNCTION DOC]
    # Array-API cross product along axis (default -1, unlike torch's first-size-3 axis
    #   default). Applies array-API type promotion, validates axis, broadcasts the
    #   inputs to a common shape, then calls torch.linalg.cross with dim=axis.
    # Inputs: x1, x2 - tensors whose size along axis is 3; axis - int.
    # Outputs: tensor of the broadcast shape.
    # Exceptions: ValueError if axis is outside [-min(ndim), max(ndim)) or if either
    #   size along axis is not 3.
    # KNOWN ISSUE: the bounds check is done before broadcasting using min/max of ndims,
    #   so a negative axis valid only for the higher-dimensional input is rejected, and
    #   the size check indexes each input with the un-broadcast axis.
    x1, x2 = _fix_promotion(x1, x2, only_scalar=False)
    if not (-min(x1.ndim, x2.ndim) <= axis < max(x1.ndim, x2.ndim)):
        raise ValueError(f"axis {axis} out of bounds for cross product of arrays with shapes {x1.shape} and {x2.shape}")
    if not (x1.shape[axis] == x2.shape[axis] == 3):
        raise ValueError(f"cross product axis must have size 3, got {x1.shape[axis]} and {x2.shape[axis]}")
    x1, x2 = torch.broadcast_tensors(x1, x2)
    return torch.linalg.cross(x1, x2, dim=axis)

def vecdot(x1: Array, x2: Array, /, *, axis: int = -1, **kwargs: object) -> Array:
    # [FUNCTION DOC]
    # Array-API vector dot product (conjugating x1 for complex) along axis. Applies type
    #   promotion and requires equal sizes along axis. For integer dtypes (unsupported
    #   by torch.linalg.vecdot) it moves axis to the end, broadcasts, and computes a
    #   batched row-by-column matrix product, returning the scalar per batch; otherwise
    #   forwards to torch.linalg.vecdot with dim=axis.
    # Inputs: x1, x2 - tensors; axis - int (default -1); kwargs - forwarded to torch.
    # Outputs: tensor with axis removed and remaining dims broadcast.
    # Exceptions: ValueError if sizes along axis differ; RuntimeError if kwargs are
    #   given with integral dtypes.
    from ._aliases import isdtype

    x1, x2 = _fix_promotion(x1, x2, only_scalar=False)

    # torch.linalg.vecdot incorrectly allows broadcasting along the contracted dimension
    if x1.shape[axis] != x2.shape[axis]:
        raise ValueError("x1 and x2 must have the same size along the given axis")

    # torch.linalg.vecdot doesn't support integer dtypes
    if isdtype(x1.dtype, 'integral') or isdtype(x2.dtype, 'integral'):
        if kwargs:
            raise RuntimeError("vecdot kwargs not supported for integral dtypes")

        x1_ = torch.moveaxis(x1, axis, -1)
        x2_ = torch.moveaxis(x2, axis, -1)
        x1_, x2_ = torch.broadcast_tensors(x1_, x2_)

        res = x1_[..., None, :] @ x2_[..., None]
        return res[..., 0, 0]
    return torch.linalg.vecdot(x1, x2, dim=axis, **kwargs)

def solve(x1: Array, x2: Array, /, **kwargs: object) -> Array:
    # [FUNCTION DOC]
    # Array-API linear solve of x1 @ result = x2. Applies type promotion, then, to
    #   defeat torch's NumPy-1 style vector-rhs heuristic, prepends a length-1 axis to
    #   x2 when x2 is not 1-D and has shape equal to x1.shape without its last dim.
    #   Forwards to torch.linalg.solve.
    # Inputs: x1 - (..., M, M) tensor; x2 - (..., M) or (..., M, K) tensor; kwargs
    #   forwarded.
    # Outputs: solution tensor. Exceptions: from torch (e.g. singular matrix).
    x1, x2 = _fix_promotion(x1, x2, only_scalar=False)
    # Torch tries to emulate NumPy 1 solve behavior by using batched 1-D solve
    # whenever
    # 1. x1.ndim - 1 == x2.ndim
    # 2. x1.shape[:-1] == x2.shape
    #
    # See linalg_solve_is_vector_rhs in
    # aten/src/ATen/native/LinearAlgebraUtils.h and
    # TORCH_META_FUNC(_linalg_solve_ex) in
    # aten/src/ATen/native/BatchLinearAlgebra.cpp in the PyTorch source code.
    #
    # The easiest way to work around this is to prepend a size 1 dimension to
    # x2, since x2 is already one dimension less than x1.
    #
    # See https://github.com/pytorch/pytorch/issues/52915
    if x2.ndim != 1 and x1.ndim - 1 == x2.ndim and x1.shape[:-1] == x2.shape:
        x2 = x2[None]
    return torch.linalg.solve(x1, x2, **kwargs)

# torch.trace doesn't support the offset argument and doesn't support stacking
def trace(x: Array, /, *, offset: int = 0, dtype: DType | None = None) -> Array:
    # [FUNCTION DOC]
    # Sum of diagonals over the last two axes, supporting offset and stacked inputs:
    #   takes torch.diagonal on the last two dims then reduces with the wrapped sum
    #   (which upcasts ints per the standard).
    # Inputs: x - (..., M, N) tensor; offset - diagonal offset; dtype - optional result
    #   dtype.
    # Outputs: tensor of shape x.shape[:-2].
    # Use our wrapped sum to make sure it does upcasting correctly
    return sum(torch.diagonal(x, offset=offset, dim1=-2, dim2=-1), axis=-1, dtype=dtype)

def vector_norm(
    x: Array,
    /,
    *,
    axis: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
    # JustFloat stands for inf | -inf, which are not valid for Literal
    ord: JustInt | JustFloat = 2,
    **kwargs: object,
) -> Array:
    # [FUNCTION DOC]
    # Array-API vector norm. When axis is an empty tuple (torch would treat it like
    #   None) the result is elementwise: abs(x), or x != 0 when ord is 0, written into
    #   kwargs['out'] if provided, else a new zeros_like tensor (real dtype for complex
    #   input). Otherwise forwards to torch.linalg.vector_norm with keepdim=keepdims.
    # Inputs: x - tensor; axis - None, int, or tuple; keepdims - bool; ord - int or
    #   +/-inf (default 2); kwargs forwarded (out honored in the empty-axis path).
    # Outputs: tensor of norms. Side effects: mutates the provided out tensor in the
    #   empty-axis path.
    # torch.vector_norm incorrectly treats axis=() the same as axis=None
    if axis == ():
        out = kwargs.get('out')
        if out is None:
            dtype = None
            if x.dtype == torch.complex64:
                dtype = torch.float32
            elif x.dtype == torch.complex128:
                dtype = torch.float64

            out = torch.zeros_like(x, dtype=dtype)

        # The norm of a single scalar works out to abs(x) in every case except
        # for ord=0, which is x != 0.
        if ord == 0:
            out[:] = (x != 0)
        else:
            out[:] = torch.abs(x)
        return out
    return torch.linalg.vector_norm(x, ord=ord, axis=axis, keepdim=keepdims, **kwargs)

__all__ += ['outer', 'matmul', 'matrix_transpose', 'tensordot',
            'cross', 'vecdot', 'solve', 'trace', 'vector_norm']

def __dir__() -> list[str]:
    # [FUNCTION DOC]
    # HELPER: Module-level dir() hook returning __all__.
    return __all__
