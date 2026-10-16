# [MODULE DOC]
# CuPy-specific array API aliases. Defines spec names that are renames of CuPy
# functions (acos, concat, pow, bool, ...), binds the generic implementations from
# common._aliases to cupy via get_xp, and implements CuPy-specific wrappers below.
# vecdot, isdtype and unstack use CuPy's native versions when present. __all__ is the
# common alias list plus the names added here. Imports cupy at import time.
from __future__ import annotations

from builtins import bool as py_bool
from typing import Literal

import cupy as cp

from ..common import _aliases, _helpers
from ..common._typing import NestedSequence, SupportsBufferProtocol
from .._internal import get_xp
from ._typing import Array, Device, DType

bool = cp.bool_

# Basic renames
acos = cp.arccos
acosh = cp.arccosh
asin = cp.arcsin
asinh = cp.arcsinh
atan = cp.arctan
atan2 = cp.arctan2
atanh = cp.arctanh
bitwise_left_shift = cp.left_shift
bitwise_invert = cp.invert
bitwise_right_shift = cp.right_shift
concat = cp.concatenate
pow = cp.power

arange = get_xp(cp)(_aliases.arange)
empty = get_xp(cp)(_aliases.empty)
empty_like = get_xp(cp)(_aliases.empty_like)
eye = get_xp(cp)(_aliases.eye)
full = get_xp(cp)(_aliases.full)
full_like = get_xp(cp)(_aliases.full_like)
linspace = get_xp(cp)(_aliases.linspace)
ones = get_xp(cp)(_aliases.ones)
ones_like = get_xp(cp)(_aliases.ones_like)
zeros = get_xp(cp)(_aliases.zeros)
zeros_like = get_xp(cp)(_aliases.zeros_like)
UniqueAllResult = get_xp(cp)(_aliases.UniqueAllResult)
UniqueCountsResult = get_xp(cp)(_aliases.UniqueCountsResult)
UniqueInverseResult = get_xp(cp)(_aliases.UniqueInverseResult)
unique_all = get_xp(cp)(_aliases.unique_all)
unique_counts = get_xp(cp)(_aliases.unique_counts)
unique_inverse = get_xp(cp)(_aliases.unique_inverse)
unique_values = get_xp(cp)(_aliases.unique_values)
std = get_xp(cp)(_aliases.std)
var = get_xp(cp)(_aliases.var)
cumulative_sum = get_xp(cp)(_aliases.cumulative_sum)
cumulative_prod = get_xp(cp)(_aliases.cumulative_prod)
clip = get_xp(cp)(_aliases.clip)
permute_dims = get_xp(cp)(_aliases.permute_dims)
reshape = get_xp(cp)(_aliases.reshape)
argsort = get_xp(cp)(_aliases.argsort)
sort = get_xp(cp)(_aliases.sort)
nonzero = get_xp(cp)(_aliases.nonzero)
matmul = get_xp(cp)(_aliases.matmul)
matrix_transpose = get_xp(cp)(_aliases.matrix_transpose)
tensordot = get_xp(cp)(_aliases.tensordot)
sign = get_xp(cp)(_aliases.sign)
finfo = get_xp(cp)(_aliases.finfo)
iinfo = get_xp(cp)(_aliases.iinfo)


# asarray also adds the copy keyword, which is not present in numpy 1.0.
def asarray(
    obj: Array | complex | NestedSequence[complex] | SupportsBufferProtocol,
    /,
    *,
    dtype: DType | None = None,
    device: Device | None = None,
    copy: py_bool | None = None,
    **kwargs: object,
) -> Array:
    # [FUNCTION DOC]
    # Spec-compliant asarray with the copy keyword. Runs inside a cupy.cuda.Device(device)
    # context (None means the current device). copy None -> cupy.asarray (copy only if
    # needed); otherwise cupy.array with copy passed through.
    # Inputs: obj - array, scalar, nested sequence or buffer; dtype - optional; device -
    # optional CuPy device or id; copy - None/True/False; kwargs forwarded to CuPy.
    # Outputs: CuPy array.
    # Exceptions: ValueError when copy is False and a new array had to be created.
    """
    Array API compatibility wrapper for asarray().

    See the corresponding documentation in the array library and/or the array API
    specification for more details.
    """
    with cp.cuda.Device(device):
        if copy is None:
            return cp.asarray(obj, dtype=dtype, **kwargs)
        else:
            res = cp.array(obj, dtype=dtype, copy=copy, **kwargs)
            if not copy and res is not obj:
                raise ValueError("Unable to avoid copy while creating an array as requested")
            return res


def astype(
    x: Array,
    dtype: DType,
    /,
    *,
    copy: py_bool = True,
    device: Device | None = None,
) -> Array:
    # [FUNCTION DOC]
    # Spec-compliant astype with a device keyword. Without device, delegates to
    # x.astype(dtype, copy). With device, casts without copying, moves with
    # _helpers.to_device and makes an explicit copy only if copy is True and the result is
    # still x itself.
    # Inputs: x - CuPy array; dtype; copy - default True; device - optional.
    # Outputs: CuPy array of dtype on device.
    # Exceptions: TypeError from to_device for unsupported device types.
    if device is None:
        return x.astype(dtype=dtype, copy=copy)
    out = _helpers.to_device(x.astype(dtype=dtype, copy=False), device)
    return out.copy() if copy and out is x else out


# cupy.count_nonzero does not have keepdims
def count_nonzero(
    x: Array,
    axis: int | tuple[int, ...] | None = None,
    keepdims: py_bool = False,
) -> Array:
   # [FUNCTION DOC]
   # count_nonzero with the keepdims option missing in CuPy. Counts non-zeros along axis;
   # if keepdims, reshapes to all-ones shape (axis None) or re-inserts the reduced axes
   # with expand_dims.
   # Inputs: x - array; axis - None, int or tuple; keepdims - default False.
   # Outputs: integer array.
   result = cp.count_nonzero(x, axis)
   if keepdims:
       if axis is None:
            return cp.reshape(result, [1]*x.ndim)
       return cp.expand_dims(result, axis)
   return result

# ceil, floor, and trunc return integers for integer inputs

def ceil(x: Array, /) -> Array:
    # [FUNCTION DOC]
    # ceil that returns a copy of x unchanged for integer dtypes (CuPy would return floats),
    # else cupy.ceil. Outputs: array with x's dtype for integer inputs.
    if cp.issubdtype(x.dtype, cp.integer):
        return x.copy()
    return cp.ceil(x)


def floor(x: Array, /) -> Array:
    # [FUNCTION DOC]
    # floor; integer inputs are returned as an unchanged copy, otherwise cupy.floor.
    if cp.issubdtype(x.dtype, cp.integer):
        return x.copy()
    return cp.floor(x)


def trunc(x: Array, /) -> Array:
    # [FUNCTION DOC]
    # trunc; integer inputs are returned as an unchanged copy, otherwise cupy.trunc.
    if cp.issubdtype(x.dtype, cp.integer):
        return x.copy()
    return cp.trunc(x)


# take_along_axis: axis defaults to -1 but in cupy (and numpy) axis is a required arg
def take_along_axis(x: Array, indices: Array, /, *, axis: int = -1) -> Array:
    # [FUNCTION DOC]
    # take_along_axis with axis defaulting to -1 (required in CuPy).
    # Inputs: x - array; indices - integer array broadcastable to x; axis - default -1.
    # Outputs: gathered array.
    return cp.take_along_axis(x, indices, axis=axis)


# https://github.com/cupy/cupy/pull/9582
def broadcast_arrays(*arrays: Array) -> tuple[Array, ...]:
    # [FUNCTION DOC]
    # broadcast_arrays returning a tuple instead of CuPy's list.
    # Outputs: tuple of broadcast arrays (views).
    return tuple(cp.broadcast_arrays(*arrays))


def meshgrid(*arrays: Array, indexing: Literal['xy', 'ij'] = 'xy') -> tuple[Array, ...]:
    # [FUNCTION DOC]
    # meshgrid returning a tuple; indexing "xy" (default) or "ij".
    # Outputs: tuple of coordinate arrays.
    return tuple(cp.meshgrid(*arrays, indexing=indexing))


# Match https://github.com/cupy/cupy/pull/9512/ until cupy v14 is the minimum
# supported version
def searchsorted(
    x1: Array,
    x2: Array | int | float,
    /,
    *,
    side: Literal['left', 'right'] = 'left',
    sorter: Array | None = None
) -> Array:
    # [FUNCTION DOC]
    # searchsorted accepting Python scalars for x2 (converted with cupy.asarray), matching
    # CuPy 14 behaviour.
    # Inputs: x1 - sorted 1-D array; x2 - array or int/float/complex scalar; side - "left"
    # or "right"; sorter - optional index array.
    # Outputs: integer index array.
    # Exceptions: NotImplementedError when x2 is neither a cupy.ndarray nor a scalar.
    if not isinstance(x2, cp.ndarray):
        if not isinstance(x2, int | float | complex):
            raise NotImplementedError(
                'Only python scalars or ndarrays are supported for x2')
        x2 = cp.asarray(x2)
    return cp.searchsorted(x1, x2, side, sorter)


# CuPy isin does not accept scalars
def isin(x1: Array | int, x2: Array | int, /, *, invert: bool = False, **kwds) -> Array:
    # [FUNCTION DOC]
    # isin that accepts Python int scalars for x1/x2 (converted with cupy.asarray).
    # Inputs: x1, x2 - arrays or ints; invert - default False; kwds forwarded.
    # Outputs: boolean array shaped like x1.
    # KNOWN ISSUE: only int scalars are converted; float/complex/bool scalars still reach
    # cupy.isin unconverted. The annotation for invert uses the module's bool, which is
    # cupy.bool_ here (harmless only because annotations are not evaluated).
    if isinstance(x1, int):
        x1 = cp.asarray(x1)
    if isinstance(x2, int):
        x2 = cp.asarray(x2)
    return cp.isin(x1, x2, invert=invert, **kwds)


# These functions are completely new here. If the library already has them
# (i.e., numpy 2.0), use the library version instead of our wrapper.
if hasattr(cp, 'vecdot'):
    vecdot = cp.vecdot
else:
    vecdot = get_xp(cp)(_aliases.vecdot)

if hasattr(cp, 'isdtype'):
    isdtype = cp.isdtype
else:
    isdtype = get_xp(cp)(_aliases.isdtype)

if hasattr(cp, 'unstack'):
    unstack = cp.unstack
else:
    unstack = get_xp(cp)(_aliases.unstack)

__all__ = _aliases.__all__ + ['asarray', 'astype',
                              'acos', 'acosh', 'asin', 'asinh', 'atan',
                              'atan2', 'atanh', 'bitwise_left_shift',
                              'bitwise_invert', 'bitwise_right_shift',
                              'bool', 'concat', 'count_nonzero', 'pow', 'sign',
                              'ceil', 'floor', 'trunc', 'take_along_axis',
                              'broadcast_arrays', 'meshgrid',
                              'searchsorted', 'isin',
]


def __dir__() -> list[str]:
    # [FUNCTION DOC]
    # HELPER: Module __dir__ hook returning __all__.
    return __all__
