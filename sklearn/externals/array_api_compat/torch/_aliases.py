# [MODULE DOC]
# Array-API compatible wrappers around torch functions that differ from the standard:
#   'dim' vs 'axis' naming, missing keepdims/axis=() handling, multi-axis reductions,
#   0-D type promotion, creation-function signatures, unique variants, dtype inspection
#   and edge cases (clip, sign, round, take). Also defines the dtype promotion table
#   used by these wrappers.
# Side effects at import: builds _int_dtypes (adding uint16/32/64 when torch >= 2.3
#   exposes them), _array_api_dtypes and a symmetric _promotion_table that includes
#   every same-dtype pair; creates the promoted two-argument elementwise wrappers and
#   several get_xp-bound generic aliases.
from __future__ import annotations

from collections.abc import Sequence
from functools import reduce as _reduce, wraps as _wraps
from builtins import all as _builtin_all, any as _builtin_any
from typing import Any, Literal
import math

import torch

from .._internal import get_xp
from ..common import _aliases
from ..common._typing import NestedSequence, SupportsBufferProtocol
from ._typing import Array, Device, DType

_int_dtypes = {
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
}
try:
    # torch >=2.3
    _int_dtypes |= {torch.uint16, torch.uint32, torch.uint64}
except AttributeError:
    pass


_array_api_dtypes = {
    torch.bool,
    *_int_dtypes,
    torch.float32,
    torch.float64,
    torch.complex64,
    torch.complex128,
}

_promotion_table = {
    # ints
    (torch.int8, torch.int16): torch.int16,
    (torch.int8, torch.int32): torch.int32,
    (torch.int8, torch.int64): torch.int64,
    (torch.int16, torch.int32): torch.int32,
    (torch.int16, torch.int64): torch.int64,
    (torch.int32, torch.int64): torch.int64,
    # ints and uints (mixed sign)
    (torch.uint8, torch.int8): torch.int16,
    (torch.uint8, torch.int16): torch.int16,
    (torch.uint8, torch.int32): torch.int32,
    (torch.uint8, torch.int64): torch.int64,
    # floats
    (torch.float32, torch.float64): torch.float64,
    # complexes
    (torch.complex64, torch.complex128): torch.complex128,
    # Mixed float and complex
    (torch.float32, torch.complex64): torch.complex64,
    (torch.float32, torch.complex128): torch.complex128,
    (torch.float64, torch.complex64): torch.complex128,
    (torch.float64, torch.complex128): torch.complex128,
}

_promotion_table.update({(b, a): c for (a, b), c in _promotion_table.items()})
_promotion_table.update({(a, a): a for a in _array_api_dtypes})


def _two_arg(f):
    # [FUNCTION DOC]
    # HELPER: Decorator factory that wraps a torch binary function so its two positional
    #   arguments go through _fix_promotion (scalar-aware array-API promotion) before
    #   the call; keyword arguments pass through.
    # Inputs: f - torch callable taking two arrays. Outputs: wrapper preserving f's
    #   metadata, given a generic compat docstring if f has none.
    @_wraps(f)
    def _f(x1, x2, /, **kwargs):
        # [FUNCTION DOC]
        # HELPER: The wrapper: promotes x1, x2 with _fix_promotion and calls f with them
        #   and kwargs.
        x1, x2 = _fix_promotion(x1, x2)
        return f(x1, x2, **kwargs)
    if _f.__doc__ is None:
        _f.__doc__ = f"""\
Array API compatibility wrapper for torch.{f.__name__}.

See the corresponding PyTorch documentation and/or the array API specification
for more details.

"""
    return _f

def _fix_promotion(x1, x2, only_scalar=True):
    # [FUNCTION DOC]
    # HELPER: Applies array-API promotion to two tensors to avoid torch's rule that a
    #   0-D tensor adopts the other operand's dtype. Does nothing unless both are
    #   tensors with array-API dtypes. With only_scalar True (default) a side is cast
    #   only when the other side is 0-D; with False both are always cast to
    #   result_type(x1, x2).
    # Outputs: the possibly-cast pair (x1, x2). Side effects: none (casting creates new
    #   tensors).
    if not isinstance(x1, torch.Tensor) or not isinstance(x2, torch.Tensor):
        return x1, x2
    if x1.dtype not in _array_api_dtypes or x2.dtype not in _array_api_dtypes:
        return x1, x2
    # If an argument is 0-D pytorch downcasts the other argument
    if not only_scalar or x1.shape == ():
        dtype = result_type(x1, x2)
        x2 = x2.to(dtype)
    if not only_scalar or x2.shape == ():
        dtype = result_type(x1, x2)
        x1 = x1.to(dtype)
    return x1, x2


_py_scalars = (bool, int, float, complex)


def result_type(*arrays_and_dtypes: Array | DType | complex) -> DType:
    # [FUNCTION DOC]
    # Array-API result_type. With one argument returns it if it is a dtype, else its
    #   .dtype. With two, delegates to _result_type. With more, moves Python scalars to
    #   the end and reduces left to right with _result_type.
    # Exceptions: ValueError with no arguments or when only scalars are given (3+ args).
    # KNOWN ISSUE: a single Python scalar argument hits the one-argument branch and
    #   fails with AttributeError instead of ValueError; two scalars are passed to
    #   torch.result_type, which rejects them.
    num = len(arrays_and_dtypes)

    if num == 0:
        raise ValueError("At least one array or dtype must be provided")

    elif num == 1:
        x = arrays_and_dtypes[0]
        if isinstance(x, torch.dtype):
            return x
        return x.dtype

    if num == 2:
        x, y = arrays_and_dtypes
        return _result_type(x, y)

    else:
        # sort scalars so that they are treated last
        scalars, others = [], []
        for x in arrays_and_dtypes:
            if isinstance(x, _py_scalars):
                scalars.append(x)
            else:
                others.append(x)
        if not others:
            raise ValueError("At least one array or dtype must be provided")

        # combine left-to-right
        return _reduce(_result_type, others + scalars)


def _result_type(x: Array | DType | complex, y: Array | DType | complex) -> DType:
    # [FUNCTION DOC]
    # HELPER: Pairwise promotion. When neither input is a Python scalar, looks up the
    #   (dtype, dtype) pair in _promotion_table; otherwise (or on a miss, e.g. cross-
    #   kind or non-standard dtypes) converts dtype arguments to empty tensors and
    #   defers to torch.result_type.
    # Outputs: torch dtype.
    if not (isinstance(x, _py_scalars) or isinstance(y, _py_scalars)):
        xdt = x if isinstance(x, torch.dtype) else x.dtype
        ydt = y if isinstance(y, torch.dtype) else y.dtype

        try:
            return _promotion_table[xdt, ydt]
        except KeyError:
            pass

    # This doesn't result_type(dtype, dtype) for non-array API dtypes
    # because torch.result_type only accepts tensors. This does however, allow
    # cross-kind promotion.
    x = torch.tensor([], dtype=x) if isinstance(x, torch.dtype) else x
    y = torch.tensor([], dtype=y) if isinstance(y, torch.dtype) else y
    return torch.result_type(x, y)


def can_cast(from_: DType | Array, to: DType, /) -> bool:
    # [FUNCTION DOC]
    # Returns torch.can_cast(from_, to), first replacing an array argument by its dtype.
    if not isinstance(from_, torch.dtype):
        from_ = from_.dtype
    return torch.can_cast(from_, to)

# Basic renames
bitwise_invert = torch.bitwise_not
newaxis = None
# torch.conj sets the conjugation bit, which breaks conversion to other
# libraries. See https://github.com/data-apis/array-api-compat/issues/173
conj = torch.conj_physical

# Two-arg elementwise functions
# These require a wrapper to do the correct type promotion on 0-D tensors
add = _two_arg(torch.add)
atan2 = _two_arg(torch.atan2)
bitwise_and = _two_arg(torch.bitwise_and)
bitwise_left_shift = _two_arg(torch.bitwise_left_shift)
bitwise_or = _two_arg(torch.bitwise_or)
bitwise_right_shift = _two_arg(torch.bitwise_right_shift)
bitwise_xor = _two_arg(torch.bitwise_xor)
copysign = _two_arg(torch.copysign)
divide = _two_arg(torch.divide)
# Also a rename. torch.equal does not broadcast
equal = _two_arg(torch.eq)
floor_divide = _two_arg(torch.floor_divide)
greater = _two_arg(torch.greater)
greater_equal = _two_arg(torch.greater_equal)
hypot = _two_arg(torch.hypot)
less = _two_arg(torch.less)
less_equal = _two_arg(torch.less_equal)
logaddexp = _two_arg(torch.logaddexp)
# logical functions are not included here because they only accept bool in the
# spec, so type promotion is irrelevant.
maximum = _two_arg(torch.maximum)
minimum = _two_arg(torch.minimum)
multiply = _two_arg(torch.multiply)
not_equal = _two_arg(torch.not_equal)
pow = _two_arg(torch.pow)
remainder = _two_arg(torch.remainder)
subtract = _two_arg(torch.subtract)


def asarray(
    obj: Array | complex | NestedSequence[complex] | SupportsBufferProtocol,
    /,
    *,
    dtype: DType | None = None,
    device: Device | None = None,
    copy: bool | None = None,
    **kwargs: Any,
) -> Array:
    # [FUNCTION DOC]
    # Array-API asarray. If device is not given and obj is a tensor, uses obj's device
    #   so the result stays on it (torch would use the default device), then forwards
    #   everything to torch.asarray.
    # Inputs: obj - tensor, scalar, nested sequence or buffer; dtype, device, copy
    #   (None/True/False); kwargs.
    # Outputs: tensor. Exceptions: from torch (e.g. copy=False needing a copy).
    # torch.asarray does not respect input->output device propagation
    # https://github.com/pytorch/pytorch/issues/150199
    if device is None and isinstance(obj, torch.Tensor):
        device = obj.device
    return torch.asarray(obj, dtype=dtype, device=device, copy=copy, **kwargs)


# These wrappers are mostly based on the fact that pytorch uses 'dim' instead
# of 'axis'.

# torch.min and torch.max return a tuple and don't support multiple axes https://github.com/pytorch/pytorch/issues/58745
def max(x: Array, /, *, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> Array:
    # [FUNCTION DOC]
    # Array-API max reduction via torch.amax (supports tuple axes and keepdims). axis=()
    #   returns a clone of x.
    # https://github.com/pytorch/pytorch/issues/29137
    if axis == ():
        return torch.clone(x)
    return torch.amax(x, axis, keepdims=keepdims)

def min(x: Array, /, *, axis: int | tuple[int, ...] |None = None, keepdims: bool = False) -> Array:
    # [FUNCTION DOC]
    # Array-API min reduction via torch.amin. axis=() returns a clone of x.
    # https://github.com/pytorch/pytorch/issues/29137
    if axis == ():
        return torch.clone(x)
    return torch.amin(x, axis, keepdims=keepdims)

unstack = get_xp(torch)(_aliases.unstack)
cumulative_sum = get_xp(torch)(_aliases.cumulative_sum)
cumulative_prod = get_xp(torch)(_aliases.cumulative_prod)
finfo = get_xp(torch)(_aliases.finfo)
iinfo = get_xp(torch)(_aliases.iinfo)


# torch.sort also returns a tuple
# https://github.com/pytorch/pytorch/issues/70921
def sort(
    x: Array,
    /,
    *,
    axis: int = -1,
    descending: bool = False,
    stable: bool = True,
    **kwargs: object,
) -> Array:
    # [FUNCTION DOC]
    # Sorted values along axis (stable by default, optional descending), taking .values
    #   from torch.sort's result tuple. Returns a new tensor.
    return torch.sort(x, dim=axis, descending=descending, stable=stable, **kwargs).values


# Wrap torch.argsort to set stable=True by default
def argsort(
    x: Array,
    /,
    *,
    axis: int = -1,
    descending: bool = False,
    stable: bool = True,
    **kwargs: object,
) -> Array:
    # [FUNCTION DOC]
    # Indices that sort x along axis; forwards to torch.argsort with stable defaulting
    #   to True.
    return torch.argsort(x, dim=axis, descending=descending, stable=stable, **kwargs)


def _normalize_axes(axis, ndim):
    # [FUNCTION DOC]
    # HELPER: Normalises a tuple of axes for an ndim-dimensional tensor: checks each is
    #   in [-ndim, ndim-1], converts negatives to positives, rejects duplicates, returns
    #   them sorted ascending.
    # Exceptions: IndexError for out-of-range axes (including any axis on a 0-D tensor)
    #   or for repeated axes.
    axes = []
    if ndim == 0 and axis:
        # Better error message in this case
        raise IndexError(f"Dimension out of range: {axis[0]}")
    lower, upper = -ndim, ndim - 1
    for a in axis:
        if a < lower or a > upper:
            # Match torch error message (e.g., from sum())
            raise IndexError(f"Dimension out of range (expected to be in range of [{lower}, {upper}], but got {a}")
        if a < 0:
            a = a + ndim
        if a in axes:
            # Use IndexError instead of RuntimeError, and "axis" instead of "dim"
            raise IndexError(f"Axis {a} appears multiple times in the list of axes")
        axes.append(a)
    return sorted(axes)

def _axis_none_keepdims(x, ndim, keepdims):
    # [FUNCTION DOC]
    # HELPER: Emulates keepdims for a full (axis=None) reduction by prepending ndim
    #   length-1 axes to the reduced result when keepdims is true; otherwise returns it
    #   unchanged.
    # Apply keepdims when axis=None
    # (https://github.com/pytorch/pytorch/issues/71209)
    # Note that this is only valid for the axis=None case.
    if keepdims:
        for _ in range(ndim):
            x = torch.unsqueeze(x, 0)
    return x

def _reduce_multiple_axes(f, x, axis, keepdims=False, **kwargs):
    # [FUNCTION DOC]
    # HELPER: Reduces over several axes with a single-axis reduction f: normalises axes,
    #   moves them (in reverse order) to the end, flattens those trailing axes into one,
    #   applies f along the last axis with kwargs, and if keepdims reinserts size-1 axes
    #   at each original position (ascending order).
    # Outputs: reduced tensor. Exceptions: IndexError from _normalize_axes.
    # Some reductions don't support multiple axes
    # (https://github.com/pytorch/pytorch/issues/56586).
    axes = _normalize_axes(axis, x.ndim)
    for a in reversed(axes):
        x = torch.movedim(x, a, -1)
    x = torch.flatten(x, -len(axes))

    out = f(x, -1, **kwargs)

    if keepdims:
        for a in axes:
            out = torch.unsqueeze(out, a)
    return out


def _sum_prod_no_axis(x: Array, dtype: DType | None) -> Array:
    # [FUNCTION DOC]
    # HELPER: Implements sum/prod with axis=() (no reduction). With explicit dtype
    #   returns a clone (same dtype) or cast copy; otherwise upcasts
    #   uint8/int8/int16/int32 to int64 and clones everything else.
    """
    Implements `sum(..., axis=())` and `prod(..., axis=())`.
    
    Works around https://github.com/pytorch/pytorch/issues/29137
    """
    if dtype is not None:
        return x.clone() if dtype == x.dtype else x.to(dtype)

    # We can't upcast uint8 according to the spec because there is no
    # torch.uint64, so at least upcast to int64 which is what prod does
    # when axis=None.
    if x.dtype in (torch.uint8, torch.int8, torch.int16, torch.int32):
        return x.to(torch.int64)

    return x.clone()


def prod(x: Array,
         /,
         *,
         axis: int | tuple[int, ...] | None = None,
         dtype: DType | None = None,
         keepdims: bool = False,
         **kwargs: object) -> Array:
    # [FUNCTION DOC]
    # Array-API product reduction. axis=() uses _sum_prod_no_axis; tuple axes use
    #   _reduce_multiple_axes with torch.prod; axis=None reduces everything and applies
    #   keepdims manually; an int axis forwards to torch.prod with keepdims.
    # Inputs: x; axis; dtype - optional accumulator/result dtype; keepdims; kwargs
    #   forwarded.

    if axis == ():
        return _sum_prod_no_axis(x, dtype)
    # torch.prod doesn't support multiple axes
    # (https://github.com/pytorch/pytorch/issues/56586).
    if isinstance(axis, tuple):
        return _reduce_multiple_axes(torch.prod, x, axis, keepdims=keepdims, dtype=dtype, **kwargs)
    if axis is None:
        # torch doesn't support keepdims with axis=None
        # (https://github.com/pytorch/pytorch/issues/71209)
        res = torch.prod(x, dtype=dtype, **kwargs)
        res = _axis_none_keepdims(res, x.ndim, keepdims)
        return res

    return torch.prod(x, axis, dtype=dtype, keepdims=keepdims, **kwargs)


def sum(x: Array,
         /,
         *,
         axis: int | tuple[int, ...] | None = None,
         dtype: DType | None = None,
         keepdims: bool = False,
         **kwargs: object) -> Array:
    # [FUNCTION DOC]
    # Array-API sum reduction. axis=() uses _sum_prod_no_axis; axis=None sums everything
    #   with manual keepdims; otherwise (int or tuple, natively supported) forwards to
    #   torch.sum.

    if axis == ():
        return _sum_prod_no_axis(x, dtype)
    if axis is None:
        # torch doesn't support keepdims with axis=None
        # (https://github.com/pytorch/pytorch/issues/71209)
        res = torch.sum(x, dtype=dtype, **kwargs)
        res = _axis_none_keepdims(res, x.ndim, keepdims)
        return res

    return torch.sum(x, axis, dtype=dtype, keepdims=keepdims, **kwargs)

def any(x: Array,
        /,
        *,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
        **kwargs: object) -> Array:
    # [FUNCTION DOC]
    # Array-API logical OR reduction returning bool. axis=() casts x to bool; tuple axes
    #   use _reduce_multiple_axes; axis=None reduces everything with manual keepdims;
    #   int axis forwards to torch.any. Always casts the result to bool (torch returns
    #   uint8 for uint8 input).
    # KNOWN ISSUE: in the int-axis branch kwargs are silently dropped.

    if axis == ():
        return x.to(torch.bool)
    # torch.any doesn't support multiple axes
    # (https://github.com/pytorch/pytorch/issues/56586).
    if isinstance(axis, tuple):
        res = _reduce_multiple_axes(torch.any, x, axis, keepdims=keepdims, **kwargs)
        return res.to(torch.bool)
    if axis is None:
        # torch doesn't support keepdims with axis=None
        # (https://github.com/pytorch/pytorch/issues/71209)
        res = torch.any(x, **kwargs)
        res = _axis_none_keepdims(res, x.ndim, keepdims)
        return res.to(torch.bool)

    # torch.any doesn't return bool for uint8
    return torch.any(x, axis, keepdims=keepdims).to(torch.bool)

def all(x: Array,
        /,
        *,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
        **kwargs: object) -> Array:
    # [FUNCTION DOC]
    # Array-API logical AND reduction returning bool; same structure as any but with
    #   torch.all.
    # KNOWN ISSUE: in the int-axis branch kwargs are silently dropped.

    if axis == ():
        return x.to(torch.bool)
    # torch.all doesn't support multiple axes
    # (https://github.com/pytorch/pytorch/issues/56586).
    if isinstance(axis, tuple):
        res = _reduce_multiple_axes(torch.all, x, axis, keepdims=keepdims, **kwargs)
        return res.to(torch.bool)
    if axis is None:
        # torch doesn't support keepdims with axis=None
        # (https://github.com/pytorch/pytorch/issues/71209)
        res = torch.all(x, **kwargs)
        res = _axis_none_keepdims(res, x.ndim, keepdims)
        return res.to(torch.bool)

    # torch.all doesn't return bool for uint8
    return torch.all(x, axis, keepdims=keepdims).to(torch.bool)

def mean(x: Array,
         /,
         *,
         axis: int | tuple[int, ...] | None = None,
         keepdims: bool = False,
         **kwargs: object) -> Array:
    # [FUNCTION DOC]
    # Array-API mean. axis=() returns a clone; axis=None reduces everything with manual
    #   keepdims; otherwise forwards to torch.mean with keepdims and kwargs.
    # https://github.com/pytorch/pytorch/issues/29137
    if axis == ():
        return torch.clone(x)
    if axis is None:
        # torch doesn't support keepdims with axis=None
        # (https://github.com/pytorch/pytorch/issues/71209)
        res = torch.mean(x, **kwargs)
        res = _axis_none_keepdims(res, x.ndim, keepdims)
        return res
    return torch.mean(x, axis, keepdims=keepdims, **kwargs)

def std(x: Array,
        /,
        *,
        axis: int | tuple[int, ...] | None = None,
        correction: float = 0.0,
        keepdims: bool = False,
        **kwargs: object) -> Array:
    # [FUNCTION DOC]
    # Array-API standard deviation. Accepts a float correction only if it is integral-
    #   valued (converted to int). axis=() returns zeros_like(x); an int axis becomes a
    #   1-tuple; axis=None reduces over all axes with manual keepdims.
    # Exceptions: NotImplementedError for non-integral float correction.
    # Note, float correction is not supported
    # https://github.com/pytorch/pytorch/issues/61492. We don't try to
    # implement it here for now.

    if isinstance(correction, float):
        _correction = int(correction)
        if correction != _correction:
            raise NotImplementedError("float correction in torch std() is not yet supported")
    else:
        _correction = correction

    # https://github.com/pytorch/pytorch/issues/29137
    if axis == ():
        return torch.zeros_like(x)
    if isinstance(axis, int):
        axis = (axis,)
    if axis is None:
        # torch doesn't support keepdims with axis=None
        # (https://github.com/pytorch/pytorch/issues/71209)
        res = torch.std(x, tuple(range(x.ndim)), correction=_correction, **kwargs)
        res = _axis_none_keepdims(res, x.ndim, keepdims)
        return res
    return torch.std(x, axis, correction=_correction, keepdims=keepdims, **kwargs)

def var(x: Array,
        /,
        *,
        axis: int | tuple[int, ...] | None = None,
        correction: float = 0.0,
        keepdims: bool = False,
        **kwargs: object) -> Array:
    # [FUNCTION DOC]
    # Array-API variance; same as std but passes correction through unchanged (no float
    #   check). axis=() returns zeros_like(x).
    # KNOWN ISSUE: inconsistent with std - non-integral float corrections are forwarded
    #   to torch instead of raising NotImplementedError (the conversion is commented
    #   out).
    # Note, float correction is not supported
    # https://github.com/pytorch/pytorch/issues/61492. We don't try to
    # implement it here for now.

    # if isinstance(correction, float):
    #     correction = int(correction)

    # https://github.com/pytorch/pytorch/issues/29137
    if axis == ():
        return torch.zeros_like(x)
    if isinstance(axis, int):
        axis = (axis,)
    if axis is None:
        # torch doesn't support keepdims with axis=None
        # (https://github.com/pytorch/pytorch/issues/71209)
        res = torch.var(x, tuple(range(x.ndim)), correction=correction, **kwargs)
        res = _axis_none_keepdims(res, x.ndim, keepdims)
        return res
    return torch.var(x, axis, correction=correction, keepdims=keepdims, **kwargs)

# torch.concat doesn't support dim=None
# https://github.com/pytorch/pytorch/issues/70925
def concat(arrays: tuple[Array, ...] | list[Array],
           /,
           *,
           axis: int | None = 0,
           **kwargs: object) -> Array:
    # [FUNCTION DOC]
    # Concatenates arrays along axis; axis=None flattens each input first and
    #   concatenates along 0. Forwards to torch.concat.
    if axis is None:
        arrays = tuple(ar.flatten() for ar in arrays)
        axis = 0
    return torch.concat(arrays, axis, **kwargs)

# torch.squeeze only accepts int dim and doesn't require it
# https://github.com/pytorch/pytorch/issues/70924. Support for tuple dim was
# added at https://github.com/pytorch/pytorch/pull/89017.
def squeeze(x: Array, /, axis: int | tuple[int, ...]) -> Array:
    # [FUNCTION DOC]
    # Removes the given size-1 axes (int or tuple). Checks every listed axis has size 1,
    #   normalises/sorts them, then squeezes one at a time adjusting each index for
    #   previously removed axes.
    # Exceptions: ValueError if any axis has size other than 1; IndexError for invalid
    #   or duplicate axes (out-of-range may surface first as IndexError from the shape
    #   lookup).
    if isinstance(axis, int):
        axis = (axis,)
    for a in axis:
        if x.shape[a] != 1:
            raise ValueError("squeezed dimensions must be equal to 1")
    axes = _normalize_axes(axis, x.ndim)
    # Remove this once pytorch 1.14 is released with the above PR #89017.
    sequence = [a - i for i, a in enumerate(axes)]
    for a in sequence:
        x = torch.squeeze(x, a)
    return x

# torch.broadcast_to uses size instead of shape
def broadcast_to(x: Array, /, shape: tuple[int, ...], **kwargs: object) -> Array:
    # [FUNCTION DOC]
    # Forwards to torch.broadcast_to(x, shape); returns a broadcast view.
    return torch.broadcast_to(x, shape, **kwargs)

# torch.permute uses dims instead of axes
def permute_dims(x: Array, /, axes: tuple[int, ...]) -> Array:
    # [FUNCTION DOC]
    # Permutes axes via torch.permute; returns a view.
    return torch.permute(x, axes)

# The axis parameter doesn't work for flip() and roll()
# https://github.com/pytorch/pytorch/issues/71210. Also torch.flip() doesn't
# accept axis=None
def flip(x: Array, /, *, axis: int | tuple[int, ...] | None = None, **kwargs: object) -> Array:
    # [FUNCTION DOC]
    # Reverses x along axis (None means all axes) using the tensor flip method, which
    #   accepts an int or tuple.
    if axis is None:
        axis = tuple(range(x.ndim))
    # torch.flip doesn't accept dim as an int but the method does
    # https://github.com/pytorch/pytorch/issues/18095
    return x.flip(axis, **kwargs)

def roll(x: Array, /, shift: int | tuple[int, ...], *, axis: int | tuple[int, ...] | None = None, **kwargs: object) -> Array:
    # [FUNCTION DOC]
    # Rolls x by shift along axis via torch.roll (positional dims argument).
    return torch.roll(x, shift, axis, **kwargs)

def nonzero(x: Array, /, **kwargs: object) -> tuple[Array, ...]:
    # [FUNCTION DOC]
    # Returns a tuple of index tensors of nonzero elements, one per dimension
    #   (torch.nonzero with as_tuple).
    # Exceptions: ValueError for 0-D input.
    if x.ndim == 0:
        raise ValueError("nonzero() does not support zero-dimensional arrays")
    return torch.nonzero(x, as_tuple=True, **kwargs)


# torch uses `dim` instead of `axis`
def diff(
    x: Array,
    /,
    *,
    axis: int = -1,
    n: int = 1,
    prepend: Array | None = None,
    append: Array | None = None,
) -> Array:
    # [FUNCTION DOC]
    # n-th discrete difference along axis with optional prepend/append arrays; forwards
    #   to torch.diff with dim=axis.
    return torch.diff(x, dim=axis, n=n, prepend=prepend, append=append)


# torch uses `dim` instead of `axis`, does not have keepdims
def count_nonzero(
    x: Array,
    /,
    *,
    axis: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
) -> Array:
    # [FUNCTION DOC]
    # Counts nonzero elements over axis (None, int or tuple) using torch.count_nonzero.
    #   When keepdims: int axis is reinserted with unsqueeze, tuple axes are restored by
    #   reshaping to x's shape with 1 at the reduced positions, axis=None prepends ndim
    #   size-1 axes.
    # Outputs: int64 tensor.
    result = torch.count_nonzero(x, dim=axis)
    if keepdims:
        if isinstance(axis, int):
            return result.unsqueeze(axis)
        elif isinstance(axis, tuple):
            n_axis = [x.ndim + ax if ax < 0 else ax for ax in axis]
            sh = [1 if i in n_axis else x.shape[i] for i in range(x.ndim)]
            return torch.reshape(result, sh)
        return _axis_none_keepdims(result, x.ndim, keepdims)
    else:
        return result


# "repeat" is torch.repeat_interleave;  also the dim argument
def repeat(x: Array, repeats: int | Array, /, *, axis: int | None = None) -> Array:
    # [FUNCTION DOC]
    # Repeats elements of x (flattened when axis is None) via torch.repeat_interleave;
    #   int8/int16 repeats tensors are first cast to int32 because torch rejects them.
    if isinstance(repeats, torch.Tensor) and repeats.dtype in (torch.int8, torch.int16):
        # torch rejects short integers for the `repeat` argument:
        # https://github.com/pytorch/pytorch/issues/151311
        repeats = repeats.to(torch.int32)
    return torch.repeat_interleave(x, repeats, axis)


def where(condition: Array, x1: Array | complex, x2: Array | complex, /) -> Array:
    # [FUNCTION DOC]
    # Elementwise selection: x1 where condition is true, else x2, after scalar-aware
    #   promotion of x1/x2.
    x1, x2 = _fix_promotion(x1, x2)
    return torch.where(condition, x1, x2)


# torch.reshape doesn't have the copy keyword
def reshape(x: Array,
            /,
            shape: tuple[int, ...],
            *,
            copy: bool | None = None,
            **kwargs: object) -> Array:
    # [FUNCTION DOC]
    # Reshapes via torch.reshape. Exceptions: NotImplementedError if copy is not None.
    if copy is not None:
        raise NotImplementedError("torch.reshape doesn't yet support the copy keyword")
    return torch.reshape(x, shape, **kwargs)

# torch.arange doesn't support returning empty arrays
# (https://github.com/pytorch/pytorch/issues/70915), and doesn't support some
# keyword argument combinations
# (https://github.com/pytorch/pytorch/issues/70914)
def arange(start: float,
           /,
           stop: float | None = None,
           step: float = 1,
           *,
           dtype: DType | None = None,
           device: Device | None = None,
           **kwargs: object) -> Array:
    # [FUNCTION DOC]
    # Array-API arange. A single argument is treated as stop with start 0. If the range
    #   is empty (step and stop-start of opposite sign or zero) returns an empty tensor
    #   of dtype (inferred int64 when start/stop/step are all ints, else float32).
    #   Otherwise calls torch.arange with dtype; if torch rejects the dtype combination
    #   (RuntimeError or NotImplementedError) it retries without dtype and casts.
    # Outputs: 1-D tensor.
    # KNOWN ISSUE: step 0 is not caught by the empty-range test and is passed to torch,
    #   which raises.
    if stop is None:
        start, stop = 0, start
    if step > 0 and stop <= start or step < 0 and stop >= start:
        if dtype is None:
            if _builtin_all(isinstance(i, int) for i in [start, stop, step]):
                dtype = torch.int64
            else:
                dtype = torch.float32
        return torch.empty(0, device=device, **kwargs).to(dtype)
    try:
        return torch.arange(start, stop, step, dtype=dtype, device=device, **kwargs)
    # torch 2.7 raises RuntimeError, 2.9 emits NotImplementedError
    except (NotImplementedError, RuntimeError):
        return torch.arange(start, stop, step, device=device, **kwargs).to(dtype)

# torch.eye does not accept None as a default for the second argument and
# doesn't support off-diagonals (https://github.com/pytorch/pytorch/issues/70910)
def eye(n_rows: int,
        n_cols: int | None = None,
        /,
        *,
        k: int = 0,
        dtype: DType | None = None,
        device: Device | None = None,
        **kwargs: object) -> Array:
    # [FUNCTION DOC]
    # Identity-like matrix of shape (n_rows, n_cols or n_rows) with ones on diagonal k:
    #   creates zeros, then fills diagonal k in place when |k| <= n_rows + n_cols (other
    #   k leave all zeros).
    if n_cols is None:
        n_cols = n_rows
    z = torch.zeros(n_rows, n_cols, dtype=dtype, device=device, **kwargs)
    if abs(k) <= n_rows + n_cols:
        z.diagonal(k).fill_(1)
    return z

# torch.linspace doesn't have the endpoint parameter
def linspace(start: float,
             stop: float,
             /,
             num: int,
             *,
             dtype: DType | None = None,
             device: Device | None = None,
             endpoint: bool = True,
             **kwargs: object) -> Array:
    # [FUNCTION DOC]
    # Evenly spaced values; when endpoint is False generates num+1 points and drops the
    #   last, else forwards to torch.linspace.
    if not endpoint:
        return torch.linspace(start, stop, num+1, dtype=dtype, device=device, **kwargs)[:-1]
    return torch.linspace(start, stop, num, dtype=dtype, device=device, **kwargs)

# torch.full does not accept an int size
# https://github.com/pytorch/pytorch/issues/70906
def full(shape: int | tuple[int, ...],
         fill_value: complex,
         *,
         dtype: DType | None = None,
         device: Device | None = None,
         **kwargs: object) -> Array:
    # [FUNCTION DOC]
    # torch.full accepting an int shape (converted to a 1-tuple).
    if isinstance(shape, int):
        shape = (shape,)

    return torch.full(shape, fill_value, dtype=dtype, device=device, **kwargs)

# ones, zeros, and empty do not accept shape as a keyword argument
def ones(shape: int | tuple[int, ...],
         *,
         dtype: DType | None = None,
         device: Device | None = None,
         **kwargs: object) -> Array:
    # [FUNCTION DOC]
    # torch.ones with shape/dtype/device keyword support.
    return torch.ones(shape, dtype=dtype, device=device, **kwargs)

def zeros(shape: int | tuple[int, ...],
         *,
         dtype: DType | None = None,
         device: Device | None = None,
         **kwargs: object) -> Array:
    # [FUNCTION DOC]
    # torch.zeros with shape/dtype/device keyword support.
    return torch.zeros(shape, dtype=dtype, device=device, **kwargs)

def empty(shape: int | tuple[int, ...],
         *,
         dtype: DType | None = None,
         device: Device | None = None,
         **kwargs: object) -> Array:
    # [FUNCTION DOC]
    # torch.empty with shape/dtype/device keyword support.
    return torch.empty(shape, dtype=dtype, device=device, **kwargs)

# tril and triu do not call the keyword argument k

def tril(x: Array, /, *, k: int = 0) -> Array:
    # [FUNCTION DOC]
    # Lower triangle of the last two axes, below/including diagonal k (torch.tril
    #   positional diagonal).
    return torch.tril(x, k)

def triu(x: Array, /, *, k: int = 0) -> Array:
    # [FUNCTION DOC]
    # Upper triangle of the last two axes, above/including diagonal k (torch.triu).
    return torch.triu(x, k)

# Functions that aren't in torch https://github.com/pytorch/pytorch/issues/58742
def expand_dims(x: Array, /, axis: int | tuple[int, ...]) -> Array:
    # [FUNCTION DOC]
    # Inserts size-1 axes. An int axis uses torch.unsqueeze. A tuple is normalised
    #   against the output ndim (x.ndim + len(axis)) and the output shape is built by
    #   putting 1 at listed positions and x's dims in order elsewhere, then reshaped.
    # Exceptions: ValueError for duplicate or out-of-range axes in the tuple form; torch
    #   errors for invalid int axes.
    if isinstance(axis, int):
        return torch.unsqueeze(x, axis)
    else:
        # follow https://github.com/numpy/numpy/blob/maintenance/2.4.x/numpy/lib/_shape_base_impl.py#L596-L602
        y_ndim = x.ndim + len(axis)

        # normalize
        n_axis = tuple(ax + y_ndim if ax < 0 else ax for ax in axis)
        if (len(n_axis) != len(set(n_axis)) or
            _builtin_any(ax < 0 or ax >= y_ndim for ax in n_axis)
        ):
            raise ValueError(f"{axis=} not allowed for {x.shape = }")

        shape_it = iter(x.shape)
        shape = [1 if ax in n_axis else next(shape_it) for ax in range(y_ndim)]

        return torch.reshape(x, shape)

def astype(
    x: Array,
    dtype: DType,
    /,
    *,
    copy: bool = True,
    device: Device | None = None,
) -> Array:
    # [FUNCTION DOC]
    # Casts x to dtype (and to device when given) via Tensor.to with copy forwarded.
    if device is not None:
        return x.to(device, dtype=dtype, copy=copy)
    return x.to(dtype=dtype, copy=copy)


def broadcast_arrays(*arrays: Array) -> tuple[Array, ...]:
    # [FUNCTION DOC]
    # Broadcasts all arrays to their common shape (torch.broadcast_shapes) and returns a
    #   tuple of broadcast views.
    # Exceptions: RuntimeError from torch for incompatible shapes.
    shape = torch.broadcast_shapes(*[a.shape for a in arrays])
    return tuple(torch.broadcast_to(a, shape) for a in arrays)

# Note that these named tuples aren't actually part of the standard namespace,
# but I don't see any issue with exporting the names here regardless.
from ..common._aliases import (UniqueAllResult, UniqueCountsResult,
                               UniqueInverseResult)

# https://github.com/pytorch/pytorch/issues/70920
def unique_all(x: Array) -> UniqueAllResult:
    # [FUNCTION DOC]
    # Not supported: always raises NotImplementedError because torch.unique cannot
    #   return first-occurrence indices. Trailing commented code is unreachable.
    # torch.unique doesn't support returning indices.
    # https://github.com/pytorch/pytorch/issues/36748. The workaround
    # suggested in that issue doesn't actually function correctly (it relies
    # on non-deterministic behavior of scatter()).
    raise NotImplementedError("unique_all() not yet implemented for pytorch (see https://github.com/pytorch/pytorch/issues/36748)")

    # values, inverse_indices, counts = torch.unique(x, return_counts=True, return_inverse=True)
    # # torch.unique incorrectly gives a 0 count for nan values.
    # # https://github.com/pytorch/pytorch/issues/94106
    # counts[torch.isnan(values)] = 1
    # return UniqueAllResult(values, indices, inverse_indices, counts)

def unique_counts(x: Array) -> UniqueCountsResult:
    # [FUNCTION DOC]
    # Unique values and their counts via torch.unique (sorted); counts of NaN values are
    #   forced to 1 to fix a torch bug. Returns UniqueCountsResult.
    # Side effects: mutates the freshly created counts tensor only.
    values, counts = torch.unique(x, return_counts=True)

    # torch.unique incorrectly gives a 0 count for nan values.
    # https://github.com/pytorch/pytorch/issues/94106
    counts[torch.isnan(values)] = 1
    return UniqueCountsResult(values, counts)

def unique_inverse(x: Array) -> UniqueInverseResult:
    # [FUNCTION DOC]
    # Unique values and inverse indices via torch.unique; returns UniqueInverseResult.
    values, inverse = torch.unique(x, return_inverse=True)
    return UniqueInverseResult(values, inverse)

def unique_values(x: Array) -> Array:
    # [FUNCTION DOC]
    # Sorted unique values via torch.unique.
    return torch.unique(x)

def matmul(x1: Array, x2: Array, /, **kwargs: object) -> Array:
    # [FUNCTION DOC]
    # Matrix product after full (not scalar-only) promotion of both operands; forwards
    #   to torch.matmul.
    # torch.matmul doesn't type promote (but differently from _fix_promotion)
    x1, x2 = _fix_promotion(x1, x2, only_scalar=False)
    return torch.matmul(x1, x2, **kwargs)

matrix_transpose = get_xp(torch)(_aliases.matrix_transpose)
_vecdot = get_xp(torch)(_aliases.vecdot)

def vecdot(x1: Array, x2: Array, /, *, axis: int = -1) -> Array:
    # [FUNCTION DOC]
    # Vector dot along axis after full promotion, delegating to the generic common
    #   vecdot bound to torch.
    x1, x2 = _fix_promotion(x1, x2, only_scalar=False)
    return _vecdot(x1, x2, axis=axis)

# torch.tensordot uses dims instead of axes
def tensordot(
    x1: Array,
    x2: Array,
    /,
    *, 
    axes: int | tuple[Sequence[int], Sequence[int]] = 2, 
    **kwargs: object,
) -> Array:
    # [FUNCTION DOC]
    # Tensor contraction after full promotion; axes (int or pair of sequences) is passed
    #   as torch's dims.
    # Note: torch.tensordot fails with integer dtypes when there is only 1
    # element in the axis (https://github.com/pytorch/pytorch/issues/84530).
    x1, x2 = _fix_promotion(x1, x2, only_scalar=False)
    return torch.tensordot(x1, x2, dims=axes, **kwargs)


def isdtype(
    dtype: DType, 
    kind: DType | str | tuple[DType | str, ...],
    *,
    _tuple: bool = True, # Disallow nested tuples
) -> bool:
    # [FUNCTION DOC]
    # Tests whether dtype belongs to kind. A tuple kind returns true if any element
    #   matches (nested tuples are compared by equality, not expanded). String kinds:
    #   'bool', 'signed integer' and 'unsigned integer' (via _int_dtypes and is_signed),
    #   'integral', 'real floating' (is_floating_point), 'complex floating', 'numeric'.
    #   Any other kind is compared with equality.
    # Exceptions: ValueError for unknown kind strings.
    """
    Returns a boolean indicating whether a provided dtype is of a specified data type ``kind``.

    Note that outside of this function, this compat library does not yet fully
    support complex numbers.

    See
    https://data-apis.org/array-api/latest/API_specification/generated/array_api.isdtype.html
    for more details
    """
    if isinstance(kind, tuple) and _tuple:
        return _builtin_any(isdtype(dtype, k, _tuple=False) for k in kind)
    elif isinstance(kind, str):
        if kind == 'bool':
            return dtype == torch.bool
        elif kind == 'signed integer':
            return dtype in _int_dtypes and dtype.is_signed
        elif kind == 'unsigned integer':
            return dtype in _int_dtypes and not dtype.is_signed
        elif kind == 'integral':
            return dtype in _int_dtypes
        elif kind == 'real floating':
            return dtype.is_floating_point
        elif kind == 'complex floating':
            return dtype.is_complex
        elif kind == 'numeric':
            return isdtype(dtype, ('integral', 'real floating', 'complex floating'))
        else:
            raise ValueError(f"Unrecognized data type kind: {kind!r}")
    else:
        return dtype == kind

def take(x: Array, indices: Array, /, *, axis: int | None = None, **kwargs: object) -> Array:
    # [FUNCTION DOC]
    # Selects indices along axis with torch.index_select, wrapping negative indices by
    #   adding the axis length. axis None is allowed only for 1-D x (uses 0).
    # Exceptions: ValueError if axis is None and x is not 1-D.
    if axis is None:
        if x.ndim != 1:
            raise ValueError("axis must be specified when ndim > 1")
        axis = 0
    # torch does not support negative indices,
    # see https://github.com/pytorch/pytorch/issues/146211
    return torch.index_select(
        x,
        axis,
        torch.where(indices < 0, indices + x.shape[axis], indices),
        **kwargs
    )


def take_along_axis(x: Array, indices: Array, /, *, axis: int = -1) -> Array:
    # [FUNCTION DOC]
    # Gathers values along axis with torch.take_along_dim after wrapping negative
    #   indices by the axis length.
    # torch does not support negative indices,
    # see https://github.com/pytorch/pytorch/issues/146211
    return torch.take_along_dim(
        x,
        torch.where(indices < 0, indices + x.shape[axis], indices),
        dim=axis
    )


def clip(
    x: Array,
    /,
    min: int | float | Array | None = None,
    max: int | float | Array | None = None,
    **kwargs
) -> Array:
    # [FUNCTION DOC]
    # Array-API clip. For integer x, int bounds outside the dtype range are dropped. If
    #   both bounds are None returns a clone. If both bounds are scalars: any NaN bound
    #   gives a NaN-filled tensor like x, else torch.clamp. Otherwise scalar bounds are
    #   converted to tensors of x's dtype/device (torch lacks mixed tensor/scalar
    #   overloads) and clamped.
    # Outputs: new tensor.
    # KNOWN ISSUE: converting a float scalar bound to an integer x dtype truncates it
    #   silently; bool and complex scalars are not recognised as scalars by _isscalar.
    def _isscalar(a: object):
        # [FUNCTION DOC]
        # HELPER: True if a is an int, a float or None.
        return isinstance(a, int | float) or a is None

    # cf clip in common/_aliases.py
    if not x.is_floating_point():
        if type(min) is int and min <= torch.iinfo(x.dtype).min:
            min = None
        if type(max) is int and max >= torch.iinfo(x.dtype).max:
            max = None

    if min is None and max is None:
        return torch.clone(x)

    min_is_scalar = _isscalar(min)
    max_is_scalar = _isscalar(max)

    if min_is_scalar and max_is_scalar:
        if (min is not None and math.isnan(min)) or (max is not None and math.isnan(max)):
            # edge case: torch.clamp(torch.zeros(1), float('nan')) -> tensor(0.)
            # https://github.com/pytorch/pytorch/issues/172067
            return torch.full_like(x, fill_value=torch.nan)
        return torch.clamp(x, min, max, **kwargs)

    # pytorch has (tensor, tensor, tensor) and (tensor, scalar, scalar) signatures,
    # but does not accept (tensor, scalar, tensor)
    a_min = min
    if min is not None and min_is_scalar:
        a_min = torch.as_tensor(min, dtype=x.dtype, device=x.device)

    a_max = max
    if max is not None and max_is_scalar:
        a_max = torch.as_tensor(max, dtype=x.dtype, device=x.device)

    return torch.clamp(x, a_min, a_max, **kwargs)


def sign(x: Array, /) -> Array:
    # [FUNCTION DOC]
    # Array-API sign. Complex: x divided by its magnitude, with exact zeros set to 0.
    #   Real: torch.sign, then NaN positions restored to NaN for floating dtypes.
    # Outputs: new tensor.
    # torch sign() does not support complex numbers and does not propagate
    # nans. See https://github.com/data-apis/array-api-compat/issues/136
    if x.dtype.is_complex:
        out = x/torch.abs(x)
        # sign(0) = 0 but the above formula would give nan
        out[x == 0+0j] = 0+0j
        return out
    else:
        out = torch.sign(x)
        if x.dtype.is_floating_point:
            out[torch.isnan(x)] = torch.nan
        return out


def round(x: Array, /, **kwargs) -> Array:
    # [FUNCTION DOC]
    # Rounds to nearest even. For complex input rounds real and imaginary parts
    #   separately and recombines; an 'out' kwarg is honoured by copying into it
    #   (mutating it) and returning it. Real inputs forward to torch.round.
    # torch.round fails for complex inputs
    # https://github.com/pytorch/pytorch/issues/58743#issuecomment-2727603845
    if x.dtype.is_complex:
        out = kwargs.pop('out', None)
        res_r = torch.round(x.real, **kwargs)
        res_i = torch.round(x.imag, **kwargs)
        res = res_r + 1j*res_i
        if out is not None:
            out.copy_(res)
            return out
        return res
    else:
        return torch.round(x, **kwargs)


def meshgrid(*arrays: Array, indexing: Literal['xy', 'ij'] = 'xy') -> tuple[Array, ...]:
    # [FUNCTION DOC]
    # Coordinate matrices from 1-D arrays, always passing indexing explicitly to avoid
    #   torch warnings; returns an empty tuple for no inputs.
    # Exceptions: ValueError for indexing other than 'xy' or 'ij'.
    # torch <= 2.9 emits a UserWarning: "torch.meshgrid: in an upcoming release, it
    # will be required to pass the indexing argument."
    # Thus always pass it explicitly.
    if indexing not in ("xy", "ij"):
        raise ValueError(f'torch.meshgrid: indexing must be one of "xy" or "ij", but received: {indexing}')
    return torch.meshgrid(*arrays, indexing=indexing) if arrays else ()


__all__ = ['asarray', 'result_type', 'can_cast',
           'permute_dims', 'bitwise_invert', 'newaxis', 'conj', 'add',
           'atan2', 'bitwise_and', 'bitwise_left_shift', 'bitwise_or',
           'bitwise_right_shift', 'bitwise_xor', 'copysign', 'count_nonzero',
           'diff', 'divide', 'round',
           'equal', 'floor_divide', 'greater', 'greater_equal', 'hypot',
           'less', 'less_equal', 'logaddexp', 'maximum', 'minimum',
           'multiply', 'not_equal', 'pow', 'remainder', 'subtract', 'max',
           'min', 'clip', 'unstack', 'cumulative_sum', 'cumulative_prod', 'sort',
           'argsort', 'prod', 'sum', 'any', 'all', 'mean', 'std', 'var', 'concat',
           'squeeze', 'broadcast_to', 'flip', 'roll', 'nonzero', 'where', 'reshape',
           'arange', 'eye', 'linspace', 'full', 'ones', 'zeros', 'empty',
           'tril', 'triu', 'expand_dims', 'astype', 'broadcast_arrays',
           'UniqueAllResult', 'UniqueCountsResult', 'UniqueInverseResult',
           'unique_all', 'unique_counts', 'unique_inverse', 'unique_values',
           'matmul', 'matrix_transpose', 'vecdot', 'tensordot', 'isdtype',
           'take', 'take_along_axis', 'sign', 'finfo', 'iinfo', 'repeat', 'meshgrid']
