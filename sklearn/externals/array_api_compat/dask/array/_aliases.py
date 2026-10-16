# [MODULE DOC]
# Dask-specific array API aliases: NumPy dtypes and can_cast/result_type re-exported
# for dask, renamed elementwise functions (acos, concat, pow, ...), generic
# common._aliases implementations bound to dask.array (or to numpy for dtype-level
# helpers such as isdtype, finfo, matmul), and Dask-specific wrappers for astype,
# arange, asarray, clip, sort, argsort and count_nonzero. Devices are not really
# supported: functions validate the device with _helpers._check_device and ignore it.
# Imports dask.array and numpy at import time.
# pyright: reportPrivateUsage=false
# pyright: reportUnknownArgumentType=false
# pyright: reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false

from __future__ import annotations

from builtins import bool as py_bool
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from typing_extensions import TypeIs

import dask.array as da
import numpy as np
from numpy import bool_ as bool
from numpy import (
    can_cast,
    complex64,
    complex128,
    float32,
    float64,
    int8,
    int16,
    int32,
    int64,
    result_type,
    uint8,
    uint16,
    uint32,
    uint64,
)

from ..._internal import get_xp
from ...common import _aliases, _helpers, array_namespace
from ...common._typing import (
    Array,
    Device,
    DType,
    NestedSequence,
    SupportsBufferProtocol,
)

isdtype = get_xp(np)(_aliases.isdtype)
unstack = get_xp(da)(_aliases.unstack)


# da.astype doesn't respect copy=True
def astype(
    x: Array,
    dtype: DType,
    /,
    *,
    copy: py_bool = True,
    device: Device | None = None,
) -> Array:
    # [FUNCTION DOC]
    # astype honouring copy semantics that dask ignores. Validates device, returns x itself
    # when copy is False and the dtype already matches, otherwise casts lazily and adds an
    # explicit .copy() when copy is True.
    # Inputs: x - dask array; dtype; copy - default True; device - None, "cpu" or
    # _DASK_DEVICE.
    # Outputs: dask array.
    # Exceptions: ValueError for unsupported devices.
    """
    Array API compatibility wrapper for astype().

    See the corresponding documentation in the array library and/or the array API
    specification for more details.
    """
    # TODO: respect device keyword?
    _helpers._check_device(da, device)

    if not copy and dtype == x.dtype:
        return x
    x = x.astype(dtype)
    return x.copy() if copy else x


# Common aliases


# This arange func is modified from the common one to
# not pass stop/step as keyword arguments, which will cause
# an error with dask
def arange(
    start: float,
    /,
    stop: float | None = None,
    step: float = 1,
    *,
    dtype: DType | None = None,
    device: Device | None = None,
    **kwargs: object,
) -> Array:
    # [FUNCTION DOC]
    # arange that passes start/stop/step positionally (dask rejects stop/step keywords).
    # When stop is None the single argument is treated as stop and start becomes 0.
    # Inputs: start; stop - optional; step - default 1; dtype - optional; device -
    # validated and ignored; kwargs forwarded to dask.array.arange.
    # Outputs: 1-D dask array.
    # Exceptions: ValueError for unsupported devices.
    """
    Array API compatibility wrapper for arange().

    See the corresponding documentation in the array library and/or the array API
    specification for more details.
    """
    # TODO: respect device keyword?
    _helpers._check_device(da, device)

    args: list[Any] = [start]
    if stop is not None:
        args.append(stop)
    else:
        # stop is None, so start is actually stop
        # prepend the default value for start which is 0
        args.insert(0, 0)
    args.append(step)

    return da.arange(*args, dtype=dtype, **kwargs)


eye = get_xp(da)(_aliases.eye)
linspace = get_xp(da)(_aliases.linspace)
UniqueAllResult = get_xp(da)(_aliases.UniqueAllResult)
UniqueCountsResult = get_xp(da)(_aliases.UniqueCountsResult)
UniqueInverseResult = get_xp(da)(_aliases.UniqueInverseResult)
unique_all = get_xp(da)(_aliases.unique_all)
unique_counts = get_xp(da)(_aliases.unique_counts)
unique_inverse = get_xp(da)(_aliases.unique_inverse)
unique_values = get_xp(da)(_aliases.unique_values)
permute_dims = get_xp(da)(_aliases.permute_dims)
std = get_xp(da)(_aliases.std)
var = get_xp(da)(_aliases.var)
cumulative_sum = get_xp(da)(_aliases.cumulative_sum)
cumulative_prod = get_xp(da)(_aliases.cumulative_prod)
empty = get_xp(da)(_aliases.empty)
empty_like = get_xp(da)(_aliases.empty_like)
full = get_xp(da)(_aliases.full)
full_like = get_xp(da)(_aliases.full_like)
ones = get_xp(da)(_aliases.ones)
ones_like = get_xp(da)(_aliases.ones_like)
zeros = get_xp(da)(_aliases.zeros)
zeros_like = get_xp(da)(_aliases.zeros_like)
reshape = get_xp(da)(_aliases.reshape)
matrix_transpose = get_xp(da)(_aliases.matrix_transpose)
vecdot = get_xp(da)(_aliases.vecdot)
nonzero = get_xp(da)(_aliases.nonzero)
matmul = get_xp(np)(_aliases.matmul)
tensordot = get_xp(np)(_aliases.tensordot)
sign = get_xp(np)(_aliases.sign)
finfo = get_xp(np)(_aliases.finfo)
iinfo = get_xp(np)(_aliases.iinfo)


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
    # asarray with copy semantics. For dask input: casts if dtype differs (ValueError if
    # copy is False), then copies only if copy is truthy. For other input: copy False
    # raises ValueError; otherwise materializes with numpy.array(copy=True) and wraps it
    # with dask.array.from_array (default chunking).
    # Inputs: obj - array-like; dtype - optional; device - validated; copy - None/bool;
    # kwargs - accepted but ignored.
    # Outputs: dask array.
    # Exceptions: ValueError for unsupported devices or impossible no-copy requests.
    # KNOWN ISSUE: extra kwargs are silently ignored rather than forwarded.
    """
    Array API compatibility wrapper for asarray().

    See the corresponding documentation in the array library and/or the array API
    specification for more details.
    """
    # TODO: respect device keyword?
    _helpers._check_device(da, device)

    if isinstance(obj, da.Array):
        if dtype is not None and dtype != obj.dtype:
            if copy is False:
                raise ValueError("Unable to avoid copy when changing dtype")
            obj = obj.astype(dtype)
        return obj.copy() if copy else obj  # pyright: ignore[reportAttributeAccessIssue]

    if copy is False:
        raise ValueError(
            "Unable to avoid copy when converting a non-dask object to dask"
        )

    # copy=None to be uniform across dask < 2024.12 and >= 2024.12
    # see https://github.com/dask/dask/pull/11524/
    obj = np.array(obj, dtype=dtype, copy=True)
    return da.from_array(obj)


# Element wise aliases
from dask.array import arccos as acos
from dask.array import arccosh as acosh
from dask.array import arcsin as asin
from dask.array import arcsinh as asinh
from dask.array import arctan as atan
from dask.array import arctan2 as atan2
from dask.array import arctanh as atanh

# Other
from dask.array import concatenate as concat
from dask.array import invert as bitwise_invert
from dask.array import left_shift as bitwise_left_shift
from dask.array import power as pow
from dask.array import right_shift as bitwise_right_shift


# dask.array.clip does not work unless all three arguments are provided.
# Furthermore, the masking workaround in common._aliases.clip cannot work with
# dask (meaning uint64 promoting to float64 is going to just be unfixed for
# now).
def clip(
    x: Array,
    /,
    min: float | Array | None = None,
    max: float | Array | None = None,
) -> Array:
    # [FUNCTION DOC]
    # clip with optional bounds. Computes the broadcast shape of x and the bounds (scalars
    # count as shape ()), broadcasts any non-None bound to it as a dask array, returns
    # positive(x) when both bounds are None, otherwise applies maximum and/or minimum and
    # casts back to x.dtype with the module's astype (which copies).
    # Inputs: x - dask array; min, max - None, Python scalar or array.
    # Outputs: dask array of x's dtype with the broadcast shape.
    # Exceptions: ValueError from broadcast_shapes for incompatible shapes.
    # KNOWN ISSUE: unknown (NaN) dask shapes are not handled, and NumPy scalar bounds are
    # not recognised as scalars by _isscalar (they work only because they have .shape).
    """
    Array API compatibility wrapper for clip().

    See the corresponding documentation in the array library and/or the array API
    specification for more details.
    """

    def _isscalar(a: float | Array | None, /) -> TypeIs[float | None]:
        # [FUNCTION DOC]
        # HELPER: Returns True when a is None or a Python int/float (bool included).
        return a is None or isinstance(a, (int, float))

    min_shape = () if _isscalar(min) else min.shape
    max_shape = () if _isscalar(max) else max.shape

    # TODO: This won't handle dask unknown shapes
    result_shape = np.broadcast_shapes(x.shape, min_shape, max_shape)

    if min is not None:
        min = da.broadcast_to(da.asarray(min), result_shape)
    if max is not None:
        max = da.broadcast_to(da.asarray(max), result_shape)

    if min is None and max is None:
        return da.positive(x)

    if min is None:
        return astype(da.minimum(x, max), x.dtype)
    if max is None:
        return astype(da.maximum(x, min), x.dtype)

    return astype(da.minimum(da.maximum(x, min), max), x.dtype)


def _ensure_single_chunk(x: Array, axis: int) -> tuple[Array, Callable[[Array], Array]]:
    # [FUNCTION DOC]
    # HELPER: Ensures x has a single chunk along axis (negative axis normalised). If it
    # already has fewer than 2 blocks there, returns x and an identity function; otherwise
    # rechunks to one chunk on axis and "auto" elsewhere, and returns a restore function
    # that applies a default rechunk() to the eventual output.
    # Inputs: x - dask array; axis - int.
    # Outputs: (rechunked array, restore callable).
    """
    Make sure that Array is not broken into multiple chunks along axis.

    Returns
    -------
    x : Array
        The input Array with a single chunk along axis.
    restore : Callable[Array, Array]
        function to apply to the output to rechunk it back into reasonable chunks
    """
    if axis < 0:
        axis += x.ndim
    if x.numblocks[axis] < 2:
        return x, lambda x: x

    # Break chunks on other axes in an attempt to keep chunk size low
    x = x.rechunk({i: -1 if i == axis else "auto" for i in range(x.ndim)})

    # Rather than reconstructing the original chunks, which can be a
    # very expensive affair, just break down oversized chunks without
    # incurring in any transfers over the network.
    # This has the downside of a risk of overchunking if the array is
    # then used in operations against other arrays that match the
    # original chunking pattern.
    return x, lambda x: x.rechunk()


def sort(
    x: Array,
    /,
    *,
    axis: int = -1,
    descending: py_bool = False,
    stable: py_bool = True,
) -> Array:
    # [FUNCTION DOC]
    # Sort along axis by rechunking that axis into a single chunk and mapping the namespace
    # of x._meta's sort (with descending and stable) over each block, then rechunking back.
    # Inputs: x - dask array; axis - default -1; descending, stable - bools.
    # Outputs: sorted dask array, same dtype and meta.
    # Side effects: none on x, but may be memory-intensive at compute time.
    """
    Array API compatibility layer around the lack of sort() in Dask.

    Warnings
    --------
    This function temporarily rechunks the array along `axis` to a single chunk.
    This can be extremely inefficient and can lead to out-of-memory errors.

    See the corresponding documentation in the array library and/or the array API
    specification for more details.
    """
    x, restore = _ensure_single_chunk(x, axis)

    meta_xp = array_namespace(x._meta)
    x = da.map_blocks(
        meta_xp.sort,
        x,
        axis=axis,
        meta=x._meta,
        dtype=x.dtype,
        descending=descending,
        stable=stable,
    )

    return restore(x)


def argsort(
    x: Array,
    /,
    *,
    axis: int = -1,
    descending: py_bool = False,
    stable: py_bool = True,
) -> Array:
    # [FUNCTION DOC]
    # Argsort along axis, implemented like sort but mapping the meta namespace's argsort;
    # the output dtype is obtained by running argsort on x._meta.
    # Inputs: as sort. Outputs: dask array of indices.
    """
    Array API compatibility layer around the lack of argsort() in Dask.

    See the corresponding documentation in the array library and/or the array API
    specification for more details.

    Warnings
    --------
    This function temporarily rechunks the array along `axis` into a single chunk.
    This can be extremely inefficient and can lead to out-of-memory errors.
    """
    x, restore = _ensure_single_chunk(x, axis)

    meta_xp = array_namespace(x._meta)
    dtype = meta_xp.argsort(x._meta).dtype
    meta = meta_xp.astype(x._meta, dtype)
    x = da.map_blocks(
        meta_xp.argsort,
        x,
        axis=axis,
        meta=meta,
        dtype=dtype,
        descending=descending,
        stable=stable,
    )

    return restore(x)


# dask.array.count_nonzero does not have keepdims
def count_nonzero(
    x: Array,
    axis: int | None = None,
    keepdims: py_bool = False,
) -> Array:
    # [FUNCTION DOC]
    # count_nonzero adding keepdims: reshapes to all-ones shape when axis is None, else
    # expands the reduced axis.
    # Inputs: x - dask array; axis - None or int; keepdims - default False.
    # Outputs: dask integer array.
    result = da.count_nonzero(x, axis)
    if keepdims:
        if axis is None:
            return da.reshape(result, [1] * x.ndim)
        return da.expand_dims(result, axis)
    return result


__all__ = [
    "count_nonzero",
    "bool",
    "int8", "int16", "int32", "int64",
    "uint8", "uint16", "uint32", "uint64",
    "float32", "float64",
    "complex64", "complex128",
    "asarray", "astype", "can_cast", "result_type",
    "pow",
    "concat",
    "acos", "acosh", "asin", "asinh", "atan", "atan2", "atanh",
    "bitwise_left_shift", "bitwise_right_shift", "bitwise_invert",
]  # fmt: skip
__all__ += _aliases.__all__

def __dir__() -> list[str]:
    # [FUNCTION DOC]
    # HELPER: Module __dir__ hook returning __all__.
    return __all__
