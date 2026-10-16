# [MODULE DOC]
# Backend-generic implementations of Array API functions whose NumPy-like counterparts
#   differ in name, signature, keyword names or return types. Each function takes the
#   wrapped library namespace as xp; per-backend subpackages bind xp with
#   _internal.get_xp. Also defines the named-tuple result types of the unique_* family
#   and a module __dir__ limited to __all__.
"""
These are functions that are just aliases of existing functions in NumPy.
"""

from __future__ import annotations

import inspect
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, NamedTuple, cast

from ._helpers import _check_device, array_namespace
from ._helpers import device as _get_device
from ._helpers import is_cupy_namespace
from ._typing import Array, Device, DType, Namespace

if TYPE_CHECKING:
    # TODO: import from typing (requires Python >=3.13)
    from typing_extensions import TypeIs

# These functions are modified from the NumPy versions.

# Creation functions add the device keyword (which does nothing for NumPy and Dask)


def arange(
    start: float,
    /,
    stop: float | None = None,
    step: float = 1,
    *,
    xp: Namespace,
    dtype: DType | None = None,
    device: Device | None = None,
    **kwargs: object,
) -> Array:
    # [FUNCTION DOC]
    # Array API creation function: validates the device keyword with _check_device (only
    #   CPU / None accepted for NumPy-like backends) and then calls the same-named
    #   function of xp with stop, step forwarded; start is positional-only, dtype and
    #   any extra keyword arguments; device itself is not forwarded.
    # Outputs: the new array. Exceptions: ValueError from _check_device for an
    #   unsupported device.
    _check_device(xp, device)
    return xp.arange(start, stop=stop, step=step, dtype=dtype, **kwargs)


def empty(
    shape: int | tuple[int, ...],
    xp: Namespace,
    *,
    dtype: DType | None = None,
    device: Device | None = None,
    **kwargs: object,
) -> Array:
    # [FUNCTION DOC]
    # Array API creation function: validates the device keyword with _check_device (only
    #   CPU / None accepted for NumPy-like backends) and then calls the same-named
    #   function of xp with shape, dtype and any extra keyword arguments; device itself
    #   is not forwarded.
    # Outputs: the new array. Exceptions: ValueError from _check_device for an
    #   unsupported device.
    _check_device(xp, device)
    return xp.empty(shape, dtype=dtype, **kwargs)


def empty_like(
    x: Array,
    /,
    xp: Namespace,
    *,
    dtype: DType | None = None,
    device: Device | None = None,
    **kwargs: object,
) -> Array:
    # [FUNCTION DOC]
    # Array API creation function: validates the device keyword with _check_device (only
    #   CPU / None accepted for NumPy-like backends) and then calls the same-named
    #   function of xp with x, dtype and any extra keyword arguments; device itself is
    #   not forwarded.
    # Outputs: the new array. Exceptions: ValueError from _check_device for an
    #   unsupported device.
    _check_device(xp, device)
    return xp.empty_like(x, dtype=dtype, **kwargs)


def eye(
    n_rows: int,
    n_cols: int | None = None,
    /,
    *,
    xp: Namespace,
    k: int = 0,
    dtype: DType | None = None,
    device: Device | None = None,
    **kwargs: object,
) -> Array:
    # [FUNCTION DOC]
    # Array API creation function: validates the device keyword with _check_device (only
    #   CPU / None accepted for NumPy-like backends) and then calls the same-named
    #   function of xp with n_rows, optional n_cols (passed as M) and diagonal offset k,
    #   dtype and any extra keyword arguments; device itself is not forwarded.
    # Outputs: the new array. Exceptions: ValueError from _check_device for an
    #   unsupported device.
    _check_device(xp, device)
    return xp.eye(n_rows, M=n_cols, k=k, dtype=dtype, **kwargs)


def full(
    shape: int | tuple[int, ...],
    fill_value: complex,
    xp: Namespace,
    *,
    dtype: DType | None = None,
    device: Device | None = None,
    **kwargs: object,
) -> Array:
    # [FUNCTION DOC]
    # Array API creation function: validates the device keyword with _check_device (only
    #   CPU / None accepted for NumPy-like backends) and then calls the same-named
    #   function of xp with shape, fill_value, dtype and any extra keyword arguments;
    #   device itself is not forwarded.
    # Outputs: the new array. Exceptions: ValueError from _check_device for an
    #   unsupported device.
    _check_device(xp, device)
    return xp.full(shape, fill_value, dtype=dtype, **kwargs)


def full_like(
    x: Array,
    /,
    fill_value: complex,
    *,
    xp: Namespace,
    dtype: DType | None = None,
    device: Device | None = None,
    **kwargs: object,
) -> Array:
    # [FUNCTION DOC]
    # Array API creation function: validates the device keyword with _check_device (only
    #   CPU / None accepted for NumPy-like backends) and then calls the same-named
    #   function of xp with x, fill_value, dtype and any extra keyword arguments; device
    #   itself is not forwarded.
    # Outputs: the new array. Exceptions: ValueError from _check_device for an
    #   unsupported device.
    _check_device(xp, device)
    return xp.full_like(x, fill_value, dtype=dtype, **kwargs)


def linspace(
    start: float,
    stop: float,
    /,
    num: int,
    *,
    xp: Namespace,
    dtype: DType | None = None,
    device: Device | None = None,
    endpoint: bool = True,
    **kwargs: object,
) -> Array:
    # [FUNCTION DOC]
    # Array API creation function: validates the device keyword with _check_device (only
    #   CPU / None accepted for NumPy-like backends) and then calls the same-named
    #   function of xp with start, stop, num and endpoint, dtype and any extra keyword
    #   arguments; device itself is not forwarded.
    # Outputs: the new array. Exceptions: ValueError from _check_device for an
    #   unsupported device.
    _check_device(xp, device)
    return xp.linspace(start, stop, num, dtype=dtype, endpoint=endpoint, **kwargs)


def ones(
    shape: int | tuple[int, ...],
    xp: Namespace,
    *,
    dtype: DType | None = None,
    device: Device | None = None,
    **kwargs: object,
) -> Array:
    # [FUNCTION DOC]
    # Array API creation function: validates the device keyword with _check_device (only
    #   CPU / None accepted for NumPy-like backends) and then calls the same-named
    #   function of xp with shape, dtype and any extra keyword arguments; device itself
    #   is not forwarded.
    # Outputs: the new array. Exceptions: ValueError from _check_device for an
    #   unsupported device.
    _check_device(xp, device)
    return xp.ones(shape, dtype=dtype, **kwargs)


def ones_like(
    x: Array,
    /,
    xp: Namespace,
    *,
    dtype: DType | None = None,
    device: Device | None = None,
    **kwargs: object,
) -> Array:
    # [FUNCTION DOC]
    # Array API creation function: validates the device keyword with _check_device (only
    #   CPU / None accepted for NumPy-like backends) and then calls the same-named
    #   function of xp with x, dtype and any extra keyword arguments; device itself is
    #   not forwarded.
    # Outputs: the new array. Exceptions: ValueError from _check_device for an
    #   unsupported device.
    _check_device(xp, device)
    return xp.ones_like(x, dtype=dtype, **kwargs)


def zeros(
    shape: int | tuple[int, ...],
    xp: Namespace,
    *,
    dtype: DType | None = None,
    device: Device | None = None,
    **kwargs: object,
) -> Array:
    # [FUNCTION DOC]
    # Array API creation function: validates the device keyword with _check_device (only
    #   CPU / None accepted for NumPy-like backends) and then calls the same-named
    #   function of xp with shape, dtype and any extra keyword arguments; device itself
    #   is not forwarded.
    # Outputs: the new array. Exceptions: ValueError from _check_device for an
    #   unsupported device.
    _check_device(xp, device)
    return xp.zeros(shape, dtype=dtype, **kwargs)


def zeros_like(
    x: Array,
    /,
    xp: Namespace,
    *,
    dtype: DType | None = None,
    device: Device | None = None,
    **kwargs: object,
) -> Array:
    # [FUNCTION DOC]
    # Array API creation function: validates the device keyword with _check_device (only
    #   CPU / None accepted for NumPy-like backends) and then calls the same-named
    #   function of xp with x, dtype and any extra keyword arguments; device itself is
    #   not forwarded.
    # Outputs: the new array. Exceptions: ValueError from _check_device for an
    #   unsupported device.
    _check_device(xp, device)
    return xp.zeros_like(x, dtype=dtype, **kwargs)


# np.unique() is split into four functions in the array API:
# unique_all, unique_counts, unique_inverse, and unique_values (this is done
# to remove polymorphic return types).

# The functions here return namedtuples (np.unique() returns a normal
# tuple).


# Note that these named tuples aren't actually part of the standard namespace,
# but I don't see any issue with exporting the names here regardless.
class UniqueAllResult(NamedTuple):
    # [CLASS DOC]
    # Named tuple (values, indices, inverse_indices, counts) returned by unique_all.
    values: Array
    indices: Array
    inverse_indices: Array
    counts: Array


class UniqueCountsResult(NamedTuple):
    # [CLASS DOC]
    # Named tuple (values, counts) returned by unique_counts.
    values: Array
    counts: Array


class UniqueInverseResult(NamedTuple):
    # [CLASS DOC]
    # Named tuple (values, inverse_indices) returned by unique_inverse.
    values: Array
    inverse_indices: Array


def _unique_kwargs(xp: Namespace) -> dict[str, bool]:
    # [FUNCTION DOC]
    # HELPER: Returns {"equal_nan": False} if xp.unique's signature has an equal_nan
    #   parameter (newer NumPy/CuPy), else an empty dict, so NaNs are treated as
    #   distinct as the standard requires.
    # Older versions of NumPy and CuPy do not have equal_nan. Rather than
    # trying to parse version numbers, just check if equal_nan is in the
    # signature.
    s = inspect.signature(xp.unique)
    if "equal_nan" in s.parameters:
        return {"equal_nan": False}
    return {}


def unique_all(x: Array, /, xp: Namespace) -> UniqueAllResult:
    # [FUNCTION DOC]
    # Calls xp.unique with indices, inverse and counts requested, reshapes the
    #   (flattened) inverse indices to x's shape, and returns a UniqueAllResult.
    kwargs = _unique_kwargs(xp)
    values, indices, inverse_indices, counts = xp.unique(
        x,
        return_counts=True,
        return_index=True,
        return_inverse=True,
        **kwargs,
    )
    # np.unique() flattens inverse indices, but they need to share x's shape
    # See https://github.com/numpy/numpy/issues/20638
    inverse_indices = inverse_indices.reshape(x.shape)
    return UniqueAllResult(
        values,
        indices,
        inverse_indices,
        counts,
    )


def unique_counts(x: Array, /, xp: Namespace) -> UniqueCountsResult:
    # [FUNCTION DOC]
    # Calls xp.unique requesting only counts and returns UniqueCountsResult(values,
    #   counts).
    kwargs = _unique_kwargs(xp)
    res = xp.unique(
        x, return_counts=True, return_index=False, return_inverse=False, **kwargs
    )

    return UniqueCountsResult(*res)


def unique_inverse(x: Array, /, xp: Namespace) -> UniqueInverseResult:
    # [FUNCTION DOC]
    # Calls xp.unique requesting only the inverse, reshapes inverse indices to x's shape
    #   and returns UniqueInverseResult.
    kwargs = _unique_kwargs(xp)
    values, inverse_indices = xp.unique(
        x,
        return_counts=False,
        return_index=False,
        return_inverse=True,
        **kwargs,
    )
    # xp.unique() flattens inverse indices, but they need to share x's shape
    # See https://github.com/numpy/numpy/issues/20638
    inverse_indices = inverse_indices.reshape(x.shape)
    return UniqueInverseResult(values, inverse_indices)


def unique_values(x: Array, /, xp: Namespace) -> Array:
    # [FUNCTION DOC]
    # Returns the sorted unique values of x (flattened) via xp.unique with no extra
    #   outputs.
    kwargs = _unique_kwargs(xp)
    return xp.unique(
        x,
        return_counts=False,
        return_index=False,
        return_inverse=False,
        **kwargs,
    )


# These functions have different keyword argument names


def std(
    x: Array,
    /,
    xp: Namespace,
    *,
    axis: int | tuple[int, ...] | None = None,
    correction: float = 0.0,  # correction instead of ddof
    keepdims: bool = False,
    **kwargs: object,
) -> Array:
    # [FUNCTION DOC]
    # Standard deviation with the Array API keyword correction mapped to NumPy's ddof;
    #   axis, keepdims and extra kwargs forwarded to xp.std.
    return xp.std(x, axis=axis, ddof=correction, keepdims=keepdims, **kwargs)


def var(
    x: Array,
    /,
    xp: Namespace,
    *,
    axis: int | tuple[int, ...] | None = None,
    correction: float = 0.0,  # correction instead of ddof
    keepdims: bool = False,
    **kwargs: object,
) -> Array:
    # [FUNCTION DOC]
    # Variance with correction mapped to ddof; otherwise forwards to xp.var.
    return xp.var(x, axis=axis, ddof=correction, keepdims=keepdims, **kwargs)


# cumulative_sum is renamed from cumsum, and adds the include_initial keyword
# argument


def cumulative_sum(
    x: Array,
    /,
    xp: Namespace,
    *,
    axis: int | None = None,
    dtype: DType | None = None,
    include_initial: bool = False,
    **kwargs: object,
) -> Array:
    # [FUNCTION DOC]
    # Array API cumulative_sum built on xp.cumsum.
    # Behavior: if axis is None, require x.ndim <= 1 and use axis 0; compute the
    #   cumulative sum with dtype; if include_initial, prepend along axis a slab of
    #   zeros (shape of x with size 1 on axis, result dtype, same device).
    # Exceptions: ValueError when axis is None and x has more than one dimension.
    # KNOWN ISSUE: for a 0-d x, axis defaults to 0 and include_initial would index an
    #   empty shape list (IndexError); the TODO notes the standard is unclear.
    wrapped_xp = array_namespace(x)

    # TODO: The standard is not clear about what should happen when x.ndim == 0.
    if axis is None:
        if x.ndim > 1:
            raise ValueError(
                "axis must be specified in cumulative_sum for more than one dimension"
            )
        axis = 0

    res = xp.cumsum(x, axis=axis, dtype=dtype, **kwargs)

    # np.cumsum does not support include_initial
    if include_initial:
        initial_shape = list(x.shape)
        initial_shape[axis] = 1
        res = xp.concatenate(
            [
                wrapped_xp.zeros(
                    shape=initial_shape, dtype=res.dtype, device=_get_device(res)
                ),
                res,
            ],
            axis=axis,
        )
    return res


def cumulative_prod(
    x: Array,
    /,
    xp: Namespace,
    *,
    axis: int | None = None,
    dtype: DType | None = None,
    include_initial: bool = False,
    **kwargs: object,
) -> Array:
    # [FUNCTION DOC]
    # Same as cumulative_sum but using xp.cumprod and prepending ones when
    #   include_initial is True. Exceptions: ValueError when axis is None and x.ndim >
    #   1.
    wrapped_xp = array_namespace(x)

    if axis is None:
        if x.ndim > 1:
            raise ValueError(
                "axis must be specified in cumulative_prod for more than one dimension"
            )
        axis = 0

    res = xp.cumprod(x, axis=axis, dtype=dtype, **kwargs)

    # np.cumprod does not support include_initial
    if include_initial:
        initial_shape = list(x.shape)
        initial_shape[axis] = 1
        res = xp.concatenate(
            [
                wrapped_xp.ones(
                    shape=initial_shape, dtype=res.dtype, device=_get_device(res)
                ),
                res,
            ],
            axis=axis,
        )
    return res


# The min and max argument names in clip are different and not optional in numpy, and type
# promotion behavior is different.
def clip(
    x: Array,
    /,
    min: float | Array | None = None,
    max: float | Array | None = None,
    *,
    xp: Namespace,
    # TODO: np.clip has other ufunc kwargs
    out: Array | None = None,
) -> Array:
    # [FUNCTION DOC]
    # Array API clip: result has x's dtype (no type promotion) and broadcast shape of x,
    #   min and max.
    # Behavior: compute shapes of min/max (scalars count as shape ()), broadcast with
    #   x's shape; for integral x drop Python-int bounds that lie outside the dtype's
    #   range; allocate out (if not given) with x's dtype on x's device and copy x into
    #   it; for each provided bound convert it to x's dtype on the device, broadcast it,
    #   and overwrite elements where out is beyond the bound or the bound is NaN.
    # Inputs: x; min, max - scalar, array or None; xp; out - optional preallocated
    #   output.
    # Outputs: out indexed with an empty tuple (a scalar for 0-d results).
    # Side effects: writes into out when provided. Bounds that overflow when cast to x's
    #   dtype behave as NumPy's cast does (documented as unspecified).
    # KNOWN ISSUE: the assert on out is a type-narrowing workaround only; NumPy integer
    #   scalars (e.g. np.int64) are not treated as scalars by _isscalar and their out-
    #   of-range check is skipped.
    def _isscalar(a: object) -> TypeIs[float | None]:
        # [FUNCTION DOC]
        # HELPER: True if a is a Python int/float (including subclasses such as bool and
        #   numpy float64) or None.
        return isinstance(a, int | float) or a is None

    min_shape = () if _isscalar(min) else min.shape
    max_shape = () if _isscalar(max) else max.shape

    wrapped_xp = array_namespace(x)

    result_shape = xp.broadcast_shapes(x.shape, min_shape, max_shape)

    # np.clip does type promotion but the array API clip requires that the
    # output have the same dtype as x. We do this instead of just downcasting
    # the result of xp.clip() to handle some corner cases better (e.g.,
    # avoiding uint64 -> float64 promotion).

    # Note: cases where min or max overflow (integer) or round (float) in the
    # wrong direction when downcasting to x.dtype are unspecified. This code
    # just does whatever NumPy does when it downcasts in the assignment, but
    # other behavior could be preferred, especially for integers. For example,
    # this code produces:

    # >>> clip(asarray(0, dtype=int8), asarray(128, dtype=int16), None)
    # -128

    # but an answer of 0 might be preferred. See
    # https://github.com/numpy/numpy/issues/24976 for more discussion on this issue.

    # At least handle the case of Python integers correctly (see
    # https://github.com/numpy/numpy/pull/26892).
    if wrapped_xp.isdtype(x.dtype, "integral"):
        if type(min) is int and min <= wrapped_xp.iinfo(x.dtype).min:
            min = None
        if type(max) is int and max >= wrapped_xp.iinfo(x.dtype).max:
            max = None

    dev = _get_device(x)
    if out is None:
        out = wrapped_xp.empty(result_shape, dtype=x.dtype, device=dev)
    assert out is not None  # workaround for a type-narrowing issue in pyright
    out[()] = x

    if min is not None:
        a = wrapped_xp.asarray(min, dtype=x.dtype, device=dev)
        a = xp.broadcast_to(a, result_shape)
        ia = (out < a) | xp.isnan(a)
        out[ia] = a[ia]

    if max is not None:
        b = wrapped_xp.asarray(max, dtype=x.dtype, device=dev)
        b = xp.broadcast_to(b, result_shape)
        ib = (out > b) | xp.isnan(b)
        out[ib] = b[ib]

    # Return a scalar for 0-D
    return out[()]


# Unlike transpose(), the axes argument to permute_dims() is required.
def permute_dims(x: Array, /, axes: tuple[int, ...], xp: Namespace) -> Array:
    # [FUNCTION DOC]
    # Returns xp.transpose of x with the required axes permutation.
    return xp.transpose(x, axes)


# np.reshape calls the keyword argument 'newshape' instead of 'shape'
def reshape(
    x: Array,
    /,
    shape: tuple[int, ...],
    xp: Namespace,
    *,
    copy: bool | None = None,
    **kwargs: object,
) -> Array:
    # [FUNCTION DOC]
    # Array API reshape with a copy keyword.
    # Behavior: copy True copies x first then reshapes; copy False creates a view and
    #   assigns the new shape in place, raising if a copy would be needed; None
    #   delegates to xp.reshape with extra kwargs.
    # Exceptions: errors from the backend when the shape is incompatible (copy False
    #   raises AttributeError/ValueError when a view is impossible).
    # KNOWN ISSUE: with copy=False any extra kwargs (e.g. order) are silently ignored.
    if copy is True:
        x = x.copy()
    elif copy is False:
        y = x.view()
        y.shape = shape
        return y
    return xp.reshape(x, shape, **kwargs)


# The descending keyword is new in sort and argsort, and 'kind' replaced with
# 'stable'
def argsort(
    x: Array,
    /,
    xp: Namespace,
    *,
    axis: int = -1,
    descending: bool = False,
    stable: bool = True,
    **kwargs: object,
) -> Array:
    # [FUNCTION DOC]
    # Array API argsort with descending and stable flags.
    # Behavior: when stable, set kind to "stable" in kwargs; ascending is a direct
    #   xp.argsort. For descending, argsort the array flipped along axis, flip the
    #   result back, and map each index i to (n-1-i) so equal elements keep their
    #   original relative order.
    # Side effects: mutates the local kwargs dict only.
    # Note: this keyword argument is different, and the default is different.
    # We set it in kwargs like this because numpy.sort uses kind='quicksort'
    # as the default whereas cupy.sort uses kind=None.
    if stable:
        kwargs["kind"] = "stable"
    if not descending:
        res = xp.argsort(x, axis=axis, **kwargs)
    else:
        # As NumPy has no native descending sort, we imitate it here. Note that
        # simply flipping the results of xp.argsort(x, ...) would not
        # respect the relative order like it would in native descending sorts.
        res = xp.flip(
            xp.argsort(xp.flip(x, axis=axis), axis=axis, **kwargs),
            axis=axis,
        )
        # Rely on flip()/argsort() to validate axis
        normalised_axis = axis if axis >= 0 else x.ndim + axis
        max_i = x.shape[normalised_axis] - 1
        res = max_i - res
    return res


def sort(
    x: Array,
    /,
    xp: Namespace,
    *,
    axis: int = -1,
    descending: bool = False,
    stable: bool = True,
    **kwargs: object,
) -> Array:
    # [FUNCTION DOC]
    # Sorts along axis (stable by default via kind "stable"); for descending simply
    #   flips the ascending result (stability of ties is irrelevant for values).
    # Note: this keyword argument is different, and the default is different.
    # We set it in kwargs like this because numpy.sort uses kind='quicksort'
    # as the default whereas cupy.sort uses kind=None.
    if stable:
        kwargs["kind"] = "stable"
    res = xp.sort(x, axis=axis, **kwargs)
    if descending:
        res = xp.flip(res, axis=axis)
    return res


# nonzero should error for zero-dimensional arrays
def nonzero(x: Array, /, xp: Namespace, **kwargs: object) -> tuple[Array, ...]:
    # [FUNCTION DOC]
    # Forwards to xp.nonzero but raises ValueError for 0-d inputs as the standard
    #   requires.
    if x.ndim == 0:
        raise ValueError("nonzero() does not support zero-dimensional arrays")
    return xp.nonzero(x, **kwargs)


# linear algebra functions


def matmul(x1: Array, x2: Array, /, xp: Namespace, **kwargs: object) -> Array:
    # [FUNCTION DOC]
    # Thin forward to xp.matmul.
    return xp.matmul(x1, x2, **kwargs)


# Unlike transpose, matrix_transpose only transposes the last two axes.
def matrix_transpose(x: Array, /, xp: Namespace) -> Array:
    # [FUNCTION DOC]
    # Swaps the last two axes; raises ValueError if x has fewer than 2 dimensions.
    if x.ndim < 2:
        raise ValueError("x must be at least 2-dimensional for matrix_transpose")
    return xp.swapaxes(x, -1, -2)


def tensordot(
    x1: Array,
    x2: Array,
    /,
    xp: Namespace,
    *,
    axes: int | tuple[Sequence[int], Sequence[int]] = 2,
    **kwargs: object,
) -> Array:
    # [FUNCTION DOC]
    # Forwards to xp.tensordot with the axes keyword (default 2).
    return xp.tensordot(x1, x2, axes=axes, **kwargs)


def vecdot(x1: Array, x2: Array, /, xp: Namespace, *, axis: int = -1) -> Array:
    # [FUNCTION DOC]
    # Conjugated vector dot product along axis with broadcasting.
    # Behavior: check equal sizes along axis; move axis to the end in both inputs;
    #   broadcast them (broadcast_tensors for torch-like, else broadcast_arrays);
    #   compute conj(x1) as row vectors times x2 as column vectors with matmul and drop
    #   the two singleton dims.
    # Exceptions: ValueError on size mismatch along axis.
    if x1.shape[axis] != x2.shape[axis]:
        raise ValueError("x1 and x2 must have the same size along the given axis")

    if hasattr(xp, "broadcast_tensors"):
        _broadcast = xp.broadcast_tensors
    else:
        _broadcast = xp.broadcast_arrays

    x1_ = xp.moveaxis(x1, axis, -1)
    x2_ = xp.moveaxis(x2, axis, -1)
    x1_, x2_ = _broadcast(x1_, x2_)

    res = xp.conj(x1_[..., None, :]) @ x2_[..., None]
    return res[..., 0, 0]


# isdtype is a new function in the 2022.12 array API specification.


def isdtype(
    dtype: DType,
    kind: DType | str | tuple[DType | str, ...],
    xp: Namespace,
    *,
    _tuple: bool = True,  # Disallow nested tuples
) -> bool:
    # [FUNCTION DOC]
    # Array API isdtype: tests whether dtype belongs to kind.
    # Behavior: a tuple kind returns True if any element matches (nested tuples are not
    #   expanded, compared by equality); string kinds map to xp.bool_ equality or
    #   issubdtype against signedinteger, unsignedinteger, integer, floating,
    #   complexfloating or number; any other kind is compared with ==.
    # Inputs: dtype; kind - dtype, kind string or tuple; xp; _tuple - private recursion
    #   guard.
    # Exceptions: ValueError for an unrecognized kind string.
    """
    Returns a boolean indicating whether a provided dtype is of a specified data type ``kind``.

    Note that outside of this function, this compat library does not yet fully
    support complex numbers.

    See
    https://data-apis.org/array-api/latest/API_specification/generated/array_api.isdtype.html
    for more details
    """
    if isinstance(kind, tuple) and _tuple:
        return any(
            isdtype(dtype, k, xp, _tuple=False)
            for k in cast("tuple[DType | str, ...]", kind)
        )
    elif isinstance(kind, str):
        if kind == "bool":
            return dtype == xp.bool_
        elif kind == "signed integer":
            return xp.issubdtype(dtype, xp.signedinteger)
        elif kind == "unsigned integer":
            return xp.issubdtype(dtype, xp.unsignedinteger)
        elif kind == "integral":
            return xp.issubdtype(dtype, xp.integer)
        elif kind == "real floating":
            return xp.issubdtype(dtype, xp.floating)
        elif kind == "complex floating":
            return xp.issubdtype(dtype, xp.complexfloating)
        elif kind == "numeric":
            return xp.issubdtype(dtype, xp.number)
        else:
            raise ValueError(f"Unrecognized data type kind: {kind!r}")
    else:
        # This will allow things that aren't required by the spec, like
        # isdtype(np.float64, float) or isdtype(np.int64, 'l'). Should we be
        # more strict here to match the type annotation? Note that the
        # array_api_strict implementation will be very strict.
        return dtype == kind


# unstack is a new function in the 2023.12 array API standard
def unstack(x: Array, /, xp: Namespace, *, axis: int = 0) -> tuple[Array, ...]:
    # [FUNCTION DOC]
    # Splits x along axis into a tuple of arrays by moving axis to the front and
    #   iterating. Raises ValueError for 0-d input.
    if x.ndim == 0:
        raise ValueError("Input array must be at least 1-d.")
    return tuple(xp.moveaxis(x, axis, 0))


# numpy 1.26 does not use the standard definition for sign on complex numbers


def sign(x: Array, /, xp: Namespace, **kwargs: object) -> Array:
    # [FUNCTION DOC]
    # Array API sign: for complex dtypes computes x divided by its absolute value and
    #   sets entries where x is 0 to 0; otherwise uses xp.sign. For CuPy with real
    #   floats, NaN entries are set back to NaN.
    # Outputs: result indexed with an empty tuple (scalar for 0-d).
    # Side effects: none on x (writes into the freshly computed output).
    if isdtype(x.dtype, "complex floating", xp=xp):
        out = (x / xp.abs(x, **kwargs))[...]
        # sign(0) = 0 but the above formula would give nan
        out[x == 0j] = 0j
    else:
        out = xp.sign(x, **kwargs)
    # CuPy sign() does not propagate nans. See
    # https://github.com/data-apis/array-api-compat/issues/136
    if is_cupy_namespace(xp) and isdtype(x.dtype, "real floating", xp=xp):
        out[xp.isnan(x)] = xp.nan
    return out[()]


def finfo(type_: DType | Array, /, xp: Namespace) -> Any:
    # [FUNCTION DOC]
    # Calls xp.finfo on type_; if that raises ValueError/TypeError (an array was passed)
    #   retries with type_.dtype.
    # It is surprisingly difficult to recognize a dtype apart from an array.
    # np.int64 is not the same as np.asarray(1).dtype!
    try:
        return xp.finfo(type_)
    except (ValueError, TypeError):
        return xp.finfo(type_.dtype)


def iinfo(type_: DType | Array, /, xp: Namespace) -> Any:
    # [FUNCTION DOC]
    # Calls xp.iinfo on type_, falling back to type_.dtype on ValueError/TypeError.
    try:
        return xp.iinfo(type_)
    except (ValueError, TypeError):
        return xp.iinfo(type_.dtype)


__all__ = [
    "arange",
    "empty",
    "empty_like",
    "eye",
    "full",
    "full_like",
    "linspace",
    "ones",
    "ones_like",
    "zeros",
    "zeros_like",
    "UniqueAllResult",
    "UniqueCountsResult",
    "UniqueInverseResult",
    "unique_all",
    "unique_counts",
    "unique_inverse",
    "unique_values",
    "std",
    "var",
    "cumulative_sum",
    "cumulative_prod",
    "clip",
    "permute_dims",
    "reshape",
    "argsort",
    "sort",
    "nonzero",
    "matmul",
    "matrix_transpose",
    "tensordot",
    "vecdot",
    "isdtype",
    "unstack",
    "sign",
    "finfo",
    "iinfo",
]

def __dir__() -> list[str]:
    # [FUNCTION DOC]
    # HELPER: Module-level dir() hook returning __all__.
    return __all__
