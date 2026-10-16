# [MODULE DOC]
# HELPER: Generic array API implementations of elementwise functions: apply_where
#   (public), isclose, nan_to_num, sinc, angle (public), deg2rad and rad2deg.
"""Array-agnostic implementations for element-wise functions."""

import typing
from collections.abc import Callable
from types import NoneType

from .._at import at
from .._lib import _compat, _helpers
from .._lib._typing import Array, ArrayNamespace
from . import _inspection

__all__ = [
    "angle",
    "apply_where",
    "deg2rad",
    "isclose",
    "nan_to_num",
    "rad2deg",
    "sinc",
]


@typing.overload
def apply_where(  # numpydoc ignore=GL08
    cond: Array,
    args: Array | tuple[Array, ...],
    f1: Callable[..., Array],
    f2: Callable[..., Array],
    /,
    *,
    kwargs: dict[str, Array] | None = None,
    xp: ArrayNamespace | None = None,
) -> Array: ...
    # [FUNCTION DOC]
    # Typing overload for apply_where with f2 given (no fill_value). No runtime
    #   behaviour.


@typing.overload
def apply_where(  # numpydoc ignore=GL08
    cond: Array,
    args: Array | tuple[Array, ...],
    f1: Callable[..., Array],
    /,
    *,
    fill_value: Array | complex,
    kwargs: dict[str, Array] | None = None,
    xp: ArrayNamespace | None = None,
) -> Array: ...
    # [FUNCTION DOC]
    # Typing overload for apply_where with fill_value given (no f2). No runtime
    #   behaviour.


def apply_where(  # numpydoc ignore=PR01,PR02
    cond: Array,
    args: Array | tuple[Array, ...],
    f1: Callable[..., Array],
    f2: Callable[..., Array] | None = None,
    /,
    *,
    fill_value: Array | complex | None = None,
    kwargs: dict[str, Array] | None = None,
    xp: ArrayNamespace | None = None,
) -> Array:
    # [FUNCTION DOC]
    # Validates that exactly one of f2/fill_value is given, packs args (single array or
    #   tuple) followed by kwargs values into one positional list remembering the kwarg
    #   keys, infers xp from cond, fill_value and args if not given, then broadcasts
    #   cond, args (and fill_value unless it is a Python scalar or None) to a common
    #   shape. On Dask runs _apply_where per chunk via map_blocks with the chunk (meta)
    #   namespace; otherwise calls _apply_where directly.
    # Exceptions: TypeError when both or neither of f2 and fill_value are given;
    #   broadcasting errors from xp.
    # KNOWN ISSUE: the docstring says f2 receives arg[cond], but the code passes the
    #   elements where cond is False.
    """
    Run one of two elementwise functions depending on a condition.

    Equivalent to ``f1(*args) if cond else fill_value`` performed elementwise
    when `fill_value` is defined, otherwise to ``f1(*args) if cond else f2(*args)``.

    Parameters
    ----------
    cond : array
        The condition, expressed as a boolean array.
    args : Array or tuple of Arrays
        Argument(s) to `f1` (and `f2`). Must be broadcastable with `cond`.
    f1 : callable
        Elementwise function of `args`, returning a single array.
        Where `cond` is True, output will be ``f1(arg0[cond], arg1[cond], ...)``.
    f2 : callable, optional
        Elementwise function of `args`, returning a single array.
        Where `cond` is False, output will be ``f2(arg0[cond], arg1[cond], ...)``.
        Mutually exclusive with `fill_value`.
    fill_value : Array or scalar, optional
        If provided, value with which to fill output array where `cond` is False.
        It does not need to be scalar; it needs however to be broadcastable with
        `cond` and `args`.
        Mutually exclusive with `f2`. You must provide one or the other.
    kwargs : dict of str : Array pairs
        Keyword argument(s) to `f1` (and `f2`). Values must be broadcastable with
        `cond`.
    xp : array_namespace, optional
        The standard-compatible namespace for `cond` and `args`. Default: infer.

    Returns
    -------
    Array
        An array with elements from the output of `f1` where `cond` is True and either
        the output of `f2` or `fill_value` where `cond` is False. The returned array has
        data type determined by type promotion rules between the output of `f1` and
        either `fill_value` or the output of `f2`.

    Notes
    -----
    ``xp.where(cond, f1(*args), f2(*args))`` requires explicitly evaluating `f1` even
    when `cond` is False, and `f2` when cond is True. This function evaluates each
    function only for their matching condition, if the backend allows for it.

    On Dask, `f1` and `f2` are applied to the individual chunks and should use functions
    from the namespace of the chunks.

    Examples
    --------
    >>> import array_api_strict as xp
    >>> import array_api_extra as xpx
    >>> a = xp.asarray([5.0, 4.0, 3.0])
    >>> b = xp.asarray([0.0, 2.0, 2.0])
    >>> def f(a, b):
    ...     return a // b
    >>> xpx.apply_where(b != 0, (a, b), f, fill_value=xp.nan)
    Array([nan,  2.,  1.], dtype=array_api_strict.float64)
    """
    # Parse and normalize arguments
    if (f2 is None) == (fill_value is None):
        msg = "Exactly one of `fill_value` or `f2` must be given."
        raise TypeError(msg)
    args_ = list(args) if isinstance(args, tuple) else [args]
    del args

    kwargs_ = {} if kwargs is None else kwargs
    kwkeys = list(kwargs_.keys())
    args_ = [*args_, *kwargs_.values()]
    del kwargs

    xp = _compat.array_namespace(cond, fill_value, *args_) if xp is None else xp

    if isinstance(fill_value, int | float | complex | NoneType):
        cond, *args_ = xp.broadcast_arrays(cond, *args_)
    else:
        cond, fill_value, *args_ = xp.broadcast_arrays(cond, fill_value, *args_)

    if _compat.is_dask_namespace(xp):
        meta_xp = _helpers.meta_namespace(cond, fill_value, *args_, xp=xp)
        # map_blocks doesn't descend into tuples of Arrays
        return xp.map_blocks(
            _apply_where, cond, f1, f2, fill_value, *args_, kwkeys=kwkeys, xp=meta_xp
        )

    return _apply_where(cond, f1, f2, fill_value, *args_, kwkeys=kwkeys, xp=xp)


def _apply_where(  # numpydoc ignore=PR01,RT01
    cond: Array,
    f1: Callable[..., Array],
    f2: Callable[..., Array] | None,
    fill_value: Array | complex | bool | None,
    *args: Array,
    kwkeys: list[str],
    xp: ArrayNamespace,
) -> Array:
    # [FUNCTION DOC]
    # HELPER: Core of apply_where (one Dask chunk or a whole array). Splits trailing
    #   args back into kwargs by kwkeys. If the backend lacks boolean indexing (e.g. JAX
    #   jit), evaluates f1 and f2 (or uses fill_value) on all elements and combines with
    #   where. Otherwise evaluates f1 only on cond-masked elements and f2 only on ~cond
    #   elements; the output dtype is the promotion of f1's result with fill_value or
    #   f2's result; output starts as a full/astype copy of fill_value or an empty array
    #   filled at ~cond with f2 results, then cond positions are set with f1 results via
    #   at().set.
    # Outputs: array of cond's shape. Side effects: none on inputs (at() copies where
    #   needed).
    """Helper of `apply_where`. On Dask, this runs on a single chunk."""

    nargs = len(args) - len(kwkeys)
    kwargs = dict(zip(kwkeys, args[nargs:], strict=True))
    args = args[:nargs]

    if not _helpers.capabilities(xp, cond)["boolean indexing"]:
        # jax.jit does not support assignment by boolean mask
        return xp.where(
            cond,
            f1(*args, **kwargs),
            f2(*args, **kwargs) if f2 is not None else fill_value,
        )

    temp1 = f1(
        *(arr[cond] for arr in args), **{key: val[cond] for key, val in kwargs.items()}
    )

    if f2 is None:
        dtype = xp.result_type(temp1, fill_value)
        if isinstance(fill_value, int | float | complex):
            out = xp.full_like(cond, dtype=dtype, fill_value=fill_value)
        else:
            out = xp.astype(fill_value, dtype, copy=True)
    else:
        ncond = ~cond
        temp2 = f2(
            *(arr[ncond] for arr in args),
            **{key: val[ncond] for key, val in kwargs.items()},
        )
        dtype = xp.result_type(temp1, temp2)
        out = xp.empty_like(cond, dtype=dtype)
        out = at(out, ncond).set(temp2)

    return at(out, cond).set(temp1)


def isclose(
    a: Array | complex,
    b: Array | complex,
    *,
    rtol: float = 1e-05,
    atol: float = 1e-08,
    equal_nan: bool = False,
    xp: ArrayNamespace,
) -> Array:  # numpydoc ignore=PR01,RT01
    # [FUNCTION DOC]
    # HELPER: Generic isclose. Converts a, b to arrays with asarrays. For
    #   floating/complex inputs uses apply_where: where either is infinite the result is
    #   True only if both are infinite with the same sign, elsewhere abs(a-b) <= atol +
    #   rtol*abs(b); NaN pairs become True if equal_nan. For bool inputs returns a==b,
    #   or all True when atol or rtol >= 1. For integers truncates atol to int and
    #   compares abs(a-b) to atol plus abs(b) integer-divided by int(1/rtol), skipping
    #   the relative term when rtol is 0 or 1/rtol exceeds b's dtype max.
    # Outputs: bool array of broadcast shape.
    # KNOWN ISSUE: for unsigned integer dtypes a-b wraps around when b > a, so abs(a-b)
    #   can be huge and close values compare False; the integer relative tolerance is an
    #   approximation (floor division).
    """See docstring in `array_api_extra._elementwise`."""
    a, b = _helpers.asarrays(a, b, xp=xp)

    a_inexact = xp.isdtype(a.dtype, ("real floating", "complex floating"))
    b_inexact = xp.isdtype(b.dtype, ("real floating", "complex floating"))
    if a_inexact or b_inexact:
        # prevent warnings on NumPy and Dask on inf - inf
        mxp = _helpers.meta_namespace(a, b, xp=xp)
        out = apply_where(
            xp.isinf(a) | xp.isinf(b),
            (a, b),
            lambda a, b: mxp.isinf(a) & mxp.isinf(b) & (mxp.sign(a) == mxp.sign(b)),  # pyright: ignore[reportUnknownArgumentType]
            # Note: inf <= inf is True!
            lambda a, b: mxp.abs(a - b) <= (atol + rtol * mxp.abs(b)),  # pyright: ignore[reportUnknownArgumentType]
            xp=xp,
        )
        if equal_nan:
            out = xp.where(xp.isnan(a) & xp.isnan(b), True, out)
        return out

    if xp.isdtype(a.dtype, "bool") or xp.isdtype(b.dtype, "bool"):
        if atol >= 1 or rtol >= 1:
            return xp.ones_like(a == b)
        return a == b

    # integer types
    atol = int(atol)
    if rtol == 0:
        return xp.abs(a - b) <= atol

    # Don't rely on OverflowError, as it is not guaranteed by the Array API.
    nrtol = int(1.0 / rtol)
    if nrtol > xp.iinfo(b.dtype).max:
        # rtol * max_int < 1, so it's inconsequential
        return xp.abs(a - b) <= atol
    return xp.abs(a - b) <= (atol + xp.abs(b) // nrtol)


def nan_to_num(  # numpydoc ignore=PR01,RT01
    x: Array,
    /,
    fill_value: float = 0.0,
    *,
    xp: ArrayNamespace,
) -> Array:
    # [FUNCTION DOC]
    # HELPER: Generic nan_to_num. Complex input: processes real and imaginary parts
    #   separately and recombines with 1j. Numeric input: replaces NaN with fill_value,
    #   +inf with the dtype's finfo max and -inf with finfo min. Other dtypes (bool) are
    #   returned unchanged.
    # Outputs: new array (input returned as-is for non-numeric).
    # KNOWN ISSUE: integer dtypes count as numeric, so they reach finfo, which raises
    #   for integer dtypes unless the public wrapper filters them out beforehand.
    """See docstring in `array_api_extra._elementwise`."""

    def perform_replacements(  # numpydoc ignore=PR01,RT01
        x: Array,
        fill_value: float,
        xp: ArrayNamespace,
    ) -> Array:
        # [FUNCTION DOC]
        # HELPER: Replaces NaN with fill_value, positive infinities with
        #   finfo(dtype).max and negative infinities with finfo(dtype).min (sign
        #   determined via signbit) using where.
        """Internal function to perform the replacements."""
        x = xp.where(xp.isnan(x), fill_value, x)

        # convert infinities to finite values
        finfo = xp.finfo(x.dtype)
        idx_posinf = xp.isinf(x) & ~xp.signbit(x)
        idx_neginf = xp.isinf(x) & xp.signbit(x)
        x = xp.where(idx_posinf, finfo.max, x)
        return xp.where(idx_neginf, finfo.min, x)

    if xp.isdtype(x.dtype, "complex floating"):
        return perform_replacements(
            xp.real(x),
            fill_value,
            xp,
        ) + 1j * perform_replacements(
            xp.imag(x),
            fill_value,
            xp,
        )

    if xp.isdtype(x.dtype, "numeric"):
        return perform_replacements(x, fill_value, xp)

    return x


def sinc(x: Array, /, *, xp: ArrayNamespace) -> Array:
    # [FUNCTION DOC]
    # HELPER: Generic normalised sinc: replaces zeros of x with the dtype's machine
    #   epsilon (so sin(y)/y tends to 1), multiplies by pi and returns sin(y)/y.
    #   Requires a real floating dtype (finfo).
    # numpydoc ignore=PR01,RT01
    """See docstring in `array_api_extra._elementwise`."""

    # no scalars in `where` - array-api#807
    y = xp.pi * xp.where(
        xp.astype(x, xp.bool),
        x,
        xp.asarray(xp.finfo(x.dtype).eps, dtype=x.dtype, device=_compat.device(x)),
    )
    return xp.sin(y) / y


def angle(z: Array, /, *, deg: bool = False, xp: ArrayNamespace | None = None) -> Array:
    # [FUNCTION DOC]
    # Angle (argument) of z via atan2(imag, real). Complex inputs use their parts; real
    #   floats use zero imaginary part; other dtypes are first cast to the namespace's
    #   default real floating dtype on z's device. Converts to degrees when deg is True.
    # Inputs: z - array; deg - bool; xp - optional namespace (inferred from z).
    # Outputs: real floating array in (-pi, pi] (or degrees).
    """
    Return the angle of the complex argument.

    Parameters
    ----------
    z : Array
        Input array.
    deg : bool, optional
        Return angle in degrees if True, radians if False (default).
    xp : array_namespace, optional
        The standard-compatible namespace for `z`. Default: infer.

    Returns
    -------
    array
        The counterclockwise angle from the positive real axis on the complex
        plane in the range ``(-pi, pi]``.

    Notes
    -----
    Real input ``x`` is interpreted as ``x + 0j``.

    Examples
    --------
    >>> import array_api_strict as xp
    >>> import array_api_extra as xpx
    >>> xpx.angle(xp.asarray([1.0, 1.0j, 1 + 1j]), xp=xp)
    Array([0.        , 1.57079633, 0.78539816], dtype=array_api_strict.float64)
    >>> xpx.angle(xp.asarray([1.0, 1.0j, 1 + 1j]), deg=True, xp=xp)
    Array([ 0., 90., 45.], dtype=array_api_strict.float64)
    """
    if xp is None:
        xp = _compat.array_namespace(z)
    if xp.isdtype(z.dtype, "complex floating"):
        zimag = xp.imag(z)
        zreal = xp.real(z)
    else:
        if not xp.isdtype(z.dtype, "real floating"):
            z = xp.astype(z, _inspection.default_dtype(xp, device=_compat.device(z)))
        zimag = xp.zeros_like(z)
        zreal = z
    a = xp.atan2(zimag, zreal)
    if deg:
        a = a * 180 / xp.pi
    return a


def deg2rad(x: Array, /, *, xp: ArrayNamespace) -> Array:
    # [FUNCTION DOC]
    # HELPER: Converts degrees to radians by multiplying by pi/180.
    # numpydoc ignore=PR01,RT01
    """See docstring in `array_api_extra._elementwise`."""
    return x * xp.pi / 180


def rad2deg(x: Array, /, *, xp: ArrayNamespace) -> Array:
    # [FUNCTION DOC]
    # HELPER: Converts radians to degrees by multiplying by 180/pi.
    # numpydoc ignore=PR01,RT01
    """See docstring in `array_api_extra._elementwise`."""
    return x * 180 / xp.pi
