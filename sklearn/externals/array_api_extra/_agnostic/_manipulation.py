# [MODULE DOC]
# HELPER: Generic array API implementations of atleast_nd, broadcast_shapes, expand_dims
#   and pad used by array_api_extra._manipulation.
"""Array-agnostic implementations for manipulation functions."""

import math
import typing
from collections.abc import Sequence

from .._at import at
from .._lib import _compat, _helpers
from .._lib._typing import Array, ArrayNamespace

__all__ = ["atleast_nd", "broadcast_shapes", "expand_dims", "pad"]


def atleast_nd(x: Array, /, *, ndim: int, xp: ArrayNamespace) -> Array:
    # [FUNCTION DOC]
    # HELPER: Recursively prepends a size-1 axis (xp.expand_dims at axis 0) until x has
    #   at least ndim dimensions; returns x unchanged if it already does.
    # numpydoc ignore=PR01,RT01
    """See docstring in `array_api_extra._manipulation`."""

    if x.ndim < ndim:
        x = xp.expand_dims(x, axis=0)
        x = atleast_nd(x, ndim=ndim, xp=xp)
    return x


# `float` in signature to accept `math.nan` for Dask.
# `int`s are still accepted as `float` is a superclass of `int` in typing
def broadcast_shapes(  # numpydoc ignore=PR01,RT01
    *shapes: tuple[float | None, ...],
) -> tuple[int | None, ...]:
    # [FUNCTION DOC]
    # HELPER: Computes the broadcast shape of several shape tuples without arrays.
    #   Right-aligns shapes; for each output axis collects the sizes present, records
    #   whether any is unknown (None or NaN, as used by Dask), discards 1/None/NaN, and
    #   requires at most one remaining distinct size. Output size is None if any input
    #   was unknown, else the remaining size, else 1. No shapes gives ().
    # Exceptions: ValueError when two different non-1 known sizes meet on an axis.
    # KNOWN ISSUE: NaN detection relies on set membership finding the math.nan object by
    #   identity; a different NaN float object (e.g. from Dask arithmetic) is neither
    #   detected nor removed and can trigger a spurious mismatch or be returned as the
    #   size.
    """See docstring in `array_api_extra._manipulation`."""
    if not shapes:
        return ()  # Match NumPy output

    ndim = max(len(shape) for shape in shapes)
    out: list[int | None] = []
    for axis in range(-ndim, 0):
        sizes = {shape[axis] for shape in shapes if axis >= -len(shape)}
        # Dask uses NaN for unknown shape, which predates the Array API spec for None
        none_size = None in sizes or math.nan in sizes  # noqa: PLW0177
        sizes -= {1, None, math.nan}
        if len(sizes) > 1:
            msg = (
                "shape mismatch: objects cannot be broadcast to a single shape: "
                f"{shapes}."
            )
            raise ValueError(msg)
        out.append(None if none_size else typing.cast(int, sizes.pop()) if sizes else 1)

    return tuple(out)


def expand_dims(
    a: Array, /, *, axis: tuple[int, ...] = (0,), xp: ArrayNamespace
) -> Array:
    # [FUNCTION DOC]
    # HELPER: Inserts size-1 axes at each position of axis, applying xp.expand_dims one
    #   axis at a time in ascending sorted order (so positions refer to the output
    #   array). Validation is done by the public wrapper.
    # numpydoc ignore=PR01,RT01
    """See docstring in `array_api_extra._manipulation`."""
    for i in sorted(axis):
        a = xp.expand_dims(a, axis=i)
    return a


def pad(
    x: Array,
    pad_width: int | tuple[int, int] | Sequence[tuple[int, int]],
    *,
    constant_values: complex = 0,
    xp: ArrayNamespace,
) -> Array:  # numpydoc ignore=PR01,RT01
    # [FUNCTION DOC]
    # HELPER: Constant padding. Normalises pad_width to one (before, after) pair per
    #   axis, computes the padded shape and the slice that addresses the original data,
    #   allocates a full array of constant_values with x's dtype/device and writes x
    #   into that slice with at().set.
    # Exceptions: ValueError if a per-axis entry is not a 2-tuple; errors from
    #   eager_shape for unknown shapes.
    """See docstring in `array_api_extra._manipulation`."""
    pad_width_seq = _helpers.normalize_pad_width(pad_width, x.ndim)

    slices: list[slice] = []
    newshape: list[int] = []
    for ax, w_tpl in enumerate(pad_width_seq):
        if len(w_tpl) != 2:
            msg = f"expect a 2-tuple (before, after), got {w_tpl}."
            raise ValueError(msg)

        sh = _helpers.eager_shape(x)[ax]

        if w_tpl[0] == 0 and w_tpl[1] == 0:
            sl = slice(None, None, None)
        else:
            stop: int | None
            start, stop = w_tpl
            stop = None if stop == 0 else -stop

            sl = slice(start, stop, None)
            sh += w_tpl[0] + w_tpl[1]

        newshape.append(sh)
        slices.append(sl)

    padded = xp.full(
        tuple(newshape),
        fill_value=constant_values,
        dtype=x.dtype,
        device=_compat.device(x),
    )
    return at(padded, tuple(slices)).set(x)
