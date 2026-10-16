# [MODULE DOC]
# HELPER: Generic array API implementations of isin, nunique, setdiff1d and union1d.
"""Array-agnostic implementations for set functions."""

from .._lib import _compat, _helpers
from .._lib._typing import Array, ArrayNamespace
from . import _inspection

__all__ = ["isin", "nunique", "setdiff1d", "union1d"]


def isin(  # numpydoc ignore=PR01,RT01
    a: Array,
    b: Array,
    /,
    *,
    assume_unique: bool = False,
    invert: bool = False,
    xp: ArrayNamespace,
) -> Array:
    # [FUNCTION DOC]
    # HELPER: Flattens a and b, runs _helpers.in1d (membership test, optionally
    #   inverted, optionally assuming unique inputs) and reshapes the boolean result
    #   back to a's original shape.
    """See docstring in `array_api_extra._set`."""
    original_a_shape = a.shape
    a = xp.reshape(a, (-1,))
    b = xp.reshape(b, (-1,))
    return xp.reshape(
        _helpers.in1d(a, b, assume_unique=assume_unique, invert=invert, xp=xp),
        original_a_shape,
    )


def nunique(x: Array, /, *, xp: ArrayNamespace) -> Array:  # numpydoc ignore=PR01,RT01
    # [FUNCTION DOC]
    # HELPER: Number of unique elements as a 0-D array. If the backend supports data-
    #   dependent shapes it uses unique_counts and returns the number of counts (or the
    #   sum of ones when the size is unknown, e.g. Dask). Otherwise it flattens, sorts,
    #   marks positions differing from the next element (cyclically) and returns the
    #   maximum of that count and 1-if-nonempty, covering the empty and all-equal cases.
    # KNOWN ISSUE: in the sort-based path every NaN differs from itself so each NaN is
    #   counted separately, while the unique_counts path may collapse NaNs depending on
    #   the backend.
    """See docstring in `array_api_extra._set`."""
    # There are 3 general use cases:
    # 1. backend has unique_counts and it returns an array with known shape
    # 2. backend has unique_counts and it returns a None-sized array;
    #    e.g. Dask, ndonnx
    # 3. backend does not have unique_counts; e.g. wrapped JAX
    if _helpers.capabilities(xp, x)["data-dependent shapes"]:
        # xp has unique_counts; O(n) complexity
        _, counts = xp.unique_counts(x)
        n = _compat.size(counts)
        if n is None:
            return xp.sum(xp.ones_like(counts))
        return xp.asarray(n, device=_compat.device(x))

    # xp does not have unique_counts; O(n*logn) complexity
    x = xp.reshape(x, (-1,))
    x = xp.sort(x, stable=False)
    mask = x != xp.roll(x, -1)
    default_int = _inspection.default_dtype(xp, "integral", device=_compat.device(x))
    return xp.maximum(
        # Special cases:
        # - array is size 0
        # - array has all elements equal to each other
        xp.astype(xp.any(~mask), default_int),
        xp.sum(xp.astype(mask, default_int)),
    )


def setdiff1d(
    x1: Array | complex,
    x2: Array | complex,
    /,
    *,
    assume_unique: bool = False,
    xp: ArrayNamespace,
) -> Array:  # numpydoc ignore=PR01,RT01
    # [FUNCTION DOC]
    # HELPER: Unique values of x1 not present in x2. Converts inputs to arrays; with
    #   assume_unique only flattens them, otherwise deduplicates with unique_values
    #   (sorted). Returns x1 values masked by inverted in1d. Requires boolean indexing
    #   (data-dependent shape).
    """See docstring in `array_api_extra._set`."""

    # https://github.com/microsoft/pyright/issues/10103
    x1_, x2_ = _helpers.asarrays(x1, x2, xp=xp)

    if assume_unique:
        x1_ = xp.reshape(x1_, (-1,))
        x2_ = xp.reshape(x2_, (-1,))
    else:
        x1_ = xp.unique_values(x1_)
        x2_ = xp.unique_values(x2_)

    return x1_[_helpers.in1d(x1_, x2_, assume_unique=True, invert=True, xp=xp)]


def union1d(a: Array, b: Array, /, *, xp: ArrayNamespace) -> Array:
    # [FUNCTION DOC]
    # HELPER: Sorted unique values of the concatenation of flattened a and b, wrapped in
    #   xp.asarray because sparse returns NumPy arrays.
    # numpydoc ignore=PR01,RT01
    """See docstring in `array_api_extra._set`."""
    a = xp.reshape(a, (-1,))
    b = xp.reshape(b, (-1,))
    # XXX: `sparse` returns NumPy arrays from `unique_values`
    return xp.asarray(xp.unique_values(xp.concat([a, b])))
