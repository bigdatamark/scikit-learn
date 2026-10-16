# [MODULE DOC]
# HELPER: Generic array API implementation of searchsorted via vectorised binary search.
"""Array-agnostic implementations for searching functions."""

import math
from typing import Literal

from .._lib import _compat
from .._lib._typing import Array, ArrayNamespace
from . import _inspection

__all__ = ["searchsorted"]


def searchsorted(
    x1: Array,
    x2: Array,
    /,
    *,
    side: Literal["left", "right"] = "left",
    xp: ArrayNamespace,
) -> Array:
    # [FUNCTION DOC]
    # HELPER: Finds insertion indices of x2 into the last axis of sorted x1 (batched,
    #   NaNs sorted last). Starts with lower bounds 0 and upper bounds equal to the
    #   number of non-NaN elements per row, then runs a fixed floor(log2(len))+1
    #   bisection steps (JIT friendly): take the midpoint, gather x1 there with
    #   take_along_axis, and move the upper or lower bound depending on x2 <= x0 (left)
    #   or x2 < x0 (right). Values that compare before the row minimum get 0; for side
    #   right NaN queries get the row length. Result cast to the default integral dtype.
    # Inputs: x1 - sorted array (..., N); x2 - queries with batch dims matching x1; side
    #   - 'left' or 'right'.
    # Outputs: integer array shaped like x2 (all zeros if N is 0).
    # numpydoc ignore=PR01,RT01
    """See docstring in `array_api_extra._searching`."""
    a = xp.full(x2.shape, 0, device=_compat.device(x1))

    if x1.shape[-1] == 0:
        return a

    n = xp.count_nonzero(~xp.isnan(x1), axis=-1, keepdims=True)
    b = xp.broadcast_to(n, x2.shape)

    compare = xp.less_equal if side == "left" else xp.less

    # while xp.any(b - a > 1):
    # refactored to for loop with ~log2(n) iterations for JAX JIT
    for _ in range(int(math.log2(x1.shape[-1])) + 1):  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
        c = (a + b) // 2
        x0 = xp.take_along_axis(x1, c, axis=-1)
        j = compare(x2, x0)
        b = xp.where(j, c, b)
        a = xp.where(j, a, c)

    out = xp.where(compare(x2, xp.min(x1, axis=-1, keepdims=True)), 0, b)
    out = xp.where(xp.isnan(x2), x1.shape[-1], out) if side == "right" else out
    return xp.astype(out, _inspection.default_dtype(xp, kind="integral"), copy=False)
