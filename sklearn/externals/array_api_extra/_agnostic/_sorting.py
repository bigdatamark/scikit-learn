# [MODULE DOC]
# HELPER: Generic array API implementations of partition and argpartition, implemented
#   as full sorts.
"""Array-agnostic implementations for sorting functions."""

from .._lib._typing import Array, ArrayNamespace

__all__ = ["argpartition", "partition"]


def partition(  # numpydoc ignore=PR01,RT01
    x: Array,
    kth: int,  # noqa: ARG001
    /,
    axis: int = -1,
    *,
    xp: ArrayNamespace,
) -> Array:
    # [FUNCTION DOC]
    # HELPER: Returns xp.sort(x, axis, unstable); kth is ignored because a full sort
    #   satisfies the partition contract (at O(n log n) cost).
    """See docstring in `array_api_extra._sorting`."""
    return xp.sort(x, axis=axis, stable=False)


def argpartition(  # numpydoc ignore=PR01,RT01
    x: Array,
    kth: int,  # noqa: ARG001
    /,
    axis: int = -1,
    *,
    xp: ArrayNamespace,
) -> Array:
    # [FUNCTION DOC]
    # HELPER: Returns xp.argsort(x, axis, unstable); kth is ignored since full sort
    #   order satisfies the argpartition contract.
    """See docstring in `array_api_extra._sorting`."""
    return xp.argsort(x, axis=axis, stable=False)
