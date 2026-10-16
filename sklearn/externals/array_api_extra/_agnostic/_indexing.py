# [MODULE DOC]
# HELPER: Generic array API implementations of diag_indices, tril_indices, triu_indices
#   and unravel_index used by the public wrappers in array_api_extra._indexing.
"""Array-agnostic implementations for indexing functions."""

from .._lib._typing import Array, ArrayNamespace, Device

__all__ = ["diag_indices", "tril_indices", "triu_indices", "unravel_index"]


def diag_indices(
    n: int, /, *, ndim: int, device: Device | None, xp: ArrayNamespace
) -> tuple[Array, ...]:  # numpydoc ignore=PR01,RT01
    # [FUNCTION DOC]
    # HELPER: Returns a tuple of ndim references to the same arange(n) array on device,
    #   indexing the main diagonal of an n^ndim array.
    """See docstring in `array_api_extra._indexing`."""
    idx = xp.arange(n, device=device)
    return (idx,) * ndim


def _tri_indices(
    n: int,
    *,
    offset: int,
    m: int | None,
    upper: bool,
    device: Device | None,
    xp: ArrayNamespace,
) -> tuple[Array, Array]:  # numpydoc ignore=PR01,RT01
    # [FUNCTION DOC]
    # HELPER: Builds an (n, cols) mask (cols = m or n) where column minus row is >=
    #   offset (upper) or <= offset (lower) and returns xp.nonzero of it.
    # Outputs: (row_indices, col_indices) in row-major order. Requires data-dependent
    #   output shapes.
    """Shared implementation for `tril_indices` and `triu_indices`."""
    cols = n if m is None else m
    rows = xp.arange(n, device=device)[:, xp.newaxis]
    cols_a = xp.arange(cols, device=device)[xp.newaxis, :]
    delta = cols_a - rows
    mask = delta >= offset if upper else delta <= offset
    r, c = xp.nonzero(mask)
    return (r, c)


def tril_indices(
    n: int,
    /,
    *,
    offset: int,
    m: int | None,
    device: Device | None,
    xp: ArrayNamespace,
) -> tuple[Array, Array]:  # numpydoc ignore=PR01,RT01
    # [FUNCTION DOC]
    # HELPER: Lower-triangle indices; calls _tri_indices with upper False.
    """See docstring in `array_api_extra._indexing`."""
    return _tri_indices(n, offset=offset, m=m, upper=False, device=device, xp=xp)


def triu_indices(
    n: int,
    /,
    *,
    offset: int,
    m: int | None,
    device: Device | None,
    xp: ArrayNamespace,
) -> tuple[Array, Array]:  # numpydoc ignore=PR01,RT01
    # [FUNCTION DOC]
    # HELPER: Upper-triangle indices; calls _tri_indices with upper True.
    """See docstring in `array_api_extra._indexing`."""
    return _tri_indices(n, offset=offset, m=m, upper=True, device=device, xp=xp)


def unravel_index(indices: Array, shape: tuple[int, ...], /) -> tuple[Array, ...]:
    # [FUNCTION DOC]
    # HELPER: Converts flat indices into coordinate arrays for a C-ordered shape:
    #   iterating dims from last to first, takes remainder by the dim as that coordinate
    #   and integer-divides indices by it.
    # Outputs: tuple of arrays (one per dim) shaped like indices. No bounds check: out-
    #   of-range indices wrap in the first coordinate.
    # numpydoc ignore=PR01,RT01
    """See docstring in `array_api_extra._indexing`."""
    coords: list[Array] = []
    for dim in reversed(shape):
        coords.append(indices % dim)
        indices = indices // dim
    return tuple(reversed(coords))
