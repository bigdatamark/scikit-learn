# [MODULE DOC]
# HELPER: Generic array API implementations of create_diagonal and one_hot used by the
#   dispatching wrappers in array_api_extra._creation.
"""Array-agnostic implementations for creation functions."""

from .._at import at
from .._lib import _compat, _helpers
from .._lib._typing import Array, ArrayNamespace

__all__ = ["create_diagonal", "one_hot"]


def create_diagonal(
    x: Array, /, *, offset: int = 0, xp: ArrayNamespace
) -> Array:  # numpydoc ignore=PR01,RT01
    # [FUNCTION DOC]
    # HELPER: Builds batched square matrices with the last axis of x on diagonal offset.
    #   Size n is the last dim plus |offset|. Allocates zeros of shape batch + (n*n,),
    #   then for every batch index writes x's row into a strided slice of the flattened
    #   matrix (step n+1, starting at offset for non-negative offsets or |offset|*n for
    #   negative ones, stopping at n*(n-offset) capped to n*n) using at(...).set, and
    #   finally reshapes to batch + (n, n).
    # Inputs: x - array of ndim >= 1 with eager (known) shape; offset - int; xp -
    #   namespace.
    # Outputs: array of x.dtype on x's device. Exceptions: from eager_shape if the shape
    #   is unknown.
    """See docstring in `array_api_extra._creation`."""
    x_shape = _helpers.eager_shape(x)
    batch_dims = x_shape[:-1]
    n = x_shape[-1] + abs(offset)
    diag = xp.zeros((*batch_dims, n**2), dtype=x.dtype, device=_compat.device(x))

    target_slice = slice(
        offset if offset >= 0 else abs(offset) * n,
        min(n * (n - offset), diag.shape[-1]),
        n + 1,
    )
    for index in _helpers.ndindex(*batch_dims):
        diag = at(diag)[(*index, target_slice)].set(x[(*index, slice(None))])
    return xp.reshape(diag, (*batch_dims, n, n))


def one_hot(
    x: Array,
    /,
    num_classes: int,
    *,
    xp: ArrayNamespace,
) -> Array:  # numpydoc ignore=PR01,RT01
    # [FUNCTION DOC]
    # HELPER: One-hot encoding by broadcasting: compares x with an extra trailing axis
    #   against arange(num_classes) of x's dtype/device.
    # Inputs: x - integer array; num_classes - int; xp - namespace.
    # Outputs: boolean array of shape x.shape + (num_classes,); out-of-range values give
    #   all-False rows.
    """See docstring in `array_api_extra._creation`."""
    # TODO: Benchmark whether this is faster on the NumPy backend:
    # if is_numpy_array(x):
    #     out = xp.zeros((x.size, num_classes), dtype=dtype)
    #     out[xp.arange(x.size), xp.reshape(x, (-1,))] = 1
    #     return xp.reshape(out, (*x.shape, num_classes))
    range_num_classes = xp.arange(num_classes, dtype=x.dtype, device=_compat.device(x))
    return x[..., xp.newaxis] == range_num_classes
