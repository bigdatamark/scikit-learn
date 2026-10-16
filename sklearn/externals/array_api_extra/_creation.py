# [MODULE DOC]
# Public delegation layer for creation functions (create_diagonal, one_hot): validates
#   inputs, dispatches to native backend implementations where available and otherwise
#   to the generic code in _agnostic._creation.
"""Delegation layer for creation functions."""

from . import _agnostic
from ._lib import _compat
from ._lib._typing import Array, ArrayNamespace, DType

__all__ = ["create_diagonal", "one_hot"]


def create_diagonal(
    x: Array, /, *, offset: int = 0, xp: ArrayNamespace | None = None
) -> Array:
    # [FUNCTION DOC]
    # Validates x is at least 1-D and infers xp. PyTorch uses torch.diag_embed on the
    #   last two axes; Dask, CuPy, NumPy and JAX use xp.diag for 1-D input; everything
    #   else (including batched input on those backends) uses the generic
    #   implementation.
    # Exceptions: ValueError for 0-D x.
    """
    Construct a diagonal array.

    Parameters
    ----------
    x : array
        An array having shape ``(*batch_dims, k)``.
    offset : int, optional
        Offset from the leading diagonal (default is ``0``).
        Use positive ints for diagonals above the leading diagonal,
        and negative ints for diagonals below the leading diagonal.
    xp : array_namespace, optional
        The standard-compatible namespace for `x`. Default: infer.

    Returns
    -------
    array
        An array having shape ``(*batch_dims, k+abs(offset), k+abs(offset))`` with `x`
        on the diagonal (offset by `offset`).

    Examples
    --------
    >>> import array_api_strict as xp
    >>> import array_api_extra as xpx
    >>> x = xp.asarray([2, 4, 8])

    >>> xpx.create_diagonal(x, xp=xp)
    Array([[2, 0, 0],
           [0, 4, 0],
           [0, 0, 8]], dtype=array_api_strict.int64)

    >>> xpx.create_diagonal(x, offset=-2, xp=xp)
    Array([[0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0],
           [2, 0, 0, 0, 0],
           [0, 4, 0, 0, 0],
           [0, 0, 8, 0, 0]], dtype=array_api_strict.int64)
    """
    if xp is None:
        xp = _compat.array_namespace(x)

    if x.ndim == 0:
        err_msg = "`x` must be at least 1-dimensional."
        raise ValueError(err_msg)

    if _compat.is_torch_namespace(xp):
        return xp.diag_embed(x, offset=offset, dim1=-2, dim2=-1)

    if (
        _compat.is_dask_namespace(xp)
        or _compat.is_cupy_namespace(xp)
        or _compat.is_numpy_namespace(xp)
        or _compat.is_jax_namespace(xp)
    ) and (x.ndim < 2):
        return xp.diag(x, k=offset)

    return _agnostic._creation.create_diagonal(x, offset=offset, xp=xp)


def one_hot(
    x: Array,
    /,
    num_classes: int,
    *,
    dtype: DType | None = None,
    axis: int = -1,
    xp: ArrayNamespace | None = None,
) -> Array:
    # [FUNCTION DOC]
    # Infers xp, requires an integral dtype, defaults dtype to the namespace's default
    #   real floating dtype on x's device. JAX delegates to jax.nn.one_hot (handling
    #   dtype and axis itself). PyTorch casts x to int64 and calls
    #   torch.nn.functional.one_hot, converting its RuntimeError (e.g. out-of-range
    #   index) to IndexError. Others use the generic broadcasting comparison. Non-JAX
    #   results are cast to dtype and the new last axis is moved to position axis when
    #   axis is not -1.
    # Exceptions: TypeError for non-integral x; IndexError on torch for invalid indices.
    """
    One-hot encode the given indices.

    Each index in the input `x` is encoded as a vector of zeros of length `num_classes`
    with the element at the given index set to one.

    Parameters
    ----------
    x : array
        An array with integral dtype whose values are between `0` and `num_classes - 1`.
    num_classes : int
        Number of classes in the one-hot dimension.
    dtype : DType, optional
        The dtype of the return value.  Defaults to the default float dtype (usually
        float64).
    axis : int, optional
        Position in the expanded axes where the new axis is placed. Default: -1.
    xp : array_namespace, optional
        The standard-compatible namespace for `x`. Default: infer.

    Returns
    -------
    array
        An array having the same shape as `x` except for a new axis at the position
        given by `axis` having size `num_classes`.  If `axis` is unspecified, it
        defaults to -1, which appends a new axis.

        If ``x < 0`` or ``x >= num_classes``, then the result is undefined, may raise
        an exception, or may even cause a bad state.  `x` is not checked.

    Examples
    --------
    >>> import array_api_extra as xpx
    >>> import array_api_strict as xp
    >>> xpx.one_hot(xp.asarray([1, 2, 0]), 3)
    Array([[0., 1., 0.],
          [0., 0., 1.],
          [1., 0., 0.]], dtype=array_api_strict.float64)
    """
    # Validate inputs.
    if xp is None:
        xp = _compat.array_namespace(x)
    if not xp.isdtype(x.dtype, "integral"):
        msg = "x must have an integral dtype."
        raise TypeError(msg)
    if dtype is None:
        dtype = _agnostic._inspection.default_dtype(xp, device=_compat.device(x))
    # Delegate where possible.
    if _compat.is_jax_namespace(xp):
        from jax.nn import one_hot as jax_one_hot

        return jax_one_hot(x, num_classes, dtype=dtype, axis=axis)
    if _compat.is_torch_namespace(xp):
        from torch.nn.functional import one_hot as torch_one_hot

        x = xp.astype(x, xp.int64)  # PyTorch only supports int64 here.
        try:
            out = torch_one_hot(x, num_classes)
        except RuntimeError as e:
            raise IndexError from e
    else:
        out = _agnostic._creation.one_hot(x, num_classes, xp=xp)
    out = xp.astype(out, dtype, copy=False)
    if axis != -1:
        out = xp.moveaxis(out, -1, axis)
    return out
