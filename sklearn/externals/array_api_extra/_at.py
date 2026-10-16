# [MODULE DOC]
# Implements xpx.at, a backend-agnostic emulation of JAX's x.at[idx].op(y) functional
#   update API: in place on writeable arrays, via copies on read-only ones, via JAX
#   .at[] on JAX, and via where() for boolean masks on Dask/jitted JAX.
"""Update operations for read-only arrays."""

from __future__ import annotations

import operator
import typing
from collections.abc import Callable
from enum import Enum
from typing import ClassVar

from ._lib import _compat, _helpers
from ._lib._typing import Array, ArrayNamespace, SetIndex

__all__ = ["at"]


class _AtOp(Enum):
    # [CLASS DOC]
    # HELPER: Enum of supported update operations; values match the JAX .at[] method
    #   names (set, add, subtract, multiply, divide, power, min, max).
    """Operations for use in `xpx.at`."""

    SET = "set"
    ADD = "add"
    SUBTRACT = "subtract"
    MULTIPLY = "multiply"
    DIVIDE = "divide"
    POWER = "power"
    MIN = "min"
    MAX = "max"

    # @override from Python 3.12
    def __str__(self) -> str:  # pyright: ignore[reportImplicitOverride]
        # [FUNCTION DOC]
        # HELPER: Returns the operation's value string (readable pytest ids).
        """
        Return string representation (useful for pytest logs).

        Returns
        -------
        str
            The operation's name.
        """
        return self.value


class _Undef(Enum):
    # [CLASS DOC]
    # HELPER: Single-member enum used as a sentinel meaning "index not given" (distinct
    #   from None, which is a valid index).
    """Sentinel for undefined values."""

    UNDEF = 0


_undef = _Undef.UNDEF


class at:  # pylint: disable=invalid-name  # numpydoc ignore=PR02
    # [CLASS DOC]
    # Lightweight holder of an array and an optional index (slots _x, _idx). Indexing it
    #   with [] fixes the index; the op methods (set, add, subtract, multiply, divide,
    #   power, min, max) perform the update according to the copy policy and return the
    #   resulting array.
    """
    Update operations for read-only arrays.

    This implements ``jax.numpy.ndarray.at`` for all writeable
    backends (those that support ``__setitem__``) and routes
    to the ``.at[]`` method for JAX arrays.

    Parameters
    ----------
    x : array
        Input array.
    idx : index, optional
        Only `array API standard compliant indices
        <https://data-apis.org/array-api/latest/API_specification/indexing.html>`_
        are supported.

        You may use two alternate syntaxes::

          import array_api_extra as xpx
          xpx.at(x, idx).set(value)  # or add(value), etc.
          xpx.at(x)[idx].set(value)

    copy : bool, optional
        None (default)
            The array parameter *may* be modified in place if it is
            possible and beneficial for performance.
            You should not reuse it after calling this function.
        True
            Ensure that the inputs are not modified.
        False
            Ensure that the update operation writes back to the input.
            Raise ``ValueError`` if a copy cannot be avoided.

    xp : array_namespace, optional
        The standard-compatible namespace for `x`. Default: infer.

    Returns
    -------
    Updated input array.

    Warnings
    --------
    (a) When you omit the ``copy`` parameter, you should never reuse the parameter
    array later on; ideally, you should reassign it immediately::

        import array_api_extra as xpx
        x = xpx.at(x, 0).set(2)

    The above best practice pattern ensures that the behaviour won't change depending
    on whether ``x`` is writeable or not, as the original ``x`` object is dereferenced
    as soon as ``xpx.at`` returns; this way there is no risk to accidentally update it
    twice.

    On the reverse, the anti-pattern below must be avoided, as it will result in
    different behaviour on read-only versus writeable arrays::

        x = xp.asarray([0, 0, 0])
        y = xpx.at(x, 0).set(2)
        z = xpx.at(x, 1).set(3)

    In the above example, both calls to ``xpx.at`` update ``x`` in place *if possible*.
    This causes the behaviour to diverge depending on whether ``x`` is writeable or not:

    - If ``x`` is writeable, then after the snippet above you'll have
      ``x == y == z == [2, 3, 0]``
    - If ``x`` is read-only, then you'll end up with
      ``x == [0, 0, 0]``, ``y == [2, 0, 0]`` and ``z == [0, 3, 0]``.

    The correct pattern to use if you want diverging outputs from the same input is
    to enforce copies::

        x = xp.asarray([0, 0, 0])
        y = xpx.at(x, 0).set(2, copy=True)  # Never updates x
        z = xpx.at(x, 1).set(3)  # May or may not update x in place
        del x  # avoid accidental reuse of x as we don't know its state anymore

    (b) The array API standard does not support integer array indices.
    The behaviour of update methods when the index is an array of integers is
    undefined and will vary between backends; this is particularly true when the
    index contains multiple occurrences of the same index, e.g.::

        import numpy as np
        import jax.numpy as jnp
        import array_api_extra as xpx
        xpx.at(np.asarray([123]), np.asarray([0, 0])).add(1)
        array([124])
        xpx.at(jnp.asarray([123]), jnp.asarray([0, 0])).add(1)
        Array([125], dtype=int32)

    See Also
    --------
    jax.numpy.ndarray.at : Equivalent array method in JAX.

    Notes
    -----
    `sparse <https://sparse.pydata.org/>`_, as well as read-only arrays from libraries
    not explicitly covered by ``array-api-compat``, are not supported by update
    methods.

    Boolean masks are supported on Dask and jitted JAX arrays exclusively
    when `idx` has the same shape as `x` and `y` is 0-dimensional.
    Note that this support is not available in JAX's native
    ``x.at[mask].set(y)``.

    This pattern::

        mask = m(x)
        x[mask] = f(x[mask])

    Can't be replaced by `at`, as it won't work on Dask and JAX inside jax.jit::

        mask = m(x)
        x = xpx.at(x, mask).set(f(x[mask]))  # Crash on Dask and jax.jit

    You should instead use::

        x = xp.where(m(x), f(x), x)

    Examples
    --------
    Given either of these equivalent expressions::

        import array_api_extra as xpx
        x = xpx.at(x)[1].add(2)
        x = xpx.at(x, 1).add(2)

    If x is a JAX array, they are the same as::

        x = x.at[1].add(2)

    If x is a read-only NumPy array, they are the same as::

        x = x.copy()
        x[1] += 2

    For other known backends, they are the same as::

        x[1] += 2
    """

    _x: Array
    _idx: SetIndex | _Undef
    __slots__: ClassVar[tuple[str, ...]] = ("_idx", "_x")

    def __init__(
        self, x: Array, idx: SetIndex | _Undef = _undef, /
    ) -> None:  # numpydoc ignore=GL08
        # [FUNCTION DOC]
        # Stores x and idx (default: the _undef sentinel). No validation or copying.
        self._x = x
        self._idx = idx

    def __getitem__(self, idx: SetIndex, /) -> typing.Self:  # numpydoc ignore=PR01,RT01
        # [FUNCTION DOC]
        # Returns a new at object with the given index, supporting at(x)[idx] syntax.
        # Exceptions: ValueError if this object already has an index.
        """
        Allow for the alternate syntax ``at(x)[start:stop:step]``.

        It looks prettier than ``at(x, slice(start, stop, step))``
        and feels more intuitive coming from the JAX documentation.
        """
        if self._idx is not _undef:
            msg = "Index has already been set"
            raise ValueError(msg)
        return type(self)(self._x, idx)

    def _op(
        self,
        at_op: _AtOp,
        in_place_op: Callable[[Array, Array | complex], Array] | None,
        out_of_place_op: Callable[[Array, Array], Array] | None,
        y: Array | complex,
        /,
        copy: bool | None,
        xp: ArrayNamespace | None,
    ) -> Array:
        # [FUNCTION DOC]
        # HELPER: Shared implementation of every update. Steps: infer xp from x and y if
        #   not given; require an index and copy in {True, False, None}. Special case:
        #   if idx is a Dask or JAX boolean array with x's shape and y converts (to x's
        #   dtype/device) to a 0-D array, compute the result with
        #   apply_where(out_of_place_op, fill_value=x) cast back to x's dtype, or
        #   where(idx, y, x) for set; with copy False write it back into x with x[()]
        #   and return x. Otherwise, if copy is True or (None and x is read-only): for
        #   JAX call x.at[idx].<op>(y) and cast to x's dtype; else copy x with
        #   asarray(copy=True) and continue. Then require x writeable, cast torch y to
        #   x's dtype, and do x[idx] = in_place_op(x[idx], y) or x[idx] = y.
        # Outputs: the updated array (x itself when updated in place).
        # Exceptions: ValueError for missing index, invalid copy, or a read-only array
        #   that cannot be updated (including copy False on read-only input); backend
        #   errors for unsupported indices (e.g. boolean masks without boolean-indexing
        #   support).
        # Side effects: mutates x in place when it is writeable and copy is not True.
        """
        Implement all update operations.

        Parameters
        ----------
        at_op : _AtOp
            Method of JAX's Array.at[].
        in_place_op : Callable[[Array, Array | complex], Array] | None
            In-place operation to apply on mutable backends::

                x[idx] = in_place_op(x[idx], y)

            If None::

                x[idx] = y

        out_of_place_op : Callable[[Array, Array], Array] | None
            Out-of-place operation to apply when idx is a boolean mask and the backend
            doesn't support in-place updates::

                x = xp.where(idx, out_of_place_op(x, y), x)

            If None::

                x = xp.where(idx, y, x)

        y : array or complex
            Right-hand side of the operation.
        copy : bool or None
            Whether to copy the input array. See the class docstring for details.
        xp : array_namespace, optional
            The array namespace for the input array. Default: infer.

        Returns
        -------
        Array
            Updated `x`.
        """
        from ._agnostic._elementwise import apply_where  # pylint: disable=cyclic-import

        x, idx = self._x, self._idx
        xp = _compat.array_namespace(x, y) if xp is None else xp

        if isinstance(idx, _Undef):
            msg = (
                "Index has not been set.\n"
                "Usage: either\n"
                "    at(x, idx).set(value)\n"
                "or\n"
                "    at(x)[idx].set(value)\n"
                "(same for all other methods)."
            )
            raise ValueError(msg)  # noqa: TRY004

        if copy not in (True, False, None):
            msg = f"copy must be True, False, or None; got {copy!r}"
            raise ValueError(msg)

        writeable = None if copy else _compat.is_writeable_array(x)

        # JAX inside jax.jit doesn't support in-place updates with boolean
        # masks; Dask exclusively supports __setitem__ but not iops.
        # We can handle the common special case of 0-dimensional y
        # with where(idx, y, x) instead.
        if (
            (_compat.is_dask_array(idx) or _compat.is_jax_array(idx))
            and idx.dtype == xp.bool
            and idx.shape == x.shape
        ):
            y_xp = xp.asarray(y, dtype=x.dtype, device=_compat.device(x))
            if y_xp.ndim == 0:
                if out_of_place_op:  # add(), subtract(), ...
                    # suppress inf warnings on Dask
                    out = apply_where(
                        idx, (x, y_xp), out_of_place_op, fill_value=x, xp=xp
                    )
                    # Undo int->float promotion on JAX after _AtOp.DIVIDE
                    out = xp.astype(out, x.dtype, copy=False)
                else:  # set()
                    out = xp.where(idx, y_xp, x)

                if copy is False:
                    x[()] = out
                    return x
                return out

            # else: this will work on eager JAX and crash on jax.jit and Dask

        if copy or (copy is None and not writeable):
            if _compat.is_jax_array(x):
                # Use JAX's at[]
                func = typing.cast(
                    Callable[[Array | complex], Array],
                    getattr(x.at[idx], at_op.value),  # type: ignore[attr-defined]  # pyright: ignore[reportAttributeAccessIssue,reportUnknownArgumentType]
                )
                out = func(y)
                # Undo int->float promotion on JAX after _AtOp.DIVIDE
                return xp.astype(out, x.dtype, copy=False)

            # Emulate at[] behaviour for non-JAX arrays
            # with a copy followed by an update

            x = xp.asarray(x, copy=True)
            # A copy of a read-only numpy array is writeable
            # Note: this assumes that a copy of a writeable array is writeable
            assert not writeable
            writeable = None

        if writeable is None:
            writeable = _compat.is_writeable_array(x)
        if not writeable:
            # sparse crashes here
            msg = f"Can't update read-only array {x}"
            raise ValueError(msg)

        # Work around bug in PyTorch where __setitem__ doesn't
        # always support mismatched dtypes
        # https://github.com/pytorch/pytorch/issues/150017
        if _compat.is_torch_array(y):
            y = xp.astype(y, x.dtype, copy=False)

        # Backends without boolean indexing (other than JAX) crash here
        if in_place_op:  # add(), subtract(), ...
            x[idx] = in_place_op(x[idx], y)
        else:  # set()
            x[idx] = y
        return x

    def set(
        self,
        y: Array | complex,
        /,
        copy: bool | None = None,
        xp: ArrayNamespace | None = None,
    ) -> Array:  # numpydoc ignore=PR01,RT01
        # [FUNCTION DOC]
        # Applies x[idx] = y via _op (no in-place or out-of-place operator).
        """Apply ``x[idx] = y`` and return the update array."""
        return self._op(_AtOp.SET, None, None, y, copy=copy, xp=xp)

    def add(
        self,
        y: Array | complex,
        /,
        copy: bool | None = None,
        xp: ArrayNamespace | None = None,
    ) -> Array:  # numpydoc ignore=PR01,RT01
        # [FUNCTION DOC]
        # Applies x[idx] += y via _op using operator.iadd (in place) / operator.add
        #   (mask path); iadd surfaces dtype incompatibilities.
        """Apply ``x[idx] += y`` and return the updated array."""

        # Note for this and all other methods based on _iop:
        # operator.iadd and operator.add subtly differ in behaviour, as
        # only iadd will trigger exceptions when y has an incompatible dtype.
        return self._op(_AtOp.ADD, operator.iadd, operator.add, y, copy=copy, xp=xp)  # pyright: ignore[reportUnknownArgumentType]

    def subtract(
        self,
        y: Array | complex,
        /,
        copy: bool | None = None,
        xp: ArrayNamespace | None = None,
    ) -> Array:  # numpydoc ignore=PR01,RT01
        # [FUNCTION DOC]
        # Applies x[idx] -= y via _op with operator.isub / operator.sub.
        """Apply ``x[idx] -= y`` and return the updated array."""
        return self._op(
            _AtOp.SUBTRACT,
            operator.isub,  # pyright: ignore[reportUnknownArgumentType]
            operator.sub,
            y,
            copy=copy,
            xp=xp,
        )

    def multiply(
        self,
        y: Array | complex,
        /,
        copy: bool | None = None,
        xp: ArrayNamespace | None = None,
    ) -> Array:  # numpydoc ignore=PR01,RT01
        # [FUNCTION DOC]
        # Applies x[idx] *= y via _op with operator.imul / operator.mul.
        """Apply ``x[idx] *= y`` and return the updated array."""
        return self._op(
            _AtOp.MULTIPLY,
            operator.imul,  # pyright: ignore[reportUnknownArgumentType]
            operator.mul,
            y,
            copy=copy,
            xp=xp,
        )

    def divide(
        self,
        y: Array | complex,
        /,
        copy: bool | None = None,
        xp: ArrayNamespace | None = None,
    ) -> Array:  # numpydoc ignore=PR01,RT01
        # [FUNCTION DOC]
        # Applies x[idx] /= y via _op with operator.itruediv / operator.truediv; JAX
        #   results are cast back to x's dtype.
        """Apply ``x[idx] /= y`` and return the updated array."""
        return self._op(
            _AtOp.DIVIDE,
            operator.itruediv,  # pyright: ignore[reportUnknownArgumentType]
            operator.truediv,  # pyright: ignore[reportUnknownArgumentType]
            y,
            copy=copy,
            xp=xp,
        )

    def power(
        self,
        y: Array | complex,
        /,
        copy: bool | None = None,
        xp: ArrayNamespace | None = None,
    ) -> Array:  # numpydoc ignore=PR01,RT01
        # [FUNCTION DOC]
        # Applies x[idx] **= y via _op with operator.ipow / operator.pow.
        """Apply ``x[idx] **= y`` and return the updated array."""
        return self._op(_AtOp.POWER, operator.ipow, operator.pow, y, copy=copy, xp=xp)  # pyright: ignore[reportUnknownArgumentType]

    def min(
        self,
        y: Array | complex,
        /,
        copy: bool | None = None,
        xp: ArrayNamespace | None = None,
    ) -> Array:  # numpydoc ignore=PR01,RT01
        # [FUNCTION DOC]
        # Applies x[idx] = minimum(x[idx], y). Infers xp from x, gets the meta namespace
        #   (the chunk library under Dask), converts y to an array and calls _op with
        #   mxp.minimum for both in-place and out-of-place forms.
        # KNOWN ISSUE: y is converted with asarray without device or dtype, so it may be
        #   created on the default device rather than x's device.
        """Apply ``x[idx] = minimum(x[idx], y)`` and return the updated array."""
        # On Dask, this function runs on the chunks, so we need to determine the
        # namespace that Dask is wrapping.
        # Note that da.minimum _incidentally_ works on NumPy, CuPy, and sparse
        # thanks to all these meta-namespaces implementing the __array_ufunc__
        # interface, but there's no guarantee that it will work for other
        # wrapped libraries in the future.
        xp = _compat.array_namespace(self._x) if xp is None else xp
        mxp = _helpers.meta_namespace(self._x, xp=xp)
        y = xp.asarray(y)
        return self._op(_AtOp.MIN, mxp.minimum, mxp.minimum, y, copy=copy, xp=xp)

    def max(
        self,
        y: Array | complex,
        /,
        copy: bool | None = None,
        xp: ArrayNamespace | None = None,
    ) -> Array:  # numpydoc ignore=PR01,RT01
        # [FUNCTION DOC]
        # Applies x[idx] = maximum(x[idx], y); same mechanism as min with mxp.maximum.
        # KNOWN ISSUE: same device concern as min for the asarray conversion of y.
        """Apply ``x[idx] = maximum(x[idx], y)`` and return the updated array."""
        # See note on min()
        xp = _compat.array_namespace(self._x) if xp is None else xp
        mxp = _helpers.meta_namespace(self._x, xp=xp)
        y = xp.asarray(y)
        return self._op(_AtOp.MAX, mxp.maximum, mxp.maximum, y, copy=copy, xp=xp)
