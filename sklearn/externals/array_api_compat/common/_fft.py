# [MODULE DOC]
# Namespace-generic wrappers around a backend's fft submodule (NumPy, CuPy, Dask) that
# make the functions conform to the array API standard. The main fix is dtype
# preservation: NumPy-like fft functions upcast float32/complex64 inputs to
# complex128/float64, so each wrapper casts the result back down to the single
# precision type matching the input. Every function takes the backend namespace as
# the keyword-or-positional argument xp; per-library modules bind it via get_xp.
# Exports the names in __all__; no import-time side effects.
from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, TypeAlias

from ._typing import Array, Device, DType, Namespace

_Norm: TypeAlias = Literal["backward", "ortho", "forward"]

# Note: NumPy fft functions improperly upcast float32 and complex64 to
# complex128, which is why we require wrapping them all here.

def fft(
    x: Array,
    /,
    xp: Namespace,
    *,
    n: int | None = None,
    axis: int = -1,
    norm: _Norm = "backward",
) -> Array:
    # [FUNCTION DOC]
    # Computes the 1-D discrete Fourier transform by delegating to xp.fft.fft.
    # Inputs: x - array; xp - backend namespace; n - optional transform length (pads or
    # truncates the axis); axis - axis to transform, default -1; norm - "backward"
    # (default), "ortho" or "forward".
    # Outputs: complex array; cast to complex64 when x is float32 or complex64, otherwise
    # whatever the backend returned.
    # Exceptions: whatever the backend raises for invalid n/axis/norm.
    res = xp.fft.fft(x, n=n, axis=axis, norm=norm)
    if x.dtype in [xp.float32, xp.complex64]:
        return res.astype(xp.complex64)
    return res

def ifft(
    x: Array,
    /,
    xp: Namespace,
    *,
    n: int | None = None,
    axis: int = -1,
    norm: _Norm = "backward",
) -> Array:
    # [FUNCTION DOC]
    # Inverse 1-D DFT via xp.fft.ifft; same parameters and complex64 downcast rule as fft.
    # Outputs: complex array (complex64 for float32/complex64 inputs).
    res = xp.fft.ifft(x, n=n, axis=axis, norm=norm)
    if x.dtype in [xp.float32, xp.complex64]:
        return res.astype(xp.complex64)
    return res

def fftn(
    x: Array,
    /,
    xp: Namespace,
    *,
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = None,
    norm: _Norm = "backward",
) -> Array:
    # [FUNCTION DOC]
    # N-dimensional DFT via xp.fft.fftn.
    # Inputs: x - array; xp - namespace; s - optional per-axis output lengths; axes -
    # optional axes to transform (None means all, or the last len(s)); norm - as in fft.
    # Outputs: complex array, downcast to complex64 for float32/complex64 inputs.
    res = xp.fft.fftn(x, s=s, axes=axes, norm=norm)
    if x.dtype in [xp.float32, xp.complex64]:
        return res.astype(xp.complex64)
    return res

def ifftn(
    x: Array,
    /,
    xp: Namespace,
    *,
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = None,
    norm: _Norm = "backward",
) -> Array:
    # [FUNCTION DOC]
    # Inverse N-dimensional DFT via xp.fft.ifftn; parameters and downcast rule as fftn.
    res = xp.fft.ifftn(x, s=s, axes=axes, norm=norm)
    if x.dtype in [xp.float32, xp.complex64]:
        return res.astype(xp.complex64)
    return res

def rfft(
    x: Array,
    /,
    xp: Namespace,
    *,
    n: int | None = None,
    axis: int = -1,
    norm: _Norm = "backward",
) -> Array:
    # [FUNCTION DOC]
    # 1-D DFT of real input via xp.fft.rfft, returning the non-negative frequency half
    # (length n//2 + 1 along axis).
    # Inputs: as in fft.
    # Outputs: complex array; complex64 when x is float32 (only float32 is checked).
    res = xp.fft.rfft(x, n=n, axis=axis, norm=norm)
    if x.dtype == xp.float32:
        return res.astype(xp.complex64)
    return res

def irfft(
    x: Array,
    /,
    xp: Namespace,
    *,
    n: int | None = None,
    axis: int = -1,
    norm: _Norm = "backward",
) -> Array:
    # [FUNCTION DOC]
    # Inverse of rfft via xp.fft.irfft; produces a real array of length n (default
    # 2*(m-1) where m is the input length along axis).
    # Outputs: real array; cast to float32 when x is complex64.
    res = xp.fft.irfft(x, n=n, axis=axis, norm=norm)
    if x.dtype == xp.complex64:
        return res.astype(xp.float32)
    return res

def rfftn(
    x: Array,
    /,
    xp: Namespace,
    *,
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = None,
    norm: _Norm = "backward",
) -> Array:
    # [FUNCTION DOC]
    # N-dimensional real-input DFT via xp.fft.rfftn; parameters as fftn.
    # Outputs: complex array; complex64 when x is float32.
    res = xp.fft.rfftn(x, s=s, axes=axes, norm=norm)
    if x.dtype == xp.float32:
        return res.astype(xp.complex64)
    return res

def irfftn(
    x: Array,
    /,
    xp: Namespace,
    *,
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = None,
    norm: _Norm = "backward",
) -> Array:
    # [FUNCTION DOC]
    # Inverse of rfftn via xp.fft.irfftn; parameters as fftn.
    # Outputs: real array; float32 when x is complex64.
    res = xp.fft.irfftn(x, s=s, axes=axes, norm=norm)
    if x.dtype == xp.complex64:
        return res.astype(xp.float32)
    return res

def hfft(
    x: Array,
    /,
    xp: Namespace,
    *,
    n: int | None = None,
    axis: int = -1,
    norm: _Norm = "backward",
) -> Array:
    # [FUNCTION DOC]
    # DFT of a Hermitian-symmetric signal via xp.fft.hfft; parameters as fft.
    # Outputs: real array; float32 when x is float32 or complex64.
    res = xp.fft.hfft(x, n=n, axis=axis, norm=norm)
    if x.dtype in [xp.float32, xp.complex64]:
        return res.astype(xp.float32)
    return res

def ihfft(
    x: Array,
    /,
    xp: Namespace,
    *,
    n: int | None = None,
    axis: int = -1,
    norm: _Norm = "backward",
) -> Array:
    # [FUNCTION DOC]
    # Inverse of hfft via xp.fft.ihfft; parameters as fft.
    # Outputs: complex array; complex64 when x is float32 or complex64.
    res = xp.fft.ihfft(x, n=n, axis=axis, norm=norm)
    if x.dtype in [xp.float32, xp.complex64]:
        return res.astype(xp.complex64)
    return res

def fftfreq(
    n: int,
    /,
    xp: Namespace,
    *,
    d: float = 1.0,
    dtype: DType | None = None,
    device: Device | None = None,
) -> Array:
    # [FUNCTION DOC]
    # Returns the sample frequencies for a DFT of length n with sample spacing d, via
    # xp.fft.fftfreq.
    # Inputs: n - window length; xp - namespace; d - spacing, default 1.0; dtype -
    # optional dtype to cast the result to; device - must be None or "cpu".
    # Outputs: 1-D array of length n, cast to dtype when given.
    # Exceptions: ValueError when device is anything other than None or "cpu".
    # KNOWN ISSUE: the "cpu" device check is applied for every backend reusing this
    # helper (e.g. CuPy), so GPU device objects are always rejected.
    if device not in ["cpu", None]:
        raise ValueError(f"Unsupported device {device!r}")
    res = xp.fft.fftfreq(n, d=d)
    if dtype is not None:
        return res.astype(dtype)
    return res

def rfftfreq(
    n: int,
    /,
    xp: Namespace,
    *,
    d: float = 1.0,
    dtype: DType | None = None,
    device: Device | None = None,
) -> Array:
    # [FUNCTION DOC]
    # Like fftfreq but returns only the non-negative frequencies (length n//2 + 1) via
    # xp.fft.rfftfreq. Same inputs, device restriction and ValueError as fftfreq.
    if device not in ["cpu", None]:
        raise ValueError(f"Unsupported device {device!r}")
    res = xp.fft.rfftfreq(n, d=d)
    if dtype is not None:
        return res.astype(dtype)
    return res

def fftshift(
    x: Array, /, xp: Namespace, *, axes: int | Sequence[int] | None = None
) -> Array:
    # [FUNCTION DOC]
    # Shifts the zero-frequency component to the center of the spectrum by delegating to
    # xp.fft.fftshift.
    # Inputs: x - array; xp - namespace; axes - int, sequence of ints, or None (all axes).
    # Outputs: shifted array, same shape and dtype.
    return xp.fft.fftshift(x, axes=axes)

def ifftshift(
    x: Array, /, xp: Namespace, *, axes: int | Sequence[int] | None = None
) -> Array:
    # [FUNCTION DOC]
    # Inverse of fftshift, delegating to xp.fft.ifftshift. Same inputs and outputs.
    return xp.fft.ifftshift(x, axes=axes)

__all__ = [
    "fft",
    "ifft",
    "fftn",
    "ifftn",
    "rfft",
    "irfft",
    "rfftn",
    "irfftn",
    "hfft",
    "ihfft",
    "fftfreq",
    "rfftfreq",
    "fftshift",
    "ifftshift",
]

def __dir__() -> list[str]:
    # [FUNCTION DOC]
    # HELPER: Module-level __dir__ hook so dir(module) lists only the public names.
    # Outputs: the __all__ list itself (not a copy).
    return __all__
