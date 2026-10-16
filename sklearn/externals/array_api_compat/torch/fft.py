# [MODULE DOC]
# Array-API compatible torch.fft namespace. Re-exports everything from torch.fft via
#   clone_module and overrides the n-dimensional transforms and shift helpers, whose
#   torch versions name the axes argument dim instead of axes.
# Side effects: importing torch and torch.fft; __all__ is built from the cloned module
#   plus the overridden names.
from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import torch  # noqa: F401
import torch.fft

from ._typing import Array
from .._internal import clone_module

__all__ = clone_module("torch.fft", globals())

# Several torch fft functions do not map axes to dim

def fftn(
    x: Array,
    /,
    *,
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = None,
    norm: Literal["backward", "ortho", "forward"] = "backward",
    **kwargs: object,
) -> Array:
    # [FUNCTION DOC]
    # N-dimensional discrete Fourier transform; forwards to torch.fft.fftn renaming axes
    #   to dim.
    # Inputs: x - tensor; s - optional output sizes per transformed axis; axes -
    #   optional axes (None means all, or the last len(s) axes); norm -
    #   "backward"/"ortho"/"forward"; kwargs passed through to torch.
    # Outputs: complex tensor. Exceptions: whatever torch raises.
    return torch.fft.fftn(x, s=s, dim=axes, norm=norm, **kwargs)

def ifftn(
    x: Array,
    /,
    *,
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = None,
    norm: Literal["backward", "ortho", "forward"] = "backward",
    **kwargs: object,
) -> Array:
    # [FUNCTION DOC]
    # Inverse n-dimensional FFT; forwards to torch.fft.ifftn with axes mapped to dim.
    #   Same inputs/outputs contract as fftn.
    return torch.fft.ifftn(x, s=s, dim=axes, norm=norm, **kwargs)

def rfftn(
    x: Array,
    /,
    *,
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = None,
    norm: Literal["backward", "ortho", "forward"] = "backward",
    **kwargs: object,
) -> Array:
    # [FUNCTION DOC]
    # N-dimensional FFT of real input; forwards to torch.fft.rfftn with axes mapped to
    #   dim. Last transformed axis of the output has size s[-1]//2+1. Same inputs as
    #   fftn.
    return torch.fft.rfftn(x, s=s, dim=axes, norm=norm, **kwargs)

def irfftn(
    x: Array,
    /,
    *,
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = None,
    norm: Literal["backward", "ortho", "forward"] = "backward",
    **kwargs: object,
) -> Array:
    # [FUNCTION DOC]
    # Inverse of rfftn; forwards to torch.fft.irfftn with axes mapped to dim. Returns a
    #   real tensor. Same inputs as fftn.
    return torch.fft.irfftn(x, s=s, dim=axes, norm=norm, **kwargs)

def fftshift(
    x: Array,
    /,
    *,
    axes: int | Sequence[int] | None = None,
    **kwargs: object,
) -> Array:
    # [FUNCTION DOC]
    # Shifts the zero-frequency component to the center along axes (int, sequence, or
    #   None for all) by forwarding to torch.fft.fftshift with dim=axes. Returns a new
    #   tensor.
    return torch.fft.fftshift(x, dim=axes, **kwargs)

def ifftshift(
    x: Array,
    /,
    *,
    axes: int | Sequence[int] | None = None,
    **kwargs: object,
) -> Array:
    # [FUNCTION DOC]
    # Inverse of fftshift; forwards to torch.fft.ifftshift with dim=axes. Returns a new
    #   tensor.
    return torch.fft.ifftshift(x, dim=axes, **kwargs)


__all__ += ["fftn", "ifftn", "rfftn", "irfftn", "fftshift", "ifftshift"]

def __dir__() -> list[str]:
    # [FUNCTION DOC]
    # HELPER: Module-level dir() hook returning __all__ so introspection lists only
    #   public names.
    return __all__
