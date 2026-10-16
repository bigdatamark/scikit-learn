# [MODULE DOC]
# Array-API-compatible numpy.fft namespace: clones numpy.fft's public names and
# overrides all transform, frequency and shift functions with the dtype-preserving
# common._fft wrappers bound to numpy. __all__ is the sorted union of both name sets.
import numpy as np

from .._internal import clone_module

__all__ = clone_module("numpy.fft", globals())

from .._internal import get_xp
from ..common import _fft

fft = get_xp(np)(_fft.fft)
ifft = get_xp(np)(_fft.ifft)
fftn = get_xp(np)(_fft.fftn)
ifftn = get_xp(np)(_fft.ifftn)
rfft = get_xp(np)(_fft.rfft)
irfft = get_xp(np)(_fft.irfft)
rfftn = get_xp(np)(_fft.rfftn)
irfftn = get_xp(np)(_fft.irfftn)
hfft = get_xp(np)(_fft.hfft)
ihfft = get_xp(np)(_fft.ihfft)
fftfreq = get_xp(np)(_fft.fftfreq)
rfftfreq = get_xp(np)(_fft.rfftfreq)
fftshift = get_xp(np)(_fft.fftshift)
ifftshift = get_xp(np)(_fft.ifftshift)


__all__ = sorted(set(__all__) | set(_fft.__all__))

def __dir__() -> list[str]:
    # [FUNCTION DOC]
    # HELPER: Module __dir__ hook returning __all__.
    return __all__

