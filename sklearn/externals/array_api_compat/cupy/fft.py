# [MODULE DOC]
# Array-API-compatible cupy.fft namespace: star-imports cupy.fft, computes the list of
# its public names by exec-ing a star import into a scratch dict (cupy.fft has no
# __all__), then overrides the transform functions with common._fft wrappers bound to
# cupy. __all__ is that list plus common._fft.__all__ (may contain duplicates).
# Side effects: imports cupy; uses exec at import time.
from cupy.fft import *  # noqa: F403

# cupy.fft doesn't have __all__. If it is added, replace this with
#
# from cupy.fft import __all__ as linalg_all
_n: dict[str, object] = {}
exec("from cupy.fft import *", _n)
del _n["__builtins__"]
fft_all = list(_n)
del _n

from ..common import _fft
from .._internal import get_xp

import cupy as cp

fft = get_xp(cp)(_fft.fft)
ifft = get_xp(cp)(_fft.ifft)
fftn = get_xp(cp)(_fft.fftn)
ifftn = get_xp(cp)(_fft.ifftn)
rfft = get_xp(cp)(_fft.rfft)
irfft = get_xp(cp)(_fft.irfft)
rfftn = get_xp(cp)(_fft.rfftn)
irfftn = get_xp(cp)(_fft.irfftn)
hfft = get_xp(cp)(_fft.hfft)
ihfft = get_xp(cp)(_fft.ihfft)
fftfreq = get_xp(cp)(_fft.fftfreq)
rfftfreq = get_xp(cp)(_fft.rfftfreq)
fftshift = get_xp(cp)(_fft.fftshift)
ifftshift = get_xp(cp)(_fft.ifftshift)

__all__ = fft_all + _fft.__all__

def __dir__() -> list[str]:
    # [FUNCTION DOC]
    # HELPER: Module __dir__ hook returning __all__.
    return __all__

