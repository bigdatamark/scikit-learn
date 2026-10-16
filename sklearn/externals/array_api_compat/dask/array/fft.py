# [MODULE DOC]
# Array-API-compatible dask.array.fft namespace: clones dask.array.fft's public names and
# overrides fftfreq and rfftfreq with common._fft wrappers bound to dask.array (adding
# dtype and device keywords; only "cpu"/None devices accepted). Imports dask at import.
from ..._internal import clone_module

__all__ = clone_module("dask.array.fft", globals())

from ...common import _fft
from ..._internal import get_xp

import dask.array as da

fftfreq = get_xp(da)(_fft.fftfreq)
rfftfreq = get_xp(da)(_fft.rfftfreq)

__all__ += ["fftfreq", "rfftfreq"]

def __dir__() -> list[str]:
    # [FUNCTION DOC]
    # HELPER: Module __dir__ hook returning __all__.
    return __all__
