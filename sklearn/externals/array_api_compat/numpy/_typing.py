# [MODULE DOC]
# NumPy-specific typing aliases: Device is the literal "cpu"; DType and Array are
# numpy.dtype and numpy.ndarray at runtime (parametrized versions under TYPE_CHECKING).
# Imports numpy; no other side effects.
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, TypeAlias

import numpy as np

Device: TypeAlias = Literal["cpu"]

if TYPE_CHECKING:

    # NumPy 1.x on Python 3.10 fails to parse np.dtype[]
    DType: TypeAlias = np.dtype[
        np.bool_
        | np.integer[Any]
        | np.float32
        | np.float64
        | np.complex64
        | np.complex128
    ]
    Array: TypeAlias = np.ndarray[Any, DType]
else:
    DType: TypeAlias = np.dtype
    Array: TypeAlias = np.ndarray

__all__ = ["Array", "DType", "Device"]


def __dir__() -> list[str]:
    # [FUNCTION DOC]
    # HELPER: Module __dir__ hook returning __all__.
    return __all__
