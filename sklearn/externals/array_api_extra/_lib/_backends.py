# [MODULE DOC]
# HELPER: Test-suite support: the Backend enum of array libraries array-api-extra is
#   tested against, and NUMPY_VERSION (first three numeric components of
#   numpy.__version__). Imports numpy and pytest at import time.
# KNOWN ISSUE: NUMPY_VERSION parsing converts each dotted component with int, so pre-
#   release versions such as "2.1.0rc1" raise ValueError at import.
"""Backends against which array-api-extra runs its tests."""

from __future__ import annotations

from enum import Enum
from typing import Any

import numpy as np
import pytest

__all__ = ["NUMPY_VERSION", "Backend"]

NUMPY_VERSION = tuple(int(v) for v in np.__version__.split(".")[:3])


class Backend(Enum):  # numpydoc ignore=PR02
    # [CLASS DOC]
    # HELPER: Enum of tested backends; values are "<module>[:<tag>]", where the tag
    #   keeps variants (strictest, readonly, gpu) distinct from the base module.
    """
    All array library backends explicitly tested by array-api-extra.

    Parameters
    ----------
    value : str
        Tag of the backend's module, in the format ``<namespace>[:<extra tag>]``.
    """

    # Use :<tag> to prevent Enum from deduplicating items with the same value
    ARRAY_API_STRICT = "array_api_strict"
    ARRAY_API_STRICTEST = "array_api_strict:strictest"
    NUMPY = "numpy"
    NUMPY_READONLY = "numpy:readonly"
    CUPY = "cupy"
    TORCH = "torch"
    TORCH_GPU = "torch:gpu"
    DASK = "dask.array"
    SPARSE = "sparse"
    JAX = "jax.numpy"
    JAX_GPU = "jax.numpy:gpu"

    @property
    def modname(self) -> str:  # numpydoc ignore=RT01
        # [FUNCTION DOC]
        # HELPER: Module name to import: the value up to the first colon.
        """Module name to be imported."""
        return self.value.split(":")[0]

    def like(self, *others: Backend) -> bool:  # numpydoc ignore=PR01,RT01
        # [FUNCTION DOC]
        # HELPER: True if any of the given backends shares this backend's module name.
        """Check if this backend uses the same module as others."""
        return any(self.modname == other.modname for other in others)

    def pytest_param(self) -> Any:
        # [FUNCTION DOC]
        # HELPER: Builds a pytest.param for this backend with id derived from the lower-
        #   cased name (":gpu"/":readonly" suffixes), adding a skipif mark for
        #   array_api_strict on NumPy < 1.26 and a thread_unsafe mark for Dask and JAX
        #   (whose functions get monkey-patched).
        # Outputs: pytest ParameterSet.
        """
        Backend as a pytest parameter.

        Returns
        -------
        pytest.mark.ParameterSet
            The backend as a pytest parameter.
        """
        id_ = (
            self.name.lower().replace("_gpu", ":gpu").replace("_readonly", ":readonly")
        )

        marks = []
        if self.like(Backend.ARRAY_API_STRICT):
            marks.append(
                pytest.mark.skipif(
                    NUMPY_VERSION < (1, 26),
                    reason="array_api_strict is untested on NumPy <1.26",
                )
            )
        if self.like(Backend.DASK, Backend.JAX):
            # Monkey-patched by lazy_xp_function
            marks.append(pytest.mark.thread_unsafe)

        return pytest.param(self, id=id_, marks=marks)  # pyright: ignore[reportUnknownArgumentType]
