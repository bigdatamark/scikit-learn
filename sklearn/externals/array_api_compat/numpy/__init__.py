# [PACKAGE DOC]
# Array-API-compatible NumPy namespace. Imports numpy.typing (so it is cloned), copies
# numpy's public names into this package with clone_module, then overrides them with
# the wrappers from ._aliases, adds __array_namespace_info__, the linalg and fft
# submodules (imported eagerly via __import__ so they replace numpy's own linalg/fft
# attributes), the linalg versions of matrix_transpose and vecdot, and
# __array_api_version__ "2025.12". __all__ is the sorted union of these names.
# Side effects at import: imports numpy, numpy.typing and the submodules.
from typing import Final

from .._internal import clone_module

# This needs to be loaded explicitly before cloning
import numpy.typing  # noqa: F401

__all__ = clone_module("numpy", globals())

# These imports may overwrite names from the import * above.
from . import _aliases
from ._aliases import *  # type: ignore[assignment,no-redef] # noqa: F403
from ._info import __array_namespace_info__  # noqa: F401

# Don't know why, but we have to do an absolute import to import linalg. If we
# instead do
#
# from . import linalg
#
# It doesn't overwrite np.linalg from above. The import is generated
# dynamically so that the library can be vendored.
__import__(__spec__.parent + ".linalg")

__import__(__spec__.parent + ".fft")

from .linalg import matrix_transpose, vecdot  # type: ignore[no-redef]  # noqa: F401

__array_api_version__: Final = "2025.12"

__all__ = sorted(
    set(__all__) 
    | set(_aliases.__all__) 
    | {"__array_api_version__", "__array_namespace_info__", "linalg", "fft"}
)

def __dir__() -> list[str]:
    # [FUNCTION DOC]
    # HELPER: Package __dir__ hook returning __all__.
    return __all__
