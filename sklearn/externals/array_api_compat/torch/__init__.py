# [PACKAGE DOC]
# Array-API-compatible PyTorch namespace. clone_module copies torch's public names into
# this package, then the wrappers from ._aliases override them and
# __array_namespace_info__ and __array_api_version__ ("2025.12") are added. __all__ is
# the sorted union of torch's names, the _aliases names and the version/info/linalg/fft
# names.
# Side effects at import: imports torch (ImportError if missing) and eagerly imports
# the .linalg and .fft submodules.
from typing import Final

from .._internal import clone_module

__all__ = clone_module("torch", globals())

# These imports may overwrite names from the import * above.
from . import _aliases
from ._aliases import * # noqa: F403
from ._info import __array_namespace_info__  # noqa: F401

# See the comment in the numpy __init__.py
__import__(__spec__.parent + '.linalg')
__import__(__spec__.parent + '.fft')

__array_api_version__: Final = '2025.12'

__all__ = sorted(
    set(__all__)
    | set(_aliases.__all__)
    | {"__array_api_version__", "__array_namespace_info__", "linalg", "fft"}
)

def __dir__() -> list[str]:
    # [FUNCTION DOC]
    # HELPER: Package __dir__ hook returning __all__.
    return __all__
