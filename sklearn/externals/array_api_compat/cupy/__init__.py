# [PACKAGE DOC]
# Array-API-compatible CuPy namespace. Re-exports everything from cupy (plus abs, max,
# min, round, which a star-import would not shadow builtins with), then overrides
# names with the wrappers in ._aliases and adds __array_namespace_info__ and
# __array_api_version__ ("2025.12"). __all__ is the sorted set of public globals minus
# helper module names, plus linalg and fft.
# Side effects at import: imports cupy (ImportError without it) and eagerly imports the
# .linalg and .fft submodules so they are attributes of the package.
from typing import Final
from cupy import * # noqa: F403

# from cupy import * doesn't overwrite these builtin names
from cupy import abs, max, min, round # noqa: F401

# These imports may overwrite names from the import * above.
from ._aliases import * # noqa: F403
from ._info import __array_namespace_info__  # noqa: F401

# See the comment in the numpy __init__.py
__import__(__spec__.parent + '.linalg')
__import__(__spec__.parent + '.fft')

__array_api_version__: Final = '2025.12'

__all__ = sorted(
    {name for name in globals() if not name.startswith("__")}
    - {"Final", "_aliases", "_info", "_typing"}
    | {"__array_api_version__", "__array_namespace_info__", "linalg", "fft"}
)

def __dir__() -> list[str]:
    # [FUNCTION DOC]
    # HELPER: Package __dir__ hook returning __all__.
    return __all__
