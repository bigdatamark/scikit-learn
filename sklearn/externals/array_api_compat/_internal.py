# [MODULE DOC]
# HELPER: Internal utilities of array_api_compat used to build per-backend wrapper
#   namespaces: a decorator that binds the xp argument to a fixed array module, and a
#   function that copies a module's public names into another module's globals.
# Module-level __dir__ restricts dir() of this module to __all__.
"""
Internal helpers
"""

import importlib
from collections.abc import Callable
from functools import wraps
from inspect import signature
from types import ModuleType
from typing import TypeVar

_T = TypeVar("_T")


def get_xp(xp: ModuleType) -> Callable[[Callable[..., _T]], Callable[..., _T]]:
    # [FUNCTION DOC]
    # HELPER: Decorator factory that pre-binds the keyword-only xp parameter of a
    #   generic implementation to a specific array module.
    # Inputs: xp - the array namespace module (e.g. numpy, cupy) to inject.
    # Outputs: a decorator; applying it to a function f returns a wrapper that calls f
    #   with all given arguments plus xp equal to the bound module.
    # Side effects: none at call time.
    """
    Decorator to automatically replace xp with the corresponding array module.

    Use like

    import numpy as np

    @get_xp(np)
    def func(x, /, xp, kwarg=None):
        return xp.func(x, kwarg=kwarg)

    Note that xp must be a keyword argument and come after all non-keyword
    arguments.

    """

    def inner(f: Callable[..., _T], /) -> Callable[..., _T]:
        # [FUNCTION DOC]
        # HELPER: The actual decorator returned by get_xp.
        # Inputs: f - a callable that accepts an xp keyword argument.
        # Outputs: wrapped_f, a functools.wraps wrapper whose __signature__ is f's
        #   signature with the xp parameter removed, and whose __doc__ is set to a
        #   generic "Array API compatibility wrapper for <name>" text when f had no
        #   docstring.
        # Exceptions: none directly; if the caller also passes xp, the wrapper raises
        #   TypeError for a duplicate keyword argument.
        # Side effects: mutates attributes (__doc__, __signature__) of the new wrapper
        #   only.
        @wraps(f)
        def wrapped_f(*args: object, **kwargs: object) -> object:
            # [FUNCTION DOC]
            # HELPER: Forwards all positional and keyword arguments to f, adding xp set
            #   to the module captured by get_xp, and returns f's result unchanged.
            return f(*args, xp=xp, **kwargs)

        sig = signature(f)
        new_sig = sig.replace(
            parameters=[par for i, par in sig.parameters.items() if i != "xp"]
        )

        if wrapped_f.__doc__ is None:
            wrapped_f.__doc__ = f"""\
Array API compatibility wrapper for {f.__name__}.

See the corresponding documentation in NumPy/CuPy and/or the array API
specification for more details.

"""
        wrapped_f.__signature__ = new_sig  # type: ignore[attr-defined] # pyright: ignore[reportAttributeAccessIssue]
        return wrapped_f  # type: ignore[return-value] # pyright: ignore[reportReturnType]

    return inner


def clone_module(mod_name: str, globals_: dict[str, object]) -> list[str]:
    # [FUNCTION DOC]
    # HELPER: Copies every public name of a module into a caller-provided globals dict,
    #   used to create a wrapper namespace that starts as a clone of the wrapped
    #   library.
    # Behavior: imports the module by name; collects names via executing a star-import
    #   into a fresh dict (honouring the module's __all__), then additionally adds every
    #   attribute from dir(module) not starting with an underscore; all collected
    #   objects are merged into globals_.
    # Inputs: mod_name - importable dotted module name; globals_ - dict to update
    #   (typically the caller's globals()).
    # Outputs: list of the collected names. KNOWN ISSUE: the star-import exec also
    #   inserts a __builtins__ entry into objs, so the returned list and globals_ update
    #   include "__builtins__".
    # Exceptions: ImportError/ModuleNotFoundError if the module cannot be imported.
    # Side effects: imports the module; mutates globals_.
    """Import everything from module, updating globals().
    Returns __all__.
    """
    mod = importlib.import_module(mod_name)
    # Neither of these two methods is sufficient by itself,
    # depending on various idiosyncrasies of the libraries we're wrapping.
    objs = {}
    exec(f"from {mod.__name__} import *", objs)

    for n in dir(mod):
        if not n.startswith("_") and hasattr(mod, n):
            objs[n] = getattr(mod, n)

    globals_.update(objs)
    return list(objs)


__all__ = ["get_xp", "clone_module"]

def __dir__() -> list[str]:
    # [FUNCTION DOC]
    # HELPER: Module-level dir() hook returning __all__ so only get_xp and clone_module
    #   are listed.
    return __all__
