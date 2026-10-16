# [PACKAGE DOC]
# Public testing subpackage of the vendored array_api_extra library. Re-exports the
# array-API-agnostic assertion helpers (assert_close, assert_close_nulp,
# assert_equal, assert_less) and the lazy-backend test tooling (lazy_xp_function,
# patch_lazy_xp_functions) from the private _testing module and lists them in
# __all__. Import side effect: imports _testing (which imports the array_api_compat
# helpers and optionally pytest/dask/jax lazily).
"""Public testing utilities."""

from ._testing import (
    assert_close,
    assert_close_nulp,
    assert_equal,
    assert_less,
    lazy_xp_function,
    patch_lazy_xp_functions,
)

__all__ = [
    "assert_close",
    "assert_close_nulp",
    "assert_equal",
    "assert_less",
    "lazy_xp_function",
    "patch_lazy_xp_functions",
]
