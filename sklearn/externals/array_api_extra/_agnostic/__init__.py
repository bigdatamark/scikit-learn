# [PACKAGE DOC]
# HELPER: Subpackage holding the backend-agnostic implementations (pure array API code)
#   that the public dispatching wrappers fall back to. Exposes the implementation
#   submodules _creation, _elementwise, _indexing, _inspection, _linalg, _manipulation,
#   _searching, _set, _sorting and _statistical; importing it imports all of them.
"""Array-agnostic function implementations."""

from . import (
    _creation,
    _elementwise,
    _indexing,
    _inspection,
    _linalg,
    _manipulation,
    _searching,
    _set,
    _sorting,
    _statistical,
)

__all__ = [
    "_creation",
    "_elementwise",
    "_indexing",
    "_inspection",
    "_linalg",
    "_manipulation",
    "_searching",
    "_set",
    "_sorting",
    "_statistical",
]
