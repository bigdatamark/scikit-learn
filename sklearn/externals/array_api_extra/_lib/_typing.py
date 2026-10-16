# [MODULE DOC]
# HELPER: Runtime placeholders for static types (Array, ArrayLike, DType, Device, Key,
#   index types, Graph, etc. all alias object; ArrayNamespace is ModuleType; TypeIs is
#   Any) and a no-op override decorator; richer definitions live in the accompanying
#   stub file.
"""Static typing helpers — better implementations in the stub file."""

from collections.abc import Callable
from types import ModuleType
from typing import Any

Array = object
ArrayLike = object
ArrayNamespace = ModuleType
DType = object
Device = object
Key = object
GetIndex = object
Graph = object
NumPyObject = object
SchedulerGetCallable = object
SetIndex = object

TypeIs = Any

__all__ = [
    "Array",
    "ArrayLike",
    "ArrayNamespace",
    "DType",
    "Device",
    "GetIndex",
    "Graph",
    "Key",
    "NumPyObject",
    "SchedulerGetCallable",
    "SetIndex",
    "TypeIs",
    "override",
]


def override(func: Callable[..., Any]) -> Callable[..., Any]:  # numpydoc ignore=GL08
    # [FUNCTION DOC]
    # HELPER: Identity decorator standing in for typing.override; returns func
    #   unchanged.
    return func
