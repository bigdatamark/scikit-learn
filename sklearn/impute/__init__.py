# [PACKAGE DOC]
# Public package for missing-value imputation. Re-exports SimpleImputer and
# MissingIndicator (from _base) and KNNImputer (from _knn); __all__ lists these three.
# IterativeImputer is experimental: it is only imported under typing.TYPE_CHECKING for
# type checkers, and accessing it at runtime goes through the module-level __getattr__,
# which raises an informative ImportError until sklearn.experimental
# enable_iterative_imputer has been imported (that module then sets the attribute).
# Import-time side effects: imports _base and _knn (and their dependencies) only.
"""Transformers for missing value imputation."""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import typing

from sklearn.impute._base import MissingIndicator, SimpleImputer
from sklearn.impute._knn import KNNImputer

if typing.TYPE_CHECKING:
    # Avoid errors in type checkers (e.g. mypy) for experimental estimators.
    # TODO: remove this check once the estimator is no longer experimental.
    from sklearn.impute._iterative import IterativeImputer  # noqa: F401

__all__ = ["KNNImputer", "MissingIndicator", "SimpleImputer"]


# TODO: remove this check once the estimator is no longer experimental.
def __getattr__(name):
    # [FUNCTION DOC]
    # Module-level attribute fallback (PEP 562), called only for names not found
    # normally.
    # Inputs: name - requested attribute name (str).
    # Outputs: never returns.
    # Exceptions: ImportError with instructions to import enable_iterative_imputer when
    # name is "IterativeImputer"; AttributeError for any other unknown name.
    if name == "IterativeImputer":
        raise ImportError(
            f"{name} is experimental and the API might change without any "
            "deprecation cycle. To use it, you need to explicitly import "
            "enable_iterative_imputer:\n"
            "from sklearn.experimental import enable_iterative_imputer"
        )
    raise AttributeError(f"module {__name__} has no attribute {name}")
