# [PACKAGE DOC]
# Public entry point of the support vector machine package. Re-exports the
# libsvm-based estimators SVC, NuSVC, SVR, NuSVR, OneClassSVM, the
# liblinear-based LinearSVC and LinearSVR (all from _classes), and the helper
# l1_min_c (from _bounds); __all__ lists exactly these names.
# Import-time side effects: imports the private submodules, which load the
# compiled libsvm/liblinear extension modules.
"""Support vector machine algorithms."""

# See http://scikit-learn.sourceforge.net/modules/svm.html for complete
# documentation.

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

from sklearn.svm._bounds import l1_min_c
from sklearn.svm._classes import (
    SVC,
    SVR,
    LinearSVC,
    LinearSVR,
    NuSVC,
    NuSVR,
    OneClassSVM,
)

__all__ = [
    "SVC",
    "SVR",
    "LinearSVC",
    "LinearSVR",
    "NuSVC",
    "NuSVR",
    "OneClassSVM",
    "l1_min_c",
]
