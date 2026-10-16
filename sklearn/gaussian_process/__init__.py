# [PACKAGE DOC]
# Public package for Gaussian process models.
# Exposes GaussianProcessClassifier (from _gpc), GaussianProcessRegressor (from _gpr)
# and the kernels submodule via __all__.
# Import-time side effects: imports the kernels, _gpc and _gpr modules; no global state.
"""Gaussian process based regression and classification."""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

from sklearn.gaussian_process import kernels
from sklearn.gaussian_process._gpc import GaussianProcessClassifier
from sklearn.gaussian_process._gpr import GaussianProcessRegressor

__all__ = ["GaussianProcessClassifier", "GaussianProcessRegressor", "kernels"]
