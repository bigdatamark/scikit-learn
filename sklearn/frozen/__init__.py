# [PACKAGE DOC]
# Public package for freezing already-fitted estimators.
# Exposes FrozenEstimator (re-exported from the private _frozen module) via __all__.
# Import-time side effects: imports sklearn.frozen._frozen only; no global state
#   changes.
# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

from sklearn.frozen._frozen import FrozenEstimator

__all__ = ["FrozenEstimator"]
