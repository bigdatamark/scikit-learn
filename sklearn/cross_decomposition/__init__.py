# [PACKAGE DOC]
# Cross decomposition package: models that find latent directions relating two data
# matrices X and y. Re-exports CCA, PLSSVD, PLSCanonical and PLSRegression from the
# private _pls module (listed in __all__). Importing it only imports _pls; no other side
# effects.
"""Algorithms for cross decomposition."""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

from sklearn.cross_decomposition._pls import CCA, PLSSVD, PLSCanonical, PLSRegression

__all__ = ["CCA", "PLSSVD", "PLSCanonical", "PLSRegression"]
