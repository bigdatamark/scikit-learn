# [PACKAGE DOC]
# Private package holding the histogram-based gradient boosting implementation
# (HistGradientBoostingRegressor/Classifier, binning, tree growing, splitting,
# histograms and prediction, partly in Cython). This __init__ exposes nothing: it
# only holds a docstring and license header, so importing it has no side effects.
# Public estimators are re-exported from sklearn.ensemble.
"""This module implements histogram-based gradient boosting estimators.

The implementation is a port from pygbm which is itself strongly inspired
from LightGBM.
"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause
