# [PACKAGE DOC]
# Public entry point of the sklearn.ensemble package. Re-exports the ensemble
# estimators implemented in private submodules: bagging (BaggingClassifier,
# BaggingRegressor), the abstract BaseEnsemble, random/extra-trees forests and
# RandomTreesEmbedding, gradient boosting (classic and histogram-based),
# IsolationForest, stacking, voting and AdaBoost estimators; __all__ lists them.
# Import-time side effects: imports all those submodules (including compiled tree and
# histogram gradient boosting extensions); no global state is modified.
"""Ensemble-based methods for classification, regression and anomaly detection."""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

from sklearn.ensemble._bagging import BaggingClassifier, BaggingRegressor
from sklearn.ensemble._base import BaseEnsemble
from sklearn.ensemble._forest import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
    RandomTreesEmbedding,
)
from sklearn.ensemble._gb import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.ensemble._hist_gradient_boosting.gradient_boosting import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.ensemble._iforest import IsolationForest
from sklearn.ensemble._stacking import StackingClassifier, StackingRegressor
from sklearn.ensemble._voting import VotingClassifier, VotingRegressor
from sklearn.ensemble._weight_boosting import AdaBoostClassifier, AdaBoostRegressor

__all__ = [
    "AdaBoostClassifier",
    "AdaBoostRegressor",
    "BaggingClassifier",
    "BaggingRegressor",
    "BaseEnsemble",
    "ExtraTreesClassifier",
    "ExtraTreesRegressor",
    "GradientBoostingClassifier",
    "GradientBoostingRegressor",
    "HistGradientBoostingClassifier",
    "HistGradientBoostingRegressor",
    "IsolationForest",
    "RandomForestClassifier",
    "RandomForestRegressor",
    "RandomTreesEmbedding",
    "StackingClassifier",
    "StackingRegressor",
    "VotingClassifier",
    "VotingRegressor",
]
