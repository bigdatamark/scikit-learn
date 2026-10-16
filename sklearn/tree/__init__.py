# [PACKAGE DOC]
# Public package for decision-tree models. Re-exports from the private submodules
# _classes and _export: BaseDecisionTree, DecisionTreeClassifier,
# DecisionTreeRegressor, ExtraTreeClassifier, ExtraTreeRegressor, and the
# visualization/export helpers export_graphviz, export_text and plot_tree; __all__
# lists exactly these names.
# Side effects at import: imports the compiled tree extension modules via _classes;
# no other global state is touched.
"""Decision tree based models for classification and regression."""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

from sklearn.tree._classes import (
    BaseDecisionTree,
    DecisionTreeClassifier,
    DecisionTreeRegressor,
    ExtraTreeClassifier,
    ExtraTreeRegressor,
)
from sklearn.tree._export import export_graphviz, export_text, plot_tree

__all__ = [
    "BaseDecisionTree",
    "DecisionTreeClassifier",
    "DecisionTreeRegressor",
    "ExtraTreeClassifier",
    "ExtraTreeRegressor",
    "export_graphviz",
    "export_text",
    "plot_tree",
]
