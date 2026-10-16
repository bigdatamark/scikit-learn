# [PACKAGE DOC]
# Public package sklearn.inspection for model inspection. Re-exports partial_dependence,
# permutation_importance, DecisionBoundaryDisplay and PartialDependenceDisplay (listed
# in __all__). Importing it imports the private computation and plotting submodules; no
# other import-time side effects (matplotlib is only imported lazily by the displays).
"""Tools for model inspection."""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

from sklearn.inspection._partial_dependence import partial_dependence
from sklearn.inspection._permutation_importance import permutation_importance
from sklearn.inspection._plot.decision_boundary import DecisionBoundaryDisplay
from sklearn.inspection._plot.partial_dependence import PartialDependenceDisplay

__all__ = [
    "DecisionBoundaryDisplay",
    "PartialDependenceDisplay",
    "partial_dependence",
    "permutation_importance",
]
