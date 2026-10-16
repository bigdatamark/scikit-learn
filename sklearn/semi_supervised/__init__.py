# [PACKAGE DOC]
# Public entry point of the semi-supervised learning package. Re-exports the
# graph-based estimators LabelPropagation and LabelSpreading (from
# _label_propagation) and the meta-estimator SelfTrainingClassifier (from
# _self_training); __all__ lists exactly these three names.
# Import-time side effects: imports both private submodules; nothing else.
"""Semi-supervised learning algorithms.

These algorithms utilize small amounts of labeled data and large amounts of unlabeled
data for classification tasks.
"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

from sklearn.semi_supervised._label_propagation import LabelPropagation, LabelSpreading
from sklearn.semi_supervised._self_training import SelfTrainingClassifier

__all__ = ["LabelPropagation", "LabelSpreading", "SelfTrainingClassifier"]
