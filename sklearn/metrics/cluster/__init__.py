# [PACKAGE DOC]
# Package: sklearn.metrics.cluster
# Cython component of sklearn.metrics.cluster:
# - _expected_mutual_info_fast: expected_mutual_information(contingency, n_samples)
#   computes the expected mutual information of two labelings under the hypergeometric
#   permutation model, summing log-gamma based terms over every contingency cell and
#   admissible count. It is called by adjusted_mutual_info_score in _supervised.py to
#   correct mutual information for chance. All other modules here are pure Python.
"""Evaluation metrics for cluster analysis results.

- Supervised evaluation uses a ground truth class values for each sample.
- Unsupervised evaluation does not use ground truths and measures the "quality" of the
  model itself.
"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

from sklearn.metrics.cluster._bicluster import consensus_score
from sklearn.metrics.cluster._supervised import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    completeness_score,
    contingency_matrix,
    entropy,
    expected_mutual_information,
    fowlkes_mallows_score,
    homogeneity_completeness_v_measure,
    homogeneity_score,
    mutual_info_score,
    normalized_mutual_info_score,
    pair_confusion_matrix,
    rand_score,
    v_measure_score,
)
from sklearn.metrics.cluster._unsupervised import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_samples,
    silhouette_score,
)

__all__ = [
    "adjusted_mutual_info_score",
    "adjusted_rand_score",
    "calinski_harabasz_score",
    "completeness_score",
    "consensus_score",
    "contingency_matrix",
    "davies_bouldin_score",
    # TODO(1.10): Remove
    "entropy",
    "expected_mutual_information",
    "fowlkes_mallows_score",
    "homogeneity_completeness_v_measure",
    "homogeneity_score",
    "mutual_info_score",
    "normalized_mutual_info_score",
    "pair_confusion_matrix",
    "rand_score",
    "silhouette_samples",
    "silhouette_score",
    "v_measure_score",
]
