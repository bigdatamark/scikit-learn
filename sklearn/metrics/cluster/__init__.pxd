# [PACKAGE DOC]
# Package: sklearn.metrics.cluster
# Cython component of sklearn.metrics.cluster:
# - _expected_mutual_info_fast: expected_mutual_information(contingency, n_samples)
#   computes the expected mutual information of two labelings under the hypergeometric
#   permutation model, summing log-gamma based terms over every contingency cell and
#   admissible count. It is called by adjusted_mutual_info_score in _supervised.py to
#   correct mutual information for chance. All other modules here are pure Python.
# (This __init__.pxd holds only documentation; it declares nothing.)
