# [PACKAGE DOC]
# Package: sklearn.cluster._hdbscan
# Cython components of the HDBSCAN implementation (driven by hdbscan.py):
# - _reachability: in-place mutual reachability transform of dense or CSR
#   distance matrices.
# - _linkage: Prim's MST over mutual reachability (from a matrix or from raw data
#   with a DistanceMetric64) and conversion of a sorted MST to a single linkage
#   hierarchy, using UnionFind from sklearn.cluster._hierarchical_fast.
# - _tree (+ _tree.pxd record types HIERARCHY_t, CONDENSED_t): condensing the
#   hierarchy, stability computation, eom/leaf cluster selection with epsilon and
#   max size, labels and probabilities.
# Pipeline: distances -> mutual reachability -> MST -> single linkage -> condensed
# tree -> flat labels.
# (This __init__.pxd holds only documentation; it declares nothing.)
