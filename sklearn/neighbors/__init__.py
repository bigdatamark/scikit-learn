# [PACKAGE DOC]
# Cython components of sklearn.neighbors:
# - _binary_tree.pxi.tp: Tempita template for the generic BinaryTree
#   (complete binary tree over a permuted index array) with k-NN, radius,
#   kernel density and two-point correlation queries, plus NeighborsHeap,
#   NodeHeap and KDE kernel utilities. It is not compiled alone but
#   textually included by the two tree modules below.
# - _kd_tree.pyx.tp -> _kd_tree: KDTree64/KDTree32/KDTree, nodes are
#   axis-aligned bounding boxes (Minkowski-family metrics only).
# - _ball_tree.pyx.tp -> _ball_tree: BallTree64/BallTree32/BallTree, nodes
#   are centroid + radius spheres (any true metric).
#   Both define the geometry hooks (init_node, min/max distance bounds)
#   that the shared BinaryTree code calls, and are used by the Python
#   estimators (NearestNeighbors, KernelDensity, etc.) via _base.py.
# - _partition_nodes.pyx/.pxd: partition_node_indices, an nth_element based
#   median split used when building BinaryTree nodes.
# - _quad_tree.pyx/.pxd: _QuadTree, an array-based 2D quadtree / 3D octree
#   with Barnes-Hut summaries, used by sklearn.manifold.TSNE (independent
#   of the binary trees).
"""The k-nearest neighbors algorithms."""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

from sklearn.neighbors._ball_tree import BallTree
from sklearn.neighbors._base import (
    VALID_METRICS,
    VALID_METRICS_SPARSE,
    sort_graph_by_row_values,
)
from sklearn.neighbors._classification import (
    KNeighborsClassifier,
    RadiusNeighborsClassifier,
)
from sklearn.neighbors._graph import (
    KNeighborsTransformer,
    RadiusNeighborsTransformer,
    kneighbors_graph,
    radius_neighbors_graph,
)
from sklearn.neighbors._kd_tree import KDTree
from sklearn.neighbors._kde import KernelDensity
from sklearn.neighbors._lof import LocalOutlierFactor
from sklearn.neighbors._nca import NeighborhoodComponentsAnalysis
from sklearn.neighbors._nearest_centroid import NearestCentroid
from sklearn.neighbors._regression import KNeighborsRegressor, RadiusNeighborsRegressor
from sklearn.neighbors._unsupervised import NearestNeighbors

__all__ = [
    "VALID_METRICS",
    "VALID_METRICS_SPARSE",
    "BallTree",
    "KDTree",
    "KNeighborsClassifier",
    "KNeighborsRegressor",
    "KNeighborsTransformer",
    "KernelDensity",
    "LocalOutlierFactor",
    "NearestCentroid",
    "NearestNeighbors",
    "NeighborhoodComponentsAnalysis",
    "RadiusNeighborsClassifier",
    "RadiusNeighborsRegressor",
    "RadiusNeighborsTransformer",
    "kneighbors_graph",
    "radius_neighbors_graph",
    "sort_graph_by_row_values",
]
