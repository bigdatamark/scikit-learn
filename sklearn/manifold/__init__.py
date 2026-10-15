# [PACKAGE DOC]
# Cython components, both used by TSNE:
# - _utils._binary_search_perplexity: per-sample binary search of Gaussian
#   precisions giving conditional affinities with a target perplexity.
# - _barnes_hut_tsne.gradient: Barnes-Hut approximation of the t-SNE KL
#   gradient (exact sparse attractive forces, tree-summarized repulsive forces,
#   OpenMP parallel), using sklearn.neighbors._quad_tree._QuadTree.
"""Data embedding techniques."""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

from sklearn.manifold._classical_mds import ClassicalMDS
from sklearn.manifold._isomap import Isomap
from sklearn.manifold._locally_linear import (
    LocallyLinearEmbedding,
    locally_linear_embedding,
)
from sklearn.manifold._mds import MDS, smacof
from sklearn.manifold._spectral_embedding import SpectralEmbedding, spectral_embedding
from sklearn.manifold._t_sne import TSNE, trustworthiness

__all__ = [
    "MDS",
    "TSNE",
    "ClassicalMDS",
    "Isomap",
    "LocallyLinearEmbedding",
    "SpectralEmbedding",
    "locally_linear_embedding",
    "smacof",
    "spectral_embedding",
    "trustworthiness",
]
