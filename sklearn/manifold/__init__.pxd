# [PACKAGE DOC]
# Package: sklearn.manifold
# Cython components, both used by TSNE:
# - _utils._binary_search_perplexity: per-sample binary search of Gaussian
#   precisions giving conditional affinities with a target perplexity.
# - _barnes_hut_tsne.gradient: Barnes-Hut approximation of the t-SNE KL
#   gradient (exact sparse attractive forces, tree-summarized repulsive forces,
#   OpenMP parallel), using sklearn.neighbors._quad_tree._QuadTree.
# (This __init__.pxd holds only documentation; it declares nothing.)
