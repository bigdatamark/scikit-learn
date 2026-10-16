# [PACKAGE DOC]
# Vendored subset of scipy.sparse.csgraph; exposes only the laplacian function from the
#   _laplacian submodule.
# Side effects: importing it imports _laplacian (and therefore numpy and scipy.sparse).
from ._laplacian import laplacian
