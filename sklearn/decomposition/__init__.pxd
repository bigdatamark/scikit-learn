# [PACKAGE DOC]
# Package: sklearn.decomposition
# Cython components:
# - _cdnmf_fast: one coordinate-descent sweep (Newton step with projection onto
#   non-negativity) used by the NMF "cd" solver for both W and H.
# - _online_lda_fast: helpers for LatentDirichletAllocation's variational
#   updates: mean absolute change, Dirichlet expectations of log variables (1D
#   in place with prior and exp, 2D without), and a fast digamma approximation.
# The two modules are independent.
# (This __init__.pxd holds only documentation; it declares nothing.)
