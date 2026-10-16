# [PACKAGE DOC]
# Package: sklearn.linear_model
# Cython components of sklearn.linear_model (all compiled via meson.build):
# - _cd_fast.pyx: coordinate descent solvers with duality-gap stopping and Gap Safe
#   screening for Lasso/ElasticNet (dense, sparse CSC with implicit centering and sample
#   weights, precomputed Gram) and their multi-task L2,1 variants; called from
#   _coordinate_descent.py.
# - _sag_fast.pyx.tp: Tempita template producing sag64/sag32, the SAG and SAGA solvers with
#   just-in-time (lazy) sparse updates and an L1 proximal step, called from _sag.py for Ridge
#   and LogisticRegression.
# - _sgd_fast.pyx.tp: Tempita template producing _plain_sgd64/_plain_sgd32, the SGD loop with
#   L2/L1/elastic-net penalties, averaging, learning-rate schedules, Passive-Aggressive steps
#   and early stopping, plus hinge/Huber/epsilon-insensitive loss types; used by
#   _stochastic_gradient.py and _passive_aggressive.py.
# The SAG and SGD modules rely on sklearn.utils._seq_dataset (sample iteration),
# sklearn.utils._weight_vector (SGD only) and sklearn._loss loss types; _cd_fast relies on
# the BLAS wrappers in sklearn.utils._cython_blas. The three modules are independent of
# each other.
# (This __init__.pxd holds only documentation; it declares nothing.)
