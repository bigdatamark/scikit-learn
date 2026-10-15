# [PACKAGE DOC]
# Package: sklearn._loss
# Cython components of sklearn._loss:
# - _loss.pyx.tp: Tempita template generating _loss.pyx, which implements nogil
#   single-sample loss/gradient/hessian kernels (link folded in) and the Cy*
#   extension types (one per loss, plus CyHalfMultinomialLoss) with
#   OpenMP-parallel array methods writing into caller-provided outputs.
# - _loss.pxd: declarations of those extension types, fused float types and
#   the double_pair struct, so other Cython modules (SGD, SAG) can cimport and
#   call the losses without the GIL.
# loss.py wraps these types in Python classes (input checks, link objects from
# link.py) used across sklearn's estimators.
# (This __init__.pxd holds only documentation; it declares nothing.)
