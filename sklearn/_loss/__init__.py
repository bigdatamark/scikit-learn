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
"""
The :mod:`sklearn._loss` module includes loss function classes suitable for
fitting classification and regression tasks.
"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

from sklearn._loss.loss import (
    AbsoluteError,
    HalfBinomialLoss,
    HalfBinomialLossArrayAPI,
    HalfGammaLoss,
    HalfMultinomialLoss,
    HalfMultinomialLossArrayAPI,
    HalfPoissonLoss,
    HalfSquaredError,
    HalfTweedieLoss,
    HalfTweedieLossIdentity,
    HuberLoss,
    PinballLoss,
)

__all__ = [
    "AbsoluteError",
    "HalfBinomialLoss",
    "HalfBinomialLossArrayAPI",
    "HalfGammaLoss",
    "HalfMultinomialLoss",
    "HalfMultinomialLossArrayAPI",
    "HalfPoissonLoss",
    "HalfSquaredError",
    "HalfTweedieLoss",
    "HalfTweedieLossIdentity",
    "HuberLoss",
    "PinballLoss",
]
