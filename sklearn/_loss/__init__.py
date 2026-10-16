# [PACKAGE DOC]
# Package sklearn._loss: private loss-function objects (value, gradient, hessian,
# link functions) used by gradient-based estimators such as HistGradientBoosting,
# GLMs and logistic regression. Re-exports the concrete loss classes from
# sklearn._loss.loss (AbsoluteError, HalfBinomialLoss and its Array API variant,
# HalfGammaLoss, HalfMultinomialLoss and its Array API variant, HalfPoissonLoss,
# HalfSquaredError, HalfTweedieLoss, HalfTweedieLossIdentity, HuberLoss,
# PinballLoss). Import-time side effects: imports the compiled _loss extension.
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
