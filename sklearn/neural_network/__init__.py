# [PACKAGE DOC]
# Public package for neural-network based models. Re-exports MLPClassifier and
# MLPRegressor (multilayer perceptrons from _multilayer_perceptron) and
# BernoulliRBM (restricted Boltzmann machine from _rbm); __all__ lists these three.
# Import-time side effects: imports those submodules only; no global state.
"""Models based on neural networks."""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

from sklearn.neural_network._multilayer_perceptron import MLPClassifier, MLPRegressor
from sklearn.neural_network._rbm import BernoulliRBM

__all__ = ["BernoulliRBM", "MLPClassifier", "MLPRegressor"]
