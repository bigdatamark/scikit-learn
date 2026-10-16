# [PACKAGE DOC]
# Public entry point of the mixture subpackage (Gaussian mixture models).
# Exposes GaussianMixture (EM-fitted finite mixture) and BayesianGaussianMixture
# (variational inference with Dirichlet / Dirichlet-process weight priors), both
# re-exported from private modules and listed in __all__.
# Side effects at import: imports the two private implementation modules only.
"""Mixture modeling algorithms."""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

from sklearn.mixture._bayesian_mixture import BayesianGaussianMixture
from sklearn.mixture._gaussian_mixture import GaussianMixture

__all__ = ["BayesianGaussianMixture", "GaussianMixture"]
