# [MODULE DOC]
# Tests shared by both mixture estimators (GaussianMixture and BayesianGaussianMixture),
#   parametrized over a default instance of each.
# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest

from sklearn.base import clone
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture


@pytest.mark.parametrize("estimator", [GaussianMixture(), BayesianGaussianMixture()])
def test_gaussian_mixture_n_iter(estimator):
    # [FUNCTION DOC]
    # Verifies n_iter_ equals max_iter when fitting with max_iter=1 on 10x5 uniform
    #   random data (a cloned estimator is used to avoid shared-state leakage between
    #   parametrizations).
    # check that n_iter is the number of iteration performed.
    estimator = clone(estimator)  # Avoid side effects from shared instances
    rng = np.random.RandomState(0)
    X = rng.rand(10, 5)
    max_iter = 1
    estimator.set_params(max_iter=max_iter)
    estimator.fit(X)
    assert estimator.n_iter_ == max_iter


@pytest.mark.parametrize("estimator", [GaussianMixture(), BayesianGaussianMixture()])
def test_mixture_n_components_greater_than_n_samples_error(estimator):
    # [FUNCTION DOC]
    # Verifies fit raises ValueError matching "Expected n_samples >= n_components" when
    #   n_components (12) exceeds n_samples (10).
    """Check error when n_components <= n_samples"""
    estimator = clone(estimator)  # Avoid side effects from shared instances
    rng = np.random.RandomState(0)
    X = rng.rand(10, 5)
    estimator.set_params(n_components=12)

    msg = "Expected n_samples >= n_components"
    with pytest.raises(ValueError, match=msg):
        estimator.fit(X)
