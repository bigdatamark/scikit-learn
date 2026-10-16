# [MODULE DOC]
# Tests for EllipticEnvelope: not-fitted errors, consistency of scores, decisions and
# predictions, and score_samples independence from contamination.
"""
Testing for Elliptic Envelope algorithm (sklearn.covariance.elliptic_envelope).
"""

import numpy as np
import pytest

from sklearn.covariance import EllipticEnvelope
from sklearn.exceptions import NotFittedError
from sklearn.utils._testing import (
    assert_almost_equal,
    assert_array_almost_equal,
    assert_array_equal,
)


def test_elliptic_envelope(global_random_seed):
    # [FUNCTION DOC]
    # Verifies on 100x10 Gaussian data with contamination 0.1: predict and
    # decision_function raise NotFittedError before fit; after fit, score_samples equals
    # minus mahalanobis, mahalanobis equals dist_, score against all-ones labels equals
    # the inlier fraction, and the number of -1 predictions equals the number of
    # negative decisions.
    rnd = np.random.RandomState(global_random_seed)
    X = rnd.randn(100, 10)
    clf = EllipticEnvelope(contamination=0.1)
    with pytest.raises(NotFittedError):
        clf.predict(X)
    with pytest.raises(NotFittedError):
        clf.decision_function(X)
    clf.fit(X)
    y_pred = clf.predict(X)
    scores = clf.score_samples(X)
    decisions = clf.decision_function(X)

    assert_array_almost_equal(scores, -clf.mahalanobis(X))
    assert_array_almost_equal(clf.mahalanobis(X), clf.dist_)
    assert_almost_equal(
        clf.score(X, np.ones(100)), (100 - y_pred[y_pred == -1].size) / 100.0
    )
    assert sum(y_pred == -1) == sum(decisions < 0)


def test_score_samples():
    # [FUNCTION DOC]
    # Verifies score_samples equals decision_function plus offset_ for two contamination
    # settings and that score_samples does not depend on contamination.
    X_train = [[1, 1], [1, 2], [2, 1]]
    clf1 = EllipticEnvelope(contamination=0.2).fit(X_train)
    clf2 = EllipticEnvelope().fit(X_train)
    assert_array_equal(
        clf1.score_samples([[2.0, 2.0]]),
        clf1.decision_function([[2.0, 2.0]]) + clf1.offset_,
    )
    assert_array_equal(
        clf2.score_samples([[2.0, 2.0]]),
        clf2.decision_function([[2.0, 2.0]]) + clf2.offset_,
    )
    assert_array_equal(
        clf1.score_samples([[2.0, 2.0]]), clf2.score_samples([[2.0, 2.0]])
    )
