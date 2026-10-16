# [MODULE DOC]
# Tests for orthogonal matching pursuit: orthogonal_mp, orthogonal_mp_gram,
# OrthogonalMatchingPursuit and OrthogonalMatchingPursuitCV.
# At import time builds a sparse-coded signal with make_sparse_coded_signal
# (random_state=0): X is a (25, 35) dictionary, y a (25, 3) target and gamma the
# (35, 3) true sparse codes with 5 nonzeros per target. X and y are scaled by 10 so
# the dictionary atoms are not unit norm; G is the Gram matrix of X and Xy is X
# transposed times y. These module globals are shared (read-only) by the tests.
# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause


import numpy as np
import pytest

from sklearn.datasets import make_sparse_coded_signal
from sklearn.linear_model import (
    LinearRegression,
    OrthogonalMatchingPursuit,
    OrthogonalMatchingPursuitCV,
    orthogonal_mp,
    orthogonal_mp_gram,
)
from sklearn.utils import check_random_state
from sklearn.utils._testing import (
    assert_allclose,
    assert_array_almost_equal,
    assert_array_equal,
    ignore_warnings,
)

n_samples, n_features, n_nonzero_coefs, n_targets = 25, 35, 5, 3
y, X, gamma = make_sparse_coded_signal(
    n_samples=n_targets,
    n_components=n_features,
    n_features=n_samples,
    n_nonzero_coefs=n_nonzero_coefs,
    random_state=0,
)
y, X, gamma = y.T, X.T, gamma.T
# Make X not of norm 1 for testing
X *= 10
y *= 10
G, Xy = np.dot(X.T, X), np.dot(X.T, y)
# this makes X (n_samples, n_features)
# and y (n_samples, 3)


def test_correct_shapes():
    # [FUNCTION DOC]
    # Verifies orthogonal_mp returns a (n_features,) vector for a 1D target and a
    # (n_features, 3) array for the 2D target.
    assert orthogonal_mp(X, y[:, 0], n_nonzero_coefs=5).shape == (n_features,)
    assert orthogonal_mp(X, y, n_nonzero_coefs=5).shape == (n_features, 3)


def test_correct_shapes_gram():
    # [FUNCTION DOC]
    # Same shape check as test_correct_shapes but for orthogonal_mp_gram given G and Xy.
    assert orthogonal_mp_gram(G, Xy[:, 0], n_nonzero_coefs=5).shape == (n_features,)
    assert orthogonal_mp_gram(G, Xy, n_nonzero_coefs=5).shape == (n_features, 3)


def test_n_nonzero_coefs():
    # [FUNCTION DOC]
    # Verifies the solution has at most n_nonzero_coefs=5 nonzero entries, both without
    #   and
    # with precompute=True.
    assert np.count_nonzero(orthogonal_mp(X, y[:, 0], n_nonzero_coefs=5)) <= 5
    assert (
        np.count_nonzero(orthogonal_mp(X, y[:, 0], n_nonzero_coefs=5, precompute=True))
        <= 5
    )


def test_tol():
    # [FUNCTION DOC]
    # Verifies that with tol=0.5 the squared residual norm of the solution is at most
    #   tol,
    # both for the direct and the precomputed-Gram path.
    tol = 0.5
    gamma = orthogonal_mp(X, y[:, 0], tol=tol)
    gamma_gram = orthogonal_mp(X, y[:, 0], tol=tol, precompute=True)
    assert np.sum((y[:, 0] - np.dot(X, gamma)) ** 2) <= tol
    assert np.sum((y[:, 0] - np.dot(X, gamma_gram)) ** 2) <= tol


def test_with_without_gram():
    # [FUNCTION DOC]
    # Verifies the direct and precompute=True solutions agree for n_nonzero_coefs=5.
    assert_array_almost_equal(
        orthogonal_mp(X, y, n_nonzero_coefs=5),
        orthogonal_mp(X, y, n_nonzero_coefs=5, precompute=True),
    )


def test_with_without_gram_tol():
    # [FUNCTION DOC]
    # Verifies the direct and precompute=True solutions agree for tol=1.0.
    assert_array_almost_equal(
        orthogonal_mp(X, y, tol=1.0), orthogonal_mp(X, y, tol=1.0, precompute=True)
    )


def test_unreachable_accuracy():
    # [FUNCTION DOC]
    # Verifies that tol=0 (unreachable precision) gives the same result as selecting all
    # n_features atoms, and that with precompute=True this situation emits a
    # RuntimeWarning about premature termination due to linear dependence.
    assert_array_almost_equal(
        orthogonal_mp(X, y, tol=0), orthogonal_mp(X, y, n_nonzero_coefs=n_features)
    )
    warning_message = (
        "Orthogonal matching pursuit ended prematurely "
        "due to linear dependence in the dictionary. "
        "The requested precision might not have been met."
    )
    with pytest.warns(RuntimeWarning, match=warning_message):
        assert_array_almost_equal(
            orthogonal_mp(X, y, tol=0, precompute=True),
            orthogonal_mp(X, y, precompute=True, n_nonzero_coefs=n_features),
        )


@pytest.mark.parametrize("positional_params", [(X, y), (G, Xy)])
@pytest.mark.parametrize(
    "keyword_params",
    [{"n_nonzero_coefs": n_features + 1}],
)
def test_bad_input(positional_params, keyword_params):
    # [FUNCTION DOC]
    # Verifies orthogonal_mp raises ValueError when n_nonzero_coefs exceeds n_features.
    # Inputs: positional_params - (X, y) or (G, Xy); keyword_params - the bad kwargs.
    # KNOWN ISSUE: the (G, Xy) case passes the Gram matrix to orthogonal_mp as if it
    #   were a
    # design matrix rather than calling orthogonal_mp_gram; it still raises because G
    #   has
    # n_features columns, but it does not test the Gram code path.
    with pytest.raises(ValueError):
        orthogonal_mp(*positional_params, **keyword_params)


def test_perfect_signal_recovery():
    # [FUNCTION DOC]
    # Verifies that with n_nonzero_coefs=5 both orthogonal_mp and orthogonal_mp_gram
    # recover exactly the support of the first true code column and its values to 2
    # decimals.
    (idx,) = gamma[:, 0].nonzero()
    gamma_rec = orthogonal_mp(X, y[:, 0], n_nonzero_coefs=5)
    gamma_gram = orthogonal_mp_gram(G, Xy[:, 0], n_nonzero_coefs=5)
    assert_array_equal(idx, np.flatnonzero(gamma_rec))
    assert_array_equal(idx, np.flatnonzero(gamma_gram))
    assert_array_almost_equal(gamma[:, 0], gamma_rec, decimal=2)
    assert_array_almost_equal(gamma[:, 0], gamma_gram, decimal=2)


def test_orthogonal_mp_gram_readonly():
    # [FUNCTION DOC]
    # Non-regression test (issue 5956): orthogonal_mp_gram must work on read-only G and
    #   Xy
    # copies with copy_Gram=False and copy_Xy=False, still recovering the true support
    #   and
    # values (2 decimals).
    # Non-regression test for:
    # https://github.com/scikit-learn/scikit-learn/issues/5956
    (idx,) = gamma[:, 0].nonzero()
    G_readonly = G.copy()
    G_readonly.setflags(write=False)
    Xy_readonly = Xy.copy()
    Xy_readonly.setflags(write=False)
    gamma_gram = orthogonal_mp_gram(
        G_readonly, Xy_readonly[:, 0], n_nonzero_coefs=5, copy_Gram=False, copy_Xy=False
    )
    assert_array_equal(idx, np.flatnonzero(gamma_gram))
    assert_array_almost_equal(gamma[:, 0], gamma_gram, decimal=2)


def test_estimator():
    # [FUNCTION DOC]
    # Verifies OrthogonalMatchingPursuit fitted attributes: coef_ and intercept_ shapes
    #   for
    # 1D and 2D targets, sparsity bounded by n_nonzero_coefs per target, consistency of
    # coefficients when refitting with fit_intercept=True, and intercept_ equal to 0
    #   with
    # fit_intercept=False.
    # KNOWN ISSUE: the variable coef_normalized and the refit with fit_intercept=True
    #   are
    # leftovers from a removed normalize option; fit_intercept was already True, so that
    # comparison only checks refit determinism.
    omp = OrthogonalMatchingPursuit(n_nonzero_coefs=n_nonzero_coefs)
    omp.fit(X, y[:, 0])
    assert omp.coef_.shape == (n_features,)
    assert omp.intercept_.shape == ()
    assert np.count_nonzero(omp.coef_) <= n_nonzero_coefs

    omp.fit(X, y)
    assert omp.coef_.shape == (n_targets, n_features)
    assert omp.intercept_.shape == (n_targets,)
    assert np.count_nonzero(omp.coef_) <= n_targets * n_nonzero_coefs

    coef_normalized = omp.coef_[0].copy()
    omp.set_params(fit_intercept=True)
    omp.fit(X, y[:, 0])
    assert_array_almost_equal(coef_normalized, omp.coef_)

    omp.set_params(fit_intercept=False)
    omp.fit(X, y[:, 0])
    assert np.count_nonzero(omp.coef_) <= n_nonzero_coefs
    assert omp.coef_.shape == (n_features,)
    assert omp.intercept_ == 0

    omp.fit(X, y)
    assert omp.coef_.shape == (n_targets, n_features)
    assert omp.intercept_ == 0
    assert np.count_nonzero(omp.coef_) <= n_targets * n_nonzero_coefs


def test_estimator_n_nonzero_coefs():
    # [FUNCTION DOC]
    # Verifies n_nonzero_coefs_ equals the requested n_nonzero_coefs when tol is unset
    #   and
    # is None when tol is given (tol takes precedence).
    """Check `n_nonzero_coefs_` correct when `tol` is and isn't set."""
    omp = OrthogonalMatchingPursuit(n_nonzero_coefs=n_nonzero_coefs)
    omp.fit(X, y[:, 0])
    assert omp.n_nonzero_coefs_ == n_nonzero_coefs

    omp = OrthogonalMatchingPursuit(n_nonzero_coefs=n_nonzero_coefs, tol=0.5)
    omp.fit(X, y[:, 0])
    assert omp.n_nonzero_coefs_ is None


def test_identical_regressors():
    # [FUNCTION DOC]
    # Verifies that duplicating column 0 into column 1 and building y from both makes
    # orthogonal_mp with 2 atoms emit the RuntimeWarning about linear dependence.
    newX = X.copy()
    newX[:, 1] = newX[:, 0]
    gamma = np.zeros(n_features)
    gamma[0] = gamma[1] = 1.0
    newy = np.dot(newX, gamma)
    warning_message = (
        "Orthogonal matching pursuit ended prematurely "
        "due to linear dependence in the dictionary. "
        "The requested precision might not have been met."
    )
    with pytest.warns(RuntimeWarning, match=warning_message):
        orthogonal_mp(newX, newy, n_nonzero_coefs=2)


def test_swapped_regressors():
    # [FUNCTION DOC]
    # Verifies selection is correct when the algorithm internally swaps columns: with
    #   true
    # coefficients on atoms 21 (1.0) and 0 (0.5), both orthogonal_mp and
    # orthogonal_mp_gram with 2 atoms return support exactly [0, 21].
    gamma = np.zeros(n_features)
    # X[:, 21] should be selected first, then X[:, 0] selected second,
    # which will take X[:, 21]'s place in case the algorithm does
    # column swapping for optimization (which is the case at the moment)
    gamma[21] = 1.0
    gamma[0] = 0.5
    new_y = np.dot(X, gamma)
    new_Xy = np.dot(X.T, new_y)
    gamma_hat = orthogonal_mp(X, new_y, n_nonzero_coefs=2)
    gamma_hat_gram = orthogonal_mp_gram(G, new_Xy, n_nonzero_coefs=2)
    assert_array_equal(np.flatnonzero(gamma_hat), [0, 21])
    assert_array_equal(np.flatnonzero(gamma_hat_gram), [0, 21])


def test_no_atoms():
    # [FUNCTION DOC]
    # Verifies an all-zero target yields an all-zero solution (warnings ignored).
    # KNOWN ISSUE: the "gram" variant calls orthogonal_mp with G and Xy instead of
    # orthogonal_mp_gram, so the Gram-based solver is not actually exercised.
    y_empty = np.zeros_like(y)
    Xy_empty = np.dot(X.T, y_empty)
    gamma_empty = ignore_warnings(orthogonal_mp)(X, y_empty, n_nonzero_coefs=1)
    gamma_empty_gram = ignore_warnings(orthogonal_mp)(G, Xy_empty, n_nonzero_coefs=1)
    assert np.all(gamma_empty == 0)
    assert np.all(gamma_empty_gram == 0)


def test_omp_path():
    # [FUNCTION DOC]
    # Verifies return_path=True gives an array of shape (n_features, n_targets, 5) whose
    # last step equals the return_path=False solution, for orthogonal_mp and
    # orthogonal_mp_gram.
    path = orthogonal_mp(X, y, n_nonzero_coefs=5, return_path=True)
    last = orthogonal_mp(X, y, n_nonzero_coefs=5, return_path=False)
    assert path.shape == (n_features, n_targets, 5)
    assert_array_almost_equal(path[:, :, -1], last)
    path = orthogonal_mp_gram(G, Xy, n_nonzero_coefs=5, return_path=True)
    last = orthogonal_mp_gram(G, Xy, n_nonzero_coefs=5, return_path=False)
    assert path.shape == (n_features, n_targets, 5)
    assert_array_almost_equal(path[:, :, -1], last)


def test_omp_return_path_prop_with_gram():
    # [FUNCTION DOC]
    # Same path check as test_omp_path for orthogonal_mp with precompute=True.
    path = orthogonal_mp(X, y, n_nonzero_coefs=5, return_path=True, precompute=True)
    last = orthogonal_mp(X, y, n_nonzero_coefs=5, return_path=False, precompute=True)
    assert path.shape == (n_features, n_targets, 5)
    assert_array_almost_equal(path[:, :, -1], last)


def test_omp_cv():
    # [FUNCTION DOC]
    # Verifies OrthogonalMatchingPursuitCV (no intercept, max_iter=10) selects the true
    # number of nonzero coefficients (5), recovers the true code, and matches a plain
    # OrthogonalMatchingPursuit fitted with the selected n_nonzero_coefs_.
    y_ = y[:, 0]
    gamma_ = gamma[:, 0]
    ompcv = OrthogonalMatchingPursuitCV(fit_intercept=False, max_iter=10)
    ompcv.fit(X, y_)
    assert ompcv.n_nonzero_coefs_ == n_nonzero_coefs
    assert_array_almost_equal(ompcv.coef_, gamma_)
    omp = OrthogonalMatchingPursuit(
        fit_intercept=False, n_nonzero_coefs=ompcv.n_nonzero_coefs_
    )
    omp.fit(X, y_)
    assert_array_almost_equal(ompcv.coef_, omp.coef_)


def test_omp_reaches_least_squares():
    # [FUNCTION DOC]
    # Verifies that on random (10, 8) data with 3 targets, OMP allowed to use all 8
    # features reproduces the ordinary least squares coefficients of LinearRegression.
    # Use small simple data; it's a sanity check but OMP can stop early
    rng = check_random_state(0)
    n_samples, n_features = (10, 8)
    n_targets = 3
    X = rng.randn(n_samples, n_features)
    Y = rng.randn(n_samples, n_targets)
    omp = OrthogonalMatchingPursuit(n_nonzero_coefs=n_features)
    lstsq = LinearRegression()
    omp.fit(X, Y)
    lstsq.fit(X, Y)
    assert_array_almost_equal(omp.coef_, lstsq.coef_)


@pytest.mark.parametrize("data_type", (np.float32, np.float64))
def test_omp_gram_dtype_match(data_type):
    # [FUNCTION DOC]
    # Verifies orthogonal_mp_gram preserves the input dtype (float32 or float64).
    # Inputs: data_type - parametrized numpy float dtype.
    # verify matching input data type and output data type
    coef = orthogonal_mp_gram(
        G.astype(data_type), Xy.astype(data_type), n_nonzero_coefs=5
    )
    assert coef.dtype == data_type


def test_omp_gram_numerical_consistency():
    # [FUNCTION DOC]
    # Intended to verify float32 and float64 inputs give close results.
    # KNOWN ISSUE: the "64" call casts G to float32 (only Xy is float64), so the test
    # compares an all-float32 run with a mixed-dtype run rather than pure float64.
    # verify numericaly consistency among np.float32 and np.float64
    coef_32 = orthogonal_mp_gram(
        G.astype(np.float32), Xy.astype(np.float32), n_nonzero_coefs=5
    )
    coef_64 = orthogonal_mp_gram(
        G.astype(np.float32), Xy.astype(np.float64), n_nonzero_coefs=5
    )
    assert_allclose(coef_32, coef_64)
