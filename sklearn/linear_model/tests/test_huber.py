# [MODULE DOC]
# Tests for HuberRegressor: equivalence to OLS for large epsilon, max_iter, the analytic
# gradient of _huber_loss_and_gradient, sample weights, sparse input, scale invariance
# of outlier detection, agreement with SGD huber loss, warm start, robustness versus
# Ridge, and boolean input.
# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest
from scipy import optimize

from sklearn.datasets import make_regression
from sklearn.linear_model import HuberRegressor, LinearRegression, Ridge, SGDRegressor
from sklearn.linear_model._huber import _huber_loss_and_gradient
from sklearn.utils._testing import (
    assert_almost_equal,
    assert_array_almost_equal,
    assert_array_equal,
)
from sklearn.utils.fixes import CSR_CONTAINERS


def make_regression_with_outliers(n_samples=50, n_features=20):
    # [FUNCTION DOC]
    # HELPER: Builds a regression problem with a few corrupted rows. Generates data with
    # make_regression (random_state=0, noise=0.05), then overwrites the features of
    # int(0.1 * n_samples) randomly chosen rows (with replacement, so possibly fewer
    # distinct rows) with Gaussian noise of std 2; targets are left untouched.
    # Inputs: n_samples (default 50), n_features (default 20).
    # Outputs: X array (n_samples, n_features), y array (n_samples,).
    # Side effects: none beyond a local RandomState(0); deterministic.
    rng = np.random.RandomState(0)
    # Generate data with outliers by replacing 10% of the samples with noise.
    X, y = make_regression(
        n_samples=n_samples, n_features=n_features, random_state=0, noise=0.05
    )

    # Replace 10% of the sample with noise.
    num_noise = int(0.1 * n_samples)
    random_samples = rng.randint(0, n_samples, num_noise)
    X[random_samples, :] = 2.0 * rng.normal(0, 1, (num_noise, X.shape[1]))
    return X, y


def test_huber_equals_lr_for_high_epsilon():
    # [FUNCTION DOC]
    # Verifies that with epsilon=1e3 and alpha=0 (no sample is an outlier, no penalty)
    # HuberRegressor matches LinearRegression coef_ (3 decimals) and intercept_ (2).
    # KNOWN ISSUE: the inline comment says "Ridge" but the comparison is Huber versus
    # LinearRegression.
    # Test that Ridge matches LinearRegression for large epsilon
    X, y = make_regression_with_outliers()
    lr = LinearRegression()
    lr.fit(X, y)
    huber = HuberRegressor(epsilon=1e3, alpha=0.0)
    huber.fit(X, y)
    assert_almost_equal(huber.coef_, lr.coef_, 3)
    assert_almost_equal(huber.intercept_, lr.intercept_, 2)


def test_huber_max_iter():
    # [FUNCTION DOC]
    # Verifies n_iter_ equals max_iter when max_iter=1 limits the optimizer.
    X, y = make_regression_with_outliers()
    huber = HuberRegressor(max_iter=1)
    huber.fit(X, y)
    assert huber.n_iter_ == huber.max_iter


def test_huber_gradient():
    # [FUNCTION DOC]
    # Verifies the gradient returned by _huber_loss_and_gradient against finite
    # differences with scipy check_grad. For 5 random draws and parameter vectors of
    # length n_features+1 (no intercept) and n_features+2 (with intercept), where the
    # last entry (scale) is made positive, calls with epsilon=0.01, alpha=0.1 and random
    # integer sample weights in {1, 2}; asserts the gradient error is ~0 (4 decimals).
    # Test that the gradient calculated by _huber_loss_and_gradient is correct
    rng = np.random.RandomState(1)
    X, y = make_regression_with_outliers()
    sample_weight = rng.randint(1, 3, (y.shape[0]))

    def loss_func(x, *args):
        # [FUNCTION DOC]
        # HELPER: Returns only the loss value of _huber_loss_and_gradient for parameters
        # x and extra positional args (X, y, epsilon, alpha, sample_weight).
        return _huber_loss_and_gradient(x, *args)[0]

    def grad_func(x, *args):
        # [FUNCTION DOC]
        # HELPER: Returns only the gradient array of _huber_loss_and_gradient for the
        # same arguments as loss_func.
        return _huber_loss_and_gradient(x, *args)[1]

    # Check using optimize.check_grad that the gradients are equal.
    for _ in range(5):
        # Check for both fit_intercept and otherwise.
        for n_features in [X.shape[1] + 1, X.shape[1] + 2]:
            w = rng.randn(n_features)
            w[-1] = np.abs(w[-1])
            grad_same = optimize.check_grad(
                loss_func, grad_func, w, X, y, 0.01, 0.1, sample_weight
            )
            assert_almost_equal(grad_same, 1e-6, 4)


@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_huber_sample_weights(csr_container):
    # [FUNCTION DOC]
    # Verifies sample_weight semantics: unit weights reproduce the unweighted fit;
    # integer weights on rows 1 and 3 match a fit where those rows are duplicated (on a
    # 5-sample problem); a CSR fit with those weights matches too. Coefficients are
    # divided by a scale computed from the first fit to make the decimal comparison
    # scale-insensitive.
    # Inputs: csr_container from parametrize.
    # Test sample_weights implementation in HuberRegressor"""

    X, y = make_regression_with_outliers()
    huber = HuberRegressor()
    huber.fit(X, y)
    huber_coef = huber.coef_
    huber_intercept = huber.intercept_

    # Rescale coefs before comparing with assert_array_almost_equal to make
    # sure that the number of decimal places used is somewhat insensitive to
    # the amplitude of the coefficients and therefore to the scale of the
    # data and the regularization parameter
    scale = max(np.mean(np.abs(huber.coef_)), np.mean(np.abs(huber.intercept_)))

    huber.fit(X, y, sample_weight=np.ones(y.shape[0]))
    assert_array_almost_equal(huber.coef_ / scale, huber_coef / scale)
    assert_array_almost_equal(huber.intercept_ / scale, huber_intercept / scale)

    X, y = make_regression_with_outliers(n_samples=5, n_features=20)
    X_new = np.vstack((X, np.vstack((X[1], X[1], X[3]))))
    y_new = np.concatenate((y, [y[1]], [y[1]], [y[3]]))
    huber.fit(X_new, y_new)
    huber_coef = huber.coef_
    huber_intercept = huber.intercept_
    sample_weight = np.ones(X.shape[0])
    sample_weight[1] = 3
    sample_weight[3] = 2
    huber.fit(X, y, sample_weight=sample_weight)

    assert_array_almost_equal(huber.coef_ / scale, huber_coef / scale)
    assert_array_almost_equal(huber.intercept_ / scale, huber_intercept / scale)

    # Test sparse implementation with sample weights.
    X_csr = csr_container(X)
    huber_sparse = HuberRegressor()
    huber_sparse.fit(X_csr, y, sample_weight=sample_weight)
    assert_array_almost_equal(huber_sparse.coef_ / scale, huber_coef / scale)


@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_huber_sparse(csr_container):
    # [FUNCTION DOC]
    # Verifies dense and CSR input give the same coef_ and identical outliers_ masks
    # (alpha=0.1).
    X, y = make_regression_with_outliers()
    huber = HuberRegressor(alpha=0.1)
    huber.fit(X, y)

    X_csr = csr_container(X)
    huber_sparse = HuberRegressor(alpha=0.1)
    huber_sparse.fit(X_csr, y)
    assert_array_almost_equal(huber_sparse.coef_, huber.coef_)
    assert_array_equal(huber.outliers_, huber_sparse.outliers_)


def test_huber_scaling_invariant():
    # [FUNCTION DOC]
    # Verifies outliers_ (not all True) is unchanged when y is doubled and when both X
    # and y are doubled, with no intercept and no penalty.
    # Test that outliers filtering is scaling independent.
    X, y = make_regression_with_outliers()
    huber = HuberRegressor(fit_intercept=False, alpha=0.0)
    huber.fit(X, y)
    n_outliers_mask_1 = huber.outliers_
    assert not np.all(n_outliers_mask_1)

    huber.fit(X, 2.0 * y)
    n_outliers_mask_2 = huber.outliers_
    assert_array_equal(n_outliers_mask_2, n_outliers_mask_1)

    huber.fit(2.0 * X, 2.0 * y)
    n_outliers_mask_3 = huber.outliers_
    assert_array_equal(n_outliers_mask_3, n_outliers_mask_1)


def test_huber_and_sgd_same_results():
    # [FUNCTION DOC]
    # Verifies HuberRegressor and SGDRegressor(loss="huber") agree to 1 decimal on
    # coef_. First fit estimates scale_, X and y are divided by it and refit so scale_
    # is ~1 (checked), since SGD's huber loss has no scale parameter; SGD runs with no
    # penalty, tol=None and 10000 iterations.
    # Test they should converge to same coefficients for same parameters

    X, y = make_regression_with_outliers(n_samples=10, n_features=2)

    # Fit once to find out the scale parameter. Scale down X and y by scale
    # so that the scale parameter is optimized to 1.0
    huber = HuberRegressor(fit_intercept=False, alpha=0.0, epsilon=1.35)
    huber.fit(X, y)
    X_scale = X / huber.scale_
    y_scale = y / huber.scale_
    huber.fit(X_scale, y_scale)
    assert_almost_equal(huber.scale_, 1.0, 3)

    sgdreg = SGDRegressor(
        alpha=0.0,
        loss="huber",
        shuffle=True,
        random_state=0,
        max_iter=10000,
        fit_intercept=False,
        epsilon=1.35,
        tol=None,
    )
    sgdreg.fit(X_scale, y_scale)
    assert_array_almost_equal(huber.coef_, sgdreg.coef_, 1)


def test_huber_warm_start():
    # [FUNCTION DOC]
    # Verifies warm_start: a second fit on the same data starts from the previous
    # solution, reports n_iter_ == 0 and leaves coef_ almost unchanged (1 decimal).
    X, y = make_regression_with_outliers()
    huber_warm = HuberRegressor(alpha=1.0, max_iter=10000, warm_start=True, tol=1e-1)

    huber_warm.fit(X, y)
    huber_warm_coef = huber_warm.coef_.copy()
    huber_warm.fit(X, y)

    # SciPy performs the tol check after doing the coef updates, so
    # these would be almost same but not equal.
    assert_array_almost_equal(huber_warm.coef_, huber_warm_coef, 1)

    assert huber_warm.n_iter_ == 0


def test_huber_better_r2_score():
    # [FUNCTION DOC]
    # Verifies robustness: using the Huber inlier mask (absolute residual below epsilon
    # * scale_), Huber scores higher R2 than Ridge on inliers, while Ridge scores higher
    # on the outliers.
    # Test that huber returns a better r2 score than non-outliers"""
    X, y = make_regression_with_outliers()
    huber = HuberRegressor(alpha=0.01)
    huber.fit(X, y)
    linear_loss = np.dot(X, huber.coef_) + huber.intercept_ - y
    mask = np.abs(linear_loss) < huber.epsilon * huber.scale_
    huber_score = huber.score(X[mask], y[mask])
    huber_outlier_score = huber.score(X[~mask], y[~mask])

    # The Ridge regressor should be influenced by the outliers and hence
    # give a worse score on the non-outliers as compared to the huber
    # regressor.
    ridge = Ridge(alpha=0.01)
    ridge.fit(X, y)
    ridge_score = ridge.score(X[mask], y[mask])
    ridge_outlier_score = ridge.score(X[~mask], y[~mask])
    assert huber_score > ridge_score

    # The huber model should also fit poorly on the outliers.
    assert ridge_outlier_score > huber_outlier_score


def test_huber_bool():
    # [FUNCTION DOC]
    # Verifies fit does not raise on boolean feature arrays (smoke test).
    # Test that it does not crash with bool data
    X, y = make_regression(n_samples=200, n_features=2, noise=4.0, random_state=0)
    X_bool = X > 0
    HuberRegressor().fit(X_bool, y)
