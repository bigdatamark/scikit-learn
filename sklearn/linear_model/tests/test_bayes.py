# [MODULE DOC]
# Tests for the Bayesian linear regressors BayesianRidge and ARDRegression: score
# (log marginal likelihood) values, hyperparameter correctness versus Ridge, posterior
# covariance sigma_, sample weights, predictive std (return_std), constant-target
# edge cases, the two ARD sigma update helpers, and dtype preservation.
# Side effects: loads the diabetes dataset at import time into module global diabetes.
# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

from math import log

import numpy as np
import pytest

from sklearn import datasets
from sklearn.linear_model import ARDRegression, BayesianRidge, Ridge
from sklearn.utils import check_random_state
from sklearn.utils._testing import (
    _convert_container,
    assert_allclose,
    assert_almost_equal,
    assert_array_almost_equal,
    assert_array_less,
)
from sklearn.utils.extmath import fast_logdet

diabetes = datasets.load_diabetes()


def test_bayesian_ridge_scores():
    # [FUNCTION DOC]
    # Verifies that with compute_score=True the scores_ attribute has one entry per
    # iteration plus the initial one, i.e. shape (n_iter_ + 1,), on the diabetes data.
    """Check scores attribute shape"""
    X, y = diabetes.data, diabetes.target

    clf = BayesianRidge(compute_score=True)
    clf.fit(X, y)

    assert clf.scores_.shape == (clf.n_iter_ + 1,)


def test_bayesian_ridge_score_values():
    # [FUNCTION DOC]
    # Verifies the first stored score equals a log marginal likelihood computed by hand.
    # Builds initial alpha_ = 1/(var(y)+eps) and lambda_ = 1, the gamma hyperprior
    #   terms,
    # and the Gaussian term using M = I/alpha + X X^T/lambda (log-determinant via
    # fast_logdet, quadratic form via a linear solve). Fits BayesianRidge with
    #   max_iter=1,
    # no intercept, and compares scores_[0] to 9 decimals.
    """Check value of score on toy example.

    Compute log marginal likelihood with equation (36) in Sparse Bayesian
    Learning and the Relevance Vector Machine (Tipping, 2001):

    - 0.5 * (log |Id/alpha + X.X^T/lambda| +
             y^T.(Id/alpha + X.X^T/lambda).y + n * log(2 * pi))
    + lambda_1 * log(lambda) - lambda_2 * lambda
    + alpha_1 * log(alpha) - alpha_2 * alpha

    and check equality with the score computed during training.
    """

    X, y = diabetes.data, diabetes.target
    n_samples = X.shape[0]
    # check with initial values of alpha and lambda (see code for the values)
    eps = np.finfo(np.float64).eps
    alpha_ = 1.0 / (np.var(y) + eps)
    lambda_ = 1.0

    # value of the parameters of the Gamma hyperpriors
    alpha_1 = 0.1
    alpha_2 = 0.1
    lambda_1 = 0.1
    lambda_2 = 0.1

    # compute score using formula of docstring
    score = lambda_1 * log(lambda_) - lambda_2 * lambda_
    score += alpha_1 * log(alpha_) - alpha_2 * alpha_
    M = 1.0 / alpha_ * np.eye(n_samples) + 1.0 / lambda_ * np.dot(X, X.T)
    M_inv_dot_y = np.linalg.solve(M, y)
    score += -0.5 * (
        fast_logdet(M) + np.dot(y.T, M_inv_dot_y) + n_samples * log(2 * np.pi)
    )

    # compute score with BayesianRidge
    clf = BayesianRidge(
        alpha_1=alpha_1,
        alpha_2=alpha_2,
        lambda_1=lambda_1,
        lambda_2=lambda_2,
        max_iter=1,
        fit_intercept=False,
        compute_score=True,
    )
    clf.fit(X, y)

    assert_almost_equal(clf.scores_[0], score, decimal=9)


def test_bayesian_ridge_parameter():
    # [FUNCTION DOC]
    # Verifies lambda_/alpha_ consistency (issue #8224): a Ridge with alpha equal to
    # lambda_ / alpha_ of a fitted BayesianRidge must yield the same coef_ and
    #   intercept_.
    # Test correctness of lambda_ and alpha_ parameters (GitHub issue #8224)
    X = np.array([[1, 1], [3, 4], [5, 7], [4, 1], [2, 6], [3, 10], [3, 2]])
    y = np.array([1, 2, 3, 2, 0, 4, 5]).T

    # A Ridge regression model using an alpha value equal to the ratio of
    # lambda_ and alpha_ from the Bayesian Ridge model must be identical
    br_model = BayesianRidge(compute_score=True).fit(X, y)
    rr_model = Ridge(alpha=br_model.lambda_ / br_model.alpha_).fit(X, y)
    assert_array_almost_equal(rr_model.coef_, br_model.coef_)
    assert_almost_equal(rr_model.intercept_, br_model.intercept_)


@pytest.mark.parametrize("n_samples, n_features", [(10, 20), (20, 10)])
def test_bayesian_covariance_matrix(n_samples, n_features, global_random_seed):
    # [FUNCTION DOC]
    # Verifies sigma_ equals the inverse of (lambda_ I + alpha_ X^T X) for a model fit
    # without intercept, on random regression data with n_samples above and below
    # n_features (issue #31093). Inputs: n_samples, n_features from parametrize;
    # global_random_seed fixture.
    """Check the posterior covariance matrix sigma_

    Non-regression test for https://github.com/scikit-learn/scikit-learn/issues/31093
    """
    X, y = datasets.make_regression(
        n_samples, n_features, random_state=global_random_seed
    )
    reg = BayesianRidge(fit_intercept=False).fit(X, y)
    covariance_matrix = np.linalg.inv(
        reg.lambda_ * np.identity(n_features) + reg.alpha_ * np.dot(X.T, X)
    )
    assert_allclose(reg.sigma_, covariance_matrix, rtol=1e-6)


def test_bayesian_sample_weights():
    # [FUNCTION DOC]
    # Same Ridge-equivalence check as test_bayesian_ridge_parameter, but with integer
    # sample weights passed to both fits.
    # Test correctness of the sample_weights method
    X = np.array([[1, 1], [3, 4], [5, 7], [4, 1], [2, 6], [3, 10], [3, 2]])
    y = np.array([1, 2, 3, 2, 0, 4, 5]).T
    w = np.array([4, 3, 3, 1, 1, 2, 3]).T

    # A Ridge regression model using an alpha value equal to the ratio of
    # lambda_ and alpha_ from the Bayesian Ridge model must be identical
    br_model = BayesianRidge(compute_score=True).fit(X, y, sample_weight=w)
    rr_model = Ridge(alpha=br_model.lambda_ / br_model.alpha_).fit(
        X, y, sample_weight=w
    )
    assert_array_almost_equal(rr_model.coef_, br_model.coef_)
    assert_almost_equal(rr_model.intercept_, br_model.intercept_)


def test_toy_bayesian_ridge_object():
    # [FUNCTION DOC]
    # Verifies BayesianRidge approximately learns the identity function on a 1-feature
    # toy set: predictions on [1, 3, 4] match to 2 decimals.
    # Test BayesianRidge on toy
    X = np.array([[1], [2], [6], [8], [10]])
    Y = np.array([1, 2, 6, 8, 10])
    clf = BayesianRidge(compute_score=True)
    clf.fit(X, Y)

    # Check that the model could approximately learn the identity function
    test = [[1], [3], [4]]
    assert_array_almost_equal(clf.predict(test), [1, 3, 4], 2)


def test_bayesian_initial_params():
    # [FUNCTION DOC]
    # Verifies alpha_init / lambda_init are used: on a cubic polynomial target with
    # Vandermonde features and a small lambda_init, the training R2 is almost 1.
    # Test BayesianRidge with initial values (alpha_init, lambda_init)
    X = np.vander(np.linspace(0, 4, 5), 4)
    y = np.array([0.0, 1.0, 0.0, -1.0, 0.0])  # y = (x^3 - 6x^2 + 8x) / 3

    # In this case, starting from the default initial values will increase
    # the bias of the fitted curve. So, lambda_init should be small.
    reg = BayesianRidge(alpha_init=1.0, lambda_init=1e-3)
    # Check the R2 score nearly equals to one.
    r2 = reg.fit(X, y).score(X, y)
    assert_almost_equal(r2, 1.0)


def test_prediction_bayesian_ridge_ard_with_constant_input(global_random_seed):
    # [FUNCTION DOC]
    # Verifies both BayesianRidge and ARDRegression predict the constant value exactly
    # (approximately) when the target vector is constant, on random 4x5 features.
    # Test BayesianRidge and ARDRegression predictions for edge case of
    # constant target vectors
    n_samples = 4
    n_features = 5
    random_state = check_random_state(global_random_seed)
    constant_value = random_state.rand()
    X = random_state.random_sample((n_samples, n_features))
    y = np.full(n_samples, constant_value, dtype=np.array(constant_value).dtype)
    expected = np.full(n_samples, constant_value, dtype=np.array(constant_value).dtype)

    for clf in [BayesianRidge(), ARDRegression()]:
        y_pred = clf.fit(X, y).predict(X)
        assert_array_almost_equal(y_pred, expected)


def test_std_bayesian_ridge_ard_with_constant_input(global_random_seed):
    # [FUNCTION DOC]
    # Verifies the predictive std returned with return_std=True is below 0.01 for both
    # estimators when trained on a constant target.
    # Test BayesianRidge and ARDRegression standard dev. for edge case of
    # constant target vector
    # The standard dev. should be relatively small (< 0.01 is tested here)
    n_samples = 10
    n_features = 5
    random_state = check_random_state(global_random_seed)
    constant_value = random_state.rand()
    X = random_state.random_sample((n_samples, n_features))
    y = np.full(n_samples, constant_value, dtype=np.array(constant_value).dtype)
    expected_upper_boundary = 0.01

    for clf in [BayesianRidge(), ARDRegression()]:
        _, y_std = clf.fit(X, y).predict(X, return_std=True)
        assert_array_less(y_std, expected_upper_boundary)


@pytest.mark.parametrize("Estimator", [BayesianRidge, ARDRegression])
def test_std_bayesian_ridge_noncentered(Estimator, global_random_seed):
    # [FUNCTION DOC]
    # Verifies predictive std is smaller at the data center (x=90) than at the origin
    #   for
    # non-centered 1-D training data in [80, 100] with intercept (issue #33757).
    # Inputs: Estimator class from parametrize; global_random_seed fixture.
    # Test BayesianRidge and ARDRegression std when data is not centered.
    # The std should be smallest at the center of the data, not at the origin.
    # Non-regression test for issue #33757
    rng = np.random.RandomState(global_random_seed)
    n_samples = 4
    X_train = np.linspace(80, 100, n_samples).reshape(-1, 1)
    y_train = X_train.reshape(-1) + 10 * rng.standard_normal(n_samples)
    model = Estimator(fit_intercept=True).fit(X_train, y_train)
    X = np.array([[0.0], [90.0]])
    _, y_std = model.predict(X, return_std=True)
    assert y_std[1] < y_std[0]


def test_update_of_sigma_in_ard():
    # [FUNCTION DOC]
    # Verifies ARDRegression updates sigma_ after the final iteration: with max_iter=1
    #   and
    # data that prunes both coefficients, sigma_ has shape (0, 0), and predict with
    # return_std=True does not raise.
    # Checks that `sigma_` is updated correctly after the last iteration
    # of the ARDRegression algorithm. See issue #10128.
    X = np.array([[1, 0], [0, 0]])
    y = np.array([0, 0])
    clf = ARDRegression(max_iter=1)
    clf.fit(X, y)
    # With the inputs above, ARDRegression prunes both of the two coefficients
    # in the first iteration. Hence, the expected shape of `sigma_` is (0, 0).
    assert clf.sigma_.shape == (0, 0)
    # Ensure that no error is thrown at prediction stage
    clf.predict(X, return_std=True)


def test_toy_ard_object():
    # [FUNCTION DOC]
    # Verifies ARDRegression approximately learns the identity on a 3-point toy set
    # (predictions on [1, 3, 4] to 2 decimals).
    # KNOWN ISSUE: the inline comment calls ARD a "classifier"; it is a regressor.
    # Test BayesianRegression ARD classifier
    X = np.array([[1], [2], [3]])
    Y = np.array([1, 2, 3])
    clf = ARDRegression(compute_score=True)
    clf.fit(X, Y)

    # Check that the model could approximately learn the identity function
    test = [[1], [3], [4]]
    assert_array_almost_equal(clf.predict(test), [1, 3, 4], 2)


@pytest.mark.parametrize("n_samples, n_features", ((10, 100), (100, 10)))
def test_ard_accuracy_on_easy_problem(global_random_seed, n_samples, n_features):
    # [FUNCTION DOC]
    # Verifies ARD recovers coef_[1] = 1 within 1e-10 when y equals the second of three
    # Gaussian features (issue #14055).
    # KNOWN ISSUE: the parametrized n_samples and n_features are ignored; the data is
    # always 250x3, so both parametrizations run the same test.
    # Check that ARD converges with reasonable accuracy on an easy problem
    # (Github issue #14055)
    X = np.random.RandomState(global_random_seed).normal(size=(250, 3))
    y = X[:, 1]

    regressor = ARDRegression()
    regressor.fit(X, y)

    abs_coef_error = np.abs(1 - regressor.coef_[1])
    assert abs_coef_error < 1e-10


@pytest.mark.parametrize("constructor_name", ["array", "pandas"])
def test_return_std(constructor_name, global_random_seed):
    # [FUNCTION DOC]
    # Verifies predictive std of both BayesianRidge and ARDRegression approximates the
    # known noise level. For noise multipliers 1, 0.1, 0.01, generates linear targets
    #   with
    # fixed weights plus Gaussian noise, fits each model and checks the std on test
    #   points
    # equals the multiplier to an increasing number of decimals (0, 1, 2). Inputs:
    # constructor_name selects numpy array or pandas DataFrame input.
    # Test return_std option for both Bayesian regressors
    rng = np.random.RandomState(global_random_seed)

    def f(X):
        # [FUNCTION DOC]
        # HELPER: Noise-free linear target: dot product of X with closure weight w plus
        #   b.
        # Inputs: X - array (n, 5). Outputs: array (n,).
        return np.dot(X, w) + b

    def f_noise(X, noise_mult):
        # [FUNCTION DOC]
        # HELPER: Linear target f(X) plus standard normal noise scaled by noise_mult.
        # Inputs: X - array (n, 5); noise_mult - float. Outputs: array (n,).
        # Side effects: consumes the enclosing rng.
        return f(X) + rng.randn(X.shape[0]) * noise_mult

    d = 5
    n_train = 50
    n_test = 10

    w = np.array([1.0, 0.0, 1.0, -1.0, 0.0])
    b = 1.0

    X = rng.random_sample((n_train, d))
    X = _convert_container(X, constructor_name)

    X_test = rng.random_sample((n_test, d))
    X_test = _convert_container(X_test, constructor_name)

    for decimal, noise_mult in enumerate([1, 0.1, 0.01]):
        y = f_noise(X, noise_mult)

        m1 = BayesianRidge()
        m1.fit(X, y)
        y_mean1, y_std1 = m1.predict(X_test, return_std=True)
        assert_array_almost_equal(y_std1, noise_mult, decimal=decimal)

        m2 = ARDRegression()
        m2.fit(X, y)
        y_mean2, y_std2 = m2.predict(X_test, return_std=True)
        assert_array_almost_equal(y_std2, noise_mult, decimal=decimal)


def test_update_sigma(global_random_seed):
    # [FUNCTION DOC]
    # Verifies the private ARD helpers _update_sigma and _update_sigma_woodbury give the
    # same posterior covariance on a random square 10x10 X with alpha=1, increasing
    # lambdas and all features kept.
    # make sure the two update_sigma() helpers are equivalent. The woodbury
    # formula is used when n_samples < n_features, and the other one is used
    # otherwise.

    rng = np.random.RandomState(global_random_seed)

    # set n_samples == n_features to avoid instability issues when inverting
    # the matrices. Using the woodbury formula would be unstable when
    # n_samples > n_features
    n_samples = n_features = 10
    X = rng.randn(n_samples, n_features)
    alpha = 1
    lmbda = np.arange(1, n_features + 1)
    keep_lambda = np.array([True] * n_features)

    reg = ARDRegression()

    sigma = reg._update_sigma(X, alpha, lmbda, keep_lambda)
    sigma_woodbury = reg._update_sigma_woodbury(X, alpha, lmbda, keep_lambda)

    np.testing.assert_allclose(sigma, sigma_woodbury)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("Estimator", [BayesianRidge, ARDRegression])
def test_dtype_match(dtype, Estimator):
    # [FUNCTION DOC]
    # Verifies float32 input is not upcast: coef_, sigma_, predicted mean and std all
    # share X's dtype, for both estimators and both float dtypes.
    # Test that np.float32 input data is not cast to np.float64 when possible
    X = np.array([[1, 1], [3, 4], [5, 7], [4, 1], [2, 6], [3, 10], [3, 2]], dtype=dtype)
    y = np.array([1, 2, 3, 2, 0, 4, 5]).T

    model = Estimator()
    # check type consistency
    model.fit(X, y)
    attributes = ["coef_", "sigma_"]
    for attribute in attributes:
        assert getattr(model, attribute).dtype == X.dtype

    y_mean, y_std = model.predict(X, return_std=True)
    assert y_mean.dtype == X.dtype
    assert y_std.dtype == X.dtype


@pytest.mark.parametrize("Estimator", [BayesianRidge, ARDRegression])
def test_dtype_correctness(Estimator):
    # [FUNCTION DOC]
    # Verifies coef_ fitted on float32 and float64 versions of the same data agree to
    # rtol 1e-4, for both estimators.
    X = np.array([[1, 1], [3, 4], [5, 7], [4, 1], [2, 6], [3, 10], [3, 2]])
    y = np.array([1, 2, 3, 2, 0, 4, 5]).T
    model = Estimator()
    coef_32 = model.fit(X.astype(np.float32), y).coef_
    coef_64 = model.fit(X.astype(np.float64), y).coef_
    np.testing.assert_allclose(coef_32, coef_64, rtol=1e-4)
