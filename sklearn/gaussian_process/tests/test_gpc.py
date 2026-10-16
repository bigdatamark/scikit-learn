# [MODULE DOC]
# Tests for GaussianProcessClassifier: predict/predict_proba consistency, log-marginal
# likelihood (LML) optimization, caching and gradients, random restarts, custom
# optimizers, multi-class (incl. n_jobs), bound-proximity warnings, fit errors and the
# latent_mean_and_variance API. Uses a 1D toy dataset X (30 points) with binary labels
# from sign(sin(x)) and 3-class labels from thresholds on sin(x), and a list of kernels
# including one with fixed hyperparameters.
"""Testing for Gaussian process classification"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import warnings

import numpy as np
import pytest
from scipy.optimize import approx_fprime

from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import (
    RBF,
    CompoundKernel,
    WhiteKernel,
)
from sklearn.gaussian_process.kernels import (
    ConstantKernel as C,
)
from sklearn.gaussian_process.tests._mini_sequence_kernel import MiniSeqKernel
from sklearn.utils._testing import (
    assert_allclose,
    assert_almost_equal,
    assert_array_equal,
)


def f(x):
    # [FUNCTION DOC]
    # HELPER: Target function used to build labels; returns sin(x) element-wise.
    return np.sin(x)


X = np.atleast_2d(np.linspace(0, 10, 30)).T
X2 = np.atleast_2d([2.0, 4.0, 5.5, 6.5, 7.5]).T
y = np.array(f(X).ravel() > 0, dtype=int)
fX = f(X).ravel()
y_mc = np.empty(y.shape, dtype=int)  # multi-class
y_mc[fX < -0.35] = 0
y_mc[(fX >= -0.35) & (fX < 0.35)] = 1
y_mc[fX > 0.35] = 2


fixed_kernel = RBF(length_scale=1.0, length_scale_bounds="fixed")
kernels = [
    RBF(length_scale=0.1),
    fixed_kernel,
    RBF(length_scale=1.0, length_scale_bounds=(1e-3, 1e3)),
    C(1.0, (1e-2, 1e2)) * RBF(length_scale=1.0, length_scale_bounds=(1e-3, 1e3)),
]
non_fixed_kernels = [kernel for kernel in kernels if kernel != fixed_kernel]


@pytest.mark.parametrize("kernel", kernels)
def test_predict_consistent(kernel):
    # [FUNCTION DOC]
    # Verifies that for binary problems predict equals predict_proba[:, 1] >= 0.5, for
    # each kernel in kernels.
    # Check binary predict decision has also predicted probability above 0.5.
    gpc = GaussianProcessClassifier(kernel=kernel).fit(X, y)
    assert_array_equal(gpc.predict(X), gpc.predict_proba(X)[:, 1] >= 0.5)


def test_predict_consistent_structured():
    # [FUNCTION DOC]
    # Same predict/predict_proba consistency check with string inputs and the fixed
    # MiniSeqKernel, i.e. GPC on structured (non-vector) data.
    # Check binary predict decision has also predicted probability above 0.5.
    X = ["A", "AB", "B"]
    y = np.array([True, False, True])
    kernel = MiniSeqKernel(baseline_similarity_bounds="fixed")
    gpc = GaussianProcessClassifier(kernel=kernel).fit(X, y)
    assert_array_equal(gpc.predict(X), gpc.predict_proba(X)[:, 1] >= 0.5)


@pytest.mark.parametrize("kernel", non_fixed_kernels)
def test_lml_improving(kernel):
    # [FUNCTION DOC]
    # Verifies that the LML at the fitted kernel_.theta is greater than at the initial
    # kernel.theta (optimization improves the objective), for non-fixed kernels.
    # Test that hyperparameter-tuning improves log-marginal likelihood.
    gpc = GaussianProcessClassifier(kernel=kernel).fit(X, y)
    assert gpc.log_marginal_likelihood(gpc.kernel_.theta) > gpc.log_marginal_likelihood(
        kernel.theta
    )


@pytest.mark.parametrize("kernel", kernels)
def test_lml_precomputed(kernel):
    # [FUNCTION DOC]
    # Verifies log_marginal_likelihood() with no theta returns the stored value equal to
    # the LML recomputed at kernel_.theta (7 decimals).
    # Test that lml of optimized kernel is stored correctly.
    gpc = GaussianProcessClassifier(kernel=kernel).fit(X, y)
    assert_almost_equal(
        gpc.log_marginal_likelihood(gpc.kernel_.theta), gpc.log_marginal_likelihood(), 7
    )


@pytest.mark.parametrize("kernel", kernels)
def test_lml_without_cloning_kernel(kernel):
    # [FUNCTION DOC]
    # Verifies that calling log_marginal_likelihood with clone_kernel=False overwrites
    # kernel_.theta with the given theta (all ones).
    # Test that clone_kernel=False has side-effects of kernel.theta.
    gpc = GaussianProcessClassifier(kernel=kernel).fit(X, y)
    input_theta = np.ones(gpc.kernel_.theta.shape, dtype=np.float64)

    gpc.log_marginal_likelihood(input_theta, clone_kernel=False)
    assert_almost_equal(gpc.kernel_.theta, input_theta, 7)


@pytest.mark.parametrize("kernel", non_fixed_kernels)
def test_converged_to_local_maximum(kernel):
    # [FUNCTION DOC]
    # Verifies a local optimum after fitting: every LML gradient component is below 1e-4
    # in absolute value or the corresponding theta sits exactly on a bound.
    # Test that we are in local maximum after hyperparameter-optimization.
    gpc = GaussianProcessClassifier(kernel=kernel).fit(X, y)

    lml, lml_gradient = gpc.log_marginal_likelihood(gpc.kernel_.theta, True)

    assert np.all(
        (np.abs(lml_gradient) < 1e-4)
        | (gpc.kernel_.theta == gpc.kernel_.bounds[:, 0])
        | (gpc.kernel_.theta == gpc.kernel_.bounds[:, 1])
    )


@pytest.mark.xfail(
    raises=AssertionError,
    reason="https://github.com/scikit-learn/scikit-learn/issues/31366",
)
@pytest.mark.parametrize("kernel", non_fixed_kernels)
@pytest.mark.parametrize("length_scale", np.logspace(-3, 3, 13))
def test_lml_gradient(kernel, length_scale):
    # [FUNCTION DOC]
    # Compares the analytic LML gradient with a scipy approx_fprime numerical gradient
    # (epsilon 1e-9) after setting the kernel's length scale to each value of a log
    # grid. Marked xfail (AssertionError) due to issue 31366.
    # Side effects: clones the kernel first to avoid mutating the shared module-level
    # one.
    # Clone the kernel object prior to mutating it to avoid any side effects between
    # GP tests:
    kernel = clone(kernel)
    length_scale_param_name = next(
        name for name in kernel.get_params() if name.endswith("length_scale")
    )
    kernel.set_params(**{length_scale_param_name: length_scale})

    # Compare analytic and numeric gradient of log marginal likelihood.
    gpr = GaussianProcessClassifier(kernel=kernel).fit(X, y)
    _, lml_gradient = gpr.log_marginal_likelihood(kernel.theta, eval_gradient=True)
    epsilon = 1e-9
    lml_gradient_approx = approx_fprime(
        kernel.theta.copy(),
        lambda theta: gpr.log_marginal_likelihood(theta, False),
        epsilon=epsilon,
    )
    assert_allclose(lml_gradient, lml_gradient_approx, rtol=1e-4, atol=epsilon * 100)


def test_random_starts(global_random_seed):
    # [FUNCTION DOC]
    # Verifies that increasing n_restarts_optimizer from 0 to 4 never decreases the LML
    # of the selected theta (up to float32 eps), on random 2-feature data with an
    # anisotropic RBF times constant kernel. Uses global_random_seed.
    # Test that an increasing number of random-starts of GP fitting only
    # increases the log marginal likelihood of the chosen theta.
    n_samples, n_features = 25, 2
    rng = np.random.RandomState(global_random_seed)
    X = rng.randn(n_samples, n_features) * 2 - 1
    y = (np.sin(X).sum(axis=1) + np.sin(3 * X).sum(axis=1)) > 0

    kernel = C(1.0, (1e-2, 1e2)) * RBF(
        length_scale=[1e-3] * n_features, length_scale_bounds=[(1e-4, 1e2)] * n_features
    )
    last_lml = -np.inf
    for n_restarts_optimizer in range(5):
        gp = GaussianProcessClassifier(
            kernel=kernel,
            n_restarts_optimizer=n_restarts_optimizer,
            random_state=global_random_seed,
        ).fit(X, y)
        lml = gp.log_marginal_likelihood(gp.kernel_.theta)
        assert lml > last_lml - np.finfo(np.float32).eps
        last_lml = lml


@pytest.mark.parametrize("kernel", non_fixed_kernels)
def test_custom_optimizer(kernel, global_random_seed):
    # [FUNCTION DOC]
    # Verifies GPC accepts a user-supplied optimizer callable on the multi-class problem
    # and that the result does not lower the LML compared to the initial theta.
    # Test that GPC can use externally defined optimizers.
    # Define a dummy optimizer that simply tests 10 random hyperparameters
    def optimizer(obj_func, initial_theta, bounds):
        # [FUNCTION DOC]
        # HELPER: Dummy optimizer: evaluates obj_func at the initial theta and at 10
        # random thetas drawn uniformly within bounds clipped to [-2, 1], returning the
        # best one.
        # Inputs: obj_func (callable(theta, eval_gradient)), initial_theta (1D), bounds
        # (n_dims, 2). Outputs: tuple (theta_opt, func_min).
        rng = np.random.RandomState(global_random_seed)
        theta_opt, func_min = (
            initial_theta,
            obj_func(initial_theta, eval_gradient=False),
        )
        for _ in range(10):
            theta = np.atleast_1d(
                rng.uniform(np.maximum(-2, bounds[:, 0]), np.minimum(1, bounds[:, 1]))
            )
            f = obj_func(theta, eval_gradient=False)
            if f < func_min:
                theta_opt, func_min = theta, f
        return theta_opt, func_min

    gpc = GaussianProcessClassifier(kernel=kernel, optimizer=optimizer)
    gpc.fit(X, y_mc)
    # Checks that optimizer improved marginal likelihood
    assert gpc.log_marginal_likelihood(
        gpc.kernel_.theta
    ) >= gpc.log_marginal_likelihood(kernel.theta)


@pytest.mark.parametrize("kernel", kernels)
def test_multi_class(kernel):
    # [FUNCTION DOC]
    # Verifies multi-class predict_proba rows sum to 1 and predict equals argmax of the
    # probabilities on X2.
    # Test GPC for multi-class classification problems.
    gpc = GaussianProcessClassifier(kernel=kernel)
    gpc.fit(X, y_mc)

    y_prob = gpc.predict_proba(X2)
    assert_almost_equal(y_prob.sum(1), 1)

    y_pred = gpc.predict(X2)
    assert_array_equal(np.argmax(y_prob, 1), y_pred)


@pytest.mark.parametrize("kernel", kernels)
def test_multi_class_n_jobs(kernel):
    # [FUNCTION DOC]
    # Verifies multi-class probabilities are identical with n_jobs=1 and n_jobs=2.
    # Test that multi-class GPC produces identical results with n_jobs>1.
    gpc = GaussianProcessClassifier(kernel=kernel)
    gpc.fit(X, y_mc)

    gpc_2 = GaussianProcessClassifier(kernel=kernel, n_jobs=2)
    gpc_2.fit(X, y_mc)

    y_prob = gpc.predict_proba(X2)
    y_prob_2 = gpc_2.predict_proba(X2)
    assert_almost_equal(y_prob, y_prob_2)


def test_warning_bounds():
    # [FUNCTION DOC]
    # Verifies ConvergenceWarning messages from Kernel._check_bounds_params: single RBF
    # hitting its upper bound; a Sum kernel producing two warnings with "k1__"/"k2__"
    # prefixed names (upper for noise, lower for length scale); and an anisotropic RBF
    # producing one warning per dimension.
    kernel = RBF(length_scale_bounds=[1e-5, 1e-3])
    gpc = GaussianProcessClassifier(kernel=kernel)
    warning_message = (
        "The optimal value found for dimension 0 of parameter "
        "length_scale is close to the specified upper bound "
        "0.001. Increasing the bound and calling fit again may "
        "find a better value."
    )
    with pytest.warns(ConvergenceWarning, match=warning_message):
        gpc.fit(X, y)

    kernel_sum = WhiteKernel(noise_level_bounds=[1e-5, 1e-3]) + RBF(
        length_scale_bounds=[1e3, 1e5]
    )
    gpc_sum = GaussianProcessClassifier(kernel=kernel_sum)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        gpc_sum.fit(X, y)

        assert len(record) == 2

        assert issubclass(record[0].category, ConvergenceWarning)
        assert (
            record[0].message.args[0] == "The optimal value found for "
            "dimension 0 of parameter "
            "k1__noise_level is close to the "
            "specified upper bound 0.001. "
            "Increasing the bound and calling "
            "fit again may find a better value."
        )

        assert issubclass(record[1].category, ConvergenceWarning)
        assert (
            record[1].message.args[0] == "The optimal value found for "
            "dimension 0 of parameter "
            "k2__length_scale is close to the "
            "specified lower bound 1000.0. "
            "Decreasing the bound and calling "
            "fit again may find a better value."
        )

    X_tile = np.tile(X, 2)
    kernel_dims = RBF(length_scale=[1.0, 2.0], length_scale_bounds=[1e1, 1e2])
    gpc_dims = GaussianProcessClassifier(kernel=kernel_dims)

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        gpc_dims.fit(X_tile, y)

        assert len(record) == 2

        assert issubclass(record[0].category, ConvergenceWarning)
        assert (
            record[0].message.args[0] == "The optimal value found for "
            "dimension 0 of parameter "
            "length_scale is close to the "
            "specified upper bound 100.0. "
            "Increasing the bound and calling "
            "fit again may find a better value."
        )

        assert issubclass(record[1].category, ConvergenceWarning)
        assert (
            record[1].message.args[0] == "The optimal value found for "
            "dimension 1 of parameter "
            "length_scale is close to the "
            "specified upper bound 100.0. "
            "Increasing the bound and calling "
            "fit again may find a better value."
        )


@pytest.mark.parametrize(
    "params, error_type, err_msg",
    [
        (
            {"kernel": CompoundKernel(0)},
            ValueError,
            "kernel cannot be a CompoundKernel",
        )
    ],
)
def test_gpc_fit_error(params, error_type, err_msg):
    # [FUNCTION DOC]
    # Verifies fit raises the expected error (ValueError when the kernel is a
    # CompoundKernel), parametrized over params/error type/message.
    """Check that expected error are raised during fit."""
    gpc = GaussianProcessClassifier(**params)
    with pytest.raises(error_type, match=err_msg):
        gpc.fit(X, y)


@pytest.mark.parametrize("kernel", kernels)
def test_gpc_latent_mean_and_variance_shape(kernel):
    # [FUNCTION DOC]
    # Verifies latent_mean_and_variance returns 1D arrays of length n_samples for binary
    # problems, for each kernel.
    """Checks that the latent mean and variance have the right shape."""
    gpc = GaussianProcessClassifier(kernel=kernel)
    gpc.fit(X, y)

    # Check that the latent mean and variance have the right shape
    latent_mean, latent_variance = gpc.latent_mean_and_variance(X)
    assert latent_mean.shape == (X.shape[0],)
    assert latent_variance.shape == (X.shape[0],)


def test_gpc_latent_mean_and_variance_complain_on_more_than_2_classes():
    # [FUNCTION DOC]
    # Verifies latent_mean_and_variance raises ValueError for a 3-class problem.
    # KNOWN ISSUE: the docstring and inline comment (copied from the previous test)
    # claim it checks shapes, but the test checks the multi-class error.
    """Checks that the latent mean and variance have the right shape."""
    gpc = GaussianProcessClassifier(kernel=RBF())
    gpc.fit(X, y_mc)

    # Check that the latent mean and variance have the right shape
    with pytest.raises(
        ValueError,
        match="Returning the mean and variance of the latent function f "
        "is only supported for binary classification",
    ):
        gpc.latent_mean_and_variance(X)


def test_latent_mean_and_variance_works_on_structured_kernels():
    # [FUNCTION DOC]
    # Smoke test: latent_mean_and_variance runs without error on string inputs with
    # MiniSeqKernel (no assertion on the output).
    X = ["A", "AB", "B"]
    y = np.array([True, False, True])
    kernel = MiniSeqKernel(baseline_similarity_bounds="fixed")
    gpc = GaussianProcessClassifier(kernel=kernel).fit(X, y)

    gpc.latent_mean_and_variance(X)
