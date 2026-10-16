# [MODULE DOC]
# Tests for sklearn.gaussian_process.kernels: analytic vs numerical gradients, theta and
# hyperparameter bookkeeping, diag/auto/cross consistency, operators, anisotropy,
# stationarity, cloning, Matern special cases, compatibility with pairwise_kernels,
# get/set_params and error handling. Shared read-only data X (5x2) and Y (6x2) and a
# list of kernels covering every kernel type, plus a PairwiseKernel per metric (except
# additive_chi2 and chi2).
"""Testing for kernels for Gaussian processes."""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

from inspect import signature

import numpy as np
import pytest

from sklearn.base import clone
from sklearn.gaussian_process.kernels import (
    RBF,
    CompoundKernel,
    ConstantKernel,
    DotProduct,
    Exponentiation,
    ExpSineSquared,
    KernelOperator,
    Matern,
    PairwiseKernel,
    RationalQuadratic,
    WhiteKernel,
    _approx_fprime,
)
from sklearn.metrics.pairwise import (
    PAIRWISE_KERNEL_FUNCTIONS,
    euclidean_distances,
    pairwise_kernels,
)
from sklearn.utils._testing import (
    assert_allclose,
    assert_almost_equal,
    assert_array_almost_equal,
    assert_array_equal,
)

X = np.random.RandomState(0).normal(0, 1, (5, 2))
Y = np.random.RandomState(0).normal(0, 1, (6, 2))
# Set shared test data as read-only to avoid unintentional in-place
# modifications that would introduce side-effects between tests.
X.flags.writeable = False
Y.flags.writeable = False

kernel_rbf_plus_white = RBF(length_scale=2.0) + WhiteKernel(noise_level=3.0)
kernels = [
    RBF(length_scale=2.0),
    RBF(length_scale_bounds=(0.5, 2.0)),
    ConstantKernel(constant_value=10.0),
    2.0 * RBF(length_scale=0.33, length_scale_bounds="fixed"),
    2.0 * RBF(length_scale=0.5),
    kernel_rbf_plus_white,
    2.0 * RBF(length_scale=[0.5, 2.0]),
    2.0 * Matern(length_scale=0.33, length_scale_bounds="fixed"),
    2.0 * Matern(length_scale=0.5, nu=0.5),
    2.0 * Matern(length_scale=1.5, nu=1.5),
    2.0 * Matern(length_scale=2.5, nu=2.5),
    2.0 * Matern(length_scale=[0.5, 2.0], nu=0.5),
    3.0 * Matern(length_scale=[2.0, 0.5], nu=1.5),
    4.0 * Matern(length_scale=[0.5, 0.5], nu=2.5),
    RationalQuadratic(length_scale=0.5, alpha=1.5),
    ExpSineSquared(length_scale=0.5, periodicity=1.5),
    DotProduct(sigma_0=2.0),
    DotProduct(sigma_0=2.0) ** 2,
    RBF(length_scale=[2.0]),
    Matern(length_scale=[2.0]),
]
for metric in PAIRWISE_KERNEL_FUNCTIONS:
    if metric in ["additive_chi2", "chi2"]:
        continue
    kernels.append(PairwiseKernel(gamma=1.0, metric=metric))


@pytest.mark.parametrize("kernel", kernels)
def test_kernel_gradient(kernel):
    # [FUNCTION DOC]
    # Verifies each kernel's gradient has shape (n_X, n_X, n_dims) and matches a
    # numerical gradient from _approx_fprime (step 1e-10) to 4 decimals.
    # Compare analytic and numeric gradient of kernels.
    kernel = clone(kernel)  # make tests independent of one-another
    K, K_gradient = kernel(X, eval_gradient=True)

    assert K_gradient.shape[0] == X.shape[0]
    assert K_gradient.shape[1] == X.shape[0]
    assert K_gradient.shape[2] == kernel.theta.shape[0]

    def eval_kernel_for_theta(theta):
        # [FUNCTION DOC]
        # HELPER: Evaluates a clone of the kernel with the given theta on X. Outputs: K
        # matrix.
        kernel_clone = kernel.clone_with_theta(theta)
        K = kernel_clone(X, eval_gradient=False)
        return K

    K_gradient_approx = _approx_fprime(kernel.theta, eval_kernel_for_theta, 1e-10)

    assert_almost_equal(K_gradient, K_gradient_approx, 4)


@pytest.mark.parametrize(
    "kernel",
    [
        kernel
        for kernel in kernels
        # skip non-basic kernels
        if not (isinstance(kernel, (KernelOperator, Exponentiation)))
    ],
)
def test_kernel_theta(kernel):
    # [FUNCTION DOC]
    # For basic (non-operator) kernels verifies: hyperparameter names match constructor
    # arguments ending in "_bounds"; theta entries are logs of hyperparameter values;
    # fixing one hyperparameter removes exactly that entry from theta and the gradient
    # leaving the others identical; and setting theta or the attribute updates the
    # other.
    # Check that parameter vector theta of kernel is set correctly.
    kernel = clone(kernel)  # make tests independent of one-another
    theta = kernel.theta
    _, K_gradient = kernel(X, eval_gradient=True)

    # Determine kernel parameters that contribute to theta
    init_sign = signature(kernel.__class__.__init__).parameters.values()
    args = [p.name for p in init_sign if p.name != "self"]
    theta_vars = map(
        lambda s: s[0 : -len("_bounds")], filter(lambda s: s.endswith("_bounds"), args)
    )
    assert set(hyperparameter.name for hyperparameter in kernel.hyperparameters) == set(
        theta_vars
    )

    # Check that values returned in theta are consistent with
    # hyperparameter values (being their logarithms)
    for i, hyperparameter in enumerate(kernel.hyperparameters):
        assert theta[i] == np.log(getattr(kernel, hyperparameter.name))

    # Fixed kernel parameters must be excluded from theta and gradient.
    for i, hyperparameter in enumerate(kernel.hyperparameters):
        # create copy with certain hyperparameter fixed
        params = kernel.get_params()
        params[hyperparameter.name + "_bounds"] = "fixed"
        kernel_class = kernel.__class__
        new_kernel = kernel_class(**params)
        # Check that theta and K_gradient are identical with the fixed
        # dimension left out
        _, K_gradient_new = new_kernel(X, eval_gradient=True)
        assert theta.shape[0] == new_kernel.theta.shape[0] + 1
        assert K_gradient.shape[2] == K_gradient_new.shape[2] + 1
        if i > 0:
            assert theta[:i] == new_kernel.theta[:i]
            assert_array_equal(K_gradient[..., :i], K_gradient_new[..., :i])
        if i + 1 < len(kernel.hyperparameters):
            assert theta[i + 1 :] == new_kernel.theta[i:]
            assert_array_equal(K_gradient[..., i + 1 :], K_gradient_new[..., i:])

    # Check that values of theta are modified correctly
    for i, hyperparameter in enumerate(kernel.hyperparameters):
        theta[i] = np.log(42)
        kernel.theta = theta
        assert_almost_equal(getattr(kernel, hyperparameter.name), 42)

        setattr(kernel, hyperparameter.name, 43)
        assert_almost_equal(kernel.theta[i], np.log(43))


@pytest.mark.parametrize(
    "kernel",
    [
        kernel
        for kernel in kernels
        # Identity is not satisfied on diagonal
        if kernel != kernel_rbf_plus_white
    ],
)
def test_auto_vs_cross(kernel):
    # [FUNCTION DOC]
    # Verifies k(X) equals k(X, X) (5 decimals) for all kernels except RBF + White.
    kernel = clone(kernel)  # make tests independent of one-another
    # Auto-correlation and cross-correlation should be consistent.
    K_auto = kernel(X)
    K_cross = kernel(X, X)
    assert_almost_equal(K_auto, K_cross, 5)


@pytest.mark.parametrize("kernel", kernels)
def test_kernel_diag(kernel):
    # [FUNCTION DOC]
    # Verifies diag(X) equals the diagonal of k(X) (5 decimals).
    kernel = clone(kernel)  # make tests independent of one-another
    # Test that diag method of kernel returns consistent results.
    K_call_diag = np.diag(kernel(X))
    K_diag = kernel.diag(X)
    assert_almost_equal(K_call_diag, K_diag, 5)


def test_kernel_operator_commutative():
    # [FUNCTION DOC]
    # Verifies scalar + kernel equals kernel + scalar and scalar * kernel equals kernel
    # * scalar on X.
    # Adding kernels and multiplying kernels should be commutative.
    # Check addition
    assert_almost_equal((RBF(2.0) + 1.0)(X), (1.0 + RBF(2.0))(X))

    # Check multiplication
    assert_almost_equal((3.0 * RBF(2.0))(X), (RBF(2.0) * 3.0)(X))


def test_kernel_anisotropic():
    # [FUNCTION DOC]
    # Verifies an anisotropic RBF equals isotropic RBFs on rescaled features, and that
    # adding log 2 to theta doubles the constant and both length scales.
    # Anisotropic kernel should be consistent with isotropic kernels.
    kernel = 3.0 * RBF([0.5, 2.0])

    K = kernel(X)
    X1 = X.copy()
    X1[:, 0] *= 4
    K1 = 3.0 * RBF(2.0)(X1)
    assert_almost_equal(K, K1)

    X2 = X.copy()
    X2[:, 1] /= 4
    K2 = 3.0 * RBF(0.5)(X2)
    assert_almost_equal(K, K2)

    # Check getting and setting via theta
    kernel.theta = kernel.theta + np.log(2)
    assert_array_equal(kernel.theta, np.log([6.0, 1.0, 4.0]))
    assert_array_equal(kernel.k2.length_scale, [1.0, 4.0])


@pytest.mark.parametrize(
    "kernel", [kernel for kernel in kernels if kernel.is_stationary()]
)
def test_kernel_stationary(kernel):
    # [FUNCTION DOC]
    # Verifies stationary kernels give a constant diagonal for k(X, X + 1).
    kernel = clone(kernel)  # make tests independent of one-another
    # Test stationarity of kernels.
    K = kernel(X, X + 1)
    assert_almost_equal(K[0, 0], np.diag(K))


@pytest.mark.parametrize("kernel", kernels)
def test_kernel_input_type(kernel):
    # [FUNCTION DOC]
    # Verifies requires_vector_input of Exponentiation and KernelOperator kernels
    # follows their sub-kernels.
    kernel = clone(kernel)  # make tests independent of one-another
    # Test whether kernels is for vectors or structured data
    if isinstance(kernel, Exponentiation):
        assert kernel.requires_vector_input == kernel.kernel.requires_vector_input
    if isinstance(kernel, KernelOperator):
        assert kernel.requires_vector_input == (
            kernel.k1.requires_vector_input or kernel.k2.requires_vector_input
        )


def test_compound_kernel_input_type():
    # [FUNCTION DOC]
    # Verifies CompoundKernel.requires_vector_input is False with only a WhiteKernel and
    # True once an RBF is added.
    kernel = CompoundKernel([WhiteKernel(noise_level=3.0)])
    assert not kernel.requires_vector_input

    kernel = CompoundKernel([WhiteKernel(noise_level=3.0), RBF(length_scale=2.0)])
    assert kernel.requires_vector_input


def check_hyperparameters_equal(kernel1, kernel2):
    # [FUNCTION DOC]
    # HELPER: Asserts that every "hyperparameter_*" attribute of either kernel is equal
    # between the two. Raises AttributeError if one kernel lacks an attribute of the
    # other.
    # Check that hyperparameters of two kernels are equal
    for attr in set(dir(kernel1) + dir(kernel2)):
        if attr.startswith("hyperparameter_"):
            attr_value1 = getattr(kernel1, attr)
            attr_value2 = getattr(kernel2, attr)
            assert attr_value1 == attr_value2


@pytest.mark.parametrize("kernel", kernels)
def test_kernel_clone(kernel):
    # [FUNCTION DOC]
    # Verifies sklearn clone yields an equal but distinct kernel with identical params
    # and hyperparameters.
    kernel = clone(kernel)  # make tests independent of one-another
    # Test that sklearn's clone works correctly on kernels.
    kernel_cloned = clone(kernel)

    # XXX: Should this be fixed?
    # This differs from the sklearn's estimators equality check.
    assert kernel == kernel_cloned
    assert id(kernel) != id(kernel_cloned)

    # Check that all constructor parameters are equal.
    assert kernel.get_params() == kernel_cloned.get_params()

    # Check that all hyperparameters are equal.
    check_hyperparameters_equal(kernel, kernel_cloned)


@pytest.mark.parametrize("kernel", kernels)
def test_kernel_clone_after_set_params(kernel):
    # [FUNCTION DOC]
    # Regression test (issue 6961): after set_params turns a scalar length_scale into a
    # 2-element list (non-isotropic kernels only), cloning still yields equal params and
    # hyperparameters.
    kernel = clone(kernel)  # make tests independent of one-another
    # This test is to verify that using set_params does not
    # break clone on kernels.
    # This used to break because in kernels such as the RBF, non-trivial
    # logic that modified the length scale used to be in the constructor
    # See https://github.com/scikit-learn/scikit-learn/issues/6961
    # for more details.
    bounds = (1e-5, 1e5)
    kernel_cloned = clone(kernel)
    params = kernel.get_params()
    # RationalQuadratic kernel is isotropic.
    isotropic_kernels = (ExpSineSquared, RationalQuadratic)
    if "length_scale" in params and not isinstance(kernel, isotropic_kernels):
        length_scale = params["length_scale"]
        if np.iterable(length_scale):
            # XXX unreached code as of v0.22
            params["length_scale"] = length_scale[0]
            params["length_scale_bounds"] = bounds
        else:
            params["length_scale"] = [length_scale] * 2
            params["length_scale_bounds"] = bounds * 2
        kernel_cloned.set_params(**params)
        kernel_cloned_clone = clone(kernel_cloned)
        assert kernel_cloned_clone.get_params() == kernel_cloned.get_params()
        assert id(kernel_cloned_clone) != id(kernel_cloned)
        check_hyperparameters_equal(kernel_cloned, kernel_cloned_clone)


def test_matern_kernel():
    # [FUNCTION DOC]
    # Verifies Matern: unit diagonal; nu=0.5 equals the absolute exponential kernel;
    # nu=inf equals RBF; nu in {0.5, 1.5, 2.5} matches the general formula at nu +
    # 1e-10; nu=100 is close to RBF (2 decimals).
    # Test consistency of Matern kernel for special values of nu.
    K = Matern(nu=1.5, length_scale=1.0)(X)
    # the diagonal elements of a matern kernel are 1
    assert_array_almost_equal(np.diag(K), np.ones(X.shape[0]))
    # matern kernel for coef0==0.5 is equal to absolute exponential kernel
    K_absexp = np.exp(-euclidean_distances(X, X, squared=False))
    K = Matern(nu=0.5, length_scale=1.0)(X)
    assert_array_almost_equal(K, K_absexp)
    # matern kernel with coef0==inf is equal to RBF kernel
    K_rbf = RBF(length_scale=1.0)(X)
    K = Matern(nu=np.inf, length_scale=1.0)(X)
    assert_array_almost_equal(K, K_rbf)
    assert_allclose(K, K_rbf)
    # test that special cases of matern kernel (coef0 in [0.5, 1.5, 2.5])
    # result in nearly identical results as the general case for coef0 in
    # [0.5 + tiny, 1.5 + tiny, 2.5 + tiny]
    tiny = 1e-10
    for nu in [0.5, 1.5, 2.5]:
        K1 = Matern(nu=nu, length_scale=1.0)(X)
        K2 = Matern(nu=nu + tiny, length_scale=1.0)(X)
        assert_array_almost_equal(K1, K2)
    # test that coef0==large is close to RBF
    large = 100
    K1 = Matern(nu=large, length_scale=1.0)(X)
    K2 = RBF(length_scale=1.0)(X)
    assert_array_almost_equal(K1, K2, decimal=2)


@pytest.mark.parametrize("kernel", kernels)
def test_kernel_versus_pairwise(kernel):
    # [FUNCTION DOC]
    # Verifies kernels used as callables in pairwise_kernels give the same auto (except
    # RBF + White) and cross kernel matrices.
    kernel = clone(kernel)  # make tests independent of one-another
    # Check that GP kernels can also be used as pairwise kernels.

    # Test auto-kernel
    if kernel != kernel_rbf_plus_white:
        # For WhiteKernel: k(X) != k(X,X). This is assumed by
        # pairwise_kernels
        K1 = kernel(X)
        K2 = pairwise_kernels(X, metric=kernel)
        assert_array_almost_equal(K1, K2)

    # Test cross-kernel
    K1 = kernel(X, Y)
    K2 = pairwise_kernels(X, Y, metric=kernel)
    assert_array_almost_equal(K1, K2)


@pytest.mark.parametrize("kernel", kernels)
def test_set_get_params(kernel):
    # [FUNCTION DOC]
    # Verifies get_params values equal exp(theta) entries for non-fixed hyperparameters,
    # and set_params to 10 reflects in theta, handling vector-valued hyperparameters.
    # KNOWN ISSUE: the check isinstance("string", type(bounds)) is an inverted way to
    # test whether bounds is a string; it works but reads misleadingly.
    kernel = clone(kernel)  # make tests independent of one-another
    # Check that set_params()/get_params() is consistent with kernel.theta.

    # Test get_params()
    index = 0
    params = kernel.get_params()
    for hyperparameter in kernel.hyperparameters:
        if isinstance("string", type(hyperparameter.bounds)):
            if hyperparameter.bounds == "fixed":
                continue
        size = hyperparameter.n_elements
        if size > 1:  # anisotropic kernels
            assert_almost_equal(
                np.exp(kernel.theta[index : index + size]), params[hyperparameter.name]
            )
            index += size
        else:
            assert_almost_equal(
                np.exp(kernel.theta[index]), params[hyperparameter.name]
            )
            index += 1
    # Test set_params()
    index = 0
    value = 10  # arbitrary value
    for hyperparameter in kernel.hyperparameters:
        if isinstance("string", type(hyperparameter.bounds)):
            if hyperparameter.bounds == "fixed":
                continue
        size = hyperparameter.n_elements
        if size > 1:  # anisotropic kernels
            kernel.set_params(**{hyperparameter.name: [value] * size})
            assert_almost_equal(
                np.exp(kernel.theta[index : index + size]), [value] * size
            )
            index += size
        else:
            kernel.set_params(**{hyperparameter.name: value})
            assert_almost_equal(np.exp(kernel.theta[index]), value)
            index += 1


@pytest.mark.parametrize("kernel", kernels)
def test_repr_kernels(kernel):
    # [FUNCTION DOC]
    # Smoke test that repr does not raise for any kernel.
    kernel = clone(kernel)  # make tests independent of one-another
    # Smoke-test for repr in kernels.

    repr(kernel)


def test_rational_quadratic_kernel():
    # [FUNCTION DOC]
    # Verifies RationalQuadratic with a vector length_scale raises AttributeError when
    # called.
    kernel = RationalQuadratic(length_scale=[1.0, 1.0])
    message = (
        "RationalQuadratic kernel only supports isotropic "
        "version, please use a single "
        "scalar for length_scale"
    )
    with pytest.raises(AttributeError, match=message):
        kernel(X)
