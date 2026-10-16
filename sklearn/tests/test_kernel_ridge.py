# [MODULE DOC]
# Tests for sklearn.kernel_ridge.KernelRidge. Builds one shared regression problem at
# import time (X, y from make_regression with 10 features, seed 0; Y stacks y twice
# as a 2-output target) and checks that KernelRidge with a linear kernel reproduces
# Ridge without intercept, plus sparse, singular, precomputed, weighted and
# multi-output cases.
import numpy as np
import pytest

from sklearn.datasets import make_regression
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.utils._testing import assert_array_almost_equal, ignore_warnings
from sklearn.utils.fixes import CSC_CONTAINERS, CSR_CONTAINERS

X, y = make_regression(n_features=10, random_state=0)
Y = np.array([y, y]).T


def test_kernel_ridge():
    # [FUNCTION DOC]
    # Verifies that a linear-kernel KernelRidge (alpha=1) gives the same predictions on
    # the training data as Ridge(alpha=1, fit_intercept=False), to ~6 decimals.
    pred = Ridge(alpha=1, fit_intercept=False).fit(X, y).predict(X)
    pred2 = KernelRidge(kernel="linear", alpha=1).fit(X, y).predict(X)
    assert_array_almost_equal(pred, pred2)


@pytest.mark.parametrize("sparse_container", [*CSR_CONTAINERS, *CSC_CONTAINERS])
def test_kernel_ridge_sparse(sparse_container):
    # [FUNCTION DOC]
    # Verifies the Ridge equivalence when X is given as each CSR/CSC sparse container;
    # Ridge uses the cholesky solver for comparison.
    # Inputs: sparse_container - parametrized sparse matrix/array constructor.
    X_sparse = sparse_container(X)
    pred = (
        Ridge(alpha=1, fit_intercept=False, solver="cholesky")
        .fit(X_sparse, y)
        .predict(X_sparse)
    )
    pred2 = KernelRidge(kernel="linear", alpha=1).fit(X_sparse, y).predict(X_sparse)
    assert_array_almost_equal(pred, pred2)


def test_kernel_ridge_singular_kernel():
    # [FUNCTION DOC]
    # Verifies the alpha=0 case: the dual solve hits a singular system and KernelRidge
    # falls back to least squares; predictions must still match Ridge(alpha=0).
    # Warnings raised during fit are suppressed via ignore_warnings.
    # alpha=0 causes a LinAlgError in computing the dual coefficients,
    # which causes a fallback to a lstsq solver. This is tested here.
    pred = Ridge(alpha=0, fit_intercept=False).fit(X, y).predict(X)
    kr = KernelRidge(kernel="linear", alpha=0)
    ignore_warnings(kr.fit)(X, y)
    pred2 = kr.predict(X)
    assert_array_almost_equal(pred, pred2)


def test_kernel_ridge_precomputed():
    # [FUNCTION DOC]
    # For linear, rbf, poly and cosine kernels, verifies that fitting on raw X and fitting
    # with kernel="precomputed" on the Gram matrix from pairwise_kernels give equal
    # predictions.
    for kernel in ["linear", "rbf", "poly", "cosine"]:
        K = pairwise_kernels(X, X, metric=kernel)
        pred = KernelRidge(kernel=kernel).fit(X, y).predict(X)
        pred2 = KernelRidge(kernel="precomputed").fit(K, y).predict(K)
        assert_array_almost_equal(pred, pred2)


def test_kernel_ridge_precomputed_kernel_unchanged():
    # [FUNCTION DOC]
    # Verifies that fitting with a precomputed kernel does not modify the user-supplied
    # Gram matrix in place (compared against a copy taken before fit).
    K = np.dot(X, X.T)
    K2 = K.copy()
    KernelRidge(kernel="precomputed").fit(K, y)
    assert_array_almost_equal(K, K2)


def test_kernel_ridge_sample_weights():
    # [FUNCTION DOC]
    # Verifies sample_weight support: random weights (seed 0) give identical predictions
    # for Ridge, linear-kernel KernelRidge and precomputed-kernel KernelRidge.
    K = np.dot(X, X.T)  # precomputed kernel
    sw = np.random.RandomState(0).rand(X.shape[0])

    pred = Ridge(alpha=1, fit_intercept=False).fit(X, y, sample_weight=sw).predict(X)
    pred2 = KernelRidge(kernel="linear", alpha=1).fit(X, y, sample_weight=sw).predict(X)
    pred3 = (
        KernelRidge(kernel="precomputed", alpha=1)
        .fit(K, y, sample_weight=sw)
        .predict(K)
    )
    assert_array_almost_equal(pred, pred2)
    assert_array_almost_equal(pred, pred3)


def test_kernel_ridge_multi_output():
    # [FUNCTION DOC]
    # Verifies 2-output targets: KernelRidge on Y matches Ridge on Y, and equals the
    # single-output prediction on y duplicated into two columns.
    pred = Ridge(alpha=1, fit_intercept=False).fit(X, Y).predict(X)
    pred2 = KernelRidge(kernel="linear", alpha=1).fit(X, Y).predict(X)
    assert_array_almost_equal(pred, pred2)

    pred3 = KernelRidge(kernel="linear", alpha=1).fit(X, y).predict(X)
    pred3 = np.array([pred3, pred3]).T
    assert_array_almost_equal(pred2, pred3)
