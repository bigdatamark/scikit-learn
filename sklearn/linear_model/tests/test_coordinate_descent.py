# [MODULE DOC]
# Tests for coordinate descent linear models: Lasso, ElasticNet, their multi-task and
# CV variants, the path functions enet_path / lasso_path, the Cython solvers in
# _cd_fast and the _set_order helper. Covers toy solutions and dual gaps, CV alpha and
# l1_ratio selection, positivity, warm starts, sparse/dense and float32/float64
# consistency, precomputed Gram handling, copy_X/check_input semantics, convergence
# warnings, sample weight semantics (including inside CV), agreement with Ridge, OLS,
# LARS and GridSearchCV, and parameter validation/deprecations.
# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import warnings
from copy import deepcopy

import joblib
import numpy as np
import pytest
from scipy import interpolate, sparse

from sklearn.base import clone
from sklearn.datasets import load_diabetes, make_regression
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import (
    ElasticNet,
    ElasticNetCV,
    Lasso,
    LassoCV,
    LassoLarsCV,
    LinearRegression,
    MultiTaskElasticNet,
    MultiTaskElasticNetCV,
    MultiTaskLasso,
    MultiTaskLassoCV,
    Ridge,
    enet_path,
    lars_path,
    lasso_path,
)
from sklearn.linear_model import _cd_fast as cd_fast
from sklearn.linear_model._coordinate_descent import _set_order
from sklearn.model_selection import (
    GridSearchCV,
    LeaveOneGroupOut,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_array
from sklearn.utils._testing import (
    TempMemmap,
    assert_allclose,
    assert_almost_equal,
    assert_array_almost_equal,
    assert_array_equal,
    assert_array_less,
    ignore_warnings,
)
from sklearn.utils.fixes import COO_CONTAINERS, CSC_CONTAINERS, CSR_CONTAINERS


@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("input_order", ["C", "F"])
def test_set_order_dense(order, input_order):
    # [FUNCTION DOC]
    # Verifies _set_order returns X and y contiguous in the requested order ("C" or "F")
    # and returns the very same objects when the input already has that order.
    """Check that _set_order returns arrays with promised order."""
    X = np.array([[0], [0], [0]], order=input_order)
    y = np.array([0, 0, 0], order=input_order)
    X2, y2 = _set_order(X, y, order=order)
    if order == "C":
        assert X2.flags["C_CONTIGUOUS"]
        assert y2.flags["C_CONTIGUOUS"]
    elif order == "F":
        assert X2.flags["F_CONTIGUOUS"]
        assert y2.flags["F_CONTIGUOUS"]

    if order == input_order:
        assert X is X2
        assert y is y2


@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("input_order", ["C", "F"])
@pytest.mark.parametrize("coo_container", COO_CONTAINERS)
def test_set_order_sparse(order, input_order, coo_container):
    # [FUNCTION DOC]
    # Verifies _set_order converts sparse inputs to CSC for order "F" and CSR for "C".
    # Inputs start as COO and are converted to CSC/CSR according to input_order.
    # KNOWN ISSUE: y is reassigned from X (y = X.asformat(...)) instead of from y, so
    #   the
    # sparse 1-D y built from coo_container is never exercised; y2 checks X's
    #   conversion.
    """Check that _set_order returns sparse matrices in promised format."""
    X = coo_container(np.array([[0], [0], [0]]))
    y = coo_container(np.array([0, 0, 0]))
    sparse_format = "csc" if input_order == "F" else "csr"
    X = X.asformat(sparse_format)
    y = X.asformat(sparse_format)
    X2, y2 = _set_order(X, y, order=order)

    format = "csc" if order == "F" else "csr"
    assert sparse.issparse(X2) and X2.format == format
    assert sparse.issparse(y2) and y2.format == format


@pytest.mark.parametrize("sparse_csc_type", [sparse.csc_array, sparse.csc_matrix])
def test_cython_solver_equivalence(sparse_csc_type):
    # [FUNCTION DOC]
    # Verifies the dense, sparse and Gram Cython solvers of _cd_fast agree for a 1-D
    #   target.
    # On centered make_regression data (dense X centered and Fortran-ordered, y
    #   centered),
    # with beta=0 (pure Lasso), tol 1e-10 and cyclic updates: at alpha_max = infinity
    #   norm
    # of X^T y all coefficients stay zero with and without gap safe screening; at
    # alpha_max/10 the dense solver without screening gives between 2 and n_features-1
    # nonzeros, and the dense solver with screening, the sparse solver (raw X plus
    #   X_mean
    # for implicit centering, both screening settings) and the Gram solver (Q = Xc^T Xc,
    # q = Xc^T y) all reproduce it.
    # Inputs: sparse_csc_type - csc_array or csc_matrix.
    # Side effects: solvers update coefficient arrays in place.
    """Test that all 3 Cython solvers for 1-d targets give same results."""
    X, y = make_regression()
    X_mean = X.mean(axis=0)
    X_centered = np.asfortranarray(X - X_mean)
    y -= y.mean()
    alpha_max = np.linalg.norm(X.T @ y, ord=np.inf)
    alpha = alpha_max / 10
    params = {
        "beta": 0,
        "max_iter": 100,
        "tol": 1e-10,
        "rng": np.random.RandomState(0),  # not used, but needed as argument
        "random": False,
        "positive": False,
    }

    def zc():
        # [FUNCTION DOC]
        # HELPER: Returns a fresh float64 zero vector of length n_features (enclosing
        #   X).
        """Create a new zero coefficient array (zc)."""
        return np.zeros(X.shape[1])

    # For alpha_max, coefficients must all be zero.
    coef_1 = zc()
    for do_screening in [True, False]:
        cd_fast.enet_coordinate_descent(
            w=coef_1,
            alpha=alpha_max,
            X=X_centered,
            y=y,
            **params,
            do_screening=do_screening,
        )
        assert_allclose(coef_1, 0)

    # Without gap safe screening rules
    coef_1 = zc()
    cd_fast.enet_coordinate_descent(
        w=coef_1, alpha=alpha, X=X_centered, y=y, **params, do_screening=False
    )
    # At least 2 coefficients are non-zero
    assert 2 <= np.sum(np.abs(coef_1) > 1e-8) < X.shape[1]

    # With gap safe screening rules
    coef_2 = zc()
    cd_fast.enet_coordinate_descent(
        w=coef_2, alpha=alpha, X=X_centered, y=y, **params, do_screening=True
    )
    assert_allclose(coef_2, coef_1)

    # Sparse
    Xs = sparse_csc_type(X)
    for do_screening in [True, False]:
        coef_3 = zc()
        cd_fast.sparse_enet_coordinate_descent(
            w=coef_3,
            alpha=alpha,
            X_data=Xs.data,
            X_indices=Xs.indices,
            X_indptr=Xs.indptr,
            y=y,
            sample_weight=None,
            X_mean=X_mean,
            **params,
            do_screening=do_screening,
        )
        assert_allclose(coef_3, coef_1)

    # Gram
    for do_screening in [True, False]:
        coef_4 = zc()
        cd_fast.enet_coordinate_descent_gram(
            w=coef_4,
            alpha=alpha,
            Q=X_centered.T @ X_centered,
            q=X_centered.T @ y,
            y=y,
            **params,
            do_screening=do_screening,
        )
        assert_allclose(coef_4, coef_1)


def test_lasso_zero():
    # [FUNCTION DOC]
    # Verifies Lasso fit on all-zero X and y yields coef 0, zero predictions and zero
    # dual gap.
    # Check that the lasso can handle zero data without crashing
    X = [[0], [0], [0]]
    y = [0, 0, 0]
    clf = Lasso(alpha=0.1).fit(X, y)
    pred = clf.predict([[1], [2], [3]])
    assert_array_almost_equal(clf.coef_, [0])
    assert_array_almost_equal(pred, [0, 0, 0])
    assert_almost_equal(clf.dual_gap_, 0)


@pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
@pytest.mark.filterwarnings("ignore::RuntimeWarning")  # overflow and similar
def test_enet_nonfinite_params():
    # [FUNCTION DOC]
    # Verifies ElasticNet raises ValueError "Coordinate descent iterations resulted in
    # non-finite parameter values" when X is scaled to near the float64 maximum,
    # causing overflow. Convergence and runtime warnings are ignored.
    # Check ElasticNet throws ValueError when dealing with non-finite parameter
    # values
    rng = np.random.RandomState(0)
    n_samples = 10
    fmax = np.finfo(np.float64).max
    X = fmax * rng.uniform(size=(n_samples, 2))
    y = rng.randint(0, 2, size=n_samples)

    clf = ElasticNet(alpha=0.1)
    msg = "Coordinate descent iterations resulted in non-finite parameter values"
    with pytest.raises(ValueError, match=msg):
        clf.fit(X, y)


def test_lasso_toy():
    # [FUNCTION DOC]
    # Verifies Lasso closed-form solutions on X = [-1, 0, 1], y = X for alpha 1e-8, 0.1,
    # 0.5 and 1 (coef 1, 0.85, 0.25, 0): coef_, predictions on [2, 3, 4] and zero dual
    # gap.
    # Test Lasso on a toy example for various values of alpha.
    # When validating this against glmnet notice that glmnet divides it
    # against nobs.

    X = [[-1], [0], [1]]
    Y = [-1, 0, 1]  # just a straight line
    T = [[2], [3], [4]]  # test sample

    clf = Lasso(alpha=1e-8)
    clf.fit(X, Y)
    pred = clf.predict(T)
    assert_array_almost_equal(clf.coef_, [1])
    assert_array_almost_equal(pred, [2, 3, 4])
    assert_almost_equal(clf.dual_gap_, 0)

    clf = Lasso(alpha=0.1)
    clf.fit(X, Y)
    pred = clf.predict(T)
    assert_array_almost_equal(clf.coef_, [0.85])
    assert_array_almost_equal(pred, [1.7, 2.55, 3.4])
    assert_almost_equal(clf.dual_gap_, 0)

    clf = Lasso(alpha=0.5)
    clf.fit(X, Y)
    pred = clf.predict(T)
    assert_array_almost_equal(clf.coef_, [0.25])
    assert_array_almost_equal(pred, [0.5, 0.75, 1.0])
    assert_almost_equal(clf.dual_gap_, 0)

    clf = Lasso(alpha=1)
    clf.fit(X, Y)
    pred = clf.predict(T)
    assert_array_almost_equal(clf.coef_, [0.0])
    assert_array_almost_equal(pred, [0, 0, 0])
    assert_almost_equal(clf.dual_gap_, 0)


def test_enet_toy():
    # [FUNCTION DOC]
    # Verifies ElasticNet toy solutions on the same line data: l1_ratio=1 with tiny
    #   alpha
    # behaves like Lasso; alpha=0.5, l1_ratio=0.3 gives coef ~0.50819 with precompute
    # False, True and an explicit Gram matrix; alpha=0.5, l1_ratio=0.5 gives ~0.45454.
    # Also checks predictions and zero dual gap.
    # Test ElasticNet for various parameters of alpha and l1_ratio.
    # Actually, the parameters alpha = 0 should not be allowed. However,
    # we test it as a border case.
    # ElasticNet is tested with and without precomputed Gram matrix

    X = np.array([[-1.0], [0.0], [1.0]])
    Y = [-1, 0, 1]  # just a straight line
    T = [[2.0], [3.0], [4.0]]  # test sample

    # this should be the same as lasso
    clf = ElasticNet(alpha=1e-8, l1_ratio=1.0)
    clf.fit(X, Y)
    pred = clf.predict(T)
    assert_array_almost_equal(clf.coef_, [1])
    assert_array_almost_equal(pred, [2, 3, 4])
    assert_almost_equal(clf.dual_gap_, 0)

    clf = ElasticNet(alpha=0.5, l1_ratio=0.3, max_iter=100, precompute=False)
    clf.fit(X, Y)
    pred = clf.predict(T)
    assert_array_almost_equal(clf.coef_, [0.50819], decimal=3)
    assert_array_almost_equal(pred, [1.0163, 1.5245, 2.0327], decimal=3)
    assert_almost_equal(clf.dual_gap_, 0)

    clf.set_params(max_iter=100, precompute=True)
    clf.fit(X, Y)  # with Gram
    pred = clf.predict(T)
    assert_array_almost_equal(clf.coef_, [0.50819], decimal=3)
    assert_array_almost_equal(pred, [1.0163, 1.5245, 2.0327], decimal=3)
    assert_almost_equal(clf.dual_gap_, 0)

    clf.set_params(max_iter=100, precompute=np.dot(X.T, X))
    clf.fit(X, Y)  # with Gram
    pred = clf.predict(T)
    assert_array_almost_equal(clf.coef_, [0.50819], decimal=3)
    assert_array_almost_equal(pred, [1.0163, 1.5245, 2.0327], decimal=3)
    assert_almost_equal(clf.dual_gap_, 0)

    clf = ElasticNet(alpha=0.5, l1_ratio=0.5)
    clf.fit(X, Y)
    pred = clf.predict(T)
    assert_array_almost_equal(clf.coef_, [0.45454], 3)
    assert_array_almost_equal(pred, [0.9090, 1.3636, 1.8181], 3)
    assert_almost_equal(clf.dual_gap_, 0)


def test_lasso_dual_gap():
    # [FUNCTION DOC]
    # Verifies dual_gap_ equals primal minus dual objective for a Lasso without
    #   intercept
    # on a 10x30 problem: primal = half mean squared residual + alpha L1 norm; dual
    #   point
    # is the residual rescaled so that the infinity norm of X^T theta / n_samples is at
    # most alpha; dual = half (mean y^2 - mean (y - rescaled R)^2).
    """
    Check that Lasso.dual_gap_ matches its objective formulation, with the
    datafit normalized by n_samples
    """
    X, y, _, _ = build_dataset(n_samples=10, n_features=30)
    n_samples = len(y)
    alpha = 0.01 * np.max(np.abs(X.T @ y)) / n_samples
    clf = Lasso(alpha=alpha, fit_intercept=False).fit(X, y)
    w = clf.coef_
    R = y - X @ w
    primal = 0.5 * np.mean(R**2) + clf.alpha * np.sum(np.abs(w))
    # dual pt: R / n_samples, dual constraint: norm(X.T @ theta, inf) <= alpha
    R /= np.max(np.abs(X.T @ R) / (n_samples * alpha))
    dual = 0.5 * (np.mean(y**2) - np.mean((y - R) ** 2))
    assert_allclose(clf.dual_gap_, primal - dual)


def build_dataset(n_samples=50, n_features=200, n_informative_features=10, n_targets=1):
    # [FUNCTION DOC]
    # HELPER: Builds an ill-posed noiseless linear regression problem. With a fixed
    # RandomState(0), draws Gaussian weights (vector, or matrix (n_features, n_targets)
    # when n_targets > 1), zeroes all but the first n_informative_features rows, draws
    # Gaussian train and test X and computes y as X times weights.
    # Inputs: n_samples (50), n_features (200), n_informative_features (10),
    # n_targets (1).
    # Outputs: X, y, X_test, y_test; y is 1-D if n_targets == 1, else (n_samples,
    # n_targets). Deterministic.
    """
    build an ill-posed linear regression problem with many noisy features and
    comparatively few samples
    """
    random_state = np.random.RandomState(0)
    if n_targets > 1:
        w = random_state.randn(n_features, n_targets)
    else:
        w = random_state.randn(n_features)
    w[n_informative_features:] = 0.0
    X = random_state.randn(n_samples, n_features)
    y = np.dot(X, w)
    X_test = random_state.randn(n_samples, n_features)
    y_test = np.dot(X_test, w)
    return X, y, X_test, y_test


def test_lasso_cv():
    # [FUNCTION DOC]
    # Verifies LassoCV with a 10-alpha grid and cv=3 selects alpha_ ~0.056 with and
    # without precompute, that LassoLarsCV selects an alpha within one grid position of
    # it, that the interpolated LARS CV MSE matches at alphas_[5] (rtol 1e-2), and that
    # test R2 exceeds 0.99.
    X, y, X_test, y_test = build_dataset()
    max_iter = 150
    clf = LassoCV(alphas=10, eps=1e-3, max_iter=max_iter, cv=3).fit(X, y)
    assert_almost_equal(clf.alpha_, 0.056, 2)

    clf = LassoCV(alphas=10, eps=1e-3, max_iter=max_iter, precompute=True, cv=3)
    clf.fit(X, y)
    assert_almost_equal(clf.alpha_, 0.056, 2)

    # Check that the lars and the coordinate descent implementation
    # select a similar alpha
    lars = LassoLarsCV(max_iter=30, cv=3).fit(X, y)
    # for this we check that they don't fall in the grid of
    # clf.alphas further than 1
    assert (
        np.abs(
            np.searchsorted(clf.alphas_[::-1], lars.alpha_)
            - np.searchsorted(clf.alphas_[::-1], clf.alpha_)
        )
        <= 1
    )
    # check that they also give a similar MSE
    mse_lars = interpolate.interp1d(lars.cv_alphas_, lars.mse_path_.T)
    assert_allclose(mse_lars(clf.alphas_[5]).mean(), clf.mse_path_[5].mean(), rtol=1e-2)

    # test set
    assert clf.score(X_test, y_test) > 0.99


def test_lasso_cv_with_some_model_selection():
    # [FUNCTION DOC]
    # Smoke test: a StandardScaler plus LassoCV(cv=ShuffleSplit) pipeline fits on the
    # diabetes data without error.
    from sklearn import datasets
    from sklearn.model_selection import ShuffleSplit

    diabetes = datasets.load_diabetes()
    X = diabetes.data
    y = diabetes.target

    pipe = make_pipeline(StandardScaler(), LassoCV(cv=ShuffleSplit(random_state=0)))
    pipe.fit(X, y)


def test_lasso_cv_positive_constraint():
    # [FUNCTION DOC]
    # Verifies LassoCV(positive=True) yields nonnegative coef_ on data where the
    # unconstrained fit has a negative coefficient.
    X, y, X_test, y_test = build_dataset()
    max_iter = 500

    # Ensure the unconstrained fit has a negative coefficient
    clf_unconstrained = LassoCV(alphas=3, eps=1e-1, max_iter=max_iter, cv=2, n_jobs=1)
    clf_unconstrained.fit(X, y)
    assert min(clf_unconstrained.coef_) < 0

    # On same data, constrained fit has non-negative coefficients
    clf_constrained = LassoCV(
        alphas=3, eps=1e-1, max_iter=max_iter, positive=True, cv=2, n_jobs=1
    )
    clf_constrained.fit(X, y)
    assert min(clf_constrained.coef_) >= 0


@pytest.mark.parametrize(
    "alphas, err_type, err_msg",
    [
        ((1, -1, -100), ValueError, r"alphas\[1\] == -1, must be >= 0.0."),
        (
            (-0.1, -1.0, -10.0),
            ValueError,
            r"alphas\[0\] == -0.1, must be >= 0.0.",
        ),
        (
            (1, 1.0, "1"),
            TypeError,
            r"alphas\[2\] must be an instance of float, not str",
        ),
    ],
)
def test_lassocv_alphas_validation(alphas, err_type, err_msg):
    # [FUNCTION DOC]
    # Verifies LassoCV fit raises the expected ValueError for negative alphas and
    # TypeError for a non-numeric alpha, with the indexed error message.
    """Check the `alphas` validation in LassoCV."""

    n_samples, n_features = 5, 5
    rng = np.random.RandomState(0)
    X = rng.randn(n_samples, n_features)
    y = rng.randint(0, 2, n_samples)
    lassocv = LassoCV(alphas=alphas)
    with pytest.raises(err_type, match=err_msg):
        lassocv.fit(X, y)


def test_lasso_path_return_models_vs_new_return_gives_same_coefficients():
    # [FUNCTION DOC]
    # Verifies lasso_path at alphas [5, 1, 0.5] matches the LARS lasso path, both
    #   linearly
    # interpolated in alpha, to 1 decimal on a 3x2 toy problem.
    # Test that lasso_path with lars_path style output gives the
    # same result

    # Some toy data
    X = np.array([[1, 2, 3.1], [2.3, 5.4, 4.3]]).T
    y = np.array([1, 2, 3.1])
    alphas = [5.0, 1.0, 0.5]

    # Use lars_path and lasso_path(new output) with 1D linear interpolation
    # to compute the same path
    alphas_lars, _, coef_path_lars = lars_path(X, y, method="lasso")
    coef_path_cont_lars = interpolate.interp1d(
        alphas_lars[::-1], coef_path_lars[:, ::-1]
    )
    alphas_lasso2, coef_path_lasso2, _ = lasso_path(X, y, alphas=alphas)
    coef_path_cont_lasso = interpolate.interp1d(
        alphas_lasso2[::-1], coef_path_lasso2[:, ::-1]
    )

    assert_array_almost_equal(
        coef_path_cont_lasso(alphas), coef_path_cont_lars(alphas), decimal=1
    )


def test_enet_path():
    # [FUNCTION DOC]
    # Verifies ElasticNetCV on a dense-truth, well-conditioned 200x100 problem picks the
    # smallest alpha and smallest l1_ratio (more ridge-like), with and without
    #   precompute,
    # and scores test R2 > 0.99; MultiTaskElasticNetCV on 3 targets scores > 0.99 with
    # coef_ shape (3, 10); and ElasticNetCV and MultiTaskElasticNetCV on a single target
    # pick the same alpha_ and l1_ratio_.
    # We use a large number of samples and of informative features so that
    # the l1_ratio selected is more toward ridge than lasso
    X, y, X_test, y_test = build_dataset(
        n_samples=200, n_features=100, n_informative_features=100
    )
    max_iter = 150

    # Here we have a small number of iterations, and thus the
    # ElasticNet might not converge. This is to speed up tests
    clf = ElasticNetCV(
        alphas=[0.01, 0.05, 0.1], eps=2e-3, l1_ratio=[0.5, 0.7], cv=3, max_iter=max_iter
    )
    clf.fit(X, y)
    # Well-conditioned settings, we should have selected our
    # smallest penalty
    assert_almost_equal(clf.alpha_, min(clf.alphas_))
    # Non-sparse ground truth: we should have selected an elastic-net
    # that is closer to ridge than to lasso
    assert clf.l1_ratio_ == min(clf.l1_ratio)

    clf = ElasticNetCV(
        alphas=[0.01, 0.05, 0.1],
        eps=2e-3,
        l1_ratio=[0.5, 0.7],
        cv=3,
        max_iter=max_iter,
        precompute=True,
    )
    clf.fit(X, y)

    # Well-conditioned settings, we should have selected our
    # smallest penalty
    assert_almost_equal(clf.alpha_, min(clf.alphas_))
    # Non-sparse ground truth: we should have selected an elastic-net
    # that is closer to ridge than to lasso
    assert clf.l1_ratio_ == min(clf.l1_ratio)

    # We are in well-conditioned settings with low noise: we should
    # have a good test-set performance
    assert clf.score(X_test, y_test) > 0.99

    # Multi-output/target case
    X, y, X_test, y_test = build_dataset(n_features=10, n_targets=3)
    clf = MultiTaskElasticNetCV(
        alphas=5, eps=2e-3, l1_ratio=[0.5, 0.7], cv=3, max_iter=max_iter
    )
    clf.fit(X, y)
    # We are in well-conditioned settings with low noise: we should
    # have a good test-set performance
    assert clf.score(X_test, y_test) > 0.99
    assert clf.coef_.shape == (3, 10)

    # Mono-output should have same cross-validated alpha_ and l1_ratio_
    # in both cases.
    X, y, _, _ = build_dataset(n_features=10)
    clf1 = ElasticNetCV(alphas=5, eps=2e-3, l1_ratio=[0.5, 0.7])
    clf1.fit(X, y)
    clf2 = MultiTaskElasticNetCV(alphas=5, eps=2e-3, l1_ratio=[0.5, 0.7])
    clf2.fit(X, y[:, np.newaxis])
    assert_almost_equal(clf1.l1_ratio_, clf2.l1_ratio_)
    assert_almost_equal(clf1.alpha_, clf2.alpha_)


def test_warm_start():
    # [FUNCTION DOC]
    # Verifies warm_start: two ElasticNet fits of max_iter=5 each give the same coef_ as
    #   a
    # single cold fit with max_iter=10 (convergence warnings ignored).
    X, y, _, _ = build_dataset()
    clf = ElasticNet(alpha=0.1, max_iter=5, warm_start=True)
    ignore_warnings(clf.fit)(X, y)
    ignore_warnings(clf.fit)(X, y)  # do a second round with 5 iterations

    clf2 = ElasticNet(alpha=0.1, max_iter=10)
    ignore_warnings(clf2.fit)(X, y)
    assert_array_almost_equal(clf2.coef_, clf.coef_)


@pytest.mark.filterwarnings("ignore:.*with no regularization.*:UserWarning")
def test_lasso_alpha_warning():
    # [FUNCTION DOC]
    # Verifies Lasso(alpha=0) emits the UserWarning recommending LinearRegression.
    X = [[-1], [0], [1]]
    Y = [-1, 0, 1]  # just a straight line

    clf = Lasso(alpha=0)
    warning_message = (
        "With alpha=0, this algorithm does not "
        "converge well. You are advised to use the "
        "LinearRegression estimator"
    )
    with pytest.warns(UserWarning, match=warning_message):
        clf.fit(X, Y)


def test_lasso_positive_constraint():
    # [FUNCTION DOC]
    # Verifies Lasso(positive=True), with and without precompute, gives nonnegative
    #   coef_
    # on data with a negative slope.
    X = [[-1], [0], [1]]
    y = [1, 0, -1]  # just a straight line with negative slope

    lasso = Lasso(alpha=0.1, positive=True)
    lasso.fit(X, y)
    assert min(lasso.coef_) >= 0

    lasso = Lasso(alpha=0.1, precompute=True, positive=True)
    lasso.fit(X, y)
    assert min(lasso.coef_) >= 0


def test_enet_positive_constraint():
    # [FUNCTION DOC]
    # Verifies ElasticNet(positive=True) gives nonnegative coef_ on a negative-slope
    #   line.
    X = [[-1], [0], [1]]
    y = [1, 0, -1]  # just a straight line with negative slope

    enet = ElasticNet(alpha=0.1, positive=True)
    enet.fit(X, y)
    assert min(enet.coef_) >= 0


def test_enet_cv_positive_constraint():
    # [FUNCTION DOC]
    # Verifies ElasticNetCV(positive=True) gives nonnegative coef_ where the
    #   unconstrained
    # fit has a negative coefficient.
    X, y, X_test, y_test = build_dataset()
    max_iter = 500

    # Ensure the unconstrained fit has a negative coefficient
    enetcv_unconstrained = ElasticNetCV(
        alphas=3, eps=1e-1, max_iter=max_iter, cv=2, n_jobs=1
    )
    enetcv_unconstrained.fit(X, y)
    assert min(enetcv_unconstrained.coef_) < 0

    # On same data, constrained fit has non-negative coefficients
    enetcv_constrained = ElasticNetCV(
        alphas=3, eps=1e-1, max_iter=max_iter, cv=2, positive=True, n_jobs=1
    )
    enetcv_constrained.fit(X, y)
    assert min(enetcv_constrained.coef_) >= 0


def test_uniform_targets():
    # [FUNCTION DOC]
    # Verifies CV estimators on constant targets (0 and 5; for multi-task, columns c and
    # 2c) predict the constant exactly and produce an alpha grid of three copies of
    # float resolution (alpha_max is zero).
    enet = ElasticNetCV(alphas=3)
    m_enet = MultiTaskElasticNetCV(alphas=3)
    lasso = LassoCV(alphas=3)
    m_lasso = MultiTaskLassoCV(alphas=3)

    models_single_task = (enet, lasso)
    models_multi_task = (m_enet, m_lasso)

    rng = np.random.RandomState(0)

    X_train = rng.random_sample(size=(10, 3))
    X_test = rng.random_sample(size=(10, 3))

    y1 = np.empty(10)
    y2 = np.empty((10, 2))

    for model in models_single_task:
        for y_values in (0, 5):
            y1.fill(y_values)
            assert_array_equal(model.fit(X_train, y1).predict(X_test), y1)
            assert_array_equal(model.alphas_, [np.finfo(float).resolution] * 3)

    for model in models_multi_task:
        for y_values in (0, 5):
            y2[:, 0].fill(y_values)
            y2[:, 1].fill(2 * y_values)
            assert_array_equal(model.fit(X_train, y2).predict(X_test), y2)
            assert_array_equal(model.alphas_, [np.finfo(float).resolution] * 3)


@pytest.mark.filterwarnings("error::sklearn.exceptions.ConvergenceWarning")
def test_multi_task_lasso_vs_skglm():
    # [FUNCTION DOC]
    # Verifies MultiTaskLasso (alpha=0.1, tol 1e-10, 5000 iterations) matches skglm 0.5
    # reference values: objective 0.4965993692547902, intercepts and per-task identical
    # coefficients, on a 5x4 Vandermonde X and 3 tasks. Fails on ConvergenceWarning.
    """Test that MultiTaskLasso gives same results as the one from skglm.

    To reproduce numbers, just use
    from skglm import MultiTaskLasso
    """
    # Numbers are with skglm version 0.5.
    n_samples, n_features, n_tasks = 5, 4, 3
    X = np.vander(np.arange(n_samples), n_features)
    Y = np.arange(n_samples * n_tasks).reshape(n_samples, n_tasks)

    def obj(W, X, y, alpha):
        # [FUNCTION DOC]
        # HELPER: Multi-task Lasso objective for a stacked matrix W whose last column is
        #   the
        # intercept: squared Frobenius norm of residuals divided by 2 * n_samples plus
        #   alpha
        # times the L2,1 norm (sum over features of the column L2 norm across tasks).
        # Inputs: W (n_tasks, n_features + 1), X, y, alpha. Outputs: float.
        # KNOWN ISSUE: uses the enclosing Y instead of the y parameter (same value
        #   here).
        intercept = W[:, -1]
        W = W[:, :-1]
        l21_norm = np.sqrt(np.sum(W**2, axis=0)).sum()
        return (
            np.linalg.norm(Y - X @ W.T - intercept, ord="fro") ** 2 / (2 * n_samples)
            + alpha * l21_norm
        )

    alpha = 0.1
    # TODO: The high number of iterations are required for convergence and show room
    # for improvement of the CD algorithm.
    m = MultiTaskLasso(alpha=alpha, tol=1e-10, max_iter=5000).fit(X, Y)
    assert_allclose(
        obj(np.c_[m.coef_, m.intercept_], X, Y, alpha=alpha),
        0.4965993692547902,
        rtol=1e-10,
    )
    assert_allclose(
        m.intercept_, [0.219942959407, 1.219942959407, 2.219942959407], rtol=1e-7
    )
    assert_allclose(
        m.coef_,
        np.tile([-0.032075014794, 0.25430904614, 2.44785152982, 0], (n_tasks, 1)),
        rtol=1e-6,
    )


def test_multi_task_lasso_and_enet():
    # [FUNCTION DOC]
    # Verifies MultiTaskLasso and MultiTaskElasticNet on two identical target columns
    #   give
    # identical per-task coef_ and a small positive dual gap, and that max_iter=1 emits
    # the "Objective did not converge" ConvergenceWarning.
    X, y, X_test, y_test = build_dataset()
    Y = np.c_[y, y]
    # Y_test = np.c_[y_test, y_test]
    clf = MultiTaskLasso(alpha=1, tol=1e-8).fit(X, Y)
    assert 0 < clf.dual_gap_ < 1e-5
    assert_array_almost_equal(clf.coef_[0], clf.coef_[1])

    clf = MultiTaskElasticNet(alpha=1, tol=1e-8).fit(X, Y)
    assert 0 < clf.dual_gap_ < 1e-5
    assert_array_almost_equal(clf.coef_[0], clf.coef_[1])

    clf = MultiTaskElasticNet(alpha=1.0, tol=1e-8, max_iter=1)
    warning_message = (
        "Objective did not converge. You might want to "
        "increase the number of iterations."
    )
    with pytest.warns(ConvergenceWarning, match=warning_message):
        clf.fit(X, Y)


def test_lasso_readonly_data():
    # [FUNCTION DOC]
    # Verifies Lasso(alpha=0.5) works on read-only memmapped X and y, reproducing the
    #   toy
    # solution coef 0.25.
    X = np.array([[-1], [0], [1]])
    Y = np.array([-1, 0, 1])  # just a straight line
    T = np.array([[2], [3], [4]])  # test sample
    with TempMemmap((X, Y)) as (X, Y):
        clf = Lasso(alpha=0.5)
        clf.fit(X, Y)
        pred = clf.predict(T)
        assert_array_almost_equal(clf.coef_, [0.25])
        assert_array_almost_equal(pred, [0.5, 0.75, 1.0])
        assert_almost_equal(clf.dual_gap_, 0)


def test_multi_task_lasso_readonly_data():
    # [FUNCTION DOC]
    # Verifies MultiTaskLasso fits on memmapped (read-only) X with identical per-task
    # coefficients.
    # KNOWN ISSUE: Y is rebuilt from the in-memory y inside the context, so only X is
    # read-only; the memmapped Y is unused.
    X, y, X_test, y_test = build_dataset()
    Y = np.c_[y, y]
    with TempMemmap((X, Y)) as (X, Y):
        Y = np.c_[y, y]
        clf = MultiTaskLasso(alpha=1, tol=1e-8).fit(X, Y)
        assert 0 < clf.dual_gap_ < 1e-5
        assert_array_almost_equal(clf.coef_[0], clf.coef_[1])


def test_enet_multitarget():
    # [FUNCTION DOC]
    # Verifies ElasticNet on a 3-target y gives, per target, the same coef_, intercept_
    #   and
    # dual_gap_ as separate single-target fits.
    n_targets = 3
    X, y, _, _ = build_dataset(
        n_samples=10, n_features=8, n_informative_features=10, n_targets=n_targets
    )
    estimator = ElasticNet(alpha=0.01)
    estimator.fit(X, y)
    coef, intercept, dual_gap = (
        estimator.coef_,
        estimator.intercept_,
        estimator.dual_gap_,
    )

    for k in range(n_targets):
        estimator.fit(X, y[:, k])
        assert_array_almost_equal(coef[k, :], estimator.coef_)
        assert_array_almost_equal(intercept[k], estimator.intercept_)
        assert_array_almost_equal(dual_gap[k], estimator.dual_gap_)


def test_multioutput_enetcv_error():
    # [FUNCTION DOC]
    # Verifies ElasticNetCV raises ValueError on a 2-D target.
    rng = np.random.RandomState(0)
    X = rng.randn(10, 2)
    y = rng.randn(10, 2)
    clf = ElasticNetCV()
    with pytest.raises(ValueError):
        clf.fit(X, y)


def test_multitask_enet_and_lasso_cv():
    # [FUNCTION DOC]
    # Verifies MultiTaskElasticNetCV and MultiTaskLassoCV select reference alphas
    #   (0.00556,
    # 0.00278) and that fitted attributes have the expected shapes: coef_ (3,
    #   n_features),
    # intercept_ (3,), mse_path_ (n_l1_ratio, n_alphas, n_folds) or (n_alphas, n_folds),
    # alphas_ (n_l1_ratio, n_alphas) or (n_alphas,); l1_ratio_ selected is 0.5.
    X, y, _, _ = build_dataset(n_features=50, n_targets=3)
    clf = MultiTaskElasticNetCV(cv=3).fit(X, y)
    assert_almost_equal(clf.alpha_, 0.00556, 3)
    clf = MultiTaskLassoCV(cv=3, tol=1e-6).fit(X, y)
    assert_almost_equal(clf.alpha_, 0.00278, 3)

    X, y, _, _ = build_dataset(n_targets=3)
    clf = MultiTaskElasticNetCV(
        alphas=10, eps=1e-3, max_iter=200, l1_ratio=[0.3, 0.5], tol=1e-3, cv=3
    )
    clf.fit(X, y)
    assert 0.5 == clf.l1_ratio_
    assert (3, X.shape[1]) == clf.coef_.shape
    assert (3,) == clf.intercept_.shape
    assert (2, 10, 3) == clf.mse_path_.shape
    assert (2, 10) == clf.alphas_.shape

    X, y, _, _ = build_dataset(n_targets=3)
    clf = MultiTaskLassoCV(alphas=10, eps=1e-3, max_iter=500, tol=1e-3, cv=3)
    clf.fit(X, y)
    assert (3, X.shape[1]) == clf.coef_.shape
    assert (3,) == clf.intercept_.shape
    assert (10, 3) == clf.mse_path_.shape
    assert 10 == len(clf.alphas_)


def test_1d_multioutput_enet_and_multitask_enet_cv():
    # [FUNCTION DOC]
    # Verifies ElasticNetCV on y and MultiTaskElasticNetCV on y as one column agree on
    # l1_ratio_, alpha_, coef_ and intercept_.
    X, y, _, _ = build_dataset(n_features=10)
    y = y[:, np.newaxis]
    clf = ElasticNetCV(alphas=5, eps=2e-3, l1_ratio=[0.5, 0.7])
    clf.fit(X, y[:, 0])
    clf1 = MultiTaskElasticNetCV(alphas=5, eps=2e-3, l1_ratio=[0.5, 0.7])
    clf1.fit(X, y)
    assert_almost_equal(clf.l1_ratio_, clf1.l1_ratio_)
    assert_almost_equal(clf.alpha_, clf1.alpha_)
    assert_almost_equal(clf.coef_, clf1.coef_[0])
    assert_almost_equal(clf.intercept_, clf1.intercept_[0])


def test_1d_multioutput_lasso_and_multitask_lasso_cv():
    # [FUNCTION DOC]
    # Same as the enet variant for LassoCV versus MultiTaskLassoCV (alpha_, coef_,
    # intercept_).
    X, y, _, _ = build_dataset(n_features=10)
    y = y[:, np.newaxis]
    clf = LassoCV(alphas=5, eps=2e-3)
    clf.fit(X, y[:, 0])
    clf1 = MultiTaskLassoCV(alphas=5, eps=2e-3)
    clf1.fit(X, y)
    assert_almost_equal(clf.alpha_, clf1.alpha_)
    assert_almost_equal(clf.coef_, clf1.coef_[0])
    assert_almost_equal(clf.intercept_, clf1.intercept_[0])


@pytest.mark.parametrize(
    "estimator", [ElasticNetCV, LassoCV, MultiTaskElasticNetCV, MultiTaskLassoCV]
)
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_sparse_input_dtype_enet_and_lassocv(estimator, csr_container):
    # [FUNCTION DOC]
    # Verifies CV estimators on CSR X in float64 and float32 select the same alpha_ and
    # coef_ (rtol 1e-5); multi-task estimators get 2 targets.
    if issubclass(estimator, (MultiTaskElasticNetCV, MultiTaskLassoCV)):
        n_targets = 2
    else:
        n_targets = 1
    X, y, _, _ = build_dataset(n_targets=n_targets, n_features=10)
    reg = estimator(alphas=5)
    reg.fit(csr_container(X), y)
    reg1 = estimator(alphas=5)
    reg1.fit(csr_container(X, dtype=np.float32), y)
    assert_allclose(reg.alpha_, reg1.alpha_, rtol=1e-5)
    assert_allclose(reg.coef_, reg1.coef_, rtol=1e-5)


def test_elasticnet_precompute_incorrect_gram():
    # [FUNCTION DOC]
    # Verifies ElasticNet raises ValueError ("Gram matrix ... did not pass validation")
    # when the precomputed Gram matrix comes from unrelated random data.
    # check that passing an invalid precomputed Gram matrix will raise an
    # error.
    X, y, _, _ = build_dataset()

    rng = np.random.RandomState(0)

    X_centered = X - np.average(X, axis=0)
    garbage = rng.standard_normal(X.shape)
    precompute = np.dot(garbage.T, garbage)

    clf = ElasticNet(alpha=0.01, precompute=precompute)
    msg = "Gram matrix.*did not pass validation.*"
    with pytest.raises(ValueError, match=msg):
        clf.fit(X_centered, y)


def test_elasticnet_precompute_gram_weighted_samples():
    # [FUNCTION DOC]
    # Verifies a user Gram matrix built from weighted-centered, sqrt-weight-scaled X
    # (weights normalized to sum to n_samples) gives the same coef_ as internal
    # computation with the same lognormal sample weights.
    # check the equivalence between passing a precomputed Gram matrix and
    # internal computation using sample weights.
    X, y, _, _ = build_dataset()

    rng = np.random.RandomState(0)
    sample_weight = rng.lognormal(size=y.shape)

    w_norm = sample_weight * (y.shape / np.sum(sample_weight))
    X_c = X - np.average(X, axis=0, weights=w_norm)
    X_r = X_c * np.sqrt(w_norm)[:, np.newaxis]
    gram = np.dot(X_r.T, X_r)

    clf1 = ElasticNet(alpha=0.01, precompute=gram)
    clf1.fit(X_c, y, sample_weight=sample_weight)

    clf2 = ElasticNet(alpha=0.01, precompute=False)
    clf2.fit(X, y, sample_weight=sample_weight)

    assert_allclose(clf1.coef_, clf2.coef_)


def test_elasticnet_precompute_gram():
    # [FUNCTION DOC]
    # Verifies the dtype-aware Gram validation accepts a float32 Gram that matches X_c^T
    # X_c only within float32 tolerance (PR #22059, issue #21997), giving the same coef_
    # as precompute=False on uncentered X.
    # Check the dtype-aware check for a precomputed Gram matrix
    # (see https://github.com/scikit-learn/scikit-learn/pull/22059
    # and https://github.com/scikit-learn/scikit-learn/issues/21997).
    # Here: (X_c.T, X_c)[2, 3] is not equal to np.dot(X_c[:, 2], X_c[:, 3])
    # but within tolerance for np.float32

    rng = np.random.RandomState(58)
    X = rng.binomial(1, 0.25, (1000, 4)).astype(np.float32)
    y = rng.rand(1000).astype(np.float32)

    X_c = X - np.average(X, axis=0)
    gram = np.dot(X_c.T, X_c)

    clf1 = ElasticNet(alpha=0.01, precompute=gram)
    clf1.fit(X_c, y)

    clf2 = ElasticNet(alpha=0.01, precompute=False)
    clf2.fit(X, y)

    assert_allclose(clf1.coef_, clf2.coef_)


@pytest.mark.parametrize("sparse_csr_type", [sparse.csr_array, sparse.csr_matrix])
@pytest.mark.parametrize("sparse_X", [True, False])
def test_warm_start_convergence(sparse_X, sparse_csr_type):
    # [FUNCTION DOC]
    # Verifies n_iter_ is reproducible on refit with warm_start=False and becomes 0 when
    # refitting with warm_start=True (dual gap checked before the loop), dense or CSR X.
    X, y, _, _ = build_dataset()
    if sparse_X:
        X = sparse_csr_type(X)
    model = ElasticNet(alpha=1e-3, tol=1e-3).fit(X, y)
    n_iter_reference = model.n_iter_

    # This dataset is not trivial enough for the model to converge in one pass.
    assert n_iter_reference > 2

    # Check that n_iter_ is invariant to multiple calls to fit
    # when warm_start=False, all else being equal.
    model.fit(X, y)
    n_iter_cold_start = model.n_iter_
    assert n_iter_cold_start == n_iter_reference

    model.set_params(warm_start=True)
    model.fit(X, y)
    n_iter_warm_start = model.n_iter_
    # coordinate descent checks dual gap before entering the main loop
    assert n_iter_warm_start == 0


def test_warm_start_convergence_with_regularizer_decrement():
    # [FUNCTION DOC]
    # Verifies on diabetes data that a weakly regularized ElasticNet needs more
    #   iterations
    # than a 10x more regularized one, and that warm starting the weak problem from the
    # strong solution converges in fewer iterations than from zero.
    X, y = load_diabetes(return_X_y=True)

    # Train a model to converge on a lightly regularized problem
    final_alpha = 1e-5
    low_reg_model = ElasticNet(alpha=final_alpha).fit(X, y)

    # Fitting a new model on a more regularized version of the same problem.
    # Fitting with high regularization is easier it should converge faster
    # in general.
    high_reg_model = ElasticNet(alpha=final_alpha * 10).fit(X, y)
    assert low_reg_model.n_iter_ > high_reg_model.n_iter_

    # Fit the solution to the original, less regularized version of the
    # problem but from the solution of the highly regularized variant of
    # the problem as a better starting point. This should also converge
    # faster than the original model that starts from zero.
    warm_low_reg_model = deepcopy(high_reg_model)
    warm_low_reg_model.set_params(warm_start=True, alpha=final_alpha)
    warm_low_reg_model.fit(X, y)
    assert low_reg_model.n_iter_ > warm_low_reg_model.n_iter_


@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_random_descent(csr_container):
    # [FUNCTION DOC]
    # Verifies selection="random" (random_state=42) and "cyclic" converge to the same
    #   coef_
    # and intercept_ with tol 1e-9 for: Gram-trick dense fit, a wide problem (X.T,
    # n_features > n_samples) without Gram, CSR input, and MultiTaskElasticNet on CSR.
    # Test that both random and cyclic selection give the same results.
    # Ensure that the test models fully converge and check a wide
    # range of conditions.

    # This uses the coordinate descent algo using the gram trick.
    X, y, _, _ = build_dataset(n_samples=50, n_features=20)
    clf_cyclic = ElasticNet(selection="cyclic", tol=1e-9)
    clf_cyclic.fit(X, y)
    clf_random = ElasticNet(selection="random", tol=1e-9, random_state=42)
    clf_random.fit(X, y)
    assert_allclose(clf_cyclic.coef_, clf_random.coef_)
    assert_allclose(clf_cyclic.intercept_, clf_random.intercept_)

    # This uses the descent algo without the gram trick
    clf_cyclic = ElasticNet(selection="cyclic", tol=1e-9)
    clf_cyclic.fit(X.T, y[:20])
    clf_random = ElasticNet(selection="random", tol=1e-9, random_state=42)
    clf_random.fit(X.T, y[:20])
    assert_allclose(clf_cyclic.coef_, clf_random.coef_)
    assert_allclose(clf_cyclic.intercept_, clf_random.intercept_)

    # Sparse Case
    clf_cyclic = ElasticNet(selection="cyclic", tol=1e-9)
    clf_cyclic.fit(csr_container(X), y)
    clf_random = ElasticNet(selection="random", tol=1e-9, random_state=42)
    clf_random.fit(csr_container(X), y)
    assert_allclose(clf_cyclic.coef_, clf_random.coef_)
    assert_allclose(clf_cyclic.intercept_, clf_random.intercept_)

    # Multioutput case.
    new_y = np.hstack((y[:, np.newaxis], y[:, np.newaxis]))
    clf_cyclic = MultiTaskElasticNet(selection="cyclic", tol=1e-9)
    clf_cyclic.fit(csr_container(X), new_y)
    clf_random = MultiTaskElasticNet(selection="random", tol=1e-9, random_state=42)
    clf_random.fit(csr_container(X), new_y)
    assert_allclose(clf_cyclic.coef_, clf_random.coef_)
    assert_allclose(clf_cyclic.intercept_, clf_random.intercept_)


@pytest.mark.parametrize("path", [enet_path, lasso_path])
@pytest.mark.parametrize("n_targets", [1, 2])
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_sparse_dense_descent_paths(path, n_targets, csr_container):
    # [FUNCTION DOC]
    # Verifies enet_path and lasso_path give the same coefficient paths for dense and
    #   CSR
    # X, for 1 and 2 targets.
    # Test that dense and sparse input give the same result for descent paths.
    X, y, _, _ = build_dataset(n_targets=n_targets, n_samples=50, n_features=20)
    csr = csr_container(X)
    _, coefs, _ = path(X, y, tol=1e-10)
    _, sparse_coefs, _ = path(csr, y, tol=1e-10)
    assert_allclose(coefs, sparse_coefs)


@pytest.mark.parametrize("path", [enet_path, lasso_path])
def test_enet_path_positive(path):
    # [FUNCTION DOC]
    # Verifies path functions with positive=True return nonnegative coefficients for a
    # single target and raise ValueError for a multi-target y.
    # Test positive parameter

    X, Y, _, _ = build_dataset(n_samples=50, n_features=50, n_targets=2)

    # For mono output
    # Test that the coefs returned by positive=True in enet_path are positive
    pos_path_coef = path(X, Y[:, 0], positive=True)[1]
    assert np.all(pos_path_coef >= 0)

    # For multi output, positive parameter is not allowed
    # Test that an error is raised
    with pytest.raises(ValueError):
        path(X, Y, positive=True)


@pytest.mark.parametrize("path_func", [enet_path, lasso_path])
def test_path_unknown_parameter(path_func):
    # [FUNCTION DOC]
    # Verifies path functions raise ValueError "Unexpected parameters in params" when
    #   given
    # normalize and fit_intercept keywords.
    """Check that passing parameter not used by the coordinate descent solver
    will raise an error."""
    X, y, _, _ = build_dataset(n_samples=50, n_features=20)
    err_msg = "Unexpected parameters in params"
    with pytest.raises(ValueError, match=err_msg):
        path_func(X, y, normalize=True, fit_intercept=True)


def test_check_input_false():
    # [FUNCTION DOC]
    # Verifies fit with check_input=False works for F-ordered float64 and float32 X, and
    # raises ValueError for C-ordered X.
    # KNOWN ISSUE: y is built from X (y = check_array(X, ...)), so the fits are
    # multi-output with a 2-D y equal to X rather than single-target.
    X, y, _, _ = build_dataset(n_samples=20, n_features=10)
    X = check_array(X, order="F", dtype="float64")
    y = check_array(X, order="F", dtype="float64")
    clf = ElasticNet(selection="cyclic", tol=1e-6)
    # Check that no error is raised if data is provided in the right format
    clf.fit(X, y, check_input=False)
    # With check_input=False, an exhaustive check is not made on y but its
    # dtype is still cast in _preprocess_data to X's dtype. So the test should
    # pass anyway
    X = check_array(X, order="F", dtype="float32")
    clf.fit(X, y, check_input=False)
    # With no input checking, providing X in C order should result in false
    # computation
    X = check_array(X, order="C", dtype="float64")
    with pytest.raises(ValueError):
        clf.fit(X, y, check_input=False)


@pytest.mark.parametrize("check_input", [True, False])
def test_enet_copy_X_True(check_input):
    # [FUNCTION DOC]
    # Verifies ElasticNet(copy_X=True) leaves an F-ordered X unchanged with check_input
    #   True
    # or False.
    X, y, _, _ = build_dataset()
    X = X.copy(order="F")

    original_X = X.copy()
    enet = ElasticNet(copy_X=True)
    enet.fit(X, y, check_input=check_input)

    assert_array_equal(original_X, X)


def test_enet_copy_X_False_check_input_False():
    # [FUNCTION DOC]
    # Verifies ElasticNet(copy_X=False) with check_input=False modifies X in place
    # (centering), i.e. no copy is made.
    X, y, _, _ = build_dataset()
    X = X.copy(order="F")

    original_X = X.copy()
    enet = ElasticNet(copy_X=False)
    enet.fit(X, y, check_input=False)

    # No copying, X is overwritten
    assert np.any(np.not_equal(original_X, X))


def test_overridden_gram_matrix():
    # [FUNCTION DOC]
    # Verifies a Gram matrix from uncentered X triggers the UserWarning that X was
    #   centered
    # and the Gram matrix is recomputed (fit_intercept=True).
    X, y, _, _ = build_dataset(n_samples=20, n_features=10)
    Gram = X.T.dot(X)
    clf = ElasticNet(selection="cyclic", tol=1e-8, precompute=Gram)
    warning_message = (
        "Gram matrix was provided but X was centered"
        " to fit intercept: recomputing Gram matrix."
    )
    with pytest.warns(UserWarning, match=warning_message):
        clf.fit(X, y)


@pytest.mark.parametrize("model", [ElasticNet, Lasso])
def test_lasso_non_float_y(model):
    # [FUNCTION DOC]
    # Verifies integer y gives exactly the same coef_ as the float version, for
    #   ElasticNet
    # and Lasso without intercept.
    X = [[0, 0], [1, 1], [-1, -1]]
    y = [0, 1, 2]
    y_float = [0.0, 1.0, 2.0]

    clf = model(fit_intercept=False)
    clf.fit(X, y)
    clf_float = model(fit_intercept=False)
    clf_float.fit(X, y_float)
    assert_array_equal(clf.coef_, clf_float.coef_)


def test_enet_float_precision():
    # [FUNCTION DOC]
    # Verifies float32 and float64 fits agree to 4 decimals in coef_ and intercept_ for
    # ElasticNet and MultiTaskElasticNet (two identical target columns), with and
    #   without
    # intercept; that coef_ keeps the input dtype; and that a precomputed Gram matches
    #   the
    # plain fit (max_iter=100, warnings ignored).
    # KNOWN ISSUE: the dtype assertion after the multi-task fit re-checks clf
    #   (ElasticNet)
    # instead of clf_multioutput.
    # Generate dataset
    X, y, X_test, y_test = build_dataset(n_samples=20, n_features=10)
    # Here we have a small number of iterations, and thus the
    # ElasticNet might not converge. This is to speed up tests

    for fit_intercept in [True, False]:
        coef = {}
        intercept = {}
        for dtype in [np.float64, np.float32]:
            clf = ElasticNet(
                alpha=0.5,
                max_iter=100,
                precompute=False,
                fit_intercept=fit_intercept,
            )

            X = dtype(X)
            y = dtype(y)
            ignore_warnings(clf.fit)(X, y)

            coef[("simple", dtype)] = clf.coef_
            intercept[("simple", dtype)] = clf.intercept_

            assert clf.coef_.dtype == dtype

            # test precompute Gram array
            Gram = X.T.dot(X)
            clf_precompute = ElasticNet(
                alpha=0.5,
                max_iter=100,
                precompute=Gram,
                fit_intercept=fit_intercept,
            )
            ignore_warnings(clf_precompute.fit)(X, y)
            assert_array_almost_equal(clf.coef_, clf_precompute.coef_)
            assert_array_almost_equal(clf.intercept_, clf_precompute.intercept_)

            # test multi task enet
            multi_y = np.hstack((y[:, np.newaxis], y[:, np.newaxis]))
            clf_multioutput = MultiTaskElasticNet(
                alpha=0.5,
                max_iter=100,
                fit_intercept=fit_intercept,
            )
            clf_multioutput.fit(X, multi_y)
            coef[("multi", dtype)] = clf_multioutput.coef_
            intercept[("multi", dtype)] = clf_multioutput.intercept_
            assert clf.coef_.dtype == dtype

        for v in ["simple", "multi"]:
            assert_array_almost_equal(
                coef[(v, np.float32)], coef[(v, np.float64)], decimal=4
            )
            assert_array_almost_equal(
                intercept[(v, np.float32)], intercept[(v, np.float64)], decimal=4
            )


def test_enet_l1_ratio():
    # [FUNCTION DOC]
    # Verifies ElasticNetCV and MultiTaskElasticNetCV with l1_ratio=0 and an automatic
    #   alpha
    # grid raise ValueError; ElasticNetCV(l1_ratio=[0], alphas=[1]) fits without any
    # warning; and with a user alpha grid, l1_ratio=0 matches l1_ratio=1e-5 to 5
    #   decimals
    # for both estimators.
    # Test that an error message is raised if an estimator that
    # uses _alpha_grid is called with l1_ratio=0
    msg = (
        "Automatic alpha grid generation is not supported for l1_ratio=0. "
        "Please supply a grid by providing your estimator with the "
        "appropriate `alphas=` argument."
    )
    X = np.array([[1, 2, 4, 5, 8], [3, 5, 7, 7, 8]]).T
    y = np.array([12, 10, 11, 21, 5])

    with pytest.raises(ValueError, match=msg):
        ElasticNetCV(l1_ratio=0, random_state=42).fit(X, y)

    with pytest.raises(ValueError, match=msg):
        MultiTaskElasticNetCV(l1_ratio=0, random_state=42).fit(X, y[:, None])

    # But no error for ElasticNetCV with l1_ratio=0 and alpha>0.
    est = ElasticNetCV(l1_ratio=[0], alphas=[1])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        est.fit(X, y)

    # Test that l1_ratio=0 is allowed if we supply a grid manually
    alphas = [0.1, 10]
    estkwds = {"alphas": alphas, "random_state": 42}
    est_desired = ElasticNetCV(l1_ratio=0.00001, **estkwds)
    est = ElasticNetCV(l1_ratio=0, **estkwds)
    est_desired.fit(X, y)
    est.fit(X, y)
    assert_array_almost_equal(est.coef_, est_desired.coef_, decimal=5)

    est_desired = MultiTaskElasticNetCV(l1_ratio=0.00001, **estkwds)
    est = MultiTaskElasticNetCV(l1_ratio=0, **estkwds)
    est.fit(X, y[:, None])
    est_desired.fit(X, y[:, None])
    assert_array_almost_equal(est.coef_, est_desired.coef_, decimal=5)


def test_coef_shape_not_zero():
    # [FUNCTION DOC]
    # Verifies Lasso without intercept on a single constant feature yields coef_ of
    #   shape
    # (1,).
    est_no_intercept = Lasso(fit_intercept=False)
    est_no_intercept.fit(np.c_[np.ones(3)], np.ones(3))
    assert est_no_intercept.coef_.shape == (1,)


def test_warm_start_multitask_lasso():
    # [FUNCTION DOC]
    # Verifies warm_start for MultiTaskLasso: two fits of 5 iterations equal one of 10.
    X, y, X_test, y_test = build_dataset()
    Y = np.c_[y, y]
    clf = MultiTaskLasso(alpha=0.1, max_iter=5, warm_start=True)
    ignore_warnings(clf.fit)(X, Y)
    ignore_warnings(clf.fit)(X, Y)  # do a second round with 5 iterations

    clf2 = MultiTaskLasso(alpha=0.1, max_iter=10)
    ignore_warnings(clf2.fit)(X, Y)
    assert_array_almost_equal(clf2.coef_, clf.coef_)


@pytest.mark.parametrize(
    "est, kwargs",
    [
        (Lasso, dict(precompute=True)),
        (Lasso, dict(precompute=False)),
    ],
)
def test_enet_coordinate_descent_raises_convergence(est, kwargs):
    # [FUNCTION DOC]
    # Verifies Lasso (precompute True and False) emits the "Objective did not converge"
    # ConvergenceWarning when warm started from bad coefficients with max_iter=1,
    # tol=1e-10 and tiny alpha.
    """Test that a warning is issued if model does not converge"""
    reg = est(
        alpha=1e-10,
        fit_intercept=False,
        warm_start=True,
        max_iter=1,
        tol=1e-10,
        **kwargs,
    )
    # Set initial coefficients to very bad values.
    reg.coef_ = np.array([1, 1, 1, 1000])
    X = np.array([[-1, -1, 1, 1], [1, 1, -1, -1]])
    y = np.array([-1, 1])
    warning_message = (
        "Objective did not converge. You might want to"
        " increase the number of iterations."
    )
    with pytest.warns(ConvergenceWarning, match=warning_message):
        reg.fit(X, y)


def test_convergence_warnings():
    # [FUNCTION DOC]
    # Verifies MultiTaskElasticNet with defaults converges without ConvergenceWarning on
    # 1000x500 Gaussian X with 3 targets (warning turned into an error).
    random_state = np.random.RandomState(0)
    X = random_state.standard_normal((1000, 500))
    y = random_state.standard_normal((1000, 3))

    # check that the model converges w/o convergence warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        MultiTaskElasticNet().fit(X, y)


@pytest.mark.parametrize(
    "estimator", [ElasticNetCV, LassoCV, MultiTaskElasticNetCV, MultiTaskLassoCV]
)
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_sparse_input_convergence_warning(estimator, csr_container):
    # [FUNCTION DOC]
    # Verifies CV estimators on float32 CSR X emit ConvergenceWarning with max_iter=1,
    # tol=0, and converge without the warning with default settings.
    if issubclass(estimator, (MultiTaskElasticNetCV, MultiTaskLassoCV)):
        n_targets = 2
    else:
        n_targets = 1
    X, y, _, _ = build_dataset(n_targets=n_targets, n_samples=50, n_features=25)

    with pytest.warns(ConvergenceWarning):
        estimator(max_iter=1, tol=0).fit(csr_container(X, dtype=np.float32), y)

    # check that the model converges w/o convergence warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        estimator().fit(csr_container(X, dtype=np.float32), y)


@pytest.mark.parametrize(
    "precompute, inner_precompute",
    [
        (True, True),
        ("auto", False),
        (False, False),
    ],
)
def test_lassoCV_does_not_set_precompute(monkeypatch, precompute, inner_precompute):
    # [FUNCTION DOC]
    # Verifies LassoCV passes the expected precompute value to the final Lasso refit
    # (True stays True; "auto" and False become False). Monkeypatches the module-level
    # Lasso with LassoMock and asserts it was called at least once.
    X, y, _, _ = build_dataset()
    calls = 0

    class LassoMock(Lasso):
        # [CLASS DOC]
        # HELPER: Lasso subclass used to intercept LassoCV's final refit and inspect the
        # precompute value it received; counts calls in the enclosing calls variable.
        def fit(self, X, y):
            # [FUNCTION DOC]
            # HELPER: Runs Lasso.fit on X, y, increments the enclosing calls counter and
            #   asserts
            # self.precompute equals the expected inner_precompute. Returns None (not
            #   self).
            # Exceptions: AssertionError on mismatch.
            super().fit(X, y)
            nonlocal calls
            calls += 1
            assert self.precompute == inner_precompute

    monkeypatch.setattr("sklearn.linear_model._coordinate_descent.Lasso", LassoMock)
    clf = LassoCV(precompute=precompute)
    clf.fit(X, y)
    assert calls > 0


def test_multi_task_lasso_cv_dtype():
    # [FUNCTION DOC]
    # Verifies MultiTaskLassoCV accepts integer binary X and recovers coef ~[1, 0, 0]
    #   for
    # two targets both equal to the first column.
    n_samples, n_features = 10, 3
    rng = np.random.RandomState(42)
    X = rng.binomial(1, 0.5, size=(n_samples, n_features))
    X = X.astype(int)  # make it explicit that X is int
    y = X[:, [0, 0]].copy()
    est = MultiTaskLassoCV(alphas=5, fit_intercept=True, tol=1e-6).fit(X, y)
    assert_array_almost_equal(est.coef_, [[1, 0, 0]] * 2, decimal=3)


@pytest.mark.parametrize("estimator", [ElasticNet, MultiTaskElasticNet])
@pytest.mark.parametrize("fit_intercept", [True, False])
@pytest.mark.parametrize("alpha", [0.01])
@pytest.mark.parametrize("precompute", [False, True])
@pytest.mark.parametrize("sparse_container", [None] + CSR_CONTAINERS)
def test_enet_sample_weight_consistency(
    estimator, fit_intercept, alpha, precompute, sparse_container, global_random_seed
):
    # [FUNCTION DOC]
    # Verifies sample_weight semantics for ElasticNet and MultiTaskElasticNet (dense or
    #   CSR,
    # precompute only for single task; multi-task with precompute returns early):
    # (1) unit weights equal no weights; (2) a scalar weight equals no weights;
    # (3) rescaling weights has no effect; (4) zero weights equal removing those samples
    # (their targets are inflated 1000x to make it matter); (5) doubling the weight of
    # the first half equals duplicating those rows.
    # Side effects: mutates local y in step 4.
    """Test that the impact of sample_weight is consistent.

    Note that this test is stricter than the common test
    check_sample_weight_equivalence alone and also tests sparse X.
    """
    rng = np.random.RandomState(global_random_seed)
    n_samples, n_features = 10, 5
    X = rng.rand(n_samples, n_features)
    y = rng.rand(n_samples)

    params = dict(
        alpha=alpha,
        fit_intercept=fit_intercept,
        tol=1e-6,
        l1_ratio=0.5,
    )

    if issubclass(estimator, MultiTaskElasticNet):
        n_tasks = 3
        y = np.tile(y[:, None], reps=(1, n_tasks))
        if precompute:
            return
    else:
        n_tasks = 1
        params["precompute"] = precompute

    if sparse_container is not None:
        X = sparse_container(X)

    reg = estimator(**params).fit(X, y)
    coef = reg.coef_.copy()
    if fit_intercept:
        intercept = reg.intercept_
    assert np.sum(coef != 0) > 1

    # 1) sample_weight=np.ones(..) should be equivalent to sample_weight=None
    sample_weight = np.ones_like(y, shape=y.shape[0])
    reg.fit(X, y, sample_weight=sample_weight)
    assert_allclose(reg.coef_, coef, rtol=1e-6)
    if fit_intercept:
        assert_allclose(reg.intercept_, intercept)

    # 2) sample_weight=None should be equivalent to sample_weight = number
    sample_weight = 123.0
    reg.fit(X, y, sample_weight=sample_weight)
    assert_allclose(reg.coef_, coef, rtol=1e-6)
    if fit_intercept:
        assert_allclose(reg.intercept_, intercept)

    # 3) scaling of sample_weight should have no effect, cf. np.average()
    sample_weight = rng.uniform(low=0.01, high=2, size=X.shape[0])
    reg = reg.fit(X, y, sample_weight=sample_weight)
    coef = reg.coef_.copy()
    if fit_intercept:
        intercept = reg.intercept_

    reg.fit(X, y, sample_weight=np.pi * sample_weight)
    assert_allclose(reg.coef_, coef, rtol=1e-6)
    if fit_intercept:
        assert_allclose(reg.intercept_, intercept)

    # 4) setting elements of sample_weight to 0 is equivalent to removing these samples
    sample_weight_0 = sample_weight.copy()
    sample_weight_0[-5:] = 0
    y[-5:] *= 1000  # to make excluding those samples important
    reg.fit(X, y, sample_weight=sample_weight_0)
    coef_0 = reg.coef_.copy()
    if fit_intercept:
        intercept_0 = reg.intercept_
    reg.fit(X[:-5], y[:-5], sample_weight=sample_weight[:-5])
    assert_allclose(reg.coef_, coef_0, rtol=1e-6)
    if fit_intercept:
        assert_allclose(reg.intercept_, intercept_0)

    # 5) check that multiplying sample_weight by 2 is equivalent to repeating
    # corresponding samples twice
    if sparse_container is not None:
        X2 = sparse.vstack([X, X[: n_samples // 2]], format="csc")
    else:
        X2 = np.concatenate([X, X[: n_samples // 2]], axis=0)
    y2 = np.concatenate([y, y[: n_samples // 2]], axis=0)
    sample_weight_1 = sample_weight.copy()
    sample_weight_1[: n_samples // 2] *= 2
    sample_weight_2 = np.concatenate(
        [sample_weight, sample_weight[: n_samples // 2]], axis=0
    )

    reg1 = estimator(**params).fit(X, y, sample_weight=sample_weight_1)
    reg2 = estimator(**params).fit(X2, y2, sample_weight=sample_weight_2)
    assert_allclose(reg1.coef_, reg2.coef_, rtol=1e-6)


@pytest.mark.parametrize("estimator", [ElasticNetCV, MultiTaskElasticNetCV])
@pytest.mark.parametrize("fit_intercept", [True, False])
@pytest.mark.parametrize("sparse_container", [None] + CSC_CONTAINERS)
def test_enet_cv_sample_weight_correctness(
    estimator, fit_intercept, sparse_container, global_random_seed
):
    # [FUNCTION DOC]
    # Verifies ElasticNetCV / MultiTaskElasticNetCV with integer sample weights (0..4)
    #   on
    # the first of three CV groups match a fit on data where those rows are repeated
    #   (and
    # zero-weight rows removed), using LeaveOneGroupOut splits computed on each dataset:
    # mse_path_, alphas_, alpha_, coef_ (atol 1e-10) and intercept_ must agree. Dense or
    # CSC X.
    """Test that ElasticNetCV with sample weights gives correct results.

    We fit the same model twice, once with weighted training data, once with repeated
    data points in the training data and check that both models converge to the
    same solution.

    Since this model uses an internal cross-validation scheme to tune the alpha
    regularization parameter, we make sure that the repetitions only occur within
    a specific CV group. Data points belonging to other CV groups stay
    unit-weighted / "unrepeated".
    """
    rng = np.random.RandomState(global_random_seed)
    n_splits, n_samples_per_cv, n_features = 3, 10, 5
    X_with_weights = rng.rand(n_splits * n_samples_per_cv, n_features)
    beta = 10 * rng.rand(n_features)
    beta[0:2] = 0
    y_with_weights = X_with_weights @ beta + rng.rand(n_splits * n_samples_per_cv)
    if issubclass(estimator, MultiTaskElasticNetCV):
        n_tasks = 3
        y_with_weights = np.tile(y_with_weights[:, None], reps=(1, n_tasks))

    if sparse_container is not None:
        X_with_weights = sparse_container(X_with_weights)
    params = dict(tol=1e-6)

    # Assign random integer weights only to the first cross-validation group.
    # The samples in the other cross-validation groups are left with unit
    # weights.

    sw = np.ones(y_with_weights.shape[0])
    sw[:n_samples_per_cv] = rng.randint(0, 5, size=n_samples_per_cv)
    groups_with_weights = np.concatenate(
        [
            np.full(n_samples_per_cv, 0),
            np.full(n_samples_per_cv, 1),
            np.full(n_samples_per_cv, 2),
        ]
    )
    splits_with_weights = list(
        LeaveOneGroupOut().split(X_with_weights, groups=groups_with_weights)
    )
    reg_with_weights = estimator(
        cv=splits_with_weights, fit_intercept=fit_intercept, **params
    )

    reg_with_weights.fit(X_with_weights, y_with_weights, sample_weight=sw)
    assert np.sum(reg_with_weights.coef_ != 0) > 1

    if sparse_container is not None:
        X_with_weights = X_with_weights.toarray()
    X_with_repetitions = np.repeat(X_with_weights, sw.astype(int), axis=0)
    if sparse_container is not None:
        X_with_repetitions = sparse_container(X_with_repetitions)

    y_with_repetitions = np.repeat(y_with_weights, sw.astype(int), axis=0)
    groups_with_repetitions = np.repeat(groups_with_weights, sw.astype(int), axis=0)

    splits_with_repetitions = list(
        LeaveOneGroupOut().split(X_with_repetitions, groups=groups_with_repetitions)
    )
    reg_with_repetitions = estimator(
        cv=splits_with_repetitions, fit_intercept=fit_intercept, **params
    )
    reg_with_repetitions.fit(X_with_repetitions, y_with_repetitions)

    # Check that the alpha selection process is the same:
    assert_allclose(reg_with_weights.mse_path_, reg_with_repetitions.mse_path_)
    assert_allclose(reg_with_weights.alphas_, reg_with_repetitions.alphas_)
    assert reg_with_weights.alpha_ == pytest.approx(reg_with_repetitions.alpha_)

    # Check that the final model coefficients are the same:
    assert_allclose(reg_with_weights.coef_, reg_with_repetitions.coef_, atol=1e-10)
    assert reg_with_weights.intercept_ == pytest.approx(reg_with_repetitions.intercept_)


@pytest.mark.parametrize(
    ["estimatorCV", "estimator"],
    [(ElasticNetCV, ElasticNet), (MultiTaskElasticNetCV, MultiTaskElasticNet)],
)
@pytest.mark.parametrize("sample_weight", [False, True])
def test_enet_cv_grid_search(estimatorCV, estimator, sample_weight):
    # [FUNCTION DOC]
    # Verifies ElasticNetCV / MultiTaskElasticNetCV select the same alpha_ and l1_ratio_
    # as GridSearchCV over the same grid with neg MSE scoring and cv=5, with and without
    # linearly increasing sample weights.
    """Test that ElasticNetCV gives same result as GridSearchCV."""
    n_samples, n_features = 200, 10
    if issubclass(estimatorCV, MultiTaskElasticNetCV):
        n_targets = 3
    else:
        n_targets = 1

    cv = 5
    X, y = make_regression(
        n_targets=n_targets,
        n_samples=n_samples,
        n_features=n_features,
        effective_rank=10,
        n_informative=n_features - 4,
        noise=10,
        random_state=0,
    )
    if sample_weight:
        sample_weight = np.linspace(1, 5, num=n_samples)
    else:
        sample_weight = None

    alphas = np.logspace(np.log10(1e-5), np.log10(1), num=10)
    l1_ratios = [0.1, 0.5, 0.9]
    reg = estimatorCV(cv=cv, alphas=alphas, l1_ratio=l1_ratios)
    reg.fit(X, y, sample_weight=sample_weight)

    param = {"alpha": alphas, "l1_ratio": l1_ratios}
    gs = GridSearchCV(
        estimator=estimator(),
        param_grid=param,
        cv=cv,
        scoring="neg_mean_squared_error",
    ).fit(X, y, sample_weight=sample_weight)

    assert reg.l1_ratio_ == pytest.approx(gs.best_params_["l1_ratio"])
    assert reg.alpha_ == pytest.approx(gs.best_params_["alpha"])


@pytest.mark.parametrize(
    ["estimator", "l1_ratio"],
    [
        (LassoCV, 0),
        (ElasticNetCV, 0.5),
        (ElasticNetCV, 1),
        (MultiTaskLassoCV, 0),
        (MultiTaskElasticNetCV, 0.5),
        (MultiTaskElasticNetCV, 1),
    ],
)
@pytest.mark.parametrize("fit_intercept", [True, False])
@pytest.mark.parametrize("precompute", [False, True])
@pytest.mark.parametrize("sparse_container", [None] + CSC_CONTAINERS)
def test_enet_cv_sample_weight_consistency(
    estimator, l1_ratio, fit_intercept, precompute, sparse_container
):
    # [FUNCTION DOC]
    # Verifies for CV estimators (LassoCV/MultiTaskLassoCV use l1_ratio placeholder 0,
    # meaning not set) that unit weights, a scalar weight and a constant weight of 2 all
    # reproduce the unweighted coef_ and intercept_, for dense or CSC X, with and
    #   without
    # intercept and precompute (precompute dropped for multi-task).
    """Test that the impact of sample_weight is consistent."""
    rng = np.random.RandomState(0)
    n_samples, n_features = 10, 5

    X = rng.rand(n_samples, n_features)
    y = X.sum(axis=1) + rng.rand(n_samples)
    params = dict(
        fit_intercept=fit_intercept,
        precompute=precompute,
        tol=1e-6,
        cv=3,
    )
    if l1_ratio > 0:
        params["l1_ratio"] = l1_ratio
    if issubclass(estimator, (MultiTaskElasticNetCV, MultiTaskLassoCV)):
        n_tasks = 3
        y = np.tile(y[:, None], reps=(1, n_tasks))
        params.pop("precompute")
    if sparse_container is not None:
        X = sparse_container(X)

    reg = estimator(**params).fit(X, y)
    coef = reg.coef_.copy()
    if fit_intercept:
        intercept = reg.intercept_
    assert np.sum(coef != 0) > 1

    # sample_weight=np.ones(..) should be equivalent to sample_weight=None
    sample_weight = np.ones(n_samples)
    reg.fit(X, y, sample_weight=sample_weight)
    assert_allclose(reg.coef_, coef, rtol=1e-6)
    if fit_intercept:
        assert_allclose(reg.intercept_, intercept)

    # sample_weight=None should be equivalent to sample_weight = number
    sample_weight = 123.0
    reg.fit(X, y, sample_weight=sample_weight)
    assert_allclose(reg.coef_, coef, rtol=1e-6)
    if fit_intercept:
        assert_allclose(reg.intercept_, intercept)

    # scaling of sample_weight should have no effect, cf. np.average()
    sample_weight = 2 * np.ones(n_samples)
    reg.fit(X, y, sample_weight=sample_weight)
    assert_allclose(reg.coef_, coef, rtol=1e-6)
    if fit_intercept:
        assert_allclose(reg.intercept_, intercept)


@pytest.mark.parametrize(
    ["estimatorCV", "estimator"],
    [
        (ElasticNetCV, ElasticNet),
        (MultiTaskElasticNetCV, MultiTaskElasticNet),
    ],
)
@pytest.mark.parametrize("X_is_sparse", [False, sparse.csc_array, sparse.csc_matrix])
@pytest.mark.parametrize("fit_intercept", [False, True])
@pytest.mark.parametrize("positive", [False, True])
@pytest.mark.parametrize("sample_weight", [np.array([1, 10, 1, 10]), None])
def test_enet_alpha_max(
    estimatorCV, estimator, X_is_sparse, fit_intercept, positive, sample_weight
):
    # [FUNCTION DOC]
    # Verifies the computed alpha_max: a CV estimator with a single alpha (alphas=1,
    #   eps=1)
    # gets all-zero coef_, while the plain estimator at 0.99 * alpha_max has some
    # coefficient above 1e-3. When positive=True, also checks alpha_max differs from the
    # unconstrained one so the data distinguishes them. Covers sparse X, intercept and
    # sample weights; multi-task skips positive=True.
    X = np.array([[3.0, -1.0], [2.0, -5.0], [5.0, -3.0], [1.0, -4.0]])
    beta = np.array([1, -2])
    y = X @ beta
    params = dict(fit_intercept=fit_intercept, positive=positive)
    if issubclass(estimator, MultiTaskElasticNet):
        n_tasks = 3
        y = np.tile(y[:, None], reps=(1, n_tasks))
        params.pop("positive")
        if positive:
            return

    if X_is_sparse:
        X = X_is_sparse(X)
    # Test alpha_max makes coefs zero.
    reg = estimatorCV(alphas=1, cv=2, eps=1, **params)
    reg.fit(X, y, sample_weight=sample_weight)
    assert_allclose(reg.coef_, 0, atol=1e-5)
    alpha_max = reg.alpha_
    # Test smaller alpha makes coefs nonzero.
    reg = estimator(alpha=0.99 * alpha_max, tol=1e-8, **params)
    reg.fit(X, y, sample_weight=sample_weight)
    assert_array_less(1e-3, np.max(np.abs(reg.coef_)))

    if positive:
        # Make sure that the positive constraint changes alpha_max,
        # i.e. test the meaningfulness of the test data.
        not_positive_alpha_max = (
            estimatorCV(alphas=1, cv=2, eps=1, **{**params, "positive": not positive})
            .fit(X, y, sample_weight=sample_weight)
            .alpha_
        )
        assert not np.isclose(alpha_max, not_positive_alpha_max), (
            "Test data cannot distinguish alpha_max between positive=True and False."
        )


@pytest.mark.parametrize("estimator", [ElasticNetCV, LassoCV])
def test_linear_models_cv_fit_with_loky(estimator):
    # [FUNCTION DOC]
    # Verifies ElasticNetCV and LassoCV fit with n_jobs=2 under the loky backend on data
    # larger than 1 MB, which triggers read-only memmapping and fancy indexing
    # (numpy#14132). Smoke test without numeric assertions.
    # LinearModelsCV.fit performs operations on fancy-indexed memmapped
    # data when using the loky backend, causing an error due to unexpected
    # behavior of fancy indexing of read-only memmaps (cf. numpy#14132).

    # Create a problem sufficiently large to cause memmapping (1MB).
    # Unfortunately the scikit-learn and joblib APIs do not make it possible to
    # change the max_nbyte of the inner Parallel call.
    X, y = make_regression(int(1e6) // 8 + 1, 1)
    assert X.nbytes > 1e6  # 1 MB
    with joblib.parallel_backend("loky"):
        estimator(n_jobs=2, cv=3).fit(X, y)


# TODO:
@pytest.mark.parametrize("check_input", [True, False])
def test_enet_sample_weight_does_not_overwrite_sample_weight(check_input):
    # [FUNCTION DOC]
    # Verifies ElasticNet.fit does not modify the caller's sample_weight array, with
    # check_input True or False.
    # KNOWN ISSUE: a bare "TODO:" comment precedes this test with no content.
    """Check that ElasticNet does not overwrite sample_weights."""

    rng = np.random.RandomState(0)
    n_samples, n_features = 10, 5

    X = rng.rand(n_samples, n_features)
    y = rng.rand(n_samples)

    sample_weight_1_25 = 1.25 * np.ones_like(y)
    sample_weight = sample_weight_1_25.copy()

    reg = ElasticNet()
    reg.fit(X, y, sample_weight=sample_weight, check_input=check_input)

    assert_array_equal(sample_weight, sample_weight_1_25)


@pytest.mark.parametrize("ridge_alpha", [1e-6, 1e-1, 1.0, 1e6])
@pytest.mark.parametrize(
    ["precompute", "n_targets"], [(False, 1), (True, 1), (False, 3)]
)
def test_enet_ridge_consistency(ridge_alpha, precompute, n_targets, global_random_seed):
    # [FUNCTION DOC]
    # Verifies ElasticNet(l1_ratio=0) (or MultiTaskElasticNet for 3 targets, without
    # weights) matches Ridge(solver="svd") when alpha is rescaled: ridge_alpha divided
    #   by
    # the sum of sample weights (single target) or by n_samples (multi-task). Very tight
    # tol; tolerance looser for the Gram-based solver.
    # Check that ElasticNet(l1_ratio=0) converges to the same solution as Ridge
    # provided that the value of alpha is adapted.

    rng = np.random.RandomState(global_random_seed)
    n_samples = 300
    X, y = make_regression(
        n_samples=n_samples,
        n_features=100,
        n_targets=n_targets,
        effective_rank=10,
        n_informative=50,
        random_state=rng,
    )
    sw = rng.uniform(low=0.01, high=10, size=X.shape[0])

    if n_targets == 1:
        sw_arg = dict(sample_weight=sw)
    else:
        # MultiTaskElasticNet does not support sample weights (yet).
        sw_arg = dict()

    ridge = Ridge(alpha=ridge_alpha, solver="svd").fit(X, y, **sw_arg)

    tol = 1e-11 if ridge_alpha >= 1e-2 else 1e-16
    if n_targets == 1:
        alpha_enet = ridge_alpha / sw.sum()
        enet = ElasticNet(alpha=alpha_enet, l1_ratio=0, precompute=precompute, tol=tol)
    else:
        alpha_enet = ridge_alpha / n_samples
        enet = MultiTaskElasticNet(alpha=alpha_enet, l1_ratio=0, tol=tol)
    enet.fit(X, y, **sw_arg)

    # The CD solver using the gram matrix (precompute = True) loses numerical precision
    # by working with the squares of matrices like Q=X'X (=gram) and
    # R^2 = y^2 + wQw - 2yQw (=square of residuals).
    rtol = 1e-5 if precompute else 5e-7
    atol = 3e-11
    assert_allclose(enet.coef_, ridge.coef_, rtol=rtol, atol=atol)
    assert_allclose(enet.intercept_, ridge.intercept_, atol=atol)


@pytest.mark.filterwarnings("ignore:With alpha=0, this algorithm:UserWarning")
@pytest.mark.parametrize("precompute", [False, True])
@pytest.mark.parametrize("effective_rank", [None, 10])
def test_enet_ols_consistency(precompute, effective_rank, global_random_seed):
    # [FUNCTION DOC]
    # Verifies ElasticNet(alpha=0) with tol 1e-15 matches weighted LinearRegression in
    # predictions, near-zero weighted squared error for both, and coefficients and
    # intercept to atol 1e-11, for full and low effective rank designs.
    """Test that ElasticNet(alpha=0) converges to the same solution as OLS."""
    rng = np.random.RandomState(global_random_seed)
    n_samples = 300
    X, y = make_regression(
        n_samples=n_samples,
        n_features=100,
        effective_rank=effective_rank,
        n_informative=50,
        random_state=rng,
    )
    sw = rng.uniform(low=0.01, high=10, size=X.shape[0])

    ols = LinearRegression().fit(X, y, sample_weight=sw)
    enet = ElasticNet(alpha=0, precompute=precompute, tol=1e-15).fit(
        X, y, sample_weight=sw
    )

    # Might be a singular problem, so check for same predictions
    assert_allclose(enet.predict(X), ols.predict(X))
    # and for similar objective function (squared error)
    se_ols = np.sum(sw * (y - ols.predict(X)) ** 2)
    se_enet = np.sum(sw * (y - enet.predict(X)) ** 2)
    assert se_ols <= 1e-19
    assert se_enet <= 1e-19
    # We check equal coefficients, but "only" with absolute tolerance.
    assert_allclose(enet.coef_, ols.coef_, atol=1e-11)
    assert_allclose(enet.intercept_, ols.intercept_, atol=1e-11)


@pytest.mark.parametrize(
    "estimator",
    [
        Lasso(alpha=1.0),
        ElasticNet(alpha=1.0, l1_ratio=0.1),
    ],
)
def test_sample_weight_invariance(estimator):
    # [FUNCTION DOC]
    # Verifies for Lasso and ElasticNet (tol 1e-12) that zeroing the weights of the
    #   first
    # third of samples equals trimming them, and that doubling all weights equals
    # duplicating the dataset (coef_ and intercept_ compared).
    rng = np.random.RandomState(42)
    X, y = make_regression(
        n_samples=100,
        n_features=300,
        effective_rank=10,
        n_informative=50,
        random_state=rng,
    )
    sw = rng.uniform(low=0.01, high=2, size=X.shape[0])
    params = dict(tol=1e-12)

    # Check that setting some weights to 0 is equivalent to trimming the
    # samples:
    cutoff = X.shape[0] // 3
    sw_with_null = sw.copy()
    sw_with_null[:cutoff] = 0.0
    X_trimmed, y_trimmed = X[cutoff:, :], y[cutoff:]
    sw_trimmed = sw[cutoff:]

    reg_trimmed = (
        clone(estimator)
        .set_params(**params)
        .fit(X_trimmed, y_trimmed, sample_weight=sw_trimmed)
    )
    reg_null_weighted = (
        clone(estimator).set_params(**params).fit(X, y, sample_weight=sw_with_null)
    )
    assert_allclose(reg_null_weighted.coef_, reg_trimmed.coef_)
    assert_allclose(reg_null_weighted.intercept_, reg_trimmed.intercept_)

    # Check that duplicating the training dataset is equivalent to multiplying
    # the weights by 2:
    X_dup = np.concatenate([X, X], axis=0)
    y_dup = np.concatenate([y, y], axis=0)
    sw_dup = np.concatenate([sw, sw], axis=0)

    reg_2sw = clone(estimator).set_params(**params).fit(X, y, sample_weight=2 * sw)
    reg_dup = (
        clone(estimator).set_params(**params).fit(X_dup, y_dup, sample_weight=sw_dup)
    )

    assert_allclose(reg_2sw.coef_, reg_dup.coef_)
    assert_allclose(reg_2sw.intercept_, reg_dup.intercept_)


@pytest.mark.parametrize(
    "EstimatorCV",
    [ElasticNetCV, LassoCV, MultiTaskElasticNetCV, MultiTaskLassoCV],
)
def test_cv_estimators_reject_params_with_no_routing_enabled(EstimatorCV):
    # [FUNCTION DOC]
    # Verifies LinearModelCV subclasses raise ValueError when extra fit params (groups)
    #   are
    # passed without metadata routing enabled.
    """Check that the models inheriting from class:`LinearModelCV` raise an
    error when any `params` are passed when routing is not enabled.
    """
    X, y = make_regression(random_state=42)
    groups = np.array([0, 1] * (len(y) // 2))
    estimator = EstimatorCV()
    msg = "is only supported if enable_metadata_routing=True"
    with pytest.raises(ValueError, match=msg):
        estimator.fit(X, y, groups=groups)


@pytest.mark.parametrize("precompute", ["auto", True, False])
def test_enet_path_check_input_false(precompute):
    # [FUNCTION DOC]
    # Smoke test: enet_path with check_input=False on F-ordered X runs for precompute
    # "auto", True and False. No assertions; outputs are discarded.
    """Test enet_path works with check_input=False and various precompute settings."""
    X, y = make_regression(n_samples=100, n_features=5, n_informative=2, random_state=0)
    X = np.asfortranarray(X)
    alphas, _, _ = enet_path(X, y, alphas=3, check_input=False, precompute=precompute)


# TODO(1.11): remove
@pytest.mark.parametrize("path_func", [lasso_path, enet_path])
def test_path_function_deprecated_n_alphas(path_func):
    # [FUNCTION DOC]
    # Verifies lasso_path / enet_path emit FutureWarning for n_alphas and for
    #   alphas=None,
    # and no warning at all for an integer alphas.
    """Check deprecation of n_alphas in favor of alphas."""
    X, y = make_regression(n_samples=9, n_features=5, n_informative=2, random_state=42)

    msg = "'n_alphas' was deprecated in 1.9 and will be removed in 1.11"
    with pytest.warns(FutureWarning, match=msg):
        path_func(X, y, n_alphas=5)

    msg = "'alphas=None' is deprecated and will be removed in 1.11"
    with pytest.warns(FutureWarning, match=msg):
        path_func(X, y, alphas=None)

    # Assert that no warning is raised when n_alphas is not used.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        path_func(X, y, alphas=5)
