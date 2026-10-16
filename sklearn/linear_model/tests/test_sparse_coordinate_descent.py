# [MODULE DOC]
# Tests for coordinate descent linear models (Lasso, ElasticNet and their CV
# variants) fitted on sparse input, mostly checking agreement with dense input,
# known toy solutions, convergence warnings and read-only buffers.
import numpy as np
import pytest
import scipy.sparse as sp
from numpy.testing import assert_allclose

from sklearn.datasets import make_regression
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet, ElasticNetCV, Lasso, LassoCV
from sklearn.utils._testing import (
    assert_almost_equal,
    assert_array_almost_equal,
    create_memmap_backed_data,
    ignore_warnings,
)
from sklearn.utils.fixes import (
    COO_CONTAINERS,
    CSC_CONTAINERS,
    LIL_CONTAINERS,
    _sparse_random_array,
)


def test_sparse_coef():
    # [FUNCTION DOC]
    # Sets coef_ manually on an unfitted ElasticNet and checks sparse_coef_ is a sparse
    # matrix whose single dense row equals coef_.
    # Check that the sparse_coef property works
    clf = ElasticNet()
    clf.coef_ = [1, 2, 3]

    assert sp.issparse(clf.sparse_coef_)
    assert clf.sparse_coef_.toarray().tolist()[0] == clf.coef_


@pytest.mark.parametrize("csc_container", CSC_CONTAINERS)
def test_lasso_zero(csc_container):
    # [FUNCTION DOC]
    # Fits Lasso on an all-zero sparse CSC X with zero targets; coef_, predictions and
    # dual_gap_ must all be 0 (no crash on empty data).
    # Check that the sparse lasso can handle zero data without crashing
    X = csc_container((3, 1))
    y = [0, 0, 0]
    T = np.array([[1], [2], [3]])
    clf = Lasso().fit(X, y)
    pred = clf.predict(T)
    assert_array_almost_equal(clf.coef_, [0])
    assert_array_almost_equal(pred, [0, 0, 0])
    assert_almost_equal(clf.dual_gap_, 0)


@pytest.mark.parametrize("with_sample_weight", [True, False])
@pytest.mark.parametrize("csc_container", CSC_CONTAINERS)
def test_enet_toy_list_input(with_sample_weight, csc_container):
    # [FUNCTION DOC]
    # On a sparse 3-point line y = x, with and without uniform sample weights: alpha=0
    # reproduces least squares (warnings ignored), and two (alpha, l1_ratio) settings
    # match precomputed coefficients and predictions, with zero dual gap.
    # KNOWN ISSUE: the name says "list input" but X is converted to a CSC container.
    # Test ElasticNet for various values of alpha and l1_ratio with list X

    X = np.array([[-1], [0], [1]])
    X = csc_container(X)
    Y = [-1, 0, 1]  # just a straight line
    T = np.array([[2], [3], [4]])  # test sample
    if with_sample_weight:
        sw = np.array([2.0, 2, 2])
    else:
        sw = None

    # this should be the same as unregularized least squares
    clf = ElasticNet(alpha=0, l1_ratio=1.0)
    # catch warning about alpha=0.
    # this is discouraged but should work.
    ignore_warnings(clf.fit)(X, Y, sample_weight=sw)
    pred = clf.predict(T)
    assert_array_almost_equal(clf.coef_, [1])
    assert_array_almost_equal(pred, [2, 3, 4])
    assert_almost_equal(clf.dual_gap_, 0)

    clf = ElasticNet(alpha=0.5, l1_ratio=0.3)
    clf.fit(X, Y, sample_weight=sw)
    pred = clf.predict(T)
    assert_array_almost_equal(clf.coef_, [0.50819], decimal=3)
    assert_array_almost_equal(pred, [1.0163, 1.5245, 2.0327], decimal=3)
    assert_almost_equal(clf.dual_gap_, 0)

    clf = ElasticNet(alpha=0.5, l1_ratio=0.5)
    clf.fit(X, Y, sample_weight=sw)
    pred = clf.predict(T)
    assert_array_almost_equal(clf.coef_, [0.45454], 3)
    assert_array_almost_equal(pred, [0.9090, 1.3636, 1.8181], 3)
    assert_almost_equal(clf.dual_gap_, 0)


@pytest.mark.parametrize("lil_container", LIL_CONTAINERS)
def test_enet_toy_explicit_sparse_input(lil_container):
    # [FUNCTION DOC]
    # Same expected values as the previous test, building X and test samples
    # element-wise in LIL containers (implicit zero for the middle sample).
    # Test ElasticNet for various values of alpha and l1_ratio with sparse X
    # training samples
    X = lil_container((3, 1))
    X[0, 0] = -1
    # X[1, 0] = 0
    X[2, 0] = 1
    Y = [-1, 0, 1]  # just a straight line (the identity function)

    # test samples
    T = lil_container((3, 1))
    T[0, 0] = 2
    T[1, 0] = 3
    T[2, 0] = 4

    # this should be the same as lasso
    clf = ElasticNet(alpha=0, l1_ratio=1.0)
    ignore_warnings(clf.fit)(X, Y)
    pred = clf.predict(T)
    assert_array_almost_equal(clf.coef_, [1])
    assert_array_almost_equal(pred, [2, 3, 4])
    assert_almost_equal(clf.dual_gap_, 0)

    clf = ElasticNet(alpha=0.5, l1_ratio=0.3)
    clf.fit(X, Y)
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


def make_sparse_data(
    sparse_container,
    n_samples=100,
    n_features=100,
    n_informative=10,
    seed=42,
    positive=False,
    n_targets=1,
):
    # [FUNCTION DOC]
    # HELPER: Builds a seeded ill-posed regression problem: ground-truth weights
    # (n_features, n_targets) with only the first n_informative nonzero (absolute
    # values if positive), Gaussian X with about 50% entries zeroed, noiseless
    # y = X w, X wrapped in sparse_container. y is raveled when n_targets is 1.
    # Outputs: (X sparse (n_samples, n_features), y (n_samples,) or (n_samples,
    # n_targets)).
    random_state = np.random.RandomState(seed)

    # build an ill-posed linear regression problem with many noisy features and
    # comparatively few samples

    # generate a ground truth model
    w = random_state.randn(n_features, n_targets)
    w[n_informative:] = 0.0  # only the top features are impacting the model
    if positive:
        w = np.abs(w)

    X = random_state.randn(n_samples, n_features)
    rnd = random_state.uniform(size=(n_samples, n_features))
    X[rnd > 0.5] = 0.0  # 50% of zeros in input signal

    # generate training ground truth labels
    y = np.dot(X, w)
    X = sparse_container(X)
    if n_targets == 1:
        y = np.ravel(y)
    return X, y


@pytest.mark.parametrize("csc_container", CSC_CONTAINERS)
@pytest.mark.parametrize(
    "alpha, fit_intercept, positive",
    [(0.1, False, False), (0.1, True, False), (1e-3, False, True), (1e-3, True, True)],
)
def test_sparse_enet_not_as_toy_dataset(csc_container, alpha, fit_intercept, positive):
    # [FUNCTION DOC]
    # For several alpha / fit_intercept / positive settings, fits ElasticNet on half
    # of sparse data: dual gap near 0, test R2 above 0.85, coef_/intercept_ equal to
    # the dense fit to 5 decimals, and fewer than 2 * n_informative nonzero coefs.
    n_samples, n_features, max_iter = 100, 100, 1000
    n_informative = 10

    X, y = make_sparse_data(
        csc_container, n_samples, n_features, n_informative, positive=positive
    )

    X_train, X_test = X[n_samples // 2 :], X[: n_samples // 2]
    y_train, y_test = y[n_samples // 2 :], y[: n_samples // 2]

    s_clf = ElasticNet(
        alpha=alpha,
        l1_ratio=0.8,
        fit_intercept=fit_intercept,
        max_iter=max_iter,
        tol=1e-7,
        positive=positive,
        warm_start=True,
    )
    s_clf.fit(X_train, y_train)

    assert_almost_equal(s_clf.dual_gap_, 0, 4)
    assert s_clf.score(X_test, y_test) > 0.85

    # check the convergence is the same as the dense version
    d_clf = ElasticNet(
        alpha=alpha,
        l1_ratio=0.8,
        fit_intercept=fit_intercept,
        max_iter=max_iter,
        tol=1e-7,
        positive=positive,
        warm_start=True,
    )
    d_clf.fit(X_train.toarray(), y_train)

    assert_almost_equal(d_clf.dual_gap_, 0, 4)
    assert d_clf.score(X_test, y_test) > 0.85

    assert_almost_equal(s_clf.coef_, d_clf.coef_, 5)
    assert_almost_equal(s_clf.intercept_, d_clf.intercept_, 5)

    # check that the coefs are sparse
    assert np.sum(s_clf.coef_ != 0.0) < 2 * n_informative


@pytest.mark.parametrize("csc_container", CSC_CONTAINERS)
def test_sparse_lasso_not_as_toy_dataset(csc_container):
    # [FUNCTION DOC]
    # Sparse Lasso without intercept converges (dual gap near 0), scores above 0.85,
    # the dense fit does too, and exactly n_informative coefficients are nonzero.
    n_samples = 100
    max_iter = 1000
    n_informative = 10
    X, y = make_sparse_data(
        csc_container, n_samples=n_samples, n_informative=n_informative
    )

    X_train, X_test = X[n_samples // 2 :], X[: n_samples // 2]
    y_train, y_test = y[n_samples // 2 :], y[: n_samples // 2]

    s_clf = Lasso(alpha=0.1, fit_intercept=False, max_iter=max_iter, tol=1e-7)
    s_clf.fit(X_train, y_train)
    assert_almost_equal(s_clf.dual_gap_, 0, 4)
    assert s_clf.score(X_test, y_test) > 0.85

    # check the convergence is the same as the dense version
    d_clf = Lasso(alpha=0.1, fit_intercept=False, max_iter=max_iter, tol=1e-7)
    d_clf.fit(X_train.toarray(), y_train)
    assert_almost_equal(d_clf.dual_gap_, 0, 4)
    assert d_clf.score(X_test, y_test) > 0.85

    # check that the coefs are sparse
    assert np.sum(s_clf.coef_ != 0.0) == n_informative


@pytest.mark.parametrize("csc_container", CSC_CONTAINERS)
def test_enet_multitarget(csc_container):
    # [FUNCTION DOC]
    # Fits ElasticNet on 3 targets jointly and checks each row of coef_, intercept_ and
    # dual_gap_ matches a separate single-target fit.
    n_targets = 3
    X, y = make_sparse_data(csc_container, n_targets=n_targets)

    estimator = ElasticNet(alpha=0.01, precompute=False)
    # XXX: There is a bug when precompute is not False!
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


@pytest.mark.parametrize("csc_container", CSC_CONTAINERS)
def test_path_parameters(csc_container):
    # [FUNCTION DOC]
    # ElasticNetCV with an integer alphas=10 keeps l1_ratio and alphas, produces 10
    # alphas_, and mse_path_ is identical for sparse and dense input.
    X, y = make_sparse_data(csc_container)
    max_iter = 50
    n_alphas = 10
    clf = ElasticNetCV(
        alphas=n_alphas,
        eps=1e-3,
        max_iter=max_iter,
        l1_ratio=0.5,
        fit_intercept=False,
    )
    clf.fit(X, y)
    assert_almost_equal(0.5, clf.l1_ratio)
    assert clf.alphas == n_alphas
    assert len(clf.alphas_) == n_alphas
    sparse_mse_path = clf.mse_path_
    # compare with dense data
    clf.fit(X.toarray(), y)
    assert_almost_equal(clf.mse_path_, sparse_mse_path)


@pytest.mark.parametrize("Model", [Lasso, ElasticNet, LassoCV, ElasticNetCV])
@pytest.mark.parametrize("fit_intercept", [False, True])
@pytest.mark.parametrize("l1_ratio", [0.5, 0])
@pytest.mark.parametrize("n_samples, n_features", [(24, 6), (6, 24)])
@pytest.mark.parametrize("with_sample_weight", [True, False])
@pytest.mark.parametrize("csc_container", CSC_CONTAINERS)
def test_sparse_dense_equality(
    Model,
    fit_intercept,
    l1_ratio,
    n_samples,
    n_features,
    with_sample_weight,
    csc_container,
):
    # [FUNCTION DOC]
    # Parametrized over models, intercept, l1_ratio, shapes and sample weights: sparse
    # and dense fits agree on coef_ and intercept_, and with intercept the weighted
    # mean prediction equals the weighted mean of y (balance property).
    # KNOWN ISSUE: combinations with l1_ratio=0 for models other than ElasticNet
    # return early and are reported as passed instead of skipped.
    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        effective_rank=n_features // 2,
        n_informative=n_features // 2,
        bias=4 * fit_intercept,
        noise=1,
        random_state=42,
    )
    if with_sample_weight:
        sw = np.abs(np.random.RandomState(42).normal(scale=10, size=y.shape))
    else:
        sw = None
    Xs = csc_container(X)
    params = {"fit_intercept": fit_intercept, "tol": 1e-6}
    if Model != ElasticNet:
        if l1_ratio == 0:
            return
    else:
        params["l1_ratio"] = l1_ratio
    reg_dense = Model(**params).fit(X, y, sample_weight=sw)
    reg_sparse = Model(**params).fit(Xs, y, sample_weight=sw)
    if fit_intercept:
        assert reg_sparse.intercept_ == pytest.approx(reg_dense.intercept_)
        # balance property
        assert np.average(reg_sparse.predict(X), weights=sw) == pytest.approx(
            np.average(y, weights=sw)
        )
    assert_allclose(reg_sparse.coef_, reg_dense.coef_)


@pytest.mark.parametrize("csc_container", CSC_CONTAINERS)
def test_same_output_sparse_dense_lasso_and_enet_cv(csc_container):
    # [FUNCTION DOC]
    # ElasticNetCV and LassoCV give the same alpha_, intercept_, mse_path_ and alphas_
    # on sparse and dense versions of the same data.
    X, y = make_sparse_data(csc_container, n_samples=40, n_features=10)
    clfs = ElasticNetCV(max_iter=100, tol=1e-7)
    clfs.fit(X, y)
    clfd = ElasticNetCV(max_iter=100, tol=1e-7)
    clfd.fit(X.toarray(), y)
    assert_allclose(clfs.alpha_, clfd.alpha_)
    assert_allclose(clfs.intercept_, clfd.intercept_)
    assert_allclose(clfs.mse_path_, clfd.mse_path_)
    assert_allclose(clfs.alphas_, clfd.alphas_)

    clfs = LassoCV(max_iter=100, cv=4, tol=1e-8)
    clfs.fit(X, y)
    clfd = LassoCV(max_iter=100, cv=4, tol=1e-8)
    clfd.fit(X.toarray(), y)
    assert_allclose(clfs.alpha_, clfd.alpha_)
    assert_allclose(clfs.intercept_, clfd.intercept_)
    assert_allclose(clfs.mse_path_, clfd.mse_path_)
    assert_allclose(clfs.alphas_, clfd.alphas_)


@pytest.mark.parametrize("coo_container", COO_CONTAINERS)
def test_same_multiple_output_sparse_dense(coo_container):
    # [FUNCTION DOC]
    # Multi-output ElasticNet trained on dense lists and on COO input gives the same
    # prediction for one sample (dense vs sparse sample).
    l = ElasticNet()
    X = [
        [0, 1, 2, 3, 4],
        [0, 2, 5, 8, 11],
        [9, 10, 11, 12, 13],
        [10, 11, 12, 13, 14],
    ]
    y = [
        [1, 2, 3, 4, 5],
        [1, 3, 6, 9, 12],
        [10, 11, 12, 13, 14],
        [11, 12, 13, 14, 15],
    ]
    l.fit(X, y)
    sample = np.array([1, 2, 3, 4, 5]).reshape(1, -1)
    predict_dense = l.predict(sample)

    l_sp = ElasticNet()
    X_sp = coo_container(X)
    l_sp.fit(X_sp, y)
    sample_sparse = coo_container(sample)
    predict_sparse = l_sp.predict(sample_sparse)

    assert_array_almost_equal(predict_sparse, predict_dense)


@pytest.mark.parametrize("csc_container", CSC_CONTAINERS)
def test_sparse_enet_coordinate_descent(csc_container):
    # [FUNCTION DOC]
    # With warm_start, very bad initial coef_ and max_iter=2, fitting sparse Lasso
    # must emit ConvergenceWarning with the expected message.
    """Test that a warning is issued if model does not converge"""
    clf = Lasso(
        alpha=1e-10, fit_intercept=False, warm_start=True, max_iter=2, tol=1e-10
    )
    # Set initial coefficients to very bad values.
    clf.coef_ = np.array([1, 1, 1, 1000])
    X = np.array([[-1, -1, 1, 1], [1, 1, -1, -1]])
    X = csc_container(X)
    y = np.array([-1, 1])
    warning_message = (
        "Objective did not converge. You might want "
        "to increase the number of iterations."
    )
    with pytest.warns(ConvergenceWarning, match=warning_message):
        clf.fit(X, y)


@pytest.mark.parametrize("copy_X", (True, False))
def test_sparse_read_only_buffer(copy_X):
    # [FUNCTION DOC]
    # Fits ElasticNet on a random CSC matrix whose data array is a read-only memmap,
    # for copy_X True and False; passes if no error is raised.
    """Test that sparse coordinate descent works for read-only buffers"""
    rng = np.random.RandomState(0)

    clf = ElasticNet(alpha=0.1, copy_X=copy_X, random_state=rng)
    X = _sparse_random_array((100, 20), format="csc", rng=rng)

    # Make X.data read-only
    X.data = create_memmap_backed_data(X.data)

    y = rng.rand(100)
    clf.fit(X, y)
