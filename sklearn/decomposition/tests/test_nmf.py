# [MODULE DOC]
# Tests for NMF, MiniBatchNMF, non_negative_factorization and _nmf internals
# (_initialize_nmf, _beta_divergence, _special_sparse_dot): non-negativity,
# reconstruction quality, transform consistency, sparse input, beta losses,
# regularization, monotone objective decrease, dtype handling, n_components
# inference, custom init validation and warnings.
# Many tests are parametrized over (NMF cd, NMF mu, MiniBatchNMF).
import re
import sys
from io import StringIO

import numpy as np
import pytest
from scipy import linalg

from sklearn.base import clone
from sklearn.decomposition import NMF, MiniBatchNMF, non_negative_factorization
from sklearn.decomposition import _nmf as nmf  # For testing internals
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils._testing import (
    assert_allclose,
    assert_almost_equal,
    assert_array_almost_equal,
    assert_array_equal,
)
from sklearn.utils.extmath import squared_norm
from sklearn.utils.fixes import CSC_CONTAINERS, CSR_CONTAINERS


@pytest.mark.parametrize(
    ["Estimator", "solver"],
    [[NMF, {"solver": "cd"}], [NMF, {"solver": "mu"}], [MiniBatchNMF, {}]],
)
def test_convergence_warning(Estimator, solver):
    # [FUNCTION DOC]
    # Checks fitting with max_iter=1 and n_components="auto" on a 2x2 ones matrix
    # emits ConvergenceWarning with the max-iterations message, for each estimator.
    convergence_warning = (
        "Maximum number of iterations 1 reached. Increase it to improve convergence."
    )
    A = np.ones((2, 2))
    with pytest.warns(ConvergenceWarning, match=convergence_warning):
        Estimator(max_iter=1, n_components="auto", **solver).fit(A)


def test_initialize_nn_output():
    # [FUNCTION DOC]
    # Checks _initialize_nmf produces no negative entries in W or H for init random,
    # nndsvd, nndsvda and nndsvdar on positive 10x10 data.
    # Test that initialization does not return negative values
    rng = np.random.mtrand.RandomState(42)
    data = np.abs(rng.randn(10, 10))
    for init in ("random", "nndsvd", "nndsvda", "nndsvdar"):
        W, H = nmf._initialize_nmf(data, 10, init=init, random_state=0)
        assert not ((W < 0).any() or (H < 0).any())


@pytest.mark.filterwarnings(
    r"ignore:The multiplicative update \('mu'\) solver cannot update zeros present in"
    r" the initialization",
)
def test_parameter_checking():
    # [FUNCTION DOC]
    # Checks non-common parameter validation: cd solver with beta_loss=1 raises
    # ValueError; negative input to fit, transform and _initialize_nmf raises the
    # "Negative values" ValueError; nndsvd-family inits with n_components larger than
    # min(n_samples, n_features) raise ValueError in NMF, MiniBatchNMF and
    # _initialize_nmf.
    # Here we only check for invalid parameter values that are not already
    # automatically tested in the common tests.

    A = np.ones((2, 2))

    msg = "Invalid beta_loss parameter: solver 'cd' does not handle beta_loss = 1.0"
    with pytest.raises(ValueError, match=msg):
        NMF(solver="cd", beta_loss=1.0).fit(A)
    msg = "Negative values in data passed to"
    with pytest.raises(ValueError, match=msg):
        NMF().fit(-A)
    clf = NMF(2, tol=0.1).fit(A)
    with pytest.raises(ValueError, match=msg):
        clf.transform(-A)
    with pytest.raises(ValueError, match=msg):
        nmf._initialize_nmf(-A, 2, "nndsvd")

    for init in ["nndsvd", "nndsvda", "nndsvdar"]:
        msg = re.escape(
            "init = '{}' can only be used when "
            "n_components <= min(n_samples, n_features)".format(init)
        )
        with pytest.raises(ValueError, match=msg):
            NMF(3, init=init).fit(A)
        with pytest.raises(ValueError, match=msg):
            MiniBatchNMF(3, init=init).fit(A)
        with pytest.raises(ValueError, match=msg):
            nmf._initialize_nmf(A, 3, init)


def test_initialize_close():
    # [FUNCTION DOC]
    # Checks that the nndsvd initialization's reconstruction error (Frobenius norm of
    # W H minus A) does not exceed the Frobenius norm of A minus its mean.
    # Test NNDSVD error
    # Test that _initialize_nmf error is less than the standard deviation of
    # the entries in the matrix.
    rng = np.random.mtrand.RandomState(42)
    A = np.abs(rng.randn(10, 10))
    W, H = nmf._initialize_nmf(A, 10, init="nndsvd")
    error = linalg.norm(np.dot(W, H) - A)
    sdev = linalg.norm(A - A.mean())
    assert error <= sdev


def test_initialize_variants():
    # [FUNCTION DOC]
    # Checks nndsvda and nndsvdar initializations equal plain nndsvd wherever nndsvd
    # is non-zero (they only fill zeros), for both W and H.
    # Test NNDSVD variants correctness
    # Test that the variants 'nndsvda' and 'nndsvdar' differ from basic
    # 'nndsvd' only where the basic version has zeros.
    rng = np.random.mtrand.RandomState(42)
    data = np.abs(rng.randn(10, 10))
    W0, H0 = nmf._initialize_nmf(data, 10, init="nndsvd")
    Wa, Ha = nmf._initialize_nmf(data, 10, init="nndsvda")
    War, Har = nmf._initialize_nmf(data, 10, init="nndsvdar", random_state=0)

    for ref, evl in ((W0, Wa), (W0, War), (H0, Ha), (H0, Har)):
        assert_almost_equal(evl[ref != 0], ref[ref != 0])


# ignore UserWarning raised when both solver='mu' and init='nndsvd'
@pytest.mark.filterwarnings(
    r"ignore:The multiplicative update \('mu'\) solver cannot update zeros present in"
    r" the initialization"
)
@pytest.mark.parametrize(
    ["Estimator", "solver"],
    [[NMF, {"solver": "cd"}], [NMF, {"solver": "mu"}], [MiniBatchNMF, {}]],
)
@pytest.mark.parametrize("init", (None, "nndsvd", "nndsvda", "nndsvdar", "random"))
@pytest.mark.parametrize("alpha_W", (0.0, 1.0))
@pytest.mark.parametrize("alpha_H", (0.0, 1.0, "same"))
def test_nmf_fit_nn_output(Estimator, solver, init, alpha_W, alpha_H):
    # [FUNCTION DOC]
    # Checks fit_transform output and components_ contain no negative values for
    # every estimator/solver, init option and combination of alpha_W and alpha_H, on
    # a small 5x2 matrix.
    # Test that the decomposition does not contain negative values
    A = np.c_[5.0 - np.arange(1, 6), 5.0 + np.arange(1, 6)]
    model = Estimator(
        n_components=2,
        init=init,
        alpha_W=alpha_W,
        alpha_H=alpha_H,
        random_state=0,
        **solver,
    )
    transf = model.fit_transform(A)
    assert not ((model.components_ < 0).any() or (transf < 0).any())


@pytest.mark.parametrize(
    ["Estimator", "solver"],
    [[NMF, {"solver": "cd"}], [NMF, {"solver": "mu"}], [MiniBatchNMF, {}]],
)
def test_nmf_fit_close(Estimator, solver):
    # [FUNCTION DOC]
    # Checks a 5-component fit (nndsvdar init, 600 iterations) on random positive
    # 6x5 data reaches reconstruction_err_ below 0.1, for each estimator.
    rng = np.random.mtrand.RandomState(42)
    # Test that the fit is not too far away
    pnmf = Estimator(
        5,
        init="nndsvdar",
        random_state=0,
        max_iter=600,
        **solver,
    )
    X = np.abs(rng.randn(6, 5))
    assert pnmf.fit(X).reconstruction_err_ < 0.1


def test_nmf_true_reconstruction():
    # [FUNCTION DOC]
    # Builds X exactly as the product of sparse non-negative W_true (15x5) and
    # H_true (5x10) and checks NMF (mu, beta_loss=1) and MiniBatchNMF (batch_size 3)
    # reach reconstruction_err_ below 0.1 and reproduce X (MiniBatchNMF with atol 1).
    # Test that the fit is not too far away from an exact solution
    # (by construction)
    n_samples = 15
    n_features = 10
    n_components = 5
    beta_loss = 1
    batch_size = 3
    max_iter = 1000

    rng = np.random.mtrand.RandomState(42)
    W_true = np.zeros([n_samples, n_components])
    W_array = np.abs(rng.randn(n_samples))
    for j in range(n_components):
        W_true[j % n_samples, j] = W_array[j % n_samples]
    H_true = np.zeros([n_components, n_features])
    H_array = np.abs(rng.randn(n_components))
    for j in range(n_features):
        H_true[j % n_components, j] = H_array[j % n_components]
    X = np.dot(W_true, H_true)

    model = NMF(
        n_components=n_components,
        solver="mu",
        beta_loss=beta_loss,
        max_iter=max_iter,
        random_state=0,
    )
    transf = model.fit_transform(X)
    X_calc = np.dot(transf, model.components_)

    assert model.reconstruction_err_ < 0.1
    assert_allclose(X, X_calc)

    mbmodel = MiniBatchNMF(
        n_components=n_components,
        beta_loss=beta_loss,
        batch_size=batch_size,
        random_state=0,
        max_iter=max_iter,
    )
    transf = mbmodel.fit_transform(X)
    X_calc = np.dot(transf, mbmodel.components_)

    assert mbmodel.reconstruction_err_ < 0.1
    assert_allclose(X, X_calc, atol=1)


@pytest.mark.parametrize("solver", ["cd", "mu"])
def test_nmf_transform(solver):
    # [FUNCTION DOC]
    # Checks NMF fit_transform and a subsequent transform give close W (atol 0.1)
    # for cd and mu solvers.
    # Test that fit_transform is equivalent to fit.transform for NMF
    # Test that NMF.transform returns close values
    rng = np.random.mtrand.RandomState(42)
    A = np.abs(rng.randn(6, 5))
    m = NMF(
        solver=solver,
        n_components=3,
        init="random",
        random_state=0,
        tol=1e-6,
    )
    ft = m.fit_transform(A)
    t = m.transform(A)
    assert_allclose(ft, t, atol=1e-1)


def test_minibatch_nmf_transform():
    # [FUNCTION DOC]
    # Checks MiniBatchNMF with fresh_restarts=True gives the same W from
    # fit_transform and transform.
    # Test that fit_transform is equivalent to fit.transform for MiniBatchNMF
    # Only guaranteed with fresh restarts
    rng = np.random.mtrand.RandomState(42)
    A = np.abs(rng.randn(6, 5))
    m = MiniBatchNMF(
        n_components=3,
        random_state=0,
        tol=1e-3,
        fresh_restarts=True,
    )
    ft = m.fit_transform(A)
    t = m.transform(A)
    assert_allclose(ft, t)


@pytest.mark.parametrize(
    ["Estimator", "solver"],
    [[NMF, {"solver": "cd"}], [NMF, {"solver": "mu"}], [MiniBatchNMF, {}]],
)
def test_nmf_transform_custom_init(Estimator, solver):
    # [FUNCTION DOC]
    # Smoke test: fit_transform with init="custom" and user W/H followed by transform
    # runs without error for each estimator.
    # Smoke test that checks if NMF.transform works with custom initialization
    random_state = np.random.RandomState(0)
    A = np.abs(random_state.randn(6, 5))
    n_components = 4
    avg = np.sqrt(A.mean() / n_components)
    H_init = np.abs(avg * random_state.randn(n_components, 5))
    W_init = np.abs(avg * random_state.randn(6, n_components))

    m = Estimator(
        n_components=n_components, init="custom", random_state=0, tol=1e-3, **solver
    )
    m.fit_transform(A, W=W_init, H=H_init)
    m.transform(A)


@pytest.mark.parametrize("solver", ("cd", "mu"))
def test_nmf_inverse_transform(solver):
    # [FUNCTION DOC]
    # Checks inverse_transform of NMF fit_transform output reconstructs A to 2
    # decimals with a full-rank (4-component) model, for cd and mu.
    # Test that NMF.inverse_transform returns close values
    random_state = np.random.RandomState(0)
    A = np.abs(random_state.randn(6, 4))
    m = NMF(
        solver=solver,
        n_components=4,
        init="random",
        random_state=0,
        max_iter=1000,
    )
    ft = m.fit_transform(A)
    A_new = m.inverse_transform(ft)
    assert_array_almost_equal(A, A_new, decimal=2)


def test_mbnmf_inverse_transform():
    # [FUNCTION DOC]
    # Checks MiniBatchNMF inverse_transform of fit_transform output is close to the
    # input (rtol 1e-3, atol 1e-2) with fresh restarts and nndsvdar init.
    # Test that MiniBatchNMF.transform followed by MiniBatchNMF.inverse_transform
    # is close to the identity
    rng = np.random.RandomState(0)
    A = np.abs(rng.randn(6, 4))
    nmf = MiniBatchNMF(
        random_state=rng,
        max_iter=500,
        init="nndsvdar",
        fresh_restarts=True,
    )
    ft = nmf.fit_transform(A)
    A_new = nmf.inverse_transform(ft)
    assert_allclose(A, A_new, rtol=1e-3, atol=1e-2)


@pytest.mark.parametrize("Estimator", [NMF, MiniBatchNMF])
def test_n_components_greater_n_features(Estimator):
    # [FUNCTION DOC]
    # Smoke test: fitting with 15 components on 10-feature data does not fail.
    # Smoke test for the case of more components than features.
    rng = np.random.mtrand.RandomState(42)
    A = np.abs(rng.randn(30, 10))
    Estimator(n_components=15, random_state=0, tol=1e-2).fit(A)


@pytest.mark.parametrize(
    ["Estimator", "solver"],
    [[NMF, {"solver": "cd"}], [NMF, {"solver": "mu"}], [MiniBatchNMF, {}]],
)
@pytest.mark.parametrize("sparse_container", CSC_CONTAINERS + CSR_CONTAINERS)
@pytest.mark.parametrize("alpha_W", (0.0, 1.0))
@pytest.mark.parametrize("alpha_H", (0.0, 1.0, "same"))
def test_nmf_sparse_input(Estimator, solver, sparse_container, alpha_W, alpha_H):
    # [FUNCTION DOC]
    # Checks dense and sparse (CSC/CSR) versions of data with zero columns give the
    # same W and H for each estimator and regularization setting (random init,
    # 100 iterations, tol 0).
    # Test that sparse matrices are accepted as input
    rng = np.random.mtrand.RandomState(42)
    A = np.abs(rng.randn(10, 10))
    A[:, 2 * np.arange(5)] = 0
    A_sparse = sparse_container(A)

    est1 = Estimator(
        n_components=5,
        init="random",
        alpha_W=alpha_W,
        alpha_H=alpha_H,
        random_state=0,
        tol=0,
        max_iter=100,
        **solver,
    )
    est2 = clone(est1)

    W1 = est1.fit_transform(A)
    W2 = est2.fit_transform(A_sparse)
    H1 = est1.components_
    H2 = est2.components_

    assert_allclose(W1, W2)
    assert_allclose(H1, H2)


@pytest.mark.parametrize(
    ["Estimator", "solver"],
    [[NMF, {"solver": "cd"}], [NMF, {"solver": "mu"}], [MiniBatchNMF, {}]],
)
@pytest.mark.parametrize("csc_container", CSC_CONTAINERS)
def test_nmf_sparse_transform(Estimator, solver, csc_container):
    # [FUNCTION DOC]
    # Regression test for issue #2124: fit_transform and transform agree
    # (atol 0.1) on sparse CSC input containing a zero entry.
    # Test that transform works on sparse data.  Issue #2124
    rng = np.random.mtrand.RandomState(42)
    A = np.abs(rng.randn(3, 2))
    A[1, 1] = 0
    A = csc_container(A)

    model = Estimator(random_state=0, n_components=2, max_iter=400, **solver)
    A_fit_tr = model.fit_transform(A)
    A_tr = model.transform(A)
    assert_allclose(A_fit_tr, A_tr, atol=1e-1)


@pytest.mark.parametrize("init", ["random", "nndsvd"])
@pytest.mark.parametrize("solver", ("cd", "mu"))
@pytest.mark.parametrize("alpha_W", (0.0, 1.0))
@pytest.mark.parametrize("alpha_H", (0.0, 1.0, "same"))
def test_non_negative_factorization_consistency(init, solver, alpha_W, alpha_H):
    # [FUNCTION DOC]
    # Checks non_negative_factorization and the NMF class produce the same W both
    # for fitting and for transform (update_H=False with the fitted H), across init,
    # solver and regularization settings.
    # Test that the function is called in the same way, either directly
    # or through the NMF class
    max_iter = 500
    rng = np.random.mtrand.RandomState(42)
    A = np.abs(rng.randn(10, 10))
    A[:, 2 * np.arange(5)] = 0

    W_nmf, H, _ = non_negative_factorization(
        A,
        init=init,
        solver=solver,
        max_iter=max_iter,
        alpha_W=alpha_W,
        alpha_H=alpha_H,
        random_state=1,
        tol=1e-2,
    )
    W_nmf_2, H, _ = non_negative_factorization(
        A,
        H=H,
        update_H=False,
        init=init,
        solver=solver,
        max_iter=max_iter,
        alpha_W=alpha_W,
        alpha_H=alpha_H,
        random_state=1,
        tol=1e-2,
    )

    model_class = NMF(
        init=init,
        solver=solver,
        max_iter=max_iter,
        alpha_W=alpha_W,
        alpha_H=alpha_H,
        random_state=1,
        tol=1e-2,
    )
    W_cls = model_class.fit_transform(A)
    W_cls_2 = model_class.transform(A)

    assert_allclose(W_nmf, W_cls)
    assert_allclose(W_nmf_2, W_cls_2)


def test_non_negative_factorization_checking():
    # [FUNCTION DOC]
    # Checks non_negative_factorization custom-init validation: negative H or W
    # raises ValueError naming the input, and an all-zero H raises the "full of
    # zeros" ValueError.
    # Note that the validity of parameter types and range of possible values
    # for scalar numerical or str parameters is already checked in the common
    # tests. Here we only check for problems that cannot be captured by simple
    # declarative constraints on the valid parameter values.

    A = np.ones((2, 2))
    # Test parameters checking in public function
    nnmf = non_negative_factorization
    msg = re.escape("Negative values in data passed to NMF (input H)")
    with pytest.raises(ValueError, match=msg):
        nnmf(A, A, -A, 2, init="custom")
    msg = re.escape("Negative values in data passed to NMF (input W)")
    with pytest.raises(ValueError, match=msg):
        nnmf(A, -A, A, 2, init="custom")
    msg = re.escape("Array passed to NMF (input H) is full of zeros")
    with pytest.raises(ValueError, match=msg):
        nnmf(A, A, 0 * A, 2, init="custom")


def _beta_divergence_dense(X, W, H, beta):
    # [FUNCTION DOC]
    # HELPER: Reference dense implementation of the beta-divergence between X and
    # W H. beta 2: half the squared Frobenius norm of the difference. Otherwise only
    # entries where X is non-zero are used, with W H clipped below at 1e-9:
    # beta 1 (generalized KL): sum of X log(X / WH) plus sum(WH) minus sum(X);
    # beta 0 (Itakura-Saito): sum of X/WH minus X.size minus sum of log(X/WH);
    # other beta: (sum X^beta + (beta-1) sum WH^beta - beta sum X WH^(beta-1)) divided
    # by beta (beta-1), where the WH^beta term uses all entries.
    # Inputs: X - dense non-negative 2D array; W, H - factor arrays; beta - float.
    # Outputs: float divergence.
    """Compute the beta-divergence of X and W.H for dense array only.

    Used as a reference for testing nmf._beta_divergence.
    """
    WH = np.dot(W, H)

    if beta == 2:
        return squared_norm(X - WH) / 2

    WH_Xnonzero = WH[X != 0]
    X_nonzero = X[X != 0]
    np.maximum(WH_Xnonzero, 1e-9, out=WH_Xnonzero)

    if beta == 1:
        res = np.sum(X_nonzero * np.log(X_nonzero / WH_Xnonzero))
        res += WH.sum() - X.sum()

    elif beta == 0:
        div = X_nonzero / WH_Xnonzero
        res = np.sum(div) - X.size - np.sum(np.log(div))
    else:
        res = (X_nonzero**beta).sum()
        res += (beta - 1) * (WH**beta).sum()
        res -= beta * (X_nonzero * (WH_Xnonzero ** (beta - 1))).sum()
        res /= beta * (beta - 1)

    return res


@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_beta_divergence(csr_container):
    # [FUNCTION DOC]
    # Checks nmf._beta_divergence on dense and CSR input matches
    # _beta_divergence_dense (7 decimals) for beta in 0, 0.5, 1, 1.5, 2, 3.
    # Compare _beta_divergence with the reference _beta_divergence_dense
    n_samples = 20
    n_features = 10
    n_components = 5
    beta_losses = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]

    # initialization
    rng = np.random.mtrand.RandomState(42)
    X = rng.randn(n_samples, n_features)
    np.clip(X, 0, None, out=X)
    X_csr = csr_container(X)
    W, H = nmf._initialize_nmf(X, n_components, init="random", random_state=42)

    for beta in beta_losses:
        ref = _beta_divergence_dense(X, W, H, beta)
        loss = nmf._beta_divergence(X, W, H, beta)
        loss_csr = nmf._beta_divergence(X_csr, W, H, beta)

        assert_almost_equal(ref, loss, decimal=7)
        assert_almost_equal(ref, loss_csr, decimal=7)


@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_special_sparse_dot(csr_container):
    # [FUNCTION DOC]
    # Checks _special_sparse_dot: with sparse X it returns W H only at X's non-zero
    # positions, matching the dense product there, with identical sparsity
    # structure (indices, indptr, shape) to X.
    # Test the function that computes np.dot(W, H), only where X is non zero.
    n_samples = 10
    n_features = 5
    n_components = 3
    rng = np.random.mtrand.RandomState(42)
    X = rng.randn(n_samples, n_features)
    np.clip(X, 0, None, out=X)
    X_csr = csr_container(X)

    W = np.abs(rng.randn(n_samples, n_components))
    H = np.abs(rng.randn(n_components, n_features))

    WH_safe = nmf._special_sparse_dot(W, H, X_csr)
    WH = nmf._special_sparse_dot(W, H, X)

    # test that both results have same values, in X_csr nonzero elements
    ii, jj = X_csr.nonzero()
    WH_safe_data = np.asarray(WH_safe[ii, jj]).ravel()
    assert_array_almost_equal(WH_safe_data, WH[ii, jj], decimal=10)

    # test that WH_safe and X_csr have the same sparse structure
    assert_array_equal(WH_safe.indices, X_csr.indices)
    assert_array_equal(WH_safe.indptr, X_csr.indptr)
    assert_array_equal(WH_safe.shape, X_csr.shape)


@pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_nmf_multiplicative_update_sparse(csr_container):
    # [FUNCTION DOC]
    # For several beta_loss values, checks mu-solver non_negative_factorization from
    # the same custom init gives the same W and H on dense and CSR input, and that
    # perturbing beta_loss by 1e-5 changes the result by at most 1e-4 (continuity).
    # Convergence warnings are ignored.
    # Compare sparse and dense input in multiplicative update NMF
    # Also test continuity of the results with respect to beta_loss parameter
    n_samples = 20
    n_features = 10
    n_components = 5
    alpha = 0.1
    l1_ratio = 0.5
    n_iter = 20

    # initialization
    rng = np.random.mtrand.RandomState(1337)
    X = rng.randn(n_samples, n_features)
    X = np.abs(X)
    X_csr = csr_container(X)
    W0, H0 = nmf._initialize_nmf(X, n_components, init="random", random_state=42)

    for beta_loss in (-1.2, 0, 0.2, 1.0, 2.0, 2.5):
        # Reference with dense array X
        W, H = W0.copy(), H0.copy()
        W1, H1, _ = non_negative_factorization(
            X,
            W,
            H,
            n_components,
            init="custom",
            update_H=True,
            solver="mu",
            beta_loss=beta_loss,
            max_iter=n_iter,
            alpha_W=alpha,
            l1_ratio=l1_ratio,
            random_state=42,
        )

        # Compare with sparse X
        W, H = W0.copy(), H0.copy()
        W2, H2, _ = non_negative_factorization(
            X_csr,
            W,
            H,
            n_components,
            init="custom",
            update_H=True,
            solver="mu",
            beta_loss=beta_loss,
            max_iter=n_iter,
            alpha_W=alpha,
            l1_ratio=l1_ratio,
            random_state=42,
        )

        assert_allclose(W1, W2, atol=1e-7)
        assert_allclose(H1, H2, atol=1e-7)

        # Compare with almost same beta_loss, since some values have a specific
        # behavior, but the results should be continuous w.r.t beta_loss
        beta_loss -= 1.0e-5
        W, H = W0.copy(), H0.copy()
        W3, H3, _ = non_negative_factorization(
            X_csr,
            W,
            H,
            n_components,
            init="custom",
            update_H=True,
            solver="mu",
            beta_loss=beta_loss,
            max_iter=n_iter,
            alpha_W=alpha,
            l1_ratio=l1_ratio,
            random_state=42,
        )

        assert_allclose(W1, W3, atol=1e-4)
        assert_allclose(H1, H3, atol=1e-4)


@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_nmf_negative_beta_loss(csr_container):
    # [FUNCTION DOC]
    # Checks the mu solver raises ValueError ("solver may diverge") for beta_loss
    # <= 0 when X has zeros but produces no NaN once X is shifted by 1e-9; for
    # positive beta_loss values it produces no NaN on dense and CSR input with zeros.
    # Test that an error is raised if beta_loss < 0 and X contains zeros.
    # Test that the output has not NaN values when the input contains zeros.
    n_samples = 6
    n_features = 5
    n_components = 3

    rng = np.random.mtrand.RandomState(42)
    X = rng.randn(n_samples, n_features)
    np.clip(X, 0, None, out=X)
    X_csr = csr_container(X)

    def _assert_nmf_no_nan(X, beta_loss):
        # [FUNCTION DOC]
        # HELPER: Runs non_negative_factorization (mu solver, random init, 1000
        # iterations) on X with the given beta_loss and asserts W and H contain no NaN.
        # Inputs: X - dense or sparse non-negative data; beta_loss - float.
        # Exceptions: propagates ValueError from non_negative_factorization.
        W, H, _ = non_negative_factorization(
            X,
            init="random",
            n_components=n_components,
            solver="mu",
            beta_loss=beta_loss,
            random_state=0,
            max_iter=1000,
        )
        assert not np.any(np.isnan(W))
        assert not np.any(np.isnan(H))

    msg = "When beta_loss <= 0 and X contains zeros, the solver may diverge."
    for beta_loss in (-0.6, 0.0):
        with pytest.raises(ValueError, match=msg):
            _assert_nmf_no_nan(X, beta_loss)
        _assert_nmf_no_nan(X + 1e-9, beta_loss)

    for beta_loss in (0.2, 1.0, 1.2, 2.0, 2.5):
        _assert_nmf_no_nan(X, beta_loss)
        _assert_nmf_no_nan(X_csr, beta_loss)


@pytest.mark.parametrize("beta_loss", [-0.5, 0.0])
def test_minibatch_nmf_negative_beta_loss(beta_loss):
    # [FUNCTION DOC]
    # Checks MiniBatchNMF.fit raises the "solver may diverge" ValueError for
    # beta_loss <= 0 when X contains zeros.
    """Check that an error is raised if beta_loss < 0 and X contains zeros."""
    rng = np.random.RandomState(0)
    X = rng.normal(size=(6, 5))
    X[X < 0] = 0

    nmf = MiniBatchNMF(beta_loss=beta_loss, random_state=0)

    msg = "When beta_loss <= 0 and X contains zeros, the solver may diverge."
    with pytest.raises(ValueError, match=msg):
        nmf.fit(X)


@pytest.mark.parametrize(
    ["Estimator", "solver"],
    [[NMF, {"solver": "cd"}], [NMF, {"solver": "mu"}], [MiniBatchNMF, {}]],
)
def test_nmf_regularization(Estimator, solver):
    # [FUNCTION DOC]
    # Checks regularization effects for each estimator: pure L1 (alpha_W 0.5,
    # alpha_H defaulting to "same") yields more near-zero entries in W and H than no
    # regularization; pure L2 yields a smaller sum of squared Frobenius norms of W
    # and H.
    # Test the effect of L1 and L2 regularizations
    n_samples = 6
    n_features = 5
    n_components = 3
    rng = np.random.mtrand.RandomState(42)
    X = np.abs(rng.randn(n_samples, n_features))

    # L1 regularization should increase the number of zeros
    l1_ratio = 1.0
    regul = Estimator(
        n_components=n_components,
        alpha_W=0.5,
        l1_ratio=l1_ratio,
        random_state=42,
        **solver,
    )
    model = Estimator(
        n_components=n_components,
        alpha_W=0.0,
        l1_ratio=l1_ratio,
        random_state=42,
        **solver,
    )

    W_regul = regul.fit_transform(X)
    W_model = model.fit_transform(X)

    H_regul = regul.components_
    H_model = model.components_

    eps = np.finfo(np.float64).eps
    W_regul_n_zeros = W_regul[W_regul <= eps].size
    W_model_n_zeros = W_model[W_model <= eps].size
    H_regul_n_zeros = H_regul[H_regul <= eps].size
    H_model_n_zeros = H_model[H_model <= eps].size

    assert W_regul_n_zeros > W_model_n_zeros
    assert H_regul_n_zeros > H_model_n_zeros

    # L2 regularization should decrease the sum of the squared norm
    # of the matrices W and H
    l1_ratio = 0.0
    regul = Estimator(
        n_components=n_components,
        alpha_W=0.5,
        l1_ratio=l1_ratio,
        random_state=42,
        **solver,
    )
    model = Estimator(
        n_components=n_components,
        alpha_W=0.0,
        l1_ratio=l1_ratio,
        random_state=42,
        **solver,
    )

    W_regul = regul.fit_transform(X)
    W_model = model.fit_transform(X)

    H_regul = regul.components_
    H_model = model.components_

    assert (linalg.norm(W_model)) ** 2.0 + (linalg.norm(H_model)) ** 2.0 > (
        linalg.norm(W_regul)
    ) ** 2.0 + (linalg.norm(H_regul)) ** 2.0


@pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
@pytest.mark.parametrize("solver", ("cd", "mu"))
def test_nmf_decreasing(solver):
    # [FUNCTION DOC]
    # Checks the regularized objective (beta divergence plus L1/L2 penalties scaled
    # by n_features for W and n_samples for H) strictly decreases over 30 successive
    # single-iteration non_negative_factorization calls, for supported
    # solver/beta_loss pairs (cd only with beta_loss 2).
    # test that the objective function is decreasing at each iteration
    n_samples = 20
    n_features = 15
    n_components = 10
    alpha = 0.1
    l1_ratio = 0.5
    tol = 0.0

    # initialization
    rng = np.random.mtrand.RandomState(42)
    X = rng.randn(n_samples, n_features)
    np.abs(X, X)
    W0, H0 = nmf._initialize_nmf(X, n_components, init="random", random_state=42)

    for beta_loss in (-1.2, 0, 0.2, 1.0, 2.0, 2.5):
        if solver != "mu" and beta_loss != 2:
            # not implemented
            continue
        W, H = W0.copy(), H0.copy()
        previous_loss = None
        for _ in range(30):
            # one more iteration starting from the previous results
            W, H, _ = non_negative_factorization(
                X,
                W,
                H,
                beta_loss=beta_loss,
                init="custom",
                n_components=n_components,
                max_iter=1,
                alpha_W=alpha,
                solver=solver,
                tol=tol,
                l1_ratio=l1_ratio,
                verbose=0,
                random_state=0,
                update_H=True,
            )

            loss = (
                nmf._beta_divergence(X, W, H, beta_loss)
                + alpha * l1_ratio * n_features * W.sum()
                + alpha * l1_ratio * n_samples * H.sum()
                + alpha * (1 - l1_ratio) * n_features * (W**2).sum()
                + alpha * (1 - l1_ratio) * n_samples * (H**2).sum()
            )
            if previous_loss is not None:
                assert previous_loss > loss
            previous_loss = loss


def test_nmf_underflow():
    # [FUNCTION DOC]
    # Regression test for an underflow in _beta_divergence: with beta=1, replacing a
    # zero entry of X by 1e-323 gives almost the same divergence.
    # Regression test for an underflow issue in _beta_divergence
    rng = np.random.RandomState(0)
    n_samples, n_features, n_components = 10, 2, 2
    X = np.abs(rng.randn(n_samples, n_features)) * 10
    W = np.abs(rng.randn(n_samples, n_components)) * 10
    H = np.abs(rng.randn(n_components, n_features))

    X[0, 0] = 0
    ref = nmf._beta_divergence(X, W, H, beta=1.0)
    X[0, 0] = 1e-323
    res = nmf._beta_divergence(X, W, H, beta=1.0)
    assert_almost_equal(res, ref)


@pytest.mark.parametrize(
    "dtype_in, dtype_out",
    [
        (np.float32, np.float32),
        (np.float64, np.float64),
        (np.int32, np.float64),
        (np.int64, np.float64),
    ],
)
@pytest.mark.parametrize(
    ["Estimator", "solver"],
    [[NMF, {"solver": "cd"}], [NMF, {"solver": "mu"}], [MiniBatchNMF, {}]],
)
def test_nmf_dtype_match(Estimator, solver, dtype_in, dtype_out):
    # [FUNCTION DOC]
    # Checks dtype handling: float32 and float64 inputs are preserved and integer
    # inputs produce float64 for transform, fit_transform and components_.
    # Check that NMF preserves dtype (float32 and float64)
    X = np.random.RandomState(0).randn(20, 15).astype(dtype_in, copy=False)
    np.abs(X, out=X)

    nmf = Estimator(
        alpha_W=1.0,
        alpha_H=1.0,
        tol=1e-2,
        random_state=0,
        **solver,
    )

    assert nmf.fit(X).transform(X).dtype == dtype_out
    assert nmf.fit_transform(X).dtype == dtype_out
    assert nmf.components_.dtype == dtype_out


@pytest.mark.parametrize(
    ["Estimator", "solver"],
    [[NMF, {"solver": "cd"}], [NMF, {"solver": "mu"}], [MiniBatchNMF, {}]],
)
def test_nmf_float32_float64_consistency(Estimator, solver):
    # [FUNCTION DOC]
    # Checks float32 and float64 fits of the same data give close W (atol 1e-5).
    # Check that the result of NMF is the same between float32 and float64
    X = np.random.RandomState(0).randn(50, 7)
    np.abs(X, out=X)
    nmf32 = Estimator(random_state=0, tol=1e-3, **solver)
    W32 = nmf32.fit_transform(X.astype(np.float32))
    nmf64 = Estimator(random_state=0, tol=1e-3, **solver)
    W64 = nmf64.fit_transform(X)

    assert_allclose(W32, W64, atol=1e-5)


@pytest.mark.parametrize("Estimator", [NMF, MiniBatchNMF])
def test_nmf_custom_init_dtype_error(Estimator):
    # [FUNCTION DOC]
    # Checks a TypeError ("should have the same dtype as X") when custom H has a
    # different dtype than X, for estimator fit and non_negative_factorization.
    # Check that an error is raise if custom H and/or W don't have the same
    # dtype as X.
    rng = np.random.RandomState(0)
    X = rng.random_sample((20, 15))
    H = rng.random_sample((15, 15)).astype(np.float32)
    W = rng.random_sample((20, 15))

    with pytest.raises(TypeError, match="should have the same dtype as X"):
        Estimator(init="custom").fit(X, H=H, W=W)

    with pytest.raises(TypeError, match="should have the same dtype as X"):
        non_negative_factorization(X, H=H, update_H=False)


@pytest.mark.parametrize("beta_loss", [-0.5, 0, 0.5, 1, 1.5, 2, 2.5])
def test_nmf_minibatchnmf_equivalence(beta_loss):
    # [FUNCTION DOC]
    # Checks MiniBatchNMF with batch_size equal to n_samples, forget_factor 0 and
    # no early stopping gives the same W as NMF with the mu solver, for several
    # beta_loss values.
    # Test that MiniBatchNMF is equivalent to NMF when batch_size = n_samples and
    # forget_factor 0.0 (stopping criterion put aside)
    rng = np.random.mtrand.RandomState(42)
    X = np.abs(rng.randn(48, 5))

    nmf = NMF(
        n_components=5,
        beta_loss=beta_loss,
        solver="mu",
        random_state=0,
        tol=0,
    )
    mbnmf = MiniBatchNMF(
        n_components=5,
        beta_loss=beta_loss,
        random_state=0,
        tol=0,
        max_no_improvement=None,
        batch_size=X.shape[0],
        forget_factor=0.0,
    )
    W = nmf.fit_transform(X)
    mbW = mbnmf.fit_transform(X)
    assert_allclose(W, mbW)


def test_minibatch_nmf_partial_fit():
    # [FUNCTION DOC]
    # Checks fit with max_iter=2 and batch_size=10 (fresh_restarts off) matches
    # repeated partial_fit calls from the same custom init: same n_steps_ and
    # components_.
    # KNOWN ISSUE: the partial_fit loop slices X[j : j + batch_size] for j in
    # range(batch_size), i.e. overlapping windows over the first 19 rows instead of
    # consecutive batches (j * batch_size); the equality only holds if the test is
    # weaker than intended.
    # Check fit / partial_fit equivalence. Applicable only with fresh restarts.
    rng = np.random.mtrand.RandomState(42)
    X = np.abs(rng.randn(100, 5))

    n_components = 5
    batch_size = 10
    max_iter = 2

    mbnmf1 = MiniBatchNMF(
        n_components=n_components,
        init="custom",
        random_state=0,
        max_iter=max_iter,
        batch_size=batch_size,
        tol=0,
        max_no_improvement=None,
        fresh_restarts=False,
    )
    mbnmf2 = MiniBatchNMF(n_components=n_components, init="custom", random_state=0)

    # Force the same init of H (W is recomputed anyway) to be able to compare results.
    W, H = nmf._initialize_nmf(
        X, n_components=n_components, init="random", random_state=0
    )

    mbnmf1.fit(X, W=W, H=H)
    for i in range(max_iter):
        for j in range(batch_size):
            mbnmf2.partial_fit(X[j : j + batch_size], W=W[:batch_size], H=H)

    assert mbnmf1.n_steps_ == mbnmf2.n_steps_
    assert_allclose(mbnmf1.components_, mbnmf2.components_)


def test_feature_names_out():
    # [FUNCTION DOC]
    # Checks get_feature_names_out returns "nmf0".."nmf2" for a 3-component NMF.
    """Check feature names out for NMF."""
    random_state = np.random.RandomState(0)
    X = np.abs(random_state.randn(10, 4))
    nmf = NMF(n_components=3).fit(X)

    names = nmf.get_feature_names_out()
    assert_array_equal([f"nmf{i}" for i in range(3)], names)


def test_minibatch_nmf_verbose():
    # [FUNCTION DOC]
    # Coverage-only test: fits MiniBatchNMF with verbose=1 while redirecting stdout
    # to a StringIO; no assertions are made on the output.
    # Side effects: temporarily replaces sys.stdout (restored in a finally block).
    # Check verbose mode of MiniBatchNMF for better coverage.
    A = np.random.RandomState(0).random_sample((100, 10))
    nmf = MiniBatchNMF(tol=1e-2, random_state=0, verbose=1)
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    try:
        nmf.fit(A)
    finally:
        sys.stdout = old_stdout


@pytest.mark.parametrize("Estimator", [NMF, MiniBatchNMF])
def test_nmf_n_components_auto(Estimator):
    # [FUNCTION DOC]
    # Checks n_components="auto" with custom W/H infers _n_components from H's
    # number of rows.
    # Check that n_components is correctly inferred
    # from the provided custom initialization.
    rng = np.random.RandomState(0)
    X = rng.random_sample((6, 5))
    W = rng.random_sample((6, 2))
    H = rng.random_sample((2, 5))
    est = Estimator(
        n_components="auto",
        init="custom",
        random_state=0,
        tol=1e-6,
    )
    est.fit_transform(X, W=W, H=H)
    assert est._n_components == H.shape[0]


def test_nmf_non_negative_factorization_n_components_auto():
    # [FUNCTION DOC]
    # Checks non_negative_factorization with n_components="auto" and custom init
    # returns W and H with the shapes of the given initial arrays.
    # Check that n_components is correctly inferred from the provided
    # custom initialization.
    rng = np.random.RandomState(0)
    X = rng.random_sample((6, 5))
    W_init = rng.random_sample((6, 2))
    H_init = rng.random_sample((2, 5))
    W, H, _ = non_negative_factorization(
        X, W=W_init, H=H_init, init="custom", n_components="auto"
    )
    assert H.shape == H_init.shape
    assert W.shape == W_init.shape


def test_nmf_n_components_auto_no_h_update():
    # [FUNCTION DOC]
    # Checks non_negative_factorization with n_components="auto" and update_H=False
    # keeps H unchanged and returns W of shape (n_samples, H rows).
    # Tests that non_negative_factorization does not fail when setting
    # n_components="auto" also tests that the inferred n_component
    # value is the right one.
    rng = np.random.RandomState(0)
    X = rng.random_sample((6, 5))
    H_true = rng.random_sample((2, 5))
    W, H, _ = non_negative_factorization(
        X, H=H_true, n_components="auto", update_H=False
    )  # should not fail
    assert_allclose(H, H_true)
    assert W.shape == (X.shape[0], H_true.shape[0])


def test_nmf_w_h_not_used_warning():
    # [FUNCTION DOC]
    # Checks RuntimeWarnings: user W/H ignored when init is not custom (with H only,
    # and with W and H), and W ignored when update_H=False.
    # Check that warnings are raised if user provided W and H are not used
    # and initialization overrides value of W or H
    rng = np.random.RandomState(0)
    X = rng.random_sample((6, 5))
    W_init = rng.random_sample((6, 2))
    H_init = rng.random_sample((2, 5))
    with pytest.warns(
        RuntimeWarning,
        match="When init!='custom', provided W or H are ignored",
    ):
        non_negative_factorization(X, H=H_init, update_H=True, n_components="auto")

    with pytest.warns(
        RuntimeWarning,
        match="When init!='custom', provided W or H are ignored",
    ):
        non_negative_factorization(
            X, W=W_init, H=H_init, update_H=True, n_components="auto"
        )

    with pytest.warns(
        RuntimeWarning, match="When update_H=False, the provided initial W is not used."
    ):
        # When update_H is False, W is ignored regardless of init
        # TODO: use the provided W when init="custom".
        non_negative_factorization(
            X, W=W_init, H=H_init, update_H=False, n_components="auto"
        )


def test_nmf_custom_init_shape_error():
    # [FUNCTION DOC]
    # Checks custom W with a wrong first or second dimension raises ValueError with
    # an informative message.
    # Check that an informative error is raised when custom initialization does not
    # have the right shape
    rng = np.random.RandomState(0)
    X = rng.random_sample((6, 5))
    H = rng.random_sample((2, 5))
    nmf = NMF(n_components=2, init="custom", random_state=0)

    with pytest.raises(ValueError, match="Array with wrong first dimension passed"):
        nmf.fit(X, H=H, W=rng.random_sample((5, 2)))

    with pytest.raises(ValueError, match="Array with wrong second dimension passed"):
        nmf.fit(X, H=H, W=rng.random_sample((6, 3)))


@pytest.mark.parametrize("init", (None, "nndsvd", "nndsvda", "nndsvdar", "random"))
@pytest.mark.parametrize("shape", ((30, 10), (10, 30)), ids=("tall", "wide"))
@pytest.mark.parametrize("solver", ("cd", "mu"))
def test_nmf_smoke(init, shape, solver):
    # [FUNCTION DOC]
    # Smoke test: NMF fit_transform with every init and solver on tall and wide data
    # returns W and components_ of the expected shapes.
    """Smoke test NMF with all inits, solvers on tall/wide arrays."""
    rng = np.random.RandomState(0)
    X = np.abs(rng.random_sample(shape))

    nmf = NMF(n_components=5, init=init, random_state=0, solver=solver)
    W = nmf.fit_transform(X)

    assert W.shape == (shape[0], 5)
    assert nmf.components_.shape == (5, shape[1])
