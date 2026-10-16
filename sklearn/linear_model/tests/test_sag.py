# [MODULE DOC] Tests for the SAG / SAGA solvers (sklearn.linear_model._sag) as used by
#   LogisticRegression and Ridge. The module contains pure-Python reference
#   implementations of SAG (dense, `sag`) and of the lazy-update "just in time" variant
#   (`sag_sparse`), plus loss/derivative helpers, and checks that the fitted
#   coefficients and intercepts of the Cython solvers match these references on dense
#   and CSR input, that the primal objectives agree with other solvers, that the
#   automatic step size is computed correctly, and that error conditions are raised.
# Side effects at import time: loads the iris dataset into the module global `iris` (not
#   used by any test in this file).
# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import math
import re

import numpy as np
import pytest

from sklearn.base import clone
from sklearn.datasets import load_iris, make_blobs, make_classification
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.linear_model._sag import get_auto_step_size
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import check_random_state, compute_class_weight
from sklearn.utils._testing import (
    assert_allclose,
    assert_almost_equal,
    assert_array_almost_equal,
)
from sklearn.utils.extmath import row_norms
from sklearn.utils.fixes import CSR_CONTAINERS

iris = load_iris()


# this is used for sag classification
def log_dloss(p, y):
    # [FUNCTION DOC]
    # HELPER: Derivative of the logistic loss log(1 + exp(-y*p)) with respect to the
    #   prediction p, for a single sample.
    # Inputs: p - scalar decision value; y - scalar label in {-1, +1}.
    # Outputs: float equal to -y / (exp(p*y) + 1); for p*y > 18 approximated by -y *
    #   exp(-p*y), for p*y < -18 approximated by -y, avoiding overflow.
    # Exceptions: none. Side effects: none.
    z = p * y
    # approximately equal and saves the computation of the log
    if z > 18.0:
        return math.exp(-z) * -y
    if z < -18.0:
        return -y
    return -y / (math.exp(z) + 1.0)


def log_loss(p, y):
    # [FUNCTION DOC]
    # HELPER: Mean logistic loss over samples.
    # Inputs: p - array of decision values; y - array of labels in {-1, +1}, same shape.
    # Outputs: scalar mean of log(1 + exp(-y*p)).
    # KNOWN ISSUE: test_sag_pobj_matches_logistic_regression passes labels {0, 1} from
    #   make_blobs, so samples of class 0 contribute a constant log(2); the comparison
    #   between solvers is still consistent since all use the same formula.
    return np.mean(np.log(1.0 + np.exp(-y * p)))


# this is used for sag regression
def squared_dloss(p, y):
    # [FUNCTION DOC]
    # HELPER: Derivative of the squared loss 0.5*(p - y)^2 with respect to p.
    # Inputs: p, y - scalars (or broadcastable arrays). Outputs: p - y.
    return p - y


def squared_loss(p, y):
    # [FUNCTION DOC]
    # HELPER: Mean squared loss with the 1/2 factor.
    # Inputs: p, y - arrays of predictions and targets of the same shape.
    # Outputs: scalar mean of 0.5*(p - y)^2.
    return np.mean(0.5 * (p - y) * (p - y))


# function for measuring the log loss
def get_pobj(w, alpha, myX, myy, loss):
    # [FUNCTION DOC]
    # HELPER: Primal objective of an L2-regularized linear model without intercept.
    # Inputs: w - coefficient array (flattened before use); alpha - L2 strength; myX -
    #   2D array (n_samples, n_features); myy - targets; loss - callable(pred, y)
    #   returning the mean data loss (e.g. log_loss or squared_loss).
    # Outputs: scalar loss(X w, y) + alpha * ||w||^2 / 2.
    # Side effects: none.
    w = w.ravel()
    pred = np.dot(myX, w)
    p = loss(pred, myy)
    p += alpha * w.dot(w) / 2.0
    return p


def sag(
    X,
    y,
    step_size,
    alpha,
    n_iter=1,
    dloss=None,
    sparse=False,
    sample_weight=None,
    fit_intercept=True,
    saga=False,
):
    # [FUNCTION DOC]
    # HELPER: Naive dense reference implementation of SAG/SAGA with optional intercept,
    #   used to check the Cython solver.
    # Algorithm: start with zero weights, zero intercept, zeros for the summed gradient
    #   and a per-sample gradient memory (full feature vector per sample, which includes
    #   the regularization term alpha*w). For n_iter epochs, draw n_samples indices
    #   uniformly at random with a fixed RandomState(77); for each drawn sample compute
    #   prediction p = x.w + intercept, the scalar loss derivative (times sample_weight
    #   if given), the update x*grad + alpha*w, its correction against the memory, add
    #   it to the summed gradient and store it. In SAGA mode additionally move weights
    #   by -correction * step_size * (1 - 1/|seen|). If fit_intercept, do the analogous
    #   memory/summed-gradient bookkeeping for the intercept and step it by step_size *
    #   summed / |seen| * decay (SAGA also subtracts the scaled correction). Finally
    #   step weights by -step_size * summed_gradient / |seen|, where |seen| is the
    #   number of distinct samples seen so far.
    # Inputs: X - dense 2D array; y - targets ({-1,+1} for log loss); step_size, alpha -
    #   floats; n_iter - epochs (default 1); dloss - derivative callable (required
    #   despite default None); sparse - if True use intercept decay 0.01 instead of 1.0;
    #   sample_weight - optional per-sample weights; fit_intercept (default True); saga
    #   (default False).
    # Outputs: tuple (weights 1D array of n_features, intercept float).
    # Side effects: none outside local state; randomness is fixed via seed 77.
    # KNOWN ISSUE: in the non-SAGA intercept branch the scaled gradient_correction is
    #   computed but never used (dead computation).
    n_samples, n_features = X.shape[0], X.shape[1]

    weights = np.zeros(X.shape[1])
    sum_gradient = np.zeros(X.shape[1])
    gradient_memory = np.zeros((n_samples, n_features))

    intercept = 0.0
    intercept_sum_gradient = 0.0
    intercept_gradient_memory = np.zeros(n_samples)

    rng = np.random.RandomState(77)
    decay = 1.0
    seen = set()

    # sparse data has a fixed decay of .01
    if sparse:
        decay = 0.01

    for epoch in range(n_iter):
        for k in range(n_samples):
            idx = int(rng.rand() * n_samples)
            # idx = k
            entry = X[idx]
            seen.add(idx)
            p = np.dot(entry, weights) + intercept
            gradient = dloss(p, y[idx])
            if sample_weight is not None:
                gradient *= sample_weight[idx]
            update = entry * gradient + alpha * weights
            gradient_correction = update - gradient_memory[idx]
            sum_gradient += gradient_correction
            gradient_memory[idx] = update
            if saga:
                weights -= gradient_correction * step_size * (1 - 1.0 / len(seen))

            if fit_intercept:
                gradient_correction = gradient - intercept_gradient_memory[idx]
                intercept_gradient_memory[idx] = gradient
                intercept_sum_gradient += gradient_correction
                gradient_correction *= step_size * (1.0 - 1.0 / len(seen))
                if saga:
                    intercept -= (
                        step_size * intercept_sum_gradient / len(seen) * decay
                    ) + gradient_correction
                else:
                    intercept -= step_size * intercept_sum_gradient / len(seen) * decay

            weights -= step_size * sum_gradient / len(seen)

    return weights, intercept


def sag_sparse(
    X,
    y,
    step_size,
    alpha,
    n_iter=1,
    dloss=None,
    sample_weight=None,
    sparse=False,
    fit_intercept=True,
    saga=False,
    random_state=0,
):
    # [FUNCTION DOC]
    # HELPER: Reference implementation of SAG/SAGA with lazy ("just-in-time") weight
    #   updates, as used for sparse data in the Cython solver.
    # Algorithm: weights are stored scaled by wscale (the accumulated factor (1 -
    #   alpha*step_size) of the L2 shrinkage) and the gradient memory stores only the
    #   scalar loss derivative per sample. c_sum holds the cumulative sum of step_size /
    #   (wscale * |seen|) per iteration, and last_updated records per feature the
    #   iteration up to which the pending summed-gradient steps were applied. At each
    #   iteration (counter >= 1) all features are first brought up to date using the
    #   difference of c_sum entries, then the prediction wscale*(x.w) + intercept and
    #   the derivative (times sample_weight) are computed, the summed gradient is
    #   corrected by x*(grad - memory), SAGA applies the extra correction step divided
    #   by wscale, the intercept is updated as in `sag`, the memory is overwritten,
    #   wscale shrinks and c_sum is extended. When wscale falls below 1e-9, pending
    #   updates are flushed, the weights are rescaled by wscale and wscale is reset to
    #   1. After all epochs the remaining pending updates are applied and the weights
    #   multiplied by wscale.
    # Inputs: X - dense 2D array; y - targets; step_size, alpha - floats; n_iter -
    #   epochs (default 1); dloss - derivative callable; sample_weight - optional
    #   weights; sparse - if True intercept decay is 0.01; fit_intercept (default True);
    #   saga (default False); random_state - seed or RandomState for sample selection
    #   (default 0).
    # Outputs: tuple (weights 1D array of n_features, intercept float).
    # Exceptions: ZeroDivisionError if step_size * alpha == 1 (wscale would become 0).
    # Side effects: consumes random numbers from random_state if a RandomState is
    #   passed.
    if step_size * alpha == 1.0:
        raise ZeroDivisionError(
            "Sparse sag does not handle the case step_size * alpha == 1"
        )
    n_samples, n_features = X.shape[0], X.shape[1]

    weights = np.zeros(n_features)
    sum_gradient = np.zeros(n_features)
    last_updated = np.zeros(n_features, dtype=int)
    gradient_memory = np.zeros(n_samples)
    rng = check_random_state(random_state)
    intercept = 0.0
    intercept_sum_gradient = 0.0
    wscale = 1.0
    decay = 1.0
    seen = set()

    c_sum = np.zeros(n_iter * n_samples)

    # sparse data has a fixed decay of .01
    if sparse:
        decay = 0.01

    counter = 0
    for epoch in range(n_iter):
        for k in range(n_samples):
            # idx = k
            idx = int(rng.rand() * n_samples)
            entry = X[idx]
            seen.add(idx)

            if counter >= 1:
                for j in range(n_features):
                    if last_updated[j] == 0:
                        weights[j] -= c_sum[counter - 1] * sum_gradient[j]
                    else:
                        weights[j] -= (
                            c_sum[counter - 1] - c_sum[last_updated[j] - 1]
                        ) * sum_gradient[j]
                    last_updated[j] = counter

            p = (wscale * np.dot(entry, weights)) + intercept
            gradient = dloss(p, y[idx])

            if sample_weight is not None:
                gradient *= sample_weight[idx]

            update = entry * gradient
            gradient_correction = update - (gradient_memory[idx] * entry)
            sum_gradient += gradient_correction
            if saga:
                for j in range(n_features):
                    weights[j] -= (
                        gradient_correction[j]
                        * step_size
                        * (1 - 1.0 / len(seen))
                        / wscale
                    )

            if fit_intercept:
                gradient_correction = gradient - gradient_memory[idx]
                intercept_sum_gradient += gradient_correction
                gradient_correction *= step_size * (1.0 - 1.0 / len(seen))
                if saga:
                    intercept -= (
                        step_size * intercept_sum_gradient / len(seen) * decay
                    ) + gradient_correction
                else:
                    intercept -= step_size * intercept_sum_gradient / len(seen) * decay

            gradient_memory[idx] = gradient

            wscale *= 1.0 - alpha * step_size
            if counter == 0:
                c_sum[0] = step_size / (wscale * len(seen))
            else:
                c_sum[counter] = c_sum[counter - 1] + step_size / (wscale * len(seen))

            if counter >= 1 and wscale < 1e-9:
                for j in range(n_features):
                    if last_updated[j] == 0:
                        weights[j] -= c_sum[counter] * sum_gradient[j]
                    else:
                        weights[j] -= (
                            c_sum[counter] - c_sum[last_updated[j] - 1]
                        ) * sum_gradient[j]
                    last_updated[j] = counter + 1
                c_sum[counter] = 0
                weights *= wscale
                wscale = 1.0

            counter += 1

    for j in range(n_features):
        if last_updated[j] == 0:
            weights[j] -= c_sum[counter - 1] * sum_gradient[j]
        else:
            weights[j] -= (
                c_sum[counter - 1] - c_sum[last_updated[j] - 1]
            ) * sum_gradient[j]
    weights *= wscale
    return weights, intercept


def get_step_size(X, alpha, fit_intercept, classification=True):
    # [FUNCTION DOC]
    # HELPER: Theoretical SAG step size used by the reference implementations.
    # Inputs: X - 2D array; alpha - L2 strength; fit_intercept - bool (counts as 1 in
    #   the Lipschitz constant); classification - True for log loss, False for squared
    #   loss.
    # Outputs: 4 / (max squared row norm + fit_intercept + 4*alpha) for classification,
    #   otherwise 1 / (max squared row norm + fit_intercept + alpha).
    if classification:
        return 4.0 / (np.max(np.sum(X * X, axis=1)) + fit_intercept + 4.0 * alpha)
    else:
        return 1.0 / (np.max(np.sum(X * X, axis=1)) + fit_intercept + alpha)


def test_classifier_matching():
    # [FUNCTION DOC] Verifies that LogisticRegression with solver "sag" (80 epochs) and
    #   "saga" (300 epochs), C = 1/(alpha*n_samples), tol 1e-11, matches both reference
    #   implementations (sag_sparse and sag with log_dloss on labels mapped to {-1, +1})
    #   to 9 decimals on coefficients and intercept, on 20 well-separated two-blob
    #   samples.
    n_samples = 20
    X, y = make_blobs(n_samples=n_samples, centers=2, random_state=0, cluster_std=0.1)
    # y must be 0 or 1
    alpha = 1.1
    fit_intercept = True
    step_size = get_step_size(X, alpha, fit_intercept)
    for solver in ["sag", "saga"]:
        if solver == "sag":
            n_iter = 80
        else:
            # SAGA variance w.r.t. stream order is higher
            n_iter = 300
        clf = LogisticRegression(
            solver=solver,
            fit_intercept=fit_intercept,
            tol=1e-11,
            C=1.0 / alpha / n_samples,
            max_iter=n_iter,
            random_state=10,
        )
        clf.fit(X, y)

        weights, intercept = sag_sparse(
            X,
            2 * y - 1,  # y must be -1 or +1
            step_size,
            alpha,
            n_iter=n_iter,
            dloss=log_dloss,
            fit_intercept=fit_intercept,
            saga=solver == "saga",
        )
        weights2, intercept2 = sag(
            X,
            2 * y - 1,  # y must be -1 or +1
            step_size,
            alpha,
            n_iter=n_iter,
            dloss=log_dloss,
            fit_intercept=fit_intercept,
            saga=solver == "saga",
        )
        weights = np.atleast_2d(weights)
        intercept = np.atleast_1d(intercept)
        weights2 = np.atleast_2d(weights2)
        intercept2 = np.atleast_1d(intercept2)

        assert_array_almost_equal(weights, clf.coef_, decimal=9)
        assert_array_almost_equal(intercept, clf.intercept_, decimal=9)
        assert_array_almost_equal(weights2, clf.coef_, decimal=9)
        assert_array_almost_equal(intercept2, clf.intercept_, decimal=9)


def test_regressor_matching():
    # [FUNCTION DOC] Verifies that Ridge(solver="sag", alpha=alpha*n_samples) fitted on
    #   a noiseless linear problem (10 x 5) matches sag_sparse and sag with
    #   squared_dloss, comparing coef_ and intercept_ with assert_allclose.
    n_samples = 10
    n_features = 5

    rng = np.random.RandomState(10)
    X = rng.normal(size=(n_samples, n_features))
    true_w = rng.normal(size=n_features)
    y = X.dot(true_w)

    alpha = 1.0
    n_iter = 100
    fit_intercept = True

    step_size = get_step_size(X, alpha, fit_intercept, classification=False)
    clf = Ridge(
        fit_intercept=fit_intercept,
        tol=0.00000000001,
        solver="sag",
        alpha=alpha * n_samples,
        max_iter=n_iter,
    )
    clf.fit(X, y)

    weights1, intercept1 = sag_sparse(
        X,
        y,
        step_size,
        alpha,
        n_iter=n_iter,
        dloss=squared_dloss,
        fit_intercept=fit_intercept,
    )
    weights2, intercept2 = sag(
        X,
        y,
        step_size,
        alpha,
        n_iter=n_iter,
        dloss=squared_dloss,
        fit_intercept=fit_intercept,
    )

    assert_allclose(weights1, clf.coef_)
    assert_allclose(intercept1, clf.intercept_)
    assert_allclose(weights2, clf.coef_)
    assert_allclose(intercept2, clf.intercept_)


@pytest.mark.filterwarnings("ignore:The max_iter was reached")
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_sag_pobj_matches_logistic_regression(csr_container):
    # [FUNCTION DOC] Verifies that the primal logistic objective (get_pobj with
    #   log_loss) of LogisticRegression fitted with sag on dense input, sag on CSR
    #   input, and the default solver agree to 4 decimals (no intercept, 20 iterations).
    """tests if the sag pobj matches log reg"""
    n_samples = 100
    alpha = 1.0
    max_iter = 20
    X, y = make_blobs(n_samples=n_samples, centers=2, random_state=0, cluster_std=0.1)

    clf1 = LogisticRegression(
        solver="sag",
        fit_intercept=False,
        tol=0.0000001,
        C=1.0 / alpha / n_samples,
        max_iter=max_iter,
        random_state=10,
    )
    clf2 = clone(clf1)
    clf3 = LogisticRegression(
        fit_intercept=False,
        tol=0.0000001,
        C=1.0 / alpha / n_samples,
        max_iter=max_iter,
        random_state=10,
    )

    clf1.fit(X, y)
    clf2.fit(csr_container(X), y)
    clf3.fit(X, y)

    pobj1 = get_pobj(clf1.coef_, alpha, X, y, log_loss)
    pobj2 = get_pobj(clf2.coef_, alpha, X, y, log_loss)
    pobj3 = get_pobj(clf3.coef_, alpha, X, y, log_loss)

    assert_array_almost_equal(pobj1, pobj2, decimal=4)
    assert_array_almost_equal(pobj2, pobj3, decimal=4)
    assert_array_almost_equal(pobj3, pobj1, decimal=4)


@pytest.mark.filterwarnings("ignore:The max_iter was reached")
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_sag_pobj_matches_ridge_regression(csr_container):
    # [FUNCTION DOC] Verifies that the primal ridge objective (get_pobj with
    #   squared_loss) of Ridge fitted with sag on dense input, sag on CSR input, and
    #   lsqr agree to 4 decimals (no intercept).
    # KNOWN ISSUE: Ridge minimizes the summed (not mean) squared loss with this alpha,
    #   so the objective computed here is not the one each solver optimizes; the check
    #   still holds since the problem is noiseless and all solutions are near the same
    #   optimum.
    """tests if the sag pobj matches ridge reg"""
    n_samples = 100
    n_features = 10
    alpha = 1.0
    n_iter = 100
    fit_intercept = False
    rng = np.random.RandomState(10)
    X = rng.normal(size=(n_samples, n_features))
    true_w = rng.normal(size=n_features)
    y = X.dot(true_w)

    clf1 = Ridge(
        fit_intercept=fit_intercept,
        tol=0.00000000001,
        solver="sag",
        alpha=alpha,
        max_iter=n_iter,
        random_state=42,
    )
    clf2 = clone(clf1)
    clf3 = Ridge(
        fit_intercept=fit_intercept,
        tol=0.00001,
        solver="lsqr",
        alpha=alpha,
        max_iter=n_iter,
        random_state=42,
    )

    clf1.fit(X, y)
    clf2.fit(csr_container(X), y)
    clf3.fit(X, y)

    pobj1 = get_pobj(clf1.coef_, alpha, X, y, squared_loss)
    pobj2 = get_pobj(clf2.coef_, alpha, X, y, squared_loss)
    pobj3 = get_pobj(clf3.coef_, alpha, X, y, squared_loss)

    assert_array_almost_equal(pobj1, pobj2, decimal=4)
    assert_array_almost_equal(pobj1, pobj3, decimal=4)
    assert_array_almost_equal(pobj3, pobj2, decimal=4)


@pytest.mark.filterwarnings("ignore:The max_iter was reached")
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_sag_regressor_computed_correctly(csr_container):
    # [FUNCTION DOC] Verifies that Ridge(solver="sag") on dense data matches the
    #   sag_sparse reference (coef to 3 decimals, intercept to 1 decimal) on a linear
    #   problem with offset 2.
    # KNOWN ISSUE: clf2 (CSR fit) and the sparse=True reference results spweights2 /
    #   spintercept2 are computed but never asserted (assertions commented out pending
    #   issue #4710), so the sparse path is not checked; the comment also ends with a
    #   stray triple quote.
    """tests if the sag regressor is computed correctly"""
    alpha = 0.1
    n_features = 10
    n_samples = 40
    max_iter = 100
    tol = 0.000001
    fit_intercept = True
    rng = np.random.RandomState(0)
    X = rng.normal(size=(n_samples, n_features))
    w = rng.normal(size=n_features)
    y = np.dot(X, w) + 2.0
    step_size = get_step_size(X, alpha, fit_intercept, classification=False)

    clf1 = Ridge(
        fit_intercept=fit_intercept,
        tol=tol,
        solver="sag",
        alpha=alpha * n_samples,
        max_iter=max_iter,
        random_state=rng,
    )
    clf2 = clone(clf1)

    clf1.fit(X, y)
    clf2.fit(csr_container(X), y)

    spweights1, spintercept1 = sag_sparse(
        X,
        y,
        step_size,
        alpha,
        n_iter=max_iter,
        dloss=squared_dloss,
        fit_intercept=fit_intercept,
        random_state=rng,
    )

    spweights2, spintercept2 = sag_sparse(
        X,
        y,
        step_size,
        alpha,
        n_iter=max_iter,
        dloss=squared_dloss,
        sparse=True,
        fit_intercept=fit_intercept,
        random_state=rng,
    )

    assert_array_almost_equal(clf1.coef_.ravel(), spweights1.ravel(), decimal=3)
    assert_almost_equal(clf1.intercept_, spintercept1, decimal=1)

    # TODO: uncomment when sparse Ridge with intercept will be fixed (#4710)
    # assert_array_almost_equal(clf2.coef_.ravel(),
    #                          spweights2.ravel(),
    #                          decimal=3)
    # assert_almost_equal(clf2.intercept_, spintercept2, decimal=1)'''


def test_get_auto_step_size():
    # [FUNCTION DOC] Verifies get_auto_step_size for "squared" and "log" losses, with
    #   and without intercept, for SAG and SAGA, against hand-computed formulas (SAGA
    #   uses 1 / (2L + min(2*n_samples*alpha, L))), after checking row_norms gives the
    #   expected maximum squared row norm. Also checks a ValueError for an unknown loss
    #   name.
    X = np.array([[1, 2, 3], [2, 3, 4], [2, 3, 2]], dtype=np.float64)
    alpha = 1.2
    fit_intercept = False
    # sum the squares of the second sample because that's the largest
    max_squared_sum = 4 + 9 + 16
    max_squared_sum_ = row_norms(X, squared=True).max()
    n_samples = X.shape[0]
    assert_almost_equal(max_squared_sum, max_squared_sum_, decimal=4)

    for saga in [True, False]:
        for fit_intercept in (True, False):
            if saga:
                L_sqr = max_squared_sum + alpha + int(fit_intercept)
                L_log = (max_squared_sum + 4.0 * alpha + int(fit_intercept)) / 4.0
                mun_sqr = min(2 * n_samples * alpha, L_sqr)
                mun_log = min(2 * n_samples * alpha, L_log)
                step_size_sqr = 1 / (2 * L_sqr + mun_sqr)
                step_size_log = 1 / (2 * L_log + mun_log)
            else:
                step_size_sqr = 1.0 / (max_squared_sum + alpha + int(fit_intercept))
                step_size_log = 4.0 / (
                    max_squared_sum + 4.0 * alpha + int(fit_intercept)
                )

            step_size_sqr_ = get_auto_step_size(
                max_squared_sum_,
                alpha,
                "squared",
                fit_intercept,
                n_samples=n_samples,
                is_saga=saga,
            )
            step_size_log_ = get_auto_step_size(
                max_squared_sum_,
                alpha,
                "log",
                fit_intercept,
                n_samples=n_samples,
                is_saga=saga,
            )

            assert_almost_equal(step_size_sqr, step_size_sqr_, decimal=4)
            assert_almost_equal(step_size_log, step_size_log_, decimal=4)

    msg = "Unknown loss function for SAG solver, got wrong instead of"
    with pytest.raises(ValueError, match=msg):
        get_auto_step_size(max_squared_sum_, alpha, "wrong", fit_intercept)


@pytest.mark.parametrize("seed", range(3))  # locally tested with 1000 seeds
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_sag_regressor(seed, csr_container):
    # [FUNCTION DOC] Verifies that Ridge(solver="sag") on dense and CSR 1-feature data
    #   reaches R^2 > 0.98 on a noiseless line and > 0.45 with added standard normal
    #   noise. Parametrized over 3 seeds and CSR container types.
    """tests if the sag regressor performs well"""
    xmin, xmax = -5, 5
    n_samples = 300
    tol = 0.001
    max_iter = 100
    alpha = 0.1
    rng = np.random.RandomState(seed)
    X = np.linspace(xmin, xmax, n_samples).reshape(n_samples, 1)

    # simple linear function without noise
    y = 0.5 * X.ravel()

    clf1 = Ridge(
        tol=tol,
        solver="sag",
        max_iter=max_iter,
        alpha=alpha * n_samples,
        random_state=rng,
    )
    clf2 = clone(clf1)
    clf1.fit(X, y)
    clf2.fit(csr_container(X), y)
    score1 = clf1.score(X, y)
    score2 = clf2.score(X, y)
    assert score1 > 0.98
    assert score2 > 0.98

    # simple linear function with noise
    y = 0.5 * X.ravel() + rng.randn(n_samples, 1).ravel()

    clf1 = Ridge(
        tol=tol,
        solver="sag",
        max_iter=max_iter,
        alpha=alpha * n_samples,
        random_state=rng,
    )
    clf2 = clone(clf1)
    clf1.fit(X, y)
    clf2.fit(csr_container(X), y)
    score1 = clf1.score(X, y)
    score2 = clf2.score(X, y)
    assert score1 > 0.45
    assert score2 > 0.45


@pytest.mark.filterwarnings("ignore:The max_iter was reached")
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_sag_classifier_computed_correctly(csr_container):
    # [FUNCTION DOC] Verifies binary LogisticRegression(solver="sag") on dense and CSR
    #   input matches sag_sparse (with sparse=False and sparse=True intercept decay
    #   respectively): coef to 2 decimals and intercept to 1 decimal, labels encoded as
    #   {-1, +1}.
    """tests if the binary classifier is computed correctly"""
    alpha = 0.1
    n_samples = 50
    n_iter = 50
    tol = 0.00001
    fit_intercept = True
    X, y = make_blobs(n_samples=n_samples, centers=2, random_state=0, cluster_std=0.1)
    step_size = get_step_size(X, alpha, fit_intercept, classification=True)
    classes = np.unique(y)
    y_tmp = np.ones(n_samples)
    y_tmp[y != classes[1]] = -1
    y = y_tmp

    clf1 = LogisticRegression(
        solver="sag",
        C=1.0 / alpha / n_samples,
        max_iter=n_iter,
        tol=tol,
        random_state=77,
        fit_intercept=fit_intercept,
    )
    clf2 = clone(clf1)

    clf1.fit(X, y)
    clf2.fit(csr_container(X), y)

    spweights, spintercept = sag_sparse(
        X,
        y,
        step_size,
        alpha,
        n_iter=n_iter,
        dloss=log_dloss,
        fit_intercept=fit_intercept,
    )
    spweights2, spintercept2 = sag_sparse(
        X,
        y,
        step_size,
        alpha,
        n_iter=n_iter,
        dloss=log_dloss,
        sparse=True,
        fit_intercept=fit_intercept,
    )

    assert_array_almost_equal(clf1.coef_.ravel(), spweights.ravel(), decimal=2)
    assert_almost_equal(clf1.intercept_, spintercept, decimal=1)

    assert_array_almost_equal(clf2.coef_.ravel(), spweights2.ravel(), decimal=2)
    assert_almost_equal(clf2.intercept_, spintercept2, decimal=1)


@pytest.mark.filterwarnings("ignore:The max_iter was reached")
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_sag_multiclass_computed_correctly(csr_container):
    # [FUNCTION DOC] Verifies OneVsRestClassifier(LogisticRegression(solver="sag")) on
    #   3-class blobs, dense and CSR, matches per-class sag_sparse references (one-vs-
    #   rest {-1,+1} labels): coef with rtol 1e-2, intercept with rtol 1e-1 (dense) or
    #   5e-1 (CSR).
    """tests if the multiclass classifier is computed correctly"""
    alpha = 0.1
    n_samples = 20
    tol = 1e-5
    max_iter = 70
    fit_intercept = True
    X, y = make_blobs(n_samples=n_samples, centers=3, random_state=0, cluster_std=0.1)
    step_size = get_step_size(X, alpha, fit_intercept, classification=True)
    classes = np.unique(y)

    clf1 = OneVsRestClassifier(
        LogisticRegression(
            solver="sag",
            C=1.0 / alpha / n_samples,
            max_iter=max_iter,
            tol=tol,
            random_state=77,
            fit_intercept=fit_intercept,
        )
    )
    clf2 = clone(clf1)

    clf1.fit(X, y)
    clf2.fit(csr_container(X), y)

    coef1 = []
    intercept1 = []
    coef2 = []
    intercept2 = []
    for cl in classes:
        y_encoded = np.ones(n_samples)
        y_encoded[y != cl] = -1

        spweights1, spintercept1 = sag_sparse(
            X,
            y_encoded,
            step_size,
            alpha,
            dloss=log_dloss,
            n_iter=max_iter,
            fit_intercept=fit_intercept,
        )
        spweights2, spintercept2 = sag_sparse(
            X,
            y_encoded,
            step_size,
            alpha,
            dloss=log_dloss,
            n_iter=max_iter,
            sparse=True,
            fit_intercept=fit_intercept,
        )
        coef1.append(spweights1)
        intercept1.append(spintercept1)

        coef2.append(spweights2)
        intercept2.append(spintercept2)

    coef1 = np.vstack(coef1)
    intercept1 = np.array(intercept1)
    coef2 = np.vstack(coef2)
    intercept2 = np.array(intercept2)

    for i, cl in enumerate(classes):
        assert_allclose(clf1.estimators_[i].coef_.ravel(), coef1[i], rtol=1e-2)
        assert_allclose(clf1.estimators_[i].intercept_, intercept1[i], rtol=1e-1)

        assert_allclose(clf2.estimators_[i].coef_.ravel(), coef2[i], rtol=1e-2)
        # Note the very crude accuracy, i.e. high rtol.
        assert_allclose(clf2.estimators_[i].intercept_, intercept2[i], rtol=5e-1)


@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_classifier_results(csr_container):
    # [FUNCTION DOC] Verifies that LogisticRegression(solver="sag") on dense and CSR
    #   input perfectly predicts the training labels (signs of a random linear function,
    #   10 samples, 20 features).
    """tests if classifier results match target"""
    alpha = 0.1
    n_features = 20
    n_samples = 10
    tol = 0.01
    max_iter = 200
    rng = np.random.RandomState(0)
    X = rng.normal(size=(n_samples, n_features))
    w = rng.normal(size=n_features)
    y = np.dot(X, w)
    y = np.sign(y)
    clf1 = LogisticRegression(
        solver="sag",
        C=1.0 / alpha / n_samples,
        max_iter=max_iter,
        tol=tol,
        random_state=77,
    )
    clf2 = clone(clf1)

    clf1.fit(X, y)
    clf2.fit(csr_container(X), y)
    pred1 = clf1.predict(X)
    pred2 = clf2.predict(X)
    assert_almost_equal(pred1, y, decimal=12)
    assert_almost_equal(pred2, y, decimal=12)


@pytest.mark.filterwarnings("ignore:The max_iter was reached")
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_binary_classifier_class_weight(csr_container):
    # [FUNCTION DOC] Verifies class_weight support of binary sag LogisticRegression: the
    #   dense and CSR fits match sag_sparse run with sample weights derived from
    #   compute_class_weight (coef to 2 decimals, intercept to 1 decimal).
    """tests binary classifier with classweights for each class"""
    alpha = 0.1
    n_samples = 50
    n_iter = 20
    tol = 0.00001
    fit_intercept = True
    X, y = make_blobs(n_samples=n_samples, centers=2, random_state=10, cluster_std=0.1)
    step_size = get_step_size(X, alpha, fit_intercept, classification=True)
    classes = np.unique(y)
    y_tmp = np.ones(n_samples)
    y_tmp[y != classes[1]] = -1
    y = y_tmp

    class_weight = {1: 0.45, -1: 0.55}
    clf1 = LogisticRegression(
        solver="sag",
        C=1.0 / alpha / n_samples,
        max_iter=n_iter,
        tol=tol,
        random_state=77,
        fit_intercept=fit_intercept,
        class_weight=class_weight,
    )
    clf2 = clone(clf1)

    clf1.fit(X, y)
    clf2.fit(csr_container(X), y)

    le = LabelEncoder()
    class_weight_ = compute_class_weight(class_weight, classes=np.unique(y), y=y)
    sample_weight = class_weight_[le.fit_transform(y)]
    spweights, spintercept = sag_sparse(
        X,
        y,
        step_size,
        alpha,
        n_iter=n_iter,
        dloss=log_dloss,
        sample_weight=sample_weight,
        fit_intercept=fit_intercept,
    )
    spweights2, spintercept2 = sag_sparse(
        X,
        y,
        step_size,
        alpha,
        n_iter=n_iter,
        dloss=log_dloss,
        sparse=True,
        sample_weight=sample_weight,
        fit_intercept=fit_intercept,
    )

    assert_array_almost_equal(clf1.coef_.ravel(), spweights.ravel(), decimal=2)
    assert_almost_equal(clf1.intercept_, spintercept, decimal=1)

    assert_array_almost_equal(clf2.coef_.ravel(), spweights2.ravel(), decimal=2)
    assert_almost_equal(clf2.intercept_, spintercept2, decimal=1)


def test_classifier_single_class():
    # [FUNCTION DOC] Verifies LogisticRegression(solver="sag") raises ValueError when y
    #   contains a single class.
    """tests if ValueError is thrown with only one class"""
    X = [[1, 2], [3, 4]]
    y = [1, 1]

    msg = "This solver needs samples of at least 2 classes in the data"
    with pytest.raises(ValueError, match=msg):
        LogisticRegression(solver="sag").fit(X, y)


def test_step_size_alpha_error():
    # [FUNCTION DOC] Verifies that LogisticRegression and Ridge with sag raise
    #   ZeroDivisionError when the step size times scaled alpha equals 1 (all-zero X, no
    #   intercept, alpha 1).
    X = [[0, 0], [0, 0]]
    y = [1, -1]
    fit_intercept = False
    alpha = 1.0
    msg = re.escape(
        "Current sag implementation does not handle the case"
        " step_size * alpha_scaled == 1"
    )

    clf1 = LogisticRegression(solver="sag", C=1.0 / alpha, fit_intercept=fit_intercept)
    with pytest.raises(ZeroDivisionError, match=msg):
        clf1.fit(X, y)

    clf2 = Ridge(fit_intercept=fit_intercept, solver="sag", alpha=alpha)
    with pytest.raises(ZeroDivisionError, match=msg):
        clf2.fit(X, y)


@pytest.mark.parametrize("solver", ["sag", "saga"])
def test_sag_classifier_raises_error(solver):
    # [FUNCTION DOC] Non-regression test: after fitting with warm_start, the
    #   coefficients are set to NaN and refitting must raise ValueError about floating-
    #   point under/overflow, for both sag and saga.
    # Following #13316, the error handling behavior changed in cython sag. This
    # is simply a non-regression test to make sure numerical errors are
    # properly raised.

    # Train a classifier on a simple problem
    rng = np.random.RandomState(42)
    X, y = make_classification(random_state=rng)
    clf = LogisticRegression(solver=solver, random_state=rng, warm_start=True)
    clf.fit(X, y)

    # Trigger a numerical error by:
    # - corrupting the fitted coefficients of the classifier
    # - fit it again starting from its current state thanks to warm_start
    clf.coef_[:] = np.nan

    with pytest.raises(ValueError, match="Floating-point under-/overflow"):
        clf.fit(X, y)
