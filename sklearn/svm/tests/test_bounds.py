# [MODULE DOC]
# Tests for sklearn.svm._bounds.l1_min_c and for the wrapped libsvm/liblinear
# random number generator in sklearn.svm._newrand (seeding, range limits and
# uniformity of bounded_rand_int_wrap).
import numpy as np
import pytest
from scipy import stats

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.svm._bounds import l1_min_c
from sklearn.svm._newrand import bounded_rand_int_wrap, set_seed_wrap
from sklearn.utils.fixes import CSR_CONTAINERS


@pytest.mark.parametrize("X_container", CSR_CONTAINERS + [np.array])
@pytest.mark.parametrize("loss", ["squared_hinge", "log"])
@pytest.mark.parametrize("intercept_label", ["no-intercept", "fit-intercept"])
def test_l1_min_c(X_container, loss, intercept_label):
    # [FUNCTION DOC]
    # Verifies l1_min_c on a tiny 4x2 dataset for dense and CSR inputs, both losses,
    # and with/without intercept (intercept_scaling=10), delegating the checks to
    # check_l1_min_c.
    intercepts = {
        "no-intercept": {"fit_intercept": False},
        "fit-intercept": {"fit_intercept": True, "intercept_scaling": 10},
    }

    X = X_container([[-1, 0], [0, 1], [1, 1], [1, 1]])
    Y = [0, 1, 1, 1]
    intercept_params = intercepts[intercept_label]
    check_l1_min_c(X, Y, loss, **intercept_params)


def check_l1_min_c(X, y, loss, fit_intercept=True, intercept_scaling=1.0):
    # [FUNCTION DOC]
    # HELPER: Computes min_c via l1_min_c, then fits the matching l1-penalized model
    # (liblinear LogisticRegression for "log", primal LinearSVC for
    # "squared_hinge") with the same intercept settings. Asserts that at C=min_c
    # all coefficients and intercepts are zero, and at C=1.01*min_c at least one
    # coefficient or intercept is non-zero.
    # Inputs: X, y data; loss name; fit_intercept, intercept_scaling as in l1_min_c.
    # Outputs: none; raises AssertionError on failure.
    min_c = l1_min_c(
        X,
        y,
        loss=loss,
        fit_intercept=fit_intercept,
        intercept_scaling=intercept_scaling,
    )

    clf = {
        "log": LogisticRegression(l1_ratio=1, solver="liblinear"),
        "squared_hinge": LinearSVC(loss="squared_hinge", penalty="l1", dual=False),
    }[loss]

    clf.fit_intercept = fit_intercept
    clf.intercept_scaling = intercept_scaling

    clf.C = min_c
    clf.fit(X, y)
    assert (np.asarray(clf.coef_) == 0).all()
    assert (np.asarray(clf.intercept_) == 0).all()

    clf.C = min_c * 1.01
    clf.fit(X, y)
    assert (np.asarray(clf.coef_) != 0).any() or (np.asarray(clf.intercept_) != 0).any()


def test_ill_posed_min_c():
    # [FUNCTION DOC]
    # Verifies that an all-zero X makes l1_min_c raise ValueError (den is zero).
    X = [[0, 0], [0, 0]]
    y = [0, 1]
    with pytest.raises(ValueError):
        l1_min_c(X, y)


_MAX_UNSIGNED_INT = 4294967295


def test_newrand_default():
    # [FUNCTION DOC]
    # Verifies that without explicit seeding, 10 draws of bounded_rand_int_wrap(100)
    # are all in [0, 100) and are not all identical.
    """Test that bounded_rand_int_wrap without seeding respects the range

    Note this test should pass either if executed alone, or in conjunctions
    with other tests that call set_seed explicit in any order: it checks
    invariants on the RNG instead of specific values.
    """
    generated = [bounded_rand_int_wrap(100) for _ in range(10)]
    assert all(0 <= x < 100 for x in generated)
    assert not all(x == generated[0] for x in generated)


@pytest.mark.thread_unsafe
@pytest.mark.parametrize("seed, expected", [(0, 54), (_MAX_UNSIGNED_INT, 9)])
def test_newrand_set_seed(seed, expected):
    # [FUNCTION DOC]
    # Verifies determinism: after set_seed_wrap(seed), the first draw in [0, 100)
    # equals a hard-coded expected value. Marked thread-unsafe (global RNG state).
    """Test that `set_seed` produces deterministic results"""
    set_seed_wrap(seed)
    generated = bounded_rand_int_wrap(100)
    assert generated == expected


@pytest.mark.parametrize("seed", [-1, _MAX_UNSIGNED_INT + 1])
def test_newrand_set_seed_overflow(seed):
    # [FUNCTION DOC]
    # Verifies that set_seed_wrap raises OverflowError for seeds outside the unsigned
    # 32-bit range (-1 and 2**32).
    """Test that `set_seed_wrap` is defined for unsigned 32bits ints"""
    with pytest.raises(OverflowError):
        set_seed_wrap(seed)


@pytest.mark.thread_unsafe
@pytest.mark.parametrize("range_, n_pts", [(_MAX_UNSIGNED_INT, 10000), (100, 25)])
def test_newrand_bounded_rand_int(range_, n_pts):
    # [FUNCTION DOC]
    # Verifies uniformity of bounded_rand_int_wrap: with seed 42, draws 100 samples
    # of n_pts values, runs a Kolmogorov-Smirnov test of each against a uniform
    # distribution on [0, range_), then asserts the resulting p-values are themselves
    # uniform (meta KS p-value > 0.05) and that their 10th percentile exceeds 0.05.
    # Side effects: reseeds the global wrapped RNG (marked thread-unsafe).
    """Test that `bounded_rand_int` follows a uniform distribution"""
    # XXX: this test is very seed sensitive: either it is wrong (too strict?)
    # or the wrapped RNG is not uniform enough, at least on some platforms.
    set_seed_wrap(42)
    n_iter = 100
    ks_pvals = []
    uniform_dist = stats.uniform(loc=0, scale=range_)
    # perform multiple samplings to make chance of outlier sampling negligible
    for _ in range(n_iter):
        # Deterministic random sampling
        sample = [bounded_rand_int_wrap(range_) for _ in range(n_pts)]
        res = stats.kstest(sample, uniform_dist.cdf)
        ks_pvals.append(res.pvalue)
    # Null hypothesis = samples come from a uniform distribution.
    # Under the null hypothesis, p-values should be uniformly distributed
    # and not concentrated on low values
    # (this may seem counter-intuitive but is backed by multiple refs)
    # So we can do two checks:

    # (1) check uniformity of p-values
    uniform_p_vals_dist = stats.uniform(loc=0, scale=1)
    res_pvals = stats.kstest(ks_pvals, uniform_p_vals_dist.cdf)
    assert res_pvals.pvalue > 0.05, (
        "Null hypothesis rejected: generated random numbers are not uniform."
        " Details: the (meta) p-value of the test of uniform distribution"
        f" of p-values is {res_pvals.pvalue} which is not > 0.05"
    )

    # (2) (safety belt) check that 90% of p-values are above 0.05
    min_10pct_pval = np.percentile(ks_pvals, q=10)
    # lower 10th quantile pvalue <= 0.05 means that the test rejects the
    # null hypothesis that the sample came from the uniform distribution
    assert min_10pct_pval > 0.05, (
        "Null hypothesis rejected: generated random numbers are not uniform. "
        f"Details: lower 10th quantile p-value of {min_10pct_pval} not > 0.05."
    )


@pytest.mark.parametrize("range_", [-1, _MAX_UNSIGNED_INT + 1])
def test_newrand_bounded_rand_int_limits(range_):
    # [FUNCTION DOC]
    # Verifies bounded_rand_int_wrap raises OverflowError for ranges outside the
    # unsigned 32-bit range (-1 and 2**32).
    """Test that `bounded_rand_int_wrap` is defined for unsigned 32bits ints"""
    with pytest.raises(OverflowError):
        bounded_rand_int_wrap(range_)
