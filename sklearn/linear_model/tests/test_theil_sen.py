# [MODULE DOC]
# Tests for TheilSenRegressor and its private helpers (modified Weiszfeld step,
# spatial median, breakdown point) on synthetic data with outliers, comparing
# against least squares and known solutions.
"""
Testing for Theil-Sen module (sklearn.linear_model.theil_sen)
"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import os
import re
import sys
from contextlib import contextmanager

import numpy as np
import pytest
from numpy.testing import (
    assert_array_almost_equal,
    assert_array_equal,
    assert_array_less,
)
from scipy.linalg import norm
from scipy.optimize import fmin_bfgs

from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression, TheilSenRegressor
from sklearn.linear_model._theil_sen import (
    _breakdown_point,
    _modified_weiszfeld_step,
    _spatial_median,
)
from sklearn.utils._testing import assert_almost_equal


@contextmanager
def no_stdout_stderr():
    # [FUNCTION DOC]
    # HELPER: Context manager that redirects sys.stdout and sys.stderr to os.devnull
    # for the duration of the block and restores them afterwards.
    # Side effects: temporarily replaces global stdout/stderr.
    # KNOWN ISSUE: no try/finally, so if the block raises, stdout/stderr remain
    # redirected to a closed devnull handle.
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    with open(os.devnull, "w") as devnull:
        sys.stdout = devnull
        sys.stderr = devnull
        yield
        devnull.flush()
        sys.stdout = old_stdout
        sys.stderr = old_stderr


def gen_toy_problem_1d(intercept=True):
    # [FUNCTION DOC]
    # HELPER: Generates a seeded 1D regression problem y = 3x + c + 0.1 noise with
    # hand-placed outliers. intercept=True: 50 samples, c=2, 4 outliers;
    # intercept=False: 100 samples, c=0.1, 5 outliers.
    # Outputs: (X (n, 1), y (n,), w=3.0, c).
    random_state = np.random.RandomState(0)
    # Linear model y = 3*x + N(2, 0.1**2)
    w = 3.0
    if intercept:
        c = 2.0
        n_samples = 50
    else:
        c = 0.1
        n_samples = 100
    x = random_state.normal(size=n_samples)
    noise = 0.1 * random_state.normal(size=n_samples)
    y = w * x + c + noise
    # Add some outliers
    if intercept:
        x[42], y[42] = (-2, 4)
        x[43], y[43] = (-2.5, 8)
        x[33], y[33] = (2.5, 1)
        x[49], y[49] = (2.1, 2)
    else:
        x[42], y[42] = (-2, 4)
        x[43], y[43] = (-2.5, 8)
        x[53], y[53] = (2.5, 1)
        x[60], y[60] = (2.1, 2)
        x[72], y[72] = (1.8, -7)
    return x[:, np.newaxis], y, w, c


def gen_toy_problem_2d():
    # [FUNCTION DOC]
    # HELPER: Seeded 100-sample problem y = 5 x1 + 10 x2 + 1 + 0.1 noise where 10
    # random indices (possibly repeated) get y replaced by 50 times normal noise.
    # Outputs: (X (100, 2), y, w, c=1.0).
    random_state = np.random.RandomState(0)
    n_samples = 100
    # Linear model y = 5*x_1 + 10*x_2 + N(1, 0.1**2)
    X = random_state.normal(size=(n_samples, 2))
    w = np.array([5.0, 10.0])
    c = 1.0
    noise = 0.1 * random_state.normal(size=n_samples)
    y = np.dot(X, w) + c + noise
    # Add some outliers
    n_outliers = n_samples // 10
    ix = random_state.randint(0, n_samples, size=n_outliers)
    y[ix] = 50 * random_state.normal(size=n_outliers)
    return X, y, w, c


def gen_toy_problem_4d():
    # [FUNCTION DOC]
    # HELPER: Same as the 2D generator with 10000 samples, 4 features and w =
    # [5, 10, 42, 7]; 10% of indices drawn as outliers. Outputs: (X, y, w, c=1.0).
    random_state = np.random.RandomState(0)
    n_samples = 10000
    # Linear model y = 5*x_1 + 10*x_2  + 42*x_3 + 7*x_4 + N(1, 0.1**2)
    X = random_state.normal(size=(n_samples, 4))
    w = np.array([5.0, 10.0, 42.0, 7.0])
    c = 1.0
    noise = 0.1 * random_state.normal(size=n_samples)
    y = np.dot(X, w) + c + noise
    # Add some outliers
    n_outliers = n_samples // 10
    ix = random_state.randint(0, n_samples, size=n_outliers)
    y[ix] = 50 * random_state.normal(size=n_outliers)
    return X, y, w, c


def test_modweiszfeld_step_1d():
    # [FUNCTION DOC]
    # Verifies _modified_weiszfeld_step in 1D: the median is a fixed point, starting
    # points above the median (also one equal to a sample) move strictly toward it, and
    # with a single sample the step returns that sample.
    X = np.array([1.0, 2.0, 3.0]).reshape(3, 1)
    # Check startvalue is element of X and solution
    median = 2.0
    new_y = _modified_weiszfeld_step(X, median)
    assert_array_almost_equal(new_y, median)
    # Check startvalue is not the solution
    y = 2.5
    new_y = _modified_weiszfeld_step(X, y)
    assert_array_less(median, new_y)
    assert_array_less(new_y, y)
    # Check startvalue is not the solution but element of X
    y = 3.0
    new_y = _modified_weiszfeld_step(X, y)
    assert_array_less(median, new_y)
    assert_array_less(new_y, y)
    # Check that a single vector is identity
    X = np.array([1.0, 2.0, 3.0]).reshape(1, 3)
    y = X[0]
    new_y = _modified_weiszfeld_step(X, y)
    assert_array_equal(y, new_y)


def test_modweiszfeld_step_2d():
    # [FUNCTION DOC]
    # Verifies two Weiszfeld iterations from (0.5, 0.5) on three 2D points against
    # precomputed values, and that the known spatial median is a fixed point.
    X = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 1.0]).reshape(3, 2)
    y = np.array([0.5, 0.5])
    # Check first two iterations
    new_y = _modified_weiszfeld_step(X, y)
    assert_array_almost_equal(new_y, np.array([1 / 3, 2 / 3]))
    new_y = _modified_weiszfeld_step(X, new_y)
    assert_array_almost_equal(new_y, np.array([0.2792408, 0.7207592]))
    # Check fix point
    y = np.array([0.21132505, 0.78867497])
    new_y = _modified_weiszfeld_step(X, y)
    assert_array_almost_equal(new_y, y)


def test_spatial_median_1d():
    # [FUNCTION DOC]
    # Checks _spatial_median equals the ordinary median in 1D, on 3 points and on
    # 1000 random integers (exact equality).
    X = np.array([1.0, 2.0, 3.0]).reshape(3, 1)
    true_median = 2.0
    _, median = _spatial_median(X)
    assert_array_almost_equal(median, true_median)
    # Test larger problem and for exact solution in 1d case
    random_state = np.random.RandomState(0)
    X = random_state.randint(100, size=(1000, 1))
    true_median = np.median(X.ravel())
    _, median = _spatial_median(X)
    assert_array_equal(median, true_median)


def test_spatial_median_2d():
    # [FUNCTION DOC]
    # Checks the 2D spatial median matches a BFGS minimizer of the sum of Euclidean
    # distances, and that hitting max_iter=30 with tol 0 emits a ConvergenceWarning
    # with the expected message.
    X = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 1.0]).reshape(3, 2)
    _, median = _spatial_median(X, max_iter=100, tol=1.0e-6)

    def cost_func(y):
        # [FUNCTION DOC]
        # HELPER: Fermat-Weber objective: sum of Euclidean distances from y to each row
        # of the enclosing X. Outputs: float.
        dists = np.array([norm(x - y) for x in X])
        return np.sum(dists)

    # Check if median is solution of the Fermat-Weber location problem
    fermat_weber = fmin_bfgs(cost_func, median, disp=False)
    assert_array_almost_equal(median, fermat_weber)
    # Check when maximum iteration is exceeded a warning is emitted
    warning_message = "Maximum number of iterations 30 reached in spatial median."
    with pytest.warns(ConvergenceWarning, match=warning_message):
        _spatial_median(X, max_iter=30, tol=0.0)


def test_theil_sen_1d():
    # [FUNCTION DOC]
    # On the 1D outlier problem, asserts least squares slope is off by more than 0.9
    # while Theil-Sen recovers slope and intercept to one decimal.
    X, y, w, c = gen_toy_problem_1d()
    # Check that Least Squares fails
    lstq = LinearRegression().fit(X, y)
    assert np.abs(lstq.coef_ - w) > 0.9
    # Check that Theil-Sen works
    theil_sen = TheilSenRegressor(random_state=0).fit(X, y)
    assert_array_almost_equal(theil_sen.coef_, w, 1)
    assert_array_almost_equal(theil_sen.intercept_, c, 1)


def test_theil_sen_1d_no_intercept():
    # [FUNCTION DOC]
    # Without intercept, least squares is off from w + c by more than 0.5 while
    # Theil-Sen recovers w + c with intercept_ exactly 0; also calls score as a
    # non-regression test for issue 18104.
    X, y, w, c = gen_toy_problem_1d(intercept=False)
    # Check that Least Squares fails
    lstq = LinearRegression(fit_intercept=False).fit(X, y)
    assert np.abs(lstq.coef_ - w - c) > 0.5
    # Check that Theil-Sen works
    theil_sen = TheilSenRegressor(fit_intercept=False, random_state=0).fit(X, y)
    assert_array_almost_equal(theil_sen.coef_, w + c, 1)
    assert_almost_equal(theil_sen.intercept_, 0.0)

    # non-regression test for #18104
    theil_sen.score(X, y)


def test_theil_sen_2d():
    # [FUNCTION DOC]
    # On the 2D outlier problem with max_subpopulation=1000, least squares coef error
    # exceeds 1 while Theil-Sen recovers coef_ and intercept_ to one decimal.
    X, y, w, c = gen_toy_problem_2d()
    # Check that Least Squares fails
    lstq = LinearRegression().fit(X, y)
    assert norm(lstq.coef_ - w) > 1.0
    # Check that Theil-Sen works
    theil_sen = TheilSenRegressor(max_subpopulation=1e3, random_state=0).fit(X, y)
    assert_array_almost_equal(theil_sen.coef_, w, 1)
    assert_array_almost_equal(theil_sen.intercept_, c, 1)


def test_calc_breakdown_point():
    # [FUNCTION DOC]
    # Checks _breakdown_point for a huge sample count and 2 subsamples approaches
    # 1 - 1/sqrt(2).
    bp = _breakdown_point(1e10, 2)
    assert np.abs(bp - 1 + 1 / (np.sqrt(2))) < 1.0e-6


@pytest.mark.parametrize(
    "param, ExceptionCls, match",
    [
        (
            {"n_subsamples": 1},
            ValueError,
            re.escape("Invalid parameter since n_features+1 > n_subsamples (2 > 1)"),
        ),
        (
            {"n_subsamples": 101},
            ValueError,
            re.escape("Invalid parameter since n_subsamples > n_samples (101 > 50)"),
        ),
    ],
)
def test_checksubparams_invalid_input(param, ExceptionCls, match):
    # [FUNCTION DOC]
    # Parametrized: n_subsamples too small (below n_features + 1) or larger than
    # n_samples raises ValueError with the exact message on fit.
    X, y, w, c = gen_toy_problem_1d()
    theil_sen = TheilSenRegressor(**param, random_state=0)
    with pytest.raises(ExceptionCls, match=match):
        theil_sen.fit(X, y)


def test_checksubparams_n_subsamples_if_less_samples_than_features():
    # [FUNCTION DOC]
    # With 10 samples and 20 features, n_subsamples=9 raises ValueError on fit.
    random_state = np.random.RandomState(0)
    n_samples, n_features = 10, 20
    X = random_state.normal(size=(n_samples, n_features))
    y = random_state.normal(size=n_samples)
    theil_sen = TheilSenRegressor(n_subsamples=9, random_state=0)
    with pytest.raises(ValueError):
        theil_sen.fit(X, y)


def test_subpopulation():
    # [FUNCTION DOC]
    # On the 4D problem with max_subpopulation=250 (random subset of subsamples),
    # Theil-Sen still recovers coef_ and intercept_ to one decimal.
    X, y, w, c = gen_toy_problem_4d()
    theil_sen = TheilSenRegressor(max_subpopulation=250, random_state=0).fit(X, y)
    assert_array_almost_equal(theil_sen.coef_, w, 1)
    assert_array_almost_equal(theil_sen.intercept_, c, 1)


def test_subsamples():
    # [FUNCTION DOC]
    # With n_subsamples equal to n_samples, Theil-Sen reduces to a single least
    # squares fit; coef_ must match LinearRegression to 9 decimals.
    X, y, w, c = gen_toy_problem_4d()
    theil_sen = TheilSenRegressor(n_subsamples=X.shape[0], random_state=0).fit(X, y)
    lstq = LinearRegression().fit(X, y)
    # Check for exact the same results as Least Squares
    assert_array_almost_equal(theil_sen.coef_, lstq.coef_, 9)


@pytest.mark.thread_unsafe  # manually captured stdout
def test_verbosity():
    # [FUNCTION DOC]
    # Smoke test that verbose fitting (with and without subpopulation) runs, output
    # silenced via no_stdout_stderr.
    X, y, w, c = gen_toy_problem_1d()
    # Check that Theil-Sen can be verbose
    with no_stdout_stderr():
        TheilSenRegressor(verbose=True, random_state=0).fit(X, y)
        TheilSenRegressor(verbose=True, max_subpopulation=10, random_state=0).fit(X, y)


def test_theil_sen_parallel():
    # [FUNCTION DOC]
    # Same as the 2D test but with n_jobs=2 and max_subpopulation=2000, checking the
    # parallel path gives accurate coefficients.
    X, y, w, c = gen_toy_problem_2d()
    # Check that Least Squares fails
    lstq = LinearRegression().fit(X, y)
    assert norm(lstq.coef_ - w) > 1.0
    # Check that Theil-Sen works
    theil_sen = TheilSenRegressor(n_jobs=2, random_state=0, max_subpopulation=2e3).fit(
        X, y
    )
    assert_array_almost_equal(theil_sen.coef_, w, 1)
    assert_array_almost_equal(theil_sen.intercept_, c, 1)


def test_less_samples_than_features():
    # [FUNCTION DOC]
    # With fewer samples than features: without intercept Theil-Sen equals least
    # squares coef_ (12 decimals); with intercept predictions interpolate y exactly.
    random_state = np.random.RandomState(0)
    n_samples, n_features = 10, 20
    X = random_state.normal(size=(n_samples, n_features))
    y = random_state.normal(size=n_samples)
    # Check that Theil-Sen falls back to Least Squares if fit_intercept=False
    theil_sen = TheilSenRegressor(fit_intercept=False, random_state=0).fit(X, y)
    lstq = LinearRegression(fit_intercept=False).fit(X, y)
    assert_array_almost_equal(theil_sen.coef_, lstq.coef_, 12)
    # Check fit_intercept=True case. This will not be equal to the Least
    # Squares solution since the intercept is calculated differently.
    theil_sen = TheilSenRegressor(fit_intercept=True, random_state=0).fit(X, y)
    y_pred = theil_sen.predict(X)
    assert_array_almost_equal(y_pred, y, 12)
