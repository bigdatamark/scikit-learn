# [MODULE DOC]
# Tests for RANSACRegressor and its helper _dynamic_max_trials.
# At import time builds a shared corrupted line dataset: 400 points on y = 0.2x + 20
# for x in [-200, 200), then shifts a random unique subset of rows (RandomState(1000),
# indices stored in module-level outliers) by 50 plus uniform [0, 10) noise in both
# coordinates. X is the (400, 1) column of x values and y the 1D targets. Most tests
# check that RANSAC flags exactly the outlier rows.
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal

from sklearn.datasets import make_regression
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import (
    LinearRegression,
    OrthogonalMatchingPursuit,
    RANSACRegressor,
    Ridge,
)
from sklearn.linear_model._ransac import _dynamic_max_trials
from sklearn.utils import check_random_state
from sklearn.utils._testing import assert_allclose
from sklearn.utils.fixes import COO_CONTAINERS, CSC_CONTAINERS, CSR_CONTAINERS

# Generate coordinates of line
X = np.arange(-200, 200)
y = 0.2 * X + 20
data = np.column_stack([X, y])

# Add some faulty data
rng = np.random.RandomState(1000)
outliers = np.unique(rng.randint(len(X), size=200))
data[outliers, :] += 50 + rng.rand(len(outliers), 2) * 10

X = data[:, 0][:, np.newaxis]
y = data[:, 1]


def test_ransac_inliers_outliers():
    # [FUNCTION DOC]
    # Verifies RANSAC with LinearRegression, min_samples=2 and residual_threshold=5
    #   yields
    # an inlier_mask_ equal to the complement of the known outlier indices.
    estimator = LinearRegression()
    ransac_estimator = RANSACRegressor(
        estimator, min_samples=2, residual_threshold=5, random_state=0
    )

    # Estimate parameters of corrupted data
    ransac_estimator.fit(X, y)

    # Ground truth / reference inlier mask
    ref_inlier_mask = np.ones_like(ransac_estimator.inlier_mask_).astype(np.bool_)
    ref_inlier_mask[outliers] = False

    assert_array_equal(ransac_estimator.inlier_mask_, ref_inlier_mask)


def test_ransac_is_data_valid():
    # [FUNCTION DOC]
    # Verifies the is_data_valid callback receives subsets of exactly min_samples rows
    #   and
    # that fit raises ValueError when it always rejects the subset.
    def is_data_valid(X, y):
        # [FUNCTION DOC]
        # HELPER: Callback asserting X and y each have 2 rows and always returning
        #   False.
        assert X.shape[0] == 2
        assert y.shape[0] == 2
        return False

    rng = np.random.RandomState(0)
    X = rng.rand(10, 2)
    y = rng.rand(10, 1)

    estimator = LinearRegression()
    ransac_estimator = RANSACRegressor(
        estimator,
        min_samples=2,
        residual_threshold=5,
        is_data_valid=is_data_valid,
        random_state=0,
    )
    with pytest.raises(ValueError):
        ransac_estimator.fit(X, y)


def test_ransac_is_model_valid():
    # [FUNCTION DOC]
    # Verifies the is_model_valid callback receives the 2-row subset and that fit raises
    # ValueError when every model is rejected.
    def is_model_valid(estimator, X, y):
        # [FUNCTION DOC]
        # HELPER: Callback asserting X and y have 2 rows; always returns False.
        assert X.shape[0] == 2
        assert y.shape[0] == 2
        return False

    estimator = LinearRegression()
    ransac_estimator = RANSACRegressor(
        estimator,
        min_samples=2,
        residual_threshold=5,
        is_model_valid=is_model_valid,
        random_state=0,
    )
    with pytest.raises(ValueError):
        ransac_estimator.fit(X, y)


def test_ransac_max_trials():
    # [FUNCTION DOC]
    # Verifies max_trials=0 makes fit raise ValueError, and that over 50 random seeds
    #   the
    # number of trials never exceeds the dynamic bound computed by _dynamic_max_trials
    #   for
    # the true inlier count, 2-sample subsets and probability 1 - 1e-9.
    estimator = LinearRegression()

    ransac_estimator = RANSACRegressor(
        estimator,
        min_samples=2,
        residual_threshold=5,
        max_trials=0,
        random_state=0,
    )
    with pytest.raises(ValueError):
        ransac_estimator.fit(X, y)

    # there is a 1e-9 chance it will take these many trials. No good reason
    # 1e-2 isn't enough, can still happen
    # 2 is the what ransac defines  as min_samples = X.shape[1] + 1
    max_trials = _dynamic_max_trials(len(X) - len(outliers), X.shape[0], 2, 1 - 1e-9)
    ransac_estimator = RANSACRegressor(estimator, min_samples=2)
    for i in range(50):
        ransac_estimator.set_params(min_samples=2, random_state=i)
        ransac_estimator.fit(X, y)
        assert ransac_estimator.n_trials_ < max_trials + 1


def test_ransac_stop_n_inliers():
    # [FUNCTION DOC]
    # Verifies stop_n_inliers=2 stops RANSAC after the first trial (n_trials_ == 1).
    estimator = LinearRegression()
    ransac_estimator = RANSACRegressor(
        estimator,
        min_samples=2,
        residual_threshold=5,
        stop_n_inliers=2,
        random_state=0,
    )
    ransac_estimator.fit(X, y)

    assert ransac_estimator.n_trials_ == 1


def test_ransac_stop_score():
    # [FUNCTION DOC]
    # Verifies stop_score=0 stops RANSAC after the first trial (n_trials_ == 1).
    estimator = LinearRegression()
    ransac_estimator = RANSACRegressor(
        estimator,
        min_samples=2,
        residual_threshold=5,
        stop_score=0,
        random_state=0,
    )
    ransac_estimator.fit(X, y)

    assert ransac_estimator.n_trials_ == 1


def test_ransac_score():
    # [FUNCTION DOC]
    # Builds a zero-target line with two outliers at indices 0 and 1 and verifies score
    #   is
    # exactly 1 on the inliers and below 1 on the outliers.
    X = np.arange(100)[:, None]
    y = np.zeros((100,))
    y[0] = 1
    y[1] = 100

    estimator = LinearRegression()
    ransac_estimator = RANSACRegressor(
        estimator, min_samples=2, residual_threshold=0.5, random_state=0
    )
    ransac_estimator.fit(X, y)

    assert ransac_estimator.score(X[2:], y[2:]) == 1
    assert ransac_estimator.score(X[:2], y[:2]) < 1


def test_ransac_predict():
    # [FUNCTION DOC]
    # On the same zero line with 2 outliers, verifies predictions are all exactly zero.
    X = np.arange(100)[:, None]
    y = np.zeros((100,))
    y[0] = 1
    y[1] = 100

    estimator = LinearRegression()
    ransac_estimator = RANSACRegressor(
        estimator, min_samples=2, residual_threshold=0.5, random_state=0
    )
    ransac_estimator.fit(X, y)

    assert_array_equal(ransac_estimator.predict(X), np.zeros(100))


def test_ransac_no_valid_data():
    # [FUNCTION DOC]
    # Verifies that when is_data_valid always rejects, fit raises ValueError ("could not
    # find a valid consensus set") and the skip counters record 5 invalid-data skips and
    # no other skips.
    def is_data_valid(X, y):
        # [FUNCTION DOC]
        # HELPER: Callback that rejects every subset (returns False).
        return False

    estimator = LinearRegression()
    ransac_estimator = RANSACRegressor(
        estimator, is_data_valid=is_data_valid, max_trials=5
    )

    msg = "RANSAC could not find a valid consensus set"
    with pytest.raises(ValueError, match=msg):
        ransac_estimator.fit(X, y)
    assert ransac_estimator.n_skips_no_inliers_ == 0
    assert ransac_estimator.n_skips_invalid_data_ == 5
    assert ransac_estimator.n_skips_invalid_model_ == 0


def test_ransac_no_valid_model():
    # [FUNCTION DOC]
    # Verifies that when is_model_valid always rejects, fit raises ValueError and
    # n_skips_invalid_model_ is 5 with other skip counters 0.
    def is_model_valid(estimator, X, y):
        # [FUNCTION DOC]
        # HELPER: Callback that rejects every model (returns False).
        return False

    estimator = LinearRegression()
    ransac_estimator = RANSACRegressor(
        estimator, is_model_valid=is_model_valid, max_trials=5
    )

    msg = "RANSAC could not find a valid consensus set"
    with pytest.raises(ValueError, match=msg):
        ransac_estimator.fit(X, y)
    assert ransac_estimator.n_skips_no_inliers_ == 0
    assert ransac_estimator.n_skips_invalid_data_ == 0
    assert ransac_estimator.n_skips_invalid_model_ == 5


def test_ransac_exceed_max_skips():
    # [FUNCTION DOC]
    # Verifies that with max_skips=3 and always-invalid data, fit raises ValueError
    #   about
    # exceeding max_skips after 4 invalid-data skips.
    def is_data_valid(X, y):
        # [FUNCTION DOC]
        # HELPER: Callback that rejects every subset (returns False).
        return False

    estimator = LinearRegression()
    ransac_estimator = RANSACRegressor(
        estimator, is_data_valid=is_data_valid, max_trials=5, max_skips=3
    )

    msg = "RANSAC skipped more iterations than `max_skips`"
    with pytest.raises(ValueError, match=msg):
        ransac_estimator.fit(X, y)
    assert ransac_estimator.n_skips_no_inliers_ == 0
    assert ransac_estimator.n_skips_invalid_data_ == 4
    assert ransac_estimator.n_skips_invalid_model_ == 0


def test_ransac_warn_exceed_max_skips():
    # [FUNCTION DOC]
    # Verifies that when the first subset is valid but later ones are not, exceeding
    # max_skips=3 emits a ConvergenceWarning (valid consensus found but exited early)
    #   and
    # records 4 invalid-data skips.
    class IsDataValid:
        # [CLASS DOC]
        # HELPER: Stateful callable accepting only the first subset it sees.
        def __init__(self):
            # [FUNCTION DOC]
            # HELPER: Initializes call_counter to 0.
            self.call_counter = 0

        def __call__(self, X, y):
            # [FUNCTION DOC]
            # HELPER: Returns True only on the first call, then False; increments
            #   call_counter
            # (side effect) on every call. Inputs X, y are ignored.
            result = self.call_counter == 0
            self.call_counter += 1
            return result

    estimator = LinearRegression()
    ransac_estimator = RANSACRegressor(
        estimator, is_data_valid=IsDataValid(), max_skips=3, max_trials=5
    )
    warning_message = (
        "RANSAC found a valid consensus set but exited "
        "early due to skipping more iterations than "
        "`max_skips`. See estimator attributes for "
        "diagnostics."
    )
    with pytest.warns(ConvergenceWarning, match=warning_message):
        ransac_estimator.fit(X, y)
    assert ransac_estimator.n_skips_no_inliers_ == 0
    assert ransac_estimator.n_skips_invalid_data_ == 4
    assert ransac_estimator.n_skips_invalid_model_ == 0


@pytest.mark.parametrize(
    "sparse_container", COO_CONTAINERS + CSR_CONTAINERS + CSC_CONTAINERS
)
def test_ransac_sparse(sparse_container):
    # [FUNCTION DOC]
    # Verifies the correct inlier mask is found when X is given as COO, CSR or CSC.
    X_sparse = sparse_container(X)

    estimator = LinearRegression()
    ransac_estimator = RANSACRegressor(
        estimator, min_samples=2, residual_threshold=5, random_state=0
    )
    ransac_estimator.fit(X_sparse, y)

    ref_inlier_mask = np.ones_like(ransac_estimator.inlier_mask_).astype(np.bool_)
    ref_inlier_mask[outliers] = False

    assert_array_equal(ransac_estimator.inlier_mask_, ref_inlier_mask)


def test_ransac_none_estimator():
    # [FUNCTION DOC]
    # Verifies estimator=None defaults to LinearRegression by comparing predictions with
    #   an
    # explicit LinearRegression base estimator.
    estimator = LinearRegression()

    ransac_estimator = RANSACRegressor(
        estimator, min_samples=2, residual_threshold=5, random_state=0
    )
    ransac_none_estimator = RANSACRegressor(
        None, min_samples=2, residual_threshold=5, random_state=0
    )

    ransac_estimator.fit(X, y)
    ransac_none_estimator.fit(X, y)

    assert_array_almost_equal(
        ransac_estimator.predict(X), ransac_none_estimator.predict(X)
    )


def test_ransac_min_n_samples():
    # [FUNCTION DOC]
    # Verifies min_samples handling: an int 2, the equivalent fraction 2/n_samples and
    #   the
    # default (None, which becomes n_features + 1 for LinearRegression) give the same
    # predictions; min_samples greater than n_samples raises ValueError; and with a
    #   Ridge
    # base estimator min_samples=None raises ValueError requiring explicit setting
    # (GH 19390).
    # KNOWN ISSUE: ransac_estimator5 is configured identically to ransac_estimator1, so
    # that comparison is redundant.
    estimator = LinearRegression()
    ransac_estimator1 = RANSACRegressor(
        estimator, min_samples=2, residual_threshold=5, random_state=0
    )
    ransac_estimator2 = RANSACRegressor(
        estimator,
        min_samples=2.0 / X.shape[0],
        residual_threshold=5,
        random_state=0,
    )
    ransac_estimator5 = RANSACRegressor(
        estimator, min_samples=2, residual_threshold=5, random_state=0
    )
    ransac_estimator6 = RANSACRegressor(estimator, residual_threshold=5, random_state=0)
    ransac_estimator7 = RANSACRegressor(
        estimator, min_samples=X.shape[0] + 1, residual_threshold=5, random_state=0
    )
    # GH #19390
    ransac_estimator8 = RANSACRegressor(
        Ridge(), min_samples=None, residual_threshold=5, random_state=0
    )

    ransac_estimator1.fit(X, y)
    ransac_estimator2.fit(X, y)
    ransac_estimator5.fit(X, y)
    ransac_estimator6.fit(X, y)

    assert_array_almost_equal(
        ransac_estimator1.predict(X), ransac_estimator2.predict(X)
    )
    assert_array_almost_equal(
        ransac_estimator1.predict(X), ransac_estimator5.predict(X)
    )
    assert_array_almost_equal(
        ransac_estimator1.predict(X), ransac_estimator6.predict(X)
    )

    with pytest.raises(ValueError):
        ransac_estimator7.fit(X, y)

    err_msg = "`min_samples` needs to be explicitly set"
    with pytest.raises(ValueError, match=err_msg):
        ransac_estimator8.fit(X, y)


def test_ransac_multi_dimensional_targets():
    # [FUNCTION DOC]
    # Verifies the correct inlier mask is found when y is stacked into 3 identical
    #   target
    # columns.
    estimator = LinearRegression()
    ransac_estimator = RANSACRegressor(
        estimator, min_samples=2, residual_threshold=5, random_state=0
    )

    # 3-D target values
    yyy = np.column_stack([y, y, y])

    # Estimate parameters of corrupted data
    ransac_estimator.fit(X, yyy)

    # Ground truth / reference inlier mask
    ref_inlier_mask = np.ones_like(ransac_estimator.inlier_mask_).astype(np.bool_)
    ref_inlier_mask[outliers] = False

    assert_array_equal(ransac_estimator.inlier_mask_, ref_inlier_mask)


def test_ransac_residual_loss():
    # [FUNCTION DOC]
    # Verifies custom loss callables: on 3-column targets, row-wise sum of absolute or
    # squared errors give the same predictions as the default loss; on 1D targets an
    # absolute-error loss assigned after construction matches the default, and
    # loss='squared_error' is also exercised.
    # KNOWN ISSUE: after fitting ransac_estimator3 with loss='squared_error' the final
    # assertion compares ransac_estimator0 with ransac_estimator2 again, so estimator3's
    # result is never checked.
    def loss_multi1(y_true, y_pred):
        # [FUNCTION DOC]
        # HELPER: Per-sample sum over target columns of absolute errors; returns 1D
        #   array.
        return np.sum(np.abs(y_true - y_pred), axis=1)

    def loss_multi2(y_true, y_pred):
        # [FUNCTION DOC]
        # HELPER: Per-sample sum over target columns of squared errors; returns 1D
        #   array.
        return np.sum((y_true - y_pred) ** 2, axis=1)

    def loss_mono(y_true, y_pred):
        # [FUNCTION DOC]
        # HELPER: Elementwise absolute error for 1D targets.
        return np.abs(y_true - y_pred)

    yyy = np.column_stack([y, y, y])

    estimator = LinearRegression()
    ransac_estimator0 = RANSACRegressor(
        estimator, min_samples=2, residual_threshold=5, random_state=0
    )
    ransac_estimator1 = RANSACRegressor(
        estimator,
        min_samples=2,
        residual_threshold=5,
        random_state=0,
        loss=loss_multi1,
    )
    ransac_estimator2 = RANSACRegressor(
        estimator,
        min_samples=2,
        residual_threshold=5,
        random_state=0,
        loss=loss_multi2,
    )

    # multi-dimensional
    ransac_estimator0.fit(X, yyy)
    ransac_estimator1.fit(X, yyy)
    ransac_estimator2.fit(X, yyy)
    assert_array_almost_equal(
        ransac_estimator0.predict(X), ransac_estimator1.predict(X)
    )
    assert_array_almost_equal(
        ransac_estimator0.predict(X), ransac_estimator2.predict(X)
    )

    # one-dimensional
    ransac_estimator0.fit(X, y)
    ransac_estimator2.loss = loss_mono
    ransac_estimator2.fit(X, y)
    assert_array_almost_equal(
        ransac_estimator0.predict(X), ransac_estimator2.predict(X)
    )
    ransac_estimator3 = RANSACRegressor(
        estimator,
        min_samples=2,
        residual_threshold=5,
        random_state=0,
        loss="squared_error",
    )
    ransac_estimator3.fit(X, y)
    assert_array_almost_equal(
        ransac_estimator0.predict(X), ransac_estimator2.predict(X)
    )


def test_ransac_default_residual_threshold():
    # [FUNCTION DOC]
    # Verifies the default residual_threshold (median absolute deviation of y) still
    # separates the known outliers exactly.
    estimator = LinearRegression()
    ransac_estimator = RANSACRegressor(estimator, min_samples=2, random_state=0)

    # Estimate parameters of corrupted data
    ransac_estimator.fit(X, y)

    # Ground truth / reference inlier mask
    ref_inlier_mask = np.ones_like(ransac_estimator.inlier_mask_).astype(np.bool_)
    ref_inlier_mask[outliers] = False

    assert_array_equal(ransac_estimator.inlier_mask_, ref_inlier_mask)


def test_ransac_dynamic_max_trials():
    # [FUNCTION DOC]
    # Verifies _dynamic_max_trials(n_inliers, n_samples, min_samples, probability)
    #   against
    # Hartley and Zisserman Table 4.3 for several outlier ratios and subset sizes, plus
    # edge cases: probability 0 gives 0 trials and probability 1 gives infinity.
    # Numbers hand-calculated and confirmed on page 119 (Table 4.3) in
    #   Hartley, R.~I. and Zisserman, A., 2004,
    #   Multiple View Geometry in Computer Vision, Second Edition,
    #   Cambridge University Press, ISBN: 0521540518

    # e = 0%, min_samples = X
    assert _dynamic_max_trials(100, 100, 2, 0.99) == 1

    # e = 5%, min_samples = 2
    assert _dynamic_max_trials(95, 100, 2, 0.99) == 2
    # e = 10%, min_samples = 2
    assert _dynamic_max_trials(90, 100, 2, 0.99) == 3
    # e = 30%, min_samples = 2
    assert _dynamic_max_trials(70, 100, 2, 0.99) == 7
    # e = 50%, min_samples = 2
    assert _dynamic_max_trials(50, 100, 2, 0.99) == 17

    # e = 5%, min_samples = 8
    assert _dynamic_max_trials(95, 100, 8, 0.99) == 5
    # e = 10%, min_samples = 8
    assert _dynamic_max_trials(90, 100, 8, 0.99) == 9
    # e = 30%, min_samples = 8
    assert _dynamic_max_trials(70, 100, 8, 0.99) == 78
    # e = 50%, min_samples = 8
    assert _dynamic_max_trials(50, 100, 8, 0.99) == 1177

    # e = 0%, min_samples = 10
    assert _dynamic_max_trials(1, 100, 10, 0) == 0
    assert _dynamic_max_trials(1, 100, 10, 1) == float("inf")


def test_ransac_fit_sample_weight():
    # [FUNCTION DOC]
    # Verifies sample_weight support: unit weights still yield the correct inlier mask;
    # fitting with integer weights equals fitting on data where each row is repeated
    # weight times (including a weighted outlier), comparing estimator_.coef_; and a
    #   base
    # estimator without sample_weight support (OrthogonalMatchingPursuit) raises
    # ValueError.
    ransac_estimator = RANSACRegressor(random_state=0)
    n_samples = y.shape[0]
    weights = np.ones(n_samples)
    ransac_estimator.fit(X, y, sample_weight=weights)
    # sanity check
    assert ransac_estimator.inlier_mask_.shape[0] == n_samples

    ref_inlier_mask = np.ones_like(ransac_estimator.inlier_mask_).astype(np.bool_)
    ref_inlier_mask[outliers] = False
    # check that mask is correct
    assert_array_equal(ransac_estimator.inlier_mask_, ref_inlier_mask)

    # check that fit(X)  = fit([X1, X2, X3],sample_weight = [n1, n2, n3]) where
    #   X = X1 repeated n1 times, X2 repeated n2 times and so forth
    random_state = check_random_state(0)
    X_ = random_state.randint(0, 200, [10, 1])
    y_ = np.ndarray.flatten(0.2 * X_ + 2)
    sample_weight = random_state.randint(0, 10, 10)
    outlier_X = random_state.randint(0, 1000, [1, 1])
    outlier_weight = random_state.randint(0, 10, 1)
    outlier_y = random_state.randint(-1000, 0, 1)

    X_flat = np.append(
        np.repeat(X_, sample_weight, axis=0),
        np.repeat(outlier_X, outlier_weight, axis=0),
        axis=0,
    )
    y_flat = np.ndarray.flatten(
        np.append(
            np.repeat(y_, sample_weight, axis=0),
            np.repeat(outlier_y, outlier_weight, axis=0),
            axis=0,
        )
    )
    ransac_estimator.fit(X_flat, y_flat)
    ref_coef_ = ransac_estimator.estimator_.coef_

    sample_weight = np.append(sample_weight, outlier_weight)
    X_ = np.append(X_, outlier_X, axis=0)
    y_ = np.append(y_, outlier_y)
    ransac_estimator.fit(X_, y_, sample_weight=sample_weight)

    assert_allclose(ransac_estimator.estimator_.coef_, ref_coef_)

    # check that if estimator.fit doesn't support
    # sample_weight, raises error
    estimator = OrthogonalMatchingPursuit()
    ransac_estimator = RANSACRegressor(estimator, min_samples=10)

    err_msg = f"{estimator.__class__.__name__} does not support sample_weight."
    with pytest.raises(ValueError, match=err_msg):
        ransac_estimator.fit(X, y, sample_weight=weights)


def test_ransac_final_model_fit_sample_weight():
    # [FUNCTION DOC]
    # Verifies the final estimator_ equals a LinearRegression refitted on the inlier
    #   rows
    # with the corresponding normalized sample weights (coef_ atol 1e-12).
    X, y = make_regression(n_samples=1000, random_state=10)
    rng = check_random_state(42)
    sample_weight = rng.randint(1, 4, size=y.shape[0])
    sample_weight = sample_weight / sample_weight.sum()
    ransac = RANSACRegressor(random_state=0)
    ransac.fit(X, y, sample_weight=sample_weight)

    final_model = LinearRegression()
    mask_samples = ransac.inlier_mask_
    final_model.fit(
        X[mask_samples], y[mask_samples], sample_weight=sample_weight[mask_samples]
    )

    assert_allclose(ransac.estimator_.coef_, final_model.coef_, atol=1e-12)


def test_perfect_horizontal_line():
    # [FUNCTION DOC]
    # Non-regression test (issue 19497): when all samples lie on y = 0 (all inliers),
    #   RANSAC
    # fits successfully with coef_ and intercept_ equal to 0.
    """Check that we can fit a line where all samples are inliers.
    Non-regression test for:
    https://github.com/scikit-learn/scikit-learn/issues/19497
    """
    X = np.arange(100)[:, None]
    y = np.zeros((100,))

    estimator = LinearRegression()
    ransac_estimator = RANSACRegressor(estimator, random_state=0)
    ransac_estimator.fit(X, y)

    assert_allclose(ransac_estimator.estimator_.coef_, 0.0)
    assert_allclose(ransac_estimator.estimator_.intercept_, 0.0)
