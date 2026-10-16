# [MODULE DOC]
# Tests for IsolationForest (sklearn.ensemble._iforest): parameter grids on dense and
# sparse input, max_samples handling and warnings, parallel consistency, outlier
# detection quality, the _average_path_length helper, warm start, chunked prediction and
# several non-regression cases. Side effects at import: loads the iris and diabetes
# datasets into module globals shared by the tests.
"""
Testing for Isolation Forest algorithm (sklearn.ensemble.iforest).
"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import warnings
from unittest.mock import Mock, patch

import numpy as np
import pytest
from joblib import parallel_backend

from sklearn.datasets import load_diabetes, load_iris, make_classification
from sklearn.ensemble import IsolationForest
from sklearn.ensemble._iforest import _average_path_length
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import ParameterGrid, train_test_split
from sklearn.utils import check_random_state
from sklearn.utils._testing import (
    assert_allclose,
    assert_array_almost_equal,
    assert_array_equal,
    ignore_warnings,
)
from sklearn.utils.fixes import CSC_CONTAINERS, CSR_CONTAINERS

# load iris & diabetes dataset
iris = load_iris()
diabetes = load_diabetes()


def test_iforest(global_random_seed):
    # [FUNCTION DOC]
    # Smoke test: fits IsolationForest with 3 trees for every combination of max_samples
    # in (0.5, 1.0, 3) and bootstrap in (True, False) on a 2-sample toy set and predicts
    # on 2 test points, with all warnings ignored; passes if nothing raises.
    """Check Isolation Forest for various parameter settings."""
    X_train = np.array([[0, 1], [1, 2]])
    X_test = np.array([[2, 1], [1, 1]])

    grid = ParameterGrid(
        {"n_estimators": [3], "max_samples": [0.5, 1.0, 3], "bootstrap": [True, False]}
    )

    with ignore_warnings():
        for params in grid:
            IsolationForest(random_state=global_random_seed, **params).fit(
                X_train
            ).predict(X_test)


@pytest.mark.parametrize("sparse_container", CSC_CONTAINERS + CSR_CONTAINERS)
def test_iforest_sparse(global_random_seed, sparse_container):
    # [FUNCTION DOC]
    # Verifies that for each max_samples / bootstrap combination, a forest trained on a
    # sparse (CSC or CSR) version of a diabetes subset predicts exactly the same labels
    # on sparse test data as a same-seed forest trained and evaluated on the dense data.
    """Check IForest for various parameter settings on sparse input."""
    rng = check_random_state(global_random_seed)
    X_train, X_test = train_test_split(diabetes.data[:50], random_state=rng)
    grid = ParameterGrid({"max_samples": [0.5, 1.0], "bootstrap": [True, False]})

    X_train_sparse = sparse_container(X_train)
    X_test_sparse = sparse_container(X_test)

    for params in grid:
        # Trained on sparse format
        sparse_classifier = IsolationForest(
            n_estimators=10, random_state=global_random_seed, **params
        ).fit(X_train_sparse)
        sparse_results = sparse_classifier.predict(X_test_sparse)

        # Trained on dense format
        dense_classifier = IsolationForest(
            n_estimators=10, random_state=global_random_seed, **params
        ).fit(X_train)
        dense_results = dense_classifier.predict(X_test)

        assert_array_equal(sparse_results, dense_results)


def test_iforest_error():
    # [FUNCTION DOC]
    # Verifies input handling on iris (150 samples): an explicit max_samples of 1000
    # emits the "max_samples will be set to n_samples" UserWarning, while "auto" and an
    # np.int64 value emit no UserWarning (warnings turned into errors); predicting with
    # fewer features than seen in fit raises ValueError.
    """Test that it gives proper exception on deficient input."""
    X = iris.data

    # The dataset has less than 256 samples, explicitly setting
    # max_samples > n_samples should result in a warning. If not set
    # explicitly there should be no warning
    warn_msg = "max_samples will be set to n_samples for estimation"
    with pytest.warns(UserWarning, match=warn_msg):
        IsolationForest(max_samples=1000).fit(X)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        IsolationForest(max_samples="auto").fit(X)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        IsolationForest(max_samples=np.int64(2)).fit(X)

    # test X_test n_features match X_train one:
    with pytest.raises(ValueError):
        IsolationForest().fit(X).predict(X[:, 1:])


def test_recalculate_max_depth():
    # [FUNCTION DOC]
    # Verifies that with default max_samples ("auto" -> min(256, n_samples) = 150 on
    # iris) each tree's max_depth equals the ceiling of log2 of the number of samples.
    """Check max_depth recalculation when max_samples is reset to n_samples"""
    X = iris.data
    clf = IsolationForest().fit(X)
    for est in clf.estimators_:
        assert est.max_depth == int(np.ceil(np.log2(X.shape[0])))


def test_max_samples_attribute():
    # [FUNCTION DOC]
    # Verifies the fitted max_samples_ attribute: equals n_samples with the default, is
    # clipped to n_samples (with the UserWarning) when 500 is requested, and equals 0.4
    # * n_samples for a float of 0.4.
    X = iris.data
    clf = IsolationForest().fit(X)
    assert clf.max_samples_ == X.shape[0]

    clf = IsolationForest(max_samples=500)
    warn_msg = "max_samples will be set to n_samples for estimation"
    with pytest.warns(UserWarning, match=warn_msg):
        clf.fit(X)
    assert clf.max_samples_ == X.shape[0]

    clf = IsolationForest(max_samples=0.4).fit(X)
    assert clf.max_samples_ == 0.4 * X.shape[0]


def test_iforest_parallel_regression(global_random_seed):
    # [FUNCTION DOC]
    # Verifies predictions do not depend on n_jobs: a forest fit with n_jobs=3 predicts
    # the same with n_jobs set to 1 and 2 afterwards, and a same-seed forest fit with
    # n_jobs=1 gives the same predictions.
    """Check parallel regression."""
    rng = check_random_state(global_random_seed)

    X_train, X_test = train_test_split(diabetes.data, random_state=rng)

    ensemble = IsolationForest(n_jobs=3, random_state=global_random_seed).fit(X_train)

    ensemble.set_params(n_jobs=1)
    y1 = ensemble.predict(X_test)
    ensemble.set_params(n_jobs=2)
    y2 = ensemble.predict(X_test)
    assert_array_almost_equal(y1, y2)

    ensemble = IsolationForest(n_jobs=1, random_state=global_random_seed).fit(X_train)

    y3 = ensemble.predict(X_test)
    assert_array_almost_equal(y1, y3)


def test_iforest_performance(global_random_seed):
    # [FUNCTION DOC]
    # Verifies detection quality: trains on 1000 points from two Gaussian blobs, tests
    # on 200 held-out inliers plus 200 uniform outliers in [-1, 1]^2, and asserts that
    # the ROC AUC of the negated decision_function exceeds 0.98.
    # KNOWN ISSUE: the inline comment speaks of "at most 6 errors" but the assertion is
    # an AUC threshold.
    """Test Isolation Forest performs well"""

    # Generate train/test data
    rng = check_random_state(global_random_seed)
    X = 0.3 * rng.randn(600, 2)
    X = rng.permutation(np.vstack((X + 2, X - 2)))
    X_train = X[:1000]

    # Generate some abnormal novel observations
    X_outliers = rng.uniform(low=-1, high=1, size=(200, 2))
    X_test = np.vstack((X[1000:], X_outliers))
    y_test = np.array([0] * 200 + [1] * 200)

    # fit the model
    clf = IsolationForest(max_samples=100, random_state=rng).fit(X_train)

    # predict scores (the lower, the more normal)
    y_pred = -clf.decision_function(X_test)

    # check that there is at most 6 errors (false positive or false negative)
    assert roc_auc_score(y_test, y_pred) > 0.98


@pytest.mark.parametrize("contamination", [0.25, "auto"])
def test_iforest_works(contamination, global_random_seed):
    # [FUNCTION DOC]
    # Verifies that on an 8-point toy set whose last two points are outliers, for
    # contamination 0.25 and "auto", the outliers get strictly higher anomaly scores
    # than all inliers and predict returns six 1s followed by two -1s. Also reused by
    # the chunking tests below.
    # toy sample (the last two samples are outliers)
    X = [[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1], [7, 4], [-5, 9]]

    # Test IsolationForest
    clf = IsolationForest(random_state=global_random_seed, contamination=contamination)
    clf.fit(X)
    decision_func = -clf.decision_function(X)
    pred = clf.predict(X)
    # assert detect outliers:
    assert np.min(decision_func[-2:]) > np.max(decision_func[:-2])
    assert_array_equal(pred, 6 * [1] + 2 * [-1])


def test_max_samples_consistency():
    # [FUNCTION DOC]
    # Verifies the public max_samples_ equals the private _max_samples computed by the
    # BaseBagging machinery after fitting on iris.
    # Make sure validated max_samples in iforest and BaseBagging are identical
    X = iris.data
    clf = IsolationForest().fit(X)
    assert clf.max_samples_ == clf._max_samples


def test_iforest_subsampled_features():
    # [FUNCTION DOC]
    # Non-regression for #5732: fitting with max_features=0.8 (feature subsampling) and
    # then predicting must not raise.
    # It tests non-regression for #5732 which failed at predict.
    rng = check_random_state(0)
    X_train, X_test, y_train, y_test = train_test_split(
        diabetes.data[:50], diabetes.target[:50], random_state=rng
    )
    clf = IsolationForest(max_features=0.8)
    clf.fit(X_train, y_train)
    clf.predict(X_test)


def test_iforest_average_path_length():
    # [FUNCTION DOC]
    # Verifies _average_path_length against the closed form 2*(ln(n-1)+gamma) -
    # 2*(n-1)/n for n=5 and n=999, the special cases 0 and 1 giving 0 and 2 giving 1,
    # vectorized input, and that the function is non-decreasing on 0..4.
    # It tests non-regression for #8549 which used the wrong formula
    # for average path length, strictly for the integer case
    # Updated to check average path length when input is <= 2 (issue #11839)
    result_one = 2.0 * (np.log(4.0) + np.euler_gamma) - 2.0 * 4.0 / 5.0
    result_two = 2.0 * (np.log(998.0) + np.euler_gamma) - 2.0 * 998.0 / 999.0
    assert_allclose(_average_path_length([0]), [0.0])
    assert_allclose(_average_path_length([1]), [0.0])
    assert_allclose(_average_path_length([2]), [1.0])
    assert_allclose(_average_path_length([5]), [result_one])
    assert_allclose(_average_path_length([999]), [result_two])
    assert_allclose(
        _average_path_length(np.array([1, 2, 5, 999])),
        [0.0, 1.0, result_one, result_two],
    )
    # _average_path_length is increasing
    avg_path_length = _average_path_length(np.arange(5))
    assert_array_equal(avg_path_length, np.sort(avg_path_length))


def test_score_samples():
    # [FUNCTION DOC]
    # Verifies that score_samples equals decision_function plus offset_ for both an
    # explicit contamination of 0.1 and the default, and that score_samples is
    # independent of contamination.
    X_train = [[1, 1], [1, 2], [2, 1]]
    clf1 = IsolationForest(contamination=0.1).fit(X_train)
    clf2 = IsolationForest().fit(X_train)
    assert_array_equal(
        clf1.score_samples([[2.0, 2.0]]),
        clf1.decision_function([[2.0, 2.0]]) + clf1.offset_,
    )
    assert_array_equal(
        clf2.score_samples([[2.0, 2.0]]),
        clf2.decision_function([[2.0, 2.0]]) + clf2.offset_,
    )
    assert_array_equal(
        clf1.score_samples([[2.0, 2.0]]), clf2.score_samples([[2.0, 2.0]])
    )


def test_iforest_warm_start():
    # [FUNCTION DOC]
    # Verifies warm_start: refitting after raising n_estimators from 10 to 20 yields 20
    # trees and keeps the first tree object identical (not refit).
    """Test iterative addition of iTrees to an iForest"""

    rng = check_random_state(0)
    X = rng.randn(20, 2)

    # fit first 10 trees
    clf = IsolationForest(
        n_estimators=10, max_samples=20, random_state=rng, warm_start=True
    )
    clf.fit(X)
    # remember the 1st tree
    tree_1 = clf.estimators_[0]
    # fit another 10 trees
    clf.set_params(n_estimators=20)
    clf.fit(X)
    # expecting 20 fitted trees and no overwritten trees
    assert len(clf.estimators_) == 20
    assert clf.estimators_[0] is tree_1


# mock get_chunk_n_rows to actually test more than one chunk (here one
# chunk has 3 rows):
@patch(
    "sklearn.ensemble._iforest.get_chunk_n_rows",
    side_effect=Mock(**{"return_value": 3}),
)
@pytest.mark.parametrize("contamination, n_predict_calls", [(0.25, 3), ("auto", 2)])
@pytest.mark.thread_unsafe  # monkeypatched code
def test_iforest_chunks_works1(
    mocked_get_chunk, contamination, n_predict_calls, global_random_seed
):
    # [FUNCTION DOC]
    # Verifies chunked scoring: with get_chunk_n_rows patched to return 3 rows, reruns
    # test_iforest_works and checks the patched function was called n_predict_calls
    # times (3 for float contamination, 2 for "auto", reflecting how many chunked
    # scoring passes fit plus decision_function plus predict perform).
    # Inputs: mocked_get_chunk - the Mock injected by the patch decorator. Thread-unsafe
    # because it monkeypatches module code.
    test_iforest_works(contamination, global_random_seed)
    assert mocked_get_chunk.call_count == n_predict_calls


# idem with chunk_size = 10 rows
@patch(
    "sklearn.ensemble._iforest.get_chunk_n_rows",
    side_effect=Mock(**{"return_value": 10}),
)
@pytest.mark.parametrize("contamination, n_predict_calls", [(0.25, 3), ("auto", 2)])
@pytest.mark.thread_unsafe  # monkeypatched code
def test_iforest_chunks_works2(
    mocked_get_chunk, contamination, n_predict_calls, global_random_seed
):
    # [FUNCTION DOC]
    # Same as test_iforest_chunks_works1 but the patched chunk size is 10 rows (larger
    # than the data), expecting the same call counts.
    test_iforest_works(contamination, global_random_seed)
    assert mocked_get_chunk.call_count == n_predict_calls


def test_iforest_with_uniform_data():
    # [FUNCTION DOC]
    # Verifies that when training data has no variation (all ones, each column constant,
    # or a single row), every prediction on training data, random data and shifted
    # constant data is inlier (1).
    """Test whether iforest predicts inliers when using uniform data"""

    # 2-d array of all 1s
    X = np.ones((100, 10))
    iforest = IsolationForest()
    iforest.fit(X)

    rng = np.random.RandomState(0)

    assert all(iforest.predict(X) == 1)
    assert all(iforest.predict(rng.randn(100, 10)) == 1)
    assert all(iforest.predict(X + 1) == 1)
    assert all(iforest.predict(X - 1) == 1)

    # 2-d array where columns contain the same value across rows
    X = np.repeat(rng.randn(1, 10), 100, 0)
    iforest = IsolationForest()
    iforest.fit(X)

    assert all(iforest.predict(X) == 1)
    assert all(iforest.predict(rng.randn(100, 10)) == 1)
    assert all(iforest.predict(np.ones((100, 10))) == 1)

    # Single row
    X = rng.randn(1, 10)
    iforest = IsolationForest()
    iforest.fit(X)

    assert all(iforest.predict(X) == 1)
    assert all(iforest.predict(rng.randn(100, 10)) == 1)
    assert all(iforest.predict(np.ones((100, 10))) == 1)


@pytest.mark.parametrize("csc_container", CSC_CONTAINERS)
def test_iforest_with_n_jobs_does_not_segfault(csc_container):
    # [FUNCTION DOC]
    # Non-regression for #23252: fitting on a large (85000 x 100) CSC matrix with
    # n_jobs=2 and max_samples=256 completes without crashing.
    """Check that Isolation Forest does not segfault with n_jobs=2

    Non-regression test for #23252
    """
    X, _ = make_classification(n_samples=85_000, n_features=100, random_state=0)
    X = csc_container(X)
    IsolationForest(n_estimators=10, max_samples=256, n_jobs=2).fit(X)


def test_iforest_preserve_feature_names():
    # [FUNCTION DOC]
    # Non-regression for #25844: fitting on a pandas DataFrame with a float
    # contamination (which triggers internal scoring of the training data) must not emit
    # the UserWarning about missing feature names. Skipped without pandas.
    """Check that feature names are preserved when contamination is not "auto".

    Feature names are required for consistency checks during scoring.

    Non-regression test for Issue #25844
    """
    pd = pytest.importorskip("pandas")
    rng = np.random.RandomState(0)

    X = pd.DataFrame(data=rng.randn(4), columns=["a"])
    model = IsolationForest(random_state=0, contamination=0.05)

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        model.fit(X)


@pytest.mark.parametrize("sparse_container", CSC_CONTAINERS + CSR_CONTAINERS)
def test_iforest_sparse_input_float_contamination(sparse_container):
    # [FUNCTION DOC]
    # Non-regression for #27626: on sorted-index sparse input with contamination 0.1,
    # fitting works and exactly 10% of training samples get a negative
    # decision_function.
    """Check that `IsolationForest` accepts sparse matrix input and float value for
    contamination.

    Non-regression test for:
    https://github.com/scikit-learn/scikit-learn/issues/27626
    """
    X, _ = make_classification(n_samples=50, n_features=4, random_state=0)
    X = sparse_container(X)
    X.sort_indices()
    contamination = 0.1
    iforest = IsolationForest(
        n_estimators=5, contamination=contamination, random_state=0
    ).fit(X)

    X_decision = iforest.decision_function(X)
    assert (X_decision < 0).sum() / X.shape[0] == pytest.approx(contamination)


@pytest.mark.parametrize("n_jobs", [1, 2])
@pytest.mark.parametrize("contamination", [0.25, "auto"])
def test_iforest_predict_parallel(global_random_seed, contamination, n_jobs):
    # [FUNCTION DOC]
    # Verifies predict parallelization: a serial forest detects the two outliers of the
    # toy set, and a forest built with n_jobs=-1 and predicted under a threading joblib
    # backend with n_jobs workers gives identical predictions.
    """Check that `IsolationForest.predict` is parallelized."""
    # toy sample (the last two samples are outliers)
    X = [[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1], [7, 4], [-5, 9]]

    # Test IsolationForest
    clf = IsolationForest(
        random_state=global_random_seed, contamination=contamination, n_jobs=None
    )
    clf.fit(X)
    decision_func = -clf.decision_function(X)
    pred = clf.predict(X)

    # assert detect outliers:
    assert np.min(decision_func[-2:]) > np.max(decision_func[:-2])
    assert_array_equal(pred, 6 * [1] + 2 * [-1])

    clf_parallel = IsolationForest(
        random_state=global_random_seed, contamination=contamination, n_jobs=-1
    )
    clf_parallel.fit(X)
    with parallel_backend("threading", n_jobs=n_jobs):
        pred_parallel = clf_parallel.predict(X)

    # assert the same results as non-parallel
    assert_array_equal(pred, pred_parallel)
