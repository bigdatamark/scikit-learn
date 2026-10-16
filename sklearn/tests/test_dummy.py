# [MODULE DOC]
# Tests for DummyClassifier and DummyRegressor: every prediction strategy, single
# and multi-output targets, sparse targets, sample weights, error messages, input
# independence and output shapes/dtypes.

import warnings

import numpy as np
import pytest
import scipy.sparse as sp

from sklearn.base import clone
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.exceptions import NotFittedError
from sklearn.utils._testing import (
    assert_almost_equal,
    assert_array_almost_equal,
    assert_array_equal,
)
from sklearn.utils.fixes import CSC_CONTAINERS
from sklearn.utils.stats import _weighted_percentile


def _check_predict_proba(clf, X, y):
    # [FUNCTION DOC]
    # HELPER: Asserts predict_proba/predict_log_proba consistency for a fitted
    # classifier. Inputs: clf - fitted classifier; X - samples; y - 1D or 2D targets
    # used to count classes per output. For each output, checks shape is
    # (n_samples, n_unique_classes), rows sum to 1, and log of proba equals
    # log_proba (divide-by-zero warnings suppressed). Single output results are wrapped
    # in a list for uniform handling.
    proba = clf.predict_proba(X)

    # We know that we can have division by zero
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "divide by zero encountered in log")
        log_proba = clf.predict_log_proba(X)

    y = np.atleast_1d(y)
    if y.ndim == 1:
        y = np.reshape(y, (-1, 1))

    n_outputs = y.shape[1]
    n_samples = len(X)

    if n_outputs == 1:
        proba = [proba]
        log_proba = [log_proba]

    for k in range(n_outputs):
        assert proba[k].shape[0] == n_samples
        assert proba[k].shape[1] == len(np.unique(y[:, k]))
        assert_array_almost_equal(proba[k].sum(axis=1), np.ones(len(X)))
        # We know that we can have division by zero
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "divide by zero encountered in log")
            assert_array_almost_equal(np.log(proba[k]), log_proba[k])


def _check_behavior_2d(clf):
    # [FUNCTION DOC]
    # HELPER: Clones clf and fits on a 4-sample dataset with 1D then 2D (two-column)
    # integer targets, asserting the prediction shape matches the target shape.
    # 1d case
    X = np.array([[0], [0], [0], [0]])  # ignored
    y = np.array([1, 2, 1, 1])
    est = clone(clf)
    est.fit(X, y)
    y_pred = est.predict(X)
    assert y.shape == y_pred.shape

    # 2d case
    y = np.array([[1, 0], [2, 0], [1, 0], [1, 3]])
    est = clone(clf)
    est.fit(X, y)
    y_pred = est.predict(X)
    assert y.shape == y_pred.shape


def _check_behavior_2d_for_constant(clf):
    # [FUNCTION DOC]
    # HELPER: Like _check_behavior_2d but only for a 5-output 2D target (for the
    # constant strategy whose constant must match the number of outputs).
    # 2d case only
    X = np.array([[0], [0], [0], [0]])  # ignored
    y = np.array([[1, 0, 5, 4, 3], [2, 0, 1, 2, 5], [1, 0, 4, 5, 2], [1, 3, 3, 2, 0]])
    est = clone(clf)
    est.fit(X, y)
    y_pred = est.predict(X)
    assert y.shape == y_pred.shape


def _check_equality_regressor(statistic, y_learn, y_pred_learn, y_test, y_pred_test):
    # [FUNCTION DOC]
    # HELPER: Asserts that train and test predictions both equal the given statistic
    # (1D or row vector) tiled over the respective number of samples.
    assert_array_almost_equal(np.tile(statistic, (y_learn.shape[0], 1)), y_pred_learn)
    assert_array_almost_equal(np.tile(statistic, (y_test.shape[0], 1)), y_pred_test)


def test_feature_names_in_and_n_features_in_(global_random_seed, n_samples=10):
    # [FUNCTION DOC]
    # Fits both dummy estimators on a one-column pandas DataFrame and checks
    # feature_names_in_ and n_features_in_ are set. Skipped without pandas.
    pd = pytest.importorskip("pandas")

    random_state = np.random.RandomState(seed=global_random_seed)

    X = pd.DataFrame([[0]] * n_samples, columns=["feature_1"])
    y = random_state.rand(n_samples)

    est = DummyRegressor().fit(X, y)
    assert hasattr(est, "feature_names_in_")
    assert hasattr(est, "n_features_in_")

    est = DummyClassifier().fit(X, y)
    assert hasattr(est, "feature_names_in_")
    assert hasattr(est, "n_features_in_")


def test_most_frequent_and_prior_strategy():
    # [FUNCTION DOC]
    # For "most_frequent" and "prior", checks predictions are the majority class, proba
    # consistency, and that predict_proba equals class_prior_ ("prior") or its one-hot
    # of the majority ("most_frequent").
    X = [[0], [0], [0], [0]]  # ignored
    y = [1, 2, 1, 1]

    for strategy in ("most_frequent", "prior"):
        clf = DummyClassifier(strategy=strategy, random_state=0)
        clf.fit(X, y)
        assert_array_equal(clf.predict(X), np.ones(len(X)))
        _check_predict_proba(clf, X, y)

        if strategy == "prior":
            assert_array_almost_equal(
                clf.predict_proba([X[0]]), clf.class_prior_.reshape((1, -1))
            )
        else:
            assert_array_almost_equal(
                clf.predict_proba([X[0]]), clf.class_prior_.reshape((1, -1)) > 0.5
            )


def test_most_frequent_and_prior_strategy_with_2d_column_y():
    # [FUNCTION DOC]
    # Non-regression: a single-column 2D y gives the same predictions as the
    # equivalent 1D y.
    # non-regression test added in
    # https://github.com/scikit-learn/scikit-learn/pull/13545
    X = [[0], [0], [0], [0]]
    y_1d = [1, 2, 1, 1]
    y_2d = [[1], [2], [1], [1]]

    for strategy in ("most_frequent", "prior"):
        clf_1d = DummyClassifier(strategy=strategy, random_state=0)
        clf_2d = DummyClassifier(strategy=strategy, random_state=0)

        clf_1d.fit(X, y_1d)
        clf_2d.fit(X, y_2d)
        assert_array_equal(clf_1d.predict(X), clf_2d.predict(X))


def test_most_frequent_and_prior_strategy_multioutput():
    # [FUNCTION DOC]
    # Checks per-output majority predictions on a 2-output target, proba consistency,
    # and 1D/2D shape behavior.
    X = [[0], [0], [0], [0]]  # ignored
    y = np.array([[1, 0], [2, 0], [1, 0], [1, 3]])

    n_samples = len(X)

    for strategy in ("prior", "most_frequent"):
        clf = DummyClassifier(strategy=strategy, random_state=0)
        clf.fit(X, y)
        assert_array_equal(
            clf.predict(X),
            np.hstack([np.ones((n_samples, 1)), np.zeros((n_samples, 1))]),
        )
        _check_predict_proba(clf, X, y)
        _check_behavior_2d(clf)


def test_stratified_strategy(global_random_seed):
    # [FUNCTION DOC]
    # Checks that "stratified" predictions on 500 samples follow the training class
    # frequencies (3/5 and 2/5) to one decimal, plus proba consistency.
    X = [[0]] * 5  # ignored
    y = [1, 2, 1, 1, 2]
    clf = DummyClassifier(strategy="stratified", random_state=global_random_seed)
    clf.fit(X, y)

    X = [[0]] * 500
    y_pred = clf.predict(X)
    p = np.bincount(y_pred) / float(len(X))
    assert_almost_equal(p[1], 3.0 / 5, decimal=1)
    assert_almost_equal(p[2], 2.0 / 5, decimal=1)
    _check_predict_proba(clf, X, y)


def test_stratified_strategy_multioutput(global_random_seed):
    # [FUNCTION DOC]
    # Same as test_stratified_strategy for each output of a 2-output target, plus
    # shape behavior.
    X = [[0]] * 5  # ignored
    y = np.array([[2, 1], [2, 2], [1, 1], [1, 2], [1, 1]])

    clf = DummyClassifier(strategy="stratified", random_state=global_random_seed)
    clf.fit(X, y)

    X = [[0]] * 500
    y_pred = clf.predict(X)

    for k in range(y.shape[1]):
        p = np.bincount(y_pred[:, k]) / float(len(X))
        assert_almost_equal(p[1], 3.0 / 5, decimal=1)
        assert_almost_equal(p[2], 2.0 / 5, decimal=1)
        _check_predict_proba(clf, X, y)

    _check_behavior_2d(clf)


def test_uniform_strategy(global_random_seed):
    # [FUNCTION DOC]
    # Checks "uniform" predictions are approximately equiprobable across the two
    # classes, plus proba consistency.
    X = [[0]] * 4  # ignored
    y = [1, 2, 1, 1]
    clf = DummyClassifier(strategy="uniform", random_state=global_random_seed)
    clf.fit(X, y)

    X = [[0]] * 500
    y_pred = clf.predict(X)
    p = np.bincount(y_pred) / float(len(X))
    assert_almost_equal(p[1], 0.5, decimal=1)
    assert_almost_equal(p[2], 0.5, decimal=1)
    _check_predict_proba(clf, X, y)


def test_uniform_strategy_multioutput(global_random_seed):
    # [FUNCTION DOC]
    # Same as test_uniform_strategy for each output of a 2-output target.
    X = [[0]] * 4  # ignored
    y = np.array([[2, 1], [2, 2], [1, 2], [1, 1]])
    clf = DummyClassifier(strategy="uniform", random_state=global_random_seed)
    clf.fit(X, y)

    X = [[0]] * 500
    y_pred = clf.predict(X)

    for k in range(y.shape[1]):
        p = np.bincount(y_pred[:, k]) / float(len(X))
        assert_almost_equal(p[1], 0.5, decimal=1)
        assert_almost_equal(p[2], 0.5, decimal=1)
        _check_predict_proba(clf, X, y)

    _check_behavior_2d(clf)


def test_string_labels():
    # [FUNCTION DOC]
    # Checks "most_frequent" works with string class labels.
    X = [[0]] * 5
    y = ["paris", "paris", "tokyo", "amsterdam", "berlin"]
    clf = DummyClassifier(strategy="most_frequent")
    clf.fit(X, y)
    assert_array_equal(clf.predict(X), ["paris"] * 5)


@pytest.mark.parametrize(
    "y,y_test",
    [
        ([2, 1, 1, 1], [2, 2, 1, 1]),
        (
            np.array([[2, 2], [1, 1], [1, 1], [1, 1]]),
            np.array([[2, 2], [2, 2], [1, 1], [1, 1]]),
        ),
    ],
)
def test_classifier_score_with_None(y, y_test):
    # [FUNCTION DOC]
    # Checks DummyClassifier accepts X=None in fit and score, giving accuracy 0.5 on
    # the given 1D and 2D targets.
    clf = DummyClassifier(strategy="most_frequent")
    clf.fit(None, y)
    assert clf.score(None, y_test) == 0.5


@pytest.mark.parametrize(
    "strategy", ["stratified", "most_frequent", "prior", "uniform", "constant"]
)
def test_classifier_prediction_independent_of_X(strategy, global_random_seed):
    # [FUNCTION DOC]
    # For every strategy with the same seed, predictions do not depend on X values.
    y = [0, 2, 1, 1]
    X1 = [[0]] * 4
    clf1 = DummyClassifier(
        strategy=strategy, random_state=global_random_seed, constant=0
    )
    clf1.fit(X1, y)
    predictions1 = clf1.predict(X1)

    X2 = [[1]] * 4
    clf2 = DummyClassifier(
        strategy=strategy, random_state=global_random_seed, constant=0
    )
    clf2.fit(X2, y)
    predictions2 = clf2.predict(X2)

    assert_array_equal(predictions1, predictions2)


def test_mean_strategy_regressor(global_random_seed):
    # [FUNCTION DOC]
    # Checks the default "mean" strategy predicts the mean of y for each sample.
    random_state = np.random.RandomState(seed=global_random_seed)

    X = [[0]] * 4  # ignored
    y = random_state.randn(4)

    reg = DummyRegressor()
    reg.fit(X, y)
    assert_array_equal(reg.predict(X), [np.mean(y)] * len(X))


def test_mean_strategy_multioutput_regressor(global_random_seed):
    # [FUNCTION DOC]
    # Checks per-output means on a 5-output target for train and test, plus shapes.
    random_state = np.random.RandomState(seed=global_random_seed)

    X_learn = random_state.randn(10, 10)
    y_learn = random_state.randn(10, 5)

    mean = np.mean(y_learn, axis=0).reshape((1, -1))

    X_test = random_state.randn(20, 10)
    y_test = random_state.randn(20, 5)

    # Correctness oracle
    est = DummyRegressor()
    est.fit(X_learn, y_learn)
    y_pred_learn = est.predict(X_learn)
    y_pred_test = est.predict(X_test)

    _check_equality_regressor(mean, y_learn, y_pred_learn, y_test, y_pred_test)
    _check_behavior_2d(est)


def test_regressor_exceptions():
    # [FUNCTION DOC]
    # Checks predict on an unfitted DummyRegressor raises NotFittedError.
    reg = DummyRegressor()
    with pytest.raises(NotFittedError):
        reg.predict([])


def test_median_strategy_regressor(global_random_seed):
    # [FUNCTION DOC]
    # Checks "median" predicts the median of y.
    random_state = np.random.RandomState(seed=global_random_seed)

    X = [[0]] * 5  # ignored
    y = random_state.randn(5)

    reg = DummyRegressor(strategy="median")
    reg.fit(X, y)
    assert_array_equal(reg.predict(X), [np.median(y)] * len(X))


def test_median_strategy_multioutput_regressor(global_random_seed):
    # [FUNCTION DOC]
    # Checks per-output medians on a 5-output target, plus shapes.
    random_state = np.random.RandomState(seed=global_random_seed)

    X_learn = random_state.randn(10, 10)
    y_learn = random_state.randn(10, 5)

    median = np.median(y_learn, axis=0).reshape((1, -1))

    X_test = random_state.randn(20, 10)
    y_test = random_state.randn(20, 5)

    # Correctness oracle
    est = DummyRegressor(strategy="median")
    est.fit(X_learn, y_learn)
    y_pred_learn = est.predict(X_learn)
    y_pred_test = est.predict(X_test)

    _check_equality_regressor(median, y_learn, y_pred_learn, y_test, y_pred_test)
    _check_behavior_2d(est)


def test_quantile_strategy_regressor(global_random_seed):
    # [FUNCTION DOC]
    # Checks "quantile" with quantile 0.5, 0, 1 and 0.3 predicts the median, min, max
    # and 30th percentile of y.
    random_state = np.random.RandomState(seed=global_random_seed)

    X = [[0]] * 5  # ignored
    y = random_state.randn(5)

    reg = DummyRegressor(strategy="quantile", quantile=0.5)
    reg.fit(X, y)
    assert_array_equal(reg.predict(X), [np.median(y)] * len(X))

    reg = DummyRegressor(strategy="quantile", quantile=0)
    reg.fit(X, y)
    assert_array_equal(reg.predict(X), [np.min(y)] * len(X))

    reg = DummyRegressor(strategy="quantile", quantile=1)
    reg.fit(X, y)
    assert_array_equal(reg.predict(X), [np.max(y)] * len(X))

    reg = DummyRegressor(strategy="quantile", quantile=0.3)
    reg.fit(X, y)
    assert_array_equal(reg.predict(X), [np.percentile(y, q=30)] * len(X))


def test_quantile_strategy_multioutput_regressor(global_random_seed):
    # [FUNCTION DOC]
    # Checks per-output 50th and 80th percentiles on a 5-output target, plus shapes.
    random_state = np.random.RandomState(seed=global_random_seed)

    X_learn = random_state.randn(10, 10)
    y_learn = random_state.randn(10, 5)

    median = np.median(y_learn, axis=0).reshape((1, -1))
    quantile_values = np.percentile(y_learn, axis=0, q=80).reshape((1, -1))

    X_test = random_state.randn(20, 10)
    y_test = random_state.randn(20, 5)

    # Correctness oracle
    est = DummyRegressor(strategy="quantile", quantile=0.5)
    est.fit(X_learn, y_learn)
    y_pred_learn = est.predict(X_learn)
    y_pred_test = est.predict(X_test)

    _check_equality_regressor(median, y_learn, y_pred_learn, y_test, y_pred_test)
    _check_behavior_2d(est)

    # Correctness oracle
    est = DummyRegressor(strategy="quantile", quantile=0.8)
    est.fit(X_learn, y_learn)
    y_pred_learn = est.predict(X_learn)
    y_pred_test = est.predict(X_test)

    _check_equality_regressor(
        quantile_values, y_learn, y_pred_learn, y_test, y_pred_test
    )
    _check_behavior_2d(est)


def test_quantile_invalid():
    # [FUNCTION DOC]
    # Checks "quantile" with quantile None raises ValueError at fit.
    X = [[0]] * 5  # ignored
    y = [0] * 5  # ignored

    est = DummyRegressor(strategy="quantile", quantile=None)
    err_msg = (
        "When using `strategy='quantile', you have to specify the desired quantile"
    )
    with pytest.raises(ValueError, match=err_msg):
        est.fit(X, y)


def test_quantile_strategy_empty_train():
    # [FUNCTION DOC]
    # Checks fitting "quantile" on empty data raises IndexError.
    est = DummyRegressor(strategy="quantile", quantile=0.4)
    with pytest.raises(IndexError):
        est.fit([], [])


def test_constant_strategy_regressor(global_random_seed):
    # [FUNCTION DOC]
    # Checks "constant" with a list or scalar constant predicts it, and that the
    # constant parameter is not converted to an ndarray (non-regression #22478).
    random_state = np.random.RandomState(seed=global_random_seed)

    X = [[0]] * 5  # ignored
    y = random_state.randn(5)

    reg = DummyRegressor(strategy="constant", constant=[43])
    reg.fit(X, y)
    assert_array_equal(reg.predict(X), [43] * len(X))

    reg = DummyRegressor(strategy="constant", constant=43)
    reg.fit(X, y)
    assert_array_equal(reg.predict(X), [43] * len(X))

    # non-regression test for #22478
    assert not isinstance(reg.constant, np.ndarray)


def test_constant_strategy_multioutput_regressor(global_random_seed):
    # [FUNCTION DOC]
    # Checks a per-output constant vector is predicted for train and test, plus shapes
    # for a 5-output target.
    random_state = np.random.RandomState(seed=global_random_seed)

    X_learn = random_state.randn(10, 10)
    y_learn = random_state.randn(10, 5)

    # test with 2d array
    constants = random_state.randn(5)

    X_test = random_state.randn(20, 10)
    y_test = random_state.randn(20, 5)

    # Correctness oracle
    est = DummyRegressor(strategy="constant", constant=constants)
    est.fit(X_learn, y_learn)
    y_pred_learn = est.predict(X_learn)
    y_pred_test = est.predict(X_test)

    _check_equality_regressor(constants, y_learn, y_pred_learn, y_test, y_pred_test)
    _check_behavior_2d_for_constant(est)


def test_y_mean_attribute_regressor():
    # [FUNCTION DOC]
    # Checks constant_ equals mean(y) after fitting with "mean".
    X = [[0]] * 5
    y = [1, 2, 4, 6, 8]
    # when strategy = 'mean'
    est = DummyRegressor(strategy="mean")
    est.fit(X, y)

    assert est.constant_ == np.mean(y)


def test_constants_not_specified_regressor():
    # [FUNCTION DOC]
    # Checks "constant" without constant raises TypeError at fit.
    X = [[0]] * 5
    y = [1, 2, 4, 6, 8]

    est = DummyRegressor(strategy="constant")
    err_msg = "Constant target value has to be specified"
    with pytest.raises(TypeError, match=err_msg):
        est.fit(X, y)


def test_constant_size_multioutput_regressor(global_random_seed):
    # [FUNCTION DOC]
    # Checks a constant of length 4 for a 5-output target raises ValueError about
    # expected shape (5, 1).
    random_state = np.random.RandomState(seed=global_random_seed)
    X = random_state.randn(10, 10)
    y = random_state.randn(10, 5)

    est = DummyRegressor(strategy="constant", constant=[1, 2, 3, 4])
    err_msg = r"Constant target value should have shape \(5, 1\)."
    with pytest.raises(ValueError, match=err_msg):
        est.fit(X, y)


def test_constant_strategy():
    # [FUNCTION DOC]
    # Checks DummyClassifier "constant" with integer and string constants predicts the
    # constant, plus proba consistency.
    X = [[0], [0], [0], [0]]  # ignored
    y = [2, 1, 2, 2]

    clf = DummyClassifier(strategy="constant", random_state=0, constant=1)
    clf.fit(X, y)
    assert_array_equal(clf.predict(X), np.ones(len(X)))
    _check_predict_proba(clf, X, y)

    X = [[0], [0], [0], [0]]  # ignored
    y = ["two", "one", "two", "two"]
    clf = DummyClassifier(strategy="constant", random_state=0, constant="one")
    clf.fit(X, y)
    assert_array_equal(clf.predict(X), np.array(["one"] * 4))
    _check_predict_proba(clf, X, y)


def test_constant_strategy_multioutput():
    # [FUNCTION DOC]
    # Checks per-output constant [1, 0] on a 2-output target.
    X = [[0], [0], [0], [0]]  # ignored
    y = np.array([[2, 3], [1, 3], [2, 3], [2, 0]])

    n_samples = len(X)

    clf = DummyClassifier(strategy="constant", random_state=0, constant=[1, 0])
    clf.fit(X, y)
    assert_array_equal(
        clf.predict(X), np.hstack([np.ones((n_samples, 1)), np.zeros((n_samples, 1))])
    )
    _check_predict_proba(clf, X, y)


@pytest.mark.parametrize(
    "y, params, err_msg",
    [
        ([2, 1, 2, 2], {"random_state": 0}, "Constant.*has to be specified"),
        ([2, 1, 2, 2], {"constant": [2, 0]}, "Constant.*should have shape"),
        (
            np.transpose([[2, 1, 2, 2], [2, 1, 2, 2]]),
            {"constant": 2},
            "Constant.*should have shape",
        ),
        (
            [2, 1, 2, 2],
            {"constant": "my-constant"},
            "constant=my-constant.*Possible values.*\\[1, 2]",
        ),
        (
            np.transpose([[2, 1, 2, 2], [2, 1, 2, 2]]),
            {"constant": [2, "unknown"]},
            "constant=\\[2, 'unknown'].*Possible values.*\\[1, 2]",
        ),
    ],
    ids=[
        "no-constant",
        "too-many-constant",
        "not-enough-output",
        "single-output",
        "multi-output",
    ],
)
def test_constant_strategy_exceptions(y, params, err_msg):
    # [FUNCTION DOC]
    # Parametrized invalid constant configurations (missing, wrong length, not a known
    # class) must raise ValueError with the given message at fit.
    X = [[0], [0], [0], [0]]

    clf = DummyClassifier(strategy="constant", **params)
    with pytest.raises(ValueError, match=err_msg):
        clf.fit(X, y)


def test_classification_sample_weight():
    # [FUNCTION DOC]
    # Checks class_prior_ is computed from sample weights.
    X = [[0], [0], [1]]
    y = [0, 1, 0]
    sample_weight = [0.1, 1.0, 0.1]

    clf = DummyClassifier(strategy="stratified").fit(X, y, sample_weight)
    assert_array_almost_equal(clf.class_prior_, [0.2 / 1.2, 1.0 / 1.2])


@pytest.mark.parametrize("csc_container", CSC_CONTAINERS)
def test_constant_strategy_sparse_target(csc_container):
    # [FUNCTION DOC]
    # Checks "constant" on a sparse CSC target returns sparse predictions with the
    # expected values.
    X = [[0]] * 5  # ignored
    y = csc_container(np.array([[0, 1], [4, 0], [1, 1], [1, 4], [1, 1]]))

    n_samples = len(X)

    clf = DummyClassifier(strategy="constant", random_state=0, constant=[1, 0])
    clf.fit(X, y)
    y_pred = clf.predict(X)
    assert sp.issparse(y_pred)
    assert_array_equal(
        y_pred.toarray(), np.hstack([np.ones((n_samples, 1)), np.zeros((n_samples, 1))])
    )


@pytest.mark.parametrize("csc_container", CSC_CONTAINERS)
def test_uniform_strategy_sparse_target_warning(global_random_seed, csc_container):
    # [FUNCTION DOC]
    # Checks "uniform" with sparse target warns that it would not save memory, and
    # predictions per output are roughly uniform over the three classes.
    X = [[0]] * 5  # ignored
    y = csc_container(np.array([[2, 1], [2, 2], [1, 4], [4, 2], [1, 1]]))

    clf = DummyClassifier(strategy="uniform", random_state=global_random_seed)
    with pytest.warns(UserWarning, match="the uniform strategy would not save memory"):
        clf.fit(X, y)

    X = [[0]] * 500
    y_pred = clf.predict(X)

    for k in range(y.shape[1]):
        p = np.bincount(y_pred[:, k]) / float(len(X))
        assert_almost_equal(p[1], 1 / 3, decimal=1)
        assert_almost_equal(p[2], 1 / 3, decimal=1)
        assert_almost_equal(p[4], 1 / 3, decimal=1)


@pytest.mark.parametrize("csc_container", CSC_CONTAINERS)
def test_stratified_strategy_sparse_target(global_random_seed, csc_container):
    # [FUNCTION DOC]
    # Checks "stratified" with sparse target returns sparse predictions following
    # class frequencies per output.
    X = [[0]] * 5  # ignored
    y = csc_container(np.array([[4, 1], [0, 0], [1, 1], [1, 4], [1, 1]]))

    clf = DummyClassifier(strategy="stratified", random_state=global_random_seed)
    clf.fit(X, y)

    X = [[0]] * 500
    y_pred = clf.predict(X)
    assert sp.issparse(y_pred)
    y_pred = y_pred.toarray()

    for k in range(y.shape[1]):
        p = np.bincount(y_pred[:, k]) / float(len(X))
        assert_almost_equal(p[1], 3.0 / 5, decimal=1)
        assert_almost_equal(p[0], 1.0 / 5, decimal=1)
        assert_almost_equal(p[4], 1.0 / 5, decimal=1)


@pytest.mark.parametrize("csc_container", CSC_CONTAINERS)
def test_most_frequent_and_prior_strategy_sparse_target(csc_container):
    # [FUNCTION DOC]
    # Checks "most_frequent" and "prior" with sparse target return sparse majority
    # predictions.
    X = [[0]] * 5  # ignored
    y = csc_container(np.array([[1, 0], [1, 3], [4, 0], [0, 1], [1, 0]]))

    n_samples = len(X)
    y_expected = np.hstack([np.ones((n_samples, 1)), np.zeros((n_samples, 1))])
    for strategy in ("most_frequent", "prior"):
        clf = DummyClassifier(strategy=strategy, random_state=0)
        clf.fit(X, y)

        y_pred = clf.predict(X)
        assert sp.issparse(y_pred)
        assert_array_equal(y_pred.toarray(), y_expected)


def test_dummy_regressor_sample_weight(global_random_seed, n_samples=10):
    # [FUNCTION DOC]
    # Checks constant_ equals the weighted mean, weighted median and weighted 95th
    # percentile (via _weighted_percentile) for the respective strategies.
    random_state = np.random.RandomState(seed=global_random_seed)

    X = [[0]] * n_samples
    y = random_state.rand(n_samples)
    sample_weight = random_state.rand(n_samples)

    est = DummyRegressor(strategy="mean").fit(X, y, sample_weight)
    assert est.constant_ == np.average(y, weights=sample_weight)

    est = DummyRegressor(strategy="median").fit(X, y, sample_weight)
    assert est.constant_ == _weighted_percentile(y, sample_weight, 50.0)

    est = DummyRegressor(strategy="quantile", quantile=0.95).fit(X, y, sample_weight)
    assert est.constant_ == _weighted_percentile(y, sample_weight, 95.0)


def test_dummy_regressor_on_3D_array():
    # [FUNCTION DOC]
    # Checks DummyRegressor fits/predicts on a 3D string array X (X is ignored).
    X = np.array([[["foo"]], [["bar"]], [["baz"]]])
    y = np.array([2, 2, 2])
    y_expected = np.array([2, 2, 2])
    cls = DummyRegressor()
    cls.fit(X, y)
    y_pred = cls.predict(X)
    assert_array_equal(y_pred, y_expected)


def test_dummy_classifier_on_3D_array():
    # [FUNCTION DOC]
    # Checks DummyClassifier "stratified" on 3D X with a single class predicts that
    # class with probability 1.
    X = np.array([[["foo"]], [["bar"]], [["baz"]]])
    y = [2, 2, 2]
    y_expected = [2, 2, 2]
    y_proba_expected = [[1], [1], [1]]
    cls = DummyClassifier(strategy="stratified")
    cls.fit(X, y)
    y_pred = cls.predict(X)
    y_pred_proba = cls.predict_proba(X)
    assert_array_equal(y_pred, y_expected)
    assert_array_equal(y_pred_proba, y_proba_expected)


def test_dummy_regressor_return_std():
    # [FUNCTION DOC]
    # Checks predict with return_std returns a pair whose std part is all zeros.
    X = [[0]] * 3  # ignored
    y = np.array([2, 2, 2])
    y_std_expected = np.array([0, 0, 0])
    cls = DummyRegressor()
    cls.fit(X, y)
    y_pred_list = cls.predict(X, return_std=True)
    # there should be two elements when return_std is True
    assert len(y_pred_list) == 2
    # the second element should be all zeros
    assert_array_equal(y_pred_list[1], y_std_expected)


@pytest.mark.parametrize(
    "y,y_test",
    [
        ([1, 1, 1, 2], [1.25] * 4),
        (np.array([[2, 2], [1, 1], [1, 1], [1, 1]]), [[1.25, 1.25]] * 4),
    ],
)
def test_regressor_score_with_None(y, y_test):
    # [FUNCTION DOC]
    # Checks DummyRegressor accepts X=None in fit and score; R2 is 1.0 when test
    # targets equal the training mean.
    reg = DummyRegressor()
    reg.fit(None, y)
    assert reg.score(None, y_test) == 1.0


@pytest.mark.parametrize("strategy", ["mean", "median", "quantile", "constant"])
def test_regressor_prediction_independent_of_X(strategy):
    # [FUNCTION DOC]
    # For each regressor strategy, predictions do not depend on X values.
    y = [0, 2, 1, 1]
    X1 = [[0]] * 4
    reg1 = DummyRegressor(strategy=strategy, constant=0, quantile=0.7)
    reg1.fit(X1, y)
    predictions1 = reg1.predict(X1)

    X2 = [[1]] * 4
    reg2 = DummyRegressor(strategy=strategy, constant=0, quantile=0.7)
    reg2.fit(X2, y)
    predictions2 = reg2.predict(X2)

    assert_array_equal(predictions1, predictions2)


@pytest.mark.parametrize(
    "strategy", ["stratified", "most_frequent", "prior", "uniform", "constant"]
)
def test_dtype_of_classifier_probas(strategy):
    # [FUNCTION DOC]
    # Checks predict_proba returns float64 for every classifier strategy.
    y = [0, 2, 1, 1]
    X = np.zeros(4)
    model = DummyClassifier(strategy=strategy, random_state=0, constant=0)
    probas = model.fit(X, y).predict_proba(X)

    assert probas.dtype == np.float64
