# [MODULE DOC]
# Tests for the mocking helpers in sklearn.utils._mocking: CheckingClassifier (input
# checks, fit params, methods_to_check, outputs) and _MockEstimatorOnOffPrediction.
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from scipy import sparse

from sklearn.datasets import load_iris
from sklearn.utils import _safe_indexing, check_array
from sklearn.utils._mocking import (
    CheckingClassifier,
    _MockEstimatorOnOffPrediction,
)
from sklearn.utils._testing import _convert_container
from sklearn.utils.fixes import CSR_CONTAINERS


@pytest.fixture
def iris():
    # [FUNCTION DOC]
    # HELPER: Fixture returning the iris dataset as an (X, y) tuple (150x4 floats, 3
    # classes).
    return load_iris(return_X_y=True)


def _success(x):
    # [FUNCTION DOC]
    # HELPER: Check callback that accepts any input; always returns True.
    return True


def _fail(x):
    # [FUNCTION DOC]
    # HELPER: Check callback that rejects any input; always returns False.
    return False


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"check_X": _success},
        {"check_y": _success},
        {"check_X": _success, "check_y": _success},
    ],
)
def test_check_on_fit_success(iris, kwargs):
    # [FUNCTION DOC]
    # Verifies fit succeeds on iris when no checks or only passing check_X / check_y
    # callbacks are given.
    X, y = iris
    CheckingClassifier(**kwargs).fit(X, y)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"check_X": _fail},
        {"check_y": _fail},
        {"check_X": _success, "check_y": _fail},
        {"check_X": _fail, "check_y": _success},
        {"check_X": _fail, "check_y": _fail},
    ],
)
def test_check_on_fit_fail(iris, kwargs):
    # [FUNCTION DOC]
    # Verifies fit raises AssertionError when check_X and/or check_y returns False.
    X, y = iris
    clf = CheckingClassifier(**kwargs)
    with pytest.raises(AssertionError):
        clf.fit(X, y)


@pytest.mark.parametrize(
    "pred_func", ["predict", "predict_proba", "decision_function", "score"]
)
def test_check_X_on_predict_success(iris, pred_func):
    # [FUNCTION DOC]
    # Verifies predict, predict_proba, decision_function and score run without error
    # when check_X passes.
    X, y = iris
    clf = CheckingClassifier(check_X=_success).fit(X, y)
    getattr(clf, pred_func)(X)


@pytest.mark.parametrize(
    "pred_func", ["predict", "predict_proba", "decision_function", "score"]
)
def test_check_X_on_predict_fail(iris, pred_func):
    # [FUNCTION DOC]
    # Fits with a passing check_X, then swaps in a failing check_X via set_params and
    # verifies each prediction method raises AssertionError.
    X, y = iris
    clf = CheckingClassifier(check_X=_success).fit(X, y)
    clf.set_params(check_X=_fail)
    with pytest.raises(AssertionError):
        getattr(clf, pred_func)(X)


@pytest.mark.parametrize("input_type", ["list", "array", "sparse", "pandas"])
def test_checking_classifier(iris, input_type):
    # [FUNCTION DOC]
    # For list, array, sparse and pandas inputs, checks fitted classes_ and
    # n_features_in_, that predictions are valid classes, that score is 0 by default and
    # 1 once foo_param is 10, and predict_proba/decision_function shapes and ranges for
    # 3 classes; then refits on only classes 0 and 1 and checks proba shape (100, 2) and
    # a 1D decision function of shape (100,).
    # Check that the CheckingClassifier outputs what we expect
    X, y = iris
    X = _convert_container(X, input_type)
    clf = CheckingClassifier()
    clf.fit(X, y)

    assert_array_equal(clf.classes_, np.unique(y))
    assert len(clf.classes_) == 3
    assert clf.n_features_in_ == 4

    y_pred = clf.predict(X)
    assert all(pred in clf.classes_ for pred in y_pred)

    assert clf.score(X) == pytest.approx(0)
    clf.set_params(foo_param=10)
    assert clf.fit(X, y).score(X) == pytest.approx(1)

    y_proba = clf.predict_proba(X)
    assert y_proba.shape == (150, 3)
    assert np.logical_and(y_proba >= 0, y_proba <= 1).all()

    y_decision = clf.decision_function(X)
    assert y_decision.shape == (150, 3)

    # check the shape in case of binary classification
    first_2_classes = np.logical_or(y == 0, y == 1)
    X = _safe_indexing(X, first_2_classes)
    y = _safe_indexing(y, first_2_classes)
    clf.fit(X, y)

    y_proba = clf.predict_proba(X)
    assert y_proba.shape == (100, 2)
    assert np.logical_and(y_proba >= 0, y_proba <= 1).all()

    y_decision = clf.decision_function(X)
    assert y_decision.shape == (100,)


@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_checking_classifier_with_params(iris, csr_container):
    # [FUNCTION DOC]
    # Verifies check_X=sparse.issparse rejects dense and accepts CSR input, and that
    # check_X=check_array with check_X_params accept_sparse=False accepts dense but
    # raises TypeError ("Sparse data was passed") for CSR input.
    X, y = iris
    X_sparse = csr_container(X)

    clf = CheckingClassifier(check_X=sparse.issparse)
    with pytest.raises(AssertionError):
        clf.fit(X, y)
    clf.fit(X_sparse, y)

    clf = CheckingClassifier(
        check_X=check_array, check_X_params={"accept_sparse": False}
    )
    clf.fit(X, y)
    with pytest.raises(TypeError, match="Sparse data was passed"):
        clf.fit(X_sparse, y)


def test_checking_classifier_fit_params(iris):
    # [FUNCTION DOC]
    # With expected_sample_weight=True, passing a sample_weight of half the sample
    # length must raise ValueError whose message exactly states actual and expected
    # shapes.
    # check the error raised when the number of samples is not the one expected
    X, y = iris
    clf = CheckingClassifier(expected_sample_weight=True)
    sample_weight = np.ones(len(X) // 2)

    msg = f"sample_weight.shape == ({len(X) // 2},), expected ({len(X)},)!"
    with pytest.raises(ValueError) as exc:
        clf.fit(X, y, sample_weight=sample_weight)
    assert exc.value.args[0] == msg


def test_checking_classifier_missing_fit_params(iris):
    # [FUNCTION DOC]
    # With expected_sample_weight=True, fitting without sample_weight must raise
    # AssertionError mentioning that sample_weight was expected.
    X, y = iris
    clf = CheckingClassifier(expected_sample_weight=True)
    err_msg = "Expected sample_weight to be passed"
    with pytest.raises(AssertionError, match=err_msg):
        clf.fit(X, y)


@pytest.mark.parametrize(
    "methods_to_check",
    [["predict"], ["predict", "predict_proba"]],
)
@pytest.mark.parametrize(
    "predict_method", ["predict", "predict_proba", "decision_function", "score"]
)
def test_checking_classifier_methods_to_check(iris, methods_to_check, predict_method):
    # [FUNCTION DOC]
    # With a check_X that fails on dense input, verifies the check is applied only to
    # the prediction methods listed in methods_to_check (raising AssertionError) and
    # bypassed for the others; fit is never in the list so it succeeds.
    # check that methods_to_check allows to bypass checks
    X, y = iris

    clf = CheckingClassifier(
        check_X=sparse.issparse,
        methods_to_check=methods_to_check,
    )

    clf.fit(X, y)
    if predict_method in methods_to_check:
        with pytest.raises(AssertionError):
            getattr(clf, predict_method)(X)
    else:
        getattr(clf, predict_method)(X)


@pytest.mark.parametrize(
    "response_methods",
    [
        ["predict"],
        ["predict", "predict_proba"],
        ["predict", "decision_function"],
        ["predict", "predict_proba", "decision_function"],
    ],
)
def test_mock_estimator_on_off_prediction(iris, response_methods):
    # [FUNCTION DOC]
    # Verifies _MockEstimatorOnOffPrediction sets classes_ after fit, exposes exactly
    # the configured response methods (each returning its own name as a string) and does
    # not expose the others.
    X, y = iris
    estimator = _MockEstimatorOnOffPrediction(response_methods=response_methods)

    estimator.fit(X, y)
    assert hasattr(estimator, "classes_")
    assert_array_equal(estimator.classes_, np.unique(y))

    possible_responses = ["predict", "predict_proba", "decision_function"]
    for response in possible_responses:
        if response in response_methods:
            assert hasattr(estimator, response)
            assert getattr(estimator, response)(X) == response
        else:
            assert not hasattr(estimator, response)
