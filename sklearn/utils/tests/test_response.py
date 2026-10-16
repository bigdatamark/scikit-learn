# [MODULE DOC]
# Tests for sklearn.utils._response: _get_response_values and
# _get_response_values_binary. Uses module-level iris data X, y (standardized in place
# with scale) and a binary subset X_binary, y_binary (first 100 rows, classes 0 and 1).
# Side effects at import: loads iris and scales it.
import warnings

import numpy as np
import pytest

from sklearn.base import clone
from sklearn.cluster import DBSCAN, KMeans
from sklearn.datasets import (
    load_iris,
    make_classification,
    make_multilabel_classification,
)
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.multioutput import ClassifierChain
from sklearn.preprocessing import scale
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils._response import _get_response_values, _get_response_values_binary
from sklearn.utils._testing import assert_allclose

X, y = load_iris(return_X_y=True)
# scale the data to avoid ConvergenceWarning with LogisticRegression
X = scale(X, copy=False)
X_binary, y_binary = X[:100], y[:100]


@pytest.mark.parametrize(
    "estimator, response_method",
    [
        (DecisionTreeRegressor(), "predict_proba"),
        (DecisionTreeRegressor(), ["predict_proba", "decision_function"]),
        (KMeans(n_clusters=2), "predict_proba"),
        (KMeans(n_clusters=2), ["predict_proba", "decision_function"]),
        (DBSCAN(), "predict"),
        (IsolationForest(), "predict_proba"),
        (IsolationForest(), ["predict_proba", "score"]),
    ],
)
def test_estimator_unsupported_response(estimator, response_method):
    # [FUNCTION DOC]
    # Verifies that _get_response_values raises AttributeError ('has none of the
    # following attributes') when the fitted estimator (regressor, clusterer, outlier
    # detector) lacks every requested response method, given as a string or a list.
    # Inputs: estimator - unfitted estimator, cloned and fit on 10 random 2-feature
    # samples; response_method - str or list of method names.
    """Check the error message with not supported response method."""
    X, y = np.random.RandomState(0).randn(10, 2), np.array([0, 1] * 5)
    estimator = clone(estimator).fit(X, y)  # clone to make test execution thread-safe
    err_msg = "has none of the following attributes:"
    with pytest.raises(AttributeError, match=err_msg):
        _get_response_values(
            estimator,
            X,
            response_method=response_method,
        )


@pytest.mark.parametrize(
    "estimator, response_method",
    [
        (LinearRegression(), "predict"),
        (KMeans(n_clusters=2, random_state=0), "predict"),
        (KMeans(n_clusters=2, random_state=0), "score"),
        (KMeans(n_clusters=2, random_state=0), ["predict", "score"]),
        (IsolationForest(random_state=0), "predict"),
        (IsolationForest(random_state=0), "decision_function"),
        (IsolationForest(random_state=0), ["decision_function", "predict"]),
    ],
)
@pytest.mark.parametrize("return_response_method_used", [True, False])
def test_estimator_get_response_values(
    estimator, response_method, return_response_method_used
):
    # [FUNCTION DOC]
    # Verifies that for non-classifiers (regressor, KMeans, IsolationForest)
    # _get_response_values returns the output of the first listed response method,
    # pos_label None, and, when return_response_method_used is True, the name of the
    # method used as a third element.
    # Inputs: estimator, response_method (str or list), return_response_method_used
    # (bool) - parametrized.
    """Check the behaviour of `_get_response_values`."""
    X, y = np.random.RandomState(0).randn(10, 2), np.array([0, 1] * 5)
    estimator = clone(estimator).fit(X, y)  # clone to make test execution thread-safe
    results = _get_response_values(
        estimator,
        X,
        response_method=response_method,
        return_response_method_used=return_response_method_used,
    )
    chosen_response_method = (
        response_method[0] if isinstance(response_method, list) else response_method
    )
    prediction_method = getattr(estimator, chosen_response_method)
    assert_allclose(results[0], prediction_method(X))
    assert results[1] is None
    if return_response_method_used:
        assert results[2] == chosen_response_method


@pytest.mark.parametrize(
    "response_method",
    ["predict_proba", "decision_function", "predict", "predict_log_proba"],
)
def test_get_response_values_classifier_unknown_pos_label(response_method):
    # [FUNCTION DOC]
    # Verifies that a binary LogisticRegression with pos_label='whatever' (not in
    # classes_) makes _get_response_values raise ValueError listing the valid labels,
    # for each response method.
    # Inputs: response_method - parametrized method name.
    """Check that `_get_response_values` raises the proper error message with
    classifier."""
    X, y = make_classification(n_samples=10, n_classes=2, random_state=0)
    classifier = LogisticRegression().fit(X, y)

    # provide a `pos_label` which is not in `y`
    err_msg = r"pos_label=whatever is not a valid label: It should be one of \[0 1\]"
    with pytest.raises(ValueError, match=err_msg):
        _get_response_values(
            classifier,
            X,
            response_method=response_method,
            pos_label="whatever",
        )


@pytest.mark.parametrize("response_method", ["predict_proba", "predict_log_proba"])
def test_get_response_values_classifier_inconsistent_y_pred_for_binary_proba(
    response_method,
):
    # [FUNCTION DOC]
    # Verifies that a classifier fit on a single class (predict_proba of shape (n, 1))
    # makes _get_response_values raise ValueError requiring two classes when using
    # predict_proba or predict_log_proba.
    # Inputs: response_method - parametrized.
    """Check that `_get_response_values` will raise an error when `y_pred` has a
    single class with `predict_proba`."""
    X, y_two_class = make_classification(n_samples=10, n_classes=2, random_state=0)
    y_single_class = np.zeros_like(y_two_class)
    classifier = DecisionTreeClassifier().fit(X, y_single_class)

    err_msg = (
        r"Got predict_proba of shape \(10, 1\), but need classifier with "
        r"two classes"
    )
    with pytest.raises(ValueError, match=err_msg):
        _get_response_values(classifier, X, response_method=response_method)


@pytest.mark.parametrize("return_response_method_used", [True, False])
def test_get_response_values_binary_classifier_decision_function(
    return_response_method_used,
):
    # [FUNCTION DOC]
    # Verifies binary decision_function handling: with pos_label None the raw scores and
    # pos_label 1 are returned; with pos_label set to classes_[0] the scores are negated
    # and pos_label 0 is returned. Also checks the optional third element naming the
    # method.
    # Inputs: return_response_method_used - bool, parametrized.
    """Check the behaviour of `_get_response_values` with `decision_function`
    and binary classifier."""
    X, y = make_classification(
        n_samples=10,
        n_classes=2,
        weights=[0.3, 0.7],
        random_state=0,
    )
    classifier = LogisticRegression().fit(X, y)
    response_method = "decision_function"

    # default `pos_label`
    results = _get_response_values(
        classifier,
        X,
        response_method=response_method,
        pos_label=None,
        return_response_method_used=return_response_method_used,
    )
    assert_allclose(results[0], classifier.decision_function(X))
    assert results[1] == 1
    if return_response_method_used:
        assert results[2] == "decision_function"

    # when forcing `pos_label=classifier.classes_[0]`
    results = _get_response_values(
        classifier,
        X,
        response_method=response_method,
        pos_label=classifier.classes_[0],
        return_response_method_used=return_response_method_used,
    )
    assert_allclose(results[0], classifier.decision_function(X) * -1)
    assert results[1] == 0
    if return_response_method_used:
        assert results[2] == "decision_function"


@pytest.mark.parametrize("return_response_method_used", [True, False])
@pytest.mark.parametrize("response_method", ["predict_proba", "predict_log_proba"])
def test_get_response_values_binary_classifier_predict_proba(
    return_response_method_used, response_method
):
    # [FUNCTION DOC]
    # Verifies binary predict_proba / predict_log_proba handling: default pos_label
    # selects column 1 and reports pos_label 1; pos_label=classes_[0] selects column 0.
    # Checks result tuple length is 3 or 2 depending on return_response_method_used.
    # Inputs: return_response_method_used - bool; response_method - str; both
    # parametrized.
    """Check that `_get_response_values` with `predict_proba` and binary
    classifier."""
    X, y = make_classification(
        n_samples=10,
        n_classes=2,
        weights=[0.3, 0.7],
        random_state=0,
    )
    classifier = LogisticRegression().fit(X, y)

    # default `pos_label`
    results = _get_response_values(
        classifier,
        X,
        response_method=response_method,
        pos_label=None,
        return_response_method_used=return_response_method_used,
    )
    assert_allclose(results[0], getattr(classifier, response_method)(X)[:, 1])
    assert results[1] == 1
    if return_response_method_used:
        assert len(results) == 3
        assert results[2] == response_method
    else:
        assert len(results) == 2

    # when forcing `pos_label=classifier.classes_[0]`
    y_pred, pos_label, *_ = _get_response_values(
        classifier,
        X,
        response_method=response_method,
        pos_label=classifier.classes_[0],
        return_response_method_used=return_response_method_used,
    )
    assert_allclose(y_pred, getattr(classifier, response_method)(X)[:, 0])
    assert pos_label == 0


@pytest.mark.parametrize(
    "estimator, X, y, err_msg, params",
    [
        (
            DecisionTreeRegressor(),
            X_binary,
            y_binary,
            "Expected 'estimator' to be a binary classifier",
            {"response_method": "auto"},
        ),
        (
            DecisionTreeClassifier(),
            X_binary,
            y_binary,
            r"pos_label=unknown is not a valid label: It should be one of \[0 1\]",
            {"response_method": "auto", "pos_label": "unknown"},
        ),
        (
            DecisionTreeClassifier(),
            X,
            y,
            "be a binary classifier. Got 3 classes instead.",
            {"response_method": "predict_proba"},
        ),
    ],
)
def test_get_response_error(estimator, X, y, err_msg, params):
    # [FUNCTION DOC]
    # Verifies the ValueError messages of _get_response_values_binary for: a regressor
    # (not a binary classifier), an unknown pos_label, and a 3-class classifier.
    # Inputs: estimator - cloned and fit on X, y; X, y - data; err_msg - regex expected;
    # params - kwargs forwarded to _get_response_values_binary.
    """Check that we raise the proper error messages in _get_response_values_binary."""

    estimator = clone(estimator).fit(X, y)  # clone to make test execution thread-safe
    with pytest.raises(ValueError, match=err_msg):
        _get_response_values_binary(estimator, X, **params)


@pytest.mark.parametrize("return_response_method_used", [True, False])
def test_get_response_predict_proba(return_response_method_used):
    # [FUNCTION DOC]
    # Verifies _get_response_values_binary with predict_proba on a decision tree:
    # default returns column 1 with pos_label 1; pos_label=0 returns column 0 with
    # pos_label 0; third element is the method name when requested.
    # Inputs: return_response_method_used - bool, parametrized.
    """Check the behaviour of `_get_response_values_binary` using `predict_proba`."""
    classifier = DecisionTreeClassifier().fit(X_binary, y_binary)
    results = _get_response_values_binary(
        classifier,
        X_binary,
        response_method="predict_proba",
        return_response_method_used=return_response_method_used,
    )
    assert_allclose(results[0], classifier.predict_proba(X_binary)[:, 1])
    assert results[1] == 1
    if return_response_method_used:
        assert results[2] == "predict_proba"

    results = _get_response_values_binary(
        classifier,
        X_binary,
        response_method="predict_proba",
        pos_label=0,
        return_response_method_used=return_response_method_used,
    )
    assert_allclose(results[0], classifier.predict_proba(X_binary)[:, 0])
    assert results[1] == 0
    if return_response_method_used:
        assert results[2] == "predict_proba"


@pytest.mark.parametrize("return_response_method_used", [True, False])
def test_get_response_decision_function(return_response_method_used):
    # [FUNCTION DOC]
    # Verifies _get_response_values_binary with decision_function on logistic
    # regression: default returns raw scores and pos_label 1; pos_label=0 returns
    # negated scores and pos_label 0; third element is the method name when requested.
    # Inputs: return_response_method_used - bool, parametrized.
    """Check the behaviour of `_get_response_values_binary` using decision_function."""
    classifier = LogisticRegression().fit(X_binary, y_binary)
    results = _get_response_values_binary(
        classifier,
        X_binary,
        response_method="decision_function",
        return_response_method_used=return_response_method_used,
    )
    assert_allclose(results[0], classifier.decision_function(X_binary))
    assert results[1] == 1
    if return_response_method_used:
        assert results[2] == "decision_function"

    results = _get_response_values_binary(
        classifier,
        X_binary,
        response_method="decision_function",
        pos_label=0,
        return_response_method_used=return_response_method_used,
    )
    assert_allclose(results[0], classifier.decision_function(X_binary) * -1)
    assert results[1] == 0
    if return_response_method_used:
        assert results[2] == "decision_function"


@pytest.mark.parametrize(
    "estimator, response_method",
    [
        (DecisionTreeClassifier(max_depth=2, random_state=0), "predict_proba"),
        (DecisionTreeClassifier(max_depth=2, random_state=0), "predict_log_proba"),
        (LogisticRegression(), "decision_function"),
    ],
)
def test_get_response_values_multiclass(estimator, response_method):
    # [FUNCTION DOC]
    # Verifies that for a 3-class estimator _get_response_values returns predictions
    # untouched with shape (n_samples, n_classes) and pos_label None; probabilities are
    # within [0, 1] and log-probabilities are non-positive.
    # Inputs: estimator - cloned and fit on iris; response_method - str.
    """Check that we can call `_get_response_values` with a multiclass estimator.
    It should return the predictions untouched.
    """
    estimator = clone(estimator).fit(X, y)  # clone to make test execution thread-safe
    predictions, pos_label = _get_response_values(
        estimator, X, response_method=response_method
    )

    assert pos_label is None
    assert predictions.shape == (X.shape[0], len(estimator.classes_))
    if response_method == "predict_proba":
        assert np.logical_and(predictions >= 0, predictions <= 1).all()
    elif response_method == "predict_log_proba":
        assert (predictions <= 0.0).all()


def test_get_response_values_with_response_list():
    # [FUNCTION DOC]
    # Verifies that when a list of response methods is given, the first one available on
    # the estimator is used: predict_proba first yields column-1 probabilities,
    # decision_function first yields raw scores; pos_label is 1 and the method name is
    # returned.
    """Check the behaviour of passing a list of responses to `_get_response_values`."""
    classifier = LogisticRegression().fit(X_binary, y_binary)

    # it should use `predict_proba`
    y_pred, pos_label, response_method = _get_response_values(
        classifier,
        X_binary,
        response_method=["predict_proba", "decision_function"],
        return_response_method_used=True,
    )
    assert_allclose(y_pred, classifier.predict_proba(X_binary)[:, 1])
    assert pos_label == 1
    assert response_method == "predict_proba"

    # it should use `decision_function`
    y_pred, pos_label, response_method = _get_response_values(
        classifier,
        X_binary,
        response_method=["decision_function", "predict_proba"],
        return_response_method_used=True,
    )
    assert_allclose(y_pred, classifier.decision_function(X_binary))
    assert pos_label == 1
    assert response_method == "decision_function"


@pytest.mark.parametrize(
    "response_method", ["predict_proba", "decision_function", "predict"]
)
def test_get_response_values_multilabel_indicator(response_method):
    # [FUNCTION DOC]
    # Verifies multilabel handling with ClassifierChain: output shape equals the
    # indicator target shape, pos_label is None; predict_proba values lie in [0, 1],
    # decision_function has values below 0 and above 1, predict returns only 0/1.
    # Inputs: response_method - parametrized.
    X, Y = make_multilabel_classification(random_state=0)
    estimator = ClassifierChain(LogisticRegression()).fit(X, Y)

    y_pred, pos_label = _get_response_values(
        estimator, X, response_method=response_method
    )
    assert pos_label is None
    assert y_pred.shape == Y.shape

    if response_method == "predict_proba":
        assert np.logical_and(y_pred >= 0, y_pred <= 1).all()
    elif response_method == "decision_function":
        # values returned by `decision_function` are not bounded in [0, 1]
        assert (y_pred < 0).sum() > 0
        assert (y_pred > 1).sum() > 0
    else:  # response_method == "predict"
        assert np.logical_or(y_pred == 0, y_pred == 1).all()


def test_response_values_type_of_target_on_classes_no_warning():
    # [FUNCTION DOC]
    # Non-regression test (#31583): fits a 30-class LogisticRegression on 120 samples
    # and calls _get_response_values with predict_proba while UserWarning is turned into
    # an error, verifying that no spurious 'unique classes > 50% of samples' warning is
    # raised from type_of_target on classes_.
    """
    Ensure `_get_response_values` doesn't raise spurious warning.

    "The number of unique classes is greater than > 50% of samples"
    warning should not be raised when calling `type_of_target(classes_)`.

    Non-regression test for issue #31583.
    """
    X = np.random.RandomState(0).randn(120, 3)
    # 30 classes, less than 50% of number of samples
    y = np.repeat(np.arange(30), 4)

    clf = LogisticRegression().fit(X, y)

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)

        _get_response_values(clf, X, response_method="predict_proba")


@pytest.mark.parametrize(
    "estimator, response_method, target_type, expected_shape",
    [
        (LogisticRegression(), "predict", "binary", (10,)),
        (LogisticRegression(), "predict_proba", "binary", (10,)),
        (LogisticRegression(), "decision_function", "binary", (10,)),
        (LogisticRegression(), "predict", "multiclass", (10,)),
        (LogisticRegression(), "predict_proba", "multiclass", (10, 4)),
        (LogisticRegression(), "decision_function", "multiclass", (10, 4)),
        (ClassifierChain(LogisticRegression()), "predict", "multilabel", (10, 2)),
        (ClassifierChain(LogisticRegression()), "predict_proba", "multilabel", (10, 2)),
        (
            ClassifierChain(LogisticRegression()),
            "decision_function",
            "multilabel",
            (10, 2),
        ),
        (IsolationForest(), "predict", "binary", (10,)),
        (IsolationForest(), "predict", "multiclass", (10,)),
        (DecisionTreeRegressor(), "predict", "binary", (10,)),
        (DecisionTreeRegressor(), "predict", "multiclass", (10,)),
        (KMeans(n_clusters=2), "predict", "binary", (10,)),
        (KMeans(n_clusters=4), "predict", "multiclass", (10,)),
    ],
)
def test_response_values_output_shape_(
    estimator, response_method, target_type, expected_shape
):
    # [FUNCTION DOC]
    # Verifies output shapes of _get_response_values: 1d (n_samples,) for binary, for
    # multiclass predict, and for outlier detection / regression / clustering; 2d
    # (n_samples, n_classes) for multiclass proba/decision; 2d (n_samples, n_outputs)
    # for multilabel.
    # Inputs: estimator; response_method; target_type - 'binary', 'multiclass' or
    # anything else for multilabel; expected_shape - tuple. Builds 10 random samples and
    # a target matching target_type.
    """
    Check that output shape corresponds to docstring description

    - for binary classification, it is a 1d array of shape `(n_samples,)`;
    - for multiclass classification
        - with response_method="predict", it is a 1d array of shape `(n_samples,)`;
        - otherwise, it is a 2d array of shape `(n_samples, n_classes)`;
    - for multilabel classification, it is a 2d array of shape `(n_samples, n_outputs)`;
    - for outlier detection, regression and clustering,
      it is a 1d array of shape `(n_samples,)`.
    """
    X = np.random.RandomState(0).randn(10, 2)
    if target_type == "binary":
        y = np.array([0, 1] * 5)
    elif target_type == "multiclass":
        y = [0, 1, 2, 3, 0, 1, 2, 3, 3, 0]
    else:  # multilabel
        y = np.array([[0, 1], [1, 0]] * 5)

    estimator = clone(estimator).fit(X, y)  # clone to make test execution thread-safe

    y_pred, _ = _get_response_values(estimator, X, response_method=response_method)

    assert y_pred.shape == expected_shape
