# [MODULE DOC]
# Tests for sklearn.model_selection._validation: cross_val_score, cross_validate,
# cross_val_predict, learning_curve, validation_curve, permutation_test_score and the
# private helpers _fit_and_score, _score and _check_is_permutation. Covers scoring
# formats, error_score handling and warnings, fit params / metadata routing, sparse and
# pandas inputs, groups, verbosity output and array API support.
# Defines several mock estimators and module-level fixtures data: X (15x2 ones), y
# (7 classes), y2 (3 balanced classes) and P (5x5 identity, used by MockClassifier).
"""Test the validation module"""

import os
import re
import tempfile
import warnings
from functools import partial
from time import sleep

import numpy as np
import pytest
from scipy.sparse import issparse

from sklearn import config_context
from sklearn.base import BaseEstimator, ClassifierMixin, clone, is_classifier
from sklearn.cluster import KMeans
from sklearn.datasets import (
    load_diabetes,
    load_digits,
    load_iris,
    make_classification,
    make_multilabel_classification,
    make_regression,
)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import FitFailedWarning, UnsetMetadataPassedError
from sklearn.impute import SimpleImputer
from sklearn.linear_model import (
    LogisticRegression,
    Ridge,
    RidgeClassifier,
    SGDClassifier,
)
from sklearn.metrics import (
    accuracy_score,
    check_scoring,
    confusion_matrix,
    explained_variance_score,
    make_scorer,
    mean_squared_error,
    precision_recall_fscore_support,
    precision_score,
    r2_score,
)
from sklearn.metrics._scorer import _MultimetricScorer
from sklearn.model_selection import (
    GridSearchCV,
    GroupKFold,
    GroupShuffleSplit,
    KFold,
    LeaveOneGroupOut,
    LeaveOneOut,
    LeavePGroupsOut,
    ShuffleSplit,
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
    cross_validate,
    learning_curve,
    permutation_test_score,
    validation_curve,
)
from sklearn.model_selection._validation import (
    _check_is_permutation,
    _fit_and_score,
    _score,
)
from sklearn.model_selection.tests.common import OneTimeSplitter
from sklearn.model_selection.tests.test_search import FailingClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, scale
from sklearn.svm import SVC, LinearSVC
from sklearn.tests.metadata_routing_common import (
    ConsumingClassifier,
    ConsumingScorer,
    ConsumingSplitter,
    _Registry,
    check_recorded_metadata,
)
from sklearn.utils import shuffle
from sklearn.utils._array_api import (
    _atol_for_type,
    move_to,
    yield_namespace_device_dtype_combinations,
)
from sklearn.utils._mocking import CheckingClassifier, MockDataFrame
from sklearn.utils._testing import (
    _array_api_for_tests,
    assert_allclose,
    assert_almost_equal,
    assert_array_almost_equal,
    assert_array_equal,
)
from sklearn.utils.estimator_checks import _NotAnArray
from sklearn.utils.fixes import COO_CONTAINERS, CSR_CONTAINERS
from sklearn.utils.validation import _num_samples


class MockImprovingEstimator(BaseEstimator):
    # [CLASS DOC]
    # HELPER: Fake estimator for learning_curve tests. Its score depends only on the
    # training size: on training data it returns 2 - train_sizes / n_max_train_sizes,
    # otherwise train_sizes / n_max_train_sizes, so train scores decrease and test scores
    # increase with more data. Training data is identified by object identity.
    """Dummy classifier to test the learning curve"""

    def __init__(self, n_max_train_sizes):
        # [FUNCTION DOC]
        # HELPER: Stores n_max_train_sizes and initializes train_sizes to 0 and X_subset None.
        self.n_max_train_sizes = n_max_train_sizes
        self.train_sizes = 0
        self.X_subset = None

    def fit(self, X_subset, y_subset=None):
        # [FUNCTION DOC]
        # HELPER: Remembers X_subset and sets train_sizes to its number of rows. Returns self.
        self.X_subset = X_subset
        self.train_sizes = X_subset.shape[0]
        return self

    def predict(self, X):
        # [FUNCTION DOC]
        # HELPER: Not supported; always raises NotImplementedError.
        raise NotImplementedError

    def score(self, X=None, Y=None):
        # [FUNCTION DOC]
        # HELPER: Returns the size-dependent score described at class level, choosing the
        # branch with _is_training_data(X). Y is ignored.
        # training score becomes worse (2 -> 1), test error better (0 -> 1)
        if self._is_training_data(X):
            return 2.0 - float(self.train_sizes) / self.n_max_train_sizes
        else:
            return float(self.train_sizes) / self.n_max_train_sizes

    def _is_training_data(self, X):
        # [FUNCTION DOC]
        # HELPER: True if X is the very object passed to the last fit.
        return X is self.X_subset


class MockIncrementalImprovingEstimator(MockImprovingEstimator):
    # [CLASS DOC]
    # HELPER: Incremental variant: partial_fit accumulates train_sizes and remembers the
    # first row of the latest batch; training data is detected by membership of that row.
    # Optionally asserts that expected fit params are received with sample-aligned length.
    """Dummy classifier that provides partial_fit"""

    def __init__(self, n_max_train_sizes, expected_fit_params=None):
        # [FUNCTION DOC]
        # HELPER: Calls parent init, sets x None and stores expected_fit_params (list or None).
        super().__init__(n_max_train_sizes)
        self.x = None
        self.expected_fit_params = expected_fit_params

    def _is_training_data(self, X):
        # [FUNCTION DOC]
        # HELPER: True if the remembered row x is contained in X (numpy "in" semantics: any
        # element match, so it is a loose check).
        return self.x in X

    def partial_fit(self, X, y=None, **params):
        # [FUNCTION DOC]
        # HELPER: Adds the batch size to train_sizes and stores the first row. If
        # expected_fit_params is set, raises AssertionError when any expected param is missing
        # or when an expected param's length differs from the batch size.
        # Outputs: None.
        # KNOWN ISSUE: returns None instead of self (callers in tests do not chain).
        self.train_sizes += X.shape[0]
        self.x = X[0]
        if self.expected_fit_params:
            missing = set(self.expected_fit_params) - set(params)
            if missing:
                raise AssertionError(
                    f"Expected fit parameter(s) {list(missing)} not seen."
                )
            for key, value in params.items():
                if key in self.expected_fit_params and _num_samples(
                    value
                ) != _num_samples(X):
                    raise AssertionError(
                        f"Fit parameter {key} has length {_num_samples(value)}"
                        f"; expected {_num_samples(X)}."
                    )


class MockEstimatorWithParameter(BaseEstimator):
    # [CLASS DOC]
    # HELPER: Fake estimator for validation_curve tests: score is param on the training data
    # and 1 - param otherwise.
    """Dummy classifier to test the validation curve"""

    def __init__(self, param=0.5):
        # [FUNCTION DOC]
        # HELPER: Sets X_subset None and stores param (default 0.5).
        self.X_subset = None
        self.param = param

    def fit(self, X_subset, y_subset):
        # [FUNCTION DOC]
        # HELPER: Remembers X_subset and its row count. Returns self.
        self.X_subset = X_subset
        self.train_sizes = X_subset.shape[0]
        return self

    def predict(self, X):
        # [FUNCTION DOC]
        # HELPER: Always raises NotImplementedError.
        raise NotImplementedError

    def score(self, X=None, y=None):
        # [FUNCTION DOC]
        # HELPER: Returns param when X is the training object, else 1 - param.
        return self.param if self._is_training_data(X) else 1 - self.param

    def _is_training_data(self, X):
        # [FUNCTION DOC]
        # HELPER: Identity check against the last fitted X_subset.
        return X is self.X_subset


class MockEstimatorWithSingleFitCallAllowed(MockEstimatorWithParameter):
    # [CLASS DOC]
    # HELPER: Variant that fails if fit is called twice on the same instance, used to check
    # that validation_curve clones before fitting.
    """Dummy classifier that disallows repeated calls of fit method"""

    def fit(self, X_subset, y_subset):
        # [FUNCTION DOC]
        # HELPER: Asserts fit_called_ is not yet set, sets it, then delegates to parent fit.
        # Exceptions: AssertionError on a second call.
        assert not hasattr(self, "fit_called_"), "fit is called the second time"
        self.fit_called_ = True
        return super().fit(X_subset, y_subset)

    def predict(self, X):
        # [FUNCTION DOC]
        # HELPER: Always raises NotImplementedError.
        raise NotImplementedError


class MockClassifier(ClassifierMixin, BaseEstimator):
    # [CLASS DOC]
    # HELPER: Dummy classifier for cross-validation tests. fit checks shapes of optional
    # fit params and records non-array params; predict returns the first column; predict_proba
    # returns the input; score is 1 / (1 + abs(a)).
    """Dummy classifier to test the cross-validation"""

    def __init__(self, a=0, allow_nd=False):
        # [FUNCTION DOC]
        # HELPER: Stores a (score control) and allow_nd (accept >2-D X by flattening).
        self.a = a
        self.allow_nd = allow_nd

    def fit(
        self,
        X,
        Y=None,
        sample_weight=None,
        class_prior=None,
        sparse_sample_weight=None,
        sparse_param=None,
        dummy_int=None,
        dummy_str=None,
        dummy_obj=None,
        callback=None,
    ):
        # [FUNCTION DOC]
        # HELPER: Stores dummy_int/str/obj, invokes callback(self) if given, flattens X when
        # allow_nd, raises ValueError for >=3-D X otherwise, and asserts that sample_weight and
        # sparse_sample_weight have one row per sample, class_prior has one entry per class
        # and sparse_param has the shape of module-level P. Sets classes_. Returns self.
        # KNOWN ISSUE: the class_prior check and classes_ use the module-level global y instead
        # of the Y argument, so they reflect the global labels, not the fitted fold.
        """The dummy arguments are to test that this fit function can
        accept non-array arguments through cross-validation, such as:
            - int
            - str (this is actually array-like)
            - object
            - function
        """
        self.dummy_int = dummy_int
        self.dummy_str = dummy_str
        self.dummy_obj = dummy_obj
        if callback is not None:
            callback(self)

        if self.allow_nd:
            X = X.reshape(len(X), -1)
        if X.ndim >= 3 and not self.allow_nd:
            raise ValueError("X cannot be d")
        if sample_weight is not None:
            assert sample_weight.shape[0] == X.shape[0], (
                "MockClassifier extra fit_param "
                "sample_weight.shape[0] is {0}, should be {1}".format(
                    sample_weight.shape[0], X.shape[0]
                )
            )
        if class_prior is not None:
            assert class_prior.shape[0] == len(np.unique(y)), (
                "MockClassifier extra fit_param class_prior.shape[0]"
                " is {0}, should be {1}".format(class_prior.shape[0], len(np.unique(y)))
            )
        if sparse_sample_weight is not None:
            fmt = (
                "MockClassifier extra fit_param sparse_sample_weight"
                ".shape[0] is {0}, should be {1}"
            )
            assert sparse_sample_weight.shape[0] == X.shape[0], fmt.format(
                sparse_sample_weight.shape[0], X.shape[0]
            )
        if sparse_param is not None:
            fmt = (
                "MockClassifier extra fit_param sparse_param.shape "
                "is ({0}, {1}), should be ({2}, {3})"
            )
            assert sparse_param.shape == P.shape, fmt.format(
                sparse_param.shape[0],
                sparse_param.shape[1],
                P.shape[0],
                P.shape[1],
            )
        self.classes_ = np.unique(y)
        return self

    def predict(self, T):
        # [FUNCTION DOC]
        # HELPER: Returns the first column of T (after flattening when allow_nd).
        if self.allow_nd:
            T = T.reshape(len(T), -1)
        return T[:, 0]

    def predict_proba(self, T):
        # [FUNCTION DOC]
        # HELPER: Returns T unchanged.
        return T

    def score(self, X=None, Y=None):
        # [FUNCTION DOC]
        # HELPER: Returns 1 / (1 + abs(a)); inputs ignored.
        return 1.0 / (1 + np.abs(self.a))

    def get_params(self, deep=False):
        # [FUNCTION DOC]
        # HELPER: Returns only a and allow_nd (deep ignored), so clone copies just these.
        return {"a": self.a, "allow_nd": self.allow_nd}


# XXX: use 2D array, since 1D X is being detected as a single sample in
# check_consistent_length
X = np.ones((15, 2))
y = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 6])
# The number of samples per class needs to be > n_splits,
# for StratifiedKFold(n_splits=3)
y2 = np.array([1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3])
P = np.eye(5)


@pytest.mark.parametrize("coo_container", COO_CONTAINERS)
def test_cross_val_score(coo_container):
    # [FUNCTION DOC]
    # Verifies cross_val_score with MockClassifier returns its constant score for dense and
    # COO sparse X with 1-D and multi-output y, that list inputs are preserved (checked by
    # CheckingClassifier), that 3-D X works with allow_nd and raises ValueError otherwise
    # with error_score="raise".
    clf = MockClassifier()
    X_sparse = coo_container(X)

    for a in range(-10, 10):
        clf.a = a
        # Smoke test
        scores = cross_val_score(clf, X, y2)
        assert_array_equal(scores, clf.score(X, y2))

        # test with multioutput y
        multioutput_y = np.column_stack([y2, y2[::-1]])
        scores = cross_val_score(clf, X_sparse, multioutput_y)
        assert_array_equal(scores, clf.score(X_sparse, multioutput_y))

        scores = cross_val_score(clf, X_sparse, y2)
        assert_array_equal(scores, clf.score(X_sparse, y2))

        # test with multioutput y
        scores = cross_val_score(clf, X_sparse, multioutput_y)
        assert_array_equal(scores, clf.score(X_sparse, multioutput_y))

    # test with X and y as list
    list_check = lambda x: isinstance(x, list)
    clf = CheckingClassifier(check_X=list_check)
    scores = cross_val_score(clf, X.tolist(), y2.tolist(), cv=3)

    clf = CheckingClassifier(check_y=list_check)
    scores = cross_val_score(clf, X, y2.tolist(), cv=3)

    # test with 3d X and
    X_3d = X[:, :, np.newaxis]
    clf = MockClassifier(allow_nd=True)
    scores = cross_val_score(clf, X_3d, y2)

    clf = MockClassifier(allow_nd=False)
    with pytest.raises(ValueError):
        cross_val_score(clf, X_3d, y2, error_score="raise")


def test_cross_validate_many_jobs():
    # [FUNCTION DOC]
    # Non-regression (#12154): cross_validate of a GridSearchCV with n_jobs=2 must not fail.
    # Smoke test only.
    # regression test for #12154: cv='warn' with n_jobs>1 trigger a copy of
    # the parameters leading to a failure in check_cv due to cv is 'warn'
    # instead of cv == 'warn'.
    X, y = load_iris(return_X_y=True)
    clf = SVC(gamma="auto")
    grid = GridSearchCV(clf, param_grid={"C": [1, 10]})
    cross_validate(grid, X, y, n_jobs=2)


def test_cross_validate_invalid_scoring_param():
    # [FUNCTION DOC]
    # Checks ValueError messages for invalid scoring specifications (tuples of callables,
    # empty tuple, duplicates, nested lists, empty dict) and that a scorer returning
    # multiple values triggers a UserWarning "Scoring failed" (score set to nan), both as
    # single scorer and inside a dict.
    X, y = make_classification(random_state=0)
    estimator = MockClassifier()

    # Test the errors
    error_message_regexp = ".*must be unique strings.*"

    # List/tuple of callables should raise a message advising users to use
    # dict of names to callables mapping
    with pytest.raises(ValueError, match=error_message_regexp):
        cross_validate(
            estimator,
            X,
            y,
            scoring=(make_scorer(precision_score), make_scorer(accuracy_score)),
        )
    with pytest.raises(ValueError, match=error_message_regexp):
        cross_validate(estimator, X, y, scoring=(make_scorer(precision_score),))

    # So should empty lists/tuples
    with pytest.raises(ValueError, match=error_message_regexp + "Empty list.*"):
        cross_validate(estimator, X, y, scoring=())

    # So should duplicated entries
    with pytest.raises(ValueError, match=error_message_regexp + "Duplicate.*"):
        cross_validate(estimator, X, y, scoring=("f1_micro", "f1_micro"))

    # Nested Lists should raise a generic error message
    with pytest.raises(ValueError, match=error_message_regexp):
        cross_validate(estimator, X, y, scoring=[[make_scorer(precision_score)]])

    # Empty dict should raise invalid scoring error
    with pytest.raises(ValueError, match="An empty dict"):
        cross_validate(estimator, X, y, scoring=(dict()))

    multiclass_scorer = make_scorer(precision_recall_fscore_support)

    # Multiclass Scorers that return multiple values are not supported yet
    # the warning message we're expecting to see
    warning_message = (
        "Scoring failed. The score on this train-test "
        f"partition for these parameters will be set to {np.nan}. "
        "Details: \n"
    )

    with pytest.warns(UserWarning, match=warning_message):
        cross_validate(estimator, X, y, scoring=multiclass_scorer)

    with pytest.warns(UserWarning, match=warning_message):
        cross_validate(estimator, X, y, scoring={"foo": multiclass_scorer})


def test_cross_validate_array_function_not_called():
    # [FUNCTION DOC]
    # Checks cross_validate works on _NotAnArray inputs, i.e. does not trigger NEP18
    # __array_function__ (which would raise).
    """Check that `__array_function__` (NEP18) is not called."""
    X = _NotAnArray([[1, 1], [1, 2], [1, 3], [1, 4], [2, 1], [2, 2], [2, 3], [2, 4]])
    y = _NotAnArray([1, 1, 1, 2, 2, 2, 1, 1])
    estimator = LogisticRegression(random_state=0)
    cross_validate(estimator, X, y, cv=2)


def test_cross_validate_nested_estimator():
    # [FUNCTION DOC]
    # Checks return_estimator=True returns a list of fitted Pipeline objects.
    # Non-regression test to ensure that nested
    # estimators are properly returned in a list
    # https://github.com/scikit-learn/scikit-learn/pull/17745
    (X, y) = load_iris(return_X_y=True)
    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer()),
            ("classifier", MockClassifier()),
        ]
    )

    results = cross_validate(pipeline, X, y, return_estimator=True)
    estimators = results["estimator"]

    assert isinstance(estimators, list)
    assert all(isinstance(estimator, Pipeline) for estimator in estimators)


@pytest.mark.parametrize("use_sparse", [False, True])
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_cross_validate(use_sparse: bool, csr_container):
    # [FUNCTION DOC]
    # Computes manual per-fold train/test MSE and R2 scores and fitted estimators with KFold
    # for a Ridge regressor and linear SVC (dense or CSR), then compares them to
    # cross_validate output via the single- and multi-metric check helpers.
    # Compute train and test mse/r2 scores
    cv = KFold()

    # Regression
    X_reg, y_reg = make_regression(n_samples=30, random_state=0)
    reg = Ridge(random_state=0)

    # Classification
    X_clf, y_clf = make_classification(n_samples=30, random_state=0)
    clf = SVC(kernel="linear", random_state=0)

    if use_sparse:
        X_reg = csr_container(X_reg)
        X_clf = csr_container(X_clf)

    for X, y, est in ((X_reg, y_reg, reg), (X_clf, y_clf, clf)):
        # It's okay to evaluate regression metrics on classification too
        mse_scorer = check_scoring(est, scoring="neg_mean_squared_error")
        r2_scorer = check_scoring(est, scoring="r2")
        train_mse_scores = []
        test_mse_scores = []
        train_r2_scores = []
        test_r2_scores = []
        fitted_estimators = []

        for train, test in cv.split(X, y):
            est = clone(est).fit(X[train], y[train])
            train_mse_scores.append(mse_scorer(est, X[train], y[train]))
            train_r2_scores.append(r2_scorer(est, X[train], y[train]))
            test_mse_scores.append(mse_scorer(est, X[test], y[test]))
            test_r2_scores.append(r2_scorer(est, X[test], y[test]))
            fitted_estimators.append(est)

        train_mse_scores = np.array(train_mse_scores)
        test_mse_scores = np.array(test_mse_scores)
        train_r2_scores = np.array(train_r2_scores)
        test_r2_scores = np.array(test_r2_scores)
        fitted_estimators = np.array(fitted_estimators)

        scores = (
            train_mse_scores,
            test_mse_scores,
            train_r2_scores,
            test_r2_scores,
            fitted_estimators,
        )

        # To ensure that the test does not suffer from
        # large statistical fluctuations due to slicing small datasets,
        # we pass the cross-validation instance
        check_cross_validate_single_metric(est, X, y, scores, cv)
        check_cross_validate_multi_metric(est, X, y, scores, cv)


def check_cross_validate_single_metric(clf, X, y, scores, cv):
    # [FUNCTION DOC]
    # HELPER: For return_train_score True/False, checks that cross_validate with a string
    # scorer and with a one-element list returns dicts of expected length and scores equal
    # to the precomputed ones; then checks return_estimator gives estimators with the same
    # coef_ (densified if sparse) and intercept_.
    # Inputs: clf, X, y, scores tuple (train/test mse, train/test r2, fitted estimators), cv.
    (
        train_mse_scores,
        test_mse_scores,
        train_r2_scores,
        test_r2_scores,
        fitted_estimators,
    ) = scores
    # Test single metric evaluation when scoring is string or singleton list
    for return_train_score, dict_len in ((True, 4), (False, 3)):
        # Single metric passed as a string
        if return_train_score:
            mse_scores_dict = cross_validate(
                clf,
                X,
                y,
                scoring="neg_mean_squared_error",
                return_train_score=True,
                cv=cv,
            )
            assert_array_almost_equal(mse_scores_dict["train_score"], train_mse_scores)
        else:
            mse_scores_dict = cross_validate(
                clf,
                X,
                y,
                scoring="neg_mean_squared_error",
                return_train_score=False,
                cv=cv,
            )
        assert isinstance(mse_scores_dict, dict)
        assert len(mse_scores_dict) == dict_len
        assert_array_almost_equal(mse_scores_dict["test_score"], test_mse_scores)

        # Single metric passed as a list
        if return_train_score:
            # It must be True by default - deprecated
            r2_scores_dict = cross_validate(
                clf, X, y, scoring=["r2"], return_train_score=True, cv=cv
            )
            assert_array_almost_equal(r2_scores_dict["train_r2"], train_r2_scores, True)
        else:
            r2_scores_dict = cross_validate(
                clf, X, y, scoring=["r2"], return_train_score=False, cv=cv
            )
        assert isinstance(r2_scores_dict, dict)
        assert len(r2_scores_dict) == dict_len
        assert_array_almost_equal(r2_scores_dict["test_r2"], test_r2_scores)

    # Test return_estimator option
    mse_scores_dict = cross_validate(
        clf, X, y, scoring="neg_mean_squared_error", return_estimator=True, cv=cv
    )
    for k, est in enumerate(mse_scores_dict["estimator"]):
        est_coef = est.coef_.copy()
        if issparse(est_coef):
            est_coef = est_coef.toarray()

        fitted_est_coef = fitted_estimators[k].coef_.copy()
        if issparse(fitted_est_coef):
            fitted_est_coef = fitted_est_coef.toarray()

        assert_almost_equal(est_coef, fitted_est_coef)
        assert_almost_equal(est.intercept_, fitted_estimators[k].intercept_)


def check_cross_validate_multi_metric(clf, X, y, scores, cv):
    # [FUNCTION DOC]
    # HELPER: For tuple, dict and callable-returning-dict scorings, with and without train
    # scores, checks the result keys, score values, ndarray types, and that fit/score times
    # are in [0, 10).
    # Test multimetric evaluation when scoring is a list / dict
    (
        train_mse_scores,
        test_mse_scores,
        train_r2_scores,
        test_r2_scores,
        fitted_estimators,
    ) = scores

    def custom_scorer(clf, X, y):
        # [FUNCTION DOC]
        # HELPER: Multi-metric callable scorer returning r2 and negative MSE of clf predictions.
        y_pred = clf.predict(X)
        return {
            "r2": r2_score(y, y_pred),
            "neg_mean_squared_error": -mean_squared_error(y, y_pred),
        }

    all_scoring = (
        ("r2", "neg_mean_squared_error"),
        {
            "r2": make_scorer(r2_score),
            "neg_mean_squared_error": "neg_mean_squared_error",
        },
        custom_scorer,
    )

    keys_sans_train = {
        "test_r2",
        "test_neg_mean_squared_error",
        "fit_time",
        "score_time",
    }
    keys_with_train = keys_sans_train.union(
        {"train_r2", "train_neg_mean_squared_error"}
    )

    for return_train_score in (True, False):
        for scoring in all_scoring:
            if return_train_score:
                # return_train_score must be True by default - deprecated
                cv_results = cross_validate(
                    clf, X, y, scoring=scoring, return_train_score=True, cv=cv
                )
                assert_array_almost_equal(cv_results["train_r2"], train_r2_scores)
                assert_array_almost_equal(
                    cv_results["train_neg_mean_squared_error"], train_mse_scores
                )
            else:
                cv_results = cross_validate(
                    clf, X, y, scoring=scoring, return_train_score=False, cv=cv
                )
            assert isinstance(cv_results, dict)
            assert set(cv_results.keys()) == (
                keys_with_train if return_train_score else keys_sans_train
            )
            assert_array_almost_equal(cv_results["test_r2"], test_r2_scores)
            assert_array_almost_equal(
                cv_results["test_neg_mean_squared_error"], test_mse_scores
            )

            # Make sure all the arrays are of np.ndarray type
            assert isinstance(cv_results["test_r2"], np.ndarray)
            assert isinstance(cv_results["test_neg_mean_squared_error"], np.ndarray)
            assert isinstance(cv_results["fit_time"], np.ndarray)
            assert isinstance(cv_results["score_time"], np.ndarray)

            # Ensure all the times are within sane limits
            assert np.all(cv_results["fit_time"] >= 0)
            assert np.all(cv_results["fit_time"] < 10)
            assert np.all(cv_results["score_time"] >= 0)
            assert np.all(cv_results["score_time"] < 10)


def test_cross_val_score_predict_groups():
    # [FUNCTION DOC]
    # Checks that group-based splitters raise ValueError "groups ... should not be None"
    # from both cross_val_score and cross_val_predict when groups is omitted.
    # Check if ValueError (when groups is None) propagates to cross_val_score
    # and cross_val_predict
    # And also check if groups is correctly passed to the cv object
    X, y = make_classification(n_samples=20, n_classes=2, random_state=0)

    clf = SVC(kernel="linear")

    group_cvs = [
        LeaveOneGroupOut(),
        LeavePGroupsOut(2),
        GroupKFold(),
        GroupShuffleSplit(),
    ]
    error_message = "The 'groups' parameter should not be None."
    for cv in group_cvs:
        with pytest.raises(ValueError, match=error_message):
            cross_val_score(estimator=clf, X=X, y=y, cv=cv)
        with pytest.raises(ValueError, match=error_message):
            cross_val_predict(estimator=clf, X=X, y=y, cv=cv)


def test_cross_val_score_pandas():
    # [FUNCTION DOC]
    # Checks cross_val_score keeps DataFrame/Series types (MockDataFrame, and pandas if
    # installed) when passing them to the estimator, via CheckingClassifier.
    # check cross_val_score doesn't destroy pandas dataframe
    types = [(MockDataFrame, MockDataFrame)]
    try:
        from pandas import DataFrame, Series

        types.append((Series, DataFrame))
    except ImportError:
        pass
    for TargetType, InputFeatureType in types:
        # X dataframe, y series
        # 3 fold cross val is used so we need at least 3 samples per class
        X_df, y_ser = InputFeatureType(X), TargetType(y2)
        check_df = lambda x: isinstance(x, InputFeatureType)
        check_series = lambda x: isinstance(x, TargetType)
        clf = CheckingClassifier(check_X=check_df, check_y=check_series)
        cross_val_score(clf, X_df, y_ser, cv=3)


def test_cross_val_score_mask():
    # [FUNCTION DOC]
    # Intended to check boolean-mask cv splits give the same scores as index splits.
    # KNOWN ISSUE: the mask arrays are built but cv_masks stores (train, test) index arrays,
    # so masks are never actually tested.
    # test that cross_val_score works with boolean masks
    svm = SVC(kernel="linear")
    iris = load_iris()
    X, y = iris.data, iris.target
    kfold = KFold(5)
    scores_indices = cross_val_score(svm, X, y, cv=kfold)
    kfold = KFold(5)
    cv_masks = []
    for train, test in kfold.split(X, y):
        mask_train = np.zeros(len(y), dtype=bool)
        mask_test = np.zeros(len(y), dtype=bool)
        mask_train[train] = 1
        mask_test[test] = 1
        cv_masks.append((train, test))
    scores_masks = cross_val_score(svm, X, y, cv=cv_masks)
    assert_array_equal(scores_indices, scores_masks)


def test_cross_val_score_precomputed():
    # [FUNCTION DOC]
    # Checks a precomputed linear kernel and a callable kernel give the same scores as
    # kernel="linear", and that ValueError is raised for non-square X and for a list kernel.
    # test for svm with precomputed kernel
    svm = SVC(kernel="precomputed")
    iris = load_iris()
    X, y = iris.data, iris.target
    linear_kernel = np.dot(X, X.T)
    score_precomputed = cross_val_score(svm, linear_kernel, y)
    svm = SVC(kernel="linear")
    score_linear = cross_val_score(svm, X, y)
    assert_array_almost_equal(score_precomputed, score_linear)

    # test with callable
    svm = SVC(kernel=lambda x, y: np.dot(x, y.T))
    score_callable = cross_val_score(svm, X, y)
    assert_array_almost_equal(score_precomputed, score_callable)

    # Error raised for non-square X
    svm = SVC(kernel="precomputed")
    with pytest.raises(ValueError):
        cross_val_score(svm, X, y)

    # test error is raised when the precomputed kernel is not array-like
    # or sparse
    with pytest.raises(ValueError):
        cross_val_score(svm, linear_kernel.tolist(), y)


@pytest.mark.parametrize("coo_container", COO_CONTAINERS)
def test_cross_val_score_fit_params(coo_container):
    # [FUNCTION DOC]
    # Passes array, sparse and non-array params (int, str, object, callback) through
    # cross_val_score params and checks MockClassifier receives them correctly (shape
    # asserts inside fit; the callback asserts the non-array values).
    clf = MockClassifier()
    n_samples = X.shape[0]
    n_classes = len(np.unique(y))

    W_sparse = coo_container(
        (np.array([1]), (np.array([1]), np.array([0]))), shape=(15, 1)
    )
    P_sparse = coo_container(np.eye(5))

    DUMMY_INT = 42
    DUMMY_STR = "42"
    DUMMY_OBJ = object()

    def assert_fit_params(clf):
        # [FUNCTION DOC]
        # HELPER: Callback asserting the classifier stored the dummy int/str/object unchanged.
        # Function to test that the values are passed correctly to the
        # classifier arguments for non-array type

        assert clf.dummy_int == DUMMY_INT
        assert clf.dummy_str == DUMMY_STR
        assert clf.dummy_obj == DUMMY_OBJ

    fit_params = {
        "sample_weight": np.ones(n_samples),
        "class_prior": np.full(n_classes, 1.0 / n_classes),
        "sparse_sample_weight": W_sparse,
        "sparse_param": P_sparse,
        "dummy_int": DUMMY_INT,
        "dummy_str": DUMMY_STR,
        "dummy_obj": DUMMY_OBJ,
        "callback": assert_fit_params,
    }
    cross_val_score(clf, X, y2, params=fit_params)


def test_cross_val_score_score_func():
    # [FUNCTION DOC]
    # Checks a custom make_scorer function is called exactly once per fold (cv=3) and its
    # return value is reported.
    clf = MockClassifier()
    _score_func_args = []

    def score_func(y_test, y_predict):
        # [FUNCTION DOC]
        # HELPER: Records its arguments and returns 1.0.
        _score_func_args.append((y_test, y_predict))
        return 1.0

    with warnings.catch_warnings(record=True):
        scoring = make_scorer(score_func)
        score = cross_val_score(clf, X, y, scoring=scoring, cv=3)
    assert_array_equal(score, [1.0, 1.0, 1.0])
    # Test that score function is called only 3 times (for cv=3)
    assert len(_score_func_args) == 3


def test_cross_val_score_with_score_func_classification():
    # [FUNCTION DOC]
    # Checks default, accuracy and f1_weighted scores of linear SVC on iris match expected
    # per-fold values.
    iris = load_iris()
    clf = SVC(kernel="linear")

    # Default score (should be the accuracy score)
    scores = cross_val_score(clf, iris.data, iris.target)
    assert_array_almost_equal(scores, [0.97, 1.0, 0.97, 0.97, 1.0], 2)

    # Correct classification score (aka. zero / one score) - should be the
    # same as the default estimator score
    zo_scores = cross_val_score(clf, iris.data, iris.target, scoring="accuracy")
    assert_array_almost_equal(zo_scores, [0.97, 1.0, 0.97, 0.97, 1.0], 2)

    # F1 score (class are balanced so f1_score should be equal to zero/one
    # score
    f1_scores = cross_val_score(clf, iris.data, iris.target, scoring="f1_weighted")
    assert_array_almost_equal(f1_scores, [0.97, 1.0, 0.97, 0.97, 1.0], 2)


def test_cross_val_score_with_score_func_regression():
    # [FUNCTION DOC]
    # Checks default, r2, neg_mean_squared_error and explained variance scores of Ridge on a
    # synthetic regression problem match expected per-fold values.
    X, y = make_regression(n_samples=30, n_features=20, n_informative=5, random_state=0)
    reg = Ridge()

    # Default score of the Ridge regression estimator
    scores = cross_val_score(reg, X, y)
    assert_array_almost_equal(scores, [0.94, 0.97, 0.97, 0.99, 0.92], 2)

    # R2 score (aka. determination coefficient) - should be the
    # same as the default estimator score
    r2_scores = cross_val_score(reg, X, y, scoring="r2")
    assert_array_almost_equal(r2_scores, [0.94, 0.97, 0.97, 0.99, 0.92], 2)

    # Mean squared error; this is a loss function, so "scores" are negative
    neg_mse_scores = cross_val_score(reg, X, y, scoring="neg_mean_squared_error")
    expected_neg_mse = np.array([-763.07, -553.16, -274.38, -273.26, -1681.99])
    assert_array_almost_equal(neg_mse_scores, expected_neg_mse, 2)

    # Explained variance
    scoring = make_scorer(explained_variance_score)
    ev_scores = cross_val_score(reg, X, y, scoring=scoring)
    assert_array_almost_equal(ev_scores, [0.94, 0.97, 0.97, 0.99, 0.92], 2)


@pytest.mark.parametrize("coo_container", COO_CONTAINERS)
def test_permutation_score(coo_container):
    # [FUNCTION DOC]
    # Checks permutation_test_score on iris: high score / low p-value, identical results
    # with constant groups and with COO sparse X, expected values with a custom scorer, and
    # low score / high p-value on non-informative labels.
    iris = load_iris()
    X = iris.data
    X_sparse = coo_container(X)
    y = iris.target
    svm = SVC(kernel="linear")
    cv = StratifiedKFold(2)

    score, scores, pvalue = permutation_test_score(
        svm, X, y, n_permutations=30, cv=cv, scoring="accuracy"
    )
    assert score > 0.9
    assert_almost_equal(pvalue, 0.0, 1)

    score_group, _, pvalue_group = permutation_test_score(
        svm,
        X,
        y,
        n_permutations=30,
        cv=cv,
        scoring="accuracy",
        groups=np.ones(y.size),
        random_state=0,
    )
    assert score_group == score
    assert pvalue_group == pvalue

    # check that we obtain the same results with a sparse representation
    svm_sparse = SVC(kernel="linear")
    cv_sparse = StratifiedKFold(2)
    score_group, _, pvalue_group = permutation_test_score(
        svm_sparse,
        X_sparse,
        y,
        n_permutations=30,
        cv=cv_sparse,
        scoring="accuracy",
        groups=np.ones(y.size),
        random_state=0,
    )

    assert score_group == score
    assert pvalue_group == pvalue

    # test with custom scoring object
    def custom_score(y_true, y_pred):
        # [FUNCTION DOC]
        # HELPER: Returns (correct - incorrect) / n_samples.
        return ((y_true == y_pred).sum() - (y_true != y_pred).sum()) / y_true.shape[0]

    scorer = make_scorer(custom_score)
    score, _, pvalue = permutation_test_score(
        svm, X, y, n_permutations=100, scoring=scorer, cv=cv, random_state=0
    )
    assert_almost_equal(score, 0.93, 2)
    assert_almost_equal(pvalue, 0.01, 3)

    # set random y
    y = np.mod(np.arange(len(y)), 3)

    score, scores, pvalue = permutation_test_score(
        svm, X, y, n_permutations=30, cv=cv, scoring="accuracy"
    )

    assert score < 0.5
    assert pvalue > 0.2


def test_permutation_test_score_allow_nans():
    # [FUNCTION DOC]
    # Smoke test: permutation_test_score with an imputing pipeline accepts NaN input.
    # Check that permutation_test_score allows input data with NaNs
    X = np.arange(200, dtype=np.float64).reshape(10, -1)
    X[2, :] = np.nan
    y = np.repeat([0, 1], X.shape[0] / 2)
    p = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="mean", missing_values=np.nan)),
            ("classifier", MockClassifier()),
        ]
    )
    permutation_test_score(p, X, y)


def test_permutation_test_score_params():
    # [FUNCTION DOC]
    # Checks sample_weight routing in permutation_test_score: missing weights raise
    # AssertionError (CheckingClassifier), wrongly sized weights raise ValueError, correct
    # weights work.
    X = np.arange(100).reshape(10, 10)
    y = np.array([0] * 5 + [1] * 5)
    clf = CheckingClassifier(expected_sample_weight=True)

    err_msg = r"Expected sample_weight to be passed"
    with pytest.raises(AssertionError, match=err_msg):
        permutation_test_score(clf, X, y)

    err_msg = r"sample_weight.shape == \(1,\), expected \(8,\)!"
    with pytest.raises(ValueError, match=err_msg):
        permutation_test_score(clf, X, y, params={"sample_weight": np.ones(1)})
    permutation_test_score(clf, X, y, params={"sample_weight": np.ones(10)})


def test_cross_val_score_allow_nans():
    # [FUNCTION DOC]
    # Smoke test: cross_val_score with an imputing pipeline accepts NaN input.
    # Check that cross_val_score allows input data with NaNs
    X = np.arange(200, dtype=np.float64).reshape(10, -1)
    X[2, :] = np.nan
    y = np.repeat([0, 1], X.shape[0] / 2)
    p = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="mean", missing_values=np.nan)),
            ("classifier", MockClassifier()),
        ]
    )
    cross_val_score(p, X, y)


def test_cross_val_score_multilabel():
    # [FUNCTION DOC]
    # Checks precision with micro, macro and samples averaging on a small multilabel
    # dataset with 1-NN gives the hand-computed per-fold values.
    X = np.array(
        [
            [-3, 4],
            [2, 4],
            [3, 3],
            [0, 2],
            [-3, 1],
            [-2, 1],
            [0, 0],
            [-2, -1],
            [-1, -2],
            [1, -2],
        ]
    )
    y = np.array(
        [[1, 1], [0, 1], [0, 1], [0, 1], [1, 1], [0, 1], [1, 0], [1, 1], [1, 0], [0, 0]]
    )
    clf = KNeighborsClassifier(n_neighbors=1)
    scoring_micro = make_scorer(precision_score, average="micro")
    scoring_macro = make_scorer(precision_score, average="macro")
    scoring_samples = make_scorer(precision_score, average="samples")
    score_micro = cross_val_score(clf, X, y, scoring=scoring_micro)
    score_macro = cross_val_score(clf, X, y, scoring=scoring_macro)
    score_samples = cross_val_score(clf, X, y, scoring=scoring_samples)
    assert_almost_equal(score_micro, [1, 1 / 2, 3 / 4, 1 / 2, 1 / 3])
    assert_almost_equal(score_macro, [1, 1 / 2, 3 / 4, 1 / 2, 1 / 4])
    assert_almost_equal(score_samples, [1, 1 / 2, 3 / 4, 1 / 2, 1 / 4])


@pytest.mark.parametrize("coo_container", COO_CONTAINERS)
def test_cross_val_predict(coo_container):
    # [FUNCTION DOC]
    # Checks cross_val_predict for Ridge on diabetes equals a manual KFold loop, returns
    # one prediction per sample for default cv, LeaveOneOut and COO sparse X, works for
    # unsupervised KMeans, raises ValueError when the cv does not partition the samples
    # (BadCV), and warns (RuntimeWarning) when a training fold misses classes with
    # predict_proba.
    X, y = load_diabetes(return_X_y=True)
    cv = KFold()

    est = Ridge()

    # Naive loop (should be same as cross_val_predict):
    preds2 = np.zeros_like(y)
    for train, test in cv.split(X, y):
        est.fit(X[train], y[train])
        preds2[test] = est.predict(X[test])

    preds = cross_val_predict(est, X, y, cv=cv)
    assert_array_almost_equal(preds, preds2)

    preds = cross_val_predict(est, X, y)
    assert len(preds) == len(y)

    cv = LeaveOneOut()
    preds = cross_val_predict(est, X, y, cv=cv)
    assert len(preds) == len(y)

    Xsp = X.copy()
    Xsp *= Xsp > np.median(Xsp)
    Xsp = coo_container(Xsp)
    preds = cross_val_predict(est, Xsp, y)
    assert_array_almost_equal(len(preds), len(y))

    preds = cross_val_predict(KMeans(n_init="auto"), X)
    assert len(preds) == len(y)

    class BadCV:
        # [CLASS DOC]
        # HELPER: Splitter yielding the same non-covering train/test pair four times.
        def split(self, X, y=None, groups=None):
            # [FUNCTION DOC]
            # HELPER: Yields train indices 0..3 and test indices 4..8, four times; ignores inputs.
            for i in range(4):
                yield np.array([0, 1, 2, 3]), np.array([4, 5, 6, 7, 8])

    with pytest.raises(ValueError):
        cross_val_predict(est, X, y, cv=BadCV())

    X, y = load_iris(return_X_y=True)

    warning_message = (
        r"Number of classes in training fold \(2\) does "
        r"not match total number of classes \(3\). "
        "Results may not be appropriate for your use case."
    )
    with pytest.warns(RuntimeWarning, match=warning_message):
        cross_val_predict(
            LogisticRegression(solver="liblinear"),
            X,
            y,
            method="predict_proba",
            cv=KFold(2),
        )


def test_cross_val_predict_decision_function_shape():
    # [FUNCTION DOC]
    # Checks decision_function output shapes: (n,) for binary, (n, 3) for iris, (n, 45) for
    # OvO SVC on digits; and ValueError messages for a binary fold with a single class and
    # for an OvO fold whose output width does not match the classes in the fold.
    X, y = make_classification(n_classes=2, n_samples=50, random_state=0)

    preds = cross_val_predict(LogisticRegression(), X, y, method="decision_function")
    assert preds.shape == (50,)

    X, y = load_iris(return_X_y=True)

    preds = cross_val_predict(LogisticRegression(), X, y, method="decision_function")
    assert preds.shape == (150, 3)

    # This specifically tests imbalanced splits for binary
    # classification with decision_function. This is only
    # applicable to classifiers that can be fit on a single
    # class.
    X = X[:100]
    y = y[:100]
    error_message = (
        "Only 1 class/es in training fold,"
        " but 2 in overall dataset. This"
        " is not supported for decision_function"
        " with imbalanced folds. To fix "
        "this, use a cross-validation technique "
        "resulting in properly stratified folds"
    )
    with pytest.raises(ValueError, match=error_message):
        cross_val_predict(
            RidgeClassifier(), X, y, method="decision_function", cv=KFold(2)
        )

    X, y = load_digits(return_X_y=True)
    est = SVC(kernel="linear", decision_function_shape="ovo")

    preds = cross_val_predict(est, X, y, method="decision_function")
    assert preds.shape == (1797, 45)

    ind = np.argsort(y)
    X, y = X[ind], y[ind]
    error_message_regexp = (
        r"Output shape \(599L?, 21L?\) of "
        "decision_function does not match number of "
        r"classes \(7\) in fold. Irregular "
        "decision_function .*"
    )
    with pytest.raises(ValueError, match=error_message_regexp):
        cross_val_predict(est, X, y, cv=KFold(n_splits=3), method="decision_function")


def test_cross_val_predict_predict_proba_shape():
    # [FUNCTION DOC]
    # Checks predict_proba output shape is (n, n_classes) for binary and 3-class data.
    X, y = make_classification(n_classes=2, n_samples=50, random_state=0)

    preds = cross_val_predict(LogisticRegression(), X, y, method="predict_proba")
    assert preds.shape == (50, 2)

    X, y = load_iris(return_X_y=True)

    preds = cross_val_predict(LogisticRegression(), X, y, method="predict_proba")
    assert preds.shape == (150, 3)


def test_cross_val_predict_predict_log_proba_shape():
    # [FUNCTION DOC]
    # Checks predict_log_proba output shape is (n, n_classes) for binary and 3-class data.
    X, y = make_classification(n_classes=2, n_samples=50, random_state=0)

    preds = cross_val_predict(LogisticRegression(), X, y, method="predict_log_proba")
    assert preds.shape == (50, 2)

    X, y = load_iris(return_X_y=True)

    preds = cross_val_predict(LogisticRegression(), X, y, method="predict_log_proba")
    assert preds.shape == (150, 3)


@pytest.mark.parametrize("coo_container", COO_CONTAINERS)
def test_cross_val_predict_input_types(coo_container):
    # [FUNCTION DOC]
    # Checks cross_val_predict output shapes for dense/COO X with 1-D and 2-D y, that list
    # X/y are passed unchanged (CheckingClassifier), that decision_function works with list
    # inputs, and that 3-D X is passed through.
    iris = load_iris()
    X, y = iris.data, iris.target
    X_sparse = coo_container(X)
    multioutput_y = np.column_stack([y, y[::-1]])

    clf = Ridge(fit_intercept=False, random_state=0)
    # 3 fold cv is used --> at least 3 samples per class
    # Smoke test
    predictions = cross_val_predict(clf, X, y)
    assert predictions.shape == (150,)

    # test with multioutput y
    predictions = cross_val_predict(clf, X_sparse, multioutput_y)
    assert predictions.shape == (150, 2)

    predictions = cross_val_predict(clf, X_sparse, y)
    assert_array_equal(predictions.shape, (150,))

    # test with multioutput y
    predictions = cross_val_predict(clf, X_sparse, multioutput_y)
    assert_array_equal(predictions.shape, (150, 2))

    # test with X and y as list
    list_check = lambda x: isinstance(x, list)
    clf = CheckingClassifier(check_X=list_check)
    predictions = cross_val_predict(clf, X.tolist(), y.tolist())

    clf = CheckingClassifier(check_y=list_check)
    predictions = cross_val_predict(clf, X, y.tolist())

    # test with X and y as list and non empty method
    predictions = cross_val_predict(
        LogisticRegression(),
        X.tolist(),
        y.tolist(),
        method="decision_function",
    )
    predictions = cross_val_predict(
        LogisticRegression(),
        X,
        y.tolist(),
        method="decision_function",
    )

    # test with 3d X and
    X_3d = X[:, :, np.newaxis]
    check_3d = lambda x: x.ndim == 3
    clf = CheckingClassifier(check_X=check_3d)
    predictions = cross_val_predict(clf, X_3d, y)
    assert_array_equal(predictions.shape, (150,))


def test_cross_val_predict_pandas():
    # [FUNCTION DOC]
    # Checks cross_val_predict keeps DataFrame/Series types (MockDataFrame and pandas if
    # available) when passed to the estimator.
    # check cross_val_score doesn't destroy pandas dataframe
    types = [(MockDataFrame, MockDataFrame)]
    try:
        from pandas import DataFrame, Series

        types.append((Series, DataFrame))
    except ImportError:
        pass
    for TargetType, InputFeatureType in types:
        # X dataframe, y series
        X_df, y_ser = InputFeatureType(X), TargetType(y2)
        check_df = lambda x: isinstance(x, InputFeatureType)
        check_series = lambda x: isinstance(x, TargetType)
        clf = CheckingClassifier(check_X=check_df, check_y=check_series)
        cross_val_predict(clf, X_df, y_ser, cv=3)


def test_cross_val_predict_unbalanced():
    # [FUNCTION DOC]
    # Checks predict_proba with a class present only in one fold: that class gets
    # probability 0 in the other fold's predictions, other classes are positive and rows
    # sum to 1.
    X, y = make_classification(
        n_samples=100,
        n_features=2,
        n_redundant=0,
        n_informative=2,
        n_clusters_per_class=1,
        random_state=1,
    )
    # Change the first sample to a new class
    y[0] = 2
    clf = LogisticRegression(random_state=1)
    cv = StratifiedKFold(n_splits=2)
    train, test = list(cv.split(X, y))
    yhat_proba = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")
    assert y[test[0]][0] == 2  # sanity check for further assertions
    assert np.all(yhat_proba[test[0]][:, 2] == 0)
    assert np.all(yhat_proba[test[0]][:, 0:1] > 0)
    assert np.all(yhat_proba[test[1]] > 0)
    assert_array_almost_equal(yhat_proba.sum(axis=1), np.ones(y.shape), decimal=12)


def test_cross_val_predict_y_none():
    # [FUNCTION DOC]
    # Checks cross_val_predict works with y=None: predict equals the first column of X and
    # predict_proba equals X (MockClassifier behavior).
    # ensure that cross_val_predict works when y is None
    mock_classifier = MockClassifier()
    rng = np.random.RandomState(42)
    X = rng.rand(100, 10)
    y_hat = cross_val_predict(mock_classifier, X, y=None, cv=5, method="predict")
    assert_allclose(X[:, 0], y_hat)
    y_hat_proba = cross_val_predict(
        mock_classifier, X, y=None, cv=5, method="predict_proba"
    )
    assert_allclose(X, y_hat_proba)


@pytest.mark.parametrize("coo_container", COO_CONTAINERS)
def test_cross_val_score_sparse_fit_params(coo_container):
    # [FUNCTION DOC]
    # Checks a sparse sample-aligned fit param (COO identity) is correctly sliced per fold;
    # MockClassifier asserts its row count and scores are all 1.
    iris = load_iris()
    X, y = iris.data, iris.target
    clf = MockClassifier()
    fit_params = {"sparse_sample_weight": coo_container(np.eye(X.shape[0]))}
    a = cross_val_score(clf, X, y, params=fit_params, cv=3)
    assert_array_equal(a, np.ones(3))


def test_learning_curve():
    # [FUNCTION DOC]
    # Checks learning_curve with MockImprovingEstimator (with and without shuffle) emits no
    # warnings, returns arrays of shape (10, 3) for scores and times (float64), train sizes
    # 2..20 and linearly varying mean train/test scores; and that a one-time splitter
    # (OneTimeSplitter) gives the same scores, proving cv.split is consumed once.
    n_samples = 30
    n_splits = 3
    X, y = make_classification(
        n_samples=n_samples,
        n_features=1,
        n_informative=1,
        n_redundant=0,
        n_classes=2,
        n_clusters_per_class=1,
        random_state=0,
    )
    estimator = MockImprovingEstimator(n_samples * ((n_splits - 1) / n_splits))
    for shuffle_train in [False, True]:
        with warnings.catch_warnings(record=True) as w:
            (
                train_sizes,
                train_scores,
                test_scores,
                fit_times,
                score_times,
            ) = learning_curve(
                estimator,
                X,
                y,
                cv=KFold(n_splits=n_splits),
                train_sizes=np.linspace(0.1, 1.0, 10),
                shuffle=shuffle_train,
                return_times=True,
            )
        if len(w) > 0:
            raise RuntimeError("Unexpected warning: %r" % w[0].message)
        assert train_scores.shape == (10, 3)
        assert test_scores.shape == (10, 3)
        assert fit_times.shape == (10, 3)
        assert score_times.shape == (10, 3)
        assert_array_equal(train_sizes, np.linspace(2, 20, 10))
        assert_array_almost_equal(train_scores.mean(axis=1), np.linspace(1.9, 1.0, 10))
        assert_array_almost_equal(test_scores.mean(axis=1), np.linspace(0.1, 1.0, 10))

        # Cannot use assert_array_almost_equal for fit and score times because
        # the values are hardware-dependent
        assert fit_times.dtype == "float64"
        assert score_times.dtype == "float64"

        # Test a custom cv splitter that can iterate only once
        with warnings.catch_warnings(record=True) as w:
            train_sizes2, train_scores2, test_scores2 = learning_curve(
                estimator,
                X,
                y,
                cv=OneTimeSplitter(n_splits=n_splits, n_samples=n_samples),
                train_sizes=np.linspace(0.1, 1.0, 10),
                shuffle=shuffle_train,
            )
        if len(w) > 0:
            raise RuntimeError("Unexpected warning: %r" % w[0].message)
        assert_array_almost_equal(train_scores2, train_scores)
        assert_array_almost_equal(test_scores2, test_scores)


def test_learning_curve_unsupervised():
    # [FUNCTION DOC]
    # Same expectations as test_learning_curve with y=None and cv=3.
    X, _ = make_classification(
        n_samples=30,
        n_features=1,
        n_informative=1,
        n_redundant=0,
        n_classes=2,
        n_clusters_per_class=1,
        random_state=0,
    )
    estimator = MockImprovingEstimator(20)
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y=None, cv=3, train_sizes=np.linspace(0.1, 1.0, 10)
    )
    assert_array_equal(train_sizes, np.linspace(2, 20, 10))
    assert_array_almost_equal(train_scores.mean(axis=1), np.linspace(1.9, 1.0, 10))
    assert_array_almost_equal(test_scores.mean(axis=1), np.linspace(0.1, 1.0, 10))


def test_learning_curve_verbose(capsys):
    # [FUNCTION DOC]
    # Checks verbose=1 prints a "[learning_curve]" line to stdout (captured by capsys).
    X, y = make_classification(
        n_samples=30,
        n_features=1,
        n_informative=1,
        n_redundant=0,
        n_classes=2,
        n_clusters_per_class=1,
        random_state=0,
    )
    estimator = MockImprovingEstimator(20)
    learning_curve(estimator, X, y, cv=3, verbose=1)
    assert "[learning_curve]" in capsys.readouterr().out


def test_learning_curve_incremental_learning_not_possible():
    # [FUNCTION DOC]
    # Checks exploit_incremental_learning=True raises ValueError for an estimator without
    # partial_fit.
    X, y = make_classification(
        n_samples=2,
        n_features=1,
        n_informative=1,
        n_redundant=0,
        n_classes=2,
        n_clusters_per_class=1,
        random_state=0,
    )
    # The mockup does not have partial_fit()
    estimator = MockImprovingEstimator(1)
    with pytest.raises(ValueError):
        learning_curve(estimator, X, y, exploit_incremental_learning=True)


def test_learning_curve_incremental_learning():
    # [FUNCTION DOC]
    # Checks incremental learning (partial_fit) yields the expected sizes and linear score
    # means, with and without shuffle.
    X, y = make_classification(
        n_samples=30,
        n_features=1,
        n_informative=1,
        n_redundant=0,
        n_classes=2,
        n_clusters_per_class=1,
        random_state=0,
    )
    estimator = MockIncrementalImprovingEstimator(20)
    for shuffle_train in [False, True]:
        train_sizes, train_scores, test_scores = learning_curve(
            estimator,
            X,
            y,
            cv=3,
            exploit_incremental_learning=True,
            train_sizes=np.linspace(0.1, 1.0, 10),
            shuffle=shuffle_train,
        )
        assert_array_equal(train_sizes, np.linspace(2, 20, 10))
        assert_array_almost_equal(train_scores.mean(axis=1), np.linspace(1.9, 1.0, 10))
        assert_array_almost_equal(test_scores.mean(axis=1), np.linspace(0.1, 1.0, 10))


def test_learning_curve_incremental_learning_unsupervised():
    # [FUNCTION DOC]
    # Same as the incremental test but with y=None.
    X, _ = make_classification(
        n_samples=30,
        n_features=1,
        n_informative=1,
        n_redundant=0,
        n_classes=2,
        n_clusters_per_class=1,
        random_state=0,
    )
    estimator = MockIncrementalImprovingEstimator(20)
    train_sizes, train_scores, test_scores = learning_curve(
        estimator,
        X,
        y=None,
        cv=3,
        exploit_incremental_learning=True,
        train_sizes=np.linspace(0.1, 1.0, 10),
    )
    assert_array_equal(train_sizes, np.linspace(2, 20, 10))
    assert_array_almost_equal(train_scores.mean(axis=1), np.linspace(1.9, 1.0, 10))
    assert_array_almost_equal(test_scores.mean(axis=1), np.linspace(0.1, 1.0, 10))


def test_learning_curve_batch_and_incremental_learning_are_equal():
    # [FUNCTION DOC]
    # Checks SGDClassifier with max_iter=1 and no shuffling gives the same mean scores in
    # batch and incremental modes.
    X, y = make_classification(
        n_samples=30,
        n_features=1,
        n_informative=1,
        n_redundant=0,
        n_classes=2,
        n_clusters_per_class=1,
        random_state=0,
    )
    train_sizes = np.linspace(0.2, 1.0, 5)
    estimator = SGDClassifier(max_iter=1, tol=None, shuffle=False)

    train_sizes_inc, train_scores_inc, test_scores_inc = learning_curve(
        estimator,
        X,
        y,
        train_sizes=train_sizes,
        cv=3,
        exploit_incremental_learning=True,
    )
    train_sizes_batch, train_scores_batch, test_scores_batch = learning_curve(
        estimator,
        X,
        y,
        cv=3,
        train_sizes=train_sizes,
        exploit_incremental_learning=False,
    )

    assert_array_equal(train_sizes_inc, train_sizes_batch)
    assert_array_almost_equal(
        train_scores_inc.mean(axis=1), train_scores_batch.mean(axis=1)
    )
    assert_array_almost_equal(
        test_scores_inc.mean(axis=1), test_scores_batch.mean(axis=1)
    )


def test_learning_curve_n_sample_range_out_of_bounds():
    # [FUNCTION DOC]
    # Checks ValueError for train_sizes containing 0, 0.0, fractions > 1, or absolute sizes
    # above the maximum training size.
    X, y = make_classification(
        n_samples=30,
        n_features=1,
        n_informative=1,
        n_redundant=0,
        n_classes=2,
        n_clusters_per_class=1,
        random_state=0,
    )
    estimator = MockImprovingEstimator(20)
    with pytest.raises(ValueError):
        learning_curve(estimator, X, y, cv=3, train_sizes=[0, 1])
    with pytest.raises(ValueError):
        learning_curve(estimator, X, y, cv=3, train_sizes=[0.0, 1.0])
    with pytest.raises(ValueError):
        learning_curve(estimator, X, y, cv=3, train_sizes=[0.1, 1.1])
    with pytest.raises(ValueError):
        learning_curve(estimator, X, y, cv=3, train_sizes=[0, 20])
    with pytest.raises(ValueError):
        learning_curve(estimator, X, y, cv=3, train_sizes=[1, 21])


def test_learning_curve_remove_duplicate_sample_sizes():
    # [FUNCTION DOC]
    # Checks duplicate absolute sizes produced from fractions are removed with a
    # RuntimeWarning and train_sizes becomes [1, 2].
    X, y = make_classification(
        n_samples=3,
        n_features=1,
        n_informative=1,
        n_redundant=0,
        n_classes=2,
        n_clusters_per_class=1,
        random_state=0,
    )
    estimator = MockImprovingEstimator(2)
    warning_message = (
        "Removed duplicate entries from 'train_sizes'. Number of ticks "
        "will be less than the size of 'train_sizes': 2 instead of 3."
    )
    with pytest.warns(RuntimeWarning, match=warning_message):
        train_sizes, _, _ = learning_curve(
            estimator, X, y, cv=3, train_sizes=np.linspace(0.33, 1.0, 3)
        )
    assert_array_equal(train_sizes, [1, 2])


def test_learning_curve_with_boolean_indices():
    # [FUNCTION DOC]
    # Checks learning_curve with an explicit KFold gives the expected sizes and scores.
    # KNOWN ISSUE: despite the name, no boolean index masks are used; it duplicates the
    # plain KFold case.
    X, y = make_classification(
        n_samples=30,
        n_features=1,
        n_informative=1,
        n_redundant=0,
        n_classes=2,
        n_clusters_per_class=1,
        random_state=0,
    )
    estimator = MockImprovingEstimator(20)
    cv = KFold(n_splits=3)
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, train_sizes=np.linspace(0.1, 1.0, 10)
    )
    assert_array_equal(train_sizes, np.linspace(2, 20, 10))
    assert_array_almost_equal(train_scores.mean(axis=1), np.linspace(1.9, 1.0, 10))
    assert_array_almost_equal(test_scores.mean(axis=1), np.linspace(0.1, 1.0, 10))


def test_learning_curve_with_shuffle():
    # [FUNCTION DOC]
    # Non-regression for PR #7506: with GroupKFold splits whose first increments miss a
    # class, shuffle=True gives expected mean scores, the unshuffled run fails with
    # ValueError under error_score="raise", and incremental learning matches batch scores.
    # Following test case was designed this way to verify the code
    # changes made in pull request: #7506.
    X = np.array(
        [
            [1, 2],
            [3, 4],
            [5, 6],
            [7, 8],
            [11, 12],
            [13, 14],
            [15, 16],
            [17, 18],
            [19, 20],
            [7, 8],
            [9, 10],
            [11, 12],
            [13, 14],
            [15, 16],
            [17, 18],
        ]
    )
    y = np.array([1, 1, 1, 2, 3, 4, 1, 1, 2, 3, 4, 1, 2, 3, 4])
    groups = np.array([1, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 4, 4, 4, 4])
    # Splits on these groups fail without shuffle as the first iteration
    # of the learning curve doesn't contain label 4 in the training set.
    estimator = SGDClassifier(
        max_iter=5, tol=None, shuffle=False, learning_rate="pa1", eta0=1
    )

    cv = GroupKFold(n_splits=2)
    train_sizes_batch, train_scores_batch, test_scores_batch = learning_curve(
        estimator,
        X,
        y,
        cv=cv,
        n_jobs=1,
        train_sizes=np.linspace(0.3, 1.0, 3),
        groups=groups,
        shuffle=True,
        random_state=2,
    )
    assert_array_almost_equal(
        train_scores_batch.mean(axis=1), np.array([0.75, 0.3, 0.36111111])
    )
    assert_array_almost_equal(
        test_scores_batch.mean(axis=1), np.array([0.36111111, 0.25, 0.25])
    )
    with pytest.raises(ValueError):
        learning_curve(
            estimator,
            X,
            y,
            cv=cv,
            n_jobs=1,
            train_sizes=np.linspace(0.3, 1.0, 3),
            groups=groups,
            error_score="raise",
        )

    train_sizes_inc, train_scores_inc, test_scores_inc = learning_curve(
        estimator,
        X,
        y,
        cv=cv,
        n_jobs=1,
        train_sizes=np.linspace(0.3, 1.0, 3),
        groups=groups,
        shuffle=True,
        random_state=2,
        exploit_incremental_learning=True,
    )
    assert_array_almost_equal(
        train_scores_inc.mean(axis=1), train_scores_batch.mean(axis=1)
    )
    assert_array_almost_equal(
        test_scores_inc.mean(axis=1), test_scores_batch.mean(axis=1)
    )


def test_learning_curve_params():
    # [FUNCTION DOC]
    # Checks sample_weight routing in learning_curve: missing weights raise AssertionError,
    # wrongly sized weights raise ValueError, correct weights work.
    X = np.arange(100).reshape(10, 10)
    y = np.array([0] * 5 + [1] * 5)
    clf = CheckingClassifier(expected_sample_weight=True)

    err_msg = r"Expected sample_weight to be passed"
    with pytest.raises(AssertionError, match=err_msg):
        learning_curve(clf, X, y, error_score="raise")

    err_msg = r"sample_weight.shape == \(1,\), expected \(2,\)!"
    with pytest.raises(ValueError, match=err_msg):
        learning_curve(
            clf, X, y, error_score="raise", params={"sample_weight": np.ones(1)}
        )
    learning_curve(
        clf, X, y, error_score="raise", params={"sample_weight": np.ones(10)}
    )


def test_learning_curve_incremental_learning_params():
    # [FUNCTION DOC]
    # Checks that in incremental mode, missing expected fit params make
    # MockIncrementalImprovingEstimator raise AssertionError (error_score="raise").
    X, y = make_classification(
        n_samples=30,
        n_features=1,
        n_informative=1,
        n_redundant=0,
        n_classes=2,
        n_clusters_per_class=1,
        random_state=0,
    )
    estimator = MockIncrementalImprovingEstimator(20, ["sample_weight"])
    err_msg = r"Expected fit parameter\(s\) \['sample_weight'\] not seen."
    with pytest.raises(AssertionError, match=err_msg):
        learning_curve(
            estimator,
            X,
            y,
            cv=3,
            exploit_incremental_learning=True,
            train_sizes=np.linspace(0.1, 1.0, 10),
            error_score="raise",
        )


def test_validation_curve():
    # [FUNCTION DOC]
    # Checks validation_curve with MockEstimatorWithParameter emits no warnings and that
    # mean train/test scores equal param_range and 1 - param_range.
    X, y = make_classification(
        n_samples=2,
        n_features=1,
        n_informative=1,
        n_redundant=0,
        n_classes=2,
        n_clusters_per_class=1,
        random_state=0,
    )
    param_range = np.linspace(0, 1, 10)
    with warnings.catch_warnings(record=True) as w:
        train_scores, test_scores = validation_curve(
            MockEstimatorWithParameter(),
            X,
            y,
            param_name="param",
            param_range=param_range,
            cv=2,
        )
    if len(w) > 0:
        raise RuntimeError("Unexpected warning: %r" % w[0].message)

    assert_array_almost_equal(train_scores.mean(axis=1), param_range)
    assert_array_almost_equal(test_scores.mean(axis=1), 1 - param_range)


def test_validation_curve_clone_estimator():
    # [FUNCTION DOC]
    # Checks validation_curve clones the estimator for every fit (the mock raises on a
    # second fit of the same instance).
    X, y = make_classification(
        n_samples=2,
        n_features=1,
        n_informative=1,
        n_redundant=0,
        n_classes=2,
        n_clusters_per_class=1,
        random_state=0,
    )

    param_range = np.linspace(1, 0, 10)
    _, _ = validation_curve(
        MockEstimatorWithSingleFitCallAllowed(),
        X,
        y,
        param_name="param",
        param_range=param_range,
        cv=2,
    )


def test_validation_curve_cv_splits_consistency():
    # [FUNCTION DOC]
    # Checks the same splits are used for every parameter value: with OneTimeSplitter and a
    # shuffled KFold, duplicated C values give identical scores; unshuffled KFold matches
    # OneTimeSplitter results.
    n_samples = 100
    n_splits = 5
    X, y = make_classification(n_samples=100, random_state=0)

    scores1 = validation_curve(
        SVC(kernel="linear", random_state=0),
        X,
        y,
        param_name="C",
        param_range=[0.1, 0.1, 0.2, 0.2],
        cv=OneTimeSplitter(n_splits=n_splits, n_samples=n_samples),
    )
    # The OneTimeSplitter is a non-re-entrant cv splitter. Unless, the
    # `split` is called for each parameter, the following should produce
    # identical results for param setting 1 and param setting 2 as both have
    # the same C value.
    assert_array_almost_equal(*np.vsplit(np.hstack(scores1)[(0, 2, 1, 3), :], 2))

    scores2 = validation_curve(
        SVC(kernel="linear", random_state=0),
        X,
        y,
        param_name="C",
        param_range=[0.1, 0.1, 0.2, 0.2],
        cv=KFold(n_splits=n_splits, shuffle=True),
    )

    # For scores2, compare the 1st and 2nd parameter's scores
    # (Since the C value for 1st two param setting is 0.1, they must be
    # consistent unless the train test folds differ between the param settings)
    assert_array_almost_equal(*np.vsplit(np.hstack(scores2)[(0, 2, 1, 3), :], 2))

    scores3 = validation_curve(
        SVC(kernel="linear", random_state=0),
        X,
        y,
        param_name="C",
        param_range=[0.1, 0.1, 0.2, 0.2],
        cv=KFold(n_splits=n_splits),
    )

    # OneTimeSplitter is basically unshuffled KFold(n_splits=5). Sanity check.
    assert_array_almost_equal(np.array(scores3), np.array(scores1))


def test_validation_curve_params():
    # [FUNCTION DOC]
    # Checks sample_weight routing in validation_curve: missing weights raise
    # AssertionError, wrongly sized weights raise ValueError, correct weights work.
    X = np.arange(100).reshape(10, 10)
    y = np.array([0] * 5 + [1] * 5)
    clf = CheckingClassifier(expected_sample_weight=True)

    err_msg = r"Expected sample_weight to be passed"
    with pytest.raises(AssertionError, match=err_msg):
        validation_curve(
            clf,
            X,
            y,
            param_name="foo_param",
            param_range=[1, 2, 3],
            error_score="raise",
        )

    err_msg = r"sample_weight.shape == \(1,\), expected \(8,\)!"
    with pytest.raises(ValueError, match=err_msg):
        validation_curve(
            clf,
            X,
            y,
            param_name="foo_param",
            param_range=[1, 2, 3],
            error_score="raise",
            params={"sample_weight": np.ones(1)},
        )
    validation_curve(
        clf,
        X,
        y,
        param_name="foo_param",
        param_range=[1, 2, 3],
        error_score="raise",
        params={"sample_weight": np.ones(10)},
    )


def test_check_is_permutation():
    # [FUNCTION DOC]
    # Checks _check_is_permutation accepts a shuffled arange and rejects arrays with a
    # missing element, a duplicated element, or an extra duplicate index.
    rng = np.random.RandomState(0)
    p = np.arange(100)
    rng.shuffle(p)
    assert _check_is_permutation(p, 100)
    assert not _check_is_permutation(np.delete(p, 23), 100)

    p[0] = 23
    assert not _check_is_permutation(p, 100)

    # Check if the additional duplicate indices are caught
    assert not _check_is_permutation(np.hstack((p, 0)), 100)


@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_cross_val_predict_sparse_prediction(csr_container):
    # [FUNCTION DOC]
    # Checks OvR multilabel cross_val_predict gives the same predictions for CSR sparse X/y
    # (densified) as for dense inputs.
    # check that cross_val_predict gives same result for sparse and dense input
    X, y = make_multilabel_classification(
        n_classes=2,
        n_labels=1,
        allow_unlabeled=False,
        return_indicator=True,
        random_state=1,
    )
    X_sparse = csr_container(X)
    y_sparse = csr_container(y)
    classif = OneVsRestClassifier(SVC(kernel="linear"))
    preds = cross_val_predict(classif, X, y, cv=10)
    preds_sparse = cross_val_predict(classif, X_sparse, y_sparse, cv=10)
    preds_sparse = preds_sparse.toarray()
    assert_array_almost_equal(preds_sparse, preds)


def check_cross_val_predict_binary(est, X, y, method):
    # [FUNCTION DOC]
    # HELPER: Builds expected out-of-fold outputs of method with a manual unshuffled
    # 3-fold KFold loop (shape (n,) for binary decision_function, (n, 2) otherwise, or
    # y.shape for 2-D y), then checks cross_val_predict matches for y, y+1, y-2 and string
    # labels.
    # Inputs: est - estimator; X, y; method - prediction method name.
    """Helper for tests of cross_val_predict with binary classification"""
    cv = KFold(n_splits=3, shuffle=False)

    # Generate expected outputs
    if y.ndim == 1:
        exp_shape = (len(X),) if method == "decision_function" else (len(X), 2)
    else:
        exp_shape = y.shape
    expected_predictions = np.zeros(exp_shape)
    for train, test in cv.split(X, y):
        est = clone(est).fit(X[train], y[train])
        expected_predictions[test] = getattr(est, method)(X[test])

    # Check actual outputs for several representations of y
    for tg in [y, y + 1, y - 2, y.astype("str")]:
        assert_allclose(
            cross_val_predict(est, X, tg, method=method, cv=cv), expected_predictions
        )


def check_cross_val_predict_multiclass(est, X, y, method):
    # [FUNCTION DOC]
    # HELPER: Builds expected out-of-fold outputs for multiclass y: columns for classes
    # absent from a training fold keep the default (float64 min for decision_function and
    # predict_log_proba, 0 for predict_proba); then compares cross_val_predict for several
    # label encodings.
    """Helper for tests of cross_val_predict with multiclass classification"""
    cv = KFold(n_splits=3, shuffle=False)

    # Generate expected outputs
    float_min = np.finfo(np.float64).min
    default_values = {
        "decision_function": float_min,
        "predict_log_proba": float_min,
        "predict_proba": 0,
    }
    expected_predictions = np.full(
        (len(X), len(set(y))), default_values[method], dtype=np.float64
    )
    _, y_enc = np.unique(y, return_inverse=True)
    for train, test in cv.split(X, y_enc):
        est = clone(est).fit(X[train], y_enc[train])
        fold_preds = getattr(est, method)(X[test])
        i_cols_fit = np.unique(y_enc[train])
        expected_predictions[np.ix_(test, i_cols_fit)] = fold_preds

    # Check actual outputs for several representations of y
    for tg in [y, y + 1, y - 2, y.astype("str")]:
        assert_allclose(
            cross_val_predict(est, X, tg, method=method, cv=cv), expected_predictions
        )


def check_cross_val_predict_multilabel(est, X, y, method):
    # [FUNCTION DOC]
    # HELPER: For 2-D targets with estimators returning a list of per-output arrays, builds
    # expected per-output arrays (1-D for binary decision_function), fills columns of the
    # classes seen in each training fold, and checks every element of the cross_val_predict
    # output list for several label encodings.
    """Check the output of cross_val_predict for 2D targets using
    Estimators which provide a predictions as a list with one
    element per class.
    """
    cv = KFold(n_splits=3, shuffle=False)

    # Create empty arrays of the correct size to hold outputs
    float_min = np.finfo(np.float64).min
    default_values = {
        "decision_function": float_min,
        "predict_log_proba": float_min,
        "predict_proba": 0,
    }
    n_targets = y.shape[1]
    expected_preds = []
    for i_col in range(n_targets):
        n_classes_in_label = len(set(y[:, i_col]))
        if n_classes_in_label == 2 and method == "decision_function":
            exp_shape = (len(X),)
        else:
            exp_shape = (len(X), n_classes_in_label)
        expected_preds.append(
            np.full(exp_shape, default_values[method], dtype=np.float64)
        )

    # Generate expected outputs
    y_enc_cols = [
        np.unique(y[:, i], return_inverse=True)[1][:, np.newaxis]
        for i in range(y.shape[1])
    ]
    y_enc = np.concatenate(y_enc_cols, axis=1)
    for train, test in cv.split(X, y_enc):
        est = clone(est).fit(X[train], y_enc[train])
        fold_preds = getattr(est, method)(X[test])
        for i_col in range(n_targets):
            fold_cols = np.unique(y_enc[train][:, i_col])
            if expected_preds[i_col].ndim == 1:
                # Decision function with <=2 classes
                expected_preds[i_col][test] = fold_preds[i_col]
            else:
                idx = np.ix_(test, fold_cols)
                expected_preds[i_col][idx] = fold_preds[i_col]

    # Check actual outputs for several representations of y
    for tg in [y, y + 1, y - 2, y.astype("str")]:
        cv_predict_output = cross_val_predict(est, X, tg, method=method, cv=cv)
        assert len(cv_predict_output) == len(expected_preds)
        for i in range(len(cv_predict_output)):
            assert_allclose(cv_predict_output[i], expected_preds[i])


def check_cross_val_predict_with_method_binary(est):
    # [FUNCTION DOC]
    # HELPER: Runs check_cross_val_predict_binary for decision_function, predict_proba and
    # predict_log_proba on a binary make_classification problem.
    # This test includes the decision_function with two classes.
    # This is a special case: it has only one column of output.
    X, y = make_classification(n_classes=2, random_state=0)
    for method in ["decision_function", "predict_proba", "predict_log_proba"]:
        check_cross_val_predict_binary(est, X, y, method)


def check_cross_val_predict_with_method_multiclass(est):
    # [FUNCTION DOC]
    # HELPER: Runs check_cross_val_predict_multiclass for the three probability/decision
    # methods on shuffled iris.
    iris = load_iris()
    X, y = iris.data, iris.target
    X, y = shuffle(X, y, random_state=0)
    for method in ["decision_function", "predict_proba", "predict_log_proba"]:
        check_cross_val_predict_multiclass(est, X, y, method)


def test_cross_val_predict_with_method():
    # [FUNCTION DOC]
    # Runs the binary and multiclass method checks with LogisticRegression.
    check_cross_val_predict_with_method_binary(LogisticRegression())
    check_cross_val_predict_with_method_multiclass(LogisticRegression())


def test_cross_val_predict_method_checking():
    # [FUNCTION DOC]
    # Non-regression for #9639: cross_val_predict must not check prediction methods before
    # fitting; runs multiclass checks with SGDClassifier(loss="log_loss").
    # Regression test for issue #9639. Tests that cross_val_predict does not
    # check estimator methods (e.g. predict_proba) before fitting
    iris = load_iris()
    X, y = iris.data, iris.target
    X, y = shuffle(X, y, random_state=0)
    for method in ["decision_function", "predict_proba", "predict_log_proba"]:
        est = SGDClassifier(loss="log_loss", random_state=2)
        check_cross_val_predict_multiclass(est, X, y, method)


def test_gridsearchcv_cross_val_predict_with_method():
    # [FUNCTION DOC]
    # Runs multiclass method checks with a GridSearchCV-wrapped LogisticRegression.
    iris = load_iris()
    X, y = iris.data, iris.target
    X, y = shuffle(X, y, random_state=0)
    est = GridSearchCV(LogisticRegression(random_state=42), {"C": [0.1, 1]}, cv=2)
    for method in ["decision_function", "predict_proba", "predict_log_proba"]:
        check_cross_val_predict_multiclass(est, X, y, method)


def test_cross_val_predict_with_method_multilabel_ovr():
    # [FUNCTION DOC]
    # Checks OvR multilabel predict_proba / decision_function outputs (2-D arrays) via the
    # binary check helper.
    # OVR does multilabel predictions, but only arrays of
    # binary indicator columns. The output of predict_proba
    # is a 2D array with shape (n_samples, n_classes).
    n_samp = 100
    n_classes = 4
    X, y = make_multilabel_classification(
        n_samples=n_samp, n_labels=3, n_classes=n_classes, n_features=5, random_state=42
    )
    est = OneVsRestClassifier(LogisticRegression(solver="liblinear", random_state=0))
    for method in ["predict_proba", "decision_function"]:
        check_cross_val_predict_binary(est, X, y, method=method)


class RFWithDecisionFunction(RandomForestClassifier):
    # [CLASS DOC]
    # HELPER: RandomForestClassifier with a fake decision_function for multioutput tasks.
    # None of the current multioutput-multiclass estimators have
    # decision function methods. Create a mock decision function
    # to test the cross_val_predict function's handling of this case.
    def decision_function(self, X):
        # [FUNCTION DOC]
        # HELPER: Returns the list of per-output predict_proba arrays, reducing binary outputs
        # to the last-class column.
        # Exceptions: AssertionError if predict_proba does not return a list (single output).
        probs = self.predict_proba(X)
        msg = "This helper should only be used on multioutput-multiclass tasks"
        assert isinstance(probs, list), msg
        probs = [p[:, -1] if p.shape[1] == 2 else p for p in probs]
        return probs


def test_cross_val_predict_with_method_multilabel_rf():
    # [FUNCTION DOC]
    # Checks multioutput-multiclass random forest predictions (3 classes in the first
    # column) for predict_proba, predict_log_proba and decision_function via the multilabel
    # helper, with warnings suppressed.
    # The RandomForest allows multiple classes in each label.
    # Output of predict_proba is a list of outputs of predict_proba
    # for each individual label.
    n_classes = 4
    X, y = make_multilabel_classification(
        n_samples=100, n_labels=3, n_classes=n_classes, n_features=5, random_state=42
    )
    y[:, 0] += y[:, 1]  # Put three classes in the first column
    for method in ["predict_proba", "predict_log_proba", "decision_function"]:
        est = RFWithDecisionFunction(n_estimators=5, random_state=0)
        with warnings.catch_warnings():
            # Suppress "RuntimeWarning: divide by zero encountered in log"
            warnings.simplefilter("ignore")
            check_cross_val_predict_multilabel(est, X, y, method=method)


def test_cross_val_predict_with_method_rare_class():
    # [FUNCTION DOC]
    # Checks multiclass predictions when one class is missing from a training fold use the
    # default fill values for that class.
    # Test a multiclass problem where one class will be missing from
    # one of the CV training sets.
    rng = np.random.RandomState(0)
    X = rng.normal(0, 1, size=(14, 10))
    y = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 3])
    est = LogisticRegression()
    for method in ["predict_proba", "predict_log_proba", "decision_function"]:
        with warnings.catch_warnings():
            # Suppress warning about too few examples of a class
            warnings.simplefilter("ignore")
            check_cross_val_predict_multiclass(est, X, y, method)


def test_cross_val_predict_with_method_multilabel_rf_rare_class():
    # [FUNCTION DOC]
    # Same as the rf multilabel test but with a class having one example, so some training
    # fold lacks it.
    # The RandomForest allows anything for the contents of the labels.
    # Output of predict_proba is a list of outputs of predict_proba
    # for each individual label.
    # In this test, the first label has a class with a single example.
    # We'll have one CV fold where the training data don't include it.
    rng = np.random.RandomState(0)
    X = rng.normal(0, 1, size=(5, 10))
    y = np.array([[0, 0], [1, 1], [2, 1], [0, 1], [1, 0]])
    for method in ["predict_proba", "predict_log_proba"]:
        est = RFWithDecisionFunction(n_estimators=5, random_state=0)
        with warnings.catch_warnings():
            # Suppress "RuntimeWarning: divide by zero encountered in log"
            warnings.simplefilter("ignore")
            check_cross_val_predict_multilabel(est, X, y, method=method)


def get_expected_predictions(X, y, cv, classes, est, method):
    # [FUNCTION DOC]
    # HELPER: Manual out-of-fold predictions: fits est on each training fold, places the
    # method outputs into the columns given by est.classes_ in a (len(test), classes) block
    # filled with 0 (predict_proba) or the float min, and assembles an (n_samples, classes)
    # array.
    # Inputs: X, y, cv splitter, classes - total number of classes, est (refitted in
    # place), method name.
    # Outputs: float array (n_samples, classes).
    # KNOWN ISSUE: relies on classes_ being integers 0..classes-1 to use them as column
    # indices.
    expected_predictions = np.zeros([len(y), classes])
    func = getattr(est, method)

    for train, test in cv.split(X, y):
        est.fit(X[train], y[train])
        expected_predictions_ = func(X[test])
        # To avoid 2 dimensional indexing
        if method == "predict_proba":
            exp_pred_test = np.zeros((len(test), classes))
        else:
            exp_pred_test = np.full(
                (len(test), classes), np.finfo(expected_predictions.dtype).min
            )
        exp_pred_test[:, est.classes_] = expected_predictions_
        expected_predictions[test] = exp_pred_test

    return expected_predictions


def test_cross_val_predict_class_subset():
    # [FUNCTION DOC]
    # Checks cross_val_predict with KFold(3)/KFold(4) on ordered labels (so training folds
    # miss classes) and on shuffled labels matches get_expected_predictions for each
    # method.
    # KNOWN ISSUE: y is overwritten with shuffled labels inside the methods loop, so only
    # the first method is tested with the ordered labels.
    X = np.arange(200).reshape(100, 2)
    y = np.array([x // 10 for x in range(100)])
    classes = 10

    kfold3 = KFold(n_splits=3)
    kfold4 = KFold(n_splits=4)

    le = LabelEncoder()

    methods = ["decision_function", "predict_proba", "predict_log_proba"]
    for method in methods:
        est = LogisticRegression()

        # Test with n_splits=3
        predictions = cross_val_predict(est, X, y, method=method, cv=kfold3)

        # Runs a naive loop (should be same as cross_val_predict):
        expected_predictions = get_expected_predictions(
            X, y, kfold3, classes, est, method
        )
        assert_array_almost_equal(expected_predictions, predictions)

        # Test with n_splits=4
        predictions = cross_val_predict(est, X, y, method=method, cv=kfold4)
        expected_predictions = get_expected_predictions(
            X, y, kfold4, classes, est, method
        )
        assert_array_almost_equal(expected_predictions, predictions)

        # Testing unordered labels
        y = shuffle(np.repeat(range(10), 10), random_state=0)
        predictions = cross_val_predict(est, X, y, method=method, cv=kfold3)
        y = le.fit_transform(y)
        expected_predictions = get_expected_predictions(
            X, y, kfold3, classes, est, method
        )
        assert_array_almost_equal(expected_predictions, predictions)


def test_score_memmap():
    # [FUNCTION DOC]
    # Checks a scorer returning a 0-d memmap is accepted while a non-scalar memmap raises
    # ValueError; cleans up the temp file with retries (Windows file locking).
    # Side effects: creates and deletes a temporary file.
    # Ensure a scalar score of memmap type is accepted
    iris = load_iris()
    X, y = iris.data, iris.target
    clf = MockClassifier()
    tf = tempfile.NamedTemporaryFile(mode="wb", delete=False)
    tf.write(b"Hello world!!!!!")
    tf.close()
    scores = np.memmap(tf.name, dtype=np.float64)
    score = np.memmap(tf.name, shape=(), mode="r", dtype=np.float64)
    try:
        cross_val_score(clf, X, y, scoring=lambda est, X, y: score)
        with pytest.raises(ValueError):
            cross_val_score(clf, X, y, scoring=lambda est, X, y: scores)
    finally:
        # Best effort to release the mmap file handles before deleting the
        # backing file under Windows
        scores, score = None, None
        for _ in range(3):
            try:
                os.unlink(tf.name)
                break
            except OSError:
                sleep(1.0)


def test_permutation_test_score_pandas():
    # [FUNCTION DOC]
    # Checks permutation_test_score keeps DataFrame/Series types when passed to the
    # estimator.
    # check permutation_test_score doesn't destroy pandas dataframe
    types = [(MockDataFrame, MockDataFrame)]
    try:
        from pandas import DataFrame, Series

        types.append((Series, DataFrame))
    except ImportError:
        pass
    for TargetType, InputFeatureType in types:
        # X dataframe, y series
        iris = load_iris()
        X, y = iris.data, iris.target
        X_df, y_ser = InputFeatureType(X), TargetType(y)
        check_df = lambda x: isinstance(x, InputFeatureType)
        check_series = lambda x: isinstance(x, TargetType)
        clf = CheckingClassifier(check_X=check_df, check_y=check_series)
        permutation_test_score(clf, X_df, y_ser)


def test_fit_and_score_failing():
    # [FUNCTION DOC]
    # Checks _fit_and_score with error_score="raise" re-raises the FailingClassifier
    # ValueError; also covers FailingClassifier.score.
    # Create a failing classifier to deliberately fail
    failing_clf = FailingClassifier(FailingClassifier.FAILING_PARAMETER)
    # dummy X data
    X = np.arange(1, 10)
    train, test = np.arange(0, 5), np.arange(5, 9)
    fit_and_score_args = dict(
        estimator=failing_clf,
        X=X,
        y=None,
        scorer=dict(),
        train=train,
        test=test,
        verbose=0,
        parameters=None,
        fit_params=None,
        score_params=None,
    )
    # passing error score to trigger the warning message
    fit_and_score_args["error_score"] = "raise"
    # check if exception was raised, with default error_score='raise'
    with pytest.raises(ValueError, match="Failing classifier failed as required"):
        _fit_and_score(**fit_and_score_args)

    assert failing_clf.score() == 0.0  # FailingClassifier coverage


def test_fit_and_score_working():
    # [FUNCTION DOC]
    # Checks return_parameters=True makes _fit_and_score return the given parameters.
    X, y = make_classification(n_samples=30, random_state=0)
    clf = SVC(kernel="linear", random_state=0)
    train, test = next(ShuffleSplit().split(X))
    # Test return_parameters option
    fit_and_score_args = dict(
        estimator=clf,
        X=X,
        y=y,
        scorer=dict(),
        train=train,
        test=test,
        verbose=0,
        parameters={"max_iter": 100, "tol": 0.1},
        fit_params=None,
        score_params=None,
        return_parameters=True,
    )
    result = _fit_and_score(**fit_and_score_args)
    assert result["parameters"] == fit_and_score_args["parameters"]


class DataDependentFailingClassifier(BaseEstimator):
    # [CLASS DOC]
    # HELPER: Estimator whose fit fails when any X value exceeds max_x_value; score is 0.
    def __init__(self, max_x_value=None):
        # [FUNCTION DOC]
        # HELPER: Stores max_x_value.
        self.max_x_value = max_x_value

    def fit(self, X, y=None):
        # [FUNCTION DOC]
        # HELPER: Raises ValueError reporting how many X values exceed max_x_value.
        # Outputs: None (does not return self).
        num_values_too_high = (X > self.max_x_value).sum()
        if num_values_too_high:
            raise ValueError(
                f"Classifier fit failed with {num_values_too_high} values too high"
            )

    def score(self, X=None, Y=None):
        # [FUNCTION DOC]
        # HELPER: Always 0.0.
        return 0.0


@pytest.mark.parametrize("error_score", [np.nan, 0])
def test_cross_validate_some_failing_fits_warning(error_score):
    # [FUNCTION DOC]
    # Checks that when 2 of 3 fits fail, cross_validate emits a FitFailedWarning summarizing
    # the failures, the error_score value and the individual error message.
    # Create a failing classifier to deliberately fail
    failing_clf = DataDependentFailingClassifier(max_x_value=8)
    # dummy X data
    X = np.arange(1, 10)
    y = np.ones(9)
    # passing error score to trigger the warning message
    cross_validate_args = [failing_clf, X, y]
    cross_validate_kwargs = {"cv": 3, "error_score": error_score}
    # check if the warning message type is as expected

    individual_fit_error_message = (
        "ValueError: Classifier fit failed with 1 values too high"
    )
    warning_message = re.compile(
        (
            "2 fits failed.+total of 3.+The score on these"
            " train-test partitions for these parameters will be set to"
            f" {cross_validate_kwargs['error_score']}.+{individual_fit_error_message}"
        ),
        flags=re.DOTALL,
    )

    with pytest.warns(FitFailedWarning, match=warning_message):
        cross_validate(*cross_validate_args, **cross_validate_kwargs)


@pytest.mark.parametrize("error_score", [np.nan, 0])
def test_cross_validate_all_failing_fits_error(error_score):
    # [FUNCTION DOC]
    # Checks that when all 7 fits fail, cross_validate raises ValueError saying all fits
    # failed and including the individual error message.
    # Create a failing classifier to deliberately fail
    failing_clf = FailingClassifier(FailingClassifier.FAILING_PARAMETER)
    # dummy X data
    X = np.arange(1, 10)
    y = np.ones(9)

    cross_validate_args = [failing_clf, X, y]
    cross_validate_kwargs = {"cv": 7, "error_score": error_score}

    individual_fit_error_message = "ValueError: Failing classifier failed as required"
    error_message = re.compile(
        (
            "All the 7 fits failed.+your model is misconfigured.+"
            f"{individual_fit_error_message}"
        ),
        flags=re.DOTALL,
    )

    with pytest.raises(ValueError, match=error_message):
        cross_validate(*cross_validate_args, **cross_validate_kwargs)


def _failing_scorer(estimator, X, y, error_msg):
    # [FUNCTION DOC]
    # HELPER: Scorer that always raises ValueError(error_msg).
    raise ValueError(error_msg)


@pytest.mark.filterwarnings("ignore:lbfgs failed to converge")
@pytest.mark.parametrize("error_score", [np.nan, 0, "raise"])
def test_cross_val_score_failing_scorer(error_score):
    # [FUNCTION DOC]
    # Checks a failing scorer re-raises with error_score="raise", otherwise emits a
    # "Scoring failed" UserWarning and returns error_score for every fold.
    # check that an estimator can fail during scoring in `cross_val_score` and
    # that we can optionally replaced it with `error_score`
    X, y = load_iris(return_X_y=True)
    clf = LogisticRegression(max_iter=5).fit(X, y)

    error_msg = "This scorer is supposed to fail!!!"
    failing_scorer = partial(_failing_scorer, error_msg=error_msg)

    if error_score == "raise":
        with pytest.raises(ValueError, match=error_msg):
            cross_val_score(
                clf, X, y, cv=3, scoring=failing_scorer, error_score=error_score
            )
    else:
        warning_msg = (
            "Scoring failed. The score on this train-test partition for "
            f"these parameters will be set to {error_score}"
        )
        with pytest.warns(UserWarning, match=warning_msg):
            scores = cross_val_score(
                clf, X, y, cv=3, scoring=failing_scorer, error_score=error_score
            )
            assert_allclose(scores, error_score)


@pytest.mark.filterwarnings("ignore:lbfgs failed to converge")
@pytest.mark.parametrize("error_score", [np.nan, 0, "raise"])
@pytest.mark.parametrize("return_train_score", [True, False])
@pytest.mark.parametrize("with_multimetric", [False, True])
def test_cross_validate_failing_scorer(
    error_score, return_train_score, with_multimetric
):
    # [FUNCTION DOC]
    # Checks failing scorers in cross_validate (single or multimetric, with and without
    # train scores): re-raise with error_score="raise"; otherwise a "Scoring failed"
    # UserWarning, failing scorer entries equal error_score and the non-failing scorer
    # (score_2) yields floats.
    # Check that an estimator can fail during scoring in `cross_validate` and
    # that we can optionally replace it with `error_score`. In the multimetric
    # case also check the result of a non-failing scorer where the other scorers
    # are failing.
    X, y = load_iris(return_X_y=True)
    clf = LogisticRegression(max_iter=5).fit(X, y)

    error_msg = "This scorer is supposed to fail!!!"
    failing_scorer = partial(_failing_scorer, error_msg=error_msg)
    if with_multimetric:
        non_failing_scorer = make_scorer(mean_squared_error)
        scoring = {
            "score_1": failing_scorer,
            "score_2": non_failing_scorer,
            "score_3": failing_scorer,
        }
    else:
        scoring = failing_scorer

    if error_score == "raise":
        with pytest.raises(ValueError, match=error_msg):
            cross_validate(
                clf,
                X,
                y,
                cv=3,
                scoring=scoring,
                return_train_score=return_train_score,
                error_score=error_score,
            )
    else:
        warning_msg = (
            "Scoring failed. The score on this train-test partition for "
            f"these parameters will be set to {error_score}"
        )
        with pytest.warns(UserWarning, match=warning_msg):
            results = cross_validate(
                clf,
                X,
                y,
                cv=3,
                scoring=scoring,
                return_train_score=return_train_score,
                error_score=error_score,
            )
            for key in results:
                if "_score" in key:
                    if "_score_2" in key:
                        # check the test (and optionally train) score for the
                        # scorer that should be non-failing
                        for i in results[key]:
                            assert isinstance(i, float)
                    else:
                        # check the test (and optionally train) score for all
                        # scorers that should be assigned to `error_score`.
                        assert_allclose(results[key], error_score)


def three_params_scorer(i, j, k):
    # [FUNCTION DOC]
    # HELPER: Scorer with signature (estimator, X, y) that always returns 3.4213.
    return 3.4213


@pytest.mark.parametrize(
    "train_score, scorer, verbose, split_prg, cdt_prg, expected",
    [
        (
            False,
            three_params_scorer,
            2,
            (1, 3),
            (0, 1),
            r"\[CV\] END ...................................................."
            r" total time=   0.\ds",
        ),
        (
            True,
            _MultimetricScorer(
                scorers={"sc1": three_params_scorer, "sc2": three_params_scorer}
            ),
            3,
            (1, 3),
            (0, 1),
            r"\[CV 2/3\] END  sc1: \(train=3.421, test=3.421\) sc2: "
            r"\(train=3.421, test=3.421\) total time=   0.\ds",
        ),
        (
            False,
            _MultimetricScorer(
                scorers={"sc1": three_params_scorer, "sc2": three_params_scorer}
            ),
            10,
            (1, 3),
            (0, 1),
            r"\[CV 2/3; 1/1\] END ....... sc1: \(test=3.421\) sc2: \(test=3.421\)"
            r" total time=   0.\ds",
        ),
    ],
)
def test_fit_and_score_verbosity(
    capsys, train_score, scorer, verbose, split_prg, cdt_prg, expected
):
    # [FUNCTION DOC]
    # Checks the "[CV ...] END" progress line printed by _fit_and_score for different
    # verbose levels, split/candidate progress, single vs multimetric scorers and
    # train-score reporting, by matching the captured stdout line against a regex (second
    # line when more than two lines are printed).
    X, y = make_classification(n_samples=30, random_state=0)
    clf = SVC(kernel="linear", random_state=0)
    train, test = next(ShuffleSplit().split(X))

    # test print without train score
    fit_and_score_args = dict(
        estimator=clf,
        X=X,
        y=y,
        scorer=scorer,
        train=train,
        test=test,
        verbose=verbose,
        parameters=None,
        fit_params=None,
        score_params=None,
        return_train_score=train_score,
        split_progress=split_prg,
        candidate_progress=cdt_prg,
    )
    _fit_and_score(**fit_and_score_args)
    out, _ = capsys.readouterr()
    outlines = out.split("\n")
    if len(outlines) > 2:
        assert re.match(expected, outlines[1])
    else:
        assert re.match(expected, outlines[0])


def test_score():
    # [FUNCTION DOC]
    # Checks _score raises ValueError "scoring must return a number, got None" when the
    # scorer returns None.
    error_message = "scoring must return a number, got None"

    def two_params_scorer(estimator, X_test):
        # [FUNCTION DOC]
        # HELPER: Scorer returning None.
        return None

    with pytest.raises(ValueError, match=error_message):
        _score(
            estimator=None,
            X_test=None,
            y_test=None,
            scorer=two_params_scorer,
            score_params=None,
            error_score=np.nan,
        )


def test_callable_multimetric_confusion_matrix_cross_validate():
    # [FUNCTION DOC]
    # Checks a callable scorer returning a dict of confusion-matrix entries produces
    # test_tn/fp/fn/tp keys in cross_validate results.
    def custom_scorer(clf, X, y):
        # [FUNCTION DOC]
        # HELPER: Returns tn, fp, fn, tp from the binary confusion matrix of clf predictions.
        y_pred = clf.predict(X)
        cm = confusion_matrix(y, y_pred)
        return {"tn": cm[0, 0], "fp": cm[0, 1], "fn": cm[1, 0], "tp": cm[1, 1]}

    X, y = make_classification(n_samples=40, n_features=4, random_state=42)
    est = LinearSVC(random_state=42)
    est.fit(X, y)
    cv_results = cross_validate(est, X, y, cv=5, scoring=custom_scorer)

    score_names = ["tn", "fp", "fn", "tp"]
    for name in score_names:
        assert "test_{}".format(name) in cv_results


def test_learning_curve_partial_fit_regressors():
    # [FUNCTION DOC]
    # Non-regression for #22981: incremental learning_curve with MLPRegressor must not fail.
    """Check that regressors with partial_fit is supported.

    Non-regression test for #22981.
    """
    X, y = make_regression(random_state=42)

    # Does not error
    learning_curve(MLPRegressor(), X, y, exploit_incremental_learning=True, cv=2)


def test_learning_curve_some_failing_fits_warning(global_random_seed):
    # [FUNCTION DOC]
    # Sorts targets so the two smallest training sizes contain one class (SVC fails),
    # then checks a FitFailedWarning "10 fits failed out of a total of 25" and that only the
    # first two train-size rows are all NaN.
    """Checks for fit failures in `learning_curve` and raises the required warning"""

    X, y = make_classification(
        n_samples=30,
        n_classes=3,
        n_informative=6,
        shuffle=False,
        random_state=global_random_seed,
    )
    # sorting the target to trigger SVC error on the 2 first splits because a single
    # class is present
    sorted_idx = np.argsort(y)
    X, y = X[sorted_idx], y[sorted_idx]

    svc = SVC()
    warning_message = "10 fits failed out of a total of 25"

    with pytest.warns(FitFailedWarning, match=warning_message):
        _, train_score, test_score, *_ = learning_curve(
            svc, X, y, cv=5, error_score=np.nan
        )

    # the first 2 splits should lead to warnings and thus np.nan scores
    for idx in range(2):
        assert np.isnan(train_score[idx]).all()
        assert np.isnan(test_score[idx]).all()

    for idx in range(2, train_score.shape[0]):
        assert not np.isnan(train_score[idx]).any()
        assert not np.isnan(test_score[idx]).any()


def test_cross_validate_return_indices(global_random_seed):
    # [FUNCTION DOC]
    # Checks return_indices=False omits "indices", and True returns per-split train/test
    # index arrays with sizes 100/50 identical to the KFold splits.
    """Check the behaviour of `return_indices` in `cross_validate`."""
    X, y = load_iris(return_X_y=True)
    X = scale(X)  # scale features for better convergence
    estimator = LogisticRegression()

    cv = KFold(n_splits=3, shuffle=True, random_state=global_random_seed)
    cv_results = cross_validate(estimator, X, y, cv=cv, n_jobs=2, return_indices=False)
    assert "indices" not in cv_results

    cv_results = cross_validate(estimator, X, y, cv=cv, n_jobs=2, return_indices=True)
    assert "indices" in cv_results
    train_indices = cv_results["indices"]["train"]
    test_indices = cv_results["indices"]["test"]
    assert len(train_indices) == cv.n_splits
    assert len(test_indices) == cv.n_splits

    assert_array_equal([indices.size for indices in train_indices], 100)
    assert_array_equal([indices.size for indices in test_indices], 50)

    for split_idx, (expected_train_idx, expected_test_idx) in enumerate(cv.split(X, y)):
        assert_array_equal(train_indices[split_idx], expected_train_idx)
        assert_array_equal(test_indices[split_idx], expected_test_idx)


# Tests for metadata routing in cross_val* and in *curve
# ======================================================


@pytest.mark.parametrize(
    "func, extra_args",
    [
        (cross_validate, {}),
        (cross_val_score, {}),
        (cross_val_predict, {}),
        (learning_curve, {}),
        (permutation_test_score, {}),
        (validation_curve, {"param_name": "alpha", "param_range": np.array([1])}),
    ],
)
@config_context(enable_metadata_routing=True)
def test_groups_with_routing_validation(func, extra_args):
    # [FUNCTION DOC]
    # With metadata routing enabled, checks every validation function raises ValueError when
    # groups is passed directly instead of through params.
    """Check that we raise an error if `groups` are passed to the cv method instead
    of `params` when metadata routing is enabled.
    """
    with pytest.raises(ValueError, match="`groups` can only be passed if"):
        func(
            estimator=ConsumingClassifier(),
            X=X,
            y=y,
            groups=[],
            **extra_args,
        )


@pytest.mark.parametrize(
    "func, extra_args",
    [
        (cross_validate, {}),
        (cross_val_score, {}),
        (cross_val_predict, {}),
        (learning_curve, {}),
        # Few permutations: this test only checks the params=None code path.
        (permutation_test_score, {"n_permutations": 5}),
        (validation_curve, {"param_name": "alpha", "param_range": np.array([1])}),
    ],
)
@config_context(enable_metadata_routing=True)
def test_cross_validate_params_none(func, extra_args):
    # [FUNCTION DOC]
    # Non-regression for #30447: with routing enabled, params=None (default) must not
    # raise in any validation function.
    """Test that no errors are raised when passing `params=None`, which is the
    default value.
    Non-regression test for: https://github.com/scikit-learn/scikit-learn/issues/30447
    """
    X, y = make_classification(n_samples=100, n_classes=2, random_state=0)
    func(estimator=ConsumingClassifier(), X=X, y=y, **extra_args)


@pytest.mark.parametrize(
    "func, extra_args",
    [
        (cross_validate, {}),
        (cross_val_score, {}),
        (cross_val_predict, {}),
        (learning_curve, {}),
        (permutation_test_score, {}),
        (validation_curve, {"param_name": "alpha", "param_range": np.array([1])}),
    ],
)
@config_context(enable_metadata_routing=True)
def test_passed_unrequested_metadata(func, extra_args):
    # [FUNCTION DOC]
    # With routing enabled, checks UnsetMetadataPassedError when metadata is passed but not
    # requested by ConsumingClassifier.fit, and (except cross_val_predict, which does not
    # score) when requested by fit/partial_fit but not by score.
    """Check that we raise an error when passing metadata that is not
    requested."""

    err_msg = re.escape(
        "[metadata] are passed but are not explicitly set as requested or not "
        "requested for ConsumingClassifier.fit, which is used within"
    )
    with pytest.raises(UnsetMetadataPassedError, match=err_msg):
        func(
            estimator=ConsumingClassifier(),
            X=X,
            y=y2,
            params=dict(metadata=[]),
            **extra_args,
        )

    # cross_val_predict doesn't use scoring
    if func == cross_val_predict:
        return

    err_msg = re.escape(
        "[metadata] are passed but are not explicitly set as requested or not "
        "requested for ConsumingClassifier.score, which is used within"
    )
    with pytest.raises(UnsetMetadataPassedError, match=err_msg):
        func(
            estimator=ConsumingClassifier()
            .set_fit_request(metadata=True)
            .set_partial_fit_request(metadata=True),
            X=X,
            y=y2,
            params=dict(metadata=[]),
            **extra_args,
        )


@pytest.mark.parametrize(
    "func, extra_args",
    [
        (cross_validate, {}),
        (cross_val_score, {}),
        (cross_val_predict, {}),
        (learning_curve, {}),
        (permutation_test_score, {"n_permutations": 20}),
        (validation_curve, {"param_name": "alpha", "param_range": np.array([1])}),
    ],
)
@config_context(enable_metadata_routing=True)
def test_validation_functions_routing(func, extra_args):
    # [FUNCTION DOC]
    # With routing enabled, passes aliased metadata for splitter, estimator (fit and
    # partial_fit) and scorer to each validation function, then checks via the registries
    # that each consumer received the correctly split metadata (check_recorded_metadata).
    # cross_val_predict gets no scorer params.
    """Check that the respective cv method is properly dispatching the metadata
    to the consumer."""
    scorer_registry = _Registry()
    scorer = ConsumingScorer(registry=scorer_registry).set_score_request(
        sample_weight="score_weights", metadata="score_metadata"
    )
    splitter_registry = _Registry()
    splitter = ConsumingSplitter(registry=splitter_registry).set_split_request(
        groups="split_groups", metadata="split_metadata"
    )
    estimator_registry = _Registry()
    estimator = (
        ConsumingClassifier(registry=estimator_registry)
        .set_fit_request(sample_weight="fit_sample_weight", metadata="fit_metadata")
        .set_partial_fit_request(
            sample_weight="fit_sample_weight", metadata="fit_metadata"
        )
    )

    n_samples = _num_samples(X)
    rng = np.random.RandomState(0)
    score_weights = rng.rand(n_samples)
    score_metadata = rng.rand(n_samples)
    split_groups = rng.randint(0, 3, n_samples)
    split_metadata = rng.rand(n_samples)
    fit_sample_weight = rng.rand(n_samples)
    fit_metadata = rng.rand(n_samples)

    scoring_args = {
        cross_validate: dict(scoring=dict(my_scorer=scorer, accuracy="accuracy")),
        cross_val_score: dict(scoring=scorer),
        learning_curve: dict(scoring=scorer),
        validation_curve: dict(scoring=scorer),
        permutation_test_score: dict(scoring=scorer),
        cross_val_predict: dict(),
    }

    params = dict(
        split_groups=split_groups,
        split_metadata=split_metadata,
        fit_sample_weight=fit_sample_weight,
        fit_metadata=fit_metadata,
    )

    if func is not cross_val_predict:
        params.update(
            score_weights=score_weights,
            score_metadata=score_metadata,
        )

    func(
        estimator,
        X=X,
        y=y,
        cv=splitter,
        **scoring_args[func],
        **extra_args,
        params=params,
    )

    if func is not cross_val_predict:
        # cross_val_predict doesn't need a scorer
        assert len(scorer_registry)
    for _scorer in scorer_registry:
        check_recorded_metadata(
            obj=_scorer,
            method="consuming_metric",
            parent=func.__name__,
            split_params=("sample_weight", "metadata"),
            sample_weight=score_weights,
            metadata=score_metadata,
        )

    assert len(splitter_registry)
    for _splitter in splitter_registry:
        check_recorded_metadata(
            obj=_splitter,
            method="split",
            parent=func.__name__,
            groups=split_groups,
            metadata=split_metadata,
        )

    assert len(estimator_registry)
    for _estimator in estimator_registry:
        check_recorded_metadata(
            obj=_estimator,
            method="fit",
            parent=func.__name__,
            split_params=("sample_weight", "metadata"),
            sample_weight=fit_sample_weight,
            metadata=fit_metadata,
        )


@config_context(enable_metadata_routing=True)
def test_learning_curve_exploit_incremental_learning_routing():
    # [FUNCTION DOC]
    # Checks learning_curve with exploit_incremental_learning routes aliased sample_weight
    # and metadata to partial_fit of the estimator, verified through the registry.
    """Test that learning_curve routes metadata to the estimator correctly while
    partial_fitting it with `exploit_incremental_learning=True`."""

    n_samples = _num_samples(X)
    rng = np.random.RandomState(0)
    fit_sample_weight = rng.rand(n_samples)
    fit_metadata = rng.rand(n_samples)

    estimator_registry = _Registry()
    estimator = ConsumingClassifier(
        registry=estimator_registry
    ).set_partial_fit_request(
        sample_weight="fit_sample_weight", metadata="fit_metadata"
    )

    learning_curve(
        estimator,
        X=X,
        y=y,
        cv=ConsumingSplitter(),
        exploit_incremental_learning=True,
        params=dict(fit_sample_weight=fit_sample_weight, fit_metadata=fit_metadata),
    )

    assert len(estimator_registry)
    for _estimator in estimator_registry:
        check_recorded_metadata(
            obj=_estimator,
            method="partial_fit",
            parent="learning_curve",
            split_params=("sample_weight", "metadata"),
            sample_weight=fit_sample_weight,
            metadata=fit_metadata,
        )


# End of metadata routing tests
# =============================


@pytest.mark.parametrize(
    "estimator",
    [Ridge(), LinearDiscriminantAnalysis()],
    ids=["Ridge", "LinearDiscriminantAnalysis"],
)
@pytest.mark.parametrize("cv", [None, 3, 5])
@pytest.mark.parametrize(
    "namespace, device_name, dtype_name",
    yield_namespace_device_dtype_combinations(),
)
def test_cross_val_predict_array_api_compliance(
    estimator, cv, namespace, device_name, dtype_name
):
    # [FUNCTION DOC]
    # Checks cross_val_predict with array API inputs (array_api_dispatch enabled) matches the
    # NumPy result within dtype tolerance, for Ridge and LDA and several cv values.
    """Test that `cross_val_predict` functions correctly with the array API
    with both a classifier and a regressor."""

    xp, device = _array_api_for_tests(namespace, device_name, dtype_name)
    if is_classifier(estimator):
        X, y = make_classification(
            n_samples=1000, n_features=5, n_classes=3, n_informative=3, random_state=42
        )
    else:
        X, y = make_regression(
            n_samples=1000, n_features=5, n_informative=3, random_state=42
        )

    X_np = X.astype(dtype_name)
    y_np = y.astype(dtype_name)
    X_xp = xp.asarray(X_np, device=device)
    y_xp = xp.asarray(y_np, device=device)

    with config_context(array_api_dispatch=True):
        pred_xp = cross_val_predict(estimator, X_xp, y_xp, cv=cv)

    pred_np = cross_val_predict(estimator, X_np, y_np, cv=cv)
    assert_allclose(
        move_to(pred_xp, xp=np, device="cpu"), pred_np, atol=_atol_for_type(dtype_name)
    )
