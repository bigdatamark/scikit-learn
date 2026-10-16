# [MODULE DOC]
# Tests for BaggingClassifier and BaggingRegressor: parameter grids on dense and
# sparse input, bootstrap of samples and features, probabilities, OOB scores,
# parallelism, warm start, sample_weight-driven index drawing, reproducibility of
# estimators_samples_, missing-value handling via pipelines, tags and metadata
# routing. At import time loads iris and diabetes and permutes them with a seeded
# RandomState (module-level shared data).
"""
Testing for the bagging ensemble module (sklearn.ensemble.bagging).
"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import re
from itertools import cycle, product

import joblib
import numpy as np
import pytest

from sklearn import config_context
from sklearn.base import BaseEstimator
from sklearn.calibration import CalibratedClassifierCV
from sklearn.datasets import load_diabetes, load_iris, make_hastie_10_2
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    AdaBoostClassifier,
    AdaBoostRegressor,
    BaggingClassifier,
    BaggingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.model_selection import GridSearchCV, ParameterGrid, train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, scale
from sklearn.random_projection import SparseRandomProjection
from sklearn.svm import SVC, SVR
from sklearn.tests.metadata_routing_common import (
    ConsumingClassifierWithOnlyPredict,
    ConsumingClassifierWithoutPredictLogProba,
    ConsumingClassifierWithoutPredictProba,
    _Registry,
    check_recorded_metadata,
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils import check_random_state
from sklearn.utils._testing import (
    assert_allclose,
    assert_array_almost_equal,
    assert_array_equal,
)
from sklearn.utils.fixes import CSC_CONTAINERS, CSR_CONTAINERS

rng = check_random_state(0)

# also load the iris dataset
# and randomly permute it
iris = load_iris()
perm = rng.permutation(iris.target.size)
iris.data = iris.data[perm]
iris.target = iris.target[perm]

# also load the diabetes dataset
# and randomly permute it
diabetes = load_diabetes()
perm = rng.permutation(diabetes.target.size)
diabetes.data = diabetes.data[perm]
diabetes.target = diabetes.target[perm]


def test_classification():
    # [FUNCTION DOC]
    # Smoke test: fits and predicts BaggingClassifier on iris for every combination of
    # max_samples, max_features, bootstrap and bootstrap_features, cycling through
    # several base classifiers (None, dummy, perceptron, tree, KNN, SVC).
    # Check classification for various parameter settings.
    rng = check_random_state(0)
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, random_state=rng
    )
    grid = ParameterGrid(
        {
            "max_samples": [0.5, 1.0],
            "max_features": [1, 4],
            "bootstrap": [True, False],
            "bootstrap_features": [True, False],
        }
    )
    estimators = [
        None,
        DummyClassifier(),
        Perceptron(max_iter=20),
        DecisionTreeClassifier(max_depth=2),
        KNeighborsClassifier(),
        SVC(),
    ]
    # Try different parameter settings with different base classifiers without
    # doing the full cartesian product to keep the test durations low.
    for params, estimator in zip(grid, cycle(estimators)):
        BaggingClassifier(
            estimator=estimator,
            random_state=rng,
            n_estimators=2,
            **params,
        ).fit(X_train, y_train).predict(X_test)


@pytest.mark.parametrize(
    "sparse_container, params, method",
    product(
        CSR_CONTAINERS + CSC_CONTAINERS,
        [
            {
                "max_samples": 0.5,
                "max_features": 2,
                "bootstrap": True,
                "bootstrap_features": True,
            },
            {
                "max_samples": 1.0,
                "max_features": 4,
                "bootstrap": True,
                "bootstrap_features": True,
            },
            {"max_features": 2, "bootstrap": False, "bootstrap_features": True},
            {"max_samples": 0.5, "bootstrap": True, "bootstrap_features": False},
        ],
        ["predict", "predict_proba", "predict_log_proba", "decision_function"],
    ),
)
def test_sparse_classification(sparse_container, params, method):
    # [FUNCTION DOC]
    # Parametrized over CSR/CSC containers, four parameter sets and four prediction
    # methods. Verifies that bagging logistic regression on sparse input gives the same
    # outputs as on dense input and that each sub-estimator was trained on the sparse
    # type.
    # KNOWN ISSUE: contains a leftover debug print of the classifier and method.
    # Check classification for various parameter settings on sparse input.
    class CustomClassifier(LogisticRegression):
        # [CLASS DOC]
        # HELPER: LogisticRegression subclass recording the type of the training matrix.
        """LogisticRegression variant that records the nature of the training set"""

        def fit(self, X, y):
            # [FUNCTION DOC]
            # HELPER: Fits the parent and stores type(X) in data_type_. Outputs: self.
            super().fit(X, y)
            self.data_type_ = type(X)
            return self

    rng = check_random_state(0)
    X_train, X_test, y_train, y_test = train_test_split(
        scale(iris.data), iris.target, random_state=rng
    )

    X_train_sparse = sparse_container(X_train)
    X_test_sparse = sparse_container(X_test)

    # Trained on sparse format
    sparse_classifier = BaggingClassifier(
        estimator=CustomClassifier(),
        random_state=1,
        **params,
    ).fit(X_train_sparse, y_train)
    print(sparse_classifier, method)
    sparse_results = getattr(sparse_classifier, method)(X_test_sparse)

    # Trained on dense format
    dense_classifier = BaggingClassifier(
        estimator=CustomClassifier(),
        random_state=1,
        **params,
    ).fit(X_train, y_train)
    dense_results = getattr(dense_classifier, method)(X_test)
    assert_array_almost_equal(sparse_results, dense_results)

    sparse_type = type(X_train_sparse)
    types = [i.data_type_ for i in sparse_classifier.estimators_]

    assert all([t == sparse_type for t in types])


def test_regression():
    # [FUNCTION DOC]
    # Smoke test: fits and predicts BaggingRegressor on 50 diabetes samples for every
    # parameter grid combination and several base regressors.
    # Check regression for various parameter settings.
    rng = check_random_state(0)
    X_train, X_test, y_train, y_test = train_test_split(
        diabetes.data[:50], diabetes.target[:50], random_state=rng
    )
    grid = ParameterGrid(
        {
            "max_samples": [0.5, 1.0],
            "max_features": [0.5, 1.0],
            "bootstrap": [True, False],
            "bootstrap_features": [True, False],
        }
    )

    for estimator in [
        None,
        DummyRegressor(),
        DecisionTreeRegressor(),
        KNeighborsRegressor(),
        SVR(),
    ]:
        for params in grid:
            BaggingRegressor(estimator=estimator, random_state=rng, **params).fit(
                X_train, y_train
            ).predict(X_test)


@pytest.mark.parametrize("sparse_container", CSR_CONTAINERS + CSC_CONTAINERS)
def test_sparse_regression(sparse_container):
    # [FUNCTION DOC]
    # Parametrized over sparse containers. Verifies sparse vs dense BaggingRegressor
    # (SVR base) predictions agree for four parameter sets and that sub-estimators were
    # trained on the sparse type.
    # KNOWN ISSUE: the sparse/dense equality assertion is repeated twice.
    # Check regression for various parameter settings on sparse input.
    rng = check_random_state(0)
    X_train, X_test, y_train, y_test = train_test_split(
        diabetes.data[:50], diabetes.target[:50], random_state=rng
    )

    class CustomSVR(SVR):
        # [CLASS DOC]
        # HELPER: SVR subclass recording the type of the training matrix.
        # KNOWN ISSUE: docstring says SVC variant although it subclasses SVR.
        """SVC variant that records the nature of the training set"""

        def fit(self, X, y):
            # [FUNCTION DOC]
            # HELPER: Fits the parent and stores type(X) in data_type_. Outputs: self.
            super().fit(X, y)
            self.data_type_ = type(X)
            return self

    parameter_sets = [
        {
            "max_samples": 0.5,
            "max_features": 2,
            "bootstrap": True,
            "bootstrap_features": True,
        },
        {
            "max_samples": 1.0,
            "max_features": 4,
            "bootstrap": True,
            "bootstrap_features": True,
        },
        {"max_features": 2, "bootstrap": False, "bootstrap_features": True},
        {"max_samples": 0.5, "bootstrap": True, "bootstrap_features": False},
    ]

    X_train_sparse = sparse_container(X_train)
    X_test_sparse = sparse_container(X_test)
    for params in parameter_sets:
        # Trained on sparse format
        sparse_classifier = BaggingRegressor(
            estimator=CustomSVR(), random_state=1, **params
        ).fit(X_train_sparse, y_train)
        sparse_results = sparse_classifier.predict(X_test_sparse)

        # Trained on dense format
        dense_results = (
            BaggingRegressor(estimator=CustomSVR(), random_state=1, **params)
            .fit(X_train, y_train)
            .predict(X_test)
        )

        sparse_type = type(X_train_sparse)
        types = [i.data_type_ for i in sparse_classifier.estimators_]

        assert_array_almost_equal(sparse_results, dense_results)
        assert all([t == sparse_type for t in types])
        assert_array_almost_equal(sparse_results, dense_results)


class DummySizeEstimator(BaseEstimator):
    # [CLASS DOC]
    # HELPER: Test estimator that records training-set size and a joblib hash of the
    # training data, and predicts ones.
    def fit(self, X, y):
        # [FUNCTION DOC]
        # HELPER: Stores X.shape[0] in training_size_ and joblib.hash(X) in training_hash_.
        # KNOWN ISSUE: does not return self (returns None), unlike sklearn convention.
        self.training_size_ = X.shape[0]
        self.training_hash_ = joblib.hash(X)

    def predict(self, X):
        # [FUNCTION DOC]
        # HELPER: Returns an array of ones of length n_samples.
        return np.ones(X.shape[0])


def test_bootstrap_samples():
    # [FUNCTION DOC]
    # Verifies that without bootstrap a bagged tree ensemble matches a single tree's
    # training score, with bootstrap the training score is lower, and that each
    # bootstrap resample has the full training size but distinct content (hashes).
    # Test that bootstrapping samples generate non-perfect base estimators.
    rng = check_random_state(0)
    X_train, X_test, y_train, y_test = train_test_split(
        diabetes.data, diabetes.target, random_state=rng
    )

    estimator = DecisionTreeRegressor().fit(X_train, y_train)

    # without bootstrap, all trees are perfect on the training set
    ensemble = BaggingRegressor(
        estimator=DecisionTreeRegressor(),
        max_samples=1.0,
        bootstrap=False,
        random_state=rng,
    ).fit(X_train, y_train)

    assert estimator.score(X_train, y_train) == ensemble.score(X_train, y_train)

    # with bootstrap, trees are no longer perfect on the training set
    ensemble = BaggingRegressor(
        estimator=DecisionTreeRegressor(),
        max_samples=1.0,
        bootstrap=True,
        random_state=rng,
    ).fit(X_train, y_train)

    assert estimator.score(X_train, y_train) > ensemble.score(X_train, y_train)

    # check that each sampling correspond to a complete bootstrap resample.
    # the size of each bootstrap should be the same as the input data but
    # the data should be different (checked using the hash of the data).
    ensemble = BaggingRegressor(estimator=DummySizeEstimator(), bootstrap=True).fit(
        X_train, y_train
    )
    training_hash = []
    for estimator in ensemble.estimators_:
        assert estimator.training_size_ == X_train.shape[0]
        training_hash.append(estimator.training_hash_)
    assert len(set(training_hash)) == len(training_hash)


def test_bootstrap_features():
    # [FUNCTION DOC]
    # Verifies that bootstrap_features=False uses every feature once per estimator and
    # bootstrap_features=True yields duplicates (fewer unique features).
    # Test that bootstrapping features may generate duplicate features.
    rng = check_random_state(0)
    X_train, X_test, y_train, y_test = train_test_split(
        diabetes.data, diabetes.target, random_state=rng
    )

    ensemble = BaggingRegressor(
        estimator=DecisionTreeRegressor(),
        max_features=1.0,
        bootstrap_features=False,
        random_state=rng,
    ).fit(X_train, y_train)

    for features in ensemble.estimators_features_:
        assert diabetes.data.shape[1] == np.unique(features).shape[0]

    ensemble = BaggingRegressor(
        estimator=DecisionTreeRegressor(),
        max_features=1.0,
        bootstrap_features=True,
        random_state=rng,
    ).fit(X_train, y_train)

    for features in ensemble.estimators_features_:
        assert diabetes.data.shape[1] > np.unique(features).shape[0]


def test_probability():
    # [FUNCTION DOC]
    # Verifies predict_proba rows sum to 1 and equal exp(predict_log_proba), both in the
    # normal case and when tiny max_samples causes missing classes in sub-estimators.
    # Predict probabilities.
    rng = check_random_state(0)
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, random_state=rng
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        # Normal case
        ensemble = BaggingClassifier(
            estimator=DecisionTreeClassifier(), random_state=rng
        ).fit(X_train, y_train)

        assert_array_almost_equal(
            np.sum(ensemble.predict_proba(X_test), axis=1), np.ones(len(X_test))
        )

        assert_array_almost_equal(
            ensemble.predict_proba(X_test), np.exp(ensemble.predict_log_proba(X_test))
        )

        # Degenerate case, where some classes are missing
        ensemble = BaggingClassifier(
            estimator=LogisticRegression(), random_state=rng, max_samples=5
        ).fit(X_train, y_train)

        assert_array_almost_equal(
            np.sum(ensemble.predict_proba(X_test), axis=1), np.ones(len(X_test))
        )

        assert_array_almost_equal(
            ensemble.predict_proba(X_test), np.exp(ensemble.predict_log_proba(X_test))
        )


def test_oob_score_classification():
    # [FUNCTION DOC]
    # Verifies the OOB score is within 0.1 of the test score for a tree and a calibrated
    # SVC, and that a single estimator triggers the "Some inputs do not have OOB scores"
    # UserWarning.
    # Check that oob prediction is a good estimation of the generalization
    # error.
    rng = check_random_state(0)
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, random_state=rng
    )

    for estimator in [
        DecisionTreeClassifier(),
        CalibratedClassifierCV(SVC(), ensemble=False),
    ]:
        clf = BaggingClassifier(
            estimator=estimator,
            n_estimators=100,
            bootstrap=True,
            oob_score=True,
            random_state=rng,
        ).fit(X_train, y_train)

        test_score = clf.score(X_test, y_test)

        assert abs(test_score - clf.oob_score_) < 0.1

        # Test with few estimators
        warn_msg = (
            "Some inputs do not have OOB scores. This probably means too few "
            "estimators were used to compute any reliable oob estimates."
        )
        with pytest.warns(UserWarning, match=warn_msg):
            clf = BaggingClassifier(
                estimator=estimator,
                n_estimators=1,
                bootstrap=True,
                oob_score=True,
                random_state=rng,
            )
            clf.fit(X_train, y_train)


def test_oob_score_regression():
    # [FUNCTION DOC]
    # Regression counterpart of the OOB test: OOB score within 0.1 of the test score
    # with 50 trees, and the warning with a single estimator.
    # Check that oob prediction is a good estimation of the generalization
    # error.
    rng = check_random_state(0)
    X_train, X_test, y_train, y_test = train_test_split(
        diabetes.data, diabetes.target, random_state=rng
    )

    clf = BaggingRegressor(
        estimator=DecisionTreeRegressor(),
        n_estimators=50,
        bootstrap=True,
        oob_score=True,
        random_state=rng,
    ).fit(X_train, y_train)

    test_score = clf.score(X_test, y_test)

    assert abs(test_score - clf.oob_score_) < 0.1

    # Test with few estimators
    warn_msg = (
        "Some inputs do not have OOB scores. This probably means too few "
        "estimators were used to compute any reliable oob estimates."
    )
    with pytest.warns(UserWarning, match=warn_msg):
        regr = BaggingRegressor(
            estimator=DecisionTreeRegressor(),
            n_estimators=1,
            bootstrap=True,
            oob_score=True,
            random_state=rng,
        )
        regr.fit(X_train, y_train)


def test_single_estimator():
    # [FUNCTION DOC]
    # Verifies a one-estimator bagging of KNN without sample/feature bootstrap predicts
    # like the plain KNN regressor.
    # Check singleton ensembles.
    rng = check_random_state(0)
    X_train, X_test, y_train, y_test = train_test_split(
        diabetes.data, diabetes.target, random_state=rng
    )

    clf1 = BaggingRegressor(
        estimator=KNeighborsRegressor(),
        n_estimators=1,
        bootstrap=False,
        bootstrap_features=False,
        random_state=rng,
    ).fit(X_train, y_train)

    clf2 = KNeighborsRegressor().fit(X_train, y_train)

    assert_array_almost_equal(clf1.predict(X_test), clf2.predict(X_test))


def test_error():
    # [FUNCTION DOC]
    # Verifies BaggingClassifier does not expose decision_function when the base
    # estimator (decision tree) lacks it.
    # Test support of decision_function
    X, y = iris.data, iris.target
    base = DecisionTreeClassifier()
    assert not hasattr(BaggingClassifier(base).fit(X, y), "decision_function")


# TODO: remove mark once loky bug is fixed:
# https://github.com/joblib/loky/issues/458
@pytest.mark.thread_unsafe
def test_parallel_classification():
    # [FUNCTION DOC]
    # Verifies predict_proba and decision_function results are identical across n_jobs
    # values at predict time and between fits with different n_jobs (same seed).
    # Check parallel classification.
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, random_state=0
    )

    ensemble = BaggingClassifier(
        DecisionTreeClassifier(), n_jobs=3, random_state=0
    ).fit(X_train, y_train)

    # predict_proba
    y1 = ensemble.predict_proba(X_test)
    ensemble.set_params(n_jobs=1)
    y2 = ensemble.predict_proba(X_test)
    assert_array_almost_equal(y1, y2)

    ensemble = BaggingClassifier(
        DecisionTreeClassifier(), n_jobs=1, random_state=0
    ).fit(X_train, y_train)

    y3 = ensemble.predict_proba(X_test)
    assert_array_almost_equal(y1, y3)

    # decision_function
    ensemble = BaggingClassifier(
        SVC(decision_function_shape="ovr"), n_jobs=3, random_state=0
    ).fit(X_train, y_train)

    decisions1 = ensemble.decision_function(X_test)
    ensemble.set_params(n_jobs=1)
    decisions2 = ensemble.decision_function(X_test)
    assert_array_almost_equal(decisions1, decisions2)

    ensemble = BaggingClassifier(
        SVC(decision_function_shape="ovr"), n_jobs=1, random_state=0
    ).fit(X_train, y_train)

    decisions3 = ensemble.decision_function(X_test)
    assert_array_almost_equal(decisions1, decisions3)


# TODO: remove mark once loky bug is fixed:
# https://github.com/joblib/loky/issues/458
@pytest.mark.thread_unsafe
def test_parallel_regression():
    # [FUNCTION DOC]
    # Verifies BaggingRegressor predictions are identical across predict-time n_jobs and
    # across fits with n_jobs 3 and 1.
    # Check parallel regression.
    rng = check_random_state(0)

    X_train, X_test, y_train, y_test = train_test_split(
        diabetes.data, diabetes.target, random_state=rng
    )

    ensemble = BaggingRegressor(DecisionTreeRegressor(), n_jobs=3, random_state=0).fit(
        X_train, y_train
    )

    ensemble.set_params(n_jobs=1)
    y1 = ensemble.predict(X_test)
    ensemble.set_params(n_jobs=2)
    y2 = ensemble.predict(X_test)
    assert_array_almost_equal(y1, y2)

    ensemble = BaggingRegressor(DecisionTreeRegressor(), n_jobs=1, random_state=0).fit(
        X_train, y_train
    )

    y3 = ensemble.predict(X_test)
    assert_array_almost_equal(y1, y3)


def test_gridsearch():
    # [FUNCTION DOC]
    # Smoke test that BaggingClassifier(SVC) can be grid-searched over n_estimators and
    # estimator__C with roc_auc scoring on a binarised iris target.
    # KNOWN ISSUE: y is the module-level iris.target array and is modified in place
    # (class 2 set to 1), which leaks into other tests using iris.
    # Check that bagging ensembles can be grid-searched.
    # Transform iris into a binary classification task
    X, y = iris.data, iris.target
    y[y == 2] = 1

    # Grid search with scoring based on decision_function
    parameters = {"n_estimators": (1, 2), "estimator__C": (1, 2)}

    GridSearchCV(BaggingClassifier(SVC()), parameters, scoring="roc_auc").fit(X, y)


# TODO: remove mark once loky bug is fixed:
# https://github.com/joblib/loky/issues/458
@pytest.mark.thread_unsafe
def test_estimator():
    # [FUNCTION DOC]
    # Verifies estimator_ defaults to DecisionTreeClassifier/Regressor when estimator is
    # None and otherwise is of the given estimator's type.
    # Check estimator and its default values.
    rng = check_random_state(0)

    # Classification
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, random_state=rng
    )

    ensemble = BaggingClassifier(None, n_jobs=3, random_state=0).fit(X_train, y_train)

    assert isinstance(ensemble.estimator_, DecisionTreeClassifier)

    ensemble = BaggingClassifier(
        DecisionTreeClassifier(), n_jobs=3, random_state=0
    ).fit(X_train, y_train)

    assert isinstance(ensemble.estimator_, DecisionTreeClassifier)

    ensemble = BaggingClassifier(Perceptron(), n_jobs=3, random_state=0).fit(
        X_train, y_train
    )

    assert isinstance(ensemble.estimator_, Perceptron)

    # Regression
    X_train, X_test, y_train, y_test = train_test_split(
        diabetes.data, diabetes.target, random_state=rng
    )

    ensemble = BaggingRegressor(None, n_jobs=3, random_state=0).fit(X_train, y_train)

    assert isinstance(ensemble.estimator_, DecisionTreeRegressor)

    ensemble = BaggingRegressor(DecisionTreeRegressor(), n_jobs=3, random_state=0).fit(
        X_train, y_train
    )

    assert isinstance(ensemble.estimator_, DecisionTreeRegressor)

    ensemble = BaggingRegressor(SVR(), n_jobs=3, random_state=0).fit(X_train, y_train)
    assert isinstance(ensemble.estimator_, SVR)


def test_bagging_with_pipeline():
    # [FUNCTION DOC]
    # Verifies that a pipeline base estimator gets an integer random_state set on its
    # nested final step.
    estimator = BaggingClassifier(
        make_pipeline(SelectKBest(k=1), DecisionTreeClassifier()), max_features=2
    )
    estimator.fit(iris.data, iris.target)
    assert isinstance(estimator[0].steps[-1][1].random_state, int)


def test_warm_start(random_state=42):
    # [FUNCTION DOC]
    # Verifies that warm-starting from 5 to 10 estimators yields the expected sizes and
    # the same set of per-estimator random seeds as a single 10-estimator fit.
    # Inputs: random_state - plain default argument (42), not a fixture.
    # Test if fitting incrementally with warm start gives a forest of the
    # right size and the same results as a normal fit.
    X, y = make_hastie_10_2(n_samples=20, random_state=1)

    clf_ws = None
    for n_estimators in [5, 10]:
        if clf_ws is None:
            clf_ws = BaggingClassifier(
                n_estimators=n_estimators, random_state=random_state, warm_start=True
            )
        else:
            clf_ws.set_params(n_estimators=n_estimators)
        clf_ws.fit(X, y)
        assert len(clf_ws) == n_estimators

    clf_no_ws = BaggingClassifier(
        n_estimators=10, random_state=random_state, warm_start=False
    )
    clf_no_ws.fit(X, y)

    assert set([tree.random_state for tree in clf_ws]) == set(
        [tree.random_state for tree in clf_no_ws]
    )


def test_warm_start_smaller_n_estimators():
    # [FUNCTION DOC]
    # Verifies warm-start refit with fewer estimators raises ValueError.
    # Test if warm start'ed second fit with smaller n_estimators raises error.
    X, y = make_hastie_10_2(n_samples=20, random_state=1)
    clf = BaggingClassifier(n_estimators=5, warm_start=True)
    clf.fit(X, y)
    clf.set_params(n_estimators=4)
    with pytest.raises(ValueError):
        clf.fit(X, y)


def test_warm_start_equal_n_estimators():
    # [FUNCTION DOC]
    # Verifies warm-start refit with the same n_estimators warns and leaves the model
    # unchanged (predictions identical even after altering X_train in place).
    # Test that nothing happens when fitting without increasing n_estimators
    X, y = make_hastie_10_2(n_samples=20, random_state=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=43)

    clf = BaggingClassifier(n_estimators=5, warm_start=True, random_state=83)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    # modify X to nonsense values, this should not change anything
    X_train += 1.0

    warn_msg = "Warm-start fitting without increasing n_estimators does not"
    with pytest.warns(UserWarning, match=warn_msg):
        clf.fit(X_train, y_train)
    assert_array_equal(y_pred, clf.predict(X_test))


def test_warm_start_equivalence():
    # [FUNCTION DOC]
    # Verifies a 5+5 warm-started classifier predicts like a 10-estimator classifier
    # with the same random_state.
    # warm started classifier with 5+5 estimators should be equivalent to
    # one classifier with 10 estimators
    X, y = make_hastie_10_2(n_samples=20, random_state=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=43)

    clf_ws = BaggingClassifier(n_estimators=5, warm_start=True, random_state=3141)
    clf_ws.fit(X_train, y_train)
    clf_ws.set_params(n_estimators=10)
    clf_ws.fit(X_train, y_train)
    y1 = clf_ws.predict(X_test)

    clf = BaggingClassifier(n_estimators=10, warm_start=False, random_state=3141)
    clf.fit(X_train, y_train)
    y2 = clf.predict(X_test)

    assert_array_almost_equal(y1, y2)


def test_warm_start_with_oob_score_fails():
    # [FUNCTION DOC]
    # Verifies that warm_start together with oob_score raises ValueError at fit.
    # Check using oob_score and warm_start simultaneously fails
    X, y = make_hastie_10_2(n_samples=20, random_state=1)
    clf = BaggingClassifier(n_estimators=5, warm_start=True, oob_score=True)
    with pytest.raises(ValueError):
        clf.fit(X, y)


def test_warning_bootstrap_sample_weight():
    # [FUNCTION DOC]
    # Verifies that fitting Bagging{Classifier,Regressor} with sample_weight and
    # bootstrap=False emits the recommendation warning to use bootstrap=True.
    X, y = iris.data, iris.target
    sample_weight = np.ones_like(y)
    clf = BaggingClassifier(bootstrap=False)
    warn_msg = (
        "When fitting BaggingClassifier with sample_weight "
        "it is recommended to use bootstrap=True"
    )
    with pytest.warns(UserWarning, match=warn_msg):
        clf.fit(X, y, sample_weight=sample_weight)

    X, y = diabetes.data, diabetes.target
    sample_weight = np.ones_like(y)
    reg = BaggingRegressor(bootstrap=False)
    warn_msg = (
        "When fitting BaggingRegressor with sample_weight "
        "it is recommended to use bootstrap=True"
    )
    with pytest.warns(UserWarning, match=warn_msg):
        reg.fit(X, y, sample_weight=sample_weight)


def test_invalid_sample_weight_max_samples_bootstrap_combinations():
    # [FUNCTION DOC]
    # Verifies (1) small total sample weight with fractional max_samples warns about a
    # low bootstrap count, and (2) bootstrap=False with weights summing above n_samples
    # raises ValueError about sampling without replacement (after the bootstrap warning).
    X, y = iris.data, iris.target

    # Case 1: small weights and fractional max_samples lead to a small
    # number of bootstrap samples, which raises a UserWarning.
    clf = BaggingClassifier(max_samples=1.0)
    sample_weight = np.ones_like(y) / (2 * len(y))
    expected_msg = (
        "Using the fractional value max_samples=1.0 when "
        r"the total sum of sample weights is 0.5(\d*) "
        r"results in a low number \(1\) of bootstrap samples. "
        "We recommend passing `max_samples` as an integer."
    )
    with pytest.warns(UserWarning, match=expected_msg):
        clf.fit(X, y, sample_weight=sample_weight)

    # Case 2: large weights and bootstrap=False would lead to sampling without
    # replacement more than the number of samples, which is not allowed.
    clf = BaggingClassifier(bootstrap=False, max_samples=1.0)
    sample_weight = np.ones_like(y)
    sample_weight[-1] = 2
    expected_msg = re.escape(
        "max_samples=151 must be <= n_samples=150 to be able to sample without "
        "replacement."
    )
    with pytest.raises(ValueError, match=expected_msg):
        with pytest.warns(
            UserWarning, match="When fitting BaggingClassifier with sample_weight"
        ):
            clf.fit(X, y, sample_weight=sample_weight)


class EstimatorAcceptingSampleWeight(BaseEstimator):
    # [CLASS DOC]
    # HELPER: Fake estimator whose fit accepts sample_weight and records its inputs.
    """Fake estimator accepting sample_weight"""

    def fit(self, X, y, sample_weight=None):
        # [FUNCTION DOC]
        # HELPER: Stores X, y and sample_weight in X_, y_, sample_weight_; returns None.
        """Record values passed during fit"""
        self.X_ = X
        self.y_ = y
        self.sample_weight_ = sample_weight

    def predict(self, X):
        # [FUNCTION DOC]
        # HELPER: No-op predict returning None.
        pass


class EstimatorRejectingSampleWeight(BaseEstimator):
    # [CLASS DOC]
    # HELPER: Fake estimator whose fit has no sample_weight parameter; records inputs.
    """Fake estimator rejecting sample_weight"""

    def fit(self, X, y):
        # [FUNCTION DOC]
        # HELPER: Stores X and y in X_ and y_; returns None.
        """Record values passed during fit"""
        self.X_ = X
        self.y_ = y

    def predict(self, X):
        # [FUNCTION DOC]
        # HELPER: No-op predict returning None.
        pass


@pytest.mark.parametrize("bagging_class", [BaggingRegressor, BaggingClassifier])
@pytest.mark.parametrize("accept_sample_weight", [False, True])
@pytest.mark.parametrize("metadata_routing", [False, True])
@pytest.mark.parametrize("max_samples", [10, 0.8])
def test_draw_indices_using_sample_weight(
    bagging_class, accept_sample_weight, metadata_routing, max_samples
):
    # [FUNCTION DOC]
    # Parametrized over bagging class, sample_weight support, metadata routing and
    # max_samples (int or fraction of total weight). With only samples 4 and 5 having
    # non-zero weight, verifies drawn indices only contain 4 and 5, have the expected
    # count, and are passed either as integer counts in sample_weight over the full data
    # (when supported) or by indexing X and y (otherwise).
    X = np.arange(100).reshape(-1, 1)
    y = np.repeat([0, 1], 50)
    # all indices except 4 and 5 have zero weight
    sample_weight = np.zeros(100)
    sample_weight[4] = 1
    sample_weight[5] = 2
    if accept_sample_weight:
        base_estimator = EstimatorAcceptingSampleWeight()
    else:
        base_estimator = EstimatorRejectingSampleWeight()

    n_samples, n_features = X.shape

    if isinstance(max_samples, float):
        # max_samples passed as a fraction of the input data. Since
        # sample_weight are provided, the effective number of samples is the
        # sum of the sample weights.
        expected_integer_max_samples = int(max_samples * sample_weight.sum())
    else:
        expected_integer_max_samples = max_samples

    with config_context(enable_metadata_routing=metadata_routing):
        # TODO(slep006): remove block when default routing is implemented
        if metadata_routing and accept_sample_weight:
            base_estimator = base_estimator.set_fit_request(sample_weight=True)
        bagging = bagging_class(base_estimator, max_samples=max_samples, n_estimators=4)
        bagging.fit(X, y, sample_weight=sample_weight)
        for estimator, samples in zip(bagging.estimators_, bagging.estimators_samples_):
            counts = np.bincount(samples, minlength=n_samples)
            assert sum(counts) == len(samples) == expected_integer_max_samples
            # only indices 4 and 5 should appear
            assert np.isin(samples, [4, 5]).all()
            if accept_sample_weight:
                # sampled indices represented through weighting
                assert estimator.X_.shape == (n_samples, n_features)
                assert estimator.y_.shape == (n_samples,)
                assert_allclose(estimator.X_, X)
                assert_allclose(estimator.y_, y)
                assert_allclose(estimator.sample_weight_, counts)
            else:
                # sampled indices represented through indexing
                assert estimator.X_.shape == (expected_integer_max_samples, n_features)
                assert estimator.y_.shape == (expected_integer_max_samples,)
                assert_allclose(estimator.X_, X[samples])
                assert_allclose(estimator.y_, y[samples])


def test_oob_score_removed_on_warm_start():
    # [FUNCTION DOC]
    # Verifies oob_score_ is removed after a warm-start refit with oob_score=False.
    X, y = make_hastie_10_2(n_samples=100, random_state=1)

    clf = BaggingClassifier(n_estimators=5, oob_score=True)
    clf.fit(X, y)

    clf.set_params(warm_start=True, oob_score=False, n_estimators=10)
    clf.fit(X, y)

    with pytest.raises(AttributeError):
        getattr(clf, "oob_score_")


def test_oob_score_consistency():
    # [FUNCTION DOC]
    # Verifies two fits with fixed random_state give the same oob_score_.
    # Make sure OOB scores are identical when random_state, estimator, and
    # training data are fixed and fitting is done twice
    X, y = make_hastie_10_2(n_samples=200, random_state=1)
    bagging = BaggingClassifier(
        KNeighborsClassifier(),
        max_samples=0.5,
        max_features=0.5,
        oob_score=True,
        random_state=1,
    )
    assert bagging.fit(X, y).oob_score_ == bagging.fit(X, y).oob_score_


def test_estimators_samples():
    # [FUNCTION DOC]
    # Verifies estimators_samples_ has one entry per estimator, length n/2, integer
    # dtype, and that refitting an estimator on the recorded samples/features reproduces
    # its coefficients.
    # Check that format of estimators_samples_ is correct and that results
    # generated at fit time can be identically reproduced at a later time
    # using data saved in object attributes.
    X, y = make_hastie_10_2(n_samples=200, random_state=1)
    bagging = BaggingClassifier(
        LogisticRegression(),
        max_samples=0.5,
        max_features=0.5,
        random_state=1,
        bootstrap=False,
    )
    bagging.fit(X, y)

    # Get relevant attributes
    estimators_samples = bagging.estimators_samples_
    estimators_features = bagging.estimators_features_
    estimators = bagging.estimators_

    # Test for correct formatting
    assert len(estimators_samples) == len(estimators)
    assert len(estimators_samples[0]) == len(X) // 2
    assert estimators_samples[0].dtype.kind == "i"

    # Re-fit single estimator to test for consistent sampling
    estimator_index = 0
    estimator_samples = estimators_samples[estimator_index]
    estimator_features = estimators_features[estimator_index]
    estimator = estimators[estimator_index]

    X_train = (X[estimator_samples])[:, estimator_features]
    y_train = y[estimator_samples]

    orig_coefs = estimator.coef_
    estimator.fit(X_train, y_train)
    new_coefs = estimator.coef_

    assert_array_almost_equal(orig_coefs, new_coefs)


def test_estimators_samples_deterministic():
    # [FUNCTION DOC]
    # Regression test (#9524): with a randomized pipeline step, refitting the first
    # estimator on its recorded samples/features reproduces the coefficients exactly.
    # This test is a regression test to check that with a random step
    # (e.g. SparseRandomProjection) and a given random state, the results
    # generated at fit time can be identically reproduced at a later time using
    # data saved in object attributes. Check issue #9524 for full discussion.

    iris = load_iris()
    X, y = iris.data, iris.target

    base_pipeline = make_pipeline(
        SparseRandomProjection(n_components=2), LogisticRegression()
    )
    clf = BaggingClassifier(estimator=base_pipeline, max_samples=0.5, random_state=0)
    clf.fit(X, y)
    pipeline_estimator_coef = clf.estimators_[0].steps[-1][1].coef_.copy()

    estimator = clf.estimators_[0]
    estimator_sample = clf.estimators_samples_[0]
    estimator_feature = clf.estimators_features_[0]

    X_train = (X[estimator_sample])[:, estimator_feature]
    y_train = y[estimator_sample]

    estimator.fit(X_train, y_train)
    assert_array_equal(estimator.steps[-1][1].coef_, pipeline_estimator_coef)


def test_max_samples_consistency():
    # [FUNCTION DOC]
    # Verifies the internal _max_samples equals the user-provided integer max_samples.
    # Make sure validated max_samples and original max_samples are identical
    # when valid integer max_samples supplied by user
    max_samples = 100
    X, y = make_hastie_10_2(n_samples=2 * max_samples, random_state=1)
    bagging = BaggingClassifier(
        KNeighborsClassifier(),
        max_samples=max_samples,
        max_features=0.5,
        random_state=1,
    )
    bagging.fit(X, y)
    assert bagging._max_samples == max_samples


def test_set_oob_score_label_encoding():
    # [FUNCTION DOC]
    # Verifies oob_score_ is identical for string, signed and 0-based integer label
    # encodings of the same targets.
    # Make sure the oob_score doesn't change when the labels change
    # See: https://github.com/scikit-learn/scikit-learn/issues/8933
    random_state = 5
    X = [[-1], [0], [1]] * 5
    Y1 = ["A", "B", "C"] * 5
    Y2 = [-1, 0, 1] * 5
    Y3 = [0, 1, 2] * 5
    x1 = (
        BaggingClassifier(oob_score=True, random_state=random_state)
        .fit(X, Y1)
        .oob_score_
    )
    x2 = (
        BaggingClassifier(oob_score=True, random_state=random_state)
        .fit(X, Y2)
        .oob_score_
    )
    x3 = (
        BaggingClassifier(oob_score=True, random_state=random_state)
        .fit(X, Y3)
        .oob_score_
    )
    assert [x1, x2] == [x3, x3]


def replace(X):
    # [FUNCTION DOC]
    # HELPER: Returns a float copy of X with non-finite values (NaN, inf, None) set to 0.
    # Used as a FunctionTransformer for missing-value tests.
    X = X.astype("float", copy=True)
    X[~np.isfinite(X)] = 0
    return X


def test_bagging_regressor_with_missing_inputs():
    # [FUNCTION DOC]
    # Verifies BaggingRegressor with an imputing pipeline handles NaN/inf/None input for
    # 1d and 2d targets with correct output shape, and that without imputation
    # ValueError is propagated by both the pipeline and the bagging wrapper.
    # Check that BaggingRegressor can accept X with missing/infinite data
    X = np.array(
        [
            [1, 3, 5],
            [2, None, 6],
            [2, np.nan, 6],
            [2, np.inf, 6],
            [2, -np.inf, 6],
        ]
    )
    y_values = [
        np.array([2, 3, 3, 3, 3]),
        np.array(
            [
                [2, 1, 9],
                [3, 6, 8],
                [3, 6, 8],
                [3, 6, 8],
                [3, 6, 8],
            ]
        ),
    ]
    for y in y_values:
        regressor = DecisionTreeRegressor()
        pipeline = make_pipeline(FunctionTransformer(replace), regressor)
        pipeline.fit(X, y).predict(X)
        bagging_regressor = BaggingRegressor(pipeline)
        y_hat = bagging_regressor.fit(X, y).predict(X)
        assert y.shape == y_hat.shape

        # Verify that exceptions can be raised by wrapper regressor
        regressor = DecisionTreeRegressor()
        pipeline = make_pipeline(regressor)
        with pytest.raises(ValueError):
            pipeline.fit(X, y)
        bagging_regressor = BaggingRegressor(pipeline)
        with pytest.raises(ValueError):
            bagging_regressor.fit(X, y)


def test_bagging_classifier_with_missing_inputs():
    # [FUNCTION DOC]
    # Classifier counterpart: imputing pipeline works for predict, predict_proba and
    # predict_log_proba; without imputation ValueError is raised at fit.
    # Check that BaggingClassifier can accept X with missing/infinite data
    X = np.array(
        [
            [1, 3, 5],
            [2, None, 6],
            [2, np.nan, 6],
            [2, np.inf, 6],
            [2, -np.inf, 6],
        ]
    )
    y = np.array([3, 6, 6, 6, 6])
    classifier = DecisionTreeClassifier()
    pipeline = make_pipeline(FunctionTransformer(replace), classifier)
    pipeline.fit(X, y).predict(X)
    bagging_classifier = BaggingClassifier(pipeline)
    bagging_classifier.fit(X, y)
    y_hat = bagging_classifier.predict(X)
    assert y.shape == y_hat.shape
    bagging_classifier.predict_log_proba(X)
    bagging_classifier.predict_proba(X)

    # Verify that exceptions can be raised by wrapper classifier
    classifier = DecisionTreeClassifier()
    pipeline = make_pipeline(classifier)
    with pytest.raises(ValueError):
        pipeline.fit(X, y)
    bagging_classifier = BaggingClassifier(pipeline)
    with pytest.raises(ValueError):
        bagging_classifier.fit(X, y)


def test_bagging_small_max_features():
    # [FUNCTION DOC]
    # Smoke test that a tiny fractional max_features (0.3 of 2 features) still fits
    # (at least one feature is drawn).
    # Check that Bagging estimator can accept low fractional max_features

    X = np.array([[1, 2], [3, 4]])
    y = np.array([1, 0])

    bagging = BaggingClassifier(LogisticRegression(), max_features=0.3, random_state=1)
    bagging.fit(X, y)


def test_bagging_get_estimators_indices(global_random_seed):
    # [FUNCTION DOC]
    # Regression test (#16436): uses a regressor that stores y (equal to sample indices)
    # and verifies the recorded indices equal estimators_samples_[0].
    # Check that Bagging estimator can generate sample indices properly
    # Non-regression test for:
    # https://github.com/scikit-learn/scikit-learn/issues/16436

    rng = np.random.RandomState(global_random_seed)
    X = rng.randn(13, 4)
    y = np.arange(13)

    class MyEstimator(DecisionTreeRegressor):
        # [CLASS DOC]
        # HELPER: DecisionTreeRegressor subclass whose fit only stores y as _sample_indices.
        """An estimator which stores y indices information at fit."""

        def fit(self, X, y):
            # [FUNCTION DOC]
            # HELPER: Stores y in _sample_indices without fitting; returns None.
            self._sample_indices = y

    clf = BaggingRegressor(estimator=MyEstimator(), n_estimators=1, random_state=0)
    clf.fit(X, y)

    assert_array_equal(clf.estimators_[0]._sample_indices, clf.estimators_samples_[0])


@pytest.mark.parametrize(
    "bagging, expected_allow_nan",
    [
        (BaggingClassifier(HistGradientBoostingClassifier(max_iter=1)), True),
        (BaggingRegressor(HistGradientBoostingRegressor(max_iter=1)), True),
        (BaggingClassifier(LogisticRegression()), False),
        (BaggingRegressor(SVR()), False),
    ],
)
def test_bagging_allow_nan_tag(bagging, expected_allow_nan):
    # [FUNCTION DOC]
    # Parametrized. Verifies the allow_nan input tag is inherited from the base estimator.
    """Check that bagging inherits allow_nan tag."""
    assert bagging.__sklearn_tags__().input_tags.allow_nan == expected_allow_nan


# Metadata Routing Tests
# ======================


@config_context(enable_metadata_routing=True)
@pytest.mark.parametrize(
    "model",
    [
        BaggingClassifier(
            estimator=RandomForestClassifier(n_estimators=1), n_estimators=1
        ),
        BaggingRegressor(
            estimator=RandomForestRegressor(n_estimators=1), n_estimators=1
        ),
    ],
)
def test_bagging_with_metadata_routing(model):
    # [FUNCTION DOC]
    # Smoke test with metadata routing enabled that bagging works with a random forest
    # base estimator.
    # KNOWN ISSUE: decorator order applies config_context to the function at definition,
    # it works but the parametrize instances are shared mutable objects.
    """Make sure that metadata routing works with non-default estimator."""
    model.fit(iris.data, iris.target)


@pytest.mark.parametrize(
    "sub_estimator, caller, callee",
    [
        (ConsumingClassifierWithoutPredictProba, "predict", "predict"),
        (
            ConsumingClassifierWithoutPredictLogProba,
            "predict_log_proba",
            "predict_proba",
        ),
        (ConsumingClassifierWithOnlyPredict, "predict_log_proba", "predict"),
    ],
)
@config_context(enable_metadata_routing=True)
def test_metadata_routing_with_dynamic_method_selection(sub_estimator, caller, callee):
    # [FUNCTION DOC]
    # Verifies metadata (sample_weight, metadata) passed to BaggingClassifier
    # predict / predict_log_proba is routed to the sub-estimator method actually used
    # (predict, predict_proba) when some prediction methods are missing, by checking the
    # registry of consuming sub-estimators.
    """Test that metadata routing works in `BaggingClassifier` with dynamic selection of
    the sub-estimator's methods. Here we test only specific test cases, where
    sub-estimator methods are not present and are not tested with `ConsumingClassifier`
    (which possesses all the methods) in
    sklearn/tests/test_metaestimators_metadata_routing.py: `BaggingClassifier.predict()`
    dynamically routes to `predict` if the sub-estimator doesn't have `predict_proba`
    and `BaggingClassifier.predict_log_proba()` dynamically routes to `predict_proba` if
    the sub-estimator doesn't have `predict_log_proba`, or to `predict`, if it doesn't
    have it.
    """
    X = np.array([[0, 2], [1, 4], [2, 6]])
    y = [1, 2, 3]
    sample_weight, metadata = [1], "a"
    registry = _Registry()
    estimator = sub_estimator(registry=registry)
    set_callee_request = "set_" + callee + "_request"
    getattr(estimator, set_callee_request)(sample_weight=True, metadata=True)

    bagging = BaggingClassifier(estimator=estimator)
    bagging.fit(X, y)
    getattr(bagging, caller)(
        X=np.array([[1, 1], [1, 3], [0, 2]]),
        sample_weight=sample_weight,
        metadata=metadata,
    )

    assert len(registry)
    for estimator in registry:
        check_recorded_metadata(
            obj=estimator,
            method=callee,
            parent=caller,
            sample_weight=sample_weight,
            metadata=metadata,
        )


# End of Metadata Routing Tests
# =============================


@pytest.mark.parametrize(
    "model",
    [
        BaggingClassifier(
            estimator=AdaBoostClassifier(n_estimators=1),
            n_estimators=1,
        ),
        BaggingRegressor(estimator=AdaBoostRegressor(n_estimators=1), n_estimators=1),
    ],
)
def test_bagging_without_support_metadata_routing(model):
    # [FUNCTION DOC]
    # Smoke test that bagging an AdaBoost estimator (no routing support) still fits.
    """Make sure that we still can use an estimator that does not implement the
    metadata routing."""
    model.fit(iris.data, iris.target)
