# [MODULE DOC]
# Tests for FrozenEstimator: delegation of prediction methods, no-op fit, metadata
#   routing through a frozen pipeline, blocking of fit_predict/fit_transform, clone
#   behaviour, fitted-state checks, tags and parameter handling.
# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import re

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from sklearn import config_context
from sklearn.base import (
    BaseEstimator,
    clone,
    is_classifier,
    is_clusterer,
    is_outlier_detector,
    is_regressor,
)
from sklearn.cluster import KMeans
from sklearn.compose import make_column_transformer
from sklearn.datasets import make_classification, make_regression
from sklearn.exceptions import NotFittedError, UnsetMetadataPassedError
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.utils._testing import set_random_state
from sklearn.utils.validation import check_is_fitted


@pytest.fixture
def regression_dataset():
    # [FUNCTION DOC]
    # HELPER: Fixture returning a default make_regression (X, y) tuple (100 x 100,
    #   random).
    return make_regression()


@pytest.fixture
def classification_dataset():
    # [FUNCTION DOC]
    # HELPER: Fixture returning a default make_classification (X, y) tuple (100 x 20,
    #   random).
    return make_classification()


@pytest.mark.parametrize(
    "estimator, dataset",
    [
        (LinearRegression(), "regression_dataset"),
        (LogisticRegression(), "classification_dataset"),
        (make_pipeline(StandardScaler(), LinearRegression()), "regression_dataset"),
        (
            make_pipeline(StandardScaler(), LogisticRegression()),
            "classification_dataset",
        ),
        (StandardScaler(), "regression_dataset"),
        (KMeans(), "regression_dataset"),
        (LocalOutlierFactor(), "regression_dataset"),
        (
            make_column_transformer(
                (StandardScaler(), [0]),
                (RobustScaler(), [1]),
            ),
            "regression_dataset",
        ),
    ],
)
@pytest.mark.parametrize(
    "method",
    ["predict", "predict_proba", "predict_log_proba", "decision_function", "transform"],
)
def test_frozen_methods(estimator, dataset, request, method):
    # [FUNCTION DOC]
    # For several estimators (regressor, classifier, pipelines, scaler, KMeans, LOF,
    #   column transformer) and each prediction/transform method name: fits a clone,
    #   wraps it, calls fit on the wrapper with dummy data and checks that every method
    #   the estimator has gives identical output via the wrapper, and that the
    #   classifier/regressor/clusterer/outlier detector type checks agree. Inputs:
    #   estimator, dataset fixture name, pytest request, method name.
    """Test that frozen.fit doesn't do anything, and that all other methods are
    exposed by the frozen estimator and return the same values as the estimator.
    """
    estimator = clone(estimator)
    X, y = request.getfixturevalue(dataset)
    set_random_state(estimator)
    estimator.fit(X, y)
    frozen = FrozenEstimator(estimator)
    # this should be no-op
    frozen.fit([[1]], [1])

    if hasattr(estimator, method):
        assert_array_equal(getattr(estimator, method)(X), getattr(frozen, method)(X))

    assert is_classifier(estimator) == is_classifier(frozen)
    assert is_regressor(estimator) == is_regressor(frozen)
    assert is_clusterer(estimator) == is_clusterer(frozen)
    assert is_outlier_detector(estimator) == is_outlier_detector(frozen)


@config_context(enable_metadata_routing=True)
def test_frozen_metadata_routing(regression_dataset):
    # [FUNCTION DOC]
    # With metadata routing enabled, checks that a frozen pipeline routes predict
    #   metadata like the original, that setting the step's predict request to False
    #   makes the frozen pipeline raise TypeError for unrouted metadata, and that
    #   resetting it to None raises UnsetMetadataPassedError. Also exercises __getitem__
    #   forwarding.
    """Test that metadata routing works with frozen estimators."""

    class ConsumesMetadata(BaseEstimator):
        # [CLASS DOC]
        # HELPER: Minimal estimator whose fit/predict accept a metadata argument and
        #   assert it is present when on_fit/on_predict is truthy.
        def __init__(self, on_fit=None, on_predict=None):
            # [FUNCTION DOC]
            # HELPER: Stores on_fit and on_predict flags (default None) unchanged.
            self.on_fit = on_fit
            self.on_predict = on_predict

        def fit(self, X, y, metadata=None):
            # [FUNCTION DOC]
            # HELPER: Asserts metadata was given if on_fit; sets fitted_ to True and
            #   returns self.
            if self.on_fit:
                assert metadata is not None
            self.fitted_ = True
            return self

        def predict(self, X, metadata=None):
            # [FUNCTION DOC]
            # HELPER: Asserts metadata was given if on_predict; returns a vector of ones
            #   of length len(X).
            if self.on_predict:
                assert metadata is not None
            return np.ones(len(X))

    X, y = regression_dataset
    pipeline = make_pipeline(
        ConsumesMetadata(on_fit=True, on_predict=True)
        .set_fit_request(metadata=True)
        .set_predict_request(metadata=True)
    )

    pipeline.fit(X, y, metadata="test")
    frozen = FrozenEstimator(pipeline)
    pipeline.predict(X, metadata="test")
    frozen.predict(X, metadata="test")

    frozen["consumesmetadata"].set_predict_request(metadata=False)
    with pytest.raises(
        TypeError,
        match=re.escape(
            "Pipeline.predict got unexpected argument(s) {'metadata'}, which are not "
            "routed to any object."
        ),
    ):
        frozen.predict(X, metadata="test")

    frozen["consumesmetadata"].set_predict_request(metadata=None)
    with pytest.raises(UnsetMetadataPassedError):
        frozen.predict(X, metadata="test")


def test_composite_fit(classification_dataset):
    # [FUNCTION DOC]
    # Checks that fit_predict and fit_transform raise AttributeError on the frozen
    #   wrapper even though the inner estimator defines them, and that the inner fit
    #   counter stays at 1.
    """Test that calling fit_transform and fit_predict doesn't call fit."""

    class Estimator(BaseEstimator):
        # [CLASS DOC]
        # HELPER: Estimator counting fit calls in _fit_counter and defining dummy
        #   fit_transform / fit_predict that must never be called.
        def fit(self, X, y):
            # [FUNCTION DOC]
            # HELPER: Increments _fit_counter (initialising it to 1 on first call);
            #   returns self.
            try:
                self._fit_counter += 1
            except AttributeError:
                self._fit_counter = 1
            return self

        def fit_transform(self, X, y=None):
            # [FUNCTION DOC]
            # HELPER: Placeholder that does nothing and returns None; must not be
            #   invoked.
            # only here to test that it doesn't get called
            ...  # pragma: no cover

        def fit_predict(self, X, y=None):
            # [FUNCTION DOC]
            # HELPER: Placeholder that does nothing and returns None; must not be
            #   invoked.
            # only here to test that it doesn't get called
            ...  # pragma: no cover

    X, y = classification_dataset
    est = Estimator().fit(X, y)
    frozen = FrozenEstimator(est)

    with pytest.raises(AttributeError):
        frozen.fit_predict(X, y)
    with pytest.raises(AttributeError):
        frozen.fit_transform(X, y)

    assert frozen._fit_counter == 1


def test_clone_frozen(regression_dataset):
    # [FUNCTION DOC]
    # Checks that clone of a frozen estimator returns a wrapper whose estimator is the
    #   very same fitted object (identity check).
    """Test that cloning a frozen estimator keeps the frozen state."""
    X, y = regression_dataset
    estimator = LinearRegression().fit(X, y)
    frozen = FrozenEstimator(estimator)
    cloned = clone(frozen)
    assert cloned.estimator is estimator


def test_check_is_fitted(regression_dataset):
    # [FUNCTION DOC]
    # Checks check_is_fitted raises NotFittedError for a frozen unfitted estimator and
    #   passes for a frozen fitted one.
    """Test that check_is_fitted works on frozen estimators."""
    X, y = regression_dataset

    estimator = LinearRegression()
    frozen = FrozenEstimator(estimator)
    with pytest.raises(NotFittedError):
        check_is_fitted(frozen)

    estimator = LinearRegression().fit(X, y)
    frozen = FrozenEstimator(estimator)
    check_is_fitted(frozen)


def test_frozen_tags():
    # [FUNCTION DOC]
    # Checks the frozen tags equal the inner tags (custom categorical input tag
    #   preserved) except _skip_test, which is True for the wrapper and False for the
    #   inner estimator.
    """Test that frozen estimators have the same tags as the original estimator
    except for the skip_test tag."""

    class Estimator(BaseEstimator):
        # [CLASS DOC]
        # HELPER: Estimator whose tags declare categorical input support.
        def __sklearn_tags__(self):
            # [FUNCTION DOC]
            # HELPER: Returns the parent tags with input_tags.categorical set to True.
            tags = super().__sklearn_tags__()
            tags.input_tags.categorical = True
            return tags

    estimator = Estimator()
    frozen = FrozenEstimator(estimator)
    frozen_tags = frozen.__sklearn_tags__()
    estimator_tags = estimator.__sklearn_tags__()

    assert frozen_tags._skip_test is True
    assert estimator_tags._skip_test is False

    assert estimator_tags.input_tags.categorical is True
    assert frozen_tags.input_tags.categorical is True


def test_frozen_params():
    # [FUNCTION DOC]
    # Checks set_params rejects nested inner parameters with ValueError, get_params
    #   returns only {"estimator": est}, and set_params(estimator=...) replaces the
    #   inner estimator.
    """Test that FrozenEstimator only exposes the estimator parameter."""
    est = LogisticRegression()
    frozen = FrozenEstimator(est)

    with pytest.raises(ValueError, match="You cannot set parameters of the inner"):
        frozen.set_params(estimator__C=1)

    assert frozen.get_params() == {"estimator": est}

    other_est = LocalOutlierFactor()
    frozen.set_params(estimator=other_est)
    assert frozen.get_params() == {"estimator": other_est}
