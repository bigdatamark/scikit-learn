# [MODULE DOC]
# Tests for the estimator tags API (sklearn.utils.Tags, get_tags) and related estimator
# checks: class-vs-instance errors, target_tags.required defaults per mixin, rejection
# of legacy _more_tags, subclassed Tags dataclasses, and errors when no concrete
# __sklearn_tags__ exists in the MRO.
import re
from dataclasses import dataclass, fields

import numpy as np
import pytest

from sklearn.base import (
    BaseEstimator,
    ClassifierMixin,
    RegressorMixin,
    TransformerMixin,
)
from sklearn.pipeline import Pipeline
from sklearn.utils import (
    Tags,
    get_tags,
)
from sklearn.utils.estimator_checks import (
    check_estimator_tags_renamed,
    check_valid_tag_types,
)


class EmptyClassifier(ClassifierMixin, BaseEstimator):
    # [CLASS DOC]
    # HELPER: Minimal classifier (ClassifierMixin + BaseEstimator) with default tags.
    pass


class EmptyTransformer(TransformerMixin, BaseEstimator):
    # [CLASS DOC]
    # HELPER: Minimal transformer (TransformerMixin + BaseEstimator) with default tags.
    pass


class EmptyRegressor(RegressorMixin, BaseEstimator):
    # [CLASS DOC]
    # HELPER: Minimal regressor (RegressorMixin + BaseEstimator) with default tags.
    pass


def test_type_error_is_thrown_for_class_vs_instance():
    # [FUNCTION DOC]
    # Verifies get_tags raises TypeError with a message naming both the expected
    # instance and the received class when passed an estimator class.
    """Test that a clearer error is raised if a class is passed instead of an instance.

    Related to the discussion in
    https://github.com/scikit-learn/scikit-learn/issues/32394#issuecomment-3375647854.
    """
    estimator_class = EmptyClassifier
    match = re.escape(
        "Expected an estimator instance (EmptyClassifier()), "
        "got estimator class instead (EmptyClassifier)."
    )
    with pytest.raises(TypeError, match=match):
        get_tags(estimator_class)


@pytest.mark.parametrize(
    "estimator, value",
    [
        [EmptyClassifier(), True],
        [EmptyTransformer(), False],
        [EmptyRegressor(), True],
        [BaseEstimator(), False],
    ],
)
def test_requires_y(estimator, value):
    # [FUNCTION DOC]
    # Verifies target_tags.required defaults: True for classifiers and regressors, False
    # for transformers and BaseEstimator.
    # Inputs: estimator - instance; value - expected bool.
    assert get_tags(estimator).target_tags.required == value


def test_no___sklearn_tags__with_more_tags():
    # [FUNCTION DOC]
    # Verifies check_estimator_tags_renamed raises TypeError for an estimator defining
    # legacy _more_tags.
    """Test that calling `get_tags` on a class that defines `_more_tags` but not
    `__sklearn_tags__` raises an error.
    """

    class MoreTagsEstimator(BaseEstimator):
        # [CLASS DOC]
        # HELPER: Estimator defining the legacy _more_tags hook.
        def _more_tags(self):
            # [FUNCTION DOC]
            # HELPER: Returns legacy tags dict {'requires_y': True}; never called.
            return {"requires_y": True}  # pragma: no cover

    with pytest.raises(
        TypeError, match="has defined either `_more_tags` or `_get_tags`"
    ):
        check_estimator_tags_renamed("MoreTagsEstimator", MoreTagsEstimator())


def test_tag_test_passes_with_inheritance():
    # [FUNCTION DOC]
    # Verifies check_valid_tag_types accepts an estimator returning a Tags subclass with
    # an extra boolean field.
    @dataclass
    class MyTags(Tags):
        # [CLASS DOC]
        # HELPER: Tags dataclass subclass adding my_tag (bool, default True).
        my_tag: bool = True

    class MyEstimator(BaseEstimator):
        # [CLASS DOC]
        # HELPER: Estimator returning MyTags from __sklearn_tags__.
        def __sklearn_tags__(self):
            # [FUNCTION DOC]
            # HELPER: Copies every field of the parent Tags into a MyTags instance, sets
            # my_tag True and returns it.
            tags_orig = super().__sklearn_tags__()
            as_dict = {
                field.name: getattr(tags_orig, field.name)
                for field in fields(tags_orig)
            }
            tags = MyTags(**as_dict)
            tags.my_tag = True
            return tags

    check_valid_tag_types("MyEstimator", MyEstimator())


def test_tags_no_sklearn_tags_concrete_implementation():
    # [FUNCTION DOC]
    # Non-regression for #30479: verifies that fitting/predicting a Pipeline whose step
    # only has mixin-provided __sklearn_tags__ (calling super with no concrete base) or
    # no __sklearn_tags__ at all raises AttributeError with 'The following error was
    # raised'; and that errors of other types (or AttributeError) raised inside a
    # concrete __sklearn_tags__ propagate unchanged from get_tags.
    """Non-regression test for:
    https://github.com/scikit-learn/scikit-learn/issues/30479

    Either the estimator doesn't implement `__sklearn_tags` or there is no class
    implementing `__sklearn_tags__` without calling `super().__sklearn_tags__()` in
    its mro. Thus, we raise an error and request to inherit from
    `BaseEstimator` that implements `__sklearn_tags__`.
    """

    X = np.array([[1, 2], [2, 3], [3, 4]])
    y = np.array([1, 0, 1])

    # 1st case, the estimator inherits from a class that only implements
    # `__sklearn_tags__` by calling `super().__sklearn_tags__()`.
    class MyEstimator(ClassifierMixin):
        # [CLASS DOC]
        # HELPER: ClassifierMixin-only estimator lacking BaseEstimator, so
        # super().__sklearn_tags__ fails.
        def __init__(self, *, param=1):
            # [FUNCTION DOC]
            # HELPER: Stores keyword-only param (default 1).
            self.param = param

        def fit(self, X, y=None):
            # [FUNCTION DOC]
            # HELPER: Sets is_fitted_ True and returns self; ignores data.
            self.is_fitted_ = True
            return self

        def predict(self, X):
            # [FUNCTION DOC]
            # HELPER: Returns an array of length n_samples filled with param.
            return np.full(shape=X.shape[0], fill_value=self.param)

    my_pipeline = Pipeline([("estimator", MyEstimator(param=1))])
    with pytest.raises(AttributeError, match="The following error was raised"):
        my_pipeline.fit(X, y).predict(X)

    # 2nd case, the estimator doesn't implement `__sklearn_tags__` at all.
    class MyEstimator2:
        # [CLASS DOC]
        # HELPER: Plain class with no __sklearn_tags__ at all.
        def __init__(self, *, param=1):
            # [FUNCTION DOC]
            # HELPER: Stores keyword-only param (default 1).
            self.param = param

        def fit(self, X, y=None):
            # [FUNCTION DOC]
            # HELPER: Sets is_fitted_ True and returns self.
            self.is_fitted_ = True
            return self

        def predict(self, X):
            # [FUNCTION DOC]
            # HELPER: Returns an array of length n_samples filled with param.
            return np.full(shape=X.shape[0], fill_value=self.param)

    my_pipeline = Pipeline([("estimator", MyEstimator2(param=1))])
    with pytest.raises(AttributeError, match="The following error was raised"):
        my_pipeline.fit(X, y).predict(X)

    # check that we still raise an error if it is not an AttributeError or related to
    # __sklearn_tags__
    class MyEstimator3(MyEstimator, BaseEstimator):
        # [CLASS DOC]
        # HELPER: Estimator with a concrete base whose __sklearn_tags__ raises a
        # configurable error.
        def __init__(self, *, param=1, error_type=AttributeError):
            # [FUNCTION DOC]
            # HELPER: Stores param (default 1) and error_type (exception class, default
            # AttributeError).
            self.param = param
            self.error_type = error_type

        def __sklearn_tags__(self):
            # [FUNCTION DOC]
            # HELPER: Calls the parent implementation, then raises error_type('test').
            # Exceptions: always raises error_type.
            super().__sklearn_tags__()
            raise self.error_type("test")

    for error_type in (AttributeError, TypeError, ValueError):
        estimator = MyEstimator3(param=1, error_type=error_type)
        with pytest.raises(error_type):
            get_tags(estimator)
