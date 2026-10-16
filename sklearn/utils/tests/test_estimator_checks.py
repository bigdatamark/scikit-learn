# [MODULE DOC]
# Tests for sklearn.utils.estimator_checks. Defines many deliberately broken toy
# estimators (each violating one API contract) and asserts the corresponding check
# function detects the violation, plus tests for check_estimator itself
# (on_fail modes, expected failures, generator, clones, tags, warnings). Written
# without a hard pytest dependency: it uses sklearn's own raises/SkipTest helpers and
# only applies pytest marks when pytest is already imported.
# We can not use pytest here, because we run
# build_tools/github/test_pytest_soft_dependency.sh on these
# tests to make sure estimator_checks works without pytest.

import importlib
import re
import sys
import unittest
import warnings
from inspect import isgenerator
from numbers import Integral, Real

import joblib
import numpy as np
import scipy.sparse as sp

from sklearn import config_context, get_config
from sklearn.base import BaseEstimator, ClassifierMixin, OutlierMixin, TransformerMixin
from sklearn.cluster import MiniBatchKMeans
from sklearn.datasets import (
    load_iris,
    make_multilabel_classification,
)
from sklearn.decomposition import PCA
from sklearn.exceptions import (
    ConvergenceWarning,
    EstimatorCheckFailedWarning,
    SkipTestWarning,
)
from sklearn.linear_model import (
    LinearRegression,
    LogisticRegression,
    MultiTaskElasticNet,
    SGDClassifier,
)
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, NuSVC
from sklearn.utils import _array_api, all_estimators, deprecated
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils._test_common.instance_generator import (
    _construct_instances,
    _get_expected_failed_checks,
)
from sklearn.utils._testing import (
    MinimalClassifier,
    MinimalRegressor,
    MinimalTransformer,
    SkipTest,
    ignore_warnings,
    raises,
)
from sklearn.utils.estimator_checks import (
    _check_name,
    _NotAnArray,
    _yield_all_checks,
    check_array_api_input,
    check_class_weight_balanced_linear_classifier,
    check_classifier_data_not_an_array,
    check_classifier_not_supporting_multiclass,
    check_classifiers_multilabel_output_format_decision_function,
    check_classifiers_multilabel_output_format_predict,
    check_classifiers_multilabel_output_format_predict_proba,
    check_classifiers_one_label_sample_weights,
    check_dataframe_column_names_consistency,
    check_decision_proba_consistency,
    check_dict_unchanged,
    check_dont_overwrite_parameters,
    check_estimator,
    check_estimator_cloneable,
    check_estimator_repr,
    check_estimator_sparse_array,
    check_estimator_sparse_matrix,
    check_estimator_sparse_tag,
    check_estimator_tags_renamed,
    check_estimators_nan_inf,
    check_estimators_overwrite_params,
    check_estimators_unfitted,
    check_fit_check_is_fitted,
    check_fit_score_takes_y,
    check_methods_sample_order_invariance,
    check_methods_subset_invariance,
    check_mixin_order,
    check_no_attributes_set_in_init,
    check_outlier_contamination,
    check_outlier_corruption,
    check_parameters_default_constructible,
    check_positive_only_tag_during_fit,
    check_regressor_data_not_an_array,
    check_requires_y_none,
    check_sample_weights_pandas_series,
    check_set_params,
    estimator_checks_generator,
    set_random_state,
)
from sklearn.utils.fixes import CSR_CONTAINERS, SPARRAY_PRESENT
from sklearn.utils.metaestimators import available_if
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import (
    check_array,
    check_is_fitted,
    check_X_y,
    validate_data,
)


def _mark_thread_unsafe_if_pytest_imported(f):
    # [FUNCTION DOC]
    # HELPER: Decorator that applies pytest.mark.thread_unsafe to f when pytest is already
    # imported (looked up in sys.modules), else returns f unchanged.
    # Inputs: f - test function. Outputs: f or the marked function.
    pytest = sys.modules.get("pytest")
    if pytest is not None:
        return pytest.mark.thread_unsafe(f)
    else:
        return f


def _mark_no_check_spmatrix_if_pytest_imported(f):
    # [FUNCTION DOC]
    # HELPER: Decorator meant to apply pytest.mark.no_check_spmatrix to f when pytest is
    # imported, else return f unchanged.
    # KNOWN ISSUE: when pytest is imported it returns the bare MarkDecorator instead of
    # applying it to f, so the decorated test is replaced by the marker object itself
    # (unlike _mark_thread_unsafe_if_pytest_imported, which calls the mark on f).
    pytest = sys.modules.get("pytest")
    if pytest is not None:
        return pytest.mark.no_check_spmatrix
    else:
        return f


class CorrectNotFittedError(ValueError):
    # [CLASS DOC]
    # HELPER: ValueError subclass (not AttributeError) used to emulate a custom
    # not-fitted error in tests; no behavior of its own.
    """Exception class to raise if estimator is used before fitting.

    Like NotFittedError, it inherits from ValueError, but not from
    AttributeError. Used for testing only.
    """


class BaseBadClassifier(ClassifierMixin, BaseEstimator):
    # [CLASS DOC]
    # HELPER: Minimal classifier base that does no validation: fit returns self and
    # predict returns ones. Subclassed by other broken estimators.
    def fit(self, X, y):
        # [FUNCTION DOC]
        # HELPER: No-op fit; ignores X and y. Outputs: self.
        return self

    def predict(self, X):
        # [FUNCTION DOC]
        # HELPER: Returns a float array of ones of length X.shape[0]; no input validation.
        return np.ones(X.shape[0])


class ChangesDict(BaseEstimator):
    # [CLASS DOC]
    # HELPER: Broken estimator that mutates its key parameter during predict, used to
    # trigger check_dict_unchanged.
    def __init__(self, key=0):
        # [FUNCTION DOC]
        # HELPER: Stores key (default 0) as a parameter.
        self.key = key

    def fit(self, X, y=None):
        # [FUNCTION DOC]
        # HELPER: Validates X, y via validate_data (sets n_features_in_). Outputs: self.
        X, y = validate_data(self, X, y)
        return self

    def predict(self, X):
        # [FUNCTION DOC]
        # HELPER: Validates X, then sets self.key to 1000 (the deliberate state change) and
        # returns ones of length n_samples.
        X = check_array(X)
        self.key = 1000
        return np.ones(X.shape[0])


class SetsWrongAttribute(BaseEstimator):
    # [CLASS DOC]
    # HELPER: Broken estimator that adds a public attribute wrong_attribute in fit, used
    # to trigger check_dont_overwrite_parameters.
    def __init__(self, acceptable_key=0):
        # [FUNCTION DOC]
        # HELPER: Stores acceptable_key (default 0).
        self.acceptable_key = acceptable_key

    def fit(self, X, y=None):
        # [FUNCTION DOC]
        # HELPER: Sets public wrong_attribute = 0 then validates data. Outputs: self.
        self.wrong_attribute = 0
        X, y = validate_data(self, X, y)
        return self


class ChangesWrongAttribute(BaseEstimator):
    # [CLASS DOC]
    # HELPER: Broken estimator whose fit overwrites its own constructor parameter, used to
    # trigger check_estimators_overwrite_params.
    def __init__(self, wrong_attribute=0):
        # [FUNCTION DOC]
        # HELPER: Stores wrong_attribute (default 0).
        self.wrong_attribute = wrong_attribute

    def fit(self, X, y=None):
        # [FUNCTION DOC]
        # HELPER: Sets wrong_attribute to 1 (mutating a parameter) then validates data.
        self.wrong_attribute = 1
        X, y = validate_data(self, X, y)
        return self


class ChangesUnderscoreAttribute(BaseEstimator):
    # [CLASS DOC]
    # HELPER: Conforming estimator that only sets a private underscore attribute in fit.
    def fit(self, X, y=None):
        # [FUNCTION DOC]
        # HELPER: Sets _good_attribute = 1 and validates data. Outputs: self.
        self._good_attribute = 1
        X, y = validate_data(self, X, y)
        return self


class RaisesErrorInSetParams(BaseEstimator):
    # [CLASS DOC]
    # HELPER: Estimator whose set_params raises ValueError for negative p, used to check
    # check_set_params tolerates (with a warning) errors raised in set_params.
    def __init__(self, p=0):
        # [FUNCTION DOC]
        # HELPER: Stores p (default 0).
        self.p = p

    def set_params(self, **kwargs):
        # [FUNCTION DOC]
        # HELPER: If p is passed, raises ValueError when p < 0, else assigns it; remaining
        # kwargs are forwarded to BaseEstimator.set_params. Outputs: self.
        if "p" in kwargs:
            p = kwargs.pop("p")
            if p < 0:
                raise ValueError("p can't be less than 0")
            self.p = p
        return super().set_params(**kwargs)

    def fit(self, X, y=None):
        # [FUNCTION DOC]
        # HELPER: Validates data. Outputs: self.
        X, y = validate_data(self, X, y)
        return self


class HasMutableParameters(BaseEstimator):
    # [CLASS DOC]
    # HELPER: Estimator with a mutable default parameter (an object instance), which
    # check_parameters_default_constructible must reject.
    def __init__(self, p=object()):
        # [FUNCTION DOC]
        # HELPER: Stores p, defaulting to a shared object() instance created at definition.
        self.p = p

    def fit(self, X, y=None):
        # [FUNCTION DOC]
        # HELPER: Validates data. Outputs: self.
        X, y = validate_data(self, X, y)
        return self


class HasImmutableParameters(BaseEstimator):
    # [CLASS DOC]
    # HELPER: Estimator whose defaults are all allowed immutable values (int, numpy int,
    # a class object).
    # Note that object is an uninitialized class, thus immutable.
    def __init__(self, p=42, q=np.int32(42), r=object):
        # [FUNCTION DOC]
        # HELPER: Stores p (42), q (np.int32 42) and r (the object class).
        self.p = p
        self.q = q
        self.r = r

    def fit(self, X, y=None):
        # [FUNCTION DOC]
        # HELPER: Validates data. Outputs: self.
        X, y = validate_data(self, X, y)
        return self


class ModifiesValueInsteadOfRaisingError(BaseEstimator):
    # [CLASS DOC]
    # HELPER: Estimator whose set_params silently clamps negative p to 0 instead of
    # raising, so get_params disagrees with what was set.
    def __init__(self, p=0):
        # [FUNCTION DOC]
        # HELPER: Stores p (default 0).
        self.p = p

    def set_params(self, **kwargs):
        # [FUNCTION DOC]
        # HELPER: If p is passed, replaces negative values by 0 and assigns it; forwards the
        # rest to BaseEstimator.set_params. Outputs: self.
        if "p" in kwargs:
            p = kwargs.pop("p")
            if p < 0:
                p = 0
            self.p = p
        return super().set_params(**kwargs)

    def fit(self, X, y=None):
        # [FUNCTION DOC]
        # HELPER: Validates data. Outputs: self.
        X, y = validate_data(self, X, y)
        return self


class ModifiesAnotherValue(BaseEstimator):
    # [CLASS DOC]
    # HELPER: Estimator whose set_params changes parameter b as a side effect of setting
    # a to None, so check_set_params detects a mismatch.
    def __init__(self, a=0, b="method1"):
        # [FUNCTION DOC]
        # HELPER: Stores a (default 0) and b (default "method1").
        self.a = a
        self.b = b

    def set_params(self, **kwargs):
        # [FUNCTION DOC]
        # HELPER: If a is passed, assigns it; when a is None also drops b from kwargs and
        # forces b to "method2". Forwards remaining kwargs to BaseEstimator.set_params.
        # Exceptions: KeyError if a is None and b is not among the kwargs.
        if "a" in kwargs:
            a = kwargs.pop("a")
            self.a = a
            if a is None:
                kwargs.pop("b")
                self.b = "method2"
        return super().set_params(**kwargs)

    def fit(self, X, y=None):
        # [FUNCTION DOC]
        # HELPER: Validates data. Outputs: self.
        X, y = validate_data(self, X, y)
        return self


class NoCheckinPredict(BaseBadClassifier):
    # [CLASS DOC]
    # HELPER: Classifier that validates in fit but not in predict (inherits the
    # unvalidated predict), so NaN/inf in predict are not rejected.
    def fit(self, X, y):
        # [FUNCTION DOC]
        # HELPER: Validates X, y. Outputs: self.
        X, y = validate_data(self, X, y)
        return self


class NoSparseClassifier(BaseBadClassifier):
    # [CLASS DOC]
    # HELPER: Classifier that accepts sparse CSR/CSC in validation but then raises an
    # uninformative ValueError for the configured sparse type, used to check
    # check_estimator_sparse_matrix/array detect non-graceful failures.
    def __init__(self, raise_for_type=None):
        # [FUNCTION DOC]
        # HELPER: Stores raise_for_type: "sparse_array", "sparse_matrix" or None.
        # raise_for_type : str, expects "sparse_array" or "sparse_matrix"
        self.raise_for_type = raise_for_type

    def fit(self, X, y):
        # [FUNCTION DOC]
        # HELPER: Validates X (accepting csr/csc), then raises ValueError "Nonsensical Error"
        # if X is a scipy sparray (resp. spmatrix) per raise_for_type. Outputs: self.
        # KNOWN ISSUE: if raise_for_type is neither value (e.g. the default None),
        # correct_type is never assigned and an UnboundLocalError is raised.
        X, y = validate_data(self, X, y, accept_sparse=["csr", "csc"])
        if self.raise_for_type == "sparse_array":
            correct_type = isinstance(X, sp.sparray)
        elif self.raise_for_type == "sparse_matrix":
            correct_type = isinstance(X, sp.spmatrix)
        if correct_type:
            raise ValueError("Nonsensical Error")
        return self

    def predict(self, X):
        # [FUNCTION DOC]
        # HELPER: Validates X and returns ones; does not check fitted state.
        X = check_array(X)
        return np.ones(X.shape[0])


class CorrectNotFittedErrorClassifier(BaseBadClassifier):
    # [CLASS DOC]
    # HELPER: Classifier that correctly raises NotFittedError (via check_is_fitted) when
    # predict is called before fit.
    def fit(self, X, y):
        # [FUNCTION DOC]
        # HELPER: Validates data and sets coef_ to ones of length n_features. Outputs: self.
        X, y = validate_data(self, X, y)
        self.coef_ = np.ones(X.shape[1])
        return self

    def predict(self, X):
        # [FUNCTION DOC]
        # HELPER: Checks fitted state, validates X, returns ones of length n_samples.
        check_is_fitted(self)
        X = check_array(X)
        return np.ones(X.shape[0])


class NoSampleWeightPandasSeriesType(BaseEstimator):
    # [CLASS DOC]
    # HELPER: Regressor-like estimator that rejects sample_weight given as a pandas Series.
    def fit(self, X, y, sample_weight=None):
        # [FUNCTION DOC]
        # HELPER: Validates data (sparse csr/csc, multi-output, numeric y), then raises
        # ValueError if sample_weight is a pandas Series (pandas imported lazily).
        # Convert data
        X, y = validate_data(
            self, X, y, accept_sparse=("csr", "csc"), multi_output=True, y_numeric=True
        )
        # Function is only called after we verify that pandas is installed
        from pandas import Series

        if isinstance(sample_weight, Series):
            raise ValueError(
                "Estimator does not accept 'sample_weight'of type pandas.Series"
            )
        return self

    def predict(self, X):
        # [FUNCTION DOC]
        # HELPER: Validates X and returns ones.
        X = check_array(X)
        return np.ones(X.shape[0])


class BadBalancedWeightsClassifier(BaseBadClassifier):
    # [CLASS DOC]
    # HELPER: Classifier whose "balanced" class weights are deliberately off by one, to
    # trigger check_class_weight_balanced_linear_classifier.
    def __init__(self, class_weight=None):
        # [FUNCTION DOC]
        # HELPER: Stores class_weight (default None).
        self.class_weight = class_weight

    def fit(self, X, y):
        # [FUNCTION DOC]
        # HELPER: Computes class weights for the label-encoded classes, adds 1 when
        # class_weight is "balanced", and stores the result as coef_. Outputs: self.
        from sklearn.preprocessing import LabelEncoder
        from sklearn.utils import compute_class_weight

        label_encoder = LabelEncoder().fit(y)
        classes = label_encoder.classes_
        class_weight = compute_class_weight(self.class_weight, classes=classes, y=y)

        # Intentionally modify the balanced class_weight
        # to simulate a bug and raise an exception
        if self.class_weight == "balanced":
            class_weight += 1.0

        # Simply assigning coef_ to the class_weight
        self.coef_ = class_weight
        return self


class BadTransformerWithoutMixin(BaseEstimator):
    # [CLASS DOC]
    # HELPER: Transformer that lacks TransformerMixin (thus no transformer tags), used to
    # check check_estimator complains about the missing tag.
    def fit(self, X, y=None):
        # [FUNCTION DOC]
        # HELPER: Validates X (sets n_features_in_). Outputs: self.
        X = validate_data(self, X)
        return self

    def transform(self, X):
        # [FUNCTION DOC]
        # HELPER: Checks fitted state and returns the validated X unchanged.
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)
        return X


class NotInvariantPredict(BaseEstimator):
    # [CLASS DOC]
    # HELPER: Estimator whose predictions depend on batch size (not subset invariant).
    def fit(self, X, y):
        # [FUNCTION DOC]
        # HELPER: Validates data. Outputs: self.
        # Convert data
        X, y = validate_data(
            self, X, y, accept_sparse=("csr", "csc"), multi_output=True, y_numeric=True
        )
        return self

    def predict(self, X):
        # [FUNCTION DOC]
        # HELPER: Returns ones when more than one sample is given, zeros for a single sample.
        # return 1 if X has more than one element else return 0
        X = check_array(X)
        if X.shape[0] > 1:
            return np.ones(X.shape[0])
        return np.zeros(X.shape[0])


class NotInvariantSampleOrder(BaseEstimator):
    # [CLASS DOC]
    # HELPER: Estimator whose predictions change when samples are permuted.
    def fit(self, X, y):
        # [FUNCTION DOC]
        # HELPER: Validates data and stores the training X in self._X. Outputs: self.
        X, y = validate_data(
            self, X, y, accept_sparse=("csr", "csc"), multi_output=True, y_numeric=True
        )
        # store the original X to check for sample order later
        self._X = X
        return self

    def predict(self, X):
        # [FUNCTION DOC]
        # HELPER: If X has the same per-column sorted content as the training X but differs
        # element-wise (a reordering), returns zeros; otherwise returns the first column.
        X = check_array(X)
        # if the input contains the same elements but different sample order,
        # then just return zeros.
        if (
            np.array_equiv(np.sort(X, axis=0), np.sort(self._X, axis=0))
            and (X != self._X).any()
        ):
            return np.zeros(X.shape[0])
        return X[:, 0]


class OneClassSampleErrorClassifier(BaseBadClassifier):
    # [CLASS DOC]
    # HELPER: Classifier whose error when sample weights leave a single class lacks the
    # word "class" unless raise_when_single_class triggers the normal message.
    """Classifier allowing to trigger different behaviors when `sample_weight` reduces
    the number of classes to 1."""

    def __init__(self, raise_when_single_class=False):
        # [FUNCTION DOC]
        # HELPER: Stores raise_when_single_class (default False).
        self.raise_when_single_class = raise_when_single_class

    def fit(self, X, y, sample_weight=None):
        # [FUNCTION DOC]
        # HELPER: Validates data, sets has_single_class_ False and classes_ via unique. Raises
        # ValueError "normal class error" if there is one class and the flag is set. If
        # sample_weight is a non-empty ndarray, counts classes with non-zero total weight;
        # if fewer than two remain raises ValueError "Nonsensical Error" (setting
        # has_single_class_ True first). Outputs: self.
        X, y = check_X_y(
            X, y, accept_sparse=("csr", "csc"), multi_output=True, y_numeric=True
        )

        self.has_single_class_ = False
        self.classes_, y = np.unique(y, return_inverse=True)
        n_classes_ = self.classes_.shape[0]
        if n_classes_ < 2 and self.raise_when_single_class:
            self.has_single_class_ = True
            raise ValueError("normal class error")

        # find the number of class after trimming
        if sample_weight is not None:
            if isinstance(sample_weight, np.ndarray) and len(sample_weight) > 0:
                n_classes_ = np.count_nonzero(np.bincount(y, sample_weight))
            if n_classes_ < 2:
                self.has_single_class_ = True
                raise ValueError("Nonsensical Error")

        return self

    def predict(self, X):
        # [FUNCTION DOC]
        # HELPER: Checks fitted, validates X; returns zeros if has_single_class_ else ones.
        check_is_fitted(self)
        X = check_array(X)
        if self.has_single_class_:
            return np.zeros(X.shape[0])
        return np.ones(X.shape[0])


class LargeSparseNotSupportedClassifier(BaseEstimator):
    # [CLASS DOC]
    # HELPER: Estimator that declares accept_large_sparse=True but fails on 64-bit index
    # arrays for the configured sparse type.
    """Estimator that claims to support large sparse data
    (accept_large_sparse=True), but doesn't"""

    def __init__(self, raise_for_type=None):
        # [FUNCTION DOC]
        # HELPER: Stores raise_for_type: "sparse_array", "sparse_matrix" or None.
        # raise_for_type : str, expects "sparse_array" or "sparse_matrix"
        self.raise_for_type = raise_for_type

    def fit(self, X, y):
        # [FUNCTION DOC]
        # HELPER: Validates X (csr/csc/coo, large sparse allowed). If X matches the configured
        # sparse type: for COO raises ValueError when row or col indices are int64; for
        # CSR/CSC fails an assert when indices or indptr are int64. Outputs: self.
        # KNOWN ISSUE: same as NoSparseClassifier, correct_type is unbound when
        # raise_for_type is None, causing UnboundLocalError.
        X, y = validate_data(
            self,
            X,
            y,
            accept_sparse=("csr", "csc", "coo"),
            accept_large_sparse=True,
            multi_output=True,
            y_numeric=True,
        )
        if self.raise_for_type == "sparse_array":
            correct_type = isinstance(X, sp.sparray)
        elif self.raise_for_type == "sparse_matrix":
            correct_type = isinstance(X, sp.spmatrix)
        if correct_type:
            if X.format == "coo":
                if X.row.dtype == "int64" or X.col.dtype == "int64":
                    raise ValueError("Estimator doesn't support 64-bit indices")
            elif X.format in ["csc", "csr"]:
                assert "int64" not in (
                    X.indices.dtype,
                    X.indptr.dtype,
                ), "Estimator doesn't support 64-bit indices"

        return self


class SparseTransformer(TransformerMixin, BaseEstimator):
    # [CLASS DOC]
    # HELPER: Transformer that returns its input wrapped in a configurable sparse
    # container (non-regression for estimators that output sparse data).
    def __init__(self, sparse_container=None):
        # [FUNCTION DOC]
        # HELPER: Stores sparse_container, a callable such as a CSR class.
        self.sparse_container = sparse_container

    def fit(self, X, y=None):
        # [FUNCTION DOC]
        # HELPER: Validates X (sets n_features_in_). Outputs: self.
        validate_data(self, X)
        return self

    def fit_transform(self, X, y=None):
        # [FUNCTION DOC]
        # HELPER: Fits then transforms X.
        return self.fit(X, y).transform(X)

    def transform(self, X):
        # [FUNCTION DOC]
        # HELPER: Checks fitted, validates X (sparse allowed) and returns
        # sparse_container(X).
        check_is_fitted(self)
        X = validate_data(self, X, accept_sparse=True, reset=False)
        return self.sparse_container(X)


class EstimatorInconsistentForPandas(BaseEstimator):
    # [CLASS DOC]
    # HELPER: Estimator that memorises a different cell depending on whether X is a
    # pandas DataFrame (first cell) or an array (row 1, col 0), so its results differ
    # between data containers.
    def fit(self, X, y):
        # [FUNCTION DOC]
        # HELPER: Sets value_ to X.iloc[0, 0] for DataFrames, otherwise to the validated
        # X[1, 0] (also when pandas is not installed). Outputs: self.
        try:
            from pandas import DataFrame

            if isinstance(X, DataFrame):
                self.value_ = X.iloc[0, 0]
            else:
                X = check_array(X)
                self.value_ = X[1, 0]
            return self

        except ImportError:
            X = check_array(X)
            self.value_ = X[1, 0]
            return self

    def predict(self, X):
        # [FUNCTION DOC]
        # HELPER: Returns an array repeating value_ once per sample.
        X = check_array(X)
        return np.array([self.value_] * X.shape[0])


class UntaggedBinaryClassifier(SGDClassifier):
    # [CLASS DOC]
    # HELPER: SGDClassifier that raises on more than two classes but does not declare it
    # in its tags, so check_estimator fails.
    # Toy classifier that only supports binary classification, will fail tests.
    def fit(self, X, y, coef_init=None, intercept_init=None, sample_weight=None):
        # [FUNCTION DOC]
        # HELPER: Fits via SGDClassifier then raises ValueError if more than 2 classes.
        super().fit(X, y, coef_init, intercept_init, sample_weight)
        if len(self.classes_) > 2:
            raise ValueError("Only 2 classes are supported")
        return self

    def partial_fit(self, X, y, classes=None, sample_weight=None):
        # [FUNCTION DOC]
        # HELPER: partial_fit via SGDClassifier then raises ValueError if more than 2 classes.
        super().partial_fit(X=X, y=y, classes=classes, sample_weight=sample_weight)
        if len(self.classes_) > 2:
            raise ValueError("Only 2 classes are supported")
        return self


class TaggedBinaryClassifier(UntaggedBinaryClassifier):
    # [CLASS DOC]
    # HELPER: Binary-only classifier that declares multi_class=False in its tags and
    # raises an informative error for non-binary targets.
    def fit(self, X, y):
        # [FUNCTION DOC]
        # HELPER: Raises ValueError unless type_of_target(y) is "binary", then fits.
        y_type = type_of_target(y, input_name="y", raise_unknown=True)
        if y_type != "binary":
            raise ValueError(
                "Only binary classification is supported. The type of the target "
                f"is {y_type}."
            )
        return super().fit(X, y)

    # Toy classifier that only supports binary classification.
    def __sklearn_tags__(self):
        # [FUNCTION DOC]
        # HELPER: Returns parent tags with classifier_tags.multi_class set False.
        tags = super().__sklearn_tags__()
        tags.classifier_tags.multi_class = False
        return tags


class RequiresPositiveXRegressor(LinearRegression):
    # [CLASS DOC]
    # HELPER: LinearRegression that rejects negative X and declares positive_only and
    # non-sparse input tags.
    def fit(self, X, y):
        # [FUNCTION DOC]
        # HELPER: Validates dense X (multi-output y), raises ValueError on any negative X,
        # then fits LinearRegression.
        # reject sparse X to be able to call (X < 0).any()
        X, y = validate_data(self, X, y, accept_sparse=False, multi_output=True)
        if (X < 0).any():
            raise ValueError("Negative values in data passed to X.")
        return super().fit(X, y)

    def __sklearn_tags__(self):
        # [FUNCTION DOC]
        # HELPER: Returns parent tags with input_tags.positive_only True and sparse False.
        tags = super().__sklearn_tags__()
        tags.input_tags.positive_only = True
        # reject sparse X to be able to call (X < 0).any()
        tags.input_tags.sparse = False
        return tags


class RequiresPositiveYRegressor(LinearRegression):
    # [CLASS DOC]
    # HELPER: LinearRegression that rejects non-positive y and declares
    # target_tags.positive_only.
    def fit(self, X, y):
        # [FUNCTION DOC]
        # HELPER: Validates data (sparse ok), raises ValueError if any y <= 0, then fits.
        X, y = validate_data(self, X, y, accept_sparse=True, multi_output=True)
        if (y <= 0).any():
            raise ValueError("negative y values not supported!")
        return super().fit(X, y)

    def __sklearn_tags__(self):
        # [FUNCTION DOC]
        # HELPER: Returns parent tags with target_tags.positive_only True.
        tags = super().__sklearn_tags__()
        tags.target_tags.positive_only = True
        return tags


class PoorScoreLogisticRegression(LogisticRegression):
    # [CLASS DOC]
    # HELPER: LogisticRegression with shifted decision function (worse score) that sets
    # the poor_score classifier tag so score checks are relaxed.
    def decision_function(self, X):
        # [FUNCTION DOC]
        # HELPER: Returns the parent decision function plus 1.
        return super().decision_function(X) + 1

    def __sklearn_tags__(self):
        # [FUNCTION DOC]
        # HELPER: Returns parent tags with classifier_tags.poor_score True.
        tags = super().__sklearn_tags__()
        tags.classifier_tags.poor_score = True
        return tags


class PartialFitChecksName(BaseEstimator):
    # [CLASS DOC]
    # HELPER: Estimator whose partial_fit validates feature names consistently across
    # calls (reset only on first call).
    def fit(self, X, y):
        # [FUNCTION DOC]
        # HELPER: Validates data (records feature names). Outputs: self.
        validate_data(self, X, y)
        return self

    def partial_fit(self, X, y):
        # [FUNCTION DOC]
        # HELPER: Validates data with reset only when not yet fitted, then sets _fitted True.
        reset = not hasattr(self, "_fitted")
        validate_data(self, X, y, reset=reset)
        self._fitted = True
        return self


class BrokenArrayAPI(BaseEstimator):
    # [CLASS DOC]
    # HELPER: Estimator returning different predictions with and without array API
    # dispatch, used to make check_array_api_input fail.
    """Make different predictions when using Numpy and the Array API"""

    def fit(self, X, y):
        # [FUNCTION DOC]
        # HELPER: No-op fit. Outputs: self.
        return self

    def predict(self, X):
        # [FUNCTION DOC]
        # HELPER: Returns [1, 2, 3] in X's namespace when array_api_dispatch is enabled,
        # else the NumPy array [3, 2, 1] (ignores X's length).
        enabled = get_config()["array_api_dispatch"]
        xp, _ = _array_api.get_namespace(X)
        if enabled:
            return xp.asarray([1, 2, 3])
        else:
            return np.array([3, 2, 1])


def test_check_array_api_input():
    # [FUNCTION DOC]
    # Verifies check_array_api_input with array_api_strict and check_values detects
    # BrokenArrayAPI's mismatch ("Not equal to tolerance"). Skips without
    # array-api-strict.
    try:
        importlib.import_module("array_api_strict")
    except ModuleNotFoundError:  # pragma: nocover
        raise SkipTest("array-api-strict is required to run this test")

    with raises(AssertionError, match="Not equal to tolerance"):
        check_array_api_input(
            "BrokenArrayAPI",
            BrokenArrayAPI(),
            array_namespace="array_api_strict",
            check_values=True,
        )


def test_not_an_array_array_function():
    # [FUNCTION DOC]
    # Verifies _NotAnArray raises TypeError when NumPy's array_function protocol is used
    # (np.sum), while np.may_share_memory always returns True for it.
    not_array = _NotAnArray(np.ones(10))
    msg = "Don't want to call array_function sum!"
    with raises(TypeError, match=msg):
        np.sum(not_array)
    # always returns True
    assert np.may_share_memory(not_array, None)


def test_check_fit_score_takes_y_works_on_deprecated_fit():
    # [FUNCTION DOC]
    # Verifies check_fit_score_takes_y runs on an estimator whose fit is wrapped by the
    # deprecated decorator (signature inspection must see through it).
    # Tests that check_fit_score_takes_y works on a class with
    # a deprecated fit method

    class TestEstimatorWithDeprecatedFitMethod(BaseEstimator):
        # [CLASS DOC]
        # HELPER: Estimator with a deprecated no-op fit(X, y).
        @deprecated("Deprecated for the purpose of testing check_fit_score_takes_y")
        def fit(self, X, y):
            # [FUNCTION DOC]
            # HELPER: Deprecated no-op fit (emits FutureWarning when called). Outputs: self.
            return self

    check_fit_score_takes_y("test", TestEstimatorWithDeprecatedFitMethod())


def test_check_estimator_with_class_removed():
    # [FUNCTION DOC]
    # Verifies check_estimator raises TypeError when given a class instead of an instance.
    """Test that passing a class instead of an instance fails."""
    msg = "Passing a class was deprecated"
    with raises(TypeError, match=msg):
        check_estimator(LogisticRegression)


def test_mutable_default_params():
    # [FUNCTION DOC]
    # Verifies check_parameters_default_constructible passes for immutable defaults and
    # raises AssertionError naming the parameter for an object() default.
    """Test that constructor cannot have mutable default parameters."""
    msg = (
        "Parameter 'p' of estimator 'HasMutableParameters' is of type "
        "object which is not allowed"
    )
    # check that the "default_constructible" test checks for mutable parameters
    check_parameters_default_constructible(
        "Immutable", HasImmutableParameters()
    )  # should pass
    with raises(AssertionError, match=msg):
        check_parameters_default_constructible("Mutable", HasMutableParameters())


@_mark_thread_unsafe_if_pytest_imported
def test_check_set_params():
    # [FUNCTION DOC]
    # Verifies check_set_params fails on clamping and on cross-parameter-modifying
    # set_params, and only warns (UserWarning) when set_params raises. Thread-unsafe.
    """Check set_params doesn't fail and sets the right values."""
    # check that values returned by get_params match set_params
    msg = "get_params result does not match what was passed to set_params"
    with raises(AssertionError, match=msg):
        check_set_params("test", ModifiesValueInsteadOfRaisingError())

    with warnings.catch_warnings(record=True) as records:
        check_set_params("test", RaisesErrorInSetParams())
    assert UserWarning in [rec.category for rec in records]

    with raises(AssertionError, match=msg):
        check_set_params("test", ModifiesAnotherValue())


def test_check_estimators_nan_inf():
    # [FUNCTION DOC]
    # Verifies check_estimators_nan_inf flags an estimator not validating predict input.
    # check that predict does input validation (doesn't accept dicts in input)
    msg = "Estimator NoCheckinPredict doesn't check for NaN and inf in predict"
    with raises(AssertionError, match=msg):
        check_estimators_nan_inf("NoCheckinPredict", NoCheckinPredict())


def test_check_dict_unchanged():
    # [FUNCTION DOC]
    # Verifies check_dict_unchanged flags an estimator mutating __dict__ in predict.
    # check that estimator state does not change
    # at transform/predict/predict_proba time
    msg = "Estimator changes __dict__ during predict"
    with raises(AssertionError, match=msg):
        check_dict_unchanged("test", ChangesDict())


def test_check_sample_weights_pandas_series():
    # [FUNCTION DOC]
    # Verifies check_sample_weights_pandas_series propagates a ValueError mentioning the
    # estimator when it rejects Series weights; silently passes without pandas.
    # check that sample_weights in fit accepts pandas.Series type
    try:
        from pandas import Series  # noqa: F401

        msg = (
            "Estimator NoSampleWeightPandasSeriesType raises error if "
            "'sample_weight' parameter is of type pandas.Series"
        )
        with raises(ValueError, match=msg):
            check_sample_weights_pandas_series(
                "NoSampleWeightPandasSeriesType", NoSampleWeightPandasSeriesType()
            )
    except ImportError:
        pass


def test_check_estimators_overwrite_params():
    # [FUNCTION DOC]
    # Verifies check_estimators_overwrite_params flags mutation of a public parameter in
    # fit and accepts setting only a private attribute.
    # check that `fit` only changes attributes that
    # are private (start with an _ or end with a _).
    msg = (
        "Estimator ChangesWrongAttribute should not change or mutate  "
        "the parameter wrong_attribute from 0 to 1 during fit."
    )
    with raises(AssertionError, match=msg):
        check_estimators_overwrite_params(
            "ChangesWrongAttribute", ChangesWrongAttribute()
        )
    check_estimators_overwrite_params("test", ChangesUnderscoreAttribute())


def test_check_dont_overwrite_parameters():
    # [FUNCTION DOC]
    # Verifies check_dont_overwrite_parameters flags a public attribute added during fit.
    # check that `fit` doesn't add any public attribute
    msg = (
        r"Estimator adds public attribute\(s\) during the fit method."
        " Estimators are only allowed to add private attributes"
        " either started with _ or ended"
        " with _ but wrong_attribute added"
    )
    with raises(AssertionError, match=msg):
        check_dont_overwrite_parameters("test", SetsWrongAttribute())


def test_check_methods_sample_order_invariance():
    # [FUNCTION DOC]
    # Verifies check_methods_sample_order_invariance flags NotInvariantSampleOrder.
    # check for sample order invariance
    name = NotInvariantSampleOrder.__name__
    method = "predict"
    msg = (
        "{method} of {name} is not invariant when applied to a dataset"
        "with different sample order."
    ).format(method=method, name=name)
    with raises(AssertionError, match=msg):
        check_methods_sample_order_invariance(
            "NotInvariantSampleOrder", NotInvariantSampleOrder()
        )


def test_check_methods_subset_invariance():
    # [FUNCTION DOC]
    # Verifies check_methods_subset_invariance flags NotInvariantPredict.
    # check for invariant method
    name = NotInvariantPredict.__name__
    method = "predict"
    msg = ("{method} of {name} is not invariant when applied to a subset.").format(
        method=method, name=name
    )
    with raises(AssertionError, match=msg):
        check_methods_subset_invariance("NotInvariantPredict", NotInvariantPredict())


def test_check_estimator_sparse_data():
    # [FUNCTION DOC]
    # Verifies sparse checks (matrix and, when available, array variants) flag a
    # classifier that fails non-gracefully on sparse input and one that claims 64-bit
    # index support but fails on it.
    # check for sparse data input handling
    name = NoSparseClassifier.__name__
    msg = "Estimator %s doesn't seem to fail gracefully on sparse data" % name
    with raises(AssertionError, match=msg):
        check_estimator_sparse_matrix(name, NoSparseClassifier("sparse_matrix"))

    if SPARRAY_PRESENT:
        with raises(AssertionError, match=msg):
            check_estimator_sparse_array(name, NoSparseClassifier("sparse_array"))

    # Large indices test on bad estimator
    msg = (
        "Estimator LargeSparseNotSupportedClassifier doesn't seem to "
        r"support \S{3}_64 matrix, and is not failing gracefully.*"
    )
    with raises(AssertionError, match=msg):
        check_estimator_sparse_matrix(
            "LargeSparseNotSupportedClassifier",
            LargeSparseNotSupportedClassifier("sparse_matrix"),
        )

    if SPARRAY_PRESENT:
        with raises(AssertionError, match=msg):
            check_estimator_sparse_array(
                "LargeSparseNotSupportedClassifier",
                LargeSparseNotSupportedClassifier("sparse_array"),
            )


def test_check_classifiers_one_label_sample_weights():
    # [FUNCTION DOC]
    # Verifies check_classifiers_one_label_sample_weights flags an error message lacking
    # "class" when sample weights reduce training data to one label.
    # check for classifiers reducing to less than two classes via sample weights
    name = OneClassSampleErrorClassifier.__name__
    msg = (
        f"{name} failed when fitted on one label after sample_weight "
        "trimming. Error message is not explicit, it should have "
        "'class'."
    )
    with raises(AssertionError, match=msg):
        check_classifiers_one_label_sample_weights(
            "OneClassSampleErrorClassifier", OneClassSampleErrorClassifier()
        )


def test_check_estimator_not_fail_fast():
    # [FUNCTION DOC]
    # Verifies check_estimator with on_fail=None returns a non-empty list of result dicts
    # with the expected keys, containing both failed and passed checks for BaseEstimator.
    """Check the contents of the results returned with on_fail!="raise".

    This results should contain details about the observed failures, expected
    or not.
    """
    check_results = check_estimator(BaseEstimator(), on_fail=None)
    assert isinstance(check_results, list)
    assert len(check_results) > 0
    assert all(
        isinstance(item, dict)
        and set(item.keys())
        == {
            "estimator",
            "check_name",
            "exception",
            "status",
            "expected_to_fail",
            "expected_to_fail_reason",
        }
        for item in check_results
    )
    # Some tests are expected to fail, some are expected to pass.
    assert any(item["status"] == "failed" for item in check_results)
    assert any(item["status"] == "passed" for item in check_results)


@_mark_no_check_spmatrix_if_pytest_imported  # pickle breaks check_spmatrix
# Some estimator checks rely on warnings in deep functions calls. This is not
# automatically detected by pytest-run-parallel shallow AST inspection, so we
# need to mark the test function as thread-unsafe.
@_mark_thread_unsafe_if_pytest_imported
def test_check_estimator():
    # [FUNCTION DOC]
    # Verifies check_estimator fails on BaseEstimator (no fit) and on the untagged binary
    # classifier, but passes on sparse-output transformers, LogisticRegression,
    # MultiTaskElasticNet, tagged binary and positive-X regressors and the poor-score
    # classifier; and that the positive-y regressor still raises its ValueError.
    # tests that the estimator actually fails on "bad" estimators.
    # not a complete test of all checks, which are very extensive.

    # check that we have a fit method
    msg = "object has no attribute 'fit'"
    with raises(AttributeError, match=msg):
        check_estimator(BaseEstimator())

    # does error on binary_only untagged estimator
    msg = "Only 2 classes are supported"
    with raises(ValueError, match=msg):
        check_estimator(UntaggedBinaryClassifier())

    for csr_container in CSR_CONTAINERS:
        # non-regression test for estimators transforming to sparse data
        check_estimator(SparseTransformer(sparse_container=csr_container))

    # doesn't error on actual estimator
    check_estimator(LogisticRegression())
    check_estimator(LogisticRegression(C=0.01))
    check_estimator(MultiTaskElasticNet())

    # doesn't error on binary_only tagged estimator
    check_estimator(TaggedBinaryClassifier())
    check_estimator(RequiresPositiveXRegressor())

    # Check regressor with requires_positive_y estimator tag
    msg = "negative y values not supported!"
    with raises(ValueError, match=msg):
        check_estimator(RequiresPositiveYRegressor())

    # Does not raise error on classifier with poor_score tag
    check_estimator(PoorScoreLogisticRegression())


def test_check_outlier_corruption():
    # [FUNCTION DOC]
    # Verifies check_outlier_corruption raises when decision values at the threshold
    # boundary are not tied as expected and passes when they are.
    # should raise AssertionError
    decision = np.array([0.0, 1.0, 1.5, 2.0])
    with raises(AssertionError):
        check_outlier_corruption(1, 2, decision)
    # should pass
    decision = np.array([0.0, 1.0, 1.0, 2.0])
    check_outlier_corruption(1, 2, decision)


def test_check_estimator_sparse_tag():
    # [FUNCTION DOC]
    # Verifies check_estimator_sparse_tag passes when the sparse tag matches actual sparse
    # support and raises AssertionError otherwise, including when fit fails on sparse
    # data with an unexpected exception type (TypeError, KeyError).
    """Test that check_estimator_sparse_tag raises error when sparse tag is
    misaligned."""

    class EstimatorWithSparseConfig(BaseEstimator):
        # [CLASS DOC]
        # HELPER: Estimator with configurable sparse tag, sparse acceptance and an optional
        # exception to raise in fit.
        def __init__(self, tag_sparse, accept_sparse, fit_error=None):
            # [FUNCTION DOC]
            # HELPER: Stores tag_sparse, accept_sparse and fit_error (default None).
            self.tag_sparse = tag_sparse
            self.accept_sparse = accept_sparse
            self.fit_error = fit_error

        def fit(self, X, y=None):
            # [FUNCTION DOC]
            # HELPER: Raises fit_error if set; otherwise validates data with accept_sparse.
            if self.fit_error:
                raise self.fit_error
            validate_data(self, X, y, accept_sparse=self.accept_sparse)
            return self

        def __sklearn_tags__(self):
            # [FUNCTION DOC]
            # HELPER: Returns parent tags with input_tags.sparse set to tag_sparse.
            tags = super().__sklearn_tags__()
            tags.input_tags.sparse = self.tag_sparse
            return tags

    test_cases = [
        {"tag_sparse": True, "accept_sparse": True, "error_type": None},
        {"tag_sparse": False, "accept_sparse": False, "error_type": None},
        {"tag_sparse": False, "accept_sparse": True, "error_type": AssertionError},
        {"tag_sparse": True, "accept_sparse": False, "error_type": AssertionError},
    ]

    for test_case in test_cases:
        estimator = EstimatorWithSparseConfig(
            test_case["tag_sparse"],
            test_case["accept_sparse"],
        )
        if test_case["error_type"] is None:
            check_estimator_sparse_tag(estimator.__class__.__name__, estimator)
        else:
            with raises(test_case["error_type"]):
                check_estimator_sparse_tag(estimator.__class__.__name__, estimator)

    # estimator `tag_sparse=accept_sparse=False` fails on sparse data
    # but does not raise the appropriate error
    for fit_error in [TypeError("unexpected error"), KeyError("other error")]:
        estimator = EstimatorWithSparseConfig(False, False, fit_error)
        with raises(AssertionError):
            check_estimator_sparse_tag(estimator.__class__.__name__, estimator)


def test_check_estimator_transformer_no_mixin():
    # [FUNCTION DOC]
    # Verifies check_estimator raises RuntimeError about missing transformer_tags for a
    # transformer without TransformerMixin (pattern passed positionally as match).
    # check that TransformerMixin is not required for transformer tests to run
    # but it fails since the tag is not set
    with raises(RuntimeError, "the `transformer_tags` tag is not set"):
        check_estimator(BadTransformerWithoutMixin())


@_mark_no_check_spmatrix_if_pytest_imported  # pickle breaks check_spmatrix
def test_check_estimator_clones():
    # [FUNCTION DOC]
    # Verifies check_estimator does not mutate the passed estimator: joblib hashes before
    # and after are equal for several estimators, both unfitted and fitted on iris.
    # check that check_estimator doesn't modify the estimator it receives

    iris = load_iris()

    for Estimator in [
        GaussianMixture,
        LinearRegression,
        SGDClassifier,
        PCA,
        MiniBatchKMeans,
    ]:
        # without fitting
        with ignore_warnings(category=ConvergenceWarning):
            est = Estimator()
            set_random_state(est)
            old_hash = joblib.hash(est)
            check_estimator(
                est, expected_failed_checks=_get_expected_failed_checks(est)
            )
        assert old_hash == joblib.hash(est)

        # with fitting
        with ignore_warnings(category=ConvergenceWarning):
            est = Estimator()
            set_random_state(est)
            est.fit(iris.data, iris.target)
            old_hash = joblib.hash(est)
            check_estimator(
                est, expected_failed_checks=_get_expected_failed_checks(est)
            )
        assert old_hash == joblib.hash(est)


def test_check_estimators_unfitted():
    # [FUNCTION DOC]
    # Verifies check_estimators_unfitted flags an estimator that does not raise when
    # unfitted, and accepts one raising NotFittedError.
    # check that a ValueError/AttributeError is raised when calling predict
    # on an unfitted estimator
    msg = "Estimator should raise a NotFittedError when calling"
    with raises(AssertionError, match=msg):
        check_estimators_unfitted("estimator", NoSparseClassifier())

    # check that CorrectNotFittedError inherit from either ValueError
    # or AttributeError
    check_estimators_unfitted("estimator", CorrectNotFittedErrorClassifier())


def test_check_no_attributes_set_in_init():
    # [FUNCTION DOC]
    # Verifies check_no_attributes_set_in_init flags extra attributes set in __init__,
    # raises AttributeError when a parameter is not stored, and accepts class-level
    # __metadata_request__ attributes and set_fit_request with routing enabled.
    class NonConformantEstimatorPrivateSet(BaseEstimator):
        # [CLASS DOC]
        # HELPER: Sets a non-parameter attribute in __init__.
        def __init__(self):
            # [FUNCTION DOC]
            # HELPER: Sets you_should_not_set_this_ to None.
            self.you_should_not_set_this_ = None

    class NonConformantEstimatorNoParamSet(BaseEstimator):
        # [CLASS DOC]
        # HELPER: Declares a parameter but does not store it.
        def __init__(self, you_should_set_this_=None):
            # [FUNCTION DOC]
            # HELPER: Ignores its you_should_set_this_ parameter.
            pass

    class ConformantEstimatorClassAttribute(BaseEstimator):
        # [CLASS DOC]
        # HELPER: Conforming estimator with a class-level metadata request for "foo" in fit.
        # making sure our __metadata_request__* class attributes are okay!
        __metadata_request__fit = {"foo": True}

        def fit(self, X, y=None):
            # [FUNCTION DOC]
            # HELPER: No-op fit. Outputs: self.
            return self  # pragma: no cover

    msg = (
        "Estimator estimator_name should not set any"
        " attribute apart from parameters during init."
        r" Found attributes \['you_should_not_set_this_'\]."
    )
    with raises(AssertionError, match=msg):
        check_no_attributes_set_in_init(
            "estimator_name", NonConformantEstimatorPrivateSet()
        )

    msg = (
        "Estimator estimator_name should store all parameters as an attribute"
        " during init"
    )
    with raises(AttributeError, match=msg):
        check_no_attributes_set_in_init(
            "estimator_name", NonConformantEstimatorNoParamSet()
        )

    # a private class attribute is okay!
    check_no_attributes_set_in_init(
        "estimator_name", ConformantEstimatorClassAttribute()
    )
    # also check if cloning an estimator which has non-default set requests is
    # fine. Setting a non-default value via `set_{method}_request` sets the
    # private _metadata_request instance attribute which is copied in `clone`.
    with config_context(enable_metadata_routing=True):
        check_no_attributes_set_in_init(
            "estimator_name",
            ConformantEstimatorClassAttribute().set_fit_request(foo=True),
        )


# Some estimator checks rely on warnings in deep functions calls. This is not
# automatically detected by pytest-run-parallel shallow AST inspection, so we
# need to mark the test function as thread-unsafe.
@_mark_thread_unsafe_if_pytest_imported
def test_check_estimator_pairwise():
    # [FUNCTION DOC]
    # Verifies check_estimator passes on SVC with precomputed kernel and on
    # KNeighborsRegressor with precomputed metric (with its expected failures).
    # check that check_estimator() works on estimator with _pairwise
    # kernel or metric

    # test precomputed kernel
    est = SVC(kernel="precomputed")
    check_estimator(est)

    # test precomputed metric
    est = KNeighborsRegressor(metric="precomputed")
    check_estimator(est, expected_failed_checks=_get_expected_failed_checks(est))


def test_check_classifier_data_not_an_array():
    # [FUNCTION DOC]
    # Verifies check_classifier_data_not_an_array flags container-dependent results.
    with raises(AssertionError, match="Not equal to tolerance"):
        check_classifier_data_not_an_array(
            "estimator_name", EstimatorInconsistentForPandas()
        )


def test_check_regressor_data_not_an_array():
    # [FUNCTION DOC]
    # Verifies check_regressor_data_not_an_array flags container-dependent results.
    with raises(AssertionError, match="Not equal to tolerance"):
        check_regressor_data_not_an_array(
            "estimator_name", EstimatorInconsistentForPandas()
        )


def test_check_dataframe_column_names_consistency():
    # [FUNCTION DOC]
    # Verifies check_dataframe_column_names_consistency raises when feature_names_in_ is
    # not set, passes for PartialFitChecksName and LogisticRegression, and raises when the
    # docstring does not document feature_names_in_.
    # Side effects: overrides the __doc__ of a LogisticRegression instance.
    err_msg = "Estimator does not have a feature_names_in_"
    with raises(ValueError, match=err_msg):
        check_dataframe_column_names_consistency("estimator_name", BaseBadClassifier())
    check_dataframe_column_names_consistency("estimator_name", PartialFitChecksName())

    lr = LogisticRegression()
    check_dataframe_column_names_consistency(lr.__class__.__name__, lr)
    lr.__doc__ = "Docstring that does not document the estimator's attributes"
    err_msg = (
        "Estimator LogisticRegression does not document its feature_names_in_ attribute"
    )
    with raises(ValueError, match=err_msg):
        check_dataframe_column_names_consistency(lr.__class__.__name__, lr)


class _BaseMultiLabelClassifierMock(ClassifierMixin, BaseEstimator):
    # [CLASS DOC]
    # HELPER: Base mock multilabel classifier returning a fixed response_output from its
    # response methods (defined in subclasses); tagged multi_label.
    def __init__(self, response_output):
        # [FUNCTION DOC]
        # HELPER: Stores response_output (the value returned by response methods).
        self.response_output = response_output

    def fit(self, X, y):
        # [FUNCTION DOC]
        # HELPER: No-op fit. Outputs: self.
        return self

    def __sklearn_tags__(self):
        # [FUNCTION DOC]
        # HELPER: Returns parent tags with classifier_tags.multi_label True.
        tags = super().__sklearn_tags__()
        tags.classifier_tags.multi_label = True
        return tags


def test_check_classifiers_multilabel_output_format_predict():
    # [FUNCTION DOC]
    # Verifies check_classifiers_multilabel_output_format_predict raises AssertionError
    # when predict returns a list, an array of wrong shape, or an array of wrong dtype.
    n_samples, test_size, n_outputs = 100, 25, 5
    _, y = make_multilabel_classification(
        n_samples=n_samples,
        n_features=2,
        n_classes=n_outputs,
        n_labels=3,
        length=50,
        allow_unlabeled=True,
        random_state=0,
    )
    y_test = y[-test_size:]

    class MultiLabelClassifierPredict(_BaseMultiLabelClassifierMock):
        # [CLASS DOC]
        # HELPER: Mock whose predict returns response_output.
        def predict(self, X):
            # [FUNCTION DOC]
            # HELPER: Returns response_output regardless of X.
            return self.response_output

    # 1. inconsistent array type
    clf = MultiLabelClassifierPredict(response_output=y_test.tolist())
    err_msg = (
        r"MultiLabelClassifierPredict.predict is expected to output a "
        r"NumPy array. Got <class 'list'> instead."
    )
    with raises(AssertionError, match=err_msg):
        check_classifiers_multilabel_output_format_predict(clf.__class__.__name__, clf)
    # 2. inconsistent shape
    clf = MultiLabelClassifierPredict(response_output=y_test[:, :-1])
    err_msg = (
        r"MultiLabelClassifierPredict.predict outputs a NumPy array of "
        r"shape \(25, 4\) instead of \(25, 5\)."
    )
    with raises(AssertionError, match=err_msg):
        check_classifiers_multilabel_output_format_predict(clf.__class__.__name__, clf)
    # 3. inconsistent dtype
    clf = MultiLabelClassifierPredict(response_output=y_test.astype(np.float64))
    err_msg = (
        r"MultiLabelClassifierPredict.predict does not output the same "
        r"dtype than the targets."
    )
    with raises(AssertionError, match=err_msg):
        check_classifiers_multilabel_output_format_predict(clf.__class__.__name__, clf)


def test_check_classifiers_multilabel_output_format_predict_proba():
    # [FUNCTION DOC]
    # Verifies check_classifiers_multilabel_output_format_predict_proba raises for sparse
    # output (ValueError), lists of wrong length, shape, dtype or non-probabilities, and
    # for arrays of wrong shape, dtype or values out of [0, 1].
    n_samples, test_size, n_outputs = 100, 25, 5
    _, y = make_multilabel_classification(
        n_samples=n_samples,
        n_features=2,
        n_classes=n_outputs,
        n_labels=3,
        length=50,
        allow_unlabeled=True,
        random_state=0,
    )
    y_test = y[-test_size:]

    class MultiLabelClassifierPredictProba(_BaseMultiLabelClassifierMock):
        # [CLASS DOC]
        # HELPER: Mock whose predict_proba returns response_output.
        def predict_proba(self, X):
            # [FUNCTION DOC]
            # HELPER: Returns response_output regardless of X.
            return self.response_output

    for csr_container in CSR_CONTAINERS:
        # 1. unknown output type
        clf = MultiLabelClassifierPredictProba(response_output=csr_container(y_test))
        err_msg = (
            f"Unknown returned type .*{csr_container.__name__}.* by "
            r"MultiLabelClassifierPredictProba.predict_proba. A list or a Numpy "
            r"array is expected."
        )
        with raises(ValueError, match=err_msg):
            check_classifiers_multilabel_output_format_predict_proba(
                clf.__class__.__name__,
                clf,
            )
    # 2. for list output
    # 2.1. inconsistent length
    clf = MultiLabelClassifierPredictProba(response_output=y_test.tolist())
    err_msg = (
        "When MultiLabelClassifierPredictProba.predict_proba returns a list, "
        "the list should be of length n_outputs and contain NumPy arrays. Got "
        f"length of {test_size} instead of {n_outputs}."
    )
    with raises(AssertionError, match=err_msg):
        check_classifiers_multilabel_output_format_predict_proba(
            clf.__class__.__name__,
            clf,
        )
    # 2.2. array of inconsistent shape
    response_output = [np.ones_like(y_test) for _ in range(n_outputs)]
    clf = MultiLabelClassifierPredictProba(response_output=response_output)
    err_msg = (
        r"When MultiLabelClassifierPredictProba.predict_proba returns a list, "
        r"this list should contain NumPy arrays of shape \(n_samples, 2\). Got "
        r"NumPy arrays of shape \(25, 5\) instead of \(25, 2\)."
    )
    with raises(AssertionError, match=err_msg):
        check_classifiers_multilabel_output_format_predict_proba(
            clf.__class__.__name__,
            clf,
        )
    # 2.3. array of inconsistent dtype
    response_output = [
        np.ones(shape=(y_test.shape[0], 2), dtype=np.int64) for _ in range(n_outputs)
    ]
    clf = MultiLabelClassifierPredictProba(response_output=response_output)
    err_msg = (
        "When MultiLabelClassifierPredictProba.predict_proba returns a list, "
        "it should contain NumPy arrays with floating dtype."
    )
    with raises(AssertionError, match=err_msg):
        check_classifiers_multilabel_output_format_predict_proba(
            clf.__class__.__name__,
            clf,
        )
    # 2.4. array does not contain probability (each row should sum to 1)
    response_output = [
        np.ones(shape=(y_test.shape[0], 2), dtype=np.float64) for _ in range(n_outputs)
    ]
    clf = MultiLabelClassifierPredictProba(response_output=response_output)
    err_msg = (
        r"When MultiLabelClassifierPredictProba.predict_proba returns a list, "
        r"each NumPy array should contain probabilities for each class and "
        r"thus each row should sum to 1"
    )
    with raises(AssertionError, match=err_msg):
        check_classifiers_multilabel_output_format_predict_proba(
            clf.__class__.__name__,
            clf,
        )
    # 3 for array output
    # 3.1. array of inconsistent shape
    clf = MultiLabelClassifierPredictProba(response_output=y_test[:, :-1])
    err_msg = (
        r"When MultiLabelClassifierPredictProba.predict_proba returns a NumPy "
        r"array, the expected shape is \(n_samples, n_outputs\). Got \(25, 4\)"
        r" instead of \(25, 5\)."
    )
    with raises(AssertionError, match=err_msg):
        check_classifiers_multilabel_output_format_predict_proba(
            clf.__class__.__name__,
            clf,
        )
    # 3.2. array of inconsistent dtype
    response_output = np.zeros_like(y_test, dtype=np.int64)
    clf = MultiLabelClassifierPredictProba(response_output=response_output)
    err_msg = (
        r"When MultiLabelClassifierPredictProba.predict_proba returns a NumPy "
        r"array, the expected data type is floating."
    )
    with raises(AssertionError, match=err_msg):
        check_classifiers_multilabel_output_format_predict_proba(
            clf.__class__.__name__,
            clf,
        )
    # 4. array does not contain probabilities
    clf = MultiLabelClassifierPredictProba(response_output=y_test * 2.0)
    err_msg = (
        r"When MultiLabelClassifierPredictProba.predict_proba returns a NumPy "
        r"array, this array is expected to provide probabilities of the "
        r"positive class and should therefore contain values between 0 and 1."
    )
    with raises(AssertionError, match=err_msg):
        check_classifiers_multilabel_output_format_predict_proba(
            clf.__class__.__name__,
            clf,
        )


def test_check_classifiers_multilabel_output_format_decision_function():
    # [FUNCTION DOC]
    # Verifies check_classifiers_multilabel_output_format_decision_function raises when
    # decision_function returns a list, an array of wrong shape, or a non-float array.
    n_samples, test_size, n_outputs = 100, 25, 5
    _, y = make_multilabel_classification(
        n_samples=n_samples,
        n_features=2,
        n_classes=n_outputs,
        n_labels=3,
        length=50,
        allow_unlabeled=True,
        random_state=0,
    )
    y_test = y[-test_size:]

    class MultiLabelClassifierDecisionFunction(_BaseMultiLabelClassifierMock):
        # [CLASS DOC]
        # HELPER: Mock whose decision_function returns response_output.
        def decision_function(self, X):
            # [FUNCTION DOC]
            # HELPER: Returns response_output regardless of X.
            return self.response_output

    # 1. inconsistent array type
    clf = MultiLabelClassifierDecisionFunction(response_output=y_test.tolist())
    err_msg = (
        r"MultiLabelClassifierDecisionFunction.decision_function is expected "
        r"to output a NumPy array. Got <class 'list'> instead."
    )
    with raises(AssertionError, match=err_msg):
        check_classifiers_multilabel_output_format_decision_function(
            clf.__class__.__name__,
            clf,
        )
    # 2. inconsistent shape
    clf = MultiLabelClassifierDecisionFunction(response_output=y_test[:, :-1])
    err_msg = (
        r"MultiLabelClassifierDecisionFunction.decision_function is expected "
        r"to provide a NumPy array of shape \(n_samples, n_outputs\). Got "
        r"\(25, 4\) instead of \(25, 5\)"
    )
    with raises(AssertionError, match=err_msg):
        check_classifiers_multilabel_output_format_decision_function(
            clf.__class__.__name__,
            clf,
        )
    # 3. inconsistent dtype
    clf = MultiLabelClassifierDecisionFunction(response_output=y_test)
    err_msg = (
        r"MultiLabelClassifierDecisionFunction.decision_function is expected "
        r"to output a floating dtype."
    )
    with raises(AssertionError, match=err_msg):
        check_classifiers_multilabel_output_format_decision_function(
            clf.__class__.__name__,
            clf,
        )


def run_tests_without_pytest():
    # [FUNCTION DOC]
    # HELPER: Runs every test_* function found on the __main__ module through unittest
    # (FunctionTestCase in a TestSuite with a TextTestRunner) so the file can be
    # executed without pytest. Side effects: runs tests and prints results to stderr.
    # KNOWN ISSUE: it is invoked from the __main__ guard placed in the middle of the
    # file, so test functions defined after that guard do not exist yet and are not run.
    """Runs the tests in this file without using pytest."""
    main_module = sys.modules["__main__"]
    test_functions = [
        getattr(main_module, name)
        for name in dir(main_module)
        if name.startswith("test_")
    ]
    test_cases = [unittest.FunctionTestCase(fn) for fn in test_functions]
    suite = unittest.TestSuite()
    suite.addTests(test_cases)
    runner = unittest.TextTestRunner()
    runner.run(suite)


def test_check_class_weight_balanced_linear_classifier():
    # [FUNCTION DOC]
    # Verifies check_class_weight_balanced_linear_classifier flags a classifier whose
    # balanced class weights are wrong.
    # check that ill-computed balanced weights raises an exception
    msg = "Classifier estimator_name is not computing class_weight=balanced properly"
    with raises(AssertionError, match=msg):
        check_class_weight_balanced_linear_classifier(
            "estimator_name", BadBalancedWeightsClassifier()
        )


@_mark_thread_unsafe_if_pytest_imported
def test_all_estimators_all_public():
    # [FUNCTION DOC]
    # Verifies all_estimators emits no warnings and returns no private estimators.
    # KNOWN ISSUE: all_estimators returns (name, class) tuples, so est.__class__.__name__
    # is always "tuple" and the privacy assertion is vacuous.
    # all_estimator should not fail when pytest is not installed and return
    # only public estimators
    with warnings.catch_warnings(record=True) as record:
        estimators = all_estimators()
    # no warnings are raised
    assert not record
    for est in estimators:
        assert not est.__class__.__name__.startswith("_")


if __name__ == "__main__":
    # This module is run as a script to check that we have no dependency on
    # pytest for estimator checks.
    run_tests_without_pytest()


def test_estimator_checks_generator_strict_none():
    # [FUNCTION DOC]
    # Verifies estimator_checks_generator with mark="xfail" and no xfail_strict produces
    # xfail marks without a "strict" kwarg, for NuSVC which has expected failures.
    # Check that no "strict" mark is included in the generated checks
    est = next(_construct_instances(NuSVC))
    expected_to_fail = _get_expected_failed_checks(est)
    # If we don't pass strict, it should not appear in the xfail mark either
    # This way the behaviour configured in pytest.ini takes precedence.
    checks = estimator_checks_generator(
        est,
        legacy=True,
        expected_failed_checks=expected_to_fail,
        mark="xfail",
    )
    # make sure we use a class that has expected failures
    assert len(expected_to_fail) > 0
    marked_checks = [c for c in checks if hasattr(c, "marks")]
    # make sure we have some checks with marks
    assert len(marked_checks) > 0

    for parameter_set in marked_checks:
        first_mark = parameter_set.marks[0]
        assert "strict" not in first_mark.kwargs


def test_estimator_checks_generator_strict_xfail_tests():
    # [FUNCTION DOC]
    # Verifies with xfail_strict=True that exactly the expected-to-fail checks of NuSVC
    # are marked as strict xfail (collected by check name from ParameterSet values).
    # Make sure that the checks generator marks tests that are expected to fail
    # as strict xfail
    est = next(_construct_instances(NuSVC))
    expected_to_fail = _get_expected_failed_checks(est)
    checks = estimator_checks_generator(
        est,
        legacy=True,
        expected_failed_checks=expected_to_fail,
        mark="xfail",
        xfail_strict=True,
    )
    # make sure we use a class that has expected failures
    assert len(expected_to_fail) > 0
    strict_xfailed_checks = []

    # xfail'ed checks are wrapped in a ParameterSet, so below we extract
    # the things we need via a bit of a crutch: len()
    marked_checks = [c for c in checks if hasattr(c, "marks")]
    # make sure we use a class that has expected failures
    assert len(expected_to_fail) > 0

    for parameter_set in marked_checks:
        _, check = parameter_set.values
        first_mark = parameter_set.marks[0]
        if first_mark.kwargs["strict"]:
            strict_xfailed_checks.append(_check_name(check))

    # all checks expected to fail are marked as strict xfail
    assert set(expected_to_fail.keys()) == set(strict_xfailed_checks)


@_mark_thread_unsafe_if_pytest_imported  # Some checks use warnings.
def test_estimator_checks_generator_skipping_tests():
    # [FUNCTION DOC]
    # Verifies with mark="skip" that every expected-to-fail NuSVC check raises SkipTest
    # when run (others may also skip). Runs the checks; thread-unsafe.
    # Make sure the checks generator skips tests that are expected to fail
    est = next(_construct_instances(NuSVC))
    expected_to_fail = _get_expected_failed_checks(est)
    checks = estimator_checks_generator(
        est, legacy=True, expected_failed_checks=expected_to_fail, mark="skip"
    )
    # making sure we use a class that has expected failures
    assert len(expected_to_fail) > 0
    skipped_checks = []
    for estimator, check in checks:
        try:
            check(estimator)
        except SkipTest:
            skipped_checks.append(_check_name(check))
    # all checks expected to fail are skipped
    # some others might also be skipped, if their dependencies are not installed.
    assert set(expected_to_fail.keys()) <= set(skipped_checks)


@_mark_thread_unsafe_if_pytest_imported
def test_xfail_count_with_no_fast_fail():
    # [FUNCTION DOC]
    # Verifies check_estimator with on_fail="warn" emits one EstimatorCheckFailedWarning
    # per expected failure (ignoring SkipTestWarning) and logs that many "xfail" results.
    """Test that the right number of xfail warnings are raised when on_fail is "warn".

    It also checks the number of raised EstimatorCheckFailedWarning, and checks the
    output of check_estimator.
    """
    est = NuSVC()
    expected_failed_checks = _get_expected_failed_checks(est)
    # This is to make sure we test a class that has some expected failures
    assert len(expected_failed_checks) > 0
    with warnings.catch_warnings(record=True) as records:
        logs = check_estimator(
            est,
            expected_failed_checks=expected_failed_checks,
            on_fail="warn",
        )
    xfail_warns = [w for w in records if w.category != SkipTestWarning]
    assert all([rec.category == EstimatorCheckFailedWarning for rec in xfail_warns])
    assert len(xfail_warns) == len(expected_failed_checks)

    xfailed = [log for log in logs if log["status"] == "xfail"]
    assert len(xfailed) == len(expected_failed_checks)


def test_check_estimator_callback():
    # [FUNCTION DOC]
    # Verifies the callback receives one call per check: xfail count equals expected
    # failures, some pass, none fail, and skipped accounts for the remainder of all checks.
    """Test that the callback is called with the right arguments."""
    call_count = {"xfail": 0, "skipped": 0, "passed": 0, "failed": 0}

    def callback(
        *,
        estimator,
        check_name,
        exception,
        status,
        expected_to_fail,
        expected_to_fail_reason,
    ):
        # [FUNCTION DOC]
        # HELPER: Keyword-only callback that asserts status is valid and increments the
        # enclosing call_count entry for that status (via nonlocal).
        assert status in ("xfail", "skipped", "passed", "failed")
        nonlocal call_count
        call_count[status] += 1

    est = NuSVC()
    expected_failed_checks = _get_expected_failed_checks(est)
    # This is to make sure we test a class that has some expected failures
    assert len(expected_failed_checks) > 0
    with warnings.catch_warnings(record=True):
        check_estimator(
            est,
            expected_failed_checks=expected_failed_checks,
            on_fail=None,
            callback=callback,
        )
    all_checks_count = len(list(estimator_checks_generator(est, legacy=True)))
    assert call_count["xfail"] == len(expected_failed_checks)
    assert call_count["passed"] > 0
    assert call_count["failed"] == 0
    assert call_count["skipped"] == (
        all_checks_count - call_count["xfail"] - call_count["passed"]
    )


# FIXME: this test should be uncommented when the checks will be granular
# enough. In 0.24, these tests fail due to low estimator performance.
def test_minimal_class_implementation_checks():
    # [FUNCTION DOC]
    # Intended to check that minimal estimators not inheriting BaseEstimator pass
    # check_estimator; currently always raises SkipTest.
    # KNOWN ISSUE: the code after the unconditional raise SkipTest is dead.
    # Check that third-party library can run tests without inheriting from
    # BaseEstimator.
    # FIXME
    raise SkipTest
    minimal_estimators = [MinimalTransformer(), MinimalRegressor(), MinimalClassifier()]
    for estimator in minimal_estimators:
        check_estimator(estimator)


def test_check_fit_check_is_fitted():
    # [FUNCTION DOC]
    # Verifies check_fit_check_is_fitted fails for an estimator that claims to be fitted
    # before fit, and passes for one using __sklearn_is_fitted__ or a fitted attribute.
    class Estimator(BaseEstimator):
        # [CLASS DOC]
        # HELPER: Estimator whose fitted-state signalling is selected by behavior:
        # "attribute", "method" or "always-true".
        def __init__(self, behavior="attribute"):
            # [FUNCTION DOC]
            # HELPER: Stores behavior (default "attribute").
            self.behavior = behavior

        def fit(self, X, y, **kwargs):
            # [FUNCTION DOC]
            # HELPER: Sets is_fitted_ ("attribute") or _is_fitted ("method"). Outputs: self.
            if self.behavior == "attribute":
                self.is_fitted_ = True
            elif self.behavior == "method":
                self._is_fitted = True
            return self

        @available_if(lambda self: self.behavior in {"method", "always-true"})
        def __sklearn_is_fitted__(self):
            # [FUNCTION DOC]
            # HELPER: Available only for "method" and "always-true"; returns True for the latter,
            # otherwise whether _is_fitted was set.
            if self.behavior == "always-true":
                return True
            return hasattr(self, "_is_fitted")

    with raises(Exception, match="passes check_is_fitted before being fit"):
        check_fit_check_is_fitted("estimator", Estimator(behavior="always-true"))

    check_fit_check_is_fitted("estimator", Estimator(behavior="method"))
    check_fit_check_is_fitted("estimator", Estimator(behavior="attribute"))


def test_check_requires_y_none():
    # [FUNCTION DOC]
    # Verifies check_requires_y_none emits no warnings for an estimator raising the
    # standard y-is-None error, and raises ValueError for one re-raising a different
    # message.
    class Estimator(BaseEstimator):
        # [CLASS DOC]
        # HELPER: Estimator whose fit only validates X and y.
        def fit(self, X, y):
            # [FUNCTION DOC]
            # HELPER: Validates X, y with check_X_y (raises when y is None).
            # KNOWN ISSUE: does not return self, unlike the estimator API convention.
            X, y = check_X_y(X, y)

    with warnings.catch_warnings(record=True) as record:
        check_requires_y_none("estimator", Estimator())

    # no warnings are raised
    assert not [r.message for r in record]

    # Make an estimator that throws the wrong error to make sure we catch it
    class EstimatorWithWrongError(BaseEstimator):
        # [CLASS DOC]
        # HELPER: Estimator that replaces the y-is-None ValueError with a wrong message.
        def fit(self, X, y):
            # [FUNCTION DOC]
            # HELPER: Calls check_X_y; on ValueError asserts it is the y-is-None message and
            # raises ValueError with an unrelated message. Returns None on success.
            try:
                X, y = check_X_y(X, y)
            except ValueError as ve:
                # This assertion is just to make sure we are catching the value error
                # that comes from wrong y (=None) and not some other value error
                assert str(ve) == (
                    "estimator requires y to be passed, but the target y is None"
                )
                # Override the error message force fail
                raise ValueError("This is the wrong message that raises error")

    err_msg = (
        "Your estimator raised a ValueError, but with the incorrect or "
        "incomplete error message to be considered a graceful fail."
    )
    with raises(ValueError, match=err_msg):
        check_requires_y_none("estimator", EstimatorWithWrongError())


def test_non_deterministic_estimator_skip_tests():
    # [FUNCTION DOC]
    # Verifies _yield_all_checks includes sample-order and subset invariance checks for
    # minimal estimators, and excludes them for subclasses tagged non_deterministic.
    # check estimators with non_deterministic tag set to True
    # will skip certain tests, refer to issue #22313 for details
    for Estimator in [MinimalTransformer, MinimalRegressor, MinimalClassifier]:
        all_tests = list(_yield_all_checks(Estimator(), legacy=True))
        assert check_methods_sample_order_invariance in all_tests
        assert check_methods_subset_invariance in all_tests

        class MyEstimator(Estimator):
            # [CLASS DOC]
            # HELPER: Subclass of a minimal estimator tagged non_deterministic.
            def __sklearn_tags__(self):
                # [FUNCTION DOC]
                # HELPER: Returns parent tags with non_deterministic True.
                tags = super().__sklearn_tags__()
                tags.non_deterministic = True
                return tags

        all_tests = list(_yield_all_checks(MyEstimator(), legacy=True))
        assert check_methods_sample_order_invariance not in all_tests
        assert check_methods_subset_invariance not in all_tests


def test_check_outlier_contamination():
    # [FUNCTION DOC]
    # Verifies check_outlier_contamination: returns None without parameter constraints,
    # requires a Real Interval constraint, accepts Interval(Real, 0, 0.5, right-closed),
    # and rejects integral, negative-lower, >1 upper or left-closed intervals.
    # Side effects: mutates the local class's _parameter_constraints dict across cases.
    """Check the test for the contamination parameter in the outlier detectors."""

    # Without any parameter constraints, the estimator will early exit the test by
    # returning None.
    class OutlierDetectorWithoutConstraint(OutlierMixin, BaseEstimator):
        # [CLASS DOC]
        # HELPER: Outlier detector with a contamination parameter and no constraints.
        """Outlier detector without parameter validation."""

        def __init__(self, contamination=0.1):
            # [FUNCTION DOC]
            # HELPER: Stores contamination (default 0.1).
            self.contamination = contamination

        def fit(self, X, y=None, sample_weight=None):
            # [FUNCTION DOC]
            # HELPER: No-op fit. Outputs: self.
            return self  # pragma: no cover

        def predict(self, X, y=None):
            # [FUNCTION DOC]
            # HELPER: Returns ones (all inliers) for each sample.
            return np.ones(X.shape[0])

    detector = OutlierDetectorWithoutConstraint()
    assert check_outlier_contamination(detector.__class__.__name__, detector) is None

    # Now, we check that with the parameter constraints, the test should only be valid
    # if an Interval constraint with bound in [0, 1] is provided.
    class OutlierDetectorWithConstraint(OutlierDetectorWithoutConstraint):
        # [CLASS DOC]
        # HELPER: Same detector with a class-level _parameter_constraints for contamination,
        # initially only StrOptions({"auto"}).
        _parameter_constraints = {"contamination": [StrOptions({"auto"})]}

    detector = OutlierDetectorWithConstraint()
    err_msg = "contamination constraints should contain a Real Interval constraint."
    with raises(AssertionError, match=err_msg):
        check_outlier_contamination(detector.__class__.__name__, detector)

    # Add a correct interval constraint and check that the test passes.
    OutlierDetectorWithConstraint._parameter_constraints["contamination"] = [
        Interval(Real, 0, 0.5, closed="right")
    ]
    detector = OutlierDetectorWithConstraint()
    check_outlier_contamination(detector.__class__.__name__, detector)

    incorrect_intervals = [
        Interval(Integral, 0, 1, closed="right"),  # not an integral interval
        Interval(Real, -1, 1, closed="right"),  # lower bound is negative
        Interval(Real, 0, 2, closed="right"),  # upper bound is greater than 1
        Interval(Real, 0, 0.5, closed="left"),  # lower bound include 0
    ]

    err_msg = r"contamination constraint should be an interval in \(0, 0.5\]"
    for interval in incorrect_intervals:
        OutlierDetectorWithConstraint._parameter_constraints["contamination"] = [
            interval
        ]
        detector = OutlierDetectorWithConstraint()
        with raises(AssertionError, match=err_msg):
            check_outlier_contamination(detector.__class__.__name__, detector)


def test_decision_proba_tie_ranking():
    # [FUNCTION DOC]
    # Non-regression (issue 24025): check_decision_proba_consistency passes for
    # SGDClassifier with log loss despite probability ties.
    """Check that in case with some probabilities ties, we relax the
    ranking comparison with the decision function.
    Non-regression test for:
    https://github.com/scikit-learn/scikit-learn/issues/24025
    """
    estimator = SGDClassifier(loss="log_loss")
    check_decision_proba_consistency("SGDClassifier", estimator)


def test_yield_all_checks_legacy():
    # [FUNCTION DOC]
    # Verifies legacy=True yields strictly more checks than legacy=False and that the
    # non-legacy check names are a subset of the legacy ones.
    # Test that _yield_all_checks with legacy=True returns more checks.
    estimator = MinimalClassifier()

    legacy_checks = list(_yield_all_checks(estimator, legacy=True))
    non_legacy_checks = list(_yield_all_checks(estimator, legacy=False))

    assert len(legacy_checks) > len(non_legacy_checks)

    def get_check_name(check):
        # [FUNCTION DOC]
        # HELPER: Returns a check's __name__, falling back to .func.__name__ for partials.
        try:
            return check.__name__
        except AttributeError:
            return check.func.__name__

    # Check that all non-legacy checks are included in legacy checks
    non_legacy_check_names = {get_check_name(check) for check in non_legacy_checks}
    legacy_check_names = {get_check_name(check) for check in legacy_checks}
    assert non_legacy_check_names.issubset(legacy_check_names)


def test_check_estimator_cloneable_error():
    # [FUNCTION DOC]
    # Verifies check_estimator_cloneable raises AssertionError when cloning fails.
    """Check that the right error is raised when the estimator is not cloneable."""

    class NotCloneable(BaseEstimator):
        # [CLASS DOC]
        # HELPER: Estimator whose __sklearn_clone__ raises NotImplementedError.
        def __sklearn_clone__(self):
            # [FUNCTION DOC]
            # HELPER: Always raises NotImplementedError.
            raise NotImplementedError("This estimator is not cloneable.")

    estimator = NotCloneable()
    msg = "Cloning of .* failed with error"
    with raises(AssertionError, match=msg):
        check_estimator_cloneable("NotCloneable", estimator)


def test_estimator_repr_error():
    # [FUNCTION DOC]
    # Verifies check_estimator_repr raises AssertionError when repr fails.
    """Check that the right error is raised when the estimator does not have a repr."""

    class NotRepr(BaseEstimator):
        # [CLASS DOC]
        # HELPER: Estimator whose __repr__ raises NotImplementedError.
        def __repr__(self):
            # [FUNCTION DOC]
            # HELPER: Always raises NotImplementedError.
            raise NotImplementedError("This estimator does not have a repr.")

    estimator = NotRepr()
    msg = "Repr of .* failed with error"
    with raises(AssertionError, match=msg):
        check_estimator_repr("NotRepr", estimator)


def test_check_estimator_tags_renamed():
    # [FUNCTION DOC]
    # Verifies check_estimator_tags_renamed raises TypeError for objects defining only
    # legacy _more_tags or _get_tags, and accepts one also defining __sklearn_tags__.
    class BadEstimator1:
        # [CLASS DOC]
        # HELPER: Defines only the legacy _more_tags.
        def _more_tags(self):
            # [FUNCTION DOC]
            # HELPER: Never called; returns None.
            return None  # pragma: no cover

    class BadEstimator2:
        # [CLASS DOC]
        # HELPER: Defines only the legacy _get_tags.
        def _get_tags(self):
            # [FUNCTION DOC]
            # HELPER: Never called; returns None.
            return None  # pragma: no cover

    class OkayEstimator:
        # [CLASS DOC]
        # HELPER: Defines both __sklearn_tags__ and _more_tags.
        def __sklearn_tags__(self):
            # [FUNCTION DOC]
            # HELPER: Never called; returns None.
            return None  # pragma: no cover

        def _more_tags(self):
            # [FUNCTION DOC]
            # HELPER: Never called; returns None.
            return None  # pragma: no cover

    msg = "has defined either `_more_tags` or `_get_tags`"
    with raises(TypeError, match=msg):
        check_estimator_tags_renamed("BadEstimator1", BadEstimator1())
    with raises(TypeError, match=msg):
        check_estimator_tags_renamed("BadEstimator2", BadEstimator2())

    # This shouldn't fail since we allow both __sklearn_tags__ and _more_tags
    # to exist so that third party estimators can easily support multiple sklearn
    # versions.
    check_estimator_tags_renamed("OkayEstimator", OkayEstimator())


def test_check_classifier_not_supporting_multiclass():
    # [FUNCTION DOC]
    # Verifies check_classifier_not_supporting_multiclass fails for an estimator whose
    # multi_class tag is False but which does not raise on multiclass targets.
    """Check that when the estimator has the wrong tags.classifier_tags.multi_class
    set, the test fails."""

    class BadEstimator(BaseEstimator):
        # [CLASS DOC]
        # HELPER: Estimator with a no-op fit and default tags.
        # KNOWN ISSUE: the comment mentions multi_output=False, but the relevant tag is
        # classifier_tags.multi_class.
        # we don't actually need to define the tag here since we're running the test
        # manually, and BaseEstimator defaults to multi_output=False.
        def fit(self, X, y):
            # [FUNCTION DOC]
            # HELPER: No-op fit. Outputs: self.
            return self

    msg = "The estimator tag `tags.classifier_tags.multi_class` is False"
    with raises(AssertionError, match=msg):
        check_classifier_not_supporting_multiclass("BadEstimator", BadEstimator())


# Test that set_output doesn't make the tests to fail.
@_mark_thread_unsafe_if_pytest_imported
def test_estimator_with_set_output():
    # [FUNCTION DOC]
    # Verifies check_estimator passes for StandardScaler with set_output pandas and
    # polars, marking the array API checks as expected failures. Raises SkipTest if
    # either library is missing.
    # Doing this since pytest is not available for this file.
    for lib in ["pandas", "polars"]:
        try:
            importlib.__import__(lib)
        except ImportError:
            raise SkipTest(f"Library {lib} is not installed")

        estimator = StandardScaler().set_output(transform=lib)
        check_estimator(
            estimator=estimator,
            expected_failed_checks={
                "check_array_api_input": (
                    "this check is expected to fail because pandas and polars"
                    " are not compatible with the array api."
                ),
                "check_array_api_mixed_inputs": (
                    "this check is expected to fail because pandas and polars"
                    " are not compatible with the array api."
                ),
                "check_array_api_same_namespace": (
                    "this check is expected to fail because pandas and polars"
                    " are not compatible with the array api."
                ),
            },
        )


def test_estimator_checks_generator():
    # [FUNCTION DOC]
    # Verifies estimator_checks_generator returns a generator object.
    """Check that checks_generator returns a generator."""
    all_instance_gen_checks = estimator_checks_generator(LogisticRegression())
    assert isgenerator(all_instance_gen_checks)


def test_check_estimator_callback_with_fast_fail_error():
    # [FUNCTION DOC]
    # Verifies check_estimator raises ValueError when a callback is given with
    # on_fail="raise".
    """Check that check_estimator fails correctly with on_fail='raise' and callback."""
    with raises(
        ValueError, match="callback cannot be provided together with on_fail='raise'"
    ):
        check_estimator(LogisticRegression(), on_fail="raise", callback=lambda: None)


def test_check_mixin_order():
    # [FUNCTION DOC]
    # Verifies check_mixin_order flags a class listing BaseEstimator before
    # TransformerMixin in its bases.
    """Test that the check raises an error when the mixin order is incorrect."""

    class BadEstimator(BaseEstimator, TransformerMixin):
        # [CLASS DOC]
        # HELPER: Transformer with mixins in the wrong order.
        def fit(self, X, y=None):
            # [FUNCTION DOC]
            # HELPER: No-op fit. Outputs: self.
            return self

    msg = "TransformerMixin comes before/left side of BaseEstimator"
    with raises(AssertionError, match=re.escape(msg)):
        check_mixin_order("BadEstimator", BadEstimator())


def test_check_positive_only_tag_during_fit():
    # [FUNCTION DOC]
    # Verifies check_positive_only_tag_during_fit flags an estimator that rejects
    # negative X while its positive_only tag is False.
    class RequiresPositiveXBadTag(RequiresPositiveXRegressor):
        # [CLASS DOC]
        # HELPER: RequiresPositiveXRegressor with input_tags.positive_only set False.
        def __sklearn_tags__(self):
            # [FUNCTION DOC]
            # HELPER: Returns parent tags with input_tags.positive_only False.
            tags = super().__sklearn_tags__()
            tags.input_tags.positive_only = False
            return tags

    with raises(
        AssertionError, match="This happens when passing negative input values as X."
    ):
        check_positive_only_tag_during_fit(
            "RequiresPositiveXBadTag", RequiresPositiveXBadTag()
        )
