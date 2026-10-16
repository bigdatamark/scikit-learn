# [MODULE DOC]
# HELPER: Test utilities: mock containers mimicking a minimal DataFrame and dummy
# estimators (CheckingClassifier, NoSampleWeightWrapper,
# _MockEstimatorOnOffPrediction) used to test meta-estimators, pipelines, CV and
# metadata routing. Import-time side effect: replaces
# CheckingClassifier.set_fit_request with a RequestMethod that accepts any key
# without validation, so arbitrary fit params can be requested.
# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils._metadata_requests import RequestMethod
from sklearn.utils.metaestimators import available_if
from sklearn.utils.validation import (
    _check_sample_weight,
    _num_samples,
    check_array,
    check_is_fitted,
    check_random_state,
)


class ArraySlicingWrapper:
    # [CLASS DOC]
    # HELPER: Stand-in for a pandas iloc indexer: slicing returns a MockDataFrame of
    # the sliced underlying array.
    """
    Parameters
    ----------
    array
    """

    def __init__(self, array):
        # [FUNCTION DOC]
        # HELPER: Stores the wrapped array in self.array.
        # Inputs: array - ndarray.
        self.array = array

    def __getitem__(self, aslice):
        # [FUNCTION DOC]
        # HELPER: Indexes the stored array with aslice and wraps the result in a
        # MockDataFrame.
        # Inputs: aslice - any NumPy-compatible key. Outputs: MockDataFrame.
        return MockDataFrame(self.array[aslice])


class MockDataFrame:
    # [CLASS DOC]
    # HELPER: Minimal DataFrame-like wrapper around an ndarray exposing values, shape,
    # ndim, len, iloc, take and array conversion, used to check that code does not
    # rely on NumPy-specific behaviour of inputs.
    """
    Parameters
    ----------
    array
    """

    # have shape and length but don't support indexing.

    def __init__(self, array):
        # [FUNCTION DOC]
        # HELPER: Stores array as array and values, copies shape and ndim, and builds an
        # ArraySlicingWrapper as iloc.
        # Inputs: array - ndarray.
        self.array = array
        self.values = array
        self.shape = array.shape
        self.ndim = array.ndim
        # ugly hack to make iloc work.
        self.iloc = ArraySlicingWrapper(array)

    def __len__(self):
        # [FUNCTION DOC]
        # HELPER: Returns the length of the underlying array (first dimension).
        return len(self.array)

    def __array__(self, dtype=None):
        # [FUNCTION DOC]
        # HELPER: Returns the underlying array for NumPy conversion.
        # Inputs: dtype - ignored.
        # KNOWN ISSUE: dtype argument is ignored and no copy parameter is accepted (NumPy 2
        # may pass copy=...).
        # Pandas data frames also are array-like: we want to make sure that
        # input validation in cross-validation does not try to call that
        # method.
        return self.array

    def __eq__(self, other):
        # [FUNCTION DOC]
        # HELPER: Elementwise equality with another MockDataFrame, wrapped in a new
        # MockDataFrame.
        # Inputs: other - object exposing array. Exceptions: AttributeError otherwise.
        return MockDataFrame(self.array == other.array)

    def __ne__(self, other):
        # [FUNCTION DOC]
        # HELPER: Negation of the truthiness of self == other.
        # KNOWN ISSUE: self == other returns a MockDataFrame whose truthiness is its
        # length, so this returns False for any non-empty frame regardless of contents.
        return not self == other

    def take(self, indices, axis=0):
        # [FUNCTION DOC]
        # HELPER: Takes entries along axis from the underlying array and wraps the result.
        # Inputs: indices - int array; axis - int, default 0. Outputs: MockDataFrame.
        return MockDataFrame(self.array.take(indices, axis=axis))


class CheckingClassifier(ClassifierMixin, BaseEstimator):
    # [CLASS DOC]
    # HELPER: Dummy classifier that runs user-supplied checks on X/y in its methods,
    # verifies expected fit params / sample_weight, and returns random predictions.
    # Score is 1.0 when foo_param > 1 else 0.0.
    # KNOWN ISSUE: class docstring documents classes_ as int; it is an ndarray of
    # unique labels.
    """Dummy classifier to test pipelining and meta-estimators.

    Checks some property of `X` and `y`in fit / predict.
    This allows testing whether pipelines / cross-validation or metaestimators
    changed the input.

    Can also be used to check if `fit_params` are passed correctly, and
    to force a certain score to be returned.

    Parameters
    ----------
    check_y, check_X : callable, default=None
        The callable used to validate `X` and `y`. These callable should return
        a bool where `False` will trigger an `AssertionError`. If `None`, the
        data is not validated. Default is `None`.

    check_y_params, check_X_params : dict, default=None
        The optional parameters to pass to `check_X` and `check_y`. If `None`,
        then no parameters are passed in.

    methods_to_check : "all" or list of str, default="all"
        The methods in which the checks should be applied. By default,
        all checks will be done on all methods (`fit`, `predict`,
        `predict_proba`, `decision_function` and `score`).

    foo_param : int, default=0
        A `foo` param. When `foo > 1`, the output of :meth:`score` will be 1
        otherwise it is 0.

    expected_sample_weight : bool, default=False
        Whether to check if a valid `sample_weight` was passed to `fit`.

    expected_fit_params : list of str, default=None
        A list of the expected parameters given when calling `fit`.

    Attributes
    ----------
    classes_ : int
        The classes seen during `fit`.

    n_features_in_ : int
        The number of features seen during `fit`.

    Examples
    --------
    >>> from sklearn.utils._mocking import CheckingClassifier

    This helper allow to assert to specificities regarding `X` or `y`. In this
    case we expect `check_X` or `check_y` to return a boolean.

    >>> from sklearn.datasets import load_iris
    >>> X, y = load_iris(return_X_y=True)
    >>> clf = CheckingClassifier(check_X=lambda x: x.shape == (150, 4))
    >>> clf.fit(X, y)
    CheckingClassifier(...)

    We can also provide a check which might raise an error. In this case, we
    expect `check_X` to return `X` and `check_y` to return `y`.

    >>> from sklearn.utils import check_array
    >>> clf = CheckingClassifier(check_X=check_array)
    >>> clf.fit(X, y)
    CheckingClassifier(...)
    """

    def __init__(
        self,
        *,
        check_y=None,
        check_y_params=None,
        check_X=None,
        check_X_params=None,
        methods_to_check="all",
        foo_param=0,
        expected_sample_weight=None,
        expected_fit_params=None,
        random_state=None,
    ):
        # [FUNCTION DOC]
        # HELPER: Stores all constructor arguments unchanged as attributes (sklearn
        # convention). random_state seeds the random predictions.
        self.check_y = check_y
        self.check_y_params = check_y_params
        self.check_X = check_X
        self.check_X_params = check_X_params
        self.methods_to_check = methods_to_check
        self.foo_param = foo_param
        self.expected_sample_weight = expected_sample_weight
        self.expected_fit_params = expected_fit_params
        self.random_state = random_state

    def _check_X_y(self, X, y=None, should_be_fitted=True):
        # [FUNCTION DOC]
        # HELPER: Optionally asserts the estimator is fitted, then applies check_X to X and,
        # when y is given, check_y to y, each with their optional params dict. A boolean
        # check result is asserted true; any other result replaces X (resp. y).
        # Inputs: X, y (optional), should_be_fitted - bool, default True.
        # Outputs: tuple (X, y) possibly transformed by the checks.
        # Exceptions: NotFittedError, AssertionError when a boolean check is False, and
        # anything the check callables raise.
        """Validate X and y and make extra check.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data set.
            `X` is checked only if `check_X` is not `None` (default is None).
        y : array-like of shape (n_samples), default=None
            The corresponding target, by default `None`.
            `y` is checked only if `check_y` is not `None` (default is None).
        should_be_fitted : bool, default=True
            Whether or not the classifier should be already fitted.
            By default True.

        Returns
        -------
        X, y
        """
        if should_be_fitted:
            check_is_fitted(self)
        if self.check_X is not None:
            params = {} if self.check_X_params is None else self.check_X_params
            checked_X = self.check_X(X, **params)
            if isinstance(checked_X, (bool, np.bool_)):
                assert checked_X
            else:
                X = checked_X
        if y is not None and self.check_y is not None:
            params = {} if self.check_y_params is None else self.check_y_params
            checked_y = self.check_y(y, **params)
            if isinstance(checked_y, (bool, np.bool_)):
                assert checked_y
            else:
                y = checked_y
        return X, y

    def fit(self, X, y, sample_weight=None, **fit_params):
        # [FUNCTION DOC]
        # HELPER: Asserts X and y have the same number of samples, runs checks if "fit" is
        # in methods_to_check (or "all"), sets n_features_in_ to the second dimension of X
        # and classes_ to unique labels of y. If expected_fit_params is set, every expected
        # name must be in fit_params and every fit param must have as many samples as X.
        # If expected_sample_weight is truthy, sample_weight must be given and valid.
        # Inputs: X - 2d array-like; y - labels; sample_weight - optional; **fit_params.
        # Outputs: self.
        # Exceptions: AssertionError on any failed expectation, ValueError from sample
        # weight validation.
        # Side effects: sets n_features_in_ and classes_.
        """Fit classifier.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vector, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        y : array-like of shape (n_samples, n_outputs) or (n_samples,), \
                default=None
            Target relative to X for classification or regression;
            None for unsupervised learning.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted.

        **fit_params : dict of string -> object
            Parameters passed to the ``fit`` method of the estimator

        Returns
        -------
        self
        """
        assert _num_samples(X) == _num_samples(y)
        if self.methods_to_check == "all" or "fit" in self.methods_to_check:
            X, y = self._check_X_y(X, y, should_be_fitted=False)
        self.n_features_in_ = np.shape(X)[1]
        self.classes_ = np.unique(check_array(y, ensure_2d=False, allow_nd=True))
        if self.expected_fit_params:
            missing = set(self.expected_fit_params) - set(fit_params)
            if missing:
                raise AssertionError(
                    f"Expected fit parameter(s) {list(missing)} not seen."
                )
            for key, value in fit_params.items():
                if _num_samples(value) != _num_samples(X):
                    raise AssertionError(
                        f"Fit parameter {key} has length {_num_samples(value)}"
                        f"; expected {_num_samples(X)}."
                    )
        if self.expected_sample_weight:
            if sample_weight is None:
                raise AssertionError("Expected sample_weight to be passed")
            _check_sample_weight(sample_weight, X)

        return self

    def predict(self, X):
        # [FUNCTION DOC]
        # HELPER: Runs checks if enabled for "predict", then returns labels drawn
        # uniformly at random from classes_ using a fresh RNG from random_state.
        # Inputs: X. Outputs: ndarray of shape (n_samples,).
        # KNOWN ISSUE: docstring says it predicts the first class; it predicts random
        # classes.
        """Predict the first class seen in `classes_`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data.

        Returns
        -------
        preds : ndarray of shape (n_samples,)
            Predictions of the first class seen in `classes_`.
        """
        if self.methods_to_check == "all" or "predict" in self.methods_to_check:
            X, y = self._check_X_y(X)
        rng = check_random_state(self.random_state)
        return rng.choice(self.classes_, size=_num_samples(X))

    def predict_proba(self, X):
        # [FUNCTION DOC]
        # HELPER: Runs checks if enabled, then returns random probabilities: absolute
        # values of standard normal draws of shape (n_samples, n_classes), normalised so
        # each row sums to 1.
        # Outputs: ndarray of shape (n_samples, n_classes).
        # KNOWN ISSUE: docstring says probability 1 for the first class; values are random.
        """Predict probabilities for each class.

        Here, the dummy classifier will provide a probability of 1 for the
        first class of `classes_` and 0 otherwise.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data.

        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
            The probabilities for each sample and class.
        """
        if self.methods_to_check == "all" or "predict_proba" in self.methods_to_check:
            X, y = self._check_X_y(X)
        rng = check_random_state(self.random_state)
        proba = rng.randn(_num_samples(X), len(self.classes_))
        proba = np.abs(proba, out=proba)
        proba /= np.sum(proba, axis=1)[:, np.newaxis]
        return proba

    def decision_function(self, X):
        # [FUNCTION DOC]
        # HELPER: Runs checks if enabled, then returns standard normal random scores of
        # shape (n_samples,) for binary problems or (n_samples, n_classes) otherwise.
        # KNOWN ISSUE: the inline comment says the binary score "should be null" but random
        # values are returned.
        """Confidence score.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data.

        Returns
        -------
        decision : ndarray of shape (n_samples,) if n_classes == 2\
                else (n_samples, n_classes)
            Confidence score.
        """
        if (
            self.methods_to_check == "all"
            or "decision_function" in self.methods_to_check
        ):
            X, y = self._check_X_y(X)
        rng = check_random_state(self.random_state)
        if len(self.classes_) == 2:
            # for binary classifier, the confidence score is related to
            # classes_[1] and therefore should be null.
            return rng.randn(_num_samples(X))
        else:
            return rng.randn(_num_samples(X), len(self.classes_))

    def score(self, X=None, Y=None):
        # [FUNCTION DOC]
        # HELPER: Runs checks on X and Y if enabled (Y as target) and returns 1.0 if
        # foo_param > 1 else 0.0.
        # Inputs: X, Y - optional. Outputs: float.
        """Fake score.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        Y : array-like of shape (n_samples, n_output) or (n_samples,)
            Target relative to X for classification or regression;
            None for unsupervised learning.

        Returns
        -------
        score : float
            Either 0 or 1 depending of `foo_param` (i.e. `foo_param > 1 =>
            score=1` otherwise `score=0`).
        """
        if self.methods_to_check == "all" or "score" in self.methods_to_check:
            self._check_X_y(X, Y)
        if self.foo_param > 1:
            score = 1.0
        else:
            score = 0.0
        return score

    def __sklearn_tags__(self):
        # [FUNCTION DOC]
        # HELPER: Extends parent tags: marks the estimator to be skipped by common tests,
        # declares it does not require 2D array input, and requires 1d labels.
        # Outputs: Tags object.
        tags = super().__sklearn_tags__()
        tags._skip_test = True
        tags.input_tags.two_d_array = False
        tags.target_tags.one_d_labels = True
        return tags


# Deactivate key validation for CheckingClassifier because we want to be able to
# call fit with arbitrary fit_params and record them. Without this change, we
# would get an error because those arbitrary params are not expected.
CheckingClassifier.set_fit_request = RequestMethod(  # pyrefly: ignore[bad-assignment]
    name="fit", keys=[], validate_keys=False
)


class NoSampleWeightWrapper(BaseEstimator):
    # [CLASS DOC]
    # HELPER: Wrapper estimator whose fit signature hides sample_weight, delegating
    # fit/predict/predict_proba to the wrapped estimator.
    """Wrap estimator which will not expose `sample_weight`.

    Parameters
    ----------
    est : estimator, default=None
        The estimator to wrap.
    """

    def __init__(self, est=None):
        # [FUNCTION DOC]
        # HELPER: Stores the wrapped estimator in est.
        self.est = est

    def fit(self, X, y):
        # [FUNCTION DOC]
        # HELPER: Fits the wrapped estimator in place and returns what its fit returns
        # (the inner estimator, not this wrapper).
        # Side effects: fits self.est.
        return self.est.fit(X, y)

    def predict(self, X):
        # [FUNCTION DOC]
        # HELPER: Delegates predict to the wrapped estimator.
        return self.est.predict(X)

    def predict_proba(self, X):
        # [FUNCTION DOC]
        # HELPER: Delegates predict_proba to the wrapped estimator.
        return self.est.predict_proba(X)

    def __sklearn_tags__(self):
        # [FUNCTION DOC]
        # HELPER: Parent tags with the skip-common-tests flag set.
        tags = super().__sklearn_tags__()
        tags._skip_test = True
        return tags


def _check_response(method):
    # [FUNCTION DOC]
    # HELPER: Builds a predicate for available_if that is true when the estimator's
    # response_methods is not None and contains the given method name.
    # Inputs: method - str. Outputs: function taking the estimator, returning bool.
    def check(self):
        # [FUNCTION DOC]
        # HELPER: The predicate closure; checks membership of the captured method name in
        # self.response_methods.
        return self.response_methods is not None and method in self.response_methods

    return check


class _MockEstimatorOnOffPrediction(BaseEstimator):
    # [CLASS DOC]
    # HELPER: Estimator whose predict / predict_proba / decision_function exist only
    # when listed in response_methods; each returns its own method name as a string.
    """Estimator for which we can turn on/off the prediction methods.

    Parameters
    ----------
    response_methods: list of \
            {"predict", "predict_proba", "decision_function"}, default=None
        List containing the response implemented by the estimator. When, the
        response is in the list, it will return the name of the response method
        when called. Otherwise, an `AttributeError` is raised. It allows to
        use `getattr` as any conventional estimator. By default, no response
        methods are mocked.
    """

    def __init__(self, response_methods=None):
        # [FUNCTION DOC]
        # HELPER: Stores response_methods (list of method names or None).
        self.response_methods = response_methods

    def fit(self, X, y):
        # [FUNCTION DOC]
        # HELPER: Sets classes_ to the unique values of y and returns self.
        self.classes_ = np.unique(y)
        return self

    @available_if(_check_response("predict"))
    def predict(self, X):
        # [FUNCTION DOC]
        # HELPER: Returns the string "predict"; AttributeError if not enabled.
        return "predict"

    @available_if(_check_response("predict_proba"))
    def predict_proba(self, X):
        # [FUNCTION DOC]
        # HELPER: Returns the string "predict_proba"; AttributeError if not enabled.
        return "predict_proba"

    @available_if(_check_response("decision_function"))
    def decision_function(self, X):
        # [FUNCTION DOC]
        # HELPER: Returns the string "decision_function"; AttributeError if not enabled.
        return "decision_function"
