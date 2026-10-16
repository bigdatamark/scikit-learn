# [MODULE DOC]
# AdaBoost ensembles. BaseWeightBoosting implements the shared boosting loop
# (sample-weight normalisation, clipping, early stopping) and delegates a single
# iteration to the subclasses' _boost: AdaBoostClassifier uses discrete SAMME,
# AdaBoostRegressor uses AdaBoost.R2 with weighted bootstrap resampling and a
# weighted-median prediction. Neither estimator supports metadata routing.
# Side effects at import: none.
"""Weight Boosting.

This module contains weight boosting estimators for both classification and
regression.

The module structure is the following:

- The `BaseWeightBoosting` base class implements a common ``fit`` method
  for all the estimators in the module. Regression and classification
  only differ from each other in the loss function that is optimized.

- :class:`~sklearn.ensemble.AdaBoostClassifier` implements adaptive boosting
  (AdaBoost-SAMME) for classification problems.

- :class:`~sklearn.ensemble.AdaBoostRegressor` implements adaptive boosting
  (AdaBoost.R2) for regression problems.
"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import warnings
from abc import ABCMeta, abstractmethod
from numbers import Integral, Real

import numpy as np

from sklearn.base import (
    ClassifierMixin,
    RegressorMixin,
    _fit_context,
    is_classifier,
    is_regressor,
)
from sklearn.ensemble._base import BaseEnsemble
from sklearn.metrics import accuracy_score, r2_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils import _safe_indexing, check_random_state
from sklearn.utils._param_validation import HasMethods, Interval, StrOptions
from sklearn.utils.extmath import softmax
from sklearn.utils.metadata_routing import (
    _raise_for_unsupported_routing,
    _RoutingNotSupportedMixin,
)
from sklearn.utils.validation import (
    _check_sample_weight,
    _num_samples,
    check_is_fitted,
    has_fit_parameter,
    validate_data,
)

__all__ = [
    "AdaBoostClassifier",
    "AdaBoostRegressor",
]


class BaseWeightBoosting(BaseEnsemble, metaclass=ABCMeta):
    # [CLASS DOC]
    # Abstract base for AdaBoost estimators. Adds learning_rate and random_state to
    # BaseEnsemble, validates input for non-fit methods, runs the boosting loop in fit,
    # provides staged_score and weight-averaged feature_importances_, and tags sparse
    # input as supported.
    """Base class for AdaBoost estimators.

    Warning: This class should not be used directly. Use derived classes
    instead.
    """

    _parameter_constraints: dict = {
        "estimator": [HasMethods(["fit", "predict"]), None],
        "n_estimators": [Interval(Integral, 1, None, closed="left")],
        "learning_rate": [Interval(Real, 0, None, closed="neither")],
        "random_state": ["random_state"],
    }

    @abstractmethod
    def __init__(
        self,
        estimator=None,
        *,
        n_estimators=50,
        estimator_params=tuple(),
        learning_rate=1.0,
        random_state=None,
    ):
        # [FUNCTION DOC]
        # Abstract constructor: forwards estimator, n_estimators and estimator_params to
        # BaseEnsemble and stores learning_rate and random_state unchanged.
        super().__init__(
            estimator=estimator,
            n_estimators=n_estimators,
            estimator_params=estimator_params,
        )

        self.learning_rate = learning_rate
        self.random_state = random_state

    def _check_X(self, X):
        # [FUNCTION DOC]
        # HELPER: Validates X for prediction methods without resetting n_features_in_ /
        # feature names: accepts CSR/CSC sparse matrices and nd arrays, keeps the dtype.
        # Outputs: validated X. Exceptions: ValueError on feature count / name mismatch.
        # Only called to validate X in non-fit methods, therefore reset=False
        return validate_data(
            self,
            X,
            accept_sparse=["csr", "csc"],
            ensure_2d=True,
            allow_nd=True,
            dtype=None,
            reset=False,
        )

    @_fit_context(
        # AdaBoost*.estimator is not validated yet
        prefer_skip_nested_validation=False
    )
    def fit(self, X, y, sample_weight=None):
        # [FUNCTION DOC]
        # Runs the boosting loop.
        # Steps: reject routed metadata; validate X, y (y numeric for regressors); validate
        # sample_weight (float64 copy, non-negative, default uniform) and normalise to sum 1;
        # validate the base estimator; reset estimators_, estimator_weights_ (zeros) and
        # estimator_errors_ (ones), both of length n_estimators. For each iteration: clip
        # weights below machine epsilon up to epsilon but keep originally-zero weights at 0;
        # call _boost; stop if it returns None weights; record weight and error; stop if the
        # error is 0, if the weight sum is non-finite (warning) or non-positive; otherwise
        # renormalise the weights (skipped on the last iteration).
        # Inputs: X (n_samples, n_features), y (n_samples,), sample_weight or None.
        # Outputs: self.
        # Exceptions: ValueError for invalid data, negative weights, unsupported estimator,
        # or (classifier) a first estimator worse than random.
        # Side effects: sets estimators_, estimator_weights_, estimator_errors_, estimator_,
        # n_features_in_ / feature_names_in_ and (classifier) classes_, n_classes_; emits a
        # UserWarning on weight overflow; consumes random_state.
        # KNOWN ISSUE: after an early stop, trailing entries of estimator_weights_ stay 0 and
        # of estimator_errors_ stay 1 although no estimator exists for them.
        """Build a boosted classifier/regressor from the training set (X, y).

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. COO, DOK, and LIL are converted to CSR.

        y : array-like of shape (n_samples,)
            The target values.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, the sample weights are initialized to
            1 / n_samples.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        _raise_for_unsupported_routing(self, "fit", sample_weight=sample_weight)
        X, y = validate_data(
            self,
            X,
            y,
            accept_sparse=["csr", "csc"],
            ensure_2d=True,
            allow_nd=True,
            dtype=None,
            y_numeric=is_regressor(self),
        )

        sample_weight = _check_sample_weight(
            sample_weight, X, dtype=np.float64, copy=True, ensure_non_negative=True
        )
        sample_weight /= sample_weight.sum()

        # Check parameters
        self._validate_estimator()

        # Clear any previous fit results
        self.estimators_ = []
        self.estimator_weights_ = np.zeros(self.n_estimators, dtype=np.float64)
        self.estimator_errors_ = np.ones(self.n_estimators, dtype=np.float64)

        # Initialization of the random number instance that will be used to
        # generate a seed at each iteration
        random_state = check_random_state(self.random_state)
        epsilon = np.finfo(sample_weight.dtype).eps

        zero_weight_mask = sample_weight == 0.0
        for iboost in range(self.n_estimators):
            # avoid extremely small sample weight, for details see issue #20320
            sample_weight = np.clip(sample_weight, a_min=epsilon, a_max=None)
            # do not clip sample weights that were exactly zero originally
            sample_weight[zero_weight_mask] = 0.0

            # Boosting step
            sample_weight, estimator_weight, estimator_error = self._boost(
                iboost, X, y, sample_weight, random_state
            )

            # Early termination
            if sample_weight is None:
                break
            self.estimator_weights_[iboost] = estimator_weight
            self.estimator_errors_[iboost] = estimator_error

            # Stop if error is zero
            if estimator_error == 0:
                break

            sample_weight_sum = np.sum(sample_weight)

            if not np.isfinite(sample_weight_sum):
                warnings.warn(
                    (
                        "Sample weights have reached infinite values,"
                        f" at iteration {iboost}, causing overflow. "
                        "Iterations stopped. Try lowering the learning rate."
                    ),
                    stacklevel=2,
                )
                break

            # Stop if the sum of sample weights has become non-positive
            if sample_weight_sum <= 0:
                break

            if iboost < self.n_estimators - 1:
                # Normalize
                sample_weight /= sample_weight_sum

        return self

    @abstractmethod
    def _boost(self, iboost, X, y, sample_weight, random_state):
        # [FUNCTION DOC]
        # Abstract single boosting iteration; must return (new sample weights, estimator
        # weight, estimator error), or (None, None, None) to stop early.
        """Implement a single boost.

        Warning: This method needs to be overridden by subclasses.

        Parameters
        ----------
        iboost : int
            The index of the current boost iteration.

        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. COO, DOK, and LIL are converted to CSR.

        y : array-like of shape (n_samples,)
            The target values (class labels).

        sample_weight : array-like of shape (n_samples,)
            The current sample weights.

        random_state : RandomState
            The current random number generator

        Returns
        -------
        sample_weight : array-like of shape (n_samples,) or None
            The reweighted sample weights.
            If None then boosting has terminated early.

        estimator_weight : float
            The weight for the current boost.
            If None then boosting has terminated early.

        error : float
            The classification error for the current boost.
            If None then boosting has terminated early.
        """
        pass

    def staged_score(self, X, y, sample_weight=None):
        # [FUNCTION DOC]
        # Generator yielding the ensemble score after each boosting stage: accuracy for
        # classifiers, R^2 for regressors, computed on staged_predict outputs.
        # Inputs: X - samples; y - true targets; sample_weight - optional weights.
        # Yields: float per stage. Exceptions: ValueError on invalid X.
        """Return staged scores for X, y.

        This generator method yields the ensemble score after each iteration of
        boosting and therefore allows monitoring, such as to determine the
        score on a test set after each boost.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. COO, DOK, and LIL are converted to CSR.

        y : array-like of shape (n_samples,)
            Labels for X.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.

        Yields
        ------
        z : float
        """
        X = self._check_X(X)

        for y_pred in self.staged_predict(X):
            if is_classifier(self):
                yield accuracy_score(y, y_pred, sample_weight=sample_weight)
            else:
                yield r2_score(y, y_pred, sample_weight=sample_weight)

    @property
    def feature_importances_(self):
        # [FUNCTION DOC]
        # Property: weighted average of the sub-estimators' feature_importances_, weights
        # being estimator_weights_ normalised by their sum.
        # Outputs: ndarray (n_features,).
        # Exceptions: ValueError when estimators_ is None or empty; AttributeError (chained)
        # when the base estimator lacks feature_importances_.
        # KNOWN ISSUE: accessing it before fit raises AttributeError on estimators_ rather
        # than the intended ValueError, because estimators_ does not exist yet.
        """The impurity-based feature importances.

        The higher, the more important the feature.
        The importance of a feature is computed as the (normalized)
        total reduction of the criterion brought by that feature.  It is also
        known as the Gini importance.

        Warning: impurity-based feature importances can be misleading for
        high cardinality features (many unique values). See
        :func:`sklearn.inspection.permutation_importance` as an alternative.

        Returns
        -------
        feature_importances_ : ndarray of shape (n_features,)
            The feature importances.
        """
        if self.estimators_ is None or len(self.estimators_) == 0:
            raise ValueError(
                "Estimator not fitted, call `fit` before `feature_importances_`."
            )

        try:
            norm = self.estimator_weights_.sum()
            return (
                sum(
                    weight * clf.feature_importances_
                    for weight, clf in zip(self.estimator_weights_, self.estimators_)
                )
                / norm
            )

        except AttributeError as e:
            raise AttributeError(
                "Unable to compute feature importances "
                "since estimator does not have a "
                "feature_importances_ attribute"
            ) from e

    def __sklearn_tags__(self):
        # [FUNCTION DOC]
        # Extends parent tags to declare sparse input support. Outputs: Tags.
        tags = super().__sklearn_tags__()
        tags.input_tags.sparse = True
        return tags


class AdaBoostClassifier(
    _RoutingNotSupportedMixin, ClassifierMixin, BaseWeightBoosting
):
    # [CLASS DOC]
    # AdaBoost classifier using the discrete SAMME algorithm. Default base estimator is
    # a depth-1 DecisionTreeClassifier. Decision function is a weighted vote where a
    # correct class gets +w and other classes get -w/(K-1); probabilities are a softmax of
    # the scaled decision function.
    """An AdaBoost classifier.

    An AdaBoost [1]_ classifier is a meta-estimator that begins by fitting a
    classifier on the original dataset and then fits additional copies of the
    classifier on the same dataset but where the weights of incorrectly
    classified instances are adjusted such that subsequent classifiers focus
    more on difficult cases.

    This class implements the algorithm based on [2]_.

    Read more in the :ref:`User Guide <adaboost>`.

    .. versionadded:: 0.14

    Parameters
    ----------
    estimator : object, default=None
        The base estimator from which the boosted ensemble is built.
        Support for sample weighting is required, as well as proper
        ``classes_`` and ``n_classes_`` attributes. If ``None``, then
        the base estimator is :class:`~sklearn.tree.DecisionTreeClassifier`
        initialized with `max_depth=1`.

        .. versionadded:: 1.2
           `base_estimator` was renamed to `estimator`.

    n_estimators : int, default=50
        The maximum number of estimators at which boosting is terminated.
        In case of perfect fit, the learning procedure is stopped early.
        Values must be in the range `[1, inf)`.

    learning_rate : float, default=1.0
        Weight applied to each classifier at each boosting iteration. A higher
        learning rate increases the contribution of each classifier. There is
        a trade-off between the `learning_rate` and `n_estimators` parameters.
        Values must be in the range `(0.0, inf)`.

    random_state : int, RandomState instance or None, default=None
        Controls the random seed given at each `estimator` at each
        boosting iteration.
        Thus, it is only used when `estimator` exposes a `random_state`.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    Attributes
    ----------
    estimator_ : estimator
        The base estimator from which the ensemble is grown.

        .. versionadded:: 1.2
           `base_estimator_` was renamed to `estimator_`.

    estimators_ : list of classifiers
        The collection of fitted sub-estimators.

    classes_ : ndarray of shape (n_classes,)
        The classes labels.

    n_classes_ : int
        The number of classes.

    estimator_weights_ : ndarray of floats
        Weights for each estimator in the boosted ensemble.

    estimator_errors_ : ndarray of floats
        Classification error for each estimator in the boosted
        ensemble.

    feature_importances_ : ndarray of shape (n_features,)
        The impurity-based feature importances if supported by the
        ``estimator`` (when based on decision trees).

        Warning: impurity-based feature importances can be misleading for
        high cardinality features (many unique values). See
        :func:`sklearn.inspection.permutation_importance` as an alternative.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    AdaBoostRegressor : An AdaBoost regressor that begins by fitting a
        regressor on the original dataset and then fits additional copies of
        the regressor on the same dataset but where the weights of instances
        are adjusted according to the error of the current prediction.

    GradientBoostingClassifier : GB builds an additive model in a forward
        stage-wise fashion. Regression trees are fit on the negative gradient
        of the binomial or multinomial deviance loss function. Binary
        classification is a special case where only a single regression tree is
        induced.

    sklearn.tree.DecisionTreeClassifier : A non-parametric supervised learning
        method used for classification.
        Creates a model that predicts the value of a target variable by
        learning simple decision rules inferred from the data features.

    References
    ----------
    .. [1] Y. Freund, R. Schapire, "A Decision-Theoretic Generalization of
           on-Line Learning and an Application to Boosting", 1995.

    .. [2] :doi:`J. Zhu, H. Zou, S. Rosset, T. Hastie, "Multi-class adaboost."
           Statistics and its Interface 2.3 (2009): 349-360.
           <10.4310/SII.2009.v2.n3.a8>`

    Examples
    --------
    >>> from sklearn.ensemble import AdaBoostClassifier
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=1000, n_features=4,
    ...                            n_informative=2, n_redundant=0,
    ...                            random_state=0, shuffle=False)
    >>> clf = AdaBoostClassifier(n_estimators=100, random_state=0)
    >>> clf.fit(X, y)
    AdaBoostClassifier(n_estimators=100, random_state=0)
    >>> clf.predict([[0, 0, 0, 0]])
    array([1])
    >>> clf.score(X, y)
    0.96

    For a detailed example of using AdaBoost to fit a sequence of DecisionTrees
    as weaklearners, please refer to
    :ref:`sphx_glr_auto_examples_ensemble_plot_adaboost_multiclass.py`.

    For a detailed example of using AdaBoost to fit a non-linearly separable
    classification dataset composed of two Gaussian quantiles clusters, please
    refer to :ref:`sphx_glr_auto_examples_ensemble_plot_adaboost_twoclass.py`.
    """

    def __init__(
        self,
        estimator=None,
        *,
        n_estimators=50,
        learning_rate=1.0,
        random_state=None,
    ):
        # [FUNCTION DOC]
        # Stores parameters by forwarding estimator, n_estimators, learning_rate and
        # random_state to the base constructor.
        super().__init__(
            estimator=estimator,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=random_state,
        )

    def _validate_estimator(self):
        # [FUNCTION DOC]
        # Validates the base estimator, defaulting to DecisionTreeClassifier(max_depth=1),
        # and sets estimator_.
        # Exceptions: ValueError if the estimator's fit lacks a sample_weight parameter.
        """Check the estimator and set the estimator_ attribute."""
        super()._validate_estimator(default=DecisionTreeClassifier(max_depth=1))

        if not has_fit_parameter(self.estimator_, "sample_weight"):
            raise ValueError(
                f"{self.estimator.__class__.__name__} doesn't support sample_weight."
            )

    def _boost(self, iboost, X, y, sample_weight, random_state):
        # [FUNCTION DOC]
        # One SAMME iteration.
        # Steps: build a new estimator (appended to estimators_, seeded from random_state) and
        # fit it with the current weights; on the first iteration set classes_ and n_classes_
        # from it; compute the weighted misclassification rate. If error <= 0 return the
        # unchanged weights with weight 1 and error 0. If error >= 1 - 1/K, remove the
        # estimator and stop (raise if no estimator remains). Otherwise estimator weight =
        # learning_rate * (log((1-err)/err) + log(K-1)); unless this is the last iteration,
        # multiply weights of misclassified, positive-weight samples by exp(estimator weight)
        # (computed in log space).
        # Outputs: (sample_weight or None, estimator_weight, estimator_error).
        # Exceptions: ValueError when the first estimator is worse than random.
        # Side effects: mutates estimators_, sets classes_ / n_classes_ on iteration 0.
        # KNOWN ISSUE: classes_ is taken from the first fitted estimator via getattr with a
        # None default; an estimator without classes_ would make len() fail with TypeError.
        """Implement a single boost.

        Perform a single boost according to the discrete SAMME algorithm and return the
        updated sample weights.

        Parameters
        ----------
        iboost : int
            The index of the current boost iteration.

        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples.

        y : array-like of shape (n_samples,)
            The target values (class labels).

        sample_weight : array-like of shape (n_samples,)
            The current sample weights.

        random_state : RandomState instance
            The RandomState instance used if the base estimator accepts a
            `random_state` attribute.

        Returns
        -------
        sample_weight : array-like of shape (n_samples,) or None
            The reweighted sample weights.
            If None then boosting has terminated early.

        estimator_weight : float
            The weight for the current boost.
            If None then boosting has terminated early.

        estimator_error : float
            The classification error for the current boost.
            If None then boosting has terminated early.
        """
        estimator = self._make_estimator(random_state=random_state)

        estimator.fit(X, y, sample_weight=sample_weight)

        y_predict = estimator.predict(X)

        if iboost == 0:
            self.classes_ = getattr(estimator, "classes_", None)
            self.n_classes_ = len(self.classes_)

        # Instances incorrectly classified
        incorrect = y_predict != y

        # Error fraction
        estimator_error = np.mean(np.average(incorrect, weights=sample_weight, axis=0))

        # Stop if classification is perfect
        if estimator_error <= 0:
            return sample_weight, 1.0, 0.0

        n_classes = self.n_classes_

        # Stop if the error is at least as bad as random guessing
        if estimator_error >= 1.0 - (1.0 / n_classes):
            self.estimators_.pop(-1)
            if len(self.estimators_) == 0:
                raise ValueError(
                    "BaseClassifier in AdaBoostClassifier "
                    "ensemble is worse than random, ensemble "
                    "can not be fit."
                )
            return None, None, None

        # Boost weight using multi-class AdaBoost SAMME alg
        estimator_weight = self.learning_rate * (
            np.log((1.0 - estimator_error) / estimator_error) + np.log(n_classes - 1.0)
        )

        # Only boost the weights if it will fit again
        if not iboost == self.n_estimators - 1:
            # Only boost positive weights
            sample_weight = np.exp(
                np.log(sample_weight)
                + estimator_weight * incorrect * (sample_weight > 0)
            )

        return sample_weight, estimator_weight, estimator_error

    def predict(self, X):
        # [FUNCTION DOC]
        # Predicts labels: for binary problems class index 1 if the decision function is
        # positive else 0; otherwise argmax of the decision function; mapped via classes_.
        # Outputs: ndarray (n_samples,). Exceptions: NotFittedError if not fitted.
        # KNOWN ISSUE: docstring calls it a weighted mean prediction; it is a weighted vote.
        """Predict classes for X.

        The predicted class of an input sample is computed as the weighted mean
        prediction of the classifiers in the ensemble.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. COO, DOK, and LIL are converted to CSR.

        Returns
        -------
        y : ndarray of shape (n_samples,)
            The predicted classes.
        """
        pred = self.decision_function(X)

        if self.n_classes_ == 2:
            return self.classes_.take(pred > 0, axis=0)

        return self.classes_.take(np.argmax(pred, axis=1), axis=0)

    def staged_predict(self, X):
        # [FUNCTION DOC]
        # Generator version of predict over staged_decision_function: yields the predicted
        # labels after each boosting stage. Outputs: ndarray (n_samples,) per stage.
        """Return staged predictions for X.

        The predicted class of an input sample is computed as the weighted mean
        prediction of the classifiers in the ensemble.

        This generator method yields the ensemble prediction after each
        iteration of boosting and therefore allows monitoring, such as to
        determine the prediction on a test set after each boost.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. COO, DOK, and LIL are converted to CSR.

        Yields
        ------
        y : generator of ndarray of shape (n_samples,)
            The predicted classes.
        """
        X = self._check_X(X)

        n_classes = self.n_classes_
        classes = self.classes_

        if n_classes == 2:
            for pred in self.staged_decision_function(X):
                yield np.array(classes.take(pred > 0, axis=0))

        else:
            for pred in self.staged_decision_function(X):
                yield np.array(classes.take(np.argmax(pred, axis=1), axis=0))

    def decision_function(self, X):
        # [FUNCTION DOC]
        # Weighted SAMME decision function. For each estimator, a sample scores w for the
        # predicted class and -w/(K-1) for the others; scores are summed and divided by the
        # total of estimator_weights_. Binary: negate the first column and sum both columns
        # to get a 1d score. Single class: returns zeros of shape (n_samples, 1).
        # Outputs: (n_samples, K) array, or (n_samples,) for binary.
        # Exceptions: NotFittedError if not fitted; ValueError on invalid X.
        # KNOWN ISSUE: docstring says binary output has shape (n_samples, 1); the code
        # returns a 1d array of shape (n_samples,).
        """Compute the decision function of ``X``.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. COO, DOK, and LIL are converted to CSR.

        Returns
        -------
        score : ndarray of shape of (n_samples, k)
            The decision function of the input samples. The order of
            outputs is the same as in the :term:`classes_` attribute.
            Binary classification is a special case with ``k == 1``,
            otherwise ``k==n_classes``. For binary classification,
            values closer to -1 or 1 mean more like the first or second
            class in ``classes_``, respectively.
        """
        check_is_fitted(self)
        X = self._check_X(X)

        n_classes = self.n_classes_
        classes = self.classes_[:, np.newaxis]

        if n_classes == 1:
            return np.zeros_like(X, shape=(X.shape[0], 1))

        pred = sum(
            np.where(
                (estimator.predict(X) == classes).T,
                w,
                -1 / (n_classes - 1) * w,
            )
            for estimator, w in zip(self.estimators_, self.estimator_weights_)
        )

        pred /= self.estimator_weights_.sum()
        if n_classes == 2:
            pred[:, 0] *= -1
            return pred.sum(axis=1)
        return pred

    def staged_decision_function(self, X):
        # [FUNCTION DOC]
        # Generator yielding the decision function after each stage, using the cumulative
        # sum of estimator weights seen so far as the normaliser; binary case collapsed to a
        # 1d score like decision_function (on a copy, so accumulation is not altered).
        # Exceptions: NotFittedError if not fitted.
        # KNOWN ISSUE: a single-class fit leads to division by zero in -1/(K-1).
        """Compute decision function of ``X`` for each boosting iteration.

        This method allows monitoring (i.e. determine error on testing set)
        after each boosting iteration.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. COO, DOK, and LIL are converted to CSR.

        Yields
        ------
        score : generator of ndarray of shape (n_samples, k)
            The decision function of the input samples. The order of
            outputs is the same as in the :term:`classes_` attribute.
            Binary classification is a special case with ``k == 1``,
            otherwise ``k==n_classes``. For binary classification,
            values closer to -1 or 1 mean more like the first or second
            class in ``classes_``, respectively.
        """
        check_is_fitted(self)
        X = self._check_X(X)

        n_classes = self.n_classes_
        classes = self.classes_[:, np.newaxis]
        pred = None
        norm = 0.0

        for weight, estimator in zip(self.estimator_weights_, self.estimators_):
            norm += weight

            current_pred = np.where(
                (estimator.predict(X) == classes).T,
                weight,
                -1 / (n_classes - 1) * weight,
            )

            if pred is None:
                pred = current_pred
            else:
                pred += current_pred

            if n_classes == 2:
                tmp_pred = np.copy(pred)
                tmp_pred[:, 0] *= -1
                yield (tmp_pred / norm).sum(axis=1)
            else:
                yield pred / norm

    @staticmethod
    def _compute_proba_from_decision(decision, n_classes):
        # [FUNCTION DOC]
        # HELPER: Static method converting a decision function into probabilities via
        # softmax of f/(K-1). Binary: builds two columns (-d/2, d/2) first.
        # Inputs: decision - (n_samples,) or (n_samples, K) array; n_classes - K.
        # Outputs: ndarray (n_samples, K) of probabilities.
        # Side effects: in the multiclass case divides decision in place and softmax runs
        # with copy False, so the passed array is modified.
        """Compute probabilities from the decision function.

        This is based eq. (15) of [1] where:
            p(y=c|X) = exp((1 / K-1) f_c(X)) / sum_k(exp((1 / K-1) f_k(X)))
                     = softmax((1 / K-1) * f(X))

        References
        ----------
        .. [1] J. Zhu, H. Zou, S. Rosset, T. Hastie, "Multi-class AdaBoost",
               2009.
        """
        if n_classes == 2:
            decision = np.vstack([-decision, decision]).T / 2
        else:
            decision /= n_classes - 1
        return softmax(decision, copy=False)

    def predict_proba(self, X):
        # [FUNCTION DOC]
        # Class probabilities from the decision function via _compute_proba_from_decision.
        # Single class: returns ones of shape (n_samples, 1) without validating X.
        # Outputs: ndarray (n_samples, n_classes). Exceptions: NotFittedError.
        # KNOWN ISSUE: docstring describes a weighted mean of sub-estimator probabilities,
        # but probabilities derive from the SAMME decision function.
        """Predict class probabilities for X.

        The predicted class probabilities of an input sample is computed as
        the weighted mean predicted class probabilities of the classifiers
        in the ensemble.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. COO, DOK, and LIL are converted to CSR.

        Returns
        -------
        p : ndarray of shape (n_samples, n_classes)
            The class probabilities of the input samples. The order of
            outputs is the same as in the :term:`classes_` attribute.
        """
        check_is_fitted(self)
        n_classes = self.n_classes_

        if n_classes == 1:
            return np.ones((_num_samples(X), 1))

        decision = self.decision_function(X)
        return self._compute_proba_from_decision(decision, n_classes)

    def staged_predict_proba(self, X):
        # [FUNCTION DOC]
        # Generator yielding class probabilities after each stage from
        # staged_decision_function.
        # KNOWN ISSUE: docstring says each yielded array has shape (n_samples,); it is
        # (n_samples, n_classes).
        """Predict class probabilities for X.

        The predicted class probabilities of an input sample is computed as
        the weighted mean predicted class probabilities of the classifiers
        in the ensemble.

        This generator method yields the ensemble predicted class probabilities
        after each iteration of boosting and therefore allows monitoring, such
        as to determine the predicted class probabilities on a test set after
        each boost.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. COO, DOK, and LIL are converted to CSR.

        Yields
        ------
        p : generator of ndarray of shape (n_samples,)
            The class probabilities of the input samples. The order of
            outputs is the same as in the :term:`classes_` attribute.
        """

        n_classes = self.n_classes_

        for decision in self.staged_decision_function(X):
            yield self._compute_proba_from_decision(decision, n_classes)

    def predict_log_proba(self, X):
        # [FUNCTION DOC]
        # Natural log of predict_proba. Outputs: ndarray (n_samples, n_classes); may contain
        # -inf where a probability underflows to 0.
        """Predict class log-probabilities for X.

        The predicted class log-probabilities of an input sample is computed as
        the weighted mean predicted class log-probabilities of the classifiers
        in the ensemble.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. COO, DOK, and LIL are converted to CSR.

        Returns
        -------
        p : ndarray of shape (n_samples, n_classes)
            The class probabilities of the input samples. The order of
            outputs is the same as in the :term:`classes_` attribute.
        """
        return np.log(self.predict_proba(X))


class AdaBoostRegressor(_RoutingNotSupportedMixin, RegressorMixin, BaseWeightBoosting):
    # [CLASS DOC]
    # AdaBoost.R2 regressor. Default base estimator is DecisionTreeRegressor(max_depth=3).
    # Each iteration fits on a weighted bootstrap sample; prediction is the weighted
    # median over sub-estimator predictions.
    """An AdaBoost regressor.

    An AdaBoost [1]_ regressor is a meta-estimator that begins by fitting a
    regressor on the original dataset and then fits additional copies of the
    regressor on the same dataset but where the weights of instances are
    adjusted according to the error of the current prediction. As such,
    subsequent regressors focus more on difficult cases.

    This class implements the algorithm known as AdaBoost.R2 [2]_.

    Read more in the :ref:`User Guide <adaboost>`.

    .. versionadded:: 0.14

    Parameters
    ----------
    estimator : object, default=None
        The base estimator from which the boosted ensemble is built.
        If ``None``, then the base estimator is
        :class:`~sklearn.tree.DecisionTreeRegressor` initialized with
        `max_depth=3`.

        .. versionadded:: 1.2
           `base_estimator` was renamed to `estimator`.

    n_estimators : int, default=50
        The maximum number of estimators at which boosting is terminated.
        In case of perfect fit, the learning procedure is stopped early.
        Values must be in the range `[1, inf)`.

    learning_rate : float, default=1.0
        Weight applied to each regressor at each boosting iteration. A higher
        learning rate increases the contribution of each regressor. There is
        a trade-off between the `learning_rate` and `n_estimators` parameters.
        Values must be in the range `(0.0, inf)`.

    loss : {'linear', 'square', 'exponential'}, default='linear'
        The loss function to use when updating the weights after each
        boosting iteration.

    random_state : int, RandomState instance or None, default=None
        Controls the random seed given at each `estimator` at each
        boosting iteration.
        Thus, it is only used when `estimator` exposes a `random_state`.
        In addition, it controls the bootstrap of the weights used to train the
        `estimator` at each boosting iteration.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    Attributes
    ----------
    estimator_ : estimator
        The base estimator from which the ensemble is grown.

        .. versionadded:: 1.2
           `base_estimator_` was renamed to `estimator_`.

    estimators_ : list of regressors
        The collection of fitted sub-estimators.

    estimator_weights_ : ndarray of floats
        Weights for each estimator in the boosted ensemble.

    estimator_errors_ : ndarray of floats
        Regression error for each estimator in the boosted ensemble.

    feature_importances_ : ndarray of shape (n_features,)
        The impurity-based feature importances if supported by the
        ``estimator`` (when based on decision trees).

        Warning: impurity-based feature importances can be misleading for
        high cardinality features (many unique values). See
        :func:`sklearn.inspection.permutation_importance` as an alternative.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    AdaBoostClassifier : An AdaBoost classifier.
    GradientBoostingRegressor : Gradient Boosting Classification Tree.
    sklearn.tree.DecisionTreeRegressor : A decision tree regressor.

    References
    ----------
    .. [1] Y. Freund, R. Schapire, "A Decision-Theoretic Generalization of
           on-Line Learning and an Application to Boosting", 1995.

    .. [2] `H. Drucker, "Improving Regressors using Boosting Techniques", 1997.
           <https://dl.acm.org/doi/10.5555/645526.657132>`_

    Examples
    --------
    >>> from sklearn.ensemble import AdaBoostRegressor
    >>> from sklearn.datasets import make_regression
    >>> X, y = make_regression(n_features=4, n_informative=2,
    ...                        random_state=0, shuffle=False)
    >>> regr = AdaBoostRegressor(random_state=0, n_estimators=100)
    >>> regr.fit(X, y)
    AdaBoostRegressor(n_estimators=100, random_state=0)
    >>> regr.predict([[0, 0, 0, 0]])
    array([4.7972])
    >>> regr.score(X, y)
    0.9771

    For a detailed example of utilizing :class:`~sklearn.ensemble.AdaBoostRegressor`
    to fit a sequence of decision trees as weak learners, please refer to
    :ref:`sphx_glr_auto_examples_ensemble_plot_adaboost_regression.py`.
    """

    _parameter_constraints: dict = {
        **BaseWeightBoosting._parameter_constraints,
        "loss": [StrOptions({"linear", "square", "exponential"})],
    }

    def __init__(
        self,
        estimator=None,
        *,
        n_estimators=50,
        learning_rate=1.0,
        loss="linear",
        random_state=None,
    ):
        # [FUNCTION DOC]
        # Stores parameters: forwards estimator, n_estimators, learning_rate, random_state to
        # the base constructor, and stores loss.
        # KNOWN ISSUE: random_state is assigned a second time redundantly.
        super().__init__(
            estimator=estimator,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=random_state,
        )

        self.loss = loss
        self.random_state = random_state

    def _validate_estimator(self):
        # [FUNCTION DOC]
        # Validates the base estimator (default DecisionTreeRegressor(max_depth=3)) and sets
        # estimator_.
        """Check the estimator and set the estimator_ attribute."""
        super()._validate_estimator(default=DecisionTreeRegressor(max_depth=3))

    def _boost(self, iboost, X, y, sample_weight, random_state):
        # [FUNCTION DOC]
        # One AdaBoost.R2 iteration.
        # Steps: create a new estimator (appended to estimators_); draw a bootstrap of
        # n_samples indices with replacement with probabilities sample_weight from
        # random_state; fit on it; predict on all of X; take absolute errors on samples
        # with positive weight, divide by their maximum (if non-zero), transform by loss
        # (linear: unchanged, square: squared, exponential: 1 - exp(-e)); average loss =
        # weighted sum of these errors. If loss <= 0: return weights unchanged, weight 1,
        # error 0. If loss >= 0.5: drop the estimator unless it is the only one and stop.
        # Otherwise beta = loss/(1-loss), estimator weight = learning_rate * log(1/beta);
        # unless last iteration, multiply positive weights by beta ** ((1 - e) *
        # learning_rate).
        # Outputs: (sample_weight or None, estimator_weight, estimator_error).
        # Side effects: mutates estimators_ and sample_weight in place; consumes randomness.
        # Exceptions: ValueError from choice if sample_weight does not sum to 1.
        """Implement a single boost for regression

        Perform a single boost according to the AdaBoost.R2 algorithm and
        return the updated sample weights.

        Parameters
        ----------
        iboost : int
            The index of the current boost iteration.

        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples.

        y : array-like of shape (n_samples,)
            The target values (class labels in classification, real numbers in
            regression).

        sample_weight : array-like of shape (n_samples,)
            The current sample weights.

        random_state : RandomState
            The RandomState instance used if the base estimator accepts a
            `random_state` attribute.
            Controls also the bootstrap of the weights used to train the weak
            learner.

        Returns
        -------
        sample_weight : array-like of shape (n_samples,) or None
            The reweighted sample weights.
            If None then boosting has terminated early.

        estimator_weight : float
            The weight for the current boost.
            If None then boosting has terminated early.

        estimator_error : float
            The regression error for the current boost.
            If None then boosting has terminated early.
        """
        estimator = self._make_estimator(random_state=random_state)

        # Weighted sampling of the training set with replacement
        bootstrap_idx = random_state.choice(
            np.arange(_num_samples(X)),
            size=_num_samples(X),
            replace=True,
            p=sample_weight,
        )

        # Fit on the bootstrapped sample and obtain a prediction
        # for all samples in the training set
        X_ = _safe_indexing(X, bootstrap_idx)
        y_ = _safe_indexing(y, bootstrap_idx)
        estimator.fit(X_, y_)
        y_predict = estimator.predict(X)

        error_vect = np.abs(y_predict - y)
        sample_mask = sample_weight > 0
        masked_sample_weight = sample_weight[sample_mask]
        masked_error_vector = error_vect[sample_mask]

        error_max = masked_error_vector.max()
        if error_max != 0:
            masked_error_vector /= error_max

        if self.loss == "square":
            masked_error_vector **= 2
        elif self.loss == "exponential":
            masked_error_vector = 1.0 - np.exp(-masked_error_vector)

        # Calculate the average loss
        estimator_error = (masked_sample_weight * masked_error_vector).sum()

        if estimator_error <= 0:
            # Stop if fit is perfect
            return sample_weight, 1.0, 0.0

        elif estimator_error >= 0.5:
            # Discard current estimator only if it isn't the only one
            if len(self.estimators_) > 1:
                self.estimators_.pop(-1)
            return None, None, None

        beta = estimator_error / (1.0 - estimator_error)

        # Boost weight using AdaBoost.R2 alg
        estimator_weight = self.learning_rate * np.log(1.0 / beta)

        if not iboost == self.n_estimators - 1:
            sample_weight[sample_mask] *= np.power(
                beta, (1.0 - masked_error_vector) * self.learning_rate
            )

        return sample_weight, estimator_weight, estimator_error

    def _get_median_predict(self, X, limit):
        # [FUNCTION DOC]
        # HELPER: Weighted median of the first limit estimators' predictions per sample:
        # sort predictions, accumulate the corresponding estimator weights, and pick the
        # first prediction whose cumulative weight reaches half the total.
        # Inputs: X - validated samples; limit - number of leading estimators to use.
        # Outputs: ndarray (n_samples,).
        # Evaluate predictions of all estimators
        predictions = np.array([est.predict(X) for est in self.estimators_[:limit]]).T

        # Sort the predictions
        sorted_idx = np.argsort(predictions, axis=1)

        # Find index of median prediction for each sample
        weight_cdf = np.cumsum(self.estimator_weights_[sorted_idx], axis=1)
        median_or_above = weight_cdf >= 0.5 * weight_cdf[:, -1][:, np.newaxis]
        median_idx = median_or_above.argmax(axis=1)

        median_estimators = sorted_idx[np.arange(_num_samples(X)), median_idx]

        # Return median predictions
        return predictions[np.arange(_num_samples(X)), median_estimators]

    def predict(self, X):
        # [FUNCTION DOC]
        # Weighted-median prediction using all fitted estimators.
        # Outputs: ndarray (n_samples,). Exceptions: NotFittedError; ValueError on bad X.
        """Predict regression value for X.

        The predicted regression value of an input sample is computed
        as the weighted median prediction of the regressors in the ensemble.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. COO, DOK, and LIL are converted to CSR.

        Returns
        -------
        y : ndarray of shape (n_samples,)
            The predicted regression values.
        """
        check_is_fitted(self)
        X = self._check_X(X)

        return self._get_median_predict(X, len(self.estimators_))

    def staged_predict(self, X):
        # [FUNCTION DOC]
        # Generator yielding the weighted-median prediction using the first 1, 2, ...,
        # len(estimators_) estimators. Exceptions: NotFittedError if not fitted.
        """Return staged predictions for X.

        The predicted regression value of an input sample is computed
        as the weighted median prediction of the regressors in the ensemble.

        This generator method yields the ensemble prediction after each
        iteration of boosting and therefore allows monitoring, such as to
        determine the prediction on a test set after each boost.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples.

        Yields
        ------
        y : generator of ndarray of shape (n_samples,)
            The predicted regression values.
        """
        check_is_fitted(self)
        X = self._check_X(X)

        for i, _ in enumerate(self.estimators_, 1):
            yield self._get_median_predict(X, limit=i)
