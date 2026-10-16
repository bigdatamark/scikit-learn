# [MODULE DOC]
# Meta-estimators that extend single-output estimators to multi-output targets.
# MultiOutputRegressor / MultiOutputClassifier fit one independent clone per target
# column (shared base _MultiOutputEstimator). ClassifierChain / RegressorChain (shared
# base _BaseChain) fit clones sequentially, each one seeing the original features plus
# the previous targets (true values or cross-validated predictions) in the chain order.
# Public names: those in __all__. No import-time side effects.
"""Multioutput regression and classification.

The estimators provided in this module are meta-estimators: they require
a base estimator to be provided in their constructor. The meta-estimator
extends single output estimators to multioutput estimators.
"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

from abc import ABCMeta, abstractmethod
from numbers import Integral

import numpy as np
import scipy.sparse as sp

from sklearn.base import (
    BaseEstimator,
    ClassifierMixin,
    MetaEstimatorMixin,
    RegressorMixin,
    _fit_context,
    clone,
    is_classifier,
)
from sklearn.model_selection import cross_val_predict
from sklearn.utils import check_random_state, get_tags
from sklearn.utils._param_validation import HasMethods, StrOptions
from sklearn.utils._response import _get_response_values
from sklearn.utils._sparse import _align_api_if_sparse
from sklearn.utils._user_interface import _print_elapsed_time
from sklearn.utils.metadata_routing import (
    MetadataRouter,
    MethodMapping,
    _manual_routing,
    _raise_for_params,
    _routing_enabled,
    process_routing,
)
from sklearn.utils.metaestimators import available_if
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.parallel import Parallel, delayed
from sklearn.utils.validation import (
    _check_method_params,
    _check_response_method,
    check_is_fitted,
    has_fit_parameter,
    validate_data,
)

__all__ = [
    "ClassifierChain",
    "MultiOutputClassifier",
    "MultiOutputRegressor",
    "RegressorChain",
]


def _fit_estimator(estimator, X, y, sample_weight=None, **fit_params):
    # [FUNCTION DOC]
    # HELPER: Clones estimator and fits it on (X, y), passing sample_weight only when it is
    # not None, plus any extra fit params.
    # Outputs: the fitted clone.
    estimator = clone(estimator)
    if sample_weight is not None:
        estimator.fit(X, y, sample_weight=sample_weight, **fit_params)
    else:
        estimator.fit(X, y, **fit_params)
    return estimator


def _partial_fit_estimator(
    estimator, X, y, classes=None, partial_fit_params=None, first_time=True
):
    # [FUNCTION DOC]
    # HELPER: Runs one partial_fit step for a single output. Clones the estimator on the
    # first call, passes classes only when given, and forwards partial_fit_params (None
    # treated as empty dict).
    # Inputs: estimator; X; y - 1-D target column; classes - labels for this output or None;
    # partial_fit_params - dict or None; first_time - bool, clone when True.
    # Outputs: the updated estimator (mutated in place when not first_time).
    partial_fit_params = {} if partial_fit_params is None else partial_fit_params
    if first_time:
        estimator = clone(estimator)

    if classes is not None:
        estimator.partial_fit(X, y, classes=classes, **partial_fit_params)
    else:
        estimator.partial_fit(X, y, **partial_fit_params)
    return estimator


def _available_if_estimator_has(attr):
    # [FUNCTION DOC]
    # HELPER: Returns an available_if decorator whose predicate is true when every fitted
    # sub-estimator has attr (if estimators_ exists) or else when the unfitted estimator
    # has attr.
    # Inputs: attr - attribute name. Outputs: available_if descriptor factory.
    """Return a function to check if the sub-estimator(s) has(have) `attr`.

    Helper for Chain implementations.
    """

    def _check(self):
        # [FUNCTION DOC]
        # HELPER: Predicate described above; returns a bool rather than raising.
        if hasattr(self, "estimators_"):
            return all(hasattr(est, attr) for est in self.estimators_)

        if hasattr(self.estimator, attr):
            return True

        return False

    return available_if(_check)


class _MultiOutputEstimator(MetaEstimatorMixin, BaseEstimator, metaclass=ABCMeta):
    # [CLASS DOC]
    # HELPER: Abstract base for the independent per-output meta-estimators. Validates that
    # estimator has fit and predict; fits one clone per column of y in parallel (joblib,
    # n_jobs). Fitted attributes: estimators_, and n_features_in_ / feature_names_in_
    # copied from the first sub-estimator when available.
    _parameter_constraints: dict = {
        "estimator": [HasMethods(["fit", "predict"])],
        "n_jobs": [Integral, None],
    }

    @abstractmethod
    def __init__(self, estimator, *, n_jobs=None):
        # [FUNCTION DOC]
        # Stores estimator and n_jobs (abstract so the base cannot be instantiated).
        self.estimator = estimator
        self.n_jobs = n_jobs

    @_available_if_estimator_has("partial_fit")
    @_fit_context(
        # MultiOutput*.estimator is not validated yet
        prefer_skip_nested_validation=False
    )
    def partial_fit(self, X, y, classes=None, sample_weight=None, **partial_fit_params):
        # [FUNCTION DOC]
        # Incremental fit. first_time is True if estimators_ does not exist. Validates y as
        # multi-output (X not validated). When routing is enabled, sample_weight is added to
        # the routed params; otherwise sample_weight is only allowed if the estimator's fit
        # accepts it and is wrapped in a manual routing object. Then, per column i in parallel,
        # calls _partial_fit_estimator with the existing estimator (or template on first call)
        # and classes[i]. On the first call copies n_features_in_ / feature_names_in_.
        # Inputs: X; y - 2-D targets; classes - list of per-output label arrays or None;
        # sample_weight - (n_samples,) or None; partial_fit_params.
        # Outputs: self.
        # Exceptions: ValueError if y is 1-D, if sample weights are unsupported, or for
        # unrouted metadata.
        # Side effects: replaces estimators_; mutates the partial_fit_params kwargs dict.
        # KNOWN ISSUE: has_fit_parameter checks the fit signature, not partial_fit, when
        # deciding whether sample_weight is supported.
        """Incrementally fit a separate model for each class output.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input data.

        y : {array-like, sparse matrix} of shape (n_samples, n_outputs)
            Multi-output targets.

        classes : list of ndarray of shape (n_outputs,), default=None
            Each array is unique classes for one output in str/int.
            Can be obtained via
            ``[np.unique(y[:, i]) for i in range(y.shape[1])]``, where `y`
            is the target matrix of the entire dataset.
            This argument is required for the first call to partial_fit
            and can be omitted in the subsequent calls.
            Note that `y` doesn't need to contain all labels in `classes`.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If `None`, then samples are equally weighted.
            Only supported if the underlying regressor supports sample
            weights.

        **partial_fit_params : dict of str -> object
            Parameters passed to the ``estimator.partial_fit`` method of each
            sub-estimator.

            Only available if `enable_metadata_routing=True`. See the
            :ref:`User Guide <metadata_routing>`.

            .. versionadded:: 1.3

        Returns
        -------
        self : object
            Returns a fitted instance.
        """
        _raise_for_params(partial_fit_params, self, "partial_fit")

        first_time = not hasattr(self, "estimators_")

        y = validate_data(self, X="no_validation", y=y, multi_output=True)

        if y.ndim == 1:
            raise ValueError(
                "y must have at least two dimensions for "
                "multi-output regression but has only one."
            )

        if _routing_enabled():
            if sample_weight is not None:
                partial_fit_params["sample_weight"] = sample_weight
            routed_params = process_routing(
                self,
                "partial_fit",
                **partial_fit_params,
            )
        else:
            if sample_weight is not None and not has_fit_parameter(
                self.estimator, "sample_weight"
            ):
                raise ValueError(
                    "Underlying estimator does not support sample weights."
                )

            sw = {"sample_weight": sample_weight} if sample_weight is not None else {}
            routed_params = _manual_routing({"estimator": {"partial_fit": sw}})

        self.estimators_ = Parallel(n_jobs=self.n_jobs)(
            delayed(_partial_fit_estimator)(
                self.estimators_[i] if not first_time else self.estimator,
                X,
                y[:, i],
                classes[i] if classes is not None else None,
                partial_fit_params=routed_params.estimator.partial_fit,
                first_time=first_time,
            )
            for i in range(y.shape[1])
        )

        if first_time and hasattr(self.estimators_[0], "n_features_in_"):
            self.n_features_in_ = self.estimators_[0].n_features_in_
        if first_time and hasattr(self.estimators_[0], "feature_names_in_"):
            self.feature_names_in_ = self.estimators_[0].feature_names_in_

        return self

    @_fit_context(
        # MultiOutput*.estimator is not validated yet
        prefer_skip_nested_validation=False
    )
    def fit(self, X, y, sample_weight=None, **fit_params):
        # [FUNCTION DOC]
        # Checks estimator has fit, validates y as multi-output (classification targets checked
        # for classifiers), rejects 1-D y. Builds routed params: via process_routing when
        # routing is enabled (sample_weight added), else sample-aligned fit_params via
        # _check_method_params plus sample_weight (after checking support). Fits
        # _fit_estimator for each column in parallel and copies feature metadata from the
        # first fitted estimator.
        # Inputs: X; y - (n_samples, n_outputs); sample_weight; fit_params.
        # Outputs: self.
        # Exceptions: ValueError for missing fit, 1-D y, unsupported sample_weight, invalid
        # classification targets.
        # KNOWN ISSUE: the "should implement a fit method" check is unreachable in practice
        # since parameter validation already requires fit and predict.
        """Fit the model to data, separately for each output variable.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input data.

        y : {array-like, sparse matrix} of shape (n_samples, n_outputs)
            Multi-output targets. An indicator matrix turns on multilabel
            estimation.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If `None`, then samples are equally weighted.
            Only supported if the underlying regressor supports sample
            weights.

        **fit_params : dict of string -> object
            Parameters passed to the ``estimator.fit`` method of each step.

            .. versionadded:: 0.23

        Returns
        -------
        self : object
            Returns a fitted instance.
        """
        if not hasattr(self.estimator, "fit"):
            raise ValueError("The base estimator should implement a fit method")

        y = validate_data(self, X="no_validation", y=y, multi_output=True)

        if is_classifier(self):
            check_classification_targets(y)

        if y.ndim == 1:
            raise ValueError(
                "y must have at least two dimensions for "
                "multi-output regression but has only one."
            )

        if _routing_enabled():
            if sample_weight is not None:
                fit_params["sample_weight"] = sample_weight
            routed_params = process_routing(
                self,
                "fit",
                **fit_params,
            )
        else:
            if sample_weight is not None and not has_fit_parameter(
                self.estimator, "sample_weight"
            ):
                raise ValueError(
                    "Underlying estimator does not support sample weights."
                )

            fit_kwargs = dict(_check_method_params(X, params=fit_params))
            if sample_weight is not None:
                fit_kwargs["sample_weight"] = sample_weight
            routed_params = _manual_routing({"estimator": {"fit": fit_kwargs}})

        self.estimators_ = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_estimator)(
                self.estimator, X, y[:, i], **routed_params.estimator.fit
            )
            for i in range(y.shape[1])
        )

        if hasattr(self.estimators_[0], "n_features_in_"):
            self.n_features_in_ = self.estimators_[0].n_features_in_
        if hasattr(self.estimators_[0], "feature_names_in_"):
            self.feature_names_in_ = self.estimators_[0].feature_names_in_

        return self

    def predict(self, X):
        # [FUNCTION DOC]
        # Checks fitted, then calls predict of every sub-estimator in parallel and returns the
        # transposed stacked result (n_samples, n_outputs).
        # Exceptions: NotFittedError; ValueError if the first estimator lacks predict
        # (effectively unreachable).
        """Predict multi-output variable using model for each target variable.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input data.

        Returns
        -------
        y : {array-like, sparse matrix} of shape (n_samples, n_outputs)
            Multi-output targets predicted across multiple predictors.
            Note: Separate models are generated for each predictor.
        """
        check_is_fitted(self)
        if not hasattr(self.estimators_[0], "predict"):
            raise ValueError("The base estimator should implement a predict method")

        y = Parallel(n_jobs=self.n_jobs)(
            delayed(e.predict)(X) for e in self.estimators_
        )

        return np.asarray(y).T

    def __sklearn_tags__(self):
        # [FUNCTION DOC]
        # Copies the wrapped estimator's sparse input tag and marks targets as multi-output,
        # not single-output.
        tags = super().__sklearn_tags__()
        tags.input_tags.sparse = get_tags(self.estimator).input_tags.sparse
        tags.target_tags.single_output = False
        tags.target_tags.multi_output = True
        return tags

    def get_metadata_routing(self):
        # [FUNCTION DOC]
        # MetadataRouter routing partial_fit->partial_fit and fit->fit to the wrapped estimator.
        """Get metadata routing of this object.

        Please check :ref:`User Guide <metadata_routing>` on how the routing
        mechanism works.

        .. versionadded:: 1.3

        Returns
        -------
        routing : MetadataRouter
            A :class:`~sklearn.utils.metadata_routing.MetadataRouter` encapsulating
            routing information.
        """
        router = MetadataRouter(owner=self).add(
            estimator=self.estimator,
            method_mapping=MethodMapping()
            .add(caller="partial_fit", callee="partial_fit")
            .add(caller="fit", callee="fit"),
        )
        return router


class MultiOutputRegressor(RegressorMixin, _MultiOutputEstimator):
    # [CLASS DOC]
    # Regressor variant of _MultiOutputEstimator; adds no fitted attributes beyond the base.
    """Multi target regression.

    This strategy consists of fitting one regressor per target. This is a
    simple strategy for extending regressors that do not natively support
    multi-target regression.

    .. versionadded:: 0.18

    Parameters
    ----------
    estimator : estimator object
        An estimator object implementing :term:`fit` and :term:`predict`.

    n_jobs : int or None, optional (default=None)
        The number of jobs to run in parallel.
        :meth:`fit`, :meth:`predict` and :meth:`partial_fit` (if supported
        by the passed estimator) will be parallelized for each target.

        When individual estimators are fast to train or predict,
        using ``n_jobs > 1`` can result in slower performance due
        to the parallelism overhead.

        ``None`` means `1` unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all available processes / threads.
        See :term:`Glossary <n_jobs>` for more details.

        .. versionchanged:: 0.20
            `n_jobs` default changed from `1` to `None`.

    Attributes
    ----------
    estimators_ : list of ``n_output`` estimators
        Estimators used for predictions.

    n_features_in_ : int
        Number of features seen during :term:`fit`. Only defined if the
        underlying `estimator` exposes such an attribute when fit.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Only defined if the
        underlying estimators expose such an attribute when fit.

        .. versionadded:: 1.0

    See Also
    --------
    RegressorChain : A multi-label model that arranges regressions into a
        chain.
    MultiOutputClassifier : Classifies each output independently rather than
        chaining.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.datasets import load_linnerud
    >>> from sklearn.multioutput import MultiOutputRegressor
    >>> from sklearn.linear_model import Ridge
    >>> X, y = load_linnerud(return_X_y=True)
    >>> regr = MultiOutputRegressor(Ridge(random_state=123)).fit(X, y)
    >>> regr.predict(X[[0]])
    array([[176, 35.1, 57.1]])
    """

    def __init__(self, estimator, *, n_jobs=None):
        # [FUNCTION DOC]
        # Forwards estimator and n_jobs to the base constructor.
        super().__init__(estimator, n_jobs=n_jobs)

    @_available_if_estimator_has("partial_fit")
    def partial_fit(self, X, y, sample_weight=None, **partial_fit_params):
        # [FUNCTION DOC]
        # Delegates to the base partial_fit without classes.
        # Outputs: None.
        # KNOWN ISSUE: the return value of super().partial_fit is discarded, so this returns
        # None instead of self as documented, breaking method chaining.
        """Incrementally fit the model to data, for each output variable.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input data.

        y : {array-like, sparse matrix} of shape (n_samples, n_outputs)
            Multi-output targets.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If `None`, then samples are equally weighted.
            Only supported if the underlying regressor supports sample
            weights.

        **partial_fit_params : dict of str -> object
            Parameters passed to the ``estimator.partial_fit`` method of each
            sub-estimator.

            Only available if `enable_metadata_routing=True`. See the
            :ref:`User Guide <metadata_routing>`.

            .. versionadded:: 1.3

        Returns
        -------
        self : object
            Returns a fitted instance.
        """
        super().partial_fit(X, y, sample_weight=sample_weight, **partial_fit_params)


class MultiOutputClassifier(ClassifierMixin, _MultiOutputEstimator):
    # [CLASS DOC]
    # Classifier variant of _MultiOutputEstimator. Adds classes_ (list of per-output class
    # arrays), predict_proba (list output) and a subset-accuracy score.
    """Multi target classification.

    This strategy consists of fitting one classifier per target. This is a
    simple strategy for extending classifiers that do not natively support
    multi-target classification.

    Parameters
    ----------
    estimator : estimator object
        An estimator object implementing :term:`fit` and :term:`predict`.
        A :term:`predict_proba` method will be exposed only if `estimator` implements
        it.

    n_jobs : int or None, optional (default=None)
        The number of jobs to run in parallel.
        :meth:`fit`, :meth:`predict` and :meth:`partial_fit` (if supported
        by the passed estimator) will be parallelized for each target.

        When individual estimators are fast to train or predict,
        using ``n_jobs > 1`` can result in slower performance due
        to the parallelism overhead.

        ``None`` means `1` unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all available processes / threads.
        See :term:`Glossary <n_jobs>` for more details.

        .. versionchanged:: 0.20
            `n_jobs` default changed from `1` to `None`.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        Class labels.

    estimators_ : list of ``n_output`` estimators
        Estimators used for predictions.

    n_features_in_ : int
        Number of features seen during :term:`fit`. Only defined if the
        underlying `estimator` exposes such an attribute when fit.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Only defined if the
        underlying estimators expose such an attribute when fit.

        .. versionadded:: 1.0

    See Also
    --------
    ClassifierChain : A multi-label model that arranges binary classifiers
        into a chain.
    MultiOutputRegressor : Fits one regressor per target variable.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.datasets import make_multilabel_classification
    >>> from sklearn.multioutput import MultiOutputClassifier
    >>> from sklearn.linear_model import LogisticRegression
    >>> X, y = make_multilabel_classification(n_classes=3, random_state=0)
    >>> clf = MultiOutputClassifier(LogisticRegression()).fit(X, y)
    >>> clf.predict(X[-2:])
    array([[1, 1, 1],
           [1, 0, 1]])
    """

    def __init__(self, estimator, *, n_jobs=None):
        # [FUNCTION DOC]
        # Forwards estimator and n_jobs to the base constructor.
        super().__init__(estimator, n_jobs=n_jobs)

    def fit(self, X, Y, sample_weight=None, **fit_params):
        # [FUNCTION DOC]
        # Runs the base fit, then sets classes_ to the list of each sub-estimator's classes_.
        # Outputs: self.
        """Fit the model to data matrix X and targets Y.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input data.

        Y : array-like of shape (n_samples, n_classes)
            The target values.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If `None`, then samples are equally weighted.
            Only supported if the underlying classifier supports sample
            weights.

        **fit_params : dict of string -> object
            Parameters passed to the ``estimator.fit`` method of each step.

            .. versionadded:: 0.23

        Returns
        -------
        self : object
            Returns a fitted instance.
        """
        super().fit(X, Y, sample_weight=sample_weight, **fit_params)
        self.classes_ = [estimator.classes_ for estimator in self.estimators_]
        return self

    def _check_predict_proba(self):
        # [FUNCTION DOC]
        # HELPER: available_if predicate; accesses predict_proba on every fitted sub-estimator
        # (or the unfitted estimator), letting AttributeError propagate; returns True otherwise.
        if hasattr(self, "estimators_"):
            # raise an AttributeError if `predict_proba` does not exist for
            # each estimator
            [getattr(est, "predict_proba") for est in self.estimators_]
            return True
        # raise an AttributeError if `predict_proba` does not exist for the
        # unfitted estimator
        getattr(self.estimator, "predict_proba")
        return True

    @available_if(_check_predict_proba)
    def predict_proba(self, X):
        # [FUNCTION DOC]
        # Checks fitted and returns a list with each sub-estimator's predict_proba(X), each of
        # shape (n_samples, n_classes_for_that_output).
        # KNOWN ISSUE: the docstring says a ValueError is raised when an estimator lacks
        # predict_proba; actually the method is hidden via available_if (AttributeError).
        """Return prediction probabilities for each class of each output.

        This method will raise a ``ValueError`` if any of the
        estimators do not have ``predict_proba``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data.

        Returns
        -------
        p : array of shape (n_samples, n_classes), or a list of n_outputs \
                such arrays if n_outputs > 1.
            The class probabilities of the input samples. The order of the
            classes corresponds to that in the attribute :term:`classes_`.

            .. versionchanged:: 0.19
                This function now returns a list of arrays where the length of
                the list is ``n_outputs``, and each array is (``n_samples``,
                ``n_classes``) for that particular output.
        """
        check_is_fitted(self)
        results = [estimator.predict_proba(X) for estimator in self.estimators_]
        return results

    def score(self, X, y):
        # [FUNCTION DOC]
        # Subset accuracy: fraction of samples whose predicted row equals the true row on all
        # outputs.
        # Inputs: X; y - 2-D array (n_samples, n_outputs), must expose ndim/shape.
        # Outputs: float.
        # Exceptions: ValueError if y is 1-D or its number of columns differs from the number of
        # fitted estimators; NotFittedError.
        """Return the mean accuracy on the given test data and labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.

        y : array-like of shape (n_samples, n_outputs)
            True values for X.

        Returns
        -------
        scores : float
            Mean accuracy of predicted target versus true target.
        """
        check_is_fitted(self)
        n_outputs_ = len(self.estimators_)
        if y.ndim == 1:
            raise ValueError(
                "y must have at least two dimensions for "
                "multi target classification but has only one"
            )
        if y.shape[1] != n_outputs_:
            raise ValueError(
                "The number of outputs of Y for fit {0} and"
                " score {1} should be same".format(n_outputs_, y.shape[1])
            )
        y_pred = self.predict(X)
        return np.mean(np.all(y == y_pred, axis=1))

    def __sklearn_tags__(self):
        # [FUNCTION DOC]
        # Sets the private _skip_test tag (marked FIXME) so common estimator checks are skipped.
        tags = super().__sklearn_tags__()
        # FIXME
        tags._skip_test = True
        return tags


def _available_if_base_estimator_has(attr):
    # [FUNCTION DOC]
    # HELPER: Returns an available_if decorator whose predicate is true if the unfitted
    # estimator has attr or all fitted estimators_ have it.
    """Return a function to check if `estimator` or `estimators_` has `attr`.

    Helper for Chain implementations.
    """

    def _check(self):
        # [FUNCTION DOC]
        # HELPER: Predicate above. If the estimator lacks attr and the model is unfitted,
        # accessing estimators_ raises AttributeError, which available_if treats as missing.
        return hasattr(self.estimator, attr) or all(
            hasattr(est, attr) for est in self.estimators_
        )

    return available_if(_check)


class _BaseChain(BaseEstimator, metaclass=ABCMeta):
    # [CLASS DOC]
    # HELPER: Abstract base for chain models. Fitted attributes: order_ (array of target
    # column indices), estimators_ (one clone per target), chain_method_ (classifier only)
    # and feature metadata set by validate_data.
    _parameter_constraints: dict = {
        "estimator": [HasMethods(["fit", "predict"])],
        "order": ["array-like", StrOptions({"random"}), None],
        "cv": ["cv_object", StrOptions({"prefit"})],
        "random_state": ["random_state"],
        "verbose": ["boolean"],
    }

    def __init__(
        self,
        estimator,
        *,
        order=None,
        cv=None,
        random_state=None,
        verbose=False,
    ):
        # [FUNCTION DOC]
        # Stores estimator, order, cv, random_state and verbose unchanged.
        self.estimator = estimator
        self.order = order
        self.cv = cv
        self.random_state = random_state
        self.verbose = verbose

    def _log_message(self, *, estimator_idx, n_estimators, processing_msg):
        # [FUNCTION DOC]
        # HELPER: Returns "(idx of n) msg" for _print_elapsed_time when verbose, else None.
        if not self.verbose:
            return None
        return f"({estimator_idx} of {n_estimators}) {processing_msg}"

    def _get_predictions(self, X, *, output_method):
        # [FUNCTION DOC]
        # HELPER: Runs the chain at prediction time. For each estimator in chain order, appends
        # the previous estimators' chain_method_ outputs (default "predict") as extra columns
        # to X (sparse hstack for sparse X; dok arrays converted to coo first), records the
        # chain feature from chain_method and the output from output_method (both via
        # _get_response_values, which gives positive-class 1-D values for binary
        # probabilities). Finally reorders output columns back to the original target order
        # using the inverse permutation of order_.
        # Inputs: X; output_method - response method name for the returned values.
        # Outputs: dense array (n_samples, n_outputs) passed through _align_api_if_sparse.
        # Exceptions: NotFittedError; validation errors.
        """Get predictions for each model in the chain."""
        check_is_fitted(self)
        X = validate_data(self, X, accept_sparse=True, reset=False)
        Y_output_chain = np.zeros((X.shape[0], len(self.estimators_)))
        Y_feature_chain = np.zeros((X.shape[0], len(self.estimators_)))

        # `RegressorChain` does not have a `chain_method_` parameter so we
        # default to "predict"
        chain_method = getattr(self, "chain_method_", "predict")
        hstack = sp.hstack if sp.issparse(X) else np.hstack
        for chain_idx, estimator in enumerate(self.estimators_):
            previous_predictions = Y_feature_chain[:, :chain_idx]
            # if `X` is a scipy sparse dok_array, we convert it to a sparse
            # coo_array format before hstacking, it's faster; see
            # https://github.com/scipy/scipy/issues/20060#issuecomment-1937007039:
            if sp.issparse(X) and not sp.isspmatrix(X) and X.format == "dok":
                X = sp.coo_array(X)
            X_aug = hstack((X, previous_predictions))

            feature_predictions, _ = _get_response_values(
                estimator,
                X_aug,
                response_method=chain_method,
            )
            Y_feature_chain[:, chain_idx] = feature_predictions

            output_predictions, _ = _get_response_values(
                estimator,
                X_aug,
                response_method=output_method,
            )
            Y_output_chain[:, chain_idx] = output_predictions

        inv_order = np.empty_like(self.order_)
        inv_order[self.order_] = np.arange(len(self.order_))
        Y_output = Y_output_chain[:, inv_order]

        return _align_api_if_sparse(Y_output)

    @abstractmethod
    def fit(self, X, Y, **fit_params):
        # [FUNCTION DOC]
        # Steps: validate X (sparse allowed) and multi-output Y; resolve order_ (None -> range,
        # "random" -> RNG permutation, tuple -> array, else must be a permutation of
        # range(n_outputs)); clone one estimator per output. Build augmented X_aug: with cv None
        # the true Y columns in chain order are appended (sparse result converted to CSR);
        # otherwise zero columns are appended (lil for sparse, coo used for sparse arrays).
        # Route fit params (process_routing or manual). Resolve chain_method_ via
        # _check_response_method when the subclass defines chain_method, else "predict". For
        # each chain position fit the estimator on X plus the first chain_idx extra columns
        # (timed/logged by _print_elapsed_time); when cv is set and not at the last position,
        # fill the next extra column with cross_val_predict output of an unfitted template
        # (second column for 2-D outputs).
        # Inputs: X; Y - (n_samples, n_outputs); fit_params.
        # Outputs: self.
        # Exceptions: ValueError("invalid order") when order is not a permutation.
        # Side effects: sets fitted attributes; consumes randomness; may print progress.
        # KNOWN ISSUE: random_state is documented to seed each base estimator but is only used
        # for the random order. The sparse cv=None path builds through lil before CSR while the
        # cv path keeps lil for the column assignment.
        """Fit the model to data matrix X and targets Y.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input data.

        Y : array-like of shape (n_samples, n_classes)
            The target values.

        **fit_params : dict of string -> object
            Parameters passed to the `fit` method of each step.

            .. versionadded:: 0.23

        Returns
        -------
        self : object
            Returns a fitted instance.
        """
        X, Y = validate_data(self, X, Y, multi_output=True, accept_sparse=True)

        random_state = check_random_state(self.random_state)
        self.order_ = self.order
        if isinstance(self.order_, tuple):
            self.order_ = np.array(self.order_)

        if self.order_ is None:
            self.order_ = np.array(range(Y.shape[1]))
        elif isinstance(self.order_, str):
            if self.order_ == "random":
                self.order_ = random_state.permutation(Y.shape[1])
        elif sorted(self.order_) != list(range(Y.shape[1])):
            raise ValueError("invalid order")

        self.estimators_ = [clone(self.estimator) for _ in range(Y.shape[1])]

        if self.cv is None:
            Y_pred_chain = Y[:, self.order_]
            if sp.issparse(X):
                X_aug = sp.hstack((X, Y_pred_chain), format="lil")
                X_aug = X_aug.tocsr()
            else:
                X_aug = np.hstack((X, Y_pred_chain))

        elif sp.issparse(X):
            # TODO: remove this condition check when the minimum supported scipy version
            # doesn't support sparse matrices anymore
            if not sp.isspmatrix(X):
                # if `X` is a scipy sparse dok_array, we convert it to a sparse
                # coo_array format before hstacking, it's faster; see
                # https://github.com/scipy/scipy/issues/20060#issuecomment-1937007039:
                if X.format == "dok":
                    X = sp.coo_array(X)
                # in case that `X` is a sparse array we create `Y_pred_chain` as a
                # sparse array format:
                Y_pred_chain = sp.coo_array((X.shape[0], Y.shape[1]))
            else:
                Y_pred_chain = sp.coo_matrix((X.shape[0], Y.shape[1]))
            X_aug = sp.hstack((X, Y_pred_chain), format="lil")

        else:
            Y_pred_chain = np.zeros((X.shape[0], Y.shape[1]))
            X_aug = np.hstack((X, Y_pred_chain))

        del Y_pred_chain

        if _routing_enabled():
            routed_params = process_routing(self, "fit", **fit_params)
        else:
            routed_params = _manual_routing({"estimator": {"fit": fit_params}})

        if hasattr(self, "chain_method"):
            chain_method = _check_response_method(
                self.estimator,
                self.chain_method,
            ).__name__
            self.chain_method_ = chain_method
        else:
            # `RegressorChain` does not have a `chain_method` parameter
            chain_method = "predict"

        for chain_idx, estimator in enumerate(self.estimators_):
            message = self._log_message(
                estimator_idx=chain_idx + 1,
                n_estimators=len(self.estimators_),
                processing_msg=f"Processing order {self.order_[chain_idx]}",
            )
            y = Y[:, self.order_[chain_idx]]
            with _print_elapsed_time("Chain", message):
                estimator.fit(
                    X_aug[:, : (X.shape[1] + chain_idx)],
                    y,
                    **routed_params.estimator.fit,
                )

            if self.cv is not None and chain_idx < len(self.estimators_) - 1:
                col_idx = X.shape[1] + chain_idx
                cv_result = cross_val_predict(
                    self.estimator,
                    X_aug[:, :col_idx],
                    y=y,
                    cv=self.cv,
                    method=chain_method,
                )
                # `predict_proba` output is 2D, we use only output for classes[-1]
                if cv_result.ndim > 1:
                    cv_result = cv_result[:, 1]
                if sp.issparse(X_aug):
                    X_aug[:, col_idx] = np.expand_dims(cv_result, 1)
                else:
                    X_aug[:, col_idx] = cv_result

        return self

    def predict(self, X):
        # [FUNCTION DOC]
        # Returns _get_predictions with output_method "predict".
        # KNOWN ISSUE: docstring mentions ClassifierChain but this is shared by RegressorChain.
        """Predict on the data matrix X using the ClassifierChain model.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input data.

        Returns
        -------
        Y_pred : array-like of shape (n_samples, n_classes)
            The predicted values.
        """
        return self._get_predictions(X, output_method="predict")

    def __sklearn_tags__(self):
        # [FUNCTION DOC]
        # Copies the wrapped estimator's sparse input tag.
        tags = super().__sklearn_tags__()
        tags.input_tags.sparse = get_tags(self.estimator).input_tags.sparse
        return tags


class ClassifierChain(MetaEstimatorMixin, ClassifierMixin, _BaseChain):
    # [CLASS DOC]
    # Classifier chain. Adds chain_method (method whose outputs feed later classifiers) and
    # predict_proba / predict_log_proba / decision_function. Fitted attribute classes_ is a
    # list of per-estimator class arrays.
    """A multi-label model that arranges binary classifiers into a chain.

    Each model makes a prediction in the order specified by the chain using
    all of the available features provided to the model plus the predictions
    of models that are earlier in the chain.

    For an example of how to use ``ClassifierChain`` and benefit from its
    ensemble, see
    :ref:`ClassifierChain on a yeast dataset
    <sphx_glr_auto_examples_multioutput_plot_classifier_chain_yeast.py>` example.

    Read more in the :ref:`User Guide <classifierchain>`.

    .. versionadded:: 0.19

    Parameters
    ----------
    estimator : estimator
        The base estimator from which the classifier chain is built.

    order : array-like of shape (n_outputs,) or 'random', default=None
        If `None`, the order will be determined by the order of columns in
        the label matrix Y.::

            order = [0, 1, 2, ..., Y.shape[1] - 1]

        The order of the chain can be explicitly set by providing a list of
        integers. For example, for a chain of length 5.::

            order = [1, 3, 2, 4, 0]

        means that the first model in the chain will make predictions for
        column 1 in the Y matrix, the second model will make predictions
        for column 3, etc.

        If order is `random` a random ordering will be used.

    cv : int, cross-validation generator or an iterable, default=None
        Determines whether to use cross validated predictions or true
        labels for the results of previous estimators in the chain.
        Possible inputs for cv are:

        - None, to use true labels when fitting,
        - integer, to specify the number of folds in a (Stratified)KFold,
        - :term:`CV splitter`,
        - an iterable yielding (train, test) splits as arrays of indices.

    chain_method : {'predict', 'predict_proba', 'predict_log_proba', \
            'decision_function'} or list of such str's, default='predict'

        Prediction method to be used by estimators in the chain for
        the 'prediction' features of previous estimators in the chain.

        - if `str`, name of the method;
        - if a list of `str`, provides the method names in order of
          preference. The method used corresponds to the first method in
          the list that is implemented by `base_estimator`.

        .. versionadded:: 1.5

    random_state : int, RandomState instance or None, optional (default=None)
        If ``order='random'``, determines random number generation for the
        chain order.
        In addition, it controls the random seed given at each `base_estimator`
        at each chaining iteration. Thus, it is only used when `base_estimator`
        exposes a `random_state`.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    verbose : bool, default=False
        If True, chain progress is output as each model is completed.

        .. versionadded:: 1.2

    Attributes
    ----------
    classes_ : list
        A list of arrays of length ``len(estimators_)`` containing the
        class labels for each estimator in the chain.

    estimators_ : list
        A list of clones of base_estimator.

    order_ : list
        The order of labels in the classifier chain.

    chain_method_ : str
        Prediction method used by estimators in the chain for the prediction
        features.

    n_features_in_ : int
        Number of features seen during :term:`fit`. Only defined if the
        underlying `base_estimator` exposes such an attribute when fit.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    RegressorChain : Equivalent for regression.
    MultiOutputClassifier : Classifies each output independently rather than
        chaining.

    References
    ----------
    Jesse Read, Bernhard Pfahringer, Geoff Holmes, Eibe Frank, "Classifier
    Chains for Multi-label Classification", 2009.

    Examples
    --------
    >>> from sklearn.datasets import make_multilabel_classification
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.multioutput import ClassifierChain
    >>> X, Y = make_multilabel_classification(
    ...    n_samples=12, n_classes=3, random_state=0
    ... )
    >>> X_train, X_test, Y_train, Y_test = train_test_split(
    ...    X, Y, random_state=0
    ... )
    >>> base_lr = LogisticRegression(solver='lbfgs', random_state=0)
    >>> chain = ClassifierChain(base_lr, order='random', random_state=0)
    >>> chain.fit(X_train, Y_train).predict(X_test)
    array([[1., 1., 0.],
           [1., 0., 0.],
           [0., 1., 0.]])
    >>> chain.predict_proba(X_test)
    array([[0.8387, 0.9431, 0.4576],
           [0.8878, 0.3684, 0.2640],
           [0.0321, 0.9935, 0.0626]])
    """

    _parameter_constraints: dict = {
        **_BaseChain._parameter_constraints,
        "chain_method": [
            list,
            tuple,
            StrOptions(
                {"predict", "predict_proba", "predict_log_proba", "decision_function"}
            ),
        ],
    }

    def __init__(
        self,
        estimator,
        *,
        order=None,
        cv=None,
        chain_method="predict",
        random_state=None,
        verbose=False,
    ):
        # [FUNCTION DOC]
        # Forwards common parameters to _BaseChain and stores chain_method.
        super().__init__(
            estimator,
            order=order,
            cv=cv,
            random_state=random_state,
            verbose=verbose,
        )
        self.chain_method = chain_method

    @_fit_context(
        # ClassifierChain.base_estimator is not validated yet
        prefer_skip_nested_validation=False
    )
    def fit(self, X, Y, **fit_params):
        # [FUNCTION DOC]
        # Rejects unrouted metadata, runs the base fit, then sets classes_ from each estimator.
        # Outputs: self.
        # Exceptions: ValueError if metadata is passed without routing enabled.
        """Fit the model to data matrix X and targets Y.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input data.

        Y : array-like of shape (n_samples, n_classes)
            The target values.

        **fit_params : dict of string -> object
            Parameters passed to the `fit` method of each step.

            Only available if `enable_metadata_routing=True`. See the
            :ref:`User Guide <metadata_routing>`.

            .. versionadded:: 1.3

        Returns
        -------
        self : object
            Class instance.
        """
        _raise_for_params(fit_params, self, "fit")

        super().fit(X, Y, **fit_params)
        self.classes_ = [estimator.classes_ for estimator in self.estimators_]
        return self

    @_available_if_base_estimator_has("predict_proba")
    def predict_proba(self, X):
        # [FUNCTION DOC]
        # Returns _get_predictions with output_method "predict_proba": positive-class
        # probability per output, shape (n_samples, n_outputs).
        """Predict probability estimates.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input data.

        Returns
        -------
        Y_prob : array-like of shape (n_samples, n_classes)
            The predicted probabilities.
        """
        return self._get_predictions(X, output_method="predict_proba")

    def predict_log_proba(self, X):
        # [FUNCTION DOC]
        # Natural log of predict_proba (zeros produce -inf with a numpy warning).
        # KNOWN ISSUE: not guarded by available_if, so on an estimator without predict_proba it
        # fails with AttributeError at call time instead of being hidden.
        """Predict logarithm of probability estimates.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input data.

        Returns
        -------
        Y_log_prob : array-like of shape (n_samples, n_classes)
            The predicted logarithm of the probabilities.
        """
        return np.log(self.predict_proba(X))

    @_available_if_base_estimator_has("decision_function")
    def decision_function(self, X):
        # [FUNCTION DOC]
        # Returns _get_predictions with output_method "decision_function".
        """Evaluate the decision_function of the models in the chain.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data.

        Returns
        -------
        Y_decision : array-like of shape (n_samples, n_classes)
            Returns the decision function of the sample for each model
            in the chain.
        """
        return self._get_predictions(X, output_method="decision_function")

    def get_metadata_routing(self):
        # [FUNCTION DOC]
        # MetadataRouter routing fit->fit to the wrapped estimator.
        """Get metadata routing of this object.

        Please check :ref:`User Guide <metadata_routing>` on how the routing
        mechanism works.

        .. versionadded:: 1.3

        Returns
        -------
        routing : MetadataRouter
            A :class:`~sklearn.utils.metadata_routing.MetadataRouter` encapsulating
            routing information.
        """

        router = MetadataRouter(owner=self).add(
            estimator=self.estimator,
            method_mapping=MethodMapping().add(caller="fit", callee="fit"),
        )
        return router

    def __sklearn_tags__(self):
        # [FUNCTION DOC]
        # Sets _skip_test (FIXME) and marks targets multi-output, not single-output.
        tags = super().__sklearn_tags__()
        # FIXME
        tags._skip_test = True
        tags.target_tags.single_output = False
        tags.target_tags.multi_output = True
        return tags


class RegressorChain(MetaEstimatorMixin, RegressorMixin, _BaseChain):
    # [CLASS DOC]
    # Regressor chain; uses "predict" outputs as chain features.
    """A multi-label model that arranges regressions into a chain.

    Each model makes a prediction in the order specified by the chain using
    all of the available features provided to the model plus the predictions
    of models that are earlier in the chain.

    Read more in the :ref:`User Guide <regressorchain>`.

    .. versionadded:: 0.20

    Parameters
    ----------
    estimator : estimator
        The base estimator from which the regressor chain is built.

    order : array-like of shape (n_outputs,) or 'random', default=None
        If `None`, the order will be determined by the order of columns in
        the label matrix Y.::

            order = [0, 1, 2, ..., Y.shape[1] - 1]

        The order of the chain can be explicitly set by providing a list of
        integers. For example, for a chain of length 5.::

            order = [1, 3, 2, 4, 0]

        means that the first model in the chain will make predictions for
        column 1 in the Y matrix, the second model will make predictions
        for column 3, etc.

        If order is 'random' a random ordering will be used.

    cv : int, cross-validation generator or an iterable, default=None
        Determines whether to use cross validated predictions or true
        labels for the results of previous estimators in the chain.
        Possible inputs for cv are:

        - None, to use true labels when fitting,
        - integer, to specify the number of folds in a (Stratified)KFold,
        - :term:`CV splitter`,
        - an iterable yielding (train, test) splits as arrays of indices.

    random_state : int, RandomState instance or None, optional (default=None)
        If ``order='random'``, determines random number generation for the
        chain order.
        In addition, it controls the random seed given at each `base_estimator`
        at each chaining iteration. Thus, it is only used when `base_estimator`
        exposes a `random_state`.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    verbose : bool, default=False
        If True, chain progress is output as each model is completed.

        .. versionadded:: 1.2

    Attributes
    ----------
    estimators_ : list
        A list of clones of base_estimator.

    order_ : list
        The order of labels in the regressor chain.

    n_features_in_ : int
        Number of features seen during :term:`fit`. Only defined if the
        underlying `base_estimator` exposes such an attribute when fit.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    ClassifierChain : Equivalent for classification.
    MultiOutputRegressor : Learns each output independently rather than
        chaining.

    Examples
    --------
    >>> from sklearn.multioutput import RegressorChain
    >>> from sklearn.linear_model import LogisticRegression
    >>> logreg = LogisticRegression(solver='lbfgs')
    >>> X, Y = [[1, 0], [0, 1], [1, 1]], [[0, 2], [1, 1], [2, 0]]
    >>> chain = RegressorChain(logreg, order=[0, 1]).fit(X, Y)
    >>> chain.predict(X)
    array([[0., 2.],
           [1., 1.],
           [2., 0.]])
    """

    @_fit_context(
        # RegressorChain.base_estimator is not validated yet
        prefer_skip_nested_validation=False
    )
    def fit(self, X, Y, **fit_params):
        # [FUNCTION DOC]
        # Delegates to the base fit and returns self.
        # KNOWN ISSUE: unlike ClassifierChain.fit, does not call _raise_for_params, so metadata
        # passed without routing is forwarded manually instead of raising.
        """Fit the model to data matrix X and targets Y.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input data.

        Y : array-like of shape (n_samples, n_classes)
            The target values.

        **fit_params : dict of string -> object
            Parameters passed to the `fit` method at each step
            of the regressor chain.

            .. versionadded:: 0.23

        Returns
        -------
        self : object
            Returns a fitted instance.
        """
        super().fit(X, Y, **fit_params)
        return self

    def get_metadata_routing(self):
        # [FUNCTION DOC]
        # MetadataRouter routing fit->fit to the wrapped estimator.
        """Get metadata routing of this object.

        Please check :ref:`User Guide <metadata_routing>` on how the routing
        mechanism works.

        .. versionadded:: 1.3

        Returns
        -------
        routing : MetadataRouter
            A :class:`~sklearn.utils.metadata_routing.MetadataRouter` encapsulating
            routing information.
        """

        router = MetadataRouter(owner=self).add(
            estimator=self.estimator,
            method_mapping=MethodMapping().add(caller="fit", callee="fit"),
        )
        return router

    def __sklearn_tags__(self):
        # [FUNCTION DOC]
        # Marks targets as multi-output, not single-output.
        tags = super().__sklearn_tags__()
        tags.target_tags.single_output = False
        tags.target_tags.multi_output = True
        return tags
