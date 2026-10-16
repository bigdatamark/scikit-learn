# [MODULE DOC]
# Stacked generalization ensembles: StackingClassifier and StackingRegressor, built on
# the abstract _BaseStacking. Base estimators are fitted on the full data (or used
# as-is when cv="prefit"); the final estimator is fitted on cross-validated base
# predictions (optionally concatenated with X when passthrough=True). Supports
# metadata routing for fit (base estimators) and predict (final estimator).
"""Stacking classifier and regressor."""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

from abc import ABCMeta, abstractmethod
from copy import deepcopy
from numbers import Integral

import numpy as np
import scipy.sparse as sparse

from sklearn.base import (
    ClassifierMixin,
    RegressorMixin,
    TransformerMixin,
    _fit_context,
    clone,
    is_classifier,
    is_regressor,
)
from sklearn.ensemble._base import _BaseHeterogeneousEnsemble, _fit_single_estimator
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.model_selection import check_cv, cross_val_predict
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import Bunch
from sklearn.utils._param_validation import HasMethods, StrOptions
from sklearn.utils._repr_html.estimator import _VisualBlock
from sklearn.utils.metadata_routing import (
    MetadataRouter,
    MethodMapping,
    _manual_routing,
    _raise_for_params,
    _routing_enabled,
    process_routing,
)
from sklearn.utils.metaestimators import available_if
from sklearn.utils.multiclass import check_classification_targets, type_of_target
from sklearn.utils.parallel import Parallel, delayed
from sklearn.utils.validation import (
    _check_feature_names_in,
    _check_response_method,
    _estimator_has,
    check_is_fitted,
    column_or_1d,
)


class _BaseStacking(TransformerMixin, _BaseHeterogeneousEnsemble, metaclass=ABCMeta):
    # [CLASS DOC]
    # Abstract base for stacking meta-estimators. Combines TransformerMixin (transform
    # returns the stacked base predictions) with _BaseHeterogeneousEnsemble (estimators
    # list of (name, estimator or "drop")). Subclasses must provide __init__,
    # _validate_final_estimator and transform. Fitted attributes: estimators_,
    # named_estimators_, stack_method_, final_estimator_, _n_feature_outs and possibly
    # feature_names_in_.
    """Base class for stacking method."""

    _parameter_constraints: dict = {
        "estimators": [list],
        "final_estimator": [None, HasMethods("fit")],
        "cv": ["cv_object", StrOptions({"prefit"})],
        "n_jobs": [None, Integral],
        "passthrough": ["boolean"],
        "verbose": ["verbose"],
    }

    @abstractmethod
    def __init__(
        self,
        estimators,
        final_estimator=None,
        *,
        cv=None,
        stack_method="auto",
        n_jobs=None,
        verbose=0,
        passthrough=False,
    ):
        # [FUNCTION DOC]
        # Abstract constructor: stores estimators (via parent), final_estimator, cv,
        # stack_method, n_jobs, verbose and passthrough without validation.
        super().__init__(estimators=estimators)
        self.final_estimator = final_estimator
        self.cv = cv
        self.stack_method = stack_method
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.passthrough = passthrough

    def _clone_final_estimator(self, default):
        # [FUNCTION DOC]
        # HELPER: Sets self.final_estimator_ to a clone of final_estimator, or a clone of
        # default when final_estimator is None.
        # Side effects: sets final_estimator_.
        if self.final_estimator is not None:
            self.final_estimator_ = clone(self.final_estimator)
        else:
            self.final_estimator_ = clone(default)

    def _concatenate_predictions(self, X, predictions):
        # [FUNCTION DOC]
        # HELPER: Builds the meta-feature matrix from a list of base predictions (one entry per
        # non-dropped estimator, in order).
        # Behavior: for each entry: a list of 2D arrays (multilabel predict_proba) contributes
        # each array without its first column; a 1D array becomes a single column; with
        # predict_proba in binary classification (2 classes) the first column is dropped;
        # otherwise the 2D array is kept. Records the number of columns contributed by each
        # block in _n_feature_outs (one entry per block, so a list entry yields several). If
        # passthrough, appends X; with sparse X the result is a sparse hstack in X's format,
        # otherwise a dense np.hstack.
        # Inputs: X - original input (array or sparse); predictions - list of arrays/lists.
        # Outputs: ndarray or sparse matrix (n_samples, n_meta_features).
        # Side effects: sets self._n_feature_outs.
        # KNOWN ISSUE: for list predictions _n_feature_outs gets one entry per target while
        # get_feature_names_out zips it with estimator names, so names misalign for
        # multilabel list outputs.
        # KNOWN ISSUE: with passthrough and dense X but a sparse prediction block, np.hstack is
        # used, which may fail; only X's sparsity is checked.
        """Concatenate the predictions of each first layer learner and
        possibly the input dataset `X`.

        If `X` is sparse and `self.passthrough` is False, the output of
        `transform` will be dense (the predictions). If `X` is sparse
        and `self.passthrough` is True, the output of `transform` will
        be sparse.

        This helper is in charge of ensuring the predictions are 2D arrays and
        it will drop one of the probability column when using probabilities
        in the binary case. Indeed, the p(y|c=0) = 1 - p(y|c=1)

        When `y` type is `"multilabel-indicator"`` and the method used is
        `predict_proba`, `preds` can be either a `ndarray` of shape
        `(n_samples, n_class)` or for some estimators a list of `ndarray`.
        This function will drop one of the probability column in this situation as well.
        """
        X_meta = []
        for est_idx, preds in enumerate(predictions):
            if isinstance(preds, list):
                # `preds` is here a list of `n_targets` 2D ndarrays of
                # `n_classes` columns. The k-th column contains the
                # probabilities of the samples belonging the k-th class.
                #
                # Since those probabilities must sum to one for each sample,
                # we can work with probabilities of `n_classes - 1` classes.
                # Hence we drop the first column.
                for pred in preds:
                    X_meta.append(pred[:, 1:])
            elif preds.ndim == 1:
                # Some estimator return a 1D array for predictions
                # which must be 2-dimensional arrays.
                X_meta.append(preds.reshape(-1, 1))
            elif (
                self.stack_method_[est_idx] == "predict_proba"
                and len(self.classes_) == 2
            ):
                # Remove the first column when using probabilities in
                # binary classification because both features `preds` are perfectly
                # collinear.
                X_meta.append(preds[:, 1:])
            else:
                X_meta.append(preds)

        self._n_feature_outs = [pred.shape[1] for pred in X_meta]
        if self.passthrough:
            X_meta.append(X)
            if sparse.issparse(X):
                return sparse.hstack(X_meta, format=X.format)

        return np.hstack(X_meta)

    @staticmethod
    def _method_name(name, estimator, method):
        # [FUNCTION DOC]
        # HELPER: Resolves the response method for one base estimator. Returns None for "drop";
        # "auto" tries predict_proba, decision_function, then predict in order; otherwise uses
        # the given method name.
        # Inputs: name - str; estimator - estimator or "drop"; method - str.
        # Outputs: str method name or None.
        # Exceptions: ValueError if the estimator implements none of the requested methods.
        if estimator == "drop":
            return None
        if method == "auto":
            method = ["predict_proba", "decision_function", "predict"]
        try:
            method_name = _check_response_method(estimator, method).__name__
        except AttributeError as e:
            raise ValueError(
                f"Underlying estimator {name} does not implement the method {method}."
            ) from e

        return method_name

    @_fit_context(
        # estimators in Stacking*.estimators are not validated yet
        prefer_skip_nested_validation=False
    )
    def fit(self, X, y, **fit_params):
        # [FUNCTION DOC]
        # Fits the stack.
        # Behavior: validates estimators and final estimator (subclass hooks); resolves
        # per-estimator fit metadata (process_routing when routing is enabled, else only
        # sample_weight is forwarded to every base estimator). If cv == "prefit", checks each
        # non-dropped estimator is fitted and uses it directly; otherwise fits clones of every
        # non-dropped estimator on the full data in parallel. Builds named_estimators_ (with
        # "drop" placeholders) and copies feature_names_in_ from fitted estimators that have
        # it. Resolves stack_method_. Obtains meta-features: prefit estimators' predictions on
        # X, or cross_val_predict out-of-fold predictions using check_cv (a cv object whose
        # random_state is None gets a fresh RandomState so every estimator sees the same
        # folds via deepcopy). Filters stack_method_ to non-dropped estimators, concatenates
        # predictions and fits final_estimator_ on them with fit_params.
        # Inputs: X - (n_samples, n_features) array or sparse; y - (n_samples,) (already
        # encoded by subclasses); fit_params - metadata such as sample_weight.
        # Outputs: self.
        # Exceptions: ValueError from validation hooks or _method_name; NotFittedError for
        # unfitted prefit estimators; parameter validation errors from _fit_context.
        # Side effects: sets estimators_, named_estimators_, stack_method_, final_estimator_,
        # _n_feature_outs, feature_names_in_; may mutate the random_state of the cv splitter
        # object returned by check_cv (which can be the user's own object).
        # KNOWN ISSUE: the final estimator receives the raw fit_params rather than parameters
        # routed to it, so with metadata routing enabled it may receive metadata it did not
        # request.
        """Fit the estimators.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vectors, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        y : array-like of shape (n_samples,)
            Target values.

        **fit_params : dict
            Dict of metadata, potentially containing sample_weight as a
            key-value pair. If sample_weight is not present, then samples are
            equally weighted. Note that sample_weight is supported only if all
            underlying estimators support sample weights.

            .. versionadded:: 1.6

        Returns
        -------
        self : object
        """
        # all_estimators contains all estimators, the one to be fitted and the
        # 'drop' string.
        names, all_estimators = self._validate_estimators()
        self._validate_final_estimator()

        stack_method = [self.stack_method] * len(all_estimators)

        if _routing_enabled():
            routed_params = process_routing(self, "fit", **fit_params)
        else:
            sw = (
                {"sample_weight": fit_params["sample_weight"]}
                if "sample_weight" in fit_params
                else {}
            )
            routed_params = _manual_routing({name: {"fit": sw} for name in names})

        if self.cv == "prefit":
            self.estimators_ = []
            for estimator in all_estimators:
                if estimator != "drop":
                    check_is_fitted(estimator)
                    self.estimators_.append(estimator)
        else:
            # Fit the base estimators on the whole training data. Those
            # base estimators will be used in transform, predict, and
            # predict_proba. They are exposed publicly.
            self.estimators_ = Parallel(n_jobs=self.n_jobs)(
                delayed(_fit_single_estimator)(
                    clone(est), X, y, routed_params[name]["fit"]
                )
                for name, est in zip(names, all_estimators)
                if est != "drop"
            )

        self.named_estimators_ = Bunch()
        est_fitted_idx = 0
        for name_est, org_est in zip(names, all_estimators):
            if org_est != "drop":
                current_estimator = self.estimators_[est_fitted_idx]
                self.named_estimators_[name_est] = current_estimator
                est_fitted_idx += 1
                if hasattr(current_estimator, "feature_names_in_"):
                    self.feature_names_in_ = current_estimator.feature_names_in_
            else:
                self.named_estimators_[name_est] = "drop"

        self.stack_method_ = [
            self._method_name(name, est, meth)
            for name, est, meth in zip(names, all_estimators, stack_method)
        ]

        if self.cv == "prefit":
            # Generate predictions from prefit models
            predictions = [
                getattr(estimator, predict_method)(X)
                for estimator, predict_method in zip(all_estimators, self.stack_method_)
                if estimator != "drop"
            ]
        else:
            # To train the meta-classifier using the most data as possible, we use
            # a cross-validation to obtain the output of the stacked estimators.
            # To ensure that the data provided to each estimator are the same,
            # we need to set the random state of the cv if there is one and we
            # need to take a copy.
            cv = check_cv(self.cv, y=y, classifier=is_classifier(self))
            if hasattr(cv, "random_state") and cv.random_state is None:
                cv.random_state = np.random.RandomState()

            predictions = Parallel(n_jobs=self.n_jobs)(
                delayed(cross_val_predict)(
                    clone(est),
                    X,
                    y,
                    cv=deepcopy(cv),
                    method=meth,
                    n_jobs=self.n_jobs,
                    params=routed_params[name]["fit"],
                    verbose=self.verbose,
                )
                for name, est, meth in zip(names, all_estimators, self.stack_method_)
                if est != "drop"
            )

        # Only not None or not 'drop' estimators will be used in transform.
        # Remove the None from the method as well.
        self.stack_method_ = [
            meth
            for (meth, est) in zip(self.stack_method_, all_estimators)
            if est != "drop"
        ]

        X_meta = self._concatenate_predictions(X, predictions)
        _fit_single_estimator(self.final_estimator_, X_meta, y, fit_params=fit_params)

        return self

    @property
    def n_features_in_(self):
        # [FUNCTION DOC]
        # Property: number of input features, taken from the first fitted base estimator.
        # Exceptions: AttributeError (wrapping NotFittedError) when not fitted.
        """Number of features seen during :term:`fit`."""
        try:
            check_is_fitted(self)
        except NotFittedError as nfe:
            raise AttributeError(
                f"{self.__class__.__name__} object has no attribute n_features_in_"
            ) from nfe
        return self.estimators_[0].n_features_in_

    def _transform(self, X):
        # [FUNCTION DOC]
        # HELPER: Checks fitted state, computes each fitted base estimator's stack_method_
        # output on X and concatenates them (plus X if passthrough) via
        # _concatenate_predictions.
        # Side effects: re-sets _n_feature_outs.
        # KNOWN ISSUE: the "est != 'drop'" filter is dead, since estimators_ never contains
        # "drop".
        """Concatenate and return the predictions of the estimators."""
        check_is_fitted(self)
        predictions = [
            getattr(est, meth)(X)
            for est, meth in zip(self.estimators_, self.stack_method_)
            if est != "drop"
        ]
        return self._concatenate_predictions(X, predictions)

    def get_feature_names_out(self, input_features=None):
        # [FUNCTION DOC]
        # Returns output feature names: for each non-dropped estimator, "<classname>_<name>" if
        # it contributes one column, else "<classname>_<name><i>" for each column; followed by
        # validated input feature names when passthrough is True.
        # Inputs: input_features - array-like of str or None.
        # Outputs: ndarray of str (object dtype).
        # Exceptions: NotFittedError if not fitted; ValueError for inconsistent
        # input_features.
        """Get output feature names for transformation.

        Parameters
        ----------
        input_features : array-like of str or None, default=None
            Input features. The input feature names are only used when `passthrough` is
            `True`.

            - If `input_features` is `None`, then `feature_names_in_` is
              used as feature names in. If `feature_names_in_` is not defined,
              then names are generated: `[x0, x1, ..., x(n_features_in_ - 1)]`.
            - If `input_features` is an array-like, then `input_features` must
              match `feature_names_in_` if `feature_names_in_` is defined.

            If `passthrough` is `False`, then only the names of `estimators` are used
            to generate the output feature names.

        Returns
        -------
        feature_names_out : ndarray of str objects
            Transformed feature names.
        """
        check_is_fitted(self, "n_features_in_")
        input_features = _check_feature_names_in(
            self, input_features, generate_names=self.passthrough
        )

        class_name = self.__class__.__name__.lower()
        non_dropped_estimators = (
            name for name, est in self.estimators if est != "drop"
        )
        meta_names = []
        for est, n_features_out in zip(non_dropped_estimators, self._n_feature_outs):
            if n_features_out == 1:
                meta_names.append(f"{class_name}_{est}")
            else:
                meta_names.extend(
                    f"{class_name}_{est}{i}" for i in range(n_features_out)
                )

        if self.passthrough:
            return np.concatenate((meta_names, input_features))

        return np.asarray(meta_names, dtype=object)

    @available_if(
        _estimator_has("predict", delegates=("final_estimator_", "final_estimator"))
    )
    def predict(self, X, **predict_params):
        # [FUNCTION DOC]
        # Checks fitted state and returns final_estimator_.predict on transform(X), forwarding
        # predict_params. Available only if the final estimator has predict.
        """Predict target for X.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vectors, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        **predict_params : dict of str -> obj
            Parameters to the `predict` called by the `final_estimator`. Note
            that this may be used to return uncertainties from some estimators
            with `return_std` or `return_cov`. Be aware that it will only
            account for uncertainty in the final estimator.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,) or (n_samples, n_output)
            Predicted targets.
        """

        check_is_fitted(self)
        return self.final_estimator_.predict(self.transform(X), **predict_params)

    def _sk_visual_block_with_final_estimator(self, final_estimator):
        # [FUNCTION DOC]
        # HELPER: Builds the HTML repr block: a parallel block of the base estimators followed
        # serially by a parallel block labeled "final_estimator".
        names, estimators = zip(*self.estimators)
        parallel = _VisualBlock("parallel", estimators, names=names, dash_wrapped=False)

        # final estimator is wrapped in a parallel block to show the label:
        # 'final_estimator' in the html repr
        final_block = _VisualBlock(
            "parallel", [final_estimator], names=["final_estimator"], dash_wrapped=False
        )
        return _VisualBlock("serial", (parallel, final_block), dash_wrapped=False)

    def get_metadata_routing(self):
        # [FUNCTION DOC]
        # Returns a MetadataRouter routing caller fit to each base estimator's fit, and caller
        # predict to predict of final_estimator_ (fitted) or final_estimator (unfitted).
        """Get metadata routing of this object.

        Please check :ref:`User Guide <metadata_routing>` on how the routing
        mechanism works.

        .. versionadded:: 1.6

        Returns
        -------
        routing : MetadataRouter
            A :class:`~sklearn.utils.metadata_routing.MetadataRouter` encapsulating
            routing information.
        """
        router = MetadataRouter(owner=self)

        # `self.estimators` is a list of (name, est) tuples
        for name, estimator in self.estimators:
            router.add(
                **{name: estimator},
                method_mapping=MethodMapping().add(callee="fit", caller="fit"),
            )

        try:
            final_estimator_ = self.final_estimator_
        except AttributeError:
            final_estimator_ = self.final_estimator

        router.add(
            final_estimator_=final_estimator_,
            method_mapping=MethodMapping().add(caller="predict", callee="predict"),
        )

        return router


class StackingClassifier(ClassifierMixin, _BaseStacking):
    # [CLASS DOC]
    # Stacking with a final classifier (default LogisticRegression). Labels are encoded
    # with LabelEncoder (one per column for multilabel-indicator targets) before fitting
    # and decoded in predict. stack_method may be "auto", "predict_proba",
    # "decision_function" or "predict". Base estimators may be regressors.
    """Stack of estimators with a final classifier.

    Stacked generalization consists in stacking the output of individual
    estimator and use a classifier to compute the final prediction. Stacking
    allows to use the strength of each individual estimator by using their
    output as input of a final estimator.

    Note that `estimators_` are fitted on the full `X` while `final_estimator_`
    is trained using cross-validated predictions of the base estimators using
    `cross_val_predict`.

    Read more in the :ref:`User Guide <stacking>`.

    .. versionadded:: 0.22

    Parameters
    ----------
    estimators : list of (str, estimator)
        Base estimators which will be stacked together. Each element of the
        list is defined as a tuple of string (i.e. name) and an estimator
        instance. An estimator can be set to 'drop' using `set_params`.

        The type of estimator is generally expected to be a classifier.
        However, one can pass a regressor for some use case (e.g. ordinal
        regression).

    final_estimator : estimator, default=None
        A classifier which will be used to combine the base estimators.
        The default classifier is a
        :class:`~sklearn.linear_model.LogisticRegression`.

    cv : int, cross-validation generator, iterable, or "prefit", default=None
        Determines the cross-validation splitting strategy used in
        `cross_val_predict` to train `final_estimator`. Possible inputs for
        cv are:

        * None, to use the default 5-fold cross validation,
        * integer, to specify the number of folds in a (Stratified) KFold,
        * An object to be used as a cross-validation generator,
        * An iterable yielding train, test splits,
        * `"prefit"`, to assume the `estimators` are prefit. In this case, the
          estimators will not be refitted.

        For integer/None inputs, if the estimator is a classifier and y is
        either binary or multiclass,
        :class:`~sklearn.model_selection.StratifiedKFold` is used.
        In all other cases, :class:`~sklearn.model_selection.KFold` is used.
        These splitters are instantiated with `shuffle=False` so the splits
        will be the same across calls.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.

        If "prefit" is passed, it is assumed that all `estimators` have
        been fitted already. The `final_estimator_` is trained on the `estimators`
        predictions on the full training set and are **not** cross validated
        predictions. Please note that if the models have been trained on the same
        data to train the stacking model, there is a very high risk of overfitting.

        .. versionadded:: 1.1
            The 'prefit' option was added in 1.1

        .. note::
           A larger number of split will provide no benefits if the number
           of training samples is large enough. Indeed, the training time
           will increase. ``cv`` is not used for model evaluation but for
           prediction.

    stack_method : {'auto', 'predict_proba', 'decision_function', 'predict'}, \
            default='auto'
        Methods called for each base estimator. It can be:

        * if 'auto', it will try to invoke, for each estimator,
          `'predict_proba'`, `'decision_function'` or `'predict'` in that
          order.
        * otherwise, one of `'predict_proba'`, `'decision_function'` or
          `'predict'`. If the method is not implemented by the estimator, it
          will raise an error.

    n_jobs : int, default=None
        The number of jobs to run in parallel for `fit` of all `estimators`.
        `None` means 1 unless in a `joblib.parallel_backend` context. -1 means
        using all processors. See :term:`Glossary <n_jobs>` for more details.

    passthrough : bool, default=False
        When False, only the predictions of estimators will be used as
        training data for `final_estimator`. When True, the
        `final_estimator` is trained on the predictions as well as the
        original training data.

    verbose : int, default=0
        Verbosity level.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,) or list of ndarray if `y` \
        is of type `"multilabel-indicator"`.
        Class labels.

    estimators_ : list of estimators
        The elements of the `estimators` parameter, having been fitted on the
        training data. If an estimator has been set to `'drop'`, it
        will not appear in `estimators_`. When `cv="prefit"`, `estimators_`
        is set to `estimators` and is not fitted again.

    named_estimators_ : :class:`~sklearn.utils.Bunch`
        Attribute to access any fitted sub-estimators by name.

    n_features_in_ : int
        Number of features seen during :term:`fit`. Only defined if the
        underlying estimator exposes such an attribute when fit.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Only defined if the
        underlying estimators expose such an attribute when fit.

        .. versionadded:: 1.0

    final_estimator_ : estimator
        The classifier fit on the output of `estimators_` and responsible for
        final predictions.

    stack_method_ : list of str
        The method used by each base estimator.

    See Also
    --------
    StackingRegressor : Stack of estimators with a final regressor.

    Notes
    -----
    When `predict_proba` is used by each estimator (i.e. most of the time for
    `stack_method='auto'` or specifically for `stack_method='predict_proba'`),
    the first column predicted by each estimator will be dropped in the case
    of a binary classification problem. Indeed, both feature will be perfectly
    collinear.

    In some cases (e.g. ordinal regression), one can pass regressors as the
    first layer of the :class:`StackingClassifier`. However, note that `y` will
    be internally encoded in a numerically increasing order or lexicographic
    order. If this ordering is not adequate, one should manually numerically
    encode the classes in the desired order.

    References
    ----------
    .. [1] Wolpert, David H. "Stacked generalization." Neural networks 5.2
       (1992): 241-259.

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.svm import LinearSVC
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.preprocessing import StandardScaler
    >>> from sklearn.pipeline import make_pipeline
    >>> from sklearn.ensemble import StackingClassifier
    >>> X, y = load_iris(return_X_y=True)
    >>> estimators = [
    ...     ('rf', RandomForestClassifier(n_estimators=10, random_state=42)),
    ...     ('svr', make_pipeline(StandardScaler(),
    ...                           LinearSVC(random_state=42)))
    ... ]
    >>> clf = StackingClassifier(
    ...     estimators=estimators, final_estimator=LogisticRegression()
    ... )
    >>> from sklearn.model_selection import train_test_split
    >>> X_train, X_test, y_train, y_test = train_test_split(
    ...     X, y, stratify=y, random_state=42
    ... )
    >>> clf.fit(X_train, y_train).score(X_test, y_test)
    0.9...
    """

    _parameter_constraints: dict = {
        **_BaseStacking._parameter_constraints,
        "stack_method": [
            StrOptions({"auto", "predict_proba", "decision_function", "predict"})
        ],
    }

    def __init__(
        self,
        estimators,
        final_estimator=None,
        *,
        cv=None,
        stack_method="auto",
        n_jobs=None,
        passthrough=False,
        verbose=0,
    ):
        # [FUNCTION DOC]
        # Stores parameters by forwarding all of them to _BaseStacking.__init__.
        super().__init__(
            estimators=estimators,
            final_estimator=final_estimator,
            cv=cv,
            stack_method=stack_method,
            n_jobs=n_jobs,
            passthrough=passthrough,
            verbose=verbose,
        )

    def _validate_final_estimator(self):
        # [FUNCTION DOC]
        # HELPER: Clones final_estimator (default LogisticRegression) into final_estimator_ and
        # raises ValueError if it is not a classifier.
        self._clone_final_estimator(default=LogisticRegression())
        if not is_classifier(self.final_estimator_):
            raise ValueError(
                "'final_estimator' parameter should be a classifier. Got {}".format(
                    self.final_estimator_
                )
            )

    def _validate_estimators(self):
        # [FUNCTION DOC]
        # HELPER: Lenient version of the parent check: estimators must be a non-empty list,
        # names valid (_validate_names), and at least one entry not "drop"; estimator types are
        # not checked. Returns (names, estimators) tuples.
        # Exceptions: ValueError for empty list or all dropped; errors from _validate_names.
        """Overload the method of `_BaseHeterogeneousEnsemble` to be more
        lenient towards the type of `estimators`.

        Regressors can be accepted for some cases such as ordinal regression.
        """
        if len(self.estimators) == 0:
            raise ValueError(
                "Invalid 'estimators' attribute, 'estimators' should be a "
                "non-empty list of (string, estimator) tuples."
            )
        names, estimators = zip(*self.estimators)
        self._validate_names(names)

        has_estimator = any(est != "drop" for est in estimators)
        if not has_estimator:
            raise ValueError(
                "All estimators are dropped. At least one is required "
                "to be an estimator."
            )

        return names, estimators

    def fit(self, X, y, **fit_params):
        # [FUNCTION DOC]
        # Rejects unsupported metadata when routing is disabled (only sample_weight allowed),
        # checks classification targets, label-encodes y (per column for multilabel-indicator,
        # setting _label_encoder to a list and classes_ to a list of arrays), and delegates to
        # _BaseStacking.fit with the encoded targets.
        # Outputs: self. Side effects: sets _label_encoder and classes_ plus base fit
        # attributes.
        # Exceptions: ValueError for invalid targets or unsupported params.
        """Fit the estimators.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vectors, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        y : array-like of shape (n_samples,)
            Target values. Note that `y` will be internally encoded in
            numerically increasing order or lexicographic order. If the order
            matter (e.g. for ordinal regression), one should numerically encode
            the target `y` before calling :term:`fit`.

        **fit_params : dict
            Parameters to pass to the underlying estimators.

            .. versionadded:: 1.6

                Only available if `enable_metadata_routing=True`, which can be
                set by using ``sklearn.set_config(enable_metadata_routing=True)``.
                See :ref:`Metadata Routing User Guide <metadata_routing>` for
                more details.

        Returns
        -------
        self : object
            Returns a fitted instance of estimator.
        """
        _raise_for_params(fit_params, self, "fit", allow=["sample_weight"])

        check_classification_targets(y)
        if type_of_target(y) == "multilabel-indicator":
            self._label_encoder = [LabelEncoder().fit(yk) for yk in y.T]
            self.classes_ = [le.classes_ for le in self._label_encoder]
            y_encoded = np.array(
                [
                    self._label_encoder[target_idx].transform(target)
                    for target_idx, target in enumerate(y.T)
                ]
            ).T
        else:
            self._label_encoder = LabelEncoder().fit(y)
            self.classes_ = self._label_encoder.classes_
            y_encoded = self._label_encoder.transform(y)

        return super().fit(X, y_encoded, **fit_params)

    @available_if(
        _estimator_has("predict", delegates=("final_estimator_", "final_estimator"))
    )
    def predict(self, X, **predict_params):
        # [FUNCTION DOC]
        # Routes predict_params to the final estimator (process_routing or passed through
        # directly), calls the base predict, and inverse-transforms the encoded predictions
        # to original labels (per column for multilabel).
        # Outputs: ndarray (n_samples,) or (n_samples, n_outputs).
        """Predict target for X.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vectors, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        **predict_params : dict of str -> obj
            Parameters to the `predict` called by the `final_estimator`. Note
            that this may be used to return uncertainties from some estimators
            with `return_std` or `return_cov`. Be aware that it will only
            account for uncertainty in the final estimator.

            - If `enable_metadata_routing=False` (default):
              Parameters directly passed to the `predict` method of the
              `final_estimator`.

            - If `enable_metadata_routing=True`: Parameters safely routed to
              the `predict` method of the `final_estimator`. See :ref:`Metadata
              Routing User Guide <metadata_routing>` for more details.

            .. versionchanged:: 1.6
                `**predict_params` can be routed via metadata routing API.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,) or (n_samples, n_output)
            Predicted targets.
        """
        if _routing_enabled():
            routed_params = process_routing(self, "predict", **predict_params)
        else:
            # TODO(SLEP6): remove when metadata routing cannot be disabled.
            routed_params = _manual_routing(
                {"final_estimator_": {"predict": predict_params}}
            )

        y_pred = super().predict(X, **routed_params.final_estimator_["predict"])
        if isinstance(self._label_encoder, list):
            # Handle the multilabel-indicator case
            y_pred = np.array(
                [
                    self._label_encoder[target_idx].inverse_transform(target)
                    for target_idx, target in enumerate(y_pred.T)
                ]
            ).T
        else:
            y_pred = self._label_encoder.inverse_transform(y_pred)
        return y_pred

    @available_if(
        _estimator_has(
            "predict_proba", delegates=("final_estimator_", "final_estimator")
        )
    )
    def predict_proba(self, X):
        # [FUNCTION DOC]
        # Returns final_estimator_.predict_proba on the stacked features. For multilabel
        # targets the per-label list is converted to an (n_samples, n_labels) array.
        # KNOWN ISSUE: the multilabel conversion keeps column 0 of each per-label probability
        # array, i.e. the probability of the negative label, which appears inverted relative
        # to the usual convention of reporting the positive class.
        """Predict class probabilities for `X` using the final estimator.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vectors, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        Returns
        -------
        probabilities : ndarray of shape (n_samples, n_classes) or \
            list of ndarray of shape (n_output,)
            The class probabilities of the input samples.
        """
        check_is_fitted(self)
        y_pred = self.final_estimator_.predict_proba(self.transform(X))

        if isinstance(self._label_encoder, list):
            # Handle the multilabel-indicator cases
            y_pred = np.array([preds[:, 0] for preds in y_pred]).T
        return y_pred

    @available_if(
        _estimator_has(
            "decision_function", delegates=("final_estimator_", "final_estimator")
        )
    )
    def decision_function(self, X):
        # [FUNCTION DOC]
        # Returns final_estimator_.decision_function on the stacked features (after checking
        # fitted state).
        """Decision function for samples in `X` using the final estimator.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vectors, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        Returns
        -------
        decisions : ndarray of shape (n_samples,), (n_samples, n_classes), \
            or (n_samples, n_classes * (n_classes-1) / 2)
            The decision function computed the final estimator.
        """
        check_is_fitted(self)
        return self.final_estimator_.decision_function(self.transform(X))

    def transform(self, X):
        # [FUNCTION DOC]
        # Returns the stacked base-estimator outputs (plus X if passthrough) via _transform.
        """Return class labels or probabilities for X for each estimator.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vectors, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        Returns
        -------
        y_preds : ndarray of shape (n_samples, n_estimators) or \
                (n_samples, n_classes * n_estimators)
            Prediction outputs for each estimator.
        """
        return self._transform(X)

    def _sk_visual_block_(self):
        # [FUNCTION DOC]
        # HELPER: Builds the HTML repr using final_estimator or a default LogisticRegression.
        # If final_estimator's default changes then this should be
        # updated.
        if self.final_estimator is None:
            final_estimator = LogisticRegression()
        else:
            final_estimator = self.final_estimator
        return super()._sk_visual_block_with_final_estimator(final_estimator)


class StackingRegressor(RegressorMixin, _BaseStacking):
    # [CLASS DOC]
    # Stacking with a final regressor (default RidgeCV). stack_method is fixed to
    # "predict", so each base estimator contributes one column per output.
    """Stack of estimators with a final regressor.

    Stacked generalization consists in stacking the output of individual
    estimator and use a regressor to compute the final prediction. Stacking
    allows to use the strength of each individual estimator by using their
    output as input of a final estimator.

    Note that `estimators_` are fitted on the full `X` while `final_estimator_`
    is trained using cross-validated predictions of the base estimators using
    `cross_val_predict`.

    Read more in the :ref:`User Guide <stacking>`.

    .. versionadded:: 0.22

    Parameters
    ----------
    estimators : list of (str, estimator)
        Base estimators which will be stacked together. Each element of the
        list is defined as a tuple of string (i.e. name) and an estimator
        instance. An estimator can be set to 'drop' using `set_params`.

    final_estimator : estimator, default=None
        A regressor which will be used to combine the base estimators.
        The default regressor is a :class:`~sklearn.linear_model.RidgeCV`.

    cv : int, cross-validation generator, iterable, or "prefit", default=None
        Determines the cross-validation splitting strategy used in
        `cross_val_predict` to train `final_estimator`. Possible inputs for
        cv are:

        * None, to use the default 5-fold cross validation,
        * integer, to specify the number of folds in a (Stratified) KFold,
        * An object to be used as a cross-validation generator,
        * An iterable yielding train, test splits,
        * `"prefit"`, to assume the `estimators` are prefit. In this case, the
          estimators will not be refitted.

        For integer/None inputs, if the estimator is a classifier and y is
        either binary or multiclass,
        :class:`~sklearn.model_selection.StratifiedKFold` is used.
        In all other cases, :class:`~sklearn.model_selection.KFold` is used.
        These splitters are instantiated with `shuffle=False` so the splits
        will be the same across calls.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.

        If "prefit" is passed, it is assumed that all `estimators` have
        been fitted already. The `final_estimator_` is trained on the `estimators`
        predictions on the full training set and are **not** cross validated
        predictions. Please note that if the models have been trained on the same
        data to train the stacking model, there is a very high risk of overfitting.

        .. versionadded:: 1.1
            The 'prefit' option was added in 1.1

        .. note::
           A larger number of split will provide no benefits if the number
           of training samples is large enough. Indeed, the training time
           will increase. ``cv`` is not used for model evaluation but for
           prediction.

    n_jobs : int, default=None
        The number of jobs to run in parallel for `fit` of all `estimators`.
        `None` means 1 unless in a `joblib.parallel_backend` context. -1 means
        using all processors. See :term:`Glossary <n_jobs>` for more details.

    passthrough : bool, default=False
        When False, only the predictions of estimators will be used as
        training data for `final_estimator`. When True, the
        `final_estimator` is trained on the predictions as well as the
        original training data.

    verbose : int, default=0
        Verbosity level.

    Attributes
    ----------
    estimators_ : list of estimators
        The elements of the `estimators` parameter, having been fitted on the
        training data. If an estimator has been set to `'drop'`, it
        will not appear in `estimators_`. When `cv="prefit"`, `estimators_`
        is set to `estimators` and is not fitted again.

    named_estimators_ : :class:`~sklearn.utils.Bunch`
        Attribute to access any fitted sub-estimators by name.

    n_features_in_ : int
        Number of features seen during :term:`fit`. Only defined if the
        underlying estimator exposes such an attribute when fit.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Only defined if the
        underlying estimators expose such an attribute when fit.

        .. versionadded:: 1.0

    final_estimator_ : estimator
        The regressor fit on the output of `estimators_` and responsible for
        final predictions.

    stack_method_ : list of str
        The method used by each base estimator.

    See Also
    --------
    StackingClassifier : Stack of estimators with a final classifier.

    References
    ----------
    .. [1] Wolpert, David H. "Stacked generalization." Neural networks 5.2
       (1992): 241-259.

    Examples
    --------
    >>> from sklearn.datasets import load_diabetes
    >>> from sklearn.linear_model import RidgeCV
    >>> from sklearn.svm import LinearSVR
    >>> from sklearn.ensemble import RandomForestRegressor
    >>> from sklearn.ensemble import StackingRegressor
    >>> X, y = load_diabetes(return_X_y=True)
    >>> estimators = [
    ...     ('lr', RidgeCV()),
    ...     ('svr', LinearSVR(random_state=42))
    ... ]
    >>> reg = StackingRegressor(
    ...     estimators=estimators,
    ...     final_estimator=RandomForestRegressor(n_estimators=10,
    ...                                           random_state=42)
    ... )
    >>> from sklearn.model_selection import train_test_split
    >>> X_train, X_test, y_train, y_test = train_test_split(
    ...     X, y, random_state=42
    ... )
    >>> reg.fit(X_train, y_train).score(X_test, y_test)
    0.3...
    """

    def __init__(
        self,
        estimators,
        final_estimator=None,
        *,
        cv=None,
        n_jobs=None,
        passthrough=False,
        verbose=0,
    ):
        # [FUNCTION DOC]
        # Forwards parameters to _BaseStacking.__init__ with stack_method="predict".
        super().__init__(
            estimators=estimators,
            final_estimator=final_estimator,
            cv=cv,
            stack_method="predict",
            n_jobs=n_jobs,
            passthrough=passthrough,
            verbose=verbose,
        )

    def _validate_final_estimator(self):
        # [FUNCTION DOC]
        # HELPER: Clones final_estimator (default RidgeCV) into final_estimator_ and raises
        # ValueError if it is not a regressor.
        self._clone_final_estimator(default=RidgeCV())
        if not is_regressor(self.final_estimator_):
            raise ValueError(
                "'final_estimator' parameter should be a regressor. Got {}".format(
                    self.final_estimator_
                )
            )

    def fit(self, X, y, **fit_params):
        # [FUNCTION DOC]
        # Rejects unsupported metadata when routing is disabled (only sample_weight allowed),
        # ravels y to 1D (warning on column vectors) and delegates to _BaseStacking.fit.
        # Outputs: self.
        # KNOWN ISSUE: column_or_1d forbids multi-output y, although predict's docstring
        # mentions (n_samples, n_output) outputs.
        """Fit the estimators.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vectors, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        y : array-like of shape (n_samples,)
            Target values.

        **fit_params : dict
            Parameters to pass to the underlying estimators.

            .. versionadded:: 1.6

                Only available if `enable_metadata_routing=True`, which can be
                set by using ``sklearn.set_config(enable_metadata_routing=True)``.
                See :ref:`Metadata Routing User Guide <metadata_routing>` for
                more details.

        Returns
        -------
        self : object
            Returns a fitted instance.
        """
        _raise_for_params(fit_params, self, "fit", allow=["sample_weight"])

        y = column_or_1d(y, warn=True)

        return super().fit(X, y, **fit_params)

    def transform(self, X):
        # [FUNCTION DOC]
        # Returns the stacked base-estimator predictions (plus X if passthrough) via
        # _transform.
        """Return the predictions for X for each estimator.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vectors, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        Returns
        -------
        y_preds : ndarray of shape (n_samples, n_estimators)
            Prediction outputs for each estimator.
        """
        return self._transform(X)

    def fit_transform(self, X, y, **fit_params):
        # [FUNCTION DOC]
        # Validates metadata like fit, then uses TransformerMixin.fit_transform (fit followed
        # by transform on the same X). Note the returned values are predictions of the
        # estimators refitted on full data, not the out-of-fold predictions used to train the
        # final estimator.
        """Fit the estimators and return the predictions for X for each estimator.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vectors, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        y : array-like of shape (n_samples,)
            Target values.

        **fit_params : dict
            Parameters to pass to the underlying estimators.

            .. versionadded:: 1.6

                Only available if `enable_metadata_routing=True`, which can be
                set by using ``sklearn.set_config(enable_metadata_routing=True)``.
                See :ref:`Metadata Routing User Guide <metadata_routing>` for
                more details.

        Returns
        -------
        y_preds : ndarray of shape (n_samples, n_estimators)
            Prediction outputs for each estimator.
        """
        _raise_for_params(fit_params, self, "fit", allow=["sample_weight"])

        return super().fit_transform(X, y, **fit_params)

    @available_if(
        _estimator_has("predict", delegates=("final_estimator_", "final_estimator"))
    )
    def predict(self, X, **predict_params):
        # [FUNCTION DOC]
        # Routes predict_params to the final estimator (process_routing or passed through
        # directly) and returns the base predict result.
        """Predict target for X.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vectors, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        **predict_params : dict of str -> obj
            Parameters to the `predict` called by the `final_estimator`. Note
            that this may be used to return uncertainties from some estimators
            with `return_std` or `return_cov`. Be aware that it will only
            account for uncertainty in the final estimator.

            - If `enable_metadata_routing=False` (default):
              Parameters directly passed to the `predict` method of the
              `final_estimator`.

            - If `enable_metadata_routing=True`: Parameters safely routed to
              the `predict` method of the `final_estimator`. See :ref:`Metadata
              Routing User Guide <metadata_routing>` for more details.

            .. versionchanged:: 1.6
                `**predict_params` can be routed via metadata routing API.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,) or (n_samples, n_output)
            Predicted targets.
        """
        if _routing_enabled():
            routed_params = process_routing(self, "predict", **predict_params)
        else:
            # TODO(SLEP6): remove when metadata routing cannot be disabled.
            routed_params = _manual_routing(
                {"final_estimator_": {"predict": predict_params}}
            )

        y_pred = super().predict(X, **routed_params.final_estimator_["predict"])

        return y_pred

    def _sk_visual_block_(self):
        # [FUNCTION DOC]
        # HELPER: Builds the HTML repr using final_estimator or a default RidgeCV.
        # If final_estimator's default changes then this should be
        # updated.
        if self.final_estimator is None:
            final_estimator = RidgeCV()
        else:
            final_estimator = self.final_estimator
        return super()._sk_visual_block_with_final_estimator(final_estimator)
