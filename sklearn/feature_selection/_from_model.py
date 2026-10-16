# [MODULE DOC]
# SelectFromModel meta-transformer: selects features whose importance (from a wrapped
#   estimator's coef_ / feature_importances_ or a custom getter) reaches a threshold,
#   optionally capped at max_features. Also contains _calculate_threshold, which turns
#   the threshold specification into a number.
# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

from copy import deepcopy
from numbers import Integral, Real

import numpy as np

from sklearn.base import BaseEstimator, MetaEstimatorMixin, _fit_context, clone
from sklearn.exceptions import NotFittedError
from sklearn.feature_selection._base import SelectorMixin, _get_feature_importances
from sklearn.utils._param_validation import HasMethods, Interval, Options
from sklearn.utils._tags import get_tags
from sklearn.utils.metadata_routing import (
    MetadataRouter,
    MethodMapping,
    _routing_enabled,
    process_routing,
)
from sklearn.utils.metaestimators import available_if
from sklearn.utils.validation import (
    _check_feature_names,
    _estimator_has,
    check_is_fitted,
    check_scalar,
)


def _calculate_threshold(estimator, importances, threshold):
    # [FUNCTION DOC]
    # HELPER: Resolves the threshold specification. None: 1e-5 if the estimator looks L1
    #   penalized (penalty attribute "l1", class name containing "Lasso", ElasticNet or
    #   LogisticRegression with l1_ratio close to 1, their CV variants with fitted
    #   l1_ratio_ close to 1), otherwise "mean". Strings: "mean", "median", or
    #   "<scale>*<mean|median>" (the reference statistic times the float scale).
    #   Anything else is converted with float.
    # Inputs: estimator - estimator used for the default heuristic; importances - 1D
    #   array; threshold - str, number or None.
    # Outputs: float threshold.
    # Exceptions: ValueError for unknown reference or string; ValueError from float.
    # KNOWN ISSUE: the inline comment says the natural default is 0 but the code uses
    #   1e-5 (intentional tolerance, misleading comment).
    """Interpret the threshold value"""

    if threshold is None:
        # determine default from estimator
        est_name = estimator.__class__.__name__
        is_l1_penalized = hasattr(estimator, "penalty") and estimator.penalty == "l1"
        is_lasso = "Lasso" in est_name
        is_elasticnet_l1_penalized = est_name == "ElasticNet" and (
            hasattr(estimator, "l1_ratio") and np.isclose(estimator.l1_ratio, 1.0)
        )
        is_elasticnetcv_l1_penalized = est_name == "ElasticNetCV" and (
            hasattr(estimator, "l1_ratio_") and np.isclose(estimator.l1_ratio_, 1.0)
        )
        is_logreg_l1_penalized = est_name == "LogisticRegression" and (
            hasattr(estimator, "l1_ratio") and np.isclose(estimator.l1_ratio, 1.0)
        )
        is_logregcv_l1_penalized = est_name == "LogisticRegressionCV" and (
            hasattr(estimator, "l1_ratio_")
            and np.all(np.isclose(estimator.l1_ratio_, 1.0))
        )
        if (
            is_l1_penalized
            or is_lasso
            or is_elasticnet_l1_penalized
            or is_elasticnetcv_l1_penalized
            or is_logreg_l1_penalized
            or is_logregcv_l1_penalized
        ):
            # the natural default threshold is 0 when l1 penalty was used
            threshold = 1e-5
        else:
            threshold = "mean"

    if isinstance(threshold, str):
        if "*" in threshold:
            scale, reference = threshold.split("*")
            scale = float(scale.strip())
            reference = reference.strip()

            if reference == "median":
                reference = np.median(importances)
            elif reference == "mean":
                reference = np.mean(importances)
            else:
                raise ValueError("Unknown reference: " + reference)

            threshold = scale * reference

        elif threshold == "median":
            threshold = np.median(importances)

        elif threshold == "mean":
            threshold = np.mean(importances)

        else:
            raise ValueError(
                "Expected threshold='mean' or threshold='median' got %s" % threshold
            )

    else:
        threshold = float(threshold)

    return threshold


class SelectFromModel(MetaEstimatorMixin, SelectorMixin, BaseEstimator):
    # [CLASS DOC]
    # Meta-estimator wrapping estimator. Fitted attributes: estimator_ (fitted clone, or
    #   deep copy when prefit), max_features_ (when max_features given), and
    #   feature_names_in_ / n_features_in_. threshold_ is computed on demand.
    """Meta-transformer for selecting features based on importance weights.

    .. versionadded:: 0.17

    Read more in the :ref:`User Guide <select_from_model>`.

    Parameters
    ----------
    estimator : object
        The base estimator from which the transformer is built.
        This can be both a fitted (if ``prefit`` is set to True)
        or a non-fitted estimator. The estimator should have a
        ``feature_importances_`` or ``coef_`` attribute after fitting.
        Otherwise, the ``importance_getter`` parameter should be used.

    threshold : str or float, default=None
        The threshold value to use for feature selection. Features whose
        absolute importance value is greater or equal are kept while the others
        are discarded. If "median" (resp. "mean"), then the ``threshold`` value
        is the median (resp. the mean) of the feature importances. A scaling
        factor (e.g., "1.25*mean") may also be used. If None and if the
        estimator has a parameter penalty set to l1, either explicitly
        or implicitly (e.g, Lasso), the threshold used is 1e-5.
        Otherwise, "mean" is used by default.

    prefit : bool, default=False
        Whether a prefit model is expected to be passed into the constructor
        directly or not.
        If `True`, `estimator` must be a fitted estimator.
        If `False`, `estimator` is fitted and updated by calling
        `fit` and `partial_fit`, respectively.

    norm_order : non-zero int, inf, -inf, default=1
        Order of the norm used to filter the vectors of coefficients below
        ``threshold`` in the case where the ``coef_`` attribute of the
        estimator is of dimension 2.

    max_features : int, callable, default=None
        The maximum number of features to select.

        - If an integer, then it specifies the maximum number of features to
          allow.
        - If a callable, then it specifies how to calculate the maximum number of
          features allowed. The callable will receive `X` as input: `max_features(X)`.
        - If `None`, then all features are kept.

        To only select based on ``max_features``, set ``threshold=-np.inf``.

        .. versionadded:: 0.20
        .. versionchanged:: 1.1
           `max_features` accepts a callable.

    importance_getter : str or callable, default='auto'
        If 'auto', uses the feature importance either through a ``coef_``
        attribute or ``feature_importances_`` attribute of estimator.

        Also accepts a string that specifies an attribute name/path
        for extracting feature importance (implemented with `attrgetter`).
        For example, give `regressor_.coef_` in case of
        :class:`~sklearn.compose.TransformedTargetRegressor`  or
        `named_steps.clf.feature_importances_` in case of
        :class:`~sklearn.pipeline.Pipeline` with its last step named `clf`.

        If `callable`, overrides the default feature importance getter.
        The callable is passed with the fitted estimator and it should
        return importance for each feature.

        .. versionadded:: 0.24

    Attributes
    ----------
    estimator_ : estimator
        The base estimator from which the transformer is built. This attribute
        exist only when `fit` has been called.

        - If `prefit=True`, it is a deep copy of `estimator`.
        - If `prefit=False`, it is a clone of `estimator` and fit on the data
          passed to `fit` or `partial_fit`.

    n_features_in_ : int
        Number of features seen during :term:`fit`. Only defined if the
        underlying estimator exposes such an attribute when fit.

        .. versionadded:: 0.24

    max_features_ : int
        Maximum number of features calculated during :term:`fit`. Only defined
        if the ``max_features`` is not `None`.

        - If `max_features` is an `int`, then `max_features_ = max_features`.
        - If `max_features` is a callable, then `max_features_ = max_features(X)`.

        .. versionadded:: 1.1

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    threshold_ : float
        The threshold value used for feature selection.

    See Also
    --------
    RFE : Recursive feature elimination based on importance weights.
    RFECV : Recursive feature elimination with built-in cross-validated
        selection of the best number of features.
    SequentialFeatureSelector : Sequential cross-validation based feature
        selection. Does not rely on importance weights.

    Notes
    -----
    Allows NaN/Inf in the input if the underlying estimator does as well.

    Examples
    --------
    >>> from sklearn.feature_selection import SelectFromModel
    >>> from sklearn.linear_model import LogisticRegression
    >>> X = [[ 0.87, -1.34,  0.31 ],
    ...      [-2.79, -0.02, -0.85 ],
    ...      [-1.34, -0.48, -2.55 ],
    ...      [ 1.92,  1.48,  0.65 ]]
    >>> y = [0, 1, 0, 1]
    >>> selector = SelectFromModel(estimator=LogisticRegression()).fit(X, y)
    >>> selector.estimator_.coef_
    array([[-0.3252,  0.8345,  0.4976]])
    >>> selector.threshold_
    np.float64(0.55249)
    >>> selector.get_support()
    array([False,  True, False])
    >>> selector.transform(X)
    array([[-1.34],
           [-0.02],
           [-0.48],
           [ 1.48]])

    Using a callable to create a selector that can use no more than half
    of the input features.

    >>> def half_callable(X):
    ...     return round(len(X[0]) / 2)
    >>> half_selector = SelectFromModel(estimator=LogisticRegression(),
    ...                                 max_features=half_callable)
    >>> _ = half_selector.fit(X, y)
    >>> half_selector.max_features_
    2
    """

    _parameter_constraints: dict = {
        "estimator": [HasMethods("fit")],
        "threshold": [Interval(Real, None, None, closed="both"), str, None],
        "prefit": ["boolean"],
        "norm_order": [
            Interval(Integral, None, -1, closed="right"),
            Interval(Integral, 1, None, closed="left"),
            Options(Real, {np.inf, -np.inf}),
        ],
        "max_features": [Interval(Integral, 0, None, closed="left"), callable, None],
        "importance_getter": [str, callable],
    }

    def __init__(
        self,
        estimator,
        *,
        threshold=None,
        prefit=False,
        norm_order=1,
        max_features=None,
        importance_getter="auto",
    ):
        # [FUNCTION DOC]
        # Stores estimator, threshold, prefit, importance_getter, norm_order and
        #   max_features as attributes.
        self.estimator = estimator
        self.threshold = threshold
        self.prefit = prefit
        self.importance_getter = importance_getter
        self.norm_order = norm_order
        self.max_features = max_features

    def _get_support_mask(self):
        # [FUNCTION DOC]
        # Uses estimator_ if fitted, else the raw estimator (prefit case without fit),
        #   and max_features_ if set, else max_features. Computes norm-aggregated
        #   importances and the threshold. If max_features is set, starts from a mask of
        #   the top max_features features by score (stable sort, earlier index wins
        #   ties), else from all True; then clears features with score below threshold.
        # Outputs: bool array (n_features,).
        # Exceptions: NotFittedError when prefit and the estimator is unfitted, or when
        #   max_features is still a callable (fit was never called); ValueError for a
        #   non-integer max_features.
        # KNOWN ISSUE: the top-k branch tests self.max_features rather than the resolved
        #   local max_features (equivalent in practice since both are None together).
        #   The callable error message mentions prefit=True though it triggers whenever
        #   fit was skipped.
        estimator = getattr(self, "estimator_", self.estimator)
        max_features = getattr(self, "max_features_", self.max_features)

        if self.prefit:
            try:
                check_is_fitted(self.estimator)
            except NotFittedError as exc:
                raise NotFittedError(
                    "When `prefit=True`, `estimator` is expected to be a fitted "
                    "estimator."
                ) from exc
        if callable(max_features):
            # This branch is executed when `transform` is called directly and thus
            # `max_features_` is not set and we fallback using `self.max_features`
            # that is not validated
            raise NotFittedError(
                "When `prefit=True` and `max_features` is a callable, call `fit` "
                "before calling `transform`."
            )
        elif max_features is not None and not isinstance(max_features, Integral):
            raise ValueError(
                f"`max_features` must be an integer. Got `max_features={max_features}` "
                "instead."
            )

        scores = _get_feature_importances(
            estimator=estimator,
            getter=self.importance_getter,
            transform_func="norm",
            norm_order=self.norm_order,
        )
        threshold = _calculate_threshold(estimator, scores, self.threshold)
        if self.max_features is not None:
            mask = np.zeros_like(scores, dtype=bool)
            candidate_indices = np.argsort(-scores, kind="mergesort")[:max_features]
            mask[candidate_indices] = True
        else:
            mask = np.ones_like(scores, dtype=bool)
        mask[scores < threshold] = False
        return mask

    def _check_max_features(self, X):
        # [FUNCTION DOC]
        # HELPER: If max_features is set, resolves it (calling it with X when callable),
        #   validates it is an integer >= 0 with check_scalar, and stores max_features_.
        # Exceptions: TypeError / ValueError from check_scalar.
        # Side effects: sets max_features_.
        if self.max_features is not None:
            if callable(self.max_features):
                max_features = self.max_features(X)
            else:  # int
                max_features = self.max_features

            check_scalar(
                max_features,
                "max_features",
                Integral,
                min_val=0,
                max_val=None,
            )
            self.max_features_ = max_features

    @_fit_context(
        # SelectFromModel.estimator is not validated yet
        prefer_skip_nested_validation=False
    )
    def fit(self, X, y=None, **fit_params):
        # [FUNCTION DOC]
        # Resolves max_features. prefit: checks the estimator is fitted and stores a
        #   deep copy as estimator_. Otherwise clones and fits the estimator on (X, y),
        #   forwarding routed fit params (routing enabled) or all fit_params. Copies
        #   feature_names_in_ from the inner estimator if it has one; otherwise records
        #   feature names from X.
        # Inputs: X - (n_samples, n_features); y - targets or None; fit_params -
        #   forwarded.
        # Outputs: self.
        # Exceptions: NotFittedError when prefit and unfitted; errors from the
        #   estimator.
        # Side effects: sets estimator_, max_features_, feature_names_in_.
        """Fit the SelectFromModel meta-transformer.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.

        y : array-like of shape (n_samples,), default=None
            The target values (integers that correspond to classes in
            classification, real numbers in regression).

        **fit_params : dict
            - If `enable_metadata_routing=False` (default): Parameters directly passed
              to the `fit` method of the sub-estimator. They are ignored if
              `prefit=True`.

            - If `enable_metadata_routing=True`: Parameters safely routed to the `fit`
              method of the sub-estimator. They are ignored if `prefit=True`.

            .. versionchanged:: 1.4
                See :ref:`Metadata Routing User Guide <metadata_routing>` for
                more details.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        self._check_max_features(X)

        if self.prefit:
            try:
                check_is_fitted(self.estimator)
            except NotFittedError as exc:
                raise NotFittedError(
                    "When `prefit=True`, `estimator` is expected to be a fitted "
                    "estimator."
                ) from exc
            self.estimator_ = deepcopy(self.estimator)
        else:
            if _routing_enabled():
                routed_params = process_routing(self, "fit", **fit_params)
                self.estimator_ = clone(self.estimator)
                self.estimator_.fit(X, y, **routed_params.estimator.fit)
            else:
                # TODO(SLEP6): remove when metadata routing cannot be disabled.
                self.estimator_ = clone(self.estimator)
                self.estimator_.fit(X, y, **fit_params)

        if hasattr(self.estimator_, "feature_names_in_"):
            self.feature_names_in_ = self.estimator_.feature_names_in_
        else:
            _check_feature_names(self, X, reset=True)

        return self

    @property
    def threshold_(self):
        # [FUNCTION DOC]
        # Property: recomputes importances from estimator_ and returns the resolved
        #   threshold.
        # KNOWN ISSUE: passes self.estimator (the unfitted template) instead of
        #   estimator_ to _calculate_threshold, so the default-threshold heuristic that
        #   checks fitted l1_ratio_ (ElasticNetCV / LogisticRegressionCV) can differ
        #   from the one used in _get_support_mask once fitted.
        # Exceptions: AttributeError if not fitted.
        """Threshold value used for feature selection."""
        scores = _get_feature_importances(
            estimator=self.estimator_,
            getter=self.importance_getter,
            transform_func="norm",
            norm_order=self.norm_order,
        )
        return _calculate_threshold(self.estimator, scores, self.threshold)

    @available_if(_estimator_has("partial_fit"))
    @_fit_context(
        # SelectFromModel.estimator is not validated yet
        prefer_skip_nested_validation=False
    )
    def partial_fit(self, X, y=None, **partial_fit_params):
        # [FUNCTION DOC]
        # Available only if the wrapped estimator has partial_fit. On the first call
        #   resolves max_features. prefit: on first call validates and deep-copies the
        #   estimator, then returns without training. Otherwise clones the estimator on
        #   the first call and calls its partial_fit with the forwarded params. Feature
        #   names are copied from the inner estimator or checked against X (reset only
        #   on first call).
        # Outputs: self.
        # Side effects: sets / updates estimator_, max_features_, feature_names_in_.
        # KNOWN ISSUE: with metadata routing enabled, estimator_ is re-cloned on every
        #   call, discarding previous incremental training.
        """Fit the SelectFromModel meta-transformer only once.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.

        y : array-like of shape (n_samples,), default=None
            The target values (integers that correspond to classes in
            classification, real numbers in regression).

        **partial_fit_params : dict
            - If `enable_metadata_routing=False` (default): Parameters directly passed
              to the `partial_fit` method of the sub-estimator.

            - If `enable_metadata_routing=True`: Parameters passed to the `partial_fit`
              method of the sub-estimator. They are ignored if `prefit=True`.

            .. versionchanged:: 1.4

                `**partial_fit_params` are routed to the sub-estimator, if
                `enable_metadata_routing=True` is set via
                :func:`~sklearn.set_config`, which allows for aliasing.

                See :ref:`Metadata Routing User Guide <metadata_routing>` for
                more details.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        first_call = not hasattr(self, "estimator_")

        if first_call:
            self._check_max_features(X)

        if self.prefit:
            if first_call:
                try:
                    check_is_fitted(self.estimator)
                except NotFittedError as exc:
                    raise NotFittedError(
                        "When `prefit=True`, `estimator` is expected to be a fitted "
                        "estimator."
                    ) from exc
                self.estimator_ = deepcopy(self.estimator)
            return self

        if first_call:
            self.estimator_ = clone(self.estimator)
        if _routing_enabled():
            routed_params = process_routing(self, "partial_fit", **partial_fit_params)
            self.estimator_ = clone(self.estimator)
            self.estimator_.partial_fit(X, y, **routed_params.estimator.partial_fit)
        else:
            # TODO(SLEP6): remove when metadata routing cannot be disabled.
            self.estimator_.partial_fit(X, y, **partial_fit_params)

        if hasattr(self.estimator_, "feature_names_in_"):
            self.feature_names_in_ = self.estimator_.feature_names_in_
        else:
            _check_feature_names(self, X, reset=first_call)

        return self

    @property
    def n_features_in_(self):
        # [FUNCTION DOC]
        # Property: number of features of the fitted inner estimator.
        # Exceptions: AttributeError (instead of NotFittedError) when unfitted, so
        #   hasattr returns False.
        """Number of features seen during `fit`."""
        # For consistency with other estimators we raise an AttributeError so
        # that hasattr() fails if the estimator isn't fitted.
        try:
            check_is_fitted(self)
        except NotFittedError as nfe:
            raise AttributeError(
                "{} object has no n_features_in_ attribute.".format(
                    self.__class__.__name__
                )
            ) from nfe

        return self.estimator_.n_features_in_

    def get_metadata_routing(self):
        # [FUNCTION DOC]
        # Builds a MetadataRouter routing fit to the estimator's fit and partial_fit to
        #   its partial_fit.
        # Outputs: MetadataRouter.
        """Get metadata routing of this object.

        Please check :ref:`User Guide <metadata_routing>` on how the routing
        mechanism works.

        .. versionadded:: 1.4

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

    def __sklearn_tags__(self):
        # [FUNCTION DOC]
        # Copies sparse and allow_nan input tags from the wrapped estimator.
        tags = super().__sklearn_tags__()
        tags.input_tags.sparse = get_tags(self.estimator).input_tags.sparse
        tags.input_tags.allow_nan = get_tags(self.estimator).input_tags.allow_nan
        return tags
