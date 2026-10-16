# [MODULE DOC]
# HELPER: Shared test utilities for metadata-routing tests. Provides functions that
# record which metadata each method received (keyed by callee and caller found via
# stack inspection) and check those records, assertions on metadata requests, and
# a set of dummy consumers (regressors, classifiers, transformers, scorer, splitter)
# and routers (meta-estimators) used across sklearn's routing test suites.
import inspect
from collections import defaultdict
from functools import partial

import numpy as np
from numpy.testing import assert_array_equal

from sklearn.base import (
    BaseEstimator,
    ClassifierMixin,
    MetaEstimatorMixin,
    RegressorMixin,
    TransformerMixin,
    clone,
)
from sklearn.metrics._scorer import _Scorer, mean_squared_error
from sklearn.model_selection import BaseCrossValidator
from sklearn.model_selection._split import GroupKFold, GroupsConsumerMixin
from sklearn.utils._metadata_requests import (
    METHODS,
    SIMPLE_METHODS,
)
from sklearn.utils.metadata_routing import (
    MetadataRouter,
    MethodMapping,
    process_routing,
)
from sklearn.utils.multiclass import _check_partial_fit_first_call


def record_metadata(obj, record_default=True, **kwargs):
    # [FUNCTION DOC]
    # HELPER: Stores the metadata kwargs received by a method on obj._records.
    # Behavior: inspects the call stack; callee is the innermost frame whose function
    # name is in METHODS or is "consuming_metric"; caller is the outermost frame above it
    # whose name is in METHODS or a model_selection validation function (cross_validate,
    # cross_val_score, ...). Falls back to stack[1] / stack[2] function names. Appends
    # the kwargs dict to obj._records[callee][caller].
    # Inputs: obj - any object to hold records; record_default - if False, drop kwargs
    # whose value is the string "default"; **kwargs - metadata to record.
    # Outputs: None.
    # Side effects: lazily creates obj._records (nested defaultdict of lists) and mutates it.
    """Utility function to store passed metadata to a method of obj.

    If record_default is False, kwargs whose values are "default" are skipped.
    This is so that checks on keyword arguments whose default was not changed
    are skipped.

    """
    validation_functions = [
        "cross_validate",
        "cross_val_score",
        "cross_val_predict",
        "learning_curve",
        "permutation_test_score",
        "validation_curve",
    ]
    stack = inspect.stack()
    # for callee, extract the innermost `function` (which is in METHODS) from the
    # stack; for caller, extract the *outermost* matching function so the expected
    # `parent` is the top-level public entry point (e.g. `cross_validate`) rather
    # than an internal helper whose name may change. Fall back to positional frames:
    callee = caller = None
    for i, frame in enumerate(stack):
        if frame.function in METHODS + ["consuming_metric"]:
            callee = frame.function
            break
    for frame in stack[i + 1 :]:
        if frame.function in METHODS + validation_functions:
            caller = frame.function
    callee = callee or stack[1].function
    caller = caller or stack[2].function

    if not hasattr(obj, "_records"):
        obj._records = defaultdict(lambda: defaultdict(list))
    if not record_default:
        kwargs = {
            key: val
            for key, val in kwargs.items()
            if not isinstance(val, str) or (val != "default")
        }
    obj._records[callee][caller].append(kwargs)


def check_recorded_metadata(
    obj, method, parent, split_params=tuple(), preserves_metadata=True, **kwargs
):
    # [FUNCTION DOC]
    # HELPER: Asserts records exist on obj for (method, parent) and that every record has
    # exactly the keys in kwargs. If preserves_metadata, also checks values: keys in
    # split_params must be a subset (np.isin) of the passed value; ndarray values must be
    # equal; other values must be the identical object.
    # Inputs: see docstring; preserves_metadata - bool, default True, skip value checks if
    # False.
    # Exceptions: AssertionError on missing records or mismatches.
    """Check whether the expected metadata is passed to the object's method.

    Parameters
    ----------
    obj : estimator object
        sub-estimator to check routed params for
    method : str
        sub-estimator's method where metadata is routed to, or otherwise in
        the context of metadata routing referred to as 'callee'
    parent : str
        the parent method which should have called `method`, or otherwise in
        the context of metadata routing referred to as 'caller'
    split_params : tuple, default=empty
        specifies any parameters which are to be checked as being a subset
        of the original values
    **kwargs : dict
        passed metadata
    """
    records = getattr(obj, "_records", dict()).get(method, dict()).get(parent, list())
    assert records, (
        f"No overlapping records. Checked routing for obj `{obj.__class__.__name__}` "
        f"with callee: `{method}`, caller: `{parent}`; records exist for callee-caller "
        f"pairs: {[(k1, k2) for k1, d in obj._records.items() for k2 in d]}."
    )
    for record in records:
        # first check that the names of the metadata passed are the same as
        # expected. The names are stored as keys in `record`.
        assert set(kwargs.keys()) == set(record.keys()), (
            f"Expected {kwargs.keys()}, got {record.keys()}"
        )
        if not preserves_metadata:
            continue
        for key, value in kwargs.items():
            recorded_value = record[key]
            # The following condition is used to check for any specified parameters
            # being a subset of the original values
            if key in split_params and recorded_value is not None:
                assert np.isin(recorded_value, value).all()
            else:
                if isinstance(recorded_value, np.ndarray):
                    assert_array_equal(recorded_value, value)
                else:
                    assert recorded_value is value, (
                        f"Expected {recorded_value} vs {value}. Method: {method}"
                    )


record_metadata_not_default = partial(record_metadata, record_default=False)


def assert_request_is_empty(metadata_request, exclude=None):
    # [FUNCTION DOC]
    # HELPER: Asserts that a MetadataRequest has no requested/aliased params on any
    # SIMPLE_METHODS method except those in exclude. For a MetadataRouter, recurses into
    # each child's router, passing exclude[name] as that child's exclusion list.
    # Exceptions: AssertionError if any method requests something.
    # KNOWN ISSUE: the condition "isinstance(alias, str) or alias is not None" reduces to
    # "alias is not None", so params explicitly set to False also count as non-empty.
    """Check if a metadata request dict is empty.

    One can exclude a method or a list of methods from the check using the
    ``exclude`` parameter. If metadata_request is a MetadataRouter, then
    ``exclude`` can be of the form ``{"object" : [method, ...]}``.
    """
    if isinstance(metadata_request, MetadataRouter):
        for name, route_mapping in metadata_request:
            if exclude is not None and name in exclude:
                _exclude = exclude[name]
            else:
                _exclude = None
            assert_request_is_empty(route_mapping.router, exclude=_exclude)
        return

    exclude = [] if exclude is None else exclude
    for method in SIMPLE_METHODS:
        if method in exclude:
            continue
        mmr = getattr(metadata_request, method)
        props = [
            prop
            for prop, alias in mmr.requests.items()
            if isinstance(alias, str) or alias is not None
        ]
        assert not props


def assert_request_equal(request, dictionary):
    # [FUNCTION DOC]
    # HELPER: Asserts that request's per-method requests equal the dict given for each
    # method in dictionary, and that every other SIMPLE_METHODS method has no requests.
    # Inputs: request - MetadataRequest; dictionary - {method: {param: alias}}.
    # Exceptions: AssertionError on mismatch.
    for method, requests in dictionary.items():
        mmr = getattr(request, method)
        assert mmr.requests == requests

    empty_methods = [method for method in SIMPLE_METHODS if method not in dictionary]
    for method in empty_methods:
        assert not len(getattr(request, method).requests)


class _Registry(list):
    # [CLASS DOC]
    # HELPER: list subclass used as a shared registry of sub-estimator instances; copy and
    # deepcopy return the same instance so clones keep appending to the same list.
    # This list is used to get a reference to the sub-estimators, which are not
    # necessarily stored on the metaestimator. We need to override __deepcopy__
    # because the sub-estimators are probably cloned, which would result in a
    # new copy of the list, but we need copy and deep copy both to return the
    # same instance.
    def __deepcopy__(self, memo):
        # [FUNCTION DOC]
        # HELPER: Returns self instead of a deep copy (memo ignored).
        return self

    def __copy__(self):
        # [FUNCTION DOC]
        # HELPER: Returns self instead of a shallow copy.
        return self

    def __repr__(self):
        # [FUNCTION DOC]
        # HELPER: Returns a string with the registry length and class names of its entries.
        return (
            f"_Registry(n={len(self)}, "
            f"estimators={[x.__class__.__name__ for x in self]})."
        )


class ConsumingRegressor(RegressorMixin, BaseEstimator):
    # [CLASS DOC]
    # HELPER: Dummy regressor whose methods record sample_weight and metadata (ignoring
    # values left at "default") and optionally register themselves in registry.
    """A regressor consuming metadata.

    Parameters
    ----------
    registry : list, default=None
        If a list, the estimator will append itself to the list in order to have
        a reference to the estimator later on. Since that reference is not
        required in all tests, registration can be skipped by leaving this value
        as None.
    """

    def __init__(self, registry=None):
        # [FUNCTION DOC]
        # HELPER: Stores registry unchanged.
        self.registry = registry

    def partial_fit(self, X, y, sample_weight="default", metadata="default"):
        # [FUNCTION DOC]
        # HELPER: Appends self to registry if set, records metadata, returns self.
        if self.registry is not None:
            self.registry.append(self)

        record_metadata_not_default(
            self, sample_weight=sample_weight, metadata=metadata
        )
        return self

    def fit(self, X, y, sample_weight="default", metadata="default"):
        # [FUNCTION DOC]
        # HELPER: Appends self to registry if set, records metadata, returns self. Sets no
        # fitted attributes.
        if self.registry is not None:
            self.registry.append(self)

        record_metadata_not_default(
            self, sample_weight=sample_weight, metadata=metadata
        )
        return self

    def predict(self, X, y=None, sample_weight="default", metadata="default"):
        # [FUNCTION DOC]
        # HELPER: Records metadata and returns zeros of shape (n_samples,).
        record_metadata_not_default(
            self, sample_weight=sample_weight, metadata=metadata
        )
        return np.zeros(shape=(len(X),))

    def score(self, X, y, sample_weight="default", metadata="default"):
        # [FUNCTION DOC]
        # HELPER: Records metadata and returns the constant score 1.
        record_metadata_not_default(
            self, sample_weight=sample_weight, metadata=metadata
        )
        return 1


class NonConsumingClassifier(ClassifierMixin, BaseEstimator):
    # [CLASS DOC]
    # HELPER: Dummy classifier whose methods accept no metadata.
    """A classifier which accepts no metadata on any method."""

    def __init__(self, alpha=0.0):
        # [FUNCTION DOC]
        # HELPER: Stores alpha (unused hyperparameter).
        self.alpha = alpha

    def fit(self, X, y):
        # [FUNCTION DOC]
        # HELPER: Sets classes_ to unique labels of y and coef_ to ones shaped like X; returns
        # self.
        self.classes_ = np.unique(y)
        self.coef_ = np.ones_like(X)
        return self

    def partial_fit(self, X, y, classes=None):
        # [FUNCTION DOC]
        # HELPER: No-op; returns self.
        return self

    def decision_function(self, X):
        # [FUNCTION DOC]
        # HELPER: Returns predict(X).
        return self.predict(X)

    def predict(self, X):
        # [FUNCTION DOC]
        # HELPER: Returns float array: first half of samples labeled 0, remaining half 1.
        y_pred = np.empty(shape=(len(X),))
        y_pred[: len(X) // 2] = 0
        y_pred[len(X) // 2 :] = 1
        return y_pred

    def predict_proba(self, X):
        # [FUNCTION DOC]
        # HELPER: Returns random float32 probabilities (n_samples, n_classes) drawn from a
        # Dirichlet distribution so rows sum to 1. Uses global numpy RNG; needs fit first.
        # dummy probabilities to support predict_proba
        y_proba = np.empty(shape=(len(X), len(self.classes_)), dtype=np.float32)
        # each row sums up to 1.0:
        y_proba[:] = np.random.dirichlet(alpha=np.ones(len(self.classes_)), size=len(X))
        return y_proba

    def predict_log_proba(self, X):
        # [FUNCTION DOC]
        # HELPER: Returns predict_proba(X) unchanged (not actually log-probabilities).
        # dummy probabilities to support predict_log_proba
        return self.predict_proba(X)


class NonConsumingRegressor(RegressorMixin, BaseEstimator):
    # [CLASS DOC]
    # HELPER: Dummy regressor accepting no metadata.
    # KNOWN ISSUE: docstring says "classifier" but this is a regressor.
    """A classifier which accepts no metadata on any method."""

    def fit(self, X, y):
        # [FUNCTION DOC]
        # HELPER: No-op; returns self.
        return self

    def partial_fit(self, X, y):
        # [FUNCTION DOC]
        # HELPER: No-op; returns self.
        return self

    def predict(self, X):
        # [FUNCTION DOC]
        # HELPER: Returns ones of shape (n_samples,).
        return np.ones(len(X))  # pragma: no cover


class ConsumingClassifier(ClassifierMixin, BaseEstimator):
    # [CLASS DOC]
    # HELPER: Dummy classifier whose methods record sample_weight and metadata (ignoring
    # "default" values) and optionally register themselves.
    """A classifier consuming metadata.

    Parameters
    ----------
    registry : list, default=None
        If a list, the estimator will append itself to the list in order to have
        a reference to the estimator later on. Since that reference is not
        required in all tests, registration can be skipped by leaving this value
        as None.

    alpha : float, default=0
        This parameter is only used to test the ``*SearchCV`` objects, and
        doesn't do anything.
    """

    def __init__(self, registry=None, alpha=0.0):
        # [FUNCTION DOC]
        # HELPER: Stores alpha and registry unchanged.
        self.alpha = alpha
        self.registry = registry

    def partial_fit(
        self, X, y, classes=None, sample_weight="default", metadata="default"
    ):
        # [FUNCTION DOC]
        # HELPER: Registers self, records metadata, validates/sets classes_ on first call via
        # _check_partial_fit_first_call; returns self.
        # Exceptions: ValueError from _check_partial_fit_first_call when classes is missing
        # on the first call or inconsistent later.
        if self.registry is not None:
            self.registry.append(self)

        record_metadata_not_default(
            self, sample_weight=sample_weight, metadata=metadata
        )
        _check_partial_fit_first_call(self, classes)
        return self

    def fit(self, X, y, sample_weight="default", metadata="default"):
        # [FUNCTION DOC]
        # HELPER: Registers self, records metadata, sets classes_ (unique y) and coef_ (ones
        # like X); returns self.
        if self.registry is not None:
            self.registry.append(self)

        record_metadata_not_default(
            self, sample_weight=sample_weight, metadata=metadata
        )

        self.classes_ = np.unique(y)
        self.coef_ = np.ones_like(X)
        return self

    def predict(self, X, sample_weight="default", metadata="default"):
        # [FUNCTION DOC]
        # HELPER: Records metadata; returns int8 array with first half 1 and second half 0.
        record_metadata_not_default(
            self, sample_weight=sample_weight, metadata=metadata
        )
        y_pred = np.empty(shape=(len(X),), dtype="int8")
        y_pred[len(X) // 2 :] = 0
        y_pred[: len(X) // 2] = 1
        return y_pred

    def predict_proba(self, X, sample_weight="default", metadata="default"):
        # [FUNCTION DOC]
        # HELPER: Records metadata; returns random Dirichlet float32 probabilities of shape
        # (n_samples, n_classes). Uses global numpy RNG.
        record_metadata_not_default(
            self, sample_weight=sample_weight, metadata=metadata
        )
        y_proba = np.empty(shape=(len(X), len(self.classes_)), dtype=np.float32)
        # each row sums up to 1.0:
        y_proba[:] = np.random.dirichlet(alpha=np.ones(len(self.classes_)), size=len(X))
        return y_proba

    def predict_log_proba(self, X, sample_weight="default", metadata="default"):
        # [FUNCTION DOC]
        # HELPER: Records metadata, then returns predict_proba(X) (which records again under
        # the predict_proba callee).
        record_metadata_not_default(
            self, sample_weight=sample_weight, metadata=metadata
        )
        return self.predict_proba(X)

    def decision_function(self, X, sample_weight="default", metadata="default"):
        # [FUNCTION DOC]
        # HELPER: Records metadata; returns float scores with first half 1 and second half 0.
        record_metadata_not_default(
            self, sample_weight=sample_weight, metadata=metadata
        )
        y_score = np.empty(shape=(len(X),))
        y_score[len(X) // 2 :] = 0
        y_score[: len(X) // 2] = 1
        return y_score

    def score(self, X, y, sample_weight="default", metadata="default"):
        # [FUNCTION DOC]
        # HELPER: Records metadata and returns constant score 1.
        record_metadata_not_default(
            self, sample_weight=sample_weight, metadata=metadata
        )
        return 1


class ConsumingClassifierWithoutPredictProba(ConsumingClassifier):
    # [CLASS DOC]
    # HELPER: ConsumingClassifier whose predict_proba attribute access raises.
    """ConsumingClassifier without a predict_proba method, but with predict_log_proba.

    Used to mimic dynamic method selection such as in the `_parallel_predict_proba()`
    function called by `BaggingClassifier`.
    """

    @property
    def predict_proba(self):
        # [FUNCTION DOC]
        # HELPER: Property; always raises AttributeError so hasattr reports no predict_proba.
        raise AttributeError("This estimator does not support predict_proba")


class ConsumingClassifierWithoutPredictLogProba(ConsumingClassifier):
    # [CLASS DOC]
    # HELPER: ConsumingClassifier whose predict_log_proba attribute access raises.
    """ConsumingClassifier without a predict_log_proba method, but with predict_proba.

    Used to mimic dynamic method selection such as in
    `BaggingClassifier.predict_log_proba()`.
    """

    @property
    def predict_log_proba(self):
        # [FUNCTION DOC]
        # HELPER: Property; always raises AttributeError so hasattr reports no
        # predict_log_proba.
        raise AttributeError("This estimator does not support predict_log_proba")


class ConsumingClassifierWithOnlyPredict(ConsumingClassifier):
    # [CLASS DOC]
    # HELPER: ConsumingClassifier exposing neither predict_proba nor predict_log_proba
    # (decision_function, predict, score remain available).
    # KNOWN ISSUE: docstring says "only a predict method" though decision_function and
    # score are inherited and still present.
    """ConsumingClassifier with only a predict method.

    Used to mimic dynamic method selection such as in
    `BaggingClassifier.predict_log_proba()`.
    """

    @property
    def predict_proba(self):
        # [FUNCTION DOC]
        # HELPER: Property; always raises AttributeError.
        raise AttributeError("This estimator does not support predict_proba")

    @property
    def predict_log_proba(self):
        # [FUNCTION DOC]
        # HELPER: Property; always raises AttributeError.
        raise AttributeError("This estimator does not support predict_log_proba")


class ConsumingTransformer(TransformerMixin, BaseEstimator):
    # [CLASS DOC]
    # HELPER: Dummy transformer recording sample_weight and metadata on all methods.
    """A transformer which accepts metadata on fit and transform.

    Parameters
    ----------
    registry : list, default=None
        If a list, the estimator will append itself to the list in order to have
        a reference to the estimator later on. Since that reference is not
        required in all tests, registration can be skipped by leaving this value
        as None.
    """

    def __init__(self, registry=None):
        # [FUNCTION DOC]
        # HELPER: Stores registry unchanged.
        self.registry = registry

    def fit(self, X, y=None, sample_weight="default", metadata="default"):
        # [FUNCTION DOC]
        # HELPER: Registers self, records metadata, sets fitted_ = True; returns self.
        if self.registry is not None:
            self.registry.append(self)

        record_metadata_not_default(
            self, sample_weight=sample_weight, metadata=metadata
        )
        self.fitted_ = True
        return self

    def transform(self, X, sample_weight="default", metadata="default"):
        # [FUNCTION DOC]
        # HELPER: Records metadata; returns X + 1.
        record_metadata_not_default(
            self, sample_weight=sample_weight, metadata=metadata
        )
        return X + 1

    def fit_transform(self, X, y, sample_weight="default", metadata="default"):
        # [FUNCTION DOC]
        # HELPER: Records metadata under fit_transform, then calls fit and transform with the
        # same metadata (each records again); returns X + 1.
        # implementing ``fit_transform`` is necessary since
        # ``TransformerMixin.fit_transform`` doesn't route any metadata to
        # ``transform``, while here we want ``transform`` to receive
        # ``sample_weight`` and ``metadata``.
        record_metadata_not_default(
            self, sample_weight=sample_weight, metadata=metadata
        )
        return self.fit(X, y, sample_weight=sample_weight, metadata=metadata).transform(
            X, sample_weight=sample_weight, metadata=metadata
        )

    def inverse_transform(self, X, sample_weight=None, metadata=None):
        # [FUNCTION DOC]
        # HELPER: Records metadata; returns X - 1.
        # KNOWN ISSUE: defaults are None rather than "default", so with the not-default
        # recorder unpassed params are still recorded as None, unlike the other methods.
        record_metadata_not_default(
            self, sample_weight=sample_weight, metadata=metadata
        )
        return X - 1


class ConsumingNoFitTransformTransformer(BaseEstimator):
    # [CLASS DOC]
    # HELPER: Metadata-consuming transformer without TransformerMixin (no fit_transform);
    # records all values including defaults.
    """A metadata consuming transformer that doesn't inherit from
    TransformerMixin, and thus doesn't implement `fit_transform`. Note that
    TransformerMixin's `fit_transform` doesn't route metadata to `transform`."""

    def __init__(self, registry=None):
        # [FUNCTION DOC]
        # HELPER: Stores registry unchanged.
        self.registry = registry

    def fit(self, X, y=None, sample_weight=None, metadata=None):
        # [FUNCTION DOC]
        # HELPER: Registers self, records sample_weight and metadata (defaults included);
        # returns self.
        if self.registry is not None:
            self.registry.append(self)

        record_metadata(self, sample_weight=sample_weight, metadata=metadata)

        return self

    def transform(self, X, sample_weight=None, metadata=None):
        # [FUNCTION DOC]
        # HELPER: Records sample_weight and metadata; returns X unchanged.
        record_metadata(self, sample_weight=sample_weight, metadata=metadata)
        return X


def consuming_metric(y_pred, y_true, registry=None, scorer=None, **kwargs):
    # [FUNCTION DOC]
    # HELPER: Metric function used by ConsumingScorer. Records kwargs (non-default) on
    # scorer if given, else on the function object itself, optionally registers that
    # holder, and returns mean_squared_error with the given sample_weight.
    # Inputs: y_pred, y_true - arrays; registry - optional list; scorer - record holder.
    # Outputs: float MSE.
    # KNOWN ISSUE: parameter names are swapped relative to the scorer's call order
    # (score_func(y_true, y_pred)); harmless since MSE is symmetric.
    # Emulates metric function used in a `_Scorer`. Records are stored on the
    # per-test `scorer` instance, not on this module-level function, so they
    # don't leak across tests.
    holder = scorer if scorer is not None else consuming_metric
    if registry is not None:
        registry.append(holder)
    record_metadata_not_default(holder, **kwargs)
    sample_weight = kwargs.get("sample_weight", None)
    return mean_squared_error(y_pred, y_true, sample_weight=sample_weight)


class ConsumingScorer(_Scorer):
    # [CLASS DOC]
    # HELPER: _Scorer using consuming_metric with response_method "predict", sign 1,
    # recording routed metadata on this scorer instance.
    def __init__(self, registry=None):
        # [FUNCTION DOC]
        # HELPER: Stores registry and builds the scorer around consuming_metric partially bound
        # to registry and self.
        self.registry = registry
        score_func = partial(consuming_metric, registry=registry, scorer=self)
        super().__init__(
            score_func=score_func, sign=1, kwargs={}, response_method="predict"
        )


class ConsumingSplitter(GroupsConsumerMixin, BaseCrossValidator):
    # [CLASS DOC]
    # HELPER: Two-fold splitter consuming groups and metadata, recording them on split.
    def __init__(self, registry=None):
        # [FUNCTION DOC]
        # HELPER: Stores registry unchanged.
        self.registry = registry

    def split(self, X, y=None, groups="default", metadata="default"):
        # [FUNCTION DOC]
        # HELPER: Registers self, records groups/metadata (non-default), then yields two splits
        # over the first half / second half of indices: first (second half as train, first
        # half as test), then the reverse.
        # Outputs: generator of (train_indices, test_indices) lists.
        if self.registry is not None:
            self.registry.append(self)

        record_metadata_not_default(self, groups=groups, metadata=metadata)

        split_index = len(X) // 2
        train_indices = list(range(0, split_index))
        test_indices = list(range(split_index, len(X)))
        yield test_indices, train_indices
        yield train_indices, test_indices

    def get_n_splits(self, X=None, y=None, groups=None, metadata=None):
        # [FUNCTION DOC]
        # HELPER: Always returns 2.
        return 2

    def _iter_test_indices(self, X=None, y=None, groups=None):
        # [FUNCTION DOC]
        # HELPER: Yields the second-half indices, then the first-half indices, as test sets.
        # Not used by split above (which is overridden).
        split_index = len(X) // 2
        train_indices = list(range(0, split_index))
        test_indices = list(range(split_index, len(X)))
        yield test_indices
        yield train_indices


class ConsumingSplitterInheritingFromGroupKFold(ConsumingSplitter, GroupKFold):
    # [CLASS DOC]
    # HELPER: ConsumingSplitter that also is a GroupKFold, for estimators (e.g.
    # TargetEncoder) that accept only specific splitter types.
    """Helper class that can be used to test TargetEncoder, that only takes specific
    splitters."""


class MetaRegressor(MetaEstimatorMixin, RegressorMixin, BaseEstimator):
    # [CLASS DOC]
    # HELPER: Meta-regressor that only routes fit metadata to a clone of estimator.
    """A meta-regressor which is only a router."""

    def __init__(self, estimator):
        # [FUNCTION DOC]
        # HELPER: Stores estimator.
        self.estimator = estimator

    def fit(self, X, y, **fit_params):
        # [FUNCTION DOC]
        # HELPER: Routes fit_params via process_routing and fits a clone into estimator_.
        # KNOWN ISSUE: does not return self (returns None), unlike the fit convention.
        params = process_routing(self, "fit", **fit_params)
        self.estimator_ = clone(self.estimator).fit(X, y, **params.estimator.fit)

    def get_metadata_routing(self):
        # [FUNCTION DOC]
        # HELPER: Returns a MetadataRouter mapping fit -> estimator.fit.
        router = MetadataRouter(owner=self).add(
            estimator=self.estimator,
            method_mapping=MethodMapping().add(caller="fit", callee="fit"),
        )
        return router


class WeightedMetaRegressor(MetaEstimatorMixin, RegressorMixin, BaseEstimator):
    # [CLASS DOC]
    # HELPER: Meta-regressor that consumes sample_weight itself and routes fit/predict
    # metadata to a cloned estimator.
    """A meta-regressor which is also a consumer."""

    def __init__(self, estimator, registry=None):
        # [FUNCTION DOC]
        # HELPER: Stores estimator and registry.
        self.estimator = estimator
        self.registry = registry

    def fit(self, X, y, sample_weight=None, **fit_params):
        # [FUNCTION DOC]
        # HELPER: Registers self, records sample_weight, routes metadata and fits a clone into
        # estimator_; returns self.
        if self.registry is not None:
            self.registry.append(self)

        record_metadata(self, sample_weight=sample_weight)
        params = process_routing(self, "fit", sample_weight=sample_weight, **fit_params)
        self.estimator_ = clone(self.estimator).fit(X, y, **params.estimator.fit)
        return self

    def predict(self, X, **predict_params):
        # [FUNCTION DOC]
        # HELPER: Routes predict_params and returns estimator_.predict output.
        params = process_routing(self, "predict", **predict_params)
        return self.estimator_.predict(X, **params.estimator.predict)

    def get_metadata_routing(self):
        # [FUNCTION DOC]
        # HELPER: Returns a router with self-request plus fit->fit and predict->predict
        # mapping to estimator.
        router = (
            MetadataRouter(owner=self)
            .add_self_request(self)
            .add(
                estimator=self.estimator,
                method_mapping=MethodMapping()
                .add(caller="fit", callee="fit")
                .add(caller="predict", callee="predict"),
            )
        )
        return router


class WeightedMetaClassifier(MetaEstimatorMixin, ClassifierMixin, BaseEstimator):
    # [CLASS DOC]
    # HELPER: Meta-classifier that consumes sample_weight in fit and routes fit metadata.
    """A meta-estimator which also consumes sample_weight itself in ``fit``."""

    def __init__(self, estimator, registry=None):
        # [FUNCTION DOC]
        # HELPER: Stores estimator and registry.
        self.estimator = estimator
        self.registry = registry

    def fit(self, X, y, sample_weight=None, **kwargs):
        # [FUNCTION DOC]
        # HELPER: Registers self, records sample_weight, routes kwargs and fits a clone into
        # estimator_; returns self.
        if self.registry is not None:
            self.registry.append(self)

        record_metadata(self, sample_weight=sample_weight)
        params = process_routing(self, "fit", sample_weight=sample_weight, **kwargs)
        self.estimator_ = clone(self.estimator).fit(X, y, **params.estimator.fit)
        return self

    def get_metadata_routing(self):
        # [FUNCTION DOC]
        # HELPER: Returns a router with self-request plus fit->fit mapping to estimator.
        router = (
            MetadataRouter(owner=self)
            .add_self_request(self)
            .add(
                estimator=self.estimator,
                method_mapping=MethodMapping().add(caller="fit", callee="fit"),
            )
        )
        return router


class MetaTransformer(MetaEstimatorMixin, TransformerMixin, BaseEstimator):
    # [CLASS DOC]
    # HELPER: Meta-transformer routing fit and transform metadata to a cloned transformer.
    """A simple meta-transformer."""

    def __init__(self, transformer):
        # [FUNCTION DOC]
        # HELPER: Stores transformer.
        self.transformer = transformer

    def fit(self, X, y=None, **fit_params):
        # [FUNCTION DOC]
        # HELPER: Routes fit_params and fits a clone into transformer_; returns self.
        params = process_routing(self, "fit", **fit_params)
        self.transformer_ = clone(self.transformer).fit(X, y, **params.transformer.fit)
        return self

    def transform(self, X, y=None, **transform_params):
        # [FUNCTION DOC]
        # HELPER: Routes transform_params and returns transformer_.transform(X); y ignored.
        params = process_routing(self, "transform", **transform_params)
        return self.transformer_.transform(X, **params.transformer.transform)

    def get_metadata_routing(self):
        # [FUNCTION DOC]
        # HELPER: Returns a router with fit->fit and transform->transform mapping.
        return MetadataRouter(owner=self).add(
            transformer=self.transformer,
            method_mapping=MethodMapping()
            .add(caller="fit", callee="fit")
            .add(caller="transform", callee="transform"),
        )
