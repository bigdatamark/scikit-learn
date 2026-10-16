# [MODULE DOC]
# Common tests for meta-estimators: (1) delegating meta-estimators expose a method iff
# their sub-estimator does, raise NotFittedError before fit and work after fit; (2)
# meta-estimators discovered via all_estimators delegate input validation to their
# inner estimator (checked by fitting with a text pipeline on a 1D list of strings).
# Import-time side effects: enumerates all estimators and constructs instances to
# build DATA_VALIDATION_META_ESTIMATORS, printing diagnostics to stdout.
"""Common tests for metaestimators"""

import functools
from contextlib import suppress
from inspect import signature

import numpy as np
import pytest

from sklearn.base import BaseEstimator, clone, is_regressor
from sklearn.datasets import make_classification
from sklearn.ensemble import BaggingClassifier
from sklearn.exceptions import NotFittedError
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import RFE, RFECV
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import MaxAbsScaler, StandardScaler
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.utils import all_estimators
from sklearn.utils._test_common.instance_generator import _construct_instances
from sklearn.utils._testing import SkipTest, set_random_state
from sklearn.utils.estimator_checks import (
    _enforce_estimator_tags_X,
    _enforce_estimator_tags_y,
)
from sklearn.utils.validation import check_is_fitted


class DelegatorData:
    # [CLASS DOC]
    # HELPER: Plain record describing one delegating meta-estimator under test.
    def __init__(
        self,
        name,
        construct,
        skip_methods=(),
        fit_args=make_classification(random_state=0),
    ):
        # [FUNCTION DOC]
        # HELPER: Stores the record fields as attributes.
        # Inputs: name - display name; construct - callable taking a sub-estimator and
        # returning the meta-estimator; skip_methods - method names not checked (implemented
        # by the meta-estimator itself); fit_args - (X, y) tuple, default a
        # make_classification dataset (seed 0) evaluated once at definition time.
        # Side effects: sets name, construct, fit_args, skip_methods on self.
        self.name = name
        self.construct = construct
        self.fit_args = fit_args
        self.skip_methods = skip_methods


# For the following meta estimators we check for the existence of relevant
# methods only if the sub estimator also contains them. Any methods that
# are implemented in the meta estimator themselves and are not dependent
# on the sub estimator are specified in the `skip_methods` parameter.
DELEGATING_METAESTIMATORS = [
    DelegatorData("Pipeline", lambda est: Pipeline([("est", est)])),
    DelegatorData(
        "GridSearchCV",
        lambda est: GridSearchCV(est, param_grid={"param": [5]}, cv=2),
        skip_methods=["score"],
    ),
    DelegatorData(
        "RandomizedSearchCV",
        lambda est: RandomizedSearchCV(
            est, param_distributions={"param": [5]}, cv=2, n_iter=1
        ),
        skip_methods=["score"],
    ),
    DelegatorData("RFE", RFE, skip_methods=["transform", "inverse_transform"]),
    DelegatorData(
        "RFECV", RFECV, skip_methods=["transform", "inverse_transform", "score"]
    ),
    DelegatorData(
        "BaggingClassifier",
        BaggingClassifier,
        skip_methods=[
            "transform",
            "inverse_transform",
            "score",
            "predict_proba",
            "predict_log_proba",
            "predict",
        ],
    ),
    DelegatorData(
        "SelfTrainingClassifier",
        lambda est: SelfTrainingClassifier(est),
        skip_methods=["transform", "inverse_transform", "predict_proba"],
    ),
]


def test_metaestimator_delegation():
    # [FUNCTION DOC]
    # For each entry of DELEGATING_METAESTIMATORS, with a dummy SubEstimator whose public
    # non-fit methods can be individually hidden: asserts the meta-estimator has each
    # non-skipped method, that calling it before fit raises NotFittedError (score is
    # called with X and y), that it runs after fit, and that hiding the method on the
    # sub-estimator also removes it from the meta-estimator.
    # Ensures specified metaestimators have methods iff subestimator does
    def hides(method):
        # [FUNCTION DOC]
        # HELPER: Decorator turning a method into a property that raises AttributeError when
        # the instance's hidden_method equals the method name, otherwise returns the method
        # bound to the instance (via functools.partial).
        # Inputs: method - function to wrap.
        # Outputs: property object.
        @property
        def wrapper(obj):
            # [FUNCTION DOC]
            # HELPER: Property getter for the hidden-able method.
            # Inputs: obj - SubEstimator instance.
            # Outputs: callable bound to obj.
            # Exceptions: AttributeError when obj.hidden_method names this method.
            if obj.hidden_method == method.__name__:
                raise AttributeError("%r is hidden" % obj.hidden_method)
            return functools.partial(method, obj)

        return wrapper

    class SubEstimator(BaseEstimator):
        # [CLASS DOC]
        # HELPER: Minimal estimator implementing every delegatable method; any one of them can
        # be hidden through hidden_method so hasattr returns False for it.
        def __init__(self, param=1, hidden_method=None):
            # [FUNCTION DOC]
            # HELPER: Stores param (dummy hyperparameter, default 1) and hidden_method (name of the
            # method to hide, default None).
            self.param = param
            self.hidden_method = hidden_method

        def fit(self, X, y=None, *args, **kwargs):
            # [FUNCTION DOC]
            # HELPER: Sets coef_ to the range of feature indices and classes_ to an empty list.
            # Outputs: True instead of self.
            # KNOWN ISSUE: returns True rather than self, unlike the estimator API; harmless here
            # because the return value is unused.
            self.coef_ = np.arange(X.shape[1])
            self.classes_ = []
            return True

        def _check_fit(self):
            # [FUNCTION DOC]
            # HELPER: Raises NotFittedError via check_is_fitted if fit has not been called.
            check_is_fitted(self)

        @hides
        def inverse_transform(self, X, *args, **kwargs):
            # [FUNCTION DOC]
            # HELPER: Checks fitted, returns X unchanged.
            self._check_fit()
            return X

        @hides
        def transform(self, X, *args, **kwargs):
            # [FUNCTION DOC]
            # HELPER: Checks fitted, returns X unchanged.
            self._check_fit()
            return X

        @hides
        def predict(self, X, *args, **kwargs):
            # [FUNCTION DOC]
            # HELPER: Checks fitted, returns an array of ones of length n_samples.
            self._check_fit()
            return np.ones(X.shape[0])

        @hides
        def predict_proba(self, X, *args, **kwargs):
            # [FUNCTION DOC]
            # HELPER: Checks fitted, returns an array of ones of length n_samples (not 2D
            # probabilities).
            self._check_fit()
            return np.ones(X.shape[0])

        @hides
        def predict_log_proba(self, X, *args, **kwargs):
            # [FUNCTION DOC]
            # HELPER: Checks fitted, returns an array of ones of length n_samples.
            self._check_fit()
            return np.ones(X.shape[0])

        @hides
        def decision_function(self, X, *args, **kwargs):
            # [FUNCTION DOC]
            # HELPER: Checks fitted, returns an array of ones of length n_samples.
            self._check_fit()
            return np.ones(X.shape[0])

        @hides
        def score(self, X, y, *args, **kwargs):
            # [FUNCTION DOC]
            # HELPER: Checks fitted, returns the constant score 1.0.
            self._check_fit()
            return 1.0

    methods = [
        k
        for k in SubEstimator.__dict__.keys()
        if not k.startswith("_") and not k.startswith("fit")
    ]
    methods.sort()

    for delegator_data in DELEGATING_METAESTIMATORS:
        delegate = SubEstimator()
        delegator = delegator_data.construct(delegate)
        for method in methods:
            if method in delegator_data.skip_methods:
                continue
            assert hasattr(delegate, method)
            assert hasattr(delegator, method), (
                "%s does not have method %r when its delegate does"
                % (
                    delegator_data.name,
                    method,
                )
            )
            # delegation before fit raises a NotFittedError
            if method == "score":
                with pytest.raises(NotFittedError):
                    getattr(delegator, method)(
                        delegator_data.fit_args[0], delegator_data.fit_args[1]
                    )
            else:
                with pytest.raises(NotFittedError):
                    getattr(delegator, method)(delegator_data.fit_args[0])

        delegator.fit(*delegator_data.fit_args)
        for method in methods:
            if method in delegator_data.skip_methods:
                continue
            # smoke test delegation
            if method == "score":
                getattr(delegator, method)(
                    delegator_data.fit_args[0], delegator_data.fit_args[1]
                )
            else:
                getattr(delegator, method)(delegator_data.fit_args[0])

        for method in methods:
            if method in delegator_data.skip_methods:
                continue
            delegate = SubEstimator(hidden_method=method)
            delegator = delegator_data.construct(delegate)
            assert not hasattr(delegate, method)
            assert not hasattr(delegator, method), (
                "%s has method %r when its delegate does not"
                % (
                    delegator_data.name,
                    method,
                )
            )


def _get_instance_with_pipeline(meta_estimator, init_params):
    # [FUNCTION DOC]
    # HELPER: Builds a new meta-estimator of the same class wired to text pipelines
    # (TfidfVectorizer followed by Ridge or LogisticRegression, depending on
    # is_regressor). Handles: single-estimator params (estimator/base_estimator/regressor;
    # SearchCV classes also get a 2-value param grid and n_iter=2 if accepted);
    # transformer_list (FeatureUnion of two Tfidf pipelines with scalers); estimators
    # (stacking/voting with two pipelines).
    # Inputs: meta_estimator - an instance; init_params - set of its constructor
    # parameter names.
    # Outputs: new estimator instance, or None if no case matches.
    """Given a single meta-estimator instance, generate an instance with a pipeline"""
    if {"estimator", "base_estimator", "regressor"} & init_params:
        if is_regressor(meta_estimator):
            estimator = make_pipeline(TfidfVectorizer(), Ridge())
            param_grid = {"ridge__alpha": [0.1, 1.0]}
        else:
            estimator = make_pipeline(TfidfVectorizer(), LogisticRegression())
            param_grid = {"logisticregression__C": [0.1, 1.0]}

        if init_params.intersection(
            {"param_grid", "param_distributions"}
        ):  # SearchCV estimators
            extra_params = {"n_iter": 2} if "n_iter" in init_params else {}
            return type(meta_estimator)(estimator, param_grid, **extra_params)
        else:
            return type(meta_estimator)(estimator)

    if "transformer_list" in init_params:
        # FeatureUnion
        transformer_list = [
            ("trans1", make_pipeline(TfidfVectorizer(), MaxAbsScaler())),
            (
                "trans2",
                make_pipeline(TfidfVectorizer(), StandardScaler(with_mean=False)),
            ),
        ]
        return type(meta_estimator)(transformer_list)

    if "estimators" in init_params:
        # stacking, voting
        if is_regressor(meta_estimator):
            estimator = [
                ("est1", make_pipeline(TfidfVectorizer(), Ridge(alpha=0.1))),
                ("est2", make_pipeline(TfidfVectorizer(), Ridge(alpha=1))),
            ]
        else:
            estimator = [
                (
                    "est1",
                    make_pipeline(TfidfVectorizer(), LogisticRegression(C=0.1)),
                ),
                ("est2", make_pipeline(TfidfVectorizer(), LogisticRegression(C=1))),
            ]
        return type(meta_estimator)(estimator)


def _generate_meta_estimator_instances_with_pipeline():
    # [FUNCTION DOC]
    # HELPER: Generator over all_estimators (sorted by name) whose constructors accept one
    # of estimator, base_estimator, regressor, transformer_list or estimators; for each
    # instance from _construct_instances (SkipTest suppressed) yields the pipeline-fed
    # variant from _get_instance_with_pipeline.
    # Outputs: yields estimator instances (possibly None).
    # Side effects: prints diagnostic information to stdout.
    # KNOWN ISSUE: docstring lists only estimator/base_estimator/estimators, but regressor
    # and transformer_list are also matched.
    """Generate instances of meta-estimators fed with a pipeline

    Are considered meta-estimators all estimators accepting one of "estimator",
    "base_estimator" or "estimators".
    """
    print("estimators: ", len(all_estimators()))
    for _, Estimator in sorted(all_estimators()):
        sig = set(signature(Estimator).parameters)

        print("\n", Estimator.__name__, sig)
        if not sig.intersection(
            {
                "estimator",
                "base_estimator",
                "regressor",
                "transformer_list",
                "estimators",
            }
        ):
            continue

        with suppress(SkipTest):
            for meta_estimator in _construct_instances(Estimator):
                print(meta_estimator)
                yield _get_instance_with_pipeline(meta_estimator, sig)


# TODO: remove data validation for the following estimators
# They should be able to work on any data and delegate data validation to
# their inner estimator(s).
DATA_VALIDATION_META_ESTIMATORS_TO_IGNORE = [
    "AdaBoostClassifier",
    "AdaBoostRegressor",
    "BaggingClassifier",
    "BaggingRegressor",
    "ClassifierChain",  # data validation is necessary
    "FrozenEstimator",  # this estimator cannot be tested like others.
    "IterativeImputer",
    "OneVsOneClassifier",  # input validation can't be avoided
    "RANSACRegressor",
    "RFE",
    "RFECV",
    "RegressorChain",  # data validation is necessary
    "SelfTrainingClassifier",
    "SequentialFeatureSelector",  # not applicable (2D data mandatory)
]

DATA_VALIDATION_META_ESTIMATORS = [
    est
    for est in _generate_meta_estimator_instances_with_pipeline()
    if est.__class__.__name__ not in DATA_VALIDATION_META_ESTIMATORS_TO_IGNORE
]


def _get_meta_estimator_id(estimator):
    # [FUNCTION DOC]
    # HELPER: pytest id function returning the estimator's class name.
    return estimator.__class__.__name__


@pytest.mark.parametrize(
    "estimator", DATA_VALIDATION_META_ESTIMATORS, ids=_get_meta_estimator_id
)
def test_meta_estimators_delegate_data_validation(estimator):
    # [FUNCTION DOC]
    # Verifies that the meta-estimator fits on a list of random string samples ("aa",
    # "bb", "cc", seed 0) with list targets (regression or 3-class) without raising a
    # validation error, i.e. validation is delegated to the inner text pipeline, and that
    # n_features_in_ is not set afterwards. The estimator is cloned first.
    # Check that meta-estimators delegate data validation to the inner
    # estimator(s).

    # clone to avoid side effects and ensure thread-safe test execution.
    estimator = clone(estimator)
    rng = np.random.RandomState(0)
    set_random_state(estimator)

    n_samples = 30
    X = rng.choice(np.array(["aa", "bb", "cc"], dtype=object), size=n_samples)

    if is_regressor(estimator):
        y = rng.normal(size=n_samples)
    else:
        y = rng.randint(3, size=n_samples)

    # We convert to lists to make sure it works on array-like
    X = _enforce_estimator_tags_X(estimator, X).tolist()
    y = _enforce_estimator_tags_y(estimator, y).tolist()

    # Calling fit should not raise any data validation exception since X is a
    # valid input datastructure for the first step of the pipeline passed as
    # base estimator to the meta estimator.
    estimator.fit(X, y)

    # n_features_in_ should not be defined since data is not tabular data.
    assert not hasattr(estimator, "n_features_in_")
