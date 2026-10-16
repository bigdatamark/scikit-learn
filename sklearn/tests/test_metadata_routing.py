# [MODULE DOC]
# Tests for the metadata routing utilities in sklearn.utils._metadata_requests and
# sklearn.utils.metadata_routing: request objects, routers, method mappings, the
# generated set_{method}_request methods, process_routing, error messages, cloning
# behavior and string representations. Uses the consumer/meta test estimators from
# sklearn.tests.metadata_routing_common. Import time: builds a fixed random dataset
# (seed 42) X (100x4), binary y, my_groups, my_weights and my_other_weights.
"""
Metadata Routing Utility Tests
"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import re

import numpy as np
import pytest

from sklearn import config_context
from sklearn.base import (
    BaseEstimator,
    clone,
)
from sklearn.exceptions import UnsetMetadataPassedError
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.tests.metadata_routing_common import (
    ConsumingClassifier,
    ConsumingRegressor,
    ConsumingTransformer,
    MetaRegressor,
    MetaTransformer,
    NonConsumingClassifier,
    WeightedMetaClassifier,
    WeightedMetaRegressor,
    _Registry,
    assert_request_equal,
    assert_request_is_empty,
    check_recorded_metadata,
)
from sklearn.utils import metadata_routing
from sklearn.utils._metadata_requests import (
    COMPOSITE_METHODS,
    METHODS,
    SIMPLE_METHODS,
    MethodMetadataRequest,
    MethodPair,
    _MetadataRequester,
    request_is_alias,
    request_is_valid,
)
from sklearn.utils.metadata_routing import (
    MetadataRequest,
    MetadataRouter,
    MethodMapping,
    _RoutingNotSupportedMixin,
    get_routing_for_object,
    process_routing,
)
from sklearn.utils.validation import check_is_fitted

rng = np.random.RandomState(42)
N, M = 100, 4
X = rng.rand(N, M)
y = rng.randint(0, 2, size=N)
my_groups = rng.randint(0, 10, size=N)
my_weights = rng.rand(N)
my_other_weights = rng.rand(N)


class SimplePipeline(BaseEstimator):
    # [CLASS DOC]
    # HELPER: Minimal router meta-estimator chaining transformers and a final predictor,
    # routing metadata to each step via process_routing. Steps are routed under the names
    # "step_<i>" and "predictor".
    """A very simple pipeline, assuming the last step is always a predictor.

    Parameters
    ----------
    steps : iterable of objects
        An iterable of transformers with the last step being a predictor.
    """

    def __init__(self, steps):
        # [FUNCTION DOC]
        # HELPER: Stores steps (list of transformers ending with a predictor), unvalidated.
        self.steps = steps

    def fit(self, X, y, **fit_params):
        # [FUNCTION DOC]
        # HELPER: Routes fit_params, then for each non-final step clones it, fits it with its
        # routed fit params, appends it to steps_ and transforms the running data with its
        # routed transform params; finally clones and fits the predictor on the transformed
        # data.
        # Outputs: self. Side effects: sets steps_ (list of fitted clones).
        # Exceptions: routing errors from process_routing for unrequested/invalid metadata.
        self.steps_ = []
        params = process_routing(self, "fit", **fit_params)
        X_transformed = X
        for i, step in enumerate(self.steps[:-1]):
            transformer = clone(step).fit(
                X_transformed, y, **params.get(f"step_{i}").fit
            )
            self.steps_.append(transformer)
            X_transformed = transformer.transform(
                X_transformed, **params.get(f"step_{i}").transform
            )

        self.steps_.append(
            clone(self.steps[-1]).fit(X_transformed, y, **params.predictor.fit)
        )
        return self

    def predict(self, X, **predict_params):
        # [FUNCTION DOC]
        # HELPER: Checks fitted, routes predict_params, applies the fitted transformers with
        # their routed transform params, and returns the last step's predict output.
        # KNOWN ISSUE: each transformer is applied to the original X instead of the running
        # X_transformed, so with more than one transformer only the last transform is
        # effectively used.
        check_is_fitted(self)
        X_transformed = X
        params = process_routing(self, "predict", **predict_params)
        for i, step in enumerate(self.steps_[:-1]):
            X_transformed = step.transform(X, **params.get(f"step_{i}").transform)

        return self.steps_[-1].predict(X_transformed, **params.predictor.predict)

    def get_metadata_routing(self):
        # [FUNCTION DOC]
        # HELPER: Builds a MetadataRouter: each non-final step gets fit->fit, fit->transform
        # and predict->transform mappings; the predictor gets fit->fit and predict->predict.
        # Outputs: MetadataRouter.
        router = MetadataRouter(owner=self)
        for i, step in enumerate(self.steps[:-1]):
            router.add(
                **{f"step_{i}": step},
                method_mapping=MethodMapping()
                .add(caller="fit", callee="fit")
                .add(caller="fit", callee="transform")
                .add(caller="predict", callee="transform"),
            )
        router.add(
            predictor=self.steps[-1],
            method_mapping=MethodMapping()
            .add(caller="fit", callee="fit")
            .add(caller="predict", callee="predict"),
        )
        return router


@config_context(enable_metadata_routing=True)
def test_assert_request_is_empty():
    # [FUNCTION DOC]
    # Verifies assert_request_is_empty: passes for fresh requests and None-alias requests,
    # fails once a string alias or True is set, honors the exclude argument (single name or
    # list), and passes for a router whose children have only default requests.
    requests = MetadataRequest(owner="test")
    assert_request_is_empty(requests)

    requests.fit.add_request(param="foo", alias=None)
    # this should still work, since None is the default value
    assert_request_is_empty(requests)

    requests.fit.add_request(param="bar", alias="value")
    with pytest.raises(AssertionError):
        # now requests is no more empty
        assert_request_is_empty(requests)

    # but one can exclude a method
    assert_request_is_empty(requests, exclude="fit")

    requests.score.add_request(param="carrot", alias=True)
    with pytest.raises(AssertionError):
        # excluding `fit` is not enough
        assert_request_is_empty(requests, exclude="fit")

    # and excluding both fit and score would avoid an exception
    assert_request_is_empty(requests, exclude=["fit", "score"])

    # test if a router is empty
    assert_request_is_empty(
        MetadataRouter(owner="test")
        .add_self_request(WeightedMetaRegressor(estimator=None))
        .add(
            estimator=ConsumingRegressor(),
            method_mapping=MethodMapping().add(caller="fit", callee="fit"),
        )
    )


@pytest.mark.parametrize(
    "estimator",
    [
        ConsumingClassifier(registry=_Registry()),
        ConsumingRegressor(registry=_Registry()),
        ConsumingTransformer(registry=_Registry()),
        WeightedMetaClassifier(estimator=ConsumingClassifier(), registry=_Registry()),
        WeightedMetaRegressor(estimator=ConsumingRegressor(), registry=_Registry()),
    ],
)
@config_context(enable_metadata_routing=True)
def test_estimator_puts_self_in_registry(estimator):
    # [FUNCTION DOC]
    # Verifies each consuming/meta test estimator records itself in its registry on fit
    # (clone is fitted on module X, y).
    """Check that an estimator puts itself in the registry upon fit."""
    estimator = clone(estimator).fit(X, y)
    assert estimator in estimator.registry


@pytest.mark.parametrize(
    "val, res",
    [
        (False, False),
        (True, False),
        (None, False),
        ("$UNUSED$", False),
        ("$WARN$", False),
        ("invalid-input", False),
        ("valid_arg", True),
    ],
)
@config_context(enable_metadata_routing=True)
def test_request_type_is_alias(val, res):
    # [FUNCTION DOC]
    # Verifies request_is_alias returns True only for valid identifier strings, not for
    # booleans, None, the UNUSED/WARN sentinels or invalid strings.
    # Test request_is_alias
    assert request_is_alias(val) == res


@pytest.mark.parametrize(
    "val, res",
    [
        (False, True),
        (True, True),
        (None, True),
        ("$UNUSED$", True),
        ("$WARN$", True),
        ("invalid-input", False),
        ("alias_arg", False),
    ],
)
@config_context(enable_metadata_routing=True)
def test_request_type_is_valid(val, res):
    # [FUNCTION DOC]
    # Verifies request_is_valid accepts booleans, None and the UNUSED/WARN sentinels, and
    # rejects arbitrary strings (including aliases).
    # Test request_is_valid
    assert request_is_valid(val) == res


@config_context(enable_metadata_routing=True)
def test_default_requests():
    # [FUNCTION DOC]
    # Verifies default requests: a class-level __metadata_request__fit overrides defaults;
    # NonConsumingClassifier has no requests; ConsumingTransformer and ConsumingClassifier
    # default every metadata to None, which counts as empty.
    class OddEstimator(BaseEstimator):
        # [CLASS DOC]
        # HELPER: Estimator whose class attribute requests sample_weight for fit by default.
        __metadata_request__fit = {
            # set a different default request
            "sample_weight": True
        }

        def fit(self, X, y=None):
            # [FUNCTION DOC]
            # HELPER: Stub fit returning self.
            return self  # pragma: no cover

    odd_request = get_routing_for_object(OddEstimator())
    assert odd_request.fit.requests == {"sample_weight": True}

    # check other test estimators
    assert not len(get_routing_for_object(NonConsumingClassifier()).fit.requests)
    assert_request_is_empty(NonConsumingClassifier().get_metadata_routing())

    trs_request = get_routing_for_object(ConsumingTransformer())
    assert trs_request.fit.requests == {
        "sample_weight": None,
        "metadata": None,
    }
    assert trs_request.transform.requests == {"metadata": None, "sample_weight": None}
    assert_request_is_empty(trs_request)

    est_request = get_routing_for_object(ConsumingClassifier())
    assert est_request.fit.requests == {
        "sample_weight": None,
        "metadata": None,
    }
    assert_request_is_empty(est_request)


@config_context(enable_metadata_routing=True)
def test_default_request_override():
    # [FUNCTION DOC]
    # Non-regression (issue 28430): a subclass's __metadata_request__split override wins
    # over the parent's default regardless of whether the class name starts lower- or
    # upper-case (ASCII ordering of MRO handling).
    """Test that default requests are correctly overridden regardless of the ASCII order
    of the class names, hence testing small and capital letter class name starts.
    Non-regression test for https://github.com/scikit-learn/scikit-learn/issues/28430
    """

    class Base(BaseEstimator):
        # [CLASS DOC]
        # HELPER: Splitter base whose default request is groups=True.
        __metadata_request__split = {"groups": True}

        def split(self, X, y=None):
            # [FUNCTION DOC]
            # HELPER: Stub split; does nothing.
            pass  # pragma: no cover

    class class_1(Base):
        # [CLASS DOC]
        # HELPER: Lower-case named subclass aliasing groups to "sample_domain".
        __metadata_request__split = {"groups": "sample_domain"}

        def split(self, X, y=None):
            # [FUNCTION DOC]
            # HELPER: Stub split; does nothing.
            pass  # pragma: no cover

    class Class_1(Base):
        # [CLASS DOC]
        # HELPER: Upper-case named subclass aliasing groups to "sample_domain".
        __metadata_request__split = {"groups": "sample_domain"}

        def split(self, X, y=None):
            # [FUNCTION DOC]
            # HELPER: Stub split; does nothing.
            pass  # pragma: no cover

    assert_request_equal(
        class_1()._get_metadata_request(), {"split": {"groups": "sample_domain"}}
    )
    assert_request_equal(
        Class_1()._get_metadata_request(), {"split": {"groups": "sample_domain"}}
    )


@config_context(enable_metadata_routing=True)
def test_process_routing_invalid_method():
    # [FUNCTION DOC]
    # Verifies process_routing raises TypeError for an unknown method name.
    with pytest.raises(TypeError, match="Can only route and process input"):
        process_routing(ConsumingClassifier(), "invalid_method", groups=my_groups)


@config_context(enable_metadata_routing=True)
def test_process_routing_invalid_object():
    # [FUNCTION DOC]
    # Verifies process_routing raises AttributeError for an object implementing neither
    # get_metadata_routing nor _get_metadata_request.
    class InvalidObject:
        # [CLASS DOC]
        # HELPER: Empty class with no routing support.
        pass

    with pytest.raises(AttributeError, match="either implement the routing method"):
        process_routing(InvalidObject(), "fit", groups=my_groups)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("default", [None, "default", []])
@config_context(enable_metadata_routing=True)
def test_process_routing_empty_params_get_with_default(method, default):
    # [FUNCTION DOC]
    # Verifies that with no params, process_routing's result returns for any method a dict
    # keyed by all METHODS, and that get with any default gives the same dict.
    empty_params = {}
    routed_params = process_routing(ConsumingClassifier(), "fit", **empty_params)

    # Behaviour should be an empty dictionary returned for each method when retrieved.
    params_for_method = routed_params[method]
    assert isinstance(params_for_method, dict)
    assert set(params_for_method.keys()) == set(METHODS)

    # No default to `get` should be equivalent to the default
    default_params_for_method = routed_params.get(method, default=default)
    assert default_params_for_method == params_for_method


@config_context(enable_metadata_routing=True)
def test_simple_metadata_routing():
    # [FUNCTION DOC]
    # Verifies basic routing in WeightedMetaClassifier: no error when the child does not
    # consume; error when the child accepts sample_weight without an explicit request;
    # no forwarding when the child sets False; forwarding when set to True or an alias
    # (checked via check_recorded_metadata on estimator_).
    # Tests that metadata is properly routed

    # The underlying estimator doesn't accept or request metadata
    clf = WeightedMetaClassifier(estimator=NonConsumingClassifier())
    clf.fit(X, y)

    # Meta-estimator consumes sample_weight, but doesn't forward it to the underlying
    # estimator
    clf = WeightedMetaClassifier(estimator=NonConsumingClassifier())
    clf.fit(X, y, sample_weight=my_weights)

    # If the estimator accepts the metadata but doesn't explicitly say it doesn't
    # need it, there's an error
    clf = WeightedMetaClassifier(estimator=ConsumingClassifier())
    err_message = (
        "[sample_weight] are passed but are not explicitly set as requested or"
        " not requested for ConsumingClassifier.fit"
    )
    with pytest.raises(ValueError, match=re.escape(err_message)):
        clf.fit(X, y, sample_weight=my_weights)

    # Explicitly saying the estimator doesn't need it, makes the error go away,
    # because in this case `WeightedMetaClassifier` consumes `sample_weight`. If
    # there was no consumer of sample_weight, passing it would result in an
    # error.
    clf = WeightedMetaClassifier(
        estimator=ConsumingClassifier().set_fit_request(sample_weight=False)
    )
    # this doesn't raise since WeightedMetaClassifier itself is a consumer,
    # and passing metadata to the consumer directly is fine regardless of its
    # metadata_request values.
    clf.fit(X, y, sample_weight=my_weights)
    check_recorded_metadata(clf.estimator_, method="fit", parent="fit")

    # Requesting a metadata will make the meta-estimator forward it correctly
    clf = WeightedMetaClassifier(
        estimator=ConsumingClassifier().set_fit_request(sample_weight=True)
    )
    clf.fit(X, y, sample_weight=my_weights)
    check_recorded_metadata(
        clf.estimator_, method="fit", parent="fit", sample_weight=my_weights
    )

    # And requesting it with an alias
    clf = WeightedMetaClassifier(
        estimator=ConsumingClassifier().set_fit_request(
            sample_weight="alternative_weight"
        )
    )
    clf.fit(X, y, alternative_weight=my_weights)
    check_recorded_metadata(
        clf.estimator_, method="fit", parent="fit", sample_weight=my_weights
    )


@config_context(enable_metadata_routing=True)
def test_nested_routing():
    # [FUNCTION DOC]
    # Verifies nested routing through SimplePipeline -> MetaTransformer/WeightedMetaRegressor
    # -> consumers, including aliases at outer and inner levels, by checking recorded
    # metadata for fit, transform and predict.
    # check if metadata is routed in a nested routing situation.
    pipeline = SimplePipeline(
        [
            MetaTransformer(
                transformer=ConsumingTransformer()
                .set_fit_request(metadata=True, sample_weight=False)
                .set_transform_request(sample_weight=True, metadata=False)
            ),
            WeightedMetaRegressor(
                estimator=ConsumingRegressor()
                .set_fit_request(sample_weight="inner_weights", metadata=False)
                .set_predict_request(sample_weight=False)
            ).set_fit_request(sample_weight="outer_weights"),
        ]
    )
    w1, w2, w3 = [1], [2], [3]
    pipeline.fit(
        X, y, metadata=my_groups, sample_weight=w1, outer_weights=w2, inner_weights=w3
    )
    check_recorded_metadata(
        pipeline.steps_[0].transformer_,
        method="fit",
        parent="fit",
        metadata=my_groups,
    )
    check_recorded_metadata(
        pipeline.steps_[0].transformer_,
        method="transform",
        parent="fit",
        sample_weight=w1,
    )
    check_recorded_metadata(
        pipeline.steps_[1], method="fit", parent="fit", sample_weight=w2
    )
    check_recorded_metadata(
        pipeline.steps_[1].estimator_, method="fit", parent="fit", sample_weight=w3
    )

    pipeline.predict(X, sample_weight=w3)
    check_recorded_metadata(
        pipeline.steps_[1].estimator_,
        method="predict",
        parent="predict",
    )


@config_context(enable_metadata_routing=True)
def test_nested_routing_conflict():
    # [FUNCTION DOC]
    # Verifies a ValueError reporting a conflict when a meta-estimator aliases
    # sample_weight while its child requests the same name sample_weight=True.
    # check if an error is raised if there's a conflict between keys
    pipeline = SimplePipeline(
        [
            MetaTransformer(
                transformer=ConsumingTransformer()
                .set_fit_request(metadata=True, sample_weight=False)
                .set_transform_request(sample_weight=True)
            ),
            WeightedMetaRegressor(
                estimator=ConsumingRegressor().set_fit_request(sample_weight=True)
            ).set_fit_request(sample_weight="outer_weights"),
        ]
    )
    w1, w2 = [1], [2]
    with pytest.raises(
        ValueError,
        match=(
            re.escape(
                "In WeightedMetaRegressor, there is a conflict on sample_weight between"
                " what is requested for this estimator and what is requested by its"
                " children. You can resolve this conflict by using an alias for the"
                " child estimators' requested metadata."
            )
        ),
    ):
        pipeline.fit(X, y, metadata=my_groups, sample_weight=w1, outer_weights=w2)


@config_context(enable_metadata_routing=True)
def test_invalid_metadata():
    # [FUNCTION DOC]
    # Verifies TypeError "unexpected argument(s)" when transform receives a metadata no
    # estimator requests, both for an unknown name and for one explicitly set to False.
    # check that passing wrong metadata raises an error
    trs = MetaTransformer(
        transformer=ConsumingTransformer().set_transform_request(sample_weight=True)
    )
    with pytest.raises(
        TypeError,
        match=(re.escape("transform got unexpected argument(s) {'other_param'}")),
    ):
        trs.fit(X, y).transform(X, other_param=my_weights)

    # passing a metadata which is not requested by any estimator should also raise
    trs = MetaTransformer(
        transformer=ConsumingTransformer().set_transform_request(sample_weight=False)
    )
    with pytest.raises(
        TypeError,
        match=(re.escape("transform got unexpected argument(s) {'sample_weight'}")),
    ):
        trs.fit(X, y).transform(X, sample_weight=my_weights)


@config_context(enable_metadata_routing=True)
def test_get_metadata_routing():
    # [FUNCTION DOC]
    # Verifies get_metadata_routing reflects class-level default requests of a
    # _MetadataRequester and updates after set_score_request (alias) and set_fit_request.
    class TestDefaults(_MetadataRequester):
        # [CLASS DOC]
        # HELPER: Requester with class-level default requests for fit, score and predict.
        __metadata_request__fit = {
            "sample_weight": None,
            "my_other_param": None,
        }
        __metadata_request__score = {
            "sample_weight": None,
            "my_param": True,
            "my_other_param": None,
        }
        __metadata_request__predict = {"my_param": True}

        def fit(self, X, y=None):
            # [FUNCTION DOC]
            # HELPER: Stub fit returning self.
            return self  # pragma: no cover

        def score(self, X, y=None):
            # [FUNCTION DOC]
            # HELPER: Stub score; does nothing.
            pass  # pragma: no cover

        def predict(self, X):
            # [FUNCTION DOC]
            # HELPER: Stub predict; does nothing.
            pass  # pragma: no cover

    expected = {
        "score": {
            "my_param": True,
            "my_other_param": None,
            "sample_weight": None,
        },
        "fit": {
            "my_other_param": None,
            "sample_weight": None,
        },
        "predict": {"my_param": True},
    }
    assert_request_equal(TestDefaults().get_metadata_routing(), expected)

    est = TestDefaults().set_score_request(my_param="other_param")
    expected = {
        "score": {
            "my_param": "other_param",
            "my_other_param": None,
            "sample_weight": None,
        },
        "fit": {
            "my_other_param": None,
            "sample_weight": None,
        },
        "predict": {"my_param": True},
    }
    assert_request_equal(est.get_metadata_routing(), expected)

    est = TestDefaults().set_fit_request(sample_weight=True)
    expected = {
        "score": {
            "my_param": True,
            "my_other_param": None,
            "sample_weight": None,
        },
        "fit": {
            "my_other_param": None,
            "sample_weight": True,
        },
        "predict": {"my_param": True},
    }
    assert_request_equal(est.get_metadata_routing(), expected)


@config_context(enable_metadata_routing=True)
def test_setting_default_requests():
    # [FUNCTION DOC]
    # Verifies how default fit requests are derived from signatures and class attributes:
    # explicit request for a non-signature param, override to True, implicit None from the
    # signature, and removal via UNUSED; other methods stay empty.
    # Test _get_default_requests method
    test_cases = dict()

    class ExplicitRequest(BaseEstimator):
        # [CLASS DOC]
        # HELPER: Requests prop via class attribute although fit only takes **kwargs.
        # `fit` doesn't accept `props` explicitly, but we want to request it
        __metadata_request__fit = {"prop": None}

        def fit(self, X, y, **kwargs):
            # [FUNCTION DOC]
            # HELPER: Stub fit returning self.
            return self

    test_cases[ExplicitRequest] = {"prop": None}

    class ExplicitRequestOverwrite(BaseEstimator):
        # [CLASS DOC]
        # HELPER: Overrides the signature-derived default of prop to True.
        # `fit` explicitly accepts `props`, but we want to change the default
        # request value from None to True
        __metadata_request__fit = {"prop": True}

        def fit(self, X, y, prop=None, **kwargs):
            # [FUNCTION DOC]
            # HELPER: Stub fit returning self.
            return self

    test_cases[ExplicitRequestOverwrite] = {"prop": True}

    class ImplicitRequest(BaseEstimator):
        # [CLASS DOC]
        # HELPER: Gets prop requested implicitly from the fit signature.
        # `fit` requests `prop` and the default None should be used
        def fit(self, X, y, prop=None, **kwargs):
            # [FUNCTION DOC]
            # HELPER: Stub fit returning self.
            return self

    test_cases[ImplicitRequest] = {"prop": None}

    class ImplicitRequestRemoval(BaseEstimator):
        # [CLASS DOC]
        # HELPER: Removes the signature-derived prop request using UNUSED.
        # `fit` (in this class or a parent) requests `prop`, but we don't want
        # it requested at all.
        __metadata_request__fit = {"prop": metadata_routing.UNUSED}

        def fit(self, X, y, prop=None, **kwargs):
            # [FUNCTION DOC]
            # HELPER: Stub fit returning self.
            return self

    test_cases[ImplicitRequestRemoval] = {}

    for Klass, requests in test_cases.items():
        assert get_routing_for_object(Klass()).fit.requests == requests
        assert_request_is_empty(Klass().get_metadata_routing(), exclude="fit")
        Klass().fit(None, None)  # for coverage


@config_context(enable_metadata_routing=True)
def test_removing_non_existing_param_raises():
    # [FUNCTION DOC]
    # Verifies that marking a non-existent param as UNUSED raises ValueError when the
    # routing is requested.
    """Test that removing a metadata using UNUSED which doesn't exist raises."""

    class InvalidRequestRemoval(BaseEstimator):
        # [CLASS DOC]
        # HELPER: Tries to remove prop which fit does not accept.
        # `fit` (in this class or a parent) requests `prop`, but we don't want
        # it requested at all.
        __metadata_request__fit = {"prop": metadata_routing.UNUSED}

        def fit(self, X, y, **kwargs):
            # [FUNCTION DOC]
            # HELPER: Stub fit returning self.
            return self

    with pytest.raises(ValueError, match="Trying to remove parameter"):
        InvalidRequestRemoval().get_metadata_routing()


def test_get_class_level_metadata_request_values():
    # [FUNCTION DOC]
    # Verifies _get_class_level_metadata_request_values: signature sniffing of a class
    # method, ignore_params filtering, inspecting an explicit callable (first argument
    # skipped), and empty result for a missing method.
    """Test `_get_class_level_metadata_request_values`, which infers metadata
    requests from callables; used for class methods in consumers and by scorers
    for custom `score_func`s.
    """

    class Dummy(BaseEstimator):
        # [CLASS DOC]
        # HELPER: Estimator whose fit accepts sample_weight and extra.
        def fit(self, X, y, sample_weight=None, extra=None):
            # [FUNCTION DOC]
            # HELPER: Stub fit returning self.
            return self

    # Baseline: sniff the class method's signature.
    assert Dummy._get_class_level_metadata_request_values("fit") == {
        "sample_weight": None,
        "extra": None,
    }

    # `ignore_params` filters names out of the result.
    assert Dummy._get_class_level_metadata_request_values(
        "fit", ignore_params={"sample_weight"}
    ) == {"extra": None}

    # `method` lets us inspect a different callable; first arg is auto-skipped.
    def score_func(y_true, y_pred, sample_weight=None):
        # [FUNCTION DOC]
        # HELPER: Dummy scoring function signature; returns 0.
        return 0  # pragma: no cover

    assert Dummy._get_class_level_metadata_request_values(
        "score", method=score_func, ignore_params={"y_pred"}
    ) == {"sample_weight": None}

    # No matching class method and no `method` callable -> empty dict.
    assert Dummy._get_class_level_metadata_request_values("predict") == {}


@config_context(enable_metadata_routing=True)
def test_method_metadata_request():
    # [FUNCTION DOC]
    # Verifies MethodMetadataRequest.add_request: non-string/bool alias raises ValueError;
    # None/False/True stored as given; alias equal to the param name collapses to True;
    # other strings stored as alias; _get_param_names returns names or aliases.
    mmr = MethodMetadataRequest(owner="test", method="fit")

    with pytest.raises(ValueError, match="The alias you're setting for"):
        mmr.add_request(param="foo", alias=1.4)

    mmr.add_request(param="foo", alias=None)
    assert mmr.requests == {"foo": None}
    mmr.add_request(param="foo", alias=False)
    assert mmr.requests == {"foo": False}
    mmr.add_request(param="foo", alias=True)
    assert mmr.requests == {"foo": True}
    mmr.add_request(param="foo", alias="foo")
    assert mmr.requests == {"foo": True}
    mmr.add_request(param="foo", alias="bar")
    assert mmr.requests == {"foo": "bar"}
    assert mmr._get_param_names(return_alias=False) == {"foo"}
    assert mmr._get_param_names(return_alias=True) == {"bar"}


@config_context(enable_metadata_routing=True)
def test_get_routing_for_object():
    # [FUNCTION DOC]
    # Verifies get_routing_for_object returns empty requests for None and arbitrary
    # objects, a copy of a MetadataRequest passed in, and the class-level requests of an
    # estimator.
    class Consumer(BaseEstimator):
        # [CLASS DOC]
        # HELPER: Estimator with a class-level fit request for prop.
        __metadata_request__fit = {"prop": None}

        def fit(self, X, y=None):
            # [FUNCTION DOC]
            # HELPER: Stub fit returning self.
            return self  # pragma: no cover

    assert_request_is_empty(get_routing_for_object(None))
    assert_request_is_empty(get_routing_for_object(object()))

    mr = MetadataRequest(owner="test")
    mr.fit.add_request(param="foo", alias="bar")
    mr_factory = get_routing_for_object(mr)
    assert_request_is_empty(mr_factory, exclude="fit")
    assert mr_factory.fit.requests == {"foo": "bar"}

    mr = get_routing_for_object(Consumer())
    assert_request_is_empty(mr, exclude="fit")
    assert mr.fit.requests == {"prop": None}


@config_context(enable_metadata_routing=True)
def test_metadata_request_consumes_method():
    # [FUNCTION DOC]
    # Verifies MetadataRequest.consumes returns the subset of given params consumed for a
    # method, using aliases when set.
    """Test that MetadataRequest().consumes() method works as expected."""
    request = MetadataRequest(owner="test")
    assert request.consumes(method="fit", params={"foo"}) == set()

    request = MetadataRequest(owner="test")
    request.fit.add_request(param="foo", alias=True)
    assert request.consumes(method="fit", params={"foo"}) == {"foo"}

    request = MetadataRequest(owner="test")
    request.fit.add_request(param="foo", alias="bar")
    assert request.consumes(method="fit", params={"bar", "foo"}) == {"bar"}


@config_context(enable_metadata_routing=True)
def test_metadata_router_consumes_method():
    # [FUNCTION DOC]
    # Verifies MetadataRouter.consumes on WeightedMetaRegressor routers: with a plain
    # request and with an alias, only consumed names are returned.
    """Test that MetadataRouter().consumes method works as expected."""
    # having it here instead of parametrizing the test since `set_fit_request`
    # is not available while collecting the tests.
    cases = [
        (
            WeightedMetaRegressor(
                estimator=ConsumingRegressor().set_fit_request(sample_weight=True)
            ),
            {"sample_weight"},
            {"sample_weight"},
        ),
        (
            WeightedMetaRegressor(
                estimator=ConsumingRegressor().set_fit_request(
                    sample_weight="my_weights"
                )
            ),
            {"my_weights", "sample_weight"},
            {"my_weights"},
        ),
    ]

    for obj, input, output in cases:
        assert obj.get_metadata_routing().consumes(method="fit", params=input) == output


@config_context(enable_metadata_routing=True)
def test_metaestimator_warnings():
    # [FUNCTION DOC]
    # Verifies that a meta-estimator whose fit request default is WARN emits the
    # "Support for ... has recently been added" UserWarning when sample_weight is passed.
    class WeightedMetaRegressorWarn(WeightedMetaRegressor):
        # [CLASS DOC]
        # HELPER: WeightedMetaRegressor with sample_weight default request WARN.
        __metadata_request__fit = {"sample_weight": metadata_routing.WARN}

    with pytest.warns(
        UserWarning, match="Support for .* has recently been added to .* class"
    ):
        WeightedMetaRegressorWarn(
            estimator=LinearRegression().set_fit_request(sample_weight=False)
        ).fit(X, y, sample_weight=my_weights)


@config_context(enable_metadata_routing=True)
def test_estimator_warnings():
    # [FUNCTION DOC]
    # Verifies the same UserWarning when a consumer (child) has a WARN default request and
    # sample_weight is passed through MetaRegressor.
    class ConsumingRegressorWarn(ConsumingRegressor):
        # [CLASS DOC]
        # HELPER: ConsumingRegressor with sample_weight default request WARN.
        __metadata_request__fit = {"sample_weight": metadata_routing.WARN}

    with pytest.warns(
        UserWarning, match="Support for .* has recently been added to .* class"
    ):
        MetaRegressor(estimator=ConsumingRegressorWarn()).fit(
            X, y, sample_weight=my_weights
        )


@config_context(enable_metadata_routing=True)
@pytest.mark.parametrize(
    "obj, string",
    [
        (
            MethodMetadataRequest(owner="test", method="fit").add_request(
                param="foo", alias="bar"
            ),
            "{'foo': 'bar'}",
        ),
        (
            MetadataRequest(owner="test"),
            "{}",
        ),
        (
            MetadataRouter(owner="test").add(
                estimator=ConsumingRegressor(),
                method_mapping=MethodMapping().add(caller="predict", callee="predict"),
            ),
            (
                "{'estimator': {'mapping': [{'caller': 'predict', 'callee':"
                " 'predict'}], 'router': {'fit': {'sample_weight': None, 'metadata':"
                " None}, 'partial_fit': {'sample_weight': None, 'metadata': None},"
                " 'predict': {'sample_weight': None, 'metadata': None}, 'score':"
                " {'sample_weight': None, 'metadata': None}}}}"
            ),
        ),
    ],
)
@config_context(enable_metadata_routing=True)
def test_string_representations(obj, string):
    # [FUNCTION DOC]
    # Verifies str() of a MethodMetadataRequest, empty MetadataRequest and a MetadataRouter
    # matches the expected dict-like strings.
    # KNOWN ISSUE: the config_context decorator is applied twice (above and below the
    # parametrize decorator); redundant but harmless.
    assert str(obj) == string


@pytest.mark.parametrize(
    "obj, method, inputs, err_cls, err_msg",
    [
        (
            MethodMapping(),
            "add",
            {"caller": "fit", "callee": "invalid"},
            ValueError,
            "Given callee",
        ),
        (
            MethodMapping(),
            "add",
            {"caller": "invalid", "callee": "fit"},
            ValueError,
            "Given caller",
        ),
        (
            MetadataRouter(owner="test"),
            "add_self_request",
            {"obj": MetadataRouter(owner="test")},
            ValueError,
            "Given `obj` is neither a `MetadataRequest` nor does it implement",
        ),
        (
            ConsumingClassifier(),
            "set_fit_request",
            {"invalid": True},
            TypeError,
            "Unexpected args",
        ),
    ],
)
@config_context(enable_metadata_routing=True)
def test_validations(obj, method, inputs, err_cls, err_msg):
    # [FUNCTION DOC]
    # Verifies invalid inputs raise: MethodMapping.add with bad callee/caller, add_self_request
    # with a router, and set_fit_request with an unknown param.
    with pytest.raises(err_cls, match=err_msg):
        getattr(obj, method)(**inputs)


@config_context(enable_metadata_routing=True)
def test_methodmapping():
    # [FUNCTION DOC]
    # Verifies MethodMapping iteration order yields (caller, callee) tuples, that every
    # METHODS pair can be added and stored as MethodPair, and its repr.
    mm = (
        MethodMapping()
        .add(caller="fit", callee="transform")
        .add(caller="fit", callee="fit")
    )

    mm_list = list(mm)
    assert mm_list[0] == ("fit", "transform")
    assert mm_list[1] == ("fit", "fit")

    mm = MethodMapping()
    for method in METHODS:
        mm.add(caller=method, callee=method)
        assert MethodPair(method, method) in mm._routes
    assert len(mm._routes) == len(METHODS)

    mm = MethodMapping().add(caller="score", callee="score")
    assert repr(mm) == "[{'caller': 'score', 'callee': 'score'}]"


@config_context(enable_metadata_routing=True)
def test_metadatarouter_add_self_request():
    # [FUNCTION DOC]
    # Verifies add_self_request stores a copy of a MetadataRequest, of an estimator's
    # routing, and only the consumer part (_get_metadata_request) of a consumer+router.
    # adding a MetadataRequest as `self` adds a copy
    request = MetadataRequest(owner="nested")
    request.fit.add_request(param="param", alias=True)
    router = MetadataRouter(owner="test").add_self_request(request)
    assert str(router._self_request) == str(request)
    # should be a copy, not the same object
    assert router._self_request is not request

    # one can add an estimator as self
    est = ConsumingRegressor().set_fit_request(sample_weight="my_weights")
    router = MetadataRouter(owner="test").add_self_request(obj=est)
    assert str(router._self_request) == str(est.get_metadata_routing())
    assert router._self_request is not est.get_metadata_routing()

    # adding a consumer+router as self should only add the consumer part
    est = WeightedMetaRegressor(
        estimator=ConsumingRegressor().set_fit_request(sample_weight="nested_weights")
    )
    router = MetadataRouter(owner="test").add_self_request(obj=est)
    # _get_metadata_request() returns the consumer part of the requests
    assert str(router._self_request) == str(est._get_metadata_request())
    # get_metadata_routing() returns the complete request set, consumer and
    # router included.
    assert str(router._self_request) != str(est.get_metadata_routing())
    # it should be a copy, not the same object
    assert router._self_request is not est._get_metadata_request()


@config_context(enable_metadata_routing=True)
def test_metadata_routing_add():
    # [FUNCTION DOC]
    # Verifies MetadataRouter.add with keyword order variations produces the expected
    # string representation for fit->fit and fit->score mappings.
    # adding one with a string `method_mapping`
    router = MetadataRouter(owner="test").add(
        est=ConsumingRegressor().set_fit_request(sample_weight="weights"),
        method_mapping=MethodMapping().add(caller="fit", callee="fit"),
    )
    assert (
        str(router)
        == "{'est': {'mapping': [{'caller': 'fit', 'callee': 'fit'}], 'router': {'fit':"
        " {'sample_weight': 'weights', 'metadata': None}, 'partial_fit':"
        " {'sample_weight': None, 'metadata': None}, 'predict': {'sample_weight':"
        " None, 'metadata': None}, 'score': {'sample_weight': None, 'metadata':"
        " None}}}}"
    )

    # adding one with an instance of MethodMapping
    router = MetadataRouter(owner="test").add(
        method_mapping=MethodMapping().add(caller="fit", callee="score"),
        est=ConsumingRegressor().set_score_request(sample_weight=True),
    )
    assert (
        str(router)
        == "{'est': {'mapping': [{'caller': 'fit', 'callee': 'score'}], 'router':"
        " {'fit': {'sample_weight': None, 'metadata': None}, 'partial_fit':"
        " {'sample_weight': None, 'metadata': None}, 'predict': {'sample_weight':"
        " None, 'metadata': None}, 'score': {'sample_weight': True, 'metadata':"
        " None}}}}"
    )


@config_context(enable_metadata_routing=True)
def test_metadata_routing_get_param_names():
    # [FUNCTION DOC]
    # Verifies MetadataRouter._get_param_names for combinations of return_alias and
    # ignore_self_request on a router with a self request and an aliased child.
    router = (
        MetadataRouter(owner="test")
        .add_self_request(
            WeightedMetaRegressor(estimator=ConsumingRegressor()).set_fit_request(
                sample_weight="self_weights"
            )
        )
        .add(
            trs=ConsumingTransformer().set_fit_request(
                sample_weight="transform_weights"
            ),
            method_mapping=MethodMapping().add(caller="fit", callee="fit"),
        )
    )

    assert (
        str(router)
        == "{'$self_request': {'fit': {'sample_weight': 'self_weights'}, 'score':"
        " {'sample_weight': None}}, 'trs': {'mapping': [{'caller': 'fit', 'callee':"
        " 'fit'}], 'router': {'fit': {'sample_weight': 'transform_weights',"
        " 'metadata': None}, 'transform': {'sample_weight': None, 'metadata': None},"
        " 'inverse_transform': {'sample_weight': None, 'metadata': None}}}}"
    )

    assert router._get_param_names(
        method="fit", return_alias=True, ignore_self_request=False
    ) == {"transform_weights", "metadata", "self_weights"}
    # return_alias=False will return original names for "self"
    assert router._get_param_names(
        method="fit", return_alias=False, ignore_self_request=False
    ) == {"sample_weight", "metadata", "transform_weights"}
    # ignoring self would remove "sample_weight"
    assert router._get_param_names(
        method="fit", return_alias=False, ignore_self_request=True
    ) == {"metadata", "transform_weights"}
    # return_alias is ignored when ignore_self_request=True
    assert router._get_param_names(
        method="fit", return_alias=True, ignore_self_request=True
    ) == router._get_param_names(
        method="fit", return_alias=False, ignore_self_request=True
    )


@config_context(enable_metadata_routing=True)
def test_method_generation():
    # [FUNCTION DOC]
    # Verifies set_{method}_request generation: absent when no method accepts metadata;
    # for an estimator whose methods all accept sample_weight, present for every simple
    # method and absent for composite methods.
    # Test if all required request methods are generated.

    # TODO: these test classes can be moved to sklearn.utils._testing once we
    # have a better idea of what the commonly used classes are.
    class SimpleEstimator(BaseEstimator):
        # [CLASS DOC]
        # HELPER: Estimator with all methods but no metadata parameters.
        # This class should have no set_{method}_request
        def fit(self, X, y):
            # [FUNCTION DOC]
            # HELPER: Stub method; does nothing, returns None.
            pass  # pragma: no cover

        def fit_transform(self, X, y):
            # [FUNCTION DOC]
            # HELPER: Stub method; does nothing, returns None.
            pass  # pragma: no cover

        def fit_predict(self, X, y):
            # [FUNCTION DOC]
            # HELPER: Stub method; does nothing, returns None.
            pass  # pragma: no cover

        def partial_fit(self, X, y):
            # [FUNCTION DOC]
            # HELPER: Stub method; does nothing, returns None.
            pass  # pragma: no cover

        def predict(self, X):
            # [FUNCTION DOC]
            # HELPER: Stub method; does nothing, returns None.
            pass  # pragma: no cover

        def predict_proba(self, X):
            # [FUNCTION DOC]
            # HELPER: Stub method; does nothing, returns None.
            pass  # pragma: no cover

        def predict_log_proba(self, X):
            # [FUNCTION DOC]
            # HELPER: Stub method; does nothing, returns None.
            pass  # pragma: no cover

        def decision_function(self, X):
            # [FUNCTION DOC]
            # HELPER: Stub method; does nothing, returns None.
            pass  # pragma: no cover

        def score(self, X, y):
            # [FUNCTION DOC]
            # HELPER: Stub method; does nothing, returns None.
            pass  # pragma: no cover

        def split(self, X, y=None):
            # [FUNCTION DOC]
            # HELPER: Stub method; does nothing, returns None.
            pass  # pragma: no cover

        def transform(self, X):
            # [FUNCTION DOC]
            # HELPER: Stub method; does nothing, returns None.
            pass  # pragma: no cover

        def inverse_transform(self, X):
            # [FUNCTION DOC]
            # HELPER: Stub method; does nothing, returns None.
            pass  # pragma: no cover

    for method in METHODS:
        assert not hasattr(SimpleEstimator(), f"set_{method}_request")

    class SimpleEstimator(BaseEstimator):
        # [CLASS DOC]
        # HELPER: Estimator whose every method accepts sample_weight (shadows the previous
        # local class of the same name).
        # This class should have every set_{method}_request
        def fit(self, X, y, sample_weight=None):
            # [FUNCTION DOC]
            # HELPER: Stub method; does nothing, returns None.
            pass  # pragma: no cover

        def fit_transform(self, X, y, sample_weight=None):
            # [FUNCTION DOC]
            # HELPER: Stub method; does nothing, returns None.
            pass  # pragma: no cover

        def fit_predict(self, X, y, sample_weight=None):
            # [FUNCTION DOC]
            # HELPER: Stub method; does nothing, returns None.
            pass  # pragma: no cover

        def partial_fit(self, X, y, sample_weight=None):
            # [FUNCTION DOC]
            # HELPER: Stub method; does nothing, returns None.
            pass  # pragma: no cover

        def predict(self, X, sample_weight=None):
            # [FUNCTION DOC]
            # HELPER: Stub method; does nothing, returns None.
            pass  # pragma: no cover

        def predict_proba(self, X, sample_weight=None):
            # [FUNCTION DOC]
            # HELPER: Stub method; does nothing, returns None.
            pass  # pragma: no cover

        def predict_log_proba(self, X, sample_weight=None):
            # [FUNCTION DOC]
            # HELPER: Stub method; does nothing, returns None.
            pass  # pragma: no cover

        def decision_function(self, X, sample_weight=None):
            # [FUNCTION DOC]
            # HELPER: Stub method; does nothing, returns None.
            pass  # pragma: no cover

        def score(self, X, y, sample_weight=None):
            # [FUNCTION DOC]
            # HELPER: Stub method; does nothing, returns None.
            pass  # pragma: no cover

        def split(self, X, y=None, sample_weight=None):
            # [FUNCTION DOC]
            # HELPER: Stub method; does nothing, returns None.
            pass  # pragma: no cover

        def transform(self, X, sample_weight=None):
            # [FUNCTION DOC]
            # HELPER: Stub method; does nothing, returns None.
            pass  # pragma: no cover

        def inverse_transform(self, X, sample_weight=None):
            # [FUNCTION DOC]
            # HELPER: Stub method; does nothing, returns None.
            pass  # pragma: no cover

    # composite methods shouldn't have a corresponding set method.
    for method in COMPOSITE_METHODS:
        assert not hasattr(SimpleEstimator(), f"set_{method}_request")

    # simple methods should have a corresponding set method.
    for method in SIMPLE_METHODS:
        assert hasattr(SimpleEstimator(), f"set_{method}_request")


@config_context(enable_metadata_routing=True)
def test_composite_methods():
    # [FUNCTION DOC]
    # Verifies composite method requests (fit_transform, fit_predict) are unions of the
    # simple ones, raise ValueError on conflicting requests between fit and predict, and
    # merge non-overlapping params.
    # Test the behavior and the values of methods (composite methods) whose
    # request values are a union of requests by other methods (simple methods).
    # fit_transform and fit_predict are the only composite methods we have in
    # scikit-learn.
    class SimpleEstimator(BaseEstimator):
        # [CLASS DOC]
        # HELPER: Estimator with fit/predict accepting foo and bar, transform accepting
        # other_param.
        # This class should have every set_{method}_request
        def fit(self, X, y, foo=None, bar=None):
            # [FUNCTION DOC]
            # HELPER: Stub; does nothing.
            pass  # pragma: no cover

        def predict(self, X, foo=None, bar=None):
            # [FUNCTION DOC]
            # HELPER: Stub; does nothing.
            pass  # pragma: no cover

        def transform(self, X, other_param=None):
            # [FUNCTION DOC]
            # HELPER: Stub; does nothing.
            pass  # pragma: no cover

    est = SimpleEstimator()
    # Since no request is set for fit or predict or transform, the request for
    # fit_transform and fit_predict should also be empty.
    assert est.get_metadata_routing().fit_transform.requests == {
        "bar": None,
        "foo": None,
        "other_param": None,
    }
    assert est.get_metadata_routing().fit_predict.requests == {"bar": None, "foo": None}

    # setting the request on only one of them should raise an error
    est.set_fit_request(foo=True, bar="test")
    with pytest.raises(ValueError, match="Conflicting metadata requests for"):
        est.get_metadata_routing().fit_predict

    # setting the request on the other one should fail if not the same as the
    # first method
    est.set_predict_request(bar=True)
    with pytest.raises(ValueError, match="Conflicting metadata requests for"):
        est.get_metadata_routing().fit_predict

    # now the requests are consistent and getting the requests for fit_predict
    # shouldn't raise.
    est.set_predict_request(foo=True, bar="test")
    est.get_metadata_routing().fit_predict

    # setting the request for a none-overlapping parameter would merge them
    # together.
    est.set_transform_request(other_param=True)
    assert est.get_metadata_routing().fit_transform.requests == {
        "bar": "test",
        "foo": True,
        "other_param": True,
    }


@config_context(enable_metadata_routing=True)
def test_no_feature_flag_raises_error():
    # [FUNCTION DOC]
    # Verifies set_fit_request raises RuntimeError when routing is disabled.
    """Test that when feature flag disabled, set_{method}_requests raises."""
    with config_context(enable_metadata_routing=False):
        with pytest.raises(RuntimeError, match="This method is only available"):
            ConsumingClassifier().set_fit_request(sample_weight=True)


@config_context(enable_metadata_routing=True)
def test_none_metadata_passed():
    # [FUNCTION DOC]
    # Verifies passing sample_weight=None to a non-requesting child does not raise.
    """Test that passing None as metadata when not requested doesn't raise"""
    MetaRegressor(estimator=ConsumingRegressor()).fit(X, y, sample_weight=None)


@config_context(enable_metadata_routing=True)
def test_no_metadata_always_works():
    # [FUNCTION DOC]
    # Non-regression (issue 28246): a child using _RoutingNotSupportedMixin works when no
    # metadata is passed and raises NotImplementedError when metadata is passed.
    """Test that when no metadata is passed, having a meta-estimator which does
    not yet support metadata routing works.

    Non-regression test for https://github.com/scikit-learn/scikit-learn/issues/28246
    """

    class Estimator(_RoutingNotSupportedMixin, BaseEstimator):
        # [CLASS DOC]
        # HELPER: Estimator that declares routing as not supported.
        def fit(self, X, y, metadata=None):
            # [FUNCTION DOC]
            # HELPER: Stub fit returning self.
            return self

    # This passes since no metadata is passed.
    MetaRegressor(estimator=Estimator()).fit(X, y)
    # This fails since metadata is passed but Estimator() does not support it.
    with pytest.raises(
        NotImplementedError, match="Estimator has not implemented metadata routing yet."
    ):
        MetaRegressor(estimator=Estimator()).fit(X, y, metadata=my_groups)


@config_context(enable_metadata_routing=True)
def test_unsetmetadatapassederror_correct():
    # [FUNCTION DOC]
    # Verifies the UnsetMetadataPassedError message names the nested consumer, its parent
    # method and the set_fit_request hint.
    """Test that UnsetMetadataPassedError raises the correct error message when
    set_{method}_request is not set in nested cases."""
    weighted_meta = WeightedMetaClassifier(estimator=ConsumingClassifier())
    pipe = SimplePipeline([weighted_meta])
    msg = re.escape(
        "[metadata] are passed but are not explicitly set as requested or not requested"
        " for ConsumingClassifier.fit, which is used within WeightedMetaClassifier.fit."
        " Call `ConsumingClassifier.set_fit_request({metadata}=True/False)` for each"
        " metadata you want to request/ignore."
    )

    with pytest.raises(UnsetMetadataPassedError, match=msg):
        pipe.fit(X, y, metadata="blah")


@config_context(enable_metadata_routing=True)
def test_unsetmetadatapassederror_correct_for_composite_methods():
    # [FUNCTION DOC]
    # Verifies the UnsetMetadataPassedError message for a composite method (fit_transform)
    # in Pipeline suggests both set_fit_request and set_transform_request.
    """Test that UnsetMetadataPassedError raises the correct error message when
    composite metadata request methods are not set in nested cases."""
    consuming_transformer = ConsumingTransformer()
    pipe = Pipeline([("consuming_transformer", consuming_transformer)])

    msg = re.escape(
        "[metadata] are passed but are not explicitly set as requested or not requested"
        " for ConsumingTransformer.fit_transform, which is used within"
        " Pipeline.fit_transform. Call"
        " `ConsumingTransformer.set_fit_request({metadata}=True/False)"
        ".set_transform_request({metadata}=True/False)`"
        " for each metadata you want to request/ignore."
    )
    with pytest.raises(UnsetMetadataPassedError, match=msg):
        pipe.fit_transform(X, y, metadata="blah")


@config_context(enable_metadata_routing=True)
def test_unbound_set_methods_work():
    # [FUNCTION DOC]
    # Non-regression (issue 28632): set_fit_request keeps working after reassigning the
    # descriptor on the class, and positional arguments raise a TypeError before and after.
    # Side effects: mutates the local class A only.
    """Tests that if the set_{method}_request is unbound, it still works.

    Also test that passing positional arguments to the set_{method}_request fails
    with the right TypeError message.

    Non-regression test for https://github.com/scikit-learn/scikit-learn/issues/28632
    """

    class A(BaseEstimator):
        # [CLASS DOC]
        # HELPER: Estimator whose fit accepts sample_weight.
        def fit(self, X, y, sample_weight=None):
            # [FUNCTION DOC]
            # HELPER: Stub fit returning self.
            return self

    error_message = re.escape(
        "set_fit_request() takes 0 positional argument but 1 were given"
    )

    # Test positional arguments error before making the descriptor method unbound.
    with pytest.raises(TypeError, match=error_message):
        A().set_fit_request(True)

    # This somehow makes the descriptor method unbound, which results in the `instance`
    # argument being None, and instead `self` being passed as a positional argument
    # to the descriptor method.
    A.set_fit_request = A.set_fit_request

    # This should pass as usual
    A().set_fit_request(sample_weight=True)

    # Test positional arguments error after making the descriptor method unbound.
    with pytest.raises(TypeError, match=error_message):
        A().set_fit_request(True)


class _UncopyableOwner:
    # [CLASS DOC]
    # HELPER: Owner stand-in that fails if deep-copied, used to check cloning of routing
    # objects keeps the owner by reference.
    """An owner-like object that fails on deepcopy.

    Used to verify that cloning routing objects does not walk into the
    estimator state. This mirrors the real-world skorch case where the
    estimator holds attributes (e.g. locally-defined torch modules) that are
    not picklable / deep-copyable.
    """

    def __deepcopy__(self, memo):
        # [FUNCTION DOC]
        # HELPER: Always raises AssertionError.
        raise AssertionError("owner must not be deep-copied")  # pragma: no cover


def test_method_metadata_request_clone_does_not_copy_owner():
    # [FUNCTION DOC]
    # Verifies cloning a MethodMetadataRequest shares the owner but deep-copies the
    # requests, so mutating the clone does not affect the original.
    owner = _UncopyableOwner()
    req = MethodMetadataRequest(owner=owner, method="fit")
    req.add_request(param="sample_weight", alias=True)

    new = clone(req)

    # owner is shared by reference, not copied
    assert new.owner is owner
    # routing state is deep-copied (independent dict)
    assert new.requests == {"sample_weight": True}
    assert new._requests is not req._requests
    # mutating the copy doesn't affect the original
    new.add_request(param="groups", alias=True)
    assert "groups" not in req.requests


def test_metadata_request_clone_does_not_copy_owner():
    # [FUNCTION DOC]
    # Verifies cloning a MetadataRequest shares the owner (also on every per-method request)
    # and copies the per-method requests independently.
    owner = _UncopyableOwner()
    req = MetadataRequest(owner=owner)
    req.fit.add_request(param="sample_weight", alias=True)

    new = clone(req)

    assert new.owner is owner
    for method in SIMPLE_METHODS:
        assert getattr(new, method).owner is owner
    assert new.fit.requests == {"sample_weight": True}
    assert new.fit is not req.fit
    new.fit.add_request(param="groups", alias=True)
    assert "groups" not in req.fit.requests


def test_metadata_router_clone_does_not_copy_owner():
    # [FUNCTION DOC]
    # Verifies cloning a MetadataRouter shares owners (also of nested requests) while the
    # route mappings are independent copies.
    owner = _UncopyableOwner()
    sub_owner = _UncopyableOwner()
    sub_req = MetadataRequest(owner=sub_owner)
    sub_req.fit.add_request(param="sample_weight", alias=True)

    router = MetadataRouter(owner=owner).add(
        est=sub_req,
        method_mapping=MethodMapping().add(caller="fit", callee="fit"),
    )

    new = clone(router)

    assert new.owner is owner
    assert new._route_mappings["est"].router.owner is sub_owner
    # routing state is independent from the original
    assert new._route_mappings is not router._route_mappings
    assert new._route_mappings["est"].router.fit.requests == {"sample_weight": True}


@config_context(enable_metadata_routing=True)
def test_get_routing_for_object_does_not_deepcopy_estimator():
    # [FUNCTION DOC]
    # Regression (skorch): get_routing_for_object on an estimator must not deep-copy it;
    # the returned routing's owner is the estimator itself.
    # Regression test for the skorch deepcopy issue: asking for routing info
    # of an estimator should not deep-copy the estimator itself.
    class Est(BaseEstimator):
        # [CLASS DOC]
        # HELPER: Estimator that fails if deep-copied.
        def fit(self, X, y, sample_weight=None):
            # [FUNCTION DOC]
            # HELPER: Stub fit returning self.
            return self  # pragma: no cover

        def __deepcopy__(self, memo):
            # [FUNCTION DOC]
            # HELPER: Always raises AssertionError.
            raise AssertionError(
                "estimator must not be deep-copied"
            )  # pragma: no cover

    est = Est().set_fit_request(sample_weight=True)
    routing = get_routing_for_object(est)
    assert routing.owner is est


@config_context(enable_metadata_routing=True)
def test_add_self_request_does_not_deepcopy_estimator():
    # [FUNCTION DOC]
    # Verifies add_self_request with an estimator does not deep-copy it; the self request
    # owner is the estimator.
    class Est(BaseEstimator):
        # [CLASS DOC]
        # HELPER: Estimator that fails if deep-copied.
        def fit(self, X, y, sample_weight=None):
            # [FUNCTION DOC]
            # HELPER: Stub fit returning self.
            return self  # pragma: no cover

        def __deepcopy__(self, memo):
            # [FUNCTION DOC]
            # HELPER: Always raises AssertionError.
            raise AssertionError(
                "estimator must not be deep-copied"
            )  # pragma: no cover

    est = Est().set_fit_request(sample_weight=True)
    # add_self_request clones the request internally; it must not reach into
    # the estimator.
    router = MetadataRouter(owner=est).add_self_request(est)
    assert router._self_request.owner is est


@config_context(enable_metadata_routing=True)
def test_removing_metadata_in_subclass_correctly_works():
    # [FUNCTION DOC]
    # Verifies that a subclass marking all score metadata UNUSED has a set_score_request
    # that rejects sample_weight with TypeError, while the parent still accepts it.
    """Test that removing a metadata with UNUSED marker affects child's method."""

    class A(ConsumingClassifier):
        # [CLASS DOC]
        # HELPER: ConsumingClassifier with all score metadata removed.
        __metadata_request__score = {
            "sample_weight": metadata_routing.UNUSED,
            "metadata": metadata_routing.UNUSED,
        }

    # Here we make sure that the parent class has the method as usual
    assert hasattr(ConsumingClassifier(), "set_score_request")
    # And that the child class doesn't have it since all metadata for the score method
    # are removed.
    with pytest.raises(
        TypeError,
        match=re.escape(
            "Unexpected args: {'sample_weight'} in score. Accepted arguments are: set()"
        ),
    ):
        A().set_score_request(sample_weight=True)


@config_context(enable_metadata_routing=True)
def test_explicitly_defined_set_method_request_is_not_overriden():
    # [FUNCTION DOC]
    # Verifies a user-defined set_score_request on a parent is not replaced by the
    # auto-generated one in a subclass.
    """Test that explicitly defined set_{method}_request is not overridden."""

    class A(BaseEstimator):
        # [CLASS DOC]
        # HELPER: Estimator defining its own set_score_request.
        def set_score_request(self, sample_weight=None, metadata=None):
            # [FUNCTION DOC]
            # HELPER: Custom setter returning self without recording anything.
            return self  # pragma: no cover

    class B(A):
        # [CLASS DOC]
        # HELPER: Subclass adding score, which would normally trigger generation.
        def score(self, X, y=None):
            # [FUNCTION DOC]
            # HELPER: Stub score; does nothing.
            pass  # pragma: no cover

    # This should work as usual since the method is explicitly defined.
    B().set_score_request(sample_weight=True)
