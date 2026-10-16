# [MODULE DOC]
# General API-conformance tests run over all estimators: the full common check
# suite (parametrize_with_checks), import/__all__ consistency, presence of test
# subpackages, feature-name, set_output, parameter validation, in-place writeable
# input and zero-sample-weight checks. At import time it instantiates estimator
# lists (column_name_estimators, GET_FEATURES_OUT_ESTIMATORS,
# ESTIMATORS_WITH_GET_FEATURE_NAMES_OUT, SET_OUTPUT_ESTIMATORS,
# ESTIMATORS_ACCEPTING_SAMPLE_WEIGHTS) and enables experimental estimators.

"""
General tests for all estimators in sklearn.
"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import os
import pkgutil
import re
import warnings
from functools import partial
from itertools import chain

import pytest
from scipy.linalg import LinAlgWarning

import sklearn
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning

# make it possible to discover experimental estimators when calling `all_estimators`
from sklearn.experimental import (
    enable_halving_search_cv,  # noqa: F401
    enable_iterative_imputer,  # noqa: F401
)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, make_pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    MinMaxScaler,
    OneHotEncoder,
    StandardScaler,
)
from sklearn.utils import all_estimators
from sklearn.utils._test_common.instance_generator import (
    _get_check_estimator_ids,
    _get_expected_failed_checks,
    _tested_estimators,
    _yield_instances_for_check,
)
from sklearn.utils._testing import (
    SkipTest,
    ignore_warnings,
)
from sklearn.utils.estimator_checks import (
    check_all_zero_sample_weights_error,
    check_dataframe_column_names_consistency,
    check_estimator,
    check_get_feature_names_out_error,
    check_global_output_transform_pandas,
    check_global_set_output_transform_polars,
    check_inplace_ensure_writeable,
    check_param_validation,
    check_set_output_transform,
    check_set_output_transform_pandas,
    check_set_output_transform_polars,
    check_transformer_get_feature_names_out,
    check_transformer_get_feature_names_out_pandas,
    parametrize_with_checks,
)
from sklearn.utils.validation import has_fit_parameter


@pytest.mark.thread_unsafe  # import side-effects
def test_all_estimator_no_base_class():
    # [FUNCTION DOC]
    # Verifies no name returned by all_estimators starts with "base" (case-insensitive),
    # i.e. abstract base classes are not discovered.
    # test that all_estimators doesn't find abstract classes.
    for name, Estimator in all_estimators():
        msg = (
            "Base estimators such as {0} should not be included in all_estimators"
        ).format(name)
        assert not name.lower().startswith("base"), msg


def _sample_func(x, y=1):
    # [FUNCTION DOC]
    # HELPER: No-op function used as a parameter value when testing estimator id
    # generation for plain functions and functools.partial objects.
    pass


class CallableEstimator(BaseEstimator):
    # [CLASS DOC]
    # HELPER: Minimal BaseEstimator subclass that is also callable; used to make sure
    # id generation treats it as an estimator (repr) rather than as a function.
    """Dummy development stub for an estimator.

    This is to make sure a callable estimator passes common tests.
    """

    def __call__(self):
        # [FUNCTION DOC]
        # HELPER: No-op call; never executed in tests.
        pass  # pragma: nocover


@pytest.mark.parametrize(
    "val, expected",
    [
        (partial(_sample_func, y=1), "_sample_func(y=1)"),
        (_sample_func, "_sample_func"),
        (partial(_sample_func, "world"), "_sample_func"),
        (LogisticRegression(C=2.0), "LogisticRegression(C=2.0)"),
        (
            LogisticRegression(
                random_state=1,
                solver="newton-cg",
                class_weight="balanced",
                warm_start=True,
            ),
            (
                "LogisticRegression(class_weight='balanced',random_state=1,"
                "solver='newton-cg',warm_start=True)"
            ),
        ),
        (CallableEstimator(), "CallableEstimator()"),
    ],
)
def test_get_check_estimator_ids(val, expected):
    # [FUNCTION DOC]
    # Verifies _get_check_estimator_ids output for partials (name plus keyword args
    # only), plain functions (name), estimators (compact repr without spaces/newlines)
    # and callable estimators.
    assert _get_check_estimator_ids(val) == expected


@pytest.mark.no_check_spmatrix  # pickle breaks check_spmatrix
@parametrize_with_checks(
    list(_tested_estimators()), expected_failed_checks=_get_expected_failed_checks
)
def test_estimators(estimator, check, request):
    # [FUNCTION DOC]
    # Runs every common estimator check on every tested estimator instance, generated
    # by parametrize_with_checks with the known expected failures, while silencing
    # FutureWarning, ConvergenceWarning, UserWarning and LinAlgWarning.
    # Common tests for estimator instances
    with ignore_warnings(
        category=(FutureWarning, ConvergenceWarning, UserWarning, LinAlgWarning)
    ):
        check(estimator)


@pytest.mark.filterwarnings(
    "ignore:Since version 1.0, it is not needed to import "
    "enable_hist_gradient_boosting anymore"
)
@pytest.mark.thread_unsafe  # import side-effects
def test_import_all_consistency():
    # [FUNCTION DOC]
    # Walks all sklearn modules (excluding tests, externals and _build_utils), imports
    # each, and asserts every name listed in its __all__ exists as an attribute.
    # Side effects: imports every sklearn submodule.
    sklearn_path = [os.path.dirname(sklearn.__file__)]
    # Smoke test to check that any name in a __all__ list is actually defined
    # in the namespace of the module or package.
    pkgs = pkgutil.walk_packages(
        path=sklearn_path, prefix="sklearn.", onerror=lambda _: None
    )
    submods = [modname for _, modname, _ in pkgs]
    for modname in submods + ["sklearn"]:
        if ".tests." in modname or "sklearn.externals" in modname:
            continue
        # Avoid test suite depending on build dependencies, for example Cython
        if "sklearn._build_utils" in modname:
            continue
        package = __import__(modname, fromlist="dummy")
        for name in getattr(package, "__all__", ()):
            assert hasattr(package, name), "Module '{0}' has no attribute '{1}'".format(
                modname, name
            )


def test_root_import_all_completeness():
    # [FUNCTION DOC]
    # Asserts each public top-level sklearn subpackage/module (not private and not in
    # utils, tests, base, conftest) is listed in sklearn.__all__.
    sklearn_path = [os.path.dirname(sklearn.__file__)]
    EXCEPTIONS = ("utils", "tests", "base", "conftest")
    for _, modname, _ in pkgutil.walk_packages(
        path=sklearn_path, onerror=lambda _: None
    ):
        if "." in modname or modname.startswith("_") or modname in EXCEPTIONS:
            continue
        assert modname in sklearn.__all__


@pytest.mark.thread_unsafe  # import side-effects
def test_all_tests_are_importable():
    # [FUNCTION DOC]
    # Asserts every sklearn subpackage, except externals, tests, private ones and data
    # resource packages, has a "tests" subpackage discoverable by walk_packages.
    # Ensure that for each contentful subpackage, there is a test directory
    # within it that is also a subpackage (i.e. a directory with __init__.py)

    HAS_TESTS_EXCEPTIONS = re.compile(
        r"""(?x)
        \.externals(\.|$)|
        \.tests(\.|$)|
        \._
        """
    )
    resource_modules = {
        "sklearn.datasets.data",
        "sklearn.datasets.descr",
        "sklearn.datasets.images",
    }
    sklearn_path = [os.path.dirname(sklearn.__file__)]
    lookup = {
        name: ispkg
        for _, name, ispkg in pkgutil.walk_packages(sklearn_path, prefix="sklearn.")
    }
    missing_tests = [
        name
        for name, ispkg in lookup.items()
        if ispkg
        and name not in resource_modules
        and not HAS_TESTS_EXCEPTIONS.search(name)
        and name + ".tests" not in lookup
    ]
    assert missing_tests == [], (
        "{0} do not have `tests` subpackages. "
        "Perhaps they require "
        "__init__.py or a meson.build "
        "in the parent "
        "directory".format(missing_tests)
    )


def test_class_support_removed():
    # [FUNCTION DOC]
    # Verifies passing an estimator class (not instance) to check_estimator or
    # parametrize_with_checks raises TypeError with the removal message.
    # Make sure passing classes to check_estimator or parametrize_with_checks
    # raises an error

    msg = "Passing a class was deprecated.* isn't supported anymore"
    with pytest.raises(TypeError, match=msg):
        check_estimator(LogisticRegression)

    with pytest.raises(TypeError, match=msg):
        parametrize_with_checks([LogisticRegression])


def _estimators_that_predict_in_fit():
    # [FUNCTION DOC]
    # HELPER: Generator over tested estimators reconfigured so that fit internally
    # predicts on held-out data: enables oob_score with bootstrap, or early_stopping
    # with n_iter_no_change 1 (MLP variants wrapped in an xfail pytest.param), or just
    # n_iter_no_change 1. Estimators with none of these parameters are skipped.
    # KNOWN ISSUE: pytest.mark.xfail is given a "msg" keyword instead of "reason".
    for estimator in _tested_estimators():
        est_params = set(estimator.get_params())
        if "oob_score" in est_params:
            yield estimator.set_params(oob_score=True, bootstrap=True)
        elif "early_stopping" in est_params:
            est = estimator.set_params(early_stopping=True, n_iter_no_change=1)
            if est.__class__.__name__ in {"MLPClassifier", "MLPRegressor"}:
                # TODO: FIX MLP to not check validation set during MLP
                yield pytest.param(
                    est, marks=pytest.mark.xfail(msg="MLP still validates in fit")
                )
            else:
                yield est
        elif "n_iter_no_change" in est_params:
            yield estimator.set_params(n_iter_no_change=1)


# NOTE: When running `check_dataframe_column_names_consistency` on a meta-estimator that
# delegates validation to a base estimator, the check is testing that the base estimator
# is checking for column name consistency.
column_name_estimators = list(
    chain(
        _tested_estimators(),
        [make_pipeline(LogisticRegression(C=1))],
        _estimators_that_predict_in_fit(),
    )
)


@pytest.mark.parametrize(
    "estimator_orig", column_name_estimators, ids=_get_check_estimator_ids
)
def test_pandas_column_name_consistency(estimator_orig):
    # [FUNCTION DOC]
    # For each estimator (skipping ColumnTransformer and those expected to fail the
    # check), runs check_dataframe_column_names_consistency on each check instance and
    # asserts no warning says the estimator "was fitted without feature names".
    if isinstance(estimator_orig, ColumnTransformer):
        pytest.skip("ColumnTransformer is not tested here")
    if "check_dataframe_column_names_consistency" in _get_expected_failed_checks(
        estimator_orig
    ):
        pytest.skip(
            "Estimator does not support check_dataframe_column_names_consistency"
        )
    for estimator in _yield_instances_for_check(
        check_dataframe_column_names_consistency, estimator_orig
    ):
        with ignore_warnings(category=(FutureWarning)):
            with warnings.catch_warnings(record=True) as record:
                check_dataframe_column_names_consistency(
                    estimator.__class__.__name__, estimator
                )
            for warning in record:
                assert "was fitted without feature names" not in str(warning.message)


# TODO: As more modules support get_feature_names_out they should be removed
# from this list to be tested
GET_FEATURES_OUT_MODULES_TO_IGNORE = [
    "ensemble",
    "kernel_approximation",
]


def _include_in_get_feature_names_out_check(transformer):
    # [FUNCTION DOC]
    # HELPER: Returns True if the transformer has get_feature_names_out, otherwise True
    # only when its top-level sklearn submodule is not in
    # GET_FEATURES_OUT_MODULES_TO_IGNORE.
    if hasattr(transformer, "get_feature_names_out"):
        return True
    module = transformer.__module__.split(".")[1]
    return module not in GET_FEATURES_OUT_MODULES_TO_IGNORE


GET_FEATURES_OUT_ESTIMATORS = [
    est
    for est in _tested_estimators("transformer")
    if _include_in_get_feature_names_out_check(est)
]


@pytest.mark.parametrize(
    "transformer", GET_FEATURES_OUT_ESTIMATORS, ids=_get_check_estimator_ids
)
def test_transformers_get_feature_names_out(transformer):
    # [FUNCTION DOC]
    # Runs check_transformer_get_feature_names_out and its pandas variant on each
    # selected transformer, ignoring FutureWarning.
    with ignore_warnings(category=(FutureWarning)):
        check_transformer_get_feature_names_out(
            transformer.__class__.__name__, transformer
        )
        check_transformer_get_feature_names_out_pandas(
            transformer.__class__.__name__, transformer
        )


ESTIMATORS_WITH_GET_FEATURE_NAMES_OUT = [
    est for est in _tested_estimators() if hasattr(est, "get_feature_names_out")
]


@pytest.mark.parametrize(
    "estimator", ESTIMATORS_WITH_GET_FEATURE_NAMES_OUT, ids=_get_check_estimator_ids
)
def test_estimators_get_feature_names_out_error(estimator):
    # [FUNCTION DOC]
    # Runs check_get_feature_names_out_error (calling it before fit must raise
    # NotFittedError) on estimators exposing get_feature_names_out.
    estimator_name = estimator.__class__.__name__
    check_get_feature_names_out_error(estimator_name, estimator)


@pytest.mark.parametrize(
    "estimator", list(_tested_estimators()), ids=_get_check_estimator_ids
)
def test_check_param_validation(estimator):
    # [FUNCTION DOC]
    # Runs check_param_validation on every tested estimator except FeatureUnion.
    if isinstance(estimator, FeatureUnion):
        pytest.skip("FeatureUnion is not tested here")
    name = estimator.__class__.__name__
    check_param_validation(name, estimator)


SET_OUTPUT_ESTIMATORS = list(
    chain(
        _tested_estimators("transformer"),
        [
            make_pipeline(StandardScaler(), MinMaxScaler()),
            OneHotEncoder(sparse_output=False),
            FunctionTransformer(feature_names_out="one-to-one"),
        ],
    )
)


@pytest.mark.parametrize(
    "estimator_orig", SET_OUTPUT_ESTIMATORS, ids=_get_check_estimator_ids
)
def test_set_output_transform(estimator_orig):
    # [FUNCTION DOC]
    # For transformers supporting set_output, runs check_set_output_transform on each
    # check instance; skips those without set_output.
    name = estimator_orig.__class__.__name__
    if not hasattr(estimator_orig, "set_output"):
        pytest.skip(
            f"Skipping check_set_output_transform for {name}: Does not support"
            " set_output API"
        )
    for estimator in _yield_instances_for_check(
        check_set_output_transform, estimator_orig
    ):
        with ignore_warnings(category=(FutureWarning)):
            check_set_output_transform(estimator.__class__.__name__, estimator)


@pytest.mark.parametrize(
    "estimator_orig", SET_OUTPUT_ESTIMATORS, ids=_get_check_estimator_ids
)
@pytest.mark.parametrize(
    "check_func",
    [
        check_set_output_transform_pandas,
        check_global_output_transform_pandas,
        check_set_output_transform_polars,
        check_global_set_output_transform_polars,
    ],
)
def test_set_output_transform_configured(estimator_orig, check_func):
    # [FUNCTION DOC]
    # Same as test_set_output_transform but parametrized over the pandas/polars local
    # (set_output) and global (config) output checks.
    name = estimator_orig.__class__.__name__
    if not hasattr(estimator_orig, "set_output"):
        pytest.skip(
            f"Skipping {check_func.__name__} for {name}: Does not support"
            " set_output API yet"
        )
    for estimator in _yield_instances_for_check(check_func, estimator_orig):
        with ignore_warnings(category=(FutureWarning)):
            check_func(estimator.__class__.__name__, estimator)


@pytest.mark.parametrize(
    "estimator", _tested_estimators(), ids=_get_check_estimator_ids
)
def test_check_inplace_ensure_writeable(estimator):
    # [FUNCTION DOC]
    # Sets copy=False or copy_X=False (skipping estimators with neither), applies
    # settings needed for in-place work (HDBSCAN precomputed brute, PCA full solver,
    # KernelPCA precomputed kernel), then runs check_inplace_ensure_writeable.
    # Side effects: mutates the shared parametrized estimator instance.
    name = estimator.__class__.__name__

    if hasattr(estimator, "copy"):
        estimator.set_params(copy=False)
    elif hasattr(estimator, "copy_X"):
        estimator.set_params(copy_X=False)
    else:
        raise SkipTest(f"{name} doesn't require writeable input.")

    # The following estimators can work inplace only with certain settings
    if name == "HDBSCAN":
        estimator.set_params(metric="precomputed", algorithm="brute")

    if name == "PCA":
        estimator.set_params(svd_solver="full")

    if name == "KernelPCA":
        estimator.set_params(kernel="precomputed")

    check_inplace_ensure_writeable(name, estimator)


ESTIMATORS_ACCEPTING_SAMPLE_WEIGHTS = [
    est for est in _tested_estimators() if has_fit_parameter(est, "sample_weight")
]


@pytest.mark.parametrize(
    "estimator", ESTIMATORS_ACCEPTING_SAMPLE_WEIGHTS, ids=_get_check_estimator_ids
)
def test_check_all_zero_sample_weights_error(estimator):
    # [FUNCTION DOC]
    # Runs check_all_zero_sample_weights_error on estimators whose fit accepts
    # sample_weight.
    name = estimator.__class__.__name__

    check_all_zero_sample_weights_error(name, estimator)
