# [MODULE DOC]
# Tests for sklearn.utils._param_validation: constraint classes (Interval, StrOptions,
# Options, HasMethods, Hidden, MissingValues and private constraints), make_constraint,
# invalid/valid value generators, and the validate_params decorator on functions,
# methods and estimators, including skipping via config and nested validation. Defines
# helper targets _func, _Class and _Estimator at module level.
from numbers import Integral, Real

import numpy as np
import pytest
from scipy.sparse import csr_array, csr_matrix

from sklearn._config import config_context, get_config
from sklearn.base import BaseEstimator, _fit_context
from sklearn.model_selection import LeaveOneOut
from sklearn.utils import deprecated
from sklearn.utils._param_validation import (
    HasMethods,
    Hidden,
    Interval,
    InvalidParameterError,
    MissingValues,
    Options,
    RealNotInt,
    StrOptions,
    _ArrayLikes,
    _Booleans,
    _Callables,
    _CVObjects,
    _InstancesOf,
    _IterablesNotString,
    _NanConstraint,
    _NoneConstraint,
    _PandasNAConstraint,
    _RandomStates,
    _SparseMatrices,
    _VerboseHelper,
    generate_invalid_param_val,
    generate_valid_param,
    make_constraint,
    validate_params,
)
from sklearn.utils.fixes import CSR_CONTAINERS


# Some helpers for the tests
@validate_params(
    {"a": [Real], "b": [Real], "c": [Real], "d": [Real]},
    prefer_skip_nested_validation=True,
)
def _func(a, b=0, *args, c, d=0, **kwargs):
    # [FUNCTION DOC]
    # HELPER: Empty function decorated with validate_params requiring a, b, c, d to be
    # Real; has positional, var-positional, keyword-only and var-keyword parameters.
    # Outputs: None.
    # Exceptions: InvalidParameterError (from the decorator) for non-Real a, b, c or d.
    """A function to test the validation of functions."""


class _Class:
    # [CLASS DOC]
    # HELPER: Plain class used as an _InstancesOf target and to host validated methods.
    """A class to test the _InstancesOf constraint and the validation of methods."""

    @validate_params({"a": [Real]}, prefer_skip_nested_validation=True)
    def _method(self, a):
        # [FUNCTION DOC]
        # HELPER: Empty method validated to require a Real a. Outputs: None.
        """A validated method"""

    @deprecated()
    @validate_params({"a": [Real]}, prefer_skip_nested_validation=True)
    def _deprecated_method(self, a):
        # [FUNCTION DOC]
        # HELPER: Empty method validated to require a Real a and wrapped by deprecated,
        # so calls emit a FutureWarning before validation. Outputs: None.
        """A deprecated validated method"""


class _Estimator(BaseEstimator):
    # [CLASS DOC]
    # HELPER: Minimal BaseEstimator whose parameter a must be Real; validation happens
    # in fit via _fit_context, not in __init__.
    """An estimator to test the validation of estimator parameters."""

    _parameter_constraints: dict = {"a": [Real]}

    def __init__(self, a):
        # [FUNCTION DOC]
        # HELPER: Stores a without validation. Side effects: sets self.a.
        self.a = a

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X=None, y=None):
        # [FUNCTION DOC]
        # HELPER: No-op fit wrapped by _fit_context, which validates parameters first.
        # Inputs: X, y - ignored (default None). Outputs: None.
        # Exceptions: InvalidParameterError if self.a is not Real.
        pass


@pytest.mark.parametrize("interval_type", [Integral, Real])
def test_interval_range(interval_type):
    # [FUNCTION DOC]
    # For Integral and Real intervals on [-2, 2], checks membership of both bounds for
    # each closed mode (left, right, both, neither).
    """Check the range of values depending on closed."""
    interval = Interval(interval_type, -2, 2, closed="left")
    assert -2 in interval
    assert 2 not in interval

    interval = Interval(interval_type, -2, 2, closed="right")
    assert -2 not in interval
    assert 2 in interval

    interval = Interval(interval_type, -2, 2, closed="both")
    assert -2 in interval
    assert 2 in interval

    interval = Interval(interval_type, -2, 2, closed="neither")
    assert -2 not in interval
    assert 2 not in interval


@pytest.mark.parametrize("interval_type", [Integral, Real])
def test_interval_large_integers(interval_type):
    # [FUNCTION DOC]
    # Non-regression for 26648: checks very large ints and their float conversions are
    # correctly classified as outside or inside open intervals, including 2**1024 which
    # cannot be converted to a float.
    """Check that Interval constraint work with large integers.

    non-regression test for #26648.
    """
    interval = Interval(interval_type, 0, 2, closed="neither")
    assert 2**65 not in interval
    assert 2**128 not in interval
    assert float(2**65) not in interval
    assert float(2**128) not in interval

    interval = Interval(interval_type, 0, 2**128, closed="neither")
    assert 2**65 in interval
    assert 2**128 not in interval
    assert float(2**65) in interval
    assert float(2**128) not in interval

    assert 2**1024 not in interval


def test_interval_inf_in_bounds():
    # [FUNCTION DOC]
    # Checks infinities are included in real intervals with an unbounded (None) side
    # closed, and excluded when both unbounded sides are open.
    """Check that inf is included iff a bound is closed and set to None.

    Only valid for real intervals.
    """
    interval = Interval(Real, 0, None, closed="right")
    assert np.inf in interval

    interval = Interval(Real, None, 0, closed="left")
    assert -np.inf in interval

    interval = Interval(Real, None, None, closed="neither")
    assert np.inf not in interval
    assert -np.inf not in interval


@pytest.mark.parametrize(
    "interval",
    [Interval(Real, 0, 1, closed="left"), Interval(Real, None, None, closed="both")],
)
def test_nan_not_in_interval(interval):
    # [FUNCTION DOC]
    # Checks NaN is never contained in a bounded or fully unbounded real interval.
    """Check that np.nan is not in any interval."""
    assert np.nan not in interval


@pytest.mark.parametrize(
    "params, error, match",
    [
        (
            {"type": Integral, "left": 1.0, "right": 2, "closed": "both"},
            TypeError,
            r"Expecting left to be an int for an interval over the integers",
        ),
        (
            {"type": Integral, "left": 1, "right": 2.0, "closed": "neither"},
            TypeError,
            "Expecting right to be an int for an interval over the integers",
        ),
        (
            {"type": Integral, "left": None, "right": 0, "closed": "left"},
            ValueError,
            r"left can't be None when closed == left",
        ),
        (
            {"type": Integral, "left": 0, "right": None, "closed": "right"},
            ValueError,
            r"right can't be None when closed == right",
        ),
        (
            {"type": Integral, "left": 1, "right": -1, "closed": "both"},
            ValueError,
            r"right can't be less than left",
        ),
    ],
)
def test_interval_errors(params, error, match):
    # [FUNCTION DOC]
    # Checks Interval construction raises TypeError for non-int bounds on integer
    # intervals and ValueError for a None bound on a closed side or right less than
    # left.
    """Check that informative errors are raised for invalid combination of parameters"""
    with pytest.raises(error, match=match):
        Interval(**params)


def test_stroptions():
    # [FUNCTION DOC]
    # Checks StrOptions accepts listed (including deprecated) options, rejects others,
    # and marks deprecated options in its string representation.
    """Sanity check for the StrOptions constraint"""
    options = StrOptions({"a", "b", "c"}, deprecated={"c"})
    assert options.is_satisfied_by("a")
    assert options.is_satisfied_by("c")
    assert not options.is_satisfied_by("d")

    assert "'c' (deprecated)" in str(options)


def test_options():
    # [FUNCTION DOC]
    # Checks Options over Real accepts listed values, rejects others and marks
    # deprecated values in its string representation.
    """Sanity check for the Options constraint"""
    options = Options(Real, {-0.5, 0.5, np.inf}, deprecated={-0.5})
    assert options.is_satisfied_by(-0.5)
    assert options.is_satisfied_by(np.inf)
    assert not options.is_satisfied_by(1.23)

    assert "-0.5 (deprecated)" in str(options)


@pytest.mark.parametrize(
    "type, expected_type_name",
    [
        (int, "int"),
        (Integral, "int"),
        (Real, "float"),
        (np.ndarray, "numpy.ndarray"),
    ],
)
def test_instances_of_type_human_readable(type, expected_type_name):
    # [FUNCTION DOC]
    # Checks _InstancesOf string representation uses a human readable type name (int,
    # float, numpy.ndarray).
    """Check the string representation of the _InstancesOf constraint."""
    constraint = _InstancesOf(type)
    assert str(constraint) == f"an instance of '{expected_type_name}'"


def test_hasmethods():
    # [FUNCTION DOC]
    # Checks HasMethods(["a", "b"]) accepts an object with both methods, rejects one
    # with only a, and has the expected string representation.
    """Check the HasMethods constraint."""
    constraint = HasMethods(["a", "b"])

    class _Good:
        # [CLASS DOC]
        # HELPER: Local class implementing both methods a and b.
        def a(self):
            # [FUNCTION DOC]
            # HELPER: No-op method (never called).
            pass  # pragma: no cover

        def b(self):
            # [FUNCTION DOC]
            # HELPER: No-op method (never called).
            pass  # pragma: no cover

    class _Bad:
        # [CLASS DOC]
        # HELPER: Local class implementing only method a.
        def a(self):
            # [FUNCTION DOC]
            # HELPER: No-op method (never called).
            pass  # pragma: no cover

    assert constraint.is_satisfied_by(_Good())
    assert not constraint.is_satisfied_by(_Bad())
    assert str(constraint) == "an object implementing 'a' and 'b'"


@pytest.mark.parametrize(
    "constraint",
    [
        Interval(Real, None, 0, closed="left"),
        Interval(Real, 0, None, closed="left"),
        Interval(Real, None, None, closed="neither"),
        StrOptions({"a", "b", "c"}),
        MissingValues(),
        MissingValues(numeric_only=True),
        _VerboseHelper(),
        HasMethods("fit"),
        _IterablesNotString(),
        _CVObjects(),
    ],
)
def test_generate_invalid_param_val(constraint):
    # [FUNCTION DOC]
    # For various constraints, checks generate_invalid_param_val returns a value the
    # constraint rejects.
    """Check that the value generated does not satisfy the constraint"""
    bad_value = generate_invalid_param_val(constraint)
    assert not constraint.is_satisfied_by(bad_value)


@pytest.mark.parametrize(
    "integer_interval, real_interval",
    [
        (
            Interval(Integral, None, 3, closed="right"),
            Interval(RealNotInt, -5, 5, closed="both"),
        ),
        (
            Interval(Integral, None, 3, closed="right"),
            Interval(RealNotInt, -5, 5, closed="neither"),
        ),
        (
            Interval(Integral, None, 3, closed="right"),
            Interval(RealNotInt, 4, 5, closed="both"),
        ),
        (
            Interval(Integral, None, 3, closed="right"),
            Interval(RealNotInt, 5, None, closed="left"),
        ),
        (
            Interval(Integral, None, 3, closed="right"),
            Interval(RealNotInt, 4, None, closed="neither"),
        ),
        (
            Interval(Integral, 3, None, closed="left"),
            Interval(RealNotInt, -5, 5, closed="both"),
        ),
        (
            Interval(Integral, 3, None, closed="left"),
            Interval(RealNotInt, -5, 5, closed="neither"),
        ),
        (
            Interval(Integral, 3, None, closed="left"),
            Interval(RealNotInt, 1, 2, closed="both"),
        ),
        (
            Interval(Integral, 3, None, closed="left"),
            Interval(RealNotInt, None, -5, closed="left"),
        ),
        (
            Interval(Integral, 3, None, closed="left"),
            Interval(RealNotInt, None, -4, closed="neither"),
        ),
        (
            Interval(Integral, -5, 5, closed="both"),
            Interval(RealNotInt, None, 1, closed="right"),
        ),
        (
            Interval(Integral, -5, 5, closed="both"),
            Interval(RealNotInt, 1, None, closed="left"),
        ),
        (
            Interval(Integral, -5, 5, closed="both"),
            Interval(RealNotInt, -10, -4, closed="neither"),
        ),
        (
            Interval(Integral, -5, 5, closed="both"),
            Interval(RealNotInt, -10, -4, closed="right"),
        ),
        (
            Interval(Integral, -5, 5, closed="neither"),
            Interval(RealNotInt, 6, 10, closed="neither"),
        ),
        (
            Interval(Integral, -5, 5, closed="neither"),
            Interval(RealNotInt, 6, 10, closed="left"),
        ),
        (
            Interval(Integral, 2, None, closed="left"),
            Interval(RealNotInt, 0, 1, closed="both"),
        ),
        (
            Interval(Integral, 1, None, closed="left"),
            Interval(RealNotInt, 0, 1, closed="both"),
        ),
    ],
)
def test_generate_invalid_param_val_2_intervals(integer_interval, real_interval):
    # [FUNCTION DOC]
    # For pairs of an Integral interval and a RealNotInt interval, checks the invalid
    # value generated for either interval satisfies neither of them.
    """Check that the value generated for an interval constraint does not satisfy any of
    the interval constraints.
    """
    bad_value = generate_invalid_param_val(constraint=real_interval)
    assert not real_interval.is_satisfied_by(bad_value)
    assert not integer_interval.is_satisfied_by(bad_value)

    bad_value = generate_invalid_param_val(constraint=integer_interval)
    assert not real_interval.is_satisfied_by(bad_value)
    assert not integer_interval.is_satisfied_by(bad_value)


@pytest.mark.parametrize(
    "constraint",
    [
        _ArrayLikes(),
        _InstancesOf(list),
        _Callables(),
        _NoneConstraint(),
        _RandomStates(),
        _SparseMatrices(),
        _Booleans(),
        Interval(Integral, None, None, closed="neither"),
    ],
)
def test_generate_invalid_param_val_all_valid(constraint):
    # [FUNCTION DOC]
    # Checks generate_invalid_param_val raises NotImplementedError for constraints that
    # accept all values of their kind (array-likes, callables, None, booleans, unbounded
    # integer interval, etc.).
    """Check that the function raises NotImplementedError when there's no invalid value
    for the constraint.
    """
    with pytest.raises(NotImplementedError):
        generate_invalid_param_val(constraint)


@pytest.mark.parametrize(
    "constraint",
    [
        _ArrayLikes(),
        _Callables(),
        _InstancesOf(list),
        _NoneConstraint(),
        _RandomStates(),
        _SparseMatrices(),
        _Booleans(),
        _VerboseHelper(),
        MissingValues(),
        MissingValues(numeric_only=True),
        StrOptions({"a", "b", "c"}),
        Options(Integral, {1, 2, 3}),
        Interval(Integral, None, None, closed="neither"),
        Interval(Integral, 0, 10, closed="neither"),
        Interval(Integral, 0, None, closed="neither"),
        Interval(Integral, None, 0, closed="neither"),
        Interval(Real, 0, 1, closed="neither"),
        Interval(Real, 0, None, closed="both"),
        Interval(Real, None, 0, closed="right"),
        HasMethods("fit"),
        _IterablesNotString(),
        _CVObjects(),
    ],
)
def test_generate_valid_param(constraint):
    # [FUNCTION DOC]
    # For many constraints, checks generate_valid_param returns a value the constraint
    # accepts.
    """Check that the value generated does satisfy the constraint."""
    value = generate_valid_param(constraint)
    assert constraint.is_satisfied_by(value)


@pytest.mark.parametrize(
    "constraint_declaration, value",
    [
        (Interval(Real, 0, 1, closed="both"), 0.42),
        (Interval(Integral, 0, None, closed="neither"), 42),
        (StrOptions({"a", "b", "c"}), "b"),
        (Options(type, {np.float32, np.float64}), np.float64),
        (callable, lambda x: x + 1),
        (None, None),
        ("array-like", [[1, 2], [3, 4]]),
        ("array-like", np.array([[1, 2], [3, 4]])),
        ("sparse matrix", csr_matrix([[1, 2], [3, 4]])),
        ("sparse matrix", csr_array([[1, 2], [3, 4]])),
        *[
            ("sparse matrix", container([[1, 2], [3, 4]]))
            for container in CSR_CONTAINERS
        ],
        ("random_state", 0),
        ("random_state", np.random.RandomState(0)),
        ("random_state", None),
        (_Class, _Class()),
        (int, 1),
        (Real, 0.5),
        ("boolean", False),
        ("verbose", 1),
        ("nan", np.nan),
        (MissingValues(), -1),
        (MissingValues(), -1.0),
        (MissingValues(), 2**1028),
        (MissingValues(), None),
        (MissingValues(), float("nan")),
        (MissingValues(), np.nan),
        (MissingValues(), "missing"),
        (HasMethods("fit"), _Estimator(a=0)),
        ("cv_object", 5),
    ],
)
def test_is_satisfied_by(constraint_declaration, value):
    # [FUNCTION DOC]
    # For a variety of constraint declarations and matching values, checks the
    # constraint built by make_constraint accepts the value.
    """Sanity check for the is_satisfied_by method"""
    constraint = make_constraint(constraint_declaration)
    assert constraint.is_satisfied_by(value)


@pytest.mark.parametrize(
    "constraint_declaration, expected_constraint_class",
    [
        (Interval(Real, 0, 1, closed="both"), Interval),
        (StrOptions({"option1", "option2"}), StrOptions),
        (Options(Real, {0.42, 1.23}), Options),
        ("array-like", _ArrayLikes),
        ("sparse matrix", _SparseMatrices),
        ("random_state", _RandomStates),
        (None, _NoneConstraint),
        (callable, _Callables),
        (int, _InstancesOf),
        ("boolean", _Booleans),
        ("verbose", _VerboseHelper),
        (MissingValues(numeric_only=True), MissingValues),
        (HasMethods("fit"), HasMethods),
        ("cv_object", _CVObjects),
        ("nan", _NanConstraint),
        (np.nan, _NanConstraint),
    ],
)
def test_make_constraint(constraint_declaration, expected_constraint_class):
    # [FUNCTION DOC]
    # Checks make_constraint maps each declaration (objects, strings, None, callable,
    # types, nan) to the expected constraint class exactly.
    """Check that make_constraint dispatches to the appropriate constraint class"""
    constraint = make_constraint(constraint_declaration)
    assert constraint.__class__ is expected_constraint_class


def test_make_constraint_unknown():
    # [FUNCTION DOC]
    # Checks an unknown string declaration makes make_constraint raise ValueError.
    """Check that an informative error is raised when an unknown constraint is passed"""
    with pytest.raises(ValueError, match="Unknown constraint"):
        make_constraint("not a valid constraint")


def test_validate_params():
    # [FUNCTION DOC]
    # Calls _func with an invalid value for each of a, b, c, d passed positionally, via
    # unpacking, as keyword-only or with extra args/kwargs, and checks
    # InvalidParameterError names the offending parameter.
    """Check that validate_params works no matter how the arguments are passed"""
    with pytest.raises(
        InvalidParameterError, match="The 'a' parameter of _func must be"
    ):
        _func("wrong", c=1)

    with pytest.raises(
        InvalidParameterError, match="The 'b' parameter of _func must be"
    ):
        _func(*[1, "wrong"], c=1)

    with pytest.raises(
        InvalidParameterError, match="The 'c' parameter of _func must be"
    ):
        _func(1, **{"c": "wrong"})

    with pytest.raises(
        InvalidParameterError, match="The 'd' parameter of _func must be"
    ):
        _func(1, c=1, d="wrong")

    # check in the presence of extra positional and keyword args
    with pytest.raises(
        InvalidParameterError, match="The 'b' parameter of _func must be"
    ):
        _func(0, *["wrong", 2, 3], c=4, **{"e": 5})

    with pytest.raises(
        InvalidParameterError, match="The 'c' parameter of _func must be"
    ):
        _func(0, *[1, 2, 3], c="four", **{"e": 5})


def test_validate_params_missing_params():
    # [FUNCTION DOC]
    # Checks a validated function with a parameter (b) absent from the constraint dict
    # can be called without error.
    """Check that no error is raised when there are parameters without
    constraints
    """

    @validate_params({"a": [int]}, prefer_skip_nested_validation=True)
    def func(a, b):
        # [FUNCTION DOC]
        # HELPER: No-op function validating only a as int. Outputs: None.
        pass

    func(1, 2)


def test_decorate_validated_function():
    # [FUNCTION DOC]
    # Wraps _func with deprecated and checks calls emit FutureWarning and that
    # validation still raises InvalidParameterError for an invalid c.
    """Check that validate_params functions can be decorated"""
    decorated_function = deprecated()(_func)

    with pytest.warns(FutureWarning, match="Function _func is deprecated"):
        decorated_function(1, 2, c=3)

    # outer decorator does not interfere with validation
    with pytest.warns(FutureWarning, match="Function _func is deprecated"):
        with pytest.raises(
            InvalidParameterError, match=r"The 'c' parameter of _func must be"
        ):
            decorated_function(1, 2, c="wrong")


def test_validate_params_method():
    # [FUNCTION DOC]
    # Checks validated methods report the qualified name in InvalidParameterError and
    # that the deprecated validated method both warns and validates.
    """Check that validate_params works with methods"""
    with pytest.raises(
        InvalidParameterError, match="The 'a' parameter of _Class._method must be"
    ):
        _Class()._method("wrong")

    # validated method can be decorated
    with pytest.warns(FutureWarning, match="Function _deprecated_method is deprecated"):
        with pytest.raises(
            InvalidParameterError,
            match="The 'a' parameter of _Class._deprecated_method must be",
        ):
            _Class()._deprecated_method("wrong")


def test_validate_params_estimator():
    # [FUNCTION DOC]
    # Checks _Estimator accepts an invalid a at construction but fit raises
    # InvalidParameterError.
    """Check that validate_params works with Estimator instances"""
    # no validation in init
    est = _Estimator("wrong")

    with pytest.raises(
        InvalidParameterError, match="The 'a' parameter of _Estimator must be"
    ):
        est.fit()


def test_stroptions_deprecated_subset():
    # [FUNCTION DOC]
    # Checks StrOptions raises ValueError when deprecated options are not a subset.
    """Check that the deprecated parameter must be a subset of options."""
    with pytest.raises(ValueError, match="deprecated options must be a subset"):
        StrOptions({"a", "b", "c"}, deprecated={"a", "d"})


def test_hidden_constraint():
    # [FUNCTION DOC]
    # Checks a Hidden(list) constraint accepts lists but is omitted from the error
    # message (which mentions only dict) for invalid values.
    """Check that internal constraints are not exposed in the error message."""

    @validate_params(
        {"param": [Hidden(list), dict]}, prefer_skip_nested_validation=True
    )
    def f(param):
        # [FUNCTION DOC]
        # HELPER: No-op function whose param accepts a hidden list or a dict.
        pass

    # list and dict are valid params
    f({"a": 1, "b": 2, "c": 3})
    f([1, 2, 3])

    with pytest.raises(
        InvalidParameterError, match="The 'param' parameter"
    ) as exc_info:
        f(param="bad")

    # the list option is not exposed in the error message
    err_msg = str(exc_info.value)
    assert "an instance of 'dict'" in err_msg
    assert "an instance of 'list'" not in err_msg


def test_hidden_stroptions():
    # [FUNCTION DOC]
    # Checks a hidden StrOptions value ("warn") is accepted but not listed in the error
    # message, while the visible option "auto" is.
    """Check that we can have 2 StrOptions constraints, one being hidden."""

    @validate_params(
        {"param": [StrOptions({"auto"}), Hidden(StrOptions({"warn"}))]},
        prefer_skip_nested_validation=True,
    )
    def f(param):
        # [FUNCTION DOC]
        # HELPER: No-op function whose param accepts "auto" or a hidden "warn".
        pass

    # "auto" and "warn" are valid params
    f("auto")
    f("warn")

    with pytest.raises(
        InvalidParameterError, match="The 'param' parameter"
    ) as exc_info:
        f(param="bad")

    # the "warn" option is not exposed in the error message
    err_msg = str(exc_info.value)
    assert "auto" in err_msg
    assert "warn" not in err_msg


def test_validate_params_set_param_constraints_attribute():
    # [FUNCTION DOC]
    # Checks validate_params stores _skl_parameter_constraints on decorated functions
    # and bound methods.
    """Check that the validate_params decorator properly sets the parameter constraints
    as attribute of the decorated function/method.
    """
    assert hasattr(_func, "_skl_parameter_constraints")
    assert hasattr(_Class()._method, "_skl_parameter_constraints")


def test_boolean_constraint_deprecated_int():
    # [FUNCTION DOC]
    # Checks a "boolean" constraint accepts Python and numpy booleans.
    # KNOWN ISSUE: the name and docstring mention a deprecation warning for int values,
    # but the test only passes booleans and never checks ints or warnings.
    """Check that validate_params raise a deprecation message but still passes
    validation when using an int for a parameter accepting a boolean.
    """

    @validate_params({"param": ["boolean"]}, prefer_skip_nested_validation=True)
    def f(param):
        # [FUNCTION DOC]
        # HELPER: No-op function with a "boolean" constrained param.
        pass

    # True/False and np.bool_(True/False) are valid params
    f(True)
    f(np.bool_(False))


def test_no_validation():
    # [FUNCTION DOC]
    # Checks param1 (int or None) is validated while param2 marked "no_validation"
    # accepts any value, including a class and its instance.
    """Check that validation can be skipped for a parameter."""

    @validate_params(
        {"param1": [int, None], "param2": "no_validation"},
        prefer_skip_nested_validation=True,
    )
    def f(param1=None, param2=None):
        # [FUNCTION DOC]
        # HELPER: No-op function with param1 validated and param2 unvalidated (both
        # default None).
        pass

    # param1 is validated
    with pytest.raises(InvalidParameterError, match="The 'param1' parameter"):
        f(param1="wrong")

    # param2 is not validated: any type is valid.
    class SomeType:
        # [CLASS DOC]
        # HELPER: Arbitrary empty type used as an unvalidated parameter value.
        pass

    f(param2=SomeType)
    f(param2=SomeType())


def test_pandas_na_constraint_with_pd_na():
    # [FUNCTION DOC]
    # Skipped without pandas. Checks _PandasNAConstraint accepts pandas.NA and rejects
    # an array.
    """Add a specific test for checking support for `pandas.NA`."""
    pd = pytest.importorskip("pandas")

    na_constraint = _PandasNAConstraint()
    assert na_constraint.is_satisfied_by(pd.NA)
    assert not na_constraint.is_satisfied_by(np.array([1, 2, 3]))


def test_iterable_not_string():
    # [FUNCTION DOC]
    # Checks _IterablesNotString accepts a list and a range but rejects a string.
    """Check that a string does not satisfy the _IterableNotString constraint."""
    constraint = _IterablesNotString()
    assert constraint.is_satisfied_by([1, 2, 3])
    assert constraint.is_satisfied_by(range(10))
    assert not constraint.is_satisfied_by("some string")


def test_cv_objects():
    # [FUNCTION DOC]
    # Checks _CVObjects accepts an int, a CV splitter, an iterable of splits and None,
    # and rejects a string.
    """Check that the _CVObjects constraint accepts all current ways
    to pass cv objects."""
    constraint = _CVObjects()
    assert constraint.is_satisfied_by(5)
    assert constraint.is_satisfied_by(LeaveOneOut())
    assert constraint.is_satisfied_by([([1, 2], [3, 4]), ([3, 4], [1, 2])])
    assert constraint.is_satisfied_by(None)
    assert not constraint.is_satisfied_by("not a CV object")


def test_third_party_estimator():
    # [FUNCTION DOC]
    # Checks a subclass of _Estimator with different __init__ parameters (b) can fit
    # without validation errors about mismatched constraint names.
    """Check that the validation from a scikit-learn estimator inherited by a third
    party estimator does not impose a match between the dict of constraints and the
    parameters of the estimator.
    """

    class ThirdPartyEstimator(_Estimator):
        # [CLASS DOC]
        # HELPER: Third-party style estimator subclassing _Estimator with its own
        # parameter b.
        def __init__(self, b):
            # [FUNCTION DOC]
            # HELPER: Stores b and initializes the parent with a=0.
            # Side effects: sets self.b and self.a.
            self.b = b
            super().__init__(a=0)

        def fit(self, X=None, y=None):
            # [FUNCTION DOC]
            # HELPER: Delegates to _Estimator.fit (which validates a). Outputs: None.
            super().fit(X, y)

    # does not raise, even though "b" is not in the constraints dict and "a" is not
    # a parameter of the estimator.
    ThirdPartyEstimator(b=0).fit()


def test_interval_real_not_int():
    # [FUNCTION DOC]
    # Checks Interval(RealNotInt, 0, 1) accepts 1.0 but rejects the int 1.
    """Check for the type RealNotInt in the Interval constraint."""
    constraint = Interval(RealNotInt, 0, 1, closed="both")
    assert constraint.is_satisfied_by(1.0)
    assert not constraint.is_satisfied_by(1)


def test_real_not_int():
    # [FUNCTION DOC]
    # Checks isinstance with RealNotInt is True for Python/numpy floats and False for
    # Python/numpy ints.
    """Check for the RealNotInt type."""
    assert isinstance(1.0, RealNotInt)
    assert not isinstance(1, RealNotInt)
    assert isinstance(np.float64(1), RealNotInt)
    assert not isinstance(np.int64(1), RealNotInt)


def test_skip_param_validation():
    # [FUNCTION DOC]
    # Checks an invalid argument raises InvalidParameterError normally but not inside
    # config_context(skip_parameter_validation=True).
    """Check that param validation can be skipped using config_context."""

    @validate_params({"a": [int]}, prefer_skip_nested_validation=True)
    def f(a):
        # [FUNCTION DOC]
        # HELPER: No-op function validating a as int.
        pass

    with pytest.raises(InvalidParameterError, match="The 'a' parameter"):
        f(a="1")

    # does not raise
    with config_context(skip_parameter_validation=True):
        f(a="1")


@pytest.mark.parametrize("prefer_skip_nested_validation", [True, False])
def test_skip_nested_validation(prefer_skip_nested_validation):
    # [FUNCTION DOC]
    # Checks the outer function g is always validated, and that its call to an inner
    # validated f with a bad value is skipped when g's prefer_skip_nested_validation is
    # True and raises InvalidParameterError otherwise.
    """Check that nested validation can be skipped."""

    @validate_params({"a": [int]}, prefer_skip_nested_validation=True)
    def f(a):
        # [FUNCTION DOC]
        # HELPER: No-op inner function validating a as int.
        pass

    @validate_params(
        {"b": [int]},
        prefer_skip_nested_validation=prefer_skip_nested_validation,
    )
    def g(b):
        # [FUNCTION DOC]
        # HELPER: Outer validated function (b int) that calls f with an invalid a.
        # Outputs: f's return value (None).
        # Exceptions: InvalidParameterError for invalid b, or from f when nested
        # validation is not skipped.
        # calls f with a bad parameter type
        return f(a="invalid_param_value")

    # Validation for g is never skipped.
    with pytest.raises(InvalidParameterError, match="The 'b' parameter"):
        g(b="invalid_param_value")

    if prefer_skip_nested_validation:
        g(b=1)  # does not raise because inner f is not validated
    else:
        with pytest.raises(InvalidParameterError, match="The 'a' parameter"):
            g(b=1)


@pytest.mark.parametrize(
    "skip_parameter_validation, prefer_skip_nested_validation, expected_skipped",
    [
        (True, True, True),
        (True, False, True),
        (False, True, True),
        (False, False, False),
    ],
)
def test_skip_nested_validation_and_config_context(
    skip_parameter_validation, prefer_skip_nested_validation, expected_skipped
):
    # [FUNCTION DOC]
    # Checks the skip_parameter_validation value seen inside a validated function is
    # True when either the global config or prefer_skip_nested_validation enables
    # skipping, and False only when both are False.
    """Check interaction between global skip and local skip."""

    @validate_params(
        {"a": [int]}, prefer_skip_nested_validation=prefer_skip_nested_validation
    )
    def g(a):
        # [FUNCTION DOC]
        # HELPER: Validated function returning the current skip_parameter_validation
        # config value as seen inside its body.
        return get_config()["skip_parameter_validation"]

    with config_context(skip_parameter_validation=skip_parameter_validation):
        actual_skipped = g(1)

    assert actual_skipped == expected_skipped
