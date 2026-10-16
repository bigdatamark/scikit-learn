# [MODULE DOC]
# Test suite for sklearn.preprocessing.FunctionTransformer: delegation of transform and
#   inverse_transform to user callables, kw_args/inv_kw_args forwarding, check_inverse
#   warnings and errors, get_feature_names_out with "one-to-one" or callable
#   feature_names_out, set_output interaction (pandas/polars), and column-name
#   consistency of dataframe outputs.
# Side effects: none at import beyond imports; several tests use the global numpy random
#   state without seeding.
import warnings

import numpy as np
import pytest

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.utils._testing import (
    _convert_container,
    assert_allclose_dense_sparse,
    assert_array_equal,
)
from sklearn.utils.fixes import CSC_CONTAINERS, CSR_CONTAINERS


def _make_func(args_store, kwargs_store, func=lambda X, *a, **k: X):
    # [FUNCTION DOC]
    # HELPER: Builds a spy callable that records how FunctionTransformer invokes it.
    # Inputs: args_store - list that receives X followed by any extra positional args;
    #   kwargs_store - dict updated with keyword args; func - callable applied to X for
    #   the return value, defaults to identity.
    # Outputs: the recording function, which returns func applied to X only (extra args
    #   are recorded but not forwarded).
    # Side effects: the returned function mutates args_store and kwargs_store on every
    #   call.
    def _func(X, *args, **kwargs):
        # [FUNCTION DOC]
        # HELPER: Recording wrapper: appends X and extra positional args to args_store,
        #   merges keyword args into kwargs_store, and returns func(X).
        args_store.append(X)
        args_store.extend(args)
        kwargs_store.update(kwargs)
        return func(X)

    return _func


def test_delegate_to_func():
    # [FUNCTION DOC]
    # Verifies transform delegates to func with only X as positional argument and no
    #   keyword arguments, and returns X unchanged for the identity spy; repeated twice
    #   with reset stores.
    # KNOWN ISSUE: the two halves are identical (both build the transformer the same
    #   way), so the second check adds no coverage.
    # (args|kwargs)_store will hold the positional and keyword arguments
    # passed to the function inside the FunctionTransformer.
    args_store = []
    kwargs_store = {}
    X = np.arange(10).reshape((5, 2))
    assert_array_equal(
        FunctionTransformer(_make_func(args_store, kwargs_store)).transform(X),
        X,
        "transform should have returned X unchanged",
    )

    # The function should only have received X.
    assert args_store == [X], (
        "Incorrect positional arguments passed to func: {args}".format(args=args_store)
    )

    assert not kwargs_store, (
        "Unexpected keyword arguments passed to func: {args}".format(args=kwargs_store)
    )

    # reset the argument stores.
    args_store[:] = []
    kwargs_store.clear()
    transformed = FunctionTransformer(
        _make_func(args_store, kwargs_store),
    ).transform(X)

    assert_array_equal(
        transformed, X, err_msg="transform should have returned X unchanged"
    )

    # The function should have received X
    assert args_store == [X], (
        "Incorrect positional arguments passed to func: {args}".format(args=args_store)
    )

    assert not kwargs_store, (
        "Unexpected keyword arguments passed to func: {args}".format(args=kwargs_store)
    )


def test_np_log():
    # [FUNCTION DOC]
    # Verifies FunctionTransformer(np.log1p).transform matches np.log1p on an integer
    #   array.
    X = np.arange(10).reshape((5, 2))

    # Test that the numpy.log example still works.
    assert_array_equal(
        FunctionTransformer(np.log1p).transform(X),
        np.log1p(X),
    )


def test_kw_arg():
    # [FUNCTION DOC]
    # Verifies kw_args given at construction are forwarded to func (np.around with
    #   decimals=3).
    X = np.linspace(0, 1, num=10).reshape((5, 2))

    F = FunctionTransformer(np.around, kw_args=dict(decimals=3))

    # Test that rounding is correct
    assert_array_equal(F.transform(X), np.around(X, decimals=3))


def test_kw_arg_update():
    # [FUNCTION DOC]
    # Verifies in-place mutation of the kw_args dict after construction is honored at
    #   transform time.
    X = np.linspace(0, 1, num=10).reshape((5, 2))

    F = FunctionTransformer(np.around, kw_args=dict(decimals=3))

    F.kw_args["decimals"] = 1

    # Test that rounding is correct
    assert_array_equal(F.transform(X), np.around(X, decimals=1))


def test_kw_arg_reset():
    # [FUNCTION DOC]
    # Verifies reassigning kw_args to a new dict after construction is honored at
    #   transform time.
    X = np.linspace(0, 1, num=10).reshape((5, 2))

    F = FunctionTransformer(np.around, kw_args=dict(decimals=3))

    F.kw_args = dict(decimals=1)

    # Test that rounding is correct
    assert_array_equal(F.transform(X), np.around(X, decimals=1))


def test_inverse_transform():
    # [FUNCTION DOC]
    # Verifies inverse_transform applies inverse_func with inv_kw_args (sqrt then around
    #   to 3 decimals).
    X = np.array([1, 4, 9, 16]).reshape((2, 2))

    # Test that inverse_transform works correctly
    F = FunctionTransformer(
        func=np.sqrt,
        inverse_func=np.around,
        inv_kw_args=dict(decimals=3),
    )
    assert_array_equal(
        F.inverse_transform(F.transform(X)),
        np.around(np.sqrt(X), decimals=3),
    )


@pytest.mark.parametrize("sparse_container", [None] + CSC_CONTAINERS + CSR_CONTAINERS)
def test_check_inverse(sparse_container):
    # [FUNCTION DOC]
    # Verifies check_inverse=True emits the "not strictly inverse" UserWarning at fit
    #   for sqrt/around, and emits no warning for the true inverse pair expm1/log1p,
    #   whose round trip must reproduce X; run for dense and every CSC/CSR container.
    # Inputs: sparse_container - None for dense, else a sparse matrix/array constructor.
    X = np.array([1, 4, 9, 16], dtype=np.float64).reshape((2, 2))
    if sparse_container is not None:
        X = sparse_container(X)

    trans = FunctionTransformer(
        func=np.sqrt,
        inverse_func=np.around,
        accept_sparse=sparse_container is not None,
        check_inverse=True,
        validate=True,
    )
    warning_message = (
        "The provided functions are not strictly"
        " inverse of each other. If you are sure you"
        " want to proceed regardless, set"
        " 'check_inverse=False'."
    )
    with pytest.warns(UserWarning, match=warning_message):
        trans.fit(X)

    trans = FunctionTransformer(
        func=np.expm1,
        inverse_func=np.log1p,
        accept_sparse=sparse_container is not None,
        check_inverse=True,
        validate=True,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        Xt = trans.fit_transform(X)

    assert_allclose_dense_sparse(X, trans.inverse_transform(Xt))


def test_check_inverse_func_or_inverse_not_provided():
    # [FUNCTION DOC]
    # Verifies no inverse check (hence no warning, warnings turned into errors) happens
    #   when either func or inverse_func is None.
    # check that we don't check inverse when one of the func or inverse is not
    # provided.
    X = np.array([1, 4, 9, 16], dtype=np.float64).reshape((2, 2))

    trans = FunctionTransformer(
        func=np.expm1, inverse_func=None, check_inverse=True, validate=True
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        trans.fit(X)
    trans = FunctionTransformer(
        func=None, inverse_func=np.expm1, check_inverse=True, validate=True
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        trans.fit(X)


def test_function_transformer_frame():
    # [FUNCTION DOC]
    # Verifies the default (identity) transformer preserves a pandas DataFrame (output
    #   has .loc). Skipped without pandas.
    pd = pytest.importorskip("pandas")
    X_df = pd.DataFrame(np.random.randn(100, 10))
    transformer = FunctionTransformer()
    X_df_trans = transformer.fit_transform(X_df)
    assert hasattr(X_df_trans, "loc")


@pytest.mark.parametrize("X_type", ["array", "series"])
def test_function_transformer_raise_error_with_mixed_dtype(X_type):
    # [FUNCTION DOC]
    # Verifies fit with check_inverse=True raises ValueError when X holds object data
    #   with mixed strings and ints, for array and pandas Series inputs.
    # Inputs: X_type - "array" or "series", container type passed to _convert_container.
    """Check that `FunctionTransformer.check_inverse` raises error on mixed dtype."""
    mapping = {"one": 1, "two": 2, "three": 3, 5: "five", 6: "six"}
    inverse_mapping = {value: key for key, value in mapping.items()}
    dtype = "object"

    data = ["one", "two", "three", "one", "one", 5, 6]
    data = _convert_container(data, X_type, column_names=["value"], dtype=dtype)

    def func(X):
        # [FUNCTION DOC]
        # HELPER: Maps each element of the 1-D input through the forward mapping dict,
        #   returning an object ndarray.
        return np.array([mapping[X[i]] for i in range(X.size)], dtype=object)

    def inverse_func(X):
        # [FUNCTION DOC]
        # HELPER: Maps each element back through inverse_mapping and converts the list
        #   to the container type X_type with column name "value" and object dtype.
        return _convert_container(
            [inverse_mapping[x] for x in X],
            X_type,
            column_names=["value"],
            dtype=dtype,
        )

    transformer = FunctionTransformer(
        func=func, inverse_func=inverse_func, validate=False, check_inverse=True
    )

    msg = "'check_inverse' is only supported when all the elements in `X` is numerical."
    with pytest.raises(ValueError, match=msg):
        transformer.fit(data)


def test_function_transformer_support_all_numerical_dataframes_check_inverse_True():
    # [FUNCTION DOC]
    # Verifies check_inverse=True works for an all-numeric DataFrame (x+2 / x-2) without
    #   error and the output equals df + 2. Skipped without pandas.
    """Check support for dataframes with only numerical values."""
    pd = pytest.importorskip("pandas")

    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    transformer = FunctionTransformer(
        func=lambda x: x + 2, inverse_func=lambda x: x - 2, check_inverse=True
    )

    # Does not raise an error
    df_out = transformer.fit_transform(df)
    assert_allclose_dense_sparse(df_out, df + 2)


def test_function_transformer_with_dataframe_and_check_inverse_True():
    # [FUNCTION DOC]
    # Non-regression for gh-25261: fit with check_inverse=True on a DataFrame with a
    #   string column raises ValueError about numerical elements. Skipped without
    #   pandas.
    """Check error is raised when check_inverse=True.

    Non-regression test for gh-25261.
    """
    pd = pytest.importorskip("pandas")
    transformer = FunctionTransformer(
        func=lambda x: x, inverse_func=lambda x: x, check_inverse=True
    )

    df_mixed = pd.DataFrame({"a": [1, 2, 3], "b": ["a", "b", "c"]})
    msg = "'check_inverse' is only supported when all the elements in `X` is numerical."
    with pytest.raises(ValueError, match=msg):
        transformer.fit(df_mixed)


@pytest.mark.parametrize(
    "X, feature_names_out, input_features, expected",
    [
        (
            # NumPy inputs, default behavior: generate names
            np.random.rand(100, 3),
            "one-to-one",
            None,
            ("x0", "x1", "x2"),
        ),
        (
            # Pandas input, default behavior: use input feature names
            {"a": np.random.rand(100), "b": np.random.rand(100)},
            "one-to-one",
            None,
            ("a", "b"),
        ),
        (
            # NumPy input, feature_names_out=callable
            np.random.rand(100, 3),
            lambda transformer, input_features: ("a", "b"),
            None,
            ("a", "b"),
        ),
        (
            # Pandas input, feature_names_out=callable
            {"a": np.random.rand(100), "b": np.random.rand(100)},
            lambda transformer, input_features: ("c", "d", "e"),
            None,
            ("c", "d", "e"),
        ),
        (
            # NumPy input, feature_names_out=callable – default input_features
            np.random.rand(100, 3),
            lambda transformer, input_features: tuple(input_features) + ("a",),
            None,
            ("x0", "x1", "x2", "a"),
        ),
        (
            # Pandas input, feature_names_out=callable – default input_features
            {"a": np.random.rand(100), "b": np.random.rand(100)},
            lambda transformer, input_features: tuple(input_features) + ("c",),
            None,
            ("a", "b", "c"),
        ),
        (
            # NumPy input, input_features=list of names
            np.random.rand(100, 3),
            "one-to-one",
            ("a", "b", "c"),
            ("a", "b", "c"),
        ),
        (
            # Pandas input, input_features=list of names
            {"a": np.random.rand(100), "b": np.random.rand(100)},
            "one-to-one",
            ("a", "b"),  # must match feature_names_in_
            ("a", "b"),
        ),
        (
            # NumPy input, feature_names_out=callable, input_features=list
            np.random.rand(100, 3),
            lambda transformer, input_features: tuple(input_features) + ("d",),
            ("a", "b", "c"),
            ("a", "b", "c", "d"),
        ),
        (
            # Pandas input, feature_names_out=callable, input_features=list
            {"a": np.random.rand(100), "b": np.random.rand(100)},
            lambda transformer, input_features: tuple(input_features) + ("c",),
            ("a", "b"),  # must match feature_names_in_
            ("a", "b", "c"),
        ),
    ],
)
@pytest.mark.parametrize("validate", [True, False])
def test_function_transformer_get_feature_names_out(
    X, feature_names_out, input_features, expected, validate
):
    # [FUNCTION DOC]
    # Verifies get_feature_names_out returns an object ndarray equal to expected for
    #   numpy and pandas inputs, with "one-to-one" or callable feature_names_out, with
    #   or without explicit input_features, and for validate True and False.
    # Inputs: X - ndarray or dict (converted to DataFrame); feature_names_out - "one-to-
    #   one" or callable(transformer, input_features); input_features - None or names
    #   tuple; expected - expected names; validate - bool.
    if isinstance(X, dict):
        pd = pytest.importorskip("pandas")
        X = pd.DataFrame(X)

    transformer = FunctionTransformer(
        feature_names_out=feature_names_out, validate=validate
    )
    transformer.fit(X)
    names = transformer.get_feature_names_out(input_features)
    assert isinstance(names, np.ndarray)
    assert names.dtype == object
    assert_array_equal(names, expected)


def test_function_transformer_get_feature_names_out_without_validation():
    # [FUNCTION DOC]
    # Verifies get_feature_names_out with explicit input_features works when
    #   validate=False on a numpy array.
    transformer = FunctionTransformer(feature_names_out="one-to-one", validate=False)
    X = np.random.rand(100, 2)
    transformer.fit_transform(X)

    names = transformer.get_feature_names_out(("a", "b"))
    assert isinstance(names, np.ndarray)
    assert names.dtype == object
    assert_array_equal(names, ("a", "b"))


def test_function_transformer_feature_names_out_is_None():
    # [FUNCTION DOC]
    # Verifies get_feature_names_out is unavailable (AttributeError) when
    #   feature_names_out is None (default).
    transformer = FunctionTransformer()
    X = np.random.rand(100, 2)
    transformer.fit_transform(X)

    msg = "This 'FunctionTransformer' has no attribute 'get_feature_names_out'"
    with pytest.raises(AttributeError, match=msg):
        transformer.get_feature_names_out()


def test_function_transformer_feature_names_out_uses_estimator():
    # [FUNCTION DOC]
    # Verifies the feature_names_out callable receives the fitted transformer (reading
    #   kw_args["n"]) and builds names from DataFrame columns plus rnd0..rnd2. Skipped
    #   without pandas.
    def add_n_random_features(X, n):
        # [FUNCTION DOC]
        # HELPER: Appends n columns of uniform random values to X (uses the global numpy
        #   random state).
        return np.concatenate([X, np.random.rand(len(X), n)], axis=1)

    def feature_names_out(transformer, input_features):
        # [FUNCTION DOC]
        # HELPER: Returns input_features followed by "rnd0".."rnd{n-1}", where n is read
        #   from transformer.kw_args.
        n = transformer.kw_args["n"]
        return list(input_features) + [f"rnd{i}" for i in range(n)]

    transformer = FunctionTransformer(
        func=add_n_random_features,
        feature_names_out=feature_names_out,
        kw_args=dict(n=3),
        validate=True,
    )
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame({"a": np.random.rand(100), "b": np.random.rand(100)})
    transformer.fit_transform(df)
    names = transformer.get_feature_names_out()

    assert isinstance(names, np.ndarray)
    assert names.dtype == object
    assert_array_equal(names, ("a", "b", "rnd0", "rnd1", "rnd2"))


def test_function_transformer_validate_inverse():
    # [FUNCTION DOC]
    # Verifies inverse_transform with validate=True does not reset n_features_in_ (stays
    #   at the input width after a forward transform that adds a column).
    """Test that function transformer does not reset estimator in
    `inverse_transform`."""

    def add_constant_feature(X):
        # [FUNCTION DOC]
        # HELPER: Appends a column of ones to the 2-D array X.
        X_one = np.ones((X.shape[0], 1))
        return np.concatenate((X, X_one), axis=1)

    def inverse_add_constant(X):
        # [FUNCTION DOC]
        # HELPER: Drops the last column of the 2-D array X.
        return X[:, :-1]

    X = np.array([[1, 2], [3, 4], [3, 4]])
    trans = FunctionTransformer(
        func=add_constant_feature,
        inverse_func=inverse_add_constant,
        validate=True,
    )
    X_trans = trans.fit_transform(X)
    assert trans.n_features_in_ == X.shape[1]

    trans.inverse_transform(X_trans)
    assert trans.n_features_in_ == X.shape[1]


@pytest.mark.parametrize(
    "feature_names_out, expected",
    [
        ("one-to-one", ["pet", "color"]),
        [lambda est, names: [f"{n}_out" for n in names], ["pet_out", "color_out"]],
    ],
)
@pytest.mark.parametrize("in_pipeline", [True, False])
def test_get_feature_names_out_dataframe_with_string_data(
    feature_names_out, expected, in_pipeline
):
    # [FUNCTION DOC]
    # Verifies get_feature_names_out on a DataFrame with string columns, standalone and
    #   inside a pipeline, returns the expected object array and the output stays a
    #   DataFrame. Skipped without pandas.
    # Inputs: feature_names_out - "one-to-one" or a suffixing callable; expected -
    #   expected names; in_pipeline - wrap in make_pipeline when True.
    """Check that get_feature_names_out works with DataFrames with string data."""
    pd = pytest.importorskip("pandas")
    X = pd.DataFrame({"pet": ["dog", "cat"], "color": ["red", "green"]})

    def func(X):
        # [FUNCTION DOC]
        # HELPER: Returns X unchanged for "one-to-one", otherwise renames its columns
        #   using the feature_names_out callable (closure over the test parameter).
        if feature_names_out == "one-to-one":
            return X
        else:
            name = feature_names_out(None, X.columns)
            return X.rename(columns=dict(zip(X.columns, name)))

    transformer = FunctionTransformer(func=func, feature_names_out=feature_names_out)
    if in_pipeline:
        transformer = make_pipeline(transformer)

    X_trans = transformer.fit_transform(X)
    assert isinstance(X_trans, pd.DataFrame)

    names = transformer.get_feature_names_out()
    assert isinstance(names, np.ndarray)
    assert names.dtype == object
    assert_array_equal(names, expected)


def test_set_output_func():
    # [FUNCTION DOC]
    # Verifies set_output behavior: no warning when feature_names_out is set or when
    #   func returns a DataFrame; a UserWarning when func returns an ndarray with pandas
    #   or polars output configured; no warning with "default" output. Skipped without
    #   pandas.
    # KNOWN ISSUE: the polars loop iteration needs polars installed but the test only
    #   skips on missing pandas.
    """Check behavior of set_output with different settings."""
    pd = pytest.importorskip("pandas")

    X = pd.DataFrame({"a": [1, 2, 3], "b": [10, 20, 100]})

    ft = FunctionTransformer(np.log, feature_names_out="one-to-one")

    # no warning is raised when feature_names_out is defined
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        ft.set_output(transform="pandas")

    X_trans = ft.fit_transform(X)
    assert isinstance(X_trans, pd.DataFrame)
    assert_array_equal(X_trans.columns, ["a", "b"])

    ft = FunctionTransformer(lambda x: 2 * x)
    ft.set_output(transform="pandas")

    # no warning is raised when func returns a panda dataframe
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        X_trans = ft.fit_transform(X)
    assert isinstance(X_trans, pd.DataFrame)
    assert_array_equal(X_trans.columns, ["a", "b"])

    # Warning is raised when func returns an ndarray
    ft_np = FunctionTransformer(lambda x: np.asarray(x))

    for transform in ("pandas", "polars"):
        ft_np.set_output(transform=transform)
        msg = (
            f"When `set_output` is configured to be '{transform}'.*{transform} "
            "DataFrame.*"
        )
        with pytest.warns(UserWarning, match=msg):
            ft_np.fit_transform(X)

    # default transform does not warn
    ft_np.set_output(transform="default")
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        ft_np.fit_transform(X)


def test_consistence_column_name_between_steps():
    # [FUNCTION DOC]
    # Non-regression for gh-27695: the pipeline FunctionTransformer(log1p, suffixing
    #   names) + StandardScaler exposes names a__log, b__log and the final output is an
    #   ndarray. Skipped without pandas.
    """Check that we have a consistence between the feature names out of
    `FunctionTransformer` and the feature names in of the next step in the pipeline.

    Non-regression test for:
    https://github.com/scikit-learn/scikit-learn/issues/27695
    """
    pd = pytest.importorskip("pandas")

    def with_suffix(_, names):
        # [FUNCTION DOC]
        # HELPER: Appends "__log" to each name; the first (transformer) argument is
        #   ignored.
        return [name + "__log" for name in names]

    pipeline = make_pipeline(
        FunctionTransformer(np.log1p, feature_names_out=with_suffix), StandardScaler()
    )

    df = pd.DataFrame([[1, 2], [3, 4], [5, 6]], columns=["a", "b"])
    X_trans = pipeline.fit_transform(df)
    assert pipeline.get_feature_names_out().tolist() == ["a__log", "b__log"]
    # StandardScaler will convert to a numpy array
    assert isinstance(X_trans, np.ndarray)


@pytest.mark.parametrize("dataframe_lib", ["pandas", "polars"])
@pytest.mark.parametrize("transform_output", ["default", "pandas", "polars"])
def test_function_transformer_overwrite_column_names(dataframe_lib, transform_output):
    # [FUNCTION DOC]
    # Verifies a callable feature_names_out renames output dataframe columns (values
    #   unchanged) for pandas/polars inputs and every transform_output setting, and the
    #   names match get_feature_names_out.
    # Inputs: dataframe_lib - "pandas"/"polars"; transform_output -
    #   "default"/"pandas"/"polars".
    # KNOWN ISSUE: the guard compares transform_output with "numpy", which is never a
    #   parameter value, so importorskip("default") is always called and the "default"
    #   case is always skipped.
    """Check that we overwrite the column names when we should."""
    lib = pytest.importorskip(dataframe_lib)
    if transform_output != "numpy":
        pytest.importorskip(transform_output)

    df = lib.DataFrame({"a": [1, 2, 3], "b": [10, 20, 100]})

    def with_suffix(_, names):
        # [FUNCTION DOC]
        # HELPER: Appends "__log" to each name; the first argument is ignored.
        return [name + "__log" for name in names]

    transformer = FunctionTransformer(feature_names_out=with_suffix).set_output(
        transform=transform_output
    )
    X_trans = transformer.fit_transform(df)
    assert_array_equal(np.asarray(X_trans), np.asarray(df))

    feature_names = transformer.get_feature_names_out()
    assert list(X_trans.columns) == with_suffix(None, df.columns)
    assert feature_names.tolist() == with_suffix(None, df.columns)


@pytest.mark.parametrize(
    "feature_names_out",
    ["one-to-one", lambda _, names: [f"{name}_log" for name in names]],
)
def test_function_transformer_overwrite_column_names_numerical(feature_names_out):
    # [FUNCTION DOC]
    # Verifies with integer pandas column names that output columns equal
    #   get_feature_names_out for "one-to-one" and a suffixing callable. Skipped without
    #   pandas.
    """Check the same as `test_function_transformer_overwrite_column_names`
    but for the specific case of pandas where column names can be numerical."""
    pd = pytest.importorskip("pandas")

    df = pd.DataFrame({0: [1, 2, 3], 1: [10, 20, 100]})

    transformer = FunctionTransformer(feature_names_out=feature_names_out)
    X_trans = transformer.fit_transform(df)
    assert_array_equal(np.asarray(X_trans), np.asarray(df))

    feature_names = transformer.get_feature_names_out()
    assert list(X_trans.columns) == list(feature_names)


@pytest.mark.parametrize("dataframe_lib", ["pandas", "polars"])
@pytest.mark.parametrize(
    "feature_names_out",
    ["one-to-one", lambda _, names: [f"{name}_log" for name in names]],
)
def test_function_transformer_error_column_inconsistent(
    dataframe_lib, feature_names_out
):
    # [FUNCTION DOC]
    # Verifies fit_transform raises ValueError when func renames a column so output
    #   columns disagree with get_feature_names_out, for pandas and polars.
    """Check that we raise an error when `func` returns a dataframe with new
    column names that become inconsistent with `get_feature_names_out`."""
    lib = pytest.importorskip(dataframe_lib)

    df = lib.DataFrame({"a": [1, 2, 3], "b": [10, 20, 100]})

    def func(df):
        # [FUNCTION DOC]
        # HELPER: Renames column "a" to "c" using the pandas or polars rename API
        #   depending on dataframe_lib.
        if dataframe_lib == "pandas":
            return df.rename(columns={"a": "c"})
        else:
            return df.rename({"a": "c"})

    transformer = FunctionTransformer(func=func, feature_names_out=feature_names_out)
    err_msg = "The output generated by `func` have different column names"
    with pytest.raises(ValueError, match=err_msg):
        transformer.fit_transform(df).columns
