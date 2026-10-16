# [MODULE DOC]
# Tests for sklearn.utils._set_output: the pandas/polars container adapters,
# _SetOutputMixin (set_output method, automatic wrapping of transform outputs,
# __init_subclass__ handling), _safe_set_output, _get_output_config,
# _wrap_data_with_container, check_library_installed and _get_adapter_from_container.
# Defines several minimal dummy estimators used as fixtures.
import importlib
from collections import namedtuple

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from sklearn._config import config_context, get_config
from sklearn.preprocessing import StandardScaler
from sklearn.utils._set_output import (
    ADAPTERS_MANAGER,
    ContainerAdapterProtocol,
    _get_adapter_from_container,
    _get_output_config,
    _safe_set_output,
    _SetOutputMixin,
    _wrap_data_with_container,
    check_library_installed,
)
from sklearn.utils.fixes import CSR_CONTAINERS


def test_pandas_adapter():
    # [FUNCTION DOC]
    # Verifies the pandas adapter: create_container builds a DataFrame from ndarray with
    # callable or array columns, reusing the original DataFrame/Series index; existing
    # DataFrame inputs keep their own index; is_supported_container accepts DataFrames
    # only; rename_columns works even with duplicate names (gh-28260); hstack
    # concatenates column-wise; inplace=False copies while inplace=True renames the
    # input object itself. Skipped without pandas.
    """Check pandas adapter has expected behavior."""
    pd = pytest.importorskip("pandas")
    X_np = np.asarray([[1, 0, 3], [0, 0, 1]])
    columns = np.asarray(["f0", "f1", "f2"], dtype=object)
    index = np.asarray([1, 2])
    X_df_orig = pd.DataFrame([[1, 2], [1, 3]], index=index)
    X_ser_orig = pd.Series([2, 3], index=index)

    adapter = ADAPTERS_MANAGER.adapters["pandas"]
    X_container = adapter.create_container(X_np, X_df_orig, columns=lambda: columns)
    assert isinstance(X_container, pd.DataFrame)
    assert_array_equal(X_container.columns, columns)
    assert_array_equal(X_container.index, index)

    # use original index when the original is a series
    X_container = adapter.create_container(X_np, X_ser_orig, columns=lambda: columns)
    assert isinstance(X_container, pd.DataFrame)
    assert_array_equal(X_container.columns, columns)
    assert_array_equal(X_container.index, index)

    # Input dataframe's index does not change
    new_columns = np.asarray(["f0", "f1"], dtype=object)
    X_df = pd.DataFrame([[1, 2], [1, 3]], index=[10, 12])
    new_df = adapter.create_container(X_df, X_df_orig, columns=new_columns)
    assert_array_equal(new_df.columns, new_columns)
    assert_array_equal(new_df.index, X_df.index)

    assert adapter.is_supported_container(X_df)
    assert not adapter.is_supported_container(X_np)

    # adapter.update_columns updates the columns
    new_columns = np.array(["a", "c"], dtype=object)
    new_df = adapter.rename_columns(X_df, new_columns)
    assert_array_equal(new_df.columns, new_columns)

    # adapter.hstack stacks the dataframes horizontally.
    X_df_1 = pd.DataFrame([[1, 2, 5], [3, 4, 6]], columns=["a", "b", "e"])
    X_df_2 = pd.DataFrame([[4], [5]], columns=["c"])
    X_stacked = adapter.hstack([X_df_1, X_df_2])

    expected_df = pd.DataFrame(
        [[1, 2, 5, 4], [3, 4, 6, 5]], columns=["a", "b", "e", "c"]
    )
    pd.testing.assert_frame_equal(X_stacked, expected_df)

    # check that we update properly the columns even with duplicate column names
    # this use-case potentially happen when using ColumnTransformer
    # non-regression test for gh-28260
    X_df = pd.DataFrame([[1, 2], [1, 3]], columns=["a", "a"])
    new_columns = np.array(["x__a", "y__a"], dtype=object)
    new_df = adapter.rename_columns(X_df, new_columns)
    assert_array_equal(new_df.columns, new_columns)

    # check the behavior of the inplace parameter in `create_container`
    # we should trigger a copy
    X_df = pd.DataFrame([[1, 2], [1, 3]], index=index)
    X_output = adapter.create_container(X_df, X_df, columns=["a", "b"], inplace=False)
    assert X_output is not X_df
    assert list(X_df.columns) == [0, 1]
    assert list(X_output.columns) == ["a", "b"]

    # the operation is inplace
    X_df = pd.DataFrame([[1, 2], [1, 3]], index=index)
    X_output = adapter.create_container(X_df, X_df, columns=["a", "b"], inplace=True)
    assert X_output is X_df
    assert list(X_df.columns) == ["a", "b"]
    assert list(X_output.columns) == ["a", "b"]


def test_polars_adapter():
    # [FUNCTION DOC]
    # Verifies the polars adapter: create_container from ndarray and from a DataFrame
    # with new columns, is_supported_container, rename_columns, hstack, and copy vs in-
    # place behavior of create_container's inplace flag. Skipped without polars.
    """Check Polars adapter has expected behavior."""
    pl = pytest.importorskip("polars")
    X_np = np.array([[1, 0, 3], [0, 0, 1]])
    columns = ["f1", "f2", "f3"]
    X_df_orig = pl.DataFrame(X_np, schema=columns, orient="row")

    adapter = ADAPTERS_MANAGER.adapters["polars"]
    X_container = adapter.create_container(X_np, X_df_orig, columns=lambda: columns)

    assert isinstance(X_container, pl.DataFrame)
    assert_array_equal(X_container.columns, columns)

    # Update columns with create_container
    new_columns = np.asarray(["a", "b", "c"], dtype=object)
    new_df = adapter.create_container(X_df_orig, X_df_orig, columns=new_columns)
    assert_array_equal(new_df.columns, new_columns)

    assert adapter.is_supported_container(X_df_orig)
    assert not adapter.is_supported_container(X_np)

    # adapter.update_columns updates the columns
    new_columns = np.array(["a", "c", "g"], dtype=object)
    new_df = adapter.rename_columns(X_df_orig, new_columns)
    assert_array_equal(new_df.columns, new_columns)

    # adapter.hstack stacks the dataframes horizontally.
    X_df_1 = pl.DataFrame([[1, 2, 5], [3, 4, 6]], schema=["a", "b", "e"], orient="row")
    X_df_2 = pl.DataFrame([[4], [5]], schema=["c"], orient="row")
    X_stacked = adapter.hstack([X_df_1, X_df_2])

    expected_df = pl.DataFrame(
        [[1, 2, 5, 4], [3, 4, 6, 5]], schema=["a", "b", "e", "c"], orient="row"
    )
    from polars.testing import assert_frame_equal

    assert_frame_equal(X_stacked, expected_df)

    # check the behavior of the inplace parameter in `create_container`
    # we should trigger a copy
    X_df = pl.DataFrame([[1, 2], [1, 3]], schema=["a", "b"], orient="row")
    X_output = adapter.create_container(X_df, X_df, columns=["c", "d"], inplace=False)
    assert X_output is not X_df
    assert list(X_df.columns) == ["a", "b"]
    assert list(X_output.columns) == ["c", "d"]

    # the operation is inplace
    X_df = pl.DataFrame([[1, 2], [1, 3]], schema=["a", "b"], orient="row")
    X_output = adapter.create_container(X_df, X_df, columns=["c", "d"], inplace=True)
    assert X_output is X_df
    assert list(X_df.columns) == ["c", "d"]
    assert list(X_output.columns) == ["c", "d"]


@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test__container_error_validation(csr_container):
    # [FUNCTION DOC]
    # Verifies that _wrap_data_with_container raises ValueError when a transformer
    # returns a sparse matrix while transform_output is 'pandas'.
    # Inputs: csr_container - CSR constructor, parametrized.
    """Check errors in _wrap_data_with_container."""
    X = np.asarray([[1, 0, 3], [0, 0, 1]])
    X_csr = csr_container(X)
    match = "The transformer outputs a scipy sparse matrix."
    with config_context(transform_output="pandas"):
        with pytest.raises(ValueError, match=match):
            _wrap_data_with_container("transform", X_csr, X, StandardScaler())


class EstimatorWithoutSetOutputAndWithoutTransform:
    # [CLASS DOC]
    # HELPER: Dummy object with neither transform nor set_output; _safe_set_output
    # should silently skip it.
    pass


class EstimatorNoSetOutputWithTransform:
    # [CLASS DOC]
    # HELPER: Dummy estimator with transform but no set_output; _safe_set_output should
    # raise for it.
    def transform(self, X, y=None):
        # [FUNCTION DOC]
        # HELPER: Identity transform; returns X unchanged. Never actually called by the
        # tests.
        return X  # pragma: no cover


class EstimatorWithSetOutput(_SetOutputMixin):
    # [CLASS DOC]
    # HELPER: Minimal transformer using _SetOutputMixin with fit, identity transform and
    # get_feature_names_out, so set_output is defined and transform output is auto-
    # wrapped.
    def fit(self, X, y=None):
        # [FUNCTION DOC]
        # HELPER: Records n_features_in_ from the number of columns of X and returns
        # self.
        # Inputs: X - 2D array-like; y - ignored.
        # Side effects: sets self.n_features_in_.
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X, y=None):
        # [FUNCTION DOC]
        # HELPER: Identity transform returning X (wrapped by the mixin according to the
        # output config).
        return X

    def get_feature_names_out(self, input_features=None):
        # [FUNCTION DOC]
        # HELPER: Returns object array ['X0', ..., 'X{n-1}'] using n_features_in_;
        # input_features is ignored.
        # Exceptions: AttributeError if not fitted.
        return np.asarray([f"X{i}" for i in range(self.n_features_in_)], dtype=object)


def test__safe_set_output():
    # [FUNCTION DOC]
    # Verifies _safe_set_output: no-op for objects without transform, ValueError 'Unable
    # to configure output' for objects with transform but no set_output, and for a mixin
    # estimator it sets the dense transform config to 'pandas' then 'default', with
    # transform=None leaving it unchanged.
    """Check _safe_set_output works as expected."""

    # Estimator without transform will not raise when setting set_output for transform.
    est = EstimatorWithoutSetOutputAndWithoutTransform()
    _safe_set_output(est, transform="pandas")

    # Estimator with transform but without set_output will raise
    est = EstimatorNoSetOutputWithTransform()
    with pytest.raises(ValueError, match="Unable to configure output"):
        _safe_set_output(est, transform="pandas")

    est = EstimatorWithSetOutput().fit(np.asarray([[1, 2, 3]]))
    _safe_set_output(est, transform="pandas")
    config = _get_output_config("transform", est)
    assert config["dense"] == "pandas"

    _safe_set_output(est, transform="default")
    config = _get_output_config("transform", est)
    assert config["dense"] == "default"

    # transform is None is a no-op, so the config remains "default"
    _safe_set_output(est, transform=None)
    config = _get_output_config("transform", est)
    assert config["dense"] == "default"


class EstimatorNoSetOutputWithTransformNoFeatureNamesOut(_SetOutputMixin):
    # [CLASS DOC]
    # HELPER: _SetOutputMixin subclass lacking get_feature_names_out, so the mixin must
    # not define set_output.
    def transform(self, X, y=None):
        # [FUNCTION DOC]
        # HELPER: Identity transform; not called.
        return X  # pragma: no cover


def test_set_output_mixin():
    # [FUNCTION DOC]
    # Verifies that a _SetOutputMixin subclass without get_feature_names_out has no
    # set_output attribute.
    """Estimator without get_feature_names_out does not define `set_output`."""
    est = EstimatorNoSetOutputWithTransformNoFeatureNamesOut()
    assert not hasattr(est, "set_output")


def test__safe_set_output_error():
    # [FUNCTION DOC]
    # Verifies that configuring an invalid output ('bad') via _safe_set_output is
    # accepted at set time but transform then raises ValueError 'output config must be
    # in'.
    """Check transform with invalid config."""
    X = np.asarray([[1, 0, 3], [0, 0, 1]])

    est = EstimatorWithSetOutput()
    _safe_set_output(est, transform="bad")

    msg = "output config must be in"
    with pytest.raises(ValueError, match=msg):
        est.transform(X)


@pytest.mark.parametrize("dataframe_lib", ["pandas", "polars"])
def test_set_output_method(dataframe_lib):
    # [FUNCTION DOC]
    # Verifies set_output(transform=None) returns self and leaves ndarray output, while
    # set_output(transform=dataframe_lib) makes transform return that library's
    # DataFrame.
    # Inputs: dataframe_lib - 'pandas' or 'polars'; skipped if not installed.
    """Check that the output is a dataframe."""
    lib = pytest.importorskip(dataframe_lib)

    X = np.asarray([[1, 0, 3], [0, 0, 1]])
    est = EstimatorWithSetOutput().fit(X)

    # transform=None is a no-op
    est2 = est.set_output(transform=None)
    assert est2 is est
    X_trans_np = est2.transform(X)
    assert isinstance(X_trans_np, np.ndarray)

    est.set_output(transform=dataframe_lib)

    X_trans_pd = est.transform(X)

    assert isinstance(X_trans_pd, lib.DataFrame)


def test_set_output_method_error():
    # [FUNCTION DOC]
    # Verifies set_output(transform='bad') leads transform to raise ValueError 'output
    # config must be in'.
    """Check transform fails with invalid transform."""

    X = np.asarray([[1, 0, 3], [0, 0, 1]])
    est = EstimatorWithSetOutput().fit(X)
    est.set_output(transform="bad")

    msg = "output config must be in"
    with pytest.raises(ValueError, match=msg):
        est.transform(X)


@pytest.mark.parametrize("transform_output", ["pandas", "polars"])
def test__get_output_config(transform_output):
    # [FUNCTION DOC]
    # Verifies the precedence of _get_output_config: global config when no estimator or
    # when the estimator has no local config (including estimators without set_output),
    # and the estimator's local set_output config otherwise, both inside and outside a
    # config_context.
    # Inputs: transform_output - 'pandas' or 'polars' (libraries need not be installed).
    """Check _get_output_config works as expected."""

    # Without a configuration set, the global config is used
    global_config = get_config()["transform_output"]
    config = _get_output_config("transform")
    assert config["dense"] == global_config

    with config_context(transform_output=transform_output):
        # with estimator=None, the global config is used
        config = _get_output_config("transform")
        assert config["dense"] == transform_output

        est = EstimatorNoSetOutputWithTransform()
        config = _get_output_config("transform", est)
        assert config["dense"] == transform_output

        est = EstimatorWithSetOutput()
        # If estimator has not config, use global config
        config = _get_output_config("transform", est)
        assert config["dense"] == transform_output

        # If estimator has a config, use local config
        est.set_output(transform="default")
        config = _get_output_config("transform", est)
        assert config["dense"] == "default"

    est.set_output(transform=transform_output)
    config = _get_output_config("transform", est)
    assert config["dense"] == transform_output


class EstimatorWithSetOutputNoAutoWrap(_SetOutputMixin, auto_wrap_output_keys=None):
    # [CLASS DOC]
    # HELPER: _SetOutputMixin subclass with auto_wrap_output_keys=None, which disables
    # wrapping and set_output.
    def transform(self, X, y=None):
        # [FUNCTION DOC]
        # HELPER: Identity transform returning the same X object.
        return X


def test_get_output_auto_wrap_false():
    # [FUNCTION DOC]
    # Verifies that with auto_wrap_output_keys=None the estimator has no set_output and
    # transform returns the very same input object.
    """Check that auto_wrap_output_keys=None does not wrap."""
    est = EstimatorWithSetOutputNoAutoWrap()
    assert not hasattr(est, "set_output")

    X = np.asarray([[1, 0, 3], [0, 0, 1]])
    assert X is est.transform(X)


def test_auto_wrap_output_keys_errors_with_incorrect_input():
    # [FUNCTION DOC]
    # Verifies that defining a _SetOutputMixin subclass with a non-tuple, non-None
    # auto_wrap_output_keys raises ValueError at class creation.
    msg = "auto_wrap_output_keys must be None or a tuple of keys."
    with pytest.raises(ValueError, match=msg):

        class BadEstimator(_SetOutputMixin, auto_wrap_output_keys="bad_parameter"):
            # [CLASS DOC]
            # HELPER: Class whose definition is expected to raise ValueError in
            # __init_subclass__.
            pass


class AnotherMixin:
    # [CLASS DOC]
    # HELPER: Mixin whose __init_subclass__ requires a custom_parameter keyword, used to
    # test cooperative __init_subclass__ chaining with _SetOutputMixin.
    def __init_subclass__(cls, custom_parameter, **kwargs):
        # [FUNCTION DOC]
        # HELPER: Forwards remaining kwargs up the MRO and stores custom_parameter as a
        # class attribute.
        # Inputs: custom_parameter - any value; kwargs - other class keywords.
        # Side effects: sets cls.custom_parameter.
        super().__init_subclass__(**kwargs)
        cls.custom_parameter = custom_parameter


def test_set_output_mixin_custom_mixin():
    # [FUNCTION DOC]
    # Verifies that _SetOutputMixin forwards unknown class keywords up the MRO: a class
    # combining it with AnotherMixin gets custom_parameter 123 and still has set_output.
    """Check that multiple init_subclasses passes parameters up."""

    class BothMixinEstimator(_SetOutputMixin, AnotherMixin, custom_parameter=123):
        # [CLASS DOC]
        # HELPER: Estimator combining both mixins with custom_parameter=123.
        def transform(self, X, y=None):
            # [FUNCTION DOC]
            # HELPER: Identity transform.
            return X

        def get_feature_names_out(self, input_features=None):
            # [FUNCTION DOC]
            # HELPER: Returns input_features unchanged.
            return input_features

    est = BothMixinEstimator()
    assert est.custom_parameter == 123
    assert hasattr(est, "set_output")


def test_set_output_mro():
    # [FUNCTION DOC]
    # Non-regression test gh-25293: in a diamond hierarchy C(A, B) where only B
    # overrides transform, verifies that the mixin wrapping does not shadow the MRO, so
    # C().transform returns 'B'.
    """Check that multi-inheritance resolves to the correct class method.

    Non-regression test gh-25293.
    """

    class Base(_SetOutputMixin):
        # [CLASS DOC]
        # HELPER: Base mixin class whose transform returns 'Base'.
        def transform(self, X):
            # [FUNCTION DOC]
            # HELPER: Returns the string 'Base'.
            return "Base"

    class A(Base):
        # [CLASS DOC]
        # HELPER: Subclass of Base without overrides.
        pass

    class B(Base):
        # [CLASS DOC]
        # HELPER: Subclass of Base overriding transform.
        def transform(self, X):
            # [FUNCTION DOC]
            # HELPER: Returns the string 'B'.
            return "B"

    class C(A, B):
        # [CLASS DOC]
        # HELPER: Diamond subclass of A and B.
        pass

    assert C().transform(None) == "B"


class EstimatorWithSetOutputIndex(_SetOutputMixin):
    # [CLASS DOC]
    # HELPER: Mixin transformer whose transform returns a pandas DataFrame with a new
    # string index, used to check that wrapping keeps the transformer's index.
    def fit(self, X, y=None):
        # [FUNCTION DOC]
        # HELPER: Records n_features_in_ and returns self.
        # Side effects: sets self.n_features_in_.
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X, y=None):
        # [FUNCTION DOC]
        # HELPER: Returns a DataFrame of X's values with index 's0', 's1', ...
        # Inputs: X - pandas DataFrame (uses to_numpy).
        # Exceptions: ImportError without pandas.
        import pandas as pd

        # transform by giving output a new index.
        return pd.DataFrame(X.to_numpy(), index=[f"s{i}" for i in range(X.shape[0])])

    def get_feature_names_out(self, input_features=None):
        # [FUNCTION DOC]
        # HELPER: Returns ['X0', ..., 'X{n-1}'] as an object array.
        return np.asarray([f"X{i}" for i in range(self.n_features_in_)], dtype=object)


def test_set_output_pandas_keep_index():
    # [FUNCTION DOC]
    # Non-regression test gh-25730: verifies that with pandas output the index produced
    # by transform ('s0', 's1') is not overwritten by the input index.
    """Check that set_output does not override index.

    Non-regression test for gh-25730.
    """
    pd = pytest.importorskip("pandas")

    X = pd.DataFrame([[1, 2, 3], [4, 5, 6]], index=[0, 1])
    est = EstimatorWithSetOutputIndex().set_output(transform="pandas")
    est.fit(X)

    X_trans = est.transform(X)
    assert_array_equal(X_trans.index, ["s0", "s1"])


class EstimatorReturnTuple(_SetOutputMixin):
    # [CLASS DOC]
    # HELPER: Mixin estimator whose transform returns a namedtuple, to check tuples are
    # not wrapped.
    def __init__(self, OutputTuple):
        # [FUNCTION DOC]
        # HELPER: Stores the namedtuple class.
        # Inputs: OutputTuple - namedtuple type with two fields.
        self.OutputTuple = OutputTuple

    def transform(self, X, y=None):
        # [FUNCTION DOC]
        # HELPER: Returns OutputTuple(X, 2 * X).
        return self.OutputTuple(X, 2 * X)


def test_set_output_named_tuple_out():
    # [FUNCTION DOC]
    # Verifies that a namedtuple returned by transform is kept as-is under the default
    # output config, with fields X and 2X.
    """Check that namedtuples are kept by default."""
    Output = namedtuple("Output", "X, Y")
    X = np.asarray([[1, 2, 3]])
    est = EstimatorReturnTuple(OutputTuple=Output)
    X_trans = est.transform(X)

    assert isinstance(X_trans, Output)
    assert_array_equal(X_trans.X, X)
    assert_array_equal(X_trans.Y, 2 * X)


class EstimatorWithListInput(_SetOutputMixin):
    # [CLASS DOC]
    # HELPER: Mixin transformer that asserts fit receives a list of lists, for checking
    # set_output with list input.
    def fit(self, X, y=None):
        # [FUNCTION DOC]
        # HELPER: Asserts X is a list, sets n_features_in_ to the length of the first
        # row, returns self.
        # Exceptions: AssertionError if X is not a list.
        assert isinstance(X, list)
        self.n_features_in_ = len(X[0])
        return self

    def transform(self, X, y=None):
        # [FUNCTION DOC]
        # HELPER: Identity transform (list returned, wrapped by mixin).
        return X

    def get_feature_names_out(self, input_features=None):
        # [FUNCTION DOC]
        # HELPER: Returns ['X0', ..., 'X{n-1}'] as an object array.
        return np.asarray([f"X{i}" for i in range(self.n_features_in_)], dtype=object)


@pytest.mark.parametrize("dataframe_lib", ["pandas", "polars"])
def test_set_output_list_input(dataframe_lib):
    # [FUNCTION DOC]
    # Non-regression test #27037: verifies that a transformer receiving list-of-lists
    # input returns a DataFrame with columns X0..X3 when set_output selects pandas or
    # polars.
    # Inputs: dataframe_lib - parametrized; skipped if missing.
    """Check set_output for list input.

    Non-regression test for #27037.
    """
    lib = pytest.importorskip(dataframe_lib)

    X = [[0, 1, 2, 3], [4, 5, 6, 7]]
    est = EstimatorWithListInput()
    est.set_output(transform=dataframe_lib)

    X_out = est.fit(X).transform(X)
    assert isinstance(X_out, lib.DataFrame)
    assert_array_equal(X_out.columns, ["X0", "X1", "X2", "X3"])


@pytest.mark.parametrize("name", sorted(ADAPTERS_MANAGER.adapters))
def test_adapter_class_has_interface(name):
    # [FUNCTION DOC]
    # Verifies each registered adapter is an instance of the runtime-checkable
    # ContainerAdapterProtocol.
    # Inputs: name - adapter key.
    """Check adapters have the correct interface."""
    assert isinstance(ADAPTERS_MANAGER.adapters[name], ContainerAdapterProtocol)


def test_check_library_installed(monkeypatch):
    # [FUNCTION DOC]
    # Verifies check_library_installed re-raises a failed pandas import as an
    # ImportError with an explanatory message, by monkeypatching importlib.import_module
    # to fail for 'pandas'.
    # Inputs: monkeypatch - pytest fixture.
    """Check import error changed."""
    orig_import_module = importlib.import_module

    def patched_import_module(name):
        # [FUNCTION DOC]
        # HELPER: Replacement import_module raising ImportError for 'pandas' and
        # delegating otherwise.
        # KNOWN ISSUE: the delegated call's result is not returned, so it returns None
        # for other modules (harmless here since only 'pandas' is requested).
        if name == "pandas":
            raise ImportError()
        orig_import_module(name, package=None)

    monkeypatch.setattr(importlib, "import_module", patched_import_module)

    msg = "Setting output container to 'pandas' requires"
    with pytest.raises(ImportError, match=msg):
        check_library_installed("pandas")


def test_get_adapter_from_container():
    # [FUNCTION DOC]
    # Verifies _get_adapter_from_container returns the pandas adapter for a DataFrame
    # and raises ValueError for an ndarray, which has no registered adapter.
    """Check the behavior fo `_get_adapter_from_container`."""
    pd = pytest.importorskip("pandas")
    X = pd.DataFrame({"a": [1, 2, 3], "b": [10, 20, 100]})
    adapter = _get_adapter_from_container(X)
    assert adapter.container_lib == "pandas"
    err_msg = "The container does not have a registered adapter in scikit-learn."
    with pytest.raises(ValueError, match=err_msg):
        _get_adapter_from_container(X.to_numpy())
