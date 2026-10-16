# [MODULE DOC]
# Tests for the dataframe detection helpers in sklearn.utils._dataframe
# (is_df_or_series, is_pandas_df, is_polars_df) across pyarrow, pandas and
# polars containers and duck-typed look-alikes.
"""Tests for dataframe detection functions."""

import numpy as np
import pytest

from sklearn._min_dependencies import dependent_packages
from sklearn.utils._dataframe import is_df_or_series, is_pandas_df, is_polars_df
from sklearn.utils._testing import _convert_container


@pytest.mark.parametrize("constructor_name", ["pyarrow", "pandas", "polars"])
def test_is_df_or_series(constructor_name):
    # [FUNCTION DOC]
    # Verifies is_df_or_series is True for pyarrow/pandas/polars tables and False
    # for a NumPy array.
    df = _convert_container([[1, 4, 2], [3, 3, 6]], constructor_name)

    assert is_df_or_series(df)
    assert not is_df_or_series(np.asarray([1, 2, 3]))


@pytest.mark.parametrize("constructor_name", ["pyarrow", "pandas", "polars"])
def test_is_pandas_df_other_libraries(constructor_name):
    # [FUNCTION DOC]
    # Verifies is_pandas_df is True only for the pandas container.
    df = _convert_container([[1, 4, 2], [3, 3, 6]], constructor_name)
    if constructor_name in ("pyarrow", "polars"):
        assert not is_pandas_df(df)
    else:
        assert is_pandas_df(df)


def test_is_pandas_df():
    # [FUNCTION DOC]
    # Verifies is_pandas_df is True for a pandas DataFrame and False for arrays and
    # scalars.
    """Check behavior of is_pandas_df when pandas is installed."""
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame([[1, 2, 3]])
    assert is_pandas_df(df)
    assert not is_pandas_df(np.asarray([1, 2, 3]))
    assert not is_pandas_df(1)


def test_is_pandas_df_pandas_not_installed(hide_available_pandas):
    # [FUNCTION DOC]
    # Verifies is_pandas_df returns False for arrays and scalars when pandas is
    # hidden by the hide_available_pandas fixture.
    """Check is_pandas_df when pandas is not installed."""

    assert not is_pandas_df(np.asarray([1, 2, 3]))
    assert not is_pandas_df(1)


@pytest.mark.parametrize(
    "constructor_name, minversion",
    [
        ("pyarrow", dependent_packages["pyarrow"][0]),
        ("pandas", dependent_packages["pandas"][0]),
        ("polars", dependent_packages["polars"][0]),
    ],
)
def test_is_polars_df_other_libraries(constructor_name, minversion):
    # [FUNCTION DOC]
    # Verifies is_polars_df is True only for the polars container (each built with
    # its minimum supported version).
    df = _convert_container(
        [[1, 4, 2], [3, 3, 6]],
        constructor_name,
        minversion=minversion,
    )
    if constructor_name in ("pyarrow", "pandas"):
        assert not is_polars_df(df)
    else:
        assert is_polars_df(df)


def test_is_polars_df_for_duck_typed_polars_dataframe():
    # [FUNCTION DOC]
    # Verifies an object with columns and schema attributes is not detected as a
    # polars DataFrame.
    """Check is_polars_df for object that looks like a polars dataframe"""

    class NotAPolarsDataFrame:
        # [CLASS DOC]
        # HELPER: Duck-typed object with columns and a string schema attribute.
        def __init__(self):
            # [FUNCTION DOC]
            # HELPER: Sets columns and schema attributes.
            self.columns = [1, 2, 3]
            self.schema = "my_schema"

    not_a_polars_df = NotAPolarsDataFrame()
    assert not is_polars_df(not_a_polars_df)


def test_is_polars_df():
    # [FUNCTION DOC]
    # Verifies an object with list columns and schema is not detected as polars.
    """Check that is_polars_df return False for non-dataframe objects."""

    class LooksLikePolars:
        # [CLASS DOC]
        # HELPER: Duck-typed object with list-valued columns and schema attributes.
        def __init__(self):
            # [FUNCTION DOC]
            # HELPER: Sets columns and schema attributes.
            self.columns = ["a", "b"]
            self.schema = ["a", "b"]

    assert not is_polars_df(LooksLikePolars())
