# [MODULE DOC]
# Tests for sklearn.datasets.fetch_covtype. All tests receive the loader through the
# fetch_covtype_fxt fixture (from the datasets conftest), which skips unless the data is
# cached locally or downloads are explicitly enabled. Checks shapes, shuffling, DESCR,
# the return_X_y and as_frame options, and the pandas-missing error message.
"""Test the covtype loader, if the data is available,
or if specifically requested via environment variable
(e.g. for CI jobs)."""

from functools import partial

import pytest

from sklearn.datasets.tests.test_common import check_return_X_y


def test_fetch(fetch_covtype_fxt, global_random_seed):
    # [FUNCTION DOC]
    # Verifies that two shuffled fetches with different seeds (global_random_seed and
    # +1) have the expected shape (581012, 54), matching shapes and equal total sums
    # (same rows, different order), 1-D targets of the right length, a DESCR starting
    # with the covtype RST label, and that return_X_y returns the same data (via
    # check_return_X_y).
    data1 = fetch_covtype_fxt(shuffle=True, random_state=global_random_seed)
    data2 = fetch_covtype_fxt(shuffle=True, random_state=global_random_seed + 1)

    X1, X2 = data1["data"], data2["data"]
    assert (581012, 54) == X1.shape
    assert X1.shape == X2.shape

    assert X1.sum() == X2.sum()

    y1, y2 = data1["target"], data2["target"]
    assert (X1.shape[0],) == y1.shape
    assert (X1.shape[0],) == y2.shape

    descr_prefix = ".. _covtype_dataset:"
    assert data1.DESCR.startswith(descr_prefix)
    assert data2.DESCR.startswith(descr_prefix)

    # test return_X_y option
    fetch_func = partial(fetch_covtype_fxt)
    check_return_X_y(data1, fetch_func)


def test_fetch_asframe(fetch_covtype_fxt):
    # [FUNCTION DOC]
    # Verifies that as_frame=True (requires pandas, skipped otherwise) yields a frame of
    # shape (581012, 55), data (581012, 54), target (581012,), and that the one-hot
    # column groups are named Wilderness_Area_0..3 and Soil_Type_0..39.
    pytest.importorskip("pandas")

    bunch = fetch_covtype_fxt(as_frame=True)
    assert hasattr(bunch, "frame")
    frame = bunch.frame
    assert frame.shape == (581012, 55)
    assert bunch.data.shape == (581012, 54)
    assert bunch.target.shape == (581012,)

    column_names = set(frame.columns)

    # enumerated names are added correctly
    assert set(f"Wilderness_Area_{i}" for i in range(4)) < column_names
    assert set(f"Soil_Type_{i}" for i in range(40)) < column_names


def test_pandas_dependency_message(fetch_covtype_fxt, hide_available_pandas):
    # [FUNCTION DOC]
    # Verifies that with pandas hidden (hide_available_pandas fixture) as_frame=True
    # raises ImportError mentioning that fetch_covtype with as_frame=True requires
    # pandas.
    expected_msg = "fetch_covtype with as_frame=True requires pandas"
    with pytest.raises(ImportError, match=expected_msg):
        fetch_covtype_fxt(as_frame=True)
