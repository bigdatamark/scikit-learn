# [MODULE DOC]
# Tests for fetch_california_housing, run through the conftest fixture
# fetch_california_housing_fxt that skips unless the data is available or network tests
# are enabled.
"""Test the california_housing loader, if the data is available,
or if specifically requested via environment variable
(e.g. for CI jobs)."""

from functools import partial

import pytest

from sklearn.datasets.tests.test_common import check_return_X_y


def test_fetch(fetch_california_housing_fxt):
    # [FUNCTION DOC]
    # Verifies data shape (20640, 8), target shape (20640,), DESCR header, and the
    # return_X_y option via check_return_X_y.
    data = fetch_california_housing_fxt()
    assert (20640, 8) == data.data.shape
    assert (20640,) == data.target.shape
    assert data.DESCR.startswith(".. _california_housing_dataset:")

    # test return_X_y option
    fetch_func = partial(fetch_california_housing_fxt)
    check_return_X_y(data, fetch_func)


def test_fetch_asframe(fetch_california_housing_fxt):
    # [FUNCTION DOC]
    # Verifies as_frame=True gives a frame of shape (20640, 9), a DataFrame data and a
    # Series target. Skipped without pandas.
    pd = pytest.importorskip("pandas")
    bunch = fetch_california_housing_fxt(as_frame=True)
    frame = bunch.frame
    assert hasattr(bunch, "frame") is True
    assert frame.shape == (20640, 9)
    assert isinstance(bunch.data, pd.DataFrame)
    assert isinstance(bunch.target, pd.Series)


def test_pandas_dependency_message(fetch_california_housing_fxt, hide_available_pandas):
    # [FUNCTION DOC]
    # Verifies that with pandas hidden (hide_available_pandas fixture), as_frame=True
    # raises ImportError with the informative message.
    # Check that pandas is imported lazily and that an informative error
    # message is raised when pandas is missing:
    expected_msg = "fetch_california_housing with as_frame=True requires pandas"
    with pytest.raises(ImportError, match=expected_msg):
        fetch_california_housing_fxt(as_frame=True)
