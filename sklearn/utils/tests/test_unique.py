# [MODULE DOC]
# Tests for sklearn.utils._unique: attach_unique stores the array's unique values in
# dtype metadata (on a view), and cached_unique reads them back; also checks check_array
# preserves that metadata.
import numpy as np
from numpy.testing import assert_array_equal

from sklearn.utils._unique import attach_unique, cached_unique
from sklearn.utils.validation import check_array


def test_attach_unique_attaches_unique_to_array():
    # [FUNCTION DOC]
    # Verifies attach_unique stores the sorted unique values under
    # dtype.metadata['unique'] while array values are unchanged.
    arr = np.array([1, 2, 2, 3, 4, 4, 5])
    arr_ = attach_unique(arr)
    assert_array_equal(arr_.dtype.metadata["unique"], np.array([1, 2, 3, 4, 5]))
    assert_array_equal(arr_, arr)


def test_cached_unique_returns_cached_unique():
    # [FUNCTION DOC]
    # Verifies cached_unique returns the metadata-stored unique values (deliberately
    # inconsistent [1, 2]) rather than recomputing them.
    my_dtype = np.dtype(np.float64, metadata={"unique": np.array([1, 2])})
    arr = np.array([1, 2, 2, 3, 4, 4, 5], dtype=my_dtype)
    assert_array_equal(cached_unique(arr), np.array([1, 2]))


def test_attach_unique_not_ndarray():
    # [FUNCTION DOC]
    # Verifies non-ndarray inputs (a list) are returned as the same object, untouched.
    """Test that when not np.ndarray, we don't touch the array."""
    arr = [1, 2, 2, 3, 4, 4, 5]
    arr_ = attach_unique(arr)
    assert arr_ is arr


def test_attach_unique_returns_view():
    # [FUNCTION DOC]
    # Verifies the returned array is a view whose base is the input array.
    """Test that attach_unique returns a view of the array."""
    arr = np.array([1, 2, 2, 3, 4, 4, 5])
    arr_ = attach_unique(arr)
    assert arr_.base is arr


def test_attach_unique_return_tuple():
    # [FUNCTION DOC]
    # Verifies return_tuple=True wraps the result in a 1-tuple and return_tuple=False
    # returns a bare ndarray, both equal to the input.
    """Test return_tuple argument of the function."""
    arr = np.array([1, 2, 2, 3, 4, 4, 5])
    arr_tuple = attach_unique(arr, return_tuple=True)
    assert isinstance(arr_tuple, tuple)
    assert len(arr_tuple) == 1
    assert_array_equal(arr_tuple[0], arr)

    arr_single = attach_unique(arr, return_tuple=False)
    assert isinstance(arr_single, np.ndarray)
    assert_array_equal(arr_single, arr)


def test_check_array_keeps_unique():
    # [FUNCTION DOC]
    # Verifies check_array on a 2D array with attached unique metadata preserves the
    # metadata and values.
    """Test that check_array keeps the unique metadata."""
    arr = np.array([[1, 2, 2, 3, 4, 4, 5]])
    arr_ = attach_unique(arr)
    arr_ = check_array(arr_)
    assert_array_equal(arr_.dtype.metadata["unique"], np.array([1, 2, 3, 4, 5]))
    assert_array_equal(arr_, arr)
