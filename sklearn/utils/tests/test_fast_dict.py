# [MODULE DOC]
# Tests for the Cython IntFloatDict mapping (int keys to float values) and its argmin
# helper from sklearn.utils._fast_dict.
"""Test fast_dict."""

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from sklearn.utils._fast_dict import IntFloatDict, argmin


def test_int_float_dict():
    # [FUNCTION DOC]
    # Verifies IntFloatDict construction from unique random intp keys and float values:
    # every key maps to its value, len matches, append adds a new key (len grows by one),
    # and appending 2000 more keys still allows correct lookup (internal growth works).
    rng = np.random.RandomState(0)
    keys = np.unique(rng.randint(100, size=10).astype(np.intp))
    values = rng.rand(len(keys))

    d = IntFloatDict(keys, values)
    for key, value in zip(keys, values):
        assert d[key] == value
    assert len(d) == len(keys)

    d.append(120, 3.0)
    assert d[120] == 3.0
    assert len(d) == len(keys) + 1
    for i in range(2000):
        d.append(i + 1000, 4.0)
    assert d[1100] == 4.0


def test_int_float_dict_argmin():
    # [FUNCTION DOC]
    # Verifies argmin on a dict with keys 0..99 mapped to values 0..99 returns the
    # (key, value) pair of the minimum value, i.e. (0, 0).
    # Test the argmin implementation on the IntFloatDict
    keys = np.arange(100, dtype=np.intp)
    values = np.arange(100, dtype=np.float64)
    d = IntFloatDict(keys, values)
    assert argmin(d) == (0, 0)


def test_to_arrays():
    # [FUNCTION DOC]
    # Verifies to_arrays returns keys and values arrays with the input dtypes (intp and
    # float64) and equal contents to the construction inputs.
    # Test that an IntFloatDict is converted into arrays
    # of keys and values correctly
    keys_in = np.array([1, 2, 3], dtype=np.intp)
    values_in = np.array([4, 5, 6], dtype=np.float64)

    d = IntFloatDict(keys_in, values_in)
    keys_out, values_out = d.to_arrays()

    assert keys_out.dtype == keys_in.dtype
    assert values_in.dtype == values_out.dtype
    assert_array_equal(keys_out, keys_in)
    assert_allclose(values_out, values_in)
