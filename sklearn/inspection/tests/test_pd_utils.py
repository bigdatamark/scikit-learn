# [MODULE DOC]
# Tests for the private partial-dependence helpers in sklearn.inspection._pd_utils:
# _check_feature_names (validation/derivation of feature names) and _get_feature_index
# (mapping a feature name or integer to a column index).
import numpy as np
import pytest

from sklearn.inspection._pd_utils import _check_feature_names, _get_feature_index
from sklearn.utils._testing import _convert_container


@pytest.mark.parametrize(
    "feature_names, array_type, expected_feature_names",
    [
        (None, "array", ["x0", "x1", "x2"]),
        (None, "pandas", ["a", "b", "c"]),
        (np.array(["a", "b", "c"]), "array", ["a", "b", "c"]),
    ],
)
def test_check_feature_names(feature_names, array_type, expected_feature_names):
    # [FUNCTION DOC]
    # Verifies _check_feature_names returns the expected list of names: generated
    # "x0".."x2"
    # for a plain array with no names, the DataFrame column names for pandas input, and
    # the
    # given names (converted to a list) when an explicit array of names is passed.
    # Inputs: parametrized feature_names, array_type ("array" or "pandas"), expected
    # list.
    # Uses unseeded np.random data (values are irrelevant to the check).
    X = np.random.randn(10, 3)
    column_names = ["a", "b", "c"]
    X = _convert_container(X, constructor_name=array_type, column_names=column_names)
    feature_names_validated = _check_feature_names(X, feature_names)
    assert feature_names_validated == expected_feature_names


def test_check_feature_names_error():
    # [FUNCTION DOC]
    # Verifies _check_feature_names raises ValueError with a "should not contain
    # duplicates" message when the provided feature_names list has a repeated entry.
    X = np.random.randn(10, 3)
    feature_names = ["a", "b", "c", "a"]
    msg = "feature_names should not contain duplicates."
    with pytest.raises(ValueError, match=msg):
        _check_feature_names(X, feature_names)


@pytest.mark.parametrize("fx, idx", [(0, 0), (1, 1), ("a", 0), ("b", 1), ("c", 2)])
def test_get_feature_index(fx, idx):
    # [FUNCTION DOC]
    # Verifies _get_feature_index maps an integer to itself and a string name to its
    # position in feature_names.
    feature_names = ["a", "b", "c"]
    assert _get_feature_index(fx, feature_names) == idx


@pytest.mark.parametrize(
    "fx, feature_names, err_msg",
    [
        ("a", None, "Cannot plot partial dependence for feature 'a'"),
        ("d", ["a", "b", "c"], "Feature 'd' not in feature_names"),
    ],
)
def test_get_feature_names_error(fx, feature_names, err_msg):
    # [FUNCTION DOC]
    # Verifies _get_feature_index raises ValueError when a string feature is given
    # without
    # feature_names, or when the name is absent from feature_names.
    with pytest.raises(ValueError, match=err_msg):
        _get_feature_index(fx, feature_names)
