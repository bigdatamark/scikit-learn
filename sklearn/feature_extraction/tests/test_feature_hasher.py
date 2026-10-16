# [MODULE DOC]
# Tests for FeatureHasher and the underlying Cython hashing transform: input
# types (dict, pair, string), seeds, signs, collisions, zero handling and the
# stateless/no-fit behaviour.
import numpy as np
import pytest
from numpy.testing import assert_array_equal

from sklearn.feature_extraction import FeatureHasher
from sklearn.feature_extraction._hashing_fast import transform as _hashing_transform
from sklearn.utils.fixes import SCIPY_VERSION_BELOW_1_12


def test_feature_hasher_dicts():
    # [FUNCTION DOC]
    # Checks the default input_type is "dict" and that hashing dicts gives the same
    # matrix as hashing the same items via input_type="pair" from a generator.
    feature_hasher = FeatureHasher(n_features=16)
    assert "dict" == feature_hasher.input_type

    raw_X = [{"foo": "bar", "dada": 42, "tzara": 37}, {"foo": "baz", "gaga": "string1"}]
    X1 = FeatureHasher(n_features=16).transform(raw_X)
    gen = (iter(d.items()) for d in raw_X)
    X2 = FeatureHasher(n_features=16, input_type="pair").transform(gen)
    assert_array_equal(X1.toarray(), X2.toarray())


def test_feature_hasher_strings():
    # [FUNCTION DOC]
    # For several power-of-two n_features, hashes mixed byte/unicode string samples
    # from a generator without sign alternation and checks the shape, per-row sums
    # (4 and 3, counting the duplicate "foo") using the row indexing style supported
    # by the installed scipy, and total nnz 6 (byte and str "foo" collide).
    # mix byte and Unicode strings; note that "foo" is a duplicate in row 0
    raw_X = [
        ["foo", "bar", "baz", "foo".encode("ascii")],
        ["bar".encode("ascii"), "baz", "quux"],
    ]

    for lg_n_features in (7, 9, 11, 16, 22):
        n_features = 2**lg_n_features

        it = (x for x in raw_X)  # iterable

        feature_hasher = FeatureHasher(
            n_features=n_features, input_type="string", alternate_sign=False
        )
        X = feature_hasher.transform(it)

        assert X.shape[0] == len(raw_X)
        assert X.shape[1] == n_features

        if SCIPY_VERSION_BELOW_1_12:
            assert X[[0], :].sum() == 4
            assert X[[1], :].sum() == 3
        else:
            assert X[0].sum() == 4
            assert X[1].sum() == 3

        assert X.nnz == 6


@pytest.mark.parametrize("input_type", ["list", "generator"])
def test_feature_hasher_single_string(input_type):
    # [FUNCTION DOC]
    # Non-regression (gh-13199): for a list and a generator of plain strings,
    # input_type="string" transform raises ValueError about single-string samples.
    """FeatureHasher raises error when a sample is a single string.

    Non-regression test for gh-13199.
    """
    msg = "Samples can not be a single string"
    raw_X = ["my_string", "another_string"]
    if input_type == "generator":
        raw_X = (x for x in raw_X)

    feature_hasher = FeatureHasher(n_features=10, input_type="string")
    with pytest.raises(ValueError, match=msg):
        feature_hasher.transform(raw_X)


def test_hashing_transform_seed():
    # [FUNCTION DOC]
    # Calls the Cython transform directly: default seed equals seed=0 (same indices
    # and indptr), while seed=1 yields different indices (asserted via an expected
    # AssertionError from assert_array_equal).
    # check the influence of the seed when computing the hashes
    raw_X = [
        ["foo", "bar", "baz", "foo".encode("ascii")],
        ["bar".encode("ascii"), "baz", "quux"],
    ]

    raw_X_ = (((f, 1) for f in x) for x in raw_X)
    indices, indptr, _ = _hashing_transform(raw_X_, 2**7, str, False)

    raw_X_ = (((f, 1) for f in x) for x in raw_X)
    indices_0, indptr_0, _ = _hashing_transform(raw_X_, 2**7, str, False, seed=0)
    assert_array_equal(indices, indices_0)
    assert_array_equal(indptr, indptr_0)

    raw_X_ = (((f, 1) for f in x) for x in raw_X)
    indices_1, _, _ = _hashing_transform(raw_X_, 2**7, str, False, seed=1)
    with pytest.raises(AssertionError):
        assert_array_equal(indices, indices_1)


def test_feature_hasher_pairs():
    # [FUNCTION DOC]
    # Hashes (name, value) pairs and checks the absolute nonzero values per row equal
    # the input magnitudes, i.e. no collisions at n_features=16 for these names.
    raw_X = (
        iter(d.items())
        for d in [{"foo": 1, "bar": 2}, {"baz": 3, "quux": 4, "foo": -1}]
    )
    feature_hasher = FeatureHasher(n_features=16, input_type="pair")
    x1, x2 = feature_hasher.transform(raw_X).toarray()
    x1_nz = sorted(np.abs(x1[x1 != 0]))
    x2_nz = sorted(np.abs(x2[x2 != 0]))
    assert [1, 2] == x1_nz
    assert [1, 3, 4] == x2_nz


def test_feature_hasher_pairs_with_string_values():
    # [FUNCTION DOC]
    # Checks pair input with string values hashes "name=value" with value 1 (sign
    # ignored), and that identical string-valued samples give identical rows.
    raw_X = (
        iter(d.items())
        for d in [{"foo": 1, "bar": "a"}, {"baz": "abc", "quux": 4, "foo": -1}]
    )
    feature_hasher = FeatureHasher(n_features=16, input_type="pair")
    x1, x2 = feature_hasher.transform(raw_X).toarray()
    x1_nz = sorted(np.abs(x1[x1 != 0]))
    x2_nz = sorted(np.abs(x2[x2 != 0]))
    assert [1, 1] == x1_nz
    assert [1, 1, 4] == x2_nz

    raw_X = (iter(d.items()) for d in [{"bax": "abc"}, {"bax": "abc"}])
    x1, x2 = feature_hasher.transform(raw_X).toarray()
    x1_nz = np.abs(x1[x1 != 0])
    x2_nz = np.abs(x2[x2 != 0])
    assert [1] == x1_nz
    assert [1] == x2_nz
    assert_array_equal(x1, x2)


def test_hash_empty_input():
    # [FUNCTION DOC]
    # Checks empty samples (list, tuple, empty iterator) with input_type="string"
    # give all-zero rows of the right shape.
    n_features = 16
    raw_X = [[], (), iter(range(0))]

    feature_hasher = FeatureHasher(n_features=n_features, input_type="string")
    X = feature_hasher.transform(raw_X)

    assert_array_equal(X.toarray(), np.zeros((len(raw_X), n_features)))


def test_hasher_zeros():
    # [FUNCTION DOC]
    # Checks a zero-valued feature produces no stored entry in the CSR output.
    # Assert that no zeros are materialized in the output.
    X = FeatureHasher().transform([{"foo": 0}])
    assert X.data.shape == (0,)


def test_hasher_alternate_sign():
    # [FUNCTION DOC]
    # Checks alternate_sign=True produces both negative and positive values and
    # alternate_sign=False only positive values for a list of characters.
    X = [list("Thequickbrownfoxjumped")]

    Xt = FeatureHasher(alternate_sign=True, input_type="string").fit_transform(X)
    assert Xt.data.min() < 0 and Xt.data.max() > 0

    Xt = FeatureHasher(alternate_sign=False, input_type="string").fit_transform(X)
    assert Xt.data.min() > 0


def test_hash_collisions():
    # [FUNCTION DOC]
    # With n_features=1 all tokens collide: with alternating signs the absolute sum
    # is smaller than the token count (cancellation); without, it equals the count.
    X = [list("Thequickbrownfoxjumped")]

    Xt = FeatureHasher(
        alternate_sign=True, n_features=1, input_type="string"
    ).fit_transform(X)
    # check that some of the hashed tokens are added
    # with an opposite sign and cancel out
    assert abs(Xt.data[0]) < len(X[0])

    Xt = FeatureHasher(
        alternate_sign=False, n_features=1, input_type="string"
    ).fit_transform(X)
    assert Xt.data[0] == len(X[0])


def test_feature_hasher_requires_fit_tag():
    # [FUNCTION DOC]
    # Checks __sklearn_tags__ reports requires_fit as False.
    """Test that FeatureHasher has requires_fit=False tag."""
    hasher = FeatureHasher()
    tags = hasher.__sklearn_tags__()
    assert not tags.requires_fit


def test_feature_hasher_transform_without_fit():
    # [FUNCTION DOC]
    # Checks transform works on an unfitted FeatureHasher and returns shape (2, 10).
    """Test that FeatureHasher can transform without fitting."""
    hasher = FeatureHasher(n_features=10)
    data = [{"dog": 1, "cat": 2}, {"dog": 2, "run": 5}]
    result = hasher.transform(data)
    assert result.shape == (2, 10)
