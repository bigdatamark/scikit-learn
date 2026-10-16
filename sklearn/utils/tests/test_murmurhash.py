# [MODULE DOC]
# Tests for sklearn.utils.murmurhash.murmurhash3_32 on ints, int arrays, bytes and str,
# plus basic collision and uniformity checks.
# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal

from sklearn.utils.murmurhash import murmurhash3_32


def test_mmhash3_int():
    # [FUNCTION DOC]
    # Checks hard-coded hash values for integer key 3 with seeds 0 (default) and 42, for
    # positive unset, False and True; positive=True yields the unsigned 32-bit value.
    assert murmurhash3_32(3) == 847579505
    assert murmurhash3_32(3, seed=0) == 847579505
    assert murmurhash3_32(3, seed=42) == -1823081949

    assert murmurhash3_32(3, positive=False) == 847579505
    assert murmurhash3_32(3, seed=0, positive=False) == 847579505
    assert murmurhash3_32(3, seed=42, positive=False) == -1823081949

    assert murmurhash3_32(3, positive=True) == 847579505
    assert murmurhash3_32(3, seed=0, positive=True) == 847579505
    assert murmurhash3_32(3, seed=42, positive=True) == 2471885347


def test_mmhash3_int_array():
    # [FUNCTION DOC]
    # Verifies that hashing a 3D int32 array element-wise equals hashing each element
    # individually (reshaped to the input shape), for seeds 0 and 42, signed and
    # unsigned.
    rng = np.random.RandomState(42)
    keys = rng.randint(-5342534, 345345, size=3 * 2 * 1).astype(np.int32)
    keys = keys.reshape((3, 2, 1))

    for seed in [0, 42]:
        expected = np.array([murmurhash3_32(int(k), seed) for k in keys.flat])
        expected = expected.reshape(keys.shape)
        assert_array_equal(murmurhash3_32(keys, seed), expected)

    for seed in [0, 42]:
        expected = np.array([murmurhash3_32(k, seed, positive=True) for k in keys.flat])
        expected = expected.reshape(keys.shape)
        assert_array_equal(murmurhash3_32(keys, seed, positive=True), expected)


def test_mmhash3_bytes():
    # [FUNCTION DOC]
    # Checks hard-coded hash values for the bytes key "foo" with seeds 0 and 42, signed
    # and unsigned.
    assert murmurhash3_32(b"foo", 0) == -156908512
    assert murmurhash3_32(b"foo", 42) == -1322301282

    assert murmurhash3_32(b"foo", 0, positive=True) == 4138058784
    assert murmurhash3_32(b"foo", 42, positive=True) == 2972666014


def test_mmhash3_unicode():
    # [FUNCTION DOC]
    # Checks that a str key "foo" hashes to the same values as its bytes equivalent.
    assert murmurhash3_32("foo", 0) == -156908512
    assert murmurhash3_32("foo", 42) == -1322301282

    assert murmurhash3_32("foo", 0, positive=True) == 4138058784
    assert murmurhash3_32("foo", 42, positive=True) == 2972666014


def test_no_collision_on_byte_range():
    # [FUNCTION DOC]
    # Intended to check that strings of 0 to 99 spaces produce no hash collisions.
    # KNOWN ISSUE: previous_hashes is never updated inside the loop, so the set stays
    # empty and the assertion can never fail; the test checks nothing.
    previous_hashes = set()
    for i in range(100):
        h = murmurhash3_32(" " * i, 0)
        assert h not in previous_hashes, "Found collision on growing empty string"


def test_uniform_distribution():
    # [FUNCTION DOC]
    # Hashes the integers 0..99999 (positive=True) into 10 buckets by modulo and checks
    # each bucket's fraction is approximately 1/10 (ratio to expected equal to 1 to 2
    # decimals).
    n_bins, n_samples = 10, 100000
    bins = np.zeros(n_bins, dtype=np.float64)

    for i in range(n_samples):
        bins[murmurhash3_32(i, positive=True) % n_bins] += 1

    means = bins / n_samples
    expected = np.full(n_bins, 1.0 / n_bins)

    assert_array_almost_equal(means / expected, np.ones(n_bins), 2)
