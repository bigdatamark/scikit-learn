# [MODULE DOC]
# Tests for KBinsDiscretizer: bin assignment for the uniform, quantile and kmeans
# strategies (with and without sample_weight), n_bins validation, encodings (ordinal,
# onehot, onehot-dense), inverse_transform, dtype handling, subsampling, feature names
# and error messages.
# Module-level X is a shared 4x4 list-of-lists toy dataset used by many tests.
import warnings

import numpy as np
import pytest
import scipy.sparse as sp

from sklearn import clone
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder
from sklearn.utils._testing import (
    assert_allclose,
    assert_allclose_dense_sparse,
    assert_array_almost_equal,
    assert_array_equal,
    ignore_warnings,
)

X = [[-2, 1.5, -4, -1], [-1, 2.5, -3, -0.5], [0, 3.5, -2, 0.5], [1, 4.5, -1, 2]]


@pytest.mark.parametrize(
    "strategy, quantile_method, expected, sample_weight",
    [
        (
            "uniform",
            "averaged_inverted_cdf",  # default
            [[0, 0, 0, 0], [1, 1, 1, 0], [2, 2, 2, 1], [2, 2, 2, 2]],
            None,
        ),
        (
            "kmeans",
            "averaged_inverted_cdf",  # default
            [[0, 0, 0, 0], [0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2]],
            None,
        ),
        (
            "quantile",
            "averaged_inverted_cdf",
            [[0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2], [2, 2, 2, 2]],
            None,
        ),
        (
            "uniform",
            "averaged_inverted_cdf",  # default
            [[0, 0, 0, 0], [1, 1, 1, 0], [2, 2, 2, 1], [2, 2, 2, 2]],
            [1, 1, 2, 1],
        ),
        (
            "uniform",
            "averaged_inverted_cdf",  # default
            [[0, 0, 0, 0], [1, 1, 1, 0], [2, 2, 2, 1], [2, 2, 2, 2]],
            [1, 1, 1, 1],
        ),
        (
            "quantile",
            "averaged_inverted_cdf",
            [[0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2], [2, 2, 2, 2]],
            [1, 1, 2, 1],
        ),
        (
            "quantile",
            "averaged_inverted_cdf",
            [[0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2], [2, 2, 2, 2]],
            [1, 1, 1, 1],
        ),
        (
            "quantile",
            "averaged_inverted_cdf",
            [[0, 0, 0, 0], [0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2]],
            [0, 1, 1, 1],
        ),
        (
            "kmeans",
            "averaged_inverted_cdf",  # default
            [[0, 0, 0, 0], [1, 1, 1, 0], [1, 1, 1, 1], [2, 2, 2, 2]],
            [1, 0, 3, 1],
        ),
        (
            "kmeans",
            "averaged_inverted_cdf",  # default
            [[0, 0, 0, 0], [0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2]],
            [1, 1, 1, 1],
        ),
    ],
)
def test_fit_transform(strategy, quantile_method, expected, sample_weight):
    # [FUNCTION DOC]
    # Verifies ordinal bin assignments of X with n_bins=3 for each strategy and
    # sample_weight setting (including zero weights) match the expected table;
    # UserWarnings about removed small bins are ignored during fit.
    est = KBinsDiscretizer(
        n_bins=3, encode="ordinal", strategy=strategy, quantile_method=quantile_method
    )
    with ignore_warnings(category=UserWarning):
        # Ignore the warning on removed small bins.
        est.fit(X, sample_weight=sample_weight)
    assert_array_equal(est.transform(X), expected)


def test_valid_n_bins():
    # [FUNCTION DOC]
    # Verifies that a Python int and a numpy integer scalar are both accepted as n_bins,
    # and that the fitted n_bins_ has the default int dtype.
    KBinsDiscretizer(n_bins=2, quantile_method="averaged_inverted_cdf").fit_transform(X)
    KBinsDiscretizer(
        n_bins=np.array([2])[0], quantile_method="averaged_inverted_cdf"
    ).fit_transform(X)
    assert KBinsDiscretizer(n_bins=2, quantile_method="averaged_inverted_cdf").fit(
        X
    ).n_bins_.dtype == np.dtype(int)


def test_invalid_n_bins_array():
    # [FUNCTION DOC]
    # Verifies the ValueError messages for an n_bins array with wrong shape (2d, or
    # length not n_features), with values below 2 (indices 0, 3 reported) and with non-
    # integer floats (indices 0, 2 reported).
    # Bad shape
    n_bins = np.full((2, 4), 2.0)
    est = KBinsDiscretizer(n_bins=n_bins, quantile_method="averaged_inverted_cdf")
    err_msg = r"n_bins must be a scalar or array of shape \(n_features,\)."
    with pytest.raises(ValueError, match=err_msg):
        est.fit_transform(X)

    # Incorrect number of features
    n_bins = [1, 2, 2]
    est = KBinsDiscretizer(n_bins=n_bins, quantile_method="averaged_inverted_cdf")
    err_msg = r"n_bins must be a scalar or array of shape \(n_features,\)."
    with pytest.raises(ValueError, match=err_msg):
        est.fit_transform(X)

    # Bad bin values
    n_bins = [1, 2, 2, 1]
    est = KBinsDiscretizer(n_bins=n_bins, quantile_method="averaged_inverted_cdf")
    err_msg = (
        "KBinsDiscretizer received an invalid number of bins "
        "at indices 0, 3. Number of bins must be at least 2, "
        "and must be an int."
    )
    with pytest.raises(ValueError, match=err_msg):
        est.fit_transform(X)

    # Float bin values
    n_bins = [2.1, 2, 2.1, 2]
    est = KBinsDiscretizer(n_bins=n_bins, quantile_method="averaged_inverted_cdf")
    err_msg = (
        "KBinsDiscretizer received an invalid number of bins "
        "at indices 0, 2. Number of bins must be at least 2, "
        "and must be an int."
    )
    with pytest.raises(ValueError, match=err_msg):
        est.fit_transform(X)


@pytest.mark.parametrize(
    "strategy, quantile_method, expected, sample_weight",
    [
        (
            "uniform",
            "averaged_inverted_cdf",  # default
            [[0, 0, 0, 0], [0, 1, 1, 0], [1, 2, 2, 1], [1, 2, 2, 2]],
            None,
        ),
        (
            "kmeans",
            "averaged_inverted_cdf",  # default
            [[0, 0, 0, 0], [0, 0, 0, 0], [1, 1, 1, 1], [1, 2, 2, 2]],
            None,
        ),
        (
            "quantile",
            "linear",
            [[0, 0, 0, 0], [0, 1, 1, 1], [1, 2, 2, 2], [1, 2, 2, 2]],
            None,
        ),
        (
            "quantile",
            "averaged_inverted_cdf",
            [[0, 0, 0, 0], [0, 1, 1, 1], [1, 2, 2, 2], [1, 2, 2, 2]],
            None,
        ),
        (
            "quantile",
            "averaged_inverted_cdf",
            [[0, 0, 0, 0], [0, 1, 1, 1], [1, 2, 2, 2], [1, 2, 2, 2]],
            [1, 1, 1, 1],
        ),
        (
            "quantile",
            "averaged_inverted_cdf",
            [[0, 0, 0, 0], [0, 0, 0, 0], [1, 1, 1, 1], [1, 1, 1, 1]],
            [0, 1, 3, 1],
        ),
        (
            "quantile",
            "averaged_inverted_cdf",
            [[0, 0, 0, 0], [0, 0, 0, 0], [1, 2, 2, 2], [1, 2, 2, 2]],
            [1, 1, 3, 1],
        ),
        (
            "kmeans",
            "averaged_inverted_cdf",  # default
            [[0, 0, 0, 0], [0, 1, 1, 0], [1, 1, 1, 1], [1, 2, 2, 2]],
            [1, 0, 3, 1],
        ),
    ],
)
def test_fit_transform_n_bins_array(strategy, quantile_method, expected, sample_weight):
    # [FUNCTION DOC]
    # Verifies per-feature n_bins [2, 3, 3, 3] yields the expected ordinal codes for
    # each strategy/weight case, that bin_edges_ is an object array of length
    # n_features, and that each feature's edges have n_bins_ + 1 entries.
    est = KBinsDiscretizer(
        n_bins=[2, 3, 3, 3],
        encode="ordinal",
        strategy=strategy,
        quantile_method=quantile_method,
    ).fit(X, sample_weight=sample_weight)
    assert_array_equal(est.transform(X), expected)

    # test the shape of bin_edges_
    n_features = np.array(X).shape[1]
    assert est.bin_edges_.shape == (n_features,)
    for bin_edges, n_bins in zip(est.bin_edges_, est.n_bins_):
        assert bin_edges.shape == (n_bins + 1,)


@pytest.mark.filterwarnings("ignore: Bins whose width are too small")
def test_kbinsdiscretizer_effect_sample_weight():
    # [FUNCTION DOC]
    # Verifies that zero sample weights exclude samples from quantile edges: with 10
    # bins only the four weighted samples (plus their midpoints) form edges [-2, -1, 0,
    # 1, 3], and the zero-weight large values fall into the last bin.
    """Check the impact of `sample_weight` one computed quantiles."""
    X = np.array([[-2], [-1], [1], [3], [500], [1000]])
    # add a large number of bins such that each sample with a non-null weight
    # will be used as bin edge
    est = KBinsDiscretizer(
        n_bins=10,
        encode="ordinal",
        strategy="quantile",
        quantile_method="averaged_inverted_cdf",
    )
    est.fit(X, sample_weight=[1, 1, 1, 1, 0, 0])
    assert_allclose(est.bin_edges_[0], [-2, -1, 0, 1, 3])
    assert_allclose(est.transform(X), [[0.0], [1.0], [3.0], [3.0], [3.0], [3.0]])


@pytest.mark.parametrize("strategy", ["kmeans", "quantile"])
def test_kbinsdiscretizer_no_mutating_sample_weight(strategy):
    # [FUNCTION DOC]
    # Verifies fit does not modify the passed sample_weight array in place for the
    # kmeans and quantile strategies, by comparing against a copy.
    """Make sure that `sample_weight` is not changed in place."""

    if strategy == "quantile":
        est = KBinsDiscretizer(
            n_bins=3,
            encode="ordinal",
            strategy=strategy,
            quantile_method="averaged_inverted_cdf",
        )
    else:
        est = KBinsDiscretizer(n_bins=3, encode="ordinal", strategy=strategy)
    sample_weight = np.array([1, 3, 1, 2], dtype=np.float64)
    sample_weight_copy = np.copy(sample_weight)
    est.fit(X, sample_weight=sample_weight)
    assert_allclose(sample_weight, sample_weight_copy)


@pytest.mark.parametrize("strategy", ["uniform", "kmeans", "quantile"])
def test_same_min_max(strategy):
    # [FUNCTION DOC]
    # Verifies a constant feature triggers the 'Feature 0 is constant' UserWarning, gets
    # n_bins_ of 1, and is transformed to all zeros.
    # Side effects: calls warnings.simplefilter('always') at function level, altering
    # the global warning filters beyond this test.
    warnings.simplefilter("always")
    X = np.array([[1, -2], [1, -1], [1, 0], [1, 1]])
    if strategy == "quantile":
        est = KBinsDiscretizer(
            strategy=strategy,
            n_bins=3,
            encode="ordinal",
            quantile_method="averaged_inverted_cdf",
        )
    else:
        est = KBinsDiscretizer(strategy=strategy, n_bins=3, encode="ordinal")
    warning_message = "Feature 0 is constant and will be replaced with 0."
    with pytest.warns(UserWarning, match=warning_message):
        est.fit(X)
    assert est.n_bins_[0] == 1
    # replace the feature with zeros
    Xt = est.transform(X)
    assert_array_equal(Xt[:, 0], np.zeros(X.shape[0]))


def test_transform_1d_behavior():
    # [FUNCTION DOC]
    # Verifies 1d input is rejected with ValueError both at fit and at transform (after
    # fitting on the 2d reshaped column).
    X = np.arange(4)
    est = KBinsDiscretizer(n_bins=2, quantile_method="averaged_inverted_cdf")
    with pytest.raises(ValueError):
        est.fit(X)

    est = KBinsDiscretizer(n_bins=2, quantile_method="averaged_inverted_cdf")
    est.fit(X.reshape(-1, 1))
    with pytest.raises(ValueError):
        est.transform(X)


@pytest.mark.parametrize("i", range(1, 9))
def test_numeric_stability(i):
    # [FUNCTION DOC]
    # Verifies uniform-quantile binning into 2 bins gives the same codes when the data
    # is scaled down by 10**i for i in 1..8 (no floating-point edge errors).
    X_init = np.array([2.0, 4.0, 6.0, 8.0, 10.0]).reshape(-1, 1)
    Xt_expected = np.array([0, 0, 1, 1, 1]).reshape(-1, 1)

    # Test up to discretizing nano units
    X = X_init / 10**i
    Xt = KBinsDiscretizer(
        n_bins=2, encode="ordinal", quantile_method="averaged_inverted_cdf"
    ).fit_transform(X)
    assert_array_equal(Xt_expected, Xt)


def test_encode_options():
    # [FUNCTION DOC]
    # Verifies onehot-dense output is dense and equals OneHotEncoder (dense) applied to
    # the ordinal codes with categories 0..n_bins-1, and onehot output is sparse and
    # equals the sparse OneHotEncoder result.
    est = KBinsDiscretizer(
        n_bins=[2, 3, 3, 3], encode="ordinal", quantile_method="averaged_inverted_cdf"
    ).fit(X)
    Xt_1 = est.transform(X)
    est = KBinsDiscretizer(
        n_bins=[2, 3, 3, 3],
        encode="onehot-dense",
        quantile_method="averaged_inverted_cdf",
    ).fit(X)
    Xt_2 = est.transform(X)
    assert not sp.issparse(Xt_2)
    assert_array_equal(
        OneHotEncoder(
            categories=[np.arange(i) for i in [2, 3, 3, 3]], sparse_output=False
        ).fit_transform(Xt_1),
        Xt_2,
    )
    est = KBinsDiscretizer(
        n_bins=[2, 3, 3, 3], encode="onehot", quantile_method="averaged_inverted_cdf"
    ).fit(X)
    Xt_3 = est.transform(X)
    assert sp.issparse(Xt_3)
    assert_array_equal(
        OneHotEncoder(
            categories=[np.arange(i) for i in [2, 3, 3, 3]], sparse_output=True
        )
        .fit_transform(Xt_1)
        .toarray(),
        Xt_3.toarray(),
    )


@pytest.mark.parametrize(
    "strategy, quantile_method, expected_2bins, expected_3bins, expected_5bins",
    [
        (
            "uniform",
            "averaged_inverted_cdf",
            [0, 0, 0, 0, 1, 1],
            [0, 0, 0, 0, 2, 2],
            [0, 0, 1, 1, 4, 4],
        ),
        (
            "kmeans",
            "averaged_inverted_cdf",
            [0, 0, 0, 0, 1, 1],
            [0, 0, 1, 1, 2, 2],
            [0, 0, 1, 2, 3, 4],
        ),
        (
            "quantile",
            "averaged_inverted_cdf",
            [0, 0, 0, 1, 1, 1],
            [0, 0, 1, 1, 2, 2],
            [0, 1, 2, 3, 4, 4],
        ),
    ],
)
def test_nonuniform_strategies(
    strategy, quantile_method, expected_2bins, expected_3bins, expected_5bins
):
    # [FUNCTION DOC]
    # Verifies ordinal codes on a skewed 1d dataset for 2, 3 and 5 bins match the
    # expected arrays per strategy.
    X = np.array([0, 0.5, 2, 3, 9, 10]).reshape(-1, 1)

    # with 2 bins
    est = KBinsDiscretizer(
        n_bins=2, strategy=strategy, quantile_method=quantile_method, encode="ordinal"
    )
    Xt = est.fit_transform(X)
    assert_array_equal(expected_2bins, Xt.ravel())

    # with 3 bins
    est = KBinsDiscretizer(
        n_bins=3, strategy=strategy, quantile_method=quantile_method, encode="ordinal"
    )
    Xt = est.fit_transform(X)
    assert_array_equal(expected_3bins, Xt.ravel())

    # with 5 bins
    est = KBinsDiscretizer(
        n_bins=5, strategy=strategy, quantile_method=quantile_method, encode="ordinal"
    )
    Xt = est.fit_transform(X)
    assert_array_equal(expected_5bins, Xt.ravel())


@pytest.mark.parametrize(
    "strategy, expected_inv,quantile_method",
    [
        (
            "uniform",
            [
                [-1.5, 2.0, -3.5, -0.5],
                [-0.5, 3.0, -2.5, -0.5],
                [0.5, 4.0, -1.5, 0.5],
                [0.5, 4.0, -1.5, 1.5],
            ],
            "averaged_inverted_cdf",  # default
        ),
        (
            "kmeans",
            [
                [-1.375, 2.125, -3.375, -0.5625],
                [-1.375, 2.125, -3.375, -0.5625],
                [-0.125, 3.375, -2.125, 0.5625],
                [0.75, 4.25, -1.25, 1.625],
            ],
            "averaged_inverted_cdf",  # default
        ),
        (
            "quantile",
            [
                [-1.5, 2.0, -3.5, -0.75],
                [-0.5, 3.0, -2.5, 0.0],
                [0.5, 4.0, -1.5, 1.25],
                [0.5, 4.0, -1.5, 1.25],
            ],
            "averaged_inverted_cdf",
        ),
    ],
)
@pytest.mark.parametrize("encode", ["ordinal", "onehot", "onehot-dense"])
def test_inverse_transform(strategy, encode, expected_inv, quantile_method):
    # [FUNCTION DOC]
    # Verifies inverse_transform maps codes to the bin centers in the expected table for
    # each strategy and each encoding.
    kbd = KBinsDiscretizer(
        n_bins=3, strategy=strategy, quantile_method=quantile_method, encode=encode
    )
    Xt = kbd.fit_transform(X)
    Xinv = kbd.inverse_transform(Xt)
    assert_array_almost_equal(expected_inv, Xinv)


@pytest.mark.parametrize("strategy", ["uniform", "kmeans", "quantile"])
def test_transform_outside_fit_range(strategy):
    # [FUNCTION DOC]
    # Verifies values outside the fit range are clipped into the first and last bins:
    # transforming -2 and 5 gives codes 0 and n_bins_ - 1.
    X = np.array([0, 1, 2, 3])[:, None]

    if strategy == "quantile":
        kbd = KBinsDiscretizer(
            n_bins=4,
            strategy=strategy,
            encode="ordinal",
            quantile_method="averaged_inverted_cdf",
        )
    else:
        kbd = KBinsDiscretizer(n_bins=4, strategy=strategy, encode="ordinal")
    kbd.fit(X)

    X2 = np.array([-2, 5])[:, None]
    X2t = kbd.transform(X2)
    assert_array_equal(X2t.max(axis=0) + 1, kbd.n_bins_)
    assert_array_equal(X2t.min(axis=0), [0])


def test_overwrite():
    # [FUNCTION DOC]
    # Verifies fit_transform does not modify X, inverse_transform does not modify Xt,
    # and inverse gives the expected bin centers.
    X = np.array([0, 1, 2, 3])[:, None]
    X_before = X.copy()

    est = KBinsDiscretizer(
        n_bins=3, quantile_method="averaged_inverted_cdf", encode="ordinal"
    )
    Xt = est.fit_transform(X)
    assert_array_equal(X, X_before)

    Xt_before = Xt.copy()
    Xinv = est.inverse_transform(Xt)
    assert_array_equal(Xt, Xt_before)
    assert_array_equal(Xinv, np.array([[0.5], [1.5], [2.5], [2.5]]))


@pytest.mark.parametrize(
    "strategy, expected_bin_edges, quantile_method",
    [
        ("quantile", [0, 1.5, 3], "averaged_inverted_cdf"),
        ("kmeans", [0, 1.5, 3], "averaged_inverted_cdf"),
    ],
)
def test_redundant_bins(strategy, expected_bin_edges, quantile_method):
    # [FUNCTION DOC]
    # Verifies that when duplicate edges occur (data with only two distinct values) a
    # 'Consider decreasing the number of bins' UserWarning is raised and redundant edges
    # are removed, leaving [0, 1.5, 3].
    X = [[0], [0], [0], [0], [3], [3]]
    kbd = KBinsDiscretizer(
        n_bins=3, strategy=strategy, quantile_method=quantile_method, subsample=None
    )
    warning_message = "Consider decreasing the number of bins."
    with pytest.warns(UserWarning, match=warning_message):
        kbd.fit(X)

    assert_array_almost_equal(kbd.bin_edges_[0], expected_bin_edges)


def test_percentile_numeric_stability():
    # [FUNCTION DOC]
    # Verifies quantile edges with quantile_method='linear' on a tiny dataset are
    # computed without numerical noise (expected evenly spaced edges after redundant-bin
    # removal warning) and transform gives the expected codes.
    X = np.array([0.05, 0.05, 0.95]).reshape(-1, 1)
    bin_edges = np.array([0.05, 0.23, 0.41, 0.59, 0.77, 0.95])
    Xt = np.array([0, 0, 4]).reshape(-1, 1)
    kbd = KBinsDiscretizer(
        n_bins=10,
        encode="ordinal",
        strategy="quantile",
        quantile_method="linear",
    )
    ## TODO: change to averaged inverted cdf, but that means we only get bin
    ## edges of 0.05 and 0.95 and nothing in between

    warning_message = "Consider decreasing the number of bins."
    with pytest.warns(UserWarning, match=warning_message):
        kbd.fit(X)

    assert_array_almost_equal(kbd.bin_edges_[0], bin_edges)
    assert_array_almost_equal(kbd.transform(X), Xt)


@pytest.mark.parametrize("in_dtype", [np.float16, np.float32, np.float64])
@pytest.mark.parametrize("out_dtype", [None, np.float32, np.float64])
@pytest.mark.parametrize("encode", ["ordinal", "onehot", "onehot-dense"])
def test_consistent_dtype(in_dtype, out_dtype, encode):
    # [FUNCTION DOC]
    # Verifies output dtype: the dtype parameter wins when set; otherwise float16 input
    # yields float64 and float32/float64 inputs are preserved; for all encodings.
    X_input = np.array(X, dtype=in_dtype)
    kbd = KBinsDiscretizer(
        n_bins=3,
        encode=encode,
        quantile_method="averaged_inverted_cdf",
        dtype=out_dtype,
    )
    kbd.fit(X_input)

    # test output dtype
    if out_dtype is not None:
        expected_dtype = out_dtype
    elif out_dtype is None and X_input.dtype == np.float16:
        # wrong numeric input dtype are cast in np.float64
        expected_dtype = np.float64
    else:
        expected_dtype = X_input.dtype
    Xt = kbd.transform(X_input)
    assert Xt.dtype == expected_dtype


@pytest.mark.parametrize("input_dtype", [np.float16, np.float32, np.float64])
@pytest.mark.parametrize("encode", ["ordinal", "onehot", "onehot-dense"])
def test_32_equal_64(input_dtype, encode):
    # [FUNCTION DOC]
    # Verifies float32 and float64 output dtypes give numerically close results (dense
    # or sparse) for every input dtype and encoding.
    # TODO this check is redundant with common checks and can be removed
    #  once #16290 is merged
    X_input = np.array(X, dtype=input_dtype)

    # 32 bit output
    kbd_32 = KBinsDiscretizer(
        n_bins=3,
        encode=encode,
        quantile_method="averaged_inverted_cdf",
        dtype=np.float32,
    )
    kbd_32.fit(X_input)
    Xt_32 = kbd_32.transform(X_input)

    # 64 bit output
    kbd_64 = KBinsDiscretizer(
        n_bins=3,
        encode=encode,
        quantile_method="averaged_inverted_cdf",
        dtype=np.float64,
    )
    kbd_64.fit(X_input)
    Xt_64 = kbd_64.transform(X_input)

    assert_allclose_dense_sparse(Xt_32, Xt_64)


def test_kbinsdiscretizer_subsample_default():
    # [FUNCTION DOC]
    # Verifies that with the default subsample on a small dataset, bin edges equal those
    # obtained with subsample=None (no subsampling happens), comparing elementwise and
    # by shape.
    # KNOWN ISSUE: the loop variable bin_kbd_with_subsampling actually holds edges from
    # the model fitted without subsampling.
    # Since the size of X is small (< 2e5), subsampling will not take place.
    X = np.array([-2, 1.5, -4, -1]).reshape(-1, 1)
    kbd_default = KBinsDiscretizer(
        n_bins=10,
        encode="ordinal",
        strategy="quantile",
        quantile_method="averaged_inverted_cdf",
    )
    kbd_default.fit(X)

    kbd_without_subsampling = clone(kbd_default)
    kbd_without_subsampling.set_params(subsample=None)
    kbd_without_subsampling.fit(X)

    for bin_kbd_default, bin_kbd_with_subsampling in zip(
        kbd_default.bin_edges_[0], kbd_without_subsampling.bin_edges_[0]
    ):
        np.testing.assert_allclose(bin_kbd_default, bin_kbd_with_subsampling)
    assert kbd_default.bin_edges_.shape == kbd_without_subsampling.bin_edges_.shape


@pytest.mark.parametrize(
    "encode, expected_names",
    [
        (
            "onehot",
            [
                f"feat{col_id}_{float(bin_id)}"
                for col_id in range(3)
                for bin_id in range(4)
            ],
        ),
        (
            "onehot-dense",
            [
                f"feat{col_id}_{float(bin_id)}"
                for col_id in range(3)
                for bin_id in range(4)
            ],
        ),
        ("ordinal", [f"feat{col_id}" for col_id in range(3)]),
    ],
)
def test_kbinsdiscrtizer_get_feature_names_out(encode, expected_names):
    # [FUNCTION DOC]
    # Non-regression for #22731: verifies get_feature_names_out with custom input names
    # has as many names as output columns and matches '<feat>_<bin as float>' for onehot
    # encodings or the plain names for ordinal.
    """Check get_feature_names_out for different settings.
    Non-regression test for #22731
    """
    X = [[-2, 1, -4], [-1, 2, -3], [0, 3, -2], [1, 4, -1]]

    kbd = KBinsDiscretizer(
        n_bins=4, encode=encode, quantile_method="averaged_inverted_cdf"
    ).fit(X)
    Xt = kbd.transform(X)

    input_features = [f"feat{i}" for i in range(3)]
    output_names = kbd.get_feature_names_out(input_features)
    assert Xt.shape[1] == output_names.shape[0]

    assert_array_equal(output_names, expected_names)


@pytest.mark.parametrize("strategy", ["uniform", "kmeans", "quantile"])
def test_kbinsdiscretizer_subsample(strategy, global_random_seed):
    # [FUNCTION DOC]
    # Verifies that on 100000 uniform samples, bin edges fitted with subsample=50000 are
    # within 1% relative tolerance of those fitted on all data, for each strategy,
    # seeded by global_random_seed.
    # Check that the bin edges are almost the same when subsampling is used.
    X = np.random.RandomState(global_random_seed).random_sample((100000, 1)) + 1

    if strategy == "quantile":
        kbd_subsampling = KBinsDiscretizer(
            strategy=strategy,
            subsample=50000,
            random_state=global_random_seed,
            quantile_method="averaged_inverted_cdf",
        )
    else:
        kbd_subsampling = KBinsDiscretizer(
            strategy=strategy, subsample=50000, random_state=global_random_seed
        )
    kbd_subsampling.fit(X)

    kbd_no_subsampling = clone(kbd_subsampling)
    kbd_no_subsampling.set_params(subsample=None)
    kbd_no_subsampling.fit(X)

    # We use a large tolerance because we can't expect the bin edges to be exactly the
    # same when subsampling is used.
    assert_allclose(
        kbd_subsampling.bin_edges_[0], kbd_no_subsampling.bin_edges_[0], rtol=1e-2
    )


def test_invalid_quantile_method_with_sample_weight():
    # [FUNCTION DOC]
    # Verifies fitting with sample_weight, strategy='quantile' and
    # quantile_method='linear' raises ValueError with the message naming the allowed
    # methods.
    X = [[-2, 1, -4], [-1, 2, -3], [0, 3, -2], [1, 4, -1]]
    expected_msg = (
        "When fitting with strategy='quantile' and sample weights, "
        "quantile_method should either be set to 'averaged_inverted_cdf' or "
        "'inverted_cdf', got quantile_method='linear' instead."
    )
    with pytest.raises(
        ValueError,
        match=expected_msg,
    ):
        KBinsDiscretizer(strategy="quantile", quantile_method="linear").fit(
            X,
            sample_weight=[1, 1, 2, 2],
        )
