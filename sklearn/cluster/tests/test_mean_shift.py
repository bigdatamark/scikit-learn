# [MODULE DOC]
# Tests for mean shift clustering: the MeanShift estimator, the mean_shift function,
# estimate_bandwidth and get_bin_seeds.
# Module-level fixture data: X is 300 two-dimensional samples drawn by make_blobs around
# 3 centers shifted by +10 (cluster_std 0.4, random_state 11), shared by most tests.
# Many tests use the global_dtype fixture to run under float32 and float64.
"""
Testing for mean shift clustering methods

"""

import warnings

import numpy as np
import pytest

from sklearn.cluster import MeanShift, estimate_bandwidth, get_bin_seeds, mean_shift
from sklearn.datasets import make_blobs
from sklearn.metrics import v_measure_score
from sklearn.utils._testing import assert_allclose, assert_array_equal

n_clusters = 3
centers = np.array([[1, 1], [-1, -1], [1, -1]]) + 10
X, _ = make_blobs(
    n_samples=300,
    n_features=2,
    centers=centers,
    cluster_std=0.4,
    shuffle=True,
    random_state=11,
)


def test_convergence_of_1d_constant_data():
    # [FUNCTION DOC]
    # Non-regression (issue 28926): fitting MeanShift on 10 identical 1D values must
    # converge, verified by n_iter_ being strictly less than max_iter.
    # Test convergence using 1D constant data
    # Non-regression test for:
    # https://github.com/scikit-learn/scikit-learn/issues/28926
    model = MeanShift()
    n_iter = model.fit(np.ones(10).reshape(-1, 1)).n_iter_
    assert n_iter < model.max_iter


def test_estimate_bandwidth():
    # [FUNCTION DOC]
    # Verifies estimate_bandwidth on the shared blobs with n_samples=200 returns a value in
    # the range [0.9, 1.5].
    # Test estimate_bandwidth
    bandwidth = estimate_bandwidth(X, n_samples=200)
    assert 0.9 <= bandwidth <= 1.5


def test_estimate_bandwidth_1sample(global_dtype):
    # [FUNCTION DOC]
    # Verifies estimate_bandwidth with n_samples=1 and quantile=0.3 (forcing n_neighbors=1)
    # returns approximately 0 (each point is its own nearest neighbor).
    # Inputs: global_dtype fixture used to cast X.
    # KNOWN ISSUE: the dtype assertion compares against X.dtype (always float64) rather
    # than global_dtype, so for float32 it only passes if the result is float64; the check
    # does not verify dtype preservation as the cast suggests.
    # Test estimate_bandwidth when n_samples=1 and quantile<1, so that
    # n_neighbors is set to 1.
    bandwidth = estimate_bandwidth(
        X.astype(global_dtype, copy=False), n_samples=1, quantile=0.3
    )

    assert bandwidth.dtype == X.dtype
    assert bandwidth == pytest.approx(0.0, abs=1e-5)


@pytest.mark.parametrize(
    "bandwidth, cluster_all, expected, first_cluster_label",
    [(1.2, True, 3, 0), (1.2, False, 4, -1)],
)
def test_mean_shift(
    global_dtype, bandwidth, cluster_all, expected, first_cluster_label
):
    # [FUNCTION DOC]
    # Parametrized over bandwidth/cluster_all: fits MeanShift and also calls mean_shift,
    # and checks the number of unique labels equals expected (3 when all points are
    # clustered, 4 including the orphan label -1 otherwise), that the smallest label equals
    # first_cluster_label, and that cluster centers keep global_dtype.
    # Test MeanShift algorithm
    X_with_global_dtype = X.astype(global_dtype, copy=False)
    ms = MeanShift(bandwidth=bandwidth, cluster_all=cluster_all)
    labels = ms.fit(X_with_global_dtype).labels_
    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)
    assert n_clusters_ == expected
    assert labels_unique[0] == first_cluster_label
    assert ms.cluster_centers_.dtype == global_dtype

    cluster_centers, labels_mean_shift = mean_shift(
        X_with_global_dtype, cluster_all=cluster_all
    )
    labels_mean_shift_unique = np.unique(labels_mean_shift)
    n_clusters_mean_shift = len(labels_mean_shift_unique)
    assert n_clusters_mean_shift == expected
    assert labels_mean_shift_unique[0] == first_cluster_label
    assert cluster_centers.dtype == global_dtype


# TODO: remove mark once loky bug is fixed:
# https://github.com/joblib/loky/issues/458
@pytest.mark.thread_unsafe
def test_parallel(global_dtype, global_random_seed):
    # [FUNCTION DOC]
    # Verifies that MeanShift with n_jobs=2 gives the same cluster centers (allclose), same
    # centers dtype and identical labels as the sequential fit, on 50 blobs generated with
    # global_random_seed. Marked thread_unsafe due to a loky bug.
    centers = np.array([[1, 1], [-1, -1], [1, -1]]) + 10
    X, _ = make_blobs(
        n_samples=50,
        n_features=2,
        centers=centers,
        cluster_std=0.4,
        shuffle=True,
        random_state=global_random_seed,
    )

    X = X.astype(global_dtype, copy=False)

    ms1 = MeanShift(n_jobs=2)
    ms1.fit(X)

    ms2 = MeanShift()
    ms2.fit(X)

    assert_allclose(ms1.cluster_centers_, ms2.cluster_centers_)
    assert ms1.cluster_centers_.dtype == ms2.cluster_centers_.dtype
    assert_array_equal(ms1.labels_, ms2.labels_)


def test_meanshift_predict(global_dtype):
    # [FUNCTION DOC]
    # Verifies fit_predict labels equal predict labels on the same data after fitting
    # with bandwidth 1.2.
    # Test MeanShift.predict
    ms = MeanShift(bandwidth=1.2)
    X_with_global_dtype = X.astype(global_dtype, copy=False)
    labels = ms.fit_predict(X_with_global_dtype)
    labels2 = ms.predict(X_with_global_dtype)
    assert_array_equal(labels, labels2)


def test_meanshift_all_orphans():
    # [FUNCTION DOC]
    # Verifies that seeds far from all data with a tiny bandwidth (0.1) make fit raise
    # ValueError with message "No point was within bandwidth=0.1".
    # init away from the data, crash with a sensible warning
    ms = MeanShift(bandwidth=0.1, seeds=[[-9, -9], [-10, -10]])
    msg = "No point was within bandwidth=0.1"
    with pytest.raises(ValueError, match=msg):
        ms.fit(
            X,
        )


def test_unfitted():
    # [FUNCTION DOC]
    # Verifies that an unfitted MeanShift has no cluster_centers_ or labels_ attribute.
    # Non-regression: before fit, there should be not fitted attributes.
    ms = MeanShift()
    assert not hasattr(ms, "cluster_centers_")
    assert not hasattr(ms, "labels_")


def test_cluster_intensity_tie(global_dtype):
    # [FUNCTION DOC]
    # Verifies deterministic tie-breaking when two clusters have equal intensity (3 points
    # each): labels follow data order, so permuting the two groups swaps label ids.
    X = np.array([[1, 1], [2, 1], [1, 0], [4, 7], [3, 5], [3, 6]], dtype=global_dtype)
    c1 = MeanShift(bandwidth=2).fit(X)

    X = np.array([[4, 7], [3, 5], [3, 6], [1, 1], [2, 1], [1, 0]], dtype=global_dtype)
    c2 = MeanShift(bandwidth=2).fit(X)
    assert_array_equal(c1.labels_, [1, 1, 1, 0, 0, 0])
    assert_array_equal(c2.labels_, [0, 0, 0, 1, 1, 1])


def test_bin_seeds(global_dtype):
    # [FUNCTION DOC]
    # Verifies get_bin_seeds: with bin size 1 and min_bin_freq 1 gives 3 expected bin
    # centers, with min_bin_freq 2 gives 2; with bin size 0.01 (every point its own bin)
    # it falls back to returning the data (warning suppressed); on tight blobs around
    # (0,0) and (1,1) with bin size 1 it returns exactly those two seeds.
    # Test the bin seeding technique which can be used in the mean shift
    # algorithm
    # Data is just 6 points in the plane
    X = np.array(
        [[1.0, 1.0], [1.4, 1.4], [1.8, 1.2], [2.0, 1.0], [2.1, 1.1], [0.0, 0.0]],
        dtype=global_dtype,
    )

    # With a bin coarseness of 1.0 and min_bin_freq of 1, 3 bins should be
    # found
    ground_truth = {(1.0, 1.0), (2.0, 1.0), (0.0, 0.0)}
    test_bins = get_bin_seeds(X, 1, 1)
    test_result = set(tuple(p) for p in test_bins)
    assert len(ground_truth.symmetric_difference(test_result)) == 0

    # With a bin coarseness of 1.0 and min_bin_freq of 2, 2 bins should be
    # found
    ground_truth = {(1.0, 1.0), (2.0, 1.0)}
    test_bins = get_bin_seeds(X, 1, 2)
    test_result = set(tuple(p) for p in test_bins)
    assert len(ground_truth.symmetric_difference(test_result)) == 0

    # With a bin size of 0.01 and min_bin_freq of 1, 6 bins should be found
    # we bail and use the whole data here.
    with warnings.catch_warnings(record=True):
        test_bins = get_bin_seeds(X, 0.01, 1)
    assert_allclose(test_bins, X)

    # tight clusters around [0, 0] and [1, 1], only get two bins
    X, _ = make_blobs(
        n_samples=100,
        n_features=2,
        centers=[[0, 0], [1, 1]],
        cluster_std=0.1,
        random_state=0,
    )
    X = X.astype(global_dtype, copy=False)
    test_bins = get_bin_seeds(X, 1)
    assert_array_equal(test_bins, [[0, 0], [1, 1]])


@pytest.mark.parametrize("max_iter", [1, 100])
def test_max_iter(max_iter):
    # [FUNCTION DOC]
    # Parametrized over max_iter (1, 100): checks mean_shift and MeanShift give the same
    # number of centers with allclose values, and that n_iter_ never exceeds max_iter.
    clusters1, _ = mean_shift(X, max_iter=max_iter)
    ms = MeanShift(max_iter=max_iter).fit(X)
    clusters2 = ms.cluster_centers_

    assert ms.n_iter_ <= ms.max_iter
    assert len(clusters1) == len(clusters2)

    for c1, c2 in zip(clusters1, clusters2):
        assert np.allclose(c1, c2)


def test_mean_shift_zero_bandwidth(global_dtype):
    # [FUNCTION DOC]
    # Verifies behavior when estimate_bandwidth returns 0 on a 1D dataset of repeated
    # values: get_bin_seeds with bin_size 0 returns the input object itself, and
    # MeanShift with and without bin seeding both recover the 3 expected groups (perfect
    # v_measure) with identical cluster centers.
    # Check that mean shift works when the estimated bandwidth is 0.
    X = np.array([1, 1, 1, 2, 2, 2, 3, 3], dtype=global_dtype).reshape(-1, 1)

    # estimate_bandwidth with default args returns 0 on this dataset
    bandwidth = estimate_bandwidth(X)
    assert bandwidth == 0

    # get_bin_seeds with a 0 bin_size should return the dataset itself
    assert get_bin_seeds(X, bin_size=bandwidth) is X

    # MeanShift with binning and a 0 estimated bandwidth should be equivalent
    # to no binning.
    ms_binning = MeanShift(bin_seeding=True, bandwidth=None).fit(X)
    ms_nobinning = MeanShift(bin_seeding=False).fit(X)
    expected_labels = np.array([0, 0, 0, 1, 1, 1, 2, 2])

    assert v_measure_score(ms_binning.labels_, expected_labels) == pytest.approx(1)
    assert v_measure_score(ms_nobinning.labels_, expected_labels) == pytest.approx(1)
    assert_allclose(ms_binning.cluster_centers_, ms_nobinning.cluster_centers_)
