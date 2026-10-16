# [MODULE DOC]
# Shared tests for KDTree and BallTree: kernel density, radius queries, two-point
# correlation, the NeighborsHeap / nodeheap_sort / simultaneous_sort helpers,
# comparison with scipy gaussian_kde, k-NN queries versus brute force across
# metrics, and pickling. Module globals: a seeded RNG, a random SPD matrix for
# mahalanobis, METRICS (name -> params, dimension 3) and the metric lists each tree
# supports.
# SPDX-License-Identifier: BSD-3-Clause

import itertools
import pickle

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_almost_equal

from sklearn.metrics import DistanceMetric
from sklearn.neighbors._ball_tree import (
    BallTree,
    kernel_norm,
)
from sklearn.neighbors._ball_tree import (
    NeighborsHeap64 as NeighborsHeapBT,
)
from sklearn.neighbors._ball_tree import (
    nodeheap_sort as nodeheap_sort_bt,
)
from sklearn.neighbors._ball_tree import (
    simultaneous_sort as simultaneous_sort_bt,
)
from sklearn.neighbors._kd_tree import (
    KDTree,
)
from sklearn.neighbors._kd_tree import (
    NeighborsHeap64 as NeighborsHeapKDT,
)
from sklearn.neighbors._kd_tree import (
    nodeheap_sort as nodeheap_sort_kdt,
)
from sklearn.neighbors._kd_tree import (
    simultaneous_sort as simultaneous_sort_kdt,
)
from sklearn.utils import check_random_state

rng = np.random.RandomState(42)
V_mahalanobis = rng.rand(3, 3)
V_mahalanobis = np.dot(V_mahalanobis, V_mahalanobis.T)

DIMENSION = 3

METRICS = {
    "euclidean": {},
    "manhattan": {},
    "minkowski": dict(p=3),
    "chebyshev": {},
    "seuclidean": dict(V=rng.random_sample(DIMENSION)),
    "mahalanobis": dict(V=V_mahalanobis),
}

KD_TREE_METRICS = ["euclidean", "manhattan", "chebyshev", "minkowski"]
BALL_TREE_METRICS = list(METRICS)


def dist_func(x1, x2, p):
    # [FUNCTION DOC]
    # HELPER: User-defined Minkowski-like distance: sum of (x1 - x2)^p, raised to 1/p.
    # Inputs: x1, x2 - 1D arrays; p - exponent. Outputs: float.
    # KNOWN ISSUE: no absolute value, so it is only a valid distance for even p (the
    # tests use p=2).
    return np.sum((x1 - x2) ** p) ** (1.0 / p)


def compute_kernel_slow(Y, X, kernel, h):
    # [FUNCTION DOC]
    # HELPER: Brute-force unnormalized-by-n KDE reference: pairwise Euclidean distances
    # from each Y row to every X row, apply the kernel with bandwidth h and sum over X,
    # times kernel_norm(h, d, kernel).
    # Outputs: (n_Y,) array. Exceptions: ValueError for unknown kernel.
    # Duplicates the helper in test_kde (which also divides by n and resolves rules).
    d = np.sqrt(((Y[:, None, :] - X) ** 2).sum(-1))
    norm = kernel_norm(h, X.shape[1], kernel)

    if kernel == "gaussian":
        return norm * np.exp(-0.5 * (d * d) / (h * h)).sum(-1)
    elif kernel == "tophat":
        return norm * (d < h).sum(-1)
    elif kernel == "epanechnikov":
        return norm * ((1.0 - (d * d) / (h * h)) * (d < h)).sum(-1)
    elif kernel == "exponential":
        return norm * (np.exp(-d / h)).sum(-1)
    elif kernel == "linear":
        return norm * ((1 - d / h) * (d < h)).sum(-1)
    elif kernel == "cosine":
        return norm * (np.cos(0.5 * np.pi * d / h) * (d < h)).sum(-1)
    else:
        raise ValueError("kernel not recognized")


def brute_force_neighbors(X, Y, k, metric, **kwargs):
    # [FUNCTION DOC]
    # HELPER: Computes full pairwise distances Y-to-X with DistanceMetric(metric,
    # **kwargs), takes the k smallest per row via argsort and returns (dist, ind), each
    # of shape (n_Y, k).
    D = DistanceMetric.get_metric(metric, **kwargs).pairwise(Y, X)
    ind = np.argsort(D, axis=1)[:, :k]
    dist = D[np.arange(Y.shape[0])[:, None], ind]
    return dist, ind


@pytest.mark.parametrize("Cls", [KDTree, BallTree])
@pytest.mark.parametrize(
    "kernel", ["gaussian", "tophat", "epanechnikov", "exponential", "linear", "cosine"]
)
@pytest.mark.parametrize("h", [0.01, 0.1, 1])
@pytest.mark.parametrize("rtol", [0, 1e-5])
@pytest.mark.parametrize("atol", [1e-6, 1e-2])
@pytest.mark.parametrize("breadth_first", [True, False])
def test_kernel_density(
    Cls, kernel, h, rtol, atol, breadth_first, n_samples=100, n_features=3
):
    # [FUNCTION DOC]
    # Checks tree.kernel_density matches compute_kernel_slow for both tree classes, all
    # kernels, several bandwidths, tolerances and breadth_first settings.
    rng = check_random_state(1)
    X = rng.random_sample((n_samples, n_features))
    Y = rng.random_sample((n_samples, n_features))
    dens_true = compute_kernel_slow(Y, X, kernel, h)

    tree = Cls(X, leaf_size=10)
    dens = tree.kernel_density(
        Y, h, atol=atol, rtol=rtol, kernel=kernel, breadth_first=breadth_first
    )
    assert_allclose(dens, dens_true, atol=atol, rtol=max(rtol, 1e-7))


@pytest.mark.parametrize("Cls", [KDTree, BallTree])
def test_neighbor_tree_query_radius(Cls, n_samples=100, n_features=10):
    # [FUNCTION DOC]
    # For 100 radii, checks query_radius around the origin returns exactly the indices
    # whose distance is <= r (with 1e-15 slack), order-independent.
    # KNOWN ISSUE: radii span rad[0]..rad[-1] (first and last sample), not min..max,
    # so the radius range depends on sample order.
    rng = check_random_state(0)
    X = 2 * rng.random_sample(size=(n_samples, n_features)) - 1
    query_pt = np.zeros(n_features, dtype=float)

    eps = 1e-15  # roundoff error can cause test to fail
    tree = Cls(X, leaf_size=5)
    rad = np.sqrt(((X - query_pt) ** 2).sum(1))

    for r in np.linspace(rad[0], rad[-1], 100):
        ind = tree.query_radius([query_pt], r + eps)[0]
        i = np.where(rad <= r + eps)[0]

        ind.sort()
        i.sort()

        assert_array_almost_equal(i, ind)


@pytest.mark.parametrize("Cls", [KDTree, BallTree])
def test_neighbor_tree_query_radius_distance(Cls, n_samples=100, n_features=10):
    # [FUNCTION DOC]
    # Checks distances returned by query_radius with return_distance equal recomputed
    # Euclidean distances for the returned indices.
    rng = check_random_state(0)
    X = 2 * rng.random_sample(size=(n_samples, n_features)) - 1
    query_pt = np.zeros(n_features, dtype=float)

    eps = 1e-15  # roundoff error can cause test to fail
    tree = Cls(X, leaf_size=5)
    rad = np.sqrt(((X - query_pt) ** 2).sum(1))

    for r in np.linspace(rad[0], rad[-1], 100):
        ind, dist = tree.query_radius([query_pt], r + eps, return_distance=True)

        ind = ind[0]
        dist = dist[0]

        d = np.sqrt(((query_pt - X[ind]) ** 2).sum(1))

        assert_array_almost_equal(d, dist)


@pytest.mark.parametrize("Cls", [KDTree, BallTree])
@pytest.mark.parametrize("dualtree", (True, False))
def test_neighbor_tree_two_point(Cls, dualtree, n_samples=100, n_features=3):
    # [FUNCTION DOC]
    # Checks two_point_correlation (single and dual tree) counts pairs within each
    # radius exactly as brute force does.
    rng = check_random_state(0)
    X = rng.random_sample((n_samples, n_features))
    Y = rng.random_sample((n_samples, n_features))
    r = np.linspace(0, 1, 10)
    tree = Cls(X, leaf_size=10)

    D = DistanceMetric.get_metric("euclidean").pairwise(Y, X)
    counts_true = [(D <= ri).sum() for ri in r]

    counts = tree.two_point_correlation(Y, r=r, dualtree=dualtree)
    assert_array_almost_equal(counts, counts_true)


@pytest.mark.parametrize("NeighborsHeap", [NeighborsHeapBT, NeighborsHeapKDT])
def test_neighbors_heap(NeighborsHeap, n_pts=5, n_nbrs=10):
    # [FUNCTION DOC]
    # Pushes 2*n_nbrs random (distance, index) pairs per row and checks the heap keeps
    # the n_nbrs smallest, returned sorted, for both trees' heap classes.
    heap = NeighborsHeap(n_pts, n_nbrs)
    rng = check_random_state(0)

    for row in range(n_pts):
        d_in = rng.random_sample(2 * n_nbrs).astype(np.float64, copy=False)
        i_in = np.arange(2 * n_nbrs, dtype=np.intp)
        for d, i in zip(d_in, i_in):
            heap.push(row, d, i)

        ind = np.argsort(d_in)
        d_in = d_in[ind]
        i_in = i_in[ind]

        d_heap, i_heap = heap.get_arrays(sort=True)

        assert_array_almost_equal(d_in[:n_nbrs], d_heap[row])
        assert_array_almost_equal(i_in[:n_nbrs], i_heap[row])


@pytest.mark.parametrize("nodeheap_sort", [nodeheap_sort_bt, nodeheap_sort_kdt])
def test_node_heap(nodeheap_sort, n_nodes=50):
    # [FUNCTION DOC]
    # Checks nodeheap_sort returns the same sorted values and permutation as argsort.
    rng = check_random_state(0)
    vals = rng.random_sample(n_nodes).astype(np.float64, copy=False)

    i1 = np.argsort(vals)
    vals2, i2 = nodeheap_sort(vals)

    assert_array_almost_equal(i1, i2)
    assert_array_almost_equal(vals[i1], vals2)


@pytest.mark.parametrize(
    "simultaneous_sort", [simultaneous_sort_bt, simultaneous_sort_kdt]
)
def test_simultaneous_sort(simultaneous_sort, n_rows=10, n_pts=201):
    # [FUNCTION DOC]
    # Checks simultaneous_sort sorts each row of dist in place and permutes ind
    # consistently, matching a numpy argsort-based reference.
    rng = check_random_state(0)
    dist = rng.random_sample((n_rows, n_pts)).astype(np.float64, copy=False)
    ind = (np.arange(n_pts) + np.zeros((n_rows, 1))).astype(np.intp, copy=False)

    dist2 = dist.copy()
    ind2 = ind.copy()

    # simultaneous sort rows using function
    simultaneous_sort(dist, ind)

    # simultaneous sort rows using numpy
    i = np.argsort(dist2, axis=1)
    row_ind = np.arange(n_rows)[:, None]
    dist2 = dist2[row_ind, i]
    ind2 = ind2[row_ind, i]

    assert_array_almost_equal(dist, dist2)
    assert_array_almost_equal(ind, ind2)


@pytest.mark.parametrize("Cls", [KDTree, BallTree])
def test_gaussian_kde(Cls, n_samples=1000):
    # [FUNCTION DOC]
    # Checks 1D gaussian kernel_density divided by n matches scipy gaussian_kde with
    # equivalent bandwidth to 3 decimals for h in 0.01, 0.1, 1.
    # Compare gaussian KDE results to scipy.stats.gaussian_kde
    from scipy.stats import gaussian_kde

    rng = check_random_state(0)
    x_in = rng.normal(0, 1, n_samples)
    x_out = np.linspace(-5, 5, 30)

    for h in [0.01, 0.1, 1]:
        tree = Cls(x_in[:, None])
        gkde = gaussian_kde(x_in, bw_method=h / np.std(x_in))

        dens_tree = tree.kernel_density(x_out[:, None], h) / n_samples
        dens_gkde = gkde.evaluate(x_out)

        assert_array_almost_equal(dens_tree, dens_gkde, decimal=3)


@pytest.mark.parametrize(
    "Cls, metric",
    itertools.chain(
        [(KDTree, metric) for metric in KD_TREE_METRICS],
        [(BallTree, metric) for metric in BALL_TREE_METRICS],
    ),
)
@pytest.mark.parametrize("k", (1, 3, 5))
@pytest.mark.parametrize("dualtree", (True, False))
@pytest.mark.parametrize("breadth_first", (True, False))
def test_nn_tree_query(Cls, metric, k, dualtree, breadth_first):
    # [FUNCTION DOC]
    # Checks query distances (not indices, due to ties) match brute force for each
    # tree/metric pair, k in 1, 3, 5, dualtree and breadth_first options.
    rng = check_random_state(0)
    X = rng.random_sample((40, DIMENSION))
    Y = rng.random_sample((10, DIMENSION))

    kwargs = METRICS[metric]

    kdt = Cls(X, leaf_size=1, metric=metric, **kwargs)
    dist1, ind1 = kdt.query(Y, k, dualtree=dualtree, breadth_first=breadth_first)
    dist2, ind2 = brute_force_neighbors(X, Y, k, metric, **kwargs)

    # don't check indices here: if there are any duplicate distances,
    # the indices may not match.  Distances should not have this problem.
    assert_array_almost_equal(dist1, dist2)


@pytest.mark.parametrize(
    "Cls, metric",
    [(KDTree, "euclidean"), (BallTree, "euclidean"), (BallTree, dist_func)],
)
@pytest.mark.parametrize("protocol", (0, 1, 2))
def test_pickle(Cls, metric, protocol):
    # [FUNCTION DOC]
    # Checks a tree (euclidean or callable metric) gives identical query results and
    # type after pickling with protocols 0-2.
    # KNOWN ISSUE: variables are named ind/dist but query returns (dist, ind); the
    # comparison is still valid.
    rng = check_random_state(0)
    X = rng.random_sample((10, 3))

    if hasattr(metric, "__call__"):
        kwargs = {"p": 2}
    else:
        kwargs = {}

    tree1 = Cls(X, leaf_size=1, metric=metric, **kwargs)

    ind1, dist1 = tree1.query(X)

    s = pickle.dumps(tree1, protocol=protocol)
    tree2 = pickle.loads(s)

    ind2, dist2 = tree2.query(X)

    assert_array_almost_equal(ind1, ind2)
    assert_array_almost_equal(dist1, dist2)

    assert isinstance(tree2, Cls)
