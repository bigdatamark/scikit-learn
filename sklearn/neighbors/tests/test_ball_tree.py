# [MODULE DOC]
# Tests for BallTree (and its float64/float32 variants BallTree64/BallTree32):
# correctness of queries against a brute-force reference for several metrics, input
# validation, custom metric errors, and float32 vs float64 numerical consistency of
# query, query_radius, kernel_density and two_point_correlation.
# Module-level setup builds a random symmetric matrix V_mahalanobis (unused here) and
# metric lists used for parametrization.
import itertools

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_almost_equal, assert_equal

from sklearn.neighbors._ball_tree import BallTree, BallTree32, BallTree64
from sklearn.utils import check_random_state
from sklearn.utils._testing import _convert_container
from sklearn.utils.validation import check_array

rng = np.random.RandomState(10)
V_mahalanobis = rng.rand(3, 3)
V_mahalanobis = np.dot(V_mahalanobis, V_mahalanobis.T)

DIMENSION = 3

METRICS = {
    "euclidean": {},
    "manhattan": {},
    "minkowski": dict(p=3),
    "chebyshev": {},
}

DISCRETE_METRICS = ["hamming", "canberra", "braycurtis"]

BOOLEAN_METRICS = [
    "jaccard",
    "dice",
    "rogerstanimoto",
    "russellrao",
    "sokalmichener",
    "sokalsneath",
]

BALL_TREE_CLASSES = [
    BallTree64,
    BallTree32,
]


def brute_force_neighbors(X, Y, k, metric, **kwargs):
    # [FUNCTION DOC]
    # HELPER: Reference k-NN: computes the full distance matrix from each row of Y to
    # each
    # row of X with DistanceMetric, then takes the k smallest per row.
    # Inputs: X - (n_fit, d); Y - (n_query, d); k - int; metric - str; kwargs - metric
    # params.
    # Outputs: (dist, ind), each (n_query, k).
    from sklearn.metrics import DistanceMetric

    X, Y = check_array(X), check_array(Y)
    D = DistanceMetric.get_metric(metric, **kwargs).pairwise(Y, X)
    ind = np.argsort(D, axis=1)[:, :k]
    dist = D[np.arange(Y.shape[0])[:, None], ind]
    return dist, ind


def test_BallTree_is_BallTree64_subclass():
    # [FUNCTION DOC]
    # Verifies that the public BallTree alias is a subclass of BallTree64.
    assert issubclass(BallTree, BallTree64)


@pytest.mark.parametrize("metric", itertools.chain(BOOLEAN_METRICS, DISCRETE_METRICS))
@pytest.mark.parametrize("array_type", ["list", "array"])
@pytest.mark.parametrize("BallTreeImplementation", BALL_TREE_CLASSES)
def test_ball_tree_query_metrics(metric, array_type, BallTreeImplementation):
    # [FUNCTION DOC]
    # For boolean and discrete metrics, with list or array inputs and both tree dtypes,
    # builds a tree on rounded random data and checks query distances for k=5 match the
    # brute-force reference.
    # KNOWN ISSUE: only distances are compared; the returned indices are not checked.
    rng = check_random_state(0)
    if metric in BOOLEAN_METRICS:
        X = rng.random_sample((40, 10)).round(0)
        Y = rng.random_sample((10, 10)).round(0)
    elif metric in DISCRETE_METRICS:
        X = (4 * rng.random_sample((40, 10))).round(0)
        Y = (4 * rng.random_sample((10, 10))).round(0)
    X = _convert_container(X, array_type)
    Y = _convert_container(Y, array_type)

    k = 5

    bt = BallTreeImplementation(X, leaf_size=1, metric=metric)
    dist1, ind1 = bt.query(Y, k)
    dist2, ind2 = brute_force_neighbors(X, Y, k, metric)
    assert_array_almost_equal(dist1, dist2)


@pytest.mark.parametrize(
    "BallTreeImplementation, decimal_tol", zip(BALL_TREE_CLASSES, [6, 5])
)
def test_query_haversine(BallTreeImplementation, decimal_tol):
    # [FUNCTION DOC]
    # Checks BallTree queries with the haversine metric on random angle pairs match the
    # brute-force reference in both distances and indices (looser tolerance for
    # float32).
    rng = check_random_state(0)
    X = 2 * np.pi * rng.random_sample((40, 2))
    bt = BallTreeImplementation(X, leaf_size=1, metric="haversine")
    dist1, ind1 = bt.query(X, k=5)
    dist2, ind2 = brute_force_neighbors(X, X, k=5, metric="haversine")

    assert_array_almost_equal(dist1, dist2, decimal=decimal_tol)
    assert_array_almost_equal(ind1, ind2)


@pytest.mark.parametrize("BallTreeImplementation", BALL_TREE_CLASSES)
def test_array_object_type(BallTreeImplementation):
    # [FUNCTION DOC]
    # Checks that a ragged object-dtype array is rejected with a ValueError at
    # construction.
    """Check that we do not accept object dtype array."""
    X = np.array([(1, 2, 3), (2, 5), (5, 5, 1, 2)], dtype=object)
    with pytest.raises(ValueError, match="setting an array element with a sequence"):
        BallTreeImplementation(X)


@pytest.mark.parametrize("BallTreeImplementation", BALL_TREE_CLASSES)
def test_bad_pyfunc_metric(BallTreeImplementation):
    # [FUNCTION DOC]
    # Checks that a callable metric returning a non-float raises TypeError with the
    # expected message, and that a one-argument callable raises TypeError.
    def wrong_returned_value(x, y):
        # [FUNCTION DOC]
        # HELPER: Invalid metric that returns a string instead of a float.
        return "1"

    def one_arg_func(x):
        # [FUNCTION DOC]
        # HELPER: Invalid metric accepting only one argument (never actually executed).
        return 1.0  # pragma: no cover

    X = np.ones((5, 2))
    msg = "Custom distance function must accept two vectors and return a float."
    with pytest.raises(TypeError, match=msg):
        BallTreeImplementation(X, metric=wrong_returned_value)

    msg = "takes 1 positional argument but 2 were given"
    with pytest.raises(TypeError, match=msg):
        BallTreeImplementation(X, metric=one_arg_func)


@pytest.mark.parametrize("metric", itertools.chain(METRICS, BOOLEAN_METRICS))
def test_ball_tree_numerical_consistency(global_random_seed, metric):
    # [FUNCTION DOC]
    # For each metric, builds float64 and float32 trees on the same data (50 features)
    # and
    # checks query distances are close and indices equal with correct dtypes, and that
    # query_radius indices (with and without distances) agree, distances close and
    # dtypes
    # preserved.
    # Results on float64 and float32 versions of a dataset must be
    # numerically close.
    X_64, X_32, Y_64, Y_32 = get_dataset_for_binary_tree(
        random_seed=global_random_seed, features=50
    )

    metric_params = METRICS.get(metric, {})
    bt_64 = BallTree64(X_64, leaf_size=1, metric=metric, **metric_params)
    bt_32 = BallTree32(X_32, leaf_size=1, metric=metric, **metric_params)

    # Test consistency with respect to the `query` method
    k = 5
    dist_64, ind_64 = bt_64.query(Y_64, k=k)
    dist_32, ind_32 = bt_32.query(Y_32, k=k)
    assert_allclose(dist_64, dist_32, rtol=1e-5)
    assert_equal(ind_64, ind_32)
    assert dist_64.dtype == np.float64
    assert dist_32.dtype == np.float32

    # Test consistency with respect to the `query_radius` method
    r = 2.38
    ind_64 = bt_64.query_radius(Y_64, r=r)
    ind_32 = bt_32.query_radius(Y_32, r=r)
    for _ind64, _ind32 in zip(ind_64, ind_32):
        assert_equal(_ind64, _ind32)

    # Test consistency with respect to the `query_radius` method
    # with return distances being true
    ind_64, dist_64 = bt_64.query_radius(Y_64, r=r, return_distance=True)
    ind_32, dist_32 = bt_32.query_radius(Y_32, r=r, return_distance=True)
    for _ind64, _ind32, _dist_64, _dist_32 in zip(ind_64, ind_32, dist_64, dist_32):
        assert_equal(_ind64, _ind32)
        assert_allclose(_dist_64, _dist_32, rtol=1e-5)
        assert _dist_64.dtype == np.float64
        assert _dist_32.dtype == np.float32


@pytest.mark.parametrize("metric", itertools.chain(METRICS, BOOLEAN_METRICS))
def test_kernel_density_numerical_consistency(global_random_seed, metric):
    # [FUNCTION DOC]
    # Checks gaussian kernel_density results agree between float64 and float32 trees for
    # each metric, with output dtypes matching the tree dtype.
    # Test consistency with respect to the `kernel_density` method
    X_64, X_32, Y_64, Y_32 = get_dataset_for_binary_tree(random_seed=global_random_seed)

    metric_params = METRICS.get(metric, {})
    bt_64 = BallTree64(X_64, leaf_size=1, metric=metric, **metric_params)
    bt_32 = BallTree32(X_32, leaf_size=1, metric=metric, **metric_params)

    kernel = "gaussian"
    h = 0.1
    density64 = bt_64.kernel_density(Y_64, h=h, kernel=kernel, breadth_first=True)
    density32 = bt_32.kernel_density(Y_32, h=h, kernel=kernel, breadth_first=True)
    assert_allclose(density64, density32, rtol=1e-5)
    assert density64.dtype == np.float64
    assert density32.dtype == np.float32


def test_two_point_correlation_numerical_consistency(global_random_seed):
    # [FUNCTION DOC]
    # Checks dual-tree two_point_correlation counts over 10 radii agree between float64
    # and
    # float32 trees.
    # Test consistency with respect to the `two_point_correlation` method
    X_64, X_32, Y_64, Y_32 = get_dataset_for_binary_tree(random_seed=global_random_seed)

    bt_64 = BallTree64(X_64, leaf_size=10)
    bt_32 = BallTree32(X_32, leaf_size=10)

    r = np.linspace(0, 1, 10)

    counts_64 = bt_64.two_point_correlation(Y_64, r=r, dualtree=True)
    counts_32 = bt_32.two_point_correlation(Y_32, r=r, dualtree=True)
    assert_allclose(counts_64, counts_32)


def get_dataset_for_binary_tree(random_seed, features=3):
    # [FUNCTION DOC]
    # HELPER: Generates seeded random X (100, features) and Y (5, features) uniform data
    # and
    # returns them in float64 and float32: (X_64, X_32, Y_64, Y_32).
    rng = np.random.RandomState(random_seed)
    _X = rng.rand(100, features)
    _Y = rng.rand(5, features)

    X_64 = _X.astype(dtype=np.float64, copy=False)
    Y_64 = _Y.astype(dtype=np.float64, copy=False)

    X_32 = _X.astype(dtype=np.float32, copy=False)
    Y_32 = _Y.astype(dtype=np.float32, copy=False)

    return X_64, X_32, Y_64, Y_32
