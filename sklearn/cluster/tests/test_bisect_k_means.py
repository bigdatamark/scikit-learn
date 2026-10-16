# [MODULE DOC]
# Tests for BisectingKMeans: correctness on toy data, sparse/dense and dtype
# equivalence, label ranges, single cluster, predict consistency and non-regression
# cases (empty bisections, single feature, callable init).
import numpy as np
import pytest

from sklearn.cluster import BisectingKMeans
from sklearn.metrics import v_measure_score
from sklearn.utils._testing import assert_allclose, assert_array_equal
from sklearn.utils.fixes import CSR_CONTAINERS


@pytest.mark.parametrize("bisecting_strategy", ["biggest_inertia", "largest_cluster"])
@pytest.mark.parametrize("init", ["k-means++", "random"])
def test_three_clusters(bisecting_strategy, init):
    # [FUNCTION DOC]
    # Verifies on 9 hand-placed 2D points forming 3 groups that the fitted centers equal
    # the expected group means (order-insensitive) and labels match the expected partition
    # (v-measure 1). Inputs: bisecting_strategy, init - parametrized options.
    """Tries to perform bisect k-means for three clusters to check
    if splitting data is performed correctly.
    """
    X = np.array(
        [[1, 1], [10, 1], [3, 1], [10, 0], [2, 1], [10, 2], [10, 8], [10, 9], [10, 10]]
    )
    bisect_means = BisectingKMeans(
        n_clusters=3,
        random_state=0,
        bisecting_strategy=bisecting_strategy,
        init=init,
    )
    bisect_means.fit(X)

    expected_centers = [[2, 1], [10, 1], [10, 9]]
    expected_labels = [0, 1, 0, 1, 0, 1, 2, 2, 2]

    assert_allclose(
        sorted(expected_centers), sorted(bisect_means.cluster_centers_.tolist())
    )
    assert_allclose(v_measure_score(expected_labels, bisect_means.labels_), 1.0)


@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_sparse(csr_container):
    # [FUNCTION DOC]
    # Verifies cluster_centers_ fitted on a thresholded random matrix are identical (atol
    # 1e-8) for CSR and dense input. Inputs: csr_container - parametrized CSR class.
    """Test Bisecting K-Means with sparse data.

    Checks if labels and centers are the same between dense and sparse.
    """

    rng = np.random.RandomState(0)

    X = rng.rand(20, 2)
    X[X < 0.8] = 0
    X_csr = csr_container(X)

    bisect_means = BisectingKMeans(n_clusters=3, random_state=0)

    bisect_means.fit(X_csr)
    sparse_centers = bisect_means.cluster_centers_

    bisect_means.fit(X)
    normal_centers = bisect_means.cluster_centers_

    # Check if results is the same for dense and sparse data
    assert_allclose(normal_centers, sparse_centers, atol=1e-8)


@pytest.mark.parametrize("n_clusters", [4, 5])
def test_n_clusters(n_clusters):
    # [FUNCTION DOC]
    # Verifies the set of labels after fitting is exactly 0..n_clusters-1 (n_clusters 4
    # or 5).
    """Test if resulting labels are in range [0, n_clusters - 1]."""

    rng = np.random.RandomState(0)
    X = rng.rand(10, 2)

    bisect_means = BisectingKMeans(n_clusters=n_clusters, random_state=0)
    bisect_means.fit(X)

    assert_array_equal(np.unique(bisect_means.labels_), np.arange(n_clusters))


def test_one_cluster():
    # [FUNCTION DOC]
    # Verifies n_clusters=1 labels everything 0 in fit and predict and the single center is
    # the data mean.
    """Test single cluster."""

    X = np.array([[1, 2], [10, 2], [10, 8]])

    bisect_means = BisectingKMeans(n_clusters=1, random_state=0).fit(X)

    # All labels from fit or predict should be equal 0
    assert all(bisect_means.labels_ == 0)
    assert all(bisect_means.predict(X) == 0)

    assert_allclose(bisect_means.cluster_centers_, X.mean(axis=0).reshape(1, -1))


@pytest.mark.parametrize("csr_container", CSR_CONTAINERS + [None])
def test_fit_predict(csr_container):
    # [FUNCTION DOC]
    # Verifies labels_ from fit equal predict on the training data, for dense and
    # thresholded CSR input. Inputs: csr_container - CSR class or None for dense.
    """Check if labels from fit(X) method are same as from fit(X).predict(X)."""
    rng = np.random.RandomState(0)

    X = rng.rand(10, 2)

    if csr_container is not None:
        X[X < 0.8] = 0
        X = csr_container(X)

    bisect_means = BisectingKMeans(n_clusters=3, random_state=0)
    bisect_means.fit(X)

    assert_array_equal(bisect_means.labels_, bisect_means.predict(X))


@pytest.mark.parametrize("csr_container", CSR_CONTAINERS + [None])
def test_dtype_preserved(csr_container, global_dtype):
    # [FUNCTION DOC]
    # Verifies cluster_centers_ keeps the input dtype (global_dtype), dense or CSR.
    """Check that centers dtype is the same as input data dtype."""
    rng = np.random.RandomState(0)
    X = rng.rand(10, 2).astype(global_dtype, copy=False)

    if csr_container is not None:
        X[X < 0.8] = 0
        X = csr_container(X)

    km = BisectingKMeans(n_clusters=3, random_state=0)
    km.fit(X)

    assert km.cluster_centers_.dtype == global_dtype


@pytest.mark.parametrize("csr_container", CSR_CONTAINERS + [None])
def test_float32_float64_equivalence(csr_container):
    # [FUNCTION DOC]
    # Verifies float32 and float64 fits of the same data give the same centers and labels,
    # dense or CSR.
    """Check that the results are the same between float32 and float64."""
    rng = np.random.RandomState(0)
    X = rng.rand(10, 2)

    if csr_container is not None:
        X[X < 0.8] = 0
        X = csr_container(X)

    km64 = BisectingKMeans(n_clusters=3, random_state=0).fit(X)
    km32 = BisectingKMeans(n_clusters=3, random_state=0).fit(X.astype(np.float32))

    assert_allclose(km32.cluster_centers_, km64.cluster_centers_)
    assert_array_equal(km32.labels_, km64.labels_)


@pytest.mark.parametrize("algorithm", ("lloyd", "elkan"))
def test_no_crash_on_empty_bisections(algorithm):
    # [FUNCTION DOC]
    # Non-regression for #27081: predicting on data scaled far outside the training range
    # (which makes some inner bisection masks empty) does not raise a division error and
    # returns labels within 0..9. Inputs: algorithm - "lloyd" or "elkan".
    # Side effects: model fit is unseeded (no random_state), so it is nondeterministic.
    # Non-regression test for:
    # https://github.com/scikit-learn/scikit-learn/issues/27081
    rng = np.random.RandomState(0)
    X_train = rng.rand(3000, 10)
    bkm = BisectingKMeans(n_clusters=10, algorithm=algorithm).fit(X_train)

    # predict on scaled data to trigger pathologic case
    # where the inner mask leads to empty bisections.
    X_test = 50 * rng.rand(100, 10)
    labels = bkm.predict(X_test)  # should not crash with idiv by 0
    assert np.isin(np.unique(labels), np.arange(10)).all()


def test_one_feature():
    # [FUNCTION DOC]
    # Non-regression for #27236: fitting single-feature data with the biggest_inertia
    # strategy does not raise.
    # Side effects: draws from the global numpy random state (unseeded).
    # Check that no error is raised when there is only one feature
    # Non-regression test for:
    # https://github.com/scikit-learn/scikit-learn/issues/27236
    X = np.random.normal(size=(128, 1))
    BisectingKMeans(bisecting_strategy="biggest_inertia", random_state=0).fit(X)


def test_bisecting_kmeans_custom_init_validation():
    # [FUNCTION DOC]
    # Regression test for #33146: fitting with a callable init that returns the first
    # n_clusters rows must not fail center-shape validation. Only checks that fit does not
    # raise; there is no assertion.
    """Test that BisectingKMeans validates center shape correctly with callable init.

    Regression test for issue #33146
    """
    rng = np.random.RandomState(42)
    X = rng.rand(100, 2)

    def my_init(X, n_clusters, random_state):
        # [FUNCTION DOC]
        # HELPER: Callable init returning the first n_clusters rows of X as the initial
        # centers; random_state is ignored.
        return X[:n_clusters]

    bisect = BisectingKMeans(n_clusters=3, init=my_init, n_init=1, random_state=rng)
    bisect.fit(X)
