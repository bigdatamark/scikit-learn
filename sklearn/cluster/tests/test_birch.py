# [MODULE DOC]
# Tests for the Birch clustering estimator: CF-tree bookkeeping, partial_fit
# equivalence, prediction, n_clusters handling, sparse input, branching factor and
# threshold invariants, dtype preservation and feature names.
"""
Tests for the birch clustering algorithm.
"""

import numpy as np
import pytest

from sklearn.cluster import AgglomerativeClustering, Birch
from sklearn.cluster.tests.common import generate_clustered_data
from sklearn.datasets import make_blobs
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import pairwise_distances_argmin, v_measure_score
from sklearn.utils._testing import assert_allclose, assert_array_equal
from sklearn.utils.fixes import CSR_CONTAINERS


def test_n_samples_leaves_roots(global_random_seed, global_dtype):
    # [FUNCTION DOC]
    # Verifies the n_samples_ counts of the root subclusters and of all leaf subclusters
    # each sum to the number of fitted samples (10 blobs points, global_dtype).
    # Sanity check for the number of samples in leaves and roots
    X, y = make_blobs(n_samples=10, random_state=global_random_seed)
    X = X.astype(global_dtype, copy=False)
    brc = Birch()
    brc.fit(X)
    n_samples_root = sum([sc.n_samples_ for sc in brc.root_.subclusters_])
    n_samples_leaves = sum(
        [sc.n_samples_ for leaf in brc._get_leaves() for sc in leaf.subclusters_]
    )
    assert n_samples_leaves == X.shape[0]
    assert n_samples_root == X.shape[0]


def test_partial_fit(global_random_seed, global_dtype):
    # [FUNCTION DOC]
    # Verifies two partial_fit calls on halves of the data (n_clusters=None) give the same
    # subcluster_centers_ as a single fit, and that a subsequent partial_fit(None) after
    # setting n_clusters=3 only runs the global clustering step and reproduces the
    # subcluster_labels_ of the full fit.
    # Test that fit is equivalent to calling partial_fit multiple times
    X, y = make_blobs(n_samples=100, random_state=global_random_seed)
    X = X.astype(global_dtype, copy=False)
    brc = Birch(n_clusters=3)
    brc.fit(X)
    brc_partial = Birch(n_clusters=None)
    brc_partial.partial_fit(X[:50])
    brc_partial.partial_fit(X[50:])
    assert_allclose(brc_partial.subcluster_centers_, brc.subcluster_centers_)

    # Test that same global labels are obtained after calling partial_fit
    # with None
    brc_partial.set_params(n_clusters=3)
    brc_partial.partial_fit(None)
    assert_array_equal(brc_partial.subcluster_labels_, brc.subcluster_labels_)


def test_birch_predict(global_random_seed, global_dtype):
    # [FUNCTION DOC]
    # Verifies on shuffled 3-cluster data that labels_ equals predict on the training data,
    # that subcluster_centers_ keeps the input dtype, and that labels match (v-measure 1)
    # assigning each sample the global label of its nearest subcluster center.
    # Test the predict method predicts the nearest centroid.
    rng = np.random.RandomState(global_random_seed)
    X = generate_clustered_data(n_clusters=3, n_features=3, n_samples_per_cluster=10)
    X = X.astype(global_dtype, copy=False)

    # n_samples * n_samples_per_cluster
    shuffle_indices = np.arange(30)
    rng.shuffle(shuffle_indices)
    X_shuffle = X[shuffle_indices, :]
    brc = Birch(n_clusters=4, threshold=1.0)
    brc.fit(X_shuffle)

    # Birch must preserve inputs' dtype
    assert brc.subcluster_centers_.dtype == global_dtype

    assert_array_equal(brc.labels_, brc.predict(X_shuffle))
    centroids = brc.subcluster_centers_
    nearest_centroid = brc.subcluster_labels_[
        pairwise_distances_argmin(X_shuffle, centroids)
    ]
    assert_allclose(v_measure_score(nearest_centroid, brc.labels_), 1.0)


def test_n_clusters(global_random_seed, global_dtype):
    # [FUNCTION DOC]
    # Verifies n_clusters=10 gives more than 10 subclusters but exactly 10 labels; passing
    # an AgglomerativeClustering(n_clusters=10) instance gives identical results; and a
    # huge threshold (too few subclusters for the default 3 clusters) emits
    # ConvergenceWarning.
    # Test that n_clusters param works properly
    X, y = make_blobs(n_samples=100, centers=10, random_state=global_random_seed)
    X = X.astype(global_dtype, copy=False)
    brc1 = Birch(n_clusters=10)
    brc1.fit(X)
    assert len(brc1.subcluster_centers_) > 10
    assert len(np.unique(brc1.labels_)) == 10

    # Test that n_clusters = Agglomerative Clustering gives
    # the same results.
    gc = AgglomerativeClustering(n_clusters=10)
    brc2 = Birch(n_clusters=gc)
    brc2.fit(X)
    assert_array_equal(brc1.subcluster_labels_, brc2.subcluster_labels_)
    assert_array_equal(brc1.labels_, brc2.labels_)

    # Test that a small number of clusters raises a warning.
    brc4 = Birch(threshold=10000.0)
    with pytest.warns(ConvergenceWarning):
        brc4.fit(X)


@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_sparse_X(global_random_seed, global_dtype, csr_container):
    # [FUNCTION DOC]
    # Verifies a CSR input yields the same labels and subcluster centers as dense input
    # and preserves dtype. Inputs: csr_container - parametrized CSR class.
    # Test that sparse and dense data give same results
    X, y = make_blobs(n_samples=100, centers=10, random_state=global_random_seed)
    X = X.astype(global_dtype, copy=False)
    brc = Birch(n_clusters=10)
    brc.fit(X)

    csr = csr_container(X)
    brc_sparse = Birch(n_clusters=10)
    brc_sparse.fit(csr)

    # Birch must preserve inputs' dtype
    assert brc_sparse.subcluster_centers_.dtype == global_dtype

    assert_array_equal(brc.labels_, brc_sparse.labels_)
    assert_allclose(brc.subcluster_centers_, brc_sparse.subcluster_centers_)


def test_partial_fit_second_call_error_checks():
    # [FUNCTION DOC]
    # Verifies a second partial_fit with a different number of features raises ValueError
    # ("X has 1 features, but Birch is expecting 2 features").
    # second partial fit calls will error when n_features is not consistent
    # with the first call
    X, y = make_blobs(n_samples=100)
    brc = Birch(n_clusters=3)
    brc.partial_fit(X, y)

    msg = "X has 1 features, but Birch is expecting 2 features"
    with pytest.raises(ValueError, match=msg):
        brc.partial_fit(X[:, [0]], y)


def check_branching_factor(node, branching_factor):
    # [FUNCTION DOC]
    # HELPER: Recursively asserts that a CF node and all its descendants hold at most
    # branching_factor subclusters. Inputs: node - _CFNode; branching_factor - int.
    # Raises AssertionError on violation.
    subclusters = node.subclusters_
    assert branching_factor >= len(subclusters)
    for cluster in subclusters:
        if cluster.child_:
            check_branching_factor(cluster.child_, branching_factor)


def test_branching_factor(global_random_seed, global_dtype):
    # [FUNCTION DOC]
    # Verifies with a tiny threshold (many subclusters) that every node respects
    # branching_factor=9, both with n_clusters=None and n_clusters=3.
    # Test that nodes have at max branching_factor number of subclusters
    X, y = make_blobs(random_state=global_random_seed)
    X = X.astype(global_dtype, copy=False)
    branching_factor = 9

    # Purposefully set a low threshold to maximize the subclusters.
    brc = Birch(n_clusters=None, branching_factor=branching_factor, threshold=0.01)
    brc.fit(X)
    check_branching_factor(brc.root_, branching_factor)
    brc = Birch(n_clusters=3, branching_factor=branching_factor, threshold=0.01)
    brc.fit(X)
    check_branching_factor(brc.root_, branching_factor)


def check_threshold(birch_instance, threshold):
    # [FUNCTION DOC]
    # HELPER: Walks the leaf linked list starting after dummy_leaf_ and asserts every leaf
    # subcluster radius is at most threshold. Inputs: fitted Birch; threshold - float.
    # Raises AssertionError on violation.
    """Use the leaf linked list for traversal"""
    current_leaf = birch_instance.dummy_leaf_.next_leaf_
    while current_leaf:
        subclusters = current_leaf.subclusters_
        for sc in subclusters:
            assert threshold >= sc.radius
        current_leaf = current_leaf.next_leaf_


def test_threshold(global_random_seed, global_dtype):
    # [FUNCTION DOC]
    # Verifies every leaf subcluster radius is at most the threshold, for thresholds 0.5
    # and 5.0.
    # KNOWN ISSUE: the inline comment says "threshold lesser than radius", the inverse of
    # what is actually checked (radius not exceeding threshold).
    # Test that the leaf subclusters have a threshold lesser than radius
    X, y = make_blobs(n_samples=80, centers=4, random_state=global_random_seed)
    X = X.astype(global_dtype, copy=False)
    brc = Birch(threshold=0.5, n_clusters=None)
    brc.fit(X)
    check_threshold(brc, 0.5)

    brc = Birch(threshold=5.0, n_clusters=None)
    brc.fit(X)
    check_threshold(brc, 5.0)


def test_birch_n_clusters_long_int():
    # [FUNCTION DOC]
    # Non-regression for #16484: fitting with n_clusters given as np.int64 does not raise.
    # Check that birch supports n_clusters with np.int64 dtype, for instance
    # coming from np.arange. #16484
    X, _ = make_blobs(random_state=0)
    n_clusters = np.int64(5)
    Birch(n_clusters=n_clusters).fit(X)


def test_feature_names_out():
    # [FUNCTION DOC]
    # Verifies get_feature_names_out returns "birch0", ... one per subcluster center.
    """Check `get_feature_names_out` for `Birch`."""
    X, _ = make_blobs(n_samples=80, n_features=4, random_state=0)
    brc = Birch(n_clusters=4)
    brc.fit(X)
    n_clusters = brc.subcluster_centers_.shape[0]

    names_out = brc.get_feature_names_out()
    assert_array_equal([f"birch{i}" for i in range(n_clusters)], names_out)


def test_transform_match_across_dtypes(global_random_seed):
    # [FUNCTION DOC]
    # Verifies fit_transform on float64 and float32 versions of the data give the same
    # distances up to atol 1e-6.
    X, _ = make_blobs(n_samples=80, n_features=4, random_state=global_random_seed)
    brc = Birch(n_clusters=4, threshold=1.1)
    Y_64 = brc.fit_transform(X)
    Y_32 = brc.fit_transform(X.astype(np.float32))

    assert_allclose(Y_64, Y_32, atol=1e-6)


def test_subcluster_dtype(global_dtype):
    # [FUNCTION DOC]
    # Verifies subcluster_centers_ has the input dtype (global_dtype).
    X = make_blobs(n_samples=80, n_features=4, random_state=0)[0].astype(
        global_dtype, copy=False
    )
    brc = Birch(n_clusters=4)
    assert brc.fit(X).subcluster_centers_.dtype == global_dtype


def test_both_subclusters_updated():
    # [FUNCTION DOC]
    # Non-regression for #23269: fitting float32 data with many duplicated points and a
    # tiny threshold (forcing node splits) completes without error.
    """Check that both subclusters are updated when a node a split, even when there are
    duplicated data points. Non-regression test for #23269.
    """

    X = np.array(
        [
            [-2.6192791, -1.5053215],
            [-2.9993038, -1.6863596],
            [-2.3724914, -1.3438171],
            [-2.336792, -1.3417323],
            [-2.4089134, -1.3290224],
            [-2.3724914, -1.3438171],
            [-3.364009, -1.8846745],
            [-2.3724914, -1.3438171],
            [-2.617677, -1.5003285],
            [-2.2960556, -1.3260119],
            [-2.3724914, -1.3438171],
            [-2.5459878, -1.4533926],
            [-2.25979, -1.3003055],
            [-2.4089134, -1.3290224],
            [-2.3724914, -1.3438171],
            [-2.4089134, -1.3290224],
            [-2.5459878, -1.4533926],
            [-2.3724914, -1.3438171],
            [-2.9720619, -1.7058647],
            [-2.336792, -1.3417323],
            [-2.3724914, -1.3438171],
        ],
        dtype=np.float32,
    )

    # no error
    Birch(branching_factor=5, threshold=1e-5, n_clusters=None).fit(X)
