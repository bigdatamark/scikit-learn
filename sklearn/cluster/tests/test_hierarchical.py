# [MODULE DOC]
# Tests for hierarchical (agglomerative) clustering: AgglomerativeClustering,
# FeatureAgglomeration, ward_tree, linkage_tree, the private tree builders in
# _TREE_BUILDERS, _hc_cut, _fix_connectivity, and the Cython helpers (mst_linkage_core,
# max_merge, average_merge, IntFloatDict). Many tests compare with scipy.hierarchy.
"""
Several basic tests for hierarchical clustering procedures

"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import itertools
import shutil
from functools import partial
from tempfile import mkdtemp

import numpy as np
import pytest
from scipy.cluster import hierarchy
from scipy.sparse.csgraph import connected_components

from sklearn.cluster import AgglomerativeClustering, FeatureAgglomeration, ward_tree
from sklearn.cluster._agglomerative import (
    _TREE_BUILDERS,
    _fix_connectivity,
    _hc_cut,
    linkage_tree,
)
from sklearn.cluster._hierarchical_fast import (
    average_merge,
    max_merge,
    mst_linkage_core,
)
from sklearn.datasets import make_circles, make_moons
from sklearn.feature_extraction.image import grid_to_graph
from sklearn.metrics import DistanceMetric
from sklearn.metrics.cluster import adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics.pairwise import (
    PAIRED_DISTANCES,
    cosine_distances,
    manhattan_distances,
    pairwise_distances,
)
from sklearn.metrics.tests.test_dist_metrics import METRICS_DEFAULT_PARAMS
from sklearn.neighbors import kneighbors_graph
from sklearn.utils._fast_dict import IntFloatDict
from sklearn.utils._testing import (
    assert_almost_equal,
    assert_array_almost_equal,
    assert_array_equal,
    create_memmap_backed_data,
    ignore_warnings,
)
from sklearn.utils.fixes import LIL_CONTAINERS


def test_linkage_misc():
    # [FUNCTION DOC]
    # Checks linkage_tree raises ValueError for an unknown linkage and for a connectivity
    # of wrong shape; smoke-fits FeatureAgglomeration; checks precomputed cosine distances
    # and a callable affinity (manhattan_distances) give the same children as the
    # corresponding string affinities.
    # Misc tests on linkage
    rng = np.random.RandomState(42)
    X = rng.normal(size=(5, 5))

    with pytest.raises(ValueError):
        linkage_tree(X, linkage="foo")

    with pytest.raises(ValueError):
        linkage_tree(X, connectivity=np.ones((4, 4)))

    # Smoke test FeatureAgglomeration
    FeatureAgglomeration().fit(X)

    # test hierarchical clustering on a precomputed distances matrix
    dis = cosine_distances(X)

    res = linkage_tree(dis, affinity="precomputed")
    assert_array_equal(res[0], linkage_tree(X, affinity="cosine")[0])

    # test hierarchical clustering on a precomputed distances matrix
    res = linkage_tree(X, affinity=manhattan_distances)
    assert_array_equal(res[0], linkage_tree(X, affinity="manhattan")[0])


def test_structured_linkage_tree():
    # [FUNCTION DOC]
    # For each tree builder with a 10x10 grid connectivity (on 100 features) checks the
    # number of merges plus leaves equals 2 * n_leaves - 1, and ValueError for a wrongly
    # shaped connectivity and for zero samples.
    # Check that we obtain the correct solution for structured linkage trees.
    rng = np.random.RandomState(0)
    mask = np.ones([10, 10], dtype=bool)
    # Avoiding a mask with only 'True' entries
    mask[4:7, 4:7] = 0
    X = rng.randn(50, 100)
    connectivity = grid_to_graph(*mask.shape)
    for tree_builder in _TREE_BUILDERS.values():
        children, n_components, n_leaves, parent = tree_builder(
            X.T, connectivity=connectivity
        )
        n_nodes = 2 * X.shape[1] - 1
        assert len(children) + n_leaves == n_nodes
        # Check that ward_tree raises a ValueError with a connectivity matrix
        # of the wrong shape
        with pytest.raises(ValueError):
            tree_builder(X.T, connectivity=np.ones((4, 4)))
        # Check that fitting with no samples raises an error
        with pytest.raises(ValueError):
            tree_builder(X.T[:0], connectivity=connectivity)


def test_unstructured_linkage_tree():
    # [FUNCTION DOC]
    # Checks ward_tree and every tree builder without connectivity, with n_clusters=10
    # (which warns since it is ignored), produce a complete tree, on a 2D X and on its
    # first row.
    # Check that we obtain the correct solution for unstructured linkage trees.
    rng = np.random.RandomState(0)
    X = rng.randn(50, 100)
    for this_X in (X, X[0]):
        # With specified a number of clusters just for the sake of
        # raising a warning and testing the warning code
        with ignore_warnings():
            with pytest.warns(UserWarning):
                children, n_nodes, n_leaves, parent = ward_tree(this_X.T, n_clusters=10)
        n_nodes = 2 * X.shape[1] - 1
        assert len(children) + n_leaves == n_nodes

    for tree_builder in _TREE_BUILDERS.values():
        for this_X in (X, X[0]):
            with ignore_warnings():
                with pytest.warns(UserWarning):
                    children, n_nodes, n_leaves, parent = tree_builder(
                        this_X.T, n_clusters=10
                    )
            n_nodes = 2 * X.shape[1] - 1
            assert len(children) + n_leaves == n_nodes


def test_height_linkage_tree():
    # [FUNCTION DOC]
    # Checks every tree builder with grid connectivity produces a complete tree.
    # KNOWN ISSUE: despite the comment, merge heights are never checked to be sorted;
    # only the node count is asserted.
    # Check that the height of the results of linkage tree is sorted.
    rng = np.random.RandomState(0)
    mask = np.ones([10, 10], dtype=bool)
    X = rng.randn(50, 100)
    connectivity = grid_to_graph(*mask.shape)
    for linkage_func in _TREE_BUILDERS.values():
        children, n_nodes, n_leaves, parent = linkage_func(
            X.T, connectivity=connectivity
        )
        n_nodes = 2 * X.shape[1] - 1
        assert len(children) + n_leaves == n_nodes


def test_zero_cosine_linkage_tree():
    # [FUNCTION DOC]
    # Checks linkage_tree with cosine affinity raises ValueError when X has a zero row.
    # Check that zero vectors in X produce an error when
    # 'cosine' affinity is used
    X = np.array([[0, 1], [0, 0]])
    msg = "Cosine affinity cannot be used when X contains zero vectors"
    with pytest.raises(ValueError, match=msg):
        linkage_tree(X, affinity="cosine")


@pytest.mark.parametrize("n_clusters, distance_threshold", [(None, 0.5), (10, None)])
@pytest.mark.parametrize("compute_distances", [True, False])
@pytest.mark.parametrize("linkage", ["ward", "complete", "average", "single"])
def test_agglomerative_clustering_distances(
    n_clusters, compute_distances, distance_threshold, linkage
):
    # [FUNCTION DOC]
    # Checks distances_ (shape n_merges) exists iff compute_distances is True or
    # distance_threshold is set, for every linkage with grid connectivity.
    # Check that when `compute_distances` is True or `distance_threshold` is
    # given, the fitted model has an attribute `distances_`.
    rng = np.random.RandomState(0)
    mask = np.ones([10, 10], dtype=bool)
    n_samples = 100
    X = rng.randn(n_samples, 50)
    connectivity = grid_to_graph(*mask.shape)

    clustering = AgglomerativeClustering(
        n_clusters=n_clusters,
        connectivity=connectivity,
        linkage=linkage,
        distance_threshold=distance_threshold,
        compute_distances=compute_distances,
    )
    clustering.fit(X)
    if compute_distances or (distance_threshold is not None):
        assert hasattr(clustering, "distances_")
        n_children = clustering.children_.shape[0]
        n_nodes = n_children + 1
        assert clustering.distances_.shape == (n_nodes - 1,)
    else:
        assert not hasattr(clustering, "distances_")


@pytest.mark.no_check_spmatrix  # pickle breaks check_spmatrix
@pytest.mark.parametrize("lil_container", LIL_CONTAINERS)
def test_agglomerative_clustering(global_random_seed, lil_container):
    # [FUNCTION DOC]
    # For each linkage with grid connectivity: fit with a memory cache dir (temp dir
    # removed afterwards) gives 10 clusters; compute_full_tree=False gives same
    # partition (NMI 1); no connectivity gives 10 clusters; a wrongly shaped LIL
    # connectivity raises ValueError. Then checks complete linkage with full
    # connectivity matches unstructured for every paired metric, and precomputed
    # distances match euclidean with connectivity.
    # KNOWN ISSUE: inline comment says TypeError on dense matrices but the code asserts a
    # ValueError for a wrongly sized sparse connectivity.
    # Check that we obtain the correct number of clusters with
    # agglomerative clustering.
    rng = np.random.RandomState(global_random_seed)
    mask = np.ones([10, 10], dtype=bool)
    n_samples = 100
    X = rng.randn(n_samples, 50)
    connectivity = grid_to_graph(*mask.shape)
    for linkage in ("ward", "complete", "average", "single"):
        clustering = AgglomerativeClustering(
            n_clusters=10, connectivity=connectivity, linkage=linkage
        )
        clustering.fit(X)
        # test caching
        try:
            tempdir = mkdtemp()
            clustering = AgglomerativeClustering(
                n_clusters=10,
                connectivity=connectivity,
                memory=tempdir,
                linkage=linkage,
            )
            clustering.fit(X)
            labels = clustering.labels_
            assert np.size(np.unique(labels)) == 10
        finally:
            shutil.rmtree(tempdir)
        # Turn caching off now
        clustering = AgglomerativeClustering(
            n_clusters=10, connectivity=connectivity, linkage=linkage
        )
        # Check that we obtain the same solution with early-stopping of the
        # tree building
        clustering.compute_full_tree = False
        clustering.fit(X)
        assert_almost_equal(normalized_mutual_info_score(clustering.labels_, labels), 1)
        clustering.connectivity = None
        clustering.fit(X)
        assert np.size(np.unique(clustering.labels_)) == 10
        # Check that we raise a TypeError on dense matrices
        clustering = AgglomerativeClustering(
            n_clusters=10,
            connectivity=lil_container(connectivity.toarray()[:10, :10]),
            linkage=linkage,
        )
        with pytest.raises(ValueError):
            clustering.fit(X)

    # Test using another metric than euclidean works with linkage complete
    for metric in PAIRED_DISTANCES.keys():
        # Compare our (structured) implementation to scipy
        clustering = AgglomerativeClustering(
            n_clusters=10,
            connectivity=np.ones((n_samples, n_samples)),
            metric=metric,
            linkage="complete",
        )
        clustering.fit(X)
        clustering2 = AgglomerativeClustering(
            n_clusters=10, connectivity=None, metric=metric, linkage="complete"
        )
        clustering2.fit(X)
        assert_almost_equal(
            normalized_mutual_info_score(clustering2.labels_, clustering.labels_), 1
        )

    # Test that using a distance matrix (affinity = 'precomputed') has same
    # results (with connectivity constraints)
    clustering = AgglomerativeClustering(
        n_clusters=10, connectivity=connectivity, linkage="complete"
    )
    clustering.fit(X)
    X_dist = pairwise_distances(X)
    clustering2 = AgglomerativeClustering(
        n_clusters=10,
        connectivity=connectivity,
        metric="precomputed",
        linkage="complete",
    )
    clustering2.fit(X_dist)
    assert_array_equal(clustering.labels_, clustering2.labels_)


def test_agglomerative_clustering_memory_mapped():
    # [FUNCTION DOC]
    # Non-regression (issue 19875): single linkage fit must work on memmap-backed data.
    """AgglomerativeClustering must work on mem-mapped dataset.

    Non-regression test for issue #19875.
    """
    rng = np.random.RandomState(0)
    Xmm = create_memmap_backed_data(rng.randn(50, 100))
    AgglomerativeClustering(metric="euclidean", linkage="single").fit(Xmm)


def test_ward_agglomeration(global_random_seed):
    # [FUNCTION DOC]
    # Checks FeatureAgglomeration with 5 clusters and grid connectivity: 5 labels,
    # transform gives 5 columns, inverse_transform has 5 unique values per row and
    # round-trips through transform; fitting on zero samples raises ValueError.
    # Check that we obtain the correct solution in a simplistic case
    rng = np.random.RandomState(global_random_seed)
    mask = np.ones([10, 10], dtype=bool)
    X = rng.randn(50, 100)
    connectivity = grid_to_graph(*mask.shape)
    agglo = FeatureAgglomeration(n_clusters=5, connectivity=connectivity)
    agglo.fit(X)
    assert np.size(np.unique(agglo.labels_)) == 5

    X_red = agglo.transform(X)
    assert X_red.shape[1] == 5
    X_full = agglo.inverse_transform(X_red)
    assert np.unique(X_full[0]).size == 5
    assert_array_almost_equal(agglo.transform(X_full), X_red)

    # Check that fitting with no samples raises a ValueError
    with pytest.raises(ValueError):
        agglo.fit(X[:0])


def test_single_linkage_clustering():
    # [FUNCTION DOC]
    # Checks single linkage perfectly separates make_moons and make_circles (NMI 1).
    # Check that we get the correct result in two emblematic cases
    moons, moon_labels = make_moons(noise=0.05, random_state=42)
    clustering = AgglomerativeClustering(n_clusters=2, linkage="single")
    clustering.fit(moons)
    assert_almost_equal(
        normalized_mutual_info_score(clustering.labels_, moon_labels), 1
    )

    circles, circle_labels = make_circles(factor=0.5, noise=0.025, random_state=42)
    clustering = AgglomerativeClustering(n_clusters=2, linkage="single")
    clustering.fit(circles)
    assert_almost_equal(
        normalized_mutual_info_score(clustering.labels_, circle_labels), 1
    )


def assess_same_labelling(cut1, cut2):
    # [FUNCTION DOC]
    # HELPER: Asserts two flat labelings define the same partition by building the
    # one-hot membership matrix of each and comparing their co-clustering matrices
    # (membership times its transpose) for exact equality.
    # Inputs: cut1, cut2 - 1D integer label arrays of equal length with labels from 0.
    # Exceptions: AssertionError if partitions differ.
    """Util for comparison with scipy"""
    co_clust = []
    for cut in [cut1, cut2]:
        n = len(cut)
        k = cut.max() + 1
        ecut = np.zeros((n, k))
        ecut[np.arange(n), cut] = 1
        co_clust.append(np.dot(ecut, ecut.T))
    assert (co_clust[0] == co_clust[1]).all()


def test_sparse_scikit_vs_scipy(global_random_seed):
    # [FUNCTION DOC]
    # For every linkage in _TREE_BUILDERS and 5 random well-ordered datasets, checks
    # children with dense full connectivity equal scipy linkage children (after sorting
    # each row) and _hc_cut into 3 clusters gives the same partition; then checks _hc_cut
    # raises ValueError when asked for more clusters than leaves.
    # Test scikit linkage with full connectivity (i.e. unstructured) vs scipy
    n, p, k = 10, 5, 3
    rng = np.random.RandomState(global_random_seed)

    # Not using a lil_matrix here, just to check that non sparse
    # matrices are well handled
    connectivity = np.ones((n, n))
    for linkage in _TREE_BUILDERS.keys():
        for i in range(5):
            X = 0.1 * rng.normal(size=(n, p))
            X -= 4.0 * np.arange(n)[:, np.newaxis]
            X -= X.mean(axis=1)[:, np.newaxis]

            out = hierarchy.linkage(X, method=linkage)

            children_ = out[:, :2].astype(int, copy=False)
            children, _, n_leaves, _ = _TREE_BUILDERS[linkage](
                X, connectivity=connectivity
            )

            # Sort the order of child nodes per row for consistency
            children.sort(axis=1)
            assert_array_equal(
                children,
                children_,
                "linkage tree differs from scipy impl for linkage: " + linkage,
            )

            cut = _hc_cut(k, children, n_leaves)
            cut_ = _hc_cut(k, children_, n_leaves)
            assess_same_labelling(cut, cut_)

    # Test error management in _hc_cut
    with pytest.raises(ValueError):
        _hc_cut(n_leaves + 1, children, n_leaves)


# Make sure our custom mst_linkage_core gives
# the same results as scipy's builtin
def test_vector_scikit_single_vs_scipy_single(global_random_seed):
    # [FUNCTION DOC]
    # Checks unstructured single linkage (mst_linkage_core based) matches scipy's single
    # linkage children and 3-cluster cut.
    n_samples, n_features, n_clusters = 10, 5, 3
    rng = np.random.RandomState(global_random_seed)
    X = 0.1 * rng.normal(size=(n_samples, n_features))
    X -= 4.0 * np.arange(n_samples)[:, np.newaxis]
    X -= X.mean(axis=1)[:, np.newaxis]

    out = hierarchy.linkage(X, method="single")
    children_scipy = out[:, :2].astype(int)

    children, _, n_leaves, _ = _TREE_BUILDERS["single"](X)

    # Sort the order of child nodes per row for consistency
    children.sort(axis=1)
    assert_array_equal(
        children,
        children_scipy,
        "linkage tree differs from scipy impl for single linkage.",
    )

    cut = _hc_cut(n_clusters, children, n_leaves)
    cut_scipy = _hc_cut(n_clusters, children_scipy, n_leaves)
    assess_same_labelling(cut, cut_scipy)


@pytest.mark.parametrize("metric_param_grid", METRICS_DEFAULT_PARAMS)
def test_mst_linkage_core_memory_mapped(metric_param_grid):
    # [FUNCTION DOC]
    # Non-regression (issue 19875): for each metric and every parameter combination in
    # METRICS_DEFAULT_PARAMS, mst_linkage_core gives identical output on an array and
    # on its memmap copy.
    """The MST-LINKAGE-CORE algorithm must work on mem-mapped dataset.

    Non-regression test for issue #19875.
    """
    rng = np.random.RandomState(seed=1)
    X = rng.normal(size=(20, 4))
    Xmm = create_memmap_backed_data(X)
    metric, param_grid = metric_param_grid
    keys = param_grid.keys()
    for vals in itertools.product(*param_grid.values()):
        kwargs = dict(zip(keys, vals))
        distance_metric = DistanceMetric.get_metric(metric, **kwargs)
        mst = mst_linkage_core(X, distance_metric)
        mst_mm = mst_linkage_core(Xmm, distance_metric)
        np.testing.assert_equal(mst, mst_mm)


def test_identical_points():
    # [FUNCTION DOC]
    # Checks duplicated points with a symmetrized kneighbors connectivity fixed by
    # _fix_connectivity are clustered correctly into 3 groups (NMI 1).
    # KNOWN ISSUE: the linkage tuple lists "average" twice and omits "complete".
    # Ensure identical points are handled correctly when using mst with
    # a sparse connectivity matrix
    X = np.array([[0, 0, 0], [0, 0, 0], [1, 1, 1], [1, 1, 1], [2, 2, 2], [2, 2, 2]])
    true_labels = np.array([0, 0, 1, 1, 2, 2])
    connectivity = kneighbors_graph(X, n_neighbors=3, include_self=False)
    connectivity = 0.5 * (connectivity + connectivity.T)
    connectivity, n_components = _fix_connectivity(X, connectivity, "euclidean")

    for linkage in ("single", "average", "average", "ward"):
        clustering = AgglomerativeClustering(
            n_clusters=3, linkage=linkage, connectivity=connectivity
        )
        clustering.fit(X)

        assert_almost_equal(
            normalized_mutual_info_score(clustering.labels_, true_labels), 1
        )


def test_connectivity_propagation():
    # [FUNCTION DOC]
    # Regression smoke test: ward with a kneighbors connectivity on near-duplicate points
    # must not crash (IndexError) due to bad connectivity propagation during merges.
    # Check that connectivity in the ward tree is propagated correctly during
    # merging.
    X = np.array(
        [
            (0.014, 0.120),
            (0.014, 0.099),
            (0.014, 0.097),
            (0.017, 0.153),
            (0.017, 0.153),
            (0.018, 0.153),
            (0.018, 0.153),
            (0.018, 0.153),
            (0.018, 0.153),
            (0.018, 0.153),
            (0.018, 0.153),
            (0.018, 0.153),
            (0.018, 0.152),
            (0.018, 0.149),
            (0.018, 0.144),
        ]
    )
    connectivity = kneighbors_graph(X, 10, include_self=False)
    ward = AgglomerativeClustering(
        n_clusters=4, connectivity=connectivity, linkage="ward"
    )
    # If changes are not propagated correctly, fit crashes with an
    # IndexError
    ward.fit(X)


def test_ward_tree_children_order(global_random_seed):
    # [FUNCTION DOC]
    # Checks ward_tree children are identical with and without full connectivity on 5
    # random datasets.
    # Check that children are ordered in the same way for both structured and
    # unstructured versions of ward_tree.

    # test on five random datasets
    n, p = 10, 5
    rng = np.random.RandomState(global_random_seed)

    connectivity = np.ones((n, n))
    for i in range(5):
        X = 0.1 * rng.normal(size=(n, p))
        X -= 4.0 * np.arange(n)[:, np.newaxis]
        X -= X.mean(axis=1)[:, np.newaxis]

        out_unstructured = ward_tree(X)
        out_structured = ward_tree(X, connectivity=connectivity)

        assert_array_equal(out_unstructured[0], out_structured[0])


def test_ward_linkage_tree_return_distance(global_random_seed):
    # [FUNCTION DOC]
    # Checks return_distance outputs agree between structured (full connectivity) and
    # unstructured ward_tree and linkage_tree on random data, then compares children and
    # distances to hard-coded scipy reference linkages for ward, complete and average.
    # KNOWN ISSUE: in the random-data loop, linkage_tree(...)[-1] is already the
    # distances array, so the further [-1] / [0] index scalar distances and the
    # "children" comparison is really a first-distance comparison. Also zip with only two
    # truth arrays silently drops the "single" option.
    # Test return_distance option on linkage and ward trees

    # test that return_distance when set true, gives same
    # output on both structured and unstructured clustering.
    n, p = 10, 5
    rng = np.random.RandomState(global_random_seed)

    connectivity = np.ones((n, n))
    for i in range(5):
        X = 0.1 * rng.normal(size=(n, p))
        X -= 4.0 * np.arange(n)[:, np.newaxis]
        X -= X.mean(axis=1)[:, np.newaxis]

        out_unstructured = ward_tree(X, return_distance=True)
        out_structured = ward_tree(X, connectivity=connectivity, return_distance=True)

        # get children
        children_unstructured = out_unstructured[0]
        children_structured = out_structured[0]

        # check if we got the same clusters
        assert_array_equal(children_unstructured, children_structured)

        # check if the distances are the same
        dist_unstructured = out_unstructured[-1]
        dist_structured = out_structured[-1]

        assert_array_almost_equal(dist_unstructured, dist_structured)

        for linkage in ["average", "complete", "single"]:
            structured_items = linkage_tree(
                X, connectivity=connectivity, linkage=linkage, return_distance=True
            )[-1]
            unstructured_items = linkage_tree(X, linkage=linkage, return_distance=True)[
                -1
            ]
            structured_dist = structured_items[-1]
            unstructured_dist = unstructured_items[-1]
            structured_children = structured_items[0]
            unstructured_children = unstructured_items[0]
            assert_array_almost_equal(structured_dist, unstructured_dist)
            assert_array_almost_equal(structured_children, unstructured_children)

    # test on the following dataset where we know the truth
    # taken from scipy/cluster/tests/hierarchy_test_data.py
    X = np.array(
        [
            [1.43054825, -7.5693489],
            [6.95887839, 6.82293382],
            [2.87137846, -9.68248579],
            [7.87974764, -6.05485803],
            [8.24018364, -6.09495602],
            [7.39020262, 8.54004355],
        ]
    )
    # truth
    linkage_X_ward = np.array(
        [
            [3.0, 4.0, 0.36265956, 2.0],
            [1.0, 5.0, 1.77045373, 2.0],
            [0.0, 2.0, 2.55760419, 2.0],
            [6.0, 8.0, 9.10208346, 4.0],
            [7.0, 9.0, 24.7784379, 6.0],
        ]
    )

    linkage_X_complete = np.array(
        [
            [3.0, 4.0, 0.36265956, 2.0],
            [1.0, 5.0, 1.77045373, 2.0],
            [0.0, 2.0, 2.55760419, 2.0],
            [6.0, 8.0, 6.96742194, 4.0],
            [7.0, 9.0, 18.77445997, 6.0],
        ]
    )

    linkage_X_average = np.array(
        [
            [3.0, 4.0, 0.36265956, 2.0],
            [1.0, 5.0, 1.77045373, 2.0],
            [0.0, 2.0, 2.55760419, 2.0],
            [6.0, 8.0, 6.55832839, 4.0],
            [7.0, 9.0, 15.44089605, 6.0],
        ]
    )

    n_samples, n_features = np.shape(X)
    connectivity_X = np.ones((n_samples, n_samples))

    out_X_unstructured = ward_tree(X, return_distance=True)
    out_X_structured = ward_tree(X, connectivity=connectivity_X, return_distance=True)

    # check that the labels are the same
    assert_array_equal(linkage_X_ward[:, :2], out_X_unstructured[0])
    assert_array_equal(linkage_X_ward[:, :2], out_X_structured[0])

    # check that the distances are correct
    assert_array_almost_equal(linkage_X_ward[:, 2], out_X_unstructured[4])
    assert_array_almost_equal(linkage_X_ward[:, 2], out_X_structured[4])

    linkage_options = ["complete", "average", "single"]
    X_linkage_truth = [linkage_X_complete, linkage_X_average]
    for linkage, X_truth in zip(linkage_options, X_linkage_truth):
        out_X_unstructured = linkage_tree(X, return_distance=True, linkage=linkage)
        out_X_structured = linkage_tree(
            X, connectivity=connectivity_X, linkage=linkage, return_distance=True
        )

        # check that the labels are the same
        assert_array_equal(X_truth[:, :2], out_X_unstructured[0])
        assert_array_equal(X_truth[:, :2], out_X_structured[0])

        # check that the distances are correct
        assert_array_almost_equal(X_truth[:, 2], out_X_unstructured[4])
        assert_array_almost_equal(X_truth[:, 2], out_X_structured[4])


def test_connectivity_fixing_non_lil():
    # [FUNCTION DOC]
    # Non-regression: a non item-assignable (COO) connectivity with 2 components must be
    # fixed, emitting a UserWarning, without crashing.
    # Check non regression of a bug if a non item assignable connectivity is
    # provided with more than one component.
    # create dummy data
    x = np.array([[0, 0], [1, 1]])
    # create a mask with several components to force connectivity fixing
    m = np.array([[True, False], [False, True]])
    c = grid_to_graph(n_x=2, n_y=2, mask=m)
    w = AgglomerativeClustering(connectivity=c, linkage="ward")
    with pytest.warns(UserWarning):
        w.fit(x)


def test_int_float_dict():
    # [FUNCTION DOC]
    # Checks IntFloatDict stores and returns given key/value pairs, and smoke-tests
    # max_merge and average_merge on two dicts.
    rng = np.random.RandomState(0)
    keys = np.unique(rng.randint(100, size=10).astype(np.intp, copy=False))
    values = rng.rand(len(keys))

    d = IntFloatDict(keys, values)
    for key, value in zip(keys, values):
        assert d[key] == value

    other_keys = np.arange(50, dtype=np.intp)[::2]
    other_values = np.full(50, 0.5)[::2]
    other = IntFloatDict(other_keys, other_values)
    # Complete smoke test
    max_merge(d, other, mask=np.ones(100, dtype=np.intp), n_a=1, n_b=1)
    average_merge(d, other, mask=np.ones(100, dtype=np.intp), n_a=1, n_b=1)


def test_connectivity_callable():
    # [FUNCTION DOC]
    # Checks a callable connectivity (partial kneighbors_graph) gives the same labels as
    # the precomputed graph.
    rng = np.random.RandomState(0)
    X = rng.rand(20, 5)
    connectivity = kneighbors_graph(X, 3, include_self=False)
    aglc1 = AgglomerativeClustering(connectivity=connectivity)
    aglc2 = AgglomerativeClustering(
        connectivity=partial(kneighbors_graph, n_neighbors=3, include_self=False)
    )
    aglc1.fit(X)
    aglc2.fit(X)
    assert_array_equal(aglc1.labels_, aglc2.labels_)


def test_connectivity_ignores_diagonal():
    # [FUNCTION DOC]
    # Checks labels are the same whether connectivity includes self-loops or not.
    rng = np.random.RandomState(0)
    X = rng.rand(20, 5)
    connectivity = kneighbors_graph(X, 3, include_self=False)
    connectivity_include_self = kneighbors_graph(X, 3, include_self=True)
    aglc1 = AgglomerativeClustering(connectivity=connectivity)
    aglc2 = AgglomerativeClustering(connectivity=connectivity_include_self)
    aglc1.fit(X)
    aglc2.fit(X)
    assert_array_equal(aglc1.labels_, aglc2.labels_)


def test_compute_full_tree():
    # [FUNCTION DOC]
    # Checks compute_full_tree="auto": with 2 clusters the full tree (n_samples - 1
    # merges) is built; with 101 clusters (above max(100, 0.02 n)) building stops early
    # with n_samples - n_clusters merges.
    # Test that the full tree is computed if n_clusters is small
    rng = np.random.RandomState(0)
    X = rng.randn(10, 2)
    connectivity = kneighbors_graph(X, 5, include_self=False)

    # When n_clusters is less, the full tree should be built
    # that is the number of merges should be n_samples - 1
    agc = AgglomerativeClustering(n_clusters=2, connectivity=connectivity)
    agc.fit(X)
    n_samples = X.shape[0]
    n_nodes = agc.children_.shape[0]
    assert n_nodes == n_samples - 1

    # When n_clusters is large, greater than max of 100 and 0.02 * n_samples.
    # we should stop when there are n_clusters.
    n_clusters = 101
    X = rng.randn(200, 2)
    connectivity = kneighbors_graph(X, 10, include_self=False)
    agc = AgglomerativeClustering(n_clusters=n_clusters, connectivity=connectivity)
    agc.fit(X)
    n_samples = X.shape[0]
    n_nodes = agc.children_.shape[0]
    assert n_nodes == n_samples - n_clusters


def test_n_components():
    # [FUNCTION DOC]
    # Checks every tree builder reports 5 connected components for identity
    # connectivity (warnings ignored).
    # Test n_components returned by linkage, average and ward tree
    rng = np.random.RandomState(0)
    X = rng.rand(5, 5)

    # Connectivity matrix having five components.
    connectivity = np.eye(5)

    for linkage_func in _TREE_BUILDERS.values():
        assert ignore_warnings(linkage_func)(X, connectivity=connectivity)[1] == 5


def test_affinity_passed_to_fix_connectivity():
    # [FUNCTION DOC]
    # Checks the callable affinity is forwarded to connectivity fixing by counting calls:
    # a 2x2 grid with 2 components leads to exactly 3 calls of the counter.
    # Test that the affinity parameter is actually passed to the pairwise
    # function

    size = 2
    rng = np.random.RandomState(0)
    X = rng.randn(size, size)
    mask = np.array([True, False, False, True])

    connectivity = grid_to_graph(n_x=size, n_y=size, mask=mask, return_as=np.ndarray)

    class FakeAffinity:
        # [CLASS DOC]
        # HELPER: Callable-affinity stand-in that counts how many times it is invoked.
        def __init__(self):
            # [FUNCTION DOC]
            # HELPER: Initializes counter to 0.
            self.counter = 0

        def increment(self, *args, **kwargs):
            # [FUNCTION DOC]
            # HELPER: Accepts any args, increments counter and returns the new count.
            # Side effects: mutates self.counter.
            self.counter += 1
            return self.counter

    fa = FakeAffinity()

    linkage_tree(X, connectivity=connectivity, affinity=fa.increment)

    assert fa.counter == 3


@pytest.mark.parametrize("linkage", ["ward", "complete", "average"])
def test_agglomerative_clustering_with_distance_threshold(linkage, global_random_seed):
    # [FUNCTION DOC]
    # With distance_threshold 10 (with and without grid connectivity) checks the number
    # of clusters equals 1 plus the number of merge distances >= threshold from the
    # tree builder, and labels equal _hc_cut at that number of clusters.
    # Check that we obtain the correct number of clusters with
    # agglomerative clustering with distance_threshold.
    rng = np.random.RandomState(global_random_seed)
    mask = np.ones([10, 10], dtype=bool)
    n_samples = 100
    X = rng.randn(n_samples, 50)
    connectivity = grid_to_graph(*mask.shape)
    # test when distance threshold is set to 10
    distance_threshold = 10
    for conn in [None, connectivity]:
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=distance_threshold,
            connectivity=conn,
            linkage=linkage,
        )
        clustering.fit(X)
        clusters_produced = clustering.labels_
        num_clusters_produced = len(np.unique(clustering.labels_))
        # test if the clusters produced match the point in the linkage tree
        # where the distance exceeds the threshold
        tree_builder = _TREE_BUILDERS[linkage]
        children, n_components, n_leaves, parent, distances = tree_builder(
            X, connectivity=conn, n_clusters=None, return_distance=True
        )
        num_clusters_at_threshold = (
            np.count_nonzero(distances >= distance_threshold) + 1
        )
        # test number of clusters produced
        assert num_clusters_at_threshold == num_clusters_produced
        # test clusters produced
        clusters_at_threshold = _hc_cut(
            n_clusters=num_clusters_produced, children=children, n_leaves=n_leaves
        )
        assert np.array_equiv(clusters_produced, clusters_at_threshold)


def test_small_distance_threshold(global_random_seed):
    # [FUNCTION DOC]
    # Checks single linkage with threshold 1.0 on 10 integer points with pairwise
    # distances verified above 0.1 leaves every point in its own cluster.
    # KNOWN ISSUE: the check uses 0.1 while the threshold is 1.0, so the precondition
    # asserted is weaker than what the outcome relies on (integer points make it hold).
    rng = np.random.RandomState(global_random_seed)
    n_samples = 10
    X = rng.randint(-300, 300, size=(n_samples, 3))
    # this should result in all data in their own clusters, given that
    # their pairwise distances are bigger than .1 (which may not be the case
    # with a different random seed).
    clustering = AgglomerativeClustering(
        n_clusters=None, distance_threshold=1.0, linkage="single"
    ).fit(X)
    # check that the pairwise distances are indeed all larger than .1
    all_distances = pairwise_distances(X, metric="minkowski", p=2)
    np.fill_diagonal(all_distances, np.inf)
    assert np.all(all_distances > 0.1)
    assert clustering.n_clusters_ == n_samples


def test_cluster_distances_with_distance_threshold(global_random_seed):
    # [FUNCTION DOC]
    # Single linkage, threshold 4: for each cluster checks every member's nearest
    # in-cluster neighbor is closer than threshold (non-singletons) and the nearest
    # out-of-cluster point is at least threshold away.
    rng = np.random.RandomState(global_random_seed)
    n_samples = 100
    X = rng.randint(-10, 10, size=(n_samples, 3))
    # check the distances within the clusters and with other clusters
    distance_threshold = 4
    clustering = AgglomerativeClustering(
        n_clusters=None, distance_threshold=distance_threshold, linkage="single"
    ).fit(X)
    labels = clustering.labels_
    D = pairwise_distances(X, metric="minkowski", p=2)
    # to avoid taking the 0 diagonal in min()
    np.fill_diagonal(D, np.inf)
    for label in np.unique(labels):
        in_cluster_mask = labels == label
        max_in_cluster_distance = (
            D[in_cluster_mask][:, in_cluster_mask].min(axis=0).max()
        )
        min_out_cluster_distance = (
            D[in_cluster_mask][:, ~in_cluster_mask].min(axis=0).min()
        )
        # single data point clusters only have that inf diagonal here
        if in_cluster_mask.sum() > 1:
            assert max_in_cluster_distance < distance_threshold
        assert min_out_cluster_distance >= distance_threshold


@pytest.mark.parametrize("linkage", ["ward", "complete", "average"])
@pytest.mark.parametrize(
    ("threshold", "y_true"), [(0.5, [1, 0]), (1.0, [1, 0]), (1.5, [0, 0])]
)
def test_agglomerative_clustering_with_distance_threshold_edge_case(
    linkage, threshold, y_true
):
    # [FUNCTION DOC]
    # Two points at distance 1: thresholds 0.5 and 1.0 give two clusters, 1.5 gives
    # one, checked with ARI for ward, complete and average.
    # test boundary case of distance_threshold matching the distance
    X = [[0], [1]]
    clusterer = AgglomerativeClustering(
        n_clusters=None, distance_threshold=threshold, linkage=linkage
    )
    y_pred = clusterer.fit_predict(X)
    assert adjusted_rand_score(y_true, y_pred) == 1


def test_dist_threshold_invalid_parameters():
    # [FUNCTION DOC]
    # Checks ValueError when both or neither of n_clusters and distance_threshold are
    # set, and when distance_threshold is used with compute_full_tree=False.
    X = [[0], [1]]
    with pytest.raises(ValueError, match="Exactly one of "):
        AgglomerativeClustering(n_clusters=None, distance_threshold=None).fit(X)

    with pytest.raises(ValueError, match="Exactly one of "):
        AgglomerativeClustering(n_clusters=2, distance_threshold=1).fit(X)

    X = [[0], [1]]
    with pytest.raises(ValueError, match="compute_full_tree must be True if"):
        AgglomerativeClustering(
            n_clusters=None, distance_threshold=1, compute_full_tree=False
        ).fit(X)


def test_invalid_shape_precomputed_dist_matrix():
    # [FUNCTION DOC]
    # Checks precomputed metric with a non-square (5, 3) matrix raises ValueError.
    # Check that an error is raised when affinity='precomputed'
    # and a non square matrix is passed (PR #16257).
    rng = np.random.RandomState(0)
    X = rng.rand(5, 3)
    with pytest.raises(
        ValueError,
        match=r"Distance matrix should be square, got matrix of shape \(5, 3\)",
    ):
        AgglomerativeClustering(metric="precomputed", linkage="complete").fit(X)


def test_precomputed_connectivity_metric_with_2_connected_components():
    # [FUNCTION DOC]
    # Non-regression (issue 16151): with a 2-component connectivity, precomputed and
    # euclidean fits both warn about completing the connectivity and give identical
    # labels and children.
    """Check that connecting components works when connectivity and
    affinity are both precomputed and the number of connected components is
    greater than 1. Non-regression test for #16151.
    """

    connectivity_matrix = np.array(
        [
            [0, 1, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0],
        ]
    )
    # ensure that connectivity_matrix has two connected components
    assert connected_components(connectivity_matrix)[0] == 2

    rng = np.random.RandomState(0)
    X = rng.randn(5, 10)

    X_dist = pairwise_distances(X)
    clusterer_precomputed = AgglomerativeClustering(
        metric="precomputed", connectivity=connectivity_matrix, linkage="complete"
    )
    msg = "Completing it to avoid stopping the tree early"
    with pytest.warns(UserWarning, match=msg):
        clusterer_precomputed.fit(X_dist)

    clusterer = AgglomerativeClustering(
        connectivity=connectivity_matrix, linkage="complete"
    )
    with pytest.warns(UserWarning, match=msg):
        clusterer.fit(X)

    assert_array_equal(clusterer.labels_, clusterer_precomputed.labels_)
    assert_array_equal(clusterer.children_, clusterer_precomputed.children_)


@pytest.mark.parametrize("Clustering", [AgglomerativeClustering, FeatureAgglomeration])
def test_agglomeration_ward_constrained_metric(Clustering):
    # [FUNCTION DOC]
    # Checks ward linkage with metric "manhattan" raises ValueError for both
    # AgglomerativeClustering and FeatureAgglomeration.
    """Check that we raise an error when 'euclidean' or 'l2' are not passed with
    ward linkage."""
    rng = np.random.RandomState(0)
    mask = np.ones([10, 10], dtype=bool)
    n_samples = 100
    X = rng.randn(n_samples, 50)
    connectivity = grid_to_graph(*mask.shape)

    clustering = Clustering(
        n_clusters=10,
        connectivity=connectivity.toarray(),
        metric="manhattan",
        linkage="ward",
    )
    with pytest.raises(ValueError):
        clustering.fit(X)


@pytest.mark.parametrize("Clustering", [AgglomerativeClustering, FeatureAgglomeration])
@pytest.mark.parametrize("metric", ["euclidean", "l2"])
def test_agglomeration_ward_euclidean(Clustering, metric):
    # [FUNCTION DOC]
    # Smoke test that ward accepts metric "euclidean" and "l2" for both estimators.
    """Check that we can pass 'euclidean' and 'l2' as metric with Ward linkage."""
    rng = np.random.RandomState(0)
    mask = np.ones([10, 10], dtype=bool)
    n_samples = 100
    X = rng.randn(n_samples, 100)
    connectivity = grid_to_graph(*mask.shape)

    clustering = Clustering(
        n_clusters=10,
        connectivity=connectivity.toarray(),
        metric=metric,
        linkage="ward",
    )
    clustering.fit(X)
