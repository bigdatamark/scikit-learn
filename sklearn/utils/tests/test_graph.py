# [MODULE DOC]
# Tests for sklearn.utils.graph._fix_connected_components, which adds edges between
# disconnected components of a sparse neighbors graph until it is fully connected.
# All tests use small 1-D point sets with gaps so kneighbors_graph yields several
# components.
import numpy as np
import pytest
from scipy.sparse.csgraph import connected_components

from sklearn.metrics.pairwise import pairwise_distances
from sklearn.neighbors import kneighbors_graph
from sklearn.utils.graph import _fix_connected_components


def test_fix_connected_components():
    # [FUNCTION DOC]
    # Verifies a 2-NN distance graph over two point clusters starts with more than one
    # connected component and ends with exactly one after the fix (feature input X).
    # Test that _fix_connected_components reduces the number of component to 1.
    X = np.array([0, 1, 2, 5, 6, 7])[:, None]
    graph = kneighbors_graph(X, n_neighbors=2, mode="distance")

    n_connected_components, labels = connected_components(graph)
    assert n_connected_components > 1

    graph = _fix_connected_components(X, graph, n_connected_components, labels)

    n_connected_components, labels = connected_components(graph)
    assert n_connected_components == 1


def test_fix_connected_components_precomputed():
    # [FUNCTION DOC]
    # Verifies the fix works with metric="precomputed" given a dense pairwise distance
    # matrix (yields one component), and that passing a sparse graph as the precomputed
    # distances raises RuntimeError mentioning "does not work with a sparse".
    # Test that _fix_connected_components accepts precomputed distance matrix.
    X = np.array([0, 1, 2, 5, 6, 7])[:, None]
    graph = kneighbors_graph(X, n_neighbors=2, mode="distance")

    n_connected_components, labels = connected_components(graph)
    assert n_connected_components > 1

    distances = pairwise_distances(X)
    graph = _fix_connected_components(
        distances, graph, n_connected_components, labels, metric="precomputed"
    )

    n_connected_components, labels = connected_components(graph)
    assert n_connected_components == 1

    # but it does not work with precomputed neighbors graph
    with pytest.raises(RuntimeError, match="does not work with a sparse"):
        _fix_connected_components(
            graph, graph, n_connected_components, labels, metric="precomputed"
        )


def test_fix_connected_components_wrong_mode():
    # [FUNCTION DOC]
    # Verifies an unrecognized mode string ("foo") raises ValueError "Unknown mode".
    # Test that the an error is raised if the mode string is incorrect.
    X = np.array([0, 1, 2, 5, 6, 7])[:, None]
    graph = kneighbors_graph(X, n_neighbors=2, mode="distance")
    n_connected_components, labels = connected_components(graph)

    with pytest.raises(ValueError, match="Unknown mode"):
        graph = _fix_connected_components(
            X, graph, n_connected_components, labels, mode="foo"
        )


def test_fix_connected_components_connectivity_mode():
    # [FUNCTION DOC]
    # Verifies mode="connectivity" on a 1-NN connectivity graph produces a graph whose
    # stored values are all 1, i.e. added edges get weight one.
    # Test that the connectivity mode fill new connections with ones.
    X = np.array([0, 1, 6, 7])[:, None]
    graph = kneighbors_graph(X, n_neighbors=1, mode="connectivity")
    n_connected_components, labels = connected_components(graph)
    graph = _fix_connected_components(
        X, graph, n_connected_components, labels, mode="connectivity"
    )
    assert np.all(graph.data == 1)


def test_fix_connected_components_distance_mode():
    # [FUNCTION DOC]
    # Verifies mode="distance" adds edges weighted by actual distances: starting from a
    # 1-NN graph whose stored distances are all 1, after the fix not all values equal 1.
    # Test that the distance mode does not fill new connections with ones.
    X = np.array([0, 1, 6, 7])[:, None]
    graph = kneighbors_graph(X, n_neighbors=1, mode="distance")
    assert np.all(graph.data == 1)

    n_connected_components, labels = connected_components(graph)
    graph = _fix_connected_components(
        X, graph, n_connected_components, labels, mode="distance"
    )
    assert not np.all(graph.data == 1)
