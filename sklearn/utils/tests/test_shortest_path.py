# [MODULE DOC]
# Tests for sklearn.utils.graph.single_source_shortest_path_length, comparing against a
# naive Floyd-Warshall reference on a random sparse graph with unit edge weights.
from collections import defaultdict

import numpy as np
from numpy.testing import assert_array_almost_equal

from sklearn.utils.graph import single_source_shortest_path_length


def floyd_warshall_slow(graph, directed=False):
    # [FUNCTION DOC]
    # HELPER: Naive O(N^3) all-pairs shortest path. Replaces zero entries by infinity
    # (no edge), sets the diagonal to 0, symmetrizes with the elementwise minimum of
    # graph and its transpose when not directed, relaxes every (i, j) through every
    # intermediate k, then maps unreachable (infinite) distances back to 0.
    # Inputs: graph - square float ndarray of edge weights (0 = no edge); directed -
    # bool, default False.
    # Outputs: square ndarray of shortest path lengths (0 for unreachable).
    # Side effects: modifies graph in place for the infinity/diagonal steps (the
    # symmetrized case then works on a new array).
    # KNOWN ISSUE: the inline comment says 'set nonzero entries to infinity' but the
    # code sets the zero entries.
    N = graph.shape[0]

    # set nonzero entries to infinity
    graph[np.where(graph == 0)] = np.inf

    # set diagonal to zero
    graph.flat[:: N + 1] = 0

    if not directed:
        graph = np.minimum(graph, graph.T)

    for k in range(N):
        for i in range(N):
            for j in range(N):
                graph[i, j] = min(graph[i, j], graph[i, k] + graph[k, j])

    graph[np.where(np.isinf(graph))] = 0

    return graph


def generate_graph(N=20):
    # [FUNCTION DOC]
    # HELPER: Builds a random symmetric N x N distance matrix (seed 0), zeroes about
    # N*N/2 randomly chosen entries (not symmetric, so the result can be asymmetric) and
    # sets the diagonal to 0.
    # Inputs: N - int number of nodes, default 20.
    # Outputs: float ndarray of shape (N, N).
    # KNOWN ISSUE: the comment calls it symmetric, but the random zeroing step breaks
    # symmetry.
    # sparse grid of distances
    rng = np.random.RandomState(0)
    dist_matrix = rng.random_sample((N, N))

    # make symmetric: distances are not direction-dependent
    dist_matrix = dist_matrix + dist_matrix.T

    # make graph sparse
    i = (rng.randint(N, size=N * N // 2), rng.randint(N, size=N * N // 2))
    dist_matrix[i] = 0

    # set diagonal to zero
    dist_matrix.flat[:: N + 1] = 0

    return dist_matrix


def test_shortest_path():
    # [FUNCTION DOC]
    # Verifies single_source_shortest_path_length against floyd_warshall_slow for every
    # source node, first on the directed 0/1 adjacency, then on its symmetrized version;
    # unreachable nodes are compared as distance 0 via a defaultdict.
    dist_matrix = generate_graph(20)
    # We compare path length and not costs (-> set distances to 0 or 1)
    dist_matrix[dist_matrix != 0] = 1

    for directed in (True, False):
        if not directed:
            dist_matrix = np.minimum(dist_matrix, dist_matrix.T)

        graph_py = floyd_warshall_slow(dist_matrix.copy(), directed)
        for i in range(dist_matrix.shape[0]):
            # Non-reachable nodes have distance 0 in graph_py
            dist_dict = defaultdict(int)
            dist_dict.update(single_source_shortest_path_length(dist_matrix, i))

            for j in range(graph_py[i].shape[0]):
                assert_array_almost_equal(dist_dict[j], graph_py[i, j])
