# [MODULE DOC]
# Compiled helpers for agglomerative clustering (sklearn/cluster/_agglomerative.py):
# Ward distance computation for connectivity-constrained Ward linkage, utilities
# to explore and cut trees/forests, merge strategies on IntFloatDict adjacency
# dictionaries (max / average linkage with connectivity), a WeightedEdge type used
# in heaps, a UnionFind structure to relabel MST edges into a scipy-style linkage
# matrix, and the MST-LINKAGE-CORE (Prim-like) algorithm for single linkage.
# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
cimport cython

from sklearn.metrics._dist_metrics cimport DistanceMetric64
from sklearn.utils._fast_dict cimport IntFloatDict
from sklearn.utils._typedefs cimport float64_t, intp_t, uint8_t

# C++
from cython.operator cimport dereference as deref, preincrement as inc
from libcpp.map cimport map as cpp_map
from libc.math cimport fmax, INFINITY


###############################################################################
# Utilities for computing the ward momentum

def compute_ward_dist(
    const float64_t[::1] m_1,
    const float64_t[:, ::1] m_2,
    const intp_t[::1] coord_row,
    const intp_t[::1] coord_col,
    float64_t[::1] res
):
    # [FUNCTION DOC]
    # HELPER: used by ward_tree to compute merge costs for connected cluster pairs.
    # Behavior: for each pair i (row = coord_row[i], col = coord_col[i]), computes the
    # Ward increase in inertia: n_r*n_c/(n_r+n_c) times the squared Euclidean distance
    # between the cluster means m_2[row]/m_1[row] and m_2[col]/m_1[col].
    # Inputs: m_1 (n_nodes,) cluster sizes (moment 0); m_2 (n_nodes, n_features)
    # per-cluster feature sums (moment 1); coord_row, coord_col (n_pairs,) node indices.
    # Outputs: res (n_pairs,) written in place with the costs; nothing returned.
    # Exceptions: none explicit; zero sizes give division by zero (inf/nan).
    # Side effects: mutates res; holds the GIL.
    cdef intp_t size_max = coord_row.shape[0]
    cdef intp_t n_features = m_2.shape[1]
    cdef intp_t i, j, row, col
    cdef float64_t pa, n

    for i in range(size_max):
        row = coord_row[i]
        col = coord_col[i]
        n = (m_1[row] * m_1[col]) / (m_1[row] + m_1[col])
        pa = 0.
        for j in range(n_features):
            pa += (m_2[row, j] / m_1[row] - m_2[col, j] / m_1[col]) ** 2
        res[i] = pa * n


###############################################################################
# Utilities for cutting and exploring a hierarchical tree

def _hc_get_descendent(intp_t node, children, intp_t n_leaves):
    # [FUNCTION DOC]
    # HELPER: used by _hc_cut / tree cutting to collect leaves under a node.
    # Behavior: if node is a leaf (< n_leaves) returns [node]. Otherwise performs an
    # iterative traversal with a Python list as stack: popping an internal node pushes
    # its two children from children[i - n_leaves]; popping a leaf appends it to the
    # result. n_indices tracks the stack size to avoid calling len.
    # Inputs: node id; children sequence of child pairs for internal nodes; n_leaves.
    # Outputs: list of leaf indices (order is traversal order, not sorted).
    # Exceptions: IndexError for invalid node ids. Side effects: none.
    """
    Function returning all the descendent leaves of a set of nodes in the tree.

    Parameters
    ----------
    node : integer
        The node for which we want the descendents.

    children : list of pairs, length n_nodes
        The children of each non-leaf node. Values less than `n_samples` refer
        to leaves of the tree. A greater value `i` indicates a node with
        children `children[i - n_samples]`.

    n_leaves : integer
        Number of leaves.

    Returns
    -------
    descendent : list of int
    """
    ind = [node]
    if node < n_leaves:
        return ind
    descendent = []

    # It is actually faster to do the accounting of the number of
    # elements is the list ourselves: len is a lengthy operation on a
    # chained list
    cdef intp_t i, n_indices = 1

    while n_indices:
        i = ind.pop()
        if i < n_leaves:
            descendent.append(i)
            n_indices -= 1
        else:
            ind.extend(children[i - n_leaves])
            n_indices += 1
    return descendent


def hc_get_heads(intp_t[:] parents, copy=True):
    """Returns the heads of the forest, as defined by parents.

    Parameters
    ----------
    parents : array of integers
        The parent structure defining the forest (ensemble of trees)
    copy : boolean
        If copy is False, the input 'parents' array is modified inplace

    Returns
    -------
    heads : array of integers of same shape as parents
        The indices in the 'parents' of the tree heads

    """
    # [FUNCTION DOC]
    # Behavior: for every node, scanning from the highest index down, follows the
    # parent chain until reaching a root (a node that is its own parent) and stores
    # that root as the node's entry. Ancestor chains are partially compressed because
    # entries already processed (higher indices) were replaced by their roots.
    # Inputs: parents (n_nodes,) intp, parents[root] == root; copy whether to work on a copy.
    # Outputs: array where entry k is the root of node k.
    # Exceptions: infinite loop if the parent structure contains a cycle other than
    # self loops. Side effects: modifies parents in place when copy is False.
    cdef intp_t parent, node0, node, size
    if copy:
        parents = np.copy(parents)
    size = parents.size

    # Start from the top of the tree and go down
    for node0 in range(size - 1, -1, -1):
        node = node0
        parent = parents[node]
        while parent != node:
            parents[node0] = parent
            node = parent
            parent = parents[node]
    return parents


def _get_parents(
    nodes,
    heads,
    const intp_t[:] parents,
    uint8_t[::1] not_visited
):
    """Returns the heads of the given nodes, as defined by parents.

    Modifies 'heads' and 'not_visited' in-place.

    Parameters
    ----------
    nodes : list of integers
        The nodes to start from
    heads : list of integers
        A list to hold the results (modified inplace)
    parents : array of integers
        The parent structure defining the tree
    not_visited
        The tree nodes to consider (modified inplace)

    """
    # [FUNCTION DOC]
    # HELPER: used by _hc_cut-style forest exploration in agglomerative code.
    # Behavior: for each starting node, follows parents to the root; if that root is
    # still flagged in not_visited, clears the flag and appends the root to heads.
    # Inputs: nodes iterable of node ids; heads Python list (output); parents (n_nodes,)
    # with roots pointing to themselves; not_visited uint8 flags (n_nodes,).
    # Outputs: heads appended in place; not_visited cleared for found roots.
    # Exceptions: none explicit. Side effects: mutates heads and not_visited.
    cdef intp_t parent, node

    for node in nodes:
        parent = parents[node]
        while parent != node:
            node = parent
            parent = parents[node]
        if not_visited[node]:
            not_visited[node] = 0
            heads.append(node)


###############################################################################
# merge strategies implemented on IntFloatDicts

# These are used in the hierarchical clustering code, to implement
# merging between two clusters, defined as a dict containing node number
# as keys and edge weights as values.


def max_merge(
    IntFloatDict a,
    IntFloatDict b,
    const intp_t[:] mask,
    intp_t n_a,
    intp_t n_b
):
    """Merge two IntFloatDicts with the max strategy: when the same key is
    present in the two dicts, the max of the two values is used.

    Parameters
    ==========
    a, b : IntFloatDict object
        The IntFloatDicts to merge
    mask : ndarray array of dtype integer and of dimension 1
        a mask for keys to ignore: if not mask[key] the corresponding key
        is skipped in the output dictionary
    n_a, n_b : float
        n_a and n_b are weights for a and b for the merge strategy.
        They are not used in the case of a max merge.

    Returns
    =======
    out : IntFloatDict object
        The IntFloatDict resulting from the merge
    """
    # [FUNCTION DOC]
    # HELPER: merge strategy for complete (max) linkage with connectivity.
    # Behavior: copies entries of a whose key passes mask into a new IntFloatDict,
    # then for each key of b passing mask inserts it or replaces the existing value by
    # the max of both values. n_a and n_b are ignored.
    # Inputs: a, b IntFloatDict (neighbor node -> linkage distance); mask indexed by key,
    # nonzero for keys still active; n_a, n_b cluster sizes (unused).
    # Outputs: new IntFloatDict with the merged distances.
    # Exceptions: IndexError possible if a key exceeds mask length. Side effects: none
    # on inputs; allocates the result.
    cdef IntFloatDict out_obj = IntFloatDict.__new__(IntFloatDict)
    cdef cpp_map[intp_t, float64_t].iterator a_it = a.my_map.begin()
    cdef cpp_map[intp_t, float64_t].iterator a_end = a.my_map.end()
    cdef intp_t key
    cdef float64_t value
    # First copy a into out
    while a_it != a_end:
        key = deref(a_it).first
        if mask[key]:
            out_obj.my_map[key] = deref(a_it).second
        inc(a_it)

    # Then merge b into out
    cdef cpp_map[intp_t, float64_t].iterator out_it = out_obj.my_map.begin()
    cdef cpp_map[intp_t, float64_t].iterator out_end = out_obj.my_map.end()
    cdef cpp_map[intp_t, float64_t].iterator b_it = b.my_map.begin()
    cdef cpp_map[intp_t, float64_t].iterator b_end = b.my_map.end()
    while b_it != b_end:
        key = deref(b_it).first
        value = deref(b_it).second
        if mask[key]:
            out_it = out_obj.my_map.find(key)
            if out_it == out_end:
                # Key not found
                out_obj.my_map[key] = value
            else:
                deref(out_it).second = fmax(deref(out_it).second, value)
        inc(b_it)
    return out_obj


def average_merge(
    IntFloatDict a,
    IntFloatDict b,
    const intp_t[:] mask,
    intp_t n_a,
    intp_t n_b
):
    """Merge two IntFloatDicts with the average strategy: when the
    same key is present in the two dicts, the weighted average of the two
    values is used.

    Parameters
    ==========
    a, b : IntFloatDict object
        The IntFloatDicts to merge
    mask : ndarray array of dtype integer and of dimension 1
        a mask for keys to ignore: if not mask[key] the corresponding key
        is skipped in the output dictionary
    n_a, n_b : float
        n_a and n_b are weights for a and b for the merge strategy.
        They are used for a weighted mean.

    Returns
    =======
    out : IntFloatDict object
        The IntFloatDict resulting from the merge
    """
    # [FUNCTION DOC]
    # HELPER: merge strategy for average linkage with connectivity.
    # Behavior: as max_merge, but for keys present in both dicts the merged value is
    # the size-weighted mean (n_a*value_a + n_b*value_b)/(n_a + n_b). Keys present in
    # only one dict keep that dict's value unchanged.
    # Inputs: a, b IntFloatDict; mask of active keys; n_a, n_b cluster sizes (weights).
    # Outputs: new IntFloatDict with merged distances.
    # Exceptions: as max_merge. Side effects: allocates the result.
    cdef IntFloatDict out_obj = IntFloatDict.__new__(IntFloatDict)
    cdef cpp_map[intp_t, float64_t].iterator a_it = a.my_map.begin()
    cdef cpp_map[intp_t, float64_t].iterator a_end = a.my_map.end()
    cdef intp_t key
    cdef float64_t value
    cdef float64_t n_out = <float64_t> (n_a + n_b)
    # First copy a into out
    while a_it != a_end:
        key = deref(a_it).first
        if mask[key]:
            out_obj.my_map[key] = deref(a_it).second
        inc(a_it)

    # Then merge b into out
    cdef cpp_map[intp_t, float64_t].iterator out_it = out_obj.my_map.begin()
    cdef cpp_map[intp_t, float64_t].iterator out_end = out_obj.my_map.end()
    cdef cpp_map[intp_t, float64_t].iterator b_it = b.my_map.begin()
    cdef cpp_map[intp_t, float64_t].iterator b_end = b.my_map.end()
    while b_it != b_end:
        key = deref(b_it).first
        value = deref(b_it).second
        if mask[key]:
            out_it = out_obj.my_map.find(key)
            if out_it == out_end:
                # Key not found
                out_obj.my_map[key] = value
            else:
                deref(out_it).second = (n_a * deref(out_it).second
                                        + n_b * value) / n_out
        inc(b_it)
    return out_obj


###############################################################################
# An edge object for fast comparisons

cdef class WeightedEdge:
    # [CLASS DOC]
    # Small comparable record (weight, a, b) representing an edge between nodes a and
    # b; instances are ordered by weight only so they can be pushed on heapq heaps in
    # linkage_tree. Attributes are public and writable.
    cdef public intp_t a
    cdef public intp_t b
    cdef public float64_t weight

    def __init__(self, float64_t weight, intp_t a, intp_t b):
        # [FUNCTION DOC]
        # Stores weight and the two endpoint ids. Inputs: weight float, a and b node ids.
        # Outputs: none. Exceptions: TypeError on non-numeric arguments.
        self.weight = weight
        self.a = a
        self.b = b

    def __richcmp__(self, WeightedEdge other, int op):
        # [FUNCTION DOC]
        # Rich comparison: compares self.weight with other.weight according to op
        # (0 <, 1 <=, 2 ==, 3 !=, 4 >, 5 >=) and returns the boolean result; endpoints
        # are ignored, so edges of equal weight compare equal.
        # Exceptions: TypeError if other is not a WeightedEdge (argument typing).
        """Cython-specific comparison method.

        op is the comparison code::
            <   0
            ==  2
            >   4
            <=  1
            !=  3
            >=  5
        """
        if op == 0:
            return self.weight < other.weight
        elif op == 1:
            return self.weight <= other.weight
        elif op == 2:
            return self.weight == other.weight
        elif op == 3:
            return self.weight != other.weight
        elif op == 4:
            return self.weight > other.weight
        elif op == 5:
            return self.weight >= other.weight

    def __repr__(self):
        # [FUNCTION DOC]
        # Returns a string "ClassName(weight=..., a=..., b=...)". No side effects.
        return "%s(weight=%f, a=%i, b=%i)" % (self.__class__.__name__,
                                              self.weight,
                                              self.a, self.b)


################################################################################
# Efficient labelling/conversion of MSTs to single linkage hierarchies

cdef class UnionFind(object):
    # [CLASS DOC]
    # Union-find over the 2N-1 nodes of a single linkage dendrogram (N leaves). Each
    # union creates a new parent node labeled next_label (N, N+1, ...), records the
    # merged size, and links both roots to it. parent entries are -1 for roots;
    # size holds 1 for leaves and the merged size for internal nodes.

    def __init__(self, N):
        # [FUNCTION DOC]
        # Allocates parent (2N-1, all -1), size (N ones followed by N-1 zeros) and sets
        # next_label to N. Inputs: N number of leaves (>= 1). Exceptions: numpy errors for
        # invalid N. Side effects: allocation.
        self.parent = np.full(2 * N - 1, -1., dtype=np.intp, order='C')
        self.next_label = N
        self.size = np.hstack((np.ones(N, dtype=np.intp),
                               np.zeros(N - 1, dtype=np.intp)))

    cdef void union(self, intp_t m, intp_t n) noexcept:
        # [FUNCTION DOC]
        # Behavior: assigns the new node id next_label as parent of roots m and n, stores
        # its size as the sum of their sizes, and increments next_label.
        # Inputs: m, n current root ids (callers must pass roots, from fast_find).
        # Outputs: none. Exceptions: none (noexcept; errors would be printed, not raised).
        # Side effects: mutates parent, size, next_label.
        self.parent[m] = self.next_label
        self.parent[n] = self.next_label
        self.size[self.next_label] = self.size[m] + self.size[n]
        self.next_label += 1
        return

    @cython.wraparound(True)
    cdef intp_t fast_find(self, intp_t n) noexcept:
        # [FUNCTION DOC]
        # Behavior: walks parent links from n up to the root (parent -1), then a second
        # pass re-links every node on the path directly to the root (path compression).
        # Inputs: n node id. Outputs: root id.
        # Exceptions: none (noexcept). Side effects: mutates parent for path compression.
        cdef intp_t p
        p = n
        # find the highest node in the linkage graph so far
        while self.parent[n] != -1:
            n = self.parent[n]
        # provide a shortcut up to the highest node
        while self.parent[p] != n:
            p, self.parent[p] = self.parent[p], n
        return n


def _single_linkage_label(const float64_t[:, :] L):
    """
    Convert a linkage array or MST to a tree by labelling clusters at merges.
    This is done by using a Union find structure to keep track of merges
    efficiently. This is the private version of the function that assumes that
    ``L`` has been properly validated. See ``single_linkage_label`` for the
    user facing version of this function.

    Parameters
    ----------
    L: array of shape (n_samples - 1, 3)
        The linkage array or MST where each row specifies two samples
        to be merged and a distance or weight at which the merge occurs. This
         array is assumed to be sorted by the distance/weight.

    Returns
    -------
    A tree in the format used by scipy.cluster.hierarchy.
    """
    # [FUNCTION DOC]
    # HELPER: unchecked implementation behind single_linkage_label (also used by HDBSCAN-style code paths).
    # Behavior: for each MST edge in order, finds the current clusters of both
    # endpoints with UnionFind.fast_find, writes a linkage row (left cluster, right
    # cluster, distance, merged size) and unions them.
    # Inputs: L (n_samples-1, 3) float64 rows (sample a, sample b, distance), assumed
    # sorted by distance and to form a spanning tree on n_samples points.
    # Outputs: float64 array (n_samples-1, 4) in scipy linkage format.
    # Exceptions: none explicit; invalid indices can produce wrong results or
    # IndexError. Side effects: allocation only.

    cdef float64_t[:, ::1] result_arr

    cdef intp_t left, left_cluster, right, right_cluster, index
    cdef float64_t delta

    result_arr = np.zeros((L.shape[0], 4), dtype=np.float64)
    U = UnionFind(L.shape[0] + 1)

    for index in range(L.shape[0]):

        left = <intp_t> L[index, 0]
        right = <intp_t> L[index, 1]
        delta = L[index, 2]

        left_cluster = U.fast_find(left)
        right_cluster = U.fast_find(right)

        result_arr[index][0] = left_cluster
        result_arr[index][1] = right_cluster
        result_arr[index][2] = delta
        result_arr[index][3] = U.size[left_cluster] + U.size[right_cluster]

        U.union(left_cluster, right_cluster)

    return np.asarray(result_arr)


@cython.wraparound(True)
def single_linkage_label(L):
    """
    Convert a linkage array or MST to a tree by labelling clusters at merges.
    This is done by using a Union find structure to keep track of merges
    efficiently.

    Parameters
    ----------
    L: array of shape (n_samples - 1, 3)
        The linkage array or MST where each row specifies two samples
        to be merged and a distance or weight at which the merge occurs. This
         array is assumed to be sorted by the distance/weight.

    Returns
    -------
    A tree in the format used by scipy.cluster.hierarchy.
    """
    # [FUNCTION DOC]
    # Behavior: validates that the endpoint columns are within range and that the
    # distance column is non-decreasing, then delegates to _single_linkage_label.
    # Inputs: L (n_samples-1, 3) array-like of MST edges.
    # Outputs: scipy-style linkage matrix (n_samples-1, 4).
    # Exceptions: ValueError for out-of-range endpoints or unsorted weights.
    # Side effects: none.
    # KNOWN ISSUE: the upper bound check allows endpoint values up to 2*L.shape[0],
    # although valid sample indices are < L.shape[0] + 1; values in between pass
    # validation but index internal dendrogram nodes in UnionFind.
    # Validate L
    if L[:, :2].min() < 0 or L[:, :2].max() >= 2 * L.shape[0] + 1:
        raise ValueError("Input MST array is not a validly formatted MST array")

    is_sorted = lambda x: np.all(x[:-1] <= x[1:])
    if not is_sorted(L[:, 2]):
        raise ValueError("Input MST array must be sorted by weight")

    return _single_linkage_label(L)


# Implements MST-LINKAGE-CORE from https://arxiv.org/abs/1109.2378
def mst_linkage_core(
        const float64_t [:, ::1] raw_data,
        DistanceMetric64 dist_metric):
    """
    Compute the necessary elements of a minimum spanning
    tree for computation of single linkage clustering. This
    represents the MST-LINKAGE-CORE algorithm (Figure 6) from
    :arxiv:`Daniel Mullner, "Modern hierarchical, agglomerative clustering
    algorithms" <1109.2378>`.

    In contrast to the scipy implementation is never computes
    a full distance matrix, generating distances only as they
    are needed and releasing them when no longer needed.

    Parameters
    ----------
    raw_data: array of shape (n_samples, n_features)
        The array of feature data to be clustered. Must be C-aligned

    dist_metric: DistanceMetric64
        A DistanceMetric64 object conforming to the API from
        ``sklearn.metrics._dist_metrics.pxd`` that will be
        used to compute distances.

    Returns
    -------
    mst_core_data: array of shape (n_samples, 3)
        An array providing information from which one
        can either compute an MST, or the linkage hierarchy
        very efficiently. See :arxiv:`Daniel Mullner, "Modern hierarchical,
        agglomerative clustering algorithms" <1109.2378>` algorithm
        MST-LINKAGE-CORE for more details.
    """
    # [FUNCTION DOC]
    # Behavior: Prim-style MST construction without a full distance matrix. Starting
    # at node 0, repeatedly marks the current node as in-tree, updates for every
    # out-of-tree node j its best distance to the tree with the distance to the
    # current node (via dist_metric.dist), and selects the out-of-tree node with the
    # minimum best distance as the next node. Each step emits the row (current node,
    # new node, distance). O(n^2) distance evaluations, O(n) extra memory.
    # Note the first column is the previously added node, not necessarily the tree
    # node that achieves the minimum distance; the rows are suitable for single
    # linkage after sorting by distance (not a literal MST edge list).
    # Inputs: raw_data (n_samples, n_features) C-contiguous float64; dist_metric a
    # DistanceMetric64.
    # Outputs: float64 array (n_samples-1, 3), unsorted by distance.
    # Exceptions: errors propagated from dist_metric.dist. Side effects: allocation; holds the GIL.
    cdef:
        intp_t n_samples = raw_data.shape[0]
        uint8_t[:] in_tree = np.zeros(n_samples, dtype=bool)
        float64_t[:, ::1] result = np.zeros((n_samples - 1, 3))

        intp_t current_node = 0
        intp_t new_node
        intp_t i
        intp_t j
        intp_t num_features = raw_data.shape[1]

        float64_t right_value
        float64_t left_value
        float64_t new_distance

        float64_t[:] current_distances = np.full(n_samples, INFINITY)

    for i in range(n_samples - 1):

        in_tree[current_node] = 1

        new_distance = INFINITY
        new_node = 0

        for j in range(n_samples):
            if in_tree[j]:
                continue

            right_value = current_distances[j]
            left_value = dist_metric.dist(&raw_data[current_node, 0],
                                          &raw_data[j, 0],
                                          num_features)

            if left_value < right_value:
                current_distances[j] = left_value

            if current_distances[j] < new_distance:
                new_distance = current_distances[j]
                new_node = j

        result[i, 0] = current_node
        result[i, 1] = new_node
        result[i, 2] = new_distance
        current_node = new_node

    return np.array(result)
