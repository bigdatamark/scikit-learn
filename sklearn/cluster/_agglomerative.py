# [MODULE DOC]
# Hierarchical agglomerative clustering: tree builders (ward_tree, linkage_tree for
# complete/average/single linkage, with or without a connectivity graph), the tree
# cutting helper _hc_cut, and the AgglomerativeClustering and FeatureAgglomeration
# estimators. Unstructured trees are delegated to scipy.cluster.hierarchy (or a
# Cython MST for single linkage); structured trees are built here with a heap of
# candidate merges restricted to connected nodes. Node ids follow the scipy
# convention: leaves are 0..n_samples-1 and the merge at step k creates node
# n_samples + k. _TREE_BUILDERS maps linkage names to builders.
"""Hierarchical Agglomerative Clustering

These routines perform some hierarchical agglomerative clustering of some
input data.
"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import warnings
from heapq import heapify, heappop, heappush, heappushpop
from numbers import Integral, Real

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components

from sklearn.base import (
    BaseEstimator,
    ClassNamePrefixFeaturesOutMixin,
    ClusterMixin,
    _fit_context,
)
from sklearn.cluster import (
    _hierarchical_fast as _hierarchical,
)
from sklearn.cluster._feature_agglomeration import AgglomerationTransform
from sklearn.metrics import DistanceMetric
from sklearn.metrics._dist_metrics import METRIC_MAPPING64
from sklearn.metrics.pairwise import _VALID_METRICS, paired_distances
from sklearn.utils import check_array
from sklearn.utils._fast_dict import IntFloatDict
from sklearn.utils._param_validation import (
    HasMethods,
    Interval,
    StrOptions,
    validate_params,
)
from sklearn.utils._sparse import _align_api_if_sparse
from sklearn.utils.graph import _fix_connected_components
from sklearn.utils.validation import check_memory, validate_data

###############################################################################
# For non fully-connected graphs


def _fix_connectivity(X, connectivity, affinity):
    # [FUNCTION DOC]
    # HELPER: Validates and prepares a connectivity graph for structured clustering:
    # checks it is (n_samples, n_samples), symmetrizes it by adding its transpose,
    # converts it to LIL, and if it has several connected components warns
    # (UserWarning) and adds edges between components via _fix_connected_components
    # using the given metric.
    # Inputs: X - ndarray (n_samples, n_features); connectivity - dense or sparse
    # (n_samples, n_samples); affinity - metric name or callable.
    # Outputs: (connectivity as LIL sparse, aligned via _align_api_if_sparse; original
    # number of connected components).
    # Exceptions: ValueError on wrong shape.
    """
    Fixes the connectivity matrix.

    The different steps are:

    - copies it
    - makes it symmetric
    - converts it to LIL if necessary
    - completes it if necessary.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Feature matrix representing `n_samples` samples to be clustered.

    connectivity : sparse matrix, default=None
        Connectivity matrix. Defines for each sample the neighboring samples
        following a given structure of the data. The matrix is assumed to
        be symmetric and only the upper triangular half is used.
        Default is `None`, i.e, the Ward algorithm is unstructured.

    affinity : {"euclidean", "precomputed"}, default="euclidean"
        Which affinity to use. At the moment `precomputed` and
        ``euclidean`` are supported. `euclidean` uses the
        negative squared Euclidean distance between points.

    Returns
    -------
    connectivity : sparse matrix
        The fixed connectivity matrix.

    n_connected_components : int
        The number of connected components in the graph.
    """
    n_samples = X.shape[0]
    if connectivity.shape[0] != n_samples or connectivity.shape[1] != n_samples:
        raise ValueError(
            "Wrong shape for connectivity matrix: %s when X is %s"
            % (connectivity.shape, X.shape)
        )

    # Make the connectivity matrix symmetric:
    connectivity = connectivity + connectivity.T

    # Convert connectivity matrix to LIL
    if not sparse.issparse(connectivity):
        connectivity = sparse.lil_array(connectivity)

    # `connectivity` is a sparse matrix at this point
    if connectivity.format != "lil":
        connectivity = connectivity.tolil()

    # Compute the number of nodes
    n_connected_components, labels = connected_components(connectivity)

    if n_connected_components > 1:
        warnings.warn(
            "the number of connected components of the "
            "connectivity matrix is %d > 1. Completing it to avoid "
            "stopping the tree early." % n_connected_components,
            stacklevel=2,
        )
        # XXX: Can we do without completing the matrix?
        connectivity = _fix_connected_components(
            X=X,
            graph=connectivity,
            n_connected_components=n_connected_components,
            component_labels=labels,
            metric=affinity,
            mode="connectivity",
        )

    return _align_api_if_sparse(connectivity), n_connected_components


def _single_linkage_tree(
    connectivity,
    n_samples,
    n_nodes,
    n_clusters,
    n_connected_components,
    return_distance,
):
    # [FUNCTION DOC]
    # HELPER: Single-linkage tree on a sparse distance graph: casts to float64, replaces
    # explicit zero distances by machine epsilon so the MST keeps them, computes the
    # minimum spanning tree, restores those zeros, sorts edges by weight (stable), and
    # converts the edge list to a scipy-style linkage via _single_linkage_label. Then
    # fills parent for each merge node below n_nodes.
    # Inputs: connectivity - sparse COO distance graph (data modified in place);
    # n_samples, n_nodes - ints; n_clusters - int or None; n_connected_components - int;
    # return_distance - bool.
    # Outputs: (children, n_connected_components, n_leaves, parent[, distances]),
    # children int (n_samples-1, 2), parent (n_nodes,).
    # KNOWN ISSUE: unlike the other structured builders, children and distances always
    # describe the full tree even when n_clusters truncates n_nodes; only parent is
    # truncated. Genuine distances equal to epsilon are also turned into 0.
    """
    Perform single linkage clustering on sparse data via the minimum
    spanning tree from scipy.sparse.csgraph, then using union-find to label.
    The parent array is then generated by walking through the tree.
    """
    from scipy.sparse.csgraph import minimum_spanning_tree

    # explicitly cast connectivity to ensure safety
    connectivity = connectivity.astype(np.float64, copy=False)

    # Ensure zero distances aren't ignored by setting them to "epsilon"
    epsilon_value = np.finfo(dtype=connectivity.data.dtype).eps
    connectivity.data[connectivity.data == 0] = epsilon_value

    # Use scipy.sparse.csgraph to generate a minimum spanning tree
    mst = minimum_spanning_tree(connectivity.tocsr())

    # Convert the graph to scipy.cluster.hierarchy array format
    mst = mst.tocoo()

    # Undo the epsilon values
    mst.data[mst.data == epsilon_value] = 0

    mst_array = np.vstack([mst.row, mst.col, mst.data]).T

    # Sort edges of the min_spanning_tree by weight
    mst_array = mst_array[np.argsort(mst_array.T[2], kind="mergesort"), :]

    # Convert edge list into standard hierarchical clustering format
    single_linkage_tree = _hierarchical._single_linkage_label(mst_array)
    children_ = single_linkage_tree[:, :2].astype(int)

    # Compute parents
    parent = np.arange(n_nodes, dtype=np.intp)
    for i, (left, right) in enumerate(children_, n_samples):
        if n_clusters is not None and i >= n_nodes:
            break
        if left < n_nodes:
            parent[left] = i
        if right < n_nodes:
            parent[right] = i

    if return_distance:
        distances = single_linkage_tree[:, 2]
        return children_, n_connected_components, n_samples, parent, distances
    return children_, n_connected_components, n_samples, parent


###############################################################################
# Hierarchical tree building functions


@validate_params(
    {
        "X": ["array-like"],
        "connectivity": ["array-like", "sparse matrix", None],
        "n_clusters": [Interval(Integral, 1, None, closed="left"), None],
        "return_distance": ["boolean"],
    },
    prefer_skip_nested_validation=True,
)
def ward_tree(X, *, connectivity=None, n_clusters=None, return_distance=False):
    # [FUNCTION DOC]
    # Builds a Ward tree. Without connectivity: warns if n_clusters is given (full tree is
    # built anyway), calls scipy hierarchy.ward on a writeable X and returns its merges.
    # With connectivity: fixes the graph, sets n_nodes = 2n-1 (or 2n-n_clusters to stop
    # early), stores for every node the count (moments_1) and feature sum (moments_2),
    # computes Ward inertia for each connected pair (upper triangle) and pushes them on a
    # heap. For each new node k: pop until both endpoints are unused, record the merge,
    # sum the moments, collect the current top-level ancestors of both children's
    # neighbours with _get_parents, link them to k, compute their Ward distance to k and
    # push them. Children are reversed per pair for consistency with scipy.
    # Inputs: X - array-like (n_samples, n_features), 1-D reshaped to a column;
    # connectivity - sparse/array or None; n_clusters - int or None; return_distance.
    # Outputs: (children (n_merges, 2), n_connected_components, n_leaves, parents or None
    # [, distances]). Structured distances are sqrt(2 * inertia) to match scipy.
    # Exceptions: ValueError if n_clusters > n_samples (structured); shape error from
    # _fix_connectivity.
    # Side effects: warnings (partial build, multiple components).
    """Ward clustering based on a Feature matrix.

    Recursively merges the pair of clusters that minimally increases
    within-cluster variance.

    The inertia matrix uses a Heapq-based representation.

    This is the structured version, that takes into account some topological
    structure between samples.

    Read more in the :ref:`User Guide <hierarchical_clustering>`.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Feature matrix representing `n_samples` samples to be clustered.

    connectivity : {array-like, sparse matrix}, default=None
        Connectivity matrix. Defines for each sample the neighboring samples
        following a given structure of the data. The matrix is assumed to
        be symmetric and only the upper triangular half is used.
        Default is None, i.e, the Ward algorithm is unstructured.

    n_clusters : int, default=None
        `n_clusters` should be less than `n_samples`.  Stop early the
        construction of the tree at `n_clusters.` This is useful to decrease
        computation time if the number of clusters is not small compared to the
        number of samples. In this case, the complete tree is not computed, thus
        the 'children' output is of limited use, and the 'parents' output should
        rather be used. This option is valid only when specifying a connectivity
        matrix.

    return_distance : bool, default=False
        If `True`, return the distance between the clusters.

    Returns
    -------
    children : ndarray of shape (n_nodes-1, 2)
        The children of each non-leaf node. Values less than `n_samples`
        correspond to leaves of the tree which are the original samples.
        A node `i` greater than or equal to `n_samples` is a non-leaf
        node and has children `children_[i - n_samples]`. Alternatively
        at the i-th iteration, children[i][0] and children[i][1]
        are merged to form node `n_samples + i`.

    n_connected_components : int
        The number of connected components in the graph.

    n_leaves : int
        The number of leaves in the tree.

    parents : ndarray of shape (n_nodes,) or None
        The parent of each node. Only returned when a connectivity matrix
        is specified, elsewhere 'None' is returned.

    distances : ndarray of shape (n_nodes-1,)
        Only returned if `return_distance` is set to `True` (for compatibility).
        The distances between the centers of the nodes. `distances[i]`
        corresponds to a weighted Euclidean distance between
        the nodes `children[i, 1]` and `children[i, 2]`. If the nodes refer to
        leaves of the tree, then `distances[i]` is their unweighted Euclidean
        distance. Distances are updated in the following way
        (from scipy.hierarchy.linkage):

        The new entry :math:`d(u,v)` is computed as follows,

        .. math::

           d(u,v) = \\sqrt{\\frac{|v|+|s|}
                               {T}d(v,s)^2
                        + \\frac{|v|+|t|}
                               {T}d(v,t)^2
                        - \\frac{|v|}
                               {T}d(s,t)^2}

        where :math:`u` is the newly joined cluster consisting of
        clusters :math:`s` and :math:`t`, :math:`v` is an unused
        cluster in the forest, :math:`T=|v|+|s|+|t|`, and
        :math:`|*|` is the cardinality of its argument. This is also
        known as the incremental algorithm.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.cluster import ward_tree
    >>> X = np.array([[1, 2], [1, 4], [1, 0],
    ...               [4, 2], [4, 4], [4, 0]])
    >>> children, n_connected_components, n_leaves, parents = ward_tree(X)
    >>> children
    array([[0, 1],
           [3, 5],
           [2, 6],
           [4, 7],
           [8, 9]])
    >>> n_connected_components
    1
    >>> n_leaves
    6
    """
    X = np.asarray(X)
    if X.ndim == 1:
        X = np.reshape(X, (-1, 1))
    n_samples, n_features = X.shape

    if connectivity is None:
        from scipy.cluster import hierarchy  # imports PIL

        if n_clusters is not None:
            warnings.warn(
                (
                    "Partial build of the tree is implemented "
                    "only for structured clustering (i.e. with "
                    "explicit connectivity). The algorithm "
                    "will build the full tree and only "
                    "retain the lower branches required "
                    "for the specified number of clusters"
                ),
                stacklevel=2,
            )
        X = np.require(X, requirements="W")
        out = hierarchy.ward(X)
        children_ = out[:, :2].astype(np.intp)

        if return_distance:
            distances = out[:, 2]
            return children_, 1, n_samples, None, distances
        else:
            return children_, 1, n_samples, None

    connectivity, n_connected_components = _fix_connectivity(
        X, connectivity, affinity="euclidean"
    )
    if n_clusters is None:
        n_nodes = 2 * n_samples - 1
    else:
        if n_clusters > n_samples:
            raise ValueError(
                "Cannot provide more clusters than samples. "
                "%i n_clusters was asked, and there are %i "
                "samples." % (n_clusters, n_samples)
            )
        n_nodes = 2 * n_samples - n_clusters

    # create inertia matrix
    coord_row = []
    coord_col = []
    A = []
    for ind, row in enumerate(connectivity.rows):
        A.append(row)
        # We keep only the upper triangular for the moments
        # Generator expressions are faster than arrays on the following
        row = [i for i in row if i < ind]
        coord_row.extend(
            len(row)
            * [
                ind,
            ]
        )
        coord_col.extend(row)

    coord_row = np.array(coord_row, dtype=np.intp, order="C")
    coord_col = np.array(coord_col, dtype=np.intp, order="C")

    # build moments as a list
    moments_1 = np.zeros(n_nodes, order="C")
    moments_1[:n_samples] = 1
    moments_2 = np.zeros((n_nodes, n_features), order="C")
    moments_2[:n_samples] = X
    inertia = np.empty(len(coord_row), dtype=np.float64, order="C")
    _hierarchical.compute_ward_dist(moments_1, moments_2, coord_row, coord_col, inertia)
    inertia = list(zip(inertia, coord_row, coord_col))
    heapify(inertia)

    # prepare the main fields
    parent = np.arange(n_nodes, dtype=np.intp)
    used_node = np.ones(n_nodes, dtype=bool)
    children = []
    if return_distance:
        distances = np.empty(n_nodes - n_samples)

    not_visited = np.empty(n_nodes, dtype=bool, order="C")

    # recursive merge loop
    for k in range(n_samples, n_nodes):
        # identify the merge
        while True:
            inert, i, j = heappop(inertia)
            if used_node[i] and used_node[j]:
                break
        parent[i], parent[j] = k, k
        children.append((i, j))
        used_node[i] = used_node[j] = False
        if return_distance:  # store inertia value
            distances[k - n_samples] = inert

        # update the moments
        moments_1[k] = moments_1[i] + moments_1[j]
        moments_2[k] = moments_2[i] + moments_2[j]

        # update the structure matrix A and the inertia matrix
        coord_col = []
        not_visited.fill(1)
        not_visited[k] = 0
        _hierarchical._get_parents(A[i], coord_col, parent, not_visited)
        _hierarchical._get_parents(A[j], coord_col, parent, not_visited)
        # List comprehension is faster than a for loop
        [A[col].append(k) for col in coord_col]
        A.append(coord_col)
        coord_col = np.array(coord_col, dtype=np.intp, order="C")
        coord_row = np.empty(coord_col.shape, dtype=np.intp, order="C")
        coord_row.fill(k)
        n_additions = len(coord_row)
        ini = np.empty(n_additions, dtype=np.float64, order="C")

        _hierarchical.compute_ward_dist(moments_1, moments_2, coord_row, coord_col, ini)

        # List comprehension is faster than a for loop
        [heappush(inertia, (ini[idx], k, coord_col[idx])) for idx in range(n_additions)]

    # Separate leaves in children (empty lists up to now)
    n_leaves = n_samples
    # sort children to get consistent output with unstructured version
    children = [c[::-1] for c in children]
    children = np.array(children)  # return numpy array for efficient caching

    if return_distance:
        # 2 is scaling factor to compare w/ unstructured version
        distances = np.sqrt(2.0 * distances)
        return children, n_connected_components, n_leaves, parent, distances
    else:
        return children, n_connected_components, n_leaves, parent


# single average and complete linkage
def linkage_tree(
    X,
    connectivity=None,
    n_clusters=None,
    linkage="complete",
    affinity="euclidean",
    return_distance=False,
):
    # [FUNCTION DOC]
    # Builds a complete, average or single linkage tree. Validates linkage and rejects
    # zero rows with cosine affinity. Without connectivity: converts precomputed or
    # callable affinities to a condensed distance vector (upper triangle), maps l2 to
    # euclidean and l1/manhattan to cityblock, then uses the Cython MST for single
    # linkage with a metric known to DistanceMetric, else scipy hierarchy.linkage. With
    # connectivity: fixes the graph, drops self-loops, sets edge weights to the pairwise
    # distances of connected pairs (or reads them from precomputed X), delegates single
    # linkage to _single_linkage_tree, otherwise keeps adjacency as one IntFloatDict per
    # node and a heap of WeightedEdge; each merge pops the smallest valid edge, records
    # it, tracks cluster sizes in used_node, and merges neighbour distances with the
    # max (complete) or size-weighted mean (average) join function, pushing new edges.
    # Inputs: X - array-like features or square distances; connectivity; n_clusters;
    # linkage - "complete" (default), "average", "single"; affinity - metric string or
    # callable (default "euclidean"); return_distance.
    # Outputs: same tuple layout as ward_tree; parents None when unstructured.
    # Exceptions: ValueError for unknown linkage, cosine with zero vectors, non-square
    # precomputed matrix; AssertionError if n_clusters > n_samples (structured).
    # KNOWN ISSUE: the n_clusters bound in the structured path is enforced with an assert
    # (skipped under python -O) instead of a ValueError as in ward_tree. The function
    # lacks validate_params unlike ward_tree.
    """Linkage agglomerative clustering based on a Feature matrix.

    The inertia matrix uses a Heapq-based representation.

    This is the structured version, that takes into account some topological
    structure between samples.

    Read more in the :ref:`User Guide <hierarchical_clustering>`.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Feature matrix representing `n_samples` samples to be clustered.

    connectivity : sparse matrix, default=None
        Connectivity matrix. Defines for each sample the neighboring samples
        following a given structure of the data. The matrix is assumed to
        be symmetric and only the upper triangular half is used.
        Default is `None`, i.e, the Ward algorithm is unstructured.

    n_clusters : int, default=None
        Stop early the construction of the tree at `n_clusters`. This is
        useful to decrease computation time if the number of clusters is
        not small compared to the number of samples. In this case, the
        complete tree is not computed, thus the 'children' output is of
        limited use, and the 'parents' output should rather be used.
        This option is valid only when specifying a connectivity matrix.

    linkage : {"average", "complete", "single"}, default="complete"
        Which linkage criteria to use. The linkage criterion determines which
        distance to use between sets of observation.
            - "average" uses the average of the distances of each observation of
              the two sets.
            - "complete" or maximum linkage uses the maximum distances between
              all observations of the two sets.
            - "single" uses the minimum of the distances between all
              observations of the two sets.

    affinity : str or callable, default='euclidean'
        Which metric to use. Can be 'euclidean', 'manhattan', or any
        distance known to paired distance (see metric.pairwise).

    return_distance : bool, default=False
        Whether or not to return the distances between the clusters.

    Returns
    -------
    children : ndarray of shape (n_nodes-1, 2)
        The children of each non-leaf node. Values less than `n_samples`
        correspond to leaves of the tree which are the original samples.
        A node `i` greater than or equal to `n_samples` is a non-leaf
        node and has children `children_[i - n_samples]`. Alternatively
        at the i-th iteration, children[i][0] and children[i][1]
        are merged to form node `n_samples + i`.

    n_connected_components : int
        The number of connected components in the graph.

    n_leaves : int
        The number of leaves in the tree.

    parents : ndarray of shape (n_nodes, ) or None
        The parent of each node. Only returned when a connectivity matrix
        is specified, elsewhere 'None' is returned.

    distances : ndarray of shape (n_nodes-1,)
        Returned when `return_distance` is set to `True`.

        distances[i] refers to the distance between children[i][0] and
        children[i][1] when they are merged.

    See Also
    --------
    ward_tree : Hierarchical clustering with ward linkage.
    """
    X = np.asarray(X)
    if X.ndim == 1:
        X = np.reshape(X, (-1, 1))
    n_samples, n_features = X.shape

    linkage_choices = {
        "complete": _hierarchical.max_merge,
        "average": _hierarchical.average_merge,
        "single": None,
    }  # Single linkage is handled differently
    try:
        join_func = linkage_choices[linkage]
    except KeyError as e:
        raise ValueError(
            "Unknown linkage option, linkage should be one of %s, but %s was given"
            % (linkage_choices.keys(), linkage)
        ) from e

    if affinity == "cosine" and np.any(~np.any(X, axis=1)):
        raise ValueError("Cosine affinity cannot be used when X contains zero vectors")

    if connectivity is None:
        from scipy.cluster import hierarchy  # imports PIL

        if n_clusters is not None:
            warnings.warn(
                (
                    "Partial build of the tree is implemented "
                    "only for structured clustering (i.e. with "
                    "explicit connectivity). The algorithm "
                    "will build the full tree and only "
                    "retain the lower branches required "
                    "for the specified number of clusters"
                ),
                stacklevel=2,
            )

        if affinity == "precomputed":
            # for the linkage function of hierarchy to work on precomputed
            # data, provide as first argument an ndarray of the shape returned
            # by sklearn.metrics.pairwise_distances.
            if X.shape[0] != X.shape[1]:
                raise ValueError(
                    f"Distance matrix should be square, got matrix of shape {X.shape}"
                )
            i, j = np.triu_indices(X.shape[0], k=1)
            X = X[i, j]
        elif affinity == "l2":
            # Translate to something understood by scipy
            affinity = "euclidean"
        elif affinity in ("l1", "manhattan"):
            affinity = "cityblock"
        elif callable(affinity):
            X = affinity(X)
            i, j = np.triu_indices(X.shape[0], k=1)
            X = X[i, j]
        if (
            linkage == "single"
            and affinity != "precomputed"
            and not callable(affinity)
            and affinity in METRIC_MAPPING64
        ):
            # We need the fast cythonized metric from neighbors
            dist_metric = DistanceMetric.get_metric(affinity)

            # The Cython routines used require contiguous arrays
            X = np.ascontiguousarray(X, dtype=np.double)

            mst = _hierarchical.mst_linkage_core(X, dist_metric)
            # Sort edges of the min_spanning_tree by weight
            mst = mst[np.argsort(mst.T[2], kind="mergesort"), :]

            # Convert edge list into standard hierarchical clustering format
            out = _hierarchical.single_linkage_label(mst)
        else:
            out = hierarchy.linkage(X, method=linkage, metric=affinity)
        children_ = out[:, :2].astype(int, copy=False)

        if return_distance:
            distances = out[:, 2]
            return children_, 1, n_samples, None, distances
        return children_, 1, n_samples, None

    connectivity, n_connected_components = _fix_connectivity(
        X, connectivity, affinity=affinity
    )
    connectivity = connectivity.tocoo()
    # Put the diagonal to zero
    diag_mask = connectivity.row != connectivity.col
    connectivity.row = connectivity.row[diag_mask]
    connectivity.col = connectivity.col[diag_mask]
    connectivity.data = connectivity.data[diag_mask]
    del diag_mask

    if affinity == "precomputed":
        distances = X[connectivity.row, connectivity.col].astype(np.float64, copy=False)
    else:
        # FIXME We compute all the distances, while we could have only computed
        # the "interesting" distances
        distances = paired_distances(
            X[connectivity.row], X[connectivity.col], metric=affinity
        )
    connectivity.data = distances

    if n_clusters is None:
        n_nodes = 2 * n_samples - 1
    else:
        assert n_clusters <= n_samples
        n_nodes = 2 * n_samples - n_clusters

    if linkage == "single":
        return _single_linkage_tree(
            connectivity,
            n_samples,
            n_nodes,
            n_clusters,
            n_connected_components,
            return_distance,
        )

    if return_distance:
        distances = np.empty(n_nodes - n_samples)
    # create inertia heap and connection matrix
    A = np.empty(n_nodes, dtype=object)
    inertia = list()

    # LIL seems to the best format to access the rows quickly,
    # without the numpy overhead of slicing CSR indices and data.
    connectivity = connectivity.tolil()
    # We are storing the graph in a list of IntFloatDict
    for ind, (data, row) in enumerate(zip(connectivity.data, connectivity.rows)):
        A[ind] = IntFloatDict(
            np.asarray(row, dtype=np.intp), np.asarray(data, dtype=np.float64)
        )
        # We keep only the upper triangular for the heap
        # Generator expressions are faster than arrays on the following
        inertia.extend(
            _hierarchical.WeightedEdge(d, ind, r) for r, d in zip(row, data) if r < ind
        )
    del connectivity

    heapify(inertia)

    # prepare the main fields
    parent = np.arange(n_nodes, dtype=np.intp)
    used_node = np.ones(n_nodes, dtype=np.intp)
    children = []

    # recursive merge loop
    for k in range(n_samples, n_nodes):
        # identify the merge
        while True:
            edge = heappop(inertia)
            if used_node[edge.a] and used_node[edge.b]:
                break
        i = edge.a
        j = edge.b

        if return_distance:
            # store distances
            distances[k - n_samples] = edge.weight

        parent[i] = parent[j] = k
        children.append((i, j))
        # Keep track of the number of elements per cluster
        n_i = used_node[i]
        n_j = used_node[j]
        used_node[k] = n_i + n_j
        used_node[i] = used_node[j] = False

        # update the structure matrix A and the inertia matrix
        # a clever 'min', or 'max' operation between A[i] and A[j]
        coord_col = join_func(A[i], A[j], used_node, n_i, n_j)
        for col, d in coord_col:
            A[col].append(k, d)
            # Here we use the information from coord_col (containing the
            # distances) to update the heap
            heappush(inertia, _hierarchical.WeightedEdge(d, k, col))
        A[k] = coord_col
        # Clear A[i] and A[j] to save memory
        A[i] = A[j] = 0

    # Separate leaves in children (empty lists up to now)
    n_leaves = n_samples

    # # return numpy array for efficient caching
    children = np.array(children)[:, ::-1]

    if return_distance:
        return children, n_connected_components, n_leaves, parent, distances
    return children, n_connected_components, n_leaves, parent


# Matching names to tree-building strategies
def _complete_linkage(*args, **kwargs):
    # [FUNCTION DOC]
    # HELPER: linkage_tree with linkage forced to "complete"; all arguments forwarded.
    kwargs["linkage"] = "complete"
    return linkage_tree(*args, **kwargs)


def _average_linkage(*args, **kwargs):
    # [FUNCTION DOC]
    # HELPER: linkage_tree with linkage forced to "average"; all arguments forwarded.
    kwargs["linkage"] = "average"
    return linkage_tree(*args, **kwargs)


def _single_linkage(*args, **kwargs):
    # [FUNCTION DOC]
    # HELPER: linkage_tree with linkage forced to "single"; all arguments forwarded.
    kwargs["linkage"] = "single"
    return linkage_tree(*args, **kwargs)


_TREE_BUILDERS = dict(
    ward=ward_tree,
    complete=_complete_linkage,
    average=_average_linkage,
    single=_single_linkage,
)

###############################################################################
# Functions for cutting hierarchical clustering tree


def _hc_cut(n_clusters, children, n_leaves):
    # [FUNCTION DOC]
    # HELPER: Cuts a full merge tree into n_clusters flat clusters: starting from the
    # root, repeatedly replace the highest-id (latest-merged) node in a max-heap of
    # current cluster roots by its two children, n_clusters-1 times; then label all
    # leaves under the i-th remaining root with i.
    # Inputs: n_clusters - int; children - int array (n_leaves-1, 2); n_leaves - int.
    # Outputs: intp array (n_leaves,) of labels in 0..n_clusters-1 (heap order).
    # Exceptions: ValueError if n_clusters > n_leaves.
    """Function cutting the ward tree for a given number of clusters.

    Parameters
    ----------
    n_clusters : int or ndarray
        The number of clusters to form.

    children : ndarray of shape (n_nodes-1, 2)
        The children of each non-leaf node. Values less than `n_samples`
        correspond to leaves of the tree which are the original samples.
        A node `i` greater than or equal to `n_samples` is a non-leaf
        node and has children `children_[i - n_samples]`. Alternatively
        at the i-th iteration, children[i][0] and children[i][1]
        are merged to form node `n_samples + i`.

    n_leaves : int
        Number of leaves of the tree.

    Returns
    -------
    labels : array [n_samples]
        Cluster labels for each point.
    """
    if n_clusters > n_leaves:
        raise ValueError(
            "Cannot extract more clusters than samples: "
            f"{n_clusters} clusters were given for a tree with {n_leaves} leaves."
        )
    # In this function, we store nodes as a heap to avoid recomputing
    # the max of the nodes: the first element is always the smallest
    # We use negated indices as heaps work on smallest elements, and we
    # are interested in largest elements
    # children[-1] is the root of the tree
    nodes = [-(max(children[-1]) + 1)]
    for _ in range(n_clusters - 1):
        # As we have a heap, nodes[0] is the smallest element
        these_children = children[-nodes[0] - n_leaves]
        # Insert the 2 children and remove the largest node
        heappush(nodes, -these_children[0])
        heappushpop(nodes, -these_children[1])
    label = np.zeros(n_leaves, dtype=np.intp)
    for i, node in enumerate(nodes):
        label[_hierarchical._hc_get_descendent(-node, children, n_leaves)] = i
    return label


###############################################################################


class AgglomerativeClustering(ClusterMixin, BaseEstimator):
    # [CLASS DOC]
    # Estimator for bottom-up clustering of samples using one of the tree builders,
    # optionally restricted by a connectivity graph and cached through memory. Either
    # n_clusters or distance_threshold selects where the tree is cut.
    """
    Agglomerative Clustering.

    Recursively merges pair of clusters of sample data; uses linkage distance.

    Read more in the :ref:`User Guide <hierarchical_clustering>`.

    Parameters
    ----------
    n_clusters : int or None, default=2
        The number of clusters to find. It must be ``None`` if
        ``distance_threshold`` is not ``None``.

    metric : str or callable, default="euclidean"
        Metric used to compute the linkage. Can be "euclidean", "l1", "l2",
        "manhattan", "cosine", or "precomputed". If linkage is "ward", only
        "euclidean" and "l2" are accepted. If "precomputed", a distance matrix is needed
        as input for the fit method. If connectivity is None, linkage is
        "single" and metric is not "precomputed" any valid pairwise distance
        metric can be assigned.

        For an example of agglomerative clustering with different metrics, see
        :ref:`sphx_glr_auto_examples_cluster_plot_agglomerative_clustering_metrics.py`.

        .. versionadded:: 1.2

    memory : str or object with the joblib.Memory interface, default=None
        Used to cache the output of the computation of the tree.
        By default, no caching is done. If a string is given, it is the
        path to the caching directory.

    connectivity : array-like, sparse matrix, or callable, default=None
        Connectivity matrix. Defines for each sample the neighboring
        samples following a given structure of the data.
        This can be a connectivity matrix itself or a callable that transforms
        the data into a connectivity matrix, such as derived from
        `kneighbors_graph`. Default is ``None``, i.e, the
        hierarchical clustering algorithm is unstructured.

        For an example of connectivity matrix using
        :class:`~sklearn.neighbors.kneighbors_graph`, see
        :ref:`sphx_glr_auto_examples_cluster_plot_ward_structured_vs_unstructured.py`.

    compute_full_tree : 'auto' or bool, default='auto'
        Stop early the construction of the tree at ``n_clusters``. This is
        useful to decrease computation time if the number of clusters is not
        small compared to the number of samples. This option is useful only
        when specifying a connectivity matrix. Note also that when varying the
        number of clusters and using caching, it may be advantageous to compute
        the full tree. It must be ``True`` if ``distance_threshold`` is not
        ``None``. By default `compute_full_tree` is "auto", which is equivalent
        to `True` when `distance_threshold` is not `None` or that `n_clusters`
        is inferior to the maximum between 100 or `0.02 * n_samples`.
        Otherwise, "auto" is equivalent to `False`.

    linkage : {'ward', 'complete', 'average', 'single'}, default='ward'
        Which linkage criterion to use. The linkage criterion determines which
        distance to use between sets of observation. The algorithm will merge
        the pairs of cluster that minimize this criterion.

        - 'ward' minimizes the variance of the clusters being merged.
        - 'average' uses the average of the distances of each observation of
          the two sets.
        - 'complete' or 'maximum' linkage uses the maximum distances between
          all observations of the two sets.
        - 'single' uses the minimum of the distances between all observations
          of the two sets.

        .. versionadded:: 0.20
            Added the 'single' option

        For examples comparing different `linkage` criteria, see
        :ref:`sphx_glr_auto_examples_cluster_plot_linkage_comparison.py`.

    distance_threshold : float, default=None
        The linkage distance threshold at or above which clusters will not be
        merged. If not ``None``, ``n_clusters`` must be ``None`` and
        ``compute_full_tree`` must be ``True``.

        .. versionadded:: 0.21

    compute_distances : bool, default=False
        Computes distances between clusters even if `distance_threshold` is not
        used. This can be used to make dendrogram visualization, but introduces
        a computational and memory overhead.

        .. versionadded:: 0.24

        For an example of dendrogram visualization, see
        :ref:`sphx_glr_auto_examples_cluster_plot_agglomerative_dendrogram.py`.

    Attributes
    ----------
    n_clusters_ : int
        The number of clusters found by the algorithm. If
        ``distance_threshold=None``, it will be equal to the given
        ``n_clusters``.

    labels_ : ndarray of shape (n_samples)
        Cluster labels for each point.

    n_leaves_ : int
        Number of leaves in the hierarchical tree.

    n_connected_components_ : int
        The estimated number of connected components in the graph.

        .. versionadded:: 0.21
            ``n_connected_components_`` was added to replace ``n_components_``.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    children_ : array-like of shape (n_samples-1, 2)
        The children of each non-leaf node. Values less than `n_samples`
        correspond to leaves of the tree which are the original samples.
        A node `i` greater than or equal to `n_samples` is a non-leaf
        node and has children `children_[i - n_samples]`. Alternatively
        at the i-th iteration, children[i][0] and children[i][1]
        are merged to form node `n_samples + i`.

    distances_ : array-like of shape (n_nodes-1,)
        Distances between nodes in the corresponding place in `children_`.
        Only computed if `distance_threshold` is used or `compute_distances`
        is set to `True`.

    See Also
    --------
    FeatureAgglomeration : Agglomerative clustering but for features instead of
        samples.
    ward_tree : Hierarchical clustering with ward linkage.

    Examples
    --------
    >>> from sklearn.cluster import AgglomerativeClustering
    >>> import numpy as np
    >>> X = np.array([[1, 2], [1, 4], [1, 0],
    ...               [4, 2], [4, 4], [4, 0]])
    >>> clustering = AgglomerativeClustering().fit(X)
    >>> clustering
    AgglomerativeClustering()
    >>> clustering.labels_
    array([1, 1, 1, 0, 0, 0])

    For a comparison of Agglomerative clustering with other clustering algorithms, see
    :ref:`sphx_glr_auto_examples_cluster_plot_cluster_comparison.py`
    """

    _parameter_constraints: dict = {
        "n_clusters": [Interval(Integral, 1, None, closed="left"), None],
        "metric": [
            StrOptions(set(_VALID_METRICS) | {"precomputed"}),
            callable,
        ],
        "memory": [str, HasMethods("cache"), None],
        "connectivity": ["array-like", "sparse matrix", callable, None],
        "compute_full_tree": [StrOptions({"auto"}), "boolean"],
        "linkage": [StrOptions(set(_TREE_BUILDERS.keys()))],
        "distance_threshold": [Interval(Real, 0, None, closed="left"), None],
        "compute_distances": ["boolean"],
    }

    def __init__(
        self,
        n_clusters=2,
        *,
        metric="euclidean",
        memory=None,
        connectivity=None,
        compute_full_tree="auto",
        linkage="ward",
        distance_threshold=None,
        compute_distances=False,
    ):
        # [FUNCTION DOC]
        # Stores the hyper-parameters unchanged.
        self.n_clusters = n_clusters
        self.distance_threshold = distance_threshold
        self.memory = memory
        self.connectivity = connectivity
        self.compute_full_tree = compute_full_tree
        self.linkage = linkage
        self.metric = metric
        self.compute_distances = compute_distances

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None):
        # [FUNCTION DOC]
        # Validates X (at least 2 samples) and delegates to _fit.
        # Inputs: X - (n_samples, n_features) or distances if metric="precomputed"; y ignored.
        # Outputs: self.
        """Fit the hierarchical clustering from features, or distance matrix.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features) or \
                (n_samples, n_samples)
            Training instances to cluster, or distances between instances if
            ``metric='precomputed'``.

        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        self : object
            Returns the fitted instance.
        """
        X = validate_data(self, X, ensure_min_samples=2)
        return self._fit(X)

    def _fit(self, X):
        # [FUNCTION DOC]
        # Fits on already-validated X. Checks that exactly one of n_clusters and
        # distance_threshold is set, that compute_full_tree is truthy with a threshold, and
        # that ward uses euclidean/l2. Evaluates a callable connectivity on X and validates
        # it. compute_full_tree is forced True without connectivity; "auto" becomes True with
        # a threshold, else n_clusters < max(100, 0.02*n_samples). Calls the cached builder
        # (n_clusters only passed when not building the full tree; distances requested when
        # thresholding or compute_distances). Labels come from _hc_cut on the full tree, or
        # from the heads of the parent array (hc_get_heads) relabelled contiguously.
        # Sets: children_, n_connected_components_, n_leaves_, distances_ (if requested),
        # n_clusters_ (number of merges with distance >= threshold, plus 1, or n_clusters),
        # labels_.
        # Exceptions: ValueError for the inconsistent parameter combinations above.
        # Side effects: may write to the joblib memory cache.
        """Fit without validation

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features) or (n_samples, n_samples)
            Training instances to cluster, or distances between instances if
            ``metric='precomputed'``.

        Returns
        -------
        self : object
            Returns the fitted instance.
        """
        memory = check_memory(self.memory)

        if not ((self.n_clusters is None) ^ (self.distance_threshold is None)):
            raise ValueError(
                "Exactly one of n_clusters and "
                "distance_threshold has to be set, and the other "
                "needs to be None."
            )

        if self.distance_threshold is not None and not self.compute_full_tree:
            raise ValueError(
                "compute_full_tree must be True if distance_threshold is set."
            )

        if self.linkage == "ward" and self.metric not in ("euclidean", "l2"):
            raise ValueError(
                f"{self.metric} was provided as metric. Ward can only "
                "work with euclidean distances (i.e. 'euclidean' and 'l2')."
            )

        tree_builder = _TREE_BUILDERS[self.linkage]

        connectivity = self.connectivity
        if self.connectivity is not None:
            if callable(self.connectivity):
                connectivity = self.connectivity(X)
            connectivity = check_array(
                connectivity, accept_sparse=["csr", "coo", "lil"]
            )

        n_samples = len(X)
        compute_full_tree = self.compute_full_tree
        if self.connectivity is None:
            compute_full_tree = True
        if compute_full_tree == "auto":
            if self.distance_threshold is not None:
                compute_full_tree = True
            else:
                # Early stopping is likely to give a speed up only for
                # a large number of clusters. The actual threshold
                # implemented here is heuristic
                compute_full_tree = self.n_clusters < max(100, 0.02 * n_samples)
        n_clusters = self.n_clusters
        if compute_full_tree:
            n_clusters = None

        # Construct the tree
        kwargs = {}
        if self.linkage != "ward":
            kwargs["linkage"] = self.linkage
            kwargs["affinity"] = self.metric

        distance_threshold = self.distance_threshold

        return_distance = (distance_threshold is not None) or self.compute_distances

        out = memory.cache(tree_builder)(
            X,
            connectivity=connectivity,
            n_clusters=n_clusters,
            return_distance=return_distance,
            **kwargs,
        )
        (self.children_, self.n_connected_components_, self.n_leaves_, parents) = out[
            :4
        ]

        if return_distance:
            self.distances_ = out[-1]

        if self.distance_threshold is not None:  # distance_threshold is used
            self.n_clusters_ = (
                np.count_nonzero(self.distances_ >= distance_threshold) + 1
            )
        else:  # n_clusters is used
            self.n_clusters_ = self.n_clusters

        # Cut the tree
        if compute_full_tree:
            self.labels_ = _hc_cut(self.n_clusters_, self.children_, self.n_leaves_)
        else:
            labels = _hierarchical.hc_get_heads(parents, copy=False)
            # copy to avoid holding a reference on the original array
            labels = np.copy(labels[:n_samples])
            # Reassign cluster numbers
            self.labels_ = np.searchsorted(np.unique(labels), labels)
        return self

    def fit_predict(self, X, y=None):
        # [FUNCTION DOC]
        # Delegates to ClusterMixin.fit_predict (fit, return labels_); overridden for docs.
        """Fit and return the result of each sample's clustering assignment.

        In addition to fitting, this method also return the result of the
        clustering assignment for each sample in the training set.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or \
                (n_samples, n_samples)
            Training instances to cluster, or distances between instances if
            ``metric='precomputed'``.

        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Cluster labels.
        """
        return super().fit_predict(X, y)


class FeatureAgglomeration(
    ClassNamePrefixFeaturesOutMixin, AgglomerationTransform, AgglomerativeClustering
):
    # [CLASS DOC]
    # Clusters features instead of samples by running AgglomerativeClustering on the
    # transposed data; transform pools features per cluster with pooling_func.
    """Agglomerate features.

    Recursively merges pair of clusters of features.

    Refer to
    :ref:`sphx_glr_auto_examples_cluster_plot_feature_agglomeration_vs_univariate_selection.py`
    for an example comparison of :class:`FeatureAgglomeration` strategy with a
    univariate feature selection strategy (based on ANOVA).

    Read more in the :ref:`User Guide <hierarchical_clustering>`.

    Parameters
    ----------
    n_clusters : int or None, default=2
        The number of clusters to find. It must be ``None`` if
        ``distance_threshold`` is not ``None``.

    metric : str or callable, default="euclidean"
        Metric used to compute the linkage. Can be "euclidean", "l1", "l2",
        "manhattan", "cosine", or "precomputed". If linkage is "ward", only
        "euclidean" and "l2" are accepted. If "precomputed", a distance matrix is needed
        as input for the fit method.

        .. versionadded:: 1.2

    memory : str or object with the joblib.Memory interface, default=None
        Used to cache the output of the computation of the tree.
        By default, no caching is done. If a string is given, it is the
        path to the caching directory.

    connectivity : array-like, sparse matrix, or callable, default=None
        Connectivity matrix. Defines for each feature the neighboring
        features following a given structure of the data.
        This can be a connectivity matrix itself or a callable that transforms
        the data into a connectivity matrix, such as derived from
        `kneighbors_graph`. Default is `None`, i.e, the
        hierarchical clustering algorithm is unstructured.

    compute_full_tree : 'auto' or bool, default='auto'
        Stop early the construction of the tree at `n_clusters`. This is useful
        to decrease computation time if the number of clusters is not small
        compared to the number of features. This option is useful only when
        specifying a connectivity matrix. Note also that when varying the
        number of clusters and using caching, it may be advantageous to compute
        the full tree. It must be ``True`` if ``distance_threshold`` is not
        ``None``. By default `compute_full_tree` is "auto", which is equivalent
        to `True` when `distance_threshold` is not `None` or that `n_clusters`
        is inferior to the maximum between 100 or `0.02 * n_samples`.
        Otherwise, "auto" is equivalent to `False`.

    linkage : {"ward", "complete", "average", "single"}, default="ward"
        Which linkage criterion to use. The linkage criterion determines which
        distance to use between sets of features. The algorithm will merge
        the pairs of cluster that minimize this criterion.

        - "ward" minimizes the variance of the clusters being merged.
        - "complete" or maximum linkage uses the maximum distances between
          all features of the two sets.
        - "average" uses the average of the distances of each feature of
          the two sets.
        - "single" uses the minimum of the distances between all features
          of the two sets.

    pooling_func : callable, default=np.mean
        This combines the values of agglomerated features into a single
        value, and should accept an array of shape [M, N] and the keyword
        argument `axis=1`, and reduce it to an array of size [M].

    distance_threshold : float, default=None
        The linkage distance threshold at or above which clusters will not be
        merged. If not ``None``, ``n_clusters`` must be ``None`` and
        ``compute_full_tree`` must be ``True``.

        .. versionadded:: 0.21

    compute_distances : bool, default=False
        Computes distances between clusters even if `distance_threshold` is not
        used. This can be used to make dendrogram visualization, but introduces
        a computational and memory overhead.

        .. versionadded:: 0.24

    Attributes
    ----------
    n_clusters_ : int
        The number of clusters found by the algorithm. If
        ``distance_threshold=None``, it will be equal to the given
        ``n_clusters``.

    labels_ : array-like of (n_features,)
        Cluster labels for each feature.

    n_leaves_ : int
        Number of leaves in the hierarchical tree.

    n_connected_components_ : int
        The estimated number of connected components in the graph.

        .. versionadded:: 0.21
            ``n_connected_components_`` was added to replace ``n_components_``.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    children_ : array-like of shape (n_nodes-1, 2)
        The children of each non-leaf node. Values less than `n_features`
        correspond to leaves of the tree which are the original samples.
        A node `i` greater than or equal to `n_features` is a non-leaf
        node and has children `children_[i - n_features]`. Alternatively
        at the i-th iteration, children[i][0] and children[i][1]
        are merged to form node `n_features + i`.

    distances_ : array-like of shape (n_nodes-1,)
        Distances between nodes in the corresponding place in `children_`.
        Only computed if `distance_threshold` is used or `compute_distances`
        is set to `True`.

    See Also
    --------
    AgglomerativeClustering : Agglomerative clustering samples instead of
        features.
    ward_tree : Hierarchical clustering with ward linkage.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn import datasets, cluster
    >>> digits = datasets.load_digits()
    >>> images = digits.images
    >>> X = np.reshape(images, (len(images), -1))
    >>> agglo = cluster.FeatureAgglomeration(n_clusters=32)
    >>> agglo.fit(X)
    FeatureAgglomeration(n_clusters=32)
    >>> X_reduced = agglo.transform(X)
    >>> X_reduced.shape
    (1797, 32)
    """

    _parameter_constraints: dict = {
        "n_clusters": [Interval(Integral, 1, None, closed="left"), None],
        "metric": [
            StrOptions(set(_VALID_METRICS) | {"precomputed"}),
            callable,
        ],
        "memory": [str, HasMethods("cache"), None],
        "connectivity": ["array-like", "sparse matrix", callable, None],
        "compute_full_tree": [StrOptions({"auto"}), "boolean"],
        "linkage": [StrOptions(set(_TREE_BUILDERS.keys()))],
        "pooling_func": [callable],
        "distance_threshold": [Interval(Real, 0, None, closed="left"), None],
        "compute_distances": ["boolean"],
    }

    def __init__(
        self,
        n_clusters=2,
        *,
        metric="euclidean",
        memory=None,
        connectivity=None,
        compute_full_tree="auto",
        linkage="ward",
        pooling_func=np.mean,
        distance_threshold=None,
        compute_distances=False,
    ):
        # [FUNCTION DOC]
        # Forwards common hyper-parameters to AgglomerativeClustering and stores pooling_func
        # (default numpy mean).
        super().__init__(
            n_clusters=n_clusters,
            memory=memory,
            connectivity=connectivity,
            compute_full_tree=compute_full_tree,
            linkage=linkage,
            metric=metric,
            distance_threshold=distance_threshold,
            compute_distances=compute_distances,
        )
        self.pooling_func = pooling_func

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None):
        # [FUNCTION DOC]
        # Validates X (at least 2 features), fits the parent clustering on X transposed (so
        # labels_ has one entry per feature) and sets _n_features_out to n_clusters_ for
        # get_feature_names_out.
        # Outputs: self.
        """Fit the hierarchical clustering on the data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data.

        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        self : object
            Returns the transformer.
        """
        X = validate_data(self, X, ensure_min_features=2)
        super()._fit(X.T)
        self._n_features_out = self.n_clusters_
        return self

    @property
    def fit_predict(self):
        # [FUNCTION DOC]
        # Property that always raises AttributeError, hiding the inherited fit_predict since
        # predicting per-sample labels makes no sense for feature clustering.
        """Fit and return the result of each sample's clustering assignment."""
        raise AttributeError
