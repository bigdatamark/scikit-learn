# [MODULE DOC]
# OPTICS (Ordering Points To Identify the Clustering Structure): the OPTICS estimator,
# the public compute_optics_graph (ordering, core distances, reachability and
# predecessors), and the two public cluster extraction methods cluster_optics_dbscan
# (fixed eps cut) and cluster_optics_xi (Xi-steep areas), plus private helpers
# implementing core distances, reachability updates and the steep-area search.
"""Ordering Points To Identify the Clustering Structure (OPTICS)

These routines execute the OPTICS algorithm, and implement various
cluster extraction methods of the ordered list.
"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import warnings
from numbers import Integral, Real

import numpy as np
from scipy.sparse import SparseEfficiencyWarning, issparse

from sklearn.base import BaseEstimator, ClusterMixin, _fit_context
from sklearn.exceptions import DataConversionWarning
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import _VALID_METRICS, PAIRWISE_BOOLEAN_FUNCTIONS
from sklearn.neighbors import NearestNeighbors, sort_graph_by_row_values
from sklearn.utils import gen_batches
from sklearn.utils._chunking import get_chunk_n_rows
from sklearn.utils._param_validation import (
    HasMethods,
    Interval,
    RealNotInt,
    StrOptions,
    validate_params,
)
from sklearn.utils.validation import check_memory, validate_data


class OPTICS(ClusterMixin, BaseEstimator):
    # [CLASS DOC]
    # Estimator combining compute_optics_graph with either Xi or DBSCAN-style extraction.
    # Hyper-parameters are stored as given and validated against _parameter_constraints
    # at fit time. Fitted attributes: ordering_, core_distances_, reachability_,
    # predecessor_, labels_, cluster_hierarchy_ (xi only), n_features_in_,
    # feature_names_in_.
    """Estimate clustering structure from vector array.

    OPTICS (Ordering Points To Identify the Clustering Structure), closely
    related to DBSCAN, finds core samples of high density and expands clusters
    from them [1]_. Unlike DBSCAN, it keeps cluster hierarchy for a variable
    neighborhood radius. Better suited for usage on large datasets than the
    current scikit-learn implementation of DBSCAN.

    Clusters are then extracted from the cluster-order using a
    DBSCAN-like method (cluster_method = 'dbscan') or an automatic
    technique proposed in [1]_ (cluster_method = 'xi').

    This implementation deviates from the original OPTICS by first performing
    k-nearest-neighborhood searches on all points to identify core sizes of
    all points (instead of computing neighbors while looping through points).
    Reachability distances to only unprocessed points are then computed, to
    construct the cluster order, similar to the original OPTICS.
    Note that we do not employ a heap to manage the expansion
    candidates, so the time complexity will be O(n^2).

    Read more in the :ref:`User Guide <optics>`.

    Parameters
    ----------
    min_samples : int > 1 or float between 0 and 1, default=5
        The number of samples in a neighborhood for a point to be considered as
        a core point. Also, up and down steep regions can't have more than
        ``min_samples`` consecutive non-steep points. Expressed as an absolute
        number or a fraction of the number of samples (rounded to be at least
        2).

    max_eps : float, default=np.inf
        The maximum distance between two samples for one to be considered as
        in the neighborhood of the other. Default value of ``np.inf`` will
        identify clusters across all scales; reducing ``max_eps`` will result
        in shorter run times.

    metric : str or callable, default='minkowski'
        Metric to use for distance computation. Any metric from scikit-learn
        or :mod:`scipy.spatial.distance` can be used.

        If `metric` is a callable function, it is called on each
        pair of instances (rows) and the resulting value recorded. The callable
        should take two arrays as input and return one value indicating the
        distance between them. This works for Scipy's metrics, but is less
        efficient than passing the metric name as a string. If metric is
        "precomputed", `X` is assumed to be a distance matrix and must be
        square.

        Valid values for metric are:

        - from scikit-learn: ['cityblock', 'cosine', 'euclidean', 'l1', 'l2',
          'manhattan']

        - from scipy.spatial.distance: ['braycurtis', 'canberra', 'chebyshev',
          'correlation', 'dice', 'hamming', 'jaccard', 'kulsinski',
          'mahalanobis', 'minkowski', 'rogerstanimoto', 'russellrao',
          'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean',
          'yule']

        Sparse matrices are only supported by scikit-learn metrics.
        See :mod:`scipy.spatial.distance` for details on these metrics.

        .. note::
           `'kulsinski'` is deprecated from SciPy 1.9 and will be removed in SciPy 1.11.

    p : float, default=2
        Parameter for the Minkowski metric from
        :class:`~sklearn.metrics.pairwise_distances`. When p = 1, this is
        equivalent to using manhattan_distance (l1), and euclidean_distance
        (l2) for p = 2. For arbitrary p, minkowski_distance (l_p) is used.

    metric_params : dict, default=None
        Additional keyword arguments for the metric function.

    cluster_method : {'xi', 'dbscan'}, default='xi'
        The extraction method used to extract clusters using the calculated
        reachability and ordering.

    eps : float, default=None
        The maximum distance between two samples for one to be considered as
        in the neighborhood of the other. By default it assumes the same value
        as ``max_eps``.
        Used only when ``cluster_method='dbscan'``.

    xi : float between 0 and 1, default=0.05
        Determines the minimum steepness on the reachability plot that
        constitutes a cluster boundary. For example, an upwards point in the
        reachability plot is defined by the ratio from one point to its
        successor being at most 1-xi.
        Used only when ``cluster_method='xi'``.

    predecessor_correction : bool, default=True
        Correct clusters according to the predecessors calculated by OPTICS
        [2]_. This parameter has minimal effect on most datasets.
        Used only when ``cluster_method='xi'``.

    min_cluster_size : int > 1 or float between 0 and 1, default=None
        Minimum number of samples in an OPTICS cluster, expressed as an
        absolute number or a fraction of the number of samples (rounded to be
        at least 2). If ``None``, the value of ``min_samples`` is used instead.
        Used only when ``cluster_method='xi'``.

    algorithm : {'auto', 'ball_tree', 'kd_tree', 'brute'}, default='auto'
        Algorithm used to compute the nearest neighbors:

        - 'ball_tree' will use :class:`~sklearn.neighbors.BallTree`.
        - 'kd_tree' will use :class:`~sklearn.neighbors.KDTree`.
        - 'brute' will use a brute-force search.
        - 'auto' (default) will attempt to decide the most appropriate
          algorithm based on the values passed to :meth:`fit` method.

        Note: fitting on sparse input will override the setting of
        this parameter, using brute force.

    leaf_size : int, default=30
        Leaf size passed to :class:`~sklearn.neighbors.BallTree` or
        :class:`~sklearn.neighbors.KDTree`. This can affect the speed of the
        construction and query, as well as the memory required to store the
        tree. The optimal value depends on the nature of the problem.

    memory : str or object with the joblib.Memory interface, default=None
        Used to cache the output of the computation of the tree.
        By default, no caching is done. If a string is given, it is the
        path to the caching directory.

    n_jobs : int, default=None
        The number of parallel jobs to run for neighbors search.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    Attributes
    ----------
    labels_ : ndarray of shape (n_samples,)
        Cluster labels for each point in the dataset given to fit().
        Noisy samples and points which are not included in a leaf cluster
        of ``cluster_hierarchy_`` are labeled as -1.

    reachability_ : ndarray of shape (n_samples,)
        Reachability distances per sample, indexed by object order. Use
        ``clust.reachability_[clust.ordering_]`` to access in cluster order.

    ordering_ : ndarray of shape (n_samples,)
        The cluster ordered list of sample indices.

    core_distances_ : ndarray of shape (n_samples,)
        Distance at which each sample becomes a core point, indexed by object
        order. Points which will never be core have a distance of inf. Use
        ``clust.core_distances_[clust.ordering_]`` to access in cluster order.

    predecessor_ : ndarray of shape (n_samples,)
        Point that a sample was reached from, indexed by object order.
        Seed points have a predecessor of -1.

    cluster_hierarchy_ : ndarray of shape (n_clusters, 2)
        The list of clusters in the form of ``[start, end]`` in each row, with
        all indices inclusive. The clusters are ordered according to
        ``(end, -start)`` (ascending) so that larger clusters encompassing
        smaller clusters come after those smaller ones. Since ``labels_`` does
        not reflect the hierarchy, usually
        ``len(cluster_hierarchy_) > np.unique(optics.labels_)``. Please also
        note that these indices are of the ``ordering_``, i.e.
        ``X[ordering_][start:end + 1]`` form a cluster.
        Only available when ``cluster_method='xi'``.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    DBSCAN : A similar clustering for a specified neighborhood radius (eps).
        Our implementation is optimized for runtime.

    References
    ----------
    .. [1] Ankerst, Mihael, Markus M. Breunig, Hans-Peter Kriegel,
       and Jörg Sander. "OPTICS: ordering points to identify the clustering
       structure." ACM SIGMOD Record 28, no. 2 (1999): 49-60.

    .. [2] Schubert, Erich, Michael Gertz.
       "Improving the Cluster Structure Extracted from OPTICS Plots." Proc. of
       the Conference "Lernen, Wissen, Daten, Analysen" (LWDA) (2018): 318-329.

    Examples
    --------
    >>> from sklearn.cluster import OPTICS
    >>> import numpy as np
    >>> X = np.array([[1, 2], [2, 5], [3, 6],
    ...               [8, 7], [8, 8], [7, 3]])
    >>> clustering = OPTICS(min_samples=2).fit(X)
    >>> clustering.labels_
    array([0, 0, 0, 1, 1, 1])

    For a more detailed example see
    :ref:`sphx_glr_auto_examples_cluster_plot_optics.py`.

    For a comparison of OPTICS with other clustering algorithms, see
    :ref:`sphx_glr_auto_examples_cluster_plot_cluster_comparison.py`
    """

    _parameter_constraints: dict = {
        "min_samples": [
            Interval(Integral, 2, None, closed="left"),
            Interval(RealNotInt, 0, 1, closed="both"),
        ],
        "max_eps": [Interval(Real, 0, None, closed="both")],
        "metric": [StrOptions(set(_VALID_METRICS) | {"precomputed"}), callable],
        "p": [Interval(Real, 1, None, closed="left")],
        "metric_params": [dict, None],
        "cluster_method": [StrOptions({"dbscan", "xi"})],
        "eps": [Interval(Real, 0, None, closed="both"), None],
        "xi": [Interval(Real, 0, 1, closed="both")],
        "predecessor_correction": ["boolean"],
        "min_cluster_size": [
            Interval(Integral, 2, None, closed="left"),
            Interval(RealNotInt, 0, 1, closed="right"),
            None,
        ],
        "algorithm": [StrOptions({"auto", "brute", "ball_tree", "kd_tree"})],
        "leaf_size": [Interval(Integral, 1, None, closed="left")],
        "memory": [str, HasMethods("cache"), None],
        "n_jobs": [Integral, None],
    }

    def __init__(
        self,
        *,
        min_samples=5,
        max_eps=np.inf,
        metric="minkowski",
        p=2,
        metric_params=None,
        cluster_method="xi",
        eps=None,
        xi=0.05,
        predecessor_correction=True,
        min_cluster_size=None,
        algorithm="auto",
        leaf_size=30,
        memory=None,
        n_jobs=None,
    ):
        # [FUNCTION DOC]
        # Stores all keyword-only hyper-parameters as attributes without validation.
        self.max_eps = max_eps
        self.min_samples = min_samples
        self.min_cluster_size = min_cluster_size
        self.algorithm = algorithm
        self.metric = metric
        self.metric_params = metric_params
        self.p = p
        self.leaf_size = leaf_size
        self.cluster_method = cluster_method
        self.eps = eps
        self.xi = xi
        self.predecessor_correction = predecessor_correction
        self.memory = memory
        self.n_jobs = n_jobs

    @_fit_context(
        # Optics.metric is not validated yet
        prefer_skip_nested_validation=False
    )
    def fit(self, X, y=None):
        # [FUNCTION DOC]
        # Steps: if the metric is a boolean pairwise metric, warn (DataConversionWarning)
        # when X is not already boolean and validate X as bool, else as float; accept CSR.
        # For a sparse precomputed matrix, copy it, write its diagonal explicitly (so each
        # point is its own neighbor, silencing SparseEfficiencyWarning) and re-sort rows by
        # value. Run compute_optics_graph through the memory cache. Then extract labels: for
        # "xi" call cluster_optics_xi and also store cluster_hierarchy_; for "dbscan" use eps
        # (defaulting to max_eps) and call cluster_optics_dbscan.
        # Inputs: X - array or sparse (n_samples, n_features) or (n_samples, n_samples) when
        # metric="precomputed"; y - ignored. Outputs: self.
        # Exceptions: ValueError if eps > max_eps for dbscan; ValueError from size checks;
        # parameter/data validation errors.
        # Side effects: sets ordering_, core_distances_, reachability_, predecessor_, labels_
        # (and cluster_hierarchy_); may write to the joblib memory cache; may warn.
        # KNOWN ISSUE: X.dtype is read before validate_data, so a list input with a boolean
        # metric raises AttributeError instead of being converted.
        """Perform OPTICS clustering.

        Extracts an ordered list of points and reachability distances, and
        performs initial clustering using ``max_eps`` distance specified at
        OPTICS object instantiation.

        Parameters
        ----------
        X : {ndarray, sparse matrix} of shape (n_samples, n_features), or \
                (n_samples, n_samples) if metric='precomputed'
            A feature array, or array of distances between samples if
            metric='precomputed'. If a sparse matrix is provided, it will be
            converted into CSR format.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : object
            Returns a fitted instance of self.
        """
        dtype = bool if self.metric in PAIRWISE_BOOLEAN_FUNCTIONS else float
        if dtype is bool and X.dtype != bool:
            msg = (
                "Data will be converted to boolean for"
                f" metric {self.metric}, to avoid this warning,"
                " you may convert the data prior to calling fit."
            )
            warnings.warn(msg, DataConversionWarning)

        X = validate_data(self, X, dtype=dtype, accept_sparse="csr")
        if self.metric == "precomputed" and issparse(X):
            X = X.copy()  # copy to avoid in-place modification
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SparseEfficiencyWarning)
                # Set each diagonal to an explicit value so each point is its
                # own neighbor
                X.setdiag(X.diagonal())
            # Sorting could be undone by .setdiag(), make sure it's sorted
            X = sort_graph_by_row_values(X, warn_when_not_sorted=False)
        memory = check_memory(self.memory)

        (
            self.ordering_,
            self.core_distances_,
            self.reachability_,
            self.predecessor_,
        ) = memory.cache(compute_optics_graph)(
            X=X,
            min_samples=self.min_samples,
            algorithm=self.algorithm,
            leaf_size=self.leaf_size,
            metric=self.metric,
            metric_params=self.metric_params,
            p=self.p,
            n_jobs=self.n_jobs,
            max_eps=self.max_eps,
        )

        # Extract clusters from the calculated orders and reachability
        if self.cluster_method == "xi":
            labels_, clusters_ = cluster_optics_xi(
                reachability=self.reachability_,
                predecessor=self.predecessor_,
                ordering=self.ordering_,
                min_samples=self.min_samples,
                min_cluster_size=self.min_cluster_size,
                xi=self.xi,
                predecessor_correction=self.predecessor_correction,
            )
            self.cluster_hierarchy_ = clusters_
        elif self.cluster_method == "dbscan":
            if self.eps is None:
                eps = self.max_eps
            else:
                eps = self.eps

            if eps > self.max_eps:
                raise ValueError(
                    "Specify an epsilon smaller than %s. Got %s." % (self.max_eps, eps)
                )

            labels_ = cluster_optics_dbscan(
                reachability=self.reachability_,
                core_distances=self.core_distances_,
                ordering=self.ordering_,
                eps=eps,
            )

        self.labels_ = labels_
        return self


def _validate_size(size, n_samples, param_name):
    # [FUNCTION DOC]
    # HELPER: Raises ValueError if size exceeds n_samples, naming param_name in the
    # message. Inputs: size - int or float fraction; n_samples - int; param_name - str.
    # Outputs: None.
    if size > n_samples:
        raise ValueError(
            "%s must be no greater than the number of samples (%d). Got %d"
            % (param_name, n_samples, size)
        )


# OPTICS helper functions
def _compute_core_distances_(X, neighbors, min_samples, working_memory):
    # [FUNCTION DOC]
    # HELPER: Computes, for each sample, the distance to its min_samples-th nearest
    # neighbor (the sample itself counts as the first), querying in row chunks sized from
    # working_memory (16 bytes per neighbor per row). Inputs: X - training data;
    # neighbors - fitted NearestNeighbors; min_samples - int; working_memory - MiB or None.
    # Outputs: float array (n_samples,).
    # KNOWN ISSUE: the docstring says never-core points get inf; this function returns
    # finite kNN distances, the inf capping happens in compute_optics_graph.
    """Compute the k-th nearest neighbor of each sample.

    Equivalent to neighbors.kneighbors(X, self.min_samples)[0][:, -1]
    but with more memory efficiency.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The data.
    neighbors : NearestNeighbors instance
        The fitted nearest neighbors estimator.
    working_memory : int, default=None
        The sought maximum memory for temporary distance matrix chunks.
        When None (default), the value of
        ``sklearn.get_config()['working_memory']`` is used.

    Returns
    -------
    core_distances : ndarray of shape (n_samples,)
        Distance at which each sample becomes a core point.
        Points which will never be core have a distance of inf.
    """
    n_samples = X.shape[0]
    core_distances = np.empty(n_samples)
    core_distances.fill(np.nan)

    chunk_n_rows = get_chunk_n_rows(
        row_bytes=16 * min_samples, max_n_rows=n_samples, working_memory=working_memory
    )
    slices = gen_batches(n_samples, chunk_n_rows)
    for sl in slices:
        core_distances[sl] = neighbors.kneighbors(X[sl], min_samples)[0][:, -1]
    return core_distances


@validate_params(
    {
        "X": [np.ndarray, "sparse matrix"],
        "min_samples": [
            Interval(Integral, 2, None, closed="left"),
            Interval(RealNotInt, 0, 1, closed="both"),
        ],
        "max_eps": [Interval(Real, 0, None, closed="both")],
        "metric": [StrOptions(set(_VALID_METRICS) | {"precomputed"}), callable],
        "p": [Interval(Real, 0, None, closed="right"), None],
        "metric_params": [dict, None],
        "algorithm": [StrOptions({"auto", "brute", "ball_tree", "kd_tree"})],
        "leaf_size": [Interval(Integral, 1, None, closed="left")],
        "n_jobs": [Integral, None],
    },
    prefer_skip_nested_validation=False,  # metric is not validated yet
)
def compute_optics_graph(
    X, *, min_samples, max_eps, metric, p, metric_params, algorithm, leaf_size, n_jobs
):
    # [FUNCTION DOC]
    # Steps: check min_samples <= n_samples; convert a fractional min_samples to
    # max(2, int(fraction * n_samples)); initialize reachability to inf and predecessor
    # to -1; fit NearestNeighbors with the given metric settings; compute core distances,
    # set those above max_eps to inf and round them to the dtype precision. Main loop,
    # n_samples times: among unprocessed points pick the one with smallest reachability
    # (lowest index on ties, including all-inf), mark it processed, append it to the
    # ordering, and if it is a core point update its unprocessed neighbors' reachability
    # with _set_reach_dist. Warn if every reachability is inf.
    # Inputs: X - ndarray or sparse; min_samples - int > 1 or fraction; max_eps - float;
    # metric, p, metric_params, algorithm, leaf_size, n_jobs - neighbor search settings.
    # Outputs: ordering (int, n_samples), core_distances_ (float), reachability_ (float,
    # first point inf), predecessor_ (int, -1 for seeds), all indexed by sample index
    # except ordering.
    # Exceptions: ValueError if min_samples > n_samples; parameter validation errors.
    # Side effects: UserWarning when all reachabilities are inf.
    # KNOWN ISSUE: p is validated here as (0, inf] while OPTICS requires p >= 1.
    """Compute the OPTICS reachability graph.

    Read more in the :ref:`User Guide <optics>`.

    Parameters
    ----------
    X : {ndarray, sparse matrix} of shape (n_samples, n_features), or \
            (n_samples, n_samples) if metric='precomputed'
        A feature array, or array of distances between samples if
        metric='precomputed'.

    min_samples : int > 1 or float between 0 and 1
        The number of samples in a neighborhood for a point to be considered
        as a core point. Expressed as an absolute number or a fraction of the
        number of samples (rounded to be at least 2).

    max_eps : float, default=np.inf
        The maximum distance between two samples for one to be considered as
        in the neighborhood of the other. Default value of ``np.inf`` will
        identify clusters across all scales; reducing ``max_eps`` will result
        in shorter run times.

    metric : str or callable, default='minkowski'
        Metric to use for distance computation. Any metric from scikit-learn
        or scipy.spatial.distance can be used.

        If metric is a callable function, it is called on each
        pair of instances (rows) and the resulting value recorded. The callable
        should take two arrays as input and return one value indicating the
        distance between them. This works for Scipy's metrics, but is less
        efficient than passing the metric name as a string. If metric is
        "precomputed", X is assumed to be a distance matrix and must be square.

        Valid values for metric are:

        - from scikit-learn: ['cityblock', 'cosine', 'euclidean', 'l1', 'l2',
          'manhattan']

        - from scipy.spatial.distance: ['braycurtis', 'canberra', 'chebyshev',
          'correlation', 'dice', 'hamming', 'jaccard', 'kulsinski',
          'mahalanobis', 'minkowski', 'rogerstanimoto', 'russellrao',
          'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean',
          'yule']

        See the documentation for scipy.spatial.distance for details on these
        metrics.

        .. note::
           `'kulsinski'` is deprecated from SciPy 1.9 and will be removed in SciPy 1.11.

    p : float, default=2
        Parameter for the Minkowski metric from
        :class:`~sklearn.metrics.pairwise_distances`. When p = 1, this is
        equivalent to using manhattan_distance (l1), and euclidean_distance
        (l2) for p = 2. For arbitrary p, minkowski_distance (l_p) is used.

    metric_params : dict, default=None
        Additional keyword arguments for the metric function.

    algorithm : {'auto', 'ball_tree', 'kd_tree', 'brute'}, default='auto'
        Algorithm used to compute the nearest neighbors:

        - 'ball_tree' will use :class:`~sklearn.neighbors.BallTree`.
        - 'kd_tree' will use :class:`~sklearn.neighbors.KDTree`.
        - 'brute' will use a brute-force search.
        - 'auto' will attempt to decide the most appropriate algorithm
          based on the values passed to `fit` method. (default)

        Note: fitting on sparse input will override the setting of
        this parameter, using brute force.

    leaf_size : int, default=30
        Leaf size passed to :class:`~sklearn.neighbors.BallTree` or
        :class:`~sklearn.neighbors.KDTree`. This can affect the speed of the
        construction and query, as well as the memory required to store the
        tree. The optimal value depends on the nature of the problem.

    n_jobs : int, default=None
        The number of parallel jobs to run for neighbors search.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    Returns
    -------
    ordering_ : array of shape (n_samples,)
        The cluster ordered list of sample indices.

    core_distances_ : array of shape (n_samples,)
        Distance at which each sample becomes a core point, indexed by object
        order. Points which will never be core have a distance of inf. Use
        ``clust.core_distances_[clust.ordering_]`` to access in cluster order.

    reachability_ : array of shape (n_samples,)
        Reachability distances per sample, indexed by object order. Use
        ``clust.reachability_[clust.ordering_]`` to access in cluster order.

    predecessor_ : array of shape (n_samples,)
        Point that a sample was reached from, indexed by object order.
        Seed points have a predecessor of -1.

    References
    ----------
    .. [1] Ankerst, Mihael, Markus M. Breunig, Hans-Peter Kriegel,
       and Jörg Sander. "OPTICS: ordering points to identify the clustering
       structure." ACM SIGMOD Record 28, no. 2 (1999): 49-60.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.cluster import compute_optics_graph
    >>> X = np.array([[1, 2], [2, 5], [3, 6],
    ...               [8, 7], [8, 8], [7, 3]])
    >>> ordering, core_distances, reachability, predecessor = compute_optics_graph(
    ...     X,
    ...     min_samples=2,
    ...     max_eps=np.inf,
    ...     metric="minkowski",
    ...     p=2,
    ...     metric_params=None,
    ...     algorithm="auto",
    ...     leaf_size=30,
    ...     n_jobs=None,
    ... )
    >>> ordering
    array([0, 1, 2, 5, 3, 4])
    >>> core_distances
    array([3.16, 1.41, 1.41, 1.        , 1.        ,
           4.12])
    >>> reachability
    array([       inf, 3.16, 1.41, 4.12, 1.        ,
           5.        ])
    >>> predecessor
    array([-1,  0,  1,  5,  3,  2])
    """
    n_samples = X.shape[0]
    _validate_size(min_samples, n_samples, "min_samples")
    if min_samples <= 1:
        min_samples = max(2, int(min_samples * n_samples))

    # Start all points as 'unprocessed' ##
    reachability_ = np.empty(n_samples)
    reachability_.fill(np.inf)
    predecessor_ = np.empty(n_samples, dtype=int)
    predecessor_.fill(-1)

    nbrs = NearestNeighbors(
        n_neighbors=min_samples,
        algorithm=algorithm,
        leaf_size=leaf_size,
        metric=metric,
        metric_params=metric_params,
        p=p,
        n_jobs=n_jobs,
    )

    nbrs.fit(X)
    # Here we first do a kNN query for each point, this differs from
    # the original OPTICS that only used epsilon range queries.
    # TODO: handle working_memory somehow?
    core_distances_ = _compute_core_distances_(
        X=X, neighbors=nbrs, min_samples=min_samples, working_memory=None
    )
    # OPTICS puts an upper limit on these, use inf for undefined.
    core_distances_[core_distances_ > max_eps] = np.inf
    np.around(
        core_distances_,
        decimals=np.finfo(core_distances_.dtype).precision,
        out=core_distances_,
    )

    # Main OPTICS loop. Not parallelizable. The order that entries are
    # written to the 'ordering_' list is important!
    # Note that this implementation is O(n^2) theoretically, but
    # supposedly with very low constant factors.
    processed = np.zeros(X.shape[0], dtype=bool)
    ordering = np.zeros(X.shape[0], dtype=int)
    for ordering_idx in range(X.shape[0]):
        # Choose next based on smallest reachability distance
        # (And prefer smaller ids on ties, possibly np.inf!)
        index = np.where(processed == 0)[0]
        point = index[np.argmin(reachability_[index])]

        processed[point] = True
        ordering[ordering_idx] = point
        if core_distances_[point] != np.inf:
            _set_reach_dist(
                core_distances_=core_distances_,
                reachability_=reachability_,
                predecessor_=predecessor_,
                point_index=point,
                processed=processed,
                X=X,
                nbrs=nbrs,
                metric=metric,
                metric_params=metric_params,
                p=p,
                max_eps=max_eps,
            )
    if np.all(np.isinf(reachability_)):
        warnings.warn(
            (
                "All reachability values are inf. Set a larger"
                " max_eps or all data will be considered outliers."
            ),
            UserWarning,
        )
    return ordering, core_distances_, reachability_, predecessor_


def _set_reach_dist(
    core_distances_,
    reachability_,
    predecessor_,
    point_index,
    processed,
    X,
    nbrs,
    metric,
    metric_params,
    p,
    max_eps,
):
    # [FUNCTION DOC]
    # HELPER: For the point point_index, finds neighbors within max_eps, keeps the
    # unprocessed ones, computes their distances to the point (reading the row of a
    # precomputed matrix, or pairwise_distances with metric_params, adding p for
    # minkowski unless already given), takes the elementwise maximum with the point's core
    # distance, rounds to dtype precision, and wherever this is strictly smaller than the
    # current reachability, overwrites reachability_ and sets predecessor_ to point_index.
    # Inputs: arrays core_distances_, reachability_, predecessor_, processed (bool);
    # X; fitted nbrs; metric settings; max_eps.
    # Outputs: None. Side effects: mutates reachability_ and predecessor_ in place.
    P = X[point_index : point_index + 1]
    # Assume that radius_neighbors is faster without distances
    # and we don't need all distances, nevertheless, this means
    # we may be doing some work twice.
    indices = nbrs.radius_neighbors(P, radius=max_eps, return_distance=False)[0]

    # Getting indices of neighbors that have not been processed
    unproc = np.compress(~np.take(processed, indices), indices)
    # Neighbors of current point are already processed.
    if not unproc.size:
        return

    # Only compute distances to unprocessed neighbors:
    if metric == "precomputed":
        dists = X[[point_index], unproc]
        if isinstance(dists, np.matrix):
            dists = np.asarray(dists)
        dists = dists.ravel()
    else:
        _params = dict() if metric_params is None else metric_params.copy()
        if metric == "minkowski" and "p" not in _params:
            # the same logic as neighbors, p is ignored if explicitly set
            # in the dict params
            _params["p"] = p
        dists = pairwise_distances(P, X[unproc], metric, n_jobs=None, **_params).ravel()

    rdists = np.maximum(dists, core_distances_[point_index])
    np.around(rdists, decimals=np.finfo(rdists.dtype).precision, out=rdists)
    improved = np.where(rdists < np.take(reachability_, unproc))
    reachability_[unproc[improved]] = rdists[improved]
    predecessor_[unproc[improved]] = point_index


@validate_params(
    {
        "reachability": [np.ndarray],
        "core_distances": [np.ndarray],
        "ordering": [np.ndarray],
        "eps": [Interval(Real, 0, None, closed="both")],
    },
    prefer_skip_nested_validation=True,
)
def cluster_optics_dbscan(*, reachability, core_distances, ordering, eps):
    # [FUNCTION DOC]
    # DBSCAN-like extraction at a given eps: a point starts a new cluster when its
    # reachability exceeds eps and its core distance is at most eps; walking the ordering,
    # labels are the running count of such starts minus one; points whose reachability
    # and core distance both exceed eps are labelled -1.
    # Inputs: reachability, core_distances - float arrays (n_samples,) by sample index;
    # ordering - int array; eps - float >= 0. Outputs: int labels (n_samples,).
    # KNOWN ISSUE: a non-core point reachable within eps that precedes every cluster
    # start in the ordering gets label -1 from the cumulative count, coinciding with
    # the noise label.
    """Perform DBSCAN extraction for an arbitrary epsilon.

    Extracting the clusters runs in linear time. Note that this results in
    ``labels_`` which are close to a :class:`~sklearn.cluster.DBSCAN` with
    similar settings and ``eps``, only if ``eps`` is close to ``max_eps``.

    Parameters
    ----------
    reachability : ndarray of shape (n_samples,)
        Reachability distances calculated by OPTICS (``reachability_``).

    core_distances : ndarray of shape (n_samples,)
        Distances at which points become core (``core_distances_``).

    ordering : ndarray of shape (n_samples,)
        OPTICS ordered point indices (``ordering_``).

    eps : float
        DBSCAN ``eps`` parameter. Must be set to < ``max_eps``. Results
        will be close to DBSCAN algorithm if ``eps`` and ``max_eps`` are close
        to one another.

    Returns
    -------
    labels_ : array of shape (n_samples,)
        The estimated labels.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.cluster import cluster_optics_dbscan, compute_optics_graph
    >>> X = np.array([[1, 2], [2, 5], [3, 6],
    ...               [8, 7], [8, 8], [7, 3]])
    >>> ordering, core_distances, reachability, predecessor = compute_optics_graph(
    ...     X,
    ...     min_samples=2,
    ...     max_eps=np.inf,
    ...     metric="minkowski",
    ...     p=2,
    ...     metric_params=None,
    ...     algorithm="auto",
    ...     leaf_size=30,
    ...     n_jobs=None,
    ... )
    >>> eps = 4.5
    >>> labels = cluster_optics_dbscan(
    ...     reachability=reachability,
    ...     core_distances=core_distances,
    ...     ordering=ordering,
    ...     eps=eps,
    ... )
    >>> labels
    array([0, 0, 0, 1, 1, 1])
    """
    n_samples = len(core_distances)
    labels = np.zeros(n_samples, dtype=int)

    far_reach = reachability > eps
    near_core = core_distances <= eps
    labels[ordering] = np.cumsum(far_reach[ordering] & near_core[ordering]) - 1
    labels[far_reach & ~near_core] = -1
    return labels


@validate_params(
    {
        "reachability": [np.ndarray],
        "predecessor": [np.ndarray],
        "ordering": [np.ndarray],
        "min_samples": [
            Interval(Integral, 2, None, closed="left"),
            Interval(RealNotInt, 0, 1, closed="both"),
        ],
        "min_cluster_size": [
            Interval(Integral, 2, None, closed="left"),
            Interval(RealNotInt, 0, 1, closed="both"),
            None,
        ],
        "xi": [Interval(Real, 0, 1, closed="both")],
        "predecessor_correction": ["boolean"],
    },
    prefer_skip_nested_validation=True,
)
def cluster_optics_xi(
    *,
    reachability,
    predecessor,
    ordering,
    min_samples,
    min_cluster_size=None,
    xi=0.05,
    predecessor_correction=True,
):
    # [FUNCTION DOC]
    # Normalizes min_samples and min_cluster_size (each checked against n_samples,
    # fractions converted to max(2, int(fraction * n_samples)), min_cluster_size defaults
    # to min_samples), runs _xi_cluster on the reachability and predecessor arrays
    # reordered by ordering, and derives flat labels with _extract_xi_labels.
    # Inputs: reachability, predecessor - arrays by sample index; ordering - int array;
    # min_samples - int > 1 or fraction; min_cluster_size - int, fraction or None;
    # xi - float in [0, 1]; predecessor_correction - bool.
    # Outputs: (labels int array (n_samples,), clusters int array (n_clusters, 2) of
    # inclusive [start, end] positions in the ordering, smaller clusters first).
    # Exceptions: ValueError when sizes exceed n_samples; parameter validation errors.
    """Automatically extract clusters according to the Xi-steep method.

    Parameters
    ----------
    reachability : ndarray of shape (n_samples,)
        Reachability distances calculated by OPTICS (`reachability_`).

    predecessor : ndarray of shape (n_samples,)
        Predecessors calculated by OPTICS.

    ordering : ndarray of shape (n_samples,)
        OPTICS ordered point indices (`ordering_`).

    min_samples : int > 1 or float between 0 and 1
        The same as the min_samples given to OPTICS. Up and down steep regions
        can't have more then ``min_samples`` consecutive non-steep points.
        Expressed as an absolute number or a fraction of the number of samples
        (rounded to be at least 2).

    min_cluster_size : int > 1 or float between 0 and 1, default=None
        Minimum number of samples in an OPTICS cluster, expressed as an
        absolute number or a fraction of the number of samples (rounded to be
        at least 2). If ``None``, the value of ``min_samples`` is used instead.

    xi : float between 0 and 1, default=0.05
        Determines the minimum steepness on the reachability plot that
        constitutes a cluster boundary. For example, an upwards point in the
        reachability plot is defined by the ratio from one point to its
        successor being at most 1-xi.

    predecessor_correction : bool, default=True
        Correct clusters based on the calculated predecessors.

    Returns
    -------
    labels : ndarray of shape (n_samples,)
        The labels assigned to samples. Points which are not included
        in any cluster are labeled as -1.

    clusters : ndarray of shape (n_clusters, 2)
        The list of clusters in the form of ``[start, end]`` in each row, with
        all indices inclusive. The clusters are ordered according to ``(end,
        -start)`` (ascending) so that larger clusters encompassing smaller
        clusters come after such nested smaller clusters. Since ``labels`` does
        not reflect the hierarchy, usually ``len(clusters) >
        np.unique(labels)``.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.cluster import cluster_optics_xi, compute_optics_graph
    >>> X = np.array([[1, 2], [2, 5], [3, 6],
    ...               [8, 7], [8, 8], [7, 3]])
    >>> ordering, core_distances, reachability, predecessor = compute_optics_graph(
    ...     X,
    ...     min_samples=2,
    ...     max_eps=np.inf,
    ...     metric="minkowski",
    ...     p=2,
    ...     metric_params=None,
    ...     algorithm="auto",
    ...     leaf_size=30,
    ...     n_jobs=None
    ... )
    >>> min_samples = 2
    >>> labels, clusters = cluster_optics_xi(
    ...     reachability=reachability,
    ...     predecessor=predecessor,
    ...     ordering=ordering,
    ...     min_samples=min_samples,
    ... )
    >>> labels
    array([0, 0, 0, 1, 1, 1])
    >>> clusters
    array([[0, 2],
           [3, 5],
           [0, 5]])
    """
    n_samples = len(reachability)
    _validate_size(min_samples, n_samples, "min_samples")
    if min_samples <= 1:
        min_samples = max(2, int(min_samples * n_samples))
    if min_cluster_size is None:
        min_cluster_size = min_samples
    _validate_size(min_cluster_size, n_samples, "min_cluster_size")
    if min_cluster_size <= 1:
        min_cluster_size = max(2, int(min_cluster_size * n_samples))

    clusters = _xi_cluster(
        reachability[ordering],
        predecessor[ordering],
        ordering,
        xi,
        min_samples,
        min_cluster_size,
        predecessor_correction,
    )
    labels = _extract_xi_labels(ordering, clusters)
    return labels, clusters


def _extend_region(steep_point, xward_point, start, min_samples):
    # [FUNCTION DOC]
    # HELPER: Starting at start, scans forward to find the end of a maximal steep area:
    # each steep point resets the counter of non-steep points and becomes the new end; a
    # point that goes in the opposite direction (xward_point) ends the scan; other points
    # are counted, and the scan stops once more than min_samples consecutive such points
    # are seen. Inputs: steep_point, xward_point - bool arrays; start - int;
    # min_samples - int. Outputs: int index of the last steep point (inclusive end).
    # KNOWN ISSUE: the docstring documents two return values (index, end) but only end
    # is returned.
    """Extend the area until it's maximal.

    It's the same function for both upward and downward reagions, depending on
    the given input parameters. Assuming:

        - steep_{upward/downward}: bool array indicating whether a point is a
          steep {upward/downward};
        - upward/downward: bool array indicating whether a point is
          upward/downward;

    To extend an upward reagion, ``steep_point=steep_upward`` and
    ``xward_point=downward`` are expected, and to extend a downward region,
    ``steep_point=steep_downward`` and ``xward_point=upward``.

    Parameters
    ----------
    steep_point : ndarray of shape (n_samples,), dtype=bool
        True if the point is steep downward (upward).

    xward_point : ndarray of shape (n_samples,), dtype=bool
        True if the point is an upward (respectively downward) point.

    start : int
        The start of the xward region.

    min_samples : int
       The same as the min_samples given to OPTICS. Up and down steep
       regions can't have more then ``min_samples`` consecutive non-steep
       points.

    Returns
    -------
    index : int
        The current index iterating over all the samples, i.e. where we are up
        to in our search.

    end : int
        The end of the region, which can be behind the index. The region
        includes the ``end`` index.
    """
    n_samples = len(steep_point)
    non_xward_points = 0
    index = start
    end = start
    # find a maximal area
    while index < n_samples:
        if steep_point[index]:
            non_xward_points = 0
            end = index
        elif not xward_point[index]:
            # it's not a steep point, but still goes up.
            non_xward_points += 1
            # region should include no more than min_samples consecutive
            # non steep xward points.
            if non_xward_points > min_samples:
                break
        else:
            return end
        index += 1
    return end


def _update_filter_sdas(sdas, mib, xi_complement, reachability_plot):
    # [FUNCTION DOC]
    # HELPER: Given the current maximum-in-between value mib, returns [] if mib is
    # infinite; otherwise keeps only steep down areas whose start reachability times
    # xi_complement is at least mib, and raises each kept area's "mib" to max(own, mib).
    # Inputs: sdas - list of dicts with start, end, mib; mib - float; xi_complement -
    # 1 - xi; reachability_plot - reachability in ordering order (with trailing inf).
    # Outputs: filtered list. Side effects: mutates the "mib" entry of kept dicts.
    """Update steep down areas (SDAs) using the new maximum in between (mib)
    value, and the given complement of xi, i.e. ``1 - xi``.
    """
    if np.isinf(mib):
        return []
    res = [
        sda for sda in sdas if mib <= reachability_plot[sda["start"]] * xi_complement
    ]
    for sda in res:
        sda["mib"] = max(sda["mib"], mib)
    return res


def _correct_predecessor(reachability_plot, predecessor_plot, ordering, s, e):
    # [FUNCTION DOC]
    # HELPER: Predecessor correction (Schubert and Gertz, Algorithm 2) on a candidate
    # cluster [s, e] in ordering positions: while s < e, accept if reachability at s
    # exceeds that at e, or if the predecessor of the point at e is among the points at
    # positions s..e-1; otherwise shrink e by one. Outputs: (s, e) when accepted, else
    # (None, None).
    """Correct for predecessors.

    Applies Algorithm 2 of [1]_.

    Input parameters are ordered by the computer OPTICS ordering.

    .. [1] Schubert, Erich, Michael Gertz.
       "Improving the Cluster Structure Extracted from OPTICS Plots." Proc. of
       the Conference "Lernen, Wissen, Daten, Analysen" (LWDA) (2018): 318-329.
    """
    while s < e:
        if reachability_plot[s] > reachability_plot[e]:
            return s, e
        p_e = predecessor_plot[e]
        for i in range(s, e):
            if p_e == ordering[i]:
                return s, e
        e -= 1
    return None, None


def _xi_cluster(
    reachability_plot,
    predecessor_plot,
    ordering,
    xi,
    min_samples,
    min_cluster_size,
    predecessor_correction,
):
    # [FUNCTION DOC]
    # HELPER: Xi-steep cluster extraction (roughly Figure 19 of the OPTICS paper, with
    # the paper's Definition 9 and 11 errata corrected). Appends inf to the reachability
    # plot; computes the ratio of each value to its successor and flags steep upward
    # (ratio <= 1 - xi), steep downward (ratio >= 1 / (1 - xi)), downward (> 1) and
    # upward (< 1). Iterates over steep indices not already consumed, maintaining mib
    # (max reachability since the last area). A steep-down start filters the SDA list,
    # extends the area and records it with mib 0. A steep-up start filters SDAs, extends
    # the up area, then for each SDA forms a candidate [SDA start, up end]: skip if the
    # value after the end times (1 - xi) is below the SDA mib; trim the start rightwards
    # or the end leftwards per Definition 11 criterion 4 depending on which side is
    # higher; optionally apply predecessor correction; reject if smaller than
    # min_cluster_size, if the start is past the SDA end, or if the end is before the up
    # start. Accepted candidates for one up area are appended in reverse so that smaller
    # nested clusters precede larger ones.
    # Inputs: reachability_plot, predecessor_plot - arrays in ordering order; ordering;
    # xi; min_samples; min_cluster_size - ints; predecessor_correction - bool.
    # Outputs: int array (n_clusters, 2) of inclusive [start, end] positions (shape (0,)
    # when empty).
    """Automatically extract clusters according to the Xi-steep method.

    This is rouphly an implementation of Figure 19 of the OPTICS paper.

    Parameters
    ----------
    reachability_plot : array-like of shape (n_samples,)
        The reachability plot, i.e. reachability ordered according to
        the calculated ordering, all computed by OPTICS.

    predecessor_plot : array-like of shape (n_samples,)
        Predecessors ordered according to the calculated ordering.

    xi : float, between 0 and 1
        Determines the minimum steepness on the reachability plot that
        constitutes a cluster boundary. For example, an upwards point in the
        reachability plot is defined by the ratio from one point to its
        successor being at most 1-xi.

    min_samples : int > 1
        The same as the min_samples given to OPTICS. Up and down steep regions
        can't have more then ``min_samples`` consecutive non-steep points.

    min_cluster_size : int > 1
        Minimum number of samples in an OPTICS cluster.

    predecessor_correction : bool
        Correct clusters based on the calculated predecessors.

    Returns
    -------
    clusters : ndarray of shape (n_clusters, 2)
        The list of clusters in the form of [start, end] in each row, with all
        indices inclusive. The clusters are ordered in a way that larger
        clusters encompassing smaller clusters come after those smaller
        clusters.
    """

    # Our implementation adds an inf to the end of reachability plot
    # this helps to find potential clusters at the end of the
    # reachability plot even if there's no upward region at the end of it.
    reachability_plot = np.hstack((reachability_plot, np.inf))

    xi_complement = 1 - xi
    sdas = []  # steep down areas, introduced in section 4.3.2 of the paper
    clusters = []
    index = 0
    mib = 0.0  # maximum in between, section 4.3.2

    # Our implementation corrects a mistake in the original
    # paper, i.e., in Definition 9 steep downward point,
    # r(p) * (1 - x1) <= r(p + 1) should be
    # r(p) * (1 - x1) >= r(p + 1)
    with np.errstate(invalid="ignore"):
        ratio = reachability_plot[:-1] / reachability_plot[1:]
        steep_upward = ratio <= xi_complement
        steep_downward = ratio >= 1 / xi_complement
        downward = ratio > 1
        upward = ratio < 1

    # the following loop is almost exactly as Figure 19 of the paper.
    # it jumps over the areas which are not either steep down or up areas
    for steep_index in iter(np.flatnonzero(steep_upward | steep_downward)):
        # just continue if steep_index has been a part of a discovered xward
        # area.
        if steep_index < index:
            continue

        mib = max(mib, np.max(reachability_plot[index : steep_index + 1]))

        # steep downward areas
        if steep_downward[steep_index]:
            sdas = _update_filter_sdas(sdas, mib, xi_complement, reachability_plot)
            D_start = steep_index
            D_end = _extend_region(steep_downward, upward, D_start, min_samples)
            D = {"start": D_start, "end": D_end, "mib": 0.0}
            sdas.append(D)
            index = D_end + 1
            mib = reachability_plot[index]

        # steep upward areas
        else:
            sdas = _update_filter_sdas(sdas, mib, xi_complement, reachability_plot)
            U_start = steep_index
            U_end = _extend_region(steep_upward, downward, U_start, min_samples)
            index = U_end + 1
            mib = reachability_plot[index]

            U_clusters = []
            for D in sdas:
                c_start = D["start"]
                c_end = U_end

                # line (**), sc2*
                if reachability_plot[c_end + 1] * xi_complement < D["mib"]:
                    continue

                # Definition 11: criterion 4
                D_max = reachability_plot[D["start"]]
                if D_max * xi_complement >= reachability_plot[c_end + 1]:
                    # Find the first index from the left side which is almost
                    # at the same level as the end of the detected cluster.
                    while (
                        reachability_plot[c_start + 1] > reachability_plot[c_end + 1]
                        and c_start < D["end"]
                    ):
                        c_start += 1
                elif reachability_plot[c_end + 1] * xi_complement >= D_max:
                    # Find the first index from the right side which is almost
                    # at the same level as the beginning of the detected
                    # cluster.
                    # Our implementation corrects a mistake in the original
                    # paper, i.e., in Definition 11 4c, r(x) < r(sD) should be
                    # r(x) > r(sD).
                    while reachability_plot[c_end - 1] > D_max and c_end > U_start:
                        c_end -= 1

                # predecessor correction
                if predecessor_correction:
                    c_start, c_end = _correct_predecessor(
                        reachability_plot, predecessor_plot, ordering, c_start, c_end
                    )
                if c_start is None:
                    continue

                # Definition 11: criterion 3.a
                if c_end - c_start + 1 < min_cluster_size:
                    continue

                # Definition 11: criterion 1
                if c_start > D["end"]:
                    continue

                # Definition 11: criterion 2
                if c_end < U_start:
                    continue

                U_clusters.append((c_start, c_end))

            # add smaller clusters first.
            U_clusters.reverse()
            clusters.extend(U_clusters)

    return np.array(clusters)


def _extract_xi_labels(ordering, clusters):
    # [FUNCTION DOC]
    # HELPER: Walks clusters (smaller first) and assigns a new label to the positions of a
    # cluster only when none of them is labelled yet, so the smallest (leaf) clusters win
    # and enclosing clusters only label otherwise-unlabelled spans; finally maps labels
    # from ordering positions back to sample indices. Inputs: ordering - int array;
    # clusters - (n_clusters, 2) array. Outputs: int labels (n_samples,), -1 for noise.
    """Extracts the labels from the clusters returned by `_xi_cluster`.
    We rely on the fact that clusters are stored
    with the smaller clusters coming before the larger ones.

    Parameters
    ----------
    ordering : array-like of shape (n_samples,)
        The ordering of points calculated by OPTICS

    clusters : array-like of shape (n_clusters, 2)
        List of clusters i.e. (start, end) tuples,
        as returned by `_xi_cluster`.

    Returns
    -------
    labels : ndarray of shape (n_samples,)
    """

    labels = np.full(len(ordering), -1, dtype=int)
    label = 0
    for c in clusters:
        if not np.any(labels[c[0] : (c[1] + 1)] != -1):
            labels[c[0] : (c[1] + 1)] = label
            label += 1
    labels[ordering] = labels.copy()
    return labels
