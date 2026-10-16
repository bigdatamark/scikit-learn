# [MODULE DOC]
# K-means clustering: k-means++ seeding (public kmeans_plusplus and private
# _kmeans_plusplus), the functional k_means wrapper, single-run Lloyd and Elkan
# solvers delegating the per-iteration work to Cython chunked kernels, the E-step
# helper _labels_inertia (plus a BLAS-threadpool-limited variant), the shared
# _BaseKMeans base, KMeans, the mini-batch update step and MiniBatchKMeans.
# Data are float32 or float64, dense C-ordered or CSR; OpenMP threads parallelize the
# Cython loops while BLAS is limited to one thread to avoid oversubscription.
"""K-means clustering."""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import warnings
from abc import ABC, abstractmethod
from numbers import Integral, Real

import numpy as np
import scipy.sparse as sp

from sklearn.base import (
    BaseEstimator,
    ClassNamePrefixFeaturesOutMixin,
    ClusterMixin,
    TransformerMixin,
    _fit_context,
)
from sklearn.cluster._k_means_common import (
    CHUNK_SIZE,
    _inertia_dense,
    _inertia_sparse,
    _is_same_clustering,
)
from sklearn.cluster._k_means_elkan import (
    elkan_iter_chunked_dense,
    elkan_iter_chunked_sparse,
    init_bounds_dense,
    init_bounds_sparse,
)
from sklearn.cluster._k_means_lloyd import (
    lloyd_iter_chunked_dense,
    lloyd_iter_chunked_sparse,
)
from sklearn.cluster._k_means_minibatch import (
    _minibatch_update_dense,
    _minibatch_update_sparse,
)
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics.pairwise import _euclidean_distances, euclidean_distances
from sklearn.utils import check_array, check_random_state
from sklearn.utils._openmp_helpers import _openmp_effective_n_threads
from sklearn.utils._param_validation import Interval, StrOptions, validate_params
from sklearn.utils.extmath import row_norms
from sklearn.utils.parallel import (
    _get_threadpool_controller,
    _threadpool_controller_decorator,
)
from sklearn.utils.sparsefuncs import mean_variance_axis
from sklearn.utils.sparsefuncs_fast import assign_rows_csr
from sklearn.utils.validation import (
    _check_sample_weight,
    _is_arraylike_not_scalar,
    check_is_fitted,
    validate_data,
)

###############################################################################
# Initialization heuristic


@validate_params(
    {
        "X": ["array-like", "sparse matrix"],
        "n_clusters": [Interval(Integral, 1, None, closed="left")],
        "sample_weight": ["array-like", None],
        "x_squared_norms": ["array-like", None],
        "random_state": ["random_state"],
        "n_local_trials": [Interval(Integral, 1, None, closed="left"), None],
    },
    prefer_skip_nested_validation=True,
)
def kmeans_plusplus(
    X,
    n_clusters,
    *,
    sample_weight=None,
    x_squared_norms=None,
    random_state=None,
    n_local_trials=None,
):
    # [FUNCTION DOC]
    # Public k-means++ seeding: validates X (dense or CSR, float32/float64) and
    # sample_weight, checks n_samples >= n_clusters, computes or validates the squared row
    # norms, resolves random_state and calls _kmeans_plusplus.
    # Inputs: X - array-like or sparse (n_samples, n_features); n_clusters - int >= 1;
    # sample_weight - array-like (n_samples,) or None (unit weights); x_squared_norms -
    # array (n_samples,) or None; random_state - seed/RandomState/None; n_local_trials -
    # int or None (default 2 + int(log(k))).
    # Outputs: (centers array (n_clusters, n_features) in X's dtype, indices int array
    # (n_clusters,)).
    # Exceptions: ValueError if n_samples < n_clusters or x_squared_norms has the wrong
    # length; validation errors.
    # Side effects: consumes random_state.
    """Init n_clusters seeds according to k-means++.

    .. versionadded:: 0.24

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        The data to pick seeds from.

    n_clusters : int
        The number of centroids to initialize.

    sample_weight : array-like of shape (n_samples,), default=None
        The weights for each observation in `X`. If `None`, all observations
        are assigned equal weight. `sample_weight` is ignored if `init`
        is a callable or a user provided array.

        .. versionadded:: 1.3

    x_squared_norms : array-like of shape (n_samples,), default=None
        Squared Euclidean norm of each data point.

    random_state : int or RandomState instance, default=None
        Determines random number generation for centroid initialization. Pass
        an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    n_local_trials : int, default=None
        The number of seeding trials for each center (except the first),
        of which the one reducing inertia the most is greedily chosen.
        Set to None to make the number of trials depend logarithmically
        on the number of seeds (2+log(k)) which is the recommended setting.
        Setting to 1 disables the greedy cluster selection and recovers the
        vanilla k-means++ algorithm which was empirically shown to work less
        well than its greedy variant.

    Returns
    -------
    centers : ndarray of shape (n_clusters, n_features)
        The initial centers for k-means.

    indices : ndarray of shape (n_clusters,)
        The index location of the chosen centers in the data array X. For a
        given index and center, X[index] = center.

    Notes
    -----
    Selects initial cluster centers for k-mean clustering in a smart way
    to speed up convergence. see: Arthur, D. and Vassilvitskii, S.
    "k-means++: the advantages of careful seeding". ACM-SIAM symposium
    on Discrete algorithms. 2007

    Examples
    --------

    >>> from sklearn.cluster import kmeans_plusplus
    >>> import numpy as np
    >>> X = np.array([[1, 2], [1, 4], [1, 0],
    ...               [10, 2], [10, 4], [10, 0]])
    >>> centers, indices = kmeans_plusplus(X, n_clusters=2, random_state=0)
    >>> centers
    array([[10,  2],
           [ 1,  0]])
    >>> indices
    array([3, 2])
    """
    # Check data
    check_array(X, accept_sparse="csr", dtype=[np.float64, np.float32])
    sample_weight = _check_sample_weight(sample_weight, X, dtype=X.dtype)

    if X.shape[0] < n_clusters:
        raise ValueError(
            f"n_samples={X.shape[0]} should be >= n_clusters={n_clusters}."
        )

    # Check parameters
    if x_squared_norms is None:
        x_squared_norms = row_norms(X, squared=True)
    else:
        x_squared_norms = check_array(x_squared_norms, dtype=X.dtype, ensure_2d=False)

    if x_squared_norms.shape[0] != X.shape[0]:
        raise ValueError(
            f"The length of x_squared_norms {x_squared_norms.shape[0]} should "
            f"be equal to the length of n_samples {X.shape[0]}."
        )

    random_state = check_random_state(random_state)

    # Call private k-means++
    centers, indices = _kmeans_plusplus(
        X, n_clusters, x_squared_norms, sample_weight, random_state, n_local_trials
    )

    return centers, indices


def _kmeans_plusplus(
    X, n_clusters, x_squared_norms, sample_weight, random_state, n_local_trials=None
):
    # [FUNCTION DOC]
    # HELPER: Greedy k-means++ (Arthur and Vassilvitskii) with sample weights. Pick the
    # first center by sampling an index with probability proportional to sample_weight;
    # keep each sample's squared distance to its closest chosen center and the weighted
    # potential (sum of weight times distance). For each further center: draw
    # n_local_trials uniform values scaled by the potential and locate them in the
    # cumulative sum of weight times closest distance (inverse-CDF sampling, clipped to
    # the last index); compute squared distances from each candidate to all samples;
    # for each candidate take the elementwise minimum with the current closest distances
    # and its weighted potential; keep the candidate with the smallest potential and
    # update closest distances and potential. Sparse rows are densified when stored.
    # Inputs: X - validated dense or CSR data; n_clusters - int; x_squared_norms -
    # (n_samples,); sample_weight - (n_samples,); random_state - RandomState;
    # n_local_trials - int or None.
    # Outputs: (centers (n_clusters, n_features) in X's dtype, indices int (n_clusters,)).
    # Side effects: consumes random_state.
    """Computational component for initialization of n_clusters by
    k-means++. Prior validation of data is assumed.

    Parameters
    ----------
    X : {ndarray, sparse matrix} of shape (n_samples, n_features)
        The data to pick seeds for.

    n_clusters : int
        The number of seeds to choose.

    sample_weight : ndarray of shape (n_samples,)
        The weights for each observation in `X`.

    x_squared_norms : ndarray of shape (n_samples,)
        Squared Euclidean norm of each data point.

    random_state : RandomState instance
        The generator used to initialize the centers.
        See :term:`Glossary <random_state>`.

    n_local_trials : int, default=None
        The number of seeding trials for each center (except the first),
        of which the one reducing inertia the most is greedily chosen.
        Set to None to make the number of trials depend logarithmically
        on the number of seeds (2+log(k)); this is the default.

    Returns
    -------
    centers : ndarray of shape (n_clusters, n_features)
        The initial centers for k-means.

    indices : ndarray of shape (n_clusters,)
        The index location of the chosen centers in the data array X. For a
        given index and center, X[index] = center.
    """
    n_samples, n_features = X.shape

    centers = np.empty((n_clusters, n_features), dtype=X.dtype)

    # Set the number of local seeding trials if none is given
    if n_local_trials is None:
        # This is what Arthur/Vassilvitskii tried, but did not report
        # specific results for other than mentioning in the conclusion
        # that it helped.
        n_local_trials = 2 + int(np.log(n_clusters))

    # Pick first center randomly and track index of point
    center_id = random_state.choice(n_samples, p=sample_weight / sample_weight.sum())
    indices = np.full(n_clusters, -1, dtype=int)
    if sp.issparse(X):
        centers[0] = X[[center_id]].toarray()
    else:
        centers[0] = X[center_id]
    indices[0] = center_id

    # Initialize list of closest distances and calculate current potential
    closest_dist_sq = _euclidean_distances(
        centers[0, np.newaxis], X, Y_norm_squared=x_squared_norms, squared=True
    )
    current_pot = closest_dist_sq @ sample_weight

    # Pick the remaining n_clusters-1 points
    for c in range(1, n_clusters):
        # Choose center candidates by sampling with probability proportional
        # to the squared distance to the closest existing center
        rand_vals = random_state.uniform(size=n_local_trials) * current_pot
        candidate_ids = np.searchsorted(
            np.cumsum(sample_weight * closest_dist_sq), rand_vals
        )
        # XXX: numerical imprecision can result in a candidate_id out of range
        np.clip(candidate_ids, None, closest_dist_sq.size - 1, out=candidate_ids)

        # Compute distances to center candidates
        distance_to_candidates = _euclidean_distances(
            X[candidate_ids], X, Y_norm_squared=x_squared_norms, squared=True
        )

        # update closest distances squared and potential for each candidate
        np.minimum(closest_dist_sq, distance_to_candidates, out=distance_to_candidates)
        candidates_pot = distance_to_candidates @ sample_weight.reshape(-1, 1)

        # Decide which candidate is the best
        best_candidate = np.argmin(candidates_pot)
        current_pot = candidates_pot[best_candidate]
        closest_dist_sq = distance_to_candidates[best_candidate]
        best_candidate = candidate_ids[best_candidate]

        # Permanently add best center candidate found in local tries
        if sp.issparse(X):
            centers[c] = X[[best_candidate]].toarray()
        else:
            centers[c] = X[best_candidate]
        indices[c] = best_candidate

    return centers, indices


###############################################################################
# K-means batch estimation by EM (expectation maximization)


def _tolerance(X, tol):
    # [FUNCTION DOC]
    # HELPER: Scales tol by the data: returns 0 if tol is 0, else tol times the mean of
    # the per-feature variances (sparse variances via mean_variance_axis).
    # Inputs: X - dense or sparse data; tol - float. Outputs: float.
    # KNOWN ISSUE: variances are unweighted, so sample_weight does not influence the
    # convergence tolerance.
    """Return a tolerance which is dependent on the dataset."""
    if tol == 0:
        return 0
    if sp.issparse(X):
        variances = mean_variance_axis(X, axis=0)[1]
    else:
        variances = np.var(X, axis=0)
    return np.mean(variances) * tol


@validate_params(
    {
        "X": ["array-like", "sparse matrix"],
        "sample_weight": ["array-like", None],
        "return_n_iter": [bool],
    },
    prefer_skip_nested_validation=False,
)
def k_means(
    X,
    n_clusters,
    *,
    sample_weight=None,
    init="k-means++",
    n_init="auto",
    max_iter=300,
    verbose=False,
    tol=1e-4,
    random_state=None,
    copy_x=True,
    algorithm="lloyd",
    return_n_iter=False,
):
    # [FUNCTION DOC]
    # Functional interface: builds KMeans with the given parameters, fits it on X with
    # sample_weight and returns its fitted attributes.
    # Inputs: as in KMeans plus X, sample_weight and return_n_iter (bool, default False).
    # Outputs: (cluster_centers_, labels_, inertia_) and also n_iter_ when return_n_iter.
    # Exceptions: those of KMeans.fit and parameter validation.
    """Perform K-means clustering algorithm.

    Read more in the :ref:`User Guide <k_means>`.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        The observations to cluster. It must be noted that the data
        will be converted to C ordering, which will cause a memory copy
        if the given data is not C-contiguous.

    n_clusters : int
        The number of clusters to form as well as the number of
        centroids to generate.

    sample_weight : array-like of shape (n_samples,), default=None
        The weights for each observation in `X`. If `None`, all observations
        are assigned equal weight. `sample_weight` is not used during
        initialization if `init` is a callable or a user provided array.

    init : {'k-means++', 'random'}, callable or array-like of shape \
            (n_clusters, n_features), default='k-means++'
        Method for initialization:

        - `'k-means++'` : selects initial cluster centers for k-mean
          clustering in a smart way to speed up convergence. See section
          Notes in k_init for more details.
        - `'random'`: choose `n_clusters` observations (rows) at random from data
          for the initial centroids.
        - If an array is passed, it should be of shape `(n_clusters, n_features)`
          and gives the initial centers.
        - If a callable is passed, it should take arguments `X`, `n_clusters` and a
          random state and return an initialization.

    n_init : 'auto' or int, default="auto"
        Number of time the k-means algorithm will be run with different
        centroid seeds. The final results will be the best output of
        n_init consecutive runs in terms of inertia.

        When `n_init='auto'`, the number of runs depends on the value of init:
        10 if using `init='random'` or `init` is a callable;
        1 if using `init='k-means++'` or `init` is an array-like.

        .. versionadded:: 1.2
           Added 'auto' option for `n_init`.

        .. versionchanged:: 1.4
           Default value for `n_init` changed to `'auto'`.

    max_iter : int, default=300
        Maximum number of iterations of the k-means algorithm to run.

    verbose : bool, default=False
        Verbosity mode.

    tol : float, default=1e-4
        Relative tolerance with regards to Frobenius norm of the difference
        in the cluster centers of two consecutive iterations to declare
        convergence.

    random_state : int, RandomState instance or None, default=None
        Determines random number generation for centroid initialization. Use
        an int to make the randomness deterministic.
        See :term:`Glossary <random_state>`.

    copy_x : bool, default=True
        When pre-computing distances it is more numerically accurate to center
        the data first. If `copy_x` is True (default), then the original data is
        not modified. If False, the original data is modified, and put back
        before the function returns, but small numerical differences may be
        introduced by subtracting and then adding the data mean. Note that if
        the original data is not C-contiguous, a copy will be made even if
        `copy_x` is False. If the original data is sparse, but not in CSR format,
        a copy will be made even if `copy_x` is False.

    algorithm : {"lloyd", "elkan"}, default="lloyd"
        K-means algorithm to use. The classical EM-style algorithm is `"lloyd"`.
        The `"elkan"` variation can be more efficient on some datasets with
        well-defined clusters, by using the triangle inequality. However it's
        more memory intensive due to the allocation of an extra array of shape
        `(n_samples, n_clusters)`.

        .. versionchanged:: 0.18
            Added Elkan algorithm

        .. versionchanged:: 1.1
            Renamed "full" to "lloyd", and deprecated "auto" and "full".
            Changed "auto" to use "lloyd" instead of "elkan".

    return_n_iter : bool, default=False
        Whether or not to return the number of iterations.

    Returns
    -------
    centroid : ndarray of shape (n_clusters, n_features)
        Centroids found at the last iteration of k-means.

    label : ndarray of shape (n_samples,)
        The `label[i]` is the code or index of the centroid the
        i'th observation is closest to.

    inertia : float
        The final value of the inertia criterion (sum of squared distances to
        the closest centroid for all observations in the training set).

    best_n_iter : int
        Number of iterations corresponding to the best results.
        Returned only if `return_n_iter` is set to True.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.cluster import k_means
    >>> X = np.array([[1, 2], [1, 4], [1, 0],
    ...               [10, 2], [10, 4], [10, 0]])
    >>> centroid, label, inertia = k_means(
    ...     X, n_clusters=2, n_init="auto", random_state=0
    ... )
    >>> centroid
    array([[10.,  2.],
           [ 1.,  2.]])
    >>> label
    array([1, 1, 1, 0, 0, 0], dtype=int32)
    >>> inertia
    16.0
    """
    est = KMeans(
        n_clusters=n_clusters,
        init=init,
        n_init=n_init,
        max_iter=max_iter,
        verbose=verbose,
        tol=tol,
        random_state=random_state,
        copy_x=copy_x,
        algorithm=algorithm,
    ).fit(X, sample_weight=sample_weight)
    if return_n_iter:
        return est.cluster_centers_, est.labels_, est.inertia_, est.n_iter_
    else:
        return est.cluster_centers_, est.labels_, est.inertia_


def _kmeans_single_elkan(
    X,
    sample_weight,
    centers_init,
    max_iter=300,
    verbose=False,
    tol=1e-4,
    n_threads=1,
):
    # [FUNCTION DOC]
    # HELPER: One Elkan k-means run (triangle-inequality accelerated) from given initial
    # centers. Allocates buffers (new centers, per-cluster weights, labels starting at -1,
    # upper bounds per sample, lower bounds per sample and cluster, center shifts); computes
    # half the pairwise center distances and, per center, the half distance to its closest
    # other center; initializes the bounds. Each iteration runs the chunked Elkan kernel
    # (assign labels, update bounds, compute new centers and shifts), recomputes the half
    # distances from the new centers, optionally prints the inertia, swaps the center
    # buffers, and stops on identical labels (strict convergence) or when the sum of
    # squared center shifts is at most tol. Without strict convergence, one extra
    # assignment-only pass makes labels consistent with the final centers. The inertia is
    # then computed.
    # Inputs: X - dense or CSR; sample_weight - (n_samples,); centers_init -
    # (n_clusters, n_features), at least 2 centers; max_iter - int; verbose - bool;
    # tol - absolute tolerance; n_threads - OpenMP threads.
    # Outputs: (labels int32 (n_samples,), inertia float, centers (n_clusters,
    # n_features), number of iterations run).
    # Side effects: prints when verbose; centers_init may be reused as a work buffer and
    # overwritten.
    """A single run of k-means elkan, assumes preparation completed prior.

    Parameters
    ----------
    X : {ndarray, sparse matrix} of shape (n_samples, n_features)
        The observations to cluster. If sparse matrix, must be in CSR format.

    sample_weight : array-like of shape (n_samples,)
        The weights for each observation in X.

    centers_init : ndarray of shape (n_clusters, n_features)
        The initial centers.

    max_iter : int, default=300
        Maximum number of iterations of the k-means algorithm to run.

    verbose : bool, default=False
        Verbosity mode.

    tol : float, default=1e-4
        Relative tolerance with regards to Frobenius norm of the difference
        in the cluster centers of two consecutive iterations to declare
        convergence.
        It's not advised to set `tol=0` since convergence might never be
        declared due to rounding errors. Use a very small number instead.

    n_threads : int, default=1
        The number of OpenMP threads to use for the computation. Parallelism is
        sample-wise on the main cython loop which assigns each sample to its
        closest center.

    Returns
    -------
    centroid : ndarray of shape (n_clusters, n_features)
        Centroids found at the last iteration of k-means.

    label : ndarray of shape (n_samples,)
        label[i] is the code or index of the centroid the
        i'th observation is closest to.

    inertia : float
        The final value of the inertia criterion (sum of squared distances to
        the closest centroid for all observations in the training set).

    n_iter : int
        Number of iterations run.
    """
    n_samples = X.shape[0]
    n_clusters = centers_init.shape[0]

    # Buffers to avoid new allocations at each iteration.
    centers = centers_init
    centers_new = np.zeros_like(centers)
    weight_in_clusters = np.zeros(n_clusters, dtype=X.dtype)
    labels = np.full(n_samples, -1, dtype=np.int32)
    labels_old = labels.copy()
    center_half_distances = euclidean_distances(centers) / 2
    distance_next_center = np.partition(
        np.asarray(center_half_distances), kth=1, axis=0
    )[1]
    upper_bounds = np.zeros(n_samples, dtype=X.dtype)
    lower_bounds = np.zeros((n_samples, n_clusters), dtype=X.dtype)
    center_shift = np.zeros(n_clusters, dtype=X.dtype)

    if sp.issparse(X):
        init_bounds = init_bounds_sparse
        elkan_iter = elkan_iter_chunked_sparse
        _inertia = _inertia_sparse
    else:
        init_bounds = init_bounds_dense
        elkan_iter = elkan_iter_chunked_dense
        _inertia = _inertia_dense

    init_bounds(
        X,
        centers,
        center_half_distances,
        labels,
        upper_bounds,
        lower_bounds,
        n_threads=n_threads,
    )

    strict_convergence = False

    for i in range(max_iter):
        elkan_iter(
            X,
            sample_weight,
            centers,
            centers_new,
            weight_in_clusters,
            center_half_distances,
            distance_next_center,
            upper_bounds,
            lower_bounds,
            labels,
            center_shift,
            n_threads,
        )

        # compute new pairwise distances between centers and closest other
        # center of each center for next iterations
        center_half_distances = euclidean_distances(centers_new) / 2
        distance_next_center = np.partition(
            np.asarray(center_half_distances), kth=1, axis=0
        )[1]

        if verbose:
            inertia = _inertia(X, sample_weight, centers, labels, n_threads)
            print(f"Iteration {i}, inertia {inertia}")

        centers, centers_new = centers_new, centers

        if np.array_equal(labels, labels_old):
            # First check the labels for strict convergence.
            if verbose:
                print(f"Converged at iteration {i}: strict convergence.")
            strict_convergence = True
            break
        else:
            # No strict convergence, check for tol based convergence.
            center_shift_tot = (center_shift**2).sum()
            if center_shift_tot <= tol:
                if verbose:
                    print(
                        f"Converged at iteration {i}: center shift "
                        f"{center_shift_tot} within tolerance {tol}."
                    )
                break

        labels_old[:] = labels

    if not strict_convergence:
        # rerun E-step so that predicted labels match cluster centers
        elkan_iter(
            X,
            sample_weight,
            centers,
            centers,
            weight_in_clusters,
            center_half_distances,
            distance_next_center,
            upper_bounds,
            lower_bounds,
            labels,
            center_shift,
            n_threads,
            update_centers=False,
        )

    inertia = _inertia(X, sample_weight, centers, labels, n_threads)

    return labels, inertia, centers, i + 1


# Threadpoolctl context to limit the number of threads in second level of
# nested parallelism (i.e. BLAS) to avoid oversubscription.
@_threadpool_controller_decorator(limits=1, user_api="blas")
def _kmeans_single_lloyd(
    X,
    sample_weight,
    centers_init,
    max_iter=300,
    verbose=False,
    tol=1e-4,
    n_threads=1,
):
    # [FUNCTION DOC]
    # HELPER: One Lloyd k-means run with BLAS limited to one thread. Each iteration runs
    # the chunked Lloyd kernel (assign each sample to its nearest center, accumulate
    # weighted means into the new-center buffer, compute center shifts), optionally
    # prints the inertia, swaps the center buffers, and stops on identical labels (strict
    # convergence) or when the sum of squared center shifts is at most tol. Without strict
    # convergence, one assignment-only pass aligns labels with the final centers before
    # computing the inertia.
    # Inputs/Outputs: same as _kmeans_single_elkan (any number of clusters).
    # Side effects: prints when verbose; centers_init may be overwritten as a work buffer.
    """A single run of k-means lloyd, assumes preparation completed prior.

    Parameters
    ----------
    X : {ndarray, sparse matrix} of shape (n_samples, n_features)
        The observations to cluster. If sparse matrix, must be in CSR format.

    sample_weight : ndarray of shape (n_samples,)
        The weights for each observation in X.

    centers_init : ndarray of shape (n_clusters, n_features)
        The initial centers.

    max_iter : int, default=300
        Maximum number of iterations of the k-means algorithm to run.

    verbose : bool, default=False
        Verbosity mode

    tol : float, default=1e-4
        Relative tolerance with regards to Frobenius norm of the difference
        in the cluster centers of two consecutive iterations to declare
        convergence.
        It's not advised to set `tol=0` since convergence might never be
        declared due to rounding errors. Use a very small number instead.

    n_threads : int, default=1
        The number of OpenMP threads to use for the computation. Parallelism is
        sample-wise on the main cython loop which assigns each sample to its
        closest center.

    Returns
    -------
    centroid : ndarray of shape (n_clusters, n_features)
        Centroids found at the last iteration of k-means.

    label : ndarray of shape (n_samples,)
        label[i] is the code or index of the centroid the
        i'th observation is closest to.

    inertia : float
        The final value of the inertia criterion (sum of squared distances to
        the closest centroid for all observations in the training set).

    n_iter : int
        Number of iterations run.
    """
    n_clusters = centers_init.shape[0]

    # Buffers to avoid new allocations at each iteration.
    centers = centers_init
    centers_new = np.zeros_like(centers)
    labels = np.full(X.shape[0], -1, dtype=np.int32)
    labels_old = labels.copy()
    weight_in_clusters = np.zeros(n_clusters, dtype=X.dtype)
    center_shift = np.zeros(n_clusters, dtype=X.dtype)

    if sp.issparse(X):
        lloyd_iter = lloyd_iter_chunked_sparse
        _inertia = _inertia_sparse
    else:
        lloyd_iter = lloyd_iter_chunked_dense
        _inertia = _inertia_dense

    strict_convergence = False

    for i in range(max_iter):
        lloyd_iter(
            X,
            sample_weight,
            centers,
            centers_new,
            weight_in_clusters,
            labels,
            center_shift,
            n_threads,
        )

        if verbose:
            inertia = _inertia(X, sample_weight, centers, labels, n_threads)
            print(f"Iteration {i}, inertia {inertia}.")

        centers, centers_new = centers_new, centers

        if np.array_equal(labels, labels_old):
            # First check the labels for strict convergence.
            if verbose:
                print(f"Converged at iteration {i}: strict convergence.")
            strict_convergence = True
            break
        else:
            # No strict convergence, check for tol based convergence.
            center_shift_tot = (center_shift**2).sum()
            if center_shift_tot <= tol:
                if verbose:
                    print(
                        f"Converged at iteration {i}: center shift "
                        f"{center_shift_tot} within tolerance {tol}."
                    )
                break

        labels_old[:] = labels

    if not strict_convergence:
        # rerun E-step so that predicted labels match cluster centers
        lloyd_iter(
            X,
            sample_weight,
            centers,
            centers,
            weight_in_clusters,
            labels,
            center_shift,
            n_threads,
            update_centers=False,
        )

    inertia = _inertia(X, sample_weight, centers, labels, n_threads)

    return labels, inertia, centers, i + 1


def _labels_inertia(X, sample_weight, centers, n_threads=1, return_inertia=True):
    # [FUNCTION DOC]
    # HELPER: E-step: assigns each sample to its nearest center using the Lloyd kernel
    # with center updates disabled, and optionally computes the weighted inertia (sum of
    # weight times squared distance to the assigned center).
    # Inputs: X - dense or CSR; sample_weight - (n_samples,); centers - (n_clusters,
    # n_features); n_threads - int; return_inertia - bool, default True.
    # Outputs: labels int32 (n_samples,), plus inertia float when return_inertia.
    """E step of the K-means EM algorithm.

    Compute the labels and the inertia of the given samples and centers.

    Parameters
    ----------
    X : {ndarray, sparse matrix} of shape (n_samples, n_features)
        The input samples to assign to the labels. If sparse matrix, must
        be in CSR format.

    sample_weight : ndarray of shape (n_samples,)
        The weights for each observation in X.

    centers : ndarray of shape (n_clusters, n_features)
        The cluster centers.

    n_threads : int, default=1
        The number of OpenMP threads to use for the computation. Parallelism is
        sample-wise on the main cython loop which assigns each sample to its
        closest center.

    return_inertia : bool, default=True
        Whether to compute and return the inertia.

    Returns
    -------
    labels : ndarray of shape (n_samples,)
        The resulting assignment.

    inertia : float
        Sum of squared distances of samples to their closest cluster center.
        Inertia is only returned if return_inertia is True.
    """
    n_samples = X.shape[0]
    n_clusters = centers.shape[0]

    labels = np.full(n_samples, -1, dtype=np.int32)
    center_shift = np.zeros(n_clusters, dtype=centers.dtype)

    if sp.issparse(X):
        _labels = lloyd_iter_chunked_sparse
        _inertia = _inertia_sparse
    else:
        _labels = lloyd_iter_chunked_dense
        _inertia = _inertia_dense

    _labels(
        X,
        sample_weight,
        centers,
        centers_new=None,
        weight_in_clusters=None,
        labels=labels,
        center_shift=center_shift,
        n_threads=n_threads,
        update_centers=False,
    )

    if return_inertia:
        inertia = _inertia(X, sample_weight, centers, labels, n_threads)
        return labels, inertia

    return labels


# Same as _labels_inertia but in a threadpool_limits context.
_labels_inertia_threadpool_limit = _threadpool_controller_decorator(
    limits=1, user_api="blas"
)(_labels_inertia)


class _BaseKMeans(
    ClassNamePrefixFeaturesOutMixin, TransformerMixin, ClusterMixin, BaseEstimator, ABC
):
    # [CLASS DOC]
    # Abstract base for KMeans and MiniBatchKMeans holding shared parameter constraints,
    # input checks against parameters, centroid initialization, predict/transform/score
    # and tags. Subclasses must implement _warn_mkl_vcomp and fit. Fitted attributes set
    # by subclasses: cluster_centers_, labels_, inertia_, n_iter_, _n_threads,
    # _n_features_out.
    """Base class for KMeans and MiniBatchKMeans"""

    _parameter_constraints: dict = {
        "n_clusters": [Interval(Integral, 1, None, closed="left")],
        "init": [StrOptions({"k-means++", "random"}), callable, "array-like"],
        "n_init": [
            StrOptions({"auto"}),
            Interval(Integral, 1, None, closed="left"),
        ],
        "max_iter": [Interval(Integral, 1, None, closed="left")],
        "tol": [Interval(Real, 0, None, closed="left")],
        "verbose": ["verbose"],
        "random_state": ["random_state"],
    }

    def __init__(
        self,
        n_clusters,
        *,
        init,
        n_init,
        max_iter,
        tol,
        verbose,
        random_state,
    ):
        # [FUNCTION DOC]
        # Stores n_clusters, init, n_init, max_iter, tol, verbose and random_state as given.
        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.tol = tol
        self.n_init = n_init
        self.verbose = verbose
        self.random_state = random_state

    def _check_params_vs_input(self, X, default_n_init=None):
        # [FUNCTION DOC]
        # Checks parameters that depend on the data and derives private values: raises
        # ValueError if n_samples < n_clusters; sets _tol from _tolerance; resolves n_init
        # "auto" to 1 for "k-means++" or an array init and to default_n_init for "random" or a
        # callable; if init is an explicit array and the resolved count is not 1, warns
        # (RuntimeWarning) and forces _n_init to 1.
        # Inputs: X - validated data; default_n_init - int used for "auto".
        # Side effects: sets self._tol and self._n_init.
        # n_clusters
        if X.shape[0] < self.n_clusters:
            raise ValueError(
                f"n_samples={X.shape[0]} should be >= n_clusters={self.n_clusters}."
            )

        # tol
        self._tol = _tolerance(X, self.tol)

        # n-init
        if self.n_init == "auto":
            if isinstance(self.init, str) and self.init == "k-means++":
                self._n_init = 1
            elif isinstance(self.init, str) and self.init == "random":
                self._n_init = default_n_init
            elif callable(self.init):
                self._n_init = default_n_init
            else:  # array-like
                self._n_init = 1
        else:
            self._n_init = self.n_init

        if _is_arraylike_not_scalar(self.init) and self._n_init != 1:
            warnings.warn(
                (
                    "Explicit initial center position passed: performing only"
                    f" one init in {self.__class__.__name__} instead of "
                    f"n_init={self._n_init}."
                ),
                RuntimeWarning,
                stacklevel=2,
            )
            self._n_init = 1

    @abstractmethod
    def _warn_mkl_vcomp(self, n_active_threads):
        # [FUNCTION DOC]
        # Abstract hook: issues the estimator-specific warning about the MKL plus vcomp
        # memory leak. Inputs: n_active_threads - int.
        """Issue an estimator specific warning when vcomp and mkl are both present

        This method is called by `_check_mkl_vcomp`.
        """

    def _check_mkl_vcomp(self, X, n_samples):
        # [FUNCTION DOC]
        # HELPER: For dense X only, computes the number of active chunks (n_samples divided by
        # CHUNK_SIZE, rounded up); if fewer than the available threads, inspects the
        # threadpool to detect the vcomp OpenMP runtime together with Intel-threaded MKL and
        # then calls _warn_mkl_vcomp. Inputs: X; n_samples - rows per kernel call.
        # Side effects: may emit a warning (see #18653).
        """Check when vcomp and mkl are both present"""
        # The BLAS call inside a prange in lloyd_iter_chunked_dense is known to
        # cause a small memory leak when there are less chunks than the number
        # of available threads. It only happens when the OpenMP library is
        # vcomp (microsoft OpenMP) and the BLAS library is MKL. see #18653
        if sp.issparse(X):
            return

        n_active_threads = int(np.ceil(n_samples / CHUNK_SIZE))
        if n_active_threads < self._n_threads:
            modules = _get_threadpool_controller().info()
            has_vcomp = "vcomp" in [module["prefix"] for module in modules]
            has_mkl = ("mkl", "intel") in [
                (module["internal_api"], module.get("threading_layer", None))
                for module in modules
            ]
            if has_vcomp and has_mkl:
                self._warn_mkl_vcomp(n_active_threads)

    def _validate_center_shape(self, X, centers, n_centroids=None):
        # [FUNCTION DOC]
        # HELPER: Raises ValueError if centers does not have n_centroids rows (default
        # n_clusters) or its column count differs from X's.
        """Check if the shape of the centers is correct."""
        if n_centroids is None:
            n_centroids = self.n_clusters

        if centers.shape[0] != n_centroids:
            raise ValueError(
                f"The shape of the initial centers {centers.shape} does not "
                f"match the number of clusters {n_centroids}."
            )
        if centers.shape[1] != X.shape[1]:
            raise ValueError(
                f"The shape of the initial centers {centers.shape} does not "
                f"match the number of features of the data {X.shape[1]}."
            )

    def _check_test_data(self, X):
        # [FUNCTION DOC]
        # HELPER: Validates data for predict/transform/score without resetting
        # n_features_in_: CSR or dense C-ordered float64/float32, no 64-bit sparse indices.
        # Outputs: the validated X.
        X = validate_data(
            self,
            X,
            accept_sparse="csr",
            reset=False,
            dtype=[np.float64, np.float32],
            order="C",
            accept_large_sparse=False,
        )
        return X

    def _init_centroids(
        self,
        X,
        x_squared_norms,
        init,
        random_state,
        sample_weight,
        init_size=None,
        n_centroids=None,
    ):
        # [FUNCTION DOC]
        # HELPER: Produces initial centers. If init_size is smaller than n_samples, first
        # draws init_size row indices with replacement and restricts X, norms and weights to
        # them. Then: "k-means++" - weighted k-means++; "random" - n_clusters distinct rows
        # sampled with probability proportional to weight; array - used as is; callable -
        # called with (X, n_clusters, random_state), converted to X's dtype and shape-checked.
        # Sparse results are densified.
        # Inputs: X; x_squared_norms; init - str, callable or validated array; random_state;
        # sample_weight; init_size - int or None; n_centroids - int or None (n_clusters).
        # Outputs: centers (n_centroids, n_features).
        # Exceptions: ValueError from shape validation for callable output.
        # Side effects: consumes random_state.
        # KNOWN ISSUE: the array branch tests self.init instead of the init argument, so a
        # caller passing an init that differs from self.init (e.g. bisecting k-means) relies
        # on both having the same kind.
        """Compute the initial centroids.

        Parameters
        ----------
        X : {ndarray, sparse matrix} of shape (n_samples, n_features)
            The input samples.

        x_squared_norms : ndarray of shape (n_samples,)
            Squared euclidean norm of each data point. Pass it if you have it
            at hands already to avoid it being recomputed here.

        init : {'k-means++', 'random'}, callable or ndarray of shape \
                (n_clusters, n_features)
            Method for initialization.

        random_state : RandomState instance
            Determines random number generation for centroid initialization.
            See :term:`Glossary <random_state>`.

        sample_weight : ndarray of shape (n_samples,)
            The weights for each observation in X. `sample_weight` is not used
            during initialization if `init` is a callable or a user provided
            array.

        init_size : int, default=None
            Number of samples to randomly sample for speeding up the
            initialization (sometimes at the expense of accuracy).

        n_centroids : int, default=None
            Number of centroids to initialize.
            If left to 'None' the number of centroids will be equal to
            number of clusters to form (self.n_clusters).

        Returns
        -------
        centers : ndarray of shape (n_clusters, n_features)
            Initial centroids of clusters.
        """
        n_samples = X.shape[0]
        n_clusters = self.n_clusters if n_centroids is None else n_centroids

        if init_size is not None and init_size < n_samples:
            init_indices = random_state.randint(0, n_samples, init_size)
            X = X[init_indices]
            x_squared_norms = x_squared_norms[init_indices]
            n_samples = X.shape[0]
            sample_weight = sample_weight[init_indices]

        if isinstance(init, str) and init == "k-means++":
            centers, _ = _kmeans_plusplus(
                X,
                n_clusters,
                random_state=random_state,
                x_squared_norms=x_squared_norms,
                sample_weight=sample_weight,
            )
        elif isinstance(init, str) and init == "random":
            seeds = random_state.choice(
                n_samples,
                size=n_clusters,
                replace=False,
                p=sample_weight / sample_weight.sum(),
            )
            centers = X[seeds]
        elif _is_arraylike_not_scalar(self.init):
            centers = init
        elif callable(init):
            centers = init(X, n_clusters, random_state=random_state)
            centers = check_array(centers, dtype=X.dtype, copy=False, order="C")
            self._validate_center_shape(X, centers, n_centroids=n_clusters)

        if sp.issparse(centers):
            centers = centers.toarray()

        return centers

    def fit_predict(self, X, y=None, sample_weight=None):
        # [FUNCTION DOC]
        # Fits on X with sample_weight and returns labels_. Outputs: int array (n_samples,).
        """Compute cluster centers and predict cluster index for each sample.

        Convenience method; equivalent to calling fit(X) followed by
        predict(X).

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            New data to transform.

        y : Ignored
            Not used, present here for API consistency by convention.

        sample_weight : array-like of shape (n_samples,), default=None
            The weights for each observation in X. If None, all observations
            are assigned equal weight.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Index of the cluster each sample belongs to.
        """
        return self.fit(X, sample_weight=sample_weight).labels_

    def predict(self, X):
        # [FUNCTION DOC]
        # Assigns each sample to its nearest fitted center with the threadpool-limited
        # E-step, using unit weights. Inputs: X (n_samples, n_features).
        # Outputs: int32 labels (n_samples,).
        # Exceptions: NotFittedError; validation errors including feature-count mismatch.
        """Predict the closest cluster each sample in X belongs to.

        In the vector quantization literature, `cluster_centers_` is called
        the code book and each value returned by `predict` is the index of
        the closest code in the code book.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            New data to predict.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Index of the cluster each sample belongs to.
        """
        check_is_fitted(self)

        X = self._check_test_data(X)

        # sample weights are not used by predict but cython helpers expect an array
        sample_weight = np.ones(X.shape[0], dtype=X.dtype)

        labels = _labels_inertia_threadpool_limit(
            X,
            sample_weight,
            self.cluster_centers_,
            n_threads=self._n_threads,
            return_inertia=False,
        )

        return labels

    def fit_transform(self, X, y=None, sample_weight=None):
        # [FUNCTION DOC]
        # Fits on X with sample_weight and returns euclidean distances from each sample to
        # every center via _transform. Outputs: array (n_samples, n_clusters).
        """Compute clustering and transform X to cluster-distance space.

        Equivalent to fit(X).transform(X), but more efficiently implemented.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            New data to transform.

        y : Ignored
            Not used, present here for API consistency by convention.

        sample_weight : array-like of shape (n_samples,), default=None
            The weights for each observation in X. If None, all observations
            are assigned equal weight.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_clusters)
            X transformed in the new space.
        """
        return self.fit(X, sample_weight=sample_weight)._transform(X)

    def transform(self, X):
        # [FUNCTION DOC]
        # Validates X and returns its euclidean distances to each fitted center (dense even
        # for sparse input). Outputs: array (n_samples, n_clusters).
        # Exceptions: NotFittedError; validation errors.
        """Transform X to a cluster-distance space.

        In the new space, each dimension is the distance to the cluster
        centers. Note that even if X is sparse, the array returned by
        `transform` will typically be dense.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            New data to transform.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_clusters)
            X transformed in the new space.
        """
        check_is_fitted(self)

        X = self._check_test_data(X)
        return self._transform(X)

    def _transform(self, X):
        # [FUNCTION DOC]
        # HELPER: Euclidean distances between X and cluster_centers_, no validation.
        """Guts of transform method; no input validation."""
        return euclidean_distances(X, self.cluster_centers_)

    def score(self, X, y=None, sample_weight=None):
        # [FUNCTION DOC]
        # Returns minus the weighted inertia of X with respect to the fitted centers.
        # Inputs: X; y - ignored; sample_weight - array-like or None. Outputs: float.
        # Exceptions: NotFittedError; validation errors.
        """Opposite of the value of X on the K-means objective.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            New data.

        y : Ignored
            Not used, present here for API consistency by convention.

        sample_weight : array-like of shape (n_samples,), default=None
            The weights for each observation in X. If None, all observations
            are assigned equal weight.

        Returns
        -------
        score : float
            Opposite of the value of X on the K-means objective.
        """
        check_is_fitted(self)

        X = self._check_test_data(X)
        sample_weight = _check_sample_weight(sample_weight, X, dtype=X.dtype)

        _, scores = _labels_inertia_threadpool_limit(
            X, sample_weight, self.cluster_centers_, self._n_threads
        )
        return -scores

    def __sklearn_tags__(self):
        # [FUNCTION DOC]
        # Extends base tags to declare sparse input support.
        tags = super().__sklearn_tags__()
        tags.input_tags.sparse = True
        return tags


class KMeans(_BaseKMeans):
    # [CLASS DOC]
    # Full-batch k-means estimator (Lloyd or Elkan). Adds copy_x and algorithm to the base
    # parameters; n_init "auto" means 10 runs for random/callable init. Fitted attributes:
    # cluster_centers_, labels_, inertia_, n_iter_, n_features_in_, feature_names_in_.
    """K-Means clustering.

    Read more in the :ref:`User Guide <k_means>`.

    Parameters
    ----------

    n_clusters : int, default=8
        The number of clusters to form as well as the number of
        centroids to generate.

        For an example of how to choose an optimal value for `n_clusters` refer to
        :ref:`sphx_glr_auto_examples_cluster_plot_kmeans_silhouette_analysis.py`.

    init : {'k-means++', 'random'}, callable or array-like of shape \
            (n_clusters, n_features), default='k-means++'
        Method for initialization:

        * 'k-means++' : selects initial cluster centroids using sampling \
            based on an empirical probability distribution of the points' \
            contribution to the overall inertia. This technique speeds up \
            convergence. The algorithm implemented is "greedy k-means++". It \
            differs from the vanilla k-means++ by making several trials at \
            each sampling step and choosing the best centroid among them.

        * 'random': choose `n_clusters` observations (rows) at random from \
        data for the initial centroids.

        * If an array is passed, it should be of shape (n_clusters, n_features)\
        and gives the initial centers.

        * If a callable is passed, it should take arguments X, n_clusters and a\
        random state and return an initialization.

        For an example of how to use the different `init` strategies, see
        :ref:`sphx_glr_auto_examples_cluster_plot_kmeans_digits.py`.

        For an evaluation of the impact of initialization, see the example
        :ref:`sphx_glr_auto_examples_cluster_plot_kmeans_stability_low_dim_dense.py`.

    n_init : 'auto' or int, default='auto'
        Number of times the k-means algorithm is run with different centroid
        seeds. The final results is the best output of `n_init` consecutive runs
        in terms of inertia. Several runs are recommended for sparse
        high-dimensional problems (see :ref:`kmeans_sparse_high_dim`).

        When `n_init='auto'`, the number of runs depends on the value of init:
        10 if using `init='random'` or `init` is a callable;
        1 if using `init='k-means++'` or `init` is an array-like.

        .. versionadded:: 1.2
           Added 'auto' option for `n_init`.

        .. versionchanged:: 1.4
           Default value for `n_init` changed to `'auto'`.

    max_iter : int, default=300
        Maximum number of iterations of the k-means algorithm for a
        single run.

    tol : float, default=1e-4
        Relative tolerance with regards to Frobenius norm of the difference
        in the cluster centers of two consecutive iterations to declare
        convergence.

    verbose : int, default=0
        Verbosity mode.

    random_state : int, RandomState instance or None, default=None
        Determines random number generation for centroid initialization. Use
        an int to make the randomness deterministic.
        See :term:`Glossary <random_state>`.

    copy_x : bool, default=True
        When pre-computing distances it is more numerically accurate to center
        the data first. If copy_x is True (default), then the original data is
        not modified. If False, the original data is modified, and put back
        before the function returns, but small numerical differences may be
        introduced by subtracting and then adding the data mean. Note that if
        the original data is not C-contiguous, a copy will be made even if
        copy_x is False. If the original data is sparse, but not in CSR format,
        a copy will be made even if copy_x is False.

    algorithm : {"lloyd", "elkan"}, default="lloyd"
        K-means algorithm to use. The classical EM-style algorithm is `"lloyd"`.
        The `"elkan"` variation can be more efficient on some datasets with
        well-defined clusters, by using the triangle inequality. However it's
        more memory intensive due to the allocation of an extra array of shape
        `(n_samples, n_clusters)`.

        .. versionchanged:: 0.18
            Added Elkan algorithm

        .. versionchanged:: 1.1
            Renamed "full" to "lloyd", and deprecated "auto" and "full".
            Changed "auto" to use "lloyd" instead of "elkan".

    Attributes
    ----------
    cluster_centers_ : ndarray of shape (n_clusters, n_features)
        Coordinates of cluster centers. If the algorithm stops before fully
        converging (see ``tol`` and ``max_iter``), these will not be
        consistent with ``labels_``.

    labels_ : ndarray of shape (n_samples,)
        Labels of each point

    inertia_ : float
        Sum of squared distances of samples to their closest cluster center,
        weighted by the sample weights if provided.

    n_iter_ : int
        Number of iterations run.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    MiniBatchKMeans : Alternative online implementation that does incremental
        updates of the centers positions using mini-batches.
        For large scale learning (say n_samples > 10k) MiniBatchKMeans is
        probably much faster than the default batch implementation.

    Notes
    -----
    The k-means problem is solved using either Lloyd's or Elkan's algorithm.

    The average complexity is given by O(k n T), where n is the number of
    samples and T is the number of iteration.

    The worst case complexity is given by O(n^(k+2/p)) with
    n = n_samples, p = n_features.
    Refer to :doi:`"How slow is the k-means method?" D. Arthur and S. Vassilvitskii -
    SoCG2006.<10.1145/1137856.1137880>` for more details.

    In practice, the k-means algorithm is very fast (one of the fastest
    clustering algorithms available), but it falls in local minima. That's why
    it can be useful to restart it several times.

    If the algorithm stops before fully converging (because of ``tol`` or
    ``max_iter``), ``labels_`` and ``cluster_centers_`` will not be consistent,
    i.e. the ``cluster_centers_`` will not be the means of the points in each
    cluster. Also, the estimator will reassign ``labels_`` after the last
    iteration to make ``labels_`` consistent with ``predict`` on the training
    set.

    Examples
    --------

    >>> from sklearn.cluster import KMeans
    >>> import numpy as np
    >>> X = np.array([[1, 2], [1, 4], [1, 0],
    ...               [10, 2], [10, 4], [10, 0]])
    >>> kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto").fit(X)
    >>> kmeans.labels_
    array([1, 1, 1, 0, 0, 0], dtype=int32)
    >>> kmeans.predict([[0, 0], [12, 3]])
    array([1, 0], dtype=int32)
    >>> kmeans.cluster_centers_
    array([[10.,  2.],
           [ 1.,  2.]])

    For examples of common problems with K-Means and how to address them see
    :ref:`sphx_glr_auto_examples_cluster_plot_kmeans_assumptions.py`.

    For a demonstration of how K-Means can be used to cluster text documents see
    :ref:`sphx_glr_auto_examples_text_plot_document_clustering.py`.

    For a comparison between K-Means and MiniBatchKMeans refer to example
    :ref:`sphx_glr_auto_examples_cluster_plot_mini_batch_kmeans.py`.

    For a comparison between K-Means and BisectingKMeans refer to example
    :ref:`sphx_glr_auto_examples_cluster_plot_bisect_kmeans.py`.
    """

    _parameter_constraints: dict = {
        **_BaseKMeans._parameter_constraints,
        "copy_x": ["boolean"],
        "algorithm": [StrOptions({"lloyd", "elkan"})],
    }

    def __init__(
        self,
        n_clusters=8,
        *,
        init="k-means++",
        n_init="auto",
        max_iter=300,
        tol=1e-4,
        verbose=0,
        random_state=None,
        copy_x=True,
        algorithm="lloyd",
    ):
        # [FUNCTION DOC]
        # Stores parameters via the base constructor plus copy_x and algorithm.
        super().__init__(
            n_clusters=n_clusters,
            init=init,
            n_init=n_init,
            max_iter=max_iter,
            tol=tol,
            verbose=verbose,
            random_state=random_state,
        )

        self.copy_x = copy_x
        self.algorithm = algorithm

    def _check_params_vs_input(self, X):
        # [FUNCTION DOC]
        # Runs the base checks with default_n_init=10 and sets _algorithm, replacing "elkan"
        # by "lloyd" with a RuntimeWarning when n_clusters is 1.
        # Side effects: sets _tol, _n_init, _algorithm.
        super()._check_params_vs_input(X, default_n_init=10)

        self._algorithm = self.algorithm
        if self._algorithm == "elkan" and self.n_clusters == 1:
            warnings.warn(
                (
                    "algorithm='elkan' doesn't make sense for a single "
                    "cluster. Using 'lloyd' instead."
                ),
                RuntimeWarning,
            )
            self._algorithm = "lloyd"

    def _warn_mkl_vcomp(self, n_active_threads):
        # [FUNCTION DOC]
        # Emits a UserWarning about the Windows MKL memory leak and suggests setting
        # OMP_NUM_THREADS to n_active_threads.
        """Warn when vcomp and mkl are both present"""
        warnings.warn(
            "KMeans is known to have a memory leak on Windows "
            "with MKL, when there are less chunks than available "
            "threads. You can avoid it by setting the environment"
            f" variable OMP_NUM_THREADS={n_active_threads}."
        )

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None, sample_weight=None):
        # [FUNCTION DOC]
        # Steps: validate X (CSR or dense C-ordered float, copied when copy_x); check
        # parameters; resolve random_state and sample_weight; get the OpenMP thread count;
        # validate an array init (copied, shape-checked). For dense X, subtract the column
        # mean from X (and from an array init) for numerical accuracy. Precompute squared
        # row norms; choose the Elkan or Lloyd single-run function (Lloyd also checks the
        # MKL/vcomp issue). For each of _n_init runs: initialize centers and run once; keep
        # the run if it is the first, or if its inertia is lower and its labels are not the
        # same partition as the current best. Afterwards add the mean back to the centers
        # (and to X when copy_x is False, restoring the caller's array). Warn
        # (ConvergenceWarning) if fewer distinct labels than n_clusters were found.
        # Inputs: X - array-like or sparse (n_samples, n_features); y - ignored;
        # sample_weight - array-like or None. Outputs: self.
        # Exceptions: ValueError on n_samples < n_clusters or bad init shape; validation
        # errors.
        # Side effects: sets cluster_centers_, _n_features_out, labels_, inertia_, n_iter_,
        # n_features_in_, _n_threads and private parameter-derived attributes; when copy_x is
        # False the caller's dense float C-contiguous array is temporarily modified in place;
        # prints when verbose.
        """Compute k-means clustering.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training instances to cluster. It must be noted that the data
            will be converted to C ordering, which will cause a memory
            copy if the given data is not C-contiguous.
            If a sparse matrix is passed, a copy will be made if it's not in
            CSR format.

        y : Ignored
            Not used, present here for API consistency by convention.

        sample_weight : array-like of shape (n_samples,), default=None
            The weights for each observation in X. If None, all observations
            are assigned equal weight. `sample_weight` is not used during
            initialization if `init` is a callable or a user provided array.

            .. versionadded:: 0.20

        Returns
        -------
        self : object
            Fitted estimator.
        """
        X = validate_data(
            self,
            X,
            accept_sparse="csr",
            dtype=[np.float64, np.float32],
            order="C",
            copy=self.copy_x,
            accept_large_sparse=False,
        )

        self._check_params_vs_input(X)

        random_state = check_random_state(self.random_state)
        sample_weight = _check_sample_weight(sample_weight, X, dtype=X.dtype)
        self._n_threads = _openmp_effective_n_threads()

        # Validate init array
        init = self.init
        init_is_array_like = _is_arraylike_not_scalar(init)
        if init_is_array_like:
            init = check_array(init, dtype=X.dtype, copy=True, order="C")
            self._validate_center_shape(X, init)

        # subtract of mean of x for more accurate distance computations
        if not sp.issparse(X):
            X_mean = X.mean(axis=0)
            # The copy was already done above
            X -= X_mean

            if init_is_array_like:
                init -= X_mean

        # precompute squared norms of data points
        x_squared_norms = row_norms(X, squared=True)

        if self._algorithm == "elkan":
            kmeans_single = _kmeans_single_elkan
        else:
            kmeans_single = _kmeans_single_lloyd
            self._check_mkl_vcomp(X, X.shape[0])

        best_inertia, best_labels = None, None

        for i in range(self._n_init):
            # Initialize centers
            centers_init = self._init_centroids(
                X,
                x_squared_norms=x_squared_norms,
                init=init,
                random_state=random_state,
                sample_weight=sample_weight,
            )
            if self.verbose:
                print("Initialization complete")

            # run a k-means once
            labels, inertia, centers, n_iter_ = kmeans_single(
                X,
                sample_weight,
                centers_init,
                max_iter=self.max_iter,
                verbose=self.verbose,
                tol=self._tol,
                n_threads=self._n_threads,
            )

            # determine if these results are the best so far
            # we chose a new run if it has a better inertia and the clustering is
            # different from the best so far (it's possible that the inertia is
            # slightly better even if the clustering is the same with potentially
            # permuted labels, due to rounding errors)
            if best_inertia is None or (
                inertia < best_inertia
                and not _is_same_clustering(labels, best_labels, self.n_clusters)
            ):
                best_labels = labels
                best_centers = centers
                best_inertia = inertia
                best_n_iter = n_iter_

        if not sp.issparse(X):
            if not self.copy_x:
                X += X_mean
            best_centers += X_mean

        distinct_clusters = len(set(best_labels))
        if distinct_clusters < self.n_clusters:
            warnings.warn(
                "Number of distinct clusters ({}) found smaller than "
                "n_clusters ({}). Possibly due to duplicate points "
                "in X.".format(distinct_clusters, self.n_clusters),
                ConvergenceWarning,
                stacklevel=2,
            )

        self.cluster_centers_ = best_centers
        self._n_features_out = self.cluster_centers_.shape[0]
        self.labels_ = best_labels
        self.inertia_ = best_inertia
        self.n_iter_ = best_n_iter
        return self


def _mini_batch_step(
    X,
    sample_weight,
    centers,
    centers_new,
    weight_sums,
    random_state,
    random_reassign=False,
    reassignment_ratio=0.01,
    verbose=False,
    n_threads=1,
):
    # [FUNCTION DOC]
    # HELPER: One mini-batch update. Assigns batch samples to the current centers and
    # computes the batch inertia; updates centers with the Cython mini-batch kernel (each
    # center moves toward the weighted mean of its newly assigned samples with a learning
    # rate based on the accumulated weight, which is incremented in weight_sums). If
    # random_reassign and reassignment_ratio > 0, flags centers whose accumulated weight is
    # below reassignment_ratio times the maximum, keeping at most half the batch size of
    # them (the lowest-weight ones), moves each flagged center to a batch sample drawn
    # uniformly without replacement, and resets their weights to the minimum weight of the
    # unflagged centers.
    # Inputs: X - batch, dense or CSR; sample_weight - (batch,); centers - current
    # centers; centers_new - output buffer (may be the same array as centers);
    # weight_sums - (n_clusters,) accumulated weights; random_state; random_reassign -
    # bool; reassignment_ratio - float; verbose - bool; n_threads - int.
    # Outputs: float batch inertia computed before the update.
    # Side effects: mutates centers_new and weight_sums; consumes random_state; prints
    # when verbose.
    # KNOWN ISSUE: with reassignment_ratio > 1 every center (including the heaviest)
    # can be flagged; the minimum over the unflagged weights is then taken over an
    # empty array and raises ValueError.
    """Incremental update of the centers for the Minibatch K-Means algorithm.

    Parameters
    ----------

    X : {ndarray, sparse matrix} of shape (n_samples, n_features)
        The original data array. If sparse, must be in CSR format.

    sample_weight : ndarray of shape (n_samples,)
        The weights for each observation in `X`.

    centers : ndarray of shape (n_clusters, n_features)
        The cluster centers before the current iteration

    centers_new : ndarray of shape (n_clusters, n_features)
        The cluster centers after the current iteration. Modified in-place.

    weight_sums : ndarray of shape (n_clusters,)
        The vector in which we keep track of the numbers of points in a
        cluster. This array is modified in place.

    random_state : RandomState instance
        Determines random number generation for low count centers reassignment.
        See :term:`Glossary <random_state>`.

    random_reassign : boolean, default=False
        If True, centers with very low counts are randomly reassigned
        to observations.

    reassignment_ratio : float, default=0.01
        Control the fraction of the maximum number of counts for a
        center to be reassigned. A higher value means that low count
        centers are more likely to be reassigned, which means that the
        model will take longer to converge, but should converge in a
        better clustering.

    verbose : bool, default=False
        Controls the verbosity.

    n_threads : int, default=1
        The number of OpenMP threads to use for the computation.

    Returns
    -------
    inertia : float
        Sum of squared distances of samples to their closest cluster center.
        The inertia is computed after finding the labels and before updating
        the centers.
    """
    # Perform label assignment to nearest centers
    # For better efficiency, it's better to run _mini_batch_step in a
    # threadpool_limit context than using _labels_inertia_threadpool_limit here
    labels, inertia = _labels_inertia(X, sample_weight, centers, n_threads=n_threads)

    # Update centers according to the labels
    if sp.issparse(X):
        _minibatch_update_sparse(
            X, sample_weight, centers, centers_new, weight_sums, labels, n_threads
        )
    else:
        _minibatch_update_dense(
            X,
            sample_weight,
            centers,
            centers_new,
            weight_sums,
            labels,
            n_threads,
        )

    # Reassign clusters that have very low weight
    if random_reassign and reassignment_ratio > 0:
        to_reassign = weight_sums < reassignment_ratio * weight_sums.max()

        # pick at most .5 * batch_size samples as new centers
        if to_reassign.sum() > 0.5 * X.shape[0]:
            indices_dont_reassign = np.argsort(weight_sums)[int(0.5 * X.shape[0]) :]
            to_reassign[indices_dont_reassign] = False
        n_reassigns = to_reassign.sum()

        if n_reassigns:
            # Pick new clusters amongst observations with uniform probability
            new_centers = random_state.choice(
                X.shape[0], replace=False, size=n_reassigns
            )
            if verbose:
                print(f"[MiniBatchKMeans] Reassigning {n_reassigns} cluster centers.")

            if sp.issparse(X):
                assign_rows_csr(
                    X,
                    new_centers.astype(np.intp, copy=False),
                    np.where(to_reassign)[0].astype(np.intp, copy=False),
                    centers_new,
                )
            else:
                centers_new[to_reassign] = X[new_centers]

        # reset counts of reassigned centers, but don't reset them too small
        # to avoid instant reassignment. This is a pretty dirty hack as it
        # also modifies the learning rates.
        weight_sums[to_reassign] = np.min(weight_sums[~to_reassign])

    return inertia


class MiniBatchKMeans(_BaseKMeans):
    # [CLASS DOC]
    # Mini-batch k-means estimator. Adds batch_size, compute_labels, max_no_improvement,
    # init_size and reassignment_ratio; n_init "auto" means 3 runs for random/callable
    # init. Fitted attributes: cluster_centers_, labels_ (when compute_labels), inertia_,
    # n_iter_, n_steps_, n_features_in_, feature_names_in_; private state _counts,
    # _ewa_inertia, _ewa_inertia_min, _no_improvement, _n_since_last_reassign.
    """
    Mini-Batch K-Means clustering.

    Read more in the :ref:`User Guide <mini_batch_kmeans>`.

    Parameters
    ----------

    n_clusters : int, default=8
        The number of clusters to form as well as the number of
        centroids to generate.

    init : {'k-means++', 'random'}, callable or array-like of shape \
            (n_clusters, n_features), default='k-means++'
        Method for initialization:

        'k-means++' : selects initial cluster centroids using sampling based on
        an empirical probability distribution of the points' contribution to the
        overall inertia. This technique speeds up convergence. The algorithm
        implemented is "greedy k-means++". It differs from the vanilla k-means++
        by making several trials at each sampling step and choosing the best centroid
        among them.

        'random': choose `n_clusters` observations (rows) at random from data
        for the initial centroids.

        If an array is passed, it should be of shape (n_clusters, n_features)
        and gives the initial centers.

        If a callable is passed, it should take arguments X, n_clusters and a
        random state and return an initialization.

        For an evaluation of the impact of initialization, see the example
        :ref:`sphx_glr_auto_examples_cluster_plot_kmeans_stability_low_dim_dense.py`.

    max_iter : int, default=100
        Maximum number of iterations over the complete dataset before
        stopping independently of any early stopping criterion heuristics.

    batch_size : int, default=1024
        Size of the mini batches.
        For faster computations, you can set `batch_size > 256 * number_of_cores`
        to enable :ref:`parallelism <lower-level-parallelism-with-openmp>`
        on all cores.

        .. versionchanged:: 1.0
           `batch_size` default changed from 100 to 1024.

    verbose : int, default=0
        Verbosity mode.

    compute_labels : bool, default=True
        Compute label assignment and inertia for the complete dataset
        once the minibatch optimization has converged in fit.

    random_state : int, RandomState instance or None, default=None
        Determines random number generation for centroid initialization and
        random reassignment. Use an int to make the randomness deterministic.
        See :term:`Glossary <random_state>`.

    tol : float, default=0.0
        Control early stopping based on the relative center changes as
        measured by a smoothed, variance-normalized of the mean center
        squared position changes. This early stopping heuristics is
        closer to the one used for the batch variant of the algorithms
        but induces a slight computational and memory overhead over the
        inertia heuristic.

        To disable convergence detection based on normalized center
        change, set tol to 0.0 (default).

    max_no_improvement : int, default=10
        Control early stopping based on the consecutive number of mini
        batches that does not yield an improvement on the smoothed inertia.

        To disable convergence detection based on inertia, set
        max_no_improvement to None.

    init_size : int, default=None
        Number of samples to randomly sample for speeding up the
        initialization (sometimes at the expense of accuracy): the
        only algorithm is initialized by running a batch KMeans on a
        random subset of the data. This needs to be larger than n_clusters.

        If `None`, the heuristic is `init_size = 3 * batch_size` if
        `3 * batch_size < n_clusters`, else `init_size = 3 * n_clusters`.

    n_init : 'auto' or int, default="auto"
        Number of random initializations that are tried.
        In contrast to KMeans, the algorithm is only run once, using the best of
        the `n_init` initializations as measured by inertia. Several runs are
        recommended for sparse high-dimensional problems (see
        :ref:`kmeans_sparse_high_dim`).

        When `n_init='auto'`, the number of runs depends on the value of init:
        3 if using `init='random'` or `init` is a callable;
        1 if using `init='k-means++'` or `init` is an array-like.

        .. versionadded:: 1.2
           Added 'auto' option for `n_init`.

        .. versionchanged:: 1.4
           Default value for `n_init` changed to `'auto'` in version.

    reassignment_ratio : float, default=0.01
        Control the fraction of the maximum number of counts for a center to
        be reassigned. A higher value means that low count centers are more
        easily reassigned, which means that the model will take longer to
        converge, but should converge in a better clustering. However, too high
        a value may cause convergence issues, especially with a small batch
        size.

    Attributes
    ----------

    cluster_centers_ : ndarray of shape (n_clusters, n_features)
        Coordinates of cluster centers.

    labels_ : ndarray of shape (n_samples,)
        Labels of each point (if compute_labels is set to True).

    inertia_ : float
        The value of the inertia criterion associated with the chosen
        partition if compute_labels is set to True. If compute_labels is set to
        False, it's an approximation of the inertia based on an exponentially
        weighted average of the batch inertiae.
        The inertia is defined as the sum of square distances of samples to
        their cluster center, weighted by the sample weights if provided.

    n_iter_ : int
        Number of iterations over the full dataset.

    n_steps_ : int
        Number of minibatches processed.

        .. versionadded:: 1.0

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    KMeans : The classic implementation of the clustering method based on the
        Lloyd's algorithm. It consumes the whole set of input data at each
        iteration.

    Notes
    -----
    See https://www.eecs.tufts.edu/~dsculley/papers/fastkmeans.pdf

    When there are too few points in the dataset, some centers may be
    duplicated, which means that a proper clustering in terms of the number
    of requesting clusters and the number of returned clusters will not
    always match. One solution is to set `reassignment_ratio=0`, which
    prevents reassignments of clusters that are too small.

    See :ref:`sphx_glr_auto_examples_cluster_plot_birch_vs_minibatchkmeans.py` for a
    comparison with :class:`~sklearn.cluster.BIRCH`.

    Examples
    --------
    >>> from sklearn.cluster import MiniBatchKMeans
    >>> import numpy as np
    >>> X = np.array([[1, 2], [1, 4], [1, 0],
    ...               [4, 2], [4, 0], [4, 4],
    ...               [4, 5], [0, 1], [2, 2],
    ...               [3, 2], [5, 5], [1, -1]])
    >>> # manually fit on batches
    >>> kmeans = MiniBatchKMeans(n_clusters=2,
    ...                          random_state=0,
    ...                          batch_size=6,
    ...                          n_init="auto")
    >>> kmeans = kmeans.partial_fit(X[0:6,:])
    >>> kmeans = kmeans.partial_fit(X[6:12,:])
    >>> kmeans.cluster_centers_
    array([[3.375, 3.  ],
           [0.75 , 0.5 ]])
    >>> kmeans.predict([[0, 0], [4, 4]])
    array([1, 0], dtype=int32)
    >>> # fit on the whole data
    >>> kmeans = MiniBatchKMeans(n_clusters=2,
    ...                          random_state=0,
    ...                          batch_size=6,
    ...                          max_iter=10,
    ...                          n_init="auto").fit(X)
    >>> kmeans.cluster_centers_
    array([[3.20967742, 3.56451613],
           [1.32758621, 0.77586207]])
    >>> kmeans.predict([[0, 0], [4, 4]])
    array([1, 0], dtype=int32)

    For a comparison of Mini-Batch K-Means clustering with other clustering algorithms,
    see :ref:`sphx_glr_auto_examples_cluster_plot_cluster_comparison.py`
    """

    _parameter_constraints: dict = {
        **_BaseKMeans._parameter_constraints,
        "batch_size": [Interval(Integral, 1, None, closed="left")],
        "compute_labels": ["boolean"],
        "max_no_improvement": [Interval(Integral, 0, None, closed="left"), None],
        "init_size": [Interval(Integral, 1, None, closed="left"), None],
        "reassignment_ratio": [Interval(Real, 0, None, closed="left")],
    }

    def __init__(
        self,
        n_clusters=8,
        *,
        init="k-means++",
        max_iter=100,
        batch_size=1024,
        verbose=0,
        compute_labels=True,
        random_state=None,
        tol=0.0,
        max_no_improvement=10,
        init_size=None,
        n_init="auto",
        reassignment_ratio=0.01,
    ):
        # [FUNCTION DOC]
        # Stores parameters via the base constructor plus the mini-batch specific ones.
        super().__init__(
            n_clusters=n_clusters,
            init=init,
            max_iter=max_iter,
            verbose=verbose,
            random_state=random_state,
            tol=tol,
            n_init=n_init,
        )

        self.max_no_improvement = max_no_improvement
        self.batch_size = batch_size
        self.compute_labels = compute_labels
        self.init_size = init_size
        self.reassignment_ratio = reassignment_ratio

    def _check_params_vs_input(self, X):
        # [FUNCTION DOC]
        # Runs base checks with default_n_init=3; sets _batch_size to min(batch_size,
        # n_samples); sets _init_size to init_size or 3 * _batch_size (raised to
        # 3 * n_clusters if smaller than n_clusters), warning with RuntimeWarning and using
        # 3 * n_clusters when a user init_size is below n_clusters; finally caps it at
        # n_samples.
        # Side effects: sets _tol, _n_init, _batch_size, _init_size.
        # KNOWN ISSUE: the reassignment_ratio < 0 ValueError branch is unreachable because
        # the parameter constraints already require it to be >= 0.
        super()._check_params_vs_input(X, default_n_init=3)

        self._batch_size = min(self.batch_size, X.shape[0])

        # init_size
        self._init_size = self.init_size
        if self._init_size is None:
            self._init_size = 3 * self._batch_size
            if self._init_size < self.n_clusters:
                self._init_size = 3 * self.n_clusters
        elif self._init_size < self.n_clusters:
            warnings.warn(
                (
                    f"init_size={self._init_size} should be larger than "
                    f"n_clusters={self.n_clusters}. Setting it to "
                    "min(3*n_clusters, n_samples)"
                ),
                RuntimeWarning,
                stacklevel=2,
            )
            self._init_size = 3 * self.n_clusters
        self._init_size = min(self._init_size, X.shape[0])

        # reassignment_ratio
        if self.reassignment_ratio < 0:
            raise ValueError(
                "reassignment_ratio should be >= 0, got "
                f"{self.reassignment_ratio} instead."
            )

    def _warn_mkl_vcomp(self, n_active_threads):
        # [FUNCTION DOC]
        # Emits a UserWarning about the Windows MKL memory leak suggesting a larger batch_size
        # or OMP_NUM_THREADS.
        """Warn when vcomp and mkl are both present"""
        warnings.warn(
            "MiniBatchKMeans is known to have a memory leak on "
            "Windows with MKL, when there are less chunks than "
            "available threads. You can prevent it by setting "
            f"batch_size >= {self._n_threads * CHUNK_SIZE} or by "
            "setting the environment variable "
            f"OMP_NUM_THREADS={n_active_threads}"
        )

    def _mini_batch_convergence(
        self, step, n_steps, n_samples, centers_squared_diff, batch_inertia
    ):
        # [FUNCTION DOC]
        # HELPER: Early stopping logic for fit. Normalizes the batch inertia by the batch size;
        # on the first step only prints and returns False. Otherwise updates the exponentially
        # weighted average inertia with alpha = min(1, 2 * batch_size / (n_samples + 1));
        # returns True if tol-based stopping is active and the squared center change is at
        # most _tol; tracks the minimum average inertia and counts steps without improvement,
        # returning True once that count reaches max_no_improvement (if not None).
        # Inputs: step - 0-based step; n_steps - total; n_samples; centers_squared_diff;
        # batch_inertia. Outputs: bool.
        # Side effects: updates _ewa_inertia, _ewa_inertia_min, _no_improvement; prints when
        # verbose.
        """Helper function to encapsulate the early stopping logic"""
        # Normalize inertia to be able to compare values when
        # batch_size changes
        batch_inertia /= self._batch_size

        # count steps starting from 1 for user friendly verbose mode.
        step = step + 1

        # Ignore first iteration because it's inertia from initialization.
        if step == 1:
            if self.verbose:
                print(
                    f"Minibatch step {step}/{n_steps}: mean batch "
                    f"inertia: {batch_inertia}"
                )
            return False

        # Compute an Exponentially Weighted Average of the inertia to
        # monitor the convergence while discarding minibatch-local stochastic
        # variability: https://en.wikipedia.org/wiki/Moving_average
        if self._ewa_inertia is None:
            self._ewa_inertia = batch_inertia
        else:
            alpha = self._batch_size * 2.0 / (n_samples + 1)
            alpha = min(alpha, 1)
            self._ewa_inertia = self._ewa_inertia * (1 - alpha) + batch_inertia * alpha

        # Log progress to be able to monitor convergence
        if self.verbose:
            print(
                f"Minibatch step {step}/{n_steps}: mean batch inertia: "
                f"{batch_inertia}, ewa inertia: {self._ewa_inertia}"
            )

        # Early stopping based on absolute tolerance on squared change of
        # centers position
        if self._tol > 0.0 and centers_squared_diff <= self._tol:
            if self.verbose:
                print(f"Converged (small centers change) at step {step}/{n_steps}")
            return True

        # Early stopping heuristic due to lack of improvement on smoothed
        # inertia
        if self._ewa_inertia_min is None or self._ewa_inertia < self._ewa_inertia_min:
            self._no_improvement = 0
            self._ewa_inertia_min = self._ewa_inertia
        else:
            self._no_improvement += 1

        if (
            self.max_no_improvement is not None
            and self._no_improvement >= self.max_no_improvement
        ):
            if self.verbose:
                print(
                    "Converged (lack of improvement in inertia) at step "
                    f"{step}/{n_steps}"
                )
            return True

        return False

    def _random_reassign(self):
        # [FUNCTION DOC]
        # HELPER: Adds _batch_size to the samples-seen counter and returns True (resetting the
        # counter) when some center has zero accumulated weight or at least 10 * n_clusters
        # samples were seen since the last reassignment; otherwise False.
        # Side effects: updates _n_since_last_reassign.
        # KNOWN ISSUE: in partial_fit the counter grows by _batch_size, which was derived from
        # the first batch, not by the actual size of the current batch.
        """Check if a random reassignment needs to be done.

        Do random reassignments each time 10 * n_clusters samples have been
        processed.

        If there are empty clusters we always want to reassign.
        """
        self._n_since_last_reassign += self._batch_size
        if (self._counts == 0).any() or self._n_since_last_reassign >= (
            10 * self.n_clusters
        ):
            self._n_since_last_reassign = 0
            return True
        return False

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None, sample_weight=None):
        # [FUNCTION DOC]
        # Steps: validate X; check parameters; resolve random_state, sample_weight and
        # threads; validate an array init; check the MKL/vcomp issue; compute squared norms;
        # draw a validation subset of _init_size rows (with replacement). For each of _n_init
        # initializations, initialize centers on a random init_size subset and keep those with
        # the lowest validation inertia. Reset counts and convergence state. Run
        # n_steps = max_iter * n_samples // _batch_size steps with BLAS limited to one thread:
        # sample a batch with replacement with probability proportional to sample_weight, run
        # _mini_batch_step with unit weights (weights already used in sampling) and random
        # reassignment as decided by _random_reassign, compute the squared center change when
        # tol > 0, swap center buffers, and stop early when _mini_batch_convergence says so.
        # Set n_steps_ and n_iter_ (passes over the data, rounded up). If compute_labels,
        # compute labels_ and inertia_ on the full data, otherwise approximate inertia_ as the
        # averaged inertia times the total weight.
        # Inputs: X - array-like or sparse; y - ignored; sample_weight - array-like or None.
        # Outputs: self.
        # Side effects: sets fitted and private attributes; consumes random_state; prints when
        # verbose.
        # KNOWN ISSUE: with compute_labels=False and a single step (e.g. max_iter=1 and
        # batch_size >= n_samples), _ewa_inertia is still None and computing inertia_ raises
        # TypeError.
        """Compute the centroids on X by chunking it into mini-batches.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training instances to cluster. It must be noted that the data
            will be converted to C ordering, which will cause a memory copy
            if the given data is not C-contiguous.
            If a sparse matrix is passed, a copy will be made if it's not in
            CSR format.

        y : Ignored
            Not used, present here for API consistency by convention.

        sample_weight : array-like of shape (n_samples,), default=None
            The weights for each observation in X. If None, all observations
            are assigned equal weight. `sample_weight` is not used during
            initialization if `init` is a callable or a user provided array.

            .. versionadded:: 0.20

        Returns
        -------
        self : object
            Fitted estimator.
        """
        X = validate_data(
            self,
            X,
            accept_sparse="csr",
            dtype=[np.float64, np.float32],
            order="C",
            accept_large_sparse=False,
        )

        self._check_params_vs_input(X)
        random_state = check_random_state(self.random_state)
        sample_weight = _check_sample_weight(sample_weight, X, dtype=X.dtype)
        self._n_threads = _openmp_effective_n_threads()
        n_samples, n_features = X.shape

        # Validate init array
        init = self.init
        if _is_arraylike_not_scalar(init):
            init = check_array(init, dtype=X.dtype, copy=True, order="C")
            self._validate_center_shape(X, init)

        self._check_mkl_vcomp(X, self._batch_size)

        # precompute squared norms of data points
        x_squared_norms = row_norms(X, squared=True)

        # Validation set for the init
        validation_indices = random_state.randint(0, n_samples, self._init_size)
        X_valid = X[validation_indices]
        sample_weight_valid = sample_weight[validation_indices]

        # perform several inits with random subsets
        best_inertia = None
        for init_idx in range(self._n_init):
            if self.verbose:
                print(f"Init {init_idx + 1}/{self._n_init} with method {init}")

            # Initialize the centers using only a fraction of the data as we
            # expect n_samples to be very large when using MiniBatchKMeans.
            cluster_centers = self._init_centroids(
                X,
                x_squared_norms=x_squared_norms,
                init=init,
                random_state=random_state,
                init_size=self._init_size,
                sample_weight=sample_weight,
            )

            # Compute inertia on a validation set.
            _, inertia = _labels_inertia_threadpool_limit(
                X_valid,
                sample_weight_valid,
                cluster_centers,
                n_threads=self._n_threads,
            )

            if self.verbose:
                print(f"Inertia for init {init_idx + 1}/{self._n_init}: {inertia}")
            if best_inertia is None or inertia < best_inertia:
                init_centers = cluster_centers
                best_inertia = inertia

        centers = init_centers
        centers_new = np.empty_like(centers)

        # Initialize counts
        self._counts = np.zeros(self.n_clusters, dtype=X.dtype)

        # Attributes to monitor the convergence
        self._ewa_inertia = None
        self._ewa_inertia_min = None
        self._no_improvement = 0

        # Initialize number of samples seen since last reassignment
        self._n_since_last_reassign = 0

        sum_of_weights = np.sum(sample_weight)

        n_steps = (self.max_iter * n_samples) // self._batch_size
        normalized_sample_weight = sample_weight / sum_of_weights
        unit_sample_weight = np.ones_like(sample_weight, shape=(self._batch_size,))

        with _get_threadpool_controller().limit(limits=1, user_api="blas"):
            # Perform the iterative optimization until convergence
            for i in range(n_steps):
                # Sample a minibatch from the full dataset
                minibatch_indices = random_state.choice(
                    n_samples,
                    self._batch_size,
                    p=normalized_sample_weight,
                    replace=True,
                )
                # Perform the actual update step on the minibatch data
                # Note: since the sampling of the minibatch is sample_weight aware,
                # we pass fixed unit weights to the `_mini_batch_step` call to avoid
                # accounting for the weights twice. Also note that `_mini_batch_step`
                # can be called with non-unit weights when the caller constructs
                # the batches explicitly by calling the public `partial_fit` method
                # instead.
                batch_inertia = _mini_batch_step(
                    X=X[minibatch_indices],
                    sample_weight=unit_sample_weight,
                    centers=centers,
                    centers_new=centers_new,
                    weight_sums=self._counts,
                    random_state=random_state,
                    random_reassign=self._random_reassign(),
                    reassignment_ratio=self.reassignment_ratio,
                    verbose=self.verbose,
                    n_threads=self._n_threads,
                )

                if self._tol > 0.0:
                    centers_squared_diff = np.sum((centers_new - centers) ** 2)
                else:
                    centers_squared_diff = 0

                centers, centers_new = centers_new, centers

                # Monitor convergence and do early stopping if necessary
                if self._mini_batch_convergence(
                    i, n_steps, n_samples, centers_squared_diff, batch_inertia
                ):
                    break

        self.cluster_centers_ = centers
        self._n_features_out = self.cluster_centers_.shape[0]

        self.n_steps_ = i + 1
        self.n_iter_ = int(np.ceil(((i + 1) * self._batch_size) / n_samples))

        if self.compute_labels:
            self.labels_, self.inertia_ = _labels_inertia_threadpool_limit(
                X,
                sample_weight,
                self.cluster_centers_,
                n_threads=self._n_threads,
            )
        else:
            self.inertia_ = self._ewa_inertia * sum_of_weights

        return self

    @_fit_context(prefer_skip_nested_validation=True)
    def partial_fit(self, X, y=None, sample_weight=None):
        # [FUNCTION DOC]
        # Updates the model with one batch. Validates X (resetting n_features_in_ only on the
        # first call), keeps a persistent _random_state, validates sample_weight and
        # initializes n_steps_ to 0 if absent. On the first call (no cluster_centers_):
        # checks parameters, threads, an array init and MKL/vcomp, initializes centers from
        # this batch (using init_size) and zeroes the counts and reassignment counter. Then
        # runs one _mini_batch_step in place on cluster_centers_ with the given weights; if
        # compute_labels, sets labels_ and inertia_ for this batch; increments n_steps_.
        # Inputs: X - batch (n_samples, n_features); y - ignored; sample_weight.
        # Outputs: self.
        # Exceptions: ValueError on feature mismatch with earlier calls or too few samples on
        # the first call; validation errors.
        # Side effects: mutates cluster_centers_, _counts, n_steps_, labels_, inertia_ and
        # private state; consumes _random_state.
        """Update k means estimate on a single mini-batch X.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training instances to cluster. It must be noted that the data
            will be converted to C ordering, which will cause a memory copy
            if the given data is not C-contiguous.
            If a sparse matrix is passed, a copy will be made if it's not in
            CSR format.

        y : Ignored
            Not used, present here for API consistency by convention.

        sample_weight : array-like of shape (n_samples,), default=None
            The weights for each observation in X. If None, all observations
            are assigned equal weight. `sample_weight` is not used during
            initialization if `init` is a callable or a user provided array.

        Returns
        -------
        self : object
            Return updated estimator.
        """
        has_centers = hasattr(self, "cluster_centers_")

        X = validate_data(
            self,
            X,
            accept_sparse="csr",
            dtype=[np.float64, np.float32],
            order="C",
            accept_large_sparse=False,
            reset=not has_centers,
        )

        self._random_state = getattr(
            self, "_random_state", check_random_state(self.random_state)
        )
        sample_weight = _check_sample_weight(sample_weight, X, dtype=X.dtype)
        self.n_steps_ = getattr(self, "n_steps_", 0)

        # precompute squared norms of data points
        x_squared_norms = row_norms(X, squared=True)

        if not has_centers:
            # this instance has not been fitted yet (fit or partial_fit)
            self._check_params_vs_input(X)
            self._n_threads = _openmp_effective_n_threads()

            # Validate init array
            init = self.init
            if _is_arraylike_not_scalar(init):
                init = check_array(init, dtype=X.dtype, copy=True, order="C")
                self._validate_center_shape(X, init)

            self._check_mkl_vcomp(X, X.shape[0])

            # initialize the cluster centers
            self.cluster_centers_ = self._init_centroids(
                X,
                x_squared_norms=x_squared_norms,
                init=init,
                random_state=self._random_state,
                init_size=self._init_size,
                sample_weight=sample_weight,
            )

            # Initialize counts
            self._counts = np.zeros(self.n_clusters, dtype=X.dtype)

            # Initialize number of samples seen since last reassignment
            self._n_since_last_reassign = 0

        with _get_threadpool_controller().limit(limits=1, user_api="blas"):
            _mini_batch_step(
                X,
                sample_weight=sample_weight,
                centers=self.cluster_centers_,
                centers_new=self.cluster_centers_,
                weight_sums=self._counts,
                random_state=self._random_state,
                random_reassign=self._random_reassign(),
                reassignment_ratio=self.reassignment_ratio,
                verbose=self.verbose,
                n_threads=self._n_threads,
            )

        if self.compute_labels:
            self.labels_, self.inertia_ = _labels_inertia_threadpool_limit(
                X,
                sample_weight,
                self.cluster_centers_,
                n_threads=self._n_threads,
            )

        self.n_steps_ += 1
        self._n_features_out = self.cluster_centers_.shape[0]

        return self
