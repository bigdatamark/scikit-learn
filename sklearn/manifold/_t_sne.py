# [MODULE DOC]
# Implementation of t-distributed Stochastic Neighbor Embedding (t-SNE) with exact
#   O(N^2) and Barnes-Hut O(N log N) gradients. Provides the joint-probability
#   computations (dense and nearest-neighbor sparse), the KL divergence objectives with
#   gradients, a momentum/gain gradient descent optimizer, the trustworthiness score and
#   the TSNE estimator. MACHINE_EPSILON (float64 eps) is used as a floor for
#   probabilities and sums.
# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

# This is the exact and Barnes-Hut t-SNE implementation. There are other
# modifications of the algorithm:
# * Fast Optimization for t-SNE:
#   https://cseweb.ucsd.edu/~lvdmaaten/workshops/nips2010/papers/vandermaaten.pdf

from numbers import Integral, Real
from time import time

import numpy as np
from scipy import linalg
from scipy.sparse import csr_array, issparse
from scipy.spatial.distance import pdist, squareform

from sklearn.base import (
    BaseEstimator,
    ClassNamePrefixFeaturesOutMixin,
    TransformerMixin,
    _fit_context,
)
from sklearn.decomposition import PCA
from sklearn.manifold import _barnes_hut_tsne, _utils
from sklearn.metrics.pairwise import _VALID_METRICS, pairwise_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import _align_api_if_sparse, check_random_state
from sklearn.utils._openmp_helpers import _openmp_effective_n_threads
from sklearn.utils._param_validation import Interval, StrOptions, validate_params
from sklearn.utils.validation import _num_samples, check_non_negative, validate_data

MACHINE_EPSILON = np.finfo(np.double).eps


def _joint_probabilities(distances, desired_perplexity, verbose):
    # [FUNCTION DOC]
    # HELPER: Converts input-space squared distances into symmetric joint probabilities
    #   for the exact method. Casts distances to float32, runs the perplexity binary
    #   search to get row-wise conditional probabilities, symmetrizes by adding the
    #   transpose, converts to condensed form, divides by the total (floored at
    #   MACHINE_EPSILON) and floors every entry at MACHINE_EPSILON.
    # Inputs: distances - square (n_samples, n_samples) array of (squared) distances;
    #   desired_perplexity - float; verbose - int verbosity passed to the binary search.
    # Outputs: condensed float array of length n_samples * (n_samples - 1) / 2.
    # Exceptions: none raised directly.
    # Side effects: may print progress from the binary search.
    # KNOWN ISSUE: the docstring describes distances as a condensed 1-D array, but the
    #   code (binary search per row, transpose, squareform) requires a square matrix.
    """Compute joint probabilities p_ij from distances.

    Parameters
    ----------
    distances : ndarray of shape (n_samples * (n_samples-1) / 2,)
        Distances of samples are stored as condensed matrices, i.e.
        we omit the diagonal and duplicate entries and store everything
        in a one-dimensional array.

    desired_perplexity : float
        Desired perplexity of the joint probability distributions.

    verbose : int
        Verbosity level.

    Returns
    -------
    P : ndarray of shape (n_samples * (n_samples-1) / 2,)
        Condensed joint probability matrix.
    """
    # Compute conditional probabilities such that they approximately match
    # the desired perplexity
    distances = distances.astype(np.float32, copy=False)
    conditional_P = _utils._binary_search_perplexity(
        distances, desired_perplexity, verbose
    )
    P = conditional_P + conditional_P.T
    sum_P = np.maximum(np.sum(P), MACHINE_EPSILON)
    P = np.maximum(squareform(P) / sum_P, MACHINE_EPSILON)
    return P


def _joint_probabilities_nn(distances, desired_perplexity, verbose):
    # [FUNCTION DOC]
    # HELPER: Sparse variant of _joint_probabilities restricted to k nearest neighbors.
    #   Sorts the CSR indices in place, reshapes the stored distances into (n_samples,
    #   k) rows (requires the same number of stored neighbors per row), runs the
    #   perplexity binary search on them, rebuilds a CSR array of conditional
    #   probabilities with the original sparsity pattern, symmetrizes it by adding its
    #   transpose, and divides by the total (floored at MACHINE_EPSILON).
    # Inputs: distances - CSR (n_samples, n_samples) of squared neighbor distances;
    #   desired_perplexity - float; verbose - int.
    # Outputs: CSR sparse matrix/array (container aligned via _align_api_if_sparse) of
    #   joint probabilities.
    # Exceptions: AssertionError if probabilities are not finite or exceed 1; ValueError
    #   from reshape if rows have differing neighbor counts.
    # Side effects: sorts indices of the input matrix in place; prints timing if verbose
    #   >= 2.
    # KNOWN ISSUE: the docstring says the dense version is O(N); it is O(N^2).
    """Compute joint probabilities p_ij from distances using just nearest
    neighbors.

    This method is approximately equal to _joint_probabilities. The latter
    is O(N), but limiting the joint probability to nearest neighbors improves
    this substantially to O(uN).

    Parameters
    ----------
    distances : sparse matrix of shape (n_samples, n_samples)
        Distances of samples to its n_neighbors nearest neighbors. All other
        distances are left to zero (and are not materialized in memory).
        Matrix should be of CSR format.

    desired_perplexity : float
        Desired perplexity of the joint probability distributions.

    verbose : int
        Verbosity level.

    Returns
    -------
    P : sparse matrix of shape (n_samples, n_samples)
        Condensed joint probability matrix with only nearest neighbors. Matrix
        will be of CSR format.
    """
    t0 = time()
    # Compute conditional probabilities such that they approximately match
    # the desired perplexity
    distances.sort_indices()
    n_samples = distances.shape[0]
    distances_data = distances.data.reshape(n_samples, -1)
    distances_data = distances_data.astype(np.float32, copy=False)
    conditional_P = _utils._binary_search_perplexity(
        distances_data, desired_perplexity, verbose
    )
    assert np.all(np.isfinite(conditional_P)), "All probabilities should be finite"

    # Symmetrize the joint probability distribution using sparse operations
    P = csr_array(
        (conditional_P.ravel(), distances.indices, distances.indptr),
        shape=(n_samples, n_samples),
    )
    P = P + P.T

    # Normalize the joint probability distribution
    sum_P = np.maximum(P.sum(), MACHINE_EPSILON)
    P /= sum_P

    assert np.all(np.abs(P.data) <= 1.0)
    if verbose >= 2:
        duration = time() - t0
        print("[t-SNE] Computed conditional probabilities in {:.3f}s".format(duration))
    return _align_api_if_sparse(P)


def _kl_divergence(
    params,
    P,
    degrees_of_freedom,
    n_samples,
    n_components,
    skip_num_points=0,
    compute_error=True,
):
    # [FUNCTION DOC]
    # HELPER: Exact t-SNE objective. Reshapes params to (n_samples, n_components),
    #   computes the condensed Student-t kernel (1 + squared distance / dof) to the
    #   power -(dof + 1) / 2, normalizes it into Q (total of the square matrix is twice
    #   the condensed sum, floored at MACHINE_EPSILON). The error is twice the dot
    #   product of P with log(max(P, eps) / Q), or NaN when compute_error is False. For
    #   each point i from skip_num_points onward, the gradient row is the sum over j of
    #   (p_ij - q_ij) times the kernel value times (y_i - y_j); the flattened gradient
    #   is multiplied by 2 (dof + 1) / dof.
    # Inputs: params - flattened embedding (n_samples * n_components,); P - condensed
    #   joint probabilities; degrees_of_freedom - positive number; n_samples,
    #   n_components - ints; skip_num_points - int (default 0); compute_error - bool
    #   (default True).
    # Outputs: tuple (kl_divergence float or NaN, gradient flat array with params'
    #   dtype).
    # Exceptions: none raised directly.
    # Side effects: none.
    # KNOWN ISSUE: the gradient buffer is allocated uninitialized, so rows below
    #   skip_num_points contain arbitrary memory instead of zeros.
    """t-SNE objective function: gradient of the KL divergence
    of p_ijs and q_ijs and the absolute error.

    Parameters
    ----------
    params : ndarray of shape (n_params,)
        Unraveled embedding.

    P : ndarray of shape (n_samples * (n_samples-1) / 2,)
        Condensed joint probability matrix.

    degrees_of_freedom : int
        Degrees of freedom of the Student's-t distribution.

    n_samples : int
        Number of samples.

    n_components : int
        Dimension of the embedded space.

    skip_num_points : int, default=0
        This does not compute the gradient for points with indices below
        `skip_num_points`. This is useful when computing transforms of new
        data where you'd like to keep the old data fixed.

    compute_error: bool, default=True
        If False, the kl_divergence is not computed and returns NaN.

    Returns
    -------
    kl_divergence : float
        Kullback-Leibler divergence of p_ij and q_ij.

    grad : ndarray of shape (n_params,)
        Unraveled gradient of the Kullback-Leibler divergence with respect to
        the embedding.
    """
    X_embedded = params.reshape(n_samples, n_components)

    # Q is a heavy-tailed distribution: Student's t-distribution
    dist = pdist(X_embedded, "sqeuclidean")
    dist /= degrees_of_freedom
    dist += 1.0
    dist **= (degrees_of_freedom + 1.0) / -2.0
    Q = np.maximum(dist / (2.0 * np.sum(dist)), MACHINE_EPSILON)

    # Optimization trick below: np.dot(x, y) is faster than
    # np.sum(x * y) because it calls BLAS

    # Objective: C (Kullback-Leibler divergence of P and Q)
    if compute_error:
        kl_divergence = 2.0 * np.dot(P, np.log(np.maximum(P, MACHINE_EPSILON) / Q))
    else:
        kl_divergence = np.nan

    # Gradient: dC/dY
    # pdist always returns double precision distances. Thus we need to take
    grad = np.ndarray((n_samples, n_components), dtype=params.dtype)
    PQd = squareform((P - Q) * dist)
    for i in range(skip_num_points, n_samples):
        grad[i] = np.dot(np.ravel(PQd[i], order="K"), X_embedded[i] - X_embedded)
    grad = grad.ravel()
    c = 2.0 * (degrees_of_freedom + 1.0) / degrees_of_freedom
    grad *= c

    return kl_divergence, grad


def _kl_divergence_bh(
    params,
    P,
    degrees_of_freedom,
    n_samples,
    n_components,
    angle=0.5,
    skip_num_points=0,
    verbose=False,
    compute_error=True,
    num_threads=1,
):
    # [FUNCTION DOC]
    # HELPER: Barnes-Hut t-SNE objective. Casts params to float32, reshapes into the
    #   embedding, extracts P's data/indices/indptr (indices as int64) and calls the
    #   Cython _barnes_hut_tsne.gradient which fills a zero-initialized gradient buffer
    #   using attractive forces over the sparse neighbors and a space-partitioning tree
    #   with opening angle angle for repulsive forces, and returns the KL error. Scales
    #   the flattened gradient by 2 (dof + 1) / dof.
    # Inputs: params - flat embedding; P - CSR joint probabilities; degrees_of_freedom,
    #   n_samples, n_components; angle - float (default 0.5); skip_num_points (default
    #   0); verbose (default False); compute_error (default True); num_threads (default
    #   1).
    # Outputs: tuple (error float, float32 flat gradient).
    # Exceptions: errors from the Cython routine on inconsistent inputs.
    # Side effects: may print from the Cython routine when verbose.
    # KNOWN ISSUE: skip_num_points is accepted but never forwarded to the gradient
    #   routine, so the gradient is computed for all points.
    """t-SNE objective function: KL divergence of p_ijs and q_ijs.

    Uses Barnes-Hut tree methods to calculate the gradient that
    runs in O(NlogN) instead of O(N^2).

    Parameters
    ----------
    params : ndarray of shape (n_params,)
        Unraveled embedding.

    P : sparse matrix of shape (n_samples, n_sample)
        Sparse approximate joint probability matrix, computed only for the
        k nearest-neighbors and symmetrized. Matrix should be of CSR format.

    degrees_of_freedom : int
        Degrees of freedom of the Student's-t distribution.

    n_samples : int
        Number of samples.

    n_components : int
        Dimension of the embedded space.

    angle : float, default=0.5
        This is the trade-off between speed and accuracy for Barnes-Hut T-SNE.
        'angle' is the angular size (referred to as theta in [3]) of a distant
        node as measured from a point. If this size is below 'angle' then it is
        used as a summary node of all points contained within it.
        This method is not very sensitive to changes in this parameter
        in the range of 0.2 - 0.8. Angle less than 0.2 has quickly increasing
        computation time and angle greater 0.8 has quickly increasing error.

    skip_num_points : int, default=0
        This does not compute the gradient for points with indices below
        `skip_num_points`. This is useful when computing transforms of new
        data where you'd like to keep the old data fixed.

    verbose : int, default=False
        Verbosity level.

    compute_error: bool, default=True
        If False, the kl_divergence is not computed and returns NaN.

    num_threads : int, default=1
        Number of threads used to compute the gradient. This is set here to
        avoid calling _openmp_effective_n_threads for each gradient step.

    Returns
    -------
    kl_divergence : float
        Kullback-Leibler divergence of p_ij and q_ij.

    grad : ndarray of shape (n_params,)
        Unraveled gradient of the Kullback-Leibler divergence with respect to
        the embedding.
    """
    params = params.astype(np.float32, copy=False)
    X_embedded = params.reshape(n_samples, n_components)

    val_P = P.data.astype(np.float32, copy=False)
    neighbors = P.indices.astype(np.int64, copy=False)
    indptr = P.indptr.astype(np.int64, copy=False)

    grad = np.zeros(X_embedded.shape, dtype=np.float32)
    error = _barnes_hut_tsne.gradient(
        val_P,
        X_embedded,
        neighbors,
        indptr,
        grad,
        angle,
        n_components,
        verbose,
        dof=degrees_of_freedom,
        compute_error=compute_error,
        num_threads=num_threads,
    )
    c = 2.0 * (degrees_of_freedom + 1.0) / degrees_of_freedom
    grad = grad.ravel()
    grad *= c

    return error, grad


def _gradient_descent(
    objective,
    p0,
    it,
    max_iter,
    n_iter_check=1,
    n_iter_without_progress=300,
    momentum=0.8,
    learning_rate=200.0,
    min_gain=0.01,
    min_grad_norm=1e-7,
    verbose=0,
    args=None,
    kwargs=None,
):
    # [FUNCTION DOC]
    # HELPER: Batch gradient descent with momentum and per-parameter adaptive gains
    #   (Jacobs delta-bar-delta style). Starting from a copy of p0 flattened, for
    #   iterations it to max_iter - 1: set kwargs["compute_error"] only on check
    #   iterations (every n_iter_check) and the last one, call the objective, increase
    #   gains by 0.2 where the previous update and the gradient have opposite signs and
    #   multiply them by 0.8 elsewhere, clip gains at min_gain, scale the gradient by
    #   the gains, update = momentum * update - learning_rate * gradient, and add it to
    #   p. On check iterations compute the norm of the gain-scaled gradient, optionally
    #   print progress, track the best error/iteration, and stop when no improvement
    #   happened for more than n_iter_without_progress iterations or when the gradient
    #   norm is at or below min_grad_norm.
    # Inputs: objective - callable(p, *args, **kwargs) returning (error, grad); p0 -
    #   initial parameters; it - starting iteration index; max_iter - exclusive end
    #   iteration; n_iter_check (1); n_iter_without_progress (300); momentum (0.8);
    #   learning_rate (200.0); min_gain (0.01); min_grad_norm (1e-7); verbose (0); args
    #   (None -> []); kwargs (None -> {}).
    # Outputs: tuple (p optimized flat array, last error evaluated, index of the last
    #   iteration run; equals it if the loop does not run).
    # Exceptions: none raised directly.
    # Side effects: mutates the kwargs dict passed in (sets "compute_error"); prints
    #   when verbose >= 2.
    """Batch gradient descent with momentum and individual gains.

    Parameters
    ----------
    objective : callable
        Should return a tuple of cost and gradient for a given parameter
        vector. When expensive to compute, the cost can optionally
        be None and can be computed every n_iter_check steps using
        the objective_error function.

    p0 : array-like of shape (n_params,)
        Initial parameter vector.

    it : int
        Current number of iterations (this function will be called more than
        once during the optimization).

    max_iter : int
        Maximum number of gradient descent iterations.

    n_iter_check : int, default=1
        Number of iterations before evaluating the global error. If the error
        is sufficiently low, we abort the optimization.

    n_iter_without_progress : int, default=300
        Maximum number of iterations without progress before we abort the
        optimization.

    momentum : float within (0.0, 1.0), default=0.8
        The momentum generates a weight for previous gradients that decays
        exponentially.

    learning_rate : float, default=200.0
        The learning rate for t-SNE is usually in the range [10.0, 1000.0]. If
        the learning rate is too high, the data may look like a 'ball' with any
        point approximately equidistant from its nearest neighbours. If the
        learning rate is too low, most points may look compressed in a dense
        cloud with few outliers.

    min_gain : float, default=0.01
        Minimum individual gain for each parameter.

    min_grad_norm : float, default=1e-7
        If the gradient norm is below this threshold, the optimization will
        be aborted.

    verbose : int, default=0
        Verbosity level.

    args : sequence, default=None
        Arguments to pass to objective function.

    kwargs : dict, default=None
        Keyword arguments to pass to objective function.

    Returns
    -------
    p : ndarray of shape (n_params,)
        Optimum parameters.

    error : float
        Optimum.

    i : int
        Last iteration.
    """
    if args is None:
        args = []
    if kwargs is None:
        kwargs = {}

    p = p0.copy().ravel()
    update = np.zeros_like(p)
    gains = np.ones_like(p)
    error = np.finfo(float).max
    best_error = np.finfo(float).max
    best_iter = i = it

    tic = time()
    for i in range(it, max_iter):
        check_convergence = (i + 1) % n_iter_check == 0
        # only compute the error when needed
        kwargs["compute_error"] = check_convergence or i == max_iter - 1

        error, grad = objective(p, *args, **kwargs)

        inc = update * grad < 0.0
        dec = np.invert(inc)
        gains[inc] += 0.2
        gains[dec] *= 0.8
        np.clip(gains, min_gain, np.inf, out=gains)
        grad *= gains
        update = momentum * update - learning_rate * grad
        p += update

        if check_convergence:
            toc = time()
            duration = toc - tic
            tic = toc
            grad_norm = linalg.norm(grad)

            if verbose >= 2:
                print(
                    "[t-SNE] Iteration %d: error = %.7f,"
                    " gradient norm = %.7f"
                    " (%s iterations in %0.3fs)"
                    % (i + 1, error, grad_norm, n_iter_check, duration)
                )

            if error < best_error:
                best_error = error
                best_iter = i
            elif i - best_iter > n_iter_without_progress:
                if verbose >= 2:
                    print(
                        "[t-SNE] Iteration %d: did not make any progress "
                        "during the last %d episodes. Finished."
                        % (i + 1, n_iter_without_progress)
                    )
                break
            if grad_norm <= min_grad_norm:
                if verbose >= 2:
                    print(
                        "[t-SNE] Iteration %d: gradient norm %f. Finished."
                        % (i + 1, grad_norm)
                    )
                break

    return p, error, i


@validate_params(
    {
        "X": ["array-like", "sparse matrix"],
        "X_embedded": ["array-like", "sparse matrix"],
        "n_neighbors": [Interval(Integral, 1, None, closed="left")],
        "metric": [StrOptions(set(_VALID_METRICS) | {"precomputed"}), callable],
    },
    prefer_skip_nested_validation=True,
)
def trustworthiness(X, X_embedded, *, n_neighbors=5, metric="euclidean"):
    # [FUNCTION DOC]
    # Computes the trustworthiness score. Validates n_neighbors < n_samples / 2,
    #   computes the full input-space distance matrix (copied when precomputed so the
    #   input is not modified), sets the diagonal to infinity and argsorts each row to
    #   get input-space neighbor rankings. Finds the n_neighbors nearest neighbors of
    #   each sample in X_embedded (excluding itself), builds an inverse permutation
    #   giving each sample's 1-based rank in the input space, subtracts n_neighbors from
    #   the ranks of embedded neighbors, sums the positive values and normalizes as in
    #   the docstring formula.
    # Inputs/Outputs: as documented; returns a Python float.
    # Exceptions: ValueError when n_neighbors >= n_samples / 2; parameter validation
    #   errors.
    # Side effects: none on inputs. Memory is O(n_samples^2).
    r"""Indicate to what extent the local structure is retained.

    The trustworthiness is within [0, 1]. It is defined as

    .. math::

        T(k) = 1 - \frac{2}{nk (2n - 3k - 1)} \sum^n_{i=1}
            \sum_{j \in \mathcal{N}_{i}^{k}} \max(0, (r(i, j) - k))

    where for each sample i, :math:`\mathcal{N}_{i}^{k}` are its k nearest
    neighbors in the output space, and every sample j is its :math:`r(i, j)`-th
    nearest neighbor in the input space. In other words, any unexpected nearest
    neighbors in the output space are penalised in proportion to their rank in
    the input space.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features) or \
        (n_samples, n_samples)
        If the metric is 'precomputed' X must be a square distance
        matrix. Otherwise it contains a sample per row.

    X_embedded : {array-like, sparse matrix} of shape (n_samples, n_components)
        Embedding of the training data in low-dimensional space.

    n_neighbors : int, default=5
        The number of neighbors that will be considered. Should be fewer than
        `n_samples / 2` to ensure the trustworthiness to lies within [0, 1], as
        mentioned in [1]_. An error will be raised otherwise.

    metric : str or callable, default='euclidean'
        Which metric to use for computing pairwise distances between samples
        from the original input space. If metric is 'precomputed', X must be a
        matrix of pairwise distances or squared distances. Otherwise, for a list
        of available metrics, see the documentation of argument metric in
        `sklearn.pairwise.pairwise_distances` and metrics listed in
        `sklearn.metrics.pairwise.PAIRWISE_DISTANCE_FUNCTIONS`. Note that the
        "cosine" metric uses :func:`~sklearn.metrics.pairwise.cosine_distances`.

        .. versionadded:: 0.20

    Returns
    -------
    trustworthiness : float
        Trustworthiness of the low-dimensional embedding.

    References
    ----------
    .. [1] Jarkko Venna and Samuel Kaski. 2001. Neighborhood
           Preservation in Nonlinear Projection Methods: An Experimental Study.
           In Proceedings of the International Conference on Artificial Neural Networks
           (ICANN '01). Springer-Verlag, Berlin, Heidelberg, 485-491.

    .. [2] Laurens van der Maaten. Learning a Parametric Embedding by Preserving
           Local Structure. Proceedings of the Twelfth International Conference on
           Artificial Intelligence and Statistics, PMLR 5:384-391, 2009.

    Examples
    --------
    >>> from sklearn.datasets import make_blobs
    >>> from sklearn.decomposition import PCA
    >>> from sklearn.manifold import trustworthiness
    >>> X, _ = make_blobs(n_samples=100, n_features=10, centers=3, random_state=42)
    >>> X_embedded = PCA(n_components=2).fit_transform(X)
    >>> print(f"{trustworthiness(X, X_embedded, n_neighbors=5):.2f}")
    0.92
    """
    n_samples = _num_samples(X)
    if n_neighbors >= n_samples / 2:
        raise ValueError(
            f"n_neighbors ({n_neighbors}) should be less than n_samples / 2"
            f" ({n_samples / 2})"
        )
    dist_X = pairwise_distances(X, metric=metric)
    if metric == "precomputed":
        dist_X = dist_X.copy()
    # we set the diagonal to np.inf to exclude the points themselves from
    # their own neighborhood
    np.fill_diagonal(dist_X, np.inf)
    ind_X = np.argsort(dist_X, axis=1)
    # `ind_X[i]` is the index of sorted distances between i and other samples
    ind_X_embedded = (
        NearestNeighbors(n_neighbors=n_neighbors)
        .fit(X_embedded)
        .kneighbors(return_distance=False)
    )

    # We build an inverted index of neighbors in the input space: For sample i,
    # we define `inverted_index[i]` as the inverted index of sorted distances:
    # inverted_index[i][ind_X[i]] = np.arange(1, n_sample + 1)
    inverted_index = np.zeros((n_samples, n_samples), dtype=int)
    ordered_indices = np.arange(n_samples + 1)
    inverted_index[ordered_indices[:-1, np.newaxis], ind_X] = ordered_indices[1:]
    ranks = (
        inverted_index[ordered_indices[:-1, np.newaxis], ind_X_embedded] - n_neighbors
    )
    t = np.sum(ranks[ranks > 0])
    t = 1.0 - t * (
        2.0 / (n_samples * n_neighbors * (2.0 * n_samples - 3.0 * n_neighbors - 1.0))
    )
    return t


class TSNE(ClassNamePrefixFeaturesOutMixin, TransformerMixin, BaseEstimator):
    # [CLASS DOC]
    # t-SNE estimator. Hyperparameters are stored unchanged in __init__ and validated
    #   through _parameter_constraints at fit time. Class constants
    #   _EXPLORATION_MAX_ITER (250, length of the early exaggeration phase) and
    #   _N_ITER_CHECK (50, error check interval) control the optimization schedule.
    #   Fitted attributes: embedding_, kl_divergence_, n_iter_, learning_rate_,
    #   n_features_in_ and feature_names_in_. Has no separate transform method; only
    #   fit_transform produces embeddings.
    """T-distributed Stochastic Neighbor Embedding.

    t-SNE [1] is a tool to visualize high-dimensional data. It converts
    similarities between data points to joint probabilities and tries
    to minimize the Kullback-Leibler divergence between the joint
    probabilities of the low-dimensional embedding and the
    high-dimensional data. t-SNE has a cost function that is not convex,
    i.e. with different initializations we can get different results.

    It is highly recommended to use another dimensionality reduction
    method (e.g. PCA for dense data or TruncatedSVD for sparse data)
    to reduce the number of dimensions to a reasonable amount (e.g. 50)
    if the number of features is very high. This will suppress some
    noise and speed up the computation of pairwise distances between
    samples. For more tips see Laurens van der Maaten's FAQ [2].

    Read more in the :ref:`User Guide <t_sne>`.

    Parameters
    ----------
    n_components : int, default=2
        Dimension of the embedded space.

    perplexity : float, default=30.0
        The perplexity is related to the number of nearest neighbors that
        is used in other manifold learning algorithms. Larger datasets
        usually require a larger perplexity. Consider selecting a value
        between 5 and 50. Different values can result in significantly
        different results. The perplexity must be less than the number
        of samples.

    early_exaggeration : float, default=12.0
        Controls how tight natural clusters in the original space are in
        the embedded space and how much space will be between them. For
        larger values, the space between natural clusters will be larger
        in the embedded space. Again, the choice of this parameter is not
        very critical. If the cost function increases during initial
        optimization, the early exaggeration factor or the learning rate
        might be too high.

    learning_rate : float or "auto", default="auto"
        The learning rate for t-SNE is usually in the range [10.0, 1000.0]. If
        the learning rate is too high, the data may look like a 'ball' with any
        point approximately equidistant from its nearest neighbours. If the
        learning rate is too low, most points may look compressed in a dense
        cloud with few outliers. If the cost function gets stuck in a bad local
        minimum increasing the learning rate may help.
        Note that many other t-SNE implementations (bhtsne, FIt-SNE, openTSNE,
        etc.) use a definition of learning_rate that is 4 times smaller than
        ours. So our learning_rate=200 corresponds to learning_rate=800 in
        those other implementations. The 'auto' option sets the learning_rate
        to `max(N / early_exaggeration / 4, 50)` where N is the sample size,
        following [4] and [5].

        .. versionchanged:: 1.2
           The default value changed to `"auto"`.

    max_iter : int, default=1000
        Maximum number of iterations for the optimization. Should be at
        least 250.

        .. versionchanged:: 1.5
            Parameter name changed from `n_iter` to `max_iter`.

    n_iter_without_progress : int, default=300
        Maximum number of iterations without progress before we abort the
        optimization, used after 250 initial iterations with early
        exaggeration. Note that progress is only checked every 50 iterations so
        this value is rounded to the next multiple of 50.

        .. versionadded:: 0.17
           parameter *n_iter_without_progress* to control stopping criteria.

    min_grad_norm : float, default=1e-7
        If the gradient norm is below this threshold, the optimization will
        be stopped.

    metric : str or callable, default='euclidean'
        The metric to use when calculating distance between instances in a
        feature array. If metric is a string, it must be one of the options
        allowed by scipy.spatial.distance.pdist for its metric parameter, or
        a metric listed in pairwise.PAIRWISE_DISTANCE_FUNCTIONS.
        If metric is "precomputed", X is assumed to be a distance matrix.
        Alternatively, if metric is a callable function, it is called on each
        pair of instances (rows) and the resulting value recorded. The callable
        should take two arrays from X as input and return a value indicating
        the distance between them. The default is "euclidean" which is
        interpreted as squared euclidean distance.

    metric_params : dict, default=None
        Additional keyword arguments for the metric function.

        .. versionadded:: 1.1

    init : {"random", "pca"} or ndarray of shape (n_samples, n_components), \
            default="pca"
        Initialization of embedding.
        PCA initialization cannot be used with precomputed distances and is
        usually more globally stable than random initialization.

        .. versionchanged:: 1.2
           The default value changed to `"pca"`.

    verbose : int, default=0
        Verbosity level.

    random_state : int, RandomState instance or None, default=None
        Determines the random number generator. Pass an int for reproducible
        results across multiple function calls. Note that different
        initializations might result in different local minima of the cost
        function. See :term:`Glossary <random_state>`.

    method : {'barnes_hut', 'exact'}, default='barnes_hut'
        By default the gradient calculation algorithm uses Barnes-Hut
        approximation running in O(NlogN) time. method='exact'
        will run on the slower, but exact, algorithm in O(N^2) time. The
        exact algorithm should be used when nearest-neighbor errors need
        to be better than 3%. However, the exact method cannot scale to
        millions of examples.

        .. versionadded:: 0.17
           Approximate optimization *method* via the Barnes-Hut.

    angle : float, default=0.5
        Only used if method='barnes_hut'
        This is the trade-off between speed and accuracy for Barnes-Hut T-SNE.
        'angle' is the angular size (referred to as theta in [3]) of a distant
        node as measured from a point. If this size is below 'angle' then it is
        used as a summary node of all points contained within it.
        This method is not very sensitive to changes in this parameter
        in the range of 0.2 - 0.8. Angle less than 0.2 has quickly increasing
        computation time and angle greater 0.8 has quickly increasing error.

    n_jobs : int, default=None
        The number of parallel jobs to run for neighbors search. This parameter
        has no impact when ``metric="precomputed"`` or
        (``metric="euclidean"`` and ``method="exact"``).
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

        .. versionadded:: 0.22

    Attributes
    ----------
    embedding_ : array-like of shape (n_samples, n_components)
        Stores the embedding vectors.

    kl_divergence_ : float
        Kullback-Leibler divergence after optimization.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    learning_rate_ : float
        Effective learning rate.

        .. versionadded:: 1.2

    n_iter_ : int
        Number of iterations run.

    See Also
    --------
    sklearn.decomposition.PCA : Principal component analysis that is a linear
        dimensionality reduction method.
    sklearn.decomposition.KernelPCA : Non-linear dimensionality reduction using
        kernels and PCA.
    MDS : Manifold learning using multidimensional scaling.
    Isomap : Manifold learning based on Isometric Mapping.
    LocallyLinearEmbedding : Manifold learning using Locally Linear Embedding.
    SpectralEmbedding : Spectral embedding for non-linear dimensionality.

    Notes
    -----
    For an example of using :class:`~sklearn.manifold.TSNE` in combination with
    :class:`~sklearn.neighbors.KNeighborsTransformer` see
    :ref:`sphx_glr_auto_examples_neighbors_approximate_nearest_neighbors.py`.

    References
    ----------

    [1] van der Maaten, L.J.P.; Hinton, G.E. Visualizing High-Dimensional Data
        Using t-SNE. Journal of Machine Learning Research 9:2579-2605, 2008.

    [2] van der Maaten, L.J.P. t-Distributed Stochastic Neighbor Embedding
        https://lvdmaaten.github.io/tsne/

    [3] L.J.P. van der Maaten. Accelerating t-SNE using Tree-Based Algorithms.
        Journal of Machine Learning Research 15(Oct):3221-3245, 2014.
        https://lvdmaaten.github.io/publications/papers/JMLR_2014.pdf

    [4] Belkina, A. C., Ciccolella, C. O., Anno, R., Halpert, R., Spidlen, J.,
        & Snyder-Cappione, J. E. (2019). Automated optimized parameters for
        T-distributed stochastic neighbor embedding improve visualization
        and analysis of large datasets. Nature Communications, 10(1), 1-12.

    [5] Kobak, D., & Berens, P. (2019). The art of using t-SNE for single-cell
        transcriptomics. Nature Communications, 10(1), 1-14.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.manifold import TSNE
    >>> X = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
    >>> X_embedded = TSNE(n_components=2, learning_rate='auto',
    ...                   init='random', perplexity=3).fit_transform(X)
    >>> X_embedded.shape
    (4, 2)
    """

    _parameter_constraints: dict = {
        "n_components": [Interval(Integral, 1, None, closed="left")],
        "perplexity": [Interval(Real, 0, None, closed="neither")],
        "early_exaggeration": [Interval(Real, 1, None, closed="left")],
        "learning_rate": [
            StrOptions({"auto"}),
            Interval(Real, 0, None, closed="neither"),
        ],
        "max_iter": [Interval(Integral, 250, None, closed="left")],
        "n_iter_without_progress": [Interval(Integral, -1, None, closed="left")],
        "min_grad_norm": [Interval(Real, 0, None, closed="left")],
        "metric": [StrOptions(set(_VALID_METRICS) | {"precomputed"}), callable],
        "metric_params": [dict, None],
        "init": [
            StrOptions({"pca", "random"}),
            np.ndarray,
        ],
        "verbose": ["verbose"],
        "random_state": ["random_state"],
        "method": [StrOptions({"barnes_hut", "exact"})],
        "angle": [Interval(Real, 0, 1, closed="both")],
        "n_jobs": [None, Integral],
    }

    # Control the number of exploration iterations with early_exaggeration on
    _EXPLORATION_MAX_ITER = 250

    # Control the number of iterations between progress checks
    _N_ITER_CHECK = 50

    def __init__(
        self,
        n_components=2,
        *,
        perplexity=30.0,
        early_exaggeration=12.0,
        learning_rate="auto",
        max_iter=1000,
        n_iter_without_progress=300,
        min_grad_norm=1e-7,
        metric="euclidean",
        metric_params=None,
        init="pca",
        verbose=0,
        random_state=None,
        method="barnes_hut",
        angle=0.5,
        n_jobs=None,
    ):
        # [FUNCTION DOC]
        # Stores every constructor argument as an attribute with the same name, without
        #   validation (scikit-learn estimator convention).
        self.n_components = n_components
        self.perplexity = perplexity
        self.early_exaggeration = early_exaggeration
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.n_iter_without_progress = n_iter_without_progress
        self.min_grad_norm = min_grad_norm
        self.metric = metric
        self.metric_params = metric_params
        self.init = init
        self.verbose = verbose
        self.random_state = random_state
        self.method = method
        self.angle = angle
        self.n_jobs = n_jobs

    def _check_params_vs_input(self, X):
        # [FUNCTION DOC]
        # HELPER: Ensures perplexity is strictly less than the number of rows of X.
        # Inputs: X - object with a shape attribute (called before data validation).
        # Exceptions: ValueError if perplexity >= n_samples.
        if self.perplexity >= X.shape[0]:
            raise ValueError(
                f"perplexity ({self.perplexity}) must be less "
                f"than n_samples ({X.shape[0]})"
            )

    def _fit(self, X, skip_num_points=0):
        # [FUNCTION DOC]
        # HELPER: Core fitting routine. Steps: set learning_rate_ (for "auto", n_samples
        #   / early_exaggeration / 4 floored at 50); validate X (barnes_hut accepts CSR
        #   and needs at least 2 samples; exact accepts CSR/CSC/COO; float32/float64).
        #   For metric="precomputed" reject init="pca", non-square X, negative entries
        #   and sparse X with method exact. Reject n_components > 3 for barnes_hut.
        #   Exact method: obtain the full distance matrix (squared Euclidean directly,
        #   otherwise the metric with metric_params and then squared), reject negative
        #   distances and compute dense joint probabilities. Barnes-Hut: use
        #   min(n_samples - 1, int(3 * perplexity + 1)) nearest neighbors, square the
        #   neighbor distances and compute sparse joint probabilities. Initialize the
        #   embedding from the given ndarray, from PCA rescaled so the first component
        #   has std 1e-4, or from Gaussian noise with std 1e-4; set degrees of freedom
        #   to max(n_components - 1, 1) and run _tsne.
        # Inputs: X - data or precomputed distances; skip_num_points - int (default 0).
        # Outputs: embedding array (n_samples, n_components).
        # Exceptions: ValueError / TypeError for the invalid combinations above;
        #   AssertionError if probabilities are invalid.
        # Side effects: sets learning_rate_, n_features_in_ (via validate_data) and,
        #   through _tsne, n_iter_ and kl_divergence_; prints progress when verbose;
        #   consumes the random state.
        # KNOWN ISSUE: in the exact path with metric="precomputed", distances is the
        #   validated X itself and is squared in place, so a float64/float32 numpy input
        #   passed by the user is modified. neighbors_nn is always None, so the
        #   neighbors argument of _tsne is unused.
        """Private function to fit the model using X as training data."""

        if self.learning_rate == "auto":
            # See issue #18018
            self.learning_rate_ = X.shape[0] / self.early_exaggeration / 4
            self.learning_rate_ = np.maximum(self.learning_rate_, 50)
        else:
            self.learning_rate_ = self.learning_rate

        if self.method == "barnes_hut":
            X = validate_data(
                self,
                X,
                accept_sparse=["csr"],
                ensure_min_samples=2,
                dtype=[np.float32, np.float64],
            )
        else:
            X = validate_data(
                self,
                X,
                accept_sparse=["csr", "csc", "coo"],
                dtype=[np.float32, np.float64],
            )
        if self.metric == "precomputed":
            if isinstance(self.init, str) and self.init == "pca":
                raise ValueError(
                    'The parameter init="pca" cannot be used with metric="precomputed".'
                )
            if X.shape[0] != X.shape[1]:
                raise ValueError("X should be a square distance matrix")

            check_non_negative(
                X,
                (
                    "TSNE.fit(). With metric='precomputed', X "
                    "should contain positive distances."
                ),
            )

            if self.method == "exact" and issparse(X):
                raise TypeError(
                    'TSNE with method="exact" does not accept sparse '
                    'precomputed distance matrix. Use method="barnes_hut" '
                    "or provide the dense distance matrix."
                )

        if self.method == "barnes_hut" and self.n_components > 3:
            raise ValueError(
                "'n_components' should be inferior to 4 for the "
                "barnes_hut algorithm as it relies on "
                "quad-tree or oct-tree."
            )
        random_state = check_random_state(self.random_state)

        n_samples = X.shape[0]

        neighbors_nn = None
        if self.method == "exact":
            # Retrieve the distance matrix, either using the precomputed one or
            # computing it.
            if self.metric == "precomputed":
                distances = X
            else:
                if self.verbose:
                    print("[t-SNE] Computing pairwise distances...")

                if self.metric == "euclidean":
                    # Euclidean is squared here, rather than using **= 2,
                    # because euclidean_distances already calculates
                    # squared distances, and returns np.sqrt(dist) for
                    # squared=False.
                    # Also, Euclidean is slower for n_jobs>1, so don't set here
                    distances = pairwise_distances(X, metric=self.metric, squared=True)
                else:
                    metric_params_ = self.metric_params or {}
                    distances = pairwise_distances(
                        X, metric=self.metric, n_jobs=self.n_jobs, **metric_params_
                    )

            if np.any(distances < 0):
                raise ValueError(
                    "All distances should be positive, the metric given is not correct"
                )

            if self.metric != "euclidean":
                distances **= 2

            # compute the joint probability distribution for the input space
            P = _joint_probabilities(distances, self.perplexity, self.verbose)
            assert np.all(np.isfinite(P)), "All probabilities should be finite"
            assert np.all(P >= 0), "All probabilities should be non-negative"
            assert np.all(P <= 1), (
                "All probabilities should be less or then equal to one"
            )

        else:
            # Compute the number of nearest neighbors to find.
            # LvdM uses 3 * perplexity as the number of neighbors.
            # In the event that we have very small # of points
            # set the neighbors to n - 1.
            n_neighbors = min(n_samples - 1, int(3.0 * self.perplexity + 1))

            if self.verbose:
                print("[t-SNE] Computing {} nearest neighbors...".format(n_neighbors))

            # Find the nearest neighbors for every point
            knn = NearestNeighbors(
                algorithm="auto",
                n_jobs=self.n_jobs,
                n_neighbors=n_neighbors,
                metric=self.metric,
                metric_params=self.metric_params,
            )
            t0 = time()
            knn.fit(X)
            duration = time() - t0
            if self.verbose:
                print(
                    "[t-SNE] Indexed {} samples in {:.3f}s...".format(
                        n_samples, duration
                    )
                )

            t0 = time()
            distances_nn = knn.kneighbors_graph(mode="distance")
            duration = time() - t0
            if self.verbose:
                print(
                    "[t-SNE] Computed neighbors for {} samples in {:.3f}s...".format(
                        n_samples, duration
                    )
                )

            # Free the memory used by the ball_tree
            del knn

            # knn return the euclidean distance but we need it squared
            # to be consistent with the 'exact' method. Note that the
            # the method was derived using the euclidean method as in the
            # input space. Not sure of the implication of using a different
            # metric.
            distances_nn.data **= 2

            # compute the joint probability distribution for the input space
            P = _joint_probabilities_nn(distances_nn, self.perplexity, self.verbose)

        if isinstance(self.init, np.ndarray):
            X_embedded = self.init
        elif self.init == "pca":
            pca = PCA(
                n_components=self.n_components,
                random_state=random_state,
            )
            # Always output a numpy array, no matter what is configured globally
            pca.set_output(transform="default")
            X_embedded = pca.fit_transform(X).astype(np.float32, copy=False)
            # PCA is rescaled so that PC1 has standard deviation 1e-4 which is
            # the default value for random initialization. See issue #18018.
            X_embedded = X_embedded / np.std(X_embedded[:, 0]) * 1e-4
        elif self.init == "random":
            # The embedding is initialized with iid samples from Gaussians with
            # standard deviation 1e-4.
            X_embedded = 1e-4 * random_state.standard_normal(
                size=(n_samples, self.n_components)
            ).astype(np.float32)

        # Degrees of freedom of the Student's t-distribution. The suggestion
        # degrees_of_freedom = n_components - 1 comes from
        # "Learning a Parametric Embedding by Preserving Local Structure"
        # Laurens van der Maaten, 2009.
        degrees_of_freedom = max(self.n_components - 1, 1)

        return self._tsne(
            P,
            degrees_of_freedom,
            n_samples,
            X_embedded=X_embedded,
            neighbors=neighbors_nn,
            skip_num_points=skip_num_points,
        )

    def _tsne(
        self,
        P,
        degrees_of_freedom,
        n_samples,
        X_embedded,
        neighbors=None,
        skip_num_points=0,
    ):
        # [FUNCTION DOC]
        # HELPER: Runs the two-stage optimization. Stage 1: multiply P in place by
        #   early_exaggeration and run _gradient_descent for _EXPLORATION_MAX_ITER
        #   iterations with momentum 0.5 (n_iter_without_progress also set to that
        #   value). Stage 2: divide P back, then continue from the next iteration up to
        #   max_iter with momentum 0.8 and the user's n_iter_without_progress. The
        #   objective is _kl_divergence_bh (with angle, verbose and OpenMP thread count)
        #   or _kl_divergence.
        # Inputs: P - joint probabilities (dense condensed or CSR); degrees_of_freedom;
        #   n_samples; X_embedded - initial embedding; neighbors (unused);
        #   skip_num_points.
        # Outputs: final embedding reshaped to (n_samples, n_components).
        # Side effects: temporarily scales P in place; sets n_iter_ (index of the last
        #   iteration, i.e. iterations run minus one) and kl_divergence_; prints when
        #   verbose.
        # KNOWN ISSUE: stage 1 always ends with it below _EXPLORATION_MAX_ITER, so the
        #   condition guarding stage 2 is always true and the remaining > 0 check is
        #   dead. The neighbors parameter is never used.
        """Runs t-SNE."""
        # t-SNE minimizes the Kullback-Leiber divergence of the Gaussians P
        # and the Student's t-distributions Q. The optimization algorithm that
        # we use is batch gradient descent with two stages:
        # * initial optimization with early exaggeration and momentum at 0.5
        # * final optimization with momentum at 0.8
        params = X_embedded.ravel()

        opt_args = {
            "it": 0,
            "n_iter_check": self._N_ITER_CHECK,
            "min_grad_norm": self.min_grad_norm,
            "learning_rate": self.learning_rate_,
            "verbose": self.verbose,
            "kwargs": dict(skip_num_points=skip_num_points),
            "args": [P, degrees_of_freedom, n_samples, self.n_components],
            "n_iter_without_progress": self._EXPLORATION_MAX_ITER,
            "max_iter": self._EXPLORATION_MAX_ITER,
            "momentum": 0.5,
        }
        if self.method == "barnes_hut":
            obj_func = _kl_divergence_bh
            opt_args["kwargs"]["angle"] = self.angle
            # Repeat verbose argument for _kl_divergence_bh
            opt_args["kwargs"]["verbose"] = self.verbose
            # Get the number of threads for gradient computation here to
            # avoid recomputing it at each iteration.
            opt_args["kwargs"]["num_threads"] = _openmp_effective_n_threads()
        else:
            obj_func = _kl_divergence

        # Learning schedule (part 1): do 250 iteration with lower momentum but
        # higher learning rate controlled via the early exaggeration parameter
        P *= self.early_exaggeration
        params, kl_divergence, it = _gradient_descent(obj_func, params, **opt_args)
        if self.verbose:
            print(
                "[t-SNE] KL divergence after %d iterations with early exaggeration: %f"
                % (it + 1, kl_divergence)
            )

        # Learning schedule (part 2): disable early exaggeration and finish
        # optimization with a higher momentum at 0.8
        P /= self.early_exaggeration
        remaining = self.max_iter - self._EXPLORATION_MAX_ITER
        if it < self._EXPLORATION_MAX_ITER or remaining > 0:
            opt_args["max_iter"] = self.max_iter
            opt_args["it"] = it + 1
            opt_args["momentum"] = 0.8
            opt_args["n_iter_without_progress"] = self.n_iter_without_progress
            params, kl_divergence, it = _gradient_descent(obj_func, params, **opt_args)

        # Save the final number of iterations
        self.n_iter_ = it

        if self.verbose:
            print(
                "[t-SNE] KL divergence after %d iterations: %f"
                % (it + 1, kl_divergence)
            )

        X_embedded = params.reshape(n_samples, self.n_components)
        self.kl_divergence_ = kl_divergence

        return X_embedded

    @_fit_context(
        # TSNE.metric is not validated yet
        prefer_skip_nested_validation=False
    )
    def fit_transform(self, X, y=None):
        # [FUNCTION DOC]
        # Checks perplexity against the number of samples, runs _fit, stores the result
        #   in embedding_ and returns it.
        # Inputs: X - data or precomputed distances; y - ignored.
        # Outputs: ndarray (n_samples, n_components).
        # Exceptions: ValueError from _check_params_vs_input and _fit; parameter
        #   validation errors via _fit_context.
        # Side effects: sets all fitted attributes.
        """Fit X into an embedded space and return that transformed output.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features) or \
            (n_samples, n_samples)
            If the metric is 'precomputed' X must be a square distance
            matrix. Otherwise it contains a sample per row. If the method
            is 'exact', X may be a sparse matrix of type 'csr', 'csc'
            or 'coo'. If the method is 'barnes_hut' and the metric is
            'precomputed', X may be a precomputed sparse graph.

        y : None
            Ignored.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_components)
            Embedding of the training data in low-dimensional space.
        """
        self._check_params_vs_input(X)
        embedding = self._fit(X)
        self.embedding_ = embedding
        return self.embedding_

    @_fit_context(
        # TSNE.metric is not validated yet
        prefer_skip_nested_validation=False
    )
    def fit(self, X, y=None):
        # [FUNCTION DOC]
        # Fits the model by calling fit_transform and returns self.
        # Side effects: same as fit_transform.
        """Fit X into an embedded space.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features) or \
            (n_samples, n_samples)
            If the metric is 'precomputed' X must be a square distance
            matrix. Otherwise it contains a sample per row. If the method
            is 'exact', X may be a sparse matrix of type 'csr', 'csc'
            or 'coo'. If the method is 'barnes_hut' and the metric is
            'precomputed', X may be a precomputed sparse graph.

        y : None
            Ignored.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        self.fit_transform(X)
        return self

    @property
    def _n_features_out(self):
        # [FUNCTION DOC]
        # Number of output features (columns of embedding_), used to build output
        #   feature names.
        # Exceptions: AttributeError if not fitted.
        """Number of transformed output features."""
        return self.embedding_.shape[1]

    def __sklearn_tags__(self):
        # [FUNCTION DOC]
        # Extends the base tags: input is pairwise when metric is "precomputed", and
        #   sparse input is supported.
        tags = super().__sklearn_tags__()
        tags.input_tags.pairwise = self.metric == "precomputed"
        tags.input_tags.sparse = True
        return tags
