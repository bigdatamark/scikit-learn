# [MODULE DOC]
# Affinity Propagation clustering (Frey & Dueck 2007): exemplar-based clustering by
# iterative exchange of responsibility and availability messages on a similarity
# matrix. Provides the private solver, the affinity_propagation function wrapper and
# the AffinityPropagation estimator.
"""Affinity Propagation clustering algorithm."""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import warnings
from numbers import Integral, Real

import numpy as np

from sklearn._config import config_context
from sklearn.base import BaseEstimator, ClusterMixin, _fit_context
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import euclidean_distances, pairwise_distances_argmin
from sklearn.utils import check_random_state
from sklearn.utils._param_validation import Interval, StrOptions, validate_params
from sklearn.utils.validation import check_is_fitted, validate_data


def _equal_similarities_and_preferences(S, preference):
    # [FUNCTION DOC]
    # HELPER: Tells whether the degenerate case holds where all preferences are equal and
    # all off-diagonal similarities are equal.
    # Inputs: S - square ndarray (n, n); preference - ndarray (scalar or (n,)).
    # Outputs: bool (numpy bool). Short-circuits: similarities are not checked if the
    # preferences differ.
    def all_equal_preferences():
        # [FUNCTION DOC]
        # HELPER: True if every entry of preference equals its first entry.
        return np.all(preference == preference.flat[0])

    def all_equal_similarities():
        # [FUNCTION DOC]
        # HELPER: True if all off-diagonal entries of S are equal (diagonal masked out).
        # Create mask to ignore diagonal of S
        mask = np.ones(S.shape, dtype=bool)
        np.fill_diagonal(mask, 0)

        return np.all(S[mask].flat == S[mask].flat[0])

    return all_equal_preferences() and all_equal_similarities()


def _affinity_propagation(
    S,
    *,
    preference,
    convergence_iter,
    max_iter,
    damping,
    verbose,
    return_n_iter,
    random_state,
):
    # [FUNCTION DOC]
    # HELPER: Core solver.
    # Behavior: if n_samples is 1 or the degenerate equal case holds, warn (UserWarning)
    # and return either every sample as its own exemplar (when the first preference
    # exceeds the similarity S[0, n-1]) or a single exemplar 0 with all labels 0.
    # Otherwise: write preference on the diagonal of S; add tiny random noise
    # (eps*S + 100*tiny, times standard normal draws) to break ties; then for up to
    # max_iter iterations: responsibilities R_new(i,k) = S(i,k) - max over k' != k of
    # (A + S)(i,k') (computed with the row max and second max), damped R = d*R +
    # (1-d)*R_new; availabilities from Rp = max(R, 0) with Rp's diagonal set to R's,
    # A_new(i,k) = min(0, column sum of Rp minus Rp(i,k)) off-diagonal and column sum
    # minus Rp(k,k) on the diagonal, damped likewise. Exemplars E are samples with
    # A(k,k) + R(k,k) > 0; their history over the last convergence_iter iterations is
    # kept circularly. Stop when, after at least convergence_iter iterations, every
    # sample's exemplar status was constant over the window and at least one exemplar
    # exists. Afterwards, if exemplars exist: assign each sample to the most similar
    # exemplar, refine each cluster's exemplar as the member maximizing the summed
    # similarity to the other members, reassign, and relabel to contiguous ids sorted by
    # exemplar index. If none: warn ConvergenceWarning, labels all -1, centers [].
    # Inputs: S - float ndarray (n, n), modified in place; preference - ndarray scalar or
    # (n,); convergence_iter, max_iter - ints; damping - float in [0.5, 1); verbose -
    # bool, prints convergence status; return_n_iter - bool; random_state - RandomState.
    # Outputs: (cluster_centers_indices, labels[, n_iter]); n_iter is the number of
    # iterations run (0 for the degenerate case).
    # Side effects: mutates S (diagonal and noise); emits warnings; consumes randomness.
    # KNOWN ISSUE: the condition it == max_iter inside the loop is unreachable since it
    # ranges up to max_iter - 1. When no exemplar exists, cluster_centers_indices is a
    # plain empty list rather than an ndarray.
    """Main affinity propagation algorithm."""
    n_samples = S.shape[0]
    if n_samples == 1 or _equal_similarities_and_preferences(S, preference):
        # It makes no sense to run the algorithm in this case, so return 1 or
        # n_samples clusters, depending on preferences
        warnings.warn(
            "All samples have mutually equal similarities. "
            "Returning arbitrary cluster center(s)."
        )
        if preference.flat[0] > S.flat[n_samples - 1]:
            return (
                (np.arange(n_samples), np.arange(n_samples), 0)
                if return_n_iter
                else (np.arange(n_samples), np.arange(n_samples))
            )
        else:
            return (
                (np.array([0]), np.array([0] * n_samples), 0)
                if return_n_iter
                else (np.array([0]), np.array([0] * n_samples))
            )

    # Place preference on the diagonal of S
    S.flat[:: (n_samples + 1)] = preference

    A = np.zeros((n_samples, n_samples))
    R = np.zeros((n_samples, n_samples))  # Initialize messages
    # Intermediate results
    tmp = np.zeros((n_samples, n_samples))

    # Remove degeneracies
    S += (
        np.finfo(S.dtype).eps * S + np.finfo(S.dtype).tiny * 100
    ) * random_state.standard_normal(size=(n_samples, n_samples))

    # Execute parallel affinity propagation updates
    e = np.zeros((n_samples, convergence_iter))

    ind = np.arange(n_samples)

    for it in range(max_iter):
        # tmp = A + S; compute responsibilities
        np.add(A, S, tmp)
        I = np.argmax(tmp, axis=1)
        Y = tmp[ind, I]  # np.max(A + S, axis=1)
        tmp[ind, I] = -np.inf
        Y2 = np.max(tmp, axis=1)

        # tmp = Rnew
        np.subtract(S, Y[:, None], tmp)
        tmp[ind, I] = S[ind, I] - Y2

        # Damping
        tmp *= 1 - damping
        R *= damping
        R += tmp

        # tmp = Rp; compute availabilities
        np.maximum(R, 0, out=tmp)
        tmp.flat[:: n_samples + 1] = R.flat[:: n_samples + 1]

        # tmp = -Anew
        tmp -= np.sum(tmp, axis=0)
        dA = np.diag(tmp).copy()
        tmp.clip(0, np.inf, tmp)
        tmp.flat[:: n_samples + 1] = dA

        # Damping
        tmp *= 1 - damping
        A *= damping
        A -= tmp

        # Check for convergence
        E = (np.diag(A) + np.diag(R)) > 0
        e[:, it % convergence_iter] = E
        K = np.sum(E, axis=0)

        if it >= convergence_iter:
            se = np.sum(e, axis=1)
            unconverged = np.sum((se == convergence_iter) + (se == 0)) != n_samples
            if (not unconverged and (K > 0)) or (it == max_iter):
                never_converged = False
                if verbose:
                    print("Converged after %d iterations." % it)
                break
    else:
        never_converged = True
        if verbose:
            print("Did not converge")

    I = np.flatnonzero(E)
    K = I.size  # Identify exemplars

    if K > 0:
        if never_converged:
            warnings.warn(
                (
                    "Affinity propagation did not converge, this model "
                    "may return degenerate cluster centers and labels."
                ),
                ConvergenceWarning,
            )
        c = np.argmax(S[:, I], axis=1)
        c[I] = np.arange(K)  # Identify clusters
        # Refine the final set of exemplars and clusters and return results
        for k in range(K):
            ii = np.asarray(c == k).nonzero()[0]
            j = np.argmax(np.sum(S[ii[:, np.newaxis], ii], axis=0))
            I[k] = ii[j]

        c = np.argmax(S[:, I], axis=1)
        c[I] = np.arange(K)
        labels = I[c]
        # Reduce labels to a sorted, gapless, list
        cluster_centers_indices = np.unique(labels)
        labels = np.searchsorted(cluster_centers_indices, labels)
    else:
        warnings.warn(
            (
                "Affinity propagation did not converge and this model "
                "will not have any cluster centers."
            ),
            ConvergenceWarning,
        )
        labels = np.array([-1] * n_samples)
        cluster_centers_indices = []

    if return_n_iter:
        return cluster_centers_indices, labels, it + 1
    else:
        return cluster_centers_indices, labels


###############################################################################
# Public API


@validate_params(
    {
        "S": ["array-like"],
        "return_n_iter": ["boolean"],
    },
    prefer_skip_nested_validation=False,
)
def affinity_propagation(
    S,
    *,
    preference=None,
    convergence_iter=15,
    max_iter=200,
    damping=0.5,
    copy=True,
    verbose=False,
    return_n_iter=False,
    random_state=None,
):
    # [FUNCTION DOC]
    # Function interface: builds AffinityPropagation with affinity="precomputed" and the
    # given parameters, fits it on S and returns its fitted attributes.
    # Inputs: see docstring; S must be a square similarity matrix.
    # Outputs: (cluster_centers_indices_, labels_) or with n_iter_ if return_n_iter.
    # Exceptions: parameter validation errors; ValueError if S is not square.
    # Side effects: with copy=False the caller's S is modified in place.
    """Perform Affinity Propagation Clustering of data.

    Read more in the :ref:`User Guide <affinity_propagation>`.

    Parameters
    ----------
    S : array-like of shape (n_samples, n_samples)
        Matrix of similarities between points.

    preference : array-like of shape (n_samples,) or float, default=None
        Preferences for each point - points with larger values of
        preferences are more likely to be chosen as exemplars. The number of
        exemplars, i.e. of clusters, is influenced by the input preferences
        value. If the preferences are not passed as arguments, they will be
        set to the median of the input similarities (resulting in a moderate
        number of clusters). For a smaller amount of clusters, this can be set
        to the minimum value of the similarities.

    convergence_iter : int, default=15
        Number of iterations with no change in the number
        of estimated clusters that stops the convergence.

    max_iter : int, default=200
        Maximum number of iterations.

    damping : float, default=0.5
        Damping factor between 0.5 and 1.

    copy : bool, default=True
        If copy is False, the affinity matrix is modified inplace by the
        algorithm, for memory efficiency.

    verbose : bool, default=False
        The verbosity level.

    return_n_iter : bool, default=False
        Whether or not to return the number of iterations.

    random_state : int, RandomState instance or None, default=None
        Pseudo-random number generator to control the starting state.
        Use an int for reproducible results across function calls.
        See the :term:`Glossary <random_state>`.

        .. versionadded:: 0.23
            this parameter was previously hardcoded as 0.

    Returns
    -------
    cluster_centers_indices : ndarray of shape (n_clusters,)
        Index of clusters centers.

    labels : ndarray of shape (n_samples,)
        Cluster labels for each point.

    n_iter : int
        Number of iterations run. Returned only if `return_n_iter` is
        set to True.

    Notes
    -----
    For an example usage,
    see :ref:`sphx_glr_auto_examples_cluster_plot_affinity_propagation.py`.
    You may also check out,
    :ref:`sphx_glr_auto_examples_applications_plot_stock_market.py`

    When the algorithm does not converge, it will still return an array of
    ``cluster_center_indices`` and labels if there are any exemplars/clusters,
    however they may be degenerate and should be used with caution.

    When all training samples have equal similarities and equal preferences,
    the assignment of cluster centers and labels depends on the preference.
    If the preference is smaller than the similarities, a single cluster center
    and label ``0`` for every sample will be returned. Otherwise, every
    training sample becomes its own cluster center and is assigned a unique
    label.

    References
    ----------
    Brendan J. Frey and Delbert Dueck, "Clustering by Passing Messages
    Between Data Points", Science Feb. 2007

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.cluster import affinity_propagation
    >>> from sklearn.metrics.pairwise import euclidean_distances
    >>> X = np.array([[1, 2], [1, 4], [1, 0],
    ...               [4, 2], [4, 4], [4, 0]])
    >>> S = -euclidean_distances(X, squared=True)
    >>> cluster_centers_indices, labels = affinity_propagation(S, random_state=0)
    >>> cluster_centers_indices
    array([0, 3])
    >>> labels
    array([0, 0, 0, 1, 1, 1])
    """
    estimator = AffinityPropagation(
        damping=damping,
        max_iter=max_iter,
        convergence_iter=convergence_iter,
        copy=copy,
        preference=preference,
        affinity="precomputed",
        verbose=verbose,
        random_state=random_state,
    ).fit(S)

    if return_n_iter:
        return estimator.cluster_centers_indices_, estimator.labels_, estimator.n_iter_
    return estimator.cluster_centers_indices_, estimator.labels_


class AffinityPropagation(ClusterMixin, BaseEstimator):
    # [CLASS DOC]
    # Estimator wrapping _affinity_propagation. With affinity="euclidean" the similarity
    # is the negative squared euclidean distance and cluster_centers_ are the exemplar
    # rows of X; with "precomputed" X is the similarity matrix and predict is unavailable.
    """Perform Affinity Propagation Clustering of data.

    Read more in the :ref:`User Guide <affinity_propagation>`.

    Parameters
    ----------
    damping : float, default=0.5
        Damping factor in the range `[0.5, 1.0)` is the extent to
        which the current value is maintained relative to
        incoming values (weighted 1 - damping). This in order
        to avoid numerical oscillations when updating these
        values (messages).

    max_iter : int, default=200
        Maximum number of iterations.

    convergence_iter : int, default=15
        Number of iterations with no change in the number
        of estimated clusters that stops the convergence.

    copy : bool, default=True
        Make a copy of input data.

    preference : array-like of shape (n_samples,) or float, default=None
        Preferences for each point - points with larger values of
        preferences are more likely to be chosen as exemplars. The number
        of exemplars, ie of clusters, is influenced by the input
        preferences value. If the preferences are not passed as arguments,
        they will be set to the median of the input similarities.

    affinity : {'euclidean', 'precomputed'}, default='euclidean'
        Which affinity to use. At the moment 'precomputed' and
        ``euclidean`` are supported. 'euclidean' uses the
        negative squared euclidean distance between points.

    verbose : bool, default=False
        Whether to be verbose.

    random_state : int, RandomState instance or None, default=None
        Pseudo-random number generator to control the starting state.
        Use an int for reproducible results across function calls.
        See the :term:`Glossary <random_state>`.

        .. versionadded:: 0.23
            this parameter was previously hardcoded as 0.

    Attributes
    ----------
    cluster_centers_indices_ : ndarray of shape (n_clusters,)
        Indices of cluster centers.

    cluster_centers_ : ndarray of shape (n_clusters, n_features)
        Cluster centers (if affinity != ``precomputed``).

    labels_ : ndarray of shape (n_samples,)
        Labels of each point.

    affinity_matrix_ : ndarray of shape (n_samples, n_samples)
        Stores the affinity matrix used in ``fit``.

    n_iter_ : int
        Number of iterations taken to converge.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    AgglomerativeClustering : Recursively merges the pair of
        clusters that minimally increases a given linkage distance.
    FeatureAgglomeration : Similar to AgglomerativeClustering,
        but recursively merges features instead of samples.
    KMeans : K-Means clustering.
    MiniBatchKMeans : Mini-Batch K-Means clustering.
    MeanShift : Mean shift clustering using a flat kernel.
    SpectralClustering : Apply clustering to a projection
        of the normalized Laplacian.

    Notes
    -----
    The algorithmic complexity of affinity propagation is quadratic
    in the number of points.

    When the algorithm does not converge, it will still return an array of
    ``cluster_center_indices`` and labels if there are any exemplars/clusters,
    however they may be degenerate and should be used with caution.

    When ``fit`` does not converge, ``cluster_centers_`` is still populated
    however it may be degenerate. In such a case, proceed with caution.
    If ``fit`` does not converge and fails to produce any ``cluster_centers_``
    then ``predict`` will label every sample as ``-1``.

    When all training samples have equal similarities and equal preferences,
    the assignment of cluster centers and labels depends on the preference.
    If the preference is smaller than the similarities, ``fit`` will result in
    a single cluster center and label ``0`` for every sample. Otherwise, every
    training sample becomes its own cluster center and is assigned a unique
    label.

    References
    ----------

    Brendan J. Frey and Delbert Dueck, "Clustering by Passing Messages
    Between Data Points", Science Feb. 2007

    Examples
    --------
    >>> from sklearn.cluster import AffinityPropagation
    >>> import numpy as np
    >>> X = np.array([[1, 2], [1, 4], [1, 0],
    ...               [4, 2], [4, 4], [4, 0]])
    >>> clustering = AffinityPropagation(random_state=5).fit(X)
    >>> clustering
    AffinityPropagation(random_state=5)
    >>> clustering.labels_
    array([0, 0, 0, 1, 1, 1])
    >>> clustering.predict([[0, 0], [4, 4]])
    array([0, 1])
    >>> clustering.cluster_centers_
    array([[1, 2],
           [4, 2]])

    For an example usage,
    see :ref:`sphx_glr_auto_examples_cluster_plot_affinity_propagation.py`.

    For a comparison of Affinity Propagation with other clustering algorithms, see
    :ref:`sphx_glr_auto_examples_cluster_plot_cluster_comparison.py`
    """

    _parameter_constraints: dict = {
        "damping": [Interval(Real, 0.5, 1.0, closed="left")],
        "max_iter": [Interval(Integral, 1, None, closed="left")],
        "convergence_iter": [Interval(Integral, 1, None, closed="left")],
        "copy": ["boolean"],
        "preference": [
            "array-like",
            Interval(Real, None, None, closed="neither"),
            None,
        ],
        "affinity": [StrOptions({"euclidean", "precomputed"})],
        "verbose": ["verbose"],
        "random_state": ["random_state"],
    }

    def __init__(
        self,
        *,
        damping=0.5,
        max_iter=200,
        convergence_iter=15,
        copy=True,
        preference=None,
        affinity="euclidean",
        verbose=False,
        random_state=None,
    ):
        # [FUNCTION DOC]
        # Stores the hyper-parameters unchanged (validated later in fit).
        self.damping = damping
        self.max_iter = max_iter
        self.convergence_iter = convergence_iter
        self.copy = copy
        self.verbose = verbose
        self.preference = preference
        self.affinity = affinity
        self.random_state = random_state

    def __sklearn_tags__(self):
        # [FUNCTION DOC]
        # Marks input as pairwise when affinity is "precomputed" and sparse-capable
        # otherwise.
        # Outputs: Tags object.
        tags = super().__sklearn_tags__()
        tags.input_tags.pairwise = self.affinity == "precomputed"
        tags.input_tags.sparse = self.affinity != "precomputed"
        return tags

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None):
        # [FUNCTION DOC]
        # Builds affinity_matrix_ (validated copy of X when precomputed, honoring copy;
        # otherwise negative squared euclidean distances of X), checks it is square, uses the
        # median of the affinity matrix as preference when preference is None, and runs
        # _affinity_propagation with return_n_iter=True.
        # Inputs: X - features or (n, n) similarities; y - ignored.
        # Outputs: self.
        # Sets: affinity_matrix_, cluster_centers_indices_, labels_, n_iter_, and
        # cluster_centers_ (copy of X rows) when not precomputed; n_features_in_ etc.
        # Exceptions: ValueError if the similarity matrix is not square.
        # Side effects: affinity_matrix_ is modified in place by the solver (diagonal holds
        # the preferences plus noise), so it is not the raw similarity matrix; with
        # precomputed and copy=False, the user's array is modified.
        """Fit the clustering from features, or affinity matrix.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features), or \
                array-like of shape (n_samples, n_samples)
            Training instances to cluster, or similarities / affinities between
            instances if ``affinity='precomputed'``. If a sparse feature matrix
            is provided, it will be converted into a sparse ``csr_matrix``.

        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        self
            Returns the instance itself.
        """
        if self.affinity == "precomputed":
            X = validate_data(self, X, copy=self.copy, force_writeable=True)
            self.affinity_matrix_ = X
        else:  # self.affinity == "euclidean"
            X = validate_data(self, X, accept_sparse="csr")
            self.affinity_matrix_ = -euclidean_distances(X, squared=True)

        if self.affinity_matrix_.shape[0] != self.affinity_matrix_.shape[1]:
            raise ValueError(
                "The matrix of similarities must be a square array. "
                f"Got {self.affinity_matrix_.shape} instead."
            )

        if self.preference is None:
            preference = np.median(self.affinity_matrix_)
        else:
            preference = self.preference
        preference = np.asarray(preference)

        random_state = check_random_state(self.random_state)

        (
            self.cluster_centers_indices_,
            self.labels_,
            self.n_iter_,
        ) = _affinity_propagation(
            self.affinity_matrix_,
            max_iter=self.max_iter,
            convergence_iter=self.convergence_iter,
            preference=preference,
            damping=self.damping,
            verbose=self.verbose,
            return_n_iter=True,
            random_state=random_state,
        )

        if self.affinity != "precomputed":
            self.cluster_centers_ = X[self.cluster_centers_indices_].copy()

        return self

    def predict(self, X):
        # [FUNCTION DOC]
        # Assigns each sample of X to the nearest cluster center (euclidean) with
        # pairwise_distances_argmin; if there are no centers, warns ConvergenceWarning and
        # returns -1 for all samples.
        # Inputs: X - array-like or sparse (n_samples, n_features).
        # Outputs: int ndarray (n_samples,).
        # Exceptions: NotFittedError; ValueError when fitted with affinity="precomputed".
        """Predict the closest cluster each sample in X belongs to.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            New data to predict. If a sparse matrix is provided, it will be
            converted into a sparse ``csr_matrix``.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Cluster labels.
        """
        check_is_fitted(self)
        X = validate_data(self, X, reset=False, accept_sparse="csr")
        if not hasattr(self, "cluster_centers_"):
            raise ValueError(
                "Predict method is not supported when affinity='precomputed'."
            )

        if self.cluster_centers_.shape[0] > 0:
            with config_context(assume_finite=True):
                return pairwise_distances_argmin(X, self.cluster_centers_)
        else:
            warnings.warn(
                (
                    "This model does not have any cluster centers "
                    "because affinity propagation did not converge. "
                    "Labeling every sample as '-1'."
                ),
                ConvergenceWarning,
            )
            return np.array([-1] * X.shape[0])

    def fit_predict(self, X, y=None):
        # [FUNCTION DOC]
        # Delegates to ClusterMixin.fit_predict (fit then return labels_); kept to override
        # the docstring.
        # Outputs: labels_ ndarray (n_samples,).
        """Fit clustering from features/affinity matrix; return cluster labels.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features), or \
                array-like of shape (n_samples, n_samples)
            Training instances to cluster, or similarities / affinities between
            instances if ``affinity='precomputed'``. If a sparse feature matrix
            is provided, it will be converted into a sparse ``csr_matrix``.

        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Cluster labels.
        """
        return super().fit_predict(X, y)
