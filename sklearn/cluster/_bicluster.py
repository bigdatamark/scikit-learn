# [MODULE DOC]
# Spectral biclustering estimators: SpectralCoclustering (Dhillon 2001) and
# SpectralBiclustering (Kluger 2003). Both normalize the data matrix, take singular
# vectors (randomized SVD or ARPACK), and run k-means on them to obtain row and
# column labels; rows_ and columns_ boolean indicator matrices describe biclusters.
"""Spectral biclustering algorithms."""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

from abc import ABCMeta, abstractmethod
from numbers import Integral

import numpy as np
from scipy.linalg import norm
from scipy.sparse import dia_array, issparse
from scipy.sparse.linalg import eigsh, svds

from sklearn.base import BaseEstimator, BiclusterMixin, _fit_context
from sklearn.cluster._kmeans import KMeans, MiniBatchKMeans
from sklearn.utils import check_random_state, check_scalar
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.extmath import _randomized_svd, make_nonnegative, safe_sparse_dot
from sklearn.utils.validation import assert_all_finite, validate_data

__all__ = ["SpectralBiclustering", "SpectralCoclustering"]


def _scale_normalize(X):
    # [FUNCTION DOC]
    # HELPER: Independent row/column scaling: makes X non-negative (shifting by its
    # minimum if needed), computes D_r = 1/sqrt(row sums) and D_c = 1/sqrt(column sums)
    # with NaN (from zero sums) replaced by 0, and returns D_r X D_c.
    # Inputs: X - dense ndarray or sparse (n_rows, n_cols).
    # Outputs: (normalized matrix of the same kind, row_diag (n_rows,), col_diag
    # (n_cols,)).
    # KNOWN ISSUE: a zero row sum gives 1/0 = inf, not NaN, so the NaN replacement does
    # not catch it and the result contains inf/NaN for empty rows or columns.
    """Normalize ``X`` by scaling rows and columns independently.

    Returns the normalized matrix and the row and column scaling
    factors.
    """
    X = make_nonnegative(X)
    row_diag = np.asarray(1.0 / np.sqrt(X.sum(axis=1))).squeeze()
    col_diag = np.asarray(1.0 / np.sqrt(X.sum(axis=0))).squeeze()
    row_diag = np.where(np.isnan(row_diag), 0, row_diag)
    col_diag = np.where(np.isnan(col_diag), 0, col_diag)
    if issparse(X):
        n_rows, n_cols = X.shape
        r = dia_array((row_diag, [0]), shape=(n_rows, n_rows))
        c = dia_array((col_diag, [0]), shape=(n_cols, n_cols))
        an = r @ X @ c
    else:
        an = row_diag[:, np.newaxis] * X * col_diag
    return an, row_diag, col_diag


def _bistochastic_normalize(X, max_iter=1000, tol=1e-5):
    # [FUNCTION DOC]
    # HELPER: Iterates _scale_normalize until the matrix stops changing (Frobenius norm
    # of the change below tol) or max_iter iterations, approximating a matrix whose rows
    # and columns each sum to a constant.
    # Inputs: X - dense or sparse; max_iter (default 1000); tol (default 1e-5).
    # Outputs: normalized matrix.
    # KNOWN ISSUE: in the sparse branch the distance compares X_scaled with the original
    # X instead of with X_new; on the first iteration it is 0, so the loop always stops
    # after a single scaling step for sparse input. The "dist is not None" check is
    # always true.
    """Normalize rows and columns of ``X`` simultaneously so that all
    rows sum to one constant and all columns sum to a different
    constant.
    """
    # According to paper, this can also be done more efficiently with
    # deviation reduction and balancing algorithms.
    X = make_nonnegative(X)
    X_scaled = X
    for _ in range(max_iter):
        X_new, _, _ = _scale_normalize(X_scaled)
        if issparse(X):
            dist = norm(X_scaled.data - X.data)
        else:
            dist = norm(X_scaled - X_new)
        X_scaled = X_new
        if dist is not None and dist < tol:
            break
    return X_scaled


def _log_normalize(X):
    # [FUNCTION DOC]
    # HELPER: Kluger log-interaction normalization: makes X non-negative with minimum 1,
    # takes the log, then subtracts row means and column means and adds the overall mean.
    # Inputs: X - dense ndarray. Outputs: ndarray of the same shape.
    # Exceptions: ValueError for sparse input.
    """Normalize ``X`` according to Kluger's log-interactions scheme."""
    X = make_nonnegative(X, min_value=1)
    if issparse(X):
        raise ValueError(
            "Cannot compute log of a sparse matrix,"
            " because log(x) diverges to -infinity as x"
            " goes to 0."
        )
    L = np.log(X)
    row_avg = L.mean(axis=1)[:, np.newaxis]
    col_avg = L.mean(axis=0)
    avg = L.mean()
    return L - row_avg - col_avg + avg


class BaseSpectral(BiclusterMixin, BaseEstimator, metaclass=ABCMeta):
    # [CLASS DOC]
    # Abstract base for the spectral bicluster estimators: shared hyper-parameters,
    # validation, SVD and k-means helpers. Subclasses implement _check_parameters and
    # _fit.
    """Base class for spectral biclustering."""

    _parameter_constraints: dict = {
        "svd_method": [StrOptions({"randomized", "arpack"})],
        "n_svd_vecs": [Interval(Integral, 0, None, closed="left"), None],
        "mini_batch": ["boolean"],
        "init": [StrOptions({"k-means++", "random"}), np.ndarray],
        "n_init": [Interval(Integral, 1, None, closed="left")],
        "random_state": ["random_state"],
    }

    @abstractmethod
    def __init__(
        self,
        n_clusters=3,
        svd_method="randomized",
        n_svd_vecs=None,
        mini_batch=False,
        init="k-means++",
        n_init=10,
        random_state=None,
    ):
        # [FUNCTION DOC]
        # Stores the shared hyper-parameters (abstract; called by subclasses).
        self.n_clusters = n_clusters
        self.svd_method = svd_method
        self.n_svd_vecs = n_svd_vecs
        self.mini_batch = mini_batch
        self.init = init
        self.n_init = n_init
        self.random_state = random_state

    @abstractmethod
    def _check_parameters(self, n_samples):
        # [FUNCTION DOC]
        # Abstract: validates data-dependent parameters given n_samples.
        """Validate parameters depending on the input data."""

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None):
        # [FUNCTION DOC]
        # Validates X (dense or CSR, float64), runs _check_parameters on the row count, then
        # _fit.
        # Inputs: X (n_samples, n_features); y ignored. Outputs: self.
        """Create a biclustering for X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : object
            SpectralBiclustering instance.
        """
        X = validate_data(self, X, accept_sparse="csr", dtype=np.float64)
        self._check_parameters(X.shape[0])
        self._fit(X)
        return self

    def _svd(self, array, n_components, n_discard):
        # [FUNCTION DOC]
        # HELPER: Returns the first n_components left/right singular vectors of array,
        # dropping the first n_discard. Randomized: _randomized_svd with n_svd_vecs as
        # n_oversamples. ARPACK: svds with ncv=n_svd_vecs; if NaNs appear (negative
        # eigenvalues), recomputes the vectors with eigsh on the Gram matrix using a
        # uniform(-1, 1) start vector.
        # Inputs: array - dense or sparse; n_components, n_discard - ints.
        # Outputs: (u (n_rows, n_components - n_discard), v (n_cols, n_components -
        # n_discard)).
        # Exceptions: ValueError if the vectors are still non-finite.
        # Side effects: uses random_state.
        # KNOWN ISSUE: the eigsh fallback uses its default k=6 rather than n_components, so
        # the number and ordering (ascending eigenvalues vs svds output) of vectors can differ
        # from the svds result.
        """Returns first `n_components` left and right singular
        vectors u and v, discarding the first `n_discard`.
        """
        if self.svd_method == "randomized":
            kwargs = {}
            if self.n_svd_vecs is not None:
                kwargs["n_oversamples"] = self.n_svd_vecs
            u, _, vt = _randomized_svd(
                array, n_components, random_state=self.random_state, **kwargs
            )

        elif self.svd_method == "arpack":
            u, _, vt = svds(array, k=n_components, ncv=self.n_svd_vecs)
            if np.any(np.isnan(vt)):
                # some eigenvalues of A * A.T are negative, causing
                # sqrt() to be np.nan. This causes some vectors in vt
                # to be np.nan.
                A = safe_sparse_dot(array.T, array)
                random_state = check_random_state(self.random_state)
                # initialize with [-1,1] as in ARPACK
                v0 = random_state.uniform(-1, 1, A.shape[0])
                _, v = eigsh(A, ncv=self.n_svd_vecs, v0=v0)
                vt = v.T
            if np.any(np.isnan(u)):
                A = safe_sparse_dot(array, array.T)
                random_state = check_random_state(self.random_state)
                # initialize with [-1,1] as in ARPACK
                v0 = random_state.uniform(-1, 1, A.shape[0])
                _, u = eigsh(A, ncv=self.n_svd_vecs, v0=v0)

        assert_all_finite(u)
        assert_all_finite(vt)
        u = u[:, n_discard:]
        vt = vt[n_discard:]
        return u, vt.T

    def _k_means(self, data, n_clusters):
        # [FUNCTION DOC]
        # HELPER: Fits KMeans (or MiniBatchKMeans if mini_batch) with init, n_init and
        # random_state on data.
        # Inputs: data - (n, d) array; n_clusters - int.
        # Outputs: (cluster_centers_, labels_).
        if self.mini_batch:
            model = MiniBatchKMeans(
                n_clusters,
                init=self.init,
                n_init=self.n_init,
                random_state=self.random_state,
            )
        else:
            model = KMeans(
                n_clusters,
                init=self.init,
                n_init=self.n_init,
                random_state=self.random_state,
            )
        model.fit(data)
        centroid = model.cluster_centers_
        labels = model.labels_
        return centroid, labels

    def __sklearn_tags__(self):
        # [FUNCTION DOC]
        # Declares sparse input support.
        tags = super().__sklearn_tags__()
        tags.input_tags.sparse = True
        return tags


class SpectralCoclustering(BaseSpectral):
    # [CLASS DOC]
    # Co-clustering: rows and columns are clustered jointly into n_clusters biclusters
    # from the singular vectors of the scale-normalized matrix.
    """Spectral Co-Clustering algorithm (Dhillon, 2001) [1]_.

    Clusters rows and columns of an array `X` to solve the relaxed
    normalized cut of the bipartite graph created from `X` as follows:
    the edge between row vertex `i` and column vertex `j` has weight
    `X[i, j]`.

    The resulting bicluster structure is block-diagonal, since each
    row and each column belongs to exactly one bicluster.

    Supports sparse matrices, as long as they are nonnegative.

    Read more in the :ref:`User Guide <spectral_coclustering>`.

    Parameters
    ----------
    n_clusters : int, default=3
        The number of biclusters to find.

    svd_method : {'randomized', 'arpack'}, default='randomized'
        Selects the algorithm for finding singular vectors. May be
        'randomized' or 'arpack'. If 'randomized', use
        :func:`sklearn.utils.extmath.randomized_svd`, which may be faster
        for large matrices. If 'arpack', use
        :func:`scipy.sparse.linalg.svds`, which is more accurate, but
        possibly slower in some cases.

    n_svd_vecs : int, default=None
        Number of vectors to use in calculating the SVD. Corresponds
        to `ncv` when `svd_method=arpack` and `n_oversamples` when
        `svd_method` is 'randomized`.

    mini_batch : bool, default=False
        Whether to use mini-batch k-means, which is faster but may get
        different results.

    init : {'k-means++', 'random'}, or ndarray of shape \
            (n_clusters, n_features), default='k-means++'
        Method for initialization of k-means algorithm; defaults to
        'k-means++'.

    n_init : int, default=10
        Number of random initializations that are tried with the
        k-means algorithm.

        If mini-batch k-means is used, the best initialization is
        chosen and the algorithm runs once. Otherwise, the algorithm
        is run for each initialization and the best solution chosen.

    random_state : int, RandomState instance, default=None
        Used for randomizing the singular value decomposition and the k-means
        initialization. Use an int to make the randomness deterministic.
        See :term:`Glossary <random_state>`.

    Attributes
    ----------
    rows_ : array-like of shape (n_row_clusters, n_rows)
        Results of the clustering. `rows[i, r]` is True if
        cluster `i` contains row `r`. Available only after calling ``fit``.

    columns_ : array-like of shape (n_column_clusters, n_columns)
        Results of the clustering, like `rows`.

    row_labels_ : array-like of shape (n_rows,)
        The bicluster label of each row.

    column_labels_ : array-like of shape (n_cols,)
        The bicluster label of each column.

    biclusters_ : tuple of two ndarrays
        The tuple contains the `rows_` and `columns_` arrays.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    SpectralBiclustering : Partitions rows and columns under the assumption
        that the data has an underlying checkerboard structure.

    References
    ----------
    .. [1] :doi:`Dhillon, Inderjit S, 2001. Co-clustering documents and words using
           bipartite spectral graph partitioning.
           <10.1145/502512.502550>`

    Examples
    --------
    >>> from sklearn.cluster import SpectralCoclustering
    >>> import numpy as np
    >>> X = np.array([[1, 1], [2, 1], [1, 0],
    ...               [4, 7], [3, 5], [3, 6]])
    >>> clustering = SpectralCoclustering(n_clusters=2, random_state=0).fit(X)
    >>> clustering.row_labels_ #doctest: +SKIP
    array([0, 1, 1, 0, 0, 0], dtype=int32)
    >>> clustering.column_labels_ #doctest: +SKIP
    array([0, 0], dtype=int32)
    >>> clustering
    SpectralCoclustering(n_clusters=2, random_state=0)

    For a more detailed example, see the following:
    :ref:`sphx_glr_auto_examples_bicluster_plot_spectral_coclustering.py`.
    """

    _parameter_constraints: dict = {
        **BaseSpectral._parameter_constraints,
        "n_clusters": [Interval(Integral, 1, None, closed="left")],
    }

    def __init__(
        self,
        n_clusters=3,
        *,
        svd_method="randomized",
        n_svd_vecs=None,
        mini_batch=False,
        init="k-means++",
        n_init=10,
        random_state=None,
    ):
        # [FUNCTION DOC]
        # Forwards hyper-parameters to BaseSpectral.
        super().__init__(
            n_clusters, svd_method, n_svd_vecs, mini_batch, init, n_init, random_state
        )

    def _check_parameters(self, n_samples):
        # [FUNCTION DOC]
        # Raises ValueError if n_clusters exceeds n_samples.
        # KNOWN ISSUE: n_clusters is not checked against the number of columns.
        if self.n_clusters > n_samples:
            raise ValueError(
                f"n_clusters should be <= n_samples={n_samples}. Got"
                f" {self.n_clusters} instead."
            )

    def _fit(self, X):
        # [FUNCTION DOC]
        # Scale-normalizes X, takes 1 + ceil(log2(n_clusters)) singular vectors discarding
        # the first, rescales them by row_diag/col_diag, stacks row and column embeddings,
        # and k-means them jointly.
        # Sets: row_labels_, column_labels_, rows_ and columns_ (n_clusters x n_rows/n_cols
        # boolean indicators).
        normalized_data, row_diag, col_diag = _scale_normalize(X)
        n_sv = 1 + int(np.ceil(np.log2(self.n_clusters)))
        u, v = self._svd(normalized_data, n_sv, n_discard=1)
        z = np.vstack((row_diag[:, np.newaxis] * u, col_diag[:, np.newaxis] * v))

        _, labels = self._k_means(z, self.n_clusters)

        n_rows = X.shape[0]
        self.row_labels_ = labels[:n_rows]
        self.column_labels_ = labels[n_rows:]

        self.rows_ = np.vstack([self.row_labels_ == c for c in range(self.n_clusters)])
        self.columns_ = np.vstack(
            [self.column_labels_ == c for c in range(self.n_clusters)]
        )


class SpectralBiclustering(BaseSpectral):
    # [CLASS DOC]
    # Checkerboard biclustering: rows and columns get separate cluster counts; the best
    # piecewise-constant singular vectors are used to project and cluster the data.
    """Spectral biclustering (Kluger, 2003) [1]_.

    Partitions rows and columns under the assumption that the data has
    an underlying checkerboard structure. For instance, if there are
    two row partitions and three column partitions, each row will
    belong to three biclusters, and each column will belong to two
    biclusters. The outer product of the corresponding row and column
    label vectors gives this checkerboard structure.

    Read more in the :ref:`User Guide <spectral_biclustering>`.

    Parameters
    ----------
    n_clusters : int or tuple (n_row_clusters, n_column_clusters), default=3
        The number of row and column clusters in the checkerboard
        structure.

    method : {'bistochastic', 'scale', 'log'}, default='bistochastic'
        Method of normalizing and converting singular vectors into
        biclusters. May be one of 'scale', 'bistochastic', or 'log'.
        The authors recommend using 'log'. If the data is sparse,
        however, log normalization will not work, which is why the
        default is 'bistochastic'.

        .. warning::
           if `method='log'`, the data must not be sparse.

    n_components : int, default=6
        Number of singular vectors to check.

    n_best : int, default=3
        Number of best singular vectors to which to project the data
        for clustering.

    svd_method : {'randomized', 'arpack'}, default='randomized'
        Selects the algorithm for finding singular vectors. May be
        'randomized' or 'arpack'. If 'randomized', uses
        :func:`~sklearn.utils.extmath.randomized_svd`, which may be faster
        for large matrices. If 'arpack', uses
        `scipy.sparse.linalg.svds`, which is more accurate, but
        possibly slower in some cases.

    n_svd_vecs : int, default=None
        Number of vectors to use in calculating the SVD. Corresponds
        to `ncv` when `svd_method=arpack` and `n_oversamples` when
        `svd_method` is 'randomized`.

    mini_batch : bool, default=False
        Whether to use mini-batch k-means, which is faster but may get
        different results.

    init : {'k-means++', 'random'} or ndarray of shape (n_clusters, n_features), \
            default='k-means++'
        Method for initialization of k-means algorithm; defaults to
        'k-means++'.

    n_init : int, default=10
        Number of random initializations that are tried with the
        k-means algorithm.

        If mini-batch k-means is used, the best initialization is
        chosen and the algorithm runs once. Otherwise, the algorithm
        is run for each initialization and the best solution chosen.

    random_state : int, RandomState instance, default=None
        Used for randomizing the singular value decomposition and the k-means
        initialization. Use an int to make the randomness deterministic.
        See :term:`Glossary <random_state>`.

    Attributes
    ----------
    rows_ : array-like of shape (n_row_clusters, n_rows)
        Results of the clustering. `rows[i, r]` is True if
        cluster `i` contains row `r`. Available only after calling ``fit``.

    columns_ : array-like of shape (n_column_clusters, n_columns)
        Results of the clustering, like `rows`.

    row_labels_ : array-like of shape (n_rows,)
        Row partition labels.

    column_labels_ : array-like of shape (n_cols,)
        Column partition labels.

    biclusters_ : tuple of two ndarrays
        The tuple contains the `rows_` and `columns_` arrays.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    SpectralCoclustering : Clusters rows and columns of an array `X` to solve the
        relaxed normalized cut of the bipartite graph created from `X`.

    References
    ----------

    .. [1] :doi:`Kluger, Yuval, et. al., 2003. Spectral biclustering of microarray
           data: coclustering genes and conditions.
           <10.1101/gr.648603>`

    Examples
    --------
    >>> from sklearn.cluster import SpectralBiclustering
    >>> import numpy as np
    >>> X = np.array([[1, 1], [2, 1], [1, 0],
    ...               [4, 7], [3, 5], [3, 6]])
    >>> clustering = SpectralBiclustering(n_clusters=2, random_state=0).fit(X)
    >>> clustering.row_labels_
    array([1, 1, 1, 0, 0, 0], dtype=int32)
    >>> clustering.column_labels_
    array([1, 0], dtype=int32)
    >>> clustering
    SpectralBiclustering(n_clusters=2, random_state=0)

    For a more detailed example, see
    :ref:`sphx_glr_auto_examples_bicluster_plot_spectral_biclustering.py`
    """

    _parameter_constraints: dict = {
        **BaseSpectral._parameter_constraints,
        "n_clusters": [Interval(Integral, 1, None, closed="left"), tuple],
        "method": [StrOptions({"bistochastic", "scale", "log"})],
        "n_components": [Interval(Integral, 1, None, closed="left")],
        "n_best": [Interval(Integral, 1, None, closed="left")],
    }

    def __init__(
        self,
        n_clusters=3,
        *,
        method="bistochastic",
        n_components=6,
        n_best=3,
        svd_method="randomized",
        n_svd_vecs=None,
        mini_batch=False,
        init="k-means++",
        n_init=10,
        random_state=None,
    ):
        # [FUNCTION DOC]
        # Forwards shared hyper-parameters and stores method, n_components, n_best.
        super().__init__(
            n_clusters, svd_method, n_svd_vecs, mini_batch, init, n_init, random_state
        )
        self.method = method
        self.n_components = n_components
        self.n_best = n_best

    def _check_parameters(self, n_samples):
        # [FUNCTION DOC]
        # Validates n_clusters (an int <= n_samples, or a pair of ints in [1, n_samples])
        # and that n_best <= n_components.
        # Exceptions: ValueError otherwise (TypeError from the pair checks is converted).
        # KNOWN ISSUE: n_column_clusters is bounded by n_samples rather than by the number of
        # features.
        if isinstance(self.n_clusters, Integral):
            if self.n_clusters > n_samples:
                raise ValueError(
                    f"n_clusters should be <= n_samples={n_samples}. Got"
                    f" {self.n_clusters} instead."
                )
        else:  # tuple
            try:
                n_row_clusters, n_column_clusters = self.n_clusters
                check_scalar(
                    n_row_clusters,
                    "n_row_clusters",
                    target_type=Integral,
                    min_val=1,
                    max_val=n_samples,
                )
                check_scalar(
                    n_column_clusters,
                    "n_column_clusters",
                    target_type=Integral,
                    min_val=1,
                    max_val=n_samples,
                )
            except (ValueError, TypeError) as e:
                raise ValueError(
                    "Incorrect parameter n_clusters has value:"
                    f" {self.n_clusters}. It should either be a single integer"
                    " or an iterable with two integers:"
                    " (n_row_clusters, n_column_clusters)"
                    " And the values are should be in the"
                    " range: (1, n_samples)"
                ) from e

        if self.n_best > self.n_components:
            raise ValueError(
                f"n_best={self.n_best} must be <= n_components={self.n_components}."
            )

    def _fit(self, X):
        # [FUNCTION DOC]
        # Normalizes X with the chosen method (bistochastic, scale or log), computes
        # n_components singular vectors (one extra, discarded, except for log), selects for
        # rows and columns the n_best vectors best approximated by piecewise-constant
        # vectors, and clusters rows by projecting X on the best right vectors and columns by
        # projecting X transposed on the best left vectors.
        # Sets: row_labels_, column_labels_, rows_ and columns_ indicators of shape
        # (n_row_clusters * n_col_clusters, n), enumerating every (row, column) cluster pair.
        n_sv = self.n_components
        if self.method == "bistochastic":
            normalized_data = _bistochastic_normalize(X)
            n_sv += 1
        elif self.method == "scale":
            normalized_data, _, _ = _scale_normalize(X)
            n_sv += 1
        elif self.method == "log":
            normalized_data = _log_normalize(X)
        n_discard = 0 if self.method == "log" else 1
        u, v = self._svd(normalized_data, n_sv, n_discard)
        ut = u.T
        vt = v.T

        try:
            n_row_clusters, n_col_clusters = self.n_clusters
        except TypeError:
            n_row_clusters = n_col_clusters = self.n_clusters

        best_ut = self._fit_best_piecewise(ut, self.n_best, n_row_clusters)

        best_vt = self._fit_best_piecewise(vt, self.n_best, n_col_clusters)

        self.row_labels_ = self._project_and_cluster(X, best_vt.T, n_row_clusters)

        self.column_labels_ = self._project_and_cluster(X.T, best_ut.T, n_col_clusters)

        self.rows_ = np.vstack(
            [
                self.row_labels_ == label
                for label in range(n_row_clusters)
                for _ in range(n_col_clusters)
            ]
        )
        self.columns_ = np.vstack(
            [
                self.column_labels_ == label
                for _ in range(n_row_clusters)
                for label in range(n_col_clusters)
            ]
        )

    def _fit_best_piecewise(self, vectors, n_best, n_clusters):
        # [FUNCTION DOC]
        # HELPER: For each vector (row of vectors), runs 1-D k-means with n_clusters and
        # replaces each entry by its centroid; returns the n_best vectors with the smallest
        # euclidean distance to their piecewise approximation.
        # Inputs: vectors - (n_vectors, length); n_best, n_clusters - ints.
        # Outputs: array (n_best, length).
        """Find the ``n_best`` vectors that are best approximated by piecewise
        constant vectors.

        The piecewise vectors are found by k-means; the best is chosen
        according to Euclidean distance.

        """

        def make_piecewise(v):
            # [FUNCTION DOC]
            # HELPER: k-means one vector as a column and return the centroid of each entry.
            centroid, labels = self._k_means(v.reshape(-1, 1), n_clusters)
            return centroid[labels].ravel()

        piecewise_vectors = np.apply_along_axis(make_piecewise, axis=1, arr=vectors)
        dists = np.apply_along_axis(norm, axis=1, arr=(vectors - piecewise_vectors))
        result = vectors[np.argsort(dists)[:n_best]]
        return result

    def _project_and_cluster(self, data, vectors, n_clusters):
        # [FUNCTION DOC]
        # HELPER: Projects data onto vectors (dot product) and returns k-means labels.
        # Inputs: data - (n, m) dense/sparse; vectors - (m, k); n_clusters.
        # Outputs: labels (n,).
        """Project ``data`` to ``vectors`` and cluster the result."""
        projected = safe_sparse_dot(data, vectors)
        _, labels = self._k_means(projected, n_clusters)
        return labels
