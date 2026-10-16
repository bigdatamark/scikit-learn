# [MODULE DOC]
# Locally Linear Embedding: reconstruction-weight helpers, the null-space eigen
# solver, the four LLE variants (standard, hessian, modified, ltsa) behind the
# public locally_linear_embedding function, and the LocallyLinearEmbedding
# transformer.
"""Locally Linear Embedding"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

from numbers import Integral, Real

import numpy as np
from scipy.linalg import eigh, qr, solve, svd
from scipy.sparse import csr_array, lil_array
from scipy.sparse.linalg import eigsh

from sklearn.base import (
    BaseEstimator,
    ClassNamePrefixFeaturesOutMixin,
    TransformerMixin,
    _fit_context,
    _UnstableArchMixin,
)
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import check_array, check_random_state
from sklearn.utils._arpack import _init_arpack_v0
from sklearn.utils._param_validation import Interval, StrOptions, validate_params
from sklearn.utils._sparse import _align_api_if_sparse
from sklearn.utils.fixes import SCIPY_VERSION_BELOW_1_15, _sparse_eye_array
from sklearn.utils.validation import FLOAT_DTYPES, check_is_fitted, validate_data


def barycenter_weights(X, Y, indices, reg=1e-3):
    # [FUNCTION DOC]
    # For each row i of X, solve for weights over its neighbors Y[indices[i]] that
    # best reconstruct X[i] and sum to 1. Per sample: center neighbors on X[i] (C),
    # form the Gram matrix G = C times C transposed, add reg times trace(G) to the
    # diagonal (or reg alone when the trace is not positive), solve G w = ones with a
    # positive-definite solver and normalize w to sum 1.
    # Inputs: X (n_samples, n_dim); Y (n_points, n_dim); indices int
    # (n_samples, n_neighbors) into Y; reg - float regularization, default 1e-3.
    # Outputs: B ndarray (n_samples, n_neighbors) of X's float dtype.
    # Exceptions: AssertionError if X and indices row counts differ; LinAlgError from
    # solve if G is singular.
    # Side effects: none.
    # KNOWN ISSUE: docstring gives indices shape as (n_samples, n_dim); it is
    # (n_samples, n_neighbors).
    """Compute barycenter weights of X from Y along the first axis

    We estimate the weights to assign to each point in Y[indices] to recover
    the point X[i]. The barycenter weights sum to 1.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_dim)

    Y : array-like, shape (n_samples, n_dim)

    indices : array-like, shape (n_samples, n_dim)
            Indices of the points in Y used to compute the barycenter

    reg : float, default=1e-3
        Amount of regularization to add for the problem to be
        well-posed in the case of n_neighbors > n_dim

    Returns
    -------
    B : array-like, shape (n_samples, n_neighbors)

    Notes
    -----
    See developers note for more information.
    """
    X = check_array(X, dtype=FLOAT_DTYPES)
    Y = check_array(Y, dtype=FLOAT_DTYPES)
    indices = check_array(indices, dtype=int)

    n_samples, n_neighbors = indices.shape
    assert X.shape[0] == n_samples

    B = np.empty((n_samples, n_neighbors), dtype=X.dtype)
    v = np.ones(n_neighbors, dtype=X.dtype)

    # this might raise a LinalgError if G is singular and has trace
    # zero
    for i, ind in enumerate(indices):
        A = Y[ind]
        C = A - X[i]  # broadcasting
        G = np.dot(C, C.T)
        trace = np.trace(G)
        if trace > 0:
            R = reg * trace
        else:
            R = reg
        G.flat[:: n_neighbors + 1] += R
        w = solve(G, v, assume_a="pos")
        B[i, :] = w / np.sum(w)
    return B


def barycenter_kneighbors_graph(X, n_neighbors, reg=1e-3, n_jobs=None):
    # [FUNCTION DOC]
    # Builds the sparse barycenter weight graph: fit NearestNeighbors with
    # n_neighbors + 1 neighbors, query the training points against themselves and drop
    # the first column (self), compute barycenter_weights of each point from its
    # neighbors, and pack them into a CSR array with exactly n_neighbors entries per
    # row.
    # Inputs: X - array or fitted-able NearestNeighbors; n_neighbors int; reg float;
    # n_jobs for the neighbor search.
    # Outputs: sparse CSR (n_samples, n_samples), API aligned via _align_api_if_sparse.
    # Side effects: fits a new NearestNeighbors (refits if X is a NearestNeighbors).
    # KNOWN ISSUE: docstring says reg is "only relevant if mode='barycenter'" and "if
    # None use the default"; there is no mode parameter and None is not handled.
    # Dropping the first neighbor assumes it is the point itself (not true with
    # duplicate points).
    """Computes the barycenter weighted graph of k-Neighbors for points in X

    Parameters
    ----------
    X : {array-like, NearestNeighbors}
        Sample data, shape = (n_samples, n_features), in the form of a
        numpy array or a NearestNeighbors object.

    n_neighbors : int
        Number of neighbors for each sample.

    reg : float, default=1e-3
        Amount of regularization when solving the least-squares
        problem. Only relevant if mode='barycenter'. If None, use the
        default.

    n_jobs : int or None, default=None
        The number of parallel jobs to run for neighbors search.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    Returns
    -------
    A : sparse matrix in CSR format, shape = [n_samples, n_samples]
        A[i, j] is assigned the weight of edge that connects i to j.

    See Also
    --------
    sklearn.neighbors.kneighbors_graph
    sklearn.neighbors.radius_neighbors_graph
    """
    knn = NearestNeighbors(n_neighbors=n_neighbors + 1, n_jobs=n_jobs).fit(X)
    X = knn._fit_X
    n_samples = knn.n_samples_fit_
    ind = knn.kneighbors(X, return_distance=False)[:, 1:]
    data = barycenter_weights(X, X, ind, reg=reg)
    indptr = np.arange(0, n_samples * n_neighbors + 1, n_neighbors)
    csr = csr_array((data.ravel(), ind.ravel(), indptr), shape=(n_samples, n_samples))
    return _align_api_if_sparse(csr)


def null_space(
    M, k, k_skip=1, eigen_solver="arpack", tol=1e-6, max_iter=100, random_state=None
):
    # [FUNCTION DOC]
    # Finds the eigenvectors of M for the smallest eigenvalues, skipping the first
    # k_skip. "auto" picks arpack when M has more than 200 rows and k + k_skip < 10,
    # else dense. arpack: shift-invert eigsh around sigma 0 for k + k_skip pairs with
    # a random initial vector from random_state. dense: densify sparse M and call eigh
    # for index range [k_skip, k + k_skip - 1], then reorder by absolute eigenvalue.
    # Inputs: M symmetric PSD (dense, sparse or LinearOperator); k int; k_skip int
    # default 1; eigen_solver; tol and max_iter (arpack only); random_state.
    # Outputs: tuple (eigenvectors ndarray (n, k), float sum of kept eigenvalues).
    # Exceptions: ValueError wrapping ARPACK RuntimeError; ValueError for an unknown
    # eigen_solver.
    # Side effects: dense path overwrites a dense input M in place (overwrite_a).
    """
    Find the null space of a matrix M.

    Parameters
    ----------
    M : {array, matrix, sparse matrix, LinearOperator}
        Input covariance matrix: should be symmetric positive semi-definite

    k : int
        Number of eigenvalues/vectors to return

    k_skip : int, default=1
        Number of low eigenvalues to skip.

    eigen_solver : {'auto', 'arpack', 'dense'}, default='arpack'
        auto : algorithm will attempt to choose the best method for input data
        arpack : use arnoldi iteration in shift-invert mode.
                    For this method, M may be a dense matrix, sparse matrix,
                    or general linear operator.
                    Warning: ARPACK can be unstable for some problems.  It is
                    best to try several random seeds in order to check results.
        dense  : use standard dense matrix operations for the eigenvalue
                    decomposition.  For this method, M must be an array
                    or matrix type.  This method should be avoided for
                    large problems.

    tol : float, default=1e-6
        Tolerance for 'arpack' method.
        Not used if eigen_solver=='dense'.

    max_iter : int, default=100
        Maximum number of iterations for 'arpack' method.
        Not used if eigen_solver=='dense'

    random_state : int, RandomState instance, default=None
        Determines the random number generator when ``solver`` == 'arpack'.
        Pass an int for reproducible results across multiple function calls.
        See :term:`Glossary <random_state>`.
    """
    if eigen_solver == "auto":
        if M.shape[0] > 200 and k + k_skip < 10:
            eigen_solver = "arpack"
        else:
            eigen_solver = "dense"

    if eigen_solver == "arpack":
        v0 = _init_arpack_v0(M.shape[0], random_state)
        try:
            eigen_values, eigen_vectors = eigsh(
                M, k + k_skip, sigma=0.0, tol=tol, maxiter=max_iter, v0=v0
            )
        except RuntimeError as e:
            raise ValueError(
                "Error in determining null-space with ARPACK. Error message: "
                "'%s'. Note that eigen_solver='arpack' can fail when the "
                "weight matrix is singular or otherwise ill-behaved. In that "
                "case, eigen_solver='dense' is recommended. See online "
                "documentation for more information." % e
            ) from e

        return eigen_vectors[:, k_skip:], np.sum(eigen_values[k_skip:])
    elif eigen_solver == "dense":
        if hasattr(M, "toarray"):
            M = M.toarray()
        eigen_values, eigen_vectors = eigh(
            M, subset_by_index=(k_skip, k + k_skip - 1), overwrite_a=True
        )
        index = np.argsort(np.abs(eigen_values))
        return eigen_vectors[:, index], np.sum(eigen_values)
    else:
        raise ValueError("Unrecognized eigen_solver '%s'" % eigen_solver)


def _locally_linear_embedding(
    X,
    *,
    n_neighbors,
    n_components,
    reg=1e-3,
    eigen_solver="auto",
    tol=1e-6,
    max_iter=100,
    method="standard",
    hessian_tol=1e-4,
    modified_tol=1e-12,
    random_state=None,
    n_jobs=None,
):
    # [FUNCTION DOC]
    # HELPER: Unvalidated implementation of locally_linear_embedding.
    # Steps: fit NearestNeighbors(n_neighbors + 1) on X (array or estimator) and use
    # its stored data; check n_components <= n_features and n_neighbors < n_samples.
    # Build the N x N alignment matrix M (lil sparse unless eigen_solver is dense):
    # standard - M = (I - W)^T (I - W) with W from barycenter_kneighbors_graph.
    # hessian - require n_neighbors > n_components + n_components(n_components+1)/2;
    # per point, center its neighbors, get the local principal directions (SVD if
    # n_neighbors > n_features, else eigh of the Gram matrix), build Yi = [ones,
    # top directions, all pairwise products of directions], QR it, take the trailing
    # columns as Hessian estimator w, normalize columns by their sums (sums below
    # hessian_tol replaced by 1) and add w w^T to the neighbor block of M.
    # modified - require n_neighbors >= n_components; per point get eigen
    # decomposition of the local Gram matrix, compute regularized weights (reg = 1e-3
    # times eigenvalue sum), eta = median ratio of small to large eigenvalue sums,
    # per point size s_i of the almost-null space, then build Householder-corrected
    # multiple weight vectors Wi and accumulate Wi Wi^T, minus row sums on the
    # point/neighbor cross entries, plus s_i on the diagonal.
    # ltsa - per point, centered neighbors give n_components top directions; Gi =
    # [constant 1/sqrt(k), directions]; subtract Gi Gi^T from neighbor block and add
    # 1 to neighbor diagonal entries.
    # Convert sparse M to CSR and return null_space(M, n_components, k_skip=1, ...).
    # Inputs: as in locally_linear_embedding.
    # Outputs: (embedding ndarray (N, n_components), float error).
    # Exceptions: ValueError for the dimension/neighbor checks above and from
    # null_space.
    # Side effects: none on inputs (neighbor subsets are copies). Overrides the reg
    # argument in the modified branch.
    # KNOWN ISSUE: the "reg" parameter is ignored for method="modified" (overwritten by
    # a fixed 1e-3 scaling), and hessian/modified/ltsa build their own kNN query
    # instead of reusing W. An unrecognized method leaves M undefined (NameError),
    # though validation upstream prevents this.
    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1, n_jobs=n_jobs)
    nbrs.fit(X)
    X = nbrs._fit_X

    N, d_in = X.shape

    if n_components > d_in:
        raise ValueError(
            "output dimension must be less than or equal to input dimension"
        )
    if n_neighbors >= N:
        raise ValueError(
            "Expected n_neighbors < n_samples, but n_samples = %d, n_neighbors = %d"
            % (N, n_neighbors)
        )

    M_sparse = eigen_solver != "dense"
    M_container_constructor = lil_array if M_sparse else np.zeros

    if method == "standard":
        W = barycenter_kneighbors_graph(
            nbrs, n_neighbors=n_neighbors, reg=reg, n_jobs=n_jobs
        )

        # we'll compute M = (I-W)'(I-W)
        # depending on the solver, we'll do this differently
        if M_sparse:
            M = _sparse_eye_array(*W.shape, format=W.format, dtype=W.dtype) - W
            M = M.T @ M  # M = (I - W)' (I - W) = W' W - W' - W + I
        else:
            M = (W.T @ W - W.T - W).toarray()
            M.flat[:: M.shape[0] + 1] += 1  # M = W' W - W' - W + I

    elif method == "hessian":
        dp = n_components * (n_components + 1) // 2

        if n_neighbors <= n_components + dp:
            raise ValueError(
                "for method='hessian', n_neighbors must be "
                "greater than "
                "[n_components * (n_components + 3) / 2]"
            )

        neighbors = nbrs.kneighbors(
            X, n_neighbors=n_neighbors + 1, return_distance=False
        )
        neighbors = neighbors[:, 1:]

        Yi = np.empty((n_neighbors, 1 + n_components + dp), dtype=np.float64)
        Yi[:, 0] = 1

        M = M_container_constructor((N, N), dtype=np.float64)

        use_svd = n_neighbors > d_in

        for i in range(N):
            Gi = X[neighbors[i]]
            Gi -= Gi.mean(0)

            # build Hessian estimator
            if use_svd:
                U = svd(Gi, full_matrices=0)[0]
            else:
                Ci = np.dot(Gi, Gi.T)
                U = eigh(Ci)[1][:, ::-1]

            Yi[:, 1 : 1 + n_components] = U[:, :n_components]

            j = 1 + n_components
            for k in range(n_components):
                Yi[:, j : j + n_components - k] = U[:, k : k + 1] * U[:, k:n_components]
                j += n_components - k

            Q, R = qr(Yi)

            w = Q[:, n_components + 1 :]
            S = w.sum(0)

            S[np.where(abs(S) < hessian_tol)] = 1
            w /= S

            nbrs_x, nbrs_y = np.meshgrid(neighbors[i], neighbors[i])
            M[nbrs_x, nbrs_y] += np.dot(w, w.T)

    elif method == "modified":
        if n_neighbors < n_components:
            raise ValueError("modified LLE requires n_neighbors >= n_components")

        neighbors = nbrs.kneighbors(
            X, n_neighbors=n_neighbors + 1, return_distance=False
        )
        neighbors = neighbors[:, 1:]

        # find the eigenvectors and eigenvalues of each local covariance
        # matrix. We want V[i] to be a [n_neighbors x n_neighbors] matrix,
        # where the columns are eigenvectors
        V = np.zeros((N, n_neighbors, n_neighbors))
        nev = min(d_in, n_neighbors)
        evals = np.zeros([N, nev])

        # choose the most efficient way to find the eigenvectors
        use_svd = n_neighbors > d_in

        if use_svd:
            for i in range(N):
                X_nbrs = X[neighbors[i]] - X[i]
                V[i], evals[i], _ = svd(X_nbrs, full_matrices=True)
            evals **= 2
        else:
            for i in range(N):
                X_nbrs = X[neighbors[i]] - X[i]
                C_nbrs = np.dot(X_nbrs, X_nbrs.T)
                evi, vi = eigh(C_nbrs)
                evals[i] = evi[::-1]
                V[i] = vi[:, ::-1]

        # find regularized weights: this is like normal LLE.
        # because we've already computed the SVD of each covariance matrix,
        # it's faster to use this rather than np.linalg.solve
        reg = 1e-3 * evals.sum(1)

        tmp = np.dot(V.transpose(0, 2, 1), np.ones(n_neighbors))
        tmp[:, :nev] /= evals + reg[:, None]
        tmp[:, nev:] /= reg[:, None]

        w_reg = np.zeros((N, n_neighbors))
        for i in range(N):
            w_reg[i] = np.dot(V[i], tmp[i])
        w_reg /= w_reg.sum(1)[:, None]

        # calculate eta: the median of the ratio of small to large eigenvalues
        # across the points.  This is used to determine s_i, below
        rho = evals[:, n_components:].sum(1) / evals[:, :n_components].sum(1)
        eta = np.median(rho)

        # find s_i, the size of the "almost null space" for each point:
        # this is the size of the largest set of eigenvalues
        # such that Sum[v; v in set]/Sum[v; v not in set] < eta
        s_range = np.zeros(N, dtype=int)
        evals_cumsum = np.cumsum(evals, 1)
        eta_range = evals_cumsum[:, -1:] / evals_cumsum[:, :-1] - 1
        for i in range(N):
            s_range[i] = np.searchsorted(eta_range[i, ::-1], eta)
        s_range += n_neighbors - nev  # number of zero eigenvalues

        # Now calculate M.
        # This is the [N x N] matrix whose null space is the desired embedding
        M = M_container_constructor((N, N), dtype=np.float64)

        for i in range(N):
            s_i = s_range[i]

            # select bottom s_i eigenvectors and calculate alpha
            Vi = V[i, :, n_neighbors - s_i :]
            alpha_i = np.linalg.norm(Vi.sum(0)) / np.sqrt(s_i)

            # compute Householder matrix which satisfies
            #  Hi*Vi.T*ones(n_neighbors) = alpha_i*ones(s)
            # using prescription from paper
            h = np.full(s_i, alpha_i) - np.dot(Vi.T, np.ones(n_neighbors))

            norm_h = np.linalg.norm(h)
            if norm_h < modified_tol:
                h *= 0
            else:
                h /= norm_h

            # Householder matrix is
            #  >> Hi = np.identity(s_i) - 2*np.outer(h,h)
            # Then the weight matrix is
            #  >> Wi = np.dot(Vi,Hi) + (1-alpha_i) * w_reg[i,:,None]
            # We do this much more efficiently:
            Wi = Vi - 2 * np.outer(np.dot(Vi, h), h) + (1 - alpha_i) * w_reg[i, :, None]

            # Update M as follows:
            # >> W_hat = np.zeros( (N,s_i) )
            # >> W_hat[neighbors[i],:] = Wi
            # >> W_hat[i] -= 1
            # >> M += np.dot(W_hat,W_hat.T)
            # We can do this much more efficiently:
            nbrs_x, nbrs_y = np.meshgrid(neighbors[i], neighbors[i])
            M[nbrs_x, nbrs_y] += np.dot(Wi, Wi.T)
            Wi_sum1 = Wi.sum(1)
            if SCIPY_VERSION_BELOW_1_15:
                M[[i], neighbors[i]] -= Wi_sum1
                M[neighbors[i], [i]] -= Wi_sum1
            else:
                M[i, neighbors[i]] -= Wi_sum1
                M[neighbors[i], i] -= Wi_sum1
            M[i, i] += s_i

    elif method == "ltsa":
        neighbors = nbrs.kneighbors(
            X, n_neighbors=n_neighbors + 1, return_distance=False
        )
        neighbors = neighbors[:, 1:]

        M = M_container_constructor((N, N), dtype=np.float64)

        use_svd = n_neighbors > d_in

        for i in range(N):
            Xi = X[neighbors[i]]
            Xi -= Xi.mean(0)

            # compute n_components largest eigenvalues of Xi @ Xi^T
            if use_svd:
                v = svd(Xi, full_matrices=True)[0]
            else:
                Ci = np.dot(Xi, Xi.T)
                v = eigh(Ci)[1][:, ::-1]

            Gi = np.zeros((n_neighbors, n_components + 1))
            Gi[:, 1:] = v[:, :n_components]
            Gi[:, 0] = 1.0 / np.sqrt(n_neighbors)

            GiGiT = np.dot(Gi, Gi.T)

            nbrs_x, nbrs_y = np.meshgrid(neighbors[i], neighbors[i])
            M[nbrs_x, nbrs_y] -= GiGiT

            M[neighbors[i], neighbors[i]] += np.ones(shape=n_neighbors)

    if M_sparse:
        M = _align_api_if_sparse(M.tocsr())

    return null_space(
        M,
        n_components,
        k_skip=1,
        eigen_solver=eigen_solver,
        tol=tol,
        max_iter=max_iter,
        random_state=random_state,
    )


@validate_params(
    {
        "X": ["array-like", NearestNeighbors],
        "n_neighbors": [Interval(Integral, 1, None, closed="left")],
        "n_components": [Interval(Integral, 1, None, closed="left")],
        "reg": [Interval(Real, 0, None, closed="left")],
        "eigen_solver": [StrOptions({"auto", "arpack", "dense"})],
        "tol": [Interval(Real, 0, None, closed="left")],
        "max_iter": [Interval(Integral, 1, None, closed="left")],
        "method": [StrOptions({"standard", "hessian", "modified", "ltsa"})],
        "hessian_tol": [Interval(Real, 0, None, closed="left")],
        "modified_tol": [Interval(Real, 0, None, closed="left")],
        "random_state": ["random_state"],
        "n_jobs": [None, Integral],
    },
    prefer_skip_nested_validation=True,
)
def locally_linear_embedding(
    X,
    *,
    n_neighbors,
    n_components,
    reg=1e-3,
    eigen_solver="auto",
    tol=1e-6,
    max_iter=100,
    method="standard",
    hessian_tol=1e-4,
    modified_tol=1e-12,
    random_state=None,
    n_jobs=None,
):
    # [FUNCTION DOC]
    # Public, parameter-validated wrapper that forwards every argument to
    # _locally_linear_embedding.
    # Inputs: see docstring (X array or NearestNeighbors; n_neighbors, n_components
    # required keyword ints).
    # Outputs: tuple (Y ndarray (n_samples, n_components), squared_error float).
    # Exceptions: InvalidParameterError from validate_params; ValueError from the
    # implementation.
    # KNOWN ISSUE: the docstring's hessian requirement formula has an unbalanced
    # parenthesis; the code requires n_neighbors > n_components*(n_components+3)/2.
    """Perform a Locally Linear Embedding analysis on the data.

    Read more in the :ref:`User Guide <locally_linear_embedding>`.

    Parameters
    ----------
    X : {array-like, NearestNeighbors}
        Sample data, shape = (n_samples, n_features), in the form of a
        numpy array or a NearestNeighbors object.

    n_neighbors : int
        Number of neighbors to consider for each point.

    n_components : int
        Number of coordinates for the manifold.

    reg : float, default=1e-3
        Regularization constant, multiplies the trace of the local covariance
        matrix of the distances.

    eigen_solver : {'auto', 'arpack', 'dense'}, default='auto'
        auto : algorithm will attempt to choose the best method for input data

        arpack : use arnoldi iteration in shift-invert mode.
                    For this method, M may be a dense matrix, sparse matrix,
                    or general linear operator.
                    Warning: ARPACK can be unstable for some problems.  It is
                    best to try several random seeds in order to check results.

        dense  : use standard dense matrix operations for the eigenvalue
                    decomposition.  For this method, M must be an array
                    or matrix type.  This method should be avoided for
                    large problems.

    tol : float, default=1e-6
        Tolerance for 'arpack' method
        Not used if eigen_solver=='dense'.

    max_iter : int, default=100
        Maximum number of iterations for the arpack solver.

    method : {'standard', 'hessian', 'modified', 'ltsa'}, default='standard'
        standard : use the standard locally linear embedding algorithm.
                   see reference [1]_
        hessian  : use the Hessian eigenmap method.  This method requires
                   n_neighbors > n_components * (1 + (n_components + 1) / 2.
                   see reference [2]_
        modified : use the modified locally linear embedding algorithm.
                   see reference [3]_
        ltsa     : use local tangent space alignment algorithm
                   see reference [4]_

    hessian_tol : float, default=1e-4
        Tolerance for Hessian eigenmapping method.
        Only used if method == 'hessian'.

    modified_tol : float, default=1e-12
        Tolerance for modified LLE method.
        Only used if method == 'modified'.

    random_state : int, RandomState instance, default=None
        Determines the random number generator when ``solver`` == 'arpack'.
        Pass an int for reproducible results across multiple function calls.
        See :term:`Glossary <random_state>`.

    n_jobs : int or None, default=None
        The number of parallel jobs to run for neighbors search.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    Returns
    -------
    Y : ndarray of shape (n_samples, n_components)
        Embedding vectors.

    squared_error : float
        Reconstruction error for the embedding vectors. Equivalent to
        ``norm(Y - W Y, 'fro')**2``, where W are the reconstruction weights.

    References
    ----------

    .. [1] Roweis, S. & Saul, L. Nonlinear dimensionality reduction
        by locally linear embedding.  Science 290:2323 (2000).
    .. [2] Donoho, D. & Grimes, C. Hessian eigenmaps: Locally
        linear embedding techniques for high-dimensional data.
        Proc Natl Acad Sci U S A.  100:5591 (2003).
    .. [3] `Zhang, Z. & Wang, J. MLLE: Modified Locally Linear
        Embedding Using Multiple Weights.
        <https://citeseerx.ist.psu.edu/doc_view/pid/0b060fdbd92cbcc66b383bcaa9ba5e5e624d7ee3>`_
    .. [4] Zhang, Z. & Zha, H. Principal manifolds and nonlinear
        dimensionality reduction via tangent space alignment.
        Journal of Shanghai Univ.  8:406 (2004)

    Examples
    --------
    >>> from sklearn.datasets import load_digits
    >>> from sklearn.manifold import locally_linear_embedding
    >>> X, _ = load_digits(return_X_y=True)
    >>> X.shape
    (1797, 64)
    >>> embedding, _ = locally_linear_embedding(X[:100],n_neighbors=5, n_components=2)
    >>> embedding.shape
    (100, 2)
    """
    return _locally_linear_embedding(
        X=X,
        n_neighbors=n_neighbors,
        n_components=n_components,
        reg=reg,
        eigen_solver=eigen_solver,
        tol=tol,
        max_iter=max_iter,
        method=method,
        hessian_tol=hessian_tol,
        modified_tol=modified_tol,
        random_state=random_state,
        n_jobs=n_jobs,
    )


class LocallyLinearEmbedding(
    ClassNamePrefixFeaturesOutMixin,
    TransformerMixin,
    _UnstableArchMixin,
    BaseEstimator,
):
    # [CLASS DOC]
    # Transformer wrapper around _locally_linear_embedding. Fit stores nbrs_,
    # embedding_, reconstruction_error_; transform maps new points by barycentric
    # interpolation of training embeddings. Marked as architecture-unstable
    # (_UnstableArchMixin) and with class-name-prefixed output feature names.
    """Locally Linear Embedding.

    Read more in the :ref:`User Guide <locally_linear_embedding>`.

    Parameters
    ----------
    n_neighbors : int, default=5
        Number of neighbors to consider for each point.

    n_components : int, default=2
        Number of coordinates for the manifold.

    reg : float, default=1e-3
        Regularization constant, multiplies the trace of the local covariance
        matrix of the distances.

    eigen_solver : {'auto', 'arpack', 'dense'}, default='auto'
        The solver used to compute the eigenvectors. The available options are:

        - `'auto'` : algorithm will attempt to choose the best method for input
          data.
        - `'arpack'` : use arnoldi iteration in shift-invert mode. For this
          method, M may be a dense matrix, sparse matrix, or general linear
          operator.
        - `'dense'`  : use standard dense matrix operations for the eigenvalue
          decomposition. For this method, M must be an array or matrix type.
          This method should be avoided for large problems.

        .. warning::
           ARPACK can be unstable for some problems.  It is best to try several
           random seeds in order to check results.

    tol : float, default=1e-6
        Tolerance for 'arpack' method
        Not used if eigen_solver=='dense'.

    max_iter : int, default=100
        Maximum number of iterations for the arpack solver.
        Not used if eigen_solver=='dense'.

    method : {'standard', 'hessian', 'modified', 'ltsa'}, default='standard'
        - `standard`: use the standard locally linear embedding algorithm. see
          reference [1]_
        - `hessian`: use the Hessian eigenmap method. This method requires
          ``n_neighbors > n_components * (1 + (n_components + 1) / 2``. see
          reference [2]_
        - `modified`: use the modified locally linear embedding algorithm.
          see reference [3]_
        - `ltsa`: use local tangent space alignment algorithm. see
          reference [4]_

    hessian_tol : float, default=1e-4
        Tolerance for Hessian eigenmapping method.
        Only used if ``method == 'hessian'``.

    modified_tol : float, default=1e-12
        Tolerance for modified LLE method.
        Only used if ``method == 'modified'``.

    neighbors_algorithm : {'auto', 'brute', 'kd_tree', 'ball_tree'}, \
                          default='auto'
        Algorithm to use for nearest neighbors search, passed to
        :class:`~sklearn.neighbors.NearestNeighbors` instance.

    random_state : int, RandomState instance, default=None
        Determines the random number generator when
        ``eigen_solver`` == 'arpack'. Pass an int for reproducible results
        across multiple function calls. See :term:`Glossary <random_state>`.

    n_jobs : int or None, default=None
        The number of parallel jobs to run.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    Attributes
    ----------
    embedding_ : array-like, shape [n_samples, n_components]
        Stores the embedding vectors

    reconstruction_error_ : float
        Reconstruction error associated with `embedding_`

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    nbrs_ : NearestNeighbors object
        Stores nearest neighbors instance, including BallTree or KDtree
        if applicable.

    See Also
    --------
    SpectralEmbedding : Spectral embedding for non-linear dimensionality
        reduction.
    TSNE : Distributed Stochastic Neighbor Embedding.

    References
    ----------

    .. [1] Roweis, S. & Saul, L. Nonlinear dimensionality reduction
        by locally linear embedding.  Science 290:2323 (2000).
    .. [2] Donoho, D. & Grimes, C. Hessian eigenmaps: Locally
        linear embedding techniques for high-dimensional data.
        Proc Natl Acad Sci U S A.  100:5591 (2003).
    .. [3] `Zhang, Z. & Wang, J. MLLE: Modified Locally Linear
        Embedding Using Multiple Weights.
        <https://citeseerx.ist.psu.edu/doc_view/pid/0b060fdbd92cbcc66b383bcaa9ba5e5e624d7ee3>`_
    .. [4] Zhang, Z. & Zha, H. Principal manifolds and nonlinear
        dimensionality reduction via tangent space alignment.
        Journal of Shanghai Univ.  8:406 (2004)

    Examples
    --------
    >>> from sklearn.datasets import load_digits
    >>> from sklearn.manifold import LocallyLinearEmbedding
    >>> X, _ = load_digits(return_X_y=True)
    >>> X.shape
    (1797, 64)
    >>> embedding = LocallyLinearEmbedding(n_components=2)
    >>> X_transformed = embedding.fit_transform(X[:100])
    >>> X_transformed.shape
    (100, 2)
    """

    _parameter_constraints: dict = {
        "n_neighbors": [Interval(Integral, 1, None, closed="left")],
        "n_components": [Interval(Integral, 1, None, closed="left")],
        "reg": [Interval(Real, 0, None, closed="left")],
        "eigen_solver": [StrOptions({"auto", "arpack", "dense"})],
        "tol": [Interval(Real, 0, None, closed="left")],
        "max_iter": [Interval(Integral, 1, None, closed="left")],
        "method": [StrOptions({"standard", "hessian", "modified", "ltsa"})],
        "hessian_tol": [Interval(Real, 0, None, closed="left")],
        "modified_tol": [Interval(Real, 0, None, closed="left")],
        "neighbors_algorithm": [StrOptions({"auto", "brute", "kd_tree", "ball_tree"})],
        "random_state": ["random_state"],
        "n_jobs": [None, Integral],
    }

    def __init__(
        self,
        *,
        n_neighbors=5,
        n_components=2,
        reg=1e-3,
        eigen_solver="auto",
        tol=1e-6,
        max_iter=100,
        method="standard",
        hessian_tol=1e-4,
        modified_tol=1e-12,
        neighbors_algorithm="auto",
        random_state=None,
        n_jobs=None,
    ):
        # [FUNCTION DOC]
        # Stores all hyperparameters unchanged (keyword-only). Side effects: sets
        # attributes on self.
        self.n_neighbors = n_neighbors
        self.n_components = n_components
        self.reg = reg
        self.eigen_solver = eigen_solver
        self.tol = tol
        self.max_iter = max_iter
        self.method = method
        self.hessian_tol = hessian_tol
        self.modified_tol = modified_tol
        self.random_state = random_state
        self.neighbors_algorithm = neighbors_algorithm
        self.n_jobs = n_jobs

    def _fit_transform(self, X):
        # [FUNCTION DOC]
        # HELPER: Creates nbrs_ with n_neighbors, neighbors_algorithm and n_jobs, validates
        # X as float, fits nbrs_, then runs _locally_linear_embedding on the fitted nbrs_
        # with the stored hyperparameters and a RandomState from random_state.
        # Inputs: X array-like (n_samples, n_features).
        # Outputs: None.
        # Side effects: sets nbrs_, n_features_in_ (and feature_names_in_), embedding_,
        # reconstruction_error_, _n_features_out.
        # KNOWN ISSUE: nbrs_ is fitted with n_neighbors but _locally_linear_embedding
        # refits a fresh NearestNeighbors(n_neighbors + 1) from it, ignoring
        # neighbors_algorithm for the actual neighbor graph.
        self.nbrs_ = NearestNeighbors(
            n_neighbors=self.n_neighbors,
            algorithm=self.neighbors_algorithm,
            n_jobs=self.n_jobs,
        )

        random_state = check_random_state(self.random_state)
        X = validate_data(self, X, dtype=float)
        self.nbrs_.fit(X)
        self.embedding_, self.reconstruction_error_ = _locally_linear_embedding(
            X=self.nbrs_,
            n_neighbors=self.n_neighbors,
            n_components=self.n_components,
            eigen_solver=self.eigen_solver,
            tol=self.tol,
            max_iter=self.max_iter,
            method=self.method,
            hessian_tol=self.hessian_tol,
            modified_tol=self.modified_tol,
            random_state=random_state,
            reg=self.reg,
            n_jobs=self.n_jobs,
        )
        self._n_features_out = self.embedding_.shape[1]

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None):
        # [FUNCTION DOC]
        # Fits via _fit_transform and returns self. y is ignored.
        """Compute the embedding vectors for data X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training set.

        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        self : object
            Fitted `LocallyLinearEmbedding` class instance.
        """
        self._fit_transform(X)
        return self

    @_fit_context(prefer_skip_nested_validation=True)
    def fit_transform(self, X, y=None):
        # [FUNCTION DOC]
        # Fits via _fit_transform and returns embedding_ (n_samples, n_components).
        # KNOWN ISSUE: docstring says it "Returns the instance itself"; it returns the
        # embedding.
        """Compute the embedding vectors for data X and transform X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training set.

        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        X_new : array-like, shape (n_samples, n_components)
            Returns the instance itself.
        """
        self._fit_transform(X)
        return self.embedding_

    def transform(self, X):
        # [FUNCTION DOC]
        # Embeds new points: validate X against the fitted feature count, find each point's
        # n_neighbors nearest training points, compute barycenter_weights with reg, and
        # output for each point the weighted sum of its neighbors' embedding_ rows.
        # Inputs: X (n_queries, n_features).
        # Outputs: ndarray (n_queries, n_components), float64.
        # Exceptions: NotFittedError; ValueError on feature mismatch.
        # KNOWN ISSUE: docstring describes X as "Training set" and the return as "the
        # instance itself"; neither is accurate. A query equal to a training point is not
        # excluded from its own neighbors (expected behavior for new points).
        """
        Transform new points into embedding space.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training set.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_components)
            Returns the instance itself.

        Notes
        -----
        Because of scaling performed by this method, it is discouraged to use
        it together with methods that are not scale-invariant (like SVMs).
        """
        check_is_fitted(self)

        X = validate_data(self, X, reset=False)
        ind = self.nbrs_.kneighbors(
            X, n_neighbors=self.n_neighbors, return_distance=False
        )
        weights = barycenter_weights(X, self.nbrs_._fit_X, ind, reg=self.reg)
        X_new = np.empty((X.shape[0], self.n_components))
        for i in range(X.shape[0]):
            X_new[i] = np.dot(self.embedding_[ind[i]].T, weights[i])
        return X_new
