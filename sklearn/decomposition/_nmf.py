# [MODULE DOC]
# Non-negative matrix factorization X ~ W H with W, H >= 0. Provides the functional API
# non_negative_factorization, the estimators NMF (coordinate descent 'cd' or
# multiplicative update 'mu' solvers) and MiniBatchNMF (online multiplicative updates),
# their shared base _BaseNMF, and private helpers for initialization (random / NNDSVD
# variants), beta-divergence evaluation, and the solver iterations. The CD inner loop is
# the Cython _update_cdnmf_fast.
# Module constant EPSILON: float32 machine epsilon, used as a floor to avoid division by
# zero / negative powers of zero. Side effects at import: none.
"""Non-negative matrix factorization."""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import itertools
import time
import warnings
from abc import ABC
from math import sqrt
from numbers import Integral, Real

import numpy as np
import scipy.sparse as sp
from scipy import linalg

from sklearn._config import config_context
from sklearn.base import (
    BaseEstimator,
    ClassNamePrefixFeaturesOutMixin,
    TransformerMixin,
    _fit_context,
)
from sklearn.decomposition._cdnmf_fast import _update_cdnmf_fast
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils import check_array, check_random_state, gen_batches
from sklearn.utils._param_validation import Interval, StrOptions, validate_params
from sklearn.utils._sparse import _align_api_if_sparse
from sklearn.utils.extmath import _randomized_svd, safe_sparse_dot, squared_norm
from sklearn.utils.validation import check_is_fitted, check_non_negative, validate_data

EPSILON = np.finfo(np.float32).eps


def norm(x):
    # [FUNCTION DOC]
    # HELPER: Frobenius (L2) norm of an array: square root of the sum of squared entries
    #   (via squared_norm). Input: x array. Output: float.
    """Dot product-based Euclidean norm implementation.

    See: http://fa.bianp.net/blog/2011/computing-the-vector-norm/

    Parameters
    ----------
    x : array-like
        Vector for which to compute the norm.
    """
    return sqrt(squared_norm(x))


def trace_dot(X, Y):
    # [FUNCTION DOC]
    # HELPER: Trace of X transposed times Y, computed as the dot product of the
    #   flattened arrays. Inputs: X, Y arrays of equal shape. Output: scalar.
    """Trace of np.dot(X, Y.T).

    Parameters
    ----------
    X : array-like
        First matrix.
    Y : array-like
        Second matrix.
    """
    return np.dot(X.ravel(), Y.ravel())


def _check_init(A, shape, whom):
    # [FUNCTION DOC]
    # HELPER: Validates a user-supplied W or H: converts with check_array, checks each
    #   dimension against shape (a dimension given as 'auto' is not checked), requires
    #   non-negative values and at least one strictly positive entry.
    # Inputs: A array-like; shape 2-tuple of ints or 'auto'; whom str for messages.
    #   Outputs: none.
    # Exceptions: ValueError for wrong dimensions, negative values, or all-zero arrays.
    # KNOWN ISSUE: the converted array is discarded, so callers keep using the original
    #   object (which may be a list, not an ndarray).
    A = check_array(A)
    if shape[0] != "auto" and A.shape[0] != shape[0]:
        raise ValueError(
            f"Array with wrong first dimension passed to {whom}. Expected {shape[0]}, "
            f"but got {A.shape[0]}."
        )
    if shape[1] != "auto" and A.shape[1] != shape[1]:
        raise ValueError(
            f"Array with wrong second dimension passed to {whom}. Expected {shape[1]}, "
            f"but got {A.shape[1]}."
        )
    check_non_negative(A, whom)
    if np.max(A) == 0:
        raise ValueError(f"Array passed to {whom} is full of zeros.")


def _beta_divergence(X, W, H, beta, square_root=False):
    # [FUNCTION DOC]
    # HELPER: Computes the beta-divergence between X and W H. Converts a string loss to
    #   its float. Frobenius (beta 2): half the squared Frobenius norm of X - WH,
    #   computed without densifying WH for sparse X via norms and trace identities.
    #   Otherwise restricts to entries where X > EPSILON (WH taken only at X's non-zeros
    #   for sparse X) and floors WH at EPSILON; beta 1 (generalized KL): sum of X
    #   log(X/WH) plus sum(WH) minus sum(X); beta 0 (Itakura-Saito): sum of X/WH minus
    #   the number of entries of X minus sum of log(X/WH); other beta: (sum X^beta -
    #   beta sum X WH^(beta-1) + (beta-1) sum WH^beta) / (beta (beta-1)), with sum
    #   WH^beta looped over columns for sparse X.
    # Inputs: X (n_samples, n_features) dense or sparse (or scalar); W (n_samples,
    #   n_components); H (n_components, n_features); beta float or string; square_root
    #   bool.
    # Outputs: float divergence, or sqrt(2 * divergence) (clipped at 0) when
    #   square_root.
    # KNOWN ISSUE: for Itakura-Saito the constant subtracted is the total number of
    #   entries of X, while the other terms skip zero entries of X; this is only
    #   consistent when X has no zeros (which the estimators enforce for beta <= 0).
    """Compute the beta-divergence of X and dot(W, H).

    Parameters
    ----------
    X : float or array-like of shape (n_samples, n_features)

    W : float or array-like of shape (n_samples, n_components)

    H : float or array-like of shape (n_components, n_features)

    beta : float or {'frobenius', 'kullback-leibler', 'itakura-saito'}
        Parameter of the beta-divergence.
        If beta == 2, this is half the Frobenius *squared* norm.
        If beta == 1, this is the generalized Kullback-Leibler divergence.
        If beta == 0, this is the Itakura-Saito divergence.
        Else, this is the general beta-divergence.

    square_root : bool, default=False
        If True, return np.sqrt(2 * res)
        For beta == 2, it corresponds to the Frobenius norm.

    Returns
    -------
        res : float
            Beta divergence of X and np.dot(X, H).
    """
    beta = _beta_loss_to_float(beta)

    # The method can be called with scalars
    if not sp.issparse(X):
        X = np.atleast_2d(X)
    W = np.atleast_2d(W)
    H = np.atleast_2d(H)

    # Frobenius norm
    if beta == 2:
        # Avoid the creation of the dense np.dot(W, H) if X is sparse.
        if sp.issparse(X):
            norm_X = np.dot(X.data, X.data)
            norm_WH = trace_dot(np.linalg.multi_dot([W.T, W, H]), H)
            cross_prod = trace_dot((X @ H.T), W)
            res = (norm_X + norm_WH - 2.0 * cross_prod) / 2.0
        else:
            res = squared_norm(X - np.dot(W, H)) / 2.0

        if square_root:
            return np.sqrt(res * 2)
        else:
            return res

    if sp.issparse(X):
        # compute np.dot(W, H) only where X is nonzero
        WH_data = _special_sparse_dot(W, H, X).data
        X_data = X.data
    else:
        WH = np.dot(W, H)
        WH_data = WH.ravel()
        X_data = X.ravel()

    # do not affect the zeros: here 0 ** (-1) = 0 and not infinity
    indices = X_data > EPSILON
    WH_data = WH_data[indices]
    X_data = X_data[indices]

    # used to avoid division by zero
    WH_data[WH_data < EPSILON] = EPSILON

    # generalized Kullback-Leibler divergence
    if beta == 1:
        # fast and memory efficient computation of np.sum(np.dot(W, H))
        sum_WH = np.dot(np.sum(W, axis=0), np.sum(H, axis=1))
        # computes np.sum(X * log(X / WH)) only where X is nonzero
        div = X_data / WH_data
        res = np.dot(X_data, np.log(div))
        # add full np.sum(np.dot(W, H)) - np.sum(X)
        res += sum_WH - X_data.sum()

    # Itakura-Saito divergence
    elif beta == 0:
        div = X_data / WH_data
        res = np.sum(div) - np.prod(X.shape) - np.sum(np.log(div))

    # beta-divergence, beta not in (0, 1, 2)
    else:
        if sp.issparse(X):
            # slow loop, but memory efficient computation of :
            # np.sum(np.dot(W, H) ** beta)
            sum_WH_beta = 0
            for i in range(X.shape[1]):
                sum_WH_beta += np.sum(np.dot(W, H[:, i]) ** beta)

        else:
            sum_WH_beta = np.sum(WH**beta)

        sum_X_WH = np.dot(X_data, WH_data ** (beta - 1))
        res = (X_data**beta).sum() - beta * sum_X_WH
        res += sum_WH_beta * (beta - 1)
        res /= beta * (beta - 1)

    if square_root:
        res = max(res, 0)  # avoid negative number due to rounding errors
        return np.sqrt(2 * res)
    else:
        return res


def _special_sparse_dot(W, H, X):
    # [FUNCTION DOC]
    # HELPER: Computes W H only at the non-zero positions of a sparse X, in batches over
    #   the non-zeros (row of W dotted with column of H), returning a CSR matrix with
    #   X's shape; for dense X returns the full dense product.
    # Inputs: W (n_samples, n_components), H (n_components, n_features), X. Outputs: CSR
    #   matrix or ndarray (n_samples, n_features).
    """Computes np.dot(W, H), only where X is non zero."""
    if sp.issparse(X):
        ii, jj = X.nonzero()
        n_vals = ii.shape[0]
        dot_vals = np.empty(n_vals)
        n_components = W.shape[1]

        batch_size = max(n_components, n_vals // n_components)
        for start in range(0, n_vals, batch_size):
            batch = slice(start, start + batch_size)
            dot_vals[batch] = np.multiply(W[ii[batch], :], H.T[jj[batch], :]).sum(
                axis=1
            )

        WH = sp.coo_array((dot_vals, (ii, jj)), shape=X.shape)
        return _align_api_if_sparse(WH.tocsr())
    else:
        return np.dot(W, H)


def _beta_loss_to_float(beta_loss):
    # [FUNCTION DOC]
    # HELPER: Maps 'frobenius' to 2, 'kullback-leibler' to 1, 'itakura-saito' to 0; non-
    #   strings pass through unchanged.
    # Exceptions: KeyError for an unknown string.
    """Convert string beta_loss to float."""
    beta_loss_map = {"frobenius": 2, "kullback-leibler": 1, "itakura-saito": 0}
    if isinstance(beta_loss, str):
        beta_loss = beta_loss_map[beta_loss]
    return beta_loss


def _initialize_nmf(X, n_components, init=None, eps=1e-6, random_state=None):
    # [FUNCTION DOC]
    # HELPER: Computes initial W, H. Checks X is non-negative; rejects NNDSVD-type inits
    #   when n_components exceeds min(n_samples, n_features); init None resolves to
    #   'nndsvda' when possible else 'random'. 'random': absolute values of standard
    #   normals scaled by sqrt(mean(X)/n_components). NNDSVD: randomized SVD of X; the
    #   first component uses sqrt(s0) times the absolute leading singular vectors; for
    #   each further component splits the singular vector pair into positive and
    #   negative parts, keeps the pair with the larger product of norms, normalises it
    #   and scales by sqrt(s_j times that product); values below eps are zeroed;
    #   'nndsvda' fills zeros with mean(X), 'nndsvdar' with small random values |mean(X)
    #   * N(0,1) / 100|.
    # Inputs: X (n_samples, n_features); n_components int; init str or None; eps float
    #   threshold; random_state.
    # Outputs: tuple W (n_samples, n_components), H (n_components, n_features).
    # Exceptions: ValueError for negative X, too many components with an NNDSVD init, or
    #   an unknown init string.
    # Side effects: consumes randomness.
    """Algorithms for NMF initialization.

    Computes an initial guess for the non-negative
    rank k matrix approximation for X: X = WH.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The data matrix to be decomposed.

    n_components : int
        The number of components desired in the approximation.

    init :  {'random', 'nndsvd', 'nndsvda', 'nndsvdar'}, default=None
        Method used to initialize the procedure.
        Valid options:

        - None: 'nndsvda' if n_components <= min(n_samples, n_features),
            otherwise 'random'.

        - 'random': non-negative random matrices, scaled with:
            sqrt(X.mean() / n_components)

        - 'nndsvd': Nonnegative Double Singular Value Decomposition (NNDSVD)
            initialization (better for sparseness)

        - 'nndsvda': NNDSVD with zeros filled with the average of X
            (better when sparsity is not desired)

        - 'nndsvdar': NNDSVD with zeros filled with small random values
            (generally faster, less accurate alternative to NNDSVDa
            for when sparsity is not desired)

        - 'custom': use custom matrices W and H

        .. versionchanged:: 1.1
            When `init=None` and n_components is less than n_samples and n_features
            defaults to `nndsvda` instead of `nndsvd`.

    eps : float, default=1e-6
        Truncate all values less then this in output to zero.

    random_state : int, RandomState instance or None, default=None
        Used when ``init`` == 'nndsvdar' or 'random'. Pass an int for
        reproducible results across multiple function calls.
        See :term:`Glossary <random_state>`.

    Returns
    -------
    W : array-like of shape (n_samples, n_components)
        Initial guesses for solving X ~= WH.

    H : array-like of shape (n_components, n_features)
        Initial guesses for solving X ~= WH.

    References
    ----------
    C. Boutsidis, E. Gallopoulos: SVD based initialization: A head start for
    nonnegative matrix factorization - Pattern Recognition, 2008
    http://tinyurl.com/nndsvd
    """
    check_non_negative(X, "NMF initialization")
    n_samples, n_features = X.shape

    if (
        init is not None
        and init != "random"
        and n_components > min(n_samples, n_features)
    ):
        raise ValueError(
            "init = '{}' can only be used when "
            "n_components <= min(n_samples, n_features)".format(init)
        )

    if init is None:
        if n_components <= min(n_samples, n_features):
            init = "nndsvda"
        else:
            init = "random"

    # Random initialization
    if init == "random":
        avg = np.sqrt(X.mean() / n_components)
        rng = check_random_state(random_state)
        H = avg * rng.standard_normal(size=(n_components, n_features)).astype(
            X.dtype, copy=False
        )
        W = avg * rng.standard_normal(size=(n_samples, n_components)).astype(
            X.dtype, copy=False
        )
        np.abs(H, out=H)
        np.abs(W, out=W)
        return W, H

    # NNDSVD initialization
    U, S, V = _randomized_svd(X, n_components, random_state=random_state)
    W = np.zeros_like(U)
    H = np.zeros_like(V)

    # The leading singular triplet is non-negative
    # so it can be used as is for initialization.
    W[:, 0] = np.sqrt(S[0]) * np.abs(U[:, 0])
    H[0, :] = np.sqrt(S[0]) * np.abs(V[0, :])

    for j in range(1, n_components):
        x, y = U[:, j], V[j, :]

        # extract positive and negative parts of column vectors
        x_p, y_p = np.maximum(x, 0), np.maximum(y, 0)
        x_n, y_n = np.abs(np.minimum(x, 0)), np.abs(np.minimum(y, 0))

        # and their norms
        x_p_nrm, y_p_nrm = norm(x_p), norm(y_p)
        x_n_nrm, y_n_nrm = norm(x_n), norm(y_n)

        m_p, m_n = x_p_nrm * y_p_nrm, x_n_nrm * y_n_nrm

        # choose update
        if m_p > m_n:
            u = x_p / x_p_nrm
            v = y_p / y_p_nrm
            sigma = m_p
        else:
            u = x_n / x_n_nrm
            v = y_n / y_n_nrm
            sigma = m_n

        lbd = np.sqrt(S[j] * sigma)
        W[:, j] = lbd * u
        H[j, :] = lbd * v

    W[W < eps] = 0
    H[H < eps] = 0

    if init == "nndsvd":
        pass
    elif init == "nndsvda":
        avg = X.mean()
        W[W == 0] = avg
        H[H == 0] = avg
    elif init == "nndsvdar":
        rng = check_random_state(random_state)
        avg = X.mean()
        W[W == 0] = abs(avg * rng.standard_normal(size=len(W[W == 0])) / 100)
        H[H == 0] = abs(avg * rng.standard_normal(size=len(H[H == 0])) / 100)
    else:
        raise ValueError(
            "Invalid init parameter: got %r instead of one of %r"
            % (init, (None, "random", "nndsvd", "nndsvda", "nndsvdar"))
        )

    return W, H


def _update_coordinate_descent(X, W, Ht, l1_reg, l2_reg, shuffle, random_state):
    # [FUNCTION DOC]
    # HELPER: One coordinate-descent sweep updating W with Ht fixed: forms HHt = Ht^T Ht
    #   and XHt = X Ht, adds l2_reg to the diagonal of HHt and subtracts l1_reg from
    #   XHt, chooses the coordinate order (random permutation when shuffle), and calls
    #   the Cython _update_cdnmf_fast.
    # Inputs: X (n_samples, n_features); W (n_samples, n_components) updated in place;
    #   Ht (n_features, n_components); l1_reg, l2_reg floats; shuffle bool; random_state
    #   RandomState.
    # Outputs: float violation (sum of projected-gradient magnitudes) from the Cython
    #   routine.
    # Side effects: mutates W in place; consumes randomness when shuffle.
    """Helper function for _fit_coordinate_descent.

    Update W to minimize the objective function, iterating once over all
    coordinates. By symmetry, to update H, one can call
    _update_coordinate_descent(X.T, Ht, W, ...).

    """
    n_components = Ht.shape[1]

    HHt = np.dot(Ht.T, Ht)
    XHt = safe_sparse_dot(X, Ht)

    # L2 regularization corresponds to increase of the diagonal of HHt
    if l2_reg != 0.0:
        # adds l2_reg only on the diagonal
        HHt.flat[:: n_components + 1] += l2_reg
    # L1 regularization corresponds to decrease of each element of XHt
    if l1_reg != 0.0:
        XHt -= l1_reg

    if shuffle:
        permutation = random_state.permutation(n_components)
    else:
        permutation = np.arange(n_components)
    # The following seems to be required on 64-bit Windows w/ Python 3.5.
    permutation = np.asarray(permutation, dtype=np.intp)
    return _update_cdnmf_fast(W, HHt, XHt, permutation)


def _fit_coordinate_descent(
    X,
    W,
    H,
    tol=1e-4,
    max_iter=200,
    l1_reg_W=0,
    l1_reg_H=0,
    l2_reg_W=0,
    l2_reg_H=0,
    update_H=True,
    verbose=0,
    shuffle=False,
    random_state=None,
):
    # [FUNCTION DOC]
    # HELPER: Coordinate-descent NMF solver. Makes W and H transposed C-contiguous and
    #   converts CSC X to CSR, then for up to max_iter iterations updates W, and if
    #   update_H updates Ht using X transposed; records the first iteration's violation,
    #   stops immediately if it is zero, else stops when the violation relative to the
    #   initial one is at most tol (printing progress when verbose).
    # Inputs: X (n_samples, n_features); W, H initial factors; tol; max_iter; l1/l2
    #   regularizations for W and H; update_H bool; verbose; shuffle; random_state.
    # Outputs: tuple (W, H, number of iterations run).
    # Side effects: may mutate the passed W in place when already contiguous; prints
    #   when verbose; consumes randomness.
    # KNOWN ISSUE: the verbose message "Converged at iteration" prints n_iter + 1, one
    #   more than the actual iteration.
    """Compute Non-negative Matrix Factorization (NMF) with Coordinate Descent

    The objective function is minimized with an alternating minimization of W
    and H. Each minimization is done with a cyclic (up to a permutation of the
    features) Coordinate Descent.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Constant matrix.

    W : array-like of shape (n_samples, n_components)
        Initial guess for the solution.

    H : array-like of shape (n_components, n_features)
        Initial guess for the solution.

    tol : float, default=1e-4
        Tolerance of the stopping condition.

    max_iter : int, default=200
        Maximum number of iterations before timing out.

    l1_reg_W : float, default=0.
        L1 regularization parameter for W.

    l1_reg_H : float, default=0.
        L1 regularization parameter for H.

    l2_reg_W : float, default=0.
        L2 regularization parameter for W.

    l2_reg_H : float, default=0.
        L2 regularization parameter for H.

    update_H : bool, default=True
        Set to True, both W and H will be estimated from initial guesses.
        Set to False, only W will be estimated.

    verbose : int, default=0
        The verbosity level.

    shuffle : bool, default=False
        If true, randomize the order of coordinates in the CD solver.

    random_state : int, RandomState instance or None, default=None
        Used to randomize the coordinates in the CD solver, when
        ``shuffle`` is set to ``True``. Pass an int for reproducible
        results across multiple function calls.
        See :term:`Glossary <random_state>`.

    Returns
    -------
    W : ndarray of shape (n_samples, n_components)
        Solution to the non-negative least squares problem.

    H : ndarray of shape (n_components, n_features)
        Solution to the non-negative least squares problem.

    n_iter : int
        The number of iterations done by the algorithm.

    References
    ----------
    .. [1] :doi:`"Fast local algorithms for large scale nonnegative matrix and tensor
       factorizations" <10.1587/transfun.E92.A.708>`
       Cichocki, Andrzej, and P. H. A. N. Anh-Huy. IEICE transactions on fundamentals
       of electronics, communications and computer sciences 92.3: 708-721, 2009.
    """
    # ensure that W and Ht are both in C order in memory and that X is csr
    W = np.ascontiguousarray(W)
    Ht = np.ascontiguousarray(H.T)
    if sp.issparse(X) and X.format == "csc":
        X = X.tocsr()

    rng = check_random_state(random_state)

    for n_iter in range(1, max_iter + 1):
        violation = 0.0

        # Update W
        violation += _update_coordinate_descent(
            X, W, Ht, l1_reg_W, l2_reg_W, shuffle, rng
        )
        # Update H
        if update_H:
            violation += _update_coordinate_descent(
                X.T, Ht, W, l1_reg_H, l2_reg_H, shuffle, rng
            )

        if n_iter == 1:
            violation_init = violation

        if violation_init == 0:
            break

        if verbose:
            print("violation:", violation / violation_init)

        if violation / violation_init <= tol:
            if verbose:
                print("Converged at iteration", n_iter + 1)
            break

    return W, Ht.T, n_iter


def _multiplicative_update_w(
    X,
    W,
    H,
    beta_loss,
    l1_reg_W,
    l2_reg_W,
    gamma,
    H_sum=None,
    HHt=None,
    XHt=None,
    update_H=True,
):
    # [FUNCTION DOC]
    # HELPER: One multiplicative update of W (Fevotte & Idier). Frobenius: numerator X
    #   H^T (reused from XHt if given, copied when H is fixed), denominator W (H H^T).
    #   Other betas: compute WH (only at X's non-zeros for sparse X), floor it at
    #   EPSILON where needed, form X times WH^(beta-2) (division for KL, squared inverse
    #   for IS), numerator is that times H^T; denominator is the row sums of H for KL,
    #   else WH^(beta-1) times H^T (computed row by row for sparse X). Adds l1_reg_W to
    #   the denominator and l2_reg_W times W, replaces zero denominators with EPSILON,
    #   raises the ratio to power gamma if gamma != 1, and multiplies W by it.
    # Inputs: X; W (n_samples, n_components); H (n_components, n_features); beta_loss
    #   float; l1_reg_W, l2_reg_W; gamma exponent in (0, 1]; H_sum, HHt, XHt optional
    #   cached quantities; update_H bool (whether caches may be consumed).
    # Outputs: tuple (W updated, H_sum, HHt, XHt) so caches can be reused when H is
    #   fixed.
    # Side effects: mutates W in place; may overwrite the cached XHt when update_H is
    #   True.
    """Update W in Multiplicative Update NMF."""
    if beta_loss == 2:
        # Numerator
        if XHt is None:
            XHt = safe_sparse_dot(X, H.T)
        if update_H:
            # avoid a copy of XHt, which will be re-computed (update_H=True)
            numerator = XHt
        else:
            # preserve the XHt, which is not re-computed (update_H=False)
            numerator = XHt.copy()

        # Denominator
        if HHt is None:
            HHt = np.dot(H, H.T)
        denominator = np.dot(W, HHt)

    else:
        # Numerator
        # if X is sparse, compute WH only where X is non zero
        WH_safe_X = _special_sparse_dot(W, H, X)
        if sp.issparse(X):
            WH_safe_X_data = WH_safe_X.data
            X_data = X.data
        else:
            WH_safe_X_data = WH_safe_X
            X_data = X
            # copy used in the Denominator
            WH = WH_safe_X.copy()
            if beta_loss - 1.0 < 0:
                WH[WH < EPSILON] = EPSILON

        # to avoid taking a negative power of zero
        if beta_loss - 2.0 < 0:
            WH_safe_X_data[WH_safe_X_data < EPSILON] = EPSILON

        if beta_loss == 1:
            np.divide(X_data, WH_safe_X_data, out=WH_safe_X_data)
        elif beta_loss == 0:
            # speeds up computation time
            # refer to /numpy/numpy/issues/9363
            WH_safe_X_data **= -1
            WH_safe_X_data **= 2
            # element-wise multiplication
            WH_safe_X_data *= X_data
        else:
            WH_safe_X_data **= beta_loss - 2
            # element-wise multiplication
            WH_safe_X_data *= X_data

        # here numerator = dot(X * (dot(W, H) ** (beta_loss - 2)), H.T)
        numerator = safe_sparse_dot(WH_safe_X, H.T)

        # Denominator
        if beta_loss == 1:
            if H_sum is None:
                H_sum = np.sum(H, axis=1)  # shape(n_components, )
            denominator = H_sum[np.newaxis, :]

        else:
            # computation of WHHt = dot(dot(W, H) ** beta_loss - 1, H.T)
            if sp.issparse(X):
                # memory efficient computation
                # (compute row by row, avoiding the dense matrix WH)
                WHHt = np.empty(W.shape)
                for i in range(X.shape[0]):
                    WHi = np.dot(W[i, :], H)
                    if beta_loss - 1 < 0:
                        WHi[WHi < EPSILON] = EPSILON
                    WHi **= beta_loss - 1
                    WHHt[i, :] = np.dot(WHi, H.T)
            else:
                WH **= beta_loss - 1
                WHHt = np.dot(WH, H.T)
            denominator = WHHt

    # Add L1 and L2 regularization
    if l1_reg_W > 0:
        denominator += l1_reg_W
    if l2_reg_W > 0:
        denominator = denominator + l2_reg_W * W
    denominator[denominator == 0] = EPSILON

    numerator /= denominator
    delta_W = numerator

    # gamma is in ]0, 1]
    if gamma != 1:
        delta_W **= gamma

    W *= delta_W

    return W, H_sum, HHt, XHt


def _multiplicative_update_h(
    X, W, H, beta_loss, l1_reg_H, l2_reg_H, gamma, A=None, B=None, rho=None
):
    # [FUNCTION DOC]
    # HELPER: One multiplicative update of H, mirroring _multiplicative_update_w with
    #   roles transposed (numerator W^T X or W^T (X WH^(beta-2)); denominator W^T W H,
    #   column sums of W (zeros replaced by 1) for KL, or W^T WH^(beta-1) computed
    #   column by column for sparse X), plus l1/l2 regularization and EPSILON flooring.
    #   Batch mode: H is multiplied by (numerator/denominator)^gamma. Online mode (A and
    #   B given): H is raised to 1/gamma, numerator times H is accumulated into A and
    #   the denominator into B after both are scaled by rho, and H becomes (A/B)^gamma.
    # Inputs: X; W; H; beta_loss; l1_reg_H; l2_reg_H; gamma; A, B optional running
    #   numerator/denominator arrays (n_components, n_features); rho forgetting factor.
    # Outputs: updated H (same object in batch mode, a new array in online mode).
    # Side effects: mutates H in place (batch mode, and the 1/gamma power in online
    #   mode); mutates A and B in place.
    """update H in Multiplicative Update NMF."""
    if beta_loss == 2:
        numerator = safe_sparse_dot(W.T, X)
        denominator = np.linalg.multi_dot([W.T, W, H])

    else:
        # Numerator
        WH_safe_X = _special_sparse_dot(W, H, X)
        if sp.issparse(X):
            WH_safe_X_data = WH_safe_X.data
            X_data = X.data
        else:
            WH_safe_X_data = WH_safe_X
            X_data = X
            # copy used in the Denominator
            WH = WH_safe_X.copy()
            if beta_loss - 1.0 < 0:
                WH[WH < EPSILON] = EPSILON

        # to avoid division by zero
        if beta_loss - 2.0 < 0:
            WH_safe_X_data[WH_safe_X_data < EPSILON] = EPSILON

        if beta_loss == 1:
            np.divide(X_data, WH_safe_X_data, out=WH_safe_X_data)
        elif beta_loss == 0:
            # speeds up computation time
            # refer to /numpy/numpy/issues/9363
            WH_safe_X_data **= -1
            WH_safe_X_data **= 2
            # element-wise multiplication
            WH_safe_X_data *= X_data
        else:
            WH_safe_X_data **= beta_loss - 2
            # element-wise multiplication
            WH_safe_X_data *= X_data

        # here numerator = dot(W.T, (dot(W, H) ** (beta_loss - 2)) * X)
        numerator = safe_sparse_dot(W.T, WH_safe_X)

        # Denominator
        if beta_loss == 1:
            W_sum = np.sum(W, axis=0)  # shape(n_components, )
            W_sum[W_sum == 0] = 1.0
            denominator = W_sum[:, np.newaxis]

        # beta_loss not in (1, 2)
        else:
            # computation of WtWH = dot(W.T, dot(W, H) ** beta_loss - 1)
            if sp.issparse(X):
                # memory efficient computation
                # (compute column by column, avoiding the dense matrix WH)
                WtWH = np.empty(H.shape)
                for i in range(X.shape[1]):
                    WHi = np.dot(W, H[:, i])
                    if beta_loss - 1 < 0:
                        WHi[WHi < EPSILON] = EPSILON
                    WHi **= beta_loss - 1
                    WtWH[:, i] = np.dot(W.T, WHi)
            else:
                WH **= beta_loss - 1
                WtWH = np.dot(W.T, WH)
            denominator = WtWH

    # Add L1 and L2 regularization
    if l1_reg_H > 0:
        denominator += l1_reg_H
    if l2_reg_H > 0:
        denominator = denominator + l2_reg_H * H
    denominator[denominator == 0] = EPSILON

    if A is not None and B is not None:
        # Updates for the online nmf
        if gamma != 1:
            H **= 1 / gamma
        numerator *= H
        A *= rho
        B *= rho
        A += numerator
        B += denominator
        H = A / B

        if gamma != 1:
            H **= gamma
    else:
        delta_H = numerator
        delta_H /= denominator
        if gamma != 1:
            delta_H **= gamma
        H *= delta_H

    return H


def _fit_multiplicative_update(
    X,
    W,
    H,
    beta_loss="frobenius",
    max_iter=200,
    tol=1e-4,
    l1_reg_W=0,
    l1_reg_H=0,
    l2_reg_W=0,
    l2_reg_H=0,
    update_H=True,
    verbose=0,
):
    # [FUNCTION DOC]
    # HELPER: Multiplicative-update NMF solver. Sets the MM exponent gamma (1/(2-beta)
    #   for beta<1, 1/(beta-1) for beta>2, else 1); computes the initial sqrt beta-
    #   divergence; then for up to max_iter iterations updates W (reusing cached H
    #   products when H is fixed), zeroes tiny W entries for beta<1, updates H when
    #   update_H and zeroes tiny H entries for beta<=1; every 10 iterations (if tol>0)
    #   recomputes the error, optionally prints it, and stops when the decrease relative
    #   to the initial error is below tol.
    # Inputs: X; W, H initial factors; beta_loss; max_iter; tol; l1/l2 regularizations;
    #   update_H; verbose.
    # Outputs: tuple (W, H, number of iterations).
    # Side effects: W and H are updated in place; prints timing when verbose.
    """Compute Non-negative Matrix Factorization with Multiplicative Update.

    The objective function is _beta_divergence(X, WH) and is minimized with an
    alternating minimization of W and H. Each minimization is done with a
    Multiplicative Update.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Constant input matrix.

    W : array-like of shape (n_samples, n_components)
        Initial guess for the solution.

    H : array-like of shape (n_components, n_features)
        Initial guess for the solution.

    beta_loss : float or {'frobenius', 'kullback-leibler', \
            'itakura-saito'}, default='frobenius'
        String must be in {'frobenius', 'kullback-leibler', 'itakura-saito'}.
        Beta divergence to be minimized, measuring the distance between X
        and the dot product WH. Note that values different from 'frobenius'
        (or 2) and 'kullback-leibler' (or 1) lead to significantly slower
        fits. Note that for beta_loss <= 0 (or 'itakura-saito'), the input
        matrix X cannot contain zeros.

    max_iter : int, default=200
        Number of iterations.

    tol : float, default=1e-4
        Tolerance of the stopping condition.

    l1_reg_W : float, default=0.
        L1 regularization parameter for W.

    l1_reg_H : float, default=0.
        L1 regularization parameter for H.

    l2_reg_W : float, default=0.
        L2 regularization parameter for W.

    l2_reg_H : float, default=0.
        L2 regularization parameter for H.

    update_H : bool, default=True
        Set to True, both W and H will be estimated from initial guesses.
        Set to False, only W will be estimated.

    verbose : int, default=0
        The verbosity level.

    Returns
    -------
    W : ndarray of shape (n_samples, n_components)
        Solution to the non-negative least squares problem.

    H : ndarray of shape (n_components, n_features)
        Solution to the non-negative least squares problem.

    n_iter : int
        The number of iterations done by the algorithm.

    References
    ----------
    Lee, D. D., & Seung, H., S. (2001). Algorithms for Non-negative Matrix
    Factorization. Adv. Neural Inform. Process. Syst.. 13.
    Fevotte, C., & Idier, J. (2011). Algorithms for nonnegative matrix
    factorization with the beta-divergence. Neural Computation, 23(9).
    """
    start_time = time.time()

    beta_loss = _beta_loss_to_float(beta_loss)

    # gamma for Maximization-Minimization (MM) algorithm [Fevotte 2011]
    if beta_loss < 1:
        gamma = 1.0 / (2.0 - beta_loss)
    elif beta_loss > 2:
        gamma = 1.0 / (beta_loss - 1.0)
    else:
        gamma = 1.0

    # used for the convergence criterion
    error_at_init = _beta_divergence(X, W, H, beta_loss, square_root=True)
    previous_error = error_at_init

    H_sum, HHt, XHt = None, None, None
    for n_iter in range(1, max_iter + 1):
        # update W
        # H_sum, HHt and XHt are saved and reused if not update_H
        W, H_sum, HHt, XHt = _multiplicative_update_w(
            X,
            W,
            H,
            beta_loss=beta_loss,
            l1_reg_W=l1_reg_W,
            l2_reg_W=l2_reg_W,
            gamma=gamma,
            H_sum=H_sum,
            HHt=HHt,
            XHt=XHt,
            update_H=update_H,
        )

        # necessary for stability with beta_loss < 1
        if beta_loss < 1:
            W[W < np.finfo(np.float64).eps] = 0.0

        # update H (only at fit or fit_transform)
        if update_H:
            H = _multiplicative_update_h(
                X,
                W,
                H,
                beta_loss=beta_loss,
                l1_reg_H=l1_reg_H,
                l2_reg_H=l2_reg_H,
                gamma=gamma,
            )

            # These values will be recomputed since H changed
            H_sum, HHt, XHt = None, None, None

            # necessary for stability with beta_loss < 1
            if beta_loss <= 1:
                H[H < np.finfo(np.float64).eps] = 0.0

        # test convergence criterion every 10 iterations
        if tol > 0 and n_iter % 10 == 0:
            error = _beta_divergence(X, W, H, beta_loss, square_root=True)

            if verbose:
                iter_time = time.time()
                print(
                    "Epoch %02d reached after %.3f seconds, error: %f"
                    % (n_iter, iter_time - start_time, error)
                )

            if (previous_error - error) / error_at_init < tol:
                break
            previous_error = error

    # do not print if we have already printed in the convergence test
    if verbose and (tol == 0 or n_iter % 10 != 0):
        end_time = time.time()
        print(
            "Epoch %02d reached after %.3f seconds." % (n_iter, end_time - start_time)
        )

    return W, H, n_iter


@validate_params(
    {
        "X": ["array-like", "sparse matrix"],
        "W": ["array-like", None],
        "H": ["array-like", None],
        "update_H": ["boolean"],
    },
    prefer_skip_nested_validation=False,
)
def non_negative_factorization(
    X,
    W=None,
    H=None,
    n_components="auto",
    *,
    init=None,
    update_H=True,
    solver="cd",
    beta_loss="frobenius",
    tol=1e-4,
    max_iter=200,
    alpha_W=0.0,
    alpha_H="same",
    l1_ratio=0.0,
    random_state=None,
    verbose=0,
    shuffle=False,
):
    # [FUNCTION DOC]
    # Functional API: builds an NMF estimator from the arguments, validates its
    # parameters, validates X (CSR/CSC, float32/64) and runs NMF._fit_transform under
    # assume_finite.
    # Inputs: X (n_samples, n_features); W, H optional initial/fixed factors;
    #   n_components; init; update_H (False solves only for W with H fixed); solver;
    #   beta_loss; tol; max_iter; alpha_W; alpha_H; l1_ratio; random_state; verbose;
    #   shuffle.
    # Outputs: tuple (W (n_samples, n_components), H (n_components, n_features),
    #   n_iter).
    # Exceptions: InvalidParameterError and ValueErrors/TypeErrors from validation and
    #   _fit_transform.
    # Side effects: ConvergenceWarning/RuntimeWarning/UserWarning as in NMF; consumes
    #   randomness.
    """Compute Non-negative Matrix Factorization (NMF).

    Find two non-negative matrices (W, H) whose product approximates the non-
    negative matrix X. This factorization can be used for example for
    dimensionality reduction, source separation or topic extraction.

    The objective function is:

    .. math::

        L(W, H) &= 0.5 * ||X - WH||_{loss}^2

                &+ alpha\\_W * l1\\_ratio * n\\_features * ||vec(W)||_1

                &+ alpha\\_H * l1\\_ratio * n\\_samples * ||vec(H)||_1

                &+ 0.5 * alpha\\_W * (1 - l1\\_ratio) * n\\_features * ||W||_{Fro}^2

                &+ 0.5 * alpha\\_H * (1 - l1\\_ratio) * n\\_samples * ||H||_{Fro}^2,

    where :math:`||A||_{Fro}^2 = \\sum_{i,j} A_{ij}^2` (Frobenius norm) and
    :math:`||vec(A)||_1 = \\sum_{i,j} abs(A_{ij})` (Elementwise L1 norm)

    The generic norm :math:`||X - WH||_{loss}^2` may represent
    the Frobenius norm or another supported beta-divergence loss.
    The choice between options is controlled by the `beta_loss` parameter.

    The regularization terms are scaled by `n_features` for `W` and by `n_samples` for
    `H` to keep their impact balanced with respect to one another and to the data fit
    term as independent as possible of the size `n_samples` of the training set.

    The objective function is minimized with an alternating minimization of W
    and H. If H is given and update_H=False, it solves for W only.

    Note that the transformed data is named W and the components matrix is named H. In
    the NMF literature, the naming convention is usually the opposite since the data
    matrix X is transposed.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        Constant matrix.

    W : array-like of shape (n_samples, n_components), default=None
        If `init='custom'`, it is used as initial guess for the solution.
        If `update_H=False`, it is initialised as an array of zeros, unless
        `solver='mu'`, then it is filled with values calculated by
        `np.sqrt(X.mean() / self._n_components)`.
        If `None`, uses the initialisation method specified in `init`.

    H : array-like of shape (n_components, n_features), default=None
        If `init='custom'`, it is used as initial guess for the solution.
        If `update_H=False`, it is used as a constant, to solve for W only.
        If `None`, uses the initialisation method specified in `init`.

    n_components : int or {'auto'} or None, default='auto'
        Number of components. If `None`, all features are kept.
        If `n_components='auto'`, the number of components is automatically inferred
        from `W` or `H` shapes.

        .. versionchanged:: 1.4
            Added `'auto'` value.

        .. versionchanged:: 1.6
            Default value changed from `None` to `'auto'`.

    init : {'random', 'nndsvd', 'nndsvda', 'nndsvdar', 'custom'}, default=None
        Method used to initialize the procedure.

        Valid options:

        - None: 'nndsvda' if n_components < n_features, otherwise 'random'.
        - 'random': non-negative random matrices, scaled with:
          `sqrt(X.mean() / n_components)`
        - 'nndsvd': Nonnegative Double Singular Value Decomposition (NNDSVD)
          initialization (better for sparseness)
        - 'nndsvda': NNDSVD with zeros filled with the average of X
          (better when sparsity is not desired)
        - 'nndsvdar': NNDSVD with zeros filled with small random values
          (generally faster, less accurate alternative to NNDSVDa
          for when sparsity is not desired)
        - 'custom': If `update_H=True`, use custom matrices W and H which must both
          be provided. If `update_H=False`, then only custom matrix H is used.

        .. versionchanged:: 0.23
            The default value of `init` changed from 'random' to None in 0.23.

        .. versionchanged:: 1.1
            When `init=None` and n_components is less than n_samples and n_features
            defaults to `nndsvda` instead of `nndsvd`.

    update_H : bool, default=True
        Set to True, both W and H will be estimated from initial guesses.
        Set to False, only W will be estimated.

    solver : {'cd', 'mu'}, default='cd'
        Numerical solver to use:

        - 'cd' is a Coordinate Descent solver that uses Fast Hierarchical
          Alternating Least Squares (Fast HALS).
        - 'mu' is a Multiplicative Update solver.

        .. versionadded:: 0.17
           Coordinate Descent solver.

        .. versionadded:: 0.19
           Multiplicative Update solver.

    beta_loss : float or {'frobenius', 'kullback-leibler', \
            'itakura-saito'}, default='frobenius'
        Beta divergence to be minimized, measuring the distance between X
        and the dot product WH. Note that values different from 'frobenius'
        (or 2) and 'kullback-leibler' (or 1) lead to significantly slower
        fits. Note that for beta_loss <= 0 (or 'itakura-saito'), the input
        matrix X cannot contain zeros. Used only in 'mu' solver.

        .. versionadded:: 0.19

    tol : float, default=1e-4
        Tolerance of the stopping condition.

    max_iter : int, default=200
        Maximum number of iterations before timing out.

    alpha_W : float, default=0.0
        Constant that multiplies the regularization terms of `W`. Set it to zero
        (default) to have no regularization on `W`.

        .. versionadded:: 1.0

    alpha_H : float or "same", default="same"
        Constant that multiplies the regularization terms of `H`. Set it to zero to
        have no regularization on `H`. If "same" (default), it takes the same value as
        `alpha_W`.

        .. versionadded:: 1.0

    l1_ratio : float, default=0.0
        The regularization mixing parameter, with 0 <= l1_ratio <= 1.
        For l1_ratio = 0 the penalty is an elementwise L2 penalty
        (aka Frobenius Norm).
        For l1_ratio = 1 it is an elementwise L1 penalty.
        For 0 < l1_ratio < 1, the penalty is a combination of L1 and L2.

    random_state : int, RandomState instance or None, default=None
        Used for NMF initialisation (when ``init`` == 'nndsvdar' or
        'random'), and in Coordinate Descent. Pass an int for reproducible
        results across multiple function calls.
        See :term:`Glossary <random_state>`.

    verbose : int, default=0
        The verbosity level.

    shuffle : bool, default=False
        If true, randomize the order of coordinates in the CD solver.

    Returns
    -------
    W : ndarray of shape (n_samples, n_components)
        Solution to the non-negative least squares problem.

    H : ndarray of shape (n_components, n_features)
        Solution to the non-negative least squares problem.

    n_iter : int
        Actual number of iterations.

    References
    ----------
    .. [1] :doi:`"Fast local algorithms for large scale nonnegative matrix and tensor
       factorizations" <10.1587/transfun.E92.A.708>`
       Cichocki, Andrzej, and P. H. A. N. Anh-Huy. IEICE transactions on fundamentals
       of electronics, communications and computer sciences 92.3: 708-721, 2009.

    .. [2] :doi:`"Algorithms for nonnegative matrix factorization with the
       beta-divergence" <10.1162/NECO_a_00168>`
       Fevotte, C., & Idier, J. (2011). Neural Computation, 23(9).

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[1,1], [2, 1], [3, 1.2], [4, 1], [5, 0.8], [6, 1]])
    >>> from sklearn.decomposition import non_negative_factorization
    >>> W, H, n_iter = non_negative_factorization(
    ...     X, n_components=2, init='random', random_state=0)
    """
    est = NMF(
        n_components=n_components,
        init=init,
        solver=solver,
        beta_loss=beta_loss,
        tol=tol,
        max_iter=max_iter,
        random_state=random_state,
        alpha_W=alpha_W,
        alpha_H=alpha_H,
        l1_ratio=l1_ratio,
        verbose=verbose,
        shuffle=shuffle,
    )
    est._validate_params()

    X = check_array(X, accept_sparse=("csr", "csc"), dtype=[np.float64, np.float32])

    with config_context(assume_finite=True):
        W, H, n_iter = est._fit_transform(X, W=W, H=H, update_H=update_H)

    return W, H, n_iter


class _BaseNMF(ClassNamePrefixFeaturesOutMixin, TransformerMixin, BaseEstimator, ABC):
    # [CLASS DOC]
    # Abstract base for NMF and MiniBatchNMF: holds common hyperparameters and parameter
    # constraints, resolves n_components and beta_loss, checks/initialises W and H,
    # computes the scaled regularization strengths, and provides fit, inverse_transform,
    # feature-name and tag support. Subclasses implement fit_transform, _fit_transform
    # and transform.
    """Base class for NMF and MiniBatchNMF."""

    _parameter_constraints: dict = {
        "n_components": [
            Interval(Integral, 1, None, closed="left"),
            None,
            StrOptions({"auto"}),
        ],
        "init": [
            StrOptions({"random", "nndsvd", "nndsvda", "nndsvdar", "custom"}),
            None,
        ],
        "beta_loss": [
            StrOptions({"frobenius", "kullback-leibler", "itakura-saito"}),
            Real,
        ],
        "tol": [Interval(Real, 0, None, closed="left")],
        "max_iter": [Interval(Integral, 1, None, closed="left")],
        "random_state": ["random_state"],
        "alpha_W": [Interval(Real, 0, None, closed="left")],
        "alpha_H": [Interval(Real, 0, None, closed="left"), StrOptions({"same"})],
        "l1_ratio": [Interval(Real, 0, 1, closed="both")],
        "verbose": ["verbose"],
    }

    def __init__(
        self,
        n_components="auto",
        *,
        init=None,
        beta_loss="frobenius",
        tol=1e-4,
        max_iter=200,
        random_state=None,
        alpha_W=0.0,
        alpha_H="same",
        l1_ratio=0.0,
        verbose=0,
    ):
        # [FUNCTION DOC]
        # Stores common hyperparameters (n_components, init, beta_loss, tol, max_iter,
        # random_state, alpha_W, alpha_H, l1_ratio, verbose) as attributes without
        # validation.
        self.n_components = n_components
        self.init = init
        self.beta_loss = beta_loss
        self.tol = tol
        self.max_iter = max_iter
        self.random_state = random_state
        self.alpha_W = alpha_W
        self.alpha_H = alpha_H
        self.l1_ratio = l1_ratio
        self.verbose = verbose

    def _check_params(self, X):
        # [FUNCTION DOC]
        # HELPER: Resolves private settings: _n_components from n_components (None
        #   becomes n_features; 'auto' is kept and resolved later in _check_w_h) and
        #   _beta_loss as a float.
        # Inputs: X (n_samples, n_features). Side effects: sets _n_components and
        #   _beta_loss.
        # n_components
        self._n_components = self.n_components
        if self._n_components is None:
            self._n_components = X.shape[1]

        # beta_loss
        self._beta_loss = _beta_loss_to_float(self.beta_loss)

    def _check_w_h(self, X, W, H, update_H):
        # [FUNCTION DOC]
        # HELPER: Prepares W and H. Custom init with update_H: validates both against
        #   expected shapes and dtype of X, inferring _n_components from H if 'auto'.
        #   With update_H False: warns that W is ignored, validates H (shape/dtype),
        #   infers _n_components, and initialises W to sqrt(mean(X)/n_components)
        #   everywhere for solver 'mu' or zeros otherwise. Otherwise: warns if W or H
        #   were given, resolves 'auto' to n_features and calls _initialize_nmf.
        # Inputs: X; W, H arrays or None; update_H bool.
        # Outputs: tuple (W, H).
        # Exceptions: ValueError from _check_init; TypeError when dtypes differ from X.
        # Side effects: may set _n_components; RuntimeWarning for ignored inputs;
        #   consumes randomness.
        # KNOWN ISSUE: the update_H=False branch reads self.solver, which only NMF
        #   defines; MiniBatchNMF would raise AttributeError there (it never takes this
        #   path because its transform uses _solve_W).
        """Check W and H, or initialize them."""
        n_samples, n_features = X.shape

        if self.init == "custom" and update_H:
            _check_init(H, (self._n_components, n_features), "NMF (input H)")
            _check_init(W, (n_samples, self._n_components), "NMF (input W)")
            if self._n_components == "auto":
                self._n_components = H.shape[0]

            if H.dtype != X.dtype or W.dtype != X.dtype:
                raise TypeError(
                    "H and W should have the same dtype as X. Got "
                    "H.dtype = {} and W.dtype = {}.".format(H.dtype, W.dtype)
                )

        elif not update_H:
            if W is not None:
                warnings.warn(
                    "When update_H=False, the provided initial W is not used.",
                    RuntimeWarning,
                )

            _check_init(H, (self._n_components, n_features), "NMF (input H)")
            if self._n_components == "auto":
                self._n_components = H.shape[0]

            if H.dtype != X.dtype:
                raise TypeError(
                    "H should have the same dtype as X. Got H.dtype = {}.".format(
                        H.dtype
                    )
                )

            # 'mu' solver should not be initialized by zeros
            if self.solver == "mu":
                avg = np.sqrt(X.mean() / self._n_components)
                W = np.full((n_samples, self._n_components), avg, dtype=X.dtype)
            else:
                W = np.zeros((n_samples, self._n_components), dtype=X.dtype)

        else:
            if W is not None or H is not None:
                warnings.warn(
                    (
                        "When init!='custom', provided W or H are ignored. Set "
                        " init='custom' to use them as initialization."
                    ),
                    RuntimeWarning,
                )

            if self._n_components == "auto":
                self._n_components = X.shape[1]

            W, H = _initialize_nmf(
                X, self._n_components, init=self.init, random_state=self.random_state
            )

        return W, H

    def _compute_regularization(self, X):
        # [FUNCTION DOC]
        # HELPER: Returns (l1_reg_W, l1_reg_H, l2_reg_W, l2_reg_H): alpha_W scaled by
        #   n_features and alpha_H (equal to alpha_W when 'same') scaled by n_samples,
        #   each split by l1_ratio into L1 and (1 - l1_ratio) into L2 parts.
        # Inputs: X used only for its shape.
        """Compute scaled regularization terms."""
        n_samples, n_features = X.shape
        alpha_W = self.alpha_W
        alpha_H = self.alpha_W if self.alpha_H == "same" else self.alpha_H

        l1_reg_W = n_features * alpha_W * self.l1_ratio
        l1_reg_H = n_samples * alpha_H * self.l1_ratio
        l2_reg_W = n_features * alpha_W * (1.0 - self.l1_ratio)
        l2_reg_H = n_samples * alpha_H * (1.0 - self.l1_ratio)

        return l1_reg_W, l1_reg_H, l2_reg_W, l2_reg_H

    def fit(self, X, y=None, **params):
        # [FUNCTION DOC]
        # Fits by calling fit_transform(X, **params) and returns self.
        """Learn a NMF model for the data X.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vector, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        y : Ignored
            Not used, present for API consistency by convention.

        **params : kwargs
            Parameters (keyword arguments) and values passed to
            the fit_transform instance.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        # param validation is done in fit_transform

        self.fit_transform(X, **params)
        return self

    def inverse_transform(self, X):
        # [FUNCTION DOC]
        # Reconstructs data as X times components_.
        # Inputs: X (n_samples, n_components) dense or sparse. Outputs: (n_samples,
        #   n_features).
        # Exceptions: NotFittedError. Note X is not validated.
        """Transform data back to its original space.

        .. versionadded:: 0.18

        Parameters
        ----------
        X : {ndarray, sparse matrix} of shape (n_samples, n_components)
            Transformed data matrix.

        Returns
        -------
        X_original : ndarray of shape (n_samples, n_features)
            Returns a data matrix of the original shape.
        """

        check_is_fitted(self)
        return X @ self.components_

    @property
    def _n_features_out(self):
        # [FUNCTION DOC]
        # HELPER: Property returning the number of output features, the row count of
        #   components_.
        """Number of transformed output features."""
        return self.components_.shape[0]

    def __sklearn_tags__(self):
        # [FUNCTION DOC]
        # Extends parent tags: non-negative input only, sparse accepted, float64/float32
        # preserved.
        tags = super().__sklearn_tags__()
        tags.input_tags.positive_only = True
        tags.input_tags.sparse = True
        tags.transformer_tags.preserves_dtype = ["float64", "float32"]
        return tags


class NMF(_BaseNMF):
    # [CLASS DOC]
    # Batch NMF estimator adding the solver ('cd' or 'mu') and shuffle hyperparameters.
    # Fitting sets components_, n_components_, reconstruction_err_, n_iter_ and
    # n_features_in_.
    """Non-Negative Matrix Factorization (NMF).

    Find two non-negative matrices, i.e. matrices with all non-negative elements, (W, H)
    whose product approximates the non-negative matrix X. This factorization can be used
    for example for dimensionality reduction, source separation or topic extraction.

    The objective function is:

    .. math::

        L(W, H) &= 0.5 * ||X - WH||_{loss}^2

                &+ alpha\\_W * l1\\_ratio * n\\_features * ||vec(W)||_1

                &+ alpha\\_H * l1\\_ratio * n\\_samples * ||vec(H)||_1

                &+ 0.5 * alpha\\_W * (1 - l1\\_ratio) * n\\_features * ||W||_{Fro}^2

                &+ 0.5 * alpha\\_H * (1 - l1\\_ratio) * n\\_samples * ||H||_{Fro}^2,

    where :math:`||A||_{Fro}^2 = \\sum_{i,j} A_{ij}^2` (Frobenius norm) and
    :math:`||vec(A)||_1 = \\sum_{i,j} abs(A_{ij})` (Elementwise L1 norm).

    The generic norm :math:`||X - WH||_{loss}` may represent
    the Frobenius norm or another supported beta-divergence loss.
    The choice between options is controlled by the `beta_loss` parameter.

    The regularization terms are scaled by `n_features` for `W` and by `n_samples` for
    `H` to keep their impact balanced with respect to one another and to the data fit
    term as independent as possible of the size `n_samples` of the training set.

    The objective function is minimized with an alternating minimization of W
    and H.

    Note that the transformed data is named W and the components matrix is named H. In
    the NMF literature, the naming convention is usually the opposite since the data
    matrix X is transposed.

    Read more in the :ref:`User Guide <NMF>`.

    Parameters
    ----------
    n_components : int or {'auto'} or None, default='auto'
        Number of components. If `None`, all features are kept.
        If `n_components='auto'`, the number of components is automatically inferred
        from W or H shapes.

        .. versionchanged:: 1.4
            Added `'auto'` value.

        .. versionchanged:: 1.6
            Default value changed from `None` to `'auto'`.

    init : {'random', 'nndsvd', 'nndsvda', 'nndsvdar', 'custom'}, default=None
        Method used to initialize the procedure.
        Valid options:

        - `None`: 'nndsvda' if n_components <= min(n_samples, n_features),
          otherwise random.

        - `'random'`: non-negative random matrices, scaled with:
          `sqrt(X.mean() / n_components)`

        - `'nndsvd'`: Nonnegative Double Singular Value Decomposition (NNDSVD)
          initialization (better for sparseness)

        - `'nndsvda'`: NNDSVD with zeros filled with the average of X
          (better when sparsity is not desired)

        - `'nndsvdar'` NNDSVD with zeros filled with small random values
          (generally faster, less accurate alternative to NNDSVDa
          for when sparsity is not desired)

        - `'custom'`: Use custom matrices `W` and `H` which must both be provided.

        .. versionchanged:: 1.1
            When `init=None` and n_components is less than n_samples and n_features
            defaults to `nndsvda` instead of `nndsvd`.

    solver : {'cd', 'mu'}, default='cd'
        Numerical solver to use:

        - 'cd' is a Coordinate Descent solver.
        - 'mu' is a Multiplicative Update solver.

        .. versionadded:: 0.17
           Coordinate Descent solver.

        .. versionadded:: 0.19
           Multiplicative Update solver.

    beta_loss : float or {'frobenius', 'kullback-leibler', \
            'itakura-saito'}, default='frobenius'
        Beta divergence to be minimized, measuring the distance between X
        and the dot product WH. Note that values different from 'frobenius'
        (or 2) and 'kullback-leibler' (or 1) lead to significantly slower
        fits. Note that for beta_loss <= 0 (or 'itakura-saito'), the input
        matrix X cannot contain zeros. Used only in 'mu' solver.

        .. versionadded:: 0.19

    tol : float, default=1e-4
        Tolerance of the stopping condition.

    max_iter : int, default=200
        Maximum number of iterations before timing out.

    random_state : int, RandomState instance or None, default=None
        Used for initialisation (when ``init`` == 'nndsvdar' or
        'random'), and in Coordinate Descent. Pass an int for reproducible
        results across multiple function calls.
        See :term:`Glossary <random_state>`.

    alpha_W : float, default=0.0
        Constant that multiplies the regularization terms of `W`. Set it to zero
        (default) to have no regularization on `W`.

        .. versionadded:: 1.0

    alpha_H : float or "same", default="same"
        Constant that multiplies the regularization terms of `H`. Set it to zero to
        have no regularization on `H`. If "same" (default), it takes the same value as
        `alpha_W`.

        .. versionadded:: 1.0

    l1_ratio : float, default=0.0
        The regularization mixing parameter, with 0 <= l1_ratio <= 1.
        For l1_ratio = 0 the penalty is an elementwise L2 penalty
        (aka Frobenius Norm).
        For l1_ratio = 1 it is an elementwise L1 penalty.
        For 0 < l1_ratio < 1, the penalty is a combination of L1 and L2.

        .. versionadded:: 0.17
           Regularization parameter *l1_ratio* used in the Coordinate Descent
           solver.

    verbose : int, default=0
        Whether to be verbose.

    shuffle : bool, default=False
        If true, randomize the order of coordinates in the CD solver.

        .. versionadded:: 0.17
           *shuffle* parameter used in the Coordinate Descent solver.

    Attributes
    ----------
    components_ : ndarray of shape (n_components, n_features)
        Factorization matrix, sometimes called 'dictionary'.

    n_components_ : int
        The number of components. It is same as the `n_components` parameter
        if it was given. Otherwise, it will be same as the number of
        features.

    reconstruction_err_ : float
        Frobenius norm of the matrix difference, or beta-divergence, between
        the training data ``X`` and the reconstructed data ``WH`` from
        the fitted model.

    n_iter_ : int
        Actual number of iterations.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    DictionaryLearning : Find a dictionary that sparsely encodes data.
    MiniBatchSparsePCA : Mini-batch Sparse Principal Components Analysis.
    PCA : Principal component analysis.
    SparseCoder : Find a sparse representation of data from a fixed,
        precomputed dictionary.
    SparsePCA : Sparse Principal Components Analysis.
    TruncatedSVD : Dimensionality reduction using truncated SVD.

    References
    ----------
    .. [1] :doi:`"Fast local algorithms for large scale nonnegative matrix and tensor
       factorizations" <10.1587/transfun.E92.A.708>`
       Cichocki, Andrzej, and P. H. A. N. Anh-Huy. IEICE transactions on fundamentals
       of electronics, communications and computer sciences 92.3: 708-721, 2009.

    .. [2] :doi:`"Algorithms for nonnegative matrix factorization with the
       beta-divergence" <10.1162/NECO_a_00168>`
       Fevotte, C., & Idier, J. (2011). Neural Computation, 23(9).

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[1, 1], [2, 1], [3, 1.2], [4, 1], [5, 0.8], [6, 1]])
    >>> from sklearn.decomposition import NMF
    >>> model = NMF(n_components=2, init='random', random_state=0)
    >>> W = model.fit_transform(X)
    >>> H = model.components_
    """

    _parameter_constraints: dict = {
        **_BaseNMF._parameter_constraints,
        "solver": [StrOptions({"mu", "cd"})],
        "shuffle": ["boolean"],
    }

    def __init__(
        self,
        n_components="auto",
        *,
        init=None,
        solver="cd",
        beta_loss="frobenius",
        tol=1e-4,
        max_iter=200,
        random_state=None,
        alpha_W=0.0,
        alpha_H="same",
        l1_ratio=0.0,
        verbose=0,
        shuffle=False,
    ):
        # [FUNCTION DOC]
        # Passes the common hyperparameters to _BaseNMF and stores solver and shuffle.
        super().__init__(
            n_components=n_components,
            init=init,
            beta_loss=beta_loss,
            tol=tol,
            max_iter=max_iter,
            random_state=random_state,
            alpha_W=alpha_W,
            alpha_H=alpha_H,
            l1_ratio=l1_ratio,
            verbose=verbose,
        )

        self.solver = solver
        self.shuffle = shuffle

    def _check_params(self, X):
        # [FUNCTION DOC]
        # HELPER: Runs the base checks, then rejects non-Frobenius losses for the 'cd'
        #   solver and warns when 'mu' is combined with init 'nndsvd' (zeros cannot be
        #   updated).
        # Outputs: self. Exceptions: ValueError. Side effects: UserWarning; sets
        #   _n_components and _beta_loss.
        super()._check_params(X)

        # solver
        if self.solver != "mu" and self.beta_loss not in (2, "frobenius"):
            # 'mu' is the only solver that handles other beta losses than 'frobenius'
            raise ValueError(
                f"Invalid beta_loss parameter: solver {self.solver!r} does not handle "
                f"beta_loss = {self.beta_loss!r}"
            )
        if self.solver == "mu" and self.init == "nndsvd":
            warnings.warn(
                (
                    "The multiplicative update ('mu') solver cannot update "
                    "zeros present in the initialization, and so leads to "
                    "poorer results when used jointly with init='nndsvd'. "
                    "You may try init='nndsvda' or init='nndsvdar' instead."
                ),
                UserWarning,
            )

        return self

    @_fit_context(prefer_skip_nested_validation=True)
    def fit_transform(self, X, y=None, W=None, H=None):
        # [FUNCTION DOC]
        # Validates X, runs _fit_transform under assume_finite, and stores
        # reconstruction_err_ (sqrt beta-divergence), n_components_, components_ and
        # n_iter_.
        # Inputs: X (n_samples, n_features); y ignored; W, H optional custom
        #   initialisations. Outputs: W (n_samples, n_components).
        # Exceptions/side effects: as in _fit_transform.
        """Learn a NMF model for the data X and returns the transformed data.

        This is more efficient than calling fit followed by transform.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vector, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        y : Ignored
            Not used, present for API consistency by convention.

        W : array-like of shape (n_samples, n_components), default=None
            If `init='custom'`, it is used as initial guess for the solution.
            If `None`, uses the initialisation method specified in `init`.

        H : array-like of shape (n_components, n_features), default=None
            If `init='custom'`, it is used as initial guess for the solution.
            If `None`, uses the initialisation method specified in `init`.

        Returns
        -------
        W : ndarray of shape (n_samples, n_components)
            Transformed data.
        """
        X = validate_data(
            self, X, accept_sparse=("csr", "csc"), dtype=[np.float64, np.float32]
        )

        with config_context(assume_finite=True):
            W, H, n_iter = self._fit_transform(X, W=W, H=H)

        self.reconstruction_err_ = _beta_divergence(
            X, W, H, self._beta_loss, square_root=True
        )

        self.n_components_ = H.shape[0]
        self.components_ = H
        self.n_iter_ = n_iter

        return W

    def _fit_transform(self, X, y=None, W=None, H=None, update_H=True):
        # [FUNCTION DOC]
        # HELPER: Core solver dispatch: checks params, rejects X containing zeros when
        #   beta_loss <= 0, prepares W/H via _check_w_h, computes regularization, runs
        #   _fit_coordinate_descent or _fit_multiplicative_update, and warns when
        #   max_iter was reached with tol > 0.
        # Inputs: X validated; y ignored; W, H optional; update_H bool.
        # Outputs: tuple (W, H, n_iter).
        # Exceptions: ValueError for invalid beta_loss/solver combinations or zeros with
        #   beta_loss <= 0; errors from _check_w_h.
        # Side effects: ConvergenceWarning; sets _n_components/_beta_loss.
        # KNOWN ISSUE: the final else branch raising "Invalid solver parameter" is
        #   unreachable because parameter validation restricts solver to 'cd' or 'mu'.
        """Learn a NMF model for the data X and returns the transformed data.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Data matrix to be decomposed

        y : Ignored

        W : array-like of shape (n_samples, n_components), default=None
            If `init='custom'`, it is used as initial guess for the solution.
            If `update_H=False`, it is initialised as an array of zeros, unless
            `solver='mu'`, then it is filled with values calculated by
            `np.sqrt(X.mean() / self._n_components)`.
            If `None`, uses the initialisation method specified in `init`.

        H : array-like of shape (n_components, n_features), default=None
            If `init='custom'`, it is used as initial guess for the solution.
            If `update_H=False`, it is used as a constant, to solve for W only.
            If `None`, uses the initialisation method specified in `init`.

        update_H : bool, default=True
            If True, both W and H will be estimated from initial guesses,
            this corresponds to a call to the 'fit_transform' method.
            If False, only W will be estimated, this corresponds to a call
            to the 'transform' method.

        Returns
        -------
        W : ndarray of shape (n_samples, n_components)
            Transformed data.

        H : ndarray of shape (n_components, n_features)
            Factorization matrix, sometimes called 'dictionary'.

        n_iter_ : int
            Actual number of iterations.
        """
        # check parameters
        self._check_params(X)

        if X.min() == 0 and self._beta_loss <= 0:
            raise ValueError(
                "When beta_loss <= 0 and X contains zeros, "
                "the solver may diverge. Please add small values "
                "to X, or use a positive beta_loss."
            )

        # initialize or check W and H
        W, H = self._check_w_h(X, W, H, update_H)

        # scale the regularization terms
        l1_reg_W, l1_reg_H, l2_reg_W, l2_reg_H = self._compute_regularization(X)

        if self.solver == "cd":
            W, H, n_iter = _fit_coordinate_descent(
                X,
                W,
                H,
                self.tol,
                self.max_iter,
                l1_reg_W,
                l1_reg_H,
                l2_reg_W,
                l2_reg_H,
                update_H=update_H,
                verbose=self.verbose,
                shuffle=self.shuffle,
                random_state=self.random_state,
            )
        elif self.solver == "mu":
            W, H, n_iter, *_ = _fit_multiplicative_update(
                X,
                W,
                H,
                self._beta_loss,
                self.max_iter,
                self.tol,
                l1_reg_W,
                l1_reg_H,
                l2_reg_W,
                l2_reg_H,
                update_H,
                self.verbose,
            )
        else:
            raise ValueError("Invalid solver parameter '%s'." % self.solver)

        if n_iter == self.max_iter and self.tol > 0:
            warnings.warn(
                "Maximum number of iterations %d reached. Increase "
                "it to improve convergence." % self.max_iter,
                ConvergenceWarning,
            )

        return W, H, n_iter

    def transform(self, X):
        # [FUNCTION DOC]
        # Validates X (non-negative, same features) and solves for W with H fixed to
        # components_ via _fit_transform(update_H=False).
        # Outputs: W (n_samples, n_components). Exceptions: NotFittedError; ValueError
        #   on negative values. Side effects: may emit ConvergenceWarning.
        """Transform the data X according to the fitted NMF model.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vector, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        Returns
        -------
        W : ndarray of shape (n_samples, n_components)
            Transformed data.
        """
        check_is_fitted(self)
        X = validate_data(
            self,
            X,
            accept_sparse=("csr", "csc"),
            dtype=[np.float64, np.float32],
            reset=False,
            ensure_non_negative=True,
        )

        with config_context(assume_finite=True):
            W, *_ = self._fit_transform(X, H=self.components_, update_H=False)

        return W


class MiniBatchNMF(_BaseNMF):
    # [CLASS DOC]
    # Online NMF estimator processing mini-batches with multiplicative updates and
    # running numerator/denominator statistics for H (Lefevre et al.). Adds batch_size,
    # max_no_improvement, forget_factor, fresh_restarts, fresh_restarts_max_iter and
    # transform_max_iter; fitting also sets n_steps_ and private state
    # (_components_numerator, _components_denominator, _ewa_cost, _ewa_cost_min,
    # _no_improvement, _batch_size, _rho, _gamma, _transform_max_iter).
    """Mini-Batch Non-Negative Matrix Factorization (NMF).

    .. versionadded:: 1.1

    Find two non-negative matrices, i.e. matrices with all non-negative elements,
    (`W`, `H`) whose product approximates the non-negative matrix `X`. This
    factorization can be used for example for dimensionality reduction, source
    separation or topic extraction.

    The objective function is:

    .. math::

        L(W, H) &= 0.5 * ||X - WH||_{loss}^2

                &+ alpha\\_W * l1\\_ratio * n\\_features * ||vec(W)||_1

                &+ alpha\\_H * l1\\_ratio * n\\_samples * ||vec(H)||_1

                &+ 0.5 * alpha\\_W * (1 - l1\\_ratio) * n\\_features * ||W||_{Fro}^2

                &+ 0.5 * alpha\\_H * (1 - l1\\_ratio) * n\\_samples * ||H||_{Fro}^2,

    where :math:`||A||_{Fro}^2 = \\sum_{i,j} A_{ij}^2` (Frobenius norm) and
    :math:`||vec(A)||_1 = \\sum_{i,j} abs(A_{ij})` (Elementwise L1 norm).

    The generic norm :math:`||X - WH||_{loss}^2` may represent
    the Frobenius norm or another supported beta-divergence loss.
    The choice between options is controlled by the `beta_loss` parameter.

    The objective function is minimized with an alternating minimization of `W`
    and `H`.

    Note that the transformed data is named `W` and the components matrix is
    named `H`. In the NMF literature, the naming convention is usually the opposite
    since the data matrix `X` is transposed.

    Read more in the :ref:`User Guide <MiniBatchNMF>`.

    Parameters
    ----------
    n_components : int or {'auto'} or None, default='auto'
        Number of components. If `None`, all features are kept.
        If `n_components='auto'`, the number of components is automatically inferred
        from W or H shapes.

        .. versionchanged:: 1.4
            Added `'auto'` value.

        .. versionchanged:: 1.6
            Default value changed from `None` to `'auto'`.

    init : {'random', 'nndsvd', 'nndsvda', 'nndsvdar', 'custom'}, default=None
        Method used to initialize the procedure.
        Valid options:

        - `None`: 'nndsvda' if `n_components <= min(n_samples, n_features)`,
          otherwise random.

        - `'random'`: non-negative random matrices, scaled with:
          `sqrt(X.mean() / n_components)`

        - `'nndsvd'`: Nonnegative Double Singular Value Decomposition (NNDSVD)
          initialization (better for sparseness).

        - `'nndsvda'`: NNDSVD with zeros filled with the average of X
          (better when sparsity is not desired).

        - `'nndsvdar'` NNDSVD with zeros filled with small random values
          (generally faster, less accurate alternative to NNDSVDa
          for when sparsity is not desired).

        - `'custom'`: Use custom matrices `W` and `H` which must both be provided.

    batch_size : int, default=1024
        Number of samples in each mini-batch. Large batch sizes
        give better long-term convergence at the cost of a slower start.

    beta_loss : float or {'frobenius', 'kullback-leibler', \
            'itakura-saito'}, default='frobenius'
        Beta divergence to be minimized, measuring the distance between `X`
        and the dot product `WH`. Note that values different from 'frobenius'
        (or 2) and 'kullback-leibler' (or 1) lead to significantly slower
        fits. Note that for `beta_loss <= 0` (or 'itakura-saito'), the input
        matrix `X` cannot contain zeros.

    tol : float, default=1e-4
        Control early stopping based on the norm of the differences in `H`
        between 2 steps. To disable early stopping based on changes in `H`, set
        `tol` to 0.0.

    max_no_improvement : int, default=10
        Control early stopping based on the consecutive number of mini batches
        that does not yield an improvement on the smoothed cost function.
        To disable convergence detection based on cost function, set
        `max_no_improvement` to None.

    max_iter : int, default=200
        Maximum number of iterations over the complete dataset before
        timing out.

    alpha_W : float, default=0.0
        Constant that multiplies the regularization terms of `W`. Set it to zero
        (default) to have no regularization on `W`.

    alpha_H : float or "same", default="same"
        Constant that multiplies the regularization terms of `H`. Set it to zero to
        have no regularization on `H`. If "same" (default), it takes the same value as
        `alpha_W`.

    l1_ratio : float, default=0.0
        The regularization mixing parameter, with 0 <= l1_ratio <= 1.
        For l1_ratio = 0 the penalty is an elementwise L2 penalty
        (aka Frobenius Norm).
        For l1_ratio = 1 it is an elementwise L1 penalty.
        For 0 < l1_ratio < 1, the penalty is a combination of L1 and L2.

    forget_factor : float, default=0.7
        Amount of rescaling of past information. Its value could be 1 with
        finite datasets. Choosing values < 1 is recommended with online
        learning as more recent batches will weight more than past batches.

    fresh_restarts : bool, default=False
        Whether to completely solve for W at each step. Doing fresh restarts will likely
        lead to a better solution for a same number of iterations but it is much slower.

    fresh_restarts_max_iter : int, default=30
        Maximum number of iterations when solving for W at each step. Only used when
        doing fresh restarts. These iterations may be stopped early based on a small
        change of W controlled by `tol`.

    transform_max_iter : int, default=None
        Maximum number of iterations when solving for W at transform time.
        If None, it defaults to `max_iter`.

    random_state : int, RandomState instance or None, default=None
        Used for initialisation (when ``init`` == 'nndsvdar' or
        'random'), and in Coordinate Descent. Pass an int for reproducible
        results across multiple function calls.
        See :term:`Glossary <random_state>`.

    verbose : int, default=0
        Whether to be verbose.

    Attributes
    ----------
    components_ : ndarray of shape (n_components, n_features)
        Factorization matrix, sometimes called 'dictionary'.

    n_components_ : int
        The number of components. It is same as the `n_components` parameter
        if it was given. Otherwise, it will be same as the number of
        features.

    reconstruction_err_ : float
        Frobenius norm of the matrix difference, or beta-divergence, between
        the training data `X` and the reconstructed data `WH` from
        the fitted model.

    n_iter_ : int
        Actual number of started iterations over the whole dataset.

    n_steps_ : int
        Number of mini-batches processed.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

    See Also
    --------
    NMF : Non-negative matrix factorization.
    MiniBatchDictionaryLearning : Finds a dictionary that can best be used to represent
        data using a sparse code.

    References
    ----------
    .. [1] :doi:`"Fast local algorithms for large scale nonnegative matrix and tensor
       factorizations" <10.1587/transfun.E92.A.708>`
       Cichocki, Andrzej, and P. H. A. N. Anh-Huy. IEICE transactions on fundamentals
       of electronics, communications and computer sciences 92.3: 708-721, 2009.

    .. [2] :doi:`"Algorithms for nonnegative matrix factorization with the
       beta-divergence" <10.1162/NECO_a_00168>`
       Fevotte, C., & Idier, J. (2011). Neural Computation, 23(9).

    .. [3] :doi:`"Online algorithms for nonnegative matrix factorization with the
       Itakura-Saito divergence" <10.1109/ASPAA.2011.6082314>`
       Lefevre, A., Bach, F., Fevotte, C. (2011). WASPA.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[1, 1], [2, 1], [3, 1.2], [4, 1], [5, 0.8], [6, 1]])
    >>> from sklearn.decomposition import MiniBatchNMF
    >>> model = MiniBatchNMF(n_components=2, init='random', random_state=0)
    >>> W = model.fit_transform(X)
    >>> H = model.components_
    """

    _parameter_constraints: dict = {
        **_BaseNMF._parameter_constraints,
        "max_no_improvement": [Interval(Integral, 1, None, closed="left"), None],
        "batch_size": [Interval(Integral, 1, None, closed="left")],
        "forget_factor": [Interval(Real, 0, 1, closed="both")],
        "fresh_restarts": ["boolean"],
        "fresh_restarts_max_iter": [Interval(Integral, 1, None, closed="left")],
        "transform_max_iter": [Interval(Integral, 1, None, closed="left"), None],
    }

    def __init__(
        self,
        n_components="auto",
        *,
        init=None,
        batch_size=1024,
        beta_loss="frobenius",
        tol=1e-4,
        max_no_improvement=10,
        max_iter=200,
        alpha_W=0.0,
        alpha_H="same",
        l1_ratio=0.0,
        forget_factor=0.7,
        fresh_restarts=False,
        fresh_restarts_max_iter=30,
        transform_max_iter=None,
        random_state=None,
        verbose=0,
    ):
        # [FUNCTION DOC]
        # Passes common hyperparameters to _BaseNMF and stores the mini-batch specific
        # ones without validation.
        super().__init__(
            n_components=n_components,
            init=init,
            beta_loss=beta_loss,
            tol=tol,
            max_iter=max_iter,
            random_state=random_state,
            alpha_W=alpha_W,
            alpha_H=alpha_H,
            l1_ratio=l1_ratio,
            verbose=verbose,
        )

        self.max_no_improvement = max_no_improvement
        self.batch_size = batch_size
        self.forget_factor = forget_factor
        self.fresh_restarts = fresh_restarts
        self.fresh_restarts_max_iter = fresh_restarts_max_iter
        self.transform_max_iter = transform_max_iter

    def _check_params(self, X):
        # [FUNCTION DOC]
        # HELPER: Base checks plus derived settings: _batch_size = min(batch_size,
        #   n_samples); _rho = forget_factor raised to (_batch_size / n_samples); the MM
        #   exponent _gamma as in the batch solver; _transform_max_iter defaulting to
        #   max_iter.
        # Outputs: self. Side effects: sets the listed private attributes.
        super()._check_params(X)

        # batch_size
        self._batch_size = min(self.batch_size, X.shape[0])

        # forget_factor
        self._rho = self.forget_factor ** (self._batch_size / X.shape[0])

        # gamma for Maximization-Minimization (MM) algorithm [Fevotte 2011]
        if self._beta_loss < 1:
            self._gamma = 1.0 / (2.0 - self._beta_loss)
        elif self._beta_loss > 2:
            self._gamma = 1.0 / (self._beta_loss - 1.0)
        else:
            self._gamma = 1.0

        # transform_max_iter
        self._transform_max_iter = (
            self.max_iter
            if self.transform_max_iter is None
            else self.transform_max_iter
        )

        return self

    def _solve_W(self, X, H, max_iter):
        # [FUNCTION DOC]
        # HELPER: Solves for W with H fixed: initialises W to
        #   sqrt(mean(X)/_n_components) everywhere, then applies up to max_iter
        #   multiplicative W updates, stopping when the relative Frobenius change of W
        #   is at most tol (if tol > 0).
        # Inputs: X (n_samples, n_features); H (n_components, n_features); max_iter int.
        # Outputs: W (n_samples, n_components).
        """Minimize the objective function w.r.t W.

        Update W with H being fixed, until convergence. This is the heart
        of `transform` but it's also used during `fit` when doing fresh restarts.
        """
        avg = np.sqrt(X.mean() / self._n_components)
        W = np.full((X.shape[0], self._n_components), avg, dtype=X.dtype)
        W_buffer = W.copy()

        # Get scaled regularization terms. Done for each minibatch to take into account
        # variable sizes of minibatches.
        l1_reg_W, _, l2_reg_W, _ = self._compute_regularization(X)

        for _ in range(max_iter):
            W, *_ = _multiplicative_update_w(
                X, W, H, self._beta_loss, l1_reg_W, l2_reg_W, self._gamma
            )

            W_diff = linalg.norm(W - W_buffer) / linalg.norm(W)
            if self.tol > 0 and W_diff <= self.tol:
                break

            W_buffer[:] = W

        return W

    def _minibatch_step(self, X, W, H, update_H):
        # [FUNCTION DOC]
        # HELPER: Processes one mini-batch: computes regularization (scaled by this
        #   batch's shape), solves W from scratch when fresh_restarts or W is None, else
        #   applies one multiplicative W update (in place); zeroes tiny W entries for
        #   beta<1; computes the batch cost (beta-divergence plus L1/L2 penalties)
        #   divided by the batch size; if update_H, performs the online H update with
        #   the running statistics and rho, writing into H in place, and zeroes tiny
        #   entries for beta<=1.
        # Inputs: X batch; W batch rows (view) or None; H (n_components, n_features);
        #   update_H bool.
        # Outputs: float batch cost (computed before the H update).
        # Side effects: mutates W (view into the full W) and H in place; updates
        #   _components_numerator and _components_denominator.
        # KNOWN ISSUE: when a fresh W is solved it is not written back into the caller's
        #   W rows, so with fresh_restarts the full W only becomes consistent through
        #   the final _solve_W in _fit_transform.
        """Perform the update of W and H for one minibatch."""
        batch_size = X.shape[0]

        # get scaled regularization terms. Done for each minibatch to take into account
        # variable sizes of minibatches.
        l1_reg_W, l1_reg_H, l2_reg_W, l2_reg_H = self._compute_regularization(X)

        # update W
        if self.fresh_restarts or W is None:
            W = self._solve_W(X, H, self.fresh_restarts_max_iter)
        else:
            W, *_ = _multiplicative_update_w(
                X, W, H, self._beta_loss, l1_reg_W, l2_reg_W, self._gamma
            )

        # necessary for stability with beta_loss < 1
        if self._beta_loss < 1:
            W[W < np.finfo(np.float64).eps] = 0.0

        batch_cost = (
            _beta_divergence(X, W, H, self._beta_loss)
            + l1_reg_W * W.sum()
            + l1_reg_H * H.sum()
            + l2_reg_W * (W**2).sum()
            + l2_reg_H * (H**2).sum()
        ) / batch_size

        # update H (only at fit or fit_transform)
        if update_H:
            H[:] = _multiplicative_update_h(
                X,
                W,
                H,
                beta_loss=self._beta_loss,
                l1_reg_H=l1_reg_H,
                l2_reg_H=l2_reg_H,
                gamma=self._gamma,
                A=self._components_numerator,
                B=self._components_denominator,
                rho=self._rho,
            )

            # necessary for stability with beta_loss < 1
            if self._beta_loss <= 1:
                H[H < np.finfo(np.float64).eps] = 0.0

        return batch_cost

    def _minibatch_convergence(
        self, X, batch_cost, H, H_buffer, n_samples, step, n_steps
    ):
        # [FUNCTION DOC]
        # HELPER: Early-stopping check after a mini-batch. Skips the first step (returns
        #   False). Updates the exponentially weighted average cost with weight
        #   min(batch_size/(n_samples+1), 1); prints progress when verbose; returns True
        #   when the relative change of H versus H_buffer is at most tol (tol > 0);
        #   otherwise tracks the best EWA cost and returns True once max_no_improvement
        #   consecutive steps fail to improve it.
        # Inputs: X batch; batch_cost float; H current; H_buffer previous H; n_samples
        #   total; step zero-based index; n_steps total planned steps.
        # Outputs: bool. Side effects: updates _ewa_cost, _ewa_cost_min,
        #   _no_improvement; prints when verbose.
        """Helper function to encapsulate the early stopping logic"""
        batch_size = X.shape[0]

        # counts steps starting from 1 for user friendly verbose mode.
        step = step + 1

        # Ignore first iteration because H is not updated yet.
        if step == 1:
            if self.verbose:
                print(f"Minibatch step {step}/{n_steps}: mean batch cost: {batch_cost}")
            return False

        # Compute an Exponentially Weighted Average of the cost function to
        # monitor the convergence while discarding minibatch-local stochastic
        # variability: https://en.wikipedia.org/wiki/Moving_average
        if self._ewa_cost is None:
            self._ewa_cost = batch_cost
        else:
            alpha = batch_size / (n_samples + 1)
            alpha = min(alpha, 1)
            self._ewa_cost = self._ewa_cost * (1 - alpha) + batch_cost * alpha

        # Log progress to be able to monitor convergence
        if self.verbose:
            print(
                f"Minibatch step {step}/{n_steps}: mean batch cost: "
                f"{batch_cost}, ewa cost: {self._ewa_cost}"
            )

        # Early stopping based on change of H
        H_diff = linalg.norm(H - H_buffer) / linalg.norm(H)
        if self.tol > 0 and H_diff <= self.tol:
            if self.verbose:
                print(f"Converged (small H change) at step {step}/{n_steps}")
            return True

        # Early stopping heuristic due to lack of improvement on smoothed
        # cost function
        if self._ewa_cost_min is None or self._ewa_cost < self._ewa_cost_min:
            self._no_improvement = 0
            self._ewa_cost_min = self._ewa_cost
        else:
            self._no_improvement += 1

        if (
            self.max_no_improvement is not None
            and self._no_improvement >= self.max_no_improvement
        ):
            if self.verbose:
                print(
                    "Converged (lack of improvement in objective function) "
                    f"at step {step}/{n_steps}"
                )
            return True

        return False

    @_fit_context(prefer_skip_nested_validation=True)
    def fit_transform(self, X, y=None, W=None, H=None):
        # [FUNCTION DOC]
        # Validates X, runs _fit_transform under assume_finite, and stores
        # reconstruction_err_, n_components_, components_, n_iter_ and n_steps_.
        # Inputs: X; y ignored; W, H optional custom initialisations. Outputs: W
        #   (n_samples, n_components).
        """Learn a NMF model for the data X and returns the transformed data.

        This is more efficient than calling fit followed by transform.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Data matrix to be decomposed.

        y : Ignored
            Not used, present here for API consistency by convention.

        W : array-like of shape (n_samples, n_components), default=None
            If `init='custom'`, it is used as initial guess for the solution.
            If `None`, uses the initialisation method specified in `init`.

        H : array-like of shape (n_components, n_features), default=None
            If `init='custom'`, it is used as initial guess for the solution.
            If `None`, uses the initialisation method specified in `init`.

        Returns
        -------
        W : ndarray of shape (n_samples, n_components)
            Transformed data.
        """
        X = validate_data(
            self, X, accept_sparse=("csr", "csc"), dtype=[np.float64, np.float32]
        )

        with config_context(assume_finite=True):
            W, H, n_iter, n_steps = self._fit_transform(X, W=W, H=H)

        self.reconstruction_err_ = _beta_divergence(
            X, W, H, self._beta_loss, square_root=True
        )

        self.n_components_ = H.shape[0]
        self.components_ = H
        self.n_iter_ = n_iter
        self.n_steps_ = n_steps

        return W

    def _fit_transform(self, X, W=None, H=None, update_H=True):
        # [FUNCTION DOC]
        # HELPER: Mini-batch fitting loop: checks non-negativity and params, rejects
        #   zeros in X when beta_loss <= 0, prepares W/H, initialises running statistics
        #   (numerator copy of H, denominator ones) and EWA state, then cycles over
        #   batches for up to max_iter epochs' worth of steps, calling _minibatch_step
        #   and (when update_H) stopping early via _minibatch_convergence; finally re-
        #   solves W fully when fresh_restarts and computes the epoch count as
        #   ceil(steps / steps per epoch).
        # Inputs: X; W, H optional; update_H bool.
        # Outputs: tuple (W, H, n_iter, n_steps).
        # Exceptions: ValueError for negative X or zeros with beta_loss <= 0; errors
        #   from _check_w_h.
        # Side effects: sets private online state; ConvergenceWarning when n_iter
        #   reaches max_iter with tol > 0; prints when verbose.
        """Learn a NMF model for the data X and returns the transformed data.

        Parameters
        ----------
        X : {ndarray, sparse matrix} of shape (n_samples, n_features)
            Data matrix to be decomposed.

        W : array-like of shape (n_samples, n_components), default=None
            If `init='custom'`, it is used as initial guess for the solution.
            If `update_H=False`, it is initialised as an array of zeros, unless
            `solver='mu'`, then it is filled with values calculated by
            `np.sqrt(X.mean() / self._n_components)`.
            If `None`, uses the initialisation method specified in `init`.

        H : array-like of shape (n_components, n_features), default=None
            If `init='custom'`, it is used as initial guess for the solution.
            If `update_H=False`, it is used as a constant, to solve for W only.
            If `None`, uses the initialisation method specified in `init`.

        update_H : bool, default=True
            If True, both W and H will be estimated from initial guesses,
            this corresponds to a call to the `fit_transform` method.
            If False, only W will be estimated, this corresponds to a call
            to the `transform` method.

        Returns
        -------
        W : ndarray of shape (n_samples, n_components)
            Transformed data.

        H : ndarray of shape (n_components, n_features)
            Factorization matrix, sometimes called 'dictionary'.

        n_iter : int
            Actual number of started iterations over the whole dataset.

        n_steps : int
            Number of mini-batches processed.
        """
        check_non_negative(X, "MiniBatchNMF (input X)")
        self._check_params(X)

        if X.min() == 0 and self._beta_loss <= 0:
            raise ValueError(
                "When beta_loss <= 0 and X contains zeros, "
                "the solver may diverge. Please add small values "
                "to X, or use a positive beta_loss."
            )

        n_samples = X.shape[0]

        # initialize or check W and H
        W, H = self._check_w_h(X, W, H, update_H)
        H_buffer = H.copy()

        # Initialize auxiliary matrices
        self._components_numerator = H.copy()
        self._components_denominator = np.ones(H.shape, dtype=H.dtype)

        # Attributes to monitor the convergence
        self._ewa_cost = None
        self._ewa_cost_min = None
        self._no_improvement = 0

        batches = gen_batches(n_samples, self._batch_size)
        batches = itertools.cycle(batches)
        n_steps_per_iter = int(np.ceil(n_samples / self._batch_size))
        n_steps = self.max_iter * n_steps_per_iter

        for i, batch in zip(range(n_steps), batches):
            batch_cost = self._minibatch_step(X[batch], W[batch], H, update_H)

            if update_H and self._minibatch_convergence(
                X[batch], batch_cost, H, H_buffer, n_samples, i, n_steps
            ):
                break

            H_buffer[:] = H

        if self.fresh_restarts:
            W = self._solve_W(X, H, self._transform_max_iter)

        n_steps = i + 1
        n_iter = int(np.ceil(n_steps / n_steps_per_iter))

        if n_iter == self.max_iter and self.tol > 0:
            warnings.warn(
                (
                    f"Maximum number of iterations {self.max_iter} reached. "
                    "Increase it to improve convergence."
                ),
                ConvergenceWarning,
            )

        return W, H, n_iter, n_steps

    def transform(self, X):
        # [FUNCTION DOC]
        # Validates X and solves for W with H fixed to components_ via _solve_W with
        # _transform_max_iter iterations.
        # Outputs: W (n_samples, n_components). Exceptions: NotFittedError.
        # KNOWN ISSUE: unlike NMF.transform, X is not checked for non-negativity.
        """Transform the data X according to the fitted MiniBatchNMF model.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Data matrix to be transformed by the model.

        Returns
        -------
        W : ndarray of shape (n_samples, n_components)
            Transformed data.
        """
        check_is_fitted(self)
        X = validate_data(
            self,
            X,
            accept_sparse=("csr", "csc"),
            dtype=[np.float64, np.float32],
            reset=False,
        )

        W = self._solve_W(X, self.components_, self._transform_max_iter)

        return W

    @_fit_context(prefer_skip_nested_validation=True)
    def partial_fit(self, X, y=None, W=None, H=None):
        # [FUNCTION DOC]
        # Online update with one chunk. On the first call validates X (resetting feature
        # info), checks params, initialises H (via _check_w_h; W is discarded) and the
        # running statistics and sets n_steps_ to 0; on later calls reuses components_.
        # Then runs one _minibatch_step with a freshly solved W, stores components_ and
        # n_components_, and increments n_steps_.
        # Inputs: X (n_samples, n_features); y ignored; W, H used only on the first call
        #   with init='custom'.
        # Outputs: self.
        # Side effects: updates components_ in place, running statistics, n_steps_.
        # KNOWN ISSUE: derived settings (_batch_size, _rho) are computed from the first
        #   chunk only, and non-negativity of X is only checked on the first call
        #   (inside initialisation).
        """Update the model using the data in `X` as a mini-batch.

        This method is expected to be called several times consecutively
        on different chunks of a dataset so as to implement out-of-core
        or online learning.

        This is especially useful when the whole dataset is too big to fit in
        memory at once (see :ref:`scaling_strategies`).

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Data matrix to be decomposed.

        y : Ignored
            Not used, present here for API consistency by convention.

        W : array-like of shape (n_samples, n_components), default=None
            If `init='custom'`, it is used as initial guess for the solution.
            Only used for the first call to `partial_fit`.

        H : array-like of shape (n_components, n_features), default=None
            If `init='custom'`, it is used as initial guess for the solution.
            Only used for the first call to `partial_fit`.

        Returns
        -------
        self
            Returns the instance itself.
        """
        has_components = hasattr(self, "components_")

        X = validate_data(
            self,
            X,
            accept_sparse=("csr", "csc"),
            dtype=[np.float64, np.float32],
            reset=not has_components,
        )

        if not has_components:
            # This instance has not been fitted yet (fit or partial_fit)
            self._check_params(X)
            _, H = self._check_w_h(X, W=W, H=H, update_H=True)

            self._components_numerator = H.copy()
            self._components_denominator = np.ones(H.shape, dtype=H.dtype)
            self.n_steps_ = 0
        else:
            H = self.components_

        self._minibatch_step(X, None, H, update_H=True)

        self.n_components_ = H.shape[0]
        self.components_ = H
        self.n_steps_ += 1

        return self
