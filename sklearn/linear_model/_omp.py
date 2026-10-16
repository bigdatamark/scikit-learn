# [MODULE DOC]
# Orthogonal Matching Pursuit (OMP): greedy sparse approximation that iteratively
# selects the feature (atom) most correlated with the current residual and refits the
# least-squares solution on the selected atoms using an incrementally updated Cholesky
# factor. Provides low-level solvers on X or on the Gram matrix, the public functions
# orthogonal_mp / orthogonal_mp_gram, the OrthogonalMatchingPursuit estimator and its
# cross-validated variant. Defines the module string premature used in RuntimeWarnings.
"""Orthogonal matching pursuit algorithms"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import warnings
from math import sqrt
from numbers import Integral, Real

import numpy as np
from scipy import linalg
from scipy.linalg.lapack import get_lapack_funcs

from sklearn.base import RegressorMixin, _fit_context
from sklearn.linear_model._base import LinearModel, MultiOutputLinearModel, _pre_fit
from sklearn.model_selection import check_cv
from sklearn.utils import as_float_array, check_array
from sklearn.utils._param_validation import Interval, StrOptions, validate_params
from sklearn.utils.metadata_routing import (
    MetadataRouter,
    MethodMapping,
    _manual_routing,
    _raise_for_params,
    _routing_enabled,
    process_routing,
)
from sklearn.utils.parallel import Parallel, delayed
from sklearn.utils.validation import FLOAT_DTYPES, validate_data

premature = (
    "Orthogonal matching pursuit ended prematurely due to linear"
    " dependence in the dictionary. The requested precision might"
    " not have been met."
)


def _cholesky_omp(X, y, n_nonzero_coefs, tol=None, copy_X=True, return_path=False):
    # [FUNCTION DOC]
    # HELPER: OMP for a single target working directly on X.
    # Steps: copy X to Fortran order (or just ensure Fortran order when copy_X is False);
    # alpha = X'y; loop: pick the column with largest absolute correlation with the
    # residual; stop with a RuntimeWarning if it is already active or its initial
    # correlation alpha squared is below machine eps; extend the lower Cholesky factor L of
    # the active Gram matrix by solving a triangular system (stop with warning if the new
    # diagonal is <= eps, i.e. linear dependence); swap the chosen column into the active
    # block of X (and alpha, indices); solve L L' gamma = alpha_active with potrs; record
    # gamma in the path; recompute the residual y - X_active gamma. Stop when the squared
    # residual norm <= tol (tol mode) or when n_nonzero_coefs atoms are active.
    # Inputs: X (n_samples, n_features); y (n_samples,); n_nonzero_coefs int (ignored when
    # tol is set); tol float or None (squared residual norm target); copy_X bool;
    # return_path bool.
    # Outputs: gamma (n_active,) coefficients for the selected atoms, indices of selected
    # atoms (n_active,), optional path coefs (max_features, n_active) whose column j holds
    # the solution with j+1 atoms, and n_active.
    # Side effects: when copy_X is False and X is already Fortran ordered, X's columns are
    # permuted in place. Emits RuntimeWarning on premature termination.
    # KNOWN ISSUE: the "inner product too small" check uses alpha (the initial X'y,
    # permuted) rather than the current residual correlation, so it tests the wrong
    # quantity.
    """Orthogonal Matching Pursuit step using the Cholesky decomposition.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Input dictionary. Columns are assumed to have unit norm.

    y : ndarray of shape (n_samples,)
        Input targets.

    n_nonzero_coefs : int
        Targeted number of non-zero elements.

    tol : float, default=None
        Targeted squared error, if not None overrides n_nonzero_coefs.

    copy_X : bool, default=True
        Whether the design matrix X must be copied by the algorithm. A false
        value is only helpful if X is already Fortran-ordered, otherwise a
        copy is made anyway.

    return_path : bool, default=False
        Whether to return every value of the nonzero coefficients along the
        forward path. Useful for cross-validation.

    Returns
    -------
    gamma : ndarray of shape (n_nonzero_coefs,)
        Non-zero elements of the solution.

    idx : ndarray of shape (n_nonzero_coefs,)
        Indices of the positions of the elements in gamma within the solution
        vector.

    coef : ndarray of shape (n_features, n_nonzero_coefs)
        The first k values of column k correspond to the coefficient value
        for the active features at that step. The lower left triangle contains
        garbage. Only returned if ``return_path=True``.

    n_active : int
        Number of active features at convergence.
    """
    if copy_X:
        X = X.copy("F")
    else:  # even if we are allowed to overwrite, still copy it if bad order
        X = np.asfortranarray(X)

    min_float = np.finfo(X.dtype).eps
    nrm2, swap = linalg.get_blas_funcs(("nrm2", "swap"), (X,))
    (potrs,) = get_lapack_funcs(("potrs",), (X,))

    alpha = np.dot(X.T, y)
    residual = y
    gamma = np.empty(0)
    n_active = 0
    indices = np.arange(X.shape[1])  # keeping track of swapping

    max_features = X.shape[1] if tol is not None else n_nonzero_coefs

    L = np.empty((max_features, max_features), dtype=X.dtype)

    if return_path:
        coefs = np.empty_like(L)

    while True:
        lam = np.argmax(np.abs(np.dot(X.T, residual)))
        if lam < n_active or alpha[lam] ** 2 < min_float:
            # atom already selected or inner product too small
            warnings.warn(premature, RuntimeWarning, stacklevel=2)
            break

        if n_active > 0:
            # Updates the Cholesky decomposition of X' X
            L[n_active, :n_active] = np.dot(X[:, :n_active].T, X[:, lam])
            linalg.solve_triangular(
                L[:n_active, :n_active],
                L[n_active, :n_active],
                trans=0,
                lower=1,
                overwrite_b=True,
                check_finite=False,
            )
            v = nrm2(L[n_active, :n_active]) ** 2
            Lkk = linalg.norm(X[:, lam]) ** 2 - v
            if Lkk <= min_float:  # selected atoms are dependent
                warnings.warn(premature, RuntimeWarning, stacklevel=2)
                break
            L[n_active, n_active] = sqrt(Lkk)
        else:
            L[0, 0] = linalg.norm(X[:, lam])

        X.T[n_active], X.T[lam] = swap(X.T[n_active], X.T[lam])
        alpha[n_active], alpha[lam] = alpha[lam], alpha[n_active]
        indices[n_active], indices[lam] = indices[lam], indices[n_active]
        n_active += 1

        # solves LL'x = X'y as a composition of two triangular systems
        gamma, _ = potrs(
            L[:n_active, :n_active], alpha[:n_active], lower=True, overwrite_b=False
        )

        if return_path:
            coefs[:n_active, n_active - 1] = gamma
        residual = y - np.dot(X[:, :n_active], gamma)
        if tol is not None and nrm2(residual) ** 2 <= tol:
            break
        elif n_active == max_features:
            break

    if return_path:
        return gamma, indices[:n_active], coefs[:, :n_active], n_active
    else:
        return gamma, indices[:n_active], n_active


def _gram_omp(
    Gram,
    Xy,
    n_nonzero_coefs,
    tol_0=None,
    tol=None,
    copy_Gram=True,
    copy_Xy=True,
    return_path=False,
):
    # [FUNCTION DOC]
    # HELPER: OMP for a single target using only the Gram matrix X'X and Xy = X'y.
    # Same greedy loop as _cholesky_omp but the correlation vector alpha is maintained as
    # Xy minus Gram times the current solution, the Cholesky row comes from Gram, and
    # rows and columns of Gram are swapped to keep active atoms first. In tol mode the
    # residual squared norm is tracked incrementally starting from tol_0 (the squared norm
    # of y): add back the previous delta, compute the new delta as gamma dotted with
    # (Gram gamma) restricted to the active set, subtract it, and stop when its absolute
    # value <= tol.
    # Inputs: Gram (n_features, n_features); Xy (n_features,); n_nonzero_coefs; tol_0 -
    # squared norm of y (needed with tol); tol; copy_Gram; copy_Xy; return_path.
    # Outputs: same tuple as _cholesky_omp.
    # Side effects: permutes Gram in place when copy_Gram is False and it is Fortran
    # ordered; Xy is copied if requested or read-only, otherwise permuted in place.
    # RuntimeWarning on premature termination.
    # KNOWN ISSUE: L[0, 0] is initialised to 1.0 before the loop but always overwritten
    # on the first iteration (dead store).
    """Orthogonal Matching Pursuit step on a precomputed Gram matrix.

    This function uses the Cholesky decomposition method.

    Parameters
    ----------
    Gram : ndarray of shape (n_features, n_features)
        Gram matrix of the input data matrix.

    Xy : ndarray of shape (n_features,)
        Input targets.

    n_nonzero_coefs : int
        Targeted number of non-zero elements.

    tol_0 : float, default=None
        Squared norm of y, required if tol is not None.

    tol : float, default=None
        Targeted squared error, if not None overrides n_nonzero_coefs.

    copy_Gram : bool, default=True
        Whether the gram matrix must be copied by the algorithm. A false
        value is only helpful if it is already Fortran-ordered, otherwise a
        copy is made anyway.

    copy_Xy : bool, default=True
        Whether the covariance vector Xy must be copied by the algorithm.
        If False, it may be overwritten.

    return_path : bool, default=False
        Whether to return every value of the nonzero coefficients along the
        forward path. Useful for cross-validation.

    Returns
    -------
    gamma : ndarray of shape (n_nonzero_coefs,)
        Non-zero elements of the solution.

    idx : ndarray of shape (n_nonzero_coefs,)
        Indices of the positions of the elements in gamma within the solution
        vector.

    coefs : ndarray of shape (n_features, n_nonzero_coefs)
        The first k values of column k correspond to the coefficient value
        for the active features at that step. The lower left triangle contains
        garbage. Only returned if ``return_path=True``.

    n_active : int
        Number of active features at convergence.
    """
    Gram = Gram.copy("F") if copy_Gram else np.asfortranarray(Gram)

    if copy_Xy or not Xy.flags.writeable:
        Xy = Xy.copy()

    min_float = np.finfo(Gram.dtype).eps
    nrm2, swap = linalg.get_blas_funcs(("nrm2", "swap"), (Gram,))
    (potrs,) = get_lapack_funcs(("potrs",), (Gram,))

    indices = np.arange(len(Gram))  # keeping track of swapping
    alpha = Xy
    tol_curr = tol_0
    delta = 0
    gamma = np.empty(0)
    n_active = 0

    max_features = len(Gram) if tol is not None else n_nonzero_coefs

    L = np.empty((max_features, max_features), dtype=Gram.dtype)

    L[0, 0] = 1.0
    if return_path:
        coefs = np.empty_like(L)

    while True:
        lam = np.argmax(np.abs(alpha))
        if lam < n_active or alpha[lam] ** 2 < min_float:
            # selected same atom twice, or inner product too small
            warnings.warn(premature, RuntimeWarning, stacklevel=3)
            break
        if n_active > 0:
            L[n_active, :n_active] = Gram[lam, :n_active]
            linalg.solve_triangular(
                L[:n_active, :n_active],
                L[n_active, :n_active],
                trans=0,
                lower=1,
                overwrite_b=True,
                check_finite=False,
            )
            v = nrm2(L[n_active, :n_active]) ** 2
            Lkk = Gram[lam, lam] - v
            if Lkk <= min_float:  # selected atoms are dependent
                warnings.warn(premature, RuntimeWarning, stacklevel=3)
                break
            L[n_active, n_active] = sqrt(Lkk)
        else:
            L[0, 0] = sqrt(Gram[lam, lam])

        Gram[n_active], Gram[lam] = swap(Gram[n_active], Gram[lam])
        Gram.T[n_active], Gram.T[lam] = swap(Gram.T[n_active], Gram.T[lam])
        indices[n_active], indices[lam] = indices[lam], indices[n_active]
        Xy[n_active], Xy[lam] = Xy[lam], Xy[n_active]
        n_active += 1
        # solves LL'x = X'y as a composition of two triangular systems
        gamma, _ = potrs(
            L[:n_active, :n_active], Xy[:n_active], lower=True, overwrite_b=False
        )
        if return_path:
            coefs[:n_active, n_active - 1] = gamma
        beta = np.dot(Gram[:, :n_active], gamma)
        alpha = Xy - beta
        if tol is not None:
            tol_curr += delta
            delta = np.inner(gamma, beta[:n_active])
            tol_curr -= delta
            if abs(tol_curr) <= tol:
                break
        elif n_active == max_features:
            break

    if return_path:
        return gamma, indices[:n_active], coefs[:, :n_active], n_active
    else:
        return gamma, indices[:n_active], n_active


@validate_params(
    {
        "X": ["array-like"],
        "y": [np.ndarray],
        "n_nonzero_coefs": [Interval(Integral, 1, None, closed="left"), None],
        "tol": [Interval(Real, 0, None, closed="left"), None],
        "precompute": ["boolean", StrOptions({"auto"})],
        "copy_X": ["boolean"],
        "return_path": ["boolean"],
        "return_n_iter": ["boolean"],
    },
    prefer_skip_nested_validation=True,
)
def orthogonal_mp(
    X,
    y,
    *,
    n_nonzero_coefs=None,
    tol=None,
    precompute=False,
    copy_X=True,
    return_path=False,
    return_n_iter=False,
):
    # [FUNCTION DOC]
    # Public OMP solver on X for one or several targets.
    # Validates X (Fortran, optionally copied) and y (reshaped to 2-D). Forces copying of
    # X for each target when there are multiple targets since the low-level solver permutes
    # columns. Defaults n_nonzero_coefs to max(10% of n_features, 1) when neither it nor tol
    # is given. precompute="auto" means use the Gram path when n_samples > n_features; the
    # Gram path computes X'X, X'y and (in tol mode) the squared norms of y and delegates to
    # orthogonal_mp_gram. Otherwise runs _cholesky_omp per target and scatters each
    # solution (or each path step) into a dense coefficient array.
    # Inputs: X (n_samples, n_features); y ndarray (n_samples,) or (n_samples, n_targets);
    # n_nonzero_coefs; tol; precompute bool or "auto"; copy_X; return_path;
    # return_n_iter.
    # Outputs: coef squeezed from (n_features, n_targets) or, with return_path,
    # (n_features, n_targets, n_steps) where n_steps is the number of atoms of the last
    # target processed; optionally n_iter (int for one target else list).
    # Exceptions: ValueError if n_nonzero_coefs > n_features without tol; parameter
    # validation errors.
    # KNOWN ISSUE: return_n_iter is ignored on the precompute path (the Gram call does
    # not forward it), so the return type changes with precompute.
    # KNOWN ISSUE: with return_path and several targets, the path array is truncated to
    # the number of steps of each target in turn, losing longer paths of earlier targets.
    r"""Orthogonal Matching Pursuit (OMP).

    Solves `n_targets` Orthogonal Matching Pursuit problems.
    Each instance of the problem fits a linear model with constraints
    imposed on the number of non-zero coefficients
    (i.e. the :math:`\\ell_0` pseudo-norm) using `n_nonzero_coefs`:

    .. math::
        \underset{w}{\operatorname{arg\,min\,}} ||y - Xw||^2_2
            \text{ subject to } ||w||_0 \leq n_{\text{nonzero_coefs}}

    When parametrized by error using the parameter `tol`:

    .. math::
        \underset{w}{\operatorname{arg\,min\,}} ||w||_0
            \text{ subject to } ||y - Xw||^2_2 \leq tol

    Read more in the :ref:`User Guide <omp>`.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Input data. Columns are assumed to have unit norm.

    y : ndarray of shape (n_samples,) or (n_samples, n_targets)
        Input targets.

    n_nonzero_coefs : int, default=None
        Maximum number of non-zero entries in the solution. If None (by
        default) this value is set to 10% of n_features.

    tol : float, default=None
        Maximum squared norm of the residual. If not None, overrides n_nonzero_coefs.

    precompute : 'auto' or bool, default=False
        Whether to perform precomputations. Improves performance when n_targets
        or n_samples is very large.

    copy_X : bool, default=True
        Whether the design matrix X must be copied by the algorithm. A false
        value is only helpful if X is already Fortran-ordered, otherwise a
        copy is made anyway.

    return_path : bool, default=False
        Whether to return every value of the nonzero coefficients along the
        forward path. Useful for cross-validation.

    return_n_iter : bool, default=False
        Whether or not to return the number of iterations.

    Returns
    -------
    coef : ndarray of shape (n_features,) or (n_features, n_targets)
        Coefficients of the OMP solution. If `return_path=True`, this contains
        the whole coefficient path. In this case its shape is
        (n_features, n_features) or (n_features, n_targets, n_features) and
        iterating over the last axis generates coefficients in increasing order
        of active features.

    n_iters : array-like or int
        Number of active features across every target. Returned only if
        `return_n_iter` is set to True.

    See Also
    --------
    OrthogonalMatchingPursuit : Orthogonal Matching Pursuit model.
    orthogonal_mp_gram : Solve OMP problems using Gram matrix and the product X.T * y.
    lars_path : Compute Least Angle Regression or Lasso path using LARS algorithm.
    sklearn.decomposition.sparse_encode : Sparse coding.

    Notes
    -----
    Orthogonal matching pursuit was introduced in S. Mallat, Z. Zhang,
    Matching pursuits with time-frequency dictionaries, IEEE Transactions on
    Signal Processing, Vol. 41, No. 12. (December 1993), pp. 3397-3415.
    (https://www.di.ens.fr/~mallat/papiers/MallatPursuit93.pdf)

    This implementation is based on Rubinstein, R., Zibulevsky, M. and Elad,
    M., Efficient Implementation of the K-SVD Algorithm using Batch Orthogonal
    Matching Pursuit Technical Report - CS Technion, April 2008.
    https://www.cs.technion.ac.il/~ronrubin/Publications/KSVD-OMP-v2.pdf

    Examples
    --------
    >>> from sklearn.datasets import make_regression
    >>> from sklearn.linear_model import orthogonal_mp
    >>> X, y = make_regression(noise=4, random_state=0)
    >>> coef = orthogonal_mp(X, y)
    >>> coef.shape
    (100,)
    >>> X[:1,] @ coef
    array([-78.68])
    """
    X = check_array(X, order="F", copy=copy_X)
    copy_X = False
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    y = check_array(y)
    if y.shape[1] > 1:  # subsequent targets will be affected
        copy_X = True
    if n_nonzero_coefs is None and tol is None:
        # default for n_nonzero_coefs is 0.1 * n_features
        # but at least one.
        n_nonzero_coefs = max(int(0.1 * X.shape[1]), 1)
    if tol is None and n_nonzero_coefs > X.shape[1]:
        raise ValueError(
            "The number of atoms cannot be more than the number of features"
        )
    if precompute == "auto":
        precompute = X.shape[0] > X.shape[1]
    if precompute:
        G = np.dot(X.T, X)
        G = np.asfortranarray(G)
        Xy = np.dot(X.T, y)
        if tol is not None:
            norms_squared = np.sum((y**2), axis=0)
        else:
            norms_squared = None
        return orthogonal_mp_gram(
            G,
            Xy,
            n_nonzero_coefs=n_nonzero_coefs,
            tol=tol,
            norms_squared=norms_squared,
            copy_Gram=copy_X,
            copy_Xy=False,
            return_path=return_path,
        )

    if return_path:
        coef = np.zeros((X.shape[1], y.shape[1], X.shape[1]))
    else:
        coef = np.zeros((X.shape[1], y.shape[1]))
    n_iters = []

    for k in range(y.shape[1]):
        out = _cholesky_omp(
            X, y[:, k], n_nonzero_coefs, tol, copy_X=copy_X, return_path=return_path
        )
        if return_path:
            _, idx, coefs, n_iter = out
            coef = coef[:, :, : len(idx)]
            for n_active, x in enumerate(coefs.T):
                coef[idx[: n_active + 1], k, n_active] = x[: n_active + 1]
        else:
            x, idx, n_iter = out
            coef[idx, k] = x
        n_iters.append(n_iter)

    if y.shape[1] == 1:
        n_iters = n_iters[0]

    if return_n_iter:
        return np.squeeze(coef), n_iters
    else:
        return np.squeeze(coef)


@validate_params(
    {
        "Gram": ["array-like"],
        "Xy": ["array-like"],
        "n_nonzero_coefs": [Interval(Integral, 0, None, closed="neither"), None],
        "tol": [Interval(Real, 0, None, closed="left"), None],
        "norms_squared": ["array-like", None],
        "copy_Gram": ["boolean"],
        "copy_Xy": ["boolean"],
        "return_path": ["boolean"],
        "return_n_iter": ["boolean"],
    },
    prefer_skip_nested_validation=True,
)
def orthogonal_mp_gram(
    Gram,
    Xy,
    *,
    n_nonzero_coefs=None,
    tol=None,
    norms_squared=None,
    copy_Gram=True,
    copy_Xy=True,
    return_path=False,
    return_n_iter=False,
):
    # [FUNCTION DOC]
    # Public OMP solver from the Gram matrix and X'y, for one or several targets.
    # Validates/copies Gram (Fortran), makes Gram copies per target when there are
    # several targets, wraps 1-D Xy (and norms_squared) into 2-D/list form, copies Xy once
    # if requested or read-only. Defaults n_nonzero_coefs to 10% of n_features. Then runs
    # _gram_omp per target and assembles coefficients like orthogonal_mp.
    # Inputs: Gram (n_features, n_features); Xy (n_features,) or (n_features, n_targets);
    # n_nonzero_coefs; tol; norms_squared (n_targets,) squared norms of y, required with
    # tol; copy_Gram; copy_Xy; return_path; return_n_iter.
    # Outputs: as orthogonal_mp.
    # Exceptions: ValueError if tol is given without norms_squared, if the number of atoms
    # is non-positive or exceeds n_features.
    # KNOWN ISSUE: the default int(0.1 * n_features) can be 0 for fewer than 10 features,
    # which then raises "number of atoms must be positive", unlike orthogonal_mp which
    # uses at least 1.
    # KNOWN ISSUE: the "tol < 0" check is unreachable because validate_params already
    # enforces tol >= 0.
    """Gram Orthogonal Matching Pursuit (OMP).

    Solves `n_targets` Orthogonal Matching Pursuit problems using only
    the precomputed Gram matrix ``X.T * X`` and the product ``X.T * y``.
    This implementation efficiently computes the solution using the
    Cholesky decomposition.

    Each instance of the problem fits a linear model with constraints
    imposed on the number of non-zero coefficients (i.e. the
    :math:`\\ell_0` pseudo-norm) using `n_nonzero_coefs` or
    on the error using the parameter `tol`.

    Read more in the :ref:`User Guide <omp>`.

    Parameters
    ----------
    Gram : array-like of shape (n_features, n_features)
        Gram matrix of the input data: `X.T * X`.

    Xy : array-like of shape (n_features,) or (n_features, n_targets)
        Input targets multiplied by `X`: `X.T * y`.

    n_nonzero_coefs : int, default=None
        Maximum number of non-zero entries in the solution. If `None` (by
        default) this value is set to 10% of n_features.

    tol : float, default=None
        Maximum squared norm of the residual. If not `None`,
        overrides `n_nonzero_coefs`.

    norms_squared : array-like of shape (n_targets,), default=None
        Squared L2 norms of the lines of `y`. Required if `tol` is not None.

    copy_Gram : bool, default=True
        Whether the gram matrix must be copied by the algorithm. A `False`
        value is only helpful if it is already Fortran-ordered, otherwise a
        copy is made anyway.

    copy_Xy : bool, default=True
        Whether the covariance vector `Xy` must be copied by the algorithm.
        If `False`, it may be overwritten.

    return_path : bool, default=False
        Whether to return every value of the nonzero coefficients along the
        forward path. Useful for cross-validation.

    return_n_iter : bool, default=False
        Whether or not to return the number of iterations.

    Returns
    -------
    coef : ndarray of shape (n_features,) or (n_features, n_targets)
        Coefficients of the OMP solution. If `return_path=True`, this contains
        the whole coefficient path. In this case its shape is
        `(n_features, n_features)` or `(n_features, n_targets, n_features)` and
        iterating over the last axis yields coefficients in increasing order
        of active features.

    n_iters : list or int
        Number of active features across every target. Returned only if
        `return_n_iter` is set to True.

    See Also
    --------
    OrthogonalMatchingPursuit : Orthogonal Matching Pursuit model (OMP).
    orthogonal_mp : Solves n_targets Orthogonal Matching Pursuit problems.
    lars_path : Compute Least Angle Regression or Lasso path using
        LARS algorithm.
    sklearn.decomposition.sparse_encode : Generic sparse coding.
        Each column of the result is the solution to a Lasso problem.

    Notes
    -----
    Orthogonal matching pursuit was introduced in G. Mallat, Z. Zhang,
    Matching pursuits with time-frequency dictionaries, IEEE Transactions on
    Signal Processing, Vol. 41, No. 12. (December 1993), pp. 3397-3415.
    (https://www.di.ens.fr/~mallat/papiers/MallatPursuit93.pdf)

    This implementation is based on Rubinstein, R., Zibulevsky, M. and Elad,
    M., Efficient Implementation of the K-SVD Algorithm using Batch Orthogonal
    Matching Pursuit Technical Report - CS Technion, April 2008.
    https://www.cs.technion.ac.il/~ronrubin/Publications/KSVD-OMP-v2.pdf

    Examples
    --------
    >>> from sklearn.datasets import make_regression
    >>> from sklearn.linear_model import orthogonal_mp_gram
    >>> X, y = make_regression(noise=4, random_state=0)
    >>> coef = orthogonal_mp_gram(X.T @ X, X.T @ y)
    >>> coef.shape
    (100,)
    >>> X[:1,] @ coef
    array([-78.68])
    """
    Gram = check_array(Gram, order="F", copy=copy_Gram)
    Xy = np.asarray(Xy)
    if Xy.ndim > 1 and Xy.shape[1] > 1:
        # or subsequent target will be affected
        copy_Gram = True
    if Xy.ndim == 1:
        Xy = Xy[:, np.newaxis]
        if tol is not None:
            norms_squared = [norms_squared]
    if copy_Xy or not Xy.flags.writeable:
        # Make the copy once instead of many times in _gram_omp itself.
        Xy = Xy.copy()

    if n_nonzero_coefs is None and tol is None:
        n_nonzero_coefs = int(0.1 * len(Gram))
    if tol is not None and norms_squared is None:
        raise ValueError(
            "Gram OMP needs the precomputed norms in order "
            "to evaluate the error sum of squares."
        )
    if tol is not None and tol < 0:
        raise ValueError("Epsilon cannot be negative")
    if tol is None and n_nonzero_coefs <= 0:
        raise ValueError("The number of atoms must be positive")
    if tol is None and n_nonzero_coefs > len(Gram):
        raise ValueError(
            "The number of atoms cannot be more than the number of features"
        )

    if return_path:
        coef = np.zeros((len(Gram), Xy.shape[1], len(Gram)), dtype=Gram.dtype)
    else:
        coef = np.zeros((len(Gram), Xy.shape[1]), dtype=Gram.dtype)

    n_iters = []
    for k in range(Xy.shape[1]):
        out = _gram_omp(
            Gram,
            Xy[:, k],
            n_nonzero_coefs,
            norms_squared[k] if tol is not None else None,
            tol,
            copy_Gram=copy_Gram,
            copy_Xy=False,
            return_path=return_path,
        )
        if return_path:
            _, idx, coefs, n_iter = out
            coef = coef[:, :, : len(idx)]
            for n_active, x in enumerate(coefs.T):
                coef[idx[: n_active + 1], k, n_active] = x[: n_active + 1]
        else:
            x, idx, n_iter = out
            coef[idx, k] = x
        n_iters.append(n_iter)

    if Xy.shape[1] == 1:
        n_iters = n_iters[0]

    if return_n_iter:
        return np.squeeze(coef), n_iters
    else:
        return np.squeeze(coef)


class OrthogonalMatchingPursuit(RegressorMixin, MultiOutputLinearModel):
    # [CLASS DOC]
    # OMP linear regression estimator (multi-output capable) stopping either after
    # n_nonzero_coefs atoms or when the residual squared norm reaches tol.
    """Orthogonal Matching Pursuit model (OMP).

    Fits a linear model with constraints imposed on the number of non-zero coefficients
    (i.e. the :math:`\\ell_0` pseudo-norm) using `n_nonzero_coefs`:

    .. math::
        \\underset{w}{\\operatorname{arg\\,min\\,}} ||y - Xw||^2_2
            \\text{ subject to } ||w||_0 \\leq n_{\\text{nonzero_coefs}}

    When parametrized by error using the parameter `tol`:

    .. math::
        \\underset{w}{\\operatorname{arg\\,min\\,}} ||w||_0
            \\text{ subject to } ||y - Xw||^2_2 \\leq tol

    Read more in the :ref:`User Guide <omp>`.

    Parameters
    ----------
    n_nonzero_coefs : int, default=None
        Maximum number of non-zero entries in the solution. Ignored if `tol` is set.
        When `None` and `tol` is also `None`, this value is either set to 10% of
        `n_features` or 1, whichever is greater.

    tol : float, default=None
        Maximum squared norm of the residual. If not None, overrides n_nonzero_coefs.

    fit_intercept : bool, default=True
        Whether to calculate the intercept for this model. If set
        to false, no intercept will be used in calculations
        (i.e. data is expected to be centered).

    precompute : 'auto' or bool, default='auto'
        Whether to use a precomputed Gram and Xy matrix to speed up
        calculations. Improves performance when :term:`n_targets` or
        :term:`n_samples` is very large.

    Attributes
    ----------
    coef_ : ndarray of shape (n_features,) or (n_targets, n_features)
        Parameter vector (w in the formula).

    intercept_ : float or ndarray of shape (n_targets,)
        Independent term in decision function.

    n_iter_ : int or array-like
        Number of active features across every target.

    n_nonzero_coefs_ : int or None
        The number of non-zero coefficients in the solution or `None` when `tol` is
        set. If `n_nonzero_coefs` is None and `tol` is None this value is either set
        to 10% of `n_features` or 1, whichever is greater.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    orthogonal_mp : Solves n_targets Orthogonal Matching Pursuit problems.
    orthogonal_mp_gram :  Solves n_targets Orthogonal Matching Pursuit
        problems using only the Gram matrix X.T * X and the product X.T * y.
    lars_path : Compute Least Angle Regression or Lasso path using LARS algorithm.
    Lars : Least Angle Regression model a.k.a. LAR.
    LassoLars : Lasso model fit with Least Angle Regression a.k.a. Lars.
    sklearn.decomposition.sparse_encode : Generic sparse coding.
        Each column of the result is the solution to a Lasso problem.
    OrthogonalMatchingPursuitCV : Cross-validated
        Orthogonal Matching Pursuit model (OMP).

    Notes
    -----
    Orthogonal matching pursuit was introduced in G. Mallat, Z. Zhang,
    Matching pursuits with time-frequency dictionaries, IEEE Transactions on
    Signal Processing, Vol. 41, No. 12. (December 1993), pp. 3397-3415.
    (https://www.di.ens.fr/~mallat/papiers/MallatPursuit93.pdf)

    This implementation is based on Rubinstein, R., Zibulevsky, M. and Elad,
    M., Efficient Implementation of the K-SVD Algorithm using Batch Orthogonal
    Matching Pursuit Technical Report - CS Technion, April 2008.
    https://www.cs.technion.ac.il/~ronrubin/Publications/KSVD-OMP-v2.pdf

    Examples
    --------
    >>> from sklearn.linear_model import OrthogonalMatchingPursuit
    >>> from sklearn.datasets import make_regression
    >>> X, y = make_regression(noise=4, random_state=0)
    >>> reg = OrthogonalMatchingPursuit().fit(X, y)
    >>> reg.score(X, y)
    0.9991
    >>> reg.predict(X[:1,])
    array([-78.3854])
    """

    _parameter_constraints: dict = {
        "n_nonzero_coefs": [Interval(Integral, 1, None, closed="left"), None],
        "tol": [Interval(Real, 0, None, closed="left"), None],
        "fit_intercept": ["boolean"],
        "precompute": [StrOptions({"auto"}), "boolean"],
    }

    def __init__(
        self,
        *,
        n_nonzero_coefs=None,
        tol=None,
        fit_intercept=True,
        precompute="auto",
    ):
        # [FUNCTION DOC]
        # Stores n_nonzero_coefs, tol, fit_intercept, precompute unchanged on self.
        self.n_nonzero_coefs = n_nonzero_coefs
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.precompute = precompute

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y):
        # [FUNCTION DOC]
        # Fits the model. Validates X/y (float dtypes, multi-output), centers data and
        # optionally precomputes the Gram matrix via _pre_fit (precompute may be "auto"),
        # determines n_nonzero_coefs_ (default max(10% n_features, 1), None in tol mode),
        # then calls orthogonal_mp (no Gram) or orthogonal_mp_gram (with squared norms of the
        # centered y in tol mode), transposes the coefficients and sets the intercept from the
        # offsets.
        # Inputs: X (n_samples, n_features); y (n_samples,) or (n_samples, n_targets).
        # Outputs: self.
        # Side effects: sets coef_, intercept_, n_iter_, n_nonzero_coefs_; RuntimeWarning
        # from the solver on premature stop.
        """Fit the model using X, y as training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Target values. Will be cast to X's dtype if necessary.

        Returns
        -------
        self : object
            Returns an instance of self.
        """
        X, y = validate_data(
            self, X, y, multi_output=True, y_numeric=True, dtype=FLOAT_DTYPES
        )
        n_features = X.shape[1]

        X, y, X_offset, y_offset, X_scale, Gram, Xy = _pre_fit(
            X,
            y,
            None,
            self.precompute,
            self.fit_intercept,
            copy=True,
            check_gram=False,
        )

        if y.ndim == 1:
            y = y[:, np.newaxis]

        if self.n_nonzero_coefs is None and self.tol is None:
            # default for n_nonzero_coefs is 0.1 * n_features
            # but at least one.
            self.n_nonzero_coefs_ = max(int(0.1 * n_features), 1)
        elif self.tol is not None:
            self.n_nonzero_coefs_ = None
        else:
            self.n_nonzero_coefs_ = self.n_nonzero_coefs

        if Gram is False:
            coef_, self.n_iter_ = orthogonal_mp(
                X,
                y,
                n_nonzero_coefs=self.n_nonzero_coefs_,
                tol=self.tol,
                precompute=False,
                copy_X=True,
                return_n_iter=True,
            )
        else:
            norms_sq = np.sum(y**2, axis=0) if self.tol is not None else None

            coef_, self.n_iter_ = orthogonal_mp_gram(
                Gram,
                Xy=Xy,
                n_nonzero_coefs=self.n_nonzero_coefs_,
                tol=self.tol,
                norms_squared=norms_sq,
                copy_Gram=True,
                copy_Xy=True,
                return_n_iter=True,
            )
        self.coef_ = coef_.T
        self._set_intercept(X_offset, y_offset, X_scale)
        return self


def _omp_path_residues(
    X_train,
    y_train,
    X_test,
    y_test,
    copy=True,
    fit_intercept=True,
    max_iter=100,
):
    # [FUNCTION DOC]
    # HELPER: Computes held-out residuals along the OMP path for one CV fold.
    # Optionally copies the arrays, centers train and test X and y with the training
    # means when fit_intercept, runs orthogonal_mp with return_path for max_iter atoms
    # (copy_X False since already copied), and returns for each path step the test
    # predictions minus y_test.
    # Inputs: X_train, y_train, X_test, y_test - arrays; copy bool; fit_intercept bool;
    # max_iter int (maximum number of atoms).
    # Outputs: ndarray (n_steps, n_test_samples) of residuals.
    # Side effects: mutates the input arrays in place when copy is False.
    """Compute the residues on left-out data for a full LARS path.

    Parameters
    ----------
    X_train : ndarray of shape (n_samples, n_features)
        The data to fit the LARS on.

    y_train : ndarray of shape (n_samples)
        The target variable to fit LARS on.

    X_test : ndarray of shape (n_samples, n_features)
        The data to compute the residues on.

    y_test : ndarray of shape (n_samples)
        The target variable to compute the residues on.

    copy : bool, default=True
        Whether X_train, X_test, y_train and y_test should be copied.  If
        False, they may be overwritten.

    fit_intercept : bool, default=True
        Whether to calculate the intercept for this model. If set
        to false, no intercept will be used in calculations
        (i.e. data is expected to be centered).

    max_iter : int, default=100
        Maximum numbers of iterations to perform, therefore maximum features
        to include. 100 by default.

    Returns
    -------
    residues : ndarray of shape (n_samples, max_features)
        Residues of the prediction on the test data.
    """

    if copy:
        X_train = X_train.copy()
        y_train = y_train.copy()
        X_test = X_test.copy()
        y_test = y_test.copy()

    if fit_intercept:
        X_mean = X_train.mean(axis=0)
        X_train -= X_mean
        X_test -= X_mean
        y_mean = y_train.mean(axis=0)
        y_train = as_float_array(y_train, copy=False)
        y_train -= y_mean
        y_test = as_float_array(y_test, copy=False)
        y_test -= y_mean

    coefs = orthogonal_mp(
        X_train,
        y_train,
        n_nonzero_coefs=max_iter,
        tol=None,
        precompute=False,
        copy_X=False,
        return_path=True,
    )
    if coefs.ndim == 1:
        coefs = coefs[:, np.newaxis]

    return np.dot(coefs.T, X_test.T) - y_test


class OrthogonalMatchingPursuitCV(RegressorMixin, LinearModel):
    # [CLASS DOC]
    # Cross-validated OMP: chooses the number of non-zero coefficients minimising mean
    # held-out squared error, then refits an OrthogonalMatchingPursuit on all data.
    # Single-target only.
    """Cross-validated Orthogonal Matching Pursuit model (OMP).

    See glossary entry for :term:`cross-validation estimator`.

    Read more in the :ref:`User Guide <omp>`.

    Parameters
    ----------
    copy : bool, default=True
        Whether the design matrix X must be copied by the algorithm. A false
        value is only helpful if X is already Fortran-ordered, otherwise a
        copy is made anyway.

    fit_intercept : bool, default=True
        Whether to calculate the intercept for this model. If set
        to false, no intercept will be used in calculations
        (i.e. data is expected to be centered).

    max_iter : int, default=None
        Maximum numbers of iterations to perform, therefore maximum features
        to include. 10% of ``n_features`` but at least 5 if available.

    cv : int, cross-validation generator or iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, to use the default 5-fold cross-validation,
        - integer, to specify the number of folds,
        - :term:`CV splitter`,
        - an iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, :class:`~sklearn.model_selection.KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.

        .. versionchanged:: 0.22
            ``cv`` default value if None changed from 3-fold to 5-fold.

    n_jobs : int, default=None
        Number of CPUs to use during the cross validation.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    verbose : bool or int, default=False
        Sets the verbosity amount.

    Attributes
    ----------
    intercept_ : float or ndarray of shape (n_targets,)
        Independent term in decision function.

    coef_ : ndarray of shape (n_features,) or (n_targets, n_features)
        Parameter vector (w in the problem formulation).

    n_nonzero_coefs_ : int
        Estimated number of non-zero coefficients giving the best mean squared
        error over the cross-validation folds.

    n_iter_ : int or array-like
        Number of active features across every target for the model refit with
        the best hyperparameters got by cross-validating across all folds.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    orthogonal_mp : Solves n_targets Orthogonal Matching Pursuit problems.
    orthogonal_mp_gram : Solves n_targets Orthogonal Matching Pursuit
        problems using only the Gram matrix X.T * X and the product X.T * y.
    lars_path : Compute Least Angle Regression or Lasso path using LARS algorithm.
    Lars : Least Angle Regression model a.k.a. LAR.
    LassoLars : Lasso model fit with Least Angle Regression a.k.a. Lars.
    OrthogonalMatchingPursuit : Orthogonal Matching Pursuit model (OMP).
    LarsCV : Cross-validated Least Angle Regression model.
    LassoLarsCV : Cross-validated Lasso model fit with Least Angle Regression.
    sklearn.decomposition.sparse_encode : Generic sparse coding.
        Each column of the result is the solution to a Lasso problem.

    Notes
    -----
    In `fit`, once the optimal number of non-zero coefficients is found through
    cross-validation, the model is fit again using the entire training set.

    Examples
    --------
    >>> from sklearn.linear_model import OrthogonalMatchingPursuitCV
    >>> from sklearn.datasets import make_regression
    >>> X, y = make_regression(n_features=100, n_informative=10,
    ...                        noise=4, random_state=0)
    >>> reg = OrthogonalMatchingPursuitCV(cv=5).fit(X, y)
    >>> reg.score(X, y)
    0.9991
    >>> reg.n_nonzero_coefs_
    np.int64(10)
    >>> reg.predict(X[:1,])
    array([-78.3854])
    """

    _parameter_constraints: dict = {
        "copy": ["boolean"],
        "fit_intercept": ["boolean"],
        "max_iter": [Interval(Integral, 0, None, closed="left"), None],
        "cv": ["cv_object"],
        "n_jobs": [Integral, None],
        "verbose": ["verbose"],
    }

    def __init__(
        self,
        *,
        copy=True,
        fit_intercept=True,
        max_iter=None,
        cv=None,
        n_jobs=None,
        verbose=False,
    ):
        # [FUNCTION DOC]
        # Stores copy, fit_intercept, max_iter, cv, n_jobs, verbose unchanged on self.
        self.copy = copy
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.cv = cv
        self.n_jobs = n_jobs
        self.verbose = verbose

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y, **fit_params):
        # [FUNCTION DOC]
        # Validates X (at least 2 features) and y, converts X to float, resolves cv and routed
        # splitter params. max_iter defaults to min(max(10% n_features, 5), n_features).
        # Computes path residuals per fold in parallel (_omp_path_residues), truncates all
        # folds to the shortest path, computes per-step MSE per fold, averages over folds and
        # picks the step with minimal mean MSE (number of atoms = index + 1). Refits
        # OrthogonalMatchingPursuit with that number of atoms.
        # Inputs: X (n_samples, n_features); y (n_samples,); **fit_params routed to the
        # splitter.
        # Outputs: self.
        # Side effects: sets n_nonzero_coefs_, coef_, intercept_, n_iter_.
        # KNOWN ISSUE: max_iter=0 is allowed by the constraints but treated like None because
        # of a truthiness test.
        """Fit the model using X, y as training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        y : array-like of shape (n_samples,)
            Target values. Will be cast to X's dtype if necessary.

        **fit_params : dict
            Parameters to pass to the underlying splitter.

            .. versionadded:: 1.4
                Only available if `enable_metadata_routing=True`,
                which can be set by using
                ``sklearn.set_config(enable_metadata_routing=True)``.
                See :ref:`Metadata Routing User Guide <metadata_routing>` for
                more details.

        Returns
        -------
        self : object
            Returns an instance of self.
        """
        _raise_for_params(fit_params, self, "fit")

        X, y = validate_data(self, X, y, y_numeric=True, ensure_min_features=2)
        X = as_float_array(X, copy=False, ensure_all_finite=False)
        cv = check_cv(self.cv, classifier=False)
        if _routing_enabled():
            routed_params = process_routing(self, "fit", **fit_params)
        else:
            # TODO(SLEP6): remove when metadata routing cannot be disabled.
            routed_params = _manual_routing({"splitter": {}})
        max_iter = (
            min(max(int(0.1 * X.shape[1]), 5), X.shape[1])
            if not self.max_iter
            else self.max_iter
        )
        cv_paths = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
            delayed(_omp_path_residues)(
                X[train],
                y[train],
                X[test],
                y[test],
                self.copy,
                self.fit_intercept,
                max_iter,
            )
            for train, test in cv.split(X, **routed_params.splitter.split)
        )

        min_early_stop = min(fold.shape[0] for fold in cv_paths)
        mse_folds = np.array(
            [(fold[:min_early_stop] ** 2).mean(axis=1) for fold in cv_paths]
        )
        best_n_nonzero_coefs = np.argmin(mse_folds.mean(axis=0)) + 1
        self.n_nonzero_coefs_ = best_n_nonzero_coefs
        omp = OrthogonalMatchingPursuit(
            n_nonzero_coefs=best_n_nonzero_coefs,
            fit_intercept=self.fit_intercept,
        ).fit(X, y)

        self.coef_ = omp.coef_
        self.intercept_ = omp.intercept_
        self.n_iter_ = omp.n_iter_
        return self

    def get_metadata_routing(self):
        # [FUNCTION DOC]
        # Returns a MetadataRouter routing fit params to the cv splitter's split method.
        """Get metadata routing of this object.

        Please check :ref:`User Guide <metadata_routing>` on how the routing
        mechanism works.

        .. versionadded:: 1.4

        Returns
        -------
        routing : MetadataRouter
            A :class:`~sklearn.utils.metadata_routing.MetadataRouter` encapsulating
            routing information.
        """

        router = MetadataRouter(owner=self).add(
            splitter=self.cv,
            method_mapping=MethodMapping().add(caller="fit", callee="split"),
        )
        return router
