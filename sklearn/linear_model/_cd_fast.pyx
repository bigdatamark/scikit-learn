# [MODULE DOC]
# Coordinate descent (CD) solvers for Elastic-Net type problems, compiled for float32
# and float64 through the fused type "floating". Public entry points (called from
# sklearn/linear_model/_coordinate_descent.py):
# - enet_coordinate_descent: dense Fortran-ordered X, single target.
# - sparse_enet_coordinate_descent: CSC sparse X, optional sample weights and implicit
# centering by X_mean (X is never densified or centered explicitly).
# - enet_coordinate_descent_gram: works on a precomputed Gram matrix Q = X'X and q = X'y.
# - enet_coordinate_descent_multi_task: dense or CSC X, multi-output Y with an L2,1
# (group) penalty (MultiTaskLasso / MultiTaskElasticNet).
# All solvers minimize 1/2 ||y - Xw||^2 + alpha * penalty_1(w) + beta/2 ||w||^2 by cyclic or
# random coordinate updates with soft thresholding, track the residual (or Q @ w) incrementally,
# use the duality gap (computed by the gap_* helpers) as stopping criterion with tolerance
# tol * ||y||^2, and optionally apply Gap Safe screening rules to discard features that are
# provably zero at the optimum. The heavy loops run without the GIL using BLAS wrappers from
# sklearn.utils._cython_blas. Non-convergence emits a ConvergenceWarning.
# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

from libc.math cimport fabs, sqrt
import numpy as np

from cython cimport floating
import warnings
from sklearn.exceptions import ConvergenceWarning

from sklearn.utils._cython_blas cimport (
    _axpy, _dot, _asum, _gemv, _nrm2, _copy, _scal
)
from sklearn.utils._cython_blas cimport ColMajor, Trans, NoTrans
from sklearn.utils._typedefs cimport int32_t, uint8_t, uint32_t
from sklearn.utils._random cimport our_rand_r


# The following two functions are shamelessly copied from the tree code.

cdef enum:
    # Max value for our rand_r replacement (near the bottom).
    # We don't use RAND_MAX because it's different across platforms and
    # particularly tiny on Windows/MSVC.
    # It corresponds to the maximum representable value for
    # 32-bit signed integers (i.e. 2^31 - 1).
    RAND_R_MAX = 2147483647


cdef inline uint32_t rand_int(uint32_t end, uint32_t* random_state) noexcept nogil:
    # [FUNCTION DOC]
    # HELPER: random coordinate selection for the CD solvers.
    # Behavior: draws the next value of the custom 32-bit generator our_rand_r and returns it
    # modulo end, i.e. an integer in [0, end) (slightly biased for large end).
    # Inputs: end, exclusive upper bound, must be > 0; random_state, pointer to the generator state.
    # Outputs: the random integer. Exceptions: none (nogil, noexcept); end == 0 is undefined.
    # Side effects: advances the generator state pointed to by random_state.
    """Generate a random integer in [0; end)."""
    return our_rand_r(random_state) % end


cdef inline floating fmax(floating x, floating y) noexcept nogil:
    # [FUNCTION DOC]
    # HELPER: scalar maximum used throughout the module.
    # Behavior: returns x if x > y, else y (so if either argument is NaN, y is returned).
    # Inputs: two floating scalars. Outputs: the larger value. Exceptions: none. Side effects: none.
    if x > y:
        return x
    return y


cdef inline floating fsign(floating f) noexcept nogil:
    # [FUNCTION DOC]
    # HELPER: sign function used in the soft-thresholding coordinate update.
    # Behavior: returns 0 for f == 0, 1 for positive and -1 for negative (NaN gives -1).
    # Inputs: f, scalar. Outputs: the sign as a floating value. Exceptions: none. Side effects: none.
    if f == 0:
        return 0
    elif f > 0:
        return 1.0
    else:
        return -1.0


cdef inline floating abs_max(int n, const floating* a) noexcept nogil:
    # [FUNCTION DOC]
    # HELPER: infinity norm of a vector, used for dual norms and multi-task stopping criteria.
    # Behavior: linear scan returning the largest absolute value among the first n entries.
    # Inputs: n, number of entries, must be >= 1 (a[0] is always read); a, pointer to the data.
    # Outputs: max of absolute values. Exceptions: none. Side effects: none.
    """np.max(np.abs(a))"""
    cdef int i
    cdef floating m = fabs(a[0])
    cdef floating d
    for i in range(1, n):
        d = fabs(a[i])
        if d > m:
            m = d
    return m


cdef inline floating max(int n, floating* a) noexcept nogil:
    # [FUNCTION DOC]
    # HELPER: maximum of a vector, used as dual norm when positive=True.
    # Behavior: linear scan returning the largest value among the first n entries (not absolute).
    # Inputs: n, must be >= 1; a, pointer to the data.
    # Outputs: the maximum. Exceptions: none. Side effects: none. Note it shadows the builtin name.
    """np.max(a)"""
    cdef int i
    cdef floating m = a[0]
    cdef floating d
    for i in range(1, n):
        d = a[i]
        if d > m:
            m = d
    return m


cdef inline floating diff_abs_max(int n, const floating* a, floating* b) noexcept nogil:
    # [FUNCTION DOC]
    # HELPER: largest absolute coefficient change, used by the multi-task solver.
    # Behavior: returns the maximum over i < n of the absolute value of a[i] - b[i].
    # Inputs: n, must be >= 1; a and b, pointers to n values each.
    # Outputs: the maximum absolute difference. Exceptions: none. Side effects: none.
    """np.max(np.abs(a - b))"""
    cdef int i
    cdef floating m = fabs(a[0] - b[0])
    cdef floating d
    for i in range(1, n):
        d = fabs(a[i] - b[i])
        if d > m:
            m = d
    return m


cdef inline floating sparse_dot(
    int32_t j,
    const floating[::1] X_data,  # in
    const int32_t[::1] X_indices,  # in
    const int32_t[::1] X_indptr,  # in
    const floating[::1] y,
) noexcept nogil:
    # [FUNCTION DOC]
    # HELPER: sparse column dot product for the multi-task solver with CSC X.
    # Behavior: sums X_data[k] * y[X_indices[k]] over the stored entries k of column j, i.e. the
    # dot product of column j of X with dense vector y (no centering applied here).
    # Inputs: j, column index; X_data, X_indices, X_indptr, CSC arrays of X; y, dense vector of
    # length n_samples. Outputs: the dot product. Exceptions: none. Side effects: none.
    """BLAS X[:, j] @ y for sparse CSC X."""
    cdef int32_t i, i_ind
    cdef int32_t startptr = X_indptr[j]
    cdef int32_t endptr = X_indptr[j + 1]
    cdef floating result = 0

    for i_ind in range(startptr, endptr):
        i = X_indices[i_ind]
        result += X_data[i_ind] * y[i]
    return result


cdef inline floating sparse_axpy(
    int32_t j,
    floating a,
    const floating[::1] X_data,  # in
    const int32_t[::1] X_indices,  # in
    const int32_t[::1] X_indptr,  # in
    floating[::1] y,  # out
) noexcept nogil:
    # [FUNCTION DOC]
    # HELPER: sparse axpy for computing the initial multi-task residual with CSC X.
    # Behavior: for each stored entry of column j, adds a * value to y at that row.
    # Inputs: j, column index; a, scalar; X_data, X_indices, X_indptr, CSC arrays of X;
    # y, dense vector of length n_samples.
    # Outputs: y is updated in place. Exceptions: none. Side effects: mutates y.
    # KNOWN ISSUE: declared to return a floating value but never returns one; callers ignore the
    # result, so it is harmless, but the return type should be void.
    """BLAS y += a * X[:, j] for sparse CSC X."""
    cdef int32_t i, i_ind
    cdef int32_t startptr = X_indptr[j]
    cdef int32_t endptr = X_indptr[j + 1]

    for i_ind in range(startptr, endptr):
        i = X_indices[i_ind]
        y[i] += a * X_data[i_ind]


message_conv = (
    "Objective did not converge. You might want to increase "
    "the number of iterations, check the scale of the "
    "features or consider increasing regularisation."
)


message_ridge = (
    "Linear regression models with a zero l1 penalization "
    "strength are more efficiently fitted using one of the "
    "solvers implemented in "
    "sklearn.linear_model.Ridge/RidgeCV instead."
)


cdef inline floating dual_gap_formulation_A(
    floating alpha,  # L1 penalty
    floating beta,  # L1 penalty
    floating w_l1_norm,
    floating w_l2_norm2,
    floating R_norm2,  # R @ R
    floating Ry,  # R @ y
    floating dual_norm_XtA,
) noexcept nogil:
    # [FUNCTION DOC]
    # HELPER: shared duality gap computation (formulation A) for all gap_* functions.
    # Behavior: primal = 1/2 (R_norm2 + beta w_l2_norm2) + alpha w_l1_norm. The dual point is the
    # residual scaled by s = alpha / dual_norm_XtA if dual_norm_XtA > alpha, else s = 1; the dual
    # value is -1/2 s^2 (R_norm2 + beta w_l2_norm2) + s Ry. Returns primal - dual.
    # Inputs: alpha, L1 (or L2,1) penalty; beta, L2 penalty (the inline comment calling it L1 is a
    # typo); w_l1_norm, L1 (or L2,1) norm of the coefficients; w_l2_norm2, squared L2 norm of w (0
    # if beta is 0); R_norm2, squared norm of the residual; Ry, residual dot target;
    # dual_norm_XtA, dual norm of X'R - beta w (infinity norm, max, or max row norm).
    # Outputs: the gap. Exceptions: none; dual_norm_XtA == 0 is safe since then s = 1.
    # Side effects: none.
    """Compute dual gap according to formulation A."""
    cdef floating gap, primal, dual
    cdef floating scale  # Scaling factor to achieve dual feasible point.

    primal = 0.5 * (R_norm2 + beta * w_l2_norm2) + alpha * w_l1_norm

    if (dual_norm_XtA > alpha):
        scale = alpha / dual_norm_XtA
    else:
        scale = 1.0
    dual = -0.5 * (scale ** 2) * (R_norm2 + beta * w_l2_norm2) + scale * Ry
    gap = primal - dual
    return gap


cdef (floating, floating) gap_enet(
    int n_samples,
    int n_features,
    const floating[::1] w,
    floating alpha,  # L1 penalty
    floating beta,  # L2 penalty
    const floating[::1, :] X,
    const floating[::1] y,
    const floating[::1] R,  # current residuals = y - X @ w
    floating[::1] XtA,  # XtA = X.T @ R - beta * w is calculated inplace
    bint positive,
) noexcept nogil:
    # [FUNCTION DOC]
    # HELPER: duality gap for the dense single-target solver enet_coordinate_descent.
    # Behavior: computes ||w||^2 (if beta > 0), ||R||^2 and R'y. If alpha == 0: XtA = X'R and its
    # squared L2 norm; with beta == 0 (OLS) returns gap = ||X'R||^2 (a gradient criterion, not a
    # gap); with beta > 0 uses formulation B: gap = ||R||^2 + beta/2 ||w||^2 - R'y + ||X'R||^2/(2 beta).
    # If alpha > 0: XtA = X'R - beta w (via copy plus gemv), dual norm = max(XtA) if positive else
    # max |XtA|, L1 norm of w, then delegates to dual_gap_formulation_A.
    # Inputs: n_samples, n_features; w, coefficients (n_features,); alpha, beta, penalties;
    # X, Fortran-ordered (n_samples, n_features); y, target (n_samples,); R, current residual
    # y - Xw; XtA, work buffer (n_features,); positive, whether coefficients are constrained >= 0.
    # Outputs: tuple (gap, dual_norm_XtA); XtA is overwritten (X'R or X'R - beta w). With alpha == 0
    # the second value is the squared L2 norm of X'R rather than a dual norm.
    # Exceptions: none (nogil). Side effects: writes XtA only.
    """Compute dual gap for use in enet_coordinate_descent.

    alpha > 0:            formulation A of the duality gap
    alpha = 0 & beta > 0: formulation B of the duality gap
    alpha = beta = 0:     OLS first order condition (=gradient)
    """
    cdef floating gap = 0.0
    cdef floating dual_norm_XtA
    cdef floating R_norm2
    cdef floating Ry
    cdef floating w_l1_norm
    cdef floating w_l2_norm2 = 0.0

    # w_l2_norm2 = w @ w
    if beta > 0:
        w_l2_norm2 = _dot(n_features, &w[0], 1, &w[0], 1)
    # R_norm2 = R @ R
    R_norm2 = _dot(n_samples, &R[0], 1, &R[0], 1)
    # Ry = R @ y
    if not (alpha == 0 and beta == 0):
        Ry = _dot(n_samples, &R[0], 1, &y[0], 1)

    if alpha == 0:
        # XtA = X.T @ R
        _gemv(
            ColMajor, Trans, n_samples, n_features, 1.0, &X[0, 0],
            n_samples, &R[0], 1, 0, &XtA[0], 1,
        )
        # ||X'R||_2^2
        dual_norm_XtA = _dot(n_features, &XtA[0], 1, &XtA[0], 1)
        if beta == 0:
            # This is OLS, no dual gap available. Resort to first order condition
            #     X'R = 0
            #     gap = ||X'R||_2^2
            # Compare with stopping criterion of LSQR.
            gap = dual_norm_XtA
            return gap, dual_norm_XtA
        # This is Ridge regression, we use formulation B for the dual gap.
        gap = R_norm2 + 0.5 * beta * w_l2_norm2 - Ry
        gap += 1 / (2 * beta) * dual_norm_XtA
        return gap, dual_norm_XtA

    # XtA = X.T @ R - beta * w
    _copy(n_features, &w[0], 1, &XtA[0], 1)
    _gemv(ColMajor, Trans, n_samples, n_features, 1.0, &X[0, 0],
          n_samples, &R[0], 1,
          -beta, &XtA[0], 1)

    # dual_norm_XtA
    if positive:
        dual_norm_XtA = max(n_features, &XtA[0])
    else:
        dual_norm_XtA = abs_max(n_features, &XtA[0])

    # w_l1_norm = np.sum(np.abs(w))
    w_l1_norm = _asum(n_features, &w[0], 1)

    gap = dual_gap_formulation_A(
        alpha=alpha,
        beta=beta,
        w_l1_norm=w_l1_norm,
        w_l2_norm2=w_l2_norm2,
        R_norm2=R_norm2,
        Ry=Ry,
        dual_norm_XtA=dual_norm_XtA,
    )
    return gap, dual_norm_XtA


def enet_coordinate_descent(
    floating[::1] w,
    floating alpha,
    floating beta,
    const floating[::1, :] X,
    const floating[::1] y,
    unsigned int max_iter,
    floating tol,
    object rng,
    bint random=0,
    bint positive=0,
    bint do_screening=1,
):
    # [FUNCTION DOC]
    # Behavior: dense Elastic-Net coordinate descent. Precomputes squared column norms of X and the
    # residual R = y - Xw, scales tol by ||y||^2 and returns immediately if the initial gap is <= tol.
    # With do_screening (forced off when alpha == 0) applies Gap Safe screening: features with zero
    # norm are set to 0 and excluded; feature j is kept if (1 - |XtA_j| / max(alpha, dual_norm)) /
    # sqrt(||X_j||^2 + beta) <= sqrt(2 gap) / alpha, otherwise its coefficient is zeroed (residual
    # corrected) and it is permanently excluded. Each epoch visits the n_active active features
    # (in order, or drawn with replacement when random=True) and sets
    # w_j = soft_threshold(X_j'R + w_j ||X_j||^2, alpha) / (||X_j||^2 + beta) (clipped at 0 when
    # positive and the unthresholded value is negative), updating R by an axpy when w_j changes.
    # After an epoch where max |dw| / max |w| <= the original tol (or all w are 0, or the last
    # epoch) the gap is recomputed; it stops if gap <= scaled tol, else re-screens.
    # Inputs: w, initial coefficients (n_features,), updated in place; alpha, L1 strength; beta, L2
    # strength; X, Fortran-ordered dense design; y, target; max_iter, maximum number of epochs;
    # tol, relative tolerance; rng, numpy RandomState used once to seed the internal generator;
    # random, random coordinate selection; positive, enforce w >= 0; do_screening, enable screening.
    # Outputs: (w as ndarray sharing memory with the input, gap, scaled tol, number of epochs run;
    # 0 if converged before the first epoch).
    # Exceptions: errors from numpy allocation or rng; emits ConvergenceWarning (not raised) when
    # max_iter is exhausted, with an extra hint to use Ridge if alpha is ~0.
    # Side effects: mutates w; allocates work arrays; releases the GIL for the main loop.
    # KNOWN ISSUE: when max_iter == 0 the loop never runs, the for/else still warns, and n_iter + 1
    # returns 1 instead of 0.
    """
    Cython version of the coordinate descent algorithm for Elastic-Net regression.

    The algorithm mostly follows [Friedman 2010].
    We minimize the primal

        P(w) = 1/2 ||y - X w||_2^2 + alpha ||w||_1 + beta/2 ||w||_2^2

    The dual for beta = 0, see e.g. [Fercoq 2015] with v = alpha * theta, is

        D(v) = -1/2 ||v||_2^2 + y' v    (formulation A)

    with dual feasible condition ||X^T v||_inf <= alpha.
    For beta > 0, one uses extended versions of X and y by adding n_features rows

        X -> (           X)    y -> (y)
             (sqrt(beta) I)         (0)

    Note that the residual R = y - X w is an important ingredient for the estimation of
    a dual feasible point v.
    At optimum of primal w* and dual v*, one has

        v* = y - X w*

    The duality gap is

        G(w, v) = P(w) - D(v) <= P(w) - P(w*)

    Strong duality holds: G(w*, v*) = 0.
    For testing convergence, one uses G(w, v) with current w and uses

        v = R                            if ||X^T R||_inf <= alpha
        v = R * alpha / ||X^T R||_inf    else

    The final stopping criterion is based on the duality gap

        tol ||y||_2^2 <= G(w, v)

    The tolerance here is multiplied by ||y||_2^2 to have an inequality that scales the
    same on both sides and because one has G(0, 0) = 1/2 ||y||_2^2.

    Note:
    The above dual D(v) and duality gap G require alpha > 0 because of the dual
    feasible condition.
    There is, however, an alternative dual formulation, see [Dünner 2016] 5.2.3 and
    https://github.com/scikit-learn/scikit-learn/issues/22836:

        D(v) = -1/2 ||v||_2^2 + y' v
               -1/(2 beta) sum_j (|X_j' v| - alpha)_+^2    (formulation B)

    The dual feasible set is v element real numbers. It requires beta > 0, but
    alpha = 0 is allowed. Strong duality holds and at optimum, v* = y - X w*.

    Returns
    -------
    w : ndarray of shape (n_features,)
        ElasticNet coefficients.
    gap : float
        Achieved dual gap.
    tol : float
        Equals input `tol` times `np.dot(y, y)`. The tolerance used for the dual gap.
    n_iter : int
        Number of coordinate descent iterations.

    References
    ----------
    .. [Friedman 2010]
       Jerome H. Friedman, Trevor Hastie, Rob Tibshirani. (2010)
       Regularization Paths for Generalized Linear Models via Coordinate Descent
       https://www.jstatsoft.org/article/view/v033i01

    .. [Fercoq 2015]
       Olivier Fercoq, Alexandre Gramfort, Joseph Salmon. (2015)
       Mind the duality gap: safer rules for the Lasso
       https://arxiv.org/abs/1505.03410

    .. [Dünner 2016]
       Celestine Dünner, Simon Forte, Martin Takác, Martin Jaggi. (2016).
       Primal-Dual Rates and Certificates. In ICML 2016.
       https://arxiv.org/abs/1602.05205
    """

    if floating is float:
        dtype = np.float32
    else:
        dtype = np.float64

    # get the data information into easy vars
    cdef unsigned int n_samples = X.shape[0]
    cdef unsigned int n_features = X.shape[1]

    # compute squared norms of the columns of X
    # same as norm2_cols_X = np.square(X).sum(axis=0)
    cdef floating[::1] norm2_cols_X = np.einsum(
        "ij,ij->j", X, X, dtype=dtype, order="C"
    )

    # initial value of the residuals
    cdef floating[::1] R = np.empty(n_samples, dtype=dtype)
    cdef floating[::1] XtA = np.empty(n_features, dtype=dtype)

    cdef floating d_j
    cdef floating Xj_theta
    cdef floating tmp
    cdef floating w_j
    cdef floating d_w_max
    cdef floating w_max
    cdef floating d_w_j
    cdef floating gap = tol + 1.0
    cdef floating d_w_tol = tol
    cdef floating dual_norm_XtA
    cdef unsigned int n_active = n_features
    cdef uint32_t[::1] active_set
    # TODO: use binset instead of array of bools
    cdef uint8_t[::1] excluded_set
    cdef unsigned int j
    cdef unsigned int n_iter = 0
    cdef unsigned int f_iter
    cdef uint32_t rand_r_state_seed = rng.randint(0, RAND_R_MAX)
    cdef uint32_t* rand_r_state = &rand_r_state_seed

    if alpha == 0:
        # No screeing without L1-penalty.
        do_screening = False

    if do_screening:
        active_set = np.empty(n_features, dtype=np.uint32)  # map [:n_active] -> j
        excluded_set = np.empty(n_features, dtype=np.uint8)

    with nogil:
        # R = y - np.dot(X, w)
        _copy(n_samples, &y[0], 1, &R[0], 1)
        _gemv(ColMajor, NoTrans, n_samples, n_features, -1.0, &X[0, 0],
              n_samples, &w[0], 1, 1.0, &R[0], 1)

        # tol *= np.dot(y, y)
        tol *= _dot(n_samples, &y[0], 1, &y[0], 1)

        # Check convergence before entering the main loop.
        gap, dual_norm_XtA = gap_enet(
            n_samples, n_features, w, alpha, beta, X, y, R, XtA, positive
        )
        if gap <= tol:
            with gil:
                return np.asarray(w), gap, tol, 0

        # Gap Safe Screening Rules, see https://arxiv.org/abs/1802.07481, Eq. 11
        if do_screening:
            n_active = 0
            for j in range(n_features):
                if norm2_cols_X[j] == 0:
                    w[j] = 0
                    excluded_set[j] = 1
                    continue
                Xj_theta = XtA[j] / fmax(alpha, dual_norm_XtA)  # X[:,j] @ dual_theta
                d_j = (1 - fabs(Xj_theta)) / sqrt(norm2_cols_X[j] + beta)
                if d_j <= sqrt(2 * gap) / alpha:
                    # include feature j
                    active_set[n_active] = j
                    excluded_set[j] = 0
                    n_active += 1
                else:
                    if w[j] != 0:
                        # R += w[j] * X[:,j]
                        _axpy(n_samples, w[j], &X[0, j], 1, &R[0], 1)
                        w[j] = 0
                    excluded_set[j] = 1

        for n_iter in range(max_iter):
            w_max = 0.0
            d_w_max = 0.0
            for f_iter in range(n_active):  # Loop over coordinates
                if random:
                    j = rand_int(n_active, rand_r_state)
                else:
                    j = f_iter

                if do_screening:
                    j = active_set[j]

                if norm2_cols_X[j] == 0.0:
                    continue

                w_j = w[j]  # Store previous value

                # tmp = X[:,j] @ (R + w_j * X[:,j])
                tmp = _dot(n_samples, &X[0, j], 1, &R[0], 1) + w_j * norm2_cols_X[j]

                if positive and tmp < 0:
                    w[j] = 0.0
                else:
                    w[j] = (fsign(tmp) * fmax(fabs(tmp) - alpha, 0)
                            / (norm2_cols_X[j] + beta))

                if w[j] != w_j:
                    # R -= (w[j] - w_j) * X[:,j] # Update residual
                    _axpy(n_samples, w_j - w[j], &X[0, j], 1, &R[0], 1)

                # update the maximum absolute coefficient update
                d_w_j = fabs(w[j] - w_j)
                d_w_max = fmax(d_w_max, d_w_j)

                w_max = fmax(w_max, fabs(w[j]))

            if (
                w_max == 0.0
                or d_w_max / w_max <= d_w_tol
                or n_iter == max_iter - 1
            ):
                # the biggest coordinate update of this iteration was smaller
                # than the tolerance: check the duality gap as ultimate
                # stopping criterion
                gap, dual_norm_XtA = gap_enet(
                    n_samples, n_features, w, alpha, beta, X, y, R, XtA, positive
                )
                if gap <= tol:
                    # return if we reached desired tolerance
                    break

                # Gap Safe Screening Rules, see https://arxiv.org/abs/1802.07481, Eq. 11
                if do_screening:
                    n_active = 0
                    for j in range(n_features):
                        if excluded_set[j]:
                            continue
                        Xj_theta = XtA[j] / fmax(alpha, dual_norm_XtA)  # X @ dual_theta
                        d_j = (1 - fabs(Xj_theta)) / sqrt(norm2_cols_X[j] + beta)
                        if d_j <= sqrt(2 * gap) / alpha:
                            # include feature j
                            active_set[n_active] = j
                            excluded_set[j] = 0
                            n_active += 1
                        else:
                            if w[j] != 0:
                                # R += w[j] * X[:,j]
                                _axpy(n_samples, w[j], &X[0, j], 1, &R[0], 1)
                                w[j] = 0
                            excluded_set[j] = 1

        else:
            # for/else, runs if for doesn't end with a `break`
            with gil:
                message = (
                    message_conv +
                    f" Duality gap: {gap:.6e}, tolerance: {tol:.3e}"
                )
                if alpha < np.finfo(np.float64).eps:
                    message += "\n" + message_ridge
                warnings.warn(message, ConvergenceWarning)

    return np.asarray(w), gap, tol, n_iter + 1


cdef inline void R_plus_wj_Xj(
    unsigned int n_samples,
    floating[::1] R,  # out
    const floating[::1] X_data,
    const int32_t[::1] X_indices,
    const int32_t[::1] X_indptr,
    const floating[::1] X_mean,
    bint center,
    const floating[::1] sample_weight,
    bint no_sample_weights,
    floating w_j,
    unsigned int j,
) noexcept nogil:
    # [FUNCTION DOC]
    # HELPER: residual update for sparse (optionally centered and weighted) X.
    # Behavior: adds w_j times column j of the implicitly centered matrix to R. Without weights:
    # R[i] += X[i, j] w_j over stored entries, then, if center, R[i] -= X_mean[j] w_j for all rows.
    # With weights each term is multiplied by sample_weight[i] (R holds weighted residuals).
    # Inputs: n_samples; R, residual vector (n_samples,); X_data, X_indices, X_indptr, CSC arrays;
    # X_mean, column means; center, whether to subtract means; sample_weight, weights (may be None
    # if no_sample_weights); no_sample_weights, flag; w_j, scalar multiplier; j, column index.
    # Outputs: R updated in place. Exceptions: none. Side effects: mutates R; O(n_samples) when center.
    """R += w_j * X[:,j]"""
    cdef int32_t i, i_ind
    cdef int32_t startptr = X_indptr[j]
    cdef int32_t endptr = X_indptr[j + 1]
    cdef floating sw
    cdef floating X_mean_j = X_mean[j]
    if no_sample_weights:
        for i_ind in range(startptr, endptr):
            i = X_indices[i_ind]
            R[i] += X_data[i_ind] * w_j
        if center:
            for i in range(n_samples):
                R[i] -= X_mean_j * w_j
    else:
        for i_ind in range(startptr, endptr):
            i = X_indices[i_ind]
            sw = sample_weight[i]
            R[i] += sw * X_data[i_ind] * w_j
        if center:
            for i in range(n_samples):
                R[i] -= sample_weight[i] * X_mean_j * w_j


cdef (floating, floating) gap_enet_sparse(
    int n_samples,
    int n_features,
    const floating[::1] w,
    floating alpha,  # L1 penalty
    floating beta,  # L2 penalty
    const floating[::1] X_data,
    const int32_t[::1] X_indices,
    const int32_t[::1] X_indptr,
    const floating[::1] y,
    const floating[::1] sample_weight,
    bint no_sample_weights,
    const floating[::1] X_mean,
    bint center,
    const floating[::1] R,  # current residuals = y - X @ w
    floating R_sum,
    floating[::1] XtA,  # XtA = X.T @ R - beta * w is calculated inplace
    bint positive,
) noexcept nogil:
    # [FUNCTION DOC]
    # HELPER: duality gap for sparse_enet_coordinate_descent.
    # Behavior: same three regimes as gap_enet (formulation A for alpha > 0, formulation B for
    # alpha == 0 and beta > 0, squared gradient norm for OLS), adapted to CSC X and weights. Since
    # R holds sample_weight times the residual, ||R||^2 in the weighted metric is the sum of
    # R[i]^2 / sw[i] over rows with nonzero weight, and R'y is used directly. X'R is computed
    # column by column from the CSC arrays, minus X_mean[j] * R_sum when center.
    # Inputs: as gap_enet plus X_data, X_indices, X_indptr (CSC X); sample_weight and
    # no_sample_weights; X_mean and center for implicit centering; R_sum, sum of R.
    # Outputs: (gap, dual_norm_XtA); XtA overwritten. Exceptions: none (nogil). Side effects: writes XtA.
    """Compute dual gap for use in sparse_enet_coordinate_descent.

    alpha > 0:            formulation A of the duality gap
    alpha = 0 & beta > 0: formulation B of the duality gap
    alpha = beta = 0:     OLS first order condition (=gradient)
    """
    cdef floating gap = 0.0
    cdef floating dual_norm_XtA
    cdef floating R_norm2
    cdef floating Ry
    cdef floating w_l1_norm
    cdef floating w_l2_norm2 = 0.0
    cdef int32_t i, i_ind, j

    # w_l2_norm2 = w @ w
    if beta > 0:
        w_l2_norm2 = _dot(n_features, &w[0], 1, &w[0], 1)
    # R_norm2 = R @ R
    if no_sample_weights:
        R_norm2 = _dot(n_samples, &R[0], 1, &R[0], 1)
    else:
        R_norm2 = 0.0
        for i in range(n_samples):
            # R is already multiplied by sample_weight
            if sample_weight[i] != 0:
                R_norm2 += (R[i] ** 2) / sample_weight[i]
    # Ry = R @ y
    if not (alpha == 0 and beta == 0):
        # Note that with sample_weight, R equals R*sw and y is just y, such that
        # Ry = (sw * R) @ y, as it should be.
        Ry = _dot(n_samples, &R[0], 1, &y[0], 1)

    if alpha == 0:
        # XtA = X.T @ R
        for j in range(n_features):
            XtA[j] = 0.0
            for i_ind in range(X_indptr[j], X_indptr[j + 1]):
                i = X_indices[i_ind]
                XtA[j] += X_data[i_ind] * R[i]

            if center:
                XtA[j] -= X_mean[j] * R_sum
        # ||X'R||_2^2
        dual_norm_XtA = _dot(n_features, &XtA[0], 1, &XtA[0], 1)
        if beta == 0:
            # This is OLS, no dual gap available. Resort to first order condition
            #     X'R = 0
            #     gap = ||X'R||_2^2
            # Compare with stopping criterion of LSQR.
            gap = dual_norm_XtA
            return gap, dual_norm_XtA
        # This is Ridge regression, we use formulation B for the dual gap.
        gap = R_norm2 + 0.5 * beta * w_l2_norm2 - Ry
        gap += 1 / (2 * beta) * dual_norm_XtA
        return gap, dual_norm_XtA

    # XtA = X.T @ R - beta * w
    # sparse X.T @ dense R
    for j in range(n_features):
        XtA[j] = 0.0
        for i_ind in range(X_indptr[j], X_indptr[j + 1]):
            i = X_indices[i_ind]
            XtA[j] += X_data[i_ind] * R[i]

        if center:
            XtA[j] -= X_mean[j] * R_sum
        XtA[j] -= beta * w[j]

    # dual_norm_XtA
    if positive:
        dual_norm_XtA = max(n_features, &XtA[0])
    else:
        dual_norm_XtA = abs_max(n_features, &XtA[0])

    # w_l1_norm = np.sum(np.abs(w))
    w_l1_norm = _asum(n_features, &w[0], 1)

    gap = dual_gap_formulation_A(
        alpha=alpha,
        beta=beta,
        w_l1_norm=w_l1_norm,
        w_l2_norm2=w_l2_norm2,
        R_norm2=R_norm2,
        Ry=Ry,
        dual_norm_XtA=dual_norm_XtA,
    )
    return gap, dual_norm_XtA


def sparse_enet_coordinate_descent(
    floating[::1] w,
    floating alpha,
    floating beta,
    const floating[::1] X_data,
    const int32_t[::1] X_indices,
    const int32_t[::1] X_indptr,
    const floating[::1] y,
    const floating[::1] sample_weight,
    const floating[::1] X_mean,
    unsigned int max_iter,
    floating tol,
    object rng,
    bint random=0,
    bint positive=0,
    bint do_screening=1,
):
    # [FUNCTION DOC]
    # Behavior: Elastic-Net coordinate descent for CSC sparse X, same algorithm and screening as
    # enet_coordinate_descent, but X is centered implicitly by X_mean and sample weights are
    # handled without square roots: R stores sw * (y - Zw), the weighted target is sw * y, and the
    # column norms are the weighted squared norms of the centered columns (computed from stored
    # entries plus a correction for the implicit zeros). center is True if any X_mean entry is
    # nonzero. The coordinate gradient is sum over stored entries of R[i] X[i, j] plus
    # w_j ||Z_j||^2, minus R_sum X_mean[j] when centering. tol is scaled by y'(sw y).
    # Inputs: w, coefficients updated in place; alpha, beta, penalties; X_data, X_indices, X_indptr,
    # CSC arrays (int32 indices); y, target; sample_weight, weights or None; X_mean, column offsets
    # (zeros for no centering); max_iter, tol, rng, random, positive, do_screening as in the dense
    # solver.
    # Outputs: (w, gap, scaled tol, number of epochs), same conventions as the dense solver.
    # Exceptions: allocation errors; ConvergenceWarning when not converged.
    # Side effects: mutates w; allocates R, XtA and column norms; releases the GIL.
    # KNOWN ISSUE: during initialization R_sum accumulates the sum of R once per feature (inside the
    # column loop), so it equals sum(R) only when sum(R) is zero at every step, i.e. when y is
    # (weighted-)centered, which the caller ensures whenever X_mean is nonzero.
    # KNOWN ISSUE: same max_iter == 0 behavior as enet_coordinate_descent (warns, returns 1).
    """Cython version of the coordinate descent algorithm for Elastic-Net

    We minimize:

        1/2 * norm(y - Z w, 2)^2 + alpha * norm(w, 1) + (beta/2) * norm(w, 2)^2

    where Z = X - X_mean.
    With sample weights sw, this becomes

        1/2 * sum(sw * (y - Z w)^2, axis=0) + alpha * norm(w, 1)
        + (beta/2) * norm(w, 2)^2

    and X_mean is the weighted average of X (per column).

    The rest is the same as enet_coordinate_descent, but for sparse X.

    Returns
    -------
    w : ndarray of shape (n_features,)
        ElasticNet coefficients.
    gap : float
        Achieved dual gap.
    tol : float
        Equals input `tol` times `np.dot(y, y)`. The tolerance used for the dual gap.
    n_iter : int
        Number of coordinate descent iterations.
    """
    # Notes for sample_weight:
    # For dense X, one centers X and y and then rescales them by sqrt(sample_weight).
    # Here, for sparse X, we get the sample_weight averaged center X_mean. We take care
    # that every calculation results as if we had rescaled y and X (and therefore also
    # X_mean) by sqrt(sample_weight) without actually calculating the square root.
    # We work with:
    #     yw = sample_weight * y
    #     R = sample_weight * residual
    #     norm2_cols_X = np.sum(sample_weight * (X - X_mean)**2, axis=0)

    if floating is float:
        dtype = np.float32
    else:
        dtype = np.float64

    # get the data information into easy vars
    cdef unsigned int n_samples = y.shape[0]
    cdef unsigned int n_features = w.shape[0]

    # compute squared norms of the columns of X
    cdef floating[::1] norm2_cols_X = np.zeros(n_features, dtype=dtype)

    # initial value of the residuals
    # R = y - Zw, weighted version R = sample_weight * (y - Zw)
    cdef floating[::1] R
    cdef floating[::1] XtA = np.empty(n_features, dtype=dtype)
    cdef const floating[::1] yw

    cdef floating d_j
    cdef floating Xj_theta
    cdef floating tmp
    cdef floating w_j
    cdef floating d_w_max
    cdef floating w_max
    cdef floating d_w_j
    cdef floating gap = tol + 1.0
    cdef floating d_w_tol = tol
    cdef floating dual_norm_XtA
    cdef floating X_mean_j
    cdef floating R_sum = 0.0
    cdef floating normalize_sum
    cdef unsigned int n_active = n_features
    cdef uint32_t[::1] active_set
    # TODO: use binset instead of array of bools
    cdef uint8_t[::1] excluded_set
    cdef int32_t i, i_ind
    cdef unsigned int j
    cdef unsigned int n_iter = 0
    cdef unsigned int f_iter
    cdef int32_t startptr = X_indptr[0]
    cdef int32_t endptr
    cdef uint32_t rand_r_state_seed = rng.randint(0, RAND_R_MAX)
    cdef uint32_t* rand_r_state = &rand_r_state_seed
    cdef bint center = False
    cdef bint no_sample_weights = sample_weight is None

    if alpha == 0:
        # No screeing without L1-penalty.
        do_screening = False

    if do_screening:
        active_set = np.empty(n_features, dtype=np.uint32)  # map [:n_active] -> j
        excluded_set = np.empty(n_features, dtype=np.uint8)

    if no_sample_weights:
        yw = y
        R = y.copy()
    else:
        yw = np.multiply(sample_weight, y)
        R = yw.copy()

    with nogil:
        # center = (X_mean != 0).any()
        for j in range(n_features):
            if X_mean[j]:
                center = True
                break

        # R = y - np.dot(X, w)
        for j in range(n_features):
            X_mean_j = X_mean[j]
            endptr = X_indptr[j + 1]
            normalize_sum = 0.0
            w_j = w[j]

            if no_sample_weights:
                for i_ind in range(startptr, endptr):
                    i = X_indices[i_ind]
                    normalize_sum += (X_data[i_ind] - X_mean_j) ** 2
                    R[i] -= X_data[i_ind] * w_j
                norm2_cols_X[j] = normalize_sum
                if center:
                    norm2_cols_X[j] += (n_samples - endptr + startptr) * X_mean_j ** 2
                    for i in range(n_samples):
                        R[i] += X_mean_j * w_j
                        R_sum += R[i]
            else:
                # R = sw * (y - np.dot(X, w))
                for i_ind in range(startptr, endptr):
                    i = X_indices[i_ind]
                    tmp = sample_weight[i]
                    # second term will be subtracted by loop over range(n_samples)
                    normalize_sum += (tmp * (X_data[i_ind] - X_mean_j) ** 2
                                      - tmp * X_mean_j ** 2)
                    R[i] -= tmp * X_data[i_ind] * w_j
                if center:
                    for i in range(n_samples):
                        normalize_sum += sample_weight[i] * X_mean_j ** 2
                        R[i] += sample_weight[i] * X_mean_j * w_j
                        R_sum += R[i]
                norm2_cols_X[j] = normalize_sum
            startptr = endptr

        # Note: No need to update R_sum from here on because the update terms cancel
        # each other: w_j * np.sum(X[:,j] - X_mean[j]) = 0. R_sum is only ever
        # needed and calculated if X_mean is provided.

        # tol *= np.dot(y, y)
        # with sample weights: tol *= y @ (sw * y)
        tol *= _dot(n_samples, &y[0], 1, &yw[0], 1)

        # Check convergence before entering the main loop.
        gap, dual_norm_XtA = gap_enet_sparse(
            n_samples,
            n_features,
            w,
            alpha,
            beta,
            X_data,
            X_indices,
            X_indptr,
            y,
            sample_weight,
            no_sample_weights,
            X_mean,
            center,
            R,
            R_sum,
            XtA,
            positive,
        )
        if gap <= tol:
            with gil:
                return np.asarray(w), gap, tol, 0

        # Gap Safe Screening Rules, see https://arxiv.org/abs/1802.07481, Eq. 11
        if do_screening:
            n_active = 0
            for j in range(n_features):
                if norm2_cols_X[j] == 0:
                    w[j] = 0
                    excluded_set[j] = 1
                    continue
                Xj_theta = XtA[j] / fmax(alpha, dual_norm_XtA)  # X[:,j] @ dual_theta
                d_j = (1 - fabs(Xj_theta)) / sqrt(norm2_cols_X[j] + beta)
                if d_j <= sqrt(2 * gap) / alpha:
                    # include feature j
                    active_set[n_active] = j
                    excluded_set[j] = 0
                    n_active += 1
                else:
                    if w[j] != 0:
                        # R += w[j] * X[:,j]
                        R_plus_wj_Xj(
                            n_samples,
                            R,
                            X_data,
                            X_indices,
                            X_indptr,
                            X_mean,
                            center,
                            sample_weight,
                            no_sample_weights,
                            w[j],
                            j,
                        )
                        w[j] = 0
                    excluded_set[j] = 1

        for n_iter in range(max_iter):
            w_max = 0.0
            d_w_max = 0.0
            for f_iter in range(n_active):  # Loop over coordinates
                if random:
                    j = rand_int(n_active, rand_r_state)
                else:
                    j = f_iter

                if do_screening:
                    j = active_set[j]

                if norm2_cols_X[j] == 0.0:
                    continue

                startptr = X_indptr[j]
                endptr = X_indptr[j + 1]
                w_j = w[j]  # Store previous value
                X_mean_j = X_mean[j]

                # tmp = X[:,j] @ (R + w_j * X[:,j])
                tmp = 0.0
                for i_ind in range(startptr, endptr):
                    i = X_indices[i_ind]
                    tmp += R[i] * X_data[i_ind]
                tmp += w_j * norm2_cols_X[j]

                if center:
                    tmp -= R_sum * X_mean_j

                if positive and tmp < 0.0:
                    w[j] = 0.0
                else:
                    w[j] = fsign(tmp) * fmax(fabs(tmp) - alpha, 0) \
                            / (norm2_cols_X[j] + beta)

                if w[j] != w_j:
                    # R -=  (w[j] - w_j) * X[:,j] # Update residual
                    R_plus_wj_Xj(
                        n_samples,
                        R,
                        X_data,
                        X_indices,
                        X_indptr,
                        X_mean,
                        center,
                        sample_weight,
                        no_sample_weights,
                        w_j - w[j],
                        j,
                    )

                # update the maximum absolute coefficient update
                d_w_j = fabs(w[j] - w_j)
                d_w_max = fmax(d_w_max, d_w_j)

                w_max = fmax(w_max, fabs(w[j]))

            if w_max == 0.0 or d_w_max / w_max <= d_w_tol or n_iter == max_iter - 1:
                # the biggest coordinate update of this iteration was smaller than
                # the tolerance: check the duality gap as ultimate stopping
                # criterion
                gap, dual_norm_XtA = gap_enet_sparse(
                    n_samples,
                    n_features,
                    w,
                    alpha,
                    beta,
                    X_data,
                    X_indices,
                    X_indptr,
                    y,
                    sample_weight,
                    no_sample_weights,
                    X_mean,
                    center,
                    R,
                    R_sum,
                    XtA,
                    positive,
                )

                if gap <= tol:
                    # return if we reached desired tolerance
                    break

                # Gap Safe Screening Rules, see https://arxiv.org/abs/1802.07481, Eq. 11
                if do_screening:
                    n_active = 0
                    for j in range(n_features):
                        if excluded_set[j]:
                            continue
                        Xj_theta = XtA[j] / fmax(alpha, dual_norm_XtA)  # X @ dual_theta
                        d_j = (1 - fabs(Xj_theta)) / sqrt(norm2_cols_X[j] + beta)
                        if d_j <= sqrt(2 * gap) / alpha:
                            # include feature j
                            active_set[n_active] = j
                            excluded_set[j] = 0
                            n_active += 1
                        else:
                            if w[j] != 0:
                                # R += w[j] * X[:,j]
                                R_plus_wj_Xj(
                                    n_samples,
                                    R,
                                    X_data,
                                    X_indices,
                                    X_indptr,
                                    X_mean,
                                    center,
                                    sample_weight,
                                    no_sample_weights,
                                    w[j],
                                    j,
                                )
                                w[j] = 0
                            excluded_set[j] = 1

        else:
            # for/else, runs if for doesn't end with a `break`
            with gil:
                message = (
                    message_conv +
                    f" Duality gap: {gap:.6e}, tolerance: {tol:.3e}"
                )
                if alpha < np.finfo(np.float64).eps:
                    message += "\n" + message_ridge
                warnings.warn(message, ConvergenceWarning)

    return np.asarray(w), gap, tol, n_iter + 1


cdef (floating, floating) gap_enet_gram(
    int n_features,
    const floating[::1] w,
    floating alpha,  # L1 penalty
    floating beta,  # L2 penalty
    const floating[::1] Qw,
    const floating[::1] q,
    const floating y_norm2,
    floating[::1] XtA,  # XtA = X.T @ R - beta * w is calculated inplace
    bint positive,
) noexcept nogil:
    # [FUNCTION DOC]
    # HELPER: duality gap for enet_coordinate_descent_gram.
    # Behavior: expresses all residual quantities through Q = X'X, q = X'y and ||y||^2:
    # ||R||^2 = ||y||^2 + w'Qw - 2 q'w, R'y = ||y||^2 - q'w and X'R = q - Qw; then applies the same
    # three regimes as gap_enet.
    # Inputs: n_features; w, coefficients; alpha, beta; Qw, current Q @ w; q, X'y; y_norm2, ||y||^2;
    # XtA, work buffer; positive.
    # Outputs: (gap, dual_norm_XtA); XtA overwritten. Exceptions: none. Side effects: writes XtA.
    # KNOWN ISSUE: the docstring says it is used in enet_coordinate_descent; it is used by
    # enet_coordinate_descent_gram. Rounding can make the computed gap negative for singular Q.
    """Compute dual gap for use in enet_coordinate_descent.

    alpha > 0:            formulation A of the duality gap
    alpha = 0 & beta > 0: formulation B of the duality gap
    alpha = beta = 0:     OLS first order condition (=gradient)
    """
    cdef floating gap = 0.0
    cdef floating dual_norm_XtA
    cdef floating R_norm2
    cdef floating Ry
    cdef floating w_l1_norm
    cdef floating w_l2_norm2 = 0.0
    cdef floating q_dot_w
    cdef floating wQw
    cdef unsigned int j

    # w_l2_norm2 = w @ w
    if beta > 0:
        w_l2_norm2 = _dot(n_features, &w[0], 1, &w[0], 1)
    # q_dot_w = w @ q
    q_dot_w = _dot(n_features, &w[0], 1, &q[0], 1)
    # wQw = w @ Q @ w
    wQw = _dot(n_features, &w[0], 1, &Qw[0], 1)
    # R_norm2 = R @ R, residual R = y - Xw
    R_norm2 = y_norm2 + wQw - 2.0 * q_dot_w
    # Ry = R @ y
    if not (alpha == 0 and beta == 0):
        # Note that R'y = (y - Xw)' y = ||y||_2^2 - w'X'y = y_norm2 - q_dot_w
        Ry = y_norm2 - q_dot_w

    if alpha == 0:
        # XtA = X'R
        for j in range(n_features):
            XtA[j] = q[j] - Qw[j]
        # ||X'R||_2^2
        dual_norm_XtA = _dot(n_features, &XtA[0], 1, &XtA[0], 1)
        if beta == 0:
            # This is OLS, no dual gap available. Resort to first order condition
            #     X'R = 0
            #     gap = ||X'R||_2^2
            # Compare with stopping criterion of LSQR.
            gap = dual_norm_XtA
            return gap, dual_norm_XtA
        # This is Ridge regression, we use formulation B for the dual gap.
        gap = R_norm2 + 0.5 * beta * w_l2_norm2 - Ry
        gap += 1 / (2 * beta) * dual_norm_XtA
        return gap, dual_norm_XtA

    # XtA = X.T @ R - beta * w = X.T @ y - X.T @ X @ w - beta * w
    for j in range(n_features):
        XtA[j] = q[j] - Qw[j] - beta * w[j]

    # dual_norm_XtA
    if positive:
        dual_norm_XtA = max(n_features, &XtA[0])
    else:
        dual_norm_XtA = abs_max(n_features, &XtA[0])

    # w_l1_norm = np.sum(np.abs(w))
    w_l1_norm = _asum(n_features, &w[0], 1)

    gap = dual_gap_formulation_A(
        alpha=alpha,
        beta=beta,
        w_l1_norm=w_l1_norm,
        w_l2_norm2=w_l2_norm2,
        R_norm2=R_norm2,
        Ry=Ry,
        dual_norm_XtA=dual_norm_XtA,
    )
    return gap, dual_norm_XtA


def enet_coordinate_descent_gram(
    floating[::1] w,
    floating alpha,
    floating beta,
    const floating[:, ::1] Q,
    const floating[::1] q,
    const floating[:] y,
    unsigned int max_iter,
    floating tol,
    object rng,
    bint random=0,
    bint positive=0,
    bint do_screening=1,
):
    # [FUNCTION DOC]
    # Behavior: Elastic-Net coordinate descent on the Gram formulation. Maintains Qw = Q @ w
    # (initialized with numpy, updated by axpy with row j of Q when w_j changes). Coordinate update:
    # w_j = soft_threshold(q_j - Qw_j + w_j Q_jj, alpha) / (Q_jj + beta), with the positive
    # constraint as in the dense solver; features with Q_jj == 0 are skipped. tol is scaled by
    # ||y||^2. Screening is identical to the dense solver with ||X_j||^2 replaced by Q_jj and the
    # screening radius computed from |gap| to tolerate negative rounding.
    # Inputs: w, coefficients updated in place; alpha, beta; Q, C-ordered (n_features, n_features)
    # Gram matrix; q, X'y (n_features,); y, target (any stride) used only for ||y||^2; max_iter, tol,
    # rng, random, positive, do_screening as in the dense solver.
    # Outputs: (w, gap, scaled tol, number of epochs).
    # Exceptions: allocation errors; ConvergenceWarning when not converged.
    # Side effects: mutates w; releases the GIL.
    # KNOWN ISSUE: the early exit before the loop requires 0 <= gap <= tol, while the in-loop check
    # accepts any gap <= tol, including negative values caused by a singular Q.
    # KNOWN ISSUE: same max_iter == 0 behavior as enet_coordinate_descent (warns, returns 1).
    """Cython version of the coordinate descent algorithm
        for Elastic-Net regression

        We minimize

        (1/2) * w^T Q w - q^T w + alpha norm(w, 1) + (beta/2) * norm(w, 2)^2
        +1/2 * y^T y

        which amount to the Elastic-Net problem when:
        Q = X^T X (Gram matrix)
        q = X^T y

    Returns
    -------
    w : ndarray of shape (n_features,)
        ElasticNet coefficients.
    gap : float
        Achieved dual gap.
    tol : float
        Equals input `tol` times `np.dot(y, y)`. The tolerance used for the dual gap.
    n_iter : int
        Number of coordinate descent iterations.
    """

    if floating is float:
        dtype = np.float32
    else:
        dtype = np.float64

    # get the data information into easy vars
    cdef unsigned int n_features = Q.shape[0]

    # initial value "Q w" which will be kept of up to date in the iterations
    cdef floating[::1] Qw = np.dot(Q, w)
    cdef floating[::1] XtA = np.zeros(n_features, dtype=dtype)
    cdef floating y_norm2 = np.dot(y, y)

    cdef floating d_j
    cdef floating radius
    cdef floating Xj_theta
    cdef floating tmp
    cdef floating w_j
    cdef floating d_w_max
    cdef floating w_max
    cdef floating d_w_j
    cdef floating gap = tol + 1.0
    cdef floating d_w_tol = tol
    cdef floating dual_norm_XtA
    cdef unsigned int n_active = n_features
    cdef uint32_t[::1] active_set
    # TODO: use binset instead of array of bools
    cdef uint8_t[::1] excluded_set
    cdef unsigned int j
    cdef unsigned int n_iter = 0
    cdef unsigned int f_iter
    cdef uint32_t rand_r_state_seed = rng.randint(0, RAND_R_MAX)
    cdef uint32_t* rand_r_state = &rand_r_state_seed

    if alpha == 0:
        # No screeing without L1-penalty.
        do_screening = False

    if do_screening:
        active_set = np.empty(n_features, dtype=np.uint32)  # map [:n_active] -> j
        excluded_set = np.empty(n_features, dtype=np.uint8)

    with nogil:
        tol *= y_norm2

        # Check convergence before entering the main loop.
        gap, dual_norm_XtA = gap_enet_gram(
            n_features, w, alpha, beta, Qw, q, y_norm2, XtA, positive
        )
        if 0 <= gap <= tol:
            # Only if gap >=0 as singular Q may cause dubious values of gap.
            with gil:
                return np.asarray(w), gap, tol, 0

        # Gap Safe Screening Rules, see https://arxiv.org/abs/1802.07481, Eq. 11
        if do_screening:
            # Due to floating point issues, gap might be negative.
            radius = sqrt(2 * fabs(gap)) / alpha
            n_active = 0
            for j in range(n_features):
                if Q[j, j] == 0:
                    w[j] = 0
                    excluded_set[j] = 1
                    continue
                Xj_theta = XtA[j] / fmax(alpha, dual_norm_XtA)  # X[:,j] @ dual_theta
                d_j = (1 - fabs(Xj_theta)) / sqrt(Q[j, j] + beta)
                if d_j <= radius:
                    # include feature j
                    active_set[n_active] = j
                    excluded_set[j] = 0
                    n_active += 1
                else:
                    if w[j] != 0:
                        # Qw -= w[j] * Q[j]  # Update Qw = Q @ w
                        _axpy(n_features, -w[j], &Q[j, 0], 1, &Qw[0], 1)
                        w[j] = 0
                    excluded_set[j] = 1

        for n_iter in range(max_iter):
            w_max = 0.0
            d_w_max = 0.0
            for f_iter in range(n_active):  # Loop over coordinates
                if random:
                    j = rand_int(n_active, rand_r_state)
                else:
                    j = f_iter

                if do_screening:
                    j = active_set[j]

                if Q[j, j] == 0.0:
                    continue

                w_j = w[j]  # Store previous value

                # if Q = X.T @ X then tmp = X[:,j] @ (y - X @ w + X[:, j] * w_j)
                tmp = q[j] - Qw[j] + w_j * Q[j, j]

                if positive and tmp < 0:
                    w[j] = 0.0
                else:
                    w[j] = fsign(tmp) * fmax(fabs(tmp) - alpha, 0) \
                        / (Q[j, j] + beta)

                if w[j] != w_j:
                    # Qw += (w[j] - w_j) * Q[j]  # Update Qw = Q @ w
                    _axpy(n_features, w[j] - w_j, &Q[j, 0], 1, &Qw[0], 1)

                # update the maximum absolute coefficient update
                d_w_j = fabs(w[j] - w_j)
                if d_w_j > d_w_max:
                    d_w_max = d_w_j

                if fabs(w[j]) > w_max:
                    w_max = fabs(w[j])

            if w_max == 0.0 or d_w_max / w_max <= d_w_tol or n_iter == max_iter - 1:
                # the biggest coordinate update of this iteration was smaller than
                # the tolerance: check the duality gap as ultimate stopping
                # criterion
                gap, dual_norm_XtA = gap_enet_gram(
                    n_features, w, alpha, beta, Qw, q, y_norm2, XtA, positive
                )

                if gap <= tol:
                    # return if we reached desired tolerance
                    break

                # Gap Safe Screening Rules, see https://arxiv.org/abs/1802.07481, Eq. 11
                if do_screening:
                    # Due to floating point issues, gap might be negative.
                    radius = sqrt(2 * fabs(gap)) / alpha
                    n_active = 0
                    for j in range(n_features):
                        if excluded_set[j]:
                            continue
                        Xj_theta = XtA[j] / fmax(alpha, dual_norm_XtA)  # X @ dual_theta
                        d_j = (1 - fabs(Xj_theta)) / sqrt(Q[j, j] + beta)
                        if d_j <= radius:
                            # include feature j
                            active_set[n_active] = j
                            excluded_set[j] = 0
                            n_active += 1
                        else:
                            if w[j] != 0:
                                # Qw -= w[j] * Q[j]  # Update Qw = Q @ w
                                _axpy(n_features, -w[j], &Q[j, 0], 1, &Qw[0], 1)
                                w[j] = 0
                            excluded_set[j] = 1

        else:
            # for/else, runs if for doesn't end with a `break`
            with gil:
                message = (
                    message_conv +
                    f" Duality gap: {gap:.6e}, tolerance: {tol:.3e}"
                )
                if alpha < np.finfo(np.float64).eps:
                    message += "\n" + message_ridge
                warnings.warn(message, ConvergenceWarning)

    return np.asarray(w), gap, tol, n_iter + 1


cdef (floating, floating) gap_enet_multi_task(
    int n_samples,
    int n_features,
    int n_tasks,
    const floating[::1, :] W,
    floating alpha,
    floating beta,
    const floating[::1, :] X,
    bint X_is_sparse,
    const floating[::1] X_data,
    const int32_t[::1] X_indices,
    const int32_t[::1] X_indptr,
    const floating[::1, :] Y,
    const floating[::1] sample_weight,
    bint no_sample_weights,
    const floating[::1] X_mean,
    bint center,
    const floating[::1, :] R,  # current residuals = y - X @ w
    const floating[::1] R_sum,
    floating[:, ::1] XtA,  # out
    floating[::1] XtA_row_norms,  # out
) noexcept nogil:
    # [FUNCTION DOC]
    # HELPER: duality gap for enet_coordinate_descent_multi_task.
    # Behavior: squared Frobenius norms of W (if beta > 0) and R (weighted by 1/sw for sparse X with
    # weights), R'Y as an elementwise sum. XtA[j, t] = X_j'R_t (dense dot or sparse_dot minus
    # X_mean[j] R_sum[t] when centered), minus beta W[t, j] when alpha > 0. For alpha == 0 the OLS
    # and formulation B regimes of gap_enet are used. Otherwise the dual norm is the maximum row L2
    # norm of XtA, the penalty norm is the L2,1 norm (sum over features of the L2 norm of W[:, j]),
    # and dual_gap_formulation_A gives the gap.
    # Inputs: n_samples, n_features, n_tasks; W (n_tasks, n_features) Fortran order; alpha, beta;
    # X dense (ignored if X_is_sparse) or CSC arrays X_data, X_indices, X_indptr; Y (n_samples,
    # n_tasks); sample_weight, no_sample_weights; X_mean, center; R, residuals; R_sum, per-task
    # residual sums (sparse only); XtA (n_features, n_tasks) and XtA_row_norms (n_features,) outputs.
    # Outputs: (gap, dual_norm_XtA); XtA always written; XtA_row_norms written only when alpha > 0.
    # Exceptions: none. Side effects: writes XtA and XtA_row_norms.
    """Compute dual gap for use in enet_coordinate_descent_multi_task.

    Parameters
    ----------
    W : memoryview of shape (n_tasks, n_features)
    X : memoryview of shape (n_samples, n_features)
    Y : memoryview of shape (n_samples, n_tasks)
    R : memoryview of shape (n_samples, n_tasks)
        Current residuals = Y - X @ W.T
    XtA : memoryview of shape (n_features, n_tasks)
        Inplace calculated as XtA = X.T @ R - beta * W.T
    XtA_row_norms : memoryview of shape n_features
        Inplace calculated as np.sqrt(np.sum(XtA ** 2, axis=1))
    """
    cdef floating gap = 0.0
    cdef floating dual_norm_XtA
    cdef floating R_norm2
    cdef floating Ry
    cdef floating w_l21_norm
    cdef floating w_l2_norm2 = 0.0
    cdef unsigned int t, j
    cdef int32_t i

    # w_l2_norm2 = linalg.norm(W, ord="fro") ** 2
    if beta > 0:
        w_l2_norm2 = _dot(n_features * n_tasks, &W[0, 0], 1, &W[0, 0], 1)
    # R_norm2 = linalg.norm(R, ord="fro") ** 2
    if not X_is_sparse or no_sample_weights:
        R_norm2 = _dot(n_samples * n_tasks, &R[0, 0], 1, &R[0, 0], 1)
    else:  # sparse X and sample_weights
        R_norm2 = 0.0
        for t in range(n_tasks):
            for i in range(n_samples):
                # R is already multiplied by sample_weight
                if sample_weight[i] != 0:
                    R_norm2 += (R[i, t] ** 2) / sample_weight[i]
    # Ry = np.sum(R * Y)
    if not (alpha == 0 and beta == 0):
        # Note that with sample_weight, R equals R*sw and y is just y, such that
        # Ry = (sw * R) @ y, as it should be.
        Ry = _dot(n_samples * n_tasks, &R[0, 0], 1, &Y[0, 0], 1)

    if alpha == 0:
        # XtA = X.T @ R
        for j in range(n_features):
            for t in range(n_tasks):
                if not X_is_sparse:
                    XtA[j, t] = _dot(n_samples, &X[0, j], 1, &R[0, t], 1)
                else:
                    XtA[j, t] = sparse_dot(j, X_data, X_indices, X_indptr, R[:, t])
                    if center:
                        XtA[j, t] -= X_mean[j] * R_sum[t]
        # ||X'R||_2^2
        dual_norm_XtA = _dot(n_features * n_tasks, &XtA[0, 0], 1, &XtA[0, 0], 1)
        if beta == 0:
            # This is OLS, no dual gap available. Resort to first order condition
            #     X'R = 0
            #     gap = ||X'R||_2^2
            # Compare with stopping criterion of LSQR.
            gap = dual_norm_XtA
            return gap, dual_norm_XtA
        # This is Ridge regression, we use formulation B for the dual gap.
        gap = R_norm2 + 0.5 * beta * w_l2_norm2 - Ry
        gap += 1 / (2 * beta) * dual_norm_XtA
        return gap, dual_norm_XtA

    # XtA = X.T @ R - beta * W.T
    for j in range(n_features):
        for t in range(n_tasks):
            if not X_is_sparse:
                XtA[j, t] = _dot(n_samples, &X[0, j], 1, &R[0, t], 1) - beta * W[t, j]
            else:
                XtA[j, t] = sparse_dot(j, X_data, X_indices, X_indptr, R[:, t])
                if center:
                    XtA[j, t] -= X_mean[j] * R_sum[t]
                XtA[j, t] -= beta * W[t, j]

    # dual_norm_XtA = np.max(np.sqrt(np.sum(XtA ** 2, axis=1)))
    dual_norm_XtA = 0.0
    for j in range(n_features):
        # np.sqrt(np.sum(XtA ** 2, axis=1))
        XtA_row_norms[j] = _nrm2(n_tasks, &XtA[j, 0], 1)
        if XtA_row_norms[j] > dual_norm_XtA:
            dual_norm_XtA = XtA_row_norms[j]

    # w_l21_norm = np.sqrt(np.sum(W ** 2, axis=0)).sum()
    w_l21_norm = 0.0
    for ii in range(n_features):
        w_l21_norm += _nrm2(n_tasks, &W[0, ii], 1)

    gap = dual_gap_formulation_A(
        alpha=alpha,
        beta=beta,
        w_l1_norm=w_l21_norm,
        w_l2_norm2=w_l2_norm2,
        R_norm2=R_norm2,
        Ry=Ry,
        dual_norm_XtA=dual_norm_XtA,
    )
    return gap, dual_norm_XtA


def enet_coordinate_descent_multi_task(
    floating[::1, :] W,
    floating alpha,
    floating beta,
    const floating[::1, :] X,
    bint X_is_sparse,
    const floating[::1] X_data,
    const int32_t[::1] X_indices,
    const int32_t[::1] X_indptr,
    const floating[::1, :] Y,
    const floating[::1] sample_weight,
    const floating[::1] X_mean,
    unsigned int max_iter,
    floating tol,
    object rng,
    bint random=0,
    bint do_screening=1,
):
    # [FUNCTION DOC]
    # Behavior: block coordinate descent for the multi-task Elastic-Net with L2,1 penalty. For dense
    # X the caller has already applied centering and sqrt(sample_weight) scaling; for CSC X,
    # centering by X_mean and weights are handled implicitly as in sparse_enet_coordinate_descent
    # (R holds weighted residuals, R_sum per task). For each active feature j the block gradient
    # tmp = X_j'R + W[:, j] ||X_j||^2 is computed per task and W[:, j] is set by group soft
    # thresholding: tmp * max(1 - alpha / ||tmp||_2, 0) / (||X_j||^2 + beta); R is updated per task
    # where the coefficient changed. Convergence uses max abs change over max abs weight, then the
    # duality gap with tol scaled by sum(Y * Yw). Gap Safe screening uses the row norms of XtA
    # (multi-task rule) and permanently excludes features.
    # Inputs: W, (n_tasks, n_features) Fortran-ordered coefficients, updated in place; alpha, beta;
    # X, dense Fortran design (dummy when sparse); X_is_sparse; X_data, X_indices, X_indptr, CSC
    # arrays (dummies when dense); Y, (n_samples, n_tasks) Fortran-ordered targets; sample_weight or
    # None (used only for sparse X); X_mean, column offsets for sparse X; max_iter, tol, rng, random,
    # do_screening as in the single-target solvers. No positive option.
    # Outputs: (W, gap, scaled tol, number of epochs).
    # Exceptions: allocation errors; ConvergenceWarning when not converged.
    # Side effects: mutates W; allocates R, XtA and work vectors; releases the GIL for the main loop.
    # KNOWN ISSUE: as in sparse_enet_coordinate_descent, R_sum is accumulated once per feature during
    # initialization and equals sum(R) only because the caller passes centered Y with nonzero X_mean.
    # KNOWN ISSUE: same max_iter == 0 behavior as enet_coordinate_descent (warns, returns 1).
    """Cython version of the coordinate descent algorithm
        for Elastic-Net multi-task regression

        We minimize

        0.5 * norm(Y - X W.T, 2)^2 + alpha * ||W.T||_21 + 0.5 * beta * norm(W.T, 2)^2

    The algorithm follows
    Noah Simon, Jerome Friedman, Trevor Hastie. 2013.
    A Blockwise Descent Algorithm for Group-penalized Multiresponse and Multinomial
    Regression
    https://doi.org/10.48550/arXiv.1311.6529

    Returns
    -------
    W : ndarray of shape (n_tasks, n_features)
        ElasticNet coefficients.
    gap : float
        Achieved dual gap.
    tol : float
        Equals input `tol` times `np.dot(y, y)`. The tolerance used for the dual gap.
    n_iter : int
        Number of coordinate descent iterations.
    """
    # Notes for sample_weight:
    # For dense X, one centers X and y and then rescales them by sqrt(sample_weight).
    # For sparse X, we get the sample_weight averaged center X_mean. We take care
    # that every calculation results as if we had rescaled y and X (and therefore also
    # X_mean) by sqrt(sample_weight) without actually calculating the square root.
    # We work with:
    #     yw = sample_weight * y
    #     R = sample_weight * residual
    #     norm2_cols_X = np.sum(sample_weight * (X - X_mean)**2, axis=0)

    if floating is float:
        dtype = np.float32
    else:
        dtype = np.float64

    # get the data information into easy vars
    cdef unsigned int n_samples = Y.shape[0]
    cdef unsigned int n_features = W.shape[1]
    cdef unsigned int n_tasks = Y.shape[1]

    # compute squared norms of the columns of X
    norm2_cols_X_array = np.empty(shape=n_features, dtype=dtype)
    cdef floating[::1] norm2_cols_X = norm2_cols_X_array

    # initial value of the residuals
    cdef floating[::1, :] R  # shape (n_samples, n_tasks)
    cdef floating[:, ::1] XtA = np.empty((n_features, n_tasks), dtype=dtype)
    cdef floating[::1] XtA_row_norms = np.empty(n_features, dtype=dtype)
    cdef const floating[::1, :] Yw

    cdef floating d_j
    cdef floating Xj_theta
    cdef floating[::1] tmp = np.empty(n_tasks, dtype=dtype)
    cdef floating[::1] w_j = np.empty(n_tasks, dtype=dtype)
    cdef floating d_w_max
    cdef floating w_max
    cdef floating d_w_j
    cdef floating nn
    cdef floating W_j_abs_max
    cdef floating gap = tol + 1.0
    cdef floating d_w_tol = tol
    cdef floating dual_norm_XtA
    cdef floating[::1] R_sum
    cdef unsigned int n_active = n_features
    cdef uint32_t[::1] active_set
    # TODO: use binset instead of array of bools
    cdef uint8_t[::1] excluded_set
    cdef unsigned int j
    cdef unsigned int t
    cdef int32_t i, i_ind, startptr, endptr
    cdef unsigned int n_iter = 0
    cdef unsigned int f_iter
    cdef uint32_t rand_r_state_seed = rng.randint(0, RAND_R_MAX)
    cdef uint32_t* rand_r_state = &rand_r_state_seed
    cdef bint center = False
    cdef bint no_sample_weights = sample_weight is None

    if alpha == 0:
        # No screeing without L1-penalty.
        do_screening = False

    if do_screening:
        active_set = np.empty(n_features, dtype=np.uint32)  # map [:n_active] -> j
        excluded_set = np.empty(n_features, dtype=np.uint8)

    if no_sample_weights or not X_is_sparse:
        Yw = Y
        R = np.copy(Y, order="F")
    else:
        Yw = np.multiply(sample_weight[:, None], Y)
        R = np.copy(Yw, order="F")

    if X_is_sparse:
        R_sum = np.zeros(n_tasks, dtype=dtype)
        # center = (X_mean != 0).any()
        for j in range(n_features):
            if X_mean[j]:
                center = True
                break

    # compute squared norms of the columns of X
    # same as norm2_cols_X = np.square(X).sum(axis=0)
    if not X_is_sparse:
        np.einsum("ij,ij->j", X, X, dtype=dtype, out=norm2_cols_X_array)
    else:
        for j in range(n_features):
            norm2_cols_X[j] = 0
            startptr = X_indptr[j]
            endptr = X_indptr[j + 1]
            if no_sample_weights:
                for i_ind in range(startptr, endptr):
                    norm2_cols_X[j] += (X_data[i_ind] - X_mean[j]) ** 2
                if center:
                    norm2_cols_X[j] += (n_samples - endptr + startptr) * X_mean[j] ** 2
            else:
                for i_ind in range(startptr, endptr):
                    i = X_indices[i_ind]
                    norm2_cols_X[j] += sample_weight[i] * (
                        (X_data[i_ind] - X_mean[j]) ** 2 - X_mean[j] ** 2
                    )
                if center:
                    for i in range(n_samples):
                        norm2_cols_X[j] += sample_weight[i] * X_mean[j] ** 2

    with nogil:
        # R = Y - X @ W.T
        # R = Y was already set above.
        for j in range(n_features):
            for t in range(n_tasks):
                if W[t, j] != 0:
                    if not X_is_sparse:
                        _axpy(n_samples, -W[t, j], &X[0, j], 1, &R[0, t], 1)
                    else:
                        if no_sample_weights:
                            sparse_axpy(j, -W[t, j], X_data, X_indices, X_indptr, R[:, t])
                        else:
                            startptr = X_indptr[j]
                            endptr = X_indptr[j + 1]
                            for i_ind in range(startptr, endptr):
                                i = X_indices[i_ind]
                                R[i, t] -= sample_weight[i] * X_data[i_ind] * W[t, j]

            if X_is_sparse and center:
                # R = Y - (X - X_mean) @ W.T
                if no_sample_weights:
                    for i in range(n_samples):
                        for t in range(n_tasks):
                            R[i, t] += X_mean[j] * W[t, j]
                            R_sum[t] += R[i, t]
                else:
                    for i in range(n_samples):
                        for t in range(n_tasks):
                            R[i, t] += sample_weight[i] * X_mean[j] * W[t, j]
                            R_sum[t] += R[i, t]

        # Note: No need to update R_sum from here on because the update terms cancel
        # each other: w_j[t] * np.sum(X[:,j] - X_mean[j]) = 0. R_sum is only ever
        # needed and calculated if X_mean is provided.

        # tol = tol * linalg.norm(Y, ord='fro') ** 2
        # with sample weights: tol *= y @ (sw * y)
        tol *= _dot(n_samples * n_tasks, &Y[0, 0], 1, &Yw[0, 0], 1)

        # Check convergence before entering the main loop.
        gap, dual_norm_XtA = gap_enet_multi_task(
            n_samples=n_samples,
            n_features=n_features,
            n_tasks=n_tasks,
            W=W,
            alpha=alpha,
            beta=beta,
            X=X,
            X_is_sparse=X_is_sparse,
            X_data=X_data,
            X_indices=X_indices,
            X_indptr=X_indptr,
            Y=Y,
            sample_weight=sample_weight,
            no_sample_weights=no_sample_weights,
            X_mean=X_mean,
            center=center,
            R=R,
            R_sum=R_sum,
            XtA=XtA,
            XtA_row_norms=XtA_row_norms,
        )
        if gap <= tol:
            with gil:
                return np.asarray(W), gap, tol, 0

        # Gap Safe Screening Rules for multi-task Lasso, see
        # https://arxiv.org/abs/1703.07285 Eq 2.2. (also arxiv:1506.03736)
        if do_screening:
            n_active = 0
            for j in range(n_features):
                if norm2_cols_X[j] == 0:
                    for t in range(n_tasks):
                        W[t, j] = 0
                    excluded_set[j] = 1
                    continue
                # Xj_theta = ||X[:,j] @ dual_theta||_2
                Xj_theta = XtA_row_norms[j] / fmax(alpha, dual_norm_XtA)
                d_j = (1 - Xj_theta) / sqrt(norm2_cols_X[j] + beta)
                if d_j <= sqrt(2 * gap) / alpha:
                    # include feature j
                    active_set[n_active] = j
                    excluded_set[j] = 0
                    n_active += 1
                else:
                    # R += W[:, j] * X[:, 1][:, None]
                    for t in range(n_tasks):
                        if W[t, j] != 0:
                            if not X_is_sparse:
                                _axpy(n_samples, W[t, j], &X[0, j], 1, &R[0, t], 1)
                            else:
                                R_plus_wj_Xj(
                                    n_samples=n_samples,
                                    R=R[:, t],
                                    X_data=X_data,
                                    X_indices=X_indices,
                                    X_indptr=X_indptr,
                                    X_mean=X_mean,
                                    center=center,
                                    sample_weight=sample_weight,
                                    no_sample_weights=no_sample_weights,
                                    w_j=W[t, j],
                                    j=j,
                                )
                            W[t, j] = 0
                    excluded_set[j] = 1

        for n_iter in range(max_iter):
            w_max = 0.0
            d_w_max = 0.0
            for f_iter in range(n_active):  # Loop over coordinates
                if random:
                    j = rand_int(n_active, rand_r_state)
                else:
                    j = f_iter

                if do_screening:
                    j = active_set[j]

                if norm2_cols_X[j] == 0.0:
                    continue

                # w_j = W[:, j] # Store previous value
                _copy(n_tasks, &W[0, j], 1, &w_j[0], 1)

                # tmp = X[:, j] @ (R + w_j * X[:,j][:, None])
                # first part: X[:, j] @ R
                #   Using BLAS Level 2:
                #   _gemv(RowMajor, Trans, n_samples, n_tasks, 1.0, &R[0, 0],
                #         n_tasks, &X[0, j], 1, 0.0, &tmp[0], 1)
                # second part: (X[:, j] @ X[:,j]) * w_j = norm2_cols * w_j
                #   Using BLAS Level 1:
                #   _axpy(n_tasks, norm2_cols[j], &w_j[0], 1, &tmp[0], 1)
                # Using BLAS Level 1 (faster for small vectors like here):
                for t in range(n_tasks):
                    if not X_is_sparse:
                        tmp[t] = _dot(n_samples, &X[0, j], 1, &R[0, t], 1)
                    else:
                        tmp[t] = sparse_dot(j, X_data, X_indices, X_indptr, R[:, t])
                    # As we have the loop already, we use it to replace the second BLAS
                    # Level 1, i.e., _axpy, too.
                    tmp[t] += w_j[t] * norm2_cols_X[j]

                # nn = sqrt(np.sum(tmp ** 2))
                nn = _nrm2(n_tasks, &tmp[0], 1)

                # W[:, j] = tmp * fmax(1. - alpha / nn, 0) / (norm2_cols_X[j] + beta)
                _copy(n_tasks, &tmp[0], 1, &W[0, j], 1)
                _scal(n_tasks, fmax(1. - alpha / nn, 0) / (norm2_cols_X[j] + beta),
                      &W[0, j], 1)

                # Update residual
                # Using numpy:
                #   R -= (W[:, j] - w_j) * X[:, j][:, None]
                # Using BLAS Level 1 and 2:
                #   _axpy(n_tasks, -1.0, &W[0, j], 1, &w_j[0], 1)
                #   _ger(RowMajor, n_samples, n_tasks, 1.0,
                #        &X[0, j], 1, &w_j, 1,
                #        &R[0, 0], n_tasks)
                # Using BLAS Level 1 (faster for small vectors like here):
                for t in range(n_tasks):
                    if W[t, j] != w_j[t]:
                        if not X_is_sparse:
                            _axpy(n_samples, w_j[t] - W[t, j], &X[0, j], 1, &R[0, t], 1)
                        else:
                            R_plus_wj_Xj(
                                n_samples=n_samples,
                                R=R[:, t],
                                X_data=X_data,
                                X_indices=X_indices,
                                X_indptr=X_indptr,
                                X_mean=X_mean,
                                center=center,
                                sample_weight=sample_weight,
                                no_sample_weights=no_sample_weights,
                                w_j=w_j[t] - W[t, j],
                                j=j,
                            )

                # update the maximum absolute coefficient update
                d_w_j = diff_abs_max(n_tasks, &W[0, j], &w_j[0])

                if d_w_j > d_w_max:
                    d_w_max = d_w_j

                W_j_abs_max = abs_max(n_tasks, &W[0, j])
                if W_j_abs_max > w_max:
                    w_max = W_j_abs_max

            if w_max == 0.0 or d_w_max / w_max <= d_w_tol or n_iter == max_iter - 1:
                # the biggest coordinate update of this iteration was smaller than
                # the tolerance: check the duality gap as ultimate stopping
                # criterion
                gap, dual_norm_XtA = gap_enet_multi_task(
                    n_samples=n_samples,
                    n_features=n_features,
                    n_tasks=n_tasks,
                    W=W,
                    alpha=alpha,
                    beta=beta,
                    X=X,
                    X_is_sparse=X_is_sparse,
                    X_data=X_data,
                    X_indices=X_indices,
                    X_indptr=X_indptr,
                    Y=Y,
                    sample_weight=sample_weight,
                    no_sample_weights=no_sample_weights,
                    X_mean=X_mean,
                    center=center,
                    R=R,
                    R_sum=R_sum,
                    XtA=XtA,
                    XtA_row_norms=XtA_row_norms,
                )
                if gap <= tol:
                    # return if we reached desired tolerance
                    break

                # Gap Safe Screening Rules for multi-task Lasso, see
                # https://arxiv.org/abs/1703.07285 Eq 2.2. (also arxiv:1506.03736)
                if do_screening:
                    n_active = 0
                    for j in range(n_features):
                        if excluded_set[j]:
                            continue
                        # Xj_theta = ||X[:,j] @ dual_theta||_2
                        Xj_theta = XtA_row_norms[j] / fmax(alpha, dual_norm_XtA)
                        d_j = (1 - Xj_theta) / sqrt(norm2_cols_X[j] + beta)
                        if d_j <= sqrt(2 * gap) / alpha:
                            # include feature j
                            active_set[n_active] = j
                            excluded_set[j] = 0
                            n_active += 1
                        else:
                            # R += W[:, j] * X[:, 1][:, None]
                            for t in range(n_tasks):
                                if W[t, j] != 0:
                                    if not X_is_sparse:
                                        _axpy(n_samples, W[t, j], &X[0, j], 1, &R[0, t], 1)
                                    else:
                                        R_plus_wj_Xj(
                                            n_samples=n_samples,
                                            R=R[:, t],
                                            X_data=X_data,
                                            X_indices=X_indices,
                                            X_indptr=X_indptr,
                                            X_mean=X_mean,
                                            center=center,
                                            sample_weight=sample_weight,
                                            no_sample_weights=no_sample_weights,
                                            w_j=W[t, j],
                                            j=j,
                                        )
                                    W[t, j] = 0
                            excluded_set[j] = 1

        else:
            # for/else, runs if for doesn't end with a `break`
            with gil:
                message = (
                    message_conv +
                    f" Duality gap: {gap:.6e}, tolerance: {tol:.3e}"
                )
                if alpha < np.finfo(np.float64).eps:
                    message += "\n" + message_ridge
                warnings.warn(message, ConvergenceWarning)

    return np.asarray(W), gap, tol, n_iter + 1
