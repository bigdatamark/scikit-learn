# [MODULE DOC]
# Graphical lasso: sparse inverse covariance (precision) estimation by l1-penalized
# Gaussian maximum likelihood, solved with the GLasso block coordinate algorithm (one
# Lasso per feature, via coordinate descent or LARS).
# Public contents: function graphical_lasso, graphical_lasso_path, estimators
# GraphicalLasso and GraphicalLassoCV (CV grid search with iterative grid refinement);
# private helpers _objective, _dual_gap, _graphical_lasso, alpha_max.
"""GraphicalLasso: sparse inverse covariance estimation with an l1-penalized
estimator.
"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import operator
import sys
import time
import warnings
from numbers import Integral, Real

import numpy as np
from scipy import linalg

from sklearn.base import _fit_context
from sklearn.covariance import EmpiricalCovariance, empirical_covariance, log_likelihood
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import _cd_fast as cd_fast
from sklearn.linear_model import lars_path_gram
from sklearn.model_selection import check_cv, cross_val_score
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
from sklearn.utils.validation import (
    _is_arraylike_not_scalar,
    check_random_state,
    check_scalar,
    validate_data,
)


# Helper functions to compute the objective and dual objective functions
# of the l1-penalized estimator
def _objective(mle, precision_, alpha):
    # [FUNCTION DOC]
    # HELPER: Computes the graphical-lasso primal objective: minus two times the
    # Gaussian log-likelihood of mle under precision_, plus p*log(2*pi), plus alpha
    # times the l1 norm of the off-diagonal entries of precision_.
    # Inputs: mle - (p, p) empirical covariance; precision_ - (p, p) precision; alpha -
    # float.
    # Outputs: float cost. Side effects: none.
    """Evaluation of the graphical-lasso objective function

    the objective function is made of a shifted scaled version of the
    normalized log-likelihood (i.e. its empirical mean over the samples) and a
    penalisation term to promote sparsity
    """
    p = precision_.shape[0]
    cost = -2.0 * log_likelihood(mle, precision_) + p * np.log(2 * np.pi)
    cost += alpha * (np.abs(precision_).sum() - np.abs(np.diag(precision_)).sum())
    return cost


def _dual_gap(emp_cov, precision_, alpha):
    # [FUNCTION DOC]
    # HELPER: Computes the duality gap used as convergence criterion: trace of emp_cov
    # times precision_ (elementwise product sum), minus p, plus alpha times the l1 norm
    # of off-diagonal precision_ entries.
    # Inputs: emp_cov, precision_ - (p, p) arrays; alpha - float. Outputs: float gap.
    """Expression of the dual gap convergence criterion

    The specific definition is given in Duchi "Projected Subgradient Methods
    for Learning Sparse Gaussians".
    """
    gap = np.sum(emp_cov * precision_)
    gap -= precision_.shape[0]
    gap += alpha * (np.abs(precision_).sum() - np.abs(np.diag(precision_)).sum())
    return gap


# The g-lasso algorithm
def _graphical_lasso(
    emp_cov,
    alpha,
    *,
    cov_init=None,
    mode="cd",
    tol=1e-4,
    enet_tol=1e-4,
    max_iter=100,
    verbose=False,
    eps=np.finfo(np.float64).eps,
):
    # [FUNCTION DOC]
    # HELPER: Core GLasso solver.
    # Behavior: if alpha is 0, return emp_cov and its exact inverse directly. Otherwise
    # start from cov_init (or emp_cov), shrink off-diagonals by 0.95 keeping emp_cov's
    # diagonal, and set precision to its pseudo-inverse. For each outer iteration, loop
    # over every feature idx: build the submatrix of the covariance without row/column
    # idx (updated incrementally, one row/column per step), solve a Lasso of the idx-th
    # emp_cov row against it (mode 'cd': gram-based coordinate descent warm-started from
    # the scaled precision column, at least 20 inner iterations, tolerance enet_tol;
    # mode 'lars': lars_path_gram with alpha_min alpha/(p-1)), then update the precision
    # diagonal and off-diagonal column/row from the coefficients and set covariance
    # row/column idx to the submatrix times coefficients. After each sweep compute dual
    # gap and objective, record them, optionally print, and stop when the absolute gap
    # is below tol.
    # Inputs: emp_cov - (p, p) array; alpha - float >= 0; cov_init - optional (p, p)
    # start; mode - 'cd' or 'lars'; tol, enet_tol - floats; max_iter - int; verbose -
    # bool; eps - small regularizer added to the precision diagonal in cd warm start /
    # passed to lars.
    # Outputs: (covariance, precision, costs list of (cost, gap), number of iterations).
    # Exceptions: FloatingPointError if the precision becomes non-finite or the cost is
    # non-finite after the first iteration (message extended with an ill-conditioning
    # note); numpy floating errors raised under the chosen errstate.
    # Side effects: emits ConvergenceWarning when max_iter is exhausted; prints if
    # verbose.
    # KNOWN ISSUE: the alpha == 0 early return gives costs as a single (cost, gap) tuple
    # and n_iter 0, unlike the list returned otherwise.
    _, n_features = emp_cov.shape
    if alpha == 0:
        # Early return without regularization
        precision_ = linalg.inv(emp_cov)
        cost = -2.0 * log_likelihood(emp_cov, precision_)
        cost += n_features * np.log(2 * np.pi)
        d_gap = np.sum(emp_cov * precision_) - n_features
        return emp_cov, precision_, (cost, d_gap), 0

    if cov_init is None:
        covariance_ = emp_cov.copy()
    else:
        covariance_ = cov_init.copy()
    # As a trivial regularization (Tikhonov like), we scale down the
    # off-diagonal coefficients of our starting point: This is needed, as
    # in the cross-validation the cov_init can easily be
    # ill-conditioned, and the CV loop blows. Beside, this takes
    # conservative stand-point on the initial conditions, and it tends to
    # make the convergence go faster.
    covariance_ *= 0.95
    diagonal = emp_cov.flat[:: n_features + 1]
    covariance_.flat[:: n_features + 1] = diagonal
    precision_ = linalg.pinvh(covariance_)

    indices = np.arange(n_features)
    i = 0  # initialize the counter to be robust to `max_iter=0`
    costs = list()
    # The different l1 regression solver have different numerical errors
    if mode == "cd":
        errors = dict(over="raise", invalid="ignore")
    else:
        errors = dict(invalid="raise")
    try:
        # be robust to the max_iter=0 edge case, see:
        # https://github.com/scikit-learn/scikit-learn/issues/4134
        d_gap = np.inf
        # set a sub_covariance buffer
        sub_covariance = np.copy(covariance_[1:, 1:], order="C")
        for i in range(max_iter):
            for idx in range(n_features):
                # To keep the contiguous matrix `sub_covariance` equal to
                # covariance_[indices != idx].T[indices != idx]
                # we only need to update 1 column and 1 line when idx changes
                if idx > 0:
                    di = idx - 1
                    sub_covariance[di] = covariance_[di][indices != idx]
                    sub_covariance[:, di] = covariance_[:, di][indices != idx]
                else:
                    sub_covariance[:] = covariance_[1:, 1:]
                row = emp_cov[idx, indices != idx]
                with np.errstate(**errors):
                    if mode == "cd":
                        # Use coordinate descent
                        coefs = -(
                            precision_[indices != idx, idx]
                            / (precision_[idx, idx] + 1000 * eps)
                        )
                        coefs, _, _, _ = cd_fast.enet_coordinate_descent_gram(
                            w=coefs,
                            alpha=alpha,
                            beta=0,
                            Q=sub_covariance,
                            q=row,
                            y=row,
                            # TODO: It is not ideal that the max_iter of the outer
                            # solver (graphical lasso) is coupled with the max_iter of
                            # the inner solver (CD). Ideally, CD has its own parameter
                            # enet_max_iter (like enet_tol). A minimum of 20 is rather
                            # arbitrary, but not unreasonable.
                            max_iter=max(20, max_iter),
                            tol=enet_tol,
                            rng=check_random_state(None),
                            random=False,
                            positive=False,
                            do_screening=True,
                        )
                    else:  # mode == "lars"
                        _, _, coefs = lars_path_gram(
                            Xy=row,
                            Gram=sub_covariance,
                            n_samples=row.size,
                            alpha_min=alpha / (n_features - 1),
                            copy_Gram=True,
                            eps=eps,
                            method="lars",
                            return_path=False,
                        )
                # Update the precision matrix
                precision_[idx, idx] = 1.0 / (
                    covariance_[idx, idx]
                    - np.dot(covariance_[indices != idx, idx], coefs)
                )
                precision_[indices != idx, idx] = -precision_[idx, idx] * coefs
                precision_[idx, indices != idx] = -precision_[idx, idx] * coefs
                coefs = np.dot(sub_covariance, coefs)
                covariance_[idx, indices != idx] = coefs
                covariance_[indices != idx, idx] = coefs
            if not np.isfinite(precision_.sum()):
                raise FloatingPointError(
                    "The system is too ill-conditioned for this solver"
                )
            d_gap = _dual_gap(emp_cov, precision_, alpha)
            cost = _objective(emp_cov, precision_, alpha)
            if verbose:
                print(
                    "[graphical_lasso] Iteration % 3i, cost % 3.2e, dual gap %.3e"
                    % (i, cost, d_gap)
                )
            costs.append((cost, d_gap))
            if np.abs(d_gap) < tol:
                break
            if not np.isfinite(cost) and i > 0:
                raise FloatingPointError(
                    "Non SPD result: the system is too ill-conditioned for this solver"
                )
        else:
            warnings.warn(
                "graphical_lasso: did not converge after %i iteration: dual gap: %.3e"
                % (max_iter, d_gap),
                ConvergenceWarning,
            )
    except FloatingPointError as e:
        e.args = (e.args[0] + ". The system is too ill-conditioned for this solver",)
        raise e

    return covariance_, precision_, costs, i + 1


def alpha_max(emp_cov):
    # [FUNCTION DOC]
    # Returns the largest absolute off-diagonal entry of emp_cov: the smallest alpha for
    # which the graphical lasso solution has an all-zero off-diagonal precision.
    # Inputs: emp_cov - (p, p) array (copied, not modified). Outputs: float.
    """Find the maximum alpha for which there are some non-zeros off-diagonal.

    Parameters
    ----------
    emp_cov : ndarray of shape (n_features, n_features)
        The sample covariance matrix.

    Notes
    -----
    This results from the bound for the all the Lasso that are solved
    in GraphicalLasso: each time, the row of cov corresponds to Xy. As the
    bound for alpha is given by `max(abs(Xy))`, the result follows.
    """
    A = np.copy(emp_cov)
    A.flat[:: A.shape[0] + 1] = 0
    return np.max(np.abs(A))


@validate_params(
    {
        "emp_cov": ["array-like"],
        "return_costs": ["boolean"],
        "return_n_iter": ["boolean"],
    },
    prefer_skip_nested_validation=False,
)
def graphical_lasso(
    emp_cov,
    alpha,
    *,
    mode="cd",
    tol=1e-4,
    enet_tol=1e-4,
    max_iter=100,
    verbose=False,
    return_costs=False,
    eps=np.finfo(np.float64).eps,
    return_n_iter=False,
):
    # [FUNCTION DOC]
    # Functional wrapper: fits a GraphicalLasso with covariance='precomputed' and
    # assume_centered=True on emp_cov, then returns a tuple (covariance_, precision_)
    # optionally followed by costs_ (if return_costs) and n_iter_ (if return_n_iter).
    # Inputs are passed through to the estimator (see docstring for meanings/defaults).
    # Exceptions: parameter validation errors; solver FloatingPointError.
    # Side effects: as GraphicalLasso.fit (warnings, verbose printing).
    """L1-penalized covariance estimator.

    Read more in the :ref:`User Guide <sparse_inverse_covariance>`.

    .. versionchanged:: v0.20
        graph_lasso has been renamed to graphical_lasso

    Parameters
    ----------
    emp_cov : array-like of shape (n_features, n_features)
        Empirical covariance from which to compute the covariance estimate.

    alpha : float
        The regularization parameter: the higher alpha, the more
        regularization, the sparser the inverse covariance.
        Range is (0, inf].

    mode : {'cd', 'lars'}, default='cd'
        The Lasso solver to use: coordinate descent or LARS. Use LARS for
        very sparse underlying graphs, where p > n. Elsewhere prefer cd
        which is more numerically stable.

    tol : float, default=1e-4
        The tolerance to declare convergence: if the dual gap goes below
        this value, iterations are stopped. Range is (0, inf].

    enet_tol : float, default=1e-4
        The tolerance for the elastic net solver used to calculate the descent
        direction. This parameter controls the accuracy of the search direction
        for a given column update, not of the overall parameter estimate. Only
        used for mode='cd'. Range is (0, inf].

    max_iter : int, default=100
        The maximum number of iterations.

    verbose : bool, default=False
        If verbose is True, the objective function and dual gap are
        printed at each iteration.

    return_costs : bool, default=False
        If return_costs is True, the objective function and dual gap
        at each iteration are returned.

    eps : float, default=eps
        The machine-precision regularization in the computation of the
        Cholesky diagonal factors. Increase this for very ill-conditioned
        systems. Default is `np.finfo(np.float64).eps`.

    return_n_iter : bool, default=False
        Whether or not to return the number of iterations.

    Returns
    -------
    covariance : ndarray of shape (n_features, n_features)
        The estimated covariance matrix.

    precision : ndarray of shape (n_features, n_features)
        The estimated (sparse) precision matrix.

    costs : list of (objective, dual_gap) pairs
        The list of values of the objective function and the dual gap at
        each iteration. Returned only if return_costs is True.

    n_iter : int
        Number of iterations. Returned only if `return_n_iter` is set to True.

    See Also
    --------
    GraphicalLasso : Sparse inverse covariance estimation
        with an l1-penalized estimator.
    GraphicalLassoCV : Sparse inverse covariance with
        cross-validated choice of the l1 penalty.

    Notes
    -----
    The algorithm employed to solve this problem is the GLasso algorithm,
    from the Friedman 2008 Biostatistics paper. It is the same algorithm
    as in the R `glasso` package.

    One possible difference with the `glasso` R package is that the
    diagonal coefficients are not penalized.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.datasets import make_sparse_spd_matrix
    >>> from sklearn.covariance import empirical_covariance, graphical_lasso
    >>> true_cov = make_sparse_spd_matrix(n_dim=3,random_state=42)
    >>> rng = np.random.RandomState(42)
    >>> X = rng.multivariate_normal(mean=np.zeros(3), cov=true_cov, size=3)
    >>> emp_cov = empirical_covariance(X, assume_centered=True)
    >>> emp_cov, _ = graphical_lasso(emp_cov, alpha=0.05)
    >>> emp_cov
    array([[ 1.687,  0.212, -0.209],
           [ 0.212,  0.221, -0.0817],
           [-0.209, -0.0817, 0.232]])
    """
    model = GraphicalLasso(
        alpha=alpha,
        mode=mode,
        covariance="precomputed",
        tol=tol,
        enet_tol=enet_tol,
        max_iter=max_iter,
        verbose=verbose,
        eps=eps,
        assume_centered=True,
    ).fit(emp_cov)

    output = [model.covariance_, model.precision_]
    if return_costs:
        output.append(model.costs_)
    if return_n_iter:
        output.append(model.n_iter_)
    return tuple(output)


class BaseGraphicalLasso(EmpiricalCovariance):
    # [CLASS DOC]
    # HELPER: Common base for GraphicalLasso and GraphicalLassoCV. Extends
    # EmpiricalCovariance's parameter constraints with solver settings (tol, enet_tol,
    # max_iter, mode, verbose, eps) and removes store_precision (precision is always
    # stored).
    _parameter_constraints: dict = {
        **EmpiricalCovariance._parameter_constraints,
        "tol": [Interval(Real, 0, None, closed="right")],
        "enet_tol": [Interval(Real, 0, None, closed="right")],
        "max_iter": [Interval(Integral, 0, None, closed="left")],
        "mode": [StrOptions({"cd", "lars"})],
        "verbose": ["verbose"],
        "eps": [Interval(Real, 0, None, closed="both")],
    }
    _parameter_constraints.pop("store_precision")

    def __init__(
        self,
        tol=1e-4,
        enet_tol=1e-4,
        max_iter=100,
        mode="cd",
        verbose=False,
        eps=np.finfo(np.float64).eps,
        assume_centered=False,
    ):
        # [FUNCTION DOC]
        # Stores solver hyperparameters as attributes without validation;
        # assume_centered is forwarded to EmpiricalCovariance.
        super().__init__(assume_centered=assume_centered)
        self.tol = tol
        self.enet_tol = enet_tol
        self.max_iter = max_iter
        self.mode = mode
        self.verbose = verbose
        self.eps = eps


class GraphicalLasso(BaseGraphicalLasso):
    # [CLASS DOC]
    # Estimator for a fixed alpha. Adds constraints for alpha (>= 0) and covariance
    # (None or 'precomputed', meaning X is already a covariance matrix).
    """Sparse inverse covariance estimation with an l1-penalized estimator.

    For a usage example see
    :ref:`sphx_glr_auto_examples_applications_plot_stock_market.py`.

    Read more in the :ref:`User Guide <sparse_inverse_covariance>`.

    .. versionchanged:: v0.20
        GraphLasso has been renamed to GraphicalLasso

    Parameters
    ----------
    alpha : float, default=0.01
        The regularization parameter: the higher alpha, the more
        regularization, the sparser the inverse covariance.
        Range is (0, inf].

    mode : {'cd', 'lars'}, default='cd'
        The Lasso solver to use: coordinate descent or LARS. Use LARS for
        very sparse underlying graphs, where p > n. Elsewhere prefer cd
        which is more numerically stable.

    covariance : "precomputed", default=None
        If covariance is "precomputed", the input data in `fit` is assumed
        to be the covariance matrix. If `None`, the empirical covariance
        is estimated from the data `X`.

        .. versionadded:: 1.3

    tol : float, default=1e-4
        The tolerance to declare convergence: if the dual gap goes below
        this value, iterations are stopped. Range is (0, inf].

    enet_tol : float, default=1e-4
        The tolerance for the elastic net solver used to calculate the descent
        direction. This parameter controls the accuracy of the search direction
        for a given column update, not of the overall parameter estimate. Only
        used for mode='cd'. Range is (0, inf].

    max_iter : int, default=100
        The maximum number of iterations.

    verbose : bool, default=False
        If verbose is True, the objective function and dual gap are
        plotted at each iteration.

    eps : float, default=eps
        The machine-precision regularization in the computation of the
        Cholesky diagonal factors. Increase this for very ill-conditioned
        systems. Default is `np.finfo(np.float64).eps`.

        .. versionadded:: 1.3

    assume_centered : bool, default=False
        If True, data are not centered before computation.
        Useful when working with data whose mean is almost, but not exactly
        zero.
        If False, data are centered before computation.

    Attributes
    ----------
    location_ : ndarray of shape (n_features,)
        Estimated location, i.e. the estimated mean.

    covariance_ : ndarray of shape (n_features, n_features)
        Estimated covariance matrix

    precision_ : ndarray of shape (n_features, n_features)
        Estimated pseudo inverse matrix.

    n_iter_ : int
        Number of iterations run.

    costs_ : list of (objective, dual_gap) pairs
        The list of values of the objective function and the dual gap at
        each iteration. Returned only if return_costs is True.

        .. versionadded:: 1.3

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    graphical_lasso : L1-penalized covariance estimator.
    GraphicalLassoCV : Sparse inverse covariance with
        cross-validated choice of the l1 penalty.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.covariance import GraphicalLasso
    >>> true_cov = np.array([[0.8, 0.0, 0.2, 0.0],
    ...                      [0.0, 0.4, 0.0, 0.0],
    ...                      [0.2, 0.0, 0.3, 0.1],
    ...                      [0.0, 0.0, 0.1, 0.7]])
    >>> np.random.seed(0)
    >>> X = np.random.multivariate_normal(mean=[0, 0, 0, 0],
    ...                                   cov=true_cov,
    ...                                   size=200)
    >>> cov = GraphicalLasso().fit(X)
    >>> np.around(cov.covariance_, decimals=3)
    array([[0.816, 0.049, 0.218, 0.019],
           [0.049, 0.364, 0.017, 0.034],
           [0.218, 0.017, 0.322, 0.093],
           [0.019, 0.034, 0.093, 0.69 ]])
    >>> np.around(cov.location_, decimals=3)
    array([0.073, 0.04 , 0.038, 0.143])
    """

    _parameter_constraints: dict = {
        **BaseGraphicalLasso._parameter_constraints,
        "alpha": [Interval(Real, 0, None, closed="both")],
        "covariance": [StrOptions({"precomputed"}), None],
    }

    def __init__(
        self,
        alpha=0.01,
        *,
        mode="cd",
        covariance=None,
        tol=1e-4,
        enet_tol=1e-4,
        max_iter=100,
        verbose=False,
        eps=np.finfo(np.float64).eps,
        assume_centered=False,
    ):
        # [FUNCTION DOC]
        # Stores alpha, covariance and the base solver settings as attributes (sklearn
        # convention: no validation here).
        super().__init__(
            tol=tol,
            enet_tol=enet_tol,
            max_iter=max_iter,
            mode=mode,
            verbose=verbose,
            eps=eps,
            assume_centered=assume_centered,
        )
        self.alpha = alpha
        self.covariance = covariance

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None):
        # [FUNCTION DOC]
        # Validates X (at least 2 samples and 2 features). If covariance is
        # 'precomputed', uses a copy of X as the empirical covariance and sets location_
        # to zeros; otherwise computes the empirical covariance (respecting
        # assume_centered) and location_ as zeros or the column means. Runs
        # _graphical_lasso with the estimator settings.
        # Inputs: X - (n_samples, n_features) data or (p, p) covariance; y ignored.
        # Outputs: self. Side effects: sets location_, covariance_, precision_, costs_,
        # n_iter_, n_features_in_ (and feature_names_in_). Exceptions: validation
        # errors, solver errors.
        """Fit the GraphicalLasso model to X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data from which to compute the covariance estimate.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        # Covariance does not make sense for a single feature
        X = validate_data(self, X, ensure_min_features=2, ensure_min_samples=2)

        if self.covariance == "precomputed":
            emp_cov = X.copy()
            self.location_ = np.zeros(X.shape[1])
        else:
            emp_cov = empirical_covariance(X, assume_centered=self.assume_centered)
            if self.assume_centered:
                self.location_ = np.zeros(X.shape[1])
            else:
                self.location_ = X.mean(0)

        self.covariance_, self.precision_, self.costs_, self.n_iter_ = _graphical_lasso(
            emp_cov,
            alpha=self.alpha,
            cov_init=None,
            mode=self.mode,
            tol=self.tol,
            enet_tol=self.enet_tol,
            max_iter=self.max_iter,
            verbose=self.verbose,
            eps=self.eps,
        )
        return self


# Cross-validation with GraphicalLasso
def graphical_lasso_path(
    X,
    alphas,
    cov_init=None,
    X_test=None,
    mode="cd",
    tol=1e-4,
    enet_tol=1e-4,
    max_iter=100,
    verbose=False,
    eps=np.finfo(np.float64).eps,
):
    # [FUNCTION DOC]
    # Computes graphical-lasso solutions along a sequence of alphas with warm starts:
    # the empirical covariance of X is computed once, and each alpha's solve starts from
    # the previous covariance (or cov_init). If a solve raises FloatingPointError, NaN
    # is stored for that alpha's covariance/precision and score -inf. When X_test is
    # given, each precision is scored by the log-likelihood of X_test's empirical
    # covariance (non-finite scores mapped to -inf).
    # Inputs: X - (n_samples, n_features); alphas - iterable of floats; cov_init -
    # optional start; X_test - optional held-out data; solver settings as in
    # _graphical_lasso; verbose
    # - int, inner solver gets verbose-1.
    # Outputs: (covariances list, precisions list) or with scores list when X_test is
    # given.
    # Side effects: writes '.' to stderr per alpha when verbose == 1, prints when > 1.
    """l1-penalized covariance estimator along a path of decreasing alphas

    Read more in the :ref:`User Guide <sparse_inverse_covariance>`.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Data from which to compute the covariance estimate.

    alphas : array-like of shape (n_alphas,)
        The list of regularization parameters, decreasing order.

    cov_init : array of shape (n_features, n_features), default=None
        The initial guess for the covariance.

    X_test : array of shape (n_test_samples, n_features), default=None
        Optional test matrix to measure generalisation error.

    mode : {'cd', 'lars'}, default='cd'
        The Lasso solver to use: coordinate descent or LARS. Use LARS for
        very sparse underlying graphs, where p > n. Elsewhere prefer cd
        which is more numerically stable.

    tol : float, default=1e-4
        The tolerance to declare convergence: if the dual gap goes below
        this value, iterations are stopped. The tolerance must be a positive
        number.

    enet_tol : float, default=1e-4
        The tolerance for the elastic net solver used to calculate the descent
        direction. This parameter controls the accuracy of the search direction
        for a given column update, not of the overall parameter estimate. Only
        used for mode='cd'. The tolerance must be a positive number.

    max_iter : int, default=100
        The maximum number of iterations. This parameter should be a strictly
        positive integer.

    verbose : int or bool, default=False
        The higher the verbosity flag, the more information is printed
        during the fitting.

    eps : float, default=eps
        The machine-precision regularization in the computation of the
        Cholesky diagonal factors. Increase this for very ill-conditioned
        systems. Default is `np.finfo(np.float64).eps`.

        .. versionadded:: 1.3

    Returns
    -------
    covariances_ : list of shape (n_alphas,) of ndarray of shape \
            (n_features, n_features)
        The estimated covariance matrices.

    precisions_ : list of shape (n_alphas,) of ndarray of shape \
            (n_features, n_features)
        The estimated (sparse) precision matrices.

    scores_ : list of shape (n_alphas,), dtype=float
        The generalisation error (log-likelihood) on the test data.
        Returned only if test data is passed.
    """
    inner_verbose = max(0, verbose - 1)
    emp_cov = empirical_covariance(X)
    if cov_init is None:
        covariance_ = emp_cov.copy()
    else:
        covariance_ = cov_init
    covariances_ = list()
    precisions_ = list()
    scores_ = list()
    if X_test is not None:
        test_emp_cov = empirical_covariance(X_test)

    for alpha in alphas:
        try:
            # Capture the errors, and move on
            covariance_, precision_, _, _ = _graphical_lasso(
                emp_cov,
                alpha=alpha,
                cov_init=covariance_,
                mode=mode,
                tol=tol,
                enet_tol=enet_tol,
                max_iter=max_iter,
                verbose=inner_verbose,
                eps=eps,
            )
            covariances_.append(covariance_)
            precisions_.append(precision_)
            if X_test is not None:
                this_score = log_likelihood(test_emp_cov, precision_)
        except FloatingPointError:
            this_score = -np.inf
            covariances_.append(np.nan)
            precisions_.append(np.nan)
        if X_test is not None:
            if not np.isfinite(this_score):
                this_score = -np.inf
            scores_.append(this_score)
        if verbose == 1:
            sys.stderr.write(".")
        elif verbose > 1:
            if X_test is not None:
                print(
                    "[graphical_lasso_path] alpha: %.2e, score: %.2e"
                    % (alpha, this_score)
                )
            else:
                print("[graphical_lasso_path] alpha: %.2e" % alpha)
    if X_test is not None:
        return covariances_, precisions_, scores_
    return covariances_, precisions_


class GraphicalLassoCV(BaseGraphicalLasso):
    # [CLASS DOC]
    # Estimator choosing alpha by cross-validated log-likelihood on an alpha grid that
    # is iteratively refined. Adds constraints for alphas (int count or array),
    # n_refinements (>= 1), cv and n_jobs.
    """Sparse inverse covariance w/ cross-validated choice of the l1 penalty.

    See glossary entry for :term:`cross-validation estimator`.

    Read more in the :ref:`User Guide <sparse_inverse_covariance>`.

    .. versionchanged:: v0.20
        GraphLassoCV has been renamed to GraphicalLassoCV

    Parameters
    ----------
    alphas : int or array-like of shape (n_alphas,), dtype=float, default=4
        If an integer is given, it fixes the number of points on the
        grids of alpha to be used. If a list is given, it gives the
        grid to be used. See the notes in the class docstring for
        more details. Range is [1, inf) for an integer.
        Range is (0, inf] for an array-like of floats.

    n_refinements : int, default=4
        The number of times the grid is refined. Not used if explicit
        values of alphas are passed. Range is [1, inf).

    cv : int, cross-validation generator or iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, to use the default 5-fold cross-validation,
        - integer, to specify the number of folds,
        - :term:`CV splitter`,
        - an iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs :class:`~sklearn.model_selection.KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.

        .. versionchanged:: 0.20
            ``cv`` default value if None changed from 3-fold to 5-fold.

    tol : float, default=1e-4
        The tolerance to declare convergence: if the dual gap goes below
        this value, iterations are stopped. Range is (0, inf].

    enet_tol : float, default=1e-4
        The tolerance for the elastic net solver used to calculate the descent
        direction. This parameter controls the accuracy of the search direction
        for a given column update, not of the overall parameter estimate. Only
        used for mode='cd'. Range is (0, inf].

    max_iter : int, default=100
        Maximum number of iterations.

    mode : {'cd', 'lars'}, default='cd'
        The Lasso solver to use: coordinate descent or LARS. Use LARS for
        very sparse underlying graphs, where number of features is greater
        than number of samples. Elsewhere prefer cd which is more numerically
        stable.

    n_jobs : int, default=None
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

        .. versionchanged:: v0.20
           `n_jobs` default changed from 1 to None

    verbose : bool, default=False
        If verbose is True, the objective function and duality gap are
        printed at each iteration.

    eps : float, default=eps
        The machine-precision regularization in the computation of the
        Cholesky diagonal factors. Increase this for very ill-conditioned
        systems. Default is `np.finfo(np.float64).eps`.

        .. versionadded:: 1.3

    assume_centered : bool, default=False
        If True, data are not centered before computation.
        Useful when working with data whose mean is almost, but not exactly
        zero.
        If False, data are centered before computation.

    Attributes
    ----------
    location_ : ndarray of shape (n_features,)
        Estimated location, i.e. the estimated mean.

    covariance_ : ndarray of shape (n_features, n_features)
        Estimated covariance matrix.

    precision_ : ndarray of shape (n_features, n_features)
        Estimated precision matrix (inverse covariance).

    costs_ : list of (objective, dual_gap) pairs
        The list of values of the objective function and the dual gap at
        each iteration. Returned only if return_costs is True.

        .. versionadded:: 1.3

    alpha_ : float
        Penalization parameter selected.

    cv_results_ : dict of ndarrays
        A dict with keys:

        alphas : ndarray of shape (n_alphas,)
            All penalization parameters explored.

        split(k)_test_score : ndarray of shape (n_alphas,)
            Log-likelihood score on left-out data across (k)th fold.

            .. versionadded:: 1.0

        mean_test_score : ndarray of shape (n_alphas,)
            Mean of scores over the folds.

            .. versionadded:: 1.0

        std_test_score : ndarray of shape (n_alphas,)
            Standard deviation of scores over the folds.

            .. versionadded:: 1.0

    n_iter_ : int
        Number of iterations run for the optimal alpha.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    graphical_lasso : L1-penalized covariance estimator.
    GraphicalLasso : Sparse inverse covariance estimation
        with an l1-penalized estimator.

    Notes
    -----
    The search for the optimal penalization parameter (`alpha`) is done on an
    iteratively refined grid: first the cross-validated scores on a grid are
    computed, then a new refined grid is centered around the maximum, and so
    on.

    One of the challenges which is faced here is that the solvers can
    fail to converge to a well-conditioned estimate. The corresponding
    values of `alpha` then come out as missing values, but the optimum may
    be close to these missing values.

    In `fit`, once the best parameter `alpha` is found through
    cross-validation, the model is fit again using the entire training set.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.covariance import GraphicalLassoCV
    >>> true_cov = np.array([[0.8, 0.0, 0.2, 0.0],
    ...                      [0.0, 0.4, 0.0, 0.0],
    ...                      [0.2, 0.0, 0.3, 0.1],
    ...                      [0.0, 0.0, 0.1, 0.7]])
    >>> np.random.seed(0)
    >>> X = np.random.multivariate_normal(mean=[0, 0, 0, 0],
    ...                                   cov=true_cov,
    ...                                   size=200)
    >>> cov = GraphicalLassoCV().fit(X)
    >>> np.around(cov.covariance_, decimals=3)
    array([[0.816, 0.051, 0.22 , 0.017],
           [0.051, 0.364, 0.018, 0.036],
           [0.22 , 0.018, 0.322, 0.094],
           [0.017, 0.036, 0.094, 0.69 ]])
    >>> np.around(cov.location_, decimals=3)
    array([0.073, 0.04 , 0.038, 0.143])

    For an example comparing :class:`sklearn.covariance.GraphicalLassoCV`,
    :func:`sklearn.covariance.ledoit_wolf` shrinkage and the empirical covariance
    on high-dimensional gaussian data, see
    :ref:`sphx_glr_auto_examples_covariance_plot_sparse_cov.py`.
    """

    _parameter_constraints: dict = {
        **BaseGraphicalLasso._parameter_constraints,
        "alphas": [Interval(Integral, 0, None, closed="left"), "array-like"],
        "n_refinements": [Interval(Integral, 1, None, closed="left")],
        "cv": ["cv_object"],
        "n_jobs": [Integral, None],
    }

    def __init__(
        self,
        *,
        alphas=4,
        n_refinements=4,
        cv=None,
        tol=1e-4,
        enet_tol=1e-4,
        max_iter=100,
        mode="cd",
        n_jobs=None,
        verbose=False,
        eps=np.finfo(np.float64).eps,
        assume_centered=False,
    ):
        # [FUNCTION DOC]
        # Stores alphas, n_refinements, cv, n_jobs and the base solver settings as
        # attributes.
        super().__init__(
            tol=tol,
            enet_tol=enet_tol,
            max_iter=max_iter,
            mode=mode,
            verbose=verbose,
            eps=eps,
            assume_centered=assume_centered,
        )
        self.alphas = alphas
        self.n_refinements = n_refinements
        self.cv = cv
        self.n_jobs = n_jobs

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None, **params):
        # [FUNCTION DOC]
        # Behavior: validate X (>= 2 features), set location_, compute the empirical
        # covariance. If alphas is an array, check each value is in (0, inf] and do a
        # single pass; else build a log-spaced decreasing grid of `alphas` points from
        # alpha_max/100 to alpha_max and do n_refinements passes. Each pass runs
        # graphical_lasso_path on every CV split in parallel (max_iter reduced to 10
        # percent, ConvergenceWarnings suppressed), merges (alpha, per-fold scores,
        # covs) into the path sorted by decreasing alpha, and picks the best mean score
        # (ties resolved to the smaller alpha; huge scores treated as NaN). The grid is
        # then narrowed around the best index (special cases at the top, at the last
        # finite score, and at the bottom where the lower bound becomes 1 percent of the
        # best alpha) and regenerated with n_alphas interior points when alphas was an
        # int. Finally appends alpha 0 scored with cross_val_score of
        # EmpiricalCovariance, builds cv_results_, sets alpha_ and refits on the full
        # empirical covariance.
        # Inputs: X - (n_samples, n_features); y - ignored except for CV splitting;
        # params - metadata routed to the splitter (requires routing enabled, else
        # error).
        # Outputs: self. Side effects: sets location_, cv_results_ (alphas,
        # splitK_test_score, mean/std_test_score), alpha_, covariance_, precision_,
        # costs_, n_iter_; prints progress when verbose. Exceptions: invalid alpha
        # values, unsupported params without routing.
        # KNOWN ISSUE: params are forwarded unchanged to cross_val_score for the alpha=0
        # column instead of the routed splitter params.
        """Fit the GraphicalLasso covariance model to X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data from which to compute the covariance estimate.

        y : Ignored
            Not used, present for API consistency by convention.

        **params : dict, default=None
            Parameters to be passed to the CV splitter and the
            cross_val_score function.

            .. versionadded:: 1.5
                Only available if `enable_metadata_routing=True`,
                which can be set by using
                ``sklearn.set_config(enable_metadata_routing=True)``.
                See :ref:`Metadata Routing User Guide <metadata_routing>` for
                more details.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        # Covariance does not make sense for a single feature
        _raise_for_params(params, self, "fit")

        X = validate_data(self, X, ensure_min_features=2)
        if self.assume_centered:
            self.location_ = np.zeros(X.shape[1])
        else:
            self.location_ = X.mean(0)
        emp_cov = empirical_covariance(X, assume_centered=self.assume_centered)

        cv = check_cv(self.cv, y, classifier=False)

        # List of (alpha, scores, covs)
        path = list()
        n_alphas = self.alphas
        inner_verbose = max(0, self.verbose - 1)

        if _is_arraylike_not_scalar(n_alphas):
            for alpha in self.alphas:
                check_scalar(
                    alpha,
                    "alpha",
                    Real,
                    min_val=0,
                    max_val=np.inf,
                    include_boundaries="right",
                )
            alphas = self.alphas
            n_refinements = 1
        else:
            n_refinements = self.n_refinements
            alpha_1 = alpha_max(emp_cov)
            alpha_0 = 1e-2 * alpha_1
            alphas = np.logspace(np.log10(alpha_0), np.log10(alpha_1), n_alphas)[::-1]

        if _routing_enabled():
            routed_params = process_routing(self, "fit", **params)
        else:
            routed_params = _manual_routing({"splitter": {}})

        t0 = time.time()
        for i in range(n_refinements):
            with warnings.catch_warnings():
                # No need to see the convergence warnings on this grid:
                # they will always be points that will not converge
                # during the cross-validation
                warnings.simplefilter("ignore", ConvergenceWarning)
                # Compute the cross-validated loss on the current grid

                # NOTE: Warm-restarting graphical_lasso_path has been tried,
                # and this did not allow to gain anything
                # (same execution time with or without).
                this_path = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
                    delayed(graphical_lasso_path)(
                        X[train],
                        alphas=alphas,
                        X_test=X[test],
                        mode=self.mode,
                        tol=self.tol,
                        enet_tol=self.enet_tol,
                        max_iter=int(0.1 * self.max_iter),
                        verbose=inner_verbose,
                        eps=self.eps,
                    )
                    for train, test in cv.split(X, y, **routed_params.splitter.split)
                )

            # Little danse to transform the list in what we need
            covs, _, scores = zip(*this_path)
            covs = zip(*covs)
            scores = zip(*scores)
            path.extend(zip(alphas, scores, covs))
            path = sorted(path, key=operator.itemgetter(0), reverse=True)

            # Find the maximum (avoid using built in 'max' function to
            # have a fully-reproducible selection of the smallest alpha
            # in case of equality)
            best_score = -np.inf
            last_finite_idx = 0
            for index, (alpha, scores, _) in enumerate(path):
                this_score = np.mean(scores)
                if this_score >= 0.1 / np.finfo(np.float64).eps:
                    this_score = np.nan
                if np.isfinite(this_score):
                    last_finite_idx = index
                if this_score >= best_score:
                    best_score = this_score
                    best_index = index

            # Refine the grid
            if best_index == 0:
                # We do not need to go back: we have chosen
                # the highest value of alpha for which there are
                # non-zero coefficients
                alpha_1 = path[0][0]
                alpha_0 = path[1][0]
            elif best_index == last_finite_idx and not best_index == len(path) - 1:
                # We have non-converged models on the upper bound of the
                # grid, we need to refine the grid there
                alpha_1 = path[best_index][0]
                alpha_0 = path[best_index + 1][0]
            elif best_index == len(path) - 1:
                alpha_1 = path[best_index][0]
                alpha_0 = 0.01 * path[best_index][0]
            else:
                alpha_1 = path[best_index - 1][0]
                alpha_0 = path[best_index + 1][0]

            if not _is_arraylike_not_scalar(n_alphas):
                alphas = np.logspace(np.log10(alpha_1), np.log10(alpha_0), n_alphas + 2)
                alphas = alphas[1:-1]

            if self.verbose and n_refinements > 1:
                print(
                    "[GraphicalLassoCV] Done refinement % 2i out of %i: % 3is"
                    % (i + 1, n_refinements, time.time() - t0)
                )

        path = list(zip(*path))
        grid_scores = list(path[1])
        alphas = list(path[0])
        # Finally, compute the score with alpha = 0
        alphas.append(0)
        grid_scores.append(
            cross_val_score(
                EmpiricalCovariance(),
                X,
                cv=cv,
                n_jobs=self.n_jobs,
                verbose=inner_verbose,
                params=params,
            )
        )
        grid_scores = np.array(grid_scores)

        self.cv_results_ = {"alphas": np.array(alphas)}

        for i in range(grid_scores.shape[1]):
            self.cv_results_[f"split{i}_test_score"] = grid_scores[:, i]

        self.cv_results_["mean_test_score"] = np.mean(grid_scores, axis=1)
        self.cv_results_["std_test_score"] = np.std(grid_scores, axis=1)

        best_alpha = alphas[best_index]
        self.alpha_ = best_alpha

        # Finally fit the model with the selected alpha
        self.covariance_, self.precision_, self.costs_, self.n_iter_ = _graphical_lasso(
            emp_cov,
            alpha=best_alpha,
            mode=self.mode,
            tol=self.tol,
            enet_tol=self.enet_tol,
            max_iter=self.max_iter,
            verbose=inner_verbose,
            eps=self.eps,
        )
        return self

    def get_metadata_routing(self):
        # [FUNCTION DOC]
        # Returns a MetadataRouter that routes fit metadata to the cv splitter's split
        # method. Outputs: MetadataRouter.
        """Get metadata routing of this object.

        Please check :ref:`User Guide <metadata_routing>` on how the routing
        mechanism works.

        .. versionadded:: 1.5

        Returns
        -------
        routing : MetadataRouter
            A :class:`~sklearn.utils.metadata_routing.MetadataRouter` encapsulating
            routing information.
        """
        router = MetadataRouter(owner=self).add(
            splitter=check_cv(self.cv),
            method_mapping=MethodMapping().add(callee="split", caller="fit"),
        )
        return router
