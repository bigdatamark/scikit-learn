# [MODULE DOC]
# Bayesian linear regression estimators with Gamma hyperpriors: BayesianRidge (single
#   shared weight precision lambda, evidence maximization via an SVD of X) and
#   ARDRegression (per-feature precisions with pruning, Automatic Relevance
#   Determination). Both use MacKay's fixed-point updates for the noise precision alpha
#   and weight precisions.
"""
Various bayesian regression
"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

from math import log
from numbers import Integral, Real

import numpy as np
from scipy import linalg
from scipy.linalg import pinvh

from sklearn.base import RegressorMixin, _fit_context
from sklearn.linear_model._base import LinearModel, _preprocess_data
from sklearn.utils import _safe_indexing
from sklearn.utils._param_validation import Interval
from sklearn.utils.extmath import fast_logdet
from sklearn.utils.validation import _check_sample_weight, validate_data

###############################################################################
# BayesianRidge regression


class BayesianRidge(RegressorMixin, LinearModel):
    # [CLASS DOC]
    # Bayesian ridge regression. fit centers (and weight-rescales) data, computes one
    #   SVD of X, then alternates posterior-mean updates and MacKay hyperparameter updates
    #   until coef changes less than tol in L1 norm. Fitted: coef_, intercept_, alpha_,
    #   lambda_, sigma_ (posterior covariance), scores_ (if compute_score), n_iter_,
    #   X_offset_, X_scale_.
    """Bayesian ridge regression.

    Fit a Bayesian ridge model. See the Notes section for details on this
    implementation and the optimization of the regularization parameters
    lambda (precision of the weights) and alpha (precision of the noise).

    Read more in the :ref:`User Guide <bayesian_regression>`.
    For an intuitive visualization of how the sinusoid is approximated by
    a polynomial using different pairs of initial values, see
    :ref:`sphx_glr_auto_examples_linear_model_plot_bayesian_ridge_curvefit.py`.

    Parameters
    ----------
    max_iter : int, default=300
        Maximum number of iterations over the complete dataset before
        stopping independently of any early stopping criterion.

        .. versionchanged:: 1.3

    tol : float, default=1e-3
        Stop the algorithm if w has converged.

    alpha_1 : float, default=1e-6
        Hyper-parameter : shape parameter for the Gamma distribution prior
        over the alpha parameter.

    alpha_2 : float, default=1e-6
        Hyper-parameter : inverse scale parameter (rate parameter) for the
        Gamma distribution prior over the alpha parameter.

    lambda_1 : float, default=1e-6
        Hyper-parameter : shape parameter for the Gamma distribution prior
        over the lambda parameter.

    lambda_2 : float, default=1e-6
        Hyper-parameter : inverse scale parameter (rate parameter) for the
        Gamma distribution prior over the lambda parameter.

    alpha_init : float, default=None
        Initial value for alpha (precision of the noise).
        If not set, alpha_init is 1/Var(y).

        .. versionadded:: 0.22

    lambda_init : float, default=None
        Initial value for lambda (precision of the weights).
        If not set, lambda_init is 1.

        .. versionadded:: 0.22

    compute_score : bool, default=False
        If True, compute the log marginal likelihood at each iteration of the
        optimization.

    fit_intercept : bool, default=True
        Whether to calculate the intercept for this model.
        The intercept is not treated as a probabilistic parameter
        and thus has no associated variance. If set
        to False, no intercept will be used in calculations
        (i.e. data is expected to be centered).

    copy_X : bool, default=True
        If True, X will be copied; else, it may be overwritten.

    verbose : bool, default=False
        Verbose mode when fitting the model.

    Attributes
    ----------
    coef_ : array-like of shape (n_features,)
        Coefficients of the regression model (mean of distribution)

    intercept_ : float
        Independent term in decision function. Set to 0.0 if
        `fit_intercept = False`.

    alpha_ : float
       Estimated precision of the noise.

    lambda_ : float
       Estimated precision of the weights.

    sigma_ : array-like of shape (n_features, n_features)
        Estimated variance-covariance matrix of the weights

    scores_ : array-like of shape (n_iter_+1,)
        If computed_score is True, value of the log marginal likelihood (to be
        maximized) at each iteration of the optimization. The array starts
        with the value of the log marginal likelihood obtained for the initial
        values of alpha and lambda and ends with the value obtained for the
        estimated alpha and lambda.

    n_iter_ : int
        The actual number of iterations to reach the stopping criterion.

    X_offset_ : ndarray of shape (n_features,)
        If `fit_intercept=True`, offset subtracted for centering data to a
        zero mean. Set to np.zeros(n_features) otherwise.

    X_scale_ : ndarray of shape (n_features,)
        Set to np.ones(n_features).

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    ARDRegression : Bayesian ARD regression.

    Notes
    -----
    There exist several strategies to perform Bayesian ridge regression. This
    implementation is based on the algorithm described in Appendix A of
    (Tipping, 2001) where updates of the regularization parameters are done as
    suggested in (MacKay, 1992). Note that according to A New
    View of Automatic Relevance Determination (Wipf and Nagarajan, 2008) these
    update rules do not guarantee that the marginal likelihood is increasing
    between two consecutive iterations of the optimization.

    References
    ----------
    D. J. C. MacKay, Bayesian Interpolation, Computation and Neural Systems,
    Vol. 4, No. 3, 1992.

    M. E. Tipping, Sparse Bayesian Learning and the Relevance Vector Machine,
    Journal of Machine Learning Research, Vol. 1, 2001.

    Examples
    --------
    >>> from sklearn import linear_model
    >>> clf = linear_model.BayesianRidge()
    >>> clf.fit([[0,0], [1, 1], [2, 2]], [0, 1, 2])
    BayesianRidge()
    >>> clf.predict([[1, 1]])
    array([1.])
    """

    _parameter_constraints: dict = {
        "max_iter": [Interval(Integral, 1, None, closed="left")],
        "tol": [Interval(Real, 0, None, closed="neither")],
        "alpha_1": [Interval(Real, 0, None, closed="left")],
        "alpha_2": [Interval(Real, 0, None, closed="left")],
        "lambda_1": [Interval(Real, 0, None, closed="left")],
        "lambda_2": [Interval(Real, 0, None, closed="left")],
        "alpha_init": [None, Interval(Real, 0, None, closed="left")],
        "lambda_init": [None, Interval(Real, 0, None, closed="left")],
        "compute_score": ["boolean"],
        "fit_intercept": ["boolean"],
        "copy_X": ["boolean"],
        "verbose": ["verbose"],
    }

    def __init__(
        self,
        *,
        max_iter=300,
        tol=1.0e-3,
        alpha_1=1.0e-6,
        alpha_2=1.0e-6,
        lambda_1=1.0e-6,
        lambda_2=1.0e-6,
        alpha_init=None,
        lambda_init=None,
        compute_score=False,
        fit_intercept=True,
        copy_X=True,
        verbose=False,
    ):
        # [FUNCTION DOC]
        # Stores hyperparameters unchanged: max_iter, tol, Gamma prior shape/rate
        #   parameters alpha_1, alpha_2, lambda_1, lambda_2, optional initial alpha_init
        #   and lambda_init, compute_score, fit_intercept, copy_X, verbose.
        self.max_iter = max_iter
        self.tol = tol
        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.alpha_init = alpha_init
        self.lambda_init = lambda_init
        self.compute_score = compute_score
        self.fit_intercept = fit_intercept
        self.copy_X = copy_X
        self.verbose = verbose

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y, sample_weight=None):
        # [FUNCTION DOC]
        # Fits the model. Steps: validate X (float, writeable) and numeric y; compute
        #   the (weighted) variance of y and sum of weights; center and rescale X and y by
        #   sqrt of sample weights via _preprocess_data. Init alpha to 1/(var(y)+eps) and
        #   lambda to 1 unless given. Compute SVD of X (full only if n_samples <
        #   n_features) and squared singular values. Loop up to max_iter: compute
        #   posterior mean coef and SSE with _update_coef_, optionally record log marginal
        #   likelihood, compute effective number of parameters gamma as the sum of
        #   alpha*eig / (lambda + alpha*eig), update lambda as (gamma + 2 lambda_1) /
        #   (squared norm of coef + 2 lambda_2) and alpha as (sw_sum - gamma + 2 alpha_1)
        #   / (SSE + 2 alpha_2); stop when the L1 change of coef is below tol (after the
        #   first iteration). Then recompute coef_ with the final hyperparameters,
        #   posterior covariance sigma_ from the full right singular vectors, and the
        #   intercept.
        # Inputs: X (n_samples, n_features); y (n_samples,); sample_weight optional.
        # Outputs: self.
        # Side effects: sets X_offset_, X_scale_, scores_ (list, converted to ndarray
        #   only when compute_score), n_iter_, alpha_, lambda_, coef_, sigma_, intercept_;
        #   prints convergence if verbose; may modify X in place when copy_X is False.
        # KNOWN ISSUE: the final log-likelihood appended after the loop uses the loop's
        #   last coef_ rather than the recomputed self.coef_ (while using the new sse_),
        #   so it mixes values from two coefficient vectors.
        """Fit the model.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data.
        y : ndarray of shape (n_samples,)
            Target values. Will be cast to X's dtype if necessary.

        sample_weight : ndarray of shape (n_samples,), default=None
            Individual weights for each sample.

            .. versionadded:: 0.20
               parameter *sample_weight* support to BayesianRidge.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        X, y = validate_data(
            self,
            X,
            y,
            dtype=[np.float64, np.float32],
            force_writeable=True,
            y_numeric=True,
        )
        dtype = X.dtype
        n_samples, n_features = X.shape

        sw_sum = n_samples
        y_var = y.var()
        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X, dtype=dtype)
            sw_sum = sample_weight.sum()
            y_mean = np.average(y, weights=sample_weight)
            y_var = np.average((y - y_mean) ** 2, weights=sample_weight)

        X, y, X_offset_, y_offset_, X_scale_, _ = _preprocess_data(
            X,
            y,
            fit_intercept=self.fit_intercept,
            copy=self.copy_X,
            sample_weight=sample_weight,
            # Sample weight can be implemented via a simple rescaling.
            rescale_with_sw=True,
        )

        self.X_offset_ = X_offset_
        self.X_scale_ = X_scale_

        # Initialization of the values of the parameters
        eps = np.finfo(np.float64).eps
        # Add `eps` in the denominator to omit division by zero
        alpha_ = self.alpha_init
        lambda_ = self.lambda_init
        if alpha_ is None:
            alpha_ = 1.0 / (y_var + eps)
        if lambda_ is None:
            lambda_ = 1.0

        # Avoid unintended type promotion to float64 with numpy 2
        alpha_ = np.asarray(alpha_, dtype=dtype)
        lambda_ = np.asarray(lambda_, dtype=dtype)

        verbose = self.verbose
        lambda_1 = self.lambda_1
        lambda_2 = self.lambda_2
        alpha_1 = self.alpha_1
        alpha_2 = self.alpha_2

        self.scores_ = list()
        coef_old_ = None

        XT_y = np.dot(X.T, y)
        # Let M, N = n_samples, n_features and K = min(M, N).
        # The posterior covariance matrix needs Vh_full: (N, N).
        # The full SVD is only required when n_samples < n_features.
        # When n_samples < n_features, K=M and full_matrices=True
        # U: (M, M), S: M, Vh_full: (N, N), Vh: (M, N)
        # When n_samples > n_features, K=N and full_matrices=False
        # U: (M, N), S: N, Vh_full: (N, N), Vh: (N, N)
        U, S, Vh_full = linalg.svd(X, full_matrices=(n_samples < n_features))
        K = len(S)
        eigen_vals_ = S**2
        eigen_vals_full = np.zeros(n_features, dtype=dtype)
        eigen_vals_full[0:K] = eigen_vals_
        Vh = Vh_full[0:K, :]

        # Convergence loop of the bayesian ridge regression
        for iter_ in range(self.max_iter):
            # update posterior mean coef_ based on alpha_ and lambda_ and
            # compute corresponding sse (sum of squared errors)
            coef_, sse_ = self._update_coef_(
                X, y, n_samples, n_features, XT_y, U, Vh, eigen_vals_, alpha_, lambda_
            )
            if self.compute_score:
                # compute the log marginal likelihood
                s = self._log_marginal_likelihood(
                    n_samples,
                    n_features,
                    sw_sum,
                    eigen_vals_,
                    alpha_,
                    lambda_,
                    coef_,
                    sse_,
                )
                self.scores_.append(s)

            # Update alpha and lambda according to (MacKay, 1992)
            gamma_ = np.sum((alpha_ * eigen_vals_) / (lambda_ + alpha_ * eigen_vals_))
            lambda_ = (gamma_ + 2 * lambda_1) / (np.sum(coef_**2) + 2 * lambda_2)
            alpha_ = (sw_sum - gamma_ + 2 * alpha_1) / (sse_ + 2 * alpha_2)

            # Check for convergence
            if iter_ != 0 and np.sum(np.abs(coef_old_ - coef_)) < self.tol:
                if verbose:
                    print("Convergence after ", str(iter_), " iterations")
                break
            coef_old_ = np.copy(coef_)

        self.n_iter_ = iter_ + 1

        # return regularization parameters and corresponding posterior mean,
        # log marginal likelihood and posterior covariance
        self.alpha_ = alpha_
        self.lambda_ = lambda_
        self.coef_, sse_ = self._update_coef_(
            X, y, n_samples, n_features, XT_y, U, Vh, eigen_vals_, alpha_, lambda_
        )
        if self.compute_score:
            # compute the log marginal likelihood
            s = self._log_marginal_likelihood(
                n_samples,
                n_features,
                sw_sum,
                eigen_vals_,
                alpha_,
                lambda_,
                coef_,
                sse_,
            )
            self.scores_.append(s)
            self.scores_ = np.array(self.scores_)

        # posterior covariance
        self.sigma_ = np.dot(
            Vh_full.T, Vh_full / (alpha_ * eigen_vals_full + lambda_)[:, np.newaxis]
        )

        self._set_intercept(X_offset_, y_offset_, X_scale_)

        return self

    def predict(self, X, return_std=False):
        # [FUNCTION DOC]
        # Predicts with the linear model; if return_std, also returns the predictive
        #   standard deviation per sample: square root of the quadratic form of the
        #   centered row with sigma_ plus 1/alpha_.
        # Inputs: X (n_samples, n_features); return_std bool, default False.
        # Outputs: y_mean (n_samples,) or tuple (y_mean, y_std).
        # KNOWN ISSUE: the std branch uses raw X with plain subtraction and np.dot, so
        #   it does not support sparse input, and ignores X_scale_ (fine only because it
        #   is all ones here).
        """Predict using the linear model.

        In addition to the mean of the predictive distribution, also its
        standard deviation can be returned.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Samples.

        return_std : bool, default=False
            Whether to return the standard deviation of posterior prediction.

        Returns
        -------
        y_mean : array-like of shape (n_samples,)
            Mean of predictive distribution of query points.

        y_std : array-like of shape (n_samples,)
            Standard deviation of predictive distribution of query points.
        """
        y_mean = self._decision_function(X)
        if not return_std:
            return y_mean
        else:
            X = X - self.X_offset_
            sigmas_squared_data = (np.dot(X, self.sigma_) * X).sum(axis=1)
            y_std = np.sqrt(sigmas_squared_data + (1.0 / self.alpha_))
            return y_mean, y_std

    def _update_coef_(
        self, X, y, n_samples, n_features, XT_y, U, Vh, eigen_vals_, alpha_, lambda_
    ):
        # [FUNCTION DOC]
        # HELPER: Computes the posterior mean of coef for given alpha_ and lambda_ using
        #   the SVD: if n_samples > n_features, V diag(1/(eig + lambda/alpha)) V^T times
        #   X^T y; otherwise X^T U diag(1/(eig + lambda/alpha)) U^T y (dual form). Also
        #   returns the sum of squared residuals on the (preprocessed, weight-rescaled)
        #   data.
        # Outputs: (coef_ of shape (n_features,), sse_ float).
        # Side effects: none.
        """Update posterior mean and compute corresponding sse (sum of squared errors).

        Posterior mean is given by coef_ = scaled_sigma_ * X.T * y where
        scaled_sigma_ = (lambda_/alpha_ * np.eye(n_features)
                         + np.dot(X.T, X))^-1
        """

        if n_samples > n_features:
            coef_ = np.linalg.multi_dot(
                [Vh.T, Vh / (eigen_vals_ + lambda_ / alpha_)[:, np.newaxis], XT_y]
            )
        else:
            coef_ = np.linalg.multi_dot(
                [X.T, U / (eigen_vals_ + lambda_ / alpha_)[None, :], U.T, y]
            )

        # Note: we do not need to explicitly use the weights in this sum because
        # y and X were preprocessed by _rescale_data to handle the weights.
        sse_ = np.sum((y - np.dot(X, coef_)) ** 2)

        return coef_, sse_

    def _log_marginal_likelihood(
        self, n_samples, n_features, sw_sum, eigen_vals, alpha_, lambda_, coef, sse
    ):
        # [FUNCTION DOC]
        # HELPER: Computes the log marginal likelihood (evidence) for the current
        #   alpha_, lambda_, coef and SSE, including the Gamma hyperprior terms. Log
        #   determinant of the posterior covariance is minus the sum of logs of (lambda +
        #   alpha * eigenvalue), padding eigenvalues with zeros up to n_features when
        #   n_samples <= n_features. sw_sum replaces n_samples in the data terms to
        #   account for weights.
        # Outputs: float score.
        # Side effects: none.
        """Log marginal likelihood."""
        alpha_1 = self.alpha_1
        alpha_2 = self.alpha_2
        lambda_1 = self.lambda_1
        lambda_2 = self.lambda_2

        # compute the log of the determinant of the posterior covariance.
        # posterior covariance is given by
        # sigma = (lambda_ * np.eye(n_features) + alpha_ * np.dot(X.T, X))^-1
        if n_samples > n_features:
            logdet_sigma = -np.sum(np.log(lambda_ + alpha_ * eigen_vals))
        else:
            logdet_sigma = np.full(n_features, lambda_, dtype=np.array(lambda_).dtype)
            logdet_sigma[:n_samples] += alpha_ * eigen_vals
            logdet_sigma = -np.sum(np.log(logdet_sigma))

        score = lambda_1 * log(lambda_) - lambda_2 * lambda_
        score += alpha_1 * log(alpha_) - alpha_2 * alpha_
        score += 0.5 * (
            n_features * log(lambda_)
            + sw_sum * log(alpha_)
            - alpha_ * sse
            - lambda_ * np.sum(coef**2)
            + logdet_sigma
            - sw_sum * log(2 * np.pi)
        )

        return score


###############################################################################
# ARD (Automatic Relevance Determination) regression


class ARDRegression(RegressorMixin, LinearModel):
    # [CLASS DOC]
    # ARD regression: one precision per weight; features whose precision exceeds
    #   threshold_lambda are pruned (coef set to 0) and excluded from later updates.
    #   Posterior covariance is computed by direct inversion or the Woodbury identity
    #   depending on the data shape.
    """Bayesian ARD regression.

    Fit the weights of a regression model, using an ARD prior. The weights of
    the regression model are assumed to be in Gaussian distributions.
    Also estimate the parameters lambda (precisions of the distributions of the
    weights) and alpha (precision of the distribution of the noise).
    The estimation is done by an iterative procedures (Evidence Maximization)

    Read more in the :ref:`User Guide <bayesian_regression>`.

    Parameters
    ----------
    max_iter : int, default=300
        Maximum number of iterations.

        .. versionchanged:: 1.3

    tol : float, default=1e-3
        Stop the algorithm if w has converged.

    alpha_1 : float, default=1e-6
        Hyper-parameter : shape parameter for the Gamma distribution prior
        over the alpha parameter.

    alpha_2 : float, default=1e-6
        Hyper-parameter : inverse scale parameter (rate parameter) for the
        Gamma distribution prior over the alpha parameter.

    lambda_1 : float, default=1e-6
        Hyper-parameter : shape parameter for the Gamma distribution prior
        over the lambda parameter.

    lambda_2 : float, default=1e-6
        Hyper-parameter : inverse scale parameter (rate parameter) for the
        Gamma distribution prior over the lambda parameter.

    compute_score : bool, default=False
        If True, compute the objective function at each step of the model.

    threshold_lambda : float, default=10 000
        Threshold for removing (pruning) weights with high precision from
        the computation.

    fit_intercept : bool, default=True
        Whether to calculate the intercept for this model. If set
        to false, no intercept will be used in calculations
        (i.e. data is expected to be centered).

    copy_X : bool, default=True
        If True, X will be copied; else, it may be overwritten.

    verbose : bool, default=False
        Verbose mode when fitting the model.

    Attributes
    ----------
    coef_ : array-like of shape (n_features,)
        Coefficients of the regression model (mean of distribution)

    alpha_ : float
       estimated precision of the noise.

    lambda_ : array-like of shape (n_features,)
       estimated precisions of the weights.

    sigma_ : array-like of shape (n_features, n_features)
        estimated variance-covariance matrix of the weights

    scores_ : float
        if computed, value of the objective function (to be maximized)

    n_iter_ : int
        The actual number of iterations to reach the stopping criterion.

        .. versionadded:: 1.3

    intercept_ : float
        Independent term in decision function. Set to 0.0 if
        ``fit_intercept = False``.

    X_offset_ : float
        If `fit_intercept=True`, offset subtracted for centering data to a
        zero mean. Set to np.zeros(n_features) otherwise.

    X_scale_ : float
        Set to np.ones(n_features).

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    BayesianRidge : Bayesian ridge regression.

    References
    ----------
    D. J. C. MacKay, Bayesian nonlinear modeling for the prediction
    competition, ASHRAE Transactions, 1994.

    R. Salakhutdinov, Lecture notes on Statistical Machine Learning,
    http://www.utstat.toronto.edu/~rsalakhu/sta4273/notes/Lecture2.pdf#page=15
    Their beta is our ``self.alpha_``
    Their alpha is our ``self.lambda_``
    ARD is a little different than the slide: only dimensions/features for
    which ``self.lambda_ < self.threshold_lambda`` are kept and the rest are
    discarded.

    Examples
    --------
    >>> from sklearn import linear_model
    >>> clf = linear_model.ARDRegression()
    >>> clf.fit([[0,0], [1, 1], [2, 2]], [0, 1, 2])
    ARDRegression()
    >>> clf.predict([[1, 1]])
    array([1.])

    -   :ref:`sphx_glr_auto_examples_linear_model_plot_ard.py` demonstrates ARD
        Regression.
    -   :ref:`sphx_glr_auto_examples_linear_model_plot_lasso_and_elasticnet.py`
        showcases ARD Regression alongside Lasso and Elastic-Net for sparse,
        correlated signals, in the presence of noise.
    """

    _parameter_constraints: dict = {
        "max_iter": [Interval(Integral, 1, None, closed="left")],
        "tol": [Interval(Real, 0, None, closed="left")],
        "alpha_1": [Interval(Real, 0, None, closed="left")],
        "alpha_2": [Interval(Real, 0, None, closed="left")],
        "lambda_1": [Interval(Real, 0, None, closed="left")],
        "lambda_2": [Interval(Real, 0, None, closed="left")],
        "compute_score": ["boolean"],
        "threshold_lambda": [Interval(Real, 0, None, closed="left")],
        "fit_intercept": ["boolean"],
        "copy_X": ["boolean"],
        "verbose": ["verbose"],
    }

    def __init__(
        self,
        *,
        max_iter=300,
        tol=1.0e-3,
        alpha_1=1.0e-6,
        alpha_2=1.0e-6,
        lambda_1=1.0e-6,
        lambda_2=1.0e-6,
        compute_score=False,
        threshold_lambda=1.0e4,
        fit_intercept=True,
        copy_X=True,
        verbose=False,
    ):
        # [FUNCTION DOC]
        # Stores hyperparameters unchanged: max_iter, tol, alpha_1, alpha_2, lambda_1,
        #   lambda_2, compute_score, threshold_lambda, fit_intercept, copy_X, verbose.
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.compute_score = compute_score
        self.threshold_lambda = threshold_lambda
        self.copy_X = copy_X
        self.verbose = verbose

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y):
        # [FUNCTION DOC]
        # Fits the model. Steps: validate X, y (at least 2 samples); center via
        #   _preprocess_data. Init alpha to 1/(var(y)+eps), all lambda to 1, all features
        #   kept. Choose _update_sigma when n_samples >= n_features else
        #   _update_sigma_woodbury. Loop up to max_iter: compute sigma for kept features,
        #   set kept coefs to alpha * sigma X_keep^T y, compute SSE and gamma = 1 - lambda
        #   * diag(sigma), update kept lambdas as (gamma + 2 lambda_1) / (coef^2 + 2
        #   lambda_2) and alpha as (n_samples - sum gamma + 2 alpha_1) / (SSE + 2
        #   alpha_2); prune features with lambda >= threshold_lambda (zero their coef);
        #   optionally record the objective; stop on L1 coef change below tol or when no
        #   feature is left. Finally recompute sigma and coef if features remain (else
        #   sigma_ is an empty 0x0 array) and set the intercept.
        # Inputs: X (n_samples, n_features); y (n_samples,).
        # Outputs: self.
        # Side effects: sets X_offset_, X_scale_, scores_ (stays a Python list),
        #   n_iter_, coef_, alpha_, sigma_ (only kept features), lambda_, intercept_;
        #   prints if verbose.
        """Fit the model according to the given training data and parameters.

        Iterative procedure to maximize the evidence

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vector, where `n_samples` is the number of samples and
            `n_features` is the number of features.
        y : array-like of shape (n_samples,)
            Target values (integers). Will be cast to X's dtype if necessary.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        X, y = validate_data(
            self,
            X,
            y,
            dtype=[np.float64, np.float32],
            force_writeable=True,
            y_numeric=True,
            ensure_min_samples=2,
        )
        dtype = X.dtype

        n_samples, n_features = X.shape
        coef_ = np.zeros(n_features, dtype=dtype)

        X, y, X_offset_, y_offset_, X_scale_, _ = _preprocess_data(
            X, y, fit_intercept=self.fit_intercept, copy=self.copy_X
        )

        self.X_offset_ = X_offset_
        self.X_scale_ = X_scale_

        # Launch the convergence loop
        keep_lambda = np.ones(n_features, dtype=bool)

        lambda_1 = self.lambda_1
        lambda_2 = self.lambda_2
        alpha_1 = self.alpha_1
        alpha_2 = self.alpha_2
        verbose = self.verbose

        # Initialization of the values of the parameters
        eps = np.finfo(np.float64).eps
        # Add `eps` in the denominator to omit division by zero if `np.var(y)`
        # is zero.
        # Explicitly set dtype to avoid unintended type promotion with numpy 2.
        alpha_ = np.asarray(1.0 / (np.var(y) + eps), dtype=dtype)
        lambda_ = np.ones(n_features, dtype=dtype)

        self.scores_ = list()
        coef_old_ = None

        def update_coeff(X, y, coef_, alpha_, keep_lambda, sigma_):
            # [FUNCTION DOC]
            # HELPER: Sets the coefficients of kept features to alpha times sigma times
            #   the kept columns of X transposed times y.
            # Side effects: modifies coef_ in place and returns it.
            coef_[keep_lambda] = alpha_ * np.linalg.multi_dot(
                [sigma_, X[:, keep_lambda].T, y]
            )
            return coef_

        update_sigma = (
            self._update_sigma
            if n_samples >= n_features
            else self._update_sigma_woodbury
        )
        # Iterative procedure of ARDRegression
        for iter_ in range(self.max_iter):
            sigma_ = update_sigma(X, alpha_, lambda_, keep_lambda)
            coef_ = update_coeff(X, y, coef_, alpha_, keep_lambda, sigma_)

            # Update alpha and lambda
            sse_ = np.sum((y - np.dot(X, coef_)) ** 2)
            gamma_ = 1.0 - lambda_[keep_lambda] * np.diag(sigma_)
            lambda_[keep_lambda] = (gamma_ + 2.0 * lambda_1) / (
                (coef_[keep_lambda]) ** 2 + 2.0 * lambda_2
            )
            alpha_ = (n_samples - gamma_.sum() + 2.0 * alpha_1) / (sse_ + 2.0 * alpha_2)

            # Prune the weights with a precision over a threshold
            keep_lambda = lambda_ < self.threshold_lambda
            coef_[~keep_lambda] = 0

            # Compute the objective function
            if self.compute_score:
                s = (lambda_1 * np.log(lambda_) - lambda_2 * lambda_).sum()
                s += alpha_1 * log(alpha_) - alpha_2 * alpha_
                s += 0.5 * (
                    fast_logdet(sigma_)
                    + n_samples * log(alpha_)
                    + np.sum(np.log(lambda_))
                )
                s -= 0.5 * (alpha_ * sse_ + (lambda_ * coef_**2).sum())
                self.scores_.append(s)

            # Check for convergence
            if iter_ > 0 and np.sum(np.abs(coef_old_ - coef_)) < self.tol:
                if verbose:
                    print("Converged after %s iterations" % iter_)
                break
            coef_old_ = np.copy(coef_)

            if not keep_lambda.any():
                break

        self.n_iter_ = iter_ + 1

        if keep_lambda.any():
            # update sigma and mu using updated params from the last iteration
            sigma_ = update_sigma(X, alpha_, lambda_, keep_lambda)
            coef_ = update_coeff(X, y, coef_, alpha_, keep_lambda, sigma_)
        else:
            sigma_ = np.array([]).reshape(0, 0)

        self.coef_ = coef_
        self.alpha_ = alpha_
        self.sigma_ = sigma_
        self.lambda_ = lambda_
        self._set_intercept(X_offset_, y_offset_, X_scale_)
        return self

    def _update_sigma_woodbury(self, X, alpha_, lambda_, keep_lambda):
        # [FUNCTION DOC]
        # HELPER: Posterior covariance for kept features when n_samples < n_features via
        #   the Woodbury identity: inverts an (n_samples x n_samples) matrix (identity /
        #   alpha plus X_keep Lambda^-1 X_keep^T) with pinvh and forms Lambda^-1 minus
        #   Lambda^-1 X_keep^T (that inverse) X_keep Lambda^-1.
        # Outputs: (n_kept, n_kept) array.
        # See slides as referenced in the docstring note
        # this function is used when n_samples < n_features and will invert
        # a matrix of shape (n_samples, n_samples) making use of the
        # woodbury formula:
        # https://en.wikipedia.org/wiki/Woodbury_matrix_identity
        n_samples = X.shape[0]
        X_keep = X[:, keep_lambda]
        inv_lambda = 1 / lambda_[keep_lambda].reshape(1, -1)
        sigma_ = pinvh(
            np.eye(n_samples, dtype=X.dtype) / alpha_
            + np.dot(X_keep * inv_lambda, X_keep.T)
        )
        sigma_ = np.dot(sigma_, X_keep * inv_lambda)
        sigma_ = -np.dot(inv_lambda.reshape(-1, 1) * X_keep.T, sigma_)
        sigma_[np.diag_indices(sigma_.shape[1])] += 1.0 / lambda_[keep_lambda]
        return sigma_

    def _update_sigma(self, X, alpha_, lambda_, keep_lambda):
        # [FUNCTION DOC]
        # HELPER: Posterior covariance for kept features by pseudo-inverting
        #   diag(lambda_keep) plus alpha times the Gram matrix of kept columns.
        # Outputs: (n_kept, n_kept) array.
        # See slides as referenced in the docstring note
        # this function is used when n_samples >= n_features and will
        # invert a matrix of shape (n_features, n_features)
        X_keep = X[:, keep_lambda]
        gram = np.dot(X_keep.T, X_keep)
        eye = np.eye(gram.shape[0], dtype=X.dtype)
        sigma_inv = lambda_[keep_lambda] * eye + alpha_ * gram
        sigma_ = pinvh(sigma_inv)
        return sigma_

    def predict(self, X, return_std=False):
        # [FUNCTION DOC]
        # Predicts with the linear model; if return_std, also returns predictive std per
        #   sample using only kept columns (lambda_ below threshold_lambda) of the
        #   centered X with sigma_, plus 1/alpha_.
        # Outputs: y_mean (n_samples,) or tuple (y_mean, y_std).
        # KNOWN ISSUE: when every feature was pruned, sigma_ is 0x0 and the std is just
        #   sqrt(1/alpha_); X must be dense for the std branch.
        """Predict using the linear model.

        In addition to the mean of the predictive distribution, also its
        standard deviation can be returned.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Samples.

        return_std : bool, default=False
            Whether to return the standard deviation of posterior prediction.

        Returns
        -------
        y_mean : array-like of shape (n_samples,)
            Mean of predictive distribution of query points.

        y_std : array-like of shape (n_samples,)
            Standard deviation of predictive distribution of query points.
        """
        y_mean = self._decision_function(X)
        if return_std is False:
            return y_mean
        else:
            col_index = self.lambda_ < self.threshold_lambda
            X = X - self.X_offset_
            X = _safe_indexing(X, indices=col_index, axis=1)
            sigmas_squared_data = (np.dot(X, self.sigma_) * X).sum(axis=1)
            y_std = np.sqrt(sigmas_squared_data + (1.0 / self.alpha_))
            return y_mean, y_std
