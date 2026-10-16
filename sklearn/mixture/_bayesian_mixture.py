# [MODULE DOC]
# Implements BayesianGaussianMixture, a variational-inference Gaussian mixture with
#   either a finite Dirichlet distribution prior or a truncated (stick-breaking)
#   Dirichlet process prior on the mixture weights, a Gaussian prior on means and a
#   Wishart prior on precisions.
# Reuses the EM driver of BaseMixture (fit loop, n_init restarts, convergence on the
#   lower bound) and the Gaussian helpers of _gaussian_mixture (sufficient statistics,
#   Cholesky precisions, log-probabilities).
# Also defines two private helpers giving log normalization constants of the Dirichlet
#   and Wishart distributions.
"""Bayesian Gaussian Mixture Model."""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import math
from numbers import Real

import numpy as np
from scipy.special import betaln, digamma, gammaln

from sklearn.mixture._base import BaseMixture, _check_shape
from sklearn.mixture._gaussian_mixture import (
    _check_precision_matrix,
    _check_precision_positivity,
    _compute_log_det_cholesky,
    _compute_precision_cholesky,
    _estimate_gaussian_parameters,
    _estimate_log_gaussian_prob,
)
from sklearn.utils import check_array
from sklearn.utils._param_validation import Interval, StrOptions


def _log_dirichlet_norm(dirichlet_concentration):
    # [FUNCTION DOC]
    # HELPER: Log normalization constant of a Dirichlet distribution: log-gamma of the
    #   sum of the concentrations minus the sum of log-gamma of each concentration.
    # Inputs: dirichlet_concentration - 1-D array of positive concentrations (one per
    #   component; the docstring's "n_samples" shape name is misleading).
    # Outputs: scalar float.
    # Exceptions/side effects: none.
    """Compute the log of the Dirichlet distribution normalization term.

    Parameters
    ----------
    dirichlet_concentration : array-like of shape (n_samples,)
        The parameters values of the Dirichlet distribution.

    Returns
    -------
    log_dirichlet_norm : float
        The log normalization of the Dirichlet distribution.
    """
    return gammaln(np.sum(dirichlet_concentration)) - np.sum(
        gammaln(dirichlet_concentration)
    )


def _log_wishart_norm(degrees_of_freedom, log_det_precisions_chol, n_features):
    # [FUNCTION DOC]
    # HELPER: Log normalization constant of the Wishart distribution for each component,
    #   with the constant pi term dropped.
    # Computes, per component, the negative of: dof times log-det of the precision
    #   Cholesky, plus dof * n_features * 0.5 * log(2), plus the sum over feature index
    #   i in [0, n_features) of log-gamma(0.5 * (dof - i)).
    # Inputs: degrees_of_freedom - array (n_components,) or scalar;
    #   log_det_precisions_chol - array (n_components,) or scalar; n_features - int.
    # Outputs: array of shape (n_components,) (or scalar-like for scalar inputs).
    # Side effects: none.
    """Compute the log of the Wishart distribution normalization term.

    Parameters
    ----------
    degrees_of_freedom : array-like of shape (n_components,)
        The number of degrees of freedom on the covariance Wishart
        distributions.

    log_det_precisions_chol : array-like of shape (n_components,)
         The determinant of the precision matrix for each component.

    n_features : int
        The number of features.

    Return
    ------
    log_wishart_norm : array-like of shape (n_components,)
        The log normalization of the Wishart distribution.
    """
    # To simplify the computation we have removed the np.log(np.pi) term
    return -(
        degrees_of_freedom * log_det_precisions_chol
        + degrees_of_freedom * n_features * 0.5 * math.log(2.0)
        + np.sum(
            gammaln(0.5 * (degrees_of_freedom - np.arange(n_features)[:, np.newaxis])),
            0,
        )
    )


class BayesianGaussianMixture(BaseMixture):
    # [CLASS DOC]
    # Variational Bayesian Gaussian mixture estimator. Subclasses BaseMixture and plugs
    #   in the variational E/M quantities: the M step updates posterior hyperparameters
    #   (weight_concentration_, mean_precision_, means_, degrees_of_freedom_,
    #   covariances_) from responsibility-weighted sufficient statistics; the E step
    #   uses expected log weights (digamma based) and expected log Gaussian densities.
    # Priors are resolved at fit time into trailing-underscore attributes by
    #   _check_parameters. covariances_ are stored normalized by degrees_of_freedom_
    #   (unlike Bishop), so precisions_cholesky_ is computed from those normalized
    #   covariances and the log-probability / lower bound compensate with a log(dof)
    #   term.
    """Variational Bayesian estimation of a Gaussian mixture.

    This class allows to infer an approximate posterior distribution over the
    parameters of a Gaussian mixture distribution. The effective number of
    components can be inferred from the data.

    This class implements two types of prior for the weights distribution: a
    finite mixture model with Dirichlet distribution and an infinite mixture
    model with the Dirichlet Process. In practice Dirichlet Process inference
    algorithm is approximated and uses a truncated distribution with a fixed
    maximum number of components (called the Stick-breaking representation).
    The number of components actually used almost always depends on the data.

    .. versionadded:: 0.18

    Read more in the :ref:`User Guide <bgmm>`.

    Parameters
    ----------
    n_components : int, default=1
        The number of mixture components. Depending on the data and the value
        of the `weight_concentration_prior` the model can decide to not use
        all the components by setting some component `weights_` to values very
        close to zero. The number of effective components is therefore smaller
        than n_components.

    covariance_type : {'full', 'tied', 'diag', 'spherical'}, default='full'
        String describing the type of covariance parameters to use.
        Must be one of:

        - 'full' (each component has its own general covariance matrix),
        - 'tied' (all components share the same general covariance matrix),
        - 'diag' (each component has its own diagonal covariance matrix),
        - 'spherical' (each component has its own single variance).

    tol : float, default=1e-3
        The convergence threshold. EM iterations will stop when the
        lower bound average gain on the likelihood (of the training data with
        respect to the model) is below this threshold.

    reg_covar : float, default=1e-6
        Non-negative regularization added to the diagonal of covariance.
        Allows to assure that the covariance matrices are all positive.

    max_iter : int, default=100
        The number of EM iterations to perform.

    n_init : int, default=1
        The number of initializations to perform. The result with the highest
        lower bound value on the likelihood is kept.

    init_params : {'kmeans', 'k-means++', 'random', 'random_from_data'}, \
    default='kmeans'
        The method used to initialize the weights, the means and the
        covariances. String must be one of:

        - 'kmeans': responsibilities are initialized using kmeans.
        - 'k-means++': use the k-means++ method to initialize.
        - 'random': responsibilities are initialized randomly.
        - 'random_from_data': initial means are randomly selected data points.

        .. versionchanged:: v1.1
            `init_params` now accepts 'random_from_data' and 'k-means++' as
            initialization methods.

    weight_concentration_prior_type : {'dirichlet_process', 'dirichlet_distribution'}, \
            default='dirichlet_process'
        String describing the type of the weight concentration prior.

    weight_concentration_prior : float or None, default=None
        The dirichlet concentration of each component on the weight
        distribution (Dirichlet). This is commonly called gamma in the
        literature. The higher concentration puts more mass in
        the center and will lead to more components being active, while a lower
        concentration parameter will lead to more mass at the edge of the
        mixture weights simplex. The value of the parameter must be greater
        than 0. If it is None, it's set to ``1. / n_components``.

    mean_precision_prior : float or None, default=None
        The precision prior on the mean distribution (Gaussian).
        Controls the extent of where means can be placed. Larger
        values concentrate the cluster means around `mean_prior`.
        The value of the parameter must be greater than 0.
        If it is None, it is set to 1.

    mean_prior : array-like, shape (n_features,), default=None
        The prior on the mean distribution (Gaussian).
        If it is None, it is set to the mean of X.

    degrees_of_freedom_prior : float or None, default=None
        The prior of the number of degrees of freedom on the covariance
        distributions (Wishart). If it is None, it's set to `n_features`.

    covariance_prior : float or array-like, default=None
        The prior on the covariance distribution (Wishart).
        If it is None, the emiprical covariance prior is initialized using the
        covariance of X. The shape depends on `covariance_type`::

                (n_features, n_features) if 'full',
                (n_features, n_features) if 'tied',
                (n_features)             if 'diag',
                float                    if 'spherical'

    random_state : int, RandomState instance or None, default=None
        Controls the random seed given to the method chosen to initialize the
        parameters (see `init_params`).
        In addition, it controls the generation of random samples from the
        fitted distribution (see the method `sample`).
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    warm_start : bool, default=False
        If 'warm_start' is True, the solution of the last fitting is used as
        initialization for the next call of fit(). This can speed up
        convergence when fit is called several times on similar problems.
        See :term:`the Glossary <warm_start>`.

    verbose : int, default=0
        Enable verbose output. If 1 then it prints the current
        initialization and each iteration step. If greater than 1 then
        it prints also the log probability and the time needed
        for each step.

    verbose_interval : int, default=10
        Number of iteration done before the next print.

    Attributes
    ----------
    weights_ : array-like of shape (n_components,)
        The weights of each mixture components.

    means_ : array-like of shape (n_components, n_features)
        The mean of each mixture component.

    covariances_ : array-like
        The covariance of each mixture component.
        The shape depends on `covariance_type`::

            (n_components,)                        if 'spherical',
            (n_features, n_features)               if 'tied',
            (n_components, n_features)             if 'diag',
            (n_components, n_features, n_features) if 'full'

    precisions_ : array-like
        The precision matrices for each component in the mixture. A precision
        matrix is the inverse of a covariance matrix. A covariance matrix is
        symmetric positive definite so the mixture of Gaussian can be
        equivalently parameterized by the precision matrices. Storing the
        precision matrices instead of the covariance matrices makes it more
        efficient to compute the log-likelihood of new samples at test time.
        The shape depends on ``covariance_type``::

            (n_components,)                        if 'spherical',
            (n_features, n_features)               if 'tied',
            (n_components, n_features)             if 'diag',
            (n_components, n_features, n_features) if 'full'

    precisions_cholesky_ : array-like
        The Cholesky decomposition of the precision matrices of each mixture
        component. A precision matrix is the inverse of a covariance matrix.
        A covariance matrix is symmetric positive definite so the mixture of
        Gaussian can be equivalently parameterized by the precision matrices.
        Storing the precision matrices instead of the covariance matrices makes
        it more efficient to compute the log-likelihood of new samples at test
        time. The shape depends on ``covariance_type``::

            (n_components,)                        if 'spherical',
            (n_features, n_features)               if 'tied',
            (n_components, n_features)             if 'diag',
            (n_components, n_features, n_features) if 'full'

    converged_ : bool
        True when convergence of the best fit of inference was reached, False otherwise.

    n_iter_ : int
        Number of step used by the best fit of inference to reach the
        convergence.

    lower_bound_ : float
        Lower bound value on the model evidence (of the training data) of the
        best fit of inference.

    lower_bounds_ : array-like of shape (`n_iter_`,)
        The list of lower bound values on the model evidence from each iteration
        of the best fit of inference.

    weight_concentration_prior_ : tuple or float
        The dirichlet concentration of each component on the weight
        distribution (Dirichlet). The type depends on
        ``weight_concentration_prior_type``::

            (float, float) if 'dirichlet_process' (Beta parameters),
            float          if 'dirichlet_distribution' (Dirichlet parameters).

        The higher concentration puts more mass in
        the center and will lead to more components being active, while a lower
        concentration parameter will lead to more mass at the edge of the
        simplex.

    weight_concentration_ : array-like of shape (n_components,)
        The dirichlet concentration of each component on the weight
        distribution (Dirichlet).

    mean_precision_prior_ : float
        The precision prior on the mean distribution (Gaussian).
        Controls the extent of where means can be placed.
        Larger values concentrate the cluster means around `mean_prior`.
        If mean_precision_prior is set to None, `mean_precision_prior_` is set
        to 1.

    mean_precision_ : array-like of shape (n_components,)
        The precision of each components on the mean distribution (Gaussian).

    mean_prior_ : array-like of shape (n_features,)
        The prior on the mean distribution (Gaussian).

    degrees_of_freedom_prior_ : float
        The prior of the number of degrees of freedom on the covariance
        distributions (Wishart).

    degrees_of_freedom_ : array-like of shape (n_components,)
        The number of degrees of freedom of each components in the model.

    covariance_prior_ : float or array-like
        The prior on the covariance distribution (Wishart).
        The shape depends on `covariance_type`::

            (n_features, n_features) if 'full',
            (n_features, n_features) if 'tied',
            (n_features)             if 'diag',
            float                    if 'spherical'

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    GaussianMixture : Finite Gaussian mixture fit with EM.

    References
    ----------

    .. [1] `Bishop, Christopher M. (2006). "Pattern recognition and machine
       learning". Vol. 4 No. 4. New York: Springer.
       <https://www.springer.com/kr/book/9780387310732>`_

    .. [2] `Hagai Attias. (2000). "A Variational Bayesian Framework for
       Graphical Models". In Advances in Neural Information Processing
       Systems 12.
       <https://proceedings.neurips.cc/paper_files/paper/1999/file/74563ba21a90da13dacf2a73e3ddefa7-Paper.pdf>`_

    .. [3] `Blei, David M. and Michael I. Jordan. (2006). "Variational
       inference for Dirichlet process mixtures". Bayesian analysis 1.1
       <https://www.cs.princeton.edu/courses/archive/fall11/cos597C/reading/BleiJordan2005.pdf>`_

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.mixture import BayesianGaussianMixture
    >>> X = np.array([[1, 2], [1, 4], [1, 0], [4, 2], [12, 4], [10, 7]])
    >>> bgm = BayesianGaussianMixture(n_components=2, random_state=42).fit(X)
    >>> bgm.means_
    array([[2.49 , 2.29],
           [8.45, 4.52 ]])
    >>> bgm.predict([[0, 0], [9, 3]])
    array([0, 1])
    """

    _parameter_constraints: dict = {
        **BaseMixture._parameter_constraints,
        "covariance_type": [StrOptions({"spherical", "tied", "diag", "full"})],
        "weight_concentration_prior_type": [
            StrOptions({"dirichlet_process", "dirichlet_distribution"})
        ],
        "weight_concentration_prior": [
            None,
            Interval(Real, 0.0, None, closed="neither"),
        ],
        "mean_precision_prior": [None, Interval(Real, 0.0, None, closed="neither")],
        "mean_prior": [None, "array-like"],
        "degrees_of_freedom_prior": [None, Interval(Real, 0.0, None, closed="neither")],
        "covariance_prior": [
            None,
            "array-like",
            Interval(Real, 0.0, None, closed="neither"),
        ],
    }

    def __init__(
        self,
        *,
        n_components=1,
        covariance_type="full",
        tol=1e-3,
        reg_covar=1e-6,
        max_iter=100,
        n_init=1,
        init_params="kmeans",
        weight_concentration_prior_type="dirichlet_process",
        weight_concentration_prior=None,
        mean_precision_prior=None,
        mean_prior=None,
        degrees_of_freedom_prior=None,
        covariance_prior=None,
        random_state=None,
        warm_start=False,
        verbose=0,
        verbose_interval=10,
    ):
        # [FUNCTION DOC]
        # Stores all constructor parameters verbatim (shared ones forwarded to
        #   BaseMixture.__init__); no validation or computation.
        # Inputs: see class docstring; all keyword-only.
        # Side effects: sets the matching public attributes on self.
        super().__init__(
            n_components=n_components,
            tol=tol,
            reg_covar=reg_covar,
            max_iter=max_iter,
            n_init=n_init,
            init_params=init_params,
            random_state=random_state,
            warm_start=warm_start,
            verbose=verbose,
            verbose_interval=verbose_interval,
        )

        self.covariance_type = covariance_type
        self.weight_concentration_prior_type = weight_concentration_prior_type
        self.weight_concentration_prior = weight_concentration_prior
        self.mean_precision_prior = mean_precision_prior
        self.mean_prior = mean_prior
        self.degrees_of_freedom_prior = degrees_of_freedom_prior
        self.covariance_prior = covariance_prior

    def _check_parameters(self, X, xp=None):
        # [FUNCTION DOC]
        # Resolves all prior hyperparameters from the constructor values and the
        #   training data by calling the weights, means, precision (degrees of freedom)
        #   and covariance-prior checkers in order.
        # Inputs: X - array (n_samples, n_features); xp - unused array-namespace
        #   argument kept for the base-class signature.
        # Exceptions: propagates ValueError from the individual checkers.
        # Side effects: sets weight_concentration_prior_, mean_precision_prior_,
        #   mean_prior_, degrees_of_freedom_prior_, covariance_prior_.
        """Check that the parameters are well defined.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        """
        self._check_weights_parameters()
        self._check_means_parameters(X)
        self._check_precision_parameters(X)
        self._checkcovariance_prior_parameter(X)

    def _check_weights_parameters(self):
        # [FUNCTION DOC]
        # Sets weight_concentration_prior_ to weight_concentration_prior, or to 1 /
        #   n_components when it is None.
        # Side effects: sets self.weight_concentration_prior_.
        """Check the parameter of the Dirichlet distribution."""
        if self.weight_concentration_prior is None:
            self.weight_concentration_prior_ = 1.0 / self.n_components
        else:
            self.weight_concentration_prior_ = self.weight_concentration_prior

    def _check_means_parameters(self, X):
        # [FUNCTION DOC]
        # Sets mean_precision_prior_ (default 1.0) and mean_prior_ (default: column
        #   means of X; otherwise the user value converted to a float array without
        #   forcing 2-D).
        # Inputs: X - array (n_samples, n_features).
        # Exceptions: ValueError via _check_shape when a provided mean_prior does not
        #   have shape (n_features,).
        # Side effects: sets self.mean_precision_prior_ and self.mean_prior_.
        """Check the parameters of the Gaussian distribution.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        """
        _, n_features = X.shape

        if self.mean_precision_prior is None:
            self.mean_precision_prior_ = 1.0
        else:
            self.mean_precision_prior_ = self.mean_precision_prior

        if self.mean_prior is None:
            self.mean_prior_ = X.mean(axis=0)
        else:
            self.mean_prior_ = check_array(
                self.mean_prior, dtype=[np.float64, np.float32], ensure_2d=False
            )
            _check_shape(self.mean_prior_, (n_features,), "means")

    def _check_precision_parameters(self, X):
        # [FUNCTION DOC]
        # Sets degrees_of_freedom_prior_: n_features when None, otherwise the user value
        #   provided it is strictly greater than n_features - 1.
        # Inputs: X - array (n_samples, n_features), used only for n_features.
        # Exceptions: ValueError when degrees_of_freedom_prior <= n_features - 1.
        # Side effects: sets self.degrees_of_freedom_prior_.
        """Check the prior parameters of the precision distribution.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        """
        _, n_features = X.shape

        if self.degrees_of_freedom_prior is None:
            self.degrees_of_freedom_prior_ = n_features
        elif self.degrees_of_freedom_prior > n_features - 1.0:
            self.degrees_of_freedom_prior_ = self.degrees_of_freedom_prior
        else:
            raise ValueError(
                "The parameter 'degrees_of_freedom_prior' "
                "should be greater than %d, but got %.3f."
                % (n_features - 1, self.degrees_of_freedom_prior)
            )

    def _checkcovariance_prior_parameter(self, X):
        # [FUNCTION DOC]
        # Sets covariance_prior_. When None it is derived empirically from X: the (at
        #   least 2-D) sample covariance for 'full' and 'tied', per-feature unbiased
        #   variances for 'diag', and the mean of those variances for 'spherical'.
        #   Otherwise the user value is converted to a float array and validated: shape
        #   (n_features, n_features) and symmetric positive definite for 'full'/'tied';
        #   shape (n_features,) and strictly positive for 'diag'; used as-is for
        #   'spherical'.
        # Inputs: X - array (n_samples, n_features).
        # Exceptions: ValueError from _check_shape, _check_precision_matrix or
        #   _check_precision_positivity.
        # Side effects: sets self.covariance_prior_.
        # KNOWN ISSUE: the method name is missing an underscore
        #   ("_checkcovariance_prior_parameter"), unlike its sibling checkers. The
        #   'spherical' user value is not validated for positivity here (only the
        #   parameter constraints apply).
        """Check the `covariance_prior_`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        """
        _, n_features = X.shape

        if self.covariance_prior is None:
            self.covariance_prior_ = {
                "full": np.atleast_2d(np.cov(X.T)),
                "tied": np.atleast_2d(np.cov(X.T)),
                "diag": np.var(X, axis=0, ddof=1),
                "spherical": np.var(X, axis=0, ddof=1).mean(),
            }[self.covariance_type]

        elif self.covariance_type in ["full", "tied"]:
            self.covariance_prior_ = check_array(
                self.covariance_prior, dtype=[np.float64, np.float32], ensure_2d=False
            )
            _check_shape(
                self.covariance_prior_,
                (n_features, n_features),
                "%s covariance_prior" % self.covariance_type,
            )
            _check_precision_matrix(self.covariance_prior_, self.covariance_type)
        elif self.covariance_type == "diag":
            self.covariance_prior_ = check_array(
                self.covariance_prior, dtype=[np.float64, np.float32], ensure_2d=False
            )
            _check_shape(
                self.covariance_prior_,
                (n_features,),
                "%s covariance_prior" % self.covariance_type,
            )
            _check_precision_positivity(self.covariance_prior_, self.covariance_type)
        # spherical case
        else:
            self.covariance_prior_ = self.covariance_prior

    def _initialize(self, X, resp):
        # [FUNCTION DOC]
        # Initializes the variational posterior from initial responsibilities: computes
        #   Gaussian sufficient statistics (nk, xk, sk) with reg_covar and then runs the
        #   weights, means and precisions update steps.
        # Inputs: X - array (n_samples, n_features); resp - array (n_samples,
        #   n_components) of initial responsibilities.
        # Side effects: sets weight_concentration_, mean_precision_, means_,
        #   degrees_of_freedom_, covariances_, precisions_cholesky_.
        """Initialization of the mixture parameters.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        resp : array-like of shape (n_samples, n_components)
        """
        nk, xk, sk = _estimate_gaussian_parameters(
            X, resp, self.reg_covar, self.covariance_type
        )

        self._estimate_weights(nk)
        self._estimate_means(nk, xk)
        self._estimate_precisions(nk, xk, sk)

    def _estimate_weights(self, nk):
        # [FUNCTION DOC]
        # Updates the weight posterior. For 'dirichlet_process' sets
        #   weight_concentration_ to a tuple of two Beta parameter arrays: first is 1 +
        #   nk; second is weight_concentration_prior_ plus, for each component k, the
        #   sum of nk over all components after k (0 for the last one). For
        #   'dirichlet_distribution' it is weight_concentration_prior_ + nk.
        # Inputs: nk - array (n_components,) of effective counts.
        # Side effects: sets self.weight_concentration_.
        """Estimate the parameters of the Dirichlet distribution.

        Parameters
        ----------
        nk : array-like of shape (n_components,)
        """
        if self.weight_concentration_prior_type == "dirichlet_process":
            # For dirichlet process weight_concentration will be a tuple
            # containing the two parameters of the beta distribution
            self.weight_concentration_ = (
                1.0 + nk,
                (
                    self.weight_concentration_prior_
                    + np.hstack((np.cumsum(nk[::-1])[-2::-1], 0))
                ),
            )
        else:
            # case Variational Gaussian mixture with dirichlet distribution
            self.weight_concentration_ = self.weight_concentration_prior_ + nk

    def _estimate_means(self, nk, xk):
        # [FUNCTION DOC]
        # Updates the Gaussian posterior on means: mean_precision_ =
        #   mean_precision_prior_ + nk; means_ is the precision-weighted average of
        #   mean_prior_ (weight mean_precision_prior_) and xk (weight nk), divided per
        #   component by mean_precision_.
        # Inputs: nk - array (n_components,); xk - array (n_components, n_features) of
        #   weighted means.
        # Side effects: sets self.mean_precision_ and self.means_.
        """Estimate the parameters of the Gaussian distribution.

        Parameters
        ----------
        nk : array-like of shape (n_components,)

        xk : array-like of shape (n_components, n_features)
        """
        self.mean_precision_ = self.mean_precision_prior_ + nk
        self.means_ = (
            self.mean_precision_prior_ * self.mean_prior_ + nk[:, np.newaxis] * xk
        ) / self.mean_precision_[:, np.newaxis]

    def _estimate_precisions(self, nk, xk, sk):
        # [FUNCTION DOC]
        # Dispatches to the Wishart update for the current covariance_type, then
        #   computes precisions_cholesky_ from the (normalized) covariances_.
        # Inputs: nk, xk as in _estimate_means; sk - covariance statistics shaped per
        #   covariance_type.
        # Exceptions: ValueError from _compute_precision_cholesky if a covariance is not
        #   positive definite.
        # Side effects: sets degrees_of_freedom_, covariances_, precisions_cholesky_.
        """Estimate the precisions parameters of the precision distribution.

        Parameters
        ----------
        nk : array-like of shape (n_components,)

        xk : array-like of shape (n_components, n_features)

        sk : array-like
            The shape depends of `covariance_type`:
            'full' : (n_components, n_features, n_features)
            'tied' : (n_features, n_features)
            'diag' : (n_components, n_features)
            'spherical' : (n_components,)
        """
        {
            "full": self._estimate_wishart_full,
            "tied": self._estimate_wishart_tied,
            "diag": self._estimate_wishart_diag,
            "spherical": self._estimate_wishart_spherical,
        }[self.covariance_type](nk, xk, sk)

        self.precisions_cholesky_ = _compute_precision_cholesky(
            self.covariances_, self.covariance_type
        )

    def _estimate_wishart_full(self, nk, xk, sk):
        # [FUNCTION DOC]
        # Full-covariance Wishart update: degrees_of_freedom_ = prior dof + nk; for each
        #   component, covariance = covariance_prior_ + nk * sk + nk *
        #   mean_precision_prior_ / mean_precision_ * outer product of (xk -
        #   mean_prior_) with itself; finally each covariance is divided by its degrees
        #   of freedom.
        # Inputs: nk (n_components,), xk (n_components, n_features), sk (n_components,
        #   n_features, n_features).
        # Side effects: sets degrees_of_freedom_ and covariances_ (allocated as
        #   float64).
        """Estimate the full Wishart distribution parameters.

        Parameters
        ----------
        nk : array-like of shape (n_components,)

        xk : array-like of shape (n_components, n_features)

        sk : array-like of shape (n_components, n_features, n_features)
        """
        _, n_features = xk.shape

        # Warning : in some Bishop book, there is a typo on the formula 10.63
        # `degrees_of_freedom_k = degrees_of_freedom_0 + Nk` is
        # the correct formula
        self.degrees_of_freedom_ = self.degrees_of_freedom_prior_ + nk

        self.covariances_ = np.empty((self.n_components, n_features, n_features))

        for k in range(self.n_components):
            diff = xk[k] - self.mean_prior_
            self.covariances_[k] = (
                self.covariance_prior_
                + nk[k] * sk[k]
                + nk[k]
                * self.mean_precision_prior_
                / self.mean_precision_[k]
                * np.outer(diff, diff)
            )

        # Contrary to the original bishop book, we normalize the covariances
        self.covariances_ /= self.degrees_of_freedom_[:, np.newaxis, np.newaxis]

    def _estimate_wishart_tied(self, nk, xk, sk):
        # [FUNCTION DOC]
        # Tied Wishart update: a single scalar degrees_of_freedom_ = prior dof + total
        #   count / n_components; single covariance = covariance_prior_ + sk * total
        #   count / n_components + mean_precision_prior_ / n_components * sum over k of
        #   (nk / mean_precision_) weighted outer products of (xk - mean_prior_); then
        #   divided by the dof.
        # Inputs: nk, xk as above; sk (n_features, n_features).
        # Side effects: sets degrees_of_freedom_ (scalar) and covariances_.
        # KNOWN ISSUE: n_features is computed but unused.
        """Estimate the tied Wishart distribution parameters.

        Parameters
        ----------
        nk : array-like of shape (n_components,)

        xk : array-like of shape (n_components, n_features)

        sk : array-like of shape (n_features, n_features)
        """
        _, n_features = xk.shape

        # Warning : in some Bishop book, there is a typo on the formula 10.63
        # `degrees_of_freedom_k = degrees_of_freedom_0 + Nk`
        # is the correct formula
        self.degrees_of_freedom_ = (
            self.degrees_of_freedom_prior_ + nk.sum() / self.n_components
        )

        diff = xk - self.mean_prior_
        self.covariances_ = (
            self.covariance_prior_
            + sk * nk.sum() / self.n_components
            + self.mean_precision_prior_
            / self.n_components
            * np.dot((nk / self.mean_precision_) * diff.T, diff)
        )

        # Contrary to the original bishop book, we normalize the covariances
        self.covariances_ /= self.degrees_of_freedom_

    def _estimate_wishart_diag(self, nk, xk, sk):
        # [FUNCTION DOC]
        # Diagonal Wishart update: degrees_of_freedom_ = prior dof + nk; covariances_ =
        #   covariance_prior_ + nk * (sk + (mean_precision_prior_ / mean_precision_) *
        #   squared (xk - mean_prior_)), elementwise per component and feature, then
        #   each row divided by its dof.
        # Inputs: nk, xk as above; sk (n_components, n_features).
        # Side effects: sets degrees_of_freedom_ and covariances_.
        # KNOWN ISSUE: n_features is computed but unused.
        """Estimate the diag Wishart distribution parameters.

        Parameters
        ----------
        nk : array-like of shape (n_components,)

        xk : array-like of shape (n_components, n_features)

        sk : array-like of shape (n_components, n_features)
        """
        _, n_features = xk.shape

        # Warning : in some Bishop book, there is a typo on the formula 10.63
        # `degrees_of_freedom_k = degrees_of_freedom_0 + Nk`
        # is the correct formula
        self.degrees_of_freedom_ = self.degrees_of_freedom_prior_ + nk

        diff = xk - self.mean_prior_
        self.covariances_ = self.covariance_prior_ + nk[:, np.newaxis] * (
            sk
            + (self.mean_precision_prior_ / self.mean_precision_)[:, np.newaxis]
            * np.square(diff)
        )

        # Contrary to the original bishop book, we normalize the covariances
        self.covariances_ /= self.degrees_of_freedom_[:, np.newaxis]

    def _estimate_wishart_spherical(self, nk, xk, sk):
        # [FUNCTION DOC]
        # Spherical Wishart update: degrees_of_freedom_ = prior dof + nk; covariances_ =
        #   covariance_prior_ + nk * (sk + mean_precision_prior_ / mean_precision_ *
        #   mean over features of squared (xk - mean_prior_)), then divided by dof.
        # Inputs: nk, xk as above; sk (n_components,).
        # Side effects: sets degrees_of_freedom_ and covariances_.
        # KNOWN ISSUE: n_features is computed but unused.
        """Estimate the spherical Wishart distribution parameters.

        Parameters
        ----------
        nk : array-like of shape (n_components,)

        xk : array-like of shape (n_components, n_features)

        sk : array-like of shape (n_components,)
        """
        _, n_features = xk.shape

        # Warning : in some Bishop book, there is a typo on the formula 10.63
        # `degrees_of_freedom_k = degrees_of_freedom_0 + Nk`
        # is the correct formula
        self.degrees_of_freedom_ = self.degrees_of_freedom_prior_ + nk

        diff = xk - self.mean_prior_
        self.covariances_ = self.covariance_prior_ + nk * (
            sk
            + self.mean_precision_prior_
            / self.mean_precision_
            * np.mean(np.square(diff), 1)
        )

        # Contrary to the original bishop book, we normalize the covariances
        self.covariances_ /= self.degrees_of_freedom_

    def _m_step(self, X, log_resp, xp=None):
        # [FUNCTION DOC]
        # Variational M step: exponentiates log_resp to responsibilities, recomputes
        #   sufficient statistics and runs the weights, means and precisions updates.
        # Inputs: X (n_samples, n_features); log_resp (n_samples, n_components); xp
        #   unused.
        # Side effects: updates all posterior attributes (see _initialize).
        # KNOWN ISSUE: n_samples is unpacked but unused.
        """M step.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        log_resp : array-like of shape (n_samples, n_components)
            Logarithm of the posterior probabilities (or responsibilities) of
            the point of each sample in X.
        """
        n_samples, _ = X.shape

        nk, xk, sk = _estimate_gaussian_parameters(
            X, np.exp(log_resp), self.reg_covar, self.covariance_type
        )
        self._estimate_weights(nk)
        self._estimate_means(nk, xk)
        self._estimate_precisions(nk, xk, sk)

    def _estimate_log_weights(self, xp=None):
        # [FUNCTION DOC]
        # Expected log mixture weights under the variational posterior. For
        #   'dirichlet_process': with a, b the Beta parameters, returns digamma(a) -
        #   digamma(a+b) plus the cumulative sum, over previous components only, of
        #   digamma(b) - digamma(a+b) (stick-breaking). Otherwise digamma(concentration)
        #   minus digamma of the total concentration.
        # Inputs: xp - unused.
        # Outputs: array (n_components,).
        # Side effects: none.
        if self.weight_concentration_prior_type == "dirichlet_process":
            digamma_sum = digamma(
                self.weight_concentration_[0] + self.weight_concentration_[1]
            )
            digamma_a = digamma(self.weight_concentration_[0])
            digamma_b = digamma(self.weight_concentration_[1])
            return (
                digamma_a
                - digamma_sum
                + np.hstack((0, np.cumsum(digamma_b - digamma_sum)[:-1]))
            )
        else:
            # case Variational Gaussian mixture with dirichlet distribution
            return digamma(self.weight_concentration_) - digamma(
                np.sum(self.weight_concentration_)
            )

    def _estimate_log_prob(self, X, xp=None):
        # [FUNCTION DOC]
        # Expected log Gaussian density per sample and component: the plug-in log
        #   Gaussian probability using precisions_cholesky_, minus 0.5 * n_features *
        #   log(dof) (undoing the covariance normalization), plus 0.5 * (log_lambda -
        #   n_features / mean_precision_), where log_lambda = n_features * log 2 + sum
        #   over i of digamma(0.5 * (dof - i)).
        # Inputs: X (n_samples, n_features); xp unused.
        # Outputs: array (n_samples, n_components).
        # Side effects: none.
        _, n_features = X.shape
        # We remove `n_features * np.log(self.degrees_of_freedom_)` because
        # the precision matrix is normalized
        log_gauss = _estimate_log_gaussian_prob(
            X, self.means_, self.precisions_cholesky_, self.covariance_type
        ) - 0.5 * n_features * np.log(self.degrees_of_freedom_)

        log_lambda = n_features * np.log(2.0) + np.sum(
            digamma(
                0.5
                * (self.degrees_of_freedom_ - np.arange(0, n_features)[:, np.newaxis])
            ),
            0,
        )

        return log_gauss + 0.5 * (log_lambda - n_features / self.mean_precision_)

    def _compute_lower_bound(self, log_resp, log_prob_norm):
        # [FUNCTION DOC]
        # Variational lower bound up to constants: minus the responsibility entropy term
        #   (sum of resp * log resp), minus the total Wishart log-normalization (for
        #   'tied' the single component value times n_components), minus the weight
        #   prior normalization (sum of log Beta function for the Dirichlet process,
        #   Dirichlet log-norm otherwise), minus 0.5 * n_features * sum of log
        #   mean_precision_. The log-det of precisions is corrected by -0.5 * n_features
        #   * log(dof) for the normalized covariances.
        # Inputs: log_resp (n_samples, n_components); log_prob_norm - unused (kept for
        #   the base-class interface).
        # Outputs: float.
        # Side effects: none.
        """Estimate the lower bound of the model.

        The lower bound on the likelihood (of the training data with respect to
        the model) is used to detect the convergence and has to increase at
        each iteration.

        Parameters
        ----------
        log_resp : array, shape (n_samples, n_components)
            Logarithm of the posterior probabilities (or responsibilities) of
            the point of each sample in X.

        log_prob_norm : float
            Logarithm of the probability of each sample in X.

        Returns
        -------
        lower_bound : float
        """
        # Contrary to the original formula, we have done some simplification
        # and removed all the constant terms.
        (n_features,) = self.mean_prior_.shape

        # We removed `.5 * n_features * np.log(self.degrees_of_freedom_)`
        # because the precision matrix is normalized.
        log_det_precisions_chol = _compute_log_det_cholesky(
            self.precisions_cholesky_, self.covariance_type, n_features
        ) - 0.5 * n_features * np.log(self.degrees_of_freedom_)

        if self.covariance_type == "tied":
            log_wishart = self.n_components * np.float64(
                _log_wishart_norm(
                    self.degrees_of_freedom_, log_det_precisions_chol, n_features
                )
            )
        else:
            log_wishart = np.sum(
                _log_wishart_norm(
                    self.degrees_of_freedom_, log_det_precisions_chol, n_features
                )
            )

        if self.weight_concentration_prior_type == "dirichlet_process":
            log_norm_weight = -np.sum(
                betaln(self.weight_concentration_[0], self.weight_concentration_[1])
            )
        else:
            log_norm_weight = _log_dirichlet_norm(self.weight_concentration_)

        return (
            -np.sum(np.exp(log_resp) * log_resp)
            - log_wishart
            - log_norm_weight
            - 0.5 * n_features * np.sum(np.log(self.mean_precision_))
        )

    def _get_parameters(self):
        # [FUNCTION DOC]
        # Returns the fitted posterior state as a tuple (weight_concentration_,
        #   mean_precision_, means_, degrees_of_freedom_, covariances_,
        #   precisions_cholesky_) used by BaseMixture to keep the best initialization.
        # Side effects: none.
        return (
            self.weight_concentration_,
            self.mean_precision_,
            self.means_,
            self.degrees_of_freedom_,
            self.covariances_,
            self.precisions_cholesky_,
        )

    def _set_parameters(self, params, xp=None):
        # [FUNCTION DOC]
        # Restores the posterior from a tuple produced by _get_parameters and derives
        #   weights_ and precisions_. For the Dirichlet process, weights are the
        #   expected stick-breaking weights (a/(a+b) times the cumulative product of
        #   earlier b/(a+b)) renormalized to sum to 1; otherwise the concentrations
        #   normalized to sum to 1. precisions_ is L L^T per component ('full'), L L^T
        #   ('tied') or the squared Cholesky values ('diag', 'spherical').
        # Inputs: params - 6-tuple; xp unused.
        # Side effects: sets the six posterior attributes plus weights_ and precisions_.
        (
            self.weight_concentration_,
            self.mean_precision_,
            self.means_,
            self.degrees_of_freedom_,
            self.covariances_,
            self.precisions_cholesky_,
        ) = params

        # Weights computation
        if self.weight_concentration_prior_type == "dirichlet_process":
            weight_dirichlet_sum = (
                self.weight_concentration_[0] + self.weight_concentration_[1]
            )
            tmp = self.weight_concentration_[1] / weight_dirichlet_sum
            self.weights_ = (
                self.weight_concentration_[0]
                / weight_dirichlet_sum
                * np.hstack((1, np.cumprod(tmp[:-1])))
            )
            self.weights_ /= np.sum(self.weights_)
        else:
            self.weights_ = self.weight_concentration_ / np.sum(
                self.weight_concentration_
            )

        # Precisions matrices computation
        if self.covariance_type == "full":
            self.precisions_ = np.array(
                [
                    np.dot(prec_chol, prec_chol.T)
                    for prec_chol in self.precisions_cholesky_
                ]
            )

        elif self.covariance_type == "tied":
            self.precisions_ = np.dot(
                self.precisions_cholesky_, self.precisions_cholesky_.T
            )
        else:
            self.precisions_ = self.precisions_cholesky_**2
