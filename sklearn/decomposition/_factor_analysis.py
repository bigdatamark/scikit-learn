# [MODULE DOC]
# Implements the FactorAnalysis estimator: a linear Gaussian latent variable model X = W
# z + mean + noise with per-feature (diagonal) noise variances, fitted by an SVD-based
# iterative maximum-likelihood procedure (Barber, Algorithm 21.1), plus an optional
# varimax/quartimax rotation of the loadings.
# Public contents: FactorAnalysis. Private helper: _ortho_rotation.
# Side effects at import: none.
"""Factor Analysis.

A latent linear variable model.

FactorAnalysis is similar to probabilistic PCA implemented by PCA.score
While PCA assumes Gaussian noise with the same variance for each
feature, the FactorAnalysis model assumes different variances for
each of them.

This implementation is based on David Barber's Book,
Bayesian Reasoning and Machine Learning,
http://www.cs.ucl.ac.uk/staff/d.barber/brml,
Algorithm 21.1
"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import warnings
from math import log, sqrt
from numbers import Integral, Real

import numpy as np
from scipy import linalg

from sklearn.base import (
    BaseEstimator,
    ClassNamePrefixFeaturesOutMixin,
    TransformerMixin,
    _fit_context,
)
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils import check_random_state
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.extmath import _randomized_svd, fast_logdet, squared_norm
from sklearn.utils.validation import check_is_fitted, validate_data


class FactorAnalysis(ClassNamePrefixFeaturesOutMixin, TransformerMixin, BaseEstimator):
    # [CLASS DOC]
    # Estimator storing hyperparameters in __init__ and learning mean_, components_
    # (n_components x n_features loadings), noise_variance_ (per-feature psi), loglike_
    # and n_iter_ in fit. Provides latent posterior means (transform), the implied
    # covariance/precision of the data, and Gaussian log-likelihoods. Output feature
    # names are prefixed by the lower-cased class name
    # (ClassNamePrefixFeaturesOutMixin).
    """Factor Analysis (FA).

    A simple linear generative model with Gaussian latent variables.

    The observations are assumed to be caused by a linear transformation of
    lower dimensional latent factors and added Gaussian noise.
    Without loss of generality the factors are distributed according to a
    Gaussian with zero mean and unit covariance. The noise is also zero mean
    and has an arbitrary diagonal covariance matrix.

    If we would restrict the model further, by assuming that the Gaussian
    noise is even isotropic (all diagonal entries are the same) we would obtain
    :class:`PCA`.

    FactorAnalysis performs a maximum likelihood estimate of the so-called
    `loading` matrix, the transformation of the latent variables to the
    observed ones, using SVD based approach.

    Read more in the :ref:`User Guide <FA>`.

    .. versionadded:: 0.13

    Parameters
    ----------
    n_components : int, default=None
        Dimensionality of latent space, the number of components
        of ``X`` that are obtained after ``transform``.
        If None, n_components is set to the number of features.

    tol : float, default=1e-2
        Stopping tolerance for log-likelihood increase.

    copy : bool, default=True
        Whether to make a copy of X. If ``False``, the input X gets overwritten
        during fitting.

    max_iter : int, default=1000
        Maximum number of iterations.

    noise_variance_init : array-like of shape (n_features,), default=None
        The initial guess of the noise variance for each feature.
        If None, it defaults to np.ones(n_features).

    svd_method : {'lapack', 'randomized'}, default='randomized'
        Which SVD method to use. If 'lapack' use standard SVD from
        scipy.linalg, if 'randomized' use fast ``randomized_svd`` function.
        Defaults to 'randomized'. For most applications 'randomized' will
        be sufficiently precise while providing significant speed gains.
        Accuracy can also be improved by setting higher values for
        `iterated_power`. If this is not sufficient, for maximum precision
        you should choose 'lapack'.

    iterated_power : int, default=3
        Number of iterations for the power method. 3 by default. Only used
        if ``svd_method`` equals 'randomized'.

    rotation : {'varimax', 'quartimax'}, default=None
        If not None, apply the indicated rotation. Currently, varimax and
        quartimax are implemented. See
        `"The varimax criterion for analytic rotation in factor analysis"
        <https://link.springer.com/article/10.1007%2FBF02289233>`_
        H. F. Kaiser, 1958.

        .. versionadded:: 0.24

    random_state : int or RandomState instance, default=0
        Only used when ``svd_method`` equals 'randomized'. Pass an int for
        reproducible results across multiple function calls.
        See :term:`Glossary <random_state>`.

    Attributes
    ----------
    components_ : ndarray of shape (n_components, n_features)
        Components with maximum variance.

    loglike_ : list of shape (n_iterations,)
        The log likelihood at each iteration.

    noise_variance_ : ndarray of shape (n_features,)
        The estimated noise variance for each feature.

    n_iter_ : int
        Number of iterations run.

    mean_ : ndarray of shape (n_features,)
        Per-feature empirical mean, estimated from the training set.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    PCA: Principal component analysis is also a latent linear variable model
        which however assumes equal noise variance for each feature.
        This extra assumption makes probabilistic PCA faster as it can be
        computed in closed form.
    FastICA: Independent component analysis, a latent variable model with
        non-Gaussian latent variables.

    References
    ----------
    - David Barber, Bayesian Reasoning and Machine Learning,
      Algorithm 21.1.

    - Christopher M. Bishop: Pattern Recognition and Machine Learning,
      Chapter 12.2.4.

    Examples
    --------
    >>> from sklearn.datasets import load_digits
    >>> from sklearn.decomposition import FactorAnalysis
    >>> X, _ = load_digits(return_X_y=True)
    >>> transformer = FactorAnalysis(n_components=7, random_state=0)
    >>> X_transformed = transformer.fit_transform(X)
    >>> X_transformed.shape
    (1797, 7)
    """

    _parameter_constraints: dict = {
        "n_components": [Interval(Integral, 0, None, closed="left"), None],
        "tol": [Interval(Real, 0.0, None, closed="left")],
        "copy": ["boolean"],
        "max_iter": [Interval(Integral, 1, None, closed="left")],
        "noise_variance_init": ["array-like", None],
        "svd_method": [StrOptions({"randomized", "lapack"})],
        "iterated_power": [Interval(Integral, 0, None, closed="left")],
        "rotation": [StrOptions({"varimax", "quartimax"}), None],
        "random_state": ["random_state"],
    }

    def __init__(
        self,
        n_components=None,
        *,
        tol=1e-2,
        copy=True,
        max_iter=1000,
        noise_variance_init=None,
        svd_method="randomized",
        iterated_power=3,
        rotation=None,
        random_state=0,
    ):
        # [FUNCTION DOC]
        # Stores each constructor argument unchanged as an attribute of the same name;
        # no validation (done by _parameter_constraints at fit time).
        # Inputs: n_components (int or None, None means n_features), tol (float, log-
        #   likelihood increase threshold, default 1e-2), copy (bool), max_iter (int,
        #   default 1000), noise_variance_init (array of length n_features or None),
        #   svd_method ('randomized' or 'lapack'), iterated_power (int, randomized SVD
        #   power iterations), rotation (None, 'varimax' or 'quartimax'), random_state
        #   (seed/RandomState, default 0).
        # Outputs: none. Side effects: sets attributes on self.
        self.n_components = n_components
        self.copy = copy
        self.tol = tol
        self.max_iter = max_iter
        self.svd_method = svd_method

        self.noise_variance_init = noise_variance_init
        self.iterated_power = iterated_power
        self.random_state = random_state
        self.rotation = rotation

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None):
        # [FUNCTION DOC]
        # Fits the model. Steps: validate X as float64 (copied when copy is True);
        # center X by its column mean (stored as mean_). Initialise psi as ones or
        # noise_variance_init. Then for up to max_iter iterations: scale X by
        # 1/(sqrt(psi)+1e-12) and 1/sqrt(n_samples); take the top n_components singular
        # values s and right vectors Vt of this matrix along with the unexplained
        # squared norm; square s; set loadings W to sqrt(max(s-1, 0)) times each row of
        # Vt, then rescale columns by sqrt(psi); compute log-likelihood as -n_samples/2
        # times (n_features*log(2 pi) + n_components + sum log s + unexplained variance
        # + sum log psi); stop when the increase over the previous value is below tol;
        # otherwise update psi to max(per-feature variance minus column sum of W
        # squared, 1e-12).
        # Inputs: X array-like (n_samples, n_features); y ignored.
        # Outputs: self.
        # Exceptions: ValueError if noise_variance_init length differs from n_features;
        #   validation errors from validate_data.
        # Side effects: sets mean_, components_ (rotated with _rotate when rotation is
        #   set), noise_variance_, loglike_ (list of per-iteration log-likelihoods),
        #   n_iter_, n_features_in_ (and feature_names_in_); mutates X in place when
        #   copy is False; emits ConvergenceWarning when max_iter is exhausted; consumes
        #   random_state for the randomized SVD.
        # KNOWN ISSUE: the ValueError message reads "dimension does not with number of
        #   features" (missing word "match").
        # KNOWN ISSUE: when the loop breaks on convergence, components_ comes from the
        #   latest iteration but noise_variance_ is the psi used to compute it (not
        #   updated), which is consistent; however when max_iter is exhausted psi has
        #   been updated once more after W was computed, so noise_variance_ and
        #   components_ come from different iterations.
        """Fit the FactorAnalysis model to X using SVD based approach.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        y : Ignored
            Ignored parameter.

        Returns
        -------
        self : object
            FactorAnalysis class instance.
        """
        X = validate_data(
            self, X, copy=self.copy, dtype=np.float64, force_writeable=True
        )

        n_samples, n_features = X.shape
        n_components = self.n_components
        if n_components is None:
            n_components = n_features

        self.mean_ = np.mean(X, axis=0)
        X -= self.mean_

        # some constant terms
        nsqrt = sqrt(n_samples)
        llconst = n_features * log(2.0 * np.pi) + n_components
        var = np.var(X, axis=0)

        if self.noise_variance_init is None:
            psi = np.ones(n_features, dtype=X.dtype)
        else:
            if len(self.noise_variance_init) != n_features:
                raise ValueError(
                    "noise_variance_init dimension does not "
                    "with number of features : %d != %d"
                    % (len(self.noise_variance_init), n_features)
                )
            psi = np.array(self.noise_variance_init)

        loglike = []
        old_ll = -np.inf
        SMALL = 1e-12

        # we'll modify svd outputs to return unexplained variance
        # to allow for unified computation of loglikelihood
        if self.svd_method == "lapack":

            def my_svd(X):
                # [FUNCTION DOC]
                # HELPER: Exact SVD backend (svd_method 'lapack'). Computes a thin SVD
                #   of the given matrix with scipy.
                # Inputs: X - scaled 2D data matrix. Closes over n_components.
                # Outputs: tuple of (top n_components singular values, matching rows of
                #   Vt, sum of squares of the remaining singular values).
                _, s, Vt = linalg.svd(X, full_matrices=False, check_finite=False)
                return (
                    s[:n_components],
                    Vt[:n_components],
                    squared_norm(s[n_components:]),
                )

        else:  # svd_method == "randomized"
            random_state = check_random_state(self.random_state)

            def my_svd(X):
                # [FUNCTION DOC]
                # HELPER: Randomized SVD backend (svd_method 'randomized') using
                #   _randomized_svd with n_components, the shared random_state and
                #   iterated_power iterations.
                # Inputs: X - scaled 2D data matrix.
                # Outputs: tuple of (singular values, Vt rows, squared Frobenius norm of
                #   X minus the sum of squared singular values, i.e. the unexplained
                #   variance).
                # Side effects: advances the shared random_state.
                _, s, Vt = _randomized_svd(
                    X,
                    n_components,
                    random_state=random_state,
                    n_iter=self.iterated_power,
                )
                return s, Vt, squared_norm(X) - squared_norm(s)

        for i in range(self.max_iter):
            # SMALL helps numerics
            sqrt_psi = np.sqrt(psi) + SMALL
            s, Vt, unexp_var = my_svd(X / (sqrt_psi * nsqrt))
            s **= 2
            # Use 'maximum' here to avoid sqrt problems.
            W = np.sqrt(np.maximum(s - 1.0, 0.0))[:, np.newaxis] * Vt
            del Vt
            W *= sqrt_psi

            # loglikelihood
            ll = llconst + np.sum(np.log(s))
            ll += unexp_var + np.sum(np.log(psi))
            ll *= -n_samples / 2.0
            loglike.append(ll)
            if (ll - old_ll) < self.tol:
                break
            old_ll = ll

            psi = np.maximum(var - np.sum(W**2, axis=0), SMALL)
        else:
            warnings.warn(
                "FactorAnalysis did not converge."
                " You might want"
                " to increase the number of iterations.",
                ConvergenceWarning,
            )

        self.components_ = W
        if self.rotation is not None:
            self.components_ = self._rotate(W)
        self.noise_variance_ = psi
        self.loglike_ = loglike
        self.n_iter_ = i + 1
        return self

    def transform(self, X):
        # [FUNCTION DOC]
        # Returns the posterior mean of the latent variables: center X by mean_, form
        # Wpsi = components_ divided column-wise by noise_variance_, compute cov_z as
        # the inverse of (identity + Wpsi times components_ transposed), and return
        # centered X times Wpsi transposed times cov_z.
        # Inputs: X array-like (n_samples, n_features).
        # Outputs: ndarray (n_samples, n_components).
        # Exceptions: NotFittedError if unfitted; validation errors on feature mismatch.
        """Apply dimensionality reduction to X using the model.

        Compute the expected mean of the latent variables.
        See Barber, 21.2.33 (or Bishop, 12.66).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_components)
            The latent variables of X.
        """
        check_is_fitted(self)

        X = validate_data(self, X, reset=False)
        Ih = np.eye(len(self.components_))

        X_transformed = X - self.mean_

        Wpsi = self.components_ / self.noise_variance_
        cov_z = linalg.inv(Ih + np.dot(Wpsi, self.components_.T))
        tmp = np.dot(X_transformed, Wpsi.T)
        X_transformed = np.dot(tmp, cov_z)

        return X_transformed

    def get_covariance(self):
        # [FUNCTION DOC]
        # Returns components_ transposed times components_ with noise_variance_ added to
        # the diagonal, an ndarray (n_features, n_features).
        # Exceptions: NotFittedError if unfitted.
        """Compute data covariance with the FactorAnalysis model.

        ``cov = components_.T * components_ + diag(noise_variance)``

        Returns
        -------
        cov : ndarray of shape (n_features, n_features)
            Estimated covariance of data.
        """
        check_is_fitted(self)

        cov = np.dot(self.components_.T, self.components_)
        cov.flat[:: len(cov) + 1] += self.noise_variance_  # modify diag inplace
        return cov

    def get_precision(self):
        # [FUNCTION DOC]
        # Returns the inverse of get_covariance(), shape (n_features, n_features). If
        # n_components is 0 returns the diagonal of 1/noise_variance_; if n_components
        # equals n_features inverts the covariance directly; otherwise uses the Woodbury
        # identity: invert the small matrix (identity + W psi^-1 W^T), then compute
        # psi^-1 minus psi^-1 W^T (that inverse) W psi^-1.
        # Exceptions: NotFittedError if unfitted; LinAlgError if a matrix is singular.
        # KNOWN ISSUE: the corner cases test the hyperparameter n_components, not the
        #   fitted number of components; with n_components None (meaning n_features) the
        #   direct-inversion shortcut is skipped and the Woodbury path is used, which is
        #   still correct but slower.
        """Compute data precision matrix with the FactorAnalysis model.

        Returns
        -------
        precision : ndarray of shape (n_features, n_features)
            Estimated precision of data.
        """
        check_is_fitted(self)

        n_features = self.components_.shape[1]

        # handle corner cases first
        if self.n_components == 0:
            return np.diag(1.0 / self.noise_variance_)
        if self.n_components == n_features:
            return linalg.inv(self.get_covariance())

        # Get precision using matrix inversion lemma
        components_ = self.components_
        precision = np.dot(components_ / self.noise_variance_, components_.T)
        precision.flat[:: len(precision) + 1] += 1.0
        precision = np.dot(components_.T, np.dot(linalg.inv(precision), components_))
        precision /= self.noise_variance_[:, np.newaxis]
        precision /= -self.noise_variance_[np.newaxis, :]
        precision.flat[:: len(precision) + 1] += 1.0 / self.noise_variance_
        return precision

    def score_samples(self, X):
        # [FUNCTION DOC]
        # Computes the Gaussian log-density of each sample under N(mean_,
        # get_covariance()): minus half the Mahalanobis form with the precision matrix,
        # minus half of (n_features*log(2 pi) minus log-determinant of the precision).
        # Inputs: X array (n_samples, n_features).
        # Outputs: ndarray (n_samples,).
        # Exceptions: NotFittedError; validation errors.
        """Compute the log-likelihood of each sample.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The data.

        Returns
        -------
        ll : ndarray of shape (n_samples,)
            Log-likelihood of each sample under the current model.
        """
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)
        Xr = X - self.mean_
        precision = self.get_precision()
        n_features = X.shape[1]
        log_like = -0.5 * (Xr * (np.dot(Xr, precision))).sum(axis=1)
        log_like -= 0.5 * (n_features * log(2.0 * np.pi) - fast_logdet(precision))
        return log_like

    def score(self, X, y=None):
        # [FUNCTION DOC]
        # Returns the mean of score_samples(X) as a float. y is ignored.
        """Compute the average log-likelihood of the samples.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The data.

        y : Ignored
            Ignored parameter.

        Returns
        -------
        ll : float
            Average log-likelihood of the samples under the current model.
        """
        return np.mean(self.score_samples(X))

    def _rotate(self, components, n_components=None, tol=1e-6):
        # [FUNCTION DOC]
        # HELPER: Applies the configured rotation (self.rotation) to the loadings via
        #   _ortho_rotation on the transposed components, returning the first
        #   self.n_components rows of the rotated (n_components, n_features) result.
        # Inputs: components (n_components, n_features); n_components unused; tol
        #   convergence tolerance.
        # KNOWN ISSUE: the n_components argument is ignored; slicing uses
        #   self.n_components instead.
        "Rotate the factor analysis solution."
        # note that tol is not exposed
        return _ortho_rotation(components.T, method=self.rotation, tol=tol)[
            : self.n_components
        ]

    @property
    def _n_features_out(self):
        # [FUNCTION DOC]
        # HELPER: Property giving the number of output features, i.e. the number of rows
        #   of components_ (used for get_feature_names_out).
        """Number of transformed output features."""
        return self.components_.shape[0]


def _ortho_rotation(components, method="varimax", tol=1e-6, max_iter=100):
    # [FUNCTION DOC]
    # HELPER: Iteratively finds an orthogonal rotation maximising the varimax or
    #   quartimax criterion. Start from the identity rotation; each iteration rotates
    #   the loadings, builds the gradient target (for varimax, the cube of rotated
    #   loadings minus rotated loadings scaled by their column mean square; for
    #   quartimax, just the cube), takes the SVD of components transposed times that
    #   target, and sets the rotation to U times V. Stops when the sum of singular
    #   values improves by less than relative tol, or after max_iter iterations.
    # Inputs: components (n_features, n_components) loading matrix; method 'varimax' or
    #   'quartimax'; tol float; max_iter int.
    # Outputs: rotated loadings transposed, shape (n_components, n_features).
    # KNOWN ISSUE: any other method string leaves tmp unset on the first iteration and
    #   raises UnboundLocalError (guarded in practice by parameter validation).
    """Return rotated components."""
    nrow, ncol = components.shape
    rotation_matrix = np.eye(ncol)
    var = 0

    for _ in range(max_iter):
        comp_rot = np.dot(components, rotation_matrix)
        if method == "varimax":
            tmp = comp_rot * np.transpose((comp_rot**2).sum(axis=0) / nrow)
        elif method == "quartimax":
            tmp = 0
        u, s, v = np.linalg.svd(np.dot(components.T, comp_rot**3 - tmp))
        rotation_matrix = np.dot(u, v)
        var_new = np.sum(s)
        if var != 0 and var_new < var * (1 + tol):
            break
        var = var_new

    return np.dot(components, rotation_matrix).T
