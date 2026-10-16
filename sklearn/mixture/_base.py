# [MODULE DOC]
# Abstract base for mixture-model estimators. Defines the shared EM / variational
# driver loop (multiple initializations, E-step / M-step alternation, convergence
# check on the lower bound), initialization strategies for responsibilities,
# scoring, prediction and sampling. Subclasses supply parameter checks,
# initialization from responsibilities, the M-step, log-probabilities, log-weights,
# lower-bound computation and parameter get/set. Supports array API namespaces.
"""Base class for mixture models."""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import warnings
from abc import ABCMeta, abstractmethod
from contextlib import nullcontext
from numbers import Integral, Real
from time import time

import numpy as np

from sklearn import cluster
from sklearn.base import BaseEstimator, DensityMixin, _fit_context
from sklearn.cluster import kmeans_plusplus
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils import check_random_state
from sklearn.utils._array_api import (
    _is_numpy_namespace,
    _logsumexp,
    _max_precision_float_dtype,
    get_namespace,
    get_namespace_and_device,
    move_to,
)
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.validation import check_is_fitted, validate_data


def _check_shape(param, param_shape, name):
    # [FUNCTION DOC]
    # HELPER: Ensures an array has exactly the expected shape.
    # Inputs: param - array; param_shape - expected shape tuple; name - parameter name
    # used in the error message.
    # Outputs: none.
    # Exceptions: ValueError if param.shape differs from param_shape.
    """Validate the shape of the input parameter 'param'.

    Parameters
    ----------
    param : array

    param_shape : tuple

    name : str
    """
    if param.shape != param_shape:
        raise ValueError(
            "The parameter '%s' should have the shape of %s, but got %s"
            % (name, param_shape, param.shape)
        )


class BaseMixture(DensityMixin, BaseEstimator, metaclass=ABCMeta):
    # [CLASS DOC]
    # Abstract mixture-model estimator (density estimator). Holds the common
    # hyperparameters (validated via _parameter_constraints) and implements fit,
    # fit_predict, score_samples, score, predict, predict_proba and sample on top of
    # abstract hooks implemented by GaussianMixture / BayesianGaussianMixture.
    # Subclasses must also provide _compute_lower_bound, means_, weights_,
    # covariances_ and covariance_type (used by fit_predict and sample).
    """Base class for mixture models.

    This abstract class specifies an interface for all mixture classes and
    provides basic common methods for mixture models.
    """

    _parameter_constraints: dict = {
        "n_components": [Interval(Integral, 1, None, closed="left")],
        "tol": [Interval(Real, 0.0, None, closed="left")],
        "reg_covar": [Interval(Real, 0.0, None, closed="left")],
        "max_iter": [Interval(Integral, 0, None, closed="left")],
        "n_init": [Interval(Integral, 1, None, closed="left")],
        "init_params": [
            StrOptions({"kmeans", "random", "random_from_data", "k-means++"})
        ],
        "random_state": ["random_state"],
        "warm_start": ["boolean"],
        "verbose": ["verbose"],
        "verbose_interval": [Interval(Integral, 1, None, closed="left")],
    }

    def __init__(
        self,
        n_components,
        tol,
        reg_covar,
        max_iter,
        n_init,
        init_params,
        random_state,
        warm_start,
        verbose,
        verbose_interval,
    ):
        # [FUNCTION DOC]
        # Stores hyperparameters unchanged as attributes (no validation, sklearn API).
        # Inputs: n_components (int >= 1), tol (float >= 0, convergence threshold on
        #   lower
        # bound change), reg_covar (float >= 0), max_iter (int >= 0), n_init (int >= 1),
        # init_params (one of kmeans, random, random_from_data, k-means++),
        # random_state, warm_start (bool), verbose (int), verbose_interval (int >= 1).
        # Side effects: sets the same-named attributes on self.
        self.n_components = n_components
        self.tol = tol
        self.reg_covar = reg_covar
        self.max_iter = max_iter
        self.n_init = n_init
        self.init_params = init_params
        self.random_state = random_state
        self.warm_start = warm_start
        self.verbose = verbose
        self.verbose_interval = verbose_interval

    @abstractmethod
    def _check_parameters(self, X, xp=None):
        # [FUNCTION DOC]
        # Abstract hook: validates subclass-specific parameters against data X.
        # Inputs: X - validated data (n_samples, n_features); xp - array namespace or
        #   None.
        # Outputs: none. Exceptions: defined by subclasses.
        """Check initial parameters of the derived class.

        Parameters
        ----------
        X : array-like of shape  (n_samples, n_features)
        """
        pass

    def _initialize_parameters(self, X, random_state, xp=None):
        # [FUNCTION DOC]
        # Builds an initial responsibility matrix (n_samples, n_components) according to
        # init_params, then delegates to _initialize(X, resp).
        # - kmeans: run KMeans (n_init=1) and one-hot encode the cluster labels (numpy).
        # - random: uniform random values, normalized so each row sums to 1.
        # - random_from_data: pick n_components distinct sample indices; each chosen
        # sample gets responsibility 1 for its own component, all others 0.
        # - k-means++: k-means++ seeding indices, one-hot as above (numpy).
        # Inputs: X - data; random_state - RandomState instance; xp - array namespace.
        # Outputs: none.
        # Side effects: consumes random_state; subclass _initialize sets model params.
        # KNOWN ISSUE: kmeans and k-means++ branches build resp with numpy regardless of
        # xp, so for non-numpy namespaces resp is not in the input namespace/device.
        # KNOWN ISSUE: with random_from_data / k-means++ rows not selected have all-zero
        # responsibilities (intended, but rows do not sum to 1).
        """Initialize the model parameters.

        Parameters
        ----------
        X : array-like of shape  (n_samples, n_features)

        random_state : RandomState
            A random number generator instance that controls the random seed
            used for the method chosen to initialize the parameters.
        """
        xp, _, device = get_namespace_and_device(X, xp=xp)
        n_samples, _ = X.shape

        if self.init_params == "kmeans":
            resp = np.zeros((n_samples, self.n_components), dtype=X.dtype)
            label = (
                cluster.KMeans(
                    n_clusters=self.n_components, n_init=1, random_state=random_state
                )
                .fit(X)
                .labels_
            )
            resp[np.arange(n_samples), label] = 1
        elif self.init_params == "random":
            resp = xp.asarray(
                random_state.uniform(size=(n_samples, self.n_components)),
                dtype=X.dtype,
                device=device,
            )
            resp /= xp.sum(resp, axis=1)[:, xp.newaxis]
        elif self.init_params == "random_from_data":
            resp = xp.zeros(
                (n_samples, self.n_components), dtype=X.dtype, device=device
            )
            indices = random_state.choice(
                n_samples, size=self.n_components, replace=False
            )
            # TODO: when array API supports __setitem__ with fancy indexing we
            # can use the previous code:
            # resp[indices, xp.arange(self.n_components)] = 1
            # Until then we use a for loop on one dimension.
            for col, index in enumerate(indices):
                resp[index, col] = 1
        elif self.init_params == "k-means++":
            resp = np.zeros((n_samples, self.n_components), dtype=X.dtype)
            _, indices = kmeans_plusplus(
                X,
                self.n_components,
                random_state=random_state,
            )
            resp[indices, np.arange(self.n_components)] = 1

        self._initialize(X, resp)

    @abstractmethod
    def _initialize(self, X, resp):
        # [FUNCTION DOC]
        # Abstract hook: initializes model parameters from data X and responsibilities
        # resp of shape (n_samples, n_components).
        """Initialize the model parameters of the derived class.

        Parameters
        ----------
        X : array-like of shape  (n_samples, n_features)

        resp : array-like of shape (n_samples, n_components)
        """
        pass

    def fit(self, X, y=None):
        # [FUNCTION DOC]
        # Fits the model by calling fit_predict and discarding the labels.
        # Inputs: X - array-like (n_samples, n_features); y - ignored.
        # Outputs: self.
        # Side effects/Exceptions: same as fit_predict.
        """Estimate model parameters with the EM algorithm.

        The method fits the model ``n_init`` times and sets the parameters with
        which the model has the largest likelihood or lower bound. Within each
        trial, the method iterates between E-step and M-step for ``max_iter``
        times until the change of likelihood or lower bound is less than
        ``tol``, otherwise, a ``ConvergenceWarning`` is raised.
        If ``warm_start`` is ``True``, then ``n_init`` is ignored and a single
        initialization is performed upon the first call. Upon consecutive
        calls, training starts where it left off.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : object
            The fitted mixture.
        """
        # parameters are validated in fit_predict
        self.fit_predict(X, y)
        return self

    @_fit_context(prefer_skip_nested_validation=True)
    def fit_predict(self, X, y=None):
        # [FUNCTION DOC]
        # Main fitting driver. Steps:
        # 1. Validate X (float64/float32, at least 2 samples); require n_samples >=
        # n_components; call _check_parameters.
        # 2. do_init is False only when warm_start is set and the model is already
        #   fitted
        # (converged_ exists); then a single run starting from current params and
        # lower_bound_, otherwise n_init runs each re-initialized.
        # 3. For each run: if max_iter is 0, take current params as best (n_iter 0).
        # Otherwise iterate up to max_iter times: E-step, M-step, compute lower bound,
        # record it, stop when the absolute change is below tol. Keep the run with
        # highest final lower bound (or the first one) as best params, n_iter,
        # lower-bound history and convergence flag.
        # 4. Warn if best run did not converge and max_iter > 0.
        # 5. Set best params and fitted attributes, run a final E-step, return the
        #   argmax
        # of log responsibilities.
        # Inputs: X - array-like (n_samples, n_features); y - ignored.
        # Outputs: integer label array (n_samples,).
        # Exceptions: ValueError if n_samples < n_components or invalid data/params.
        # Side effects: sets converged_, n_iter_, lower_bound_, lower_bounds_, model
        # params; emits ConvergenceWarning; prints when verbose; consumes randomness.
        # KNOWN ISSUE: when max_iter is 0, max_lower_bound stays -inf so lower_bound_ is
        # -inf and best_params is overwritten by each init (the last one wins).
        """Estimate model parameters using X and predict the labels for X.

        The method fits the model ``n_init`` times and sets the parameters with
        which the model has the largest likelihood or lower bound. Within each
        trial, the method iterates between E-step and M-step for `max_iter`
        times until the change of likelihood or lower bound is less than
        `tol`, otherwise, a :class:`~sklearn.exceptions.ConvergenceWarning` is
        raised. After fitting, it predicts the most probable label for the
        input data points.

        .. versionadded:: 0.20

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        labels : array, shape (n_samples,)
            Component labels.
        """
        xp, _ = get_namespace(X)
        X = validate_data(self, X, dtype=[xp.float64, xp.float32], ensure_min_samples=2)
        if X.shape[0] < self.n_components:
            raise ValueError(
                "Expected n_samples >= n_components "
                f"but got n_components = {self.n_components}, "
                f"n_samples = {X.shape[0]}"
            )
        self._check_parameters(X, xp=xp)

        # if we enable warm_start, we will have a unique initialisation
        do_init = not (self.warm_start and hasattr(self, "converged_"))
        n_init = self.n_init if do_init else 1

        max_lower_bound = -xp.inf
        best_lower_bounds = []
        self.converged_ = False

        random_state = check_random_state(self.random_state)

        n_samples, _ = X.shape
        for init in range(n_init):
            self._print_verbose_msg_init_beg(init)

            if do_init:
                self._initialize_parameters(X, random_state, xp=xp)

            lower_bound = -xp.inf if do_init else self.lower_bound_
            current_lower_bounds = []

            if self.max_iter == 0:
                best_params = self._get_parameters()
                best_n_iter = 0
            else:
                converged = False
                for n_iter in range(1, self.max_iter + 1):
                    prev_lower_bound = lower_bound

                    log_prob_norm, log_resp = self._e_step(X, xp=xp)
                    self._m_step(X, log_resp, xp=xp)
                    lower_bound = self._compute_lower_bound(log_resp, log_prob_norm)
                    current_lower_bounds.append(lower_bound)

                    change = lower_bound - prev_lower_bound
                    self._print_verbose_msg_iter_end(n_iter, change)

                    if abs(change) < self.tol:
                        converged = True
                        break

                self._print_verbose_msg_init_end(lower_bound, converged)

                if lower_bound > max_lower_bound or max_lower_bound == -xp.inf:
                    max_lower_bound = lower_bound
                    best_params = self._get_parameters()
                    best_n_iter = n_iter
                    best_lower_bounds = current_lower_bounds
                    self.converged_ = converged

        # Should only warn about convergence if max_iter > 0, otherwise
        # the user is assumed to have used 0-iters initialization
        # to get the initial means.
        if not self.converged_ and self.max_iter > 0:
            warnings.warn(
                (
                    "Best performing initialization did not converge. "
                    "Try different init parameters, or increase max_iter, "
                    "tol, or check for degenerate data."
                ),
                ConvergenceWarning,
            )

        self._set_parameters(best_params, xp=xp)
        self.n_iter_ = best_n_iter
        self.lower_bound_ = max_lower_bound
        self.lower_bounds_ = best_lower_bounds

        # Always do a final e-step to guarantee that the labels returned by
        # fit_predict(X) are always consistent with fit(X).predict(X)
        # for any value of max_iter and tol (and any random_state).
        _, log_resp = self._e_step(X, xp=xp)

        return xp.argmax(log_resp, axis=1)

    def _e_step(self, X, xp=None):
        # [FUNCTION DOC]
        # E-step: computes per-sample log normalizers and log responsibilities.
        # Inputs: X - (n_samples, n_features); xp - namespace.
        # Outputs: tuple (mean of per-sample log p(x) as 0-d array/scalar, log_resp of
        # shape (n_samples, n_components)).
        """E step.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        log_prob_norm : float
            Mean of the logarithms of the probabilities of each sample in X

        log_responsibility : array, shape (n_samples, n_components)
            Logarithm of the posterior probabilities (or responsibilities) of
            the point of each sample in X.
        """
        xp, _ = get_namespace(X, xp=xp)
        log_prob_norm, log_resp = self._estimate_log_prob_resp(X, xp=xp)
        return xp.mean(log_prob_norm), log_resp

    @abstractmethod
    def _m_step(self, X, log_resp):
        # [FUNCTION DOC]
        # Abstract hook: updates model parameters from X and log responsibilities.
        # Note: called with an xp keyword by fit_predict although the abstract signature
        # lacks it; subclasses must accept xp.
        """M step.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        log_resp : array-like of shape (n_samples, n_components)
            Logarithm of the posterior probabilities (or responsibilities) of
            the point of each sample in X.
        """
        pass

    @abstractmethod
    def _get_parameters(self):
        # [FUNCTION DOC]
        # Abstract hook: returns a tuple snapshot of the current model parameters.
        pass

    @abstractmethod
    def _set_parameters(self, params):
        # [FUNCTION DOC]
        # Abstract hook: restores model parameters from a snapshot from _get_parameters.
        # Note: called with an xp keyword by fit_predict; subclasses must accept it.
        pass

    def score_samples(self, X):
        # [FUNCTION DOC]
        # Per-sample log-likelihood: log-sum-exp over components of weighted log-probs.
        # Inputs: X - array-like (n_samples, n_features).
        # Outputs: array (n_samples,).
        # Exceptions: NotFittedError if not fitted; ValueError on feature mismatch.
        """Compute the log-likelihood of each sample.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.

        Returns
        -------
        log_prob : array, shape (n_samples,)
            Log-likelihood of each sample in `X` under the current model.
        """
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)

        return _logsumexp(self._estimate_weighted_log_prob(X), axis=1)

    def score(self, X, y=None):
        # [FUNCTION DOC]
        # Average log-likelihood of X under the model (mean of score_samples).
        # Inputs: X - array-like (n_samples, n_features); y - ignored.
        # Outputs: Python float.
        """Compute the per-sample average log-likelihood of the given data X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_dimensions)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        log_likelihood : float
            Log-likelihood of `X` under the Gaussian mixture model.
        """
        xp, _ = get_namespace(X)
        return float(xp.mean(self.score_samples(X)))

    def predict(self, X):
        # [FUNCTION DOC]
        # Hard component assignment: argmax over components of weighted log-probs.
        # Inputs: X - array-like (n_samples, n_features).
        # Outputs: integer label array (n_samples,).
        # Exceptions: NotFittedError if not fitted.
        """Predict the labels for the data samples in X using trained model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.

        Returns
        -------
        labels : array, shape (n_samples,)
            Component labels.
        """
        check_is_fitted(self)
        xp, _ = get_namespace(X)
        X = validate_data(self, X, reset=False)
        return xp.argmax(self._estimate_weighted_log_prob(X), axis=1)

    def predict_proba(self, X):
        # [FUNCTION DOC]
        # Posterior component probabilities (exponentiated log responsibilities).
        # Inputs: X - array-like (n_samples, n_features).
        # Outputs: array (n_samples, n_components), rows sum to 1.
        # Exceptions: NotFittedError if not fitted.
        """Evaluate the components' density for each sample.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.

        Returns
        -------
        resp : array, shape (n_samples, n_components)
            Density of each Gaussian component for each sample in X.
        """
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)
        xp, _ = get_namespace(X)
        _, log_resp = self._estimate_log_prob_resp(X, xp=xp)
        return xp.exp(log_resp)

    def sample(self, n_samples=1):
        # [FUNCTION DOC]
        # Draws samples from the fitted mixture. Component counts are drawn from a
        # multinomial with the fitted weights; then for each component samples are drawn
        # from a multivariate normal with that mean and covariance (full: per-component
        # matrix; tied: shared matrix; diag/spherical: mean plus standard normal noise
        # scaled by sqrt of the variances). Sampling is done in numpy on CPU, then
        # converted to the namespace/device of means_.
        # Inputs: n_samples - int, default 1.
        # Outputs: (X of shape (n_samples, n_features) in the max-precision float dtype,
        # y int64 component labels (n_samples,), grouped by component in order).
        # Exceptions: NotFittedError; ValueError if n_samples < 1.
        # Side effects: uses a RandomState built from self.random_state (deterministic
        #   for
        # an int seed; repeated calls give the same samples).
        # KNOWN ISSUE: the error message formats self.n_components instead of n_samples.
        """Generate random samples from the fitted Gaussian distribution.

        Parameters
        ----------
        n_samples : int, default=1
            Number of samples to generate.

        Returns
        -------
        X : array, shape (n_samples, n_features)
            Randomly generated sample.

        y : array, shape (nsamples,)
            Component labels.
        """
        check_is_fitted(self)
        xp, _, device_ = get_namespace_and_device(self.means_)

        if n_samples < 1:
            raise ValueError(
                "Invalid value for 'n_samples': %d . The sampling requires at "
                "least one sample." % (self.n_components)
            )

        _, n_features = self.means_.shape
        rng = check_random_state(self.random_state)
        n_samples_comp = rng.multinomial(
            n_samples, move_to(self.weights_, xp=np, device="cpu")
        )

        if self.covariance_type == "full":
            X = np.vstack(
                [
                    rng.multivariate_normal(mean, covariance, int(sample))
                    for (mean, covariance, sample) in zip(
                        move_to(self.means_, xp=np, device="cpu"),
                        move_to(self.covariances_, xp=np, device="cpu"),
                        n_samples_comp,
                    )
                ]
            )
        elif self.covariance_type == "tied":
            X = np.vstack(
                [
                    rng.multivariate_normal(
                        mean,
                        move_to(self.covariances_, xp=np, device="cpu"),
                        int(sample),
                    )
                    for (mean, sample) in zip(
                        move_to(self.means_, xp=np, device="cpu"), n_samples_comp
                    )
                ]
            )
        else:
            X = np.vstack(
                [
                    mean
                    + rng.standard_normal(size=(sample, n_features))
                    * np.sqrt(covariance)
                    for (mean, covariance, sample) in zip(
                        move_to(self.means_, xp=np, device="cpu"),
                        move_to(self.covariances_, xp=np, device="cpu"),
                        n_samples_comp,
                    )
                ]
            )

        y = xp.concat(
            [
                xp.full(int(n_samples_comp[i]), i, dtype=xp.int64, device=device_)
                for i in range(len(n_samples_comp))
            ]
        )

        max_float_dtype = _max_precision_float_dtype(xp=xp, device=device_)
        return xp.asarray(X, dtype=max_float_dtype, device=device_), y

    def _estimate_weighted_log_prob(self, X, xp=None):
        # [FUNCTION DOC]
        # Returns log P(X | Z) plus log weights, shape (n_samples, n_components).
        # Inputs: X - data; xp - namespace.
        """Estimate the weighted log-probabilities, log P(X | Z) + log weights.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        weighted_log_prob : array, shape (n_samples, n_component)
        """
        return self._estimate_log_prob(X, xp=xp) + self._estimate_log_weights(xp=xp)

    @abstractmethod
    def _estimate_log_weights(self, xp=None):
        # [FUNCTION DOC]
        # Abstract hook: log mixing weights (EM) or expected log weights (VB), shape
        # (n_components,).
        """Estimate log-weights in EM algorithm, E[ log pi ] in VB algorithm.

        Returns
        -------
        log_weight : array, shape (n_components, )
        """
        pass

    @abstractmethod
    def _estimate_log_prob(self, X, xp=None):
        # [FUNCTION DOC]
        # Abstract hook: per-sample per-component log-likelihood log P(X | Z), shape
        # (n_samples, n_components).
        """Estimate the log-probabilities log P(X | Z).

        Compute the log-probabilities per each component for each sample.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        log_prob : array, shape (n_samples, n_component)
        """
        pass

    def _estimate_log_prob_resp(self, X, xp=None):
        # [FUNCTION DOC]
        # Computes per-sample log normalizer (log-sum-exp of weighted log-probs over
        # components) and log responsibilities (weighted log-probs minus the
        #   normalizer),
        # suppressing numpy underflow warnings when using numpy.
        # Inputs: X - data; xp - namespace.
        # Outputs: (log_prob_norm (n_samples,), log_resp (n_samples, n_components)).
        """Estimate log probabilities and responsibilities for each sample.

        Compute the log probabilities, weighted log probabilities per
        component and responsibilities for each sample in X with respect to
        the current state of the model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        log_prob_norm : array, shape (n_samples,)
            log p(X)

        log_responsibilities : array, shape (n_samples, n_components)
            logarithm of the responsibilities
        """
        xp, _ = get_namespace(X, xp=xp)
        weighted_log_prob = self._estimate_weighted_log_prob(X, xp=xp)
        log_prob_norm = _logsumexp(weighted_log_prob, axis=1, xp=xp)

        # There is no errstate equivalent for warning/error management in array API
        context_manager = (
            np.errstate(under="ignore") if _is_numpy_namespace(xp) else nullcontext()
        )
        with context_manager:
            # ignore underflow
            log_resp = weighted_log_prob - log_prob_norm[:, xp.newaxis]
        return log_prob_norm, log_resp

    def _print_verbose_msg_init_beg(self, n_init):
        # [FUNCTION DOC]
        # HELPER: Prints the initialization index when verbose >= 1; for verbose >= 2
        # also records start timestamps.
        # Inputs: n_init - index of the current initialization.
        # Side effects: stdout; sets _init_prev_time and _iter_prev_time.
        """Print verbose message on initialization."""
        if self.verbose == 1:
            print("Initialization %d" % n_init)
        elif self.verbose >= 2:
            print("Initialization %d" % n_init)
            self._init_prev_time = time()
            self._iter_prev_time = self._init_prev_time

    def _print_verbose_msg_iter_end(self, n_iter, diff_ll):
        # [FUNCTION DOC]
        # HELPER: Every verbose_interval iterations prints the iteration number; at
        # verbose >= 2 also elapsed time since last print and lower-bound change.
        # Inputs: n_iter - iteration number; diff_ll - lower bound change.
        # Side effects: stdout; updates _iter_prev_time.
        # KNOWN ISSUE: docstring says "on initialization" but it reports iteration end.
        """Print verbose message on initialization."""
        if n_iter % self.verbose_interval == 0:
            if self.verbose == 1:
                print("  Iteration %d" % n_iter)
            elif self.verbose >= 2:
                cur_time = time()
                print(
                    "  Iteration %d\t time lapse %.5fs\t ll change %.5f"
                    % (n_iter, cur_time - self._iter_prev_time, diff_ll)
                )
                self._iter_prev_time = cur_time

    def _print_verbose_msg_init_end(self, lb, init_has_converged):
        # [FUNCTION DOC]
        # HELPER: Prints whether the initialization converged; at verbose >= 2 also
        # elapsed time of the run and final lower bound.
        # Inputs: lb - final lower bound; init_has_converged - bool.
        # Side effects: stdout.
        """Print verbose message on the end of iteration."""
        converged_msg = "converged" if init_has_converged else "did not converge"
        if self.verbose == 1:
            print(f"Initialization {converged_msg}.")
        elif self.verbose >= 2:
            t = time() - self._init_prev_time
            print(
                f"Initialization {converged_msg}. time lapse {t:.5f}s\t lower bound"
                f" {lb:.5f}."
            )
