# [MODULE DOC]
# Maximum-likelihood (empirical) covariance estimation: the log_likelihood and
# empirical_covariance functions and the EmpiricalCovariance estimator, which is
# the base class for the other covariance estimators in this package (they reuse
# _set_covariance, get_precision, score, error_norm and mahalanobis).
# KNOWN ISSUE: the "# avoid division truncation" comment near the imports is a
# stale leftover (it used to precede a __future__ division import).
"""
Maximum likelihood covariance estimator.

"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

# avoid division truncation
import warnings

import numpy as np
from scipy import linalg

from sklearn import config_context
from sklearn.base import BaseEstimator, _fit_context
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.utils import check_array, metadata_routing
from sklearn.utils._param_validation import validate_params
from sklearn.utils.extmath import fast_logdet
from sklearn.utils.validation import validate_data


@validate_params(
    {
        "emp_cov": [np.ndarray],
        "precision": [np.ndarray],
    },
    prefer_skip_nested_validation=True,
)
def log_likelihood(emp_cov, precision):
    # [FUNCTION DOC]
    # Gaussian average log-likelihood of data summarised by emp_cov under a model with
    # the given precision: computes minus the sum of the elementwise product of
    # emp_cov and precision (i.e. minus the trace of their product), adds the log
    # determinant of precision (fast_logdet, -inf if not positive definite), subtracts
    # p * log(2*pi) with p = number of features, and halves the result.
    # Inputs: emp_cov - ndarray (p, p); precision - ndarray (p, p).
    # Outputs: float.
    # Exceptions: InvalidParameterError from validate_params if not ndarrays.
    """Compute the sample mean of the log_likelihood under a covariance model.

    Computes the empirical expected log-likelihood, allowing for universal
    comparison (beyond this software package), and accounts for normalization
    terms and scaling.

    Parameters
    ----------
    emp_cov : ndarray of shape (n_features, n_features)
        Maximum Likelihood Estimator of covariance.

    precision : ndarray of shape (n_features, n_features)
        The precision matrix of the covariance model to be tested.

    Returns
    -------
    log_likelihood_ : float
        Sample mean of the log-likelihood.
    """
    p = precision.shape[0]
    log_likelihood_ = -np.sum(emp_cov * precision) + fast_logdet(precision)
    log_likelihood_ -= p * np.log(2 * np.pi)
    log_likelihood_ /= 2.0
    return log_likelihood_


@validate_params(
    {
        "X": ["array-like"],
        "assume_centered": ["boolean"],
    },
    prefer_skip_nested_validation=True,
)
def empirical_covariance(X, *, assume_centered=False):
    # [FUNCTION DOC]
    # Computes the MLE covariance (normalised by n_samples). X is validated with
    # check_array (2D not required, non-finite allowed); a 1D input becomes one row.
    # If assume_centered, returns X^T X / n_samples; otherwise numpy covariance of the
    # columns with bias (divide by n). A 0-d result (single feature) is wrapped to
    # shape (1, 1).
    # Inputs: X - array-like (n_samples, n_features) or 1D; assume_centered - bool.
    # Outputs: ndarray (n_features, n_features).
    # Side effects: emits UserWarning when only one sample is available.
    """Compute the Maximum likelihood covariance estimator.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Data from which to compute the covariance estimate.

    assume_centered : bool, default=False
        If `True`, data will not be centered before computation.
        Useful when working with data whose mean is almost, but not exactly
        zero.
        If `False`, data will be centered before computation.

    Returns
    -------
    covariance : ndarray of shape (n_features, n_features)
        Empirical covariance (Maximum Likelihood Estimator).

    Examples
    --------
    >>> from sklearn.covariance import empirical_covariance
    >>> X = [[1,1,1],[1,1,1],[1,1,1],
    ...      [0,0,0],[0,0,0],[0,0,0]]
    >>> empirical_covariance(X)
    array([[0.25, 0.25, 0.25],
           [0.25, 0.25, 0.25],
           [0.25, 0.25, 0.25]])
    """
    X = check_array(X, ensure_2d=False, ensure_all_finite=False)

    if X.ndim == 1:
        X = np.reshape(X, (1, -1))

    if X.shape[0] == 1:
        warnings.warn(
            "Only one sample available. You may want to reshape your data array"
        )

    if assume_centered:
        covariance = np.dot(X.T, X) / X.shape[0]
    else:
        covariance = np.cov(X.T, bias=1)

    if covariance.ndim == 0:
        covariance = np.array([[covariance]])
    return covariance


class EmpiricalCovariance(BaseEstimator):
    # [CLASS DOC]
    # Maximum-likelihood covariance estimator. fit stores location_ (mean or zeros),
    # covariance_ and optionally precision_ (pseudo-inverse). Class attributes:
    # __metadata_request__score marks score's X_test as unused for metadata routing;
    # _parameter_constraints validates store_precision and assume_centered as bools.
    """Maximum likelihood covariance estimator.

    Read more in the :ref:`User Guide <covariance>`.

    Parameters
    ----------
    store_precision : bool, default=True
        Specifies if the estimated precision is stored.

    assume_centered : bool, default=False
        If True, data are not centered before computation.
        Useful when working with data whose mean is almost, but not exactly
        zero.
        If False (default), data are centered before computation.

    Attributes
    ----------
    location_ : ndarray of shape (n_features,)
        Estimated location, i.e. the estimated mean.

    covariance_ : ndarray of shape (n_features, n_features)
        Estimated covariance matrix.

    precision_ : ndarray of shape (n_features, n_features)
        Estimated pseudo-inverse matrix.
        (stored only if store_precision is True)

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    EllipticEnvelope : An object for detecting outliers in
        a Gaussian distributed dataset.
    GraphicalLasso : Sparse inverse covariance estimation
        with an l1-penalized estimator.
    LedoitWolf : LedoitWolf Estimator.
    MinCovDet : Minimum Covariance Determinant
        (robust estimator of covariance).
    OAS : Oracle Approximating Shrinkage Estimator.
    ShrunkCovariance : Covariance estimator with shrinkage.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.covariance import EmpiricalCovariance
    >>> from sklearn.datasets import make_gaussian_quantiles
    >>> real_cov = np.array([[.8, .3],
    ...                      [.3, .4]])
    >>> rng = np.random.RandomState(0)
    >>> X = rng.multivariate_normal(mean=[0, 0],
    ...                             cov=real_cov,
    ...                             size=500)
    >>> cov = EmpiricalCovariance().fit(X)
    >>> cov.covariance_
    array([[0.7569, 0.2818],
           [0.2818, 0.3928]])
    >>> cov.location_
    array([0.0622, 0.0193])
    """

    # X_test should have been called X
    __metadata_request__score = {"X_test": metadata_routing.UNUSED}

    _parameter_constraints: dict = {
        "store_precision": ["boolean"],
        "assume_centered": ["boolean"],
    }

    def __init__(self, *, store_precision=True, assume_centered=False):
        # [FUNCTION DOC]
        # Stores the hyperparameters store_precision (bool, default True) and
        # assume_centered (bool, default False) on self; no validation.
        self.store_precision = store_precision
        self.assume_centered = assume_centered

    def _set_covariance(self, covariance):
        # [FUNCTION DOC]
        # HELPER: validates covariance with check_array, stores it as covariance_ and sets
        # precision_ to its pseudo-inverse (scipy pinvh) if store_precision, else None.
        # Inputs: covariance - array-like (n_features, n_features), assumed symmetric.
        # Side effects: sets covariance_ and precision_.
        # KNOWN ISSUE: docstring says precision is stored "only if invertible"; the code
        # always stores the pseudo-inverse when store_precision is True.
        """Saves the covariance and precision estimates

        Storage is done accordingly to `self.store_precision`.
        Precision stored only if invertible.

        Parameters
        ----------
        covariance : array-like of shape (n_features, n_features)
            Estimated covariance matrix to be stored, and from which precision
            is computed.
        """
        covariance = check_array(covariance)
        # set covariance
        self.covariance_ = covariance
        # set precision
        if self.store_precision:
            self.precision_ = linalg.pinvh(covariance, check_finite=False)
        else:
            self.precision_ = None

    def get_precision(self):
        # [FUNCTION DOC]
        # Returns precision_ if store_precision, else computes pinvh(covariance_) on the
        # fly (not cached).
        # Outputs: ndarray (n_features, n_features).
        # Exceptions: AttributeError if not fitted.
        """Getter for the precision matrix.

        Returns
        -------
        precision_ : array-like of shape (n_features, n_features)
            The precision matrix associated to the current covariance object.
        """
        if self.store_precision:
            precision = self.precision_
        else:
            precision = linalg.pinvh(self.covariance_, check_finite=False)
        return precision

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None):
        # [FUNCTION DOC]
        # Validates X (validate_data, sets n_features_in_/feature_names_in_), sets
        # location_ to zeros if assume_centered else the column means, computes
        # empirical_covariance with the same assume_centered flag and stores it via
        # _set_covariance.
        # Inputs: X - array-like (n_samples, n_features); y - ignored.
        # Outputs: self.
        # Side effects: sets location_, covariance_, precision_ and feature metadata.
        """Fit the maximum likelihood covariance estimator to X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
          Training data, where `n_samples` is the number of samples and
          `n_features` is the number of features.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        X = validate_data(self, X)
        if self.assume_centered:
            self.location_ = np.zeros(X.shape[1])
        else:
            self.location_ = X.mean(0)
        covariance = empirical_covariance(X, assume_centered=self.assume_centered)
        self._set_covariance(covariance)

        return self

    def score(self, X_test, y=None):
        # [FUNCTION DOC]
        # Average Gaussian log-likelihood of X_test: validates X_test (reset=False),
        # centers it by location_, computes its empirical covariance with
        # assume_centered=True and evaluates log_likelihood with get_precision().
        # Inputs: X_test - array-like (n_samples, n_features); y - ignored.
        # Outputs: float.
        # Exceptions: validation errors on feature mismatch; AttributeError if not fitted.
        """Compute the log-likelihood of `X_test` under the estimated Gaussian model.

        The Gaussian model is defined by its mean and covariance matrix which are
        represented respectively by `self.location_` and `self.covariance_`.

        Parameters
        ----------
        X_test : array-like of shape (n_samples, n_features)
            Test data of which we compute the likelihood, where `n_samples` is
            the number of samples and `n_features` is the number of features.
            `X_test` is assumed to be drawn from the same distribution than
            the data used in fit (including centering).

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        res : float
            The log-likelihood of `X_test` with `self.location_` and `self.covariance_`
            as estimators of the Gaussian model mean and covariance matrix respectively.
        """
        X_test = validate_data(self, X_test, reset=False)
        # compute empirical covariance of the test set
        test_cov = empirical_covariance(X_test - self.location_, assume_centered=True)
        # compute log likelihood
        res = log_likelihood(test_cov, self.get_precision())

        return res

    def error_norm(self, comp_cov, norm="frobenius", scaling=True, squared=True):
        # [FUNCTION DOC]
        # Distance between comp_cov and covariance_: error = comp_cov - covariance_;
        # squared norm is the sum of squared entries ("frobenius") or the largest
        # singular value of error^T error ("spectral", i.e. squared 2-norm). If scaling,
        # divides by n_features (error.shape[0]); returns the square root if not squared.
        # Inputs: comp_cov - array (n_features, n_features); norm - str; scaling, squared
        # - bools, both default True.
        # Outputs: float.
        # Exceptions: NotImplementedError for any other norm value.
        """Compute the Mean Squared Error between two covariance estimators.

        Parameters
        ----------
        comp_cov : array-like of shape (n_features, n_features)
            The covariance to compare with.

        norm : {"frobenius", "spectral"}, default="frobenius"
            The type of norm used to compute the error. Available error types:
            - 'frobenius' (default): sqrt(tr(A^t.A))
            - 'spectral': sqrt(max(eigenvalues(A^t.A))
            where A is the error ``(comp_cov - self.covariance_)``.

        scaling : bool, default=True
            If True (default), the squared error norm is divided by n_features.
            If False, the squared error norm is not rescaled.

        squared : bool, default=True
            Whether to compute the squared error norm or the error norm.
            If True (default), the squared error norm is returned.
            If False, the error norm is returned.

        Returns
        -------
        result : float
            The Mean Squared Error (in the sense of the Frobenius norm) between
            `self` and `comp_cov` covariance estimators.
        """
        # compute the error
        error = comp_cov - self.covariance_
        # compute the error norm
        if norm == "frobenius":
            squared_norm = np.sum(error**2)
        elif norm == "spectral":
            squared_norm = np.amax(linalg.svdvals(np.dot(error.T, error)))
        else:
            raise NotImplementedError(
                "Only spectral and frobenius norms are implemented"
            )
        # optionally scale the error norm
        if scaling:
            squared_norm = squared_norm / error.shape[0]
        # finally get either the squared norm or the norm
        if squared:
            result = squared_norm
        else:
            result = np.sqrt(squared_norm)

        return result

    def mahalanobis(self, X):
        # [FUNCTION DOC]
        # Squared Mahalanobis distances of each row of X to location_ using
        # get_precision() as VI in pairwise_distances (with finiteness checks disabled),
        # reshaped to 1D and squared.
        # Inputs: X - array-like (n_samples, n_features).
        # Outputs: ndarray (n_samples,).
        # Exceptions: validation errors on feature mismatch; AttributeError if not fitted.
        """Compute the squared Mahalanobis distances of given observations.

        For a detailed example of how outliers affects the Mahalanobis distance,
        see :ref:`sphx_glr_auto_examples_covariance_plot_mahalanobis_distances.py`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The observations, the Mahalanobis distances of the which we
            compute. Observations are assumed to be drawn from the same
            distribution than the data used in fit.

        Returns
        -------
        dist : ndarray of shape (n_samples,)
            Squared Mahalanobis distances of the observations.
        """
        X = validate_data(self, X, reset=False)

        precision = self.get_precision()
        with config_context(assume_finite=True):
            # compute mahalanobis distances
            dist = pairwise_distances(
                X, self.location_[np.newaxis, :], metric="mahalanobis", VI=precision
            )

        return np.reshape(dist, (len(X),)) ** 2
