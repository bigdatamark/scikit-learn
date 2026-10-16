# [MODULE DOC]
# Defines EllipticEnvelope, an outlier detector that fits a robust (Minimum
# Covariance Determinant) Gaussian model and flags points whose Mahalanobis
# distance is in the top `contamination` fraction of the training distances.
# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

from numbers import Real

import numpy as np

from sklearn.base import OutlierMixin, _fit_context
from sklearn.covariance._robust_covariance import MinCovDet
from sklearn.metrics import accuracy_score
from sklearn.utils._param_validation import Interval
from sklearn.utils.validation import check_is_fitted


class EllipticEnvelope(OutlierMixin, MinCovDet):
    # [CLASS DOC]
    # Outlier detector built on MinCovDet. fit runs the MCD fit, then sets offset_ to
    # the percentile (100 * contamination) of the negated training distances dist_.
    # Scores are negated squared Mahalanobis distances; decision_function subtracts
    # offset_ so that 0 is the inlier/outlier threshold.
    # Class attribute _parameter_constraints extends MinCovDet's with contamination in
    # (0, 0.5].
    """An object for detecting outliers in a Gaussian distributed dataset.

    Read more in the :ref:`User Guide <outlier_detection>`.

    Parameters
    ----------
    store_precision : bool, default=True
        Specify if the estimated precision is stored.

    assume_centered : bool, default=False
        If True, the support of robust location and covariance estimates
        is computed, and a covariance estimate is recomputed from it,
        without centering the data.
        Useful to work with data whose mean is significantly equal to
        zero but is not exactly zero.
        If False, the robust location and covariance are directly computed
        with the FastMCD algorithm without additional treatment.

    support_fraction : float, default=None
        The proportion of points to be included in the support of the raw
        MCD estimate. If None, the minimum value of support_fraction will
        be used within the algorithm: `(n_samples + n_features + 1) / 2 * n_samples`.
        Range is (0, 1).

    contamination : float, default=0.1
        The amount of contamination of the data set, i.e. the proportion
        of outliers in the data set. Range is (0, 0.5].

    random_state : int, RandomState instance or None, default=None
        Determines the pseudo random number generator for shuffling
        the data. Pass an int for reproducible results across multiple function
        calls. See :term:`Glossary <random_state>`.

    Attributes
    ----------
    location_ : ndarray of shape (n_features,)
        Estimated robust location.

    covariance_ : ndarray of shape (n_features, n_features)
        Estimated robust covariance matrix.

    precision_ : ndarray of shape (n_features, n_features)
        Estimated pseudo inverse matrix.
        (stored only if store_precision is True)

    support_ : ndarray of shape (n_samples,)
        A mask of the observations that have been used to compute the
        robust estimates of location and shape.

    offset_ : float
        Offset used to define the decision function from the raw scores.
        We have the relation: ``decision_function = score_samples - offset_``.
        The offset depends on the contamination parameter and is defined in
        such a way we obtain the expected number of outliers (samples with
        decision function < 0) in training.

        .. versionadded:: 0.20

    raw_location_ : ndarray of shape (n_features,)
        The raw robust estimated location before correction and re-weighting.

    raw_covariance_ : ndarray of shape (n_features, n_features)
        The raw robust estimated covariance before correction and re-weighting.

    raw_support_ : ndarray of shape (n_samples,)
        A mask of the observations that have been used to compute
        the raw robust estimates of location and shape, before correction
        and re-weighting.

    dist_ : ndarray of shape (n_samples,)
        Mahalanobis distances of the training set (on which :meth:`fit` is
        called) observations.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    EmpiricalCovariance : Maximum likelihood covariance estimator.
    GraphicalLasso : Sparse inverse covariance estimation
        with an l1-penalized estimator.
    LedoitWolf : LedoitWolf Estimator.
    MinCovDet : Minimum Covariance Determinant
        (robust estimator of covariance).
    OAS : Oracle Approximating Shrinkage Estimator.
    ShrunkCovariance : Covariance estimator with shrinkage.

    Notes
    -----
    Outlier detection from covariance estimation may break or not
    perform well in high-dimensional settings. In particular, one will
    always take care to work with ``n_samples > n_features ** 2``.

    References
    ----------
    .. [1] Rousseeuw, P.J., Van Driessen, K. "A fast algorithm for the
       minimum covariance determinant estimator" Technometrics 41(3), 212
       (1999)

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.covariance import EllipticEnvelope
    >>> true_cov = np.array([[.8, .3],
    ...                      [.3, .4]])
    >>> X = np.random.RandomState(0).multivariate_normal(mean=[0, 0],
    ...                                                  cov=true_cov,
    ...                                                  size=500)
    >>> cov = EllipticEnvelope(random_state=0).fit(X)
    >>> # predict returns 1 for an inlier and -1 for an outlier
    >>> cov.predict([[0, 0],
    ...              [3, 3]])
    array([ 1, -1])
    >>> cov.covariance_
    array([[0.8102, 0.2736],
           [0.2736, 0.3330]])
    >>> cov.location_
    array([0.0769 , 0.0397])
    """

    _parameter_constraints: dict = {
        **MinCovDet._parameter_constraints,
        "contamination": [Interval(Real, 0, 0.5, closed="right")],
    }

    def __init__(
        self,
        *,
        store_precision=True,
        assume_centered=False,
        support_fraction=None,
        contamination=0.1,
        random_state=None,
    ):
        # [FUNCTION DOC]
        # Stores hyperparameters only (sklearn convention, no validation here).
        # Inputs: store_precision, assume_centered, support_fraction, random_state -
        # forwarded to MinCovDet.__init__; contamination - float in (0, 0.5], expected
        # proportion of outliers, default 0.1.
        # Side effects: sets the matching attributes on self.
        super().__init__(
            store_precision=store_precision,
            assume_centered=assume_centered,
            support_fraction=support_fraction,
            random_state=random_state,
        )
        self.contamination = contamination

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None):
        # [FUNCTION DOC]
        # Fits the robust covariance (MinCovDet.fit on X) then computes offset_ as the
        # (100 * contamination)-th percentile of -dist_ (negated squared robust
        # Mahalanobis distances of the training samples).
        # Inputs: X - array-like (n_samples, n_features); y - ignored.
        # Outputs: self.
        # Exceptions: parameter validation errors via _fit_context; those of MinCovDet.fit.
        # Side effects: sets all MinCovDet fitted attributes plus offset_.
        """Fit the EllipticEnvelope model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        super().fit(X)
        self.offset_ = np.percentile(-self.dist_, 100.0 * self.contamination)
        return self

    def decision_function(self, X):
        # [FUNCTION DOC]
        # Returns score_samples(X) minus offset_; negative values indicate outliers.
        # Inputs: X - array-like (n_samples, n_features).
        # Outputs: ndarray (n_samples,).
        # Exceptions: NotFittedError if not fitted.
        # KNOWN ISSUE: docstring says "shifted Mahalanobis distances"; the values are
        # actually negated squared Mahalanobis distances shifted by offset_.
        """Compute the decision function of the given observations.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data matrix.

        Returns
        -------
        decision : ndarray of shape (n_samples,)
            Decision function of the samples.
            It is equal to the shifted Mahalanobis distances.
            The threshold for being an outlier is 0, which ensures a
            compatibility with other outlier detection algorithms.
        """
        check_is_fitted(self)
        negative_mahal_dist = self.score_samples(X)
        return negative_mahal_dist - self.offset_

    def score_samples(self, X):
        # [FUNCTION DOC]
        # Returns the negation of self.mahalanobis(X), i.e. minus the squared Mahalanobis
        # distance of each sample to the robust location under the robust precision.
        # Inputs: X - array-like (n_samples, n_features).
        # Outputs: ndarray (n_samples,).
        # Exceptions: NotFittedError if not fitted.
        """Compute the negative Mahalanobis distances.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data matrix.

        Returns
        -------
        negative_mahal_distances : array-like of shape (n_samples,)
            Opposite of the Mahalanobis distances.
        """
        check_is_fitted(self)
        return -self.mahalanobis(X)

    def predict(self, X):
        # [FUNCTION DOC]
        # Labels samples: computes decision_function, then returns an int array filled
        # with -1 and set to +1 where the decision value is >= 0.
        # Inputs: X - array-like (n_samples, n_features).
        # Outputs: ndarray of int (n_samples,) with values in {-1, 1}.
        # Exceptions: NotFittedError if not fitted.
        """
        Predict labels (1 inlier, -1 outlier) of X according to fitted model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data matrix.

        Returns
        -------
        is_inlier : ndarray of shape (n_samples,)
            Returns -1 for anomalies/outliers and +1 for inliers.
        """
        values = self.decision_function(X)
        is_inlier = np.full(values.shape[0], -1, dtype=int)
        is_inlier[values >= 0] = 1

        return is_inlier

    def score(self, X, y, sample_weight=None):
        # [FUNCTION DOC]
        # Accuracy of predict(X) against y (expected in {-1, 1} labels), computed with
        # accuracy_score, optionally weighted.
        # Inputs: X - array-like (n_samples, n_features); y - true labels (n_samples,);
        # sample_weight - optional array (n_samples,), default None.
        # Outputs: float mean (weighted) accuracy.
        # Overrides the log-likelihood score inherited from EmpiricalCovariance.
        """Return the mean accuracy on the given test data and labels.

        In multi-label classification, this is the subset accuracy
        which is a harsh metric since you require for each sample that
        each label set be correctly predicted.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            True labels for X.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.

        Returns
        -------
        score : float
            Mean accuracy of self.predict(X) w.r.t. y.
        """
        return accuracy_score(y, self.predict(X), sample_weight=sample_weight)
