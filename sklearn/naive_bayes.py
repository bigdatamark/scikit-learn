# [MODULE DOC]
# Naive Bayes classifiers: Gaussian, Multinomial, Complement, Bernoulli and Categorical.
# All share _BaseNB, which turns a per-class joint log likelihood (log prior plus
# class-conditional log likelihood) into predictions and (log-)probabilities. The
# discrete variants share _BaseDiscreteNB, which accumulates weighted per-class counts
# (sufficient statistics) in fit/partial_fit, then derives smoothed feature log
# probabilities and class log priors from them. GaussianNB supports the Array API;
# the discrete variants are NumPy/SciPy-sparse only.
# Side effects at import: none beyond imports.
"""Naive Bayes algorithms.

These are supervised learning methods based on applying Bayes' theorem with strong
(naive) feature independence assumptions.
"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import warnings
from abc import ABCMeta, abstractmethod
from numbers import Integral, Real

import numpy as np

import sklearn.externals.array_api_extra as xpx
from sklearn.base import BaseEstimator, ClassifierMixin, _fit_context
from sklearn.preprocessing import LabelBinarizer, binarize, label_binarize
from sklearn.utils._array_api import (
    _average,
    _find_matching_floating_dtype,
    _isin,
    _logsumexp,
    get_namespace,
    get_namespace_and_device,
    move_to,
    size,
)
from sklearn.utils._param_validation import Interval
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.utils.multiclass import _check_partial_fit_first_call
from sklearn.utils.validation import (
    _check_n_features,
    _check_sample_weight,
    check_is_fitted,
    check_non_negative,
    validate_data,
)

__all__ = [
    "BernoulliNB",
    "CategoricalNB",
    "ComplementNB",
    "GaussianNB",
    "MultinomialNB",
]


class _BaseNB(ClassifierMixin, BaseEstimator, metaclass=ABCMeta):
    # [CLASS DOC]
    # Abstract base for all naive Bayes estimators. Subclasses implement
    # _joint_log_likelihood (unnormalized log posterior per class) and _check_X (input
    # validation for prediction). This class provides the four predict* methods on top.
    """Abstract base class for naive Bayes estimators"""

    @abstractmethod
    def _joint_log_likelihood(self, X):
        # [FUNCTION DOC]
        # Abstract: return the unnormalized log posterior log P(c) + log P(x|c).
        # Inputs: X - already validated array of shape (n_samples, n_features).
        # Outputs: array of shape (n_samples, n_classes), columns ordered as classes_.
        """Compute the unnormalized posterior log probability of X

        I.e. ``log P(c) + log P(x|c)`` for all rows x of X, as an array-like of
        shape (n_samples, n_classes).

        Public methods predict, predict_proba, predict_log_proba, and
        predict_joint_log_proba pass the input through _check_X before handing it
        over to _joint_log_likelihood. The term "joint log likelihood" is used
        interchangibly with "joint log probability".
        """

    @abstractmethod
    def _check_X(self, X):
        # [FUNCTION DOC]
        # Abstract: validate/convert X for prediction (no reset of n_features_in_).
        # Inputs: X - raw input. Outputs: validated X suitable for
        # _joint_log_likelihood.
        """To be overridden in subclasses with the actual checks.

        Only used in predict* methods.
        """

    def predict_joint_log_proba(self, X):
        # [FUNCTION DOC]
        # Returns the joint log probability log P(x, y) for each sample and class.
        # Steps: check fitted, validate X via _check_X, delegate to
        # _joint_log_likelihood.
        # Inputs: X - array-like (n_samples, n_features).
        # Outputs: array (n_samples, n_classes).
        # Exceptions: NotFittedError if not fitted; validation errors from _check_X.
        """Return joint log probability estimates for the test vector X.

        For each row x of X and class y, the joint log probability is given by
        ``log P(x, y) = log P(y) + log P(x|y),``
        where ``log P(y)`` is the class prior probability and ``log P(x|y)`` is
        the class-conditional probability.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        C : ndarray of shape (n_samples, n_classes)
            Returns the joint log-probability of the samples for each class in
            the model. The columns correspond to the classes in sorted
            order, as they appear in the attribute :term:`classes_`.
        """
        check_is_fitted(self)
        X = self._check_X(X)
        return self._joint_log_likelihood(X)

    def predict(self, X):
        # [FUNCTION DOC]
        # Predicts the class label per sample as the class with largest joint log
        # likelihood.
        # Steps: check fitted, validate X, compute jll, argmax along the class axis,
        # index
        # classes_ with the winning indices. When labels are strings the indices are
        # moved to
        # NumPy on CPU first because string classes_ is a NumPy array.
        # Inputs: X - array-like (n_samples, n_features).
        # Outputs: array (n_samples,) of labels taken from classes_.
        # Exceptions: NotFittedError; validation errors.
        """
        Perform classification on an array of test vectors X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        C : ndarray of shape (n_samples,)
            Predicted target values for X.
        """
        check_is_fitted(self)
        xp, _ = get_namespace(X)
        X = self._check_X(X)
        jll = self._joint_log_likelihood(X)
        pred_indices = xp.argmax(jll, axis=1)
        if isinstance(self.classes_[0], str):
            pred_indices = move_to(pred_indices, xp=np, device="cpu")
        return self.classes_[pred_indices]

    def predict_log_proba(self, X):
        # [FUNCTION DOC]
        # Returns normalized log posterior probabilities.
        # Steps: check fitted, validate X, compute jll, subtract per-row logsumexp of
        # jll
        # (i.e. log P(x)) from every column.
        # Inputs: X - array-like (n_samples, n_features).
        # Outputs: array (n_samples, n_classes); each row's exp sums to 1.
        # Exceptions: NotFittedError; validation errors.
        """
        Return log-probability estimates for the test vector X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        C : array-like of shape (n_samples, n_classes)
            Returns the log-probability of the samples for each class in
            the model. The columns correspond to the classes in sorted
            order, as they appear in the attribute :term:`classes_`.
        """
        check_is_fitted(self)
        xp, _ = get_namespace(X)
        X = self._check_X(X)
        jll = self._joint_log_likelihood(X)
        # normalize by P(x) = P(f_1, ..., f_n)
        log_prob_x = _logsumexp(jll, axis=1, xp=xp)
        return jll - xpx.atleast_nd(log_prob_x, ndim=2).T

    def predict_proba(self, X):
        # [FUNCTION DOC]
        # Returns posterior probabilities as the exponential of predict_log_proba.
        # Inputs: X - array-like (n_samples, n_features).
        # Outputs: array (n_samples, n_classes), rows sum to 1.
        """
        Return probability estimates for the test vector X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        C : array-like of shape (n_samples, n_classes)
            Returns the probability of the samples for each class in
            the model. The columns correspond to the classes in sorted
            order, as they appear in the attribute :term:`classes_`.
        """
        xp, _ = get_namespace(X)
        return xp.exp(self.predict_log_proba(X))


class GaussianNB(_BaseNB):
    # [CLASS DOC]
    # Naive Bayes with per-class, per-feature Gaussian likelihoods (diagonal
    # covariance).
    # Stores theta_ (means), var_ (variances plus epsilon_), class_count_ and
    # class_prior_. Supports incremental learning and the Array API.
    """
    Gaussian Naive Bayes (GaussianNB).

    Can perform online updates to model parameters via :meth:`partial_fit`.
    For details on algorithm used to update feature means and variance online,
    see `Stanford CS tech report STAN-CS-79-773 by Chan, Golub, and LeVeque
    <http://i.stanford.edu/pub/cstr/reports/cs/tr/79/773/CS-TR-79-773.pdf>`_.

    Read more in the :ref:`User Guide <gaussian_naive_bayes>`.

    Parameters
    ----------
    priors : array-like of shape (n_classes,), default=None
        Prior probabilities of the classes. If specified, the priors are not
        adjusted according to the data.

    var_smoothing : float, default=1e-9
        Portion of the largest variance of all features that is added to
        variances for calculation stability.

        .. versionadded:: 0.20

    Attributes
    ----------
    class_count_ : ndarray of shape (n_classes,)
        number of training samples observed in each class.

    class_prior_ : ndarray of shape (n_classes,)
        probability of each class.

    classes_ : ndarray of shape (n_classes,)
        class labels known to the classifier.

    epsilon_ : float
        absolute additive value to variances.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    var_ : ndarray of shape (n_classes, n_features)
        Variance of each feature per class.

        .. versionadded:: 1.0

    theta_ : ndarray of shape (n_classes, n_features)
        mean of each feature per class.

    See Also
    --------
    BernoulliNB : Naive Bayes classifier for multivariate Bernoulli models.
    CategoricalNB : Naive Bayes classifier for categorical features.
    ComplementNB : Complement Naive Bayes classifier.
    MultinomialNB : Naive Bayes classifier for multinomial models.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    >>> Y = np.array([1, 1, 1, 2, 2, 2])
    >>> from sklearn.naive_bayes import GaussianNB
    >>> clf = GaussianNB()
    >>> clf.fit(X, Y)
    GaussianNB()
    >>> print(clf.predict([[-0.8, -1]]))
    [1]
    >>> clf_pf = GaussianNB()
    >>> clf_pf.partial_fit(X, Y, np.unique(Y))
    GaussianNB()
    >>> print(clf_pf.predict([[-0.8, -1]]))
    [1]
    """

    _parameter_constraints: dict = {
        "priors": ["array-like", None],
        "var_smoothing": [Interval(Real, 0, None, closed="left")],
    }

    def __init__(self, *, priors=None, var_smoothing=1e-9):
        # [FUNCTION DOC]
        # Stores hyper-parameters unchanged (sklearn convention, no validation here).
        # Inputs: priors - optional class priors (n_classes,); var_smoothing - float,
        # fraction
        # of the largest feature variance added to all variances.
        self.priors = priors
        self.var_smoothing = var_smoothing

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y, sample_weight=None):
        # [FUNCTION DOC]
        # Fits from scratch: validates y alone, then calls _partial_fit with classes set
        # to the
        # unique values of y and _refit=True (which discards any previous classes_).
        # Inputs: X (n_samples, n_features); y (n_samples,); sample_weight optional
        # (n_samples,).
        # Outputs: self.
        # Exceptions: see _partial_fit; parameter validation via _fit_context.
        # Side effects: sets all fitted attributes.
        """Fit Gaussian Naive Bayes according to X, y.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vectors, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        y : array-like of shape (n_samples,)
            Target values.

        sample_weight : array-like of shape (n_samples,), default=None
            Weights applied to individual samples (1. for unweighted).

            .. versionadded:: 0.17
               Gaussian Naive Bayes supports fitting with *sample_weight*.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        y = validate_data(self, y=y)
        xp_y, _ = get_namespace(y)
        return self._partial_fit(
            X, y, xp_y.unique_values(y), _refit=True, sample_weight=sample_weight
        )

    def _check_X(self, X):
        # [FUNCTION DOC]
        # Validates X for prediction without resetting n_features_in_ (dense only).
        # Outputs: validated array.
        """Validate X, used only in predict* methods."""
        return validate_data(self, X, reset=False)

    @staticmethod
    def _update_mean_variance(n_past, mu, var, X, sample_weight=None):
        # [FUNCTION DOC]
        # HELPER: Online combination of a running Gaussian mean/variance with a new
        # batch
        # (Chan, Golub and LeVeque pairwise update).
        # Steps: if X has no rows, or the sample weights sum to ~0, return mu and var
        # unchanged.
        # Otherwise compute the (weighted) mean and biased variance of the batch along
        # axis 0
        # and its (weighted) count n_new. If n_past is 0, return the batch statistics.
        # Else total mean is the count-weighted average of both means; total sum of
        # squared
        # deviations is old_ssd plus new_ssd plus n_new*n_past/n_total times the squared
        # difference of the means; total variance is that divided by n_total.
        # Inputs: n_past - (weighted) count already seen; mu, var - arrays
        # (n_features,);
        # X - batch (n_samples, n_features); sample_weight - optional (n_samples,).
        # Outputs: tuple (total_mu, total_var), each (n_features,).
        # Side effects: none.
        """Compute online update of Gaussian mean and variance.

        Given starting sample count, mean, and variance, a new set of
        points X, and optionally sample weights, return the updated mean and
        variance. (NB - each dimension (column) in X is treated as independent
        -- you get variance, not covariance).

        Can take scalar mean and variance, or vector mean and variance to
        simultaneously update a number of independent Gaussians.

        See Stanford CS tech report STAN-CS-79-773 by Chan, Golub, and LeVeque:

        http://i.stanford.edu/pub/cstr/reports/cs/tr/79/773/CS-TR-79-773.pdf

        Parameters
        ----------
        n_past : int
            Number of samples represented in old mean and variance. If sample
            weights were given, this should contain the sum of sample
            weights represented in old mean and variance.

        mu : array-like of shape (number of Gaussians,)
            Means for Gaussians in original set.

        var : array-like of shape (number of Gaussians,)
            Variances for Gaussians in original set.

        sample_weight : array-like of shape (n_samples,), default=None
            Weights applied to individual samples (1. for unweighted).

        Returns
        -------
        total_mu : array-like of shape (number of Gaussians,)
            Updated mean for each Gaussian over the combined set.

        total_var : array-like of shape (number of Gaussians,)
            Updated variance for each Gaussian over the combined set.
        """
        xp, _ = get_namespace(X)
        if X.shape[0] == 0:
            return mu, var

        # Compute (potentially weighted) mean and variance of new datapoints
        if sample_weight is not None:
            n_new = float(xp.sum(sample_weight))
            if np.isclose(n_new, 0.0):
                return mu, var
            new_mu = _average(X, axis=0, weights=sample_weight, xp=xp)
            new_var = _average((X - new_mu) ** 2, axis=0, weights=sample_weight, xp=xp)
        else:
            n_new = X.shape[0]
            new_var = xp.var(X, axis=0)
            new_mu = xp.mean(X, axis=0)

        if n_past == 0:
            return new_mu, new_var

        n_total = float(n_past + n_new)

        # Combine mean of old and new data, taking into consideration
        # (weighted) number of observations
        total_mu = (n_new * new_mu + n_past * mu) / n_total

        # Combine variance of old and new data, taking into consideration
        # (weighted) number of observations. This is achieved by combining
        # the sum-of-squared-differences (ssd)
        old_ssd = n_past * var
        new_ssd = n_new * new_var
        total_ssd = old_ssd + new_ssd + (n_new * n_past / n_total) * (mu - new_mu) ** 2
        total_var = total_ssd / n_total

        return total_mu, total_var

    @_fit_context(prefer_skip_nested_validation=True)
    def partial_fit(self, X, y, classes=None, sample_weight=None):
        # [FUNCTION DOC]
        # Incremental fit on a batch; thin wrapper calling _partial_fit with
        # _refit=False.
        # Inputs: X, y as in fit; classes - all possible labels, required on the first
        # call;
        # sample_weight optional.
        # Outputs: self.
        # Exceptions: ValueError if classes is missing on first call or inconsistent
        # later.
        """Incremental fit on a batch of samples.

        This method is expected to be called several times consecutively
        on different chunks of a dataset so as to implement out-of-core
        or online learning.

        This is especially useful when the whole dataset is too big to fit in
        memory at once.

        This method has some performance and numerical stability overhead,
        hence it is better to call partial_fit on chunks of data that are
        as large as possible (as long as fitting in the memory budget) to
        hide the overhead.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vectors, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        y : array-like of shape (n_samples,)
            Target values.

        classes : array-like of shape (n_classes,), default=None
            List of all the classes that can possibly appear in the y vector.

            Must be provided at the first call to partial_fit, can be omitted
            in subsequent calls.

        sample_weight : array-like of shape (n_samples,), default=None
            Weights applied to individual samples (1. for unweighted).

            .. versionadded:: 0.17

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        return self._partial_fit(
            X, y, classes, _refit=False, sample_weight=sample_weight
        )

    def _partial_fit(self, X, y, classes=None, _refit=False, sample_weight=None):
        # [FUNCTION DOC]
        # Core fitting routine shared by fit and partial_fit.
        # Steps: if _refit, clear classes_. Determine whether this is the first call
        # (sets
        # classes_ on first call). Validate X, y (reset feature info only on the first
        # call),
        # pick a floating dtype, validate sample_weight. Set epsilon_ to var_smoothing
        # times
        # the largest per-feature variance of THIS batch. On first call allocate zero
        # theta_,
        # var_ (n_classes, n_features) and class_count_, and either validate and store
        # user priors or zero class_prior_. On later calls check the feature count
        # matches
        # and subtract epsilon_ from var_. Verify every label in y belongs to classes_.
        # For
        # each label present, select its rows (and weights) and merge them into that
        # class's mean/variance with _update_mean_variance, then add the (weighted)
        # count to
        # class_count_. Finally add epsilon_ back to var_, and if no priors were given
        # set
        # class_prior_ to normalized class_count_.
        # Inputs: X, y, classes, sample_weight as in partial_fit; _refit - bool, start
        # over.
        # Outputs: self.
        # Exceptions: ValueError for wrong number of priors, priors not summing to 1,
        # negative priors, feature-count mismatch, labels not in classes_.
        # Side effects: sets/updates classes_, theta_, var_, class_count_, class_prior_,
        # epsilon_, n_features_in_ (and feature_names_in_).
        # KNOWN ISSUE: on a later partial_fit call, the epsilon_ subtracted from var_ is
        # the
        # newly computed one (from the current batch), not the one added on the previous
        # call,
        # so var_ drifts by the difference between the two epsilons across batches.
        """Actual implementation of Gaussian NB fitting.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vectors, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        y : array-like of shape (n_samples,)
            Target values.

        classes : array-like of shape (n_classes,), default=None
            List of all the classes that can possibly appear in the y vector.

            Must be provided at the first call to partial_fit, can be omitted
            in subsequent calls.

        _refit : bool, default=False
            If true, act as though this were the first time we called
            _partial_fit (ie, throw away any past fitting and start over).

        sample_weight : array-like of shape (n_samples,), default=None
            Weights applied to individual samples (1. for unweighted).

        Returns
        -------
        self : object
        """
        if _refit:
            self.classes_ = None

        first_call = _check_partial_fit_first_call(self, classes)
        X, y = validate_data(self, X, y, reset=first_call)
        xp, _, device_ = get_namespace_and_device(X)
        float_dtype = _find_matching_floating_dtype(X, xp=xp)
        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X, dtype=float_dtype)

        xp_y, _ = get_namespace(y)
        # If the ratio of data variance between dimensions is too small, it
        # will cause numerical errors. To address this, we artificially
        # boost the variance by epsilon, a small fraction of the standard
        # deviation of the largest dimension.
        self.epsilon_ = self.var_smoothing * xp.max(xp.var(X, axis=0))

        if first_call:
            # This is the first call to partial_fit:
            # initialize various cumulative counters
            n_features = X.shape[1]
            n_classes = self.classes_.shape[0]
            self.theta_ = xp.zeros(
                (n_classes, n_features), dtype=float_dtype, device=device_
            )
            self.var_ = xp.zeros(
                (n_classes, n_features), dtype=float_dtype, device=device_
            )

            self.class_count_ = xp.zeros(n_classes, dtype=float_dtype, device=device_)

            # Initialise the class prior
            # Take into account the priors
            if self.priors is not None:
                priors = xp.asarray(self.priors, dtype=float_dtype, device=device_)
                # Check that the provided prior matches the number of classes
                if priors.shape[0] != n_classes:
                    raise ValueError("Number of priors must match number of classes.")
                # Check that the sum is 1
                if not xpx.isclose(xp.sum(priors), 1.0):
                    raise ValueError("The sum of the priors should be 1.")
                # Check that the priors are non-negative
                if xp.any(priors < 0):
                    raise ValueError("Priors must be non-negative.")
                self.class_prior_ = priors
            else:
                # Initialize the priors to zeros for each class
                self.class_prior_ = xp.zeros(
                    self.classes_.shape[0], dtype=float_dtype, device=device_
                )
        else:
            if X.shape[1] != self.theta_.shape[1]:
                msg = "Number of features %d does not match previous data %d."
                raise ValueError(msg % (X.shape[1], self.theta_.shape[1]))
            # Put epsilon back in each time
            self.var_[:, :] -= self.epsilon_

        classes = self.classes_

        unique_y = xp_y.unique_values(y)
        unique_y_in_classes = _isin(unique_y, classes, xp=xp_y)

        if not xp_y.all(unique_y_in_classes):
            raise ValueError(
                "The target label(s) %s in y do not exist in the initial classes %s"
                % (unique_y[~unique_y_in_classes], classes)
            )

        for y_i in unique_y:
            i = int(xp_y.searchsorted(classes, y_i))
            y_i_mask = xp.asarray(y == y_i, device=device_)
            X_i = X[y_i_mask]

            if sample_weight is not None:
                sw_i = sample_weight[y_i_mask]
                N_i = xp.sum(sw_i)
            else:
                sw_i = None
                N_i = X_i.shape[0]

            new_theta, new_sigma = self._update_mean_variance(
                self.class_count_[i], self.theta_[i, :], self.var_[i, :], X_i, sw_i
            )

            self.theta_[i, :] = new_theta
            self.var_[i, :] = new_sigma
            self.class_count_[i] += N_i

        self.var_[:, :] += self.epsilon_

        # Update if only no priors is provided
        if self.priors is None:
            # Empirical prior, with sample_weight taken into account
            self.class_prior_ = self.class_count_ / xp.sum(self.class_count_)

        return self

    def _joint_log_likelihood(self, X):
        # [FUNCTION DOC]
        # For each class i, computes log prior plus the Gaussian log density summed over
        # features: -0.5 times the sum of log(2*pi*var) minus 0.5 times the sum of
        # squared
        # deviations from theta_ divided by var. Stacks per-class columns.
        # Inputs: X - validated (n_samples, n_features).
        # Outputs: array (n_samples, n_classes).
        xp, _ = get_namespace(X)
        joint_log_likelihood = []
        for i in range(size(self.classes_)):
            jointi = xp.log(self.class_prior_[i])
            n_ij = -0.5 * xp.sum(xp.log(2.0 * xp.pi * self.var_[i, :]))
            n_ij = n_ij - 0.5 * xp.sum(
                ((X - self.theta_[i, :]) ** 2) / (self.var_[i, :]), axis=1
            )
            joint_log_likelihood.append(jointi + n_ij)

        joint_log_likelihood = xp.stack(joint_log_likelihood).T
        return joint_log_likelihood

    def __sklearn_tags__(self):
        # [FUNCTION DOC]
        # Extends parent tags to declare Array API support.
        # Outputs: Tags object.
        tags = super().__sklearn_tags__()
        tags.array_api_support = True
        return tags


class _BaseDiscreteNB(_BaseNB):
    # [CLASS DOC]
    # Abstract base for count-based naive Bayes models (Multinomial, Complement,
    # Bernoulli, Categorical). Implements fit/partial_fit: binarize labels into a one-
    # hot
    # matrix Y (weighted by sample_weight), let the subclass accumulate counts (_count),
    # smooth via alpha (_update_feature_log_prob), and set class_log_prior_.
    """Abstract base class for naive Bayes on discrete/categorical data

    Any estimator based on this class should provide:

    __init__
    _joint_log_likelihood(X) as per _BaseNB
    _update_feature_log_prob(alpha)
    _count(X, Y)
    """

    _parameter_constraints: dict = {
        "alpha": [Interval(Real, 0, None, closed="left"), "array-like"],
        "fit_prior": ["boolean"],
        "class_prior": ["array-like", None],
        "force_alpha": ["boolean"],
    }

    def __init__(self, alpha=1.0, fit_prior=True, class_prior=None, force_alpha=True):
        # [FUNCTION DOC]
        # Stores hyper-parameters alpha, fit_prior, class_prior, force_alpha unchanged.
        self.alpha = alpha
        self.fit_prior = fit_prior
        self.class_prior = class_prior
        self.force_alpha = force_alpha

    @abstractmethod
    def _count(self, X, Y):
        # [FUNCTION DOC]
        # Abstract: add counts from X and one-hot/weighted label matrix Y (n_samples,
        # n_classes) into class_count_, feature_count_ and any model-specific counters.
        # Side effects: mutates the count attributes in place.
        """Update counts that are used to calculate probabilities.

        The counts make up a sufficient statistic extracted from the data.
        Accordingly, this method is called each time `fit` or `partial_fit`
        update the model. `class_count_` and `feature_count_` must be updated
        here along with any model specific counts.

        Parameters
        ----------
        X : {ndarray, sparse matrix} of shape (n_samples, n_features)
            The input samples.
        Y : ndarray of shape (n_samples, n_classes)
            Binarized class labels.
        """

    @abstractmethod
    def _update_feature_log_prob(self, alpha):
        # [FUNCTION DOC]
        # Abstract: recompute feature_log_prob_ from the accumulated counts using
        # smoothing
        # value alpha (scalar or per-feature array). Side effects: sets
        # feature_log_prob_.
        """Update feature log probabilities based on counts.

        This method is called each time `fit` or `partial_fit` update the
        model.

        Parameters
        ----------
        alpha : float
            smoothing parameter. See :meth:`_check_alpha`.
        """

    def _check_X(self, X):
        # [FUNCTION DOC]
        # Validates X for prediction, accepting CSR sparse input, no reset.
        """Validate X, used only in predict* methods."""
        return validate_data(self, X, accept_sparse="csr", reset=False)

    def _check_X_y(self, X, y, reset=True):
        # [FUNCTION DOC]
        # Validates X and y for fitting, accepting CSR sparse X.
        # Inputs: reset - whether to (re)set n_features_in_/feature_names_in_.
        # Outputs: tuple (X, y).
        """Validate X and y in fit methods."""
        return validate_data(self, X, y, accept_sparse="csr", reset=reset)

    def _update_class_log_prior(self, class_prior=None):
        # [FUNCTION DOC]
        # Sets class_log_prior_. If class_prior is given, use its log. Else if
        # fit_prior, use
        # log of class_count_ minus log of the total count (log(0) warnings for classes
        # not
        # yet observed are silenced, giving -inf). Otherwise use a uniform prior, -log
        # of the
        # number of classes for every class.
        # Inputs: class_prior - optional array (n_classes,).
        # Exceptions: ValueError if class_prior length differs from the number of
        # classes.
        # Side effects: sets class_log_prior_.
        """Update class log priors.

        The class log priors are based on `class_prior`, class count or the
        number of classes. This method is called each time `fit` or
        `partial_fit` update the model.
        """
        n_classes = len(self.classes_)
        if class_prior is not None:
            if len(class_prior) != n_classes:
                raise ValueError("Number of priors must match number of classes.")
            self.class_log_prior_ = np.log(class_prior)
        elif self.fit_prior:
            with warnings.catch_warnings():
                # silence the warning when count is 0 because class was not yet
                # observed
                warnings.simplefilter("ignore", RuntimeWarning)
                log_class_count = np.log(self.class_count_)

            # empirical prior, with sample_weight taken into account
            self.class_log_prior_ = log_class_count - np.log(self.class_count_.sum())
        else:
            self.class_log_prior_ = np.full(n_classes, -np.log(n_classes))

    def _check_alpha(self):
        # [FUNCTION DOC]
        # Returns the effective smoothing value. Converts non-scalar alpha to an array;
        # if an
        # array, checks its length equals n_features_in_ and that no entry is negative.
        # If the
        # minimum alpha is below 1e-10 and force_alpha is False, warns and clips alpha
        # up to
        # 1e-10.
        # Outputs: float or ndarray (n_features,).
        # Exceptions: ValueError for wrong array length or negative entries.
        # Side effects: UserWarning when clipping.
        # KNOWN ISSUE: the error text says values must be "greater than 0" but the check
        # only
        # rejects negatives, so zeros are accepted.
        alpha = (
            np.asarray(self.alpha) if not isinstance(self.alpha, Real) else self.alpha
        )
        alpha_min = np.min(alpha)
        if isinstance(alpha, np.ndarray):
            if not alpha.shape[0] == self.n_features_in_:
                raise ValueError(
                    "When alpha is an array, it should contains `n_features`. "
                    f"Got {alpha.shape[0]} elements instead of {self.n_features_in_}."
                )
            # check that all alpha are positive
            if alpha_min < 0:
                raise ValueError("All values in alpha must be greater than 0.")
        alpha_lower_bound = 1e-10
        if alpha_min < alpha_lower_bound and not self.force_alpha:
            warnings.warn(
                "alpha too small will result in numeric errors, setting alpha ="
                f" {alpha_lower_bound:.1e}. Use `force_alpha=True` to keep alpha"
                " unchanged."
            )
            return np.maximum(alpha, alpha_lower_bound)
        return alpha

    @_fit_context(prefer_skip_nested_validation=True)
    def partial_fit(self, X, y, classes=None, sample_weight=None):
        # [FUNCTION DOC]
        # Incremental fit on a batch.
        # Steps: first call is detected by absence of classes_; validate X, y (resetting
        # feature info only on first call). If first call (classes required), allocate
        # counters
        # via _init_counters. One-hot encode y against classes_; a single column is
        # expanded to
        # two complementary columns for binary problems or set to ones for one class.
        # Check row
        # counts match, convert Y to float64, and multiply each row by its sample
        # weight.
        # Then _count, _check_alpha, _update_feature_log_prob, _update_class_log_prior.
        # Inputs: X (n_samples, n_features), dense or sparse; y (n_samples,); classes -
        # all
        # labels, required on first call; sample_weight optional (n_samples,).
        # Outputs: self.
        # Exceptions: ValueError if classes missing/inconsistent, or X and y lengths
        # differ.
        # Side effects: updates counters and fitted attributes.
        """Incremental fit on a batch of samples.

        This method is expected to be called several times consecutively
        on different chunks of a dataset so as to implement out-of-core
        or online learning.

        This is especially useful when the whole dataset is too big to fit in
        memory at once.

        This method has some performance overhead hence it is better to call
        partial_fit on chunks of data that are as large as possible
        (as long as fitting in the memory budget) to hide the overhead.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vectors, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        y : array-like of shape (n_samples,)
            Target values.

        classes : array-like of shape (n_classes,), default=None
            List of all the classes that can possibly appear in the y vector.

            Must be provided at the first call to partial_fit, can be omitted
            in subsequent calls.

        sample_weight : array-like of shape (n_samples,), default=None
            Weights applied to individual samples (1. for unweighted).

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        first_call = not hasattr(self, "classes_")

        X, y = self._check_X_y(X, y, reset=first_call)
        _, n_features = X.shape

        if _check_partial_fit_first_call(self, classes):
            # This is the first call to partial_fit:
            # initialize various cumulative counters
            n_classes = len(classes)
            self._init_counters(n_classes, n_features)

        Y = label_binarize(y, classes=self.classes_)
        if Y.shape[1] == 1:
            if len(self.classes_) == 2:
                Y = np.concatenate((1 - Y, Y), axis=1)
            else:  # degenerate case: just one class
                Y = np.ones_like(Y)

        if X.shape[0] != Y.shape[0]:
            msg = "X.shape[0]=%d and y.shape[0]=%d are incompatible."
            raise ValueError(msg % (X.shape[0], y.shape[0]))

        # label_binarize() returns arrays with dtype=np.int64.
        # We convert it to np.float64 to support sample_weight consistently
        Y = Y.astype(np.float64, copy=False)
        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X)
            sample_weight = np.atleast_2d(sample_weight)
            Y *= sample_weight.T

        class_prior = self.class_prior

        # Count raw events from data before updating the class log prior
        # and feature log probas
        self._count(X, Y)

        # XXX: OPTIM: we could introduce a public finalization method to
        # be called by the user explicitly just once after several consecutive
        # calls to partial_fit and prior any call to predict[_[log_]proba]
        # to avoid computing the smooth log probas at each call to partial fit
        alpha = self._check_alpha()
        self._update_feature_log_prob(alpha)
        self._update_class_log_prior(class_prior=class_prior)
        return self

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y, sample_weight=None):
        # [FUNCTION DOC]
        # Fits from scratch. Validates X, y; one-hot encodes y with LabelBinarizer
        # (setting
        # classes_), handling the 1-column binary/single-class cases as in partial_fit;
        # when
        # sample_weight is given converts Y to float and scales rows by weights. Resets
        # counters, then _count, smoothing via _check_alpha/_update_feature_log_prob,
        # and
        # _update_class_log_prior.
        # Inputs: X, y, optional sample_weight. Outputs: self.
        # Side effects: sets classes_, counters, feature_log_prob_, class_log_prior_.
        """Fit Naive Bayes classifier according to X, y.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vectors, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        y : array-like of shape (n_samples,)
            Target values.

        sample_weight : array-like of shape (n_samples,), default=None
            Weights applied to individual samples (1. for unweighted).

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        X, y = self._check_X_y(X, y)
        _, n_features = X.shape

        labelbin = LabelBinarizer()
        Y = labelbin.fit_transform(y)
        self.classes_ = labelbin.classes_
        if Y.shape[1] == 1:
            if len(self.classes_) == 2:
                Y = np.concatenate((1 - Y, Y), axis=1)
            else:  # degenerate case: just one class
                Y = np.ones_like(Y)

        # LabelBinarizer().fit_transform() returns arrays with dtype=np.int64.
        # We convert it to np.float64 to support sample_weight consistently;
        # this means we also don't have to cast X to floating point
        if sample_weight is not None:
            Y = Y.astype(np.float64, copy=False)
            sample_weight = _check_sample_weight(sample_weight, X)
            sample_weight = np.atleast_2d(sample_weight)
            Y *= sample_weight.T

        class_prior = self.class_prior

        # Count raw events from data before updating the class log prior
        # and feature log probas
        n_classes = Y.shape[1]
        self._init_counters(n_classes, n_features)
        self._count(X, Y)
        alpha = self._check_alpha()
        self._update_feature_log_prob(alpha)
        self._update_class_log_prior(class_prior=class_prior)
        return self

    def _init_counters(self, n_classes, n_features):
        # [FUNCTION DOC]
        # HELPER: Allocates zero class_count_ (n_classes,) and feature_count_
        # (n_classes,
        # n_features) as float64. Side effects: overwrites those attributes.
        self.class_count_ = np.zeros(n_classes, dtype=np.float64)
        self.feature_count_ = np.zeros((n_classes, n_features), dtype=np.float64)

    def __sklearn_tags__(self):
        # [FUNCTION DOC]
        # Declares sparse input support and marks the classifier as possibly poor-
        # scoring for
        # common tests.
        tags = super().__sklearn_tags__()
        tags.input_tags.sparse = True
        tags.classifier_tags.poor_score = True
        return tags


class MultinomialNB(_BaseDiscreteNB):
    # [CLASS DOC]
    # Multinomial naive Bayes for count features (e.g. word counts). Feature log
    # probability is the log of smoothed per-class feature counts over their row total.
    """
    Naive Bayes classifier for multinomial models.

    The multinomial Naive Bayes classifier is suitable for classification with
    discrete features (e.g., word counts for text classification). The
    multinomial distribution normally requires integer feature counts. However,
    in practice, fractional counts such as tf-idf may also work.

    Read more in the :ref:`User Guide <multinomial_naive_bayes>`.

    Parameters
    ----------
    alpha : float or array-like of shape (n_features,), default=1.0
        Additive (Laplace/Lidstone) smoothing parameter
        (set alpha=0 and force_alpha=True, for no smoothing).

    force_alpha : bool, default=True
        If False and alpha is less than 1e-10, it will set alpha to
        1e-10. If True, alpha will remain unchanged. This may cause
        numerical errors if alpha is too close to 0.

        .. versionadded:: 1.2
        .. versionchanged:: 1.4
           The default value of `force_alpha` changed to `True`.

    fit_prior : bool, default=True
        Whether to learn class prior probabilities or not.
        If false, a uniform prior will be used.

    class_prior : array-like of shape (n_classes,), default=None
        Prior probabilities of the classes. If specified, the priors are not
        adjusted according to the data.

    Attributes
    ----------
    class_count_ : ndarray of shape (n_classes,)
        Number of samples encountered for each class during fitting. This
        value is weighted by the sample weight when provided.

    class_log_prior_ : ndarray of shape (n_classes,)
        Smoothed empirical log probability for each class.

    classes_ : ndarray of shape (n_classes,)
        Class labels known to the classifier

    feature_count_ : ndarray of shape (n_classes, n_features)
        Number of samples encountered for each (class, feature)
        during fitting. This value is weighted by the sample weight when
        provided.

    feature_log_prob_ : ndarray of shape (n_classes, n_features)
        Empirical log probability of features
        given a class, ``P(x_i|y)``.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    BernoulliNB : Naive Bayes classifier for multivariate Bernoulli models.
    CategoricalNB : Naive Bayes classifier for categorical features.
    ComplementNB : Complement Naive Bayes classifier.
    GaussianNB : Gaussian Naive Bayes.

    References
    ----------
    C.D. Manning, P. Raghavan and H. Schuetze (2008). Introduction to
    Information Retrieval. Cambridge University Press, pp. 234-265.
    https://nlp.stanford.edu/IR-book/html/htmledition/naive-bayes-text-classification-1.html

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.RandomState(1)
    >>> X = rng.randint(5, size=(6, 100))
    >>> y = np.array([1, 2, 3, 4, 5, 6])
    >>> from sklearn.naive_bayes import MultinomialNB
    >>> clf = MultinomialNB()
    >>> clf.fit(X, y)
    MultinomialNB()
    >>> print(clf.predict(X[2:3]))
    [3]
    """

    def __init__(
        self, *, alpha=1.0, force_alpha=True, fit_prior=True, class_prior=None
    ):
        # [FUNCTION DOC]
        # Forwards keyword-only hyper-parameters to _BaseDiscreteNB.__init__.
        super().__init__(
            alpha=alpha,
            fit_prior=fit_prior,
            class_prior=class_prior,
            force_alpha=force_alpha,
        )

    def __sklearn_tags__(self):
        # [FUNCTION DOC]
        # Declares that input must be non-negative.
        tags = super().__sklearn_tags__()
        tags.input_tags.positive_only = True
        return tags

    def _count(self, X, Y):
        # [FUNCTION DOC]
        # Adds Y transposed times X to feature_count_ and column sums of Y to
        # class_count_.
        # Exceptions: ValueError if X contains negative values.
        # Side effects: mutates feature_count_, class_count_.
        """Count and smooth feature occurrences."""
        check_non_negative(X, "MultinomialNB (input X)")
        self.feature_count_ += safe_sparse_dot(Y.T, X)
        self.class_count_ += Y.sum(axis=0)

    def _update_feature_log_prob(self, alpha):
        # [FUNCTION DOC]
        # Adds alpha to feature_count_, then sets feature_log_prob_ to the log of each
        # entry
        # minus the log of its row sum. Side effects: sets feature_log_prob_.
        """Apply smoothing to raw counts and recompute log probabilities"""
        smoothed_fc = self.feature_count_ + alpha
        smoothed_cc = smoothed_fc.sum(axis=1)

        self.feature_log_prob_ = np.log(smoothed_fc) - np.log(
            smoothed_cc.reshape(-1, 1)
        )

    def _joint_log_likelihood(self, X):
        # [FUNCTION DOC]
        # Returns X times feature_log_prob_ transposed plus class_log_prior_, shape
        # (n_samples, n_classes).
        """Calculate the posterior log probability of the samples X"""
        return safe_sparse_dot(X, self.feature_log_prob_.T) + self.class_log_prior_


class ComplementNB(_BaseDiscreteNB):
    # [CLASS DOC]
    # Complement naive Bayes (Rennie et al. 2003): weights are estimated from counts of
    # all
    # classes except the target one; suited to imbalanced text data. Priors are ignored
    # except when only one class is present.
    """The Complement Naive Bayes classifier described in Rennie et al. (2003).

    The Complement Naive Bayes classifier was designed to correct the "severe
    assumptions" made by the standard Multinomial Naive Bayes classifier. It is
    particularly suited for imbalanced data sets.

    Read more in the :ref:`User Guide <complement_naive_bayes>`.

    .. versionadded:: 0.20

    Parameters
    ----------
    alpha : float or array-like of shape (n_features,), default=1.0
        Additive (Laplace/Lidstone) smoothing parameter
        (set alpha=0 and force_alpha=True, for no smoothing).

    force_alpha : bool, default=True
        If False and alpha is less than 1e-10, it will set alpha to
        1e-10. If True, alpha will remain unchanged. This may cause
        numerical errors if alpha is too close to 0.

        .. versionadded:: 1.2
        .. versionchanged:: 1.4
           The default value of `force_alpha` changed to `True`.

    fit_prior : bool, default=True
        Only used in edge case with a single class in the training set.

    class_prior : array-like of shape (n_classes,), default=None
        Prior probabilities of the classes. Not used.

    norm : bool, default=False
        Whether or not a second normalization of the weights is performed. The
        default behavior mirrors the implementations found in Mahout and Weka,
        which do not follow the full algorithm described in Table 9 of the
        paper.

    Attributes
    ----------
    class_count_ : ndarray of shape (n_classes,)
        Number of samples encountered for each class during fitting. This
        value is weighted by the sample weight when provided.

    class_log_prior_ : ndarray of shape (n_classes,)
        Smoothed empirical log probability for each class. Only used in edge
        case with a single class in the training set.

    classes_ : ndarray of shape (n_classes,)
        Class labels known to the classifier

    feature_all_ : ndarray of shape (n_features,)
        Number of samples encountered for each feature during fitting. This
        value is weighted by the sample weight when provided.

    feature_count_ : ndarray of shape (n_classes, n_features)
        Number of samples encountered for each (class, feature) during fitting.
        This value is weighted by the sample weight when provided.

    feature_log_prob_ : ndarray of shape (n_classes, n_features)
        Empirical weights for class complements.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    BernoulliNB : Naive Bayes classifier for multivariate Bernoulli models.
    CategoricalNB : Naive Bayes classifier for categorical features.
    GaussianNB : Gaussian Naive Bayes.
    MultinomialNB : Naive Bayes classifier for multinomial models.

    References
    ----------
    Rennie, J. D., Shih, L., Teevan, J., & Karger, D. R. (2003).
    Tackling the poor assumptions of naive bayes text classifiers. In ICML
    (Vol. 3, pp. 616-623).
    https://people.csail.mit.edu/jrennie/papers/icml03-nb.pdf

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.RandomState(1)
    >>> X = rng.randint(5, size=(6, 100))
    >>> y = np.array([1, 2, 3, 4, 5, 6])
    >>> from sklearn.naive_bayes import ComplementNB
    >>> clf = ComplementNB()
    >>> clf.fit(X, y)
    ComplementNB()
    >>> print(clf.predict(X[2:3]))
    [3]
    """

    _parameter_constraints: dict = {
        **_BaseDiscreteNB._parameter_constraints,
        "norm": ["boolean"],
    }

    def __init__(
        self,
        *,
        alpha=1.0,
        force_alpha=True,
        fit_prior=True,
        class_prior=None,
        norm=False,
    ):
        # [FUNCTION DOC]
        # Forwards the shared hyper-parameters to the base class and stores norm.
        super().__init__(
            alpha=alpha,
            force_alpha=force_alpha,
            fit_prior=fit_prior,
            class_prior=class_prior,
        )
        self.norm = norm

    def __sklearn_tags__(self):
        # [FUNCTION DOC]
        # Declares that input must be non-negative.
        tags = super().__sklearn_tags__()
        tags.input_tags.positive_only = True
        return tags

    def _count(self, X, Y):
        # [FUNCTION DOC]
        # Like MultinomialNB._count and additionally sets feature_all_ to the per-
        # feature
        # total over classes.
        # Exceptions: ValueError on negative X. Side effects: mutates counts, sets
        # feature_all_.
        """Count feature occurrences."""
        check_non_negative(X, "ComplementNB (input X)")
        self.feature_count_ += safe_sparse_dot(Y.T, X)
        self.class_count_ += Y.sum(axis=0)
        self.feature_all_ = self.feature_count_.sum(axis=0)

    def _update_feature_log_prob(self, alpha):
        # [FUNCTION DOC]
        # Complement count per class is feature_all_ plus alpha minus the class's own
        # count;
        # take log of each row normalized to sum 1. If norm, divide each row by its sum
        # (second normalization); otherwise negate it, so that argmax in predict picks
        # the
        # class with smallest complement weight.
        # Side effects: sets feature_log_prob_.
        """Apply smoothing to raw counts and compute the weights."""
        comp_count = self.feature_all_ + alpha - self.feature_count_
        logged = np.log(comp_count / comp_count.sum(axis=1, keepdims=True))
        # _BaseNB.predict uses argmax, but ComplementNB operates with argmin.
        if self.norm:
            summed = logged.sum(axis=1, keepdims=True)
            feature_log_prob = logged / summed
        else:
            feature_log_prob = -logged
        self.feature_log_prob_ = feature_log_prob

    def _joint_log_likelihood(self, X):
        # [FUNCTION DOC]
        # Returns X times feature_log_prob_ transposed; the class log prior is added
        # only
        # when there is a single class.
        # Outputs: array (n_samples, n_classes) of class scores (not true log
        # likelihoods).
        """Calculate the class scores for the samples in X."""
        jll = safe_sparse_dot(X, self.feature_log_prob_.T)
        if len(self.classes_) == 1:
            jll += self.class_log_prior_
        return jll


class BernoulliNB(_BaseDiscreteNB):
    # [CLASS DOC]
    # Naive Bayes for binary/boolean features; optionally binarizes inputs with a
    # threshold before fitting and predicting. Penalizes absence of features explicitly.
    """Naive Bayes classifier for multivariate Bernoulli models.

    Like MultinomialNB, this classifier is suitable for discrete data. The
    difference is that while MultinomialNB works with occurrence counts,
    BernoulliNB is designed for binary/boolean features.

    Read more in the :ref:`User Guide <bernoulli_naive_bayes>`.

    Parameters
    ----------
    alpha : float or array-like of shape (n_features,), default=1.0
        Additive (Laplace/Lidstone) smoothing parameter
        (set alpha=0 and force_alpha=True, for no smoothing).

    force_alpha : bool, default=True
        If False and alpha is less than 1e-10, it will set alpha to
        1e-10. If True, alpha will remain unchanged. This may cause
        numerical errors if alpha is too close to 0.

        .. versionadded:: 1.2
        .. versionchanged:: 1.4
           The default value of `force_alpha` changed to `True`.

    binarize : float or None, default=0.0
        Threshold for binarizing (mapping to booleans) of sample features.
        If None, input is presumed to already consist of binary vectors.

    fit_prior : bool, default=True
        Whether to learn class prior probabilities or not.
        If false, a uniform prior will be used.

    class_prior : array-like of shape (n_classes,), default=None
        Prior probabilities of the classes. If specified, the priors are not
        adjusted according to the data.

    Attributes
    ----------
    class_count_ : ndarray of shape (n_classes,)
        Number of samples encountered for each class during fitting. This
        value is weighted by the sample weight when provided.

    class_log_prior_ : ndarray of shape (n_classes,)
        Log probability of each class (smoothed).

    classes_ : ndarray of shape (n_classes,)
        Class labels known to the classifier

    feature_count_ : ndarray of shape (n_classes, n_features)
        Number of samples encountered for each (class, feature)
        during fitting. This value is weighted by the sample weight when
        provided.

    feature_log_prob_ : ndarray of shape (n_classes, n_features)
        Empirical log probability of features given a class, P(x_i|y).

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    CategoricalNB : Naive Bayes classifier for categorical features.
    ComplementNB : The Complement Naive Bayes classifier
        described in Rennie et al. (2003).
    GaussianNB : Gaussian Naive Bayes (GaussianNB).
    MultinomialNB : Naive Bayes classifier for multinomial models.

    References
    ----------
    C.D. Manning, P. Raghavan and H. Schuetze (2008). Introduction to
    Information Retrieval. Cambridge University Press, pp. 234-265.
    https://nlp.stanford.edu/IR-book/html/htmledition/the-bernoulli-model-1.html

    A. McCallum and K. Nigam (1998). A comparison of event models for naive
    Bayes text classification. Proc. AAAI/ICML-98 Workshop on Learning for
    Text Categorization, pp. 41-48.

    V. Metsis, I. Androutsopoulos and G. Paliouras (2006). Spam filtering with
    naive Bayes -- Which naive Bayes? 3rd Conf. on Email and Anti-Spam (CEAS).

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.RandomState(1)
    >>> X = rng.randint(5, size=(6, 100))
    >>> Y = np.array([1, 2, 3, 4, 4, 5])
    >>> from sklearn.naive_bayes import BernoulliNB
    >>> clf = BernoulliNB()
    >>> clf.fit(X, Y)
    BernoulliNB()
    >>> print(clf.predict(X[2:3]))
    [3]
    """

    _parameter_constraints: dict = {
        **_BaseDiscreteNB._parameter_constraints,
        "binarize": [None, Interval(Real, 0, None, closed="left")],
    }

    def __init__(
        self,
        *,
        alpha=1.0,
        force_alpha=True,
        binarize=0.0,
        fit_prior=True,
        class_prior=None,
    ):
        # [FUNCTION DOC]
        # Forwards the shared hyper-parameters to the base class and stores binarize.
        super().__init__(
            alpha=alpha,
            fit_prior=fit_prior,
            class_prior=class_prior,
            force_alpha=force_alpha,
        )
        self.binarize = binarize

    def _check_X(self, X):
        # [FUNCTION DOC]
        # Base validation, then binarizes X with threshold binarize unless it is None.
        """Validate X, used only in predict* methods."""
        X = super()._check_X(X)
        if self.binarize is not None:
            X = binarize(X, threshold=self.binarize)
        return X

    def _check_X_y(self, X, y, reset=True):
        # [FUNCTION DOC]
        # Base validation of X and y, then binarizes X unless binarize is None.
        # Outputs: tuple (X, y).
        X, y = super()._check_X_y(X, y, reset=reset)
        if self.binarize is not None:
            X = binarize(X, threshold=self.binarize)
        return X, y

    def _count(self, X, Y):
        # [FUNCTION DOC]
        # Adds Y transposed times X to feature_count_ and column sums of Y to
        # class_count_.
        # Side effects: mutates counts.
        """Count and smooth feature occurrences."""
        self.feature_count_ += safe_sparse_dot(Y.T, X)
        self.class_count_ += Y.sum(axis=0)

    def _update_feature_log_prob(self, alpha):
        # [FUNCTION DOC]
        # Sets feature_log_prob_ to log of (feature count plus alpha) minus log of
        # (class count
        # plus 2*alpha), i.e. smoothed log P(feature = 1 | class).
        """Apply smoothing to raw counts and recompute log probabilities"""
        smoothed_fc = self.feature_count_ + alpha
        smoothed_cc = self.class_count_ + alpha * 2

        self.feature_log_prob_ = np.log(smoothed_fc) - np.log(
            smoothed_cc.reshape(-1, 1)
        )

    def _joint_log_likelihood(self, X):
        # [FUNCTION DOC]
        # Computes for each sample and class the sum over features of x*log p +
        # (1-x)*log(1-p), rewritten as X times (log p - log(1-p)) transposed plus the
        # per-class
        # sum of log(1-p), plus class_log_prior_.
        # Exceptions: ValueError if X's feature count differs from the fitted one.
        # Outputs: array (n_samples, n_classes).
        """Calculate the posterior log probability of the samples X"""
        n_features = self.feature_log_prob_.shape[1]
        n_features_X = X.shape[1]

        if n_features_X != n_features:
            raise ValueError(
                "Expected input with %d features, got %d instead"
                % (n_features, n_features_X)
            )

        neg_prob = np.log(1 - np.exp(self.feature_log_prob_))
        # Compute  neg_prob · (1 - X).T  as  ∑neg_prob - X · neg_prob
        jll = safe_sparse_dot(X, (self.feature_log_prob_ - neg_prob).T)
        jll += self.class_log_prior_ + neg_prob.sum(axis=1)

        return jll


class CategoricalNB(_BaseDiscreteNB):
    # [CLASS DOC]
    # Naive Bayes for categorical features encoded as non-negative integers 0..n-1 per
    # feature. Keeps per-feature count matrices category_count_ (n_classes,
    # n_categories_i) and per-feature log probability matrices.
    """Naive Bayes classifier for categorical features.

    The categorical Naive Bayes classifier is suitable for classification with
    discrete features that are categorically distributed. The categories of
    each feature are drawn from a categorical distribution.

    Read more in the :ref:`User Guide <categorical_naive_bayes>`.

    Parameters
    ----------
    alpha : float, default=1.0
        Additive (Laplace/Lidstone) smoothing parameter
        (set alpha=0 and force_alpha=True, for no smoothing).

    force_alpha : bool, default=True
        If False and alpha is less than 1e-10, it will set alpha to
        1e-10. If True, alpha will remain unchanged. This may cause
        numerical errors if alpha is too close to 0.

        .. versionadded:: 1.2
        .. versionchanged:: 1.4
           The default value of `force_alpha` changed to `True`.

    fit_prior : bool, default=True
        Whether to learn class prior probabilities or not.
        If false, a uniform prior will be used.

    class_prior : array-like of shape (n_classes,), default=None
        Prior probabilities of the classes. If specified, the priors are not
        adjusted according to the data.

    min_categories : int or array-like of shape (n_features,), default=None
        Minimum number of categories per feature.

        - integer: Sets the minimum number of categories per feature to
          `n_categories` for each features.
        - array-like: shape (n_features,) where `n_categories[i]` holds the
          minimum number of categories for the ith column of the input.
        - None (default): Determines the number of categories automatically
          from the training data.

        .. versionadded:: 0.24

    Attributes
    ----------
    category_count_ : list of arrays of shape (n_features,)
        Holds arrays of shape (n_classes, n_categories of respective feature)
        for each feature. Each array provides the number of samples
        encountered for each class and category of the specific feature.

    class_count_ : ndarray of shape (n_classes,)
        Number of samples encountered for each class during fitting. This
        value is weighted by the sample weight when provided.

    class_log_prior_ : ndarray of shape (n_classes,)
        Smoothed empirical log probability for each class.

    classes_ : ndarray of shape (n_classes,)
        Class labels known to the classifier

    feature_log_prob_ : list of arrays of shape (n_features,)
        Holds arrays of shape (n_classes, n_categories of respective feature)
        for each feature. Each array provides the empirical log probability
        of categories given the respective feature and class, ``P(x_i|y)``.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    n_categories_ : ndarray of shape (n_features,), dtype=np.int64
        Number of categories for each feature. This value is
        inferred from the data or set by the minimum number of categories.

        .. versionadded:: 0.24

    See Also
    --------
    BernoulliNB : Naive Bayes classifier for multivariate Bernoulli models.
    ComplementNB : Complement Naive Bayes classifier.
    GaussianNB : Gaussian Naive Bayes.
    MultinomialNB : Naive Bayes classifier for multinomial models.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.RandomState(1)
    >>> X = rng.randint(5, size=(6, 100))
    >>> y = np.array([1, 2, 3, 4, 5, 6])
    >>> from sklearn.naive_bayes import CategoricalNB
    >>> clf = CategoricalNB()
    >>> clf.fit(X, y)
    CategoricalNB()
    >>> print(clf.predict(X[2:3]))
    [3]
    """

    _parameter_constraints: dict = {
        **_BaseDiscreteNB._parameter_constraints,
        "min_categories": [
            None,
            "array-like",
            Interval(Integral, 1, None, closed="left"),
        ],
        "alpha": [Interval(Real, 0, None, closed="left")],
    }

    def __init__(
        self,
        *,
        alpha=1.0,
        force_alpha=True,
        fit_prior=True,
        class_prior=None,
        min_categories=None,
    ):
        # [FUNCTION DOC]
        # Forwards the shared hyper-parameters to the base class and stores
        # min_categories.
        super().__init__(
            alpha=alpha,
            force_alpha=force_alpha,
            fit_prior=fit_prior,
            class_prior=class_prior,
        )
        self.min_categories = min_categories

    def fit(self, X, y, sample_weight=None):
        # [FUNCTION DOC]
        # Delegates to _BaseDiscreteNB.fit (exists to provide the specific docstring).
        # Outputs: self.
        """Fit Naive Bayes classifier according to X, y.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vectors, where `n_samples` is the number of samples and
            `n_features` is the number of features. Here, each feature of X is
            assumed to be from a different categorical distribution.
            It is further assumed that all categories of each feature are
            represented by the numbers 0, ..., n - 1, where n refers to the
            total number of categories for the given feature. This can, for
            instance, be achieved with the help of OrdinalEncoder.

        y : array-like of shape (n_samples,)
            Target values.

        sample_weight : array-like of shape (n_samples,), default=None
            Weights applied to individual samples (1. for unweighted).

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        return super().fit(X, y, sample_weight=sample_weight)

    def partial_fit(self, X, y, classes=None, sample_weight=None):
        # [FUNCTION DOC]
        # Delegates to _BaseDiscreteNB.partial_fit (exists for the specific docstring).
        # Outputs: self.
        """Incremental fit on a batch of samples.

        This method is expected to be called several times consecutively
        on different chunks of a dataset so as to implement out-of-core
        or online learning.

        This is especially useful when the whole dataset is too big to fit in
        memory at once.

        This method has some performance overhead hence it is better to call
        partial_fit on chunks of data that are as large as possible
        (as long as fitting in the memory budget) to hide the overhead.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vectors, where `n_samples` is the number of samples and
            `n_features` is the number of features. Here, each feature of X is
            assumed to be from a different categorical distribution.
            It is further assumed that all categories of each feature are
            represented by the numbers 0, ..., n - 1, where n refers to the
            total number of categories for the given feature. This can, for
            instance, be achieved with the help of OrdinalEncoder.

        y : array-like of shape (n_samples,)
            Target values.

        classes : array-like of shape (n_classes,), default=None
            List of all the classes that can possibly appear in the y vector.

            Must be provided at the first call to partial_fit, can be omitted
            in subsequent calls.

        sample_weight : array-like of shape (n_samples,), default=None
            Weights applied to individual samples (1. for unweighted).

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        return super().partial_fit(X, y, classes, sample_weight=sample_weight)

    def __sklearn_tags__(self):
        # [FUNCTION DOC]
        # Declares categorical, dense-only, non-negative input.
        tags = super().__sklearn_tags__()
        tags.input_tags.categorical = True
        tags.input_tags.sparse = False
        tags.input_tags.positive_only = True
        return tags

    def _check_X(self, X):
        # [FUNCTION DOC]
        # Validates X for prediction as a finite dense integer array, no reset.
        # Exceptions: ValueError on sparse input, non-finite or negative values.
        """Validate X, used only in predict* methods."""
        X = validate_data(
            self,
            X,
            dtype="int",
            accept_sparse=False,
            ensure_all_finite=True,
            reset=False,
        )
        check_non_negative(X, "CategoricalNB (input X)")
        return X

    def _check_X_y(self, X, y, reset=True):
        # [FUNCTION DOC]
        # Validates X (dense, finite, integer) and y for fitting, then rejects negatives
        # in X.
        # Outputs: tuple (X, y).
        X, y = validate_data(
            self,
            X,
            y,
            dtype="int",
            accept_sparse=False,
            ensure_all_finite=True,
            reset=reset,
        )
        check_non_negative(X, "CategoricalNB (input X)")
        return X, y

    def _init_counters(self, n_classes, n_features):
        # [FUNCTION DOC]
        # HELPER: Sets class_count_ to zeros (n_classes,) and category_count_ to a list
        # with
        # one empty (n_classes, 0) array per feature; columns are grown during _count.
        self.class_count_ = np.zeros(n_classes, dtype=np.float64)
        self.category_count_ = [np.zeros((n_classes, 0)) for _ in range(n_features)]

    @staticmethod
    def _validate_n_categories(X, min_categories):
        # [FUNCTION DOC]
        # HELPER: Number of categories per feature is the column max of X plus 1; if
        # min_categories is given, take the element-wise maximum with it.
        # Inputs: X - int array (n_samples, n_features); min_categories - None, int or
        # array-like (n_features,).
        # Outputs: int array (n_features,).
        # Exceptions: ValueError if min_categories is not signed-integer typed or its
        # shape
        # does not broadcast to (n_features,) without changing shape.
        # rely on max for n_categories categories are encoded between 0...n-1
        n_categories_X = X.max(axis=0) + 1
        min_categories_ = np.array(min_categories)
        if min_categories is not None:
            if not np.issubdtype(min_categories_.dtype, np.signedinteger):
                raise ValueError(
                    "'min_categories' should have integral type. Got "
                    f"{min_categories_.dtype} instead."
                )
            n_categories_ = np.maximum(n_categories_X, min_categories_, dtype=np.int64)
            if n_categories_.shape != n_categories_X.shape:
                raise ValueError(
                    f"'min_categories' should have shape ({X.shape[1]},"
                    ") when an array-like is provided. Got"
                    f" {min_categories_.shape} instead."
                )
            return n_categories_
        else:
            return n_categories_X

    def _count(self, X, Y):
        # [FUNCTION DOC]
        # Adds Y's column sums to class_count_, recomputes n_categories_ from the
        # current batch,
        # and for each feature pads its count matrix with zero columns up to
        # n_categories_ and
        # adds per-class weighted category counts.
        # Side effects: mutates class_count_, category_count_, sets n_categories_.
        # KNOWN ISSUE: n_categories_ is recomputed from the current batch only, so after
        # a
        # partial_fit with a smaller max category it can be lower than the width of an
        # existing
        # category_count_ matrix (the matrix is never shrunk, so this is only cosmetic).
        def _update_cat_count_dims(cat_count, highest_feature):
            # [FUNCTION DOC]
            # HELPER: Pads cat_count with zero columns on the right so it has at least
            # highest_feature+1 columns. Outputs: the padded (or original) array.
            diff = highest_feature + 1 - cat_count.shape[1]
            if diff > 0:
                # we append a column full of zeros for each new category
                return np.pad(cat_count, [(0, 0), (0, diff)], "constant")
            return cat_count

        def _update_cat_count(X_feature, Y, cat_count, n_classes):
            # [FUNCTION DOC]
            # HELPER: For each class j, select samples with nonzero Y[:, j], bincount
            # their
            # category values (weighted by Y entries unless Y has int64 dtype), and add
            # the nonzero
            # counts into row j of cat_count. Side effects: mutates cat_count in place.
            for j in range(n_classes):
                mask = Y[:, j].astype(bool)
                if Y.dtype.type == np.int64:
                    weights = None
                else:
                    weights = Y[mask, j]
                counts = np.bincount(X_feature[mask], weights=weights)
                indices = np.nonzero(counts)[0]
                cat_count[j, indices] += counts[indices]

        self.class_count_ += Y.sum(axis=0)
        self.n_categories_ = self._validate_n_categories(X, self.min_categories)
        for i in range(self.n_features_in_):
            X_feature = X[:, i]
            self.category_count_[i] = _update_cat_count_dims(
                self.category_count_[i], self.n_categories_[i] - 1
            )
            _update_cat_count(
                X_feature, Y, self.category_count_[i], self.class_count_.shape[0]
            )

    def _update_feature_log_prob(self, alpha):
        # [FUNCTION DOC]
        # For each feature, add alpha to its category counts and take log of each entry
        # minus
        # log of its row sum. Side effects: sets feature_log_prob_ as a list of
        # (n_classes,
        # n_categories_i) arrays.
        feature_log_prob = []
        for i in range(self.n_features_in_):
            smoothed_cat_count = self.category_count_[i] + alpha
            smoothed_class_count = smoothed_cat_count.sum(axis=1)
            feature_log_prob.append(
                np.log(smoothed_cat_count) - np.log(smoothed_class_count.reshape(-1, 1))
            )
        self.feature_log_prob_ = feature_log_prob

    def _joint_log_likelihood(self, X):
        # [FUNCTION DOC]
        # Checks feature count, then for each feature adds the log probability of each
        # sample's observed category (for every class) and finally adds
        # class_log_prior_.
        # Outputs: array (n_samples, n_classes).
        # Exceptions: ValueError on feature mismatch; IndexError if a category exceeds
        # those
        # seen during fit.
        _check_n_features(self, X, reset=False)
        jll = np.zeros((X.shape[0], self.class_count_.shape[0]))
        for i in range(self.n_features_in_):
            indices = X[:, i]
            jll += self.feature_log_prob_[i][:, indices].T
        total_ll = jll + self.class_log_prior_
        return total_ll
