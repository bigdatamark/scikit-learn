# [MODULE DOC]
# Gallery script demonstrating a custom estimator that implements __sklearn_is_fitted__
#   so check_is_fitted uses it instead of looking for trailing-underscore attributes.
# Only defines the class; nothing is executed at import.
"""
========================================
`__sklearn_is_fitted__` as Developer API
========================================

The `__sklearn_is_fitted__` method is a convention used in scikit-learn for
checking whether an estimator object has been fitted or not. This method is
typically implemented in custom estimator classes that are built on top of
scikit-learn's base classes like `BaseEstimator` or its subclasses.

Developers should use :func:`~sklearn.utils.validation.check_is_fitted`
at the beginning of all methods except `fit`. If they need to customize or
speed-up the check, they can implement the `__sklearn_is_fitted__` method as
shown below.

In this example the custom estimator showcases the usage of the
`__sklearn_is_fitted__` method and the `check_is_fitted` utility function
as developer APIs. The `__sklearn_is_fitted__` method checks fitted status
by verifying the presence of the `_is_fitted` attribute.
"""

# %%
# An example custom estimator implementing a simple classifier
# ------------------------------------------------------------
# This code snippet defines a custom estimator class called `CustomEstimator`
# that extends both the `BaseEstimator` and `ClassifierMixin` classes from
# scikit-learn and showcases the usage of the `__sklearn_is_fitted__` method
# and the `check_is_fitted` utility function.

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted


class CustomEstimator(BaseEstimator, ClassifierMixin):
    # [CLASS DOC]
    # Toy classifier: stores sorted classes on fit, predicts the first class for every
    #   sample, and returns a constant score. Fitted state is flagged via the private
    #   _is_fitted attribute.
    def __init__(self, parameter=1):
        # [FUNCTION DOC]
        # Stores parameter (unused by the model; default 1).
        self.parameter = parameter

    def fit(self, X, y):
        # [FUNCTION DOC]
        # Records classes_ as the sorted list of unique labels in y and sets _is_fitted
        #   to True; X is ignored.
        # Outputs: self. Side effects: sets classes_ and _is_fitted.
        """
        Fit the estimator to the training data.
        """
        self.classes_ = sorted(set(y))
        # Custom attribute to track if the estimator is fitted
        self._is_fitted = True
        return self

    def predict(self, X):
        # [FUNCTION DOC]
        # Returns a list of length len(X) filled with classes_[0] (smallest class).
        # Exceptions: NotFittedError when not fitted.
        """
        Perform Predictions

        If the estimator is not fitted, then raise NotFittedError
        """
        check_is_fitted(self)
        # Perform prediction logic
        predictions = [self.classes_[0]] * len(X)
        return predictions

    def score(self, X, y):
        # [FUNCTION DOC]
        # Returns the constant 0.5 regardless of X and y.
        # Exceptions: NotFittedError when not fitted.
        # KNOWN ISSUE: overrides ClassifierMixin.score with a placeholder rather than
        #   computing accuracy.
        """
        Calculate Score

        If the estimator is not fitted, then raise NotFittedError
        """
        check_is_fitted(self)
        # Perform scoring logic
        return 0.5

    def __sklearn_is_fitted__(self):
        # [FUNCTION DOC]
        # Returns True only if the _is_fitted attribute exists and is truthy; used by
        #   check_is_fitted.
        """
        Check fitted status and return a Boolean value.
        """
        return hasattr(self, "_is_fitted") and self._is_fitted
