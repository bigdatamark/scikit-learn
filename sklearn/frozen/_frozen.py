# [MODULE DOC]
# Implements FrozenEstimator, a meta-estimator that wraps an already fitted estimator so
# that fitting it (directly, via clone, or inside a pipeline) leaves it unchanged, while
# delegating all other attribute access to the wrapped estimator.
# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

from copy import deepcopy

from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError
from sklearn.utils import get_tags
from sklearn.utils.metaestimators import available_if
from sklearn.utils.validation import check_is_fitted


def _estimator_has(attr):
    # [FUNCTION DOC]
    # HELPER: Builds a predicate for available_if that checks the wrapped estimator
    #   exposes
    # a given attribute.
    # Inputs: attr - attribute name (str).
    # Outputs: a function taking self that returns True if self.estimator has attr.
    # Exceptions: the returned check lets AttributeError from getattr propagate, which
    #   makes
    # available_if hide the decorated method.
    """Check that final_estimator has `attr`.

    Used together with `available_if`.
    """

    def check(self):
        # [FUNCTION DOC]
        # HELPER: Looks up attr on self.estimator (result discarded) and returns True.
        # Inputs: self - a FrozenEstimator. Outputs: True.
        # Exceptions: AttributeError if the wrapped estimator lacks attr.
        # raise original `AttributeError` if `attr` does not exist
        getattr(self.estimator, attr)
        return True

    return check


class FrozenEstimator(BaseEstimator):
    # [CLASS DOC]
    # Wrapper storing a single parameter, estimator. fit is a no-op (only checks the
    #   inner
    # estimator is fitted), clone returns the same object, fit_predict/fit_transform are
    # blocked, and any other attribute is forwarded through __getattr__.
    # Tags are copied from the inner estimator with _skip_test set, so common estimator
    # checks are skipped.
    """Estimator that wraps a fitted estimator to prevent re-fitting.

    This meta-estimator takes an estimator and freezes it, in the sense that calling
    `fit` on it has no effect. `fit_predict` and `fit_transform` are also disabled.
    All other methods are delegated to the original estimator and original estimator's
    attributes are accessible as well.

    This is particularly useful when you have a fitted or a pre-trained model as a
    transformer in a pipeline, and you'd like `pipeline.fit` to have no effect on this
    step.

    Parameters
    ----------
    estimator : estimator
        The estimator which is to be kept frozen.

    See Also
    --------
    None: No similar entry in the scikit-learn documentation.

    Examples
    --------
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.frozen import FrozenEstimator
    >>> from sklearn.linear_model import LogisticRegression
    >>> X, y = make_classification(random_state=0)
    >>> clf = LogisticRegression(random_state=0).fit(X, y)
    >>> frozen_clf = FrozenEstimator(clf)
    >>> frozen_clf.fit(X, y)  # No-op
    FrozenEstimator(estimator=LogisticRegression(random_state=0))
    >>> frozen_clf.predict(X)  # Predictions from `clf.predict`
    array(...)
    """

    def __init__(self, estimator):
        # [FUNCTION DOC]
        # Stores the given estimator unchanged on self.estimator (no validation, no
        #   copy).
        # Inputs: estimator - any estimator object, expected to be fitted already.
        self.estimator = estimator

    @available_if(_estimator_has("__getitem__"))
    def __getitem__(self, *args, **kwargs):
        # [FUNCTION DOC]
        # Forwards indexing (e.g. pipeline slicing / step access) to the inner
        #   estimator.
        # Only available when the inner estimator defines __getitem__ (via
        #   available_if).
        # Inputs: arbitrary positional/keyword args passed through. Outputs: whatever
        #   the inner
        # __getitem__ returns.
        """__getitem__ is defined in :class:`~sklearn.pipeline.Pipeline` and \
            :class:`~sklearn.compose.ColumnTransformer`.
        """
        return self.estimator.__getitem__(*args, **kwargs)

    def __getattr__(self, name):
        # [FUNCTION DOC]
        # Fallback attribute lookup (only invoked when normal lookup fails): forwards to
        #   the
        # inner estimator.
        # Inputs: name - attribute name (str).
        # Outputs: the inner estimator's attribute value.
        # Exceptions: AttributeError for "fit_predict" or "fit_transform" (explicitly
        #   blocked),
        # or when the inner estimator lacks the attribute.
        # `estimator`'s attributes are now accessible except `fit_predict` and
        # `fit_transform`
        if name in ["fit_predict", "fit_transform"]:
            raise AttributeError(f"{name} is not available for frozen estimators.")
        return getattr(self.estimator, name)

    def __sklearn_clone__(self):
        # [FUNCTION DOC]
        # Customises sklearn.base.clone so cloning returns this very instance (no copy),
        #   keeping
        # the fitted state of the inner estimator.
        # Outputs: self.
        return self

    def __sklearn_is_fitted__(self):
        # [FUNCTION DOC]
        # Reports fitted status by delegating to check_is_fitted on the inner estimator.
        # Outputs: True if the inner estimator is fitted, False if NotFittedError was
        #   raised.
        try:
            check_is_fitted(self.estimator)
            return True
        except NotFittedError:
            return False

    def fit(self, X, y, *args, **kwargs):
        # [FUNCTION DOC]
        # No-op fit: verifies the inner estimator is fitted and returns self; all
        #   arguments are
        # ignored.
        # Inputs: X, y, *args, **kwargs - ignored.
        # Outputs: self.
        # Exceptions: NotFittedError if the inner estimator is not fitted.
        # Side effects: none.
        """No-op.

        As a frozen estimator, calling `fit` has no effect.

        Parameters
        ----------
        X : object
            Ignored.

        y : object
            Ignored.

        *args : tuple
            Additional positional arguments. Ignored, but present for API compatibility
            with `self.estimator`.

        **kwargs : dict
            Additional keyword arguments. Ignored, but present for API compatibility
            with `self.estimator`.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        check_is_fitted(self.estimator)
        return self

    def set_params(self, **kwargs):
        # [FUNCTION DOC]
        # Allows replacing only the estimator parameter; any other key is rejected.
        # Inputs: **kwargs - may contain "estimator"; other keys trigger an error.
        # Outputs: None.
        # Exceptions: ValueError if any key other than "estimator" is passed (raised
        #   after the
        # estimator replacement, if any, has already been applied).
        # Side effects: replaces self.estimator when a non-None estimator is given;
        #   passing
        # estimator=None is silently ignored.
        # KNOWN ISSUE: the docstring says it returns self, but the method returns None,
        #   so
        # chained calls such as set_params(...).fit(...) fail.
        # KNOWN ISSUE: on ValueError, a provided estimator has already been assigned,
        #   leaving the
        # object partially updated.
        """Set the parameters of this estimator.

        The only valid key here is `estimator`. You cannot set the parameters of the
        inner estimator.

        Parameters
        ----------
        **kwargs : dict
            Estimator parameters.

        Returns
        -------
        self : FrozenEstimator
            This estimator.
        """
        estimator = kwargs.pop("estimator", None)
        if estimator is not None:
            self.estimator = estimator
        if kwargs:
            raise ValueError(
                "You cannot set parameters of the inner estimator in a frozen "
                "estimator since calling `fit` has no effect. You can use "
                "`frozenestimator.estimator.set_params` to set parameters of the inner "
                "estimator."
            )

    def get_params(self, deep=True):
        # [FUNCTION DOC]
        # Returns only the wrapper's own parameter, never the inner estimator's
        #   parameters.
        # Inputs: deep - ignored. Outputs: dict with the single key "estimator".
        """Get parameters for this estimator.

        Returns a `{"estimator": estimator}` dict. The parameters of the inner
        estimator are not included.

        Parameters
        ----------
        deep : bool, default=True
            Ignored.

        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        return {"estimator": self.estimator}

    def __sklearn_tags__(self):
        # [FUNCTION DOC]
        # Returns a deep copy of the inner estimator's tags with _skip_test set to True.
        # Outputs: Tags object. Side effects: none (the inner tags are copied, not
        #   mutated).
        tags = deepcopy(get_tags(self.estimator))
        tags._skip_test = True
        return tags
