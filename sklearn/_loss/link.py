# [MODULE DOC]
# Defines the Interval dataclass (value-range description with open/closed ends)
# and invertible, differentiable link functions used by the loss classes in
# sklearn._loss.loss: identity, log, logit, half-logit and symmetric multinomial
# logit. _LINKS maps short string names to link classes. Most links dispatch via
# get_namespace so they work with Array API inputs.
"""
Module contains classes for invertible (and differentiable) link functions.
"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

from abc import ABC, abstractmethod
from dataclasses import dataclass
from math import ulp

from scipy.stats import gmean

from sklearn.utils._array_api import _expit, _logit, get_namespace
from sklearn.utils.extmath import softmax


@dataclass
class Interval:
    # [CLASS DOC]
    # Dataclass describing a real interval [low, high] where each end may be inclusive
    # or exclusive (fields low, high: float; low_inclusive, high_inclusive: bool).
    # Used to describe the valid domains of y_true / y_pred for losses and links.
    low: float
    high: float
    low_inclusive: bool
    high_inclusive: bool

    def __post_init__(self):
        # [FUNCTION DOC]
        # Dataclass post-init validation.
        # Exceptions: ValueError if low > high. Equal bounds are allowed.
        """Check that low <= high"""
        if self.low > self.high:
            raise ValueError(
                f"One must have low <= high; got low={self.low}, high={self.high}."
            )

    def includes(self, x):
        # [FUNCTION DOC]
        # Checks that every element of x lies in the interval, respecting inclusiveness of
        # each end; short-circuits after the lower-bound check fails.
        # Inputs: x - array (any shape, any Array API namespace).
        # Outputs: Python bool. NaN values make the result False.
        """Test whether all values of x are in interval range.

        Parameters
        ----------
        x : ndarray
            Array whose elements are tested to be in interval range.

        Returns
        -------
        result : bool
        """
        xp, _ = get_namespace(x)
        if self.low_inclusive:
            low = xp.greater_equal(x, self.low)
        else:
            low = xp.greater(x, self.low)

        if not xp.all(low):
            return False

        if self.high_inclusive:
            high = xp.less_equal(x, self.high)
        else:
            high = xp.less(x, self.high)

        # Note: np.all returns numpy.bool_
        return bool(xp.all(high))


def _inclusive_low_high(interval):
    # [FUNCTION DOC]
    # HELPER: Returns two floats strictly inside the interval, used in tests to sample
    # values. Infinite ends are replaced by -1e10 / 1e10; finite ends are nudged inward
    # by a relative and absolute step eps = 10 ulp(1), with the sign of the end
    # deciding the direction of the relative step.
    # Inputs: interval - Interval. Outputs: tuple (low, high) of floats.
    # KNOWN ISSUE: for finite ends the result is strictly inside even when the end is
    # inclusive, so the name "inclusive" is somewhat misleading.
    """Generate values low and high to be within the interval range.

    This is used in tests only.

    Returns
    -------
    low, high : tuple of floats
        The returned values low and high lie within the interval.
    """
    eps = 10 * ulp(1)
    if interval.low == -float("inf"):
        low = -1e10
    elif interval.low < 0:
        low = interval.low * (1 - eps) + eps
    else:
        low = interval.low * (1 + eps) + eps

    if interval.high == float("inf"):
        high = 1e10
    elif interval.high < 0:
        high = interval.high * (1 + eps) - eps
    else:
        high = interval.high * (1 - eps) - eps

    return float(low), float(high)


class BaseLink(ABC):
    # [CLASS DOC]
    # Abstract base for link functions with the convention raw_prediction = link(y_pred)
    # and y_pred = inverse(raw_prediction). Class attributes: is_multiclass (False,
    # used in tests) and interval_y_pred (default the open real line), the domain of
    # y_pred. Subclasses must implement link and inverse.
    """Abstract base class for differentiable, invertible link functions.

    Convention:
        - link function g: raw_prediction = g(y_pred)
        - inverse link h: y_pred = h(raw_prediction)

    For (generalized) linear models, `raw_prediction = X @ coef` is the so
    called linear predictor, and `y_pred = h(raw_prediction)` is the predicted
    conditional (on X) expected value of the target `y_true`.

    The methods are not implemented as staticmethods in case a link function needs
    parameters.
    """

    is_multiclass = False  # used for testing only

    # Usually, raw_prediction may be any real number and y_pred is an open
    # interval.
    # interval_raw_prediction = Interval(-float("inf"), float("inf"), False, False)
    interval_y_pred = Interval(-float("inf"), float("inf"), False, False)

    @abstractmethod
    def link(self, y_pred):
        # [FUNCTION DOC]
        # Abstract: element-wise link g applied to y_pred; returns an array of the same
        # shape. No implementation (body is the docstring only).
        """Compute the link function g(y_pred).

        The link function maps (predicted) target values to raw predictions,
        i.e. `g(y_pred) = raw_prediction`.

        Parameters
        ----------
        y_pred : array
            Predicted target values.

        Returns
        -------
        array
            Output array, element-wise link function.
        """

    @abstractmethod
    def inverse(self, raw_prediction):
        # [FUNCTION DOC]
        # Abstract: element-wise inverse link h applied to raw_prediction; returns an array
        # of the same shape. No implementation.
        """Compute the inverse link function h(raw_prediction).

        The inverse link function maps raw predictions to predicted target
        values, i.e. `h(raw_prediction) = y_pred`.

        Parameters
        ----------
        raw_prediction : array
            Raw prediction values (in link space).

        Returns
        -------
        array
            Output array, element-wise inverse link function.
        """


class IdentityLink(BaseLink):
    # [CLASS DOC]
    # Identity link: g(x) = x; inverse is the same function (class attribute alias).
    # y_pred domain is the whole real line.
    """The identity link function g(x)=x."""

    def link(self, y_pred):
        # [FUNCTION DOC]
        # Returns y_pred itself (no copy; the caller receives the same object). Also used
        # as inverse.
        return y_pred  # TODO: Should we copy?

    inverse = link


class LogLink(BaseLink):
    # [CLASS DOC]
    # Log link g(x) = log(x), inverse exp; y_pred domain is the open interval (0, inf).
    """The log link function g(x)=log(x)."""

    interval_y_pred = Interval(0, float("inf"), False, False)

    def link(self, y_pred):
        # [FUNCTION DOC]
        # Element-wise natural logarithm of y_pred in its array namespace. Values <= 0
        # produce -inf/NaN (with numpy warnings), not exceptions.
        xp, _ = get_namespace(y_pred)
        return xp.log(y_pred)

    def inverse(self, raw_prediction):
        # [FUNCTION DOC]
        # Element-wise exponential of raw_prediction in its array namespace; may overflow
        # to inf for large inputs.
        xp, _ = get_namespace(raw_prediction)
        return xp.exp(raw_prediction)


class LogitLink(BaseLink):
    # [CLASS DOC]
    # Logit link g(x) = log(x / (1 - x)), inverse the logistic sigmoid (expit);
    # y_pred domain is (0, 1).
    """The logit link function g(x)=logit(x)."""

    interval_y_pred = Interval(0, 1, False, False)

    def link(self, y_pred):
        # [FUNCTION DOC]
        # Element-wise logit of y_pred via the namespace-aware _logit helper.
        return _logit(y_pred)

    def inverse(self, raw_prediction):
        # [FUNCTION DOC]
        # Element-wise logistic sigmoid of raw_prediction via the namespace-aware _expit.
        return _expit(raw_prediction)


class HalfLogitLink(BaseLink):
    # [CLASS DOC]
    # Half-logit link g(x) = logit(x) / 2 (exponential loss), inverse expit(2 * raw);
    # y_pred domain is (0, 1).
    """Half the logit link function g(x)=1/2 * logit(x).

    Used for the exponential loss.
    """

    interval_y_pred = Interval(0, 1, False, False)

    def link(self, y_pred):
        # [FUNCTION DOC]
        # Returns half the element-wise logit of y_pred.
        return 0.5 * _logit(y_pred)

    def inverse(self, raw_prediction):
        # [FUNCTION DOC]
        # Returns expit of twice raw_prediction, element-wise.
        return _expit(2 * raw_prediction)


class MultinomialLogit(BaseLink):
    # [CLASS DOC]
    # Symmetric multinomial logit link for arrays of shape (n_samples, n_classes):
    # link is log of probabilities divided by their per-row geometric mean (so each row
    # of raw predictions sums to zero), inverse is the row-wise softmax.
    # is_multiclass is True; y_pred domain per entry is (0, 1).
    """The symmetric multinomial logit function.

    Convention:
        - y_pred.shape = raw_prediction.shape = (n_samples, n_classes)

    Notes:
        - The inverse link h is the softmax function.
        - The sum is over the second axis, i.e. axis=1 (n_classes).

    We have to choose additional constraints in order to make

        y_pred[k] = exp(raw_pred[k]) / sum(exp(raw_pred[k]), k=0..n_classes-1)

    for n_classes classes identifiable and invertible.
    We choose the symmetric side constraint where the geometric mean response
    is set as reference category, see [2]:

    The symmetric multinomial logit link function for a single data point is
    then defined as

        raw_prediction[k] = g(y_pred[k]) = log(y_pred[k]/gmean(y_pred))
        = log(y_pred[k]) - mean(log(y_pred)).

    Note that this is equivalent to the definition in [1] and implies mean
    centered raw predictions:

        sum(raw_prediction[k], k=0..n_classes-1) = 0.

    For linear models with raw_prediction = X @ coef, this corresponds to
    sum(coef[k], k=0..n_classes-1) = 0, i.e. the sum over classes for every
    feature is zero.

    Reference
    ---------
    .. [1] Friedman, Jerome; Hastie, Trevor; Tibshirani, Robert. "Additive
        logistic regression: a statistical view of boosting" Ann. Statist.
        28 (2000), no. 2, 337--407. doi:10.1214/aos/1016218223.
        https://projecteuclid.org/euclid.aos/1016218223

    .. [2] Zahid, Faisal Maqbool and Gerhard Tutz. "Ridge estimation for
        multinomial logit models with symmetric side constraints."
        Computational Statistics 28 (2013): 1017-1034.
        http://epub.ub.uni-muenchen.de/11001/1/tr067.pdf
    """

    is_multiclass = True
    interval_y_pred = Interval(0, 1, False, False)

    def symmetrize_raw_prediction(self, raw_prediction):
        # [FUNCTION DOC]
        # Centers each row of raw_prediction (n_samples, n_classes) by subtracting its mean
        # over classes, giving the symmetric (zero-sum) parametrisation; returns a new array.
        xp, _ = get_namespace(raw_prediction)
        return raw_prediction - xp.mean(raw_prediction, axis=1)[:, None]

    def link(self, y_pred):
        # [FUNCTION DOC]
        # Computes, for each row, the geometric mean over classes with scipy's gmean and
        # returns the element-wise log of y_pred divided by that row mean.
        # Inputs: y_pred - (n_samples, n_classes) positive probabilities.
        # Outputs: array of the same shape with rows summing to ~0.
        # KNOWN ISSUE: scipy.stats.gmean is NumPy based, so this path is not truly Array
        # API compliant despite the get_namespace call.
        xp, _ = get_namespace(y_pred)
        # geometric mean as reference category
        gm = gmean(y_pred, axis=1)
        return xp.log(y_pred / gm[:, None])

    def inverse(self, raw_prediction):
        # [FUNCTION DOC]
        # Row-wise softmax of raw_prediction (n_samples, n_classes) via
        # sklearn.utils.extmath.softmax; returns probabilities with rows summing to 1.
        return softmax(raw_prediction)


_LINKS = {
    "identity": IdentityLink,
    "log": LogLink,
    "logit": LogitLink,
    "half_logit": HalfLogitLink,
    "multinomial_logit": MultinomialLogit,
}
