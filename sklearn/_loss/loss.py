# [MODULE DOC]
# Private loss classes used by linear models, GLMs, SGD and (Hist)GradientBoosting.
# Each loss composes a Cython CyLossFunction (closs) doing the element-wise,
# optionally OpenMP-parallel loss/gradient/hessian computations, with a link object
# from sklearn._loss.link mapping raw predictions to predictions. BaseLoss wraps
# the Cython calls (allocation of outputs, squeezing (n, 1) shapes). Concrete
# classes set domains (interval_y_true / interval_y_pred), hessian flags, and
# override fit_intercept_only, constant_to_optimal_zero and predict_proba where
# needed. *ArrayAPI variants reimplement computations with Array API namespaces in
# pure Python. _LOSSES maps string names to loss classes.
"""
This module contains loss classes suitable for fitting.

It is not part of the public API.
Specific losses are used for regression, binary classification or multiclass
classification.
"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

# Goals:
# - Provide a common private module for loss functions/classes.
# - To be used in:
#   - LogisticRegression
#   - PoissonRegressor, GammaRegressor, TweedieRegressor
#   - HistGradientBoostingRegressor, HistGradientBoostingClassifier
#   - GradientBoostingRegressor, GradientBoostingClassifier
#   - SGDRegressor, SGDClassifier
# - Replace link module of GLMs.

import numbers

import numpy as np
from scipy.special import xlogy

from sklearn._loss._loss import (
    CyAbsoluteError,
    CyExponentialLoss,
    CyHalfBinomialLoss,
    CyHalfGammaLoss,
    CyHalfMultinomialLoss,
    CyHalfPoissonLoss,
    CyHalfSquaredError,
    CyHalfTweedieLoss,
    CyHalfTweedieLossIdentity,
    CyHuberLoss,
    CyPinballLoss,
)
from sklearn._loss.link import (
    HalfLogitLink,
    IdentityLink,
    Interval,
    LogitLink,
    LogLink,
    MultinomialLogit,
)
from sklearn.externals.array_api_extra import one_hot
from sklearn.utils import check_scalar
from sklearn.utils._array_api import (
    _average,
    _logsumexp,
    _ravel,
)
from sklearn.utils.extmath import softmax
from sklearn.utils.stats import _weighted_percentile


# Note: The shape of raw_prediction for multiclass classifications are
# - GradientBoostingClassifier: (n_samples, n_classes)
# - HistGradientBoostingClassifier: (n_classes, n_samples)
#
# Note: Instead of inheritance like
#
#    class BaseLoss(BaseLink, CyLossFunction):
#    ...
#
#    # Note: Naturally, we would inherit in the following order
#    #     class HalfSquaredError(IdentityLink, CyHalfSquaredError, BaseLoss)
#    #   But because of https://github.com/cython/cython/issues/4350 we set BaseLoss as
#    #   the last one. This, of course, changes the MRO.
#    class HalfSquaredError(IdentityLink, CyHalfSquaredError, BaseLoss):
#
# we use composition. This way we improve maintainability by avoiding the above
# mentioned Cython edge case and have easier to understand code (which method calls
# which code).
class BaseLoss:
    # [CLASS DOC]
    # Base class composing a Cython loss (closs) and a link. Class attributes:
    # differentiable (True) and is_multiclass (False). Instance attributes set in
    # __init__: closs, link, n_classes, xp, device, approx_hessian=False,
    # constant_hessian=False, interval_y_true = open real line, interval_y_pred taken
    # from the link. Output-array arguments (*_out) are filled in place when provided.
    """Base class for a loss function of 1-dimensional targets.

    Conventions:

        - y_true.shape = sample_weight.shape = (n_samples,)
        - y_pred.shape = raw_prediction.shape = (n_samples,)
        - If is_multiclass is true (multiclass classification), then
          y_pred.shape = raw_prediction.shape = (n_samples, n_classes)
          Note that this corresponds to the return value of decision_function.

    y_true, y_pred, sample_weight and raw_prediction must either be all float64
    or all float32.
    gradient and hessian must be either both float64 or both float32.

    Note that y_pred = link.inverse(raw_prediction).

    Specific loss classes can inherit specific link classes to satisfy
    BaseLink's abstractmethods.

    Parameters
    ----------
    closs: CyLossFunction
        For example, a CyLossFunction; hence the name "c"loss.
    link : BaseLink
    sample_weight : {None, ndarray}
        If sample_weight is None, the hessian might be constant.
    n_classes : {None, int}
        The number of classes for classification, else None.
    xp : module, default=None
        Array namespace module.
    device : device, default=None
        A device object (see the "Device Support" section of the array API spec).

    Attributes
    ----------
    closs: CyLossFunction
        For example, a CyLossFunction; hence the name "c"loss.
    link : BaseLink
    n_classes : {None, int}
        The number of classes for classification, else None.
    xp : module or None
        Array namespace module. Ignored by the Cython implementation.
    device : device or None
        A device object. Ignored by the Cython implementation.
    interval_y_true : Interval
        Valid interval for y_true
    interval_y_pred : Interval
        Valid Interval for y_pred
    differentiable : bool
        Indicates whether or not loss function is differentiable in
        raw_prediction everywhere.
    approx_hessian : bool
        Indicates whether the hessian is approximated or exact. If,
        approximated, it should be larger or equal to the exact one.
    constant_hessian : bool
        Indicates whether the hessian is one for this loss.
    is_multiclass : bool
        Indicates whether n_classes > 2 is allowed.
    """

    # For gradient boosted decision trees:
    # If differentiable = False for a loss, the leaves values are required to
    # be updated once the tree has been trained. The trees are trained to
    # predict a Newton-Raphson step (see grower._finalize_leaf()). But for
    # some losses (e.g. least absolute deviation) we need to adjust the tree
    # values to account for the "line search" of the gradient descent
    # procedure. See the original paper Greedy Function Approximation: A
    # Gradient Boosting Machine by Friedman
    # (https://statweb.stanford.edu/~jhf/ftp/trebst.pdf) for the theory.
    differentiable = True
    is_multiclass = False

    def __init__(self, closs, link, n_classes=None, xp=None, device=None):
        # [FUNCTION DOC]
        # Stores the given closs, link, n_classes, xp and device and initialises the flags
        # and intervals listed in the class doc. No validation.
        self.closs = closs
        self.link = link
        self.n_classes = n_classes
        self.xp = xp  # simplifies array API versions
        self.device = device  # simplifies array API versions
        self.approx_hessian = False
        self.constant_hessian = False
        self.interval_y_true = Interval(-np.inf, np.inf, False, False)
        self.interval_y_pred = self.link.interval_y_pred

    def in_y_true_range(self, y):
        # [FUNCTION DOC]
        # Returns True if all values of y lie in interval_y_true (see Interval.includes).
        """Return True if y is in the valid range of y_true.

        Parameters
        ----------
        y : ndarray
        """
        return self.interval_y_true.includes(y)

    def in_y_pred_range(self, y):
        # [FUNCTION DOC]
        # Returns True if all values of y lie in interval_y_pred.
        """Return True if y is in the valid range of y_pred.

        Parameters
        ----------
        y : ndarray
        """
        return self.interval_y_pred.includes(y)

    def loss(
        self,
        y_true,
        raw_prediction,
        sample_weight=None,
        loss_out=None,
        n_threads=1,
    ):
        # [FUNCTION DOC]
        # Allocates loss_out like y_true if missing, squeezes a (n_samples, 1)
        # raw_prediction to 1d, then calls closs.loss which writes the (optionally
        # sample-weighted) per-sample losses into loss_out.
        # Outputs: loss_out (n_samples,). Side effects: writes into the provided loss_out.
        # Exceptions: errors from the Cython layer on dtype/shape/contiguity mismatch.
        """Compute the pointwise loss value for each input.

        Parameters
        ----------
        y_true : C-contiguous array of shape (n_samples,)
            Observed, true target values.
        raw_prediction : C-contiguous array of shape (n_samples,) or array of \
            shape (n_samples, n_classes)
            Raw prediction values (in link space).
        sample_weight : None or C-contiguous array of shape (n_samples,)
            Sample weights.
        loss_out : None or C-contiguous array of shape (n_samples,)
            A location into which the result is stored. If None, a new array
            might be created.
        n_threads : int, default=1
            Might use openmp thread parallelism.

        Returns
        -------
        loss : array of shape (n_samples,)
            Element-wise loss function.
        """
        if loss_out is None:
            loss_out = np.empty_like(y_true)
        # Be graceful to shape (n_samples, 1) -> (n_samples,)
        if raw_prediction.ndim == 2 and raw_prediction.shape[1] == 1:
            raw_prediction = raw_prediction.squeeze(1)

        self.closs.loss(
            y_true=y_true,
            raw_prediction=raw_prediction,
            sample_weight=sample_weight,
            loss_out=loss_out,
            n_threads=n_threads,
        )
        return loss_out

    def loss_gradient(
        self,
        y_true,
        raw_prediction,
        sample_weight=None,
        loss_out=None,
        gradient_out=None,
        n_threads=1,
    ):
        # [FUNCTION DOC]
        # Allocates missing outputs (loss like y_true and gradient like raw_prediction; when
        # only one output is given, the other is allocated with its dtype), squeezes
        # (n_samples, 1) raw_prediction and gradient_out to 1d, then calls
        # closs.loss_gradient to fill both.
        # Outputs: tuple (loss_out, gradient_out). Side effects: writes provided buffers.
        # Note: the returned gradient may be a squeezed view of the passed gradient_out.
        """Compute loss and gradient w.r.t. raw_prediction for each input.

        Parameters
        ----------
        y_true : C-contiguous array of shape (n_samples,)
            Observed, true target values.
        raw_prediction : C-contiguous array of shape (n_samples,) or array of \
            shape (n_samples, n_classes)
            Raw prediction values (in link space).
        sample_weight : None or C-contiguous array of shape (n_samples,)
            Sample weights.
        loss_out : None or C-contiguous array of shape (n_samples,)
            A location into which the loss is stored. If None, a new array
            might be created.
        gradient_out : None or C-contiguous array of shape (n_samples,) or array \
            of shape (n_samples, n_classes)
            A location into which the gradient is stored. If None, a new array
            might be created.
        n_threads : int, default=1
            Might use openmp thread parallelism.

        Returns
        -------
        loss : array of shape (n_samples,)
            Element-wise loss function.

        gradient : array of shape (n_samples,) or (n_samples, n_classes)
            Element-wise gradients.
        """
        if loss_out is None:
            if gradient_out is None:
                loss_out = np.empty_like(y_true)
                gradient_out = np.empty_like(raw_prediction)
            else:
                loss_out = np.empty_like(y_true, dtype=gradient_out.dtype)
        elif gradient_out is None:
            gradient_out = np.empty_like(raw_prediction, dtype=loss_out.dtype)

        # Be graceful to shape (n_samples, 1) -> (n_samples,)
        if raw_prediction.ndim == 2 and raw_prediction.shape[1] == 1:
            raw_prediction = raw_prediction.squeeze(1)
        if gradient_out.ndim == 2 and gradient_out.shape[1] == 1:
            gradient_out = gradient_out.squeeze(1)

        self.closs.loss_gradient(
            y_true=y_true,
            raw_prediction=raw_prediction,
            sample_weight=sample_weight,
            loss_out=loss_out,
            gradient_out=gradient_out,
            n_threads=n_threads,
        )
        return loss_out, gradient_out

    def gradient(
        self,
        y_true,
        raw_prediction,
        sample_weight=None,
        gradient_out=None,
        n_threads=1,
    ):
        # [FUNCTION DOC]
        # Allocates gradient_out like raw_prediction if missing, squeezes (n_samples, 1)
        # arrays, and calls closs.gradient to fill per-sample (weighted) gradients.
        # Outputs: gradient_out. Side effects: writes into the provided gradient_out.
        """Compute gradient of loss w.r.t raw_prediction for each input.

        Parameters
        ----------
        y_true : C-contiguous array of shape (n_samples,)
            Observed, true target values.
        raw_prediction : C-contiguous array of shape (n_samples,) or array of \
            shape (n_samples, n_classes)
            Raw prediction values (in link space).
        sample_weight : None or C-contiguous array of shape (n_samples,)
            Sample weights.
        gradient_out : None or C-contiguous array of shape (n_samples,) or array \
            of shape (n_samples, n_classes)
            A location into which the result is stored. If None, a new array
            might be created.
        n_threads : int, default=1
            Might use openmp thread parallelism.

        Returns
        -------
        gradient : array of shape (n_samples,) or (n_samples, n_classes)
            Element-wise gradients.
        """
        if gradient_out is None:
            gradient_out = np.empty_like(raw_prediction)

        # Be graceful to shape (n_samples, 1) -> (n_samples,)
        if raw_prediction.ndim == 2 and raw_prediction.shape[1] == 1:
            raw_prediction = raw_prediction.squeeze(1)
        if gradient_out.ndim == 2 and gradient_out.shape[1] == 1:
            gradient_out = gradient_out.squeeze(1)

        self.closs.gradient(
            y_true=y_true,
            raw_prediction=raw_prediction,
            sample_weight=sample_weight,
            gradient_out=gradient_out,
            n_threads=n_threads,
        )
        return gradient_out

    def gradient_hessian(
        self,
        y_true,
        raw_prediction,
        sample_weight=None,
        gradient_out=None,
        hessian_out=None,
        n_threads=1,
    ):
        # [FUNCTION DOC]
        # Allocates missing gradient/hessian outputs (like raw_prediction, or like the
        # other provided output), squeezes (n_samples, 1) arrays, and calls
        # closs.gradient_hessian to fill both.
        # Outputs: tuple (gradient_out, hessian_out). Side effects: writes provided buffers.
        """Compute gradient and hessian of loss w.r.t raw_prediction.

        Parameters
        ----------
        y_true : C-contiguous array of shape (n_samples,)
            Observed, true target values.
        raw_prediction : C-contiguous array of shape (n_samples,) or array of \
            shape (n_samples, n_classes)
            Raw prediction values (in link space).
        sample_weight : None or C-contiguous array of shape (n_samples,)
            Sample weights.
        gradient_out : None or C-contiguous array of shape (n_samples,) or array \
            of shape (n_samples, n_classes)
            A location into which the gradient is stored. If None, a new array
            might be created.
        hessian_out : None or C-contiguous array of shape (n_samples,) or array \
            of shape (n_samples, n_classes)
            A location into which the hessian is stored. If None, a new array
            might be created.
        n_threads : int, default=1
            Might use openmp thread parallelism.

        Returns
        -------
        gradient : arrays of shape (n_samples,) or (n_samples, n_classes)
            Element-wise gradients.

        hessian : arrays of shape (n_samples,) or (n_samples, n_classes)
            Element-wise hessians.
        """
        if gradient_out is None:
            if hessian_out is None:
                gradient_out = np.empty_like(raw_prediction)
                hessian_out = np.empty_like(raw_prediction)
            else:
                gradient_out = np.empty_like(hessian_out)
        elif hessian_out is None:
            hessian_out = np.empty_like(gradient_out)

        # Be graceful to shape (n_samples, 1) -> (n_samples,)
        if raw_prediction.ndim == 2 and raw_prediction.shape[1] == 1:
            raw_prediction = raw_prediction.squeeze(1)
        if gradient_out.ndim == 2 and gradient_out.shape[1] == 1:
            gradient_out = gradient_out.squeeze(1)
        if hessian_out.ndim == 2 and hessian_out.shape[1] == 1:
            hessian_out = hessian_out.squeeze(1)

        self.closs.gradient_hessian(
            y_true=y_true,
            raw_prediction=raw_prediction,
            sample_weight=sample_weight,
            gradient_out=gradient_out,
            hessian_out=hessian_out,
            n_threads=n_threads,
        )
        return gradient_out, hessian_out

    def __call__(self, y_true, raw_prediction, sample_weight=None, n_threads=1):
        # [FUNCTION DOC]
        # Returns the mean per-sample loss, weighted by sample_weight via np.average
        # (the per-sample loss is computed unweighted to avoid double weighting).
        # Outputs: numpy float scalar.
        # Exceptions: ZeroDivisionError from np.average if the weights sum to zero.
        """Compute the weighted average loss.

        Parameters
        ----------
        y_true : C-contiguous array of shape (n_samples,)
            Observed, true target values.
        raw_prediction : C-contiguous array of shape (n_samples,) or array of \
            shape (n_samples, n_classes)
            Raw prediction values (in link space).
        sample_weight : None or C-contiguous array of shape (n_samples,)
            Sample weights.
        n_threads : int, default=1
            Might use openmp thread parallelism.

        Returns
        -------
        loss : float
            Mean or averaged loss function.
        """
        return np.average(
            self.loss(
                y_true=y_true,
                raw_prediction=raw_prediction,
                sample_weight=None,
                loss_out=None,
                n_threads=n_threads,
            ),
            weights=sample_weight,
        )

    def fit_intercept_only(self, y_true, sample_weight=None):
        # [FUNCTION DOC]
        # Default intercept-only fit: takes the weighted mean of y_true over axis 0, clips
        # it into interval_y_pred (exclusive finite ends are moved inward by 10 machine eps
        # of the mean's dtype; infinite ends are not clipped) and applies the link.
        # Inputs: y_true (n_samples,) or (n_samples, k); sample_weight optional.
        # Outputs: scalar or array in raw-prediction space.
        """Compute raw_prediction of an intercept-only model.

        This can be used as initial estimates of predictions, i.e. before the
        first iteration in fit.

        Parameters
        ----------
        y_true : array-like of shape (n_samples,)
            Observed, true target values.
        sample_weight : None or array of shape (n_samples,)
            Sample weights.

        Returns
        -------
        raw_prediction : numpy scalar or array of shape (n_classes,)
            Raw predictions of an intercept-only model.
        """
        # As default, take weighted average of the target over the samples
        # axis=0 and then transform into link-scale (raw_prediction).
        y_pred = np.average(y_true, weights=sample_weight, axis=0)
        eps = 10 * np.finfo(y_pred.dtype).eps

        if self.interval_y_pred.low == -np.inf:
            a_min = None
        elif self.interval_y_pred.low_inclusive:
            a_min = self.interval_y_pred.low
        else:
            a_min = self.interval_y_pred.low + eps

        if self.interval_y_pred.high == np.inf:
            a_max = None
        elif self.interval_y_pred.high_inclusive:
            a_max = self.interval_y_pred.high
        else:
            a_max = self.interval_y_pred.high - eps

        if a_min is None and a_max is None:
            return self.link.link(y_pred)
        else:
            return self.link.link(np.clip(y_pred, a_min, a_max))

    def constant_to_optimal_zero(self, y_true, sample_weight=None):
        # [FUNCTION DOC]
        # Returns the per-sample constant term that, when added to the loss, makes the loss
        # of perfect predictions zero. Base implementation returns zeros like y_true;
        # sample_weight is ignored here.
        # KNOWN ISSUE: the docstring says the constant is added to raw predictions; it is
        # actually added to the loss values.
        """Calculate term dropped in loss.

        With this term added, the loss of perfect predictions is zero.

        Parameters
        ----------
        y_true : array-like of shape (n_samples,)
            Observed, true target values.

        sample_weight : None or array of shape (n_samples,), default=None
            Sample weights.

        Returns
        -------
        constant : ndarray of shape (n_samples,)
            Constant value to be added to raw predictions so that the loss
            of perfect predictions becomes zero.
        """
        return np.zeros_like(y_true)

    def init_gradient_and_hessian(self, n_samples, dtype=np.float64, order="F"):
        # [FUNCTION DOC]
        # Allocates the gradient array of shape (n_samples,) or (n_samples, n_classes) for
        # multiclass losses, uninitialised, with the given dtype and memory order. The
        # hessian is a length-1 array of ones when constant_hessian is True, otherwise an
        # uninitialised array of the gradient's shape.
        # Outputs: tuple (gradient, hessian).
        # Exceptions: ValueError if dtype is not np.float32 or np.float64.
        # KNOWN ISSUE: the docstring calls the outputs C-contiguous although the default
        # order is 'F'.
        """Initialize arrays for gradients and hessians.

        Unless hessians are constant, arrays are initialized with undefined values.

        Parameters
        ----------
        n_samples : int
            The number of samples, usually passed to `fit()`.
        dtype : {np.float64, np.float32}, default=np.float64
            The dtype of the arrays gradient and hessian.
        order : {'C', 'F'}, default='F'
            Order of the arrays gradient and hessian. The default 'F' makes the arrays
            contiguous along samples.

        Returns
        -------
        gradient : C-contiguous array of shape (n_samples,) or array of shape \
            (n_samples, n_classes)
            Empty array (allocated but not initialized) to be used as argument
            gradient_out.
        hessian : C-contiguous array of shape (n_samples,), array of shape
            (n_samples, n_classes) or shape (1,)
            Empty (allocated but not initialized) array to be used as argument
            hessian_out.
            If constant_hessian is True (e.g. `HalfSquaredError`), the array is
            initialized to ``1``.
        """
        if dtype not in (np.float32, np.float64):
            raise ValueError(
                "Valid options for 'dtype' are np.float32 and np.float64. "
                f"Got dtype={dtype} instead."
            )

        if self.is_multiclass:
            shape = (n_samples, self.n_classes)
        else:
            shape = (n_samples,)
        gradient = np.empty(shape=shape, dtype=dtype, order=order)

        if self.constant_hessian:
            # If the hessians are constant, we consider them equal to 1.
            # - This is correct for HalfSquaredError
            # - For AbsoluteError, hessians are actually 0, but they are
            #   always ignored anyway.
            hessian = np.ones(shape=(1,), dtype=dtype)
        else:
            hessian = np.empty(shape=shape, dtype=dtype, order=order)

        return gradient, hessian


class HalfSquaredError(BaseLoss):
    # [CLASS DOC]
    # Half squared error loss with identity link. constant_hessian is True only when
    # the constructor receives sample_weight=None (unit hessian).
    """Half squared error with identity link, for regression.

    Domain:
    y_true and y_pred all real numbers

    Link:
    y_pred = raw_prediction

    For a given sample x_i, half squared error is defined as::

        loss(x_i) = 0.5 * (y_true_i - raw_prediction_i)**2

    The factor of 0.5 simplifies the computation of gradients and results in a
    unit hessian (and is consistent with what is done in LightGBM). It is also
    half the Normal distribution deviance.
    """

    def __init__(self, sample_weight=None, xp=None, device=None):
        # [FUNCTION DOC]
        # Builds the base loss with CyHalfSquaredError and IdentityLink; sample_weight is
        # used only to decide constant_hessian (not stored).
        super().__init__(
            closs=CyHalfSquaredError(), link=IdentityLink(), xp=xp, device=device
        )
        self.constant_hessian = sample_weight is None


class AbsoluteError(BaseLoss):
    # [CLASS DOC]
    # Absolute error with identity link; differentiable is False and the hessian is
    # approximated (set to 1). constant_hessian when no sample_weight is given.
    # Intercept-only fit is the (weighted) median.
    """Absolute error with identity link, for regression.

    Domain:
    y_true and y_pred all real numbers

    Link:
    y_pred = raw_prediction

    For a given sample x_i, the absolute error is defined as::

        loss(x_i) = |y_true_i - raw_prediction_i|

    Note that the exact hessian = 0 almost everywhere (except at one point, therefore
    differentiable = False). Optimization routines like in HGBT, however, need a
    hessian > 0. Therefore, we assign 1.
    """

    differentiable = False

    def __init__(self, sample_weight=None, xp=None, device=None):
        # [FUNCTION DOC]
        # Builds the base loss with CyAbsoluteError and IdentityLink, sets approx_hessian
        # True and constant_hessian to whether sample_weight is None.
        super().__init__(
            closs=CyAbsoluteError(), link=IdentityLink(), xp=xp, device=device
        )
        self.approx_hessian = True
        self.constant_hessian = sample_weight is None

    def fit_intercept_only(self, y_true, sample_weight=None):
        # [FUNCTION DOC]
        # Intercept-only fit for absolute error: np.median over axis 0 when sample_weight is
        # None, else the weighted 50th percentile via _weighted_percentile.
        # Outputs: scalar or per-column array.
        """Compute raw_prediction of an intercept-only model.

        This is the weighted median of the target, i.e. over the samples
        axis=0.
        """
        if sample_weight is None:
            return np.median(y_true, axis=0)
        else:
            return _weighted_percentile(y_true, sample_weight, 50)


class PinballLoss(BaseLoss):
    # [CLASS DOC]
    # Pinball (quantile) loss with identity link; quantile is stored in the Cython
    # object (closs.quantile). Not differentiable, approximate unit hessian,
    # constant_hessian when sample_weight is None.
    """Quantile loss aka pinball loss, for regression.

    Domain:
    y_true and y_pred all real numbers
    quantile in (0, 1)

    Link:
    y_pred = raw_prediction

    For a given sample x_i, the pinball loss is defined as::

        loss(x_i) = rho_{quantile}(y_true_i - raw_prediction_i)

        rho_{quantile}(u) = u * (quantile - 1_{u<0})
                          = -u *(1 - quantile)  if u < 0
                             u * quantile       if u >= 0

    Note: 2 * PinballLoss(quantile=0.5) equals AbsoluteError().

    Note that the exact hessian = 0 almost everywhere (except at one point, therefore
    differentiable = False). Optimization routines like in HGBT, however, need a
    hessian > 0. Therefore, we assign 1.

    Additional Attributes
    ---------------------
    quantile : float
        The quantile level of the quantile to be estimated. Must be in range (0, 1).
    """

    differentiable = False

    def __init__(self, sample_weight=None, quantile=0.5, xp=None, device=None):
        # [FUNCTION DOC]
        # Validates quantile as a real number strictly inside (0, 1) with check_scalar,
        # builds CyPinballLoss(quantile) with IdentityLink and sets approx_hessian True and
        # constant_hessian to whether sample_weight is None.
        # Exceptions: TypeError / ValueError from check_scalar.
        check_scalar(
            quantile,
            "quantile",
            target_type=numbers.Real,
            min_val=0,
            max_val=1,
            include_boundaries="neither",
        )
        super().__init__(
            closs=CyPinballLoss(quantile=float(quantile)),
            link=IdentityLink(),
            xp=xp,
            device=device,
        )
        self.approx_hessian = True
        self.constant_hessian = sample_weight is None

    def fit_intercept_only(self, y_true, sample_weight=None):
        # [FUNCTION DOC]
        # Intercept-only fit: the (weighted) percentile of y_true at 100 * quantile, via
        # np.percentile over axis 0 or _weighted_percentile when weights are given.
        # KNOWN ISSUE: the docstring says "weighted median", but it computes the quantile
        # given by closs.quantile.
        """Compute raw_prediction of an intercept-only model.

        This is the weighted median of the target, i.e. over the samples
        axis=0.
        """
        if sample_weight is None:
            return np.percentile(y_true, 100 * self.closs.quantile, axis=0)
        else:
            return _weighted_percentile(
                y_true, sample_weight, 100 * self.closs.quantile
            )


class HuberLoss(BaseLoss):
    # [CLASS DOC]
    # Huber loss with identity link. The threshold delta lives in closs.delta (mutable
    # by callers such as HGBT, which recompute it from the quantile of absolute
    # residuals); quantile is stored on the Python object. Not differentiable,
    # approximate hessian, never constant hessian.
    # KNOWN ISSUE: the docstring lists "quantile in (0, 1)" under Domain and has a
    # mis-indented "Reference" heading.
    """Huber loss, for regression.

    Domain:
    y_true and y_pred all real numbers
    quantile in (0, 1)

    Link:
    y_pred = raw_prediction

    For a given sample x_i, the Huber loss is defined as::

        loss(x_i) = 1/2 * abserr**2            if abserr <= delta
                    delta * (abserr - delta/2) if abserr > delta

        abserr = |y_true_i - raw_prediction_i|
        delta = quantile(abserr, self.quantile)

    Note: HuberLoss(quantile=1) equals HalfSquaredError and HuberLoss(quantile=0)
    equals delta * (AbsoluteError() - delta/2).

    Additional Attributes
    ---------------------
    quantile : float
        The quantile level which defines the breaking point `delta` to distinguish
        between absolute error and squared error. Must be in range (0, 1).

     Reference
    ---------
    .. [1] Friedman, J.H. (2001). :doi:`Greedy function approximation: A gradient
      boosting machine <10.1214/aos/1013203451>`.
      Annals of Statistics, 29, 1189-1232.
    """

    differentiable = False

    def __init__(
        self, sample_weight=None, quantile=0.9, delta=0.5, xp=None, device=None
    ):
        # [FUNCTION DOC]
        # Validates quantile in the open interval (0, 1) with check_scalar, stores it as
        # self.quantile, builds CyHuberLoss(delta) with IdentityLink, and sets
        # approx_hessian True and constant_hessian False. sample_weight is unused.
        # Exceptions: TypeError / ValueError from check_scalar. delta is not validated.
        check_scalar(
            quantile,
            "quantile",
            target_type=numbers.Real,
            min_val=0,
            max_val=1,
            include_boundaries="neither",
        )
        self.quantile = quantile  # This is better stored outside of Cython.
        super().__init__(
            closs=CyHuberLoss(delta=float(delta)),
            link=IdentityLink(),
            xp=xp,
            device=device,
        )
        self.approx_hessian = True
        self.constant_hessian = False

    def fit_intercept_only(self, y_true, sample_weight=None):
        # [FUNCTION DOC]
        # Intercept-only fit following Friedman (2001): computes the (weighted) median of
        # y_true, then adds the (weighted) mean of the residuals from the median clipped
        # in magnitude to closs.delta (sign times min(delta, |residual|)).
        # Outputs: scalar (1d y_true).
        # KNOWN ISSUE: the docstring claims it returns the weighted median; it returns the
        # median plus a clipped-residual correction.
        """Compute raw_prediction of an intercept-only model.

        This is the weighted median of the target, i.e. over the samples
        axis=0.
        """
        # See formula before algo 4 in Friedman (2001), but we apply it to y_true,
        # not to the residual y_true - raw_prediction. An estimator like
        # HistGradientBoostingRegressor might then call it on the residual, e.g.
        # fit_intercept_only(y_true - raw_prediction).
        if sample_weight is None:
            median = np.percentile(y_true, 50, axis=0)
        else:
            median = _weighted_percentile(y_true, sample_weight, 50)
        diff = y_true - median
        term = np.sign(diff) * np.minimum(self.closs.delta, np.abs(diff))
        return median + np.average(term, weights=sample_weight)


class HalfPoissonLoss(BaseLoss):
    # [CLASS DOC]
    # Half Poisson deviance with log link; y_true in [0, inf), y_pred in (0, inf).
    """Half Poisson deviance loss with log-link, for regression.

    Domain:
    y_true in non-negative real numbers
    y_pred in positive real numbers

    Link:
    y_pred = exp(raw_prediction)

    For a given sample x_i, half the Poisson deviance is defined as::

        loss(x_i) = y_true_i * log(y_true_i/exp(raw_prediction_i))
                    - y_true_i + exp(raw_prediction_i)

    Half the Poisson deviance is actually the negative log-likelihood up to
    constant terms (not involving raw_prediction) and simplifies the
    computation of the gradients.
    We also skip the constant term `y_true_i * log(y_true_i) - y_true_i`.
    """

    def __init__(self, sample_weight=None, xp=None, device=None):
        # [FUNCTION DOC]
        # Builds CyHalfPoissonLoss with LogLink and sets interval_y_true to [0, inf).
        # sample_weight is unused.
        super().__init__(
            closs=CyHalfPoissonLoss(), link=LogLink(), xp=xp, device=device
        )
        self.interval_y_true = Interval(0, np.inf, True, False)

    def constant_to_optimal_zero(self, y_true, sample_weight=None):
        # [FUNCTION DOC]
        # Returns per-sample y*log(y) - y (xlogy gives 0 at y = 0), multiplied by
        # sample_weight if given. Outputs: array (n_samples,).
        term = xlogy(y_true, y_true) - y_true
        if sample_weight is not None:
            term *= sample_weight
        return term


class HalfGammaLoss(BaseLoss):
    # [CLASS DOC]
    # Half Gamma deviance with log link; y_true and y_pred in (0, inf).
    """Half Gamma deviance loss with log-link, for regression.

    Domain:
    y_true and y_pred in positive real numbers

    Link:
    y_pred = exp(raw_prediction)

    For a given sample x_i, half Gamma deviance loss is defined as::

        loss(x_i) = log(exp(raw_prediction_i)/y_true_i)
                    + y_true/exp(raw_prediction_i) - 1

    Half the Gamma deviance is actually proportional to the negative log-
    likelihood up to constant terms (not involving raw_prediction) and
    simplifies the computation of the gradients.
    We also skip the constant term `-log(y_true_i) - 1`.
    """

    def __init__(self, sample_weight=None, xp=None, device=None):
        # [FUNCTION DOC]
        # Builds CyHalfGammaLoss with LogLink and sets interval_y_true to (0, inf).
        super().__init__(closs=CyHalfGammaLoss(), link=LogLink(), xp=xp, device=device)
        self.interval_y_true = Interval(0, np.inf, False, False)

    def constant_to_optimal_zero(self, y_true, sample_weight=None):
        # [FUNCTION DOC]
        # Returns per-sample -log(y) - 1, multiplied by sample_weight if given.
        term = -np.log(y_true) - 1
        if sample_weight is not None:
            term *= sample_weight
        return term


class HalfTweedieLoss(BaseLoss):
    # [CLASS DOC]
    # Half Tweedie deviance with log link, parametrised by power (stored in
    # closs.power). interval_y_true depends on power: real line for power <= 0,
    # [0, inf) for 0 < power < 2, (0, inf) for power >= 2.
    """Half Tweedie deviance loss with log-link, for regression.

    Domain:
    y_true in real numbers for power <= 0
    y_true in non-negative real numbers for 0 < power < 2
    y_true in positive real numbers for 2 <= power
    y_pred in positive real numbers
    power in real numbers

    Link:
    y_pred = exp(raw_prediction)

    For a given sample x_i, half Tweedie deviance loss with p=power is defined
    as::

        loss(x_i) = max(y_true_i, 0)**(2-p) / (1-p) / (2-p)
                    - y_true_i * exp(raw_prediction_i)**(1-p) / (1-p)
                    + exp(raw_prediction_i)**(2-p) / (2-p)

    Taking the limits for p=0, 1, 2 gives HalfSquaredError with a log link,
    HalfPoissonLoss and HalfGammaLoss.

    We also skip constant terms, but those are different for p=0, 1, 2.
    Therefore, the loss is not continuous in `power`.

    Note furthermore that although no Tweedie distribution exists for
    0 < power < 1, it still gives a strictly consistent scoring function for
    the expectation.
    """

    def __init__(self, sample_weight=None, power=1.5, xp=None, device=None):
        # [FUNCTION DOC]
        # Builds CyHalfTweedieLoss(power) with LogLink and sets interval_y_true by power as
        # described in the class doc. power is not validated beyond float conversion.
        super().__init__(
            closs=CyHalfTweedieLoss(power=float(power)),
            link=LogLink(),
            xp=xp,
            device=device,
        )
        if self.closs.power <= 0:
            self.interval_y_true = Interval(-np.inf, np.inf, False, False)
        elif self.closs.power < 2:
            self.interval_y_true = Interval(0, np.inf, True, False)
        else:
            self.interval_y_true = Interval(0, np.inf, False, False)

    def constant_to_optimal_zero(self, y_true, sample_weight=None):
        # [FUNCTION DOC]
        # Dispatches to HalfSquaredError (power 0, zeros), HalfPoissonLoss (power 1) or
        # HalfGammaLoss (power 2); otherwise returns max(y, 0)^(2-p) / ((1-p)(2-p)),
        # multiplied by sample_weight if given.
        if self.closs.power == 0:
            return HalfSquaredError().constant_to_optimal_zero(
                y_true=y_true, sample_weight=sample_weight
            )
        elif self.closs.power == 1:
            return HalfPoissonLoss().constant_to_optimal_zero(
                y_true=y_true, sample_weight=sample_weight
            )
        elif self.closs.power == 2:
            return HalfGammaLoss().constant_to_optimal_zero(
                y_true=y_true, sample_weight=sample_weight
            )
        else:
            p = self.closs.power
            term = np.power(np.maximum(y_true, 0), 2 - p) / (1 - p) / (2 - p)
            if sample_weight is not None:
                term *= sample_weight
            return term


class HalfTweedieLossIdentity(BaseLoss):
    # [CLASS DOC]
    # Half Tweedie deviance with identity link (y_pred = raw_prediction). y_true domain
    # as for HalfTweedieLoss; y_pred domain is the real line for power 0, else (0, inf).
    # Its loss already includes the constant term, so the base
    # constant_to_optimal_zero (zeros) applies.
    """Half Tweedie deviance loss with identity link, for regression.

    Domain:
    y_true in real numbers for power <= 0
    y_true in non-negative real numbers for 0 < power < 2
    y_true in positive real numbers for 2 <= power
    y_pred in positive real numbers for power != 0
    y_pred in real numbers for power = 0
    power in real numbers

    Link:
    y_pred = raw_prediction

    For a given sample x_i, half Tweedie deviance loss with p=power is defined
    as::

        loss(x_i) = max(y_true_i, 0)**(2-p) / (1-p) / (2-p)
                    - y_true_i * raw_prediction_i**(1-p) / (1-p)
                    + raw_prediction_i**(2-p) / (2-p)

    Note that the minimum value of this loss is 0.

    Note furthermore that although no Tweedie distribution exists for
    0 < power < 1, it still gives a strictly consistent scoring function for
    the expectation.
    """

    def __init__(self, sample_weight=None, power=1.5, xp=None, device=None):
        # [FUNCTION DOC]
        # Builds CyHalfTweedieLossIdentity(power) with IdentityLink and overrides
        # interval_y_true and interval_y_pred depending on power.
        super().__init__(
            closs=CyHalfTweedieLossIdentity(power=float(power)),
            link=IdentityLink(),
            xp=xp,
            device=device,
        )
        if self.closs.power <= 0:
            self.interval_y_true = Interval(-np.inf, np.inf, False, False)
        elif self.closs.power < 2:
            self.interval_y_true = Interval(0, np.inf, True, False)
        else:
            self.interval_y_true = Interval(0, np.inf, False, False)

        if self.closs.power == 0:
            self.interval_y_pred = Interval(-np.inf, np.inf, False, False)
        else:
            self.interval_y_pred = Interval(0, np.inf, False, False)


class HalfBinomialLoss(BaseLoss):
    # [CLASS DOC]
    # Half binomial deviance (log loss) with logit link for binary classification;
    # n_classes is 2, y_true in [0, 1] (inclusive), y_pred in (0, 1).
    """Half Binomial deviance loss with logit link, for binary classification.

    This is also know as binary cross entropy, log-loss and logistic loss.

    Domain:
    y_true in [0, 1], i.e. regression on the unit interval
    y_pred in (0, 1), i.e. boundaries excluded

    Link:
    y_pred = expit(raw_prediction)

    For a given sample x_i, half Binomial deviance is defined as the negative
    log-likelihood of the Binomial/Bernoulli distribution and can be expressed
    as::

        loss(x_i) = log(1 + exp(raw_pred_i)) - y_true_i * raw_pred_i

    See The Elements of Statistical Learning, by Hastie, Tibshirani, Friedman,
    section 4.4.1 (about logistic regression).

    Note that the formulation works for classification, y = {0, 1}, as well as
    logistic regression, y = [0, 1].
    If you add `constant_to_optimal_zero` to the loss, you get half the
    Bernoulli/binomial deviance.

    More details: Inserting the predicted probability y_pred = expit(raw_prediction)
    in the loss gives the well known::

        loss(x_i) = - y_true_i * log(y_pred_i) - (1 - y_true_i) * log(1 - y_pred_i)
    """

    def __init__(self, sample_weight=None, xp=None, device=None):
        # [FUNCTION DOC]
        # Builds CyHalfBinomialLoss with LogitLink and n_classes=2, sets interval_y_true to
        # the closed interval [0, 1].
        super().__init__(
            closs=CyHalfBinomialLoss(),
            link=LogitLink(),
            n_classes=2,
            xp=xp,
            device=device,
        )
        self.interval_y_true = Interval(0, 1, True, True)

    def constant_to_optimal_zero(self, y_true, sample_weight=None):
        # [FUNCTION DOC]
        # Returns y*log(y) + (1-y)*log(1-y) per sample (zero for y in {0, 1}), multiplied
        # by sample_weight if given.
        # This is non-zero only if y_true is neither 0 nor 1.
        term = xlogy(y_true, y_true) + xlogy(1 - y_true, 1 - y_true)
        if sample_weight is not None:
            term *= sample_weight
        return term

    def predict_proba(self, raw_prediction):
        # [FUNCTION DOC]
        # Squeezes (n_samples, 1) input, then returns an (n_samples, 2) array with column 1
        # = expit(raw_prediction) and column 0 = its complement, in raw_prediction's dtype.
        """Predict probabilities.

        Parameters
        ----------
        raw_prediction : array of shape (n_samples,) or (n_samples, 1)
            Raw prediction values (in link space).

        Returns
        -------
        proba : array of shape (n_samples, 2)
            Element-wise class probabilities.
        """
        # Be graceful to shape (n_samples, 1) -> (n_samples,)
        if raw_prediction.ndim == 2 and raw_prediction.shape[1] == 1:
            raw_prediction = raw_prediction.squeeze(1)
        proba = np.empty((raw_prediction.shape[0], 2), dtype=raw_prediction.dtype)
        proba[:, 1] = self.link.inverse(raw_prediction)
        proba[:, 0] = 1 - proba[:, 1]
        return proba


class HalfMultinomialLoss(BaseLoss):
    # [CLASS DOC]
    # Multinomial cross-entropy with softmax inverse link and symmetric multinomial
    # logit link; is_multiclass True. y_true holds integer-valued class labels in
    # [0, n_classes). Hessian methods return only the diagonal per class. Caches
    # class_indexing_offsets, y_true_int and y_true_one_hot are reserved for the Array
    # API subclass.
    """Categorical cross-entropy loss, for multiclass classification.

    Domain:
    y_true in {0, 1, 2, 3, .., n_classes - 1}
    y_pred has n_classes elements, each element in (0, 1)

    Link:
    y_pred = softmax(raw_prediction)

    Note: We assume y_true to be already label encoded. The inverse link is
    softmax. But the full link function is the symmetric multinomial logit
    function.

    For a given sample x_i, the categorical cross-entropy loss is defined as
    the negative log-likelihood of the multinomial distribution, it
    generalizes the binary cross-entropy to more than 2 classes::

        loss_i = log(sum(exp(raw_pred_{i, k}), k=0..n_classes-1))
                - sum(y_true_{i, k} * raw_pred_{i, k}, k=0..n_classes-1)

    See [1].

    Note that for the hessian, we calculate only the diagonal part in the
    classes: If the full hessian for classes k and l and sample i is H_i_k_l,
    we calculate H_i_k_k, i.e. k=l.

    Parameters
    ----------
    sample_weight : {None, ndarray}
        If sample_weight is None, the hessian might be constant.

    n_classes : {None, int}
        The number of classes for classification, else None.

    xp : module or None
        Array namespace module. Ignored by the Cython implementation.

    device : device or None
        A device object. Ignored by the Cython implementation.

    References
    ----------
    .. [1] :arxiv:`Simon, Noah, J. Friedman and T. Hastie.
        "A Blockwise Descent Algorithm for Group-penalized Multiresponse and
        Multinomial Regression".
        <1311.6529>`
    """

    is_multiclass = True

    def __init__(self, sample_weight=None, n_classes=3, xp=None, device=None):
        # [FUNCTION DOC]
        # Builds CyHalfMultinomialLoss with MultinomialLogit and n_classes (default 3),
        # sets interval_y_true to [0, inf), interval_y_pred to (0, 1) and the three Array
        # API caches to None.
        super().__init__(
            closs=CyHalfMultinomialLoss(),
            link=MultinomialLogit(),
            n_classes=n_classes,
            xp=xp,
            device=device,
        )
        self.interval_y_true = Interval(0, np.inf, True, False)
        self.interval_y_pred = Interval(0, 1, False, False)
        # These instance variables are specifically used for the array API
        # methods to store certain intermediate values in order to avoid
        # having to recompute them repeatedly.
        self.class_indexing_offsets = None
        self.y_true_int = None
        self.y_true_one_hot = None

    def in_y_true_range(self, y):
        # [FUNCTION DOC]
        # Returns True if y lies in [0, inf) and all values are integral (compared after
        # casting to int). Uses NumPy directly.
        """Return True if y is in the valid range of y_true.

        Parameters
        ----------
        y : ndarray
        """
        return self.interval_y_true.includes(y) and np.all(y.astype(int) == y)

    def fit_intercept_only(self, y_true, sample_weight=None):
        # [FUNCTION DOC]
        # Intercept-only fit: for each class k computes the weighted frequency of
        # y_true == k, clips it to [eps, 1 - eps] (dtype eps of y_true), and returns the
        # symmetric multinomial logit of the frequency vector as a 1d array (n_classes,).
        # KNOWN ISSUE: docstring says "softmax of the weighted average"; it is the link
        # (inverse of softmax) of class frequencies. y_true must be floating, since
        # np.finfo is called on its dtype.
        """Compute raw_prediction of an intercept-only model.

        This is the softmax of the weighted average of the target, i.e. over
        the samples axis=0.

        Parameters
        ----------
        y_true : array-like of shape (n_samples,)
            Observed, true target values.

        sample_weight : None or array of shape (n_samples,), default=None
            Sample weights.

        Returns
        -------
        raw_prediction : numpy scalar or array of shape (n_classes,)
            Raw predictions of an intercept-only model.
        """
        out = np.zeros(self.n_classes, dtype=y_true.dtype)
        eps = np.finfo(y_true.dtype).eps
        for k in range(self.n_classes):
            out[k] = np.average(y_true == k, weights=sample_weight, axis=0)
            out[k] = np.clip(out[k], eps, 1 - eps)
        return self.link.link(out[None, :]).reshape(-1)

    def predict_proba(self, raw_prediction):
        # [FUNCTION DOC]
        # Returns the row-wise softmax of raw_prediction (n_samples, n_classes).
        """Predict probabilities.

        Parameters
        ----------
        raw_prediction : array of shape (n_samples, n_classes)
            Raw prediction values (in link space).

        Returns
        -------
        proba : array of shape (n_samples, n_classes)
            Element-wise class probabilities.
        """
        return self.link.inverse(raw_prediction)

    def gradient_proba(
        self,
        y_true,
        raw_prediction,
        sample_weight=None,
        gradient_out=None,
        proba_out=None,
        n_threads=1,
    ):
        # [FUNCTION DOC]
        # Allocates missing gradient/proba outputs (like raw_prediction, or like the other
        # provided output) and calls closs.gradient_proba, which fills per-sample class
        # probabilities (softmax) and gradients (proba minus one-hot of y_true, times
        # sample_weight).
        # Outputs: tuple (gradient_out, proba_out), both (n_samples, n_classes).
        # Side effects: writes provided buffers. No (n, 1) squeezing here.
        """Compute gradient and class probabilities fow raw_prediction.

        Parameters
        ----------
        y_true : C-contiguous array of shape (n_samples,)
            Observed, true target values.
        raw_prediction : array of shape (n_samples, n_classes)
            Raw prediction values (in link space).
        sample_weight : None or C-contiguous array of shape (n_samples,)
            Sample weights.
        gradient_out : None or array of shape (n_samples, n_classes)
            A location into which the gradient is stored. If None, a new array
            might be created.
        proba_out : None or array of shape (n_samples, n_classes)
            A location into which the class probabilities are stored. If None,
            a new array might be created.
        n_threads : int, default=1
            Might use openmp thread parallelism.

        Returns
        -------
        gradient : array of shape (n_samples, n_classes)
            Element-wise gradients.

        proba : array of shape (n_samples, n_classes)
            Element-wise class probabilities.
        """
        if gradient_out is None:
            if proba_out is None:
                gradient_out = np.empty_like(raw_prediction)
                proba_out = np.empty_like(raw_prediction)
            else:
                gradient_out = np.empty_like(proba_out)
        elif proba_out is None:
            proba_out = np.empty_like(gradient_out)

        self.closs.gradient_proba(
            y_true=y_true,
            raw_prediction=raw_prediction,
            sample_weight=sample_weight,
            gradient_out=gradient_out,
            proba_out=proba_out,
            n_threads=n_threads,
        )
        return gradient_out, proba_out


class ExponentialLoss(BaseLoss):
    # [CLASS DOC]
    # Exponential (boosting) loss with half-logit link for binary classification;
    # n_classes 2, y_true in [0, 1] inclusive. Not listed in the package __all__ but
    # registered in _LOSSES.
    """Exponential loss with (half) logit link, for binary classification.

    This is also know as boosting loss.

    Domain:
    y_true in [0, 1], i.e. regression on the unit interval
    y_pred in (0, 1), i.e. boundaries excluded

    Link:
    y_pred = expit(2 * raw_prediction)

    For a given sample x_i, the exponential loss is defined as::

        loss(x_i) = y_true_i * exp(-raw_pred_i)) + (1 - y_true_i) * exp(raw_pred_i)

    See:
    - J. Friedman, T. Hastie, R. Tibshirani.
      "Additive logistic regression: a statistical view of boosting (With discussion
      and a rejoinder by the authors)." Ann. Statist. 28 (2) 337 - 407, April 2000.
      https://doi.org/10.1214/aos/1016218223
    - A. Buja, W. Stuetzle, Y. Shen. (2005).
      "Loss Functions for Binary Class Probability Estimation and Classification:
      Structure and Applications."

    Note that the formulation works for classification, y = {0, 1}, as well as
    "exponential logistic" regression, y = [0, 1].
    Note that this is a proper scoring rule, but without it's canonical link.

    More details: Inserting the predicted probability
    y_pred = expit(2 * raw_prediction) in the loss gives::

        loss(x_i) = y_true_i * sqrt((1 - y_pred_i) / y_pred_i)
            + (1 - y_true_i) * sqrt(y_pred_i / (1 - y_pred_i))
    """

    def __init__(self, sample_weight=None, xp=None, device=None):
        # [FUNCTION DOC]
        # Builds CyExponentialLoss with HalfLogitLink and n_classes=2, and sets
        # interval_y_true to [0, 1].
        super().__init__(
            closs=CyExponentialLoss(),
            link=HalfLogitLink(),
            n_classes=2,
            xp=xp,
            device=device,
        )
        self.interval_y_true = Interval(0, 1, True, True)

    def constant_to_optimal_zero(self, y_true, sample_weight=None):
        # [FUNCTION DOC]
        # Returns -2 * sqrt(y * (1 - y)) per sample (zero for y in {0, 1}), multiplied by
        # sample_weight if given.
        # This is non-zero only if y_true is neither 0 nor 1.
        term = -2 * np.sqrt(y_true * (1 - y_true))
        if sample_weight is not None:
            term *= sample_weight
        return term

    def predict_proba(self, raw_prediction):
        # [FUNCTION DOC]
        # Squeezes (n_samples, 1) input and returns an (n_samples, 2) array: column 1 is
        # expit(2 * raw_prediction) and column 0 its complement.
        """Predict probabilities.

        Parameters
        ----------
        raw_prediction : array of shape (n_samples,) or (n_samples, 1)
            Raw prediction values (in link space).

        Returns
        -------
        proba : array of shape (n_samples, 2)
            Element-wise class probabilities.
        """
        # Be graceful to shape (n_samples, 1) -> (n_samples,)
        if raw_prediction.ndim == 2 and raw_prediction.shape[1] == 1:
            raw_prediction = raw_prediction.squeeze(1)
        proba = np.empty((raw_prediction.shape[0], 2), dtype=raw_prediction.dtype)
        proba[:, 1] = self.link.inverse(raw_prediction)
        proba[:, 0] = 1 - proba[:, 1]
        return proba


_LOSSES = {
    "squared_error": HalfSquaredError,
    "absolute_error": AbsoluteError,
    "pinball_loss": PinballLoss,
    "huber_loss": HuberLoss,
    "poisson_loss": HalfPoissonLoss,
    "gamma_loss": HalfGammaLoss,
    "tweedie_loss": HalfTweedieLoss,
    "binomial_loss": HalfBinomialLoss,
    "multinomial_loss": HalfMultinomialLoss,
    "exponential_loss": ExponentialLoss,
}


class ArrayAPILossMixin:
    # [CLASS DOC]
    # Mixin placed before a NumPy/Cython loss in the MRO to replace __call__, loss,
    # loss_gradient and gradient with pure Array API implementations built on
    # self._compute_loss / self._compute_gradient using self.xp. *_out buffers and
    # n_threads are ignored; fresh arrays are returned.
    # KNOWN ISSUE: gradient_hessian, gradient_proba and predict_proba are not
    # overridden, so they still go through the Cython / NumPy code paths.
    """Mixin for loss classes that are compatible with the array API.

    Currently this mixin redefines methods:
    - __call__(...)
    - loss(...)
    - loss_gradient(...)
    - gradient(...)

    such that they work according to the array API specification.
    It uses the attributes self.xp and self.device from BaseLoss and it assumes that
    methods self._compute_loss and self._compute_gradient are implemented.
    """

    def __call__(
        self,
        y_true,
        raw_prediction,
        sample_weight=None,
        n_threads=1,
    ):
        # [FUNCTION DOC]
        # Computes the unweighted per-sample loss via self.loss and returns its
        # sample_weight-weighted average (sklearn _average in namespace xp) as a Python
        # float.
        """Compute the weighted average loss for the array API losses.

        Parameters
        ----------
        y_true : C-contiguous array of shape (n_samples,)
            Observed, true target values.
        raw_prediction : C-contiguous array of shape (n_samples,) or array of \
            shape (n_samples, n_classes)
            Raw prediction values (in link space).
        sample_weight : None or C-contiguous array of shape (n_samples,)
            Sample weights.
        n_threads : int, default=1
            Ignored by the array API implementation.

        Returns
        -------
        loss : float
            Mean or averaged loss function.
        """
        loss_xp = self.loss(
            y_true=y_true, raw_prediction=raw_prediction, sample_weight=None
        )
        return float(_average(loss_xp, weights=sample_weight, xp=self.xp))

    def loss(
        self,
        y_true,
        raw_prediction,
        sample_weight=None,
        loss_out=None,
        n_threads=1,
    ):
        # [FUNCTION DOC]
        # Returns self._compute_loss(y_true, raw_prediction, sample_weight): per-sample loss,
        # multiplied by sample_weight when given. loss_out and n_threads are ignored.
        """Compute the pointwise loss value for each input.

        Parameters
        ----------
        y_true : C-contiguous array of shape (n_samples,)
            Observed, true target values.
        raw_prediction : C-contiguous array of shape (n_samples,) or array of \
            shape (n_samples, n_classes)
            Raw prediction values (in link space).
        sample_weight : None or C-contiguous array of shape (n_samples,)
            Sample weights.
        loss_out : None or C-contiguous array of shape (n_samples,)
            Ignored by the array API implementation.
        n_threads : int, default=1
            Ignored by the array API implementation.

        Returns
        -------
        loss : array of shape (n_samples,)
            Element-wise loss function.
        """
        return self._compute_loss(
            y_true=y_true,
            raw_prediction=raw_prediction,
            sample_weight=sample_weight,
        )

    def loss_gradient(
        self,
        y_true,
        raw_prediction,
        sample_weight=None,
        loss_out=None,
        gradient_out=None,
        n_threads=1,
    ):
        # [FUNCTION DOC]
        # Returns the tuple (loss, gradient) from _compute_loss and _compute_gradient;
        # the output buffers and n_threads are ignored.
        """Compute loss and gradient w.r.t. raw_prediction for each input.

        Parameters
        ----------
        y_true : C-contiguous array of shape (n_samples,)
            Observed, true target values.
        raw_prediction : C-contiguous array of shape (n_samples,) or array of \
            shape (n_samples, n_classes)
            Raw prediction values (in link space).
        sample_weight : None or C-contiguous array of shape (n_samples,)
            Sample weights.
        loss_out : None or C-contiguous array of shape (n_samples,)
            Ignored by the array API implementation.
        gradient_out : None or C-contiguous array of shape (n_samples,) or array \
            of shape (n_samples, n_classes)
            Ignored by the array API implementation.
        n_threads : int, default=1
            Ignored by the array API implementation.

        Returns
        -------
        loss : array of shape (n_samples,)
            Element-wise loss function.

        gradient : array of shape (n_samples,) or (n_samples, n_classes)
            Element-wise gradients.
        """
        loss = self._compute_loss(
            y_true=y_true,
            raw_prediction=raw_prediction,
            sample_weight=sample_weight,
        )
        gradient = self._compute_gradient(
            y_true=y_true,
            raw_prediction=raw_prediction,
            sample_weight=sample_weight,
        )
        return loss, gradient

    def gradient(
        self,
        y_true,
        raw_prediction,
        sample_weight=None,
        gradient_out=None,
        n_threads=1,
    ):
        # [FUNCTION DOC]
        # Returns self._compute_gradient(...) (weighted per-sample gradient); output
        # buffer and n_threads are ignored.
        """Compute gradient of loss w.r.t raw_prediction for each input.

        Parameters
        ----------
        y_true : C-contiguous array of shape (n_samples,)
            Observed, true target values.
        raw_prediction : C-contiguous array of shape (n_samples,) or array of \
            shape (n_samples, n_classes)
            Raw prediction values (in link space).
        sample_weight : None or C-contiguous array of shape (n_samples,)
            Sample weights.
        gradient_out : None or C-contiguous array of shape (n_samples,) or array \
            of shape (n_samples, n_classes)
            Ignored by the array API implementation.
        n_threads : int, default=1
            Ignored by the array API implementation.

        Returns
        -------
        gradient : array of shape (n_samples,) or (n_samples, n_classes)
            Element-wise gradients.
        """
        return self._compute_gradient(
            y_true=y_true,
            raw_prediction=raw_prediction,
            sample_weight=sample_weight,
        )


def _log1pexp(raw_prediction, raw_prediction_exp, xp):
    # [FUNCTION DOC]
    # HELPER: Element-wise, numerically stable log(1 + exp(x)) for Array API arrays,
    # given x and precomputed exp(x). Piecewise by dtype-dependent cutoffs c0 < c1 <
    # c2 < c3 (float64: -37, -2, 18, 33.3; any other dtype: -17, -1, 9, 14.6):
    # x <= c0 -> exp(x); x <= c1 -> log1p(exp(x)); x <= c2 -> log(1 + exp(x));
    # x <= c3 -> x + 1/exp(x); else x.
    # Outputs: array of the same shape as raw_prediction.
    # KNOWN ISSUE: the docstring says it returns a float; it returns an array. All
    # branches are evaluated eagerly, so overflow in exp(x) for large x may emit
    # warnings even though that branch is discarded.
    """Numerically stable version of log(1 + exp(x)) that is compatible with
    the array API.

    Parameters
    ----------
    raw_prediction : C-contiguous array of shape (n_samples,) or array of \
        shape (n_samples, n_classes)
        Raw prediction values (in link space).
    raw_prediction_exp : C-contiguous array of shape (n_samples,) or array of \
        shape (n_samples, n_classes)
        Exponential of the raw prediction values.
    xp : module, default=None
        Array namespace module.

    Returns
    -------
    log1pexp : float
        Numerically stable value for log(1 + exp(raw_prediction)).
    """

    # The "magic constants" used here are different for float64 and float32
    # dtypes. For float64, we simply use the values that are present in the
    # Cython loss module and the details can be found there. For float32,
    # we use the `scipy.optimize.brentq` with `xtol=1e-7`to deduce the valid
    # cutoff for each of the different cases that are handled. The trick is
    # to define for each special case a function that subtracts
    # `np.log1p(np.exp(x, dtype=np.float32))` from the special case under
    # consideration. Additionally the resulting values that are very close to
    # zero are set to -1.
    # Consider as an example the case `x + exp(-x)`:
    #
    #     def x_plus_exp_negx(x):
    #         x = np.float32(x)
    #         val = (
    #             x + np.exp(-x, dtype=np.float32))
    #             - np.log1p(np.exp(x, dtype=np.float32)
    #         )
    #         if np.isclose(val, 0, atol=1e-16):
    #             val = -1
    #         return val
    #
    #
    #     x_cutoff = brentq(x_plus_exp_negx, 1, 20, xtol=1e-7)
    #
    # The bounds used in the `brentq` function for each case respectively are
    # acquired through the referenced paper:
    # https://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf
    # Compared to the reference, we have the additional case distinction x <= -2
    # in the float64 case. Since we don't have the reference bounds for this,
    # we estimate the value as approximately x <= -1 for float32.
    constants = (
        [-37, -2, 18, 33.3]
        if raw_prediction.dtype == xp.float64
        else [-17, -1, 9, 14.6]
    )
    return xp.where(
        raw_prediction <= constants[0],
        raw_prediction_exp,
        xp.where(
            raw_prediction <= constants[1],
            xp.log1p(raw_prediction_exp),
            xp.where(
                raw_prediction <= constants[2],
                xp.log(1.0 + raw_prediction_exp),
                xp.where(
                    raw_prediction <= constants[3],
                    raw_prediction + 1 / raw_prediction_exp,
                    raw_prediction,
                ),
            ),
        ),
    )


class HalfBinomialLossArrayAPI(ArrayAPILossMixin, HalfBinomialLoss):
    # [CLASS DOC]
    # Array API version of HalfBinomialLoss: loss and gradient computed in self.xp
    # with the stable _log1pexp formulation; exp(raw_prediction) is shared between
    # loss and gradient in loss_gradient.
    """A version of the HalfBinomialLoss that is compatible with the array API."""

    def loss_gradient(
        self,
        y_true,
        raw_prediction,
        sample_weight=None,
        loss_out=None,
        gradient_out=None,
        n_threads=1,
    ):
        # [FUNCTION DOC]
        # Computes exp(raw_prediction) once and passes it to _compute_loss and
        # _compute_gradient. Returns (loss, gradient); buffers and n_threads ignored.
        raw_prediction_exp = self.xp.exp(raw_prediction)
        loss = self._compute_loss(
            y_true=y_true,
            raw_prediction=raw_prediction,
            sample_weight=sample_weight,
            raw_prediction_exp=raw_prediction_exp,
        )
        gradient = self._compute_gradient(
            y_true=y_true,
            raw_prediction=raw_prediction,
            sample_weight=sample_weight,
            raw_prediction_exp=raw_prediction_exp,
        )
        return loss, gradient

    def _compute_loss(
        self,
        y_true,
        raw_prediction,
        sample_weight=None,
        raw_prediction_exp=None,
    ):
        # [FUNCTION DOC]
        # HELPER: Per-sample loss log(1 + exp(x)) - y * x using _log1pexp; exp(x) is computed
        # if not supplied. Multiplied in place by sample_weight when given.
        # Outputs: array (n_samples,).
        if raw_prediction_exp is None:
            raw_prediction_exp = self.xp.exp(raw_prediction)
        log1pexp = _log1pexp(
            raw_prediction=raw_prediction,
            raw_prediction_exp=raw_prediction_exp,
            xp=self.xp,
        )
        loss = log1pexp - y_true * raw_prediction
        if sample_weight is not None:
            loss *= sample_weight
        return loss

    def _compute_gradient(
        self,
        y_true,
        raw_prediction,
        sample_weight=None,
        raw_prediction_exp=None,
    ):
        # [FUNCTION DOC]
        # HELPER: Per-sample gradient expit(x) - y. For x above a cutoff (-37 for float64,
        # -17 otherwise) it uses ((1 - y) - y * exp(-x)) / (1 + exp(-x)); below it uses
        # exp(x) - y (accurate for very negative x). Multiplied in place by sample_weight.
        # Outputs: array (n_samples,).
        xp = self.xp
        if raw_prediction_exp is None:
            raw_prediction_exp = xp.exp(raw_prediction)
        neg_raw_prediction_exp = 1 / raw_prediction_exp
        grad = xp.where(
            raw_prediction > (-37 if raw_prediction.dtype == xp.float64 else -17),
            ((1 - y_true) - y_true * neg_raw_prediction_exp)
            / (1 + neg_raw_prediction_exp),
            raw_prediction_exp - y_true,
        )
        if sample_weight is not None:
            grad *= sample_weight
        return grad


class HalfMultinomialLossArrayAPI(ArrayAPILossMixin, HalfMultinomialLoss):
    # [CLASS DOC]
    # Array API version of HalfMultinomialLoss. Caches integer labels (y_true_int),
    # flat-index offsets (row index * n_classes) and the one-hot label matrix on the
    # instance across calls.
    # KNOWN ISSUE: caches are filled on first use and never invalidated, so calling the
    # same instance with a different y_true (or different n_samples) silently reuses
    # stale labels/offsets.
    """A version of the HalfMultinomialLoss that is compatible with the array API.

    Parameters
    ----------
    sample_weight : {None, ndarray}
        If sample_weight is None, the hessian might be constant.

    n_classes : {None, int}
        The number of classes for classification, else None.

    xp : module or None
        Array namespace module.

    device : device or None
        A device object.
    """

    def __init__(self, sample_weight=None, n_classes=3, xp=None, device=None):
        # [FUNCTION DOC]
        # Calls HalfMultinomialLoss.__init__ with n_classes, xp and device (sample_weight
        # is dropped) and resets the three caches to None (redundant with the parent).
        super().__init__(n_classes=n_classes, xp=xp, device=device)
        # These instance variables are specifically to store certain
        # intermediate values in order to avoid having to recompute
        # them repeatedly.

        # Used when computing the multinomial loss.
        self.class_indexing_offsets = None
        self.y_true_int = None

        # Used when computing the gradient.
        self.y_true_one_hot = None

    def _compute_loss(
        self,
        y_true,
        raw_prediction,
        sample_weight=None,
    ):
        # [FUNCTION DOC]
        # HELPER: Per-sample loss logsumexp over classes minus the raw prediction of the
        # true class. The true-class entries are gathered from the flattened (row-major)
        # raw_prediction at index label + row * n_classes, using cached int labels and
        # offsets. Multiplied in place by sample_weight if given.
        # Inputs: y_true (n_samples,) label-encoded; raw_prediction (n_samples, n_classes).
        # Outputs: array (n_samples,). Side effects: fills caches on first call.
        xp = self.xp
        device = self.device
        log_sum_exp = _logsumexp(raw_prediction, axis=1, xp=xp)
        if self.y_true_int is None:
            self.y_true_int = xp.asarray(y_true, dtype=xp.int64, device=device)

        if self.class_indexing_offsets is None:
            self.class_indexing_offsets = (
                xp.arange(y_true.shape[0], device=device) * self.n_classes
            )
        true_label_probs = xp.take(
            _ravel(raw_prediction), self.y_true_int + self.class_indexing_offsets
        )
        loss = log_sum_exp - true_label_probs
        if sample_weight is not None:
            loss *= sample_weight
        return loss

    def _compute_gradient(
        self,
        y_true,
        raw_prediction,
        sample_weight=None,
    ):
        # [FUNCTION DOC]
        # HELPER: Gradient softmax(raw_prediction) minus the cached one-hot encoding of
        # y_true (built with array_api_extra one_hot in raw_prediction's dtype), scaled
        # row-wise by sample_weight when given.
        # Outputs: array (n_samples, n_classes). Side effects: fills caches on first call.
        xp = self.xp
        device_ = self.device
        if self.y_true_one_hot is None:
            if self.y_true_int is None:
                self.y_true_int = xp.asarray(y_true, dtype=xp.int64, device=device_)

            self.y_true_one_hot = one_hot(
                self.y_true_int,
                num_classes=self.n_classes,
                dtype=raw_prediction.dtype,
            )
        grad = softmax(raw_prediction)
        # TODO: once incremental assignment for multiple integer array
        # indices is part of a released version of the array API
        # spec and array-api-strict has been updated accordingly,
        # we can further avoid allocating a big (n_samples, n_classes)
        # array for the one-hot encoded y_true and instead use one of the
        # following (the latter should allow for JAX support):
        # grad[xp.arange(y_true.shape[0]), y_true_int] -= 1
        # xpx.at(grad)[xp.arange(y_true.shape[0]), y_true_int].add(-1)
        # See: https://github.com/data-apis/array-api/issues/864
        grad -= self.y_true_one_hot
        if sample_weight is not None:
            grad *= sample_weight[:, None]
        return grad


class HalfPoissonLossArrayAPI(ArrayAPILossMixin, HalfPoissonLoss):
    # [CLASS DOC]
    # Array API version of HalfPoissonLoss (log link), sharing exp(raw_prediction)
    # between loss and gradient. Not exported in the package __init__.
    """A version of the HalfPoissonLoss that is compatible with the array API."""

    def loss_gradient(
        self,
        y_true,
        raw_prediction,
        sample_weight=None,
        loss_out=None,
        gradient_out=None,
        n_threads=1,
    ):
        # [FUNCTION DOC]
        # Computes exp(raw_prediction) once and returns (loss, gradient) from the helpers;
        # buffers and n_threads ignored.
        raw_prediction_exp = self.xp.exp(raw_prediction)
        loss = self._compute_loss(
            y_true=y_true,
            raw_prediction=raw_prediction,
            sample_weight=sample_weight,
            raw_prediction_exp=raw_prediction_exp,
        )
        gradient = self._compute_gradient(
            y_true=y_true,
            raw_prediction=raw_prediction,
            sample_weight=sample_weight,
            raw_prediction_exp=raw_prediction_exp,
        )
        return loss, gradient

    def _compute_loss(
        self,
        y_true,
        raw_prediction,
        sample_weight=None,
        raw_prediction_exp=None,
    ):
        # [FUNCTION DOC]
        # HELPER: Per-sample loss exp(x) - y * x (constant term omitted), multiplied in
        # place by sample_weight when given.
        if raw_prediction_exp is None:
            raw_prediction_exp = self.xp.exp(raw_prediction)
        loss = raw_prediction_exp - y_true * raw_prediction
        if sample_weight is not None:
            loss *= sample_weight
        return loss

    def _compute_gradient(
        self,
        y_true,
        raw_prediction,
        sample_weight=None,
        raw_prediction_exp=None,
    ):
        # [FUNCTION DOC]
        # HELPER: Per-sample gradient exp(x) - y, multiplied in place by sample_weight when
        # given.
        if raw_prediction_exp is None:
            raw_prediction_exp = self.xp.exp(raw_prediction)
        grad = raw_prediction_exp - y_true
        if sample_weight is not None:
            grad *= sample_weight
        return grad
