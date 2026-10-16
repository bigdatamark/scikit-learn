# [MODULE DOC]
# Provides LinearModelLoss, which turns a pointwise BaseLoss (from sklearn._loss) into
# the objective of a linear model (weighted average loss of X @ coef + intercept plus
# an L2 penalty) together with its gradient, full hessian and hessian-vector product,
# for use by scipy / Newton solvers (e.g. in LogisticRegression and GLMs). Also
# provides sandwich_dot used to build hessians.
"""
Loss functions for linear models with raw_prediction = X @ coef
"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
from scipy import sparse

from sklearn.utils._array_api import (
    get_namespace,
    get_namespace_and_device,
    move_to,
)
from sklearn.utils._sparse import _align_api_if_sparse
from sklearn.utils.extmath import safe_sparse_dot, squared_norm


def sandwich_dot(X, W):
    # [FUNCTION DOC]
    # HELPER: Computes the matrix X transposed times diag(W) times X.
    # For sparse X, multiplies X on the left by a sparse diagonal matrix built from W,
    # then by X transposed with a dense result (aligned to the sparse array API).
    # For dense X, scales each row of X by its weight and multiplies by X transposed
    # (BLAS GEMM).
    # Inputs: X - dense or sparse (n_samples, n_features); W - 1-D (n_samples,).
    # Outputs: dense ndarray (n_features, n_features).
    # Side effects: none.
    """Compute the sandwich product X.T @ diag(W) @ X."""
    # TODO: This "sandwich product" is the main computational bottleneck for solvers
    # that use the full hessian matrix. Here, thread parallelism would pay-off the
    # most.
    # While a dedicated Cython routine could exploit the symmetry, it is very hard to
    # beat BLAS GEMM, even thought the latter cannot exploit the symmetry, unless one
    # pays the price of taking square roots and implements
    #    sqrtWX = sqrt(W)[: None] * X
    #    return sqrtWX.T @ sqrtWX
    # which (might) detect the symmetry and use BLAS SYRK under the hood.
    n_samples = X.shape[0]
    if sparse.issparse(X):
        return _align_api_if_sparse(
            safe_sparse_dot(
                X.T,
                sparse.dia_array((W, 0), shape=(n_samples, n_samples)) @ X,
                dense_output=True,
            )
        )
    else:
        # np.einsum may use less memory but the following, using BLAS matrix
        # multiplication (GEMM), is by far faster.
        WX = W[:, None] * X
        return X.T @ WX


class LinearModelLoss:
    # [CLASS DOC]
    # Objective wrapper for linear models; holds a base_loss and the fit_intercept flag.
    # Coefficient layout conventions (intercept last; multiclass coef either 2-D
    # (n_classes, n_dof) or raveled in Fortran order so classes are contiguous) are given
    # in the class docstring and used by every method.
    """General class for loss functions with raw_prediction = X @ coef + intercept.

    Note that raw_prediction is also known as linear predictor.

    The loss is the average of per sample losses and includes a term for L2
    regularization::

        loss = 1 / s_sum * sum_i s_i loss(y_i, X_i @ coef + intercept)
               + 1/2 * l2_reg_strength * ||coef||_2^2

    with sample weights s_i=1 if sample_weight=None and s_sum=sum_i s_i.

    Gradient and hessian, for simplicity without intercept, are::

        gradient = 1 / s_sum * X.T @ loss.gradient + l2_reg_strength * coef
        hessian = 1 / s_sum * X.T @ diag(loss.hessian) @ X
                  + l2_reg_strength * identity

    Conventions:
        if fit_intercept:
            n_dof =  n_features + 1
        else:
            n_dof = n_features

        if base_loss.is_multiclass:
            coef.shape = (n_classes, n_dof) or ravelled (n_classes * n_dof,)
        else:
            coef.shape = (n_dof,)

        The intercept term is at the end of the coef array:
        if base_loss.is_multiclass:
            if coef.shape (n_classes, n_dof):
                intercept = coef[:, -1]
            if coef.shape (n_classes * n_dof,)
                intercept = coef[n_classes * n_features:] = coef[(n_dof-1):]
            intercept.shape = (n_classes,)
        else:
            intercept = coef[-1]

        Shape of gradient follows shape of coef.
        gradient.shape = coef.shape

        But hessian (to make our lives simpler) are always 2-d:
        if base_loss.is_multiclass:
            hessian.shape = (n_classes * n_dof, n_classes * n_dof)
        else:
            hessian.shape = (n_dof, n_dof)

    Note: if coef has shape (n_classes * n_dof,), the classes are expected to be
    contiguous, i.e. the 2d-array can be reconstructed as

        coef.reshape((n_classes, -1), order="F")

    The option order="F" makes coef[:, i] contiguous. This, in turn, makes the
    coefficients without intercept, coef[:, :-1], contiguous and speeds up
    matrix-vector computations.

    Note: If the average loss per sample is wanted instead of the sum of the loss per
    sample, one can simply use a rescaled sample_weight such that
    sum(sample_weight) = 1.

    Parameters
    ----------
    base_loss : instance of class BaseLoss from sklearn._loss.
    fit_intercept : bool
    """

    def __init__(self, base_loss, fit_intercept):
        # [FUNCTION DOC]
        # Stores base_loss (BaseLoss instance) and fit_intercept (bool). No validation.
        self.base_loss = base_loss
        self.fit_intercept = fit_intercept

    def init_zero_coef(self, X, dtype=None):
        # [FUNCTION DOC]
        # Allocates an all-zero coefficient array with n_dof = n_features (+1 with intercept).
        # Inputs: X - only its number of columns is used; dtype - array dtype, default None.
        # Outputs: ndarray (n_classes, n_dof) in Fortran order for multiclass losses, else
        # (n_dof,).
        # KNOWN ISSUE: the docstring says dtype=None means "same dtype as X", but the code
        # passes dtype straight to np.zeros, so None yields float64 regardless of X.
        """Allocate coef of correct shape with zeros.

        Parameters:
        -----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training data.
        dtype : data-type, default=None
            Overrides the data type of coef. With dtype=None, coef will have the same
            dtype as X.

        Returns
        -------
        coef : ndarray of shape (n_dof,) or (n_classes, n_dof)
            Coefficients of a linear model.
        """
        n_features = X.shape[1]
        n_classes = self.base_loss.n_classes
        if self.fit_intercept:
            n_dof = n_features + 1
        else:
            n_dof = n_features
        if self.base_loss.is_multiclass:
            coef = np.zeros(shape=(n_classes, n_dof), dtype=dtype, order="F")
        else:
            coef = np.zeros(shape=n_dof, dtype=dtype)
        return coef

    def weight_intercept(self, coef):
        # [FUNCTION DOC]
        # Splits coef into weights (without intercept) and intercept.
        # Non-multiclass: intercept is the last entry when fit_intercept, otherwise 0.0 and
        # weights is coef itself. Multiclass: a 1-D coef is reshaped (Fortran order) to
        # (n_classes, n_dof); intercept is the last column or 0.0.
        # Inputs: coef in any of the documented layouts.
        # Outputs: (weights, intercept); weights are views of coef, not copies.
        """Helper function to get coefficients and intercept.

        Parameters
        ----------
        coef : ndarray of shape (n_dof,), (n_classes, n_dof) or (n_classes * n_dof,)
            Coefficients of a linear model.
            If shape (n_classes * n_dof,), the classes of one feature are contiguous,
            i.e. one reconstructs the 2d-array via
            coef.reshape((n_classes, -1), order="F").

        Returns
        -------
        weights : ndarray of shape (n_features,) or (n_classes, n_features)
            Coefficients without intercept term.
        intercept : float or ndarray of shape (n_classes,)
            Intercept terms.
        """
        if not self.base_loss.is_multiclass:
            if self.fit_intercept:
                intercept = coef[-1]
                weights = coef[:-1]
            else:
                intercept = 0.0
                weights = coef
        else:
            # reshape to (n_classes, n_dof)
            if coef.ndim == 1:
                weights = coef.reshape((self.base_loss.n_classes, -1), order="F")
            else:
                weights = coef
            if self.fit_intercept:
                intercept = weights[:, -1]
                weights = weights[:, :-1]
            else:
                intercept = 0.0

        return weights, intercept

    def weight_intercept_raw(self, coef, X):
        # [FUNCTION DOC]
        # Splits coef via weight_intercept and computes the raw (linear) predictions
        # X times weights (transposed for multiclass) plus intercept. weights and intercept
        # are converted to X's array namespace/device/dtype for the product only.
        # Inputs: coef; X (n_samples, n_features), dense, sparse or array-API array.
        # Outputs: (weights, intercept, raw_prediction) where weights/intercept are the
        # original numpy objects and raw_prediction is (n_samples,) or (n_samples, n_classes)
        # in X's namespace.
        """Helper function to get coefficients, intercept and raw_prediction.

        Parameters
        ----------
        coef : ndarray of shape (n_dof,), (n_classes, n_dof) or (n_classes * n_dof,)
            Coefficients of a linear model.
            If shape (n_classes * n_dof,), the classes of one feature are contiguous,
            i.e. one reconstructs the 2d-array via
            coef.reshape((n_classes, -1), order="F").
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training data.

        Returns
        -------
        weights : ndarray of shape (n_features,) or (n_classes, n_features)
            Coefficients without intercept term.
        intercept : float or ndarray of shape (n_classes,)
            Intercept terms.
        raw_prediction : ndarray of shape (n_samples,) or \
            (n_samples, n_classes)
        """
        weights, intercept = self.weight_intercept(coef)
        xp, _, device_ = get_namespace_and_device(X)

        # The `weights` and `intercept` are only converted internally to the
        # array API because the relevant `scipy.optimize` functions do not
        # currently support the array API and we have to ensure that the final
        # values returned to the respective `scipy.optimize` function are in
        # the `numpy` namespace.
        weights_xp = xp.asarray(weights, dtype=X.dtype, device=device_)
        intercept_xp = xp.asarray(intercept, dtype=X.dtype, device=device_)
        if not self.base_loss.is_multiclass:
            raw_prediction = X @ weights_xp + intercept_xp
        else:
            # weights has shape (n_classes, n_dof)
            raw_prediction = X @ weights_xp.T + intercept_xp

        return weights, intercept, raw_prediction

    def l2_penalty(self, weights, l2_reg_strength):
        # [FUNCTION DOC]
        # Returns 0.5 * l2_reg_strength * squared Frobenius / Euclidean norm of weights as a
        # Python float. Inputs: weights 1-D or 2-D array; l2_reg_strength float.
        """Compute L2 penalty term l2_reg_strength/2 *||w||_2^2."""
        norm2_w = weights @ weights if weights.ndim == 1 else squared_norm(weights)
        return float(0.5 * l2_reg_strength * norm2_w)

    def loss(
        self,
        coef,
        X,
        y,
        sample_weight=None,
        l2_reg_strength=0.0,
        n_threads=1,
        raw_prediction=None,
    ):
        # [FUNCTION DOC]
        # Computes the objective: sum of pointwise (weighted) losses divided by the weight
        # sum (n_samples if unweighted) plus the L2 penalty when l2_reg_strength > 0.
        # Uses raw_prediction if given, otherwise computes it from coef and X.
        # Inputs: coef; X; y (n_samples,); sample_weight or None; l2_reg_strength float;
        # n_threads int; raw_prediction optional precomputed linear predictor.
        # Outputs: float loss.
        """Compute the loss as weighted average over point-wise losses.

        Parameters
        ----------
        coef : ndarray of shape (n_dof,), (n_classes, n_dof) or (n_classes * n_dof,)
            Coefficients of a linear model.
            If shape (n_classes * n_dof,), the classes of one feature are contiguous,
            i.e. one reconstructs the 2d-array via
            coef.reshape((n_classes, -1), order="F").
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training data.
        y : contiguous array of shape (n_samples,)
            Observed, true target values.
        sample_weight : None or contiguous array of shape (n_samples,), default=None
            Sample weights.
        l2_reg_strength : float, default=0.0
            L2 regularization strength
        n_threads : int, default=1
            Number of OpenMP threads to use.
        raw_prediction : C-contiguous array of shape (n_samples,) or array of \
            shape (n_samples, n_classes)
            Raw prediction values (in link space). If provided, these are used. If
            None, then raw_prediction = X @ coef + intercept is calculated.

        Returns
        -------
        loss : float
            Weighted average of losses per sample, plus penalty.
        """
        n_samples = X.shape[0]
        if raw_prediction is None:
            weights, intercept, raw_prediction = self.weight_intercept_raw(coef, X)
        else:
            weights, intercept = self.weight_intercept(coef)

        loss = self.base_loss.loss(
            y_true=y,
            raw_prediction=raw_prediction,
            sample_weight=sample_weight,
            n_threads=n_threads,
        )
        xp, _ = get_namespace(X, y, sample_weight)
        sw_sum = n_samples if sample_weight is None else xp.sum(sample_weight)
        loss = float(xp.sum(loss) / sw_sum)

        if l2_reg_strength > 0:
            loss += self.l2_penalty(weights, l2_reg_strength)

        return loss

    def loss_gradient(
        self,
        coef,
        X,
        y,
        sample_weight=None,
        l2_reg_strength=0.0,
        n_threads=1,
        raw_prediction=None,
    ):
        # [FUNCTION DOC]
        # Computes the objective (as in loss) and its gradient with respect to coef in one
        # pass. Pointwise gradients are divided by the weight sum. Non-multiclass: feature
        # part is X transposed times pointwise gradient plus l2_reg_strength * weights,
        # intercept part is the sum of pointwise gradients. Multiclass: same per class
        # (pointwise gradient transposed times X), assembled into an (n_classes, n_dof)
        # Fortran array and raveled if coef was 1-D. Results are moved back to numpy/CPU.
        # Inputs: same as loss.
        # Outputs: (loss float, gradient ndarray with coef's shape).
        # Side effects: none on inputs (pointwise gradient array is a fresh temporary).
        """Computes the sum of loss and gradient w.r.t. coef.

        Parameters
        ----------
        coef : ndarray of shape (n_dof,), (n_classes, n_dof) or (n_classes * n_dof,)
            Coefficients of a linear model.
            If shape (n_classes * n_dof,), the classes of one feature are contiguous,
            i.e. one reconstructs the 2d-array via
            coef.reshape((n_classes, -1), order="F").
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training data.
        y : contiguous array of shape (n_samples,)
            Observed, true target values.
        sample_weight : None or contiguous array of shape (n_samples,), default=None
            Sample weights.
        l2_reg_strength : float, default=0.0
            L2 regularization strength
        n_threads : int, default=1
            Number of OpenMP threads to use.
        raw_prediction : C-contiguous array of shape (n_samples,) or array of \
            shape (n_samples, n_classes)
            Raw prediction values (in link space). If provided, these are used. If
            None, then raw_prediction = X @ coef + intercept is calculated.

        Returns
        -------
        loss : float
            Weighted average of losses per sample, plus penalty.

        gradient : ndarray of shape coef.shape
             The gradient of the loss.
        """
        (n_samples, n_features), n_classes = X.shape, self.base_loss.n_classes
        n_dof = n_features + int(self.fit_intercept)

        if raw_prediction is None:
            weights, intercept, raw_prediction = self.weight_intercept_raw(coef, X)
        else:
            weights, intercept = self.weight_intercept(coef)

        loss, grad_pointwise = self.base_loss.loss_gradient(
            y_true=y,
            raw_prediction=raw_prediction,
            sample_weight=sample_weight,
            n_threads=n_threads,
        )
        xp, _ = get_namespace(X, y, sample_weight)
        sw_sum = n_samples if sample_weight is None else xp.sum(sample_weight)
        loss = float(xp.sum(loss) / sw_sum)
        loss += self.l2_penalty(weights, l2_reg_strength)

        grad_pointwise /= sw_sum

        if not self.base_loss.is_multiclass:
            grad = np.empty_like(coef, dtype=weights.dtype)
            X_grad = X.T @ grad_pointwise
            grad[:n_features] = (
                move_to(X_grad, xp=np, device="cpu") + l2_reg_strength * weights
            )
            if self.fit_intercept:
                grad[-1] = xp.sum(grad_pointwise)
        else:
            # The final value of `grad` needs to be in the `numpy` namespace
            # because the relevant `scipy.optimize` functions do not currently
            # support the array API.
            grad = np.empty((n_classes, n_dof), dtype=weights.dtype, order="F")
            # grad_pointwise.shape = (n_samples, n_classes)
            grad_X = grad_pointwise.T @ X
            grad[:, :n_features] = (
                move_to(grad_X, xp=np, device="cpu") + l2_reg_strength * weights
            )
            if self.fit_intercept:
                grad[:, -1] = move_to(
                    xp.sum(grad_pointwise, axis=0), xp=np, device="cpu"
                )
            if coef.ndim == 1:
                grad = grad.ravel(order="F")

        return loss, grad

    def gradient(
        self,
        coef,
        X,
        y,
        sample_weight=None,
        l2_reg_strength=0.0,
        n_threads=1,
        raw_prediction=None,
    ):
        # [FUNCTION DOC]
        # Same gradient as loss_gradient but without computing the loss value, and without
        # the array-API conversions (assumes numpy inputs).
        # Inputs: same as loss. Outputs: gradient ndarray with coef's shape.
        """Computes the gradient w.r.t. coef.

        Parameters
        ----------
        coef : ndarray of shape (n_dof,), (n_classes, n_dof) or (n_classes * n_dof,)
            Coefficients of a linear model.
            If shape (n_classes * n_dof,), the classes of one feature are contiguous,
            i.e. one reconstructs the 2d-array via
            coef.reshape((n_classes, -1), order="F").
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training data.
        y : contiguous array of shape (n_samples,)
            Observed, true target values.
        sample_weight : None or contiguous array of shape (n_samples,), default=None
            Sample weights.
        l2_reg_strength : float, default=0.0
            L2 regularization strength
        n_threads : int, default=1
            Number of OpenMP threads to use.
        raw_prediction : C-contiguous array of shape (n_samples,) or array of \
            shape (n_samples, n_classes)
            Raw prediction values (in link space). If provided, these are used. If
            None, then raw_prediction = X @ coef + intercept is calculated.

        Returns
        -------
        gradient : ndarray of shape coef.shape
             The gradient of the loss.
        """
        (n_samples, n_features), n_classes = X.shape, self.base_loss.n_classes
        n_dof = n_features + int(self.fit_intercept)

        if raw_prediction is None:
            weights, intercept, raw_prediction = self.weight_intercept_raw(coef, X)
        else:
            weights, intercept = self.weight_intercept(coef)

        grad_pointwise = self.base_loss.gradient(
            y_true=y,
            raw_prediction=raw_prediction,
            sample_weight=sample_weight,
            n_threads=n_threads,
        )
        sw_sum = n_samples if sample_weight is None else np.sum(sample_weight)
        grad_pointwise /= sw_sum

        if not self.base_loss.is_multiclass:
            grad = np.empty_like(coef, dtype=weights.dtype)
            grad[:n_features] = X.T @ grad_pointwise + l2_reg_strength * weights
            if self.fit_intercept:
                grad[-1] = grad_pointwise.sum()
            return grad
        else:
            grad = np.empty((n_classes, n_dof), dtype=weights.dtype, order="F")
            # gradient.shape = (n_samples, n_classes)
            grad[:, :n_features] = grad_pointwise.T @ X + l2_reg_strength * weights
            if self.fit_intercept:
                grad[:, -1] = grad_pointwise.sum(axis=0)
            if coef.ndim == 1:
                return grad.ravel(order="F")
            else:
                return grad

    def gradient_hessian(
        self,
        coef,
        X,
        y,
        sample_weight=None,
        l2_reg_strength=0.0,
        n_threads=1,
        gradient_out=None,
        hessian_out=None,
        raw_prediction=None,
    ):
        # [FUNCTION DOC]
        # Computes the gradient and the full 2-D hessian of the objective.
        # Validates or allocates the output buffers (gradient must match coef.shape, F-ordered
        # for multiclass; hessian must be (coef.size, coef.size), contiguous for multiclass).
        # Non-multiclass: pointwise gradient and hessian are divided by the weight sum; the
        # hessian warning flag is set if more than 25% (weighted) of pointwise hessians are
        # non-positive, and absolute values are used. Gradient is computed as in
        # loss_gradient; if the warning is set the function returns early without filling the
        # hessian. Otherwise the feature block is sandwich_dot(X, h), the L2 strength is added
        # to the feature diagonal (through a flat strided view), and with intercept the last
        # row/column are X transposed times h and the corner is the sum of h.
        # Multiclass (multinomial assumed): uses predicted probabilities; for each class pair
        # (k, l) builds the block X' diag(h_kl) X with h_kk = p_k(1-p_k)*sw and
        # h_kl = -p_k p_l * sw, written with stride n_classes so classes are interleaved per
        # feature, plus intercept rows/columns; mirrors the upper class triangle to the lower;
        # adds L2 to the feature diagonal. Warning flag is always False.
        # Inputs: as loss, plus gradient_out / hessian_out optional preallocated arrays.
        # Outputs: (gradient, hessian 2-D (n, n), hessian_warning bool).
        # Exceptions: ValueError for wrongly shaped or non-contiguous output buffers.
        # Side effects: writes into gradient_out / hessian_out when provided.
        # KNOWN ISSUE: the docstring gives the multiclass hessian shape as
        # (n_classes, n_dof, n_dof, n_classes) but the returned array is always 2-D.
        # KNOWN ISSUE: on the early return with hessian_warning True, the returned hessian is
        # uninitialised memory (np.empty) or the untouched hessian_out.
        # KNOWN ISSUE: the hessian_out shape error message formats the expected shape as a
        # nested tuple string.
        """Computes gradient and hessian w.r.t. coef.

        Parameters
        ----------
        coef : ndarray of shape (n_dof,), (n_classes, n_dof) or (n_classes * n_dof,)
            Coefficients of a linear model.
            If shape (n_classes * n_dof,), the classes of one feature are contiguous,
            i.e. one reconstructs the 2d-array via
            coef.reshape((n_classes, -1), order="F").
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training data.
        y : contiguous array of shape (n_samples,)
            Observed, true target values.
        sample_weight : None or contiguous array of shape (n_samples,), default=None
            Sample weights.
        l2_reg_strength : float, default=0.0
            L2 regularization strength
        n_threads : int, default=1
            Number of OpenMP threads to use.
        gradient_out : None or ndarray of shape coef.shape
            A location into which the gradient is stored. If None, a new array
            might be created.
        hessian_out : None or ndarray of shape (n_dof, n_dof) or \
            (n_classes * n_dof, n_classes * n_dof)
            A location into which the hessian is stored. If None, a new array
            might be created.
        raw_prediction : C-contiguous array of shape (n_samples,) or array of \
            shape (n_samples, n_classes)
            Raw prediction values (in link space). If provided, these are used. If
            None, then raw_prediction = X @ coef + intercept is calculated.

        Returns
        -------
        gradient : ndarray of shape coef.shape
             The gradient of the loss.

        hessian : ndarray of shape (n_dof, n_dof) or \
            (n_classes, n_dof, n_dof, n_classes)
            Hessian matrix.

        hessian_warning : bool
            True if pointwise hessian has more than 25% of its elements non-positive.
        """
        (n_samples, n_features), n_classes = X.shape, self.base_loss.n_classes
        n_dof = n_features + int(self.fit_intercept)
        if raw_prediction is None:
            weights, intercept, raw_prediction = self.weight_intercept_raw(coef, X)
        else:
            weights, intercept = self.weight_intercept(coef)
        sw_sum = n_samples if sample_weight is None else np.sum(sample_weight)

        # Allocate gradient.
        if gradient_out is None:
            grad = np.empty_like(coef, dtype=weights.dtype, order="F")
        elif gradient_out.shape != coef.shape:
            raise ValueError(
                f"gradient_out is required to have shape coef.shape = {coef.shape}; "
                f"got {gradient_out.shape}."
            )
        elif self.base_loss.is_multiclass and not gradient_out.flags.f_contiguous:
            raise ValueError("gradient_out must be F-contiguous.")
        else:
            grad = gradient_out
        # Allocate hessian.
        n = coef.size  # for multinomial this equals n_dof * n_classes
        if hessian_out is None:
            hess = np.empty((n, n), dtype=weights.dtype)
        elif hessian_out.shape != (n, n):
            raise ValueError(
                f"hessian_out is required to have shape ({n, n}); got "
                f"{hessian_out.shape=}."
            )
        elif self.base_loss.is_multiclass and (
            not hessian_out.flags.c_contiguous and not hessian_out.flags.f_contiguous
        ):
            raise ValueError("hessian_out must be contiguous.")
        else:
            hess = hessian_out

        if not self.base_loss.is_multiclass:
            grad_pointwise, hess_pointwise = self.base_loss.gradient_hessian(
                y_true=y,
                raw_prediction=raw_prediction,
                sample_weight=sample_weight,
                n_threads=n_threads,
            )
            grad_pointwise /= sw_sum
            hess_pointwise /= sw_sum

            # For non-canonical link functions and far away from the optimum, the
            # pointwise hessian can be negative. We take care that 75% of the hessian
            # entries are positive.
            hessian_warning = (
                np.average(hess_pointwise <= 0, weights=sample_weight) > 0.25
            )
            hess_pointwise = np.abs(hess_pointwise)

            grad[:n_features] = X.T @ grad_pointwise + l2_reg_strength * weights
            if self.fit_intercept:
                grad[-1] = grad_pointwise.sum()

            if hessian_warning:
                # Exit early without computing the hessian.
                return grad, hess, hessian_warning

            hess[:n_features, :n_features] = sandwich_dot(X, hess_pointwise)

            if l2_reg_strength > 0:
                # The L2 penalty enters the Hessian on the diagonal only. To add those
                # terms, we use a flattened view of the array.
                order = "C" if hess.flags.c_contiguous else "F"
                hess.reshape(-1, order=order)[: (n_features * n_dof) : (n_dof + 1)] += (
                    l2_reg_strength
                )

            if self.fit_intercept:
                # With intercept included as added column to X, the hessian becomes
                # hess = (X, 1)' @ diag(h) @ (X, 1)
                #      = (X' @ diag(h) @ X, X' @ h)
                #        (           h @ X, sum(h))
                # The left upper part has already been filled, it remains to compute
                # the last row and the last column.
                Xh = X.T @ hess_pointwise
                hess[:-1, -1] = Xh
                hess[-1, :-1] = Xh
                hess[-1, -1] = hess_pointwise.sum()
        else:
            # Here we may safely assume HalfMultinomialLoss aka categorical
            # cross-entropy.
            # HalfMultinomialLoss computes only the diagonal part of the hessian, i.e.
            # diagonal in the classes. Here, we want the full hessian. Therefore, we
            # call gradient_proba.
            grad_pointwise, proba = self.base_loss.gradient_proba(
                y_true=y,
                raw_prediction=raw_prediction,
                sample_weight=sample_weight,
                n_threads=n_threads,
            )
            grad_pointwise /= sw_sum
            grad = grad.reshape((n_classes, n_dof), order="F")
            grad[:, :n_features] = grad_pointwise.T @ X + l2_reg_strength * weights
            if self.fit_intercept:
                grad[:, -1] = grad_pointwise.sum(axis=0)
            if coef.ndim == 1:
                grad = grad.ravel(order="F")

            # The full hessian matrix, i.e. not only the diagonal part, dropping most
            # indices, is given by:
            #
            #   hess = X' @ h @ X
            #
            # Here, h is a priori a 4-dimensional matrix of shape
            # (n_samples, n_samples, n_classes, n_classes). It is diagonal its first
            # two dimensions (the ones with n_samples), i.e. it is
            # effectively a 3-dimensional matrix (n_samples, n_classes, n_classes).
            #
            #   h = diag(p) - p' p
            #
            # or with indices k and l for classes
            #
            #   h_kl = p_k * delta_kl - p_k * p_l
            #
            # with p_k the (predicted) probability for class k. Only the dimension in
            # n_samples multiplies with X.
            # For 3 classes and n_samples = 1, this looks like ("@" is a bit misused
            # here):
            #
            #   hess = X' @ (h00 h10 h20) @ X
            #               (h10 h11 h12)
            #               (h20 h12 h22)
            #        = (X' @ diag(h00) @ X, X' @ diag(h10), X' @ diag(h20))
            #          (X' @ diag(h10) @ X, X' @ diag(h11), X' @ diag(h12))
            #          (X' @ diag(h20) @ X, X' @ diag(h12), X' @ diag(h22))
            #
            # Now coef of shape (n_classes * n_dof) is contiguous in n_classes.
            # Therefore, we want the hessian to follow this convention, too, i.e.
            #     hess[:n_classes, :n_classes] = (x0' @ h00 @ x0, x0' @ h10 @ x0, ..)
            #                                    (x0' @ h10 @ x0, x0' @ h11 @ x0, ..)
            #                                    (x0' @ h20 @ x0, x0' @ h12 @ x0, ..)
            # is the first feature, x0, for all classes. In our implementation, we
            # still want to take advantage of BLAS "X.T @ X". Therefore, we have some
            # index/slicing battle to fight.
            if sample_weight is not None:
                sw = sample_weight / sw_sum
            else:
                sw = 1.0 / sw_sum

            for k in range(n_classes):
                # Diagonal terms (in classes) hess_kk.
                # Note that this also writes to some of the lower triangular part.
                h = proba[:, k] * (1 - proba[:, k]) * sw
                hess[
                    k : n_classes * n_features : n_classes,
                    k : n_classes * n_features : n_classes,
                ] = sandwich_dot(X, h)
                if self.fit_intercept:
                    # See above in the non multiclass case.
                    Xh = X.T @ h
                    hess[
                        k : n_classes * n_features : n_classes,
                        n_classes * n_features + k,
                    ] = Xh
                    hess[
                        n_classes * n_features + k,
                        k : n_classes * n_features : n_classes,
                    ] = Xh
                    hess[n_classes * n_features + k, n_classes * n_features + k] = (
                        h.sum()
                    )
                # Off diagonal terms (in classes) hess_kl.
                for l in range(k + 1, n_classes):
                    # Upper triangle (in classes).
                    h = -proba[:, k] * proba[:, l] * sw
                    hess[
                        k : n_classes * n_features : n_classes,
                        l : n_classes * n_features : n_classes,
                    ] = sandwich_dot(X, h)
                    if self.fit_intercept:
                        Xh = X.T @ h
                        hess[
                            k : n_classes * n_features : n_classes,
                            n_classes * n_features + l,
                        ] = Xh
                        hess[
                            n_classes * n_features + k,
                            l : n_classes * n_features : n_classes,
                        ] = Xh
                        hess[n_classes * n_features + k, n_classes * n_features + l] = (
                            h.sum()
                        )
                    # Fill lower triangle (in classes).
                    hess[l::n_classes, k::n_classes] = hess[k::n_classes, l::n_classes]

            if l2_reg_strength > 0:
                # See above in the non multiclass case.
                order = "C" if hess.flags.c_contiguous else "F"
                hess.reshape(-1, order=order)[
                    : (n_classes**2 * n_features * n_dof) : (n_classes * n_dof + 1)
                ] += l2_reg_strength

            # The pointwise hessian is always non-negative for the multinomial loss.
            hessian_warning = False

        return grad, hess, hessian_warning

    def gradient_hessian_product(
        self, coef, X, y, sample_weight=None, l2_reg_strength=0.0, n_threads=1
    ):
        # [FUNCTION DOC]
        # Computes the gradient and returns a closure hessp(s) computing the hessian-vector
        # product without forming the hessian. Gradient is computed as in loss_gradient.
        # Non-multiclass: precomputes h (pointwise hessian / weight sum), hX (rows of X
        # scaled by h), sum of h and column sums of hX for the intercept.
        # Multiclass: precomputes the probabilities; gradient raveled in Fortran order when
        # coef is 1-D.
        # Inputs: coef, X, y, sample_weight, l2_reg_strength, n_threads (as in loss).
        # Outputs: (gradient with coef's shape, hessp callable).
        # Side effects: the closure captures X, proba/hX and other arrays (memory held while
        # the closure lives).
        """Computes gradient and hessp (hessian product function) w.r.t. coef.

        Parameters
        ----------
        coef : ndarray of shape (n_dof,), (n_classes, n_dof) or (n_classes * n_dof,)
            Coefficients of a linear model.
            If shape (n_classes * n_dof,), the classes of one feature are contiguous,
            i.e. one reconstructs the 2d-array via
            coef.reshape((n_classes, -1), order="F").
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training data.
        y : contiguous array of shape (n_samples,)
            Observed, true target values.
        sample_weight : None or contiguous array of shape (n_samples,), default=None
            Sample weights.
        l2_reg_strength : float, default=0.0
            L2 regularization strength
        n_threads : int, default=1
            Number of OpenMP threads to use.

        Returns
        -------
        gradient : ndarray of shape coef.shape
             The gradient of the loss.

        hessp : callable
            Function that takes in a vector input of shape of gradient and
            and returns matrix-vector product with hessian.
        """
        (n_samples, n_features), n_classes = X.shape, self.base_loss.n_classes
        n_dof = n_features + int(self.fit_intercept)
        weights, intercept, raw_prediction = self.weight_intercept_raw(coef, X)
        sw_sum = n_samples if sample_weight is None else np.sum(sample_weight)

        if not self.base_loss.is_multiclass:
            grad_pointwise, hess_pointwise = self.base_loss.gradient_hessian(
                y_true=y,
                raw_prediction=raw_prediction,
                sample_weight=sample_weight,
                n_threads=n_threads,
            )
            grad_pointwise /= sw_sum
            hess_pointwise /= sw_sum
            grad = np.empty_like(coef, dtype=weights.dtype)
            grad[:n_features] = X.T @ grad_pointwise + l2_reg_strength * weights
            if self.fit_intercept:
                grad[-1] = grad_pointwise.sum()

            # Precompute as much as possible: hX, hX_sum and hessian_sum
            hessian_sum = hess_pointwise.sum()
            if sparse.issparse(X):
                hX = (
                    sparse.dia_array((hess_pointwise, 0), shape=(n_samples, n_samples))
                    @ X
                )
            else:
                hX = hess_pointwise[:, np.newaxis] * X

            if self.fit_intercept:
                # Calculate the double derivative with respect to intercept.
                # Note: In case hX is sparse, hX.sum is a matrix object.
                hX_sum = np.squeeze(np.asarray(hX.sum(axis=0)))
                # prevent squeezing to zero-dim array if n_features == 1
                hX_sum = np.atleast_1d(hX_sum)

            # With intercept included and l2_reg_strength = 0, hessp returns
            # res = (X, 1)' @ diag(h) @ (X, 1) @ s
            #     = (X, 1)' @ (hX @ s[:n_features], sum(h) * s[-1])
            # res[:n_features] = X' @ hX @ s[:n_features] + sum(h) * s[-1]
            # res[-1] = 1' @ hX @ s[:n_features] + sum(h) * s[-1]
            def hessp(s):
                # [FUNCTION DOC]
                # HELPER: Non-multiclass hessian-vector product. Feature part: X transposed times
                # (hX times the feature part of s) plus l2_reg_strength times that part; with
                # intercept adds s_last times column sums of hX, and the last entry is column sums of
                # hX dotted with the feature part plus sum(h) times s_last.
                # Inputs: s - 1-D vector of length n_dof. Outputs: new vector of the same shape.
                ret = np.empty_like(s)
                if sparse.issparse(X):
                    ret[:n_features] = X.T @ (hX @ s[:n_features])
                else:
                    ret[:n_features] = np.linalg.multi_dot([X.T, hX, s[:n_features]])
                ret[:n_features] += l2_reg_strength * s[:n_features]

                if self.fit_intercept:
                    ret[:n_features] += s[-1] * hX_sum
                    ret[-1] = hX_sum @ s[:n_features] + hessian_sum * s[-1]
                return ret

        else:
            # Here we may safely assume HalfMultinomialLoss aka categorical
            # cross-entropy.
            # HalfMultinomialLoss computes only the diagonal part of the hessian, i.e.
            # diagonal in the classes. Here, we want the matrix-vector product of the
            # full hessian. Therefore, we call gradient_proba.
            grad_pointwise, proba = self.base_loss.gradient_proba(
                y_true=y,
                raw_prediction=raw_prediction,
                sample_weight=sample_weight,
                n_threads=n_threads,
            )
            grad_pointwise /= sw_sum
            grad = np.empty((n_classes, n_dof), dtype=weights.dtype, order="F")
            grad[:, :n_features] = grad_pointwise.T @ X + l2_reg_strength * weights
            if self.fit_intercept:
                grad[:, -1] = grad_pointwise.sum(axis=0)

            # Full hessian-vector product, i.e. not only the diagonal part of the
            # hessian. Derivation with some index battle for input vector s:
            #   - sample index i
            #   - feature indices j, m
            #   - class indices k, l
            #   - 1_{k=l} is one if k=l else 0
            #   - p_i_k is the (predicted) probability that sample i belongs to class k
            #     for all i: sum_k p_i_k = 1
            #   - s_l_m is input vector for class l and feature m
            #   - X' = X transposed
            #
            # Note: Hessian with dropping most indices is just:
            #       X' @ p_k (1(k=l) - p_l) @ X
            #
            # result_{k j} = sum_{i, l, m} Hessian_{i, k j, m l} * s_l_m
            #   = sum_{i, l, m} (X')_{ji} * p_i_k * (1_{k=l} - p_i_l)
            #                   * X_{im} s_l_m
            #   = sum_{i, m} (X')_{ji} * p_i_k
            #                * (X_{im} * s_k_m - sum_l p_i_l * X_{im} * s_l_m)
            #
            # See also https://github.com/scikit-learn/scikit-learn/pull/3646#discussion_r17461411
            def hessp(s):
                # [FUNCTION DOC]
                # HELPER: Multinomial hessian-vector product. Reshapes s to (n_classes, n_dof)
                # (Fortran order), splits the intercept column, computes for each sample
                # t = X s' + s_intercept, subtracts the probability-weighted row mean of t, multiplies
                # by the probabilities and the sample weights, then returns per class
                # (t' X) / weight_sum + l2_reg_strength * s and the intercept part as column sums of t
                # divided by the weight sum.
                # Inputs: s - flattened (n_classes * n_dof,) or 2-D vector.
                # Outputs: product with the same layout as coef (raveled in Fortran order when coef
                # was 1-D).
                s = s.reshape((n_classes, -1), order="F")  # shape = (n_classes, n_dof)
                if self.fit_intercept:
                    s_intercept = s[:, -1]
                    s = s[:, :-1]  # shape = (n_classes, n_features)
                else:
                    s_intercept = 0
                tmp = X @ s.T + s_intercept  # X_{im} * s_k_m
                tmp -= (proba * tmp).sum(axis=1)[:, np.newaxis]  # - sum_l ..
                tmp *= proba  # * p_i_k
                if sample_weight is not None:
                    tmp *= sample_weight[:, np.newaxis]
                # hess_prod = empty_like(grad), but we ravel grad below and this
                # function is run after that.
                hess_prod = np.empty((n_classes, n_dof), dtype=weights.dtype, order="F")
                hess_prod[:, :n_features] = (tmp.T @ X) / sw_sum + l2_reg_strength * s
                if self.fit_intercept:
                    hess_prod[:, -1] = tmp.sum(axis=0) / sw_sum
                if coef.ndim == 1:
                    return hess_prod.ravel(order="F")
                else:
                    return hess_prod

            if coef.ndim == 1:
                return grad.ravel(order="F"), hessp

        return grad, hessp
