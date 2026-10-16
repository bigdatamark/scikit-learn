# [MODULE DOC]
# HELPER: Activation functions, their derivatives and loss functions used by the
# MLP estimators. All activations/derivatives operate in place on dense numpy
# arrays. Exposes registries: ACTIVATIONS (name -> in-place activation),
# DERIVATIVES (name -> in-place derivative; no softmax/exp entries since those are
# only used on output layers) and LOSS_FUNCTIONS (name -> loss).
"""Utilities for the neural network modules"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
from scipy.special import expit as logistic_sigmoid
from scipy.special import xlogy


def inplace_identity(X):
    # [FUNCTION DOC]
    # HELPER: Identity activation; intentionally does nothing.
    # Inputs: X - array. Outputs: None. Side effects: none.
    """Simply leave the input array unchanged.

    Parameters
    ----------
    X : {array-like, sparse matrix}, shape (n_samples, n_features)
        Data, where `n_samples` is the number of samples
        and `n_features` is the number of features.
    """
    # Nothing to do


def inplace_exp(X):
    # [FUNCTION DOC]
    # HELPER: Replaces each element of X with its exponential, writing into X.
    # Outputs: None. Side effects: mutates X. May overflow to inf for large values.
    # KNOWN ISSUE: docstring advertises sparse matrices, but writing the ufunc output
    # into a sparse matrix is not supported; effectively dense arrays only.
    """Compute the exponential inplace.

    Parameters
    ----------
    X : {array-like, sparse matrix}, shape (n_samples, n_features)
        The input data.
    """
    np.exp(X, out=X)


def inplace_logistic(X):
    # [FUNCTION DOC]
    # HELPER: Replaces each element of X with the logistic sigmoid 1/(1+exp(-x))
    # (scipy expit), writing into X. Outputs: None. Side effects: mutates X.
    # KNOWN ISSUE: docstring claims sparse support; only dense arrays work.
    """Compute the logistic function inplace.

    Parameters
    ----------
    X : {array-like, sparse matrix}, shape (n_samples, n_features)
        The input data.
    """
    logistic_sigmoid(X, out=X)


def inplace_tanh(X):
    # [FUNCTION DOC]
    # HELPER: Replaces each element of X with tanh(x), in place.
    # Outputs: None. Side effects: mutates X.
    # KNOWN ISSUE: docstring claims sparse support; only dense arrays work.
    """Compute the hyperbolic tan function inplace.

    Parameters
    ----------
    X : {array-like, sparse matrix}, shape (n_samples, n_features)
        The input data.
    """
    np.tanh(X, out=X)


def inplace_relu(X):
    # [FUNCTION DOC]
    # HELPER: Replaces each element of X with max(x, 0), in place.
    # Outputs: None. Side effects: mutates X.
    # KNOWN ISSUE: docstring claims sparse support; only dense arrays work.
    """Compute the rectified linear unit function inplace.

    Parameters
    ----------
    X : {array-like, sparse matrix}, shape (n_samples, n_features)
        The input data.
    """
    np.maximum(X, 0, out=X)


def inplace_softmax(X):
    # [FUNCTION DOC]
    # HELPER: Row-wise softmax in place: subtract each row's maximum (numerical
    # stability), exponentiate into X, then divide each row by its sum.
    # Inputs: X - 2D float array (n_samples, n_classes).
    # Outputs: None. Side effects: mutates X so each row sums to 1.
    # KNOWN ISSUE: docstring claims sparse support; row max/broadcast subtraction
    # would not work on sparse matrices.
    """Compute the K-way softmax function inplace.

    Parameters
    ----------
    X : {array-like, sparse matrix}, shape (n_samples, n_features)
        The input data.
    """
    tmp = X - X.max(axis=1)[:, np.newaxis]
    np.exp(tmp, out=X)
    X /= X.sum(axis=1)[:, np.newaxis]


ACTIVATIONS = {
    "identity": inplace_identity,
    "exp": inplace_exp,
    "tanh": inplace_tanh,
    "logistic": inplace_logistic,
    "relu": inplace_relu,
    "softmax": inplace_softmax,
}


def inplace_identity_derivative(Z, delta):
    # [FUNCTION DOC]
    # HELPER: Backprop through identity; the derivative is 1 so delta is left as is.
    # Outputs: None. Side effects: none.
    """Apply the derivative of the identity function: do nothing.

    Parameters
    ----------
    Z : {array-like, sparse matrix}, shape (n_samples, n_features)
        The data which was output from the identity activation function during
        the forward pass.

    delta : {array-like}, shape (n_samples, n_features)
         The backpropagated error signal to be modified inplace.
    """
    # Nothing to do


def inplace_logistic_derivative(Z, delta):
    # [FUNCTION DOC]
    # HELPER: Multiplies delta element-wise by Z*(1-Z), the sigmoid derivative written
    # in terms of the activation output Z. Inputs: Z, delta - same-shape dense arrays.
    # Outputs: None. Side effects: mutates delta.
    """Apply the derivative of the logistic sigmoid function.

    It exploits the fact that the derivative is a simple function of the output
    value from logistic function.

    Parameters
    ----------
    Z : {array-like, sparse matrix}, shape (n_samples, n_features)
        The data which was output from the logistic activation function during
        the forward pass.

    delta : {array-like}, shape (n_samples, n_features)
         The backpropagated error signal to be modified inplace.
    """
    delta *= Z
    delta *= 1 - Z


def inplace_tanh_derivative(Z, delta):
    # [FUNCTION DOC]
    # HELPER: Multiplies delta element-wise by (1 - Z squared), the tanh derivative in
    # terms of its output Z. Outputs: None. Side effects: mutates delta.
    """Apply the derivative of the hyperbolic tanh function.

    It exploits the fact that the derivative is a simple function of the output
    value from hyperbolic tangent.

    Parameters
    ----------
    Z : {array-like, sparse matrix}, shape (n_samples, n_features)
        The data which was output from the hyperbolic tangent activation
        function during the forward pass.

    delta : {array-like}, shape (n_samples, n_features)
         The backpropagated error signal to be modified inplace.
    """
    delta *= 1 - Z**2


def inplace_relu_derivative(Z, delta):
    # [FUNCTION DOC]
    # HELPER: Sets delta to 0 wherever the ReLU output Z equals 0 (derivative 0 there,
    # 1 elsewhere). Outputs: None. Side effects: mutates delta.
    """Apply the derivative of the relu function.

    It exploits the fact that the derivative is a simple function of the output
    value from rectified linear units activation function.

    Parameters
    ----------
    Z : {array-like, sparse matrix}, shape (n_samples, n_features)
        The data which was output from the rectified linear units activation
        function during the forward pass.

    delta : {array-like}, shape (n_samples, n_features)
         The backpropagated error signal to be modified inplace.
    """
    delta[Z == 0] = 0


DERIVATIVES = {
    "identity": inplace_identity_derivative,
    "tanh": inplace_tanh_derivative,
    "logistic": inplace_logistic_derivative,
    "relu": inplace_relu_derivative,
}


def squared_loss(y_true, y_pred, sample_weight=None):
    # [FUNCTION DOC]
    # HELPER: Half mean squared error: per output column, the (optionally
    # sample_weight-weighted) average over samples of squared residuals, then mean
    # over columns, times 0.5.
    # Inputs: y_true, y_pred - arrays of matching shape (n_samples,) or
    # (n_samples, n_outputs); sample_weight - optional (n_samples,) weights.
    # Outputs: float. Exceptions: numpy errors on shape mismatch or zero weight sum.
    """Compute the squared loss for regression.

    Parameters
    ----------
    y_true : array-like or label indicator matrix
        Ground truth (correct) values.

    y_pred : array-like or label indicator matrix
        Predicted values, as returned by a regression estimator.

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.

    Returns
    -------
    loss : float
        The degree to which the samples are correctly predicted.
    """
    return (
        0.5 * np.average((y_true - y_pred) ** 2, weights=sample_weight, axis=0).mean()
    )


def poisson_loss(y_true, y_pred, sample_weight=None):
    # [FUNCTION DOC]
    # HELPER: Half Poisson deviance (without the factor 2): per column, weighted
    # average over samples of y_true*log(y_true/y_pred) - y_true + y_pred (xlogy gives
    # 0 when y_true is 0), then summed over columns.
    # Inputs: y_true non-negative, y_pred positive arrays; optional sample_weight.
    # Outputs: float; inf/nan if y_pred has zeros.
    # KNOWN ISSUE: docstring calls it "half of the" deviance but the expression is the
    # unit deviance divided by 2 only implicitly (no factor 2 is ever applied), and
    # columns are summed rather than averaged unlike squared_loss.
    """Compute (half of the) Poisson deviance loss for regression.

    Parameters
    ----------
    y_true : array-like or label indicator matrix
        Ground truth (correct) labels.

    y_pred : array-like or label indicator matrix
        Predicted values, as returned by a regression estimator.

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.

    Returns
    -------
    loss : float
        The degree to which the samples are correctly predicted.
    """
    # TODO: Decide what to do with the term `xlogy(y_true, y_true) - y_true`. For now,
    # it is included. But the _loss module doesn't use it (for performance reasons) and
    # only adds it as return of constant_to_optimal_zero (mainly for testing).
    return np.average(
        xlogy(y_true, y_true / y_pred) - y_true + y_pred, weights=sample_weight, axis=0
    ).sum()


def log_loss(y_true, y_prob, sample_weight=None):
    # [FUNCTION DOC]
    # HELPER: Multiclass cross-entropy. Clips y_prob to [eps, 1-eps] (eps of y_prob's
    # dtype); if y_prob or y_true has a single column, expands it to two columns
    # (1-p, p). Returns minus the per-column weighted sample average of
    # y_true*log(y_prob), summed over classes.
    # Inputs: y_true - 2D indicator array (n_samples, n_classes or 1); y_prob - 2D
    # float array same shape; sample_weight - optional (n_samples,).
    # Outputs: float. Exceptions: IndexError if inputs are 1D (shape[1] accessed).
    # Side effects: none (clip returns a copy).
    """Compute Logistic loss for classification.

    Parameters
    ----------
    y_true : array-like or label indicator matrix
        Ground truth (correct) labels.

    y_prob : array-like of float, shape = (n_samples, n_classes)
        Predicted probabilities, as returned by a classifier's
        predict_proba method.

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.

    Returns
    -------
    loss : float
        The degree to which the samples are correctly predicted.
    """
    eps = np.finfo(y_prob.dtype).eps
    y_prob = np.clip(y_prob, eps, 1 - eps)
    if y_prob.shape[1] == 1:
        y_prob = np.append(1 - y_prob, y_prob, axis=1)

    if y_true.shape[1] == 1:
        y_true = np.append(1 - y_true, y_true, axis=1)

    return -np.average(xlogy(y_true, y_prob), weights=sample_weight, axis=0).sum()


def binary_log_loss(y_true, y_prob, sample_weight=None):
    # [FUNCTION DOC]
    # HELPER: Element-wise binary cross-entropy for multilabel outputs: clip y_prob to
    # [eps, 1-eps], compute y*log(p) + (1-y)*log(1-p), take weighted average over
    # samples per column, sum over columns, negate.
    # Inputs: y_true, y_prob - same-shape arrays (n_samples, n_labels); optional
    # sample_weight. Outputs: float. Side effects: none.
    """Compute binary logistic loss for classification.

    This is identical to log_loss in binary classification case,
    but is kept for its use in multilabel case.

    Parameters
    ----------
    y_true : array-like or label indicator matrix
        Ground truth (correct) labels.

    y_prob : array-like of float, shape = (n_samples, 1)
        Predicted probabilities, as returned by a classifier's
        predict_proba method.

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.

    Returns
    -------
    loss : float
        The degree to which the samples are correctly predicted.
    """
    eps = np.finfo(y_prob.dtype).eps
    y_prob = np.clip(y_prob, eps, 1 - eps)
    return -np.average(
        xlogy(y_true, y_prob) + xlogy(1 - y_true, 1 - y_prob),
        weights=sample_weight,
        axis=0,
    ).sum()


LOSS_FUNCTIONS = {
    "squared_error": squared_loss,
    "poisson": poisson_loss,
    "log_loss": log_loss,
    "binary_log_loss": binary_log_loss,
}
