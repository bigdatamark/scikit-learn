# [MODULE DOC]
# Partial Least Squares family (following Wegelin's 2000 review, section 4.1): the _PLS
# base implementing the iterative NIPALS/SVD algorithm with regression or canonical
# deflation, its public variants PLSRegression, PLSCanonical and CCA (mode B), the
# separate single-SVD estimator PLSSVD, and private numerical helpers.
"""
The :mod:`sklearn.pls` module implements Partial Least Squares (PLS).
"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import warnings
from abc import ABCMeta, abstractmethod
from numbers import Integral, Real

import numpy as np
from scipy.linalg import pinv, svd

from sklearn.base import (
    BaseEstimator,
    ClassNamePrefixFeaturesOutMixin,
    MultiOutputMixin,
    RegressorMixin,
    TransformerMixin,
    _fit_context,
)
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils import check_array, check_consistent_length
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.extmath import svd_flip
from sklearn.utils.validation import FLOAT_DTYPES, check_is_fitted, validate_data

__all__ = ["PLSSVD", "PLSCanonical", "PLSRegression"]


def _pinv2_old(a):
    # [FUNCTION DOC]
    # HELPER: Pseudo-inverse reproducing old scipy pinv2 behavior: thin SVD of a, rank =
    # number of singular values above max(s) times a dtype factor (1e3 for float32, 1e6
    # for float64) times machine eps, then V times diag(1/s) times U conjugate-
    # transposed on that rank.
    # Inputs: a - 2-D float array (m, n). Outputs: (n, m) pseudo-inverse.
    # Exceptions: KeyError for dtypes other than float/double (factor lookup).
    # Used previous scipy pinv2 that was updated in:
    # https://github.com/scipy/scipy/pull/10067
    # We can not set `cond` or `rcond` for pinv2 in scipy >= 1.3 to keep the
    # same behavior of pinv2 for scipy < 1.3, because the condition used to
    # determine the rank is dependent on the output of svd.
    u, s, vh = svd(a, full_matrices=False, check_finite=False)

    t = u.dtype.char.lower()
    factor = {"f": 1e3, "d": 1e6}
    cond = np.max(s) * factor[t] * np.finfo(t).eps
    rank = np.sum(s > cond)

    u = u[:, :rank]
    u /= s[:rank]
    return np.transpose(np.conjugate(np.dot(u, vh[:rank])))


def _get_first_singular_vectors_power_method(
    X, y, mode="A", max_iter=500, tol=1e-06, norm_y_weights=False
):
    # [FUNCTION DOC]
    # HELPER: NIPALS power iteration for the first pair of X and y weight vectors.
    # Behavior: start y_score at the first column of y with any entry above eps (raise
    # StopIteration 'y residual is constant' if none). In mode 'B' precompute pseudo-
    # inverses of X and y. Each iteration: x_weights = X transposed times y_score
    # divided by its squared norm (mode A) or X_pinv times y_score (mode B), normalize;
    # x_score = X times x_weights; y_weights analogous from y and x_score; optionally
    # normalize y_weights; y_score = y times y_weights divided by its squared norm. Stop
    # when the squared change in x_weights is below tol or y has a single column.
    # Inputs: X - (n, p); y - (n, q); mode 'A'/'B'; max_iter - int; tol - float;
    # norm_y_weights - bool.
    # Outputs: (x_weights (p,), y_weights (q,), n_iter int).
    # Exceptions: StopIteration when y is constant. Side effects: ConvergenceWarning
    # when n_iter equals max_iter.
    # KNOWN ISSUE: the warning is also emitted when convergence happens exactly on the
    # last allowed iteration.
    """Return the first left and right singular vectors of X'y.

    Provides an alternative to the svd(X'y) and uses the power method instead.
    With norm_y_weights to True and in mode A, this corresponds to the
    algorithm section 11.3 of the Wegelin's review, except this starts at the
    "update saliences" part.
    """

    eps = np.finfo(X.dtype).eps
    try:
        y_score = next(col for col in y.T if np.any(np.abs(col) > eps))
    except StopIteration as e:
        raise StopIteration("y residual is constant") from e

    x_weights_old = 100  # init to big value for first convergence check

    if mode == "B":
        # Precompute pseudo inverse matrices
        # Basically: X_pinv = (X.T X)^-1 X.T
        # Which requires inverting a (n_features, n_features) matrix.
        # As a result, and as detailed in the Wegelin's review, CCA (i.e. mode
        # B) will be unstable if n_features > n_samples or n_targets >
        # n_samples
        X_pinv, y_pinv = _pinv2_old(X), _pinv2_old(y)

    for i in range(max_iter):
        if mode == "B":
            x_weights = np.dot(X_pinv, y_score)
        else:
            x_weights = np.dot(X.T, y_score) / np.dot(y_score, y_score)

        x_weights /= np.sqrt(np.dot(x_weights, x_weights)) + eps
        x_score = np.dot(X, x_weights)

        if mode == "B":
            y_weights = np.dot(y_pinv, x_score)
        else:
            y_weights = np.dot(y.T, x_score) / np.dot(x_score.T, x_score)

        if norm_y_weights:
            y_weights /= np.sqrt(np.dot(y_weights, y_weights)) + eps

        y_score = np.dot(y, y_weights) / (np.dot(y_weights, y_weights) + eps)

        x_weights_diff = x_weights - x_weights_old
        if np.dot(x_weights_diff, x_weights_diff) < tol or y.shape[1] == 1:
            break
        x_weights_old = x_weights

    n_iter = i + 1
    if n_iter == max_iter:
        warnings.warn("Maximum number of iterations reached", ConvergenceWarning)

    return x_weights, y_weights, n_iter


def _get_first_singular_vectors_svd(X, y):
    # [FUNCTION DOC]
    # HELPER: Returns the first left and right singular vectors of the cross-covariance
    # X transposed times y via a full SVD.
    # Inputs: X - (n, p); y - (n, q). Outputs: (u (p,), v (q,)).
    """Return the first left and right singular vectors of X'y.

    Here the whole SVD is computed.
    """
    C = np.dot(X.T, y)
    U, _, Vt = svd(C, full_matrices=False)
    return U[:, 0], Vt[0, :]


def _center_scale_xy(X, y, scale=True):
    # [FUNCTION DOC]
    # HELPER: Centers X and y in place by their column means and, if scale, divides by
    # column standard deviations (ddof 1, zeros replaced by 1); otherwise stds are ones.
    # Inputs: X - (n, p) float array; y - (n, q) float array; scale - bool.
    # Outputs: (X, y, x_mean, y_mean, x_std, y_std). Side effects: mutates X and y.
    """Center X, y and scale if the scale parameter==True

    Returns
    -------
        X, y, x_mean, y_mean, x_std, y_std
    """
    # center
    x_mean = X.mean(axis=0)
    X -= x_mean
    y_mean = y.mean(axis=0)
    y -= y_mean
    # scale
    if scale:
        x_std = X.std(axis=0, ddof=1)
        x_std[x_std == 0.0] = 1.0
        X /= x_std
        y_std = y.std(axis=0, ddof=1)
        y_std[y_std == 0.0] = 1.0
        y /= y_std
    else:
        x_std = np.ones(X.shape[1])
        y_std = np.ones(y.shape[1])
    return X, y, x_mean, y_mean, x_std, y_std


def _svd_flip_1d(u, v):
    # [FUNCTION DOC]
    # HELPER: In place, multiplies u and v by the sign of the entry of u with largest
    # absolute value, making that entry positive.
    # Inputs: u, v - 1-D arrays. Outputs: None. Side effects: mutates u and v.
    """Same as svd_flip but works on 1d arrays, and is inplace"""
    # svd_flip would force us to convert to 2d array and would also return 2d
    # arrays. We don't want that.
    biggest_abs_val_idx = np.argmax(np.abs(u))
    sign = np.sign(u[biggest_abs_val_idx])
    u *= sign
    v *= sign


class _PLS(
    ClassNamePrefixFeaturesOutMixin,
    TransformerMixin,
    RegressorMixin,
    MultiOutputMixin,
    BaseEstimator,
    metaclass=ABCMeta,
):
    # [CLASS DOC]
    # HELPER: Abstract base of PLSRegression, PLSCanonical and CCA; transformer and
    # multi-output regressor. Declares constraints for n_components, scale,
    # deflation_mode, mode, algorithm, max_iter, tol and copy.
    """Partial Least Squares (PLS)

    This class implements the generic PLS algorithm.

    Main ref: Wegelin, a survey of Partial Least Squares (PLS) methods,
    with emphasis on the two-block case
    https://stat.uw.edu/sites/default/files/files/reports/2000/tr371.pdf
    """

    _parameter_constraints: dict = {
        "n_components": [Interval(Integral, 1, None, closed="left")],
        "scale": ["boolean"],
        "deflation_mode": [StrOptions({"regression", "canonical"})],
        "mode": [StrOptions({"A", "B"})],
        "algorithm": [StrOptions({"svd", "nipals"})],
        "max_iter": [Interval(Integral, 1, None, closed="left")],
        "tol": [Interval(Real, 0, None, closed="left")],
        "copy": ["boolean"],
    }

    @abstractmethod
    def __init__(
        self,
        n_components=2,
        *,
        scale=True,
        deflation_mode="regression",
        mode="A",
        algorithm="nipals",
        max_iter=500,
        tol=1e-06,
        copy=True,
    ):
        # [FUNCTION DOC]
        # Stores all hyperparameters as attributes (abstract: subclasses fix some of
        # them).
        self.n_components = n_components
        self.deflation_mode = deflation_mode
        self.mode = mode
        self.scale = scale
        self.algorithm = algorithm
        self.max_iter = max_iter
        self.tol = tol
        self.copy = copy

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y):
        # [FUNCTION DOC]
        # Behavior: validate X and y (float64, copies if copy, >= 2 samples), reshape
        # 1-D y to a column and remember that predictions should be 1-D. Check
        # n_components does not exceed min(n, p) (regression deflation) or min(n, p, q)
        # (canonical). Center/scale X and y. For each component k: get weights (nipals:
        # zero near-constant y columns, then power method; stop early with a warning if
        # the y residual is constant; svd: first singular vectors), sign-flip, compute
        # x_scores = Xk times x_weights and y_scores = yk times y_weights (divided by
        # the squared y_weights norm unless canonical), x_loadings by regressing Xk on
        # x_scores and deflate Xk; deflate yk on y_scores (canonical) or x_scores
        # (regression); store the column k of every matrix. Finally compute x_rotations_
        # = x_weights_ times pinv(x_loadings_ transposed times x_weights_), likewise
        # y_rotations_, coef_ = x_rotations_ times y_loadings_ transposed rescaled to
        # original units (shape (q, p)), intercept_ = y mean.
        # Inputs: X - (n, p); y - (n,) or (n, q). Outputs: self.
        # Exceptions: ValueError for too large n_components or invalid input.
        # Side effects: sets x/y_weights_, x/y_loadings_, x/y_rotations_, coef_,
        # intercept_, n_iter_ (list, nipals only), private _x_scores, _y_scores,
        # _x/_y_mean, _x/_y_std, _predict_1d, _norm_y_weights, _n_features_out; may
        # modify X/y in place when copy is False; warnings on non-convergence or
        # constant residual.
        """Fit model to data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vectors, where `n_samples` is the number of samples and
            `n_features` is the number of predictors.

        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Target vectors, where `n_samples` is the number of samples and
            `n_targets` is the number of response variables.

        Returns
        -------
        self : object
            Fitted model.
        """
        check_consistent_length(X, y)
        X = validate_data(
            self,
            X,
            dtype=np.float64,
            force_writeable=True,
            copy=self.copy,
            ensure_min_samples=2,
        )
        y = check_array(
            y,
            input_name="y",
            dtype=np.float64,
            force_writeable=True,
            copy=self.copy,
            ensure_2d=False,
        )
        if y.ndim == 1:
            self._predict_1d = True
            y = y.reshape(-1, 1)
        else:
            self._predict_1d = False

        n = X.shape[0]
        p = X.shape[1]
        q = y.shape[1]

        n_components = self.n_components
        # With PLSRegression n_components is bounded by the rank of (X.T X) see
        # Wegelin page 25. With CCA and PLSCanonical, n_components is bounded
        # by the rank of X and the rank of y: see Wegelin page 12
        rank_upper_bound = (
            min(n, p) if self.deflation_mode == "regression" else min(n, p, q)
        )
        if n_components > rank_upper_bound:
            raise ValueError(
                f"`n_components` upper bound is {rank_upper_bound}. "
                f"Got {n_components} instead. Reduce `n_components`."
            )

        self._norm_y_weights = self.deflation_mode == "canonical"  # 1.1
        norm_y_weights = self._norm_y_weights

        # Scale (in place)
        Xk, yk, self._x_mean, self._y_mean, self._x_std, self._y_std = _center_scale_xy(
            X, y, self.scale
        )

        self.x_weights_ = np.zeros((p, n_components))  # U
        self.y_weights_ = np.zeros((q, n_components))  # V
        self._x_scores = np.zeros((n, n_components))  # Xi
        self._y_scores = np.zeros((n, n_components))  # Omega
        self.x_loadings_ = np.zeros((p, n_components))  # Gamma
        self.y_loadings_ = np.zeros((q, n_components))  # Delta
        self.n_iter_ = []

        # This whole thing corresponds to the algorithm in section 4.1 of the
        # review from Wegelin. See above for a notation mapping from code to
        # paper.
        y_eps = np.finfo(yk.dtype).eps
        for k in range(n_components):
            # Find first left and right singular vectors of the X.T.dot(y)
            # cross-covariance matrix.
            if self.algorithm == "nipals":
                # Replace columns that are all close to zero with zeros
                yk_mask = np.all(np.abs(yk) < 10 * y_eps, axis=0)
                yk[:, yk_mask] = 0.0

                try:
                    (
                        x_weights,
                        y_weights,
                        n_iter_,
                    ) = _get_first_singular_vectors_power_method(
                        Xk,
                        yk,
                        mode=self.mode,
                        max_iter=self.max_iter,
                        tol=self.tol,
                        norm_y_weights=norm_y_weights,
                    )
                except StopIteration as e:
                    if str(e) != "y residual is constant":
                        raise
                    warnings.warn(f"y residual is constant at iteration {k}")
                    break

                self.n_iter_.append(n_iter_)

            elif self.algorithm == "svd":
                x_weights, y_weights = _get_first_singular_vectors_svd(Xk, yk)

            # inplace sign flip for consistency across solvers and archs
            _svd_flip_1d(x_weights, y_weights)

            # compute scores, i.e. the projections of X and y
            x_scores = np.dot(Xk, x_weights)
            if norm_y_weights:
                y_ss = 1
            else:
                y_ss = np.dot(y_weights, y_weights)
            y_scores = np.dot(yk, y_weights) / y_ss

            # Deflation: subtract rank-one approx to obtain Xk+1 and yk+1
            x_loadings = np.dot(x_scores, Xk) / np.dot(x_scores, x_scores)
            Xk -= np.outer(x_scores, x_loadings)

            if self.deflation_mode == "canonical":
                # regress yk on y_score
                y_loadings = np.dot(y_scores, yk) / np.dot(y_scores, y_scores)
                yk -= np.outer(y_scores, y_loadings)
            if self.deflation_mode == "regression":
                # regress yk on x_score
                y_loadings = np.dot(x_scores, yk) / np.dot(x_scores, x_scores)
                yk -= np.outer(x_scores, y_loadings)

            self.x_weights_[:, k] = x_weights
            self.y_weights_[:, k] = y_weights
            self._x_scores[:, k] = x_scores
            self._y_scores[:, k] = y_scores
            self.x_loadings_[:, k] = x_loadings
            self.y_loadings_[:, k] = y_loadings

        # X was approximated as Xi . Gamma.T + X_(R+1)
        # Xi . Gamma.T is a sum of n_components rank-1 matrices. X_(R+1) is
        # whatever is left to fully reconstruct X, and can be 0 if X is of rank
        # n_components.
        # Similarly, y was approximated as Omega . Delta.T + y_(R+1)

        # Compute transformation matrices (rotations_). See User Guide.
        self.x_rotations_ = np.dot(
            self.x_weights_,
            pinv(np.dot(self.x_loadings_.T, self.x_weights_), check_finite=False),
        )
        self.y_rotations_ = np.dot(
            self.y_weights_,
            pinv(np.dot(self.y_loadings_.T, self.y_weights_), check_finite=False),
        )
        self.coef_ = np.dot(self.x_rotations_, self.y_loadings_.T)
        self.coef_ = (self.coef_ * self._y_std).T / self._x_std
        self.intercept_ = self._y_mean
        self._n_features_out = self.x_rotations_.shape[1]
        return self

    def transform(self, X, y=None, copy=True):
        # [FUNCTION DOC]
        # Centers/scales X with training statistics and projects with x_rotations_; if y
        # given (1-D reshaped), does the same with y_rotations_.
        # Inputs: X - (n, p); y - optional (n,) or (n, q); copy - if False X and y may
        # be modified in place. Outputs: x_scores (n, n_components) or a tuple
        # (x_scores, y_scores).
        # Exceptions: NotFittedError, feature mismatch errors.
        """Apply the dimension reduction.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to transform.

        y : array-like of shape (n_samples, n_targets), default=None
            Target vectors.

        copy : bool, default=True
            Whether to copy `X` and `y`, or perform in-place normalization.

        Returns
        -------
        x_scores, y_scores : array-like or tuple of array-like
            Return `x_scores` if `y` is not given, `(x_scores, y_scores)` otherwise.
        """
        check_is_fitted(self)
        X = validate_data(self, X, copy=copy, dtype=FLOAT_DTYPES, reset=False)
        # Normalize
        X -= self._x_mean
        X /= self._x_std
        # Apply rotation
        x_scores = np.dot(X, self.x_rotations_)
        if y is not None:
            y = check_array(
                y, input_name="y", ensure_2d=False, copy=copy, dtype=FLOAT_DTYPES
            )
            if y.ndim == 1:
                y = y.reshape(-1, 1)
            y -= self._y_mean
            y /= self._y_std
            y_scores = np.dot(y, self.y_rotations_)
            return x_scores, y_scores

        return x_scores

    def inverse_transform(self, X, y=None):
        # [FUNCTION DOC]
        # Maps latent scores back: X times x_loadings_ transposed, then multiply by x
        # std and add x mean; likewise for y with y_loadings_ when given.
        # Inputs: X - (n, n_components); y - optional (n, n_components) 2-D.
        # Outputs: X_reconstructed (n, p) or a tuple with y_reconstructed (n, q).
        """Transform data back to its original space.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_components)
            New data, where `n_samples` is the number of samples
            and `n_components` is the number of pls components.

        y : array-like of shape (n_samples,) or (n_samples, n_components)
            New target, where `n_samples` is the number of samples
            and `n_components` is the number of pls components.

        Returns
        -------
        X_original : ndarray of shape (n_samples, n_features)
            Return the reconstructed `X` data.

        y_original : ndarray of shape (n_samples, n_targets)
            Return the reconstructed `X` target. Only returned when `y` is given.

        Notes
        -----
        This transformation will only be exact if `n_components=n_features`.
        """
        check_is_fitted(self)
        X = check_array(X, input_name="X", dtype=FLOAT_DTYPES)
        # From pls space to original space
        X_reconstructed = np.matmul(X, self.x_loadings_.T)
        # Denormalize
        X_reconstructed *= self._x_std
        X_reconstructed += self._x_mean

        if y is not None:
            y = check_array(y, input_name="y", dtype=FLOAT_DTYPES)
            # From pls space to original space
            y_reconstructed = np.matmul(y, self.y_loadings_.T)
            # Denormalize
            y_reconstructed *= self._y_std
            y_reconstructed += self._y_mean
            return X_reconstructed, y_reconstructed

        return X_reconstructed

    def predict(self, X, copy=True):
        # [FUNCTION DOC]
        # Centers X (no scaling, coef_ already includes it) and returns X times coef_
        # transposed plus intercept_, raveled to 1-D if fit received 1-D y.
        # Inputs: X - (n, p); copy - if False X is centered in place. Outputs: (n,) or
        # (n, q).
        """Predict targets of given samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.

        copy : bool, default=True
            Whether to copy `X` or perform in-place normalization.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,) or (n_samples, n_targets)
            Returns predicted values.

        Notes
        -----
        This call requires the estimation of a matrix of shape
        `(n_features, n_targets)`, which may be an issue in high dimensional
        space.
        """
        check_is_fitted(self)
        X = validate_data(self, X, copy=copy, dtype=FLOAT_DTYPES, reset=False)
        # Only center X but do not scale it since the coefficients are already scaled
        X -= self._x_mean
        y_pred = X @ self.coef_.T + self.intercept_
        return y_pred.ravel() if self._predict_1d else y_pred

    def fit_transform(self, X, y=None):
        # [FUNCTION DOC]
        # Fits on (X, y) and returns transform(X, y).
        # Outputs: x_scores or (x_scores, y_scores).
        # KNOWN ISSUE: y defaults to None but fit requires y, so calling without y fails
        # in validation.
        """Learn and apply the dimension reduction on the train data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vectors, where `n_samples` is the number of samples and
            `n_features` is the number of predictors.

        y : array-like of shape (n_samples, n_targets), default=None
            Target vectors, where `n_samples` is the number of samples and
            `n_targets` is the number of response variables.

        Returns
        -------
        self : ndarray of shape (n_samples, n_components)
            Return `x_scores` if `y` is not given, `(x_scores, y_scores)` otherwise.
        """
        return self.fit(X, y).transform(X, y)

    def __sklearn_tags__(self):
        # [FUNCTION DOC]
        # Extends parent tags: marks regressor score as poor (poor_score) and target as
        # not required.
        tags = super().__sklearn_tags__()
        tags.regressor_tags.poor_score = True
        tags.target_tags.required = False
        return tags


class PLSRegression(_PLS):
    # [CLASS DOC]
    # PLS2/PLS1 regression: regression deflation, mode A, NIPALS; those three parameters
    # are removed from the constraints.
    """PLS regression.

    PLSRegression is also known as PLS2 or PLS1, depending on the number of
    targets.

    For a comparison between other cross decomposition algorithms, see
    :ref:`sphx_glr_auto_examples_cross_decomposition_plot_compare_cross_decomposition.py`.

    Read more in the :ref:`User Guide <cross_decomposition>`.

    .. versionadded:: 0.8

    Parameters
    ----------
    n_components : int, default=2
        Number of components to keep. Should be in `[1, n_features]`.

    scale : bool, default=True
        Whether to scale `X` and `y`.

    max_iter : int, default=500
        The maximum number of iterations of the power method when
        `algorithm='nipals'`. Ignored otherwise.

    tol : float, default=1e-06
        The tolerance used as convergence criteria in the power method: the
        algorithm stops whenever the squared norm of `u_i - u_{i-1}` is less
        than `tol`, where `u` corresponds to the left singular vector.

    copy : bool, default=True
        Whether to copy `X` and `y` in :term:`fit` before applying centering,
        and potentially scaling. If `False`, these operations will be done
        inplace, modifying both arrays.

    Attributes
    ----------
    x_weights_ : ndarray of shape (n_features, n_components)
        The left singular vectors of the cross-covariance matrices of each
        iteration.

    y_weights_ : ndarray of shape (n_targets, n_components)
        The right singular vectors of the cross-covariance matrices of each
        iteration.

    x_loadings_ : ndarray of shape (n_features, n_components)
        The loadings of `X`.

    y_loadings_ : ndarray of shape (n_targets, n_components)
        The loadings of `y`.

    x_scores_ : ndarray of shape (n_samples, n_components)
        The transformed training samples.

    y_scores_ : ndarray of shape (n_samples, n_components)
        The transformed training targets.

    x_rotations_ : ndarray of shape (n_features, n_components)
        The projection matrix used to transform `X`.

    y_rotations_ : ndarray of shape (n_targets, n_components)
        The projection matrix used to transform `y`.

    coef_ : ndarray of shape (n_target, n_features)
        The coefficients of the linear model such that `y` is approximated as
        `y = X @ coef_.T + intercept_`.

    intercept_ : ndarray of shape (n_targets,)
        The intercepts of the linear model such that `y` is approximated as
        `y = X @ coef_.T + intercept_`.

        .. versionadded:: 1.1

    n_iter_ : list of shape (n_components,)
        Number of iterations of the power method, for each
        component.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    PLSCanonical : Partial Least Squares transformer and regressor.

    Examples
    --------
    >>> from sklearn.cross_decomposition import PLSRegression
    >>> X = [[0., 0., 1.], [1.,0.,0.], [2.,2.,2.], [2.,5.,4.]]
    >>> y = [[0.1, -0.2], [0.9, 1.1], [6.2, 5.9], [11.9, 12.3]]
    >>> pls2 = PLSRegression(n_components=2)
    >>> pls2.fit(X, y)
    PLSRegression()
    >>> y_pred = pls2.predict(X)

    For a comparison between PLS Regression and :class:`~sklearn.decomposition.PCA`, see
    :ref:`sphx_glr_auto_examples_cross_decomposition_plot_pcr_vs_pls.py`.
    """

    _parameter_constraints: dict = {**_PLS._parameter_constraints}
    for param in ("deflation_mode", "mode", "algorithm"):
        _parameter_constraints.pop(param)

    # This implementation provides the same results that 3 PLS packages
    # provided in the R language (R-project):
    #     - "mixOmics" with function pls(X, y, mode = "regression")
    #     - "plspm " with function plsreg2(X, y)
    #     - "pls" with function oscorespls.fit(X, y)

    def __init__(
        self, n_components=2, *, scale=True, max_iter=500, tol=1e-06, copy=True
    ):
        # [FUNCTION DOC]
        # Forwards n_components, scale, max_iter, tol, copy to _PLS with deflation_mode
        # 'regression', mode 'A', algorithm 'nipals'.
        super().__init__(
            n_components=n_components,
            scale=scale,
            deflation_mode="regression",
            mode="A",
            algorithm="nipals",
            max_iter=max_iter,
            tol=tol,
            copy=copy,
        )

    def fit(self, X, y):
        # [FUNCTION DOC]
        # Runs _PLS.fit, then exposes x_scores_ and y_scores_ as public attributes.
        # Outputs: self.
        """Fit model to data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vectors, where `n_samples` is the number of samples and
            `n_features` is the number of predictors.

        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Target vectors, where `n_samples` is the number of samples and
            `n_targets` is the number of response variables.

        Returns
        -------
        self : object
            Fitted model.
        """
        super().fit(X, y)
        # expose the fitted attributes `x_scores_` and `y_scores_`
        self.x_scores_ = self._x_scores
        self.y_scores_ = self._y_scores
        return self


class PLSCanonical(_PLS):
    # [CLASS DOC]
    # Symmetric PLS with canonical deflation and mode A; algorithm remains configurable
    # ('nipals' or 'svd').
    """Partial Least Squares transformer and regressor.

    For a comparison between other cross decomposition algorithms, see
    :ref:`sphx_glr_auto_examples_cross_decomposition_plot_compare_cross_decomposition.py`.

    Read more in the :ref:`User Guide <cross_decomposition>`.

    .. versionadded:: 0.8

    Parameters
    ----------
    n_components : int, default=2
        Number of components to keep. Should be in `[1, min(n_samples,
        n_features, n_targets)]`.

    scale : bool, default=True
        Whether to scale `X` and `y`.

    algorithm : {'nipals', 'svd'}, default='nipals'
        The algorithm used to estimate the first singular vectors of the
        cross-covariance matrix. 'nipals' uses the power method while 'svd'
        will compute the whole SVD.

    max_iter : int, default=500
        The maximum number of iterations of the power method when
        `algorithm='nipals'`. Ignored otherwise.

    tol : float, default=1e-06
        The tolerance used as convergence criteria in the power method: the
        algorithm stops whenever the squared norm of `u_i - u_{i-1}` is less
        than `tol`, where `u` corresponds to the left singular vector.

    copy : bool, default=True
        Whether to copy `X` and `y` in fit before applying centering, and
        potentially scaling. If False, these operations will be done inplace,
        modifying both arrays.

    Attributes
    ----------
    x_weights_ : ndarray of shape (n_features, n_components)
        The left singular vectors of the cross-covariance matrices of each
        iteration.

    y_weights_ : ndarray of shape (n_targets, n_components)
        The right singular vectors of the cross-covariance matrices of each
        iteration.

    x_loadings_ : ndarray of shape (n_features, n_components)
        The loadings of `X`.

    y_loadings_ : ndarray of shape (n_targets, n_components)
        The loadings of `y`.

    x_rotations_ : ndarray of shape (n_features, n_components)
        The projection matrix used to transform `X`.

    y_rotations_ : ndarray of shape (n_targets, n_components)
        The projection matrix used to transform `y`.

    coef_ : ndarray of shape (n_targets, n_features)
        The coefficients of the linear model such that `y` is approximated as
        `y = X @ coef_.T + intercept_`.

    intercept_ : ndarray of shape (n_targets,)
        The intercepts of the linear model such that `y` is approximated as
        `y = X @ coef_.T + intercept_`.

        .. versionadded:: 1.1

    n_iter_ : list of shape (n_components,)
        Number of iterations of the power method, for each
        component. Empty if `algorithm='svd'`.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    CCA : Canonical Correlation Analysis.
    PLSSVD : Partial Least Square SVD.

    Examples
    --------
    >>> from sklearn.cross_decomposition import PLSCanonical
    >>> X = [[0., 0., 1.], [1.,0.,0.], [2.,2.,2.], [2.,5.,4.]]
    >>> y = [[0.1, -0.2], [0.9, 1.1], [6.2, 5.9], [11.9, 12.3]]
    >>> plsca = PLSCanonical(n_components=2)
    >>> plsca.fit(X, y)
    PLSCanonical()
    >>> X_c, y_c = plsca.transform(X, y)
    """

    _parameter_constraints: dict = {**_PLS._parameter_constraints}
    for param in ("deflation_mode", "mode"):
        _parameter_constraints.pop(param)

    # This implementation provides the same results that the "plspm" package
    # provided in the R language (R-project), using the function plsca(X, y).
    # Results are equal or collinear with the function
    # ``pls(..., mode = "canonical")`` of the "mixOmics" package. The
    # difference relies in the fact that mixOmics implementation does not
    # exactly implement the Wold algorithm since it does not normalize
    # y_weights to one.

    def __init__(
        self,
        n_components=2,
        *,
        scale=True,
        algorithm="nipals",
        max_iter=500,
        tol=1e-06,
        copy=True,
    ):
        # [FUNCTION DOC]
        # Forwards parameters to _PLS with deflation_mode 'canonical' and mode 'A'.
        super().__init__(
            n_components=n_components,
            scale=scale,
            deflation_mode="canonical",
            mode="A",
            algorithm=algorithm,
            max_iter=max_iter,
            tol=tol,
            copy=copy,
        )


class CCA(_PLS):
    # [CLASS DOC]
    # Canonical Correlation Analysis: canonical deflation with mode B (uses pseudo-
    # inverses, unstable when features or targets exceed samples), NIPALS only.
    """Canonical Correlation Analysis, also known as "Mode B" PLS.

    For a comparison between other cross decomposition algorithms, see
    :ref:`sphx_glr_auto_examples_cross_decomposition_plot_compare_cross_decomposition.py`.

    Read more in the :ref:`User Guide <cross_decomposition>`.

    Parameters
    ----------
    n_components : int, default=2
        Number of components to keep. Should be in `[1, min(n_samples,
        n_features, n_targets)]`.

    scale : bool, default=True
        Whether to scale `X` and `y`.

    max_iter : int, default=500
        The maximum number of iterations of the power method.

    tol : float, default=1e-06
        The tolerance used as convergence criteria in the power method: the
        algorithm stops whenever the squared norm of `u_i - u_{i-1}` is less
        than `tol`, where `u` corresponds to the left singular vector.

    copy : bool, default=True
        Whether to copy `X` and `y` in fit before applying centering, and
        potentially scaling. If False, these operations will be done inplace,
        modifying both arrays.

    Attributes
    ----------
    x_weights_ : ndarray of shape (n_features, n_components)
        The left singular vectors of the cross-covariance matrices of each
        iteration.

    y_weights_ : ndarray of shape (n_targets, n_components)
        The right singular vectors of the cross-covariance matrices of each
        iteration.

    x_loadings_ : ndarray of shape (n_features, n_components)
        The loadings of `X`.

    y_loadings_ : ndarray of shape (n_targets, n_components)
        The loadings of `y`.

    x_rotations_ : ndarray of shape (n_features, n_components)
        The projection matrix used to transform `X`.

    y_rotations_ : ndarray of shape (n_targets, n_components)
        The projection matrix used to transform `y`.

    coef_ : ndarray of shape (n_targets, n_features)
        The coefficients of the linear model such that `y` is approximated as
        `y = X @ coef_.T + intercept_`.

    intercept_ : ndarray of shape (n_targets,)
        The intercepts of the linear model such that `y` is approximated as
        `y = X @ coef_.T + intercept_`.

        .. versionadded:: 1.1

    n_iter_ : list of shape (n_components,)
        Number of iterations of the power method, for each
        component.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    PLSCanonical : Partial Least Squares transformer and regressor.
    PLSSVD : Partial Least Square SVD.

    Examples
    --------
    >>> from sklearn.cross_decomposition import CCA
    >>> X = [[0., 0., 1.], [1.,0.,0.], [2.,2.,2.], [3.,5.,4.]]
    >>> y = [[0.1, -0.2], [0.9, 1.1], [6.2, 5.9], [11.9, 12.3]]
    >>> cca = CCA(n_components=1)
    >>> cca.fit(X, y)
    CCA(n_components=1)
    >>> X_c, y_c = cca.transform(X, y)
    """

    _parameter_constraints: dict = {**_PLS._parameter_constraints}
    for param in ("deflation_mode", "mode", "algorithm"):
        _parameter_constraints.pop(param)

    def __init__(
        self, n_components=2, *, scale=True, max_iter=500, tol=1e-06, copy=True
    ):
        # [FUNCTION DOC]
        # Forwards parameters to _PLS with deflation_mode 'canonical', mode 'B',
        # algorithm 'nipals'.
        super().__init__(
            n_components=n_components,
            scale=scale,
            deflation_mode="canonical",
            mode="B",
            algorithm="nipals",
            max_iter=max_iter,
            tol=tol,
            copy=copy,
        )


class PLSSVD(ClassNamePrefixFeaturesOutMixin, TransformerMixin, BaseEstimator):
    # [CLASS DOC]
    # Transformer computing all components at once from one SVD of the cross-covariance
    # matrix (no deflation). Constraints: n_components >= 1, scale, copy.
    """Partial Least Square SVD.

    This transformer simply performs an SVD on the cross-covariance matrix
    `X'y`. It is able to project both the training data `X` and the targets
    `y`. The training data `X` is projected on the left singular vectors, while
    the targets are projected on the right singular vectors.

    Read more in the :ref:`User Guide <cross_decomposition>`.

    .. versionadded:: 0.8

    Parameters
    ----------
    n_components : int, default=2
        The number of components to keep. Should be in `[1,
        min(n_samples, n_features, n_targets)]`.

    scale : bool, default=True
        Whether to scale `X` and `y`.

    copy : bool, default=True
        Whether to copy `X` and `y` in fit before applying centering, and
        potentially scaling. If `False`, these operations will be done inplace,
        modifying both arrays.

    Attributes
    ----------
    x_weights_ : ndarray of shape (n_features, n_components)
        The left singular vectors of the SVD of the cross-covariance matrix.
        Used to project `X` in :meth:`transform`.

    y_weights_ : ndarray of (n_targets, n_components)
        The right singular vectors of the SVD of the cross-covariance matrix.
        Used to project `X` in :meth:`transform`.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    PLSCanonical : Partial Least Squares transformer and regressor.
    CCA : Canonical Correlation Analysis.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.cross_decomposition import PLSSVD
    >>> X = np.array([[0., 0., 1.],
    ...               [1., 0., 0.],
    ...               [2., 2., 2.],
    ...               [2., 5., 4.]])
    >>> y = np.array([[0.1, -0.2],
    ...               [0.9, 1.1],
    ...               [6.2, 5.9],
    ...               [11.9, 12.3]])
    >>> pls = PLSSVD(n_components=2).fit(X, y)
    >>> X_c, y_c = pls.transform(X, y)
    >>> X_c.shape, y_c.shape
    ((4, 2), (4, 2))
    """

    _parameter_constraints: dict = {
        "n_components": [Interval(Integral, 1, None, closed="left")],
        "scale": ["boolean"],
        "copy": ["boolean"],
    }

    def __init__(self, n_components=2, *, scale=True, copy=True):
        # [FUNCTION DOC]
        # Stores n_components, scale and copy.
        self.n_components = n_components
        self.scale = scale
        self.copy = copy

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y):
        # [FUNCTION DOC]
        # Validates X and y like _PLS.fit (1-D y reshaped), checks n_components <=
        # min(n, p, q), centers/scales both, computes the SVD of X transposed times y,
        # keeps the first n_components singular vector pairs with deterministic signs
        # (svd_flip).
        # Outputs: self. Exceptions: ValueError for too large n_components.
        # Side effects: sets x_weights_ (p, k), y_weights_ (q, k), _x/_y_mean,
        # _x/_y_std, _n_features_out; may modify inputs in place when copy is False.
        """Fit model to data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training samples.

        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Targets.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        check_consistent_length(X, y)
        X = validate_data(
            self,
            X,
            dtype=np.float64,
            force_writeable=True,
            copy=self.copy,
            ensure_min_samples=2,
        )
        y = check_array(
            y,
            input_name="y",
            dtype=np.float64,
            force_writeable=True,
            copy=self.copy,
            ensure_2d=False,
        )
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        # we'll compute the SVD of the cross-covariance matrix = X.T.dot(y)
        # This matrix rank is at most min(n_samples, n_features, n_targets) so
        # n_components cannot be bigger than that.
        n_components = self.n_components
        rank_upper_bound = min(X.shape[0], X.shape[1], y.shape[1])
        if n_components > rank_upper_bound:
            raise ValueError(
                f"`n_components` upper bound is {rank_upper_bound}. "
                f"Got {n_components} instead. Reduce `n_components`."
            )

        X, y, self._x_mean, self._y_mean, self._x_std, self._y_std = _center_scale_xy(
            X, y, self.scale
        )

        # Compute SVD of cross-covariance matrix
        C = np.dot(X.T, y)
        U, s, Vt = svd(C, full_matrices=False)
        U = U[:, :n_components]
        Vt = Vt[:n_components]
        U, Vt = svd_flip(U, Vt)
        V = Vt.T

        self.x_weights_ = U
        self.y_weights_ = V
        self._n_features_out = self.x_weights_.shape[1]
        return self

    def transform(self, X, y=None):
        # [FUNCTION DOC]
        # Standardizes X with training stats and multiplies by x_weights_; if y is given
        # (1-D reshaped), also returns standardized y times y_weights_.
        # Outputs: x_scores (n, k) or (x_scores, y_scores).
        """
        Apply the dimensionality reduction.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to be transformed.

        y : array-like of shape (n_samples,) or (n_samples, n_targets), \
                default=None
            Targets.

        Returns
        -------
        x_scores : array-like or tuple of array-like
            The transformed data `X_transformed` if `y is not None`,
            `(X_transformed, y_transformed)` otherwise.
        """
        check_is_fitted(self)
        X = validate_data(self, X, dtype=np.float64, reset=False)
        Xr = (X - self._x_mean) / self._x_std
        x_scores = np.dot(Xr, self.x_weights_)
        if y is not None:
            y = check_array(y, input_name="y", ensure_2d=False, dtype=np.float64)
            if y.ndim == 1:
                y = y.reshape(-1, 1)
            yr = (y - self._y_mean) / self._y_std
            y_scores = np.dot(yr, self.y_weights_)
            return x_scores, y_scores
        return x_scores

    def fit_transform(self, X, y=None):
        # [FUNCTION DOC]
        # Fits on (X, y) and returns transform(X, y).
        # KNOWN ISSUE: y defaults to None but fit requires y.
        """Learn and apply the dimensionality reduction.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training samples.

        y : array-like of shape (n_samples,) or (n_samples, n_targets), \
                default=None
            Targets.

        Returns
        -------
        out : array-like or tuple of array-like
            The transformed data `X_transformed` if `y is not None`,
            `(X_transformed, y_transformed)` otherwise.
        """
        return self.fit(X, y).transform(X, y)
