# [MODULE DOC]
# Isotonic (monotonic) regression: the public solver function isotonic_regression, the
# helper check_increasing that guesses the monotonic direction, and the
# IsotonicRegression estimator which fits a 1-d monotonic step function and predicts by
# linear interpolation between fitted thresholds. Uses SciPy's
# optimize.isotonic_regression when SciPy >= 1.12, else a Cython PAVA fallback.
"""Isotonic regression for obtaining monotonic fit to data."""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import math
import warnings
from numbers import Real

import numpy as np
from scipy import interpolate, optimize
from scipy.stats import spearmanr

from sklearn._isotonic import _inplace_contiguous_isotonic_regression, _make_unique
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin, _fit_context
from sklearn.utils import check_array, check_consistent_length, metadata_routing
from sklearn.utils._param_validation import Interval, StrOptions, validate_params
from sklearn.utils.fixes import parse_version, sp_base_version
from sklearn.utils.validation import _check_sample_weight, check_is_fitted

__all__ = ["IsotonicRegression", "check_increasing", "isotonic_regression"]


@validate_params(
    {
        "x": ["array-like"],
        "y": ["array-like"],
    },
    prefer_skip_nested_validation=True,
)
def check_increasing(x, y):
    # [FUNCTION DOC]
    # Decides the monotonic direction: computes Spearman rho between x and y and returns
    # whether rho >= 0 (numpy bool). If rho is not exactly +/-1 and there are more than
    # 3
    # samples, computes a 95% confidence interval via the Fisher transform (arctanh of
    # rho,
    # standard error 1/sqrt(n-3), +/-1.96 SE, mapped back with tanh) and emits a
    # UserWarning
    # when the interval endpoints have different signs.
    # Inputs: x, y - array-likes of equal length n_samples.
    # Outputs: boolean. Exceptions: validation errors from validate_params only.
    # KNOWN ISSUE: if rho is NaN (constant x or y) and n > 3, math.log receives NaN and
    # returns NaN silently; result is False and the sign comparison of NaN endpoints may
    # trigger a misleading warning.
    """Determine whether y is monotonically correlated with x.

    y is found increasing or decreasing with respect to x based on a Spearman
    correlation test.

    Parameters
    ----------
    x : array-like of shape (n_samples,)
            Training data.

    y : array-like of shape (n_samples,)
        Training target.

    Returns
    -------
    increasing_bool : boolean
        Whether the relationship is increasing or decreasing.

    Notes
    -----
    The Spearman correlation coefficient is estimated from the data, and the
    sign of the resulting estimate is used as the result.

    In the event that the 95% confidence interval based on Fisher transform
    spans zero, a warning is raised.

    References
    ----------
    Fisher transformation. Wikipedia.
    https://en.wikipedia.org/wiki/Fisher_transformation

    Examples
    --------
    >>> from sklearn.isotonic import check_increasing
    >>> x, y = [1, 2, 3, 4, 5], [2, 4, 6, 8, 10]
    >>> check_increasing(x, y)
    np.True_
    >>> y = [10, 8, 6, 4, 2]
    >>> check_increasing(x, y)
    np.False_
    """

    # Calculate Spearman rho estimate and set return accordingly.
    rho, _ = spearmanr(x, y)
    increasing_bool = rho >= 0

    # Run Fisher transform to get the rho CI, but handle rho=+/-1
    if rho not in [-1.0, 1.0] and len(x) > 3:
        F = 0.5 * math.log((1.0 + rho) / (1.0 - rho))
        F_se = 1 / math.sqrt(len(x) - 3)

        # Use a 95% CI, i.e., +/-1.96 S.E.
        # https://en.wikipedia.org/wiki/Fisher_transformation
        rho_0 = math.tanh(F - 1.96 * F_se)
        rho_1 = math.tanh(F + 1.96 * F_se)

        # Warn if the CI spans zero.
        if np.sign(rho_0) != np.sign(rho_1):
            warnings.warn(
                "Confidence interval of the Spearman "
                "correlation coefficient spans zero. "
                "Determination of ``increasing`` may be "
                "suspect."
            )

    return increasing_bool


@validate_params(
    {
        "y": ["array-like"],
        "sample_weight": ["array-like", None],
        "y_min": [Interval(Real, None, None, closed="both"), None],
        "y_max": [Interval(Real, None, None, closed="both"), None],
        "increasing": ["boolean"],
    },
    prefer_skip_nested_validation=True,
)
def isotonic_regression(
    y, *, sample_weight=None, y_min=None, y_max=None, increasing=True
):
    # [FUNCTION DOC]
    # Solves weighted isotonic regression of the sequence y (order given by position).
    # Steps: validate y as 1-d float64/float32. With SciPy >= 1.12 call
    # optimize.isotonic_regression with the given weights and direction and cast back to
    # y's
    # dtype. Otherwise reverse y (and weights) when decreasing, validate/copy
    # sample_weight
    # (ones if None), run the in-place Cython PAVA, and reverse back. Finally, if y_min
    # or
    # y_max is given, clip in place to [y_min, y_max] (missing bound treated as -/+inf).
    # Inputs: y (n_samples,); sample_weight (n_samples,) or None; y_min, y_max real or
    # None;
    # increasing bool (keyword only).
    # Outputs: new ndarray (n_samples,) with the fitted monotonic values.
    # Exceptions: validation errors for bad parameters/arrays.
    # KNOWN ISSUE: in the SciPy branch sample_weight is passed through without
    # _check_sample_weight, so dtype/shape validation is left to SciPy.
    """Solve the isotonic regression model.

    Read more in the :ref:`User Guide <isotonic>`.

    Parameters
    ----------
    y : array-like of shape (n_samples,)
        The data.

    sample_weight : array-like of shape (n_samples,), default=None
        Weights on each point of the regression.
        If None, weight is set to 1 (equal weights).

    y_min : float, default=None
        Lower bound on the lowest predicted value (the minimum value may
        still be higher). If not set, defaults to -inf.

    y_max : float, default=None
        Upper bound on the highest predicted value (the maximum may still be
        lower). If not set, defaults to +inf.

    increasing : bool, default=True
        Whether to compute ``y_`` is increasing (if set to True) or decreasing
        (if set to False).

    Returns
    -------
    y_ : ndarray of shape (n_samples,)
        Isotonic fit of y.

    References
    ----------
    "Active set algorithms for isotonic regression; A unifying framework"
    by Michael J. Best and Nilotpal Chakravarti, section 3.

    Examples
    --------
    >>> from sklearn.isotonic import isotonic_regression
    >>> isotonic_regression([5, 3, 1, 2, 8, 10, 7, 9, 6, 4])
    array([2.75   , 2.75   , 2.75   , 2.75   , 7.33,
           7.33, 7.33, 7.33, 7.33, 7.33])
    """
    y = check_array(y, ensure_2d=False, input_name="y", dtype=[np.float64, np.float32])
    if sp_base_version >= parse_version("1.12.0"):
        res = optimize.isotonic_regression(
            y=y, weights=sample_weight, increasing=increasing
        )
        y = np.asarray(res.x, dtype=y.dtype)
    else:
        # TODO: remove this branch when Scipy 1.12 is the minimum supported version
        # Also remove _inplace_contiguous_isotonic_regression.
        order = np.s_[:] if increasing else np.s_[::-1]
        y = np.array(y[order], dtype=y.dtype)
        sample_weight = _check_sample_weight(sample_weight, y, dtype=y.dtype, copy=True)
        sample_weight = np.ascontiguousarray(sample_weight[order])
        _inplace_contiguous_isotonic_regression(y, sample_weight)
        y = y[order]

    if y_min is not None or y_max is not None:
        # Older versions of np.clip don't accept None as a bound, so use np.inf
        if y_min is None:
            y_min = -np.inf
        if y_max is None:
            y_max = np.inf
        np.clip(y, y_min, y_max, y)
    return y


class IsotonicRegression(RegressorMixin, TransformerMixin, BaseEstimator):
    # [CLASS DOC]
    # Estimator fitting a monotonic piecewise-linear function of one feature. After fit,
    # stores the deduplicated thresholds and an interp1d function f_ used for transform
    # and
    # predict (identical outputs). Declares the "T" argument of predict/transform as
    # unused
    # for metadata routing. f_ is dropped on pickling and rebuilt on unpickling.
    """Isotonic regression model.

    Read more in the :ref:`User Guide <isotonic>`.

    .. versionadded:: 0.13

    Parameters
    ----------
    y_min : float, default=None
        Lower bound on the lowest predicted value (the minimum value may
        still be higher). If not set, defaults to -inf.

    y_max : float, default=None
        Upper bound on the highest predicted value (the maximum may still be
        lower). If not set, defaults to +inf.

    increasing : bool or 'auto', default=True
        Determines whether the predictions should be constrained to increase
        or decrease with `X`. 'auto' will decide based on the Spearman
        correlation estimate's sign.

    out_of_bounds : {'nan', 'clip', 'raise'}, default='nan'
        Handles how `X` values outside of the training domain are handled
        during prediction.

        - 'nan', predictions will be NaN.
        - 'clip', predictions will be set to the value corresponding to
          the nearest train interval endpoint.
        - 'raise', a `ValueError` is raised.

    Attributes
    ----------
    X_min_ : float
        Minimum value of input array `X_` for left bound.

    X_max_ : float
        Maximum value of input array `X_` for right bound.

    X_thresholds_ : ndarray of shape (n_thresholds,)
        Unique ascending `X` values used to interpolate
        the y = f(X) monotonic function.

        .. versionadded:: 0.24

    y_thresholds_ : ndarray of shape (n_thresholds,)
        De-duplicated `y` values suitable to interpolate the y = f(X)
        monotonic function.

        .. versionadded:: 0.24

    f_ : function
        The stepwise interpolating function that covers the input domain ``X``.

    increasing_ : bool
        Inferred value for ``increasing``.

    See Also
    --------
    sklearn.linear_model.LinearRegression : Ordinary least squares Linear
        Regression.
    sklearn.ensemble.HistGradientBoostingRegressor : Gradient boosting that
        is a non-parametric model accepting monotonicity constraints.
    isotonic_regression : Function to solve the isotonic regression model.

    Notes
    -----
    Ties are broken using the secondary method from de Leeuw, 1977.

    References
    ----------
    Isotonic Median Regression: A Linear Programming Approach
    Nilotpal Chakravarti
    Mathematics of Operations Research
    Vol. 14, No. 2 (May, 1989), pp. 303-308

    Isotone Optimization in R : Pool-Adjacent-Violators
    Algorithm (PAVA) and Active Set Methods
    de Leeuw, Hornik, Mair
    Journal of Statistical Software 2009

    Correctness of Kruskal's algorithms for monotone regression with ties
    de Leeuw, Psychometrica, 1977

    Examples
    --------
    >>> from sklearn.datasets import make_regression
    >>> from sklearn.isotonic import IsotonicRegression
    >>> X, y = make_regression(n_samples=10, n_features=1, random_state=41)
    >>> iso_reg = IsotonicRegression().fit(X, y)
    >>> iso_reg.predict([.1, .2])
    array([1.8628, 3.7256])
    """

    # T should have been called X
    __metadata_request__predict = {"T": metadata_routing.UNUSED}
    __metadata_request__transform = {"T": metadata_routing.UNUSED}

    _parameter_constraints: dict = {
        "y_min": [Interval(Real, None, None, closed="both"), None],
        "y_max": [Interval(Real, None, None, closed="both"), None],
        "increasing": ["boolean", StrOptions({"auto"})],
        "out_of_bounds": [StrOptions({"nan", "clip", "raise"})],
    }

    def __init__(self, *, y_min=None, y_max=None, increasing=True, out_of_bounds="nan"):
        # [FUNCTION DOC]
        # Stores constructor arguments as attributes without validation (sklearn
        # convention).
        self.y_min = y_min
        self.y_max = y_max
        self.increasing = increasing
        self.out_of_bounds = out_of_bounds

    def _check_input_data_shape(self, X):
        # [FUNCTION DOC]
        # HELPER: Ensures X is 1-d or 2-d with exactly one column.
        # Exceptions: ValueError otherwise. Outputs: None.
        if not (X.ndim == 1 or (X.ndim == 2 and X.shape[1] == 1)):
            msg = (
                "Isotonic regression input X should be a 1d array or "
                "2d array with 1 feature"
            )
            raise ValueError(msg)

    def _build_f(self, X, y):
        # [FUNCTION DOC]
        # HELPER: Builds self.f_ from thresholds X, y. If there is a single point, f_ is
        # a
        # function returning y repeated to the input's shape (constant prediction);
        # otherwise a
        # linear scipy interp1d with bounds_error set when out_of_bounds == "raise"
        # (default
        # fill is NaN, implementing the "nan" option).
        # Side effects: sets self.f_.
        # KNOWN ISSUE: in the single-point case out_of_bounds="raise" is not honoured;
        # any input
        # gets the constant value.
        """Build the f_ interp1d function."""

        bounds_error = self.out_of_bounds == "raise"
        if len(y) == 1:
            # single y, constant prediction
            self.f_ = lambda x: y.repeat(x.shape)
        else:
            self.f_ = interpolate.interp1d(
                X, y, kind="linear", bounds_error=bounds_error
            )

    def _build_y(self, X, y, sample_weight, trim_duplicates=True):
        # [FUNCTION DOC]
        # HELPER: Core fit computation. Checks X shape and flattens it; sets increasing_
        # (via check_increasing when increasing == "auto"); validates sample_weight
        # (ones if
        # None) and drops samples with non-positive weight; sorts by X then y; merges
        # duplicate X
        # values with _make_unique (weighted mean y, summed weight); runs
        # isotonic_regression on
        # the merged values with y_min, y_max and increasing_; records X_min_ and
        # X_max_. When
        # trim_duplicates is True, removes interior points whose y equals both
        # neighbours'
        # (they do not change the interpolation).
        # Inputs: X, y validated arrays; sample_weight array or None; trim_duplicates
        # bool.
        # Outputs: tuple (X_thresholds, y_thresholds) of 1-d arrays.
        # Side effects: sets increasing_, X_min_, X_max_; may warn via check_increasing.
        """Build the y_ IsotonicRegression."""
        self._check_input_data_shape(X)
        X = X.reshape(-1)  # use 1d view

        # Determine increasing if auto-determination requested
        if self.increasing == "auto":
            self.increasing_ = check_increasing(X, y)
        else:
            self.increasing_ = self.increasing

        # If sample_weights is passed, removed zero-weight values and clean
        # order
        sample_weight = _check_sample_weight(sample_weight, X, dtype=X.dtype)
        mask = sample_weight > 0
        X, y, sample_weight = X[mask], y[mask], sample_weight[mask]

        order = np.lexsort((y, X))
        X, y, sample_weight = [array[order] for array in [X, y, sample_weight]]
        unique_X, unique_y, unique_sample_weight = _make_unique(X, y, sample_weight)

        X = unique_X
        y = isotonic_regression(
            unique_y,
            sample_weight=unique_sample_weight,
            y_min=self.y_min,
            y_max=self.y_max,
            increasing=self.increasing_,
        )

        # Handle the left and right bounds on X
        self.X_min_, self.X_max_ = np.min(X), np.max(X)

        if trim_duplicates:
            # Remove unnecessary points for faster prediction
            keep_data = np.ones((len(y),), dtype=bool)
            # Aside from the 1st and last point, remove points whose y values
            # are equal to both the point before and the point after it.
            keep_data[1:-1] = np.logical_or(
                np.not_equal(y[1:-1], y[:-2]), np.not_equal(y[1:-1], y[2:])
            )
            return X[keep_data], y[keep_data]
        else:
            # The ability to turn off trim_duplicates is only used to it make
            # easier to unit test that removing duplicates in y does not have
            # any impact the resulting interpolation function (besides
            # prediction speed).
            return X, y

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y, sample_weight=None):
        # [FUNCTION DOC]
        # Validates X (non-sparse, float64/float32, 1-d allowed) and y (same dtype),
        # checks
        # consistent lengths with sample_weight, computes thresholds with _build_y,
        # stores them
        # as X_thresholds_ / y_thresholds_ (so the model can be pickled) and builds f_.
        # Outputs: self. Side effects: sets the fitted attributes listed above.
        # Exceptions: parameter validation errors; ValueError for bad shapes/lengths.
        """Fit the model using X, y as training data.

        Parameters
        ----------
        X : array-like of shape (n_samples,) or (n_samples, 1)
            Training data.

            .. versionchanged:: 0.24
               Also accepts 2d array with 1 feature.

        y : array-like of shape (n_samples,)
            Training target.

        sample_weight : array-like of shape (n_samples,), default=None
            Weights. If set to None, all weights will be set to 1 (equal
            weights).

        Returns
        -------
        self : object
            Returns an instance of self.

        Notes
        -----
        X is stored for future use, as :meth:`transform` needs X to interpolate
        new input data.
        """
        check_params = dict(accept_sparse=False, ensure_2d=False)
        X = check_array(
            X, input_name="X", dtype=[np.float64, np.float32], **check_params
        )
        y = check_array(y, input_name="y", dtype=X.dtype, **check_params)
        check_consistent_length(X, y, sample_weight)

        # Transform y by running the isotonic regression algorithm and
        # transform X accordingly.
        X, y = self._build_y(X, y, sample_weight)

        # It is necessary to store the non-redundant part of the training set
        # on the model to make it possible to support model persistence via
        # the pickle module as the object built by scipy.interp1d is not
        # picklable directly.
        self.X_thresholds_, self.y_thresholds_ = X, y

        # Build the interpolation function
        self._build_f(X, y)
        return self

    def _transform(self, T):
        # [FUNCTION DOC]
        # HELPER: Shared implementation of transform/predict. Validates T with the dtype
        # of
        # X_thresholds_ (float64 if unfitted), checks shape and flattens it, clips to
        # [X_min_, X_max_] when out_of_bounds == "clip", evaluates f_ and casts the
        # result to
        # T's dtype. Outputs: 1-d ndarray (n_samples,).
        # Exceptions: ValueError for bad shape or (with "raise") out-of-range values.
        # KNOWN ISSUE: no check_is_fitted call, so an unfitted estimator fails with
        # AttributeError on f_ instead of NotFittedError.
        """`_transform` is called by both `transform` and `predict` methods.

        Since `transform` is wrapped to output arrays of specific types (e.g.
        NumPy arrays, pandas DataFrame), we cannot make `predict` call `transform`
        directly.

        The above behaviour could be changed in the future, if we decide to output
        other type of arrays when calling `predict`.
        """
        if hasattr(self, "X_thresholds_"):
            dtype = self.X_thresholds_.dtype
        else:
            dtype = np.float64

        T = check_array(T, dtype=dtype, ensure_2d=False)

        self._check_input_data_shape(T)
        T = T.reshape(-1)  # use 1d view

        if self.out_of_bounds == "clip":
            T = np.clip(T, self.X_min_, self.X_max_)

        res = self.f_(T)

        # on scipy 0.17, interp1d up-casts to float64, so we cast back
        res = res.astype(T.dtype)

        return res

    def transform(self, T):
        # [FUNCTION DOC]
        # Public transform; delegates to _transform (output may be wrapped by
        # set_output).
        """Transform new data by linear interpolation.

        Parameters
        ----------
        T : array-like of shape (n_samples,) or (n_samples, 1)
            Data to transform.

            .. versionchanged:: 0.24
               Also accepts 2d array with 1 feature.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            The transformed data.
        """
        return self._transform(T)

    def predict(self, T):
        # [FUNCTION DOC]
        # Public predict; delegates to _transform (not wrapped by set_output).
        """Predict new data by linear interpolation.

        Parameters
        ----------
        T : array-like of shape (n_samples,) or (n_samples, 1)
            Data to transform.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Transformed data.
        """
        return self._transform(T)

    # We implement get_feature_names_out here instead of using
    # `ClassNamePrefixFeaturesOutMixin`` because `input_features` are ignored.
    # `input_features` are ignored because `IsotonicRegression` accepts 1d
    # arrays and the semantics of `feature_names_in_` are not clear for 1d arrays.
    def get_feature_names_out(self, input_features=None):
        # [FUNCTION DOC]
        # Returns object ndarray with the single name "<lowercased class name>0";
        # input_features
        # is ignored. Exceptions: NotFittedError if f_ is missing.
        """Get output feature names for transformation.

        Parameters
        ----------
        input_features : array-like of str or None, default=None
            Ignored.

        Returns
        -------
        feature_names_out : ndarray of str objects
            An ndarray with one string i.e. ["isotonicregression0"].
        """
        check_is_fitted(self, "f_")
        class_name = self.__class__.__name__.lower()
        return np.asarray([f"{class_name}0"], dtype=object)

    def __getstate__(self):
        # [FUNCTION DOC]
        # Returns the parent's pickled state dict without the unpicklable f_ entry.
        """Pickle-protocol - return state of the estimator."""
        state = super().__getstate__()
        # remove interpolation method
        state.pop("f_", None)
        return state

    def __setstate__(self, state):
        # [FUNCTION DOC]
        # Restores state via the parent, then rebuilds f_ from the stored thresholds if
        # present.
        # Side effects: sets self.f_.
        """Pickle-protocol - set state of the estimator.

        We need to rebuild the interpolation function.
        """
        super().__setstate__(state)
        if hasattr(self, "X_thresholds_") and hasattr(self, "y_thresholds_"):
            self._build_f(self.X_thresholds_, self.y_thresholds_)

    def __sklearn_tags__(self):
        # [FUNCTION DOC]
        # Extends parent tags: accepts 1-d arrays and declares 2-d arrays as not
        # supported.
        tags = super().__sklearn_tags__()
        tags.input_tags.one_d_array = True
        tags.input_tags.two_d_array = False
        return tags
