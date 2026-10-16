# [MODULE DOC]
# Theil-Sen robust multivariate linear regression. Fits ordinary least squares on
# many subsets of the samples and returns the spatial (L1) median of the resulting
# coefficient vectors. Contents: Weiszfeld spatial-median helpers, breakdown point
# approximation, batched LAPACK least squares helper and TheilSenRegressor.
"""
A Theil-Sen Estimator for Multiple Linear Regression Model
"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import warnings
from itertools import combinations
from numbers import Integral, Real

import numpy as np
from joblib import effective_n_jobs
from scipy import linalg
from scipy.linalg.lapack import get_lapack_funcs
from scipy.special import binom

from sklearn.base import RegressorMixin, _fit_context
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model._base import LinearModel
from sklearn.utils import check_random_state
from sklearn.utils._param_validation import Interval
from sklearn.utils.parallel import Parallel, delayed
from sklearn.utils.validation import validate_data

_EPSILON = np.finfo(np.double).eps


def _modified_weiszfeld_step(X, x_old):
    # [FUNCTION DOC]
    # HELPER: One iteration of the modified Weiszfeld algorithm for the spatial median.
    # Computes the Euclidean distance of each row of X to x_old, drops rows closer
    # than machine epsilon (and records whether x_old coincides with a sample). The
    # candidate is the distance-weighted mean of the remaining rows (weights 1/dist).
    # The result blends the candidate and x_old: if x_old is not a sample it is the
    # candidate; otherwise the factors are max(0, 1 - 1/r) and min(1, 1/r) where r is
    # the norm of the sum of unit vectors from x_old to the remaining samples.
    # If r is below epsilon the candidate becomes the scalar 1.0 and r is set to 1.
    # Inputs: X - ndarray (n_samples, n_features); x_old - ndarray (n_features,).
    # Outputs: ndarray (n_features,), the new iterate.
    # KNOWN ISSUE: in the degenerate branch new_direction is the scalar 1.0, so the
    # returned point is not a meaningful location (it is only reached when x_old is a
    # sample, where its coefficient becomes 0, so the result is x_old; if x_old is not a
    # sample it returns a vector of ones).
    """Modified Weiszfeld step.

    This function defines one iteration step in order to approximate the
    spatial median (L1 median). It is a form of an iteratively re-weighted
    least squares method.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Training vector, where `n_samples` is the number of samples and
        `n_features` is the number of features.

    x_old : ndarray of shape = (n_features,)
        Current start vector.

    Returns
    -------
    x_new : ndarray of shape (n_features,)
        New iteration step.

    References
    ----------
    - On Computation of Spatial Median for Robust Data Mining, 2005
      T. Kärkkäinen and S. Äyrämö
      http://users.jyu.fi/~samiayr/pdf/ayramo_eurogen05.pdf
    """
    diff = X - x_old
    diff_norm = np.sqrt(np.sum(diff**2, axis=1))
    mask = diff_norm >= _EPSILON
    # x_old equals one of our samples
    is_x_old_in_X = int(mask.sum() < X.shape[0])

    diff = diff[mask]
    diff_norm = diff_norm[mask][:, np.newaxis]
    quotient_norm = linalg.norm(np.sum(diff / diff_norm, axis=0))

    if quotient_norm > _EPSILON:  # to avoid division by zero
        new_direction = np.sum(X[mask, :] / diff_norm, axis=0) / np.sum(
            1 / diff_norm, axis=0
        )
    else:
        new_direction = 1.0
        quotient_norm = 1.0

    return (
        max(0.0, 1.0 - is_x_old_in_X / quotient_norm) * new_direction
        + min(1.0, is_x_old_in_X / quotient_norm) * x_old
    )


def _spatial_median(X, max_iter=300, tol=1.0e-3):
    # [FUNCTION DOC]
    # HELPER: Iterates _modified_weiszfeld_step starting from the column mean until the
    # squared change between iterates is below tol squared or max_iter is reached.
    # If X has a single column, returns the ordinary median directly.
    # Inputs: X - ndarray (n_points, n_dims); max_iter - int, default 300;
    # tol - float, default 1e-3 (compared against the squared step norm after squaring).
    # Outputs: tuple (n_iter, spatial_median); n_iter is the zero-based index of the
    # last iteration (1 in the single-column case); spatial_median is (n_dims,).
    # Side effects: ConvergenceWarning when the loop exhausts max_iter.
    # KNOWN ISSUE: the docstring lists the returns as (spatial_median, n_iter) but the
    # code returns (n_iter, spatial_median). n_iter is the loop index, so it is one less
    # than the number of iterations performed. With max_iter=0 (allowed by
    # TheilSenRegressor constraints) the loop never runs and n_iter and spatial_median
    # are undefined, raising UnboundLocalError.
    """Spatial median (L1 median).

    The spatial median is member of a class of so-called M-estimators which
    are defined by an optimization problem. Given a number of p points in an
    n-dimensional space, the point x minimizing the sum of all distances to the
    p other points is called spatial median.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Training vector, where `n_samples` is the number of samples and
        `n_features` is the number of features.

    max_iter : int, default=300
        Maximum number of iterations.

    tol : float, default=1.e-3
        Stop the algorithm if spatial_median has converged.

    Returns
    -------
    spatial_median : ndarray of shape = (n_features,)
        Spatial median.

    n_iter : int
        Number of iterations needed.

    References
    ----------
    - On Computation of Spatial Median for Robust Data Mining, 2005
      T. Kärkkäinen and S. Äyrämö
      http://users.jyu.fi/~samiayr/pdf/ayramo_eurogen05.pdf
    """
    if X.shape[1] == 1:
        return 1, np.median(X.ravel(), keepdims=True)

    tol **= 2  # We are computing the tol on the squared norm
    spatial_median_old = np.mean(X, axis=0)

    for n_iter in range(max_iter):
        spatial_median = _modified_weiszfeld_step(X, spatial_median_old)
        if np.sum((spatial_median_old - spatial_median) ** 2) < tol:
            break
        else:
            spatial_median_old = spatial_median
    else:
        warnings.warn(
            "Maximum number of iterations {max_iter} reached in "
            "spatial median for TheilSen regressor."
            "".format(max_iter=max_iter),
            ConvergenceWarning,
        )
    return n_iter, spatial_median


def _breakdown_point(n_samples, n_subsamples):
    # [FUNCTION DOC]
    # HELPER: Approximates the breakdown point as
    # 1 - (0.5^(1/n_subsamples) * (n_samples - n_subsamples + 1) + n_subsamples - 1)
    # divided by n_samples.
    # Inputs: n_samples, n_subsamples - positive ints. Outputs: float.
    """Approximation of the breakdown point.

    Parameters
    ----------
    n_samples : int
        Number of samples.

    n_subsamples : int
        Number of subsamples to consider.

    Returns
    -------
    breakdown_point : float
        Approximation of breakdown point.
    """
    return (
        1
        - (
            0.5 ** (1 / n_subsamples) * (n_samples - n_subsamples + 1)
            + n_subsamples
            - 1
        )
        / n_samples
    )


def _lstsq(X, y, indices, fit_intercept):
    # [FUNCTION DOC]
    # HELPER: Solves one least squares problem per row of indices using the LAPACK
    # gelss routine. A design buffer of shape (n_subsamples, n_features [+1]) has a
    # leading column of ones when fit_intercept is true; for each subset its other
    # columns are filled with the selected rows of X, and a zero-padded target buffer
    # (length max(n_subsamples, n_dims)) is filled with y[subset]. The first n_dims
    # entries of the gelss solution are stored.
    # Inputs: X - ndarray (n_samples, n_features); y - ndarray (n_samples,);
    # indices - 2D int array (n_subsets, n_subsamples); fit_intercept - bool.
    # Outputs: float ndarray (n_subsets, n_features + intercept), intercept first.
    # Side effects: none outside local buffers (buffers are reused across subsets).
    """Least Squares Estimator for TheilSenRegressor class.

    This function calculates the least squares method on a subset of rows of X
    and y defined by the indices array. Optionally, an intercept column is
    added if intercept is set to true.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Design matrix, where `n_samples` is the number of samples and
        `n_features` is the number of features.

    y : ndarray of shape (n_samples,)
        Target vector, where `n_samples` is the number of samples.

    indices : ndarray of shape (n_subpopulation, n_subsamples)
        Indices of all subsamples with respect to the chosen subpopulation.

    fit_intercept : bool
        Fit intercept or not.

    Returns
    -------
    weights : ndarray of shape (n_subpopulation, n_features + intercept)
        Solution matrix of n_subpopulation solved least square problems.
    """
    fit_intercept = int(fit_intercept)
    n_features = X.shape[1] + fit_intercept
    n_subsamples = indices.shape[1]
    weights = np.empty((indices.shape[0], n_features))
    X_subpopulation = np.ones((n_subsamples, n_features))
    # gelss need to pad y_subpopulation to be of the max dim of X_subpopulation
    y_subpopulation = np.zeros((max(n_subsamples, n_features)))
    (lstsq,) = get_lapack_funcs(("gelss",), (X_subpopulation, y_subpopulation))

    for index, subset in enumerate(indices):
        X_subpopulation[:, fit_intercept:] = X[subset, :]
        y_subpopulation[:n_subsamples] = y[subset]
        weights[index] = lstsq(X_subpopulation, y_subpopulation)[1][:n_features]

    return weights


class TheilSenRegressor(RegressorMixin, LinearModel):
    # [CLASS DOC]
    # Estimator implementing Theil-Sen regression. Hyperparameters are validated by
    # _parameter_constraints (max_subpopulation is a Real >= 1, max_iter >= 0, tol >= 0).
    # Fitted attributes: coef_, intercept_, breakdown_, n_iter_, n_subpopulation_ and
    # the standard n_features_in_ / feature_names_in_. Prediction comes from LinearModel.
    """Theil-Sen Estimator: robust multivariate regression model.

    The algorithm calculates least square solutions on subsets with size
    n_subsamples of the samples in X. Any value of n_subsamples between the
    number of features and samples leads to an estimator with a compromise
    between robustness and efficiency. Since the number of least square
    solutions is "n_samples choose n_subsamples", it can be extremely large
    and can therefore be limited with max_subpopulation. If this limit is
    reached, the subsets are chosen randomly. In a final step, the spatial
    median (or L1 median) is calculated of all least square solutions.

    Read more in the :ref:`User Guide <theil_sen_regression>`.

    Parameters
    ----------
    fit_intercept : bool, default=True
        Whether to calculate the intercept for this model. If set
        to false, no intercept will be used in calculations.

    max_subpopulation : int, default=1e4
        Instead of computing with a set of cardinality 'n choose k', where n is
        the number of samples and k is the number of subsamples (at least
        number of features), consider only a stochastic subpopulation of a
        given maximal size if 'n choose k' is larger than max_subpopulation.
        For other than small problem sizes this parameter will determine
        memory usage and runtime if n_subsamples is not changed. Note that the
        data type should be int but floats such as 1e4 can be accepted too.

    n_subsamples : int, default=None
        Number of samples to calculate the parameters. This is at least the
        number of features (plus 1 if fit_intercept=True) and the number of
        samples as a maximum. A lower number leads to a higher breakdown
        point and a low efficiency while a high number leads to a low
        breakdown point and a high efficiency. If None, take the
        minimum number of subsamples leading to maximal robustness.
        If n_subsamples is set to n_samples, Theil-Sen is identical to least
        squares.

    max_iter : int, default=300
        Maximum number of iterations for the calculation of spatial median.

    tol : float, default=1e-3
        Tolerance when calculating spatial median.

    random_state : int, RandomState instance or None, default=None
        A random number generator instance to define the state of the random
        permutations generator. Pass an int for reproducible output across
        multiple function calls.
        See :term:`Glossary <random_state>`.

    n_jobs : int, default=None
        Number of CPUs to use during the cross validation.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    verbose : bool, default=False
        Verbose mode when fitting the model.

    Attributes
    ----------
    coef_ : ndarray of shape (n_features,)
        Coefficients of the regression model (median of distribution).

    intercept_ : float
        Estimated intercept of regression model.

    breakdown_ : float
        Approximated breakdown point.

    n_iter_ : int
        Number of iterations needed for the spatial median.

    n_subpopulation_ : int
        Number of combinations taken into account from 'n choose k', where n is
        the number of samples and k is the number of subsamples.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    HuberRegressor : Linear regression model that is robust to outliers.
    RANSACRegressor : RANSAC (RANdom SAmple Consensus) algorithm.
    SGDRegressor : Fitted by minimizing a regularized empirical loss with SGD.

    References
    ----------
    - Theil-Sen Estimators in a Multiple Linear Regression Model, 2009
      Xin Dang, Hanxiang Peng, Xueqin Wang and Heping Zhang
      http://home.olemiss.edu/~xdang/papers/MTSE.pdf

    Examples
    --------
    >>> from sklearn.linear_model import TheilSenRegressor
    >>> from sklearn.datasets import make_regression
    >>> X, y = make_regression(
    ...     n_samples=200, n_features=2, noise=4.0, random_state=0)
    >>> reg = TheilSenRegressor(random_state=0).fit(X, y)
    >>> reg.score(X, y)
    0.9884
    >>> reg.predict(X[:1,])
    array([-31.5871])
    """

    _parameter_constraints: dict = {
        "fit_intercept": ["boolean"],
        # target_type should be Integral but can accept Real for backward compatibility
        "max_subpopulation": [Interval(Real, 1, None, closed="left")],
        "n_subsamples": [None, Integral],
        "max_iter": [Interval(Integral, 0, None, closed="left")],
        "tol": [Interval(Real, 0.0, None, closed="left")],
        "random_state": ["random_state"],
        "n_jobs": [None, Integral],
        "verbose": ["verbose"],
    }

    def __init__(
        self,
        *,
        fit_intercept=True,
        max_subpopulation=1e4,
        n_subsamples=None,
        max_iter=300,
        tol=1.0e-3,
        random_state=None,
        n_jobs=None,
        verbose=False,
    ):
        # [FUNCTION DOC]
        # Stores the hyperparameters unchanged (sklearn convention: no validation here).
        # Inputs: see class docstring. Side effects: sets the attributes of the same names.
        self.fit_intercept = fit_intercept
        self.max_subpopulation = max_subpopulation
        self.n_subsamples = n_subsamples
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose

    def _check_subparams(self, n_samples, n_features):
        # [FUNCTION DOC]
        # HELPER: Resolves the effective subset size and number of subsets.
        # n_dim is n_features plus 1 if fit_intercept. If n_subsamples is given it must be
        # <= n_samples; when n_samples >= n_features it must be >= n_dim, otherwise (fewer
        # samples than features) it must equal n_samples. If None it defaults to
        # min(n_dim, n_samples). n_subpopulation is min(max_subpopulation, round(binomial
        # coefficient n_samples choose n_subsamples), at least 1) cast to int.
        # Inputs: n_samples, n_features - ints.
        # Outputs: tuple (n_subsamples, n_subpopulation).
        # Exceptions: ValueError on any of the violated constraints above.
        # KNOWN ISSUE: the n_samples >= n_features test ignores the intercept column, so
        # with n_samples == n_features and fit_intercept the defaults give n_subsamples =
        # n_samples < n_dim (underdetermined subproblems) without an error.
        n_subsamples = self.n_subsamples

        if self.fit_intercept:
            n_dim = n_features + 1
        else:
            n_dim = n_features

        if n_subsamples is not None:
            if n_subsamples > n_samples:
                raise ValueError(
                    "Invalid parameter since n_subsamples > "
                    "n_samples ({0} > {1}).".format(n_subsamples, n_samples)
                )
            if n_samples >= n_features:
                if n_dim > n_subsamples:
                    plus_1 = "+1" if self.fit_intercept else ""
                    raise ValueError(
                        "Invalid parameter since n_features{0} "
                        "> n_subsamples ({1} > {2})."
                        "".format(plus_1, n_dim, n_subsamples)
                    )
            else:  # if n_samples < n_features
                if n_subsamples != n_samples:
                    raise ValueError(
                        "Invalid parameter since n_subsamples != "
                        "n_samples ({0} != {1}) while n_samples "
                        "< n_features.".format(n_subsamples, n_samples)
                    )
        else:
            n_subsamples = min(n_dim, n_samples)

        all_combinations = max(1, np.rint(binom(n_samples, n_subsamples)))
        n_subpopulation = int(min(self.max_subpopulation, all_combinations))

        return n_subsamples, n_subpopulation

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y):
        # [FUNCTION DOC]
        # Fits the model. Validates X, y (numeric y), resolves subset parameters, stores
        # breakdown_ and n_subpopulation_, optionally prints diagnostics. If the total
        # number of combinations is at most max_subpopulation, enumerates all
        # n_subsamples-combinations of sample indices; otherwise draws n_subpopulation_
        # random subsets without replacement (subsets may repeat). Splits the subsets into
        # effective n_jobs chunks, runs _lstsq on each in parallel, stacks the solutions
        # and takes their spatial median. The first coordinate is intercept_ when
        # fit_intercept, else intercept_ is 0.0 and all coordinates are coef_.
        # Inputs: X - array-like (n_samples, n_features); y - array-like (n_samples,).
        # Outputs: self.
        # Exceptions: ValueError from validation or _check_subparams.
        # Side effects: sets n_features_in_, n_subpopulation_, breakdown_, n_iter_,
        # intercept_, coef_; consumes random_state; prints when verbose; may warn
        # ConvergenceWarning.
        # KNOWN ISSUE: enumerating all combinations materializes a Python list of tuples,
        # which can be large near the max_subpopulation threshold.
        """Fit linear model.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data.
        y : ndarray of shape (n_samples,)
            Target values.

        Returns
        -------
        self : returns an instance of self.
            Fitted `TheilSenRegressor` estimator.
        """
        random_state = check_random_state(self.random_state)
        X, y = validate_data(self, X, y, y_numeric=True)
        n_samples, n_features = X.shape
        n_subsamples, self.n_subpopulation_ = self._check_subparams(
            n_samples, n_features
        )
        self.breakdown_ = _breakdown_point(n_samples, n_subsamples)

        if self.verbose:
            print("Breakdown point: {0}".format(self.breakdown_))
            print("Number of samples: {0}".format(n_samples))
            tol_outliers = int(self.breakdown_ * n_samples)
            print("Tolerable outliers: {0}".format(tol_outliers))
            print("Number of subpopulations: {0}".format(self.n_subpopulation_))

        # Determine indices of subpopulation
        if np.rint(binom(n_samples, n_subsamples)) <= self.max_subpopulation:
            indices = list(combinations(range(n_samples), n_subsamples))
        else:
            indices = [
                random_state.choice(n_samples, size=n_subsamples, replace=False)
                for _ in range(self.n_subpopulation_)
            ]

        n_jobs = effective_n_jobs(self.n_jobs)
        index_list = np.array_split(indices, n_jobs)
        weights = Parallel(n_jobs=n_jobs, verbose=self.verbose)(
            delayed(_lstsq)(X, y, index_list[job], self.fit_intercept)
            for job in range(n_jobs)
        )
        weights = np.vstack(weights)
        self.n_iter_, coefs = _spatial_median(
            weights, max_iter=self.max_iter, tol=self.tol
        )

        if self.fit_intercept:
            self.intercept_ = coefs[0]
            self.coef_ = coefs[1:]
        else:
            self.intercept_ = 0.0
            self.coef_ = coefs

        return self
