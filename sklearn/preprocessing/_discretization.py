# [MODULE DOC]
# Defines KBinsDiscretizer, which bins each continuous feature into intervals (uniform,
#   quantile or 1D k-means edges) and encodes bin indices as ordinal values or one-hot
#   columns via an internal OneHotEncoder.
# Side effects at import: none (KMeans is imported lazily inside fit to avoid import
#   cycles).
# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause


import warnings
from numbers import Integral

import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin, _fit_context
from sklearn.preprocessing._encoders import OneHotEncoder
from sklearn.utils import resample
from sklearn.utils._param_validation import Interval, Options, StrOptions
from sklearn.utils.stats import _weighted_percentile
from sklearn.utils.validation import (
    _check_feature_names_in,
    _check_sample_weight,
    check_array,
    check_is_fitted,
    validate_data,
)


class KBinsDiscretizer(TransformerMixin, BaseEstimator):
    # [CLASS DOC]
    # Per-feature binning transformer. fit learns bin_edges_ (object array of 1D float
    #   edge arrays) and n_bins_ (int array, possibly reduced after merging tiny bins),
    #   and for one-hot encodings a pre-fitted private _encoder (OneHotEncoder with
    #   categories 0..n_bins_-1).
    # KNOWN ISSUE: the docstring mentions "ignored features" with empty edge arrays, but
    #   no feature-ignoring mechanism exists in the code.
    """
    Bin continuous data into intervals.

    Read more in the :ref:`User Guide <preprocessing_discretization>`.

    .. versionadded:: 0.20

    Parameters
    ----------
    n_bins : int or array-like of shape (n_features,), default=5
        The number of bins to produce. Raises ValueError if ``n_bins < 2``.

    encode : {'onehot', 'onehot-dense', 'ordinal'}, default='onehot'
        Method used to encode the transformed result.

        - 'onehot': Encode the transformed result with one-hot encoding
          and return a sparse matrix. Ignored features are always
          stacked to the right.
        - 'onehot-dense': Encode the transformed result with one-hot encoding
          and return a dense array. Ignored features are always
          stacked to the right.
        - 'ordinal': Return the bin identifier encoded as an integer value.

    strategy : {'uniform', 'quantile', 'kmeans'}, default='quantile'
        Strategy used to define the widths of the bins.

        - 'uniform': All bins in each feature have identical widths.
        - 'quantile': All bins in each feature have the same number of points.
        - 'kmeans': Values in each bin have the same nearest center of a 1D
          k-means cluster.

        For an example of the different strategies see:
        :ref:`sphx_glr_auto_examples_preprocessing_plot_discretization_strategies.py`.

    quantile_method : {"inverted_cdf", "averaged_inverted_cdf",
            "closest_observation", "interpolated_inverted_cdf", "hazen",
            "weibull", "linear", "median_unbiased", "normal_unbiased"},
            default="averaged_inverted_cdf"
            Method to pass on to np.percentile calculation when using
            strategy="quantile". Only `averaged_inverted_cdf` and `inverted_cdf`
            support the use of `sample_weight != None` when subsampling is not
            active.

            .. versionadded:: 1.7

            .. versionchanged:: 1.9
                The default value changed from `"linear"` to `"averaged_inverted_cdf"`.

    dtype : {np.float32, np.float64}, default=None
        The desired data-type for the output. If None, output dtype is
        consistent with input dtype. Only np.float32 and np.float64 are
        supported.

        .. versionadded:: 0.24

    subsample : int or None, default=200_000
        Maximum number of samples, used to fit the model, for computational
        efficiency.
        `subsample=None` means that all the training samples are used when
        computing the quantiles that determine the binning thresholds.
        Since quantile computation relies on sorting each column of `X` and
        that sorting has an `n log(n)` time complexity,
        it is recommended to use subsampling on datasets with a
        very large number of samples.

        .. versionchanged:: 1.3
            The default value of `subsample` changed from `None` to `200_000` when
            `strategy="quantile"`.

        .. versionchanged:: 1.5
            The default value of `subsample` changed from `None` to `200_000` when
            `strategy="uniform"` or `strategy="kmeans"`.

    random_state : int, RandomState instance or None, default=None
        Determines random number generation for subsampling.
        Pass an int for reproducible results across multiple function calls.
        See the `subsample` parameter for more details.
        See :term:`Glossary <random_state>`.

        .. versionadded:: 1.1

    Attributes
    ----------
    bin_edges_ : ndarray of ndarray of shape (n_features,)
        The edges of each bin. Contain arrays of varying shapes ``(n_bins_, )``
        Ignored features will have empty arrays.

    n_bins_ : ndarray of shape (n_features,), dtype=np.int64
        Number of bins per feature. Bins whose width are too small
        (i.e., <= 1e-8) are removed with a warning.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    Binarizer : Class used to bin values as ``0`` or
        ``1`` based on a parameter ``threshold``.

    Notes
    -----

    For a visualization of discretization on different datasets refer to
    :ref:`sphx_glr_auto_examples_preprocessing_plot_discretization_classification.py`.
    On the effect of discretization on linear models see:
    :ref:`sphx_glr_auto_examples_preprocessing_plot_discretization.py`.

    In bin edges for feature ``i``, the first and last values are used only for
    ``inverse_transform``. During transform, bin edges are extended to::

      np.concatenate([-np.inf, bin_edges_[i][1:-1], np.inf])

    You can combine ``KBinsDiscretizer`` with
    :class:`~sklearn.compose.ColumnTransformer` if you only want to preprocess
    part of the features.

    ``KBinsDiscretizer`` might produce constant features (e.g., when
    ``encode = 'onehot'`` and certain bins do not contain any data).
    These features can be removed with feature selection algorithms
    (e.g., :class:`~sklearn.feature_selection.VarianceThreshold`).

    Examples
    --------
    >>> from sklearn.preprocessing import KBinsDiscretizer
    >>> X = [[-2, 1, -4,   -1],
    ...      [-1, 2, -3, -0.5],
    ...      [ 0, 3, -2,  0.5],
    ...      [ 1, 4, -1,    2]]
    >>> est = KBinsDiscretizer(
    ...     n_bins=3, encode='ordinal', strategy='uniform'
    ... )
    >>> est.fit(X)
    KBinsDiscretizer(...)
    >>> Xt = est.transform(X)
    >>> Xt  # doctest: +SKIP
    array([[ 0., 0., 0., 0.],
           [ 1., 1., 1., 0.],
           [ 2., 2., 2., 1.],
           [ 2., 2., 2., 2.]])

    Sometimes it may be useful to convert the data back into the original
    feature space. The ``inverse_transform`` function converts the binned
    data into the original feature space. Each value will be equal to the mean
    of the two bin edges.

    >>> est.bin_edges_[0]
    array([-2., -1.,  0.,  1.])
    >>> est.inverse_transform(Xt)
    array([[-1.5,  1.5, -3.5, -0.5],
           [-0.5,  2.5, -2.5, -0.5],
           [ 0.5,  3.5, -1.5,  0.5],
           [ 0.5,  3.5, -1.5,  1.5]])

    While this preprocessing step can be an optimization, it is important
    to note the array returned by ``inverse_transform`` will have an internal type
    of ``np.float64`` or ``np.float32``, denoted by the ``dtype`` input argument.
    This can drastically increase the memory usage of the array. See the
    :ref:`sphx_glr_auto_examples_cluster_plot_face_compress.py`
    where `KBinsDescretizer` is used to cluster the image into bins and increases
    the size of the image by 8x.
    """

    _parameter_constraints: dict = {
        "n_bins": [Interval(Integral, 2, None, closed="left"), "array-like"],
        "encode": [StrOptions({"onehot", "onehot-dense", "ordinal"})],
        "strategy": [StrOptions({"uniform", "quantile", "kmeans"})],
        "quantile_method": [
            StrOptions(
                {
                    "inverted_cdf",
                    "averaged_inverted_cdf",
                    "closest_observation",
                    "interpolated_inverted_cdf",
                    "hazen",
                    "weibull",
                    "linear",
                    "median_unbiased",
                    "normal_unbiased",
                }
            )
        ],
        "dtype": [Options(type, {np.float64, np.float32}), None],
        "subsample": [Interval(Integral, 1, None, closed="left"), None],
        "random_state": ["random_state"],
    }

    def __init__(
        self,
        n_bins=5,
        *,
        encode="onehot",
        strategy="quantile",
        quantile_method="averaged_inverted_cdf",
        dtype=None,
        subsample=200_000,
        random_state=None,
    ):
        # [FUNCTION DOC]
        # Stores constructor parameters verbatim; validation happens in fit.
        self.n_bins = n_bins
        self.encode = encode
        self.strategy = strategy
        self.quantile_method = quantile_method
        self.dtype = dtype
        self.subsample = subsample
        self.random_state = random_state

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None, sample_weight=None):
        # [FUNCTION DOC]
        # Behavior: validate X as numeric; choose output dtype (self.dtype or X.dtype);
        #   validate sample_weight; if subsample is set and n_samples exceeds it,
        #   resample X with replacement (weighted by sample_weight, using random_state)
        #   and drop the weights afterwards. Resolve per-feature n_bins via
        #   _validate_n_bins.
        # For each feature: compute min/max (ignoring zero-weight samples for non-
        #   quantile strategies); constant features warn, get n_bins 1 and edges [-inf,
        #   inf]. 'uniform': evenly spaced edges from min to max. 'quantile':
        #   percentiles at evenly spaced levels (np.percentile with quantile_method
        #   unless 'linear', or _weighted_percentile with average for
        #   'averaged_inverted_cdf' when weighted). 'kmeans': 1D KMeans with centers
        #   initialised at uniform bin midpoints, sorted centers, edges at midpoints
        #   between centers plus min and max. For quantile/kmeans, drop edges closer
        #   than 1e-8 to the previous one and warn if the bin count shrank.
        # Inputs: X (n_samples, n_features) numeric; y ignored; sample_weight optional
        #   (n_samples,).
        # Outputs: self. Sets bin_edges_, n_bins_, n_features_in_, feature_names_in_,
        #   and _encoder when encode contains 'onehot' (fitted on a single zero row).
        # Exceptions: ValueError for weighted quantile with an unsupported
        #   quantile_method, and from _validate_n_bins.
        # Side effects: UserWarnings for constant features and removed bins; randomness
        #   via random_state when subsampling (KMeans has no random_state but init is
        #   deterministic).
        # KNOWN ISSUE: constant features get edges [-inf, inf], so inverse_transform
        #   produces NaN (mean of -inf and inf) for them.
        # KNOWN ISSUE: refitting with encode='ordinal' after a one-hot fit leaves the
        #   old _encoder attribute, so get_feature_names_out would still return one-hot
        #   names.
        """
        Fit the estimator.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to be discretized.

        y : None
            Ignored. This parameter exists only for compatibility with
            :class:`~sklearn.pipeline.Pipeline`.

        sample_weight : ndarray of shape (n_samples,)
            Contains weight values to be associated with each sample.

            .. versionadded:: 1.3

            .. versionchanged:: 1.7
               Added support for strategy="uniform".

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        X = validate_data(self, X, dtype="numeric")

        if self.dtype in (np.float64, np.float32):
            output_dtype = self.dtype
        else:  # self.dtype is None
            output_dtype = X.dtype

        n_samples, n_features = X.shape

        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X, dtype=X.dtype)

        if self.subsample is not None and n_samples > self.subsample:
            # Take a subsample of `X`
            # When resampling, it is important to subsample **with replacement** to
            # preserve the distribution, in particular in the presence of a few data
            # points with large weights. You can check this by setting `replace=False`
            # in sklearn.utils.test.test_indexing.test_resample_weighted and check that
            # it fails as a justification for this claim.
            X = resample(
                X,
                replace=True,
                n_samples=self.subsample,
                random_state=self.random_state,
                sample_weight=sample_weight,
            )
            # Since we already used the weights when resampling when provided,
            # we set them back to `None` to avoid accounting for the weights twice
            # in subsequent operations to compute weight-aware bin edges with
            # quantiles or k-means.
            sample_weight = None

        n_features = X.shape[1]
        n_bins = self._validate_n_bins(n_features)

        bin_edges = np.zeros(n_features, dtype=object)

        quantile_method = self.quantile_method

        if (
            self.strategy == "quantile"
            and quantile_method not in ["inverted_cdf", "averaged_inverted_cdf"]
            and sample_weight is not None
        ):
            raise ValueError(
                "When fitting with strategy='quantile' and sample weights, "
                "quantile_method should either be set to 'averaged_inverted_cdf' or "
                f"'inverted_cdf', got quantile_method='{quantile_method}' instead."
            )

        if self.strategy != "quantile" and sample_weight is not None:
            # Prepare a mask to filter out zero-weight samples when extracting
            # the min and max values of each columns which are needed for the
            # "uniform" and "kmeans" strategies.
            nnz_weight_mask = sample_weight != 0
        else:
            # Otherwise, all samples are used. Use a slice to avoid creating a
            # new array.
            nnz_weight_mask = slice(None)

        for jj in range(n_features):
            column = X[:, jj]
            col_min = column[nnz_weight_mask].min()
            col_max = column[nnz_weight_mask].max()

            if col_min == col_max:
                warnings.warn(
                    "Feature %d is constant and will be replaced with 0." % jj
                )
                n_bins[jj] = 1
                bin_edges[jj] = np.array([-np.inf, np.inf])
                continue

            if self.strategy == "uniform":
                bin_edges[jj] = np.linspace(col_min, col_max, n_bins[jj] + 1)

            elif self.strategy == "quantile":
                percentile_levels = np.linspace(0, 100, n_bins[jj] + 1)

                # method="linear" is the implicit default for any numpy
                # version. So we keep it version independent in that case by
                # using an empty param dict.
                percentile_kwargs = {}
                if quantile_method != "linear" and sample_weight is None:
                    percentile_kwargs["method"] = quantile_method

                if sample_weight is None:
                    bin_edges[jj] = np.asarray(
                        np.percentile(column, percentile_levels, **percentile_kwargs),
                        dtype=np.float64,
                    )
                else:
                    average = (
                        True if quantile_method == "averaged_inverted_cdf" else False
                    )
                    bin_edges[jj] = _weighted_percentile(
                        column, sample_weight, percentile_levels, average=average
                    )
            elif self.strategy == "kmeans":
                from sklearn.cluster import KMeans  # fixes import loops

                # Deterministic initialization with uniform spacing
                uniform_edges = np.linspace(col_min, col_max, n_bins[jj] + 1)
                init = (uniform_edges[1:] + uniform_edges[:-1])[:, None] * 0.5

                # 1D k-means procedure
                km = KMeans(n_clusters=n_bins[jj], init=init, n_init=1)
                centers = km.fit(
                    column[:, None], sample_weight=sample_weight
                ).cluster_centers_[:, 0]
                # Must sort, centers may be unsorted even with sorted init
                centers.sort()
                bin_edges[jj] = (centers[1:] + centers[:-1]) * 0.5
                bin_edges[jj] = np.r_[col_min, bin_edges[jj], col_max]

            # Remove bins whose width are too small (i.e., <= 1e-8)
            if self.strategy in ("quantile", "kmeans"):
                mask = np.ediff1d(bin_edges[jj], to_begin=np.inf) > 1e-8
                bin_edges[jj] = bin_edges[jj][mask]
                if len(bin_edges[jj]) - 1 != n_bins[jj]:
                    warnings.warn(
                        "Bins whose width are too small (i.e., <= "
                        "1e-8) in feature %d are removed. Consider "
                        "decreasing the number of bins." % jj
                    )
                    n_bins[jj] = len(bin_edges[jj]) - 1

        self.bin_edges_ = bin_edges
        self.n_bins_ = n_bins

        if "onehot" in self.encode:
            self._encoder = OneHotEncoder(
                categories=[np.arange(i) for i in self.n_bins_],
                sparse_output=self.encode == "onehot",
                dtype=output_dtype,
            )
            # Fit the OneHotEncoder with toy datasets
            # so that it's ready for use after the KBinsDiscretizer is fitted
            self._encoder.fit(np.zeros((1, len(self.n_bins_))))

        return self

    def _validate_n_bins(self, n_features):
        # [FUNCTION DOC]
        # HELPER: Expands n_bins into an int array of length n_features.
        # Inputs: n_features - int. Outputs: int ndarray (n_features,); a scalar n_bins
        #   is broadcast.
        # Exceptions: ValueError if the array form is not 1D of length n_features, or
        #   contains values < 2 or non-integral values (detected by comparing the int-
        #   cast copy with the original).
        """Returns n_bins_, the number of bins per feature."""
        orig_bins = self.n_bins
        if isinstance(orig_bins, Integral):
            return np.full(n_features, orig_bins, dtype=int)

        n_bins = check_array(orig_bins, dtype=int, copy=True, ensure_2d=False)

        if n_bins.ndim > 1 or n_bins.shape[0] != n_features:
            raise ValueError("n_bins must be a scalar or array of shape (n_features,).")

        bad_nbins_value = (n_bins < 2) | (n_bins != orig_bins)

        violating_indices = np.where(bad_nbins_value)[0]
        if violating_indices.shape[0] > 0:
            indices = ", ".join(str(i) for i in violating_indices)
            raise ValueError(
                "{} received an invalid number "
                "of bins at indices {}. Number of bins "
                "must be at least 2, and must be an int.".format(
                    KBinsDiscretizer.__name__, indices
                )
            )
        return n_bins

    def transform(self, X):
        # [FUNCTION DOC]
        # Validates X (reset=False, copy, float dtype self.dtype or float64/float32).
        #   For each feature, replaces values with the bin index found by right-sided
        #   binary search into the interior edges (so out-of-range values map to the
        #   first/last bin). Returns the float ordinal array for 'ordinal'; otherwise
        #   one-hot encodes it, temporarily setting _encoder.dtype to the data dtype and
        #   restoring the original in a finally block.
        # Outputs: dense float array (ordinal or 'onehot-dense') or sparse matrix
        #   ('onehot').
        # KNOWN ISSUE: the encoder dtype is temporarily set to the input dtype, so the
        #   configured output dtype passed to the encoder at fit is effectively
        #   overridden by the validated data dtype.
        """
        Discretize the data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to be discretized.

        Returns
        -------
        Xt : {ndarray, sparse matrix}, dtype={np.float32, np.float64}
            Data in the binned space. Will be a sparse matrix if
            `self.encode='onehot'` and ndarray otherwise.
        """
        check_is_fitted(self)

        # check input and attribute dtypes
        dtype = (np.float64, np.float32) if self.dtype is None else self.dtype
        Xt = validate_data(self, X, copy=True, dtype=dtype, reset=False)

        bin_edges = self.bin_edges_
        for jj in range(Xt.shape[1]):
            Xt[:, jj] = np.searchsorted(bin_edges[jj][1:-1], Xt[:, jj], side="right")

        if self.encode == "ordinal":
            return Xt

        dtype_init = None
        if "onehot" in self.encode:
            dtype_init = self._encoder.dtype
            self._encoder.dtype = Xt.dtype
        try:
            Xt_enc = self._encoder.transform(Xt)
        finally:
            # revert the initial dtype to avoid modifying self.
            self._encoder.dtype = dtype_init
        return Xt_enc

    def inverse_transform(self, X):
        # [FUNCTION DOC]
        # For one-hot encodings, first decodes X with _encoder.inverse_transform.
        #   Converts to a float copy, checks the column count, then maps each bin index
        #   (cast to int64) to the midpoint of its bin edges.
        # Outputs: float ndarray (n_samples, n_features).
        # Exceptions: ValueError on wrong number of features; IndexError for out-of-
        #   range bin indices.
        """
        Transform discretized data back to original feature space.

        Note that this function does not regenerate the original data
        due to discretization rounding.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Transformed data in the binned space.

        Returns
        -------
        X_original : ndarray, dtype={np.float32, np.float64}
            Data in the original feature space.
        """

        check_is_fitted(self)

        if "onehot" in self.encode:
            X = self._encoder.inverse_transform(X)

        Xinv = check_array(X, copy=True, dtype=(np.float64, np.float32))
        n_features = self.n_bins_.shape[0]
        if Xinv.shape[1] != n_features:
            raise ValueError(
                "Incorrect number of features. Expecting {}, received {}.".format(
                    n_features, Xinv.shape[1]
                )
            )

        for jj in range(n_features):
            bin_edges = self.bin_edges_[jj]
            bin_centers = (bin_edges[1:] + bin_edges[:-1]) * 0.5
            Xinv[:, jj] = bin_centers[(Xinv[:, jj]).astype(np.int64)]

        return Xinv

    def get_feature_names_out(self, input_features=None):
        # [FUNCTION DOC]
        # Requires n_features_in_. Resolves input names via _check_feature_names_in;
        #   returns the encoder's one-hot names when _encoder exists, otherwise the
        #   input names (ordinal encoding).
        """Get output feature names.

        Parameters
        ----------
        input_features : array-like of str or None, default=None
            Input features.

            - If `input_features` is `None`, then `feature_names_in_` is
              used as feature names in. If `feature_names_in_` is not defined,
              then the following input feature names are generated:
              `["x0", "x1", ..., "x(n_features_in_ - 1)"]`.
            - If `input_features` is an array-like, then `input_features` must
              match `feature_names_in_` if `feature_names_in_` is defined.

        Returns
        -------
        feature_names_out : ndarray of str objects
            Transformed feature names.
        """
        check_is_fitted(self, "n_features_in_")
        input_features = _check_feature_names_in(self, input_features)
        if hasattr(self, "_encoder"):
            return self._encoder.get_feature_names_out(input_features)

        # ordinal encoding
        return input_features
