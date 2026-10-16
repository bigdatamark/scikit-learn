# [MODULE DOC]
# Test suite for PolynomialFeatures and SplineTransformer
#   (sklearn.preprocessing._polynomial): parameter validation, knot placement, B-spline
#   properties (partition of unity, periodicity, smoothness, extrapolation modes,
#   missing-value handling, sparse output), polynomial expansion for dense/CSR/CSC input
#   including index-dtype promotion and overflow edge cases of the Cython CSR expansion,
#   feature names, and array API compliance.
import re
import sys

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy import sparse
from scipy.interpolate import BSpline

from sklearn._config import config_context
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    KBinsDiscretizer,
    PolynomialFeatures,
    SplineTransformer,
)
from sklearn.preprocessing._csr_polynomial_expansion import (
    _get_sizeof_LARGEST_INT_t,
)
from sklearn.utils._array_api import (
    _is_numpy_namespace,
    get_namespace,
    move_to,
    yield_namespace_device_dtype_combinations,
)
from sklearn.utils._array_api import (
    device as array_api_device,
)
from sklearn.utils._mask import _get_mask
from sklearn.utils._testing import (
    _array_api_for_tests,
    assert_allclose_dense_sparse,
    assert_array_almost_equal,
)
from sklearn.utils.fixes import (
    CSC_CONTAINERS,
    CSR_CONTAINERS,
    _sparse_random_array,
)


@pytest.mark.parametrize("est", (PolynomialFeatures, SplineTransformer))
def test_polynomial_and_spline_array_order(est):
    # [FUNCTION DOC]
    # Verifies the output memory layout: default and order="C" give C-contiguous output,
    #   order="F" gives Fortran-ordered output, for both PolynomialFeatures and
    #   SplineTransformer.
    """Test that output array has the given order."""
    X = np.arange(10).reshape(5, 2)

    def is_c_contiguous(a):
        # [FUNCTION DOC]
        # HELPER: Returns True when the array is C-contiguous, tested as its transpose
        #   being Fortran-ordered.
        return np.isfortran(a.T)

    assert is_c_contiguous(est().fit_transform(X))
    assert is_c_contiguous(est(order="C").fit_transform(X))
    assert np.isfortran(est(order="F").fit_transform(X))


@pytest.mark.parametrize(
    "params, err_msg",
    [
        ({"knots": [[1]]}, r"Number of knots, knots.shape\[0\], must be >= 2."),
        ({"knots": [[1, 1], [2, 2]]}, r"knots.shape\[1\] == n_features is violated"),
        ({"knots": [[1], [0]]}, "knots must be sorted without duplicates."),
    ],
)
def test_spline_transformer_input_validation(params, err_msg):
    # [FUNCTION DOC]
    # Verifies SplineTransformer.fit raises ValueError with the given message for too
    #   few knots, knots of the wrong width, and unsorted knots.
    """Test that we raise errors for invalid input in SplineTransformer."""
    X = [[1], [2]]

    with pytest.raises(ValueError, match=err_msg):
        SplineTransformer(**params).fit(X)


@pytest.mark.parametrize("extrapolation", ["continue", "periodic"])
def test_spline_transformer_integer_knots(extrapolation):
    # [FUNCTION DOC]
    # Verifies integer-valued explicit knots are accepted for "continue" and "periodic"
    #   extrapolation (smoke test: fit_transform must not raise).
    """Test that SplineTransformer accepts integer value knot positions."""
    X = np.arange(20).reshape(10, 2)
    knots = [[0, 1], [1, 2], [5, 5], [11, 10], [12, 11]]
    _ = SplineTransformer(
        degree=3, knots=knots, extrapolation=extrapolation
    ).fit_transform(X)


def test_spline_transformer_feature_names():
    # [FUNCTION DOC]
    # Verifies get_feature_names_out produces "<feature>_sp_<i>" names, n_knots + degree
    #   - 1 per feature with include_bias, one fewer without, using default and custom
    #   input names.
    """Test that SplineTransformer generates correct feature names."""
    X = np.arange(20).reshape(10, 2)
    splt = SplineTransformer(n_knots=3, degree=3, include_bias=True).fit(X)
    feature_names = splt.get_feature_names_out()
    assert_array_equal(
        feature_names,
        [
            "x0_sp_0",
            "x0_sp_1",
            "x0_sp_2",
            "x0_sp_3",
            "x0_sp_4",
            "x1_sp_0",
            "x1_sp_1",
            "x1_sp_2",
            "x1_sp_3",
            "x1_sp_4",
        ],
    )

    splt = SplineTransformer(n_knots=3, degree=3, include_bias=False).fit(X)
    feature_names = splt.get_feature_names_out(["a", "b"])
    assert_array_equal(
        feature_names,
        [
            "a_sp_0",
            "a_sp_1",
            "a_sp_2",
            "a_sp_3",
            "b_sp_0",
            "b_sp_1",
            "b_sp_2",
            "b_sp_3",
        ],
    )


@pytest.mark.parametrize(
    "extrapolation",
    ["constant", "linear", "continue", "periodic"],
)
@pytest.mark.parametrize("degree", [2, 3])
def test_split_transform_feature_names_extrapolation_degree(extrapolation, degree):
    # [FUNCTION DOC]
    # Non-regression for gh-25292: number of feature names equals n_features_out_ and
    #   the transform width, for every extrapolation mode and degree 2/3.
    """Test feature names are correct for different extrapolations and degree.

    Non-regression test for gh-25292.
    """
    X = np.arange(20).reshape(10, 2)
    splt = SplineTransformer(degree=degree, extrapolation=extrapolation).fit(X)
    feature_names = splt.get_feature_names_out(["a", "b"])
    assert len(feature_names) == splt.n_features_out_

    X_trans = splt.transform(X)
    assert X_trans.shape[1] == len(feature_names)


@pytest.mark.parametrize("degree", range(1, 5))
@pytest.mark.parametrize("n_knots", range(3, 5))
@pytest.mark.parametrize("knots", ["uniform", "quantile"])
@pytest.mark.parametrize("extrapolation", ["constant", "periodic"])
def test_spline_transformer_unity_decomposition(degree, n_knots, knots, extrapolation):
    # [FUNCTION DOC]
    # Verifies spline basis values sum to 1 per row on training and in-range test points
    #   (partition of unity) for uniform/quantile knots and constant/periodic
    #   extrapolation; periodic runs add degree to n_knots.
    """Test that B-splines are indeed a decomposition of unity.

    Splines basis functions must sum up to 1 per row, if we stay in between boundaries.
    """
    X = np.linspace(0, 1, 100)[:, None]
    # make the boundaries 0 and 1 part of X_train, for sure.
    X_train = np.r_[[[0]], X[::2, :], [[1]]]
    X_test = X[1::2, :]

    if extrapolation == "periodic":
        n_knots = n_knots + degree  # periodic splines require degree < n_knots

    splt = SplineTransformer(
        n_knots=n_knots,
        degree=degree,
        knots=knots,
        include_bias=True,
        extrapolation=extrapolation,
    )
    splt.fit(X_train)
    for X in [X_train, X_test]:
        assert_allclose(np.sum(splt.transform(X), axis=1), 1)


@pytest.mark.parametrize(["bias", "intercept"], [(True, False), (False, True)])
def test_spline_transformer_linear_regression(bias, intercept):
    # [FUNCTION DOC]
    # Verifies a SplineTransformer + LinearRegression pipeline fits sin(x)+2 within rtol
    #   1e-3, with the bias provided by either the splines or the regression intercept.
    """Test that B-splines fit a sinusodial curve pretty well."""
    X = np.linspace(0, 10, 100)[:, None]
    y = np.sin(X[:, 0]) + 2  # +2 to avoid the value 0 in assert_allclose
    pipe = Pipeline(
        steps=[
            (
                "spline",
                SplineTransformer(
                    n_knots=15,
                    degree=3,
                    include_bias=bias,
                    extrapolation="constant",
                ),
            ),
            ("ols", LinearRegression(fit_intercept=intercept)),
        ]
    )
    pipe.fit(X, y)
    assert_allclose(pipe.predict(X), y, rtol=1e-3)


@pytest.mark.parametrize(
    ["knots", "n_knots", "sample_weight", "expected_knots"],
    [
        ("uniform", 3, None, np.array([[0, 2], [3, 8], [6, 14]])),
        (
            "uniform",
            3,
            np.array([0, 0, 1, 1, 0, 3, 1]),
            np.array([[2, 2], [4, 8], [6, 14]]),
        ),
        ("uniform", 4, None, np.array([[0, 2], [2, 6], [4, 10], [6, 14]])),
        ("quantile", 3, None, np.array([[0, 2], [3, 3], [6, 14]])),
        (
            "quantile",
            3,
            np.array([0, 0, 1, 1, 0, 3, 1]),
            np.array([[2, 2], [5, 8], [6, 14]]),
        ),
    ],
)
def test_spline_transformer_get_base_knot_positions(
    knots, n_knots, sample_weight, expected_knots
):
    # [FUNCTION DOC]
    # Verifies the private static _get_base_knot_positions returns the expected uniform
    #   or (weighted) quantile knot positions, with and without sample_weight (zero
    #   weights exclude samples from the range).
    """Check the behaviour to find knot positions with and without sample_weight."""
    X = np.array([[0, 2], [0, 2], [2, 2], [3, 3], [4, 6], [5, 8], [6, 14]])
    base_knots = SplineTransformer._get_base_knot_positions(
        X=X, knots=knots, n_knots=n_knots, sample_weight=sample_weight
    )
    assert_allclose(base_knots, expected_knots)


@pytest.mark.parametrize(["bias", "intercept"], [(True, False), (False, True)])
def test_spline_transformer_periodic_linear_regression(bias, intercept):
    # [FUNCTION DOC]
    # Verifies a periodic spline regression fits a periodic function on [0, 1] and
    #   extrapolates periodically on [-1, 2]: predictions match the function and repeat
    #   with period 1.
    """Test that B-splines fit a periodic curve pretty well."""

    # "+ 3" to avoid the value 0 in assert_allclose
    def f(x):
        # [FUNCTION DOC]
        # HELPER: Periodic target sin(2 pi x) - sin(8 pi x) + 3 (offset avoids zeros for
        #   relative tolerance).
        return np.sin(2 * np.pi * x) - np.sin(8 * np.pi * x) + 3

    X = np.linspace(0, 1, 101)[:, None]
    pipe = Pipeline(
        steps=[
            (
                "spline",
                SplineTransformer(
                    n_knots=20,
                    degree=3,
                    include_bias=bias,
                    extrapolation="periodic",
                ),
            ),
            ("ols", LinearRegression(fit_intercept=intercept)),
        ]
    )
    pipe.fit(X, f(X[:, 0]))

    # Generate larger array to check periodic extrapolation
    X_ = np.linspace(-1, 2, 301)[:, None]
    predictions = pipe.predict(X_)
    assert_allclose(predictions, f(X_[:, 0]), atol=0.01, rtol=0.01)
    assert_allclose(predictions[0:100], predictions[100:200], rtol=1e-3)


def test_spline_transformer_periodic_spline_backport():
    # [FUNCTION DOC]
    # Verifies the periodic extrapolation of SplineTransformer with explicit knots
    #   equals scipy BSpline with extrapolate "periodic" on an equivalent extended knot
    #   vector and identity-like coefficients.
    """Test that the backport of extrapolate="periodic" works correctly"""
    X = np.linspace(-2, 3.5, 10)[:, None]
    degree = 2

    # Use periodic extrapolation backport in SplineTransformer
    transformer = SplineTransformer(
        degree=degree, extrapolation="periodic", knots=[[-1.0], [0.0], [1.0]]
    )
    Xt = transformer.fit_transform(X)

    # Use periodic extrapolation in BSpline
    coef = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    spl = BSpline(np.arange(-3, 4), coef, degree, "periodic")
    Xspl = spl(X[:, 0])
    assert_allclose(Xt, Xspl)


def test_spline_transformer_periodic_splines_periodicity():
    # [FUNCTION DOC]
    # Verifies that two periodic knot sets that are shifts of each other (same period)
    #   produce the same basis up to a column permutation.
    """Test if shifted knots result in the same transformation up to permutation."""
    X = np.linspace(0, 10, 101)[:, None]

    transformer_1 = SplineTransformer(
        degree=3,
        extrapolation="periodic",
        knots=[[0.0], [1.0], [3.0], [4.0], [5.0], [8.0]],
    )

    transformer_2 = SplineTransformer(
        degree=3,
        extrapolation="periodic",
        knots=[[1.0], [3.0], [4.0], [5.0], [8.0], [9.0]],
    )

    Xt_1 = transformer_1.fit_transform(X)
    Xt_2 = transformer_2.fit_transform(X)

    assert_allclose(Xt_1, Xt_2[:, [4, 0, 1, 2, 3]])


@pytest.mark.parametrize("degree", [3, 5])
def test_spline_transformer_periodic_splines_smoothness(degree):
    # [FUNCTION DOC]
    # Verifies periodic splines are degree-1 times continuously differentiable across
    #   knots by bounding successive finite differences of the transform by 10 times the
    #   grid step, and that the next derivative order shows spikes (> 1) at the knots.
    """Test that spline transformation is smooth at first / last knot."""
    X = np.linspace(-2, 10, 10_000)[:, None]

    transformer = SplineTransformer(
        degree=degree,
        extrapolation="periodic",
        knots=[[0.0], [1.0], [3.0], [4.0], [5.0], [8.0]],
    )
    Xt = transformer.fit_transform(X)

    delta = (X.max() - X.min()) / len(X)
    tol = 10 * delta

    dXt = Xt
    # We expect splines of degree `degree` to be (`degree`-1) times
    # continuously differentiable. I.e. for d = 0, ..., `degree` - 1 the d-th
    # derivative should be continuous. This is the case if the (d+1)-th
    # numerical derivative is reasonably small (smaller than `tol` in absolute
    # value). We thus compute d-th numeric derivatives for d = 1, ..., `degree`
    # and compare them to `tol`.
    #
    # Note that the 0-th derivative is the function itself, such that we are
    # also checking its continuity.
    for d in range(1, degree + 1):
        # Check continuity of the (d-1)-th derivative
        diff = np.diff(dXt, axis=0)
        assert np.abs(diff).max() < tol
        # Compute d-th numeric derivative
        dXt = diff / delta

    # As degree `degree` splines are not `degree` times continuously
    # differentiable at the knots, the `degree + 1`-th numeric derivative
    # should have spikes at the knots.
    diff = np.diff(dXt, axis=0)
    assert np.abs(diff).max() > 1


@pytest.mark.parametrize(["bias", "intercept"], [(True, False), (False, True)])
@pytest.mark.parametrize("degree", [1, 2, 3, 4, 5])
def test_spline_transformer_extrapolation(bias, intercept, degree):
    # [FUNCTION DOC]
    # Verifies extrapolation modes on a linear target: "constant" clamps predictions to
    #   the boundary values, "linear" continues the line, and "error" raises ValueError
    #   for out-of-range inputs.
    """Test that B-spline extrapolation works correctly."""
    # we use a straight line for that
    X = np.linspace(-1, 1, 100)[:, None]
    y = X.squeeze()

    # 'constant'
    pipe = Pipeline(
        [
            [
                "spline",
                SplineTransformer(
                    n_knots=4,
                    degree=degree,
                    include_bias=bias,
                    extrapolation="constant",
                ),
            ],
            ["ols", LinearRegression(fit_intercept=intercept)],
        ]
    )
    pipe.fit(X, y)
    assert_allclose(pipe.predict([[-10], [5]]), [-1, 1])

    # 'linear'
    pipe = Pipeline(
        [
            [
                "spline",
                SplineTransformer(
                    n_knots=4,
                    degree=degree,
                    include_bias=bias,
                    extrapolation="linear",
                ),
            ],
            ["ols", LinearRegression(fit_intercept=intercept)],
        ]
    )
    pipe.fit(X, y)
    assert_allclose(pipe.predict([[-10], [5]]), [-10, 5])

    # 'error'
    splt = SplineTransformer(
        n_knots=4, degree=degree, include_bias=bias, extrapolation="error"
    )
    splt.fit(X)
    msg = "`X` contains values beyond the limits of the knots"
    with pytest.raises(ValueError, match=msg):
        splt.transform([[-10]])
    with pytest.raises(ValueError, match=msg):
        splt.transform([[5]])


def test_spline_transformer_kbindiscretizer(global_random_seed):
    # [FUNCTION DOC]
    # Verifies a degree-0 spline with quantile knots equals the one-hot dense
    #   KBinsDiscretizer output with quantile strategy (averaged_inverted_cdf), up to
    #   rtol 1e-13, on seeded normal data.
    """Test that a B-spline of degree=0 is equivalent to KBinsDiscretizer."""
    rng = np.random.RandomState(global_random_seed)
    X = rng.randn(200).reshape(200, 1)
    n_bins = 5
    n_knots = n_bins + 1

    splt = SplineTransformer(
        n_knots=n_knots, degree=0, knots="quantile", include_bias=True
    )
    splines = splt.fit_transform(X)

    kbd = KBinsDiscretizer(
        n_bins=n_bins,
        encode="onehot-dense",
        strategy="quantile",
        quantile_method="averaged_inverted_cdf",
    )
    kbins = kbd.fit_transform(X)

    # Though they should be exactly equal, we test approximately with high
    # accuracy.
    assert_allclose(splines, kbins, rtol=1e-13)


@pytest.mark.parametrize("degree", range(1, 3))
@pytest.mark.parametrize("knots", ["uniform", "quantile"])
@pytest.mark.parametrize(
    "extrapolation", ["error", "constant", "linear", "continue", "periodic"]
)
@pytest.mark.parametrize("include_bias", [False, True])
def test_spline_transformer_sparse_output(
    degree, knots, extrapolation, include_bias, global_random_seed
):
    # [FUNCTION DOC]
    # Verifies sparse_output=True gives a CSR matrix equal to the dense output, in range
    #   and in the extrapolation regime; with extrapolation "error", dense raises the
    #   knots-limit message and sparse raises "Out of bounds".
    # Inputs: degree, knots, extrapolation, include_bias - SplineTransformer params;
    #   global_random_seed - fixture seeding the data.
    rng = np.random.RandomState(global_random_seed)
    X = rng.randn(200).reshape(40, 5)

    splt_dense = SplineTransformer(
        degree=degree,
        knots=knots,
        extrapolation=extrapolation,
        include_bias=include_bias,
        sparse_output=False,
    )
    splt_sparse = SplineTransformer(
        degree=degree,
        knots=knots,
        extrapolation=extrapolation,
        include_bias=include_bias,
        sparse_output=True,
    )

    splt_dense.fit(X)
    splt_sparse.fit(X)

    X_trans_sparse = splt_sparse.transform(X)
    X_trans_dense = splt_dense.transform(X)
    assert sparse.issparse(X_trans_sparse) and X_trans_sparse.format == "csr"
    assert_allclose(X_trans_dense, X_trans_sparse.toarray())

    # extrapolation regime
    X_min = np.amin(X, axis=0)
    X_max = np.amax(X, axis=0)
    X_extra = np.r_[
        np.linspace(X_min - 5, X_min, 10), np.linspace(X_max, X_max + 5, 10)
    ]
    if extrapolation == "error":
        msg = "`X` contains values beyond the limits of the knots"
        with pytest.raises(ValueError, match=msg):
            splt_dense.transform(X_extra)
        msg = "Out of bounds"
        with pytest.raises(ValueError, match=msg):
            splt_sparse.transform(X_extra)
    else:
        assert_allclose(
            splt_dense.transform(X_extra), splt_sparse.transform(X_extra).toarray()
        )


@pytest.mark.parametrize("n_knots", [5, 10])
@pytest.mark.parametrize("include_bias", [True, False])
@pytest.mark.parametrize("degree", [3, 4])
@pytest.mark.parametrize(
    "extrapolation", ["error", "constant", "linear", "continue", "periodic"]
)
@pytest.mark.parametrize("sparse_output", [False, True])
def test_spline_transformer_n_features_out(
    n_knots, include_bias, degree, extrapolation, sparse_output
):
    # [FUNCTION DOC]
    # Verifies transform output width equals n_features_out_ across n_knots,
    #   include_bias, degree, extrapolation and sparse_output combinations.
    """Test that transform results in n_features_out_ features."""
    splt = SplineTransformer(
        n_knots=n_knots,
        degree=degree,
        include_bias=include_bias,
        extrapolation=extrapolation,
        sparse_output=sparse_output,
    )
    X = np.linspace(0, 1, 10)[:, None]
    splt.fit(X)

    assert splt.transform(X).shape[1] == splt.n_features_out_


@pytest.mark.parametrize(
    "extrapolation", ["error", "constant", "linear", "continue", "periodic"]
)
@pytest.mark.parametrize("sparse_output", [False, True])
def test_spline_transformer_handles_missing_values(extrapolation, sparse_output):
    # [FUNCTION DOC]
    # Verifies handle_missing: "error" raises on NaN; "zeros" gives consistent results
    #   for fit_transform vs fit+transform, with unit sample_weight and on row subsets,
    #   encodes NaN entries as all-zero spline columns, keeps values in [0, 1], and
    #   leaves non-missing outputs equal to those obtained without NaN.
    """Test that SplineTransformer handles missing values correctly.
    We only test for knots="uniform", since for "quantile" the metrics are calculated
    differently with nans present and a different result is thus expected.
    """
    X = np.array([[1, 1], [2, 2], [3, 3], [4, 5], [4, 4]], dtype=np.float64)
    X_nan = X.copy()
    X_nan[3, 0] = np.nan

    # Check correct error message for handle_missing="error":
    msg = "Input X contains NaN values and `SplineTransformer` is configured to error"
    with pytest.raises(ValueError, match=re.escape(msg)):
        spline = SplineTransformer(
            degree=2,
            n_knots=3,
            handle_missing="error",
            extrapolation=extrapolation,
        )
        spline.fit_transform(X_nan)

    # Test correct results for handle_missing="zeros"
    spline = SplineTransformer(
        degree=2,
        n_knots=3,
        handle_missing="zeros",
        extrapolation=extrapolation,
        sparse_output=sparse_output,
    )

    # Check `fit_transform` does the same as `fit` and then `transform`:
    X_nan_transform = spline.fit_transform(X_nan)
    X_nan_fit_then_transform = spline.fit(X_nan).transform(X_nan)
    assert_allclose_dense_sparse(X_nan_transform, X_nan_fit_then_transform)

    # Check that missing values are handled the same when sample_weight is passed:
    X_nan_transform_with_sample_weight = spline.fit_transform(
        X_nan, sample_weight=[1, 1, 1, 1, 1]
    )
    assert_allclose_dense_sparse(X_nan_transform, X_nan_transform_with_sample_weight)

    # Check that `transform` works as expected when the passed data has a different
    # shape than the training data passed to `fit`:
    X_nan_transform_same_shape = spline.fit_transform(X_nan)[::2]
    X_nan_transform_different_shapes = spline.fit(X_nan).transform(X_nan[::2])
    assert_allclose_dense_sparse(
        X_nan_transform_same_shape, X_nan_transform_different_shapes
    )

    # Check that the masked nan-values are 0s:
    nan_mask = _get_mask(X_nan, np.nan)
    encoded_nan_mask = np.repeat(nan_mask, spline.bsplines_[0].c.shape[1], axis=1)
    assert (X_nan_transform[encoded_nan_mask] == 0).all()

    # Check the nan handling doesn't change that B-Splines basis functions are always in
    # the interval [0, 1]:
    X_nan_transform = spline.fit_transform(X_nan)
    if sparse.issparse(X_nan_transform):
        X_nan_transform = X_nan_transform.toarray()
    assert (X_nan_transform >= 0).all()
    assert (X_nan_transform <= 1).all()

    # Check that additional nan values don't change the calculation of the other
    # splines. Note: this assertion only holds as long as no np.nan value constructs the
    # min or max value of the data space (in this case, SplineTransformer's stats would
    # be calculated based on the other values and thus differ from another
    # SplineTransformer fit on the whole range).
    X_transform = spline.fit_transform(X)
    X_nan_transform = spline.fit_transform(X_nan)
    assert_allclose_dense_sparse(
        X_transform[~encoded_nan_mask], X_nan_transform[~encoded_nan_mask]
    )


@pytest.mark.parametrize(
    "extrapolation", ["error", "constant", "linear", "continue", "periodic"]
)
@pytest.mark.parametrize("sparse_output", [False, True])
def test_spline_transformer_handles_all_nans(extrapolation, sparse_output):
    # [FUNCTION DOC]
    # Verifies with handle_missing="zeros" that a fully-missing feature column (and
    #   partially missing ones) are encoded as zeros after fitting on the same data.
    # KNOWN ISSUE: the array X is built but never used.
    """Test that SplineTransformer encodes missing values to zeros even for
    all-nan-features."""

    X = np.array([[1, 1], [2, 2], [3, 3], [4, 5], [4, 4]])
    X_nan_full_column = np.array([[np.nan, np.nan], [np.nan, 1]])

    spline = SplineTransformer(
        degree=2,
        n_knots=3,
        handle_missing="zeros",
        extrapolation=extrapolation,
        sparse_output=sparse_output,
    )
    spline.fit(X_nan_full_column)

    all_missing_column_encoded = spline.transform(X_nan_full_column)
    nan_mask = _get_mask(X_nan_full_column, np.nan)
    encoded_nan_mask = np.repeat(nan_mask, spline.bsplines_[0].c.shape[1], axis=1)
    assert (all_missing_column_encoded[encoded_nan_mask] == 0).all()


@pytest.mark.parametrize(
    "params, err_msg",
    [
        ({"degree": (-1, 2)}, r"degree=\(min_degree, max_degree\) must"),
        ({"degree": (0, 1.5)}, r"degree=\(min_degree, max_degree\) must"),
        ({"degree": (3, 2)}, r"degree=\(min_degree, max_degree\) must"),
        ({"degree": (1, 2, 3)}, r"int or tuple \(min_degree, max_degree\)"),
    ],
)
def test_polynomial_features_input_validation(params, err_msg):
    # [FUNCTION DOC]
    # Verifies PolynomialFeatures.fit raises ValueError for invalid degree tuples
    #   (negative, non-integer, min greater than max, wrong length).
    """Test that we raise errors for invalid input in PolynomialFeatures."""
    X = [[1], [2]]

    with pytest.raises(ValueError, match=err_msg):
        PolynomialFeatures(**params).fit(X)


@pytest.fixture()
def single_feature_degree3():
    # [FUNCTION DOC]
    # HELPER: Fixture returning X (6x1 column 0..5) and P, the matrix of its powers 0 to
    #   3 as columns.
    X = np.arange(6)[:, np.newaxis]
    P = np.hstack([np.ones_like(X), X, X**2, X**3])
    return X, P


@pytest.mark.parametrize(
    "degree, include_bias, interaction_only, indices",
    [
        (3, True, False, slice(None, None)),
        (3, False, False, slice(1, None)),
        (3, True, True, [0, 1]),
        (3, False, True, [1]),
        ((2, 3), True, False, [0, 2, 3]),
        ((2, 3), False, False, [2, 3]),
        ((2, 3), True, True, [0]),
        ((2, 3), False, True, []),
    ],
)
@pytest.mark.parametrize("X_container", [None] + CSR_CONTAINERS + CSC_CONTAINERS)
def test_polynomial_features_one_feature(
    single_feature_degree3,
    degree,
    include_bias,
    interaction_only,
    indices,
    X_container,
):
    # [FUNCTION DOC]
    # Verifies PolynomialFeatures on one feature selects the expected power columns of P
    #   for each degree/include_bias/interaction_only combination, for dense, CSR and
    #   CSC input, and that powers_ has shape (n_output_features_, n_features_in_).
    """Test PolynomialFeatures on single feature up to degree 3."""
    X, P = single_feature_degree3
    if X_container is not None:
        X = X_container(X)
    tf = PolynomialFeatures(
        degree=degree, include_bias=include_bias, interaction_only=interaction_only
    ).fit(X)
    out = tf.transform(X)
    if X_container is not None:
        out = out.toarray()
    assert_allclose(out, P[:, indices])
    if tf.n_output_features_ > 0:
        assert tf.powers_.shape == (tf.n_output_features_, tf.n_features_in_)


@pytest.fixture()
def two_features_degree3():
    # [FUNCTION DOC]
    # HELPER: Fixture returning X (3x2 array 0..5) and P, the 10 monomials of degree up
    #   to 3 of its two columns in PolynomialFeatures output order (bias, degree 1,
    #   degree 2, degree 3).
    X = np.arange(6).reshape((3, 2))
    x1 = X[:, :1]
    x2 = X[:, 1:]
    P = np.hstack(
        [
            x1**0 * x2**0,  # 0
            x1**1 * x2**0,  # 1
            x1**0 * x2**1,  # 2
            x1**2 * x2**0,  # 3
            x1**1 * x2**1,  # 4
            x1**0 * x2**2,  # 5
            x1**3 * x2**0,  # 6
            x1**2 * x2**1,  # 7
            x1**1 * x2**2,  # 8
            x1**0 * x2**3,  # 9
        ]
    )
    return X, P


@pytest.mark.parametrize(
    "degree, include_bias, interaction_only, indices",
    [
        (2, True, False, slice(0, 6)),
        (2, False, False, slice(1, 6)),
        (2, True, True, [0, 1, 2, 4]),
        (2, False, True, [1, 2, 4]),
        ((2, 2), True, False, [0, 3, 4, 5]),
        ((2, 2), False, False, [3, 4, 5]),
        ((2, 2), True, True, [0, 4]),
        ((2, 2), False, True, [4]),
        (3, True, False, slice(None, None)),
        (3, False, False, slice(1, None)),
        (3, True, True, [0, 1, 2, 4]),
        (3, False, True, [1, 2, 4]),
        ((2, 3), True, False, [0, 3, 4, 5, 6, 7, 8, 9]),
        ((2, 3), False, False, slice(3, None)),
        ((2, 3), True, True, [0, 4]),
        ((2, 3), False, True, [4]),
        ((3, 3), True, False, [0, 6, 7, 8, 9]),
        ((3, 3), False, False, [6, 7, 8, 9]),
        ((3, 3), True, True, [0]),
        ((3, 3), False, True, []),  # would need 3 input features
    ],
)
@pytest.mark.parametrize("X_container", [None] + CSR_CONTAINERS + CSC_CONTAINERS)
def test_polynomial_features_two_features(
    two_features_degree3,
    degree,
    include_bias,
    interaction_only,
    indices,
    X_container,
):
    # [FUNCTION DOC]
    # Verifies PolynomialFeatures on two features selects the expected monomial columns
    #   of P for each degree/include_bias/interaction_only combination, for dense, CSR
    #   and CSC input, plus the powers_ shape.
    """Test PolynomialFeatures on 2 features up to degree 3."""
    X, P = two_features_degree3
    if X_container is not None:
        X = X_container(X)
    tf = PolynomialFeatures(
        degree=degree, include_bias=include_bias, interaction_only=interaction_only
    ).fit(X)
    out = tf.transform(X)
    if X_container is not None:
        out = out.toarray()
    assert_allclose(out, P[:, indices])
    if tf.n_output_features_ > 0:
        assert tf.powers_.shape == (tf.n_output_features_, tf.n_features_in_)


def test_polynomial_feature_names():
    # [FUNCTION DOC]
    # Verifies get_feature_names_out naming ("1", "x0", "x0^2", "x0 x1", ...) for
    #   several degrees, degree ranges and interaction_only, that the count matches the
    #   transform width, and that unicode input names pass through.
    X = np.arange(30).reshape(10, 3)
    poly = PolynomialFeatures(degree=2, include_bias=True).fit(X)
    feature_names = poly.get_feature_names_out()
    assert_array_equal(
        ["1", "x0", "x1", "x2", "x0^2", "x0 x1", "x0 x2", "x1^2", "x1 x2", "x2^2"],
        feature_names,
    )
    assert len(feature_names) == poly.transform(X).shape[1]

    poly = PolynomialFeatures(degree=3, include_bias=False).fit(X)
    feature_names = poly.get_feature_names_out(["a", "b", "c"])
    assert_array_equal(
        [
            "a",
            "b",
            "c",
            "a^2",
            "a b",
            "a c",
            "b^2",
            "b c",
            "c^2",
            "a^3",
            "a^2 b",
            "a^2 c",
            "a b^2",
            "a b c",
            "a c^2",
            "b^3",
            "b^2 c",
            "b c^2",
            "c^3",
        ],
        feature_names,
    )
    assert len(feature_names) == poly.transform(X).shape[1]

    poly = PolynomialFeatures(degree=(2, 3), include_bias=False).fit(X)
    feature_names = poly.get_feature_names_out(["a", "b", "c"])
    assert_array_equal(
        [
            "a^2",
            "a b",
            "a c",
            "b^2",
            "b c",
            "c^2",
            "a^3",
            "a^2 b",
            "a^2 c",
            "a b^2",
            "a b c",
            "a c^2",
            "b^3",
            "b^2 c",
            "b c^2",
            "c^3",
        ],
        feature_names,
    )
    assert len(feature_names) == poly.transform(X).shape[1]

    poly = PolynomialFeatures(
        degree=(3, 3), include_bias=True, interaction_only=True
    ).fit(X)
    feature_names = poly.get_feature_names_out(["a", "b", "c"])
    assert_array_equal(["1", "a b c"], feature_names)
    assert len(feature_names) == poly.transform(X).shape[1]

    # test some unicode
    poly = PolynomialFeatures(degree=1, include_bias=True).fit(X)
    feature_names = poly.get_feature_names_out(["\u0001F40D", "\u262e", "\u05d0"])
    assert_array_equal(["1", "\u0001F40D", "\u262e", "\u05d0"], feature_names)


@pytest.mark.parametrize(
    ["deg", "include_bias", "interaction_only", "dtype"],
    [
        (1, True, False, int),
        (2, True, False, int),
        (2, True, False, np.float32),
        (2, True, False, np.float64),
        (3, False, False, np.float64),
        (3, False, True, np.float64),
        (4, False, False, np.float64),
        (4, False, True, np.float64),
    ],
)
@pytest.mark.parametrize("csc_container", CSC_CONTAINERS)
def test_polynomial_features_csc_X(
    deg, include_bias, interaction_only, dtype, csc_container, global_random_seed
):
    # [FUNCTION DOC]
    # Verifies CSC input gives CSC output with the same dtype and values as dense input,
    #   for several degree/bias/interaction/dtype settings on seeded random binary data.
    rng = np.random.RandomState(global_random_seed)
    X = rng.randint(0, 2, (100, 2))
    X_csc = csc_container(X)

    est = PolynomialFeatures(
        deg, include_bias=include_bias, interaction_only=interaction_only
    )
    Xt_csc = est.fit_transform(X_csc.astype(dtype))
    Xt_dense = est.fit_transform(X.astype(dtype))

    assert sparse.issparse(Xt_csc) and Xt_csc.format == "csc"
    assert Xt_csc.dtype == Xt_dense.dtype
    assert_array_almost_equal(Xt_csc.toarray(), Xt_dense)


@pytest.mark.parametrize(
    ["deg", "include_bias", "interaction_only", "dtype"],
    [
        (1, True, False, int),
        (2, True, False, int),
        (2, True, False, np.float32),
        (2, True, False, np.float64),
        (3, False, False, np.float64),
        (3, False, True, np.float64),
    ],
)
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_polynomial_features_csr_X(
    deg, include_bias, interaction_only, dtype, csr_container, global_random_seed
):
    # [FUNCTION DOC]
    # Verifies CSR input gives CSR output with the same dtype and values as dense input,
    #   for several degree/bias/interaction/dtype settings on seeded random binary data.
    rng = np.random.RandomState(global_random_seed)
    X = rng.randint(0, 2, (100, 2))
    X_csr = csr_container(X)

    est = PolynomialFeatures(
        deg, include_bias=include_bias, interaction_only=interaction_only
    )
    Xt_csr = est.fit_transform(X_csr.astype(dtype))
    Xt_dense = est.fit_transform(X.astype(dtype, copy=False))

    assert sparse.issparse(Xt_csr) and Xt_csr.format == "csr"
    assert Xt_csr.dtype == Xt_dense.dtype
    assert_array_almost_equal(Xt_csr.toarray(), Xt_dense)


@pytest.mark.parametrize("n_features", [1, 4, 5])
@pytest.mark.parametrize(
    "min_degree, max_degree", [(0, 1), (0, 2), (1, 3), (0, 4), (3, 4)]
)
@pytest.mark.parametrize("interaction_only", [True, False])
@pytest.mark.parametrize("include_bias", [True, False])
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_num_combinations(
    n_features, min_degree, max_degree, interaction_only, include_bias, csr_container
):
    # [FUNCTION DOC]
    # Verifies n_output_features_ computed at fit equals the number of combinations
    #   yielded by the private _combinations generator, using a CSR matrix with a single
    #   non-zero in the last column.
    # KNOWN ISSUE: the min_degree parameter is never used (degree is set to max_degree
    #   only and _combinations gets min_degree 0), so the (min_degree, max_degree) pairs
    #   only vary max_degree.
    """
    Test that n_output_features_ is calculated correctly.
    """
    x = csr_container(([1], ([0], [n_features - 1])))
    est = PolynomialFeatures(
        degree=max_degree,
        interaction_only=interaction_only,
        include_bias=include_bias,
    )
    est.fit(x)
    num_combos = est.n_output_features_

    combos = PolynomialFeatures._combinations(
        n_features=n_features,
        min_degree=0,
        max_degree=max_degree,
        interaction_only=interaction_only,
        include_bias=include_bias,
    )
    assert num_combos == sum([1 for _ in combos])


@pytest.mark.parametrize(
    ["deg", "include_bias", "interaction_only", "dtype"],
    [
        (2, True, False, np.float32),
        (2, True, False, np.float64),
        (3, False, False, np.float64),
        (3, False, True, np.float64),
    ],
)
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_polynomial_features_csr_X_floats(
    deg, include_bias, interaction_only, dtype, csr_container, global_random_seed
):
    # [FUNCTION DOC]
    # Verifies CSR float32/float64 expansion matches dense results and dtype on a seeded
    #   1000x10 random sparse matrix.
    X_csr = csr_container(
        _sparse_random_array((1000, 10), density=0.5, random_state=global_random_seed)
    )
    X = X_csr.toarray()

    est = PolynomialFeatures(
        deg, include_bias=include_bias, interaction_only=interaction_only
    )
    Xt_csr = est.fit_transform(X_csr.astype(dtype))
    Xt_dense = est.fit_transform(X.astype(dtype))

    assert sparse.issparse(Xt_csr) and Xt_csr.format == "csr"
    assert Xt_csr.dtype == Xt_dense.dtype
    assert_array_almost_equal(Xt_csr.toarray(), Xt_dense)


@pytest.mark.parametrize(
    ["zero_row_index", "deg", "interaction_only"],
    [
        (0, 2, True),
        (1, 2, True),
        (2, 2, True),
        (0, 3, True),
        (1, 3, True),
        (2, 3, True),
        (0, 2, False),
        (1, 2, False),
        (2, 2, False),
        (0, 3, False),
        (1, 3, False),
        (2, 3, False),
    ],
)
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_polynomial_features_csr_X_zero_row(
    zero_row_index, deg, interaction_only, csr_container, global_random_seed
):
    # [FUNCTION DOC]
    # Verifies CSR expansion handles an all-zero row at each position (first, middle,
    #   last) and matches the dense result.
    # Side effects: assigns zeros into a CSR row (may emit a scipy sparse efficiency
    #   warning).
    X_csr = csr_container(
        _sparse_random_array((3, 10), density=1.0, random_state=global_random_seed)
    )
    X_csr[zero_row_index, :] = 0.0
    X = X_csr.toarray()

    est = PolynomialFeatures(deg, include_bias=False, interaction_only=interaction_only)
    Xt_csr = est.fit_transform(X_csr)
    Xt_dense = est.fit_transform(X)

    assert sparse.issparse(Xt_csr) and Xt_csr.format == "csr"
    assert Xt_csr.dtype == Xt_dense.dtype
    assert_array_almost_equal(Xt_csr.toarray(), Xt_dense)


# This degree should always be one more than the highest degree supported by
# _csr_expansion.
@pytest.mark.parametrize(
    ["include_bias", "interaction_only"],
    [(True, True), (True, False), (False, True), (False, False)],
)
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_polynomial_features_csr_X_degree_4(
    include_bias, interaction_only, csr_container, global_random_seed
):
    # [FUNCTION DOC]
    # Verifies degree 4 (beyond what the Cython CSR expansion supports directly) on CSR
    #   input still returns CSR output matching the dense result.
    X_csr = csr_container(
        _sparse_random_array((1000, 10), density=0.5, random_state=global_random_seed)
    )
    X = X_csr.toarray()

    est = PolynomialFeatures(
        4, include_bias=include_bias, interaction_only=interaction_only
    )
    Xt_csr = est.fit_transform(X_csr)
    Xt_dense = est.fit_transform(X)

    assert sparse.issparse(Xt_csr) and Xt_csr.format == "csr"
    assert Xt_csr.dtype == Xt_dense.dtype
    assert_array_almost_equal(Xt_csr.toarray(), Xt_dense)


@pytest.mark.parametrize(
    ["deg", "dim", "interaction_only"],
    [
        (2, 1, True),
        (2, 2, True),
        (3, 1, True),
        (3, 2, True),
        (3, 3, True),
        (2, 1, False),
        (2, 2, False),
        (3, 1, False),
        (3, 2, False),
        (3, 3, False),
    ],
)
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_polynomial_features_csr_X_dim_edges(
    deg, dim, interaction_only, csr_container, global_random_seed
):
    # [FUNCTION DOC]
    # Verifies CSR expansion at dimension edge cases (1 to 3 features, degree 2/3)
    #   matches dense results.
    X_csr = csr_container(
        _sparse_random_array((1000, dim), density=0.5, random_state=global_random_seed)
    )
    X = X_csr.toarray()

    est = PolynomialFeatures(deg, interaction_only=interaction_only)
    Xt_csr = est.fit_transform(X_csr)
    Xt_dense = est.fit_transform(X)

    assert sparse.issparse(Xt_csr) and Xt_csr.format == "csr"
    assert Xt_csr.dtype == Xt_dense.dtype
    assert_array_almost_equal(Xt_csr.toarray(), Xt_dense)


@pytest.mark.parametrize("interaction_only", [True, False])
@pytest.mark.parametrize("include_bias", [True, False])
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_csr_polynomial_expansion_index_overflow_non_regression(
    interaction_only, include_bias, csr_container
):
    # [FUNCTION DOC]
    # Non-regression for gh-16803: degree-2 expansion of a CSR matrix with 120001
    #   features promotes indices/indptr to int64 when output column indices exceed
    #   int32, and the data, row and column positions of non-zeros match values computed
    #   by hand; on platforms where the output cannot be indexed, checks the ValueError
    #   instead and returns early.
    """Check the automatic index dtype promotion to `np.int64` when needed.

    This ensures that sufficiently large input configurations get
    properly promoted to use `np.int64` for index and indptr representation
    while preserving data integrity. Non-regression test for gh-16803.

    Note that this is only possible for Python runtimes with a 64 bit address
    space. On 32 bit platforms, a `ValueError` is raised instead.
    """

    def degree_2_calc(d, i, j):
        # [FUNCTION DOC]
        # HELPER: Returns the zero-based degree-2 output column offset (before adding
        #   the degree-1 block) for the product of input columns i <= j among d
        #   features, using the interaction-only or full triangular indexing formula
        #   depending on interaction_only.
        if interaction_only:
            return d * i - (i**2 + 3 * i) // 2 - 1 + j
        else:
            return d * i - (i**2 + i) // 2 + j

    n_samples = 13
    n_features = 120001
    data_dtype = np.float32
    data = np.arange(1, 5, dtype=np.int64)
    row = np.array([n_samples - 2, n_samples - 2, n_samples - 1, n_samples - 1])
    # An int64 dtype is required to avoid overflow error on Windows within the
    # `degree_2_calc` function.
    col = np.array(
        [n_features - 2, n_features - 1, n_features - 2, n_features - 1], dtype=np.int64
    )
    X = csr_container(
        (data, (row, col)),
        shape=(n_samples, n_features),
        dtype=data_dtype,
    )
    pf = PolynomialFeatures(
        interaction_only=interaction_only, include_bias=include_bias, degree=2
    )

    # Calculate the number of combinations a-priori, and if needed check for
    # the correct ValueError and terminate the test early.
    num_combinations = pf._num_combinations(
        n_features=n_features,
        min_degree=0,
        max_degree=2,
        interaction_only=pf.interaction_only,
        include_bias=pf.include_bias,
    )
    if num_combinations > np.iinfo(np.intp).max:
        msg = (
            r"The output that would result from the current configuration would have"
            r" \d* features which is too large to be indexed"
        )
        with pytest.raises(ValueError, match=msg):
            pf.fit(X)
        return
    X_trans = pf.fit_transform(X)
    row_nonzero, col_nonzero = X_trans.nonzero()
    n_degree_1_features_out = n_features + include_bias
    max_degree_2_idx = (
        degree_2_calc(n_features, col[int(not interaction_only)], col[1])
        + n_degree_1_features_out
    )

    # Account for bias of all samples except last one which will be handled
    # separately since there are distinct data values before it
    data_target = [1] * (n_samples - 2) if include_bias else []
    col_nonzero_target = [0] * (n_samples - 2) if include_bias else []

    for i in range(2):
        x = data[2 * i]
        y = data[2 * i + 1]
        x_idx = col[2 * i]
        y_idx = col[2 * i + 1]
        if include_bias:
            data_target.append(1)
            col_nonzero_target.append(0)
        data_target.extend([x, y])
        col_nonzero_target.extend(
            [x_idx + int(include_bias), y_idx + int(include_bias)]
        )
        if not interaction_only:
            data_target.extend([x * x, x * y, y * y])
            col_nonzero_target.extend(
                [
                    degree_2_calc(n_features, x_idx, x_idx) + n_degree_1_features_out,
                    degree_2_calc(n_features, x_idx, y_idx) + n_degree_1_features_out,
                    degree_2_calc(n_features, y_idx, y_idx) + n_degree_1_features_out,
                ]
            )
        else:
            data_target.extend([x * y])
            col_nonzero_target.append(
                degree_2_calc(n_features, x_idx, y_idx) + n_degree_1_features_out
            )

    nnz_per_row = int(include_bias) + 3 + 2 * int(not interaction_only)

    assert pf.n_output_features_ == max_degree_2_idx + 1
    assert X_trans.dtype == data_dtype
    assert X_trans.shape == (n_samples, max_degree_2_idx + 1)
    assert X_trans.indptr.dtype == X_trans.indices.dtype == np.int64
    # Ensure that dtype promotion was actually required:
    assert X_trans.indices.max() > np.iinfo(np.int32).max

    row_nonzero_target = list(range(n_samples - 2)) if include_bias else []
    row_nonzero_target.extend(
        [n_samples - 2] * nnz_per_row + [n_samples - 1] * nnz_per_row
    )

    assert_allclose(X_trans.data, data_target)
    assert_array_equal(row_nonzero, row_nonzero_target)
    assert_array_equal(col_nonzero, col_nonzero_target)


@pytest.mark.parametrize(
    "degree, n_features",
    [
        # Needs promotion to int64 when interaction_only=False
        (2, 65535),
        (3, 2344),
        # This guarantees that the intermediate operation when calculating
        # output columns would overflow a C-long, hence checks that python-
        # longs are being used.
        (2, int(np.sqrt(np.iinfo(np.int64).max) + 1)),
        (3, 65535),
        # This case tests the second clause of the overflow check which
        # takes into account the value of `n_features` itself.
        (2, int(np.sqrt(np.iinfo(np.int64).max))),
    ],
)
@pytest.mark.parametrize("interaction_only", [True, False])
@pytest.mark.parametrize("include_bias", [True, False])
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_csr_polynomial_expansion_index_overflow(
    degree, n_features, interaction_only, include_bias, csr_container
):
    # [FUNCTION DOC]
    # Verifies index-dtype promotion edge cases: for a single non-zero in the last of
    #   n_features columns, the output has the expected index dtype (int32 or int64 by
    #   number of combinations), nnz, values at hand-computed degree-1/2/3 indices and
    #   n_output_features_; configurations not indexable by intp must raise ValueError.
    """Tests known edge-cases to the dtype promotion strategy and custom
    Cython code, including a current bug in the upstream
    `scipy.sparse.hstack`.
    """
    data = [1.0]
    # Use int32 indices as much as we can
    indices_dtype = np.int32 if n_features - 1 <= np.iinfo(np.int32).max else np.int64
    row = np.array([0], dtype=indices_dtype)
    col = np.array([n_features - 1], dtype=indices_dtype)

    # First degree index
    expected_indices = [
        n_features - 1 + int(include_bias),
    ]
    # Second degree index
    expected_indices.append(n_features * (n_features + 1) // 2 + expected_indices[0])
    # Third degree index
    expected_indices.append(
        n_features * (n_features + 1) * (n_features + 2) // 6 + expected_indices[1]
    )

    X = csr_container((data, (row, col)))
    pf = PolynomialFeatures(
        interaction_only=interaction_only, include_bias=include_bias, degree=degree
    )

    # Calculate the number of combinations a-priori, and if needed check for
    # the correct ValueError and terminate the test early.
    num_combinations = pf._num_combinations(
        n_features=n_features,
        min_degree=0,
        max_degree=degree,
        interaction_only=pf.interaction_only,
        include_bias=pf.include_bias,
    )
    if num_combinations > np.iinfo(np.intp).max:
        msg = (
            r"The output that would result from the current configuration would have"
            r" \d* features which is too large to be indexed"
        )
        with pytest.raises(ValueError, match=msg):
            pf.fit(X)
        return

    X_trans = pf.fit_transform(X)

    expected_dtype = np.int64 if num_combinations > np.iinfo(np.int32).max else np.int32
    # Terms higher than first degree
    non_bias_terms = 1 + (degree - 1) * int(not interaction_only)
    expected_nnz = int(include_bias) + non_bias_terms
    assert X_trans.dtype == X.dtype
    assert X_trans.shape == (1, pf.n_output_features_)
    assert X_trans.indptr.dtype == X_trans.indices.dtype == expected_dtype
    assert X_trans.nnz == expected_nnz

    if include_bias:
        assert X_trans[0, 0] == pytest.approx(1.0)
    for idx in range(non_bias_terms):
        assert X_trans[0, expected_indices[idx]] == pytest.approx(1.0)

    offset = interaction_only * n_features
    if degree == 3:
        offset *= 1 + n_features
    assert pf.n_output_features_ == expected_indices[degree - 1] + 1 - offset


@pytest.mark.parametrize("interaction_only", [True, False])
@pytest.mark.parametrize("include_bias", [True, False])
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_csr_polynomial_expansion_too_large_to_index(
    interaction_only, include_bias, csr_container
):
    # [FUNCTION DOC]
    # Verifies fit and fit_transform raise the "too large to be indexed" ValueError when
    #   n_features is half the int64 max with degree (2, 2).
    n_features = np.iinfo(np.int64).max // 2
    data = [1.0]
    row = [0]
    col = [n_features - 1]
    X = csr_container((data, (row, col)))
    pf = PolynomialFeatures(
        interaction_only=interaction_only, include_bias=include_bias, degree=(2, 2)
    )
    msg = (
        r"The output that would result from the current configuration would have \d*"
        r" features which is too large to be indexed"
    )
    with pytest.raises(ValueError, match=msg):
        pf.fit(X)
    with pytest.raises(ValueError, match=msg):
        pf.fit_transform(X)


@pytest.mark.parametrize("sparse_container", CSR_CONTAINERS + CSC_CONTAINERS)
def test_polynomial_features_behaviour_on_zero_degree(sparse_container):
    # [FUNCTION DOC]
    # Verifies degree 0 or (0, 0) with include_bias=False raises ValueError, and degree
    #   0 with include_bias=True outputs a single column of ones for dense and sparse
    #   input.
    """Check that PolynomialFeatures raises error when degree=0 and include_bias=False,
    and output a single constant column when include_bias=True
    """
    X = np.ones((10, 2))
    poly = PolynomialFeatures(degree=0, include_bias=False)
    err_msg = (
        "Setting degree to zero and include_bias to False would result in"
        " an empty output array."
    )
    with pytest.raises(ValueError, match=err_msg):
        poly.fit_transform(X)

    poly = PolynomialFeatures(degree=(0, 0), include_bias=False)
    err_msg = (
        "Setting both min_degree and max_degree to zero and include_bias to"
        " False would result in an empty output array."
    )
    with pytest.raises(ValueError, match=err_msg):
        poly.fit_transform(X)

    for _X in [X, sparse_container(X)]:
        poly = PolynomialFeatures(degree=0, include_bias=True)
        output = poly.fit_transform(_X)
        # convert to dense array if needed
        if sparse.issparse(output):
            output = output.toarray()
        assert_array_equal(output, np.ones((X.shape[0], 1)))


def test_sizeof_LARGEST_INT_t():
    # [FUNCTION DOC]
    # Verifies the size of the Cython LARGEST_INT_t type is 8 bytes on Windows or 32-bit
    #   (non-emscripten) platforms and 16 bytes otherwise.
    # On Windows, scikit-learn is typically compiled with MSVC that
    # does not support int128 arithmetic (at the time of writing):
    # https://stackoverflow.com/a/6761962/163740
    if sys.platform == "win32" or (
        sys.maxsize <= 2**32 and sys.platform != "emscripten"
    ):
        expected_size = 8
    else:
        expected_size = 16

    assert _get_sizeof_LARGEST_INT_t() == expected_size


@pytest.mark.xfail(
    sys.platform == "win32",
    reason=(
        "On Windows, scikit-learn is typically compiled with MSVC that does not support"
        " int128 arithmetic (at the time of writing)"
    ),
    run=True,
)
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_csr_polynomial_expansion_windows_fail(csr_container):
    # [FUNCTION DOC]
    # Verifies degree-3 CSR expansion where intermediate index arithmetic overflows
    #   int64 still gives 1.0 at the hand-computed degree 1/2/3 indices (requires int128
    #   support; xfail on Windows), or raises the indexing ValueError on 32-bit
    #   platforms.
    # Minimum needed to ensure integer overflow occurs while guaranteeing an
    # int64-indexable output.
    n_features = int(np.iinfo(np.int64).max ** (1 / 3) + 3)
    data = [1.0]
    row = [0]
    col = [n_features - 1]

    # First degree index
    expected_indices = [
        n_features - 1,
    ]
    # Second degree index
    expected_indices.append(
        int(n_features * (n_features + 1) // 2 + expected_indices[0])
    )
    # Third degree index
    expected_indices.append(
        int(n_features * (n_features + 1) * (n_features + 2) // 6 + expected_indices[1])
    )

    X = csr_container((data, (row, col)))
    pf = PolynomialFeatures(interaction_only=False, include_bias=False, degree=3)
    if sys.maxsize <= 2**32:
        msg = (
            r"The output that would result from the current configuration would"
            r" have \d*"
            r" features which is too large to be indexed"
        )
        with pytest.raises(ValueError, match=msg):
            pf.fit_transform(X)
    else:
        X_trans = pf.fit_transform(X)
        for idx in range(3):
            assert X_trans[0, expected_indices[idx]] == pytest.approx(1.0)


@pytest.mark.parametrize(
    "array_namespace, device_name, dtype_name",
    yield_namespace_device_dtype_combinations(),
)
@pytest.mark.parametrize("interaction_only", [True, False])
@pytest.mark.parametrize("include_bias", [True, False])
@pytest.mark.parametrize("degree", [2, (2, 2), 3, (3, 3)])
def test_polynomial_features_array_api_compliance(
    two_features_degree3,
    degree,
    include_bias,
    interaction_only,
    array_namespace,
    device_name,
    dtype_name,
):
    # [FUNCTION DOC]
    # Verifies PolynomialFeatures under array_api_dispatch gives the same values as
    #   NumPy, and keeps the namespace, device and dtype of the input, for several
    #   degree/bias/interaction settings.
    """Test array API compliance for PolynomialFeatures on 2 features up to degree 3."""
    xp, device = _array_api_for_tests(array_namespace, device_name, dtype_name)
    X, _ = two_features_degree3
    X_np = X.astype(dtype_name)
    X_xp = xp.asarray(X_np, device=device)
    with config_context(array_api_dispatch=True):
        tf_np = PolynomialFeatures(
            degree=degree, include_bias=include_bias, interaction_only=interaction_only
        ).fit(X_np)

        tf_xp = PolynomialFeatures(
            degree=degree, include_bias=include_bias, interaction_only=interaction_only
        ).fit(X_xp)
        out_np = tf_np.transform(X_np)
        out_xp = tf_xp.transform(X_xp)
        assert_allclose(move_to(out_xp, xp=np, device="cpu"), out_np)
        assert get_namespace(out_xp)[0].__name__ == xp.__name__
        assert array_api_device(out_xp) == array_api_device(X_xp)
        assert out_xp.dtype == X_xp.dtype


@pytest.mark.parametrize(
    "array_namespace, device_name, dtype_name",
    yield_namespace_device_dtype_combinations(),
)
def test_polynomial_features_array_api_raises_on_order_F(
    array_namespace, device_name, dtype_name
):
    # [FUNCTION DOC]
    # Verifies transform with order="F" raises ValueError for non-NumPy array API
    #   namespaces and succeeds for NumPy.
    """Test that PolynomialFeatures with order='F' raises ValueError on
    array API namespaces other than numpy."""
    xp, device = _array_api_for_tests(array_namespace, device_name, dtype_name)
    X = np.arange(6).reshape((3, 2)).astype(dtype_name)
    X_xp = xp.asarray(X, device=device)
    msg = "PolynomialFeatures does not support order='F' for non-numpy arrays"
    with config_context(array_api_dispatch=True):
        pf = PolynomialFeatures(order="F").fit(X_xp)
        if _is_numpy_namespace(xp):  # Numpy should not raise
            pf.transform(X_xp)
        else:
            with pytest.raises(ValueError, match=msg):
                pf.transform(X_xp)
