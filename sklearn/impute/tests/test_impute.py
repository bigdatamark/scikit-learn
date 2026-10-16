# [MODULE DOC]
# Tests for SimpleImputer, IterativeImputer, MissingIndicator and KNNImputer
# keep_empty_features behaviour: statistics per strategy on dense and sparse inputs,
# dtype handling and error messages, copy semantics, IterativeImputer convergence,
# ordering, clipping, posterior sampling and min/max bounds, missing indicator output
# formats, inverse_transform, the _most_frequent helper and pandas nullable types.
import io
import re
import warnings
from itertools import product

import numpy as np
import pytest
from scipy import sparse
from scipy.stats import kstest

from sklearn import tree
from sklearn.datasets import load_diabetes
from sklearn.dummy import DummyRegressor
from sklearn.exceptions import ConvergenceWarning

# make IterativeImputer available
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer, MissingIndicator, SimpleImputer
from sklearn.impute._base import _most_frequent
from sklearn.linear_model import ARDRegression, BayesianRidge, RidgeCV
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline, make_union
from sklearn.random_projection import _sparse_random_matrix
from sklearn.utils._testing import (
    _convert_container,
    assert_allclose,
    assert_allclose_dense_sparse,
    assert_array_almost_equal,
    assert_array_equal,
)
from sklearn.utils.fixes import (
    BSR_CONTAINERS,
    COO_CONTAINERS,
    CSC_CONTAINERS,
    CSR_CONTAINERS,
    LIL_CONTAINERS,
    SCIPY_VERSION_BELOW_1_12,
)


def _assert_array_equal_and_same_dtype(x, y):
    # [FUNCTION DOC]
    # HELPER: Asserts two arrays are exactly equal and have the same dtype.
    # Exceptions: AssertionError on mismatch.
    assert_array_equal(x, y)
    assert x.dtype == y.dtype


def _assert_allclose_and_same_dtype(x, y):
    # [FUNCTION DOC]
    # HELPER: Asserts two arrays are numerically close and have the same dtype.
    # Exceptions: AssertionError on mismatch.
    assert_allclose(x, y)
    assert x.dtype == y.dtype


def _check_statistics(
    X, X_true, strategy, statistics, missing_values, sparse_container
):
    # [FUNCTION DOC]
    # HELPER: Fits SimpleImputer with the given strategy and missing_values on dense X
    # and on sparse_container(X), and asserts statistics_ equals statistics and the
    # transformed data equals X_true (densifying sparse output). Uses almost-equality
    # when X or X_true is float.
    # Inputs: X, X_true - 2-D arrays; strategy - str; statistics - expected per-feature
    # values; missing_values - marker; sparse_container - sparse matrix/array class.
    # Exceptions: AssertionError with a message naming the parameters and sparsity.
    """Utility function for testing imputation for a given strategy.

    Test with dense and sparse arrays

    Check that:
        - the statistics (mean, median, mode) are correct
        - the missing values are imputed correctly"""

    err_msg = "Parameters: strategy = %s, missing_values = %s, sparse = {0}" % (
        strategy,
        missing_values,
    )

    assert_ae = assert_array_equal

    if X.dtype.kind == "f" or X_true.dtype.kind == "f":
        assert_ae = assert_array_almost_equal

    # Normal matrix
    imputer = SimpleImputer(missing_values=missing_values, strategy=strategy)
    X_trans = imputer.fit(X).transform(X.copy())
    assert_ae(imputer.statistics_, statistics, err_msg=err_msg.format(False))
    assert_ae(X_trans, X_true, err_msg=err_msg.format(False))

    # Sparse matrix
    imputer = SimpleImputer(missing_values=missing_values, strategy=strategy)
    imputer.fit(sparse_container(X))
    X_trans = imputer.transform(sparse_container(X.copy()))

    if sparse.issparse(X_trans):
        X_trans = X_trans.toarray()

    assert_ae(imputer.statistics_, statistics, err_msg=err_msg.format(True))
    assert_ae(X_trans, X_true, err_msg=err_msg.format(True))


@pytest.mark.parametrize("strategy", ["mean", "median", "most_frequent", "constant"])
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_imputation_shape(strategy, csr_container):
    # [FUNCTION DOC]
    # Verifies output shape (10, 2) for SimpleImputer on CSR and dense input and for
    # IterativeImputer with each initial_strategy, on data with all-NaN even rows.
    # Note: uses the unseeded global RNG.
    # Verify the shapes of the imputed matrix for different strategies.
    X = np.random.randn(10, 2)
    X[::2] = np.nan

    imputer = SimpleImputer(strategy=strategy)
    X_imputed = imputer.fit_transform(csr_container(X))
    assert X_imputed.shape == (10, 2)
    X_imputed = imputer.fit_transform(X)
    assert X_imputed.shape == (10, 2)

    iterative_imputer = IterativeImputer(initial_strategy=strategy)
    X_imputed = iterative_imputer.fit_transform(X)
    assert X_imputed.shape == (10, 2)


@pytest.mark.parametrize("strategy", ["mean", "median", "most_frequent"])
def test_imputation_deletion_warning(strategy):
    # [FUNCTION DOC]
    # Verifies transform emits a UserWarning "Skipping..." when a column was entirely
    # missing during fit, for mean/median/most_frequent.
    X = np.ones((3, 5))
    X[:, 0] = np.nan
    imputer = SimpleImputer(strategy=strategy).fit(X)

    with pytest.warns(UserWarning, match="Skipping"):
        imputer.transform(X)


@pytest.mark.parametrize("strategy", ["mean", "median", "most_frequent"])
def test_imputation_deletion_warning_feature_names(strategy):
    # [FUNCTION DOC]
    # Verifies feature_names_in_ is set from a DataFrame and the skipped-feature warning
    # names the all-missing column 'b'.
    pd = pytest.importorskip("pandas")

    missing_values = np.nan
    feature_names = np.array(["a", "b", "c", "d"], dtype=object)
    X = pd.DataFrame(
        [
            [missing_values, missing_values, 1, missing_values],
            [4, missing_values, 2, 10],
        ],
        columns=feature_names,
    )

    imputer = SimpleImputer(strategy=strategy).fit(X)

    # check SimpleImputer returning feature name attribute correctly
    assert_array_equal(imputer.feature_names_in_, feature_names)

    # ensure that skipped feature warning includes feature name
    with pytest.warns(
        UserWarning, match=r"Skipping features without any observed values: \['b'\]"
    ):
        imputer.transform(X)


@pytest.mark.parametrize("strategy", ["mean", "median", "most_frequent", "constant"])
@pytest.mark.parametrize("csc_container", CSC_CONTAINERS)
def test_imputation_error_sparse_0(strategy, csc_container):
    # [FUNCTION DOC]
    # Verifies SimpleImputer raises "Provide a dense array" for CSC input with
    # missing_values=0 in fit, and in transform after fitting on dense data.
    # check that error are raised when missing_values = 0 and input is sparse
    X = np.ones((3, 5))
    X[0] = 0
    X = csc_container(X)

    imputer = SimpleImputer(strategy=strategy, missing_values=0)
    with pytest.raises(ValueError, match="Provide a dense array"):
        imputer.fit(X)

    imputer.fit(X.toarray())
    with pytest.raises(ValueError, match="Provide a dense array"):
        imputer.transform(X)


def safe_median(arr, *args, **kwargs):
    # [FUNCTION DOC]
    # HELPER: Median that returns NaN for empty input instead of erroring; otherwise
    # delegates to np.median with the extra arguments.
    # Outputs: float or NaN.
    # np.median([]) raises a TypeError for numpy >= 1.10.1
    length = arr.size if hasattr(arr, "size") else len(arr)
    return np.nan if length == 0 else np.median(arr, *args, **kwargs)


def safe_mean(arr, *args, **kwargs):
    # [FUNCTION DOC]
    # HELPER: Mean that returns NaN for empty input instead of warning; otherwise
    # delegates to np.mean. Outputs: float or NaN.
    # np.mean([]) raises a RuntimeWarning for numpy >= 1.10.1
    length = arr.size if hasattr(arr, "size") else len(arr)
    return np.nan if length == 0 else np.mean(arr, *args, **kwargs)


@pytest.mark.parametrize("csc_container", CSC_CONTAINERS)
def test_imputation_mean_median(csc_container):
    # [FUNCTION DOC]
    # Verifies mean and median strategies with NaN markers: builds a 100x20 matrix whose
    # columns mix random signed values, explicit zeros and NaNs in varying proportions
    # (including all-zero and all-NaN columns), computes the expected statistic per
    # column with safe_mean/safe_median over values and zeros, builds the expected
    # imputed matrix, shuffles both identically per column, drops columns that end up
    # all NaN, and checks via _check_statistics on dense and CSC input.
    # KNOWN ISSUE: the branch for test_missing_values equal to 0 is unreachable (only
    # NaN is tested), as its own comment notes. The comment "Mean doesn't support
    # columns containing NaNs, median does" does not match the code, which keeps columns
    # with any NaN for mean.
    # Test imputation using the mean and median strategies, when
    # missing_values != 0.
    rng = np.random.RandomState(0)

    dim = 10
    dec = 10
    shape = (dim * dim, dim + dec)

    zeros = np.zeros(shape[0])
    values = np.arange(1, shape[0] + 1)
    values[4::2] = -values[4::2]

    tests = [
        ("mean", np.nan, lambda z, v, p: safe_mean(np.hstack((z, v)))),
        ("median", np.nan, lambda z, v, p: safe_median(np.hstack((z, v)))),
    ]

    for strategy, test_missing_values, true_value_fun in tests:
        X = np.empty(shape)
        X_true = np.empty(shape)
        true_statistics = np.empty(shape[1])

        # Create a matrix X with columns
        #    - with only zeros,
        #    - with only missing values
        #    - with zeros, missing values and values
        # And a matrix X_true containing all true values
        for j in range(shape[1]):
            nb_zeros = (j - dec + 1 > 0) * (j - dec + 1) * (j - dec + 1)
            nb_missing_values = max(shape[0] + dec * dec - (j + dec) * (j + dec), 0)
            nb_values = shape[0] - nb_zeros - nb_missing_values

            z = zeros[:nb_zeros]
            p = np.repeat(test_missing_values, nb_missing_values)
            v = values[rng.permutation(len(values))[:nb_values]]

            true_statistics[j] = true_value_fun(z, v, p)

            # Create the columns
            X[:, j] = np.hstack((v, z, p))

            if 0 == test_missing_values:
                # XXX unreached code as of v0.22
                X_true[:, j] = np.hstack(
                    (v, np.repeat(true_statistics[j], nb_missing_values + nb_zeros))
                )
            else:
                X_true[:, j] = np.hstack(
                    (v, z, np.repeat(true_statistics[j], nb_missing_values))
                )

            # Shuffle them the same way
            np.random.RandomState(j).shuffle(X[:, j])
            np.random.RandomState(j).shuffle(X_true[:, j])

        # Mean doesn't support columns containing NaNs, median does
        if strategy == "median":
            cols_to_keep = ~np.isnan(X_true).any(axis=0)
        else:
            cols_to_keep = ~np.isnan(X_true).all(axis=0)

        X_true = X_true[:, cols_to_keep]

        _check_statistics(
            X, X_true, strategy, true_statistics, test_missing_values, csc_container
        )


@pytest.mark.parametrize("csc_container", CSC_CONTAINERS)
def test_imputation_median_special_cases(csc_container):
    # [FUNCTION DOC]
    # Verifies median imputation on sparse boundary cases (implicit zeros, odd/even
    # counts, negative/positive mixes) against hand-computed statistics, dense and CSC.
    # Test median imputation with sparse boundary cases
    X = np.array(
        [
            [0, np.nan, np.nan],  # odd: implicit zero
            [5, np.nan, np.nan],  # odd: explicit nonzero
            [0, 0, np.nan],  # even: average two zeros
            [-5, 0, np.nan],  # even: avg zero and neg
            [0, 5, np.nan],  # even: avg zero and pos
            [4, 5, np.nan],  # even: avg nonzeros
            [-4, -5, np.nan],  # even: avg negatives
            [-1, 2, np.nan],  # even: crossing neg and pos
        ]
    ).transpose()

    X_imputed_median = np.array(
        [
            [0, 0, 0],
            [5, 5, 5],
            [0, 0, 0],
            [-5, 0, -2.5],
            [0, 5, 2.5],
            [4, 5, 4.5],
            [-4, -5, -4.5],
            [-1, 2, 0.5],
        ]
    ).transpose()
    statistics_median = [0, 5, 0, -2.5, 2.5, 4.5, -4.5, 0.5]

    _check_statistics(
        X, X_imputed_median, "median", statistics_median, np.nan, csc_container
    )


@pytest.mark.parametrize("strategy", ["mean", "median"])
@pytest.mark.parametrize("dtype", [None, object, str])
def test_imputation_mean_median_error_invalid_type(strategy, dtype):
    # [FUNCTION DOC]
    # Verifies mean/median raise ValueError "non-numeric data" on string-containing
    # numpy arrays (default, object and str dtypes).
    X = np.array([["a", "b", 3], [4, "e", 6], ["g", "h", 9]], dtype=dtype)
    msg = "non-numeric data:\ncould not convert string to float:"
    with pytest.raises(ValueError, match=msg):
        imputer = SimpleImputer(strategy=strategy)
        imputer.fit_transform(X)


@pytest.mark.parametrize("strategy", ["mean", "median"])
@pytest.mark.parametrize("constructor_name", ["list", "pandas"])
def test_imputation_mean_median_error_invalid_type_list_pandas(
    strategy, constructor_name
):
    # [FUNCTION DOC]
    # Same as above for list and DataFrame inputs.
    X = [["a", "b", 3], [4, "e", 6], ["g", "h", 9]]
    X = _convert_container(X, constructor_name)
    msg = "non-numeric data:\ncould not convert string to float:"
    with pytest.raises(ValueError, match=msg):
        imputer = SimpleImputer(strategy=strategy)
        imputer.fit_transform(X)


@pytest.mark.parametrize("strategy", ["constant", "most_frequent"])
@pytest.mark.parametrize("dtype", [str, np.dtype("U"), np.dtype("S")])
def test_imputation_const_mostf_error_invalid_types(strategy, dtype):
    # [FUNCTION DOC]
    # Verifies most_frequent/constant raise "SimpleImputer does not support data" for
    # numpy string dtypes (str, U, S) containing NaN.
    # Test imputation on non-numeric data using "most_frequent" and "constant"
    # strategy
    X = np.array(
        [
            [np.nan, np.nan, "a", "f"],
            [np.nan, "c", np.nan, "d"],
            [np.nan, "b", "d", np.nan],
            [np.nan, "c", "d", "h"],
        ],
        dtype=dtype,
    )

    err_msg = "SimpleImputer does not support data"
    with pytest.raises(ValueError, match=err_msg):
        imputer = SimpleImputer(strategy=strategy)
        imputer.fit(X).transform(X)


@pytest.mark.parametrize("csc_container", CSC_CONTAINERS)
def test_imputation_most_frequent(csc_container):
    # [FUNCTION DOC]
    # Verifies most_frequent on integers with marker -1 picks the smallest of tied modes
    # and drops the all-missing first column (statistic NaN), dense and CSC.
    # Test imputation using the most-frequent strategy.
    X = np.array(
        [
            [-1, -1, 0, 5],
            [-1, 2, -1, 3],
            [-1, 1, 3, -1],
            [-1, 2, 3, 7],
        ]
    )

    X_true = np.array(
        [
            [2, 0, 5],
            [2, 3, 3],
            [1, 3, 3],
            [2, 3, 7],
        ]
    )

    # scipy.stats.mode, used in SimpleImputer, doesn't return the first most
    # frequent as promised in the doc but the lowest most frequent. When this
    # test will fail after an update of scipy, SimpleImputer will need to be
    # updated to be consistent with the new (correct) behaviour
    _check_statistics(X, X_true, "most_frequent", [np.nan, 2, 3, 3], -1, csc_container)


@pytest.mark.parametrize("marker", [None, np.nan, "NAN", "", 0])
def test_imputation_most_frequent_objects(marker):
    # [FUNCTION DOC]
    # Verifies most_frequent on object arrays for several markers (None, NaN, "NAN", "",
    # 0), dropping the all-missing column and filling with the modal string.
    # Test imputation using the most-frequent strategy.
    X = np.array(
        [
            [marker, marker, "a", "f"],
            [marker, "c", marker, "d"],
            [marker, "b", "d", marker],
            [marker, "c", "d", "h"],
        ],
        dtype=object,
    )

    X_true = np.array(
        [
            ["c", "a", "f"],
            ["c", "d", "d"],
            ["b", "d", "d"],
            ["c", "d", "h"],
        ],
        dtype=object,
    )

    imputer = SimpleImputer(missing_values=marker, strategy="most_frequent")
    X_trans = imputer.fit(X).transform(X)

    assert_array_equal(X_trans, X_true)


@pytest.mark.parametrize("dtype", [object, "category"])
def test_imputation_most_frequent_pandas(dtype):
    # [FUNCTION DOC]
    # Verifies most_frequent on DataFrames read from CSV with object and category
    # dtypes, dropping the empty Cat4 column.
    # Test imputation using the most frequent strategy on pandas df
    pd = pytest.importorskip("pandas")

    f = io.StringIO("Cat1,Cat2,Cat3,Cat4\n,i,x,\na,,y,\na,j,,\nb,j,x,")

    df = pd.read_csv(f, dtype=dtype)

    X_true = np.array(
        [["a", "i", "x"], ["a", "j", "y"], ["a", "j", "x"], ["b", "j", "x"]],
        dtype=object,
    )

    imputer = SimpleImputer(strategy="most_frequent")
    X_trans = imputer.fit_transform(df)

    assert_array_equal(X_trans, X_true)


@pytest.mark.parametrize("X_data, missing_value", [(1, 0), (1.0, np.nan)])
def test_imputation_constant_error_invalid_type(X_data, missing_value):
    # [FUNCTION DOC]
    # Verifies constant strategy raises ValueError when fill_value "x" cannot be cast to
    # the float data dtype.
    # Verify that exceptions are raised on invalid fill_value type
    X = np.full((3, 5), X_data, dtype=float)
    X[0, 0] = missing_value

    fill_value = "x"
    err_msg = f"fill_value={fill_value!r} (of type {type(fill_value)!r}) cannot be cast"
    with pytest.raises(ValueError, match=re.escape(err_msg)):
        imputer = SimpleImputer(
            missing_values=missing_value, strategy="constant", fill_value=fill_value
        )
        imputer.fit_transform(X)


@pytest.mark.parametrize("keep_empty_features", [True, False])
def test_imputation_constant_integer(keep_empty_features):
    # [FUNCTION DOC]
    # Verifies constant strategy with integer data and fill_value 0; the all-missing
    # last column is kept (filled) only when keep_empty_features.
    # Test imputation using the constant strategy on integers
    X = np.array([[-1, 2, 3, -1], [4, -1, 5, -1], [6, 7, -1, -1], [8, 9, 0, -1]])

    X_true = np.array([[0, 2, 3, 0], [4, 0, 5, 0], [6, 7, 0, 0], [8, 9, 0, 0]])
    if not keep_empty_features:
        X_true = X_true[:, :-1]

    imputer = SimpleImputer(
        missing_values=-1,
        strategy="constant",
        fill_value=0,
        keep_empty_features=keep_empty_features,
    )
    X_trans = imputer.fit_transform(X)

    assert_array_equal(X_trans, X_true)


@pytest.mark.parametrize("array_constructor", CSR_CONTAINERS + [np.asarray])
@pytest.mark.parametrize("keep_empty_features", [True, False])
def test_imputation_constant_float(array_constructor, keep_empty_features):
    # [FUNCTION DOC]
    # Verifies constant strategy with float data (dense and CSR) and fill_value -1, with
    # and without keeping the empty column.
    # Test imputation using the constant strategy on floats
    X = np.array(
        [
            [np.nan, 1.1, 0, np.nan],
            [1.2, np.nan, 1.3, np.nan],
            [0, 0, np.nan, np.nan],
            [1.4, 1.5, 0, np.nan],
        ]
    )

    X_true = np.array(
        [[-1, 1.1, 0, -1], [1.2, -1, 1.3, -1], [0, 0, -1, -1], [1.4, 1.5, 0, -1]]
    )
    if not keep_empty_features:
        X_true = X_true[:, :-1]

    X = array_constructor(X)

    X_true = array_constructor(X_true)

    imputer = SimpleImputer(
        strategy="constant", fill_value=-1, keep_empty_features=keep_empty_features
    )
    X_trans = imputer.fit_transform(X)

    assert_allclose_dense_sparse(X_trans, X_true)


@pytest.mark.parametrize("marker", [None, np.nan, "NAN", "", 0])
@pytest.mark.parametrize("keep_empty_features", [True, False])
def test_imputation_constant_object(marker, keep_empty_features):
    # [FUNCTION DOC]
    # Verifies constant strategy on object data with various markers fills with
    # "missing", dropping or keeping the empty column.
    # Test imputation using the constant strategy on objects
    X = np.array(
        [
            [marker, "a", "b", marker],
            ["c", marker, "d", marker],
            ["e", "f", marker, marker],
            ["g", "h", "i", marker],
        ],
        dtype=object,
    )

    X_true = np.array(
        [
            ["missing", "a", "b", "missing"],
            ["c", "missing", "d", "missing"],
            ["e", "f", "missing", "missing"],
            ["g", "h", "i", "missing"],
        ],
        dtype=object,
    )
    if not keep_empty_features:
        X_true = X_true[:, :-1]

    imputer = SimpleImputer(
        missing_values=marker,
        strategy="constant",
        fill_value="missing",
        keep_empty_features=keep_empty_features,
    )
    X_trans = imputer.fit_transform(X)

    assert_array_equal(X_trans, X_true)


@pytest.mark.parametrize("dtype", [object, "category"])
@pytest.mark.parametrize("keep_empty_features", [True, False])
def test_imputation_constant_pandas(dtype, keep_empty_features):
    # [FUNCTION DOC]
    # Verifies constant strategy on DataFrames (object/category) uses the default fill
    # "missing_value", dropping or keeping the empty column.
    # Test imputation using the constant strategy on pandas df
    pd = pytest.importorskip("pandas")

    f = io.StringIO("Cat1,Cat2,Cat3,Cat4\n,i,x,\na,,y,\na,j,,\nb,j,x,")

    df = pd.read_csv(f, dtype=dtype)

    X_true = np.array(
        [
            ["missing_value", "i", "x", "missing_value"],
            ["a", "missing_value", "y", "missing_value"],
            ["a", "j", "missing_value", "missing_value"],
            ["b", "j", "x", "missing_value"],
        ],
        dtype=object,
    )
    if not keep_empty_features:
        X_true = X_true[:, :-1]

    imputer = SimpleImputer(
        strategy="constant", keep_empty_features=keep_empty_features
    )
    X_trans = imputer.fit_transform(df)

    assert_array_equal(X_trans, X_true)


@pytest.mark.parametrize("X", [[[1], [2]], [[1], [np.nan]]])
def test_iterative_imputer_one_feature(X):
    # [FUNCTION DOC]
    # Verifies IterativeImputer exits early (n_iter_ 0) when there is a single feature,
    # with and without missing values.
    # KNOWN ISSUE: the parametrized X is only used in the first fit; the remaining
    # assertions repeat the same fixed inputs for every parametrization.
    # check we exit early when there is a single feature
    imputer = IterativeImputer().fit(X)
    assert imputer.n_iter_ == 0
    imputer = IterativeImputer()
    imputer.fit([[1], [2]])
    assert imputer.n_iter_ == 0
    imputer.fit([[1], [np.nan]])
    assert imputer.n_iter_ == 0


def test_imputation_pipeline_grid_search():
    # [FUNCTION DOC]
    # Smoke test: SimpleImputer in a Pipeline with a decision tree is grid-searched over
    # strategies on a sparse random matrix whose first stored value is the marker.
    # Test imputation within a pipeline + gridsearch.
    X = _sparse_random_matrix(100, 100, density=0.10)
    missing_values = X.data[0]

    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(missing_values=missing_values)),
            ("tree", tree.DecisionTreeRegressor(random_state=0)),
        ]
    )

    parameters = {"imputer__strategy": ["mean", "median", "most_frequent"]}

    Y = _sparse_random_matrix(100, 1, density=0.10).toarray()
    gs = GridSearchCV(pipeline, parameters)
    gs.fit(X, Y)


def test_imputation_copy():
    # [FUNCTION DOC]
    # Verifies copy semantics of SimpleImputer: copy=True never aliases (dense, CSR);
    # copy=False aliases dense and CSC inputs (mutating output changes input) but CSR
    # input is still copied.
    # Test imputation with copy
    X_orig = _sparse_random_matrix(5, 5, density=0.75, random_state=0)

    # copy=True, dense => copy
    X = X_orig.copy().toarray()
    imputer = SimpleImputer(missing_values=0, strategy="mean", copy=True)
    Xt = imputer.fit(X).transform(X)
    Xt[0, 0] = -1
    assert not np.all(X == Xt)

    # copy=True, sparse csr => copy
    X = X_orig.copy()
    imputer = SimpleImputer(missing_values=X.data[0], strategy="mean", copy=True)
    Xt = imputer.fit(X).transform(X)
    Xt.data[0] = -1
    assert not np.all(X.data == Xt.data)

    # copy=False, dense => no copy
    X = X_orig.copy().toarray()
    imputer = SimpleImputer(missing_values=0, strategy="mean", copy=False)
    Xt = imputer.fit(X).transform(X)
    Xt[0, 0] = -1
    assert_array_almost_equal(X, Xt)

    # copy=False, sparse csc => no copy
    X = X_orig.copy().tocsc()
    imputer = SimpleImputer(missing_values=X.data[0], strategy="mean", copy=False)
    Xt = imputer.fit(X).transform(X)
    Xt.data[0] = -1
    assert_array_almost_equal(X.data, Xt.data)

    # copy=False, sparse csr => copy
    X = X_orig.copy()
    imputer = SimpleImputer(missing_values=X.data[0], strategy="mean", copy=False)
    Xt = imputer.fit(X).transform(X)
    Xt.data[0] = -1
    assert not np.all(X.data == Xt.data)

    # Note: If X is sparse and if missing_values=0, then a (dense) copy of X is
    # made, even if copy=False.


def test_iterative_imputer_zero_iters():
    # [FUNCTION DOC]
    # Verifies max_iter=0 returns the initial imputation, and that forcing n_iter_ to 0
    # on a fitted imputer makes transform equal to the initial imputation.
    rng = np.random.RandomState(0)

    n = 100
    d = 10
    X = _sparse_random_matrix(n, d, density=0.10, random_state=rng).toarray()
    missing_flag = X == 0
    X[missing_flag] = np.nan

    imputer = IterativeImputer(max_iter=0)
    X_imputed = imputer.fit_transform(X)
    # with max_iter=0, only initial imputation is performed
    assert_allclose(X_imputed, imputer.initial_imputer_.transform(X))

    # repeat but force n_iter_ to 0
    imputer = IterativeImputer(max_iter=5).fit(X)
    # transformed should not be equal to initial imputation
    assert not np.all(imputer.transform(X) == imputer.initial_imputer_.transform(X))

    imputer.n_iter_ = 0
    # now they should be equal as only initial imputation is done
    assert_allclose(imputer.transform(X), imputer.initial_imputer_.transform(X))


def test_iterative_imputer_verbose():
    # [FUNCTION DOC]
    # Smoke test that verbose levels 1 and 2 run through fit and transform.
    rng = np.random.RandomState(0)

    n = 100
    d = 3
    X = _sparse_random_matrix(n, d, density=0.10, random_state=rng).toarray()
    imputer = IterativeImputer(missing_values=0, max_iter=1, verbose=1)
    imputer.fit(X)
    imputer.transform(X)
    imputer = IterativeImputer(missing_values=0, max_iter=1, verbose=2)
    imputer.fit(X)
    imputer.transform(X)


def test_iterative_imputer_all_missing():
    # [FUNCTION DOC]
    # Verifies that when every entry is missing, the output equals the initial
    # imputation.
    n = 100
    d = 3
    X = np.zeros((n, d))
    imputer = IterativeImputer(missing_values=0, max_iter=1)
    X_imputed = imputer.fit_transform(X)
    assert_allclose(X_imputed, imputer.initial_imputer_.transform(X))


@pytest.mark.parametrize(
    "imputation_order", ["random", "roman", "ascending", "descending", "arabic"]
)
def test_iterative_imputer_imputation_order(imputation_order):
    # [FUNCTION DOC]
    # Verifies the visiting order per imputation_order with skip_complete (column 0
    # complete): number of steps per round equals n_features_with_missing_, roman visits
    # 1..9, arabic 9..1, random differs between rounds, ascending/descending perform
    # 2*(d-1) steps.
    rng = np.random.RandomState(0)
    n = 100
    d = 10
    max_iter = 2
    X = _sparse_random_matrix(n, d, density=0.10, random_state=rng).toarray()
    X[:, 0] = 1  # this column should not be discarded by IterativeImputer

    imputer = IterativeImputer(
        missing_values=0,
        max_iter=max_iter,
        n_nearest_features=5,
        sample_posterior=False,
        skip_complete=True,
        min_value=0,
        max_value=1,
        verbose=1,
        imputation_order=imputation_order,
        random_state=rng,
    )
    imputer.fit_transform(X)
    ordered_idx = [i.feat_idx for i in imputer.imputation_sequence_]

    assert len(ordered_idx) // imputer.n_iter_ == imputer.n_features_with_missing_

    if imputation_order == "roman":
        assert np.all(ordered_idx[: d - 1] == np.arange(1, d))
    elif imputation_order == "arabic":
        assert np.all(ordered_idx[: d - 1] == np.arange(d - 1, 0, -1))
    elif imputation_order == "random":
        ordered_idx_round_1 = ordered_idx[: d - 1]
        ordered_idx_round_2 = ordered_idx[d - 1 :]
        assert ordered_idx_round_1 != ordered_idx_round_2
    elif "ending" in imputation_order:
        assert len(ordered_idx) == max_iter * (d - 1)


@pytest.mark.parametrize(
    "estimator", [None, DummyRegressor(), BayesianRidge(), ARDRegression(), RidgeCV()]
)
def test_iterative_imputer_estimators(estimator):
    # [FUNCTION DOC]
    # Verifies each step in imputation_sequence_ stores a distinct estimator instance of
    # the requested type (BayesianRidge when None).
    rng = np.random.RandomState(0)

    n = 100
    d = 10
    X = _sparse_random_matrix(n, d, density=0.10, random_state=rng).toarray()

    imputer = IterativeImputer(
        missing_values=0, max_iter=1, estimator=estimator, random_state=rng
    )
    imputer.fit_transform(X)

    # check that types are correct for estimators
    hashes = []
    for triplet in imputer.imputation_sequence_:
        expected_type = (
            type(estimator) if estimator is not None else type(BayesianRidge())
        )
        assert isinstance(triplet.estimator, expected_type)
        hashes.append(id(triplet.estimator))

    # check that each estimator is unique
    assert len(set(hashes)) == len(hashes)


def test_iterative_imputer_clip():
    # [FUNCTION DOC]
    # Verifies min_value/max_value clipping: imputed entries span exactly [0.1, 0.2] and
    # observed entries are unchanged.
    rng = np.random.RandomState(0)
    n = 100
    d = 10
    X = _sparse_random_matrix(n, d, density=0.10, random_state=rng).toarray()

    imputer = IterativeImputer(
        missing_values=0, max_iter=1, min_value=0.1, max_value=0.2, random_state=rng
    )

    Xt = imputer.fit_transform(X)
    assert_allclose(np.min(Xt[X == 0]), 0.1)
    assert_allclose(np.max(Xt[X == 0]), 0.2)
    assert_allclose(Xt[X != 0], X[X != 0])


def test_iterative_imputer_clip_truncnorm():
    # [FUNCTION DOC]
    # Same as test_iterative_imputer_clip but with sample_posterior, random order and
    # n_nearest_features (truncated normal sampling path).
    rng = np.random.RandomState(0)
    n = 100
    d = 10
    X = _sparse_random_matrix(n, d, density=0.10, random_state=rng).toarray()
    X[:, 0] = 1

    imputer = IterativeImputer(
        missing_values=0,
        max_iter=2,
        n_nearest_features=5,
        sample_posterior=True,
        min_value=0.1,
        max_value=0.2,
        verbose=1,
        imputation_order="random",
        random_state=rng,
    )
    Xt = imputer.fit_transform(X)
    assert_allclose(np.min(Xt[X == 0]), 0.1)
    assert_allclose(np.max(Xt[X == 0]), 0.2)
    assert_allclose(Xt[X != 0], X[X != 0])


def test_iterative_imputer_truncated_normal_posterior():
    # [FUNCTION DOC]
    # Verifies posterior samples for a single missing value with bounds [0, 0.5] stay
    # within bounds and, after standardization, pass a Kolmogorov-Smirnov normality
    # check over 100 transforms.
    # KNOWN ISSUE: kstest is computed once before the zero-sigma guard and that result
    # is discarded; the first call is dead code (and would divide by zero if sigma were
    # 0).
    #  test that the values that are imputed using `sample_posterior=True`
    #  with boundaries (`min_value` and `max_value` are not None) are drawn
    #  from a distribution that looks gaussian via the Kolmogorov Smirnov test.
    #  note that starting from the wrong random seed will make this test fail
    #  because random sampling doesn't occur at all when the imputation
    #  is outside of the (min_value, max_value) range
    rng = np.random.RandomState(42)

    X = rng.normal(size=(5, 5))
    X[0][0] = np.nan

    imputer = IterativeImputer(
        min_value=0, max_value=0.5, sample_posterior=True, random_state=rng
    )

    imputer.fit_transform(X)
    # generate multiple imputations for the single missing value
    imputations = np.array([imputer.transform(X)[0][0] for _ in range(100)])

    assert all(imputations >= 0)
    assert all(imputations <= 0.5)

    mu, sigma = imputations.mean(), imputations.std()
    ks_statistic, p_value = kstest((imputations - mu) / sigma, "norm")
    if sigma == 0:
        sigma += 1e-12
    ks_statistic, p_value = kstest((imputations - mu) / sigma, "norm")
    # we want to fail to reject null hypothesis
    # null hypothesis: distributions are the same
    assert ks_statistic < 0.2 or p_value > 0.1, "The posterior does appear to be normal"


@pytest.mark.parametrize("strategy", ["mean", "median", "most_frequent"])
def test_iterative_imputer_missing_at_transform(strategy):
    # [FUNCTION DOC]
    # Verifies that a feature with no missing values at fit time is imputed at transform
    # only by the initial imputer (matches a SimpleImputer with the same strategy).
    rng = np.random.RandomState(0)
    n = 100
    d = 10
    X_train = rng.randint(low=0, high=3, size=(n, d))
    X_test = rng.randint(low=0, high=3, size=(n, d))

    X_train[:, 0] = 1  # definitely no missing values in 0th column
    X_test[0, 0] = 0  # definitely missing value in 0th column

    imputer = IterativeImputer(
        missing_values=0, max_iter=1, initial_strategy=strategy, random_state=rng
    ).fit(X_train)
    initial_imputer = SimpleImputer(missing_values=0, strategy=strategy).fit(X_train)

    # if there were no missing values at time of fit, then imputer will
    # only use the initial imputer for that feature at transform
    assert_allclose(
        imputer.transform(X_test)[:, 0], initial_imputer.transform(X_test)[:, 0]
    )


def test_iterative_imputer_transform_stochasticity():
    # [FUNCTION DOC]
    # Verifies transform is stochastic with sample_posterior (means differ between
    # calls) and deterministic, independent of random_state, when sampling, neighbor
    # subsampling and random ordering are all disabled.
    rng1 = np.random.RandomState(0)
    rng2 = np.random.RandomState(1)
    n = 100
    d = 10
    X = _sparse_random_matrix(n, d, density=0.10, random_state=rng1).toarray()

    # when sample_posterior=True, two transforms shouldn't be equal
    imputer = IterativeImputer(
        missing_values=0, max_iter=1, sample_posterior=True, random_state=rng1
    )
    imputer.fit(X)

    X_fitted_1 = imputer.transform(X)
    X_fitted_2 = imputer.transform(X)

    # sufficient to assert that the means are not the same
    assert np.mean(X_fitted_1) != pytest.approx(np.mean(X_fitted_2))

    # when sample_posterior=False, and n_nearest_features=None
    # and imputation_order is not random
    # the two transforms should be identical even if rng are different
    imputer1 = IterativeImputer(
        missing_values=0,
        max_iter=1,
        sample_posterior=False,
        n_nearest_features=None,
        imputation_order="ascending",
        random_state=rng1,
    )

    imputer2 = IterativeImputer(
        missing_values=0,
        max_iter=1,
        sample_posterior=False,
        n_nearest_features=None,
        imputation_order="ascending",
        random_state=rng2,
    )
    imputer1.fit(X)
    imputer2.fit(X)

    X_fitted_1a = imputer1.transform(X)
    X_fitted_1b = imputer1.transform(X)
    X_fitted_2 = imputer2.transform(X)

    assert_allclose(X_fitted_1a, X_fitted_1b)
    assert_allclose(X_fitted_1a, X_fitted_2)


def test_iterative_imputer_no_missing():
    # [FUNCTION DOC]
    # Verifies an all-NaN first column is dropped and fit().transform() equals
    # fit_transform.
    rng = np.random.RandomState(0)
    X = rng.rand(100, 100)
    X[:, 0] = np.nan
    m1 = IterativeImputer(max_iter=10, random_state=rng)
    m2 = IterativeImputer(max_iter=10, random_state=rng)
    pred1 = m1.fit(X).transform(X)
    pred2 = m2.fit_transform(X)
    # should exclude the first column entirely
    assert_allclose(X[:, 1:], pred1)
    # fit and fit_transform should both be identical
    assert_allclose(pred1, pred2)


def test_iterative_imputer_rank_one():
    # [FUNCTION DOC]
    # Verifies recovery of a rank-one matrix with half the entries missing within atol
    # 0.02.
    rng = np.random.RandomState(0)
    d = 50
    A = rng.rand(d, 1)
    B = rng.rand(1, d)
    X = np.dot(A, B)
    nan_mask = rng.rand(d, d) < 0.5
    X_missing = X.copy()
    X_missing[nan_mask] = np.nan

    imputer = IterativeImputer(max_iter=5, verbose=1, random_state=rng)
    X_filled = imputer.fit_transform(X_missing)
    assert_allclose(X_filled, X, atol=0.02)


@pytest.mark.parametrize("rank", [3, 5])
def test_iterative_imputer_transform_recovery(rank):
    # [FUNCTION DOC]
    # Verifies that an imputer fitted on half of a low-rank (3 or 5) matrix recovers
    # missing entries of the other half within atol 0.1.
    rng = np.random.RandomState(0)
    n = 70
    d = 70
    A = rng.rand(n, rank)
    B = rng.rand(rank, d)
    X_filled = np.dot(A, B)
    nan_mask = rng.rand(n, d) < 0.5
    X_missing = X_filled.copy()
    X_missing[nan_mask] = np.nan

    # split up data in half
    n = n // 2
    X_train = X_missing[:n]
    X_test_filled = X_filled[n:]
    X_test = X_missing[n:]

    imputer = IterativeImputer(
        max_iter=5, imputation_order="descending", verbose=1, random_state=rng
    ).fit(X_train)
    X_test_est = imputer.transform(X_test)
    assert_allclose(X_test_filled, X_test_est, atol=0.1)


def test_iterative_imputer_additive_matrix():
    # [FUNCTION DOC]
    # Verifies recovery of a matrix built as sums of pairs of random columns with 25%
    # missing, fit on half and transform on the other half.
    rng = np.random.RandomState(0)
    n = 100
    d = 10
    A = rng.randn(n, d)
    B = rng.randn(n, d)
    X_filled = np.zeros(A.shape)
    for i in range(d):
        for j in range(d):
            X_filled[:, (i + j) % d] += (A[:, i] + B[:, j]) / 2
    # a quarter is randomly missing
    nan_mask = rng.rand(n, d) < 0.25
    X_missing = X_filled.copy()
    X_missing[nan_mask] = np.nan

    # split up data
    n = n // 2
    X_train = X_missing[:n]
    X_test_filled = X_filled[n:]
    X_test = X_missing[n:]

    imputer = IterativeImputer(max_iter=10, verbose=1, random_state=rng).fit(X_train)
    X_test_est = imputer.transform(X_test)
    assert_allclose(X_test_filled, X_test_est, rtol=1e-3, atol=0.01)


def test_iterative_imputer_early_stopping():
    # [FUNCTION DOC]
    # Verifies early stopping: with tol 1e-2 the sequence length equals d * n_iter_;
    # rerunning with max_iter equal to that n_iter_ gives the same result; tol 0 runs
    # all max_iter rounds.
    rng = np.random.RandomState(0)
    n = 50
    d = 5
    A = rng.rand(n, 1)
    B = rng.rand(1, d)
    X = np.dot(A, B)
    nan_mask = rng.rand(n, d) < 0.5
    X_missing = X.copy()
    X_missing[nan_mask] = np.nan

    imputer = IterativeImputer(
        max_iter=100, tol=1e-2, sample_posterior=False, verbose=1, random_state=rng
    )
    X_filled_100 = imputer.fit_transform(X_missing)
    assert len(imputer.imputation_sequence_) == d * imputer.n_iter_

    imputer = IterativeImputer(
        max_iter=imputer.n_iter_, sample_posterior=False, verbose=1, random_state=rng
    )
    X_filled_early = imputer.fit_transform(X_missing)
    assert_allclose(X_filled_100, X_filled_early, atol=1e-7)

    imputer = IterativeImputer(
        max_iter=100, tol=0, sample_posterior=False, verbose=1, random_state=rng
    )
    imputer.fit(X_missing)
    assert imputer.n_iter_ == imputer.max_iter


def test_iterative_imputer_catch_warning():
    # [FUNCTION DOC]
    # Verifies that a constant feature does not cause a RuntimeWarning (warnings turned
    # into errors) when computing correlations with n_nearest_features, and output has
    # no NaN.
    # KNOWN ISSUE: the comment says it checks that a warning is caught; it actually
    # checks that no RuntimeWarning escapes.
    # check that we catch a RuntimeWarning due to a division by zero when a
    # feature is constant in the dataset
    X, y = load_diabetes(return_X_y=True)
    n_samples, n_features = X.shape

    # simulate that a feature only contain one category during fit
    X[:, 3] = 1

    # add some missing values
    rng = np.random.RandomState(0)
    missing_rate = 0.15
    for feat in range(n_features):
        sample_idx = rng.choice(
            np.arange(n_samples), size=int(n_samples * missing_rate), replace=False
        )
        X[sample_idx, feat] = np.nan

    imputer = IterativeImputer(n_nearest_features=5, sample_posterior=True)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        X_fill = imputer.fit_transform(X, y)
    assert not np.any(np.isnan(X_fill))


@pytest.mark.parametrize(
    "min_value, max_value, correct_output",
    [
        (0, 100, np.array([[0] * 3, [100] * 3])),
        (None, None, np.array([[-np.inf] * 3, [np.inf] * 3])),
        (-np.inf, np.inf, np.array([[-np.inf] * 3, [np.inf] * 3])),
        ([-5, 5, 10], [100, 200, 300], np.array([[-5, 5, 10], [100, 200, 300]])),
        (
            [-5, -np.inf, 10],
            [100, 200, np.inf],
            np.array([[-5, -np.inf, 10], [100, 200, np.inf]]),
        ),
    ],
    ids=["scalars", "None-default", "inf", "lists", "lists-with-inf"],
)
def test_iterative_imputer_min_max_array_like(min_value, max_value, correct_output):
    # [FUNCTION DOC]
    # Verifies scalar, None, infinite and list bounds are expanded to per-feature arrays
    # in _min_value/_max_value with the expected values.
    # check that passing scalar or array-like
    # for min_value and max_value in IterativeImputer works
    X = np.random.RandomState(0).randn(10, 3)
    imputer = IterativeImputer(min_value=min_value, max_value=max_value)
    imputer.fit(X)

    assert isinstance(imputer._min_value, np.ndarray) and isinstance(
        imputer._max_value, np.ndarray
    )
    assert (imputer._min_value.shape[0] == X.shape[1]) and (
        imputer._max_value.shape[0] == X.shape[1]
    )

    assert_allclose(correct_output[0, :], imputer._min_value)
    assert_allclose(correct_output[1, :], imputer._max_value)


@pytest.mark.parametrize(
    "min_value, max_value, err_msg",
    [
        (100, 0, "min_value >= max_value."),
        (np.inf, -np.inf, "min_value >= max_value."),
        ([-5, 5], [100, 200, 0], "_value' should be of shape"),
        ([-5, 5, 5], [100, 200], "_value' should be of shape"),
    ],
)
def test_iterative_imputer_catch_min_max_error(min_value, max_value, err_msg):
    # [FUNCTION DOC]
    # Verifies ValueError when min >= max and when array bounds have the wrong length.
    # KNOWN ISSUE: the comment is copied from the previous test and describes success,
    # not error cases.
    # check that passing scalar or array-like
    # for min_value and max_value in IterativeImputer works
    X = np.random.random((10, 3))
    imputer = IterativeImputer(min_value=min_value, max_value=max_value)
    with pytest.raises(ValueError, match=err_msg):
        imputer.fit(X)


@pytest.mark.parametrize(
    "min_max_1, min_max_2",
    [([None, None], [-np.inf, np.inf]), ([-10, 10], [[-10] * 4, [10] * 4])],
    ids=["None-vs-inf", "Scalar-vs-vector"],
)
def test_iterative_imputer_min_max_array_like_imputation(min_max_1, min_max_2):
    # [FUNCTION DOC]
    # Verifies None vs infinite bounds and scalar vs per-feature bounds produce the same
    # imputation.
    # KNOWN ISSUE: only column 0 of the outputs is compared.
    # Test that None/inf and scalar/vector give the same imputation
    X_train = np.array(
        [
            [np.nan, 2, 2, 1],
            [10, np.nan, np.nan, 7],
            [3, 1, np.nan, 1],
            [np.nan, 4, 2, np.nan],
        ]
    )
    X_test = np.array(
        [[np.nan, 2, np.nan, 5], [2, 4, np.nan, np.nan], [np.nan, 1, 10, 1]]
    )
    imputer1 = IterativeImputer(
        min_value=min_max_1[0], max_value=min_max_1[1], random_state=0
    )
    imputer2 = IterativeImputer(
        min_value=min_max_2[0], max_value=min_max_2[1], random_state=0
    )
    X_test_imputed1 = imputer1.fit(X_train).transform(X_test)
    X_test_imputed2 = imputer2.fit(X_train).transform(X_test)
    assert_allclose(X_test_imputed1[:, 0], X_test_imputed2[:, 0])


@pytest.mark.parametrize("skip_complete", [True, False])
def test_iterative_imputer_skip_non_missing(skip_complete):
    # [FUNCTION DOC]
    # Non-regression for #14383: a feature missing only at transform is imputed with the
    # training mean when skip_complete, and with model predictions ([11, 7, 12])
    # otherwise.
    # check the imputing strategy when missing data are present in the
    # testing set only.
    # taken from: https://github.com/scikit-learn/scikit-learn/issues/14383
    rng = np.random.RandomState(0)
    X_train = np.array([[5, 2, 2, 1], [10, 1, 2, 7], [3, 1, 1, 1], [8, 4, 2, 2]])
    X_test = np.array([[np.nan, 2, 4, 5], [np.nan, 4, 1, 2], [np.nan, 1, 10, 1]])
    imputer = IterativeImputer(
        initial_strategy="mean", skip_complete=skip_complete, random_state=rng
    )
    X_test_est = imputer.fit(X_train).transform(X_test)
    if skip_complete:
        # impute with the initial strategy: 'mean'
        assert_allclose(X_test_est[:, 0], np.mean(X_train[:, 0]))
    else:
        assert_allclose(X_test_est[:, 0], [11, 7, 12], rtol=1e-4)


@pytest.mark.parametrize("rs_imputer", [None, 1, np.random.RandomState(seed=1)])
@pytest.mark.parametrize("rs_estimator", [None, 1, np.random.RandomState(seed=1)])
def test_iterative_imputer_dont_set_random_state(rs_imputer, rs_estimator):
    # [FUNCTION DOC]
    # Intended to verify that IterativeImputer does not overwrite the random_state of
    # the user-supplied estimator.
    # KNOWN ISSUE: the ZeroEstimator is never passed to IterativeImputer, so the test
    # cannot fail; it only checks an untouched object.
    class ZeroEstimator:
        # [CLASS DOC]
        # HELPER: Minimal regressor carrying a random_state attribute and predicting
        # zeros.
        def __init__(self, random_state):
            # [FUNCTION DOC]
            # HELPER: Stores random_state.
            self.random_state = random_state

        def fit(self, *args, **kgards):
            # [FUNCTION DOC]
            # HELPER: No-op fit returning self.
            return self

        def predict(self, X):
            # [FUNCTION DOC]
            # HELPER: Returns a zero vector of length n_samples.
            return np.zeros(X.shape[0])

    estimator = ZeroEstimator(random_state=rs_estimator)
    imputer = IterativeImputer(random_state=rs_imputer)
    X_train = np.zeros((10, 3))
    imputer.fit(X_train)
    assert estimator.random_state == rs_estimator


@pytest.mark.parametrize(
    "X_fit, X_trans, params, msg_err",
    [
        (
            np.array([[-1, 1], [1, 2]]),
            np.array([[-1, 1], [1, -1]]),
            {"features": "missing-only", "sparse": "auto"},
            "have missing values in transform but have no missing values in fit",
        ),
        (
            np.array([["a", "b"], ["c", "a"]], dtype=str),
            np.array([["a", "b"], ["c", "a"]], dtype=str),
            {},
            "MissingIndicator does not support data with dtype",
        ),
    ],
)
def test_missing_indicator_error(X_fit, X_trans, params, msg_err):
    # [FUNCTION DOC]
    # Verifies MissingIndicator raises ValueError when transform data has missing values
    # in features not missing at fit (features="missing-only"), and for unsupported str
    # dtype.
    indicator = MissingIndicator(missing_values=-1)
    indicator.set_params(**params)
    with pytest.raises(ValueError, match=msg_err):
        indicator.fit(X_fit).transform(X_trans)


def _generate_missing_indicator_cases():
    # [FUNCTION DOC]
    # HELPER: Builds parametrization tuples (array constructor, missing value, dtype)
    # over dense and all sparse containers, excluding sparse containers with missing
    # value 0.
    # Outputs: list of 3-tuples.
    missing_values_dtypes = [(0, np.int32), (np.nan, np.float64), (-1, np.int32)]
    arr_types = (
        [np.array]
        + CSC_CONTAINERS
        + CSR_CONTAINERS
        + COO_CONTAINERS
        + LIL_CONTAINERS
        + BSR_CONTAINERS
    )
    return [
        (arr_type, missing_values, dtype)
        for arr_type, (missing_values, dtype) in product(
            arr_types, missing_values_dtypes
        )
        if not (missing_values == 0 and arr_type is not np.array)
    ]


@pytest.mark.parametrize(
    "arr_type, missing_values, dtype", _generate_missing_indicator_cases()
)
@pytest.mark.parametrize(
    "param_features, n_features, features_indices",
    [("missing-only", 3, np.array([0, 1, 2])), ("all", 3, np.array([0, 1, 2]))],
)
def test_missing_indicator_new(
    missing_values, arr_type, dtype, param_features, n_features, features_indices
):
    # [FUNCTION DOC]
    # Verifies MissingIndicator masks for fit and transform data across containers and
    # dtypes: shape, features_, values, bool dtype, ndarray output with sparse=False and
    # CSC output equal to the dense one with sparse=True.
    X_fit = np.array([[missing_values, missing_values, 1], [4, 2, missing_values]])
    X_trans = np.array([[missing_values, missing_values, 1], [4, 12, 10]])
    X_fit_expected = np.array([[1, 1, 0], [0, 0, 1]])
    X_trans_expected = np.array([[1, 1, 0], [0, 0, 0]])

    # convert the input to the right array format and right dtype
    X_fit = arr_type(X_fit).astype(dtype)
    X_trans = arr_type(X_trans).astype(dtype)
    X_fit_expected = X_fit_expected.astype(dtype)
    X_trans_expected = X_trans_expected.astype(dtype)

    indicator = MissingIndicator(
        missing_values=missing_values, features=param_features, sparse=False
    )
    X_fit_mask = indicator.fit_transform(X_fit)
    X_trans_mask = indicator.transform(X_trans)

    assert X_fit_mask.shape[1] == n_features
    assert X_trans_mask.shape[1] == n_features

    assert_array_equal(indicator.features_, features_indices)
    assert_allclose(X_fit_mask, X_fit_expected[:, features_indices])
    assert_allclose(X_trans_mask, X_trans_expected[:, features_indices])

    assert X_fit_mask.dtype == bool
    assert X_trans_mask.dtype == bool
    assert isinstance(X_fit_mask, np.ndarray)
    assert isinstance(X_trans_mask, np.ndarray)

    indicator.set_params(sparse=True)
    X_fit_mask_sparse = indicator.fit_transform(X_fit)
    X_trans_mask_sparse = indicator.transform(X_trans)

    assert X_fit_mask_sparse.dtype == bool
    assert X_trans_mask_sparse.dtype == bool
    assert X_fit_mask_sparse.format == "csc"
    assert X_trans_mask_sparse.format == "csc"
    assert_allclose(X_fit_mask_sparse.toarray(), X_fit_mask)
    assert_allclose(X_trans_mask_sparse.toarray(), X_trans_mask)


@pytest.mark.parametrize(
    "arr_type",
    CSC_CONTAINERS + CSR_CONTAINERS + COO_CONTAINERS + LIL_CONTAINERS + BSR_CONTAINERS,
)
def test_missing_indicator_raise_on_sparse_with_missing_0(arr_type):
    # [FUNCTION DOC]
    # Verifies MissingIndicator rejects sparse input with missing_values=0 in
    # fit_transform, and in transform after fitting on dense data.
    # test for sparse input and missing_value == 0

    missing_values = 0
    X_fit = np.array([[missing_values, missing_values, 1], [4, missing_values, 2]])
    X_trans = np.array([[missing_values, missing_values, 1], [4, 12, 10]])

    # convert the input to the right array format
    X_fit_sparse = arr_type(X_fit)
    X_trans_sparse = arr_type(X_trans)

    indicator = MissingIndicator(missing_values=missing_values)

    with pytest.raises(ValueError, match="Sparse input with missing_values=0"):
        indicator.fit_transform(X_fit_sparse)

    indicator.fit_transform(X_fit)
    with pytest.raises(ValueError, match="Sparse input with missing_values=0"):
        indicator.transform(X_trans_sparse)


@pytest.mark.parametrize("param_sparse", [True, False, "auto"])
@pytest.mark.parametrize(
    "arr_type, missing_values",
    [(np.array, 0)]
    + list(
        product(
            CSC_CONTAINERS
            + CSR_CONTAINERS
            + COO_CONTAINERS
            + LIL_CONTAINERS
            + BSR_CONTAINERS,
            [np.nan],
        )
    ),
)
def test_missing_indicator_sparse_param(arr_type, missing_values, param_sparse):
    # [FUNCTION DOC]
    # Verifies output container per sparse parameter: True gives CSC, False gives
    # ndarray, "auto" follows the input sparsity (dense for dense input with marker 0).
    # check the format of the output with different sparse parameter
    X_fit = np.array([[missing_values, missing_values, 1], [4, missing_values, 2]])
    X_trans = np.array([[missing_values, missing_values, 1], [4, 12, 10]])
    X_fit = arr_type(X_fit).astype(np.float64)
    X_trans = arr_type(X_trans).astype(np.float64)

    indicator = MissingIndicator(missing_values=missing_values, sparse=param_sparse)
    X_fit_mask = indicator.fit_transform(X_fit)
    X_trans_mask = indicator.transform(X_trans)

    if param_sparse is True:
        assert X_fit_mask.format == "csc"
        assert X_trans_mask.format == "csc"
    elif param_sparse == "auto" and missing_values == 0:
        assert isinstance(X_fit_mask, np.ndarray)
        assert isinstance(X_trans_mask, np.ndarray)
    elif param_sparse is False:
        assert isinstance(X_fit_mask, np.ndarray)
        assert isinstance(X_trans_mask, np.ndarray)
    else:
        if sparse.issparse(X_fit):
            assert X_fit_mask.format == "csc"
            assert X_trans_mask.format == "csc"
        else:
            assert isinstance(X_fit_mask, np.ndarray)
            assert isinstance(X_trans_mask, np.ndarray)


def test_missing_indicator_string():
    # [FUNCTION DOC]
    # Verifies MissingIndicator on object data with string marker "a" and
    # features="all".
    X = np.array([["a", "b", "c"], ["b", "c", "a"]], dtype=object)
    indicator = MissingIndicator(missing_values="a", features="all")
    X_trans = indicator.fit_transform(X)
    assert_array_equal(X_trans, np.array([[True, False, False], [False, False, True]]))


@pytest.mark.parametrize(
    "X, missing_values, X_trans_exp",
    [
        (
            np.array([["a", "b"], ["b", "a"]], dtype=object),
            "a",
            np.array([["b", "b", True, False], ["b", "b", False, True]], dtype=object),
        ),
        (
            np.array([[np.nan, 1.0], [1.0, np.nan]]),
            np.nan,
            np.array([[1.0, 1.0, True, False], [1.0, 1.0, False, True]]),
        ),
        (
            np.array([[np.nan, "b"], ["b", np.nan]], dtype=object),
            np.nan,
            np.array([["b", "b", True, False], ["b", "b", False, True]], dtype=object),
        ),
        (
            np.array([[None, "b"], ["b", None]], dtype=object),
            None,
            np.array([["b", "b", True, False], ["b", "b", False, True]], dtype=object),
        ),
    ],
)
def test_missing_indicator_with_imputer(X, missing_values, X_trans_exp):
    # [FUNCTION DOC]
    # Verifies a FeatureUnion of most_frequent SimpleImputer and MissingIndicator on
    # string, float, object-with-NaN and None-marker data gives the expected
    # concatenation.
    trans = make_union(
        SimpleImputer(missing_values=missing_values, strategy="most_frequent"),
        MissingIndicator(missing_values=missing_values),
    )
    X_trans = trans.fit_transform(X)
    assert_array_equal(X_trans, X_trans_exp)


@pytest.mark.parametrize("imputer_constructor", [SimpleImputer, IterativeImputer])
@pytest.mark.parametrize(
    "imputer_missing_values, missing_value, err_msg",
    [
        ("NaN", np.nan, "Input X contains NaN"),
        ("-1", -1, "types are expected to be both numerical."),
    ],
)
def test_inconsistent_dtype_X_missing_values(
    imputer_constructor, imputer_missing_values, missing_value, err_msg
):
    # [FUNCTION DOC]
    # Non-regression for #11390: string markers ("NaN", "-1") with numeric X raise a
    # clear ValueError for SimpleImputer and IterativeImputer.
    # regression test for issue #11390. Comparison between incoherent dtype
    # for X and missing_values was not raising a proper error.
    rng = np.random.RandomState(42)
    X = rng.randn(10, 10)
    X[0, 0] = missing_value

    imputer = imputer_constructor(missing_values=imputer_missing_values)

    with pytest.raises(ValueError, match=err_msg):
        imputer.fit_transform(X)


def test_missing_indicator_no_missing():
    # [FUNCTION DOC]
    # Verifies features="missing-only" returns zero columns when nothing is missing
    # (#13491).
    # check that all features are dropped if there are no missing values when
    # features='missing-only' (#13491)
    X = np.array([[1, 1], [1, 1]])

    mi = MissingIndicator(features="missing-only", missing_values=-1)
    Xt = mi.fit_transform(X)

    assert Xt.shape[1] == 0


@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_missing_indicator_sparse_no_explicit_zeros(csr_container):
    # [FUNCTION DOC]
    # Verifies sparse mask output contains no explicit zeros (nnz equals number of
    # True).
    # Check that non missing values don't become explicit zeros in the mask
    # generated by missing indicator when X is sparse. (#13491)
    X = csr_container([[0, 1, 2], [1, 2, 0], [2, 0, 1]])

    mi = MissingIndicator(features="all", missing_values=1)
    Xt = mi.fit_transform(X)

    assert Xt.nnz == Xt.sum()


@pytest.mark.parametrize("imputer_constructor", [SimpleImputer, IterativeImputer])
def test_imputer_without_indicator(imputer_constructor):
    # [FUNCTION DOC]
    # Verifies indicator_ is None when add_indicator is False.
    X = np.array([[1, 1], [1, 1]])
    imputer = imputer_constructor()
    imputer.fit(X)

    assert imputer.indicator_ is None


@pytest.mark.parametrize(
    "arr_type",
    CSC_CONTAINERS + CSR_CONTAINERS + COO_CONTAINERS + LIL_CONTAINERS + BSR_CONTAINERS,
)
def test_simple_imputation_add_indicator_sparse_matrix(arr_type):
    # [FUNCTION DOC]
    # Verifies mean imputation with add_indicator on each sparse container returns a
    # sparse result with imputed values followed by indicator columns.
    X_sparse = arr_type([[np.nan, 1, 5], [2, np.nan, 1], [6, 3, np.nan], [1, 2, 9]])
    X_true = np.array(
        [
            [3.0, 1.0, 5.0, 1.0, 0.0, 0.0],
            [2.0, 2.0, 1.0, 0.0, 1.0, 0.0],
            [6.0, 3.0, 5.0, 0.0, 0.0, 1.0],
            [1.0, 2.0, 9.0, 0.0, 0.0, 0.0],
        ]
    )

    imputer = SimpleImputer(missing_values=np.nan, add_indicator=True)
    X_trans = imputer.fit_transform(X_sparse)

    assert sparse.issparse(X_trans)
    assert X_trans.shape == X_true.shape
    assert_allclose(X_trans.toarray(), X_true)


@pytest.mark.parametrize(
    "strategy, expected", [("most_frequent", "b"), ("constant", "missing_value")]
)
def test_simple_imputation_string_list(strategy, expected):
    # [FUNCTION DOC]
    # Verifies most_frequent/constant on a list of strings with NaN gives an object
    # array with the expected fill.
    X = [["a", "b"], ["c", np.nan]]

    X_true = np.array([["a", "b"], ["c", expected]], dtype=object)

    imputer = SimpleImputer(strategy=strategy)
    X_trans = imputer.fit_transform(X)

    assert_array_equal(X_trans, X_true)


@pytest.mark.parametrize(
    "order, idx_order",
    [("ascending", [3, 4, 2, 0, 1]), ("descending", [1, 0, 2, 4, 3])],
)
def test_imputation_order(order, idx_order):
    # [FUNCTION DOC]
    # Non-regression for #15393: ascending/descending orders by missing fraction give
    # the expected feature index sequence for one round (ConvergenceWarning expected).
    # regression test for #15393
    rng = np.random.RandomState(42)
    X = rng.rand(100, 5)
    X[:50, 1] = np.nan
    X[:30, 0] = np.nan
    X[:20, 2] = np.nan
    X[:10, 4] = np.nan

    with pytest.warns(ConvergenceWarning):
        trs = IterativeImputer(max_iter=1, imputation_order=order, random_state=0).fit(
            X
        )
        idx = [x.feat_idx for x in trs.imputation_sequence_]
        assert idx == idx_order


@pytest.mark.parametrize("missing_value", [-1, np.nan])
def test_simple_imputation_inverse_transform(missing_value):
    # [FUNCTION DOC]
    # Verifies inverse_transform with add_indicator restores the original missing
    # markers on training data, new data and data with many missing entries, for -1 and
    # NaN markers.
    # Test inverse_transform feature for np.nan
    X_1 = np.array(
        [
            [9, missing_value, 3, -1],
            [4, -1, 5, 4],
            [6, 7, missing_value, -1],
            [8, 9, 0, missing_value],
        ]
    )

    X_2 = np.array(
        [
            [5, 4, 2, 1],
            [2, 1, missing_value, 3],
            [9, missing_value, 7, 1],
            [6, 4, 2, missing_value],
        ]
    )

    X_3 = np.array(
        [
            [1, missing_value, 5, 9],
            [missing_value, 4, missing_value, missing_value],
            [2, missing_value, 7, missing_value],
            [missing_value, 3, missing_value, 8],
        ]
    )

    X_4 = np.array(
        [
            [1, 1, 1, 3],
            [missing_value, 2, missing_value, 1],
            [2, 3, 3, 4],
            [missing_value, 4, missing_value, 2],
        ]
    )

    imputer = SimpleImputer(
        missing_values=missing_value, strategy="mean", add_indicator=True
    )

    X_1_trans = imputer.fit_transform(X_1)
    X_1_inv_trans = imputer.inverse_transform(X_1_trans)

    X_2_trans = imputer.transform(X_2)  # test on new data
    X_2_inv_trans = imputer.inverse_transform(X_2_trans)

    assert_array_equal(X_1_inv_trans, X_1)
    assert_array_equal(X_2_inv_trans, X_2)

    for X in [X_3, X_4]:
        X_trans = imputer.fit_transform(X)
        X_inv_trans = imputer.inverse_transform(X_trans)
        assert_array_equal(X_inv_trans, X)


@pytest.mark.parametrize("missing_value", [-1, np.nan])
def test_simple_imputation_inverse_transform_exceptions(missing_value):
    # [FUNCTION DOC]
    # Verifies inverse_transform raises ValueError mentioning add_indicator when it is
    # False.
    X_1 = np.array(
        [
            [9, missing_value, 3, -1],
            [4, -1, 5, 4],
            [6, 7, missing_value, -1],
            [8, 9, 0, missing_value],
        ]
    )

    imputer = SimpleImputer(missing_values=missing_value, strategy="mean")
    X_1_trans = imputer.fit_transform(X_1)
    with pytest.raises(
        ValueError, match=f"Got 'add_indicator={imputer.add_indicator}'"
    ):
        imputer.inverse_transform(X_1_trans)


@pytest.mark.parametrize(
    "expected,array,dtype,extra_value,n_repeat",
    [
        # array of object dtype
        ("extra_value", ["a", "b", "c"], object, "extra_value", 2),
        (
            "most_frequent_value",
            ["most_frequent_value", "most_frequent_value", "value"],
            object,
            "extra_value",
            1,
        ),
        ("a", ["min_value", "min_valuevalue"], object, "a", 2),
        ("min_value", ["min_value", "min_value", "value"], object, "z", 2),
        # array of numeric dtype
        (10, [1, 2, 3], int, 10, 2),
        (1, [1, 1, 2], int, 10, 1),
        (10, [20, 20, 1], int, 10, 2),
        (1, [1, 1, 20], int, 10, 2),
    ],
)
def test_most_frequent(expected, array, dtype, extra_value, n_repeat):
    # [FUNCTION DOC]
    # Verifies _most_frequent for object and numeric arrays combined with an extra value
    # repeated n_repeat times, including ties (smallest value wins).
    assert expected == _most_frequent(
        np.array(array, dtype=dtype), extra_value, n_repeat
    )


@pytest.mark.parametrize(
    "expected,array",
    [
        ("a", ["a", "b"]),
        (1, [1, 2]),
        (None, [None, "a"]),
        (None, [None, 1]),
        (None, [None, "a", 1]),
        (1, [1, "1"]),
        (1, ["1", 1]),
    ],
)
def test_most_frequent_tie_object(expected, array):
    # [FUNCTION DOC]
    # Non-regression for #31717: tie breaking in _most_frequent on mixed-type object
    # arrays returns the first value in the array.
    """Check the tie breaking behavior of the most frequent strategy.

    Non-regression test for issue #31717.
    """
    assert expected == _most_frequent(np.array(array, dtype=object), None, 0)


@pytest.mark.parametrize(
    "initial_strategy", ["mean", "median", "most_frequent", "constant"]
)
def test_iterative_imputer_keep_empty_features(initial_strategy):
    # [FUNCTION DOC]
    # Verifies an all-missing feature is imputed with 0 for every initial_strategy when
    # keep_empty_features, in fit_transform and transform.
    """Check the behaviour of the iterative imputer with different initial strategy
    and keeping empty features (i.e. features containing only missing values).
    """
    X = np.array([[1, np.nan, 2], [3, np.nan, np.nan]])

    imputer = IterativeImputer(
        initial_strategy=initial_strategy, keep_empty_features=True
    )
    X_imputed = imputer.fit_transform(X)
    assert_allclose(X_imputed[:, 1], 0)
    X_imputed = imputer.transform(X)
    assert_allclose(X_imputed[:, 1], 0)


@pytest.mark.parametrize("keep_empty_features", [True, False])
def test_iterative_imputer_constant_fill_value(keep_empty_features):
    # [FUNCTION DOC]
    # Verifies fill_value is propagated to the initial SimpleImputer: statistics_ equal
    # 100 for all columns when keeping empty features, else NaN for the empty column.
    """Check that we propagate properly the parameter `fill_value`."""
    X = np.array([[-1, 2, 3, -1], [4, -1, 5, -1], [6, 7, -1, -1], [8, 9, 0, -1]])

    fill_value = 100
    imputer = IterativeImputer(
        missing_values=-1,
        initial_strategy="constant",
        fill_value=fill_value,
        max_iter=0,
        keep_empty_features=keep_empty_features,
    )
    imputer.fit_transform(X)

    if keep_empty_features:
        assert_array_equal(imputer.initial_imputer_.statistics_, fill_value)
    else:
        assert_array_equal(imputer.initial_imputer_.statistics_[:-1], fill_value)
        assert np.isnan(imputer.initial_imputer_.statistics_[-1])


def test_iterative_imputer_min_max_value_remove_empty():
    # [FUNCTION DOC]
    # Non-regression for #29355: per-feature min/max arrays are correctly realigned
    # after an empty column is dropped, checking imputed values hit the expected bounds.
    """Check that we properly apply the empty feature mask to `min_value` and
    `max_value`.

    Non-regression test for https://github.com/scikit-learn/scikit-learn/issues/29355
    """
    # Intentionally make column 2 as a missing column, then the bound of the imputed
    # value of column 3 should be (4, 5)
    X = np.array(
        [
            [1, 2, np.nan, np.nan],
            [4, 5, np.nan, 6],
            [7, 8, np.nan, np.nan],
            [10, 11, np.nan, 12],
        ]
    )
    min_value = [-np.inf, -np.inf, -np.inf, 4]
    max_value = [np.inf, np.inf, np.inf, 5]

    X_imputed = IterativeImputer(
        min_value=min_value,
        max_value=max_value,
        keep_empty_features=False,
    ).fit_transform(X)

    X_without_missing_column = np.delete(X, 2, axis=1)
    assert X_imputed.shape == X_without_missing_column.shape
    assert np.min(X_imputed[np.isnan(X_without_missing_column)]) == pytest.approx(4)
    assert np.max(X_imputed[np.isnan(X_without_missing_column)]) == pytest.approx(5)

    # Intentionally make column 3 as a missing column, then the bound of the imputed
    # value of column 2 should be (3.5, 6)
    X = np.array(
        [
            [1, 2, np.nan, np.nan],
            [4, 5, 6, np.nan],
            [7, 8, np.nan, np.nan],
            [10, 11, 12, np.nan],
        ]
    )
    min_value = [-np.inf, -np.inf, 3.5, -np.inf]
    max_value = [np.inf, np.inf, 6, np.inf]

    X_imputed = IterativeImputer(
        min_value=min_value,
        max_value=max_value,
        keep_empty_features=False,
    ).fit_transform(X)

    X_without_missing_column = X[:, :3]
    assert X_imputed.shape == X_without_missing_column.shape
    assert np.min(X_imputed[np.isnan(X_without_missing_column)]) == pytest.approx(3.5)
    assert np.max(X_imputed[np.isnan(X_without_missing_column)]) == pytest.approx(6)


@pytest.mark.parametrize("keep_empty_features", [True, False])
def test_knn_imputer_keep_empty_features(keep_empty_features):
    # [FUNCTION DOC]
    # Verifies KNNImputer keeps the empty column filled with 0 when keep_empty_features,
    # and drops it otherwise, for fit_transform and transform.
    """Check the behaviour of `keep_empty_features` for `KNNImputer`."""
    X = np.array([[1, np.nan, 2], [3, np.nan, np.nan]])

    imputer = KNNImputer(keep_empty_features=keep_empty_features)

    for method in ["fit_transform", "transform"]:
        X_imputed = getattr(imputer, method)(X)
        if keep_empty_features:
            assert X_imputed.shape == X.shape
            assert_array_equal(X_imputed[:, 1], 0)
        else:
            assert X_imputed.shape == (X.shape[0], X.shape[1] - 1)


def test_simple_impute_pd_na():
    # [FUNCTION DOC]
    # Verifies SimpleImputer with pd.NA marker on pandas string, Int64 and float64
    # columns (constant, mean, median), checking values and output dtype (object or
    # float64).
    pd = pytest.importorskip("pandas")

    # Impute pandas array of string types.
    df = pd.DataFrame({"feature": pd.Series(["abc", None, "de"], dtype="string")})
    imputer = SimpleImputer(missing_values=pd.NA, strategy="constant", fill_value="na")
    _assert_array_equal_and_same_dtype(
        imputer.fit_transform(df), np.array([["abc"], ["na"], ["de"]], dtype=object)
    )

    # Impute pandas array of string types without any missing values.
    df = pd.DataFrame({"feature": pd.Series(["abc", "de", "fgh"], dtype="string")})
    imputer = SimpleImputer(fill_value="ok", strategy="constant")
    _assert_array_equal_and_same_dtype(
        imputer.fit_transform(df), np.array([["abc"], ["de"], ["fgh"]], dtype=object)
    )

    # Impute pandas array of integer types.
    df = pd.DataFrame({"feature": pd.Series([1, None, 3], dtype="Int64")})
    imputer = SimpleImputer(missing_values=pd.NA, strategy="constant", fill_value=-1)
    _assert_allclose_and_same_dtype(
        imputer.fit_transform(df), np.array([[1], [-1], [3]], dtype="float64")
    )

    # Use `np.nan` also works.
    imputer = SimpleImputer(missing_values=np.nan, strategy="constant", fill_value=-1)
    _assert_allclose_and_same_dtype(
        imputer.fit_transform(df), np.array([[1], [-1], [3]], dtype="float64")
    )

    # Impute pandas array of integer types with 'median' strategy.
    df = pd.DataFrame({"feature": pd.Series([1, None, 2, 3], dtype="Int64")})
    imputer = SimpleImputer(missing_values=pd.NA, strategy="median")
    _assert_allclose_and_same_dtype(
        imputer.fit_transform(df), np.array([[1], [2], [2], [3]], dtype="float64")
    )

    # Impute pandas array of integer types with 'mean' strategy.
    df = pd.DataFrame({"feature": pd.Series([1, None, 2], dtype="Int64")})
    imputer = SimpleImputer(missing_values=pd.NA, strategy="mean")
    _assert_allclose_and_same_dtype(
        imputer.fit_transform(df), np.array([[1], [1.5], [2]], dtype="float64")
    )

    # Impute pandas array of float types.
    df = pd.DataFrame({"feature": pd.Series([1.0, None, 3.0], dtype="float64")})
    imputer = SimpleImputer(missing_values=pd.NA, strategy="constant", fill_value=-2.0)
    _assert_allclose_and_same_dtype(
        imputer.fit_transform(df), np.array([[1.0], [-2.0], [3.0]], dtype="float64")
    )

    # Impute pandas array of float types with 'median' strategy.
    df = pd.DataFrame({"feature": pd.Series([1.0, None, 2.0, 3.0], dtype="float64")})
    imputer = SimpleImputer(missing_values=pd.NA, strategy="median")
    _assert_allclose_and_same_dtype(
        imputer.fit_transform(df),
        np.array([[1.0], [2.0], [2.0], [3.0]], dtype="float64"),
    )


def test_missing_indicator_feature_names_out():
    # [FUNCTION DOC]
    # Verifies MissingIndicator feature names are prefixed "missingindicator_" for
    # columns with missing values.
    """Check that missing indicator return the feature names with a prefix."""
    pd = pytest.importorskip("pandas")

    missing_values = np.nan
    X = pd.DataFrame(
        [
            [missing_values, missing_values, 1, missing_values],
            [4, missing_values, 2, 10],
        ],
        columns=["a", "b", "c", "d"],
    )

    indicator = MissingIndicator(missing_values=missing_values).fit(X)
    feature_names = indicator.get_feature_names_out()
    expected_names = ["missingindicator_a", "missingindicator_b", "missingindicator_d"]
    assert_array_equal(expected_names, feature_names)


def test_imputer_lists_fit_transform():
    # [FUNCTION DOC]
    # Non-regression for #19572: fitted on string lists, transform of an all-NaN row
    # returns object dtype with the modal values.
    """Check transform uses object dtype when fitted on an object dtype.

    Non-regression test for #19572.
    """

    X = [["a", "b"], ["c", "b"], ["a", "a"]]
    imp_frequent = SimpleImputer(strategy="most_frequent").fit(X)
    X_trans = imp_frequent.transform([[np.nan, np.nan]])
    assert X_trans.dtype == object
    assert_array_equal(X_trans, [["a", "b"]])


@pytest.mark.parametrize("dtype_test", [np.float32, np.float64])
def test_imputer_transform_preserves_numeric_dtype(dtype_test):
    # [FUNCTION DOC]
    # Verifies transform output keeps the input float32/float64 dtype regardless of fit
    # dtype.
    """Check transform preserves numeric dtype independent of fit dtype."""
    X = np.asarray(
        [[1.2, 3.4, np.nan], [np.nan, 1.2, 1.3], [4.2, 2, 1]], dtype=np.float64
    )
    imp = SimpleImputer().fit(X)

    X_test = np.asarray([[np.nan, np.nan, np.nan]], dtype=dtype_test)
    X_trans = imp.transform(X_test)
    assert X_trans.dtype == dtype_test


@pytest.mark.parametrize("array_type", ["array", "sparse"])
@pytest.mark.parametrize("strategy", ["mean", "median", "most_frequent"])
@pytest.mark.parametrize("keep_empty_features", [True, False])
def test_simple_imputer_keep_empty_features(strategy, array_type, keep_empty_features):
    # [FUNCTION DOC]
    # Verifies keep_empty_features for mean/median/most_frequent on dense and sparse
    # inputs: shape kept and empty column filled with 0, or dropped (handles older scipy
    # indexing).
    """Check the behaviour of `keep_empty_features` with all strategies but
    'constant'.
    """
    X = np.array([[np.nan, 2], [np.nan, 3], [np.nan, 6]])
    X = _convert_container(X, array_type)
    imputer = SimpleImputer(strategy=strategy, keep_empty_features=keep_empty_features)

    for method in ["fit_transform", "transform"]:
        X_imputed = getattr(imputer, method)(X)
        if keep_empty_features:
            assert X_imputed.shape == X.shape
            if SCIPY_VERSION_BELOW_1_12 and array_type == "sparse":
                constant_feature = X_imputed[:, [0]].toarray()
            else:
                col0 = X_imputed[:, 0]
                constant_feature = col0.toarray() if array_type == "sparse" else col0
            assert_array_equal(constant_feature, 0)
        else:
            assert X_imputed.shape == (X.shape[0], X.shape[1] - 1)


@pytest.mark.parametrize("csc_container", CSC_CONTAINERS)
def test_imputation_custom(csc_container):
    # [FUNCTION DOC]
    # Verifies a callable strategy (np.min) imputes column minima, dense and CSC.
    X = np.array(
        [
            [1.1, 1.1, 1.1],
            [3.9, 1.2, np.nan],
            [np.nan, 1.3, np.nan],
            [0.1, 1.4, 1.4],
            [4.9, 1.5, 1.5],
            [np.nan, 1.6, 1.6],
        ]
    )

    X_true = np.array(
        [
            [1.1, 1.1, 1.1],
            [3.9, 1.2, 1.1],
            [0.1, 1.3, 1.1],
            [0.1, 1.4, 1.4],
            [4.9, 1.5, 1.5],
            [0.1, 1.6, 1.6],
        ]
    )

    imputer = SimpleImputer(missing_values=np.nan, strategy=np.min)
    X_trans = imputer.fit_transform(X)
    assert_array_equal(X_trans, X_true)

    # Sparse matrix
    imputer = SimpleImputer(missing_values=np.nan, strategy=np.min)
    X_trans = imputer.fit_transform(csc_container(X))
    assert_array_equal(X_trans.toarray(), X_true)


def test_simple_imputer_constant_fill_value_casting():
    # [FUNCTION DOC]
    # Non-regression for #28309: fill_value incompatible with integer data raises at fit
    # and at transform with clear messages; compatible float fill values keep float32
    # output.
    """Check that we raise a proper error message when we cannot cast the fill value
    to the input data type. Otherwise, check that the casting is done properly.

    Non-regression test for:
    https://github.com/scikit-learn/scikit-learn/issues/28309
    """
    # cannot cast fill_value at fit
    fill_value = 1.5
    X_int64 = np.array([[1, 2, 3], [2, 3, 4]], dtype=np.int64)
    imputer = SimpleImputer(
        strategy="constant", fill_value=fill_value, missing_values=2
    )
    err_msg = f"fill_value={fill_value!r} (of type {type(fill_value)!r}) cannot be cast"
    with pytest.raises(ValueError, match=re.escape(err_msg)):
        imputer.fit(X_int64)

    # cannot cast fill_value at transform
    X_float64 = np.array([[1, 2, 3], [2, 3, 4]], dtype=np.float64)
    imputer.fit(X_float64)
    err_msg = (
        f"The dtype of the filling value (i.e. {imputer._fill_dtype!r}) cannot be cast"
    )
    with pytest.raises(ValueError, match=re.escape(err_msg)):
        imputer.transform(X_int64)

    # check that no error is raised when having the same kind of dtype
    fill_value_list = [np.float64(1.5), 1.5, 1]
    X_float32 = X_float64.astype(np.float32)

    for fill_value in fill_value_list:
        imputer = SimpleImputer(
            strategy="constant", fill_value=fill_value, missing_values=2
        )
        X_trans = imputer.fit_transform(X_float32)
        assert X_trans.dtype == X_float32.dtype


@pytest.mark.parametrize("strategy", ["mean", "median", "most_frequent", "constant"])
def test_iterative_imputer_no_empty_features(strategy):
    # [FUNCTION DOC]
    # Non-regression for #29375: without empty features, keep_empty_features True/False
    # give identical IterativeImputer output.
    """Check the behaviour of `keep_empty_features` with no empty features.

    With no-empty features, we should get the same imputation whatever the
    parameter `keep_empty_features`.

    Non-regression test for:
    https://github.com/scikit-learn/scikit-learn/issues/29375
    """
    X = np.array([[np.nan, 0, 1], [2, np.nan, 3], [4, 5, np.nan]])

    imputer_drop_empty_features = IterativeImputer(
        initial_strategy=strategy, fill_value=1, keep_empty_features=False
    )

    imputer_keep_empty_features = IterativeImputer(
        initial_strategy=strategy, fill_value=1, keep_empty_features=True
    )

    assert_allclose(
        imputer_drop_empty_features.fit_transform(X),
        imputer_keep_empty_features.fit_transform(X),
    )


@pytest.mark.parametrize("strategy", ["mean", "median", "most_frequent", "constant"])
@pytest.mark.parametrize(
    "X_test",
    [
        np.array([[1, 2, 3, 4], [5, 6, 7, 8]]),  # without empty feature
        np.array([[np.nan, 2, 3, 4], [np.nan, 6, 7, 8]]),  # empty feature at column 0
        np.array([[1, 2, 3, np.nan], [5, 6, 7, np.nan]]),  # empty feature at column 3
    ],
)
def test_iterative_imputer_with_empty_features(strategy, X_test):
    # [FUNCTION DOC]
    # Non-regression for #29375: with an empty training column, keep_empty_features
    # keeps it filled with 0 and the rest matches the dropping imputer; transform output
    # widths match training widths for test data with or without empty columns.
    """Check the behaviour of `keep_empty_features` in the presence of empty features.

    With `keep_empty_features=True`, the empty feature will be imputed with the value
    defined by the initial imputation.

    Non-regression test for:
    https://github.com/scikit-learn/scikit-learn/issues/29375
    """
    X_train = np.array(
        [[np.nan, np.nan, 0, 1], [np.nan, 2, np.nan, 3], [np.nan, 4, 5, np.nan]]
    )

    imputer_drop_empty_features = IterativeImputer(
        initial_strategy=strategy, fill_value=0, keep_empty_features=False
    )
    X_train_drop_empty_features = imputer_drop_empty_features.fit_transform(X_train)
    X_test_drop_empty_features = imputer_drop_empty_features.transform(X_test)

    imputer_keep_empty_features = IterativeImputer(
        initial_strategy=strategy, fill_value=0, keep_empty_features=True
    )
    X_train_keep_empty_features = imputer_keep_empty_features.fit_transform(X_train)
    X_test_keep_empty_features = imputer_keep_empty_features.transform(X_test)

    assert_allclose(X_train_drop_empty_features, X_train_keep_empty_features[:, 1:])
    assert_allclose(X_train_keep_empty_features[:, 0], 0)

    assert X_train_drop_empty_features.shape[1] == X_test_drop_empty_features.shape[1]
    assert X_train_keep_empty_features.shape[1] == X_test_keep_empty_features.shape[1]
