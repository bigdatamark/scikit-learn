# [MODULE DOC]
# Tests for sklearn.utils.validation: check_array/check_X_y conversion and error paths
# (dtype, order, copy, sparse formats, pandas/narwhals dataframes, NaN/inf handling),
# sample/feature counting, fitted-state checks, scalar checks, PSD eigenvalue checks,
# sample_weight handling, feature-name validation, categorical/monotonic helpers and
# Array API variants. Mostly uses pytest parametrization and pytest.raises/warns.
"""Tests for input validation functions"""

import numbers
import re
import warnings
from itertools import product
from operator import itemgetter
from tempfile import NamedTemporaryFile

import narwhals.stable.v2 as nw
import numpy as np
import pytest
import scipy.sparse as sp

import sklearn
from sklearn._config import config_context
from sklearn.base import BaseEstimator
from sklearn.datasets import make_blobs
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import NotFittedError, PositiveSpectrumWarning
from sklearn.linear_model import ARDRegression

# TODO: add this estimator into the _mocking module in a further refactoring
from sklearn.metrics.tests.test_score_objects import EstimatorWithFit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.random_projection import _sparse_random_matrix
from sklearn.svm import SVR
from sklearn.utils import (
    _safe_indexing,
    as_float_array,
    check_array,
    check_symmetric,
    check_X_y,
    deprecated,
)
from sklearn.utils._array_api import (
    _is_numpy_namespace,
    move_to,
    yield_namespace_device_dtype_combinations,
)
from sklearn.utils._mocking import (
    MockDataFrame,
    _MockEstimatorOnOffPrediction,
)
from sklearn.utils._testing import (
    TempMemmap,
    _array_api_for_tests,
    _convert_container,
    assert_allclose,
    assert_allclose_dense_sparse,
    assert_array_equal,
    create_memmap_backed_data,
    skip_if_array_api_compat_not_configured,
)
from sklearn.utils.estimator_checks import _NotAnArray
from sklearn.utils.fixes import (
    COO_CONTAINERS,
    CSC_CONTAINERS,
    CSR_CONTAINERS,
    DIA_CONTAINERS,
    DOK_CONTAINERS,
    _sparse_random_array,
)
from sklearn.utils.validation import (
    FLOAT_DTYPES,
    _allclose_dense_sparse,
    _check_categorical_features,
    _check_feature_names_in,
    _check_method_params,
    _check_pos_label_consistency,
    _check_psd_eigenvalues,
    _check_response_method,
    _check_sample_weight,
    _check_y,
    _deprecate_positional_args,
    _estimator_has,
    _get_feature_names,
    _is_fitted,
    _num_features,
    _num_samples,
    _to_object_array,
    assert_all_finite,
    check_consistent_length,
    check_is_fitted,
    check_memory,
    check_non_negative,
    check_random_state,
    check_scalar,
    column_or_1d,
    has_fit_parameter,
    validate_data,
)


def test_make_rng():
    # [FUNCTION DOC]
    # Verifies check_random_state: None and np.random map to the global RandomState,
    # ints seed a new RandomState (draws compared to an equally seeded one), instances
    # are returned as is, and an invalid seed raises ValueError.
    # Check the check_random_state utility function behavior
    assert check_random_state(None) is np.random.mtrand._rand
    assert check_random_state(np.random) is np.random.mtrand._rand

    rng_42 = np.random.RandomState(42)
    assert check_random_state(42).randint(100) == rng_42.randint(100)

    rng_42 = np.random.RandomState(42)
    assert check_random_state(rng_42) is rng_42

    rng_42 = np.random.RandomState(42)
    assert check_random_state(43).randint(100) != rng_42.randint(100)

    with pytest.raises(ValueError):
        check_random_state("some invalid seed")


def test_as_float_array():
    # [FUNCTION DOC]
    # Verifies as_float_array dtype selection: int32 and small int/bool dtypes give
    # float32, int64 and object give float64, a copy is made for int64 even with
    # copy=False, float32 input is returned as is with copy=False, Fortran order
    # survives a copy, and copies of dense matrices are independent of the source.
    # Test function for as_float_array
    X = np.ones((3, 10), dtype=np.int32)
    X = X + np.arange(10, dtype=np.int32)
    X2 = as_float_array(X, copy=False)
    assert X2.dtype == np.float32
    # Another test
    X = X.astype(np.int64)
    X2 = as_float_array(X, copy=True)
    # Checking that the array wasn't overwritten
    assert as_float_array(X, copy=False) is not X
    assert X2.dtype == np.float64
    # Test int dtypes <= 32bit
    tested_dtypes = [bool, np.int8, np.int16, np.int32, np.uint8, np.uint16, np.uint32]
    for dtype in tested_dtypes:
        X = X.astype(dtype)
        X2 = as_float_array(X)
        assert X2.dtype == np.float32

    # Test object dtype
    X = X.astype(object)
    X2 = as_float_array(X, copy=True)
    assert X2.dtype == np.float64

    # Here, X is of the right type, it shouldn't be modified
    X = np.ones((3, 2), dtype=np.float32)
    assert as_float_array(X, copy=False) is X
    # Test that if X is fortran ordered it stays
    X = np.asfortranarray(X)
    assert np.isfortran(as_float_array(X, copy=True))

    # Test the copy parameter with some matrices
    matrices = [
        sp.csc_matrix(np.arange(5)).toarray(),
        sp.csc_array([np.arange(5)]).toarray(),
        _sparse_random_matrix(10, 10, density=0.10).toarray(),
    ]
    for M in matrices:
        N = as_float_array(M, copy=True)
        N[0, 0] = np.nan
        assert not np.isnan(M).any()


@pytest.mark.filterwarnings("ignore::scipy.sparse.SparseEfficiencyWarning")
@pytest.mark.parametrize(
    "X", [np.random.random((10, 2)), _sparse_random_array((10, 2), format="csr")]
)
def test_as_float_array_nan(X):
    # [FUNCTION DOC]
    # Verifies as_float_array with ensure_all_finite="allow-nan" keeps NaNs for dense
    # and CSR inputs (result equal to input). Inputs: X - parametrized dense/sparse
    # array.
    X = X.copy()
    X[5, 0] = np.nan
    X[6, 1] = np.nan
    X_converted = as_float_array(X, ensure_all_finite="allow-nan")
    assert_allclose_dense_sparse(X_converted, X)


def test_np_matrix():
    # [FUNCTION DOC]
    # Verifies as_float_array never returns np.matrix for ndarray, csc_matrix or
    # csc_array inputs.
    # Confirm that input validation code does not return np.matrix
    X = np.arange(12).reshape(3, 4)

    assert not isinstance(as_float_array(X), np.matrix)
    assert not isinstance(as_float_array(sp.csc_matrix(X)), np.matrix)
    assert not isinstance(as_float_array(sp.csc_array(X)), np.matrix)


def test_memmap():
    # Confirm that input validation code doesn't copy memory mapped arrays

    # [FUNCTION DOC]
    # Verifies check_array, np.asarray and as_float_array(copy=False) do not copy a
    # float32 memmap: writes through the result are visible in the memmap.
    # Side effects: creates a temporary file.
    asflt = lambda x: as_float_array(x, copy=False)

    with NamedTemporaryFile(prefix="sklearn-test") as tmp:
        M = np.memmap(tmp, shape=(10, 10), dtype=np.float32)
        M[:] = 0

        for f in (check_array, np.asarray, asflt):
            X = f(M)
            X[:] = 1
            assert_array_equal(X.ravel(), M.ravel())
            X[:] = 0


def test_ordering():
    # [FUNCTION DOC]
    # Verifies check_array enforces C/F order for a matrix and its transpose, with and
    # without copy, and returns a new object when copy=True.
    # KNOWN ISSUE: the trailing sparse block only builds a CSR with non-contiguous data
    # and asserts that precondition; it never calls a validation function, so it tests
    # nothing.
    # Check that ordering is enforced correctly by validation utilities.
    # We need to check each validation utility, because a 'copy' without
    # 'order=K' will kill the ordering.
    X = np.ones((10, 5))
    for A in X, X.T:
        for copy in (True, False):
            B = check_array(A, order="C", copy=copy)
            assert B.flags["C_CONTIGUOUS"]
            B = check_array(A, order="F", copy=copy)
            assert B.flags["F_CONTIGUOUS"]
            if copy:
                assert A is not B

    X = sp.csr_array(X)
    X.data = X.data[::-1]
    assert not X.data.flags["C_CONTIGUOUS"]


@pytest.mark.parametrize(
    "value, ensure_all_finite",
    [(np.inf, False), (np.nan, "allow-nan"), (np.nan, False)],
)
@pytest.mark.parametrize("retype", [np.asarray, sp.csr_array, sp.csr_matrix])
def test_check_array_ensure_all_finite_valid(value, ensure_all_finite, retype):
    # [FUNCTION DOC]
    # Verifies check_array accepts inf with ensure_all_finite=False and NaN with
    # "allow-nan"/False, for dense, csr_array and csr_matrix, returning equal data.
    X = np.array(np.arange(4).reshape(2, 2).astype(float))
    X[0, 0] = value
    X = retype(X)
    X_checked = check_array(X, ensure_all_finite=ensure_all_finite, accept_sparse=True)
    assert_allclose_dense_sparse(X, X_checked)


@pytest.mark.parametrize(
    "value, input_name, ensure_all_finite, match_msg",
    [
        (np.inf, "", True, "Input contains infinity"),
        (np.inf, "X", True, "Input X contains infinity"),
        (np.inf, "sample_weight", True, "Input sample_weight contains infinity"),
        (np.inf, "X", "allow-nan", "Input X contains infinity"),
        (np.nan, "", True, "Input contains NaN"),
        (np.nan, "X", True, "Input X contains NaN"),
        (np.nan, "y", True, "Input y contains NaN"),
        (
            np.nan,
            "",
            "allow-inf",
            "ensure_all_finite should be a bool or 'allow-nan'",
        ),
        (np.nan, "", 1, "Input contains NaN"),
    ],
)
@pytest.mark.parametrize("retype", [np.asarray, sp.csr_array, sp.csr_matrix])
def test_check_array_ensure_all_finite_invalid(
    value, input_name, ensure_all_finite, match_msg, retype
):
    # [FUNCTION DOC]
    # Verifies check_array error messages for inf/NaN, including the input_name in the
    # message, rejection of an invalid ensure_all_finite string, and that a truthy
    # non-bool (1) behaves like True; for dense and CSR inputs.
    X = np.array(np.arange(4).reshape(2, 2).astype(np.float64))
    X[0, 0] = value
    X = retype(X)
    with pytest.raises(ValueError, match=match_msg):
        check_array(
            X,
            input_name=input_name,
            ensure_all_finite=ensure_all_finite,
            accept_sparse=True,
        )


@pytest.mark.parametrize("input_name", ["X", "y", "sample_weight"])
@pytest.mark.parametrize("retype", [np.asarray, sp.csr_array, sp.csr_matrix])
def test_check_array_links_to_imputer_doc_only_for_X(input_name, retype):
    # [FUNCTION DOC]
    # Verifies the extended NaN message pointing to imputation docs is appended only
    # when input_name is "X" (with an estimator), for dense and sparse inputs, and that
    # SVR.fit produces the same extended message.
    data = np.array(np.arange(4).reshape(2, 2).astype(np.float64))
    data[0, 0] = np.nan
    data = retype(data)
    estimator = SVR()
    extended_msg = (
        f"\n{estimator.__class__.__name__} does not accept missing values"
        " encoded as NaN natively. For supervised learning, you might want"
        " to consider sklearn.ensemble.HistGradientBoostingClassifier and Regressor"
        " which accept missing values encoded as NaNs natively."
        " Alternatively, it is possible to preprocess the"
        " data, for instance by using an imputer transformer in a pipeline"
        " or drop samples with missing values. See"
        " https://scikit-learn.org/stable/modules/impute.html"
        " You can find a list of all estimators that handle NaN values"
        " at the following page:"
        " https://scikit-learn.org/stable/modules/impute.html"
        "#estimators-that-handle-nan-values"
    )

    with pytest.raises(ValueError, match=f"Input {input_name} contains NaN") as ctx:
        check_array(
            data,
            estimator=estimator,
            input_name=input_name,
            accept_sparse=True,
        )

    if input_name == "X":
        assert extended_msg in ctx.value.args[0]
    else:
        assert extended_msg not in ctx.value.args[0]

    if input_name == "X":
        # Verify that _validate_data is automatically called with the right argument
        # to generate the same exception:
        with pytest.raises(ValueError, match=f"Input {input_name} contains NaN") as ctx:
            SVR().fit(data, np.ones(data.shape[0]))
        assert extended_msg in ctx.value.args[0]


def test_check_array_ensure_all_finite_object():
    # [FUNCTION DOC]
    # Verifies object-dtype arrays containing NaN pass unchanged (same object) with
    # "allow-nan" or False and raise "Input contains NaN" with True.
    X = np.array([["a", "b", np.nan]], dtype=object).T

    X_checked = check_array(X, dtype=None, ensure_all_finite="allow-nan")
    assert X is X_checked

    X_checked = check_array(X, dtype=None, ensure_all_finite=False)
    assert X is X_checked

    with pytest.raises(ValueError, match="Input contains NaN"):
        check_array(X, dtype=None, ensure_all_finite=True)


@pytest.mark.parametrize(
    "X, err_msg",
    [
        (
            np.array([[1, np.nan]]),
            "Input contains NaN.",
        ),
        (
            np.array([[1, np.nan]]),
            "Input contains NaN.",
        ),
        (
            np.array([[1, np.inf]]),
            "Input contains infinity or a value too large for.*int",
        ),
        (np.array([[1, np.nan]], dtype=object), "cannot convert float NaN to integer"),
    ],
)
@pytest.mark.parametrize("ensure_all_finite", [True, False])
def test_check_array_ensure_all_finite_object_unsafe_casting(
    X, err_msg, ensure_all_finite
):
    # [FUNCTION DOC]
    # Verifies casting arrays containing NaN/inf to int raises ValueError regardless of
    # ensure_all_finite (float input: finiteness message; object input: numpy cast
    # error).
    # KNOWN ISSUE: the first two parametrized cases are identical duplicates.
    # casting a float array containing NaN or inf to int dtype should
    # raise an error irrespective of the ensure_all_finite parameter.
    with pytest.raises(ValueError, match=err_msg):
        check_array(X, dtype=int, ensure_all_finite=ensure_all_finite)


def test_check_array_series_err_msg():
    # [FUNCTION DOC]
    # Verifies a pandas Series with ensure_2d=True raises the "Expected a 2-dimensional
    # container" message naming the Series type. Skipped without pandas.
    """
    Check that we raise a proper error message when passing a Series and we expect a
    2-dimensional container.

    Non-regression test for:
    https://github.com/scikit-learn/scikit-learn/issues/27498
    """
    pd = pytest.importorskip("pandas")
    ser = pd.Series([1, 2, 3])
    msg = f"Expected a 2-dimensional container but got {type(ser)} instead."
    with pytest.raises(ValueError, match=msg):
        check_array(ser, ensure_2d=True)


@pytest.mark.filterwarnings("ignore:Can't check dok sparse matrix for nan or inf")
def test_check_array():
    # [FUNCTION DOC]
    # Broad check_array test: sparse rejected by default (TypeError); ensure_2d errors
    # for 1D, scalar and 1D sparse rows; nd arrays rejected unless allow_nd; for all
    # combinations of dense inputs, dtypes, orders and copy flags checks dtype,
    # contiguity and copy/no-copy identity; for sparse formats and accept_sparse lists
    # checks dtype, format conversion to the first accepted format and copy semantics;
    # lists and array-likes (_NotAnArray) become ndarrays; negative values raise with
    # ensure_non_negative and the message includes input and estimator names.
    # accept_sparse == False
    # raise error on sparse inputs
    X = [[1, 2], [3, 4]]
    X_csr = sp.csr_array(X)
    with pytest.raises(TypeError):
        check_array(X_csr)

    # ensure_2d=False
    X_array = check_array([0, 1, 2], ensure_2d=False)
    assert X_array.ndim == 1
    # ensure_2d=True with 1d array
    with pytest.raises(ValueError, match="Expected 2D array, got 1D array instead"):
        check_array([0, 1, 2], ensure_2d=True)

    # ensure_2d=True with scalar array
    with pytest.raises(ValueError, match="Expected 2D array, got scalar array instead"):
        check_array(10, ensure_2d=True)

    # ensure_2d=True with 1d sparse array
    if hasattr(sp, "csr_array"):
        sparse_row = next(iter(sp.csr_array(X)))
        if sparse_row.ndim == 1:
            # In scipy 1.14 and later, sparse row is 1D while it was 2D before.
            with pytest.raises(ValueError, match="Expected 2D input, got"):
                check_array(sparse_row, accept_sparse=True, ensure_2d=True)

    # don't allow ndim > 3
    X_ndim = np.arange(8).reshape(2, 2, 2)
    with pytest.raises(ValueError):
        check_array(X_ndim)
    check_array(X_ndim, allow_nd=True)  # doesn't raise

    # dtype and order enforcement.
    X_C = np.arange(4).reshape(2, 2).copy("C")
    X_F = X_C.copy("F")
    X_int = X_C.astype(int)
    X_float = X_C.astype(float)
    Xs = [X_C, X_F, X_int, X_float]
    dtypes = [np.int32, int, float, np.float32, None, bool, object]
    orders = ["C", "F", None]
    copys = [True, False]

    for X, dtype, order, copy in product(Xs, dtypes, orders, copys):
        X_checked = check_array(X, dtype=dtype, order=order, copy=copy)
        if dtype is not None:
            assert X_checked.dtype == dtype
        else:
            assert X_checked.dtype == X.dtype
        if order == "C":
            assert X_checked.flags["C_CONTIGUOUS"]
            assert not X_checked.flags["F_CONTIGUOUS"]
        elif order == "F":
            assert X_checked.flags["F_CONTIGUOUS"]
            assert not X_checked.flags["C_CONTIGUOUS"]
        if copy:
            assert X is not X_checked
        else:
            # doesn't copy if it was already good
            if (
                X.dtype == X_checked.dtype
                and X_checked.flags["C_CONTIGUOUS"] == X.flags["C_CONTIGUOUS"]
                and X_checked.flags["F_CONTIGUOUS"] == X.flags["F_CONTIGUOUS"]
            ):
                assert X is X_checked

    # allowed sparse != None

    # try different type of sparse format
    Xs = []
    Xs.extend(
        [
            sparse_container(X_C)
            for sparse_container in CSR_CONTAINERS
            + CSC_CONTAINERS
            + COO_CONTAINERS
            + DOK_CONTAINERS
        ]
    )
    Xs.extend([Xs[0].astype(np.int64), Xs[0].astype(np.float64)])

    accept_sparses = [["csr", "coo"], ["coo", "dok"]]
    # scipy sparse matrices do not support the object dtype so
    # this dtype is skipped in this loop
    non_object_dtypes = [dt for dt in dtypes if dt is not object]
    for X, dtype, accept_sparse, copy in product(
        Xs, non_object_dtypes, accept_sparses, copys
    ):
        X_checked = check_array(X, dtype=dtype, accept_sparse=accept_sparse, copy=copy)
        if dtype is not None:
            assert X_checked.dtype == dtype
        else:
            assert X_checked.dtype == X.dtype
        if X.format in accept_sparse:
            # no change if allowed
            assert X.format == X_checked.format
        else:
            # got converted
            assert X_checked.format == accept_sparse[0]
        if copy:
            assert X is not X_checked
        else:
            # doesn't copy if it was already good
            if X.dtype == X_checked.dtype and X.format == X_checked.format:
                assert X is X_checked

    # other input formats
    # convert lists to arrays
    X_dense = check_array([[1, 2], [3, 4]])
    assert isinstance(X_dense, np.ndarray)
    # raise on too deep lists
    with pytest.raises(ValueError):
        check_array(X_ndim.tolist())
    check_array(X_ndim.tolist(), allow_nd=True)  # doesn't raise

    # convert weird stuff to arrays
    X_no_array = _NotAnArray(X_dense)
    result = check_array(X_no_array)
    assert isinstance(result, np.ndarray)

    # check negative values when ensure_non_negative=True
    X_neg = check_array([[1, 2], [-3, 4]])
    err_msg = "Negative values in data passed to X in RandomForestRegressor"
    with pytest.raises(ValueError, match=err_msg):
        check_array(
            X_neg,
            ensure_non_negative=True,
            input_name="X",
            estimator=RandomForestRegressor(),
        )


@pytest.mark.parametrize(
    "X",
    [
        [["1", "2"], ["3", "4"]],
        np.array([["1", "2"], ["3", "4"]], dtype="U"),
        np.array([["1", "2"], ["3", "4"]], dtype="S"),
        [[b"1", b"2"], [b"3", b"4"]],
        np.array([[b"1", b"2"], [b"3", b"4"]], dtype="V1"),
    ],
)
def test_check_array_numeric_error(X):
    # [FUNCTION DOC]
    # Verifies dtype="numeric" rejects string/bytes/void inputs (lists and arrays) with
    # the bytes/strings error message.
    """Test that check_array errors when it receives an array of bytes/string
    while a numeric dtype is required."""
    expected_msg = r"dtype='numeric' is not compatible with arrays of bytes/strings"
    with pytest.raises(ValueError, match=expected_msg):
        check_array(X, dtype="numeric")


def test_check_array_pandas_string_dtype_numeric_error():
    # [FUNCTION DOC]
    # Verifies string-typed pandas DataFrames (all-string and mixed) and string Series
    # raise with dtype="numeric", while dtype=None returns object arrays equal to the
    # values.
    """check_array raises an error for pandas StringDtype with dtype='numeric'.

    Non-regression test for pandas 3 where string columns use StringDtype
    instead of object dtype. check_array should reject string data when
    dtype='numeric' is requested.
    """
    pd = pytest.importorskip("pandas")

    # DataFrame with all string columns
    df_str = pd.DataFrame({"a": ["x", "y", "z"], "b": ["1", "2", "3"]})
    with pytest.raises(ValueError):
        check_array(df_str, dtype="numeric")

    # DataFrame with mixed string/numeric columns
    df_mixed = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": ["x", "y", "z"]})
    with pytest.raises(ValueError):
        check_array(df_mixed, dtype="numeric")

    # Series with string dtype
    s_str = pd.Series(["a", "b", "c"])
    with pytest.raises(ValueError):
        check_array(s_str, dtype="numeric", ensure_2d=False)

    # String data with dtype=None should still work
    result = check_array(df_str, dtype=None)
    assert result.dtype == np.object_
    assert_array_equal(result, df_str.values)

    result = check_array(s_str, dtype=None, ensure_2d=False)
    assert result.dtype == np.object_
    assert_array_equal(result, s_str.values)


@pytest.mark.parametrize(
    "pd_dtype", ["Int8", "Int16", "UInt8", "UInt16", "Float32", "Float64"]
)
@pytest.mark.parametrize(
    "dtype, expected_dtype",
    [
        ([np.float32, np.float64], np.float32),
        (np.float64, np.float64),
        ("numeric", np.float64),
    ],
)
def test_check_array_pandas_na_support(pd_dtype, dtype, expected_dtype):
    # [FUNCTION DOC]
    # Verifies nullable pandas extension dtypes with pd.NA are converted to NaN floats
    # with the expected dtype (first of a dtype list, float64, or float64 for "numeric")
    # when NaN is allowed, and raise when ensure_all_finite=True.
    # Test pandas numerical extension arrays with pd.NA
    pd = pytest.importorskip("pandas")

    if pd_dtype in {"Float32", "Float64"}:
        # Extension dtypes with Floats was added in 1.2
        pd = pytest.importorskip("pandas", minversion="1.2")

    X_np = np.array(
        [[1, 2, 3, np.nan, np.nan], [np.nan, np.nan, 8, 4, 6], [1, 2, 3, 4, 5]]
    ).T

    # Creates dataframe with numerical extension arrays with pd.NA
    X = pd.DataFrame(X_np, dtype=pd_dtype, columns=["a", "b", "c"])
    # column c has no nans
    X["c"] = X["c"].astype("float")
    X_checked = check_array(X, ensure_all_finite="allow-nan", dtype=dtype)
    assert_allclose(X_checked, X_np)
    assert X_checked.dtype == expected_dtype

    X_checked = check_array(X, ensure_all_finite=False, dtype=dtype)
    assert_allclose(X_checked, X_np)
    assert X_checked.dtype == expected_dtype

    msg = "Input contains NaN"
    with pytest.raises(ValueError, match=msg):
        check_array(X, ensure_all_finite=True)


def test_check_array_panadas_na_support_series():
    # [FUNCTION DOC]
    # Verifies an Int64 Series with pd.NA raises when finiteness is enforced and
    # converts to float64 (or the requested float32) with NaN otherwise.
    """Check check_array is correct with pd.NA in a series."""
    pd = pytest.importorskip("pandas")

    X_int64 = pd.Series([1, 2, pd.NA], dtype="Int64")

    msg = "Input contains NaN"
    with pytest.raises(ValueError, match=msg):
        check_array(X_int64, ensure_all_finite=True, ensure_2d=False)

    X_out = check_array(X_int64, ensure_all_finite=False, ensure_2d=False)
    assert_allclose(X_out, [1, 2, np.nan])
    assert X_out.dtype == np.float64

    X_out = check_array(
        X_int64, ensure_all_finite=False, ensure_2d=False, dtype=np.float32
    )
    assert_allclose(X_out, [1, 2, np.nan])
    assert X_out.dtype == np.float32


def test_check_array_pandas_dtype_casting():
    # [FUNCTION DOC]
    # Verifies dtype resolution for DataFrames: homogeneous float32 stays float32; mixed
    # float16/int16/float32 resolve to float32; all int16 stays int16 but becomes
    # float64 with FLOAT_DTYPES; a Categorical column converts to int64 (float64 with
    # FLOAT_DTYPES).
    # test that data-frames with homogeneous dtype are not upcast
    pd = pytest.importorskip("pandas")
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
    X_df = pd.DataFrame(X)
    assert check_array(X_df).dtype == np.float32
    assert check_array(X_df, dtype=FLOAT_DTYPES).dtype == np.float32

    X_df = X_df.astype({0: np.float16})
    assert_array_equal(X_df.dtypes, (np.float16, np.float32, np.float32))
    assert check_array(X_df).dtype == np.float32
    assert check_array(X_df, dtype=FLOAT_DTYPES).dtype == np.float32

    X_df = X_df.astype({0: np.int16})
    # float16, int16, float32 casts to float32
    assert check_array(X_df).dtype == np.float32
    assert check_array(X_df, dtype=FLOAT_DTYPES).dtype == np.float32

    X_df = X_df.astype({2: np.float16})
    # float16, int16, float16 casts to float32
    assert check_array(X_df).dtype == np.float32
    assert check_array(X_df, dtype=FLOAT_DTYPES).dtype == np.float32

    X_df = X_df.astype(np.int16)
    assert check_array(X_df).dtype == np.int16
    # we're not using upcasting rules for determining
    # the target type yet, so we cast to the default of float64
    assert check_array(X_df, dtype=FLOAT_DTYPES).dtype == np.float64

    # check that we handle pandas dtypes in a semi-reasonable way
    # this is actually tricky because we can't really know that this
    # should be integer ahead of converting it.
    cat_df = pd.DataFrame({"cat_col": pd.Categorical([1, 2, 3])})
    assert check_array(cat_df).dtype == np.int64
    assert check_array(cat_df, dtype=FLOAT_DTYPES).dtype == np.float64


def test_check_array_on_mock_dataframe():
    # [FUNCTION DOC]
    # Verifies check_array on MockDataFrame keeps the underlying dtype and honors an
    # explicit float32 dtype.
    arr = np.array([[0.2, 0.7], [0.6, 0.5], [0.4, 0.1], [0.7, 0.2]])
    mock_df = MockDataFrame(arr)
    checked_arr = check_array(mock_df)
    assert checked_arr.dtype == arr.dtype
    checked_arr = check_array(mock_df, dtype=np.float32)
    assert checked_arr.dtype == np.dtype(np.float32)


def test_check_array_dtype_stability():
    # [FUNCTION DOC]
    # Verifies integer lists stay integer after check_array, 2D and 1D.
    # test that lists with ints don't get converted to floats
    X = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    assert check_array(X).dtype.kind == "i"
    assert check_array(X, ensure_2d=False).dtype.kind == "i"


def test_check_array_dtype_warning():
    # [FUNCTION DOC]
    # Verifies no warnings are emitted (warnings turned into errors) when casting ints
    # to float64, when a float32 input matches a dtype list (returned as the same
    # object), when copying, and when a CSC float32 input is converted to CSR (new
    # object).
    X_int_list = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    X_float32 = np.asarray(X_int_list, dtype=np.float32)
    X_int64 = np.asarray(X_int_list, dtype=np.int64)
    X_csr_float32 = sp.csr_array(X_float32)
    X_csc_float32 = sp.csc_array(X_float32)
    X_csc_int32 = sp.csc_array(X_int64, dtype=np.int32)
    integer_data = [X_int64, X_csc_int32]
    float32_data = [X_float32, X_csr_float32, X_csc_float32]
    with warnings.catch_warnings():
        warnings.simplefilter("error")

        for X in integer_data:
            X_checked = check_array(X, dtype=np.float64, accept_sparse=True)
            assert X_checked.dtype == np.float64

        for X in float32_data:
            X_checked = check_array(
                X, dtype=[np.float64, np.float32], accept_sparse=True
            )
            assert X_checked.dtype == np.float32
            assert X_checked is X

            X_checked = check_array(
                X,
                dtype=[np.float64, np.float32],
                accept_sparse=["csr", "dok"],
                copy=True,
            )
            assert X_checked.dtype == np.float32
            assert X_checked is not X

        X_checked = check_array(
            X_csc_float32,
            dtype=[np.float64, np.float32],
            accept_sparse=["csr", "dok"],
            copy=False,
        )
        assert X_checked.dtype == np.float32
        assert X_checked is not X_csc_float32
        assert X_checked.format == "csr"


def test_check_array_accept_sparse_type_exception():
    # [FUNCTION DOC]
    # Verifies accept_sparse error paths: False gives TypeError suggesting toarray; a
    # non-str/bool/list value gives ValueError; empty list/tuple gives ValueError; a
    # list containing a non-string (an SVR) gives TypeError mentioning SVR (raised by
    # scipy).
    X = [[1, 2], [3, 4]]
    X_csr = sp.csr_array(X)
    invalid_type = SVR()

    msg = (
        "Sparse data was passed, but dense data is required. "
        r"Use '.toarray\(\)' to convert to a dense numpy array."
    )
    with pytest.raises(TypeError, match=msg):
        check_array(X_csr, accept_sparse=False)

    msg = (
        "Parameter 'accept_sparse' should be a string, "
        "boolean or list of strings. You provided 'accept_sparse=.*'."
    )
    with pytest.raises(ValueError, match=msg):
        check_array(X_csr, accept_sparse=invalid_type)

    msg = (
        "When providing 'accept_sparse' as a tuple or list, "
        "it must contain at least one string value."
    )
    with pytest.raises(ValueError, match=msg):
        check_array(X_csr, accept_sparse=[])
    with pytest.raises(ValueError, match=msg):
        check_array(X_csr, accept_sparse=())
    with pytest.raises(TypeError, match="SVR"):
        check_array(X_csr, accept_sparse=[invalid_type])


def test_check_array_accept_sparse_no_exception():
    # [FUNCTION DOC]
    # Verifies CSR input is accepted with accept_sparse True, "csr", ["csr"] and
    # ("csr",).
    X = [[1, 2], [3, 4]]
    X_csr = sp.csr_array(X)

    check_array(X_csr, accept_sparse=True)
    check_array(X_csr, accept_sparse="csr")
    check_array(X_csr, accept_sparse=["csr"])
    check_array(X_csr, accept_sparse=("csr",))


@pytest.fixture(params=["csr", "csc", "coo", "bsr"])
def X_64bit(request):
    # [FUNCTION DOC]
    # HELPER: Fixture yielding a random 20x10 sparse array in csr/csc/coo/bsr format
    # (parametrized) whose index arrays were cast to int64 (coords for scipy >= 1.13
    # COO, row/col for older COO, indices/indptr otherwise).
    # Inputs: request - pytest request carrying the format. Yields: sparse array.
    X = _sparse_random_array((20, 10), format=request.param)

    if request.param == "coo":
        if hasattr(X, "coords"):
            # for scipy >= 1.13 .coords is a new attribute and is a tuple. The
            # .col and .row attributes do not seem to be able to change the
            # dtype, for more details see https://github.com/scipy/scipy/pull/18530/
            # and https://github.com/scipy/scipy/pull/20003 where .indices was
            # renamed to .coords
            X.coords = tuple(v.astype("int64") for v in X.coords)
        else:
            # scipy < 1.13
            X.row = X.row.astype("int64")
            X.col = X.col.astype("int64")
    else:
        X.indices = X.indices.astype("int64")
        X.indptr = X.indptr.astype("int64")

    yield X


def test_check_array_accept_large_sparse_no_exception(X_64bit):
    # [FUNCTION DOC]
    # Verifies int64-indexed sparse input passes with accept_large_sparse=True.
    # When large sparse are allowed
    check_array(X_64bit, accept_large_sparse=True, accept_sparse=True)


def test_check_array_accept_large_sparse_raise_exception(X_64bit):
    # [FUNCTION DOC]
    # Verifies int64-indexed sparse input raises the 32-bit indices ValueError with
    # accept_large_sparse=False.
    # When large sparse are not allowed
    msg = (
        "Only sparse matrices with 32-bit integer indices "
        "are accepted. Got int64 indices. Please do report"
    )
    with pytest.raises(ValueError, match=msg):
        check_array(X_64bit, accept_sparse=True, accept_large_sparse=False)


def test_check_array_min_samples_and_features_messages():
    # empty list is considered 2D by default:
    # [FUNCTION DOC]
    # Verifies min-sample/min-feature error messages of check_array/check_X_y: empty 2D
    # list (0 features), empty 1D list (0 samples), scalar with ensure_2d=False
    # (TypeError from _num_samples), ensure_min_samples=2 with or without ensure_2d,
    # ensure_min_features =3 even with allow_nd, a (10, 0) array, and that 4D data with
    # a zero axis is not checked for features.
    msg = r"0 feature\(s\) \(shape=\(1, 0\)\) while a minimum of 1 is required."
    with pytest.raises(ValueError, match=msg):
        check_array([[]])

    # If considered a 1D collection when ensure_2d=False, then the minimum
    # number of samples will break:
    msg = r"0 sample\(s\) \(shape=\(0,\)\) while a minimum of 1 is required."
    with pytest.raises(ValueError, match=msg):
        check_array([], ensure_2d=False)

    # Invalid edge case when checking the default minimum sample of a scalar
    msg = re.escape(
        (
            "Input should have at least 1 dimension i.e. satisfy "
            "`len(x.shape) > 0`, got scalar `array(42)` instead."
        )
    )
    with pytest.raises(TypeError, match=msg):
        check_array(42, ensure_2d=False)

    # Simulate a model that would need at least 2 samples to be well defined
    X = np.ones((1, 10))
    y = np.ones(1)
    msg = r"1 sample\(s\) \(shape=\(1, 10\)\) while a minimum of 2 is required."
    with pytest.raises(ValueError, match=msg):
        check_X_y(X, y, ensure_min_samples=2)

    # The same message is raised if the data has 2 dimensions even if this is
    # not mandatory
    with pytest.raises(ValueError, match=msg):
        check_X_y(X, y, ensure_min_samples=2, ensure_2d=False)

    # Simulate a model that would require at least 3 features (e.g. SelectKBest
    # with k=3)
    X = np.ones((10, 2))
    y = np.ones(2)
    msg = r"2 feature\(s\) \(shape=\(10, 2\)\) while a minimum of 3 is required."
    with pytest.raises(ValueError, match=msg):
        check_X_y(X, y, ensure_min_features=3)

    # Only the feature check is enabled whenever the number of dimensions is 2
    # even if allow_nd is enabled:
    with pytest.raises(ValueError, match=msg):
        check_X_y(X, y, ensure_min_features=3, allow_nd=True)

    # Simulate a case where a pipeline stage as trimmed all the features of a
    # 2D dataset.
    X = np.empty(0).reshape(10, 0)
    y = np.ones(10)
    msg = r"0 feature\(s\) \(shape=\(10, 0\)\) while a minimum of 1 is required."
    with pytest.raises(ValueError, match=msg):
        check_X_y(X, y)

    # nd-data is not checked for any minimum number of features by default:
    X = np.ones((10, 0, 28, 28))
    y = np.ones(10)
    X_checked, y_checked = check_X_y(X, y, allow_nd=True)
    assert_array_equal(X, X_checked)
    assert_array_equal(y, y_checked)


def test_check_array_complex_data_error():
    # [FUNCTION DOC]
    # Verifies complex data is rejected with "Complex data not supported" for ndarrays,
    # nested lists/tuples, lists/tuples of arrays, MockDataFrame, sparse COO, and for y
    # via _check_y.
    X = np.array([[1 + 2j, 3 + 4j, 5 + 7j], [2 + 3j, 4 + 5j, 6 + 7j]])
    with pytest.raises(ValueError, match="Complex data not supported"):
        check_array(X)

    # list of lists
    X = [[1 + 2j, 3 + 4j, 5 + 7j], [2 + 3j, 4 + 5j, 6 + 7j]]
    with pytest.raises(ValueError, match="Complex data not supported"):
        check_array(X)

    # tuple of tuples
    X = ((1 + 2j, 3 + 4j, 5 + 7j), (2 + 3j, 4 + 5j, 6 + 7j))
    with pytest.raises(ValueError, match="Complex data not supported"):
        check_array(X)

    # list of np arrays
    X = [np.array([1 + 2j, 3 + 4j, 5 + 7j]), np.array([2 + 3j, 4 + 5j, 6 + 7j])]
    with pytest.raises(ValueError, match="Complex data not supported"):
        check_array(X)

    # tuple of np arrays
    X = (np.array([1 + 2j, 3 + 4j, 5 + 7j]), np.array([2 + 3j, 4 + 5j, 6 + 7j]))
    with pytest.raises(ValueError, match="Complex data not supported"):
        check_array(X)

    # dataframe
    X = MockDataFrame(np.array([[1 + 2j, 3 + 4j, 5 + 7j], [2 + 3j, 4 + 5j, 6 + 7j]]))
    with pytest.raises(ValueError, match="Complex data not supported"):
        check_array(X)

    # sparse matrix
    X = sp.coo_array([[0, 1 + 2j], [0, 0]])
    with pytest.raises(ValueError, match="Complex data not supported"):
        check_array(X)

    # target variable does not always go through check_array but should
    # never accept complex data either.
    y = np.array([1 + 2j, 3 + 4j, 5 + 7j, 2 + 3j, 4 + 5j, 6 + 7j])
    with pytest.raises(ValueError, match="Complex data not supported"):
        _check_y(y)


def test_has_fit_parameter():
    # [FUNCTION DOC]
    # Verifies has_fit_parameter on classes and instances (KNeighborsClassifier lacks
    # sample_weight; RandomForestRegressor and SVR have it) and on a class whose fit is
    # wrapped by the deprecated decorator.
    assert not has_fit_parameter(KNeighborsClassifier, "sample_weight")
    assert has_fit_parameter(RandomForestRegressor, "sample_weight")
    assert has_fit_parameter(SVR, "sample_weight")
    assert has_fit_parameter(SVR(), "sample_weight")

    class TestClassWithDeprecatedFitMethod:
        # [CLASS DOC]
        # HELPER: Minimal class with a deprecated fit accepting sample_weight.
        @deprecated("Deprecated for the purpose of testing has_fit_parameter")
        def fit(self, X, y, sample_weight=None):
            # [FUNCTION DOC]
            # HELPER: No-op fit; the decorator must preserve the signature.
            pass

    assert has_fit_parameter(TestClassWithDeprecatedFitMethod, "sample_weight"), (
        "has_fit_parameter fails for class with deprecated fit method."
    )


def test_check_symmetric():
    # [FUNCTION DOC]
    # Verifies check_symmetric: non-2D input raises; for dense and every sparse format
    # an asymmetric array warns by default, raises with raise_exception, and without
    # warning returns the symmetrized array (sparse output keeps the input format).
    arr_sym = np.array([[0, 1], [1, 2]])
    arr_bad = np.ones(2)
    arr_asym = np.array([[0, 2], [0, 2]])

    test_arrays = {
        "dense": arr_asym,
        "dok": sp.dok_array(arr_asym),
        "csr": sp.csr_array(arr_asym),
        "csc": sp.csc_array(arr_asym),
        "coo": sp.coo_array(arr_asym),
        "lil": sp.lil_array(arr_asym),
        "bsr": sp.bsr_array(arr_asym),
    }

    # check error for bad inputs
    with pytest.raises(ValueError):
        check_symmetric(arr_bad)

    # check that asymmetric arrays are properly symmetrized
    for arr_format, arr in test_arrays.items():
        # Check for warnings and errors
        with pytest.warns(UserWarning):
            check_symmetric(arr)
        with pytest.raises(ValueError):
            check_symmetric(arr, raise_exception=True)

        output = check_symmetric(arr, raise_warning=False)
        if sp.issparse(output):
            assert output.format == arr_format
            assert_array_equal(output.toarray(), arr_sym)
        else:
            assert_array_equal(output, arr_sym)


def test_check_is_fitted_with_is_fitted():
    # [FUNCTION DOC]
    # Verifies check_is_fitted honors a custom __sklearn_is_fitted__: raises before fit,
    # passes after.
    class Estimator(BaseEstimator):
        # [CLASS DOC]
        # HELPER: Estimator whose fitted state is a private _is_fitted flag reported by
        # __sklearn_is_fitted__.
        def fit(self, **kwargs):
            # [FUNCTION DOC]
            # HELPER: Sets self._is_fitted to True and returns self.
            self._is_fitted = True
            return self

        def __sklearn_is_fitted__(self):
            # [FUNCTION DOC]
            # HELPER: Returns True only when the _is_fitted flag exists and is truthy.
            return hasattr(self, "_is_fitted") and self._is_fitted

    with pytest.raises(NotFittedError):
        check_is_fitted(Estimator())
    check_is_fitted(Estimator().fit())


def test_check_is_fitted_stateless():
    # [FUNCTION DOC]
    # Verifies check_is_fitted does not raise for an unfitted estimator whose tags set
    # requires_fit to False.
    """Check that check_is_fitted passes for stateless estimators."""

    class StatelessEstimator(BaseEstimator):
        # [CLASS DOC]
        # HELPER: Estimator declaring requires_fit=False in its tags.
        def fit(self, **kwargs):
            # [FUNCTION DOC]
            # HELPER: No-op fit returning self (never called).
            return self  # pragma: no cover

        def __sklearn_tags__(self):
            # [FUNCTION DOC]
            # HELPER: Returns parent tags with requires_fit set to False.
            tags = super().__sklearn_tags__()
            tags.requires_fit = False
            return tags

    check_is_fitted(StatelessEstimator())


def test_check_is_fitted():
    # [FUNCTION DOC]
    # Verifies check_is_fitted raises TypeError for a class or a string, NotFittedError
    # for unfitted ARDRegression/SVR (catchable as ValueError and AttributeError) with
    # custom msg formatting of the class name, and returns None after fitting on
    # make_blobs data.
    # Check is TypeError raised when non estimator instance passed
    with pytest.raises(TypeError):
        check_is_fitted(ARDRegression)
    with pytest.raises(TypeError):
        check_is_fitted("SVR")

    ard = ARDRegression()
    svr = SVR()

    try:
        with pytest.raises(NotFittedError):
            check_is_fitted(ard)
        with pytest.raises(NotFittedError):
            check_is_fitted(svr)
    except ValueError:
        assert False, "check_is_fitted failed with ValueError"

    # NotFittedError is a subclass of both ValueError and AttributeError
    msg = "Random message %(name)s, %(name)s"
    match = "Random message ARDRegression, ARDRegression"
    with pytest.raises(ValueError, match=match):
        check_is_fitted(ard, msg=msg)

    msg = "Another message %(name)s, %(name)s"
    match = "Another message SVR, SVR"
    with pytest.raises(AttributeError, match=match):
        check_is_fitted(svr, msg=msg)

    ard.fit(*make_blobs())
    svr.fit(*make_blobs())

    assert check_is_fitted(ard) is None
    assert check_is_fitted(svr) is None


def test_check_is_fitted_attributes():
    # [FUNCTION DOC]
    # Verifies _is_fitted and check_is_fitted with explicit attributes and all_or_any:
    # none present fails for all/any, one present passes only with any, both present
    # passes in all modes. Side effects: sets attributes on the test estimator.
    class MyEstimator(BaseEstimator):
        # [CLASS DOC]
        # HELPER: Estimator whose fit sets nothing.
        def fit(self, X, y):
            # [FUNCTION DOC]
            # HELPER: Returns self without setting attributes.
            return self

    msg = "not fitted"
    est = MyEstimator()

    assert not _is_fitted(est, attributes=["a_", "b_"])
    with pytest.raises(NotFittedError, match=msg):
        check_is_fitted(est, attributes=["a_", "b_"])
    assert not _is_fitted(est, attributes=["a_", "b_"], all_or_any=all)
    with pytest.raises(NotFittedError, match=msg):
        check_is_fitted(est, attributes=["a_", "b_"], all_or_any=all)
    assert not _is_fitted(est, attributes=["a_", "b_"], all_or_any=any)
    with pytest.raises(NotFittedError, match=msg):
        check_is_fitted(est, attributes=["a_", "b_"], all_or_any=any)

    est.a_ = "a"
    assert not _is_fitted(est, attributes=["a_", "b_"])
    with pytest.raises(NotFittedError, match=msg):
        check_is_fitted(est, attributes=["a_", "b_"])
    assert not _is_fitted(est, attributes=["a_", "b_"], all_or_any=all)
    with pytest.raises(NotFittedError, match=msg):
        check_is_fitted(est, attributes=["a_", "b_"], all_or_any=all)
    assert _is_fitted(est, attributes=["a_", "b_"], all_or_any=any)
    check_is_fitted(est, attributes=["a_", "b_"], all_or_any=any)

    est.b_ = "b"
    assert _is_fitted(est, attributes=["a_", "b_"])
    check_is_fitted(est, attributes=["a_", "b_"])
    assert _is_fitted(est, attributes=["a_", "b_"], all_or_any=all)
    check_is_fitted(est, attributes=["a_", "b_"], all_or_any=all)
    assert _is_fitted(est, attributes=["a_", "b_"], all_or_any=any)
    check_is_fitted(est, attributes=["a_", "b_"], all_or_any=any)


@pytest.mark.parametrize(
    "wrap", [itemgetter(0), list, tuple], ids=["single", "list", "tuple"]
)
def test_check_is_fitted_with_attributes(wrap):
    # [FUNCTION DOC]
    # Verifies attributes may be given as a str, list or tuple (parametrized wrap):
    # missing coef_ raises before fit, passes after fit, and an undefined attribute
    # raises.
    ard = ARDRegression()
    with pytest.raises(NotFittedError, match="is not fitted yet"):
        check_is_fitted(ard, wrap(["coef_"]))

    ard.fit(*make_blobs())

    # Does not raise
    check_is_fitted(ard, wrap(["coef_"]))

    # Raises when using attribute that is not defined
    with pytest.raises(NotFittedError, match="is not fitted yet"):
        check_is_fitted(ard, wrap(["coef_bad_"]))


def test_check_consistent_length():
    # [FUNCTION DOC]
    # Verifies check_consistent_length accepts equal lengths of lists, tuples, arrays
    # and sparse inputs, raises ValueError on mismatched lengths and TypeError for ints,
    # arbitrary objects, 0-d arrays and estimators (even ones defining __len__).
    """Test that `check_consistent_length` raises on inconsistent lengths and wrong
    input types trigger TypeErrors."""
    check_consistent_length([1], [2], [3], [4], [5])
    check_consistent_length([[1, 2], [[1, 2]]], [1, 2], ["a", "b"])
    check_consistent_length([1], (2,), np.array([3]), sp.csr_array((1, 2)))
    with pytest.raises(ValueError, match="inconsistent numbers of samples"):
        check_consistent_length([1, 2], [1])
    with pytest.raises(TypeError, match=r"got <\w+ 'int'>"):
        check_consistent_length([1, 2], 1)
    with pytest.raises(TypeError, match=r"got <\w+ 'object'>"):
        check_consistent_length([1, 2], object())
    with pytest.raises(TypeError):
        check_consistent_length([1, 2], np.array(1))
    # Despite ensembles having __len__ they must raise TypeError
    with pytest.raises(TypeError, match="Expected sequence or array-like"):
        check_consistent_length([1, 2], RandomForestRegressor())
    # XXX: We should have a test with a string, but what is correct behaviour?


@pytest.mark.parametrize(
    "array_namespace, device_name, dtype_name",
    yield_namespace_device_dtype_combinations(),
)
def test_check_consistent_length_array_api(array_namespace, device_name, dtype_name):
    # [FUNCTION DOC]
    # Verifies check_consistent_length with Array API inputs (all namespace/device/dtype
    # combos) mixed with lists, object arrays and sparse, plus the mismatch error, under
    # array_api_dispatch.
    """Test that check_consistent_length works with different array types."""
    xp, device = _array_api_for_tests(array_namespace, device_name, dtype_name)

    with config_context(array_api_dispatch=True):
        check_consistent_length(
            xp.asarray([1, 2, 3], device=device),
            xp.asarray([[1, 1], [2, 2], [3, 3]], device=device),
            [1, 2, 3],
            ["a", "b", "c"],
            np.asarray(("a", "b", "c"), dtype=object),
            sp.csr_array([[0, 1], [1, 0], [0, 0]]),
        )

        with pytest.raises(ValueError, match="inconsistent numbers of samples"):
            check_consistent_length(
                xp.asarray([1, 2], device=device),
                xp.asarray([1], device=device),
            )


def test_check_dataframe_fit_attribute():
    # [FUNCTION DOC]
    # Verifies a DataFrame with a column named "fit" is not mistaken for an estimator by
    # _num_samples (via check_consistent_length).
    # check pandas.DataFrame with 'fit' column does not raise error
    # https://github.com/scikit-learn/scikit-learn/issues/8415
    # This essentially tests _num_samples.
    pd = pytest.importorskip("pandas")
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    X_df = pd.DataFrame(X, columns=["a", "b", "fit"])
    check_consistent_length(X_df)


def test_suppress_validation():
    # [FUNCTION DOC]
    # Verifies assert_all_finite is skipped when the global assume_finite config is True
    # and active again after resetting it.
    # Side effects: mutates global sklearn config (not restored if an assertion fails).
    X = np.array([0, np.inf])
    with pytest.raises(ValueError):
        assert_all_finite(X)
    sklearn.set_config(assume_finite=True)
    assert_all_finite(X)
    sklearn.set_config(assume_finite=False)
    with pytest.raises(ValueError):
        assert_all_finite(X)


def test_check_array_series():
    # [FUNCTION DOC]
    # Verifies check_array on a 1D pandas Series returns the values and that a
    # categorical Series with dtype=None returns an object array of the categories'
    # values.
    # regression test that check_array works on pandas Series
    pd = pytest.importorskip("pandas")
    res = check_array(pd.Series([1, 2, 3]), ensure_2d=False)
    assert_array_equal(res, np.array([1, 2, 3]))

    # with categorical dtype (not a numpy dtype) (GH12699)
    s = pd.Series(["a", "b", "c"]).astype("category")
    res = check_array(s, dtype=None, ensure_2d=False)
    assert_array_equal(res, np.array(["a", "b", "c"], dtype=object))


@pytest.mark.parametrize(
    "dtype", ((np.float64, np.float32), np.float64, None, "numeric")
)
@pytest.mark.parametrize("bool_dtype", ("bool", "boolean"))
def test_check_dataframe_mixed_float_dtypes(dtype, bool_dtype):
    # pandas.DataFrame will coerce a boolean into a object, this is a mismatch
    # with np.result_type which will return a float
    # check_array needs to explicitly check for bool dtype in a dataframe for
    # this situation
    # https://github.com/scikit-learn/scikit-learn/issues/15787

    # [FUNCTION DOC]
    # Verifies DataFrames mixing int, float and bool/boolean columns convert to float64
    # with correct values for various dtype requests (regression for bool coercion to
    # object).
    pd = pytest.importorskip("pandas")

    df = pd.DataFrame(
        {
            "int": [1, 2, 3],
            "float": [0, 0.1, 2.1],
            "bool": pd.Series([True, False, True], dtype=bool_dtype),
        },
        columns=["int", "float", "bool"],
    )

    array = check_array(df, dtype=dtype)
    assert array.dtype == np.float64
    expected_array = np.array(
        [[1.0, 0.0, 1.0], [2.0, 0.1, 0.0], [3.0, 2.1, 1.0]], dtype=float
    )
    assert_allclose_dense_sparse(array, expected_array)


def test_check_dataframe_with_only_bool():
    # [FUNCTION DOC]
    # Verifies a bool-only DataFrame stays bool with dtype=None and that bool plus int
    # columns give int64 with dtype="numeric".
    """Check that pandas.DataFrame with bool return a boolean arrays."""
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame({"bool": [True, False, True]})

    array = check_array(df, dtype=None)
    assert array.dtype == np.bool_
    assert_array_equal(array, [[True], [False], [True]])

    # common dtype is int for bool + int
    df = pd.DataFrame(
        {"bool": [True, False, True], "int": [1, 2, 3]},
        columns=["bool", "int"],
    )
    array = check_array(df, dtype="numeric")
    assert array.dtype == np.int64
    assert_array_equal(array, [[1, 1], [0, 2], [1, 3]])


def test_check_dataframe_with_only_boolean():
    # [FUNCTION DOC]
    # Verifies a nullable "boolean" DataFrame converts to float64 with dtype=None.
    """Check that pandas.DataFrame with boolean return a float array with dtype=None"""
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame({"bool": pd.Series([True, False, True], dtype="boolean")})

    array = check_array(df, dtype=None)
    assert array.dtype == np.float64
    assert_array_equal(array, [[True], [False], [True]])


class DummyMemory:
    # [CLASS DOC]
    # HELPER: Minimal object exposing a joblib.Memory-like cache method.
    def cache(self, func):
        # [FUNCTION DOC]
        # HELPER: Returns func unchanged (no caching).
        return func


class WrongDummyMemory:
    # [CLASS DOC]
    # HELPER: Object without a cache method, used to trigger check_memory errors.
    pass


def test_check_memory(tmp_path):
    # [FUNCTION DOC]
    # Verifies check_memory: a path string and None produce joblib.Memory with that
    # location, an object with cache is returned as is, and an int or an object without
    # cache raises ValueError with the documented message.
    # Inputs: tmp_path - pytest temporary directory.
    cache_directory = str(tmp_path / "cache_directory")
    memory = check_memory(cache_directory)
    assert memory.location == cache_directory

    memory = check_memory(None)
    assert memory.location is None

    dummy = DummyMemory()
    memory = check_memory(dummy)
    assert memory is dummy

    msg = (
        "'memory' should be None, a string or have the same interface as"
        " joblib.Memory. Got memory='1' instead."
    )
    with pytest.raises(ValueError, match=msg):
        check_memory(1)
    dummy = WrongDummyMemory()
    msg = (
        "'memory' should be None, a string or have the same interface as"
        " joblib.Memory. Got memory='{}' instead.".format(dummy)
    )
    with pytest.raises(ValueError, match=msg):
        check_memory(dummy)


@pytest.mark.parametrize("copy", [True, False])
def test_check_array_memmap(copy):
    # [FUNCTION DOC]
    # Verifies check_array on a read-only memmap shares memory and stays read-only when
    # copy=False, and yields an independent writeable array when copy=True.
    X = np.ones((4, 4))
    with TempMemmap(X, mmap_mode="r") as X_memmap:
        X_checked = check_array(X_memmap, copy=copy)
        assert np.may_share_memory(X_memmap, X_checked) == (not copy)
        assert X_checked.flags["WRITEABLE"] == copy


@pytest.mark.parametrize(
    "estimator_name, estimator_value, delegates, expected_result, expected_exception",
    [
        (
            "estimator_",
            type("SubEstimator", (), {"attribute_present": True}),
            None,  # default delegates - ["estimator_", "estimator"]
            True,  # expected_result is True b/c delegate and attribute are present
            None,  # expected_exception not relevant for this case
        ),
        (
            "estimator",
            type("SubEstimator", (), {"attribute_present": True}),
            None,  # default delegates - ["estimator_", "estimator"]
            True,  # expected_result is True b/c delegate and attribute are present
            None,  # expected_exception not relevant for this case
        ),
        (
            "estimators_",
            [
                type("SubEstimator", (), {"attribute_present": True})
            ],  # list of sub-estimators
            ["estimators_"],
            True,  # expected_result is True b/c delegate and attribute are present
            None,  # expected_exception not relevant for this case
        ),
        (
            "custom_estimator",  # custom estimator attribute name
            type("SubEstimator", (), {"attribute_present": True}),
            ["custom_estimator"],  # custom delegates
            True,  # expected_result is True b/c delegate and attribute are present
            None,  # expected_exception not relevant for this case
        ),
        (
            "no_estimator",  # no estimator attribute name
            type("SubEstimator", (), {"attribute_present": True}),
            None,  # default delegates - ["estimator_", "estimator"]
            None,  # expected_result is not relevant for this case
            ValueError,  # should raise ValueError b/c no estimator found from delegates
        ),
        (
            "estimator",
            type("SubEstimator", (), {"attribute_absent": True}),  # attribute_absent
            None,  # default delegates - ["estimator_", "estimator"]
            None,  # expected_result is not relevant for this case
            AttributeError,  # should raise AttributeError b/c attribute is absent
        ),
    ],
    ids=[
        "fitted_estimator_with_default_delegates",
        "estimator_with_default_delegates",
        "list_of_estimators_with_estimators_",
        "custom_estimator_with_custom_delegates",
        "no_estimator_with_default_delegates",
        "estimator_with_default_delegates_but_absent_attribute",
    ],
)
def test_estimator_has(
    estimator_name, estimator_value, delegates, expected_result, expected_exception
):
    # [FUNCTION DOC]
    # Verifies _estimator_has: finds the attribute on default delegates estimator_ and
    # estimator, on the first element of a list delegate, on custom delegates; raises
    # ValueError when no delegate exists and AttributeError when the attribute is
    # missing.
    # Inputs: parametrized delegate name/value, delegates, expected result/exception.
    """
    Tests the _estimator_has function by verifying:
    - Functionality with default and custom delegates.
    - Raises ValueError if delegates are missing.
    - Raises AttributeError if the specified attribute is missing.
    """

    # always checks for attribute - "attribute_present"
    # ["estimator_", "estimator"] is default value for delegates
    if delegates is None:
        check = _estimator_has("attribute_present")
    else:
        check = _estimator_has("attribute_present", delegates=delegates)

    class MockEstimator:
        # [CLASS DOC]
        # HELPER: Empty class on which the delegate attribute is set dynamically.
        pass

    a = MockEstimator()
    setattr(a, estimator_name, estimator_value)

    if expected_exception:
        with pytest.raises(expected_exception):
            check(a)
    else:
        assert check(a) == expected_result


@pytest.mark.parametrize(
    "retype",
    [
        np.asarray,
        sp.csr_matrix,
        sp.csc_matrix,
        sp.coo_matrix,
        sp.lil_matrix,
        sp.bsr_matrix,
        sp.dok_matrix,
        sp.dia_matrix,
        sp.csr_array,
        sp.csc_array,
        sp.coo_array,
        sp.lil_array,
        sp.bsr_array,
        sp.dok_array,
        sp.dia_array,
    ],
)
def test_check_non_negative(retype):
    # [FUNCTION DOC]
    # Verifies check_non_negative on dense and every sparse matrix/array format: passes
    # for non-negative data and all-zero data, and raises "Negative values in data"
    # after a negative entry is introduced.
    A = np.array([[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    X = retype(A)
    check_non_negative(X, "")
    X = retype([[0, 0], [0, 0]])
    check_non_negative(X, "")

    A[0, 0] = -1
    X = retype(A)
    with pytest.raises(ValueError, match="Negative "):
        check_non_negative(X, "")


def test_check_X_y_informative_error():
    # [FUNCTION DOC]
    # Verifies check_X_y with y=None raises ValueError naming "estimator" by default and
    # the estimator class name when an estimator is given.
    X = np.ones((2, 2))
    y = None
    msg = "estimator requires y to be passed, but the target y is None"
    with pytest.raises(ValueError, match=msg):
        check_X_y(X, y)

    msg = "RandomForestRegressor requires y to be passed, but the target y is None"
    with pytest.raises(ValueError, match=msg):
        check_X_y(X, y, estimator=RandomForestRegressor())


def test_num_samples_on_non_standard_shape():
    # [FUNCTION DOC]
    # Verifies _num_samples falls back to len when shape[0] is not numeric, raises
    # TypeError when there is neither a numeric shape nor __len__, and converts objects
    # implementing only __array__ (1D and 2D) to count samples.
    """Test _num_samples on different non standard input X."""

    class TestNonNumericShape:
        # [CLASS DOC]
        # HELPER: Object with a non-numeric shape tuple and a length of 3.
        def __init__(self):
            # [FUNCTION DOC]
            # HELPER: Sets shape to a tuple containing a string.
            self.shape = ("not numeric",)

        def __len__(self):
            # [FUNCTION DOC]
            # HELPER: Always returns 3.
            return len([1, 2, 3])

    X = TestNonNumericShape()
    assert _num_samples(X) == len(X)

    # check that it gives a good error if there's no __len__
    class TestNoLenWeirdShape:
        # [CLASS DOC]
        # HELPER: Object with a non-numeric shape and no __len__.
        def __init__(self):
            # [FUNCTION DOC]
            # HELPER: Sets shape to a tuple containing a string.
            self.shape = ("not numeric",)

    with pytest.raises(TypeError, match="Expected sequence or array-like"):
        _num_samples(TestNoLenWeirdShape())

    class TestNoLenNoShapeButArrayProtocol:
        # [CLASS DOC]
        # HELPER: Wrapper exposing only the __array__ protocol around stored data.
        def __init__(self, x):
            # [FUNCTION DOC]
            # HELPER: Stores the wrapped data x.
            self.x = x

        def __array__(self, dtype=None, copy=None):
            # [FUNCTION DOC]
            # HELPER: Returns the stored data as an ndarray of the requested dtype; copy
            # is accepted for the numpy 2 protocol but ignored.
            return np.asarray(self.x, dtype=dtype)  # copy needs numpy >= 2.0

    X = TestNoLenNoShapeButArrayProtocol(np.arange(3))
    assert _num_samples(X) == 3
    X = TestNoLenNoShapeButArrayProtocol(np.arange(6).reshape(3, 2))
    assert _num_samples(X) == 3


@pytest.mark.parametrize(
    "constructor_name", ["list", "tuple", "array", "series", "polars_series"]
)
def test_num_samples_on_1d(constructor_name):
    # [FUNCTION DOC]
    # Verifies _num_samples returns 3 for 1D list, tuple, array, pandas and polars
    # Series.
    """Test _num_samples on different 1d input X."""
    X = _convert_container(list(range(3)), constructor_name)
    assert _num_samples(X) == 3


@pytest.mark.parametrize(
    "constructor_name",
    ["list", "tuple", "array", "sparse", "pandas", "pyarrow", "polars"],
)
def test_num_samples_on_dataframe_likes(constructor_name):
    # [FUNCTION DOC]
    # Verifies _num_samples returns 3 for 2D list, tuple, array, sparse, pandas, pyarrow
    # and polars containers.
    """Test _num_samples on different dataframe-like input X."""
    X = _convert_container([[1, 11], [2, 22], [3, 33]], constructor_name)
    assert _num_samples(X) == 3


@pytest.mark.parametrize("x", [2, 3, 2.5, 5])
def test_check_scalar_valid(x):
    # [FUNCTION DOC]
    # Verifies check_scalar returns valid Real values within inclusive bounds [2, 5]
    # without emitting any warning.
    """Test that check_scalar returns no error/warning if valid inputs are
    provided"""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        scalar = check_scalar(
            x,
            "test_name",
            target_type=numbers.Real,
            min_val=2,
            max_val=5,
            include_boundaries="both",
        )
    assert scalar == x


@pytest.mark.parametrize(
    "x, target_name, target_type, min_val, max_val, include_boundaries, err_msg",
    [
        (
            1,
            "test_name1",
            float,
            2,
            4,
            "neither",
            TypeError("test_name1 must be an instance of float, not int."),
        ),
        (
            None,
            "test_name1",
            numbers.Real,
            2,
            4,
            "neither",
            TypeError("test_name1 must be an instance of float, not NoneType."),
        ),
        (
            None,
            "test_name1",
            numbers.Integral,
            2,
            4,
            "neither",
            TypeError("test_name1 must be an instance of int, not NoneType."),
        ),
        (
            1,
            "test_name1",
            (float, bool),
            2,
            4,
            "neither",
            TypeError("test_name1 must be an instance of {float, bool}, not int."),
        ),
        (
            1,
            "test_name2",
            int,
            2,
            4,
            "neither",
            ValueError("test_name2 == 1, must be > 2."),
        ),
        (
            5,
            "test_name3",
            int,
            2,
            4,
            "neither",
            ValueError("test_name3 == 5, must be < 4."),
        ),
        (
            2,
            "test_name4",
            int,
            2,
            4,
            "right",
            ValueError("test_name4 == 2, must be > 2."),
        ),
        (
            4,
            "test_name5",
            int,
            2,
            4,
            "left",
            ValueError("test_name5 == 4, must be < 4."),
        ),
        (
            4,
            "test_name6",
            int,
            2,
            4,
            "bad parameter value",
            ValueError(
                "Unknown value for `include_boundaries`: 'bad parameter value'. "
                "Possible values are: ('left', 'right', 'both', 'neither')."
            ),
        ),
        (
            4,
            "test_name7",
            int,
            None,
            4,
            "left",
            ValueError(
                "`include_boundaries`='left' without specifying explicitly `min_val` "
                "is inconsistent."
            ),
        ),
        (
            4,
            "test_name8",
            int,
            2,
            None,
            "right",
            ValueError(
                "`include_boundaries`='right' without specifying explicitly `max_val` "
                "is inconsistent."
            ),
        ),
    ],
)
def test_check_scalar_invalid(
    x, target_name, target_type, min_val, max_val, include_boundaries, err_msg
):
    # [FUNCTION DOC]
    # Verifies check_scalar error type and exact message for wrong types (incl. tuple of
    # types and numbers.Real/Integral readable names), out-of-bound values with each
    # include_boundaries mode, an unknown include_boundaries value, and inconsistent
    # boundary specs missing min_val/max_val.
    """Test that check_scalar returns the right error if a wrong input is
    given"""
    with pytest.raises(Exception) as raised_error:
        check_scalar(
            x,
            target_name,
            target_type=target_type,
            min_val=min_val,
            max_val=max_val,
            include_boundaries=include_boundaries,
        )
    assert str(raised_error.value) == str(err_msg)
    assert isinstance(raised_error.value, type(err_msg))


_psd_cases_valid = {
    "nominal": ((1, 2), np.array([1, 2]), None, ""),
    "nominal_np_array": (np.array([1, 2]), np.array([1, 2]), None, ""),
    "insignificant_imag": (
        (5, 5e-5j),
        np.array([5, 0]),
        PositiveSpectrumWarning,
        "There are imaginary parts in eigenvalues \\(1e\\-05 of the maximum real part",
    ),
    "insignificant neg": ((5, -5e-5), np.array([5, 0]), PositiveSpectrumWarning, ""),
    "insignificant neg float32": (
        np.array([1, -1e-6], dtype=np.float32),
        np.array([1, 0], dtype=np.float32),
        PositiveSpectrumWarning,
        "There are negative eigenvalues \\(1e\\-06 of the maximum positive",
    ),
    "insignificant neg float64": (
        np.array([1, -1e-10], dtype=np.float64),
        np.array([1, 0], dtype=np.float64),
        PositiveSpectrumWarning,
        "There are negative eigenvalues \\(1e\\-10 of the maximum positive",
    ),
    "insignificant pos": (
        (5, 4e-12),
        np.array([5, 0]),
        PositiveSpectrumWarning,
        "the largest eigenvalue is more than 1e\\+12 times the smallest",
    ),
}


@pytest.mark.parametrize(
    "lambdas, expected_lambdas, w_type, w_msg",
    list(_psd_cases_valid.values()),
    ids=list(_psd_cases_valid.keys()),
)
@pytest.mark.parametrize("enable_warnings", [True, False])
def test_check_psd_eigenvalues_valid(
    lambdas, expected_lambdas, w_type, w_msg, enable_warnings
):
    # Test that ``_check_psd_eigenvalues`` returns the right output for valid
    # input, possibly raising the right warning

    # [FUNCTION DOC]
    # Verifies _check_psd_eigenvalues cleans valid spectra (drops small imaginary parts,
    # zeroes insignificant negatives and tiny positives, precision-aware) and emits
    # PositiveSpectrumWarning with the expected message only when enable_warnings is
    # True; with warnings disabled any such warning would be an error.
    if not enable_warnings:
        w_type = None

    if w_type is None:
        with warnings.catch_warnings():
            warnings.simplefilter("error", PositiveSpectrumWarning)
            lambdas_fixed = _check_psd_eigenvalues(
                lambdas, enable_warnings=enable_warnings
            )
    else:
        with pytest.warns(w_type, match=w_msg):
            lambdas_fixed = _check_psd_eigenvalues(
                lambdas, enable_warnings=enable_warnings
            )

    assert_allclose(expected_lambdas, lambdas_fixed)


_psd_cases_invalid = {
    "significant_imag": (
        (5, 5j),
        ValueError,
        "There are significant imaginary parts in eigenv",
    ),
    "all negative": (
        (-5, -1),
        ValueError,
        "All eigenvalues are negative \\(maximum is -1",
    ),
    "significant neg": (
        (5, -1),
        ValueError,
        "There are significant negative eigenvalues",
    ),
    "significant neg float32": (
        np.array([3e-4, -2e-6], dtype=np.float32),
        ValueError,
        "There are significant negative eigenvalues",
    ),
    "significant neg float64": (
        np.array([1e-5, -2e-10], dtype=np.float64),
        ValueError,
        "There are significant negative eigenvalues",
    ),
}


@pytest.mark.parametrize(
    "lambdas, err_type, err_msg",
    list(_psd_cases_invalid.values()),
    ids=list(_psd_cases_invalid.keys()),
)
def test_check_psd_eigenvalues_invalid(lambdas, err_type, err_msg):
    # Test that ``_check_psd_eigenvalues`` raises the right error for invalid
    # input

    # [FUNCTION DOC]
    # Verifies _check_psd_eigenvalues raises ValueError for significant imaginary parts,
    # all-negative spectra and significant negative eigenvalues (float32 and float64).
    with pytest.raises(err_type, match=err_msg):
        _check_psd_eigenvalues(lambdas)


def _check_sample_weight_common(xp):
    # [FUNCTION DOC]
    # HELPER: Shared assertions for _check_sample_weight under namespace xp: None gives
    # ones, a scalar gives a constant array, 2D weights and wrong length raise, float32
    # weights keep their dtype, negative weights raise with ensure_non_negative, and all
    # zero weights raise.
    # Inputs: xp - array namespace. Side effects: none beyond assertions.
    # Common checks between numpy/array api tests
    # for check_sample_weight
    # check None input
    sample_weight = _check_sample_weight(None, X=xp.ones((5, 2)))
    assert_allclose(move_to(sample_weight, xp=np, device="cpu"), np.ones(5))

    # check numbers input
    sample_weight = _check_sample_weight(2.0, X=xp.ones((5, 2)))
    assert_allclose(move_to(sample_weight, xp=np, device="cpu"), 2 * np.ones(5))

    # check wrong number of dimensions
    with pytest.raises(ValueError, match=r"Sample weights must be 1D array or scalar"):
        _check_sample_weight(xp.ones((2, 4)), X=xp.ones((2, 2)))

    # check incorrect n_samples
    msg = re.escape(f"sample_weight.shape == {xp.ones(4).shape}, expected (2,)!")
    with pytest.raises(ValueError, match=msg):
        _check_sample_weight(xp.ones(4), X=xp.ones((2, 2)))

    # float32 dtype is preserved
    X = xp.ones((5, 2))
    sample_weight = xp.ones(5, dtype=xp.float32)
    sample_weight = _check_sample_weight(sample_weight, X)
    assert sample_weight.dtype == xp.float32

    # check negative weight when ensure_non_negative=True
    X = xp.ones((5, 2))
    sample_weight = xp.ones(_num_samples(X))
    sample_weight[-1] = -10
    err_msg = "Negative values in data passed to `sample_weight`"
    with pytest.raises(ValueError, match=err_msg):
        _check_sample_weight(sample_weight, X, ensure_non_negative=True)

    # check error raised when allow_all_zero_weights=False
    X = xp.ones((5, 2))
    sample_weight = xp.zeros(_num_samples(X))
    err_msg = "Sample weights must contain at least one non-zero number."
    with pytest.raises(ValueError, match=err_msg):
        _check_sample_weight(sample_weight, X, allow_all_zero_weights=False)


def test_check_sample_weight():
    # [FUNCTION DOC]
    # Verifies _check_sample_weight returns a C-contiguous array for strided input, runs
    # the common checks on numpy, and maps an int dtype request to float64.
    # check array order
    sample_weight = np.ones(10)[::2]
    assert not sample_weight.flags["C_CONTIGUOUS"]
    sample_weight = _check_sample_weight(sample_weight, X=np.ones((5, 1)))
    assert sample_weight.flags["C_CONTIGUOUS"]

    _check_sample_weight_common(np)

    # int dtype will be converted to float64 instead
    X = np.ones((5, 2), dtype=int)
    sample_weight = _check_sample_weight(None, X, dtype=X.dtype)
    assert sample_weight.dtype == np.float64


@pytest.mark.parametrize(
    "array_namespace, device_name, dtype_name",
    yield_namespace_device_dtype_combinations(),
)
def test_check_sample_weight_array_api(array_namespace, device_name, dtype_name):
    # [FUNCTION DOC]
    # Verifies _check_sample_weight under Array API dispatch for every namespace/device/
    # dtype combination: contiguity for numpy and the common checks.
    xp, device = _array_api_for_tests(array_namespace, device_name, dtype_name)
    with config_context(array_api_dispatch=True):
        # check array order
        sample_weight = xp.ones(10)[::2]
        if _is_numpy_namespace(xp):
            assert not sample_weight.flags["C_CONTIGUOUS"]
        sample_weight = _check_sample_weight(sample_weight, X=xp.ones((5, 1)))
        if _is_numpy_namespace(xp):
            assert sample_weight.flags["C_CONTIGUOUS"]

        _check_sample_weight_common(xp)


@pytest.mark.parametrize("y_true", [[0], [0, 1], [-1, 1], [1, 1, 1], [-1, -1, -1]])
def test_check_pos_label_consistency(y_true):
    # [FUNCTION DOC]
    # Verifies pos_label None resolves to 1 for label sets within {0,1} or {-1,1}.
    assert _check_pos_label_consistency(None, y_true) == 1


@pytest.mark.parametrize(
    "array_namespace, device_name, dtype_name",
    yield_namespace_device_dtype_combinations(),
)
@pytest.mark.parametrize("y_true", [[0], [0, 1], [-1, 1], [1, 1, 1], [-1, -1, -1]])
def test_check_pos_label_consistency_array_api(
    array_namespace, device_name, dtype_name, y_true
):
    # [FUNCTION DOC]
    # Same as test_check_pos_label_consistency with Array API inputs under dispatch.
    xp, device = _array_api_for_tests(array_namespace, device_name, dtype_name)
    with config_context(array_api_dispatch=True):
        arr = xp.asarray(y_true, device=device)
        assert _check_pos_label_consistency(None, arr) == 1


@pytest.mark.parametrize("y_true", [[2, 3, 4], [-10], [0, -1]])
def test_check_pos_label_consistency_invalid(y_true):
    # [FUNCTION DOC]
    # Verifies ambiguous label sets raise when pos_label is None, and an explicit
    # pos_label is returned unchanged.
    with pytest.raises(ValueError, match="y_true takes value in"):
        _check_pos_label_consistency(None, y_true)
    # Make sure we only raise if pos_label is None
    assert _check_pos_label_consistency("a", y_true) == "a"


@pytest.mark.parametrize(
    "array_namespace, device_name, dtype_name",
    yield_namespace_device_dtype_combinations(),
)
@pytest.mark.parametrize("y_true", [[2, 3, 4], [-10], [0, -1]])
def test_check_pos_label_consistency_invalid_array_api(
    array_namespace, device_name, dtype_name, y_true
):
    # [FUNCTION DOC]
    # Same as test_check_pos_label_consistency_invalid with Array API inputs under
    # dispatch.
    xp, device = _array_api_for_tests(array_namespace, device_name, dtype_name)
    with config_context(array_api_dispatch=True):
        arr = xp.asarray(y_true, device=device)
        with pytest.raises(ValueError, match="y_true takes value in"):
            _check_pos_label_consistency(None, arr)
        # Make sure we only raise if pos_label is None
        assert _check_pos_label_consistency("a", arr) == "a"


CS_SPARSE = [sp.csr_array, sp.csr_matrix, sp.csc_array, sp.csc_matrix]


@pytest.mark.parametrize("toarray", [np.array] + CS_SPARSE)
def test_allclose_dense_sparse_equals(toarray):
    # [FUNCTION DOC]
    # Verifies _allclose_dense_sparse is True for identical dense or CSR/CSC inputs.
    base = np.arange(9).reshape(3, 3)
    x, y = toarray(base), toarray(base)
    assert _allclose_dense_sparse(x, y)


@pytest.mark.parametrize("toarray", [np.array] + CS_SPARSE)
def test_allclose_dense_sparse_not_equals(toarray):
    # [FUNCTION DOC]
    # Verifies _allclose_dense_sparse is False when values differ by 1.
    base = np.arange(9).reshape(3, 3)
    x, y = toarray(base), toarray(base + 1)
    assert not _allclose_dense_sparse(x, y)


@pytest.mark.parametrize("toarray", CS_SPARSE)
def test_allclose_dense_sparse_raise(toarray):
    # [FUNCTION DOC]
    # Verifies comparing dense with sparse raises ValueError.
    x = np.arange(9).reshape(3, 3)
    y = toarray(x + 1)

    msg = "Can only compare two sparse matrices, not a sparse matrix and an array"
    with pytest.raises(ValueError, match=msg):
        _allclose_dense_sparse(x, y)


def test_deprecate_positional_args_warns_for_function():
    # [FUNCTION DOC]
    # Verifies _deprecate_positional_args on functions warns with the names/values of
    # keyword-only args passed positionally, including with defaulted positional params
    # and keyword-only params without defaults.
    @_deprecate_positional_args
    def f1(a, b, *, c=1, d=1):
        # [FUNCTION DOC]
        # HELPER: No-op function with two positional and two keyword-only params.
        pass

    with pytest.warns(FutureWarning, match=r"Pass c=3 as keyword args"):
        f1(1, 2, 3)

    with pytest.warns(FutureWarning, match=r"Pass c=3, d=4 as keyword args"):
        f1(1, 2, 3, 4)

    @_deprecate_positional_args
    def f2(a=1, *, b=1, c=1, d=1):
        # [FUNCTION DOC]
        # HELPER: No-op function with one defaulted positional and three keyword-only
        # params.
        pass

    with pytest.warns(FutureWarning, match=r"Pass b=2 as keyword args"):
        f2(1, 2)

    # The * is place before a keyword only argument without a default value
    @_deprecate_positional_args
    def f3(a, *, b, c=1, d=1):
        # [FUNCTION DOC]
        # HELPER: No-op function whose first keyword-only param has no default.
        pass

    with pytest.warns(FutureWarning, match=r"Pass b=2 as keyword args"):
        f3(1, 2)


def test_deprecate_positional_args_warns_for_function_version():
    # [FUNCTION DOC]
    # Verifies the version passed to the decorator appears in the FutureWarning message.
    @_deprecate_positional_args(version="1.1")
    def f1(a, *, b):
        # [FUNCTION DOC]
        # HELPER: No-op function with one positional and one keyword-only param.
        pass

    with pytest.warns(
        FutureWarning, match=r"From version 1.1 passing these as positional"
    ):
        f1(1, 2)


def test_deprecate_positional_args_warns_for_class():
    # [FUNCTION DOC]
    # Verifies the decorator works on __init__ methods (self counted as positional) and
    # reports one or two keyword-only args passed positionally.
    class A1:
        # [CLASS DOC]
        # HELPER: Class whose decorated __init__ has two required positional params.
        @_deprecate_positional_args
        def __init__(self, a, b, *, c=1, d=1):
            # [FUNCTION DOC]
            # HELPER: No-op constructor.
            pass

    with pytest.warns(FutureWarning, match=r"Pass c=3 as keyword args"):
        A1(1, 2, 3)

    with pytest.warns(FutureWarning, match=r"Pass c=3, d=4 as keyword args"):
        A1(1, 2, 3, 4)

    class A2:
        # [CLASS DOC]
        # HELPER: Class whose decorated __init__ has defaulted positional params.
        @_deprecate_positional_args
        def __init__(self, a=1, b=1, *, c=1, d=1):
            # [FUNCTION DOC]
            # HELPER: No-op constructor.
            pass

    with pytest.warns(FutureWarning, match=r"Pass c=3 as keyword args"):
        A2(1, 2, 3)

    with pytest.warns(FutureWarning, match=r"Pass c=3, d=4 as keyword args"):
        A2(1, 2, 3, 4)


@pytest.mark.parametrize("indices", [None, [1, 3]])
def test_check_method_params(indices):
    # [FUNCTION DOC]
    # Verifies _check_method_params subsets sample-aligned params (list, array, sparse
    # column vectors) with indices (all rows when None) and passes through row vectors,
    # scalars, strings and None unchanged (same object).
    # Side effects: uses the unseeded global numpy RNG for X.
    X = np.random.randn(4, 2)
    _params = {
        "list": [1, 2, 3, 4],
        "array": np.array([1, 2, 3, 4]),
        "sparse-col2": sp.csc_matrix([[1, 2, 3, 4]]).T,
        "sparse-row2": sp.csc_matrix([[1, 2, 3, 4]]),
        "sparse-col": sp.csc_array([[1, 2, 3, 4]]).T,
        "sparse-row": sp.csc_array([[1, 2, 3, 4]]),
        "scalar-int": 1,
        "scalar-str": "xxx",
        "None": None,
    }
    result = _check_method_params(X, params=_params, indices=indices)
    indices_ = indices if indices is not None else list(range(X.shape[0]))

    for key in ["sparse-row", "sparse-row2", "scalar-int", "scalar-str", "None"]:
        assert result[key] is _params[key]

    assert result["list"] == _safe_indexing(_params["list"], indices_)
    assert_array_equal(result["array"], _safe_indexing(_params["array"], indices_))
    assert_allclose_dense_sparse(
        result["sparse-col"], _safe_indexing(_params["sparse-col"], indices_)
    )
    assert_allclose_dense_sparse(
        result["sparse-col2"], _safe_indexing(_params["sparse-col2"], indices_)
    )


@pytest.mark.parametrize("convert_to_narwhals", [False, True])
@pytest.mark.parametrize("sp_format", [True, "csr", "csc", "coo", "bsr"])
def test_check_sparse_pandas_sp_format(convert_to_narwhals, sp_format):
    # [FUNCTION DOC]
    # Verifies a fully sparse pandas DataFrame (optionally wrapped by narwhals) converts
    # to a sparse container in the requested format (COO when accept_sparse is True)
    # with equal values.
    # check_array converts pandas.DataFrame with only sparse arrays into sparse matrix
    pd = pytest.importorskip("pandas")
    sp_mat = _sparse_random_matrix(10, 3)

    sdf = pd.DataFrame.sparse.from_spmatrix(sp_mat)
    if convert_to_narwhals:
        sdf = nw.from_native(sdf)
    result = check_array(sdf, accept_sparse=sp_format)

    if sp_format is True:
        # by default pandas converts to coo when accept_sparse is True
        sp_format = "coo"

    assert sp.issparse(result)
    assert result.format == sp_format
    assert_allclose_dense_sparse(sp_mat, result)


def test_check_array_pd_sparse_dataframe_warning():
    # [FUNCTION DOC]
    # Verifies a DataFrame with some sparse columns warns that it will be densified,
    # while a fully sparse DataFrame raises no UserWarning.
    """Test that check_array warns on pandas dataframe with sparse columns."""
    pd = pytest.importorskip("pandas")

    # Warning is raised only when some of the columns are sparse, not all of them.
    # Construct a pandas.DataFrame with first column dense, all others sparse.
    df = pd.DataFrame({"col_0": np.linspace(0, 1, 10)})
    for i in range(1, 4):
        arr = np.zeros(10)
        arr[:4] = np.arange(4)
        arr = pd.arrays.SparseArray(arr, fill_value=0)
        df[f"col_{i}"] = arr

    msg = "pandas.DataFrame with sparse columns found."
    with pytest.warns(UserWarning, match=msg):
        check_array(df, accept_sparse=True)

    # No warning when the whole dataframe is sparse
    df = df.drop(columns="col_0")
    assert hasattr(df, "sparse")

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        check_array(df, accept_sparse=True)


@pytest.mark.parametrize(
    "ntype1, ntype2",
    [
        ("longdouble", "float16"),
        ("float16", "float32"),
        ("float32", "double"),
        ("int16", "int32"),
        ("int32", "long"),
        ("byte", "uint16"),
        ("ushort", "uint32"),
        ("uint32", "uint64"),
        ("uint8", "int8"),
    ],
)
def test_check_pandas_sparse_mixed_dtypes(ntype1, ntype2):
    # [FUNCTION DOC]
    # Verifies check_array accepts fully sparse DataFrames whose two columns have
    # different numeric dtypes (only that no error is raised).
    """Check that pandas.DataFrame having sparse extension arrays with mixed dtypes
    works."""
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame(
        {
            "col1": pd.arrays.SparseArray([0, 1, 0], dtype=ntype1, fill_value=0),
            "col2": pd.arrays.SparseArray([1, 0, 1], dtype=ntype2, fill_value=0),
        }
    )
    check_array(df, accept_sparse=["csr", "csc"])


@pytest.mark.parametrize(
    "ntype1, ntype2, expected_subtype",
    [
        ("double", "longdouble", np.floating),
        ("single", "float32", np.floating),
        ("double", "float64", np.floating),
        ("int8", "byte", np.integer),
        ("short", "int16", np.integer),
        ("intc", "int32", np.integer),
        ("intp", "long", np.integer),
        ("int", "long", np.integer),
        ("int64", "longlong", np.integer),
        ("int_", "intp", np.integer),
        ("ubyte", "uint8", np.unsignedinteger),
        ("uint16", "ushort", np.unsignedinteger),
        ("uintc", "uint32", np.unsignedinteger),
        ("uint", "uint64", np.unsignedinteger),
        ("uintp", "ulonglong", np.unsignedinteger),
    ],
)
def test_check_pandas_sparse_valid(ntype1, ntype2, expected_subtype):
    # [FUNCTION DOC]
    # Verifies fully sparse DataFrames with safely combinable mixed dtypes convert to a
    # sparse result whose dtype is of the expected kind (floating, signed or unsigned).
    # check that we support the conversion of sparse dataframe with mixed
    # type which can be converted safely.
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame(
        {
            "col1": pd.arrays.SparseArray([0, 1, 0], dtype=ntype1, fill_value=0),
            "col2": pd.arrays.SparseArray([1, 0, 1], dtype=ntype2, fill_value=0),
        }
    )
    arr = check_array(df, accept_sparse=["csr", "csc"])
    assert np.issubdtype(arr.dtype, expected_subtype)


@pytest.mark.parametrize(
    "constructor_name",
    ["list", "tuple", "array", "pandas", "sparse_csr", "sparse_csc"],
)
def test_num_features(constructor_name):
    # [FUNCTION DOC]
    # Verifies _num_features returns 3 for 2x3 list, tuple, array, DataFrame, CSR and
    # CSC.
    """Check _num_features for array-likes."""
    X = [[1, 2, 3], [4, 5, 6]]
    X = _convert_container(X, constructor_name)
    assert _num_features(X) == 3


@pytest.mark.parametrize(
    "X",
    [
        [1, 2, 3],
        ["a", "b", "c"],
        [False, True, False],
        [1.0, 3.4, 4.0],
        [{"a": 1}, {"b": 2}, {"c": 3}],
    ],
    ids=["int", "str", "bool", "float", "dict"],
)
@pytest.mark.parametrize("constructor_name", ["list", "tuple", "array", "series"])
def test_num_features_errors_1d_containers(X, constructor_name):
    # [FUNCTION DOC]
    # Verifies _num_features raises TypeError for 1D containers with a message naming
    # the container type plus the shape (for array-likes with shape) or the sample type
    # (str or dict samples).
    X = _convert_container(X, constructor_name)
    if constructor_name == "array":
        expected_type_name = "numpy.ndarray"
    elif constructor_name == "series":
        expected_type_name = "pandas.*Series"
    else:
        expected_type_name = constructor_name
    message = (
        f"Unable to find the number of features from X of type {expected_type_name}"
    )
    if hasattr(X, "shape"):
        message += re.escape(" with shape (3,)")
    elif isinstance(X[0], str):
        message += " where the samples are of type str"
    elif isinstance(X[0], dict):
        message += " where the samples are of type dict"
    with pytest.raises(TypeError, match=message):
        _num_features(X)


@pytest.mark.parametrize("X", [1, "b", False, 3.0], ids=["int", "str", "bool", "float"])
def test_num_features_errors_scalars(X):
    # [FUNCTION DOC]
    # Verifies _num_features raises TypeError naming the type for scalar inputs.
    msg = f"Unable to find the number of features from X of type {type(X).__qualname__}"
    with pytest.raises(TypeError, match=msg):
        _num_features(X)


@pytest.mark.parametrize(
    "names",
    [list(range(2)), range(2), None, [["a", "b"], ["c", "d"]]],
    ids=["list-int", "range", "default", "MultiIndex"],
)
def test_get_feature_names_pandas_with_ints_no_warning(names):
    # [FUNCTION DOC]
    # Verifies _get_feature_names returns None without FutureWarning for DataFrames with
    # integer, range, default or MultiIndex columns.
    """Get feature names with pandas.DataFrames without warning.

    Column names with consistent dtypes will not warn, such as int or MultiIndex.
    """
    pd = pytest.importorskip("pandas")
    X = pd.DataFrame([[1, 2], [4, 5], [5, 6]], columns=names)

    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        names = _get_feature_names(X)
    assert names is None


@pytest.mark.parametrize(
    "constructor_name, minversion",
    [("pyarrow", "13.0.0"), ("pandas", "1.5.0"), ("polars", "0.18.2")],
)
def test_get_feature_names_4_dataframes(constructor_name, minversion):
    # [FUNCTION DOC]
    # Verifies _get_feature_names returns string column names for pyarrow, pandas and
    # polars tables (skipped below the given minimum versions).
    """Test _get_features_names on dataframes."""
    data = [[1, 4, 2], [3, 3, 6]]
    columns = ["col_0", "col_1", "col_2"]
    df = _convert_container(
        data, constructor_name, column_names=columns, minversion=minversion
    )
    feature_names = _get_feature_names(df)

    assert_array_equal(feature_names, columns)


def test_get_feature_names_numpy():
    # [FUNCTION DOC]
    # Verifies _get_feature_names returns None for numpy arrays.
    """Get feature names return None for numpy arrays."""
    X = np.array([[1, 2, 3], [4, 5, 6]])
    names = _get_feature_names(X)
    assert names is None


@pytest.mark.parametrize(
    "names, dtypes",
    [
        (["a", 1], "['int', 'str']"),
        (["pizza", ("a", "b")], "['str', 'tuple']"),
    ],
    ids=["str-int", "str-tuple"],
)
def test_get_feature_names_invalid_dtypes(names, dtypes):
    # [FUNCTION DOC]
    # Verifies _get_feature_names raises TypeError with the sorted list of name types
    # when column names mix strings with ints or tuples.
    """Get feature names errors when the feature names have mixed dtypes"""
    pd = pytest.importorskip("pandas")
    X = pd.DataFrame([[1, 2], [4, 5], [5, 6]], columns=names)

    msg = re.escape(
        "Feature names are only supported if all input features have string names, "
        f"but your input has {dtypes} as feature name / column name types. "
        "If you want feature names to be stored and validated, you must convert "
        "them all to strings, by using X.columns = X.columns.astype(str) for "
        "example. Otherwise you can remove feature / column names from your input "
        "data, or convert them all to a non-string data type."
    )
    with pytest.raises(TypeError, match=msg):
        names = _get_feature_names(X)


class PassthroughTransformer(BaseEstimator):
    # [CLASS DOC]
    # HELPER: Minimal transformer that validates X in fit (recording n_features_in_ and
    # feature_names_in_), returns X unchanged in transform and delegates
    # get_feature_names_out to _check_feature_names_in.
    def fit(self, X, y=None):
        # [FUNCTION DOC]
        # HELPER: Calls validate_data with reset=True; returns self. Side effects: sets
        # n_features_in_ and possibly feature_names_in_.
        validate_data(self, X, reset=True)
        return self

    def transform(self, X):
        # [FUNCTION DOC]
        # HELPER: Returns X unchanged.
        return X

    def get_feature_names_out(self, input_features=None):
        # [FUNCTION DOC]
        # HELPER: Returns _check_feature_names_in(self, input_features) (names generated
        # when needed).
        return _check_feature_names_in(self, input_features)


@pytest.mark.parametrize(
    ["constructor_name", "feature_names", "msg"],
    [
        ("array", ["x0", "x1", "x2"], "input_features should have length equal to"),
        ("pandas", ["a", "b", "c"], "input_features is not equal to"),
        ("polars", ["a", "b", "c"], "input_features is not equal to"),
    ],
)
def test_check_feature_names_in(constructor_name, feature_names, msg):
    # [FUNCTION DOC]
    # Verifies _check_feature_names_in via PassthroughTransformer: generated "x" names
    # for arrays and column names for pandas/polars; wrong input_features raise the
    # length or equality message; after deleting n_features_in_, arrays cannot generate
    # names while dataframes still return stored names.
    """Check behavior of check_feature_names_in for arrays."""
    X = np.array([[0.0, 1.0, 2.0]])
    X = _convert_container(X, constructor_name, column_names=["a", "b", "c"])
    est = PassthroughTransformer().fit(X)

    names = est.get_feature_names_out()
    assert_array_equal(names, feature_names)

    incorrect_len_names = feature_names[:2]
    with pytest.raises(ValueError, match=msg):
        est.get_feature_names_out(incorrect_len_names)

    # remove n_feature_in_
    del est.n_features_in_
    if constructor_name == "array":
        with pytest.raises(ValueError, match="Unable to generate feature names"):
            est.get_feature_names_out()
    else:
        assert_array_equal(est.get_feature_names_out(), feature_names)


def test_check_response_method_unknown_method():
    # [FUNCTION DOC]
    # Verifies _check_response_method raises AttributeError naming the estimator and the
    # unknown method.
    """Check the error message when passing an unknown response method."""
    err_msg = (
        "RandomForestRegressor has none of the following attributes: unknown_method."
    )
    with pytest.raises(AttributeError, match=err_msg):
        _check_response_method(RandomForestRegressor(), "unknown_method")


@pytest.mark.parametrize(
    "response_method", ["decision_function", "predict_proba", "predict"]
)
def test_check_response_method_not_supported_response_method(response_method):
    # [FUNCTION DOC]
    # Verifies AttributeError for prediction methods an estimator (EstimatorWithFit)
    # lacks.
    """Check the error message when a response method is not supported by the
    estimator."""
    err_msg = (
        f"EstimatorWithFit has none of the following attributes: {response_method}."
    )
    with pytest.raises(AttributeError, match=err_msg):
        _check_response_method(EstimatorWithFit(), response_method)


def test_check_response_method_list_str():
    # [FUNCTION DOC]
    # Verifies a list of method names: error when none exist, otherwise the first
    # existing method in list order is returned (checked through the mock's returned
    # method name).
    """Check that we can pass a list of ordered method."""
    method_implemented = ["predict_proba"]
    my_estimator = _MockEstimatorOnOffPrediction(method_implemented)

    X = "mocking_data"

    # raise an error when no methods are defined
    response_method = ["decision_function", "predict"]
    err_msg = (
        "_MockEstimatorOnOffPrediction has none of the following attributes: "
        f"{', '.join(response_method)}."
    )
    with pytest.raises(AttributeError, match=err_msg):
        _check_response_method(my_estimator, response_method)(X)

    # check that we don't get issue when one of the method is defined
    response_method = ["decision_function", "predict_proba"]
    method_name_predicting = _check_response_method(my_estimator, response_method)(X)
    assert method_name_predicting == "predict_proba"

    # check the order of the methods returned
    method_implemented = ["predict_proba", "predict"]
    my_estimator = _MockEstimatorOnOffPrediction(method_implemented)
    response_method = ["decision_function", "predict", "predict_proba"]
    method_name_predicting = _check_response_method(my_estimator, response_method)(X)
    assert method_name_predicting == "predict"


def test_boolean_series_remains_boolean():
    # [FUNCTION DOC]
    # Regression test: a bool pandas Series stays bool through check_array.
    """Regression test for gh-25145"""
    pd = pytest.importorskip("pandas")
    res = check_array(pd.Series([True, False]), ensure_2d=False)
    expected = np.array([True, False])

    assert res.dtype == expected.dtype
    assert_array_equal(res, expected)


@pytest.mark.parametrize("input_values", [[0, 1, 0, 1, 0, np.nan], [0, 1, 0, 1, 0, 1]])
def test_pandas_array_returns_ndarray(input_values):
    # [FUNCTION DOC]
    # Verifies a nullable Int32 pandas array (with or without NA) converts to a floating
    # ndarray with matching values when dtype=None.
    """Check pandas array with extensions dtypes returns a numeric ndarray.

    Non-regression test for gh-25637.
    """
    pd = pytest.importorskip("pandas")
    input_series = pd.array(input_values, dtype="Int32")
    result = check_array(
        input_series,
        dtype=None,
        ensure_2d=False,
        allow_nd=False,
        ensure_all_finite=False,
    )
    assert np.issubdtype(result.dtype.kind, np.floating)
    assert_allclose(result, input_values)


@skip_if_array_api_compat_not_configured
def test_check_array_array_api_has_non_finite():
    # [FUNCTION DOC]
    # Verifies NaN and inf detection in check_array for array_api_strict float32 arrays
    # under Array API dispatch.
    """Checks that Array API arrays checks non-finite correctly."""
    xp = pytest.importorskip("array_api_strict")

    X_nan = xp.asarray([[xp.nan, 1, 0], [0, xp.nan, 3]], dtype=xp.float32)
    with config_context(array_api_dispatch=True):
        with pytest.raises(ValueError, match="Input contains NaN."):
            check_array(X_nan)

    X_inf = xp.asarray([[xp.inf, 1, 0], [0, xp.inf, 3]], dtype=xp.float32)
    with config_context(array_api_dispatch=True):
        with pytest.raises(ValueError, match="infinity or a value too large"):
            check_array(X_inf)


@pytest.mark.parametrize(
    "extension_dtype, regular_dtype",
    [
        ("boolean", "bool"),
        ("Int64", "int64"),
        ("Float64", "float64"),
        ("category", "object"),
    ],
)
@pytest.mark.parametrize("include_object", [True, False])
def test_check_array_multiple_extensions(
    extension_dtype, regular_dtype, include_object
):
    # [FUNCTION DOC]
    # Verifies DataFrames using pandas extension dtypes (boolean, Int64, Float64,
    # category) produce the same check_array output as their regular-dtype counterparts,
    # with and without an extra object column.
    """Check pandas extension arrays give the same result as non-extension arrays."""
    pd = pytest.importorskip("pandas")
    X_regular = pd.DataFrame(
        {
            "a": pd.Series([1, 0, 1, 0], dtype=regular_dtype),
            "c": pd.Series([9, 8, 7, 6], dtype="int64"),
        }
    )
    if include_object:
        X_regular["b"] = pd.Series(["a", "b", "c", "d"], dtype="object")

    X_extension = X_regular.assign(a=X_regular["a"].astype(extension_dtype))

    X_regular_checked = check_array(X_regular, dtype=None)
    X_extension_checked = check_array(X_extension, dtype=None)
    assert_array_equal(X_regular_checked, X_extension_checked)


@pytest.mark.parametrize(
    "sparse_container",
    CSR_CONTAINERS + CSC_CONTAINERS + COO_CONTAINERS + DIA_CONTAINERS,
)
@pytest.mark.parametrize("output_format", ["csr", "csc", "coo"])
def test_check_array_dia_to_int32_indexed_csr_csc_coo(sparse_container, output_format):
    # [FUNCTION DOC]
    # Verifies converting int32-indexed CSR/CSC/COO/DIA inputs to csr/csc/coo keeps
    # int32 index arrays (exercises the DIA index dtype preservation workaround).
    # KNOWN ISSUE: for COO inputs only row is cast to int32 (col keeps its dtype), so
    # the input setup is not fully int32 as the comment implies.
    """Check the consistency of the indices dtype with sparse matrices/arrays."""
    X = sparse_container([[0, 1], [1, 0]], dtype=np.float64)

    # Explicitly set the dtype of the indexing arrays
    if hasattr(X, "offsets"):  # DIA matrix
        X.offsets = X.offsets.astype(np.int32)
    elif hasattr(X, "row") and hasattr(X, "col"):  # COO matrix
        X.row = X.row.astype(np.int32)
    elif hasattr(X, "indices") and hasattr(X, "indptr"):  # CSR or CSC matrix
        X.indices = X.indices.astype(np.int32)
        X.indptr = X.indptr.astype(np.int32)

    X_checked = check_array(X, accept_sparse=output_format)
    if output_format == "coo":
        assert X_checked.row.dtype == np.int32
        assert X_checked.col.dtype == np.int32
    else:  # output_format in ["csr", "csc"]
        assert X_checked.indices.dtype == np.int32
        assert X_checked.indptr.dtype == np.int32


@pytest.mark.parametrize("sequence", [[np.array(1), np.array(2)], [[1, 2], [3, 4]]])
def test_to_object_array(sequence):
    # [FUNCTION DOC]
    # Verifies _to_object_array returns a 1D object ndarray even for nested sequences or
    # lists of 0-d arrays.
    out = _to_object_array(sequence)
    assert isinstance(out, np.ndarray)
    assert out.dtype.kind == "O"
    assert out.ndim == 1


def test_column_or_1d():
    # [FUNCTION DOC]
    # Verifies column_or_1d ravels binary/multiclass/continuous targets (including
    # column vectors) and raises ValueError for multilabel and multioutput shapes.
    EXAMPLES = [
        ("binary", ["spam", "egg", "spam"]),
        ("binary", [0, 1, 0, 1]),
        ("continuous", np.arange(10) / 20.0),
        ("multiclass", [1, 2, 3]),
        ("multiclass", [0, 1, 2, 2, 0]),
        ("multiclass", [[1], [2], [3]]),
        ("multilabel-indicator", [[0, 1, 0], [0, 0, 1]]),
        ("multiclass-multioutput", [[1, 2, 3]]),
        ("multiclass-multioutput", [[1, 1], [2, 2], [3, 1]]),
        ("multiclass-multioutput", [[5, 1], [4, 2], [3, 1]]),
        ("multiclass-multioutput", [[1, 2, 3]]),
        ("continuous-multioutput", np.arange(30).reshape((-1, 3))),
    ]

    for y_type, y in EXAMPLES:
        if y_type in ["binary", "multiclass", "continuous"]:
            assert_array_equal(column_or_1d(y), np.ravel(y))
        else:
            with pytest.raises(ValueError):
                column_or_1d(y)


def test_check_array_writeable_np():
    # [FUNCTION DOC]
    # Verifies force_writeable: a writeable ndarray is returned without copy; a
    # read-only one is copied into a writeable array. Side effects: unseeded global RNG.
    """Check the behavior of check_array when a writeable array is requested
    without copy if possible, on numpy arrays.
    """
    X = np.random.uniform(size=(10, 10))

    out = check_array(X, copy=False, force_writeable=True)
    # X is already writeable, no copy is needed
    assert np.may_share_memory(out, X)
    assert out.flags.writeable

    X.flags.writeable = False

    out = check_array(X, copy=False, force_writeable=True)
    # X is not writeable, a copy is made
    assert not np.may_share_memory(out, X)
    assert out.flags.writeable


def test_check_array_writeable_mmap():
    # [FUNCTION DOC]
    # Verifies force_writeable on memmaps: writeable mmap returned without copy,
    # read-only mmap copied. Side effects: temporary memmap files, unseeded global RNG.
    """Check the behavior of check_array when a writeable array is requested
    without copy if possible, on a memory-map.

    A common situation is when a meta-estimators run in parallel using multiprocessing
    with joblib, which creates read-only memory-maps of large arrays.
    """
    X = np.random.uniform(size=(10, 10))

    mmap = create_memmap_backed_data(X, mmap_mode="w+")
    out = check_array(mmap, copy=False, force_writeable=True)
    # mmap is already writeable, no copy is needed
    assert np.may_share_memory(out, mmap)
    assert out.flags.writeable

    mmap = create_memmap_backed_data(X, mmap_mode="r")
    out = check_array(mmap, copy=False, force_writeable=True)
    # mmap is read-only, a copy is made
    assert not np.may_share_memory(out, mmap)
    assert out.flags.writeable


def test_check_array_writeable_df():
    # [FUNCTION DOC]
    # Verifies force_writeable on DataFrames: no copy when backed by a writeable array,
    # a copy when backed by a read-only array.
    """Check the behavior of check_array when a writeable array is requested
    without copy if possible, on a dataframe.
    """
    pd = pytest.importorskip("pandas")

    X = np.random.uniform(size=(10, 10))
    df = pd.DataFrame(X, copy=False)

    out = check_array(df, copy=False, force_writeable=True)
    # df is backed by a writeable array, no copy is needed
    assert np.may_share_memory(out, df)
    assert out.flags.writeable

    X.flags.writeable = False
    df = pd.DataFrame(X, copy=False)

    out = check_array(df, copy=False, force_writeable=True)
    # df is backed by a read-only array, a copy is made
    assert not np.may_share_memory(out, df)
    assert out.flags.writeable


@skip_if_array_api_compat_not_configured
def test_check_array_on_sparse_inputs_with_array_api_enabled():
    # [FUNCTION DOC]
    # Verifies sparse input is still accepted (and rejected without accept_sparse) when
    # Array API dispatch is enabled.
    X_sp = sp.csr_array([[0, 1, 0], [1, 0, 1]])
    with config_context(array_api_dispatch=True):
        assert sp.issparse(check_array(X_sp, accept_sparse=True))

        with pytest.raises(TypeError):
            check_array(X_sp)


@pytest.mark.parametrize(
    ["X", "estimator", "expected_error_message"],
    [
        (
            np.array([[[1, 2], [3, 4]], [[1, 2], [3, 4]]]),
            RandomForestRegressor(),
            "Found array with dim 3, while dim <= 2 is required by "
            "RandomForestRegressor.",
        ),
        (
            np.array([[[1, 2], [3, 4]], [[1, 2], [3, 4]]]),
            None,
            "Found array with dim 3, while dim <= 2 is required.",
        ),
    ],
)
def test_check_array_allow_nd_errors(X, estimator, expected_error_message):
    # [FUNCTION DOC]
    # Verifies the 3D-input error message with and without an estimator name suffix.
    with pytest.raises(ValueError, match=expected_error_message):
        check_array(X, estimator=estimator)


@pytest.mark.parametrize(
    ["categorical_features", "expected_msg"],
    [
        (
            [b"hello", b"world"],
            re.escape(
                "categorical_features must be an array-like of bool, int or str, "
                "got: bytes40."
            ),
        ),
        (
            np.array([b"hello", 1.3], dtype=object),
            re.escape(
                "categorical_features must be an array-like of bool, int or str, "
                "got: bytes, float."
            ),
        ),
        (
            [0, -1],
            re.escape(
                "categorical_features set as integer indices must be in "
                "[0, n_features - 1]"
            ),
        ),
        (
            [True, True, False, False, True],
            re.escape(
                "categorical_features set as a boolean mask must have shape "
                "(n_features,)"
            ),
        ),
    ],
)
def test_check_categorical_features_raises(categorical_features, expected_msg):
    # [FUNCTION DOC]
    # Verifies _check_categorical_features errors for bytes dtype, mixed object types,
    # negative indices and wrong-length boolean masks on a 10x10 integer array.
    """Test that check_categorical_features raises expected errors."""
    rng = np.random.RandomState(0)
    n_samples, n_features = 10, 10
    X = rng.randint(0, 3, size=(n_samples, n_features))

    with pytest.raises(ValueError, match=expected_msg):
        _check_categorical_features(X, categorical_features)


@pytest.mark.parametrize(
    ["categorical_features", "on_array"],
    [
        ([False, True, True, False], True),
        ([1, 2], True),
        (["b", "c"], False),
        ("from_dtype", False),
    ],
)
@pytest.mark.parametrize("constructor_name", ["array", "pandas", "polars"])
def test_check_categorical_features(categorical_features, on_array, constructor_name):
    # [FUNCTION DOC]
    # Verifies _check_categorical_features returns mask [F, T, T, F] for a bool mask,
    # int indices, column names and "from_dtype" on arrays/pandas/polars inputs with b
    # and c categorical; name-based and dtype-based specs are skipped (early return) for
    # arrays.
    """Test that check_categorical_features returns as expected on simple data."""
    rng = np.random.RandomState(0)
    n_samples, n_features = 30, 4
    X = rng.randint(0, 3, size=(n_samples, n_features))
    if constructor_name == "array" and not on_array:
        return
    elif constructor_name == "polars":
        X = X.astype(str)
    X = _convert_container(
        X,
        constructor_name,
        column_names=["a", "b", "c", "d"],
        categorical_feature_names=["b", "c"],
    )

    result = _check_categorical_features(X, categorical_features)
    assert_allclose(result, [False, True, True, False])


def test_num_samples_pa_chunked_array():
    # [FUNCTION DOC]
    # Regression test: _num_samples returns 3 for a pyarrow ChunkedArray.
    # https://github.com/scikit-learn/scikit-learn/issues/33993
    pytest.importorskip("pyarrow")
    from pyarrow import chunked_array

    result = _num_samples(chunked_array([[0.1, 0.2, 0.3]]))
    assert result == 3
