# [MODULE DOC]
# Tests for sklearn.utils.sparsefuncs and sklearn.utils.sparsefuncs_fast: (incremental,
# weighted, NaN-aware) mean/variance along an axis, row/column scaling and swapping,
# min/max, nonzero counting, CSC column medians, row normalization and norms, row
# assignment, the implicitly column-centered LinearOperator, and sparse_matmul_to_dense.
# Results are compared to dense NumPy equivalents for CSR/CSC matrix and array
# containers.
import numpy as np
import pytest
import scipy.sparse as sp
from numpy.random import RandomState
from numpy.testing import assert_array_almost_equal, assert_array_equal
from scipy import linalg

from sklearn.datasets import make_classification
from sklearn.utils._testing import assert_allclose
from sklearn.utils.fixes import (
    CSC_CONTAINERS,
    CSR_CONTAINERS,
    LIL_CONTAINERS,
    _sparse_eye_array,
    _sparse_random_array,
)
from sklearn.utils.sparsefuncs import (
    _implicit_column_offset,
    count_nonzero,
    csc_median_axis_0,
    incr_mean_variance_axis,
    inplace_column_scale,
    inplace_row_scale,
    inplace_swap_column,
    inplace_swap_row,
    mean_variance_axis,
    min_max_axis,
    sparse_matmul_to_dense,
)
from sklearn.utils.sparsefuncs_fast import (
    assign_rows_csr,
    csr_row_norms,
    inplace_csr_row_normalize_l1,
    inplace_csr_row_normalize_l2,
)


@pytest.mark.parametrize("csc_container", CSC_CONTAINERS)
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
@pytest.mark.parametrize("lil_container", LIL_CONTAINERS)
def test_mean_variance_axis0(csc_container, csr_container, lil_container):
    # [FUNCTION DOC]
    # Verifies mean_variance_axis along axis 0 on a small sparsified 5x4 dataset:
    # TypeError for LIL input; for CSR and CSC inputs of float32, float64, int32 and
    # int64 dtypes, output dtype is float32 for float32 input and float64 otherwise, and
    # values match np.mean/np.var.
    # Inputs: csc_container, csr_container, lil_container - sparse constructors.
    X, _ = make_classification(5, 4, random_state=0)
    # Sparsify the array a little bit
    X[0, 0] = 0
    X[2, 1] = 0
    X[4, 3] = 0
    X_lil = lil_container(X)
    X_lil[1, 0] = 0
    X[1, 0] = 0

    with pytest.raises(TypeError):
        mean_variance_axis(X_lil, axis=0)

    X_csr = csr_container(X_lil)
    X_csc = csc_container(X_lil)

    expected_dtypes = [
        (np.float32, np.float32),
        (np.float64, np.float64),
        (np.int32, np.float64),
        (np.int64, np.float64),
    ]

    for input_dtype, output_dtype in expected_dtypes:
        X_test = X.astype(input_dtype)
        for X_sparse in (X_csr, X_csc):
            X_sparse = X_sparse.astype(input_dtype)
            X_means, X_vars = mean_variance_axis(X_sparse, axis=0)
            assert X_means.dtype == output_dtype
            assert X_vars.dtype == output_dtype
            assert_array_almost_equal(X_means, np.mean(X_test, axis=0))
            assert_array_almost_equal(X_vars, np.var(X_test, axis=0))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("sparse_constructor", CSC_CONTAINERS + CSR_CONTAINERS)
def test_mean_variance_axis0_precision(dtype, sparse_constructor):
    # [FUNCTION DOC]
    # Non-regression (#19766): a 1000x1 column of constant 100 with 10 NaNs and random
    # positive weights must yield a weighted variance below machine epsilon of the
    # dtype, i.e. no catastrophic precision loss.
    # Inputs: dtype - float32/float64; sparse_constructor - CSR or CSC.
    # Check that there's no big loss of precision when the real variance is
    # exactly 0. (#19766)
    rng = np.random.RandomState(0)
    X = np.full(fill_value=100.0, shape=(1000, 1), dtype=dtype)
    # Add some missing records which should be ignored:
    missing_indices = rng.choice(np.arange(X.shape[0]), 10, replace=False)
    X[missing_indices, 0] = np.nan
    X = sparse_constructor(X)

    # Random positive weights:
    sample_weight = rng.rand(X.shape[0]).astype(dtype)

    _, var = mean_variance_axis(X, weights=sample_weight, axis=0)

    assert var < np.finfo(dtype).eps


@pytest.mark.parametrize("csc_container", CSC_CONTAINERS)
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
@pytest.mark.parametrize("lil_container", LIL_CONTAINERS)
def test_mean_variance_axis1(csc_container, csr_container, lil_container):
    # [FUNCTION DOC]
    # Intended to verify mean_variance_axis along axis 1: checks TypeError for LIL input
    # with axis=1, then dtype and value checks.
    # KNOWN ISSUE: the value/dtype loop calls mean_variance_axis with axis=0 and
    # compares to axis-0 NumPy stats, so axis=1 results are never actually checked; the
    # test duplicates test_mean_variance_axis0.
    X, _ = make_classification(5, 4, random_state=0)
    # Sparsify the array a little bit
    X[0, 0] = 0
    X[2, 1] = 0
    X[4, 3] = 0
    X_lil = lil_container(X)
    X_lil[1, 0] = 0
    X[1, 0] = 0

    with pytest.raises(TypeError):
        mean_variance_axis(X_lil, axis=1)

    X_csr = csr_container(X_lil)
    X_csc = csc_container(X_lil)

    expected_dtypes = [
        (np.float32, np.float32),
        (np.float64, np.float64),
        (np.int32, np.float64),
        (np.int64, np.float64),
    ]

    for input_dtype, output_dtype in expected_dtypes:
        X_test = X.astype(input_dtype)
        for X_sparse in (X_csr, X_csc):
            X_sparse = X_sparse.astype(input_dtype)
            X_means, X_vars = mean_variance_axis(X_sparse, axis=0)
            assert X_means.dtype == output_dtype
            assert X_vars.dtype == output_dtype
            assert_array_almost_equal(X_means, np.mean(X_test, axis=0))
            assert_array_almost_equal(X_vars, np.var(X_test, axis=0))


@pytest.mark.parametrize(
    ["Xw", "X", "weights"],
    [
        ([[0, 0, 1], [0, 2, 3]], [[0, 0, 1], [0, 2, 3]], [1, 1, 1]),
        ([[0, 0, 1], [0, 1, 1]], [[0, 0, 0, 1], [0, 1, 1, 1]], [1, 2, 1]),
        ([[0, 0, 1], [0, 1, 1]], [[0, 0, 1], [0, 1, 1]], None),
        (
            [[0, np.nan, 2], [0, np.nan, np.nan]],
            [[0, np.nan, 2], [0, np.nan, np.nan]],
            [1.0, 1.0, 1.0],
        ),
        (
            [[0, 0], [1, np.nan], [2, 0], [0, 3], [np.nan, np.nan], [np.nan, 2]],
            [
                [0, 0, 0],
                [1, 1, np.nan],
                [2, 2, 0],
                [0, 0, 3],
                [np.nan, np.nan, np.nan],
                [np.nan, np.nan, 2],
            ],
            [2.0, 1.0],
        ),
        (
            [[1, 0, 1], [0, 3, 1]],
            [[1, 0, 0, 0, 1], [0, 3, 3, 3, 1]],
            np.array([1, 3, 1]),
        ),
    ],
)
@pytest.mark.parametrize("sparse_constructor", CSC_CONTAINERS + CSR_CONTAINERS)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_incr_mean_variance_axis_weighted_axis1(
    Xw, X, weights, sparse_constructor, dtype
):
    # [FUNCTION DOC]
    # Verifies that weighted incr_mean_variance_axis along axis 1 on Xw with
    # integer/float column weights equals the unweighted result on X where columns are
    # repeated according to the weights (NaNs ignored); also equals mean_variance_axis
    # on X; checks output dtypes equal the input dtype, and repeats for a second
    # incremental round starting from the first round's statistics.
    # Inputs: Xw, X, weights - parametrized equivalent pairs; sparse_constructor; dtype.
    axis = 1
    Xw_sparse = sparse_constructor(Xw).astype(dtype)
    X_sparse = sparse_constructor(X).astype(dtype)

    last_mean = np.zeros(np.shape(Xw)[0], dtype=dtype)
    last_var = np.zeros_like(last_mean, dtype=dtype)
    last_n = np.zeros_like(last_mean, dtype=np.int64)
    means0, vars0, n_incr0 = incr_mean_variance_axis(
        X=X_sparse,
        axis=axis,
        last_mean=last_mean,
        last_var=last_var,
        last_n=last_n,
        weights=None,
    )

    means_w0, vars_w0, n_incr_w0 = incr_mean_variance_axis(
        X=Xw_sparse,
        axis=axis,
        last_mean=last_mean,
        last_var=last_var,
        last_n=last_n,
        weights=weights,
    )

    assert means_w0.dtype == dtype
    assert vars_w0.dtype == dtype
    assert n_incr_w0.dtype == dtype

    means_simple, vars_simple = mean_variance_axis(X=X_sparse, axis=axis)

    assert_array_almost_equal(means0, means_w0)
    assert_array_almost_equal(means0, means_simple)
    assert_array_almost_equal(vars0, vars_w0)
    assert_array_almost_equal(vars0, vars_simple)
    assert_array_almost_equal(n_incr0, n_incr_w0)

    # check second round for incremental
    means1, vars1, n_incr1 = incr_mean_variance_axis(
        X=X_sparse,
        axis=axis,
        last_mean=means0,
        last_var=vars0,
        last_n=n_incr0,
        weights=None,
    )

    means_w1, vars_w1, n_incr_w1 = incr_mean_variance_axis(
        X=Xw_sparse,
        axis=axis,
        last_mean=means_w0,
        last_var=vars_w0,
        last_n=n_incr_w0,
        weights=weights,
    )

    assert_array_almost_equal(means1, means_w1)
    assert_array_almost_equal(vars1, vars_w1)
    assert_array_almost_equal(n_incr1, n_incr_w1)

    assert means_w1.dtype == dtype
    assert vars_w1.dtype == dtype
    assert n_incr_w1.dtype == dtype


@pytest.mark.parametrize(
    ["Xw", "X", "weights"],
    [
        ([[0, 0, 1], [0, 2, 3]], [[0, 0, 1], [0, 2, 3]], [1, 1]),
        ([[0, 0, 1], [0, 1, 1]], [[0, 0, 1], [0, 1, 1], [0, 1, 1]], [1, 2]),
        ([[0, 0, 1], [0, 1, 1]], [[0, 0, 1], [0, 1, 1]], None),
        (
            [[0, np.nan, 2], [0, np.nan, np.nan]],
            [[0, np.nan, 2], [0, np.nan, np.nan]],
            [1.0, 1.0],
        ),
        (
            [[0, 0, 1, np.nan, 2, 0], [0, 3, np.nan, np.nan, np.nan, 2]],
            [
                [0, 0, 1, np.nan, 2, 0],
                [0, 0, 1, np.nan, 2, 0],
                [0, 3, np.nan, np.nan, np.nan, 2],
            ],
            [2.0, 1.0],
        ),
        (
            [[1, 0, 1], [0, 0, 1]],
            [[1, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1]],
            np.array([1, 3]),
        ),
    ],
)
@pytest.mark.parametrize("sparse_constructor", CSC_CONTAINERS + CSR_CONTAINERS)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_incr_mean_variance_axis_weighted_axis0(
    Xw, X, weights, sparse_constructor, dtype
):
    # [FUNCTION DOC]
    # Same as the axis-1 variant but along axis 0: weighted rows of Xw are equivalent to
    # rows repeated in X; checks first-round equality with mean_variance_axis and second
    # incremental round consistency, plus output dtypes.
    # Inputs: Xw, X, weights; sparse_constructor; dtype.
    axis = 0
    Xw_sparse = sparse_constructor(Xw).astype(dtype)
    X_sparse = sparse_constructor(X).astype(dtype)

    last_mean = np.zeros(np.size(Xw, 1), dtype=dtype)
    last_var = np.zeros_like(last_mean)
    last_n = np.zeros_like(last_mean, dtype=np.int64)
    means0, vars0, n_incr0 = incr_mean_variance_axis(
        X=X_sparse,
        axis=axis,
        last_mean=last_mean,
        last_var=last_var,
        last_n=last_n,
        weights=None,
    )

    means_w0, vars_w0, n_incr_w0 = incr_mean_variance_axis(
        X=Xw_sparse,
        axis=axis,
        last_mean=last_mean,
        last_var=last_var,
        last_n=last_n,
        weights=weights,
    )

    assert means_w0.dtype == dtype
    assert vars_w0.dtype == dtype
    assert n_incr_w0.dtype == dtype

    means_simple, vars_simple = mean_variance_axis(X=X_sparse, axis=axis)

    assert_array_almost_equal(means0, means_w0)
    assert_array_almost_equal(means0, means_simple)
    assert_array_almost_equal(vars0, vars_w0)
    assert_array_almost_equal(vars0, vars_simple)
    assert_array_almost_equal(n_incr0, n_incr_w0)

    # check second round for incremental
    means1, vars1, n_incr1 = incr_mean_variance_axis(
        X=X_sparse,
        axis=axis,
        last_mean=means0,
        last_var=vars0,
        last_n=n_incr0,
        weights=None,
    )

    means_w1, vars_w1, n_incr_w1 = incr_mean_variance_axis(
        X=Xw_sparse,
        axis=axis,
        last_mean=means_w0,
        last_var=vars_w0,
        last_n=n_incr_w0,
        weights=weights,
    )

    assert_array_almost_equal(means1, means_w1)
    assert_array_almost_equal(vars1, vars_w1)
    assert_array_almost_equal(n_incr1, n_incr_w1)

    assert means_w1.dtype == dtype
    assert vars_w1.dtype == dtype
    assert n_incr_w1.dtype == dtype


@pytest.mark.parametrize("csc_container", CSC_CONTAINERS)
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
@pytest.mark.parametrize("lil_container", LIL_CONTAINERS)
def test_incr_mean_variance_axis(csc_container, csr_container, lil_container):
    # [FUNCTION DOC]
    # For axis 0 and 1, on random 0/1 data: checks TypeError for nonsensical positional
    # arguments and for LIL input; checks that a single-row (or column) input gives the
    # same means/vars as mean_variance_axis with n equal to X.shape[axis] for CSR and
    # CSC; then for the full stacked data and several input dtypes verifies output dtype
    # (float32 for float32 input, else float64) and values.
    # Inputs: csc_container, csr_container, lil_container.
    for axis in [0, 1]:
        rng = np.random.RandomState(0)
        n_features = 50
        n_samples = 10
        if axis == 0:
            data_chunks = [rng.randint(0, 2, size=n_features) for i in range(n_samples)]
        else:
            data_chunks = [rng.randint(0, 2, size=n_samples) for i in range(n_features)]

        # default params for incr_mean_variance
        last_mean = np.zeros(n_features) if axis == 0 else np.zeros(n_samples)
        last_var = np.zeros_like(last_mean)
        last_n = np.zeros_like(last_mean, dtype=np.int64)

        # Test errors
        X = np.array(data_chunks[0])
        X = np.atleast_2d(X)
        X = X.T if axis == 1 else X
        X_lil = lil_container(X)
        X_csr = csr_container(X_lil)

        with pytest.raises(TypeError):
            incr_mean_variance_axis(
                X=axis, axis=last_mean, last_mean=last_var, last_var=last_n
            )
        with pytest.raises(TypeError):
            incr_mean_variance_axis(
                X_lil, axis=axis, last_mean=last_mean, last_var=last_var, last_n=last_n
            )

        # Test _incr_mean_and_var with a 1 row input
        X_means, X_vars = mean_variance_axis(X_csr, axis)
        X_means_incr, X_vars_incr, n_incr = incr_mean_variance_axis(
            X_csr, axis=axis, last_mean=last_mean, last_var=last_var, last_n=last_n
        )
        assert_array_almost_equal(X_means, X_means_incr)
        assert_array_almost_equal(X_vars, X_vars_incr)
        # X.shape[axis] picks # samples
        assert_array_equal(X.shape[axis], n_incr)

        X_csc = csc_container(X_lil)
        X_means, X_vars = mean_variance_axis(X_csc, axis)
        assert_array_almost_equal(X_means, X_means_incr)
        assert_array_almost_equal(X_vars, X_vars_incr)
        assert_array_equal(X.shape[axis], n_incr)

        # Test _incremental_mean_and_var with whole data
        X = np.vstack(data_chunks)
        X = X.T if axis == 1 else X
        X_lil = lil_container(X)
        X_csr = csr_container(X_lil)
        X_csc = csc_container(X_lil)

        expected_dtypes = [
            (np.float32, np.float32),
            (np.float64, np.float64),
            (np.int32, np.float64),
            (np.int64, np.float64),
        ]

        for input_dtype, output_dtype in expected_dtypes:
            for X_sparse in (X_csr, X_csc):
                X_sparse = X_sparse.astype(input_dtype)
                last_mean = last_mean.astype(output_dtype)
                last_var = last_var.astype(output_dtype)
                X_means, X_vars = mean_variance_axis(X_sparse, axis)
                X_means_incr, X_vars_incr, n_incr = incr_mean_variance_axis(
                    X_sparse,
                    axis=axis,
                    last_mean=last_mean,
                    last_var=last_var,
                    last_n=last_n,
                )
                assert X_means_incr.dtype == output_dtype
                assert X_vars_incr.dtype == output_dtype
                assert_array_almost_equal(X_means, X_means_incr)
                assert_array_almost_equal(X_vars, X_vars_incr)
                assert_array_equal(X.shape[axis], n_incr)


@pytest.mark.parametrize("sparse_constructor", CSC_CONTAINERS + CSR_CONTAINERS)
def test_incr_mean_variance_axis_dim_mismatch(sparse_constructor):
    # [FUNCTION DOC]
    # Non-regression for PR #18655: on a 60x4 random matrix, axis 0 matches NumPy stats;
    # axis=1 with last_mean sized to n_features raises ValueError; inconsistent sizes of
    # last_mean, last_var, last_n raise ValueError.
    # Inputs: sparse_constructor - CSR or CSC.
    """Check that we raise proper error when axis=1 and the dimension mismatch.
    Non-regression test for:
    https://github.com/scikit-learn/scikit-learn/pull/18655
    """
    n_samples, n_features = 60, 4
    rng = np.random.RandomState(42)
    X = sparse_constructor(rng.rand(n_samples, n_features))

    last_mean = np.zeros(n_features)
    last_var = np.zeros_like(last_mean)
    last_n = np.zeros(last_mean.shape, dtype=np.int64)

    kwargs = dict(last_mean=last_mean, last_var=last_var, last_n=last_n)
    mean0, var0, _ = incr_mean_variance_axis(X, axis=0, **kwargs)
    assert_allclose(np.mean(X.toarray(), axis=0), mean0)
    assert_allclose(np.var(X.toarray(), axis=0), var0)

    # test ValueError if axis=1 and last_mean.size == n_features
    with pytest.raises(ValueError):
        incr_mean_variance_axis(X, axis=1, **kwargs)

    # test inconsistent shapes of last_mean, last_var, last_n
    kwargs = dict(last_mean=last_mean[:-1], last_var=last_var, last_n=last_n)
    with pytest.raises(ValueError):
        incr_mean_variance_axis(X, axis=0, **kwargs)


@pytest.mark.parametrize(
    "X1, X2",
    [
        (
            _sparse_random_array((5, 2), density=0.8, format="csr", rng=0),
            _sparse_random_array((13, 2), density=0.8, format="csr", rng=0),
        ),
        (
            _sparse_random_array((5, 2), density=0.8, format="csr", rng=0),
            sp.hstack(
                [
                    np.full((13, 1), fill_value=np.nan),
                    _sparse_random_array((13, 1), density=0.8, rng=42),
                ],
                format="csr",
            ),
        ),
    ],
)
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_incr_mean_variance_axis_equivalence_mean_variance(X1, X2, csr_container):
    # [FUNCTION DOC]
    # Non-regression for #16448: two incremental updates on X1 then X2 (second may
    # include an all-NaN column) must equal nanmean, nanvar and non-NaN counts on the
    # vertically stacked data.
    # Inputs: X1, X2 - random sparse arrays; csr_container.
    # non-regression test for:
    # https://github.com/scikit-learn/scikit-learn/issues/16448
    # check that computing the incremental mean and variance is equivalent to
    # computing the mean and variance on the stacked dataset.
    X1 = csr_container(X1)
    X2 = csr_container(X2)
    axis = 0
    last_mean, last_var = np.zeros(X1.shape[1]), np.zeros(X1.shape[1])
    last_n = np.zeros(X1.shape[1], dtype=np.int64)
    updated_mean, updated_var, updated_n = incr_mean_variance_axis(
        X1, axis=axis, last_mean=last_mean, last_var=last_var, last_n=last_n
    )
    updated_mean, updated_var, updated_n = incr_mean_variance_axis(
        X2, axis=axis, last_mean=updated_mean, last_var=updated_var, last_n=updated_n
    )
    X = sp.vstack([X1, X2])
    assert_allclose(updated_mean, np.nanmean(X.toarray(), axis=axis))
    assert_allclose(updated_var, np.nanvar(X.toarray(), axis=axis))
    assert_allclose(updated_n, np.count_nonzero(~np.isnan(X.toarray()), axis=0))


def test_incr_mean_variance_no_new_n():
    # [FUNCTION DOC]
    # Verifies that updating statistics with an empty (0-row) matrix leaves mean,
    # variance and counts unchanged.
    # check the behaviour when we update the variance with an empty matrix
    axis = 0
    X1 = _sparse_random_array((5, 1), density=0.8, format="csr", rng=0)
    X2 = _sparse_random_array((0, 1), density=0.8, format="csr", rng=0)
    last_mean, last_var = np.zeros(X1.shape[1]), np.zeros(X1.shape[1])
    last_n = np.zeros(X1.shape[1], dtype=np.int64)
    last_mean, last_var, last_n = incr_mean_variance_axis(
        X1, axis=axis, last_mean=last_mean, last_var=last_var, last_n=last_n
    )
    # update statistic with a column which should ignored
    updated_mean, updated_var, updated_n = incr_mean_variance_axis(
        X2, axis=axis, last_mean=last_mean, last_var=last_var, last_n=last_n
    )
    assert_allclose(updated_mean, last_mean)
    assert_allclose(updated_var, last_var)
    assert_allclose(updated_n, last_n)


def test_incr_mean_variance_n_float():
    # [FUNCTION DOC]
    # Verifies that a scalar last_n of 0 is accepted and broadcast, giving new_n equal
    # to n_samples for every feature.
    # check the behaviour when last_n is just a number
    axis = 0
    X = _sparse_random_array((5, 2), density=0.8, format="csr", rng=0)
    last_mean, last_var = np.zeros(X.shape[1]), np.zeros(X.shape[1])
    last_n = 0
    _, _, new_n = incr_mean_variance_axis(
        X, axis=axis, last_mean=last_mean, last_var=last_var, last_n=last_n
    )
    assert_allclose(new_n, np.full(X.shape[1], X.shape[0]))


@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("sparse_constructor", CSC_CONTAINERS + CSR_CONTAINERS)
def test_incr_mean_variance_axis_ignore_nan(axis, sparse_constructor):
    # [FUNCTION DOC]
    # Verifies that NaN entries are ignored by incr_mean_variance_axis: data with NaNs
    # inserted (and the same non-NaN values per feature) yields the same updated means,
    # variances and counts as the NaN-free data, starting from identical prior
    # statistics (copied because they are modified in place). Uses transposes for axis
    # 1.
    # Inputs: axis - 0 or 1; sparse_constructor.
    old_means = np.array([535.0, 535.0, 535.0, 535.0])
    old_variances = np.array([4225.0, 4225.0, 4225.0, 4225.0])
    old_sample_count = np.array([2, 2, 2, 2], dtype=np.int64)

    X = sparse_constructor(
        np.array([[170, 170, 170, 170], [430, 430, 430, 430], [300, 300, 300, 300]])
    )

    X_nan = sparse_constructor(
        np.array(
            [
                [170, np.nan, 170, 170],
                [np.nan, 170, 430, 430],
                [430, 430, np.nan, 300],
                [300, 300, 300, np.nan],
            ]
        )
    )

    # we avoid creating specific data for axis 0 and 1: translating the data is
    # enough.
    if axis:
        X = X.T
        X_nan = X_nan.T

    # take a copy of the old statistics since they are modified in place.
    X_means, X_vars, X_sample_count = incr_mean_variance_axis(
        X,
        axis=axis,
        last_mean=old_means.copy(),
        last_var=old_variances.copy(),
        last_n=old_sample_count.copy(),
    )
    X_nan_means, X_nan_vars, X_nan_sample_count = incr_mean_variance_axis(
        X_nan,
        axis=axis,
        last_mean=old_means.copy(),
        last_var=old_variances.copy(),
        last_n=old_sample_count.copy(),
    )

    assert_allclose(X_nan_means, X_means)
    assert_allclose(X_nan_vars, X_vars)
    assert_allclose(X_nan_sample_count, X_sample_count)


@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_mean_variance_illegal_axis(csr_container):
    # [FUNCTION DOC]
    # Verifies ValueError for axis values -3, 2 and -1 in both mean_variance_axis and
    # incr_mean_variance_axis.
    # Inputs: csr_container.
    X, _ = make_classification(5, 4, random_state=0)
    # Sparsify the array a little bit
    X[0, 0] = 0
    X[2, 1] = 0
    X[4, 3] = 0
    X_csr = csr_container(X)
    with pytest.raises(ValueError):
        mean_variance_axis(X_csr, axis=-3)
    with pytest.raises(ValueError):
        mean_variance_axis(X_csr, axis=2)
    with pytest.raises(ValueError):
        mean_variance_axis(X_csr, axis=-1)

    with pytest.raises(ValueError):
        incr_mean_variance_axis(
            X_csr, axis=-3, last_mean=None, last_var=None, last_n=None
        )

    with pytest.raises(ValueError):
        incr_mean_variance_axis(
            X_csr, axis=2, last_mean=None, last_var=None, last_n=None
        )

    with pytest.raises(ValueError):
        incr_mean_variance_axis(
            X_csr, axis=-1, last_mean=None, last_var=None, last_n=None
        )


@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_densify_rows(csr_container):
    # [FUNCTION DOC]
    # Verifies assign_rows_csr copies selected CSR rows (0, 2, 3) densely into rows (1,
    # 3, 4) of a preallocated ones array, leaving other rows untouched, for float32 and
    # float64.
    # Inputs: csr_container.
    for dtype in (np.float32, np.float64):
        X = csr_container(
            [[0, 3, 0], [2, 4, 0], [0, 0, 0], [9, 8, 7], [4, 0, 5]], dtype=dtype
        )
        X_rows = np.array([0, 2, 3], dtype=np.intp)
        out = np.ones((6, X.shape[1]), dtype=dtype)
        out_rows = np.array([1, 3, 4], dtype=np.intp)

        expect = np.ones_like(out)
        expect[out_rows] = X[X_rows, :].toarray()

        assign_rows_csr(X, X_rows, out_rows, out)
        assert_array_equal(out, expect)


def test_inplace_column_scale():
    # [FUNCTION DOC]
    # Verifies inplace_column_scale multiplies each column of CSR and CSC matrices by
    # the scale vector, matching dense scaling, for float64 and float32, and raises
    # TypeError for LIL input. Uses an unseeded _sparse_random_array for X.
    rng = np.random.RandomState(0)
    X = _sparse_random_array((100, 200), density=0.05)
    Xr = X.tocsr()
    Xc = X.tocsc()
    XA = X.toarray()
    scale = rng.rand(200)
    XA *= scale

    inplace_column_scale(Xc, scale)
    inplace_column_scale(Xr, scale)
    assert_array_almost_equal(Xr.toarray(), Xc.toarray())
    assert_array_almost_equal(XA, Xc.toarray())
    assert_array_almost_equal(XA, Xr.toarray())
    with pytest.raises(TypeError):
        inplace_column_scale(X.tolil(), scale)

    X = X.astype(np.float32)
    scale = scale.astype(np.float32)
    Xr = X.tocsr()
    Xc = X.tocsc()
    XA = X.toarray()
    XA *= scale
    inplace_column_scale(Xc, scale)
    inplace_column_scale(Xr, scale)
    assert_array_almost_equal(Xr.toarray(), Xc.toarray())
    assert_array_almost_equal(XA, Xc.toarray())
    assert_array_almost_equal(XA, Xr.toarray())
    with pytest.raises(TypeError):
        inplace_column_scale(X.tolil(), scale)


def test_inplace_row_scale():
    # [FUNCTION DOC]
    # Verifies inplace_row_scale multiplies each row of CSR and CSC matrices by the
    # scale vector, matching dense scaling, for float64 and float32.
    # KNOWN ISSUE: the TypeError checks call inplace_column_scale instead of
    # inplace_row_scale, so the LIL rejection of inplace_row_scale is not tested.
    rng = np.random.RandomState(0)
    X = _sparse_random_array((100, 200), density=0.05)
    Xr = X.tocsr()
    Xc = X.tocsc()
    XA = X.toarray()
    scale = rng.rand(100)
    XA *= scale.reshape(-1, 1)

    inplace_row_scale(Xc, scale)
    inplace_row_scale(Xr, scale)
    assert_array_almost_equal(Xr.toarray(), Xc.toarray())
    assert_array_almost_equal(XA, Xc.toarray())
    assert_array_almost_equal(XA, Xr.toarray())
    with pytest.raises(TypeError):
        inplace_column_scale(X.tolil(), scale)

    X = X.astype(np.float32)
    scale = scale.astype(np.float32)
    Xr = X.tocsr()
    Xc = X.tocsc()
    XA = X.toarray()
    XA *= scale.reshape(-1, 1)
    inplace_row_scale(Xc, scale)
    inplace_row_scale(Xr, scale)
    assert_array_almost_equal(Xr.toarray(), Xc.toarray())
    assert_array_almost_equal(XA, Xc.toarray())
    assert_array_almost_equal(XA, Xr.toarray())
    with pytest.raises(TypeError):
        inplace_column_scale(X.tolil(), scale)


@pytest.mark.parametrize("csc_container", CSC_CONTAINERS)
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_inplace_swap_row(csc_container, csr_container):
    # [FUNCTION DOC]
    # Verifies inplace_swap_row on CSR and CSC (including negative index -1) matches a
    # BLAS swap of dense rows, for float64 and float32.
    # KNOWN ISSUE: the TypeError check calls inplace_swap_row with only the LIL matrix
    # and no row indices, so the TypeError comes from missing arguments rather than from
    # LIL rejection.
    X = np.array(
        [[0, 3, 0], [2, 4, 0], [0, 0, 0], [9, 8, 7], [4, 0, 5]], dtype=np.float64
    )
    X_csr = csr_container(X)
    X_csc = csc_container(X)

    swap = linalg.get_blas_funcs(("swap",), (X,))
    swap = swap[0]
    X[0], X[-1] = swap(X[0], X[-1])
    inplace_swap_row(X_csr, 0, -1)
    inplace_swap_row(X_csc, 0, -1)
    assert_array_equal(X_csr.toarray(), X_csc.toarray())
    assert_array_equal(X, X_csc.toarray())
    assert_array_equal(X, X_csr.toarray())

    X[2], X[3] = swap(X[2], X[3])
    inplace_swap_row(X_csr, 2, 3)
    inplace_swap_row(X_csc, 2, 3)
    assert_array_equal(X_csr.toarray(), X_csc.toarray())
    assert_array_equal(X, X_csc.toarray())
    assert_array_equal(X, X_csr.toarray())
    with pytest.raises(TypeError):
        inplace_swap_row(X_csr.tolil())

    X = np.array(
        [[0, 3, 0], [2, 4, 0], [0, 0, 0], [9, 8, 7], [4, 0, 5]], dtype=np.float32
    )
    X_csr = csr_container(X)
    X_csc = csc_container(X)
    swap = linalg.get_blas_funcs(("swap",), (X,))
    swap = swap[0]
    X[0], X[-1] = swap(X[0], X[-1])
    inplace_swap_row(X_csr, 0, -1)
    inplace_swap_row(X_csc, 0, -1)
    assert_array_equal(X_csr.toarray(), X_csc.toarray())
    assert_array_equal(X, X_csc.toarray())
    assert_array_equal(X, X_csr.toarray())
    X[2], X[3] = swap(X[2], X[3])
    inplace_swap_row(X_csr, 2, 3)
    inplace_swap_row(X_csc, 2, 3)
    assert_array_equal(X_csr.toarray(), X_csc.toarray())
    assert_array_equal(X, X_csc.toarray())
    assert_array_equal(X, X_csr.toarray())
    with pytest.raises(TypeError):
        inplace_swap_row(X_csr.tolil())


@pytest.mark.parametrize("csc_container", CSC_CONTAINERS)
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_inplace_swap_column(csc_container, csr_container):
    # [FUNCTION DOC]
    # Verifies inplace_swap_column on CSR and CSC (including index -1) matches a BLAS
    # swap of dense columns, for float64 and float32.
    # KNOWN ISSUE: the TypeError check omits the column indices, so it raises for
    # missing arguments rather than testing LIL rejection.
    X = np.array(
        [[0, 3, 0], [2, 4, 0], [0, 0, 0], [9, 8, 7], [4, 0, 5]], dtype=np.float64
    )
    X_csr = csr_container(X)
    X_csc = csc_container(X)

    swap = linalg.get_blas_funcs(("swap",), (X,))
    swap = swap[0]
    X[:, 0], X[:, -1] = swap(X[:, 0], X[:, -1])
    inplace_swap_column(X_csr, 0, -1)
    inplace_swap_column(X_csc, 0, -1)
    assert_array_equal(X_csr.toarray(), X_csc.toarray())
    assert_array_equal(X, X_csc.toarray())
    assert_array_equal(X, X_csr.toarray())

    X[:, 0], X[:, 1] = swap(X[:, 0], X[:, 1])
    inplace_swap_column(X_csr, 0, 1)
    inplace_swap_column(X_csc, 0, 1)
    assert_array_equal(X_csr.toarray(), X_csc.toarray())
    assert_array_equal(X, X_csc.toarray())
    assert_array_equal(X, X_csr.toarray())
    with pytest.raises(TypeError):
        inplace_swap_column(X_csr.tolil())

    X = np.array(
        [[0, 3, 0], [2, 4, 0], [0, 0, 0], [9, 8, 7], [4, 0, 5]], dtype=np.float32
    )
    X_csr = csr_container(X)
    X_csc = csc_container(X)
    swap = linalg.get_blas_funcs(("swap",), (X,))
    swap = swap[0]
    X[:, 0], X[:, -1] = swap(X[:, 0], X[:, -1])
    inplace_swap_column(X_csr, 0, -1)
    inplace_swap_column(X_csc, 0, -1)
    assert_array_equal(X_csr.toarray(), X_csc.toarray())
    assert_array_equal(X, X_csc.toarray())
    assert_array_equal(X, X_csr.toarray())
    X[:, 0], X[:, 1] = swap(X[:, 0], X[:, 1])
    inplace_swap_column(X_csr, 0, 1)
    inplace_swap_column(X_csc, 0, 1)
    assert_array_equal(X_csr.toarray(), X_csc.toarray())
    assert_array_equal(X, X_csc.toarray())
    assert_array_equal(X, X_csr.toarray())
    with pytest.raises(TypeError):
        inplace_swap_column(X_csr.tolil())


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("axis", [0, 1, None])
@pytest.mark.parametrize("sparse_format", CSC_CONTAINERS + CSR_CONTAINERS)
@pytest.mark.parametrize(
    "missing_values, min_func, max_func, ignore_nan",
    [(0, np.min, np.max, False), (np.nan, np.nanmin, np.nanmax, True)],
)
@pytest.mark.parametrize("large_indices", [True, False])
def test_min_max(
    dtype,
    axis,
    sparse_format,
    missing_values,
    min_func,
    max_func,
    ignore_nan,
    large_indices,
):
    # [FUNCTION DOC]
    # Verifies min_max_axis matches np.min/np.max (or nanmin/nanmax with ignore_nan when
    # the missing value is NaN) along axis 0, 1 and None, with int32 or int64 index
    # arrays.
    # Inputs: dtype; axis; sparse_format - CSR/CSC constructor; missing_values,
    # min_func, max_func, ignore_nan - paired; large_indices - bool.
    X = np.array(
        [
            [0, 3, 0],
            [2, -1, missing_values],
            [0, 0, 0],
            [9, missing_values, 7],
            [4, 0, 5],
        ],
        dtype=dtype,
    )
    X_sparse = sparse_format(X)

    if large_indices:
        X_sparse.indices = X_sparse.indices.astype("int64")
        X_sparse.indptr = X_sparse.indptr.astype("int64")

    mins_sparse, maxs_sparse = min_max_axis(X_sparse, axis=axis, ignore_nan=ignore_nan)
    assert_array_equal(mins_sparse, min_func(X, axis=axis))
    assert_array_equal(maxs_sparse, max_func(X, axis=axis))


@pytest.mark.parametrize("csc_container", CSC_CONTAINERS)
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_min_max_axis_errors(csc_container, csr_container):
    # [FUNCTION DOC]
    # Verifies min_max_axis raises TypeError for LIL input and ValueError for axis 2 or
    # -3.
    # Inputs: csc_container, csr_container.
    X = np.array(
        [[0, 3, 0], [2, -1, 0], [0, 0, 0], [9, 8, 7], [4, 0, 5]], dtype=np.float64
    )
    X_csr = csr_container(X)
    X_csc = csc_container(X)
    with pytest.raises(TypeError):
        min_max_axis(X_csr.tolil(), axis=0)
    with pytest.raises(ValueError):
        min_max_axis(X_csr, axis=2)
    with pytest.raises(ValueError):
        min_max_axis(X_csc, axis=-3)


@pytest.mark.parametrize("csc_container", CSC_CONTAINERS)
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_count_nonzero(csc_container, csr_container):
    # [FUNCTION DOC]
    # Verifies count_nonzero on CSR (axes 0, 1, -1, -2, None; with and without
    # sample_weight) matches dense counts/weighted sums, raises TypeError for CSC and
    # ValueError for axis 2, and returns the same dtype for both axes, also with int64
    # indices (tolerating a casting TypeError on 32-bit platforms).
    # Inputs: csc_container, csr_container.
    # Side effects: mutates X_csr index dtype in place.
    X = np.array(
        [[0, 3, 0], [2, -1, 0], [0, 0, 0], [9, 8, 7], [4, 0, 5]], dtype=np.float64
    )
    X_csr = csr_container(X)
    X_csc = csc_container(X)
    X_nonzero = X != 0
    sample_weight = [0.5, 0.2, 0.3, 0.1, 0.1]
    X_nonzero_weighted = X_nonzero * np.array(sample_weight)[:, None]

    for axis in [0, 1, -1, -2, None]:
        assert_array_almost_equal(
            count_nonzero(X_csr, axis=axis), X_nonzero.sum(axis=axis)
        )
        assert_array_almost_equal(
            count_nonzero(X_csr, axis=axis, sample_weight=sample_weight),
            X_nonzero_weighted.sum(axis=axis),
        )

    with pytest.raises(TypeError):
        count_nonzero(X_csc)
    with pytest.raises(ValueError):
        count_nonzero(X_csr, axis=2)

    assert count_nonzero(X_csr, axis=0).dtype == count_nonzero(X_csr, axis=1).dtype
    assert (
        count_nonzero(X_csr, axis=0, sample_weight=sample_weight).dtype
        == count_nonzero(X_csr, axis=1, sample_weight=sample_weight).dtype
    )

    # Check dtypes with large sparse matrices too
    # XXX: test fails on 32bit (Windows/Linux)
    try:
        X_csr.indices = X_csr.indices.astype(np.int64)
        X_csr.indptr = X_csr.indptr.astype(np.int64)
        assert count_nonzero(X_csr, axis=0).dtype == count_nonzero(X_csr, axis=1).dtype
        assert (
            count_nonzero(X_csr, axis=0, sample_weight=sample_weight).dtype
            == count_nonzero(X_csr, axis=1, sample_weight=sample_weight).dtype
        )
    except TypeError as e:
        assert "according to the rule 'safe'" in e.args[0] and np.intp().nbytes < 8, e


@pytest.mark.parametrize("csc_container", CSC_CONTAINERS)
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_csc_row_median(csc_container, csr_container):
    # [FUNCTION DOC]
    # Verifies csc_median_axis_0 matches np.median along axis 0 on dense random data, on
    # sparse data with negatives, and on toy inputs with even/odd row counts; raises
    # TypeError for CSR input.
    # Inputs: csc_container, csr_container.
    # Test csc_row_median actually calculates the median.

    # Test that it gives the same output when X is dense.
    rng = np.random.RandomState(0)
    X = rng.rand(100, 50)
    dense_median = np.median(X, axis=0)
    csc = csc_container(X)
    sparse_median = csc_median_axis_0(csc)
    assert_array_equal(sparse_median, dense_median)

    # Test that it gives the same output when X is sparse
    X = rng.rand(51, 100)
    X[X < 0.7] = 0.0
    ind = rng.randint(0, 50, 10)
    X[ind] = -X[ind]
    csc = csc_container(X)
    dense_median = np.median(X, axis=0)
    sparse_median = csc_median_axis_0(csc)
    assert_array_equal(sparse_median, dense_median)

    # Test for toy data.
    X = [[0, -2], [-1, -1], [1, 0], [2, 1]]
    csc = csc_container(X)
    assert_array_equal(csc_median_axis_0(csc), np.array([0.5, -0.5]))
    X = [[0, -2], [-1, -5], [1, -3]]
    csc = csc_container(X)
    assert_array_equal(csc_median_axis_0(csc), np.array([0.0, -3]))

    # Test that it raises an Error for non-csc matrices.
    with pytest.raises(TypeError):
        csc_median_axis_0(csr_container(X))


@pytest.mark.parametrize(
    "inplace_csr_row_normalize",
    (inplace_csr_row_normalize_l1, inplace_csr_row_normalize_l2),
)
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_inplace_normalize(csr_container, inplace_csr_row_normalize):
    # [FUNCTION DOC]
    # Verifies inplace_csr_row_normalize_l1/l2 keep dtype and produce rows whose L1 norm
    # (of absolute values, or of squared values for L2) is 1, with int32 and then int64
    # indices. The same matrix is normalized repeatedly across index dtypes.
    # Inputs: csr_container; inplace_csr_row_normalize - the function under test.
    if issubclass(sp.csr_matrix, csr_container):
        ones = np.ones((10, 1))
    else:
        ones = np.ones(10)
    rs = RandomState(10)

    for dtype in (np.float64, np.float32):
        X = rs.randn(10, 5).astype(dtype)
        X_csr = csr_container(X)
        for index_dtype in [np.int32, np.int64]:
            # csr_matrix will use int32 indices by default,
            # up-casting those to int64 when necessary
            if index_dtype is np.int64:
                X_csr.indptr = X_csr.indptr.astype(index_dtype)
                X_csr.indices = X_csr.indices.astype(index_dtype)
            assert X_csr.indices.dtype == index_dtype
            assert X_csr.indptr.dtype == index_dtype
            inplace_csr_row_normalize(X_csr)
            assert X_csr.dtype == dtype
            if inplace_csr_row_normalize is inplace_csr_row_normalize_l2:
                X_csr.data **= 2
            assert_array_almost_equal(np.abs(X_csr).sum(axis=1), ones)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_csr_row_norms(dtype):
    # [FUNCTION DOC]
    # Verifies csr_row_norms returns squared row L2 norms matching
    # scipy.sparse.linalg.norm squared, preserving dtype (rtol 1e-6 for float32, 1e-7
    # for float64).
    # Inputs: dtype.
    # checks that csr_row_norms returns the same output as
    # scipy.sparse.linalg.norm, and that the dype is the same as X.dtype.
    X = _sparse_random_array((100, 10), format="csr", dtype=dtype, rng=42)

    scipy_norms = sp.linalg.norm(X, axis=1) ** 2
    norms = csr_row_norms(X)

    assert norms.dtype == dtype
    rtol = 1e-6 if dtype == np.float32 else 1e-7
    assert_allclose(norms, scipy_norms, rtol=rtol)


@pytest.fixture(scope="module", params=CSR_CONTAINERS + CSC_CONTAINERS)
def centered_matrices(request):
    # [FUNCTION DOC]
    # HELPER: Module-scoped fixture parametrized over CSR and CSC containers. Builds a
    # 500x100 random sparse matrix (density 0.1, seed 42), its column means, and returns
    # the pair (LinearOperator from _implicit_column_offset, explicitly centered dense
    # array).
    # Inputs: request - pytest request carrying the container.
    # Outputs: tuple (LinearOperator, ndarray of shape (500, 100)).
    """Returns equivalent tuple[sp.linalg.LinearOperator, np.ndarray]."""
    sparse_container = request.param

    rng = np.random.default_rng(42)

    X_sparse = sparse_container(
        _sparse_random_array((500, 100), density=0.1, format="csr", rng=rng)
    )
    X_dense = X_sparse.toarray()
    mu = np.asarray(X_sparse.mean(axis=0)).ravel()

    X_sparse_centered = _implicit_column_offset(X_sparse, mu)
    X_dense_centered = X_dense - mu

    return X_sparse_centered, X_dense_centered


def test_implicit_center_matmat(global_random_seed, centered_matrices):
    # [FUNCTION DOC]
    # Verifies the implicitly centered operator's matmat and @ with a random (100, 50)
    # matrix equal the dense centered product.
    # Inputs: global_random_seed; centered_matrices fixture.
    X_sparse_centered, X_dense_centered = centered_matrices
    rng = np.random.default_rng(global_random_seed)
    Y = rng.standard_normal((X_dense_centered.shape[1], 50))
    assert_allclose(X_dense_centered @ Y, X_sparse_centered.matmat(Y))
    assert_allclose(X_dense_centered @ Y, X_sparse_centered @ Y)


def test_implicit_center_matvec(global_random_seed, centered_matrices):
    # [FUNCTION DOC]
    # Verifies matvec and @ with a random vector of length 100 equal the dense centered
    # product.
    # Inputs: global_random_seed; centered_matrices.
    X_sparse_centered, X_dense_centered = centered_matrices
    rng = np.random.default_rng(global_random_seed)
    y = rng.standard_normal(X_dense_centered.shape[1])
    assert_allclose(X_dense_centered @ y, X_sparse_centered.matvec(y))
    assert_allclose(X_dense_centered @ y, X_sparse_centered @ y)


def test_implicit_center_rmatmat(global_random_seed, centered_matrices):
    # [FUNCTION DOC]
    # Verifies rmatmat and transpose @ with a random (500, 50) matrix equal the dense
    # centered transpose product.
    # Inputs: global_random_seed; centered_matrices.
    X_sparse_centered, X_dense_centered = centered_matrices
    rng = np.random.default_rng(global_random_seed)
    Y = rng.standard_normal((X_dense_centered.shape[0], 50))
    assert_allclose(X_dense_centered.T @ Y, X_sparse_centered.rmatmat(Y))
    assert_allclose(X_dense_centered.T @ Y, X_sparse_centered.T @ Y)


def test_implit_center_rmatvec(global_random_seed, centered_matrices):
    # [FUNCTION DOC]
    # Verifies rmatvec and transpose @ with a random vector of length 500 equal the
    # dense centered transpose product.
    # Inputs: global_random_seed; centered_matrices.
    # KNOWN ISSUE: test name misspells 'implicit'.
    X_sparse_centered, X_dense_centered = centered_matrices
    rng = np.random.default_rng(global_random_seed)
    y = rng.standard_normal(X_dense_centered.shape[0])
    assert_allclose(X_dense_centered.T @ y, X_sparse_centered.rmatvec(y))
    assert_allclose(X_dense_centered.T @ y, X_sparse_centered.T @ y)


@pytest.mark.parametrize(
    ["A", "B", "out", "msg"],
    [
        (
            _sparse_eye_array(3, format="csr"),
            _sparse_eye_array(2, format="csr"),
            None,
            "Shapes must fulfil",
        ),
        (
            _sparse_eye_array(2, format="csr"),
            _sparse_eye_array(2, format="csr"),
            np.eye(3),
            "Shape of out",
        ),
        (
            _sparse_eye_array(2, format="coo"),
            _sparse_eye_array(2, format="csr"),
            None,
            "Input 'A' must",
        ),
        (
            _sparse_eye_array(2, format="csr"),
            _sparse_eye_array(2, format="coo"),
            None,
            "Input 'B' must",
        ),
        (
            _sparse_eye_array(2, format="csr", dtype=np.int32),
            _sparse_eye_array(2, format="csr"),
            None,
            "Dtype of A and B",
        ),
        (
            _sparse_eye_array(2, format="csr", dtype=np.float32),
            _sparse_eye_array(2, format="csr", dtype=np.float64),
            None,
            "Dtype of A and B",
        ),
    ],
)
def test_sparse_matmul_to_dense_raises(A, B, out, msg):
    # [FUNCTION DOC]
    # Verifies sparse_matmul_to_dense raises ValueError for incompatible shapes, wrong
    # out shape, non-CSR/CSC A or B, and mismatched or non-float dtypes.
    # Inputs: A, B - sparse arrays; out - ndarray or None; msg - expected regex.
    """Test that sparse_matmul_to_dense raises when it should."""
    with pytest.raises(ValueError, match=msg):
        sparse_matmul_to_dense(A, B, out=out)


@pytest.mark.parametrize("out_is_None", [False, True])
@pytest.mark.parametrize("a_container", CSC_CONTAINERS + CSR_CONTAINERS)
@pytest.mark.parametrize("b_container", CSC_CONTAINERS + CSR_CONTAINERS)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_sparse_matmul_to_dense(
    global_random_seed, out_is_None, a_container, b_container, dtype
):
    # [FUNCTION DOC]
    # Verifies sparse_matmul_to_dense of random half-sparse A (10x19) and B (19x13) in
    # any CSR/CSC combination equals the dense product (atol 1e-7), and that when out is
    # given it is filled with the result.
    # Inputs: global_random_seed; out_is_None - bool; a_container, b_container; dtype.
    """Test that sparse_matmul_to_dense computes correctly."""
    rng = np.random.default_rng(global_random_seed)
    n1, n2, n3 = 10, 19, 13
    a_dense = rng.standard_normal((n1, n2)).astype(dtype)
    b_dense = rng.standard_normal((n2, n3)).astype(dtype)
    a_dense.flat[rng.choice([False, True], size=n1 * n2, p=[0.5, 0.5])] = 0
    b_dense.flat[rng.choice([False, True], size=n2 * n3, p=[0.5, 0.5])] = 0
    a = a_container(a_dense)
    b = b_container(b_dense)
    if out_is_None:
        out = None
    else:
        out = np.empty((n1, n3), dtype=dtype)

    result = sparse_matmul_to_dense(a, b, out=out)
    # Use atol to account for the wide range of values in the computed matrix.
    assert_allclose(result, a_dense @ b_dense, atol=1e-7)
    if not out_is_None:
        assert_allclose(out, result, atol=1e-7)
