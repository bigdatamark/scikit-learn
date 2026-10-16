# [MODULE DOC]
# Tests for sklearn.metrics.pairwise: pairwise_distances, pairwise_kernels, paired
#   distances, chunked and argmin reductions, individual distance and kernel functions
#   (euclidean, nan_euclidean, cosine, haversine, chi2, ...), input checkers
#   check_pairwise_arrays/check_paired_arrays, and Array API dispatch support.
# Results are mostly compared against scipy.spatial.distance (cdist, pdist) or direct
#   per-pair reference computations.
import warnings
from types import GeneratorType

import numpy as np
import pytest
from numpy import linalg
from scipy.sparse import issparse
from scipy.spatial.distance import (
    cdist,
    cityblock,
    cosine,
    minkowski,
    pdist,
    squareform,
)

from sklearn import config_context
from sklearn.exceptions import DataConversionWarning
from sklearn.metrics.pairwise import (
    PAIRED_DISTANCES,
    PAIRWISE_BOOLEAN_FUNCTIONS,
    PAIRWISE_DISTANCE_FUNCTIONS,
    PAIRWISE_KERNEL_FUNCTIONS,
    _euclidean_distances_upcast,
    additive_chi2_kernel,
    check_paired_arrays,
    check_pairwise_arrays,
    chi2_kernel,
    cosine_distances,
    cosine_similarity,
    euclidean_distances,
    haversine_distances,
    laplacian_kernel,
    linear_kernel,
    manhattan_distances,
    nan_euclidean_distances,
    paired_cosine_distances,
    paired_distances,
    paired_euclidean_distances,
    paired_manhattan_distances,
    pairwise_distances,
    pairwise_distances_argmin,
    pairwise_distances_argmin_min,
    pairwise_distances_chunked,
    pairwise_kernels,
    polynomial_kernel,
    rbf_kernel,
    sigmoid_kernel,
)
from sklearn.preprocessing import normalize
from sklearn.utils._array_api import (
    get_namespace,
    move_to,
    xpx,
    yield_namespace_device_dtype_combinations,
)
from sklearn.utils._testing import (
    _array_api_for_tests,
    assert_allclose,
    assert_almost_equal,
    assert_array_equal,
    ignore_warnings,
)
from sklearn.utils.fixes import (
    BSR_CONTAINERS,
    COO_CONTAINERS,
    CSC_CONTAINERS,
    CSR_CONTAINERS,
    DOK_CONTAINERS,
)
from sklearn.utils.parallel import Parallel, delayed


def test_pairwise_distances_for_dense_data(global_dtype):
    # [FUNCTION DOC]
    # Verifies pairwise_distances on dense data equals the dedicated functions
    #   (euclidean with and without Y, nan_euclidean with NaNs, tuple inputs, haversine)
    #   and that string metrics match scipy callables (cityblock, manhattan, cosine),
    #   checking dtype preservation where applicable.
    # Test the pairwise_distance helper function.
    rng = np.random.RandomState(0)

    # Euclidean distance should be equivalent to calling the function.
    X = rng.random_sample((5, 4)).astype(global_dtype, copy=False)
    S = pairwise_distances(X, metric="euclidean")
    S2 = euclidean_distances(X)
    assert_allclose(S, S2)
    assert S.dtype == S2.dtype == global_dtype

    # Euclidean distance, with Y != X.
    Y = rng.random_sample((2, 4)).astype(global_dtype, copy=False)
    S = pairwise_distances(X, Y, metric="euclidean")
    S2 = euclidean_distances(X, Y)
    assert_allclose(S, S2)
    assert S.dtype == S2.dtype == global_dtype

    # Check to ensure NaNs work with pairwise_distances.
    X_masked = rng.random_sample((5, 4)).astype(global_dtype, copy=False)
    Y_masked = rng.random_sample((2, 4)).astype(global_dtype, copy=False)
    X_masked[0, 0] = np.nan
    Y_masked[0, 0] = np.nan
    S_masked = pairwise_distances(X_masked, Y_masked, metric="nan_euclidean")
    S2_masked = nan_euclidean_distances(X_masked, Y_masked)
    assert_allclose(S_masked, S2_masked)
    assert S_masked.dtype == S2_masked.dtype == global_dtype

    # Test with tuples as X and Y
    X_tuples = tuple([tuple([v for v in row]) for row in X])
    Y_tuples = tuple([tuple([v for v in row]) for row in Y])
    S2 = pairwise_distances(X_tuples, Y_tuples, metric="euclidean")
    assert_allclose(S, S2)
    assert S.dtype == S2.dtype == global_dtype

    # Test haversine distance
    # The data should be valid latitude and longitude
    # haversine converts to float64 currently so we don't check dtypes.
    X = rng.random_sample((5, 2)).astype(global_dtype, copy=False)
    X[:, 0] = (X[:, 0] - 0.5) * 2 * np.pi / 2
    X[:, 1] = (X[:, 1] - 0.5) * 2 * np.pi
    S = pairwise_distances(X, metric="haversine")
    S2 = haversine_distances(X)
    assert_allclose(S, S2)

    # Test haversine distance, with Y != X
    Y = rng.random_sample((2, 2)).astype(global_dtype, copy=False)
    Y[:, 0] = (Y[:, 0] - 0.5) * 2 * np.pi / 2
    Y[:, 1] = (Y[:, 1] - 0.5) * 2 * np.pi
    S = pairwise_distances(X, Y, metric="haversine")
    S2 = haversine_distances(X, Y)
    assert_allclose(S, S2)

    # "cityblock" uses scikit-learn metric, cityblock (function) is
    # scipy.spatial.
    # The metric functions from scipy converts to float64 so we don't check the dtypes.
    S = pairwise_distances(X, metric="cityblock")
    S2 = pairwise_distances(X, metric=cityblock)
    assert S.shape[0] == S.shape[1]
    assert S.shape[0] == X.shape[0]
    assert_allclose(S, S2)

    # The manhattan metric should be equivalent to cityblock.
    S = pairwise_distances(X, Y, metric="manhattan")
    S2 = pairwise_distances(X, Y, metric=cityblock)
    assert S.shape[0] == X.shape[0]
    assert S.shape[1] == Y.shape[0]
    assert_allclose(S, S2)

    # Test cosine as a string metric versus cosine callable
    # The string "cosine" uses sklearn.metric,
    # while the function cosine is scipy.spatial
    S = pairwise_distances(X, Y, metric="cosine")
    S2 = pairwise_distances(X, Y, metric=cosine)
    assert S.shape[0] == X.shape[0]
    assert S.shape[1] == Y.shape[0]
    assert_allclose(S, S2)


@pytest.mark.parametrize(
    "array_namespace, device_name, dtype_name",
    yield_namespace_device_dtype_combinations(),
)
@pytest.mark.parametrize("metric", ["cosine", "euclidean", "manhattan"])
def test_pairwise_distances_array_api(array_namespace, device_name, dtype_name, metric):
    # [FUNCTION DOC]
    # Verifies pairwise_distances under array_api_dispatch returns arrays in the input
    #   namespace, device and dtype, matching the numpy result, with and without Y.
    # Test array API support in pairwise_distances.
    xp, device = _array_api_for_tests(array_namespace, device_name, dtype_name)

    rng = np.random.RandomState(0)
    # Euclidean distance should be equivalent to calling the function.
    X_np = rng.random_sample((5, 4)).astype(dtype_name, copy=False)
    Y_np = rng.random_sample((5, 4)).astype(dtype_name, copy=False)
    X_xp = xp.asarray(X_np, device=device)
    Y_xp = xp.asarray(Y_np, device=device)

    with config_context(array_api_dispatch=True):
        # Test with Y=None
        D_xp = pairwise_distances(X_xp, metric=metric)
        D_xp_np = move_to(D_xp, xp=np, device="cpu")
        assert get_namespace(D_xp)[0].__name__ == xp.__name__
        assert D_xp.device == X_xp.device
        assert D_xp.dtype == X_xp.dtype

        D_np = pairwise_distances(X_np, metric=metric)
        assert_allclose(D_xp_np, D_np)

        # Test with Y=Y_np/Y_xp
        D_xp = pairwise_distances(X_xp, Y=Y_xp, metric=metric)
        D_xp_np = move_to(D_xp, xp=np, device="cpu")
        assert get_namespace(D_xp)[0].__name__ == xp.__name__
        assert D_xp.device == X_xp.device
        assert D_xp.dtype == X_xp.dtype

        D_np = pairwise_distances(X_np, Y=Y_np, metric=metric)
        assert_allclose(D_xp_np, D_np)


def test_pairwise_distances_array_api_no_warnings():
    # [FUNCTION DOC]
    # Regression test for #33829: verifies pairwise_distances on array_api_strict inputs
    #   emits no warnings at all (warnings turned into errors).
    # Regression test for https://github.com/scikit-learn/scikit-learn/issues/33829
    # pairwise_distances should not emit cross-library dtype comparison warnings
    # when called with Array API inputs under array_api_dispatch=True.
    xp, device = _array_api_for_tests("array_api_strict")

    rng = np.random.RandomState(0)
    X_np = rng.random_sample((5, 4))
    Y_np = rng.random_sample((3, 4))
    X_xp = xp.asarray(X_np, device=device)
    Y_xp = xp.asarray(Y_np, device=device)

    with config_context(array_api_dispatch=True):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            pairwise_distances(X_xp, Y_xp, metric="euclidean")


@pytest.mark.parametrize("coo_container", COO_CONTAINERS)
@pytest.mark.parametrize("csc_container", CSC_CONTAINERS)
@pytest.mark.parametrize("bsr_container", BSR_CONTAINERS)
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_pairwise_distances_for_sparse_data(
    coo_container, csc_container, bsr_container, csr_container, global_dtype
):
    # [FUNCTION DOC]
    # Verifies pairwise_distances on sparse inputs (CSR, CSC, BSR, COO) matches
    #   euclidean_distances, cosine_distances and manhattan_distances, documents that
    #   manhattan does not preserve float32 (asserting the dtype check fails), checks
    #   scipy minkowski string vs callable on dense data, and that scipy metrics raise
    #   TypeError on sparse input.
    # Test the pairwise_distance helper function.
    rng = np.random.RandomState(0)
    X = rng.random_sample((5, 4)).astype(global_dtype, copy=False)
    Y = rng.random_sample((2, 4)).astype(global_dtype, copy=False)

    # Test with sparse X and Y,
    # currently only supported for Euclidean, L1 and cosine.
    X_sparse = csr_container(X)
    Y_sparse = csr_container(Y)

    S = pairwise_distances(X_sparse, Y_sparse, metric="euclidean")
    S2 = euclidean_distances(X_sparse, Y_sparse)
    assert_allclose(S, S2)
    assert S.dtype == S2.dtype == global_dtype

    S = pairwise_distances(X_sparse, Y_sparse, metric="cosine")
    S2 = cosine_distances(X_sparse, Y_sparse)
    assert_allclose(S, S2)
    assert S.dtype == S2.dtype == global_dtype

    S = pairwise_distances(X_sparse, csc_container(Y), metric="manhattan")
    S2 = manhattan_distances(bsr_container(X), coo_container(Y))
    assert_allclose(S, S2)
    if global_dtype == np.float64:
        assert S.dtype == S2.dtype == global_dtype
    else:
        # TODO Fix manhattan_distances to preserve dtype.
        # currently pairwise_distances uses manhattan_distances but converts the result
        # back to the input dtype
        with pytest.raises(AssertionError):
            assert S.dtype == S2.dtype == global_dtype

    S2 = manhattan_distances(X, Y)
    assert_allclose(S, S2)
    if global_dtype == np.float64:
        assert S.dtype == S2.dtype == global_dtype
    else:
        # TODO Fix manhattan_distances to preserve dtype.
        # currently pairwise_distances uses manhattan_distances but converts the result
        # back to the input dtype
        with pytest.raises(AssertionError):
            assert S.dtype == S2.dtype == global_dtype

    # Test with scipy.spatial.distance metric, with a kwd
    kwds = {"p": 2.0}
    S = pairwise_distances(X, Y, metric="minkowski", **kwds)
    S2 = pairwise_distances(X, Y, metric=minkowski, **kwds)
    assert_allclose(S, S2)

    # same with Y = None
    kwds = {"p": 2.0}
    S = pairwise_distances(X, metric="minkowski", **kwds)
    S2 = pairwise_distances(X, metric=minkowski, **kwds)
    assert_allclose(S, S2)

    # Test that scipy distance metrics throw an error if sparse matrix given
    with pytest.raises(TypeError):
        pairwise_distances(X_sparse, metric="minkowski")
    with pytest.raises(TypeError):
        pairwise_distances(X, Y_sparse, metric="minkowski")


# Some scipy metrics are deprecated (depending on the scipy version) but we
# still want to test them.
@ignore_warnings(category=DeprecationWarning)
@pytest.mark.parametrize("metric", PAIRWISE_BOOLEAN_FUNCTIONS)
def test_pairwise_boolean_distance(metric):
    # [FUNCTION DOC]
    # Verifies boolean metrics give zero distance between identical boolean-converted
    #   rows, emit DataConversionWarning when non-boolean X or Y is converted, and no
    #   warning when X is already boolean and Y is None.
    # KNOWN ISSUE: the result of xpx.nan_to_num is discarded, so NaNs in res are not
    #   actually replaced before the comparison (NaN != 0 would count as non-zero).
    # test that we convert to boolean arrays for boolean distances
    rng = np.random.RandomState(0)
    X = rng.randn(5, 4)
    Y = X.copy()
    Y[0, 0] = 1 - Y[0, 0]

    # ignore conversion to boolean in pairwise_distances
    with ignore_warnings(category=DataConversionWarning):
        for Z in [Y, None]:
            res = pairwise_distances(X, Z, metric=metric)
            xpx.nan_to_num(res, fill_value=0)
            assert np.sum(res != 0) == 0

    # non-boolean arrays are converted to boolean for boolean
    # distance metrics with a data conversion warning
    msg = "Data was converted to boolean for metric %s" % metric
    with pytest.warns(DataConversionWarning, match=msg):
        pairwise_distances(X, metric=metric)

    # Check that the warning is raised if X is boolean by Y is not boolean:
    with pytest.warns(DataConversionWarning, match=msg):
        pairwise_distances(X.astype(bool), Y=Y, metric=metric)

    # Check that no warning is raised if X is already boolean and Y is None:
    with warnings.catch_warnings():
        warnings.simplefilter("error", DataConversionWarning)
        pairwise_distances(X.astype(bool), metric=metric)


def test_no_data_conversion_warning():
    # [FUNCTION DOC]
    # Verifies a non-boolean metric (minkowski) does not raise DataConversionWarning.
    # No warnings issued if metric is not a boolean distance function
    rng = np.random.RandomState(0)
    X = rng.randn(5, 4)
    with warnings.catch_warnings():
        warnings.simplefilter("error", DataConversionWarning)
        pairwise_distances(X, metric="minkowski")


@pytest.mark.parametrize("func", [pairwise_distances, pairwise_kernels])
def test_pairwise_precomputed(func):
    # [FUNCTION DOC]
    # Verifies metric="precomputed" for pairwise_distances and pairwise_kernels: wrong
    #   shapes raise ValueError, the input is returned without copying when already
    #   float, int input yields a float result, and lists are converted to ndarray.
    # Test correct shape
    with pytest.raises(ValueError, match=".* shape .*"):
        func(np.zeros((5, 3)), metric="precomputed")
    # with two args
    with pytest.raises(ValueError, match=".* shape .*"):
        func(np.zeros((5, 3)), np.zeros((4, 4)), metric="precomputed")
    # even if shape[1] agrees (although thus second arg is spurious)
    with pytest.raises(ValueError, match=".* shape .*"):
        func(np.zeros((5, 3)), np.zeros((4, 3)), metric="precomputed")

    # Test not copied (if appropriate dtype)
    S = np.zeros((5, 5))
    S2 = func(S, metric="precomputed")
    assert S is S2
    # with two args
    S = np.zeros((5, 3))
    S2 = func(S, np.zeros((3, 3)), metric="precomputed")
    assert S is S2

    # Test always returns float dtype
    S = func(np.array([[1]], dtype="int"), metric="precomputed")
    assert "f" == S.dtype.kind

    # Test converts list to array-like
    S = func([[1.0]], metric="precomputed")
    assert isinstance(S, np.ndarray)


def test_pairwise_precomputed_non_negative():
    # [FUNCTION DOC]
    # Verifies a precomputed distance matrix with negative values raises ValueError.
    # Test non-negative values
    with pytest.raises(ValueError, match=".* non-negative values.*"):
        pairwise_distances(np.full((5, 5), -1), metric="precomputed")


_minkowski_kwds = {"w": np.arange(1, 5).astype("double", copy=False), "p": 1}


def callable_rbf_kernel(x, y, **kwds):
    # [FUNCTION DOC]
    # HELPER: Callable version of rbf_kernel operating on two single samples, used as a
    #   custom kernel metric.
    # Inputs: x, y - 1-D (or 2-D) arrays of any supported namespace; kwds - forwarded to
    #   rbf_kernel (e.g. gamma).
    # Promotes inputs to 2-D, computes rbf_kernel and returns its single entry as a
    #   Python float.
    # Exceptions: ValueError if the kernel result is 0-dimensional or does not contain
    #   exactly one element.
    xp, _ = get_namespace(x, y)
    # Callable version of pairwise.rbf_kernel.
    K = rbf_kernel(
        xpx.atleast_nd(x, ndim=2, xp=xp), xpx.atleast_nd(y, ndim=2, xp=xp), **kwds
    )
    # unpack the output since this is a scalar packed in a 0-dim array
    # Note below is array API version of numpys `item()`
    if K.ndim > 0:
        K_flat = xp.reshape(K, (-1,))
        if K_flat.shape == (1,):
            return float(K_flat[0])
    raise ValueError("can only convert an array of size 1 to a Python scalar")


@pytest.mark.parametrize(
    "func, metric, kwds",
    [
        (pairwise_distances, "euclidean", {}),
        (
            pairwise_distances,
            minkowski,
            _minkowski_kwds,
        ),
        (
            pairwise_distances,
            "minkowski",
            _minkowski_kwds,
        ),
        (pairwise_kernels, "polynomial", {"degree": 1}),
        (pairwise_kernels, callable_rbf_kernel, {"gamma": 0.1}),
    ],
)
@pytest.mark.parametrize("dtype", [np.float64, np.float32, int])
def test_pairwise_parallel(func, metric, kwds, dtype):
    # [FUNCTION DOC]
    # Verifies pairwise_distances / pairwise_kernels give the same result with n_jobs=1
    #   and n_jobs=2, with and without Y, for several metrics (string and callable) and
    #   dtypes including int.
    rng = np.random.RandomState(0)
    X = np.array(5 * rng.random_sample((5, 4)), dtype=dtype)
    Y = np.array(5 * rng.random_sample((3, 4)), dtype=dtype)

    S = func(X, metric=metric, n_jobs=1, **kwds)
    S2 = func(X, metric=metric, n_jobs=2, **kwds)
    assert_allclose(S, S2)

    S = func(X, Y, metric=metric, n_jobs=1, **kwds)
    S2 = func(X, Y, metric=metric, n_jobs=2, **kwds)
    assert_allclose(S, S2)


@pytest.mark.parametrize(
    "array_namespace, device_name, dtype_name",
    yield_namespace_device_dtype_combinations(),
)
@pytest.mark.parametrize(
    "func, metric, kwds",
    [
        (pairwise_distances, "euclidean", {}),
        (pairwise_distances, "manhattan", {}),
        (pairwise_kernels, "polynomial", {"degree": 1}),
        (pairwise_kernels, callable_rbf_kernel, {"gamma": 0.1}),
        (pairwise_kernels, "laplacian", {"gamma": 0.1}),
    ],
)
def test_pairwise_parallel_array_api(
    func, metric, kwds, array_namespace, device_name, dtype_name
):
    # [FUNCTION DOC]
    # Verifies n_jobs=1 and n_jobs=2 give the same result under array_api_dispatch with
    #   the input namespace/device/dtype preserved, and match the numpy result.
    # KNOWN ISSUE: the first loop iteration sets Y_xp and Y_np to None and the second
    #   iteration keeps them None, so the Y-provided case is never tested; the numpy
    #   reference also never receives Y.
    xp, device = _array_api_for_tests(array_namespace, device_name, dtype_name)
    rng = np.random.RandomState(0)
    X_np = np.array(5 * rng.random_sample((5, 4)), dtype=dtype_name)
    Y_np = np.array(5 * rng.random_sample((3, 4)), dtype=dtype_name)
    X_xp = xp.asarray(X_np, device=device)
    Y_xp = xp.asarray(Y_np, device=device)

    with config_context(array_api_dispatch=True):
        for y_val in (None, "not none"):
            Y_xp = None if y_val is None else Y_xp
            Y_np = None if y_val is None else Y_np

            n_job1_xp = func(X_xp, Y_xp, metric=metric, n_jobs=1, **kwds)
            n_job1_xp_np = move_to(n_job1_xp, xp=np, device="cpu")
            assert get_namespace(n_job1_xp)[0].__name__ == xp.__name__
            assert n_job1_xp.device == X_xp.device
            assert n_job1_xp.dtype == X_xp.dtype

            n_job2_xp = func(X_xp, Y_xp, metric=metric, n_jobs=2, **kwds)
            n_job2_xp_np = move_to(n_job2_xp, xp=np, device="cpu")
            assert get_namespace(n_job2_xp)[0].__name__ == xp.__name__
            assert n_job2_xp.device == X_xp.device
            assert n_job2_xp.dtype == X_xp.dtype

            n_job2_np = func(X_np, metric=metric, n_jobs=2, **kwds)

            assert_allclose(n_job1_xp_np, n_job2_xp_np)
            assert_allclose(n_job2_xp_np, n_job2_np)


def test_pairwise_callable_nonstrict_metric():
    # [FUNCTION DOC]
    # Verifies pairwise_distances with a callable returning 5 for any pair computes the
    #   diagonal instead of assuming zero.
    # paired_distances should allow callable metric where metric(x, x) != 0
    # Knowing that the callable is a strict metric would allow the diagonal to
    # be left uncalculated and set to 0.
    assert pairwise_distances([[1.0]], metric=lambda x, y: 5)[0, 0] == 5


# Test with all metrics that should be in PAIRWISE_KERNEL_FUNCTIONS.
@pytest.mark.parametrize(
    "metric",
    ["rbf", "laplacian", "sigmoid", "polynomial", "linear", "chi2", "additive_chi2"],
)
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_pairwise_kernels(metric, csr_container):
    # [FUNCTION DOC]
    # Verifies pairwise_kernels with each kernel name matches the function in
    #   PAIRWISE_KERNEL_FUNCTIONS for Y=None, Y given, tuple inputs and (except chi2
    #   variants) CSR inputs.
    # Test the pairwise_kernels helper function.

    rng = np.random.RandomState(0)
    X = rng.random_sample((5, 4))
    Y = rng.random_sample((2, 4))
    function = PAIRWISE_KERNEL_FUNCTIONS[metric]
    # Test with Y=None
    K1 = pairwise_kernels(X, metric=metric)
    K2 = function(X)
    assert_allclose(K1, K2)
    # Test with Y=Y
    K1 = pairwise_kernels(X, Y=Y, metric=metric)
    K2 = function(X, Y=Y)
    assert_allclose(K1, K2)
    # Test with tuples as X and Y
    X_tuples = tuple([tuple([v for v in row]) for row in X])
    Y_tuples = tuple([tuple([v for v in row]) for row in Y])
    K2 = pairwise_kernels(X_tuples, Y_tuples, metric=metric)
    assert_allclose(K1, K2)

    # Test with sparse X and Y
    X_sparse = csr_container(X)
    Y_sparse = csr_container(Y)
    if metric in ["chi2", "additive_chi2"]:
        # these don't support sparse matrices yet
        return
    K1 = pairwise_kernels(X_sparse, Y=Y_sparse, metric=metric)
    assert_allclose(K1, K2)


@pytest.mark.parametrize(
    "array_namespace, device_name, dtype_name",
    yield_namespace_device_dtype_combinations(),
)
@pytest.mark.parametrize(
    "metric",
    ["rbf", "sigmoid", "polynomial", "linear", "laplacian", "chi2", "additive_chi2"],
)
def test_pairwise_kernels_array_api(metric, array_namespace, device_name, dtype_name):
    # [FUNCTION DOC]
    # Verifies pairwise_kernels under array_api_dispatch preserves namespace, device and
    #   dtype and matches numpy results for each kernel, with and without Y.
    # Test array API support in pairwise_kernels.
    xp, device = _array_api_for_tests(array_namespace, device_name, dtype_name)

    rng = np.random.RandomState(0)
    X_np = 10 * rng.random_sample((5, 4))
    X_np = X_np.astype(dtype_name, copy=False)
    Y_np = 10 * rng.random_sample((2, 4))
    Y_np = Y_np.astype(dtype_name, copy=False)
    X_xp = xp.asarray(X_np, device=device)
    Y_xp = xp.asarray(Y_np, device=device)

    with config_context(array_api_dispatch=True):
        # Test with Y=None
        K_xp = pairwise_kernels(X_xp, metric=metric)
        K_xp_np = move_to(K_xp, xp=np, device="cpu")
        assert get_namespace(K_xp)[0].__name__ == xp.__name__
        assert K_xp.device == X_xp.device
        assert K_xp.dtype == X_xp.dtype

        K_np = pairwise_kernels(X_np, metric=metric)
        assert_allclose(K_xp_np, K_np)

        # Test with Y=Y_np/Y_xp
        K_xp = pairwise_kernels(X_xp, Y=Y_xp, metric=metric)
        K_xp_np = move_to(K_xp, xp=np, device="cpu")
        assert get_namespace(K_xp)[0].__name__ == xp.__name__
        assert K_xp.device == X_xp.device
        assert K_xp.dtype == X_xp.dtype

        K_np = pairwise_kernels(X_np, Y=Y_np, metric=metric)
        assert_allclose(K_xp_np, K_np)


def test_pairwise_kernels_callable():
    # [FUNCTION DOC]
    # Verifies pairwise_kernels with callable_rbf_kernel and gamma matches rbf_kernel
    #   for Y distinct and Y equal to X.
    # Test the pairwise_kernels helper function
    # with a callable function, with given keywords.
    rng = np.random.RandomState(0)
    X = rng.random_sample((5, 4))
    Y = rng.random_sample((2, 4))

    metric = callable_rbf_kernel
    kwds = {"gamma": 0.1}
    K1 = pairwise_kernels(X, Y=Y, metric=metric, **kwds)
    K2 = rbf_kernel(X, Y=Y, **kwds)
    assert_allclose(K1, K2)

    # callable function, X=Y
    K1 = pairwise_kernels(X, Y=X, metric=metric, **kwds)
    K2 = rbf_kernel(X, Y=X, **kwds)
    assert_allclose(K1, K2)


def test_pairwise_kernels_filter_param():
    # [FUNCTION DOC]
    # Verifies filter_params=True drops unknown keyword arguments for the rbf kernel and
    #   that without it the extra keyword raises TypeError.
    rng = np.random.RandomState(0)
    X = rng.random_sample((5, 4))
    Y = rng.random_sample((2, 4))
    K = rbf_kernel(X, Y, gamma=0.1)
    params = {"gamma": 0.1, "blabla": ":)"}
    K2 = pairwise_kernels(X, Y, metric="rbf", filter_params=True, **params)
    assert_allclose(K, K2)

    with pytest.raises(TypeError):
        pairwise_kernels(X, Y, metric="rbf", **params)


@pytest.mark.parametrize("metric, func", PAIRED_DISTANCES.items())
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_paired_distances(metric, func, csr_container):
    # [FUNCTION DOC]
    # Verifies paired_distances for each PAIRED_DISTANCES metric matches the underlying
    #   function on dense and CSR input, and equals the diagonal of the corresponding
    #   pairwise distance when available.
    # Test the pairwise_distance helper function.
    rng = np.random.RandomState(0)
    # Euclidean distance should be equivalent to calling the function.
    X = rng.random_sample((5, 4))
    # Euclidean distance, with Y != X.
    Y = rng.random_sample((5, 4))

    S = paired_distances(X, Y, metric=metric)
    S2 = func(X, Y)
    assert_allclose(S, S2)
    S3 = func(csr_container(X), csr_container(Y))
    assert_allclose(S, S3)
    if metric in PAIRWISE_DISTANCE_FUNCTIONS:
        # Check the pairwise_distances implementation
        # gives the same value
        distances = PAIRWISE_DISTANCE_FUNCTIONS[metric](X, Y)
        distances = np.diag(distances)
        assert_allclose(distances, S)


def test_paired_distances_callable(global_dtype):
    # [FUNCTION DOC]
    # Verifies paired_distances with a callable manhattan matches metric="manhattan",
    #   and that X and Y with different numbers of rows raise ValueError.
    # Test the paired_distance helper function
    # with the callable implementation
    rng = np.random.RandomState(0)
    # Euclidean distance should be equivalent to calling the function.
    X = rng.random_sample((5, 4)).astype(global_dtype, copy=False)
    # Euclidean distance, with Y != X.
    Y = rng.random_sample((5, 4)).astype(global_dtype, copy=False)

    S = paired_distances(X, Y, metric="manhattan")
    S2 = paired_distances(X, Y, metric=lambda x, y: np.abs(x - y).sum(axis=0))
    assert_allclose(S, S2)

    # Test that a value error is raised when the lengths of X and Y should not
    # differ
    Y = rng.random_sample((3, 4))
    with pytest.raises(ValueError):
        paired_distances(X, Y)


# XXX: thread-safety bug tracked at:
# https://github.com/scikit-learn/scikit-learn/issues/31884
@pytest.mark.thread_unsafe
@pytest.mark.parametrize("dok_container", DOK_CONTAINERS)
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_pairwise_distances_argmin_min(dok_container, csr_container, global_dtype):
    # [FUNCTION DOC]
    # Verifies pairwise_distances_argmin_min and pairwise_distances_argmin on a tiny 1-D
    #   example for euclidean, sqeuclidean (both spellings), manhattan and scipy
    #   minkowski, on dense and sparse (DOK/CSR) inputs returning ndarrays; then on
    #   random data compares to a naive argmin along axis 0, checks axis/argument-swap
    #   symmetry and C- vs F-contiguous equivalence.
    # Check pairwise minimum distances computation for any metric
    X = np.asarray([[0], [1]], dtype=global_dtype)
    Y = np.asarray([[-2], [3]], dtype=global_dtype)

    Xsp = dok_container(X)
    Ysp = csr_container(Y, dtype=global_dtype)

    expected_idx = [0, 1]
    expected_vals = [2, 2]
    expected_vals_sq = [4, 4]

    # euclidean metric
    idx, vals = pairwise_distances_argmin_min(X, Y, metric="euclidean")
    idx2 = pairwise_distances_argmin(X, Y, metric="euclidean")
    assert_allclose(idx, expected_idx)
    assert_allclose(idx2, expected_idx)
    assert_allclose(vals, expected_vals)
    # sparse matrix case
    idxsp, valssp = pairwise_distances_argmin_min(Xsp, Ysp, metric="euclidean")
    idxsp2 = pairwise_distances_argmin(Xsp, Ysp, metric="euclidean")
    assert_allclose(idxsp, expected_idx)
    assert_allclose(idxsp2, expected_idx)
    assert_allclose(valssp, expected_vals)
    # We don't want np.matrix here
    assert type(idxsp) == np.ndarray
    assert type(valssp) == np.ndarray

    # Squared Euclidean metric
    idx, vals = pairwise_distances_argmin_min(X, Y, metric="sqeuclidean")
    idx2, vals2 = pairwise_distances_argmin_min(
        X, Y, metric="euclidean", metric_kwargs={"squared": True}
    )
    idx3 = pairwise_distances_argmin(X, Y, metric="sqeuclidean")
    idx4 = pairwise_distances_argmin(
        X, Y, metric="euclidean", metric_kwargs={"squared": True}
    )

    assert_allclose(vals, expected_vals_sq)
    assert_allclose(vals2, expected_vals_sq)

    assert_allclose(idx, expected_idx)
    assert_allclose(idx2, expected_idx)
    assert_allclose(idx3, expected_idx)
    assert_allclose(idx4, expected_idx)

    # Non-euclidean scikit-learn metric
    idx, vals = pairwise_distances_argmin_min(X, Y, metric="manhattan")
    idx2 = pairwise_distances_argmin(X, Y, metric="manhattan")
    assert_allclose(idx, expected_idx)
    assert_allclose(idx2, expected_idx)
    assert_allclose(vals, expected_vals)
    # sparse matrix case
    idxsp, valssp = pairwise_distances_argmin_min(Xsp, Ysp, metric="manhattan")
    idxsp2 = pairwise_distances_argmin(Xsp, Ysp, metric="manhattan")
    assert_allclose(idxsp, expected_idx)
    assert_allclose(idxsp2, expected_idx)
    assert_allclose(valssp, expected_vals)

    # Non-euclidean Scipy distance (callable)
    idx, vals = pairwise_distances_argmin_min(
        X, Y, metric=minkowski, metric_kwargs={"p": 2}
    )
    assert_allclose(idx, expected_idx)
    assert_allclose(vals, expected_vals)

    # Non-euclidean Scipy distance (string)
    idx, vals = pairwise_distances_argmin_min(
        X, Y, metric="minkowski", metric_kwargs={"p": 2}
    )
    assert_allclose(idx, expected_idx)
    assert_allclose(vals, expected_vals)

    # Compare with naive implementation
    rng = np.random.RandomState(0)
    X = rng.randn(97, 149)
    Y = rng.randn(111, 149)

    dist = pairwise_distances(X, Y, metric="manhattan")
    dist_orig_ind = dist.argmin(axis=0)
    dist_orig_val = dist[dist_orig_ind, range(len(dist_orig_ind))]

    dist_chunked_ind, dist_chunked_val = pairwise_distances_argmin_min(
        X, Y, axis=0, metric="manhattan"
    )
    assert_allclose(dist_orig_ind, dist_chunked_ind, rtol=1e-7)
    assert_allclose(dist_orig_val, dist_chunked_val, rtol=1e-7)

    # Changing the axis and permuting datasets must give the same results
    argmin_0, dist_0 = pairwise_distances_argmin_min(X, Y, axis=0)
    argmin_1, dist_1 = pairwise_distances_argmin_min(Y, X, axis=1)

    assert_allclose(dist_0, dist_1)
    assert_array_equal(argmin_0, argmin_1)

    argmin_0, dist_0 = pairwise_distances_argmin_min(X, X, axis=0)
    argmin_1, dist_1 = pairwise_distances_argmin_min(X, X, axis=1)

    assert_allclose(dist_0, dist_1)
    assert_array_equal(argmin_0, argmin_1)

    # Changing the axis and permuting datasets must give the same results
    argmin_0 = pairwise_distances_argmin(X, Y, axis=0)
    argmin_1 = pairwise_distances_argmin(Y, X, axis=1)

    assert_array_equal(argmin_0, argmin_1)

    argmin_0 = pairwise_distances_argmin(X, X, axis=0)
    argmin_1 = pairwise_distances_argmin(X, X, axis=1)

    assert_array_equal(argmin_0, argmin_1)

    # F-contiguous arrays must be supported and must return identical results.
    argmin_C_contiguous = pairwise_distances_argmin(X, Y)
    argmin_F_contiguous = pairwise_distances_argmin(
        np.asfortranarray(X), np.asfortranarray(Y)
    )

    assert_array_equal(argmin_C_contiguous, argmin_F_contiguous)


def _reduce_func(dist, start):
    # [FUNCTION DOC]
    # HELPER: reduce_func for pairwise_distances_chunked that keeps the first 100
    #   columns of each distance chunk; start is ignored.
    return dist[:, :100]


def test_pairwise_distances_chunked_reduce(global_dtype):
    # [FUNCTION DOC]
    # Verifies pairwise_distances_chunked with a reduce_func and tiny working_memory
    #   returns a generator of several chunks preserving dtype whose vertical stack
    #   equals the reduced full pairwise distances.
    rng = np.random.RandomState(0)
    X = rng.random_sample((400, 4)).astype(global_dtype, copy=False)
    # Reduced Euclidean distance
    S = pairwise_distances(X)[:, :100]
    S_chunks = pairwise_distances_chunked(
        X, None, reduce_func=_reduce_func, working_memory=2**-16
    )
    assert isinstance(S_chunks, GeneratorType)
    S_chunks = list(S_chunks)
    assert len(S_chunks) > 1
    assert S_chunks[0].dtype == X.dtype

    # atol is for diagonal where S is explicitly zeroed on the diagonal
    assert_allclose(np.vstack(S_chunks), S, atol=1e-7)


def test_pairwise_distances_chunked_reduce_none(global_dtype):
    # [FUNCTION DOC]
    # Verifies a reduce_func returning None yields a generator of several None chunks.
    # check that the reduce func is allowed to return None
    rng = np.random.RandomState(0)
    X = rng.random_sample((10, 4)).astype(global_dtype, copy=False)
    S_chunks = pairwise_distances_chunked(
        X, None, reduce_func=lambda dist, start: None, working_memory=2**-16
    )
    assert isinstance(S_chunks, GeneratorType)
    S_chunks = list(S_chunks)
    assert len(S_chunks) > 1
    assert all(chunk is None for chunk in S_chunks)


@pytest.mark.parametrize(
    "good_reduce",
    [
        lambda D, start: list(D),
        lambda D, start: np.array(D),
        lambda D, start: (list(D), list(D)),
    ]
    + [
        lambda D, start, scipy_csr_type=scipy_csr_type: scipy_csr_type(D)
        for scipy_csr_type in CSR_CONTAINERS
    ]
    + [
        lambda D, start, scipy_dok_type=scipy_dok_type: (
            scipy_dok_type(D),
            np.array(D),
            list(D),
        )
        for scipy_dok_type in DOK_CONTAINERS
    ],
)
def test_pairwise_distances_chunked_reduce_valid(good_reduce):
    # [FUNCTION DOC]
    # Verifies various valid reduce_func return types (list, array, tuple of lists, CSR,
    #   tuple with DOK/array/list) are accepted for the first chunk.
    X = np.arange(10).reshape(-1, 1)
    S_chunks = pairwise_distances_chunked(
        X, None, reduce_func=good_reduce, working_memory=64
    )
    next(S_chunks)


@pytest.mark.parametrize(
    ("bad_reduce", "err_type", "message"),
    [
        (
            lambda D, s: np.concatenate([D, D[-1:]]),
            ValueError,
            r"length 11\..* input: 10\.",
        ),
        (
            lambda D, s: (D, np.concatenate([D, D[-1:]])),
            ValueError,
            r"length \(10, 11\)\..* input: 10\.",
        ),
        (lambda D, s: (D[:9], D), ValueError, r"length \(9, 10\)\..* input: 10\."),
        (
            lambda D, s: 7,
            TypeError,
            r"returned 7\. Expected sequence\(s\) of length 10\.",
        ),
        (
            lambda D, s: (7, 8),
            TypeError,
            r"returned \(7, 8\)\. Expected sequence\(s\) of length 10\.",
        ),
        (
            lambda D, s: (np.arange(10), 9),
            TypeError,
            r", 9\)\. Expected sequence\(s\) of length 10\.",
        ),
    ],
)
def test_pairwise_distances_chunked_reduce_invalid(
    global_dtype, bad_reduce, err_type, message
):
    # [FUNCTION DOC]
    # Verifies reduce_func outputs of wrong length raise ValueError and non-sequence
    #   outputs raise TypeError, with messages naming the expected length 10.
    X = np.arange(10).reshape(-1, 1).astype(global_dtype, copy=False)
    S_chunks = pairwise_distances_chunked(
        X, None, reduce_func=bad_reduce, working_memory=64
    )
    with pytest.raises(err_type, match=message):
        next(S_chunks)


def check_pairwise_distances_chunked(X, Y, working_memory, metric="euclidean"):
    # [FUNCTION DOC]
    # HELPER: Checks pairwise_distances_chunked for given X, Y, working_memory (MiB) and
    #   metric.
    # Asserts the result is a generator, every block's byte size is at most the larger
    #   of working_memory and one row of float64 distances (in MiB), and the stacked
    #   blocks equal pairwise_distances within atol 1e-7.
    # Exceptions: AssertionError on violation.
    gen = pairwise_distances_chunked(X, Y, working_memory=working_memory, metric=metric)
    assert isinstance(gen, GeneratorType)
    blockwise_distances = list(gen)
    Y = X if Y is None else Y
    min_block_mib = len(Y) * 8 * 2**-20

    for block in blockwise_distances:
        memory_used = block.nbytes
        assert memory_used <= max(working_memory, min_block_mib) * 2**20

    blockwise_distances = np.vstack(blockwise_distances)
    S = pairwise_distances(X, Y, metric=metric)
    assert_allclose(blockwise_distances, S, atol=1e-7)


@pytest.mark.parametrize("metric", ("euclidean", "l2", "sqeuclidean"))
def test_pairwise_distances_chunked_diagonal(metric, global_dtype):
    # [FUNCTION DOC]
    # Verifies chunked euclidean/l2/sqeuclidean distances of a large-scale dataset with
    #   itself have exactly-zero diagonal across multiple chunks.
    rng = np.random.RandomState(0)
    X = rng.normal(size=(1000, 10), scale=1e10).astype(global_dtype, copy=False)
    chunks = list(pairwise_distances_chunked(X, working_memory=1, metric=metric))
    assert len(chunks) > 1
    assert_allclose(np.diag(np.vstack(chunks)), 0, rtol=1e-10)


@pytest.mark.parametrize("metric", ("euclidean", "l2", "sqeuclidean"))
def test_parallel_pairwise_distances_diagonal(metric, global_dtype):
    # [FUNCTION DOC]
    # Verifies parallel (n_jobs=2) euclidean-type pairwise distances keep a zero
    #   diagonal.
    rng = np.random.RandomState(0)
    X = rng.normal(size=(1000, 10), scale=1e10).astype(global_dtype, copy=False)
    distances = pairwise_distances(X, metric=metric, n_jobs=2)
    assert_allclose(np.diag(distances), 0, atol=1e-10)


def test_parallel_pairwise_distances_y_norm_squared():
    # [FUNCTION DOC]
    # Non-regression test for #33877: verifies Y_norm_squared passed to parallel
    #   pairwise_distances is sliced together with Y, giving the same result as
    #   n_jobs=1.
    """Check that Y_norm_squared is correctly sliced alongside Y.

    Non-regression test for issue #33877.
    """
    rng = np.random.RandomState(42)
    X = rng.rand(13, 4)
    Y = rng.rand(15, 4)
    Y_norm_squared = (Y**2).sum(axis=1)

    D_single = pairwise_distances(
        X, Y, metric="euclidean", n_jobs=1, Y_norm_squared=Y_norm_squared
    )
    D_parallel = pairwise_distances(
        X, Y, metric="euclidean", n_jobs=2, Y_norm_squared=Y_norm_squared
    )
    assert_allclose(D_parallel, D_single)


@pytest.mark.filterwarnings("ignore:Could not adhere to working_memory config")
def test_pairwise_distances_chunked(global_dtype):
    # [FUNCTION DOC]
    # Verifies pairwise_distances_chunked via check_pairwise_distances_chunked for many
    #   working_memory sizes, list inputs, Y given, a huge working_memory and cityblock,
    #   and that metric="precomputed" yields the input matrix itself as the single
    #   chunk.
    # Test the pairwise_distance helper function.
    rng = np.random.RandomState(0)
    # Euclidean distance should be equivalent to calling the function.
    X = rng.random_sample((200, 4)).astype(global_dtype, copy=False)
    check_pairwise_distances_chunked(X, None, working_memory=1, metric="euclidean")
    # Test small amounts of memory
    for power in range(-16, 0):
        check_pairwise_distances_chunked(
            X, None, working_memory=2**power, metric="euclidean"
        )
    # X as list
    check_pairwise_distances_chunked(
        X.tolist(), None, working_memory=1, metric="euclidean"
    )
    # Euclidean distance, with Y != X.
    Y = rng.random_sample((100, 4)).astype(global_dtype, copy=False)
    check_pairwise_distances_chunked(X, Y, working_memory=1, metric="euclidean")
    check_pairwise_distances_chunked(
        X.tolist(), Y.tolist(), working_memory=1, metric="euclidean"
    )
    # absurdly large working_memory
    check_pairwise_distances_chunked(X, Y, working_memory=10000, metric="euclidean")
    # "cityblock" uses scikit-learn metric, cityblock (function) is
    # scipy.spatial.
    check_pairwise_distances_chunked(X, Y, working_memory=1, metric="cityblock")

    # Test precomputed returns all at once
    D = pairwise_distances(X)
    gen = pairwise_distances_chunked(D, working_memory=2**-16, metric="precomputed")
    assert isinstance(gen, GeneratorType)
    assert next(gen) is D
    with pytest.raises(StopIteration):
        next(gen)


@pytest.mark.parametrize(
    "x_array_constr",
    [np.array] + CSR_CONTAINERS,
    ids=["dense"] + [container.__name__ for container in CSR_CONTAINERS],
)
@pytest.mark.parametrize(
    "y_array_constr",
    [np.array] + CSR_CONTAINERS,
    ids=["dense"] + [container.__name__ for container in CSR_CONTAINERS],
)
def test_euclidean_distances_known_result(x_array_constr, y_array_constr):
    # [FUNCTION DOC]
    # Verifies euclidean_distances of [[0]] to [[1],[2]] is [[1, 2]] for every dense/CSR
    #   combination.
    # Check the pairwise Euclidean distances computation on known result
    X = x_array_constr([[0]])
    Y = y_array_constr([[1], [2]])
    D = euclidean_distances(X, Y)
    assert_allclose(D, [[1.0, 2.0]])


@pytest.mark.parametrize(
    "y_array_constr",
    [np.array] + CSR_CONTAINERS,
    ids=["dense"] + [container.__name__ for container in CSR_CONTAINERS],
)
def test_euclidean_distances_with_norms(global_dtype, y_array_constr):
    # [FUNCTION DOC]
    # Verifies euclidean_distances gives the same result when correct float64 X/Y
    #   squared norms are supplied, and a different result when zero norms are supplied.
    # check that we still get the right answers with {X,Y}_norm_squared
    # and that we get a wrong answer with wrong {X,Y}_norm_squared
    rng = np.random.RandomState(0)
    X = rng.random_sample((10, 10)).astype(global_dtype, copy=False)
    Y = rng.random_sample((20, 10)).astype(global_dtype, copy=False)

    # norms will only be used if their dtype is float64
    X_norm_sq = (X.astype(np.float64) ** 2).sum(axis=1).reshape(1, -1)
    Y_norm_sq = (Y.astype(np.float64) ** 2).sum(axis=1).reshape(1, -1)

    Y = y_array_constr(Y)

    D1 = euclidean_distances(X, Y)
    D2 = euclidean_distances(X, Y, X_norm_squared=X_norm_sq)
    D3 = euclidean_distances(X, Y, Y_norm_squared=Y_norm_sq)
    D4 = euclidean_distances(X, Y, X_norm_squared=X_norm_sq, Y_norm_squared=Y_norm_sq)
    assert_allclose(D2, D1)
    assert_allclose(D3, D1)
    assert_allclose(D4, D1)

    # check we get the wrong answer with wrong {X,Y}_norm_squared
    wrong_D = euclidean_distances(
        X,
        Y,
        X_norm_squared=np.zeros_like(X_norm_sq),
        Y_norm_squared=np.zeros_like(Y_norm_sq),
    )
    with pytest.raises(AssertionError):
        assert_allclose(wrong_D, D1)


@pytest.mark.parametrize("symmetric", [True, False])
def test_euclidean_distances_float32_norms(global_random_seed, symmetric):
    # [FUNCTION DOC]
    # Non-regression test for #27621: verifies float32-precomputed squared norms on
    #   float64 data give results matching no norms, for symmetric and non-symmetric
    #   cases.
    # Non-regression test for #27621
    rng = np.random.RandomState(global_random_seed)
    X = rng.random_sample((10, 10))
    Y = X if symmetric else rng.random_sample((20, 10))
    X_norm_sq = (X.astype(np.float32) ** 2).sum(axis=1).reshape(1, -1)
    Y_norm_sq = (Y.astype(np.float32) ** 2).sum(axis=1).reshape(1, -1)
    D1 = euclidean_distances(X, Y)
    D2 = euclidean_distances(X, Y, X_norm_squared=X_norm_sq)
    D3 = euclidean_distances(X, Y, Y_norm_squared=Y_norm_sq)
    D4 = euclidean_distances(X, Y, X_norm_squared=X_norm_sq, Y_norm_squared=Y_norm_sq)
    assert_allclose(D2, D1)
    assert_allclose(D3, D1)
    assert_allclose(D4, D1)


def test_euclidean_distances_norm_shapes():
    # [FUNCTION DOC]
    # Verifies squared norms are accepted as 1-D, column and row vectors, and that norms
    #   of the wrong length raise ValueError naming X or Y.
    # Check all accepted shapes for the norms or appropriate error messages.
    rng = np.random.RandomState(0)
    X = rng.random_sample((10, 10))
    Y = rng.random_sample((20, 10))

    X_norm_squared = (X**2).sum(axis=1)
    Y_norm_squared = (Y**2).sum(axis=1)

    D1 = euclidean_distances(
        X, Y, X_norm_squared=X_norm_squared, Y_norm_squared=Y_norm_squared
    )
    D2 = euclidean_distances(
        X,
        Y,
        X_norm_squared=X_norm_squared.reshape(-1, 1),
        Y_norm_squared=Y_norm_squared.reshape(-1, 1),
    )
    D3 = euclidean_distances(
        X,
        Y,
        X_norm_squared=X_norm_squared.reshape(1, -1),
        Y_norm_squared=Y_norm_squared.reshape(1, -1),
    )

    assert_allclose(D2, D1)
    assert_allclose(D3, D1)

    with pytest.raises(ValueError, match="Incompatible dimensions for X"):
        euclidean_distances(X, Y, X_norm_squared=X_norm_squared[:5])
    with pytest.raises(ValueError, match="Incompatible dimensions for Y"):
        euclidean_distances(X, Y, Y_norm_squared=Y_norm_squared[:5])


@pytest.mark.parametrize(
    "x_array_constr",
    [np.array] + CSR_CONTAINERS,
    ids=["dense"] + [container.__name__ for container in CSR_CONTAINERS],
)
@pytest.mark.parametrize(
    "y_array_constr",
    [np.array] + CSR_CONTAINERS,
    ids=["dense"] + [container.__name__ for container in CSR_CONTAINERS],
)
def test_euclidean_distances(global_dtype, x_array_constr, y_array_constr):
    # [FUNCTION DOC]
    # Verifies euclidean_distances on sparse-ish data matches scipy cdist for every
    #   dense/CSR combination, preserving dtype.
    # check that euclidean distances gives same result as scipy cdist
    # when X and Y != X are provided
    rng = np.random.RandomState(0)
    X = rng.random_sample((100, 10)).astype(global_dtype, copy=False)
    X[X < 0.8] = 0
    Y = rng.random_sample((10, 10)).astype(global_dtype, copy=False)
    Y[Y < 0.8] = 0

    expected = cdist(X, Y)

    X = x_array_constr(X)
    Y = y_array_constr(Y)
    distances = euclidean_distances(X, Y)

    # the default rtol=1e-7 is too close to the float32 precision
    # and fails due to rounding errors.
    assert_allclose(distances, expected, rtol=1e-6)
    assert distances.dtype == global_dtype


@pytest.mark.parametrize(
    "x_array_constr",
    [np.array] + CSR_CONTAINERS,
    ids=["dense"] + [container.__name__ for container in CSR_CONTAINERS],
)
def test_euclidean_distances_sym(global_dtype, x_array_constr):
    # [FUNCTION DOC]
    # Verifies euclidean_distances(X) matches squareform(pdist(X)) for dense and CSR
    #   input, preserving dtype.
    # check that euclidean distances gives same result as scipy pdist
    # when only X is provided
    rng = np.random.RandomState(0)
    X = rng.random_sample((100, 10)).astype(global_dtype, copy=False)
    X[X < 0.8] = 0

    expected = squareform(pdist(X))

    X = x_array_constr(X)
    distances = euclidean_distances(X)

    # the default rtol=1e-7 is too close to the float32 precision
    # and fails due to rounding errors.
    assert_allclose(distances, expected, rtol=1e-6)
    assert distances.dtype == global_dtype


@pytest.mark.parametrize("batch_size", [None, 5, 7, 101])
@pytest.mark.parametrize(
    "x_array_constr",
    [np.array] + CSR_CONTAINERS,
    ids=["dense"] + [container.__name__ for container in CSR_CONTAINERS],
)
@pytest.mark.parametrize(
    "y_array_constr",
    [np.array] + CSR_CONTAINERS,
    ids=["dense"] + [container.__name__ for container in CSR_CONTAINERS],
)
def test_euclidean_distances_upcast(batch_size, x_array_constr, y_array_constr):
    # [FUNCTION DOC]
    # Verifies _euclidean_distances_upcast on float32 data with several batch sizes,
    #   after clipping and square root, matches cdist for every dense/CSR combination
    #   (#13910).
    # check batches handling when Y != X (#13910)
    rng = np.random.RandomState(0)
    X = rng.random_sample((100, 10)).astype(np.float32)
    X[X < 0.8] = 0
    Y = rng.random_sample((10, 10)).astype(np.float32)
    Y[Y < 0.8] = 0

    expected = cdist(X, Y)

    X = x_array_constr(X)
    Y = y_array_constr(Y)
    distances = _euclidean_distances_upcast(X, Y=Y, batch_size=batch_size)
    distances = np.sqrt(np.maximum(distances, 0))

    # the default rtol=1e-7 is too close to the float32 precision
    # and fails due to rounding errors.
    assert_allclose(distances, expected, rtol=1e-6)


@pytest.mark.parametrize("batch_size", [None, 5, 7, 101])
@pytest.mark.parametrize(
    "x_array_constr",
    [np.array] + CSR_CONTAINERS,
    ids=["dense"] + [container.__name__ for container in CSR_CONTAINERS],
)
def test_euclidean_distances_upcast_sym(batch_size, x_array_constr):
    # [FUNCTION DOC]
    # Verifies _euclidean_distances_upcast with Y equal to X matches squareform(pdist)
    #   for several batch sizes.
    # check batches handling when X is Y (#13910)
    rng = np.random.RandomState(0)
    X = rng.random_sample((100, 10)).astype(np.float32)
    X[X < 0.8] = 0

    expected = squareform(pdist(X))

    X = x_array_constr(X)
    distances = _euclidean_distances_upcast(X, Y=X, batch_size=batch_size)
    distances = np.sqrt(np.maximum(distances, 0))

    # the default rtol=1e-7 is too close to the float32 precision
    # and fails due to rounding errors.
    assert_allclose(distances, expected, rtol=1e-6)


@pytest.mark.parametrize(
    "dtype, eps, rtol",
    [
        (np.float32, 1e-4, 1e-5),
        pytest.param(
            np.float64,
            1e-8,
            0.99,
            marks=pytest.mark.xfail(reason="failing due to lack of precision"),
        ),
    ],
)
@pytest.mark.parametrize("dim", [1, 1000000])
def test_euclidean_distances_extreme_values(dtype, eps, rtol, dim):
    # [FUNCTION DOC]
    # Verifies euclidean_distances is accurate for nearly equal vectors of dimension 1
    #   or 1e6 in float32 (float64 case marked xfail).
    # KNOWN ISSUE: the parametrized rtol is ignored; the assertion uses a hard-coded
    #   rtol of 1e-5.
    # check that euclidean distances is correct with float32 input thanks to
    # upcasting. On float64 there are still precision issues.
    X = np.array([[1.0] * dim], dtype=dtype)
    Y = np.array([[1.0 + eps] * dim], dtype=dtype)

    distances = euclidean_distances(X, Y)
    expected = cdist(X, Y)

    assert_allclose(distances, expected, rtol=1e-5)


@pytest.mark.parametrize("squared", [True, False])
def test_nan_euclidean_distances_equal_to_euclidean_distance(squared):
    # [FUNCTION DOC]
    # Verifies nan_euclidean_distances equals euclidean_distances on data without
    #   missing values, squared or not.
    # with no nan values
    rng = np.random.RandomState(1337)
    X = rng.randn(3, 4)
    Y = rng.randn(4, 4)

    normal_distance = euclidean_distances(X, Y=Y, squared=squared)
    nan_distance = nan_euclidean_distances(X, Y=Y, squared=squared)
    assert_allclose(normal_distance, nan_distance)


@pytest.mark.parametrize("X", [np.array([[np.inf, 0]]), np.array([[0, -np.inf]])])
@pytest.mark.parametrize("Y", [np.array([[np.inf, 0]]), np.array([[0, -np.inf]]), None])
def test_nan_euclidean_distances_infinite_values(X, Y):
    # [FUNCTION DOC]
    # Verifies nan_euclidean_distances raises ValueError with the exact infinity message
    #   when X or Y contains +/-inf.
    with pytest.raises(ValueError) as excinfo:
        nan_euclidean_distances(X, Y=Y)

    exp_msg = "Input contains infinity or a value too large for dtype('float64')."
    assert exp_msg == str(excinfo.value)


@pytest.mark.parametrize(
    "X, X_diag, missing_value",
    [
        (np.array([[0, 1], [1, 0]]), np.sqrt(2), np.nan),
        (np.array([[0, 1], [1, np.nan]]), np.sqrt(2), np.nan),
        (np.array([[np.nan, 1], [1, np.nan]]), np.nan, np.nan),
        (np.array([[np.nan, 1], [np.nan, 0]]), np.sqrt(2), np.nan),
        (np.array([[0, np.nan], [1, np.nan]]), np.sqrt(2), np.nan),
        (np.array([[0, 1], [1, 0]]), np.sqrt(2), -1),
        (np.array([[0, 1], [1, -1]]), np.sqrt(2), -1),
        (np.array([[-1, 1], [1, -1]]), np.nan, -1),
        (np.array([[-1, 1], [-1, 0]]), np.sqrt(2), -1),
        (np.array([[0, -1], [1, -1]]), np.sqrt(2), -1),
    ],
)
def test_nan_euclidean_distances_2x2(X, X_diag, missing_value):
    # [FUNCTION DOC]
    # Verifies nan_euclidean_distances on 2x2 inputs with NaN or -1 as missing marker
    #   against known distance matrices, squared and unsquared, and with Y given as X or
    #   a copy of X.
    exp_dist = np.array([[0.0, X_diag], [X_diag, 0]])

    dist = nan_euclidean_distances(X, missing_values=missing_value)
    assert_allclose(exp_dist, dist)

    dist_sq = nan_euclidean_distances(X, squared=True, missing_values=missing_value)
    assert_allclose(exp_dist**2, dist_sq)

    dist_two = nan_euclidean_distances(X, X, missing_values=missing_value)
    assert_allclose(exp_dist, dist_two)

    dist_two_copy = nan_euclidean_distances(X, X.copy(), missing_values=missing_value)
    assert_allclose(exp_dist, dist_two_copy)


@pytest.mark.parametrize("missing_value", [np.nan, -1])
def test_nan_euclidean_distances_complete_nan(missing_value):
    # [FUNCTION DOC]
    # Verifies a row with all values missing yields NaN distances (including to itself)
    #   while other rows behave normally.
    X = np.array([[missing_value, missing_value], [0, 1]])

    exp_dist = np.array([[np.nan, np.nan], [np.nan, 0]])

    dist = nan_euclidean_distances(X, missing_values=missing_value)
    assert_allclose(exp_dist, dist)

    dist = nan_euclidean_distances(X, X.copy(), missing_values=missing_value)
    assert_allclose(exp_dist, dist)


@pytest.mark.parametrize("missing_value", [np.nan, -1])
def test_nan_euclidean_distances_not_trivial(missing_value):
    # [FUNCTION DOC]
    # Verifies nan_euclidean_distances symmetry, the weighted formula (scaling by total
    #   over present coordinates) on hand-computed pairs, equality between Y omitted, Y
    #   is X and Y a copy, and copy=True vs copy=False.
    # KNOWN ISSUE: the copy comparison omits missing_values, so for missing_value -1 it
    #   treats -1 as an ordinary value.
    X = np.array(
        [
            [1.0, missing_value, 3.0, 4.0, 2.0],
            [missing_value, 4.0, 6.0, 1.0, missing_value],
            [3.0, missing_value, missing_value, missing_value, 1.0],
        ]
    )

    Y = np.array(
        [
            [missing_value, 7.0, 7.0, missing_value, 2.0],
            [missing_value, missing_value, 5.0, 4.0, 7.0],
            [missing_value, missing_value, missing_value, 4.0, 5.0],
        ]
    )

    # Check for symmetry
    D1 = nan_euclidean_distances(X, Y, missing_values=missing_value)
    D2 = nan_euclidean_distances(Y, X, missing_values=missing_value)

    assert_almost_equal(D1, D2.T)

    # Check with explicit formula and squared=True
    assert_allclose(
        nan_euclidean_distances(
            X[:1], Y[:1], squared=True, missing_values=missing_value
        ),
        [[5.0 / 2.0 * ((7 - 3) ** 2 + (2 - 2) ** 2)]],
    )

    # Check with explicit formula and squared=False
    assert_allclose(
        nan_euclidean_distances(
            X[1:2], Y[1:2], squared=False, missing_values=missing_value
        ),
        [[np.sqrt(5.0 / 2.0 * ((6 - 5) ** 2 + (1 - 4) ** 2))]],
    )

    # Check when Y = X is explicitly passed
    D3 = nan_euclidean_distances(X, missing_values=missing_value)
    D4 = nan_euclidean_distances(X, X, missing_values=missing_value)
    D5 = nan_euclidean_distances(X, X.copy(), missing_values=missing_value)
    assert_allclose(D3, D4)
    assert_allclose(D4, D5)

    # Check copy = True against copy = False
    D6 = nan_euclidean_distances(X, Y, copy=True)
    D7 = nan_euclidean_distances(X, Y, copy=False)
    assert_allclose(D6, D7)


@pytest.mark.parametrize("missing_value", [np.nan, -1])
def test_nan_euclidean_distances_one_feature_match_positive(missing_value):
    # [FUNCTION DOC]
    # Verifies that when only one coordinate is shared, squared distances are non-
    #   negative and unsquared distances are close to 0.
    # First feature is the only feature that is non-nan and in both
    # samples. The result of `nan_euclidean_distances` with squared=True
    # should be non-negative. The non-squared version should all be close to 0.
    X = np.array(
        [
            [-122.27, 648.0, missing_value, 37.85],
            [-122.27, missing_value, 2.34701493, missing_value],
        ]
    )

    dist_squared = nan_euclidean_distances(
        X, missing_values=missing_value, squared=True
    )
    assert np.all(dist_squared >= 0)

    dist = nan_euclidean_distances(X, missing_values=missing_value, squared=False)
    assert_allclose(dist, 0.0)


def test_cosine_distances():
    # [FUNCTION DOC]
    # Verifies cosine_distances lies in [0, 2], is zero on the diagonal and for
    #   identical rows, is 2 for opposite rows, and stays valid for a large random
    #   matrix.
    # Check the pairwise Cosine distances computation
    rng = np.random.RandomState(1337)
    x = np.abs(rng.rand(910))
    XA = np.vstack([x, x])
    D = cosine_distances(XA)
    assert_allclose(D, [[0.0, 0.0], [0.0, 0.0]], atol=1e-10)
    # check that all elements are in [0, 2]
    assert np.all(D >= 0.0)
    assert np.all(D <= 2.0)
    # check that diagonal elements are equal to 0
    assert_allclose(D[np.diag_indices_from(D)], [0.0, 0.0])

    XB = np.vstack([x, -x])
    D2 = cosine_distances(XB)
    # check that all elements are in [0, 2]
    assert np.all(D2 >= 0.0)
    assert np.all(D2 <= 2.0)
    # check that diagonal elements are equal to 0 and non diagonal to 2
    assert_allclose(D2, [[0.0, 2.0], [2.0, 0.0]])

    # check large random matrix
    X = np.abs(rng.rand(1000, 5000))
    D = cosine_distances(X)
    # check that diagonal elements are equal to 0
    assert_allclose(D[np.diag_indices_from(D)], [0.0] * D.shape[0])
    assert np.all(D >= 0.0)
    assert np.all(D <= 2.0)


def test_haversine_distances():
    # [FUNCTION DOC]
    # Verifies haversine_distances matches a slow per-pair reference and raises
    #   ValueError for data with other than 2 features.
    # Check haversine distance with distances computation
    def slow_haversine_distances(x, y):
        # [FUNCTION DOC]
        # HELPER: Reference haversine distance between points x and y given as
        #   (latitude, longitude) in radians; returns the great-circle angle as a float.
        diff_lat = y[0] - x[0]
        diff_lon = y[1] - x[1]
        a = np.sin(diff_lat / 2) ** 2 + (
            np.cos(x[0]) * np.cos(y[0]) * np.sin(diff_lon / 2) ** 2
        )
        c = 2 * np.arcsin(np.sqrt(a))
        return c

    rng = np.random.RandomState(0)
    X = rng.random_sample((5, 2))
    Y = rng.random_sample((10, 2))
    D1 = np.array([[slow_haversine_distances(x, y) for y in Y] for x in X])
    D2 = haversine_distances(X, Y)
    assert_allclose(D1, D2)
    # Test haversine distance does not accept X where n_feature != 2
    X = rng.random_sample((10, 3))
    err_msg = "Haversine distance only valid in 2 dimensions"
    with pytest.raises(ValueError, match=err_msg):
        haversine_distances(X)


# Paired distances


def test_paired_euclidean_distances():
    # [FUNCTION DOC]
    # Verifies paired_euclidean_distances on a tiny known example.
    # Check the paired Euclidean distances computation
    X = [[0], [0]]
    Y = [[1], [2]]
    D = paired_euclidean_distances(X, Y)
    assert_allclose(D, [1.0, 2.0])


def test_paired_manhattan_distances():
    # [FUNCTION DOC]
    # Verifies paired_manhattan_distances on a tiny known example.
    # Check the paired manhattan distances computation
    X = [[0], [0]]
    Y = [[1], [2]]
    D = paired_manhattan_distances(X, Y)
    assert_allclose(D, [1.0, 2.0])


def test_paired_cosine_distances():
    # [FUNCTION DOC]
    # Verifies paired_cosine_distances on a tiny known example (expected 0.5 because the
    #   zero vector has undefined direction).
    # KNOWN ISSUE: the inline comment says manhattan although the cosine function is
    #   tested.
    # Check the paired manhattan distances computation
    X = [[0], [0]]
    Y = [[1], [2]]
    D = paired_cosine_distances(X, Y)
    assert_allclose(D, [0.5, 0.5])


def test_chi_square_kernel():
    # [FUNCTION DOC]
    # Verifies additive_chi2_kernel and chi2_kernel against per-pair formulas, unit
    #   diagonal and off-diagonal in (0, 1), float32 preservation, integer input
    #   converted to float with zeros handled, similarity ordering, and ValueError on
    #   negative inputs or mismatched feature counts.
    rng = np.random.RandomState(0)
    X = rng.random_sample((5, 4))
    Y = rng.random_sample((10, 4))
    K_add = additive_chi2_kernel(X, Y)
    gamma = 0.1
    K = chi2_kernel(X, Y, gamma=gamma)
    assert K.dtype == float
    for i, x in enumerate(X):
        for j, y in enumerate(Y):
            chi2 = -np.sum((x - y) ** 2 / (x + y))
            chi2_exp = np.exp(gamma * chi2)
            assert_almost_equal(K_add[i, j], chi2)
            assert_almost_equal(K[i, j], chi2_exp)

    # check diagonal is ones for data with itself
    K = chi2_kernel(Y)
    assert_array_equal(np.diag(K), 1)
    # check off-diagonal is < 1 but > 0:
    assert np.all(K > 0)
    assert np.all(K - np.diag(np.diag(K)) < 1)
    # check that float32 is preserved
    X = rng.random_sample((5, 4)).astype(np.float32)
    Y = rng.random_sample((10, 4)).astype(np.float32)
    K = chi2_kernel(X, Y)
    assert K.dtype == np.float32

    # check integer type gets converted,
    # check that zeros are handled
    X = rng.random_sample((10, 4)).astype(np.int32)
    K = chi2_kernel(X, X)
    assert np.isfinite(K).all()
    assert K.dtype == float

    # check that kernel of similar things is greater than dissimilar ones
    X = [[0.3, 0.7], [1.0, 0]]
    Y = [[0, 1], [0.9, 0.1]]
    K = chi2_kernel(X, Y)
    assert K[0, 0] > K[0, 1]
    assert K[1, 1] > K[1, 0]

    # test negative input
    with pytest.raises(ValueError):
        chi2_kernel([[0, -1]])
    with pytest.raises(ValueError):
        chi2_kernel([[0, -1]], [[-1, -1]])
    with pytest.raises(ValueError):
        chi2_kernel([[0, 1]], [[-1, -1]])

    # different n_features in X and Y
    with pytest.raises(ValueError):
        chi2_kernel([[0, 1]], [[0.2, 0.2, 0.6]])


@pytest.mark.parametrize(
    "kernel",
    (
        linear_kernel,
        polynomial_kernel,
        rbf_kernel,
        laplacian_kernel,
        sigmoid_kernel,
        cosine_similarity,
    ),
)
def test_kernel_symmetry(kernel):
    # [FUNCTION DOC]
    # Verifies common kernels produce a symmetric matrix on X with itself.
    # KNOWN ISSUE: the third positional argument 15 is interpreted as rtol, making the
    #   check extremely loose (probably meant as a decimal count).
    # Valid kernels should be symmetric
    rng = np.random.RandomState(0)
    X = rng.random_sample((5, 4))
    K = kernel(X, X)
    assert_allclose(K, K.T, 15)


@pytest.mark.parametrize(
    "kernel",
    (
        linear_kernel,
        polynomial_kernel,
        rbf_kernel,
        laplacian_kernel,
        sigmoid_kernel,
        cosine_similarity,
    ),
)
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_kernel_sparse(kernel, csr_container):
    # [FUNCTION DOC]
    # Verifies common kernels give the same result on dense and CSR inputs.
    rng = np.random.RandomState(0)
    X = rng.random_sample((5, 4))
    X_sparse = csr_container(X)
    K = kernel(X, X)
    K2 = kernel(X_sparse, X_sparse)
    assert_allclose(K, K2)


def test_linear_kernel():
    # [FUNCTION DOC]
    # Verifies the diagonal of linear_kernel(X, X) equals squared row norms.
    rng = np.random.RandomState(0)
    X = rng.random_sample((5, 4))
    K = linear_kernel(X, X)
    # the diagonal elements of a linear kernel are their squared norm
    assert_allclose(K.flat[::6], [linalg.norm(x) ** 2 for x in X])


def test_rbf_kernel():
    # [FUNCTION DOC]
    # Verifies the diagonal of rbf_kernel(X, X) is all ones.
    rng = np.random.RandomState(0)
    X = rng.random_sample((5, 4))
    K = rbf_kernel(X, X)
    # the diagonal elements of an rbf kernel are 1
    assert_allclose(K.flat[::6], np.ones(5))


def test_laplacian_kernel():
    # [FUNCTION DOC]
    # Verifies laplacian_kernel(X, X) has unit diagonal and off-diagonal values in (0,
    #   1).
    rng = np.random.RandomState(0)
    X = rng.random_sample((5, 4))
    K = laplacian_kernel(X, X)
    # the diagonal elements of a laplacian kernel are 1
    assert_allclose(np.diag(K), np.ones(5))

    # off-diagonal elements are < 1 but > 0:
    assert np.all(K > 0)
    assert np.all(K - np.diag(np.diag(K)) < 1)


@pytest.mark.parametrize(
    "metric, pairwise_func",
    [("linear", linear_kernel), ("cosine", cosine_similarity)],
)
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_pairwise_similarity_sparse_output(metric, pairwise_func, csr_container):
    # [FUNCTION DOC]
    # Verifies linear_kernel/cosine_similarity with dense_output=False on CSR returns
    #   sparse output that equals the dense output and pairwise_kernels.
    rng = np.random.RandomState(0)
    X = rng.random_sample((5, 4))
    Y = rng.random_sample((3, 4))
    Xcsr = csr_container(X)
    Ycsr = csr_container(Y)

    # should be sparse
    K1 = pairwise_func(Xcsr, Ycsr, dense_output=False)
    assert issparse(K1)

    # should be dense, and equal to K1
    K2 = pairwise_func(X, Y, dense_output=True)
    assert not issparse(K2)
    assert_allclose(K1.toarray(), K2)

    # show the kernel output equal to the sparse.toarray()
    K3 = pairwise_kernels(X, Y=Y, metric=metric)
    assert_allclose(K1.toarray(), K3)


@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_cosine_similarity(csr_container):
    # [FUNCTION DOC]
    # Verifies the cosine kernel equals the linear kernel on L2-normalized data, for
    #   dense and CSR X with Y None or given.
    # Test the cosine_similarity.

    rng = np.random.RandomState(0)
    X = rng.random_sample((5, 4))
    Y = rng.random_sample((3, 4))
    Xcsr = csr_container(X)
    Ycsr = csr_container(Y)

    for X_, Y_ in ((X, None), (X, Y), (Xcsr, None), (Xcsr, Ycsr)):
        # Test that the cosine is kernel is equal to a linear kernel when data
        # has been previously normalized by L2-norm.
        K1 = pairwise_kernels(X_, Y=Y_, metric="cosine")
        X_ = normalize(X_)
        if Y_ is not None:
            Y_ = normalize(Y_)
        K2 = pairwise_kernels(X_, Y=Y_, metric="linear")
        assert_allclose(K1, K2)


def test_check_dense_matrices():
    # [FUNCTION DOC]
    # Verifies check_pairwise_arrays with Y None returns the same object twice and
    #   preserves values.
    # Ensure that pairwise array check works for dense matrices.
    # Check that if XB is None, XB is returned as reference to XA
    XA = np.resize(np.arange(40), (5, 8))
    XA_checked, XB_checked = check_pairwise_arrays(XA, None)
    assert XA_checked is XB_checked
    assert_array_equal(XA, XA_checked)


def test_check_XB_returned():
    # [FUNCTION DOC]
    # Verifies check_pairwise_arrays and check_paired_arrays return inputs with
    #   unchanged values.
    # Ensure that if XA and XB are given correctly, they return as equal.
    # Check that if XB is not None, it is returned equal.
    # Note that the second dimension of XB is the same as XA.
    XA = np.resize(np.arange(40), (5, 8))
    XB = np.resize(np.arange(32), (4, 8))
    XA_checked, XB_checked = check_pairwise_arrays(XA, XB)
    assert_array_equal(XA, XA_checked)
    assert_array_equal(XB, XB_checked)

    XB = np.resize(np.arange(40), (5, 8))
    XA_checked, XB_checked = check_paired_arrays(XA, XB)
    assert_array_equal(XA, XA_checked)
    assert_array_equal(XB, XB_checked)


def test_check_different_dimensions():
    # [FUNCTION DOC]
    # Verifies mismatched feature counts raise ValueError in check_pairwise_arrays and
    #   mismatched row counts raise ValueError in check_paired_arrays.
    # Ensure an error is raised if the dimensions are different.
    XA = np.resize(np.arange(45), (5, 9))
    XB = np.resize(np.arange(32), (4, 8))
    with pytest.raises(ValueError):
        check_pairwise_arrays(XA, XB)

    XB = np.resize(np.arange(4 * 9), (4, 9))
    with pytest.raises(ValueError):
        check_paired_arrays(XA, XB)


def test_check_invalid_dimensions():
    # [FUNCTION DOC]
    # Verifies check_pairwise_arrays raises ValueError for inputs with different feature
    #   counts.
    # KNOWN ISSUE: despite the comment about 1D arrays, both blocks use identical 2-D
    #   inputs, so the second check duplicates the first.
    # Ensure an error is raised on 1D input arrays.
    # The modified tests are not 1D. In the old test, the array was internally
    # converted to 2D anyways
    XA = np.arange(45).reshape(9, 5)
    XB = np.arange(32).reshape(4, 8)
    with pytest.raises(ValueError):
        check_pairwise_arrays(XA, XB)
    XA = np.arange(45).reshape(9, 5)
    XB = np.arange(32).reshape(4, 8)
    with pytest.raises(ValueError):
        check_pairwise_arrays(XA, XB)


@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_check_sparse_arrays(csr_container):
    # [FUNCTION DOC]
    # Verifies check_pairwise_arrays keeps CSR inputs sparse and value-identical,
    #   including when the same matrix is passed twice.
    # Ensures that checks return valid sparse matrices.
    rng = np.random.RandomState(0)
    XA = rng.random_sample((5, 4))
    XA_sparse = csr_container(XA)
    XB = rng.random_sample((5, 4))
    XB_sparse = csr_container(XB)
    XA_checked, XB_checked = check_pairwise_arrays(XA_sparse, XB_sparse)
    # compare their difference because testing csr matrices for
    # equality with '==' does not work as expected.
    assert issparse(XA_checked)
    assert abs(XA_sparse - XA_checked).sum() == 0
    assert issparse(XB_checked)
    assert abs(XB_sparse - XB_checked).sum() == 0

    XA_checked, XA_2_checked = check_pairwise_arrays(XA_sparse, XA_sparse)
    assert issparse(XA_checked)
    assert abs(XA_sparse - XA_checked).sum() == 0
    assert issparse(XA_2_checked)
    assert abs(XA_2_checked - XA_checked).sum() == 0


def tuplify(X):
    # [FUNCTION DOC]
    # HELPER: Recursively converts an n-dimensional array into nested tuples of its
    #   elements.
    # Inputs: X - numpy array. Outputs: nested tuple mirroring the array shape.
    # Turns a numpy matrix (any n-dimensional array) into tuples.
    s = X.shape
    if len(s) > 1:
        # Tuplify each sub-array in the input.
        return tuple(tuplify(row) for row in X)
    else:
        # Single dimension input, just return tuple of contents.
        return tuple(r for r in X)


def test_check_tuple_input():
    # [FUNCTION DOC]
    # Verifies check_pairwise_arrays accepts nested tuple inputs and returns arrays
    #   equal to them.
    # Ensures that checks return valid tuples.
    rng = np.random.RandomState(0)
    XA = rng.random_sample((5, 4))
    XA_tuples = tuplify(XA)
    XB = rng.random_sample((5, 4))
    XB_tuples = tuplify(XB)
    XA_checked, XB_checked = check_pairwise_arrays(XA_tuples, XB_tuples)
    assert_array_equal(XA_tuples, XA_checked)
    assert_array_equal(XB_tuples, XB_checked)


def test_check_preserve_type():
    # [FUNCTION DOC]
    # Verifies check_pairwise_arrays keeps float32 when all inputs are float32 and
    #   upcasts both to float64 when either is float64.
    # Ensures that type float32 is preserved.
    XA = np.resize(np.arange(40), (5, 8)).astype(np.float32)
    XB = np.resize(np.arange(40), (5, 8)).astype(np.float32)

    XA_checked, XB_checked = check_pairwise_arrays(XA, None)
    assert XA_checked.dtype == np.float32

    # both float32
    XA_checked, XB_checked = check_pairwise_arrays(XA, XB)
    assert XA_checked.dtype == np.float32
    assert XB_checked.dtype == np.float32

    # mismatched A
    XA_checked, XB_checked = check_pairwise_arrays(XA.astype(float), XB)
    assert XA_checked.dtype == float
    assert XB_checked.dtype == float

    # mismatched B
    XA_checked, XB_checked = check_pairwise_arrays(XA, XB.astype(float))
    assert XA_checked.dtype == float
    assert XB_checked.dtype == float


@pytest.mark.parametrize("n_jobs", [1, 2])
@pytest.mark.parametrize("metric", ["seuclidean", "mahalanobis"])
@pytest.mark.parametrize(
    "dist_function", [pairwise_distances, pairwise_distances_chunked]
)
def test_pairwise_distances_data_derived_params(n_jobs, metric, dist_function):
    # [FUNCTION DOC]
    # Verifies pairwise_distances and pairwise_distances_chunked with
    #   seuclidean/mahalanobis (parameters derived from the data) give the same results
    #   as pdist, sequentially and in parallel, with small working_memory forcing
    #   several chunks.
    # check that pairwise_distances give the same result in sequential and
    # parallel, when metric has data-derived parameters.
    with config_context(working_memory=0.1):  # to have more than 1 chunk
        rng = np.random.RandomState(0)
        X = rng.random_sample((100, 10))

        expected_dist = squareform(pdist(X, metric=metric))
        dist = np.vstack(tuple(dist_function(X, metric=metric, n_jobs=n_jobs)))

        assert_allclose(dist, expected_dist)


@pytest.mark.parametrize("metric", ["seuclidean", "mahalanobis"])
def test_pairwise_distances_data_derived_params_error(metric):
    # [FUNCTION DOC]
    # Verifies passing Y with seuclidean/mahalanobis but without V/VI raises ValueError
    #   requiring that parameter.
    # check that pairwise_distances raises an error when Y is passed but
    # metric has data-derived params that are not provided by the user.
    rng = np.random.RandomState(0)
    X = rng.random_sample((100, 10))
    Y = rng.random_sample((100, 10))

    with pytest.raises(
        ValueError,
        match=rf"The '(V|VI)' parameter is required for the {metric} metric",
    ):
        pairwise_distances(X, Y, metric=metric)


@pytest.mark.parametrize(
    "metric",
    [
        "braycurtis",
        "canberra",
        "chebyshev",
        "correlation",
        "hamming",
        "mahalanobis",
        "minkowski",
        "seuclidean",
        "sqeuclidean",
        "cityblock",
        "cosine",
        "euclidean",
    ],
)
@pytest.mark.parametrize("y_is_x", [True, False], ids=["Y is X", "Y is not X"])
def test_numeric_pairwise_distances_datatypes(metric, global_dtype, y_is_x):
    # [FUNCTION DOC]
    # Verifies pairwise_distances matches scipy pdist/cdist for many scipy metrics in
    #   both dtypes, precomputing V or VI from stacked X and Y when Y is not X (#15730).
    # Check that pairwise distances gives the same result as pdist and cdist
    # regardless of input datatype when using any scipy metric for comparing
    # numeric vectors
    #
    # This test is necessary because pairwise_distances used to throw an
    # error when using metric='seuclidean' and the input data was not
    # of type np.float64 (#15730)

    rng = np.random.RandomState(0)

    X = rng.random_sample((5, 4)).astype(global_dtype, copy=False)

    params = {}
    if y_is_x:
        Y = X
        expected_dist = squareform(pdist(X, metric=metric))
    else:
        Y = rng.random_sample((5, 4)).astype(global_dtype, copy=False)
        expected_dist = cdist(X, Y, metric=metric)
        # precompute parameters for seuclidean & mahalanobis when x is not y
        if metric == "seuclidean":
            params = {"V": np.var(np.vstack([X, Y]), axis=0, ddof=1, dtype=np.float64)}
        elif metric == "mahalanobis":
            params = {"VI": np.linalg.inv(np.cov(np.vstack([X, Y]).T)).T}

    dist = pairwise_distances(X, Y, metric=metric, **params)

    assert_allclose(dist, expected_dist)


@pytest.mark.parametrize(
    "pairwise_distances_func",
    [pairwise_distances, pairwise_distances_argmin, pairwise_distances_argmin_min],
)
def test_nan_euclidean_support(pairwise_distances_func):
    # [FUNCTION DOC]
    # Verifies pairwise_distances, pairwise_distances_argmin and
    #   pairwise_distances_argmin_min with nan_euclidean produce no NaN in their output
    #   on data containing NaN.
    """Check that `nan_euclidean` is lenient with `nan` values."""

    X = [[0, 1], [1, np.nan], [2, 3], [3, 5]]
    output = pairwise_distances_func(X, X, metric="nan_euclidean")

    assert not np.isnan(output).any()


def test_nan_euclidean_constant_input_argmin():
    # [FUNCTION DOC]
    # Verifies argmin with nan_euclidean on all-NaN data equals the argmin on constant
    #   zero data.
    """Check that the behavior of constant input is the same in the case of
    full of nan vector and full of zero vector.
    """

    X_nan = [[np.nan, np.nan], [np.nan, np.nan], [np.nan, np.nan]]
    argmin_nan = pairwise_distances_argmin(X_nan, X_nan, metric="nan_euclidean")

    X_const = [[0, 0], [0, 0], [0, 0]]
    argmin_const = pairwise_distances_argmin(X_const, X_const, metric="nan_euclidean")

    assert_allclose(argmin_nan, argmin_const)


@pytest.mark.parametrize(
    "X,Y,expected_distance",
    [
        (
            ["a", "ab", "abc"],
            None,
            [[0.0, 1.0, 2.0], [1.0, 0.0, 1.0], [2.0, 1.0, 0.0]],
        ),
        (
            ["a", "ab", "abc"],
            ["a", "ab"],
            [[0.0, 1.0], [1.0, 0.0], [2.0, 1.0]],
        ),
    ],
)
def test_pairwise_dist_custom_metric_for_string(X, Y, expected_distance):
    # [FUNCTION DOC]
    # Verifies pairwise_distances accepts lists of strings with a custom callable
    #   metric, with and without Y.
    """Check pairwise_distances with lists of strings as input."""

    def dummy_string_similarity(x, y):
        # [FUNCTION DOC]
        # HELPER: Custom metric returning the absolute difference of the two strings'
        #   lengths.
        return np.abs(len(x) - len(y))

    actual_distance = pairwise_distances(X=X, Y=Y, metric=dummy_string_similarity)
    assert_allclose(actual_distance, expected_distance)


def test_pairwise_dist_custom_metric_for_bool():
    # [FUNCTION DOC]
    # Verifies boolean input is passed unchanged (not converted to float) to a custom
    #   metric relying on bitwise operations.
    """Check that pairwise_distances does not convert boolean input to float
    when using a custom metric.
    """

    def dummy_bool_dist(v1, v2):
        # [FUNCTION DOC]
        # HELPER: Custom boolean distance equal to one minus the Jaccard similarity of
        #   two boolean vectors.
        # dummy distance func using `&` and thus relying on the input data being boolean
        return 1 - (v1 & v2).sum() / (v1 | v2).sum()

    X = np.array([[1, 0, 0, 0], [1, 0, 1, 0], [1, 1, 1, 1]], dtype=bool)

    expected_distance = np.array(
        [
            [0.0, 0.5, 0.75],
            [0.5, 0.0, 0.5],
            [0.75, 0.5, 0.0],
        ]
    )

    actual_distance = pairwise_distances(X=X, metric=dummy_bool_dist)
    assert_allclose(actual_distance, expected_distance)


# TODO: remove mark once loky bug is fixed:
# https://github.com/joblib/loky/issues/458
@pytest.mark.thread_unsafe
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_sparse_manhattan_readonly_dataset(csr_container):
    # [FUNCTION DOC]
    # Non-regression test for #7981: verifies manhattan_distances on CSR matrices
    #   memory-mapped read-only by joblib (n_jobs=2, max_nbytes=0) runs without error.
    # Non-regression test for: https://github.com/scikit-learn/scikit-learn/issues/7981
    matrices1 = [csr_container(np.ones((5, 5)))]
    matrices2 = [csr_container(np.ones((5, 5)))]
    # Joblib memory maps datasets which makes them read-only.
    # The following call was reporting as failing in #7981, but this must pass.
    Parallel(n_jobs=2, max_nbytes=0)(
        delayed(manhattan_distances)(m1, m2) for m1, m2 in zip(matrices1, matrices2)
    )
