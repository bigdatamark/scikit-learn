# [MODULE DOC]
# Tests for spectral biclustering estimators (SpectralCoclustering,
# SpectralBiclustering), BiclusterMixin helpers (get_submatrix, get_shape,
# get_indices) and the private normalization helpers in sklearn.cluster._bicluster.
"""Testing for Spectral Biclustering methods"""

import numpy as np
import pytest
from scipy.sparse import issparse

from sklearn.base import BaseEstimator, BiclusterMixin, clone
from sklearn.cluster import SpectralBiclustering, SpectralCoclustering
from sklearn.cluster._bicluster import (
    _bistochastic_normalize,
    _log_normalize,
    _scale_normalize,
)
from sklearn.datasets import make_biclusters, make_checkerboard
from sklearn.metrics import consensus_score, v_measure_score
from sklearn.model_selection import ParameterGrid
from sklearn.utils._testing import (
    assert_almost_equal,
    assert_array_almost_equal,
    assert_array_equal,
)
from sklearn.utils.fixes import CSR_CONTAINERS


class MockBiclustering(BiclusterMixin, BaseEstimator):
    # [CLASS DOC]
    # HELPER: Minimal bicluster estimator with fixed indices, used to test
    # BiclusterMixin.get_submatrix without fitting anything.
    # Mock object for testing get_submatrix.
    def __init__(self):
        # [FUNCTION DOC]
        # HELPER: No-op constructor (no parameters, no attributes set).
        pass

    def get_indices(self, i):
        # [FUNCTION DOC]
        # HELPER: Ignores the bicluster index i and always returns row indices [0, 1, 4] and
        # column indices [2, 3] as integer arrays.
        # Overridden to reproduce old get_submatrix test.
        return (
            np.where([True, True, False, False, True])[0],
            np.where([False, False, True, True])[0],
        )


@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_get_submatrix(csr_container):
    # [FUNCTION DOC]
    # Verifies get_submatrix on a 5x4 arange matrix given as ndarray, CSR matrix and nested
    # list returns rows [0, 1, 4] x columns [2, 3], and that writing into the returned
    # submatrix does not modify the original input (it is a copy).
    data = np.arange(20).reshape(5, 4)
    model = MockBiclustering()

    for X in (data, csr_container(data), data.tolist()):
        submatrix = model.get_submatrix(0, X)
        if issparse(submatrix):
            submatrix = submatrix.toarray()
        assert_array_equal(submatrix, [[2, 3], [6, 7], [18, 19]])
        submatrix[:] = -1
        if issparse(X):
            X = X.toarray()
        assert np.all(X != -1)


def _test_shape_indices(model):
    # [FUNCTION DOC]
    # HELPER: For each bicluster index in range(model.n_clusters), asserts get_shape
    # returns (number of row indices, number of column indices) consistent with
    # get_indices. Input: a fitted bicluster model. Raises AssertionError on mismatch.
    # KNOWN ISSUE: for SpectralBiclustering n_clusters may be a tuple and the number of
    # biclusters is the product of its entries; here an int 3 is used, so only the first 3
    # of the 9 biclusters are checked.
    # Test get_shape and get_indices on fitted model.
    for i in range(model.n_clusters):
        m, n = model.get_shape(i)
        i_ind, j_ind = model.get_indices(i)
        assert len(i_ind) == m
        assert len(j_ind) == n


@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_spectral_coclustering(global_random_seed, csr_container):
    # [FUNCTION DOC]
    # Verifies SpectralCoclustering recovers 3 planted biclusters perfectly (consensus score
    # 1) on a thresholded nonnegative make_biclusters matrix, dense and CSR, over a grid of
    # svd_method, n_svd_vecs and mini_batch settings; rows_/columns_ assign each row and
    # column to exactly one bicluster, and shape/indices helpers agree.
    # Test Dhillon's Spectral CoClustering on a simple problem.
    param_grid = {
        "svd_method": ["randomized", "arpack"],
        "n_svd_vecs": [None, 20],
        "mini_batch": [False, True],
        "init": ["k-means++"],
        "n_init": [10],
    }
    S, rows, cols = make_biclusters(
        (30, 30), 3, noise=0.1, random_state=global_random_seed
    )
    S -= S.min()  # needs to be nonnegative before making it sparse
    S = np.where(S < 1, 0, S)  # threshold some values
    for mat in (S, csr_container(S)):
        for kwargs in ParameterGrid(param_grid):
            model = SpectralCoclustering(
                n_clusters=3, random_state=global_random_seed, **kwargs
            )
            model.fit(mat)

            assert model.rows_.shape == (3, 30)
            assert_array_equal(model.rows_.sum(axis=0), np.ones(30))
            assert_array_equal(model.columns_.sum(axis=0), np.ones(30))
            assert consensus_score(model.biclusters_, (rows, cols)) == 1

            _test_shape_indices(model)


@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_spectral_biclustering(global_random_seed, csr_container):
    # [FUNCTION DOC]
    # Verifies SpectralBiclustering on a 3x3 checkerboard (9 biclusters) for each single
    # non-default parameter value: rows_ and columns_ have shape (9, 30), each row/column
    # belongs to 3 biclusters, consensus score is 1; for CSR input with method="log" a
    # ValueError is expected instead.
    # Test Kluger methods on a checkerboard dataset.
    S, rows, cols = make_checkerboard(
        (30, 30), 3, noise=0.5, random_state=global_random_seed
    )

    non_default_params = {
        "method": ["scale", "log"],
        "svd_method": ["arpack"],
        "n_svd_vecs": [20],
        "mini_batch": [True],
    }

    for mat in (S, csr_container(S)):
        for param_name, param_values in non_default_params.items():
            for param_value in param_values:
                model = SpectralBiclustering(
                    n_clusters=3,
                    n_init=3,
                    init="k-means++",
                    random_state=global_random_seed,
                )
                model.set_params(**dict([(param_name, param_value)]))

                if issparse(mat) and model.get_params().get("method") == "log":
                    # cannot take log of sparse matrix
                    with pytest.raises(ValueError):
                        model.fit(mat)
                    continue
                else:
                    model.fit(mat)

                assert model.rows_.shape == (9, 30)
                assert model.columns_.shape == (9, 30)
                assert_array_equal(model.rows_.sum(axis=0), np.repeat(3, 30))
                assert_array_equal(model.columns_.sum(axis=0), np.repeat(3, 30))
                assert consensus_score(model.biclusters_, (rows, cols)) == 1

                _test_shape_indices(model)


def _do_scale_test(scaled):
    # [FUNCTION DOC]
    # HELPER: Asserts that the row sums of a 100x100 (dense or sparse) matrix are all
    # roughly equal to their mean and likewise for column sums (1 decimal tolerance).
    # Raises AssertionError otherwise.
    """Check that rows sum to one constant, and columns to another."""
    row_sum = scaled.sum(axis=1)
    col_sum = scaled.sum(axis=0)
    if issparse(scaled):
        row_sum = np.asarray(row_sum).squeeze()
        col_sum = np.asarray(col_sum).squeeze()
    assert_array_almost_equal(row_sum, np.tile(row_sum.mean(), 100), decimal=1)
    assert_array_almost_equal(col_sum, np.tile(col_sum.mean(), 100), decimal=1)


def _do_bistochastic_test(scaled):
    # [FUNCTION DOC]
    # HELPER: Runs _do_scale_test and additionally asserts that mean row sum and mean
    # column sum agree to 1 decimal.
    """Check that rows and columns sum to the same constant."""
    _do_scale_test(scaled)
    assert_almost_equal(scaled.sum(axis=0).mean(), scaled.sum(axis=1).mean(), decimal=1)


@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_scale_normalize(global_random_seed, csr_container):
    # [FUNCTION DOC]
    # Verifies _scale_normalize on a random 100x100 matrix (dense and CSR) yields constant
    # row sums and constant column sums, and preserves sparsity.
    generator = np.random.RandomState(global_random_seed)
    X = generator.rand(100, 100)
    for mat in (X, csr_container(X)):
        scaled, _, _ = _scale_normalize(mat)
        _do_scale_test(scaled)
        if issparse(mat):
            assert issparse(scaled)


@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_bistochastic_normalize(global_random_seed, csr_container):
    # [FUNCTION DOC]
    # Verifies _bistochastic_normalize yields rows and columns summing to the same constant
    # for dense and CSR input, and preserves sparsity.
    generator = np.random.RandomState(global_random_seed)
    X = generator.rand(100, 100)
    for mat in (X, csr_container(X)):
        scaled = _bistochastic_normalize(mat)
        _do_bistochastic_test(scaled)
        if issparse(mat):
            assert issparse(scaled)


def test_log_normalize(global_random_seed):
    # [FUNCTION DOC]
    # Verifies that the log-normalized random matrix plus 1 is bistochastic.
    # adding any constant to a log-scaled matrix should make it
    # bistochastic
    generator = np.random.RandomState(global_random_seed)
    mat = generator.rand(100, 100)
    scaled = _log_normalize(mat) + 1
    _do_bistochastic_test(scaled)


def test_fit_best_piecewise(global_random_seed):
    # [FUNCTION DOC]
    # Verifies _fit_best_piecewise with n_best=2 selects the two vectors that are best
    # approximated by piecewise-constant vectors (the first two rows of the input).
    model = SpectralBiclustering(random_state=global_random_seed)
    vectors = np.array([[0, 0, 0, 1, 1, 1], [2, 2, 2, 3, 3, 3], [0, 1, 2, 3, 4, 5]])
    best = model._fit_best_piecewise(vectors, n_best=2, n_clusters=2)
    assert_array_equal(best, vectors[:2])


@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_project_and_cluster(global_random_seed, csr_container):
    # [FUNCTION DOC]
    # Verifies _project_and_cluster separates the two distinct row patterns of a small
    # matrix (dense and CSR) into 2 clusters with perfect v-measure.
    model = SpectralBiclustering(random_state=global_random_seed)
    data = np.array([[1, 1, 1], [1, 1, 1], [3, 6, 3], [3, 6, 3]])
    vectors = np.array([[1, 0], [0, 1], [0, 0]])
    for mat in (data, csr_container(data)):
        labels = model._project_and_cluster(mat, vectors, n_clusters=2)
        assert_almost_equal(v_measure_score(labels, [0, 0, 1, 1]), 1.0)


def test_perfect_checkerboard(global_random_seed):
    # [FUNCTION DOC]
    # Verifies SpectralBiclustering (arpack) perfectly recovers noise-free 3x3
    # checkerboards of shapes 30x30, 40x30 and 30x40 (consensus score 1).
    # XXX Previously failed on build bot (not reproducible)
    model = SpectralBiclustering(
        3, svd_method="arpack", random_state=global_random_seed
    )

    S, rows, cols = make_checkerboard(
        (30, 30), 3, noise=0, random_state=global_random_seed
    )
    model.fit(S)
    assert consensus_score(model.biclusters_, (rows, cols)) == 1

    S, rows, cols = make_checkerboard(
        (40, 30), 3, noise=0, random_state=global_random_seed
    )
    model.fit(S)
    assert consensus_score(model.biclusters_, (rows, cols)) == 1

    S, rows, cols = make_checkerboard(
        (30, 40), 3, noise=0, random_state=global_random_seed
    )
    model.fit(S)
    assert consensus_score(model.biclusters_, (rows, cols)) == 1


@pytest.mark.parametrize(
    "params, type_err, err_msg",
    [
        (
            {"n_clusters": 6},
            ValueError,
            "n_clusters should be <= n_samples=5",
        ),
        (
            {"n_clusters": (3, 3, 3)},
            ValueError,
            "Incorrect parameter n_clusters",
        ),
        (
            {"n_clusters": (3, 6)},
            ValueError,
            "Incorrect parameter n_clusters",
        ),
        (
            {"n_components": 3, "n_best": 4},
            ValueError,
            "n_best=4 must be <= n_components=3",
        ),
    ],
)
def test_spectralbiclustering_parameter_validation(params, type_err, err_msg):
    # [FUNCTION DOC]
    # Verifies fit on a 5x5 matrix raises the expected error for invalid n_clusters (too
    # large, wrong tuple length, tuple entry too large) and n_best > n_components.
    # Inputs: params - constructor kwargs; type_err - exception class; err_msg - regex.
    """Check parameters validation in `SpectralBiClustering`"""
    data = np.arange(25).reshape((5, 5))
    model = SpectralBiclustering(**params)
    with pytest.raises(type_err, match=err_msg):
        model.fit(data)


@pytest.mark.parametrize("est", (SpectralBiclustering(), SpectralCoclustering()))
def test_n_features_in_(est):
    # [FUNCTION DOC]
    # Verifies n_features_in_ is absent before fit and equals 3 after fitting a 3x3 matrix,
    # for a clone of each bicluster estimator. Inputs: est - estimator instance.
    X, _, _ = make_biclusters((3, 3), 3, random_state=0)

    est = clone(est)
    assert not hasattr(est, "n_features_in_")
    est.fit(X)
    assert est.n_features_in_ == 3
