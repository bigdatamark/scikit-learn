# [MODULE DOC]
# Tests for mutual_reachability_graph in sklearn.cluster._hdbscan._reachability:
# input-format validation, in-place operation, dense/sparse equivalence and dtype
# preservation. Inputs are symmetric positive semi-definite matrices built as the
# Gram matrix of a seeded random square matrix.
import numpy as np
import pytest

from sklearn.cluster._hdbscan._reachability import mutual_reachability_graph
from sklearn.utils._testing import (
    _convert_container,
    assert_allclose,
)


def test_mutual_reachability_graph_error_sparse_format():
    # [FUNCTION DOC]
    # Verifies that passing a CSC sparse distance matrix raises ValueError with the
    # message "Only sparse CSR matrices are supported".
    """Check that we raise an error if the sparse format is not CSR."""
    rng = np.random.RandomState(0)
    X = rng.randn(10, 10)
    X = X.T @ X
    np.fill_diagonal(X, 0.0)
    X = _convert_container(X, "sparse_csc")

    err_msg = "Only sparse CSR matrices are supported"
    with pytest.raises(ValueError, match=err_msg):
        mutual_reachability_graph(X)


@pytest.mark.parametrize("array_type", ["array", "sparse_csr"])
def test_mutual_reachability_graph_inplace(array_type):
    # [FUNCTION DOC]
    # Verifies the function returns the very same object it was given (identity check via
    # id) for both a dense array and a CSR matrix, i.e. the computation is in place.
    # Inputs: array_type - "array" or "sparse_csr" (parametrized).
    """Check that the operation is happening inplace."""
    rng = np.random.RandomState(0)
    X = rng.randn(10, 10)
    X = X.T @ X
    np.fill_diagonal(X, 0.0)
    X = _convert_container(X, array_type)

    mr_graph = mutual_reachability_graph(X)

    assert id(mr_graph) == id(X)


def test_mutual_reachability_graph_equivalence_dense_sparse():
    # [FUNCTION DOC]
    # Verifies that with min_samples=3 the dense and CSR implementations produce the same
    # graph (sparse result densified and compared with assert_allclose).
    # KNOWN ISSUE: unlike the other tests the diagonal is not zeroed here, so the input is
    # not a proper distance matrix; the test still checks equivalence only.
    """Check that we get the same results for dense and sparse implementation."""
    rng = np.random.RandomState(0)
    X = rng.randn(5, 5)
    X_dense = X.T @ X
    X_sparse = _convert_container(X_dense, "sparse_csr")

    mr_graph_dense = mutual_reachability_graph(X_dense, min_samples=3)
    mr_graph_sparse = mutual_reachability_graph(X_sparse, min_samples=3)

    assert_allclose(mr_graph_dense, mr_graph_sparse.toarray())


@pytest.mark.parametrize("array_type", ["array", "sparse_csr"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_mutual_reachability_graph_preserves_dtype(array_type, dtype):
    # [FUNCTION DOC]
    # Verifies the output dtype equals the input dtype (float32 or float64) for dense and
    # CSR inputs, relying on the fused-type implementation.
    # Inputs: array_type - "array" or "sparse_csr"; dtype - np.float32 or np.float64.
    """Check that the computation preserve dtype thanks to fused types."""
    rng = np.random.RandomState(0)
    X = rng.randn(10, 10)
    X = (X.T @ X).astype(dtype)
    np.fill_diagonal(X, 0.0)
    X = _convert_container(X, array_type)

    assert X.dtype == dtype
    mr_graph = mutual_reachability_graph(X)
    assert mr_graph.dtype == dtype
