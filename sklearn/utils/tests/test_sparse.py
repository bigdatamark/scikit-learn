# [MODULE DOC]
# Tests for the sparse_interface configuration option: sparse outputs from utilities,
# transformers, functions and estimator properties follow the configured scipy sparse
# API (sparray vs spmatrix).
import numpy as np
import pytest
from scipy.sparse import csc_array, csc_matrix, csr_array, csr_matrix

import sklearn


@pytest.mark.parametrize(
    ["sparse_interface", "x", "result_type"],
    [
        ("sparray", csr_array([[1, 2, 3]]), csr_array),
        ("sparray", csr_matrix([[1, 2, 3]]), csr_array),
        ("sparray", csc_array([[1, 2, 3]]), csc_array),
        ("sparray", csc_matrix([[1, 2, 3]]), csc_array),
        ("spmatrix", csr_array([[1, 2, 3]]), csr_matrix),
        ("spmatrix", csr_matrix([[1, 2, 3]]), csr_matrix),
        ("spmatrix", csc_array([[1, 2, 3]]), csc_matrix),
        ("spmatrix", csc_matrix([[1, 2, 3]]), csc_matrix),
    ],
)
def test_align_api_if_sparse(sparse_interface, x, result_type):
    # [FUNCTION DOC]
    # Verifies that sklearn.utils._align_api_if_sparse converts CSR/CSC arrays and
    # matrices to the array or matrix class (preserving format) selected by the
    # sparse_interface config.
    # Inputs: sparse_interface - 'sparray' or 'spmatrix'; x - sparse input; result_type
    # - expected class.
    with sklearn.config_context(sparse_interface=sparse_interface):
        result = sklearn.utils._align_api_if_sparse(x)
        assert isinstance(result, result_type)


@pytest.mark.parametrize(
    ["sparse_interface", "x", "result_type"],
    [
        ("sparray", np.array([[1, 2, 3]]), np.ndarray),
        ("spmatrix", np.array([[1, 2, 3]]), np.ndarray),
    ],
)
def test_ndarray_align_api_if_sparse(sparse_interface, x, result_type):
    # [FUNCTION DOC]
    # Verifies that _align_api_if_sparse leaves dense ndarrays as ndarrays under either
    # sparse_interface setting.
    # Inputs: sparse_interface, x, result_type - parametrized.
    with sklearn.config_context(sparse_interface=sparse_interface):
        result = sklearn.utils._align_api_if_sparse(x)
        assert isinstance(result, result_type)


@pytest.mark.parametrize(
    ["sparse_interface", "result_type"],
    [("sparray", csr_array), ("spmatrix", csr_matrix)],
)
def test_transform_returns_sparse(sparse_interface, result_type):
    # [FUNCTION DOC]
    # Verifies that CountVectorizer.fit_transform returns csr_array or csr_matrix
    # according to sparse_interface.
    # Inputs: sparse_interface, result_type - parametrized.
    corpus = [
        "This is the first document.",
        "This document is the second document.",
        "And this is the third one.",
        "Is this the first document?",
    ]
    with sklearn.config_context(sparse_interface=sparse_interface):
        vectorizer = sklearn.feature_extraction.text.CountVectorizer()
        X = vectorizer.fit_transform(corpus)
        assert isinstance(X, result_type)


@pytest.mark.parametrize(
    ["sparse_interface", "result_type"],
    [("sparray", csr_array), ("spmatrix", csr_matrix)],
)
def test_function_returns_sparse(sparse_interface, result_type):
    # [FUNCTION DOC]
    # Verifies that the private barycenter_kneighbors_graph function returns the CSR
    # class selected by sparse_interface.
    # Inputs: sparse_interface, result_type - parametrized.
    with sklearn.config_context(sparse_interface=sparse_interface):
        X, y = sklearn.datasets.make_regression(n_features=2, random_state=0)
        X = sklearn.manifold._locally_linear.barycenter_kneighbors_graph(X, 1)
        assert isinstance(X, result_type)


@pytest.mark.parametrize(
    ["sparse_interface", "result_type"],
    [("sparray", csr_array), ("spmatrix", csr_matrix)],
)
def test_estimator_property_sparse(sparse_interface, result_type):
    # [FUNCTION DOC]
    # Verifies that the fitted ElasticNet.sparse_coef_ property returns the CSR class
    # selected by sparse_interface.
    # Inputs: sparse_interface, result_type - parametrized.
    with sklearn.config_context(sparse_interface=sparse_interface):
        X, y = sklearn.datasets.make_regression(n_features=2, random_state=0)
        regr = sklearn.linear_model.ElasticNet(random_state=0)
        regr.fit(X, y)
        # check spec_coeff property
        assert isinstance(regr.sparse_coef_, result_type)
