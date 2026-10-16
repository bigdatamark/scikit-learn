# [MODULE DOC]
# Tests for KNeighborsTransformer and RadiusNeighborsTransformer from _graph.py: output
# shapes, CSR format, per-row sorting, explicit storage of self-distances, and output
# feature names.
import numpy as np
import pytest

from sklearn.metrics import euclidean_distances
from sklearn.neighbors import KNeighborsTransformer, RadiusNeighborsTransformer
from sklearn.neighbors._base import _is_sorted_by_data
from sklearn.utils._testing import assert_array_equal


def test_transformer_result():
    # [FUNCTION DOC]
    # For both modes, checks that KNeighborsTransformer outputs CSR graphs of the right
    # shape with n_neighbors (plus one in distance mode) entries per row, sorted by
    # data,
    # for fit_transform and transform on new data; and that RadiusNeighborsTransformer
    # outputs correctly shaped sorted CSR graphs whose entry count differs from the k-NN
    # count.
    # Test the number of neighbors returned
    n_neighbors = 5
    n_samples_fit = 20
    n_queries = 18
    n_features = 10

    rng = np.random.RandomState(42)
    X = rng.randn(n_samples_fit, n_features)
    X2 = rng.randn(n_queries, n_features)
    radius = np.percentile(euclidean_distances(X), 10)

    # with n_neighbors
    for mode in ["distance", "connectivity"]:
        add_one = mode == "distance"
        nnt = KNeighborsTransformer(n_neighbors=n_neighbors, mode=mode)
        Xt = nnt.fit_transform(X)
        assert Xt.shape == (n_samples_fit, n_samples_fit)
        assert Xt.data.shape == (n_samples_fit * (n_neighbors + add_one),)
        assert Xt.format == "csr"
        assert _is_sorted_by_data(Xt)

        X2t = nnt.transform(X2)
        assert X2t.shape == (n_queries, n_samples_fit)
        assert X2t.data.shape == (n_queries * (n_neighbors + add_one),)
        assert X2t.format == "csr"
        assert _is_sorted_by_data(X2t)

    # with radius
    for mode in ["distance", "connectivity"]:
        add_one = mode == "distance"
        nnt = RadiusNeighborsTransformer(radius=radius, mode=mode)
        Xt = nnt.fit_transform(X)
        assert Xt.shape == (n_samples_fit, n_samples_fit)
        assert not Xt.data.shape == (n_samples_fit * (n_neighbors + add_one),)
        assert Xt.format == "csr"
        assert _is_sorted_by_data(Xt)

        X2t = nnt.transform(X2)
        assert X2t.shape == (n_queries, n_samples_fit)
        assert not X2t.data.shape == (n_queries * (n_neighbors + add_one),)
        assert X2t.format == "csr"
        assert _is_sorted_by_data(X2t)


def _has_explicit_diagonal(X):
    # [FUNCTION DOC]
    # HELPER: Returns True when the number of explicitly stored diagonal entries equals
    # the
    # number of rows (converts to COO and counts entries with row equal to column).
    # Inputs: X - sparse matrix. Outputs: bool.
    """Return True if the diagonal is explicitly stored"""
    X = X.tocoo()
    explicit = X.row[X.row == X.col]
    return len(explicit) == X.shape[0]


def test_explicit_diagonal():
    # [FUNCTION DOC]
    # Checks that in distance mode the training graph (via fit_transform and transform)
    # stores each sample's zero self-distance explicitly as the first entry of its row,
    # while transforming new data does not produce a full explicit diagonal.
    # Test that the diagonal is explicitly stored in the sparse graph
    n_neighbors = 5
    n_samples_fit, n_samples_transform, n_features = 20, 18, 10
    rng = np.random.RandomState(42)
    X = rng.randn(n_samples_fit, n_features)
    X2 = rng.randn(n_samples_transform, n_features)

    nnt = KNeighborsTransformer(n_neighbors=n_neighbors)
    Xt = nnt.fit_transform(X)
    assert _has_explicit_diagonal(Xt)
    assert np.all(Xt.data.reshape(n_samples_fit, n_neighbors + 1)[:, 0] == 0)

    Xt = nnt.transform(X)
    assert _has_explicit_diagonal(Xt)
    assert np.all(Xt.data.reshape(n_samples_fit, n_neighbors + 1)[:, 0] == 0)

    # Using transform on new data should not always have zero diagonal
    X2t = nnt.transform(X2)
    assert not _has_explicit_diagonal(X2t)


@pytest.mark.parametrize("Klass", (KNeighborsTransformer, RadiusNeighborsTransformer))
def test_graph_feature_names_out(Klass):
    # [FUNCTION DOC]
    # Checks get_feature_names_out returns the lowercase class name followed by an index
    # for each of the n_samples_fit_ output columns.
    """Check `get_feature_names_out` for transformers defined in `_graph.py`."""

    n_samples_fit = 20
    n_features = 10
    rng = np.random.RandomState(42)
    X = rng.randn(n_samples_fit, n_features)

    est = Klass().fit(X)
    names_out = est.get_feature_names_out()

    class_name_lower = Klass.__name__.lower()
    expected_names_out = np.array(
        [f"{class_name_lower}{i}" for i in range(est.n_samples_fit_)],
        dtype=object,
    )
    assert_array_equal(names_out, expected_names_out)
