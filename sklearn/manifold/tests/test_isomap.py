# [MODULE DOC]
# Tests for sklearn.manifold.Isomap: distance preservation on a grid, agreement of
#   reconstruction_error with a manually computed kernel error, transform of noisy data,
#   pipelines (including precomputed KNeighborsTransformer input), metric handling,
#   sparse input, precomputed radius graphs, dtype preservation, parameter validation,
#   behavior with disconnected neighbor graphs and feature names. Module-level
#   eigen_solvers and path_methods list the solver/path options used in
#   parametrizations.
import math
from itertools import product

import numpy as np
import pytest

from sklearn import clone, datasets, manifold, neighbors, pipeline, preprocessing
from sklearn.datasets import make_blobs
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.utils._testing import (
    assert_allclose,
    assert_allclose_dense_sparse,
    assert_array_equal,
)
from sklearn.utils.fixes import CSR_CONTAINERS, _sparse_random_array

eigen_solvers = ["auto", "dense", "arpack"]
path_methods = ["auto", "FW", "D"]


def create_sample_data(dtype, n_pts=25, add_noise=False):
    # [FUNCTION DOC]
    # HELPER: Builds a regular 2-D grid of points with integer coordinates.
    # Inputs: dtype - numpy dtype for the output; n_pts - number of points (default 25),
    #   the grid side is the integer square root, so n_pts should be a perfect square;
    #   add_noise - if True, append a third column of Gaussian noise (scale 0.1, seed
    #   0).
    # Outputs: array of shape (side*side, 2), or (n_pts, 3) with noise.
    # Exceptions: concatenation fails if n_pts is not a perfect square and add_noise is
    #   True.
    # Side effects: none (uses a local RandomState).
    # grid of equidistant points in 2D, n_components = n_dim
    n_per_side = int(math.sqrt(n_pts))
    X = np.array(list(product(range(n_per_side), repeat=2))).astype(dtype, copy=False)
    if add_noise:
        # add noise in a third dimension
        rng = np.random.RandomState(0)
        noise = 0.1 * rng.randn(n_pts, 1).astype(dtype, copy=False)
        X = np.concatenate((X, noise), 1)
    return X


@pytest.mark.parametrize("n_neighbors, radius", [(24, None), (None, np.inf)])
@pytest.mark.parametrize("eigen_solver", eigen_solvers)
@pytest.mark.parametrize("path_method", path_methods)
def test_isomap_simple_grid(
    global_dtype, n_neighbors, radius, eigen_solver, path_method
):
    # [FUNCTION DOC]
    # Parametrized over neighbor mode (24 neighbors or infinite radius), eigen solver
    #   and path method. With all points as neighbors on a flat 5x5 grid, the geodesic
    #   distances equal Euclidean ones, so the neighbor distance graph of the 2-D
    #   embedding must equal the graph of the input (exactly for float64, atol 1e-5 for
    #   float32).
    # Isomap should preserve distances when all neighbors are used
    n_pts = 25
    X = create_sample_data(global_dtype, n_pts=n_pts, add_noise=False)

    # distances from each point to all others
    if n_neighbors is not None:
        G = neighbors.kneighbors_graph(X, n_neighbors, mode="distance")
    else:
        G = neighbors.radius_neighbors_graph(X, radius, mode="distance")

    clf = manifold.Isomap(
        n_neighbors=n_neighbors,
        radius=radius,
        n_components=2,
        eigen_solver=eigen_solver,
        path_method=path_method,
    )
    clf.fit(X)

    if n_neighbors is not None:
        G_iso = neighbors.kneighbors_graph(clf.embedding_, n_neighbors, mode="distance")
    else:
        G_iso = neighbors.radius_neighbors_graph(
            clf.embedding_, radius, mode="distance"
        )
    atol = 1e-5 if global_dtype == np.float32 else 0
    assert_allclose_dense_sparse(G, G_iso, atol=atol)


@pytest.mark.parametrize("n_neighbors, radius", [(24, None), (None, np.inf)])
@pytest.mark.parametrize("eigen_solver", eigen_solvers)
@pytest.mark.parametrize("path_method", path_methods)
def test_isomap_reconstruction_error(
    global_dtype, n_neighbors, radius, eigen_solver, path_method
):
    # [FUNCTION DOC]
    # Same parametrization as test_isomap_simple_grid but on the grid with a noisy third
    #   dimension. Computes the centered kernel of minus half the squared neighbor-graph
    #   distances for both the input and the embedding, and checks that the Frobenius
    #   norm of their difference divided by n_pts equals Isomap.reconstruction_error().
    #   Skipped for float32 because of numerical instability.
    # KNOWN ISSUE: since float32 is skipped, the float32 branch of the atol selection is
    #   dead code.
    if global_dtype is np.float32:
        pytest.skip(
            "Skipping test due to numerical instabilities on float32 data"
            "from KernelCenterer used in the reconstruction_error method"
        )

    # Same setup as in test_isomap_simple_grid, with an added dimension
    n_pts = 25
    X = create_sample_data(global_dtype, n_pts=n_pts, add_noise=True)

    # compute input kernel
    if n_neighbors is not None:
        G = neighbors.kneighbors_graph(X, n_neighbors, mode="distance").toarray()
    else:
        G = neighbors.radius_neighbors_graph(X, radius, mode="distance").toarray()
    centerer = preprocessing.KernelCenterer()
    K = centerer.fit_transform(-0.5 * G**2)

    clf = manifold.Isomap(
        n_neighbors=n_neighbors,
        radius=radius,
        n_components=2,
        eigen_solver=eigen_solver,
        path_method=path_method,
    )
    clf.fit(X)

    # compute output kernel
    if n_neighbors is not None:
        G_iso = neighbors.kneighbors_graph(clf.embedding_, n_neighbors, mode="distance")
    else:
        G_iso = neighbors.radius_neighbors_graph(
            clf.embedding_, radius, mode="distance"
        )
    G_iso = G_iso.toarray()
    K_iso = centerer.fit_transform(-0.5 * G_iso**2)

    # make sure error agrees
    reconstruction_error = np.linalg.norm(K - K_iso) / n_pts
    atol = 1e-5 if global_dtype == np.float32 else 0
    assert_allclose(reconstruction_error, clf.reconstruction_error(), atol=atol)


@pytest.mark.parametrize("n_neighbors, radius", [(2, None), (None, 0.5)])
def test_transform(global_dtype, n_neighbors, radius):
    # [FUNCTION DOC]
    # Fits Isomap with 10 components on a 200-sample S-curve (2 neighbors or radius
    #   0.5), then transforms a copy perturbed with Gaussian noise of scale 0.01, and
    #   checks that the RMS difference between the two embeddings is below twice the
    #   noise scale.
    n_samples = 200
    n_components = 10
    noise_scale = 0.01

    # Create S-curve dataset
    X, y = datasets.make_s_curve(n_samples, random_state=0)

    X = X.astype(global_dtype, copy=False)

    # Compute isomap embedding
    iso = manifold.Isomap(
        n_components=n_components, n_neighbors=n_neighbors, radius=radius
    )
    X_iso = iso.fit_transform(X)

    # Re-embed a noisy version of the points
    rng = np.random.RandomState(0)
    noise = noise_scale * rng.randn(*X.shape)
    X_iso2 = iso.transform(X + noise)

    # Make sure the rms error on re-embedding is comparable to noise_scale
    assert np.sqrt(np.mean((X_iso - X_iso2) ** 2)) < 2 * noise_scale


@pytest.mark.parametrize("n_neighbors, radius", [(2, None), (None, 10.0)])
def test_pipeline(n_neighbors, radius, global_dtype):
    # [FUNCTION DOC]
    # Smoke test: Isomap (k-neighbors or radius mode) followed by KNeighborsClassifier
    #   in a Pipeline fits make_blobs data and scores above 0.9 on the training set.
    # check that Isomap works fine as a transformer in a Pipeline
    # only checks that no error is raised.
    # TODO check that it actually does something useful
    X, y = datasets.make_blobs(random_state=0)
    X = X.astype(global_dtype, copy=False)
    clf = pipeline.Pipeline(
        [
            ("isomap", manifold.Isomap(n_neighbors=n_neighbors, radius=radius)),
            ("clf", neighbors.KNeighborsClassifier()),
        ]
    )
    clf.fit(X, y)
    assert 0.9 < clf.score(X, y)


def test_pipeline_with_nearest_neighbors_transformer(global_dtype):
    # [FUNCTION DOC]
    # Verifies that a pipeline KNeighborsTransformer(mode="distance") followed by Isomap
    #   with metric="precomputed" gives the same fit_transform and transform results (on
    #   a second blobs dataset) as a single Isomap computing neighbors itself.
    # Test chaining NearestNeighborsTransformer and Isomap with
    # neighbors_algorithm='precomputed'
    algorithm = "auto"
    n_neighbors = 10

    X, _ = datasets.make_blobs(random_state=0)
    X2, _ = datasets.make_blobs(random_state=1)

    X = X.astype(global_dtype, copy=False)
    X2 = X2.astype(global_dtype, copy=False)

    # compare the chained version and the compact version
    est_chain = pipeline.make_pipeline(
        neighbors.KNeighborsTransformer(
            n_neighbors=n_neighbors, algorithm=algorithm, mode="distance"
        ),
        manifold.Isomap(n_neighbors=n_neighbors, metric="precomputed"),
    )
    est_compact = manifold.Isomap(
        n_neighbors=n_neighbors, neighbors_algorithm=algorithm
    )

    Xt_chain = est_chain.fit_transform(X)
    Xt_compact = est_compact.fit_transform(X)
    assert_allclose(Xt_chain, Xt_compact)

    Xt_chain = est_chain.transform(X2)
    Xt_compact = est_compact.transform(X2)
    assert_allclose(Xt_chain, Xt_compact)


@pytest.mark.parametrize(
    "metric, p, is_euclidean",
    [
        ("euclidean", 2, True),
        ("manhattan", 1, False),
        ("minkowski", 1, False),
        ("minkowski", 2, True),
        (lambda x1, x2: np.sqrt(np.sum(x1**2 + x2**2)), 2, False),
    ],
)
def test_different_metric(global_dtype, metric, p, is_euclidean):
    # [FUNCTION DOC]
    # Parametrized over metric/p combinations including a callable. Checks that
    #   Euclidean equivalent settings (euclidean, minkowski with p=2) reproduce the
    #   default Isomap embedding, and that non-Euclidean ones produce a different
    #   embedding (the comparison is expected to fail with "Not equal to tolerance").
    # Isomap must work on various metric parameters work correctly
    # and must default to euclidean.
    X, _ = datasets.make_blobs(random_state=0)
    X = X.astype(global_dtype, copy=False)

    reference = manifold.Isomap().fit_transform(X)
    embedding = manifold.Isomap(metric=metric, p=p).fit_transform(X)

    if is_euclidean:
        assert_allclose(embedding, reference)
    else:
        with pytest.raises(AssertionError, match="Not equal to tolerance"):
            assert_allclose(embedding, reference)


def test_isomap_clone_bug():
    # [FUNCTION DOC]
    # Regression test for #6062: after set_params(n_neighbors=k) and refitting, the
    #   internal nbrs_ NearestNeighbors object must use the new k, for k in 10, 15, 20.
    # Side effects: draws data from the global numpy random state (unseeded).
    # regression test for bug reported in #6062
    model = manifold.Isomap()
    for n_neighbors in [10, 15, 20]:
        model.set_params(n_neighbors=n_neighbors)
        model.fit(np.random.rand(50, 2))
        assert model.nbrs_.n_neighbors == n_neighbors


@pytest.mark.parametrize("eigen_solver", eigen_solvers)
@pytest.mark.parametrize("path_method", path_methods)
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_sparse_input(
    global_dtype, eigen_solver, path_method, global_random_seed, csr_container
):
    # [FUNCTION DOC]
    # Fits Isomap (2 components, 8 neighbors) on a random 100x3 sparse CSR matrix
    #   (density 0.1) and on its dense version, for every eigen solver, path method and
    #   CSR container, and checks the embeddings agree within rtol/atol 1e-4.
    # TODO: compare results on dense and sparse data as proposed in:
    # https://github.com/scikit-learn/scikit-learn/pull/23585#discussion_r968388186
    X = csr_container(
        _sparse_random_array(
            (100, 3),
            density=0.1,
            format="csr",
            dtype=global_dtype,
            random_state=global_random_seed,
        )
    )

    iso_dense = manifold.Isomap(
        n_components=2,
        eigen_solver=eigen_solver,
        path_method=path_method,
        n_neighbors=8,
    )
    iso_sparse = clone(iso_dense)

    X_trans_dense = iso_dense.fit_transform(X.toarray())
    X_trans_sparse = iso_sparse.fit_transform(X)

    assert_allclose(X_trans_sparse, X_trans_dense, rtol=1e-4, atol=1e-4)


def test_isomap_fit_precomputed_radius_graph(global_dtype):
    # [FUNCTION DOC]
    # Verifies that fitting on a precomputed radius-neighbors distance graph (radius 10)
    #   of an S-curve gives the same embedding as fitting on raw data with metric
    #   "minkowski" and the same radius.
    # Isomap.fit_transform must yield similar result when using
    # a precomputed distance matrix.

    X, y = datasets.make_s_curve(200, random_state=0)
    X = X.astype(global_dtype, copy=False)
    radius = 10

    g = neighbors.radius_neighbors_graph(X, radius=radius, mode="distance")
    isomap = manifold.Isomap(n_neighbors=None, radius=radius, metric="precomputed")
    isomap.fit(g)
    precomputed_result = isomap.embedding_

    isomap = manifold.Isomap(n_neighbors=None, radius=radius, metric="minkowski")
    result = isomap.fit_transform(X)
    atol = 1e-5 if global_dtype == np.float32 else 0
    assert_allclose(precomputed_result, result, atol=atol)


def test_isomap_fitted_attributes_dtype(global_dtype):
    # [FUNCTION DOC]
    # Verifies dist_matrix_ and embedding_ keep the dtype of the input (global_dtype).
    """Check that the fitted attributes are stored accordingly to the
    data type of X."""
    iso = manifold.Isomap(n_neighbors=2)

    X = np.array([[1, 2], [3, 4], [5, 6]], dtype=global_dtype)

    iso.fit(X)

    assert iso.dist_matrix_.dtype == global_dtype
    assert iso.embedding_.dtype == global_dtype


def test_isomap_dtype_equivalence():
    # [FUNCTION DOC]
    # Verifies dist_matrix_ is numerically the same when fitting float32 and float64
    #   versions of the same tiny dataset.
    """Check the equivalence of the results with 32 and 64 bits input."""
    iso_32 = manifold.Isomap(n_neighbors=2)
    X_32 = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
    iso_32.fit(X_32)

    iso_64 = manifold.Isomap(n_neighbors=2)
    X_64 = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float64)
    iso_64.fit(X_64)

    assert_allclose(iso_32.dist_matrix_, iso_64.dist_matrix_)


def test_isomap_raise_error_when_neighbor_and_radius_both_set():
    # [FUNCTION DOC]
    # Verifies fit_transform raises ValueError "Both n_neighbors and radius are
    #   provided" when both parameters are set (digits data).
    # Isomap.fit_transform must raise a ValueError if
    # radius and n_neighbors are provided.

    X, _ = datasets.load_digits(return_X_y=True)
    isomap = manifold.Isomap(n_neighbors=3, radius=5.5)
    msg = "Both n_neighbors and radius are provided"
    with pytest.raises(ValueError, match=msg):
        isomap.fit_transform(X)


def test_multiple_connected_components():
    # [FUNCTION DOC]
    # Verifies that a UserWarning mentioning "number of connected components" is emitted
    #   when the 2-nearest-neighbor graph of 1-D data with two separated clusters is
    #   disconnected.
    # Test that a warning is raised when the graph has multiple components
    X = np.array([0, 1, 2, 5, 6, 7])[:, None]
    with pytest.warns(UserWarning, match="number of connected components"):
        manifold.Isomap(n_neighbors=2).fit(X)


def test_multiple_connected_components_metric_precomputed(global_dtype):
    # [FUNCTION DOC]
    # With metric="precomputed": a dense full distance matrix of disconnected data only
    #   triggers the connected-components UserWarning, whereas a sparse precomputed
    #   neighbors graph that is disconnected raises RuntimeError (it cannot be
    #   completed).
    # Test that an error is raised when the graph has multiple components
    # and when X is a precomputed neighbors graph.
    X = np.array([0, 1, 2, 5, 6, 7])[:, None].astype(global_dtype, copy=False)

    # works with a precomputed distance matrix (dense)
    X_distances = pairwise_distances(X)
    with pytest.warns(UserWarning, match="number of connected components"):
        manifold.Isomap(n_neighbors=1, metric="precomputed").fit(X_distances)

    # does not work with a precomputed neighbors graph (sparse)
    X_graph = neighbors.kneighbors_graph(X, n_neighbors=2, mode="distance")
    with pytest.raises(RuntimeError, match="number of connected components"):
        manifold.Isomap(n_neighbors=1, metric="precomputed").fit(X_graph)


def test_get_feature_names_out():
    # [FUNCTION DOC]
    # Verifies get_feature_names_out returns "isomap0", "isomap1" for 2 components.
    """Check get_feature_names_out for Isomap."""
    X, y = make_blobs(random_state=0, n_features=4)
    n_components = 2

    iso = manifold.Isomap(n_components=n_components)
    iso.fit_transform(X)
    names = iso.get_feature_names_out()
    assert_array_equal([f"isomap{i}" for i in range(n_components)], names)
