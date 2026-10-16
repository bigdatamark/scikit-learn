# [MODULE DOC]
# Tests for t-SNE (sklearn.manifold.TSNE and the private helpers in _t_sne,
#   _barnes_hut_tsne and _utils): the gradient descent stopping rules, perplexity binary
#   search, joint probabilities, KL divergence and its gradient (exact and Barnes-Hut),
#   trustworthiness, input validation for precomputed/sparse distances, parameter
#   effects (early_exaggeration, max_iter, min_grad_norm, n_iter_without_progress,
#   n_jobs), metrics, verbose output and dtype handling. Module level builds X_2d_grid,
#   a 10x10 uniform grid in the unit square used by test_uniform_grid.
import re
import sys
from io import StringIO

import numpy as np
import pytest
import scipy.sparse as sp
from numpy.testing import assert_allclose
from scipy.optimize import check_grad
from scipy.spatial.distance import pdist, squareform

from sklearn import config_context
from sklearn.datasets import make_blobs
from sklearn.manifold import (
    TSNE,
    _barnes_hut_tsne,
)
from sklearn.manifold._t_sne import (
    _gradient_descent,
    _joint_probabilities,
    _joint_probabilities_nn,
    _kl_divergence,
    _kl_divergence_bh,
    trustworthiness,
)
from sklearn.manifold._utils import _binary_search_perplexity
from sklearn.metrics.pairwise import (
    cosine_distances,
    manhattan_distances,
    pairwise_distances,
)
from sklearn.neighbors import NearestNeighbors, kneighbors_graph
from sklearn.utils import check_random_state
from sklearn.utils._testing import (
    assert_almost_equal,
    assert_array_almost_equal,
    assert_array_equal,
    skip_if_32bit,
)
from sklearn.utils.fixes import CSR_CONTAINERS, LIL_CONTAINERS

x = np.linspace(0, 1, 10)
xx, yy = np.meshgrid(x, x)
X_2d_grid = np.hstack(
    [
        xx.ravel().reshape(-1, 1),
        yy.ravel().reshape(-1, 1),
    ]
)


def test_gradient_descent_stops(capsys):
    # [FUNCTION DOC]
    # Exercises the three stopping criteria of _gradient_descent with zero learning rate
    #   and verbose=2, inspecting returned error/iteration and captured stdout: gradient
    #   norm below min_grad_norm stops at iteration 0 ("gradient norm"); a flat
    #   objective stops after n_iter_without_progress=10 iterations at it 11 ("did not
    #   make any progress"); and max_iter=11 stops at iteration 10 ("Iteration 10").
    # Test stopping conditions of gradient descent.
    class ObjectiveSmallGradient:
        # [CLASS DOC]
        # HELPER: Callable objective whose error decreases by 0.1 per call starting at
        #   1.0 and whose gradient is always the tiny constant 1e-5.
        def __init__(self):
            # [FUNCTION DOC]
            # HELPER: Sets the call counter it to -1 so the first call reports iteration
            #   0.
            self.it = -1

        def __call__(self, _, compute_error=True):
            # [FUNCTION DOC]
            # HELPER: Increments the counter and returns the error (10 - it) / 10 and a
            #   gradient array holding 1e-5. Inputs: parameters (ignored), compute_error
            #   (ignored).
            # Side effects: mutates self.it.
            self.it += 1
            return (10 - self.it) / 10.0, np.array([1e-5])

    def flat_function(_, compute_error=True):
        # [FUNCTION DOC]
        # HELPER: Objective with constant error 0.0 and gradient of ones, so no progress
        #   is ever made. Inputs are ignored.
        return 0.0, np.ones(1)

    # Gradient norm
    _, error, it = _gradient_descent(
        ObjectiveSmallGradient(),
        np.zeros(1),
        0,
        max_iter=100,
        n_iter_without_progress=100,
        momentum=0.0,
        learning_rate=0.0,
        min_gain=0.0,
        min_grad_norm=1e-5,
        verbose=2,
    )
    assert error == 1.0
    assert it == 0
    assert "gradient norm" in capsys.readouterr().out

    # Maximum number of iterations without improvement
    _, error, it = _gradient_descent(
        flat_function,
        np.zeros(1),
        0,
        max_iter=100,
        n_iter_without_progress=10,
        momentum=0.0,
        learning_rate=0.0,
        min_gain=0.0,
        min_grad_norm=0.0,
        verbose=2,
    )
    assert error == 0.0
    assert it == 11
    assert "did not make any progress" in capsys.readouterr().out

    # Maximum number of iterations
    _, error, it = _gradient_descent(
        ObjectiveSmallGradient(),
        np.zeros(1),
        0,
        max_iter=11,
        n_iter_without_progress=100,
        momentum=0.0,
        learning_rate=0.0,
        min_gain=0.0,
        min_grad_norm=0.0,
        verbose=2,
    )
    assert error == 0.0
    assert it == 10
    assert "Iteration 10" in capsys.readouterr().out


def test_binary_search():
    # [FUNCTION DOC]
    # Verifies _binary_search_perplexity on 50 random 5-D points: the mean over rows of
    #   the exponential of the entropy of each conditional distribution equals the
    #   desired perplexity 25 to 3 decimals.
    # Test if the binary search finds Gaussians with desired perplexity.
    random_state = check_random_state(0)
    data = random_state.randn(50, 5)
    distances = pairwise_distances(data).astype(np.float32)
    desired_perplexity = 25.0
    P = _binary_search_perplexity(distances, desired_perplexity, verbose=0)
    P = np.maximum(P, np.finfo(np.double).eps)
    mean_perplexity = np.mean(
        [np.exp(-np.sum(P[i] * np.log(P[i]))) for i in range(P.shape[0])]
    )
    assert_almost_equal(mean_perplexity, desired_perplexity, decimal=3)


def test_binary_search_underflow():
    # [FUNCTION DOC]
    # Regression for #19471: for a single float32 row of 90 large values (offset 100),
    #   the base-2 perplexity of the resulting distribution (excluding the first entry)
    #   equals 30 to 3 decimals, i.e. no underflow.
    # Test if the binary search finds Gaussians with desired perplexity.
    # A more challenging case than the one above, producing numeric
    # underflow in float precision (see issue #19471 and PR #19472).
    random_state = check_random_state(42)
    data = random_state.randn(1, 90).astype(np.float32) + 100
    desired_perplexity = 30.0
    P = _binary_search_perplexity(data, desired_perplexity, verbose=0)
    perplexity = 2 ** -np.nansum(P[0, 1:] * np.log2(P[0, 1:]))
    assert_almost_equal(perplexity, desired_perplexity, decimal=3)


def test_binary_search_neighbors():
    # [FUNCTION DOC]
    # Verifies the nearest-neighbor variant: running the search on the k-NN distance
    #   rows with all n-1 neighbors reproduces the dense conditional probabilities
    #   restricted to those neighbors (4 decimals); with fewer neighbors (150 to 199)
    #   the largest 10k probabilities match the dense ones to 2 decimals.
    # KNOWN ISSUE: inside the loop the assertion comparing P1_nn and P2 is
    #   loop-invariant (it re-checks the full-neighbor result instead of P2k), so it
    #   adds nothing.
    # Binary perplexity search approximation.
    # Should be approximately equal to the slow method when we use
    # all points as neighbors.
    n_samples = 200
    desired_perplexity = 25.0
    random_state = check_random_state(0)
    data = random_state.randn(n_samples, 2).astype(np.float32, copy=False)
    distances = pairwise_distances(data)
    P1 = _binary_search_perplexity(distances, desired_perplexity, verbose=0)

    # Test that when we use all the neighbors the results are identical
    n_neighbors = n_samples - 1
    nn = NearestNeighbors().fit(data)
    distance_graph = nn.kneighbors_graph(n_neighbors=n_neighbors, mode="distance")
    distances_nn = distance_graph.data.astype(np.float32, copy=False)
    distances_nn = distances_nn.reshape(n_samples, n_neighbors)
    P2 = _binary_search_perplexity(distances_nn, desired_perplexity, verbose=0)

    indptr = distance_graph.indptr
    P1_nn = np.array(
        [
            P1[k, distance_graph.indices[indptr[k] : indptr[k + 1]]]
            for k in range(n_samples)
        ]
    )
    assert_array_almost_equal(P1_nn, P2, decimal=4)

    # Test that the highest P_ij are the same when fewer neighbors are used
    for k in np.linspace(150, n_samples - 1, 5):
        k = int(k)
        topn = k * 10  # check the top 10 * k entries out of k * k entries
        distance_graph = nn.kneighbors_graph(n_neighbors=k, mode="distance")
        distances_nn = distance_graph.data.astype(np.float32, copy=False)
        distances_nn = distances_nn.reshape(n_samples, k)
        P2k = _binary_search_perplexity(distances_nn, desired_perplexity, verbose=0)
        assert_array_almost_equal(P1_nn, P2, decimal=2)
        idx = np.argsort(P1.ravel())[::-1]
        P1top = P1.ravel()[idx][:topn]
        idx = np.argsort(P2k.ravel())[::-1]
        P2top = P2k.ravel()[idx][:topn]
        assert_array_almost_equal(P1top, P2top, decimal=2)


def test_binary_perplexity_stability():
    # [FUNCTION DOC]
    # Regression for an uninitialized P array: 100 repeated runs of
    #   _binary_search_perplexity and _joint_probabilities_nn on the same 10-NN
    #   distances must give results equal (4 decimals) to the first run.
    # Binary perplexity search should be stable.
    # The binary_search_perplexity had a bug wherein the P array
    # was uninitialized, leading to sporadically failing tests.
    n_neighbors = 10
    n_samples = 100
    random_state = check_random_state(0)
    data = random_state.randn(n_samples, 5)
    nn = NearestNeighbors().fit(data)
    distance_graph = nn.kneighbors_graph(n_neighbors=n_neighbors, mode="distance")
    distances = distance_graph.data.astype(np.float32, copy=False)
    distances = distances.reshape(n_samples, n_neighbors)
    last_P = None
    desired_perplexity = 3
    for _ in range(100):
        P = _binary_search_perplexity(distances.copy(), desired_perplexity, verbose=0)
        P1 = _joint_probabilities_nn(distance_graph, desired_perplexity, verbose=0)
        # Convert the sparse matrix to a dense one for testing
        P1 = P1.toarray()
        if last_P is None:
            last_P = P
            last_P1 = P1
        else:
            assert_array_almost_equal(P, last_P, decimal=4)
            assert_array_almost_equal(P1, last_P1, decimal=4)


def test_gradient():
    # [FUNCTION DOC]
    # Uses scipy check_grad to verify the analytic gradient of _kl_divergence matches
    #   finite differences (to 5 decimals) for 50 random 2-D embeddings with joint
    #   probabilities from a random symmetric non-negative matrix with zero diagonal.
    # Test gradient of Kullback-Leibler divergence.
    random_state = check_random_state(0)

    n_samples = 50
    n_features = 2
    n_components = 2
    alpha = 1.0

    distances = random_state.randn(n_samples, n_features).astype(np.float32)
    distances = np.abs(distances.dot(distances.T))
    np.fill_diagonal(distances, 0.0)
    X_embedded = random_state.randn(n_samples, n_components).astype(np.float32)

    P = _joint_probabilities(distances, desired_perplexity=25.0, verbose=0)

    def fun(params):
        # [FUNCTION DOC]
        # HELPER: Returns the KL divergence value for flattened embedding params.
        return _kl_divergence(params, P, alpha, n_samples, n_components)[0]

    def grad(params):
        # [FUNCTION DOC]
        # HELPER: Returns the flattened KL divergence gradient for embedding params.
        return _kl_divergence(params, P, alpha, n_samples, n_components)[1]

    assert_almost_equal(check_grad(fun, grad, X_embedded.ravel()), 0.0, decimal=5)


def test_trustworthiness():
    # [FUNCTION DOC]
    # Verifies trustworthiness is 1 for an affine transform of the data, below 0.6 for a
    #   random shuffle of 1-D points, and exactly 0.2 for a specific 5-point permutation
    #   with 1 neighbor.
    # Test trustworthiness score.
    random_state = check_random_state(0)

    # Affine transformation
    X = random_state.randn(100, 2)
    assert trustworthiness(X, 5.0 + X / 10.0) == 1.0

    # Randomly shuffled
    X = np.arange(100).reshape(-1, 1)
    X_embedded = X.copy()
    random_state.shuffle(X_embedded)
    assert trustworthiness(X, X_embedded) < 0.6

    # Completely different
    X = np.arange(5).reshape(-1, 1)
    X_embedded = np.array([[0], [2], [4], [1], [3]])
    assert_almost_equal(trustworthiness(X, X_embedded, n_neighbors=1), 0.2)


def test_trustworthiness_n_neighbors_error():
    # [FUNCTION DOC]
    # Non-regression for #18567: n_neighbors=5 with 7 samples raises ValueError (must be
    #   less than n_samples / 2), while n_neighbors=3 returns a score in [0, 1].
    """Raise an error when n_neighbors >= n_samples / 2.

    Non-regression test for #18567.
    """
    regex = "n_neighbors .+ should be less than .+"
    rng = np.random.RandomState(42)
    X = rng.rand(7, 4)
    X_embedded = rng.rand(7, 2)
    with pytest.raises(ValueError, match=regex):
        trustworthiness(X, X_embedded, n_neighbors=5)

    trust = trustworthiness(X, X_embedded, n_neighbors=3)
    assert 0 <= trust <= 1


@pytest.mark.parametrize("method", ["exact", "barnes_hut"])
@pytest.mark.parametrize("init", ("random", "pca"))
def test_preserve_trustworthiness_approximately(method, init):
    # [FUNCTION DOC]
    # For exact and barnes_hut methods and random/pca init, a 2-D embedding of 50 random
    #   2-D points must have 1-neighbor trustworthiness above 0.85.
    # Nearest neighbors should be preserved approximately.
    random_state = check_random_state(0)
    n_components = 2
    X = random_state.randn(50, n_components).astype(np.float32)
    tsne = TSNE(
        n_components=n_components,
        init=init,
        random_state=0,
        method=method,
        max_iter=700,
        learning_rate="auto",
    )
    X_embedded = tsne.fit_transform(X)
    t = trustworthiness(X, X_embedded, n_neighbors=1)
    assert t > 0.85


def test_optimization_minimizes_kl_divergence():
    # [FUNCTION DOC]
    # Verifies kl_divergence_ is non-increasing as max_iter grows from 250 to 300 to 350
    #   with otherwise identical settings on blob data.
    """t-SNE should give a lower KL divergence with more iterations."""
    random_state = check_random_state(0)
    X, _ = make_blobs(n_features=3, random_state=random_state)
    kl_divergences = []
    for max_iter in [250, 300, 350]:
        tsne = TSNE(
            n_components=2,
            init="random",
            perplexity=10,
            learning_rate=100.0,
            max_iter=max_iter,
            random_state=0,
        )
        tsne.fit_transform(X)
        kl_divergences.append(tsne.kl_divergence_)
    assert kl_divergences[1] <= kl_divergences[0]
    assert kl_divergences[2] <= kl_divergences[1]


@pytest.mark.parametrize("method", ["exact", "barnes_hut"])
@pytest.mark.parametrize("init", ["random", "pca"])
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_fit_transform_csr_matrix(method, init, csr_container):
    # [FUNCTION DOC]
    # Fits TSNE on a 50x3 random matrix with ~25 zeros converted to CSR, for each
    #   method, init and CSR container, and checks 1-neighbor trustworthiness is close
    #   to 1 (rtol 0.11).
    # TODO: compare results on dense and sparse data as proposed in:
    # https://github.com/scikit-learn/scikit-learn/pull/23585#discussion_r968388186
    # X can be a sparse matrix.
    rng = check_random_state(0)
    X = rng.randn(50, 3)
    X[(rng.randint(0, 50, 25), rng.randint(0, 3, 25))] = 0.0
    X_csr = csr_container(X)
    tsne = TSNE(
        n_components=2,
        init=init,
        perplexity=10,
        learning_rate=100.0,
        random_state=0,
        method=method,
        max_iter=750,
    )
    X_embedded = tsne.fit_transform(X_csr)
    assert_allclose(trustworthiness(X_csr, X_embedded, n_neighbors=1), 1.0, rtol=1.1e-1)


def test_preserve_trustworthiness_approximately_with_precomputed_distances():
    # [FUNCTION DOC]
    # For 3 seeds, fits TSNE with metric="precomputed" on a distance matrix of 80 random
    #   2-D points and checks precomputed trustworthiness above 0.95.
    # KNOWN ISSUE: the second positional argument of squareform is force, not a metric,
    #   so "sqeuclidean" has no effect and D holds plain Euclidean distances.
    # Nearest neighbors should be preserved approximately.
    random_state = check_random_state(0)
    for i in range(3):
        X = random_state.randn(80, 2)
        D = squareform(pdist(X), "sqeuclidean")
        tsne = TSNE(
            n_components=2,
            perplexity=2,
            learning_rate=100.0,
            early_exaggeration=2.0,
            metric="precomputed",
            random_state=i,
            verbose=0,
            max_iter=500,
            init="random",
        )
        X_embedded = tsne.fit_transform(D)
        t = trustworthiness(D, X_embedded, n_neighbors=1, metric="precomputed")
        assert t > 0.95


def test_trustworthiness_not_euclidean_metric():
    # [FUNCTION DOC]
    # Verifies trustworthiness with metric="cosine" equals trustworthiness computed from
    #   the precomputed cosine distance matrix.
    # Test trustworthiness with a metric different from 'euclidean' and
    # 'precomputed'
    random_state = check_random_state(0)
    X = random_state.randn(100, 2)
    assert trustworthiness(X, X, metric="cosine") == trustworthiness(
        pairwise_distances(X, metric="cosine"), X, metric="precomputed"
    )


@pytest.mark.parametrize(
    "method, retype",
    [
        ("exact", np.asarray),
        ("barnes_hut", np.asarray),
        *[("barnes_hut", csr_container) for csr_container in CSR_CONTAINERS],
    ],
)
@pytest.mark.parametrize(
    "D, message_regex",
    [
        ([[0.0], [1.0]], ".* square distance matrix"),
        ([[0.0, -1.0], [1.0, 0.0]], ".* positive.*"),
    ],
)
def test_bad_precomputed_distances(method, D, retype, message_regex):
    # [FUNCTION DOC]
    # Verifies metric="precomputed" raises ValueError for a non-square matrix ("square
    #   distance matrix") and for a matrix with negative entries ("positive"), for exact
    #   (dense) and barnes_hut (dense and CSR) methods.
    tsne = TSNE(
        metric="precomputed",
        method=method,
        init="random",
        random_state=42,
        perplexity=1,
    )
    with pytest.raises(ValueError, match=message_regex):
        tsne.fit_transform(retype(D))


@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_exact_no_precomputed_sparse(csr_container):
    # [FUNCTION DOC]
    # Verifies method="exact" with a sparse precomputed distance matrix raises TypeError
    #   mentioning "sparse".
    tsne = TSNE(
        metric="precomputed",
        method="exact",
        init="random",
        random_state=42,
        perplexity=1,
    )
    with pytest.raises(TypeError, match="sparse"):
        tsne.fit_transform(csr_container([[0, 5], [5, 0]]))


@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_high_perplexity_precomputed_sparse_distances(csr_container):
    # [FUNCTION DOC]
    # Verifies a sparse precomputed graph with only 1 stored neighbor per sample raises
    #   ValueError requiring 3 neighbors per sample (the count implied by perplexity=1).
    # Perplexity should be less than 50
    dist = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    bad_dist = csr_container(dist)
    tsne = TSNE(metric="precomputed", init="random", random_state=42, perplexity=1)
    msg = "3 neighbors per samples are required, but some samples have only 1"
    with pytest.raises(ValueError, match=msg):
        tsne.fit_transform(bad_dist)


@pytest.mark.filterwarnings(
    "ignore:Precomputed sparse input was not sorted by "
    "row values:sklearn.exceptions.EfficiencyWarning"
)
@pytest.mark.parametrize("sparse_container", CSR_CONTAINERS + LIL_CONTAINERS)
def test_sparse_precomputed_distance(sparse_container):
    # [FUNCTION DOC]
    # Verifies a full kneighbors_graph (100 neighbors including self) equals the dense
    #   distance matrix and that TSNE on it (CSR or LIL) gives the same embedding as on
    #   the dense matrix. Unsorted-input EfficiencyWarning is filtered.
    """Make sure that TSNE works identically for sparse and dense matrix"""
    random_state = check_random_state(0)
    X = random_state.randn(100, 2)

    D_sparse = kneighbors_graph(X, n_neighbors=100, mode="distance", include_self=True)
    D = pairwise_distances(X)
    assert sp.issparse(D_sparse)
    assert_almost_equal(D_sparse.toarray(), D)

    tsne = TSNE(
        metric="precomputed", random_state=0, init="random", learning_rate="auto"
    )
    Xt_dense = tsne.fit_transform(D)

    Xt_sparse = tsne.fit_transform(sparse_container(D_sparse))
    assert_almost_equal(Xt_dense, Xt_sparse)


def test_non_positive_computed_distances():
    # [FUNCTION DOC]
    # Verifies a callable metric returning negative distances triggers ValueError "All
    #   distances ... metric given" in exact mode.
    # Computed distance matrices must be positive.
    def metric(x, y):
        # [FUNCTION DOC]
        # HELPER: Dummy metric that always returns -1.
        return -1

    # Negative computed distances should be caught even if result is squared
    tsne = TSNE(metric=metric, method="exact", perplexity=1)
    X = np.array([[0.0, 0.0], [1.0, 1.0]])
    with pytest.raises(ValueError, match="All distances .*metric given.*"):
        tsne.fit_transform(X)


def test_init_ndarray():
    # [FUNCTION DOC]
    # Verifies an all-zeros ndarray init on constant data yields an all-zeros embedding
    #   (no gradient), i.e. the given init is used as is.
    # Initialize TSNE with ndarray and test fit
    tsne = TSNE(init=np.zeros((100, 2)), learning_rate="auto")
    X_embedded = tsne.fit_transform(np.ones((100, 5)))
    assert_array_equal(np.zeros((100, 2)), X_embedded)


def test_init_ndarray_precomputed():
    # [FUNCTION DOC]
    # Smoke test that fitting with an ndarray init and metric="precomputed" works (no
    #   FutureWarning expected from _fit, though not explicitly asserted).
    # Initialize TSNE with ndarray and metric 'precomputed'
    # Make sure no FutureWarning is thrown from _fit
    tsne = TSNE(
        init=np.zeros((100, 2)),
        metric="precomputed",
        learning_rate=50.0,
    )
    tsne.fit(np.zeros((100, 100)))


def test_pca_initialization_not_compatible_with_precomputed_kernel():
    # [FUNCTION DOC]
    # Verifies init="pca" with metric="precomputed" raises ValueError.
    # Precomputed distance matrices cannot use PCA initialization.
    tsne = TSNE(metric="precomputed", init="pca", perplexity=1)
    with pytest.raises(
        ValueError,
        match='The parameter init="pca" cannot be used with metric="precomputed".',
    ):
        tsne.fit_transform(np.array([[0.0], [1.0]]))


def test_n_components_range():
    # [FUNCTION DOC]
    # Verifies barnes_hut with n_components=4 raises ValueError about 'n_components'.
    # barnes_hut method should only be used with n_components <= 3
    tsne = TSNE(n_components=4, method="barnes_hut", perplexity=1)
    with pytest.raises(ValueError, match="'n_components' should be .*"):
        tsne.fit_transform(np.array([[0.0], [1.0]]))


def test_early_exaggeration_used():
    # [FUNCTION DOC]
    # For both methods, embeddings with early_exaggeration 1 and 10 (same seed, pca
    #   init) must differ.
    # check that the ``early_exaggeration`` parameter has an effect
    random_state = check_random_state(0)
    n_components = 2
    methods = ["exact", "barnes_hut"]
    X = random_state.randn(25, n_components).astype(np.float32)
    for method in methods:
        tsne = TSNE(
            n_components=n_components,
            perplexity=1,
            learning_rate=100.0,
            init="pca",
            random_state=0,
            method=method,
            early_exaggeration=1.0,
            max_iter=250,
        )
        X_embedded1 = tsne.fit_transform(X)
        tsne = TSNE(
            n_components=n_components,
            perplexity=1,
            learning_rate=100.0,
            init="pca",
            random_state=0,
            method=method,
            early_exaggeration=10.0,
            max_iter=250,
        )
        X_embedded2 = tsne.fit_transform(X)

        assert not np.allclose(X_embedded1, X_embedded2)


def test_max_iter_used():
    # [FUNCTION DOC]
    # For both methods and max_iter 251 and 500, checks n_iter_ equals max_iter - 1 (the
    #   final iteration index).
    # check that the ``max_iter`` parameter has an effect
    random_state = check_random_state(0)
    n_components = 2
    methods = ["exact", "barnes_hut"]
    X = random_state.randn(25, n_components).astype(np.float32)
    for method in methods:
        for max_iter in [251, 500]:
            tsne = TSNE(
                n_components=n_components,
                perplexity=1,
                learning_rate=0.5,
                init="random",
                random_state=0,
                method=method,
                early_exaggeration=1.0,
                max_iter=max_iter,
            )
            tsne.fit_transform(X)

            assert tsne.n_iter_ == max_iter - 1


@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_answer_gradient_two_points(csr_container):
    # [FUNCTION DOC]
    # Checks the Barnes-Hut gradient for 2 points against reference values from the
    #   original implementation (via _run_answer_test).
    # Test the tree with only a single set of children.
    #
    # These tests & answers have been checked against the reference
    # implementation by LvdM.
    pos_input = np.array([[1.0, 0.0], [0.0, 1.0]])
    pos_output = np.array(
        [[-4.961291e-05, -1.072243e-04], [9.259460e-05, 2.702024e-04]]
    )
    neighbors = np.array([[1], [0]])
    grad_output = np.array(
        [[-2.37012478e-05, -6.29044398e-05], [2.37012478e-05, 6.29044398e-05]]
    )
    _run_answer_test(pos_input, pos_output, neighbors, grad_output, csr_container)


@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_answer_gradient_four_points(csr_container):
    # [FUNCTION DOC]
    # Checks the Barnes-Hut gradient for 4 points (multi-level tree) against reference
    #   values.
    # Four points tests the tree with multiple levels of children.
    #
    # These tests & answers have been checked against the reference
    # implementation by LvdM.
    pos_input = np.array([[1.0, 0.0], [0.0, 1.0], [5.0, 2.0], [7.3, 2.2]])
    pos_output = np.array(
        [
            [6.080564e-05, -7.120823e-05],
            [-1.718945e-04, -4.000536e-05],
            [-2.271720e-04, 8.663310e-05],
            [-1.032577e-04, -3.582033e-05],
        ]
    )
    neighbors = np.array([[1, 2, 3], [0, 2, 3], [1, 0, 3], [1, 2, 0]])
    grad_output = np.array(
        [
            [5.81128448e-05, -7.78033454e-06],
            [-5.81526851e-05, 7.80976444e-06],
            [4.24275173e-08, -3.69569698e-08],
            [-2.58720939e-09, 7.52706374e-09],
        ]
    )
    _run_answer_test(pos_input, pos_output, neighbors, grad_output, csr_container)


@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_skip_num_points_gradient(csr_container):
    # [FUNCTION DOC]
    # Intends to check that skip_num_points=2 zeroes the gradient of the first two
    #   points, using the four-point data.
    # KNOWN ISSUE: _run_answer_test ignores its skip_num_points argument (always passes
    #   0 to gradient) and compares only to 4 decimals while the gradients are of order
    #   1e-5, so this test cannot detect whether skipping works.
    # Test the kwargs option skip_num_points.
    #
    # Skip num points should make it such that the Barnes_hut gradient
    # is not calculated for indices below skip_num_point.
    # Aside from skip_num_points=2 and the first two gradient rows
    # being set to zero, these data points are the same as in
    # test_answer_gradient_four_points()
    pos_input = np.array([[1.0, 0.0], [0.0, 1.0], [5.0, 2.0], [7.3, 2.2]])
    pos_output = np.array(
        [
            [6.080564e-05, -7.120823e-05],
            [-1.718945e-04, -4.000536e-05],
            [-2.271720e-04, 8.663310e-05],
            [-1.032577e-04, -3.582033e-05],
        ]
    )
    neighbors = np.array([[1, 2, 3], [0, 2, 3], [1, 0, 3], [1, 2, 0]])
    grad_output = np.array(
        [
            [0.0, 0.0],
            [0.0, 0.0],
            [4.24275173e-08, -3.69569698e-08],
            [-2.58720939e-09, 7.52706374e-09],
        ]
    )
    _run_answer_test(
        pos_input, pos_output, neighbors, grad_output, csr_container, False, 0.1, 2
    )


def _run_answer_test(
    pos_input,
    pos_output,
    neighbors,
    grad_output,
    csr_container,
    verbose=False,
    perplexity=0.1,
    skip_num_points=0,
):
    # [FUNCTION DOC]
    # HELPER: Computes joint probabilities from the Euclidean distances of pos_input
    #   with the given perplexity, converts them to a CSR matrix, runs the Cython
    #   Barnes-Hut gradient for embedding pos_output (theta 0.5, 2 components, verbose
    #   1) and asserts the result equals grad_output to 4 decimals.
    # Inputs: pos_input, pos_output - (n, 2) arrays; neighbors - neighbor index array;
    #   grad_output - expected gradient; csr_container - sparse constructor; verbose
    #   (False), perplexity (0.1), skip_num_points (0).
    # Exceptions: AssertionError on mismatch.
    # KNOWN ISSUE: the neighbors argument is overwritten by the CSR indices and
    #   skip_num_points is never forwarded (0 is hard-coded).
    distances = pairwise_distances(pos_input).astype(np.float32)
    args = distances, perplexity, verbose
    pos_output = pos_output.astype(np.float32)
    neighbors = neighbors.astype(np.int64, copy=False)
    pij_input = _joint_probabilities(*args)
    pij_input = squareform(pij_input).astype(np.float32)
    grad_bh = np.zeros(pos_output.shape, dtype=np.float32)

    P = csr_container(pij_input)

    neighbors = P.indices.astype(np.int64)
    indptr = P.indptr.astype(np.int64)

    _barnes_hut_tsne.gradient(
        P.data, pos_output, neighbors, indptr, grad_bh, 0.5, 2, 1, skip_num_points=0
    )
    assert_array_almost_equal(grad_bh, grad_output, decimal=4)


@pytest.mark.thread_unsafe  # manually captured stdout
def test_verbose():
    # [FUNCTION DOC]
    # Redirects sys.stdout to a StringIO, fits TSNE with verbose=2, restores stdout, and
    #   checks expected progress messages are printed. Marked thread_unsafe.
    # Verbose options write to stdout.
    random_state = check_random_state(0)
    tsne = TSNE(verbose=2, perplexity=4)
    X = random_state.randn(5, 2)

    old_stdout = sys.stdout
    sys.stdout = StringIO()
    try:
        tsne.fit_transform(X)
    finally:
        out = sys.stdout.getvalue()
        sys.stdout.close()
        sys.stdout = old_stdout

    assert "[t-SNE]" in out
    assert "nearest neighbors..." in out
    assert "Computed conditional probabilities" in out
    assert "Mean sigma" in out
    assert "early exaggeration" in out


def test_chebyshev_metric():
    # [FUNCTION DOC]
    # Smoke test (#3526): a metric that cannot be squared (chebyshev) is accepted.
    # t-SNE should allow metrics that cannot be squared (issue #3526).
    random_state = check_random_state(0)
    tsne = TSNE(metric="chebyshev", perplexity=4)
    X = random_state.randn(5, 2)
    tsne.fit_transform(X)


def test_reduction_to_one_component():
    # [FUNCTION DOC]
    # Verifies n_components=1 (#4154) produces a finite embedding_.
    # t-SNE should allow reduction to one component (issue #4154).
    random_state = check_random_state(0)
    tsne = TSNE(n_components=1, perplexity=4)
    X = random_state.randn(5, 2)
    X_embedded = tsne.fit(X).embedding_
    assert np.all(np.isfinite(X_embedded))


@pytest.mark.parametrize("method", ["barnes_hut", "exact"])
@pytest.mark.parametrize("dt", [np.float32, np.float64])
def test_64bit(method, dt):
    # [FUNCTION DOC]
    # Verifies the embedding dtype is float32 for both float32 and float64 input and
    #   both methods, since the Cython code is single precision.
    # Ensure 64bit arrays are handled correctly.
    random_state = check_random_state(0)

    X = random_state.randn(10, 2).astype(dt, copy=False)
    tsne = TSNE(
        n_components=2,
        perplexity=2,
        learning_rate=100.0,
        random_state=0,
        method=method,
        verbose=0,
        max_iter=300,
        init="random",
    )
    X_embedded = tsne.fit_transform(X)
    effective_type = X_embedded.dtype

    # tsne cython code is only single precision, so the output will
    # always be single precision, irrespectively of the input dtype
    assert effective_type == np.float32


@pytest.mark.parametrize("method", ["barnes_hut", "exact"])
def test_kl_divergence_not_nan(method):
    # [FUNCTION DOC]
    # Verifies kl_divergence_ is not NaN when max_iter (503) is not a multiple of the
    #   error check interval, i.e. the error is computed at the final iteration.
    # Ensure kl_divergence_ is computed at last iteration
    # even though max_iter % n_iter_check != 0, i.e. 1003 % 50 != 0
    random_state = check_random_state(0)

    X = random_state.randn(50, 2)
    tsne = TSNE(
        n_components=2,
        perplexity=2,
        learning_rate=100.0,
        random_state=0,
        method=method,
        verbose=0,
        max_iter=503,
        init="random",
    )
    tsne.fit_transform(X)

    assert not np.isnan(tsne.kl_divergence_)


def test_barnes_hut_angle():
    # [FUNCTION DOC]
    # For 2 and 3 components, verifies that with angle=0 the Barnes-Hut joint
    #   probabilities (from a full k-NN graph) equal the exact ones (5 decimals) and the
    #   KL divergences agree to 3 decimals.
    # KNOWN ISSUE: the exact and Barnes-Hut gradients are computed but never compared.
    # When Barnes-Hut's angle=0 this corresponds to the exact method.
    angle = 0.0
    perplexity = 10
    n_samples = 100
    for n_components in [2, 3]:
        n_features = 5
        degrees_of_freedom = float(n_components - 1.0)

        random_state = check_random_state(0)
        data = random_state.randn(n_samples, n_features)
        distances = pairwise_distances(data)
        params = random_state.randn(n_samples, n_components)
        P = _joint_probabilities(distances, perplexity, verbose=0)
        kl_exact, grad_exact = _kl_divergence(
            params, P, degrees_of_freedom, n_samples, n_components
        )

        n_neighbors = n_samples - 1
        distances_csr = (
            NearestNeighbors()
            .fit(data)
            .kneighbors_graph(n_neighbors=n_neighbors, mode="distance")
        )
        P_bh = _joint_probabilities_nn(distances_csr, perplexity, verbose=0)
        kl_bh, grad_bh = _kl_divergence_bh(
            params,
            P_bh,
            degrees_of_freedom,
            n_samples,
            n_components,
            angle=angle,
            skip_num_points=0,
            verbose=0,
        )

        P = squareform(P)
        P_bh = P_bh.toarray()
        assert_array_almost_equal(P_bh, P, decimal=5)
        assert_almost_equal(kl_exact, kl_bh, decimal=3)


@skip_if_32bit
def test_n_iter_without_progress(capsys):
    # [FUNCTION DOC]
    # Skipped on 32-bit. With n_iter_without_progress=-1, an extremely large learning
    #   rate, _N_ITER_CHECK forced to 1 and no exploration phase, verbose output must
    #   report that no progress was made "during the last -1 episodes", for both
    #   methods.
    # Use a dummy negative n_iter_without_progress and check output on stdout
    random_state = check_random_state(0)
    X = random_state.randn(100, 10)
    for method in ["barnes_hut", "exact"]:
        tsne = TSNE(
            n_iter_without_progress=-1,
            verbose=2,
            learning_rate=1e8,
            random_state=0,
            method=method,
            max_iter=351,
            init="random",
        )
        tsne._N_ITER_CHECK = 1
        tsne._EXPLORATION_MAX_ITER = 0
        tsne.fit_transform(X)

        # The output needs to contain the value of n_iter_without_progress
        assert (
            "did not make any progress during the last -1 episodes. Finished."
            in capsys.readouterr().out
        )


def test_min_grad_norm(capsys):
    # [FUNCTION DOC]
    # Parses the "gradient norm" values from verbose exact-mode output (up to the
    #   "Finished" line) and checks at most one of them is at or below
    #   min_grad_norm=0.002, since the optimizer stops as soon as this happens.
    # Make sure that the parameter min_grad_norm is used correctly
    random_state = check_random_state(0)
    X = random_state.randn(100, 2)
    min_grad_norm = 0.002
    tsne = TSNE(min_grad_norm=min_grad_norm, verbose=2, random_state=0, method="exact")

    tsne.fit_transform(X)
    lines_out = capsys.readouterr().out.split("\n")

    # extract the gradient norm from the verbose output
    gradient_norm_values = []
    for line in lines_out:
        # When the computation is Finished just an old gradient norm value
        # is repeated that we do not need to store
        if "Finished" in line:
            break

        start_grad_norm = line.find("gradient norm")
        if start_grad_norm >= 0:
            line = line[start_grad_norm:]
            line = line.replace("gradient norm = ", "").split(" ")[0]
            gradient_norm_values.append(float(line))

    # Compute how often the gradient norm is smaller than min_grad_norm
    gradient_norm_values = np.array(gradient_norm_values)
    n_smaller_gradient_norms = len(
        gradient_norm_values[gradient_norm_values <= min_grad_norm]
    )

    # The gradient norm can be smaller than min_grad_norm at most once,
    # because in the moment it becomes smaller the optimization stops
    assert n_smaller_gradient_norms <= 1


def test_accessible_kl_divergence(capsys):
    # [FUNCTION DOC]
    # Parses the error of the last "Iteration" line in verbose output and checks it
    #   equals kl_divergence_ to 5 decimals.
    # KNOWN ISSUE: if no Iteration line carries an error, error is unbound or stays an
    #   empty string and the test fails with an unrelated exception.
    # Ensures that the accessible kl_divergence matches the computed value
    random_state = check_random_state(0)
    X = random_state.randn(50, 2)
    tsne = TSNE(
        n_iter_without_progress=2,
        verbose=2,
        random_state=0,
        method="exact",
        max_iter=500,
    )

    tsne.fit_transform(X)
    # The output needs to contain the accessible kl_divergence as the error at
    # the last iteration
    for line in capsys.readouterr().out.split("\n")[::-1]:
        if "Iteration" in line:
            _, _, error = line.partition("error = ")
            if error:
                error, _, _ = error.partition(",")
                break
    assert_almost_equal(tsne.kl_divergence_, float(error), decimal=5)


@pytest.mark.parametrize("method", ["barnes_hut", "exact"])
def test_uniform_grid(method):
    # [FUNCTION DOC]
    # For each method and 3 seeds, embeds X_2d_grid with perplexity 50 and checks with
    #   assert_uniform_grid that the result is roughly uniform; on failure it retries
    #   once starting from the obtained embedding (tsne.init set to Y) to tolerate bad
    #   local minima.
    """Make sure that TSNE can approximately recover a uniform 2D grid

    Due to ties in distances between point in X_2d_grid, this test is platform
    dependent for ``method='barnes_hut'`` due to numerical imprecision.

    Also, t-SNE is not assured to converge to the right solution because bad
    initialization can lead to convergence to bad local minimum (the
    optimization problem is non-convex). To avoid breaking the test too often,
    we re-run t-SNE from the final point when the convergence is not good
    enough.
    """
    seeds = range(3)
    max_iter = 500
    for seed in seeds:
        tsne = TSNE(
            n_components=2,
            init="random",
            random_state=seed,
            perplexity=50,
            max_iter=max_iter,
            method=method,
            learning_rate="auto",
        )
        Y = tsne.fit_transform(X_2d_grid)

        try_name = "{}_{}".format(method, seed)
        try:
            assert_uniform_grid(Y, try_name)
        except AssertionError:
            # If the test fails a first time, re-run with init=Y to see if
            # this was caused by a bad initialization. Note that this will
            # also run an early_exaggeration step.
            try_name += ":rerun"
            tsne.init = Y
            Y = tsne.fit_transform(X_2d_grid)
            assert_uniform_grid(Y, try_name)


def assert_uniform_grid(Y, try_name=None):
    # [FUNCTION DOC]
    # HELPER: Asserts that the nearest-neighbor distances within embedding Y are all
    #   above 0.1 and that their min and max are within factors 0.5 and 2 of their mean.
    # Inputs: Y - (n, d) embedding; try_name - label used as assertion message (default
    #   None).
    # Exceptions: AssertionError when the spacing is not uniform enough.
    # Ensure that the resulting embedding leads to approximately
    # uniformly spaced points: the distance to the closest neighbors
    # should be non-zero and approximately constant.
    nn = NearestNeighbors(n_neighbors=1).fit(Y)
    dist_to_nn = nn.kneighbors(return_distance=True)[0].ravel()
    assert dist_to_nn.min() > 0.1

    smallest_to_mean = dist_to_nn.min() / np.mean(dist_to_nn)
    largest_to_mean = dist_to_nn.max() / np.mean(dist_to_nn)

    assert smallest_to_mean > 0.5, try_name
    assert largest_to_mean < 2, try_name


def test_bh_match_exact():
    # [FUNCTION DOC]
    # With angle=0, perplexity 29.5 on 30 samples (all points neighbors) and no
    #   exploration phase, exact and barnes_hut must run the same number of iterations
    #   and produce the same embedding (rtol 1e-4).
    # check that the ``barnes_hut`` method match the exact one when
    # ``angle = 0`` and ``perplexity > n_samples / 3``
    random_state = check_random_state(0)
    n_features = 10
    X = random_state.randn(30, n_features).astype(np.float32)
    X_embeddeds = {}
    max_iter = {}
    for method in ["exact", "barnes_hut"]:
        tsne = TSNE(
            n_components=2,
            method=method,
            learning_rate=1.0,
            init="random",
            random_state=0,
            max_iter=251,
            perplexity=29.5,
            angle=0,
        )
        # Kill the early_exaggeration
        tsne._EXPLORATION_MAX_ITER = 0
        X_embeddeds[method] = tsne.fit_transform(X)
        max_iter[method] = tsne.n_iter_

    assert max_iter["exact"] == max_iter["barnes_hut"]
    assert_allclose(X_embeddeds["exact"], X_embeddeds["barnes_hut"], rtol=1e-4)


def test_gradient_bh_multithread_match_sequential():
    # [FUNCTION DOC]
    # Intends to check that _kl_divergence_bh gives the same KL value and gradient with
    #   2 and 4 threads as with 1 thread (angle 3).
    # KNOWN ISSUE: the gradient assertion compares grad_multithread with itself, so only
    #   the KL value is effectively checked.
    # check that the bh gradient with different num_threads gives the same
    # results

    n_features = 10
    n_samples = 30
    n_components = 2
    degrees_of_freedom = 1

    angle = 3
    perplexity = 5

    random_state = check_random_state(0)
    data = random_state.randn(n_samples, n_features).astype(np.float32)
    params = random_state.randn(n_samples, n_components)

    n_neighbors = n_samples - 1
    distances_csr = (
        NearestNeighbors()
        .fit(data)
        .kneighbors_graph(n_neighbors=n_neighbors, mode="distance")
    )
    P_bh = _joint_probabilities_nn(distances_csr, perplexity, verbose=0)
    kl_sequential, grad_sequential = _kl_divergence_bh(
        params,
        P_bh,
        degrees_of_freedom,
        n_samples,
        n_components,
        angle=angle,
        skip_num_points=0,
        verbose=0,
        num_threads=1,
    )
    for num_threads in [2, 4]:
        kl_multithread, grad_multithread = _kl_divergence_bh(
            params,
            P_bh,
            degrees_of_freedom,
            n_samples,
            n_components,
            angle=angle,
            skip_num_points=0,
            verbose=0,
            num_threads=num_threads,
        )

        assert_allclose(kl_multithread, kl_sequential, rtol=1e-6)
        assert_allclose(grad_multithread, grad_multithread)


@pytest.mark.parametrize(
    "metric, dist_func",
    [("manhattan", manhattan_distances), ("cosine", cosine_distances)],
)
@pytest.mark.parametrize("method", ["barnes_hut", "exact"])
def test_tsne_with_different_distance_metrics(metric, dist_func, method):
    # [FUNCTION DOC]
    # For manhattan and cosine metrics and both methods, verifies TSNE with the named
    #   metric equals TSNE on the corresponding precomputed distance matrix; barnes_hut
    #   with manhattan is xfailed due to slightly different distance computations.
    """Make sure that TSNE works for different distance metrics"""

    if method == "barnes_hut" and metric == "manhattan":
        # The distances computed by `manhattan_distances` differ slightly from those
        # computed internally by NearestNeighbors via the PairwiseDistancesReduction
        # Cython code-based. This in turns causes T-SNE to converge to a different
        # solution but this should not impact the qualitative results as both
        # methods.
        # NOTE: it's probably not valid from a mathematical point of view to use the
        # Manhattan distance for T-SNE...
        # TODO: re-enable this test if/when `manhattan_distances` is refactored to
        # reuse the same underlying Cython code NearestNeighbors.
        # For reference, see:
        # https://github.com/scikit-learn/scikit-learn/pull/23865/files#r925721573
        pytest.xfail(
            "Distance computations are different for method == 'barnes_hut' and metric"
            " == 'manhattan', but this is expected."
        )

    random_state = check_random_state(0)
    n_components_original = 3
    n_components_embedding = 2
    X = random_state.randn(50, n_components_original).astype(np.float32)
    X_transformed_tsne = TSNE(
        metric=metric,
        method=method,
        n_components=n_components_embedding,
        random_state=0,
        max_iter=300,
        init="random",
        learning_rate="auto",
    ).fit_transform(X)
    X_transformed_tsne_precomputed = TSNE(
        metric="precomputed",
        method=method,
        n_components=n_components_embedding,
        random_state=0,
        max_iter=300,
        init="random",
        learning_rate="auto",
    ).fit_transform(dist_func(X))
    assert_array_equal(X_transformed_tsne, X_transformed_tsne_precomputed)


@pytest.mark.parametrize("method", ["exact", "barnes_hut"])
def test_tsne_n_jobs(method):
    # [FUNCTION DOC]
    # Verifies n_jobs=1 and n_jobs=2 give the same embedding for each method.
    """Make sure that the n_jobs parameter doesn't impact the output"""
    random_state = check_random_state(0)
    n_features = 10
    X = random_state.randn(30, n_features)
    X_tr_ref = TSNE(
        n_components=2,
        method=method,
        perplexity=25.0,
        angle=0,
        n_jobs=1,
        random_state=0,
        init="random",
        learning_rate="auto",
    ).fit_transform(X)
    X_tr = TSNE(
        n_components=2,
        method=method,
        perplexity=25.0,
        angle=0,
        n_jobs=2,
        random_state=0,
        init="random",
        learning_rate="auto",
    ).fit_transform(X)

    assert_allclose(X_tr_ref, X_tr)


def test_tsne_with_mahalanobis_distance():
    # [FUNCTION DOC]
    # Verifies metric="mahalanobis" without parameters raises ValueError requiring V or
    #   VI, and with metric_params V set to the data covariance gives the same embedding
    #   as the precomputed Mahalanobis distance matrix.
    """Make sure that method_parameters works with mahalanobis distance."""
    random_state = check_random_state(0)
    n_samples, n_features = 300, 10
    X = random_state.randn(n_samples, n_features)
    default_params = {
        "perplexity": 40,
        "max_iter": 250,
        "learning_rate": "auto",
        "init": "random",
        "n_components": 3,
        "random_state": 0,
    }

    tsne = TSNE(metric="mahalanobis", **default_params)
    msg = "Must provide either V or VI for Mahalanobis distance"
    with pytest.raises(ValueError, match=msg):
        tsne.fit_transform(X)

    precomputed_X = squareform(pdist(X, metric="mahalanobis"), checks=True)
    X_trans_expected = TSNE(metric="precomputed", **default_params).fit_transform(
        precomputed_X
    )

    X_trans = TSNE(
        metric="mahalanobis", metric_params={"V": np.cov(X.T)}, **default_params
    ).fit_transform(X)
    assert_allclose(X_trans, X_trans_expected)


@pytest.mark.parametrize("perplexity", (20, 30))
def test_tsne_perplexity_validation(perplexity):
    # [FUNCTION DOC]
    # Verifies perplexity 20 or 30 on 20 samples raises ValueError "perplexity (...)
    #   must be less than n_samples (20)".
    """Make sure that perplexity > n_samples results in a ValueError"""

    random_state = check_random_state(0)
    X = random_state.randn(20, 2)
    est = TSNE(
        learning_rate="auto",
        init="pca",
        perplexity=perplexity,
        random_state=random_state,
    )
    msg = re.escape(f"perplexity ({perplexity}) must be less than n_samples (20)")
    with pytest.raises(ValueError, match=msg):
        est.fit_transform(X)


def test_tsne_works_with_pandas_output():
    # [FUNCTION DOC]
    # Non-regression for gh-25365: fit_transform works under transform_output="pandas"
    #   (skipped without pandas).
    """Make sure that TSNE works when the output is set to "pandas".

    Non-regression test for gh-25365.
    """
    pytest.importorskip("pandas")
    with config_context(transform_output="pandas"):
        arr = np.arange(35 * 4).reshape(35, 4)
        TSNE(n_components=2).fit_transform(arr)
