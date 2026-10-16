# [MODULE DOC]
# Tests for the synthetic data generators in sklearn.datasets (make_classification,
# make_multilabel_classification, make_regression, make_blobs, the Friedman problems,
# low-rank / sparse-coded / SPD matrices, manifolds, biclusters, moons and circles).
# Checks output shapes, statistical/geometric properties, determinism under random_state
# and input validation errors.
import re
from collections import defaultdict
from functools import partial

import numpy as np
import pytest
import scipy.sparse as sp

from sklearn.datasets import (
    make_biclusters,
    make_blobs,
    make_checkerboard,
    make_circles,
    make_classification,
    make_friedman1,
    make_friedman2,
    make_friedman3,
    make_hastie_10_2,
    make_low_rank_matrix,
    make_moons,
    make_multilabel_classification,
    make_regression,
    make_s_curve,
    make_sparse_coded_signal,
    make_sparse_spd_matrix,
    make_sparse_uncorrelated,
    make_spd_matrix,
    make_swiss_roll,
)
from sklearn.utils._testing import (
    assert_allclose,
    assert_allclose_dense_sparse,
    assert_almost_equal,
    assert_array_almost_equal,
    assert_array_equal,
)
from sklearn.utils.validation import assert_all_finite


def test_make_classification():
    # [FUNCTION DOC]
    # Verifies with weights [0.1, 0.25] (third class implied) and 3 classes that the
    # weights list is not mutated, shapes are right and class counts are exactly
    # 10/25/65; then that 31 informative features with hypercube produce 2000 distinct
    # rows (row uniqueness via a structured-dtype view).
    weights = [0.1, 0.25]
    X, y = make_classification(
        n_samples=100,
        n_features=20,
        n_informative=5,
        n_redundant=1,
        n_repeated=1,
        n_classes=3,
        n_clusters_per_class=1,
        hypercube=False,
        shift=None,
        scale=None,
        weights=weights,
        random_state=0,
    )

    assert weights == [0.1, 0.25]
    assert X.shape == (100, 20), "X shape mismatch"
    assert y.shape == (100,), "y shape mismatch"
    assert np.unique(y).shape == (3,), "Unexpected number of classes"
    assert sum(y == 0) == 10, "Unexpected number of samples in class #0"
    assert sum(y == 1) == 25, "Unexpected number of samples in class #1"
    assert sum(y == 2) == 65, "Unexpected number of samples in class #2"

    # Test for n_features > 30
    X, y = make_classification(
        n_samples=2000,
        n_features=31,
        n_informative=31,
        n_redundant=0,
        n_repeated=0,
        hypercube=True,
        scale=0.5,
        random_state=0,
    )

    assert X.shape == (2000, 31), "X shape mismatch"
    assert y.shape == (2000,), "y shape mismatch"
    assert (
        np.unique(X.view([("", X.dtype)] * X.shape[1]))
        .view(X.dtype)
        .reshape(-1, X.shape[1])
        .shape[0]
        == 2000
    ), "Unexpected number of unique rows"


def test_make_classification_informative_features():
    # [FUNCTION DOC]
    # With a huge class_sep and no shift, scale, flips or shuffling, for several
    # (n_informative, weights, n_clusters_per_class) settings and both hypercube values,
    # verifies shapes; that clustering samples by the sign pattern of their features
    # gives exactly n_classes times n_clusters_per_class clusters with
    # n_clusters_per_class per class; class proportions match weights; and that cluster
    # centroids lie on hypercube vertices (scaled by class_sep) only when hypercube is
    # True. Finally verifies ValueError when 2 informative features cannot host the
    # requested clusters.
    # KNOWN ISSUE: the proportion check uses floor division of observed proportions by
    # weights and compares to 1, which is a loose check rather than an equality of
    # ratios.
    """Test the construction of informative features in make_classification

    Also tests `n_clusters_per_class`, `n_classes`, `hypercube` and
    fully-specified `weights`.
    """
    # Create very separate clusters; check that vertices are unique and
    # correspond to classes
    class_sep = 1e6
    make = partial(
        make_classification,
        class_sep=class_sep,
        n_redundant=0,
        n_repeated=0,
        flip_y=0,
        shift=0,
        scale=1,
        shuffle=False,
    )

    for n_informative, weights, n_clusters_per_class in [
        (2, [1], 1),
        (2, [1 / 3] * 3, 1),
        (2, [1 / 4] * 4, 1),
        (2, [1 / 2] * 2, 2),
        (2, [3 / 4, 1 / 4], 2),
        (10, [1 / 3] * 3, 10),
        (64, [1], 1),
    ]:
        n_classes = len(weights)
        n_clusters = n_classes * n_clusters_per_class
        n_samples = n_clusters * 50

        for hypercube in (False, True):
            X, y = make(
                n_samples=n_samples,
                n_classes=n_classes,
                weights=weights,
                n_features=n_informative,
                n_informative=n_informative,
                n_clusters_per_class=n_clusters_per_class,
                hypercube=hypercube,
                random_state=0,
            )

            assert X.shape == (n_samples, n_informative)
            assert y.shape == (n_samples,)

            # Cluster by sign, viewed as strings to allow uniquing
            signs = np.sign(X)
            signs = signs.view(dtype="|S{0}".format(signs.strides[0])).ravel()
            unique_signs, cluster_index = np.unique(signs, return_inverse=True)

            assert len(unique_signs) == n_clusters, (
                "Wrong number of clusters, or not in distinct quadrants"
            )

            clusters_by_class = defaultdict(set)
            for cluster, cls in zip(cluster_index, y):
                clusters_by_class[cls].add(cluster)
            for clusters in clusters_by_class.values():
                assert len(clusters) == n_clusters_per_class, (
                    "Wrong number of clusters per class"
                )
            assert len(clusters_by_class) == n_classes, "Wrong number of classes"

            assert_array_almost_equal(
                np.bincount(y) / len(y) // weights,
                [1] * n_classes,
                err_msg="Wrong number of samples per class",
            )

            # Ensure on vertices of hypercube
            for cluster in range(len(unique_signs)):
                centroid = X[cluster_index == cluster].mean(axis=0)
                if hypercube:
                    assert_array_almost_equal(
                        np.abs(centroid) / class_sep,
                        np.ones(n_informative),
                        decimal=5,
                        err_msg="Clusters are not centered on hypercube vertices",
                    )
                else:
                    with pytest.raises(AssertionError):
                        assert_array_almost_equal(
                            np.abs(centroid) / class_sep,
                            np.ones(n_informative),
                            decimal=5,
                            err_msg=(
                                "Clusters should not be centered on hypercube vertices"
                            ),
                        )

    with pytest.raises(ValueError):
        make(n_features=2, n_informative=2, n_classes=5, n_clusters_per_class=1)
    with pytest.raises(ValueError):
        make(n_features=2, n_informative=2, n_classes=3, n_clusters_per_class=2)


def test_make_classification_return_x_y():
    # [FUNCTION DOC]
    # Verifies that return_X_y=False returns a Bunch with DESCR (equal to the function
    # docstring), parameters (equal to the passed kwargs), feature_info, X and y, with X
    # and y identical to the return_X_y=True output, and that feature_info contains the
    # right number of informative, redundant and repeated entries.
    """
    Test that make_classification returns a Bunch when return_X_y is False.

    Also that bunch.X is the same as X
    """

    kwargs = {
        "n_samples": 100,
        "n_features": 20,
        "n_informative": 5,
        "n_redundant": 1,
        "n_repeated": 1,
        "n_classes": 3,
        "n_clusters_per_class": 2,
        "weights": None,
        "flip_y": 0.01,
        "class_sep": 1.0,
        "hypercube": True,
        "shift": 0.0,
        "scale": 1.0,
        "shuffle": True,
        "random_state": 42,
        "return_X_y": True,
    }

    X, y = make_classification(**kwargs)

    kwargs["return_X_y"] = False
    bunch = make_classification(**kwargs)

    assert (
        hasattr(bunch, "DESCR")
        and hasattr(bunch, "parameters")
        and hasattr(bunch, "feature_info")
        and hasattr(bunch, "X")
        and hasattr(bunch, "y")
    )

    def count(str_):
        # [FUNCTION DOC]
        # HELPER: Counts occurrences of str_ in bunch.feature_info.
        return bunch.feature_info.count(str_)

    assert np.array_equal(X, bunch.X)
    assert np.array_equal(y, bunch.y)
    assert bunch.DESCR == make_classification.__doc__
    assert bunch.parameters == kwargs
    assert count("informative") == kwargs["n_informative"]
    assert count("redundant") == kwargs["n_redundant"]
    assert count("repeated") == kwargs["n_repeated"]


@pytest.mark.parametrize(
    "weights, err_type, err_msg",
    [
        ([], ValueError, "Weights specified but incompatible with number of classes."),
        (
            [0.25, 0.75, 0.1],
            ValueError,
            "Weights specified but incompatible with number of classes.",
        ),
        (
            np.array([]),
            ValueError,
            "Weights specified but incompatible with number of classes.",
        ),
        (
            np.array([0.25, 0.75, 0.1]),
            ValueError,
            "Weights specified but incompatible with number of classes.",
        ),
        (
            np.random.random(3),
            ValueError,
            "Weights specified but incompatible with number of classes.",
        ),
    ],
)
def test_make_classification_weights_type(weights, err_type, err_msg):
    # [FUNCTION DOC]
    # Parametrized over weight lists/arrays whose length is incompatible with the
    # default 2 classes; verifies ValueError with the incompatibility message.
    with pytest.raises(err_type, match=err_msg):
        make_classification(weights=weights)


@pytest.mark.parametrize("kwargs", [{}, {"n_classes": 3, "n_informative": 3}])
def test_make_classification_weights_array_or_list_ok(kwargs):
    # [FUNCTION DOC]
    # Verifies weights given as a list or a numpy array produce identical X and y, with
    # the default and a 3-class configuration (weights of length n_classes - 1).
    X1, y1 = make_classification(weights=[0.1, 0.9], random_state=0, **kwargs)
    X2, y2 = make_classification(weights=np.array([0.1, 0.9]), random_state=0, **kwargs)
    assert_almost_equal(X1, X2)
    assert_almost_equal(y1, y2)


def test_make_multilabel_classification_return_sequences():
    # [FUNCTION DOC]
    # Verifies with return_indicator=False that labels are sequences, with minimum
    # length 0 when allow_unlabeled else 1, at most 3 labels, and max label index 2 when
    # unlabeled samples are not allowed.
    for allow_unlabeled, min_length in zip((True, False), (0, 1)):
        X, Y = make_multilabel_classification(
            n_samples=100,
            n_features=20,
            n_classes=3,
            random_state=0,
            return_indicator=False,
            allow_unlabeled=allow_unlabeled,
        )
        assert X.shape == (100, 20), "X shape mismatch"
        if not allow_unlabeled:
            assert max([max(y) for y in Y]) == 2
        assert min([len(y) for y in Y]) == min_length
        assert max([len(y) for y in Y]) <= 3


def test_make_multilabel_classification_return_indicator():
    # [FUNCTION DOC]
    # Verifies the dense indicator output shape (25, 3) and positive column sums for
    # both allow_unlabeled values; then with return_distributions=True verifies
    # identical X, Y and the class prior (sums to 1, shape 3) and per-class word
    # distributions (20 x 3, columns sum to 1).
    # KNOWN ISSUE: the column-sum check against min_length does not test per-sample
    # label counts, and the final call reuses allow_unlabeled leaked from the loop
    # (False).
    for allow_unlabeled, min_length in zip((True, False), (0, 1)):
        X, Y = make_multilabel_classification(
            n_samples=25,
            n_features=20,
            n_classes=3,
            random_state=0,
            allow_unlabeled=allow_unlabeled,
        )
        assert X.shape == (25, 20), "X shape mismatch"
        assert Y.shape == (25, 3), "Y shape mismatch"
        assert np.all(np.sum(Y, axis=0) > min_length)

    # Also test return_distributions and return_indicator with True
    X2, Y2, p_c, p_w_c = make_multilabel_classification(
        n_samples=25,
        n_features=20,
        n_classes=3,
        random_state=0,
        allow_unlabeled=allow_unlabeled,
        return_distributions=True,
    )

    assert_array_almost_equal(X, X2)
    assert_array_equal(Y, Y2)
    assert p_c.shape == (3,)
    assert_almost_equal(p_c.sum(), 1)
    assert p_w_c.shape == (20, 3)
    assert_almost_equal(p_w_c.sum(axis=0), [1] * 3)


@pytest.mark.parametrize("allow_unlabeled", [True, False])
@pytest.mark.parametrize("sparse_feature", [True, False])
def test_make_multilabel_classification_return_sparse(allow_unlabeled, sparse_feature):
    # [FUNCTION DOC]
    # Verifies return_indicator="sparse" gives a sparse Y and that X is sparse exactly
    # when sparse=True, for both allow_unlabeled values.
    X, Y = make_multilabel_classification(
        n_samples=25,
        n_features=20,
        n_classes=3,
        random_state=0,
        sparse=sparse_feature,
        return_indicator="sparse",
        allow_unlabeled=allow_unlabeled,
    )
    assert X.shape == (25, 20), "X shape mismatch"
    assert Y.shape == (25, 3), "Y shape mismatch"
    assert sp.issparse(Y)
    assert sp.issparse(X) if sparse_feature else not sp.issparse(X)


def test_make_hastie_10_2():
    # [FUNCTION DOC]
    # Verifies shapes (100, 10) and two distinct labels.
    X, y = make_hastie_10_2(n_samples=100, random_state=0)
    assert X.shape == (100, 10), "X shape mismatch"
    assert y.shape == (100,), "y shape mismatch"
    assert np.unique(y).shape == (2,), "Unexpected number of classes"


def test_make_regression(global_random_seed):
    # [FUNCTION DOC]
    # Verifies shapes, that exactly 3 coefficients are nonzero, and that the residual of
    # y minus X times coef has standard deviation about 1 (the noise level); also that
    # n_features=1 works with the default n_informative larger than n_features.
    X, y, c = make_regression(
        n_samples=200,
        n_features=10,
        n_informative=3,
        effective_rank=5,
        coef=True,
        bias=0.0,
        noise=1.0,
        random_state=global_random_seed,
    )

    assert X.shape == (200, 10), "X shape mismatch"
    assert y.shape == (200,), "y shape mismatch"
    assert c.shape == (10,), "coef shape mismatch"
    assert sum(c != 0.0) == 3, "Unexpected number of informative features"

    # Test that y ~= np.dot(X, c) + bias + N(0, 1.0).
    assert_almost_equal(np.std(y - np.dot(X, c)), 1.0, decimal=1)

    # Test with small number of features.
    X, y = make_regression(n_samples=100, n_features=1)  # n_informative=3
    assert X.shape == (100, 1)


def test_make_regression_multitarget(global_random_seed):
    # [FUNCTION DOC]
    # Same as test_make_regression for 3 targets: shapes, 3 nonzero coefficients per
    # target and unit noise standard deviation.
    X, y, c = make_regression(
        n_samples=100,
        n_features=10,
        n_informative=3,
        n_targets=3,
        coef=True,
        noise=1.0,
        random_state=global_random_seed,
    )

    assert X.shape == (100, 10), "X shape mismatch"
    assert y.shape == (100, 3), "y shape mismatch"
    assert c.shape == (10, 3), "coef shape mismatch"
    assert_array_equal(sum(c != 0.0), 3, "Unexpected number of informative features")

    # Test that y ~= np.dot(X, c) + bias + N(0, 1.0)
    assert_almost_equal(np.std(y - np.dot(X, c)), 1.0, decimal=1)


def test_make_blobs(global_random_seed):
    # [FUNCTION DOC]
    # Verifies shapes, three labels, and that each blob's spread around its given center
    # approximates its given standard deviation.
    cluster_stds = np.array([0.05, 0.2, 0.4])
    cluster_centers = np.array([[0.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    X, y = make_blobs(
        random_state=global_random_seed,
        n_samples=50,
        n_features=2,
        centers=cluster_centers,
        cluster_std=cluster_stds,
    )

    assert X.shape == (50, 2), "X shape mismatch"
    assert y.shape == (50,), "y shape mismatch"
    assert np.unique(y).shape == (3,), "Unexpected number of blobs"
    for i, (ctr, std) in enumerate(zip(cluster_centers, cluster_stds)):
        assert_almost_equal((X[y == i] - ctr).std(), std, 1, "Unexpected std")


def test_make_blobs_n_samples_list():
    # [FUNCTION DOC]
    # Verifies a list n_samples yields that exact count per blob.
    n_samples = [50, 30, 20]
    X, y = make_blobs(n_samples=n_samples, n_features=2, random_state=0)

    assert X.shape == (sum(n_samples), 2), "X shape mismatch"
    assert all(np.bincount(y, minlength=len(n_samples)) == n_samples), (
        "Incorrect number of samples per blob"
    )


def test_make_blobs_n_samples_list_with_centers(global_random_seed):
    # [FUNCTION DOC]
    # Verifies per-blob counts and per-blob standard deviations with explicit centers.
    n_samples = [20, 20, 20]
    centers = np.array([[0.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    cluster_stds = np.array([0.05, 0.2, 0.4])
    X, y = make_blobs(
        n_samples=n_samples,
        centers=centers,
        cluster_std=cluster_stds,
        random_state=global_random_seed,
    )

    assert X.shape == (sum(n_samples), 2), "X shape mismatch"
    assert all(np.bincount(y, minlength=len(n_samples)) == n_samples), (
        "Incorrect number of samples per blob"
    )
    for i, (ctr, std) in enumerate(zip(centers, cluster_stds)):
        assert_almost_equal((X[y == i] - ctr).std(), std, 1, "Unexpected std")


@pytest.mark.parametrize(
    "n_samples", [[5, 3, 0], np.array([5, 3, 0]), tuple([5, 3, 0])]
)
def test_make_blobs_n_samples_centers_none(n_samples):
    # [FUNCTION DOC]
    # Parametrized over list/array/tuple n_samples including a zero; verifies counts per
    # blob when centers is None (centers generated from len(n_samples)).
    centers = None
    X, y = make_blobs(n_samples=n_samples, centers=centers, random_state=0)

    assert X.shape == (sum(n_samples), 2), "X shape mismatch"
    assert all(np.bincount(y, minlength=len(n_samples)) == n_samples), (
        "Incorrect number of samples per blob"
    )


def test_make_blobs_return_centers():
    # [FUNCTION DOC]
    # Verifies return_centers=True returns centers of shape (n_blobs, n_features).
    n_samples = [10, 20]
    n_features = 3
    X, y, centers = make_blobs(
        n_samples=n_samples, n_features=n_features, return_centers=True, random_state=0
    )

    assert centers.shape == (len(n_samples), n_features)


def test_make_blobs_error():
    # [FUNCTION DOC]
    # Verifies ValueError messages when n_samples length mismatches centers, cluster_std
    # length mismatches centers, and centers is an int while n_samples is a list.
    n_samples = [20, 20, 20]
    centers = np.array([[0.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    cluster_stds = np.array([0.05, 0.2, 0.4])
    wrong_centers_msg = re.escape(
        "Length of `n_samples` not consistent with number of centers. "
        f"Got n_samples = {n_samples} and centers = {centers[:-1]}"
    )
    with pytest.raises(ValueError, match=wrong_centers_msg):
        make_blobs(n_samples, centers=centers[:-1])
    wrong_std_msg = re.escape(
        "Length of `clusters_std` not consistent with number of centers. "
        f"Got centers = {centers} and cluster_std = {cluster_stds[:-1]}"
    )
    with pytest.raises(ValueError, match=wrong_std_msg):
        make_blobs(n_samples, centers=centers, cluster_std=cluster_stds[:-1])
    wrong_type_msg = "Parameter `centers` must be array-like. Got {!r} instead".format(
        3
    )
    with pytest.raises(ValueError, match=wrong_type_msg):
        make_blobs(n_samples, centers=3)


def test_make_friedman1(global_random_seed):
    # [FUNCTION DOC]
    # Verifies shapes and that noise-free y equals the Friedman #1 formula on the first
    # five features.
    X, y = make_friedman1(
        n_samples=5, n_features=10, noise=0.0, random_state=global_random_seed
    )

    assert X.shape == (5, 10), "X shape mismatch"
    assert y.shape == (5,), "y shape mismatch"

    assert_array_almost_equal(
        y,
        10 * np.sin(np.pi * X[:, 0] * X[:, 1])
        + 20 * (X[:, 2] - 0.5) ** 2
        + 10 * X[:, 3]
        + 5 * X[:, 4],
    )


def test_make_friedman2(global_random_seed):
    # [FUNCTION DOC]
    # Verifies shapes and that noise-free y equals the Friedman #2 formula.
    X, y = make_friedman2(n_samples=5, noise=0.0, random_state=global_random_seed)

    assert X.shape == (5, 4), "X shape mismatch"
    assert y.shape == (5,), "y shape mismatch"

    assert_array_almost_equal(
        y, (X[:, 0] ** 2 + (X[:, 1] * X[:, 2] - 1 / (X[:, 1] * X[:, 3])) ** 2) ** 0.5
    )


def test_make_friedman3(global_random_seed):
    # [FUNCTION DOC]
    # Verifies shapes and that noise-free y equals the Friedman #3 formula.
    X, y = make_friedman3(n_samples=5, noise=0.0, random_state=global_random_seed)

    assert X.shape == (5, 4), "X shape mismatch"
    assert y.shape == (5,), "y shape mismatch"

    assert_array_almost_equal(
        y, np.arctan((X[:, 1] * X[:, 2] - 1 / (X[:, 1] * X[:, 3])) / X[:, 0])
    )


def test_make_low_rank_matrix():
    # [FUNCTION DOC]
    # Verifies shape and that the singular values sum to roughly the effective rank 5.
    # KNOWN ISSUE: the check is one-sided (sum minus 5 below 0.1) so a much smaller sum
    # would also pass; the message speaks about rank although it tests the singular
    # value sum.
    X = make_low_rank_matrix(
        n_samples=50,
        n_features=25,
        effective_rank=5,
        tail_strength=0.01,
        random_state=0,
    )

    assert X.shape == (50, 25), "X shape mismatch"

    from numpy.linalg import svd

    u, s, v = svd(X)
    assert sum(s) - 5 < 0.1, "X rank is not approximately 5"


def test_make_sparse_coded_signal(global_random_seed):
    # [FUNCTION DOC]
    # Verifies shapes of Y, D and X, exactly 3 nonzeros per code row, Y equals X times D
    # and the dictionary atoms (rows of D) have unit norm.
    Y, D, X = make_sparse_coded_signal(
        n_samples=5,
        n_components=8,
        n_features=10,
        n_nonzero_coefs=3,
        random_state=global_random_seed,
    )
    assert Y.shape == (5, 10), "Y shape mismatch"
    assert D.shape == (8, 10), "D shape mismatch"
    assert X.shape == (5, 8), "X shape mismatch"
    for row in X:
        assert len(np.flatnonzero(row)) == 3, "Non-zero coefs mismatch"
    assert_allclose(Y, X @ D)
    assert_allclose(np.sqrt((D**2).sum(axis=1)), np.ones(D.shape[0]))


def test_make_sparse_uncorrelated():
    # [FUNCTION DOC]
    # Verifies output shapes only.
    X, y = make_sparse_uncorrelated(n_samples=5, n_features=10, random_state=0)

    assert X.shape == (5, 10), "X shape mismatch"
    assert y.shape == (5,), "y shape mismatch"


def test_make_spd_matrix(global_random_seed):
    # [FUNCTION DOC]
    # Verifies a square symmetric matrix with all eigenvalues positive.
    X = make_spd_matrix(n_dim=5, random_state=global_random_seed)

    assert X.shape == (5, 5), "X shape mismatch"
    assert_array_almost_equal(X, X.T)

    from numpy.linalg import eig

    eigenvalues, _ = eig(X)
    assert np.all(eigenvalues > 0), "X is not positive-definite"


@pytest.mark.parametrize("norm_diag", [True, False])
@pytest.mark.parametrize(
    "sparse_format", [None, "bsr", "coo", "csc", "csr", "dia", "dok", "lil"]
)
def test_make_sparse_spd_matrix(norm_diag, sparse_format, global_random_seed):
    # [FUNCTION DOC]
    # Parametrized over norm_diag and sparse_format (None or each scipy format).
    # Verifies shape, dense vs sparse type and format, symmetry, positive eigenvalues
    # (computed on the dense form) and unit diagonal when norm_diag.
    n_dim = 5
    X = make_sparse_spd_matrix(
        n_dim=n_dim,
        norm_diag=norm_diag,
        sparse_format=sparse_format,
        random_state=global_random_seed,
    )

    assert X.shape == (n_dim, n_dim), "X shape mismatch"
    if sparse_format is None:
        assert not sp.issparse(X)
        assert_allclose(X, X.T)
        Xarr = X
    else:
        assert sp.issparse(X) and X.format == sparse_format
        assert_allclose_dense_sparse(X, X.T)
        Xarr = X.toarray()

    from numpy.linalg import eig

    # Do not use scipy.sparse.linalg.eigs because it cannot find all eigenvalues
    eigenvalues, _ = eig(Xarr)
    assert np.all(eigenvalues > 0), "X is not positive-definite"

    if norm_diag:
        # Check that leading diagonal elements are 1
        assert_array_almost_equal(Xarr.diagonal(), np.ones(n_dim))


@pytest.mark.parametrize("hole", [False, True])
def test_make_swiss_roll(global_random_seed, hole):
    # [FUNCTION DOC]
    # For hole False/True verifies shapes and that the first and third coordinates equal
    # t times cos(t) and t times sin(t) with no noise.
    X, t = make_swiss_roll(
        n_samples=5, noise=0.0, random_state=global_random_seed, hole=hole
    )

    assert X.shape == (5, 3)
    assert t.shape == (5,)
    assert_array_almost_equal(X[:, 0], t * np.cos(t))
    assert_array_almost_equal(X[:, 2], t * np.sin(t))


def test_make_s_curve(global_random_seed):
    # [FUNCTION DOC]
    # Verifies shapes and the S-curve coordinate formulas in terms of t with no noise.
    X, t = make_s_curve(n_samples=5, noise=0.0, random_state=global_random_seed)

    assert X.shape == (5, 3), "X shape mismatch"
    assert t.shape == (5,), "t shape mismatch"
    assert_array_almost_equal(X[:, 0], np.sin(t))
    assert_array_almost_equal(X[:, 2], np.sign(t) * (np.cos(t) - 1))


def test_make_biclusters():
    # [FUNCTION DOC]
    # Verifies shapes of the data and the row/column indicator arrays, finiteness and
    # reproducibility for a fixed random_state.
    X, rows, cols = make_biclusters(
        shape=(100, 100), n_clusters=4, shuffle=True, random_state=0
    )
    assert X.shape == (100, 100), "X shape mismatch"
    assert rows.shape == (4, 100), "rows shape mismatch"
    assert cols.shape == (
        4,
        100,
    ), "columns shape mismatch"
    assert_all_finite(X)
    assert_all_finite(rows)
    assert_all_finite(cols)

    X2, _, _ = make_biclusters(
        shape=(100, 100), n_clusters=4, shuffle=True, random_state=0
    )
    assert_array_almost_equal(X, X2)


def test_make_checkerboard():
    # [FUNCTION DOC]
    # Verifies shapes for n_clusters=(20, 5) (indicator arrays have 100 biclusters),
    # then finiteness and reproducibility with n_clusters=2.
    X, rows, cols = make_checkerboard(
        shape=(100, 100), n_clusters=(20, 5), shuffle=True, random_state=0
    )
    assert X.shape == (100, 100), "X shape mismatch"
    assert rows.shape == (100, 100), "rows shape mismatch"
    assert cols.shape == (
        100,
        100,
    ), "columns shape mismatch"

    X, rows, cols = make_checkerboard(
        shape=(100, 100), n_clusters=2, shuffle=True, random_state=0
    )
    assert_all_finite(X)
    assert_all_finite(rows)
    assert_all_finite(cols)

    X1, _, _ = make_checkerboard(
        shape=(100, 100), n_clusters=2, shuffle=True, random_state=0
    )
    X2, _, _ = make_checkerboard(
        shape=(100, 100), n_clusters=2, shuffle=True, random_state=0
    )
    assert_array_almost_equal(X1, X2)


def test_make_moons(global_random_seed):
    # [FUNCTION DOC]
    # Verifies each unshuffled point lies on the unit circle centered at (0, 0) for
    # label 0 or (1, 0.5) for label 1.
    X, y = make_moons(3, shuffle=False, random_state=global_random_seed)
    for x, label in zip(X, y):
        center = [0.0, 0.0] if label == 0 else [1.0, 0.5]
        dist_sqr = ((x - center) ** 2).sum()
        assert_almost_equal(
            dist_sqr, 1.0, err_msg="Point is not on expected unit circle"
        )


def test_make_moons_unbalanced():
    # [FUNCTION DOC]
    # Verifies tuple n_samples gives 7 and 5 samples per moon with correct shapes, and a
    # one-element tuple raises ValueError.
    X, y = make_moons(n_samples=(7, 5))
    assert np.sum(y == 0) == 7 and np.sum(y == 1) == 5, (
        "Number of samples in a moon is wrong"
    )
    assert X.shape == (12, 2), "X shape mismatch"
    assert y.shape == (12,), "y shape mismatch"

    with pytest.raises(
        ValueError,
        match=r"`n_samples` can be either an int or a two-element tuple.",
    ):
        make_moons(n_samples=(10,))


def test_make_circles():
    # [FUNCTION DOC]
    # For odd and even n_samples verifies shapes, that outer points (label 0) have
    # squared radius 1 and inner points squared radius factor squared, and the per-
    # circle counts (the outer circle gets the smaller half when odd).
    # KNOWN ISSUE: the expected-distance line is duplicated (harmless dead statement).
    factor = 0.3

    for n_samples, n_outer, n_inner in [(7, 3, 4), (8, 4, 4)]:
        # Testing odd and even case, because in the past make_circles always
        # created an even number of samples.
        X, y = make_circles(n_samples, shuffle=False, noise=None, factor=factor)
        assert X.shape == (n_samples, 2), "X shape mismatch"
        assert y.shape == (n_samples,), "y shape mismatch"
        center = [0.0, 0.0]
        for x, label in zip(X, y):
            dist_sqr = ((x - center) ** 2).sum()
            dist_exp = 1.0 if label == 0 else factor**2
            dist_exp = 1.0 if label == 0 else factor**2
            assert_almost_equal(
                dist_sqr, dist_exp, err_msg="Point is not on expected circle"
            )

        assert X[y == 0].shape == (
            n_outer,
            2,
        ), "Samples not correctly distributed across circles."
        assert X[y == 1].shape == (
            n_inner,
            2,
        ), "Samples not correctly distributed across circles."


def test_make_circles_unbalanced():
    # [FUNCTION DOC]
    # Verifies tuple n_samples (2, 8) gives 2 samples with label 0 and 8 with label 1,
    # total shapes, and ValueError for a one-element tuple.
    # KNOWN ISSUE: the assertion messages call label 0 the inner circle, but label 0 is
    # the outer circle (see test_make_circles).
    X, y = make_circles(n_samples=(2, 8))

    assert np.sum(y == 0) == 2, "Number of samples in inner circle is wrong"
    assert np.sum(y == 1) == 8, "Number of samples in outer circle is wrong"
    assert X.shape == (10, 2), "X shape mismatch"
    assert y.shape == (10,), "y shape mismatch"

    with pytest.raises(
        ValueError,
        match="When a tuple, n_samples must have exactly two elements.",
    ):
        make_circles(n_samples=(10,))
