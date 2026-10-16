# [MODULE DOC]
# HELPER: Shared data-generation utilities for the clustering test suite.
"""
Common utilities for testing clustering.

"""

import numpy as np

###############################################################################
# Generate sample data


def generate_clustered_data(
    seed=0, n_clusters=3, n_features=2, n_samples_per_cluster=20, std=0.4
):
    # [FUNCTION DOC]
    # HELPER: Builds a toy dataset of well separated Gaussian blobs. Uses four fixed
    # centers (rows of a 4x4 0/1/-1 matrix shifted by +10 so data is not centered); for
    # each of the first n_clusters centers, takes its first n_features coordinates and adds
    # std times standard normal noise for n_samples_per_cluster points; blobs are stacked
    # in order.
    # Inputs: seed - RandomState seed (default 0); n_clusters - number of blobs, default 3;
    # n_features - dimensionality, default 2; n_samples_per_cluster - default 20;
    # std - noise scale, default 0.4.
    # Outputs: float array of shape (n_clusters * n_samples_per_cluster, n_features).
    # Exceptions: IndexError if n_clusters > 4; broadcasting ValueError if n_features > 4.
    # Side effects: none (private RandomState).
    prng = np.random.RandomState(seed)

    # the data is voluntary shifted away from zero to check clustering
    # algorithm robustness with regards to non centered data
    means = (
        np.array(
            [
                [1, 1, 1, 0],
                [-1, -1, 0, 1],
                [1, -1, 1, 1],
                [-1, 1, 1, 0],
            ]
        )
        + 10
    )

    X = np.empty((0, n_features))
    for i in range(n_clusters):
        X = np.r_[
            X,
            means[i][:n_features] + std * prng.randn(n_samples_per_cluster, n_features),
        ]
    return X
