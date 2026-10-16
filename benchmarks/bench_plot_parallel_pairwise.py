# [MODULE DOC]
# Benchmark script comparing single-core (n_jobs=1) versus all-cores (n_jobs=-1)
# runtime of pairwise_distances (euclidean) and pairwise_kernels (rbf) for 1000 to
# 5000 random samples with 300 features. Runs at import time and shows plots.
# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import time

import matplotlib.pyplot as plt

from sklearn.metrics.pairwise import pairwise_distances, pairwise_kernels
from sklearn.utils import check_random_state


def plot(func):
    # [FUNCTION DOC]
    # Times func on random data (seed 0) for each sample size with n_jobs=1 and -1 and
    # plots both curves in a new figure named after func.
    # Input: func - callable accepting (X, n_jobs). Output: None.
    # Side effects: creates a matplotlib figure.
    random_state = check_random_state(0)
    one_core = []
    multi_core = []
    sample_sizes = range(1000, 6000, 1000)

    for n_samples in sample_sizes:
        X = random_state.rand(n_samples, 300)

        start = time.time()
        func(X, n_jobs=1)
        one_core.append(time.time() - start)

        start = time.time()
        func(X, n_jobs=-1)
        multi_core.append(time.time() - start)

    plt.figure("scikit-learn parallel %s benchmark results" % func.__name__)
    plt.plot(sample_sizes, one_core, label="one core")
    plt.plot(sample_sizes, multi_core, label="multi core")
    plt.xlabel("n_samples")
    plt.ylabel("Time (s)")
    plt.title("Parallel %s" % func.__name__)
    plt.legend()


def euclidean_distances(X, n_jobs):
    # [FUNCTION DOC]
    # HELPER: Euclidean pairwise distance matrix of X with itself, n_jobs workers.
    # Output: array (n_samples, n_samples).
    return pairwise_distances(X, metric="euclidean", n_jobs=n_jobs)


def rbf_kernels(X, n_jobs):
    # [FUNCTION DOC]
    # HELPER: RBF kernel matrix (gamma 0.1) of X with itself using n_jobs workers.
    # Output: array (n_samples, n_samples).
    return pairwise_kernels(X, metric="rbf", n_jobs=n_jobs, gamma=0.1)


plot(euclidean_distances)
plot(rbf_kernels)
plt.show()
