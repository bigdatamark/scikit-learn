# [MODULE DOC]
# Benchmark script timing scipy's exact thin SVD against randomized_svd (n_iter 0
# and n_iter 3, rank 50) on low-rank matrices over a 4x4 grid of n_samples and
# n_features in [2, 1000]; plots the timings as 3D surfaces.
# KNOWN ISSUE: the __main__ block calls fig.gca(projection="3d"), removed in
# matplotlib 3.6; and it reshapes timings (ordered samples-major) to
# (n_samples, n_features) while the meshgrid is (n_features, n_samples), so the
# surface axes are transposed.
"""Benchmarks of Singular Value Decomposition (Exact and Approximate)

The data is mostly low rank but is a fat infinite tail.
"""

import gc
from collections import defaultdict
from time import time

import numpy as np
from scipy.linalg import svd

from sklearn.datasets import make_low_rank_matrix
from sklearn.utils.extmath import randomized_svd


def compute_bench(samples_range, features_range, n_iter=3, rank=50):
    # [FUNCTION DOC]
    # For each (n_samples, n_features) pair, generates an unseeded low-rank matrix
    # (effective_rank=rank, tail_strength 0.2) and times scipy svd and randomized_svd
    # with n_iter 0 and the given n_iter, collecting garbage before each run.
    # Inputs: samples_range, features_range int sequences; n_iter (default 3); rank
    # (default 50, also the number of components requested).
    # Output: defaultdict label -> list of seconds in samples-major grid order.
    # Side effects: prints progress; unseeded randomness.
    it = 0

    results = defaultdict(lambda: [])

    max_it = len(samples_range) * len(features_range)
    for n_samples in samples_range:
        for n_features in features_range:
            it += 1
            print("====================")
            print("Iteration %03d of %03d" % (it, max_it))
            print("====================")
            X = make_low_rank_matrix(
                n_samples, n_features, effective_rank=rank, tail_strength=0.2
            )

            gc.collect()
            print("benchmarking scipy svd: ")
            tstart = time()
            svd(X, full_matrices=False)
            results["scipy svd"].append(time() - tstart)

            gc.collect()
            print("benchmarking scikit-learn randomized_svd: n_iter=0")
            tstart = time()
            randomized_svd(X, rank, n_iter=0)
            results["scikit-learn randomized_svd (n_iter=0)"].append(time() - tstart)

            gc.collect()
            print("benchmarking scikit-learn randomized_svd: n_iter=%d " % n_iter)
            tstart = time()
            randomized_svd(X, rank, n_iter=n_iter)
            results["scikit-learn randomized_svd (n_iter=%d)" % n_iter].append(
                time() - tstart
            )

    return results


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import axes3d  # register the 3d projection  # noqa: F401

    samples_range = np.linspace(2, 1000, 4).astype(int)
    features_range = np.linspace(2, 1000, 4).astype(int)
    results = compute_bench(samples_range, features_range)

    label = "scikit-learn singular value decomposition benchmark results"
    fig = plt.figure(label)
    ax = fig.gca(projection="3d")
    for c, (label, timings) in zip("rbg", sorted(results.items())):
        X, Y = np.meshgrid(samples_range, features_range)
        Z = np.asarray(timings).reshape(samples_range.shape[0], features_range.shape[0])
        # plot the actual surface
        ax.plot_surface(X, Y, Z, rstride=8, cstride=8, alpha=0.3, color=c)
        # dummy point plot to stick the legend to since surface plot do not
        # support legends (yet?)
        ax.plot([1], [1], [1], color=c, label=label)

    ax.set_xlabel("n_samples")
    ax.set_ylabel("n_features")
    ax.set_zlabel("Time (s)")
    ax.legend()
    plt.show()
