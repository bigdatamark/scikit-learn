# [MODULE DOC]
# Script comparing coordinate-descent Lasso and LassoLars fit times (alpha 0.01, no
# intercept): first 10 features with n_samples from 100 to 1e6 (precomputed Gram),
# then 2000 samples with 500..3000 features (no precompute); plots both curves.
# Side effects: unseeded random data, prints progress, shows a figure.
"""
Benchmarks of Lasso vs LassoLars

First, we fix a training set and increase the number of
samples. Then we plot the computation time as function of
the number of samples.

In the second benchmark, we increase the number of dimensions of the
training set. Then we plot the computation time as function of
the number of dimensions.

In both cases, only 10% of the features are informative.
"""

import gc
from time import time

import numpy as np

from sklearn.datasets import make_regression


def compute_bench(alpha, n_samples, n_features, precompute):
    # [FUNCTION DOC]
    # HELPER: For every (ns, nf) pair in the cartesian product of n_samples and
    # n_features, generates make_regression data with nf // 10 informative features and
    # noise 0.1, normalizes each column to unit L2 norm, then times Lasso and LassoLars
    # fits with the given alpha and precompute.
    # Inputs: alpha float; n_samples, n_features iterables of ints; precompute bool.
    # Outputs: (lasso_times, lars_times) lists of seconds, one entry per pair.
    # Side effects: prints, calls gc.collect.
    # KNOWN ISSUE: Lasso and LassoLars are imported only inside the main block, so calling
    # this function after a plain import raises NameError. The printed iteration total
    # is max(len) rather than the product of lengths.
    lasso_results = []
    lars_lasso_results = []

    it = 0

    for ns in n_samples:
        for nf in n_features:
            it += 1
            print("==================")
            print("Iteration %s of %s" % (it, max(len(n_samples), len(n_features))))
            print("==================")
            n_informative = nf // 10
            X, Y, coef_ = make_regression(
                n_samples=ns,
                n_features=nf,
                n_informative=n_informative,
                noise=0.1,
                coef=True,
            )

            X /= np.sqrt(np.sum(X**2, axis=0))  # Normalize data

            gc.collect()
            print("- benchmarking Lasso")
            clf = Lasso(alpha=alpha, fit_intercept=False, precompute=precompute)
            tstart = time()
            clf.fit(X, Y)
            lasso_results.append(time() - tstart)

            gc.collect()
            print("- benchmarking LassoLars")
            clf = LassoLars(alpha=alpha, fit_intercept=False, precompute=precompute)
            tstart = time()
            clf.fit(X, Y)
            lars_lasso_results.append(time() - tstart)

    return lasso_results, lars_lasso_results


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    from sklearn.linear_model import Lasso, LassoLars

    alpha = 0.01  # regularization parameter

    n_features = 10
    list_n_samples = np.linspace(100, 1000000, 5).astype(int)
    lasso_results, lars_lasso_results = compute_bench(
        alpha, list_n_samples, [n_features], precompute=True
    )

    plt.figure("scikit-learn LASSO benchmark results")
    plt.subplot(211)
    plt.plot(list_n_samples, lasso_results, "b-", label="Lasso")
    plt.plot(list_n_samples, lars_lasso_results, "r-", label="LassoLars")
    plt.title("precomputed Gram matrix, %d features, alpha=%s" % (n_features, alpha))
    plt.legend(loc="upper left")
    plt.xlabel("number of samples")
    plt.ylabel("Time (s)")
    plt.axis("tight")

    n_samples = 2000
    list_n_features = np.linspace(500, 3000, 5).astype(int)
    lasso_results, lars_lasso_results = compute_bench(
        alpha, [n_samples], list_n_features, precompute=False
    )
    plt.subplot(212)
    plt.plot(list_n_features, lasso_results, "b-", label="Lasso")
    plt.plot(list_n_features, lars_lasso_results, "r-", label="LassoLars")
    plt.title("%d samples, alpha=%s" % (n_samples, alpha))
    plt.legend(loc="upper left")
    plt.xlabel("number of features")
    plt.ylabel("Time (s)")
    plt.axis("tight")
    plt.show()
