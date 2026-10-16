# [MODULE DOC]
# Benchmark script timing DecisionTreeClassifier and DecisionTreeRegressor fit plus
# predict on random Gaussian data: first with 10 features and growing n_samples
# (20000..110000), then with 100 samples and growing dimension (1000..5500). Results
# are accumulated in module-level lists and plotted.
# Side effects: unseeded randomness, prints, matplotlib window.
# KNOWN ISSUE: the first plot's x axis is range(0, n*step, step) (0..90000) while
# actual sample counts are 20000..110000.
"""
To run this, you'll need to have installed.

  * scikit-learn

Does two benchmarks

First, we fix a training set, increase the number of
samples to classify and plot number of classified samples as a
function of time.

In the second benchmark, we increase the number of dimensions of the
training set, classify a sample and plot the time taken as a function
of the number of dimensions.
"""

import gc
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

# to store the results
scikit_classifier_results = []
scikit_regressor_results = []

mu_second = 0.0 + 10**6  # number of microseconds in a second


def bench_scikit_tree_classifier(X, Y):
    # [FUNCTION DOC]
    # Fits a default DecisionTreeClassifier on (X, Y), predicts X, and appends the
    # elapsed wall-clock seconds to scikit_classifier_results.
    # Inputs: X (n_samples, n_features); Y class labels. Output: None.
    # Side effects: mutates the global result list; gc.collect.
    """Benchmark with scikit-learn decision tree classifier"""

    from sklearn.tree import DecisionTreeClassifier

    gc.collect()

    # start time
    tstart = datetime.now()
    clf = DecisionTreeClassifier()
    clf.fit(X, Y).predict(X)
    delta = datetime.now() - tstart
    # stop time

    scikit_classifier_results.append(delta.seconds + delta.microseconds / mu_second)


def bench_scikit_tree_regressor(X, Y):
    # [FUNCTION DOC]
    # Same as the classifier benchmark with DecisionTreeRegressor; appends to
    # scikit_regressor_results. Inputs: X; Y continuous targets.
    """Benchmark with scikit-learn decision tree regressor"""

    from sklearn.tree import DecisionTreeRegressor

    gc.collect()

    # start time
    tstart = datetime.now()
    clf = DecisionTreeRegressor()
    clf.fit(X, Y).predict(X)
    delta = datetime.now() - tstart
    # stop time

    scikit_regressor_results.append(delta.seconds + delta.microseconds / mu_second)


if __name__ == "__main__":
    print("============================================")
    print("Warning: this is going to take a looong time")
    print("============================================")

    n = 10
    step = 10000
    n_samples = 10000
    dim = 10
    n_classes = 10
    for i in range(n):
        print("============================================")
        print("Entering iteration %s of %s" % (i, n))
        print("============================================")
        n_samples += step
        X = np.random.randn(n_samples, dim)
        Y = np.random.randint(0, n_classes, (n_samples,))
        bench_scikit_tree_classifier(X, Y)
        Y = np.random.randn(n_samples)
        bench_scikit_tree_regressor(X, Y)

    xx = range(0, n * step, step)
    plt.figure("scikit-learn tree benchmark results")
    plt.subplot(211)
    plt.title("Learning with varying number of samples")
    plt.plot(xx, scikit_classifier_results, "g-", label="classification")
    plt.plot(xx, scikit_regressor_results, "r-", label="regression")
    plt.legend(loc="upper left")
    plt.xlabel("number of samples")
    plt.ylabel("Time (s)")

    scikit_classifier_results = []
    scikit_regressor_results = []
    n = 10
    step = 500
    start_dim = 500
    n_classes = 10

    dim = start_dim
    for i in range(0, n):
        print("============================================")
        print("Entering iteration %s of %s" % (i, n))
        print("============================================")
        dim += step
        X = np.random.randn(100, dim)
        Y = np.random.randint(0, n_classes, (100,))
        bench_scikit_tree_classifier(X, Y)
        Y = np.random.randn(100)
        bench_scikit_tree_regressor(X, Y)

    xx = np.arange(start_dim, start_dim + n * step, step)
    plt.subplot(212)
    plt.title("Learning in high dimensional spaces")
    plt.plot(xx, scikit_classifier_results, "g-", label="classification")
    plt.plot(xx, scikit_regressor_results, "r-", label="regression")
    plt.legend(loc="upper left")
    plt.xlabel("number of dimensions")
    plt.ylabel("Time (s)")
    plt.axis("tight")
    plt.show()
