# [MODULE DOC]
# Script comparing scikit-learn Lasso with glmnet-python Lasso (alpha 0.1): first with
# 1000 features and growing training size (500..10000), then 500 training samples and
# growing feature count (100..2000); prints timing/RMSE/coef error and plots times.
# Side effects: unseeded random data, prints, matplotlib windows; needs glmnet.
# KNOWN ISSUE: n_informative is computed as a float (n_features / 10); make_regression
# validates it as an integer, so current scikit-learn raises an error.
# KNOWN ISSUE: the first plot's x values start at 0 while training sizes start at step.
# KNOWN ISSUE: in the second benchmark n_samples is (i*step)+1000 but only the first 500
# rows are used for training; the comment's description is otherwise accurate.
"""
To run this, you'll need to have installed.

  * glmnet-python
  * scikit-learn (of course)

Does two benchmarks

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

alpha = 0.1
# alpha = 0.01


def rmse(a, b):
    # [FUNCTION DOC]
    # HELPER: Root mean squared error between arrays a and b (same shape). Outputs: float.
    return np.sqrt(np.mean((a - b) ** 2))


def bench(factory, X, Y, X_test, Y_test, ref_coef):
    # [FUNCTION DOC]
    # HELPER: Runs garbage collection, fits factory(alpha=alpha) on (X, Y) timing the fit,
    # prints duration, test RMSE and mean absolute difference between ref_coef and the
    # fitted coef_.
    # Inputs: factory - estimator class accepting alpha; X, Y training data; X_test,
    # Y_test held-out data; ref_coef - true coefficients (n_features,).
    # Outputs: fit duration in seconds (float). Side effects: prints.
    gc.collect()

    # start time
    tstart = time()
    clf = factory(alpha=alpha).fit(X, Y)
    delta = time() - tstart
    # stop time

    print("duration: %0.3fs" % delta)
    print("rmse: %f" % rmse(Y_test, clf.predict(X_test)))
    print("mean coef abs diff: %f" % abs(ref_coef - clf.coef_.ravel()).mean())
    return delta


if __name__ == "__main__":
    # Delayed import of matplotlib.pyplot
    import matplotlib.pyplot as plt
    from glmnet.elastic_net import Lasso as GlmnetLasso

    from sklearn.linear_model import Lasso as ScikitLasso

    scikit_results = []
    glmnet_results = []
    n = 20
    step = 500
    n_features = 1000
    n_informative = n_features / 10
    n_test_samples = 1000
    for i in range(1, n + 1):
        print("==================")
        print("Iteration %s of %s" % (i, n))
        print("==================")

        X, Y, coef_ = make_regression(
            n_samples=(i * step) + n_test_samples,
            n_features=n_features,
            noise=0.1,
            n_informative=n_informative,
            coef=True,
        )

        X_test = X[-n_test_samples:]
        Y_test = Y[-n_test_samples:]
        X = X[: (i * step)]
        Y = Y[: (i * step)]

        print("benchmarking scikit-learn: ")
        scikit_results.append(bench(ScikitLasso, X, Y, X_test, Y_test, coef_))
        print("benchmarking glmnet: ")
        glmnet_results.append(bench(GlmnetLasso, X, Y, X_test, Y_test, coef_))

    plt.clf()
    xx = range(0, n * step, step)
    plt.title("Lasso regression on sample dataset (%d features)" % n_features)
    plt.plot(xx, scikit_results, "b-", label="scikit-learn")
    plt.plot(xx, glmnet_results, "r-", label="glmnet")
    plt.legend()
    plt.xlabel("number of samples to classify")
    plt.ylabel("Time (s)")
    plt.show()

    # now do a benchmark where the number of points is fixed
    # and the variable is the number of features

    scikit_results = []
    glmnet_results = []
    n = 20
    step = 100
    n_samples = 500

    for i in range(1, n + 1):
        print("==================")
        print("Iteration %02d of %02d" % (i, n))
        print("==================")
        n_features = i * step
        n_informative = n_features / 10

        X, Y, coef_ = make_regression(
            n_samples=(i * step) + n_test_samples,
            n_features=n_features,
            noise=0.1,
            n_informative=n_informative,
            coef=True,
        )

        X_test = X[-n_test_samples:]
        Y_test = Y[-n_test_samples:]
        X = X[:n_samples]
        Y = Y[:n_samples]

        print("benchmarking scikit-learn: ")
        scikit_results.append(bench(ScikitLasso, X, Y, X_test, Y_test, coef_))
        print("benchmarking glmnet: ")
        glmnet_results.append(bench(GlmnetLasso, X, Y, X_test, Y_test, coef_))

    xx = np.arange(100, 100 + n * step, step)
    plt.figure("scikit-learn vs. glmnet benchmark results")
    plt.title("Regression in high dimensional spaces (%d samples)" % n_samples)
    plt.plot(xx, scikit_results, "b-", label="scikit-learn")
    plt.plot(xx, glmnet_results, "r-", label="glmnet")
    plt.legend()
    plt.xlabel("number of features")
    plt.ylabel("Time (s)")
    plt.axis("tight")
    plt.show()
