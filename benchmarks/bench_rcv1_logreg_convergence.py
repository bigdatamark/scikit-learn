# [MODULE DOC]
# Top-level benchmark script comparing convergence of logistic regression solvers
# (liblinear primal/dual, sag, newton-cg, lbfgs, SGD log loss, and lightning SVRG/SAG
# when installed and fit_intercept is False) on RCV1 "CCAT" vs rest (labels +/-1).
# For increasing max_iter, records training loss, train/test accuracy and fit time,
# then plots score/loss curves vs time. Train/test split swaps the LYRL2004 split
# (first 23149 rows are the test set).
# Side effects at import: downloads RCV1, fits models, creates a joblib cache in
# the current directory, prints, shows plots.
# KNOWN ISSUE: Memory(cachedir=...) is not accepted by recent joblib. The SGD
# regularization uses n_samples of the full dataset while models train on the
# smaller post-split X, and bench_one's C conversion uses that same global.
# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import gc
import time

import matplotlib.pyplot as plt
import numpy as np
from joblib import Memory

from sklearn.datasets import fetch_rcv1
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.linear_model._sag import get_auto_step_size

try:
    import lightning.classification as lightning_clf
except ImportError:
    lightning_clf = None

m = Memory(cachedir=".", verbose=0)


# compute logistic loss
def get_loss(w, intercept, myX, myy, C):
    # [FUNCTION DOC]
    # HELPER: L2-regularized logistic loss averaged over samples: mean of
    # log(1 + exp(-y * (X w + intercept))) plus ||w||^2 / (2 C n_samples).
    # Inputs: w coefficients (any shape, flattened); intercept scalar or array; myX
    # (n_samples, n_features) dense or sparse; myy labels in {-1, +1}; C float.
    # Output: float. Side effects: prints the data and penalty terms.
    n_samples = myX.shape[0]
    w = w.ravel()
    p = np.mean(np.log(1.0 + np.exp(-myy * (myX.dot(w) + intercept))))
    print("%f + %f" % (p, w.dot(w) / 2.0 / C / n_samples))
    p += w.dot(w) / 2.0 / C / n_samples
    return p


# We use joblib to cache individual fits. Note that we do not pass the dataset
# as argument as the hashing would be too slow, so we assume that the dataset
# never changes.
@m.cache()
def bench_one(name, clf_type, clf_params, n_iter):
    # [FUNCTION DOC]
    # Builds an estimator from class and params, sets max_iter (or n_iter as fallback)
    # and random_state 42, fits on global X, y, and measures fit time. Derives C from
    # alpha for SGD-like models (1/(alpha*n_samples)) or uses clf.C.
    # Inputs: name (cache key only), clf_type, clf_params dict, n_iter.
    # Output: tuple (train_loss, train_score, test_score, duration).
    # Side effects: reads globals X, y, X_test, y_test, n_samples; results cached by
    # joblib without hashing the data.
    clf = clf_type(**clf_params)
    try:
        clf.set_params(max_iter=n_iter, random_state=42)
    except Exception:
        clf.set_params(n_iter=n_iter, random_state=42)

    st = time.time()
    clf.fit(X, y)
    end = time.time()

    try:
        C = 1.0 / clf.alpha / n_samples
    except Exception:
        C = clf.C

    try:
        intercept = clf.intercept_
    except Exception:
        intercept = 0.0

    train_loss = get_loss(clf.coef_, intercept, X, y, C)
    train_score = clf.score(X, y)
    test_score = clf.score(X_test, y_test)
    duration = end - st

    return train_loss, train_score, test_score, duration


def bench(clfs):
    # [FUNCTION DOC]
    # For each 7-tuple (name, estimator, iter_range, and four result lists), runs
    # bench_one for every n_iter and appends the results to the tuple's lists.
    # Output: the same clfs list. Side effects: mutates the result lists, prints.
    for (
        name,
        clf,
        iter_range,
        train_losses,
        train_scores,
        test_scores,
        durations,
    ) in clfs:
        print("training %s" % name)
        clf_type = type(clf)
        clf_params = clf.get_params()

        for n_iter in iter_range:
            gc.collect()

            train_loss, train_score, test_score, duration = bench_one(
                name, clf_type, clf_params, n_iter
            )

            train_losses.append(train_loss)
            train_scores.append(train_score)
            test_scores.append(test_score)
            durations.append(duration)
            print("classifier: %s" % name)
            print("train_loss: %.8f" % train_loss)
            print("train_score: %.8f" % train_score)
            print("test_score: %.8f" % test_score)
            print("time for fit: %.8f seconds" % duration)
            print("")

        print("")
    return clfs


def plot_train_losses(clfs):
    # [FUNCTION DOC]
    # Plots train loss vs fit duration for each classifier in a new figure.
    # Input: clfs list of 7-tuples as filled by bench.
    plt.figure()
    for name, _, _, train_losses, _, _, durations in clfs:
        plt.plot(durations, train_losses, "-o", label=name)
        plt.legend(loc=0)
        plt.xlabel("seconds")
        plt.ylabel("train loss")


def plot_train_scores(clfs):
    # [FUNCTION DOC]
    # Plots train accuracy vs fit duration per classifier; y range fixed to
    # 0.92-0.96. Input: clfs list of 7-tuples.
    plt.figure()
    for name, _, _, _, train_scores, _, durations in clfs:
        plt.plot(durations, train_scores, "-o", label=name)
        plt.legend(loc=0)
        plt.xlabel("seconds")
        plt.ylabel("train score")
        plt.ylim((0.92, 0.96))


def plot_test_scores(clfs):
    # [FUNCTION DOC]
    # Plots test accuracy vs fit duration per classifier; y range fixed to
    # 0.92-0.96. Input: clfs list of 7-tuples.
    plt.figure()
    for name, _, _, _, _, test_scores, durations in clfs:
        plt.plot(durations, test_scores, "-o", label=name)
        plt.legend(loc=0)
        plt.xlabel("seconds")
        plt.ylabel("test score")
        plt.ylim((0.92, 0.96))


def plot_dloss(clfs):
    # [FUNCTION DOC]
    # Takes the smallest final train loss over all classifiers as the reference and
    # plots log10 of the absolute gap to it vs fit duration per classifier.
    # Input: clfs list of 7-tuples. Side effects: new figure.
    plt.figure()
    pobj_final = []
    for name, _, _, train_losses, _, _, durations in clfs:
        pobj_final.append(train_losses[-1])

    indices = np.argsort(pobj_final)
    pobj_best = pobj_final[indices[0]]

    for name, _, _, train_losses, _, _, durations in clfs:
        log_pobj = np.log(abs(np.array(train_losses) - pobj_best)) / np.log(10)

        plt.plot(durations, log_pobj, "-o", label=name)
        plt.legend(loc=0)
        plt.xlabel("seconds")
        plt.ylabel("log(best - train_loss)")


def get_max_squared_sum(X):
    # [FUNCTION DOC]
    # HELPER: Maximum over rows of the sum of squared entries. Input: X dense or
    # sparse (n_samples, n_features). Output: float.
    """Get the maximum row-wise sum of squares"""
    return np.sum(X**2, axis=1).max()


rcv1 = fetch_rcv1()
X = rcv1.data
n_samples, n_features = X.shape

# consider the binary classification problem 'CCAT' vs the rest
ccat_idx = rcv1.target_names.tolist().index("CCAT")
y = rcv1.target.tocsc()[:, ccat_idx].toarray().ravel().astype(np.float64)
y[y == 0] = -1

# parameters
C = 1.0
fit_intercept = True
tol = 1.0e-14

# max_iter range
sgd_iter_range = list(range(1, 121, 10))
newton_iter_range = list(range(1, 25, 3))
lbfgs_iter_range = list(range(1, 242, 12))
liblinear_iter_range = list(range(1, 37, 3))
liblinear_dual_iter_range = list(range(1, 85, 6))
sag_iter_range = list(range(1, 37, 3))

clfs = [
    (
        "LR-liblinear",
        LogisticRegression(
            C=C,
            tol=tol,
            solver="liblinear",
            fit_intercept=fit_intercept,
            intercept_scaling=1,
        ),
        liblinear_iter_range,
        [],
        [],
        [],
        [],
    ),
    (
        "LR-liblinear-dual",
        LogisticRegression(
            C=C,
            tol=tol,
            dual=True,
            solver="liblinear",
            fit_intercept=fit_intercept,
            intercept_scaling=1,
        ),
        liblinear_dual_iter_range,
        [],
        [],
        [],
        [],
    ),
    (
        "LR-SAG",
        LogisticRegression(C=C, tol=tol, solver="sag", fit_intercept=fit_intercept),
        sag_iter_range,
        [],
        [],
        [],
        [],
    ),
    (
        "LR-newton-cg",
        LogisticRegression(
            C=C, tol=tol, solver="newton-cg", fit_intercept=fit_intercept
        ),
        newton_iter_range,
        [],
        [],
        [],
        [],
    ),
    (
        "LR-lbfgs",
        LogisticRegression(C=C, tol=tol, solver="lbfgs", fit_intercept=fit_intercept),
        lbfgs_iter_range,
        [],
        [],
        [],
        [],
    ),
    (
        "SGD",
        SGDClassifier(
            alpha=1.0 / C / n_samples,
            penalty="l2",
            loss="log_loss",
            fit_intercept=fit_intercept,
            verbose=0,
        ),
        sgd_iter_range,
        [],
        [],
        [],
        [],
    ),
]


if lightning_clf is not None and not fit_intercept:
    alpha = 1.0 / C / n_samples
    # compute the same step_size than in LR-sag
    max_squared_sum = get_max_squared_sum(X)
    step_size = get_auto_step_size(max_squared_sum, alpha, "log", fit_intercept)

    clfs.append(
        (
            "Lightning-SVRG",
            lightning_clf.SVRGClassifier(
                alpha=alpha, eta=step_size, tol=tol, loss="log"
            ),
            sag_iter_range,
            [],
            [],
            [],
            [],
        )
    )
    clfs.append(
        (
            "Lightning-SAG",
            lightning_clf.SAGClassifier(
                alpha=alpha, eta=step_size, tol=tol, loss="log"
            ),
            sag_iter_range,
            [],
            [],
            [],
            [],
        )
    )

    # We keep only 200 features, to have a dense dataset,
    # and compare to lightning SAG, which seems incorrect in the sparse case.
    X_csc = X.tocsc()
    nnz_in_each_features = X_csc.indptr[1:] - X_csc.indptr[:-1]
    X = X_csc[:, np.argsort(nnz_in_each_features)[-200:]]
    X = X.toarray()
    print("dataset: %.3f MB" % (X.nbytes / 1e6))


# Split training and testing. Switch train and test subset compared to
# LYRL2004 split, to have a larger training dataset.
n = 23149
X_test = X[:n, :]
y_test = y[:n]
X = X[n:, :]
y = y[n:]

clfs = bench(clfs)

plot_train_scores(clfs)
plot_test_scores(clfs)
plot_train_losses(clfs)
plot_dloss(clfs)
plt.show()
