# [MODULE DOC]
# Gallery example measuring prediction latency of regressors (SGDRegressor,
# RandomForestRegressor, SVR) on synthetic standardized regression data: per-instance
# (atomic) vs. bulk latency boxplots, influence of the number of features on Ridge
# bulk latency (90th percentile), and single-row prediction throughput.
# Side effects: prints, wall-clock timing (results vary per machine), forces garbage
# collection, uses the global numpy RNG via make_regression without a seed, plots.
"""
==================
Prediction Latency
==================

This is an example showing the prediction latency of various scikit-learn
estimators.

The goal is to measure the latency one can expect when doing predictions
either in bulk or atomic (i.e. one by one) mode.

The plots represent the distribution of the prediction latency as a boxplot.

"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import gc
import time
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.utils import shuffle


def _not_in_sphinx():
    # [FUNCTION DOC]
    # HELPER: returns True when the module global __file__ exists (script run, not
    # sphinx).
    # KNOWN ISSUE: defined but never used in this example (dead code).
    # Hack to detect whether we are running by the sphinx builder
    return "__file__" in globals()


# %%
# Benchmark and plot helper functions
# -----------------------------------


def atomic_benchmark_estimator(estimator, X_test, verbose=False):
    # [FUNCTION DOC]
    # Predicts each row of X_test separately (2-D slice of one row) and records the
    # wall-clock time per call. Inputs: estimator - fitted, has predict; X_test - 2-D
    # array;
    # verbose - print min/median/max. Outputs: float array of shape (n_instances,),
    # seconds.
    """Measure runtime prediction of each instance."""
    n_instances = X_test.shape[0]
    runtimes = np.zeros(n_instances, dtype=float)
    for i in range(n_instances):
        instance = X_test[[i], :]
        start = time.time()
        estimator.predict(instance)
        runtimes[i] = time.time() - start
    if verbose:
        print(
            "atomic_benchmark runtimes:",
            min(runtimes),
            np.percentile(runtimes, 50),
            max(runtimes),
        )
    return runtimes


def bulk_benchmark_estimator(estimator, X_test, n_bulk_repeats, verbose):
    # [FUNCTION DOC]
    # Predicts the whole X_test n_bulk_repeats times and records each duration divided
    # by
    # the number of rows (per-instance time). Inputs as atomic_benchmark_estimator plus
    # n_bulk_repeats - int; verbose required here. Outputs: float array
    # (n_bulk_repeats,),
    # seconds per instance.
    """Measure runtime prediction of the whole input."""
    n_instances = X_test.shape[0]
    runtimes = np.zeros(n_bulk_repeats, dtype=float)
    for i in range(n_bulk_repeats):
        start = time.time()
        estimator.predict(X_test)
        runtimes[i] = time.time() - start
    runtimes = np.array(list(map(lambda x: x / float(n_instances), runtimes)))
    if verbose:
        print(
            "bulk_benchmark runtimes:",
            min(runtimes),
            np.percentile(runtimes, 50),
            max(runtimes),
        )
    return runtimes


def benchmark_estimator(estimator, X_test, n_bulk_repeats=30, verbose=False):
    # [FUNCTION DOC]
    # Runs atomic then bulk benchmarks and returns both runtime arrays.
    """
    Measure runtimes of prediction in both atomic and bulk mode.

    Parameters
    ----------
    estimator : already trained estimator supporting `predict()`
    X_test : test input
    n_bulk_repeats : how many times to repeat when evaluating bulk mode

    Returns
    -------
    atomic_runtimes, bulk_runtimes : a pair of `np.array` which contain the
    runtimes in seconds.

    """
    atomic_runtimes = atomic_benchmark_estimator(estimator, X_test, verbose)
    bulk_runtimes = bulk_benchmark_estimator(estimator, X_test, n_bulk_repeats, verbose)
    return atomic_runtimes, bulk_runtimes


def generate_dataset(n_train, n_test, n_features, noise=0.1, verbose=False):
    # [FUNCTION DOC]
    # Creates n_train + n_test samples with make_regression (unseeded), splits with
    # random_state 13, shuffles the training part, standardizes X and y using statistics
    # fitted on training data, and runs gc.collect.
    # Inputs: n_train, n_test, n_features - ints; noise - float; verbose - bool.
    # Outputs: X_train, y_train, X_test, y_test (y as 1-D arrays).
    """Generate a regression dataset with the given parameters."""
    if verbose:
        print("generating dataset...")

    X, y, coef = make_regression(
        n_samples=n_train + n_test, n_features=n_features, noise=noise, coef=True
    )

    random_seed = 13
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=n_train, test_size=n_test, random_state=random_seed
    )
    X_train, y_train = shuffle(X_train, y_train, random_state=random_seed)

    X_scaler = StandardScaler()
    X_train = X_scaler.fit_transform(X_train)
    X_test = X_scaler.transform(X_test)

    y_scaler = StandardScaler()
    y_train = y_scaler.fit_transform(y_train[:, None])[:, 0]
    y_test = y_scaler.transform(y_test[:, None])[:, 0]

    gc.collect()
    if verbose:
        print("ok")
    return X_train, y_train, X_test, y_test


def boxplot_runtimes(runtimes, pred_type, configuration):
    # [FUNCTION DOC]
    # Draws a black boxplot per estimator of runtimes (microseconds), x labels showing
    # name
    # and complexity from each estimator config's complexity_computer, red "+" fliers,
    # title with pred_type and feature count, then shows the figure.
    # Inputs: runtimes - list of arrays; pred_type - label str; configuration -
    # benchmark
    # config dict with "estimators" and "n_features".
    # KNOWN ISSUE: docstring documents a nonexistent cls_names parameter and omits
    # configuration.
    """
    Plot a new `Figure` with boxplots of prediction runtimes.

    Parameters
    ----------
    runtimes : list of `np.array` of latencies in micro-seconds
    cls_names : list of estimator class names that generated the runtimes
    pred_type : 'bulk' or 'atomic'

    """

    fig, ax1 = plt.subplots(figsize=(10, 6))
    bp = plt.boxplot(
        runtimes,
    )

    cls_infos = [
        "%s\n(%d %s)"
        % (
            estimator_conf["name"],
            estimator_conf["complexity_computer"](estimator_conf["instance"]),
            estimator_conf["complexity_label"],
        )
        for estimator_conf in configuration["estimators"]
    ]
    plt.setp(ax1, xticklabels=cls_infos)
    plt.setp(bp["boxes"], color="black")
    plt.setp(bp["whiskers"], color="black")
    plt.setp(bp["fliers"], color="red", marker="+")

    ax1.yaxis.grid(True, linestyle="-", which="major", color="lightgrey", alpha=0.5)

    ax1.set_axisbelow(True)
    ax1.set_title(
        "Prediction Time per Instance - %s, %d feats."
        % (pred_type.capitalize(), configuration["n_features"])
    )
    ax1.set_ylabel("Prediction Time (us)")

    plt.show()


def benchmark(configuration):
    # [FUNCTION DOC]
    # Generates data from configuration, fits each configured estimator instance,
    # benchmarks
    # atomic and bulk latency, converts to microseconds and draws one boxplot per mode.
    # Side effects: fits (mutates) the estimator instances in configuration, prints,
    # plots.
    """Run the whole benchmark."""
    X_train, y_train, X_test, y_test = generate_dataset(
        configuration["n_train"], configuration["n_test"], configuration["n_features"]
    )

    stats = {}
    for estimator_conf in configuration["estimators"]:
        print("Benchmarking", estimator_conf["instance"])
        estimator_conf["instance"].fit(X_train, y_train)
        gc.collect()
        a, b = benchmark_estimator(estimator_conf["instance"], X_test)
        stats[estimator_conf["name"]] = {"atomic": a, "bulk": b}

    cls_names = [
        estimator_conf["name"] for estimator_conf in configuration["estimators"]
    ]
    runtimes = [1e6 * stats[clf_name]["atomic"] for clf_name in cls_names]
    boxplot_runtimes(runtimes, "atomic", configuration)
    runtimes = [1e6 * stats[clf_name]["bulk"] for clf_name in cls_names]
    boxplot_runtimes(runtimes, "bulk (%d)" % configuration["n_test"], configuration)


def n_feature_influence(estimators, n_train, n_test, n_features, percentile):
    # [FUNCTION DOC]
    # For each feature count: generates data, fits each estimator, measures bulk latency
    # over 30 repeats and stores the given percentile in microseconds.
    # Outputs: nested defaultdict estimator name -> {n_features -> latency in us}.
    # Side effects: refits the estimators, prints.
    """
    Estimate influence of the number of features on prediction time.

    Parameters
    ----------

    estimators : dict of (name (str), estimator) to benchmark
    n_train : nber of training instances (int)
    n_test : nber of testing instances (int)
    n_features : list of feature-space dimensionality to test (int)
    percentile : percentile at which to measure the speed (int [0-100])

    Returns:
    --------

    percentiles : dict(estimator_name,
                       dict(n_features, percentile_perf_in_us))

    """
    percentiles = defaultdict(defaultdict)
    for n in n_features:
        print("benchmarking with %d features" % n)
        X_train, y_train, X_test, y_test = generate_dataset(n_train, n_test, n)
        for cls_name, estimator in estimators.items():
            estimator.fit(X_train, y_train)
            gc.collect()
            runtimes = bulk_benchmark_estimator(estimator, X_test, 30, False)
            percentiles[cls_name][n] = 1e6 * np.percentile(runtimes, percentile)
    return percentiles


def plot_n_features_influence(percentiles, percentile):
    # [FUNCTION DOC]
    # Plots percentile latency vs. number of features, one line per estimator, and shows
    # it.
    # KNOWN ISSUE: only three colors are defined; more than three estimators raise
    # IndexError.
    # Lines carry no legend labels, so estimators are not identifiable on the plot.
    fig, ax1 = plt.subplots(figsize=(10, 6))
    colors = ["r", "g", "b"]
    for i, cls_name in enumerate(percentiles.keys()):
        x = np.array(sorted(percentiles[cls_name].keys()))
        y = np.array([percentiles[cls_name][n] for n in x])
        plt.plot(
            x,
            y,
            color=colors[i],
        )
    ax1.yaxis.grid(True, linestyle="-", which="major", color="lightgrey", alpha=0.5)
    ax1.set_axisbelow(True)
    ax1.set_title("Evolution of Prediction Time with #Features")
    ax1.set_xlabel("#Features")
    ax1.set_ylabel("Prediction Time at %d%%-ile (us)" % percentile)
    plt.show()


def benchmark_throughputs(configuration, duration_secs=0.1):
    # [FUNCTION DOC]
    # For each configured estimator: fits it, then repeatedly predicts the first test
    # row
    # for duration_secs seconds of wall-clock time and records predictions per second.
    # Inputs: configuration - benchmark config; duration_secs - float, default 0.1.
    # Outputs: dict estimator name -> throughput (float).
    """benchmark throughput for different estimators."""
    X_train, y_train, X_test, y_test = generate_dataset(
        configuration["n_train"], configuration["n_test"], configuration["n_features"]
    )
    throughputs = dict()
    for estimator_config in configuration["estimators"]:
        estimator_config["instance"].fit(X_train, y_train)
        start_time = time.time()
        n_predictions = 0
        while (time.time() - start_time) < duration_secs:
            estimator_config["instance"].predict(X_test[[0]])
            n_predictions += 1
        throughputs[estimator_config["name"]] = n_predictions / duration_secs
    return throughputs


def plot_benchmark_throughput(throughputs, configuration):
    # [FUNCTION DOC]
    # Bar chart of throughputs per estimator labelled with name and complexity; y limit
    # at
    # 1.2 times the max; shows the figure.
    # KNOWN ISSUE: only three colors; tick positions are computed with linspace assuming
    # offsets that do not match the default centred bars, so labels may be misaligned.
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["r", "g", "b"]
    cls_infos = [
        "%s\n(%d %s)"
        % (
            estimator_conf["name"],
            estimator_conf["complexity_computer"](estimator_conf["instance"]),
            estimator_conf["complexity_label"],
        )
        for estimator_conf in configuration["estimators"]
    ]
    cls_values = [
        throughputs[estimator_conf["name"]]
        for estimator_conf in configuration["estimators"]
    ]
    plt.bar(range(len(throughputs)), cls_values, width=0.5, color=colors)
    ax.set_xticks(np.linspace(0.25, len(throughputs) - 0.75, len(throughputs)))
    ax.set_xticklabels(cls_infos, fontsize=10)
    ymax = max(cls_values) * 1.2
    ax.set_ylim((0, ymax))
    ax.set_ylabel("Throughput (predictions/sec)")
    ax.set_title(
        "Prediction Throughput for different estimators (%d features)"
        % configuration["n_features"]
    )
    plt.show()


# %%
# Benchmark bulk/atomic prediction speed for various regressors
# -------------------------------------------------------------

configuration = {
    "n_train": int(1e3),
    "n_test": int(1e2),
    "n_features": int(1e2),
    "estimators": [
        {
            "name": "Linear Model",
            "instance": SGDRegressor(
                penalty="elasticnet", alpha=0.01, l1_ratio=0.25, tol=1e-4
            ),
            "complexity_label": "non-zero coefficients",
            "complexity_computer": lambda clf: np.count_nonzero(clf.coef_),
        },
        {
            "name": "RandomForest",
            "instance": RandomForestRegressor(),
            "complexity_label": "estimators",
            "complexity_computer": lambda clf: clf.n_estimators,
        },
        {
            "name": "SVR",
            "instance": SVR(kernel="rbf"),
            "complexity_label": "support vectors",
            "complexity_computer": lambda clf: len(clf.support_vectors_),
        },
    ],
}
benchmark(configuration)

# %%
# Benchmark n_features influence on prediction speed
# --------------------------------------------------

percentile = 90
percentiles = n_feature_influence(
    {"ridge": Ridge()},
    configuration["n_train"],
    configuration["n_test"],
    [100, 250, 500],
    percentile,
)
plot_n_features_influence(percentiles, percentile)

# %%
# Benchmark throughput
# --------------------

throughputs = benchmark_throughputs(configuration)
plot_benchmark_throughput(throughputs, configuration)
