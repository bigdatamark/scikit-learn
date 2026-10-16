# [MODULE DOC]
# Script comparing IsolationForest decision_function time across branches. "bench"
# subcommand writes <bench_results>/<branch>.csv with fit/predict times over a grid of
# test sizes, feature counts, contamination levels and n_jobs (threading backend);
# "plot" subcommand reads two such CSVs and saves a side-by-side seaborn line plot.
"""
==========================================
IsolationForest prediction benchmark
==========================================
A test of IsolationForest on classical anomaly detection datasets.

The benchmark is run as follows:
1. The dataset is randomly split into a training set and a test set, both
assumed to contain outliers.
2. Isolation Forest is trained on the training set fixed at 1000 samples.
3. The test samples are scored using the trained model at:
    - 1000, 10000, 50000 samples
    - 10, 100, 1000 features
    - 0.01, 0.1, 0.5 contamination
    - 1, 2, 3, 4 n_jobs

We compare the prediction time at the very end.

Here are instructions for running this benchmark to compare runtime against main branch:

1. Build and run on a branch or main, e.g. for a branch named `pr`:

```bash
python bench_isolation_forest_predict.py bench ~/bench_results pr
```

2. Plotting to compare two branches `pr` and `main`:

```bash
python bench_isolation_forest_predict.py plot ~/bench_results pr main results_image.png
```
"""

import argparse
from collections import defaultdict
from pathlib import Path
from time import time

import numpy as np
import pandas as pd
from joblib import parallel_config

from sklearn.ensemble import IsolationForest

print(__doc__)


def get_data(
    n_samples_train, n_samples_test, n_features, contamination=0.1, random_state=0
):
    # [FUNCTION DOC]
    # HELPER: Generates two Gaussian clusters (std 0.3, centered at +2 and -2 in every
    # feature) for train and test, then replaces floor(contamination * n_samples_test)
    # randomly chosen test rows with uniform noise in [-4, 4].
    # Inputs: n_samples_train, n_samples_test, n_features ints; contamination float;
    # random_state int. Outputs: (X_train, X_test) with 2 * n_samples_* rows each.
    # KNOWN ISSUE: because each cluster has n_samples_* rows, the arrays have twice the
    # named sample count and outliers are only drawn from the first half (the +2 cluster).
    """Function based on code from: https://scikit-learn.org/stable/
    auto_examples/ensemble/plot_isolation_forest.html#sphx-glr-auto-
    examples-ensemble-plot-isolation-forest-py
    """
    rng = np.random.RandomState(random_state)

    X = 0.3 * rng.randn(n_samples_train, n_features)
    X_train = np.r_[X + 2, X - 2]

    X = 0.3 * rng.randn(n_samples_test, n_features)
    X_test = np.r_[X + 2, X - 2]

    n_outliers = int(np.floor(contamination * n_samples_test))
    X_outliers = rng.uniform(low=-4, high=4, size=(n_outliers, n_features))

    outlier_idx = rng.choice(np.arange(0, n_samples_test), n_outliers, replace=False)
    X_test[outlier_idx, :] = X_outliers

    return X_train, X_test


def plot(args):
    # [FUNCTION DOC]
    # HELPER: Loads <pr_name>.csv and <main_name>.csv from bench_results, merges them on
    # (n_samples_test, n_jobs), and draws predict time vs n_samples_test per n_jobs for
    # each branch; saves to bench_results/image_path.
    # Inputs: args namespace (bench_results, pr_name, main_name, image_path).
    # Side effects: reads CSVs, prints n_jobs values, writes an image.
    # KNOWN ISSUE: merging only on n_samples_test and n_jobs creates a cartesian product
    # over n_features/contamination rows; seaborn then averages over those pairings.
    import matplotlib.pyplot as plt
    import seaborn as sns

    bench_results = Path(args.bench_results)
    pr_name = args.pr_name
    main_name = args.main_name
    image_path = args.image_path

    results_path = Path(bench_results)
    pr_path = results_path / f"{pr_name}.csv"
    main_path = results_path / f"{main_name}.csv"
    image_path = results_path / image_path

    df_pr = pd.read_csv(pr_path).assign(branch=pr_name)
    df_main = pd.read_csv(main_path).assign(branch=main_name)

    # Merge the two datasets on the common columns
    merged_data = pd.merge(
        df_pr,
        df_main,
        on=["n_samples_test", "n_jobs"],
        suffixes=("_pr", "_main"),
    )

    # Set up the plotting grid
    sns.set(style="whitegrid", context="notebook", font_scale=1.5)

    # Create a figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(18, 6), sharex=True, sharey=True)

    # Plot predict time as a function of n_samples_test with different n_jobs
    print(merged_data["n_jobs"].unique())
    ax = axes[0]
    sns.lineplot(
        data=merged_data,
        x="n_samples_test",
        y="predict_time_pr",
        hue="n_jobs",
        style="n_jobs",
        markers="o",
        ax=ax,
        legend="full",
    )
    ax.set_title(f"Predict Time vs. n_samples_test - {pr_name} branch")
    ax.set_ylabel("Predict Time (Seconds)")
    ax.set_xlabel("n_samples_test")

    ax = axes[1]
    sns.lineplot(
        data=merged_data,
        x="n_samples_test",
        y="predict_time_main",
        hue="n_jobs",
        style="n_jobs",
        markers="X",
        dashes=True,
        ax=ax,
        legend=None,
    )
    ax.set_title(f"Predict Time vs. n_samples_test - {main_name} branch")
    ax.set_ylabel("Predict Time")
    ax.set_xlabel("n_samples_test")

    # Adjust layout and display the plots
    plt.tight_layout()
    fig.savefig(image_path, bbox_inches="tight")
    print(f"Saved image to {image_path}")


def bench(args):
    # [FUNCTION DOC]
    # HELPER: For every combination of n_samples_test (1000, 10000, 50000), n_features
    # (10, 100, 1000), contamination (0.01, 0.1, 0.5) and n_jobs (1-4), generates data,
    # fits IsolationForest on 1000 (x2) training rows and times decision_function under
    # a threading parallel_config with n_jobs. Writes results to <bench_results>/<branch>.csv.
    # Inputs: args namespace (bench_results, branch). Side effects: prints, writes CSV.
    # KNOWN ISSUE: the "clearcache" loop does nothing useful; the model is fit with
    # n_jobs=-1 so only prediction respects the n_jobs grid value.
    results_dir = Path(args.bench_results)
    branch = args.branch
    random_state = 1

    results = defaultdict(list)

    # Loop over all datasets for fitting and scoring the estimator:
    n_samples_train = 1000
    for n_samples_test in [
        1000,
        10000,
        50000,
    ]:
        for n_features in [10, 100, 1000]:
            for contamination in [0.01, 0.1, 0.5]:
                for n_jobs in [1, 2, 3, 4]:
                    X_train, X_test = get_data(
                        n_samples_train,
                        n_samples_test,
                        n_features,
                        contamination,
                        random_state,
                    )

                    print("--- Fitting the IsolationForest estimator...")
                    model = IsolationForest(n_jobs=-1, random_state=random_state)
                    tstart = time()
                    model.fit(X_train)
                    fit_time = time() - tstart

                    # clearcache
                    for _ in range(1000):
                        1 + 1
                    with parallel_config("threading", n_jobs=n_jobs):
                        tstart = time()
                        model.decision_function(X_test)  # the lower, the more abnormal
                        predict_time = time() - tstart

                    results["predict_time"].append(predict_time)
                    results["fit_time"].append(fit_time)
                    results["n_samples_train"].append(n_samples_train)
                    results["n_samples_test"].append(n_samples_test)
                    results["n_features"].append(n_features)
                    results["contamination"].append(contamination)
                    results["n_jobs"].append(n_jobs)

    df = pd.DataFrame(results)
    df.to_csv(results_dir / f"{branch}.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # parse arguments for benchmarking
    subparsers = parser.add_subparsers()
    bench_parser = subparsers.add_parser("bench")
    bench_parser.add_argument("bench_results")
    bench_parser.add_argument("branch")
    bench_parser.set_defaults(func=bench)

    # parse arguments for plotting
    plot_parser = subparsers.add_parser("plot")
    plot_parser.add_argument("bench_results")
    plot_parser.add_argument("pr_name")
    plot_parser.add_argument("main_name")
    plot_parser.add_argument("image_path")
    plot_parser.set_defaults(func=plot)

    # enable the parser and run the relevant function
    args = parser.parse_args()
    args.func(args)
