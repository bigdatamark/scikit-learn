# [MODULE DOC]
# HELPER: Shared infrastructure for the ASV benchmarks: configuration loading from
# config.json / env vars, on-disk cache of fitted estimators, and abstract mixins
# (Benchmark, Estimator, Predictor, Transformer) that concrete benchmarks combine.
# Side effects at import: get_from_config runs when Benchmark is defined, creating cache
# directories next to this file.
import itertools
import json
import os
import pickle
import timeit
from abc import ABC, abstractmethod
from multiprocessing import cpu_count
from pathlib import Path

import numpy as np


def get_from_config():
    # [FUNCTION DOC]
    # HELPER: Reads config.json (dropping every line containing "//" as a comment) and
    # overrides values with env vars SKLBENCH_PROFILE, SKLBENCH_NJOBS (JSON list),
    # SKLBENCH_SAVE_ESTIMATORS, SKLBENCH_BASE_COMMIT, SKLBENCH_PREDICT, SKLBENCH_TRANSFORM.
    # Empty n_jobs list means 1..cpu_count. save_dir is the first 8 chars of ASV_COMMIT
    # (default "new").
    # Outputs: tuple (profile, n_jobs_vals, save_estimators, save_dir, base_commit,
    # bench_predict, bench_transform).
    # Exceptions: FileNotFoundError / JSONDecodeError / KeyError on bad config.
    # Side effects: creates cache/, cache/estimators/, cache/tmp/ and optionally
    # cache/estimators/<save_dir>.
    # KNOWN ISSUE: env var overrides are raw strings, so e.g. SKLBENCH_PREDICT="False" or
    # "0" is truthy; also any line containing "//" (e.g. a URL) is dropped entirely.
    """Get benchmarks configuration from the config.json file"""
    current_path = Path(__file__).resolve().parent

    config_path = current_path / "config.json"
    with open(config_path, "r") as config_file:
        config_file = "".join(line for line in config_file if line and "//" not in line)
        config = json.loads(config_file)

    profile = os.getenv("SKLBENCH_PROFILE", config["profile"])

    n_jobs_vals_env = os.getenv("SKLBENCH_NJOBS")
    if n_jobs_vals_env:
        n_jobs_vals = json.loads(n_jobs_vals_env)
    else:
        n_jobs_vals = config["n_jobs_vals"]
    if not n_jobs_vals:
        n_jobs_vals = list(range(1, 1 + cpu_count()))

    cache_path = current_path / "cache"
    cache_path.mkdir(exist_ok=True)
    (cache_path / "estimators").mkdir(exist_ok=True)
    (cache_path / "tmp").mkdir(exist_ok=True)

    save_estimators = os.getenv("SKLBENCH_SAVE_ESTIMATORS", config["save_estimators"])
    save_dir = os.getenv("ASV_COMMIT", "new")[:8]

    if save_estimators:
        (cache_path / "estimators" / save_dir).mkdir(exist_ok=True)

    base_commit = os.getenv("SKLBENCH_BASE_COMMIT", config["base_commit"])

    bench_predict = os.getenv("SKLBENCH_PREDICT", config["bench_predict"])
    bench_transform = os.getenv("SKLBENCH_TRANSFORM", config["bench_transform"])

    return (
        profile,
        n_jobs_vals,
        save_estimators,
        save_dir,
        base_commit,
        bench_predict,
        bench_transform,
    )


def get_estimator_path(benchmark, directory, params, save=False):
    # [FUNCTION DOC]
    # HELPER: Builds the pickle path for a fitted estimator.
    # Inputs: benchmark - benchmark instance (its class name is used); directory - subdir
    # name under cache/estimators; params - iterable of parameter values; save - if True
    # use cache/estimators/<directory>, else cache/tmp (directory ignored).
    # Outputs: pathlib.Path "<ClassName>_estimator_<p1>_<p2>...pkl". No side effects.
    """Get path of pickled fitted estimator"""
    path = Path(__file__).resolve().parent / "cache"
    path = (path / "estimators" / directory) if save else (path / "tmp")

    filename = (
        benchmark.__class__.__name__
        + "_estimator_"
        + "_".join(list(map(str, params)))
        + ".pkl"
    )

    return path / filename


def clear_tmp():
    # [FUNCTION DOC]
    # HELPER: Deletes every entry in cache/tmp.
    # Exceptions: fails on subdirectories (unlink) or if the dir does not exist.
    # Side effects: file deletion.
    """Clean the tmp directory"""
    path = Path(__file__).resolve().parent / "cache" / "tmp"
    for child in path.iterdir():
        child.unlink()


class Benchmark(ABC):
    # [CLASS DOC]
    # Base mixin carrying ASV timing settings: wall-clock timer, 1 process, 500 s timeout,
    # and class attributes from get_from_config. The profile sets warmup_time, repeat,
    # number, min_run_count and data_size ("fast"/"regular" -> small, "large_scale" ->
    # large). Subclasses must define params.
    # KNOWN ISSUE: an unknown profile name leaves data_size undefined, so make_data in
    # subclasses fails with AttributeError.
    """Abstract base class for all the benchmarks"""

    timer = timeit.default_timer  # wall time
    processes = 1
    timeout = 500

    (
        profile,
        n_jobs_vals,
        save_estimators,
        save_dir,
        base_commit,
        bench_predict,
        bench_transform,
    ) = get_from_config()

    if profile == "fast":
        warmup_time = 0
        repeat = 1
        number = 1
        min_run_count = 1
        data_size = "small"
    elif profile == "regular":
        warmup_time = 1
        repeat = (3, 100, 30)
        data_size = "small"
    elif profile == "large_scale":
        warmup_time = 1
        repeat = 3
        number = 1
        data_size = "large"

    @property
    @abstractmethod
    def params(self):
        # [FUNCTION DOC]
        # Abstract property: list of parameter value lists; ASV benchmarks the cartesian
        # product and passes each combination as positional args.
        pass


class Estimator(ABC):
    # [CLASS DOC]
    # Mixin defining the estimator benchmark workflow: setup_cache fits and pickles one
    # estimator per parameter combination; setup reloads it per run; provides fit timing,
    # peak memory and train/test score tracking. Subclasses implement make_data,
    # make_estimator and make_scorers (setting train_scorer and test_scorer).
    """Abstract base class for all benchmarks of estimators"""

    @abstractmethod
    def make_data(self, params):
        # [FUNCTION DOC]
        # Abstract: return (X, X_val, y, y_val) for a parameter combination.
        """Return the dataset for a combination of parameters"""
        # The datasets are cached using joblib.Memory so it's fast and can be
        # called for each repeat
        pass

    @abstractmethod
    def make_estimator(self, params):
        # [FUNCTION DOC]
        # Abstract: return an unfitted estimator configured from params.
        """Return an instance of the estimator for a combination of parameters"""
        pass

    def skip(self, params):
        # [FUNCTION DOC]
        # Default skip predicate: never skip. Inputs: params tuple. Outputs: False.
        """Return True if the benchmark should be skipped for these params"""
        return False

    def setup_cache(self):
        # [FUNCTION DOC]
        # ASV setup_cache hook (once per class): clears cache/tmp, then for each non-skipped
        # parameter combination builds data and estimator, fits on (X, y) and pickles it to
        # get_estimator_path(..., save_dir, params, save_estimators).
        # Side effects: file deletion and writes.
        """Pickle a fitted estimator for all combinations of parameters"""
        # This is run once per benchmark class.

        clear_tmp()

        param_grid = list(itertools.product(*self.params))

        for params in param_grid:
            if self.skip(params):
                continue

            estimator = self.make_estimator(params)
            X, _, y, _ = self.make_data(params)

            estimator.fit(X, y)

            est_path = get_estimator_path(
                self, Benchmark.save_dir, params, Benchmark.save_estimators
            )
            with est_path.open(mode="wb") as f:
                pickle.dump(estimator, f)

    def setup(self, *params):
        # [FUNCTION DOC]
        # ASV setup hook per combination/repeat: raises NotImplementedError when skip(params)
        # is True (ASV treats it as skipped); otherwise sets self.X, X_val, y, y_val, loads
        # self.estimator from the pickle, and calls make_scorers.
        """Generate dataset and load the fitted estimator"""
        # This is run once per combination of parameters and per repeat so we
        # need to avoid doing expensive operations there.

        if self.skip(params):
            raise NotImplementedError

        self.X, self.X_val, self.y, self.y_val = self.make_data(params)

        est_path = get_estimator_path(
            self, Benchmark.save_dir, params, Benchmark.save_estimators
        )
        with est_path.open(mode="rb") as f:
            self.estimator = pickle.load(f)

        self.make_scorers()

    def time_fit(self, *args):
        # [FUNCTION DOC]
        # Benchmark: time of refitting the estimator on (X, y). Mutates self.estimator.
        self.estimator.fit(self.X, self.y)

    def peakmem_fit(self, *args):
        # [FUNCTION DOC]
        # Benchmark: peak memory of refitting the estimator on (X, y).
        self.estimator.fit(self.X, self.y)

    def track_train_score(self, *args):
        # [FUNCTION DOC]
        # Tracks train score: train_scorer(y, prediction on X), prediction None if the
        # estimator has no predict. Outputs: float.
        if hasattr(self.estimator, "predict"):
            y_pred = self.estimator.predict(self.X)
        else:
            y_pred = None
        return float(self.train_scorer(self.y, y_pred))

    def track_test_score(self, *args):
        # [FUNCTION DOC]
        # Tracks validation score: test_scorer(y_val, prediction on X_val) as float.
        if hasattr(self.estimator, "predict"):
            y_val_pred = self.estimator.predict(self.X_val)
        else:
            y_val_pred = None
        return float(self.test_scorer(self.y_val, y_val_pred))


class Predictor(ABC):
    # [CLASS DOC]
    # Mixin adding predict benchmarks, defined only if bench_predict is enabled; the
    # regression check against base_commit exists only if base_commit is not None.
    """Abstract base class for benchmarks of estimators implementing predict"""

    if Benchmark.bench_predict:

        def time_predict(self, *args):
            # [FUNCTION DOC]
            # Benchmark: time of predict on training X.
            self.estimator.predict(self.X)

        def peakmem_predict(self, *args):
            # [FUNCTION DOC]
            # Benchmark: peak memory of predict on training X.
            self.estimator.predict(self.X)

        if Benchmark.base_commit is not None:

            def track_same_prediction(self, *args):
                # [FUNCTION DOC]
                # Tracks whether predictions on X_val match (np.allclose) those of the estimator
                # pickled for base_commit with the same params. Outputs: bool.
                # Exceptions: FileNotFoundError if the base estimator was not saved.
                est_path = get_estimator_path(self, Benchmark.base_commit, args, True)
                with est_path.open(mode="rb") as f:
                    estimator_base = pickle.load(f)

                y_val_pred_base = estimator_base.predict(self.X_val)
                y_val_pred = self.estimator.predict(self.X_val)

                return np.allclose(y_val_pred_base, y_val_pred)

    @property
    @abstractmethod
    def params(self):
        # [FUNCTION DOC]
        # Abstract property: benchmark parameter grid.
        pass


class Transformer(ABC):
    # [CLASS DOC]
    # Mixin adding transform benchmarks; same conditional structure as Predictor using
    # bench_transform.
    """Abstract base class for benchmarks of estimators implementing transform"""

    if Benchmark.bench_transform:

        def time_transform(self, *args):
            # [FUNCTION DOC]
            # Benchmark: time of transform on training X.
            self.estimator.transform(self.X)

        def peakmem_transform(self, *args):
            # [FUNCTION DOC]
            # Benchmark: peak memory of transform on training X.
            self.estimator.transform(self.X)

        if Benchmark.base_commit is not None:

            def track_same_transform(self, *args):
                # [FUNCTION DOC]
                # Tracks whether transform(X_val) matches (np.allclose) the base_commit estimator's.
                # Outputs: bool. Exceptions: FileNotFoundError if base estimator missing.
                est_path = get_estimator_path(self, Benchmark.base_commit, args, True)
                with est_path.open(mode="rb") as f:
                    estimator_base = pickle.load(f)

                X_val_t_base = estimator_base.transform(self.X_val)
                X_val_t = self.estimator.transform(self.X_val)

                return np.allclose(X_val_t_base, X_val_t)

    @property
    @abstractmethod
    def params(self):
        # [FUNCTION DOC]
        # Abstract property: benchmark parameter grid.
        pass
