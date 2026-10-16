# [MODULE DOC]
# ASV benchmarks for RandomForest, GradientBoosting and HistGradientBoosting
# classifiers.
from sklearn.ensemble import (
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)

from .common import Benchmark, Estimator, Predictor
from .datasets import (
    _20newsgroups_highdim_dataset,
    _20newsgroups_lowdim_dataset,
    _synth_classification_dataset,
)
from .utils import make_gen_classif_scorers


class RandomForestClassifierBenchmark(Predictor, Estimator, Benchmark):
    # [CLASS DOC]
    # RandomForestClassifier benchmark over representation and n_jobs.
    """
    Benchmarks for RandomForestClassifier.
    """

    param_names = ["representation", "n_jobs"]
    params = (["dense", "sparse"], Benchmark.n_jobs_vals)

    def setup_cache(self):
        # [FUNCTION DOC]
        # Re-declares Estimator.setup_cache so ASV runs the cache step for this class.
        super().setup_cache()

    def make_data(self, params):
        # [FUNCTION DOC]
        # Sparse: full 20newsgroups TF-IDF; dense: SVD-reduced 20newsgroups (100 dims).
        representation, n_jobs = params

        if representation == "sparse":
            data = _20newsgroups_highdim_dataset()
        else:
            data = _20newsgroups_lowdim_dataset()

        return data

    def make_estimator(self, params):
        # [FUNCTION DOC]
        # RandomForest with 500 trees (large data) or 100, min_samples_split 10, log2 max
        # features, given n_jobs, seed 0.
        representation, n_jobs = params

        n_estimators = 500 if Benchmark.data_size == "large" else 100

        estimator = RandomForestClassifier(
            n_estimators=n_estimators,
            min_samples_split=10,
            max_features="log2",
            n_jobs=n_jobs,
            random_state=0,
        )

        return estimator

    def make_scorers(self):
        # [FUNCTION DOC]
        # Uses balanced accuracy scorers (make_gen_classif_scorers).
        make_gen_classif_scorers(self)


class GradientBoostingClassifierBenchmark(Predictor, Estimator, Benchmark):
    # [CLASS DOC]
    # GradientBoostingClassifier benchmark over representation.
    """
    Benchmarks for GradientBoostingClassifier.
    """

    param_names = ["representation"]
    params = (["dense", "sparse"],)

    def setup_cache(self):
        # [FUNCTION DOC]
        # Re-declares Estimator.setup_cache so ASV runs the cache step for this class.
        super().setup_cache()

    def make_data(self, params):
        # [FUNCTION DOC]
        # Sparse: full 20newsgroups TF-IDF; dense: SVD-reduced 20newsgroups.
        (representation,) = params

        if representation == "sparse":
            data = _20newsgroups_highdim_dataset()
        else:
            data = _20newsgroups_lowdim_dataset()

        return data

    def make_estimator(self, params):
        # [FUNCTION DOC]
        # GradientBoosting with 100 (large) or 10 stages, log2 max features, subsample 0.5,
        # seed 0.
        (representation,) = params

        n_estimators = 100 if Benchmark.data_size == "large" else 10

        estimator = GradientBoostingClassifier(
            n_estimators=n_estimators,
            max_features="log2",
            subsample=0.5,
            random_state=0,
        )

        return estimator

    def make_scorers(self):
        # [FUNCTION DOC]
        # Uses balanced accuracy scorers (make_gen_classif_scorers).
        make_gen_classif_scorers(self)


class HistGradientBoostingClassifierBenchmark(Predictor, Estimator, Benchmark):
    # [CLASS DOC]
    # HistGradientBoostingClassifier benchmark with no parameters (a single run).
    """
    Benchmarks for HistGradientBoostingClassifier.
    """

    param_names = []
    params = ()

    def setup_cache(self):
        # [FUNCTION DOC]
        # Re-declares Estimator.setup_cache so ASV runs the cache step for this class.
        super().setup_cache()

    def make_data(self, params):
        # [FUNCTION DOC]
        # Synthetic 5-class data with 10000 samples and 100 features.
        data = _synth_classification_dataset(
            n_samples=10000, n_features=100, n_classes=5
        )

        return data

    def make_estimator(self, params):
        # [FUNCTION DOC]
        # HGB with 100 iterations, 15 leaf nodes, no early stopping, seed 0.
        estimator = HistGradientBoostingClassifier(
            max_iter=100, max_leaf_nodes=15, early_stopping=False, random_state=0
        )

        return estimator

    def make_scorers(self):
        # [FUNCTION DOC]
        # Uses balanced accuracy scorers (make_gen_classif_scorers).
        make_gen_classif_scorers(self)
