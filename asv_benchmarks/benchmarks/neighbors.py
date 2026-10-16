# [MODULE DOC]
# ASV benchmark for KNeighborsClassifier.
from sklearn.neighbors import KNeighborsClassifier

from .common import Benchmark, Estimator, Predictor
from .datasets import _20newsgroups_lowdim_dataset
from .utils import make_gen_classif_scorers


class KNeighborsClassifierBenchmark(Predictor, Estimator, Benchmark):
    # [CLASS DOC]
    # KNeighborsClassifier benchmark over algorithm (brute, kd_tree, ball_tree), dimension
    # (low, high) and n_jobs.
    """
    Benchmarks for KNeighborsClassifier.
    """

    param_names = ["algorithm", "dimension", "n_jobs"]
    params = (["brute", "kd_tree", "ball_tree"], ["low", "high"], Benchmark.n_jobs_vals)

    def setup_cache(self):
        # [FUNCTION DOC]
        # Re-declares Estimator.setup_cache so ASV runs the cache step for this class.
        super().setup_cache()

    def make_data(self, params):
        # [FUNCTION DOC]
        # SVD-reduced 20newsgroups with 10/50 components (small) or 40/200 (large) for low/high
        # dimension.
        algorithm, dimension, n_jobs = params

        if Benchmark.data_size == "large":
            n_components = 40 if dimension == "low" else 200
        else:
            n_components = 10 if dimension == "low" else 50

        data = _20newsgroups_lowdim_dataset(n_components=n_components)

        return data

    def make_estimator(self, params):
        # [FUNCTION DOC]
        # KNeighborsClassifier with the given algorithm and n_jobs.
        algorithm, dimension, n_jobs = params

        estimator = KNeighborsClassifier(algorithm=algorithm, n_jobs=n_jobs)

        return estimator

    def make_scorers(self):
        # [FUNCTION DOC]
        # Uses balanced accuracy scorers (make_gen_classif_scorers).
        make_gen_classif_scorers(self)
