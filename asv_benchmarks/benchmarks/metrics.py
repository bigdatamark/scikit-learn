# [MODULE DOC]
# ASV benchmark for sklearn.metrics.pairwise.pairwise_distances.
from sklearn.metrics.pairwise import pairwise_distances

from .common import Benchmark
from .datasets import _random_dataset


class PairwiseDistancesBenchmark(Benchmark):
    # [CLASS DOC]
    # Pairwise distance benchmark over representation, metric (cosine, euclidean,
    # manhattan, correlation) and n_jobs. Uses only Benchmark settings (no estimator).
    """
    Benchmarks for pairwise distances.
    """

    param_names = ["representation", "metric", "n_jobs"]
    params = (
        ["dense", "sparse"],
        ["cosine", "euclidean", "manhattan", "correlation"],
        Benchmark.n_jobs_vals,
    )

    def setup(self, *params):
        # [FUNCTION DOC]
        # ASV setup: raises NotImplementedError (skip) for sparse + correlation; picks
        # n_samples (4000/12000 small, 8000/24000 large; lower for manhattan/correlation),
        # loads a random 1000-feature dataset and stores X and pdist_params (metric, n_jobs).
        # Side effects: sets self.X, X_val, y, y_val, pdist_params.
        representation, metric, n_jobs = params

        if representation == "sparse" and metric == "correlation":
            raise NotImplementedError

        if Benchmark.data_size == "large":
            if metric in ("manhattan", "correlation"):
                n_samples = 8000
            else:
                n_samples = 24000
        else:
            if metric in ("manhattan", "correlation"):
                n_samples = 4000
            else:
                n_samples = 12000

        data = _random_dataset(n_samples=n_samples, representation=representation)
        self.X, self.X_val, self.y, self.y_val = data

        self.pdist_params = {"metric": metric, "n_jobs": n_jobs}

    def time_pairwise_distances(self, *args):
        # [FUNCTION DOC]
        # Benchmark: time of computing all pairwise distances within X.
        pairwise_distances(self.X, **self.pdist_params)

    def peakmem_pairwise_distances(self, *args):
        # [FUNCTION DOC]
        # Benchmark: peak memory of the same computation.
        pairwise_distances(self.X, **self.pdist_params)
