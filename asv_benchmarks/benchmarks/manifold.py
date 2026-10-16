# [MODULE DOC]
# ASV benchmark for t-SNE fitting.
from sklearn.manifold import TSNE

from .common import Benchmark, Estimator
from .datasets import _digits_dataset


class TSNEBenchmark(Estimator, Benchmark):
    # [CLASS DOC]
    # TSNE benchmark over method (exact, barnes_hut). Not a Predictor/Transformer, so only
    # fit time/memory and KL divergence are tracked.
    """
    Benchmarks for t-SNE.
    """

    param_names = ["method"]
    params = (["exact", "barnes_hut"],)

    def setup_cache(self):
        # [FUNCTION DOC]
        # Re-declares Estimator.setup_cache so ASV runs the cache step for this class.
        super().setup_cache()

    def make_data(self, params):
        # [FUNCTION DOC]
        # Digits dataset, truncated to 500 samples for the exact method, full otherwise.
        (method,) = params

        n_samples = 500 if method == "exact" else None

        return _digits_dataset(n_samples=n_samples)

    def make_estimator(self, params):
        # [FUNCTION DOC]
        # TSNE with the given method and seed 0 (other parameters default).
        (method,) = params

        estimator = TSNE(random_state=0, method=method)

        return estimator

    def make_scorers(self):
        # [FUNCTION DOC]
        # Both scorers return the fitted kl_divergence_ (the "test" score is not computed on
        # validation data).
        self.train_scorer = lambda _, __: self.estimator.kl_divergence_
        self.test_scorer = lambda _, __: self.estimator.kl_divergence_
