# [MODULE DOC]
# ASV benchmarks for KMeans and MiniBatchKMeans (fit, predict, transform, inertia).
from sklearn.cluster import KMeans, MiniBatchKMeans

from .common import Benchmark, Estimator, Predictor, Transformer
from .datasets import _20newsgroups_highdim_dataset, _blobs_dataset
from .utils import neg_mean_inertia


class KMeansBenchmark(Predictor, Transformer, Estimator, Benchmark):
    # [CLASS DOC]
    # KMeans benchmark over representation (dense blobs / sparse 20newsgroups),
    # algorithm (lloyd, elkan) and init (random, k-means++).
    """
    Benchmarks for KMeans.
    """

    param_names = ["representation", "algorithm", "init"]
    params = (["dense", "sparse"], ["lloyd", "elkan"], ["random", "k-means++"])

    def setup_cache(self):
        # [FUNCTION DOC]
        # Re-declares Estimator.setup_cache so ASV runs the cache step for this class.
        super().setup_cache()

    def make_data(self, params):
        # [FUNCTION DOC]
        # Sparse: first 8000 20newsgroups TF-IDF docs; dense: blobs with 20 clusters.
        # Outputs: (X, X_val, y, y_val) tuple.
        representation, algorithm, init = params

        if representation == "sparse":
            data = _20newsgroups_highdim_dataset(n_samples=8000)
        else:
            data = _blobs_dataset(n_clusters=20)

        return data

    def make_estimator(self, params):
        # [FUNCTION DOC]
        # KMeans with 20 clusters, one init, tol 0, seed 0, max_iter 30 (sparse) or 100
        # (dense), plus the given algorithm and init.
        representation, algorithm, init = params

        max_iter = 30 if representation == "sparse" else 100

        estimator = KMeans(
            n_clusters=20,
            algorithm=algorithm,
            init=init,
            n_init=1,
            max_iter=max_iter,
            tol=0,
            random_state=0,
        )

        return estimator

    def make_scorers(self):
        # [FUNCTION DOC]
        # Sets scorers to negative mean inertia of X / X_val against the fitted centers,
        # using predicted labels (scorer arguments ignored).
        self.train_scorer = lambda _, __: neg_mean_inertia(
            self.X, self.estimator.predict(self.X), self.estimator.cluster_centers_
        )
        self.test_scorer = lambda _, __: neg_mean_inertia(
            self.X_val,
            self.estimator.predict(self.X_val),
            self.estimator.cluster_centers_,
        )


class MiniBatchKMeansBenchmark(Predictor, Transformer, Estimator, Benchmark):
    # [CLASS DOC]
    # MiniBatchKMeans benchmark over representation and init.
    """
    Benchmarks for MiniBatchKMeans.
    """

    param_names = ["representation", "init"]
    params = (["dense", "sparse"], ["random", "k-means++"])

    def setup_cache(self):
        # [FUNCTION DOC]
        # Re-declares Estimator.setup_cache so ASV runs the cache step for this class.
        super().setup_cache()

    def make_data(self, params):
        # [FUNCTION DOC]
        # Sparse: full 20newsgroups TF-IDF; dense: blobs with 20 clusters.
        representation, init = params

        if representation == "sparse":
            data = _20newsgroups_highdim_dataset()
        else:
            data = _blobs_dataset(n_clusters=20)

        return data

    def make_estimator(self, params):
        # [FUNCTION DOC]
        # MiniBatchKMeans with 20 clusters, batch_size 1000, no early stopping on improvement,
        # no label computation, seed 0, max_iter 5 (sparse) or 2 (dense).
        representation, init = params

        max_iter = 5 if representation == "sparse" else 2

        estimator = MiniBatchKMeans(
            n_clusters=20,
            init=init,
            n_init=1,
            max_iter=max_iter,
            batch_size=1000,
            max_no_improvement=None,
            compute_labels=False,
            random_state=0,
        )

        return estimator

    def make_scorers(self):
        # [FUNCTION DOC]
        # Same negative mean inertia scorers as KMeansBenchmark.make_scorers.
        self.train_scorer = lambda _, __: neg_mean_inertia(
            self.X, self.estimator.predict(self.X), self.estimator.cluster_centers_
        )
        self.test_scorer = lambda _, __: neg_mean_inertia(
            self.X_val,
            self.estimator.predict(self.X_val),
            self.estimator.cluster_centers_,
        )
