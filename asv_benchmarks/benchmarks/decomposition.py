# [MODULE DOC]
# ASV benchmarks for PCA, DictionaryLearning and MiniBatchDictionaryLearning.
from sklearn.decomposition import PCA, DictionaryLearning, MiniBatchDictionaryLearning

from .common import Benchmark, Estimator, Transformer
from .datasets import _mnist_dataset, _olivetti_faces_dataset
from .utils import make_dict_learning_scorers, make_pca_scorers


class PCABenchmark(Transformer, Estimator, Benchmark):
    # [CLASS DOC]
    # PCA benchmark on MNIST over svd_solver (full, arpack, randomized).
    """
    Benchmarks for PCA.
    """

    param_names = ["svd_solver"]
    params = (["full", "arpack", "randomized"],)

    def setup_cache(self):
        # [FUNCTION DOC]
        # Re-declares Estimator.setup_cache so ASV runs the cache step for this class.
        super().setup_cache()

    def make_data(self, params):
        # [FUNCTION DOC]
        # Returns the MNIST dataset (params ignored).
        return _mnist_dataset()

    def make_estimator(self, params):
        # [FUNCTION DOC]
        # PCA with 32 components, the given solver, seed 0.
        (svd_solver,) = params

        estimator = PCA(n_components=32, svd_solver=svd_solver, random_state=0)

        return estimator

    def make_scorers(self):
        # [FUNCTION DOC]
        # Uses explained-variance-ratio scorers (make_pca_scorers).
        make_pca_scorers(self)


class DictionaryLearningBenchmark(Transformer, Estimator, Benchmark):
    # [CLASS DOC]
    # DictionaryLearning benchmark on Olivetti faces over fit_algorithm (lars, cd) and
    # n_jobs values from the config.
    """
    Benchmarks for DictionaryLearning.
    """

    param_names = ["fit_algorithm", "n_jobs"]
    params = (["lars", "cd"], Benchmark.n_jobs_vals)

    def setup_cache(self):
        # [FUNCTION DOC]
        # Re-declares Estimator.setup_cache so ASV runs the cache step for this class.
        super().setup_cache()

    def make_data(self, params):
        # [FUNCTION DOC]
        # Returns the Olivetti faces dataset (params ignored).
        return _olivetti_faces_dataset()

    def make_estimator(self, params):
        # [FUNCTION DOC]
        # DictionaryLearning with 15 atoms, alpha 0.1, transform_alpha 1, 20 iterations,
        # tol 1e-16 (effectively runs all iterations), seed 0, given algorithm and n_jobs.
        fit_algorithm, n_jobs = params

        estimator = DictionaryLearning(
            n_components=15,
            fit_algorithm=fit_algorithm,
            alpha=0.1,
            transform_alpha=1,
            max_iter=20,
            tol=1e-16,
            random_state=0,
            n_jobs=n_jobs,
        )

        return estimator

    def make_scorers(self):
        # [FUNCTION DOC]
        # Uses reconstruction-error scorers (make_dict_learning_scorers).
        make_dict_learning_scorers(self)


class MiniBatchDictionaryLearningBenchmark(Transformer, Estimator, Benchmark):
    # [CLASS DOC]
    # MiniBatchDictionaryLearning benchmark on Olivetti faces over fit_algorithm and
    # n_jobs.
    """
    Benchmarks for MiniBatchDictionaryLearning
    """

    param_names = ["fit_algorithm", "n_jobs"]
    params = (["lars", "cd"], Benchmark.n_jobs_vals)

    def setup_cache(self):
        # [FUNCTION DOC]
        # Re-declares Estimator.setup_cache so ASV runs the cache step for this class.
        super().setup_cache()

    def make_data(self, params):
        # [FUNCTION DOC]
        # Returns the Olivetti faces dataset (params ignored).
        return _olivetti_faces_dataset()

    def make_estimator(self, params):
        # [FUNCTION DOC]
        # MiniBatchDictionaryLearning with 15 atoms, alpha 0.1, batch_size 3, seed 0, given
        # algorithm and n_jobs.
        fit_algorithm, n_jobs = params

        estimator = MiniBatchDictionaryLearning(
            n_components=15,
            fit_algorithm=fit_algorithm,
            alpha=0.1,
            batch_size=3,
            random_state=0,
            n_jobs=n_jobs,
        )

        return estimator

    def make_scorers(self):
        # [FUNCTION DOC]
        # Uses reconstruction-error scorers (make_dict_learning_scorers).
        make_dict_learning_scorers(self)
