# [MODULE DOC]
# ASV benchmarks for linear models: LogisticRegression, Ridge, LinearRegression,
# SGDRegressor, ElasticNet and Lasso.
from sklearn.linear_model import (
    ElasticNet,
    Lasso,
    LinearRegression,
    LogisticRegression,
    Ridge,
    SGDRegressor,
)

from .common import Benchmark, Estimator, Predictor
from .datasets import (
    _20newsgroups_highdim_dataset,
    _20newsgroups_lowdim_dataset,
    _synth_regression_dataset,
    _synth_regression_sparse_dataset,
)
from .utils import make_gen_classif_scorers, make_gen_reg_scorers


class LogisticRegressionBenchmark(Predictor, Estimator, Benchmark):
    # [CLASS DOC]
    # LogisticRegression benchmark over representation, solver (lbfgs, saga) and n_jobs.
    """
    Benchmarks for LogisticRegression.
    """

    param_names = ["representation", "solver", "n_jobs"]
    params = (["dense", "sparse"], ["lbfgs", "saga"], Benchmark.n_jobs_vals)

    def setup_cache(self):
        # [FUNCTION DOC]
        # Re-declares Estimator.setup_cache so ASV runs the cache step for this class.
        super().setup_cache()

    def make_data(self, params):
        # [FUNCTION DOC]
        # Large: 10000 sparse 20newsgroups docs or SVD-reduced to 1000 dims; otherwise 2500
        # sparse docs or the default 100-dim reduction.
        # KNOWN ISSUE: n_components=1e3 is a float; TruncatedSVD validates n_components as an
        # integer, so the large dense case likely raises InvalidParameterError.
        representation, solver, n_jobs = params

        if Benchmark.data_size == "large":
            if representation == "sparse":
                data = _20newsgroups_highdim_dataset(n_samples=10000)
            else:
                data = _20newsgroups_lowdim_dataset(n_components=1e3)
        else:
            if representation == "sparse":
                data = _20newsgroups_highdim_dataset(n_samples=2500)
            else:
                data = _20newsgroups_lowdim_dataset()

        return data

    def make_estimator(self, params):
        # [FUNCTION DOC]
        # LogisticRegression with the solver, l1_ratio 0 for lbfgs (L2) and 1 for saga (L1),
        # tol 0.01, given n_jobs, seed 0.
        representation, solver, n_jobs = params

        l1_ratio = 0 if solver == "lbfgs" else 1

        estimator = LogisticRegression(
            solver=solver,
            l1_ratio=l1_ratio,
            tol=0.01,
            n_jobs=n_jobs,
            random_state=0,
        )

        return estimator

    def make_scorers(self):
        # [FUNCTION DOC]
        # Uses balanced accuracy scorers (make_gen_classif_scorers).
        make_gen_classif_scorers(self)


class RidgeBenchmark(Predictor, Estimator, Benchmark):
    # [CLASS DOC]
    # Ridge benchmark over representation and seven solvers.
    """
    Benchmarks for Ridge.
    """

    param_names = ["representation", "solver"]
    params = (
        ["dense", "sparse"],
        ["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"],
    )

    def setup_cache(self):
        # [FUNCTION DOC]
        # Re-declares Estimator.setup_cache so ASV runs the cache step for this class.
        super().setup_cache()

    def make_data(self, params):
        # [FUNCTION DOC]
        # Dense: 500000 x 100 synthetic regression; sparse: 100000 x 10000, density 0.005.
        representation, solver = params

        if representation == "dense":
            data = _synth_regression_dataset(n_samples=500000, n_features=100)
        else:
            data = _synth_regression_sparse_dataset(
                n_samples=100000, n_features=10000, density=0.005
            )

        return data

    def make_estimator(self, params):
        # [FUNCTION DOC]
        # Ridge with the given solver, no intercept, seed 0.
        representation, solver = params

        estimator = Ridge(solver=solver, fit_intercept=False, random_state=0)

        return estimator

    def make_scorers(self):
        # [FUNCTION DOC]
        # Uses R2 scorers (make_gen_reg_scorers).
        make_gen_reg_scorers(self)

    def skip(self, params):
        # [FUNCTION DOC]
        # Skips the sparse + svd combination (svd does not support sparse input).
        representation, solver = params

        if representation == "sparse" and solver == "svd":
            return True
        return False


class LinearRegressionBenchmark(Predictor, Estimator, Benchmark):
    # [CLASS DOC]
    # LinearRegression benchmark over representation.
    """
    Benchmarks for Linear Regression.
    """

    param_names = ["representation"]
    params = (["dense", "sparse"],)

    def setup_cache(self):
        # [FUNCTION DOC]
        # Re-declares Estimator.setup_cache so ASV runs the cache step for this class.
        super().setup_cache()

    def make_data(self, params):
        # [FUNCTION DOC]
        # Dense: 1000000 x 100; sparse: 10000 x 100000, density 0.01.
        (representation,) = params

        if representation == "dense":
            data = _synth_regression_dataset(n_samples=1000000, n_features=100)
        else:
            data = _synth_regression_sparse_dataset(
                n_samples=10000, n_features=100000, density=0.01
            )

        return data

    def make_estimator(self, params):
        # [FUNCTION DOC]
        # Default LinearRegression (params ignored).
        estimator = LinearRegression()

        return estimator

    def make_scorers(self):
        # [FUNCTION DOC]
        # Uses R2 scorers (make_gen_reg_scorers).
        make_gen_reg_scorers(self)


class SGDRegressorBenchmark(Predictor, Estimator, Benchmark):
    # [CLASS DOC]
    # SGDRegressor benchmark over representation.
    """
    Benchmark for SGD
    """

    param_names = ["representation"]
    params = (["dense", "sparse"],)

    def setup_cache(self):
        # [FUNCTION DOC]
        # Re-declares Estimator.setup_cache so ASV runs the cache step for this class.
        super().setup_cache()

    def make_data(self, params):
        # [FUNCTION DOC]
        # Dense: 100000 x 200; sparse: 100000 x 1000, density 0.01.
        (representation,) = params

        if representation == "dense":
            data = _synth_regression_dataset(n_samples=100000, n_features=200)
        else:
            data = _synth_regression_sparse_dataset(
                n_samples=100000, n_features=1000, density=0.01
            )

        return data

    def make_estimator(self, params):
        # [FUNCTION DOC]
        # SGDRegressor with tol None (runs all epochs), 60 epochs dense or 300 sparse, seed 0.
        (representation,) = params

        max_iter = 60 if representation == "dense" else 300

        estimator = SGDRegressor(max_iter=max_iter, tol=None, random_state=0)

        return estimator

    def make_scorers(self):
        # [FUNCTION DOC]
        # Uses R2 scorers (make_gen_reg_scorers).
        make_gen_reg_scorers(self)


class ElasticNetBenchmark(Predictor, Estimator, Benchmark):
    # [CLASS DOC]
    # ElasticNet benchmark over representation and precompute.
    """
    Benchmarks for ElasticNet.
    """

    param_names = ["representation", "precompute"]
    params = (["dense", "sparse"], [True, False])

    def setup_cache(self):
        # [FUNCTION DOC]
        # Re-declares Estimator.setup_cache so ASV runs the cache step for this class.
        super().setup_cache()

    def make_data(self, params):
        # [FUNCTION DOC]
        # Dense: 1000000 x 100; sparse: 50000 x 5000, density 0.01.
        representation, precompute = params

        if representation == "dense":
            data = _synth_regression_dataset(n_samples=1000000, n_features=100)
        else:
            data = _synth_regression_sparse_dataset(
                n_samples=50000, n_features=5000, density=0.01
            )

        return data

    def make_estimator(self, params):
        # [FUNCTION DOC]
        # ElasticNet with alpha 0.001, given precompute, seed 0.
        representation, precompute = params

        estimator = ElasticNet(precompute=precompute, alpha=0.001, random_state=0)

        return estimator

    def make_scorers(self):
        # [FUNCTION DOC]
        # Uses R2 scorers (make_gen_reg_scorers).
        make_gen_reg_scorers(self)

    def skip(self, params):
        # [FUNCTION DOC]
        # Skips sparse with precompute False.
        representation, precompute = params

        if representation == "sparse" and precompute is False:
            return True
        return False


class LassoBenchmark(Predictor, Estimator, Benchmark):
    # [CLASS DOC]
    # Lasso benchmark over representation and precompute.
    """
    Benchmarks for Lasso.
    """

    param_names = ["representation", "precompute"]
    params = (["dense", "sparse"], [True, False])

    def setup_cache(self):
        # [FUNCTION DOC]
        # Re-declares Estimator.setup_cache so ASV runs the cache step for this class.
        super().setup_cache()

    def make_data(self, params):
        # [FUNCTION DOC]
        # Dense: 1000000 x 100; sparse: 50000 x 5000, density 0.01.
        representation, precompute = params

        if representation == "dense":
            data = _synth_regression_dataset(n_samples=1000000, n_features=100)
        else:
            data = _synth_regression_sparse_dataset(
                n_samples=50000, n_features=5000, density=0.01
            )

        return data

    def make_estimator(self, params):
        # [FUNCTION DOC]
        # Lasso with alpha 0.001, given precompute, seed 0.
        representation, precompute = params

        estimator = Lasso(precompute=precompute, alpha=0.001, random_state=0)

        return estimator

    def make_scorers(self):
        # [FUNCTION DOC]
        # Uses R2 scorers (make_gen_reg_scorers).
        make_gen_reg_scorers(self)

    def skip(self, params):
        # [FUNCTION DOC]
        # Skips sparse with precompute False.
        representation, precompute = params

        if representation == "sparse" and precompute is False:
            return True
        return False
