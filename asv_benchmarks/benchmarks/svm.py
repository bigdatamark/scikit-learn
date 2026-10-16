# [MODULE DOC]
# ASV benchmark for SVC.
from sklearn.svm import SVC

from .common import Benchmark, Estimator, Predictor
from .datasets import _synth_classification_dataset
from .utils import make_gen_classif_scorers


class SVCBenchmark(Predictor, Estimator, Benchmark):
    # [CLASS DOC]
    # SVC benchmark over kernel (linear, poly, rbf, sigmoid).
    """Benchmarks for SVC."""

    param_names = ["kernel"]
    params = (["linear", "poly", "rbf", "sigmoid"],)

    def setup_cache(self):
        # [FUNCTION DOC]
        # Re-declares Estimator.setup_cache so ASV runs the cache step for this class.
        super().setup_cache()

    def make_data(self, params):
        # [FUNCTION DOC]
        # Default synthetic classification data (1000 x 10000, 2 classes).
        return _synth_classification_dataset()

    def make_estimator(self, params):
        # [FUNCTION DOC]
        # SVC capped at 100 iterations with tol 1e-16 (so it always hits the cap), given
        # kernel, gamma "scale", seed 0. Emits ConvergenceWarning when fitted.
        (kernel,) = params

        estimator = SVC(
            max_iter=100, tol=1e-16, kernel=kernel, random_state=0, gamma="scale"
        )

        return estimator

    def make_scorers(self):
        # [FUNCTION DOC]
        # Uses balanced accuracy scorers (make_gen_classif_scorers).
        make_gen_classif_scorers(self)
