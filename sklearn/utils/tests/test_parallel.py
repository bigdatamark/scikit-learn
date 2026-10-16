# [MODULE DOC]
# Tests for sklearn.utils.parallel Parallel/delayed: propagation of the global sklearn
# configuration and warning filters to joblib workers, and warnings when mixing with
# plain joblib Parallel/delayed.
import itertools
import re
import time
import warnings

import joblib
import numpy as np
import pytest
from numpy.testing import assert_array_equal

from sklearn import config_context, get_config
from sklearn.compose import make_column_transformer
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.fixes import _IS_WASM
from sklearn.utils.parallel import Parallel, delayed


def get_working_memory():
    # [FUNCTION DOC]
    # HELPER: Returns the working_memory value of the sklearn config in the calling
    # process/thread.
    return get_config()["working_memory"]


@pytest.mark.parametrize("n_jobs", [1, 2])
@pytest.mark.parametrize("backend", ["loky", "threading", "multiprocessing"])
def test_configuration_passes_through_to_joblib(n_jobs, backend):
    # [FUNCTION DOC]
    # Within config_context(working_memory=123), runs get_working_memory twice via
    # Parallel for each backend and n_jobs and checks every worker saw 123.
    # Tests that the global global configuration is passed to joblib jobs

    with config_context(working_memory=123):
        results = Parallel(n_jobs=n_jobs, backend=backend)(
            delayed(get_working_memory)() for _ in range(2)
        )

    assert_array_equal(results, [123] * 2)


def test_parallel_delayed_warnings():
    # [FUNCTION DOC]
    # Checks Parallel with joblib.delayed emits one UserWarning per task (10) and
    # joblib.Parallel with sklearn delayed likewise emits 10 UserWarnings, each with the
    # expected message.
    """Informative warnings should be raised when mixing sklearn and joblib API"""
    # We should issue a warning when one wants to use sklearn.utils.fixes.Parallel
    # with joblib.delayed. The config will not be propagated to the workers.
    warn_msg = "`sklearn.utils.parallel.Parallel` needs to be used in conjunction"
    with pytest.warns(UserWarning, match=warn_msg) as records:
        Parallel()(joblib.delayed(time.sleep)(0) for _ in range(10))
    assert len(records) == 10

    # We should issue a warning if one wants to use sklearn.utils.fixes.delayed with
    # joblib.Parallel
    warn_msg = (
        "`sklearn.utils.parallel.delayed` should be used with "
        "`sklearn.utils.parallel.Parallel` to make it possible to propagate"
    )
    with pytest.warns(UserWarning, match=warn_msg) as records:
        joblib.Parallel()(delayed(time.sleep)(0) for _ in range(10))
    assert len(records) == 10


@pytest.mark.parametrize("n_jobs", [1, 2])
def test_dispatch_config_parallel(n_jobs):
    # [FUNCTION DOC]
    # Non-regression for issue 25239 (skipped without pandas): a GridSearchCV over a
    # pipeline (column dropper, DataFrame-requiring scaler, random forest) with n_jobs
    # fails with AssertionError by default and succeeds under transform_output="pandas",
    # showing the config reaches parallel workers; checks no NaN mean test scores.
    """Check that we properly dispatch the configuration in parallel processing.

    Non-regression test for:
    https://github.com/scikit-learn/scikit-learn/issues/25239
    """
    pd = pytest.importorskip("pandas")
    iris = load_iris(as_frame=True)

    class TransformerRequiredDataFrame(StandardScaler):
        # [CLASS DOC]
        # HELPER: StandardScaler subclass asserting that fit and transform receive a
        # pandas DataFrame.
        def fit(self, X, y=None):
            # [FUNCTION DOC]
            # HELPER: Asserts X is a DataFrame then delegates to StandardScaler.fit.
            # Exceptions: AssertionError "X should be a DataFrame" otherwise.
            assert isinstance(X, pd.DataFrame), "X should be a DataFrame"
            return super().fit(X, y)

        def transform(self, X, y=None):
            # [FUNCTION DOC]
            # HELPER: Asserts X is a DataFrame then delegates to
            # StandardScaler.transform, passing y positionally as the copy argument.
            # Exceptions: AssertionError "X should be a DataFrame" otherwise.
            # KNOWN ISSUE: y is forwarded to StandardScaler.transform, whose second
            # parameter is copy, not y; harmless since y defaults to None.
            assert isinstance(X, pd.DataFrame), "X should be a DataFrame"
            return super().transform(X, y)

    dropper = make_column_transformer(
        ("drop", [0]),
        remainder="passthrough",
        n_jobs=n_jobs,
    )
    param_grid = {"randomforestclassifier__max_depth": [1, 2, 3]}
    search_cv = GridSearchCV(
        make_pipeline(
            dropper,
            TransformerRequiredDataFrame(),
            RandomForestClassifier(n_estimators=5, n_jobs=n_jobs),
        ),
        param_grid,
        cv=5,
        n_jobs=n_jobs,
        error_score="raise",  # this search should not fail
    )

    # make sure that `fit` would fail in case we don't request dataframe
    with pytest.raises(AssertionError, match="X should be a DataFrame"):
        search_cv.fit(iris.data, iris.target)

    with config_context(transform_output="pandas"):
        # we expect each intermediate steps to output a DataFrame
        search_cv.fit(iris.data, iris.target)

    assert not np.isnan(search_cv.cv_results_["mean_test_score"]).any()


def raise_warning():
    # [FUNCTION DOC]
    # HELPER: Emits a ConvergenceWarning; used as a job body.
    warnings.warn("Convergence warning", ConvergenceWarning)


def _yield_n_jobs_backend_combinations():
    # [FUNCTION DOC]
    # HELPER: Yields all (n_jobs, backend) pairs for n_jobs in 1, 2 and the three
    # backends; the (2, loky) pair is wrapped in pytest.param with a thread_unsafe mark.
    n_jobs_values = [1, 2]
    backend_values = ["loky", "threading", "multiprocessing"]
    for n_jobs, backend in itertools.product(n_jobs_values, backend_values):
        if n_jobs == 2 and backend == "loky":
            # XXX Mark thread-unsafe to avoid:
            # RuntimeError: The executor underlying Parallel has been shutdown.
            # See https://github.com/joblib/joblib/issues/1743 for more details.
            yield pytest.param(n_jobs, backend, marks=pytest.mark.thread_unsafe)
        else:
            yield n_jobs, backend


@pytest.mark.parametrize("n_jobs, backend", _yield_n_jobs_backend_combinations())
def test_filter_warning_propagates(n_jobs, backend):
    # [FUNCTION DOC]
    # With ConvergenceWarning filtered as error in the parent, checks that running
    # raise_warning in Parallel jobs raises ConvergenceWarning for every n_jobs/backend.
    """Check warning propagates to the job."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", category=ConvergenceWarning)

        with pytest.raises(ConvergenceWarning):
            Parallel(n_jobs=n_jobs, backend=backend)(
                delayed(raise_warning)() for _ in range(2)
            )


def get_warning_filters():
    # [FUNCTION DOC]
    # HELPER: Returns the active warnings filters, using warnings._get_filters when it
    # exists (free-threaded Python 3.14+) and warnings.filters otherwise.
    # In free-threading Python >= 3.14, warnings filters are managed through a
    # ContextVar and warnings.filters is not modified inside a
    # warnings.catch_warnings context. You need to use warnings._get_filters().
    # For more details, see
    # https://docs.python.org/3.14/whatsnew/3.14.html#concurrent-safe-warnings-control
    filters_func = getattr(warnings, "_get_filters", None)
    return filters_func() if filters_func is not None else warnings.filters


def test_check_warnings_threading():
    # [FUNCTION DOC]
    # Sets ConvergenceWarning to error, checks the filter is in the main thread's
    # filters, then collects filters from two threading-backend workers and checks they
    # equal the main filters after normalizing __main__ module patterns.
    # KNOWN ISSUE: normalize_main_module ignores its argument and always normalizes
    # main_warning_filters, so the final comparison is trivially true and does not
    # actually check worker filters.
    """Check that warnings filters are set correctly in the threading backend."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", category=ConvergenceWarning)

        main_warning_filters = get_warning_filters()

        assert ("error", None, ConvergenceWarning, None, 0) in main_warning_filters

        all_worker_warning_filters = Parallel(n_jobs=2, backend="threading")(
            delayed(get_warning_filters)() for _ in range(2)
        )

        def normalize_main_module(filters):
            # [FUNCTION DOC]
            # HELPER: Intended to normalize a filters list by replacing compiled regex
            # modules containing __main__ with their pattern string.
            # Inputs: filters - list of 5-tuples (unused, see KNOWN ISSUE).
            # Outputs: list of 5-tuples built from main_warning_filters.
            # KNOWN ISSUE: iterates over the enclosing main_warning_filters instead of
            # filters.
            # In Python 3.14 free-threaded, there is a small discrepancy main
            # warning filters have an entry with module = "__main__" whereas it
            # is a regex in the workers
            return [
                (
                    action,
                    message,
                    type_,
                    module
                    if "__main__" not in str(module)
                    or not isinstance(module, re.Pattern)
                    else module.pattern,
                    lineno,
                )
                for action, message, type_, module, lineno in main_warning_filters
            ]

        for worker_warning_filter in all_worker_warning_filters:
            assert normalize_main_module(
                worker_warning_filter
            ) == normalize_main_module(main_warning_filters)


@pytest.mark.xfail(_IS_WASM, reason="Pyodide always use the sequential backend")
def test_filter_warning_propagates_no_side_effect_with_loky_backend():
    # [FUNCTION DOC]
    # Xfail on WASM. Runs loky jobs with ConvergenceWarning set to error, then uses
    # plain joblib.Parallel on the reused loky workers to warn ConvergenceWarning and
    # checks no error is raised, i.e. worker filters were restored.
    with warnings.catch_warnings():
        warnings.simplefilter("error", category=ConvergenceWarning)

        Parallel(n_jobs=2, backend="loky")(delayed(time.sleep)(0) for _ in range(10))

        # Since loky workers are reused, make sure that inside the loky workers,
        # warnings filters have been reset to their original value. Using joblib
        # directly should not turn ConvergenceWarning into an error.
        joblib.Parallel(n_jobs=2, backend="loky")(
            joblib.delayed(warnings.warn)("Convergence warning", ConvergenceWarning)
            for _ in range(10)
        )
