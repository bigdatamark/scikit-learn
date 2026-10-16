# [MODULE DOC]
# Tests for the ScoringMonitor callback: logged scores per task for simple and
# meta-estimators (string, multi-metric and callable scoring; list and pandas output;
# with and without lineage rows), get_logs selection/types/errors, sample weights,
# ancestor retrieval through task id paths, use from functions without callback
# support, and listener cleanup on garbage collection.
# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest

from sklearn.base import clone
from sklearn.callback import ScoringMonitor
from sklearn.callback._scoring_monitor import ScoringMonitorLog
from sklearn.callback.tests._utils import (
    MaxIterEstimator,
    MetaEstimator,
    WhileEstimator,
)
from sklearn.datasets import make_regression
from sklearn.metrics import check_scoring, make_scorer
from sklearn.utils._testing import assert_allclose
from sklearn.utils.parallel import Parallel, delayed


def _make_expected_output_MaxIterEstimator(
    max_iter, scoring, as_pandas, X, y, root_task_id=0
):
    # [FUNCTION DOC]
    # HELPER: Builds the log a ScoringMonitor should produce for one MaxIterEstimator fit:
    # one row for the root "fit" task (estimator fitted with max_iter) followed by one row
    # per iteration i (estimator refitted with max_iter=i+1), each holding task metadata
    # and the scores computed on X, y.
    # Inputs: max_iter - int; scoring - str, callable scorer or tuple of names; as_pandas -
    # bool; X, y - evaluation data; root_task_id - id of the root task (default 0).
    # Outputs: list of dicts, or a pandas DataFrame if as_pandas.
    # Side effects: skips the test if as_pandas and pandas is missing.
    # KNOWN ISSUE: the reference estimators are fitted without X, y; this matches only
    # because MaxIterEstimator's fitted state does not depend on the data.
    """Generate the expected output of a ScoringMonitor on a MaxIterEstimator."""
    scorer = check_scoring(None, scoring)
    if isinstance(scoring, str):
        score_names = [scoring]
    elif callable(scoring):
        score_names = ["score"]
    else:  # multi-scorer
        score_names = list(scorer._scorers.keys())

    est_name = MaxIterEstimator.__name__

    expected_log = []

    # fit root task
    fitted_est = MaxIterEstimator(max_iter=max_iter).fit()
    log_item = {
        "task_id_path": (root_task_id,),
        "parent_task_id_path": (),
        "estimator_name": est_name,
        "task_name": "fit",
        "task_id": root_task_id,
        "sequential_subtasks": True,
    }
    scores = scorer(fitted_est, X, y)
    if not isinstance(scores, dict):
        scores = {score_names[0]: scores}
    expected_log.append({**log_item, **scores})

    # fit loop iterations
    for i in range(max_iter):
        fitted_est = MaxIterEstimator(max_iter=i + 1).fit()
        log_item = {
            "task_id_path": (root_task_id, i),
            "parent_task_id_path": (root_task_id,),
            "estimator_name": est_name,
            "task_name": f"iteration {i}",
            "task_id": i,
            "sequential_subtasks": True,
        }
        scores = scorer(fitted_est, X, y)
        if not isinstance(scores, dict):
            scores = {score_names[0]: scores}
        expected_log.append({**log_item, **scores})

    if as_pandas:
        pd = pytest.importorskip("pandas")
        expected_log = pd.DataFrame(expected_log)

    return expected_log


def _make_expected_output_MetaEstimator(
    n_outer, n_inner, max_iter, scoring, as_pandas, include_lineage, X, y
):
    # [FUNCTION DOC]
    # HELPER: Builds the expected MetaEstimator log: for each outer i and inner j, the
    # MaxIterEstimator log with root id j, with task paths prefixed by (0, i); if
    # include_lineage, adds the meta root "fit" row and one "outer" row per i. Rows are
    # sorted by parent path length then parent path (stable sort).
    # Inputs: n_outer, n_inner, max_iter - ints; scoring, as_pandas, include_lineage, X, y.
    # Outputs: list of dicts or DataFrame.
    """Generate the expected output of a ScoringMonitor on a MetaEstimator.

    The sub-estimators are expected to be MaxIterEstimator.
    """
    meta_est_name = MetaEstimator.__name__

    expected_log = []
    for i in range(n_outer):
        for j in range(n_inner):
            estimator_log = _make_expected_output_MaxIterEstimator(
                max_iter, scoring, False, X, y, root_task_id=j
            )
            for row in estimator_log:
                row["task_id_path"] = (0, i) + row["task_id_path"]
                row["parent_task_id_path"] = (0, i) + row["parent_task_id_path"]
                expected_log.append(row)

    if include_lineage:
        # fit root task
        expected_log.append(
            {
                "task_id_path": (0,),
                "parent_task_id_path": (),
                "estimator_name": meta_est_name,
                "task_name": "fit",
                "task_id": 0,
                "sequential_subtasks": False,
            }
        )
        # outer tasks
        for i in range(n_outer):
            expected_log.append(
                {
                    "task_id_path": (0, i),
                    "parent_task_id_path": (0,),
                    "estimator_name": meta_est_name,
                    "task_name": "outer",
                    "task_id": i,
                    "sequential_subtasks": True,
                }
            )

    sorting_key = lambda x: (len(x["parent_task_id_path"]), x["parent_task_id_path"])
    expected_log.sort(key=sorting_key)

    if as_pandas:
        pd = pytest.importorskip("pandas")
        expected_log = pd.DataFrame(expected_log)

    return expected_log


def _custom_score_func(y_true, y_pred):
    # [FUNCTION DOC]
    # HELPER: Dummy metric always returning 0, wrapped as custom_score.
    # Inputs: y_true, y_pred - ignored. Outputs: 0.
    """Custom score to test the ScoringMonitor with a callable."""
    return 0


custom_score = make_scorer(_custom_score_func)


def test_score_after_fit():
    # [FUNCTION DOC]
    # Verifies each logged per-iteration r2 equals the score of an estimator fitted from
    # scratch with max_iter = i+1.
    """Check that the logged scores are the same as if the fit ended at each step."""
    X, y = make_regression(n_samples=100, n_features=2, random_state=0)

    max_iter = 10
    callback = ScoringMonitor(scoring="r2")
    estimator = MaxIterEstimator(max_iter=max_iter).set_callbacks(callback)
    estimator.fit(X=X, y=y)

    log = callback.get_logs()
    # select the rows corresponding to the iteration tasks
    iter_log = [row for row in log.data if row["task_name"].startswith("iteration")]

    scorer = check_scoring(None, "r2")
    for i in range(max_iter):
        estimator = MaxIterEstimator(max_iter=i + 1).fit(X, y)
        score = scorer(estimator, X, y)
        assert_allclose(score, iter_log[i]["r2"])


@pytest.mark.parametrize(
    "scoring",
    ["neg_mean_squared_error", ("neg_mean_squared_error", "r2"), custom_score],
)
@pytest.mark.parametrize("as_pandas", [True, False])
def test_logged_values(scoring, as_pandas):
    # [FUNCTION DOC]
    # Verifies the full log (list or DataFrame) of a MaxIterEstimator fit equals
    # _make_expected_output_MaxIterEstimator.
    # Inputs: scoring, as_pandas - parametrized.
    """Test that the correct values are logged with a simple estimator."""
    if as_pandas:
        pytest.importorskip("pandas")

    max_iter = 3
    callback = ScoringMonitor(scoring=scoring)
    estimator = MaxIterEstimator(max_iter=max_iter).set_callbacks(callback)
    X, y = make_regression(n_samples=100, n_features=2, random_state=0)

    estimator.fit(X=X, y=y)

    attr = "data" if not as_pandas else "data_as_pandas"
    log = getattr(callback.get_logs(), attr)
    expected_log = _make_expected_output_MaxIterEstimator(
        max_iter, scoring, as_pandas, X, y
    )

    if as_pandas:
        assert log.equals(expected_log)
    else:
        assert log == expected_log


@pytest.mark.parametrize("prefer", ["processes", "threads"])
@pytest.mark.parametrize(
    "scoring",
    ["neg_mean_squared_error", ("neg_mean_squared_error", "r2"), custom_score],
)
@pytest.mark.parametrize("as_pandas", [True, False])
@pytest.mark.parametrize("include_lineage", [True, False])
def test_logged_values_meta_estimator(prefer, scoring, as_pandas, include_lineage):
    # [FUNCTION DOC]
    # Verifies the log of a callback on the sub-estimator of a parallel MetaEstimator
    # equals _make_expected_output_MetaEstimator, with and without lineage rows.
    # Inputs: prefer, scoring, as_pandas, include_lineage - parametrized.
    """Test that the correct values are logged with a meta-estimator."""
    if as_pandas:
        pytest.importorskip("pandas")

    n_outer, n_inner, max_iter = 3, 2, 5
    callback = ScoringMonitor(scoring=scoring)
    est = MaxIterEstimator(max_iter=max_iter).set_callbacks(callback)
    meta_est = MetaEstimator(
        est, n_outer=n_outer, n_inner=n_inner, n_jobs=2, prefer=prefer
    )
    X, y = make_regression(n_samples=100, n_features=2, random_state=0)

    meta_est.fit(X=X, y=y)

    attr = "data" if not as_pandas else "data_as_pandas"
    log = callback.get_logs(include_lineage=include_lineage)
    log = getattr(log, attr)
    expected_log = _make_expected_output_MetaEstimator(
        n_outer, n_inner, max_iter, scoring, as_pandas, include_lineage, X, y
    )

    if as_pandas:
        assert log.equals(expected_log)
    else:
        assert log == expected_log


@pytest.mark.parametrize("select", ["all", "most_recent"])
def test_get_logs_output_type_no_fit(select):
    # [FUNCTION DOC]
    # Verifies get_logs raises ValueError before any fit for both select modes.
    # Inputs: select - parametrized.
    """Check that get_logs raises an error before fit."""
    callback = ScoringMonitor(scoring="neg_mean_squared_error")

    with pytest.raises(ValueError, match="No logs to retrieve"):
        callback.get_logs(select=select)


@pytest.mark.parametrize("as_pandas", [True, False])
def test_get_logs_output_type(as_pandas):
    # [FUNCTION DOC]
    # After two fits, verifies select="all" returns a list of 2 ScoringMonitorLog,
    # select="most_recent" a single one, and data/data_as_pandas have the right types.
    # Inputs: as_pandas - parametrized.
    """Test the type of the get_logs output."""
    if as_pandas:
        pd = pytest.importorskip("pandas")

    callback = ScoringMonitor(scoring="neg_mean_squared_error")
    estimator = MaxIterEstimator().set_callbacks(callback)
    X, y = make_regression(n_samples=100, n_features=2, random_state=0)

    # 2 runs
    estimator.fit(X, y)
    estimator.fit(X, y)

    logs_all = callback.get_logs(select="all")
    assert isinstance(logs_all, list)
    assert len(logs_all) == 2
    assert all(isinstance(log, ScoringMonitorLog) for log in logs_all)

    expected_data_type = pd.DataFrame if as_pandas else list
    attr = "data" if not as_pandas else "data_as_pandas"
    assert all(isinstance(getattr(log, attr), expected_data_type) for log in logs_all)

    log_most_recent = callback.get_logs(select="most_recent")
    assert isinstance(log_most_recent, ScoringMonitorLog)
    assert isinstance(getattr(log_most_recent, attr), expected_data_type)


@pytest.mark.parametrize("select", ["all", "most_recent"])
def test_estimator_without_reconstruction_attributes(select):
    # [FUNCTION DOC]
    # Verifies that for an estimator that never provides reconstruction_attributes no log
    # is recorded, so get_logs raises ValueError.
    # Inputs: select - parametrized.
    """Smoke test on an estimator which does not provide reconstruction_attributes."""
    callback = ScoringMonitor(scoring="r2")
    WhileEstimator().set_callbacks(callback).fit()

    with pytest.raises(ValueError, match="No logs to retrieve"):
        callback.get_logs(select=select)


def test_scoringmonitor_sample_weights():
    # [FUNCTION DOC]
    # Verifies that fitting with non-uniform integer sample weights yields a different log
    # than fitting without weights.
    """Check that the ScoringMonitor works with sample weights."""
    rng = np.random.RandomState(0)
    X, y = make_regression(n_samples=100, n_features=2, random_state=rng)
    sample_weight = rng.randint(0, 5, size=X.shape[0])

    # no sample weights
    callback = ScoringMonitor(scoring="r2")
    MaxIterEstimator().set_callbacks(callback).fit(X=X, y=y)
    log_no_sw = callback.get_logs().data

    # sample weights
    callback = ScoringMonitor(scoring="r2")
    MaxIterEstimator().set_callbacks(callback).fit(
        X=X, y=y, sample_weight=sample_weight
    )
    log_sw = callback.get_logs().data

    assert log_no_sw != log_sw


def _get_ancestors_info_path(log, parent_task_id_path):
    # [FUNCTION DOC]
    # HELPER: Walks up the task tree in a list-of-dicts log from parent_task_id_path to
    # the root, collecting estimator_name, task_name and task_id of each ancestor.
    # Inputs: log - list of row dicts; parent_task_id_path - tuple.
    # Outputs: list of dicts ordered root first.
    # Exceptions: KeyError if a path is missing from the log.
    log = {row["task_id_path"]: row for row in log}
    ancestors_info_path = []
    while parent_task_id_path:
        task_info = log[parent_task_id_path]
        ancestors_info_path.append(
            {
                "estimator_name": task_info["estimator_name"],
                "task_name": task_info["task_name"],
                "task_id": task_info["task_id"],
            }
        )
        parent_task_id_path = task_info["parent_task_id_path"]
    ancestors_info_path.reverse()
    return ancestors_info_path


def _get_ancestors_info_path_pandas(log, parent_task_id_path):
    # [FUNCTION DOC]
    # HELPER: Same as _get_ancestors_info_path for a DataFrame log, indexing rows by
    # task_id_path.
    # Inputs: log - DataFrame; parent_task_id_path - tuple. Outputs: list of dicts.
    log = log.set_index("task_id_path", drop=False)
    ancestors_info_path = []
    while parent_task_id_path:
        task_info = log.loc[[parent_task_id_path]].iloc[0]
        ancestors_info_path.append(
            {
                "estimator_name": task_info["estimator_name"],
                "task_name": task_info["task_name"],
                "task_id": task_info["task_id"],
            }
        )
        parent_task_id_path = task_info["parent_task_id_path"]
    ancestors_info_path.reverse()
    return ancestors_info_path


@pytest.mark.parametrize("as_pandas", [True, False])
def test_get_logs_include_lineage_ancestor_retrieval(as_pandas):
    # [FUNCTION DOC]
    # Verifies that with include_lineage, the max_iter sibling rows under path (0, 1, 1)
    # are found and their ancestors (meta fit, outer #1, inner fit #1) can be recovered
    # from the log.
    # Inputs: as_pandas - parametrized.
    """Check that ancestor task info can be retrieved from a given task."""
    if as_pandas:
        pytest.importorskip("pandas")

    n_outer, n_inner, max_iter = 2, 3, 5
    callback = ScoringMonitor(scoring="r2")
    est = MaxIterEstimator(max_iter=max_iter).set_callbacks(callback)
    meta_est = MetaEstimator(est, n_outer=n_outer, n_inner=n_inner)
    X, y = make_regression(n_samples=100, n_features=2, random_state=0)

    meta_est.fit(X=X, y=y)

    attr = "data" if not as_pandas else "data_as_pandas"
    log = callback.get_logs(include_lineage=True)
    log = getattr(log, attr)

    # pick an arbitrary parent of iteration tasks
    parent_path = (0, 1, 1)
    expected_ancestors_info_path = [
        {
            "estimator_name": "MetaEstimator",
            "task_name": "fit",
            "task_id": 0,
        },
        {
            "estimator_name": "MetaEstimator",
            "task_name": "outer",
            "task_id": 1,
        },
        {
            "estimator_name": "MaxIterEstimator",
            "task_name": "fit",
            "task_id": 1,
        },
    ]

    if as_pandas:
        sibling_rows = log.loc[log["parent_task_id_path"] == parent_path]
        ancestors_info_path = _get_ancestors_info_path_pandas(log, parent_path)
    else:
        sibling_rows = [row for row in log if row["parent_task_id_path"] == parent_path]
        ancestors_info_path = _get_ancestors_info_path(log, parent_path)

    assert len(sibling_rows) == max_iter
    assert ancestors_info_path == expected_ancestors_info_path


@pytest.mark.parametrize("backend", ["threading", "loky"])
def test_scoring_monitor_no_callback_support(backend):
    # [FUNCTION DOC]
    # Verifies that parallel clone-and-fit from a function without callback support
    # produces n_fits separate logs, each equal to the single-fit expected log.
    # Inputs: backend - parametrized.
    """Test ScoringMonitor within a parallelized function not supporting callbacks.

    The expected result is multiple separate logs, similar to the result of multiple
    sequential fit executions.
    """

    def clone_and_fit(estimator, X, y):
        # [FUNCTION DOC]
        # HELPER: Clones the estimator and fits it on X, y.
        clone(estimator).fit(X, y)

    def func(estimator, X, y, n_fits):
        # [FUNCTION DOC]
        # HELPER: Runs clone_and_fit n_fits times via joblib Parallel (2 jobs).
        Parallel(n_jobs=2, backend=backend)(
            delayed(clone_and_fit)(estimator, X, y) for _ in range(n_fits)
        )

    max_iter = 3
    n_fits = 4

    callback = ScoringMonitor(scoring="r2")
    X, y = make_regression(n_samples=100, n_features=2, random_state=0)
    func(
        MaxIterEstimator(max_iter=max_iter).set_callbacks(callback), X, y, n_fits=n_fits
    )
    log_all = callback.get_logs(select="all")

    expected_log = _make_expected_output_MaxIterEstimator(
        max_iter, scoring="r2", as_pandas=False, X=X, y=y
    )

    assert len(log_all) == n_fits
    for log in log_all:
        assert log.data == expected_log
        assert log.estimator_name == "MaxIterEstimator"


def test_scoring_monitor_listener_closed_on_gc():
    # [FUNCTION DOC]
    # Verifies that a ScoringMonitor's listener address is registered in _listeners and
    # _message_consumers on creation and removed once the callback is deleted.
    """Check that listener is closed when callback is garbage collected."""
    from sklearn.callback._transport import _listeners, _message_consumers

    callback = ScoringMonitor(scoring="r2")
    listener_address = callback._listener_handle.address
    assert listener_address in _listeners
    assert listener_address in _message_consumers

    del callback
    assert listener_address not in _listeners
    assert listener_address not in _message_consumers
