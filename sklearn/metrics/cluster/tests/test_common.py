# [MODULE DOC]
# Common invariance tests run over every clustering metric. Metrics are registered in
# SUPERVISED_METRICS / UNSUPERVISED_METRICS dictionaries and grouped by property lists
# (symmetric, non-symmetric, normalized); module-level y1, y2 are random 3-class
# labelings of 30 samples from a seeded RandomState. Also runs array API compliance
# checks for the metrics that support it.
from functools import partial
from itertools import chain

import numpy as np
import pytest

from sklearn.metrics.cluster import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    calinski_harabasz_score,
    completeness_score,
    davies_bouldin_score,
    fowlkes_mallows_score,
    homogeneity_score,
    mutual_info_score,
    normalized_mutual_info_score,
    rand_score,
    silhouette_score,
    v_measure_score,
)
from sklearn.metrics.tests.test_common import check_array_api_metric
from sklearn.utils._array_api import (
    yield_namespace_device_dtype_combinations,
)
from sklearn.utils._testing import assert_allclose

# Dictionaries of metrics
# ------------------------
# The goal of having those dictionaries is to have an easy way to call a
# particular metric and associate a name to each function:
#   - SUPERVISED_METRICS: all supervised cluster metrics - (when given a
# ground truth value)
#   - UNSUPERVISED_METRICS: all unsupervised cluster metrics
#
# Those dictionaries will be used to test systematically some invariance
# properties, e.g. invariance toward several input layout.
#

SUPERVISED_METRICS = {
    "adjusted_mutual_info_score": adjusted_mutual_info_score,
    "adjusted_rand_score": adjusted_rand_score,
    "rand_score": rand_score,
    "completeness_score": completeness_score,
    "homogeneity_score": homogeneity_score,
    "mutual_info_score": mutual_info_score,
    "normalized_mutual_info_score": normalized_mutual_info_score,
    "v_measure_score": v_measure_score,
    "fowlkes_mallows_score": fowlkes_mallows_score,
}

UNSUPERVISED_METRICS = {
    "silhouette_score": silhouette_score,
    "silhouette_manhattan": partial(silhouette_score, metric="manhattan"),
    "calinski_harabasz_score": calinski_harabasz_score,
    "davies_bouldin_score": davies_bouldin_score,
}

# Lists of metrics with common properties
# ---------------------------------------
# Lists of metrics with common properties are used to test systematically some
# functionalities and invariance, e.g. SYMMETRIC_METRICS lists all metrics
# that are symmetric with respect to their input argument y_true and y_pred.
#
# --------------------------------------------------------------------
# Symmetric with respect to their input arguments y_true and y_pred.
# Symmetric metrics only apply to supervised clusters.
SYMMETRIC_METRICS = [
    "adjusted_rand_score",
    "rand_score",
    "v_measure_score",
    "mutual_info_score",
    "adjusted_mutual_info_score",
    "normalized_mutual_info_score",
    "fowlkes_mallows_score",
]

NON_SYMMETRIC_METRICS = ["homogeneity_score", "completeness_score"]

# Metrics whose upper bound is 1
NORMALIZED_METRICS = [
    "adjusted_rand_score",
    "rand_score",
    "homogeneity_score",
    "completeness_score",
    "v_measure_score",
    "adjusted_mutual_info_score",
    "fowlkes_mallows_score",
    "normalized_mutual_info_score",
]


rng = np.random.RandomState(0)
y1 = rng.randint(3, size=30)
y2 = rng.randint(3, size=30)


def test_symmetric_non_symmetric_union():
    # [FUNCTION DOC]
    # Verifies that the symmetric and non-symmetric lists together cover exactly the
    # supervised metrics, so no metric escapes the symmetry tests.
    assert sorted(SYMMETRIC_METRICS + NON_SYMMETRIC_METRICS) == sorted(
        SUPERVISED_METRICS
    )


@pytest.mark.parametrize(
    "metric_name, y1, y2", [(name, y1, y2) for name in SYMMETRIC_METRICS]
)
def test_symmetry(metric_name, y1, y2):
    # [FUNCTION DOC]
    # For each symmetric metric, checks that swapping the two labelings gives
    # approximately the same score.
    metric = SUPERVISED_METRICS[metric_name]
    assert metric(y1, y2) == pytest.approx(metric(y2, y1))


@pytest.mark.parametrize(
    "metric_name, y1, y2", [(name, y1, y2) for name in NON_SYMMETRIC_METRICS]
)
def test_non_symmetry(metric_name, y1, y2):
    # [FUNCTION DOC]
    # For homogeneity and completeness, checks that swapping the labelings changes the
    # score.
    metric = SUPERVISED_METRICS[metric_name]
    assert metric(y1, y2) != pytest.approx(metric(y2, y1))


@pytest.mark.parametrize("metric_name", NORMALIZED_METRICS)
def test_normalized_output(metric_name):
    # [FUNCTION DOC]
    # For metrics bounded by 1, checks scores on small imperfect labelings are strictly
    # between 0 and 1, identical labelings give 1, and a single-cluster versus
    # all-singleton pair (both orders) gives a non-negative score.
    upper_bound_1 = [0, 0, 0, 1, 1, 1]
    upper_bound_2 = [0, 0, 0, 1, 1, 1]
    metric = SUPERVISED_METRICS[metric_name]
    assert metric([0, 0, 0, 1, 1], [0, 0, 0, 1, 2]) > 0.0
    assert metric([0, 0, 1, 1, 2], [0, 0, 1, 1, 1]) > 0.0
    assert metric([0, 0, 0, 1, 2], [0, 1, 1, 1, 1]) < 1.0
    assert metric([0, 0, 0, 1, 2], [0, 1, 1, 1, 1]) < 1.0
    assert metric(upper_bound_1, upper_bound_2) == pytest.approx(1.0)

    lower_bound_1 = [0, 0, 0, 0, 0, 0]
    lower_bound_2 = [0, 1, 2, 3, 4, 5]
    score = np.array(
        [metric(lower_bound_1, lower_bound_2), metric(lower_bound_2, lower_bound_1)]
    )
    assert not (score < 0).any()


@pytest.mark.parametrize("metric_name", chain(SUPERVISED_METRICS, UNSUPERVISED_METRICS))
def test_permute_labels(metric_name):
    # [FUNCTION DOC]
    # Checks every metric is invariant to swapping label values 0 and 1: for supervised
    # metrics in either or both labelings, for unsupervised metrics in the cluster
    # labels with a random integer feature matrix.
    # Side effects: uses the global unseeded numpy random generator for X.
    # All clustering metrics do not change score due to permutations of labels
    # that is when 0 and 1 exchanged.
    y_label = np.array([0, 0, 0, 1, 1, 0, 1])
    y_pred = np.array([1, 0, 1, 0, 1, 1, 0])
    if metric_name in SUPERVISED_METRICS:
        metric = SUPERVISED_METRICS[metric_name]
        score_1 = metric(y_pred, y_label)
        assert_allclose(score_1, metric(1 - y_pred, y_label))
        assert_allclose(score_1, metric(1 - y_pred, 1 - y_label))
        assert_allclose(score_1, metric(y_pred, 1 - y_label))
    else:
        metric = UNSUPERVISED_METRICS[metric_name]
        X = np.random.randint(10, size=(7, 10))
        score_1 = metric(X, y_pred)
        assert_allclose(score_1, metric(X, 1 - y_pred))


@pytest.mark.parametrize("metric_name", chain(SUPERVISED_METRICS, UNSUPERVISED_METRICS))
# For all clustering metrics Input parameters can be both
# in the form of arrays lists, positive, negative or string
def test_format_invariance(metric_name):
    # [FUNCTION DOC]
    # Checks every metric gives identical scores whether labels are numpy int arrays,
    # int lists, string lists, object string arrays, or shifted negative/positive ints;
    # for unsupervised metrics also that integer and float X agree.
    # Side effects: uses the global unseeded numpy random generator for X.
    y_true = [0, 0, 0, 0, 1, 1, 1, 1]
    y_pred = [0, 1, 2, 3, 4, 5, 6, 7]

    def generate_formats(y):
        # [FUNCTION DOC]
        # HELPER: Generator yielding (labels, description) pairs of the same labeling in
        # six encodings: int array, int list, string list, object string array, values
        # minus one, values plus one.
        y = np.array(y)
        yield y, "array of ints"
        yield y.tolist(), "list of ints"
        yield [str(x) + "-a" for x in y.tolist()], "list of strs"
        yield (
            np.array([str(x) + "-a" for x in y.tolist()], dtype=object),
            "array of strs",
        )
        yield y - 1, "including negative ints"
        yield y + 1, "strictly positive ints"

    if metric_name in SUPERVISED_METRICS:
        metric = SUPERVISED_METRICS[metric_name]
        score_1 = metric(y_true, y_pred)
        y_true_gen = generate_formats(y_true)
        y_pred_gen = generate_formats(y_pred)
        for (y_true_fmt, fmt_name), (y_pred_fmt, _) in zip(y_true_gen, y_pred_gen):
            assert score_1 == metric(y_true_fmt, y_pred_fmt)
    else:
        metric = UNSUPERVISED_METRICS[metric_name]
        X = np.random.randint(10, size=(8, 10))
        score_1 = metric(X, y_true)
        assert score_1 == metric(X.astype(float), y_true)
        y_true_gen = generate_formats(y_true)
        for y_true_fmt, fmt_name in y_true_gen:
            assert score_1 == metric(X, y_true_fmt)


@pytest.mark.parametrize("metric", SUPERVISED_METRICS.values())
def test_single_sample(metric):
    # [FUNCTION DOC]
    # Smoke test: every supervised metric must run without error on all single-sample
    # labelings with labels 0/1.
    # only the supervised metrics support single sample
    for i, j in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        metric([i], [j])


@pytest.mark.parametrize(
    "metric_name, metric_func", dict(SUPERVISED_METRICS, **UNSUPERVISED_METRICS).items()
)
def test_inf_nan_input(metric_name, metric_func):
    # [FUNCTION DOC]
    # Checks that every metric raises ValueError mentioning NaN or infinity when the
    # label input contains inf/nan.
    # KNOWN ISSUE: the pytest.raises block wraps the whole loop, so only the first
    # invalid case is actually exercised; later cases never run.
    if metric_name in SUPERVISED_METRICS:
        invalids = [
            ([0, 1], [np.inf, np.inf]),
            ([0, 1], [np.nan, np.nan]),
            ([0, 1], [np.nan, np.inf]),
        ]
    else:
        X = np.random.randint(10, size=(2, 10))
        invalids = [(X, [np.inf, np.inf]), (X, [np.nan, np.nan]), (X, [np.nan, np.inf])]
    with pytest.raises(ValueError, match=r"contains (NaN|infinity)"):
        for args in invalids:
            metric_func(*args)


@pytest.mark.parametrize("name", chain(SUPERVISED_METRICS, UNSUPERVISED_METRICS))
def test_returned_value_consistency(name):
    # [FUNCTION DOC]
    # Checks each metric returns a built-in Python float rather than a numpy float
    # scalar, using seeded random labels and features.
    """Ensure that the returned values of all metrics are consistent.

    It can only be a float. It should not be a numpy float64 or float32.
    """

    rng = np.random.RandomState(0)
    X = rng.randint(10, size=(20, 10))
    labels_true = rng.randint(0, 3, size=(20,))
    labels_pred = rng.randint(0, 3, size=(20,))

    if name in SUPERVISED_METRICS:
        metric = SUPERVISED_METRICS[name]
        score = metric(labels_true, labels_pred)
    else:
        metric = UNSUPERVISED_METRICS[name]
        score = metric(X, labels_pred)

    assert isinstance(score, float)
    assert not isinstance(score, (np.float64, np.float32))


def check_array_api_unsupervised_metric(
    metric, array_namespace, device_name, dtype_name
):
    # [FUNCTION DOC]
    # HELPER: Array API checker for unsupervised metrics; builds a random integer X of
    # shape (7, 10) and fixed labels, then delegates to check_array_api_metric for the
    # given namespace, device and dtype.
    # Side effects: uses the global unseeded numpy random generator.
    y_pred = np.array([1, 0, 1, 0, 1, 1, 0])
    X = np.random.randint(10, size=(7, 10))

    check_array_api_metric(
        metric,
        array_namespace,
        device_name,
        dtype_name,
        a_np=X,
        b_np=y_pred,
    )


array_api_metric_checkers = {
    calinski_harabasz_score: [
        check_array_api_unsupervised_metric,
    ],
    davies_bouldin_score: [
        check_array_api_unsupervised_metric,
    ],
}


def yield_metric_checker_combinations(metric_checkers=array_api_metric_checkers):
    # [FUNCTION DOC]
    # HELPER: Yields (metric, checker) pairs flattened from a mapping of metric to list
    # of checker functions; defaults to array_api_metric_checkers.
    for metric, checkers in metric_checkers.items():
        for checker in checkers:
            yield metric, checker


@pytest.mark.parametrize(
    "array_namespace, device_name, dtype_name",
    yield_namespace_device_dtype_combinations(),
)
@pytest.mark.parametrize("metric, check_func", yield_metric_checker_combinations())
def test_array_api_compliance(
    metric, array_namespace, device_name, dtype_name, check_func
):
    # [FUNCTION DOC]
    # Runs each registered array API checker for each metric across all available
    # namespace/device/dtype combinations (unavailable ones are skipped by the checker).
    check_func(metric, array_namespace, device_name, dtype_name)
