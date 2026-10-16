# [MODULE DOC]
# Tests for ranking and threshold-based metrics in sklearn.metrics._ranking: ROC
# curve / ROC AUC (binary, partial, multiclass ovr/ovo, multilabel), auc,
# precision-recall curve and average precision, DET curve, confusion matrix and
# arbitrary metrics evaluated at thresholds, label ranking metrics (LRAP,
# coverage error, ranking loss), DCG / NDCG and top-k accuracy. Includes slow
# reference implementations used as oracles and reusable check_* helpers.
import math
import re

import numpy as np
import pytest
from scipy import stats

from sklearn import datasets
from sklearn.datasets import make_multilabel_classification
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    confusion_matrix,
    confusion_matrix_at_thresholds,
    coverage_error,
    dcg_score,
    det_curve,
    label_ranking_average_precision_score,
    label_ranking_loss,
    metric_at_thresholds,
    ndcg_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    top_k_accuracy_score,
)
from sklearn.metrics._ranking import (
    _dcg_sample_scores,
    _ndcg_sample_scores,
    _sort_inputs_and_compute_classification_thresholds,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.random_projection import _sparse_random_matrix
from sklearn.utils._testing import (
    _convert_container,
    assert_allclose,
    assert_almost_equal,
    assert_array_almost_equal,
    assert_array_equal,
)
from sklearn.utils.extmath import softmax
from sklearn.utils.fixes import CSR_CONTAINERS
from sklearn.utils.validation import (
    check_array,
    check_consistent_length,
    check_random_state,
)

###############################################################################
# Utilities for testing

CURVE_FUNCS = [
    confusion_matrix_at_thresholds,
    det_curve,
    precision_recall_curve,
    roc_curve,
]


def make_prediction(dataset=None, binary=False):
    # [FUNCTION DOC]
    # HELPER: Builds a deterministic noisy classification task (iris by default),
    # shuffles samples with seed 37, appends 200 random noise features per original
    # feature (seed 0), trains LogisticRegression on the first half and predicts the
    # second half.
    # Inputs: dataset - sklearn Bunch with data/target or None for iris; binary - if
    # True keep only classes 0 and 1 and return positive-class probabilities.
    # Outputs: (y_true, y_pred hard labels, y_score probabilities: 1d when binary,
    # (n, n_classes) otherwise) for the held-out half.
    """Make some classification predictions on a toy dataset using an SVC

    If binary is True restrict to a binary classification problem instead of a
    multiclass classification problem
    """

    if dataset is None:
        # import some data to play with
        dataset = datasets.load_iris()

    X = dataset.data
    y = dataset.target

    if binary:
        # restrict to a binary classification task
        X, y = X[y < 2], y[y < 2]

    n_samples, n_features = X.shape
    p = np.arange(n_samples)

    rng = check_random_state(37)
    rng.shuffle(p)
    X, y = X[p], y[p]
    half = int(n_samples / 2)

    # add noisy features to make the problem harder and avoid perfect results
    rng = np.random.RandomState(0)
    X = np.c_[X, rng.randn(n_samples, 200 * n_features)]

    # run classifier, get class probabilities and label predictions
    clf = LogisticRegression(random_state=0)
    y_score = clf.fit(X[:half], y[:half]).predict_proba(X[half:])

    if binary:
        # only interested in probabilities of the positive case
        # XXX: do we really want a special API for the binary case?
        y_score = y_score[:, 1]

    y_pred = clf.predict(X[half:])
    y_true = y[half:]
    return y_true, y_pred, y_score


###############################################################################
# Tests


def _auc(y_true, y_score):
    # [FUNCTION DOC]
    # HELPER: Reference ROC AUC: fraction of (positive, negative) pairs where the
    # positive sample scores strictly higher (ties count as 0, unlike roc_auc_score).
    # Inputs: y_true - 1d array with two classes (the larger is positive); y_score.
    # Outputs: float.
    """Alternative implementation to check for correctness of
    `roc_auc_score`."""
    pos_label = np.unique(y_true)[1]

    # Count the number of times positive samples are correctly ranked above
    # negative samples.
    pos = y_score[y_true == pos_label]
    neg = y_score[y_true != pos_label]
    diff_matrix = pos.reshape(1, -1) - neg.reshape(-1, 1)
    n_correct = np.sum(diff_matrix > 0)

    return n_correct / float(len(pos) * len(neg))


def _average_precision(y_true, y_score):
    # [FUNCTION DOC]
    # HELPER: Naive average precision: sort by decreasing score; for each positive at
    # position i add precision of the top i+1 items; divide by number of positives.
    # Inputs: y_true - binary 1d array (larger label positive); y_score - 1d array.
    # Outputs: float. Ties are ordered arbitrarily by argsort, so results on ties are
    # imprecise (as documented).
    """Alternative implementation to check for correctness of
    `average_precision_score`.

    Note that this implementation fails on some edge cases.
    For example, for constant predictions e.g. [0.5, 0.5, 0.5],
    y_true = [1, 0, 0] returns an average precision of 0.33...
    but y_true = [0, 0, 1] returns 1.0.
    """
    pos_label = np.unique(y_true)[1]
    n_pos = np.sum(y_true == pos_label)
    order = np.argsort(y_score)[::-1]
    y_score = y_score[order]
    y_true = y_true[order]

    score = 0
    for i in range(len(y_score)):
        if y_true[i] == pos_label:
            # Compute precision up to document i
            # i.e, percentage of relevant documents up to document i.
            prec = 0
            for j in range(0, i + 1):
                if y_true[j] == pos_label:
                    prec += 1.0
            prec /= i + 1.0
            score += prec

    return score / n_pos


def _average_precision_slow(y_true, y_score):
    # [FUNCTION DOC]
    # HELPER: Average precision as the sum over the PR curve (reversed to increasing
    # recall) of precision times the recall increment.
    # Inputs: y_true, y_score - binary labels and scores. Outputs: float.
    """A second alternative implementation of average precision that closely
    follows the Wikipedia article's definition (see References). This should
    give identical results as `average_precision_score` for all inputs.

    References
    ----------
    .. [1] `Wikipedia entry for the Average precision
       <https://en.wikipedia.org/wiki/Average_precision>`_
    """
    precision, recall, threshold = precision_recall_curve(y_true, y_score)
    precision = list(reversed(precision))
    recall = list(reversed(recall))
    average_precision = 0
    for i in range(1, len(precision)):
        average_precision += precision[i] * (recall[i] - recall[i - 1])
    return average_precision


def _partial_roc_auc_score(y_true, y_predict, max_fpr):
    # [FUNCTION DOC]
    # HELPER: Reference standardized partial ROC AUC up to max_fpr: truncates the ROC
    # curve at max_fpr (linearly interpolating TPR there), integrates it, then applies
    # McClish correction mapping the area to [0.5, 1].
    # Inputs: y_true, y_predict - binary labels and scores; max_fpr - float in (0, 1].
    # Outputs: float.
    """Alternative implementation to check for correctness of `roc_auc_score`
    with `max_fpr` set.
    """

    def _partial_roc(y_true, y_predict, max_fpr):
        # [FUNCTION DOC]
        # HELPER: Returns (fpr, tpr) points with fpr <= max_fpr plus an interpolated
        # endpoint at max_fpr between the last point inside and the first point outside.
        # KNOWN ISSUE: if no fpr exceeds max_fpr (max_fpr == 1), argmax returns 0 so the
        # interpolation uses indices -1 and 0, i.e. non-increasing x points, for
        # which np.interp behaviour is not defined; an extra point at max_fpr is added.
        fpr, tpr, _ = roc_curve(y_true, y_predict)
        new_fpr = fpr[fpr <= max_fpr]
        new_fpr = np.append(new_fpr, max_fpr)
        new_tpr = tpr[fpr <= max_fpr]
        idx_out = np.argmax(fpr > max_fpr)
        idx_in = idx_out - 1
        x_interp = [fpr[idx_in], fpr[idx_out]]
        y_interp = [tpr[idx_in], tpr[idx_out]]
        new_tpr = np.append(new_tpr, np.interp(max_fpr, x_interp, y_interp))
        return (new_fpr, new_tpr)

    new_fpr, new_tpr = _partial_roc(y_true, y_predict, max_fpr)
    partial_auc = auc(new_fpr, new_tpr)

    # Formula (5) from McClish 1989
    fpr1 = 0
    fpr2 = max_fpr
    min_area = 0.5 * (fpr2 - fpr1) * (fpr2 + fpr1)
    max_area = fpr2 - fpr1
    return 0.5 * (1 + (partial_auc - min_area) / (max_area - min_area))


@pytest.mark.parametrize("drop", [True, False])
def test_roc_curve(drop):
    # [FUNCTION DOC]
    # Verifies roc_curve AUC matches the pairwise reference _auc (2 decimals) and
    # roc_auc_score, and that fpr, tpr, thresholds have equal shapes.
    # Inputs: drop - parametrized drop_intermediate.
    # Test Area under Receiver Operating Characteristic (ROC) curve
    y_true, _, y_score = make_prediction(binary=True)
    expected_auc = _auc(y_true, y_score)

    fpr, tpr, thresholds = roc_curve(y_true, y_score, drop_intermediate=drop)
    roc_auc = auc(fpr, tpr)
    assert_array_almost_equal(roc_auc, expected_auc, decimal=2)
    assert_almost_equal(roc_auc, roc_auc_score(y_true, y_score))
    assert fpr.shape == tpr.shape
    assert fpr.shape == thresholds.shape


def test_roc_curve_end_points():
    # [FUNCTION DOC]
    # Verifies the ROC curve starts at fpr 0 and ends at fpr 1 for integer scores with
    # many ties, with consistent output shapes.
    # Make sure that roc_curve returns a curve start at 0 and ending and
    # 1 even in corner cases
    rng = np.random.RandomState(0)
    y_true = np.array([0] * 50 + [1] * 50)
    y_pred = rng.randint(3, size=100)
    fpr, tpr, thr = roc_curve(y_true, y_pred, drop_intermediate=True)
    assert fpr[0] == 0
    assert fpr[-1] == 1
    assert fpr.shape == tpr.shape
    assert fpr.shape == thr.shape


def test_roc_returns_consistency():
    # [FUNCTION DOC]
    # Verifies that returned TPR values equal the TPR recomputed by thresholding the
    # scores at each returned threshold (score >= t).
    # Test whether the returned threshold matches up with tpr
    # make small toy dataset
    y_true, _, y_score = make_prediction(binary=True)
    fpr, tpr, thresholds = roc_curve(y_true, y_score)

    # use the given thresholds to determine the tpr
    tpr_correct = []
    for t in thresholds:
        tp = np.sum((y_score >= t) & y_true)
        p = np.sum(y_true)
        tpr_correct.append(1.0 * tp / p)

    # compare tpr and tpr_correct to see if the thresholds' order was correct
    assert_array_almost_equal(tpr, tpr_correct, decimal=2)
    assert fpr.shape == tpr.shape
    assert fpr.shape == thresholds.shape


def test_roc_curve_multi():
    # [FUNCTION DOC]
    # Verifies roc_curve raises ValueError on multiclass targets.
    # roc_curve not applicable for multi-class problems
    y_true, _, y_score = make_prediction(binary=False)

    with pytest.raises(ValueError):
        roc_curve(y_true, y_score)


def test_roc_curve_confidence():
    # [FUNCTION DOC]
    # Verifies roc_curve on shifted (negative-valued) scores yields AUC about 0.90.
    # roc_curve for confidence scores
    y_true, _, y_score = make_prediction(binary=True)

    fpr, tpr, thresholds = roc_curve(y_true, y_score - 0.5)
    roc_auc = auc(fpr, tpr)
    assert_array_almost_equal(roc_auc, 0.90, decimal=2)
    assert fpr.shape == tpr.shape
    assert fpr.shape == thresholds.shape


def test_roc_curve_hard():
    # [FUNCTION DOC]
    # Verifies AUC 0.5 for constant predictions (all ones or all zeros) and about 0.78
    # for hard label predictions.
    # roc_curve for hard decisions
    y_true, pred, y_score = make_prediction(binary=True)

    # always predict one
    trivial_pred = np.ones(y_true.shape)
    fpr, tpr, thresholds = roc_curve(y_true, trivial_pred)
    roc_auc = auc(fpr, tpr)
    assert_array_almost_equal(roc_auc, 0.50, decimal=2)
    assert fpr.shape == tpr.shape
    assert fpr.shape == thresholds.shape

    # always predict zero
    trivial_pred = np.zeros(y_true.shape)
    fpr, tpr, thresholds = roc_curve(y_true, trivial_pred)
    roc_auc = auc(fpr, tpr)
    assert_array_almost_equal(roc_auc, 0.50, decimal=2)
    assert fpr.shape == tpr.shape
    assert fpr.shape == thresholds.shape

    # hard decisions
    fpr, tpr, thresholds = roc_curve(y_true, pred)
    roc_auc = auc(fpr, tpr)
    assert_array_almost_equal(roc_auc, 0.78, decimal=2)
    assert fpr.shape == tpr.shape
    assert fpr.shape == thresholds.shape


def test_roc_curve_one_label():
    # [FUNCTION DOC]
    # Verifies UndefinedMetricWarning and all-NaN fpr when y_true has no negatives,
    # and all-NaN tpr when it has no positives.
    y_true = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    y_pred = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    # assert there are warnings
    expected_message = (
        "No negative samples in y_true, false positive value should be meaningless"
    )
    with pytest.warns(UndefinedMetricWarning, match=expected_message):
        fpr, tpr, thresholds = roc_curve(y_true, y_pred)

    # all true labels, all fpr should be nan
    assert_array_equal(fpr, np.full(len(thresholds), np.nan))
    assert fpr.shape == tpr.shape
    assert fpr.shape == thresholds.shape

    # assert there are warnings
    expected_message = (
        "No positive samples in y_true, true positive value should be meaningless"
    )
    with pytest.warns(UndefinedMetricWarning, match=expected_message):
        fpr, tpr, thresholds = roc_curve([1 - x for x in y_true], y_pred)
    # all negative labels, all tpr should be nan
    assert_array_equal(tpr, np.full(len(thresholds), np.nan))
    assert fpr.shape == tpr.shape
    assert fpr.shape == thresholds.shape


def test_roc_curve_toydata():
    # [FUNCTION DOC]
    # Verifies ROC curves and AUC on tiny binary cases (perfect, inverted, ties), the
    # single-class cases (warnings, NaN AUC), and multilabel averaging (macro /
    # weighted warn for single-class columns; samples / micro values).
    # KNOWN ISSUE: the curve outputs are unpacked as tpr, fpr although roc_curve
    # returns fpr first, so variable names are swapped (assertions are consistent
    # with the actual order). The local name auc shadows the imported auc function.
    # Binary classification
    y_true = [0, 1]
    y_score = [0, 1]
    tpr, fpr, _ = roc_curve(y_true, y_score)
    roc_auc = roc_auc_score(y_true, y_score)
    assert_array_almost_equal(tpr, [0, 0, 1])
    assert_array_almost_equal(fpr, [0, 1, 1])
    assert_almost_equal(roc_auc, 1.0)

    y_true = [0, 1]
    y_score = [1, 0]
    tpr, fpr, _ = roc_curve(y_true, y_score)
    roc_auc = roc_auc_score(y_true, y_score)
    assert_array_almost_equal(tpr, [0, 1, 1])
    assert_array_almost_equal(fpr, [0, 0, 1])
    assert_almost_equal(roc_auc, 0.0)

    y_true = [1, 0]
    y_score = [1, 1]
    tpr, fpr, _ = roc_curve(y_true, y_score)
    roc_auc = roc_auc_score(y_true, y_score)
    assert_array_almost_equal(tpr, [0, 1])
    assert_array_almost_equal(fpr, [0, 1])
    assert_almost_equal(roc_auc, 0.5)

    y_true = [1, 0]
    y_score = [1, 0]
    tpr, fpr, _ = roc_curve(y_true, y_score)
    roc_auc = roc_auc_score(y_true, y_score)
    assert_array_almost_equal(tpr, [0, 0, 1])
    assert_array_almost_equal(fpr, [0, 1, 1])
    assert_almost_equal(roc_auc, 1.0)

    y_true = [1, 0]
    y_score = [0.5, 0.5]
    tpr, fpr, _ = roc_curve(y_true, y_score)
    roc_auc = roc_auc_score(y_true, y_score)
    assert_array_almost_equal(tpr, [0, 1])
    assert_array_almost_equal(fpr, [0, 1])
    assert_almost_equal(roc_auc, 0.5)

    # case with no positive samples
    y_true = [0, 0]
    y_score = [0.25, 0.75]
    # assert UndefinedMetricWarning because of no positive sample in y_true
    expected_message = (
        "No positive samples in y_true, true positive value should be meaningless"
    )
    with pytest.warns(UndefinedMetricWarning, match=expected_message):
        tpr, fpr, _ = roc_curve(y_true, y_score)
    assert_array_almost_equal(tpr, [0.0, 0.5, 1.0])
    assert_array_almost_equal(fpr, [np.nan, np.nan, np.nan])
    expected_message = (
        "Only one class is present in y_true. "
        "ROC AUC score is not defined in that case."
    )
    with pytest.warns(UndefinedMetricWarning, match=expected_message):
        auc = roc_auc_score(y_true, y_score)
    assert math.isnan(auc)

    # case with no negative samples
    y_true = [1, 1]
    y_score = [0.25, 0.75]
    # assert UndefinedMetricWarning because of no negative sample in y_true
    expected_message = (
        "No negative samples in y_true, false positive value should be meaningless"
    )
    with pytest.warns(UndefinedMetricWarning, match=expected_message):
        tpr, fpr, _ = roc_curve(y_true, y_score)
    assert_array_almost_equal(tpr, [np.nan, np.nan, np.nan])
    assert_array_almost_equal(fpr, [0.0, 0.5, 1.0])
    expected_message = (
        "Only one class is present in y_true. "
        "ROC AUC score is not defined in that case."
    )
    with pytest.warns(UndefinedMetricWarning, match=expected_message):
        auc = roc_auc_score(y_true, y_score)
    assert math.isnan(auc)

    # Multi-label classification task
    y_true = np.array([[0, 1], [0, 1]])
    y_score = np.array([[0, 1], [0, 1]])
    with pytest.warns(UndefinedMetricWarning, match=expected_message):
        roc_auc_score(y_true, y_score, average="macro")
    with pytest.warns(UndefinedMetricWarning, match=expected_message):
        roc_auc_score(y_true, y_score, average="weighted")
    assert_almost_equal(roc_auc_score(y_true, y_score, average="samples"), 1.0)
    assert_almost_equal(roc_auc_score(y_true, y_score, average="micro"), 1.0)

    y_true = np.array([[0, 1], [0, 1]])
    y_score = np.array([[0, 1], [1, 0]])
    with pytest.warns(UndefinedMetricWarning, match=expected_message):
        roc_auc_score(y_true, y_score, average="macro")
    with pytest.warns(UndefinedMetricWarning, match=expected_message):
        roc_auc_score(y_true, y_score, average="weighted")
    assert_almost_equal(roc_auc_score(y_true, y_score, average="samples"), 0.5)
    assert_almost_equal(roc_auc_score(y_true, y_score, average="micro"), 0.5)

    y_true = np.array([[1, 0], [0, 1]])
    y_score = np.array([[0, 1], [1, 0]])
    assert_almost_equal(roc_auc_score(y_true, y_score, average="macro"), 0)
    assert_almost_equal(roc_auc_score(y_true, y_score, average="weighted"), 0)
    assert_almost_equal(roc_auc_score(y_true, y_score, average="samples"), 0)
    assert_almost_equal(roc_auc_score(y_true, y_score, average="micro"), 0)

    y_true = np.array([[1, 0], [0, 1]])
    y_score = np.array([[0.5, 0.5], [0.5, 0.5]])
    assert_almost_equal(roc_auc_score(y_true, y_score, average="macro"), 0.5)
    assert_almost_equal(roc_auc_score(y_true, y_score, average="weighted"), 0.5)
    assert_almost_equal(roc_auc_score(y_true, y_score, average="samples"), 0.5)
    assert_almost_equal(roc_auc_score(y_true, y_score, average="micro"), 0.5)


def test_roc_curve_drop_intermediate():
    # [FUNCTION DOC]
    # Verifies drop_intermediate keeps only the corner thresholds, including with
    # repeated scores.
    # Test that drop_intermediate drops the correct thresholds
    y_true = [0, 0, 0, 0, 1, 1]
    y_score = [0.0, 0.2, 0.5, 0.6, 0.7, 1.0]
    tpr, fpr, thresholds = roc_curve(y_true, y_score, drop_intermediate=True)
    assert_array_almost_equal(thresholds, [np.inf, 1.0, 0.7, 0.0])

    # Test dropping thresholds with repeating scores
    y_true = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
    y_score = [0.0, 0.1, 0.6, 0.6, 0.7, 0.8, 0.9, 0.6, 0.7, 0.8, 0.9, 0.9, 1.0]
    tpr, fpr, thresholds = roc_curve(y_true, y_score, drop_intermediate=True)
    assert_array_almost_equal(thresholds, [np.inf, 1.0, 0.9, 0.7, 0.6, 0.0])


def test_roc_curve_fpr_tpr_increasing():
    # [FUNCTION DOC]
    # Verifies fpr and tpr are non-decreasing with float sample weights that could
    # introduce rounding errors.
    # Ensure that fpr and tpr returned by roc_curve are increasing.
    # Construct an edge case with float y_score and sample_weight
    # when some adjacent values of fpr and tpr are actually the same.
    y_true = [0, 0, 1, 1, 1]
    y_score = [0.1, 0.7, 0.3, 0.4, 0.5]
    sample_weight = np.repeat(0.2, 5)
    fpr, tpr, _ = roc_curve(y_true, y_score, sample_weight=sample_weight)
    assert (np.diff(fpr) < 0).sum() == 0
    assert (np.diff(tpr) < 0).sum() == 0


def test_auc():
    # [FUNCTION DOC]
    # Verifies trapezoidal auc on simple increasing and decreasing x sequences.
    # Test Area Under Curve (AUC) computation
    x = [0, 1]
    y = [0, 1]
    assert_array_almost_equal(auc(x, y), 0.5)
    x = [1, 0]
    y = [0, 1]
    assert_array_almost_equal(auc(x, y), 0.5)
    x = [1, 0, 0]
    y = [0, 1, 1]
    assert_array_almost_equal(auc(x, y), 0.5)
    x = [0, 1]
    y = [1, 1]
    assert_array_almost_equal(auc(x, y), 1)
    x = [0, 0.5, 1]
    y = [0, 0.5, 1]
    assert_array_almost_equal(auc(x, y), 0.5)


def test_auc_errors():
    # [FUNCTION DOC]
    # Verifies auc raises ValueError on mismatched shapes, fewer than 2 points, and
    # non-monotonic x (with the exact message).
    # Incompatible shapes
    with pytest.raises(ValueError):
        auc([0.0, 0.5, 1.0], [0.1, 0.2])

    # Too few x values
    with pytest.raises(ValueError):
        auc([0.0], [0.1])

    # x is not in order
    x = [2, 1, 3, 4]
    y = [5, 6, 7, 8]
    error_message = "x is neither increasing nor decreasing : {}".format(np.array(x))
    with pytest.raises(ValueError, match=re.escape(error_message)):
        auc(x, y)


@pytest.mark.parametrize(
    "y_true, labels",
    [
        (np.array([0, 1, 0, 2]), [0, 1, 2]),
        (np.array([0, 1, 0, 2]), None),
        (["a", "b", "a", "c"], ["a", "b", "c"]),
        (["a", "b", "a", "c"], None),
    ],
)
def test_multiclass_ovo_roc_auc_toydata(y_true, labels):
    # [FUNCTION DOC]
    # Verifies one-vs-one multiclass ROC AUC (macro and prevalence-weighted) against
    # manual pairwise binary AUC averages, for int / string labels with or without
    # explicit labels, and NotImplementedError for average None.
    # Inputs: y_true, labels - parametrized.
    # Tests the one-vs-one multiclass ROC AUC algorithm
    # on a small example, representative of an expected use case.
    y_scores = np.array(
        [[0.1, 0.8, 0.1], [0.3, 0.4, 0.3], [0.35, 0.5, 0.15], [0, 0.2, 0.8]]
    )

    # Used to compute the expected output.
    # Consider labels 0 and 1:
    # positive label is 0, negative label is 1
    score_01 = roc_auc_score([1, 0, 1], [0.1, 0.3, 0.35])
    # positive label is 1, negative label is 0
    score_10 = roc_auc_score([0, 1, 0], [0.8, 0.4, 0.5])
    average_score_01 = (score_01 + score_10) / 2

    # Consider labels 0 and 2:
    score_02 = roc_auc_score([1, 1, 0], [0.1, 0.35, 0])
    score_20 = roc_auc_score([0, 0, 1], [0.1, 0.15, 0.8])
    average_score_02 = (score_02 + score_20) / 2

    # Consider labels 1 and 2:
    score_12 = roc_auc_score([1, 0], [0.4, 0.2])
    score_21 = roc_auc_score([0, 1], [0.3, 0.8])
    average_score_12 = (score_12 + score_21) / 2

    # Unweighted, one-vs-one multiclass ROC AUC algorithm
    ovo_unweighted_score = (average_score_01 + average_score_02 + average_score_12) / 3
    assert_almost_equal(
        roc_auc_score(y_true, y_scores, labels=labels, multi_class="ovo"),
        ovo_unweighted_score,
    )

    # Weighted, one-vs-one multiclass ROC AUC algorithm
    # Each term is weighted by the prevalence for the positive label.
    pair_scores = [average_score_01, average_score_02, average_score_12]
    prevalence = [0.75, 0.75, 0.50]
    ovo_weighted_score = np.average(pair_scores, weights=prevalence)
    assert_almost_equal(
        roc_auc_score(
            y_true, y_scores, labels=labels, multi_class="ovo", average="weighted"
        ),
        ovo_weighted_score,
    )

    # Check that average=None raises NotImplemented error
    error_message = "average=None is not implemented for multi_class='ovo'."
    with pytest.raises(NotImplementedError, match=error_message):
        roc_auc_score(y_true, y_scores, labels=labels, multi_class="ovo", average=None)


@pytest.mark.parametrize(
    "y_true, labels",
    [
        (np.array([0, 2, 0, 2]), [0, 1, 2]),
        (np.array(["a", "d", "a", "d"]), ["a", "b", "d"]),
    ],
)
def test_multiclass_ovo_roc_auc_toydata_binary(y_true, labels):
    # [FUNCTION DOC]
    # Verifies ovo ROC AUC when y_true contains only two of the declared labels equals
    # the averaged pairwise AUC of those two, for both macro and weighted averages.
    # Inputs: y_true, labels - parametrized.
    # Tests the one-vs-one multiclass ROC AUC algorithm for binary y_true
    #
    # on a small example, representative of an expected use case.
    y_scores = np.array(
        [[0.2, 0.0, 0.8], [0.6, 0.0, 0.4], [0.55, 0.0, 0.45], [0.4, 0.0, 0.6]]
    )

    # Used to compute the expected output.
    # Consider labels 0 and 1:
    # positive label is 0, negative label is 1
    score_01 = roc_auc_score([1, 0, 1, 0], [0.2, 0.6, 0.55, 0.4])
    # positive label is 1, negative label is 0
    score_10 = roc_auc_score([0, 1, 0, 1], [0.8, 0.4, 0.45, 0.6])
    ovo_score = (score_01 + score_10) / 2

    assert_almost_equal(
        roc_auc_score(y_true, y_scores, labels=labels, multi_class="ovo"), ovo_score
    )

    # Weighted, one-vs-one multiclass ROC AUC algorithm
    assert_almost_equal(
        roc_auc_score(
            y_true, y_scores, labels=labels, multi_class="ovo", average="weighted"
        ),
        ovo_score,
    )


@pytest.mark.parametrize(
    "y_true, labels",
    [
        (np.array([0, 1, 2, 2]), None),
        (["a", "b", "c", "c"], None),
        ([0, 1, 2, 2], [0, 1, 2]),
        (["a", "b", "c", "c"], ["a", "b", "c"]),
    ],
)
def test_multiclass_ovr_roc_auc_toydata(y_true, labels):
    # [FUNCTION DOC]
    # Verifies one-vs-rest multiclass ROC AUC per class (average None), macro and
    # prevalence-weighted averages against manually computed binary AUCs.
    # Inputs: y_true, labels - parametrized.
    # Tests the unweighted, one-vs-rest multiclass ROC AUC algorithm
    # on a small example, representative of an expected use case.
    y_scores = np.array(
        [[1.0, 0.0, 0.0], [0.1, 0.5, 0.4], [0.1, 0.1, 0.8], [0.3, 0.3, 0.4]]
    )
    # Compute the expected result by individually computing the 'one-vs-rest'
    # ROC AUC scores for classes 0, 1, and 2.
    out_0 = roc_auc_score([1, 0, 0, 0], y_scores[:, 0])
    out_1 = roc_auc_score([0, 1, 0, 0], y_scores[:, 1])
    out_2 = roc_auc_score([0, 0, 1, 1], y_scores[:, 2])
    assert_almost_equal(
        roc_auc_score(y_true, y_scores, multi_class="ovr", labels=labels, average=None),
        [out_0, out_1, out_2],
    )

    # Compute unweighted results (default behaviour is average="macro")
    result_unweighted = (out_0 + out_1 + out_2) / 3.0
    assert_almost_equal(
        roc_auc_score(y_true, y_scores, multi_class="ovr", labels=labels),
        result_unweighted,
    )

    # Tests the weighted, one-vs-rest multiclass ROC AUC algorithm
    # on the same input (Provost & Domingos, 2000)
    result_weighted = out_0 * 0.25 + out_1 * 0.25 + out_2 * 0.5
    assert_almost_equal(
        roc_auc_score(
            y_true, y_scores, multi_class="ovr", labels=labels, average="weighted"
        ),
        result_weighted,
    )


@pytest.mark.parametrize(
    "multi_class, average",
    [
        ("ovr", "macro"),
        ("ovr", "micro"),
        ("ovo", "macro"),
    ],
)
def test_perfect_imperfect_chance_multiclass_roc_auc(multi_class, average):
    # [FUNCTION DOC]
    # Verifies multiclass ROC AUC is 1 for a perfect ranking, < 1 for an imperfect
    # one, and 0.5 for uniform scores.
    # Inputs: multi_class, average - parametrized.
    # KNOWN ISSUE: comment says chance level is 5.0; the asserted value is 0.5.
    y_true = np.array([3, 1, 2, 0])

    # Perfect classifier (from a ranking point of view) has roc_auc_score = 1.0
    y_perfect = [
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.75, 0.05, 0.05, 0.15],
    ]
    assert_almost_equal(
        roc_auc_score(y_true, y_perfect, multi_class=multi_class, average=average),
        1.0,
    )

    # Imperfect classifier has roc_auc_score < 1.0
    y_imperfect = [
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    assert (
        roc_auc_score(y_true, y_imperfect, multi_class=multi_class, average=average)
        < 1.0
    )

    # Chance level classifier has roc_auc_score = 5.0
    y_chance = 0.25 * np.ones((4, 4))
    assert roc_auc_score(
        y_true, y_chance, multi_class=multi_class, average=average
    ) == pytest.approx(0.5)


def test_micro_averaged_ovr_roc_auc(global_random_seed):
    # [FUNCTION DOC]
    # Verifies micro-averaged ovr ROC AUC equals the AUC of the ROC curve on the
    # flattened one-hot targets and scores (Dirichlet-sampled predictions).
    # Inputs: global_random_seed fixture.
    seed = global_random_seed
    # Let's generate a set of random predictions and matching true labels such
    # that the predictions are not perfect. To make the problem more interesting,
    # we use an imbalanced class distribution (by using different parameters
    # in the Dirichlet prior (conjugate prior of the multinomial distribution).
    y_pred = stats.dirichlet.rvs([2.0, 1.0, 0.5], size=1000, random_state=seed)
    y_true = np.asarray(
        [
            stats.multinomial.rvs(n=1, p=y_pred_i, random_state=seed).argmax()
            for y_pred_i in y_pred
        ]
    )
    y_onehot = label_binarize(y_true, classes=[0, 1, 2])
    fpr, tpr, _ = roc_curve(y_onehot.ravel(), y_pred.ravel())
    roc_auc_by_hand = auc(fpr, tpr)
    roc_auc_auto = roc_auc_score(y_true, y_pred, multi_class="ovr", average="micro")
    assert roc_auc_by_hand == pytest.approx(roc_auc_auto)


@pytest.mark.parametrize(
    "msg, y_true, labels",
    [
        ("Parameter 'labels' must be unique", np.array([0, 1, 2, 2]), [0, 2, 0]),
        (
            "Parameter 'labels' must be unique",
            np.array(["a", "b", "c", "c"]),
            ["a", "a", "b"],
        ),
        (
            (
                "Number of classes in y_true not equal to the number of columns "
                "in 'y_score'"
            ),
            np.array([0, 2, 0, 2]),
            None,
        ),
        (
            "Parameter 'labels' must be ordered",
            np.array(["a", "b", "c", "c"]),
            ["a", "c", "b"],
        ),
        (
            (
                "Number of given labels, 2, not equal to the number of columns in "
                "'y_score', 3"
            ),
            np.array([0, 1, 2, 2]),
            [0, 1],
        ),
        (
            (
                "Number of given labels, 2, not equal to the number of columns in "
                "'y_score', 3"
            ),
            np.array(["a", "b", "c", "c"]),
            ["a", "b"],
        ),
        (
            (
                "Number of given labels, 4, not equal to the number of columns in "
                "'y_score', 3"
            ),
            np.array([0, 1, 2, 2]),
            [0, 1, 2, 3],
        ),
        (
            (
                "Number of given labels, 4, not equal to the number of columns in "
                "'y_score', 3"
            ),
            np.array(["a", "b", "c", "c"]),
            ["a", "b", "c", "d"],
        ),
        (
            "'y_true' contains labels not in parameter 'labels'",
            np.array(["a", "b", "c", "e"]),
            ["a", "b", "c"],
        ),
        (
            "'y_true' contains labels not in parameter 'labels'",
            np.array(["a", "b", "c", "d"]),
            ["a", "b", "c"],
        ),
        (
            "'y_true' contains labels not in parameter 'labels'",
            np.array([0, 1, 2, 3]),
            [0, 1, 2],
        ),
    ],
)
@pytest.mark.parametrize("multi_class", ["ovo", "ovr"])
def test_roc_auc_score_multiclass_labels_error(msg, y_true, labels, multi_class):
    # [FUNCTION DOC]
    # Verifies ValueError messages for invalid labels (non-unique, unordered, wrong
    # count, missing y_true labels) or class / column mismatch in multiclass ROC AUC.
    # Inputs: msg, y_true, labels, multi_class - parametrized.
    y_scores = np.array(
        [[0.1, 0.8, 0.1], [0.3, 0.4, 0.3], [0.35, 0.5, 0.15], [0, 0.2, 0.8]]
    )

    with pytest.raises(ValueError, match=msg):
        roc_auc_score(y_true, y_scores, labels=labels, multi_class=multi_class)


@pytest.mark.parametrize(
    "msg, kwargs",
    [
        (
            (
                r"average must be one of \('macro', 'weighted', None\) for "
                r"multiclass problems"
            ),
            {"average": "samples", "multi_class": "ovo"},
        ),
        (
            (
                r"average must be one of \('micro', 'macro', 'weighted', None\) for "
                r"multiclass problems"
            ),
            {"average": "samples", "multi_class": "ovr"},
        ),
        (
            (
                r"sample_weight is not supported for multiclass one-vs-one "
                r"ROC AUC, 'sample_weight' must be None in this case"
            ),
            {"multi_class": "ovo", "sample_weight": []},
        ),
        (
            (
                r"Partial AUC computation not available in multiclass setting, "
                r"'max_fpr' must be set to `None`, received `max_fpr=0.5` "
                r"instead"
            ),
            {"multi_class": "ovo", "max_fpr": 0.5},
        ),
        (r"multi_class must be in \('ovo', 'ovr'\)", {}),
    ],
)
def test_roc_auc_score_multiclass_error(msg, kwargs):
    # [FUNCTION DOC]
    # Verifies ValueError for unsupported multiclass options: samples average,
    # sample_weight with ovo, max_fpr, and missing multi_class.
    # Inputs: msg, kwargs - parametrized.
    # Test that roc_auc_score function returns an error when trying
    # to compute multiclass AUC for parameters where an output
    # is not defined.
    rng = check_random_state(404)
    y_score = rng.rand(20, 3)
    y_prob = softmax(y_score)
    y_true = rng.randint(0, 3, size=20)
    with pytest.raises(ValueError, match=msg):
        roc_auc_score(y_true, y_prob, **kwargs)


def test_auc_score_non_binary_class():
    # [FUNCTION DOC]
    # Verifies roc_auc_score warns UndefinedMetricWarning when y_true contains a single
    # class (0, 1 or -1).
    # KNOWN ISSUE: the comment says an error is returned; the test checks a warning.
    # Test that roc_auc_score function returns an error when trying
    # to compute AUC for non-binary class values.
    rng = check_random_state(404)
    y_pred = rng.rand(10)
    # y_true contains only one class value
    y_true = np.zeros(10, dtype="int")
    warn_message = (
        "Only one class is present in y_true. "
        "ROC AUC score is not defined in that case."
    )
    with pytest.warns(UndefinedMetricWarning, match=warn_message):
        roc_auc_score(y_true, y_pred)
    y_true = np.ones(10, dtype="int")
    with pytest.warns(UndefinedMetricWarning, match=warn_message):
        roc_auc_score(y_true, y_pred)
    y_true = np.full(10, -1, dtype="int")
    with pytest.warns(UndefinedMetricWarning, match=warn_message):
        roc_auc_score(y_true, y_pred)


def test_sort_inputs_and_compute_classification_thresholds_input_validation():
    # [FUNCTION DOC]
    # Verifies the private helper raises ValueError on inconsistent lengths and NaN
    # scores.
    """Test `_sort_inputs_and_compute_classification_thresholds` input validation."""
    # Inconsistent lengths
    y_true = np.array([0, 1, 0])
    y_score = np.array([0.1, 0.9])

    with pytest.raises(ValueError, match="inconsistent numbers of samples"):
        _sort_inputs_and_compute_classification_thresholds(y_true, y_score)

    # Non-finite value
    y_true = np.array([0, 1, 0, 1])
    y_score = np.array([0.1, np.nan, 0.3, 0.7])

    with pytest.raises(ValueError, match="Input.*contains NaN"):
        _sort_inputs_and_compute_classification_thresholds(y_true, y_score)


def test_sort_inputs_and_compute_classification_thresholds_zero_weights():
    # [FUNCTION DOC]
    # Verifies zero-weight samples are dropped, weight output is None without weights,
    # and all-zero weights raise ValueError.
    """Test zero weights in `_sort_inputs_and_compute_classification_thresholds`."""
    y_true = np.array([0, 1, 0, 1, 0, 1])
    y_score = np.array([0.1, 0.9, 0.3, 0.7, 0.5, 0.2])
    # Indices 0 and 4 zero weight
    sample_weight = np.array([0.0, 2.0, 1.0, 1.5, 0.0, 0.8])

    y_true_sorted, y_score_sorted, weight_sorted, _ = (
        _sort_inputs_and_compute_classification_thresholds(
            y_true, y_score, sample_weight
        )
    )

    assert len(y_true_sorted) == len(y_score_sorted) == len(weight_sorted) == 4
    assert 0.1 not in y_score_sorted
    assert 0.5 not in y_score_sorted

    # Check default `sample_weight=None` gives None
    _, _, weight, _ = _sort_inputs_and_compute_classification_thresholds(
        y_true, y_score
    )
    assert weight is None

    # All zero weights raises error
    y_true = np.array([0, 1, 0])
    y_score = np.array([0.1, 0.9, 0.3])
    sample_weight = np.array([0.0, 0.0, 0.0])

    with pytest.raises(ValueError, match="Sample weights must contain at least"):
        _sort_inputs_and_compute_classification_thresholds(
            y_true, y_score, sample_weight
        )


def test_sort_inputs_and_compute_classification_thresholds_sorting():
    # [FUNCTION DOC]
    # Verifies stable descending sort of scores with aligned labels and weights, the
    # threshold indices (last index of each distinct score), and a single threshold
    # index for constant scores.
    """Test sorting in `_sort_inputs_and_compute_classification_thresholds`."""
    y_true = np.array([0, 1, 0, 1, 1])
    y_score = np.array([0.1, 0.9, 0.3, 0.7, 0.3])
    sample_weight = np.array([1.0, 2.0, 1.5, 0.5, 0.3])

    y_true_sorted, y_score_sorted, weight_sorted, threshold_idxs = (
        _sort_inputs_and_compute_classification_thresholds(
            y_true, y_score, sample_weight
        )
    )
    # Check descending sort
    assert np.all(y_score_sorted[:-1] >= y_score_sorted[1:])
    assert_array_equal(weight_sorted, np.array([2.0, 0.5, 1.5, 0.3, 1.0]))
    assert_array_equal(threshold_idxs, np.array([0, 1, 3, 4]))
    # Check stable sort
    assert_array_equal(y_score_sorted[2:4], [0.3, 0.3])
    assert_array_equal(y_true_sorted[2:4], [0, 1])

    # All identical scores
    y_score_same = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
    _, _, _, threshold_idxs = _sort_inputs_and_compute_classification_thresholds(
        y_true, y_score_same
    )
    # Threshold is the final index
    assert_array_equal(threshold_idxs, np.array([4]))


def test_confusion_matrix_at_thresholds(global_random_seed):
    # [FUNCTION DOC]
    # Smoke test: output arrays have equal length and tp+fn equals positives, tn+fp
    # equals negatives, and all four sum to n_samples at every threshold.
    # Inputs: global_random_seed fixture.
    """Smoke test for confusion_matrix_at_thresholds."""
    rng = np.random.RandomState(global_random_seed)

    n_samples = 100
    y_true = rng.randint(0, 2, size=100)
    y_score = rng.uniform(size=100)

    n_pos = np.sum(y_true)
    n_neg = n_samples - n_pos

    tns, fps, fns, tps, thresholds = confusion_matrix_at_thresholds(y_true, y_score)

    assert len(tns) == len(fps) == len(fns) == len(tps) == len(thresholds)
    assert_allclose(tps + fns, n_pos)
    assert_allclose(tns + fps, n_neg)
    assert_allclose(tns + fps + fns + tps, n_samples)


@pytest.mark.parametrize("curve_func", CURVE_FUNCS)
def test_confusion_matrix_at_thresholds_multiclass_error(curve_func):
    # [FUNCTION DOC]
    # Verifies every curve function rejects multiclass y_true with ValueError.
    # Inputs: curve_func - parametrized.
    rng = check_random_state(404)
    y_true = rng.randint(0, 3, size=10)
    y_pred = rng.rand(10)
    msg = "multiclass format is not supported"
    with pytest.raises(ValueError, match=msg):
        curve_func(y_true, y_pred)


@pytest.mark.parametrize("curve_func", CURVE_FUNCS)
def test_confusion_matrix_at_thresholds_implicit_pos_label(curve_func):
    # [FUNCTION DOC]
    # Verifies string labels without pos_label raise an informative ValueError for
    # unicode and object dtypes, and float 0/1 labels behave like integer labels.
    # Inputs: curve_func - parametrized.
    # Check that using string class labels raises an informative
    # error for any supported string dtype:
    msg = (
        "y_true takes value in {'a', 'b'} and pos_label is "
        "not specified: either make y_true take "
        "value in {0, 1} or {-1, 1} or pass pos_label "
        "explicitly."
    )
    with pytest.raises(ValueError, match=msg):
        curve_func(np.array(["a", "b"], dtype="<U1"), [0.0, 1.0])

    with pytest.raises(ValueError, match=msg):
        curve_func(np.array(["a", "b"], dtype=object), [0.0, 1.0])

    # Check that it is possible to use floating point class labels
    # that are interpreted similarly to integer class labels:
    y_pred = [0.0, 1.0, 0.2, 0.42]
    int_curve = curve_func([0, 1, 1, 0], y_pred)
    float_curve = curve_func([0.0, 1.0, 1.0, 0.0], y_pred)
    for int_curve_part, float_curve_part in zip(int_curve, float_curve):
        np.testing.assert_allclose(int_curve_part, float_curve_part)


@pytest.mark.filterwarnings("ignore:Support for labels represented as bytes")
@pytest.mark.parametrize("curve_func", [precision_recall_curve, roc_curve])
@pytest.mark.parametrize("labels_type", ["list", "array"])
def test_confusion_matrix_at_thresholds_implicit_bytes_pos_label(
    curve_func, labels_type
):
    # [FUNCTION DOC]
    # Verifies bytes labels (list or array) raise TypeError.
    # Inputs: curve_func, labels_type - parametrized.
    # Check that using bytes class labels raises an informative
    # error for any supported string dtype:
    labels = _convert_container([b"a", b"b"], labels_type)
    msg = "Support for labels represented as bytes is not supported"
    with pytest.raises(TypeError, match=msg):
        curve_func(labels, [0.0, 1.0])


@pytest.mark.parametrize("curve_func", CURVE_FUNCS)
def test_confusion_matrix_at_thresholds_zero_sample_weight(curve_func):
    # [FUNCTION DOC]
    # Verifies a zero-weight sample gives the same curves as removing it.
    # Inputs: curve_func - parametrized.
    y_true = [0, 0, 1, 1, 1]
    y_score = [0.1, 0.2, 0.3, 0.4, 0.5]
    sample_weight = [1, 1, 1, 0.5, 0]

    result_1 = curve_func(y_true, y_score, sample_weight=sample_weight)
    result_2 = curve_func(y_true[:-1], y_score[:-1], sample_weight=sample_weight[:-1])

    for arr_1, arr_2 in zip(result_1, result_2):
        assert_allclose(arr_1, arr_2)


@pytest.mark.parametrize("drop", [True, False])
def test_precision_recall_curve(drop):
    # [FUNCTION DOC]
    # Verifies PR curve properties via _test_precision_recall_curve, the first point
    # (precision = class prior, recall 1) on an unbalanced subset, invariance to
    # {-1, 1} labels without mutating y_true, and exact curves on a toy example with
    # and without drop_intermediate.
    # Inputs: drop - parametrized.
    # Side effects: mutates the local y_true (0 -> -1) intentionally.
    y_true, _, y_score = make_prediction(binary=True)
    _test_precision_recall_curve(y_true, y_score, drop)

    # Make sure the first point of the Precision-Recall on the right is:
    # (p=1.0, r=class balance) on a non-balanced dataset [1:]
    p, r, t = precision_recall_curve(y_true[1:], y_score[1:], drop_intermediate=drop)
    assert r[0] == 1.0
    assert p[0] == y_true[1:].mean()

    # Use {-1, 1} for labels; make sure original labels aren't modified
    y_true[np.where(y_true == 0)] = -1
    y_true_copy = y_true.copy()
    _test_precision_recall_curve(y_true, y_score, drop)
    assert_array_equal(y_true_copy, y_true)

    labels = [1, 0, 0, 1]
    predict_probas = [1, 2, 3, 4]
    p, r, t = precision_recall_curve(labels, predict_probas, drop_intermediate=drop)
    if drop:
        assert_allclose(p, [0.5, 0.33333333, 1.0, 1.0])
        assert_allclose(r, [1.0, 0.5, 0.5, 0.0])
        assert_allclose(t, [1, 2, 4])
    else:
        assert_allclose(p, [0.5, 0.33333333, 0.5, 1.0, 1.0])
        assert_allclose(r, [1.0, 0.5, 0.5, 0.5, 0.0])
        assert_allclose(t, [1, 2, 3, 4])
    assert p.size == r.size
    assert p.size == t.size + 1


def _test_precision_recall_curve(y_true, y_score, drop):
    # [FUNCTION DOC]
    # HELPER: Checks average precision from the slow reference equals about 0.869 and
    # average_precision_score, that the naive reference agrees to 2 decimals, shape
    # relations (precision/recall have one more element than thresholds), and the
    # constant-score smoke case.
    # Inputs: y_true, y_score - binary prediction data; drop - drop_intermediate flag.
    # Test Precision-Recall and area under PR curve
    p, r, thresholds = precision_recall_curve(y_true, y_score, drop_intermediate=drop)
    precision_recall_auc = _average_precision_slow(y_true, y_score)
    assert_array_almost_equal(precision_recall_auc, 0.869, 3)
    assert_array_almost_equal(
        precision_recall_auc, average_precision_score(y_true, y_score)
    )
    # `_average_precision` is not very precise in case of 0.5 ties: be tolerant
    assert_almost_equal(
        _average_precision(y_true, y_score), precision_recall_auc, decimal=2
    )
    assert p.size == r.size
    assert p.size == thresholds.size + 1
    # Smoke test in the case of proba having only one value
    p, r, thresholds = precision_recall_curve(
        y_true, np.zeros_like(y_score), drop_intermediate=drop
    )
    assert p.size == r.size
    assert p.size == thresholds.size + 1


@pytest.mark.parametrize("drop", [True, False])
def test_precision_recall_curve_toydata(drop):
    # [FUNCTION DOC]
    # Verifies PR curves and average precision on tiny binary cases (with numpy errors
    # raised as exceptions), the no-positive case warnings and values, and multilabel
    # macro / weighted / samples / micro averages including degenerate columns.
    # Inputs: drop - parametrized.
    with np.errstate(all="raise"):
        # Binary classification
        y_true = [0, 1]
        y_score = [0, 1]
        p, r, _ = precision_recall_curve(y_true, y_score, drop_intermediate=drop)
        auc_prc = average_precision_score(y_true, y_score)
        assert_array_almost_equal(p, [0.5, 1, 1])
        assert_array_almost_equal(r, [1, 1, 0])
        assert_almost_equal(auc_prc, 1.0)

        y_true = [0, 1]
        y_score = [1, 0]
        p, r, _ = precision_recall_curve(y_true, y_score, drop_intermediate=drop)
        auc_prc = average_precision_score(y_true, y_score)
        assert_array_almost_equal(p, [0.5, 0.0, 1.0])
        assert_array_almost_equal(r, [1.0, 0.0, 0.0])
        # Here we are doing a terrible prediction: we are always getting
        # it wrong, hence the average_precision_score is the accuracy at
        # chance: 50%
        assert_almost_equal(auc_prc, 0.5)

        y_true = [1, 0]
        y_score = [1, 1]
        p, r, _ = precision_recall_curve(y_true, y_score, drop_intermediate=drop)
        auc_prc = average_precision_score(y_true, y_score)
        assert_array_almost_equal(p, [0.5, 1])
        assert_array_almost_equal(r, [1.0, 0])
        assert_almost_equal(auc_prc, 0.5)

        y_true = [1, 0]
        y_score = [1, 0]
        p, r, _ = precision_recall_curve(y_true, y_score, drop_intermediate=drop)
        auc_prc = average_precision_score(y_true, y_score)
        assert_array_almost_equal(p, [0.5, 1, 1])
        assert_array_almost_equal(r, [1, 1, 0])
        assert_almost_equal(auc_prc, 1.0)

        y_true = [1, 0]
        y_score = [0.5, 0.5]
        p, r, _ = precision_recall_curve(y_true, y_score, drop_intermediate=drop)
        auc_prc = average_precision_score(y_true, y_score)
        assert_array_almost_equal(p, [0.5, 1])
        assert_array_almost_equal(r, [1, 0.0])
        assert_almost_equal(auc_prc, 0.5)

        y_true = [0, 0]
        y_score = [0.25, 0.75]
        with pytest.warns(UserWarning, match="No positive class found in y_true"):
            p, r, _ = precision_recall_curve(y_true, y_score, drop_intermediate=drop)
        with pytest.warns(UserWarning, match="No positive class found in y_true"):
            auc_prc = average_precision_score(y_true, y_score)
        assert_allclose(p, [0, 0, 1])
        assert_allclose(r, [1, 1, 0])
        assert_allclose(auc_prc, 0)

        y_true = [1, 1]
        y_score = [0.25, 0.75]
        p, r, _ = precision_recall_curve(y_true, y_score, drop_intermediate=drop)
        assert_almost_equal(average_precision_score(y_true, y_score), 1.0)
        assert_array_almost_equal(p, [1.0, 1.0, 1.0])
        assert_array_almost_equal(r, [1, 0.5, 0.0])

        # Multi-label classification task
        y_true = np.array([[0, 1], [0, 1]])
        y_score = np.array([[0, 1], [0, 1]])
        with pytest.warns(UserWarning, match="No positive class found in y_true"):
            assert_allclose(
                average_precision_score(y_true, y_score, average="macro"), 0.5
            )
        with pytest.warns(UserWarning, match="No positive class found in y_true"):
            assert_allclose(
                average_precision_score(y_true, y_score, average="weighted"), 1.0
            )
        assert_allclose(
            average_precision_score(y_true, y_score, average="samples"), 1.0
        )
        assert_allclose(average_precision_score(y_true, y_score, average="micro"), 1.0)

        y_true = np.array([[0, 1], [0, 1]])
        y_score = np.array([[0, 1], [1, 0]])
        with pytest.warns(UserWarning, match="No positive class found in y_true"):
            assert_allclose(
                average_precision_score(y_true, y_score, average="macro"), 0.5
            )
        with pytest.warns(UserWarning, match="No positive class found in y_true"):
            assert_allclose(
                average_precision_score(y_true, y_score, average="weighted"), 1.0
            )
        assert_allclose(
            average_precision_score(y_true, y_score, average="samples"), 0.75
        )
        assert_allclose(average_precision_score(y_true, y_score, average="micro"), 0.5)

        y_true = np.array([[1, 0], [0, 1]])
        y_score = np.array([[0, 1], [1, 0]])
        assert_almost_equal(
            average_precision_score(y_true, y_score, average="macro"), 0.5
        )
        assert_almost_equal(
            average_precision_score(y_true, y_score, average="weighted"), 0.5
        )
        assert_almost_equal(
            average_precision_score(y_true, y_score, average="samples"), 0.5
        )
        assert_almost_equal(
            average_precision_score(y_true, y_score, average="micro"), 0.5
        )

        y_true = np.array([[0, 0], [0, 0]])
        y_score = np.array([[0, 1], [0, 1]])
        with pytest.warns(UserWarning, match="No positive class found in y_true"):
            assert_allclose(
                average_precision_score(y_true, y_score, average="macro"), 0.0
            )
        assert_allclose(
            average_precision_score(y_true, y_score, average="weighted"), 0.0
        )
        with pytest.warns(UserWarning, match="No positive class found in y_true"):
            assert_allclose(
                average_precision_score(y_true, y_score, average="samples"), 0.0
            )
        with pytest.warns(UserWarning, match="No positive class found in y_true"):
            assert_allclose(
                average_precision_score(y_true, y_score, average="micro"), 0.0
            )

        y_true = np.array([[1, 1], [1, 1]])
        y_score = np.array([[0, 1], [0, 1]])
        assert_allclose(average_precision_score(y_true, y_score, average="macro"), 1.0)
        assert_allclose(
            average_precision_score(y_true, y_score, average="weighted"), 1.0
        )
        assert_allclose(
            average_precision_score(y_true, y_score, average="samples"), 1.0
        )
        assert_allclose(average_precision_score(y_true, y_score, average="micro"), 1.0)

        y_true = np.array([[1, 0], [0, 1]])
        y_score = np.array([[0.5, 0.5], [0.5, 0.5]])
        assert_almost_equal(
            average_precision_score(y_true, y_score, average="macro"), 0.5
        )
        assert_almost_equal(
            average_precision_score(y_true, y_score, average="weighted"), 0.5
        )
        assert_almost_equal(
            average_precision_score(y_true, y_score, average="samples"), 0.5
        )
        assert_almost_equal(
            average_precision_score(y_true, y_score, average="micro"), 0.5
        )

    with np.errstate(all="ignore"):
        # if one class is never present weighted should not be NaN
        y_true = np.array([[0, 0], [0, 1]])
        y_score = np.array([[0, 0], [0, 1]])
        with pytest.warns(UserWarning, match="No positive class found in y_true"):
            assert_allclose(
                average_precision_score(y_true, y_score, average="weighted"), 1
            )


def test_precision_recall_curve_drop_intermediate():
    # [FUNCTION DOC]
    # Verifies drop_intermediate keeps only needed thresholds, keeps endpoints when
    # all labels are negative, and keeps all thresholds when all are positive.
    """Check the behaviour of the `drop_intermediate` parameter."""
    y_true = [0, 0, 0, 0, 1, 1]
    y_score = [0.0, 0.2, 0.5, 0.6, 0.7, 1.0]
    precision, recall, thresholds = precision_recall_curve(
        y_true, y_score, drop_intermediate=True
    )
    assert_allclose(thresholds, [0.0, 0.7, 1.0])

    # Test dropping thresholds with repeating scores
    y_true = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
    y_score = [0.0, 0.1, 0.6, 0.6, 0.7, 0.8, 0.9, 0.6, 0.7, 0.8, 0.9, 0.9, 1.0]
    precision, recall, thresholds = precision_recall_curve(
        y_true, y_score, drop_intermediate=True
    )
    assert_allclose(thresholds, [0.0, 0.6, 0.7, 0.8, 0.9, 1.0])

    # Test all false keeps only endpoints
    y_true = [0, 0, 0, 0]
    y_score = [0.0, 0.1, 0.2, 0.3]
    precision, recall, thresholds = precision_recall_curve(
        y_true, y_score, drop_intermediate=True
    )
    assert_allclose(thresholds, [0.0, 0.3])

    # Test all true keeps all thresholds
    y_true = [1, 1, 1, 1]
    y_score = [0.0, 0.1, 0.2, 0.3]
    precision, recall, thresholds = precision_recall_curve(
        y_true, y_score, drop_intermediate=True
    )
    assert_allclose(thresholds, [0.0, 0.1, 0.2, 0.3])


def test_average_precision_constant_values():
    # [FUNCTION DOC]
    # Verifies a constant predictor's average precision equals the positive rate.
    # Check the average_precision_score of a constant predictor is
    # the TPR

    # Generate a dataset with 25% of positives
    y_true = np.zeros(100, dtype=int)
    y_true[::4] = 1
    # And a constant score
    y_score = np.ones(100)
    # The precision is then the fraction of positive whatever the recall
    # is, as there is only one threshold:
    assert average_precision_score(y_true, y_score) == 0.25


def test_average_precision_score_binary_pos_label_errors():
    # [FUNCTION DOC]
    # Verifies ValueError when pos_label is not present in binary y_true.
    # Raise an error when pos_label is not in binary y_true
    y_true = np.array([0, 1])
    y_pred = np.array([0, 1])
    err_msg = re.escape("pos_label=2 is not a valid label. It should be one of [0 1]")
    with pytest.raises(ValueError, match=err_msg):
        average_precision_score(y_true, y_pred, pos_label=2)


def test_average_precision_score_multilabel_pos_label_errors():
    # [FUNCTION DOC]
    # Verifies ValueError for pos_label other than 1 with multilabel-indicator y_true.
    # Raise an error for multilabel-indicator y_true with
    # pos_label other than 1
    y_true = np.array([[1, 0], [0, 1], [0, 1], [1, 0]])
    y_pred = np.array([[0.9, 0.1], [0.1, 0.9], [0.8, 0.2], [0.2, 0.8]])
    err_msg = (
        "Parameter pos_label is fixed to 1 for multilabel-indicator y_true. "
        "Do not set pos_label or set pos_label to 1."
    )
    with pytest.raises(ValueError, match=err_msg):
        average_precision_score(y_true, y_pred, pos_label=0)


def test_average_precision_score_multiclass_pos_label_errors():
    # [FUNCTION DOC]
    # Verifies ValueError for pos_label other than 1 with multiclass y_true.
    # Raise an error for multiclass y_true with pos_label other than 1
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_score = np.array(
        [
            [0.5, 0.2, 0.1],
            [0.4, 0.5, 0.3],
            [0.1, 0.2, 0.6],
            [0.2, 0.3, 0.5],
            [0.2, 0.3, 0.5],
            [0.2, 0.3, 0.5],
        ]
    )
    err_msg = (
        "Parameter pos_label is fixed to 1 for multiclass y_true. "
        "Do not set pos_label or set pos_label to 1."
    )
    with pytest.raises(ValueError, match=err_msg):
        average_precision_score(y_true, y_score, pos_label=3)


def test_multiclass_ranking_metrics_raise_for_incorrect_shape_of_y_score():
    # [FUNCTION DOC]
    # Verifies average precision, ovr ROC AUC and top-k accuracy raise ValueError for
    # a 1d y_score with multiclass y_true.
    """Test ranking metrics, with multiclass support, raise if shape `y_score` is 1D."""
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_score = np.array([0.5, 0.4, 0.8, 0.9, 0.8, 0.7])

    msg = re.escape("`y_score` needs to be of shape `(n_samples, n_classes)`")
    with pytest.raises(ValueError, match=msg):
        average_precision_score(y_true, y_score)
    with pytest.raises(ValueError, match=msg):
        roc_auc_score(y_true, y_score, multi_class="ovr")
    with pytest.raises(ValueError, match=msg):
        top_k_accuracy_score(y_true, y_score)


def test_score_scale_invariance():
    # [FUNCTION DOC]
    # Verifies ROC AUC and average precision are exactly unchanged by scaling scores
    # up / down and shifting them.
    # Test that average_precision_score and roc_auc_score are invariant by
    # the scaling or shifting of probabilities
    # This test was expanded (added scaled_down) in response to github
    # issue #3864 (and others), where overly aggressive rounding was causing
    # problems for users with very small y_score values
    y_true, _, y_score = make_prediction(binary=True)

    roc_auc = roc_auc_score(y_true, y_score)
    roc_auc_scaled_up = roc_auc_score(y_true, 100 * y_score)
    roc_auc_scaled_down = roc_auc_score(y_true, 1e-6 * y_score)
    roc_auc_shifted = roc_auc_score(y_true, y_score - 10)
    assert roc_auc == roc_auc_scaled_up
    assert roc_auc == roc_auc_scaled_down
    assert roc_auc == roc_auc_shifted

    pr_auc = average_precision_score(y_true, y_score)
    pr_auc_scaled_up = average_precision_score(y_true, 100 * y_score)
    pr_auc_scaled_down = average_precision_score(y_true, 1e-6 * y_score)
    pr_auc_shifted = average_precision_score(y_true, y_score - 10)
    assert pr_auc == pr_auc_scaled_up
    assert pr_auc == pr_auc_scaled_down
    assert pr_auc == pr_auc_shifted


@pytest.mark.parametrize(
    "y_true,y_score,expected_fpr,expected_fnr",
    [
        ([0, 0, 1], [0, 0.5, 1], [0], [0]),
        ([0, 0, 1], [0, 0.25, 0.5], [0], [0]),
        ([0, 0, 1], [0.5, 0.75, 1], [0], [0]),
        ([0, 0, 1], [0.25, 0.5, 0.75], [0], [0]),
        ([0, 1, 0], [0, 0.5, 1], [0.5, 0.5, 0], [0, 1, 1]),
        ([0, 1, 0], [0, 0.25, 0.5], [0.5, 0.5, 0], [0, 1, 1]),
        ([0, 1, 0], [0.5, 0.75, 1], [0.5, 0.5, 0], [0, 1, 1]),
        ([0, 1, 0], [0.25, 0.5, 0.75], [0.5, 0.5, 0], [0, 1, 1]),
        ([0, 1, 1], [0, 0.5, 1], [0.0], [0]),
        ([0, 1, 1], [0, 0.25, 0.5], [0], [0]),
        ([0, 1, 1], [0.5, 0.75, 1], [0], [0]),
        ([0, 1, 1], [0.25, 0.5, 0.75], [0], [0]),
        ([1, 0, 0], [0, 0.5, 1], [1, 1, 0.5, 0], [0, 1, 1, 1]),
        ([1, 0, 0], [0, 0.25, 0.5], [1, 1, 0.5, 0], [0, 1, 1, 1]),
        ([1, 0, 0], [0.5, 0.75, 1], [1, 1, 0.5, 0], [0, 1, 1, 1]),
        ([1, 0, 0], [0.25, 0.5, 0.75], [1, 1, 0.5, 0], [0, 1, 1, 1]),
        ([1, 0, 1], [0, 0.5, 1], [1, 1, 0], [0, 0.5, 0.5]),
        ([1, 0, 1], [0, 0.25, 0.5], [1, 1, 0], [0, 0.5, 0.5]),
        ([1, 0, 1], [0.5, 0.75, 1], [1, 1, 0], [0, 0.5, 0.5]),
        ([1, 0, 1], [0.25, 0.5, 0.75], [1, 1, 0], [0, 0.5, 0.5]),
    ],
)
def test_det_curve_toydata(y_true, y_score, expected_fpr, expected_fnr):
    # [FUNCTION DOC]
    # Verifies DET curve fpr / fnr on small examples and increasing thresholds.
    # Inputs: y_true, y_score, expected_fpr, expected_fnr - parametrized.
    # Check on a batch of small examples.
    fpr, fnr, thresholds = det_curve(y_true, y_score)

    assert_allclose(fpr, expected_fpr)
    assert_allclose(fnr, expected_fnr)
    assert np.all(np.diff(thresholds) >= 0)


@pytest.mark.parametrize(
    ["y_true", "y_score", "expected_fpr", "expected_fnr", "drop_intermediate"],
    [
        # drop when true positives do not change from the previous or subsequent point
        ([1, 0, 0], [0, 0.5, 1], [1, 1, 0.5, 0.0], [0, 1, 1, 1], False),
        ([1, 0, 0], [0, 0.5, 1], [1, 1, 0.0], [0, 1, 1], True),
        ([1, 0, 0], [0, 0.25, 0.5], [1, 1, 0.5, 0.0], [0, 1, 1, 1], False),
        ([1, 0, 0], [0, 0.25, 0.5], [1, 1, 0.0], [0, 1, 1], True),
        # do nothing otherwise
        ([1, 0, 1], [0, 0.5, 1], [1, 1, 0], [0, 0.5, 0.5], False),
        ([1, 0, 1], [0, 0.5, 1], [1, 1, 0], [0, 0.5, 0.5], True),
        ([1, 0, 1], [0, 0.25, 0.5], [1, 1, 0], [0, 0.5, 0.5], False),
        ([1, 0, 1], [0, 0.25, 0.5], [1, 1, 0], [0, 0.5, 0.5], True),
    ],
)
def test_det_curve_drop_intermediate(
    y_true, y_score, expected_fpr, expected_fnr, drop_intermediate
):
    # [FUNCTION DOC]
    # Verifies drop_intermediate removes collinear DET points only where tp does not
    # change.
    # Inputs: parametrized data and drop_intermediate flag.
    # Check on a batch of small examples.
    fpr, fnr, _ = det_curve(y_true, y_score, drop_intermediate=drop_intermediate)

    assert_allclose(fpr, expected_fpr)
    assert_allclose(fnr, expected_fnr)


@pytest.mark.parametrize(
    "y_true,y_score,expected_fpr,expected_fnr",
    [
        ([1, 0], [0.5, 0.5], [1, 0], [0, 1]),
        ([0, 1], [0.5, 0.5], [1, 0], [0, 1]),
        ([0, 0, 1], [0.25, 0.5, 0.5], [0.5, 0], [0, 1]),
        ([0, 1, 0], [0.25, 0.5, 0.5], [0.5, 0], [0, 1]),
        ([0, 1, 1], [0.25, 0.5, 0.5], [0], [0]),
        ([1, 0, 0], [0.25, 0.5, 0.5], [1, 1, 0], [0, 1, 1]),
        ([1, 0, 1], [0.25, 0.5, 0.5], [1, 1, 0], [0, 0.5, 1]),
        ([1, 1, 0], [0.25, 0.5, 0.5], [1, 1, 0], [0, 0.5, 1]),
    ],
)
def test_det_curve_tie_handling(y_true, y_score, expected_fpr, expected_fnr):
    # [FUNCTION DOC]
    # Verifies DET curve outputs with tied scores.
    # Inputs: parametrized data and expected curves.
    fpr, fnr, _ = det_curve(y_true, y_score)

    assert_allclose(fpr, expected_fpr)
    assert_allclose(fnr, expected_fnr)


def test_det_curve_sanity_check():
    # [FUNCTION DOC]
    # Verifies duplicating every sample leaves the DET curve unchanged.
    # Exactly duplicated inputs yield the same result.
    assert_allclose(
        det_curve([0, 0, 1], [0, 0.5, 1]),
        det_curve([0, 0, 0, 0, 1, 1], [0, 0, 0.5, 0.5, 1, 1]),
    )


@pytest.mark.parametrize("y_score", [(0), (0.25), (0.5), (0.75), (1)])
def test_det_curve_constant_scores(y_score):
    # [FUNCTION DOC]
    # Verifies DET curve for constant scores is fpr [1, 0], fnr [0, 1], thresholds
    # [score, inf].
    # Inputs: y_score - parametrized constant.
    fpr, fnr, threshold = det_curve(
        y_true=[0, 1, 0, 1, 0, 1], y_score=np.full(6, y_score)
    )

    assert_allclose(fpr, [1, 0])
    assert_allclose(fnr, [0, 1])
    assert_allclose(threshold, [y_score, np.inf])


@pytest.mark.parametrize(
    "y_true",
    [
        ([0, 0, 0, 0, 0, 1]),
        ([0, 0, 0, 0, 1, 1]),
        ([0, 0, 0, 1, 1, 1]),
        ([0, 0, 1, 1, 1, 1]),
        ([0, 1, 1, 1, 1, 1]),
    ],
)
def test_det_curve_perfect_scores(y_true):
    # [FUNCTION DOC]
    # Verifies perfect scores give a single point fpr 0, fnr 0.
    # Inputs: y_true - parametrized.
    fpr, fnr, _ = det_curve(y_true=y_true, y_score=y_true)

    assert_allclose(fpr, [0])
    assert_allclose(fnr, [0])


@pytest.mark.parametrize(
    "y_true, y_pred, err_msg",
    [
        ([0, 1], [0, 0.5, 1], "inconsistent numbers of samples"),
        ([0, 1, 1], [0, 0.5], "inconsistent numbers of samples"),
        ([0, 0, 0], [0, 0.5, 1], "Only one class is present in y_true"),
        ([1, 1, 1], [0, 0.5, 1], "Only one class is present in y_true"),
        (
            ["cancer", "cancer", "not cancer"],
            [0.2, 0.3, 0.8],
            "pos_label is not specified",
        ),
    ],
)
def test_det_curve_bad_input(y_true, y_pred, err_msg):
    # [FUNCTION DOC]
    # Verifies ValueError for inconsistent lengths, single-class y_true and string
    # labels without pos_label.
    # Inputs: y_true, y_pred, err_msg - parametrized.
    # input variables with inconsistent numbers of samples
    with pytest.raises(ValueError, match=err_msg):
        det_curve(y_true, y_pred)


def test_det_curve_pos_label():
    # [FUNCTION DOC]
    # Verifies pos_label changes the first threshold and that swapping the positive
    # class (with complementary scores) swaps and reverses fpr / fnr.
    y_true = ["cancer"] * 3 + ["not cancer"] * 7
    y_pred_pos_not_cancer = np.array([0.1, 0.4, 0.6, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.9])
    y_pred_pos_cancer = 1 - y_pred_pos_not_cancer

    fpr_pos_cancer, fnr_pos_cancer, th_pos_cancer = det_curve(
        y_true,
        y_pred_pos_cancer,
        pos_label="cancer",
    )
    fpr_pos_not_cancer, fnr_pos_not_cancer, th_pos_not_cancer = det_curve(
        y_true,
        y_pred_pos_not_cancer,
        pos_label="not cancer",
    )

    # check that the first threshold will change depending which label we
    # consider positive
    assert th_pos_cancer[0] == pytest.approx(0.4)
    assert th_pos_not_cancer[0] == pytest.approx(0.2)

    # check for the symmetry of the fpr and fnr
    assert_allclose(fpr_pos_cancer, fnr_pos_not_cancer[::-1])
    assert_allclose(fnr_pos_cancer, fpr_pos_not_cancer[::-1])


def check_lrap_toy(lrap_score):
    # [FUNCTION DOC]
    # HELPER: Checks an LRAP implementation on many hand-computed single-sample cases,
    # including tie handling.
    # Inputs: lrap_score - callable(y_true, y_score) -> float.
    # Check on several small example that it works
    assert_almost_equal(lrap_score([[0, 1]], [[0.25, 0.75]]), 1)
    assert_almost_equal(lrap_score([[0, 1]], [[0.75, 0.25]]), 1 / 2)
    assert_almost_equal(lrap_score([[1, 1]], [[0.75, 0.25]]), 1)

    assert_almost_equal(lrap_score([[0, 0, 1]], [[0.25, 0.5, 0.75]]), 1)
    assert_almost_equal(lrap_score([[0, 1, 0]], [[0.25, 0.5, 0.75]]), 1 / 2)
    assert_almost_equal(lrap_score([[0, 1, 1]], [[0.25, 0.5, 0.75]]), 1)
    assert_almost_equal(lrap_score([[1, 0, 0]], [[0.25, 0.5, 0.75]]), 1 / 3)
    assert_almost_equal(
        lrap_score([[1, 0, 1]], [[0.25, 0.5, 0.75]]), (2 / 3 + 1 / 1) / 2
    )
    assert_almost_equal(
        lrap_score([[1, 1, 0]], [[0.25, 0.5, 0.75]]), (2 / 3 + 1 / 2) / 2
    )

    assert_almost_equal(lrap_score([[0, 0, 1]], [[0.75, 0.5, 0.25]]), 1 / 3)
    assert_almost_equal(lrap_score([[0, 1, 0]], [[0.75, 0.5, 0.25]]), 1 / 2)
    assert_almost_equal(
        lrap_score([[0, 1, 1]], [[0.75, 0.5, 0.25]]), (1 / 2 + 2 / 3) / 2
    )
    assert_almost_equal(lrap_score([[1, 0, 0]], [[0.75, 0.5, 0.25]]), 1)
    assert_almost_equal(lrap_score([[1, 0, 1]], [[0.75, 0.5, 0.25]]), (1 + 2 / 3) / 2)
    assert_almost_equal(lrap_score([[1, 1, 0]], [[0.75, 0.5, 0.25]]), 1)
    assert_almost_equal(lrap_score([[1, 1, 1]], [[0.75, 0.5, 0.25]]), 1)

    assert_almost_equal(lrap_score([[0, 0, 1]], [[0.5, 0.75, 0.25]]), 1 / 3)
    assert_almost_equal(lrap_score([[0, 1, 0]], [[0.5, 0.75, 0.25]]), 1)
    assert_almost_equal(lrap_score([[0, 1, 1]], [[0.5, 0.75, 0.25]]), (1 + 2 / 3) / 2)
    assert_almost_equal(lrap_score([[1, 0, 0]], [[0.5, 0.75, 0.25]]), 1 / 2)
    assert_almost_equal(
        lrap_score([[1, 0, 1]], [[0.5, 0.75, 0.25]]), (1 / 2 + 2 / 3) / 2
    )
    assert_almost_equal(lrap_score([[1, 1, 0]], [[0.5, 0.75, 0.25]]), 1)
    assert_almost_equal(lrap_score([[1, 1, 1]], [[0.5, 0.75, 0.25]]), 1)

    # Tie handling
    assert_almost_equal(lrap_score([[1, 0]], [[0.5, 0.5]]), 0.5)
    assert_almost_equal(lrap_score([[0, 1]], [[0.5, 0.5]]), 0.5)
    assert_almost_equal(lrap_score([[1, 1]], [[0.5, 0.5]]), 1)

    assert_almost_equal(lrap_score([[0, 0, 1]], [[0.25, 0.5, 0.5]]), 0.5)
    assert_almost_equal(lrap_score([[0, 1, 0]], [[0.25, 0.5, 0.5]]), 0.5)
    assert_almost_equal(lrap_score([[0, 1, 1]], [[0.25, 0.5, 0.5]]), 1)
    assert_almost_equal(lrap_score([[1, 0, 0]], [[0.25, 0.5, 0.5]]), 1 / 3)
    assert_almost_equal(
        lrap_score([[1, 0, 1]], [[0.25, 0.5, 0.5]]), (2 / 3 + 1 / 2) / 2
    )
    assert_almost_equal(
        lrap_score([[1, 1, 0]], [[0.25, 0.5, 0.5]]), (2 / 3 + 1 / 2) / 2
    )
    assert_almost_equal(lrap_score([[1, 1, 1]], [[0.25, 0.5, 0.5]]), 1)

    assert_almost_equal(lrap_score([[1, 1, 0]], [[0.5, 0.5, 0.5]]), 2 / 3)

    assert_almost_equal(lrap_score([[1, 1, 1, 0]], [[0.5, 0.5, 0.5, 0.5]]), 3 / 4)


def check_zero_or_all_relevant_labels(lrap_score):
    # [FUNCTION DOC]
    # HELPER: Checks LRAP is 1 when a sample has no or all relevant labels (random and
    # tied scores) and for the single-label degenerate case.
    # Inputs: lrap_score - callable.
    random_state = check_random_state(0)

    for n_labels in range(2, 5):
        y_score = random_state.uniform(size=(1, n_labels))
        y_score_ties = np.zeros_like(y_score)

        # No relevant labels
        y_true = np.zeros((1, n_labels))
        assert lrap_score(y_true, y_score) == 1.0
        assert lrap_score(y_true, y_score_ties) == 1.0

        # Only relevant labels
        y_true = np.ones((1, n_labels))
        assert lrap_score(y_true, y_score) == 1.0
        assert lrap_score(y_true, y_score_ties) == 1.0

    # Degenerate case: only one label
    assert_almost_equal(
        lrap_score([[1], [0], [1], [0]], [[0.5], [0.5], [0.5], [0.5]]), 1.0
    )


def check_lrap_error_raised(lrap_score):
    # [FUNCTION DOC]
    # HELPER: Checks an LRAP implementation raises ValueError for non-multilabel
    # targets and mismatching y_true / y_score shapes.
    # Inputs: lrap_score - callable.
    # Raise value error if not appropriate format
    with pytest.raises(ValueError):
        lrap_score([0, 1, 0], [0.25, 0.3, 0.2])
    with pytest.raises(ValueError):
        lrap_score([0, 1, 2], [[0.25, 0.75, 0.0], [0.7, 0.3, 0.0], [0.8, 0.2, 0.0]])
    with pytest.raises(ValueError):
        lrap_score(
            [(0), (1), (2)], [[0.25, 0.75, 0.0], [0.7, 0.3, 0.0], [0.8, 0.2, 0.0]]
        )

    # Check that y_true.shape != y_score.shape raise the proper exception
    with pytest.raises(ValueError):
        lrap_score([[0, 1], [0, 1]], [0, 1])
    with pytest.raises(ValueError):
        lrap_score([[0, 1], [0, 1]], [[0, 1]])
    with pytest.raises(ValueError):
        lrap_score([[0, 1], [0, 1]], [[0], [1]])
    with pytest.raises(ValueError):
        lrap_score([[0, 1]], [[0, 1], [0, 1]])
    with pytest.raises(ValueError):
        lrap_score([[0], [1]], [[0, 1], [0, 1]])
    with pytest.raises(ValueError):
        lrap_score([[0, 1], [0, 1]], [[0], [1]])


def check_lrap_only_ties(lrap_score):
    # [FUNCTION DOC]
    # HELPER: Checks that with all scores tied, LRAP equals n_relevant / n_labels for
    # every contiguous block of relevant labels.
    # Inputs: lrap_score - callable.
    # Check tie handling in score
    # Basic check with only ties and increasing label space
    for n_labels in range(2, 10):
        y_score = np.ones((1, n_labels))

        # Check for growing number of consecutive relevant
        for n_relevant in range(1, n_labels):
            # Check for a bunch of positions
            for pos in range(n_labels - n_relevant):
                y_true = np.zeros((1, n_labels))
                y_true[0, pos : pos + n_relevant] = 1
                assert_almost_equal(lrap_score(y_true, y_score), n_relevant / n_labels)


def check_lrap_without_tie_and_increasing_score(lrap_score):
    # [FUNCTION DOC]
    # HELPER: Checks LRAP for strictly decreasing scores against closed-form values for
    # first-and-last relevant labels and every contiguous block of relevant labels.
    # Inputs: lrap_score - callable.
    # KNOWN ISSUE: the name says increasing score but the scores are decreasing.
    # Check that Label ranking average precision works for various
    # Basic check with increasing label space size and decreasing score
    for n_labels in range(2, 10):
        y_score = n_labels - (np.arange(n_labels).reshape((1, n_labels)) + 1)

        # First and last
        y_true = np.zeros((1, n_labels))
        y_true[0, 0] = 1
        y_true[0, -1] = 1
        assert_almost_equal(lrap_score(y_true, y_score), (2 / n_labels + 1) / 2)

        # Check for growing number of consecutive relevant label
        for n_relevant in range(1, n_labels):
            # Check for a bunch of position
            for pos in range(n_labels - n_relevant):
                y_true = np.zeros((1, n_labels))
                y_true[0, pos : pos + n_relevant] = 1
                assert_almost_equal(
                    lrap_score(y_true, y_score),
                    sum(
                        (r + 1) / ((pos + r + 1) * n_relevant)
                        for r in range(n_relevant)
                    ),
                )


def _my_lrap(y_true, y_score):
    # [FUNCTION DOC]
    # HELPER: Reference LRAP: for each sample, compute tie-aware ranks (tied labels get
    # the worst rank of their group); if no or all labels are relevant the sample
    # scores 1; otherwise average over relevant labels of (relevant labels ranked at
    # or above it) / its rank. Returns the mean over samples.
    # Inputs: y_true - (n_samples, n_labels) binary; y_score - same shape.
    # Outputs: float.
    # Exceptions: ValueError from validation on inconsistent lengths / bad arrays.
    """Simple implementation of label ranking average precision"""
    check_consistent_length(y_true, y_score)
    y_true = check_array(y_true)
    y_score = check_array(y_score)
    n_samples, n_labels = y_true.shape
    score = np.empty((n_samples,))
    for i in range(n_samples):
        # The best rank correspond to 1. Rank higher than 1 are worse.
        # The best inverse ranking correspond to n_labels.
        unique_rank, inv_rank = np.unique(y_score[i], return_inverse=True)
        n_ranks = unique_rank.size
        rank = n_ranks - inv_rank

        # Rank need to be corrected to take into account ties
        # ex: rank 1 ex aequo means that both label are rank 2.
        corr_rank = np.bincount(rank, minlength=n_ranks + 1).cumsum()
        rank = corr_rank[rank]

        relevant = y_true[i].nonzero()[0]
        if relevant.size == 0 or relevant.size == n_labels:
            score[i] = 1
            continue

        score[i] = 0.0
        for label in relevant:
            # Let's count the number of relevant label with better rank
            # (smaller rank).
            n_ranked_above = sum(rank[r] <= rank[label] for r in relevant)

            # Weight by the rank of the actual label
            score[i] += n_ranked_above / rank[label]

        score[i] /= relevant.size

    return score.mean()


def check_alternative_lrap_implementation(
    lrap_score, n_classes=5, n_samples=20, random_state=0
):
    # [FUNCTION DOC]
    # HELPER: Compares label_ranking_average_precision_score with _my_lrap on random
    # multilabel targets using sparse-random (tied) scores and uniform scores.
    # Inputs: lrap_score (unused), n_classes, n_samples, random_state.
    # KNOWN ISSUE: the lrap_score argument is ignored; the public function is always
    # used directly.
    _, y_true = make_multilabel_classification(
        n_features=1,
        allow_unlabeled=False,
        random_state=random_state,
        n_classes=n_classes,
        n_samples=n_samples,
    )

    # Score with ties
    y_score = _sparse_random_matrix(
        n_components=y_true.shape[0],
        n_features=y_true.shape[1],
        random_state=random_state,
    )

    if hasattr(y_score, "toarray"):
        y_score = y_score.toarray()
    score_lrap = label_ranking_average_precision_score(y_true, y_score)
    score_my_lrap = _my_lrap(y_true, y_score)
    assert_almost_equal(score_lrap, score_my_lrap)

    # Uniform score
    random_state = check_random_state(random_state)
    y_score = random_state.uniform(size=(n_samples, n_classes))
    score_lrap = label_ranking_average_precision_score(y_true, y_score)
    score_my_lrap = _my_lrap(y_true, y_score)
    assert_almost_equal(score_lrap, score_my_lrap)


@pytest.mark.parametrize(
    "check",
    (
        check_lrap_toy,
        check_lrap_without_tie_and_increasing_score,
        check_lrap_only_ties,
        check_zero_or_all_relevant_labels,
    ),
)
@pytest.mark.parametrize("func", (label_ranking_average_precision_score, _my_lrap))
def test_label_ranking_avp(check, func):
    # [FUNCTION DOC]
    # Runs each LRAP check helper against both the public implementation and _my_lrap.
    # Inputs: check, func - parametrized.
    check(func)


def test_lrap_error_raised():
    # [FUNCTION DOC]
    # Runs check_lrap_error_raised on the public LRAP function.
    check_lrap_error_raised(label_ranking_average_precision_score)


@pytest.mark.parametrize("n_samples", (1, 2, 8, 20))
@pytest.mark.parametrize("n_classes", (2, 5, 10))
@pytest.mark.parametrize("random_state", range(1))
def test_alternative_lrap_implementation(n_samples, n_classes, random_state):
    # [FUNCTION DOC]
    # Runs check_alternative_lrap_implementation for several sizes.
    # Inputs: n_samples, n_classes, random_state - parametrized.
    check_alternative_lrap_implementation(
        label_ranking_average_precision_score, n_classes, n_samples, random_state
    )


def test_lrap_sample_weighting_zero_labels():
    # [FUNCTION DOC]
    # Verifies weighted LRAP equals the weighted mean of per-sample APs including a
    # sample with no relevant labels (AP 1) given weight 0.
    # Degenerate sample labeling (e.g., zero labels for a sample) is a valid
    # special case for lrap (the sample is considered to achieve perfect
    # precision), but this case is not tested in test_common.
    # For these test samples, the APs are 0.5, 0.75, and 1.0 (default for zero
    # labels).
    y_true = np.array([[1, 0, 0, 0], [1, 0, 0, 1], [0, 0, 0, 0]], dtype=bool)
    y_score = np.array(
        [[0.3, 0.4, 0.2, 0.1], [0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]]
    )
    samplewise_lraps = np.array([0.5, 0.75, 1.0])
    sample_weight = np.array([1.0, 1.0, 0.0])

    assert_almost_equal(
        label_ranking_average_precision_score(
            y_true, y_score, sample_weight=sample_weight
        ),
        np.sum(sample_weight * samplewise_lraps) / np.sum(sample_weight),
    )


def test_coverage_error():
    # [FUNCTION DOC]
    # Verifies coverage_error on many hand-computed toy cases and multi-sample cases.
    # Toy case
    assert_almost_equal(coverage_error([[0, 1]], [[0.25, 0.75]]), 1)
    assert_almost_equal(coverage_error([[0, 1]], [[0.75, 0.25]]), 2)
    assert_almost_equal(coverage_error([[1, 1]], [[0.75, 0.25]]), 2)
    assert_almost_equal(coverage_error([[0, 0]], [[0.75, 0.25]]), 0)

    assert_almost_equal(coverage_error([[0, 0, 0]], [[0.25, 0.5, 0.75]]), 0)
    assert_almost_equal(coverage_error([[0, 0, 1]], [[0.25, 0.5, 0.75]]), 1)
    assert_almost_equal(coverage_error([[0, 1, 0]], [[0.25, 0.5, 0.75]]), 2)
    assert_almost_equal(coverage_error([[0, 1, 1]], [[0.25, 0.5, 0.75]]), 2)
    assert_almost_equal(coverage_error([[1, 0, 0]], [[0.25, 0.5, 0.75]]), 3)
    assert_almost_equal(coverage_error([[1, 0, 1]], [[0.25, 0.5, 0.75]]), 3)
    assert_almost_equal(coverage_error([[1, 1, 0]], [[0.25, 0.5, 0.75]]), 3)
    assert_almost_equal(coverage_error([[1, 1, 1]], [[0.25, 0.5, 0.75]]), 3)

    assert_almost_equal(coverage_error([[0, 0, 0]], [[0.75, 0.5, 0.25]]), 0)
    assert_almost_equal(coverage_error([[0, 0, 1]], [[0.75, 0.5, 0.25]]), 3)
    assert_almost_equal(coverage_error([[0, 1, 0]], [[0.75, 0.5, 0.25]]), 2)
    assert_almost_equal(coverage_error([[0, 1, 1]], [[0.75, 0.5, 0.25]]), 3)
    assert_almost_equal(coverage_error([[1, 0, 0]], [[0.75, 0.5, 0.25]]), 1)
    assert_almost_equal(coverage_error([[1, 0, 1]], [[0.75, 0.5, 0.25]]), 3)
    assert_almost_equal(coverage_error([[1, 1, 0]], [[0.75, 0.5, 0.25]]), 2)
    assert_almost_equal(coverage_error([[1, 1, 1]], [[0.75, 0.5, 0.25]]), 3)

    assert_almost_equal(coverage_error([[0, 0, 0]], [[0.5, 0.75, 0.25]]), 0)
    assert_almost_equal(coverage_error([[0, 0, 1]], [[0.5, 0.75, 0.25]]), 3)
    assert_almost_equal(coverage_error([[0, 1, 0]], [[0.5, 0.75, 0.25]]), 1)
    assert_almost_equal(coverage_error([[0, 1, 1]], [[0.5, 0.75, 0.25]]), 3)
    assert_almost_equal(coverage_error([[1, 0, 0]], [[0.5, 0.75, 0.25]]), 2)
    assert_almost_equal(coverage_error([[1, 0, 1]], [[0.5, 0.75, 0.25]]), 3)
    assert_almost_equal(coverage_error([[1, 1, 0]], [[0.5, 0.75, 0.25]]), 2)
    assert_almost_equal(coverage_error([[1, 1, 1]], [[0.5, 0.75, 0.25]]), 3)

    # Non trivial case
    assert_almost_equal(
        coverage_error([[0, 1, 0], [1, 1, 0]], [[0.1, 10.0, -3], [0, 1, 3]]),
        (1 + 3) / 2.0,
    )

    assert_almost_equal(
        coverage_error(
            [[0, 1, 0], [1, 1, 0], [0, 1, 1]], [[0.1, 10, -3], [0, 1, 3], [0, 2, 0]]
        ),
        (1 + 3 + 3) / 3.0,
    )

    assert_almost_equal(
        coverage_error(
            [[0, 1, 0], [1, 1, 0], [0, 1, 1]], [[0.1, 10, -3], [3, 1, 3], [0, 2, 0]]
        ),
        (1 + 3 + 3) / 3.0,
    )


def test_coverage_tie_handling():
    # [FUNCTION DOC]
    # Verifies coverage_error with tied scores counts all tied labels.
    assert_almost_equal(coverage_error([[0, 0]], [[0.5, 0.5]]), 0)
    assert_almost_equal(coverage_error([[1, 0]], [[0.5, 0.5]]), 2)
    assert_almost_equal(coverage_error([[0, 1]], [[0.5, 0.5]]), 2)
    assert_almost_equal(coverage_error([[1, 1]], [[0.5, 0.5]]), 2)

    assert_almost_equal(coverage_error([[0, 0, 0]], [[0.25, 0.5, 0.5]]), 0)
    assert_almost_equal(coverage_error([[0, 0, 1]], [[0.25, 0.5, 0.5]]), 2)
    assert_almost_equal(coverage_error([[0, 1, 0]], [[0.25, 0.5, 0.5]]), 2)
    assert_almost_equal(coverage_error([[0, 1, 1]], [[0.25, 0.5, 0.5]]), 2)
    assert_almost_equal(coverage_error([[1, 0, 0]], [[0.25, 0.5, 0.5]]), 3)
    assert_almost_equal(coverage_error([[1, 0, 1]], [[0.25, 0.5, 0.5]]), 3)
    assert_almost_equal(coverage_error([[1, 1, 0]], [[0.25, 0.5, 0.5]]), 3)
    assert_almost_equal(coverage_error([[1, 1, 1]], [[0.25, 0.5, 0.5]]), 3)


@pytest.mark.parametrize(
    "y_true, y_score",
    [
        ([1, 0, 1], [0.25, 0.5, 0.5]),
        ([1, 0, 1], [[0.25, 0.5, 0.5]]),
        ([[1, 0, 1]], [0.25, 0.5, 0.5]),
    ],
)
def test_coverage_1d_error_message(y_true, y_score):
    # [FUNCTION DOC]
    # Non-regression: coverage_error raises ValueError for 1d y_true or y_score.
    # Inputs: y_true, y_score - parametrized.
    # Non-regression test for:
    # https://github.com/scikit-learn/scikit-learn/issues/23368
    with pytest.raises(ValueError, match=r"Expected 2D array, got 1D array instead"):
        coverage_error(y_true, y_score)


def test_label_ranking_loss():
    # [FUNCTION DOC]
    # Verifies label_ranking_loss on toy cases, 0 for degenerate all-or-nothing rows,
    # and multi-sample cases.
    assert_almost_equal(label_ranking_loss([[0, 1]], [[0.25, 0.75]]), 0)
    assert_almost_equal(label_ranking_loss([[0, 1]], [[0.75, 0.25]]), 1)

    assert_almost_equal(label_ranking_loss([[0, 0, 1]], [[0.25, 0.5, 0.75]]), 0)
    assert_almost_equal(label_ranking_loss([[0, 1, 0]], [[0.25, 0.5, 0.75]]), 1 / 2)
    assert_almost_equal(label_ranking_loss([[0, 1, 1]], [[0.25, 0.5, 0.75]]), 0)
    assert_almost_equal(label_ranking_loss([[1, 0, 0]], [[0.25, 0.5, 0.75]]), 2 / 2)
    assert_almost_equal(label_ranking_loss([[1, 0, 1]], [[0.25, 0.5, 0.75]]), 1 / 2)
    assert_almost_equal(label_ranking_loss([[1, 1, 0]], [[0.25, 0.5, 0.75]]), 2 / 2)

    # Undefined metrics -  the ranking doesn't matter
    assert_almost_equal(label_ranking_loss([[0, 0]], [[0.75, 0.25]]), 0)
    assert_almost_equal(label_ranking_loss([[1, 1]], [[0.75, 0.25]]), 0)
    assert_almost_equal(label_ranking_loss([[0, 0]], [[0.5, 0.5]]), 0)
    assert_almost_equal(label_ranking_loss([[1, 1]], [[0.5, 0.5]]), 0)

    assert_almost_equal(label_ranking_loss([[0, 0, 0]], [[0.5, 0.75, 0.25]]), 0)
    assert_almost_equal(label_ranking_loss([[1, 1, 1]], [[0.5, 0.75, 0.25]]), 0)
    assert_almost_equal(label_ranking_loss([[0, 0, 0]], [[0.25, 0.5, 0.5]]), 0)
    assert_almost_equal(label_ranking_loss([[1, 1, 1]], [[0.25, 0.5, 0.5]]), 0)

    # Non trivial case
    assert_almost_equal(
        label_ranking_loss([[0, 1, 0], [1, 1, 0]], [[0.1, 10.0, -3], [0, 1, 3]]),
        (0 + 2 / 2) / 2.0,
    )

    assert_almost_equal(
        label_ranking_loss(
            [[0, 1, 0], [1, 1, 0], [0, 1, 1]], [[0.1, 10, -3], [0, 1, 3], [0, 2, 0]]
        ),
        (0 + 2 / 2 + 1 / 2) / 3.0,
    )

    assert_almost_equal(
        label_ranking_loss(
            [[0, 1, 0], [1, 1, 0], [0, 1, 1]], [[0.1, 10, -3], [3, 1, 3], [0, 2, 0]]
        ),
        (0 + 2 / 2 + 1 / 2) / 3.0,
    )


@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_label_ranking_loss_sparse(csr_container):
    # [FUNCTION DOC]
    # Verifies label_ranking_loss accepts sparse CSR y_true.
    # Inputs: csr_container - parametrized.
    assert_almost_equal(
        label_ranking_loss(
            csr_container(np.array([[0, 1, 0], [1, 1, 0]])), [[0.1, 10, -3], [3, 1, 3]]
        ),
        (0 + 2 / 2) / 2.0,
    )


def test_ranking_appropriate_input_shape():
    # [FUNCTION DOC]
    # Verifies label_ranking_loss raises ValueError on mismatched shapes.
    # Check that y_true.shape != y_score.shape raise the proper exception
    with pytest.raises(ValueError):
        label_ranking_loss([[0, 1], [0, 1]], [0, 1])
    with pytest.raises(ValueError):
        label_ranking_loss([[0, 1], [0, 1]], [[0, 1]])
    with pytest.raises(ValueError):
        label_ranking_loss([[0, 1], [0, 1]], [[0], [1]])
    with pytest.raises(ValueError):
        label_ranking_loss([[0, 1]], [[0, 1], [0, 1]])
    with pytest.raises(ValueError):
        label_ranking_loss([[0], [1]], [[0, 1], [0, 1]])
    with pytest.raises(ValueError):
        label_ranking_loss([[0, 1], [0, 1]], [[0], [1]])


def test_ranking_loss_ties_handling():
    # [FUNCTION DOC]
    # Verifies label_ranking_loss counts tied relevant / irrelevant pairs as errors.
    # Tie handling
    assert_almost_equal(label_ranking_loss([[1, 0]], [[0.5, 0.5]]), 1)
    assert_almost_equal(label_ranking_loss([[0, 1]], [[0.5, 0.5]]), 1)
    assert_almost_equal(label_ranking_loss([[0, 0, 1]], [[0.25, 0.5, 0.5]]), 1 / 2)
    assert_almost_equal(label_ranking_loss([[0, 1, 0]], [[0.25, 0.5, 0.5]]), 1 / 2)
    assert_almost_equal(label_ranking_loss([[0, 1, 1]], [[0.25, 0.5, 0.5]]), 0)
    assert_almost_equal(label_ranking_loss([[1, 0, 0]], [[0.25, 0.5, 0.5]]), 1)
    assert_almost_equal(label_ranking_loss([[1, 0, 1]], [[0.25, 0.5, 0.5]]), 1)
    assert_almost_equal(label_ranking_loss([[1, 1, 0]], [[0.25, 0.5, 0.5]]), 1)


def test_dcg_score():
    # [FUNCTION DOC]
    # Runs _test_dcg_score_for on multilabel targets with inverted scores and on
    # random continuous relevance and scores.
    _, y_true = make_multilabel_classification(random_state=0, n_classes=10)
    y_score = -y_true + 1
    _test_dcg_score_for(y_true, y_score)
    y_true, y_score = np.random.RandomState(0).random_sample((2, 100, 10))
    _test_dcg_score_for(y_true, y_score)


def _test_dcg_score_for(y_true, y_score):
    # [FUNCTION DOC]
    # HELPER: Checks per-sample DCG is bounded by the ideal DCG (also with k 5),
    # output shapes, and that the ideal DCG equals sorted relevance divided by
    # log2(rank + 1).
    # Inputs: y_true, y_score - (n_samples, n_labels) arrays.
    discount = np.log2(np.arange(y_true.shape[1]) + 2)
    ideal = _dcg_sample_scores(y_true, y_true)
    score = _dcg_sample_scores(y_true, y_score)
    assert (score <= ideal).all()
    assert (_dcg_sample_scores(y_true, y_true, k=5) <= ideal).all()
    assert ideal.shape == (y_true.shape[0],)
    assert score.shape == (y_true.shape[0],)
    assert ideal == pytest.approx((np.sort(y_true)[:, ::-1] / discount).sum(axis=1))


def test_dcg_ties():
    # [FUNCTION DOC]
    # Verifies DCG tie handling: with ties, tied items share averaged gains over their
    # discount positions; with ignore_ties, ties are broken in reverse index order.
    y_true = np.asarray([np.arange(5)])
    y_score = np.zeros(y_true.shape)
    dcg = _dcg_sample_scores(y_true, y_score)
    dcg_ignore_ties = _dcg_sample_scores(y_true, y_score, ignore_ties=True)
    discounts = 1 / np.log2(np.arange(2, 7))
    assert dcg == pytest.approx([discounts.sum() * y_true.mean()])
    assert dcg_ignore_ties == pytest.approx([(discounts * y_true[:, ::-1]).sum()])
    y_score[0, 3:] = 1
    dcg = _dcg_sample_scores(y_true, y_score)
    dcg_ignore_ties = _dcg_sample_scores(y_true, y_score, ignore_ties=True)
    assert dcg_ignore_ties == pytest.approx([(discounts * y_true[:, ::-1]).sum()])
    assert dcg == pytest.approx(
        [
            discounts[:2].sum() * y_true[0, 3:].mean()
            + discounts[2:].sum() * y_true[0, :3].mean()
        ]
    )


def test_ndcg_ignore_ties_with_k():
    # [FUNCTION DOC]
    # Intended to check NDCG with k and ignore_ties.
    # KNOWN ISSUE: compares the same expression with itself, so it only checks the
    # call does not fail.
    a = np.arange(12).reshape((2, 6))
    assert ndcg_score(a, a, k=3, ignore_ties=True) == pytest.approx(
        ndcg_score(a, a, k=3, ignore_ties=True)
    )


def test_ndcg_negative_ndarray_error():
    # [FUNCTION DOC]
    # Verifies ndcg_score raises ValueError for negative y_true.
    """Check `ndcg_score` exception when `y_true` contains negative values."""
    y_true = np.array([[-0.89, -0.53, -0.47, 0.39, 0.56]])
    y_score = np.array([[0.07, 0.31, 0.75, 0.33, 0.27]])
    expected_message = "ndcg_score should not be used on negative y_true values"
    with pytest.raises(ValueError, match=expected_message):
        ndcg_score(y_true, y_score)


def test_ndcg_invariant():
    # [FUNCTION DOC]
    # Verifies NDCG is 1 for scores equal to relevance plus small noise, equal with and
    # without ignore_ties, and invariant to adding a constant.
    y_true = np.arange(70).reshape(7, 10)
    y_score = y_true + np.random.RandomState(0).uniform(-0.2, 0.2, size=y_true.shape)
    ndcg = ndcg_score(y_true, y_score)
    ndcg_no_ties = ndcg_score(y_true, y_score, ignore_ties=True)
    assert ndcg == pytest.approx(ndcg_no_ties)
    assert ndcg == pytest.approx(1.0)
    y_score += 1000
    assert ndcg_score(y_true, y_score) == pytest.approx(1.0)


@pytest.mark.parametrize("ignore_ties", [True, False])
def test_ndcg_toy_examples(ignore_ties):
    # [FUNCTION DOC]
    # Verifies per-sample DCG / NDCG and averaged dcg_score / ndcg_score against
    # closed-form values (log2 and log10 discounts) for one-hot and constant
    # relevance.
    # Inputs: ignore_ties - parametrized.
    y_true = 3 * np.eye(7)[:5]
    y_score = np.tile(np.arange(6, -1, -1), (5, 1))
    y_score_noisy = y_score + np.random.RandomState(0).uniform(
        -0.2, 0.2, size=y_score.shape
    )
    assert _dcg_sample_scores(
        y_true, y_score, ignore_ties=ignore_ties
    ) == pytest.approx(3 / np.log2(np.arange(2, 7)))
    assert _dcg_sample_scores(
        y_true, y_score_noisy, ignore_ties=ignore_ties
    ) == pytest.approx(3 / np.log2(np.arange(2, 7)))
    assert _ndcg_sample_scores(
        y_true, y_score, ignore_ties=ignore_ties
    ) == pytest.approx(1 / np.log2(np.arange(2, 7)))
    assert _dcg_sample_scores(
        y_true, y_score, log_base=10, ignore_ties=ignore_ties
    ) == pytest.approx(3 / np.log10(np.arange(2, 7)))
    assert ndcg_score(y_true, y_score, ignore_ties=ignore_ties) == pytest.approx(
        (1 / np.log2(np.arange(2, 7))).mean()
    )
    assert dcg_score(y_true, y_score, ignore_ties=ignore_ties) == pytest.approx(
        (3 / np.log2(np.arange(2, 7))).mean()
    )
    y_true = 3 * np.ones((5, 7))
    expected_dcg_score = (3 / np.log2(np.arange(2, 9))).sum()
    assert _dcg_sample_scores(
        y_true, y_score, ignore_ties=ignore_ties
    ) == pytest.approx(expected_dcg_score * np.ones(5))
    assert _ndcg_sample_scores(
        y_true, y_score, ignore_ties=ignore_ties
    ) == pytest.approx(np.ones(5))
    assert dcg_score(y_true, y_score, ignore_ties=ignore_ties) == pytest.approx(
        expected_dcg_score
    )
    assert ndcg_score(y_true, y_score, ignore_ties=ignore_ties) == pytest.approx(1.0)


def test_ndcg_error_single_document():
    # [FUNCTION DOC]
    # Verifies ndcg_score raises ValueError with a single document.
    """Check that we raise an informative error message when trying to
    compute NDCG with a single document."""
    err_msg = (
        "Computing NDCG is only meaningful when there is more than 1 document. "
        "Got 1 instead."
    )
    with pytest.raises(ValueError, match=err_msg):
        ndcg_score([[1]], [[1]])


def test_ndcg_score():
    # [FUNCTION DOC]
    # Runs _test_ndcg_score_for on multilabel and random continuous data.
    _, y_true = make_multilabel_classification(random_state=0, n_classes=10)
    y_score = -y_true + 1
    _test_ndcg_score_for(y_true, y_score)
    y_true, y_score = np.random.RandomState(0).random_sample((2, 100, 10))
    _test_ndcg_score_for(y_true, y_score)


def _test_ndcg_score_for(y_true, y_score):
    # [FUNCTION DOC]
    # HELPER: Checks NDCG <= ideal, ideal is 1 (0 for all-zero rows), NDCG equals DCG
    # divided by ideal DCG for non-zero rows, 0 for all-zero rows, and output shapes.
    # Inputs: y_true, y_score - (n_samples, n_labels) arrays.
    ideal = _ndcg_sample_scores(y_true, y_true)
    score = _ndcg_sample_scores(y_true, y_score)
    assert (score <= ideal).all()
    all_zero = (y_true == 0).all(axis=1)
    assert ideal[~all_zero] == pytest.approx(np.ones((~all_zero).sum()))
    assert ideal[all_zero] == pytest.approx(np.zeros(all_zero.sum()))
    assert score[~all_zero] == pytest.approx(
        _dcg_sample_scores(y_true, y_score)[~all_zero]
        / _dcg_sample_scores(y_true, y_true)[~all_zero]
    )
    assert score[all_zero] == pytest.approx(np.zeros(all_zero.sum()))
    assert ideal.shape == (y_true.shape[0],)
    assert score.shape == (y_true.shape[0],)


def test_partial_roc_auc_score():
    # [FUNCTION DOC]
    # Verifies roc_auc_score with max_fpr: perfect predictions give 1, invalid max_fpr
    # raise ValueError, max_fpr 1 equals unconstrained AUC, a known 0.5 case, and
    # agreement with _partial_roc_auc_score for several max_fpr.
    # Check `roc_auc_score` for max_fpr != `None`
    y_true = np.array([0, 0, 1, 1])
    assert roc_auc_score(y_true, y_true, max_fpr=1) == 1
    assert roc_auc_score(y_true, y_true, max_fpr=0.001) == 1
    with pytest.raises(ValueError):
        assert roc_auc_score(y_true, y_true, max_fpr=-0.1)
    with pytest.raises(ValueError):
        assert roc_auc_score(y_true, y_true, max_fpr=1.1)
    with pytest.raises(ValueError):
        assert roc_auc_score(y_true, y_true, max_fpr=0)

    y_scores = np.array([0.1, 0, 0.1, 0.01])
    roc_auc_with_max_fpr_one = roc_auc_score(y_true, y_scores, max_fpr=1)
    unconstrained_roc_auc = roc_auc_score(y_true, y_scores)
    assert roc_auc_with_max_fpr_one == unconstrained_roc_auc
    assert roc_auc_score(y_true, y_scores, max_fpr=0.3) == 0.5

    y_true, y_pred, _ = make_prediction(binary=True)
    for max_fpr in np.linspace(1e-4, 1, 5):
        assert_almost_equal(
            roc_auc_score(y_true, y_pred, max_fpr=max_fpr),
            _partial_roc_auc_score(y_true, y_pred, max_fpr),
        )


@pytest.mark.parametrize(
    "y_true, k, true_score",
    [
        ([0, 1, 2, 3], 1, 0.25),
        ([0, 1, 2, 3], 2, 0.5),
        ([0, 1, 2, 3], 3, 0.75),
    ],
)
def test_top_k_accuracy_score(y_true, k, true_score):
    # [FUNCTION DOC]
    # Verifies multiclass top-k accuracy values for k 1..3.
    # Inputs: y_true, k, true_score - parametrized.
    y_score = np.array(
        [
            [0.4, 0.3, 0.2, 0.1],
            [0.1, 0.3, 0.4, 0.2],
            [0.4, 0.1, 0.2, 0.3],
            [0.3, 0.2, 0.4, 0.1],
        ]
    )
    score = top_k_accuracy_score(y_true, y_score, k=k)
    assert score == pytest.approx(true_score)


@pytest.mark.parametrize(
    "y_score, k, true_score",
    [
        (np.array([-1, -1, 1, 1]), 1, 1),
        (np.array([-1, 1, -1, 1]), 1, 0.5),
        (np.array([-1, 1, -1, 1]), 2, 1),
        (np.array([0.2, 0.2, 0.7, 0.7]), 1, 1),
        (np.array([0.2, 0.7, 0.2, 0.7]), 1, 0.5),
        (np.array([0.2, 0.7, 0.2, 0.7]), 2, 1),
    ],
)
def test_top_k_accuracy_score_binary(y_score, k, true_score):
    # [FUNCTION DOC]
    # Verifies binary top-k accuracy with 1d scores equals thresholded accuracy for
    # k 1 (threshold 0.5 for probabilities, 0 otherwise) and 1 for k 2.
    # Inputs: y_score, k, true_score - parametrized.
    y_true = [0, 0, 1, 1]

    threshold = 0.5 if y_score.min() >= 0 and y_score.max() <= 1 else 0
    y_pred = (y_score > threshold).astype(np.int64) if k == 1 else y_true

    score = top_k_accuracy_score(y_true, y_score, k=k)
    score_acc = accuracy_score(y_true, y_pred)

    assert score == score_acc == pytest.approx(true_score)


@pytest.mark.parametrize(
    "y_true, true_score, labels",
    [
        (np.array([0, 1, 1, 2]), 0.75, [0, 1, 2, 3]),
        (np.array([0, 1, 1, 1]), 0.5, [0, 1, 2, 3]),
        (np.array([1, 1, 1, 1]), 0.5, [0, 1, 2, 3]),
        (np.array(["a", "e", "e", "a"]), 0.75, ["a", "b", "d", "e"]),
    ],
)
@pytest.mark.parametrize("labels_as_ndarray", [True, False])
def test_top_k_accuracy_score_multiclass_with_labels(
    y_true, true_score, labels, labels_as_ndarray
):
    # [FUNCTION DOC]
    # Verifies top-2 accuracy with explicit labels (list or ndarray) when y_true lacks
    # some labels.
    # Inputs: y_true, true_score, labels, labels_as_ndarray - parametrized.
    """Test when labels and y_score are multiclass."""
    if labels_as_ndarray:
        labels = np.asarray(labels)
    y_score = np.array(
        [
            [0.4, 0.3, 0.2, 0.1],
            [0.1, 0.3, 0.4, 0.2],
            [0.4, 0.1, 0.2, 0.3],
            [0.3, 0.2, 0.4, 0.1],
        ]
    )

    score = top_k_accuracy_score(y_true, y_score, k=2, labels=labels)
    assert score == pytest.approx(true_score)


def test_top_k_accuracy_score_increasing():
    # [FUNCTION DOC]
    # Verifies top-k accuracy strictly increases with k on train and test sets.
    # Make sure increasing k leads to a higher score
    X, y = datasets.make_classification(
        n_classes=10, n_samples=1000, n_informative=10, random_state=0
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    clf = LogisticRegression(random_state=0)
    clf.fit(X_train, y_train)

    for X, y in zip((X_train, X_test), (y_train, y_test)):
        scores = [
            top_k_accuracy_score(y, clf.predict_proba(X), k=k) for k in range(2, 10)
        ]

        assert np.all(np.diff(scores) > 0)


@pytest.mark.parametrize(
    "y_true, k, true_score",
    [
        ([0, 1, 2, 3], 1, 0.25),
        ([0, 1, 2, 3], 2, 0.5),
        ([0, 1, 2, 3], 3, 1),
    ],
)
def test_top_k_accuracy_score_ties(y_true, k, true_score):
    # [FUNCTION DOC]
    # Verifies ties in scores are resolved in favour of higher label indices.
    # Inputs: y_true, k, true_score - parametrized.
    # Make sure highest indices labels are chosen first in case of ties
    y_score = np.array(
        [
            [5, 5, 7, 0],
            [1, 5, 5, 5],
            [0, 0, 3, 3],
            [1, 1, 1, 1],
        ]
    )
    assert top_k_accuracy_score(y_true, y_score, k=k) == pytest.approx(true_score)


@pytest.mark.parametrize(
    "y_true, k",
    [
        ([0, 1, 2, 3], 4),
        ([0, 1, 2, 3], 5),
    ],
)
def test_top_k_accuracy_score_warning(y_true, k):
    # [FUNCTION DOC]
    # Verifies UndefinedMetricWarning and score 1 when k >= n_classes.
    # Inputs: y_true, k - parametrized.
    y_score = np.array(
        [
            [0.4, 0.3, 0.2, 0.1],
            [0.1, 0.4, 0.3, 0.2],
            [0.2, 0.1, 0.4, 0.3],
            [0.3, 0.2, 0.1, 0.4],
        ]
    )
    expected_message = (
        r"'k' \(\d+\) greater than or equal to 'n_classes' \(\d+\) will result in a "
        "perfect score and is therefore meaningless."
    )
    with pytest.warns(UndefinedMetricWarning, match=expected_message):
        score = top_k_accuracy_score(y_true, y_score, k=k)
    assert score == 1


@pytest.mark.parametrize(
    "y_true, y_score, labels, msg",
    [
        (
            [0, 0.57, 1, 2],
            [
                [0.2, 0.1, 0.7],
                [0.4, 0.3, 0.3],
                [0.3, 0.4, 0.3],
                [0.4, 0.5, 0.1],
            ],
            None,
            "y type must be 'binary' or 'multiclass', got 'continuous'",
        ),
        (
            [0, 1, 2, 3],
            [
                [0.2, 0.1, 0.7],
                [0.4, 0.3, 0.3],
                [0.3, 0.4, 0.3],
                [0.4, 0.5, 0.1],
            ],
            None,
            r"Number of classes in 'y_true' \(4\) not equal to the number of "
            r"classes in 'y_score' \(3\).",
        ),
        (
            ["c", "c", "a", "b"],
            [
                [0.2, 0.1, 0.7],
                [0.4, 0.3, 0.3],
                [0.3, 0.4, 0.3],
                [0.4, 0.5, 0.1],
            ],
            ["a", "b", "c", "c"],
            "Parameter 'labels' must be unique.",
        ),
        (
            ["c", "c", "a", "b"],
            [
                [0.2, 0.1, 0.7],
                [0.4, 0.3, 0.3],
                [0.3, 0.4, 0.3],
                [0.4, 0.5, 0.1],
            ],
            ["a", "c", "b"],
            "Parameter 'labels' must be ordered.",
        ),
        (
            [0, 0, 1, 2],
            [
                [0.2, 0.1, 0.7],
                [0.4, 0.3, 0.3],
                [0.3, 0.4, 0.3],
                [0.4, 0.5, 0.1],
            ],
            [0, 1, 2, 3],
            r"Number of given labels \(4\) not equal to the number of classes in "
            r"'y_score' \(3\).",
        ),
        (
            [0, 0, 1, 2],
            [
                [0.2, 0.1, 0.7],
                [0.4, 0.3, 0.3],
                [0.3, 0.4, 0.3],
                [0.4, 0.5, 0.1],
            ],
            [0, 1, 3],
            "'y_true' contains labels not in parameter 'labels'.",
        ),
        (
            [0, 1],
            [[0.5, 0.2, 0.2], [0.3, 0.4, 0.2]],
            None,
            (
                "`y_true` is binary while y_score is 2d with 3 classes. If"
                " `y_true` does not contain all the labels, `labels` must be provided"
            ),
        ),
    ],
)
def test_top_k_accuracy_score_error(y_true, y_score, labels, msg):
    # [FUNCTION DOC]
    # Verifies ValueError messages for continuous y_true, class count mismatches,
    # invalid labels and binary y_true with 3-column scores and no labels.
    # Inputs: y_true, y_score, labels, msg - parametrized.
    with pytest.raises(ValueError, match=msg):
        top_k_accuracy_score(y_true, y_score, k=2, labels=labels)


@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_label_ranking_avg_precision_score_should_allow_csr_matrix_for_y_true_input(
    csr_container,
):
    # [FUNCTION DOC]
    # Non-regression: LRAP accepts sparse CSR y_true (value 2/3).
    # Inputs: csr_container - parametrized.
    # Test that label_ranking_avg_precision_score accept sparse y_true.
    # Non-regression test for #22575
    y_true = csr_container([[1, 0, 0], [0, 0, 1]])
    y_score = np.array([[0.5, 0.9, 0.6], [0, 0, 1]])
    result = label_ranking_average_precision_score(y_true, y_score)
    assert result == pytest.approx(2 / 3)


@pytest.mark.parametrize(
    "metric", [average_precision_score, det_curve, precision_recall_curve, roc_curve]
)
@pytest.mark.parametrize(
    "classes", [(False, True), (0, 1), (0.0, 1.0), ("zero", "one")]
)
def test_ranking_metric_pos_label_types(metric, classes):
    # [FUNCTION DOC]
    # Verifies ranking metrics accept bool, int, float and string pos_label and return
    # no NaNs.
    # Inputs: metric, classes - parametrized.
    """Check that the metric works with different types of `pos_label`.

    We can expect `pos_label` to be a bool, an integer, a float, a string.
    No error should be raised for those types.
    """
    rng = np.random.RandomState(42)
    n_samples, pos_label = 10, classes[-1]
    y_true = rng.choice(classes, size=n_samples, replace=True)
    y_proba = rng.rand(n_samples)
    result = metric(y_true, y_proba, pos_label=pos_label)
    if isinstance(result, float):
        assert not np.isnan(result)
    else:
        metric_1, metric_2, thresholds = result
        assert not np.isnan(metric_1).any()
        assert not np.isnan(metric_2).any()
        assert not np.isnan(thresholds).any()


def test_roc_curve_with_probability_estimates(global_random_seed):
    # [FUNCTION DOC]
    # Non-regression: the first ROC threshold is inf for probability scores.
    # Inputs: global_random_seed fixture.
    """Check that thresholds do not exceed 1.0 when `y_score` is a probability
    estimate.

    Non-regression test for:
    https://github.com/scikit-learn/scikit-learn/issues/26193
    """
    rng = np.random.RandomState(global_random_seed)
    y_true = rng.randint(0, 2, size=10)
    y_score = rng.rand(10)
    _, _, thresholds = roc_curve(y_true, y_score)
    assert np.isinf(thresholds[0])


def _dummy_metric(y_true, y_pred, sample_weight=None):
    # [FUNCTION DOC]
    # HELPER: Metric returning a tuple (weighted sum of y_pred, weighted sum of y_true);
    # weights default to ones.
    # Inputs: y_true, y_pred - arrays; sample_weight - array or None.
    """Dummy metric that returns a tuple of two values."""
    if sample_weight is None:
        sample_weight = np.ones_like(y_pred)
    y_pred_sum = np.sum(y_pred * sample_weight)
    y_true_sum = np.sum(y_true * sample_weight)
    return (y_pred_sum, y_true_sum)


@pytest.mark.parametrize(
    "metric_func",
    [
        accuracy_score,
        precision_score,
        roc_auc_score,
        # Test metric that returns tuple instead of single float
        _dummy_metric,
    ],
)
@pytest.mark.parametrize("sample_weight", [None, np.array([1, 2, 1, 0, 2])])
def test_metric_at_thresholds(metric_func, sample_weight):
    # [FUNCTION DOC]
    # Verifies metric_at_thresholds returns, for descending unique non-zero-weight
    # scores as thresholds, the metric of predictions score >= threshold, for scalar
    # and tuple-valued metrics with and without sample_weight.
    # Inputs: metric_func, sample_weight - parametrized.
    """Test `metric_at_thresholds` outputs correct."""
    y_true = np.array([0, 0, 1, 1, 1])
    y_score = np.array([0.1, 0.6, 0.4, 0.9, 0.4])

    metric_values, thresholds = metric_at_thresholds(
        y_true, y_score, metric_func, sample_weight=sample_weight
    )

    # Calculate expected scores manually at each threshold
    expected_scores = []
    for threshold in thresholds:
        y_pred = (y_score >= threshold).astype(int)
        expected_scores.append(metric_func(y_true, y_pred, sample_weight=sample_weight))

    assert len(metric_values) == len(thresholds)
    # Thresholds are descending
    assert np.all(np.diff(thresholds) <= 0)
    # Thresholds correspond to unique `y_score`s
    if sample_weight is not None:
        # Filter out 0 weight in `y_score`
        assert_allclose(
            thresholds, np.sort(np.unique(y_score[sample_weight != 0]))[::-1]
        )
    else:
        assert_allclose(thresholds, np.sort(np.unique(y_score))[::-1])
    assert_allclose(metric_values, expected_scores)


def _dummy_metric_no_sample_weight(y_true, y_pred):
    # [FUNCTION DOC]
    # HELPER: Metric without sample_weight; returns (sum of y_pred, sum of y_true).
    """Dummy metric that does not accept `sample_weight`."""
    return (np.sum(y_pred), np.sum(y_true))


def test_metric_at_thresholds_sample_weight_error():
    # [FUNCTION DOC]
    # Verifies TypeError when sample_weight is given and the metric does not accept it.
    """Test `TypeError` is raised when `metric_func` does not take `sample_weight`."""
    y_true = np.array([0, 0, 1, 1, 1])
    y_score = np.array([0.1, 0.4, 0.35, 0.6, 0.9])
    sample_weight = np.array([1, 2, 3, 1, 2])

    with pytest.raises(TypeError, match="got an unexpected keyword argument"):
        _, _ = metric_at_thresholds(
            y_true, y_score, _dummy_metric_no_sample_weight, sample_weight=sample_weight
        )


@pytest.mark.parametrize("normalize", [True, False])
def test_metric_at_thresholds_metric_params(normalize):
    # [FUNCTION DOC]
    # Verifies metric_params are forwarded to the metric (accuracy normalize).
    # Inputs: normalize - parametrized.
    """Test `metric_params` passed correctly to `metric_at_thresholds`."""
    y_true = np.array([0, 0, 1, 1, 1])
    y_score = np.array([0.1, 0.4, 0.35, 0.6, 0.9])

    metric_values, thresholds = metric_at_thresholds(
        y_true, y_score, accuracy_score, metric_params={"normalize": normalize}
    )

    expected_values = []
    for threshold in thresholds:
        y_pred = (y_score >= threshold).astype(int)
        expected_values.append(accuracy_score(y_true, y_pred, normalize=normalize))

    assert_allclose(metric_values, expected_values)
    assert len(thresholds) == len(np.unique(y_score))


@pytest.mark.parametrize("pos_label", [0, 1])
def test_metric_at_thresholds_pos_label(pos_label):
    # [FUNCTION DOC]
    # Verifies pos_label in metric_params is forwarded to precision_score.
    # Inputs: pos_label - parametrized.
    """Test `pos_label` is passed correctly to `metric_at_thresholds`."""
    y_true = np.array([0, 0, 1, 1, 1])
    y_score = np.array([0.1, 0.6, 0.4, 0.9, 0.4])

    metric_values, thresholds = metric_at_thresholds(
        y_true,
        y_score,
        precision_score,
        metric_params={"pos_label": pos_label, "zero_division": 0},
    )

    expected_scores = []
    for threshold in thresholds:
        y_pred = (y_score >= threshold).astype(int)
        expected_scores.append(
            precision_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
        )

    assert_allclose(metric_values, expected_scores)


def test_metric_at_thresholds_y_score_order():
    # [FUNCTION DOC]
    # Verifies results are invariant to permuting samples (no duplicate scores).
    """Test `y_score` order does not effect `metric_at_thresholds`."""
    y_true = np.array([0, 0, 1, 1, 1])
    y_score = np.array([0.9, 0.6, 0.5, 0.1, 0.4])
    sample_weight = np.array([1, 2, 3, 1, 2])
    # Permutate `y_true`, `y_score` and `sample_weight`
    rng = check_random_state(42)
    perm_indices = rng.permutation(len(y_score))
    y_true_perm = y_true[perm_indices]
    y_score_perm = y_score[perm_indices]
    sample_weight_perm = sample_weight[perm_indices]

    metric_1, thresh_1 = metric_at_thresholds(
        y_true, y_score, accuracy_score, sample_weight=sample_weight
    )
    metric_2, thresh_2 = metric_at_thresholds(
        y_true_perm, y_score_perm, accuracy_score, sample_weight=sample_weight_perm
    )

    assert_allclose(metric_1, metric_2)
    assert_allclose(thresh_1, thresh_2)


def test_metric_at_thresholds_y_score_order_duplicate_y_score():
    # [FUNCTION DOC]
    # Verifies with duplicate scores, differing y_true order does not change results,
    # while differing sample weights among duplicates does change metric values but
    # not thresholds.
    """Test duplicate `y_score` edge cases in `metric_at_thresholds`.

    If there are duplicate `y_score` values and `y_true` differs between
    these duplicate values, `y_score` order will not affect metric output.

    However, if there are duplicate `y_score` values and `sample_weight` differs
    between these duplicate values, `y_score` order can affect metric output,
    as stable sort preserves relative order.
    """
    # duplicate scores
    y_score = np.array([0.6, 0.9, 0.1, 0.4, 0.6])
    # `y_true` differs between duplicates
    y_true_1 = np.array([1, 0, 1, 1, 0])
    y_true_2 = np.array([0, 0, 1, 1, 1])

    metric_1, thresh_1 = metric_at_thresholds(y_true_1, y_score, accuracy_score)
    metric_2, thresh_2 = metric_at_thresholds(y_true_2, y_score, accuracy_score)

    assert_allclose(thresh_1, thresh_2)
    assert_allclose(metric_1, metric_2)

    # `sample_weight` differs between duplicates
    sample_weight_1 = np.array([1, 2, 1, 0, 2])
    sample_weight_2 = np.array([2, 2, 1, 0, 1])

    metric_1, thresh_1 = metric_at_thresholds(
        y_true_1, y_score, accuracy_score, sample_weight=sample_weight_1
    )
    metric_2, thresh_2 = metric_at_thresholds(
        y_true_1, y_score, accuracy_score, sample_weight=sample_weight_2
    )

    # Thresholds should still be the same
    assert_allclose(thresh_1, thresh_2)
    # Metric output differs for the test data
    with pytest.raises(AssertionError):
        assert_allclose(metric_1, metric_2)


def test_metric_at_thresholds_consistency_with_confusion_matrix():
    # [FUNCTION DOC]
    # Verifies confusion_matrix evaluated at thresholds matches the counts returned by
    # confusion_matrix_at_thresholds, including the array-valued metric output shape.
    """Test `metric_at_thresholds` consistency with `confusion_matrix_at_thresholds`.

    This also checks output when `metric_func` returns a tuple of arrays.
    """
    y_true = np.array([0, 0, 1, 1, 1])
    y_score = np.array([0.1, 0.4, 0.4, 0.6, 0.9])

    tns, fps, fns, tps, thresholds_cm = confusion_matrix_at_thresholds(
        y_true,
        y_score,
    )

    metric_values, thresholds = metric_at_thresholds(
        y_true, y_score, confusion_matrix, metric_params={"labels": [0, 1]}
    )

    assert_array_equal(thresholds, thresholds_cm)

    # As `labels=[0, 1]` -> [TN, FP, FN, TP]
    assert_array_equal(metric_values[:, 0, 0], tns)
    assert_array_equal(metric_values[:, 0, 1], fps)
    assert_array_equal(metric_values[:, 1, 0], fns)
    assert_array_equal(metric_values[:, 1, 1], tps)


def test_metric_at_thresholds_with_nan_outputs():
    # [FUNCTION DOC]
    # Verifies NaN metric values (recall with no positives, zero_division NaN) are
    # propagated.
    """Test `metric_at_thresholds` with NaN output."""
    # No positive labels means recall undefined (TP + FN = 0) at all thresholds
    y_true = np.array([0, 0, 0, 0, 0])
    y_score = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

    metric_values, _ = metric_at_thresholds(
        y_true, y_score, recall_score, metric_params={"zero_division": np.nan}
    )

    assert np.all(np.isnan(metric_values))


# TODO(1.11): remove this test
def test_confusion_matrix_at_thresholds_positional_args_deprecation():
    # [FUNCTION DOC]
    # Verifies passing pos_label positionally emits a FutureWarning (to be removed).
    y_true = np.array([0, 1, 1, 0])
    y_score = np.array([0.2, 0.1, 0.7, 0.7])
    with pytest.warns(FutureWarning, match="Pass pos_label=None as keyword arg"):
        confusion_matrix_at_thresholds(y_true, y_score, None)
