# [MODULE DOC]
# Tests for the classification metrics in sklearn.metrics._classification: accuracy,
#   precision/recall/F-scores, confusion matrices, classification_report, Cohen's kappa,
#   Matthews correlation, Jaccard, Hamming, zero-one, hinge, log and Brier losses,
#   likelihood ratios, D2 scores and the _check_targets helper, including warnings,
#   zero_division handling and array API support. Module-level helper make_prediction
#   builds reproducible SVC predictions on iris.
import re
import warnings
from functools import partial
from itertools import chain, permutations, product

import numpy as np
import pytest
from scipy import linalg, sparse
from scipy.spatial.distance import hamming as sp_hamming
from scipy.stats import bernoulli

from sklearn import datasets, svm
from sklearn.base import config_context
from sklearn.calibration import CalibratedClassifierCV
from sklearn.datasets import make_multilabel_classification
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    class_likelihood_ratios,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    hamming_loss,
    hinge_loss,
    jaccard_score,
    log_loss,
    make_scorer,
    matthews_corrcoef,
    multilabel_confusion_matrix,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    zero_one_loss,
)
from sklearn.metrics._classification import (
    _check_targets,
    d2_brier_score,
    d2_log_loss_score,
)
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelBinarizer, label_binarize
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils._array_api import (
    device as array_api_device,
)
from sklearn.utils._array_api import (
    get_namespace,
    yield_namespace_device_dtype_combinations,
)
from sklearn.utils._mocking import MockDataFrame
from sklearn.utils._testing import (
    _array_api_for_tests,
    assert_allclose,
    assert_almost_equal,
    assert_array_almost_equal,
    assert_array_equal,
    ignore_warnings,
)
from sklearn.utils.extmath import _nanaverage
from sklearn.utils.fixes import CSC_CONTAINERS, CSR_CONTAINERS
from sklearn.utils.validation import check_random_state

###############################################################################
# Utilities for testing


def make_prediction(dataset=None, binary=False):
    # [FUNCTION DOC]
    # HELPER: Builds deterministic predictions for tests. Loads iris unless a dataset is
    #   given, optionally keeps classes 0 and 1 only, shuffles with seed 37, appends 200
    #   * n_features Gaussian noise columns (seed 0), fits a CalibratedClassifierCV
    #   around a linear SVC on the first half and predicts the second half.
    # Inputs: dataset - Bunch with data/target or None; binary - bool.
    # Outputs: (y_true, y_pred, y_pred_proba); y_pred_proba is the positive-class column
    #   in the binary case, else (n_samples, n_classes).
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
    clf = CalibratedClassifierCV(
        svm.SVC(kernel="linear", random_state=0), ensemble=False, cv=3
    )
    y_pred_proba = clf.fit(X[:half], y[:half]).predict_proba(X[half:])

    if binary:
        # only interested in probabilities of the positive case
        # XXX: do we really want a special API for the binary case?
        y_pred_proba = y_pred_proba[:, 1]

    y_pred = clf.predict(X[half:])
    y_true = y[half:]
    return y_true, y_pred, y_pred_proba


###############################################################################
# Tests


def test_classification_report_dictionary_output():
    # [FUNCTION DOC]
    # Verifies classification_report(output_dict=True) on iris predictions matches a
    #   hard-coded nested dict of per-class and averaged precision/recall/f1/support and
    #   a float accuracy.
    # KNOWN ISSUE: the trailing isinstance checks inspect expected_report instead of
    #   report, so they do not verify the types of the returned values.
    # Test performance report with dictionary output
    iris = datasets.load_iris()
    y_true, y_pred, _ = make_prediction(dataset=iris, binary=False)

    # print classification report with class names
    expected_report = {
        "setosa": {
            "precision": 0.82608695652173914,
            "recall": 0.79166666666666663,
            "f1-score": 0.8085106382978724,
            "support": 24,
        },
        "versicolor": {
            "precision": 0.33333333333333331,
            "recall": 0.096774193548387094,
            "f1-score": 0.15000000000000002,
            "support": 31,
        },
        "virginica": {
            "precision": 0.41860465116279072,
            "recall": 0.90000000000000002,
            "f1-score": 0.57142857142857151,
            "support": 20,
        },
        "macro avg": {
            "f1-score": 0.5099797365754813,
            "precision": 0.5260083136726211,
            "recall": 0.596146953405018,
            "support": 75,
        },
        "accuracy": 0.5333333333333333,
        "weighted avg": {
            "f1-score": 0.47310435663627154,
            "precision": 0.5137535108414785,
            "recall": 0.5333333333333333,
            "support": 75,
        },
    }

    report = classification_report(
        y_true,
        y_pred,
        labels=np.arange(len(iris.target_names)),
        target_names=iris.target_names,
        output_dict=True,
    )

    # assert the 2 dicts are equal.
    assert report.keys() == expected_report.keys()
    for key in expected_report:
        if key == "accuracy":
            assert isinstance(report[key], float)
            assert report[key] == expected_report[key]
        else:
            assert report[key].keys() == expected_report[key].keys()
            for metric in expected_report[key]:
                assert_almost_equal(expected_report[key][metric], report[key][metric])

    assert isinstance(expected_report["setosa"]["precision"], float)
    assert isinstance(expected_report["macro avg"]["precision"], float)
    assert isinstance(expected_report["setosa"]["support"], int)
    assert isinstance(expected_report["macro avg"]["support"], int)


@pytest.mark.parametrize("zero_division", ["warn", 0, 1, np.nan])
def test_classification_report_zero_division_warning(zero_division):
    # [FUNCTION DOC]
    # Verifies that with an unseen predicted label, zero_division="warn" emits several
    #   "Use `zero_division`" warnings, while 0, 1 and NaN emit none.
    y_true, y_pred = ["a", "b", "c"], ["a", "b", "d"]
    with warnings.catch_warnings(record=True) as record:
        # We need "always" instead of "once" for free-threaded with
        # pytest-run-parallel to capture all the warnings in the
        # zero_division="warn" case.
        warnings.filterwarnings("always", message=".+Use `zero_division`")
        classification_report(
            y_true, y_pred, zero_division=zero_division, output_dict=True
        )
        if zero_division == "warn":
            assert len(record) > 1
            for item in record:
                msg = "Use `zero_division` parameter to control this behavior."
                assert msg in str(item.message)
        else:
            assert not record


@pytest.mark.parametrize(
    "labels, show_micro_avg", [([0], True), ([0, 1], False), ([0, 1, 2], False)]
)
def test_classification_report_labels_subset_superset(labels, show_micro_avg):
    # [FUNCTION DOC]
    # Non-regression for #27927: a labels subset makes the report show "micro avg"
    #   instead of "accuracy"; equal or superset labels show "accuracy".
    """Check the behaviour of passing `labels` as a superset or subset of the labels.
    WHen a superset, we expect to show the "accuracy" in the report while it should be
    the micro-averaging if this is a subset.

    Non-regression test for:
    https://github.com/scikit-learn/scikit-learn/issues/27927
    """

    y_true, y_pred = [0, 1], [0, 1]

    report = classification_report(y_true, y_pred, labels=labels, output_dict=True)
    if show_micro_avg:
        assert "micro avg" in report
        assert "accuracy" not in report
    else:  # accuracy should be shown
        assert "accuracy" in report
        assert "micro avg" not in report


def test_multilabel_accuracy_score_subset_accuracy():
    # [FUNCTION DOC]
    # Verifies accuracy_score on multilabel indicator matrices is subset accuracy (row
    #   must match exactly).
    # Dense label indicator matrix format
    y1 = np.array([[0, 1, 1], [1, 0, 1]])
    y2 = np.array([[0, 0, 1], [1, 0, 1]])

    assert accuracy_score(y1, y2) == 0.5
    assert accuracy_score(y1, y1) == 1
    assert accuracy_score(y2, y2) == 1
    assert accuracy_score(y2, np.logical_not(y2)) == 0
    assert accuracy_score(y1, np.logical_not(y1)) == 0
    assert accuracy_score(y1, np.zeros(y1.shape)) == 0
    assert accuracy_score(y2, np.zeros(y1.shape)) == 0


def test_precision_recall_f1_score_binary():
    # [FUNCTION DOC]
    # Verifies per-class and binary-average precision, recall, F1 and F-beta on
    #   make_prediction binary output against known values and the F-beta formula, with
    #   all warnings turned into errors.
    # Test Precision Recall and F1 Score for binary classification task
    y_true, y_pred, _ = make_prediction(binary=True)

    # detailed measures for each class
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, average=None)
    assert_array_almost_equal(p, [0.73, 0.85], 2)
    assert_array_almost_equal(r, [0.88, 0.68], 2)
    assert_array_almost_equal(f, [0.80, 0.76], 2)
    assert_array_equal(s, [25, 25])

    # individual scoring function that can be used for grid search: in the
    # binary class case the score is the value of the measure for the positive
    # class (e.g. label == 1). This is deprecated for average != 'binary'.
    for kwargs in [{}, {"average": "binary"}]:
        with warnings.catch_warnings():
            warnings.simplefilter("error")

            ps = precision_score(y_true, y_pred, **kwargs)
            assert_array_almost_equal(ps, 0.85, 2)

            rs = recall_score(y_true, y_pred, **kwargs)
            assert_array_almost_equal(rs, 0.68, 2)

            fs = f1_score(y_true, y_pred, **kwargs)
            assert_array_almost_equal(fs, 0.76, 2)

            assert_almost_equal(
                fbeta_score(y_true, y_pred, beta=2, **kwargs),
                (1 + 2**2) * ps * rs / (2**2 * ps + rs),
                2,
            )


@pytest.mark.filterwarnings(r"ignore::sklearn.exceptions.UndefinedMetricWarning")
def test_precision_recall_f_binary_single_class():
    # [FUNCTION DOC]
    # Verifies PRF metrics when only one class appears: all-positive input scores 1,
    #   all-negative scores 0, and beta=inf matches a very large beta.
    # Test precision, recall and F-scores behave with a single positive or
    # negative class
    # Such a case may occur with non-stratified cross-validation
    assert 1.0 == precision_score([1, 1], [1, 1])
    assert 1.0 == recall_score([1, 1], [1, 1])
    assert 1.0 == f1_score([1, 1], [1, 1])
    assert 1.0 == fbeta_score([1, 1], [1, 1], beta=0)

    assert 0.0 == precision_score([-1, -1], [-1, -1])
    assert 0.0 == recall_score([-1, -1], [-1, -1])
    assert 0.0 == f1_score([-1, -1], [-1, -1])
    assert 0.0 == fbeta_score([-1, -1], [-1, -1], beta=float("inf"))
    assert fbeta_score([-1, -1], [-1, -1], beta=float("inf")) == pytest.approx(
        fbeta_score([-1, -1], [-1, -1], beta=1e5)
    )


@pytest.mark.filterwarnings(r"ignore::sklearn.exceptions.UndefinedMetricWarning")
def test_precision_recall_f_extra_labels():
    # [FUNCTION DOC]
    # Verifies recall with extra labels absent from the data: zeros appear for them with
    #   average=None, macro is diluted, micro/weighted/samples unchanged; invalid labels
    #   raise ValueError for multilabel input; and non-regression #10307 for samples
    #   average with a labels subset.
    # Test handling of explicit additional (not in input) labels to PRF
    y_true = [1, 3, 3, 2]
    y_pred = [1, 1, 3, 2]
    y_true_bin = label_binarize(y_true, classes=np.arange(5))
    y_pred_bin = label_binarize(y_pred, classes=np.arange(5))
    data = [(y_true, y_pred), (y_true_bin, y_pred_bin)]

    for i, (y_true, y_pred) in enumerate(data):
        # No average: zeros in array
        actual = recall_score(y_true, y_pred, labels=[0, 1, 2, 3, 4], average=None)
        assert_array_almost_equal([0.0, 1.0, 1.0, 0.5, 0.0], actual)

        # Macro average is changed
        actual = recall_score(y_true, y_pred, labels=[0, 1, 2, 3, 4], average="macro")
        assert_array_almost_equal(np.mean([0.0, 1.0, 1.0, 0.5, 0.0]), actual)

        # No effect otherwise
        for average in ["micro", "weighted", "samples"]:
            if average == "samples" and i == 0:
                continue
            assert_almost_equal(
                recall_score(y_true, y_pred, labels=[0, 1, 2, 3, 4], average=average),
                recall_score(y_true, y_pred, labels=None, average=average),
            )

    # Error when introducing invalid label in multilabel case
    # (although it would only affect performance if average='macro'/None)
    for average in [None, "macro", "micro", "samples"]:
        with pytest.raises(ValueError):
            recall_score(y_true_bin, y_pred_bin, labels=np.arange(6), average=average)
        with pytest.raises(ValueError):
            recall_score(
                y_true_bin, y_pred_bin, labels=np.arange(-1, 4), average=average
            )

    # tests non-regression on issue #10307
    y_true = np.array([[0, 1, 1], [1, 0, 0]])
    y_pred = np.array([[1, 1, 1], [1, 0, 1]])
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, average="samples", labels=[0, 1]
    )
    assert_almost_equal(np.array([p, r, f]), np.array([3 / 4, 1, 5 / 6]))


@pytest.mark.filterwarnings(r"ignore::sklearn.exceptions.UndefinedMetricWarning")
def test_precision_recall_f_ignored_labels():
    # [FUNCTION DOC]
    # Verifies requesting a labels subset ([1, 3]) for recall gives the expected per-
    #   class, macro, weighted and micro values (multiclass and binarised), and that
    #   they differ from using all labels.
    # Test a subset of labels may be requested for PRF
    y_true = [1, 1, 2, 3]
    y_pred = [1, 3, 3, 3]
    y_true_bin = label_binarize(y_true, classes=np.arange(5))
    y_pred_bin = label_binarize(y_pred, classes=np.arange(5))
    data = [(y_true, y_pred), (y_true_bin, y_pred_bin)]

    for i, (y_true, y_pred) in enumerate(data):
        recall_13 = partial(recall_score, y_true, y_pred, labels=[1, 3])
        recall_all = partial(recall_score, y_true, y_pred, labels=None)

        assert_array_almost_equal([0.5, 1.0], recall_13(average=None))
        assert_almost_equal((0.5 + 1.0) / 2, recall_13(average="macro"))
        assert_almost_equal((0.5 * 2 + 1.0 * 1) / 3, recall_13(average="weighted"))
        assert_almost_equal(2.0 / 3, recall_13(average="micro"))

        # ensure the above were meaningful tests:
        for average in ["macro", "weighted", "micro"]:
            assert recall_13(average=average) != recall_all(average=average)


def test_average_precision_score_non_binary_class():
    # [FUNCTION DOC]
    # Verifies average_precision_score raises ValueError for multiclass-multioutput
    #   y_true.
    """Test multiclass-multiouptut for `average_precision_score`."""
    y_true = np.array(
        [
            [2, 2, 1],
            [1, 2, 0],
            [0, 1, 2],
            [1, 2, 1],
            [2, 0, 1],
            [1, 2, 1],
        ]
    )
    y_score = np.array(
        [
            [0.7, 0.2, 0.1],
            [0.4, 0.3, 0.3],
            [0.1, 0.8, 0.1],
            [0.2, 0.3, 0.5],
            [0.4, 0.4, 0.2],
            [0.1, 0.2, 0.7],
        ]
    )
    err_msg = "multiclass-multioutput format is not supported"
    with pytest.raises(ValueError, match=err_msg):
        average_precision_score(y_true, y_score, pos_label=2)


@pytest.mark.parametrize(
    "y_true, y_score",
    [
        (
            [0, 0, 1, 2],
            np.array(
                [
                    [0.7, 0.2, 0.1],
                    [0.4, 0.3, 0.3],
                    [0.1, 0.8, 0.1],
                    [0.2, 0.3, 0.5],
                ]
            ),
        ),
        (
            [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
            [0, 0.1, 0.1, 0.4, 0.5, 0.6, 0.6, 0.9, 0.9, 1, 1],
        ),
    ],
)
def test_average_precision_score_duplicate_values(y_true, y_score):
    # [FUNCTION DOC]
    # Verifies average_precision_score is exactly 1 for perfectly ranked scores
    #   containing duplicate values (multiclass and binary).
    """
    Duplicate values with precision-recall require a different
    processing than when computing the AUC of a ROC, because the
    precision-recall curve is a decreasing curve
    The following situation corresponds to a perfect
    test statistic, the average_precision_score should be 1.
    """
    assert average_precision_score(y_true, y_score) == 1


@pytest.mark.parametrize(
    "y_true, y_score",
    [
        (
            [2, 2, 1, 1, 0],
            np.array(
                [
                    [0.2, 0.3, 0.5],
                    [0.2, 0.3, 0.5],
                    [0.4, 0.5, 0.3],
                    [0.4, 0.5, 0.3],
                    [0.8, 0.5, 0.3],
                ]
            ),
        ),
        (
            [0, 1, 1],
            [0.5, 0.5, 0.6],
        ),
    ],
)
def test_average_precision_score_tied_values(y_true, y_score):
    # [FUNCTION DOC]
    # Verifies average_precision_score is not 1 when ties mix positive and negative
    #   samples.
    # Here if we go from left to right in y_true, the 0 values are
    # separated from the 1 values, so it appears that we've
    # correctly sorted our classifications. But in fact the first two
    # values have the same score (0.5) and so the first two values
    # could be swapped around, creating an imperfect sorting. This
    # imperfection should come through in the end score, making it less
    # than one.
    assert average_precision_score(y_true, y_score) != 1.0


def test_precision_recall_f_unused_pos_label():
    # [FUNCTION DOC]
    # Verifies a UserWarning is raised when pos_label is non-default but average is not
    #   "binary".
    # Check warning that pos_label unused when set to non-default value
    # but average != 'binary'; even if data is binary.

    msg = (
        r"Note that pos_label \(set to 2\) is "
        r"ignored when average != 'binary' \(got 'macro'\). You "
        r"may use labels=\[pos_label\] to specify a single "
        "positive class."
    )
    with pytest.warns(UserWarning, match=msg):
        precision_recall_fscore_support(
            [1, 2, 1], [1, 2, 2], pos_label=2, average="macro"
        )


def test_confusion_matrix_binary():
    # [FUNCTION DOC]
    # Verifies the binary confusion matrix and Matthews correlation (against the closed-
    #   form formula and 0.57) for integer and string labels.
    # Test confusion matrix - binary classification case
    y_true, y_pred, _ = make_prediction(binary=True)

    def test(y_true, y_pred):
        # [FUNCTION DOC]
        # HELPER: Asserts the confusion matrix equals the known counts and that
        #   matthews_corrcoef matches the value computed from them.
        cm = confusion_matrix(y_true, y_pred)
        assert_array_equal(cm, [[22, 3], [8, 17]])

        tp, fp, fn, tn = cm.flatten()
        num = tp * tn - fp * fn
        den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

        true_mcc = 0 if den == 0 else num / den
        mcc = matthews_corrcoef(y_true, y_pred)
        assert_array_almost_equal(mcc, true_mcc, decimal=2)
        assert_array_almost_equal(mcc, 0.57, decimal=2)

    test(y_true, y_pred)
    test([str(y) for y in y_true], [str(y) for y in y_pred])


def test_multilabel_confusion_matrix_binary():
    # [FUNCTION DOC]
    # Verifies multilabel_confusion_matrix on binary predictions gives one 2x2 matrix
    #   per class, for int and string labels.
    # Test multilabel confusion matrix - binary classification case
    y_true, y_pred, _ = make_prediction(binary=True)

    def test(y_true, y_pred):
        # [FUNCTION DOC]
        # HELPER: Asserts the per-class 2x2 matrices equal the expected counts.
        cm = multilabel_confusion_matrix(y_true, y_pred)
        assert_array_equal(cm, [[[17, 8], [3, 22]], [[22, 3], [8, 17]]])

    test(y_true, y_pred)
    test([str(y) for y in y_true], [str(y) for y in y_pred])


def test_multilabel_confusion_matrix_multiclass():
    # [FUNCTION DOC]
    # Verifies multilabel_confusion_matrix on multiclass predictions with inferred
    #   labels, a reordered labels list and a superset containing an absent label (all-
    #   negative matrix).
    # Test multilabel confusion matrix - multi-class case
    y_true, y_pred, _ = make_prediction(binary=False)

    def test(y_true, y_pred, string_type=False):
        # [FUNCTION DOC]
        # HELPER: Runs the three label-order checks; string_type selects string label
        #   lists.
        # compute confusion matrix with default labels introspection
        cm = multilabel_confusion_matrix(y_true, y_pred)
        assert_array_equal(
            cm, [[[47, 4], [5, 19]], [[38, 6], [28, 3]], [[30, 25], [2, 18]]]
        )

        # compute confusion matrix with explicit label ordering
        labels = ["0", "2", "1"] if string_type else [0, 2, 1]
        cm = multilabel_confusion_matrix(y_true, y_pred, labels=labels)
        assert_array_equal(
            cm, [[[47, 4], [5, 19]], [[30, 25], [2, 18]], [[38, 6], [28, 3]]]
        )

        # compute confusion matrix with super set of present labels
        labels = ["0", "2", "1", "3"] if string_type else [0, 2, 1, 3]
        cm = multilabel_confusion_matrix(y_true, y_pred, labels=labels)
        assert_array_equal(
            cm,
            [
                [[47, 4], [5, 19]],
                [[30, 25], [2, 18]],
                [[38, 6], [28, 3]],
                [[75, 0], [0, 0]],
            ],
        )

    test(y_true, y_pred)
    test([str(y) for y in y_true], [str(y) for y in y_pred], string_type=True)


@pytest.mark.parametrize("csc_container", CSC_CONTAINERS)
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_multilabel_confusion_matrix_multilabel(csc_container, csr_container):
    # [FUNCTION DOC]
    # Verifies multilabel_confusion_matrix on indicator input across dense/CSR/CSC
    #   combinations, samplewise mode, label selection and sample weights.
    # Test multilabel confusion matrix - multilabel-indicator case

    y_true = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 0]])
    y_pred = np.array([[1, 0, 0], [0, 1, 1], [0, 0, 1]])
    y_true_csr = csr_container(y_true)
    y_pred_csr = csr_container(y_pred)
    y_true_csc = csc_container(y_true)
    y_pred_csc = csc_container(y_pred)

    # cross test different types
    sample_weight = np.array([2, 1, 3])
    real_cm = [[[1, 0], [1, 1]], [[1, 0], [1, 1]], [[0, 2], [1, 0]]]
    trues = [y_true, y_true_csr, y_true_csc]
    preds = [y_pred, y_pred_csr, y_pred_csc]

    for y_true_tmp in trues:
        for y_pred_tmp in preds:
            cm = multilabel_confusion_matrix(y_true_tmp, y_pred_tmp)
            assert_array_equal(cm, real_cm)

    # test support for samplewise
    cm = multilabel_confusion_matrix(y_true, y_pred, samplewise=True)
    assert_array_equal(cm, [[[1, 0], [1, 1]], [[1, 1], [0, 1]], [[0, 1], [2, 0]]])

    # test support for labels
    cm = multilabel_confusion_matrix(y_true, y_pred, labels=[2, 0])
    assert_array_equal(cm, [[[0, 2], [1, 0]], [[1, 0], [1, 1]]])

    # test support for labels with samplewise
    cm = multilabel_confusion_matrix(y_true, y_pred, labels=[2, 0], samplewise=True)
    assert_array_equal(cm, [[[0, 0], [1, 1]], [[1, 1], [0, 0]], [[0, 1], [1, 0]]])

    # test support for sample_weight with sample_wise
    cm = multilabel_confusion_matrix(
        y_true, y_pred, sample_weight=sample_weight, samplewise=True
    )
    assert_array_equal(cm, [[[2, 0], [2, 2]], [[1, 1], [0, 1]], [[0, 3], [6, 0]]])


def test_multilabel_confusion_matrix_errors():
    # [FUNCTION DOC]
    # Verifies ValueError for mismatched or 2-D sample_weight, out-of-range labels,
    #   samplewise on non-multilabel data and multiclass-multioutput input.
    y_true = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 0]])
    y_pred = np.array([[1, 0, 0], [0, 1, 1], [0, 0, 1]])

    # Bad sample_weight
    with pytest.raises(ValueError, match="inconsistent numbers of samples"):
        multilabel_confusion_matrix(y_true, y_pred, sample_weight=[1, 2])
    with pytest.raises(ValueError, match="Sample weights must be 1D array or scalar"):
        multilabel_confusion_matrix(
            y_true, y_pred, sample_weight=[[1, 2, 3], [2, 3, 4], [3, 4, 5]]
        )

    # Bad labels
    err_msg = r"All labels must be in \[0, n labels\)"
    with pytest.raises(ValueError, match=err_msg):
        multilabel_confusion_matrix(y_true, y_pred, labels=[-1])
    err_msg = r"All labels must be in \[0, n labels\)"
    with pytest.raises(ValueError, match=err_msg):
        multilabel_confusion_matrix(y_true, y_pred, labels=[3])

    # Using samplewise outside multilabel
    with pytest.raises(ValueError, match="Samplewise metrics"):
        multilabel_confusion_matrix([0, 1, 2], [1, 2, 0], samplewise=True)

    # Bad y_type
    err_msg = "multiclass-multioutput is not supported"
    with pytest.raises(ValueError, match=err_msg):
        multilabel_confusion_matrix([[0, 1, 2], [2, 1, 0]], [[1, 2, 0], [1, 0, 2]])


@pytest.mark.parametrize(
    "normalize, cm_dtype, expected_results",
    [
        ("true", "f", 0.333333333),
        ("pred", "f", 0.333333333),
        ("all", "f", 0.1111111111),
        (None, "i", 2),
    ],
)
def test_confusion_matrix_normalize(normalize, cm_dtype, expected_results):
    # [FUNCTION DOC]
    # Verifies each confusion_matrix normalize mode gives the expected constant value
    #   and dtype kind (float when normalised, int otherwise).
    y_test = [0, 1, 2] * 6
    y_pred = list(chain(*permutations([0, 1, 2])))
    cm = confusion_matrix(y_test, y_pred, normalize=normalize)
    assert_allclose(cm, expected_results)
    assert cm.dtype.kind == cm_dtype


def test_confusion_matrix_normalize_single_class():
    # [FUNCTION DOC]
    # Verifies normalisation with a class never predicted (or never true) yields correct
    #   sums and no RuntimeWarning from division by zero.
    y_test = [0, 0, 0, 0, 1, 1, 1, 1]
    y_pred = [0, 0, 0, 0, 0, 0, 0, 0]

    cm_true = confusion_matrix(y_test, y_pred, normalize="true")
    assert cm_true.sum() == pytest.approx(2.0)

    # additionally check that no warnings are raised due to a division by zero
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        cm_pred = confusion_matrix(y_test, y_pred, normalize="pred")

    assert cm_pred.sum() == pytest.approx(1.0)

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        confusion_matrix(y_pred, y_test, normalize="true")


def test_confusion_matrix_single_label():
    # [FUNCTION DOC]
    # Verifies confusion_matrix warns when only one label is present.
    """Test `confusion_matrix` warns when only one label found."""
    y_test = [0, 0, 0, 0]
    y_pred = [0, 0, 0, 0]

    with pytest.warns(UserWarning, match="A single label was found in"):
        confusion_matrix(y_pred, y_test)


@pytest.mark.parametrize(
    "params, warn_msg",
    [
        # When `fp == 0` and `tp != 0`, LR+ is undefined
        (
            {
                "y_true": np.array([1, 1, 1, 0, 0, 0]),
                "y_pred": np.array([1, 1, 1, 0, 0, 0]),
            },
            "`positive_likelihood_ratio` is ill-defined and set to `np.nan`.",
        ),
        # When `fp == 0` and `tp == 0`, LR+ is undefined
        (
            {
                "y_true": np.array([1, 1, 1, 0, 0, 0]),
                "y_pred": np.array([0, 0, 0, 0, 0, 0]),
            },
            (
                "No samples were predicted for the positive class and "
                "`positive_likelihood_ratio` is set to `np.nan`."
            ),
        ),
        # When `tn == 0`, LR- is undefined
        (
            {
                "y_true": np.array([1, 1, 1, 0, 0, 0]),
                "y_pred": np.array([0, 0, 0, 1, 1, 1]),
            },
            "`negative_likelihood_ratio` is ill-defined and set to `np.nan`.",
        ),
        # When `tp + fn == 0` both ratios are undefined
        (
            {
                "y_true": np.array([0, 0, 0, 0, 0, 0]),
                "y_pred": np.array([1, 1, 1, 0, 0, 0]),
            },
            "No samples of the positive class are present in `y_true`.",
        ),
    ],
)
def test_likelihood_ratios_warnings(params, warn_msg):
    # [FUNCTION DOC]
    # Verifies class_likelihood_ratios emits the expected UserWarning for each ill-
    #   defined case (fp=0, no positive predictions, tn=0, no positives).
    # likelihood_ratios must raise warnings when at
    # least one of the ratios is ill-defined.

    with pytest.warns(UserWarning, match=warn_msg):
        class_likelihood_ratios(**params)


@pytest.mark.parametrize(
    "params, err_msg",
    [
        (
            {
                "y_true": np.array([0, 1, 0, 1, 0]),
                "y_pred": np.array([1, 1, 0, 0, 2]),
            },
            (
                "class_likelihood_ratios only supports binary classification "
                "problems, got targets of type: multiclass"
            ),
        ),
    ],
)
def test_likelihood_ratios_errors(params, err_msg):
    # [FUNCTION DOC]
    # Verifies class_likelihood_ratios raises ValueError for non-binary targets.
    # likelihood_ratios must raise error when attempting
    # non-binary classes to avoid Simpson's paradox
    with pytest.raises(ValueError, match=err_msg):
        class_likelihood_ratios(**params)


def test_likelihood_ratios():
    # [FUNCTION DOC]
    # Verifies LR+ and LR- on a constructed confusion matrix, the perfect-prediction
    #   limit (LR+ NaN, LR- 0) and the effect of zero sample weights.
    # Build confusion matrix with tn=9, fp=8, fn=1, tp=2,
    # sensitivity=2/3, specificity=9/17, prevalence=3/20,
    # LR+=34/24, LR-=17/27
    y_true = np.array([1] * 3 + [0] * 17)
    y_pred = np.array([1] * 2 + [0] * 10 + [1] * 8)

    pos, neg = class_likelihood_ratios(y_true, y_pred)
    assert_allclose(pos, 34 / 24)
    assert_allclose(neg, 17 / 27)

    # Build limit case with y_pred = y_true
    pos, neg = class_likelihood_ratios(y_true, y_true)
    assert_array_equal(pos, np.nan * 2)
    assert_allclose(neg, np.zeros(2), rtol=1e-12)

    # Ignore last 5 samples to get tn=9, fp=3, fn=1, tp=2,
    # sensitivity=2/3, specificity=9/12, prevalence=3/20,
    # LR+=24/9, LR-=12/27
    sample_weight = np.array([1.0] * 15 + [0.0] * 5)
    pos, neg = class_likelihood_ratios(y_true, y_pred, sample_weight=sample_weight)
    assert_allclose(pos, 24 / 9)
    assert_allclose(neg, 12 / 27)


def test_likelihood_ratios_replace_undefined_by_worst():
    # [FUNCTION DOC]
    # Verifies replace_undefined_by=1 yields 1.0 for an undefined LR+ (fp=0) and an
    #   undefined LR- (tn=0).
    """Test that class_likelihood_ratios returns the worst scores `1.0` for both LR+ and
    LR- when `replace_undefined_by=1` is set."""
    # This data causes fp=0 (0 false positives) in the confusion_matrix and a division
    # by zero that affects the positive_likelihood_ratio:
    y_true = np.array([1, 1, 0])
    y_pred = np.array([1, 0, 0])

    positive_likelihood_ratio, _ = class_likelihood_ratios(
        y_true, y_pred, replace_undefined_by=1
    )
    assert positive_likelihood_ratio == pytest.approx(1.0)

    # This data causes tn=0 (0 true negatives) in the confusion_matrix and a division
    # by zero that affects the negative_likelihood_ratio:
    y_true = np.array([1, 0, 0])
    y_pred = np.array([1, 1, 1])

    _, negative_likelihood_ratio = class_likelihood_ratios(
        y_true, y_pred, replace_undefined_by=1
    )
    assert negative_likelihood_ratio == pytest.approx(1.0)


@pytest.mark.parametrize(
    "replace_undefined_by",
    [
        {"LR+": 0.0},
        {"LR-": 0.0},
        {"LR+": -5.0, "LR-": 0.0},
        {"LR+": 1.0, "LR-": "nan"},
        {"LR+": 0.0, "LR-": 0.0},
        {"LR+": 1.0, "LR-": 2.0},
    ],
)
def test_likelihood_ratios_wrong_dict_replace_undefined_by(replace_undefined_by):
    # [FUNCTION DOC]
    # Verifies ValueError for malformed or out-of-range replace_undefined_by
    #   dictionaries.
    """Test that class_likelihood_ratios raises a `ValueError` if the input dict for
    `replace_undefined_by` is in the wrong format or contains impossible values."""
    y_true = np.array([1, 0])
    y_pred = np.array([1, 0])

    msg = "The dictionary passed as `replace_undefined_by` needs to be in the form"
    with pytest.raises(ValueError, match=msg):
        class_likelihood_ratios(
            y_true, y_pred, replace_undefined_by=replace_undefined_by
        )


@pytest.mark.parametrize(
    "replace_undefined_by, expected",
    [
        ({"LR+": 1.0, "LR-": 1.0}, 1.0),
        ({"LR+": np.inf, "LR-": 0.0}, np.inf),
        ({"LR+": 2.0, "LR-": 0.0}, 2.0),
        ({"LR+": np.nan, "LR-": np.nan}, np.nan),
        (np.nan, np.nan),
    ],
)
def test_likelihood_ratios_replace_undefined_by_0_fp(replace_undefined_by, expected):
    # [FUNCTION DOC]
    # Verifies the LR+ replacement value is returned when fp=0, for several dicts and a
    #   scalar NaN.
    """Test that the `replace_undefined_by` param returns the right value for the
    positive_likelihood_ratio as defined by the user."""
    # This data causes fp=0 (0 false positives) in the confusion_matrix and a division
    # by zero that affects the positive_likelihood_ratio:
    y_true = np.array([1, 1, 0])
    y_pred = np.array([1, 0, 0])

    positive_likelihood_ratio, _ = class_likelihood_ratios(
        y_true, y_pred, replace_undefined_by=replace_undefined_by
    )

    if np.isnan(expected):
        assert np.isnan(positive_likelihood_ratio)
    else:
        assert positive_likelihood_ratio == pytest.approx(expected)


@pytest.mark.parametrize(
    "replace_undefined_by, expected",
    [
        ({"LR+": 1.0, "LR-": 1.0}, 1.0),
        ({"LR+": np.inf, "LR-": 0.0}, 0.0),
        ({"LR+": np.inf, "LR-": 0.5}, 0.5),
        ({"LR+": np.nan, "LR-": np.nan}, np.nan),
        (np.nan, np.nan),
    ],
)
def test_likelihood_ratios_replace_undefined_by_0_tn(replace_undefined_by, expected):
    # [FUNCTION DOC]
    # Verifies the LR- replacement value is returned when tn=0, for several dicts and a
    #   scalar NaN.
    """Test that the `replace_undefined_by` param returns the right value for the
    negative_likelihood_ratio as defined by the user."""
    # This data causes tn=0 (0 true negatives) in the confusion_matrix and a division
    # by zero that affects the negative_likelihood_ratio:
    y_true = np.array([1, 0, 0])
    y_pred = np.array([1, 1, 1])

    _, negative_likelihood_ratio = class_likelihood_ratios(
        y_true, y_pred, replace_undefined_by=replace_undefined_by
    )

    if np.isnan(expected):
        assert np.isnan(negative_likelihood_ratio)
    else:
        assert negative_likelihood_ratio == pytest.approx(expected)


def test_cohen_kappa():
    # [FUNCTION DOC]
    # Verifies cohen_kappa_score against Artstein and Poesio (2008) examples: binary
    #   symmetry, ignoring extra labels via labels, perfect agreement, multiclass, and
    #   linear/quadratic weights.
    # These label vectors reproduce the contingency matrix from Artstein and
    # Poesio (2008), Table 1: np.array([[20, 20], [10, 50]]).
    y1 = np.array([0] * 40 + [1] * 60)
    y2 = np.array([0] * 20 + [1] * 20 + [0] * 10 + [1] * 50)
    kappa = cohen_kappa_score(y1, y2)
    assert_almost_equal(kappa, 0.348, decimal=3)
    assert kappa == cohen_kappa_score(y2, y1)

    # Add spurious labels and ignore them.
    y1 = np.append(y1, [2] * 4)
    y2 = np.append(y2, [2] * 4)
    assert cohen_kappa_score(y1, y2, labels=[0, 1]) == kappa

    assert_almost_equal(cohen_kappa_score(y1, y1), 1.0)

    # Multiclass example: Artstein and Poesio, Table 4.
    y1 = np.array([0] * 46 + [1] * 44 + [2] * 10)
    y2 = np.array([0] * 52 + [1] * 32 + [2] * 16)
    assert_almost_equal(cohen_kappa_score(y1, y2), 0.8013, decimal=4)

    # Weighting example: none, linear, quadratic.
    y1 = np.array([0] * 46 + [1] * 44 + [2] * 10)
    y2 = np.array([0] * 50 + [1] * 40 + [2] * 10)
    assert_almost_equal(cohen_kappa_score(y1, y2), 0.9315, decimal=4)
    assert_almost_equal(cohen_kappa_score(y1, y2, weights="linear"), 0.9412, decimal=4)
    assert_almost_equal(
        cohen_kappa_score(y1, y2, weights="quadratic"), 0.9541, decimal=4
    )


@ignore_warnings(category=UndefinedMetricWarning)
@pytest.mark.parametrize(
    "test_case",
    [
        # annotator y2 does not assign any label specified in `labels` (note: also
        # applicable if `labels` is default and `y2` does not contain any label that is
        # in `y1`):
        ([1] * 5 + [2] * 5, [3] * 10, [1, 2], None),
        # both inputs (`y1` and `y2`) only have one label:
        ([3] * 10, [3] * 10, None, None),
        # both inputs only have one label in common that is also in `labels`:
        ([1] * 5 + [2] * 5, [1] * 5 + [3] * 5, [1, 2], None),
        # like the last test case, but with `weights="linear"` (note that
        # weights="linear" and weights="quadratic" are different branches, though the
        # latter is so similar to the former that the test case is skipped here):
        ([1] * 5 + [2] * 5, [1] * 5 + [3] * 5, [1, 2], "linear"),
    ],
)
@pytest.mark.parametrize("replace_undefined_by", [0.0, np.nan])
def test_cohen_kappa_undefined(test_case, replace_undefined_by):
    # [FUNCTION DOC]
    # Verifies cohen_kappa_score returns replace_undefined_by (0 or NaN) in each zero-
    #   division scenario.
    """Test that cohen_kappa_score handles divisions by 0 correctly by returning the
    `replace_undefined_by` param. (The first test case covers the first possible
    location in the function for an occurrence of a division by zero, the last three
    test cases cover a zero division in the second possible location in the
    function."""

    y1, y2, labels, weights = test_case
    y1, y2 = np.array(y1), np.array(y2)

    score = cohen_kappa_score(
        y1,
        y2,
        labels=labels,
        weights=weights,
        replace_undefined_by=replace_undefined_by,
    )
    assert_allclose(score, replace_undefined_by, equal_nan=True)


def test_cohen_kappa_zero_division_warning():
    # [FUNCTION DOC]
    # Verifies the two UndefinedMetricWarning messages of cohen_kappa_score for its
    #   zero-division cases.
    """Test that cohen_kappa_score raises UndefinedMetricWarning when a division by 0
    occurs."""

    labels = [1, 2]
    y1 = np.array([1] * 5 + [2] * 5)
    y2 = np.array([3] * 10)
    with pytest.warns(
        UndefinedMetricWarning,
        match="`y2` contains no labels that are present in both `y1` and `labels`.",
    ):
        cohen_kappa_score(y1, y2, labels=labels)

    labels = [1, 2]
    y1 = np.array([1] * 5 + [2] * 5)
    y2 = np.array([1] * 5 + [3] * 5)
    with pytest.warns(
        UndefinedMetricWarning,
        match="`y1`, `y2` and `labels` have only one label in common.",
    ):
        cohen_kappa_score(y1, y2, labels=labels)


def test_cohen_kappa_score_error_wrong_label():
    # [FUNCTION DOC]
    # Verifies ValueError when none of labels appears in y1.
    """Test that correct error is raised when users pass labels that are not in y1."""
    labels = [1, 2]
    y1 = np.array(["a"] * 5 + ["b"] * 5)
    y2 = np.array(["b"] * 10)
    with pytest.raises(
        ValueError, match="At least one label in `labels` must be present in `y1`"
    ):
        cohen_kappa_score(y1, y2, labels=labels)


@pytest.mark.parametrize("zero_division", [0, 1, np.nan])
@pytest.mark.parametrize("y_true, y_pred", [([0], [0])])
@pytest.mark.parametrize(
    "metric",
    [
        f1_score,
        partial(fbeta_score, beta=1),
        precision_score,
        recall_score,
    ],
)
def test_zero_division_nan_no_warning(metric, y_true, y_pred, zero_division):
    # [FUNCTION DOC]
    # Verifies F1/F-beta/precision/recall return zero_division (0, 1 or NaN) without
    #   warnings for a degenerate single negative sample.
    """Check the behaviour of `zero_division` when setting to 0, 1 or np.nan.
    No warnings should be raised.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = metric(y_true, y_pred, zero_division=zero_division)

    if np.isnan(zero_division):
        assert np.isnan(result)
    else:
        assert result == zero_division


@pytest.mark.parametrize("y_true, y_pred", [([0], [0])])
@pytest.mark.parametrize(
    "metric",
    [
        f1_score,
        partial(fbeta_score, beta=1),
        precision_score,
        recall_score,
    ],
)
def test_zero_division_nan_warning(metric, y_true, y_pred):
    # [FUNCTION DOC]
    # Verifies zero_division="warn" emits UndefinedMetricWarning and returns 0.0.
    """Check the behaviour of `zero_division` when setting to "warn".
    A `UndefinedMetricWarning` should be raised.
    """
    with pytest.warns(UndefinedMetricWarning):
        result = metric(y_true, y_pred, zero_division="warn")
    assert result == 0.0


def test_matthews_corrcoef_against_numpy_corrcoef(global_random_seed):
    # [FUNCTION DOC]
    # Verifies binary matthews_corrcoef equals the Pearson correlation from np.corrcoef
    #   on random labels.
    rng = np.random.RandomState(global_random_seed)
    y_true = rng.randint(0, 2, size=20)
    y_pred = rng.randint(0, 2, size=20)

    assert_almost_equal(
        matthews_corrcoef(y_true, y_pred), np.corrcoef(y_true, y_pred)[0, 1], 10
    )


def test_matthews_corrcoef_against_jurman(global_random_seed):
    # [FUNCTION DOC]
    # Verifies weighted matthews_corrcoef equals the multiclass definition of Jurman et
    #   al. (2012) computed from the confusion matrix.
    # Check that the multiclass matthews_corrcoef agrees with the definition
    # presented in Jurman, Riccadonna, Furlanello, (2012). A Comparison of MCC
    # and CEN Error Measures in MultiClass Prediction
    rng = np.random.RandomState(global_random_seed)
    y_true = rng.randint(0, 2, size=20)
    y_pred = rng.randint(0, 2, size=20)
    sample_weight = rng.rand(20)

    C = confusion_matrix(y_true, y_pred, sample_weight=sample_weight)
    N = len(C)
    cov_ytyp = sum(
        [
            C[k, k] * C[m, l] - C[l, k] * C[k, m]
            for k in range(N)
            for m in range(N)
            for l in range(N)
        ]
    )
    cov_ytyt = sum(
        [
            C[:, k].sum()
            * np.sum([C[g, f] for f in range(N) for g in range(N) if f != k])
            for k in range(N)
        ]
    )
    cov_ypyp = np.sum(
        [
            C[k, :].sum()
            * np.sum([C[f, g] for f in range(N) for g in range(N) if f != k])
            for k in range(N)
        ]
    )
    mcc_jurman = cov_ytyp / np.sqrt(cov_ytyt * cov_ypyp)
    mcc_ours = matthews_corrcoef(y_true, y_pred, sample_weight=sample_weight)

    assert_almost_equal(mcc_ours, mcc_jurman, 10)


def test_matthews_corrcoef(global_random_seed):
    # [FUNCTION DOC]
    # Verifies binary matthews_corrcoef on string labels: 1 for identical, -1 for
    #   inverted vectors, 0 for constant predictions with a warning-free result, and
    #   sample-weight behaviour.
    rng = np.random.RandomState(global_random_seed)
    y_true = ["a" if i == 0 else "b" for i in rng.randint(0, 2, size=20)]

    # corrcoef of same vectors must be 1
    assert_almost_equal(matthews_corrcoef(y_true, y_true), 1.0)

    # corrcoef, when the two vectors are opposites of each other, should be -1
    y_true_inv = ["b" if i == "a" else "a" for i in y_true]
    assert_almost_equal(matthews_corrcoef(y_true, y_true_inv), -1)

    y_true_inv2 = label_binarize(y_true, classes=["a", "b"])
    y_true_inv2 = np.where(y_true_inv2, "a", "b")
    assert_almost_equal(matthews_corrcoef(y_true, y_true_inv2), -1)

    # For the zero vector case, the corrcoef cannot be calculated and should
    # output 0
    assert_almost_equal(matthews_corrcoef([0, 0, 0, 0], [0, 0, 0, 0]), 0.0)

    # And also for any other vector with 0 variance
    assert_almost_equal(matthews_corrcoef(y_true, ["a"] * len(y_true)), 0.0)

    # These two vectors have 0 correlation and hence mcc should be 0
    y_1 = [1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1]
    y_2 = [1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1]
    assert_almost_equal(matthews_corrcoef(y_1, y_2), 0.0)

    # Check that sample weight is able to selectively exclude
    mask = [1] * 10 + [0] * 10
    # Now the first half of the vector elements are alone given a weight of 1
    # and hence the mcc will not be a perfect 0 as in the previous case
    with pytest.raises(AssertionError):
        assert_almost_equal(matthews_corrcoef(y_1, y_2, sample_weight=mask), 0.0)


def test_matthews_corrcoef_multiclass(global_random_seed):
    # [FUNCTION DOC]
    # Verifies multiclass matthews_corrcoef: 1 for identical string labels, known
    #   negative values for adversarial predictions, 0 for zero-variance predictions or
    #   truth and for uncorrelated vectors, -1 when sample weights mask a third class,
    #   and 0 when weights leave a degenerate subset.
    rng = np.random.RandomState(global_random_seed)
    ord_a = ord("a")
    n_classes = 4
    y_true = [chr(ord_a + i) for i in rng.randint(0, n_classes, size=20)]

    # corrcoef of same vectors must be 1
    assert_almost_equal(matthews_corrcoef(y_true, y_true), 1.0)

    # with multiclass > 2 it is not possible to achieve -1
    y_true = [0, 0, 1, 1, 2, 2]
    y_pred_bad = [2, 2, 0, 0, 1, 1]
    assert_almost_equal(matthews_corrcoef(y_true, y_pred_bad), -0.5)

    # Maximizing false positives and negatives minimizes the MCC
    # The minimum will be different for depending on the input
    y_true = [0, 0, 1, 1, 2, 2]
    y_pred_min = [1, 1, 0, 0, 0, 0]
    assert_almost_equal(matthews_corrcoef(y_true, y_pred_min), -12 / np.sqrt(24 * 16))

    # Zero variance will result in an mcc of zero
    y_true = [0, 1, 2]
    y_pred = [3, 3, 3]
    assert_almost_equal(matthews_corrcoef(y_true, y_pred), 0.0)

    # Also for ground truth with zero variance
    y_true = [3, 3, 3]
    y_pred = [0, 1, 2]
    assert_almost_equal(matthews_corrcoef(y_true, y_pred), 0.0)

    # These two vectors have 0 correlation and hence mcc should be 0
    y_1 = [0, 1, 2, 0, 1, 2, 0, 1, 2]
    y_2 = [1, 1, 1, 2, 2, 2, 0, 0, 0]
    assert_almost_equal(matthews_corrcoef(y_1, y_2), 0.0)

    # We can test that binary assumptions hold using the multiclass computation
    # by masking the weight of samples not in the first two classes

    # Masking the last label should let us get an MCC of -1
    y_true = [0, 0, 1, 1, 2]
    y_pred = [1, 1, 0, 0, 2]
    sample_weight = [1, 1, 1, 1, 0]
    assert_almost_equal(
        matthews_corrcoef(y_true, y_pred, sample_weight=sample_weight), -1
    )

    # For the zero vector case, the corrcoef cannot be calculated and should
    # output 0
    y_true = [0, 0, 1, 2]
    y_pred = [0, 0, 1, 2]
    sample_weight = [1, 1, 0, 0]
    assert_almost_equal(
        matthews_corrcoef(y_true, y_pred, sample_weight=sample_weight), 0.0
    )


@pytest.mark.parametrize("n_points", [100, 10000])
def test_matthews_corrcoef_overflow(n_points, global_random_seed):
    # [FUNCTION DOC]
    # Non-regression for #9622: matthews_corrcoef must not overflow for large n_points;
    #   checks perfect agreement gives 1 for binary and multiclass repeated arrays and
    #   that random noisy binary predictions match an independent rate-based formula.
    # https://github.com/scikit-learn/scikit-learn/issues/9622
    rng = np.random.RandomState(global_random_seed)

    def mcc_safe(y_true, y_pred):
        # [FUNCTION DOC]
        # HELPER: Reference binary MCC computed from rates (normalised by n_points) to
        #   avoid integer overflow.
        # KNOWN ISSUE: false_pos reads entry [1, 0] (true 1, predicted 0), which is
        #   actually a false negative, and false_neg reads [0, 1], a false positive; the
        #   names are swapped but the product in the denominator is symmetric so the
        #   result is still correct.
        conf_matrix = confusion_matrix(y_true, y_pred)
        true_pos = conf_matrix[1, 1]
        false_pos = conf_matrix[1, 0]
        false_neg = conf_matrix[0, 1]
        n_points = len(y_true)
        pos_rate = (true_pos + false_neg) / n_points
        activity = (true_pos + false_pos) / n_points
        mcc_numerator = true_pos / n_points - pos_rate * activity
        mcc_denominator = activity * pos_rate * (1 - activity) * (1 - pos_rate)
        return mcc_numerator / np.sqrt(mcc_denominator)

    def random_ys(n_points):  # binary
        # [FUNCTION DOC]
        # HELPER: Draws n_points uniform values as true scores and adds uniform noise of
        #   width 0.2 for predictions, thresholding both at 0.5. Uses the enclosing
        #   test's rng.
        # Outputs: (y_true, y_pred) boolean arrays.
        x_true = rng.random_sample(n_points)
        x_pred = x_true + 0.2 * (rng.random_sample(n_points) - 0.5)
        y_true = x_true > 0.5
        y_pred = x_pred > 0.5
        return y_true, y_pred

    arr = np.repeat([0.0, 1.0], n_points)  # binary
    assert_almost_equal(matthews_corrcoef(arr, arr), 1.0)
    arr = np.repeat([0.0, 1.0, 2.0], n_points)  # multiclass
    assert_almost_equal(matthews_corrcoef(arr, arr), 1.0)

    y_true, y_pred = random_ys(n_points)
    assert_almost_equal(matthews_corrcoef(y_true, y_true), 1.0)
    assert_almost_equal(matthews_corrcoef(y_true, y_pred), mcc_safe(y_true, y_pred))


def test_precision_recall_f1_score_multiclass():
    # [FUNCTION DOC]
    # Verifies multiclass per-class and micro/macro/weighted precision, recall and F1 on
    #   make_prediction output, that average="samples" raises ValueError for non-
    #   multilabel data, and that explicit label ordering permutes per-class results.
    # Test Precision Recall and F1 Score for multiclass classification task
    y_true, y_pred, _ = make_prediction(binary=False)

    # compute scores with default labels introspection
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, average=None)
    assert_array_almost_equal(p, [0.83, 0.33, 0.42], 2)
    assert_array_almost_equal(r, [0.79, 0.09, 0.90], 2)
    assert_array_almost_equal(f, [0.81, 0.15, 0.57], 2)
    assert_array_equal(s, [24, 31, 20])

    # averaging tests
    ps = precision_score(y_true, y_pred, pos_label=1, average="micro")
    assert_array_almost_equal(ps, 0.53, 2)

    rs = recall_score(y_true, y_pred, average="micro")
    assert_array_almost_equal(rs, 0.53, 2)

    fs = f1_score(y_true, y_pred, average="micro")
    assert_array_almost_equal(fs, 0.53, 2)

    ps = precision_score(y_true, y_pred, average="macro")
    assert_array_almost_equal(ps, 0.53, 2)

    rs = recall_score(y_true, y_pred, average="macro")
    assert_array_almost_equal(rs, 0.60, 2)

    fs = f1_score(y_true, y_pred, average="macro")
    assert_array_almost_equal(fs, 0.51, 2)

    ps = precision_score(y_true, y_pred, average="weighted")
    assert_array_almost_equal(ps, 0.51, 2)

    rs = recall_score(y_true, y_pred, average="weighted")
    assert_array_almost_equal(rs, 0.53, 2)

    fs = f1_score(y_true, y_pred, average="weighted")
    assert_array_almost_equal(fs, 0.47, 2)

    with pytest.raises(ValueError):
        precision_score(y_true, y_pred, average="samples")
    with pytest.raises(ValueError):
        recall_score(y_true, y_pred, average="samples")
    with pytest.raises(ValueError):
        f1_score(y_true, y_pred, average="samples")
    with pytest.raises(ValueError):
        fbeta_score(y_true, y_pred, average="samples", beta=0.5)

    # same prediction but with and explicit label ordering
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 2, 1], average=None
    )
    assert_array_almost_equal(p, [0.83, 0.41, 0.33], 2)
    assert_array_almost_equal(r, [0.79, 0.90, 0.10], 2)
    assert_array_almost_equal(f, [0.81, 0.57, 0.15], 2)
    assert_array_equal(s, [24, 20, 31])


@pytest.mark.parametrize("average", ["samples", "micro", "macro", "weighted", None])
def test_precision_refcall_f1_score_multilabel_unordered_labels(average):
    # [FUNCTION DOC]
    # Verifies multilabel PRF accepts an unsorted labels list: all-wrong predictions
    #   give zeros for every average and supports follow the given label order.
    # test that labels need not be sorted in the multilabel case
    y_true = np.array([[1, 1, 0, 0]])
    y_pred = np.array([[0, 0, 1, 1]])
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=[3, 0, 1, 2], warn_for=[], average=average
    )
    assert_array_equal(p, 0)
    assert_array_equal(r, 0)
    assert_array_equal(f, 0)
    if average is None:
        assert_array_equal(s, [0, 1, 1, 0])


def test_precision_recall_f1_score_binary_averaged():
    # [FUNCTION DOC]
    # Verifies on binary data that macro averages equal the mean of per-class scores and
    #   weighted averages equal support-weighted means.
    y_true = np.array([0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1])
    y_pred = np.array([1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1])

    # compute scores with default labels introspection
    ps, rs, fs, _ = precision_recall_fscore_support(y_true, y_pred, average=None)
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="macro")
    assert p == np.mean(ps)
    assert r == np.mean(rs)
    assert f == np.mean(fs)
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")
    support = np.bincount(y_true)
    assert p == np.average(ps, weights=support)
    assert r == np.average(rs, weights=support)
    assert f == np.average(fs, weights=support)


def test_zero_precision_recall():
    # [FUNCTION DOC]
    # Verifies macro precision/recall/F1 are 0 (no NaN or floating point error) when all
    #   predictions are wrong, with numpy errors set to raise.
    # Side effects: temporarily changes global numpy error settings, restored in a
    #   finally block.
    # Check that pathological cases do not bring NaNs

    old_error_settings = np.seterr(all="raise")

    try:
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([2, 0, 1, 1, 2, 0])

        assert_almost_equal(precision_score(y_true, y_pred, average="macro"), 0.0, 2)
        assert_almost_equal(recall_score(y_true, y_pred, average="macro"), 0.0, 2)
        assert_almost_equal(f1_score(y_true, y_pred, average="macro"), 0.0, 2)

    finally:
        np.seterr(**old_error_settings)


def test_confusion_matrix_multiclass_subset_labels():
    # [FUNCTION DOC]
    # Verifies confusion_matrix with label subsets, reordering, and a label absent from
    #   the data (zero row and column).
    # Test confusion matrix - multi-class case with subset of labels
    y_true, y_pred, _ = make_prediction(binary=False)

    # compute confusion matrix with only first two labels considered
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    assert_array_equal(cm, [[19, 4], [4, 3]])

    # compute confusion matrix with explicit label ordering for only subset
    # of labels
    cm = confusion_matrix(y_true, y_pred, labels=[2, 1])
    assert_array_equal(cm, [[18, 2], [24, 3]])

    # a label not in y_true should result in zeros for that row/column
    extra_label = np.max(y_true) + 1
    cm = confusion_matrix(y_true, y_pred, labels=[2, extra_label])
    assert_array_equal(cm, [[18, 0], [0, 0]])


@pytest.mark.parametrize(
    "labels, err_msg",
    [
        ([], "'labels' should contain at least one label."),
        ([3, 4], "At least one label specified must be in y_true"),
    ],
    ids=["empty list", "unknown labels"],
)
def test_confusion_matrix_error(labels, err_msg):
    # [FUNCTION DOC]
    # Verifies ValueError for an empty labels list and for labels none of which appear
    #   in y_true.
    y_true, y_pred, _ = make_prediction(binary=False)
    with pytest.raises(ValueError, match=err_msg):
        confusion_matrix(y_true, y_pred, labels=labels)


def test_confusion_matrix_dtype():
    # [FUNCTION DOC]
    # Verifies confusion_matrix dtype: int64 for no or integer/bool weights, float64 for
    #   float/object weights; uint32 max weights accumulate without overflow while int64
    #   max weights wrap around (documented overflow).
    y = [0, 1, 1]
    weight = np.ones(len(y))
    # confusion_matrix returns int64 by default
    cm = confusion_matrix(y, y)
    assert cm.dtype == np.int64
    # The dtype of confusion_matrix is always 64 bit
    for dtype in [np.bool_, np.int32, np.uint64]:
        cm = confusion_matrix(y, y, sample_weight=weight.astype(dtype, copy=False))
        assert cm.dtype == np.int64
    for dtype in [np.float32, np.float64, None, object]:
        cm = confusion_matrix(y, y, sample_weight=weight.astype(dtype, copy=False))
        assert cm.dtype == np.float64

    # np.iinfo(np.uint32).max should be accumulated correctly
    weight = np.full(len(y), 4294967295, dtype=np.uint32)
    cm = confusion_matrix(y, y, sample_weight=weight)
    assert cm[0, 0] == 4294967295
    assert cm[1, 1] == 8589934590

    # np.iinfo(np.int64).max should cause an overflow
    weight = np.full(len(y), 9223372036854775807, dtype=np.int64)
    cm = confusion_matrix(y, y, sample_weight=weight)
    assert cm[0, 0] == 9223372036854775807
    assert cm[1, 1] == -2


@pytest.mark.parametrize("dtype", ["Int64", "Float64", "boolean"])
def test_confusion_matrix_pandas_nullable(dtype):
    # [FUNCTION DOC]
    # Non-regression for gh-25635: confusion_matrix with pandas nullable dtype y_true
    #   matches the ndarray result. Skipped without pandas.
    """Checks that confusion_matrix works with pandas nullable dtypes.

    Non-regression test for gh-25635.
    """
    pd = pytest.importorskip("pandas")

    y_ndarray = np.array([1, 0, 0, 1, 0, 1, 1, 0, 1])
    y_true = pd.Series(y_ndarray, dtype=dtype)
    y_predicted = pd.Series([0, 0, 1, 1, 0, 1, 1, 1, 1], dtype="int64")

    output = confusion_matrix(y_true, y_predicted)
    expected_output = confusion_matrix(y_ndarray, y_predicted)

    assert_array_equal(output, expected_output)


def test_classification_report_multiclass():
    # [FUNCTION DOC]
    # Verifies the text classification_report for iris predictions with target names
    #   matches an exact expected string.
    # Test performance report
    iris = datasets.load_iris()
    y_true, y_pred, _ = make_prediction(dataset=iris, binary=False)

    # print classification report with class names
    expected_report = """\
              precision    recall  f1-score   support

      setosa       0.83      0.79      0.81        24
  versicolor       0.33      0.10      0.15        31
   virginica       0.42      0.90      0.57        20

    accuracy                           0.53        75
   macro avg       0.53      0.60      0.51        75
weighted avg       0.51      0.53      0.47        75
"""
    report = classification_report(
        y_true,
        y_pred,
        labels=np.arange(len(iris.target_names)),
        target_names=iris.target_names,
    )
    assert report == expected_report


def test_classification_report_multiclass_balanced():
    # [FUNCTION DOC]
    # Verifies the text report for a balanced 3-class confusion where every score is
    #   0.33.
    y_true, y_pred = [0, 0, 0, 1, 1, 1, 2, 2, 2], [0, 1, 2, 0, 1, 2, 0, 1, 2]

    expected_report = """\
              precision    recall  f1-score   support

           0       0.33      0.33      0.33         3
           1       0.33      0.33      0.33         3
           2       0.33      0.33      0.33         3

    accuracy                           0.33         9
   macro avg       0.33      0.33      0.33         9
weighted avg       0.33      0.33      0.33         9
"""
    report = classification_report(y_true, y_pred)
    assert report == expected_report


def test_classification_report_multiclass_with_label_detection():
    # [FUNCTION DOC]
    # Verifies the text report with labels inferred from data (numeric row names).
    iris = datasets.load_iris()
    y_true, y_pred, _ = make_prediction(dataset=iris, binary=False)

    # print classification report with label detection
    expected_report = """\
              precision    recall  f1-score   support

           0       0.83      0.79      0.81        24
           1       0.33      0.10      0.15        31
           2       0.42      0.90      0.57        20

    accuracy                           0.53        75
   macro avg       0.53      0.60      0.51        75
weighted avg       0.51      0.53      0.47        75
"""
    report = classification_report(y_true, y_pred)
    assert report == expected_report


def test_classification_report_multiclass_with_digits():
    # [FUNCTION DOC]
    # Verifies the digits=5 option changes the number of decimals in the text report.
    # Test performance report with added digits in floating point values
    iris = datasets.load_iris()
    y_true, y_pred, _ = make_prediction(dataset=iris, binary=False)

    # print classification report with class names
    expected_report = """\
              precision    recall  f1-score   support

      setosa    0.82609   0.79167   0.80851        24
  versicolor    0.33333   0.09677   0.15000        31
   virginica    0.41860   0.90000   0.57143        20

    accuracy                        0.53333        75
   macro avg    0.52601   0.59615   0.50998        75
weighted avg    0.51375   0.53333   0.47310        75
"""
    report = classification_report(
        y_true,
        y_pred,
        labels=np.arange(len(iris.target_names)),
        target_names=iris.target_names,
        digits=5,
    )
    assert report == expected_report


def test_classification_report_multiclass_with_string_label():
    # [FUNCTION DOC]
    # Verifies the text report with string class labels, and with target_names
    #   overriding them.
    y_true, y_pred, _ = make_prediction(binary=False)

    y_true = np.array(["blue", "green", "red"])[y_true]
    y_pred = np.array(["blue", "green", "red"])[y_pred]

    expected_report = """\
              precision    recall  f1-score   support

        blue       0.83      0.79      0.81        24
       green       0.33      0.10      0.15        31
         red       0.42      0.90      0.57        20

    accuracy                           0.53        75
   macro avg       0.53      0.60      0.51        75
weighted avg       0.51      0.53      0.47        75
"""
    report = classification_report(y_true, y_pred)
    assert report == expected_report

    expected_report = """\
              precision    recall  f1-score   support

           a       0.83      0.79      0.81        24
           b       0.33      0.10      0.15        31
           c       0.42      0.90      0.57        20

    accuracy                           0.53        75
   macro avg       0.53      0.60      0.51        75
weighted avg       0.51      0.53      0.47        75
"""
    report = classification_report(y_true, y_pred, target_names=["a", "b", "c"])
    assert report == expected_report


def test_classification_report_multiclass_with_unicode_label():
    # [FUNCTION DOC]
    # Verifies the text report renders non-ASCII label names correctly.
    y_true, y_pred, _ = make_prediction(binary=False)

    labels = np.array(["blue\xa2", "green\xa2", "red\xa2"])
    y_true = labels[y_true]
    y_pred = labels[y_pred]

    expected_report = """\
              precision    recall  f1-score   support

       blue\xa2       0.83      0.79      0.81        24
      green\xa2       0.33      0.10      0.15        31
        red\xa2       0.42      0.90      0.57        20

    accuracy                           0.53        75
   macro avg       0.53      0.60      0.51        75
weighted avg       0.51      0.53      0.47        75
"""
    report = classification_report(y_true, y_pred)
    assert report == expected_report


def test_classification_report_multiclass_with_long_string_label():
    # [FUNCTION DOC]
    # Verifies the first column of the text report widens to fit a long label.
    y_true, y_pred, _ = make_prediction(binary=False)

    labels = np.array(["blue", "green" * 5, "red"])
    y_true = labels[y_true]
    y_pred = labels[y_pred]

    expected_report = """\
                           precision    recall  f1-score   support

                     blue       0.83      0.79      0.81        24
greengreengreengreengreen       0.33      0.10      0.15        31
                      red       0.42      0.90      0.57        20

                 accuracy                           0.53        75
                macro avg       0.53      0.60      0.51        75
             weighted avg       0.51      0.53      0.47        75
"""

    report = classification_report(y_true, y_pred)
    assert report == expected_report


def test_classification_report_labels_target_names_unequal_length():
    # [FUNCTION DOC]
    # Verifies a UserWarning when labels and target_names lengths differ.
    y_true = [0, 0, 2, 0, 0]
    y_pred = [0, 2, 2, 0, 0]
    target_names = ["class 0", "class 1", "class 2"]

    msg = "labels size, 2, does not match size of target_names, 3"
    with pytest.warns(UserWarning, match=msg):
        classification_report(y_true, y_pred, labels=[0, 2], target_names=target_names)


def test_classification_report_no_labels_target_names_unequal_length():
    # [FUNCTION DOC]
    # Verifies ValueError when inferred class count differs from target_names length and
    #   labels is not given.
    y_true = [0, 0, 2, 0, 0]
    y_pred = [0, 2, 2, 0, 0]
    target_names = ["class 0", "class 1", "class 2"]

    err_msg = (
        "Number of classes, 2, does not "
        "match size of target_names, 3. "
        "Try specifying the labels parameter"
    )
    with pytest.raises(ValueError, match=err_msg):
        classification_report(y_true, y_pred, target_names=target_names)


@pytest.mark.filterwarnings(r"ignore::sklearn.exceptions.UndefinedMetricWarning")
def test_multilabel_classification_report():
    # [FUNCTION DOC]
    # Verifies the text report for multilabel indicator data (two seeded
    #   make_multilabel_classification draws), including micro and samples averages.
    n_classes = 4
    n_samples = 50

    _, y_true = make_multilabel_classification(
        n_features=1, n_samples=n_samples, n_classes=n_classes, random_state=0
    )

    _, y_pred = make_multilabel_classification(
        n_features=1, n_samples=n_samples, n_classes=n_classes, random_state=1
    )

    expected_report = """\
              precision    recall  f1-score   support

           0       0.50      0.67      0.57        24
           1       0.51      0.74      0.61        27
           2       0.29      0.08      0.12        26
           3       0.52      0.56      0.54        27

   micro avg       0.50      0.51      0.50       104
   macro avg       0.45      0.51      0.46       104
weighted avg       0.45      0.51      0.46       104
 samples avg       0.46      0.42      0.40       104
"""

    report = classification_report(y_true, y_pred)
    assert report == expected_report


def test_multilabel_zero_one_loss_subset():
    # [FUNCTION DOC]
    # Verifies zero_one_loss on multilabel indicators is the fraction of rows not
    #   exactly matched.
    # Dense label indicator matrix format
    y1 = np.array([[0, 1, 1], [1, 0, 1]])
    y2 = np.array([[0, 0, 1], [1, 0, 1]])

    assert zero_one_loss(y1, y2) == 0.5
    assert zero_one_loss(y1, y1) == 0
    assert zero_one_loss(y2, y2) == 0
    assert zero_one_loss(y2, np.logical_not(y2)) == 1
    assert zero_one_loss(y1, np.logical_not(y1)) == 1
    assert zero_one_loss(y1, np.zeros(y1.shape)) == 1
    assert zero_one_loss(y2, np.zeros(y1.shape)) == 1


def test_multilabel_hamming_loss():
    # [FUNCTION DOC]
    # Verifies hamming_loss on multilabel indicators, with sample weights, and agreement
    #   with scipy's hamming on 1-D rows.
    # Dense label indicator matrix format
    y1 = np.array([[0, 1, 1], [1, 0, 1]])
    y2 = np.array([[0, 0, 1], [1, 0, 1]])
    w = np.array([1, 3])

    assert hamming_loss(y1, y2) == 1 / 6
    assert hamming_loss(y1, y1) == 0
    assert hamming_loss(y2, y2) == 0
    assert hamming_loss(y2, 1 - y2) == 1
    assert hamming_loss(y1, 1 - y1) == 1
    assert hamming_loss(y1, np.zeros(y1.shape)) == 4 / 6
    assert hamming_loss(y2, np.zeros(y1.shape)) == 0.5
    assert hamming_loss(y1, y2, sample_weight=w) == 1.0 / 12
    assert hamming_loss(y1, 1 - y2, sample_weight=w) == 11.0 / 12
    assert hamming_loss(y1, np.zeros_like(y1), sample_weight=w) == 2.0 / 3
    # sp_hamming only works with 1-D arrays
    assert hamming_loss(y1[0], y2[0]) == sp_hamming(y1[0], y2[0])


def test_jaccard_score_validation():
    # [FUNCTION DOC]
    # Verifies jaccard_score input validation errors: invalid pos_label,
    #   average="binary" with multilabel or multiclass targets, samples average outside
    #   multilabel, plus a UserWarning that pos_label is ignored with average="micro";
    #   UndefinedMetricWarning is turned into an error throughout.
    with warnings.catch_warnings():
        warnings.simplefilter("error", category=UndefinedMetricWarning)

        y_true = np.array([0, 1, 0, 1, 1])
        y_pred = np.array([0, 1, 0, 1, 1])
        err_msg = re.escape(
            "pos_label=2 is not a valid label. It should be one of [0 1]"
        )
        with pytest.raises(ValueError, match=err_msg):
            jaccard_score(y_true, y_pred, average="binary", pos_label=2)

        y_true = np.array([[0, 1, 1], [1, 0, 0]])
        y_pred = np.array([[1, 1, 1], [1, 0, 1]])
        msg1 = (
            r"Target is multilabel-indicator but average='binary'. "
            r"Please choose another average setting, one of \[None, "
            r"'micro', 'macro', 'weighted', 'samples'\]."
        )
        with pytest.raises(ValueError, match=msg1):
            jaccard_score(y_true, y_pred, average="binary", pos_label=-1)

        y_true = np.array([0, 1, 1, 0, 2])
        y_pred = np.array([1, 1, 1, 1, 0])
        msg2 = (
            r"Target is multiclass but average='binary'. Please choose "
            r"another average setting, one of \[None, 'micro', 'macro', "
            r"'weighted'\]."
        )
        with pytest.raises(ValueError, match=msg2):
            jaccard_score(y_true, y_pred, average="binary")
        msg3 = (
            "Samplewise metrics are not available outside of multilabel classification."
        )
        with pytest.raises(ValueError, match=msg3):
            jaccard_score(y_true, y_pred, average="samples")

        msg = (
            r"Note that pos_label \(set to 3\) is ignored when "
            r"average != 'binary' \(got 'micro'\). You may use "
            r"labels=\[pos_label\] to specify a single positive "
            "class."
        )
        with pytest.warns(UserWarning, match=msg):
            jaccard_score(y_true, y_pred, average="micro", pos_label=3)


def test_multilabel_jaccard_score():
    # [FUNCTION DOC]
    # Verifies multilabel jaccard_score for samples, macro, micro, weighted and per-
    #   label averages against hand-computed values, label range errors ("Got 4 > 2",
    #   "Got -1 < 0"), and the UndefinedMetricWarning for labels or samples with no true
    #   or predicted entries.
    with warnings.catch_warnings():
        warnings.simplefilter("error", category=UndefinedMetricWarning)
        # Dense label indicator matrix format
        y1 = np.array([[0, 1, 1], [1, 0, 1]])
        y2 = np.array([[0, 0, 1], [1, 0, 1]])

        # size(y1 \inter y2) = [1, 2]
        # size(y1 \union y2) = [2, 2]

        assert jaccard_score(y1, y2, average="samples") == 0.75
        assert jaccard_score(y1, y1, average="samples") == 1
        assert jaccard_score(y2, y2, average="samples") == 1
        assert jaccard_score(y2, np.logical_not(y2), average="samples") == 0
        assert jaccard_score(y1, np.logical_not(y1), average="samples") == 0
        assert jaccard_score(y1, np.zeros(y1.shape), average="samples") == 0
        assert jaccard_score(y2, np.zeros(y1.shape), average="samples") == 0

        y_true = np.array([[0, 1, 1], [1, 0, 0]])
        y_pred = np.array([[1, 1, 1], [1, 0, 1]])
        # average='macro'
        assert_almost_equal(jaccard_score(y_true, y_pred, average="macro"), 2.0 / 3)
        # average='micro'
        assert_almost_equal(jaccard_score(y_true, y_pred, average="micro"), 3.0 / 5)
        # average='samples'
        assert_almost_equal(jaccard_score(y_true, y_pred, average="samples"), 7.0 / 12)
        assert_almost_equal(
            jaccard_score(y_true, y_pred, average="samples", labels=[0, 2]), 1.0 / 2
        )
        assert_almost_equal(
            jaccard_score(y_true, y_pred, average="samples", labels=[1, 2]), 1.0 / 2
        )
        # average=None
        assert_array_equal(
            jaccard_score(y_true, y_pred, average=None),
            np.array([1.0 / 2, 1.0, 1.0 / 2]),
        )

        y_true = np.array([[0, 1, 1], [1, 0, 1]])
        y_pred = np.array([[1, 1, 1], [1, 0, 1]])
        assert_almost_equal(jaccard_score(y_true, y_pred, average="macro"), 5.0 / 6)
        # average='weighted'
        assert_almost_equal(jaccard_score(y_true, y_pred, average="weighted"), 7.0 / 8)

        msg2 = "Got 4 > 2"
        with pytest.raises(ValueError, match=msg2):
            jaccard_score(y_true, y_pred, labels=[4], average="macro")
        msg3 = "Got -1 < 0"
        with pytest.raises(ValueError, match=msg3):
            jaccard_score(y_true, y_pred, labels=[-1], average="macro")

        msg = (
            "Jaccard is ill-defined and being set to 0.0 in labels "
            "with no true or predicted samples."
        )

    with pytest.warns(UndefinedMetricWarning, match=msg):
        assert (
            jaccard_score(np.array([[0, 1]]), np.array([[0, 1]]), average="macro")
            == 0.5
        )

    msg = (
        "Jaccard is ill-defined and being set to 0.0 in samples "
        "with no true or predicted labels."
    )

    with pytest.warns(UndefinedMetricWarning, match=msg):
        assert (
            jaccard_score(
                np.array([[0, 0], [1, 1]]),
                np.array([[0, 0], [1, 1]]),
                average="samples",
            )
            == 0.5
        )


def test_multiclass_jaccard_score():
    # [FUNCTION DOC]
    # Verifies multiclass jaccard_score on string labels equals jaccard_score on the
    #   LabelBinarizer-encoded indicators for every average and label subset, without
    #   UndefinedMetricWarning.
    # KNOWN ISSUE: the final weighted-average comparison with 0 on all-zero input is not
    #   asserted, so it only checks that the call does not raise.
    with warnings.catch_warnings():
        warnings.simplefilter("error", category=UndefinedMetricWarning)

        y_true = ["ant", "ant", "cat", "cat", "ant", "cat", "bird", "bird"]
        y_pred = ["cat", "ant", "cat", "cat", "ant", "bird", "bird", "cat"]
        labels = ["ant", "bird", "cat"]
        lb = LabelBinarizer()
        lb.fit(labels)
        y_true_bin = lb.transform(y_true)
        y_pred_bin = lb.transform(y_pred)
        multi_jaccard_score = partial(jaccard_score, y_true, y_pred)
        bin_jaccard_score = partial(jaccard_score, y_true_bin, y_pred_bin)
        multi_labels_list = [
            ["ant", "bird"],
            ["ant", "cat"],
            ["cat", "bird"],
            ["ant"],
            ["bird"],
            ["cat"],
            None,
        ]
        bin_labels_list = [[0, 1], [0, 2], [2, 1], [0], [1], [2], None]

        with warnings.catch_warnings():
            warnings.simplefilter("error", category=UndefinedMetricWarning)
            # other than average='samples'/'none-samples', test everything else here
            for average in ("macro", "weighted", "micro", None):
                for m_label, b_label in zip(multi_labels_list, bin_labels_list):
                    assert_almost_equal(
                        multi_jaccard_score(average=average, labels=m_label),
                        bin_jaccard_score(average=average, labels=b_label),
                    )

            y_true = np.array([[0, 0], [0, 0], [0, 0]])
            y_pred = np.array([[0, 0], [0, 0], [0, 0]])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UndefinedMetricWarning)
        jaccard_score(y_true, y_pred, average="weighted") == 0


def test_average_binary_jaccard_score():
    # [FUNCTION DOC]
    # Verifies binary jaccard_score for no true positives, the all-negative case (0 with
    #   UndefinedMetricWarning), pos_label=0, and a general example for both pos_label
    #   values.
    with warnings.catch_warnings():
        warnings.simplefilter("error", category=UndefinedMetricWarning)

        # tp=0, fp=0, fn=1, tn=0
        assert jaccard_score([1], [0], average="binary") == 0.0
        # tp=0, fp=0, fn=0, tn=1
        msg = (
            "Jaccard is ill-defined and being set to 0.0 due to "
            "no true or predicted samples"
        )
        with pytest.warns(UndefinedMetricWarning, match=msg):
            assert jaccard_score([0, 0], [0, 0], average="binary") == 0.0

        # tp=1, fp=0, fn=0, tn=0 (pos_label=0)
        assert jaccard_score([0], [0], pos_label=0, average="binary") == 1.0
        y_true = np.array([1, 0, 1, 1, 0])
        y_pred = np.array([1, 0, 1, 1, 1])
        assert_almost_equal(jaccard_score(y_true, y_pred, average="binary"), 3.0 / 4)
        assert_almost_equal(
            jaccard_score(y_true, y_pred, average="binary", pos_label=0), 1.0 / 2
        )


def test_jaccard_score_zero_division_warning():
    # [FUNCTION DOC]
    # Verifies samples-averaged jaccard_score with an all-zero sample emits the
    #   zero_division UndefinedMetricWarning and returns 0.
    # check that we raised a warning with default behavior if a zero division
    # happens
    y_true = np.array([[1, 0, 1], [0, 0, 0]])
    y_pred = np.array([[0, 0, 0], [0, 0, 0]])
    msg = (
        "Jaccard is ill-defined and being set to 0.0 in "
        "samples with no true or predicted labels."
        " Use `zero_division` parameter to control this behavior."
    )
    with pytest.warns(UndefinedMetricWarning, match=msg):
        score = jaccard_score(y_true, y_pred, average="samples", zero_division="warn")
        assert score == pytest.approx(0.0)


@pytest.mark.parametrize("zero_division, expected_score", [(0, 0), (1, 0.5)])
def test_jaccard_score_zero_division_set_value(zero_division, expected_score):
    # [FUNCTION DOC]
    # Verifies an explicit zero_division (0 or 1) suppresses the warning and sets the
    #   empty sample's score accordingly.
    # check that we don't issue warning by passing the zero_division parameter
    y_true = np.array([[1, 0, 1], [0, 0, 0]])
    y_pred = np.array([[0, 0, 0], [0, 0, 0]])
    with warnings.catch_warnings():
        warnings.simplefilter("error", UndefinedMetricWarning)
        score = jaccard_score(
            y_true, y_pred, average="samples", zero_division=zero_division
        )
    assert score == pytest.approx(expected_score)


@pytest.mark.filterwarnings(r"ignore::sklearn.exceptions.UndefinedMetricWarning")
def test_precision_recall_f1_score_multilabel_1():
    # [FUNCTION DOC]
    # Verifies PRF and F2 on a crafted multilabel example (per-class, macro, micro,
    #   weighted and samples averages) against hand-derived counts.
    # Test precision_recall_f1_score on a crafted multilabel example
    # First crafted example

    y_true = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1]])
    y_pred = np.array([[0, 1, 0, 0], [0, 1, 0, 0], [1, 0, 1, 0]])

    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, average=None)

    # tp = [0, 1, 1, 0]
    # fn = [1, 0, 0, 1]
    # fp = [1, 1, 0, 0]
    # Check per class

    assert_array_almost_equal(p, [0.0, 0.5, 1.0, 0.0], 2)
    assert_array_almost_equal(r, [0.0, 1.0, 1.0, 0.0], 2)
    assert_array_almost_equal(f, [0.0, 1 / 1.5, 1, 0.0], 2)
    assert_array_almost_equal(s, [1, 1, 1, 1], 2)

    f2 = fbeta_score(y_true, y_pred, beta=2, average=None)
    support = s
    assert_array_almost_equal(f2, [0, 0.83, 1, 0], 2)

    # Check macro
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, average="macro")
    assert_almost_equal(p, 1.5 / 4)
    assert_almost_equal(r, 0.5)
    assert_almost_equal(f, 2.5 / 1.5 * 0.25)
    assert s is None
    assert_almost_equal(
        fbeta_score(y_true, y_pred, beta=2, average="macro"), np.mean(f2)
    )

    # Check micro
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, average="micro")
    assert_almost_equal(p, 0.5)
    assert_almost_equal(r, 0.5)
    assert_almost_equal(f, 0.5)
    assert s is None
    assert_almost_equal(
        fbeta_score(y_true, y_pred, beta=2, average="micro"),
        (1 + 4) * p * r / (4 * p + r),
    )

    # Check weighted
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, average="weighted")
    assert_almost_equal(p, 1.5 / 4)
    assert_almost_equal(r, 0.5)
    assert_almost_equal(f, 2.5 / 1.5 * 0.25)
    assert s is None
    assert_almost_equal(
        fbeta_score(y_true, y_pred, beta=2, average="weighted"),
        np.average(f2, weights=support),
    )
    # Check samples
    # |h(x_i) inter y_i | = [0, 1, 1]
    # |y_i| = [1, 1, 2]
    # |h(x_i)| = [1, 1, 2]
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, average="samples")
    assert_almost_equal(p, 0.5)
    assert_almost_equal(r, 0.5)
    assert_almost_equal(f, 0.5)
    assert s is None
    assert_almost_equal(fbeta_score(y_true, y_pred, beta=2, average="samples"), 0.5)


@pytest.mark.filterwarnings(r"ignore::sklearn.exceptions.UndefinedMetricWarning")
def test_precision_recall_f1_score_multilabel_2():
    # [FUNCTION DOC]
    # Verifies PRF and F2 on a second crafted multilabel example with a label having
    #   zero support, for all averages.
    # Test precision_recall_f1_score on a crafted multilabel example 2
    # Second crafted example
    y_true = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 1, 1, 0]])
    y_pred = np.array([[0, 0, 0, 1], [0, 0, 0, 1], [1, 1, 0, 0]])

    # tp = [ 0.  1.  0.  0.]
    # fp = [ 1.  0.  0.  2.]
    # fn = [ 1.  1.  1.  0.]

    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, average=None)
    assert_array_almost_equal(p, [0.0, 1.0, 0.0, 0.0], 2)
    assert_array_almost_equal(r, [0.0, 0.5, 0.0, 0.0], 2)
    assert_array_almost_equal(f, [0.0, 0.66, 0.0, 0.0], 2)
    assert_array_almost_equal(s, [1, 2, 1, 0], 2)

    f2 = fbeta_score(y_true, y_pred, beta=2, average=None)
    support = s
    assert_array_almost_equal(f2, [0, 0.55, 0, 0], 2)

    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, average="micro")
    assert_almost_equal(p, 0.25)
    assert_almost_equal(r, 0.25)
    assert_almost_equal(f, 2 * 0.25 * 0.25 / 0.5)
    assert s is None
    assert_almost_equal(
        fbeta_score(y_true, y_pred, beta=2, average="micro"),
        (1 + 4) * p * r / (4 * p + r),
    )

    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, average="macro")
    assert_almost_equal(p, 0.25)
    assert_almost_equal(r, 0.125)
    assert_almost_equal(f, 2 / 12)
    assert s is None
    assert_almost_equal(
        fbeta_score(y_true, y_pred, beta=2, average="macro"), np.mean(f2)
    )

    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, average="weighted")
    assert_almost_equal(p, 2 / 4)
    assert_almost_equal(r, 1 / 4)
    assert_almost_equal(f, 2 / 3 * 2 / 4)
    assert s is None
    assert_almost_equal(
        fbeta_score(y_true, y_pred, beta=2, average="weighted"),
        np.average(f2, weights=support),
    )

    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, average="samples")
    # Check samples
    # |h(x_i) inter y_i | = [0, 0, 1]
    # |y_i| = [1, 1, 2]
    # |h(x_i)| = [1, 1, 2]

    assert_almost_equal(p, 1 / 6)
    assert_almost_equal(r, 1 / 6)
    assert_almost_equal(f, 2 / 4 * 1 / 3)
    assert s is None
    assert_almost_equal(
        fbeta_score(y_true, y_pred, beta=2, average="samples"), 0.1666, 2
    )


@pytest.mark.filterwarnings(r"ignore::sklearn.exceptions.UndefinedMetricWarning")
@pytest.mark.parametrize(
    "zero_division, zero_division_expected",
    [("warn", 0), (0, 0), (1, 1), (np.nan, np.nan)],
)
def test_precision_recall_f1_score_with_an_empty_prediction(
    zero_division, zero_division_expected
):
    # [FUNCTION DOC]
    # Verifies PRF and F2 when one sample has no predicted labels and one label has no
    #   support, for each zero_division setting; macro averaging excludes NaN entries
    #   (via _nanaverage) and the zero_division value is substituted for undefined
    #   precision/recall.
    y_true = np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 1, 1, 0]])
    y_pred = np.array([[0, 0, 0, 0], [0, 0, 0, 1], [0, 1, 1, 0]])

    # true_pos = [ 0.  1.  1.  0.]
    # false_pos = [ 0.  0.  0.  1.]
    # false_neg = [ 1.  1.  0.  0.]

    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=zero_division
    )

    assert_array_almost_equal(p, [zero_division_expected, 1.0, 1.0, 0.0], 2)
    assert_array_almost_equal(r, [0.0, 0.5, 1.0, zero_division_expected], 2)
    expected_f = 0
    assert_array_almost_equal(f, [expected_f, 1 / 1.5, 1, expected_f], 2)
    assert_array_almost_equal(s, [1, 2, 1, 0], 2)

    f2 = fbeta_score(y_true, y_pred, beta=2, average=None, zero_division=zero_division)
    support = s
    assert_array_almost_equal(f2, [expected_f, 0.55, 1, expected_f], 2)

    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=zero_division
    )

    value_to_sum = 0 if np.isnan(zero_division_expected) else zero_division_expected
    values_to_average = 3 + (not np.isnan(zero_division_expected))

    assert_almost_equal(p, (2 + value_to_sum) / values_to_average)
    assert_almost_equal(r, (1.5 + value_to_sum) / values_to_average)
    expected_f = (2 / 3 + 1) / 4
    assert_almost_equal(f, expected_f)
    assert s is None
    assert_almost_equal(
        fbeta_score(
            y_true,
            y_pred,
            beta=2,
            average="macro",
            zero_division=zero_division,
        ),
        _nanaverage(f2, weights=None),
    )

    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, average="micro", zero_division=zero_division
    )
    assert_almost_equal(p, 2 / 3)
    assert_almost_equal(r, 0.5)
    assert_almost_equal(f, 2 / 3 / (2 / 3 + 0.5))
    assert s is None
    assert_almost_equal(
        fbeta_score(
            y_true, y_pred, beta=2, average="micro", zero_division=zero_division
        ),
        (1 + 4) * p * r / (4 * p + r),
    )

    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=zero_division
    )
    assert_almost_equal(p, 3 / 4 if zero_division_expected == 0 else 1.0)
    assert_almost_equal(r, 0.5)
    values_to_average = 4
    assert_almost_equal(f, (2 * 2 / 3 + 1) / values_to_average)
    assert s is None
    assert_almost_equal(
        fbeta_score(
            y_true, y_pred, beta=2, average="weighted", zero_division=zero_division
        ),
        _nanaverage(f2, weights=support),
    )

    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, average="samples")
    # |h(x_i) inter y_i | = [0, 0, 2]
    # |y_i| = [1, 1, 2]
    # |h(x_i)| = [0, 1, 2]
    assert_almost_equal(p, 1 / 3)
    assert_almost_equal(r, 1 / 3)
    assert_almost_equal(f, 1 / 3)
    assert s is None
    expected_result = 0.333
    assert_almost_equal(
        fbeta_score(
            y_true, y_pred, beta=2, average="samples", zero_division=zero_division
        ),
        expected_result,
        2,
    )


@pytest.mark.parametrize("beta", [1])
@pytest.mark.parametrize("average", ["macro", "micro", "weighted", "samples"])
@pytest.mark.parametrize("zero_division", [0, 1, np.nan])
def test_precision_recall_f1_no_labels(beta, average, zero_division):
    # [FUNCTION DOC]
    # Verifies that with all-zero multilabel truth and predictions, PRF and F-beta
    #   return the zero_division value (or NaN) for every average, without
    #   UndefinedMetricWarning, and support is None.
    y_true = np.zeros((20, 3))
    y_pred = np.zeros_like(y_true)

    with warnings.catch_warnings():
        warnings.simplefilter("error", category=UndefinedMetricWarning)

        p, r, f, s = precision_recall_fscore_support(
            y_true,
            y_pred,
            average=average,
            beta=beta,
            zero_division=zero_division,
        )
        fbeta = fbeta_score(
            y_true,
            y_pred,
            beta=beta,
            average=average,
            zero_division=zero_division,
        )
    assert s is None

    # if zero_division = nan, check that all metrics are nan and exit
    if np.isnan(zero_division):
        for metric in [p, r, f, fbeta]:
            assert np.isnan(metric)
        return

    zero_division = float(zero_division)
    assert_almost_equal(p, zero_division)
    assert_almost_equal(r, zero_division)
    assert_almost_equal(f, zero_division)

    assert_almost_equal(fbeta, float(zero_division))


@pytest.mark.parametrize("average", ["macro", "micro", "weighted", "samples"])
def test_precision_recall_f1_no_labels_check_warnings(average):
    # [FUNCTION DOC]
    # Verifies that with all-zero multilabel data and default zero_division, PRF and
    #   F-beta warn with UndefinedMetricWarning and return 0 for each average.
    y_true = np.zeros((20, 3))
    y_pred = np.zeros_like(y_true)

    func = precision_recall_fscore_support
    with pytest.warns(UndefinedMetricWarning):
        p, r, f, s = func(y_true, y_pred, average=average, beta=1.0)

    assert_almost_equal(p, 0)
    assert_almost_equal(r, 0)
    assert_almost_equal(f, 0)
    assert s is None

    with pytest.warns(UndefinedMetricWarning):
        fbeta = fbeta_score(y_true, y_pred, average=average, beta=1.0)

    assert_almost_equal(fbeta, 0)


@pytest.mark.parametrize("zero_division", [0, 1, np.nan])
def test_precision_recall_f1_no_labels_average_none(zero_division):
    # [FUNCTION DOC]
    # Verifies average=None on all-zero multilabel data returns per-label zero_division
    #   values (0, 1 or NaN), zero supports, and no warnings.
    y_true = np.zeros((20, 3))
    y_pred = np.zeros_like(y_true)

    # tp = [0, 0, 0]
    # fn = [0, 0, 0]
    # fp = [0, 0, 0]
    # support = [0, 0, 0]
    # |y_hat_i inter y_i | = [0, 0, 0]
    # |y_i| = [0, 0, 0]
    # |y_hat_i| = [0, 0, 0]

    with warnings.catch_warnings():
        warnings.simplefilter("error", category=UndefinedMetricWarning)

        p, r, f, s = precision_recall_fscore_support(
            y_true,
            y_pred,
            average=None,
            beta=1.0,
            zero_division=zero_division,
        )
        fbeta = fbeta_score(
            y_true, y_pred, beta=1.0, average=None, zero_division=zero_division
        )

    zero_division = np.float64(zero_division)
    assert_array_almost_equal(p, [zero_division, zero_division, zero_division], 2)
    assert_array_almost_equal(r, [zero_division, zero_division, zero_division], 2)
    assert_array_almost_equal(f, [zero_division, zero_division, zero_division], 2)
    assert_array_almost_equal(s, [0, 0, 0], 2)

    assert_array_almost_equal(fbeta, [zero_division, zero_division, zero_division], 2)


def test_precision_recall_f1_no_labels_average_none_warn():
    # [FUNCTION DOC]
    # Verifies average=None on all-zero multilabel data with default zero_division warns
    #   and returns per-label zeros.
    y_true = np.zeros((20, 3))
    y_pred = np.zeros_like(y_true)

    # tp = [0, 0, 0]
    # fn = [0, 0, 0]
    # fp = [0, 0, 0]
    # support = [0, 0, 0]
    # |y_hat_i inter y_i | = [0, 0, 0]
    # |y_i| = [0, 0, 0]
    # |y_hat_i| = [0, 0, 0]

    with pytest.warns(UndefinedMetricWarning):
        p, r, f, s = precision_recall_fscore_support(
            y_true, y_pred, average=None, beta=1
        )

    assert_array_almost_equal(p, [0, 0, 0], 2)
    assert_array_almost_equal(r, [0, 0, 0], 2)
    assert_array_almost_equal(f, [0, 0, 0], 2)
    assert_array_almost_equal(s, [0, 0, 0], 2)

    with pytest.warns(UndefinedMetricWarning):
        fbeta = fbeta_score(y_true, y_pred, beta=1, average=None)

    assert_array_almost_equal(fbeta, [0, 0, 0], 2)


def test_prf_warnings():
    # [FUNCTION DOC]
    # Verifies the exact UndefinedMetricWarning messages of
    #   precision_recall_fscore_support for ill-defined precision or recall in per-
    #   label, per-sample, micro and binary settings, and the three warnings (precision,
    #   recall, F-score, popped in reverse order) for an all-negative binary input.
    # average of per-label scores
    f, w = precision_recall_fscore_support, UndefinedMetricWarning
    for average in [None, "weighted", "macro"]:
        msg = (
            "Precision is ill-defined and "
            "being set to 0.0 in labels with no predicted samples."
            " Use `zero_division` parameter to control"
            " this behavior."
        )
        with pytest.warns(w, match=msg):
            f([0, 1, 2], [1, 1, 2], average=average)

        msg = (
            "Recall is ill-defined and "
            "being set to 0.0 in labels with no true samples."
            " Use `zero_division` parameter to control"
            " this behavior."
        )
        with pytest.warns(w, match=msg):
            f([1, 1, 2], [0, 1, 2], average=average)

    # average of per-sample scores
    msg = (
        "Precision is ill-defined and "
        "being set to 0.0 in samples with no predicted labels."
        " Use `zero_division` parameter to control"
        " this behavior."
    )
    with pytest.warns(w, match=msg):
        f(np.array([[1, 0], [1, 0]]), np.array([[1, 0], [0, 0]]), average="samples")

    msg = (
        "Recall is ill-defined and "
        "being set to 0.0 in samples with no true labels."
        " Use `zero_division` parameter to control"
        " this behavior."
    )
    with pytest.warns(w, match=msg):
        f(np.array([[1, 0], [0, 0]]), np.array([[1, 0], [1, 0]]), average="samples")

    # single score: micro-average
    msg = (
        "Precision is ill-defined and "
        "being set to 0.0 due to no predicted samples."
        " Use `zero_division` parameter to control"
        " this behavior."
    )
    with pytest.warns(w, match=msg):
        f(np.array([[1, 1], [1, 1]]), np.array([[0, 0], [0, 0]]), average="micro")

    msg = (
        "Recall is ill-defined and "
        "being set to 0.0 due to no true samples."
        " Use `zero_division` parameter to control"
        " this behavior."
    )
    with pytest.warns(w, match=msg):
        f(np.array([[0, 0], [0, 0]]), np.array([[1, 1], [1, 1]]), average="micro")

    # single positive label
    msg = (
        "Precision is ill-defined and "
        "being set to 0.0 due to no predicted samples."
        " Use `zero_division` parameter to control"
        " this behavior."
    )
    with pytest.warns(w, match=msg):
        f([1, 1], [-1, -1], average="binary")

    msg = (
        "Recall is ill-defined and "
        "being set to 0.0 due to no true samples."
        " Use `zero_division` parameter to control"
        " this behavior."
    )
    with pytest.warns(w, match=msg):
        f([-1, -1], [1, 1], average="binary")

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        precision_recall_fscore_support([0, 0], [0, 0], average="binary")
        msg = (
            "F-score is ill-defined and being set to 0.0 due to no true nor "
            "predicted samples. Use `zero_division` parameter to control this"
            " behavior."
        )
        assert str(record.pop().message) == msg
        msg = (
            "Recall is ill-defined and "
            "being set to 0.0 due to no true samples."
            " Use `zero_division` parameter to control"
            " this behavior."
        )
        assert str(record.pop().message) == msg
        msg = (
            "Precision is ill-defined and "
            "being set to 0.0 due to no predicted samples."
            " Use `zero_division` parameter to control"
            " this behavior."
        )
        assert str(record.pop().message) == msg


@pytest.mark.parametrize("zero_division", [0, 1, np.nan])
def test_prf_no_warnings_if_zero_division_set(zero_division):
    # [FUNCTION DOC]
    # Verifies that setting zero_division to 0, 1 or NaN suppresses all
    #   UndefinedMetricWarnings in the same scenarios as test_prf_warnings.
    with warnings.catch_warnings():
        warnings.simplefilter("error", category=UndefinedMetricWarning)

        # average of per-label scores
        for average in [None, "weighted", "macro"]:
            precision_recall_fscore_support(
                [0, 1, 2], [1, 1, 2], average=average, zero_division=zero_division
            )

            precision_recall_fscore_support(
                [1, 1, 2], [0, 1, 2], average=average, zero_division=zero_division
            )

        # average of per-sample scores
        precision_recall_fscore_support(
            np.array([[1, 0], [1, 0]]),
            np.array([[1, 0], [0, 0]]),
            average="samples",
            zero_division=zero_division,
        )

        precision_recall_fscore_support(
            np.array([[1, 0], [0, 0]]),
            np.array([[1, 0], [1, 0]]),
            average="samples",
            zero_division=zero_division,
        )

        # single score: micro-average
        precision_recall_fscore_support(
            np.array([[1, 1], [1, 1]]),
            np.array([[0, 0], [0, 0]]),
            average="micro",
            zero_division=zero_division,
        )

        precision_recall_fscore_support(
            np.array([[0, 0], [0, 0]]),
            np.array([[1, 1], [1, 1]]),
            average="micro",
            zero_division=zero_division,
        )

        # single positive label
        precision_recall_fscore_support(
            [1, 1], [-1, -1], average="binary", zero_division=zero_division
        )

        precision_recall_fscore_support(
            [-1, -1], [1, 1], average="binary", zero_division=zero_division
        )

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always", category=UndefinedMetricWarning)
        precision_recall_fscore_support(
            [0, 0], [0, 0], average="binary", zero_division=zero_division
        )
        assert len(record) == 0


@pytest.mark.parametrize("zero_division", ["warn", 0, 1, np.nan])
def test_recall_warnings(zero_division):
    # [FUNCTION DOC]
    # Verifies recall_score does not warn when predictions are empty but truth is not,
    #   and warns only for zero_division="warn" when there are no true samples; also
    #   checks the default call on all-negative binary input warns.
    with warnings.catch_warnings():
        warnings.simplefilter("error", category=UndefinedMetricWarning)

        recall_score(
            np.array([[1, 1], [1, 1]]),
            np.array([[0, 0], [0, 0]]),
            average="micro",
            zero_division=zero_division,
        )

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always", category=UndefinedMetricWarning)
        recall_score(
            np.array([[0, 0], [0, 0]]),
            np.array([[1, 1], [1, 1]]),
            average="micro",
            zero_division=zero_division,
        )
        if zero_division == "warn":
            assert (
                str(record.pop().message) == "Recall is ill-defined and "
                "being set to 0.0 due to no true samples."
                " Use `zero_division` parameter to control"
                " this behavior."
            )
        else:
            assert len(record) == 0

        recall_score([0, 0], [0, 0])
        if zero_division == "warn":
            assert (
                str(record.pop().message) == "Recall is ill-defined and "
                "being set to 0.0 due to no true samples."
                " Use `zero_division` parameter to control"
                " this behavior."
            )


@pytest.mark.parametrize("zero_division", ["warn", 0, 1, np.nan])
def test_precision_warnings(zero_division):
    # [FUNCTION DOC]
    # Verifies precision_score warns only for zero_division="warn" when there are no
    #   predicted samples (including the default binary all-negative case), and never
    #   warns when predictions exist.
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always", category=UndefinedMetricWarning)
        precision_score(
            np.array([[1, 1], [1, 1]]),
            np.array([[0, 0], [0, 0]]),
            average="micro",
            zero_division=zero_division,
        )
        if zero_division == "warn":
            assert (
                str(record.pop().message) == "Precision is ill-defined and "
                "being set to 0.0 due to no predicted samples."
                " Use `zero_division` parameter to control"
                " this behavior."
            )
        else:
            assert len(record) == 0

        precision_score([0, 0], [0, 0])
        if zero_division == "warn":
            assert (
                str(record.pop().message) == "Precision is ill-defined and "
                "being set to 0.0 due to no predicted samples."
                " Use `zero_division` parameter to control"
                " this behavior."
            )

    with warnings.catch_warnings():
        warnings.simplefilter("error", category=UndefinedMetricWarning)

        precision_score(
            np.array([[0, 0], [0, 0]]),
            np.array([[1, 1], [1, 1]]),
            average="micro",
            zero_division=zero_division,
        )


@pytest.mark.parametrize("zero_division", ["warn", 0, 1, np.nan])
def test_fscore_warnings(zero_division):
    # [FUNCTION DOC]
    # Verifies F1 and F2 do not warn when only one of precision or recall is undefined
    #   and warn (only with zero_division="warn") when there are neither true nor
    #   predicted samples.
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always", category=UndefinedMetricWarning)

        for score in [f1_score, partial(fbeta_score, beta=2)]:
            score(
                np.array([[1, 1], [1, 1]]),
                np.array([[0, 0], [0, 0]]),
                average="micro",
                zero_division=zero_division,
            )
            assert len(record) == 0

            score(
                np.array([[0, 0], [0, 0]]),
                np.array([[1, 1], [1, 1]]),
                average="micro",
                zero_division=zero_division,
            )
            assert len(record) == 0

            score(
                np.array([[0, 0], [0, 0]]),
                np.array([[0, 0], [0, 0]]),
                average="micro",
                zero_division=zero_division,
            )
            if zero_division == "warn":
                assert (
                    str(record.pop().message) == "F-score is ill-defined and "
                    "being set to 0.0 due to no true nor predicted "
                    "samples. Use `zero_division` parameter to "
                    "control this behavior."
                )
            else:
                assert len(record) == 0


def test_prf_average_binary_data_non_binary():
    # [FUNCTION DOC]
    # Verifies the default average="binary" raises ValueError with a hint for multiclass
    #   and multilabel-indicator targets for precision, recall, F1 and F2.
    # Error if user does not explicitly set non-binary average mode
    y_true_mc = [1, 2, 3, 3]
    y_pred_mc = [1, 2, 3, 1]
    msg_mc = (
        r"Target is multiclass but average='binary'. Please "
        r"choose another average setting, one of \["
        r"None, 'micro', 'macro', 'weighted'\]."
    )
    y_true_ind = np.array([[0, 1, 1], [1, 0, 0], [0, 0, 1]])
    y_pred_ind = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
    msg_ind = (
        r"Target is multilabel-indicator but average='binary'. Please "
        r"choose another average setting, one of \["
        r"None, 'micro', 'macro', 'weighted', 'samples'\]."
    )

    for y_true, y_pred, msg in [
        (y_true_mc, y_pred_mc, msg_mc),
        (y_true_ind, y_pred_ind, msg_ind),
    ]:
        for metric in [
            precision_score,
            recall_score,
            f1_score,
            partial(fbeta_score, beta=2),
        ]:
            with pytest.raises(ValueError, match=msg):
                metric(y_true, y_pred)


def test__check_targets():
    # [FUNCTION DOC]
    # Verifies _check_targets over every ordered pair of example target types: allowed
    #   combinations return the merged type (binary plus multiclass gives multiclass),
    #   multilabel outputs are CSR and others are squeezed copies of the input,
    #   mismatched lengths raise ValueError; disallowed combinations raise ValueError
    #   with a "mix of" or "is not supported" message; and the legacy sequence-of-
    #   sequences format is rejected.
    # Check that _check_targets correctly merges target types, squeezes
    # output and fails if input lengths differ.
    IND = "multilabel-indicator"
    MC = "multiclass"
    BIN = "binary"
    CNT = "continuous"
    MMC = "multiclass-multioutput"
    MCN = "continuous-multioutput"
    # all of length 3
    EXAMPLES = [
        # must not be considered binary
        (IND, np.array([[0, 1], [1, 0], [1, 1]])),
        (MC, [2, 3, 1]),
        (BIN, [0, 1, 1]),
        (CNT, [0.0, 1.5, 1.0]),
        (MC, np.array([[2], [3], [1]])),
        (BIN, np.array([[0], [1], [1]])),
        (CNT, np.array([[0.0], [1.5], [1.0]])),
        (MMC, np.array([[0, 2], [1, 3], [2, 3]])),
        (MCN, np.array([[0.5, 2.0], [1.1, 3.0], [2.0, 3.0]])),
    ]
    # expected type given input types, or None for error
    # (types will be tried in either order)
    EXPECTED = {
        (IND, IND): IND,
        (MC, MC): MC,
        (BIN, BIN): BIN,
        (MC, IND): None,
        (BIN, IND): None,
        (BIN, MC): MC,
        # Disallowed types
        (CNT, CNT): None,
        (MMC, MMC): None,
        (MCN, MCN): None,
        (IND, CNT): None,
        (MC, CNT): None,
        (BIN, CNT): None,
        (MMC, CNT): None,
        (MCN, CNT): None,
        (IND, MMC): None,
        (MC, MMC): None,
        (BIN, MMC): None,
        (MCN, MMC): None,
        (IND, MCN): None,
        (MC, MCN): None,
        (BIN, MCN): None,
    }

    for (type1, y1), (type2, y2) in product(EXAMPLES, repeat=2):
        try:
            expected = EXPECTED[type1, type2]
        except KeyError:
            expected = EXPECTED[type2, type1]
        if expected is None:
            with pytest.raises(ValueError):
                _check_targets(y1, y2)

            if type1 != type2:
                err_msg = (
                    "Classification metrics can't handle a mix "
                    "of {0} and {1} targets".format(type1, type2)
                )
                with pytest.raises(ValueError, match=err_msg):
                    _check_targets(y1, y2)

            else:
                if type1 not in (BIN, MC, IND):
                    err_msg = "{0} is not supported".format(type1)
                    with pytest.raises(ValueError, match=err_msg):
                        _check_targets(y1, y2)

        else:
            merged_type, _, y1out, y2out, _ = _check_targets(y1, y2)
            assert merged_type == expected
            if merged_type.startswith("multilabel"):
                assert y1out.format == "csr"
                assert y2out.format == "csr"
            else:
                assert_array_equal(y1out, np.squeeze(y1))
                assert_array_equal(y2out, np.squeeze(y2))
            with pytest.raises(ValueError):
                _check_targets(y1[:-1], y2)

    # Make sure seq of seq is not supported
    y1 = [(1, 2), (0, 2, 3)]
    y2 = [(2,), (0, 2)]
    msg = (
        "You appear to be using a legacy multi-label data representation. "
        "Sequence of sequences are no longer supported; use a binary array"
        " or sparse matrix instead - the MultiLabelBinarizer"
        " transformer can convert to this format."
    )
    with pytest.raises(ValueError, match=msg):
        _check_targets(y1, y2)


def test__check_targets_raises_on_empty_inputs():
    # [FUNCTION DOC]
    # Verifies _check_targets raises ValueError on empty arrays.
    msg = "Found empty input array (e.g., `y_true` or `y_pred`) while a minimum of 1"
    with pytest.raises(ValueError, match=re.escape(msg)):
        _check_targets(np.array([]), np.array([]))


def test__check_targets_multiclass_with_both_y_true_and_y_pred_binary():
    # [FUNCTION DOC]
    # Non-regression for #8098: two binary label arrays with different label sets ({0,
    #   1} and {0, -1}) merge to "multiclass".
    # https://github.com/scikit-learn/scikit-learn/issues/8098
    y_true = [0, 1]
    y_pred = [0, -1]
    assert _check_targets(y_true, y_pred)[0] == "multiclass"


@pytest.mark.parametrize(
    "y, target_type",
    [
        (sparse.csr_matrix([[1], [0], [1], [0]]), "binary"),
        (sparse.csr_matrix([[0], [1], [2], [1]]), "multiclass"),
        (sparse.csr_matrix([[1, 0, 1], [0, 1, 0], [1, 1, 0]]), "multilabel"),
    ],
)
def test__check_targets_sparse_inputs(y, target_type):
    # [FUNCTION DOC]
    # Verifies sparse binary and multiclass targets raise TypeError while sparse
    #   multilabel targets are accepted and returned as CSR with type "multilabel-
    #   indicator".
    """Check correct behaviour when different target types are sparse."""
    if target_type in ("binary", "multiclass"):
        with pytest.raises(
            TypeError, match="Sparse input is only supported when targets"
        ):
            _check_targets(y, y)
    else:
        # This should not raise an error
        y_type, _, y_true_out, y_pred_out, _ = _check_targets(y, y)

        assert y_type == "multilabel-indicator"
        assert y_true_out.format == "csr"
        assert y_pred_out.format == "csr"


def test_hinge_loss_binary():
    # [FUNCTION DOC]
    # Verifies binary hinge_loss for labels in {-1, 1} and for {0, 2} (mapped to -1/+1)
    #   against a hand-computed value.
    y_true = np.array([-1, 1, 1, -1])
    pred_decision = np.array([-8.5, 0.5, 1.5, -0.3])
    assert hinge_loss(y_true, pred_decision) == 1.2 / 4

    y_true = np.array([0, 2, 2, 0])
    pred_decision = np.array([-8.5, 0.5, 1.5, -0.3])
    assert hinge_loss(y_true, pred_decision) == 1.2 / 4


def test_hinge_loss_multiclass():
    # [FUNCTION DOC]
    # Verifies multiclass hinge_loss (Crammer-Singer: 1 minus true-class score plus the
    #   best other score, clipped at 0, averaged) against hand-computed losses.
    pred_decision = np.array(
        [
            [+0.36, -0.17, -0.58, -0.99],
            [-0.54, -0.37, -0.48, -0.58],
            [-1.45, -0.58, -0.38, -0.17],
            [-0.54, -0.38, -0.48, -0.58],
            [-2.36, -0.79, -0.27, +0.24],
            [-1.45, -0.58, -0.38, -0.17],
        ]
    )
    y_true = np.array([0, 1, 2, 1, 3, 2])
    dummy_losses = np.array(
        [
            1 - pred_decision[0][0] + pred_decision[0][1],
            1 - pred_decision[1][1] + pred_decision[1][2],
            1 - pred_decision[2][2] + pred_decision[2][3],
            1 - pred_decision[3][1] + pred_decision[3][2],
            1 - pred_decision[4][3] + pred_decision[4][2],
            1 - pred_decision[5][2] + pred_decision[5][3],
        ]
    )
    np.clip(dummy_losses, 0, None, out=dummy_losses)
    dummy_hinge_loss = np.mean(dummy_losses)
    assert hinge_loss(y_true, pred_decision) == dummy_hinge_loss


def test_hinge_loss_multiclass_missing_labels_with_labels_none():
    # [FUNCTION DOC]
    # Verifies multiclass hinge_loss raises ValueError when pred_decision has more
    #   columns than labels present in y_true and labels is not given.
    y_true = np.array([0, 1, 2, 2])
    pred_decision = np.array(
        [
            [+1.27, 0.034, -0.68, -1.40],
            [-1.45, -0.58, -0.38, -0.17],
            [-2.36, -0.79, -0.27, +0.24],
            [-2.36, -0.79, -0.27, +0.24],
        ]
    )
    error_message = (
        "Please include all labels in y_true or pass labels as third argument"
    )
    with pytest.raises(ValueError, match=error_message):
        hinge_loss(y_true, pred_decision)


def test_hinge_loss_multiclass_no_consistent_pred_decision_shape():
    # [FUNCTION DOC]
    # Verifies ValueError messages for a 1-D pred_decision with multiclass target and
    #   for a column count inconsistent with labels.
    # test for inconsistency between multiclass problem and pred_decision
    # argument
    y_true = np.array([2, 1, 0, 1, 0, 1, 1])
    pred_decision = np.array([0, 1, 2, 1, 0, 2, 1])
    error_message = (
        "The shape of pred_decision cannot be 1d array"
        "with a multiclass target. pred_decision shape "
        "must be (n_samples, n_classes), that is "
        "(7, 3). Got: (7,)"
    )
    with pytest.raises(ValueError, match=re.escape(error_message)):
        hinge_loss(y_true=y_true, pred_decision=pred_decision)

    # test for inconsistency between pred_decision shape and labels number
    pred_decision = np.array([[0, 1], [0, 1], [0, 1], [0, 1], [2, 0], [0, 1], [1, 0]])
    labels = [0, 1, 2]
    error_message = (
        "The shape of pred_decision is not "
        "consistent with the number of classes. "
        "With a multiclass target, pred_decision "
        "shape must be (n_samples, n_classes), that is "
        "(7, 3). Got: (7, 2)"
    )
    with pytest.raises(ValueError, match=re.escape(error_message)):
        hinge_loss(y_true=y_true, pred_decision=pred_decision, labels=labels)


def test_hinge_loss_multiclass_with_missing_labels():
    # [FUNCTION DOC]
    # Verifies hinge_loss with explicit labels including a class absent from y_true
    #   matches hand-computed losses.
    pred_decision = np.array(
        [
            [+0.36, -0.17, -0.58, -0.99],
            [-0.55, -0.38, -0.48, -0.58],
            [-1.45, -0.58, -0.38, -0.17],
            [-0.55, -0.38, -0.48, -0.58],
            [-1.45, -0.58, -0.38, -0.17],
        ]
    )
    y_true = np.array([0, 1, 2, 1, 2])
    labels = np.array([0, 1, 2, 3])
    dummy_losses = np.array(
        [
            1 - pred_decision[0][0] + pred_decision[0][1],
            1 - pred_decision[1][1] + pred_decision[1][2],
            1 - pred_decision[2][2] + pred_decision[2][3],
            1 - pred_decision[3][1] + pred_decision[3][2],
            1 - pred_decision[4][2] + pred_decision[4][3],
        ]
    )
    np.clip(dummy_losses, 0, None, out=dummy_losses)
    dummy_hinge_loss = np.mean(dummy_losses)
    assert hinge_loss(y_true, pred_decision, labels=labels) == dummy_hinge_loss


def test_hinge_loss_multiclass_missing_labels_only_two_unq_in_y_true():
    # [FUNCTION DOC]
    # Non-regression for #17630: hinge_loss with explicit labels works when y_true has
    #   only two of three classes.
    # non-regression test for:
    # https://github.com/scikit-learn/scikit-learn/issues/17630
    # check that we can compute the hinge loss when providing an array
    # with labels allowing to not have all labels in y_true
    pred_decision = np.array(
        [
            [+0.36, -0.17, -0.58],
            [-0.15, -0.58, -0.48],
            [-1.45, -0.58, -0.38],
            [-0.55, -0.78, -0.42],
            [-1.45, -0.58, -0.38],
        ]
    )
    y_true = np.array([0, 2, 2, 0, 2])
    labels = np.array([0, 1, 2])
    dummy_losses = np.array(
        [
            1 - pred_decision[0][0] + pred_decision[0][1],
            1 - pred_decision[1][2] + pred_decision[1][0],
            1 - pred_decision[2][2] + pred_decision[2][1],
            1 - pred_decision[3][0] + pred_decision[3][2],
            1 - pred_decision[4][2] + pred_decision[4][1],
        ]
    )
    np.clip(dummy_losses, 0, None, out=dummy_losses)
    dummy_hinge_loss = np.mean(dummy_losses)
    assert_almost_equal(
        hinge_loss(y_true, pred_decision, labels=labels), dummy_hinge_loss
    )


def test_hinge_loss_multiclass_invariance_lists():
    # [FUNCTION DOC]
    # Verifies multiclass hinge_loss accepts string labels and list inputs and gives the
    #   hand-computed value.
    # Currently, invariance of string and integer labels cannot be tested
    # in common invariance tests because invariance tests for multiclass
    # decision functions is not implemented yet.
    y_true = ["blue", "green", "red", "green", "white", "red"]
    pred_decision = [
        [+0.36, -0.17, -0.58, -0.99],
        [-0.55, -0.38, -0.48, -0.58],
        [-1.45, -0.58, -0.38, -0.17],
        [-0.55, -0.38, -0.48, -0.58],
        [-2.36, -0.79, -0.27, +0.24],
        [-1.45, -0.58, -0.38, -0.17],
    ]
    dummy_losses = np.array(
        [
            1 - pred_decision[0][0] + pred_decision[0][1],
            1 - pred_decision[1][1] + pred_decision[1][2],
            1 - pred_decision[2][2] + pred_decision[2][3],
            1 - pred_decision[3][1] + pred_decision[3][2],
            1 - pred_decision[4][3] + pred_decision[4][2],
            1 - pred_decision[5][2] + pred_decision[5][3],
        ]
    )
    np.clip(dummy_losses, 0, None, out=dummy_losses)
    dummy_hinge_loss = np.mean(dummy_losses)
    assert hinge_loss(y_true, pred_decision) == dummy_hinge_loss


def test_log_loss():
    # [FUNCTION DOC]
    # Verifies log_loss: binary string labels against scipy bernoulli logpmf, a known
    #   multiclass value with normalize True/False, errors for class-count mismatch,
    #   labels missing y_true values, single-label y_true without labels and
    #   inconsistent lengths, and correct results with explicit labels including more
    #   labels than present.
    # binary case with symbolic labels ("no" < "yes")
    y_true = ["no", "no", "no", "yes", "yes", "yes"]
    y_proba = np.array(
        [[0.5, 0.5], [0.1, 0.9], [0.01, 0.99], [0.9, 0.1], [0.75, 0.25], [0.001, 0.999]]
    )
    loss = log_loss(y_true, y_proba)
    loss_true = -np.mean(bernoulli.logpmf(np.array(y_true) == "yes", y_proba[:, 1]))
    assert_allclose(loss, loss_true)

    # multiclass case; adapted from http://bit.ly/RJJHWA
    y_true = [1, 0, 2]
    y_proba = [[0.2, 0.7, 0.1], [0.6, 0.2, 0.2], [0.6, 0.1, 0.3]]
    loss = log_loss(y_true, y_proba, normalize=True)
    assert_allclose(loss, 0.6904911)

    # check that we got all the shapes and axes right
    # by doubling the length of y_true and y_proba
    y_true *= 2
    y_proba *= 2
    loss = log_loss(y_true, y_proba, normalize=False)
    assert_allclose(loss, 0.6904911 * 6)

    # raise error if number of classes are not equal.
    y_true = [1, 0, 2]
    y_proba = [[0.3, 0.7], [0.6, 0.4], [0.4, 0.6]]
    with pytest.raises(ValueError):
        log_loss(y_true, y_proba)

    # raise error if labels do not contain all values of y_true
    y_true = ["a", "b", "c"]
    y_proba = [[0.9, 0.1, 0.0], [0.1, 0.9, 0.0], [0.1, 0.1, 0.8]]
    labels = ["a", "c", "d"]
    error_str = (
        "y_true contains values {'b'} not belonging to the passed "
        "labels ['a', 'c', 'd']."
    )
    with pytest.raises(ValueError, match=re.escape(error_str)):
        log_loss(y_true, y_proba, labels=labels)

    # case when y_true is a string array object
    y_true = ["ham", "spam", "spam", "ham"]
    y_proba = [[0.3, 0.7], [0.6, 0.4], [0.4, 0.6], [0.7, 0.3]]
    loss = log_loss(y_true, y_proba)
    assert_allclose(loss, 0.7469410)

    # test labels option

    y_true = [2, 2]
    y_proba = [[0.2, 0.8], [0.6, 0.4]]
    y_score = np.array([[0.1, 0.9], [0.1, 0.9]])
    error_str = (
        "y_true contains only one label (2). Please provide the list of all "
        "expected class labels explicitly through the labels argument."
    )
    with pytest.raises(ValueError, match=re.escape(error_str)):
        log_loss(y_true, y_proba)

    y_proba = [[0.2, 0.8], [0.6, 0.4], [0.7, 0.3]]
    error_str = "Found input variables with inconsistent numbers of samples: [3, 2]"
    with pytest.raises(ValueError, match=re.escape(error_str)):
        log_loss(y_true, y_proba)

    # works when the labels argument is used

    true_log_loss = -np.mean(np.log(y_score[:, 1]))
    calculated_log_loss = log_loss(y_true, y_score, labels=[1, 2])
    assert_allclose(calculated_log_loss, true_log_loss)

    # ensure labels work when len(np.unique(y_true)) != y_proba.shape[1]
    y_true = [1, 2, 2]
    y_score2 = [[0.7, 0.1, 0.2], [0.2, 0.7, 0.1], [0.1, 0.7, 0.2]]
    loss = log_loss(y_true, y_score2, labels=[1, 2, 3])
    assert_allclose(loss, -np.log(0.7))


@pytest.mark.parametrize("dtype", [np.float64, np.float32, np.float16])
def test_log_loss_eps(dtype):
    # [FUNCTION DOC]
    # Non-regression for #24315: log_loss stays finite for completely wrong 0/1
    #   probabilities in float64, float32 and float16 thanks to dtype-dependent
    #   clipping.
    """Check the behaviour internal eps that changes depending on the input dtype.

    Non-regression test for:
    https://github.com/scikit-learn/scikit-learn/issues/24315
    """
    y_true = np.array([0, 1], dtype=dtype)
    y_proba = np.array([1, 0], dtype=dtype)

    loss = log_loss(y_true, y_proba)
    assert np.isfinite(loss)


@pytest.mark.parametrize("dtype", [np.float64, np.float32, np.float16])
def test_log_loss_not_probabilities_warning(dtype):
    # [FUNCTION DOC]
    # Verifies log_loss warns when probability rows do not sum to one, for several float
    #   dtypes.
    """Check that log_loss raises a warning when y_proba values don't sum to 1."""
    y_true = np.array([0, 1, 1, 0])
    y_proba = np.array([[0.2, 0.7], [0.6, 0.3], [0.4, 0.7], [0.8, 0.3]], dtype=dtype)

    with pytest.warns(UserWarning, match="The y_prob values do not sum to one."):
        log_loss(y_true, y_proba)


@pytest.mark.parametrize(
    "y_true, y_proba",
    [
        ([0, 1, 0], [0, 1, 0]),
        ([0, 1, 0], [[1, 0], [0, 1], [1, 0]]),
        ([0, 1, 2], [[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
    ],
)
def test_log_loss_perfect_predictions(y_true, y_proba):
    # [FUNCTION DOC]
    # Verifies log_loss is approximately 0 for perfect predictions given as 1-D or 2-D
    #   probabilities.
    """Check that log_loss returns 0 for perfect predictions."""
    # Because of the clipping, the result is not exactly 0
    assert log_loss(y_true, y_proba) == pytest.approx(0)


def test_log_loss_pandas_input():
    # [FUNCTION DOC]
    # Non-regression for gh-5715: log_loss accepts MockDataFrame and, when pandas is
    #   installed, Series/DataFrame inputs.
    # case when input is a pandas series and dataframe gh-5715
    y_tr = np.array(["ham", "spam", "spam", "ham"])
    y_pr = np.array([[0.3, 0.7], [0.6, 0.4], [0.4, 0.6], [0.7, 0.3]])
    types = [(MockDataFrame, MockDataFrame)]
    try:
        from pandas import DataFrame, Series

        types.append((Series, DataFrame))
    except ImportError:
        pass
    for TrueInputType, PredInputType in types:
        # y_proba dataframe, y_true series
        y_true, y_proba = TrueInputType(y_tr), PredInputType(y_pr)
        loss = log_loss(y_true, y_proba)
        assert_allclose(loss, 0.7469410)


def test_log_loss_warnings():
    # [FUNCTION DOC]
    # Verifies log_loss warns when labels are passed in non-lexicographic order.
    expected_message = re.escape(
        "Labels passed were ['spam', 'eggs', 'ham']. But this function "
        "assumes labels are ordered lexicographically. "
        "Pass the ordered labels=['eggs', 'ham', 'spam'] and ensure that "
        "the columns of y_prob correspond to this ordering."
    )
    with pytest.warns(UserWarning, match=expected_message):
        log_loss(
            ["eggs", "spam", "ham"],
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            labels=["spam", "eggs", "ham"],
        )


# TODO(1.11): Remove
def test_log_loss_y_pred_deprecation():
    # [FUNCTION DOC]
    # Verifies the deprecated y_pred keyword of log_loss emits FutureWarning, and
    #   passing both y_pred and y_proba raises ValueError.
    """Test `y_pred` deprecation in favor of `y_proba` for `log_loss`."""
    y_true = np.array([0, 1, 1, 0])
    y_proba = np.array([[0.1, 0.9], [0.9, 0.1], [0.8, 0.2], [0.35, 0.65]])

    # Check no error raised
    log_loss(y_true, y_proba)

    msg = "`y_pred` was renamed to `y_proba` in version 1.9 and will be removed "
    with pytest.warns(FutureWarning, match=re.escape(msg)):
        log_loss(y_true, y_pred=y_proba)

    msg = "Cannot use both `y_pred` and `y_proba`. `y_pred` is deprecated, "
    with pytest.raises(ValueError, match=re.escape(msg)):
        log_loss(y_true, y_pred=y_proba, y_proba=y_proba)


def test_brier_score_loss_binary():
    # [FUNCTION DOC]
    # Verifies binary brier_score_loss equals the mean squared difference, is invariant
    #   to label encoding and to two-column inputs, honours scale_by_half (False doubles
    #   it), and handles a single sample with pos_label.
    # Check brier_score_loss function
    y_true = np.array([0, 1, 1, 0, 1, 1])
    y_prob = np.array([0.1, 0.8, 0.9, 0.3, 1.0, 0.95])
    true_score = linalg.norm(y_true - y_prob) ** 2 / len(y_true)

    assert_almost_equal(brier_score_loss(y_true, y_true), 0.0)
    assert_almost_equal(brier_score_loss(y_true, y_prob), true_score)
    assert_almost_equal(brier_score_loss(1.0 + y_true, y_prob), true_score)
    assert_almost_equal(brier_score_loss(2 * y_true - 1, y_prob), true_score)

    # check that using (n_samples, 2) y_prob or y_true gives the same score
    y_prob_reshaped = np.column_stack((1 - y_prob, y_prob))
    y_true_reshaped = np.column_stack((1 - y_true, y_true))
    assert_almost_equal(brier_score_loss(y_true, y_prob_reshaped), true_score)
    assert_almost_equal(brier_score_loss(y_true_reshaped, y_prob_reshaped), true_score)

    # check scale_by_half argument
    assert_almost_equal(
        brier_score_loss(y_true, y_prob, scale_by_half="auto"), true_score
    )
    assert_almost_equal(
        brier_score_loss(y_true, y_prob, scale_by_half=True), true_score
    )
    assert_almost_equal(
        brier_score_loss(y_true, y_prob, scale_by_half=False), 2 * true_score
    )

    # calculate correctly when there's only one class in y_true
    assert_almost_equal(brier_score_loss([-1], [0.4]), 0.4**2)
    assert_almost_equal(brier_score_loss([0], [0.4]), 0.4**2)
    assert_almost_equal(brier_score_loss([1], [0.4]), (1 - 0.4) ** 2)
    assert_almost_equal(brier_score_loss(["foo"], [0.4], pos_label="bar"), 0.4**2)
    assert_almost_equal(
        brier_score_loss(["foo"], [0.4], pos_label="foo"),
        (1 - 0.4) ** 2,
    )


def test_brier_score_loss_multiclass():
    # [FUNCTION DOC]
    # Verifies multiclass brier_score_loss values: with explicit labels including an
    #   unused class, a known example, 0 for perfect and 2 for perfectly wrong
    #   predictions.
    # test cases for multi-class
    assert_almost_equal(
        brier_score_loss(
            ["eggs", "spam", "ham"],
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0]],
            labels=["eggs", "ham", "spam", "yams"],
        ),
        2 / 3,
    )

    assert_almost_equal(
        brier_score_loss(
            [1, 0, 2], [[0.2, 0.7, 0.1], [0.6, 0.2, 0.2], [0.6, 0.1, 0.3]]
        ),
        0.41333333,
    )

    # check perfect predictions for 3 classes
    assert_almost_equal(
        brier_score_loss(
            [0, 1, 2], [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        ),
        0,
    )

    # check perfectly incorrect predictions for 3 classes
    assert_almost_equal(
        brier_score_loss(
            [0, 1, 2], [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
        ),
        2,
    )


def test_brier_score_loss_invalid_inputs():
    # [FUNCTION DOC]
    # Verifies brier_score_loss raises ValueError for length mismatch, probabilities
    #   outside [0, 1], multiclass y_true with 1-D y_prob, class-count mismatches with
    #   y_true or labels, and single-label y_true without labels; and that passing
    #   labels resolves the last case.
    # binary case
    y_true = np.array([0, 1, 1, 0, 1, 1])
    y_prob = np.array([0.1, 0.8, 0.9, 0.3, 1.0, 0.95])
    with pytest.raises(ValueError):
        # bad length of y_prob
        brier_score_loss(y_true, y_prob[1:])
    with pytest.raises(ValueError):
        # y_pred has value greater than 1
        brier_score_loss(y_true, y_prob + 1.0)
    with pytest.raises(ValueError):
        # y_pred has value less than 0
        brier_score_loss(y_true, y_prob - 1.0)

    # multiclass case
    y_true = np.array([1, 0, 2])
    y_prob = np.array([[0.2, 0.7, 0.1], [0.6, 0.2, 0.2], [0.6, 0.1, 0.3]])
    with pytest.raises(ValueError):
        # bad length of y_pred
        brier_score_loss(y_true, y_prob[1:])
    with pytest.raises(ValueError):
        # y_pred has value greater than 1
        brier_score_loss(y_true, y_prob + 1.0)
    with pytest.raises(ValueError):
        # y_pred has value less than 0
        brier_score_loss(y_true, y_prob - 1.0)

    # raise an error for multiclass y_true and binary y_prob
    y_true = np.array([0, 1, 2, 0])
    y_prob = np.array([0.8, 0.6, 0.4, 0.2])
    error_message = re.escape(
        "The type of the target inferred from y_true is multiclass "
        "but should be binary according to the shape of y_prob."
    )
    with pytest.raises(ValueError, match=error_message):
        brier_score_loss(y_true, y_prob)

    # raise an error for wrong number of classes
    y_true = [0, 1, 2]
    y_prob = [[1, 0], [0, 1], [0, 1]]
    error_message = (
        "y_true and y_prob contain different number of "
        "classes: 3 vs 2. Please provide the true "
        "labels explicitly through the labels argument. "
        "Classes found in "
        "y_true: [0 1 2]"
    )
    with pytest.raises(ValueError, match=re.escape(error_message)):
        brier_score_loss(y_true, y_prob)

    y_true = ["eggs", "spam", "ham"]
    y_prob = [[1, 0, 0], [0, 1, 0], [0, 1, 0]]
    labels = ["eggs", "spam", "ham", "yams"]
    error_message = (
        "The number of classes in labels is different "
        "from that in y_prob. Classes found in "
        "labels: ['eggs' 'ham' 'spam' 'yams']"
    )
    with pytest.raises(ValueError, match=re.escape(error_message)):
        brier_score_loss(y_true, y_prob, labels=labels)

    # raise error message when there's only one class in y_true
    y_true = ["eggs"]
    y_prob = [[0.9, 0.1]]
    error_message = (
        "y_true contains only one label (eggs). Please "
        "provide the list of all expected class labels explicitly through the "
        "labels argument."
    )
    with pytest.raises(ValueError, match=re.escape(error_message)):
        brier_score_loss(y_true, y_prob)

    # error is fixed when labels is specified
    assert_almost_equal(brier_score_loss(y_true, y_prob, labels=["eggs", "ham"]), 0.01)


def test_brier_score_loss_warnings():
    # [FUNCTION DOC]
    # Verifies brier_score_loss warns when labels are not in lexicographic order.
    expected_message = re.escape(
        "Labels passed were ['spam', 'eggs', 'ham']. But this function "
        "assumes labels are ordered lexicographically. "
        "Pass the ordered labels=['eggs', 'ham', 'spam'] and ensure that "
        "the columns of y_prob correspond to this ordering."
    )
    with pytest.warns(UserWarning, match=expected_message):
        brier_score_loss(
            ["eggs", "spam", "ham"],
            [
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
            ],
            labels=["spam", "eggs", "ham"],
        )


def test_balanced_accuracy_score_unseen():
    # [FUNCTION DOC]
    # Verifies balanced_accuracy_score warns when y_pred contains a class absent from
    #   y_true.
    msg = "y_pred contains classes not in y_true"
    with pytest.warns(UserWarning, match=msg):
        balanced_accuracy_score([0, 0, 0], [0, 0, 1])


@pytest.mark.parametrize(
    "y_true,y_pred",
    [
        (["a", "b", "a", "b"], ["a", "a", "a", "b"]),
        (["a", "b", "c", "b"], ["a", "a", "a", "b"]),
        (["a", "a", "a", "b"], ["a", "b", "c", "b"]),
    ],
)
def test_balanced_accuracy_score(y_true, y_pred):
    # [FUNCTION DOC]
    # Verifies balanced_accuracy_score equals macro recall over y_true's classes and
    #   that adjusted=True rescales it by chance level computed from a constant
    #   prediction.
    macro_recall = recall_score(
        y_true, y_pred, average="macro", labels=np.unique(y_true)
    )
    with ignore_warnings():
        # Warnings are tested in test_balanced_accuracy_score_unseen
        balanced = balanced_accuracy_score(y_true, y_pred)
    assert balanced == pytest.approx(macro_recall)
    adjusted = balanced_accuracy_score(y_true, y_pred, adjusted=True)
    chance = balanced_accuracy_score(y_true, np.full_like(y_true, y_true[0]))
    assert adjusted == (balanced - chance) / (1 - chance)


@pytest.mark.parametrize(
    "metric",
    [
        jaccard_score,
        f1_score,
        partial(fbeta_score, beta=0.5),
        precision_recall_fscore_support,
        precision_score,
        recall_score,
        brier_score_loss,
    ],
)
@pytest.mark.parametrize(
    "classes", [(False, True), (0, 1), (0.0, 1.0), ("zero", "one")]
)
def test_classification_metric_pos_label_types(metric, classes):
    # [FUNCTION DOC]
    # Verifies binary metrics accept bool, int, float and string pos_label without NaN
    #   results (brier_score_loss gets random probabilities).
    """Check that the metric works with different types of `pos_label`.

    We can expect `pos_label` to be a bool, an integer, a float, a string.
    No error should be raised for those types.
    """
    rng = np.random.RandomState(42)
    n_samples, pos_label = 10, classes[-1]
    y_true = rng.choice(classes, size=n_samples, replace=True)
    if metric is brier_score_loss:
        # brier score loss requires probabilities
        y_pred = rng.uniform(size=n_samples)
    else:
        y_pred = y_true.copy()
    result = metric(y_true, y_pred, pos_label=pos_label)
    assert not np.any(np.isnan(result))


@pytest.mark.parametrize(
    "y_true, y_pred, expected_score",
    [
        (np.array([0, 1]), np.array([1, 0]), 0.0),
        (np.array([0, 1]), np.array([0, 1]), 1.0),
        (np.array([0, 1]), np.array([0, 0]), 0.0),
        (np.array([0, 0]), np.array([0, 0]), 1.0),
    ],
)
def test_f1_for_small_binary_inputs_with_zero_division(y_true, y_pred, expected_score):
    # [FUNCTION DOC]
    # Non-regression for #26965: f1_score with zero_division=1.0 on two-sample binary
    #   inputs returns the expected scores.
    """Check the behaviour of `zero_division` for f1-score.

    Non-regression test for:
    https://github.com/scikit-learn/scikit-learn/issues/26965
    """
    assert f1_score(y_true, y_pred, zero_division=1.0) == pytest.approx(expected_score)


# TODO: remove mark once loky bug is fixed:
# https://github.com/joblib/loky/issues/458
@pytest.mark.thread_unsafe
@pytest.mark.parametrize(
    "scoring",
    [
        make_scorer(f1_score, zero_division=np.nan),
        make_scorer(fbeta_score, beta=2, zero_division=np.nan),
        make_scorer(precision_score, zero_division=np.nan),
        make_scorer(recall_score, zero_division=np.nan),
    ],
)
def test_classification_metric_division_by_zero_nan_validation(scoring):
    # [FUNCTION DOC]
    # Non-regression for #27563: scorers built with zero_division=np.nan work in
    #   cross_val_score with n_jobs=2, where NaN identity is not preserved across
    #   processes. Passes if no error is raised.
    """Check that we validate `np.nan` properly for classification metrics.

    With `n_jobs=2` in cross-validation, the `np.nan` used for the singleton will be
    different in the sub-process and we should not use the `is` operator but
    `math.isnan`.

    Non-regression test for:
    https://github.com/scikit-learn/scikit-learn/issues/27563
    """
    X, y = datasets.make_classification(random_state=0)
    classifier = DecisionTreeClassifier(max_depth=3, random_state=0).fit(X, y)
    cross_val_score(classifier, X, y, scoring=scoring, n_jobs=2, error_score="raise")


def test_d2_log_loss_score():
    # [FUNCTION DOC]
    # Verifies d2_log_loss_score equals 1 minus the ratio of model log loss to the log
    #   loss of a null model predicting (weighted) class proportions, with and without
    #   sample weights; that good predictions score in (0.5, 1), poor ones below 0, and
    #   class-proportion predictions exactly 0; and that string labels and multiclass
    #   targets behave consistently.
    y_true = [0, 0, 0, 1, 1, 1]
    y_true_string = ["no", "no", "no", "yes", "yes", "yes"]
    y_proba = np.array(
        [
            [0.5, 0.5],
            [0.9, 0.1],
            [0.4, 0.6],
            [0.6, 0.4],
            [0.35, 0.65],
            [0.01, 0.99],
        ]
    )
    y_proba_null = np.array(
        [
            [0.5, 0.5],
            [0.5, 0.5],
            [0.5, 0.5],
            [0.5, 0.5],
            [0.5, 0.5],
            [0.5, 0.5],
        ]
    )
    d2_score = d2_log_loss_score(y_true=y_true, y_proba=y_proba)
    log_likelihood = log_loss(y_true=y_true, y_proba=y_proba, normalize=False)
    log_likelihood_null = log_loss(y_true=y_true, y_proba=y_proba_null, normalize=False)
    d2_score_true = 1 - log_likelihood / log_likelihood_null
    assert d2_score == pytest.approx(d2_score_true)

    # check that using sample weight also gives the correct d2 score
    sample_weight = np.array([2, 1, 3, 4, 3, 1])
    y_proba_null[:, 0] = sample_weight[:3].sum() / sample_weight.sum()
    y_proba_null[:, 1] = sample_weight[3:].sum() / sample_weight.sum()
    d2_score = d2_log_loss_score(
        y_true=y_true, y_proba=y_proba, sample_weight=sample_weight
    )
    log_likelihood = log_loss(
        y_true=y_true,
        y_proba=y_proba,
        sample_weight=sample_weight,
        normalize=False,
    )
    log_likelihood_null = log_loss(
        y_true=y_true,
        y_proba=y_proba_null,
        sample_weight=sample_weight,
        normalize=False,
    )
    d2_score_true = 1 - log_likelihood / log_likelihood_null
    assert d2_score == pytest.approx(d2_score_true)

    # check if good predictions give a relatively higher value for the d2 score
    y_proba = np.array(
        [
            [0.9, 0.1],
            [0.8, 0.2],
            [0.9, 0.1],
            [0.1, 0.9],
            [0.2, 0.8],
            [0.1, 0.9],
        ]
    )
    d2_score = d2_log_loss_score(y_true, y_proba)
    assert 0.5 < d2_score < 1.0
    # check that a similar value is obtained for string labels
    d2_score_string = d2_log_loss_score(y_true_string, y_proba)
    assert d2_score_string == pytest.approx(d2_score)

    # check if poor predictions gives a relatively low value for the d2 score
    y_proba = np.array(
        [
            [0.5, 0.5],
            [0.1, 0.9],
            [0.1, 0.9],
            [0.9, 0.1],
            [0.75, 0.25],
            [0.1, 0.9],
        ]
    )
    d2_score = d2_log_loss_score(y_true, y_proba)
    assert d2_score < 0
    # check that a similar value is obtained for string labels
    d2_score_string = d2_log_loss_score(y_true_string, y_proba)
    assert d2_score_string == pytest.approx(d2_score)

    # check if simply using the average of the classes as the predictions
    # gives a d2 score of 0
    y_true = [0, 0, 0, 1, 1, 1]
    y_proba = np.array(
        [
            [0.5, 0.5],
            [0.5, 0.5],
            [0.5, 0.5],
            [0.5, 0.5],
            [0.5, 0.5],
            [0.5, 0.5],
        ]
    )
    d2_score = d2_log_loss_score(y_true, y_proba)
    assert d2_score == 0
    d2_score_string = d2_log_loss_score(y_true_string, y_proba)
    assert d2_score_string == 0

    # check if simply using the average of the classes as the predictions
    # gives a d2 score of 0 when the positive class has a higher proportion
    y_true = [0, 1, 1, 1]
    y_true_string = ["no", "yes", "yes", "yes"]
    y_proba = np.array([[0.25, 0.75], [0.25, 0.75], [0.25, 0.75], [0.25, 0.75]])
    d2_score = d2_log_loss_score(y_true, y_proba)
    assert d2_score == 0
    d2_score_string = d2_log_loss_score(y_true_string, y_proba)
    assert d2_score_string == 0
    sample_weight = [2, 2, 2, 2]
    d2_score_with_sample_weight = d2_log_loss_score(
        y_true, y_proba, sample_weight=sample_weight
    )
    assert d2_score_with_sample_weight == 0

    # check that the d2 scores seem correct when more than 2
    # labels are specified
    y_true = ["high", "high", "low", "neutral"]
    sample_weight = [1.4, 0.6, 0.8, 0.2]

    y_proba = np.array(
        [
            [0.8, 0.1, 0.1],
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
        ]
    )
    d2_score = d2_log_loss_score(y_true, y_proba)
    assert 0.5 < d2_score < 1.0
    d2_score = d2_log_loss_score(y_true, y_proba, sample_weight=sample_weight)
    assert 0.5 < d2_score < 1.0

    y_proba = np.array(
        [
            [0.2, 0.5, 0.3],
            [0.1, 0.7, 0.2],
            [0.1, 0.1, 0.8],
            [0.2, 0.7, 0.1],
        ]
    )
    d2_score = d2_log_loss_score(y_true, y_proba)
    assert d2_score < 0
    d2_score = d2_log_loss_score(y_true, y_proba, sample_weight=sample_weight)
    assert d2_score < 0


def test_d2_log_loss_score_missing_labels():
    # [FUNCTION DOC]
    # Non-regression for #30713: d2_log_loss_score with labels including a class absent
    #   from y_true matches the manually computed weighted null-model ratio.
    """Check that d2_log_loss_score works when not all labels are present in y_true

    non-regression test for https://github.com/scikit-learn/scikit-learn/issues/30713
    """
    y_true = [2, 0, 2, 0]
    labels = [0, 1, 2]
    sample_weight = [1.4, 0.6, 0.7, 0.3]
    y_proba = np.tile([1, 0, 0], (4, 1))

    log_loss_obs = log_loss(y_true, y_proba, sample_weight=sample_weight, labels=labels)

    # Null model consists of weighted average of the classes.
    # Given that the sum of the weights is 3,
    # - weighted average of 0s is (0.6 + 0.3) / 3 = 0.3
    # - weighted average of 1s is 0
    # - weighted average of 2s is (1.4 + 0.7) / 3 = 0.7
    y_proba_null = np.tile([0.3, 0, 0.7], (4, 1))
    log_loss_null = log_loss(
        y_true, y_proba_null, sample_weight=sample_weight, labels=labels
    )

    expected_d2_score = 1 - log_loss_obs / log_loss_null
    d2_score = d2_log_loss_score(
        y_true, y_proba, sample_weight=sample_weight, labels=labels
    )
    assert_allclose(d2_score, expected_d2_score)


def test_d2_log_loss_score_label_order():
    # [FUNCTION DOC]
    # Verifies d2_log_loss_score is unchanged when the labels list is permuted.
    """Check that d2_log_loss_score doesn't depend on the order of the labels."""
    y_true = [2, 0, 2, 0]
    y_proba = np.tile([1, 0, 0], (4, 1))

    d2_score = d2_log_loss_score(y_true, y_proba, labels=[0, 1, 2])
    d2_score_other = d2_log_loss_score(y_true, y_proba, labels=[0, 2, 1])

    assert_allclose(d2_score, d2_score_other)


def test_d2_log_loss_score_raises():
    # [FUNCTION DOC]
    # Verifies d2_log_loss_score raises ValueError for class-count mismatches (with
    #   y_true or labels), inconsistent lengths, single-label y_true and a single-label
    #   labels list, and warns with UndefinedMetricWarning for fewer than two samples.
    """Test that d2_log_loss_score raises the appropriate errors on
    invalid inputs."""
    y_true = [0, 1, 2]
    y_proba = [[0.2, 0.8], [0.5, 0.5], [0.4, 0.6]]
    err = "contain different number of classes"
    with pytest.raises(ValueError, match=err):
        d2_log_loss_score(y_true, y_proba)

    # check error if the number of classes in labels do not match the number
    # of classes in y_proba.
    y_true = [0, 1, 2]
    y_proba = [[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]]
    labels = [0, 1, 2]
    err = "number of classes in labels is different"
    with pytest.raises(ValueError, match=err):
        d2_log_loss_score(y_true, y_proba, labels=labels)

    # check error if y_true and y_proba do not have equal lengths
    y_true = [0, 1, 2]
    y_proba = [[0.5, 0.5, 0.5], [0.6, 0.3, 0.1]]
    err = "inconsistent numbers of samples"
    with pytest.raises(ValueError, match=err):
        d2_log_loss_score(y_true, y_proba)

    # check warning for samples < 2
    y_true = [1]
    y_proba = [[0.5, 0.5]]
    err = "score is not well-defined"
    with pytest.warns(UndefinedMetricWarning, match=err):
        d2_log_loss_score(y_true, y_proba)

    # check error when y_true only has 1 label
    y_true = [1, 1, 1]
    y_proba = [[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]]
    err = "y_true contains only one label"
    with pytest.raises(ValueError, match=err):
        d2_log_loss_score(y_true, y_proba)

    # check error when y_true only has 1 label and labels also has
    # only 1 label
    y_true = [1, 1, 1]
    labels = [1]
    y_proba = [[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]]
    err = "The labels array needs to contain at least two"
    with pytest.raises(ValueError, match=err):
        d2_log_loss_score(y_true, y_proba, labels=labels)


# TODO(1.11): Remove
def test_d2_log_loss_score_y_pred_deprecation():
    # [FUNCTION DOC]
    # Verifies the deprecated y_pred keyword of d2_log_loss_score emits FutureWarning
    #   and that passing both y_pred and y_proba raises ValueError.
    """Test `y_pred` deprecation in favor of `y_proba` for `d2_log_loss_score`."""
    y_true = np.array([0, 1, 1, 0])
    y_proba = np.array([[0.1, 0.9], [0.9, 0.1], [0.8, 0.2], [0.35, 0.65]])

    # Check no error raised
    d2_log_loss_score(y_true, y_proba)

    msg = "`y_pred` was renamed to `y_proba` in version 1.9 and will be removed "
    with pytest.warns(FutureWarning, match=re.escape(msg)):
        d2_log_loss_score(y_true, y_pred=y_proba)

    msg = "Cannot use both `y_pred` and `y_proba`. `y_pred` is deprecated, "
    with pytest.raises(ValueError, match=re.escape(msg)):
        d2_log_loss_score(y_true, y_pred=y_proba, y_proba=y_proba)


def test_d2_brier_score():
    # [FUNCTION DOC]
    # Verifies d2_brier_score equals 1 minus the ratio of the model Brier loss to that
    #   of a 0.5 reference in the binary case, is 0 for constant predictions equal to
    #   the (weighted) class proportions for binary and multiclass, int and string
    #   labels, and exceeds 0.5 for good multiclass predictions.
    """Test that d2_brier_score gives expected outcomes in both the binary and
    multiclass settings.
    """
    # Binary targets
    sample_weight = [2, 2, 3, 1, 1, 1]
    y_true = [0, 1, 1, 0, 0, 1]
    y_true_string = ["no", "yes", "yes", "no", "no", "yes"]

    # check that the value of the returned d2 score is correct
    y_proba = [0.3, 0.5, 0.6, 0.7, 0.9, 0.8]
    y_proba_ref = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    d2_score = d2_brier_score(y_true=y_true, y_proba=y_proba)
    brier_score_model = brier_score_loss(y_true=y_true, y_proba=y_proba)
    brier_score_ref = brier_score_loss(y_true=y_true, y_proba=y_proba_ref)
    d2_score_expected = 1 - brier_score_model / brier_score_ref
    assert pytest.approx(d2_score) == d2_score_expected

    # check that a model which gives a constant prediction equal to the
    # proportion of the positive class should get a d2 score of 0
    y_proba = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    d2_score = d2_brier_score(y_true=y_true, y_proba=y_proba)
    assert d2_score == 0
    d2_score = d2_brier_score(y_true=y_true_string, y_proba=y_proba, pos_label="yes")
    assert d2_score == 0

    # check that a model which gives a constant prediction equal to the
    # proportion of the positive class should get a d2 score of 0
    # when we also provide sample weight
    y_proba = [0.6, 0.6, 0.6, 0.6, 0.6, 0.6]
    d2_score = d2_brier_score(
        y_true=y_true, y_proba=y_proba, sample_weight=sample_weight
    )
    assert d2_score == 0
    d2_score = d2_brier_score(
        y_true=y_true_string,
        y_proba=y_proba,
        sample_weight=sample_weight,
        pos_label="yes",
    )
    assert d2_score == 0

    # Multiclass targets
    sample_weight = [2, 1, 3, 1, 1, 2, 1, 4, 1, 4]
    y_true = [3, 3, 2, 2, 2, 1, 1, 1, 1, 0]
    y_true_string = ["dd", "dd", "cc", "cc", "cc", "bb", "bb", "bb", "bb", "aa"]

    # check that a model which gives a constant prediction equal to the
    # proportion of the given labels gives a d2 score of 0 when we also
    # provide sample weight
    y_proba = [
        [0.2, 0.4, 0.25, 0.15],
        [0.2, 0.4, 0.25, 0.15],
        [0.2, 0.4, 0.25, 0.15],
        [0.2, 0.4, 0.25, 0.15],
        [0.2, 0.4, 0.25, 0.15],
        [0.2, 0.4, 0.25, 0.15],
        [0.2, 0.4, 0.25, 0.15],
        [0.2, 0.4, 0.25, 0.15],
        [0.2, 0.4, 0.25, 0.15],
        [0.2, 0.4, 0.25, 0.15],
    ]
    d2_score = d2_brier_score(
        y_true=y_true, y_proba=y_proba, sample_weight=sample_weight
    )
    assert d2_score == 0
    d2_score = d2_brier_score(
        y_true=y_true_string,
        y_proba=y_proba,
        sample_weight=sample_weight,
    )
    assert d2_score == 0

    # check that a model which gives generally good predictions has
    # a d2 score that is greater than 0.5
    y_proba = [
        [0.1, 0.2, 0.2, 0.5],
        [0.1, 0.2, 0.2, 0.5],
        [0.1, 0.2, 0.5, 0.2],
        [0.1, 0.2, 0.5, 0.2],
        [0.1, 0.2, 0.5, 0.2],
        [0.2, 0.5, 0.2, 0.1],
        [0.2, 0.5, 0.2, 0.1],
        [0.2, 0.5, 0.2, 0.1],
        [0.2, 0.5, 0.2, 0.1],
        [0.5, 0.2, 0.2, 0.1],
    ]
    d2_score = d2_brier_score(
        y_true=y_true, y_proba=y_proba, sample_weight=sample_weight
    )
    assert d2_score > 0.5
    d2_score = d2_brier_score(
        y_true=y_true_string,
        y_proba=y_proba,
        sample_weight=sample_weight,
    )
    assert d2_score > 0.5


def test_d2_brier_score_with_labels():
    # [FUNCTION DOC]
    # Verifies d2_brier_score with explicit labels including an absent class: proportion
    #   predictions score 0 regardless of labels order, and confidently wrong
    #   predictions score -3.
    """Test that d2_brier_score gives expected outcomes when labels are passed"""
    # Check when labels are provided and some labels may not be present inside
    # y_true, the d2 score is 0, when we use the label proportions based on
    # y_true as the predictions
    y_true = [0, 2, 0, 2]
    labels = [0, 1, 2]
    y_proba = [
        [0.5, 0, 0.5],
        [0.5, 0, 0.5],
        [0.5, 0, 0.5],
        [0.5, 0, 0.5],
    ]
    d2_score = d2_brier_score(y_true=y_true, y_proba=y_proba, labels=labels)
    assert d2_score == 0

    # Also confirm that the order of the labels does not affect the d2 score
    labels = [2, 0, 1]
    new_d2_score = d2_brier_score(y_true=y_true, y_proba=y_proba, labels=labels)
    assert new_d2_score == pytest.approx(d2_score)

    # Check that a simple model with wrong predictions gives a negative d2 score
    y_proba = [
        [0, 0, 1],
        [1, 0, 0],
        [0, 0, 1],
        [1, 0, 0],
    ]
    neg_d2_score = d2_brier_score(y_true=y_true, y_proba=y_proba, labels=labels)
    assert pytest.approx(neg_d2_score) == -3


@pytest.mark.parametrize(
    "y_true, y_pred, labels, error_msg",
    [
        (
            [1, 2, 1, 3],
            [0.8, 0.6, 0.4, 0.2],
            None,
            "inferred from y_true is multiclass but should be binary",
        ),
        (
            ["yes", "no", "yes", "no"],
            [0.8, 0.6, 0.4, 0.2],
            None,
            "pos_label is not specified",
        ),
        (
            [0, 1, 0, 0, 1, 1, 0],
            [0.8, 0.6, 0.4, 0.2],
            None,
            "variables with inconsistent numbers of samples",
        ),
        (
            [0, 1, 0, 1],
            [1.8, 0.6, 0.4, 0.2],
            None,
            "y_prob contains values greater than 1",
        ),
        (
            [0, 1, 0, 1],
            [-0.8, 0.6, 0.4, 0.2],
            None,
            "y_prob contains values less than 0",
        ),
        (
            [1, 1, 1],
            [[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]],
            None,
            "y_true contains only one label",
        ),
        (
            [[1, 0, 1, 0], [2, 3, 3, 2]],
            [[0.3, 0.3, 0.2, 0.2], [0.4, 0.1, 0.3, 0.2]],
            None,
            "Multioutput target data is not supported",
        ),
        (
            [1, 2, 0],
            [[0.5, 0.3, 0.2], [0.5, 0.3, 0.2], [0.5, 0.3, 0.2]],
            [0, 2],
            "not belonging to the passed labels",
        ),
        (
            [0, 0, 0],
            [[0.5, 0.3, 0.2], [0.5, 0.3, 0.2], [0.5, 0.3, 0.2]],
            [0],
            "labels array needs to contain at least two",
        ),
    ],
)
def test_d2_brier_score_raises(y_true, y_pred, labels, error_msg):
    # [FUNCTION DOC]
    # Verifies d2_brier_score raises ValueError with the expected message for each
    #   invalid input case (type mismatch, missing pos_label for strings, length
    #   mismatch, probabilities out of range, single label, multioutput, labels missing
    #   y_true values, single-entry labels).
    """Test that d2_brier_score raises the appropriate errors
    on invalid inputs."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    with pytest.raises(ValueError, match=error_msg):
        d2_brier_score(y_true, y_pred, labels=labels)


def test_d2_brier_score_warning_on_less_than_two_samples():
    # [FUNCTION DOC]
    # Verifies d2_brier_score warns with UndefinedMetricWarning for a single sample.
    """Test that d2_brier_score emits a warning when there are less than
    two samples"""
    y_true = np.array([1])
    y_pred = np.array([0.8])
    warning_message = "not well-defined with less than two samples"
    with pytest.warns(UndefinedMetricWarning, match=warning_message):
        d2_brier_score(y_true, y_pred)


@pytest.mark.parametrize(
    "array_namespace, device_name, dtype_name",
    yield_namespace_device_dtype_combinations(),
)
def test_confusion_matrix_array_api(array_namespace, device_name, dtype_name):
    # [FUNCTION DOC]
    # Verifies confusion_matrix under array_api_dispatch returns an array in the same
    #   namespace and on the same device as the input for every available
    #   namespace/device/dtype, using labels that force index conversion.
    """Test that `confusion_matrix` works for all array types when `labels` are passed
    such that the inner boolean `need_index_conversion` evaluates to `True`."""
    xp, device = _array_api_for_tests(array_namespace, device_name, dtype_name)

    y_true = xp.asarray([1, 2, 3], device=device)
    y_pred = xp.asarray([4, 5, 6], device=device)
    labels = xp.asarray([1, 2, 3], device=device)

    with config_context(array_api_dispatch=True):
        result = confusion_matrix(y_true, y_pred, labels=labels)
        assert get_namespace(result)[0] == get_namespace(y_pred)[0]
        assert array_api_device(result) == array_api_device(y_pred)
