# [MODULE DOC]
# Tests for SelfTrainingClassifier. Module-level fixtures: an iris train/test
# split; y_train_missing_labels keeps the first 50 training labels and sets the
# rest to -1; y_train_missing_strings is the same with classes mapped to "A",
# "B", "C" in an object array, unlabeled entries kept as integer -1.
from math import ceil

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.datasets import load_iris, make_blobs
from sklearn.ensemble import StackingClassifier
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.svm import SVC
from sklearn.tests.test_pipeline import SimpleEstimator
from sklearn.tree import DecisionTreeClassifier

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

# load the iris dataset and randomly permute it
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, random_state=0
)

n_labeled_samples = 50

y_train_missing_labels = y_train.copy()
y_train_missing_labels[n_labeled_samples:] = -1
mapping = {0: "A", 1: "B", 2: "C", -1: "-1"}
y_train_missing_strings = np.vectorize(mapping.get)(y_train_missing_labels).astype(
    object
)
y_train_missing_strings[y_train_missing_labels == -1] = -1


def test_warns_k_best():
    # [FUNCTION DOC]
    # Verifies a UserWarning when k_best exceeds the number of unlabeled samples and
    # that everything gets labeled in one go (termination "all_labeled").
    st = SelfTrainingClassifier(KNeighborsClassifier(), criterion="k_best", k_best=1000)
    with pytest.warns(UserWarning, match="k_best is larger than"):
        st.fit(X_train, y_train_missing_labels)

    assert st.termination_condition_ == "all_labeled"


@pytest.mark.parametrize(
    "estimator",
    [KNeighborsClassifier(), LogisticRegression()],
)
@pytest.mark.parametrize("selection_crit", ["threshold", "k_best"])
def test_classification(estimator, selection_crit):
    # [FUNCTION DOC]
    # For KNN and LogisticRegression with both criteria, verifies numeric and string
    # labels yield equivalent predictions, probabilities and termination condition;
    # originally labeled samples have labeled_iter_ 0 and keep their labels; max
    # labeled_iter_ <= n_iter_ <= max_iter; labeled_iter_ and transduction_ shapes
    # match.
    estimator = clone(estimator)  # Avoid side effects from previous tests.
    # Check classification for various parameter settings.
    # Also assert that predictions for strings and numerical labels are equal.
    # Also test for multioutput classification
    threshold = 0.75
    max_iter = 10
    st = SelfTrainingClassifier(
        estimator, max_iter=max_iter, threshold=threshold, criterion=selection_crit
    )
    st.fit(X_train, y_train_missing_labels)
    pred = st.predict(X_test)
    proba = st.predict_proba(X_test)

    st_string = SelfTrainingClassifier(
        estimator, max_iter=max_iter, criterion=selection_crit, threshold=threshold
    )
    st_string.fit(X_train, y_train_missing_strings)
    pred_string = st_string.predict(X_test)
    proba_string = st_string.predict_proba(X_test)

    assert_array_equal(np.vectorize(mapping.get)(pred), pred_string)
    assert_array_equal(proba, proba_string)

    assert st.termination_condition_ == st_string.termination_condition_
    # Check consistency between labeled_iter, n_iter and max_iter
    labeled = y_train_missing_labels != -1
    # assert that labeled samples have labeled_iter = 0
    assert_array_equal(st.labeled_iter_ == 0, labeled)
    # assert that labeled samples do not change label during training
    assert_array_equal(y_train_missing_labels[labeled], st.transduction_[labeled])

    # assert that the max of the iterations is less than the total amount of
    # iterations
    assert np.max(st.labeled_iter_) <= st.n_iter_ <= max_iter
    assert np.max(st_string.labeled_iter_) <= st_string.n_iter_ <= max_iter

    # check shapes
    assert st.labeled_iter_.shape == st.transduction_.shape
    assert st_string.labeled_iter_.shape == st_string.transduction_.shape


def test_k_best():
    # [FUNCTION DOC]
    # With 1-NN, k_best=10, max_iter=None and a single labeled sample, verifies the
    # number of iterations is ceil((n-1)/10), that each intermediate iteration
    # labels exactly 10 samples, the last one the remainder, and termination is
    # "all_labeled".
    # KNOWN ISSUE: the remainder assertion assumes (n-1) is not a multiple of 10;
    # otherwise the final iteration labels 10, not 0.
    st = SelfTrainingClassifier(
        KNeighborsClassifier(n_neighbors=1),
        criterion="k_best",
        k_best=10,
        max_iter=None,
    )
    y_train_only_one_label = np.copy(y_train)
    y_train_only_one_label[1:] = -1
    n_samples = y_train.shape[0]

    n_expected_iter = ceil((n_samples - 1) / 10)
    st.fit(X_train, y_train_only_one_label)
    assert st.n_iter_ == n_expected_iter

    # Check labeled_iter_
    assert np.sum(st.labeled_iter_ == 0) == 1
    for i in range(1, n_expected_iter):
        assert np.sum(st.labeled_iter_ == i) == 10
    assert np.sum(st.labeled_iter_ == n_expected_iter) == (n_samples - 1) % 10
    assert st.termination_condition_ == "all_labeled"


def test_sanity_classification():
    # [FUNCTION DOC]
    # Verifies self-training a calibrated SVC on partly labeled data gives different
    # and more accurate test predictions than the same estimator trained only on
    # the samples beyond the first 50.
    estimator = CalibratedClassifierCV(SVC(gamma="scale"), ensemble=False)
    estimator.fit(X_train[n_labeled_samples:], y_train[n_labeled_samples:])

    st = SelfTrainingClassifier(estimator)
    st.fit(X_train, y_train_missing_labels)

    pred1, pred2 = estimator.predict(X_test), st.predict(X_test)
    assert not np.array_equal(pred1, pred2)
    score_supervised = accuracy_score(estimator.predict(X_test), y_test)
    score_self_training = accuracy_score(st.predict(X_test), y_test)

    assert score_self_training > score_supervised


def test_none_iter():
    # [FUNCTION DOC]
    # Verifies with max_iter=None and threshold 0.55 that all samples get labeled in
    # fewer than 10 iterations.
    # Check that the all samples were labeled after a 'reasonable' number of
    # iterations.
    st = SelfTrainingClassifier(KNeighborsClassifier(), threshold=0.55, max_iter=None)
    st.fit(X_train, y_train_missing_labels)

    assert st.n_iter_ < 10
    assert st.termination_condition_ == "all_labeled"


@pytest.mark.parametrize(
    "estimator",
    [
        KNeighborsClassifier(),
        CalibratedClassifierCV(SVC(gamma="scale", random_state=0), ensemble=False),
    ],
)
@pytest.mark.parametrize("y", [y_train_missing_labels, y_train_missing_strings])
def test_zero_iterations(estimator, y):
    # [FUNCTION DOC]
    # Verifies max_iter=0 is equivalent to fitting the base estimator on the labeled
    # subset only (same predictions), for numeric and string labels, and that the
    # termination condition is "max_iter".
    estimator = clone(estimator)  # Avoid side effects from previous tests.
    # Check classification for zero iterations.
    # Fitting a SelfTrainingClassifier with zero iterations should give the
    # same results as fitting a supervised classifier.
    # This also asserts that string arrays work as expected.

    clf1 = SelfTrainingClassifier(estimator, max_iter=0)

    clf1.fit(X_train, y)

    clf2 = estimator.fit(X_train[:n_labeled_samples], y[:n_labeled_samples])

    assert_array_equal(clf1.predict(X_test), clf2.predict(X_test))
    assert clf1.termination_condition_ == "max_iter"


def test_prefitted_throws_error():
    # [FUNCTION DOC]
    # Verifies predict on an unfitted SelfTrainingClassifier raises NotFittedError
    # even if the wrapped estimator was already fitted.
    # Test that passing a pre-fitted classifier and calling predict throws an
    # error
    knn = KNeighborsClassifier()
    knn.fit(X_train, y_train)
    st = SelfTrainingClassifier(knn)
    with pytest.raises(
        NotFittedError,
        match="This SelfTrainingClassifier instance is not fitted yet",
    ):
        st.predict(X_train)


@pytest.mark.parametrize("max_iter", range(1, 5))
def test_labeled_iter(max_iter):
    # [FUNCTION DOC]
    # Verifies for max_iter 1..4 that exactly the 50 originally labeled samples
    # have labeled_iter_ 0 and max labeled_iter_ <= n_iter_ <= max_iter.
    # Check that the amount of datapoints labeled in iteration 0 is equal to
    # the amount of labeled datapoints we passed.
    st = SelfTrainingClassifier(KNeighborsClassifier(), max_iter=max_iter)

    st.fit(X_train, y_train_missing_labels)
    amount_iter_0 = len(st.labeled_iter_[st.labeled_iter_ == 0])
    assert amount_iter_0 == n_labeled_samples
    # Check that the max of the iterations is less than the total amount of
    # iterations
    assert np.max(st.labeled_iter_) <= st.n_iter_ <= max_iter


def test_no_unlabeled():
    # [FUNCTION DOC]
    # Verifies fully labeled y warns "y contains no unlabeled samples", reproduces
    # the plain KNN predictions, sets all labeled_iter_ to 0 and terminates with
    # "all_labeled".
    # Test that training on a fully labeled dataset produces the same results
    # as training the classifier by itself.
    knn = KNeighborsClassifier()
    knn.fit(X_train, y_train)
    st = SelfTrainingClassifier(knn)
    with pytest.warns(UserWarning, match="y contains no unlabeled samples"):
        st.fit(X_train, y_train)
    assert_array_equal(knn.predict(X_test), st.predict(X_test))
    # Assert that all samples were labeled in iteration 0 (since there were no
    # unlabeled samples).
    assert np.all(st.labeled_iter_ == 0)
    assert st.termination_condition_ == "all_labeled"


def test_early_stopping():
    # [FUNCTION DOC]
    # Verifies that when no unlabeled prediction exceeds the threshold the loop
    # stops after one iteration with termination "no_change".
    lr = LogisticRegression()
    st = SelfTrainingClassifier(lr)
    X_train_easy = [[1], [0], [1], [0.5]]
    y_train_easy = [1, 0, -1, -1]
    # X = [[0.5]] cannot be predicted on with a high confidence, so training
    # stops early
    st.fit(X_train_easy, y_train_easy)
    assert st.n_iter_ == 1
    assert st.termination_condition_ == "no_change"


def test_strings_dtype():
    # [FUNCTION DOC]
    # Verifies fitting with a unicode string label array raises ValueError mentioning
    # dtype.
    clf = SelfTrainingClassifier(KNeighborsClassifier())
    X, y = make_blobs(n_samples=30, random_state=0, cluster_std=0.1)
    labels_multiclass = ["one", "two", "three"]

    y_strings = np.take(labels_multiclass, y)

    with pytest.raises(ValueError, match="dtype"):
        clf.fit(X, y_strings)


@pytest.mark.parametrize("verbose", [True, False])
def test_verbose(capsys, verbose):
    # [FUNCTION DOC]
    # Verifies progress lines containing "iteration" are printed to stdout only when
    # verbose is True (captured with capsys).
    clf = SelfTrainingClassifier(KNeighborsClassifier(), verbose=verbose)
    clf.fit(X_train, y_train_missing_labels)

    captured = capsys.readouterr()

    if verbose:
        assert "iteration" in captured.out
    else:
        assert "iteration" not in captured.out


def test_verbose_k_best(capsys):
    # [FUNCTION DOC]
    # Verifies verbose output with k_best=10 lists "added 10 new labels" for each
    # intermediate iteration and the remainder count for the final iteration.
    st = SelfTrainingClassifier(
        KNeighborsClassifier(n_neighbors=1),
        criterion="k_best",
        k_best=10,
        verbose=True,
        max_iter=None,
    )

    y_train_only_one_label = np.copy(y_train)
    y_train_only_one_label[1:] = -1
    n_samples = y_train.shape[0]

    n_expected_iter = ceil((n_samples - 1) / 10)
    st.fit(X_train, y_train_only_one_label)

    captured = capsys.readouterr()

    msg = "End of iteration {}, added {} new labels."
    for i in range(1, n_expected_iter):
        assert msg.format(i, 10) in captured.out

    assert msg.format(n_expected_iter, (n_samples - 1) % 10) in captured.out


def test_k_best_selects_best():
    # [FUNCTION DOC]
    # Verifies one k_best iteration adds exactly the 10 unlabeled samples on which a
    # separately fitted LogisticRegression is most confident.
    # Tests that the labels added by st really are the 10 best labels.
    est = LogisticRegression(random_state=0)
    st = SelfTrainingClassifier(est, criterion="k_best", max_iter=1, k_best=10)
    has_label = y_train_missing_labels != -1
    st.fit(X_train, y_train_missing_labels)

    got_label = ~has_label & (st.transduction_ != -1)

    est.fit(X_train[has_label], y_train_missing_labels[has_label])
    pred = est.predict_proba(X_train[~has_label])
    max_proba = np.max(pred, axis=1)

    most_confident_est = X_train[~has_label][np.argsort(max_proba)[-10:]]
    added_by_st = X_train[np.where(got_label)].tolist()

    for row in most_confident_est.tolist():
        assert row in added_by_st


def test_estimator_meta_estimator():
    # [FUNCTION DOC]
    # Non-regression (#19119): a StackingClassifier exposing predict_proba works as
    # the wrapped estimator; one built from SVCs (no predict_proba) makes fit raise
    # AttributeError.
    # Check that a meta-estimator relying on an estimator implementing
    # `predict_proba` will work even if it does not expose this method before being
    # fitted.
    # Non-regression test for:
    # https://github.com/scikit-learn/scikit-learn/issues/19119

    estimator = StackingClassifier(
        estimators=[
            ("clf_1", LogisticRegression()),
            ("clf_2", LogisticRegression()),
        ],
        final_estimator=LogisticRegression(),
        cv=2,
    )

    assert hasattr(estimator, "predict_proba")
    clf = SelfTrainingClassifier(estimator=estimator)
    clf.fit(X_train, y_train_missing_labels)
    clf.predict_proba(X_test)

    estimator = StackingClassifier(
        estimators=[
            ("svc_1", SVC()),
            ("svc_2", SVC()),
        ],
        final_estimator=SVC(),
        cv=2,
    )

    assert not hasattr(estimator, "predict_proba")
    clf = SelfTrainingClassifier(estimator=estimator)
    with pytest.raises(AttributeError):
        clf.fit(X_train, y_train_missing_labels)


def test_self_training_estimator_attribute_error():
    # [FUNCTION DOC]
    # Non-regression (#28108): fit with an SVC lacking predict_proba raises
    # AttributeError naming predict_proba; decision_function on a wrapped
    # DecisionTreeClassifier raises the meta-estimator AttributeError whose cause is
    # the inner estimator's AttributeError.
    """Check that we raise the proper AttributeErrors when the `estimator`
    does not implement the `predict_proba` method, which is called from within
    `fit`, or `decision_function`, which is decorated with `available_if`.

    Non-regression test for:
    https://github.com/scikit-learn/scikit-learn/issues/28108
    """
    # `SVC` with `probability=False` does not implement 'predict_proba' that
    # is required internally in `fit` of `SelfTrainingClassifier`. We expect
    # an AttributeError to be raised.
    estimator = SVC(gamma="scale")
    self_training = SelfTrainingClassifier(estimator)

    with pytest.raises(AttributeError, match="has no attribute 'predict_proba'"):
        self_training.fit(X_train, y_train_missing_labels)

    # `DecisionTreeClassifier` does not implement 'decision_function' and
    # should raise an AttributeError
    self_training = SelfTrainingClassifier(estimator=DecisionTreeClassifier())

    outer_msg = "This 'SelfTrainingClassifier' has no attribute 'decision_function'"
    inner_msg = "'DecisionTreeClassifier' object has no attribute 'decision_function'"
    with pytest.raises(AttributeError, match=outer_msg) as exec_info:
        self_training.fit(X_train, y_train_missing_labels).decision_function(X_train)
    assert isinstance(exec_info.value.__cause__, AttributeError)
    assert inner_msg in str(exec_info.value.__cause__)


# Metadata routing tests
# =================================================================


@pytest.mark.filterwarnings("ignore:y contains no unlabeled samples:UserWarning")
@pytest.mark.parametrize(
    "method", ["decision_function", "predict_log_proba", "predict_proba", "predict"]
)
def test_routing_passed_metadata_not_supported(method):
    # [FUNCTION DOC]
    # Verifies that with metadata routing disabled, passing metadata to fit or to
    # each prediction method (on an instance marked fitted via fitted_params_)
    # raises ValueError saying it requires enable_metadata_routing=True.
    """Test that the right error message is raised when metadata is passed while
    not supported when `enable_metadata_routing=False`."""
    est = SelfTrainingClassifier(estimator=SimpleEstimator())
    with pytest.raises(
        ValueError, match="is only supported if enable_metadata_routing=True"
    ):
        est.fit([[1], [1]], [1, 1], sample_weight=[1], prop="a")

    est = SelfTrainingClassifier(estimator=SimpleEstimator())
    with pytest.raises(
        ValueError, match="is only supported if enable_metadata_routing=True"
    ):
        # make sure that the estimator thinks it is already fitted
        est.fitted_params_ = True
        getattr(est, method)([[1]], sample_weight=[1], prop="a")


# End of routing tests
# ====================
