# [MODULE DOC]
# Tests for sklearn.multiclass meta-estimators: OneVsRestClassifier (OvR),
# OneVsOneClassifier (OvO) and OutputCodeClassifier (ECOC). Covers fitting/predicting
# with various base estimators, partial_fit, multilabel and sparse targets,
# decision_function/predict_proba consistency, tie breaking, precomputed kernels and
# pairwise handling, and error messages. Import time: loads iris and shuffles it with
# a fixed permutation (seed 0); n_classes = 3.
from re import escape

import numpy as np
import pytest
import scipy.sparse as sp
from numpy.testing import assert_allclose

from sklearn import datasets, svm
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.datasets import load_breast_cancer
from sklearn.exceptions import NotFittedError
from sklearn.impute import SimpleImputer
from sklearn.linear_model import (
    ElasticNet,
    Lasso,
    LinearRegression,
    LogisticRegression,
    Perceptron,
    Ridge,
    SGDClassifier,
)
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.multiclass import (
    OneVsOneClassifier,
    OneVsRestClassifier,
    OutputCodeClassifier,
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils import (
    check_array,
    shuffle,
)
from sklearn.utils._mocking import CheckingClassifier
from sklearn.utils._testing import assert_almost_equal, assert_array_equal
from sklearn.utils.fixes import (
    COO_CONTAINERS,
    CSC_CONTAINERS,
    CSR_CONTAINERS,
    DOK_CONTAINERS,
    LIL_CONTAINERS,
)
from sklearn.utils.multiclass import check_classification_targets, type_of_target

iris = datasets.load_iris()
rng = np.random.RandomState(0)
perm = rng.permutation(iris.target.size)
iris.data = iris.data[perm]
iris.target = iris.target[perm]
n_classes = 3


def test_ovr_exceptions():
    # [FUNCTION DOC]
    # Verifies OvR raises NotFittedError on predict before fit and ValueError for
    # multioutput (non-binary 2D integer or float) targets.
    ovr = OneVsRestClassifier(LinearSVC(random_state=0))

    # test predicting without fitting
    with pytest.raises(NotFittedError):
        ovr.predict([])

    # Fail on multioutput data
    msg = "Multioutput target data is not supported with label binarization"
    with pytest.raises(ValueError, match=msg):
        X = np.array([[1, 0], [0, 1]])
        y = np.array([[1, 2], [3, 1]])
        OneVsRestClassifier(MultinomialNB()).fit(X, y)

    with pytest.raises(ValueError, match=msg):
        X = np.array([[1, 0], [0, 1]])
        y = np.array([[1.5, 2.4], [3.1, 0.8]])
        OneVsRestClassifier(MultinomialNB()).fit(X, y)


def test_check_classification_targets():
    # [FUNCTION DOC]
    # Regression (#5782): check_classification_targets raises a ValueError whose message
    # contains the type_of_target of a continuous target.
    # Test that check_classification_target return correct type. #5782
    y = np.array([0.0, 1.1, 2.0, 3.0])
    msg = type_of_target(y)
    with pytest.raises(ValueError, match=msg):
        check_classification_targets(y)


def test_ovr_ties():
    # [FUNCTION DOC]
    # Non-regression (#14124): when all binary decision values tie (all zeros), OvR predict
    # matches argmax of decision_function, i.e. picks the first class.
    """Check that ties-breaking matches np.argmax behavior

    Non-regression test for issue #14124
    """

    class Dummy(BaseEstimator):
        # [CLASS DOC]
        # HELPER: Estimator with constant zero decision_function.
        def fit(self, X, y):
            # [FUNCTION DOC]
            # HELPER: No-op fit returning self.
            return self

        def decision_function(self, X):
            # [FUNCTION DOC]
            # HELPER: Returns zeros of length n_samples.
            return np.zeros(len(X))

    X = np.array([[0], [0], [0], [0]])
    y = np.array([0, 1, 2, 3])
    clf = OneVsRestClassifier(Dummy()).fit(X, y)
    assert_array_equal(clf.predict(X), np.argmax(clf.decision_function(X), axis=1))


def test_ovr_fit_predict():
    # [FUNCTION DOC]
    # Verifies OvR with LinearSVC builds n_classes estimators and matches the accuracy of
    # LinearSVC alone on iris; with MultinomialNB accuracy exceeds 0.65.
    # A classifier which implements decision_function.
    ovr = OneVsRestClassifier(LinearSVC(random_state=0))
    pred = ovr.fit(iris.data, iris.target).predict(iris.data)
    assert len(ovr.estimators_) == n_classes

    clf = LinearSVC(random_state=0)
    pred2 = clf.fit(iris.data, iris.target).predict(iris.data)
    assert np.mean(iris.target == pred) == np.mean(iris.target == pred2)

    # A classifier which implements predict_proba.
    ovr = OneVsRestClassifier(MultinomialNB())
    pred = ovr.fit(iris.data, iris.target).predict(iris.data)
    assert np.mean(iris.target == pred) > 0.65


def test_ovr_partial_fit():
    # [FUNCTION DOC]
    # Verifies OvR partial_fit over two batches matches full fit predictions with
    # MultinomialNB, works with SGDClassifier when batches miss some classes (equal
    # accuracy to fit), and that partial_fit is unavailable for a base estimator without
    # it (SVC).
    # Side effects: uses the unseeded global NumPy RNG for one toy dataset.
    # Test if partial_fit is working as intended
    X, y = shuffle(iris.data, iris.target, random_state=0)
    ovr = OneVsRestClassifier(MultinomialNB())
    ovr.partial_fit(X[:100], y[:100], np.unique(y))
    ovr.partial_fit(X[100:], y[100:])
    pred = ovr.predict(X)
    ovr2 = OneVsRestClassifier(MultinomialNB())
    pred2 = ovr2.fit(X, y).predict(X)

    assert_almost_equal(pred, pred2)
    assert len(ovr.estimators_) == len(np.unique(y))
    assert np.mean(y == pred) > 0.65

    # Test when mini batches doesn't have all classes
    # with SGDClassifier
    X = np.abs(np.random.randn(14, 2))
    y = [1, 1, 1, 1, 2, 3, 3, 0, 0, 2, 3, 1, 2, 3]

    ovr = OneVsRestClassifier(
        SGDClassifier(max_iter=1, tol=None, shuffle=False, random_state=0)
    )
    ovr.partial_fit(X[:7], y[:7], np.unique(y))
    ovr.partial_fit(X[7:], y[7:])
    pred = ovr.predict(X)
    ovr1 = OneVsRestClassifier(
        SGDClassifier(max_iter=1, tol=None, shuffle=False, random_state=0)
    )
    pred1 = ovr1.fit(X, y).predict(X)
    assert np.mean(pred == y) == np.mean(pred1 == y)

    # test partial_fit only exists if estimator has it:
    ovr = OneVsRestClassifier(SVC())
    assert not hasattr(ovr, "partial_fit")


def test_ovr_partial_fit_exceptions():
    # [FUNCTION DOC]
    # Verifies OvR partial_fit raises ValueError when a later batch contains a class not in
    # the classes declared in the first call.
    # Side effects: uses the unseeded global NumPy RNG.
    ovr = OneVsRestClassifier(MultinomialNB())
    X = np.abs(np.random.randn(14, 2))
    y = [1, 1, 1, 1, 2, 3, 3, 0, 0, 2, 3, 1, 2, 3]
    ovr.partial_fit(X[:7], y[:7], np.unique(y))
    # If a new class that was not in the first call of partial fit is seen
    # it should raise ValueError
    y1 = [5] + y[7:-1]
    msg = r"Mini-batch contains \[.+\] while classes must be subset of \[.+\]"
    with pytest.raises(ValueError, match=msg):
        ovr.partial_fit(X=X[7:], y=y1)


def test_ovr_ovo_regressor():
    # [FUNCTION DOC]
    # Verifies OvR and OvO accept regressors without decision_function (DecisionTreeRegressor):
    # correct number of estimators, predictions limited to classes 0-2, accuracy > 0.9.
    # test that ovr and ovo work on regressors which don't have a decision_
    # function
    ovr = OneVsRestClassifier(DecisionTreeRegressor())
    pred = ovr.fit(iris.data, iris.target).predict(iris.data)
    assert len(ovr.estimators_) == n_classes
    assert_array_equal(np.unique(pred), [0, 1, 2])
    # we are doing something sensible
    assert np.mean(pred == iris.target) > 0.9

    ovr = OneVsOneClassifier(DecisionTreeRegressor())
    pred = ovr.fit(iris.data, iris.target).predict(iris.data)
    assert len(ovr.estimators_) == n_classes * (n_classes - 1) / 2
    assert_array_equal(np.unique(pred), [0, 1, 2])
    # we are doing something sensible
    assert np.mean(pred == iris.target) > 0.9


@pytest.mark.parametrize(
    "sparse_container",
    CSR_CONTAINERS + CSC_CONTAINERS + COO_CONTAINERS + DOK_CONTAINERS + LIL_CONTAINERS,
)
def test_ovr_fit_predict_sparse(sparse_container):
    # [FUNCTION DOC]
    # Verifies multilabel OvR with MultinomialNB gives the same predictions whether Y is
    # dense or the parametrized sparse format (output is then sparse), that predict equals
    # predict_proba > 0.5, and that for SVC predict equals decision_function > 0.
    base_clf = MultinomialNB(alpha=1)

    X, Y = datasets.make_multilabel_classification(
        n_samples=100,
        n_features=20,
        n_classes=5,
        n_labels=3,
        length=50,
        allow_unlabeled=True,
        random_state=0,
    )

    X_train, Y_train = X[:80], Y[:80]
    X_test = X[80:]

    clf = OneVsRestClassifier(base_clf).fit(X_train, Y_train)
    Y_pred = clf.predict(X_test)

    clf_sprs = OneVsRestClassifier(base_clf).fit(X_train, sparse_container(Y_train))
    Y_pred_sprs = clf_sprs.predict(X_test)

    assert clf.multilabel_
    assert sp.issparse(Y_pred_sprs)
    assert_array_equal(Y_pred_sprs.toarray(), Y_pred)

    # Test predict_proba
    Y_proba = clf_sprs.predict_proba(X_test)

    # predict assigns a label if the probability that the
    # sample has the label is greater than 0.5.
    pred = Y_proba > 0.5
    assert_array_equal(pred, Y_pred_sprs.toarray())

    # Test decision_function
    clf = svm.SVC()
    clf_sprs = OneVsRestClassifier(clf).fit(X_train, sparse_container(Y_train))
    dec_pred = (clf_sprs.decision_function(X_test) > 0).astype(int)
    assert_array_equal(dec_pred, clf_sprs.predict(X_test).toarray())


def test_ovr_always_present():
    # [FUNCTION DOC]
    # Verifies OvR handles labels always present (warns, constant predictor gives ones in
    # decision_function/predict_proba) and labels always absent (warns, probability zero).
    # Test that ovr works with classes that are always present or absent.
    # Note: tests is the case where _ConstantPredictor is utilised
    X = np.ones((10, 2))
    X[:5, :] = 0

    # Build an indicator matrix where two features are always on.
    # As list of lists, it would be: [[int(i >= 5), 2, 3] for i in range(10)]
    y = np.zeros((10, 3))
    y[5:, 0] = 1
    y[:, 1] = 1
    y[:, 2] = 1

    ovr = OneVsRestClassifier(LogisticRegression())
    msg = r"Label .+ is present in all training examples"
    with pytest.warns(UserWarning, match=msg):
        ovr.fit(X, y)
    y_pred = ovr.predict(X)
    assert_array_equal(np.array(y_pred), np.array(y))
    y_pred = ovr.decision_function(X)
    assert np.unique(y_pred[:, -2:]) == 1
    y_pred = ovr.predict_proba(X)
    assert_array_equal(y_pred[:, -1], np.ones(X.shape[0]))

    # y has a constantly absent label
    y = np.zeros((10, 2))
    y[5:, 0] = 1  # variable label
    ovr = OneVsRestClassifier(LogisticRegression())

    msg = r"Label not 1 is present in all training examples"
    with pytest.warns(UserWarning, match=msg):
        ovr.fit(X, y)
    y_pred = ovr.predict_proba(X)
    assert_array_equal(y_pred[:, -1], np.zeros(X.shape[0]))


def test_ovr_multiclass():
    # [FUNCTION DOC]
    # Verifies OvR on a toy 3-class string-label problem for several base estimators:
    # classes_ is correct, a query near the "eggs" pattern predicts "eggs", and the same
    # works with an indicator-matrix target.
    # Toy dataset where features correspond directly to labels.
    X = np.array([[0, 0, 5], [0, 5, 0], [3, 0, 0], [0, 0, 6], [6, 0, 0]])
    y = ["eggs", "spam", "ham", "eggs", "ham"]
    Y = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 0, 1], [1, 0, 0]])

    classes = set("ham eggs spam".split())

    for base_clf in (
        MultinomialNB(),
        LinearSVC(random_state=0),
        LinearRegression(),
        Ridge(),
        ElasticNet(),
    ):
        clf = OneVsRestClassifier(base_clf).fit(X, y)
        assert set(clf.classes_) == classes
        y_pred = clf.predict(np.array([[0, 0, 4]]))[0]
        assert_array_equal(y_pred, ["eggs"])

        # test input as label indicator matrix
        clf = OneVsRestClassifier(base_clf).fit(X, Y)
        y_pred = clf.predict([[0, 0, 4]])[0]
        assert_array_equal(y_pred, [0, 0, 1])


def test_ovr_binary():
    # [FUNCTION DOC]
    # Verifies OvR on a binary string-label problem for several base estimators: classes_,
    # prediction, 1D decision_function of shape (5,), predict_proba consistency for
    # probabilistic estimators, and indicator column target.
    # Toy dataset where features correspond directly to labels.
    X = np.array([[0, 0, 5], [0, 5, 0], [3, 0, 0], [0, 0, 6], [6, 0, 0]])
    y = ["eggs", "spam", "spam", "eggs", "spam"]
    Y = np.array([[0, 1, 1, 0, 1]]).T

    classes = set("eggs spam".split())

    def conduct_test(base_clf, test_predict_proba=False):
        # [FUNCTION DOC]
        # HELPER: Runs the binary checks for one base estimator.
        # Inputs: base_clf - estimator; test_predict_proba - bool, default False, also checks
        # predict_proba has 2 columns and argmax matches predict.
        # Exceptions: AssertionError on failure.
        clf = OneVsRestClassifier(base_clf).fit(X, y)
        assert set(clf.classes_) == classes
        y_pred = clf.predict(np.array([[0, 0, 4]]))[0]
        assert_array_equal(y_pred, ["eggs"])
        if hasattr(base_clf, "decision_function"):
            dec = clf.decision_function(X)
            assert dec.shape == (5,)

        if test_predict_proba:
            X_test = np.array([[0, 0, 4]])
            probabilities = clf.predict_proba(X_test)
            assert 2 == len(probabilities[0])
            assert clf.classes_[np.argmax(probabilities, axis=1)] == clf.predict(X_test)

        # test input as label indicator matrix
        clf = OneVsRestClassifier(base_clf).fit(X, Y)
        y_pred = clf.predict([[3, 0, 0]])[0]
        assert y_pred == 1

    for base_clf in (
        LinearSVC(random_state=0),
        LinearRegression(),
        Ridge(),
        ElasticNet(),
    ):
        conduct_test(base_clf)

    for base_clf in (
        MultinomialNB(),
        LogisticRegression(),
    ):
        conduct_test(base_clf, test_predict_proba=True)


def test_ovr_multilabel():
    # [FUNCTION DOC]
    # Verifies OvR on a toy multilabel indicator problem predicts [0, 1, 1] for a query and
    # sets multilabel_ for several base estimators.
    # Toy dataset where features correspond directly to labels.
    X = np.array([[0, 4, 5], [0, 5, 0], [3, 3, 3], [4, 0, 6], [6, 0, 0]])
    y = np.array([[0, 1, 1], [0, 1, 0], [1, 1, 1], [1, 0, 1], [1, 0, 0]])

    for base_clf in (
        MultinomialNB(),
        LinearSVC(random_state=0),
        LinearRegression(),
        Ridge(),
        ElasticNet(),
        Lasso(alpha=0.5),
    ):
        clf = OneVsRestClassifier(base_clf).fit(X, y)
        y_pred = clf.predict([[0, 4, 4]])[0]
        assert_array_equal(y_pred, [0, 1, 1])
        assert clf.multilabel_


def test_ovr_fit_predict_svc():
    # [FUNCTION DOC]
    # Verifies OvR with SVC on iris has 3 estimators and training accuracy > 0.9.
    ovr = OneVsRestClassifier(svm.SVC())
    ovr.fit(iris.data, iris.target)
    assert len(ovr.estimators_) == 3
    assert ovr.score(iris.data, iris.target) > 0.9


def test_ovr_multilabel_dataset():
    # [FUNCTION DOC]
    # Verifies multilabel OvR with MultinomialNB reaches fixed micro precision/recall (to 2
    # decimals) with and without unlabeled samples.
    base_clf = MultinomialNB(alpha=1)
    for au, prec, recall in zip((True, False), (0.51, 0.66), (0.51, 0.80)):
        X, Y = datasets.make_multilabel_classification(
            n_samples=100,
            n_features=20,
            n_classes=5,
            n_labels=2,
            length=50,
            allow_unlabeled=au,
            random_state=0,
        )
        X_train, Y_train = X[:80], Y[:80]
        X_test, Y_test = X[80:], Y[80:]
        clf = OneVsRestClassifier(base_clf).fit(X_train, Y_train)
        Y_pred = clf.predict(X_test)

        assert clf.multilabel_
        assert_almost_equal(
            precision_score(Y_test, Y_pred, average="micro"), prec, decimal=2
        )
        assert_almost_equal(
            recall_score(Y_test, Y_pred, average="micro"), recall, decimal=2
        )


def test_ovr_multilabel_predict_proba():
    # [FUNCTION DOC]
    # Verifies multilabel OvR predict equals predict_proba > 0.5, and that predict_proba is
    # unavailable when the base estimator lacks it (SVR, SVC without probability), before
    # and after fit.
    base_clf = MultinomialNB(alpha=1)
    for au in (False, True):
        X, Y = datasets.make_multilabel_classification(
            n_samples=100,
            n_features=20,
            n_classes=5,
            n_labels=3,
            length=50,
            allow_unlabeled=au,
            random_state=0,
        )
        X_train, Y_train = X[:80], Y[:80]
        X_test = X[80:]
        clf = OneVsRestClassifier(base_clf).fit(X_train, Y_train)

        # Decision function only estimator.
        decision_only = OneVsRestClassifier(svm.SVR()).fit(X_train, Y_train)
        assert not hasattr(decision_only, "predict_proba")

        # Estimator with predict_proba disabled, depending on parameters.
        decision_only = OneVsRestClassifier(svm.SVC())
        assert not hasattr(decision_only, "predict_proba")
        decision_only.fit(X_train, Y_train)
        assert not hasattr(decision_only, "predict_proba")
        assert hasattr(decision_only, "decision_function")

        Y_pred = clf.predict(X_test)
        Y_proba = clf.predict_proba(X_test)

        # predict assigns a label if the probability that the
        # sample has the label is greater than 0.5.
        pred = Y_proba > 0.5
        assert_array_equal(pred, Y_pred)


def test_ovr_single_label_predict_proba():
    # [FUNCTION DOC]
    # Verifies single-label OvR predict_proba rows sum to 1 and argmax matches predict;
    # predict_proba is unavailable for SVR.
    base_clf = MultinomialNB(alpha=1)
    X, Y = iris.data, iris.target
    X_train, Y_train = X[:80], Y[:80]
    X_test = X[80:]
    clf = OneVsRestClassifier(base_clf).fit(X_train, Y_train)

    # Decision function only estimator.
    decision_only = OneVsRestClassifier(svm.SVR()).fit(X_train, Y_train)
    assert not hasattr(decision_only, "predict_proba")

    Y_pred = clf.predict(X_test)
    Y_proba = clf.predict_proba(X_test)

    assert_almost_equal(Y_proba.sum(axis=1), 1.0)
    # predict assigns a label if the probability that the
    # sample has the label with the greatest predictive probability.
    pred = Y_proba.argmax(axis=1)
    assert not (pred - Y_pred).any()


def test_ovr_single_label_predict_proba_zero():
    # [FUNCTION DOC]
    # Verifies OvR predict_proba returns all zeros (no normalization by a zero sum) when
    # each binary estimator always gives the positive class probability 0.
    # KNOWN ISSUE: docstring typo "predic_proba".
    """Check that predic_proba returns all zeros when the base estimator
    never predicts the positive class.
    """

    class NaiveBinaryClassifier(BaseEstimator, ClassifierMixin):
        # [CLASS DOC]
        # HELPER: Binary classifier whose predict_proba always assigns probability 1 to the
        # negative class and 0 to the positive.
        def fit(self, X, y):
            # [FUNCTION DOC]
            # HELPER: Stores classes_ as the unique labels; returns self.
            self.classes_ = np.unique(y)
            return self

        def predict_proba(self, X):
            # [FUNCTION DOC]
            # HELPER: Returns an (n_samples, 2) array with column 0 ones and column 1 zeros.
            proba = np.ones((len(X), 2))
            # Probability of being the positive class is always 0
            proba[:, 1] = 0
            return proba

    base_clf = NaiveBinaryClassifier()
    X, y = iris.data, iris.target  # Three-class problem with 150 samples

    clf = OneVsRestClassifier(base_clf).fit(X, y)
    y_proba = clf.predict_proba(X)

    assert_allclose(y_proba, 0.0)


def test_ovr_multilabel_decision_function():
    # [FUNCTION DOC]
    # Verifies multilabel OvR with SVC: predict equals decision_function > 0.
    X, Y = datasets.make_multilabel_classification(
        n_samples=100,
        n_features=20,
        n_classes=5,
        n_labels=3,
        length=50,
        allow_unlabeled=True,
        random_state=0,
    )
    X_train, Y_train = X[:80], Y[:80]
    X_test = X[80:]
    clf = OneVsRestClassifier(svm.SVC()).fit(X_train, Y_train)
    assert_array_equal(
        (clf.decision_function(X_test) > 0).astype(int), clf.predict(X_test)
    )


def test_ovr_single_label_decision_function():
    # [FUNCTION DOC]
    # Verifies binary OvR with SVC: predict equals decision_function > 0.
    X, Y = datasets.make_classification(n_samples=100, n_features=20, random_state=0)
    X_train, Y_train = X[:80], Y[:80]
    X_test = X[80:]
    clf = OneVsRestClassifier(svm.SVC()).fit(X_train, Y_train)
    assert_array_equal(clf.decision_function(X_test).ravel() > 0, clf.predict(X_test))


def test_ovr_gridsearch():
    # [FUNCTION DOC]
    # Verifies GridSearchCV over estimator__C works with OvR and the chosen C is from the
    # grid.
    ovr = OneVsRestClassifier(LinearSVC(random_state=0))
    Cs = [0.1, 0.5, 0.8]
    cv = GridSearchCV(ovr, {"estimator__C": Cs})
    cv.fit(iris.data, iris.target)
    best_C = cv.best_estimator_.estimators_[0].C
    assert best_C in Cs


def test_ovr_pipeline():
    # [FUNCTION DOC]
    # Verifies OvR wrapping a one-step Pipeline gives the same predictions as wrapping the
    # decision tree directly.
    # Test with pipeline of length one
    # This test is needed because the multiclass estimators may fail to detect
    # the presence of predict_proba or decision_function.
    clf = Pipeline([("tree", DecisionTreeClassifier())])
    ovr_pipe = OneVsRestClassifier(clf)
    ovr_pipe.fit(iris.data, iris.target)
    ovr = OneVsRestClassifier(DecisionTreeClassifier())
    ovr.fit(iris.data, iris.target)
    assert_array_equal(ovr.predict(iris.data), ovr_pipe.predict(iris.data))


def test_ovo_exceptions():
    # [FUNCTION DOC]
    # Verifies OvO predict before fit raises NotFittedError.
    ovo = OneVsOneClassifier(LinearSVC(random_state=0))
    with pytest.raises(NotFittedError):
        ovo.predict([])


def test_ovo_fit_on_list():
    # [FUNCTION DOC]
    # Verifies OvO fit/predict on Python lists gives the same predictions as on arrays.
    # Test that OneVsOne fitting works with a list of targets and yields the
    # same output as predict from an array
    ovo = OneVsOneClassifier(LinearSVC(random_state=0))
    prediction_from_array = ovo.fit(iris.data, iris.target).predict(iris.data)
    iris_data_list = [list(a) for a in iris.data]
    prediction_from_list = ovo.fit(iris_data_list, list(iris.target)).predict(
        iris_data_list
    )
    assert_array_equal(prediction_from_array, prediction_from_list)


def test_ovo_fit_predict():
    # [FUNCTION DOC]
    # Verifies OvO builds n_classes*(n_classes-1)/2 estimators with LinearSVC and with
    # MultinomialNB.
    # A classifier which implements decision_function.
    ovo = OneVsOneClassifier(LinearSVC(random_state=0))
    ovo.fit(iris.data, iris.target).predict(iris.data)
    assert len(ovo.estimators_) == n_classes * (n_classes - 1) / 2

    # A classifier which implements predict_proba.
    ovo = OneVsOneClassifier(MultinomialNB())
    ovo.fit(iris.data, iris.target).predict(iris.data)
    assert len(ovo.estimators_) == n_classes * (n_classes - 1) / 2


def test_ovo_partial_fit_predict():
    # [FUNCTION DOC]
    # Verifies OvO partial_fit: two batches match full fit; batches with only some classes
    # still match; a 5-class toy problem matches; a batch with a class not in the declared
    # classes raises ValueError; partial_fit is absent for SVC.
    # Side effects: uses the unseeded global NumPy RNG.
    # KNOWN ISSUE: after the second scenario, estimators_ length is compared to the number
    # of classes, which equals the pair count only because there are 3 classes.
    temp = datasets.load_iris()
    X, y = temp.data, temp.target
    ovo1 = OneVsOneClassifier(MultinomialNB())
    ovo1.partial_fit(X[:100], y[:100], np.unique(y))
    ovo1.partial_fit(X[100:], y[100:])
    pred1 = ovo1.predict(X)

    ovo2 = OneVsOneClassifier(MultinomialNB())
    ovo2.fit(X, y)
    pred2 = ovo2.predict(X)
    assert len(ovo1.estimators_) == n_classes * (n_classes - 1) / 2
    assert np.mean(y == pred1) > 0.65
    assert_almost_equal(pred1, pred2)

    # Test when mini-batches have binary target classes
    ovo1 = OneVsOneClassifier(MultinomialNB())
    ovo1.partial_fit(X[:60], y[:60], np.unique(y))
    ovo1.partial_fit(X[60:], y[60:])
    pred1 = ovo1.predict(X)
    ovo2 = OneVsOneClassifier(MultinomialNB())
    pred2 = ovo2.fit(X, y).predict(X)

    assert_almost_equal(pred1, pred2)
    assert len(ovo1.estimators_) == len(np.unique(y))
    assert np.mean(y == pred1) > 0.65

    ovo = OneVsOneClassifier(MultinomialNB())
    X = np.random.rand(14, 2)
    y = [1, 1, 2, 3, 3, 0, 0, 4, 4, 4, 4, 4, 2, 2]
    ovo.partial_fit(X[:7], y[:7], [0, 1, 2, 3, 4])
    ovo.partial_fit(X[7:], y[7:])
    pred = ovo.predict(X)
    ovo2 = OneVsOneClassifier(MultinomialNB())
    pred2 = ovo2.fit(X, y).predict(X)
    assert_almost_equal(pred, pred2)

    # raises error when mini-batch does not have classes from all_classes
    ovo = OneVsOneClassifier(MultinomialNB())
    error_y = [0, 1, 2, 3, 4, 5, 2]
    message_re = escape(
        "Mini-batch contains {0} while it must be subset of {1}".format(
            np.unique(error_y), np.unique(y)
        )
    )
    with pytest.raises(ValueError, match=message_re):
        ovo.partial_fit(X[:7], error_y, np.unique(y))

    # test partial_fit only exists if estimator has it:
    ovr = OneVsOneClassifier(SVC())
    assert not hasattr(ovr, "partial_fit")


def test_ovo_decision_function():
    # [FUNCTION DOC]
    # Verifies OvO decision_function: 1D for binary, (n_samples, n_classes) for multiclass
    # with argmax equal to predict; rounding the decision gives the pairwise vote counts
    # recomputed from estimators_; votes take only values 0-2 while decision values are
    # nearly all distinct (> 146 unique per class).
    n_samples = iris.data.shape[0]

    ovo_clf = OneVsOneClassifier(LinearSVC(random_state=0))
    # first binary
    ovo_clf.fit(iris.data, iris.target == 0)
    decisions = ovo_clf.decision_function(iris.data)
    assert decisions.shape == (n_samples,)

    # then multi-class
    ovo_clf.fit(iris.data, iris.target)
    decisions = ovo_clf.decision_function(iris.data)

    assert decisions.shape == (n_samples, n_classes)
    assert_array_equal(decisions.argmax(axis=1), ovo_clf.predict(iris.data))

    # Compute the votes
    votes = np.zeros((n_samples, n_classes))

    k = 0
    for i in range(n_classes):
        for j in range(i + 1, n_classes):
            pred = ovo_clf.estimators_[k].predict(iris.data)
            votes[pred == 0, i] += 1
            votes[pred == 1, j] += 1
            k += 1

    # Extract votes and verify
    assert_array_equal(votes, np.round(decisions))

    for class_idx in range(n_classes):
        # For each sample and each class, there only 3 possible vote levels
        # because they are only 3 distinct class pairs thus 3 distinct
        # binary classifiers.
        # Therefore, sorting predictions based on votes would yield
        # mostly tied predictions:
        assert set(votes[:, class_idx]).issubset(set([0.0, 1.0, 2.0]))

        # The OVO decision function on the other hand is able to resolve
        # most of the ties on this data as it combines both the vote counts
        # and the aggregated confidence levels of the binary classifiers
        # to compute the aggregate decision function. The iris dataset
        # has 150 samples with a couple of duplicates. The OvO decisions
        # can resolve most of the ties:
        assert len(np.unique(decisions[:, class_idx])) > 146


def test_ovo_gridsearch():
    # [FUNCTION DOC]
    # Verifies GridSearchCV over estimator__C works with OvO.
    ovo = OneVsOneClassifier(LinearSVC(random_state=0))
    Cs = [0.1, 0.5, 0.8]
    cv = GridSearchCV(ovo, {"estimator__C": Cs})
    cv.fit(iris.data, iris.target)
    best_C = cv.best_estimator_.estimators_[0].C
    assert best_C in Cs


def test_ovo_ties():
    # [FUNCTION DOC]
    # Verifies OvO tie breaking uses normalized confidences: for a sample with one vote per
    # class the prediction is the argmax of the confidence part, others follow votes.
    # Test that ties are broken using the decision function,
    # not defaulting to the smallest label
    X = np.array([[1, 2], [2, 1], [-2, 1], [-2, -1]])
    y = np.array([2, 0, 1, 2])
    multi_clf = OneVsOneClassifier(Perceptron(shuffle=False, max_iter=4, tol=None))
    ovo_prediction = multi_clf.fit(X, y).predict(X)
    ovo_decision = multi_clf.decision_function(X)

    # Classifiers are in order 0-1, 0-2, 1-2
    # Use decision_function to compute the votes and the normalized
    # sum_of_confidences, which is used to disambiguate when there is a tie in
    # votes.
    votes = np.round(ovo_decision)
    normalized_confidences = ovo_decision - votes

    # For the first point, there is one vote per class
    assert_array_equal(votes[0, :], 1)
    # For the rest, there is no tie and the prediction is the argmax
    assert_array_equal(np.argmax(votes[1:], axis=1), ovo_prediction[1:])
    # For the tie, the prediction is the class with the highest score
    assert ovo_prediction[0] == normalized_confidences[0].argmax()


def test_ovo_ties2():
    # [FUNCTION DOC]
    # Verifies tie breaking is not biased to the first labels: cyclically relabeling the
    # toy problem makes each label win the tied sample once.
    # test that ties can not only be won by the first two labels
    X = np.array([[1, 2], [2, 1], [-2, 1], [-2, -1]])
    y_ref = np.array([2, 0, 1, 2])

    # cycle through labels so that each label wins once
    for i in range(3):
        y = (y_ref + i) % 3
        multi_clf = OneVsOneClassifier(Perceptron(shuffle=False, max_iter=4, tol=None))
        ovo_prediction = multi_clf.fit(X, y).predict(X)
        assert ovo_prediction[0] == i % 3


def test_ovo_string_y():
    # [FUNCTION DOC]
    # Verifies OvO preserves string labels on a 4-class identity dataset.
    # Test that the OvO doesn't mess up the encoding of string labels
    X = np.eye(4)
    y = np.array(["a", "b", "c", "d"])

    ovo = OneVsOneClassifier(LinearSVC())
    ovo.fit(X, y)
    assert_array_equal(y, ovo.predict(X))


def test_ovo_one_class():
    # [FUNCTION DOC]
    # Verifies OvO fit raises ValueError for a single-class target.
    # Test error for OvO with one class
    X = np.eye(4)
    y = np.array(["a"] * 4)

    ovo = OneVsOneClassifier(LinearSVC())
    msg = "when only one class"
    with pytest.raises(ValueError, match=msg):
        ovo.fit(X, y)


def test_ovo_float_y():
    # [FUNCTION DOC]
    # Verifies OvO fit raises "Unknown label type" ValueError for continuous targets.
    # Test that the OvO errors on float targets
    X = iris.data
    y = iris.data[:, 0]

    ovo = OneVsOneClassifier(LinearSVC())
    msg = "Unknown label type"
    with pytest.raises(ValueError, match=msg):
        ovo.fit(X, y)


def test_ecoc_exceptions():
    # [FUNCTION DOC]
    # Verifies ECOC predict before fit raises NotFittedError.
    ecoc = OutputCodeClassifier(LinearSVC(random_state=0))
    with pytest.raises(NotFittedError):
        ecoc.predict([])


def test_ecoc_fit_predict():
    # [FUNCTION DOC]
    # Verifies ECOC with code_size=2 builds 2*n_classes estimators for LinearSVC and
    # MultinomialNB.
    # A classifier which implements decision_function.
    ecoc = OutputCodeClassifier(LinearSVC(random_state=0), code_size=2, random_state=0)
    ecoc.fit(iris.data, iris.target).predict(iris.data)
    assert len(ecoc.estimators_) == n_classes * 2

    # A classifier which implements predict_proba.
    ecoc = OutputCodeClassifier(MultinomialNB(), code_size=2, random_state=0)
    ecoc.fit(iris.data, iris.target).predict(iris.data)
    assert len(ecoc.estimators_) == n_classes * 2


def test_ecoc_gridsearch():
    # [FUNCTION DOC]
    # Verifies GridSearchCV over estimator__C works with ECOC.
    ecoc = OutputCodeClassifier(LinearSVC(random_state=0), random_state=0)
    Cs = [0.1, 0.5, 0.8]
    cv = GridSearchCV(ecoc, {"estimator__C": Cs})
    cv.fit(iris.data, iris.target)
    best_C = cv.best_estimator_.estimators_[0].C
    assert best_C in Cs


def test_ecoc_float_y():
    # [FUNCTION DOC]
    # Verifies ECOC fit raises "Unknown label type" ValueError for continuous targets.
    # Test that the OCC errors on float targets
    X = iris.data
    y = iris.data[:, 0]

    ovo = OutputCodeClassifier(LinearSVC())
    msg = "Unknown label type"
    with pytest.raises(ValueError, match=msg):
        ovo.fit(X, y)


@pytest.mark.parametrize("csc_container", CSC_CONTAINERS)
def test_ecoc_delegate_sparse_base_estimator(csc_container):
    # [FUNCTION DOC]
    # Non-regression (#17218): ECOC passes sparse input through to the base estimator, so a
    # base estimator rejecting sparse raises TypeError in fit and predict; with LinearSVC
    # sparse fit/predict works and 4 estimators are built (default code_size 1.5 * 3).
    # Non-regression test for
    # https://github.com/scikit-learn/scikit-learn/issues/17218
    X, y = iris.data, iris.target
    X_sp = csc_container(X)

    # create an estimator that does not support sparse input
    base_estimator = CheckingClassifier(
        check_X=check_array,
        check_X_params={"ensure_2d": True, "accept_sparse": False},
    )
    ecoc = OutputCodeClassifier(base_estimator, random_state=0)

    with pytest.raises(TypeError, match="Sparse data was passed"):
        ecoc.fit(X_sp, y)

    ecoc.fit(X, y)
    with pytest.raises(TypeError, match="Sparse data was passed"):
        ecoc.predict(X_sp)

    # smoke test to check when sparse input should be supported
    ecoc = OutputCodeClassifier(LinearSVC(random_state=0))
    ecoc.fit(X_sp, y).predict(X_sp)
    assert len(ecoc.estimators_) == 4


def test_pairwise_indices():
    # [FUNCTION DOC]
    # Verifies OvO with a precomputed-kernel SVC stores pairwise_indices_, each covering
    # the samples of two of the three classes (size * n_est / (n_est - 1) equals n_samples).
    clf_precomputed = svm.SVC(kernel="precomputed")
    X, y = iris.data, iris.target

    ovr_false = OneVsOneClassifier(clf_precomputed)
    linear_kernel = np.dot(X, X.T)
    ovr_false.fit(linear_kernel, y)

    n_estimators = len(ovr_false.estimators_)
    precomputed_indices = ovr_false.pairwise_indices_

    for idx in precomputed_indices:
        assert (
            idx.shape[0] * n_estimators / (n_estimators - 1) == linear_kernel.shape[0]
        )


def test_pairwise_n_features_in():
    # [FUNCTION DOC]
    # Verifies n_features_in_ of meta and base estimators for OvR/OvO on the design matrix
    # (4) and a precomputed kernel (149 for meta and OvR bases; 99 or 100 for OvO bases,
    # since OvO subsets kernel rows and columns per class pair).
    # KNOWN ISSUE: in the OvO precomputed block, n_classes_ and estimator count are checked
    # on ovr_precomputed instead of ovo_precomputed.
    """Check the n_features_in_ attributes of the meta and base estimators

    When the training data is a regular design matrix, everything is intuitive.
    However, when the training data is a precomputed kernel matrix, the
    multiclass strategy can resample the kernel matrix of the underlying base
    estimator both row-wise and column-wise and this has a non-trivial impact
    on the expected value for the n_features_in_ of both the meta and the base
    estimators.
    """
    X, y = iris.data, iris.target

    # Remove the last sample to make the classes not exactly balanced and make
    # the test more interesting.
    assert y[-1] == 0
    X = X[:-1]
    y = y[:-1]

    # Fitting directly on the design matrix:
    assert X.shape == (149, 4)

    clf_notprecomputed = svm.SVC(kernel="linear").fit(X, y)
    assert clf_notprecomputed.n_features_in_ == 4

    ovr_notprecomputed = OneVsRestClassifier(clf_notprecomputed).fit(X, y)
    assert ovr_notprecomputed.n_features_in_ == 4
    for est in ovr_notprecomputed.estimators_:
        assert est.n_features_in_ == 4

    ovo_notprecomputed = OneVsOneClassifier(clf_notprecomputed).fit(X, y)
    assert ovo_notprecomputed.n_features_in_ == 4
    assert ovo_notprecomputed.n_classes_ == 3
    assert len(ovo_notprecomputed.estimators_) == 3
    for est in ovo_notprecomputed.estimators_:
        assert est.n_features_in_ == 4

    # When working with precomputed kernels we have one "feature" per training
    # sample:
    K = X @ X.T
    assert K.shape == (149, 149)

    clf_precomputed = svm.SVC(kernel="precomputed").fit(K, y)
    assert clf_precomputed.n_features_in_ == 149

    ovr_precomputed = OneVsRestClassifier(clf_precomputed).fit(K, y)
    assert ovr_precomputed.n_features_in_ == 149
    assert ovr_precomputed.n_classes_ == 3
    assert len(ovr_precomputed.estimators_) == 3
    for est in ovr_precomputed.estimators_:
        assert est.n_features_in_ == 149

    # This becomes really interesting with OvO and precomputed kernel together:
    # internally, OvO will drop the samples of the classes not part of the pair
    # of classes under consideration for a given binary classifier. Since we
    # use a precomputed kernel, it will also drop the matching columns of the
    # kernel matrix, and therefore we have fewer "features" as result.
    #
    # Since class 0 has 49 samples, and class 1 and 2 have 50 samples each, a
    # single OvO binary classifier works with a sub-kernel matrix of shape
    # either (99, 99) or (100, 100).
    ovo_precomputed = OneVsOneClassifier(clf_precomputed).fit(K, y)
    assert ovo_precomputed.n_features_in_ == 149
    assert ovr_precomputed.n_classes_ == 3
    assert len(ovr_precomputed.estimators_) == 3
    assert ovo_precomputed.estimators_[0].n_features_in_ == 99  # class 0 vs class 1
    assert ovo_precomputed.estimators_[1].n_features_in_ == 99  # class 0 vs class 2
    assert ovo_precomputed.estimators_[2].n_features_in_ == 100  # class 1 vs class 2


@pytest.mark.parametrize(
    "MultiClassClassifier", [OneVsRestClassifier, OneVsOneClassifier]
)
def test_pairwise_tag(MultiClassClassifier):
    # [FUNCTION DOC]
    # Verifies the pairwise input tag of OvR/OvO mirrors whether the base SVC uses a
    # precomputed kernel.
    clf_precomputed = svm.SVC(kernel="precomputed")
    clf_notprecomputed = svm.SVC()

    ovr_false = MultiClassClassifier(clf_notprecomputed)
    assert not ovr_false.__sklearn_tags__().input_tags.pairwise

    ovr_true = MultiClassClassifier(clf_precomputed)
    assert ovr_true.__sklearn_tags__().input_tags.pairwise


@pytest.mark.parametrize(
    "MultiClassClassifier", [OneVsRestClassifier, OneVsOneClassifier]
)
def test_pairwise_cross_val_score(MultiClassClassifier):
    # [FUNCTION DOC]
    # Verifies cross_val_score of OvR/OvO is identical with a linear-kernel SVC on X and a
    # precomputed-kernel SVC on the linear Gram matrix (requires correct pairwise
    # subsetting in CV).
    clf_precomputed = svm.SVC(kernel="precomputed")
    clf_notprecomputed = svm.SVC(kernel="linear")

    X, y = iris.data, iris.target

    multiclass_clf_notprecomputed = MultiClassClassifier(clf_notprecomputed)
    multiclass_clf_precomputed = MultiClassClassifier(clf_precomputed)

    linear_kernel = np.dot(X, X.T)
    score_not_precomputed = cross_val_score(
        multiclass_clf_notprecomputed, X, y, error_score="raise"
    )
    score_precomputed = cross_val_score(
        multiclass_clf_precomputed, linear_kernel, y, error_score="raise"
    )
    assert_array_equal(score_precomputed, score_not_precomputed)


@pytest.mark.parametrize(
    "MultiClassClassifier", [OneVsRestClassifier, OneVsOneClassifier]
)
# FIXME: we should move this test in `estimator_checks` once we are able
# to construct meta-estimator instances
def test_support_missing_values(MultiClassClassifier):
    # [FUNCTION DOC]
    # Smoke test: OvR/OvO wrapping an imputer+LogisticRegression pipeline fits and scores
    # on iris with ~10% NaNs (seed 42), i.e. missing-value validation is delegated.
    # smoke test to check that pipeline OvR and OvO classifiers are letting
    # the validation of missing values to
    # the underlying pipeline or classifiers
    rng = np.random.RandomState(42)
    X, y = iris.data, iris.target
    X = np.copy(X)  # Copy to avoid that the original data is modified
    mask = rng.choice([1, 0], X.shape, p=[0.1, 0.9]).astype(bool)
    X[mask] = np.nan
    lr = make_pipeline(SimpleImputer(), LogisticRegression(random_state=rng))

    MultiClassClassifier(lr).fit(X, y).score(X, y)


@pytest.mark.parametrize("make_y", [np.ones, np.zeros])
def test_constant_int_target(make_y):
    # [FUNCTION DOC]
    # Non-regression (#21869): OvR on a constant integer column target (all ones or all
    # zeros) fits without error and predict_proba gives probability 1 for column 0.
    """Check that constant y target does not raise.

    Non-regression test for #21869
    """
    X = np.ones((10, 2))
    y = make_y((10, 1), dtype=np.int32)
    ovr = OneVsRestClassifier(LogisticRegression())

    ovr.fit(X, y)
    y_pred = ovr.predict_proba(X)
    expected = np.zeros((X.shape[0], 2))
    expected[:, 0] = 1
    assert_allclose(y_pred, expected)


def test_ovo_consistent_binary_classification():
    # [FUNCTION DOC]
    # Non-regression (#13617): OvO on a binary problem (breast cancer) with a
    # distance-weighted KNN predicts identically to the KNN alone.
    """Check that ovo is consistent with binary classifier.

    Non-regression test for #13617.
    """
    X, y = load_breast_cancer(return_X_y=True)

    clf = KNeighborsClassifier(n_neighbors=8, weights="distance")
    ovo = OneVsOneClassifier(clf)

    clf.fit(X, y)
    ovo.fit(X, y)

    assert_array_equal(clf.predict(X), ovo.predict(X))


def test_multiclass_estimator_attribute_error():
    # [FUNCTION DOC]
    # Non-regression (#28108): calling partial_fit on OvR with a base estimator lacking it
    # raises AttributeError naming OneVsRestClassifier, chained from an inner
    # AttributeError naming LogisticRegression.
    """Check that we raise the proper AttributeError when the final estimator
    does not implement the `partial_fit` method, which is decorated with
    `available_if`.

    Non-regression test for:
    https://github.com/scikit-learn/scikit-learn/issues/28108
    """
    iris = datasets.load_iris()

    # LogisticRegression does not implement 'partial_fit' and should raise an
    # AttributeError
    clf = OneVsRestClassifier(estimator=LogisticRegression(random_state=42))

    outer_msg = "This 'OneVsRestClassifier' has no attribute 'partial_fit'"
    inner_msg = "'LogisticRegression' object has no attribute 'partial_fit'"
    with pytest.raises(AttributeError, match=outer_msg) as exec_info:
        clf.partial_fit(iris.data, iris.target)
    assert isinstance(exec_info.value.__cause__, AttributeError)
    assert inner_msg in str(exec_info.value.__cause__)
