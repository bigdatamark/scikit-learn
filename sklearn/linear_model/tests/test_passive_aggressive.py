# [MODULE DOC]
# Tests for the deprecated PassiveAggressiveClassifier and PassiveAggressiveRegressor.
# At import time loads iris and shuffles it with RandomState(12) into module-level X and
# y. Contains MyPassiveAggressive, a reference implementation of the PA-I/PA-II
# updates, plus tests for accuracy, partial_fit, refitting, class weights, deprecation
# warnings and equivalence with SGDClassifier/SGDRegressor using the pa1/pa2 learning
# rates. Most tests silence the FutureWarning raised by the deprecated classes.
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.sparse import issparse

from sklearn.base import ClassifierMixin
from sklearn.datasets import load_iris, make_classification, make_regression
from sklearn.linear_model import (
    PassiveAggressiveClassifier,
    PassiveAggressiveRegressor,
    SGDClassifier,
    SGDRegressor,
)
from sklearn.linear_model._base import SPARSE_INTERCEPT_DECAY
from sklearn.linear_model._stochastic_gradient import DEFAULT_EPSILON
from sklearn.utils import check_random_state
from sklearn.utils._testing import (
    assert_almost_equal,
    assert_array_equal,
)
from sklearn.utils.fixes import CSR_CONTAINERS

iris = load_iris()
random_state = check_random_state(12)
indices = np.arange(iris.data.shape[0])
random_state.shuffle(indices)
X = iris.data[indices]
y = iris.target[indices]


# TODO(1.10): Move to test_sgd.py
class MyPassiveAggressive(ClassifierMixin):
    # [CLASS DOC]
    # HELPER: Reference single-output passive-aggressive learner used to check coef_ of
    #   the
    # library estimators. Supports losses hinge, squared_hinge (classification, labels
    #   in
    # {-1, +1}) and epsilon_insensitive, squared_epsilon_insensitive (regression).
    def __init__(
        self,
        C=1.0,
        epsilon=DEFAULT_EPSILON,
        loss="hinge",
        fit_intercept=True,
        n_iter=1,
        random_state=None,
    ):
        # [FUNCTION DOC]
        # HELPER: Stores hyperparameters. Inputs: C - aggressiveness cap (float, default
        #   1.0);
        # epsilon - insensitivity margin for regression losses (default
        #   DEFAULT_EPSILON);
        # loss - one of the four losses above (default "hinge"); fit_intercept - bool;
        # n_iter - number of epochs (default 1); random_state - accepted but unused.
        # KNOWN ISSUE: random_state is not stored, so the object does not follow the
        #   estimator
        # convention of keeping every init parameter as an attribute.
        self.C = C
        self.epsilon = epsilon
        self.loss = loss
        self.fit_intercept = fit_intercept
        self.n_iter = n_iter

    def fit(self, X, y):
        # [FUNCTION DOC]
        # HELPER: Online PA training. Initializes w (float64 zeros) and b (0). If X is
        #   sparse,
        # densifies it and scales intercept updates by SPARSE_INTERCEPT_DECAY to mimic
        #   SGD.
        # For each epoch and each sample in order: compute p = project(sample); the loss
        #   is
        # the hinge loss max(0, 1 - y*p) for classification losses or the epsilon-
        #   insensitive
        # loss max(0, |p - y| - epsilon) otherwise. The step is min(C, loss / squared
        #   norm)
        # for hinge / epsilon_insensitive (PA-I) or loss / (squared norm + 1/(2C)) for
        #   the
        # squared variants (PA-II); it is multiplied by y (classification) or by the
        #   sign of
        # (y - p) (regression). Then w gets step times the sample and, if fit_intercept,
        #   b gets
        # intercept_decay times step.
        # Inputs: X - dense or sparse 2D array; y - 1D target. Outputs: None.
        # Side effects: sets self.w and self.b.
        # Exceptions: an unsupported loss leaves step undefined and raises NameError (or
        # UnboundLocalError); a zero-norm sample with PA-I loss divides by zero.
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features, dtype=np.float64)
        self.b = 0.0

        # Mimic SGD's behavior for intercept
        intercept_decay = 1.0
        if issparse(X):
            intercept_decay = SPARSE_INTERCEPT_DECAY
            X = X.toarray()

        for t in range(self.n_iter):
            for i in range(n_samples):
                p = self.project(X[i])
                if self.loss in ("hinge", "squared_hinge"):
                    loss = max(1 - y[i] * p, 0)
                else:
                    loss = max(np.abs(p - y[i]) - self.epsilon, 0)

                sqnorm = np.dot(X[i], X[i])

                if self.loss in ("hinge", "epsilon_insensitive"):
                    step = min(self.C, loss / sqnorm)
                elif self.loss in ("squared_hinge", "squared_epsilon_insensitive"):
                    step = loss / (sqnorm + 1.0 / (2 * self.C))

                if self.loss in ("hinge", "squared_hinge"):
                    step *= y[i]
                else:
                    step *= np.sign(y[i] - p)

                self.w += step * X[i]
                if self.fit_intercept:
                    self.b += intercept_decay * step

    def project(self, X):
        # [FUNCTION DOC]
        # HELPER: Returns the dot product of X with w plus b (scalar for one sample, 1D
        #   for 2D X).
        return np.dot(X, self.w) + self.b


@pytest.mark.filterwarnings("ignore::FutureWarning")
@pytest.mark.parametrize("average", [False, True])
@pytest.mark.parametrize("fit_intercept", [True, False])
@pytest.mark.parametrize("csr_container", [None, *CSR_CONTAINERS])
def test_classifier_accuracy(csr_container, fit_intercept, average):
    # [FUNCTION DOC]
    # Verifies PassiveAggressiveClassifier (C=1, 30 epochs, tol=None) reaches training
    # accuracy above 0.79 on iris for dense and CSR input, with and without intercept
    #   and
    # averaging; when average=True also checks the private averaged/standard coef and
    # intercept attributes exist.
    data = csr_container(X) if csr_container is not None else X
    clf = PassiveAggressiveClassifier(
        C=1.0,
        max_iter=30,
        fit_intercept=fit_intercept,
        random_state=1,
        average=average,
        tol=None,
    )
    clf.fit(data, y)
    score = clf.score(data, y)
    assert score > 0.79
    if average:
        assert hasattr(clf, "_average_coef")
        assert hasattr(clf, "_average_intercept")
        assert hasattr(clf, "_standard_intercept")
        assert hasattr(clf, "_standard_coef")


@pytest.mark.filterwarnings("ignore::FutureWarning")
@pytest.mark.parametrize("average", [False, True])
@pytest.mark.parametrize("csr_container", [None, *CSR_CONTAINERS])
def test_classifier_partial_fit(csr_container, average):
    # [FUNCTION DOC]
    # Verifies 30 calls of partial_fit (classes given) reach accuracy above 0.79 on iris
    # for dense/CSR input and both average settings, and that averaged attributes exist
    # when average=True.
    classes = np.unique(y)
    data = csr_container(X) if csr_container is not None else X
    clf = PassiveAggressiveClassifier(random_state=0, average=average, max_iter=5)
    for t in range(30):
        clf.partial_fit(data, y, classes)
    score = clf.score(data, y)
    assert score > 0.79
    if average:
        assert hasattr(clf, "_average_coef")
        assert hasattr(clf, "_average_intercept")
        assert hasattr(clf, "_standard_intercept")
        assert hasattr(clf, "_standard_coef")


@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_classifier_refit():
    # [FUNCTION DOC]
    # Verifies refitting on a different number of features and string labels (iris
    #   target
    # names) works and updates classes_ accordingly.
    # Classifier can be retrained on different labels and features.
    clf = PassiveAggressiveClassifier(max_iter=5).fit(X, y)
    assert_array_equal(clf.classes_, np.unique(y))

    clf.fit(X[:, :-1], iris.target_names[y])
    assert_array_equal(clf.classes_, iris.target_names)


# TODO(1.10): Move to test_sgd.py
@pytest.mark.filterwarnings("ignore::FutureWarning")
@pytest.mark.parametrize("csr_container", [None, *CSR_CONTAINERS])
@pytest.mark.parametrize("loss", ("hinge", "squared_hinge"))
def test_classifier_correctness(loss, csr_container):
    # [FUNCTION DOC]
    # Verifies PassiveAggressiveClassifier (4 epochs, no shuffle, tol=None) learns the
    #   same
    # weights as MyPassiveAggressive on the binary +1/-1 iris problem for hinge and
    # squared_hinge, dense and CSR input.
    y_bin = y.copy()
    y_bin[y != 1] = -1
    data = csr_container(X) if csr_container is not None else X

    clf1 = MyPassiveAggressive(loss=loss, n_iter=4)
    clf1.fit(data, y_bin)

    clf2 = PassiveAggressiveClassifier(loss=loss, max_iter=4, shuffle=False, tol=None)
    clf2.fit(data, y_bin)

    assert_allclose(clf1.w, clf2.coef_.ravel())


@pytest.mark.filterwarnings("ignore::FutureWarning")
@pytest.mark.parametrize(
    "response_method", ["predict_proba", "predict_log_proba", "transform"]
)
def test_classifier_undefined_methods(response_method):
    # [FUNCTION DOC]
    # Verifies predict_proba, predict_log_proba and transform are not available
    # (attribute access raises AttributeError).
    clf = PassiveAggressiveClassifier(max_iter=100)
    with pytest.raises(AttributeError):
        getattr(clf, response_method)


@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_class_weights():
    # [FUNCTION DOC]
    # Verifies class_weight effect on a 5-point 2D dataset: without weights the point
    # (0.2, -1) is predicted as class 1; with class 1 down-weighted to 0.001 the
    #   boundary
    # moves and it is predicted as -1.
    # Test class weights.
    X2 = np.array([[-1.0, -1.0], [-1.0, 0], [-0.8, -1.0], [1.0, 1.0], [1.0, 0.0]])
    y2 = [1, 1, 1, -1, -1]

    clf = PassiveAggressiveClassifier(
        C=0.1, max_iter=100, class_weight=None, random_state=100
    )
    clf.fit(X2, y2)
    assert_array_equal(clf.predict([[0.2, -1.0]]), np.array([1]))

    # we give a small weights to class 1
    clf = PassiveAggressiveClassifier(
        C=0.1, max_iter=100, class_weight={1: 0.001}, random_state=100
    )
    clf.fit(X2, y2)

    # now the hyperplane should rotate clock-wise and
    # the prediction on this point should shift
    assert_array_equal(clf.predict([[0.2, -1.0]]), np.array([-1]))


@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_partial_fit_weight_class_balanced():
    # [FUNCTION DOC]
    # Verifies partial_fit with class_weight='balanced' raises ValueError.
    # partial_fit with class_weight='balanced' not supported
    clf = PassiveAggressiveClassifier(class_weight="balanced", max_iter=100)
    with pytest.raises(ValueError):
        clf.partial_fit(X, y, classes=np.unique(y))


@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_equal_class_weight():
    # [FUNCTION DOC]
    # Verifies that on an already balanced 4-sample dataset, class_weight=None,
    # 'balanced' and equal explicit weights {0: 0.5, 1: 0.5} give coef_ equal to 2
    #   decimals.
    X2 = [[1, 0], [1, 0], [0, 1], [0, 1]]
    y2 = [0, 0, 1, 1]
    clf = PassiveAggressiveClassifier(C=0.1, tol=None, class_weight=None)
    clf.fit(X2, y2)

    # Already balanced, so "balanced" weights should have no effect
    clf_balanced = PassiveAggressiveClassifier(C=0.1, tol=None, class_weight="balanced")
    clf_balanced.fit(X2, y2)

    clf_weighted = PassiveAggressiveClassifier(
        C=0.1, tol=None, class_weight={0: 0.5, 1: 0.5}
    )
    clf_weighted.fit(X2, y2)

    # should be similar up to some epsilon due to learning rate schedule
    assert_almost_equal(clf.coef_, clf_weighted.coef_, decimal=2)
    assert_almost_equal(clf.coef_, clf_balanced.coef_, decimal=2)


@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_wrong_class_weight_label():
    # [FUNCTION DOC]
    # Verifies fit raises ValueError when class_weight names a label (0) absent from y.
    # ValueError due to wrong class_weight label.
    X2 = np.array([[-1.0, -1.0], [-1.0, 0], [-0.8, -1.0], [1.0, 1.0], [1.0, 0.0]])
    y2 = [1, 1, 1, -1, -1]

    clf = PassiveAggressiveClassifier(class_weight={0: 0.5}, max_iter=100)
    with pytest.raises(ValueError):
        clf.fit(X2, y2)


@pytest.mark.filterwarnings("ignore::FutureWarning")
@pytest.mark.parametrize("average", [False, True])
@pytest.mark.parametrize("fit_intercept", [True, False])
@pytest.mark.parametrize("csr_container", [None, *CSR_CONTAINERS])
def test_regressor_mse(csr_container, fit_intercept, average):
    # [FUNCTION DOC]
    # Verifies PassiveAggressiveRegressor (5 epochs) fitted on +1/-1 iris targets has
    # training MSE below 1.7 for dense/CSR input, with/without intercept and averaging,
    # and that averaged attributes exist when average=True.
    y_bin = y.copy()
    y_bin[y != 1] = -1

    data = csr_container(X) if csr_container is not None else X
    reg = PassiveAggressiveRegressor(
        C=1.0,
        fit_intercept=fit_intercept,
        random_state=0,
        average=average,
        max_iter=5,
    )
    reg.fit(data, y_bin)
    pred = reg.predict(data)
    assert np.mean((pred - y_bin) ** 2) < 1.7
    if average:
        assert hasattr(reg, "_average_coef")
        assert hasattr(reg, "_average_intercept")
        assert hasattr(reg, "_standard_intercept")
        assert hasattr(reg, "_standard_coef")


@pytest.mark.filterwarnings("ignore::FutureWarning")
@pytest.mark.parametrize("average", [False, True])
@pytest.mark.parametrize("csr_container", [None, *CSR_CONTAINERS])
def test_regressor_partial_fit(csr_container, average):
    # [FUNCTION DOC]
    # Verifies 50 partial_fit calls give training MSE below 1.7 for dense/CSR input and
    #   both
    # average settings, plus averaged attributes when average=True.
    y_bin = y.copy()
    y_bin[y != 1] = -1

    data = csr_container(X) if csr_container is not None else X
    reg = PassiveAggressiveRegressor(random_state=0, average=average, max_iter=100)
    for t in range(50):
        reg.partial_fit(data, y_bin)
    pred = reg.predict(data)
    assert np.mean((pred - y_bin) ** 2) < 1.7
    if average:
        assert hasattr(reg, "_average_coef")
        assert hasattr(reg, "_average_intercept")
        assert hasattr(reg, "_standard_intercept")
        assert hasattr(reg, "_standard_coef")


# TODO(1.10): Move to test_sgd.py
@pytest.mark.filterwarnings("ignore::FutureWarning")
@pytest.mark.parametrize("csr_container", [None, *CSR_CONTAINERS])
@pytest.mark.parametrize("loss", ("epsilon_insensitive", "squared_epsilon_insensitive"))
def test_regressor_correctness(loss, csr_container):
    # [FUNCTION DOC]
    # Verifies PassiveAggressiveRegressor (4 epochs, no shuffle, tol=None) matches
    # MyPassiveAggressive weights for both epsilon-insensitive losses, dense and CSR.
    y_bin = y.copy()
    y_bin[y != 1] = -1
    data = csr_container(X) if csr_container is not None else X

    reg1 = MyPassiveAggressive(loss=loss, n_iter=4)
    reg1.fit(data, y_bin)

    reg2 = PassiveAggressiveRegressor(loss=loss, max_iter=4, shuffle=False, tol=None)
    reg2.fit(data, y_bin)

    assert_allclose(reg1.w, reg2.coef_.ravel())


@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_regressor_undefined_methods():
    # [FUNCTION DOC]
    # Verifies calling transform on the regressor raises AttributeError.
    reg = PassiveAggressiveRegressor(max_iter=100)
    with pytest.raises(AttributeError):
        reg.transform(X)


# TODO(1.10): remove
@pytest.mark.parametrize(
    "Estimator", [PassiveAggressiveClassifier, PassiveAggressiveRegressor]
)
def test_class_deprecation(Estimator):
    # [FUNCTION DOC]
    # Verifies instantiating either PassiveAggressive estimator emits a FutureWarning
    # stating the class is deprecated.
    # Check that we raise the proper deprecation warning.

    with pytest.warns(FutureWarning, match="Class PassiveAggressive.+is deprecated"):
        Estimator()


@pytest.mark.parametrize(["loss", "lr"], [("hinge", "pa1"), ("squared_hinge", "pa2")])
def test_passive_aggressive_classifier_vs_sgd(loss, lr):
    # [FUNCTION DOC]
    # Verifies PassiveAggressiveClassifier with hinge/squared_hinge and C=0.987 gives
    #   the
    # same decision_function as SGDClassifier with hinge loss, no penalty and
    #   learning_rate
    # pa1/pa2 with eta0=0.987 on a make_classification dataset.
    """Test that both are equivalent."""
    X, y = make_classification(
        n_samples=100, n_features=10, n_informative=5, random_state=1234
    )
    pa = PassiveAggressiveClassifier(loss=loss, C=0.987, random_state=42).fit(X, y)
    sgd = SGDClassifier(
        loss="hinge", penalty=None, learning_rate=lr, eta0=0.987, random_state=42
    ).fit(X, y)
    assert_allclose(pa.decision_function(X), sgd.decision_function(X))


@pytest.mark.parametrize(
    ["loss", "lr"],
    [("epsilon_insensitive", "pa1"), ("squared_epsilon_insensitive", "pa2")],
)
def test_passive_aggressive_regressor_vs_sgd(loss, lr):
    # [FUNCTION DOC]
    # Verifies PassiveAggressiveRegressor with an epsilon-insensitive loss (epsilon
    #   0.123,
    # C 0.987) predicts the same as SGDRegressor with epsilon_insensitive loss, no
    #   penalty
    # and learning_rate pa1/pa2 on a make_regression dataset.
    """Test that both are equivalent."""
    X, y = make_regression(
        n_samples=100, n_features=10, n_informative=5, random_state=1234
    )
    pa = PassiveAggressiveRegressor(
        loss=loss, epsilon=0.123, C=0.987, random_state=42
    ).fit(X, y)
    sgd = SGDRegressor(
        loss="epsilon_insensitive",
        epsilon=0.123,
        penalty=None,
        learning_rate=lr,
        eta0=0.987,
        random_state=42,
    ).fit(X, y)
    assert_allclose(pa.predict(X), sgd.predict(X))
