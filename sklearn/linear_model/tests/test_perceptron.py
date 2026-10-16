# [MODULE DOC]
# Tests for sklearn.linear_model.Perceptron.
# At import time loads the iris dataset and shuffles it with a fixed RandomState(12),
# producing module-level X (150x4 features) and y (3-class labels) shared by all tests.
# Contents: MyPerceptron, a minimal reference implementation, and tests for accuracy,
# equivalence with the reference, absence of probability methods and l1_ratio handling.
import numpy as np
import pytest

from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron
from sklearn.utils import check_random_state
from sklearn.utils._testing import assert_allclose, assert_array_almost_equal
from sklearn.utils.fixes import CSR_CONTAINERS

iris = load_iris()
random_state = check_random_state(12)
indices = np.arange(iris.data.shape[0])
random_state.shuffle(indices)
X = iris.data[indices]
y = iris.target[indices]


class MyPerceptron:
    # [CLASS DOC]
    # HELPER: Reference binary perceptron (no learning rate, no regularization, no
    # shuffling) used to cross-check the coefficients learned by Perceptron.
    # Expects labels in {-1, +1}. Attributes w (weights) and b (bias) are set by fit.
    def __init__(self, n_iter=1):
        # [FUNCTION DOC]
        # HELPER: Stores n_iter (int, default 1), the number of passes over the data.
        self.n_iter = n_iter

    def fit(self, X, y):
        # [FUNCTION DOC]
        # HELPER: Trains the perceptron. Initializes w to a float64 zero vector of
        #   length
        # n_features and b to 0. For each of n_iter epochs, visits samples in order;
        #   when the
        # predicted sign differs from y[i], adds y[i] times the sample to w and y[i] to
        #   b.
        # Inputs: X - dense 2D array (n_samples, n_features); y - labels in {-1, +1}.
        # Outputs: None (does not return self).
        # Side effects: sets self.w and self.b.
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features, dtype=np.float64)
        self.b = 0.0

        for t in range(self.n_iter):
            for i in range(n_samples):
                if self.predict(X[i])[0] != y[i]:
                    self.w += y[i] * X[i]
                    self.b += y[i]

    def project(self, X):
        # [FUNCTION DOC]
        # HELPER: Returns the raw decision values, the dot product of X with w plus b.
        # Inputs: X - 2D array (n, n_features). Outputs: 1D array of length n.
        return np.dot(X, self.w) + self.b

    def predict(self, X):
        # [FUNCTION DOC]
        # HELPER: Promotes X to at least 2D and returns the sign (-1, 0 or +1) of
        #   project(X).
        # Outputs: 1D float array of length n.
        X = np.atleast_2d(X)
        return np.sign(self.project(X))


@pytest.mark.parametrize("container", CSR_CONTAINERS + [np.array])
def test_perceptron_accuracy(container):
    # [FUNCTION DOC]
    # Verifies that Perceptron (max_iter=100, tol=None, no shuffling) fits the
    #   multiclass
    # iris data with training accuracy above 0.7, for dense arrays and every CSR
    #   container.
    # Inputs: container - parametrized constructor applied to X.
    data = container(X)
    clf = Perceptron(max_iter=100, tol=None, shuffle=False)
    clf.fit(data, y)
    score = clf.score(data, y)
    assert score > 0.7


def test_perceptron_correctness():
    # [FUNCTION DOC]
    # Verifies that Perceptron with 2 epochs, no shuffling and tol=None learns the same
    # weights as MyPerceptron with n_iter=2 on a binary version of iris (class 1 vs rest
    # encoded as +1/-1), comparing the raveled coef_ to w almost-equal.
    y_bin = y.copy()
    y_bin[y != 1] = -1

    clf1 = MyPerceptron(n_iter=2)
    clf1.fit(X, y_bin)

    clf2 = Perceptron(max_iter=2, shuffle=False, tol=None)
    clf2.fit(X, y_bin)

    assert_array_almost_equal(clf1.w, clf2.coef_.ravel())


def test_undefined_methods():
    # [FUNCTION DOC]
    # Verifies that accessing predict_proba and predict_log_proba on an (unfitted)
    # Perceptron raises AttributeError, i.e. the methods are not exposed.
    clf = Perceptron(max_iter=100)
    for meth in ("predict_proba", "predict_log_proba"):
        with pytest.raises(AttributeError):
            getattr(clf, meth)


def test_perceptron_l1_ratio():
    # [FUNCTION DOC]
    # Verifies l1_ratio is honored with penalty='elasticnet': l1_ratio 0 and 0.15 give
    # different training scores; l1_ratio=1 matches penalty='l1' coefficients and
    # l1_ratio=0 matches penalty='l2' coefficients (assert_allclose).
    """Check that `l1_ratio` has an impact when `penalty='elasticnet'`"""
    clf1 = Perceptron(l1_ratio=0, penalty="elasticnet")
    clf1.fit(X, y)

    clf2 = Perceptron(l1_ratio=0.15, penalty="elasticnet")
    clf2.fit(X, y)

    assert clf1.score(X, y) != clf2.score(X, y)

    # check that the bounds of elastic net which should correspond to an l1 or
    # l2 penalty depending of `l1_ratio` value.
    clf_l1 = Perceptron(penalty="l1").fit(X, y)
    clf_elasticnet = Perceptron(l1_ratio=1, penalty="elasticnet").fit(X, y)
    assert_allclose(clf_l1.coef_, clf_elasticnet.coef_)

    clf_l2 = Perceptron(penalty="l2").fit(X, y)
    clf_elasticnet = Perceptron(l1_ratio=0, penalty="elasticnet").fit(X, y)
    assert_allclose(clf_l2.coef_, clf_elasticnet.coef_)
