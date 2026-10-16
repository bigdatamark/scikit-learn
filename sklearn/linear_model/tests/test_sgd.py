# [MODULE DOC]
# Tests for SGDClassifier, SGDRegressor and SGDOneClassSVM (dense and sparse
# input), plus the Cython loss functions in _sgd_fast. Defines factory functions
# that inject deterministic defaults (random_state=42, tol=None, max_iter=5),
# sparse wrapper subclasses that convert X to CSR, module-level toy datasets
# (X/Y, X2/Y2 string labels, X3/X4 redundant features, X5/Y5 binary, iris), and
# reference ASGD implementations for checking averaging.
import pickle
import warnings
from unittest.mock import Mock

import joblib
import numpy as np
import pytest
import scipy.sparse as sp
from scipy.optimize import minimize

from sklearn import datasets, linear_model, metrics
from sklearn.base import clone, is_classifier
from sklearn.datasets import make_blobs
from sklearn.exceptions import ConvergenceWarning
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import _sgd_fast as sgd_fast
from sklearn.linear_model import _stochastic_gradient
from sklearn.model_selection import (
    RandomizedSearchCV,
    ShuffleSplit,
    StratifiedShuffleSplit,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler, scale
from sklearn.svm import OneClassSVM
from sklearn.utils import get_tags
from sklearn.utils._sparse import _align_api_if_sparse
from sklearn.utils._testing import (
    assert_allclose,
    assert_almost_equal,
    assert_array_almost_equal,
    assert_array_equal,
)
from sklearn.utils.fixes import _sparse_random_array


def _update_kwargs(kwargs):
    # [FUNCTION DOC]
    # HELPER: Fills in test defaults in the kwargs dict when absent: random_state=42,
    # tol=None, max_iter=5.
    # Side effects: mutates kwargs in place; returns None.
    if "random_state" not in kwargs:
        kwargs["random_state"] = 42

    if "tol" not in kwargs:
        kwargs["tol"] = None
    if "max_iter" not in kwargs:
        kwargs["max_iter"] = 5


class _SparseSGDClassifier(linear_model.SGDClassifier):
    # [CLASS DOC]
    # HELPER: SGDClassifier subclass converting every X to a CSR array (API-aligned)
    # before fit, partial_fit, decision_function and predict_proba, so the shared tests
    # exercise the sparse code paths.
    def fit(self, X, y, *args, **kw):
        # [FUNCTION DOC]
        # HELPER: Converts X to CSR and delegates to SGDClassifier.fit. Outputs: self.
        X = _align_api_if_sparse(sp.csr_array(X))
        return super().fit(X, y, *args, **kw)

    def partial_fit(self, X, y, *args, **kw):
        # [FUNCTION DOC]
        # HELPER: Converts X to CSR and delegates to SGDClassifier.partial_fit.
        X = _align_api_if_sparse(sp.csr_array(X))
        return super().partial_fit(X, y, *args, **kw)

    def decision_function(self, X):
        # [FUNCTION DOC]
        # HELPER: Converts X to CSR and returns the parent decision_function.
        X = _align_api_if_sparse(sp.csr_array(X))
        return super().decision_function(X)

    def predict_proba(self, X):
        # [FUNCTION DOC]
        # HELPER: Converts X to CSR and returns the parent predict_proba.
        # KNOWN ISSUE: overriding predict_proba unconditionally means hasattr checks on
        # this wrapper differ from the real estimator (noted in test_sgd_proba).
        X = _align_api_if_sparse(sp.csr_array(X))
        return super().predict_proba(X)


class _SparseSGDRegressor(linear_model.SGDRegressor):
    # [CLASS DOC]
    # HELPER: SGDRegressor subclass converting X to CSR before fit and partial_fit.
    def fit(self, X, y, *args, **kw):
        # [FUNCTION DOC]
        # HELPER: Converts X to CSR and delegates to SGDRegressor.fit.
        X = _align_api_if_sparse(sp.csr_array(X))
        return linear_model.SGDRegressor.fit(self, X, y, *args, **kw)

    def partial_fit(self, X, y, *args, **kw):
        # [FUNCTION DOC]
        # HELPER: Converts X to CSR and delegates to SGDRegressor.partial_fit.
        X = _align_api_if_sparse(sp.csr_array(X))
        return linear_model.SGDRegressor.partial_fit(self, X, y, *args, **kw)

    def decision_function(self, X, *args, **kw):
        # [FUNCTION DOC]
        # HELPER: Aligns X's sparse API (without converting dense X to CSR) and delegates
        # to SGDRegressor.decision_function.
        # KNOWN ISSUE: unlike the other wrappers it does not convert to CSR, and the code
        # comment says it is untested.
        # XXX untested as of v0.22
        return linear_model.SGDRegressor.decision_function(
            self, _align_api_if_sparse(X), *args, **kw
        )


class _SparseSGDOneClassSVM(linear_model.SGDOneClassSVM):
    # [CLASS DOC]
    # HELPER: SGDOneClassSVM subclass converting X to CSR before fit, partial_fit and
    # decision_function.
    def fit(self, X, *args, **kw):
        # [FUNCTION DOC]
        # HELPER: Converts X to CSR and delegates to SGDOneClassSVM.fit.
        X = _align_api_if_sparse(sp.csr_array(X))
        return linear_model.SGDOneClassSVM.fit(self, X, *args, **kw)

    def partial_fit(self, X, *args, **kw):
        # [FUNCTION DOC]
        # HELPER: Converts X to CSR and delegates to SGDOneClassSVM.partial_fit.
        X = _align_api_if_sparse(sp.csr_array(X))
        return linear_model.SGDOneClassSVM.partial_fit(self, X, *args, **kw)

    def decision_function(self, X, *args, **kw):
        # [FUNCTION DOC]
        # HELPER: Converts X to CSR and delegates to SGDOneClassSVM.decision_function.
        X = _align_api_if_sparse(sp.csr_array(X))
        return linear_model.SGDOneClassSVM.decision_function(self, X, *args, **kw)


def SGDClassifier(**kwargs):
    # [FUNCTION DOC]
    # HELPER: Factory returning linear_model.SGDClassifier with test defaults from
    # _update_kwargs applied to the given keyword arguments.
    _update_kwargs(kwargs)
    return linear_model.SGDClassifier(**kwargs)


def SGDRegressor(**kwargs):
    # [FUNCTION DOC]
    # HELPER: Factory returning linear_model.SGDRegressor with test defaults applied.
    _update_kwargs(kwargs)
    return linear_model.SGDRegressor(**kwargs)


def SGDOneClassSVM(**kwargs):
    # [FUNCTION DOC]
    # HELPER: Factory returning linear_model.SGDOneClassSVM with test defaults applied.
    _update_kwargs(kwargs)
    return linear_model.SGDOneClassSVM(**kwargs)


def SparseSGDClassifier(**kwargs):
    # [FUNCTION DOC]
    # HELPER: Factory returning the CSR-converting classifier wrapper with test
    # defaults applied.
    _update_kwargs(kwargs)
    return _SparseSGDClassifier(**kwargs)


def SparseSGDRegressor(**kwargs):
    # [FUNCTION DOC]
    # HELPER: Factory returning the CSR-converting regressor wrapper with defaults.
    _update_kwargs(kwargs)
    return _SparseSGDRegressor(**kwargs)


def SparseSGDOneClassSVM(**kwargs):
    # [FUNCTION DOC]
    # HELPER: Factory returning the CSR-converting one-class SVM wrapper with defaults.
    _update_kwargs(kwargs)
    return _SparseSGDOneClassSVM(**kwargs)


# Test Data

# test sample 1
X = np.array([[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1]])
Y = [1, 1, 1, 2, 2, 2]
T = np.array([[-1, -1], [2, 2], [3, 2]])
true_result = [1, 2, 2]

# test sample 2; string class labels
X2 = np.array(
    [
        [-1, 1],
        [-0.75, 0.5],
        [-1.5, 1.5],
        [1, 1],
        [0.75, 0.5],
        [1.5, 1.5],
        [-1, -1],
        [0, -0.5],
        [1, -1],
    ]
)
Y2 = ["one"] * 3 + ["two"] * 3 + ["three"] * 3
T2 = np.array([[-1.5, 0.5], [1, 2], [0, -2]])
true_result2 = ["one", "two", "three"]

# test sample 3
X3 = np.array(
    [
        [1, 1, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 1, 1],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0, 0],
    ]
)
Y3 = np.array([1, 1, 1, 1, 2, 2, 2, 2])

# test sample 4 - two more or less redundant feature groups
X4 = np.array(
    [
        [1, 0.9, 0.8, 0, 0, 0],
        [1, 0.84, 0.98, 0, 0, 0],
        [1, 0.96, 0.88, 0, 0, 0],
        [1, 0.91, 0.99, 0, 0, 0],
        [0, 0, 0, 0.89, 0.91, 1],
        [0, 0, 0, 0.79, 0.84, 1],
        [0, 0, 0, 0.91, 0.95, 1],
        [0, 0, 0, 0.93, 1, 1],
    ]
)
Y4 = np.array([1, 1, 1, 1, 2, 2, 2, 2])

iris = datasets.load_iris()

# test sample 5 - test sample 1 as binary classification problem
X5 = np.array([[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1]])
Y5 = [1, 1, 1, 2, 2, 2]
true_result5 = [0, 1, 1]


###############################################################################
# Common Test Case to classification and regression


# a simple implementation of ASGD to use for testing
# uses squared loss to find the gradient
def asgd(klass, X, y, eta, alpha, weight_init=None, intercept_init=0.0):
    # [FUNCTION DOC]
    # HELPER: Reference averaged SGD for squared loss with L2 penalty, one pass in
    # order. For each sample: prediction = dot(x, w) + b, gradient = prediction - y;
    # shrink w by (1 - eta*alpha), subtract eta*gradient*x from w, subtract
    # eta*gradient*decay from b (decay 0.01 for the sparse classifier/regressor
    # factories, else 1); update running means of w and b.
    # Inputs: klass - factory (selects decay); X (n, d) dense; y (n,); eta, alpha
    # floats; weight_init (d,) or None (zeros); intercept_init float.
    # Outputs: (average weights (d,), average intercept float).
    # Side effects: mutates weight_init in place when it is provided.
    if weight_init is None:
        weights = np.zeros(X.shape[1])
    else:
        weights = weight_init

    average_weights = np.zeros(X.shape[1])
    intercept = intercept_init
    average_intercept = 0.0
    decay = 1.0

    # sparse data has a fixed decay of .01
    if klass in (SparseSGDClassifier, SparseSGDRegressor):
        decay = 0.01

    for i, entry in enumerate(X):
        p = np.dot(entry, weights)
        p += intercept
        gradient = p - y[i]
        weights *= 1.0 - (eta * alpha)
        weights += -(eta * gradient * entry)
        intercept += -(eta * gradient) * decay

        average_weights *= i
        average_weights += weights
        average_weights /= i + 1.0

        average_intercept *= i
        average_intercept += intercept
        average_intercept /= i + 1.0

    return average_weights, average_intercept


def _test_warm_start(klass, X, Y, lr):
    # [FUNCTION DOC]
    # HELPER: Checks that explicit warm start (passing coef_init/intercept_init from a
    # previous fit) and implicit warm_start=True followed by set_params(alpha) yield
    # identical t_ and coef_ for the given learning-rate schedule.
    # Inputs: klass factory, X, Y data, lr learning_rate name.
    # Test that explicit warm restart...
    clf = klass(alpha=0.01, eta0=0.01, shuffle=False, learning_rate=lr)
    clf.fit(X, Y)

    clf2 = klass(alpha=0.001, eta0=0.01, shuffle=False, learning_rate=lr)
    clf2.fit(X, Y, coef_init=clf.coef_.copy(), intercept_init=clf.intercept_.copy())

    # ... and implicit warm restart are equivalent.
    clf3 = klass(
        alpha=0.01, eta0=0.01, shuffle=False, warm_start=True, learning_rate=lr
    )
    clf3.fit(X, Y)

    assert clf3.t_ == clf.t_
    assert_array_almost_equal(clf3.coef_, clf.coef_)

    clf3.set_params(alpha=0.001)
    clf3.fit(X, Y)

    assert clf3.t_ == clf2.t_
    assert_array_almost_equal(clf3.coef_, clf2.coef_)


@pytest.mark.parametrize(
    "klass", [SGDClassifier, SparseSGDClassifier, SGDRegressor, SparseSGDRegressor]
)
@pytest.mark.parametrize("lr", ["constant", "optimal", "invscaling", "adaptive"])
def test_warm_start(klass, lr):
    # [FUNCTION DOC]
    # Runs _test_warm_start on the toy data for dense/sparse classifier and regressor
    # across all four non-PA learning-rate schedules.
    _test_warm_start(klass, X, Y, lr)


@pytest.mark.parametrize(
    "klass", [SGDClassifier, SparseSGDClassifier, SGDRegressor, SparseSGDRegressor]
)
def test_input_format(klass):
    # [FUNCTION DOC]
    # Fitting on a 1D target works, while a 2D target with two columns raises
    # ValueError.
    # Input format tests.
    clf = klass(alpha=0.01, shuffle=False)
    clf.fit(X, Y)
    Y_ = np.array(Y)[:, np.newaxis]

    Y_ = np.c_[Y_, Y_]
    with pytest.raises(ValueError):
        clf.fit(X, Y_)


@pytest.mark.parametrize("lr", ["pa1", "pa2"])
@pytest.mark.parametrize(
    ["est", "loss"], [(SGDClassifier, "squared_hinge"), (SGDRegressor, "squared_error")]
)
def test_learning_rate_PA_raises(lr, est, loss):
    # [FUNCTION DOC]
    # Passive-aggressive learning rates pa1/pa2 combined with a disallowed loss
    # (squared_hinge for classifier, squared_error for regressor) raise ValueError.
    """Test that SGD raises with forbidden loss for passive-aggressive algo."""
    est = est(loss=loss, learning_rate=lr)
    with pytest.raises(ValueError):
        est.fit(X, Y)


@pytest.mark.parametrize(
    "klass", [SGDClassifier, SparseSGDClassifier, SGDRegressor, SparseSGDRegressor]
)
def test_clone(klass):
    # [FUNCTION DOC]
    # A cloned estimator with penalty changed via set_params produces the same coef_
    # as one constructed directly with that penalty.
    # Test whether clone works ok.
    clf = klass(alpha=0.01, penalty="l1")
    clf = clone(clf)
    clf.set_params(penalty="l2")
    clf.fit(X, Y)

    clf2 = klass(alpha=0.01, penalty="l2")
    clf2.fit(X, Y)

    assert_array_equal(clf.coef_, clf2.coef_)


@pytest.mark.parametrize(
    "klass",
    [
        SGDClassifier,
        SparseSGDClassifier,
        SGDRegressor,
        SparseSGDRegressor,
        SGDOneClassSVM,
        SparseSGDOneClassSVM,
    ],
)
def test_plain_has_no_average_attr(klass):
    # [FUNCTION DOC]
    # With average=True the private _average_coef/_average_intercept/_standard_*
    # attributes exist after fit; with the default (no averaging) none of them exist.
    clf = klass(average=True, eta0=0.01)
    clf.fit(X, Y)

    assert hasattr(clf, "_average_coef")
    assert hasattr(clf, "_average_intercept")
    assert hasattr(clf, "_standard_intercept")
    assert hasattr(clf, "_standard_coef")

    clf = klass()
    clf.fit(X, Y)

    assert not hasattr(clf, "_average_coef")
    assert not hasattr(clf, "_average_intercept")
    assert not hasattr(clf, "_standard_intercept")
    assert not hasattr(clf, "_standard_coef")


@pytest.mark.parametrize(
    "klass",
    [
        SGDClassifier,
        SparseSGDClassifier,
        SGDRegressor,
        SparseSGDRegressor,
        SGDOneClassSVM,
        SparseSGDOneClassSVM,
    ],
)
def test_late_onset_averaging_not_reached(klass):
    # [FUNCTION DOC]
    # With average=600 and only 100 partial_fit passes over 6 samples, averaging never
    # starts, so coef_ and intercept_ (or offset_) equal those of a non-averaged model.
    clf1 = klass(average=600)
    clf2 = klass()
    for _ in range(100):
        if is_classifier(clf1):
            clf1.partial_fit(X, Y, classes=np.unique(Y))
            clf2.partial_fit(X, Y, classes=np.unique(Y))
        else:
            clf1.partial_fit(X, Y)
            clf2.partial_fit(X, Y)

    assert_array_almost_equal(clf1.coef_, clf2.coef_, decimal=16)
    if klass in [SGDClassifier, SparseSGDClassifier, SGDRegressor, SparseSGDRegressor]:
        assert_almost_equal(clf1.intercept_, clf2.intercept_, decimal=16)
    elif klass in [SGDOneClassSVM, SparseSGDOneClassSVM]:
        assert_allclose(clf1.offset_, clf2.offset_)


@pytest.mark.parametrize(
    "klass", [SGDClassifier, SparseSGDClassifier, SGDRegressor, SparseSGDRegressor]
)
def test_late_onset_averaging_reached(klass):
    # [FUNCTION DOC]
    # With average=7 over two epochs on 6 samples, averaging begins in the second
    # epoch: coef_ and intercept_ must equal asgd applied to one more epoch starting
    # from a non-averaged one-epoch model's parameters.
    eta0 = 0.001
    alpha = 0.0001
    Y_encode = np.array(Y)
    Y_encode[Y_encode == 1] = -1.0
    Y_encode[Y_encode == 2] = 1.0

    clf1 = klass(
        average=7,
        learning_rate="constant",
        loss="squared_error",
        eta0=eta0,
        alpha=alpha,
        max_iter=2,
        shuffle=False,
    )
    clf2 = klass(
        average=False,
        learning_rate="constant",
        loss="squared_error",
        eta0=eta0,
        alpha=alpha,
        max_iter=1,
        shuffle=False,
    )

    clf1.fit(X, Y_encode)
    clf2.fit(X, Y_encode)

    average_weights, average_intercept = asgd(
        klass,
        X,
        Y_encode,
        eta0,
        alpha,
        weight_init=clf2.coef_.ravel(),
        intercept_init=clf2.intercept_,
    )

    assert_array_almost_equal(clf1.coef_.ravel(), average_weights.ravel(), decimal=16)
    assert_almost_equal(clf1.intercept_, average_intercept, decimal=16)


@pytest.mark.parametrize(
    "klass", [SGDClassifier, SparseSGDClassifier, SGDRegressor, SparseSGDRegressor]
)
def test_early_stopping(klass):
    # [FUNCTION DOC]
    # On iris restricted to two classes, with and without early_stopping and tol=1e-3,
    # training stops before max_iter=1000.
    X = iris.data[iris.target > 0]
    Y = iris.target[iris.target > 0]
    for early_stopping in [True, False]:
        max_iter = 1000
        clf = klass(early_stopping=early_stopping, tol=1e-3, max_iter=max_iter).fit(
            X, Y
        )
        assert clf.n_iter_ < max_iter


@pytest.mark.parametrize(
    "klass", [SGDClassifier, SparseSGDClassifier, SGDRegressor, SparseSGDRegressor]
)
def test_adaptive_longer_than_constant(klass):
    # [FUNCTION DOC]
    # On iris, learning_rate="adaptive" runs more epochs than "constant" with the
    # same eta0 and tol.
    clf1 = klass(learning_rate="adaptive", eta0=0.01, tol=1e-3, max_iter=100)
    clf1.fit(iris.data, iris.target)
    clf2 = klass(learning_rate="constant", eta0=0.01, tol=1e-3, max_iter=100)
    clf2.fit(iris.data, iris.target)
    assert clf1.n_iter_ > clf2.n_iter_


@pytest.mark.parametrize(
    "klass", [SGDClassifier, SparseSGDClassifier, SGDRegressor, SparseSGDRegressor]
)
def test_validation_set_not_used_for_training(klass):
    # [FUNCTION DOC]
    # With early_stopping and tol=None, the model trained on the full data must equal
    # one trained (without early stopping) only on the training split produced by the
    # same (Stratified)ShuffleSplit seed, sorted to undo shuffling.
    X, Y = iris.data, iris.target
    validation_fraction = 0.4
    seed = 42
    shuffle = False
    max_iter = 10
    clf1 = klass(
        early_stopping=True,
        random_state=np.random.RandomState(seed),
        validation_fraction=validation_fraction,
        learning_rate="constant",
        eta0=0.01,
        tol=None,
        max_iter=max_iter,
        shuffle=shuffle,
    )
    clf1.fit(X, Y)
    assert clf1.n_iter_ == max_iter

    clf2 = klass(
        early_stopping=False,
        random_state=np.random.RandomState(seed),
        learning_rate="constant",
        eta0=0.01,
        tol=None,
        max_iter=max_iter,
        shuffle=shuffle,
    )

    if is_classifier(clf2):
        cv = StratifiedShuffleSplit(test_size=validation_fraction, random_state=seed)
    else:
        cv = ShuffleSplit(test_size=validation_fraction, random_state=seed)
    idx_train, idx_val = next(cv.split(X, Y))
    idx_train = np.sort(idx_train)  # remove shuffling
    clf2.fit(X[idx_train], Y[idx_train])
    assert clf2.n_iter_ == max_iter

    assert_array_equal(clf1.coef_, clf2.coef_)


@pytest.mark.parametrize(
    "klass", [SGDClassifier, SparseSGDClassifier, SGDRegressor, SparseSGDRegressor]
)
def test_n_iter_no_change(klass):
    # [FUNCTION DOC]
    # n_iter_ is non-decreasing as n_iter_no_change grows (2, 3, 10), with and without
    # early stopping.
    X, Y = iris.data, iris.target
    # test that n_iter_ increases monotonically with n_iter_no_change
    for early_stopping in [True, False]:
        n_iter_list = [
            klass(
                early_stopping=early_stopping,
                n_iter_no_change=n_iter_no_change,
                tol=1e-4,
                max_iter=1000,
            )
            .fit(X, Y)
            .n_iter_
            for n_iter_no_change in [2, 3, 10]
        ]
        assert_array_equal(n_iter_list, sorted(n_iter_list))


@pytest.mark.parametrize(
    "klass", [SGDClassifier, SparseSGDClassifier, SGDRegressor, SparseSGDRegressor]
)
def test_not_enough_sample_for_early_stopping(klass):
    # [FUNCTION DOC]
    # validation_fraction=0.99 on 8 samples leaves an empty split and fit raises
    # ValueError.
    # test an error is raised if the training or validation set is empty
    clf = klass(early_stopping=True, validation_fraction=0.99)
    with pytest.raises(ValueError):
        clf.fit(X3, Y3)


@pytest.mark.parametrize("Estimator", [SGDClassifier, SGDRegressor])
@pytest.mark.parametrize("l1_ratio", [0, 0.7, 1])
def test_sgd_l1_ratio_not_used(Estimator, l1_ratio):
    # [FUNCTION DOC]
    # With penalty="l1", the value of l1_ratio (None, 0, 0.7, 1) does not change coef_.
    """Check that l1_ratio is not used when penalty is not 'elasticnet'"""
    clf1 = Estimator(penalty="l1", l1_ratio=None, random_state=0).fit(X, Y)
    clf2 = Estimator(penalty="l1", l1_ratio=l1_ratio, random_state=0).fit(X, Y)

    assert_allclose(clf1.coef_, clf2.coef_)


@pytest.mark.parametrize(
    "Estimator", [SGDClassifier, SparseSGDClassifier, SGDRegressor, SparseSGDRegressor]
)
def test_sgd_failing_penalty_validation(Estimator):
    # [FUNCTION DOC]
    # penalty="elasticnet" with l1_ratio=None raises ValueError with the expected
    # message.
    clf = Estimator(penalty="elasticnet", l1_ratio=None)
    with pytest.raises(
        ValueError, match="l1_ratio must be set when penalty is 'elasticnet'"
    ):
        clf.fit(X, Y)


# TODO(1.10): remove this test
@pytest.mark.parametrize(
    "klass",
    [
        SGDClassifier,
        SparseSGDClassifier,
        SGDRegressor,
        SparseSGDRegressor,
        SGDOneClassSVM,
        SparseSGDOneClassSVM,
    ],
)
def test_power_t_limits(klass):
    # [FUNCTION DOC]
    # Negative power_t emits the deprecation FutureWarning; power_t=0.5 emits no
    # warning at all.
    """Check that a warning is raised when `power_t` is negative."""

    # Check that negative values of `power_t` raise a warning
    clf = klass(power_t=-1.0)
    with pytest.warns(
        FutureWarning, match="Negative values for `power_t` are deprecated"
    ):
        clf.fit(X, Y)

    # Check that values of 'power_t in range [0, inf) do not raise a warning
    with warnings.catch_warnings(record=True) as w:
        clf = klass(power_t=0.5)
        clf.fit(X, Y)
    assert len(w) == 0


###############################################################################
# Classification Test Case


@pytest.mark.parametrize("klass", [SGDClassifier, SparseSGDClassifier])
def test_sgd_clf(klass):
    # [FUNCTION DOC]
    # For hinge, squared_hinge, log_loss and modified_huber, the classifier predicts
    # the expected labels on T.
    # Check that SGD gives any results :-)

    for loss in ("hinge", "squared_hinge", "log_loss", "modified_huber"):
        clf = klass(
            penalty="l2",
            alpha=0.01,
            fit_intercept=True,
            loss=loss,
            max_iter=10,
            shuffle=True,
        )
        clf.fit(X, Y)
        # assert_almost_equal(clf.coef_[0], clf.coef_[1], decimal=7)
        assert_array_equal(clf.predict(T), true_result)


@pytest.mark.parametrize(
    "klass", [SGDClassifier, SparseSGDClassifier, SGDOneClassSVM, SparseSGDOneClassSVM]
)
def test_provide_coef(klass):
    # [FUNCTION DOC]
    # A coef_init with the wrong shape raises ValueError "Provided coef_init does not
    # match dataset".
    """Check that the shape of `coef_init` is validated."""
    with pytest.raises(ValueError, match="Provided coef_init does not match dataset"):
        klass().fit(X, Y, coef_init=np.zeros((3,)))


@pytest.mark.parametrize(
    "klass, fit_params",
    [
        (SGDClassifier, {"intercept_init": np.zeros((3,))}),
        (SparseSGDClassifier, {"intercept_init": np.zeros((3,))}),
        (SGDOneClassSVM, {"offset_init": np.zeros((3,))}),
        (SparseSGDOneClassSVM, {"offset_init": np.zeros((3,))}),
    ],
)
def test_set_intercept_offset(klass, fit_params):
    # [FUNCTION DOC]
    # An intercept_init/offset_init of the wrong length raises ValueError.
    """Check that `intercept_init` or `offset_init` is validated."""
    sgd_estimator = klass()
    with pytest.raises(ValueError, match="does not match dataset"):
        sgd_estimator.fit(X, Y, **fit_params)


@pytest.mark.parametrize(
    "klass", [SGDClassifier, SparseSGDClassifier, SGDRegressor, SparseSGDRegressor]
)
def test_sgd_early_stopping_with_partial_fit(klass):
    # [FUNCTION DOC]
    # partial_fit with early_stopping=True raises ValueError.
    """Check that we raise an error for `early_stopping` used with
    `partial_fit`.
    """
    err_msg = "early_stopping should be False with partial_fit"
    with pytest.raises(ValueError, match=err_msg):
        klass(early_stopping=True).partial_fit(X, Y)


@pytest.mark.parametrize(
    "klass, fit_params",
    [
        (SGDClassifier, {"intercept_init": 0}),
        (SparseSGDClassifier, {"intercept_init": 0}),
        (SGDOneClassSVM, {"offset_init": 0}),
        (SparseSGDOneClassSVM, {"offset_init": 0}),
    ],
)
def test_set_intercept_offset_binary(klass, fit_params):
    # [FUNCTION DOC]
    # A scalar intercept_init/offset_init is accepted for binary classification /
    # one-class fitting (no error).
    """Check that we can pass a scaler with binary classification to
    `intercept_init` or `offset_init`."""
    klass().fit(X5, Y5, **fit_params)


@pytest.mark.parametrize("klass", [SGDClassifier, SparseSGDClassifier])
def test_average_binary_computed_correctly(klass):
    # [FUNCTION DOC]
    # Binary averaged SGD with squared loss, constant rate and one epoch matches the
    # asgd reference weights and intercept to 14 decimals.
    # Checks the SGDClassifier correctly computes the average weights
    eta = 0.1
    alpha = 2.0
    n_samples = 20
    n_features = 10
    rng = np.random.RandomState(0)
    X = rng.normal(size=(n_samples, n_features))
    w = rng.normal(size=n_features)

    clf = klass(
        loss="squared_error",
        learning_rate="constant",
        eta0=eta,
        alpha=alpha,
        fit_intercept=True,
        max_iter=1,
        average=True,
        shuffle=False,
    )

    # simple linear function without noise
    y = np.dot(X, w)
    y = np.sign(y)

    clf.fit(X, y)

    average_weights, average_intercept = asgd(klass, X, y, eta, alpha)
    average_weights = average_weights.reshape(1, -1)
    assert_array_almost_equal(clf.coef_, average_weights, decimal=14)
    assert_almost_equal(clf.intercept_, average_intercept, decimal=14)


@pytest.mark.parametrize("klass", [SGDClassifier, SparseSGDClassifier])
def test_set_intercept_to_intercept(klass):
    # [FUNCTION DOC]
    # Passing a previously fitted intercept_ as intercept_init works for both binary
    # and multiclass data (shape consistency).
    # Checks intercept_ shape consistency for the warm starts
    # Inconsistent intercept_ shape.
    clf = klass().fit(X5, Y5)
    klass().fit(X5, Y5, intercept_init=clf.intercept_)
    clf = klass().fit(X, Y)
    klass().fit(X, Y, intercept_init=clf.intercept_)


@pytest.mark.parametrize("klass", [SGDClassifier, SparseSGDClassifier])
def test_sgd_at_least_two_labels(klass):
    # [FUNCTION DOC]
    # Fitting on a target with a single class raises ValueError.
    # Target must have at least two labels
    clf = klass(alpha=0.01, max_iter=20)
    with pytest.raises(ValueError):
        clf.fit(X2, np.ones(9))


@pytest.mark.parametrize("klass", [SGDClassifier, SparseSGDClassifier])
def test_partial_fit_weight_class_balanced(klass):
    # [FUNCTION DOC]
    # partial_fit with class_weight="balanced" raises ValueError with the full advisory
    # message.
    # partial_fit with class_weight='balanced' not supported"""
    regex = (
        r"class_weight 'balanced' is not supported for "
        r"partial_fit\. In order to use 'balanced' weights, "
        r"use compute_class_weight\('balanced', classes=classes, y=y\). "
        r"In place of y you can use a large enough sample "
        r"of the full training set target to properly "
        r"estimate the class frequency distributions\. "
        r"Pass the resulting weights as the class_weight "
        r"parameter\."
    )
    with pytest.raises(ValueError, match=regex):
        klass(class_weight="balanced").partial_fit(X, Y, classes=np.unique(Y))


@pytest.mark.parametrize("klass", [SGDClassifier, SparseSGDClassifier])
def test_sgd_multiclass(klass):
    # [FUNCTION DOC]
    # One-vs-all on 3 string classes yields coef_ (3, 2), intercept_ (3,), decision
    # function (1, 3) and correct predictions on T2.
    # Multi-class test case
    clf = klass(alpha=0.01, max_iter=20).fit(X2, Y2)
    assert clf.coef_.shape == (3, 2)
    assert clf.intercept_.shape == (3,)
    assert clf.decision_function([[0, 0]]).shape == (1, 3)
    pred = clf.predict(T2)
    assert_array_equal(pred, true_result2)


@pytest.mark.parametrize("klass", [SGDClassifier, SparseSGDClassifier])
def test_sgd_multiclass_average(klass):
    # [FUNCTION DOC]
    # Each one-vs-all binary problem of an averaged multiclass fit matches the asgd
    # reference coefficients and intercept for that class.
    eta = 0.001
    alpha = 0.01
    # Multi-class average test case
    clf = klass(
        loss="squared_error",
        learning_rate="constant",
        eta0=eta,
        alpha=alpha,
        fit_intercept=True,
        max_iter=1,
        average=True,
        shuffle=False,
    )

    np_Y2 = np.array(Y2)
    clf.fit(X2, np_Y2)
    classes = np.unique(np_Y2)

    for i, cl in enumerate(classes):
        y_i = np.ones(np_Y2.shape[0])
        y_i[np_Y2 != cl] = -1
        average_coef, average_intercept = asgd(klass, X2, y_i, eta, alpha)
        assert_array_almost_equal(average_coef, clf.coef_[i], decimal=16)
        assert_almost_equal(average_intercept, clf.intercept_[i], decimal=16)


@pytest.mark.parametrize("klass", [SGDClassifier, SparseSGDClassifier])
def test_sgd_multiclass_with_init_coef(klass):
    # [FUNCTION DOC]
    # Multiclass fit with zero coef_init/intercept_init gives correct shapes and
    # predictions.
    # KNOWN ISSUE: "assert clf.intercept_.shape, (3,)" always passes (tuple message
    # form), so the intercept shape is not actually checked.
    # Multi-class test case
    clf = klass(alpha=0.01, max_iter=20)
    clf.fit(X2, Y2, coef_init=np.zeros((3, 2)), intercept_init=np.zeros(3))
    assert clf.coef_.shape == (3, 2)
    assert clf.intercept_.shape, (3,)
    pred = clf.predict(T2)
    assert_array_equal(pred, true_result2)


@pytest.mark.parametrize("klass", [SGDClassifier, SparseSGDClassifier])
def test_sgd_multiclass_njobs(klass):
    # [FUNCTION DOC]
    # Same checks as test_sgd_multiclass with n_jobs=2 (parallel OvA).
    # Multi-class test case with multi-core support
    clf = klass(alpha=0.01, max_iter=20, n_jobs=2).fit(X2, Y2)
    assert clf.coef_.shape == (3, 2)
    assert clf.intercept_.shape == (3,)
    assert clf.decision_function([[0, 0]]).shape == (1, 3)
    pred = clf.predict(T2)
    assert_array_equal(pred, true_result2)


@pytest.mark.parametrize("klass", [SGDClassifier, SparseSGDClassifier])
def test_set_coef_multiclass(klass):
    # [FUNCTION DOC]
    # Mismatched coef_init or intercept_init shapes raise ValueError on multiclass
    # data; correctly shaped ones are accepted.
    # Checks coef_init and intercept_init shape for multi-class
    # problems
    # Provided coef_ does not match dataset
    clf = klass()
    with pytest.raises(ValueError):
        clf.fit(X2, Y2, coef_init=np.zeros((2, 2)))

    # Provided coef_ does match dataset
    clf = klass().fit(X2, Y2, coef_init=np.zeros((3, 2)))

    # Provided intercept_ does not match dataset
    clf = klass()
    with pytest.raises(ValueError):
        clf.fit(X2, Y2, intercept_init=np.zeros((1,)))

    # Provided intercept_ does match dataset.
    clf = klass().fit(X2, Y2, intercept_init=np.zeros((3,)))


@pytest.mark.parametrize("klass", [SGDClassifier, SparseSGDClassifier])
def test_sgd_predict_proba_method_access(klass):
    # [FUNCTION DOC]
    # For every classifier loss, predict_proba/predict_log_proba are available only
    # for log_loss and modified_huber; otherwise attribute access raises
    # AttributeError whose __cause__ carries the "probability estimates are not
    # available" message.
    # KNOWN ISSUE: the klass parameter is unused (always the dense SGDClassifier
    # factory), so the sparse parametrization duplicates the dense test.
    # Checks that SGDClassifier predict_proba and predict_log_proba methods
    # can either be accessed or raise an appropriate error message
    # otherwise. See
    # https://github.com/scikit-learn/scikit-learn/issues/10938 for more
    # details.
    for loss in linear_model.SGDClassifier.loss_functions:
        clf = SGDClassifier(loss=loss)
        if loss in ("log_loss", "modified_huber"):
            assert hasattr(clf, "predict_proba")
            assert hasattr(clf, "predict_log_proba")
        else:
            inner_msg = "probability estimates are not available for loss={!r}".format(
                loss
            )
            assert not hasattr(clf, "predict_proba")
            assert not hasattr(clf, "predict_log_proba")
            with pytest.raises(
                AttributeError, match="has no attribute 'predict_proba'"
            ) as exec_info:
                clf.predict_proba

            assert isinstance(exec_info.value.__cause__, AttributeError)
            assert inner_msg in str(exec_info.value.__cause__)

            with pytest.raises(
                AttributeError, match="has no attribute 'predict_log_proba'"
            ) as exec_info:
                clf.predict_log_proba
            assert isinstance(exec_info.value.__cause__, AttributeError)
            assert inner_msg in str(exec_info.value.__cause__)


@pytest.mark.parametrize("klass", [SGDClassifier, SparseSGDClassifier])
def test_sgd_proba(klass):
    # [FUNCTION DOC]
    # Checks probability outputs: the hinge-loss model exposes no predict_proba; for
    # log_loss and modified_huber binary models the positive-class probability is
    # above/below 0.5 on clearly positive/negative points and log-probabilities are
    # ordered the same way; for multiclass log_loss the probability argmax/argsort
    # matches decision_function, probabilities sum to 1 and are non-negative, and
    # predict_log_proba equals log of predict_proba; for multiclass modified_huber
    # the argmax (argmin for sparse) agrees with the decision function and a point
    # with all decisions below -1 gets uniform probabilities.
    # KNOWN ISSUE: the sparse branch compares argmin instead of argmax with an
    # unexplained "XXX" comment, and the uniform-probability check is conditional, so
    # it may silently not run.
    # Check SGD.predict_proba

    # Hinge loss does not allow for conditional prob estimate.
    # We cannot use the factory here, because it defines predict_proba
    # anyway.
    clf = SGDClassifier(loss="hinge", alpha=0.01, max_iter=10, tol=None).fit(X, Y)
    assert not hasattr(clf, "predict_proba")
    assert not hasattr(clf, "predict_log_proba")

    # log and modified_huber losses can output probability estimates
    # binary case
    for loss in ["log_loss", "modified_huber"]:
        clf = klass(loss=loss, alpha=0.01, max_iter=10)
        clf.fit(X, Y)
        p = clf.predict_proba([[3, 2]])
        assert p[0, 1] > 0.5
        p = clf.predict_proba([[-1, -1]])
        assert p[0, 1] < 0.5

        # If predict_proba is 0, we get "RuntimeWarning: divide by zero encountered
        # in log". We avoid it here.
        with np.errstate(divide="ignore"):
            p = clf.predict_log_proba([[3, 2]])
            assert p[0, 1] > p[0, 0]
            p = clf.predict_log_proba([[-1, -1]])
            assert p[0, 1] < p[0, 0]

    # log loss multiclass probability estimates
    clf = klass(loss="log_loss", alpha=0.01, max_iter=10).fit(X2, Y2)

    d = clf.decision_function([[0.1, -0.1], [0.3, 0.2]])
    p = clf.predict_proba([[0.1, -0.1], [0.3, 0.2]])
    assert_array_equal(np.argmax(p, axis=1), np.argmax(d, axis=1))
    assert_almost_equal(p[0].sum(), 1)
    assert np.all(p[0] >= 0)

    p = clf.predict_proba([[-1, -1]])
    d = clf.decision_function([[-1, -1]])
    assert_array_equal(np.argsort(p[0]), np.argsort(d[0]))

    lp = clf.predict_log_proba([[3, 2]])
    p = clf.predict_proba([[3, 2]])
    assert_array_almost_equal(np.log(p), lp)

    lp = clf.predict_log_proba([[-1, -1]])
    p = clf.predict_proba([[-1, -1]])
    assert_array_almost_equal(np.log(p), lp)

    # Modified Huber multiclass probability estimates; requires a separate
    # test because the hard zero/one probabilities may destroy the
    # ordering present in decision_function output.
    clf = klass(loss="modified_huber", alpha=0.01, max_iter=10)
    clf.fit(X2, Y2)
    d = clf.decision_function([[3, 2]])
    p = clf.predict_proba([[3, 2]])
    if klass != SparseSGDClassifier:
        assert np.argmax(d, axis=1) == np.argmax(p, axis=1)
    else:  # XXX the sparse test gets a different X2 (?)
        assert np.argmin(d, axis=1) == np.argmin(p, axis=1)

    # the following sample produces decision_function values < -1,
    # which would cause naive normalization to fail (see comment
    # in SGDClassifier.predict_proba)
    x = X.mean(axis=0)
    d = clf.decision_function([x])
    if np.all(d < -1):  # XXX not true in sparse test case (why?)
        p = clf.predict_proba([x])
        assert_array_almost_equal(p[0], [1 / 3.0] * 3)


@pytest.mark.no_check_spmatrix  # pickle breaks check_spmatrix
@pytest.mark.parametrize("klass", [SGDClassifier, SparseSGDClassifier])
def test_sgd_l1(klass):
    # [FUNCTION DOC]
    # On shuffled X4 with strong L1 penalty, the middle (redundant) coefficients are
    # exactly zero and training predictions are perfect; after sparsify() coef_ is
    # sparse and predictions are unchanged, also after a pickle round trip.
    # Test L1 regularization
    n = len(X4)
    rng = np.random.RandomState(13)
    idx = np.arange(n)
    rng.shuffle(idx)

    X = X4[idx, :]
    Y = Y4[idx]

    clf = klass(
        penalty="l1",
        alpha=0.2,
        fit_intercept=False,
        max_iter=2000,
        tol=None,
        shuffle=False,
    )
    clf.fit(X, Y)
    assert_array_equal(clf.coef_[0, 1:-1], np.zeros((4,)))
    pred = clf.predict(X)
    assert_array_equal(pred, Y)

    # test sparsify with dense inputs
    clf.sparsify()
    assert sp.issparse(clf.coef_)
    pred = clf.predict(X)
    assert_array_equal(pred, Y)

    # pickle and unpickle with sparse coef_
    clf = pickle.loads(pickle.dumps(clf))
    assert sp.issparse(clf.coef_)
    pred = clf.predict(X)
    assert_array_equal(pred, Y)


@pytest.mark.parametrize("klass", [SGDClassifier, SparseSGDClassifier])
def test_class_weights(klass):
    # [FUNCTION DOC]
    # Down-weighting class 1 via class_weight={1: 0.001} flips the prediction of a
    # borderline point from 1 to -1 compared with the unweighted model.
    # Test class weights.
    X = np.array([[-1.0, -1.0], [-1.0, 0], [-0.8, -1.0], [1.0, 1.0], [1.0, 0.0]])
    y = [1, 1, 1, -1, -1]

    clf = klass(alpha=0.1, max_iter=1000, fit_intercept=False, class_weight=None)
    clf.fit(X, y)
    assert_array_equal(clf.predict([[0.2, -1.0]]), np.array([1]))

    # we give a small weights to class 1
    clf = klass(alpha=0.1, max_iter=1000, fit_intercept=False, class_weight={1: 0.001})
    clf.fit(X, y)

    # now the hyperplane should rotate clock-wise and
    # the prediction on this point should shift
    assert_array_equal(clf.predict([[0.2, -1.0]]), np.array([-1]))


@pytest.mark.parametrize("klass", [SGDClassifier, SparseSGDClassifier])
def test_equal_class_weight(klass):
    # [FUNCTION DOC]
    # Equal class weights of 0.5 on deduplicated data give coefficients close (2
    # decimals) to the unweighted model on the duplicated data.
    # Test if equal class weights approx. equals no class weights.
    X = [[1, 0], [1, 0], [0, 1], [0, 1]]
    y = [0, 0, 1, 1]
    clf = klass(alpha=0.1, max_iter=1000, class_weight=None)
    clf.fit(X, y)

    X = [[1, 0], [0, 1]]
    y = [0, 1]
    clf_weighted = klass(alpha=0.1, max_iter=1000, class_weight={0: 0.5, 1: 0.5})
    clf_weighted.fit(X, y)

    # should be similar up to some epsilon due to learning rate schedule
    assert_almost_equal(clf.coef_, clf_weighted.coef_, decimal=2)


@pytest.mark.parametrize("klass", [SGDClassifier, SparseSGDClassifier])
def test_wrong_class_weight_label(klass):
    # [FUNCTION DOC]
    # class_weight referring to a label absent from y raises ValueError.
    # ValueError due to not existing class label.
    clf = klass(alpha=0.1, max_iter=1000, class_weight={0: 0.5})
    with pytest.raises(ValueError):
        clf.fit(X, Y)


@pytest.mark.parametrize("klass", [SGDClassifier, SparseSGDClassifier])
def test_weights_multiplied(klass):
    # [FUNCTION DOC]
    # Fitting with class_weight plus sample_weight equals fitting with sample weights
    # pre-multiplied by the class weights.
    # Tests that class_weight and sample_weight are multiplicative
    class_weights = {1: 0.6, 2: 0.3}
    rng = np.random.RandomState(0)
    sample_weights = rng.random_sample(Y4.shape[0])
    multiplied_together = np.copy(sample_weights)
    multiplied_together[Y4 == 1] *= class_weights[1]
    multiplied_together[Y4 == 2] *= class_weights[2]

    clf1 = klass(alpha=0.1, max_iter=20, class_weight=class_weights)
    clf2 = klass(alpha=0.1, max_iter=20)

    clf1.fit(X4, Y4, sample_weight=sample_weights)
    clf2.fit(X4, Y4, sample_weight=multiplied_together)

    assert_almost_equal(clf1.coef_, clf2.coef_)


@pytest.mark.parametrize("klass", [SGDClassifier, SparseSGDClassifier])
def test_balanced_weight(klass):
    # [FUNCTION DOC]
    # On shuffled, scaled iris, "balanced" class weights give the same F1 (about 0.96)
    # and coefficients as no weighting; on an artificially imbalanced dataset (class 0
    # repeated 11 times) unweighted F1 drops below 0.96 while balanced exceeds it.
    # Test class weights for imbalanced data"""
    # compute reference metrics on iris dataset that is quite balanced by
    # default
    X, y = iris.data, iris.target
    X = scale(X)
    idx = np.arange(X.shape[0])
    rng = np.random.RandomState(6)
    rng.shuffle(idx)
    X = X[idx]
    y = y[idx]
    clf = klass(alpha=0.0001, max_iter=1000, class_weight=None, shuffle=False).fit(X, y)
    f1 = metrics.f1_score(y, clf.predict(X), average="weighted")
    assert_almost_equal(f1, 0.96, decimal=1)

    # make the same prediction using balanced class_weight
    clf_balanced = klass(
        alpha=0.0001, max_iter=1000, class_weight="balanced", shuffle=False
    ).fit(X, y)
    f1 = metrics.f1_score(y, clf_balanced.predict(X), average="weighted")
    assert_almost_equal(f1, 0.96, decimal=1)

    # Make sure that in the balanced case it does not change anything
    # to use "balanced"
    assert_array_almost_equal(clf.coef_, clf_balanced.coef_, 6)

    # build a very very imbalanced dataset out of iris data
    X_0 = X[y == 0, :]
    y_0 = y[y == 0]

    X_imbalanced = np.vstack([X] + [X_0] * 10)
    y_imbalanced = np.concatenate([y] + [y_0] * 10)

    # fit a model on the imbalanced data without class weight info
    clf = klass(max_iter=1000, class_weight=None, shuffle=False)
    clf.fit(X_imbalanced, y_imbalanced)
    y_pred = clf.predict(X)
    assert metrics.f1_score(y, y_pred, average="weighted") < 0.96

    # fit a model with balanced class_weight enabled
    clf = klass(max_iter=1000, class_weight="balanced", shuffle=False)
    clf.fit(X_imbalanced, y_imbalanced)
    y_pred = clf.predict(X)
    assert metrics.f1_score(y, y_pred, average="weighted") > 0.96


@pytest.mark.parametrize("klass", [SGDClassifier, SparseSGDClassifier])
def test_sample_weights(klass):
    # [FUNCTION DOC]
    # Small sample weights on class 1 samples flip the prediction of a borderline
    # point, mirroring test_class_weights.
    # Test weights on individual samples
    X = np.array([[-1.0, -1.0], [-1.0, 0], [-0.8, -1.0], [1.0, 1.0], [1.0, 0.0]])
    y = [1, 1, 1, -1, -1]

    clf = klass(alpha=0.1, max_iter=1000, fit_intercept=False)
    clf.fit(X, y)
    assert_array_equal(clf.predict([[0.2, -1.0]]), np.array([1]))

    # we give a small weights to class 1
    clf.fit(X, y, sample_weight=[0.001] * 3 + [1] * 2)

    # now the hyperplane should rotate clock-wise and
    # the prediction on this point should shift
    assert_array_equal(clf.predict([[0.2, -1.0]]), np.array([-1]))


@pytest.mark.parametrize(
    "klass", [SGDClassifier, SparseSGDClassifier, SGDOneClassSVM, SparseSGDOneClassSVM]
)
def test_wrong_sample_weights(klass):
    # [FUNCTION DOC]
    # A sample_weight longer than the number of samples raises ValueError for
    # classifiers and one-class SVMs.
    # Test if ValueError is raised if sample_weight has wrong shape
    if klass in [SGDClassifier, SparseSGDClassifier]:
        clf = klass(alpha=0.1, max_iter=1000, fit_intercept=False)
    elif klass in [SGDOneClassSVM, SparseSGDOneClassSVM]:
        clf = klass(nu=0.1, max_iter=1000, fit_intercept=False)
    # provided sample_weight too long
    with pytest.raises(ValueError):
        clf.fit(X, Y, sample_weight=np.arange(7))


@pytest.mark.parametrize("klass", [SGDClassifier, SparseSGDClassifier])
def test_partial_fit_exception(klass):
    # [FUNCTION DOC]
    # First partial_fit call without classes raises ValueError.
    clf = klass(alpha=0.01)
    # classes was not specified
    with pytest.raises(ValueError):
        clf.partial_fit(X3, Y3)


@pytest.mark.parametrize("klass", [SGDClassifier, SparseSGDClassifier])
def test_partial_fit_binary(klass):
    # [FUNCTION DOC]
    # Binary partial_fit on a third of the data sets coef_ (1, 2), intercept_ (1,) and
    # decision_function shape (1,); a second call continues training and predictions
    # on T are correct.
    # KNOWN ISSUE: "assert id1, id2" always passes, so the "coef_ not reallocated"
    # check is a no-op.
    third = X.shape[0] // 3
    clf = klass(alpha=0.01)
    classes = np.unique(Y)

    clf.partial_fit(X[:third], Y[:third], classes=classes)
    assert clf.coef_.shape == (1, X.shape[1])
    assert clf.intercept_.shape == (1,)
    assert clf.decision_function([[0, 0]]).shape == (1,)
    id1 = id(clf.coef_.data)

    clf.partial_fit(X[third:], Y[third:])
    id2 = id(clf.coef_.data)
    # check that coef_ haven't been re-allocated
    assert id1, id2

    y_pred = clf.predict(T)
    assert_array_equal(y_pred, true_result)


@pytest.mark.parametrize("klass", [SGDClassifier, SparseSGDClassifier])
def test_partial_fit_multiclass(klass):
    # [FUNCTION DOC]
    # Multiclass partial_fit gives coef_ (3, 2), intercept_ (3,), decision_function
    # (1, 3) and continues on a second call.
    # KNOWN ISSUE: same no-op "assert id1, id2" reallocation check.
    third = X2.shape[0] // 3
    clf = klass(alpha=0.01)
    classes = np.unique(Y2)

    clf.partial_fit(X2[:third], Y2[:third], classes=classes)
    assert clf.coef_.shape == (3, X2.shape[1])
    assert clf.intercept_.shape == (3,)
    assert clf.decision_function([[0, 0]]).shape == (1, 3)
    id1 = id(clf.coef_.data)

    clf.partial_fit(X2[third:], Y2[third:])
    id2 = id(clf.coef_.data)
    # check that coef_ haven't been re-allocated
    assert id1, id2


@pytest.mark.parametrize("klass", [SGDClassifier, SparseSGDClassifier])
def test_partial_fit_multiclass_average(klass):
    # [FUNCTION DOC]
    # Multiclass partial_fit with average=n_samples keeps correct coef_/intercept_
    # shapes over two calls.
    third = X2.shape[0] // 3
    clf = klass(alpha=0.01, average=X2.shape[0])
    classes = np.unique(Y2)

    clf.partial_fit(X2[:third], Y2[:third], classes=classes)
    assert clf.coef_.shape == (3, X2.shape[1])
    assert clf.intercept_.shape == (3,)

    clf.partial_fit(X2[third:], Y2[third:])
    assert clf.coef_.shape == (3, X2.shape[1])
    assert clf.intercept_.shape == (3,)


@pytest.mark.parametrize("klass", [SGDClassifier, SparseSGDClassifier])
def test_fit_then_partial_fit(klass):
    # [FUNCTION DOC]
    # Non-regression for issue 2496: partial_fit after fit on multiclass data does not
    # raise (fit used to produce Fortran-ordered coef_).
    # Partial_fit should work after initial fit in the multiclass case.
    # Non-regression test for #2496; fit would previously produce a
    # Fortran-ordered coef_ that subsequent partial_fit couldn't handle.
    clf = klass()
    clf.fit(X2, Y2)
    clf.partial_fit(X2, Y2)  # no exception here


@pytest.mark.parametrize("klass", [SGDClassifier, SparseSGDClassifier])
@pytest.mark.parametrize("lr", ["constant", "optimal", "invscaling", "adaptive"])
def test_partial_fit_equal_fit_classif(klass, lr):
    # [FUNCTION DOC]
    # For each learning-rate schedule and both binary and multiclass data, two
    # partial_fit passes equal a fit with max_iter=2: same t_ and decision values to 2
    # decimals.
    for X_, Y_, T_ in ((X, Y, T), (X2, Y2, T2)):
        clf = klass(alpha=0.01, eta0=0.01, max_iter=2, learning_rate=lr, shuffle=False)
        clf.fit(X_, Y_)
        y_pred = clf.decision_function(T_)
        t = clf.t_

        classes = np.unique(Y_)
        clf = klass(alpha=0.01, eta0=0.01, learning_rate=lr, shuffle=False)
        for i in range(2):
            clf.partial_fit(X_, Y_, classes=classes)
        y_pred2 = clf.decision_function(T_)

        assert clf.t_ == t
        assert_array_almost_equal(y_pred, y_pred2, decimal=2)


@pytest.mark.parametrize("klass", [SGDClassifier, SparseSGDClassifier])
def test_regression_losses(klass):
    # [FUNCTION DOC]
    # Classifier with regression losses (epsilon_insensitive,
    # squared_epsilon_insensitive, huber, squared_error) achieves perfect training
    # accuracy on the toy data.
    random_state = np.random.RandomState(1)
    clf = klass(
        alpha=0.01,
        learning_rate="constant",
        eta0=0.1,
        loss="epsilon_insensitive",
        random_state=random_state,
    )
    clf.fit(X, Y)
    assert 1.0 == np.mean(clf.predict(X) == Y)

    clf = klass(
        alpha=0.01,
        learning_rate="constant",
        eta0=0.1,
        loss="squared_epsilon_insensitive",
        random_state=random_state,
    )
    clf.fit(X, Y)
    assert 1.0 == np.mean(clf.predict(X) == Y)

    clf = klass(alpha=0.01, loss="huber", random_state=random_state)
    clf.fit(X, Y)
    assert 1.0 == np.mean(clf.predict(X) == Y)

    clf = klass(
        alpha=0.01,
        learning_rate="constant",
        eta0=0.01,
        loss="squared_error",
        random_state=random_state,
    )
    clf.fit(X, Y)
    assert 1.0 == np.mean(clf.predict(X) == Y)


@pytest.mark.parametrize("klass", [SGDClassifier, SparseSGDClassifier])
def test_warm_start_multiclass(klass):
    # [FUNCTION DOC]
    # Runs _test_warm_start on the 3-class X2/Y2 data with the optimal schedule.
    _test_warm_start(klass, X2, Y2, "optimal")


@pytest.mark.parametrize("klass", [SGDClassifier, SparseSGDClassifier])
def test_multiple_fit(klass):
    # [FUNCTION DOC]
    # Refitting the same estimator with fewer features and different (string) labels
    # works without error.
    # Test multiple calls of fit w/ different shaped inputs.
    clf = klass(alpha=0.01, shuffle=False)
    clf.fit(X, Y)
    assert hasattr(clf, "coef_")

    # Non-regression test: try fitting with a different label set.
    y = [["ham", "spam"][i] for i in LabelEncoder().fit_transform(Y)]
    clf.fit(X[:, :-1], y)


###############################################################################
# Regression Test Case


@pytest.mark.parametrize("klass", [SGDRegressor, SparseSGDRegressor])
def test_sgd_reg(klass):
    # [FUNCTION DOC]
    # Regressor on points along the diagonal yields equal coefficients for the two
    # identical features.
    # Check that SGD gives any results.
    clf = klass(alpha=0.1, max_iter=2, fit_intercept=False)
    clf.fit([[0, 0], [1, 1], [2, 2]], [0, 1, 2])
    assert clf.coef_[0] == clf.coef_[1]


@pytest.mark.parametrize("klass", [SGDRegressor, SparseSGDRegressor])
def test_sgd_averaged_computed_correctly(klass):
    # [FUNCTION DOC]
    # Averaged regressor (squared loss, constant rate, one epoch) matches the asgd
    # reference coefficients and intercept to 16 decimals.
    # Tests the average regressor matches the naive implementation

    eta = 0.001
    alpha = 0.01
    n_samples = 20
    n_features = 10
    rng = np.random.RandomState(0)
    X = rng.normal(size=(n_samples, n_features))
    w = rng.normal(size=n_features)

    # simple linear function without noise
    y = np.dot(X, w)

    clf = klass(
        loss="squared_error",
        learning_rate="constant",
        eta0=eta,
        alpha=alpha,
        fit_intercept=True,
        max_iter=1,
        average=True,
        shuffle=False,
    )

    clf.fit(X, y)
    average_weights, average_intercept = asgd(klass, X, y, eta, alpha)

    assert_array_almost_equal(clf.coef_, average_weights, decimal=16)
    assert_almost_equal(clf.intercept_, average_intercept, decimal=16)


@pytest.mark.parametrize("klass", [SGDRegressor, SparseSGDRegressor])
def test_sgd_averaged_partial_fit(klass):
    # [FUNCTION DOC]
    # Two partial_fit calls over halves of the data give the same averaged parameters
    # as the asgd reference over the full data.
    # Tests whether the partial fit yields the same average as the fit
    eta = 0.001
    alpha = 0.01
    n_samples = 20
    n_features = 10
    rng = np.random.RandomState(0)
    X = rng.normal(size=(n_samples, n_features))
    w = rng.normal(size=n_features)

    # simple linear function without noise
    y = np.dot(X, w)

    clf = klass(
        loss="squared_error",
        learning_rate="constant",
        eta0=eta,
        alpha=alpha,
        fit_intercept=True,
        max_iter=1,
        average=True,
        shuffle=False,
    )

    clf.partial_fit(X[: int(n_samples / 2)][:], y[: int(n_samples / 2)])
    clf.partial_fit(X[int(n_samples / 2) :][:], y[int(n_samples / 2) :])
    average_weights, average_intercept = asgd(klass, X, y, eta, alpha)

    assert_array_almost_equal(clf.coef_, average_weights, decimal=16)
    assert_almost_equal(clf.intercept_[0], average_intercept, decimal=16)


@pytest.mark.parametrize("klass", [SGDRegressor, SparseSGDRegressor])
def test_average_sparse(klass):
    # [FUNCTION DOC]
    # Same as the previous test on X3, which contains many zeros, exercising sparse
    # averaging with lazy updates.
    # Checks the average weights on data with 0s

    eta = 0.001
    alpha = 0.01
    clf = klass(
        loss="squared_error",
        learning_rate="constant",
        eta0=eta,
        alpha=alpha,
        fit_intercept=True,
        max_iter=1,
        average=True,
        shuffle=False,
    )

    n_samples = Y3.shape[0]

    clf.partial_fit(X3[: int(n_samples / 2)][:], Y3[: int(n_samples / 2)])
    clf.partial_fit(X3[int(n_samples / 2) :][:], Y3[int(n_samples / 2) :])
    average_weights, average_intercept = asgd(klass, X3, Y3, eta, alpha)

    assert_array_almost_equal(clf.coef_, average_weights, decimal=16)
    assert_almost_equal(clf.intercept_, average_intercept, decimal=16)


@pytest.mark.parametrize("klass", [SGDRegressor, SparseSGDRegressor])
def test_sgd_least_squares_fit(klass):
    # [FUNCTION DOC]
    # Squared-loss regressor fits y = 0.5x with R2 above 0.99 noiseless and above 0.5
    # with unit Gaussian noise.
    xmin, xmax = -5, 5
    n_samples = 100
    rng = np.random.RandomState(0)
    X = np.linspace(xmin, xmax, n_samples).reshape(n_samples, 1)

    # simple linear function without noise
    y = 0.5 * X.ravel()

    clf = klass(loss="squared_error", alpha=0.1, max_iter=20, fit_intercept=False)
    clf.fit(X, y)
    score = clf.score(X, y)
    assert score > 0.99

    # simple linear function with noise
    y = 0.5 * X.ravel() + rng.randn(n_samples, 1).ravel()

    clf = klass(loss="squared_error", alpha=0.1, max_iter=20, fit_intercept=False)
    clf.fit(X, y)
    score = clf.score(X, y)
    assert score > 0.5


@pytest.mark.parametrize("klass", [SGDRegressor, SparseSGDRegressor])
def test_sgd_epsilon_insensitive(klass):
    # [FUNCTION DOC]
    # Same R2 thresholds as the least-squares test using epsilon_insensitive loss with
    # epsilon=0.01.
    xmin, xmax = -5, 5
    n_samples = 100
    rng = np.random.RandomState(0)
    X = np.linspace(xmin, xmax, n_samples).reshape(n_samples, 1)

    # simple linear function without noise
    y = 0.5 * X.ravel()

    clf = klass(
        loss="epsilon_insensitive",
        epsilon=0.01,
        alpha=0.1,
        max_iter=20,
        fit_intercept=False,
    )
    clf.fit(X, y)
    score = clf.score(X, y)
    assert score > 0.99

    # simple linear function with noise
    y = 0.5 * X.ravel() + rng.randn(n_samples, 1).ravel()

    clf = klass(
        loss="epsilon_insensitive",
        epsilon=0.01,
        alpha=0.1,
        max_iter=20,
        fit_intercept=False,
    )
    clf.fit(X, y)
    score = clf.score(X, y)
    assert score > 0.5


@pytest.mark.parametrize("klass", [SGDRegressor, SparseSGDRegressor])
def test_sgd_huber_fit(klass):
    # [FUNCTION DOC]
    # Same R2 thresholds using huber loss with epsilon=0.1.
    xmin, xmax = -5, 5
    n_samples = 100
    rng = np.random.RandomState(0)
    X = np.linspace(xmin, xmax, n_samples).reshape(n_samples, 1)

    # simple linear function without noise
    y = 0.5 * X.ravel()

    clf = klass(loss="huber", epsilon=0.1, alpha=0.1, max_iter=20, fit_intercept=False)
    clf.fit(X, y)
    score = clf.score(X, y)
    assert score > 0.99

    # simple linear function with noise
    y = 0.5 * X.ravel() + rng.randn(n_samples, 1).ravel()

    clf = klass(loss="huber", epsilon=0.1, alpha=0.1, max_iter=20, fit_intercept=False)
    clf.fit(X, y)
    score = clf.score(X, y)
    assert score > 0.5


@pytest.mark.parametrize("klass", [SGDRegressor, SparseSGDRegressor])
def test_elasticnet_convergence(klass):
    # [FUNCTION DOC]
    # For alpha in {0.01, 0.001} and several l1_ratio values, SGD with elasticnet
    # penalty reaches coefficients within 2 decimals of coordinate-descent ElasticNet
    # on noiseless data.
    # Check that the SGD output is consistent with coordinate descent

    n_samples, n_features = 1000, 5
    rng = np.random.RandomState(0)
    X = rng.randn(n_samples, n_features)
    # ground_truth linear model that generate y from X and to which the
    # models should converge if the regularizer would be set to 0.0
    ground_truth_coef = rng.randn(n_features)
    y = np.dot(X, ground_truth_coef)

    # XXX: alpha = 0.1 seems to cause convergence problems
    for alpha in [0.01, 0.001]:
        for l1_ratio in [0.5, 0.8, 1.0]:
            cd = linear_model.ElasticNet(
                alpha=alpha, l1_ratio=l1_ratio, fit_intercept=False
            )
            cd.fit(X, y)
            sgd = klass(
                penalty="elasticnet",
                max_iter=50,
                alpha=alpha,
                l1_ratio=l1_ratio,
                fit_intercept=False,
            )
            sgd.fit(X, y)
            err_msg = (
                "cd and sgd did not converge to comparable "
                "results for alpha=%f and l1_ratio=%f" % (alpha, l1_ratio)
            )
            assert_almost_equal(cd.coef_, sgd.coef_, decimal=2, err_msg=err_msg)


@pytest.mark.parametrize("klass", [SGDRegressor, SparseSGDRegressor])
def test_partial_fit(klass):
    # [FUNCTION DOC]
    # Regressor partial_fit sets coef_ (n_features,), intercept_ (1,), predict shape
    # (1,) and continues on a second call.
    # KNOWN ISSUE: "assert id1, id2" always passes (no-op reallocation check).
    third = X.shape[0] // 3
    clf = klass(alpha=0.01)

    clf.partial_fit(X[:third], Y[:third])
    assert clf.coef_.shape == (X.shape[1],)
    assert clf.intercept_.shape == (1,)
    assert clf.predict([[0, 0]]).shape == (1,)
    id1 = id(clf.coef_.data)

    clf.partial_fit(X[third:], Y[third:])
    id2 = id(clf.coef_.data)
    # check that coef_ haven't been re-allocated
    assert id1, id2


@pytest.mark.parametrize("klass", [SGDRegressor, SparseSGDRegressor])
@pytest.mark.parametrize("lr", ["constant", "optimal", "invscaling", "adaptive"])
def test_partial_fit_equal_fit(klass, lr):
    # [FUNCTION DOC]
    # Regressor: two partial_fit passes equal fit with max_iter=2 (same t_, predictions
    # to 2 decimals) for each learning-rate schedule.
    clf = klass(alpha=0.01, max_iter=2, eta0=0.01, learning_rate=lr, shuffle=False)
    clf.fit(X, Y)
    y_pred = clf.predict(T)
    t = clf.t_

    clf = klass(alpha=0.01, eta0=0.01, learning_rate=lr, shuffle=False)
    for i in range(2):
        clf.partial_fit(X, Y)
    y_pred2 = clf.predict(T)

    assert clf.t_ == t
    assert_array_almost_equal(y_pred, y_pred2, decimal=2)


@pytest.mark.parametrize("klass", [SGDRegressor, SparseSGDRegressor])
def test_loss_function_epsilon(klass):
    # [FUNCTION DOC]
    # After set_params(epsilon=0.1), the huber entry in loss_functions reflects the new
    # epsilon (loss table is built from current params).
    clf = klass(epsilon=0.9)
    clf.set_params(epsilon=0.1)
    assert clf.loss_functions["huber"][1] == 0.1


###############################################################################
# SGD One Class SVM Test Case


# a simple implementation of ASGD to use for testing SGDOneClassSVM
def asgd_oneclass(klass, X, eta, nu, coef_init=None, offset_init=0.0):
    # [FUNCTION DOC]
    # HELPER: Reference averaged SGD for the one-class SVM objective, one pass in order.
    # intercept starts as 1 - offset_init. For each sample: score = dot(x, coef) +
    # intercept; hinge gradient is -1 if score <= 1 else 0; shrink coef by
    # max(0, 1 - eta*nu), subtract eta*gradient*x from coef, subtract
    # eta*(nu + gradient)*decay from intercept (decay 0.01 for the sparse factory);
    # update running means.
    # Inputs: klass factory; X (n, d); eta, nu floats; coef_init (d,) or None;
    # offset_init float or array.
    # Outputs: (average coef (d,), 1 - average intercept as the offset).
    # Side effects: mutates coef_init in place when provided.
    if coef_init is None:
        coef = np.zeros(X.shape[1])
    else:
        coef = coef_init

    average_coef = np.zeros(X.shape[1])
    offset = offset_init
    intercept = 1 - offset
    average_intercept = 0.0
    decay = 1.0

    # sparse data has a fixed decay of .01
    if klass == SparseSGDOneClassSVM:
        decay = 0.01

    for i, entry in enumerate(X):
        p = np.dot(entry, coef)
        p += intercept
        if p <= 1.0:
            gradient = -1
        else:
            gradient = 0
        coef *= max(0, 1.0 - eta * nu)
        coef += -(eta * gradient * entry)
        intercept += -(eta * (nu + gradient)) * decay

        average_coef *= i
        average_coef += coef
        average_coef /= i + 1.0

        average_intercept *= i
        average_intercept += intercept
        average_intercept /= i + 1.0

    return average_coef, 1 - average_intercept


@pytest.mark.parametrize("klass", [SGDOneClassSVM, SparseSGDOneClassSVM])
def _test_warm_start_oneclass(klass, X, lr):
    # [FUNCTION DOC]
    # HELPER: One-class analogue of _test_warm_start: explicit coef_init/offset_init
    # warm start equals warm_start=True after set_params(nu), comparing t_ and coef_.
    # KNOWN ISSUE: decorated with pytest.mark.parametrize although it is a helper not
    # collected as a test; the decorator has no effect and is misleading.
    # Test that explicit warm restart...
    clf = klass(nu=0.5, eta0=0.01, shuffle=False, learning_rate=lr)
    clf.fit(X)

    clf2 = klass(nu=0.1, eta0=0.01, shuffle=False, learning_rate=lr)
    clf2.fit(X, coef_init=clf.coef_.copy(), offset_init=clf.offset_.copy())

    # ... and implicit warm restart are equivalent.
    clf3 = klass(nu=0.5, eta0=0.01, shuffle=False, warm_start=True, learning_rate=lr)
    clf3.fit(X)

    assert clf3.t_ == clf.t_
    assert_allclose(clf3.coef_, clf.coef_)

    clf3.set_params(nu=0.1)
    clf3.fit(X)

    assert clf3.t_ == clf2.t_
    assert_allclose(clf3.coef_, clf2.coef_)


@pytest.mark.parametrize("klass", [SGDOneClassSVM, SparseSGDOneClassSVM])
@pytest.mark.parametrize("lr", ["constant", "optimal", "invscaling", "adaptive"])
def test_warm_start_oneclass(klass, lr):
    # [FUNCTION DOC]
    # Runs _test_warm_start_oneclass for dense/sparse models and all four schedules.
    _test_warm_start_oneclass(klass, X, lr)


@pytest.mark.parametrize("klass", [SGDOneClassSVM, SparseSGDOneClassSVM])
def test_clone_oneclass(klass):
    # [FUNCTION DOC]
    # Cloned one-class model with nu changed by set_params equals one built with that
    # nu.
    # Test whether clone works ok.
    clf = klass(nu=0.5)
    clf = clone(clf)
    clf.set_params(nu=0.1)
    clf.fit(X)

    clf2 = klass(nu=0.1)
    clf2.fit(X)

    assert_array_equal(clf.coef_, clf2.coef_)


@pytest.mark.parametrize("klass", [SGDOneClassSVM, SparseSGDOneClassSVM])
def test_partial_fit_oneclass(klass):
    # [FUNCTION DOC]
    # One-class partial_fit gives coef_ (2,), offset_ (1,), predict shape (1,); a
    # second call updates coef_ in place (identity check), and a feature-count
    # mismatch raises ValueError.
    third = X.shape[0] // 3
    clf = klass(nu=0.1)

    clf.partial_fit(X[:third])
    assert clf.coef_.shape == (X.shape[1],)
    assert clf.offset_.shape == (1,)
    assert clf.predict([[0, 0]]).shape == (1,)
    previous_coefs = clf.coef_

    clf.partial_fit(X[third:])
    # check that coef_ haven't been re-allocated
    assert clf.coef_ is previous_coefs

    # raises ValueError if number of features does not match previous data
    with pytest.raises(ValueError):
        clf.partial_fit(X[:, 1])


@pytest.mark.parametrize("klass", [SGDOneClassSVM, SparseSGDOneClassSVM])
@pytest.mark.parametrize("lr", ["constant", "optimal", "invscaling", "adaptive"])
def test_partial_fit_equal_fit_oneclass(klass, lr):
    # [FUNCTION DOC]
    # Two partial_fit passes equal fit with max_iter=2: t_, decision values, coef_ and
    # offset_ all match, per schedule.
    clf = klass(nu=0.05, max_iter=2, eta0=0.01, learning_rate=lr, shuffle=False)
    clf.fit(X)
    y_scores = clf.decision_function(T)
    t = clf.t_
    coef = clf.coef_
    offset = clf.offset_

    clf = klass(nu=0.05, eta0=0.01, max_iter=1, learning_rate=lr, shuffle=False)
    for _ in range(2):
        clf.partial_fit(X)
    y_scores2 = clf.decision_function(T)

    assert clf.t_ == t
    assert_allclose(y_scores, y_scores2)
    assert_allclose(clf.coef_, coef)
    assert_allclose(clf.offset_, offset)


@pytest.mark.parametrize("klass", [SGDOneClassSVM, SparseSGDOneClassSVM])
def test_late_onset_averaging_reached_oneclass(klass):
    # [FUNCTION DOC]
    # average=7 over two epochs equals asgd_oneclass run for one epoch from the
    # parameters of a non-averaged one-epoch model.
    # Test average
    eta0 = 0.001
    nu = 0.05

    # 2 passes over the training set but average only at second pass
    clf1 = klass(
        average=7, learning_rate="constant", eta0=eta0, nu=nu, max_iter=2, shuffle=False
    )
    # 1 pass over the training set with no averaging
    clf2 = klass(
        average=False,
        learning_rate="constant",
        eta0=eta0,
        nu=nu,
        max_iter=1,
        shuffle=False,
    )

    clf1.fit(X)
    clf2.fit(X)

    # Start from clf2 solution, compute averaging using asgd function and
    # compare with clf1 solution
    average_coef, average_offset = asgd_oneclass(
        klass, X, eta0, nu, coef_init=clf2.coef_.ravel(), offset_init=clf2.offset_
    )

    assert_allclose(clf1.coef_.ravel(), average_coef.ravel())
    assert_allclose(clf1.offset_, average_offset)


@pytest.mark.parametrize("klass", [SGDOneClassSVM, SparseSGDOneClassSVM])
def test_sgd_averaged_computed_correctly_oneclass(klass):
    # [FUNCTION DOC]
    # Averaged one-class fit for one epoch matches asgd_oneclass coef_ and offset_.
    # Tests the average SGD One-Class SVM matches the naive implementation
    eta = 0.001
    nu = 0.05
    n_samples = 20
    n_features = 10
    rng = np.random.RandomState(0)
    X = rng.normal(size=(n_samples, n_features))

    clf = klass(
        learning_rate="constant",
        eta0=eta,
        nu=nu,
        fit_intercept=True,
        max_iter=1,
        average=True,
        shuffle=False,
    )

    clf.fit(X)
    average_coef, average_offset = asgd_oneclass(klass, X, eta, nu)

    assert_allclose(clf.coef_, average_coef)
    assert_allclose(clf.offset_, average_offset)


@pytest.mark.parametrize("klass", [SGDOneClassSVM, SparseSGDOneClassSVM])
def test_sgd_averaged_partial_fit_oneclass(klass):
    # [FUNCTION DOC]
    # Two partial_fit halves give the same averaged parameters as asgd_oneclass on the
    # full data.
    # Tests whether the partial fit yields the same average as the fit
    eta = 0.001
    nu = 0.05
    n_samples = 20
    n_features = 10
    rng = np.random.RandomState(0)
    X = rng.normal(size=(n_samples, n_features))

    clf = klass(
        learning_rate="constant",
        eta0=eta,
        nu=nu,
        fit_intercept=True,
        max_iter=1,
        average=True,
        shuffle=False,
    )

    clf.partial_fit(X[: int(n_samples / 2)][:])
    clf.partial_fit(X[int(n_samples / 2) :][:])
    average_coef, average_offset = asgd_oneclass(klass, X, eta, nu)

    assert_allclose(clf.coef_, average_coef)
    assert_allclose(clf.offset_, average_offset)


@pytest.mark.parametrize("klass", [SGDOneClassSVM, SparseSGDOneClassSVM])
def test_average_sparse_oneclass(klass):
    # [FUNCTION DOC]
    # Same as above on X3 (data with many zeros).
    # Checks the average coef on data with 0s
    eta = 0.001
    nu = 0.01
    clf = klass(
        learning_rate="constant",
        eta0=eta,
        nu=nu,
        fit_intercept=True,
        max_iter=1,
        average=True,
        shuffle=False,
    )

    n_samples = X3.shape[0]

    clf.partial_fit(X3[: int(n_samples / 2)])
    clf.partial_fit(X3[int(n_samples / 2) :])
    average_coef, average_offset = asgd_oneclass(klass, X3, eta, nu)

    assert_allclose(clf.coef_, average_coef)
    assert_allclose(clf.offset_, average_offset)


def test_ocsvm_vs_sgdocsvm():
    # [FUNCTION DOC]
    # SGDOneClassSVM on a Nystroem RBF feature map approximates kernel OneClassSVM on
    # two Gaussian blobs: at least 99% prediction agreement and decision-function
    # correlation of at least 0.9 on test data.
    # Checks SGDOneClass SVM gives a good approximation of kernelized
    # One-Class SVM
    nu = 0.05
    gamma = 2.0
    random_state = 42

    # Generate train and test data
    rng = np.random.RandomState(random_state)
    X = 0.3 * rng.randn(500, 2)
    X_train = np.r_[X + 2, X - 2]
    X = 0.3 * rng.randn(100, 2)
    X_test = np.r_[X + 2, X - 2]

    # One-Class SVM
    clf = OneClassSVM(gamma=gamma, kernel="rbf", nu=nu)
    clf.fit(X_train)
    y_pred_ocsvm = clf.predict(X_test)
    dec_ocsvm = clf.decision_function(X_test).reshape(1, -1)

    # SGDOneClassSVM using kernel approximation
    max_iter = 15
    transform = Nystroem(gamma=gamma, random_state=random_state)
    clf_sgd = SGDOneClassSVM(
        nu=nu,
        shuffle=True,
        fit_intercept=True,
        max_iter=max_iter,
        random_state=random_state,
        tol=None,
    )
    pipe_sgd = make_pipeline(transform, clf_sgd)
    pipe_sgd.fit(X_train)
    y_pred_sgdocsvm = pipe_sgd.predict(X_test)
    dec_sgdocsvm = pipe_sgd.decision_function(X_test).reshape(1, -1)

    assert np.mean(y_pred_sgdocsvm == y_pred_ocsvm) >= 0.99
    corrcoef = np.corrcoef(np.concatenate((dec_ocsvm, dec_sgdocsvm)))[0, 1]
    assert corrcoef >= 0.9


def test_sgd_oneclass_convergence():
    # [FUNCTION DOC]
    # Non-regression for issue 30027: on iris, for several nu, the tol-based stopping
    # criterion does not stop before more than 6 epochs.
    # Check that the optimization does not end early and that the stopping criterion
    # is working. Non-regression test for #30027
    for nu in [0.1, 0.5, 0.9]:
        # no need for large max_iter
        model = SGDOneClassSVM(
            nu=nu, max_iter=100, tol=1e-3, learning_rate="constant", eta0=1e-3
        )
        model.fit(iris.data)
        # 6 is the minimal number of iterations that should be surpassed, after which
        # the optimization can stop
        assert model.n_iter_ > 6


@pytest.mark.parametrize("eta0, max_iter", [(1e-3, 10000), (3e-4, 20000)])
def test_sgd_oneclass_vs_linear_oneclass(eta0, max_iter):
    # [FUNCTION DOC]
    # For two (eta0, max_iter) settings and several nu, SGDOneClassSVM on iris
    # correlates with linear-kernel OneClassSVM in decision values (>0.99), predictions
    # (>0.95) and [coef, -offset] vs [coef, intercept] (>0.99), and the outlier share
    # matches nu within 0.01.
    # Test convergence vs. liblinear `OneClassSVM` with kernel="linear"
    for nu in [0.1, 0.5, 0.9]:
        # allow enough iterations, small dataset
        model = SGDOneClassSVM(
            nu=nu, max_iter=max_iter, tol=None, learning_rate="constant", eta0=eta0
        )
        model_ref = OneClassSVM(kernel="linear", nu=nu, tol=1e-6)  # reference model
        model.fit(iris.data)
        model_ref.fit(iris.data)

        preds = model.predict(iris.data)
        dec_fn = model.decision_function(iris.data)

        preds_ref = model_ref.predict(iris.data)
        dec_fn_ref = model_ref.decision_function(iris.data)

        dec_fn_corr = np.corrcoef(dec_fn, dec_fn_ref)[0, 1]
        preds_corr = np.corrcoef(preds, preds_ref)[0, 1]
        # check weights and intercept concatenated together for correlation
        coef_corr = np.corrcoef(
            np.concatenate([model.coef_, -model.offset_]),
            np.concatenate([model_ref.coef_.flatten(), model_ref.intercept_]),
        )[0, 1]
        # share of predicted 1's
        share_ones = (preds == 1).sum() / len(preds)

        assert dec_fn_corr > 0.99
        assert preds_corr > 0.95
        assert coef_corr > 0.99
        assert_allclose(1 - share_ones, nu, atol=1e-2)


@pytest.mark.parametrize("nu", [0.1, 0.9])
def test_sgd_oneclass_vs_linear_oneclass_offsets_match(nu):
    # [FUNCTION DOC]
    # On standardized iris, SGDOneClassSVM offset_ matches linear OneClassSVM offset_
    # within 1.3e-6 absolute tolerance for nu 0.1 and 0.9.
    # KNOWN ISSUE: the docstring credits a suitable eta0, but learning_rate="optimal"
    # does not use eta0.
    """Test that the `offset_` of `SGDOneClassSVM` is close to the `offset_`
    of `OneClassSVM` with `kernel="linear"`, given enough iterations and a
    suitable value for the `eta0` parameter, while also ensuring that the
    dataset is scaled.
    """
    X = iris.data
    X_scaled = StandardScaler().fit_transform(X)
    model = SGDOneClassSVM(
        nu=nu,
        max_iter=40000,
        tol=None,
        learning_rate="optimal",
        eta0=1e-6,
        random_state=42,
    )
    model_ref = OneClassSVM(kernel="linear", nu=nu, tol=5e-6)
    model.fit(X_scaled)
    model_ref.fit(X_scaled)
    assert_allclose(model.offset_, model_ref.offset_, atol=1.3e-6)


def test_l1_ratio():
    # [FUNCTION DOC]
    # Elasticnet with l1_ratio near 1 matches the pure L1 model and with l1_ratio near
    # 0 matches the pure L2 model (same seed and epochs).
    # Test if l1 ratio extremes match L1 and L2 penalty settings.
    X, y = datasets.make_classification(
        n_samples=1000, n_features=100, n_informative=20, random_state=1234
    )

    # test if elasticnet with l1_ratio near 1 gives same result as pure l1
    est_en = SGDClassifier(
        alpha=0.001,
        penalty="elasticnet",
        tol=None,
        max_iter=6,
        l1_ratio=0.9999999999,
        random_state=42,
    ).fit(X, y)
    est_l1 = SGDClassifier(
        alpha=0.001, penalty="l1", max_iter=6, random_state=42, tol=None
    ).fit(X, y)
    assert_array_almost_equal(est_en.coef_, est_l1.coef_)

    # test if elasticnet with l1_ratio near 0 gives same result as pure l2
    est_en = SGDClassifier(
        alpha=0.001,
        penalty="elasticnet",
        tol=None,
        max_iter=6,
        l1_ratio=0.0000000001,
        random_state=42,
    ).fit(X, y)
    est_l2 = SGDClassifier(
        alpha=0.001, penalty="l2", max_iter=6, random_state=42, tol=None
    ).fit(X, y)
    assert_array_almost_equal(est_en.coef_, est_l2.coef_)


def test_underflow_or_overflow():
    # [FUNCTION DOC]
    # With floating-point errors set to raise: a model on MinMax-scaled data with two
    # features of magnitude 1e300 fits with finite coefs, while fitting the unscaled
    # data raises ValueError suggesting scaling.
    with np.errstate(all="raise"):
        # Generate some weird data with hugely unscaled features
        rng = np.random.RandomState(0)
        n_samples = 100
        n_features = 10

        X = rng.normal(size=(n_samples, n_features))
        X[:, :2] *= 1e300
        assert np.isfinite(X).all()

        # Use MinMaxScaler to scale the data without introducing a numerical
        # instability (computing the standard deviation naively is not possible
        # on this data)
        X_scaled = MinMaxScaler().fit_transform(X)
        assert np.isfinite(X_scaled).all()

        # Define a ground truth on the scaled data
        ground_truth = rng.normal(size=n_features)
        y = (np.dot(X_scaled, ground_truth) > 0.0).astype(np.int32)
        assert_array_equal(np.unique(y), [0, 1])

        model = SGDClassifier(alpha=0.1, loss="squared_hinge", max_iter=500)

        # smoke test: model is stable on scaled data
        model.fit(X_scaled, y)
        assert np.isfinite(model.coef_).all()

        # model is numerically unstable on unscaled data
        msg_regxp = (
            r"Floating-point under-/overflow occurred at epoch #.*"
            " Scaling input data with StandardScaler or MinMaxScaler"
            " might help."
        )
        with pytest.raises(ValueError, match=msg_regxp):
            model.fit(X, y)


def test_numerical_stability_large_gradient():
    # [FUNCTION DOC]
    # squared_hinge elasticnet model on iris fits without floating-point errors (all
    # errors raise) and yields finite coefficients.
    # Non regression test case for numerical stability on scaled problems
    # where the gradient can still explode with some losses
    model = SGDClassifier(
        loss="squared_hinge",
        max_iter=10,
        shuffle=True,
        penalty="elasticnet",
        l1_ratio=0.3,
        alpha=0.01,
        eta0=0.001,
        random_state=0,
        tol=None,
    )
    with np.errstate(all="raise"):
        model.fit(iris.data, iris.target)
    assert np.isfinite(model.coef_).all()


@pytest.mark.parametrize("penalty", ["l2", "l1", "elasticnet"])
def test_large_regularization(penalty):
    # [FUNCTION DOC]
    # With alpha=1e5 and each penalty type, fitting raises no floating-point error and
    # coefficients collapse to zero.
    # Non regression tests for numerical stability issues caused by large
    # regularization parameters
    model = SGDClassifier(
        alpha=1e5,
        learning_rate="constant",
        eta0=0.1,
        penalty=penalty,
        shuffle=False,
        tol=None,
        max_iter=6,
    )
    with np.errstate(all="raise"):
        model.fit(iris.data, iris.target)
    assert_array_almost_equal(model.coef_, np.zeros_like(model.coef_))


def test_tol_parameter():
    # [FUNCTION DOC]
    # tol=None runs exactly max_iter epochs; tol=0 stops early but after more than 5;
    # a larger tol stops sooner (more than 3); max_iter=3 with tol=1e-3 emits
    # ConvergenceWarning with n_iter_ equal to 3.
    # Test that the tol parameter behaves as expected
    X = StandardScaler().fit_transform(iris.data)
    y = iris.target == 1

    # With tol is None, the number of iteration should be equal to max_iter
    max_iter = 42
    model_0 = SGDClassifier(tol=None, random_state=0, max_iter=max_iter)
    model_0.fit(X, y)
    assert max_iter == model_0.n_iter_

    # If tol is not None, the number of iteration should be less than max_iter
    max_iter = 2000
    model_1 = SGDClassifier(tol=0, random_state=0, max_iter=max_iter)
    model_1.fit(X, y)
    assert max_iter > model_1.n_iter_
    assert model_1.n_iter_ > 5

    # A larger tol should yield a smaller number of iteration
    model_2 = SGDClassifier(tol=0.1, random_state=0, max_iter=max_iter)
    model_2.fit(X, y)
    assert model_1.n_iter_ > model_2.n_iter_
    assert model_2.n_iter_ > 3

    # Strict tolerance and small max_iter should trigger a warning
    model_3 = SGDClassifier(max_iter=3, tol=1e-3, random_state=0)
    warning_message = (
        "Maximum number of iteration reached before "
        "convergence. Consider increasing max_iter to "
        "improve the fit."
    )
    with pytest.warns(ConvergenceWarning, match=warning_message):
        model_3.fit(X, y)
    assert model_3.n_iter_ == 3


def _test_loss_common(loss_function, cases):
    # [FUNCTION DOC]
    # HELPER: For each (p, y, expected_loss, expected_dloss) case, asserts the Cython
    # loss object's py_loss and py_dloss match the expected values.
    # Test the different loss functions
    # cases is a list of (p, y, expected)
    for p, y, expected_loss, expected_dloss in cases:
        assert_almost_equal(loss_function.py_loss(p, y), expected_loss)
        assert_almost_equal(loss_function.py_dloss(p, y), expected_dloss)


def test_loss_hinge():
    # [FUNCTION DOC]
    # Table tests for Hinge with threshold 1 (hinge) and 0 (perceptron), including
    # boundary points where the derivative is still nonzero.
    # Test Hinge (hinge / perceptron)
    # hinge
    loss = sgd_fast.Hinge(1.0)
    cases = [
        # (p, y, expected_loss, expected_dloss)
        (1.1, 1.0, 0.0, 0.0),
        (-2.0, -1.0, 0.0, 0.0),
        (1.0, 1.0, 0.0, -1.0),
        (-1.0, -1.0, 0.0, 1.0),
        (0.5, 1.0, 0.5, -1.0),
        (2.0, -1.0, 3.0, 1.0),
        (-0.5, -1.0, 0.5, 1.0),
        (0.0, 1.0, 1, -1.0),
    ]
    _test_loss_common(loss, cases)

    # perceptron
    loss = sgd_fast.Hinge(0.0)
    cases = [
        # (p, y, expected_loss, expected_dloss)
        (1.0, 1.0, 0.0, 0.0),
        (-0.1, -1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, -1.0),
        (0.0, -1.0, 0.0, 1.0),
        (0.5, -1.0, 0.5, 1.0),
        (2.0, -1.0, 2.0, 1.0),
        (-0.5, 1.0, 0.5, -1.0),
        (-1.0, 1.0, 1.0, -1.0),
    ]
    _test_loss_common(loss, cases)


def test_gradient_squared_hinge():
    # [FUNCTION DOC]
    # Table tests for SquaredHinge loss and gradient values.
    # Test SquaredHinge
    loss = sgd_fast.SquaredHinge(1.0)
    cases = [
        # (p, y, expected_loss, expected_dloss)
        (1.0, 1.0, 0.0, 0.0),
        (-2.0, -1.0, 0.0, 0.0),
        (1.0, -1.0, 4.0, 4.0),
        (-1.0, 1.0, 4.0, -4.0),
        (0.5, 1.0, 0.25, -1.0),
        (0.5, -1.0, 2.25, 3.0),
    ]
    _test_loss_common(loss, cases)


def test_loss_modified_huber():
    # [FUNCTION DOC]
    # Table tests for ModifiedHuber: zero loss beyond the margin, quadratic region and
    # linear region (loss -4 z, gradient -4 y) for margins below -1.
    # (p, y, expected_loss, expected_dloss)
    loss = sgd_fast.ModifiedHuber()
    cases = [
        # (p, y, expected_loss, expected_dloss)
        (1.0, 1.0, 0.0, 0.0),
        (-1.0, -1.0, 0.0, 0.0),
        (2.0, 1.0, 0.0, 0.0),
        (0.0, 1.0, 1.0, -2.0),
        (-1.0, 1.0, 4.0, -4.0),
        (0.5, -1.0, 2.25, 3.0),
        (-2.0, 1.0, 8, -4.0),
        (-3.0, 1.0, 12, -4.0),
    ]
    _test_loss_common(loss, cases)


def test_loss_epsilon_insensitive():
    # [FUNCTION DOC]
    # Table tests for EpsilonInsensitive(0.1): zero inside the epsilon tube, linear
    # excess and sign gradient outside.
    # Test EpsilonInsensitive
    loss = sgd_fast.EpsilonInsensitive(0.1)
    cases = [
        # (p, y, expected_loss, expected_dloss)
        (0.0, 0.0, 0.0, 0.0),
        (0.1, 0.0, 0.0, 0.0),
        (-2.05, -2.0, 0.0, 0.0),
        (3.05, 3.0, 0.0, 0.0),
        (2.2, 2.0, 0.1, 1.0),
        (2.0, -1.0, 2.9, 1.0),
        (2.0, 2.2, 0.1, -1.0),
        (-2.0, 1.0, 2.9, -1.0),
    ]
    _test_loss_common(loss, cases)


def test_loss_squared_epsilon_insensitive():
    # [FUNCTION DOC]
    # Table tests for SquaredEpsilonInsensitive(0.1): squared excess beyond epsilon and
    # gradient twice the signed excess.
    # Test SquaredEpsilonInsensitive
    loss = sgd_fast.SquaredEpsilonInsensitive(0.1)
    cases = [
        # (p, y, expected_loss, expected_dloss)
        (0.0, 0.0, 0.0, 0.0),
        (0.1, 0.0, 0.0, 0.0),
        (-2.05, -2.0, 0.0, 0.0),
        (3.05, 3.0, 0.0, 0.0),
        (2.2, 2.0, 0.01, 0.2),
        (2.0, -1.0, 8.41, 5.8),
        (2.0, 2.2, 0.01, -0.2),
        (-2.0, 1.0, 8.41, -5.8),
    ]
    _test_loss_common(loss, cases)


def test_multi_thread_multi_class_and_early_stopping():
    # [FUNCTION DOC]
    # Non-regression: multiclass early stopping with n_jobs=2 stops shortly after
    # n_iter_no_change epochs (strictly between 100 and 120) and scores above 0.8.
    # This is a non-regression test for a bad interaction between
    # early stopping internal attribute and thread-based parallelism.
    clf = SGDClassifier(
        alpha=1e-3,
        tol=1e-3,
        max_iter=1000,
        early_stopping=True,
        n_iter_no_change=100,
        random_state=0,
        n_jobs=2,
    )
    clf.fit(iris.data, iris.target)
    assert clf.n_iter_ > clf.n_iter_no_change
    assert clf.n_iter_ < clf.n_iter_no_change + 20
    assert clf.score(iris.data, iris.target) > 0.8


def test_multi_core_gridsearch_and_early_stopping():
    # [FUNCTION DOC]
    # Non-regression: RandomizedSearchCV with n_jobs=2 over alpha and n_iter_no_change
    # on an early-stopping classifier runs and reaches best_score_ above 0.8.
    # This is a non-regression test for a bad interaction between
    # early stopping internal attribute and process-based multi-core
    # parallelism.
    param_grid = {
        "alpha": np.logspace(-4, 4, 9),
        "n_iter_no_change": [5, 10, 50],
    }

    clf = SGDClassifier(tol=1e-2, max_iter=1000, early_stopping=True, random_state=0)
    search = RandomizedSearchCV(clf, param_grid, n_iter=5, n_jobs=2, random_state=0)
    search.fit(iris.data, iris.target)
    assert search.best_score_ > 0.8


@pytest.mark.parametrize("backend", ["loky", "multiprocessing", "threading"])
def test_SGDClassifier_fit_for_all_backends(backend):
    # [FUNCTION DOC]
    # Smoke test that multiclass fit under loky, multiprocessing and threading joblib
    # backends (n_jobs=4) yields the same coef_ as a sequential fit on sparse data
    # with 20 classes, guarding against writes into read-only memmaps.
    # KNOWN ISSUE: the comment says 50000 features, but the data has 2000 features,
    # so coef_ may not exceed joblib's memmapping threshold as intended.
    # This is a non-regression smoke test. In the multi-class case,
    # SGDClassifier.fit fits each class in a one-versus-all fashion using
    # joblib.Parallel.  However, each OvA step updates the coef_ attribute of
    # the estimator in-place. Internally, SGDClassifier calls Parallel using
    # require='sharedmem'. This test makes sure SGDClassifier.fit works
    # consistently even when the user asks for a backend that does not provide
    # sharedmem semantics.

    # We further test a case where memmapping would have been used if
    # SGDClassifier.fit was called from a loky or multiprocessing backend. In
    # this specific case, in-place modification of clf.coef_ would have caused
    # a segmentation fault when trying to write in a readonly memory mapped
    # buffer.

    random_state = np.random.RandomState(42)

    # Create a classification problem with 50000 features and 20 classes. Using
    # loky or multiprocessing this make the clf.coef_ exceed the threshold
    # above which memmaping is used in joblib and loky (1MB as of 2018/11/1).
    X = _sparse_random_array((500, 2000), density=0.02, format="csr", rng=random_state)
    y = random_state.choice(20, 500)

    # Begin by fitting an SGD classifier sequentially
    clf_sequential = SGDClassifier(max_iter=1000, n_jobs=1, random_state=42)
    clf_sequential.fit(X, y)

    # Fit an SGDClassifier using the specified backend, and make sure the
    # coefficients are equal to those obtained using a sequential fit
    clf_parallel = SGDClassifier(max_iter=1000, n_jobs=4, random_state=42)
    with joblib.parallel_backend(backend=backend):
        clf_parallel.fit(X, y)
    assert_array_almost_equal(clf_sequential.coef_, clf_parallel.coef_)


@pytest.mark.parametrize(
    "Estimator", [linear_model.SGDClassifier, linear_model.SGDRegressor]
)
def test_sgd_random_state(Estimator, global_random_seed):
    # [FUNCTION DOC]
    # For classifier and regressor with one epoch (ConvergenceWarning expected), the
    # same seed gives identical coef_ and a different seed changes coef_ by more than
    # 1 in some entry.
    # Train the same model on the same data without converging and check that we
    # get reproducible results by fixing the random seed.
    if Estimator == linear_model.SGDRegressor:
        X, y = datasets.make_regression(random_state=global_random_seed)
    else:
        X, y = datasets.make_classification(random_state=global_random_seed)

    # Fitting twice a model with the same hyper-parameters on the same training
    # set with the same seed leads to the same results deterministically.

    est = Estimator(random_state=global_random_seed, max_iter=1)
    with pytest.warns(ConvergenceWarning):
        coef_same_seed_a = est.fit(X, y).coef_
        assert est.n_iter_ == 1

    est = Estimator(random_state=global_random_seed, max_iter=1)
    with pytest.warns(ConvergenceWarning):
        coef_same_seed_b = est.fit(X, y).coef_
        assert est.n_iter_ == 1

    assert_allclose(coef_same_seed_a, coef_same_seed_b)

    # Fitting twice a model with the same hyper-parameters on the same training
    # set but with different random seed leads to different results after one
    # epoch because of the random shuffling of the dataset.

    est = Estimator(random_state=global_random_seed + 1, max_iter=1)
    with pytest.warns(ConvergenceWarning):
        coef_other_seed = est.fit(X, y).coef_
        assert est.n_iter_ == 1

    assert np.abs(coef_same_seed_a - coef_other_seed).max() > 1.0


def test_validation_mask_correctly_subsets(monkeypatch):
    # [FUNCTION DOC]
    # Non-regression for issue 23255: monkeypatches _ValidationScoreCallback with a
    # Mock wrapper and checks the validation X and y passed to it have
    # validation_fraction times n_samples rows.
    # Side effects: patches the module attribute via monkeypatch (restored after).
    """Test that data passed to validation callback correctly subsets.

    Non-regression test for #23255.
    """
    X, Y = iris.data, iris.target
    n_samples = X.shape[0]
    validation_fraction = 0.2
    clf = linear_model.SGDClassifier(
        early_stopping=True,
        tol=1e-3,
        max_iter=1000,
        validation_fraction=validation_fraction,
    )

    mock = Mock(side_effect=_stochastic_gradient._ValidationScoreCallback)
    monkeypatch.setattr(_stochastic_gradient, "_ValidationScoreCallback", mock)
    clf.fit(X, Y)

    X_val, y_val = mock.call_args[0][1:3]
    assert X_val.shape[0] == int(n_samples * validation_fraction)
    assert y_val.shape[0] == int(n_samples * validation_fraction)


def test_sgd_error_on_zero_validation_weight():
    # [FUNCTION DOC]
    # Non-regression for issue 17229: all-zero sample weights make the validation set
    # weights all zero and fit raises ValueError with the expected message.
    # Test that SGDClassifier raises error when all the validation samples
    # have zero sample_weight. Non-regression test for #17229.
    X, Y = iris.data, iris.target
    sample_weight = np.zeros_like(Y)
    validation_fraction = 0.4

    clf = linear_model.SGDClassifier(
        early_stopping=True, validation_fraction=validation_fraction, random_state=0
    )

    error_message = (
        "The sample weights for validation set are all zero, consider using a"
        " different random state."
    )
    with pytest.raises(ValueError, match=error_message):
        clf.fit(X, Y, sample_weight=sample_weight)


@pytest.mark.parametrize("Estimator", [SGDClassifier, SGDRegressor])
def test_sgd_verbose(Estimator):
    # [FUNCTION DOC]
    # Non-regression for issue 25249: fitting with verbose=1 runs without error.
    """non-regression test for gh #25249"""
    Estimator(verbose=1).fit(X, Y)


@pytest.mark.parametrize(
    "SGDEstimator",
    [
        SGDClassifier,
        SparseSGDClassifier,
        SGDRegressor,
        SparseSGDRegressor,
        SGDOneClassSVM,
        SparseSGDOneClassSVM,
    ],
)
@pytest.mark.parametrize("data_type", (np.float32, np.float64))
def test_sgd_dtype_match(SGDEstimator, data_type):
    # [FUNCTION DOC]
    # For each estimator and float32/float64 input, coef_ keeps the input dtype.
    _X = X.astype(data_type)
    _Y = np.array(Y, dtype=data_type)
    sgd_model = SGDEstimator()
    sgd_model.fit(_X, _Y)
    assert sgd_model.coef_.dtype == data_type


@pytest.mark.parametrize(
    "SGDEstimator",
    [
        SGDClassifier,
        SparseSGDClassifier,
        SGDRegressor,
        SparseSGDRegressor,
        SGDOneClassSVM,
        SparseSGDOneClassSVM,
    ],
)
def test_sgd_numerical_consistency(SGDEstimator):
    # [FUNCTION DOC]
    # Fitting the same model on float64 and float32 data gives coef_ equal within
    # default tolerance.
    X_64 = X.astype(dtype=np.float64)
    Y_64 = np.array(Y, dtype=np.float64)

    X_32 = X.astype(dtype=np.float32)
    Y_32 = np.array(Y, dtype=np.float32)

    sgd_64 = SGDEstimator(max_iter=22, shuffle=False)
    sgd_64.fit(X_64, Y_64)

    sgd_32 = SGDEstimator(max_iter=22, shuffle=False)
    sgd_32.fit(X_32, Y_32)

    assert_allclose(sgd_64.coef_, sgd_32.coef_)


def test_sgd_one_class_svm_estimator_type():
    # [FUNCTION DOC]
    # SGDOneClassSVM tags report estimator_type "outlier_detector" (mixin order).
    """Check that SGDOneClassSVM has the correct estimator type.

    Non-regression test for if the mixin was not on the left.
    """
    sgd_ocsvm = SGDOneClassSVM()
    assert get_tags(sgd_ocsvm).estimator_type == "outlier_detector"


def test_sgd_one_class_svm_formulation_with_scipy_minimize():
    # [FUNCTION DOC]
    # Minimizes the one-class objective (mean hinge on all-positive labels plus L2 term
    # plus nu times intercept) with Nelder-Mead and checks a long constant-rate
    # SGDOneClassSVM fit reaches matching coef_ (rtol 5e-3) and offset_ = 1 minus the
    # intercept (rtol 1e-2).
    """Test that SGDOneClassSVM minimizes the correct objective function."""
    nu = 0.5
    hinge_threshold = 1.0
    n_samples, n_features = 300, 3
    random_seed = 42

    def objective(w, X, y, alpha):
        # [FUNCTION DOC]
        # HELPER: Objective for scipy: splits w into weights and intercept, computes the
        # mean of max(hinge_threshold - y times score, 0), adds 0.5 alpha ||weights||^2 and
        # alpha times the intercept. Outputs: float.
        weights = w[:-1]
        intercept = w[-1]
        p = X @ weights + intercept
        z = p * y
        avg_loss = np.mean(np.maximum(hinge_threshold - z, 0.0))
        reg = 0.5 * alpha * weights @ weights
        obj = avg_loss + reg + intercept * alpha
        return obj

    X, _ = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        random_state=random_seed,
    )
    y = np.ones(n_samples, dtype=X.dtype)
    w0 = np.zeros(n_features + 1)
    scipy_output = minimize(
        objective,
        w0,
        method="Nelder-Mead",
        args=(X, y, nu),
        options={"maxiter": 1000},
    )
    w_out = scipy_output.x
    expected_coef = w_out[:-1]
    expected_offset = 1 - w_out[-1]

    model = SGDOneClassSVM(
        nu=nu,
        learning_rate="constant",
        max_iter=4000,
        tol=None,
        eta0=1e-4,
        random_state=random_seed,
    )
    model.fit(X, y)

    assert_allclose(model.coef_, expected_coef, rtol=5e-3)
    assert_allclose(model.offset_, expected_offset, rtol=1e-2)
