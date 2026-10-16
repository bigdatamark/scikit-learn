# [MODULE DOC]
# Test suite for sklearn.model_selection._search: ParameterGrid, ParameterSampler,
#   GridSearchCV, RandomizedSearchCV, the Halving*SearchCV variants and BaseSearchCV.
# Covers parameter validation, cv_results_ structure/dtypes/masking, refit (bool, str,
#   callable), multi-metric and callable scoring, error_score handling, sample_weight,
#   metadata routing, pairwise tags, HTML repr, array API dispatch, and the callback
#   hooks.
# Module-level fixtures: a tiny 4-sample dataset X (2 features) and y (labels 1/2)
#   shared by many tests; one_hot_encoder, ordinal_encoder and ma_with_tuples are module
#   objects reused as parametrize inputs and expectations.
# Side effects at import: enables experimental halving search via
#   enable_halving_search_cv.
"""Test the search module"""

import pickle
import re
import sys
import warnings
from collections.abc import Iterable, Sized
from functools import partial
from io import StringIO
from itertools import chain, product
from types import GeneratorType

import numpy as np
import pytest
from scipy.stats import bernoulli, expon, randint, uniform

from sklearn import config_context
from sklearn.base import BaseEstimator, ClassifierMixin, clone, is_classifier
from sklearn.callback.tests._utils import (
    MaxIterEstimator,
    NoCallbackEstimator,
    RecordingAutoPropagatedCallback,
    RecordingCallback,
    skip_callback_test_if_wasm,
)
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.datasets import (
    make_blobs,
    make_classification,
    make_multilabel_classification,
)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.exceptions import FitFailedWarning
from sklearn.experimental import enable_halving_search_cv  # noqa: F401
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import (
    LinearRegression,
    LogisticRegression,
    Ridge,
    SGDClassifier,
)
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    make_scorer,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.model_selection import (
    GridSearchCV,
    GroupKFold,
    GroupShuffleSplit,
    HalvingGridSearchCV,
    HalvingRandomSearchCV,
    KFold,
    LeaveOneGroupOut,
    LeavePGroupsOut,
    ParameterGrid,
    ParameterSampler,
    RandomizedSearchCV,
    StratifiedKFold,
    StratifiedShuffleSplit,
    train_test_split,
)
from sklearn.model_selection._search import (
    BaseSearchCV,
    _yield_masked_array_for_each_param,
)
from sklearn.model_selection.tests.common import OneTimeSplitter
from sklearn.naive_bayes import ComplementNB
from sklearn.neighbors import KernelDensity, KNeighborsClassifier, LocalOutlierFactor
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    SplineTransformer,
    StandardScaler,
)
from sklearn.svm import SVC, LinearSVC
from sklearn.tests.metadata_routing_common import (
    ConsumingScorer,
    _Registry,
    check_recorded_metadata,
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils._array_api import (
    yield_namespace_device_dtype_combinations,
)
from sklearn.utils._mocking import CheckingClassifier, MockDataFrame
from sklearn.utils._testing import (
    MinimalClassifier,
    MinimalRegressor,
    MinimalTransformer,
    _array_api_for_tests,
    assert_allclose,
    assert_allclose_dense_sparse,
    assert_almost_equal,
    assert_array_almost_equal,
    assert_array_equal,
    set_random_state,
)
from sklearn.utils.estimator_checks import _enforce_estimator_tags_y
from sklearn.utils.fixes import CSR_CONTAINERS
from sklearn.utils.validation import _num_samples


# Neither of the following two estimators inherit from BaseEstimator,
# to test hyperparameter search on user-defined classifiers.
class MockClassifier(ClassifierMixin, BaseEstimator):
    # [CLASS DOC]
    # HELPER: minimal classifier with one hyperparameter foo_param; its score is 1.0
    #   when foo_param > 1 else 0.0, so the best grid point is the smallest foo_param
    #   above 1 (tie-breaking checks).
    # KNOWN ISSUE: the comment above says neither estimator inherits from BaseEstimator,
    #   but both MockClassifier and LinearSVCNoScore do.
    """Dummy classifier to test the parameter search algorithms"""

    def __init__(self, foo_param=0):
        # [FUNCTION DOC]
        # HELPER: stores foo_param (default 0) unchanged.
        self.foo_param = foo_param

    def fit(self, X, Y):
        # [FUNCTION DOC]
        # HELPER: asserts X and Y have the same length, sets classes_ to the unique
        #   labels of Y, returns self.
        # Exceptions: AssertionError on length mismatch.
        assert len(X) == len(Y)
        self.classes_ = np.unique(Y)
        return self

    def predict(self, T):
        # [FUNCTION DOC]
        # HELPER: returns the number of rows of T (an int, not an array); also aliased
        #   as predict_proba, predict_log_proba and decision_function.
        return T.shape[0]

    def transform(self, X):
        # [FUNCTION DOC]
        # HELPER: returns X shifted by adding foo_param elementwise.
        return X + self.foo_param

    def inverse_transform(self, X):
        # [FUNCTION DOC]
        # HELPER: returns X shifted by subtracting foo_param elementwise (inverse of
        #   transform).
        return X - self.foo_param

    predict_proba = predict
    predict_log_proba = predict
    decision_function = predict

    def score(self, X=None, Y=None):
        # [FUNCTION DOC]
        # HELPER: ignores X and Y; returns 1.0 if foo_param > 1 else 0.0.
        if self.foo_param > 1:
            score = 1.0
        else:
            score = 0.0
        return score

    def get_params(self, deep=False):
        # [FUNCTION DOC]
        # HELPER: returns a dict with only foo_param; deep is ignored.
        return {"foo_param": self.foo_param}

    def set_params(self, **params):
        # [FUNCTION DOC]
        # HELPER: sets foo_param from params and returns self.
        # Exceptions: KeyError if foo_param is not among params; other keys are silently
        #   ignored.
        self.foo_param = params["foo_param"]
        return self


class LinearSVCNoScore(LinearSVC):
    # [CLASS DOC]
    # HELPER: LinearSVC whose score attribute is hidden so that hasattr(est, "score") is
    #   False, used to test search behaviour without an estimator score method.
    """A LinearSVC classifier that has no score method."""

    @property
    def score(self):
        # [FUNCTION DOC]
        # HELPER: property that always raises AttributeError, hiding the inherited score
        #   method.
        raise AttributeError


X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
y = np.array([1, 1, 2, 2])


def assert_grid_iter_equals_getitem(grid):
    # [FUNCTION DOC]
    # HELPER: asserts that iterating over grid yields the same sequence as indexing
    #   grid[i] for i in range(len(grid)).
    assert list(grid) == [grid[i] for i in range(len(grid))]


@pytest.mark.parametrize("klass", [ParameterGrid, partial(ParameterSampler, n_iter=10)])
@pytest.mark.parametrize(
    "input, error_type, error_message",
    [
        (0, TypeError, r"Parameter .* a dict or a list, got: 0 of type int"),
        ([{"foo": [0]}, 0], TypeError, r"Parameter .* is not a dict \(0\)"),
        (
            {"foo": 0},
            TypeError,
            r"Parameter (grid|distribution) for parameter 'foo' (is not|needs to be) "
            r"(a list or a numpy array|iterable or a distribution).*",
        ),
    ],
)
def test_validate_parameter_input(klass, input, error_type, error_message):
    # [FUNCTION DOC]
    # Verifies that ParameterGrid and ParameterSampler reject invalid param
    #   specifications (non dict/list, list containing non-dict, non-iterable values)
    #   with the expected exception type and message.
    with pytest.raises(error_type, match=error_message):
        klass(input)


def test_parameter_grid():
    # [FUNCTION DOC]
    # Verifies ParameterGrid basics: Iterable/Sized, length equals product of value list
    #   sizes, repeated iteration yields the full cartesian product, an empty dict
    #   yields a single empty point (IndexError beyond it), and a list of grids
    #   including an empty dict is concatenated in order.
    # Test basic properties of ParameterGrid.
    params1 = {"foo": [1, 2, 3]}
    grid1 = ParameterGrid(params1)
    assert isinstance(grid1, Iterable)
    assert isinstance(grid1, Sized)
    assert len(grid1) == 3
    assert_grid_iter_equals_getitem(grid1)

    params2 = {"foo": [4, 2], "bar": ["ham", "spam", "eggs"]}
    grid2 = ParameterGrid(params2)
    assert len(grid2) == 6

    # loop to assert we can iterate over the grid multiple times
    for i in range(2):
        # tuple + chain transforms {"a": 1, "b": 2} to ("a", 1, "b", 2)
        points = set(tuple(chain(*(sorted(p.items())))) for p in grid2)
        assert points == set(
            ("bar", x, "foo", y) for x, y in product(params2["bar"], params2["foo"])
        )
    assert_grid_iter_equals_getitem(grid2)

    # Special case: empty grid (useful to get default estimator settings)
    empty = ParameterGrid({})
    assert len(empty) == 1
    assert list(empty) == [{}]
    assert_grid_iter_equals_getitem(empty)
    with pytest.raises(IndexError):
        empty[1]

    has_empty = ParameterGrid([{"C": [1, 10]}, {}, {"C": [0.5]}])
    assert len(has_empty) == 4
    assert list(has_empty) == [{"C": 1}, {"C": 10}, {}, {"C": 0.5}]
    assert_grid_iter_equals_getitem(has_empty)


def test_grid_search():
    # [FUNCTION DOC]
    # Verifies GridSearchCV with MockClassifier picks foo_param=2 (smallest among tied
    #   best), records param_foo_param values, smoke tests delegated methods, and raises
    #   ValueError for an invalid scoring string. Temporarily redirects sys.stdout to
    #   swallow verbose output.
    # Test that the best estimator contains the right value for foo_param
    clf = MockClassifier()
    grid_search = GridSearchCV(clf, {"foo_param": [1, 2, 3]}, cv=2, verbose=3)
    # make sure it selects the smallest parameter in case of ties
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    grid_search.fit(X, y)
    sys.stdout = old_stdout
    assert grid_search.best_estimator_.foo_param == 2

    assert_array_equal(grid_search.cv_results_["param_foo_param"].data, [1, 2, 3])

    # Smoke test the score etc:
    grid_search.score(X, y)
    grid_search.predict_proba(X)
    grid_search.decision_function(X)
    grid_search.transform(X)

    # Test exception handling on scoring
    grid_search.scoring = "sklearn"
    with pytest.raises(ValueError):
        grid_search.fit(X, y)


def test_grid_search_pipeline_steps():
    # [FUNCTION DOC]
    # Verifies estimators given as grid values are cloned: cv_results_ entries are
    #   unfitted instances distinct from best_estimator_, and the user's param_grid
    #   objects are not fitted.
    # check that parameters that are estimators are cloned before fitting
    pipe = Pipeline([("regressor", LinearRegression())])
    param_grid = {"regressor": [LinearRegression(), Ridge()]}
    grid_search = GridSearchCV(pipe, param_grid, cv=2)
    grid_search.fit(X, y)
    regressor_results = grid_search.cv_results_["param_regressor"]
    assert isinstance(regressor_results[0], LinearRegression)
    assert isinstance(regressor_results[1], Ridge)
    assert not hasattr(regressor_results[0], "coef_")
    assert not hasattr(regressor_results[1], "coef_")
    assert regressor_results[0] is not grid_search.best_estimator_
    assert regressor_results[1] is not grid_search.best_estimator_
    # check that we didn't modify the parameter grid that was passed
    assert not hasattr(param_grid["regressor"][0], "coef_")
    assert not hasattr(param_grid["regressor"][1], "coef_")


@pytest.mark.parametrize("SearchCV", [GridSearchCV, RandomizedSearchCV])
def test_SearchCV_with_fit_params(SearchCV):
    # [FUNCTION DOC]
    # Verifies fit params are forwarded and sliced per split for Grid/RandomizedSearchCV
    #   via CheckingClassifier: missing param or wrong-length param raises
    #   AssertionError (error_score="raise"), correct ones pass.
    X = np.arange(100).reshape(10, 10)
    y = np.array([0] * 5 + [1] * 5)
    clf = CheckingClassifier(expected_fit_params=["spam", "eggs"])
    searcher = SearchCV(clf, {"foo_param": [1, 2, 3]}, cv=2, error_score="raise")

    # The CheckingClassifier generates an assertion error if
    # a parameter is missing or has length != len(X).
    err_msg = r"Expected fit parameter\(s\) \['eggs'\] not seen."
    with pytest.raises(AssertionError, match=err_msg):
        searcher.fit(X, y, spam=np.ones(10))

    err_msg = "Fit parameter spam has length 1; expected"
    with pytest.raises(AssertionError, match=err_msg):
        searcher.fit(X, y, spam=np.ones(1), eggs=np.zeros(10))
    searcher.fit(X, y, spam=np.ones(10), eggs=np.zeros(10))


def test_grid_search_no_score():
    # [FUNCTION DOC]
    # Verifies that with an explicit scoring, an estimator lacking score gives the same
    #   best_params_ and score as one with score; without scoring, fit raises TypeError
    #   mentioning "no scoring".
    # Test grid-search on classifier that has no score function.
    clf = LinearSVC(random_state=0)
    X, y = make_blobs(random_state=0, centers=2)
    Cs = [0.1, 1, 10]
    clf_no_score = LinearSVCNoScore(random_state=0)
    grid_search = GridSearchCV(clf, {"C": Cs}, scoring="accuracy")
    grid_search.fit(X, y)

    grid_search_no_score = GridSearchCV(clf_no_score, {"C": Cs}, scoring="accuracy")
    # smoketest grid search
    grid_search_no_score.fit(X, y)

    # check that best params are equal
    assert grid_search_no_score.best_params_ == grid_search.best_params_
    # check that we can call score and that it gives the correct result
    assert grid_search.score(X, y) == grid_search_no_score.score(X, y)

    # giving no scoring function raises an error
    grid_search_no_score = GridSearchCV(clf_no_score, {"C": Cs})
    with pytest.raises(TypeError, match="no scoring"):
        grid_search_no_score.fit([[1]])


def test_grid_search_score_method():
    # [FUNCTION DOC]
    # Verifies SearchCV.score uses the scoring parameter: scoring=None equals accuracy,
    #   and roc_auc results match whether or not the estimator has a score method;
    #   sanity checks that the two metrics differ.
    X, y = make_classification(n_samples=100, n_classes=2, flip_y=0.2, random_state=0)
    clf = LinearSVC(random_state=0)
    grid = {"C": [0.1]}

    search_no_scoring = GridSearchCV(clf, grid, scoring=None).fit(X, y)
    search_accuracy = GridSearchCV(clf, grid, scoring="accuracy").fit(X, y)
    search_no_score_method_auc = GridSearchCV(
        LinearSVCNoScore(), grid, scoring="roc_auc"
    ).fit(X, y)
    search_auc = GridSearchCV(clf, grid, scoring="roc_auc").fit(X, y)

    # Check warning only occurs in situation where behavior changed:
    # estimator requires score method to compete with scoring parameter
    score_no_scoring = search_no_scoring.score(X, y)
    score_accuracy = search_accuracy.score(X, y)
    score_no_score_auc = search_no_score_method_auc.score(X, y)
    score_auc = search_auc.score(X, y)

    # ensure the test is sane
    assert score_auc < 1.0
    assert score_accuracy < 1.0
    assert score_auc != score_accuracy

    assert_almost_equal(score_accuracy, score_no_scoring)
    assert_almost_equal(score_auc, score_no_score_auc)


def test_grid_search_groups():
    # [FUNCTION DOC]
    # Verifies group-based CV splitters raise ValueError when groups is None and work
    #   when groups is passed, while non-group splitters work without groups.
    # Check if ValueError (when groups is None) propagates to GridSearchCV
    # And also check if groups is correctly passed to the cv object
    rng = np.random.RandomState(0)

    X, y = make_classification(n_samples=15, n_classes=2, random_state=0)
    groups = rng.randint(0, 3, 15)

    clf = LinearSVC(random_state=0)
    grid = {"C": [1]}

    group_cvs = [
        LeaveOneGroupOut(),
        LeavePGroupsOut(2),
        GroupKFold(n_splits=3),
        GroupShuffleSplit(),
    ]
    error_msg = "The 'groups' parameter should not be None."
    for cv in group_cvs:
        gs = GridSearchCV(clf, grid, cv=cv)
        with pytest.raises(ValueError, match=error_msg):
            gs.fit(X, y)
        gs.fit(X, y, groups=groups)

    non_group_cvs = [StratifiedKFold(), StratifiedShuffleSplit()]
    for cv in non_group_cvs:
        gs = GridSearchCV(clf, grid, cv=cv)
        # Should not raise an error
        gs.fit(X, y)


def test_classes__property():
    # [FUNCTION DOC]
    # Verifies classes_ on the search equals best_estimator_.classes_ for classifiers
    #   and is absent for regressors, before fit, and when refit=False.
    # Test that classes_ property matches best_estimator_.classes_
    X = np.arange(100).reshape(10, 10)
    y = np.array([0] * 5 + [1] * 5)
    Cs = [0.1, 1, 10]

    grid_search = GridSearchCV(LinearSVC(random_state=0), {"C": Cs})
    grid_search.fit(X, y)
    assert_array_equal(grid_search.best_estimator_.classes_, grid_search.classes_)

    # Test that regressors do not have a classes_ attribute
    grid_search = GridSearchCV(Ridge(), {"alpha": [1.0, 2.0]})
    grid_search.fit(X, y)
    assert not hasattr(grid_search, "classes_")

    # Test that the grid searcher has no classes_ attribute before it's fit
    grid_search = GridSearchCV(LinearSVC(random_state=0), {"C": Cs})
    assert not hasattr(grid_search, "classes_")

    # Test that the grid searcher has no classes_ attribute without a refit
    grid_search = GridSearchCV(LinearSVC(random_state=0), {"C": Cs}, refit=False)
    grid_search.fit(X, y)
    assert not hasattr(grid_search, "classes_")


def test_trivial_cv_results_attr():
    # [FUNCTION DOC]
    # Verifies cv_results_ is set when the grid / sampled space has a single point.
    # Test search over a "grid" with only one point.
    clf = MockClassifier()
    grid_search = GridSearchCV(clf, {"foo_param": [1]}, cv=2)
    grid_search.fit(X, y)
    assert hasattr(grid_search, "cv_results_")

    random_search = RandomizedSearchCV(clf, {"foo_param": [0]}, n_iter=1, cv=2)
    random_search.fit(X, y)
    assert hasattr(random_search, "cv_results_")


def test_no_refit():
    # [FUNCTION DOC]
    # Verifies refit=False keeps best_index_/best_params_ but not best_estimator_, and
    #   that predict-like methods raise AttributeError chained from an inner
    #   AttributeError explaining refit=False. Also checks invalid refit values with
    #   dict multi-metric scoring raise ValueError.
    # Test that GSCV can be used for model selection alone without refitting
    clf = MockClassifier()
    for scoring in [None, ["accuracy", "precision"]]:
        grid_search = GridSearchCV(clf, {"foo_param": [1, 2, 3]}, refit=False, cv=2)
        grid_search.fit(X, y)
        assert (
            not hasattr(grid_search, "best_estimator_")
            and hasattr(grid_search, "best_index_")
            and hasattr(grid_search, "best_params_")
        )

        # Make sure the functions predict/transform etc. raise meaningful
        # error messages
        for fn_name in (
            "predict",
            "predict_proba",
            "predict_log_proba",
            "transform",
            "inverse_transform",
        ):
            outer_msg = f"has no attribute '{fn_name}'"
            inner_msg = (
                f"`refit=False`. {fn_name} is available only after "
                "refitting on the best parameters"
            )
            with pytest.raises(AttributeError, match=outer_msg) as exec_info:
                getattr(grid_search, fn_name)(X)

            assert isinstance(exec_info.value.__cause__, AttributeError)
            assert inner_msg in str(exec_info.value.__cause__)

    # Test that an invalid refit param raises appropriate error messages
    error_msg = (
        "For multi-metric scoring, the parameter refit must be set to a scorer key"
    )
    for refit in [True, "recall", "accuracy"]:
        with pytest.raises(ValueError, match=error_msg):
            GridSearchCV(
                clf, {}, refit=refit, scoring={"acc": "accuracy", "prec": "precision"}
            ).fit(X, y)


def test_grid_search_error():
    # [FUNCTION DOC]
    # Verifies fit raises ValueError when X and y have different lengths.
    # Test that grid search will capture errors on data with different length
    X_, y_ = make_classification(n_samples=200, n_features=100, random_state=0)

    clf = LinearSVC()
    cv = GridSearchCV(clf, {"C": [0.1, 1.0]})
    with pytest.raises(ValueError):
        cv.fit(X_[:180], y_)


def test_grid_search_one_grid_point():
    # [FUNCTION DOC]
    # Verifies a single-point grid yields a best_estimator_ identical (dual_coef_) to
    #   directly fitting SVC with those params.
    X_, y_ = make_classification(n_samples=200, n_features=100, random_state=0)
    param_dict = {"C": [1.0], "kernel": ["rbf"], "gamma": [0.1]}

    clf = SVC(gamma="auto")
    cv = GridSearchCV(clf, param_dict)
    cv.fit(X_, y_)

    clf = SVC(C=1.0, kernel="rbf", gamma=0.1)
    clf.fit(X_, y_)

    assert_array_equal(clf.dual_coef_, cv.best_estimator_.dual_coef_)


def test_grid_search_when_param_grid_includes_range():
    # [FUNCTION DOC]
    # Verifies a range object is accepted as grid values and foo_param=2 is chosen.
    # KNOWN ISSUE: the initial assignment of grid_search to None is dead code.
    # Test that the best estimator contains the right value for foo_param
    clf = MockClassifier()
    grid_search = None
    grid_search = GridSearchCV(clf, {"foo_param": range(1, 4)}, cv=2)
    grid_search.fit(X, y)
    assert grid_search.best_estimator_.foo_param == 2


def test_grid_search_bad_param_grid():
    # [FUNCTION DOC]
    # Verifies param grid validation at fit: scalar value and string raise TypeError
    #   with a wrap-in-list hint, empty list raises ValueError, and a 2D array raises
    #   ValueError.
    X, y = make_classification(n_samples=10, n_features=5, random_state=0)
    param_dict = {"C": 1}
    clf = SVC(gamma="auto")
    error_msg = re.escape(
        "Parameter grid for parameter 'C' needs to be a list or "
        "a numpy array, but got 1 (of type int) instead. Single "
        "values need to be wrapped in a list with one element."
    )
    search = GridSearchCV(clf, param_dict)
    with pytest.raises(TypeError, match=error_msg):
        search.fit(X, y)

    param_dict = {"C": []}
    clf = SVC()
    error_msg = re.escape(
        "Parameter grid for parameter 'C' need to be a non-empty sequence, got: []"
    )
    search = GridSearchCV(clf, param_dict)
    with pytest.raises(ValueError, match=error_msg):
        search.fit(X, y)

    param_dict = {"C": "1,2,3"}
    clf = SVC(gamma="auto")
    error_msg = re.escape(
        "Parameter grid for parameter 'C' needs to be a list or a numpy array, "
        "but got '1,2,3' (of type str) instead. Single values need to be "
        "wrapped in a list with one element."
    )
    search = GridSearchCV(clf, param_dict)
    with pytest.raises(TypeError, match=error_msg):
        search.fit(X, y)

    param_dict = {"C": np.ones((3, 2))}
    clf = SVC()
    search = GridSearchCV(clf, param_dict)
    with pytest.raises(ValueError):
        search.fit(X, y)


@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_grid_search_sparse(csr_container):
    # [FUNCTION DOC]
    # Verifies dense vs sparse (COO from CSR container) training give the same chosen C
    #   and >= 90% identical predictions.
    # Test that grid search works with both dense and sparse matrices
    X_, y_ = make_classification(n_samples=200, n_features=100, random_state=0)

    clf = LinearSVC()
    cv = GridSearchCV(clf, {"C": [0.1, 1.0]})
    cv.fit(X_[:180], y_[:180])
    y_pred = cv.predict(X_[180:])
    C = cv.best_estimator_.C

    X_ = csr_container(X_)
    clf = LinearSVC()
    cv = GridSearchCV(clf, {"C": [0.1, 1.0]})
    cv.fit(X_[:180].tocoo(), y_[:180])
    y_pred2 = cv.predict(X_[180:])
    C2 = cv.best_estimator_.C

    assert np.mean(y_pred == y_pred2) >= 0.9
    assert C == C2


@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_grid_search_sparse_scoring(csr_container):
    # [FUNCTION DOC]
    # Verifies dense vs sparse with scoring="f1" give identical predictions and C, and
    #   that a custom loss scorer (greater_is_better=False, negated f1) yields the same
    #   choice.
    X_, y_ = make_classification(n_samples=200, n_features=100, random_state=0)

    clf = LinearSVC()
    cv = GridSearchCV(clf, {"C": [0.1, 1.0]}, scoring="f1")
    cv.fit(X_[:180], y_[:180])
    y_pred = cv.predict(X_[180:])
    C = cv.best_estimator_.C

    X_ = csr_container(X_)
    clf = LinearSVC()
    cv = GridSearchCV(clf, {"C": [0.1, 1.0]}, scoring="f1")
    cv.fit(X_[:180], y_[:180])
    y_pred2 = cv.predict(X_[180:])
    C2 = cv.best_estimator_.C

    assert_array_equal(y_pred, y_pred2)
    assert C == C2
    # Smoke test the score
    # np.testing.assert_allclose(f1_score(cv.predict(X_[:180]), y[:180]),
    #                            cv.score(X_[:180], y[:180]))

    # test loss where greater is worse
    def f1_loss(y_true_, y_pred_):
        # [FUNCTION DOC]
        # HELPER: returns minus the f1 score of the given labels and predictions.
        return -f1_score(y_true_, y_pred_)

    F1Loss = make_scorer(f1_loss, greater_is_better=False)
    cv = GridSearchCV(clf, {"C": [0.1, 1.0]}, scoring=F1Loss)
    cv.fit(X_[:180], y_[:180])
    y_pred3 = cv.predict(X_[180:])
    C3 = cv.best_estimator_.C

    assert C == C3
    assert_array_equal(y_pred, y_pred3)


def test_grid_search_precomputed_kernel():
    # [FUNCTION DOC]
    # Verifies grid search on SVC with a precomputed linear kernel fits and predicts on
    #   a test kernel; a list (non array) kernel raises ValueError.
    # KNOWN ISSUE: the asserts best_score_ >= 0 and accuracy >= 0 are trivially true and
    #   check little.
    # Test that grid search works when the input features are given in the
    # form of a precomputed kernel matrix
    X_, y_ = make_classification(n_samples=200, n_features=100, random_state=0)

    # compute the training kernel matrix corresponding to the linear kernel
    K_train = np.dot(X_[:180], X_[:180].T)
    y_train = y_[:180]

    clf = SVC(kernel="precomputed")
    cv = GridSearchCV(clf, {"C": [0.1, 1.0]})
    cv.fit(K_train, y_train)

    assert cv.best_score_ >= 0

    # compute the test kernel matrix
    K_test = np.dot(X_[180:], X_[:180].T)
    y_test = y_[180:]

    y_pred = cv.predict(K_test)

    assert np.mean(y_pred == y_test) >= 0

    # test error is raised when the precomputed kernel is not array-like
    # or sparse
    with pytest.raises(ValueError):
        cv.fit(K_train.tolist(), y_train)


def test_grid_search_precomputed_kernel_error_nonsquare():
    # [FUNCTION DOC]
    # Verifies a non-square precomputed training kernel raises ValueError at fit.
    # Test that grid search returns an error with a non-square precomputed
    # training kernel matrix
    K_train = np.zeros((10, 20))
    y_train = np.ones((10,))
    clf = SVC(kernel="precomputed")
    cv = GridSearchCV(clf, {"C": [0.1, 1.0]})
    with pytest.raises(ValueError):
        cv.fit(K_train, y_train)


class BrokenClassifier(BaseEstimator):
    # [CLASS DOC]
    # HELPER: estimator that fails (AssertionError) if fit is called twice on the same
    #   instance; detects missing cloning before refit.
    """Broken classifier that cannot be fit twice"""

    def __init__(self, parameter=None):
        # [FUNCTION DOC]
        # HELPER: stores parameter unchanged.
        self.parameter = parameter

    def fit(self, X, y):
        # [FUNCTION DOC]
        # HELPER: asserts has_been_fit_ is not set, then sets it to True. Returns None
        #   (not self).
        assert not hasattr(self, "has_been_fit_")
        self.has_been_fit_ = True

    def predict(self, X):
        # [FUNCTION DOC]
        # HELPER: returns a zero vector with one entry per row of X.
        return np.zeros(X.shape[0])


@pytest.mark.filterwarnings("ignore::sklearn.exceptions.UndefinedMetricWarning")
def test_refit():
    # [FUNCTION DOC]
    # Regression test: refit=True with BrokenClassifier must not refit an already-fitted
    #   instance (would raise AssertionError).
    # Regression test for bug in refitting
    # Simulates re-fitting a broken estimator; this used to break with
    # sparse SVMs.
    X = np.arange(100).reshape(10, 10)
    y = np.array([0] * 5 + [1] * 5)

    clf = GridSearchCV(
        BrokenClassifier(), [{"parameter": [0, 1]}], scoring="precision", refit=True
    )
    clf.fit(X, y)


def test_refit_callable():
    # [FUNCTION DOC]
    # Verifies refit=callable: the callable receives cv_results_ with all standard keys,
    #   its returned index becomes best_index_, and best_score_ is not set.
    """
    Test refit=callable, which adds flexibility in identifying the
    "best" estimator.
    """

    def refit_callable(cv_results):
        # [FUNCTION DOC]
        # HELPER: fits an internal GridSearchCV to check that every key of its
        #   cv_results_ is present in the given cv_results, then returns the argmin of
        #   mean_test_score.
        # Exceptions: AssertionError if the internal best_index_ is 0 or keys are
        #   missing.
        """
        A dummy function tests `refit=callable` interface.
        Return the index of a model that has the least
        `mean_test_score`.
        """
        # Fit a dummy clf with `refit=True` to get a list of keys in
        # clf.cv_results_.
        X, y = make_classification(n_samples=100, n_features=4, random_state=42)
        clf = GridSearchCV(
            LinearSVC(random_state=42),
            {"C": [0.01, 0.1, 1]},
            scoring="precision",
            refit=True,
        )
        clf.fit(X, y)
        # Ensure that `best_index_ != 0` for this dummy clf
        assert clf.best_index_ != 0

        # Assert every key matches those in `cv_results`
        for key in clf.cv_results_.keys():
            assert key in cv_results

        return cv_results["mean_test_score"].argmin()

    X, y = make_classification(n_samples=100, n_features=4, random_state=42)
    clf = GridSearchCV(
        LinearSVC(random_state=42),
        {"C": [0.01, 0.1, 1]},
        scoring="precision",
        refit=refit_callable,
    )
    clf.fit(X, y)

    assert clf.best_index_ == 0
    # Ensure `best_score_` is disabled when using `refit=callable`
    assert not hasattr(clf, "best_score_")


def test_refit_callable_invalid_type():
    # [FUNCTION DOC]
    # Verifies a refit callable returning a non-integer raises TypeError.
    """
    Test implementation catches the errors when 'best_index_' returns an
    invalid result.
    """

    def refit_callable_invalid_type(cv_results):
        # [FUNCTION DOC]
        # HELPER: ignores cv_results and returns None.
        """
        A dummy function tests when returned 'best_index_' is not integer.
        """
        return None

    X, y = make_classification(n_samples=100, n_features=4, random_state=42)

    clf = GridSearchCV(
        LinearSVC(random_state=42),
        {"C": [0.1, 1]},
        scoring="precision",
        refit=refit_callable_invalid_type,
    )
    with pytest.raises(TypeError, match="best_index_ returned is not an integer"):
        clf.fit(X, y)


@pytest.mark.parametrize("out_bound_value", [-1, 2])
@pytest.mark.parametrize("search_cv", [RandomizedSearchCV, GridSearchCV])
def test_refit_callable_out_bound(out_bound_value, search_cv):
    # [FUNCTION DOC]
    # Verifies a refit callable returning an out-of-range index (-1 or 2 for 2
    #   candidates) raises IndexError.
    """
    Test implementation catches the errors when 'best_index_' returns an
    out of bound result.
    """

    def refit_callable_out_bound(cv_results):
        # [FUNCTION DOC]
        # HELPER: returns the closure value out_bound_value.
        """
        A dummy function tests when returned 'best_index_' is out of bounds.
        """
        return out_bound_value

    X, y = make_classification(n_samples=100, n_features=4, random_state=42)

    clf = search_cv(
        LinearSVC(random_state=42),
        {"C": [0.1, 1]},
        scoring="precision",
        refit=refit_callable_out_bound,
    )
    with pytest.raises(IndexError, match="best_index_ index out of range"):
        clf.fit(X, y)


def test_refit_callable_multi_metric():
    # [FUNCTION DOC]
    # Verifies refit=callable with dict multi-metric scoring: callable output sets
    #   best_index_ and best_score_ is absent.
    """
    Test refit=callable in multiple metric evaluation setting
    """

    def refit_callable(cv_results):
        # [FUNCTION DOC]
        # HELPER: asserts mean_test_prec exists and returns its argmin.
        """
        A dummy function tests `refit=callable` interface.
        Return the index of a model that has the least
        `mean_test_prec`.
        """
        assert "mean_test_prec" in cv_results
        return cv_results["mean_test_prec"].argmin()

    X, y = make_classification(n_samples=100, n_features=4, random_state=42)
    scoring = {"Accuracy": make_scorer(accuracy_score), "prec": "precision"}
    clf = GridSearchCV(
        LinearSVC(random_state=42),
        {"C": [0.01, 0.1, 1]},
        scoring=scoring,
        refit=refit_callable,
    )
    clf.fit(X, y)

    assert clf.best_index_ == 0
    # Ensure `best_score_` is disabled when using `refit=callable`
    assert not hasattr(clf, "best_score_")


def test_gridsearch_nd():
    # [FUNCTION DOC]
    # Verifies n-dimensional X (4D) and y (3D) are passed through to the estimator's fit
    #   unchanged in trailing shape.
    # KNOWN ISSUE: the comment says "Pass X as list" but X is a 4D array; final score
    #   uses the module-level X, y.
    # Pass X as list in GridSearchCV
    X_4d = np.arange(10 * 5 * 3 * 2).reshape(10, 5, 3, 2)
    y_3d = np.arange(10 * 7 * 11).reshape(10, 7, 11)

    def check_X(x):
        # [FUNCTION DOC]
        # HELPER: True if trailing shape is (5, 3, 2).
        return x.shape[1:] == (5, 3, 2)

    def check_y(x):
        # [FUNCTION DOC]
        # HELPER: True if trailing shape is (7, 11).
        return x.shape[1:] == (7, 11)

    clf = CheckingClassifier(
        check_X=check_X,
        check_y=check_y,
        methods_to_check=["fit"],
    )
    grid_search = GridSearchCV(clf, {"foo_param": [1, 2, 3]})
    grid_search.fit(X_4d, y_3d).score(X, y)
    assert hasattr(grid_search, "cv_results_")


def test_X_as_list():
    # [FUNCTION DOC]
    # Verifies a Python list X is passed to the estimator's fit as a list (no
    #   conversion).
    # Pass X as list in GridSearchCV
    X = np.arange(100).reshape(10, 10)
    y = np.array([0] * 5 + [1] * 5)

    clf = CheckingClassifier(
        check_X=lambda x: isinstance(x, list),
        methods_to_check=["fit"],
    )
    cv = KFold(n_splits=3)
    grid_search = GridSearchCV(clf, {"foo_param": [1, 2, 3]}, cv=cv)
    grid_search.fit(X.tolist(), y).score(X, y)
    assert hasattr(grid_search, "cv_results_")


def test_y_as_list():
    # [FUNCTION DOC]
    # Verifies a Python list y is passed to the estimator's fit as a list.
    # Pass y as list in GridSearchCV
    X = np.arange(100).reshape(10, 10)
    y = np.array([0] * 5 + [1] * 5)

    clf = CheckingClassifier(
        check_y=lambda x: isinstance(x, list),
        methods_to_check=["fit"],
    )
    cv = KFold(n_splits=3)
    grid_search = GridSearchCV(clf, {"foo_param": [1, 2, 3]}, cv=cv)
    grid_search.fit(X, y.tolist()).score(X, y)
    assert hasattr(grid_search, "cv_results_")


def test_pandas_input():
    # [FUNCTION DOC]
    # Verifies DataFrame/Series-like inputs (MockDataFrame and, if installed, pandas)
    #   reach the estimator without conversion through fit, score and predict.
    # KNOWN ISSUE: comment mentions cross_val_score but the test uses GridSearchCV.
    # check cross_val_score doesn't destroy pandas dataframe
    types = [(MockDataFrame, MockDataFrame)]
    try:
        from pandas import DataFrame, Series

        types.append((DataFrame, Series))
    except ImportError:
        pass

    X = np.arange(100).reshape(10, 10)
    y = np.array([0] * 5 + [1] * 5)

    for InputFeatureType, TargetType in types:
        # X dataframe, y series
        X_df, y_ser = InputFeatureType(X), TargetType(y)

        def check_df(x):
            # [FUNCTION DOC]
            # HELPER: True if x is an instance of the current InputFeatureType (loop
            #   closure).
            return isinstance(x, InputFeatureType)

        def check_series(x):
            # [FUNCTION DOC]
            # HELPER: True if x is an instance of the current TargetType (loop closure).
            return isinstance(x, TargetType)

        clf = CheckingClassifier(check_X=check_df, check_y=check_series)

        grid_search = GridSearchCV(clf, {"foo_param": [1, 2, 3]})
        grid_search.fit(X_df, y_ser).score(X_df, y_ser)
        grid_search.predict(X_df)
        assert hasattr(grid_search, "cv_results_")


def test_unsupervised_grid_search():
    # [FUNCTION DOC]
    # Verifies KMeans grid search on blobs: supervised clustering metrics (multi and
    #   single metric) select n_clusters=3; default scoring (KMeans score, no y) selects
    #   4.
    # test grid-search with unsupervised estimator
    X, y = make_blobs(n_samples=50, random_state=0)
    km = KMeans(random_state=0, init="random", n_init=1)

    # Multi-metric evaluation unsupervised
    scoring = ["adjusted_rand_score", "fowlkes_mallows_score"]
    for refit in ["adjusted_rand_score", "fowlkes_mallows_score"]:
        grid_search = GridSearchCV(
            km, param_grid=dict(n_clusters=[2, 3, 4]), scoring=scoring, refit=refit
        )
        grid_search.fit(X, y)
        # Both ARI and FMS can find the right number :)
        assert grid_search.best_params_["n_clusters"] == 3

    # Single metric evaluation unsupervised
    grid_search = GridSearchCV(
        km, param_grid=dict(n_clusters=[2, 3, 4]), scoring="fowlkes_mallows_score"
    )
    grid_search.fit(X, y)
    assert grid_search.best_params_["n_clusters"] == 3

    # Now without a score, and without y
    grid_search = GridSearchCV(km, param_grid=dict(n_clusters=[2, 3, 4]))
    grid_search.fit(X)
    assert grid_search.best_params_["n_clusters"] == 4


def test_gridsearch_no_predict():
    # [FUNCTION DOC]
    # Verifies search works with an estimator lacking predict (KernelDensity) when a
    #   custom scoring callable is given.
    # test grid-search with an estimator without predict.
    # slight duplication of a test from KDE
    def custom_scoring(estimator, X):
        # [FUNCTION DOC]
        # HELPER: returns 42 if the estimator bandwidth is 0.1, else 0; X ignored.
        return 42 if estimator.bandwidth == 0.1 else 0

    X, _ = make_blobs(cluster_std=0.1, random_state=1, centers=[[0, 1], [1, 0], [0, 0]])
    search = GridSearchCV(
        KernelDensity(),
        param_grid=dict(bandwidth=[0.01, 0.1, 1]),
        scoring=custom_scoring,
    )
    search.fit(X)
    assert search.best_params_["bandwidth"] == 0.1
    assert search.best_score_ == 42


def test_param_sampler():
    # [FUNCTION DOC]
    # Verifies ParameterSampler yields n_iter dicts within the given
    #   choices/distribution ranges, and is reproducible across iterations with a fixed
    #   random_state for both lists and distributions.
    # test basic properties of param sampler
    param_distributions = {"kernel": ["rbf", "linear"], "C": uniform(0, 1)}
    sampler = ParameterSampler(
        param_distributions=param_distributions, n_iter=10, random_state=0
    )
    samples = [x for x in sampler]
    assert len(samples) == 10
    for sample in samples:
        assert sample["kernel"] in ["rbf", "linear"]
        assert 0 <= sample["C"] <= 1

    # test that repeated calls yield identical parameters
    param_distributions = {"C": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
    sampler = ParameterSampler(
        param_distributions=param_distributions, n_iter=3, random_state=0
    )
    assert [x for x in sampler] == [x for x in sampler]

    param_distributions = {"C": uniform(0, 1)}
    sampler = ParameterSampler(
        param_distributions=param_distributions, n_iter=10, random_state=0
    )
    assert [x for x in sampler] == [x for x in sampler]


def check_cv_results_array_types(
    search, param_keys, score_keys, expected_cv_results_kinds
):
    # [FUNCTION DOC]
    # HELPER: asserts that param_* entries of search.cv_results_ are MaskedArrays with
    #   the expected dtype kinds, score/time entries are plain float64 arrays, and each
    #   rank_test_* is int32.
    # Inputs: search - fitted search; param_keys, score_keys - tuples of key names;
    #   expected_cv_results_kinds - dict key to numpy dtype kind char.
    # Check if the search `cv_results`'s array are of correct types
    cv_results = search.cv_results_
    assert all(isinstance(cv_results[param], np.ma.MaskedArray) for param in param_keys)
    assert {
        key: cv_results[key].dtype.kind for key in param_keys
    } == expected_cv_results_kinds
    assert not any(isinstance(cv_results[key], np.ma.MaskedArray) for key in score_keys)
    assert all(
        cv_results[key].dtype == np.float64
        for key in score_keys
        if not key.startswith("rank")
    )

    scorer_keys = search.scorer_.keys() if search.multimetric_ else ["score"]

    for key in scorer_keys:
        assert cv_results["rank_test_%s" % key].dtype == np.int32


def check_cv_results_keys(cv_results, param_keys, score_keys, n_cand, extra_keys=()):
    # [FUNCTION DOC]
    # HELPER: asserts cv_results keys are exactly param_keys + score_keys + extra_keys +
    #   "params", and each param/score array has shape (n_cand,).
    # Test the search.cv_results_ contains all the required results
    all_keys = param_keys + score_keys + extra_keys
    assert_array_equal(sorted(cv_results.keys()), sorted(all_keys + ("params",)))
    assert all(cv_results[key].shape == (n_cand,) for key in param_keys + score_keys)


def test_grid_search_cv_results():
    # [FUNCTION DOC]
    # Verifies GridSearchCV.cv_results_ for a two-subgrid SVC search: keys, dtypes,
    #   ranks >= 1, and masking of params not present in a candidate's subgrid (poly
    #   masks C/gamma, rbf masks degree).
    # KNOWN ISSUE: two asserts wrap a generator in parentheses (assert of a generator
    #   object), which is always true; the score range checks never execute.
    X, y = make_classification(n_samples=50, n_features=4, random_state=42)

    n_grid_points = 6
    params = [
        dict(
            kernel=[
                "rbf",
            ],
            C=[1, 10],
            gamma=[0.1, 1],
        ),
        dict(
            kernel=[
                "poly",
            ],
            degree=[1, 2],
        ),
    ]

    param_keys = ("param_C", "param_degree", "param_gamma", "param_kernel")
    score_keys = (
        "mean_test_score",
        "mean_train_score",
        "rank_test_score",
        "split0_test_score",
        "split1_test_score",
        "split2_test_score",
        "split0_train_score",
        "split1_train_score",
        "split2_train_score",
        "std_test_score",
        "std_train_score",
        "mean_fit_time",
        "std_fit_time",
        "mean_score_time",
        "std_score_time",
    )
    n_candidates = n_grid_points

    search = GridSearchCV(SVC(), cv=3, param_grid=params, return_train_score=True)
    search.fit(X, y)
    cv_results = search.cv_results_
    # Check if score and timing are reasonable
    assert all(cv_results["rank_test_score"] >= 1)
    assert (all(cv_results[k] >= 0) for k in score_keys if k != "rank_test_score")
    assert (
        all(cv_results[k] <= 1)
        for k in score_keys
        if "time" not in k and k != "rank_test_score"
    )
    # Check cv_results structure
    expected_cv_results_kinds = {
        "param_C": "i",
        "param_degree": "i",
        "param_gamma": "f",
        "param_kernel": "O",
    }
    check_cv_results_array_types(
        search, param_keys, score_keys, expected_cv_results_kinds
    )
    check_cv_results_keys(cv_results, param_keys, score_keys, n_candidates)
    # Check masking
    cv_results = search.cv_results_

    poly_results = [
        (
            cv_results["param_C"].mask[i]
            and cv_results["param_gamma"].mask[i]
            and not cv_results["param_degree"].mask[i]
        )
        for i in range(n_candidates)
        if cv_results["param_kernel"][i] == "poly"
    ]
    assert all(poly_results)
    assert len(poly_results) == 2

    rbf_results = [
        (
            not cv_results["param_C"].mask[i]
            and not cv_results["param_gamma"].mask[i]
            and cv_results["param_degree"].mask[i]
        )
        for i in range(n_candidates)
        if cv_results["param_kernel"][i] == "rbf"
    ]
    assert all(rbf_results)
    assert len(rbf_results) == 4


def test_random_search_cv_results():
    # [FUNCTION DOC]
    # Same structure, dtype and masking checks as the grid version for
    #   RandomizedSearchCV with 30 iterations over continuous and discrete
    #   distributions.
    X, y = make_classification(n_samples=50, n_features=4, random_state=42)

    n_search_iter = 30

    params = [
        {"kernel": ["rbf"], "C": expon(scale=10), "gamma": expon(scale=0.1)},
        {"kernel": ["poly"], "degree": [2, 3]},
    ]
    param_keys = ("param_C", "param_degree", "param_gamma", "param_kernel")
    score_keys = (
        "mean_test_score",
        "mean_train_score",
        "rank_test_score",
        "split0_test_score",
        "split1_test_score",
        "split2_test_score",
        "split0_train_score",
        "split1_train_score",
        "split2_train_score",
        "std_test_score",
        "std_train_score",
        "mean_fit_time",
        "std_fit_time",
        "mean_score_time",
        "std_score_time",
    )
    n_candidates = n_search_iter

    search = RandomizedSearchCV(
        SVC(),
        n_iter=n_search_iter,
        cv=3,
        param_distributions=params,
        return_train_score=True,
    )
    search.fit(X, y)
    cv_results = search.cv_results_
    # Check results structure
    expected_cv_results_kinds = {
        "param_C": "f",
        "param_degree": "i",
        "param_gamma": "f",
        "param_kernel": "O",
    }
    check_cv_results_array_types(
        search, param_keys, score_keys, expected_cv_results_kinds
    )
    check_cv_results_keys(cv_results, param_keys, score_keys, n_candidates)
    assert all(
        (
            cv_results["param_C"].mask[i]
            and cv_results["param_gamma"].mask[i]
            and not cv_results["param_degree"].mask[i]
        )
        for i in range(n_candidates)
        if cv_results["param_kernel"][i] == "poly"
    )
    assert all(
        (
            not cv_results["param_C"].mask[i]
            and not cv_results["param_gamma"].mask[i]
            and cv_results["param_degree"].mask[i]
        )
        for i in range(n_candidates)
        if cv_results["param_kernel"][i] == "rbf"
    )


@pytest.mark.parametrize(
    "SearchCV, specialized_params",
    [
        (GridSearchCV, {"param_grid": {"C": [1, 10]}}),
        (RandomizedSearchCV, {"param_distributions": {"C": [1, 10]}, "n_iter": 2}),
    ],
)
def test_search_default_iid(SearchCV, specialized_params):
    # [FUNCTION DOC]
    # Verifies mean/std test scores are unweighted across folds using a hand-crafted
    #   non-iid 2-fold split (fold scores 1 and 1/3), and train scores are perfect.
    # KNOWN ISSUE: the in-code TODO notes the test name (iid) does not reflect what is
    #   tested.
    # Test the IID parameter  TODO: Clearly this test does something else???
    # noise-free simple 2d-data
    X, y = make_blobs(
        centers=[[0, 0], [1, 0], [0, 1], [1, 1]],
        random_state=0,
        cluster_std=0.1,
        shuffle=False,
        n_samples=80,
    )
    # split dataset into two folds that are not iid
    # first one contains data of all 4 blobs, second only from two.
    mask = np.ones(X.shape[0], dtype=bool)
    mask[np.where(y == 1)[0][::2]] = 0
    mask[np.where(y == 2)[0][::2]] = 0
    # this leads to perfect classification on one fold and a score of 1/3 on
    # the other
    # create "cv" for splits
    cv = [[mask, ~mask], [~mask, mask]]

    common_params = {"estimator": SVC(), "cv": cv, "return_train_score": True}
    search = SearchCV(**common_params, **specialized_params)
    search.fit(X, y)

    test_cv_scores = np.array(
        [
            search.cv_results_["split%d_test_score" % s][0]
            for s in range(search.n_splits_)
        ]
    )
    test_mean = search.cv_results_["mean_test_score"][0]
    test_std = search.cv_results_["std_test_score"][0]

    train_cv_scores = np.array(
        [
            search.cv_results_["split%d_train_score" % s][0]
            for s in range(search.n_splits_)
        ]
    )
    train_mean = search.cv_results_["mean_train_score"][0]
    train_std = search.cv_results_["std_train_score"][0]

    assert search.cv_results_["param_C"][0] == 1
    # scores are the same as above
    assert_allclose(test_cv_scores, [1, 1.0 / 3.0])
    assert_allclose(train_cv_scores, [1, 1])
    # Unweighted mean/std is used
    assert test_mean == pytest.approx(np.mean(test_cv_scores))
    assert test_std == pytest.approx(np.std(test_cv_scores))

    # For the train scores, we do not take a weighted mean irrespective of
    # i.i.d. or not
    assert train_mean == pytest.approx(1)
    assert train_std == pytest.approx(0)


def test_grid_search_cv_results_multimetric():
    # [FUNCTION DOC]
    # Verifies multi-metric GridSearchCV cv_results_ equals the merged results of
    #   separate accuracy and recall single-metric searches.
    X, y = make_classification(n_samples=50, n_features=4, random_state=42)

    n_splits = 3
    params = [
        dict(
            kernel=[
                "rbf",
            ],
            C=[1, 10],
            gamma=[0.1, 1],
        ),
        dict(
            kernel=[
                "poly",
            ],
            degree=[1, 2],
        ),
    ]

    grid_searches = []
    for scoring in (
        {"accuracy": make_scorer(accuracy_score), "recall": make_scorer(recall_score)},
        "accuracy",
        "recall",
    ):
        grid_search = GridSearchCV(
            SVC(), cv=n_splits, param_grid=params, scoring=scoring, refit=False
        )
        grid_search.fit(X, y)
        grid_searches.append(grid_search)

    compare_cv_results_multimetric_with_single(*grid_searches)


def test_random_search_cv_results_multimetric():
    # [FUNCTION DOC]
    # Verifies multi-metric RandomizedSearchCV matches single-metric searches, with
    #   refit False and refit True ("accuracy" for multi-metric), and that refit methods
    #   agree.
    # KNOWN ISSUE: refit is reassigned to "accuracy" inside the inner loop, so the
    #   subsequent single-metric searches in that outer iteration also get
    #   refit="accuracy" rather than True.
    X, y = make_classification(n_samples=50, n_features=4, random_state=42)

    n_splits = 3
    n_search_iter = 30

    params = dict(C=np.logspace(-4, 1, 3))
    for refit in (True, False):
        random_searches = []
        for scoring in (("accuracy", "recall"), "accuracy", "recall"):
            # If True, for multi-metric pass refit='accuracy'
            if refit and isinstance(scoring, tuple):
                refit = "accuracy"
            clf = LogisticRegression(random_state=42)
            random_search = RandomizedSearchCV(
                clf,
                n_iter=n_search_iter,
                cv=n_splits,
                param_distributions=params,
                scoring=scoring,
                refit=refit,
                random_state=0,
            )
            random_search.fit(X, y)
            random_searches.append(random_search)

        compare_cv_results_multimetric_with_single(*random_searches)
        compare_refit_methods_when_refit_with_acc(
            random_searches[0], random_searches[1], refit
        )


def compare_cv_results_multimetric_with_single(search_multi, search_acc, search_rec):
    # [FUNCTION DOC]
    # HELPER: asserts search_multi is multimetric with scorers accuracy and recall,
    #   timing entries are <= 1, and all non-time cv_results_ entries equal those of
    #   search_acc and search_rec after renaming their _score suffix to _accuracy /
    #   _recall.
    """Compare multi-metric cv_results with the ensemble of multiple
    single metric cv_results from single metric grid/random search"""

    assert search_multi.multimetric_
    assert_array_equal(sorted(search_multi.scorer_), ("accuracy", "recall"))

    cv_results_multi = search_multi.cv_results_
    cv_results_acc_rec = {
        re.sub("_score$", "_accuracy", k): v for k, v in search_acc.cv_results_.items()
    }
    cv_results_acc_rec.update(
        {re.sub("_score$", "_recall", k): v for k, v in search_rec.cv_results_.items()}
    )

    # Check if score and timing are reasonable, also checks if the keys
    # are present
    assert all(
        (
            np.all(cv_results_multi[k] <= 1)
            for k in (
                "mean_score_time",
                "std_score_time",
                "mean_fit_time",
                "std_fit_time",
            )
        )
    )

    # Compare the keys, other than time keys, among multi-metric and
    # single metric grid search results. np.testing.assert_equal performs a
    # deep nested comparison of the two cv_results dicts
    np.testing.assert_equal(
        {k: v for k, v in cv_results_multi.items() if not k.endswith("_time")},
        {k: v for k, v in cv_results_acc_rec.items() if not k.endswith("_time")},
    )


def compare_refit_methods_when_refit_with_acc(search_multi, search_acc, refit):
    # [FUNCTION DOC]
    # HELPER: asserts refit settings; when refit is truthy checks predict,
    #   predict_proba, predict_log_proba, score and best_index_/best_score_/best_params_
    #   agree between search_multi and search_acc on new blob data. Returns early when
    #   refit is falsy.
    """Compare refit multi-metric search methods with single metric methods"""
    assert search_acc.refit == refit
    if refit:
        assert search_multi.refit == "accuracy"
    else:
        assert not search_multi.refit
        return  # search cannot predict/score without refit

    X, y = make_blobs(n_samples=100, n_features=4, random_state=42)
    for method in ("predict", "predict_proba", "predict_log_proba"):
        assert_almost_equal(
            getattr(search_multi, method)(X), getattr(search_acc, method)(X)
        )
    assert_almost_equal(search_multi.score(X, y), search_acc.score(X, y))
    for key in ("best_index_", "best_score_", "best_params_"):
        assert getattr(search_multi, key) == getattr(search_acc, key)


@pytest.mark.parametrize(
    "search_cv",
    [
        RandomizedSearchCV(
            estimator=DecisionTreeClassifier(),
            param_distributions={"max_depth": [5, 10]},
        ),
        GridSearchCV(
            estimator=DecisionTreeClassifier(), param_grid={"max_depth": [5, 10]}
        ),
    ],
)
def test_search_cv_score_samples_error(search_cv):
    # [FUNCTION DOC]
    # Verifies score_samples raises AttributeError (chained from the inner estimator's
    #   AttributeError) when the estimator lacks score_samples.
    X, y = make_blobs(n_samples=100, n_features=4, random_state=42)
    search_cv = clone(search_cv)
    search_cv.fit(X, y)

    # Make sure to error out when underlying estimator does not implement
    # the method `score_samples`
    outer_msg = f"'{search_cv.__class__.__name__}' has no attribute 'score_samples'"
    inner_msg = "'DecisionTreeClassifier' object has no attribute 'score_samples'"

    with pytest.raises(AttributeError, match=outer_msg) as exec_info:
        search_cv.score_samples(X)
    assert isinstance(exec_info.value.__cause__, AttributeError)
    assert inner_msg == str(exec_info.value.__cause__)


def test_unsupported_sample_weight_scorer():
    # [FUNCTION DOC]
    # Verifies a UserWarning is emitted when sample_weight is passed to fit but the
    #   scorer (plain function, make_scorer, or one entry of a multi-metric dict) does
    #   not accept sample_weight.
    """Checks that fitting with sample_weight raises a warning if the scorer does not
    support sample_weight"""

    def fake_score_func(y_true, y_pred):
        # [FUNCTION DOC]
        # HELPER: constant score 0.5 with no sample_weight parameter.
        "Fake scoring function that does not support sample_weight"
        return 0.5

    fake_scorer = make_scorer(fake_score_func)

    X, y = make_classification(n_samples=10, n_features=4, random_state=42)
    sw = np.ones_like(y)
    search_cv = GridSearchCV(estimator=LogisticRegression(), param_grid={"C": [1, 10]})
    # function
    search_cv.set_params(scoring=fake_score_func)
    with pytest.warns(UserWarning, match="does not support sample_weight"):
        search_cv.fit(X, y, sample_weight=sw)
    # scorer
    search_cv.set_params(scoring=fake_scorer)
    with pytest.warns(UserWarning, match="does not support sample_weight"):
        search_cv.fit(X, y, sample_weight=sw)
    # multi-metric evaluation
    search_cv.set_params(
        scoring=dict(fake=fake_scorer, accuracy="accuracy"), refit=False
    )
    # only fake scorer does not support sample_weight
    with pytest.warns(
        UserWarning, match=r"The scoring fake=.* does not support sample_weight"
    ):
        search_cv.fit(X, y, sample_weight=sw)


@pytest.mark.parametrize(
    "estimator",
    [
        GridSearchCV(estimator=LogisticRegression(), param_grid={"C": [1, 10, 100]}),
        RandomizedSearchCV(
            estimator=Ridge(), param_distributions={"alpha": [1, 0.1, 0.01]}
        ),
    ],
)
def test_search_cv_sample_weight_equivalence(estimator):
    # [FUNCTION DOC]
    # Verifies that fitting with integer sample_weight equals fitting on data with rows
    #   repeated by weight: cv_results_ scores, best_score_/best_index_ and predict-like
    #   outputs match. Uses LeaveOneGroupOut so splits correspond.
    estimator_weighted = clone(estimator)
    estimator_repeated = clone(estimator)
    set_random_state(estimator_weighted, random_state=0)
    set_random_state(estimator_repeated, random_state=0)

    rng = np.random.RandomState(42)
    n_classes = 3
    n_samples_per_group = 30
    n_groups = 4
    n_samples = n_groups * n_samples_per_group
    X = rng.rand(n_samples, n_samples * 2)
    y = rng.randint(0, n_classes, size=n_samples)
    sw = rng.randint(0, 5, size=n_samples)
    # we use groups with LeaveOneGroupOut to ensure that
    # the splits are the same in the repeated/weighted datasets
    groups = np.tile(np.arange(n_groups), n_samples_per_group)

    X_weighted = X
    y_weighted = y
    groups_weighted = groups
    splits_weighted = list(LeaveOneGroupOut().split(X_weighted, groups=groups_weighted))
    estimator_weighted.set_params(cv=splits_weighted)
    # repeat samples according to weights
    X_repeated = X_weighted.repeat(repeats=sw, axis=0)
    y_repeated = y_weighted.repeat(repeats=sw)
    groups_repeated = groups_weighted.repeat(repeats=sw)
    splits_repeated = list(LeaveOneGroupOut().split(X_repeated, groups=groups_repeated))
    estimator_repeated.set_params(cv=splits_repeated)

    y_weighted = _enforce_estimator_tags_y(estimator_weighted, y_weighted)
    y_repeated = _enforce_estimator_tags_y(estimator_repeated, y_repeated)

    estimator_repeated.fit(X_repeated, y=y_repeated, sample_weight=None)
    estimator_weighted.fit(X_weighted, y=y_weighted, sample_weight=sw)

    # check that scores stored in cv_results_
    # are equal for the weighted/repeated datasets
    score_keys = [
        key for key in estimator_repeated.cv_results_ if key.endswith("score")
    ]
    for key in score_keys:
        s1 = estimator_repeated.cv_results_[key]
        s2 = estimator_weighted.cv_results_[key]
        err_msg = f"{key} values are not equal for weighted/repeated datasets"
        assert_allclose(s1, s2, err_msg=err_msg)

    for key in ["best_score_", "best_index_"]:
        s1 = getattr(estimator_repeated, key)
        s2 = getattr(estimator_weighted, key)
        err_msg = f"{key} values are not equal for weighted/repeated datasets"
        assert_almost_equal(s1, s2, err_msg=err_msg)

    for method in ["predict_proba", "decision_function", "predict", "transform"]:
        if hasattr(estimator, method):
            s1 = getattr(estimator_repeated, method)(X)
            s2 = getattr(estimator_weighted, method)(X)
            err_msg = (
                f"Comparing the output of {method} revealed that fitting "
                "with `sample_weight` is not equivalent to fitting with removed "
                "or repeated data points."
            )
            assert_allclose_dense_sparse(s1, s2, err_msg=err_msg)


@pytest.mark.parametrize(
    "search_cv",
    [
        RandomizedSearchCV(
            estimator=LocalOutlierFactor(novelty=True),
            param_distributions={"n_neighbors": [5, 10]},
            scoring="precision",
        ),
        GridSearchCV(
            estimator=LocalOutlierFactor(novelty=True),
            param_grid={"n_neighbors": [5, 10]},
            scoring="precision",
        ),
    ],
)
def test_search_cv_score_samples_method(search_cv):
    # [FUNCTION DOC]
    # Verifies search.score_samples equals best_estimator_.score_samples for
    #   LocalOutlierFactor(novelty=True) on inliers plus uniform outliers.
    search_cv = clone(search_cv)  # Avoid side effects from previous tests.
    # Set parameters
    rng = np.random.RandomState(42)
    n_samples = 300
    outliers_fraction = 0.15
    n_outliers = int(outliers_fraction * n_samples)
    n_inliers = n_samples - n_outliers

    # Create dataset
    X = make_blobs(
        n_samples=n_inliers,
        n_features=2,
        centers=[[0, 0], [0, 0]],
        cluster_std=0.5,
        random_state=0,
    )[0]
    # Add some noisy points
    X = np.concatenate([X, rng.uniform(low=-6, high=6, size=(n_outliers, 2))], axis=0)

    # Define labels to be able to score the estimator with `search_cv`
    y_true = np.array([1] * n_samples)
    y_true[-n_outliers:] = -1

    # Fit on data
    search_cv.fit(X, y_true)

    # Verify that the stand alone estimator yields the same results
    # as the ones obtained with *SearchCV
    assert_allclose(
        search_cv.score_samples(X), search_cv.best_estimator_.score_samples(X)
    )


def test_search_cv_results_rank_tie_breaking():
    # [FUNCTION DOC]
    # Verifies tied mean test scores get the same (min) rank: ranks [1, 1, 3] for C
    #   values 1, 1.001, 0.001.
    X, y = make_blobs(n_samples=50, random_state=42)

    # The two C values are close enough to give similar models
    # which would result in a tie of their mean cv-scores
    param_grid = {"C": [1, 1.001, 0.001]}

    grid_search = GridSearchCV(SVC(), param_grid=param_grid, return_train_score=True)
    random_search = RandomizedSearchCV(
        SVC(), n_iter=3, param_distributions=param_grid, return_train_score=True
    )

    for search in (grid_search, random_search):
        search.fit(X, y)
        cv_results = search.cv_results_
        # Check tie breaking strategy -
        # Check that there is a tie in the mean scores between
        # candidates 1 and 2 alone
        assert_almost_equal(
            cv_results["mean_test_score"][0], cv_results["mean_test_score"][1]
        )
        assert_almost_equal(
            cv_results["mean_train_score"][0], cv_results["mean_train_score"][1]
        )
        assert not np.allclose(
            cv_results["mean_test_score"][1], cv_results["mean_test_score"][2]
        )
        assert not np.allclose(
            cv_results["mean_train_score"][1], cv_results["mean_train_score"][2]
        )
        # 'min' rank should be assigned to the tied candidates
        assert_almost_equal(search.cv_results_["rank_test_score"], [1, 1, 3])


def test_search_cv_results_none_param():
    # [FUNCTION DOC]
    # Verifies None is kept as a parameter value in param_random_state.
    X, y = [[1], [2], [3], [4], [5]], [0, 0, 0, 0, 1]
    estimators = (DecisionTreeRegressor(), DecisionTreeClassifier())
    est_parameters = {"random_state": [0, None]}
    cv = KFold()

    for est in estimators:
        grid_search = GridSearchCV(
            est,
            est_parameters,
            cv=cv,
        ).fit(X, y)
        assert_array_equal(grid_search.cv_results_["param_random_state"], [0, None])


@pytest.mark.filterwarnings("ignore::sklearn.exceptions.FitFailedWarning")
def test_search_cv_timing():
    # [FUNCTION DOC]
    # Verifies fit/score timing entries are non-negative and < 1 s, failed fits (C=0)
    #   record score time 0, and refit_time_ is a non-negative float.
    svc = LinearSVC(random_state=0)

    X = [
        [
            1,
        ],
        [
            2,
        ],
        [
            3,
        ],
        [
            4,
        ],
    ]
    y = [0, 1, 1, 0]

    gs = GridSearchCV(svc, {"C": [0, 1]}, cv=2, error_score=0)
    rs = RandomizedSearchCV(svc, {"C": [0, 1]}, cv=2, error_score=0, n_iter=2)

    for search in (gs, rs):
        search.fit(X, y)
        for key in ["mean_fit_time", "std_fit_time"]:
            # NOTE The precision of time.time in windows is not high
            # enough for the fit/score times to be non-zero for trivial X and y
            assert np.all(search.cv_results_[key] >= 0)
            assert np.all(search.cv_results_[key] < 1)

        for key in ["mean_score_time", "std_score_time"]:
            assert search.cv_results_[key][1] >= 0
            assert search.cv_results_[key][0] == 0.0
            assert np.all(search.cv_results_[key] < 1)

        assert hasattr(search, "refit_time_")
        assert isinstance(search.refit_time_, float)
        assert search.refit_time_ >= 0


def test_grid_search_correct_score_results():
    # [FUNCTION DOC]
    # Verifies per-split test scores for f1 and roc_auc equal scores recomputed manually
    #   with StratifiedKFold, and expected keys exist.
    # test that correct scores are used
    n_splits = 3
    clf = LinearSVC(random_state=0)
    X, y = make_blobs(random_state=0, centers=2)
    Cs = [0.1, 1, 10]
    for score in ["f1", "roc_auc"]:
        grid_search = GridSearchCV(clf, {"C": Cs}, scoring=score, cv=n_splits)
        cv_results = grid_search.fit(X, y).cv_results_

        # Test scorer names
        result_keys = list(cv_results.keys())
        expected_keys = ("mean_test_score", "rank_test_score") + tuple(
            "split%d_test_score" % cv_i for cv_i in range(n_splits)
        )
        assert all(np.isin(expected_keys, result_keys))

        cv = StratifiedKFold(n_splits=n_splits)
        n_splits = grid_search.n_splits_
        for candidate_i, C in enumerate(Cs):
            clf.set_params(C=C)
            cv_scores = np.array(
                [
                    grid_search.cv_results_["split%d_test_score" % s][candidate_i]
                    for s in range(n_splits)
                ]
            )
            for i, (train, test) in enumerate(cv.split(X, y)):
                clf.fit(X[train], y[train])
                if score == "f1":
                    correct_score = f1_score(y[test], clf.predict(X[test]))
                elif score == "roc_auc":
                    dec = clf.decision_function(X[test])
                    correct_score = roc_auc_score(y[test], dec)
                assert_almost_equal(correct_score, cv_scores[i])


def test_pickle():
    # [FUNCTION DOC]
    # Verifies fitted Grid/RandomizedSearchCV round-trips through pickle with identical
    #   predictions.
    # Test that a fit search can be pickled
    clf = MockClassifier()
    grid_search = GridSearchCV(clf, {"foo_param": [1, 2, 3]}, refit=True, cv=2)
    grid_search.fit(X, y)
    grid_search_pickled = pickle.loads(pickle.dumps(grid_search))
    assert_array_almost_equal(grid_search.predict(X), grid_search_pickled.predict(X))

    random_search = RandomizedSearchCV(
        clf, {"foo_param": [1, 2, 3]}, refit=True, n_iter=3, cv=2
    )
    random_search.fit(X, y)
    random_search_pickled = pickle.loads(pickle.dumps(random_search))
    assert_array_almost_equal(
        random_search.predict(X), random_search_pickled.predict(X)
    )


def test_grid_search_with_multioutput_data():
    # [FUNCTION DOC]
    # Verifies per-split scores for multi-output tree estimators match manual KFold
    #   fit/score for both grid and randomized search.
    # Test search with multi-output estimator

    X, y = make_multilabel_classification(return_indicator=True, random_state=0)

    est_parameters = {"max_depth": [1, 2, 3, 4]}
    cv = KFold()

    estimators = [
        DecisionTreeRegressor(random_state=0),
        DecisionTreeClassifier(random_state=0),
    ]

    # Test with grid search cv
    for est in estimators:
        grid_search = GridSearchCV(est, est_parameters, cv=cv)
        grid_search.fit(X, y)
        res_params = grid_search.cv_results_["params"]
        for cand_i in range(len(res_params)):
            est.set_params(**res_params[cand_i])

            for i, (train, test) in enumerate(cv.split(X, y)):
                est.fit(X[train], y[train])
                correct_score = est.score(X[test], y[test])
                assert_almost_equal(
                    correct_score,
                    grid_search.cv_results_["split%d_test_score" % i][cand_i],
                )

    # Test with a randomized search
    for est in estimators:
        random_search = RandomizedSearchCV(est, est_parameters, cv=cv, n_iter=3)
        random_search.fit(X, y)
        res_params = random_search.cv_results_["params"]
        for cand_i in range(len(res_params)):
            est.set_params(**res_params[cand_i])

            for i, (train, test) in enumerate(cv.split(X, y)):
                est.fit(X[train], y[train])
                correct_score = est.score(X[test], y[test])
                assert_almost_equal(
                    correct_score,
                    random_search.cv_results_["split%d_test_score" % i][cand_i],
                )


def test_predict_proba_disabled():
    # [FUNCTION DOC]
    # Verifies predict_proba is unavailable when the best estimator (SVC with
    #   probability=False) lacks it.
    # Test predict_proba when disabled on estimator.
    X = np.arange(20).reshape(5, -1)
    y = [0, 0, 1, 1, 1]
    clf = SVC()
    gs = GridSearchCV(clf, {}, cv=2).fit(X, y)
    assert not hasattr(gs, "predict_proba")


def test_grid_search_allows_nans():
    # [FUNCTION DOC]
    # Verifies NaNs in X are accepted when a pipeline imputes them (smoke test, no
    #   assertions).
    # Test GridSearchCV with SimpleImputer
    X = np.arange(20, dtype=np.float64).reshape(5, -1)
    X[2, :] = np.nan
    y = [0, 0, 1, 1, 1]
    p = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="mean", missing_values=np.nan)),
            ("classifier", MockClassifier()),
        ]
    )
    GridSearchCV(p, {"classifier__foo_param": [1, 2, 3]}, cv=2).fit(X, y)


class FailingClassifier(BaseEstimator):
    # [CLASS DOC]
    # HELPER: estimator whose fit raises ValueError when parameter equals
    #   FAILING_PARAMETER (2); score is always 0.0.
    """Classifier that raises a ValueError on fit()"""

    FAILING_PARAMETER = 2

    def __init__(self, parameter=None):
        # [FUNCTION DOC]
        # HELPER: stores parameter unchanged.
        self.parameter = parameter

    def fit(self, X, y=None):
        # [FUNCTION DOC]
        # HELPER: raises ValueError("Failing classifier failed as required") if
        #   parameter equals FAILING_PARAMETER; otherwise returns None (not self).
        if self.parameter == FailingClassifier.FAILING_PARAMETER:
            raise ValueError("Failing classifier failed as required")

    def predict(self, X):
        # [FUNCTION DOC]
        # HELPER: returns zeros, one per row of X.
        return np.zeros(X.shape[0])

    def score(self, X=None, Y=None):
        # [FUNCTION DOC]
        # HELPER: always returns 0.0.
        return 0.0


def test_grid_search_failing_classifier():
    # [FUNCTION DOC]
    # Verifies numeric error_score (0.0 and nan): FitFailedWarning with summary of 5/15
    #   failed fits, failed candidate split scores equal error_score, and nan-scored
    #   candidate is ranked last and not chosen.
    # GridSearchCV with on_error != 'raise'
    # Ensures that a warning is raised and score reset where appropriate.

    X, y = make_classification(n_samples=20, n_features=10, random_state=0)

    clf = FailingClassifier()

    # refit=False because we only want to check that errors caused by fits
    # to individual folds will be caught and warnings raised instead. If
    # refit was done, then an exception would be raised on refit and not
    # caught by grid_search (expected behavior), and this would cause an
    # error in this test.
    gs = GridSearchCV(
        clf,
        [{"parameter": [0, 1, 2]}],
        scoring="accuracy",
        refit=False,
        error_score=0.0,
    )

    warning_message = re.compile(
        "5 fits failed.+total of 15.+The score on these"
        r" train-test partitions for these parameters will be set to 0\.0.+"
        "5 fits failed with the following error.+ValueError.+Failing classifier failed"
        " as required",
        flags=re.DOTALL,
    )
    with pytest.warns(FitFailedWarning, match=warning_message):
        gs.fit(X, y)
    n_candidates = len(gs.cv_results_["params"])

    # Ensure that grid scores were set to zero as required for those fits
    # that are expected to fail.
    def get_cand_scores(i):
        # [FUNCTION DOC]
        # HELPER: returns array of per-split test scores for candidate index i from the
        #   current gs (closure, re-bound later).
        return np.array(
            [gs.cv_results_["split%d_test_score" % s][i] for s in range(gs.n_splits_)]
        )

    assert all(
        (
            np.all(get_cand_scores(cand_i) == 0.0)
            for cand_i in range(n_candidates)
            if gs.cv_results_["param_parameter"][cand_i]
            == FailingClassifier.FAILING_PARAMETER
        )
    )

    gs = GridSearchCV(
        clf,
        [{"parameter": [0, 1, 2]}],
        scoring="accuracy",
        refit=False,
        error_score=float("nan"),
    )
    warning_message = re.compile(
        "5 fits failed.+total of 15.+The score on these"
        r" train-test partitions for these parameters will be set to nan.+"
        "5 fits failed with the following error.+ValueError.+Failing classifier failed"
        " as required",
        flags=re.DOTALL,
    )
    with pytest.warns(FitFailedWarning, match=warning_message):
        gs.fit(X, y)
    n_candidates = len(gs.cv_results_["params"])
    assert all(
        np.all(np.isnan(get_cand_scores(cand_i)))
        for cand_i in range(n_candidates)
        if gs.cv_results_["param_parameter"][cand_i]
        == FailingClassifier.FAILING_PARAMETER
    )

    ranks = gs.cv_results_["rank_test_score"]

    # Check that succeeded estimators have lower ranks
    assert ranks[0] <= 2 and ranks[1] <= 2
    # Check that failed estimator has the highest rank
    assert ranks[clf.FAILING_PARAMETER] == 3
    assert gs.best_index_ != clf.FAILING_PARAMETER


def test_grid_search_classifier_all_fits_fail():
    # [FUNCTION DOC]
    # Verifies that when all 15 fits fail, fit raises ValueError summarising the errors
    #   even with numeric error_score.
    X, y = make_classification(n_samples=20, n_features=10, random_state=0)

    clf = FailingClassifier()

    gs = GridSearchCV(
        clf,
        [{"parameter": [FailingClassifier.FAILING_PARAMETER] * 3}],
        error_score=0.0,
    )

    warning_message = re.compile(
        (
            "All the 15 fits failed.+15 fits failed with the following"
            " error.+ValueError.+Failing classifier failed as required"
        ),
        flags=re.DOTALL,
    )
    with pytest.raises(ValueError, match=warning_message):
        gs.fit(X, y)


def test_grid_search_failing_classifier_raise():
    # [FUNCTION DOC]
    # Verifies error_score="raise" propagates the estimator's ValueError.
    # GridSearchCV with on_error == 'raise' raises the error

    X, y = make_classification(n_samples=20, n_features=10, random_state=0)

    clf = FailingClassifier()

    # refit=False because we want to test the behaviour of the grid search part
    gs = GridSearchCV(
        clf,
        [{"parameter": [0, 1, 2]}],
        scoring="accuracy",
        refit=False,
        error_score="raise",
    )

    # FailingClassifier issues a ValueError so this is what we look for.
    with pytest.raises(ValueError):
        gs.fit(X, y)


def test_parameters_sampler_replacement():
    # [FUNCTION DOC]
    # Verifies ParameterSampler with all-list params samples without replacement: warns
    #   and truncates when n_iter exceeds grid size, equals the full grid when n_iter ==
    #   size, gives 99 unique samples from a 1000-point grid, and terminates with a
    #   distribution.
    # raise warning if n_iter is bigger than total parameter space
    params = [
        {"first": [0, 1], "second": ["a", "b", "c"]},
        {"third": ["two", "values"]},
    ]
    sampler = ParameterSampler(params, n_iter=9)
    n_iter = 9
    grid_size = 8
    expected_warning = (
        "The total space of parameters %d is smaller "
        "than n_iter=%d. Running %d iterations. For "
        "exhaustive searches, use GridSearchCV." % (grid_size, n_iter, grid_size)
    )
    with pytest.warns(UserWarning, match=expected_warning):
        list(sampler)

    # degenerates to GridSearchCV if n_iter the same as grid_size
    sampler = ParameterSampler(params, n_iter=8)
    samples = list(sampler)
    assert len(samples) == 8
    for values in ParameterGrid(params):
        assert values in samples
    assert len(ParameterSampler(params, n_iter=1000)) == 8

    # test sampling without replacement in a large grid
    params = {"a": range(10), "b": range(10), "c": range(10)}
    sampler = ParameterSampler(params, n_iter=99, random_state=42)
    samples = list(sampler)
    assert len(samples) == 99
    hashable_samples = ["a%db%dc%d" % (p["a"], p["b"], p["c"]) for p in samples]
    assert len(set(hashable_samples)) == 99

    # doesn't go into infinite loops
    params_distribution = {"first": bernoulli(0.5), "second": ["a", "b", "c"]}
    sampler = ParameterSampler(params_distribution, n_iter=7)
    samples = list(sampler)
    assert len(samples) == 7


def test_stochastic_gradient_loss_param():
    # [FUNCTION DOC]
    # Verifies predict_proba availability follows the loss chosen in the grid (log_loss
    #   -> available after fit, hinge -> never), not the base estimator's loss.
    # Make sure the predict_proba works when loss is specified
    # as one of the parameters in the param_grid.
    param_grid = {
        "loss": ["log_loss"],
    }
    X = np.arange(24).reshape(6, -1)
    y = [0, 0, 0, 1, 1, 1]
    clf = GridSearchCV(
        estimator=SGDClassifier(loss="hinge"), param_grid=param_grid, cv=3
    )

    # When the estimator is not fitted, `predict_proba` is not available as the
    # loss is 'hinge'.
    assert not hasattr(clf, "predict_proba")
    clf.fit(X, y)
    clf.predict_proba(X)
    clf.predict_log_proba(X)

    # Make sure `predict_proba` is not available when setting loss=['hinge']
    # in param_grid
    param_grid = {
        "loss": ["hinge"],
    }
    clf = GridSearchCV(
        estimator=SGDClassifier(loss="hinge"), param_grid=param_grid, cv=3
    )
    assert not hasattr(clf, "predict_proba")
    clf.fit(X, y)
    assert not hasattr(clf, "predict_proba")


def test_search_train_scores_set_to_false():
    # [FUNCTION DOC]
    # Smoke test that fit works with default return_train_score.
    # KNOWN ISSUE: no assertions; the name suggests train scores are checked but nothing
    #   is verified.
    X = np.arange(6).reshape(6, -1)
    y = [0, 0, 0, 1, 1, 1]
    clf = LinearSVC(random_state=0)

    gs = GridSearchCV(clf, param_grid={"C": [0.1, 0.2]}, cv=3)
    gs.fit(X, y)


def test_grid_search_cv_splits_consistency():
    # [FUNCTION DOC]
    # Verifies cv accepts a one-time splitter and a generator, split is called once (not
    #   per candidate) so results match KFold, and identical candidates get identical
    #   per-split train/test scores.
    # Check if a one time iterable is accepted as a cv parameter.
    n_samples = 100
    n_splits = 5
    X, y = make_classification(n_samples=n_samples, random_state=0)

    gs = GridSearchCV(
        LinearSVC(random_state=0),
        param_grid={"C": [0.1, 0.2, 0.3]},
        cv=OneTimeSplitter(n_splits=n_splits, n_samples=n_samples),
        return_train_score=True,
    )
    gs.fit(X, y)

    gs2 = GridSearchCV(
        LinearSVC(random_state=0),
        param_grid={"C": [0.1, 0.2, 0.3]},
        cv=KFold(n_splits=n_splits),
        return_train_score=True,
    )
    gs2.fit(X, y)

    # Give generator as a cv parameter
    assert isinstance(
        KFold(n_splits=n_splits, shuffle=True, random_state=0).split(X, y),
        GeneratorType,
    )
    gs3 = GridSearchCV(
        LinearSVC(random_state=0),
        param_grid={"C": [0.1, 0.2, 0.3]},
        cv=KFold(n_splits=n_splits, shuffle=True, random_state=0).split(X, y),
        return_train_score=True,
    )
    gs3.fit(X, y)

    gs4 = GridSearchCV(
        LinearSVC(random_state=0),
        param_grid={"C": [0.1, 0.2, 0.3]},
        cv=KFold(n_splits=n_splits, shuffle=True, random_state=0),
        return_train_score=True,
    )
    gs4.fit(X, y)

    def _pop_time_keys(cv_results):
        # [FUNCTION DOC]
        # HELPER: removes the four timing keys from cv_results in place and returns it.
        #   Side effect: mutates the fitted searches' cv_results_.
        for key in (
            "mean_fit_time",
            "std_fit_time",
            "mean_score_time",
            "std_score_time",
        ):
            cv_results.pop(key)
        return cv_results

    # Check if generators are supported as cv and
    # that the splits are consistent
    np.testing.assert_equal(
        _pop_time_keys(gs3.cv_results_), _pop_time_keys(gs4.cv_results_)
    )

    # OneTimeSplitter is a non-re-entrant cv where split can be called only
    # once if ``cv.split`` is called once per param setting in GridSearchCV.fit
    # the 2nd and 3rd parameter will not be evaluated as no train/test indices
    # will be generated for the 2nd and subsequent cv.split calls.
    # This is a check to make sure cv.split is not called once per param
    # setting.
    np.testing.assert_equal(
        {k: v for k, v in gs.cv_results_.items() if not k.endswith("_time")},
        {k: v for k, v in gs2.cv_results_.items() if not k.endswith("_time")},
    )

    # Check consistency of folds across the parameters
    gs = GridSearchCV(
        LinearSVC(random_state=0),
        param_grid={"C": [0.1, 0.1, 0.2, 0.2]},
        cv=KFold(n_splits=n_splits, shuffle=True),
        return_train_score=True,
    )
    gs.fit(X, y)

    # As the first two param settings (C=0.1) and the next two param
    # settings (C=0.2) are same, the test and train scores must also be
    # same as long as the same train/test indices are generated for all
    # the cv splits, for both param setting
    for score_type in ("train", "test"):
        per_param_scores = {}
        for param_i in range(4):
            per_param_scores[param_i] = [
                gs.cv_results_["split%d_%s_score" % (s, score_type)][param_i]
                for s in range(5)
            ]

        assert_array_almost_equal(per_param_scores[0], per_param_scores[1])
        assert_array_almost_equal(per_param_scores[2], per_param_scores[3])


def test_transform_inverse_transform_round_trip():
    # [FUNCTION DOC]
    # Verifies inverse_transform(transform(X)) returns X through the best estimator.
    clf = MockClassifier()
    grid_search = GridSearchCV(clf, {"foo_param": [1, 2, 3]}, cv=2, verbose=3)

    grid_search.fit(X, y)
    X_round_trip = grid_search.inverse_transform(grid_search.transform(X))
    assert_array_equal(X, X_round_trip)


def test_custom_run_search():
    # [FUNCTION DOC]
    # Verifies a BaseSearchCV subclass with a custom _run_search calling evaluate twice
    #   yields cumulative results equal to GridSearchCV over the union grid, and all
    #   public fitted attributes (except a few) match.
    def check_results(results, gscv):
        # [FUNCTION DOC]
        # HELPER: asserts results have the same keys as gscv.cv_results_ and all non-
        #   time values are equal (exact for object arrays, allclose otherwise). Side
        #   effect: converts results values to arrays in place.
        exp_results = gscv.cv_results_
        assert sorted(results.keys()) == sorted(exp_results)
        for k in results:
            if not k.endswith("_time"):
                # XXX: results['params'] is a list :|
                results[k] = np.asanyarray(results[k])
                if results[k].dtype.kind == "O":
                    assert_array_equal(
                        exp_results[k], results[k], err_msg="Checking " + k
                    )
                else:
                    assert_allclose(exp_results[k], results[k], err_msg="Checking " + k)

    def fit_grid(param_grid):
        # [FUNCTION DOC]
        # HELPER: fits and returns GridSearchCV(clf, param_grid,
        #   return_train_score=True) on the enclosing X, y.
        return GridSearchCV(clf, param_grid, return_train_score=True).fit(X, y)

    class CustomSearchCV(BaseSearchCV):
        # [CLASS DOC]
        # HELPER: BaseSearchCV subclass that evaluates two hard-coded candidate batches
        #   and checks intermediate results.
        def __init__(self, estimator, **kwargs):
            # [FUNCTION DOC]
            # HELPER: forwards to BaseSearchCV.__init__.
            super().__init__(estimator, **kwargs)

        def _run_search(self, evaluate):
            # [FUNCTION DOC]
            # HELPER: calls evaluate for max_depth candidates, checks against a grid
            #   search, then for min_samples_split candidates, checking that results
            #   accumulate.
            results = evaluate([{"max_depth": 1}, {"max_depth": 2}])
            check_results(results, fit_grid({"max_depth": [1, 2]}))
            results = evaluate([{"min_samples_split": 5}, {"min_samples_split": 10}])
            check_results(
                results,
                fit_grid([{"max_depth": [1, 2]}, {"min_samples_split": [5, 10]}]),
            )

    # Using regressor to make sure each score differs
    clf = DecisionTreeRegressor(random_state=0)
    X, y = make_classification(n_samples=100, n_informative=4, random_state=0)
    mycv = CustomSearchCV(clf, return_train_score=True).fit(X, y)
    gscv = fit_grid([{"max_depth": [1, 2]}, {"min_samples_split": [5, 10]}])

    results = mycv.cv_results_
    check_results(results, gscv)
    for attr in dir(gscv):
        if (
            attr[0].islower()
            and attr[-1:] == "_"
            and attr
            not in {
                "cv_results_",
                "best_estimator_",
                "refit_time_",
                "classes_",
                "scorer_",
            }
        ):
            assert getattr(gscv, attr) == getattr(mycv, attr), (
                "Attribute %s not equal" % attr
            )


def test__custom_fit_no_run_search():
    # [FUNCTION DOC]
    # Verifies a subclass overriding fit need not implement _run_search, while one that
    #   does neither raises NotImplementedError at fit.
    class NoRunSearchSearchCV(BaseSearchCV):
        # [CLASS DOC]
        # HELPER: subclass with a no-op fit.
        def __init__(self, estimator, **kwargs):
            # [FUNCTION DOC]
            # HELPER: forwards to BaseSearchCV.__init__.
            super().__init__(estimator, **kwargs)

        def fit(self, X, y=None, groups=None, **fit_params):
            # [FUNCTION DOC]
            # HELPER: returns self without doing anything.
            return self

    # this should not raise any exceptions
    NoRunSearchSearchCV(SVC()).fit(X, y)

    class BadSearchCV(BaseSearchCV):
        # [CLASS DOC]
        # HELPER: subclass without _run_search.
        def __init__(self, estimator, **kwargs):
            # [FUNCTION DOC]
            # HELPER: forwards to BaseSearchCV.__init__.
            super().__init__(estimator, **kwargs)

    with pytest.raises(NotImplementedError, match="_run_search not implemented."):
        # this should raise a NotImplementedError
        BadSearchCV(SVC(), cv=KFold(n_splits=2)).fit(X, y)


# TODO: remove mark once loky bug is fixed:
# https://github.com/joblib/loky/issues/458
@pytest.mark.thread_unsafe
def test_empty_cv_iterator_error():
    # [FUNCTION DOC]
    # Verifies an exhausted cv generator leads to ValueError "No fits were performed".
    # Use global X, y

    # create cv
    cv = KFold(n_splits=3).split(X)

    # pop all of it, this should cause the expected ValueError
    [u for u in cv]
    # cv is empty now

    train_size = 100
    ridge = RandomizedSearchCV(Ridge(), {"alpha": [1e-3, 1e-2, 1e-1]}, cv=cv, n_jobs=4)

    # assert that this raises an error
    with pytest.raises(
        ValueError,
        match=(
            "No fits were performed. "
            "Was the CV iterator empty\\? "
            "Were there no candidates\\?"
        ),
    ):
        ridge.fit(X[:train_size], y[:train_size])


# TODO: remove mark once loky bug is fixed:
# https://github.com/joblib/loky/issues/458
def test_random_search_bad_cv():
    # [FUNCTION DOC]
    # Verifies ValueError when cv.get_n_splits disagrees with the number of splits
    #   produced.
    # KNOWN ISSUE: the TODO comment above refers to a thread_unsafe mark that this test
    #   does not have.
    # Use global X, y

    class BrokenKFold(KFold):
        # [CLASS DOC]
        # HELPER: KFold whose get_n_splits always reports 1.
        def get_n_splits(self, *args, **kw):
            # [FUNCTION DOC]
            # HELPER: returns 1 regardless of arguments.
            return 1

    # create bad cv
    cv = BrokenKFold(n_splits=3)

    train_size = 100
    ridge = RandomizedSearchCV(Ridge(), {"alpha": [1e-3, 1e-2, 1e-1]}, cv=cv, n_jobs=4)

    # assert that this raises an error
    with pytest.raises(
        ValueError,
        match=(
            "cv.split and cv.get_n_splits return "
            "inconsistent results. Expected \\d+ "
            "splits, got \\d+"
        ),
    ):
        ridge.fit(X[:train_size], y[:train_size])


@pytest.mark.parametrize("return_train_score", [False, True])
@pytest.mark.parametrize(
    "SearchCV, specialized_params",
    [
        (GridSearchCV, {"param_grid": {"max_depth": [2, 3, 5, 8]}}),
        (
            RandomizedSearchCV,
            {"param_distributions": {"max_depth": [2, 3, 5, 8]}, "n_iter": 4},
        ),
    ],
)
def test_searchcv_raise_warning_with_non_finite_score(
    SearchCV, specialized_params, return_train_score
):
    # [FUNCTION DOC]
    # Verifies a UserWarning per score set (test and optionally train) when some scores
    #   are NaN, and that NaN-mean candidates share the worst rank.
    # Non-regression test for:
    # https://github.com/scikit-learn/scikit-learn/issues/10529
    # Check that we raise a UserWarning when a non-finite score is
    # computed in the SearchCV
    X, y = make_classification(n_classes=2, random_state=0)

    class FailingScorer:
        # [CLASS DOC]
        # HELPER: stateful scorer returning NaN on every 5th call, 1 otherwise.
        """Scorer that will fail for some split but not all."""

        def __init__(self):
            # [FUNCTION DOC]
            # HELPER: initialises call counter n_counts to 0.
            self.n_counts = 0

        def __call__(self, estimator, X, y):
            # [FUNCTION DOC]
            # HELPER: increments n_counts; returns NaN when it is a multiple of 5, else
            #   1.
            self.n_counts += 1
            if self.n_counts % 5 == 0:
                return np.nan
            return 1

    grid = SearchCV(
        DecisionTreeClassifier(),
        scoring=FailingScorer(),
        cv=3,
        return_train_score=return_train_score,
        **specialized_params,
    )

    with pytest.warns(UserWarning) as warn_msg:
        grid.fit(X, y)

    set_with_warning = ["test", "train"] if return_train_score else ["test"]
    assert len(warn_msg) == len(set_with_warning)
    for msg, dataset in zip(warn_msg, set_with_warning):
        assert f"One or more of the {dataset} scores are non-finite" in str(msg.message)

    # all non-finite scores should be equally ranked last
    last_rank = grid.cv_results_["rank_test_score"].max()
    non_finite_mask = np.isnan(grid.cv_results_["mean_test_score"])
    assert_array_equal(grid.cv_results_["rank_test_score"][non_finite_mask], last_rank)
    # all finite scores should be better ranked than the non-finite scores
    assert np.all(grid.cv_results_["rank_test_score"][~non_finite_mask] < last_rank)


def test_callable_multimetric_confusion_matrix():
    # [FUNCTION DOC]
    # Verifies a callable returning a dict of metrics creates mean_test_<name> keys and
    #   refit="fp" makes score return that metric.
    # Test callable with many metrics inserts the correct names and metrics
    # into the search cv object
    def custom_scorer(clf, X, y):
        # [FUNCTION DOC]
        # HELPER: returns dict of tn, fp, fn, tp from the binary confusion matrix of clf
        #   predictions.
        y_pred = clf.predict(X)
        cm = confusion_matrix(y, y_pred)
        return {"tn": cm[0, 0], "fp": cm[0, 1], "fn": cm[1, 0], "tp": cm[1, 1]}

    X, y = make_classification(n_samples=40, n_features=4, random_state=42)
    est = LinearSVC(random_state=42)
    search = GridSearchCV(est, {"C": [0.1, 1]}, scoring=custom_scorer, refit="fp")

    search.fit(X, y)

    score_names = ["tn", "fp", "fn", "tp"]
    for name in score_names:
        assert "mean_test_{}".format(name) in search.cv_results_

    y_pred = search.predict(X)
    cm = confusion_matrix(y, y_pred)
    assert search.score(X, y) == pytest.approx(cm[0, 1])


def test_callable_multimetric_same_as_list_of_strings():
    # [FUNCTION DOC]
    # Verifies a callable dict scorer gives the same best_score_, best_index_ and score
    #   as the equivalent list of metric names.
    # Test callable multimetric is the same as a list of strings
    def custom_scorer(est, X, y):
        # [FUNCTION DOC]
        # HELPER: returns dict with recall and accuracy of est predictions.
        y_pred = est.predict(X)
        return {
            "recall": recall_score(y, y_pred),
            "accuracy": accuracy_score(y, y_pred),
        }

    X, y = make_classification(n_samples=40, n_features=4, random_state=42)
    est = LinearSVC(random_state=42)
    search_callable = GridSearchCV(
        est, {"C": [0.1, 1]}, scoring=custom_scorer, refit="recall"
    )
    search_str = GridSearchCV(
        est, {"C": [0.1, 1]}, scoring=["recall", "accuracy"], refit="recall"
    )

    search_callable.fit(X, y)
    search_str.fit(X, y)

    assert search_callable.best_score_ == pytest.approx(search_str.best_score_)
    assert search_callable.best_index_ == search_str.best_index_
    assert search_callable.score(X, y) == pytest.approx(search_str.score(X, y))


def test_callable_single_metric_same_as_single_string():
    # [FUNCTION DOC]
    # Verifies a callable single-value scorer, scoring="recall" and scoring=["recall"]
    #   give the same best_score_, best_index_ and score.
    # Tests callable scorer is the same as scoring with a single string
    def custom_scorer(est, X, y):
        # [FUNCTION DOC]
        # HELPER: returns recall of est predictions.
        y_pred = est.predict(X)
        return recall_score(y, y_pred)

    X, y = make_classification(n_samples=40, n_features=4, random_state=42)
    est = LinearSVC(random_state=42)
    search_callable = GridSearchCV(
        est, {"C": [0.1, 1]}, scoring=custom_scorer, refit=True
    )
    search_str = GridSearchCV(est, {"C": [0.1, 1]}, scoring="recall", refit="recall")
    search_list_str = GridSearchCV(
        est, {"C": [0.1, 1]}, scoring=["recall"], refit="recall"
    )
    search_callable.fit(X, y)
    search_str.fit(X, y)
    search_list_str.fit(X, y)

    assert search_callable.best_score_ == pytest.approx(search_str.best_score_)
    assert search_callable.best_index_ == search_str.best_index_
    assert search_callable.score(X, y) == pytest.approx(search_str.score(X, y))

    assert search_list_str.best_score_ == pytest.approx(search_str.best_score_)
    assert search_list_str.best_index_ == search_str.best_index_
    assert search_list_str.score(X, y) == pytest.approx(search_str.score(X, y))


def test_callable_multimetric_error_on_invalid_key():
    # [FUNCTION DOC]
    # Verifies ValueError when refit names a key absent from the callable scorer's
    #   returned dict.
    # Raises when the callable scorer does not return a dict with `refit` key.
    def bad_scorer(est, X, y):
        # [FUNCTION DOC]
        # HELPER: returns {"bad_name": 1}.
        return {"bad_name": 1}

    X, y = make_classification(n_samples=40, n_features=4, random_state=42)
    clf = GridSearchCV(
        LinearSVC(random_state=42),
        {"C": [0.1, 1]},
        scoring=bad_scorer,
        refit="good_name",
    )

    msg = (
        "For multi-metric scoring, the parameter refit must be set to a "
        "scorer key or a callable to refit"
    )
    with pytest.raises(ValueError, match=msg):
        clf.fit(X, y)


def test_callable_multimetric_error_failing_clf():
    # [FUNCTION DOC]
    # Verifies FitFailedWarning with a dict callable scorer and that failing candidates
    #   get error_score (0.1) in mean_test_acc.
    # Warns when there is an estimator the fails to fit with a float
    # error_score
    def custom_scorer(est, X, y):
        # [FUNCTION DOC]
        # HELPER: returns {"acc": 1}.
        return {"acc": 1}

    X, y = make_classification(n_samples=20, n_features=10, random_state=0)

    clf = FailingClassifier()
    gs = GridSearchCV(
        clf,
        [{"parameter": [0, 1, 2]}],
        scoring=custom_scorer,
        refit=False,
        error_score=0.1,
    )

    warning_message = re.compile(
        "5 fits failed.+total of 15.+The score on these"
        r" train-test partitions for these parameters will be set to 0\.1",
        flags=re.DOTALL,
    )
    with pytest.warns(FitFailedWarning, match=warning_message):
        gs.fit(X, y)

    assert_allclose(gs.cv_results_["mean_test_acc"], [1, 1, 0.1])


def test_callable_multimetric_clf_all_fits_fail():
    # [FUNCTION DOC]
    # Verifies ValueError "All the 15 fits failed" with a dict callable scorer when
    #   every fit fails.
    # Warns and raises when all estimator fails to fit.
    def custom_scorer(est, X, y):
        # [FUNCTION DOC]
        # HELPER: returns {"acc": 1}.
        return {"acc": 1}

    X, y = make_classification(n_samples=20, n_features=10, random_state=0)

    clf = FailingClassifier()

    gs = GridSearchCV(
        clf,
        [{"parameter": [FailingClassifier.FAILING_PARAMETER] * 3}],
        scoring=custom_scorer,
        refit=False,
        error_score=0.1,
    )

    individual_fit_error_message = "ValueError: Failing classifier failed as required"
    error_message = re.compile(
        (
            "All the 15 fits failed.+your model is misconfigured.+"
            f"{individual_fit_error_message}"
        ),
        flags=re.DOTALL,
    )

    with pytest.raises(ValueError, match=error_message):
        gs.fit(X, y)


def test_n_features_in():
    # [FUNCTION DOC]
    # Verifies n_features_in_ is absent before fit and delegated from the best estimator
    #   after fit for grid and randomized search.
    # make sure grid search and random search delegate n_features_in to the
    # best estimator
    n_features = 4
    X, y = make_classification(n_features=n_features)
    gbdt = HistGradientBoostingClassifier()
    param_grid = {"max_iter": [3, 4]}
    gs = GridSearchCV(gbdt, param_grid)
    rs = RandomizedSearchCV(gbdt, param_grid, n_iter=1)
    assert not hasattr(gs, "n_features_in_")
    assert not hasattr(rs, "n_features_in_")
    gs.fit(X, y)
    rs.fit(X, y)
    assert gs.n_features_in_ == n_features
    assert rs.n_features_in_ == n_features


@pytest.mark.parametrize("pairwise", [True, False])
def test_search_cv_pairwise_property_delegated_to_base_estimator(pairwise):
    # [FUNCTION DOC]
    # Verifies the search's input_tags.pairwise tag mirrors the estimator's tag
    #   (parametrized True/False).
    """
    Test implementation of BaseSearchCV has the pairwise tag
    which matches the pairwise tag of its estimator.
    This test make sure pairwise tag is delegated to the base estimator.

    Non-regression test for issue #13920.
    """

    class TestEstimator(BaseEstimator):
        # [CLASS DOC]
        # HELPER: estimator whose pairwise tag is the closure value pairwise.
        def __sklearn_tags__(self):
            # [FUNCTION DOC]
            # HELPER: returns parent tags with input_tags.pairwise set to the closure
            #   pairwise.
            tags = super().__sklearn_tags__()
            tags.input_tags.pairwise = pairwise
            return tags

    est = TestEstimator()
    attr_message = "BaseSearchCV pairwise tag must match estimator"
    cv = GridSearchCV(est, {"n_neighbors": [10]})
    assert pairwise == cv.__sklearn_tags__().input_tags.pairwise, attr_message


def test_search_cv__pairwise_property_delegated_to_base_estimator():
    # [FUNCTION DOC]
    # Verifies the pairwise tag follows the estimator after toggling its pairwise param
    #   via set_params.
    # KNOWN ISSUE: docstring/message mention a _pairwise property, but only the tag is
    #   checked.
    """
    Test implementation of BaseSearchCV has the pairwise property
    which matches the pairwise tag of its estimator.
    This test make sure pairwise tag is delegated to the base estimator.

    Non-regression test for issue #13920.
    """

    class EstimatorPairwise(BaseEstimator):
        # [CLASS DOC]
        # HELPER: estimator with a pairwise hyperparameter controlling its pairwise tag.
        def __init__(self, pairwise=True):
            # [FUNCTION DOC]
            # HELPER: stores pairwise (default True).
            self.pairwise = pairwise

        def __sklearn_tags__(self):
            # [FUNCTION DOC]
            # HELPER: returns parent tags with input_tags.pairwise set to self.pairwise.
            tags = super().__sklearn_tags__()
            tags.input_tags.pairwise = self.pairwise
            return tags

    est = EstimatorPairwise()
    attr_message = "BaseSearchCV _pairwise property must match estimator"

    for _pairwise_setting in [True, False]:
        est.set_params(pairwise=_pairwise_setting)
        cv = GridSearchCV(est, {"n_neighbors": [10]})
        assert _pairwise_setting == cv.__sklearn_tags__().input_tags.pairwise, (
            attr_message
        )


def test_search_cv_pairwise_property_equivalence_of_precomputed():
    # [FUNCTION DOC]
    # Verifies KNN search on raw features and on a precomputed euclidean distance matrix
    #   yield identical predictions (pairwise splitting works).
    """
    Test implementation of BaseSearchCV has the pairwise tag
    which matches the pairwise tag of its estimator.
    This test ensures the equivalence of 'precomputed'.

    Non-regression test for issue #13920.
    """
    n_samples = 50
    n_splits = 2
    X, y = make_classification(n_samples=n_samples, random_state=0)
    grid_params = {"n_neighbors": [10]}

    # defaults to euclidean metric (minkowski p = 2)
    clf = KNeighborsClassifier()
    cv = GridSearchCV(clf, grid_params, cv=n_splits)
    cv.fit(X, y)
    preds_original = cv.predict(X)

    # precompute euclidean metric to validate pairwise is working
    X_precomputed = euclidean_distances(X)
    clf = KNeighborsClassifier(metric="precomputed")
    cv = GridSearchCV(clf, grid_params, cv=n_splits)
    cv.fit(X_precomputed, y)
    preds_precomputed = cv.predict(X_precomputed)

    attr_message = "GridSearchCV not identical with precomputed metric"
    assert (preds_original == preds_precomputed).all(), attr_message


@pytest.mark.parametrize(
    "SearchCV, param_search",
    [(GridSearchCV, {"a": [0.1, 0.01]}), (RandomizedSearchCV, {"a": uniform(1, 3)})],
)
def test_scalar_fit_param(SearchCV, param_search):
    # [FUNCTION DOC]
    # Verifies a scalar fit param is passed unsliced to the estimator
    #   (best_estimator_.r_ == 42).
    # unofficially sanctioned tolerance for scalar values in fit_params
    # non-regression test for:
    # https://github.com/scikit-learn/scikit-learn/issues/15805
    class TestEstimator(ClassifierMixin, BaseEstimator):
        # [CLASS DOC]
        # HELPER: classifier storing the fit kwarg r as r_ and predicting zeros.
        def __init__(self, a=None):
            # [FUNCTION DOC]
            # HELPER: stores a (default None).
            self.a = a

        def fit(self, X, y, r=None):
            # [FUNCTION DOC]
            # HELPER: sets r_ to r; returns None.
            self.r_ = r

        def predict(self, X):
            # [FUNCTION DOC]
            # HELPER: returns float zeros of length len(X).
            return np.zeros(shape=(len(X)))

    model = SearchCV(TestEstimator(), param_search)
    X, y = make_classification(random_state=42)
    model.fit(X, y, r=42)
    assert model.best_estimator_.r_ == 42


@pytest.mark.parametrize(
    "SearchCV, param_search",
    [
        (GridSearchCV, {"alpha": [0.1, 0.01]}),
        (RandomizedSearchCV, {"alpha": uniform(0.01, 0.1)}),
    ],
)
def test_scalar_fit_param_compat(SearchCV, param_search):
    # [FUNCTION DOC]
    # Verifies non-sample-aligned fit params (scalar, callable, tuple of arrays) are
    #   passed through unmodified (tuple kept as tuple).
    # check support for scalar values in fit_params, for instance in LightGBM
    # that do not exactly respect the scikit-learn API contract but that we do
    # not want to break without an explicit deprecation cycle and API
    # recommendations for implementing early stopping with a user provided
    # validation set. non-regression test for:
    # https://github.com/scikit-learn/scikit-learn/issues/15805
    X_train, X_valid, y_train, y_valid = train_test_split(
        *make_classification(random_state=42), random_state=42
    )

    class _FitParamClassifier(SGDClassifier):
        # [CLASS DOC]
        # HELPER: SGDClassifier whose fit asserts on the extra fit params it receives.
        def fit(
            self,
            X,
            y,
            sample_weight=None,
            tuple_of_arrays=None,
            scalar_param=None,
            callable_param=None,
        ):
            # [FUNCTION DOC]
            # HELPER: fits SGDClassifier with sample_weight, then asserts scalar_param >
            #   0, callable_param is callable, tuple_of_arrays is a tuple of a 2D and a
            #   1D array. Returns self.
            # Exceptions: AssertionError on unexpected params (or TypeError if None
            #   compared).
            super().fit(X, y, sample_weight=sample_weight)
            assert scalar_param > 0
            assert callable(callable_param)

            # The tuple of arrays should be preserved as tuple.
            assert isinstance(tuple_of_arrays, tuple)
            assert tuple_of_arrays[0].ndim == 2
            assert tuple_of_arrays[1].ndim == 1
            return self

    def _fit_param_callable():
        # [FUNCTION DOC]
        # HELPER: no-op callable used as a fit param.
        pass

    model = SearchCV(_FitParamClassifier(), param_search)

    # NOTE: `fit_params` should be data dependent (e.g. `sample_weight`) which
    # is not the case for the following parameters. But this abuse is common in
    # popular third-party libraries and we should tolerate this behavior for
    # now and be careful not to break support for those without following
    # proper deprecation cycle.
    fit_params = {
        "tuple_of_arrays": (X_valid, y_valid),
        "callable_param": _fit_param_callable,
        "scalar_param": 42,
    }
    model.fit(X_train, y_train, **fit_params)


# FIXME: Replace this test with a full `check_estimator` once we have API only
# checks.
@pytest.mark.filterwarnings("ignore:The total space of parameters 4 is")
@pytest.mark.parametrize("SearchCV", [GridSearchCV, RandomizedSearchCV])
@pytest.mark.parametrize("Predictor", [MinimalRegressor, MinimalClassifier])
def test_search_cv_using_minimal_compatible_estimator(SearchCV, Predictor):
    # [FUNCTION DOC]
    # Verifies search works with minimal non-BaseEstimator estimators in a Pipeline:
    #   best_params_ keys match, predictions are constant (1 for classifier, mean for
    #   regressor) and score equals accuracy / r2.
    # Check that third-party library can run tests without inheriting from
    # BaseEstimator.
    rng = np.random.RandomState(0)
    X, y = rng.randn(25, 2), np.array([0] * 5 + [1] * 20)

    model = Pipeline(
        [("transformer", MinimalTransformer()), ("predictor", Predictor())]
    )

    params = {
        "transformer__param": [1, 10],
        "predictor__parama": [1, 10],
    }
    search = SearchCV(model, params, error_score="raise")
    search.fit(X, y)

    assert search.best_params_.keys() == params.keys()

    y_pred = search.predict(X)
    if is_classifier(search):
        assert_array_equal(y_pred, 1)
        assert search.score(X, y) == pytest.approx(accuracy_score(y, y_pred))
    else:
        assert_allclose(y_pred, y.mean())
        assert search.score(X, y) == pytest.approx(r2_score(y, y_pred))


@pytest.mark.parametrize("return_train_score", [True, False])
def test_search_cv_verbose_3(capsys, return_train_score):
    # [FUNCTION DOC]
    # Verifies verbose=3 prints one score line per split (3), formatted with train and
    #   test when return_train_score is True. Uses capsys.
    """Check that search cv with verbose>2 shows the score for single
    metrics. non-regression test for #19658."""
    X, y = make_classification(n_samples=100, n_classes=2, flip_y=0.2, random_state=0)
    clf = LinearSVC(random_state=0)
    grid = {"C": [0.1]}

    GridSearchCV(
        clf,
        grid,
        scoring="accuracy",
        verbose=3,
        cv=3,
        return_train_score=return_train_score,
    ).fit(X, y)
    captured = capsys.readouterr().out
    if return_train_score:
        match = re.findall(r"score=\(train=[\d\.]+, test=[\d.]+\)", captured)
    else:
        match = re.findall(r"score=[\d\.]+", captured)
    assert len(match) == 3


@pytest.mark.parametrize(
    "SearchCV, param_search",
    [
        (GridSearchCV, "param_grid"),
        (RandomizedSearchCV, "param_distributions"),
        (HalvingGridSearchCV, "param_grid"),
    ],
)
def test_search_estimator_param(SearchCV, param_search):
    # [FUNCTION DOC]
    # Verifies estimator objects in the param grid are not mutated by setting nested
    #   params (clf__C), while best_estimator_ has the requested C.
    # test that SearchCV object doesn't change the object given in the parameter grid
    X, y = make_classification(random_state=42)

    params = {"clf": [LinearSVC()], "clf__C": [0.01]}
    orig_C = params["clf"][0].C

    pipe = Pipeline([("trs", MinimalTransformer()), ("clf", None)])

    param_grid_search = {param_search: params}
    gs = SearchCV(pipe, refit=True, cv=2, scoring="accuracy", **param_grid_search).fit(
        X, y
    )

    # testing that the original object in params is not changed
    assert params["clf"][0].C == orig_C
    # testing that the GS is setting the parameter of the step correctly
    assert gs.best_estimator_.named_steps["clf"].C == 0.01


# TODO: remove mark once loky bug is fixed:
# https://github.com/joblib/loky/issues/458
@pytest.mark.thread_unsafe
def test_search_with_2d_array():
    # [FUNCTION DOC]
    # Verifies tuple-valued params (ngram_range) are stored as tuples in an object
    #   MaskedArray in cv_results_.
    parameter_grid = {
        "vect__ngram_range": ((1, 1), (1, 2)),  # unigrams or bigrams
        "vect__norm": ("l1", "l2"),
    }
    pipeline = Pipeline(
        [
            ("vect", TfidfVectorizer()),
            ("clf", ComplementNB()),
        ]
    )
    random_search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=parameter_grid,
        n_iter=3,
        random_state=0,
        n_jobs=2,
        verbose=1,
        cv=3,
    )
    data_train = ["one", "two", "three", "four", "five"]
    data_target = [0, 0, 1, 0, 1]
    random_search.fit(data_train, data_target)
    result = random_search.cv_results_["param_vect__ngram_range"]
    expected_data = np.empty(3, dtype=object)
    expected_data[:] = [(1, 2), (1, 2), (1, 1)]
    np.testing.assert_array_equal(result.data, expected_data)


def test_search_html_repr():
    # [FUNCTION DOC]
    # Verifies HTML repr shows the original pipeline when unfitted or refit=False, and
    #   the best estimator after fitting with refit=True.
    """Test different HTML representations for GridSearchCV."""
    X, y = make_classification(random_state=42)

    pipeline = Pipeline([("scale", StandardScaler()), ("clf", DummyClassifier())])
    param_grid = {"clf": [DummyClassifier(), LogisticRegression()]}

    # Unfitted shows the original pipeline
    search_cv = GridSearchCV(pipeline, param_grid=param_grid, refit=False)
    with config_context(display="diagram"):
        repr_html = search_cv._repr_html_()
        assert "<div>DummyClassifier</div>" in repr_html

    # Fitted with `refit=False` shows the original pipeline
    search_cv.fit(X, y)
    with config_context(display="diagram"):
        repr_html = search_cv._repr_html_()
        assert "<div>DummyClassifier</div>" in repr_html

    # Fitted with `refit=True` shows the best estimator
    search_cv = GridSearchCV(pipeline, param_grid=param_grid, refit=True)
    search_cv.fit(X, y)
    with config_context(display="diagram"):
        repr_html = search_cv._repr_html_()
        assert "<div>DummyClassifier</div>" not in repr_html
        assert "<div>LogisticRegression</div>" in repr_html


# Metadata Routing Tests
# ======================


@pytest.mark.parametrize(
    "SearchCV, param_search",
    [
        (GridSearchCV, "param_grid"),
        (RandomizedSearchCV, "param_distributions"),
    ],
)
@config_context(enable_metadata_routing=True)
def test_multi_metric_search_forwards_metadata(SearchCV, param_search):
    # [FUNCTION DOC]
    # Verifies with metadata routing enabled that score metadata is routed to a
    #   ConsumingScorer in multi-metric scoring and recorded with split-sliced values.
    """Test that *SearchCV forwards metadata correctly when passed multiple metrics."""
    X, y = make_classification(random_state=42)
    n_samples = _num_samples(X)
    rng = np.random.RandomState(0)
    score_weights = rng.rand(n_samples)
    score_metadata = rng.rand(n_samples)

    est = LinearSVC()
    param_grid_search = {param_search: {"C": [1]}}

    scorer_registry = _Registry()
    scorer = ConsumingScorer(registry=scorer_registry).set_score_request(
        sample_weight="score_weights", metadata="score_metadata"
    )
    scoring = dict(my_scorer=scorer, accuracy="accuracy")
    SearchCV(est, refit="accuracy", cv=2, scoring=scoring, **param_grid_search).fit(
        X, y, score_weights=score_weights, score_metadata=score_metadata
    )
    assert len(scorer_registry)
    for _scorer in scorer_registry:
        check_recorded_metadata(
            obj=_scorer,
            method="consuming_metric",
            parent="fit",
            split_params=("sample_weight", "metadata"),
            sample_weight=score_weights,
            metadata=score_metadata,
        )


@pytest.mark.parametrize(
    "SearchCV, param_search",
    [
        (GridSearchCV, "param_grid"),
        (RandomizedSearchCV, "param_distributions"),
        (HalvingGridSearchCV, "param_grid"),
    ],
)
def test_score_rejects_params_with_no_routing_enabled(SearchCV, param_search):
    # [FUNCTION DOC]
    # Verifies SearchCV.score raises ValueError when extra params are passed without
    #   metadata routing enabled.
    """*SearchCV should reject **params when metadata routing is not enabled
    since this is added only when routing is enabled."""
    X, y = make_classification(random_state=42)
    est = LinearSVC()
    param_grid_search = {param_search: {"C": [1]}}

    gs = SearchCV(est, cv=2, **param_grid_search).fit(X, y)

    with pytest.raises(ValueError, match="is only supported if"):
        gs.score(X, y, metadata=1)


# End of Metadata Routing Tests
# =============================


def test_cv_results_dtype_issue_29074():
    # [FUNCTION DOC]
    # Non-regression: params whose values make dtype inference fail (None, dicts, lists,
    #   strings) yield object dtype param_* arrays.
    """Non-regression test for https://github.com/scikit-learn/scikit-learn/issues/29074"""

    class MetaEstimator(BaseEstimator, ClassifierMixin):
        # [CLASS DOC]
        # HELPER: meta-classifier wrapping base_clf with four unused hyperparameters.
        def __init__(
            self,
            base_clf,
            parameter1=None,
            parameter2=None,
            parameter3=None,
            parameter4=None,
        ):
            # [FUNCTION DOC]
            # HELPER: stores base_clf and parameter1..parameter4 unchanged.
            self.base_clf = base_clf
            self.parameter1 = parameter1
            self.parameter2 = parameter2
            self.parameter3 = parameter3
            self.parameter4 = parameter4

        def fit(self, X, y=None):
            # [FUNCTION DOC]
            # HELPER: fits base_clf in place (not cloned) and returns self.
            self.base_clf.fit(X, y)
            return self

        def score(self, X, y):
            # [FUNCTION DOC]
            # HELPER: delegates to base_clf.score.
            return self.base_clf.score(X, y)

    # Values of param_grid are such that np.result_type gives slightly
    # different errors, in particular ValueError and TypeError
    param_grid = {
        "parameter1": [None, {"option": "A"}, {"option": "B"}],
        "parameter2": [None, [1, 2]],
        "parameter3": [{"a": 1}],
        "parameter4": ["str1", "str2"],
    }
    grid_search = GridSearchCV(
        estimator=MetaEstimator(LogisticRegression()),
        param_grid=param_grid,
        cv=3,
    )

    X, y = make_blobs(random_state=0)
    grid_search.fit(X, y)
    for param in param_grid:
        assert grid_search.cv_results_[f"param_{param}"].dtype == object


def test_search_with_estimators_issue_29157():
    # [FUNCTION DOC]
    # Non-regression: searching over encoder estimators (which have a dtype param) gives
    #   an object dtype param array. Skipped without pandas.
    """Check cv_results_ for estimators with a `dtype` parameter, e.g. OneHotEncoder."""
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame(
        {
            "numeric_1": [1, 2, 3, 4, 5],
            "object_1": ["a", "a", "a", "a", "a"],
            "target": [1.0, 4.1, 2.0, 3.0, 1.0],
        }
    )
    X = df.drop("target", axis=1)
    y = df["target"]
    enc = ColumnTransformer(
        [("enc", OneHotEncoder(sparse_output=False), ["object_1"])],
        remainder="passthrough",
    )
    pipe = Pipeline(
        [
            ("enc", enc),
            ("regressor", LinearRegression()),
        ]
    )
    grid_params = {
        "enc__enc": [
            OneHotEncoder(sparse_output=False),
            OrdinalEncoder(),
        ]
    }
    grid_search = GridSearchCV(pipe, grid_params, cv=2)
    grid_search.fit(X, y)
    assert grid_search.cv_results_["param_enc__enc"].dtype == object


def test_cv_results_multi_size_array():
    # [FUNCTION DOC]
    # Non-regression: array-valued params of different shapes (spline knots) are stored
    #   in an object dtype param array.
    """Check that GridSearchCV works with params that are arrays of different sizes.

    Non-regression test for #29277.
    """
    n_features = 10
    X, y = make_classification(n_features=10)

    spline_reg_pipe = make_pipeline(
        SplineTransformer(extrapolation="periodic"),
        LogisticRegression(),
    )

    n_knots_list = [n_features * i for i in [10, 11, 12]]
    knots_list = [
        np.linspace(0, np.pi * 2, n_knots).reshape((-1, n_features))
        for n_knots in n_knots_list
    ]
    spline_reg_pipe_cv = GridSearchCV(
        estimator=spline_reg_pipe,
        param_grid={
            "splinetransformer__knots": knots_list,
        },
    )

    spline_reg_pipe_cv.fit(X, y)
    assert (
        spline_reg_pipe_cv.cv_results_["param_splinetransformer__knots"].dtype == object
    )


@pytest.mark.parametrize(
    "array_namespace, device_name, dtype_name",
    yield_namespace_device_dtype_combinations(),
)
@pytest.mark.parametrize("SearchCV", [GridSearchCV, RandomizedSearchCV])
def test_array_api_search_cv_classifier(
    SearchCV, array_namespace, device_name, dtype_name
):
    # [FUNCTION DOC]
    # Smoke test that Grid/RandomizedSearchCV fit and score LinearDiscriminantAnalysis
    #   under array_api_dispatch for each namespace/device/dtype.
    xp, device = _array_api_for_tests(array_namespace, device_name, dtype_name)

    X = np.arange(100).reshape((10, 10))
    X_np = X.astype(dtype_name)
    X_xp = xp.asarray(X_np, device=device)

    # y should always be an integer, no matter what `dtype_name` is
    y_np = np.array([0] * 5 + [1] * 5)
    y_xp = xp.asarray(y_np, device=device)

    with config_context(array_api_dispatch=True):
        searcher = SearchCV(
            LinearDiscriminantAnalysis(),
            {"tol": [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]},
            cv=2,
            error_score="raise",
        )
        searcher.fit(X_xp, y_xp)
        searcher.score(X_xp, y_xp)


# Construct these outside the tests so that the same object is used
# for both input and `expected`
one_hot_encoder = OneHotEncoder()
ordinal_encoder = OrdinalEncoder()

# If we construct this directly via `MaskedArray`, the list of tuples
# gets auto-converted to a 2D array.
ma_with_tuples = np.ma.MaskedArray(np.empty(2), mask=True, dtype=object)
ma_with_tuples[0] = (1, 2)
ma_with_tuples[1] = (3, 4)


@pytest.mark.parametrize(
    ("candidate_params", "expected"),
    [
        pytest.param(
            [{"foo": 1}, {"foo": 2}],
            [
                ("param_foo", np.ma.MaskedArray(np.array([1, 2]))),
            ],
            id="simple numeric, single param",
        ),
        pytest.param(
            [{"foo": 1, "bar": 3}, {"foo": 2, "bar": 4}, {"foo": 3}],
            [
                ("param_foo", np.ma.MaskedArray(np.array([1, 2, 3]))),
                (
                    "param_bar",
                    np.ma.MaskedArray(np.array([3, 4, 0]), mask=[False, False, True]),
                ),
            ],
            id="simple numeric, one param is missing in one round",
        ),
        pytest.param(
            [{"foo": [[1], [2], [3]]}, {"foo": [[1], [2]]}],
            [
                (
                    "param_foo",
                    np.ma.MaskedArray([[[1], [2], [3]], [[1], [2]]], dtype=object),
                ),
            ],
            id="lists of different lengths",
        ),
        pytest.param(
            [{"foo": (1, 2)}, {"foo": (3, 4)}],
            [
                (
                    "param_foo",
                    ma_with_tuples,
                ),
            ],
            id="lists tuples",
        ),
        pytest.param(
            [{"foo": ordinal_encoder}, {"foo": one_hot_encoder}],
            [
                (
                    "param_foo",
                    np.ma.MaskedArray([ordinal_encoder, one_hot_encoder], dtype=object),
                ),
            ],
            id="estimators",
        ),
    ],
)
def test_yield_masked_array_for_each_param(candidate_params, expected):
    # [FUNCTION DOC]
    # Verifies _yield_masked_array_for_each_param produces correct keys, dtypes, values
    #   and masks for numeric, missing, ragged list, tuple and estimator parameter
    #   values.
    result = list(_yield_masked_array_for_each_param(candidate_params))
    for (key, value), (expected_key, expected_value) in zip(result, expected):
        assert key == expected_key
        assert value.dtype == expected_value.dtype
        np.testing.assert_array_equal(value, expected_value)
        np.testing.assert_array_equal(value.mask, expected_value.mask)


def test_yield_masked_array_no_runtime_warning():
    # [FUNCTION DOC]
    # Non-regression: building the masked arrays for 1000 candidates emits no
    #   RuntimeWarning (warnings turned into errors).
    # non-regression test for https://github.com/scikit-learn/scikit-learn/issues/29929
    candidate_params = [{"param": i} for i in range(1000)]
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        list(_yield_masked_array_for_each_param(candidate_params))


def _searchcv_callback_test_cases(estimator, scoring):
    # [FUNCTION DOC]
    # HELPER: returns four unfitted searches (Grid, Randomized, HalvingGrid,
    #   HalvingRandom) over max_iter 1..3 with cv=2 for the given estimator class and
    #   scoring; randomized ones use random_state=42, halving ones
    #   aggressive_elimination.
    return [
        GridSearchCV(estimator(), {"max_iter": [1, 2, 3]}, cv=2, scoring=scoring),
        RandomizedSearchCV(
            estimator(),
            {"max_iter": randint(1, 4)},
            cv=2,
            n_iter=3,
            scoring=scoring,
            random_state=42,
        ),
        HalvingGridSearchCV(
            estimator(),
            {"max_iter": [1, 2, 3]},
            cv=2,
            aggressive_elimination=True,
            scoring=scoring,
        ),
        HalvingRandomSearchCV(
            estimator(),
            {"max_iter": randint(1, 4)},
            cv=2,
            aggressive_elimination=True,
            scoring=scoring,
            random_state=42,
        ),
    ]


@pytest.mark.parametrize("refit", [True, False])
@pytest.mark.parametrize(
    "search",
    _searchcv_callback_test_cases(NoCallbackEstimator, "accuracy"),
)
@skip_callback_test_if_wasm
def test_search_callbacks_no_propagation(search, refit):
    # [FUNCTION DOC]
    # Verifies callback hook counts when the inner estimator does not support callbacks:
    #   one setup/teardown and on_fit_task_begin/end called once per search task (root,
    #   search, optional refit, halving rounds, candidate-split evaluations).
    """Check number of hook calls when the sub-estimator doesn't support callbacks."""
    callbacks = [RecordingCallback(), RecordingAutoPropagatedCallback()]
    search = clone(search).set_params(refit=refit).set_callbacks(*callbacks).fit(X, y)

    # defining expected values
    root = 1
    search_task = 1
    n_splits = search.n_splits_  # 2

    if "Halving" in search.__class__.__name__:
        n_halving_rounds = search.n_iterations_
        n_evaluations = sum(n_cand * n_splits for n_cand in search.n_candidates_)
        expected = root + (search_task + refit) + n_halving_rounds + n_evaluations
    else:  # GridSearchCV, RandomizedSearchCV
        n_candidates = 3
        n_evaluations = n_candidates * n_splits
        expected = root + (search_task + refit) + n_evaluations

    # we expect only the hooks from `search` called:
    for callback in callbacks:
        assert callback.count_hooks("setup") == 1
        assert callback.count_hooks("on_fit_task_begin") == expected
        assert callback.count_hooks("on_fit_task_end") == expected
        assert callback.count_hooks("teardown") == 1


@pytest.mark.parametrize("refit", [True, False])
@pytest.mark.parametrize(
    "search",
    _searchcv_callback_test_cases(MaxIterEstimator, "r2"),
)
@skip_callback_test_if_wasm
def test_search_callbacks_propagation(search, refit):
    # [FUNCTION DOC]
    # Verifies hook counts with a callback-aware inner estimator: plain callback sees
    #   only search tasks; auto-propagated callback additionally sees max_iter tasks per
    #   candidate-split fit and best_estimator_.n_iter_ tasks for refit.
    # KNOWN ISSUE: the else-branch comment names TestingAutoPropagatedCallback but the
    #   class is RecordingAutoPropagatedCallback.
    """Check number of hook calls when the sub-estimator does support callbacks."""
    callbacks = [RecordingCallback(), RecordingAutoPropagatedCallback()]
    search = clone(search).set_params(refit=refit).set_callbacks(*callbacks).fit(X, y)

    # defining expected values
    root = 1
    search_task = 1
    n_splits = search.n_splits_  # 2

    if "Halving" in search.__class__.__name__:
        n_halving_rounds = search.n_iterations_
        n_evaluations = sum(n_cand * n_splits for n_cand in search.n_candidates_)
        searchcv_tasks = root + (search_task + refit) + n_halving_rounds + n_evaluations
    else:  # GridSearchCV, RandomizedSearchCV
        n_candidates = 3
        n_evaluations = n_candidates * n_splits
        searchcv_tasks = root + (search_task + refit) + n_evaluations

    for callback in callbacks:
        assert callback.count_hooks("setup") == 1
        if callback.__class__.__name__ == "RecordingCallback":
            # Without propagation we expect only hook calls from the *SearchCV class.
            assert callback.count_hooks("on_fit_task_begin") == searchcv_tasks
            assert callback.count_hooks("on_fit_task_end") == searchcv_tasks
        else:  # TestingAutoPropagatedCallback
            # With propagation we expect additional calls from each inner estimator.
            # Each MaxIterEstimator has 1 root + max_iter tasks, but we ignore the root
            # because it's the same as the evaluation leaf of the searchcv class.
            # There are n_splits * n_candidates such inner estimators.
            search_inner_tasks = sum(
                p["max_iter"]
                for p in search.cv_results_["params"]
                for _ in range(n_splits)
            )
            refit_inner_tasks = search.best_estimator_.n_iter_ if refit else 0
            expected = searchcv_tasks + search_inner_tasks + refit_inner_tasks
            assert callback.count_hooks("on_fit_task_begin") == expected
            assert callback.count_hooks("on_fit_task_end") == expected
        assert callback.count_hooks("teardown") == 1


@skip_callback_test_if_wasm
def test_search_callbacks_receive_sample_weight():
    # [FUNCTION DOC]
    # Verifies sample_weight appears in the metadata given to hooks for candidate-split
    #   evaluation and refit tasks, and refit receives the full weights unchanged.
    """Test that `sample_weight` gets passed to `callback.on_fit_task_*`.

    Note this tests all *SearchCV classes that inherit from `BaseSearchCV`.
    """
    callback = RecordingCallback()
    search = GridSearchCV(
        MaxIterEstimator(), {"max_iter": [1, 2, 3]}, cv=2, scoring="accuracy"
    ).set_callbacks(callback)
    sample_weight = np.random.RandomState(0).randint(0, 5, size=y.shape[0])
    search.fit(X, y, sample_weight=sample_weight)

    evaluation_records = [
        entry
        for entry in callback.record
        if entry["context"].task_name == "candidate-split-evaluation"
    ]
    assert evaluation_records
    refit_records = [
        entry
        for entry in callback.record
        if entry["context"].task_name == "refit-with-best-params"
    ]
    assert refit_records

    for entry in evaluation_records:
        assert "sample_weight" in entry["kwargs"]["metadata"]

    for entry in refit_records:
        assert "sample_weight" in entry["kwargs"]["metadata"]
        passed_weights = entry["kwargs"]["metadata"]["sample_weight"]
        assert_array_equal(passed_weights, sample_weight)


@pytest.mark.parametrize(
    "search",
    _searchcv_callback_test_cases(MaxIterEstimator, "r2"),
)
@skip_callback_test_if_wasm
def test_search_callbacks_receive_search_instance(search):
    # [FUNCTION DOC]
    # Verifies every recorded hook call receives the search object itself as estimator.
    """Test that all hooks receive the search instance as `estimator` argument."""
    callback = RecordingCallback()
    search = clone(search).set_callbacks(callback).fit(X, y)

    for entry in callback.record:
        assert entry["estimator"] is search


@skip_callback_test_if_wasm
def test_search_callbacks_with_partial_fit_failures():
    # [FUNCTION DOC]
    # Verifies that with numeric error_score and some failing fits, a FitFailedWarning
    #   is emitted and all 6 tasks (root, search, 4 evaluations) still trigger begin/end
    #   hooks.
    """Check callbacks hooks are called when some candidate fits fail.

    When error_score != "raise" and not all fit fail, the whole search doesn't raise so
    all callback invocations are expected.
    """

    class MayFailClassifier(ClassifierMixin, BaseEstimator):
        # [CLASS DOC]
        # HELPER: classifier that raises in fit when fail is True, predicts zeros
        #   otherwise.
        def __init__(self, fail=False):
            # [FUNCTION DOC]
            # HELPER: stores fail (default False).
            self.fail = fail

        def fit(self, X, y):
            # [FUNCTION DOC]
            # HELPER: sets classes_ to unique y, raises RuntimeError if fail, else
            #   returns self.
            self.classes_ = np.unique(y)
            if self.fail:
                raise RuntimeError("MayFailClassifier.fit failed")
            return self

        def predict(self, X):
            # [FUNCTION DOC]
            # HELPER: returns integer zeros of length len(X).
            return np.zeros(len(X), dtype=int)

    callback = RecordingCallback()
    search = GridSearchCV(
        MayFailClassifier(),
        {"fail": [False, True]},
        cv=2,
        refit=False,
        error_score=0.0,
    ).set_callbacks(callback)

    with pytest.warns(FitFailedWarning, match="2 fits failed out of a total of 4."):
        search.fit(X, y)

    expected_n_tasks = 1 + 1 + 4  # root + search + 4 candidate-split evaluations
    assert callback.count_hooks("on_fit_task_begin") == expected_n_tasks
    assert callback.count_hooks("on_fit_task_end") == expected_n_tasks
