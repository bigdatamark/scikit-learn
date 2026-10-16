# [MODULE DOC]
# Tests for sklearn.base: clone semantics, repr/str, get_params/set_params,
# is_classifier/is_regressor/is_clusterer, scoring with sample weights, pickling with
# version warnings, __sklearn_tags__ inheritance, set_output / feature-name
# validation, n_features_in_ checks, HTML repr and related BaseEstimator behavior.
# Defines several small dummy estimator classes used by these tests.
# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import pickle
import re
import warnings

import numpy as np
import pytest
import scipy.sparse as sp
from numpy.testing import assert_allclose

import sklearn
from sklearn import config_context, datasets
from sklearn.base import (
    BaseEstimator,
    OutlierMixin,
    TransformerMixin,
    clone,
    is_classifier,
    is_clusterer,
    is_regressor,
)
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.metrics import get_scorer
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils import deprecated
from sklearn.utils._mocking import MockDataFrame
from sklearn.utils._set_output import _get_output_config
from sklearn.utils._testing import (
    _convert_container,
    assert_array_equal,
)
from sklearn.utils.validation import _check_n_features, validate_data


#############################################################################
# A few test classes
class MyEstimator(BaseEstimator):
    # [CLASS DOC]
    # HELPER: Minimal estimator with two params, l1 and empty, used in clone/repr tests.
    def __init__(self, l1=0, empty=None):
        # [FUNCTION DOC]
        # HELPER: Stores l1 (default 0) and empty (default None) unchanged.
        self.l1 = l1
        self.empty = empty


class K(BaseEstimator):
    # [CLASS DOC]
    # HELPER: Minimal estimator with params c and d, used as a nested sub-estimator.
    def __init__(self, c=None, d=None):
        # [FUNCTION DOC]
        # HELPER: Stores c and d unchanged.
        self.c = c
        self.d = d


class T(BaseEstimator):
    # [CLASS DOC]
    # HELPER: Minimal estimator with params a and b, used to hold nested estimators.
    def __init__(self, a=None, b=None):
        # [FUNCTION DOC]
        # HELPER: Stores a and b unchanged.
        self.a = a
        self.b = b


class NaNTag(BaseEstimator):
    # [CLASS DOC]
    # HELPER: Estimator whose tags set input_tags.allow_nan to True.
    def __sklearn_tags__(self):
        # [FUNCTION DOC]
        # HELPER: Returns parent tags with allow_nan set True.
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        return tags


class NoNaNTag(BaseEstimator):
    # [CLASS DOC]
    # HELPER: Estimator whose tags set input_tags.allow_nan to False.
    def __sklearn_tags__(self):
        # [FUNCTION DOC]
        # HELPER: Returns parent tags with allow_nan set False.
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = False
        return tags


class OverrideTag(NaNTag):
    # [CLASS DOC]
    # HELPER: Subclass of NaNTag overriding allow_nan back to False.
    def __sklearn_tags__(self):
        # [FUNCTION DOC]
        # HELPER: Calls NaNTag's tags then sets allow_nan False.
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = False
        return tags


class DiamondOverwriteTag(NaNTag, NoNaNTag):
    # [CLASS DOC]
    # HELPER: Diamond inheritance (NaNTag, NoNaNTag); per MRO NaNTag runs last so
    # allow_nan ends True.
    pass


class InheritDiamondOverwriteTag(DiamondOverwriteTag):
    # [CLASS DOC]
    # HELPER: Subclass of DiamondOverwriteTag inheriting its tag resolution unchanged.
    pass


class ModifyInitParams(BaseEstimator):
    # [CLASS DOC]
    # HELPER: Estimator that copies its parameter in __init__, violating the clone
    # contract (stored param is not the passed object).
    """Deprecated behavior.
    Equal parameters but with a type cast.
    Doesn't fulfill a is a
    """

    def __init__(self, a=np.array([0])):
        # [FUNCTION DOC]
        # HELPER: Stores a copy of a (default array [0]).
        self.a = a.copy()


class Buggy(BaseEstimator):
    # [CLASS DOC]
    # HELPER: Estimator ignoring its constructor argument (always sets a = 1).
    "A buggy estimator that does not set its parameters right."

    def __init__(self, a=None):
        # [FUNCTION DOC]
        # HELPER: Sets self.a to 1 regardless of the passed a.
        self.a = 1


class NoEstimator:
    # [CLASS DOC]
    # HELPER: Object with fit/predict but no get_params; not a valid estimator.
    def __init__(self):
        # [FUNCTION DOC]
        # HELPER: Does nothing.
        pass

    def fit(self, X=None, y=None):
        # [FUNCTION DOC]
        # HELPER: Returns self.
        return self

    def predict(self, X=None):
        # [FUNCTION DOC]
        # HELPER: Returns None.
        return None


class VargEstimator(BaseEstimator):
    # [CLASS DOC]
    # HELPER: Estimator with *vargs in __init__, which get_params rejects.
    """scikit-learn estimators shouldn't have vargs."""

    def __init__(self, *vargs):
        # [FUNCTION DOC]
        # HELPER: Accepts and ignores positional varargs.
        pass


#############################################################################
# The tests


def test_clone():
    # [FUNCTION DOC]
    # Verifies clone returns a different object with equal get_params for SelectFpr,
    # including when alpha is a numpy array.
    # Tests that clone creates a correct deep copy.
    # We create an estimator, make a copy of its original state
    # (which, in this case, is the current state of the estimator),
    # and check that the obtained copy is a correct deep copy.

    from sklearn.feature_selection import SelectFpr, f_classif

    selector = SelectFpr(f_classif, alpha=0.1)
    new_selector = clone(selector)
    assert selector is not new_selector
    assert selector.get_params() == new_selector.get_params()

    selector = SelectFpr(f_classif, alpha=np.zeros((10, 2)))
    new_selector = clone(selector)
    assert selector is not new_selector


def test_clone_2():
    # [FUNCTION DOC]
    # Verifies clone does not copy non-parameter attributes set on the instance.
    # Tests that clone doesn't copy everything.
    # We first create an estimator, give it an own attribute, and
    # make a copy of its original state. Then we check that the copy doesn't
    # have the specific attribute we manually added to the initial estimator.

    from sklearn.feature_selection import SelectFpr, f_classif

    selector = SelectFpr(f_classif, alpha=0.1)
    selector.own_attribute = "test"
    new_selector = clone(selector)
    assert not hasattr(new_selector, "own_attribute")


def test_clone_buggy():
    # [FUNCTION DOC]
    # Verifies clone raises RuntimeError for Buggy, VargEstimator and ModifyInitParams,
    # and TypeError for an object without get_params (NoEstimator).
    # Check that clone raises an error on buggy estimators.
    buggy = Buggy()
    buggy.a = 2
    with pytest.raises(RuntimeError):
        clone(buggy)

    no_estimator = NoEstimator()
    with pytest.raises(TypeError):
        clone(no_estimator)

    varg_est = VargEstimator()
    with pytest.raises(RuntimeError):
        clone(varg_est)

    est = ModifyInitParams()
    with pytest.raises(RuntimeError):
        clone(est)


def test_clone_empty_array():
    # [FUNCTION DOC]
    # Verifies cloning preserves an empty numpy array and 1x1 sparse csr matrix/array
    # parameters (data compared).
    # Regression test for cloning estimators with empty arrays
    clf = MyEstimator(empty=np.array([]))
    clf2 = clone(clf)
    assert_array_equal(clf.empty, clf2.empty)

    clf = MyEstimator(empty=sp.csr_matrix(np.array([[0]])))
    clf2 = clone(clf)
    assert_array_equal(clf.empty.data, clf2.empty.data)

    clf = MyEstimator(empty=sp.csr_array(np.array([[0]])))
    clf2 = clone(clf)
    assert_array_equal(clf.empty.data, clf2.empty.data)


def test_clone_nan():
    # [FUNCTION DOC]
    # Verifies a np.nan parameter survives clone as the identical object.
    # Regression test for cloning estimators with default parameter as np.nan
    clf = MyEstimator(empty=np.nan)
    clf2 = clone(clf)

    assert clf.empty is clf2.empty


def test_clone_dict():
    # [FUNCTION DOC]
    # Verifies cloning a dict clones the estimators it contains.
    # test that clone creates a clone of a dict
    orig = {"a": MyEstimator()}
    cloned = clone(orig)
    assert orig["a"] is not cloned["a"]


def test_clone_sparse_matrices():
    # [FUNCTION DOC]
    # Verifies that for every scipy.sparse *_matrix / *_array class, a parameter holding
    # such a 5x5 identity is cloned to the same class with equal contents.
    sparse_matrix_classes = [
        cls
        for name in dir(sp)
        if name.endswith("_matrix") or name.endswith("_array")
        if type(cls := getattr(sp, name)) is type
    ]

    for cls in sparse_matrix_classes:
        sparse_matrix = cls(np.eye(5))
        clf = MyEstimator(empty=sparse_matrix)
        clf_cloned = clone(clf)
        assert clf.empty.__class__ is clf_cloned.empty.__class__
        assert_array_equal(clf.empty.toarray(), clf_cloned.empty.toarray())


def test_clone_estimator_types():
    # [FUNCTION DOC]
    # Verifies a class (type) parameter is kept by identity, not cloned.
    # Check that clone works for parameters that are types rather than
    # instances
    clf = MyEstimator(empty=MyEstimator)
    clf2 = clone(clf)

    assert clf.empty is clf2.empty


def test_clone_class_rather_than_instance():
    # [FUNCTION DOC]
    # Verifies cloning a class instead of an instance raises TypeError with a clear message.
    # Check that clone raises expected error message when
    # cloning class rather than instance
    msg = "You should provide an instance of scikit-learn estimator"
    with pytest.raises(TypeError, match=msg):
        clone(MyEstimator)


# TODO (1.11): remove svc test for predict_proba after it is deprecated
def test_conditional_attrs_not_in_dir():
    # [FUNCTION DOC]
    # Verifies __dir__ hides conditionally available methods: set_output absent for
    # LabelEncoder but present for StandardScaler, and predict_proba on SVC follows the
    # probability flag.
    # Test that __dir__ includes only relevant attributes. #28558

    encoder = LabelEncoder()
    assert "set_output" not in dir(encoder)

    scalar = StandardScaler()
    assert "set_output" in dir(scalar)

    svc = SVC(probability=False)
    assert "predict_proba" not in dir(svc)

    svc.probability = True
    assert "predict_proba" in dir(svc)


def test_repr():
    # [FUNCTION DOC]
    # Verifies repr of nested estimators ("T(a=K(), b=K())") and that a very long list
    # parameter is ellipsized to a repr of length 485.
    # Smoke test the repr of the base estimator.
    my_estimator = MyEstimator()
    repr(my_estimator)
    test = T(K(), K())
    assert repr(test) == "T(a=K(), b=K())"

    some_est = T(a=["long_params"] * 1000)
    assert len(repr(some_est)) == 485


def test_str():
    # [FUNCTION DOC]
    # Smoke test that str() works on an estimator.
    # Smoke test the str of the base estimator
    my_estimator = MyEstimator()
    str(my_estimator)


def test_get_params():
    # [FUNCTION DOC]
    # Verifies deep get_params exposes nested "a__d" while shallow does not, that
    # set_params(a__d=2) sets the nested value, and invalid nested param raises ValueError.
    test = T(K(), K)

    assert "a__d" in test.get_params(deep=True)
    assert "a__d" not in test.get_params(deep=False)

    test.set_params(a__d=2)
    assert test.a.d == 2

    with pytest.raises(ValueError):
        test.set_params(a__a=2)


@pytest.mark.parametrize(
    "estimator, expected_result",
    [
        (SVC(), True),
        (GridSearchCV(SVC(), {"C": [0.1, 1]}), True),
        (Pipeline([("svc", SVC())]), True),
        (Pipeline([("svc_cv", GridSearchCV(SVC(), {"C": [0.1, 1]}))]), True),
        (SVR(), False),
        (GridSearchCV(SVR(), {"C": [0.1, 1]}), False),
        (Pipeline([("svr", SVR())]), False),
        (Pipeline([("svr_cv", GridSearchCV(SVR(), {"C": [0.1, 1]}))]), False),
    ],
)
def test_is_classifier(estimator, expected_result):
    # [FUNCTION DOC]
    # Verifies is_classifier on plain, grid-searched and pipelined SVC/SVR.
    # Inputs: estimator, expected_result - parametrized.
    assert is_classifier(estimator) == expected_result


@pytest.mark.parametrize(
    "estimator, expected_result",
    [
        (SVR(), True),
        (GridSearchCV(SVR(), {"C": [0.1, 1]}), True),
        (Pipeline([("svr", SVR())]), True),
        (Pipeline([("svr_cv", GridSearchCV(SVR(), {"C": [0.1, 1]}))]), True),
        (SVC(), False),
        (GridSearchCV(SVC(), {"C": [0.1, 1]}), False),
        (Pipeline([("svc", SVC())]), False),
        (Pipeline([("svc_cv", GridSearchCV(SVC(), {"C": [0.1, 1]}))]), False),
    ],
)
def test_is_regressor(estimator, expected_result):
    # [FUNCTION DOC]
    # Verifies is_regressor on plain, grid-searched and pipelined SVR/SVC.
    assert is_regressor(estimator) == expected_result


@pytest.mark.parametrize(
    "estimator, expected_result",
    [
        (KMeans(), True),
        (GridSearchCV(KMeans(), {"n_clusters": [3, 8]}), True),
        (Pipeline([("km", KMeans())]), True),
        (Pipeline([("km_cv", GridSearchCV(KMeans(), {"n_clusters": [3, 8]}))]), True),
        (SVC(), False),
        (GridSearchCV(SVC(), {"C": [0.1, 1]}), False),
        (Pipeline([("svc", SVC())]), False),
        (Pipeline([("svc_cv", GridSearchCV(SVC(), {"C": [0.1, 1]}))]), False),
    ],
)
def test_is_clusterer(estimator, expected_result):
    # [FUNCTION DOC]
    # Verifies is_clusterer on plain, grid-searched and pipelined KMeans/SVC.
    assert is_clusterer(estimator) == expected_result


def test_set_params():
    # [FUNCTION DOC]
    # Verifies Pipeline.set_params raises ValueError for unknown nested parameter names
    # and unknown step names.
    # test nested estimator parameter setting
    clf = Pipeline([("svc", SVC())])

    # non-existing parameter in svc
    with pytest.raises(ValueError):
        clf.set_params(svc__stupid_param=True)

    # non-existing parameter of pipeline
    with pytest.raises(ValueError):
        clf.set_params(svm__stupid_param=True)

    # we don't currently catch if the things in pipeline are estimators
    # bad_pipeline = Pipeline([("bad", NoEstimator())])
    # with pytest.raises(AttributeError):
    #    bad_pipeline.set_params(bad__stupid_param=True)


def test_set_params_passes_all_parameters():
    # [FUNCTION DOC]
    # Regression test (#9944): nested set_params on Pipeline and GridSearchCV passes all
    # nested params in a single call to the sub-estimator; asserted inside the override.
    # Make sure all parameters are passed together to set_params
    # of nested estimator. Regression test for #9944

    class TestDecisionTree(DecisionTreeClassifier):
        # [CLASS DOC]
        # HELPER: DecisionTreeClassifier whose set_params asserts it received exactly the
        # expected kwargs (closure variable expected_kwargs).
        def set_params(self, **kwargs):
            # [FUNCTION DOC]
            # HELPER: Delegates to parent set_params, asserts kwargs equal expected_kwargs,
            # returns self. Raises AssertionError otherwise.
            super().set_params(**kwargs)
            # expected_kwargs is in test scope
            assert kwargs == expected_kwargs
            return self

    expected_kwargs = {"max_depth": 5, "min_samples_leaf": 2}
    for est in [
        Pipeline([("estimator", TestDecisionTree())]),
        GridSearchCV(TestDecisionTree(), {}),
    ]:
        est.set_params(estimator__max_depth=5, estimator__min_samples_leaf=2)


def test_set_params_updates_valid_params():
    # [FUNCTION DOC]
    # Verifies set_params replacing estimator and setting estimator__C in one call applies
    # C to the new SVC rather than the old tree.
    # Check that set_params tries to set SVC().C, not
    # DecisionTreeClassifier().C
    gscv = GridSearchCV(DecisionTreeClassifier(), {})
    gscv.set_params(estimator=SVC(), estimator__C=42.0)
    assert gscv.estimator.C == 42.0


@pytest.mark.parametrize(
    "tree,dataset",
    [
        (
            DecisionTreeClassifier(max_depth=2, random_state=0),
            datasets.make_classification(random_state=0),
        ),
        (
            DecisionTreeRegressor(max_depth=2, random_state=0),
            datasets.make_regression(random_state=0),
        ),
    ],
)
def test_score_sample_weight(tree, dataset):
    # [FUNCTION DOC]
    # Verifies tree.score differs with random integer sample_weight vs. unweighted, for a
    # classifier and a regressor.
    tree = clone(tree)  # avoid side effects from previous tests.
    rng = np.random.RandomState(0)
    # check that the score with and without sample weights are different
    X, y = dataset

    tree.fit(X, y)
    # generate random sample weights
    sample_weight = rng.randint(1, 10, size=len(y))
    score_unweighted = tree.score(X, y)
    score_weighted = tree.score(X, y, sample_weight=sample_weight)
    msg = "Unweighted and weighted scores are unexpectedly equal"
    assert score_unweighted != score_weighted, msg


def test_clone_pandas_dataframe():
    # [FUNCTION DOC]
    # Verifies cloning an estimator holding a MockDataFrame parameter keeps an equal
    # frame and scalar param.
    class DummyEstimator(TransformerMixin, BaseEstimator):
        # [CLASS DOC]
        # HELPER: Transformer with a dataframe param and a scalar param; fit/transform no-ops.
        """This is a dummy class for generating numerical features

        This feature extractor extracts numerical features from pandas data
        frame.

        Parameters
        ----------

        df: pandas data frame
            The pandas data frame parameter.

        Notes
        -----
        """

        def __init__(self, df=None, scalar_param=1):
            # [FUNCTION DOC]
            # HELPER: Stores df and scalar_param.
            self.df = df
            self.scalar_param = scalar_param

        def fit(self, X, y=None):
            # [FUNCTION DOC]
            # HELPER: No-op returning None.
            pass

        def transform(self, X):
            # [FUNCTION DOC]
            # HELPER: No-op returning None.
            pass

    # build and clone estimator
    d = np.arange(10)
    df = MockDataFrame(d)
    e = DummyEstimator(df, scalar_param=1)
    cloned_e = clone(e)

    # the test
    assert (e.df == cloned_e.df).values.all()
    assert e.scalar_param == cloned_e.scalar_param


def test_clone_protocol():
    # [FUNCTION DOC]
    # Verifies the __sklearn_clone__ protocol using a frozen wrapper: attribute access
    # delegates to the fitted PCA, fit/fit_transform leave components_ unchanged, and
    # clone returns the same object.
    """Checks that clone works with `__sklearn_clone__` protocol."""

    class FrozenEstimator(BaseEstimator):
        # [CLASS DOC]
        # HELPER: Wrapper around a fitted estimator that ignores refitting and clones to itself.
        def __init__(self, fitted_estimator):
            # [FUNCTION DOC]
            # HELPER: Stores the fitted estimator.
            self.fitted_estimator = fitted_estimator

        def __getattr__(self, name):
            # [FUNCTION DOC]
            # HELPER: Delegates missing attribute lookups to fitted_estimator.
            return getattr(self.fitted_estimator, name)

        def __sklearn_clone__(self):
            # [FUNCTION DOC]
            # HELPER: Returns self, making clone a no-op.
            return self

        def fit(self, *args, **kwargs):
            # [FUNCTION DOC]
            # HELPER: Ignores arguments; returns self.
            return self

        def fit_transform(self, *args, **kwargs):
            # [FUNCTION DOC]
            # HELPER: Returns fitted_estimator.transform of the arguments without refitting.
            return self.fitted_estimator.transform(*args, **kwargs)

    X = np.array([[-1, -1], [-2, -1], [-3, -2]])
    pca = PCA().fit(X)
    components = pca.components_

    frozen_pca = FrozenEstimator(pca)
    assert_allclose(frozen_pca.components_, components)

    # Calling PCA methods such as `get_feature_names_out` still works
    assert_array_equal(frozen_pca.get_feature_names_out(), pca.get_feature_names_out())

    # Fitting on a new data does not alter `components_`
    X_new = np.asarray([[-1, 2], [3, 4], [1, 2]])
    frozen_pca.fit(X_new)
    assert_allclose(frozen_pca.components_, components)

    # `fit_transform` does not alter state
    frozen_pca.fit_transform(X_new)
    assert_allclose(frozen_pca.components_, components)

    # Cloning estimator is a no-op
    clone_frozen_pca = clone(frozen_pca)
    assert clone_frozen_pca is frozen_pca
    assert_allclose(clone_frozen_pca.components_, components)


def test_pickle_version_warning_is_not_raised_with_matching_version():
    # [FUNCTION DOC]
    # Verifies pickles contain _sklearn_version and that unpickling with matching version
    # emits no warning (warnings as errors) and gives the same score.
    iris = datasets.load_iris()
    tree = DecisionTreeClassifier().fit(iris.data, iris.target)
    tree_pickle = pickle.dumps(tree)
    assert b"_sklearn_version" in tree_pickle

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        tree_restored = pickle.loads(tree_pickle)

    # test that we can predict with the restored decision tree classifier
    score_of_original = tree.score(iris.data, iris.target)
    score_of_restored = tree_restored.score(iris.data, iris.target)
    assert score_of_original == score_of_restored


class TreeBadVersion(DecisionTreeClassifier):
    # [CLASS DOC]
    # HELPER: Tree whose pickle state reports _sklearn_version "something".
    def __getstate__(self):
        # [FUNCTION DOC]
        # HELPER: Returns a copy of __dict__ with _sklearn_version overridden.
        return dict(self.__dict__.items(), _sklearn_version="something")


pickle_error_message = (
    "Trying to unpickle estimator {estimator} from "
    "version {old_version} when using version "
    "{current_version}. This might "
    "lead to breaking code or invalid results. "
    "Use at your own risk."
)


def test_pickle_version_warning_is_issued_upon_different_version():
    # [FUNCTION DOC]
    # Verifies unpickling with a different version emits InconsistentVersionWarning with
    # correct estimator name, original and current version attributes.
    iris = datasets.load_iris()
    tree = TreeBadVersion().fit(iris.data, iris.target)
    tree_pickle_other = pickle.dumps(tree)
    message = pickle_error_message.format(
        estimator="TreeBadVersion",
        old_version="something",
        current_version=sklearn.__version__,
    )
    with pytest.warns(UserWarning, match=message) as warning_record:
        pickle.loads(tree_pickle_other)

    message = warning_record.list[0].message
    assert isinstance(message, InconsistentVersionWarning)
    assert message.estimator_name == "TreeBadVersion"
    assert message.original_sklearn_version == "something"
    assert message.current_sklearn_version == sklearn.__version__


class TreeNoVersion(DecisionTreeClassifier):
    # [CLASS DOC]
    # HELPER: Tree whose pickle state omits _sklearn_version (pre-0.18 style).
    def __getstate__(self):
        # [FUNCTION DOC]
        # HELPER: Returns __dict__ directly (no version key).
        return self.__dict__


def test_pickle_version_warning_is_issued_when_no_version_info_in_pickle():
    # [FUNCTION DOC]
    # Verifies unpickling a version-less pickle warns with old_version "pre-0.18".
    iris = datasets.load_iris()
    # TreeNoVersion has no getstate, like pre-0.18
    tree = TreeNoVersion().fit(iris.data, iris.target)

    tree_pickle_noversion = pickle.dumps(tree)
    assert b"_sklearn_version" not in tree_pickle_noversion
    message = pickle_error_message.format(
        estimator="TreeNoVersion",
        old_version="pre-0.18",
        current_version=sklearn.__version__,
    )
    # check we got the warning about using pre-0.18 pickle
    with pytest.warns(UserWarning, match=message):
        pickle.loads(tree_pickle_noversion)


# The test modifies global state by changing the TreeNoVersion class
@pytest.mark.thread_unsafe
def test_pickle_version_no_warning_is_issued_with_non_sklearn_estimator():
    # [FUNCTION DOC]
    # Verifies no warning when the unpickled class's module is not sklearn, by temporarily
    # setting TreeNoVersion.__module__ to "notsklearn" (restored in finally).
    # Side effects: temporarily mutates a module-level class (thread_unsafe).
    iris = datasets.load_iris()
    tree = TreeNoVersion().fit(iris.data, iris.target)
    tree_pickle_noversion = pickle.dumps(tree)
    try:
        module_backup = TreeNoVersion.__module__
        TreeNoVersion.__module__ = "notsklearn"

        with warnings.catch_warnings():
            warnings.simplefilter("error")

            pickle.loads(tree_pickle_noversion)
    finally:
        TreeNoVersion.__module__ = module_backup


class DontPickleAttributeMixin:
    # [CLASS DOC]
    # HELPER: Mixin whose pickling sets _attribute_not_pickled to None and whose unpickling
    # sets _restored True.
    def __getstate__(self):
        # [FUNCTION DOC]
        # HELPER: Returns a copy of __dict__ with _attribute_not_pickled set to None.
        data = self.__dict__.copy()
        data["_attribute_not_pickled"] = None
        return data

    def __setstate__(self, state):
        # [FUNCTION DOC]
        # HELPER: Sets state["_restored"] = True (mutating state) and updates __dict__.
        state["_restored"] = True
        self.__dict__.update(state)


class MultiInheritanceEstimator(DontPickleAttributeMixin, BaseEstimator):
    # [CLASS DOC]
    # HELPER: Estimator combining DontPickleAttributeMixin with BaseEstimator, to test that
    # a mixin's __getstate__/__setstate__ take precedence.
    def __init__(self, attribute_pickled=5):
        # [FUNCTION DOC]
        # HELPER: Stores attribute_pickled (default 5) and sets _attribute_not_pickled to None.
        self.attribute_pickled = attribute_pickled
        self._attribute_not_pickled = None


def test_pickling_when_getstate_is_overwritten_by_mixin():
    # [FUNCTION DOC]
    # Verifies pickle round-trip uses the mixin's getstate/setstate: the public param is
    # kept, _attribute_not_pickled becomes None and _restored is set.
    estimator = MultiInheritanceEstimator()
    estimator._attribute_not_pickled = "this attribute should not be pickled"

    serialized = pickle.dumps(estimator)
    estimator_restored = pickle.loads(serialized)
    assert estimator_restored.attribute_pickled == 5
    assert estimator_restored._attribute_not_pickled is None
    assert estimator_restored._restored


def test_pickling_when_getstate_is_overwritten_by_mixin_outside_of_sklearn():
    # [FUNCTION DOC]
    # Verifies, with the class's __module__ temporarily set to "notsklearn", that the mixin
    # __getstate__ returns no _sklearn_version and __setstate__ restores given values.
    # Side effects: mutates MultiInheritanceEstimator.__module__, restored in finally.
    # KNOWN ISSUE: not marked thread_unsafe though it mutates a module-level class, and if
    # construction failed, old_mod would be undefined in the finally clause.
    try:
        estimator = MultiInheritanceEstimator()
        text = "this attribute should not be pickled"
        estimator._attribute_not_pickled = text
        old_mod = type(estimator).__module__
        type(estimator).__module__ = "notsklearn"

        serialized = estimator.__getstate__()
        assert serialized == {"_attribute_not_pickled": None, "attribute_pickled": 5}

        serialized["attribute_pickled"] = 4
        estimator.__setstate__(serialized)
        assert estimator.attribute_pickled == 4
        assert estimator._restored
    finally:
        type(estimator).__module__ = old_mod


class SingleInheritanceEstimator(BaseEstimator):
    # [CLASS DOC]
    # HELPER: Estimator overriding __getstate__ to blank _attribute_not_pickled.
    def __init__(self, attribute_pickled=5):
        # [FUNCTION DOC]
        # HELPER: Stores attribute_pickled and sets _attribute_not_pickled to None.
        self.attribute_pickled = attribute_pickled
        self._attribute_not_pickled = None

    def __getstate__(self):
        # [FUNCTION DOC]
        # HELPER: Returns BaseEstimator's state with _attribute_not_pickled set to None.
        state = super().__getstate__()
        state["_attribute_not_pickled"] = None
        return state


def test_pickling_works_when_getstate_is_overwritten_in_the_child_class():
    # [FUNCTION DOC]
    # Verifies a child-class __getstate__ override is honored during pickle round-trip.
    estimator = SingleInheritanceEstimator()
    estimator._attribute_not_pickled = "this attribute should not be pickled"

    serialized = pickle.dumps(estimator)
    estimator_restored = pickle.loads(serialized)
    assert estimator_restored.attribute_pickled == 5
    assert estimator_restored._attribute_not_pickled is None


def test_tag_inheritance():
    # [FUNCTION DOC]
    # Verifies allow_nan resolution for NaNTag, NoNaNTag, OverrideTag and the diamond
    # classes (diamond resolves to True).
    # KNOWN ISSUE: the comment says changing tags by inheritance "is not allowed", but the
    # assertions show overriding via inheritance works as normal MRO.
    # test that changing tags by inheritance is not allowed

    nan_tag_est = NaNTag()
    no_nan_tag_est = NoNaNTag()
    assert nan_tag_est.__sklearn_tags__().input_tags.allow_nan
    assert not no_nan_tag_est.__sklearn_tags__().input_tags.allow_nan

    redefine_tags_est = OverrideTag()
    assert not redefine_tags_est.__sklearn_tags__().input_tags.allow_nan

    diamond_tag_est = DiamondOverwriteTag()
    assert diamond_tag_est.__sklearn_tags__().input_tags.allow_nan

    inherit_diamond_tag_est = InheritDiamondOverwriteTag()
    assert inherit_diamond_tag_est.__sklearn_tags__().input_tags.allow_nan


def test_raises_on_get_params_non_attribute():
    # [FUNCTION DOC]
    # Verifies get_params raises AttributeError when __init__ does not store a declared
    # parameter as an attribute.
    class MyEstimator(BaseEstimator):
        # [CLASS DOC]
        # HELPER: Estimator declaring param in __init__ but never storing it.
        def __init__(self, param=5):
            # [FUNCTION DOC]
            # HELPER: Ignores param.
            pass

        def fit(self, X, y=None):
            # [FUNCTION DOC]
            # HELPER: Returns self.
            return self

    est = MyEstimator()
    msg = "'MyEstimator' object has no attribute 'param'"

    with pytest.raises(AttributeError, match=msg):
        est.get_params()


def test_repr_mimebundle_():
    # [FUNCTION DOC]
    # Verifies _repr_mimebundle_ includes text/html by default and only text/plain under
    # display="text".
    # Checks the display configuration flag controls the json output
    tree = DecisionTreeClassifier()
    output = tree._repr_mimebundle_()
    assert "text/plain" in output
    assert "text/html" in output

    with config_context(display="text"):
        output = tree._repr_mimebundle_()
        assert "text/plain" in output
        assert "text/html" not in output


def test_repr_html_wraps():
    # [FUNCTION DOC]
    # Verifies _repr_html_ contains a style tag and raises AttributeError under
    # display="text".
    # Checks the display configuration flag controls the html output
    tree = DecisionTreeClassifier()

    output = tree._repr_html_()
    assert "<style>" in output

    with config_context(display="text"):
        msg = "_repr_html_ is only defined when"
        with pytest.raises(AttributeError, match=msg):
            output = tree._repr_html_()


def test_n_features_in_validation():
    # [FUNCTION DOC]
    # Verifies _check_n_features sets n_features_in_ on reset and raises ValueError for
    # featureless input when reset=False.
    """Check that `_check_n_features` validates data when reset=False"""
    est = MyEstimator()
    X_train = [[1, 2, 3], [4, 5, 6]]
    _check_n_features(est, X_train, reset=True)

    assert est.n_features_in_ == 3

    msg = "X does not contain any features, but MyEstimator is expecting 3 features"
    with pytest.raises(ValueError, match=msg):
        _check_n_features(est, "invalid X", reset=False)


def test_n_features_in_no_validation():
    # [FUNCTION DOC]
    # Verifies _check_n_features neither sets n_features_in_ nor raises for invalid input
    # when the attribute was never defined.
    """Check that `_check_n_features` does not validate data when
    n_features_in_ is not defined."""
    est = MyEstimator()
    _check_n_features(est, "invalid X", reset=True)

    assert not hasattr(est, "n_features_in_")

    # does not raise
    _check_n_features(est, "invalid X", reset=False)


def test_feature_names_in():
    # [FUNCTION DOC]
    # Verifies feature_names_in_ handling by validate_data with pandas: stored on fit with
    # DataFrame, removed on ndarray refit, mismatch raises ValueError, ndarray/DataFrame
    # mixing warns, integer column names are silently ignored, and mixed str/int column
    # names raise TypeError on fit and transform. Skipped without pandas.
    """Check that feature_name_in are recorded by `_validate_data`"""
    pd = pytest.importorskip("pandas")
    iris = datasets.load_iris()
    X_np = iris.data
    df = pd.DataFrame(X_np, columns=iris.feature_names)

    class NoOpTransformer(TransformerMixin, BaseEstimator):
        # [CLASS DOC]
        # HELPER: Transformer that only runs validate_data.
        def fit(self, X, y=None):
            # [FUNCTION DOC]
            # HELPER: validate_data with reset; returns self.
            validate_data(self, X)
            return self

        def transform(self, X):
            # [FUNCTION DOC]
            # HELPER: validate_data with reset=False; returns X unchanged.
            validate_data(self, X, reset=False)
            return X

    # fit on dataframe saves the feature names
    trans = NoOpTransformer().fit(df)
    assert_array_equal(trans.feature_names_in_, df.columns)

    # fit again but on ndarray does not keep the previous feature names (see #21383)
    trans.fit(X_np)
    assert not hasattr(trans, "feature_names_in_")

    trans.fit(df)
    msg = "The feature names should match those that were passed"
    df_bad = pd.DataFrame(X_np, columns=iris.feature_names[::-1])
    with pytest.raises(ValueError, match=msg):
        trans.transform(df_bad)

    # warns when fitted on dataframe and transforming an ndarray
    msg = (
        "X does not have valid feature names, but NoOpTransformer was "
        "fitted with feature names"
    )
    with pytest.warns(UserWarning, match=msg):
        trans.transform(X_np)

    # warns when fitted on an ndarray and transforming dataframe
    msg = "X has feature names, but NoOpTransformer was fitted without feature names"
    trans = NoOpTransformer().fit(X_np)
    with pytest.warns(UserWarning, match=msg):
        trans.transform(df)

    # fit on dataframe with all integer feature names works without warning
    df_int_names = pd.DataFrame(X_np)
    trans = NoOpTransformer()
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        trans.fit(df_int_names)

    # fit on dataframe with no feature names or all integer feature names
    # -> do not warn on transform
    Xs = [X_np, df_int_names]
    for X in Xs:
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            trans.transform(X)

    # fit on dataframe with feature names that are mixed raises an error:
    df_mixed = pd.DataFrame(X_np, columns=["a", "b", 1, 2])
    trans = NoOpTransformer()
    msg = re.escape(
        "Feature names are only supported if all input features have string names, "
        "but your input has ['int', 'str'] as feature name / column name types. "
        "If you want feature names to be stored and validated, you must convert "
        "them all to strings, by using X.columns = X.columns.astype(str) for "
        "example. Otherwise you can remove feature / column names from your input "
        "data, or convert them all to a non-string data type."
    )
    with pytest.raises(TypeError, match=msg):
        trans.fit(df_mixed)

    # transform on feature names that are mixed also raises:
    with pytest.raises(TypeError, match=msg):
        trans.transform(df_mixed)


def test_validate_data_skip_check_array():
    # [FUNCTION DOC]
    # Verifies validate_data skip_check_array: False converts X/y to ndarrays, True returns
    # the original objects, for X only, y only and both; calling with neither raises
    # ValueError.
    """Check skip_check_array option of _validate_data."""

    pd = pytest.importorskip("pandas")
    iris = datasets.load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target)

    class NoOpTransformer(TransformerMixin, BaseEstimator):
        # [CLASS DOC]
        # HELPER: Empty transformer used as validate_data target.
        pass

    no_op = NoOpTransformer()
    X_np_out = validate_data(no_op, df, skip_check_array=False)
    assert isinstance(X_np_out, np.ndarray)
    assert_allclose(X_np_out, df.to_numpy())

    X_df_out = validate_data(no_op, df, skip_check_array=True)
    assert X_df_out is df

    y_np_out = validate_data(no_op, y=y, skip_check_array=False)
    assert isinstance(y_np_out, np.ndarray)
    assert_allclose(y_np_out, y.to_numpy())

    y_series_out = validate_data(no_op, y=y, skip_check_array=True)
    assert y_series_out is y

    X_np_out, y_np_out = validate_data(no_op, df, y, skip_check_array=False)
    assert isinstance(X_np_out, np.ndarray)
    assert_allclose(X_np_out, df.to_numpy())
    assert isinstance(y_np_out, np.ndarray)
    assert_allclose(y_np_out, y.to_numpy())

    X_df_out, y_series_out = validate_data(no_op, df, y, skip_check_array=True)
    assert X_df_out is df
    assert y_series_out is y

    msg = "Validation should be done on X, y or both."
    with pytest.raises(ValueError, match=msg):
        validate_data(no_op)


def test_clone_keeps_output_config():
    # [FUNCTION DOC]
    # Verifies clone preserves the set_output transform config.
    """Check that clone keeps the set_output config."""

    ss = StandardScaler().set_output(transform="pandas")
    config = _get_output_config("transform", ss)

    ss_clone = clone(ss)
    config_clone = _get_output_config("transform", ss_clone)
    assert config == config_clone


class _Empty:
    # [CLASS DOC]
    # HELPER: Plain empty class used as a base.
    pass


class EmptyEstimator(_Empty, BaseEstimator):
    # [CLASS DOC]
    # HELPER: Estimator with no attributes, combining _Empty and BaseEstimator.
    pass


@pytest.mark.parametrize("estimator", [BaseEstimator(), EmptyEstimator()])
def test_estimator_empty_instance_dict(estimator):
    # [FUNCTION DOC]
    # Verifies __getstate__ on an attribute-less estimator returns only _sklearn_version
    # and that pickling BaseEstimator() works (gh-25188).
    """Check that ``__getstate__`` returns an empty ``dict`` with an empty
    instance.

    Python 3.11+ changed behaviour by returning ``None`` instead of raising an
    ``AttributeError``. Non-regression test for gh-25188.
    """
    state = estimator.__getstate__()
    expected = {"_sklearn_version": sklearn.__version__}
    assert state == expected

    # this should not raise
    pickle.loads(pickle.dumps(BaseEstimator()))


def test_estimator_getstate_using_slots_error_message():
    # [FUNCTION DOC]
    # Verifies __getstate__ and pickle.dumps raise TypeError for estimators using __slots__.
    """Using a `BaseEstimator` with `__slots__` is not supported."""

    class WithSlots:
        # [CLASS DOC]
        # HELPER: Class with __slots__ ("x",).
        __slots__ = ("x",)

    class Estimator(BaseEstimator, WithSlots):
        # [CLASS DOC]
        # HELPER: BaseEstimator subclass mixed with a slotted class.
        pass

    msg = (
        "You cannot use `__slots__` in objects inheriting from "
        "`sklearn.base.BaseEstimator`"
    )

    with pytest.raises(TypeError, match=msg):
        Estimator().__getstate__()

    with pytest.raises(TypeError, match=msg):
        pickle.dumps(Estimator())


@pytest.mark.parametrize(
    "constructor_name, minversion",
    [
        ("pandas", "1.5.0"),
        ("pyarrow", "13.0.0"),
        ("polars", "0.20.23"),
    ],
)
def test_feature_names_in_on_dataframes(constructor_name, minversion):
    # [FUNCTION DOC]
    # Verifies feature_names_in_ is set from pandas, pyarrow and polars frames, transform
    # returns equal values (not checked for pyarrow), and mismatched names raise ValueError.
    # Inputs: constructor_name, minversion - parametrized container library and minimum
    # version.
    """Test that feature_names_in_ is correctly set for dataframe X."""
    data = [[1, 4, 2], [3, 3, 6]]
    columns = ["col_0", "col_1", "col_2"]
    df = _convert_container(
        data, constructor_name, column_names=columns, minversion=minversion
    )

    class NoOpTransformer(TransformerMixin, BaseEstimator):
        # [CLASS DOC]
        # HELPER: Transformer validating input via validate_data.
        def fit(self, X, y=None):
            # [FUNCTION DOC]
            # HELPER: validate_data with reset; returns self.
            validate_data(self, X)
            return self

        def transform(self, X):
            # [FUNCTION DOC]
            # HELPER: Returns the validated array (reset=False).
            return validate_data(self, X, reset=False)

    no_op = NoOpTransformer()
    no_op.fit(df)
    assert_array_equal(no_op.feature_names_in_, columns)
    X_out = no_op.transform(df)

    if constructor_name != "pyarrow":
        # pyarrow does not work with `np.asarray`
        # https://github.com/apache/arrow/issues/34886
        assert_allclose(df, X_out)

    bad_names = ["a", "b", "c"]
    df_bad = _convert_container(data, constructor_name, column_names=bad_names)
    with pytest.raises(ValueError, match="The feature names should match"):
        no_op.transform(df_bad)


@config_context(enable_metadata_routing=True)
def test_transformer_fit_transform_with_metadata_in_transform():
    # [FUNCTION DOC]
    # Verifies with routing enabled that TransformerMixin.fit_transform warns when metadata
    # is passed and transform requests it, and does not warn when none is passed.
    """Test that having a transformer with metadata for transform raises a
    warning when calling fit_transform."""

    class CustomTransformer(BaseEstimator, TransformerMixin):
        # [CLASS DOC]
        # HELPER: Transformer accepting prop in fit and transform.
        def fit(self, X, y=None, prop=None):
            # [FUNCTION DOC]
            # HELPER: Returns self.
            return self

        def transform(self, X, prop=None):
            # [FUNCTION DOC]
            # HELPER: Returns X.
            return X

    # passing the metadata to `fit_transform` should raise a warning since it
    # could potentially be consumed by `transform`
    with pytest.warns(UserWarning, match="`transform` method which consumes metadata"):
        CustomTransformer().set_transform_request(prop=True).fit_transform(
            [[1]], [1], prop=1
        )

    # not passing a metadata which can potentially be consumed by `transform` should
    # not raise a warning
    with warnings.catch_warnings(record=True) as record:
        CustomTransformer().set_transform_request(prop=True).fit_transform([[1]], [1])
        assert len(record) == 0


@config_context(enable_metadata_routing=True)
def test_outlier_mixin_fit_predict_with_metadata_in_predict():
    # [FUNCTION DOC]
    # Verifies with routing enabled that OutlierMixin.fit_predict warns when metadata is
    # passed and predict requests it, and does not warn otherwise.
    """Test that having an OutlierMixin with metadata for predict raises a
    warning when calling fit_predict."""

    class CustomOutlierDetector(BaseEstimator, OutlierMixin):
        # [CLASS DOC]
        # HELPER: Outlier detector accepting prop in fit and predict.
        def fit(self, X, y=None, prop=None):
            # [FUNCTION DOC]
            # HELPER: Returns self.
            return self

        def predict(self, X, prop=None):
            # [FUNCTION DOC]
            # HELPER: Returns X.
            return X

    # passing the metadata to `fit_predict` should raise a warning since it
    # could potentially be consumed by `predict`
    with pytest.warns(UserWarning, match="`predict` method which consumes metadata"):
        CustomOutlierDetector().set_predict_request(prop=True).fit_predict(
            [[1]], [1], prop=1
        )

    # not passing a metadata which can potentially be consumed by `predict` should
    # not raise a warning
    with warnings.catch_warnings(record=True) as record:
        CustomOutlierDetector().set_predict_request(prop=True).fit_predict([[1]], [1])
        assert len(record) == 0


def test_get_params_html():
    # [FUNCTION DOC]
    # Verifies _get_params_html returns all params and lists only "empty" as non-default.
    """Check the behaviour of the `_get_params_html` method."""
    est = MyEstimator(empty="test")

    assert est._get_params_html() == {"l1": 0, "empty": "test"}
    assert est._get_params_html().non_default == ("empty",)


def test_get_fitted_attr_html():
    # [FUNCTION DOC]
    # Verifies _get_fitted_attr_html for a fitted PCA: type/value/shape entries, 9 fitted
    # attributes, and non-trailing-underscore attributes excluded.
    """Check the behaviour of the `_get_fitted_attr_html` method."""
    X = np.array([[-1, -1], [-2, -1], [-3, -2]])
    pca = PCA().fit(X)
    pca._not_a_fitted_attr = "x"

    fitted_attr_html = pca._get_fitted_attr_html()
    assert fitted_attr_html["n_features_in_"] == {"type_name": "int", "value": 2}
    assert pca._not_a_fitted_attr not in fitted_attr_html
    assert len(fitted_attr_html) == 9
    assert fitted_attr_html["components_"]["type_name"] == ("ndarray")
    assert fitted_attr_html["components_"]["shape"] == (2, 2)
    assert_allclose(fitted_attr_html["components_"]["value"], pca.components_)


def test_get_fitted_attr_html_no_warnings():
    # [FUNCTION DOC]
    # Verifies building the fitted-attribute HTML for a deprecated fitted property emits no
    # warnings and still reports its value (issue #34056).
    """Check that deprecated fitted attrs don't generate a warning in the repr.

    Non-regression test for
    https://github.com/scikit-learn/scikit-learn/issues/34056
    """

    class Estimator(BaseEstimator):
        # [CLASS DOC]
        # HELPER: Estimator with a deprecated fitted property foo_.
        @deprecated("Attribute `foo` is deprecated")
        @property
        def foo_(self):
            # [FUNCTION DOC]
            # HELPER: Returns 1 (access emits a deprecation warning).
            return 1

    model = Estimator()
    with warnings.catch_warnings():
        warnings.simplefilter("error")

        html = model._get_fitted_attr_html()

    assert html["foo_"] == {"type_name": "int", "value": 1}


def make_estimator_with_param(default_value):
    # [FUNCTION DOC]
    # HELPER: Builds and returns a new estimator class whose single param defaults to
    # default_value.
    class DynamicEstimator(BaseEstimator):
        # [CLASS DOC]
        # HELPER: Dynamically created estimator with one param.
        def __init__(self, param=default_value):
            # [FUNCTION DOC]
            # HELPER: Stores param.
            self.param = param

    return DynamicEstimator


@pytest.mark.parametrize(
    "default_value, test_value",
    [
        ((), (1,)),
        ((), [1]),
        ((), np.array([1])),
        ((1, 2), (3, 4)),
        ((1, 2), [3, 4]),
        ((1, 2), np.array([3, 4])),
        (None, 1),
        (None, []),
        (None, lambda x: x),
        (np.nan, 1.0),
        (np.nan, np.array([np.nan])),
        ("abc", "def"),
        ("abc", ["abc"]),
        (True, False),
        (1, 2),
        (1, [1]),
        (1, np.array([1])),
        (1.0, 2.0),
        (1.0, [1.0]),
        (1.0, np.array([1.0])),
        ([1, 2], [3]),
        (np.array([1]), [2, 3]),
        (None, KFold()),
        (None, get_scorer("accuracy")),
    ],
)
def test_param_is_non_default(default_value, test_value):
    # [FUNCTION DOC]
    # Verifies values differing from the default (various types: tuples, lists, arrays,
    # NaN, callables, splitters, scorers) are reported as non-default (issue #31525).
    """Check that we detect non-default parameters with various types.

    Non-regression test for:
    https://github.com/scikit-learn/scikit-learn/issues/31525
    """
    estimator = make_estimator_with_param(default_value)(param=test_value)
    non_default = estimator._get_params_html().non_default
    assert "param" in non_default


def test_param_is_non_default_when_pandas_NA():
    # [FUNCTION DOC]
    # Verifies pandas.NA is reported as non-default against default 0 (issue #32312).
    """Check that we detect pandas.Na as non-default parameter.

    Non-regression test for:
    https://github.com/scikit-learn/scikit-learn/issues/32312
    """
    pd = pytest.importorskip("pandas")

    estimator = make_estimator_with_param(default_value=0)(param=pd.NA)
    non_default = estimator._get_params_html().non_default
    assert "param" in non_default


@pytest.mark.parametrize(
    "default_value, test_value",
    [
        (None, None),
        ((), ()),
        ((), []),
        ((), np.array([])),
        ((1, 2, 3), (1, 2, 3)),
        ((1, 2, 3), [1, 2, 3]),
        ((1, 2, 3), np.array([1, 2, 3])),
        (np.nan, np.nan),
        ("abc", "abc"),
        (True, True),
        (1, 1),
        (1.0, 1.0),
        (2, 2.0),
    ],
)
def test_param_is_default(default_value, test_value):
    # [FUNCTION DOC]
    # Verifies values equal to the default, including array-likes with equal elements,
    # NaN vs NaN and 2 vs 2.0, are not reported as non-default.
    """Check that we detect the default parameters and values in an array-like will
    be reported as default as well.

    Non-regression test for:
    https://github.com/scikit-learn/scikit-learn/issues/31525
    """
    estimator = make_estimator_with_param(default_value)(param=test_value)
    non_default = estimator._get_params_html().non_default
    assert "param" not in non_default
