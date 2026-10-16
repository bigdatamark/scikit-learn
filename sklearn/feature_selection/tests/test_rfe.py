# [MODULE DOC]
# Tests for RFE and RFECV: elimination with dense/sparse data, step and fraction
#   handling, metadata routing to fit/score, cv_results_ contents and sizes, ranking
#   formulas, n_jobs/threading consistency, groups, importance_getter, NaN/inf support,
#   PLS and pipelines, warnings, sample weights and sparse coefficients. Most tests
#   augment iris with 6 Gaussian noise columns and expect the 4 real features to be kept.
"""
Testing Recursive feature elimination
"""

import re
from operator import attrgetter

import numpy as np
import pytest
from joblib import parallel_backend
from numpy.testing import assert_allclose, assert_array_almost_equal, assert_array_equal

from sklearn.base import BaseEstimator, ClassifierMixin, is_classifier
from sklearn.compose import TransformedTargetRegressor
from sklearn.cross_decomposition import CCA, PLSCanonical, PLSRegression
from sklearn.datasets import load_iris, make_classification, make_friedman1
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, RFECV
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import get_scorer, make_scorer, zero_one_loss
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR, LinearSVR
from sklearn.utils import check_random_state
from sklearn.utils._testing import ignore_warnings
from sklearn.utils.fixes import CSR_CONTAINERS


class MockClassifier(ClassifierMixin, BaseEstimator):
    # [CLASS DOC]
    # HELPER: Dummy classifier with uniform coef_ (all ones) so every feature is equally
    #   important; predict/predict_proba/decision_function/transform all return ones,
    #   score returns 0.0, and tags allow NaN.
    """
    Dummy classifier to test recursive feature elimination
    """

    def __init__(self, foo_param=0):
        # [FUNCTION DOC]
        # HELPER: Stores foo_param (default 0), an unused parameter.
        self.foo_param = foo_param

    def fit(self, X, y):
        # [FUNCTION DOC]
        # HELPER: Asserts X and y have equal length, sets coef_ to ones of length
        #   n_features and classes_ to the sorted unique labels; returns self.
        assert len(X) == len(y)
        self.coef_ = np.ones(X.shape[1], dtype=np.float64)
        self.classes_ = sorted(set(y))
        return self

    def predict(self, T):
        # [FUNCTION DOC]
        # HELPER: Returns a vector of ones of length n_samples of T (also aliased as
        #   predict_proba, decision_function and transform, so those return 1-D output
        #   too).
        return np.ones(T.shape[0])

    predict_proba = predict
    decision_function = predict
    transform = predict

    def score(self, X=None, y=None):
        # [FUNCTION DOC]
        # HELPER: Always returns 0.0; X and y ignored.
        return 0.0

    def get_params(self, deep=True):
        # [FUNCTION DOC]
        # HELPER: Returns {"foo_param": foo_param}; deep ignored.
        return {"foo_param": self.foo_param}

    def set_params(self, **params):
        # [FUNCTION DOC]
        # HELPER: Ignores all params and returns self.
        # KNOWN ISSUE: parameters are silently not set, unlike a real estimator.
        return self

    def __sklearn_tags__(self):
        # [FUNCTION DOC]
        # HELPER: Returns parent tags with input_tags.allow_nan set to True.
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        return tags


def test_rfe_features_importance():
    # [FUNCTION DOC]
    # Checks RFE with a random forest (feature_importances_) and with a linear SVC
    #   (coef_) select the same 4 features on noisy iris, and that ranking_ covers all
    #   features.
    generator = check_random_state(0)
    iris = load_iris()
    # Add some irrelevant features. Random seed is set to make sure that
    # irrelevant features are always irrelevant.
    X = np.c_[iris.data, generator.normal(size=(len(iris.data), 6))]
    y = iris.target

    clf = RandomForestClassifier(n_estimators=20, random_state=generator, max_depth=2)
    rfe = RFE(estimator=clf, n_features_to_select=4, step=0.1)
    rfe.fit(X, y)
    assert len(rfe.ranking_) == X.shape[1]

    clf_svc = SVC(kernel="linear")
    rfe_svc = RFE(estimator=clf_svc, n_features_to_select=4, step=0.1)
    rfe_svc.fit(X, y)

    # Check if the supports are equal
    assert_array_equal(rfe.get_support(), rfe_svc.get_support())


@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_rfe(csr_container):
    # [FUNCTION DOC]
    # Checks RFE with linear SVC on noisy iris recovers exactly the 4 iris columns, that
    #   predict/score match an SVC fit on the original data, and that dense and CSR input
    #   give the same transform.
    generator = check_random_state(0)
    iris = load_iris()
    # Add some irrelevant features. Random seed is set to make sure that
    # irrelevant features are always irrelevant.
    X = np.c_[iris.data, generator.normal(size=(len(iris.data), 6))]
    X_sparse = csr_container(X)
    y = iris.target

    # dense model
    clf = SVC(kernel="linear")
    rfe = RFE(estimator=clf, n_features_to_select=4, step=0.1)
    rfe.fit(X, y)
    X_r = rfe.transform(X)
    clf.fit(X_r, y)
    assert len(rfe.ranking_) == X.shape[1]

    # sparse model
    clf_sparse = SVC(kernel="linear")
    rfe_sparse = RFE(estimator=clf_sparse, n_features_to_select=4, step=0.1)
    rfe_sparse.fit(X_sparse, y)
    X_r_sparse = rfe_sparse.transform(X_sparse)

    assert X_r.shape == iris.data.shape
    assert_array_almost_equal(X_r[:10], iris.data[:10])

    assert_array_almost_equal(rfe.predict(X), clf.predict(iris.data))
    assert rfe.score(X, y) == clf.score(iris.data, iris.target)
    assert_array_almost_equal(X_r, X_r_sparse.toarray())


def test_RFE_fit_score_params():
    # [FUNCTION DOC]
    # Checks extra fit/score keyword arguments (prop) are forwarded to the estimator:
    #   missing prop raises the estimator's ValueError in fit or score; passing it works.
    # Make sure RFE passes the metadata down to fit and score methods of the
    # underlying estimator
    class TestEstimator(BaseEstimator, ClassifierMixin):
        # [CLASS DOC]
        # HELPER: Classifier requiring a prop kwarg in fit and score; wraps a linear SVC
        #   and exposes its coef_.
        def fit(self, X, y, prop=None):
            # [FUNCTION DOC]
            # HELPER: Raises ValueError if prop is None; otherwise fits a linear SVC,
            #   copies its coef_ and returns self.
            if prop is None:
                raise ValueError("fit: prop cannot be None")
            self.svc_ = SVC(kernel="linear").fit(X, y)
            self.coef_ = self.svc_.coef_
            return self

        def score(self, X, y, prop=None):
            # [FUNCTION DOC]
            # HELPER: Raises ValueError if prop is None; otherwise returns the inner SVC
            #   accuracy.
            if prop is None:
                raise ValueError("score: prop cannot be None")
            return self.svc_.score(X, y)

    X, y = load_iris(return_X_y=True)
    with pytest.raises(ValueError, match="fit: prop cannot be None"):
        RFE(estimator=TestEstimator()).fit(X, y)
    with pytest.raises(ValueError, match="score: prop cannot be None"):
        RFE(estimator=TestEstimator()).fit(X, y, prop="foo").score(X, y)

    RFE(estimator=TestEstimator()).fit(X, y, prop="foo").score(X, y, prop="foo")


def test_rfe_percent_n_features():
    # [FUNCTION DOC]
    # Checks n_features_to_select=0.4 on 10 features gives the same ranking and support
    #   as 4.
    # test that the results are the same
    generator = check_random_state(0)
    iris = load_iris()
    # Add some irrelevant features. Random seed is set to make sure that
    # irrelevant features are always irrelevant.
    X = np.c_[iris.data, generator.normal(size=(len(iris.data), 6))]
    y = iris.target
    # there are 10 features in the data. We select 40%.
    clf = SVC(kernel="linear")
    rfe_num = RFE(estimator=clf, n_features_to_select=4, step=0.1)
    rfe_num.fit(X, y)

    rfe_perc = RFE(estimator=clf, n_features_to_select=0.4, step=0.1)
    rfe_perc.fit(X, y)

    assert_array_equal(rfe_perc.ranking_, rfe_num.ranking_)
    assert_array_equal(rfe_perc.support_, rfe_num.support_)


def test_rfe_mockclassifier():
    # [FUNCTION DOC]
    # Checks RFE works with MockClassifier (all-equal importances) and keeps 4 columns.
    generator = check_random_state(0)
    iris = load_iris()
    # Add some irrelevant features. Random seed is set to make sure that
    # irrelevant features are always irrelevant.
    X = np.c_[iris.data, generator.normal(size=(len(iris.data), 6))]
    y = iris.target

    # dense model
    clf = MockClassifier()
    rfe = RFE(estimator=clf, n_features_to_select=4, step=0.1)
    rfe.fit(X, y)
    X_r = rfe.transform(X)
    clf.fit(X_r, y)
    assert len(rfe.ranking_) == X.shape[1]
    assert X_r.shape == iris.data.shape


@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_rfecv(csr_container):
    # [FUNCTION DOC]
    # On noisy iris checks RFECV: cv_results_ arrays have one entry per feature count,
    #   the transform recovers iris data for default scoring, a custom loss scorer, and an
    #   accuracy scorer; a constant scorer leads to selecting a single feature (fewest
    #   features on ties); step=2 gives 6 entries; dense/sparse and fractional step all
    #   recover iris.
    generator = check_random_state(0)
    iris = load_iris()
    # Add some irrelevant features. Random seed is set to make sure that
    # irrelevant features are always irrelevant.
    X = np.c_[iris.data, generator.normal(size=(len(iris.data), 6))]
    y = list(iris.target)  # regression test: list should be supported

    # Test using the score function
    rfecv = RFECV(estimator=SVC(kernel="linear"), step=1)
    rfecv.fit(X, y)
    # non-regression test for missing worst feature:

    for key in rfecv.cv_results_.keys():
        assert len(rfecv.cv_results_[key]) == X.shape[1]

    assert len(rfecv.ranking_) == X.shape[1]
    X_r = rfecv.transform(X)

    # All the noisy variable were filtered out
    assert_array_equal(X_r, iris.data)

    # same in sparse
    rfecv_sparse = RFECV(estimator=SVC(kernel="linear"), step=1)
    X_sparse = csr_container(X)
    rfecv_sparse.fit(X_sparse, y)
    X_r_sparse = rfecv_sparse.transform(X_sparse)
    assert_array_equal(X_r_sparse.toarray(), iris.data)

    # Test using a customized loss function
    scoring = make_scorer(zero_one_loss, greater_is_better=False)
    rfecv = RFECV(estimator=SVC(kernel="linear"), step=1, scoring=scoring)
    ignore_warnings(rfecv.fit)(X, y)
    X_r = rfecv.transform(X)
    assert_array_equal(X_r, iris.data)

    # Test using a scorer
    scorer = get_scorer("accuracy")
    rfecv = RFECV(estimator=SVC(kernel="linear"), step=1, scoring=scorer)
    rfecv.fit(X, y)
    X_r = rfecv.transform(X)
    assert_array_equal(X_r, iris.data)

    # Test fix on cv_results_
    def test_scorer(estimator, X, y):
        # [FUNCTION DOC]
        # HELPER: Scorer that always returns 1.0, used to force CV score ties.
        return 1.0

    rfecv = RFECV(estimator=SVC(kernel="linear"), step=1, scoring=test_scorer)
    rfecv.fit(X, y)

    # In the event of cross validation score ties, the expected behavior of
    # RFECV is to return the FEWEST features that maximize the CV score.
    # Because test_scorer always returns 1.0 in this example, RFECV should
    # reduce the dimensionality to a single feature (i.e. n_features_ = 1)
    assert rfecv.n_features_ == 1

    # Same as the first two tests, but with step=2
    rfecv = RFECV(estimator=SVC(kernel="linear"), step=2)
    rfecv.fit(X, y)

    for key in rfecv.cv_results_.keys():
        assert len(rfecv.cv_results_[key]) == 6

    assert len(rfecv.ranking_) == X.shape[1]
    X_r = rfecv.transform(X)
    assert_array_equal(X_r, iris.data)

    rfecv_sparse = RFECV(estimator=SVC(kernel="linear"), step=2)
    X_sparse = csr_container(X)
    rfecv_sparse.fit(X_sparse, y)
    X_r_sparse = rfecv_sparse.transform(X_sparse)
    assert_array_equal(X_r_sparse.toarray(), iris.data)

    # Verifying that steps < 1 don't blow up.
    rfecv_sparse = RFECV(estimator=SVC(kernel="linear"), step=0.2)
    X_sparse = csr_container(X)
    rfecv_sparse.fit(X_sparse, y)
    X_r_sparse = rfecv_sparse.transform(X_sparse)
    assert_array_equal(X_r_sparse.toarray(), iris.data)


def test_rfecv_mockclassifier():
    # [FUNCTION DOC]
    # Checks RFECV with MockClassifier and list y yields cv_results_ entries and
    #   ranking_ of length n_features.
    generator = check_random_state(0)
    iris = load_iris()
    X = np.c_[iris.data, generator.normal(size=(len(iris.data), 6))]
    y = list(iris.target)  # regression test: list should be supported

    # Test using the score function
    rfecv = RFECV(estimator=MockClassifier(), step=1)
    rfecv.fit(X, y)
    # non-regression test for missing worst feature:

    for key in rfecv.cv_results_.keys():
        assert len(rfecv.cv_results_[key]) == X.shape[1]

    assert len(rfecv.ranking_) == X.shape[1]


def test_rfecv_verbose_output():
    # [FUNCTION DOC]
    # Redirects stdout to a StringIO and checks RFECV with verbose=1 writes something.
    # KNOWN ISSUE: sys.stdout is replaced and never restored, leaking into later tests
    #   (pytest capture usually masks it).
    # Check verbose=1 is producing an output.
    import sys
    from io import StringIO

    sys.stdout = StringIO()

    generator = check_random_state(0)
    iris = load_iris()
    X = np.c_[iris.data, generator.normal(size=(len(iris.data), 6))]
    y = list(iris.target)

    rfecv = RFECV(estimator=SVC(kernel="linear"), step=1, verbose=1)
    rfecv.fit(X, y)

    verbose_output = sys.stdout
    verbose_output.seek(0)
    assert len(verbose_output.readline()) > 0


def test_rfecv_cv_results_size(global_random_seed):
    # [FUNCTION DOC]
    # For several (step, min_features_to_select) pairs with MockClassifier, checks each
    #   cv_results_ array has ceil((n_features - min)/step) + 1 entries and n_features_ >=
    #   min.
    generator = check_random_state(global_random_seed)
    iris = load_iris()
    X = np.c_[iris.data, generator.normal(size=(len(iris.data), 6))]
    y = list(iris.target)  # regression test: list should be supported

    # Non-regression test for varying combinations of step and
    # min_features_to_select.
    for step, min_features_to_select in [[2, 1], [2, 2], [3, 3]]:
        rfecv = RFECV(
            estimator=MockClassifier(),
            step=step,
            min_features_to_select=min_features_to_select,
        )
        rfecv.fit(X, y)

        score_len = np.ceil((X.shape[1] - min_features_to_select) / step) + 1

        for key in rfecv.cv_results_.keys():
            assert len(rfecv.cv_results_[key]) == score_len

        assert len(rfecv.ranking_) == X.shape[1]
        assert rfecv.n_features_ >= min_features_to_select


def test_rfe_estimator_tags():
    # [FUNCTION DOC]
    # Checks RFE wrapping a classifier is a classifier (so cross_val_score stratifies)
    #   and achieves min CV accuracy above 0.7 on iris.
    rfe = RFE(SVC(kernel="linear"))
    assert is_classifier(rfe)
    # make sure that cross-validation is stratified
    iris = load_iris()
    score = cross_val_score(rfe, iris.data, iris.target)
    assert score.min() > 0.7


def test_rfe_min_step(global_random_seed):
    # [FUNCTION DOC]
    # Checks step as a tiny fraction (rounding to 0, treated as 1), a fraction, and an
    #   int all end with n_features // 2 selected (the default).
    n_features = 10
    X, y = make_friedman1(
        n_samples=50, n_features=n_features, random_state=global_random_seed
    )
    n_samples, n_features = X.shape
    estimator = SVR(kernel="linear")

    # Test when floor(step * n_features) <= 0
    selector = RFE(estimator, step=0.01)
    sel = selector.fit(X, y)
    assert sel.support_.sum() == n_features // 2

    # Test when step is between (0,1) and floor(step * n_features) > 0
    selector = RFE(estimator, step=0.20)
    sel = selector.fit(X, y)
    assert sel.support_.sum() == n_features // 2

    # Test when step is an integer
    selector = RFE(estimator, step=5)
    sel = selector.fit(X, y)
    assert sel.support_.sum() == n_features // 2


def test_number_of_subsets_of_features(global_random_seed):
    # [FUNCTION DOC]
    # Checks two equivalent formulas for the number of elimination iterations agree with
    #   max(ranking_) for RFE and with the length of each cv_results_ array for RFECV, for
    #   divisible and non-divisible step cases.
    # In RFE, 'number_of_subsets_of_features'
    # = the number of iterations in '_fit'
    # = max(ranking_)
    # = 1 + (n_features + step - n_features_to_select - 1) // step
    # After optimization #4534, this number
    # = 1 + np.ceil((n_features - n_features_to_select) / float(step))
    # This test case is to test their equivalence, refer to #4534 and #3824

    def formula1(n_features, n_features_to_select, step):
        # [FUNCTION DOC]
        # HELPER: Integer-division formula 1 + (n_features + step - n_features_to_select
        #   - 1) // step. Outputs: int.
        return 1 + ((n_features + step - n_features_to_select - 1) // step)

    def formula2(n_features, n_features_to_select, step):
        # [FUNCTION DOC]
        # HELPER: Ceiling formula 1 + ceil((n_features - n_features_to_select) / step).
        # Outputs: float.
        return 1 + np.ceil((n_features - n_features_to_select) / float(step))

    # RFE
    # Case 1, n_features - n_features_to_select is divisible by step
    # Case 2, n_features - n_features_to_select is not divisible by step
    n_features_list = [11, 11]
    n_features_to_select_list = [3, 3]
    step_list = [2, 3]
    for n_features, n_features_to_select, step in zip(
        n_features_list, n_features_to_select_list, step_list
    ):
        generator = check_random_state(global_random_seed)
        X = generator.normal(size=(100, n_features))
        y = generator.rand(100).round()
        rfe = RFE(
            estimator=SVC(kernel="linear"),
            n_features_to_select=n_features_to_select,
            step=step,
        )
        rfe.fit(X, y)
        # this number also equals to the maximum of ranking_
        assert np.max(rfe.ranking_) == formula1(n_features, n_features_to_select, step)
        assert np.max(rfe.ranking_) == formula2(n_features, n_features_to_select, step)

    # In RFECV, 'fit' calls 'RFE._fit'
    # 'number_of_subsets_of_features' of RFE
    # = the size of each score in 'cv_results_' of RFECV
    # = the number of iterations of the for loop before optimization #4534

    # RFECV, n_features_to_select = 1
    # Case 1, n_features - 1 is divisible by step
    # Case 2, n_features - 1 is not divisible by step

    n_features_to_select = 1
    n_features_list = [11, 10]
    step_list = [2, 2]
    for n_features, step in zip(n_features_list, step_list):
        generator = check_random_state(global_random_seed)
        X = generator.normal(size=(100, n_features))
        y = generator.rand(100).round()
        rfecv = RFECV(estimator=SVC(kernel="linear"), step=step)
        rfecv.fit(X, y)

        for key in rfecv.cv_results_.keys():
            assert len(rfecv.cv_results_[key]) == formula1(
                n_features, n_features_to_select, step
            )
            assert len(rfecv.cv_results_[key]) == formula2(
                n_features, n_features_to_select, step
            )


def test_rfe_cv_n_jobs(global_random_seed):
    # [FUNCTION DOC]
    # Checks RFECV with n_jobs=2 reproduces the ranking_ and cv_results_ of n_jobs=None.
    generator = check_random_state(global_random_seed)
    iris = load_iris()
    X = np.c_[iris.data, generator.normal(size=(len(iris.data), 6))]
    y = iris.target

    rfecv = RFECV(estimator=SVC(kernel="linear"))
    rfecv.fit(X, y)
    rfecv_ranking = rfecv.ranking_

    rfecv_cv_results_ = rfecv.cv_results_

    rfecv.set_params(n_jobs=2)
    rfecv.fit(X, y)
    assert_array_almost_equal(rfecv.ranking_, rfecv_ranking)

    assert rfecv_cv_results_.keys() == rfecv.cv_results_.keys()
    for key in rfecv_cv_results_.keys():
        assert rfecv_cv_results_[key] == pytest.approx(rfecv.cv_results_[key])


def test_rfe_cv_groups():
    # [FUNCTION DOC]
    # Checks groups are forwarded to a GroupKFold cv and fitting selects at least one
    #   feature.
    generator = check_random_state(0)
    iris = load_iris()
    number_groups = 4
    groups = np.floor(np.linspace(0, number_groups, len(iris.target)))
    X = iris.data
    y = (iris.target > 0).astype(int)

    est_groups = RFECV(
        estimator=RandomForestClassifier(random_state=generator),
        step=1,
        scoring="accuracy",
        cv=GroupKFold(n_splits=2),
    )
    est_groups.fit(X, y, groups=groups)
    assert est_groups.n_features_ > 0


@pytest.mark.parametrize(
    "importance_getter", [attrgetter("regressor_.coef_"), "regressor_.coef_"]
)
@pytest.mark.parametrize("selector, expected_n_features", [(RFE, 5), (RFECV, 4)])
def test_rfe_wrapped_estimator(importance_getter, selector, expected_n_features):
    # [FUNCTION DOC]
    # Non-regression: importance_getter given as attrgetter or dotted string into a
    #   TransformedTargetRegressor works; RFE keeps 5 and RFECV keeps 4 features.
    # Non-regression test for
    # https://github.com/scikit-learn/scikit-learn/issues/15312
    X, y = make_friedman1(n_samples=50, n_features=10, random_state=0)
    estimator = LinearSVR(random_state=0)

    log_estimator = TransformedTargetRegressor(
        regressor=estimator, func=np.log, inverse_func=np.exp
    )

    selector = selector(log_estimator, importance_getter=importance_getter)
    sel = selector.fit(X, y)
    assert sel.support_.sum() == expected_n_features


@pytest.mark.parametrize(
    "importance_getter, err_type",
    [
        ("auto", ValueError),
        ("random", AttributeError),
        (lambda x: x.importance, AttributeError),
    ],
)
@pytest.mark.parametrize("Selector", [RFE, RFECV])
def test_rfe_importance_getter_validation(importance_getter, err_type, Selector):
    # [FUNCTION DOC]
    # Checks "auto" raises ValueError when the estimator has no
    #   coef_/feature_importances_, and an invalid attribute path or callable raises
    #   AttributeError, for RFE and RFECV.
    X, y = make_friedman1(n_samples=50, n_features=10, random_state=42)
    estimator = LinearSVR()
    log_estimator = TransformedTargetRegressor(
        regressor=estimator, func=np.log, inverse_func=np.exp
    )

    with pytest.raises(err_type):
        model = Selector(log_estimator, importance_getter=importance_getter)
        model.fit(X, y)


@pytest.mark.parametrize("cv", [None, 5])
def test_rfe_allow_nan_inf_in_x(cv):
    # [FUNCTION DOC]
    # Checks RFE/RFECV fit and transform accept NaN and inf in X when the estimator
    #   allows NaN.
    iris = load_iris()
    X = iris.data
    y = iris.target

    # add nan and inf value to X
    X[0][0] = np.nan
    X[0][1] = np.inf

    clf = MockClassifier()
    if cv is not None:
        rfe = RFECV(estimator=clf, cv=cv)
    else:
        rfe = RFE(estimator=clf)
    rfe.fit(X, y)
    rfe.transform(X)


def test_w_pipeline_2d_coef_():
    # [FUNCTION DOC]
    # Checks RFE on a pipeline using a 2-D (multiclass) coef_ via a dotted
    #   importance_getter selects 2 features.
    pipeline = make_pipeline(StandardScaler(), LogisticRegression())

    data, y = load_iris(return_X_y=True)
    sfm = RFE(
        pipeline,
        n_features_to_select=2,
        importance_getter="named_steps.logisticregression.coef_",
    )

    sfm.fit(data, y)
    assert sfm.transform(data).shape[1] == 2


def test_rfecv_std_and_mean(global_random_seed):
    # [FUNCTION DOC]
    # Checks mean_test_score and std_test_score in cv_results_ equal the mean and std
    #   across the splitN_test_score entries.
    generator = check_random_state(global_random_seed)
    iris = load_iris()
    X = np.c_[iris.data, generator.normal(size=(len(iris.data), 6))]
    y = iris.target

    rfecv = RFECV(estimator=SVC(kernel="linear"))
    rfecv.fit(X, y)
    split_keys = [
        key
        for key in rfecv.cv_results_.keys()
        if re.search(r"split\d+_test_score", key)
    ]
    cv_scores = np.asarray([rfecv.cv_results_[key] for key in split_keys])
    expected_mean = np.mean(cv_scores, axis=0)
    expected_std = np.std(cv_scores, axis=0)

    assert_allclose(rfecv.cv_results_["mean_test_score"], expected_mean)
    assert_allclose(rfecv.cv_results_["std_test_score"], expected_std)


@pytest.mark.parametrize(
    ["min_features_to_select", "n_features", "step", "cv_results_n_features"],
    [
        [1, 4, 1, np.array([1, 2, 3, 4])],
        [1, 5, 1, np.array([1, 2, 3, 4, 5])],
        [1, 4, 2, np.array([1, 2, 4])],
        [1, 5, 2, np.array([1, 3, 5])],
        [1, 4, 3, np.array([1, 4])],
        [1, 5, 3, np.array([1, 2, 5])],
        [1, 4, 4, np.array([1, 4])],
        [1, 5, 4, np.array([1, 5])],
        [4, 4, 2, np.array([4])],
        [4, 5, 1, np.array([4, 5])],
        [4, 5, 2, np.array([4, 5])],
    ],
)
def test_rfecv_cv_results_n_features(
    min_features_to_select,
    n_features,
    step,
    cv_results_n_features,
):
    # [FUNCTION DOC]
    # Parametrized over min_features_to_select, n_features and step: checks
    #   cv_results_["n_features"] lists the expected feature counts (always including the
    #   full count) and all cv_results_ arrays have the same length.
    X, y = make_classification(
        n_samples=20, n_features=n_features, n_informative=n_features, n_redundant=0
    )
    rfecv = RFECV(
        estimator=SVC(kernel="linear"),
        step=step,
        min_features_to_select=min_features_to_select,
    )
    rfecv.fit(X, y)
    assert_array_equal(rfecv.cv_results_["n_features"], cv_results_n_features)
    assert all(
        len(value) == len(rfecv.cv_results_["n_features"])
        for value in rfecv.cv_results_.values()
    )


@pytest.mark.parametrize("ClsRFE", [RFE, RFECV])
def test_multioutput(ClsRFE):
    # [FUNCTION DOC]
    # Smoke test: RFE/RFECV fit with a random forest on 2-output binary targets.
    # Note: uses unseeded global np.random data.
    X = np.random.normal(size=(10, 3))
    y = np.random.randint(2, size=(10, 2))
    clf = RandomForestClassifier(n_estimators=5)
    rfe_test = ClsRFE(clf)
    rfe_test.fit(X, y)


@pytest.mark.parametrize("ClsRFE", [RFE, RFECV])
def test_pipeline_with_nans(ClsRFE):
    # [FUNCTION DOC]
    # Non-regression: RFE/RFECV with a pipeline containing an imputer fits data with a
    #   NaN.
    """Check that RFE works with pipeline that accept nans.

    Non-regression test for gh-21743.
    """
    X, y = load_iris(return_X_y=True)
    X[0, 0] = np.nan

    pipe = make_pipeline(
        SimpleImputer(),
        StandardScaler(),
        LogisticRegression(),
    )

    fs = ClsRFE(
        estimator=pipe,
        importance_getter="named_steps.logisticregression.coef_",
    )
    fs.fit(X, y)


@pytest.mark.parametrize("ClsRFE", [RFE, RFECV])
@pytest.mark.parametrize("PLSEstimator", [CCA, PLSCanonical, PLSRegression])
def test_rfe_pls(ClsRFE, PLSEstimator):
    # [FUNCTION DOC]
    # Non-regression: RFE/RFECV with one-component CCA/PLSCanonical/PLSRegression fit
    #   and score above 0.5 on Friedman1.
    """Check the behaviour of RFE with PLS estimators.

    Non-regression test for:
    https://github.com/scikit-learn/scikit-learn/issues/12410
    """
    X, y = make_friedman1(n_samples=50, n_features=10, random_state=0)
    estimator = PLSEstimator(n_components=1)
    selector = ClsRFE(estimator, step=1).fit(X, y)
    assert selector.score(X, y) > 0.5


def test_rfe_estimator_attribute_error():
    # [FUNCTION DOC]
    # Non-regression: decision_function on RFE wrapping LinearRegression raises
    #   AttributeError naming RFE, chained from the inner estimator's AttributeError.
    """Check that we raise the proper AttributeError when the estimator
    does not implement the `decision_function` method, which is decorated with
    `available_if`.

    Non-regression test for:
    https://github.com/scikit-learn/scikit-learn/issues/28108
    """
    iris = load_iris()

    # `LinearRegression` does not implement 'decision_function' and should raise an
    # AttributeError
    rfe = RFE(estimator=LinearRegression())

    outer_msg = "This 'RFE' has no attribute 'decision_function'"
    inner_msg = "'LinearRegression' object has no attribute 'decision_function'"
    with pytest.raises(AttributeError, match=outer_msg) as exec_info:
        rfe.fit(iris.data, iris.target).decision_function(iris.data)
    assert isinstance(exec_info.value.__cause__, AttributeError)
    assert inner_msg in str(exec_info.value.__cause__)


@pytest.mark.parametrize(
    "ClsRFE, param", [(RFE, "n_features_to_select"), (RFECV, "min_features_to_select")]
)
def test_rfe_n_features_to_select_warning(ClsRFE, param):
    # [FUNCTION DOC]
    # Checks a UserWarning "<param>=21 > n_features=20" is emitted when the requested
    #   count exceeds the number of features, for RFE and RFECV.
    """Check if the correct warning is raised when trying to initialize a RFE
    object with an n_features_to_select attribute larger than the number of
    features present in the X variable that is passed to the fit method
    """
    X, y = make_classification(n_features=20, random_state=0)

    with pytest.warns(UserWarning, match=f"{param}=21 > n_features=20"):
        # Create RFE/RFECV with n_features_to_select/min_features_to_select
        # larger than the number of features present in the X variable
        clsrfe = ClsRFE(estimator=LogisticRegression(), **{param: 21})
        clsrfe.fit(X, y)


def test_rfe_with_sample_weight():
    # [FUNCTION DOC]
    # Checks weight 2 on the first half of samples gives the same ranking as duplicating
    #   those samples, and unit weights give a different ranking than the duplicated data.
    """Test that `RFE` works correctly with sample weights."""
    X, y = make_classification(random_state=0)
    n_samples = X.shape[0]

    # Assign the first half of the samples with twice the weight
    sample_weight = np.ones_like(y)
    sample_weight[: n_samples // 2] = 2

    # Duplicate the first half of the data samples to replicate the effect
    # of sample weights for comparison
    X2 = np.concatenate([X, X[: n_samples // 2]], axis=0)
    y2 = np.concatenate([y, y[: n_samples // 2]])

    estimator = SVC(kernel="linear")

    rfe_sw = RFE(estimator=estimator, step=0.1)
    rfe_sw.fit(X, y, sample_weight=sample_weight)

    rfe = RFE(estimator=estimator, step=0.1)
    rfe.fit(X2, y2)

    assert_array_equal(rfe_sw.ranking_, rfe.ranking_)

    # Also verify that when sample weights are not doubled the results
    # are different from the duplicated data
    rfe_sw_2 = RFE(estimator=estimator, step=0.1)
    sample_weight_2 = np.ones_like(y)
    rfe_sw_2.fit(X, y, sample_weight=sample_weight_2)

    assert not np.array_equal(rfe_sw_2.ranking_, rfe.ranking_)


def test_rfe_with_joblib_threading_backend(global_random_seed):
    # [FUNCTION DOC]
    # Checks RFECV with n_jobs=2 gives the same ranking_ under the threading joblib
    #   backend as with the default backend.
    X, y = make_classification(random_state=global_random_seed)

    clf = LogisticRegression()
    rfe = RFECV(
        estimator=clf,
        n_jobs=2,
    )

    rfe.fit(X, y)
    ranking_ref = rfe.ranking_

    with parallel_backend("threading"):
        rfe.fit(X, y)

    assert_array_equal(ranking_ref, rfe.ranking_)


def test_results_per_cv_in_rfecv(global_random_seed):
    # [FUNCTION DOC]
    # Checks per-split cv_results_ arrays (test score, support, ranking) have equal
    #   lengths across folds.
    """
    Test that the results of RFECV are consistent across the different folds
    in terms of length of the arrays.
    """
    X, y = make_classification(random_state=global_random_seed)

    clf = LogisticRegression()
    rfecv = RFECV(
        estimator=clf,
        n_jobs=2,
        cv=5,
    )

    rfecv.fit(X, y)

    assert len(rfecv.cv_results_["split1_test_score"]) == len(
        rfecv.cv_results_["split2_test_score"]
    )
    assert len(rfecv.cv_results_["split1_support"]) == len(
        rfecv.cv_results_["split2_support"]
    )
    assert len(rfecv.cv_results_["split1_ranking"]) == len(
        rfecv.cv_results_["split2_ranking"]
    )


@pytest.mark.parametrize(
    "feature_importance",
    [
        lambda estimator: estimator.sparsify().coef_,
        lambda estimator: estimator.sparsify().coef_.tocsc(),
    ],
)
def test_rfe_sparse_coef(feature_importance):
    # [FUNCTION DOC]
    # Checks an importance_getter returning sparse coef_ (CSR or CSC) yields the same
    #   support and ranking as the dense default.
    X = [[0, 1, 3], [1, 0, 0], [2, 0, 4], [0, 2, 4]]
    y = [0, 1, 2, 3]

    estimator = LogisticRegression()
    selector_sparse = RFE(
        estimator, n_features_to_select=1, importance_getter=feature_importance
    )
    selector_dense = RFE(estimator, n_features_to_select=1)
    selector_sparse.fit(X, y)
    selector_dense.fit(X, y)

    assert_array_equal(selector_sparse.support_, selector_dense.support_)
    assert_array_equal(selector_sparse.ranking_, selector_dense.ranking_)
