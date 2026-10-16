# [MODULE DOC]
# Tests for estimator pretty printing (sklearn.utils._pprint._EstimatorPrettyPrinter and
# BaseEstimator.__repr__): full and changed-only reprs, nested estimators, depth limits,
# element-count ellipsis, character-count (N_CHAR_MAX) ellipsis, builtin PrettyPrinter
# compatibility, kwargs-based estimators and repr call counts. Defines simplified copies
# of several estimator constructors (with frozen, older defaults) so expected reprs stay
# stable.
import re
from pprint import PrettyPrinter

import numpy as np
import pytest

from sklearn import config_context
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import make_pipeline
from sklearn.utils._pprint import _EstimatorPrettyPrinter


# Constructors excerpted to test pprinting
class LogisticRegression(BaseEstimator):
    # [CLASS DOC]
    # HELPER: Stripped-down LogisticRegression with only a constructor (old default
    # values) and a no-op fit, used as a repr target.
    def __init__(
        self,
        C=1.0,
        l1_ratio=0,
        dual=False,
        tol=1e-4,
        fit_intercept=True,
        intercept_scaling=1,
        class_weight=None,
        random_state=None,
        solver="warn",
        max_iter=100,
        multi_class="warn",
        verbose=0,
        warm_start=False,
        n_jobs=None,
    ):
        # [FUNCTION DOC]
        # HELPER: Stores every constructor argument unchanged as an attribute of the
        # same name.
        # Inputs: the listed hyperparameters with their old defaults.
        # Side effects: sets one attribute per parameter.
        self.C = C
        self.l1_ratio = l1_ratio
        self.dual = dual
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.intercept_scaling = intercept_scaling
        self.class_weight = class_weight
        self.random_state = random_state
        self.solver = solver
        self.max_iter = max_iter
        self.multi_class = multi_class
        self.verbose = verbose
        self.warm_start = warm_start
        self.n_jobs = n_jobs

    def fit(self, X, y):
        # [FUNCTION DOC]
        # HELPER: No-op fit. Inputs: X, y - ignored. Outputs: self.
        return self


class StandardScaler(TransformerMixin, BaseEstimator):
    # [CLASS DOC]
    # HELPER: Stripped-down StandardScaler (copy, with_mean, with_std) used as a
    # pipeline step in repr tests.
    def __init__(self, copy=True, with_mean=True, with_std=True):
        # [FUNCTION DOC]
        # HELPER: Stores copy, with_mean and with_std (all default True) as attributes.
        self.with_mean = with_mean
        self.with_std = with_std
        self.copy = copy

    def transform(self, X, copy=None):
        # [FUNCTION DOC]
        # HELPER: Stub transform. Inputs: X, copy - ignored. Outputs: self.
        # Note: returns self rather than transformed data; it is never used for actual
        # data transformation in these tests.
        return self


class RFE(BaseEstimator):
    # [CLASS DOC]
    # HELPER: Stripped-down RFE used to build deeply nested estimators.
    def __init__(self, estimator, n_features_to_select=None, step=1, verbose=0):
        # [FUNCTION DOC]
        # HELPER: Stores estimator, n_features_to_select (default None), step (default
        # 1) and verbose (default 0) as attributes.
        self.estimator = estimator
        self.n_features_to_select = n_features_to_select
        self.step = step
        self.verbose = verbose


class GridSearchCV(BaseEstimator):
    # [CLASS DOC]
    # HELPER: Stripped-down GridSearchCV holding an estimator and param_grid; used to
    # test reprs of nested estimators and parameter grids.
    def __init__(
        self,
        estimator,
        param_grid,
        scoring=None,
        n_jobs=None,
        iid="warn",
        refit=True,
        cv="warn",
        verbose=0,
        pre_dispatch="2*n_jobs",
        error_score="raise-deprecating",
        return_train_score=False,
    ):
        # [FUNCTION DOC]
        # HELPER: Stores all constructor arguments (old defaults such as iid="warn",
        # cv="warn", error_score="raise-deprecating") as attributes.
        self.estimator = estimator
        self.param_grid = param_grid
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.iid = iid
        self.refit = refit
        self.cv = cv
        self.verbose = verbose
        self.pre_dispatch = pre_dispatch
        self.error_score = error_score
        self.return_train_score = return_train_score


class CountVectorizer(BaseEstimator):
    # [CLASS DOC]
    # HELPER: Stripped-down CountVectorizer used to test ellipsis of a large vocabulary
    # dict.
    def __init__(
        self,
        input="content",
        encoding="utf-8",
        decode_error="strict",
        strip_accents=None,
        lowercase=True,
        preprocessor=None,
        tokenizer=None,
        stop_words=None,
        token_pattern=r"(?u)\b\w\w+\b",
        ngram_range=(1, 1),
        analyzer="word",
        max_df=1.0,
        min_df=1,
        max_features=None,
        vocabulary=None,
        binary=False,
        dtype=np.int64,
    ):
        # [FUNCTION DOC]
        # HELPER: Stores all constructor arguments (including vocabulary and dtype) as
        # attributes.
        self.input = input
        self.encoding = encoding
        self.decode_error = decode_error
        self.strip_accents = strip_accents
        self.preprocessor = preprocessor
        self.tokenizer = tokenizer
        self.analyzer = analyzer
        self.lowercase = lowercase
        self.token_pattern = token_pattern
        self.stop_words = stop_words
        self.max_df = max_df
        self.min_df = min_df
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.vocabulary = vocabulary
        self.binary = binary
        self.dtype = dtype


class Pipeline(BaseEstimator):
    # [CLASS DOC]
    # HELPER: Stripped-down Pipeline holding a list of (name, estimator) steps.
    def __init__(self, steps, memory=None):
        # [FUNCTION DOC]
        # HELPER: Stores steps and memory (default None) as attributes.
        self.steps = steps
        self.memory = memory


class SVC(BaseEstimator):
    # [CLASS DOC]
    # HELPER: Stripped-down SVC with old defaults (e.g. gamma="auto_deprecated").
    def __init__(
        self,
        C=1.0,
        kernel="rbf",
        degree=3,
        gamma="auto_deprecated",
        coef0=0.0,
        shrinking=True,
        probability=False,
        tol=1e-3,
        cache_size=200,
        class_weight=None,
        verbose=False,
        max_iter=-1,
        decision_function_shape="ovr",
        random_state=None,
    ):
        # [FUNCTION DOC]
        # HELPER: Stores all constructor arguments as attributes.
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.tol = tol
        self.C = C
        self.shrinking = shrinking
        self.probability = probability
        self.cache_size = cache_size
        self.class_weight = class_weight
        self.verbose = verbose
        self.max_iter = max_iter
        self.decision_function_shape = decision_function_shape
        self.random_state = random_state


class PCA(BaseEstimator):
    # [CLASS DOC]
    # HELPER: Stripped-down PCA.
    def __init__(
        self,
        n_components=None,
        copy=True,
        whiten=False,
        svd_solver="auto",
        tol=0.0,
        iterated_power="auto",
        random_state=None,
    ):
        # [FUNCTION DOC]
        # HELPER: Stores all constructor arguments as attributes.
        self.n_components = n_components
        self.copy = copy
        self.whiten = whiten
        self.svd_solver = svd_solver
        self.tol = tol
        self.iterated_power = iterated_power
        self.random_state = random_state


class NMF(BaseEstimator):
    # [CLASS DOC]
    # HELPER: Stripped-down NMF.
    def __init__(
        self,
        n_components=None,
        init=None,
        solver="cd",
        beta_loss="frobenius",
        tol=1e-4,
        max_iter=200,
        random_state=None,
        alpha=0.0,
        l1_ratio=0.0,
        verbose=0,
        shuffle=False,
    ):
        # [FUNCTION DOC]
        # HELPER: Stores all constructor arguments as attributes.
        self.n_components = n_components
        self.init = init
        self.solver = solver
        self.beta_loss = beta_loss
        self.tol = tol
        self.max_iter = max_iter
        self.random_state = random_state
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.verbose = verbose
        self.shuffle = shuffle


class SimpleImputer(BaseEstimator):
    # [CLASS DOC]
    # HELPER: Stripped-down SimpleImputer whose missing_values defaults to NaN.
    def __init__(
        self,
        missing_values=np.nan,
        strategy="mean",
        fill_value=None,
        verbose=0,
        copy=True,
    ):
        # [FUNCTION DOC]
        # HELPER: Stores all constructor arguments as attributes.
        self.missing_values = missing_values
        self.strategy = strategy
        self.fill_value = fill_value
        self.verbose = verbose
        self.copy = copy


@config_context(print_changed_only=False)
def test_basic():
    # [FUNCTION DOC]
    # With print_changed_only=False, checks the default LogisticRegression repr wraps
    # over several lines exactly as expected.
    # Basic pprint test
    lr = LogisticRegression()
    expected = """
LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, l1_ratio=0, max_iter=100,
                   multi_class='warn', n_jobs=None, random_state=None,
                   solver='warn', tol=0.0001, verbose=0, warm_start=False)"""

    expected = expected[1:]  # remove first \n
    assert lr.__repr__() == expected


def test_changed_only():
    # [FUNCTION DOC]
    # Checks changed-only reprs: only non-default params shown (wrapping when long), a
    # float NaN missing_values counts as the default NaN, and array-valued params of
    # LogisticRegressionCV do not raise (issue 13583).
    # Make sure the changed_only param is correctly used when True (default)
    lr = LogisticRegression(C=99)
    expected = """LogisticRegression(C=99)"""
    assert lr.__repr__() == expected

    # Check with a repr that doesn't fit on a single line
    lr = LogisticRegression(
        C=99, class_weight=0.4, fit_intercept=False, tol=1234, verbose=True
    )
    expected = """
LogisticRegression(C=99, class_weight=0.4, fit_intercept=False, tol=1234,
                   verbose=True)"""
    expected = expected[1:]  # remove first \n
    assert lr.__repr__() == expected

    imputer = SimpleImputer(missing_values=0)
    expected = """SimpleImputer(missing_values=0)"""
    assert imputer.__repr__() == expected

    # Defaults to np.nan, trying with float('NaN')
    imputer = SimpleImputer(missing_values=float("NaN"))
    expected = """SimpleImputer()"""
    assert imputer.__repr__() == expected

    # make sure array parameters don't throw error (see #13583)
    repr(
        LogisticRegressionCV(
            Cs=np.array([0.1, 1]),
            use_legacy_attributes=False,
            scoring="neg_log_loss",  # TODO(1.11): remove because it is default now
        )
    )


@config_context(print_changed_only=False)
def test_pipeline():
    # [FUNCTION DOC]
    # With print_changed_only=False, checks the full multi-line repr of a pipeline made
    # by make_pipeline (real Pipeline, including transform_input and verbose).
    # Render a pipeline object
    pipeline = make_pipeline(StandardScaler(), LogisticRegression(C=999))
    expected = """
Pipeline(memory=None,
         steps=[('standardscaler',
                 StandardScaler(copy=True, with_mean=True, with_std=True)),
                ('logisticregression',
                 LogisticRegression(C=999, class_weight=None, dual=False,
                                    fit_intercept=True, intercept_scaling=1,
                                    l1_ratio=0, max_iter=100,
                                    multi_class='warn', n_jobs=None,
                                    random_state=None, solver='warn',
                                    tol=0.0001, verbose=0, warm_start=False))],
         transform_input=None, verbose=False)"""

    expected = expected[1:]  # remove first \n
    assert pipeline.__repr__() == expected


@config_context(print_changed_only=False)
def test_deeply_nested():
    # [FUNCTION DOC]
    # With print_changed_only=False, checks the repr of seven nested RFE estimators
    # around a LogisticRegression matches the expected indentation.
    # Render a deeply nested estimator
    rfe = RFE(RFE(RFE(RFE(RFE(RFE(RFE(LogisticRegression())))))))
    expected = """
RFE(estimator=RFE(estimator=RFE(estimator=RFE(estimator=RFE(estimator=RFE(estimator=RFE(estimator=LogisticRegression(C=1.0,
                                                                                                                     class_weight=None,
                                                                                                                     dual=False,
                                                                                                                     fit_intercept=True,
                                                                                                                     intercept_scaling=1,
                                                                                                                     l1_ratio=0,
                                                                                                                     max_iter=100,
                                                                                                                     multi_class='warn',
                                                                                                                     n_jobs=None,
                                                                                                                     random_state=None,
                                                                                                                     solver='warn',
                                                                                                                     tol=0.0001,
                                                                                                                     verbose=0,
                                                                                                                     warm_start=False),
                                                                                        n_features_to_select=None,
                                                                                        step=1,
                                                                                        verbose=0),
                                                                          n_features_to_select=None,
                                                                          step=1,
                                                                          verbose=0),
                                                            n_features_to_select=None,
                                                            step=1, verbose=0),
                                              n_features_to_select=None, step=1,
                                              verbose=0),
                                n_features_to_select=None, step=1, verbose=0),
                  n_features_to_select=None, step=1, verbose=0),
    n_features_to_select=None, step=1, verbose=0)"""

    expected = expected[1:]  # remove first \n
    assert rfe.__repr__() == expected


@pytest.mark.parametrize(
    ("print_changed_only", "expected"),
    [
        (True, "RFE(estimator=RFE(...))"),
        (
            False,
            "RFE(estimator=RFE(...), n_features_to_select=None, step=1, verbose=0)",
        ),
    ],
)
def test_print_estimator_max_depth(print_changed_only, expected):
    # [FUNCTION DOC]
    # Checks _EstimatorPrettyPrinter(depth=1) abbreviates nested estimators to "..."
    # both with and without print_changed_only.
    with config_context(print_changed_only=print_changed_only):
        pp = _EstimatorPrettyPrinter(depth=1)

        rfe = RFE(RFE(RFE(RFE(RFE(LogisticRegression())))))
        assert pp.pformat(rfe) == expected


@config_context(print_changed_only=False)
def test_gridsearch():
    # [FUNCTION DOC]
    # With print_changed_only=False, checks the repr of a GridSearchCV over SVC with a
    # list-of-dicts param_grid.
    # render a gridsearch
    param_grid = [
        {"kernel": ["rbf"], "gamma": [1e-3, 1e-4], "C": [1, 10, 100, 1000]},
        {"kernel": ["linear"], "C": [1, 10, 100, 1000]},
    ]
    gs = GridSearchCV(SVC(), param_grid, cv=5)

    expected = """
GridSearchCV(cv=5, error_score='raise-deprecating',
             estimator=SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
                           decision_function_shape='ovr', degree=3,
                           gamma='auto_deprecated', kernel='rbf', max_iter=-1,
                           probability=False, random_state=None, shrinking=True,
                           tol=0.001, verbose=False),
             iid='warn', n_jobs=None,
             param_grid=[{'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001],
                          'kernel': ['rbf']},
                         {'C': [1, 10, 100, 1000], 'kernel': ['linear']}],
             pre_dispatch='2*n_jobs', refit=True, return_train_score=False,
             scoring=None, verbose=0)"""

    expected = expected[1:]  # remove first \n
    assert gs.__repr__() == expected


@config_context(print_changed_only=False)
def test_gridsearch_pipeline():
    # [FUNCTION DOC]
    # With print_changed_only=False and a compact printer, checks the repr of a
    # GridSearchCV over a PCA/SVC Pipeline whose grid contains estimators, after
    # replacing the chi2 function address with a placeholder.
    # render a pipeline inside a gridsearch
    pp = _EstimatorPrettyPrinter(compact=True, indent=1, indent_at_name=True)

    pipeline = Pipeline([("reduce_dim", PCA()), ("classify", SVC())])
    N_FEATURES_OPTIONS = [2, 4, 8]
    C_OPTIONS = [1, 10, 100, 1000]
    param_grid = [
        {
            "reduce_dim": [PCA(iterated_power=7), NMF()],
            "reduce_dim__n_components": N_FEATURES_OPTIONS,
            "classify__C": C_OPTIONS,
        },
        {
            "reduce_dim": [SelectKBest(chi2)],
            "reduce_dim__k": N_FEATURES_OPTIONS,
            "classify__C": C_OPTIONS,
        },
    ]
    gspipeline = GridSearchCV(pipeline, cv=3, n_jobs=1, param_grid=param_grid)
    expected = """
GridSearchCV(cv=3, error_score='raise-deprecating',
             estimator=Pipeline(memory=None,
                                steps=[('reduce_dim',
                                        PCA(copy=True, iterated_power='auto',
                                            n_components=None,
                                            random_state=None,
                                            svd_solver='auto', tol=0.0,
                                            whiten=False)),
                                       ('classify',
                                        SVC(C=1.0, cache_size=200,
                                            class_weight=None, coef0=0.0,
                                            decision_function_shape='ovr',
                                            degree=3, gamma='auto_deprecated',
                                            kernel='rbf', max_iter=-1,
                                            probability=False,
                                            random_state=None, shrinking=True,
                                            tol=0.001, verbose=False))]),
             iid='warn', n_jobs=1,
             param_grid=[{'classify__C': [1, 10, 100, 1000],
                          'reduce_dim': [PCA(copy=True, iterated_power=7,
                                             n_components=None,
                                             random_state=None,
                                             svd_solver='auto', tol=0.0,
                                             whiten=False),
                                         NMF(alpha=0.0, beta_loss='frobenius',
                                             init=None, l1_ratio=0.0,
                                             max_iter=200, n_components=None,
                                             random_state=None, shuffle=False,
                                             solver='cd', tol=0.0001,
                                             verbose=0)],
                          'reduce_dim__n_components': [2, 4, 8]},
                         {'classify__C': [1, 10, 100, 1000],
                          'reduce_dim': [SelectKBest(k=10,
                                                     score_func=<function chi2 at some_address>)],
                          'reduce_dim__k': [2, 4, 8]}],
             pre_dispatch='2*n_jobs', refit=True, return_train_score=False,
             scoring=None, verbose=0)"""  # noqa: E501

    expected = expected[1:]  # remove first \n
    repr_ = pp.pformat(gspipeline)
    # Remove address of '<function chi2 at 0x.....>' for reproducibility
    repr_ = re.sub("function chi2 at 0x.*>", "function chi2 at some_address>", repr_)
    assert repr_ == expected


@config_context(print_changed_only=False)
def test_n_max_elements_to_show():
    # [FUNCTION DOC]
    # With n_max_elements_to_show=30, checks a vocabulary dict and a param_grid list of
    # 30 elements are printed fully, while 31 elements are truncated with "...".
    n_max_elements_to_show = 30
    pp = _EstimatorPrettyPrinter(
        compact=True,
        indent=1,
        indent_at_name=True,
        n_max_elements_to_show=n_max_elements_to_show,
    )

    # No ellipsis
    vocabulary = {i: i for i in range(n_max_elements_to_show)}
    vectorizer = CountVectorizer(vocabulary=vocabulary)

    expected = r"""
CountVectorizer(analyzer='word', binary=False, decode_error='strict',
                dtype=<class 'numpy.int64'>, encoding='utf-8', input='content',
                lowercase=True, max_df=1.0, max_features=None, min_df=1,
                ngram_range=(1, 1), preprocessor=None, stop_words=None,
                strip_accents=None, token_pattern='(?u)\\b\\w\\w+\\b',
                tokenizer=None,
                vocabulary={0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7,
                            8: 8, 9: 9, 10: 10, 11: 11, 12: 12, 13: 13, 14: 14,
                            15: 15, 16: 16, 17: 17, 18: 18, 19: 19, 20: 20,
                            21: 21, 22: 22, 23: 23, 24: 24, 25: 25, 26: 26,
                            27: 27, 28: 28, 29: 29})"""

    expected = expected[1:]  # remove first \n
    assert pp.pformat(vectorizer) == expected

    # Now with ellipsis
    vocabulary = {i: i for i in range(n_max_elements_to_show + 1)}
    vectorizer = CountVectorizer(vocabulary=vocabulary)

    expected = r"""
CountVectorizer(analyzer='word', binary=False, decode_error='strict',
                dtype=<class 'numpy.int64'>, encoding='utf-8', input='content',
                lowercase=True, max_df=1.0, max_features=None, min_df=1,
                ngram_range=(1, 1), preprocessor=None, stop_words=None,
                strip_accents=None, token_pattern='(?u)\\b\\w\\w+\\b',
                tokenizer=None,
                vocabulary={0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7,
                            8: 8, 9: 9, 10: 10, 11: 11, 12: 12, 13: 13, 14: 14,
                            15: 15, 16: 16, 17: 17, 18: 18, 19: 19, 20: 20,
                            21: 21, 22: 22, 23: 23, 24: 24, 25: 25, 26: 26,
                            27: 27, 28: 28, 29: 29, ...})"""

    expected = expected[1:]  # remove first \n
    assert pp.pformat(vectorizer) == expected

    # Also test with lists
    param_grid = {"C": list(range(n_max_elements_to_show))}
    gs = GridSearchCV(SVC(), param_grid)
    expected = """
GridSearchCV(cv='warn', error_score='raise-deprecating',
             estimator=SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
                           decision_function_shape='ovr', degree=3,
                           gamma='auto_deprecated', kernel='rbf', max_iter=-1,
                           probability=False, random_state=None, shrinking=True,
                           tol=0.001, verbose=False),
             iid='warn', n_jobs=None,
             param_grid={'C': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
                               15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
                               27, 28, 29]},
             pre_dispatch='2*n_jobs', refit=True, return_train_score=False,
             scoring=None, verbose=0)"""

    expected = expected[1:]  # remove first \n
    assert pp.pformat(gs) == expected

    # Now with ellipsis
    param_grid = {"C": list(range(n_max_elements_to_show + 1))}
    gs = GridSearchCV(SVC(), param_grid)
    expected = """
GridSearchCV(cv='warn', error_score='raise-deprecating',
             estimator=SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
                           decision_function_shape='ovr', degree=3,
                           gamma='auto_deprecated', kernel='rbf', max_iter=-1,
                           probability=False, random_state=None, shrinking=True,
                           tol=0.001, verbose=False),
             iid='warn', n_jobs=None,
             param_grid={'C': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
                               15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
                               27, 28, 29, ...]},
             pre_dispatch='2*n_jobs', refit=True, return_train_score=False,
             scoring=None, verbose=0)"""

    expected = expected[1:]  # remove first \n
    assert pp.pformat(gs) == expected


@config_context(print_changed_only=False)
def test_bruteforce_ellipsis():
    # [FUNCTION DOC]
    # With print_changed_only=False, checks __repr__ with N_CHAR_MAX inserts the "..."
    # ellipsis between left and right parts when they are on different lines or the same
    # line, keeps the full right line, and omits the ellipsis when N_CHAR_MAX equals the
    # non-blank character count or when adding it would not shorten the repr.
    # KNOWN ISSUE: the comment before the N_CHAR_MAX = n_nonblank - 4 case says "- 10",
    # which does not match the code.
    # Check that the bruteforce ellipsis (used when the number of non-blank
    # characters exceeds N_CHAR_MAX) renders correctly.

    lr = LogisticRegression()

    # test when the left and right side of the ellipsis aren't on the same
    # line.
    expected = """
LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                   in...
                   multi_class='warn', n_jobs=None, random_state=None,
                   solver='warn', tol=0.0001, verbose=0, warm_start=False)"""

    expected = expected[1:]  # remove first \n
    assert lr.__repr__(N_CHAR_MAX=150) == expected

    # test with very small N_CHAR_MAX
    # Note that N_CHAR_MAX is not strictly enforced, but it's normal: to avoid
    # weird reprs we still keep the whole line of the right part (after the
    # ellipsis).
    expected = """
Lo...
                   solver='warn', tol=0.0001, verbose=0, warm_start=False)"""

    expected = expected[1:]  # remove first \n
    assert lr.__repr__(N_CHAR_MAX=4) == expected

    # test with N_CHAR_MAX == number of non-blank characters: In this case we
    # don't want ellipsis
    full_repr = lr.__repr__(N_CHAR_MAX=float("inf"))
    n_nonblank = len("".join(full_repr.split()))
    assert lr.__repr__(N_CHAR_MAX=n_nonblank) == full_repr
    assert "..." not in full_repr

    # test with N_CHAR_MAX == number of non-blank characters - 10: the left and
    # right side of the ellispsis are on different lines. In this case we
    # want to expend the whole line of the right side
    expected = """
LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, l1_ratio=0,...00,
                   multi_class='warn', n_jobs=None, random_state=None,
                   solver='warn', tol=0.0001, verbose=0, warm_start=False)"""
    expected = expected[1:]  # remove first \n
    assert lr.__repr__(N_CHAR_MAX=n_nonblank - 10) == expected

    # test with N_CHAR_MAX == number of non-blank characters - 10: the left and
    # right side of the ellispsis are on the same line. In this case we don't
    # want to expend the whole line of the right side, just add the ellispsis
    # between the 2 sides.
    expected = """
LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, l1_ratio=0, max...r=100,
                   multi_class='warn', n_jobs=None, random_state=None,
                   solver='warn', tol=0.0001, verbose=0, warm_start=False)"""
    expected = expected[1:]  # remove first \n
    assert lr.__repr__(N_CHAR_MAX=n_nonblank - 4) == expected

    # test with N_CHAR_MAX == number of non-blank characters - 2: the left and
    # right side of the ellispsis are on the same line, but adding the ellipsis
    # would actually make the repr longer. So we don't add the ellipsis.
    expected = """
LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, l1_ratio=0, max_iter=100,
                   multi_class='warn', n_jobs=None, random_state=None,
                   solver='warn', tol=0.0001, verbose=0, warm_start=False)"""
    expected = expected[1:]  # remove first \n
    assert lr.__repr__(N_CHAR_MAX=n_nonblank - 2) == expected


def test_builtin_prettyprinter():
    # [FUNCTION DOC]
    # Non-regression: checks the standard library PrettyPrinter can print an estimator
    # without error.
    # non regression test than ensures we can still use the builtin
    # PrettyPrinter class for estimators (as done e.g. by joblib).
    # Used to be a bug

    PrettyPrinter().pprint(LogisticRegression())


def test_kwargs_in_init():
    # [FUNCTION DOC]
    # Non-regression for issue 17206: checks changed-only and full reprs of an estimator
    # whose extra parameters arrive through **kwargs.
    # Make sure the changed_only=True mode is OK when an argument is passed as
    # kwargs.
    # Non-regression test for
    # https://github.com/scikit-learn/scikit-learn/issues/17206

    class WithKWargs(BaseEstimator):
        # [CLASS DOC]
        # HELPER: LightGBM-like estimator storing extra **kwargs parameters in
        # _other_params and exposing them through get_params/set_params.
        # Estimator with a kwargs argument. These need to hack around
        # set_params and get_params. Here we mimic what LightGBM does.
        def __init__(self, a="willchange", b="unchanged", **kwargs):
            # [FUNCTION DOC]
            # HELPER: Stores a and b, initializes _other_params to an empty dict and
            # applies kwargs via set_params.
            # Side effects: sets a, b, _other_params and one attribute per kwarg.
            self.a = a
            self.b = b
            self._other_params = {}
            self.set_params(**kwargs)

        def get_params(self, deep=True):
            # [FUNCTION DOC]
            # HELPER: Returns BaseEstimator.get_params(deep) updated with _other_params.
            # Outputs: dict of parameter names to values.
            params = super().get_params(deep=deep)
            params.update(self._other_params)
            return params

        def set_params(self, **params):
            # [FUNCTION DOC]
            # HELPER: Sets each given parameter as an attribute and records it in
            # _other_params.
            # Outputs: self. Side effects: mutates attributes and _other_params.
            for key, value in params.items():
                setattr(self, key, value)
                self._other_params[key] = value
            return self

    est = WithKWargs(a="something", c="abcd", d=None)

    expected = "WithKWargs(a='something', c='abcd', d=None)"
    assert est.__repr__() == expected

    with config_context(print_changed_only=False):
        expected = "WithKWargs(a='something', b='unchanged', c='abcd', d=None)"
        assert est.__repr__() == expected


def test_complexity_print_changed_only():
    # [FUNCTION DOC]
    # Non-regression for issue 18490: counts DummyEstimator.__repr__ calls when printing
    # a nested estimator/pipeline with print_changed_only False and True and checks the
    # counts are equal.
    # Make sure `__repr__` is called the same amount of times
    # whether `print_changed_only` is True or False
    # Non-regression test for
    # https://github.com/scikit-learn/scikit-learn/issues/18490

    class DummyEstimator(TransformerMixin, BaseEstimator):
        # [CLASS DOC]
        # HELPER: Transformer that counts its __repr__ calls in the class attribute
        # nb_times_repr_called.
        nb_times_repr_called = 0

        def __init__(self, estimator=None):
            # [FUNCTION DOC]
            # HELPER: Stores estimator (default None).
            self.estimator = estimator

        def __repr__(self):
            # [FUNCTION DOC]
            # HELPER: Increments the class-level repr counter then returns the
            # BaseEstimator repr.
            # Side effects: mutates DummyEstimator.nb_times_repr_called.
            DummyEstimator.nb_times_repr_called += 1
            return super().__repr__()

        def transform(self, X, copy=None):  # pragma: no cover
            # [FUNCTION DOC]
            # HELPER: Identity transform (never called). Outputs: X.
            return X

    estimator = DummyEstimator(
        make_pipeline(DummyEstimator(DummyEstimator()), DummyEstimator(), "passthrough")
    )
    with config_context(print_changed_only=False):
        repr(estimator)
        nb_repr_print_changed_only_false = DummyEstimator.nb_times_repr_called

    DummyEstimator.nb_times_repr_called = 0
    with config_context(print_changed_only=True):
        repr(estimator)
        nb_repr_print_changed_only_true = DummyEstimator.nb_times_repr_called

    assert nb_repr_print_changed_only_false == nb_repr_print_changed_only_true
