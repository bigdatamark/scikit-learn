# [MODULE DOC]
# Tests for the successive halving searches HalvingGridSearchCV and
# HalvingRandomSearchCV: iteration/resource schedules (min/max resources,
# aggressive elimination, factor), custom resource parameters, candidate sampling,
# input validation, the private _SubsampleMetaSplitter and _top_k helpers,
# cv_results_ consistency with the tournament logic, group support, NaN-score
# handling and best-index selection. Defines two fast dummy classifiers used as
# base estimators. Importing enables the experimental halving search API.
from math import ceil

import numpy as np
import pytest
from scipy.stats import expon, norm, randint

from sklearn.datasets import make_classification
from sklearn.dummy import DummyClassifier
from sklearn.experimental import enable_halving_search_cv  # noqa: F401
from sklearn.model_selection import (
    GroupKFold,
    GroupShuffleSplit,
    HalvingGridSearchCV,
    HalvingRandomSearchCV,
    KFold,
    LeaveOneGroupOut,
    LeavePGroupsOut,
    ShuffleSplit,
    StratifiedKFold,
    StratifiedShuffleSplit,
)
from sklearn.model_selection._search_successive_halving import (
    _SubsampleMetaSplitter,
    _top_k,
)
from sklearn.model_selection.tests.test_search import (
    check_cv_results_array_types,
    check_cv_results_keys,
)
from sklearn.svm import SVC, LinearSVC


class FastClassifier(DummyClassifier):
    # [CLASS DOC]
    # HELPER: DummyClassifier that accepts arbitrary keyword parameters named a..z so
    # they can be used in parameter grids at negligible fit cost.
    # _parameter_constraints extends the parent's with "no_validation" for each letter.
    """Dummy classifier that accepts parameters a, b, ... z.

    These parameter don't affect the predictions and are useful for fast
    grid searching."""

    # update the constraints such that we accept all parameters from a to z
    _parameter_constraints: dict = {
        **DummyClassifier._parameter_constraints,
        **{chr(key): "no_validation" for key in range(ord("a"), ord("z") + 1)},
    }

    def __init__(
        self, strategy="stratified", random_state=None, constant=None, **kwargs
    ):
        # [FUNCTION DOC]
        # HELPER: Forwards strategy, random_state and constant to DummyClassifier.
        # Inputs: strategy, random_state, constant as in DummyClassifier; **kwargs - any
        # letter parameters, accepted and discarded.
        # Side effects: sets the three parent attributes only.
        # KNOWN ISSUE: kwargs are dropped, so letter params are never stored at init.
        super().__init__(
            strategy=strategy, random_state=random_state, constant=constant
        )

    def get_params(self, deep=False):
        # [FUNCTION DOC]
        # HELPER: Returns the parent params plus a placeholder value "whatever" for every
        # letter a..z so the search accepts these names in set_params.
        # Inputs: deep - forwarded to the parent get_params.
        # Outputs: dict of parameter names to values.
        # KNOWN ISSUE: letter values reported are always "whatever", even after set_params
        # assigned a different value, so clone/get_params do not round-trip them.
        params = super().get_params(deep=deep)
        for char in range(ord("a"), ord("z") + 1):
            params[chr(char)] = "whatever"
        return params


class SometimesFailClassifier(DummyClassifier):
    # [CLASS DOC]
    # HELPER: DummyClassifier that can be told to raise during fit or predict, plus
    # extra dummy params n_estimators (usable as a resource) and a, to produce failing
    # candidates (NaN scores) in searches.
    def __init__(
        self,
        strategy="stratified",
        random_state=None,
        constant=None,
        n_estimators=10,
        fail_fit=False,
        fail_predict=False,
        a=0,
    ):
        # [FUNCTION DOC]
        # HELPER: Stores fail_fit, fail_predict, n_estimators and a as attributes, then
        # forwards strategy, random_state and constant to DummyClassifier.
        # Inputs: fail_fit / fail_predict - bools (default False); n_estimators - int
        # (default 10); a - any (default 0); the rest as in DummyClassifier.
        self.fail_fit = fail_fit
        self.fail_predict = fail_predict
        self.n_estimators = n_estimators
        self.a = a

        super().__init__(
            strategy=strategy, random_state=random_state, constant=constant
        )

    def fit(self, X, y):
        # [FUNCTION DOC]
        # HELPER: Fits like DummyClassifier unless fail_fit is set.
        # Inputs: X, y - training data. Outputs: self.
        # Exceptions: generic Exception("fitting failed") when fail_fit is True.
        if self.fail_fit:
            raise Exception("fitting failed")
        return super().fit(X, y)

    def predict(self, X):
        # [FUNCTION DOC]
        # HELPER: Predicts like DummyClassifier unless fail_predict is set.
        # Inputs: X - samples. Outputs: array of predicted labels.
        # Exceptions: generic Exception("predict failed") when fail_predict is True.
        if self.fail_predict:
            raise Exception("predict failed")
        return super().predict(X)


@pytest.mark.filterwarnings("ignore::sklearn.exceptions.FitFailedWarning")
@pytest.mark.filterwarnings("ignore:Scoring failed:UserWarning")
@pytest.mark.filterwarnings("ignore:One or more of the:UserWarning")
@pytest.mark.parametrize("HalvingSearch", (HalvingGridSearchCV, HalvingRandomSearchCV))
@pytest.mark.parametrize("fail_at", ("fit", "predict"))
def test_nan_handling(HalvingSearch, fail_at):
    # [FUNCTION DOC]
    # Verifies that candidates failing in fit or predict get NaN mean scores, that the
    # best params never come from a failing candidate, and that all NaN-scored
    # candidates share one rank which is the worst rank. Uses n_estimators as the
    # resource with a grid over fail_<stage> and a. Warnings about failures ignored.
    """Check the selection of the best scores in presence of failure represented by
    NaN values."""
    n_samples = 1_000
    X, y = make_classification(n_samples=n_samples, random_state=0)

    search = HalvingSearch(
        SometimesFailClassifier(),
        {f"fail_{fail_at}": [False, True], "a": range(3)},
        resource="n_estimators",
        max_resources=6,
        min_resources=1,
        factor=2,
    )

    search.fit(X, y)

    # estimators that failed during fit/predict should always rank lower
    # than ones where the fit/predict succeeded
    assert not search.best_params_[f"fail_{fail_at}"]
    scores = search.cv_results_["mean_test_score"]
    ranks = search.cv_results_["rank_test_score"]

    # some scores should be NaN
    assert np.isnan(scores).any()

    unique_nan_ranks = np.unique(ranks[np.isnan(scores)])
    # all NaN scores should have the same rank
    assert unique_nan_ranks.shape[0] == 1
    # NaNs should have the lowest rank
    assert (unique_nan_ranks[0] >= ranks).all()


@pytest.mark.parametrize("Est", (HalvingGridSearchCV, HalvingRandomSearchCV))
@pytest.mark.parametrize(
    (
        "aggressive_elimination,"
        "max_resources,"
        "expected_n_iterations,"
        "expected_n_required_iterations,"
        "expected_n_possible_iterations,"
        "expected_n_remaining_candidates,"
        "expected_n_candidates,"
        "expected_n_resources,"
    ),
    [
        # notice how it loops at the beginning
        # also, the number of candidates evaluated at the last iteration is
        # <= factor
        (True, "limited", 4, 4, 3, 1, [60, 20, 7, 3], [20, 20, 60, 180]),
        # no aggressive elimination: we end up with less iterations, and
        # the number of candidates at the last iter is > factor, which isn't
        # ideal
        (False, "limited", 3, 4, 3, 3, [60, 20, 7], [20, 60, 180]),
        #  # When the amount of resource isn't limited, aggressive_elimination
        #  # has no effect. Here the default min_resources='exhaust' will take
        #  # over.
        (True, "unlimited", 4, 4, 4, 1, [60, 20, 7, 3], [37, 111, 333, 999]),
        (False, "unlimited", 4, 4, 4, 1, [60, 20, 7, 3], [37, 111, 333, 999]),
    ],
)
def test_aggressive_elimination(
    Est,
    aggressive_elimination,
    max_resources,
    expected_n_iterations,
    expected_n_required_iterations,
    expected_n_possible_iterations,
    expected_n_remaining_candidates,
    expected_n_candidates,
    expected_n_resources,
):
    # [FUNCTION DOC]
    # Verifies the aggressive_elimination parameter: with limited max_resources (180)
    # and aggressive elimination, early iterations repeat the minimum resource until
    # the final round has at most factor candidates; without it fewer iterations run.
    # With unlimited resources the flag has no effect. Checks n_iterations_,
    # n_required_iterations_, n_possible_iterations_, n_resources_, n_candidates_,
    # n_remaining_candidates_ against parametrized expectations, and that the
    # remaining count equals the ceiling of last candidates count over factor.
    # Random search is configured with 60 candidates and min_resources "exhaust".
    # Test the aggressive_elimination parameter.

    n_samples = 1000
    X, y = make_classification(n_samples=n_samples, random_state=0)
    param_grid = {"a": ("l1", "l2"), "b": list(range(30))}
    base_estimator = FastClassifier()

    if max_resources == "limited":
        max_resources = 180
    else:
        max_resources = n_samples

    sh = Est(
        base_estimator,
        param_grid,
        aggressive_elimination=aggressive_elimination,
        max_resources=max_resources,
        factor=3,
    )
    sh.set_params(verbose=True)  # just for test coverage

    if Est is HalvingRandomSearchCV:
        # same number of candidates as with the grid
        sh.set_params(n_candidates=2 * 30, min_resources="exhaust")

    sh.fit(X, y)

    assert sh.n_iterations_ == expected_n_iterations
    assert sh.n_required_iterations_ == expected_n_required_iterations
    assert sh.n_possible_iterations_ == expected_n_possible_iterations
    assert sh.n_resources_ == expected_n_resources
    assert sh.n_candidates_ == expected_n_candidates
    assert sh.n_remaining_candidates_ == expected_n_remaining_candidates
    assert ceil(sh.n_candidates_[-1] / sh.factor) == sh.n_remaining_candidates_


@pytest.mark.parametrize("Est", (HalvingGridSearchCV, HalvingRandomSearchCV))
@pytest.mark.parametrize(
    (
        "min_resources,"
        "max_resources,"
        "expected_n_iterations,"
        "expected_n_possible_iterations,"
        "expected_n_resources,"
    ),
    [
        # with enough resources
        ("smallest", "auto", 2, 4, [20, 60]),
        # with enough resources but min_resources set manually
        (50, "auto", 2, 3, [50, 150]),
        # without enough resources, only one iteration can be done
        ("smallest", 30, 1, 1, [20]),
        # with exhaust: use as much resources as possible at the last iter
        ("exhaust", "auto", 2, 2, [333, 999]),
        ("exhaust", 1000, 2, 2, [333, 999]),
        ("exhaust", 999, 2, 2, [333, 999]),
        ("exhaust", 600, 2, 2, [200, 600]),
        ("exhaust", 599, 2, 2, [199, 597]),
        ("exhaust", 300, 2, 2, [100, 300]),
        ("exhaust", 60, 2, 2, [20, 60]),
        ("exhaust", 50, 1, 1, [20]),
        ("exhaust", 20, 1, 1, [20]),
    ],
)
def test_min_max_resources(
    Est,
    min_resources,
    max_resources,
    expected_n_iterations,
    expected_n_possible_iterations,
    expected_n_resources,
):
    # [FUNCTION DOC]
    # Verifies how min_resources ("smallest", int, "exhaust") and max_resources
    # ("auto" or int) determine the number of iterations, possible iterations and the
    # resources per iteration for 6 candidates with factor 3; with "exhaust" the
    # possible and actual iteration counts must equal the resource schedule length.
    # Test the min_resources and max_resources parameters, and how they affect
    # the number of resources used at each iteration
    n_samples = 1000
    X, y = make_classification(n_samples=n_samples, random_state=0)
    param_grid = {"a": [1, 2], "b": [1, 2, 3]}
    base_estimator = FastClassifier()

    sh = Est(
        base_estimator,
        param_grid,
        factor=3,
        min_resources=min_resources,
        max_resources=max_resources,
    )
    if Est is HalvingRandomSearchCV:
        sh.set_params(n_candidates=6)  # same number as with the grid

    sh.fit(X, y)

    expected_n_required_iterations = 2  # given 6 combinations and factor = 3
    assert sh.n_iterations_ == expected_n_iterations
    assert sh.n_required_iterations_ == expected_n_required_iterations
    assert sh.n_possible_iterations_ == expected_n_possible_iterations
    assert sh.n_resources_ == expected_n_resources
    if min_resources == "exhaust":
        assert sh.n_possible_iterations_ == sh.n_iterations_ == len(sh.n_resources_)


@pytest.mark.parametrize("Est", (HalvingRandomSearchCV, HalvingGridSearchCV))
@pytest.mark.parametrize(
    "max_resources, n_iterations, n_possible_iterations",
    [
        ("auto", 5, 9),  # all resources are used
        (1024, 5, 9),
        (700, 5, 8),
        (512, 5, 8),
        (511, 5, 7),
        (32, 4, 4),
        (31, 3, 3),
        (16, 3, 3),
        (4, 1, 1),  # max_resources == min_resources, only one iteration is
        # possible
    ],
)
def test_n_iterations(Est, max_resources, n_iterations, n_possible_iterations):
    # [FUNCTION DOC]
    # Verifies the number of actual and possible iterations for 20 candidates,
    # factor 2 and min_resources 4 across several max_resources values; required
    # iterations must always be 5.
    # test the number of actual iterations that were run depending on
    # max_resources

    n_samples = 1024
    X, y = make_classification(n_samples=n_samples, random_state=1)
    param_grid = {"a": [1, 2], "b": list(range(10))}
    base_estimator = FastClassifier()
    factor = 2

    sh = Est(
        base_estimator,
        param_grid,
        cv=2,
        factor=factor,
        max_resources=max_resources,
        min_resources=4,
    )
    if Est is HalvingRandomSearchCV:
        sh.set_params(n_candidates=20)  # same as for HalvingGridSearchCV
    sh.fit(X, y)
    assert sh.n_required_iterations_ == 5
    assert sh.n_iterations_ == n_iterations
    assert sh.n_possible_iterations_ == n_possible_iterations


@pytest.mark.parametrize("Est", (HalvingRandomSearchCV, HalvingGridSearchCV))
def test_resource_parameter(Est):
    # [FUNCTION DOC]
    # Verifies using a custom estimator parameter (c) as the resource: resources are
    # 1, 3, 9 and each cv_results_ row reports the resource equal to the c value in
    # params and param_c. Also checks ValueError for an unsupported resource name and
    # for a resource that is also part of the searched grid.
    # Test the resource parameter

    n_samples = 1000
    X, y = make_classification(n_samples=n_samples, random_state=0)
    param_grid = {"a": [1, 2], "b": list(range(10))}
    base_estimator = FastClassifier()
    sh = Est(base_estimator, param_grid, cv=2, resource="c", max_resources=10, factor=3)
    sh.fit(X, y)
    assert set(sh.n_resources_) == set([1, 3, 9])
    for r_i, params, param_c in zip(
        sh.cv_results_["n_resources"],
        sh.cv_results_["params"],
        sh.cv_results_["param_c"],
    ):
        assert r_i == params["c"] == param_c

    with pytest.raises(
        ValueError, match="Cannot use resource=1234 which is not supported "
    ):
        sh = HalvingGridSearchCV(
            base_estimator, param_grid, cv=2, resource="1234", max_resources=10
        )
        sh.fit(X, y)

    with pytest.raises(
        ValueError,
        match=(
            "Cannot use parameter c as the resource since it is part "
            "of the searched parameters."
        ),
    ):
        param_grid = {"a": [1, 2], "b": [1, 2], "c": [1, 3]}
        sh = HalvingGridSearchCV(
            base_estimator, param_grid, cv=2, resource="c", max_resources=10
        )
        sh.fit(X, y)


@pytest.mark.parametrize(
    "max_resources, n_candidates, expected_n_candidates",
    [
        (512, "exhaust", 128),  # generate exactly as much as needed
        (32, "exhaust", 8),
        (32, 8, 8),
        (32, 7, 7),  # ask for less than what we could
        (32, 9, 9),  # ask for more than 'reasonable'
    ],
)
def test_random_search(max_resources, n_candidates, expected_n_candidates):
    # [FUNCTION DOC]
    # Verifies HalvingRandomSearchCV generates the expected number of initial
    # candidates for integer or "exhaust" n_candidates, and that "exhaust" makes the
    # last iteration use exactly max_resources.
    # Test random search and make sure the number of generated candidates is
    # as expected

    n_samples = 1024
    X, y = make_classification(n_samples=n_samples, random_state=0)
    param_grid = {"a": norm, "b": norm}
    base_estimator = FastClassifier()
    sh = HalvingRandomSearchCV(
        base_estimator,
        param_grid,
        n_candidates=n_candidates,
        cv=2,
        max_resources=max_resources,
        factor=2,
        min_resources=4,
    )
    sh.fit(X, y)
    assert sh.n_candidates_[0] == expected_n_candidates
    if n_candidates == "exhaust":
        # Make sure 'exhaust' makes the last iteration use as much resources as
        # we can
        assert sh.n_resources_[-1] == max_resources


@pytest.mark.parametrize(
    "param_distributions, expected_n_candidates",
    [
        ({"a": [1, 2]}, 2),  # all lists, sample less than n_candidates
        ({"a": randint(1, 3)}, 10),  # not all list, respect n_candidates
    ],
)
def test_random_search_discrete_distributions(
    param_distributions, expected_n_candidates
):
    # [FUNCTION DOC]
    # Verifies that with all-list distributions the random search samples at most
    # the grid size (2) even when n_candidates is 10, while with a non-list
    # distribution it samples exactly n_candidates.
    # Make sure random search samples the appropriate number of candidates when
    # we ask for more than what's possible. How many parameters are sampled
    # depends whether the distributions are 'all lists' or not (see
    # ParameterSampler for details). This is somewhat redundant with the checks
    # in ParameterSampler but interaction bugs were discovered during
    # development of SH

    n_samples = 1024
    X, y = make_classification(n_samples=n_samples, random_state=0)
    base_estimator = FastClassifier()
    sh = HalvingRandomSearchCV(base_estimator, param_distributions, n_candidates=10)
    sh.fit(X, y)
    assert sh.n_candidates_[0] == expected_n_candidates


@pytest.mark.parametrize("Est", (HalvingGridSearchCV, HalvingRandomSearchCV))
@pytest.mark.parametrize(
    "params, expected_error_message",
    [
        (
            {"resource": "not_a_parameter"},
            "Cannot use resource=not_a_parameter which is not supported",
        ),
        (
            {"resource": "a", "max_resources": 100},
            "Cannot use parameter a as the resource since it is part of",
        ),
        (
            {"max_resources": "auto", "resource": "b"},
            "resource can only be 'n_samples' when max_resources='auto'",
        ),
        (
            {"min_resources": 15, "max_resources": 14},
            "min_resources_=15 is greater than max_resources_=14",
        ),
        ({"cv": KFold(shuffle=True)}, "must yield consistent folds"),
        ({"cv": ShuffleSplit()}, "must yield consistent folds"),
    ],
)
def test_input_errors(Est, params, expected_error_message):
    # [FUNCTION DOC]
    # Verifies that fit raises ValueError with the expected message for invalid
    # resource names, resource in the grid, non-sample resource with "auto"
    # max_resources, min_resources above max_resources, and CV splitters yielding
    # inconsistent folds (shuffled KFold, ShuffleSplit).
    base_estimator = FastClassifier()
    param_grid = {"a": [1]}
    X, y = make_classification(100)

    sh = Est(base_estimator, param_grid, **params)

    with pytest.raises(ValueError, match=expected_error_message):
        sh.fit(X, y)


@pytest.mark.parametrize(
    "params, expected_error_message",
    [
        (
            {"n_candidates": "exhaust", "min_resources": "exhaust"},
            "cannot be both set to 'exhaust'",
        ),
    ],
)
def test_input_errors_randomized(params, expected_error_message):
    # [FUNCTION DOC]
    # Verifies HalvingRandomSearchCV.fit raises ValueError when n_candidates and
    # min_resources are both "exhaust".
    # tests specific to HalvingRandomSearchCV

    base_estimator = FastClassifier()
    param_grid = {"a": [1]}
    X, y = make_classification(100)

    sh = HalvingRandomSearchCV(base_estimator, param_grid, **params)

    with pytest.raises(ValueError, match=expected_error_message):
        sh.fit(X, y)


@pytest.mark.parametrize(
    "fraction, subsample_test, expected_train_size, expected_test_size",
    [
        (0.5, True, 40, 10),
        (0.5, False, 40, 20),
        (0.2, True, 16, 4),
        (0.2, False, 16, 20),
    ],
)
def test_subsample_splitter_shapes(
    fraction, subsample_test, expected_train_size, expected_test_size
):
    # [FUNCTION DOC]
    # Verifies train/test sizes from _SubsampleMetaSplitter over 5-fold KFold on 100
    # samples: train is subsampled by fraction; test is subsampled only when
    # subsample_test is True (then train+test equals fraction of n_samples),
    # otherwise it keeps the full fold size.
    # Make sure splits returned by SubsampleMetaSplitter are of appropriate
    # size

    n_samples = 100
    X, y = make_classification(n_samples)
    cv = _SubsampleMetaSplitter(
        base_cv=KFold(5),
        fraction=fraction,
        subsample_test=subsample_test,
        random_state=None,
    )

    for train, test in cv.split(X, y):
        assert train.shape[0] == expected_train_size
        assert test.shape[0] == expected_test_size
        if subsample_test:
            assert train.shape[0] + test.shape[0] == int(n_samples * fraction)
        else:
            assert test.shape[0] == n_samples // cv.base_cv.get_n_splits()


@pytest.mark.parametrize("subsample_test", (True, False))
def test_subsample_splitter_determinism(subsample_test):
    # [FUNCTION DOC]
    # Verifies determinism of _SubsampleMetaSplitter across two split calls with
    # random_state None: training sets differ; test sets differ when subsampled and
    # are identical (indices and data) when not subsampled.
    # Make sure _SubsampleMetaSplitter is consistent across calls to split():
    # - we're OK having training sets differ (they're always sampled with a
    #   different fraction anyway)
    # - when we don't subsample the test set, we want it to be always the same.
    #   This check is the most important. This is ensured by the determinism
    #   of the base_cv.

    # Note: we could force both train and test splits to be always the same if
    # we drew an int seed in _SubsampleMetaSplitter.__init__

    n_samples = 100
    X, y = make_classification(n_samples)
    cv = _SubsampleMetaSplitter(
        base_cv=KFold(5), fraction=0.5, subsample_test=subsample_test, random_state=None
    )

    folds_a = list(cv.split(X, y, groups=None))
    folds_b = list(cv.split(X, y, groups=None))

    for (train_a, test_a), (train_b, test_b) in zip(folds_a, folds_b):
        assert not np.all(train_a == train_b)

        if subsample_test:
            assert not np.all(test_a == test_b)
        else:
            assert np.all(test_a == test_b)
            assert np.all(X[test_a] == X[test_b])


@pytest.mark.parametrize(
    "k, itr, expected",
    [
        (1, 0, ["c"]),
        (2, 0, ["a", "c"]),
        (4, 0, ["d", "b", "a", "c"]),
        (10, 0, ["d", "b", "a", "c"]),
        (1, 1, ["e"]),
        (2, 1, ["f", "e"]),
        (10, 1, ["f", "e"]),
        (1, 2, ["i"]),
        (10, 2, ["g", "h", "i"]),
    ],
)
def test_top_k(k, itr, expected):
    # [FUNCTION DOC]
    # Verifies _top_k returns the k best params (by mean_test_score, ordered from
    # lowest to highest score) among entries of a given iteration, truncating when k
    # exceeds the number of entries, using a hand-built results dict.
    results = {  # this isn't a 'real world' result dict
        "iter": [0, 0, 0, 0, 1, 1, 2, 2, 2],
        "mean_test_score": [4, 3, 5, 1, 11, 10, 5, 6, 9],
        "params": ["a", "b", "c", "d", "e", "f", "g", "h", "i"],
    }
    got = _top_k(results, k=k, itr=itr)
    assert np.all(got == expected)


@pytest.mark.parametrize("Est", (HalvingRandomSearchCV, HalvingGridSearchCV))
def test_cv_results(Est):
    # [FUNCTION DOC]
    # Verifies cv_results_ follows the tournament logic: iter and n_resources are
    # ndarrays; with random tie-free scores, candidates discarded at one iteration are
    # never evaluated later, the number kept matches n_candidates_, every discarded
    # score is below every kept score, and best_params_ comes from the last iteration
    # even though a higher score exists earlier. Requires pandas (skipped otherwise).
    # Randomness: seeded RandomState drives the scorer.
    # test that the cv_results_ matches correctly the logic of the
    # tournament: in particular that the candidates continued in each
    # successive iteration are those that were best in the previous iteration
    pd = pytest.importorskip("pandas")

    rng = np.random.RandomState(0)

    n_samples = 1000
    X, y = make_classification(n_samples=n_samples, random_state=0)
    param_grid = {"a": ("l1", "l2"), "b": list(range(30))}
    base_estimator = FastClassifier()

    # generate random scores: we want to avoid ties, which would otherwise
    # mess with the ordering and make testing harder
    def scorer(est, X, y):
        # [FUNCTION DOC]
        # HELPER: Scorer ignoring its inputs and returning a uniform random float from
        # the enclosing rng, to avoid score ties.
        return rng.rand()

    sh = Est(base_estimator, param_grid, factor=2, scoring=scorer)
    if Est is HalvingRandomSearchCV:
        # same number of candidates as with the grid
        sh.set_params(n_candidates=2 * 30, min_resources="exhaust")

    sh.fit(X, y)

    # non-regression check for
    # https://github.com/scikit-learn/scikit-learn/issues/19203
    assert isinstance(sh.cv_results_["iter"], np.ndarray)
    assert isinstance(sh.cv_results_["n_resources"], np.ndarray)

    cv_results_df = pd.DataFrame(sh.cv_results_)

    # just make sure we don't have ties
    assert len(cv_results_df["mean_test_score"].unique()) == len(cv_results_df)

    cv_results_df["params_str"] = cv_results_df["params"].apply(str)
    table = cv_results_df.pivot(
        index="params_str", columns="iter", values="mean_test_score"
    )

    # table looks like something like this:
    # iter                    0      1       2        3   4   5
    # params_str
    # {'a': 'l2', 'b': 23} 0.75    NaN     NaN      NaN NaN NaN
    # {'a': 'l1', 'b': 30} 0.90  0.875     NaN      NaN NaN NaN
    # {'a': 'l1', 'b': 0}  0.75    NaN     NaN      NaN NaN NaN
    # {'a': 'l2', 'b': 3}  0.85  0.925  0.9125  0.90625 NaN NaN
    # {'a': 'l1', 'b': 5}  0.80    NaN     NaN      NaN NaN NaN
    # ...

    # where a NaN indicates that the candidate wasn't evaluated at a given
    # iteration, because it wasn't part of the top-K at some previous
    # iteration. We here make sure that candidates that aren't in the top-k at
    # any given iteration are indeed not evaluated at the subsequent
    # iterations.
    nan_mask = pd.isna(table)
    n_iter = sh.n_iterations_
    for it in range(n_iter - 1):
        already_discarded_mask = nan_mask[it]

        # make sure that if a candidate is already discarded, we don't evaluate
        # it later
        assert (
            already_discarded_mask & nan_mask[it + 1] == already_discarded_mask
        ).all()

        # make sure that the number of discarded candidate is correct
        discarded_now_mask = ~already_discarded_mask & nan_mask[it + 1]
        kept_mask = ~already_discarded_mask & ~discarded_now_mask
        assert kept_mask.sum() == sh.n_candidates_[it + 1]

        # make sure that all discarded candidates have a lower score than the
        # kept candidates
        discarded_max_score = table[it].where(discarded_now_mask).max()
        kept_min_score = table[it].where(kept_mask).min()
        assert discarded_max_score < kept_min_score

    # We now make sure that the best candidate is chosen only from the last
    # iteration.
    # We also make sure this is true even if there were higher scores in
    # earlier rounds (this isn't generally the case, but worth ensuring it's
    # possible).

    last_iter = cv_results_df["iter"].max()
    idx_best_last_iter = cv_results_df[cv_results_df["iter"] == last_iter][
        "mean_test_score"
    ].idxmax()
    idx_best_all_iters = cv_results_df["mean_test_score"].idxmax()

    assert sh.best_params_ == cv_results_df.iloc[idx_best_last_iter]["params"]
    assert (
        cv_results_df.iloc[idx_best_last_iter]["mean_test_score"]
        < cv_results_df.iloc[idx_best_all_iters]["mean_test_score"]
    )
    assert (
        cv_results_df.iloc[idx_best_last_iter]["params"]
        != cv_results_df.iloc[idx_best_all_iters]["params"]
    )


@pytest.mark.parametrize("Est", (HalvingGridSearchCV, HalvingRandomSearchCV))
def test_base_estimator_inputs(Est):
    # [FUNCTION DOC]
    # Verifies base estimators receive the right number of samples and params at each
    # iteration: a bookkeeping subclass records fit/predict sample counts and
    # set_params calls; after deduplicating per split, unique sample counts must match
    # n_resources_, their counts n_candidates_, and the recorded params and sample
    # counts must match cv_results_ rows. Requires pandas.
    # make sure that the base estimators are passed the correct parameters and
    # number of samples at each iteration.
    pd = pytest.importorskip("pandas")

    passed_n_samples_fit = []
    passed_n_samples_predict = []
    passed_params = []

    class FastClassifierBookKeeping(FastClassifier):
        # [CLASS DOC]
        # HELPER: FastClassifier that appends observed sample counts and params to lists
        # of the enclosing test (closure mutation) on fit, predict and set_params.
        def fit(self, X, y):
            # [FUNCTION DOC]
            # HELPER: Records the number of training samples, then fits normally. Returns self.
            passed_n_samples_fit.append(X.shape[0])
            return super().fit(X, y)

        def predict(self, X):
            # [FUNCTION DOC]
            # HELPER: Records the number of predicted samples, then predicts normally.
            passed_n_samples_predict.append(X.shape[0])
            return super().predict(X)

        def set_params(self, **params):
            # [FUNCTION DOC]
            # HELPER: Records the params dict, then delegates to the parent set_params.
            passed_params.append(params)
            return super().set_params(**params)

    n_samples = 1024
    n_splits = 2
    X, y = make_classification(n_samples=n_samples, random_state=0)
    param_grid = {"a": ("l1", "l2"), "b": list(range(30))}
    base_estimator = FastClassifierBookKeeping()

    sh = Est(
        base_estimator,
        param_grid,
        factor=2,
        cv=n_splits,
        return_train_score=False,
        refit=False,
    )
    if Est is HalvingRandomSearchCV:
        # same number of candidates as with the grid
        sh.set_params(n_candidates=2 * 30, min_resources="exhaust")

    sh.fit(X, y)

    assert len(passed_n_samples_fit) == len(passed_n_samples_predict)
    passed_n_samples = [
        x + y for (x, y) in zip(passed_n_samples_fit, passed_n_samples_predict)
    ]

    # Lists are of length n_splits * n_iter * n_candidates_at_i.
    # Each chunk of size n_splits corresponds to the n_splits folds for the
    # same candidate at the same iteration, so they contain equal values. We
    # subsample such that the lists are of length n_iter * n_candidates_at_it
    passed_n_samples = passed_n_samples[::n_splits]
    passed_params = passed_params[::n_splits]

    cv_results_df = pd.DataFrame(sh.cv_results_)

    assert len(passed_params) == len(passed_n_samples) == len(cv_results_df)

    uniques, counts = np.unique(passed_n_samples, return_counts=True)
    assert (sh.n_resources_ == uniques).all()
    assert (sh.n_candidates_ == counts).all()

    assert (cv_results_df["params"] == passed_params).all()
    assert (cv_results_df["n_resources"] == passed_n_samples).all()


@pytest.mark.parametrize("Est", (HalvingGridSearchCV, HalvingRandomSearchCV))
def test_groups_support(Est):
    # [FUNCTION DOC]
    # Verifies group-based CV splitters raise ValueError when groups is None and work
    # when groups are given, and non-group splitters work without groups.
    # Check if ValueError (when groups is None) propagates to
    # HalvingGridSearchCV and HalvingRandomSearchCV
    # And also check if groups is correctly passed to the cv object
    rng = np.random.RandomState(0)

    X, y = make_classification(n_samples=50, n_classes=2, random_state=0)
    groups = rng.randint(0, 3, 50)

    clf = LinearSVC(random_state=0)
    grid = {"C": [1]}

    group_cvs = [
        LeaveOneGroupOut(),
        LeavePGroupsOut(2),
        GroupKFold(n_splits=3),
        GroupShuffleSplit(random_state=0),
    ]
    error_msg = "The 'groups' parameter should not be None."
    for cv in group_cvs:
        gs = Est(clf, grid, cv=cv, random_state=0)
        with pytest.raises(ValueError, match=error_msg):
            gs.fit(X, y)
        gs.fit(X, y, groups=groups)

    non_group_cvs = [StratifiedKFold(), StratifiedShuffleSplit(random_state=0)]
    for cv in non_group_cvs:
        gs = Est(clf, grid, cv=cv)
        # Should not raise an error
        gs.fit(X, y)


@pytest.mark.parametrize("SearchCV", [HalvingRandomSearchCV, HalvingGridSearchCV])
def test_min_resources_null(SearchCV):
    # [FUNCTION DOC]
    # Verifies fit raises ValueError mentioning min_resources_=0 when X is empty and
    # min_resources is "smallest".
    """Check that we raise an error if the minimum resources is set to 0."""
    base_estimator = FastClassifier()
    param_grid = {"a": [1]}
    X = np.empty(0).reshape(0, 3)

    search = SearchCV(base_estimator, param_grid, min_resources="smallest")

    err_msg = "min_resources_=0: you might have passed an empty dataset X."
    with pytest.raises(ValueError, match=err_msg):
        search.fit(X, [])


@pytest.mark.parametrize("SearchCV", [HalvingGridSearchCV, HalvingRandomSearchCV])
def test_select_best_index(SearchCV):
    # [FUNCTION DOC]
    # Verifies _select_best_index (called unbound with None for self and refit) picks
    # the highest score among the last iteration only (index 8), not the global max.
    """Check the selection strategy of the halving search."""
    results = {  # this isn't a 'real world' result dict
        "iter": np.array([0, 0, 0, 0, 1, 1, 2, 2, 2]),
        "mean_test_score": np.array([4, 3, 5, 1, 11, 10, 5, 6, 9]),
        "params": np.array(["a", "b", "c", "d", "e", "f", "g", "h", "i"]),
    }

    # we expect the index of 'i'
    best_index = SearchCV._select_best_index(None, None, results)
    assert best_index == 8


def test_halving_random_search_list_of_dicts():
    # [FUNCTION DOC]
    # Verifies HalvingRandomSearchCV with a list of parameter-distribution dicts:
    # cv_results_ has the expected keys and dtype kinds, and param columns are masked
    # exactly for parameters absent from the dict the candidate came from (rbf vs
    # poly kernel).
    """Check the behaviour of the `HalvingRandomSearchCV` with `param_distribution`
    being a list of dictionary.
    """
    X, y = make_classification(n_samples=150, n_features=4, random_state=42)

    params = [
        {"kernel": ["rbf"], "C": expon(scale=10), "gamma": expon(scale=0.1)},
        {"kernel": ["poly"], "degree": [2, 3]},
    ]
    param_keys = (
        "param_C",
        "param_degree",
        "param_gamma",
        "param_kernel",
    )
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
    extra_keys = ("n_resources", "iter")

    search = HalvingRandomSearchCV(
        SVC(), cv=3, param_distributions=params, return_train_score=True, random_state=0
    )
    search.fit(X, y)
    n_candidates = sum(search.n_candidates_)
    cv_results = search.cv_results_
    # Check results structure
    check_cv_results_keys(cv_results, param_keys, score_keys, n_candidates, extra_keys)
    expected_cv_results_kinds = {
        "param_C": "f",
        "param_degree": "i",
        "param_gamma": "f",
        "param_kernel": "O",
    }
    check_cv_results_array_types(
        search, param_keys, score_keys, expected_cv_results_kinds
    )

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
