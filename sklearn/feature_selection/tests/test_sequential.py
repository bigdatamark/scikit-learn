# [MODULE DOC]
# Tests for SequentialFeatureSelector: validation of n_features_to_select and tol, the
#   "auto"/tol stopping rule, float fractions, correct selection on a toy problem, sparse,
#   NaN and pipeline support, unsupervised estimators, y validation, generator cv and
#   rejection of fit params without metadata routing.
import numpy as np
import pytest
from numpy.testing import assert_array_equal

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs, make_classification, make_regression
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneGroupOut, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.fixes import CSR_CONTAINERS


def test_bad_n_features_to_select():
    # [FUNCTION DOC]
    # Checks fit raises ValueError when n_features_to_select equals the number of
    #   features.
    n_features = 5
    X, y = make_regression(n_features=n_features)
    sfs = SequentialFeatureSelector(LinearRegression(), n_features_to_select=n_features)
    with pytest.raises(ValueError, match="n_features_to_select must be < n_features"):
        sfs.fit(X, y)


@pytest.mark.parametrize("direction", ("forward", "backward"))
@pytest.mark.parametrize("n_features_to_select", (1, 5, 9, "auto"))
def test_n_features_to_select(direction, n_features_to_select):
    # [FUNCTION DOC]
    # For both directions and n_features_to_select in 1, 5, 9, "auto" (half of 10
    #   features when tol is None), checks the support size, n_features_to_select_ and
    #   transformed width.
    # Make sure n_features_to_select is respected

    n_features = 10
    X, y = make_regression(n_features=n_features, random_state=0)
    sfs = SequentialFeatureSelector(
        LinearRegression(),
        n_features_to_select=n_features_to_select,
        direction=direction,
        cv=2,
    )
    sfs.fit(X, y)

    if n_features_to_select == "auto":
        n_features_to_select = n_features // 2

    assert sfs.get_support(indices=True).shape[0] == n_features_to_select
    assert sfs.n_features_to_select_ == n_features_to_select
    assert sfs.transform(X).shape[1] == n_features_to_select


@pytest.mark.parametrize("direction", ("forward", "backward"))
def test_n_features_to_select_auto(direction):
    # [FUNCTION DOC]
    # With "auto" and tol=1e-3, checks at most n_features-1 features are selected and
    #   that the support size equals n_features_to_select_, for both directions.
    # KNOWN ISSUE: the docstring mentions different values for tol but only one is
    #   tested.
    """Check the behaviour of `n_features_to_select="auto"` with different
    values for the parameter `tol`.
    """

    n_features = 10
    tol = 1e-3
    X, y = make_regression(n_features=n_features, random_state=0)
    sfs = SequentialFeatureSelector(
        LinearRegression(),
        n_features_to_select="auto",
        tol=tol,
        direction=direction,
        cv=2,
    )
    sfs.fit(X, y)

    max_features_to_select = n_features - 1

    assert sfs.get_support(indices=True).shape[0] <= max_features_to_select
    assert sfs.n_features_to_select_ <= max_features_to_select
    assert sfs.transform(X).shape[1] <= max_features_to_select
    assert sfs.get_support(indices=True).shape[0] == sfs.n_features_to_select_


@pytest.mark.parametrize("direction", ("forward", "backward"))
def test_n_features_to_select_stopping_criterion(direction):
    # [FUNCTION DOC]
    # On 50 features (10 informative) with "auto"/tol, compares 2-fold CV scores of: all
    #   features, selected features, selected plus one random unselected feature, and
    #   selected minus one random feature. Checks selection is at least as good as using
    #   all features; forward: adding does not gain more than tol and removing loses at
    #   least tol; backward: neither adding nor removing gains more than tol.
    """Check the behaviour stopping criterion for feature selection
    depending on the values of `n_features_to_select` and `tol`.

    When `direction` is `'forward'`, select a new features at random
    among those not currently selected in selector.support_,
    build a new version of the data that includes all the features
    in selector.support_ + this newly selected feature.
    And check that the cross-validation score of the model trained on
    this new dataset variant is lower than the model with
    the selected forward selected features or at least does not improve
    by more than the tol margin.

    When `direction` is `'backward'`, instead of adding a new feature
    to selector.support_, try to remove one of those selected features at random
    And check that the cross-validation score is either decreasing or
    not improving by more than the tol margin.
    """

    X, y = make_regression(n_features=50, n_informative=10, random_state=0)

    tol = 1e-3

    sfs = SequentialFeatureSelector(
        LinearRegression(),
        n_features_to_select="auto",
        tol=tol,
        direction=direction,
        cv=2,
    )
    sfs.fit(X, y)
    selected_X = sfs.transform(X)

    rng = np.random.RandomState(0)

    added_candidates = list(set(range(X.shape[1])) - set(sfs.get_support(indices=True)))
    added_X = np.hstack(
        [
            selected_X,
            (X[:, rng.choice(added_candidates)])[:, np.newaxis],
        ]
    )

    removed_candidate = rng.choice(list(range(sfs.n_features_to_select_)))
    removed_X = np.delete(selected_X, removed_candidate, axis=1)

    plain_cv_score = cross_val_score(LinearRegression(), X, y, cv=2).mean()
    sfs_cv_score = cross_val_score(LinearRegression(), selected_X, y, cv=2).mean()
    added_cv_score = cross_val_score(LinearRegression(), added_X, y, cv=2).mean()
    removed_cv_score = cross_val_score(LinearRegression(), removed_X, y, cv=2).mean()

    assert sfs_cv_score >= plain_cv_score

    if direction == "forward":
        assert (sfs_cv_score - added_cv_score) <= tol
        assert (sfs_cv_score - removed_cv_score) >= tol
    else:
        assert (added_cv_score - sfs_cv_score) <= tol
        assert (removed_cv_score - sfs_cv_score) <= tol


@pytest.mark.parametrize("direction", ("forward", "backward"))
@pytest.mark.parametrize(
    "n_features_to_select, expected",
    (
        (0.1, 1),
        (1.0, 10),
        (0.5, 5),
    ),
)
def test_n_features_to_select_float(direction, n_features_to_select, expected):
    # [FUNCTION DOC]
    # Checks float fractions 0.1, 0.5, 1.0 of 10 features give n_features_to_select_ 1,
    #   5, 10.
    # Test passing a float as n_features_to_select
    X, y = make_regression(n_features=10)
    sfs = SequentialFeatureSelector(
        LinearRegression(),
        n_features_to_select=n_features_to_select,
        direction=direction,
        cv=2,
    )
    sfs.fit(X, y)
    assert sfs.n_features_to_select_ == expected


@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("direction", ("forward", "backward"))
@pytest.mark.parametrize(
    "n_features_to_select, expected_selected_features",
    [
        (2, [0, 2]),  # f1 is dropped since it has no predictive power
        (1, [2]),  # f2 is more predictive than f0 so it's kept
    ],
)
def test_sanity(seed, direction, n_features_to_select, expected_selected_features):
    # [FUNCTION DOC]
    # For 10 seeds and both directions, with y = 3*f0 - 10*f2 (f1 irrelevant), checks
    #   selecting 2 features yields [0, 2] and selecting 1 yields [2].
    # Basic sanity check: 3 features, only f0 and f2 are correlated with the
    # target, f2 having a stronger correlation than f0. We expect f1 to be
    # dropped, and f2 to always be selected.

    rng = np.random.RandomState(seed)
    n_samples = 100
    X = rng.randn(n_samples, 3)
    y = 3 * X[:, 0] - 10 * X[:, 2]

    sfs = SequentialFeatureSelector(
        LinearRegression(),
        n_features_to_select=n_features_to_select,
        direction=direction,
        cv=2,
    )
    sfs.fit(X, y)
    assert_array_equal(sfs.get_support(indices=True), expected_selected_features)


@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_sparse_support(csr_container):
    # [FUNCTION DOC]
    # Smoke test: fit and transform succeed on CSR input with "auto".
    # Make sure sparse data is supported

    X, y = make_regression(n_features=10)
    X = csr_container(X)
    sfs = SequentialFeatureSelector(
        LinearRegression(), n_features_to_select="auto", cv=2
    )
    sfs.fit(X, y)
    sfs.transform(X)


def test_nan_support():
    # [FUNCTION DOC]
    # Checks fitting works on data with ~50% NaNs when the estimator supports NaN
    #   (HistGradientBoostingRegressor) and raises "Input X contains NaN" ValueError with
    #   LinearRegression.
    # Make sure nans are OK if the underlying estimator supports nans

    rng = np.random.RandomState(0)
    n_samples, n_features = 40, 4
    X, y = make_regression(n_samples, n_features, random_state=0)
    nan_mask = rng.randint(0, 2, size=(n_samples, n_features), dtype=bool)
    X[nan_mask] = np.nan
    sfs = SequentialFeatureSelector(
        HistGradientBoostingRegressor(), n_features_to_select="auto", cv=2
    )
    sfs.fit(X, y)
    sfs.transform(X)

    with pytest.raises(ValueError, match="Input X contains NaN"):
        # LinearRegression does not support nans
        SequentialFeatureSelector(
            LinearRegression(), n_features_to_select="auto", cv=2
        ).fit(X, y)


def test_pipeline_support():
    # [FUNCTION DOC]
    # Smoke test: a pipeline can be the wrapped estimator, and the selector can be a
    #   pipeline step; fit and transform succeed both ways.
    # Make sure that pipelines can be passed into SFS and that SFS can be
    # passed into a pipeline

    n_samples, n_features = 50, 3
    X, y = make_regression(n_samples, n_features, random_state=0)

    # pipeline in SFS
    pipe = make_pipeline(StandardScaler(), LinearRegression())
    sfs = SequentialFeatureSelector(pipe, n_features_to_select="auto", cv=2)
    sfs.fit(X, y)
    sfs.transform(X)

    # SFS in pipeline
    sfs = SequentialFeatureSelector(
        LinearRegression(), n_features_to_select="auto", cv=2
    )
    pipe = make_pipeline(StandardScaler(), sfs)
    pipe.fit(X, y)
    pipe.transform(X)


@pytest.mark.parametrize("n_features_to_select", (2, 3))
def test_unsupervised_model_fit(n_features_to_select):
    # [FUNCTION DOC]
    # Checks fitting with KMeans and no y works and yields the requested number of
    #   features.
    # Make sure that models without classification labels are not being
    # validated

    X, y = make_blobs(n_features=4)
    sfs = SequentialFeatureSelector(
        KMeans(n_init=1),
        n_features_to_select=n_features_to_select,
    )
    sfs.fit(X)
    assert sfs.transform(X).shape[1] == n_features_to_select


@pytest.mark.parametrize("y", ("no_validation", 1j, 99.9, np.nan, 3))
def test_no_y_validation_model_fit(y):
    # [FUNCTION DOC]
    # Checks that scalar or string y values (string, complex, float, NaN, int) are
    #   rejected by fit with TypeError or ValueError when using KMeans.
    # Make sure that other non-conventional y labels are not accepted

    X, clusters = make_blobs(n_features=6)
    sfs = SequentialFeatureSelector(
        KMeans(),
        n_features_to_select=3,
    )

    with pytest.raises((TypeError, ValueError)):
        sfs.fit(X, y)


def test_forward_neg_tol_error():
    # [FUNCTION DOC]
    # Checks negative tol with forward direction raises "tol must be strictly positive".
    """Check that we raise an error when tol<0 and direction='forward'"""
    X, y = make_regression(n_features=10, random_state=0)
    sfs = SequentialFeatureSelector(
        LinearRegression(),
        n_features_to_select="auto",
        direction="forward",
        tol=-1e-3,
    )

    with pytest.raises(ValueError, match="tol must be strictly positive"):
        sfs.fit(X, y)


def test_backward_neg_tol():
    # [FUNCTION DOC]
    # Non-regression: backward selection with negative tol removes some but not all
    #   features and the refit training R^2 on the reduced data is lower than on all
    #   features.
    """Check that SequentialFeatureSelector works negative tol

    non-regression test for #25525
    """
    X, y = make_regression(n_features=10, random_state=0)
    lr = LinearRegression()
    initial_score = lr.fit(X, y).score(X, y)

    sfs = SequentialFeatureSelector(
        lr,
        n_features_to_select="auto",
        direction="backward",
        tol=-1e-3,
    )
    Xr = sfs.fit_transform(X, y)
    new_score = lr.fit(Xr, y).score(Xr, y)

    assert 0 < sfs.get_support().sum() < X.shape[1]
    assert new_score < initial_score


def test_cv_generator_support():
    # [FUNCTION DOC]
    # Non-regression: passing a one-shot split generator (LeaveOneGroupOut splits) as cv
    #   does not raise during fit.
    """Check that no exception raised when cv is generator

    non-regression test for #25957
    """
    X, y = make_classification(random_state=0)

    groups = np.zeros_like(y, dtype=int)
    groups[y.size // 2 :] = 1

    cv = LeaveOneGroupOut()
    splits = cv.split(X, y, groups=groups)

    knc = KNeighborsClassifier(n_neighbors=5)

    sfs = SequentialFeatureSelector(knc, n_features_to_select=5, cv=splits)
    sfs.fit(X, y)


def test_fit_rejects_params_with_no_routing_enabled():
    # [FUNCTION DOC]
    # Checks passing sample_weight to fit without metadata routing raises ValueError
    #   ("is only supported if").
    X, y = make_classification(random_state=42)
    est = LinearRegression()
    sfs = SequentialFeatureSelector(estimator=est)

    with pytest.raises(ValueError, match="is only supported if"):
        sfs.fit(X, y, sample_weight=np.ones_like(y))
