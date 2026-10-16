# [MODULE DOC]
# Tests for monotonicity constraints (monotonic_cst) in single decision/extra trees
# and the forest ensembles built on them: prediction monotonicity on shifted
# inputs, error messages for unsupported settings, and structural checks that node
# values respect monotonic bounds, plus self-tests of those structural checkers.
import numpy as np
import pytest

from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.tree import (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
    ExtraTreeClassifier,
    ExtraTreeRegressor,
)
from sklearn.utils._testing import assert_allclose
from sklearn.utils.fixes import CSC_CONTAINERS

TREE_CLASSIFIER_CLASSES = [DecisionTreeClassifier, ExtraTreeClassifier]
TREE_REGRESSOR_CLASSES = [DecisionTreeRegressor, ExtraTreeRegressor]
TREE_BASED_CLASSIFIER_CLASSES = TREE_CLASSIFIER_CLASSES + [
    RandomForestClassifier,
    ExtraTreesClassifier,
]
TREE_BASED_REGRESSOR_CLASSES = TREE_REGRESSOR_CLASSES + [
    RandomForestRegressor,
    ExtraTreesRegressor,
]


@pytest.mark.parametrize("TreeClassifier", TREE_BASED_CLASSIFIER_CLASSES)
@pytest.mark.parametrize(
    "sparse_splitter, with_missing",
    [
        (False, False),
        (True, False),
        (False, True),
    ],
    ids=["dense-without-missing", "sparse-without-missing", "dense-with-missing"],
)
@pytest.mark.parametrize("depth_first_builder", (True, False))
@pytest.mark.parametrize("csc_container", CSC_CONTAINERS)
def test_monotonic_constraints_classifications(
    TreeClassifier,
    sparse_splitter,
    depth_first_builder,
    with_missing,
    global_random_seed,
    csc_container,
):
    # [FUNCTION DOC]
    # Fits a binary classifier on 900 of 1000 make_classification samples with
    # increasing constraint on feature 0 and decreasing on feature 1, using either the
    # depth-first builder or best-first (max_leaf_nodes set), optionally with ~50% NaNs
    # or a sparse CSC training matrix. Checks probabilities lie in [0, 1] and sum to 1,
    # and that shifting feature 0 (resp. 1) by +/-10 on the test set moves the
    # positive-class probability in the constrained direction.
    # Inputs: parametrized estimator class, sparse/missing flags, builder flag, CSC
    # container and global_random_seed fixture. Forests use 5 estimators.
    n_samples = 1000
    n_samples_train = 900
    X, y = make_classification(
        n_samples=n_samples,
        n_classes=2,
        n_features=5,
        n_informative=5,
        n_redundant=0,
        random_state=global_random_seed,
    )
    X_train, y_train = X[:n_samples_train], y[:n_samples_train]
    X_test, _ = X[n_samples_train:], y[n_samples_train:]

    X_test_0incr, X_test_0decr = np.copy(X_test), np.copy(X_test)
    X_test_1incr, X_test_1decr = np.copy(X_test), np.copy(X_test)
    X_test_0incr[:, 0] += 10
    X_test_0decr[:, 0] -= 10
    X_test_1incr[:, 1] += 10
    X_test_1decr[:, 1] -= 10
    monotonic_cst = np.zeros(X.shape[1])
    monotonic_cst[0] = 1
    monotonic_cst[1] = -1

    if depth_first_builder:
        est = TreeClassifier(max_depth=None, monotonic_cst=monotonic_cst)
    else:
        est = TreeClassifier(
            max_depth=None,
            monotonic_cst=monotonic_cst,
            max_leaf_nodes=n_samples_train,
        )
    if hasattr(est, "random_state"):
        est.set_params(random_state=global_random_seed)
    if hasattr(est, "n_estimators"):
        est.set_params(n_estimators=5)
    if with_missing:
        generator = np.random.default_rng(seed=global_random_seed)
        mask = generator.choice(2, size=X_train.shape).astype(bool)
        X_train[mask] = np.nan
    if sparse_splitter:
        X_train = csc_container(X_train)
    est.fit(X_train, y_train)
    proba_test = est.predict_proba(X_test)

    assert np.logical_and(proba_test >= 0.0, proba_test <= 1.0).all(), (
        "Probability should always be in [0, 1] range."
    )
    assert_allclose(proba_test.sum(axis=1), 1.0)

    # Monotonic increase constraint, it applies to the positive class
    assert np.all(est.predict_proba(X_test_0incr)[:, 1] >= proba_test[:, 1])
    assert np.all(est.predict_proba(X_test_0decr)[:, 1] <= proba_test[:, 1])

    # Monotonic decrease constraint, it applies to the positive class
    assert np.all(est.predict_proba(X_test_1incr)[:, 1] <= proba_test[:, 1])
    assert np.all(est.predict_proba(X_test_1decr)[:, 1] >= proba_test[:, 1])


@pytest.mark.parametrize("TreeRegressor", TREE_BASED_REGRESSOR_CLASSES)
@pytest.mark.parametrize(
    "sparse_splitter, with_missing",
    [
        (False, False),
        (True, False),
        (False, True),
    ],
    ids=["dense-without-missing", "sparse-without-missing", "dense-with-missing"],
)
@pytest.mark.parametrize("depth_first_builder", (True, False))
@pytest.mark.parametrize("criterion", ("absolute_error", "squared_error"))
@pytest.mark.parametrize("csc_container", CSC_CONTAINERS)
def test_monotonic_constraints_regressions(
    TreeRegressor,
    sparse_splitter,
    depth_first_builder,
    with_missing,
    criterion,
    global_random_seed,
    csc_container,
):
    # [FUNCTION DOC]
    # Regression analogue of the classification test: fits on make_regression data with
    # +1 on feature 0 and -1 on feature 1 for absolute_error and squared_error; checks
    # predictions do not decrease when feature 0 is increased by 10 and do not increase
    # when feature 1 is increased by 10. Best-first variant uses max_depth=8.
    n_samples = 1000
    n_samples_train = 900
    # Build a regression task using 5 informative features
    X, y = make_regression(
        n_samples=n_samples,
        n_features=5,
        n_informative=5,
        random_state=global_random_seed,
    )
    train = np.arange(n_samples_train)
    test = np.arange(n_samples_train, n_samples)
    X_train = X[train]
    y_train = y[train]
    X_test = np.copy(X[test])
    X_test_incr = np.copy(X_test)
    X_test_decr = np.copy(X_test)
    X_test_incr[:, 0] += 10
    X_test_decr[:, 1] += 10
    monotonic_cst = np.zeros(X.shape[1])
    monotonic_cst[0] = 1
    monotonic_cst[1] = -1

    if depth_first_builder:
        est = TreeRegressor(
            max_depth=None,
            monotonic_cst=monotonic_cst,
            criterion=criterion,
        )
    else:
        est = TreeRegressor(
            max_depth=8,
            monotonic_cst=monotonic_cst,
            criterion=criterion,
            max_leaf_nodes=n_samples_train,
        )
    if hasattr(est, "random_state"):
        est.set_params(random_state=global_random_seed)
    if hasattr(est, "n_estimators"):
        est.set_params(n_estimators=5)
    if with_missing:
        generator = np.random.default_rng(seed=global_random_seed)
        mask = generator.choice(2, size=X_train.shape).astype(bool)
        X_train[mask] = np.nan
    if sparse_splitter:
        X_train = csc_container(X_train)
    est.fit(X_train, y_train)
    y = est.predict(X_test)
    # Monotonic increase constraint
    y_incr = est.predict(X_test_incr)
    # y_incr should always be greater than y
    assert np.all(y_incr >= y)

    # Monotonic decrease constraint
    y_decr = est.predict(X_test_decr)
    # y_decr should always be lower than y
    assert np.all(y_decr <= y)


@pytest.mark.parametrize("TreeClassifier", TREE_BASED_CLASSIFIER_CLASSES)
def test_multiclass_raises(TreeClassifier):
    # [FUNCTION DOC]
    # Checks that fitting a classifier with monotonic_cst on a 3-class target raises
    # ValueError about multiclass classification.
    X, y = make_classification(
        n_samples=100, n_features=5, n_classes=3, n_informative=3, random_state=0
    )
    y[0] = 0
    monotonic_cst = np.zeros(X.shape[1])
    monotonic_cst[0] = -1
    monotonic_cst[1] = 1
    est = TreeClassifier(max_depth=None, monotonic_cst=monotonic_cst, random_state=0)

    msg = "Monotonicity constraints are not supported with multiclass classification"
    with pytest.raises(ValueError, match=msg):
        est.fit(X, y)


@pytest.mark.parametrize("TreeClassifier", TREE_BASED_CLASSIFIER_CLASSES)
def test_multiple_output_raises(TreeClassifier):
    # [FUNCTION DOC]
    # Checks that fitting a classifier with monotonic_cst on a 2d (multi-output) target
    # raises ValueError about multiple output (raised before the shape mismatch between
    # the 2-element constraint and 5 features is checked).
    X = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]
    y = [[1, 0, 1, 0, 1], [1, 0, 1, 0, 1]]

    est = TreeClassifier(
        max_depth=None, monotonic_cst=np.array([-1, 1]), random_state=0
    )
    msg = "Monotonicity constraints are not supported with multiple output"
    with pytest.raises(ValueError, match=msg):
        est.fit(X, y)


@pytest.mark.parametrize("TreeClassifier", TREE_BASED_CLASSIFIER_CLASSES)
def test_bad_monotonic_cst_raises(TreeClassifier):
    # [FUNCTION DOC]
    # Checks ValueError messages for monotonic_cst with the wrong length, with values
    # outside {-1, 0, 1}, and with a non-integer value (0.8 shown in the message).
    X = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]
    y = [1, 0, 1, 0, 1]

    msg = "monotonic_cst has shape 3 but the input data X has 2 features."
    est = TreeClassifier(
        max_depth=None, monotonic_cst=np.array([-1, 1, 0]), random_state=0
    )
    with pytest.raises(ValueError, match=msg):
        est.fit(X, y)

    msg = "monotonic_cst must be None or an array-like of -1, 0 or 1."
    est = TreeClassifier(
        max_depth=None, monotonic_cst=np.array([-2, 2]), random_state=0
    )
    with pytest.raises(ValueError, match=msg):
        est.fit(X, y)

    est = TreeClassifier(
        max_depth=None, monotonic_cst=np.array([-1, 0.8]), random_state=0
    )
    with pytest.raises(ValueError, match=msg + "(.*)0.8]"):
        est.fit(X, y)


def assert_1d_reg_tree_children_monotonic_bounded(tree_, monotonic_sign):
    # [FUNCTION DOC]
    # HELPER: Asserts that, for a 1-feature regression tree, each split node's children
    # values are ordered according to monotonic_sign (1 increasing, -1 decreasing), and
    # that the inner grandchildren (right child of left child, left child of right child,
    # when those children are splits) are bounded by the mean of the two children values.
    # Inputs: tree_ - fitted Tree object; monotonic_sign - 1 or -1 (0 checks nothing).
    # Exceptions: AssertionError on violation.
    values = tree_.value
    for i in range(tree_.node_count):
        if tree_.children_left[i] > i and tree_.children_right[i] > i:
            # Check monotonicity on children
            i_left = tree_.children_left[i]
            i_right = tree_.children_right[i]
            if monotonic_sign == 1:
                assert values[i_left] <= values[i_right]
            elif monotonic_sign == -1:
                assert values[i_left] >= values[i_right]
            val_middle = (values[i_left] + values[i_right]) / 2
            # Check bounds on grand-children, filtering out leaf nodes
            if tree_.feature[i_left] >= 0:
                i_left_right = tree_.children_right[i_left]
                if monotonic_sign == 1:
                    assert values[i_left_right] <= val_middle
                elif monotonic_sign == -1:
                    assert values[i_left_right] >= val_middle
            if tree_.feature[i_right] >= 0:
                i_right_left = tree_.children_left[i_right]
                if monotonic_sign == 1:
                    assert val_middle <= values[i_right_left]
                elif monotonic_sign == -1:
                    assert val_middle >= values[i_right_left]


def test_assert_1d_reg_tree_children_monotonic_bounded():
    # [FUNCTION DOC]
    # Verifies the 1d checker detects violations: an unconstrained tree fit on a sine
    # wave fails the check for both +1 and -1.
    X = np.linspace(-1, 1, 7).reshape(-1, 1)
    y = np.sin(2 * np.pi * X.ravel())

    reg = DecisionTreeRegressor(max_depth=None, random_state=0).fit(X, y)

    with pytest.raises(AssertionError):
        assert_1d_reg_tree_children_monotonic_bounded(reg.tree_, 1)

    with pytest.raises(AssertionError):
        assert_1d_reg_tree_children_monotonic_bounded(reg.tree_, -1)


def assert_1d_reg_monotonic(clf, monotonic_sign, min_x, max_x, n_steps):
    # [FUNCTION DOC]
    # HELPER: Predicts on n_steps evenly spaced points in [min_x, max_x] and asserts
    # consecutive predictions are non-decreasing (sign 1) or non-increasing (sign -1).
    # Inputs: clf - fitted 1-feature regressor; monotonic_sign; min_x, max_x - floats;
    # n_steps - int. Exceptions: AssertionError on violation.
    X_grid = np.linspace(min_x, max_x, n_steps).reshape(-1, 1)
    y_pred_grid = clf.predict(X_grid)
    if monotonic_sign == 1:
        assert (np.diff(y_pred_grid) >= 0.0).all()
    elif monotonic_sign == -1:
        assert (np.diff(y_pred_grid) <= 0.0).all()


@pytest.mark.parametrize("TreeRegressor", TREE_REGRESSOR_CLASSES)
def test_1d_opposite_monotonicity_cst_data(TreeRegressor):
    # [FUNCTION DOC]
    # Checks that fitting increasing data with a decreasing constraint (and vice versa)
    # yields a single-leaf tree predicting the target mean, which is 0.
    # Check that positive monotonic data with negative monotonic constraint
    # yield constant predictions, equal to the average of target values
    X = np.linspace(-2, 2, 10).reshape(-1, 1)
    y = X.ravel()
    clf = TreeRegressor(monotonic_cst=[-1])
    clf.fit(X, y)
    assert clf.tree_.node_count == 1
    assert clf.tree_.value[0] == 0.0

    # Swap monotonicity
    clf = TreeRegressor(monotonic_cst=[1])
    clf.fit(X, -y)
    assert clf.tree_.node_count == 1
    assert clf.tree_.value[0] == 0.0


@pytest.mark.parametrize("TreeRegressor", TREE_REGRESSOR_CLASSES)
@pytest.mark.parametrize("with_missing", (True, False))
@pytest.mark.parametrize("monotonic_sign", (-1, 1))
@pytest.mark.parametrize("depth_first_builder", (True, False))
@pytest.mark.parametrize("criterion", ("absolute_error", "squared_error"))
def test_1d_tree_nodes_values(
    TreeRegressor,
    with_missing,
    monotonic_sign,
    depth_first_builder,
    criterion,
    global_random_seed,
):
    # [FUNCTION DOC]
    # Fits a 1-feature regressor on uniform random data under a monotonic constraint
    # (both builders, both criteria, optional 20% NaNs) and verifies both the structural
    # bounds via assert_1d_reg_tree_children_monotonic_bounded and monotone predictions
    # on a 100-point grid over the observed feature range.
    # Adaptation from test_nodes_values in test_monotonic_constraints.py
    # in sklearn.ensemble._hist_gradient_boosting
    # Build a single tree with only one feature, and make sure the node
    # values respect the monotonicity constraints.

    # Considering the following tree with a monotonic +1 constraint, we
    # should have:
    #
    #       root
    #      /    \
    #     a      b
    #    / \    / \
    #   c   d  e   f
    #
    #        a <=  root  <= b
    # c <= d <= (a + b) / 2 <= e <= f

    rng = np.random.RandomState(global_random_seed)
    n_samples = 1000
    n_features = 1
    X = rng.rand(n_samples, n_features)
    y = rng.rand(n_samples)

    if depth_first_builder:
        # No max_leaf_nodes, default depth first tree builder
        clf = TreeRegressor(
            monotonic_cst=[monotonic_sign],
            criterion=criterion,
            random_state=global_random_seed,
        )
    else:
        # max_leaf_nodes triggers best first tree builder
        clf = TreeRegressor(
            monotonic_cst=[monotonic_sign],
            max_leaf_nodes=n_samples,
            criterion=criterion,
            random_state=global_random_seed,
        )
    if with_missing:
        generator = np.random.default_rng(seed=global_random_seed)
        mask = generator.choice(2, size=X.shape, p=[0.8, 0.2]).astype(bool)
        X[mask] = np.nan
    clf.fit(X, y)

    assert_1d_reg_tree_children_monotonic_bounded(clf.tree_, monotonic_sign)
    min_x, max_x = np.nanmin(X), np.nanmax(X)
    assert_1d_reg_monotonic(clf, monotonic_sign, min_x, max_x, 100)


def assert_nd_reg_tree_children_monotonic_bounded(tree_, monotonic_cst):
    # [FUNCTION DOC]
    # HELPER: Traverses node ids in increasing order (parents precede children) carrying
    # per-node lower/upper bounds initialised to -inf/inf. Asserts each node value lies
    # within its bounds (compared in float32 to tolerate rounding). For split nodes on an
    # unconstrained feature, children inherit the parent bounds; for +1 features asserts
    # left value <= right value and bounds the left child above and the right child below
    # by the mean of the two children values; -1 features are symmetric.
    # Inputs: tree_ - fitted single-output regression Tree; monotonic_cst - sequence of
    # -1/0/1 per feature. Exceptions: AssertionError on violation; ValueError for an
    # invalid constraint value.
    upper_bound = np.full(tree_.node_count, np.inf)
    lower_bound = np.full(tree_.node_count, -np.inf)
    for i in range(tree_.node_count):
        feature = tree_.feature[i]
        node_value = tree_.value[i][0][0]  # unpack value from nx1x1 array
        # While building the tree, the computed middle value is slightly
        # different from the average of the siblings values, because
        # sum_right / weighted_n_right
        # is slightly different from the value of the right sibling.
        # This can cause a discrepancy up to numerical noise when clipping,
        # which is resolved by comparing with some loss of precision.
        assert np.float32(node_value) <= np.float32(upper_bound[i])
        assert np.float32(node_value) >= np.float32(lower_bound[i])

        if feature < 0:
            # Leaf: nothing to do
            continue

        # Split node: check and update bounds for the children.
        i_left = tree_.children_left[i]
        i_right = tree_.children_right[i]
        # unpack value from nx1x1 array (middle_value after clipping)
        middle_value = (tree_.value[i_left][0][0] + tree_.value[i_right][0][0]) / 2

        if monotonic_cst[feature] == 0:
            # Feature without monotonicity constraint: propagate bounds
            # down the tree to both children.
            # Otherwise, with 2 features and a monotonic increase constraint
            # (encoded by +1) on feature 0, the following tree can be accepted,
            # although it does not respect the monotonic increase constraint:
            #
            #                      X[0] <= 0
            #                      value = 100
            #                     /            \
            #          X[0] <= -1                X[1] <= 0
            #          value = 50                value = 150
            #        /            \             /            \
            #    leaf           leaf           leaf          leaf
            #    value = 25     value = 75     value = 50    value = 250

            lower_bound[i_left] = lower_bound[i]
            upper_bound[i_left] = upper_bound[i]
            lower_bound[i_right] = lower_bound[i]
            upper_bound[i_right] = upper_bound[i]

        elif monotonic_cst[feature] == 1:
            # Feature with constraint: check monotonicity
            assert tree_.value[i_left] <= tree_.value[i_right]

            # Propagate bounds down the tree to both children.
            lower_bound[i_left] = lower_bound[i]
            upper_bound[i_left] = middle_value
            lower_bound[i_right] = middle_value
            upper_bound[i_right] = upper_bound[i]

        elif monotonic_cst[feature] == -1:
            # Feature with constraint: check monotonicity
            assert tree_.value[i_left] >= tree_.value[i_right]

            # Update and propagate bounds down the tree to both children.
            lower_bound[i_left] = middle_value
            upper_bound[i_left] = upper_bound[i]
            lower_bound[i_right] = lower_bound[i]
            upper_bound[i_right] = middle_value

        else:  # pragma: no cover
            raise ValueError(f"monotonic_cst[{feature}]={monotonic_cst[feature]}")


def test_assert_nd_reg_tree_children_monotonic_bounded():
    # [FUNCTION DOC]
    # Verifies the n-d checker: on a sine fit it fails for +1 and -1 but passes for 0,
    # and on cubic data it fails for the constraint opposite to the data trend in both
    # sign directions.
    # Check that assert_nd_reg_tree_children_monotonic_bounded can detect
    # non-monotonic tree predictions.
    X = np.linspace(0, 2 * np.pi, 30).reshape(-1, 1)
    y = np.sin(X).ravel()
    reg = DecisionTreeRegressor(max_depth=None, random_state=0).fit(X, y)

    with pytest.raises(AssertionError):
        assert_nd_reg_tree_children_monotonic_bounded(reg.tree_, [1])

    with pytest.raises(AssertionError):
        assert_nd_reg_tree_children_monotonic_bounded(reg.tree_, [-1])

    assert_nd_reg_tree_children_monotonic_bounded(reg.tree_, [0])

    # Check that assert_nd_reg_tree_children_monotonic_bounded raises
    # when the data (and therefore the model) is naturally monotonic in the
    # opposite direction.
    X = np.linspace(-5, 5, 5).reshape(-1, 1)
    y = X.ravel() ** 3
    reg = DecisionTreeRegressor(max_depth=None, random_state=0).fit(X, y)

    with pytest.raises(AssertionError):
        assert_nd_reg_tree_children_monotonic_bounded(reg.tree_, [-1])

    # For completeness, check that the converse holds when swapping the sign.
    reg = DecisionTreeRegressor(max_depth=None, random_state=0).fit(X, -y)

    with pytest.raises(AssertionError):
        assert_nd_reg_tree_children_monotonic_bounded(reg.tree_, [1])


@pytest.mark.parametrize("TreeRegressor", TREE_REGRESSOR_CLASSES)
@pytest.mark.parametrize("with_missing", (True, False))
@pytest.mark.parametrize("monotonic_sign", (-1, 1))
@pytest.mark.parametrize("depth_first_builder", (True, False))
@pytest.mark.parametrize("criterion", ("absolute_error", "squared_error"))
def test_nd_tree_nodes_values(
    TreeRegressor,
    with_missing,
    monotonic_sign,
    depth_first_builder,
    criterion,
    global_random_seed,
):
    # [FUNCTION DOC]
    # Fits a 2-feature regressor with a constraint on feature 0 only (both builders, both
    # criteria, optional 20% NaNs) and checks node values with
    # assert_nd_reg_tree_children_monotonic_bounded.
    # Build tree with several features, and make sure the nodes
    # values respect the monotonicity constraints.

    # Considering the following tree with a monotonic increase constraint on X[0],
    # we should have:
    #
    #            root
    #           X[0]<=t
    #          /       \
    #         a         b
    #     X[0]<=u   X[1]<=v
    #    /       \   /     \
    #   c        d  e       f
    #
    # i)   a <= root <= b
    # ii)  c <= a <= d <= (a+b)/2
    # iii) (a+b)/2 <= min(e,f)
    # For iii) we check that each node value is within the proper lower and
    # upper bounds.

    rng = np.random.RandomState(global_random_seed)
    n_samples = 1000
    n_features = 2
    monotonic_cst = [monotonic_sign, 0]
    X = rng.rand(n_samples, n_features)
    y = rng.rand(n_samples)

    if depth_first_builder:
        # No max_leaf_nodes, default depth first tree builder
        clf = TreeRegressor(
            monotonic_cst=monotonic_cst,
            criterion=criterion,
            random_state=global_random_seed,
        )
    else:
        # max_leaf_nodes triggers best first tree builder
        clf = TreeRegressor(
            monotonic_cst=monotonic_cst,
            max_leaf_nodes=n_samples,
            criterion=criterion,
            random_state=global_random_seed,
        )
    if with_missing:
        generator = np.random.default_rng(seed=global_random_seed)
        mask = generator.choice(2, size=X.shape, p=[0.8, 0.2]).astype(bool)
        X[mask] = np.nan
    clf.fit(X, y)
    assert_nd_reg_tree_children_monotonic_bounded(clf.tree_, monotonic_cst)
