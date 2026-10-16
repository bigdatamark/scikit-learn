# [MODULE DOC]
# Tests for TreeGrower of histogram gradient boosting: step-by-step growth on a
# synthetic pre-binned dataset, conversion to TreePredictor, stopping criteria
# (min_samples_leaf, max_depth), input/parameter validation, missing-value routing,
# categorical splits (and equivalence with one-hot encoding) and interaction
# constraints. n_threads is the effective OpenMP thread count.
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from pytest import approx

from sklearn.ensemble._hist_gradient_boosting.binning import _BinMapper
from sklearn.ensemble._hist_gradient_boosting.common import (
    G_H_DTYPE,
    X_BINNED_DTYPE,
    X_BITSET_INNER_DTYPE,
    X_DTYPE,
    Y_DTYPE,
)
from sklearn.ensemble._hist_gradient_boosting.grower import TreeGrower
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils._openmp_helpers import _openmp_effective_n_threads

n_threads = _openmp_effective_n_threads()


def _make_training_data(n_bins=256, constant_hessian=True):
    # [FUNCTION DOC]
    # HELPER: Generates 10000 pre-binned samples with 2 features (uniform ints in
    # [0, n_bins - 2], Fortran-ordered) and a target from a 3-leaf ground-truth tree.
    # Gradients equal the target (square loss around an initial prediction of 0).
    # Inputs: n_bins - int (default 256); constant_hessian - bool (default True): if
    # True hessians has shape (1,), else one per sample (all ones).
    # Outputs: tuple (X_binned uint8 (10000, 2), gradients (10000,), hessians).
    # Side effects: none; uses a fixed RandomState(42).
    rng = np.random.RandomState(42)
    n_samples = 10000

    # Generate some test data directly binned so as to test the grower code
    # independently of the binning logic.
    X_binned = rng.randint(0, n_bins - 1, size=(n_samples, 2), dtype=X_BINNED_DTYPE)
    X_binned = np.asfortranarray(X_binned)

    def true_decision_function(input_features):
        # [FUNCTION DOC]
        # HELPER: Ground-truth tree: returns -1 if feature 0 <= n_bins // 2, otherwise -1 if
        # feature 1 <= n_bins // 3 else 1. Input: one row of 2 binned values.
        """Ground truth decision function

        This is a very simple yet asymmetric decision tree. Therefore the
        grower code should have no trouble recovering the decision function
        from 10000 training samples.
        """
        if input_features[0] <= n_bins // 2:
            return -1
        else:
            return -1 if input_features[1] <= n_bins // 3 else 1

    target = np.array([true_decision_function(x) for x in X_binned], dtype=Y_DTYPE)

    # Assume a square loss applied to an initial model that always predicts 0
    # (hardcoded for this test):
    all_gradients = target.astype(G_H_DTYPE)
    shape_hessians = 1 if constant_hessian else all_gradients.shape
    all_hessians = np.ones(shape=shape_hessians, dtype=G_H_DTYPE)

    return X_binned, all_gradients, all_hessians


def _check_children_consistency(parent, left, right):
    # [FUNCTION DOC]
    # HELPER: Asserts that left/right are the parent's children and that their
    # sample_indices form a disjoint partition of the parent's sample_indices.
    # Make sure the samples are correctly dispatched from a parent to its
    # children
    assert parent.left_child is left
    assert parent.right_child is right

    # each sample from the parent is propagated to one of the two children
    assert len(left.sample_indices) + len(right.sample_indices) == len(
        parent.sample_indices
    )

    assert set(left.sample_indices).union(set(right.sample_indices)) == set(
        parent.sample_indices
    )

    # samples are sent either to the left or the right node, never to both
    assert set(left.sample_indices).intersection(set(right.sample_indices)) == set()


@pytest.mark.parametrize(
    "n_bins, constant_hessian, stopping_param, shrinkage",
    [
        (11, True, "min_gain_to_split", 0.5),
        (11, False, "min_gain_to_split", 1.0),
        (11, True, "max_leaf_nodes", 1.0),
        (11, False, "max_leaf_nodes", 0.1),
        (42, True, "max_leaf_nodes", 0.01),
        (42, False, "max_leaf_nodes", 1.0),
        (256, True, "min_gain_to_split", 1.0),
        (256, True, "max_leaf_nodes", 0.1),
    ],
)
def test_grow_tree(n_bins, constant_hessian, stopping_param, shrinkage):
    # [FUNCTION DOC]
    # Drives the grower manually with split_next: checks the root's best split is on
    # feature 0 at bin n_bins // 2, the children split roughly 50/50, the left child is a
    # finalized leaf when min_gain_to_split > 0, the right child splits on feature 1 at
    # bin n_bins // 3 with expected size proportions, no node remains splittable, and after
    # _apply_shrinkage leaf values equal +/- shrinkage. Parametrized over n_bins, hessian
    # type, stopping criterion and shrinkage.
    X_binned, all_gradients, all_hessians = _make_training_data(
        n_bins=n_bins, constant_hessian=constant_hessian
    )
    n_samples = X_binned.shape[0]

    if stopping_param == "max_leaf_nodes":
        stopping_param = {"max_leaf_nodes": 3}
    else:
        stopping_param = {"min_gain_to_split": 0.01}

    grower = TreeGrower(
        X_binned,
        all_gradients,
        all_hessians,
        n_bins=n_bins,
        shrinkage=shrinkage,
        min_samples_leaf=1,
        **stopping_param,
    )

    # The root node is not yet split, but the best possible split has
    # already been evaluated:
    assert grower.root.left_child is None
    assert grower.root.right_child is None

    root_split = grower.root.split_info
    assert root_split.feature_idx == 0
    assert root_split.bin_idx == n_bins // 2
    assert len(grower.splittable_nodes) == 1

    # Calling split next applies the next split and computes the best split
    # for each of the two newly introduced children nodes.
    left_node, right_node = grower.split_next()

    # All training samples have ben split in the two nodes, approximately
    # 50%/50%
    _check_children_consistency(grower.root, left_node, right_node)
    assert len(left_node.sample_indices) > 0.4 * n_samples
    assert len(left_node.sample_indices) < 0.6 * n_samples

    if grower.min_gain_to_split > 0:
        # The left node is too pure: there is no gain to split it further.
        assert left_node.split_info.gain < grower.min_gain_to_split
        assert left_node in grower.finalized_leaves

    # The right node can still be split further, this time on feature #1
    split_info = right_node.split_info
    assert split_info.gain > 1.0
    assert split_info.feature_idx == 1
    assert split_info.bin_idx == n_bins // 3
    assert right_node.left_child is None
    assert right_node.right_child is None

    # The right split has not been applied yet. Let's do it now:
    assert len(grower.splittable_nodes) == 1
    right_left_node, right_right_node = grower.split_next()
    _check_children_consistency(right_node, right_left_node, right_right_node)
    assert len(right_left_node.sample_indices) > 0.1 * n_samples
    assert len(right_left_node.sample_indices) < 0.2 * n_samples

    assert len(right_right_node.sample_indices) > 0.2 * n_samples
    assert len(right_right_node.sample_indices) < 0.4 * n_samples

    # All the leaves are pure, it is not possible to split any further:
    assert not grower.splittable_nodes

    grower._apply_shrinkage()

    # Check the values of the leaves:
    assert grower.root.left_child.value == approx(shrinkage)
    assert grower.root.right_child.left_child.value == approx(shrinkage)
    assert grower.root.right_child.right_child.value == approx(-shrinkage, rel=1e-3)


def test_predictor_from_grower():
    # [FUNCTION DOC]
    # Verifies that a 3-leaf tree converts into a 5-node predictor with 3 leaves, that
    # binned probe points reach the expected leaves, and that the training set
    # predictions equal the negated gradients exactly.
    # Build a tree on the toy 3-leaf dataset to extract the predictor.
    n_bins = 256
    X_binned, all_gradients, all_hessians = _make_training_data(n_bins=n_bins)
    grower = TreeGrower(
        X_binned,
        all_gradients,
        all_hessians,
        n_bins=n_bins,
        shrinkage=1.0,
        max_leaf_nodes=3,
        min_samples_leaf=5,
    )
    grower.grow()
    assert grower.n_nodes == 5  # (2 decision nodes + 3 leaves)

    # Check that the node structure can be converted into a predictor
    # object to perform predictions at scale
    # We pass undefined binning_thresholds because we won't use predict anyway
    predictor = grower.make_predictor(
        binning_thresholds=np.zeros((X_binned.shape[1], n_bins))
    )
    assert predictor.nodes.shape[0] == 5
    assert predictor.nodes["is_leaf"].sum() == 3

    # Probe some predictions for each leaf of the tree
    # each group of 3 samples corresponds to a condition in _make_training_data
    input_data = np.array(
        [
            [0, 0],
            [42, 99],
            [128, 254],
            [129, 0],
            [129, 85],
            [254, 85],
            [129, 86],
            [129, 254],
            [242, 100],
        ],
        dtype=np.uint8,
    )
    missing_values_bin_idx = n_bins - 1
    predictions = predictor.predict_binned(
        input_data, missing_values_bin_idx, n_threads
    )
    expected_targets = [1, 1, 1, 1, 1, 1, -1, -1, -1]
    assert np.allclose(predictions, expected_targets)

    # Check that training set can be recovered exactly:
    predictions = predictor.predict_binned(X_binned, missing_values_bin_idx, n_threads)
    assert np.allclose(predictions, -all_gradients)


@pytest.mark.parametrize(
    "n_samples, min_samples_leaf, n_bins, constant_hessian, noise",
    [
        (11, 10, 7, True, 0),
        (13, 10, 42, False, 0),
        (56, 10, 255, True, 0.1),
        (101, 3, 7, True, 0),
        (200, 42, 42, False, 0),
        (300, 55, 255, True, 0.1),
        (300, 301, 255, True, 0.1),
    ],
)
def test_min_samples_leaf(n_samples, min_samples_leaf, n_bins, constant_hessian, noise):
    # [FUNCTION DOC]
    # Verifies that every leaf of the resulting predictor has count >= min_samples_leaf,
    # or that the tree is a single root leaf with all samples when n_samples <
    # min_samples_leaf. Uses a linear target on 3 normal features (optionally noisy),
    # binned with _BinMapper.
    rng = np.random.RandomState(seed=0)
    # data = linear target, 3 features, 1 irrelevant.
    X = rng.normal(size=(n_samples, 3))
    y = X[:, 0] - X[:, 1]
    if noise:
        y_scale = y.std()
        y += rng.normal(scale=noise, size=n_samples) * y_scale
    mapper = _BinMapper(n_bins=n_bins)
    X = mapper.fit_transform(X)

    all_gradients = y.astype(G_H_DTYPE)
    shape_hessian = 1 if constant_hessian else all_gradients.shape
    all_hessians = np.ones(shape=shape_hessian, dtype=G_H_DTYPE)
    grower = TreeGrower(
        X,
        all_gradients,
        all_hessians,
        n_bins=n_bins,
        shrinkage=1.0,
        min_samples_leaf=min_samples_leaf,
        max_leaf_nodes=n_samples,
    )
    grower.grow()
    predictor = grower.make_predictor(binning_thresholds=mapper.bin_thresholds_)

    if n_samples >= min_samples_leaf:
        for node in predictor.nodes:
            if node["is_leaf"]:
                assert node["count"] >= min_samples_leaf
    else:
        assert predictor.nodes.shape[0] == 1
        assert predictor.nodes[0]["is_leaf"]
        assert predictor.nodes[0]["count"] == n_samples


@pytest.mark.parametrize("n_samples, min_samples_leaf", [(99, 50), (100, 50)])
def test_min_samples_leaf_root(n_samples, min_samples_leaf):
    # [FUNCTION DOC]
    # Verifies that the root is split only when n_samples >= 2 * min_samples_leaf
    # (at least 2 leaves for 100/50, a single leaf for 99/50).
    # Make sure root node isn't split if n_samples is not at least twice
    # min_samples_leaf
    rng = np.random.RandomState(seed=0)

    n_bins = 256

    # data = linear target, 3 features, 1 irrelevant.
    X = rng.normal(size=(n_samples, 3))
    y = X[:, 0] - X[:, 1]
    mapper = _BinMapper(n_bins=n_bins)
    X = mapper.fit_transform(X)

    all_gradients = y.astype(G_H_DTYPE)
    all_hessians = np.ones(shape=1, dtype=G_H_DTYPE)
    grower = TreeGrower(
        X,
        all_gradients,
        all_hessians,
        n_bins=n_bins,
        shrinkage=1.0,
        min_samples_leaf=min_samples_leaf,
        max_leaf_nodes=n_samples,
    )
    grower.grow()
    if n_samples >= min_samples_leaf * 2:
        assert len(grower.finalized_leaves) >= 2
    else:
        assert len(grower.finalized_leaves) == 1


def assert_is_stump(grower):
    # [FUNCTION DOC]
    # HELPER: Asserts that both children of the grower's root are leaves (no children),
    # i.e. the tree is a stump.
    # To assert that stumps are created when max_depth=1
    for leaf in (grower.root.left_child, grower.root.right_child):
        assert leaf.left_child is None
        assert leaf.right_child is None


@pytest.mark.parametrize("max_depth", [1, 2, 3])
def test_max_depth(max_depth):
    # [FUNCTION DOC]
    # Verifies that the deepest finalized leaf has depth exactly max_depth (1, 2, 3) and
    # that max_depth=1 yields a stump.
    # Make sure max_depth parameter works as expected
    rng = np.random.RandomState(seed=0)

    n_bins = 256
    n_samples = 1000

    # data = linear target, 3 features, 1 irrelevant.
    X = rng.normal(size=(n_samples, 3))
    y = X[:, 0] - X[:, 1]
    mapper = _BinMapper(n_bins=n_bins)
    X = mapper.fit_transform(X)

    all_gradients = y.astype(G_H_DTYPE)
    all_hessians = np.ones(shape=1, dtype=G_H_DTYPE)
    grower = TreeGrower(X, all_gradients, all_hessians, max_depth=max_depth)
    grower.grow()

    depth = max(leaf.depth for leaf in grower.finalized_leaves)
    assert depth == max_depth

    if max_depth == 1:
        assert_is_stump(grower)


def test_input_validation():
    # [FUNCTION DOC]
    # Verifies TreeGrower raises NotImplementedError for non-uint8 X_binned and
    # ValueError for a C-contiguous X_binned.
    X_binned, all_gradients, all_hessians = _make_training_data()

    X_binned_float = X_binned.astype(np.float32)
    with pytest.raises(NotImplementedError, match="X_binned must be of type uint8"):
        TreeGrower(X_binned_float, all_gradients, all_hessians)

    X_binned_C_array = np.ascontiguousarray(X_binned)
    with pytest.raises(
        ValueError, match="X_binned should be passed as Fortran contiguous array"
    ):
        TreeGrower(X_binned_C_array, all_gradients, all_hessians)


def test_init_parameters_validation():
    # [FUNCTION DOC]
    # Verifies TreeGrower raises ValueError for negative min_gain_to_split and negative
    # min_hessian_to_split.
    X_binned, all_gradients, all_hessians = _make_training_data()
    with pytest.raises(ValueError, match="min_gain_to_split=-1 must be positive"):
        TreeGrower(X_binned, all_gradients, all_hessians, min_gain_to_split=-1)

    with pytest.raises(ValueError, match="min_hessian_to_split=-1 must be positive"):
        TreeGrower(X_binned, all_gradients, all_hessians, min_hessian_to_split=-1)


def test_missing_value_predict_only():
    # [FUNCTION DOC]
    # Verifies that when no missing values were seen in training, NaNs at predict time
    # follow, at each node, the child with more samples: all-NaN predictions equal the
    # value of the leaf reached by that majority path.
    # Make sure that missing values are supported at predict time even if they
    # were not encountered in the training data: the missing values are
    # assigned to whichever child has the most samples.

    rng = np.random.RandomState(0)
    n_samples = 100
    X_binned = rng.randint(0, 256, size=(n_samples, 1), dtype=np.uint8)
    X_binned = np.asfortranarray(X_binned)

    gradients = rng.normal(size=n_samples).astype(G_H_DTYPE)
    hessians = np.ones(shape=1, dtype=G_H_DTYPE)

    grower = TreeGrower(
        X_binned, gradients, hessians, min_samples_leaf=5, has_missing_values=False
    )
    grower.grow()

    # We pass undefined binning_thresholds because we won't use predict anyway
    predictor = grower.make_predictor(
        binning_thresholds=np.zeros((X_binned.shape[1], X_binned.max() + 1))
    )

    # go from root to a leaf, always following node with the most samples.
    # That's the path nans are supposed to take
    node = predictor.nodes[0]
    while not node["is_leaf"]:
        left = predictor.nodes[node["left"]]
        right = predictor.nodes[node["right"]]
        node = left if left["count"] > right["count"] else right

    prediction_main_path = node["value"]

    # now build X_test with only nans, and make sure all predictions are equal
    # to prediction_main_path
    all_nans = np.full(shape=(n_samples, 1), fill_value=np.nan)
    known_cat_bitsets = np.zeros((0, 8), dtype=X_BITSET_INNER_DTYPE)
    f_idx_map = np.zeros(0, dtype=np.uint32)

    y_pred = predictor.predict(all_nans, known_cat_bitsets, f_idx_map, n_threads)
    assert np.all(y_pred == prediction_main_path)


def test_split_on_nan_with_infinite_values():
    # [FUNCTION DOC]
    # Verifies that with a "split on nan" root (threshold +inf, bin threshold
    # n_bins_non_missing - 1) a +inf sample goes left with the finite values while NaNs go
    # right, for both predict and predict_binned (predictions equal negated gradients).
    # Make sure the split on nan situations are respected even when there are
    # samples with +inf values (we set the threshold to +inf when we have a
    # split on nan so this test makes sure this does not introduce edge-case
    # bugs). We need to use the private API so that we can also test
    # predict_binned().

    X = np.array([0, 1, np.inf, np.nan, np.nan]).reshape(-1, 1)
    # the gradient values will force a split on nan situation
    gradients = np.array([0, 0, 0, 100, 100], dtype=G_H_DTYPE)
    hessians = np.ones(shape=1, dtype=G_H_DTYPE)

    bin_mapper = _BinMapper()
    X_binned = bin_mapper.fit_transform(X)

    n_bins_non_missing = 3
    has_missing_values = True
    grower = TreeGrower(
        X_binned,
        gradients,
        hessians,
        n_bins_non_missing=n_bins_non_missing,
        has_missing_values=has_missing_values,
        min_samples_leaf=1,
        n_threads=n_threads,
    )

    grower.grow()

    predictor = grower.make_predictor(binning_thresholds=bin_mapper.bin_thresholds_)

    # sanity check: this was a split on nan
    assert predictor.nodes[0]["num_threshold"] == np.inf
    assert predictor.nodes[0]["bin_threshold"] == n_bins_non_missing - 1

    known_cat_bitsets, f_idx_map = bin_mapper.make_known_categories_bitsets()

    # Make sure in particular that the +inf sample is mapped to the left child
    # Note that lightgbm "fails" here and will assign the inf sample to the
    # right child, even though it's a "split on nan" situation.
    predictions = predictor.predict(X, known_cat_bitsets, f_idx_map, n_threads)
    predictions_binned = predictor.predict_binned(
        X_binned,
        missing_values_bin_idx=bin_mapper.missing_values_bin_idx_,
        n_threads=n_threads,
    )
    np.testing.assert_allclose(predictions, -gradients)
    np.testing.assert_allclose(predictions_binned, -gradients)


def test_grow_tree_categories():
    # [FUNCTION DOC]
    # Verifies categorical split growth on a 2-category feature: 3 nodes, a categorical
    # root with 23 samples, left child the larger one, binned bitset containing bin 1 and
    # raw bitset containing category 9, missing_go_to_left set, and missing values (binned
    # index 6 or raw NaN) predicted with the left leaf value -1.
    # Check that the grower produces the right predictor tree when a split is
    # categorical
    X_binned = np.array([[0, 1] * 11 + [1]], dtype=X_BINNED_DTYPE).T
    X_binned = np.asfortranarray(X_binned)

    all_gradients = np.array([10, 1] * 11 + [1], dtype=G_H_DTYPE)
    all_hessians = np.ones(1, dtype=G_H_DTYPE)
    is_categorical = np.ones(1, dtype=np.uint8)

    grower = TreeGrower(
        X_binned,
        all_gradients,
        all_hessians,
        n_bins=4,
        shrinkage=1.0,
        min_samples_leaf=1,
        is_categorical=is_categorical,
        n_threads=n_threads,
    )
    grower.grow()
    assert grower.n_nodes == 3

    categories = [np.array([4, 9], dtype=X_DTYPE)]
    predictor = grower.make_predictor(binning_thresholds=categories)
    root = predictor.nodes[0]
    assert root["count"] == 23
    assert root["depth"] == 0
    assert root["is_categorical"]

    left, right = predictor.nodes[root["left"]], predictor.nodes[root["right"]]

    # arbitrary validation, but this means ones go to the left.
    assert left["count"] >= right["count"]

    # check binned category value (1)
    expected_binned_cat_bitset = [2**1] + [0] * 7
    binned_cat_bitset = predictor.binned_left_cat_bitsets
    assert_array_equal(binned_cat_bitset[0], expected_binned_cat_bitset)

    # check raw category value (9)
    expected_raw_cat_bitsets = [2**9] + [0] * 7
    raw_cat_bitsets = predictor.raw_left_cat_bitsets
    assert_array_equal(raw_cat_bitsets[0], expected_raw_cat_bitsets)

    # Note that since there was no missing values during training, the missing
    # values aren't part of the bitsets. However, we expect the missing values
    # to go to the biggest child (i.e. the left one).
    # The left child has a value of -1 = negative gradient.
    assert root["missing_go_to_left"]

    # make sure binned missing values are mapped to the left child during
    # prediction
    prediction_binned = predictor.predict_binned(
        np.asarray([[6]]).astype(X_BINNED_DTYPE),
        missing_values_bin_idx=6,
        n_threads=n_threads,
    )
    assert_allclose(prediction_binned, [-1])  # negative gradient

    # make sure raw missing values are mapped to the left child during
    # prediction
    known_cat_bitsets = np.zeros((1, 8), dtype=np.uint32)  # ignored anyway
    f_idx_map = np.array([0], dtype=np.uint32)
    prediction = predictor.predict(
        np.array([[np.nan]]), known_cat_bitsets, f_idx_map, n_threads
    )
    assert_allclose(prediction, [-1])


@pytest.mark.parametrize("min_samples_leaf", (1, 20))
@pytest.mark.parametrize("n_unique_categories", (2, 10, 100))
@pytest.mark.parametrize("target", ("binary", "random", "equal"))
def test_ohe_equivalence(min_samples_leaf, n_unique_categories, target):
    # [FUNCTION DOC]
    # Verifies that a tree using a native categorical split predicts the same as a tree
    # grown on the one-hot encoded feature, with depth no larger (strictly smaller for
    # the "binary" parity target with more than 2 categories). Parametrized over
    # min_samples_leaf, number of categories and target type.
    # Make sure that native categorical splits are equivalent to using a OHE,
    # when given enough depth

    rng = np.random.RandomState(0)
    n_samples = 10_000
    X_binned = rng.randint(0, n_unique_categories, size=(n_samples, 1), dtype=np.uint8)

    X_ohe = OneHotEncoder(sparse_output=False).fit_transform(X_binned)
    X_ohe = np.asfortranarray(X_ohe).astype(np.uint8)

    if target == "equal":
        gradients = X_binned.reshape(-1)
    elif target == "binary":
        gradients = (X_binned % 2).reshape(-1)
    else:
        gradients = rng.randn(n_samples)
    gradients = gradients.astype(G_H_DTYPE)

    hessians = np.ones(shape=1, dtype=G_H_DTYPE)

    grower_params = {
        "min_samples_leaf": min_samples_leaf,
        "max_depth": None,
        "max_leaf_nodes": None,
    }

    grower = TreeGrower(
        X_binned, gradients, hessians, is_categorical=[True], **grower_params
    )
    grower.grow()
    # we pass undefined bin_thresholds because we won't use predict()
    predictor = grower.make_predictor(
        binning_thresholds=np.zeros((1, n_unique_categories))
    )
    preds = predictor.predict_binned(
        X_binned, missing_values_bin_idx=255, n_threads=n_threads
    )

    grower_ohe = TreeGrower(X_ohe, gradients, hessians, **grower_params)
    grower_ohe.grow()
    predictor_ohe = grower_ohe.make_predictor(
        binning_thresholds=np.zeros((X_ohe.shape[1], n_unique_categories))
    )
    preds_ohe = predictor_ohe.predict_binned(
        X_ohe, missing_values_bin_idx=255, n_threads=n_threads
    )

    assert predictor.get_max_depth() <= predictor_ohe.get_max_depth()
    if target == "binary" and n_unique_categories > 2:
        # OHE needs more splits to achieve the same predictions
        assert predictor.get_max_depth() < predictor_ohe.get_max_depth()

    np.testing.assert_allclose(preds, preds_ohe)


def test_grower_interaction_constraints():
    # [FUNCTION DOC]
    # Verifies interaction constraints over 20 random seeds: root children's
    # allowed_features equal the union of constraint sets containing the root feature,
    # each child's interaction_cst_indices is a subset of its parent's, every split
    # feature lies in the root's constraint set, and across seeds all 6 features are used
    # at the root at least once.
    """Check that grower respects interaction constraints."""
    n_features = 6
    interaction_cst = [{0, 1}, {1, 2}, {3, 4, 5}]
    n_samples = 10
    n_bins = 6
    root_feature_splits = []

    def get_all_children(node):
        # [FUNCTION DOC]
        # HELPER: Recursively returns a list of all descendant nodes (pre-order, left before
        # right) of a grower node; empty list for a leaf.
        res = []
        if node.is_leaf:
            return res
        for n in [node.left_child, node.right_child]:
            res.append(n)
            res.extend(get_all_children(n))
        return res

    for seed in range(20):
        rng = np.random.RandomState(seed)

        X_binned = rng.randint(
            0, n_bins - 1, size=(n_samples, n_features), dtype=X_BINNED_DTYPE
        )
        X_binned = np.asfortranarray(X_binned)
        gradients = rng.normal(size=n_samples).astype(G_H_DTYPE)
        hessians = np.ones(shape=1, dtype=G_H_DTYPE)

        grower = TreeGrower(
            X_binned,
            gradients,
            hessians,
            n_bins=n_bins,
            min_samples_leaf=1,
            interaction_cst=interaction_cst,
            n_threads=n_threads,
        )
        grower.grow()

        root_feature_idx = grower.root.split_info.feature_idx
        root_feature_splits.append(root_feature_idx)

        feature_idx_to_constraint_set = {
            0: {0, 1},
            1: {0, 1, 2},
            2: {1, 2},
            3: {3, 4, 5},
            4: {3, 4, 5},
            5: {3, 4, 5},
        }

        root_constraint_set = feature_idx_to_constraint_set[root_feature_idx]
        for node in (grower.root.left_child, grower.root.right_child):
            # Root's children's allowed_features must be the root's constraints set.
            assert_array_equal(node.allowed_features, list(root_constraint_set))
        for node in get_all_children(grower.root):
            if node.is_leaf:
                continue
            # Ensure that each node uses a subset of features of its parent node.
            parent_interaction_cst_indices = set(node.interaction_cst_indices)
            right_interactions_cst_indices = set(
                node.right_child.interaction_cst_indices
            )
            left_interactions_cst_indices = set(node.left_child.interaction_cst_indices)

            assert right_interactions_cst_indices.issubset(
                parent_interaction_cst_indices
            )
            assert left_interactions_cst_indices.issubset(
                parent_interaction_cst_indices
            )
            # The features used for split must have been present in the root's
            # constraint set.
            assert node.split_info.feature_idx in root_constraint_set

    # Make sure that every feature is used at least once as split for the root node.
    assert (
        len(set(root_feature_splits))
        == len(set().union(*interaction_cst))
        == n_features
    )
