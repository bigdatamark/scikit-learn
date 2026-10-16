# [MODULE DOC]
# Tests for NeighborhoodComponentsAnalysis: learned transform behaviour on toy
# data, gradient correctness, parameter/init validation, auto-init selection,
# warm start, verbose output, degenerate class configurations, callbacks,
# convergence warning, numpy scalar params and feature names.
# Import-time side effects: loads iris, permutes it with a seeded RNG and marks the
# resulting module-global arrays read-only; defines EPS (unused here).
"""
Testing for Neighborhood Component Analysis module (sklearn.neighbors.nca)
"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import re

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal
from scipy.optimize import check_grad

from sklearn import clone
from sklearn.datasets import load_iris, make_blobs, make_classification
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import check_random_state
from sklearn.utils.validation import validate_data

rng = check_random_state(0)
# Load and shuffle the iris dataset.
iris = load_iris()
perm = rng.permutation(iris.target.size)
iris_data = iris.data[perm]
iris_target = iris.target[perm]
# Avoid having test data introducing dependencies between tests.
iris_data.flags.writeable = False
iris_target.flags.writeable = False
EPS = np.finfo(float).eps


def test_simple_example():
    # [FUNCTION DOC]
    # Checks that after fitting on 4 points where opposite labels are adjacent, each
    # point's nearest other point in the transformed space has the same label.
    """Test on a simple example.

    Puts four points in the input space where the opposite labels points are
    next to each other. After transform the samples from the same class
    should be next to each other.

    """
    X = np.array([[0, 0], [0, 1], [2, 0], [2, 1]])
    y = np.array([1, 0, 1, 0])
    nca = NeighborhoodComponentsAnalysis(
        n_components=2, init="identity", random_state=42
    )
    nca.fit(X, y)
    X_t = nca.transform(X)
    assert_array_equal(pairwise_distances(X_t).argsort()[:, 1], np.array([2, 3, 0, 1]))


def test_toy_example_collapse_points():
    # [FUNCTION DOC]
    # Two same-class points plus a different-class point at their midpoint: checks the
    # fitted transform collapses all three to one point and the final loss (recorded
    # via a callback) is -1 within 1e-10.
    # KNOWN ISSUE: leftover debug print of X_t.
    """Test on a toy example of three points that should collapse

    We build a simple example: two points from the same class and a point from
    a different class in the middle of them. On this simple example, the new
    (transformed) points should all collapse into one single point. Indeed, the
    objective is 2/(1 + exp(d/2)), with d the euclidean distance between the
    two samples from the same class. This is maximized for d=0 (because d>=0),
    with an objective equal to 1 (loss=-1.).

    """
    rng = np.random.RandomState(42)
    input_dim = 5
    two_points = rng.randn(2, input_dim)
    X = np.vstack([two_points, two_points.mean(axis=0)[np.newaxis, :]])
    y = [0, 0, 1]

    class LossStorer:
        # [CLASS DOC]
        # HELPER: Callback holder that recomputes the NCA loss at each optimizer iteration
        # using a separate unfitted NCA instance.
        def __init__(self, X, y):
            # [FUNCTION DOC]
            # HELPER: Sets loss to inf, creates a fake NCA with n_iter_ set to inf, validates
            # X/y with it, label-encodes y and stores the same-class boolean mask.
            # Side effects: sets self.loss, self.fake_nca, self.X, self.same_class_mask.
            self.loss = np.inf  # initialize the loss to very high
            # Initialize a fake NCA and variables needed to compute the loss:
            self.fake_nca = NeighborhoodComponentsAnalysis()
            self.fake_nca.n_iter_ = np.inf
            self.X, y = validate_data(self.fake_nca, X, y, ensure_min_samples=2)
            y = LabelEncoder().fit_transform(y)
            self.same_class_mask = y[:, np.newaxis] == y[np.newaxis, :]

        def callback(self, transformation, n_iter):
            # [FUNCTION DOC]
            # HELPER: Calls the fake NCA's _loss_grad_lbfgs with sign -1 on the flattened
            # transformation and stores the loss in self.loss (n_iter is ignored).
            """Stores the last value of the loss function"""
            self.loss, _ = self.fake_nca._loss_grad_lbfgs(
                transformation, self.X, self.same_class_mask, -1.0
            )

    loss_storer = LossStorer(X, y)
    nca = NeighborhoodComponentsAnalysis(random_state=42, callback=loss_storer.callback)
    X_t = nca.fit_transform(X, y)
    print(X_t)
    # test that points are collapsed into one point
    assert_array_almost_equal(X_t - X_t[0], 0.0)
    assert abs(loss_storer.loss + 1) < 1e-10


def test_finite_differences(global_random_seed):
    # [FUNCTION DOC]
    # Checks _loss_grad_lbfgs's gradient against scipy check_grad finite differences
    # for a random transformation with random output dimension (abs tol 1e-4).
    """Test gradient of loss function

    Assert that the gradient is almost equal to its finite differences
    approximation.
    """
    # Initialize the transformation `M`, as well as `X` and `y` and `NCA`
    rng = np.random.RandomState(global_random_seed)
    X, y = make_classification(random_state=global_random_seed)
    M = rng.randn(rng.randint(1, X.shape[1] + 1), X.shape[1])
    nca = NeighborhoodComponentsAnalysis()
    nca.n_iter_ = 0
    mask = y[:, np.newaxis] == y[np.newaxis, :]

    def fun(M):
        # [FUNCTION DOC]
        # HELPER: Returns the loss for flattened transformation M.
        return nca._loss_grad_lbfgs(M, X, mask)[0]

    def grad(M):
        # [FUNCTION DOC]
        # HELPER: Returns the gradient for flattened transformation M.
        return nca._loss_grad_lbfgs(M, X, mask)[1]

    # compare the gradient to a finite difference approximation
    diff = check_grad(fun, grad, M.ravel())
    assert diff == pytest.approx(0.0, abs=1e-4)


def test_params_validation():
    # [FUNCTION DOC]
    # Checks fit raises ValueError with exact messages when init has more output than
    # input dimensions and when n_components exceeds n_features.
    # Test that invalid parameters raise value error
    X = np.arange(12).reshape(4, 3)
    y = [1, 1, 2, 2]
    NCA = NeighborhoodComponentsAnalysis
    rng = np.random.RandomState(42)

    init = rng.rand(5, 3)
    msg = (
        f"The output dimensionality ({init.shape[0]}) "
        "of the given linear transformation `init` cannot be "
        f"greater than its input dimensionality ({init.shape[1]})."
    )
    with pytest.raises(ValueError, match=re.escape(msg)):
        NCA(init=init).fit(X, y)
    n_components = 10
    msg = (
        "The preferred dimensionality of the projected space "
        f"`n_components` ({n_components}) cannot be greater "
        f"than the given data dimensionality ({X.shape[1]})!"
    )
    with pytest.raises(ValueError, match=re.escape(msg)):
        NCA(n_components=n_components).fit(X, y)


def test_transformation_dimensions():
    # [FUNCTION DOC]
    # Checks fit fails when init's input dimension mismatches X or its output dim
    # exceeds its input dim, and succeeds with a square matching init.
    X = np.arange(12).reshape(4, 3)
    y = [1, 1, 2, 2]

    # Fail if transformation input dimension does not match inputs dimensions
    transformation = np.array([[1, 2], [3, 4]])
    with pytest.raises(ValueError):
        NeighborhoodComponentsAnalysis(init=transformation).fit(X, y)

    # Fail if transformation output dimension is larger than
    # transformation input dimension
    transformation = np.array([[1, 2], [3, 4], [5, 6]])
    # len(transformation) > len(transformation[0])
    with pytest.raises(ValueError):
        NeighborhoodComponentsAnalysis(init=transformation).fit(X, y)

    # Pass otherwise
    transformation = np.arange(9).reshape(3, 3)
    NeighborhoodComponentsAnalysis(init=transformation).fit(X, y)


def test_n_components():
    # [FUNCTION DOC]
    # Checks ValueError messages when n_components mismatches init's output dimension
    # or exceeds n_features, and that a valid n_components with identity init fits.
    rng = np.random.RandomState(42)
    X = np.arange(12).reshape(4, 3)
    y = [1, 1, 2, 2]

    init = rng.rand(X.shape[1] - 1, 3)

    # n_components = X.shape[1] != transformation.shape[0]
    n_components = X.shape[1]
    nca = NeighborhoodComponentsAnalysis(init=init, n_components=n_components)
    msg = (
        "The preferred dimensionality of the projected space "
        f"`n_components` ({n_components}) does not match the output "
        "dimensionality of the given linear transformation "
        f"`init` ({init.shape[0]})!"
    )
    with pytest.raises(ValueError, match=re.escape(msg)):
        nca.fit(X, y)

    # n_components > X.shape[1]
    n_components = X.shape[1] + 2
    nca = NeighborhoodComponentsAnalysis(init=init, n_components=n_components)
    msg = (
        "The preferred dimensionality of the projected space "
        f"`n_components` ({n_components}) cannot be greater than "
        f"the given data dimensionality ({X.shape[1]})!"
    )
    with pytest.raises(ValueError, match=re.escape(msg)):
        nca.fit(X, y)

    # n_components < X.shape[1]
    nca = NeighborhoodComponentsAnalysis(n_components=2, init="identity")
    nca.fit(X, y)


def test_init_transformation():
    # [FUNCTION DOC]
    # Checks every named init (identity, random, auto, pca, lda) and a valid array init
    # fit, and that array inits with wrong input dim, output > input dim, or output dim
    # different from n_components raise ValueError with exact messages.
    rng = np.random.RandomState(42)
    X, y = make_blobs(n_samples=30, centers=6, n_features=5, random_state=0)

    # Start learning from scratch
    nca = NeighborhoodComponentsAnalysis(init="identity")
    nca.fit(X, y)

    # Initialize with random
    nca_random = NeighborhoodComponentsAnalysis(init="random")
    nca_random.fit(X, y)

    # Initialize with auto
    nca_auto = NeighborhoodComponentsAnalysis(init="auto")
    nca_auto.fit(X, y)

    # Initialize with PCA
    nca_pca = NeighborhoodComponentsAnalysis(init="pca")
    nca_pca.fit(X, y)

    # Initialize with LDA
    nca_lda = NeighborhoodComponentsAnalysis(init="lda")
    nca_lda.fit(X, y)

    init = rng.rand(X.shape[1], X.shape[1])
    nca = NeighborhoodComponentsAnalysis(init=init)
    nca.fit(X, y)

    # init.shape[1] must match X.shape[1]
    init = rng.rand(X.shape[1], X.shape[1] + 1)
    nca = NeighborhoodComponentsAnalysis(init=init)
    msg = (
        f"The input dimensionality ({init.shape[1]}) of the given "
        "linear transformation `init` must match the "
        f"dimensionality of the given inputs `X` ({X.shape[1]})."
    )
    with pytest.raises(ValueError, match=re.escape(msg)):
        nca.fit(X, y)

    # init.shape[0] must be <= init.shape[1]
    init = rng.rand(X.shape[1] + 1, X.shape[1])
    nca = NeighborhoodComponentsAnalysis(init=init)
    msg = (
        f"The output dimensionality ({init.shape[0]}) of the given "
        "linear transformation `init` cannot be "
        f"greater than its input dimensionality ({init.shape[1]})."
    )
    with pytest.raises(ValueError, match=re.escape(msg)):
        nca.fit(X, y)

    # init.shape[0] must match n_components
    init = rng.rand(X.shape[1], X.shape[1])
    n_components = X.shape[1] - 2
    nca = NeighborhoodComponentsAnalysis(init=init, n_components=n_components)
    msg = (
        "The preferred dimensionality of the "
        f"projected space `n_components` ({n_components}) "
        "does not match the output dimensionality of the given "
        f"linear transformation `init` ({init.shape[0]})!"
    )
    with pytest.raises(ValueError, match=re.escape(msg)):
        nca.fit(X, y)


@pytest.mark.parametrize("n_samples", [3, 5, 7, 11])
@pytest.mark.parametrize("n_features", [3, 5, 7, 11])
@pytest.mark.parametrize("n_classes", [5, 7, 11])
@pytest.mark.parametrize("n_components", [3, 5, 7, 11])
def test_auto_init(n_samples, n_features, n_classes, n_components):
    # [FUNCTION DOC]
    # For valid sample/feature/class/component combinations, checks init="auto" gives
    # the same components_ (max_iter=1) as lda when n_components <= min(n_classes-1,
    # n_features), pca when n_components < min(n_features, n_samples), else identity.
    # Invalid combinations are skipped silently.
    # Test that auto choose the init as expected with every configuration
    # of order of n_samples, n_features, n_classes and n_components.
    rng = np.random.RandomState(42)
    nca_base = NeighborhoodComponentsAnalysis(
        init="auto", n_components=n_components, max_iter=1, random_state=rng
    )
    if n_classes >= n_samples:
        pass
        # n_classes > n_samples is impossible, and n_classes == n_samples
        # throws an error from lda but is an absurd case
    else:
        X = rng.randn(n_samples, n_features)
        y = np.tile(range(n_classes), n_samples // n_classes + 1)[:n_samples]
        if n_components > n_features:
            # this would return a ValueError, which is already tested in
            # test_params_validation
            pass
        else:
            nca = clone(nca_base)
            nca.fit(X, y)
            if n_components <= min(n_classes - 1, n_features):
                nca_other = clone(nca_base).set_params(init="lda")
            elif n_components < min(n_features, n_samples):
                nca_other = clone(nca_base).set_params(init="pca")
            else:
                nca_other = clone(nca_base).set_params(init="identity")
            nca_other.fit(X, y)
            assert_array_almost_equal(nca.components_, nca_other.components_)


def test_warm_start_validation():
    # [FUNCTION DOC]
    # Checks refitting a warm-started model on data with fewer features raises the
    # dimensionality-mismatch ValueError.
    X, y = make_classification(
        n_samples=30,
        n_features=5,
        n_classes=4,
        n_redundant=0,
        n_informative=5,
        random_state=0,
    )

    nca = NeighborhoodComponentsAnalysis(warm_start=True, max_iter=5)
    nca.fit(X, y)

    X_less_features, y = make_classification(
        n_samples=30,
        n_features=4,
        n_classes=4,
        n_redundant=0,
        n_informative=4,
        random_state=0,
    )
    msg = (
        f"The new inputs dimensionality ({X_less_features.shape[1]}) "
        "does not match the input dimensionality of the previously learned "
        f"transformation ({nca.components_.shape[1]})."
    )
    with pytest.raises(ValueError, match=re.escape(msg)):
        nca.fit(X_less_features, y)


def test_warm_start_effectiveness():
    # [FUNCTION DOC]
    # Fits to convergence then refits with max_iter=1; checks the warm-started model's
    # components change by less than 3 (L1) and change less than the cold-started one.
    # Side effects: mutates max_iter on fitted estimators.
    # A 1-iteration second fit on same data should give almost same result
    # with warm starting, and quite different result without warm starting.

    nca_warm = NeighborhoodComponentsAnalysis(warm_start=True, random_state=0)
    nca_warm.fit(iris_data, iris_target)
    transformation_warm = nca_warm.components_
    nca_warm.max_iter = 1
    nca_warm.fit(iris_data, iris_target)
    transformation_warm_plus_one = nca_warm.components_

    nca_cold = NeighborhoodComponentsAnalysis(warm_start=False, random_state=0)
    nca_cold.fit(iris_data, iris_target)
    transformation_cold = nca_cold.components_
    nca_cold.max_iter = 1
    nca_cold.fit(iris_data, iris_target)
    transformation_cold_plus_one = nca_cold.components_

    diff_warm = np.sum(np.abs(transformation_warm_plus_one - transformation_warm))
    diff_cold = np.sum(np.abs(transformation_cold_plus_one - transformation_cold))
    assert diff_warm < 3.0, (
        "Transformer changed significantly after one "
        "iteration even though it was warm-started."
    )

    assert diff_cold > diff_warm, (
        "Cold-started transformer changed less "
        "significantly than warm-started "
        "transformer after one iteration."
    )


@pytest.mark.parametrize(
    "init_name", ["pca", "lda", "identity", "random", "precomputed"]
)
def test_verbose(init_name, capsys):
    # [FUNCTION DOC]
    # Captures stdout with verbose=1 for each init; checks the pca/lda timing line,
    # header and separator lines, per-iteration line format and final "Training took"
    # line.
    # assert there is proper output when verbose = 1, for every initialization
    # except auto because auto will call one of the others
    rng = np.random.RandomState(42)
    X, y = make_blobs(n_samples=30, centers=6, n_features=5, random_state=0)
    regexp_init = r"... done in \ *\d+\.\d{2}s"
    msgs = {
        "pca": "Finding principal components" + regexp_init,
        "lda": "Finding most discriminative components" + regexp_init,
    }
    if init_name == "precomputed":
        init = rng.randn(X.shape[1], X.shape[1])
    else:
        init = init_name
    nca = NeighborhoodComponentsAnalysis(verbose=1, init=init)
    nca.fit(X, y)
    out, _ = capsys.readouterr()

    # check output
    lines = re.split("\n+", out)
    # if pca or lda init, an additional line is printed, so we test
    # it and remove it to test the rest equally among initializations
    if init_name in ["pca", "lda"]:
        assert re.match(msgs[init_name], lines[0])
        lines = lines[1:]
    assert lines[0] == "[NeighborhoodComponentsAnalysis]"
    header = "{:>10} {:>20} {:>10}".format("Iteration", "Objective Value", "Time(s)")
    assert lines[1] == "[NeighborhoodComponentsAnalysis] {}".format(header)
    assert lines[2] == "[NeighborhoodComponentsAnalysis] {}".format("-" * len(header))
    for line in lines[3:-2]:
        # The following regex will match for instance:
        # '[NeighborhoodComponentsAnalysis]  0    6.988936e+01   0.01'
        assert re.match(
            r"\[NeighborhoodComponentsAnalysis\] *\d+ *\d\.\d{6}e"
            r"[+|-]\d+\ *\d+\.\d{2}",
            line,
        )
    assert re.match(
        r"\[NeighborhoodComponentsAnalysis\] Training took\ *\d+\.\d{2}s\.",
        lines[-2],
    )
    assert lines[-1] == ""


def test_no_verbose(capsys):
    # [FUNCTION DOC]
    # Checks fitting with default verbose prints nothing.
    # assert by default there is no output (verbose=0)
    nca = NeighborhoodComponentsAnalysis()
    nca.fit(iris_data, iris_target)
    out, _ = capsys.readouterr()
    # check output
    assert out == ""


def test_singleton_class():
    # [FUNCTION DOC]
    # Checks fit works with one singleton class, with two singleton classes, and with
    # only singleton classes, where identity init must leave the transform as identity.
    X = iris_data.copy()
    y = iris_target.copy()

    # one singleton class
    singleton_class = 1
    (ind_singleton,) = np.where(y == singleton_class)
    y[ind_singleton] = 2
    y[ind_singleton[0]] = singleton_class

    nca = NeighborhoodComponentsAnalysis(max_iter=30)
    nca.fit(X, y)

    # One non-singleton class
    (ind_1,) = np.where(y == 1)
    (ind_2,) = np.where(y == 2)
    y[ind_1] = 0
    y[ind_1[0]] = 1
    y[ind_2] = 0
    y[ind_2[0]] = 2

    nca = NeighborhoodComponentsAnalysis(max_iter=30)
    nca.fit(X, y)

    # Only singleton classes
    (ind_0,) = np.where(y == 0)
    (ind_1,) = np.where(y == 1)
    (ind_2,) = np.where(y == 2)
    X = X[[ind_0[0], ind_1[0], ind_2[0]]]
    y = y[[ind_0[0], ind_1[0], ind_2[0]]]

    nca = NeighborhoodComponentsAnalysis(init="identity", max_iter=30)
    nca.fit(X, y)
    assert_array_equal(X, nca.transform(X))


def test_one_class():
    # [FUNCTION DOC]
    # Checks fitting on a single class with identity init leaves transform identity
    # (no gradient signal).
    X = iris_data[iris_target == 0]
    y = iris_target[iris_target == 0]

    nca = NeighborhoodComponentsAnalysis(
        max_iter=30, n_components=X.shape[1], init="identity"
    )
    nca.fit(X, y)
    assert_array_equal(X, nca.transform(X))


def test_callback(capsys):
    # [FUNCTION DOC]
    # Checks the user callback is invoked with a flattened (n_features^2,) array and
    # that its printed output for n_iter=1 appears in stdout.
    max_iter = 10

    def my_cb(transformation, n_iter):
        # [FUNCTION DOC]
        # HELPER: Asserts flattened transformation shape and prints remaining iterations.
        assert transformation.shape == (iris_data.shape[1] ** 2,)
        rem_iter = max_iter - n_iter
        print("{} iterations remaining...".format(rem_iter))

    # assert that my_cb is called
    nca = NeighborhoodComponentsAnalysis(max_iter=max_iter, callback=my_cb, verbose=1)
    nca.fit(iris_data, iris_target)
    out, _ = capsys.readouterr()

    # check output
    assert "{} iterations remaining...".format(max_iter - 1) in out


def test_expected_transformation_shape():
    # [FUNCTION DOC]
    # Checks the transformation passed to the callback has n_features^2 elements.
    """Test that the transformation has the expected shape."""
    X = iris_data
    y = iris_target

    class TransformationStorer:
        # [CLASS DOC]
        # HELPER: Callback holder storing the last transformation seen by the optimizer.
        def __init__(self, X, y):
            # [FUNCTION DOC]
            # HELPER: Builds a fake NCA, validates X/y, stores X and same-class mask.
            # KNOWN ISSUE: fake_nca, X and same_class_mask are never used by this test.
            # Initialize a fake NCA and variables needed to call the loss
            # function:
            self.fake_nca = NeighborhoodComponentsAnalysis()
            self.fake_nca.n_iter_ = np.inf
            self.X, y = validate_data(self.fake_nca, X, y, ensure_min_samples=2)
            y = LabelEncoder().fit_transform(y)
            self.same_class_mask = y[:, np.newaxis] == y[np.newaxis, :]

        def callback(self, transformation, n_iter):
            # [FUNCTION DOC]
            # HELPER: Stores the received transformation in self.transformation.
            """Stores the last value of the transformation taken as input by
            the optimizer"""
            self.transformation = transformation

    transformation_storer = TransformationStorer(X, y)
    cb = transformation_storer.callback
    nca = NeighborhoodComponentsAnalysis(max_iter=5, callback=cb)
    nca.fit(X, y)
    assert transformation_storer.transformation.size == X.shape[1] ** 2


def test_convergence_warning():
    # [FUNCTION DOC]
    # Checks fitting with max_iter=2 emits ConvergenceWarning "[...] NCA did not
    # converge".
    nca = NeighborhoodComponentsAnalysis(max_iter=2, verbose=1)
    cls_name = nca.__class__.__name__
    msg = "[{}] NCA did not converge".format(cls_name)
    with pytest.warns(ConvergenceWarning, match=re.escape(msg)):
        nca.fit(iris_data, iris_target)


@pytest.mark.parametrize(
    "param, value",
    [
        ("n_components", np.int32(3)),
        ("max_iter", np.int32(100)),
        ("tol", np.float32(0.0001)),
    ],
)
def test_parameters_valid_types(param, value):
    # [FUNCTION DOC]
    # Checks numpy int32/float32 values for n_components, max_iter and tol are accepted
    # by fit without error.
    # check that no error is raised when parameters have numpy integer or
    # floating types.
    nca = NeighborhoodComponentsAnalysis(**{param: value})

    X = iris_data
    y = iris_target

    nca.fit(X, y)


@pytest.mark.parametrize("n_components", [None, 2])
def test_nca_feature_names_out(n_components):
    # [FUNCTION DOC]
    # Checks get_feature_names_out returns lowercase class name plus index for
    # n_components (or n_features when None) outputs (regression #28293).
    """Check `get_feature_names_out` for `NeighborhoodComponentsAnalysis`.

    Non-regression test for:
    https://github.com/scikit-learn/scikit-learn/issues/28293
    """

    X = iris_data
    y = iris_target

    est = NeighborhoodComponentsAnalysis(n_components=n_components).fit(X, y)
    names_out = est.get_feature_names_out()

    class_name_lower = est.__class__.__name__.lower()

    if n_components is not None:
        expected_n_features = n_components
    else:
        expected_n_features = X.shape[1]

    expected_names_out = np.array(
        [f"{class_name_lower}{i}" for i in range(expected_n_features)],
        dtype=object,
    )

    assert_array_equal(names_out, expected_names_out)
