# [MODULE DOC]
# Tests for random sampling utilities: sample_without_replacement (all methods),
# _random_choice_csc and the pure-Python rand_r reimplementation _our_rand_r_py.
# Includes reusable check_* helpers applied to each sampling method.
import numpy as np
import pytest
import scipy.sparse as sp
from numpy.testing import assert_array_almost_equal
from scipy.special import comb

from sklearn.utils._random import _our_rand_r_py
from sklearn.utils.random import _random_choice_csc, sample_without_replacement


###############################################################################
# test custom sampling without replacement algorithm
###############################################################################
def test_invalid_sample_without_replacement_algorithm():
    # [FUNCTION DOC]
    # Verifies sample_without_replacement raises ValueError for an unknown method name.
    with pytest.raises(ValueError):
        sample_without_replacement(5, 4, "unknown")


def test_sample_without_replacement_algorithms():
    # [FUNCTION DOC]
    # For each method (auto, tracking_selection, reservoir_sampling, pool), wraps
    # sample_without_replacement with that method fixed and runs the edge-case, basic
    # and distribution checks on the wrapper.
    methods = ("auto", "tracking_selection", "reservoir_sampling", "pool")

    for m in methods:

        def sample_without_replacement_method(
            n_population, n_samples, random_state=None
        ):
            # [FUNCTION DOC]
            # HELPER: Closure forwarding to sample_without_replacement with method fixed
            # to the loop's current m (called within the same iteration, so late binding
            # is harmless).
            # Inputs: n_population, n_samples - ints; random_state - seed/None (default
            # None).
            # Outputs: 1D int array of n_samples distinct indices in [0, n_population).
            return sample_without_replacement(
                n_population, n_samples, method=m, random_state=random_state
            )

        check_edge_case_of_sample_int(sample_without_replacement_method)
        check_sample_int(sample_without_replacement_method)
        check_sample_int_distribution(sample_without_replacement_method)


def check_edge_case_of_sample_int(sample_without_replacement):
    # [FUNCTION DOC]
    # HELPER: Checks a sampler raises ValueError when n_samples > n_population or either
    # is negative, and returns arrays of the right length for equal sizes (0,0 and 1,1)
    # and for n_population larger than n_samples (5,0 and 5,1).
    # Inputs: sample_without_replacement - callable(n_population, n_samples).
    # Exceptions: AssertionError (via pytest) when expectations fail.
    # n_population < n_sample
    with pytest.raises(ValueError):
        sample_without_replacement(0, 1)
    with pytest.raises(ValueError):
        sample_without_replacement(1, 2)

    # n_population == n_samples
    assert sample_without_replacement(0, 0).shape == (0,)

    assert sample_without_replacement(1, 1).shape == (1,)

    # n_population >= n_samples
    assert sample_without_replacement(5, 0).shape == (0,)
    assert sample_without_replacement(5, 1).shape == (1,)

    # n_population < 0 or n_samples < 0
    with pytest.raises(ValueError):
        sample_without_replacement(-1, 5)
    with pytest.raises(ValueError):
        sample_without_replacement(5, -1)


def check_sample_int(sample_without_replacement):
    # [FUNCTION DOC]
    # HELPER: For every n_samples from 0 to 100 with population 100, checks the output
    # has the requested length, all unique values, all below n_population; also checks
    # the 0/0 case is empty.
    # Inputs: sample_without_replacement - callable(n_population, n_samples).
    # This test is heavily inspired from test_random.py of python-core.
    #
    # For the entire allowable range of 0 <= k <= N, validate that
    # the sample is of the correct length and contains only unique items
    n_population = 100

    for n_samples in range(n_population + 1):
        s = sample_without_replacement(n_population, n_samples)
        assert len(s) == n_samples
        unique = np.unique(s)
        assert np.size(unique) == n_samples
        assert np.all(unique < n_population)

    # test edge case n_population == n_samples == 0
    assert np.size(sample_without_replacement(0, 0)) == 0


def check_sample_int_distribution(sample_without_replacement):
    # [FUNCTION DOC]
    # HELPER: For each n_samples in 0..9 with population 10, repeatedly samples (up to
    # 10000 trials) collecting distinct subsets as frozensets, stopping once all C(10,
    # n_samples) combinations have been seen.
    # Inputs: sample_without_replacement - callable(n_population, n_samples).
    # Exceptions: AssertionError if not all combinations appear within n_trials.
    # This test is heavily inspired from test_random.py of python-core.
    #
    # For the entire allowable range of 0 <= k <= N, validate that
    # sample generates all possible permutations
    n_population = 10

    # a large number of trials prevents false negatives without slowing normal
    # case
    n_trials = 10000

    for n_samples in range(n_population):
        # Counting the number of combinations is not as good as counting the
        # the number of permutations. However, it works with sampling algorithm
        # that does not provide a random permutation of the subset of integer.
        n_expected = comb(n_population, n_samples, exact=True)

        output = {}
        for i in range(n_trials):
            output[frozenset(sample_without_replacement(n_population, n_samples))] = (
                None
            )

            if len(output) == n_expected:
                break
        else:
            raise AssertionError(
                "number of combinations != number of expected (%s != %s)"
                % (len(output), n_expected)
            )


def test_random_choice_csc(n_samples=10000, random_state=24):
    # [FUNCTION DOC]
    # Draws n_samples (default 10000) rows with _random_choice_csc using a fixed
    # random_state (default 24) and checks the result is sparse and per-column empirical
    # class frequencies match to 1 decimal for: explicit probabilities, implicit uniform
    # probabilities over list classes, degenerate 0/1 probabilities, and single-class
    # outputs.
    # Explicit class probabilities
    classes = [np.array([0, 1]), np.array([0, 1, 2])]
    class_probabilities = [np.array([0.5, 0.5]), np.array([0.6, 0.1, 0.3])]

    got = _random_choice_csc(n_samples, classes, class_probabilities, random_state)
    assert sp.issparse(got)

    for k in range(len(classes)):
        p = np.bincount(got[:, [k]].toarray().ravel()) / float(n_samples)
        assert_array_almost_equal(class_probabilities[k], p, decimal=1)

    # Implicit class probabilities
    classes = [[0, 1], [1, 2]]  # test for array-like support
    class_probabilities = [np.array([0.5, 0.5]), np.array([0, 1 / 2, 1 / 2])]

    got = _random_choice_csc(
        n_samples=n_samples, classes=classes, random_state=random_state
    )
    assert sp.issparse(got)

    for k in range(len(classes)):
        p = np.bincount(got[:, [k]].toarray().ravel()) / float(n_samples)
        assert_array_almost_equal(class_probabilities[k], p, decimal=1)

    # Edge case probabilities 1.0 and 0.0
    classes = [np.array([0, 1]), np.array([0, 1, 2])]
    class_probabilities = [np.array([0.0, 1.0]), np.array([0.0, 1.0, 0.0])]

    got = _random_choice_csc(n_samples, classes, class_probabilities, random_state)
    assert sp.issparse(got)

    for k in range(len(classes)):
        p = (
            np.bincount(
                got[:, [k]].toarray().ravel(), minlength=len(class_probabilities[k])
            )
            / n_samples
        )
        assert_array_almost_equal(class_probabilities[k], p, decimal=1)

    # One class target data
    classes = [[1], [0]]  # test for array-like support
    class_probabilities = [np.array([0.0, 1.0]), np.array([1.0])]

    got = _random_choice_csc(
        n_samples=n_samples, classes=classes, random_state=random_state
    )
    assert sp.issparse(got)

    for k in range(len(classes)):
        p = np.bincount(got[:, [k]].toarray().ravel()) / n_samples
        assert_array_almost_equal(class_probabilities[k], p, decimal=1)


def test_random_choice_csc_errors():
    # [FUNCTION DOC]
    # Verifies _random_choice_csc raises ValueError for mismatched classes/probabilities
    # lengths, string class labels, float class labels and probabilities not summing to
    # 1.
    # the length of an array in classes and class_probabilities is mismatched
    classes = [np.array([0, 1]), np.array([0, 1, 2, 3])]
    class_probabilities = [np.array([0.5, 0.5]), np.array([0.6, 0.1, 0.3])]
    with pytest.raises(ValueError):
        _random_choice_csc(4, classes, class_probabilities, 1)

    # the class dtype is not supported
    classes = [np.array(["a", "1"]), np.array(["z", "1", "2"])]
    class_probabilities = [np.array([0.5, 0.5]), np.array([0.6, 0.1, 0.3])]
    with pytest.raises(ValueError):
        _random_choice_csc(4, classes, class_probabilities, 1)

    # the class dtype is not supported
    classes = [np.array([4.2, 0.1]), np.array([0.1, 0.2, 9.4])]
    class_probabilities = [np.array([0.5, 0.5]), np.array([0.6, 0.1, 0.3])]
    with pytest.raises(ValueError):
        _random_choice_csc(4, classes, class_probabilities, 1)

    # Given probabilities don't sum to 1
    classes = [np.array([0, 1]), np.array([0, 1, 2])]
    class_probabilities = [np.array([0.5, 0.6]), np.array([0.6, 0.1, 0.3])]
    with pytest.raises(ValueError):
        _random_choice_csc(4, classes, class_probabilities, 1)


def test_our_rand_r():
    # [FUNCTION DOC]
    # Checks _our_rand_r_py returns known reference values for seeds 1273642419 and 0.
    assert 131541053 == _our_rand_r_py(1273642419)
    assert 270369 == _our_rand_r_py(0)
