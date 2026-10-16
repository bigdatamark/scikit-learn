# [MODULE DOC]
# Tests for the Cython WeightVector32/WeightVector64 helpers used by SGD-style solvers.
# Only dtype invariance of the wrapped weight buffers is checked here.
import numpy as np
import pytest

from sklearn.utils._weight_vector import (
    WeightVector32,
    WeightVector64,
)


@pytest.mark.parametrize(
    "dtype, WeightVector",
    [
        (np.float32, WeightVector32),
        (np.float64, WeightVector64),
    ],
)
def test_type_invariance(dtype, WeightVector):
    # [FUNCTION DOC]
    # Verifies that WeightVector32/64 keep the dtype of the weight and average-weight
    # arrays they wrap: builds random float32/float64 arrays of length 100, wraps them,
    # and asserts the w and aw memoryviews convert back to arrays of the same dtype.
    # Inputs: dtype - numpy float dtype; WeightVector - matching Cython class
    # (parametrized).
    # Side effects: uses the global numpy RNG (unseeded).
    """Check the `dtype` consistency of `WeightVector`."""
    weights = np.random.rand(100).astype(dtype)
    average_weights = np.random.rand(100).astype(dtype)

    weight_vector = WeightVector(weights, average_weights)

    assert np.asarray(weight_vector.w).dtype is np.dtype(dtype)
    assert np.asarray(weight_vector.aw).dtype is np.dtype(dtype)
