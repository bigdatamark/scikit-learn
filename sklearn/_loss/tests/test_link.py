# [MODULE DOC]
# Tests for sklearn._loss.link: Interval validation and membership, the
# _inclusive_low_high helper, round-trip consistency of every link in _LINKS, and
# agreement of Array API inputs with NumPy results.
import numpy as np
import pytest
from numpy.testing import assert_allclose

from sklearn import config_context
from sklearn._loss.link import (
    _LINKS,
    HalfLogitLink,
    Interval,
    MultinomialLogit,
    _inclusive_low_high,
)
from sklearn.utils._array_api import (
    _atol_for_type,
    move_to,
    yield_namespace_device_dtype_combinations,
)
from sklearn.utils._testing import _array_api_for_tests

LINK_FUNCTIONS = list(_LINKS.values())


def test_interval_raises():
    # [FUNCTION DOC]
    # Verifies that constructing Interval(1, 0, ...) raises ValueError with the exact
    # "low <= high" message.
    """Test that interval with low > high raises ValueError."""
    with pytest.raises(
        ValueError, match="One must have low <= high; got low=1, high=0."
    ):
        Interval(1, 0, False, False)


@pytest.mark.parametrize(
    "interval",
    [
        Interval(0, 1, False, False),
        Interval(0, 1, False, True),
        Interval(0, 1, True, False),
        Interval(0, 1, True, True),
        Interval(-float("inf"), float("inf"), False, False),
        Interval(-float("inf"), float("inf"), False, True),
        Interval(-float("inf"), float("inf"), True, False),
        Interval(-float("inf"), float("inf"), True, True),
        Interval(-10, -1, False, False),
        Interval(-10, -1, False, True),
        Interval(-10, -1, True, False),
        Interval(-10, -1, True, True),
    ],
)
def test_is_in_range(interval):
    # [FUNCTION DOC]
    # For bounded, unbounded and negative intervals with all inclusiveness combos,
    # verifies that a linspace between the _inclusive_low_high values is included, and
    # that appending low, high or both is included exactly when the corresponding ends
    # are inclusive (infinite bounds are never included as values are checked
    # against strict/non-strict comparisons with inf).
    # Inputs: interval - parametrized Interval.
    """Test that low and high are always within the interval used for linspace."""
    low, high = _inclusive_low_high(interval)
    x = np.linspace(low, high, num=10)

    assert interval.includes(x)

    # x contains lower bound
    assert interval.includes(np.r_[x, interval.low]) == interval.low_inclusive

    # x contains upper bound
    assert interval.includes(np.r_[x, interval.high]) == interval.high_inclusive

    # x contains upper and lower bound
    assert interval.includes(np.r_[x, interval.low, interval.high]) == (
        interval.low_inclusive and interval.high_inclusive
    )


@pytest.mark.parametrize("link", LINK_FUNCTIONS)
def test_link_inverse_identity(link, global_random_seed):
    # [FUNCTION DOC]
    # For every link class, draws random raw predictions (uniform in [-20, 20], or
    # [-10, 10] for HalfLogitLink; shape (100, 10) and symmetrized for the multiclass
    # link) and checks with assert_allclose that link(inverse(raw)) recovers raw and
    # inverse(link(y_pred)) recovers y_pred.
    # Inputs: link - link class; global_random_seed - fixture seed.
    """Test that link of inverse gives identity."""
    rng = np.random.RandomState(global_random_seed)
    link = link()
    n_samples, n_classes = 100, None
    # The values for `raw_prediction` are limited from -20 to 20 because in the
    # class `LogitLink` the term `expit(x)` comes very close to 1 for large
    # positive x and therefore loses precision.
    if link.is_multiclass:
        n_classes = 10
        raw_prediction = rng.uniform(low=-20, high=20, size=(n_samples, n_classes))
        if isinstance(link, MultinomialLogit):
            raw_prediction = link.symmetrize_raw_prediction(raw_prediction)
    elif isinstance(link, HalfLogitLink):
        raw_prediction = rng.uniform(low=-10, high=10, size=(n_samples))
    else:
        raw_prediction = rng.uniform(low=-20, high=20, size=(n_samples))

    assert_allclose(link.link(link.inverse(raw_prediction)), raw_prediction)
    y_pred = link.inverse(raw_prediction)
    assert_allclose(link.inverse(link.link(y_pred)), y_pred)


@pytest.mark.parametrize(
    "namespace, device_name, dtype_name",
    yield_namespace_device_dtype_combinations(),
)
@pytest.mark.parametrize("link", LINK_FUNCTIONS)
def test_link_inverse_array_api(
    namespace, device_name, dtype_name, link, global_random_seed
):
    # [FUNCTION DOC]
    # For every Array API namespace/device/dtype combination and every link, checks
    # under array_api_dispatch that inverse and link computed on namespace arrays match
    # the NumPy results after moving back to CPU. Non-float64 dtypes halve the inputs to
    # avoid overflow and use looser rtol (1e-3 multiclass, 1e-4 otherwise).
    # Skips (via _array_api_for_tests) when the namespace/device is unavailable.
    # KNOWN ISSUE: atol is only applied to the link comparison, not to the inverse one.
    """Test that link and inverse link give same result for array API inputs."""
    rng = np.random.RandomState(global_random_seed)
    link = link()
    n_samples, n_classes = 100, None
    # The values for `raw_prediction` are limited from -20 to 20 because in the
    # class `LogitLink` the term `expit(x)` comes very close to 1 for large
    # positive x and therefore loses precision.
    if link.is_multiclass:
        n_classes = 10
        raw_prediction = rng.uniform(low=-20, high=20, size=(n_samples, n_classes))
        if isinstance(link, MultinomialLogit):
            raw_prediction = link.symmetrize_raw_prediction(raw_prediction)
    elif isinstance(link, HalfLogitLink):
        raw_prediction = rng.uniform(low=-10, high=10, size=(n_samples))
    else:
        raw_prediction = rng.uniform(low=-20, high=20, size=(n_samples))

    xp, device = _array_api_for_tests(namespace, device_name, dtype_name)
    if dtype_name != "float64":
        raw_prediction *= 0.5  # avoid overflow
        rtol = 1e-3 if n_classes else 1e-4
    else:
        rtol = 1e-8
    atol = _atol_for_type(dtype_name)

    with config_context(array_api_dispatch=True):
        raw_prediction_xp = xp.asarray(raw_prediction.astype(dtype_name), device=device)
        assert_allclose(
            move_to(link.inverse(raw_prediction_xp), xp=np, device="cpu"),
            link.inverse(raw_prediction),
            rtol=rtol,
        )

        y_pred = link.inverse(raw_prediction)
        y_pred_xp = xp.asarray(y_pred.astype(dtype_name), device=device)
        assert_allclose(
            move_to(link.link(y_pred_xp), xp=np, device="cpu"),
            link.link(y_pred),
            rtol=rtol,
            atol=atol,
        )
