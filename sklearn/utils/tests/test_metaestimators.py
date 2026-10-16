# [MODULE DOC]
# Tests for sklearn.utils.metaestimators.available_if: docstring propagation,
# conditional attribute presence, unbound-method calls and picklability.
import pickle

import pytest

from sklearn.utils.metaestimators import available_if


class AvailableParameterEstimator:
    # [CLASS DOC]
    # HELPER: Minimal mock estimator whose available_func method exists only when the
    # available attribute is truthy (via available_if with a lambda checking it).
    """This estimator's `available` parameter toggles the presence of a method"""

    def __init__(self, available=True, return_value=1):
        # [FUNCTION DOC]
        # HELPER: Stores the constructor arguments unchanged.
        # Inputs: available - bool (default True) controlling method availability;
        # return_value - any value (default 1) returned by available_func.
        # Side effects: sets self.available and self.return_value.
        self.available = available
        self.return_value = return_value

    @available_if(lambda est: est.available)
    def available_func(self):
        # [FUNCTION DOC]
        # HELPER: Mock method guarded by available_if; returns self.return_value.
        # Exceptions: AttributeError on access when self.available is falsy.
        """This is a mock available_if function"""
        return self.return_value


def test_available_if_docstring():
    # [FUNCTION DOC]
    # Verifies the wrapped method's docstring is preserved when accessed through the
    # class
    # __dict__ (descriptor object), through the class, and through an instance.
    assert "This is a mock available_if function" in str(
        AvailableParameterEstimator.__dict__["available_func"].__doc__
    )
    assert "This is a mock available_if function" in str(
        AvailableParameterEstimator.available_func.__doc__
    )
    assert "This is a mock available_if function" in str(
        AvailableParameterEstimator().available_func.__doc__
    )


def test_available_if():
    # [FUNCTION DOC]
    # Verifies hasattr reports available_func present for available=True and absent for
    # available=False.
    assert hasattr(AvailableParameterEstimator(), "available_func")
    assert not hasattr(AvailableParameterEstimator(available=False), "available_func")


def test_available_if_unbound_method():
    # [FUNCTION DOC]
    # Non-regression for issue 20614: calling the descriptor as an unbound method with
    # an
    # explicit instance works when available, and raises AttributeError with the
    # expected
    # message when the instance has available=False.
    # This is a non regression test for:
    # https://github.com/scikit-learn/scikit-learn/issues/20614
    # to make sure that decorated functions can be used as an unbound method,
    # for instance when monkeypatching.
    est = AvailableParameterEstimator()
    AvailableParameterEstimator.available_func(est)

    est = AvailableParameterEstimator(available=False)
    with pytest.raises(
        AttributeError,
        match="This 'AvailableParameterEstimator' has no attribute 'available_func'",
    ):
        AvailableParameterEstimator.available_func(est)


def test_available_if_methods_can_be_pickled():
    # [FUNCTION DOC]
    # Non-regression for issue 21344: pickles and unpickles the bound available_func of
    # an
    # instance and checks the unpickled callable returns the configured return_value
    # (10).
    """Check that available_if methods can be pickled.

    Non-regression test for #21344.
    """
    return_value = 10
    est = AvailableParameterEstimator(available=True, return_value=return_value)
    pickled_bytes = pickle.dumps(est.available_func)
    unpickled_func = pickle.loads(pickled_bytes)
    assert unpickled_func() == return_value
