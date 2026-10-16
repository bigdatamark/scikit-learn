# [MODULE DOC]
# Tests for sklearn.utils.deprecation: the deprecated decorator on classes,
# methods, properties, __init__, __new__ and functions, the _is_deprecated
# helper, pickling and preserved class signatures. Defines mock classes at
# module level (decorating them emits no warning until use).
# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause


import pickle
from inspect import signature

import pytest

from sklearn.utils.deprecation import _is_deprecated, deprecated


@deprecated("qwerty")
class MockClass1:
    # [CLASS DOC]
    # HELPER: Empty class deprecated with message "qwerty".
    pass


class MockClass2:
    # [CLASS DOC]
    # HELPER: Class with a deprecated method and a deprecated property.
    @deprecated("mockclass2_method")
    def method(self):
        # [FUNCTION DOC]
        # HELPER: Deprecated no-op method.
        pass

    @deprecated("n_features_ is deprecated")
    @property
    def n_features_(self):
        # [FUNCTION DOC]
        # HELPER: Deprecated property returning 10.
        """Number of input features."""
        return 10


class MockClass3:
    # [CLASS DOC]
    # HELPER: Class whose __init__ is deprecated with the default message.
    @deprecated()
    def __init__(self):
        # [FUNCTION DOC]
        # HELPER: Deprecated no-op constructor.
        pass


class MockClass4:
    # [CLASS DOC]
    # HELPER: Plain, non-deprecated class.
    pass


class MockClass5(MockClass1):
    # [CLASS DOC]
    # HELPER: Subclass of the deprecated MockClass1 with its own __init__.
    """Inherit from deprecated class but does not call super().__init__."""

    def __init__(self, a):
        # [FUNCTION DOC]
        # HELPER: Stores a without calling the parent constructor.
        self.a = a


@deprecated("a message")
class MockClass6:
    # [CLASS DOC]
    # HELPER: Deprecated class overriding __new__, to check the decorator forwards
    # constructor arguments.
    """A deprecated class that overrides __new__."""

    def __new__(cls, *args, **kwargs):
        # [FUNCTION DOC]
        # HELPER: Asserts at least one positional argument was passed, then creates the
        # instance via object.__new__ without arguments.
        assert len(args) > 0
        return super().__new__(cls)


@deprecated()
def mock_function():
    # [FUNCTION DOC]
    # HELPER: Deprecated function returning 10.
    return 10


def test_deprecated():
    # [FUNCTION DOC]
    # Verifies FutureWarnings with the expected messages for instantiating deprecated
    # classes (also via subclass and with __new__), calling a deprecated method,
    # deprecated __init__ and deprecated function (return value preserved).
    with pytest.warns(FutureWarning, match="qwerty"):
        MockClass1()
    with pytest.warns(FutureWarning, match="mockclass2_method"):
        MockClass2().method()
    with pytest.warns(FutureWarning, match="deprecated"):
        MockClass3()
    with pytest.warns(FutureWarning, match="qwerty"):
        MockClass5(42)
    with pytest.warns(FutureWarning, match="a message"):
        MockClass6(42)
    with pytest.warns(FutureWarning, match="deprecated"):
        val = mock_function()
    assert val == 10


def test_is_deprecated():
    # [FUNCTION DOC]
    # Verifies _is_deprecated detects wrapped __new__, methods, __init__ and
    # functions, and not plain __init__.
    # Test if _is_deprecated helper identifies wrapping via deprecated
    # NOTE it works only for class methods and functions
    assert _is_deprecated(MockClass1.__new__)
    assert _is_deprecated(MockClass2().method)
    assert _is_deprecated(MockClass3.__init__)
    assert not _is_deprecated(MockClass4.__init__)
    assert _is_deprecated(MockClass5.__new__)
    assert _is_deprecated(mock_function)


def test_pickle():
    # [FUNCTION DOC]
    # Verifies a deprecated function survives a pickle round trip.
    pickle.loads(pickle.dumps(mock_function))


def test_deprecated_class_signature():
    # [FUNCTION DOC]
    # Verifies a deprecated class keeps its __init__ signature (a, b, c).
    @deprecated()
    class MockClass:
        # [CLASS DOC]
        # HELPER: Locally defined deprecated class with a three-parameter constructor.
        def __init__(self, a, b=1, c=2):
            # [FUNCTION DOC]
            # HELPER: No-op constructor taking a, b=1, c=2.
            pass

    assert list(signature(MockClass).parameters.keys()) == ["a", "b", "c"]
