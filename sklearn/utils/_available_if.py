# [MODULE DOC]
# Implements the available_if decorator: a descriptor that makes a method appear only
# when a user-supplied check on the instance succeeds, so hasattr reflects availability
# (used by meta-estimators to expose e.g. predict_proba only when the inner estimator has it).
# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

from functools import update_wrapper, wraps
from types import MethodType


class _AvailableIfDescriptor:
    # [CLASS DOC]
    # HELPER: Non-data descriptor wrapping a function fn. On instance access it runs check(obj)
    # and either returns fn bound to obj or raises AttributeError.
    # Attributes: fn (wrapped function), check (callable taking the instance),
    # attribute_name (name used in error messages).
    """Implements a conditional property using the descriptor protocol.

    Using this class to create a decorator will raise an ``AttributeError``
    if check(self) returns a falsey value. Note that if check raises an error
    this will also result in hasattr returning false.

    See https://docs.python.org/3/howto/descriptor.html for an explanation of
    descriptors.
    """

    def __init__(self, fn, check, attribute_name):
        # [FUNCTION DOC]
        # Stores fn, check and attribute_name.
        # Side effects: copies fn metadata (__doc__, __name__, __wrapped__ ...) onto the
        # descriptor via update_wrapper so documentation tools see the method docstring.
        self.fn = fn
        self.check = check
        self.attribute_name = attribute_name

        # update the docstring of the descriptor
        update_wrapper(self, fn)

    def _check(self, obj, owner):
        # [FUNCTION DOC]
        # Runs check on obj and raises AttributeError (message names the owner class and
        # attribute_name) when the result is falsy. Any exception raised by check is converted
        # to AttributeError, chained from the original.
        # Inputs: obj - the instance; owner - its class (only its __name__ is used).
        # Outputs: None on success.
        attr_err_msg = (
            f"This {owner.__name__!r} has no attribute {self.attribute_name!r}"
        )
        try:
            check_result = self.check(obj)
        except Exception as e:
            raise AttributeError(attr_err_msg) from e

        if not check_result:
            raise AttributeError(attr_err_msg)

    def __get__(self, obj, owner=None):
        # [FUNCTION DOC]
        # Descriptor protocol access. Instance access: runs _check and returns fn bound to obj as
        # a MethodType. Class access (obj is None): returns a plain wrapper function that runs
        # _check on its first positional argument before calling fn, so the method stays usable
        # unbound (e.g. for monkeypatching) and its docstring is reachable.
        # Exceptions: AttributeError from _check when unavailable.
        if obj is not None:
            # delegate only on instances, not the classes.
            # this is to allow access to the docstrings.
            self._check(obj, owner=owner)
            out = MethodType(self.fn, obj)

        else:
            # This makes it possible to use the decorated method as an unbound method,
            # for instance when monkeypatching.
            @wraps(self.fn)
            def out(*args, **kwargs):
                # [FUNCTION DOC]
                # HELPER: Unbound wrapper; treats the first positional argument as the instance, runs the
                # availability check on it, then forwards all arguments to fn and returns its result.
                # Exceptions: AttributeError when unavailable; IndexError if called with no arguments.
                self._check(args[0], owner=owner)
                return self.fn(*args, **kwargs)

        return out


def available_if(check):
    # [FUNCTION DOC]
    # Decorator factory: returns a callable that wraps a method fn into an
    # _AvailableIfDescriptor using check and fn's name as attribute_name.
    # Inputs: check - callable taking the instance, returning truthy when available.
    # Outputs: the decorator (a lambda) producing the descriptor.
    """An attribute that is available only if check returns a truthy value.

    Parameters
    ----------
    check : callable
        When passed the object with the decorated method, this should return
        a truthy value if the attribute is available, and either return False
        or raise an AttributeError if not available.

    Returns
    -------
    callable
        Callable makes the decorated method available if `check` returns
        a truthy value, otherwise the decorated method is unavailable.

    Examples
    --------
    >>> from sklearn.utils.metaestimators import available_if
    >>> class HelloIfEven:
    ...    def __init__(self, x):
    ...        self.x = x
    ...
    ...    def _x_is_even(self):
    ...        return self.x % 2 == 0
    ...
    ...    @available_if(_x_is_even)
    ...    def say_hello(self):
    ...        print("Hello")
    ...
    >>> obj = HelloIfEven(1)
    >>> hasattr(obj, "say_hello")
    False
    >>> obj.x = 2
    >>> hasattr(obj, "say_hello")
    True
    >>> obj.say_hello()
    Hello
    """
    return lambda fn: _AvailableIfDescriptor(fn, check, attribute_name=fn.__name__)
