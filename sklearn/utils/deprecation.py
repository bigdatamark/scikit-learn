# [MODULE DOC]
# Deprecation helpers: the public `deprecated` decorator (for functions, classes and
# properties, emitting FutureWarning on use) and the private _is_deprecated check.
# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import functools
import warnings
from inspect import signature

__all__ = ["deprecated"]


class deprecated:
    # [CLASS DOC]
    # Decorator factory object. An instance stores the extra message; calling it with
    # the object to deprecate dispatches on the object kind (class, property, other
    # callable) and returns a wrapped object emitting FutureWarning on use.
    # KNOWN ISSUE: the docstring says a warning is added to the decorated object's
    # docstring, but no docstring modification is performed by the code.
    """Decorator to mark a function or class as deprecated.

    Issue a warning when the function is called/the class is instantiated and
    adds a warning to the docstring.

    The optional extra argument will be appended to the deprecation message
    and the docstring. Note: to use this with the default value for extra, put
    in an empty of parentheses:

    Examples
    --------
    >>> from sklearn.utils import deprecated
    >>> deprecated()
    <sklearn.utils.deprecation.deprecated object at ...>
    >>> @deprecated()
    ... def some_function(): pass

    Parameters
    ----------
    extra : str, default=''
          To be added to the deprecation messages.
    """

    # Adapted from https://wiki.python.org/moin/PythonDecoratorLibrary,
    # but with many changes.

    def __init__(self, extra=""):
        # [FUNCTION DOC]
        # Inputs: extra - str, default "", appended to the deprecation message.
        # Side effects: stores it as self.extra.
        self.extra = extra

    def __call__(self, obj):
        # [FUNCTION DOC]
        # Dispatches: a type goes to _decorate_class, a property object to
        # _decorate_property, anything else to _decorate_fun.
        # Inputs: obj - class, property or callable.
        # Outputs: the decorated object (the same class object for classes, a new
        # wrapper otherwise).
        """Call method

        Parameters
        ----------
        obj : object
        """
        if isinstance(obj, type):
            return self._decorate_class(obj)
        elif isinstance(obj, property):
            # Note that this is only triggered properly if the `deprecated`
            # decorator is placed before the `property` decorator, like so:
            #
            # @deprecated(msg)
            # @property
            # def deprecated_attribute_(self):
            #     ...
            return self._decorate_property(obj)
        else:
            return self._decorate_fun(obj)

    def _decorate_class(self, cls):
        # [FUNCTION DOC]
        # HELPER: Replaces cls.__new__ with a wrapper that issues FutureWarning("Class
        # <name> is deprecated[; extra]") on each instantiation, then calls the
        # original __new__ (passing no extra args when the original is object.__new__,
        # which rejects them). Saves the original signature of cls into
        # cls.__signature__ so introspection is unchanged.
        # Inputs: cls - class.
        # Outputs: the same class object, mutated.
        # Side effects: mutates cls.__new__ and cls.__signature__.
        msg = "Class %s is deprecated" % cls.__name__
        if self.extra:
            msg += "; %s" % self.extra

        new = cls.__new__
        sig = signature(cls)

        def wrapped(cls, *args, **kwargs):
            # [FUNCTION DOC]
            # HELPER: Replacement __new__: warns, then delegates to the saved original
            # __new__.
            # Inputs: cls - class being instantiated; *args/**kwargs - constructor
            # arguments.
            # Outputs: the new, uninitialised instance.
            # Side effects: emits FutureWarning.
            warnings.warn(msg, category=FutureWarning)
            if new is object.__new__:
                return object.__new__(cls)

            return new(cls, *args, **kwargs)

        cls.__new__ = wrapped

        wrapped.__name__ = "__new__"
        # Restore the original signature, see PEP 362.
        cls.__signature__ = sig

        return cls

    def _decorate_fun(self, fun):
        # [FUNCTION DOC]
        # HELPER: Wraps a function in a functools.wraps wrapper that emits
        # FutureWarning("Function <name> is deprecated[; extra]") and then calls it.
        # Inputs: fun - callable with a __name__.
        # Outputs: the wrapper, with __wrapped__ set to fun.
        """Decorate function fun"""

        msg = "Function %s is deprecated" % fun.__name__
        if self.extra:
            msg += "; %s" % self.extra

        @functools.wraps(fun)
        def wrapped(*args, **kwargs):
            # [FUNCTION DOC]
            # HELPER: Emits the FutureWarning and returns fun(*args, **kwargs).
            warnings.warn(msg, category=FutureWarning)
            return fun(*args, **kwargs)

        # Add a reference to the wrapped function so that we can introspect
        # on function arguments in Python 2 (already works in Python 3)
        wrapped.__wrapped__ = fun

        return wrapped

    def _decorate_property(self, prop):
        # [FUNCTION DOC]
        # HELPER: Builds a new property whose getter emits FutureWarning with message
        # self.extra, then returns the original getter's value.
        # Inputs: prop - property object.
        # Outputs: a new read-only property.
        # KNOWN ISSUE: only fget is preserved (setter/deleter are dropped) and the
        # message is just extra, so an empty extra produces a warning with an empty
        # message.
        msg = self.extra

        @property
        @functools.wraps(prop.fget)
        def wrapped(*args, **kwargs):
            # [FUNCTION DOC]
            # HELPER: Property getter: warns, then returns prop.fget(*args, **kwargs).
            warnings.warn(msg, category=FutureWarning)
            return prop.fget(*args, **kwargs)

        return wrapped


def _is_deprecated(func):
    # [FUNCTION DOC]
    # HELPER: Heuristically detects a function produced by `deprecated` by checking
    # whether any string stored in its closure cells contains the substring
    # "deprecated" (the wrapper closes over msg).
    # Inputs: func - any object.
    # Outputs: bool.
    # Exceptions: ValueError from cell_contents on an empty closure cell is not
    # handled.
    # KNOWN ISSUE: property wrappers close over extra only, so they are detected only
    # if extra happens to contain "deprecated"; unrelated closures holding such a
    # string give false positives.
    """Helper to check if func is wrapped by our deprecated decorator"""
    closures = getattr(func, "__closure__", [])
    if closures is None:
        closures = []
    is_deprecated = "deprecated" in "".join(
        [c.cell_contents for c in closures if isinstance(c.cell_contents, str)]
    )
    return is_deprecated
