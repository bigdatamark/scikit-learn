# [MODULE DOC]
# Defines Bunch, the dict subclass with attribute access returned by dataset loaders and
# many sklearn functions, including support for deprecated keys that warn on access.
# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import warnings


class Bunch(dict):
    # [CLASS DOC]
    # Dict subclass where attribute get/set map to item get/set. Keeps a hidden
    # _deprecated_key_to_warnings mapping in the instance __dict__ (not in the dict items).
    """Container object exposing keys as attributes.

    Bunch objects are sometimes used as an output for functions and methods.
    They extend dictionaries by enabling values to be accessed by key,
    `bunch["value_key"]`, or by an attribute, `bunch.value_key`.

    Examples
    --------
    >>> from sklearn.utils import Bunch
    >>> b = Bunch(a=1, b=2)
    >>> b['b']
    2
    >>> b.b
    2
    >>> b.a = 3
    >>> b['a']
    3
    >>> b.c = 6
    >>> b['c']
    6
    """

    def __init__(self, **kwargs):
        # [FUNCTION DOC]
        # Initialises the dict from keyword arguments and creates an empty deprecated-key map
        # directly in __dict__ (bypassing __setattr__, which would store it as an item).
        super().__init__(kwargs)

        # Map from deprecated key to warning message
        self.__dict__["_deprecated_key_to_warnings"] = {}

    def __getitem__(self, key):
        # [FUNCTION DOC]
        # Item lookup that emits a FutureWarning with the registered message when key is a
        # deprecated key, then returns the stored value.
        # Exceptions: KeyError for missing keys.
        if key in self.__dict__.get("_deprecated_key_to_warnings", {}):
            warnings.warn(
                self._deprecated_key_to_warnings[key],
                FutureWarning,
            )
        return super().__getitem__(key)

    def _set_deprecated(self, value, *, new_key, deprecated_key, warning_message):
        # [FUNCTION DOC]
        # HELPER: Stores value under both new_key and deprecated_key and registers
        # warning_message for deprecated_key so later item/attribute access warns.
        # Side effects: mutates the dict and the deprecated-key map.
        """Set key in dictionary to be deprecated with its warning message."""
        self.__dict__["_deprecated_key_to_warnings"][deprecated_key] = warning_message
        self[new_key] = self[deprecated_key] = value

    def __setattr__(self, key, value):
        # [FUNCTION DOC]
        # Attribute assignment stores the value as a dict item instead of in __dict__.
        self[key] = value

    def __dir__(self):
        # [FUNCTION DOC]
        # Returns the dict keys so tab completion lists the stored items.
        # KNOWN ISSUE: returns a keys view rather than a list of all attributes, so regular
        # methods are hidden from dir().
        return self.keys()

    def __getattr__(self, key):
        # [FUNCTION DOC]
        # Fallback attribute lookup (only reached when normal lookup fails): returns the item
        # with that key via __getitem__ (so deprecated keys also warn).
        # Exceptions: AttributeError when the key is absent.
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def __setstate__(self, state):
        # [FUNCTION DOC]
        # No-op unpickling hook: ignores any pickled __dict__ state (old 0.16 pickles); dict
        # items are restored by dict pickling itself.
        # KNOWN ISSUE: as a consequence, an unpickled Bunch lacks _deprecated_key_to_warnings;
        # __getitem__ tolerates that, but _set_deprecated would raise KeyError on it.
        # Bunch pickles generated with scikit-learn 0.16.* have a non
        # empty __dict__. This causes a surprising behaviour when
        # loading these pickles scikit-learn 0.17: reading bunch.key
        # uses __dict__ but assigning to bunch.key use __setattr__ and
        # only changes bunch['key']. More details can be found at:
        # https://github.com/scikit-learn/scikit-learn/issues/6196.
        # Overriding __setstate__ to be a noop has the effect of
        # ignoring the pickled __dict__
        pass
