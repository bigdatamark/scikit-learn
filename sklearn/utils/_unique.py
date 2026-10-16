# [MODULE DOC]
# HELPER: Caching of the unique values of label arrays. Unique values are stored in
# the numpy dtype metadata of a view of the array so nested calls can reuse them
# instead of recomputing them; non-numpy arrays fall back to the array namespace
# unique_values.
# Side effects at import: none.
# [MODULE DOC]
# HELPER: Caching of the unique values of label arrays. Unique values are stored in
# the numpy dtype metadata of a view of the array so nested calls can reuse them
# instead of recomputing them; non-numpy arrays fall back to the array namespace
# unique_values.
# Side effects at import: none.
# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np

from sklearn.utils._array_api import get_namespace


def _attach_unique(y):
    # [FUNCTION DOC]
    # HELPER: Returns a view of a single array whose dtype metadata carries its unique
    # values.
    # Steps: return y unchanged if it is not a numpy ndarray; return y unchanged if
    # its dtype metadata already contains "unique" (a TypeError/AttributeError from a
    # missing metadata dict is swallowed); otherwise compute the sorted unique values,
    # build a copy of the dtype with metadata {"unique": values} and return a view of
    # y with that dtype.
    # Inputs: y - any object, typically a 1d numpy array of labels.
    # Outputs: y itself or a view of y sharing its memory.
    # Exceptions: none raised directly.
    # Side effects: none on y; the view shares data so later writes make the cache
    # stale.
    # [FUNCTION DOC]
    # HELPER: Returns a view of a single array whose dtype metadata carries its unique
    # values.
    # Steps: return y unchanged if it is not a numpy ndarray; return y unchanged if
    # its dtype metadata already contains "unique" (a TypeError/AttributeError from a
    # missing metadata dict is swallowed); otherwise compute the sorted unique values,
    # build a copy of the dtype with metadata {"unique": values} and return a view of
    # y with that dtype.
    # Inputs: y - any object, typically a 1d numpy array of labels.
    # Outputs: y itself or a view of y sharing its memory.
    # Exceptions: none raised directly.
    # Side effects: none on y; the view shares data so later writes make the cache
    # stale.
    """Attach unique values of y to y and return the result.

    The result is a view of y, and the metadata (unique) is not attached to y.
    """
    if not isinstance(y, np.ndarray):
        return y
    try:
        # avoid recalculating unique in nested calls.
        if "unique" in y.dtype.metadata:
            return y
    except (AttributeError, TypeError):
        pass

    unique = np.unique(y)
    unique_dtype = np.dtype(y.dtype, metadata={"unique": unique})
    return y.view(dtype=unique_dtype)


def attach_unique(*ys, return_tuple=False):
    # [FUNCTION DOC]
    # Applies _attach_unique to every positional array.
    # Inputs: *ys - arrays; return_tuple - bool, default False, force a tuple for one
    # input.
    # Outputs: the single view when exactly one array was given and return_tuple is
    # False, otherwise a tuple of views/originals in input order.
    # Exceptions/side effects: none.
    # [FUNCTION DOC]
    # Applies _attach_unique to every positional array.
    # Inputs: *ys - arrays; return_tuple - bool, default False, force a tuple for one
    # input.
    # Outputs: the single view when exactly one array was given and return_tuple is
    # False, otherwise a tuple of views/originals in input order.
    # Exceptions/side effects: none.
    """Attach unique values of ys to ys and return the results.

    The result is a view of y, and the metadata (unique) is not attached to y.

    IMPORTANT: The output of this function should NEVER be returned in functions.
    This is to avoid this pattern:

    .. code:: python

        y = np.array([1, 2, 3])
        y = attach_unique(y)
        y[1] = -1
        # now np.unique(y) will be different from cached_unique(y)

    Parameters
    ----------
    *ys : sequence of array-like
        Input data arrays.

    return_tuple : bool, default=False
        If True, always return a tuple even if there is only one array.

    Returns
    -------
    ys : tuple of array-like or array-like
        Input data with unique values attached.
    """
    res = tuple(_attach_unique(y) for y in ys)
    if len(res) == 1 and not return_tuple:
        return res[0]
    return res


def _cached_unique(y, xp=None):
    # [FUNCTION DOC]
    # HELPER: Returns the unique values of one array, using the dtype metadata cache
    # when present, otherwise the namespace function unique_values (resolved via
    # get_namespace unless xp is given).
    # Inputs: y - array of any supported namespace; xp - optional namespace module.
    # Outputs: array of unique values (numpy array when taken from the cache).
    # Exceptions: whatever get_namespace / unique_values raise for unsupported input.
    # Side effects: none; y is not modified.
    # KNOWN ISSUE: only AttributeError is caught here, unlike _attach_unique; this is
    # fine because the explicit None check prevents the TypeError case.
    # [FUNCTION DOC]
    # HELPER: Returns the unique values of one array, using the dtype metadata cache
    # when present, otherwise the namespace function unique_values (resolved via
    # get_namespace unless xp is given).
    # Inputs: y - array of any supported namespace; xp - optional namespace module.
    # Outputs: array of unique values (numpy array when taken from the cache).
    # Exceptions: whatever get_namespace / unique_values raise for unsupported input.
    # Side effects: none; y is not modified.
    # KNOWN ISSUE: only AttributeError is caught here, unlike _attach_unique; this is
    # fine because the explicit None check prevents the TypeError case.
    """Return the unique values of y.

    Use the cached values from dtype.metadata if present.

    This function does NOT cache the values in y, i.e. it doesn't change y.

    Call `attach_unique` to attach the unique values to y.
    """
    try:
        if y.dtype.metadata is not None and "unique" in y.dtype.metadata:
            return y.dtype.metadata["unique"]
    except AttributeError:
        # in case y is not a numpy array
        pass
    xp, _ = get_namespace(y, xp=xp)
    return xp.unique_values(y)


def cached_unique(*ys, xp=None):
    # [FUNCTION DOC]
    # Applies _cached_unique to each positional array with the shared xp.
    # Inputs: *ys - arrays; xp - optional precomputed array namespace.
    # Outputs: the unique values for a single input, else a tuple of them.
    # Exceptions/side effects: none beyond _cached_unique.
    # [FUNCTION DOC]
    # Applies _cached_unique to each positional array with the shared xp.
    # Inputs: *ys - arrays; xp - optional precomputed array namespace.
    # Outputs: the unique values for a single input, else a tuple of them.
    # Exceptions/side effects: none beyond _cached_unique.
    """Return the unique values of ys.

    Use the cached values from dtype.metadata if present.

    This function does NOT cache the values in y, i.e. it doesn't change y.

    Call `attach_unique` to attach the unique values to y.

    Parameters
    ----------
    *ys : sequence of array-like
        Input data arrays.

    xp : module, default=None
        Precomputed array namespace module. When passed, typically from a caller
        that has already performed inspection of its own inputs, skips array
        namespace inspection.

    Returns
    -------
    res : tuple of array-like or array-like
        Unique values of ys.
    """
    res = tuple(_cached_unique(y, xp=xp) for y in ys)
    if len(res) == 1:
        return res[0]
    return res
