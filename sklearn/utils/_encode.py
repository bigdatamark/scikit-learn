# [MODULE DOC]
# HELPER: Low-level encoding utilities shared by label/ordinal/one-hot encoders: computing
# sorted unique values with consistent handling of missing values (None and NaN),
# mapping values to integer codes, detecting unknown categories and counting occurrences.
# Object/string inputs use pure-Python set/dict logic; numeric inputs use array API ops.
# Convention: missing values are sorted to the end, None before NaN, and NaN is unique.
# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

from collections import Counter
from contextlib import suppress
from typing import NamedTuple

import numpy as np

from sklearn.utils._array_api import _isin, device, get_namespace, xpx
from sklearn.utils._missing import is_scalar_nan


def _unique(values, *, return_inverse=False, return_counts=False):
    # [FUNCTION DOC]
    # HELPER: Dispatches to _unique_python for object dtype and _unique_np otherwise.
    # Inputs: values - 1d array; return_inverse, return_counts - bools.
    # Outputs: uniques alone, or a tuple (uniques[, inverse][, counts]).
    # Exceptions: TypeError from _unique_python for mixed, non-comparable object values.
    """Helper function to find unique values with support for python objects.

    Uses pure python method for object dtype, and numpy method for
    all other dtypes.

    Parameters
    ----------
    values : ndarray
        Values to check for unknowns.

    return_inverse : bool, default=False
        If True, also return the indices of the unique values.

    return_counts : bool, default=False
        If True, also return the number of times each unique item appears in
        values.

    Returns
    -------
    unique : ndarray
        The sorted unique values.

    unique_inverse : ndarray
        The indices to reconstruct the original array from the unique array.
        Only provided if `return_inverse` is True.

    unique_counts : ndarray
        The number of times each of the unique values comes up in the original
        array. Only provided if `return_counts` is True.
    """
    if values.dtype == object:
        return _unique_python(
            values, return_inverse=return_inverse, return_counts=return_counts
        )
    # numerical
    return _unique_np(
        values, return_inverse=return_inverse, return_counts=return_counts
    )


def _unique_np(values, return_inverse=False, return_counts=False):
    # [FUNCTION DOC]
    # HELPER: Array API unique using unique_all / unique_inverse / unique_counts /
    # unique_values depending on the flags. Because NaNs are not deduplicated, if the last
    # unique is NaN, truncates uniques after the first NaN (found by searchsorted), maps
    # inverse indices beyond it to that position and sums the NaN counts into one entry.
    # Inputs/Outputs: as _unique (numeric arrays, any array API namespace).
    # Side effects: mutates the freshly computed inverse array in place (not the input).
    """Helper function to find unique values for numpy arrays that correctly
    accounts for nans. See `_unique` documentation for details."""
    xp, _ = get_namespace(values)

    inverse, counts = None, None

    if return_inverse and return_counts:
        uniques, _, inverse, counts = xp.unique_all(values)
    elif return_inverse:
        uniques, inverse = xp.unique_inverse(values)
    elif return_counts:
        uniques, counts = xp.unique_counts(values)
    else:
        uniques = xp.unique_values(values)

    # np.unique will have duplicate missing values at the end of `uniques`
    # here we clip the nans and remove it from uniques
    if uniques.size and is_scalar_nan(uniques[-1]):
        nan_idx = xp.searchsorted(uniques, xp.nan)
        uniques = uniques[: nan_idx + 1]
        if return_inverse:
            inverse[inverse > nan_idx] = nan_idx

        if return_counts:
            counts[nan_idx] = xp.sum(counts[nan_idx:])
            counts = counts[: nan_idx + 1]

    ret = (uniques,)

    if return_inverse:
        ret += (inverse,)

    if return_counts:
        ret += (counts,)

    return ret[0] if len(ret) == 1 else ret


class MissingValues(NamedTuple):
    # [CLASS DOC]
    # HELPER: NamedTuple recording whether a set of values contained NaN (nan) and/or None
    # (none).
    """Data class for missing data information"""

    nan: bool
    none: bool

    def to_list(self):
        # [FUNCTION DOC]
        # Returns a list containing None (if none) followed by np.nan (if nan).
        """Convert tuple to a list where None is always first."""
        output = []
        if self.none:
            output.append(None)
        if self.nan:
            output.append(np.nan)
        return output


def _extract_missing(values):
    # [FUNCTION DOC]
    # HELPER: Splits a set into non-missing values and a MissingValues summary. Missing
    # means None or a scalar NaN.
    # Inputs: values - set.
    # Outputs: (set without missing values, MissingValues); returns the same set object
    # when nothing is missing, a new set otherwise.
    # KNOWN ISSUE: assumes at most one NaN object besides None; distinct NaN objects in a set
    # are all removed, which is the desired effect, but the comment's reasoning is loose.
    """Extract missing values from `values`.

    Parameters
    ----------
    values: set
        Set of values to extract missing from.

    Returns
    -------
    output: set
        Set with missing values extracted.

    missing_values: MissingValues
        Object with missing value information.
    """
    missing_values_set = {
        value for value in values if value is None or is_scalar_nan(value)
    }

    if not missing_values_set:
        return values, MissingValues(nan=False, none=False)

    if None in missing_values_set:
        if len(missing_values_set) == 1:
            output_missing_values = MissingValues(nan=False, none=True)
        else:
            # If there is more than one missing value, then it has to be
            # float('nan') or np.nan
            output_missing_values = MissingValues(nan=True, none=True)
    else:
        output_missing_values = MissingValues(nan=True, none=False)

    # create set without the missing values
    output = values - missing_values_set
    return output, output_missing_values


class _nandict(dict):
    # [CLASS DOC]
    # HELPER: dict whose lookups of any NaN key fall back to the value stored under the first
    # NaN key of the initial mapping (NaN != NaN breaks normal hashing equality).
    """Dictionary with support for nans."""

    def __init__(self, mapping):
        # [FUNCTION DOC]
        # Builds the dict from mapping and records the value of the first NaN key as nan_value
        # (attribute set only if such a key exists).
        super().__init__(mapping)
        for key, value in mapping.items():
            if is_scalar_nan(key):
                self.nan_value = value
                break

    def __missing__(self, key):
        # [FUNCTION DOC]
        # Returns nan_value for a missing NaN key when available; otherwise raises KeyError(key).
        if hasattr(self, "nan_value") and is_scalar_nan(key):
            return self.nan_value
        raise KeyError(key)


def _map_to_integer(values, uniques):
    # [FUNCTION DOC]
    # HELPER: Encodes each element of values as its position in uniques using a _nandict
    # lookup table, returned as an array in the namespace/device of values.
    # Exceptions: KeyError for values absent from uniques.
    """Map values based on its position in uniques."""
    xp, _ = get_namespace(values, uniques)
    table = _nandict({val: i for i, val in enumerate(uniques)})
    return xp.asarray([table[v] for v in values], device=device(values))


def _unique_python(values, *, return_inverse, return_counts):
    # [FUNCTION DOC]
    # HELPER: Unique for object arrays: builds a set, strips missing values, sorts the rest,
    # appends None then NaN as present, and converts back to an array of values.dtype.
    # Optionally appends inverse codes (_map_to_integer) and counts (_get_counts).
    # Exceptions: TypeError listing the encountered type names when sorting fails because of
    # mixed types (e.g. str and int).
    # Only used in `_uniques`, see docstring there for details
    try:
        uniques_set = set(values)
        uniques_set, missing_values = _extract_missing(uniques_set)

        uniques = sorted(uniques_set)
        uniques.extend(missing_values.to_list())
        uniques = np.array(uniques, dtype=values.dtype)
    except TypeError:
        types = sorted(t.__qualname__ for t in set(type(v) for v in values))
        raise TypeError(
            "Encoders require their input argument must be uniformly "
            f"strings or numbers. Got {types}"
        )
    ret = (uniques,)

    if return_inverse:
        ret += (_map_to_integer(values, uniques),)

    if return_counts:
        ret += (_get_counts(values, uniques),)

    return ret[0] if len(ret) == 1 else ret


def _encode(values, *, uniques, check_unknown=True):
    # [FUNCTION DOC]
    # HELPER: Encodes values to integer codes in [0, len(uniques) - 1]. Non-numeric dtypes
    # use a dictionary lookup; numeric dtypes optionally check for unknowns then use
    # searchsorted on the (assumed sorted) uniques.
    # Inputs: values, uniques - 1d arrays; check_unknown - bool (only for numeric).
    # Outputs: integer array of codes.
    # Exceptions: ValueError mentioning previously unseen labels when unknowns are found.
    # KNOWN ISSUE: with check_unknown=False and numeric unknowns, searchsorted silently
    # returns wrong (possibly out of range) codes.
    """Helper function to encode values into [0, n_uniques - 1].

    Uses pure python method for object dtype, and numpy method for
    all other dtypes.
    The numpy method has the limitation that the `uniques` need to
    be sorted. Importantly, this is not checked but assumed to already be
    the case. The calling method needs to ensure this for all non-object
    values.

    Parameters
    ----------
    values : ndarray
        Values to encode.
    uniques : ndarray
        The unique values in `values`. If the dtype is not object, then
        `uniques` needs to be sorted.
    check_unknown : bool, default=True
        If True, check for values in `values` that are not in `unique`
        and raise an error. This is ignored for object dtype, and treated as
        True in this case. This parameter is useful for
        _BaseEncoder._transform() to avoid calling _check_unknown()
        twice.

    Returns
    -------
    encoded : ndarray
        Encoded values
    """
    xp, _ = get_namespace(values, uniques)
    if not xp.isdtype(values.dtype, "numeric"):
        try:
            return _map_to_integer(values, uniques)
        except KeyError as e:
            raise ValueError(f"y contains previously unseen labels: {e}")
    else:
        if check_unknown:
            diff = _check_unknown(values, uniques)
            if diff:
                raise ValueError(f"y contains previously unseen labels: {diff}")
        return xp.searchsorted(uniques, values)


def _check_unknown(values, known_values, return_mask=False):
    # [FUNCTION DOC]
    # HELPER: Finds values not present in known_values. Non-numeric path: set difference of
    # non-missing values, plus None/NaN if present in values but not in known_values; mask
    # built per element with is_valid. Numeric path: setdiff1d of unique values, mask via
    # _isin; if known_values contains NaN, NaN is removed from diff and NaN positions set valid.
    # Inputs: values, known_values (unique) arrays; return_mask - bool.
    # Outputs: diff (list), plus valid_mask boolean array when return_mask.
    # KNOWN ISSUE: the numeric branch calls isnan on known_values, which fails for integer
    # dtypes in strict array API namespaces; also docstring typo 'know_values'.
    """
    Helper function to check for unknowns in values to be encoded.

    Uses pure python method for object dtype, and numpy method for
    all other dtypes.

    Parameters
    ----------
    values : array
        Values to check for unknowns.
    known_values : array
        Known values. Must be unique.
    return_mask : bool, default=False
        If True, return a mask of the same shape as `values` indicating
        the valid values.

    Returns
    -------
    diff : list
        The unique values present in `values` and not in `know_values`.
    valid_mask : boolean array
        Additionally returned if ``return_mask=True``.

    """
    xp, _ = get_namespace(values, known_values)
    valid_mask = None

    if not xp.isdtype(values.dtype, "numeric"):
        values_set = set(values)
        values_set, missing_in_values = _extract_missing(values_set)

        uniques_set = set(known_values)
        uniques_set, missing_in_uniques = _extract_missing(uniques_set)
        diff = values_set - uniques_set

        nan_in_diff = missing_in_values.nan and not missing_in_uniques.nan
        none_in_diff = missing_in_values.none and not missing_in_uniques.none

        def is_valid(value):
            # [FUNCTION DOC]
            # HELPER: True if value is a known non-missing value, or is None/NaN while the known
            # values contain None/NaN respectively.
            return (
                value in uniques_set
                or (missing_in_uniques.none and value is None)
                or (missing_in_uniques.nan and is_scalar_nan(value))
            )

        if return_mask:
            if diff or nan_in_diff or none_in_diff:
                valid_mask = xp.array([is_valid(value) for value in values])
            else:
                valid_mask = xp.ones(len(values), dtype=xp.bool)

        diff = list(diff)
        if none_in_diff:
            diff.append(None)
        if nan_in_diff:
            diff.append(np.nan)
    else:
        unique_values = xp.unique_values(values)
        diff = xpx.setdiff1d(unique_values, known_values, assume_unique=True, xp=xp)
        if return_mask:
            if diff.size:
                valid_mask = _isin(values, known_values, xp)
            else:
                valid_mask = xp.ones(len(values), dtype=xp.bool)

        # check for nans in the known_values
        if xp.any(xp.isnan(known_values)):
            diff_is_nan = xp.isnan(diff)
            if xp.any(diff_is_nan):
                # removes nan from valid_mask
                if diff.size and return_mask:
                    is_nan = xp.isnan(values)
                    valid_mask[is_nan] = 1

                # remove nan from diff
                diff = diff[~diff_is_nan]
        diff = list(diff)

    if return_mask:
        return diff, valid_mask
    return diff


class _NaNCounter(Counter):
    # [CLASS DOC]
    # HELPER: Counter that counts NaN items separately (nan_count) so any NaN key returns the
    # total NaN count.
    """Counter with support for nan values."""

    def __init__(self, items):
        # [FUNCTION DOC]
        # Counts items, routing NaNs through _generate_items into nan_count.
        super().__init__(self._generate_items(items))

    def _generate_items(self, items):
        # [FUNCTION DOC]
        # HELPER: Yields non-NaN items; for each NaN increments nan_count (created lazily).
        # Side effects: sets/increments self.nan_count.
        """Generate items without nans. Stores the nan counts separately."""
        for item in items:
            if not is_scalar_nan(item):
                yield item
                continue
            if not hasattr(self, "nan_count"):
                self.nan_count = 0
            self.nan_count += 1

    def __missing__(self, key):
        # [FUNCTION DOC]
        # Returns nan_count for a NaN key when any NaN was seen, otherwise raises KeyError.
        # KNOWN ISSUE: overrides Counter's default of returning 0 for missing keys; callers
        # must suppress KeyError (as _get_counts does).
        if hasattr(self, "nan_count") and is_scalar_nan(key):
            return self.nan_count
        raise KeyError(key)


def _get_counts(values, uniques):
    # [FUNCTION DOC]
    # HELPER: Counts occurrences of each element of uniques in values, in uniques order.
    # Object/str dtype: uses _NaNCounter, missing entries count 0. Numeric: gets unique values
    # and counts via _unique_np, locates which uniques occur (treating trailing NaN as
    # present when both end in NaN), and scatters the counts via searchsorted.
    # Outputs: int64 array of length len(uniques).
    # KNOWN ISSUE: numeric path indexes unique_values[-1], raising IndexError for empty
    # values; it uses NumPy functions directly, not the array API namespace.
    """Get the count of each of the `uniques` in `values`.

    The counts will use the order passed in by `uniques`. For non-object dtypes,
    `uniques` is assumed to be sorted and `np.nan` is at the end.
    """
    if values.dtype.kind in "OU":
        counter = _NaNCounter(values)
        output = np.zeros(len(uniques), dtype=np.int64)
        for i, item in enumerate(uniques):
            with suppress(KeyError):
                output[i] = counter[item]
        return output

    unique_values, counts = _unique_np(values, return_counts=True)

    # Recorder unique_values based on input: `uniques`
    uniques_in_values = np.isin(uniques, unique_values, assume_unique=True)
    if np.isnan(unique_values[-1]) and np.isnan(uniques[-1]):
        uniques_in_values[-1] = True

    unique_valid_indices = np.searchsorted(unique_values, uniques[uniques_in_values])
    output = np.zeros_like(uniques, dtype=np.int64)
    output[uniques_in_values] = counts[unique_valid_indices]
    return output
