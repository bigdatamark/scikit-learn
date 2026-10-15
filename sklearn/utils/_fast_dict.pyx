# [MODULE DOC]
# IntFloatDict: a dict-like extension type with intp keys and float64 values,
# stored in a C++ std::map (ordered by key, logarithmic lookup/insert). Used by
# agglomerative clustering (sklearn.cluster._hierarchical_fast) to hold
# per-cluster linkage distances. Also provides argmin over such a dict.
"""
Uses C++ map containers for fast dict-like behavior with keys being
integers, and values float.
"""
# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

# C++
from cython.operator cimport dereference as deref, preincrement as inc
from libcpp.utility cimport pair
from libcpp.map cimport map as cpp_map

import numpy as np

from sklearn.utils._typedefs cimport float64_t, intp_t


###############################################################################
# An object to be used in Python

# Lookup is faster than dict (up to 10 times), and so is full traversal
# (up to 50 times), and assignment (up to 6 times), but creation is
# slower (up to 3 times). Also, a large benefit is that memory
# consumption is reduced a lot compared to a Python dict

cdef class IntFloatDict:
    # [CLASS DOC]
    # Ordered mapping intp -> float64 backed by a C++ std::map. Supports len,
    # item get/set, iteration in increasing key order, bulk export to arrays,
    # update from another IntFloatDict, copy, and insert-if-absent (append).

    def __init__(
        self,
        intp_t[:] keys,
        float64_t[:] values,
    ):
        # [FUNCTION DOC]
        # Behavior: insert keys[i] -> values[i] for i in range(len(values)); later
        # duplicates overwrite earlier ones.
        # Inputs: keys (intp 1d), values (float64 1d); should have equal length.
        # Outputs: none. Exceptions: buffer type errors from memoryview conversion.
        # KNOWN ISSUE: lengths are not checked; if keys is shorter than values, reads go
        # out of bounds unless bounds checking is enabled (comment in code admits this).
        cdef int i
        cdef int size = values.size
        # Should check that sizes for keys and values are equal, and
        # after should boundcheck(False)
        for i in range(size):
            self.my_map[keys[i]] = values[i]

    def __len__(self):
        # [FUNCTION DOC]
        # Return the number of entries in the map.
        return self.my_map.size()

    def __getitem__(self, int key):
        # [FUNCTION DOC]
        # Behavior: look up key in the map. Inputs: key (C int).
        # Outputs: the float value. Exceptions: KeyError if absent.
        cdef cpp_map[intp_t, float64_t].iterator it = self.my_map.find(key)
        if it == self.my_map.end():
            # The key is not in the dict
            raise KeyError('%i' % key)
        return deref(it).second

    def __setitem__(self, int key, float value):
        # [FUNCTION DOC]
        # Behavior: set map[key] = value (insert or overwrite).
        # Inputs: key (C int), value (C float, so precision is reduced to single
        # precision before being stored as float64). Exceptions: none.
        self.my_map[key] = value

    # Cython 0.20 generates buggy code below. Commenting this out for now
    # and relying on the to_arrays method
    # def __iter__(self):
    #     cdef cpp_map[intp_t, float64_t].iterator it = self.my_map.begin()
    #     cdef cpp_map[intp_t, float64_t].iterator end = self.my_map.end()
    #     while it != end:
    #         yield deref(it).first, deref(it).second
    #         inc(it)

    def __iter__(self):
        # [FUNCTION DOC]
        # Behavior: generator yielding (key, value) pairs in increasing key order. It
        # first snapshots the map into newly allocated arrays, so later mutation does
        # not affect the iteration. Exceptions: none. Side effects: allocates arrays.
        cdef int size = self.my_map.size()
        cdef intp_t [:] keys = np.empty(size, dtype=np.intp)
        cdef float64_t [:] values = np.empty(size, dtype=np.float64)
        self._to_arrays(keys, values)
        cdef int idx
        cdef intp_t key
        cdef float64_t value
        for idx in range(size):
            key = keys[idx]
            value = values[idx]
            yield key, value

    def to_arrays(self):
        # [FUNCTION DOC]
        # Behavior: return (keys, values) NumPy arrays (intp, float64) of length
        # len(self) holding the map contents in increasing key order.
        # Exceptions: none. Side effects: allocates two arrays.
        """Return the key, value representation of the IntFloatDict
           object.

           Returns
           =======
           keys : ndarray, shape (n_items, ), dtype=int
                The indices of the data points
           values : ndarray, shape (n_items, ), dtype=float
                The values of the data points
        """
        cdef int size = self.my_map.size()
        keys = np.empty(size, dtype=np.intp)
        values = np.empty(size, dtype=np.float64)
        self._to_arrays(keys, values)
        return keys, values

    cdef _to_arrays(self, intp_t [:] keys, float64_t [:] values):
        # [FUNCTION DOC]
        # HELPER: internal fill routine used by to_arrays and __iter__.
        # Behavior: walk the map in key order writing keys[i], values[i].
        # Inputs: keys, values writable memoryviews of length at least len(self).
        # Outputs: keys and values written in place. Exceptions: none checked.
        # Internal version of to_arrays that takes already-initialized arrays
        cdef cpp_map[intp_t, float64_t].iterator it = self.my_map.begin()
        cdef cpp_map[intp_t, float64_t].iterator end = self.my_map.end()
        cdef int index = 0
        while it != end:
            keys[index] = deref(it).first
            values[index] = deref(it).second
            inc(it)
            index += 1

    def update(self, IntFloatDict other):
        # [FUNCTION DOC]
        # Behavior: for every entry in other, set self[key] = value (overwrite).
        # Inputs: other, an IntFloatDict. Outputs: none; mutates self.
        # Exceptions: TypeError if other is not an IntFloatDict.
        cdef cpp_map[intp_t, float64_t].iterator it = other.my_map.begin()
        cdef cpp_map[intp_t, float64_t].iterator end = other.my_map.end()
        while it != end:
            self.my_map[deref(it).first] = deref(it).second
            inc(it)

    def copy(self):
        # [FUNCTION DOC]
        # Behavior: return a new IntFloatDict whose map is a deep copy of self's map
        # (C++ map assignment copies). Exceptions: none.
        cdef IntFloatDict out_obj = IntFloatDict.__new__(IntFloatDict)
        # The '=' operator is a copy operator for C++ maps
        out_obj.my_map = self.my_map
        return out_obj

    def append(self, intp_t key, float64_t value):
        # [FUNCTION DOC]
        # Behavior: insert key -> value only if key is not already present (C++ map
        # insert semantics; existing values are kept, unlike __setitem__).
        # Inputs: key (intp), value (float64). Outputs: none; mutates self.
        # Construct our arguments
        cdef pair[intp_t, float64_t] args
        args.first = key
        args.second = value
        self.my_map.insert(args)


###############################################################################
# operation on dict

def argmin(IntFloatDict d):
    # [FUNCTION DOC]
    # Behavior: scan the dict and return (key, value) of the smallest value; ties
    # keep the smallest key (strict less-than, key order traversal).
    # Inputs: d, an IntFloatDict. Outputs: (min_key, min_value); for an empty dict
    # returns (-1, inf). Exceptions: TypeError if d is not an IntFloatDict.
    cdef cpp_map[intp_t, float64_t].iterator it = d.my_map.begin()
    cdef cpp_map[intp_t, float64_t].iterator end = d.my_map.end()
    cdef intp_t min_key = -1
    cdef float64_t min_value = np.inf
    while it != end:
        if deref(it).second < min_value:
            min_value = deref(it).second
            min_key = deref(it).first
        inc(it)
    return min_key, min_value
