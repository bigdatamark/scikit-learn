# [MODULE DOC]
# Declaration of IntFloatDict, a Python-visible extension type wrapping a C++
# std::map from intp keys to float64 values, so other Cython modules (e.g.
# hierarchical clustering) can access the underlying map directly.
# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Uses C++ map containers for fast dict-like behavior with keys being
integers, and values float.
"""

from libcpp.map cimport map as cpp_map

from sklearn.utils._typedefs cimport float64_t, intp_t


###############################################################################
# An object to be used in Python

cdef class IntFloatDict:
    # [CLASS DOC]
    # Ordered integer-to-float mapping backed by a C++ std::map attribute my_map.
    cdef cpp_map[intp_t, float64_t] my_map
    cdef _to_arrays(self, intp_t [:] keys, float64_t [:] values)
    # [FUNCTION DOC]
    # Fill preallocated keys and values arrays with the map contents in key order.
