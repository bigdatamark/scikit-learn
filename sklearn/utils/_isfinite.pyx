# [MODULE DOC]
# Fast check whether a contiguous 1d float32/float64 array contains NaN or
# infinite values, releasing the GIL. Used by sklearn.utils.validation
# (_assert_all_finite) to validate inputs. Returns a FiniteStatus enum code
# (all_finite = 0, has_nan = 1, has_infinite = 2).
# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

from libc.math cimport isnan, isinf
from cython cimport floating


cpdef enum FiniteStatus:
    all_finite = 0
    has_nan = 1
    has_infinite = 2


def cy_isfinite(floating[::1] a, bint allow_nan=False):
    # [FUNCTION DOC]
    # Behavior: Python entry point; releases the GIL and calls _isfinite.
    # Inputs: a, C-contiguous 1d floating array (non-empty); allow_nan, if true NaN
    # is ignored and only infinities are reported.
    # Outputs: FiniteStatus value (int). Exceptions: buffer dtype/contiguity errors
    # from memoryview conversion. Precondition: a is non-empty (bounds checking
    # is disabled in release builds; caller only passes arrays with size > 0).
    # Side effects: GIL released during the scan.
    cdef FiniteStatus result
    with nogil:
        result = _isfinite(a, allow_nan)
    return result


cdef inline FiniteStatus _isfinite(floating[::1] a, bint allow_nan) noexcept nogil:
    # [FUNCTION DOC]
    # HELPER: dispatches to the allow_nan or disable_nan scan loop.
    # Inputs: a (contiguous memoryview), allow_nan. Outputs: FiniteStatus.
    # Exceptions: none (noexcept, nogil). Taking the address of element 0 of an
    # empty array is undefined behavior in release builds (no bounds checks).
    cdef floating* a_ptr = &a[0]
    cdef Py_ssize_t length = len(a)
    if allow_nan:
        return _isfinite_allow_nan(a_ptr, length)
    else:
        return _isfinite_disable_nan(a_ptr, length)


cdef inline FiniteStatus _isfinite_allow_nan(floating* a_ptr,
                                             Py_ssize_t length) noexcept nogil:
    # [FUNCTION DOC]
    # HELPER: scan length values from a_ptr; return has_infinite on the first
    # infinity, else all_finite. NaNs are ignored. No exceptions, nogil.
    cdef Py_ssize_t i
    cdef floating v
    for i in range(length):
        v = a_ptr[i]
        if isinf(v):
            return FiniteStatus.has_infinite
    return FiniteStatus.all_finite


cdef inline FiniteStatus _isfinite_disable_nan(floating* a_ptr,
                                               Py_ssize_t length) noexcept nogil:
    # [FUNCTION DOC]
    # HELPER: scan length values from a_ptr; return has_nan on the first NaN or
    # has_infinite on the first infinity (whichever comes first), else
    # all_finite. No exceptions, nogil.
    cdef Py_ssize_t i
    cdef floating v
    for i in range(length):
        v = a_ptr[i]
        if isnan(v):
            return FiniteStatus.has_nan
        elif isinf(v):
            return FiniteStatus.has_infinite
    return FiniteStatus.all_finite
