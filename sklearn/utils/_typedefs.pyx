# [MODULE DOC]
# Implementation module for _typedefs; the types are declared in _typedefs.pxd.
# It only contains a testing helper verifying that each C alias maps to the
# expected NumPy dtype.
# _typedefs is a declaration only module
#
# The functions implemented here are for testing purpose only.


import numpy as np


ctypedef fused testing_type_t:
    float32_t
    float64_t
    int8_t
    int32_t
    int64_t
    intp_t
    uint8_t
    uint32_t
    uint64_t


def testing_make_array_from_typed_val(testing_type_t val):
    # [FUNCTION DOC]
    # HELPER: test-only function.
    # Behavior: view the address of the typed scalar val as a length-1 typed array
    # and return it as a NumPy array, so tests can check the resulting dtype.
    # Inputs: val, a scalar of one of the fused testing types (dispatch by caller
    # or by the Python value). Outputs: 1-element ndarray.
    # Exceptions: TypeError/OverflowError on conversion.
    # KNOWN ISSUE: casting a pointer to a typed memoryview does not copy, so the
    # returned array appears to reference the function's local variable val after
    # the function returns (dangling memory); tests only check the dtype, but the
    # value is not reliable.
    cdef testing_type_t[:] val_view = <testing_type_t[:1]>&val
    return np.asarray(val_view)
