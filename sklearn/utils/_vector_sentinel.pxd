# [MODULE DOC]
# Declares vector_to_nd_array, which turns a C++ std::vector (float64, intp,
# int32 or int64 elements) into a 1d NumPy array without copying, by moving the
# vector's buffer into a sentinel Python object that the array keeps as base.
cimport numpy as cnp

from libcpp.vector cimport vector
from sklearn.utils._typedefs cimport intp_t, float64_t, int32_t, int64_t

ctypedef fused vector_typed:
    vector[float64_t]
    vector[intp_t]
    vector[int32_t]
    vector[int64_t]

cdef cnp.ndarray vector_to_nd_array(vector_typed * vect_ptr)
# [FUNCTION DOC]
# Move the contents of *vect_ptr into a new 1d ndarray (zero copy); the vector
# is left empty. See _vector_sentinel.pyx.
