# [MODULE DOC]
# Declaration of simultaneous_sort, a nogil introsort that sorts a floating
# values array and permutes a parallel intp indices array identically. Used by
# neighbors / pairwise-distance reduction code to sort distances with their
# sample indices.
from sklearn.utils._typedefs cimport intp_t

from cython cimport floating

cdef void simultaneous_sort(
    floating* values,
    intp_t* indices,
    intp_t n,
    bint use_three_way_partition=*,
) noexcept nogil
# [FUNCTION DOC]
# Sort values[0:n] ascending in place, applying the same permutation to
# indices; use_three_way_partition selects 3-way partitioning (default False).
