# [MODULE DOC]
# Declaration of heap_push, a nogil routine for maintaining fixed-size
# max-heaps stored as parallel value/index arrays (used e.g. by neighbors
# search and pairwise-distance reductions to keep the k smallest values).
# Heap routines, used in various Cython implementations.

from cython cimport floating

from sklearn.utils._typedefs cimport intp_t


cdef int heap_push(
    floating* values,
    intp_t* indices,
    intp_t size,
    floating val,
    intp_t val_idx,
) noexcept nogil
# [FUNCTION DOC]
# Push (val, val_idx) into a fixed-size max-heap if val is smaller than the
# current maximum; returns 0. See _heap.pyx.
