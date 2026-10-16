# [MODULE DOC]
# Declarations for _hierarchical_fast.pyx: exposes the UnionFind extension type
# so other Cython modules can cimport it and call its C-level methods.
from sklearn.utils._typedefs cimport intp_t

    # [CLASS DOC]
    # Union-find (disjoint set) structure over 2*n-1 node ids used while building a
    # single-linkage dendrogram; new merged clusters receive consecutive labels
    # starting at n. Attributes: next_label (next id to assign), parent (parent link
    # per node, -1 for roots), size (number of leaves under each node).
cdef class UnionFind:
    cdef intp_t next_label
    cdef intp_t[:] parent
    cdef intp_t[:] size

    # [FUNCTION DOC]
    # Declaration: merges the clusters rooted at m and n into a new node next_label.
    # See the implementation in _hierarchical_fast.pyx.
    cdef void union(self, intp_t m, intp_t n) noexcept
    # [FUNCTION DOC]
    # Declaration: returns the root of n with path compression.
    # See the implementation in _hierarchical_fast.pyx.
    cdef intp_t fast_find(self, intp_t n) noexcept
