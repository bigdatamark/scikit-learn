# [MODULE DOC]
# Declaration of partition_node_indices, the partial-sort routine used by
# the binary trees (KDTree/BallTree) to split node index ranges at build time.
# Implemented in _partition_nodes.pyx on top of C++ std::nth_element.
from cython cimport floating
from sklearn.utils._typedefs cimport float64_t, intp_t

cdef int partition_node_indices(
        const floating *data,
        intp_t *node_indices,
        intp_t split_dim,
        intp_t split_index,
        intp_t n_features,
        intp_t n_points) except -1
# [FUNCTION DOC]
# Rearranges node_indices in place so the element at split_index is the one a
# stable sort on column split_dim would put there, with smaller-or-equal
# values before and larger-or-equal after. Returns 0; -1 on error.
