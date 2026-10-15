# [MODULE DOC]
# Declaration file for the compiled tree module (_tree.pyx). It exposes to other
# Cython modules (e.g. ensemble gradient boosting, _export helpers via Python) the
# Node and ParentInfo C structs, the Tree extension type (array-of-structs storage
# of a fitted binary tree plus prediction/inspection methods), the TreeBuilder base
# type (growing strategy driving a Splitter) and the cost-complexity pruning entry
# point _build_pruned_tree. Implementations live in _tree.pyx.
# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

# See _tree.pyx for details.

import numpy as np
cimport numpy as cnp

from sklearn.utils._typedefs cimport (
    float32_t, float64_t, intp_t, int32_t, uint8_t, uint32_t
)
from sklearn.tree._splitter cimport Splitter, SplitRecord


cdef struct Node:
    # [CLASS DOC]
    # Plain C record for one tree node, stored contiguously in Tree.nodes. Internal
    # nodes hold child ids, split feature and threshold; leaves have both children
    # equal to TREE_LEAF (-1) and feature/threshold equal to TREE_UNDEFINED (-2).
    # missing_go_to_left tells prediction which child receives samples whose split
    # feature is NaN. The field layout is mirrored by NODE_DTYPE for numpy views and
    # pickling, so changing it breaks pickle compatibility.
    # Base storage structure for the nodes in a Tree object

    intp_t left_child                    # id of the left child of the node
    intp_t right_child                   # id of the right child of the node
    intp_t feature                       # Feature used for splitting the node
    float64_t threshold                  # Threshold value at the node
    float64_t impurity                   # Impurity of the node (i.e., the value of the criterion)
    intp_t n_node_samples                # Number of samples at the node
    float64_t weighted_n_node_samples    # Weighted number of samples at the node
    uint8_t missing_go_to_left     # Whether features have missing values


cdef struct ParentInfo:
    # [CLASS DOC]
    # Plain C record describing the node about to be split, handed by the builder
    # to Splitter.node_split. impurity is the node's own impurity, lower_bound and
    # upper_bound are the admissible value range under monotonic constraints, and
    # n_constant_features is the count of features known constant (updated in place
    # by the splitter and reused for children).
    # Structure to store information about the parent of a node
    # This is passed to the splitter, to provide information about the previous split

    float64_t lower_bound           # the lower bound of the parent's impurity
    float64_t upper_bound           # the upper bound of the parent's impurity
    float64_t impurity              # the impurity of the parent
    intp_t n_constant_features      # the number of constant features found in parent

cdef class Tree:
    # [CLASS DOC]
    # Declaration of the Tree extension type: a growable array of Node structs plus a
    # parallel float64 value buffer of shape (capacity, n_outputs, max_n_classes).
    # See _tree.pyx for full method documentation.
    # The Tree object is a binary tree structure constructed by the
    # TreeBuilder. The tree structure is used for predictions and
    # feature importances.

    # Input/Output layout
    cdef public intp_t n_features        # Number of features in X
    cdef intp_t* n_classes               # Number of classes in y[:, k]
    cdef public intp_t n_outputs         # Number of outputs in y
    cdef public intp_t max_n_classes     # max(n_classes)

    # Inner structures: values are stored separately from node structure,
    # since size is determined at runtime.
    cdef public intp_t max_depth         # Max depth of the tree
    cdef public intp_t node_count        # Counter for node IDs
    cdef public intp_t capacity          # Capacity of tree, in terms of nodes
    cdef Node* nodes                     # Array of nodes
    cdef float64_t* value                # (capacity, n_outputs, max_n_classes) array of values
    cdef intp_t value_stride             # = n_outputs * max_n_classes

    # Methods
    cdef intp_t _add_node(self, intp_t parent, bint is_left, bint is_leaf,
                          intp_t feature, float64_t threshold, float64_t impurity,
                          intp_t n_node_samples,
                          float64_t weighted_n_node_samples,
                          uint8_t missing_go_to_left) except -1 nogil
    # [FUNCTION DOC]
    # Append a node and link it to its parent; returns the new node id, or INTPTR_MAX
    # when growing storage failed. Callable without the GIL. See _tree.pyx.
    cdef int _resize(self, intp_t capacity) except -1 nogil
    # [FUNCTION DOC]
    # Resize node/value buffers to capacity, raising MemoryError (re-acquiring the
    # GIL) on failure. Returns 0 on success, -1 with an exception set otherwise.
    cdef int _resize_c(self, intp_t capacity=*) except -1 nogil
    # [FUNCTION DOC]
    # Low-level resize; the default capacity (INTPTR_MAX) means "double the current
    # capacity, or 3 if empty". Returns 0 on success. See _tree.pyx.

    cdef cnp.ndarray _get_value_ndarray(self)
    # [FUNCTION DOC]
    # Return a numpy view (node_count, n_outputs, max_n_classes) over the value buffer.
    cdef cnp.ndarray _get_node_ndarray(self)
    # [FUNCTION DOC]
    # Return a numpy structured-array view (dtype NODE_DTYPE) over the node buffer.

    cpdef cnp.ndarray predict(self, object X)
    # [FUNCTION DOC]
    # Return the stored value of the leaf reached by each row of X.

    cpdef cnp.ndarray apply(self, object X)
    # [FUNCTION DOC]
    # Return the leaf id reached by each row of X (dense or CSR float32 input).
    cdef cnp.ndarray _apply_dense(self, object X)
    # [FUNCTION DOC]
    # Leaf lookup for a dense float32 numpy array.
    cdef cnp.ndarray _apply_sparse_csr(self, object X)
    # [FUNCTION DOC]
    # Leaf lookup for a CSR float32 sparse matrix.

    cpdef object decision_path(self, object X)
    # [FUNCTION DOC]
    # Return a CSR indicator matrix (n_samples, node_count) of nodes visited per row.
    cdef object _decision_path_dense(self, object X)
    # [FUNCTION DOC]
    # Decision path for a dense float32 numpy array.
    cdef object _decision_path_sparse_csr(self, object X)
    # [FUNCTION DOC]
    # Decision path for a CSR float32 sparse matrix.

    cpdef compute_node_depths(self)
    # [FUNCTION DOC]
    # Return the depth of each node (root has depth 1) as an int64 array.
    cpdef compute_feature_importances(self, normalize=*)
    # [FUNCTION DOC]
    # Return impurity-decrease-based feature importances, optionally normalized to sum to 1.


# =============================================================================
# Tree builder
# =============================================================================

cdef class TreeBuilder:
    # [CLASS DOC]
    # Declaration of the abstract tree-growing strategy. Holds the Splitter and the
    # stopping parameters shared by DepthFirstTreeBuilder and BestFirstTreeBuilder.
    # The TreeBuilder recursively builds a Tree object from training samples,
    # using a Splitter object for splitting internal nodes and assigning
    # values to leaves.
    #
    # This class controls the various stopping criteria and the node splitting
    # evaluation order, e.g. depth-first or best-first.

    cdef Splitter splitter              # Splitting algorithm

    cdef intp_t min_samples_split       # Minimum number of samples in an internal node
    cdef intp_t min_samples_leaf        # Minimum number of samples in a leaf
    cdef float64_t min_weight_leaf         # Minimum weight in a leaf
    cdef intp_t max_depth               # Maximal tree depth
    cdef float64_t min_impurity_decrease   # Impurity threshold for early stopping

    cpdef build(
        self,
        Tree tree,
        object X,
        const float64_t[:, ::1] y,
        const float64_t[:] sample_weight=*,
        const uint8_t[::1] missing_values_in_feature_mask=*,
    )
    # [FUNCTION DOC]
    # Grow the given Tree in place from training data X, y (and optional sample
    # weights / missing-value mask). Implemented by subclasses.

    cdef _check_input(
        self,
        object X,
        const float64_t[:, ::1] y,
        const float64_t[:] sample_weight,
    )
    # [FUNCTION DOC]
    # Coerce X to float32 (CSC sorted with int32 indices if sparse, Fortran order if
    # dense) and sample_weight to contiguous; returns the (X, y, sample_weight) tuple.


# =============================================================================
# Tree pruning
# =============================================================================

# The private function allows any external caller to prune the tree and return
# a new tree with the pruned nodes. The pruned tree is a new tree object.
#
# .. warning:: this function is not backwards compatible and may change without
#              notice.
cdef void _build_pruned_tree(
    Tree tree,  # OUT
    Tree orig_tree,
    const uint8_t[:] leaves_in_subtree,
    intp_t capacity
)
# [FUNCTION DOC]
# Copy orig_tree into the empty output tree, turning every node flagged in
# leaves_in_subtree into a leaf and dropping its descendants; capacity is the
# initial node allocation. Raises MemoryError or ValueError. See _tree.pyx.
