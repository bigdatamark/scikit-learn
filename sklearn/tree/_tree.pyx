# [MODULE DOC]
# Core compiled implementation of decision trees.
# Contents:
# - Tree: array-based binary tree (Node structs in one malloc'd buffer plus a
#   parallel float64 value buffer of shape (capacity, n_outputs, max_n_classes)).
#   Provides prediction (apply, predict, decision_path), feature importances,
#   node depths, partial dependence, and pickling with cross-bitness/endianness
#   checks (module-level _check_* helpers).
# - TreeBuilder with two strategies: DepthFirstTreeBuilder (explicit stack,
#   grows until stopping criteria) and BestFirstTreeBuilder (max-heap frontier
#   ordered by impurity improvement, bounded by max_leaf_nodes). Both drive a
#   Splitter (sklearn.tree._splitter) which in turn uses a Criterion
#   (sklearn.tree._criterion) and partitions sample indices in place. Both
#   propagate monotonic-constraint bounds to children via Criterion.middle_value.
# - Minimal cost-complexity pruning: _cost_complexity_prune computes the weakest
#   link sequence, driven by a controller (_AlphaPruner to stop at ccp_alpha,
#   _PathFinder to record the full path), and _build_pruned_tree copies the
#   retained part of a tree into a new Tree.
# Conventions: TREE_LEAF (-1) marks absent children, TREE_UNDEFINED (-2) marks
# unused feature/threshold and the root's parent. Input X must be float32.
# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

from cpython cimport Py_INCREF, PyObject, PyTypeObject

from libc.math cimport INFINITY, isnan
from libc.stdlib cimport free
from libc.string cimport memcpy
from libc.string cimport memset
from libc.stdint cimport INTPTR_MAX
from libcpp.vector cimport vector
from libcpp.algorithm cimport pop_heap
from libcpp.algorithm cimport push_heap
from libcpp.stack cimport stack
from libcpp cimport bool

import struct

import numpy as np
cimport numpy as cnp
cnp.import_array()

from scipy.sparse import issparse
from scipy.sparse import csr_array

from sklearn.utils import _align_api_if_sparse

from sklearn.tree._utils cimport safe_realloc
from sklearn.tree._utils cimport sizet_ptr_to_ndarray

cdef extern from "numpy/arrayobject.h":
    object PyArray_NewFromDescr(PyTypeObject* subtype, cnp.dtype descr,
                                int nd, cnp.npy_intp* dims,
                                cnp.npy_intp* strides,
                                void* data, int flags, object obj)
    # [FUNCTION DOC]
    # External numpy C-API: create an ndarray of a given dtype descriptor over an
    # existing data pointer (no copy). Steals a reference to descr. Used to wrap the
    # Node buffer as a structured array.
    int PyArray_SetBaseObject(cnp.ndarray arr, PyObject* obj)
    # [FUNCTION DOC]
    # External numpy C-API: set arr's base object (steals a reference to obj) so the
    # owner stays alive while the view exists. Returns negative on failure.

# =============================================================================
# Types and constants
# =============================================================================

cdef float64_t EPSILON = np.finfo('double').eps

# Some handy constants (BestFirstTreeBuilder)
cdef bint IS_FIRST = 1
cdef bint IS_NOT_FIRST = 0
cdef bint IS_LEFT = 1
cdef bint IS_NOT_LEFT = 0

TREE_LEAF = -1
TREE_UNDEFINED = -2
cdef intp_t _TREE_LEAF = TREE_LEAF
cdef intp_t _TREE_UNDEFINED = TREE_UNDEFINED

# Build the corresponding numpy dtype for Node.
# This works by casting `dummy` to an array of Node of length 1, which numpy
# can construct a `dtype`-object for. See https://stackoverflow.com/q/62448946
# for a more detailed explanation.
cdef Node dummy
NODE_DTYPE = np.asarray(<Node[:1]>(&dummy)).dtype

cdef inline void _init_parent_record(ParentInfo* record) noexcept nogil:
    # [FUNCTION DOC]
    # HELPER: initializes the ParentInfo used for the root before tree building.
    # Sets n_constant_features to 0, impurity to +inf, lower_bound to -inf and
    # upper_bound to +inf (i.e. no monotonic bounds). Mutates record in place;
    # no GIL needed; cannot raise.
    record.n_constant_features = 0
    record.impurity = INFINITY
    record.lower_bound = -INFINITY
    record.upper_bound = INFINITY

# =============================================================================
# TreeBuilder
# =============================================================================

cdef class TreeBuilder:
    # [CLASS DOC]
    # Abstract base for tree-growing strategies. Subclasses set the splitter and
    # stopping parameters in __cinit__ and implement build. Provides the shared
    # input coercion helper _check_input.
    """Interface for different tree building strategies."""

    cpdef build(
        self,
        Tree tree,
        object X,
        const float64_t[:, ::1] y,
        const float64_t[:] sample_weight=None,
        const uint8_t[::1] missing_values_in_feature_mask=None,
    ):
        # [FUNCTION DOC]
        # Abstract build entry point; does nothing in the base class and returns None.
        # Subclasses grow tree in place from X, y.
        """Build a decision tree from the training set (X, y)."""
        pass

    cdef inline _check_input(
        self,
        object X,
        const float64_t[:, ::1] y,
        const float64_t[:] sample_weight,
    ):
        # [FUNCTION DOC]
        # HELPER: input coercion used at the start of every build.
        # Sparse X: converted to CSC with sorted indices, data cast to contiguous
        # float32; raises ValueError if indices or indptr are not int32. Dense X: if not
        # float32, copied to a Fortran-ordered float32 array (column access is what the
        # splitter needs). sample_weight: if given and its base buffer is not contiguous,
        # copied to a C-contiguous float64 array. y is returned unchanged.
        # Outputs: tuple (X, y, sample_weight). Side effects: X.sort_indices may sort the
        # caller's matrix in place only if tocsc returned the same object (already CSC).
        """Check input dtype, layout and format"""
        if issparse(X):
            X = X.tocsc()
            X.sort_indices()

            if X.data.dtype != np.float32:
                X.data = np.ascontiguousarray(X.data, dtype=np.float32)

            if X.indices.dtype != np.int32 or X.indptr.dtype != np.int32:
                raise ValueError("No support for np.int64 index based "
                                 "sparse matrices")

        elif X.dtype != np.float32:
            # since we have to copy we will make it fortran for efficiency
            X = np.asfortranarray(X, dtype=np.float32)

        if sample_weight is not None and not sample_weight.base.flags.contiguous:
            sample_weight = np.asarray(sample_weight, dtype=np.float64, order="C")

        return X, y, sample_weight

# Depth first builder ---------------------------------------------------------
# A record on the stack for depth-first tree growing
cdef struct StackRecord:
    # [CLASS DOC]
    # Stack entry for depth-first growth: the sample range [start, end) of a node to
    # create, its depth, parent node id and side, its precomputed impurity, the
    # inherited constant-feature count and its monotonic value bounds.
    intp_t start
    intp_t end
    intp_t depth
    intp_t parent
    bint is_left
    float64_t impurity
    intp_t n_constant_features
    float64_t lower_bound
    float64_t upper_bound

cdef class DepthFirstTreeBuilder(TreeBuilder):
    # [CLASS DOC]
    # Grows a tree depth-first using an explicit stack (left child processed before
    # right). Each popped node becomes a leaf or is split; there is no limit on the
    # number of leaves other than the stopping criteria.
    """Build a decision tree in depth-first fashion."""

    def __cinit__(self, Splitter splitter, intp_t min_samples_split,
                  intp_t min_samples_leaf, float64_t min_weight_leaf,
                  intp_t max_depth, float64_t min_impurity_decrease):
        # [FUNCTION DOC]
        # Store the splitter and stopping parameters: min_samples_split,
        # min_samples_leaf, min_weight_leaf, max_depth, min_impurity_decrease.
        # No validation is done here; cannot raise beyond argument conversion.
        self.splitter = splitter
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_leaf = min_weight_leaf
        self.max_depth = max_depth
        self.min_impurity_decrease = min_impurity_decrease

    cpdef build(
        self,
        Tree tree,
        object X,
        const float64_t[:, ::1] y,
        const float64_t[:] sample_weight=None,
        const uint8_t[::1] missing_values_in_feature_mask=None,
    ):
        # [FUNCTION DOC]
        # Algorithm:
        # 1. Coerce inputs with _check_input. Pre-size the tree to a full binary tree of
        #    depth tree.max_depth if that is at most 10 (2^(d+1)-1 nodes), else 2047.
        # 2. splitter.init with X, y, weights and missing mask.
        # 3. Push the root record (all samples, depth 0, no parent, infinite bounds).
        #    Pop records until the stack is empty. For each: splitter.node_reset to the
        #    range (gives weighted count); mark leaf if depth >= max_depth, n samples <
        #    min_samples_split, < 2*min_samples_leaf, or weight < 2*min_weight_leaf. For
        #    the root only, compute impurity via splitter.node_impurity; otherwise use
        #    the impurity stored by the parent. Impurity <= EPSILON also makes a leaf.
        #    If still splittable, call splitter.node_split; becomes a leaf if no valid
        #    split was found (pos >= end) or improvement + EPSILON < min_impurity_decrease.
        # 4. tree._add_node, then store node value via splitter.node_value, clipped to
        #    the record's bounds when monotonic constraints are active.
        # 5. For split nodes compute children bounds: unconstrained feature copies the
        #    parent bounds; +1 constraint gives left upper and right lower bound =
        #    criterion.middle_value; -1 the mirror. Push right then left child records
        #    (left popped first) with impurities from the split record.
        # 6. Finally shrink tree capacity to node_count and set tree.max_depth.
        # Inputs: tree (empty Tree, its max_depth used only for initial sizing), X, y
        # (n_samples, n_outputs) float64, optional sample_weight and per-feature uint8
        # missing-value mask. Output: tree mutated in place; returns None.
        # Exceptions: MemoryError if node allocation fails; errors from the splitter.
        # Side effects: main loop runs without the GIL; splitter state and its sample
        # index permutation are mutated.
        """Build a decision tree from the training set (X, y)."""

        # check input
        X, y, sample_weight = self._check_input(X, y, sample_weight)

        # Initial capacity
        cdef intp_t init_capacity

        if tree.max_depth <= 10:
            init_capacity = <intp_t> (2 ** (tree.max_depth + 1)) - 1
        else:
            init_capacity = 2047

        tree._resize(init_capacity)

        # Parameters
        cdef Splitter splitter = self.splitter
        cdef intp_t max_depth = self.max_depth
        cdef intp_t min_samples_leaf = self.min_samples_leaf
        cdef float64_t min_weight_leaf = self.min_weight_leaf
        cdef intp_t min_samples_split = self.min_samples_split
        cdef float64_t min_impurity_decrease = self.min_impurity_decrease

        # Recursive partition (without actual recursion)
        splitter.init(X, y, sample_weight, missing_values_in_feature_mask)

        cdef intp_t start
        cdef intp_t end
        cdef intp_t depth
        cdef intp_t parent
        cdef bint is_left
        cdef intp_t n_node_samples = splitter.n_samples
        cdef float64_t weighted_n_node_samples
        cdef SplitRecord split
        cdef intp_t node_id

        cdef float64_t middle_value
        cdef float64_t left_child_min
        cdef float64_t left_child_max
        cdef float64_t right_child_min
        cdef float64_t right_child_max
        cdef bint is_leaf
        cdef bint first = 1
        cdef intp_t max_depth_seen = -1
        cdef int rc = 0

        cdef stack[StackRecord] builder_stack
        cdef StackRecord stack_record

        cdef ParentInfo parent_record
        _init_parent_record(&parent_record)

        with nogil:
            # push root node onto stack
            builder_stack.push({
                "start": 0,
                "end": n_node_samples,
                "depth": 0,
                "parent": _TREE_UNDEFINED,
                "is_left": 0,
                "impurity": INFINITY,
                "n_constant_features": 0,
                "lower_bound": -INFINITY,
                "upper_bound": INFINITY,
            })

            while not builder_stack.empty():
                stack_record = builder_stack.top()
                builder_stack.pop()

                start = stack_record.start
                end = stack_record.end
                depth = stack_record.depth
                parent = stack_record.parent
                is_left = stack_record.is_left
                parent_record.impurity = stack_record.impurity
                parent_record.n_constant_features = stack_record.n_constant_features
                parent_record.lower_bound = stack_record.lower_bound
                parent_record.upper_bound = stack_record.upper_bound

                n_node_samples = end - start
                splitter.node_reset(start, end, &weighted_n_node_samples)

                is_leaf = (depth >= max_depth or
                           n_node_samples < min_samples_split or
                           n_node_samples < 2 * min_samples_leaf or
                           weighted_n_node_samples < 2 * min_weight_leaf)

                if first:
                    parent_record.impurity = splitter.node_impurity()
                    first = 0

                # impurity == 0 with tolerance due to rounding errors
                is_leaf = is_leaf or parent_record.impurity <= EPSILON

                if not is_leaf:
                    splitter.node_split(
                        &parent_record,
                        &split,
                    )
                    # If EPSILON=0 in the below comparison, float precision
                    # issues stop splitting, producing trees that are
                    # dissimilar to v0.18
                    is_leaf = (is_leaf or split.pos >= end or
                               (split.improvement + EPSILON <
                                min_impurity_decrease))

                node_id = tree._add_node(parent, is_left, is_leaf, split.feature,
                                         split.threshold, parent_record.impurity,
                                         n_node_samples, weighted_n_node_samples,
                                         split.missing_go_to_left)

                if node_id == INTPTR_MAX:
                    rc = -1
                    break

                # Store value for all nodes, to facilitate tree/model
                # inspection and interpretation
                splitter.node_value(tree.value + node_id * tree.value_stride)
                if splitter.with_monotonic_cst:
                    splitter.clip_node_value(tree.value + node_id * tree.value_stride, parent_record.lower_bound, parent_record.upper_bound)

                if not is_leaf:
                    if (
                        not splitter.with_monotonic_cst or
                        splitter.monotonic_cst[split.feature] == 0
                    ):
                        # Split on a feature with no monotonicity constraint

                        # Current bounds must always be propagated to both children.
                        # If a monotonic constraint is active, bounds are used in
                        # node value clipping.
                        left_child_min = right_child_min = parent_record.lower_bound
                        left_child_max = right_child_max = parent_record.upper_bound
                    elif splitter.monotonic_cst[split.feature] == 1:
                        # Split on a feature with monotonic increase constraint
                        left_child_min = parent_record.lower_bound
                        right_child_max = parent_record.upper_bound

                        # Lower bound for right child and upper bound for left child
                        # are set to the same value.
                        middle_value = splitter.criterion.middle_value()
                        right_child_min = middle_value
                        left_child_max = middle_value
                    else:  # i.e. splitter.monotonic_cst[split.feature] == -1
                        # Split on a feature with monotonic decrease constraint
                        right_child_min = parent_record.lower_bound
                        left_child_max = parent_record.upper_bound

                        # Lower bound for left child and upper bound for right child
                        # are set to the same value.
                        middle_value = splitter.criterion.middle_value()
                        left_child_min = middle_value
                        right_child_max = middle_value

                    # Push right child on stack
                    builder_stack.push({
                        "start": split.pos,
                        "end": end,
                        "depth": depth + 1,
                        "parent": node_id,
                        "is_left": 0,
                        "impurity": split.impurity_right,
                        "n_constant_features": parent_record.n_constant_features,
                        "lower_bound": right_child_min,
                        "upper_bound": right_child_max,
                    })

                    # Push left child on stack
                    builder_stack.push({
                        "start": start,
                        "end": split.pos,
                        "depth": depth + 1,
                        "parent": node_id,
                        "is_left": 1,
                        "impurity": split.impurity_left,
                        "n_constant_features": parent_record.n_constant_features,
                        "lower_bound": left_child_min,
                        "upper_bound": left_child_max,
                    })

                if depth > max_depth_seen:
                    max_depth_seen = depth

            if rc >= 0:
                rc = tree._resize_c(tree.node_count)

            if rc >= 0:
                tree.max_depth = max_depth_seen
        if rc == -1:
            raise MemoryError()


# Best first builder ----------------------------------------------------------
cdef struct FrontierRecord:
    # [CLASS DOC]
    # Frontier entry for best-first growth: an already-created node (node_id) with
    # its sample range, best split position, depth, leaf flag, impurities of node
    # and prospective children, improvement (heap key), monotonic bounds and the
    # middle value used to derive children bounds.
    # Record of information of a Node, the frontier for a split. Those records are
    # maintained in a heap to access the Node with the best improvement in impurity,
    # allowing growing trees greedily on this improvement.
    intp_t node_id
    intp_t start
    intp_t end
    intp_t pos
    intp_t depth
    bint is_leaf
    float64_t impurity
    float64_t impurity_left
    float64_t impurity_right
    float64_t improvement
    float64_t lower_bound
    float64_t upper_bound
    float64_t middle_value

cdef inline bool _compare_records(
    const FrontierRecord& left,
    const FrontierRecord& right,
):
    # [FUNCTION DOC]
    # HELPER: heap ordering for the best-first frontier.
    # Returns true when left has strictly smaller improvement than right, which makes
    # the standard heap functions produce a max-heap on improvement. Pure, no GIL use.
    return left.improvement < right.improvement

cdef inline void _add_to_frontier(
    FrontierRecord rec,
    vector[FrontierRecord]& frontier,
) noexcept nogil:
    # [FUNCTION DOC]
    # HELPER: pushes rec onto the frontier max-heap (vector push then heap sift-up
    # by improvement). Mutates frontier; may allocate (C++ vector growth). Declared
    # noexcept, so an allocation failure in the vector would not be reported.
    """Adds record `rec` to the priority queue `frontier`."""
    frontier.push_back(rec)
    push_heap(frontier.begin(), frontier.end(), &_compare_records)


cdef class BestFirstTreeBuilder(TreeBuilder):
    # [CLASS DOC]
    # Grows a tree best-first: nodes are evaluated (split searched) when created and
    # put in a max-heap on impurity improvement; the best is expanded next until the
    # heap is empty or max_leaf_nodes - 1 splits have been made.
    """Build a decision tree in best-first fashion.

    The best node to expand is given by the node at the frontier that has the
    highest impurity improvement.
    """
    cdef intp_t max_leaf_nodes

    def __cinit__(self, Splitter splitter, intp_t min_samples_split,
                  intp_t min_samples_leaf,  min_weight_leaf,
                  intp_t max_depth, intp_t max_leaf_nodes,
                  float64_t min_impurity_decrease):
        # [FUNCTION DOC]
        # Store splitter, stopping parameters and max_leaf_nodes. min_weight_leaf is
        # untyped in the signature and converted to float64 on assignment.
        self.splitter = splitter
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_leaf = min_weight_leaf
        self.max_depth = max_depth
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease

    cpdef build(
        self,
        Tree tree,
        object X,
        const float64_t[:, ::1] y,
        const float64_t[:] sample_weight=None,
        const uint8_t[::1] missing_values_in_feature_mask=None,
    ):
        # [FUNCTION DOC]
        # Algorithm:
        # 1. Coerce inputs, splitter.init, allow max_leaf_nodes - 1 split nodes and
        #    pre-size the tree to 2*max_leaf_nodes - 1 nodes.
        # 2. Create and evaluate the root via _add_split_node and push it on the heap.
        # 3. Pop the highest-improvement record. If it was flagged leaf or the split
        #    budget is exhausted, convert the node into a leaf (children TREE_LEAF,
        #    feature/threshold TREE_UNDEFINED). Otherwise derive children bounds as in
        #    depth-first growth (using the record's stored middle_value), decrement the
        #    budget and create/evaluate left [start, pos) and right [pos, end) children
        #    via _add_split_node, pushing both on the heap. The parent Node pointer is
        #    re-fetched after the left child because the node buffer may be reallocated.
        # 4. Shrink capacity to node_count and set tree.max_depth (max depth popped).
        # Inputs/Outputs: as DepthFirstTreeBuilder.build; tree mutated in place.
        # Exceptions: MemoryError on allocation failure.
        # Side effects: main loop runs without the GIL; splitter state mutated.
        """Build a decision tree from the training set (X, y)."""

        # check input
        X, y, sample_weight = self._check_input(X, y, sample_weight)

        # Parameters
        cdef Splitter splitter = self.splitter
        cdef intp_t max_leaf_nodes = self.max_leaf_nodes

        # Recursive partition (without actual recursion)
        splitter.init(X, y, sample_weight, missing_values_in_feature_mask)

        cdef vector[FrontierRecord] frontier
        cdef FrontierRecord record
        cdef FrontierRecord split_node_left
        cdef FrontierRecord split_node_right
        cdef float64_t left_child_min
        cdef float64_t left_child_max
        cdef float64_t right_child_min
        cdef float64_t right_child_max

        cdef intp_t n_node_samples = splitter.n_samples
        cdef intp_t max_split_nodes = max_leaf_nodes - 1
        cdef bint is_leaf
        cdef intp_t max_depth_seen = -1
        cdef int rc = 0
        cdef Node* node

        cdef ParentInfo parent_record
        _init_parent_record(&parent_record)

        # Initial capacity
        cdef intp_t init_capacity = max_split_nodes + max_leaf_nodes
        tree._resize(init_capacity)

        with nogil:
            # add root to frontier
            rc = self._add_split_node(
                splitter=splitter,
                tree=tree,
                start=0,
                end=n_node_samples,
                is_first=IS_FIRST,
                is_left=IS_LEFT,
                parent=NULL,
                depth=0,
                parent_record=&parent_record,
                res=&split_node_left,
            )
            if rc >= 0:
                _add_to_frontier(split_node_left, frontier)

            while not frontier.empty():
                pop_heap(frontier.begin(), frontier.end(), &_compare_records)
                record = frontier.back()
                frontier.pop_back()

                node = &tree.nodes[record.node_id]
                is_leaf = (record.is_leaf or max_split_nodes <= 0)

                if is_leaf:
                    # Node is not expandable; set node as leaf
                    node.left_child = _TREE_LEAF
                    node.right_child = _TREE_LEAF
                    node.feature = _TREE_UNDEFINED
                    node.threshold = _TREE_UNDEFINED

                else:
                    # Node is expandable

                    if (
                        not splitter.with_monotonic_cst or
                        splitter.monotonic_cst[node.feature] == 0
                    ):
                        # Split on a feature with no monotonicity constraint

                        # Current bounds must always be propagated to both children.
                        # If a monotonic constraint is active, bounds are used in
                        # node value clipping.
                        left_child_min = right_child_min = record.lower_bound
                        left_child_max = right_child_max = record.upper_bound
                    elif splitter.monotonic_cst[node.feature] == 1:
                        # Split on a feature with monotonic increase constraint
                        left_child_min = record.lower_bound
                        right_child_max = record.upper_bound

                        # Lower bound for right child and upper bound for left child
                        # are set to the same value.
                        right_child_min = record.middle_value
                        left_child_max = record.middle_value
                    else:  # i.e. splitter.monotonic_cst[split.feature] == -1
                        # Split on a feature with monotonic decrease constraint
                        right_child_min = record.lower_bound
                        left_child_max = record.upper_bound

                        # Lower bound for left child and upper bound for right child
                        # are set to the same value.
                        left_child_min = record.middle_value
                        right_child_max = record.middle_value

                    # Decrement number of split nodes available
                    max_split_nodes -= 1

                    # Compute left split node
                    parent_record.lower_bound = left_child_min
                    parent_record.upper_bound = left_child_max
                    parent_record.impurity = record.impurity_left
                    rc = self._add_split_node(
                        splitter=splitter,
                        tree=tree,
                        start=record.start,
                        end=record.pos,
                        is_first=IS_NOT_FIRST,
                        is_left=IS_LEFT,
                        parent=node,
                        depth=record.depth + 1,
                        parent_record=&parent_record,
                        res=&split_node_left,
                    )
                    if rc == -1:
                        break

                    # tree.nodes may have changed
                    node = &tree.nodes[record.node_id]

                    # Compute right split node
                    parent_record.lower_bound = right_child_min
                    parent_record.upper_bound = right_child_max
                    parent_record.impurity = record.impurity_right
                    rc = self._add_split_node(
                        splitter=splitter,
                        tree=tree,
                        start=record.pos,
                        end=record.end,
                        is_first=IS_NOT_FIRST,
                        is_left=IS_NOT_LEFT,
                        parent=node,
                        depth=record.depth + 1,
                        parent_record=&parent_record,
                        res=&split_node_right,
                    )
                    if rc == -1:
                        break

                    # Add nodes to queue
                    _add_to_frontier(split_node_left, frontier)
                    _add_to_frontier(split_node_right, frontier)

                if record.depth > max_depth_seen:
                    max_depth_seen = record.depth

            if rc >= 0:
                rc = tree._resize_c(tree.node_count)

            if rc >= 0:
                tree.max_depth = max_depth_seen

        if rc == -1:
            raise MemoryError()

    cdef inline int _add_split_node(
        self,
        Splitter splitter,
        Tree tree,
        intp_t start,
        intp_t end,
        bint is_first,
        bint is_left,
        Node* parent,
        intp_t depth,
        ParentInfo* parent_record,
        FrontierRecord* res
    ) except -1 nogil:
        # [FUNCTION DOC]
        # HELPER: creates one node for best-first growth and evaluates its best split.
        # Behavior: node_reset the splitter on [start, end); reset
        # parent_record.n_constant_features to 0; if is_first compute node impurity from
        # the splitter, otherwise use parent_record.impurity. Apply the same leaf rules
        # as depth-first (depth, sample counts, weight, impurity <= EPSILON); if not a
        # leaf call node_split and turn into a leaf when no valid split or improvement
        # is below min_impurity_decrease. Add the node (parent id derived from the parent
        # pointer, or TREE_UNDEFINED), store (and optionally clip) its value, and fill
        # res with node id, range, depth, impurity, bounds, middle_value and either the
        # split info or leaf defaults (pos=end, improvement 0, child impurities = node).
        # Inputs: splitter, tree, range, flags, parent Node pointer (may be NULL), depth,
        # parent_record (impurity and bounds for this node; mutated), res (output).
        # Returns 0 on success, -1 if the tree could not grow (no exception set here).
        # Runs without the GIL.
        """Adds node w/ partition ``[start, end)`` to the frontier. """
        cdef SplitRecord split
        cdef intp_t node_id
        cdef intp_t n_node_samples
        cdef float64_t min_impurity_decrease = self.min_impurity_decrease
        cdef float64_t weighted_n_node_samples
        cdef bint is_leaf

        splitter.node_reset(start, end, &weighted_n_node_samples)

        # reset n_constant_features for this specific split before beginning split search
        parent_record.n_constant_features = 0

        if is_first:
            parent_record.impurity = splitter.node_impurity()

        n_node_samples = end - start
        is_leaf = (depth >= self.max_depth or
                   n_node_samples < self.min_samples_split or
                   n_node_samples < 2 * self.min_samples_leaf or
                   weighted_n_node_samples < 2 * self.min_weight_leaf or
                   parent_record.impurity <= EPSILON  # impurity == 0 with tolerance
                   )

        if not is_leaf:
            splitter.node_split(
                parent_record,
                &split,
            )
            # If EPSILON=0 in the below comparison, float precision issues stop
            # splitting early, producing trees that are dissimilar to v0.18
            is_leaf = (is_leaf or split.pos >= end or
                       split.improvement + EPSILON < min_impurity_decrease)

        node_id = tree._add_node(parent - tree.nodes
                                 if parent != NULL
                                 else _TREE_UNDEFINED,
                                 is_left, is_leaf,
                                 split.feature, split.threshold, parent_record.impurity,
                                 n_node_samples, weighted_n_node_samples,
                                 split.missing_go_to_left)
        if node_id == INTPTR_MAX:
            return -1

        # compute values also for split nodes (might become leaves later).
        splitter.node_value(tree.value + node_id * tree.value_stride)
        if splitter.with_monotonic_cst:
            splitter.clip_node_value(tree.value + node_id * tree.value_stride, parent_record.lower_bound, parent_record.upper_bound)

        res.node_id = node_id
        res.start = start
        res.end = end
        res.depth = depth
        res.impurity = parent_record.impurity
        res.lower_bound = parent_record.lower_bound
        res.upper_bound = parent_record.upper_bound
        res.middle_value = splitter.criterion.middle_value()

        if not is_leaf:
            # is split node
            res.pos = split.pos
            res.is_leaf = 0
            res.improvement = split.improvement
            res.impurity_left = split.impurity_left
            res.impurity_right = split.impurity_right

        else:
            # is leaf => 0 improvement
            res.pos = end
            res.is_leaf = 1
            res.improvement = 0.0
            res.impurity_left = parent_record.impurity
            res.impurity_right = parent_record.impurity

        return 0


# =============================================================================
# Tree
# =============================================================================

cdef class Tree:
    # [CLASS DOC]
    # Array-based binary decision tree. Owns three malloc'd buffers: n_classes
    # (n_outputs entries), nodes (capacity Node structs) and value (capacity *
    # value_stride float64, value_stride = n_outputs * max_n_classes). Node 0 is the
    # root; children always have larger ids than their parent. Python-visible
    # properties return numpy views over these buffers (kept alive by referencing
    # the Tree), so they become dangling if the buffers are reallocated later.
    """Array-based representation of a binary decision tree.

    The binary tree is represented as a number of parallel arrays. The i-th
    element of each array holds information about the node `i`. Node 0 is the
    tree's root. You can find a detailed description of all arrays in
    `_tree.pxd`. NOTE: Some of the arrays only apply to either leaves or split
    nodes, resp. In this case the values of nodes of the other type are
    arbitrary!

    Attributes
    ----------
    node_count : intp_t
        The number of nodes (internal nodes + leaves) in the tree.

    capacity : intp_t
        The current capacity (i.e., size) of the arrays, which is at least as
        great as `node_count`.

    max_depth : intp_t
        The depth of the tree, i.e. the maximum depth of its leaves.

    children_left : array of intp_t, shape [node_count]
        children_left[i] holds the node id of the left child of node i.
        For leaves, children_left[i] == TREE_LEAF. Otherwise,
        children_left[i] > i. This child handles the case where
        X[:, feature[i]] <= threshold[i].

    children_right : array of intp_t, shape [node_count]
        children_right[i] holds the node id of the right child of node i.
        For leaves, children_right[i] == TREE_LEAF. Otherwise,
        children_right[i] > i. This child handles the case where
        X[:, feature[i]] > threshold[i].

    n_leaves : intp_t
        Number of leaves in the tree.

    feature : array of intp_t, shape [node_count]
        feature[i] holds the feature to split on, for the internal node i.

    threshold : array of float64_t, shape [node_count]
        threshold[i] holds the threshold for the internal node i.

    value : array of float64_t, shape [node_count, n_outputs, max_n_classes]
        Contains the constant prediction value of each node.

    impurity : array of float64_t, shape [node_count]
        impurity[i] holds the impurity (i.e., the value of the splitting
        criterion) at node i.

    n_node_samples : array of intp_t, shape [node_count]
        n_node_samples[i] holds the number of training samples reaching node i.

    weighted_n_node_samples : array of float64_t, shape [node_count]
        weighted_n_node_samples[i] holds the weighted number of training samples
        reaching node i.

    missing_go_to_left : array of bool, shape [node_count]
        missing_go_to_left[i] holds a bool indicating whether or not there were
        missing values at node i.
    """
    # Wrap for outside world.
    # WARNING: these reference the current `nodes` and `value` buffers, which
    # must not be freed by a subsequent memory allocation.
    # (i.e. through `_resize` or `__setstate__`)
    @property
    def n_classes(self):
        # [FUNCTION DOC]
        # Property: copy of the per-output number of classes as an intp numpy array of
        # length n_outputs. Allocates a new array; cannot fail except on memory.
        return sizet_ptr_to_ndarray(self.n_classes, self.n_outputs)

    @property
    def children_left(self):
        # [FUNCTION DOC]
        # Property: view of left child ids for the first node_count nodes (TREE_LEAF for leaves).
        return self._get_node_ndarray()['left_child'][:self.node_count]

    @property
    def children_right(self):
        # [FUNCTION DOC]
        # Property: view of right child ids for the first node_count nodes (TREE_LEAF for leaves).
        return self._get_node_ndarray()['right_child'][:self.node_count]

    @property
    def n_leaves(self):
        # [FUNCTION DOC]
        # Property: number of leaves, counted as nodes whose left and right child are both -1.
        return np.sum(np.logical_and(
            self.children_left == -1,
            self.children_right == -1))

    @property
    def feature(self):
        # [FUNCTION DOC]
        # Property: view of split feature indices (TREE_UNDEFINED for leaves).
        return self._get_node_ndarray()['feature'][:self.node_count]

    @property
    def threshold(self):
        # [FUNCTION DOC]
        # Property: view of split thresholds (TREE_UNDEFINED for leaves).
        return self._get_node_ndarray()['threshold'][:self.node_count]

    @property
    def impurity(self):
        # [FUNCTION DOC]
        # Property: view of node impurities as computed by the criterion during fit.
        return self._get_node_ndarray()['impurity'][:self.node_count]

    @property
    def n_node_samples(self):
        # [FUNCTION DOC]
        # Property: view of the unweighted training sample counts per node.
        return self._get_node_ndarray()['n_node_samples'][:self.node_count]

    @property
    def weighted_n_node_samples(self):
        # [FUNCTION DOC]
        # Property: view of weighted training sample counts per node.
        return self._get_node_ndarray()['weighted_n_node_samples'][:self.node_count]

    @property
    def missing_go_to_left(self):
        # [FUNCTION DOC]
        # Property: view of per-node flags telling whether NaN goes to the left child
        # (meaningful only for split nodes; zero for leaves).
        return self._get_node_ndarray()['missing_go_to_left'][:self.node_count]

    @property
    def value(self):
        # [FUNCTION DOC]
        # Property: view of node values, shape (node_count, n_outputs, max_n_classes).
        # For classifiers these are weighted class fractions; for regressors the mean
        # (or median for MAE) per output.
        return self._get_value_ndarray()[:self.node_count]

    # TODO: Convert n_classes to cython.integral memory view once
    #  https://github.com/cython/cython/issues/5243 is fixed
    def __cinit__(self, intp_t n_features, cnp.ndarray n_classes, intp_t n_outputs):
        # [FUNCTION DOC]
        # Constructor. Validates n_classes (1-D integer array; cast to the platform intp
        # dtype if it is a 4 or 8 byte int, else ValueError via _check_n_classes), copies
        # it into a malloc'd buffer, computes max_n_classes and value_stride, and starts
        # with an empty tree (no node or value storage, capacity 0, max_depth 0).
        # Inputs: n_features, n_classes array of length n_outputs (use 1 per output for
        # regression), n_outputs. Raises ValueError or MemoryError.
        """Constructor."""
        cdef intp_t dummy = 0
        size_t_dtype = np.array(dummy).dtype

        n_classes = _check_n_classes(n_classes, size_t_dtype)

        # Input/Output layout
        self.n_features = n_features
        self.n_outputs = n_outputs
        self.n_classes = NULL
        safe_realloc(&self.n_classes, n_outputs)

        self.max_n_classes = np.max(n_classes)
        self.value_stride = n_outputs * self.max_n_classes

        cdef intp_t k
        for k in range(n_outputs):
            self.n_classes[k] = n_classes[k]

        # Inner structures
        self.max_depth = 0
        self.node_count = 0
        self.capacity = 0
        self.value = NULL
        self.nodes = NULL

    def __dealloc__(self):
        # [FUNCTION DOC]
        # Destructor: frees the n_classes, value and nodes buffers. Any numpy views
        # still alive keep a reference to the Tree, so this only runs once they are gone.
        """Destructor."""
        # Free all inner structures
        free(self.n_classes)
        free(self.value)
        free(self.nodes)

    def __reduce__(self):
        # [FUNCTION DOC]
        # Pickle support: returns (Tree, constructor args (n_features, n_classes copy,
        # n_outputs), state from __getstate__).
        """Reduce re-implementation, for pickling."""
        return (Tree, (self.n_features,
                       sizet_ptr_to_ndarray(self.n_classes, self.n_outputs),
                       self.n_outputs), self.__getstate__())

    def __getstate__(self):
        # [FUNCTION DOC]
        # Pickle support: returns a dict with max_depth, node_count, and numpy views of
        # the node array ("nodes", NODE_DTYPE) and value array ("values"). Only the
        # first node_count entries are included; capacity is rebuilt on load.
        """Getstate re-implementation, for pickling."""
        d = {}
        # capacity is inferred during the __setstate__ using nodes
        d["max_depth"] = self.max_depth
        d["node_count"] = self.node_count
        d["nodes"] = self._get_node_ndarray()
        d["values"] = self._get_value_ndarray()
        return d

    def __setstate__(self, d):
        # [FUNCTION DOC]
        # Unpickle: restores max_depth and node_count, then validates the node array
        # (1-D, C-contiguous, dtype equal to or bitness/endianness-compatible with
        # NODE_DTYPE, cast if needed) and value array (shape (n_nodes, n_outputs,
        # max_n_classes), C-contiguous, float64 possibly byte-swapped). Resizes buffers
        # to exactly n_nodes and memcpys both arrays in.
        # Raises ValueError for old-format state (no "nodes") or incompatible arrays,
        # MemoryError if resizing fails. Frees/reallocates the internal buffers.
        """Setstate re-implementation, for unpickling."""
        self.max_depth = d["max_depth"]
        self.node_count = d["node_count"]

        if 'nodes' not in d:
            raise ValueError('You have loaded Tree version which '
                             'cannot be imported')

        node_ndarray = d['nodes']
        value_ndarray = d['values']

        value_shape = (node_ndarray.shape[0], self.n_outputs,
                       self.max_n_classes)

        node_ndarray = _check_node_ndarray(node_ndarray, expected_dtype=NODE_DTYPE)
        value_ndarray = _check_value_ndarray(
            value_ndarray,
            expected_dtype=np.dtype(np.float64),
            expected_shape=value_shape
        )

        self.capacity = node_ndarray.shape[0]
        if self._resize_c(self.capacity) != 0:
            raise MemoryError("resizing tree to %d" % self.capacity)

        memcpy(self.nodes, cnp.PyArray_DATA(node_ndarray),
               self.capacity * sizeof(Node))
        memcpy(self.value, cnp.PyArray_DATA(value_ndarray),
               self.capacity * self.value_stride * sizeof(float64_t))

    cdef int _resize(self, intp_t capacity) except -1 nogil:
        # [FUNCTION DOC]
        # Resize node and value buffers via _resize_c; if that reports failure, acquire
        # the GIL and raise MemoryError. Callable without the GIL.
        # Returns 0 on success (implicitly), -1 with exception on failure.
        # KNOWN ISSUE: the docstring says capacity == -1 doubles the size; doubling is
        # actually triggered by INTPTR_MAX (the _resize_c default), and -1 would be
        # passed through as a negative capacity.
        """Resize all inner arrays to `capacity`, if `capacity` == -1, then
           double the size of the inner arrays.

        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.
        """
        if self._resize_c(capacity) != 0:
            # Acquire gil only if we need to raise
            with gil:
                raise MemoryError()

    cdef int _resize_c(self, intp_t capacity=INTPTR_MAX) except -1 nogil:
        # [FUNCTION DOC]
        # Reallocate nodes to capacity Node structs and value to capacity*value_stride
        # doubles. No-op if capacity equals the current one and buffers exist. The
        # sentinel INTPTR_MAX (default) means grow: 3 if empty, else double. Newly added
        # regions of both buffers are zero-filled (zeros make classifier argmax and
        # pickles deterministic). If capacity < node_count, node_count is truncated.
        # Returns 0. Allocation goes through safe_realloc, which raises MemoryError
        # (acquiring the GIL) rather than returning -1, so callers checking for non-zero
        # mainly see the exception path. Mutates buffers, capacity, node_count.
        """Guts of _resize

        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.
        """
        if capacity == self.capacity and self.nodes != NULL:
            return 0

        if capacity == INTPTR_MAX:
            if self.capacity == 0:
                capacity = 3  # default initial value
            else:
                capacity = 2 * self.capacity

        safe_realloc(&self.nodes, capacity)
        safe_realloc(&self.value, capacity * self.value_stride)

        if capacity > self.capacity:
            # value memory is initialised to 0 to enable classifier argmax
            memset(<void*>(self.value + self.capacity * self.value_stride), 0,
                   (capacity - self.capacity) * self.value_stride *
                   sizeof(float64_t))
            # node memory is initialised to 0 to ensure deterministic pickle (padding in Node struct)
            memset(<void*>(self.nodes + self.capacity), 0, (capacity - self.capacity) * sizeof(Node))

        # if capacity smaller than node_count, adjust the counter
        if capacity < self.node_count:
            self.node_count = capacity

        self.capacity = capacity
        return 0

    cdef intp_t _add_node(self, intp_t parent, bint is_left, bint is_leaf,
                          intp_t feature, float64_t threshold, float64_t impurity,
                          intp_t n_node_samples,
                          float64_t weighted_n_node_samples,
                          uint8_t missing_go_to_left) except -1 nogil:
        # [FUNCTION DOC]
        # Append a node at id node_count, growing storage (doubling) if full. Sets
        # impurity and sample counts; if parent is not TREE_UNDEFINED, writes the new id
        # into the parent's left or right child slot per is_left. Leaves get children
        # TREE_LEAF and feature/threshold TREE_UNDEFINED; split nodes get feature,
        # threshold and missing_go_to_left (children filled in later by their own
        # _add_node calls). Increments node_count.
        # Returns the new node id, or INTPTR_MAX if growing failed. Runs without the GIL.
        # KNOWN ISSUE: the docstring says it returns (size_t)(-1) on error; the code
        # returns INTPTR_MAX, which is what callers test for.
        """Add a node to the tree.

        The new node registers itself as the child of its parent.

        Returns (size_t)(-1) on error.
        """
        cdef intp_t node_id = self.node_count

        if node_id >= self.capacity:
            if self._resize_c() != 0:
                return INTPTR_MAX

        cdef Node* node = &self.nodes[node_id]
        node.impurity = impurity
        node.n_node_samples = n_node_samples
        node.weighted_n_node_samples = weighted_n_node_samples

        if parent != _TREE_UNDEFINED:
            if is_left:
                self.nodes[parent].left_child = node_id
            else:
                self.nodes[parent].right_child = node_id

        if is_leaf:
            node.left_child = _TREE_LEAF
            node.right_child = _TREE_LEAF
            node.feature = _TREE_UNDEFINED
            node.threshold = _TREE_UNDEFINED

        else:
            # left_child and right_child will be set later
            node.feature = feature
            node.threshold = threshold
            node.missing_go_to_left = missing_go_to_left

        self.node_count += 1

        return node_id

    cpdef cnp.ndarray predict(self, object X):
        # [FUNCTION DOC]
        # Return value[leaf of each sample] via apply then take along axis 0. Output
        # shape (n_samples, n_outputs, max_n_classes), reshaped to (n_samples,
        # max_n_classes) when n_outputs == 1. Raises what apply raises.
        """Predict target for X."""
        out = self._get_value_ndarray().take(self.apply(X), axis=0,
                                             mode='clip')
        if self.n_outputs == 1:
            out = out.reshape(X.shape[0], self.max_n_classes)
        return out

    cpdef cnp.ndarray apply(self, object X):
        # [FUNCTION DOC]
        # Dispatch to _apply_sparse_csr for scipy sparse input, else _apply_dense.
        # Returns an intp array of leaf ids, one per row.
        """Finds the terminal region (=leaf node) for each sample in X."""
        if issparse(X):
            return self._apply_sparse_csr(X)
        else:
            return self._apply_dense(X)

    cdef inline cnp.ndarray _apply_dense(self, object X):
        # [FUNCTION DOC]
        # For each row, start at the root and descend until a leaf: NaN feature value
        # goes to the child given by missing_go_to_left, otherwise value <= threshold
        # goes left, else right. Returns intp array of leaf ids (length n_samples).
        # Inputs: X must be a float32 numpy ndarray of shape (n_samples, >= n_features).
        # Raises ValueError for wrong type or dtype. Traversal runs without the GIL.
        """Finds the terminal region (=leaf node) for each sample in X."""

        # Check input
        if not isinstance(X, np.ndarray):
            raise ValueError("X should be in np.ndarray format, got %s"
                             % type(X))

        if X.dtype != np.float32:
            raise ValueError("X.dtype should be np.float32, got %s" % X.dtype)

        # Extract input
        cdef const float32_t[:, :] X_ndarray = X
        cdef intp_t n_samples = X.shape[0]
        cdef float32_t X_i_node_feature

        # Initialize output
        cdef intp_t[:] out = np.zeros(n_samples, dtype=np.intp)

        # Initialize auxiliary data-structure
        cdef Node* node = NULL
        cdef intp_t i = 0

        with nogil:
            for i in range(n_samples):
                node = self.nodes
                # While node not a leaf
                while node.left_child != _TREE_LEAF:
                    X_i_node_feature = X_ndarray[i, node.feature]
                    # ... and node.right_child != _TREE_LEAF:
                    if isnan(X_i_node_feature):
                        if node.missing_go_to_left:
                            node = &self.nodes[node.left_child]
                        else:
                            node = &self.nodes[node.right_child]
                    elif X_i_node_feature <= node.threshold:
                        node = &self.nodes[node.left_child]
                    else:
                        node = &self.nodes[node.right_child]

                out[i] = <intp_t>(node - self.nodes)  # node offset

        return np.asarray(out)

    cdef inline cnp.ndarray _apply_sparse_csr(self, object X):
        # [FUNCTION DOC]
        # Leaf lookup for CSR float32 input. Uses two scratch arrays of length
        # n_features: X_sample (dense values of the current row) and feature_to_sample
        # (the last row index that set each feature, initialised to -1), so a feature is
        # non-zero for row i iff feature_to_sample[f] == i; absent features read as 0.
        # Descends with value <= threshold going left. Returns intp array of leaf ids.
        # Raises ValueError for non-CSR or non-float32 input; MemoryError on allocation.
        # Side effects: allocates and frees scratch buffers; traversal without the GIL.
        # KNOWN ISSUE: NaN handling differs from the dense path: missing_go_to_left is
        # ignored and an explicitly stored NaN always goes right (NaN <= t is false).
        """Finds the terminal region (=leaf node) for each sample in sparse X.
        """
        # Check input
        if not (issparse(X) and X.format == 'csr'):
            raise ValueError("X should be in CSR sparse format, got %s"
                             % type(X))

        if X.dtype != np.float32:
            raise ValueError("X.dtype should be np.float32, got %s" % X.dtype)

        # Extract input
        cdef const float32_t[:] X_data = X.data
        cdef const int32_t[:] X_indices = X.indices
        cdef const int32_t[:] X_indptr = X.indptr

        cdef intp_t n_samples = X.shape[0]
        cdef intp_t n_features = X.shape[1]

        # Initialize output
        cdef intp_t[:] out = np.zeros(n_samples, dtype=np.intp)

        # Initialize auxiliary data-structure
        cdef float32_t feature_value = 0.
        cdef Node* node = NULL
        cdef float32_t* X_sample = NULL
        cdef intp_t i = 0
        cdef int32_t k = 0

        # feature_to_sample as a data structure records the last seen sample
        # for each feature; functionally, it is an efficient way to identify
        # which features are nonzero in the present sample.
        cdef intp_t* feature_to_sample = NULL

        safe_realloc(&X_sample, n_features)
        safe_realloc(&feature_to_sample, n_features)

        with nogil:
            memset(feature_to_sample, -1, n_features * sizeof(intp_t))

            for i in range(n_samples):
                node = self.nodes

                for k in range(X_indptr[i], X_indptr[i + 1]):
                    feature_to_sample[X_indices[k]] = i
                    X_sample[X_indices[k]] = X_data[k]

                # While node not a leaf
                while node.left_child != _TREE_LEAF:
                    # ... and node.right_child != _TREE_LEAF:
                    if feature_to_sample[node.feature] == i:
                        feature_value = X_sample[node.feature]

                    else:
                        feature_value = 0.

                    if feature_value <= node.threshold:
                        node = &self.nodes[node.left_child]
                    else:
                        node = &self.nodes[node.right_child]

                out[i] = <intp_t>(node - self.nodes)  # node offset

            # Free auxiliary arrays
            free(X_sample)
            free(feature_to_sample)

        return np.asarray(out)

    cpdef object decision_path(self, object X):
        # [FUNCTION DOC]
        # Dispatch to the sparse or dense decision-path implementation depending on
        # whether X is a scipy sparse matrix.
        """Finds the decision path (=node) for each sample in X."""
        if issparse(X):
            return self._decision_path_sparse_csr(X)
        else:
            return self._decision_path_dense(X)

    cdef inline object _decision_path_dense(self, object X):
        # [FUNCTION DOC]
        # Build the node-indicator matrix for dense float32 X: traverse each row like
        # _apply_dense, appending every visited node id (internal and final leaf) to
        # indices, with indptr delimiting rows. indices is preallocated to
        # n_samples*(1+max_depth) and truncated. Returns a CSR array of ones, shape
        # (n_samples, node_count), passed through _align_api_if_sparse.
        # Raises ValueError on wrong type/dtype. Traversal runs without the GIL.
        """Finds the decision path (=node) for each sample in X."""

        # Check input
        if not isinstance(X, np.ndarray):
            raise ValueError("X should be in np.ndarray format, got %s"
                             % type(X))

        if X.dtype != np.float32:
            raise ValueError("X.dtype should be np.float32, got %s" % X.dtype)

        # Extract input
        cdef const float32_t[:, :] X_ndarray = X
        cdef intp_t n_samples = X.shape[0]
        cdef float32_t X_i_node_feature

        # Initialize output
        cdef intp_t[:] indptr = np.zeros(n_samples + 1, dtype=np.intp)
        cdef intp_t[:] indices = np.zeros(
            n_samples * (1 + self.max_depth), dtype=np.intp
        )

        # Initialize auxiliary data-structure
        cdef Node* node = NULL
        cdef intp_t i = 0

        with nogil:
            for i in range(n_samples):
                node = self.nodes
                indptr[i + 1] = indptr[i]

                # Add all external nodes
                while node.left_child != _TREE_LEAF:
                    # ... and node.right_child != _TREE_LEAF:
                    indices[indptr[i + 1]] = <intp_t>(node - self.nodes)
                    indptr[i + 1] += 1

                    X_i_node_feature = X_ndarray[i, node.feature]
                    if isnan(X_i_node_feature):
                        if node.missing_go_to_left:
                            node = &self.nodes[node.left_child]
                        else:
                            node = &self.nodes[node.right_child]
                    elif X_i_node_feature <= node.threshold:
                        node = &self.nodes[node.left_child]
                    else:
                        node = &self.nodes[node.right_child]

                # Add the leave node
                indices[indptr[i + 1]] = <intp_t>(node - self.nodes)
                indptr[i + 1] += 1

        indices = indices[:indptr[n_samples]]
        cdef intp_t[:] data = np.ones(shape=len(indices), dtype=np.intp)
        out = csr_array((data, indices, indptr),
                        shape=(n_samples, self.node_count))

        return _align_api_if_sparse(out)

    cdef inline object _decision_path_sparse_csr(self, object X):
        # [FUNCTION DOC]
        # Sparse CSR counterpart of _decision_path_dense, using the same per-row
        # densification trick as _apply_sparse_csr. Same output format.
        # Raises ValueError on wrong format/dtype, MemoryError on allocation.
        # KNOWN ISSUE: like _apply_sparse_csr, ignores missing_go_to_left for NaN values.
        """Finds the decision path (=node) for each sample in X."""

        # Check input
        if not (issparse(X) and X.format == "csr"):
            raise ValueError("X should be in CSR sparse format, got %s"
                             % type(X))

        if X.dtype != np.float32:
            raise ValueError("X.dtype should be np.float32, got %s" % X.dtype)

        # Extract input
        cdef const float32_t[:] X_data = X.data
        cdef const int32_t[:] X_indices = X.indices
        cdef const int32_t[:] X_indptr = X.indptr

        cdef intp_t n_samples = X.shape[0]
        cdef intp_t n_features = X.shape[1]

        # Initialize output
        cdef intp_t[:] indptr = np.zeros(n_samples + 1, dtype=np.intp)
        cdef intp_t[:] indices = np.zeros(
            n_samples * (1 + self.max_depth), dtype=np.intp
        )

        # Initialize auxiliary data-structure
        cdef float32_t feature_value = 0.
        cdef Node* node = NULL
        cdef float32_t* X_sample = NULL
        cdef intp_t i = 0
        cdef int32_t k = 0

        # feature_to_sample as a data structure records the last seen sample
        # for each feature; functionally, it is an efficient way to identify
        # which features are nonzero in the present sample.
        cdef intp_t* feature_to_sample = NULL

        safe_realloc(&X_sample, n_features)
        safe_realloc(&feature_to_sample, n_features)

        with nogil:
            memset(feature_to_sample, -1, n_features * sizeof(intp_t))

            for i in range(n_samples):
                node = self.nodes
                indptr[i + 1] = indptr[i]

                for k in range(X_indptr[i], X_indptr[i + 1]):
                    feature_to_sample[X_indices[k]] = i
                    X_sample[X_indices[k]] = X_data[k]

                # While node not a leaf
                while node.left_child != _TREE_LEAF:
                    # ... and node.right_child != _TREE_LEAF:

                    indices[indptr[i + 1]] = <intp_t>(node - self.nodes)
                    indptr[i + 1] += 1

                    if feature_to_sample[node.feature] == i:
                        feature_value = X_sample[node.feature]

                    else:
                        feature_value = 0.

                    if feature_value <= node.threshold:
                        node = &self.nodes[node.left_child]
                    else:
                        node = &self.nodes[node.right_child]

                # Add the leave node
                indices[indptr[i + 1]] = <intp_t>(node - self.nodes)
                indptr[i + 1] += 1

            # Free auxiliary arrays
            free(X_sample)
            free(feature_to_sample)

        indices = indices[:indptr[n_samples]]
        cdef intp_t[:] data = np.ones(shape=len(indices), dtype=np.intp)
        out = csr_array((data, indices, indptr),
                        shape=(n_samples, self.node_count))

        return _align_api_if_sparse(out)

    cpdef compute_node_depths(self):
        # [FUNCTION DOC]
        # Single pass over nodes in id order (parents precede children): root depth is
        # 1 and each child's depth is its parent's plus one. Returns an int64 array of
        # length node_count. Note depths are 1-based, unlike Tree.max_depth (0-based).
        # Assumes node_count >= 1.
        """Compute the depth of each node in a tree.

        .. versionadded:: 1.3

        Returns
        -------
        depths : ndarray of shape (self.node_count,), dtype=np.int64
            The depth of each node in the tree.
        """
        cdef:
            cnp.int64_t[::1] depths = np.empty(self.node_count, dtype=np.int64)
            cnp.npy_intp[:] children_left = self.children_left
            cnp.npy_intp[:] children_right = self.children_right
            cnp.npy_intp node_id
            cnp.npy_intp node_count = self.node_count
            cnp.int64_t depth

        depths[0] = 1  # init root node
        for node_id in range(node_count):
            if children_left[node_id] != _TREE_LEAF:
                depth = depths[node_id] + 1
                depths[children_left[node_id]] = depth
                depths[children_right[node_id]] = depth

        return depths.base

    cpdef compute_feature_importances(self, normalize=True):
        # [FUNCTION DOC]
        # For each split node add weighted_n_node_samples*impurity of the node minus the
        # same quantity for both children to importances[feature]; then divide all by the
        # root's weighted sample count. If normalize and the total is positive, scale to
        # sum to 1. Returns a float64 array of length n_features.
        # Side effects: accumulation loop runs without the GIL.
        """Computes the importance of each feature (aka variable)."""
        cdef Node* left
        cdef Node* right
        cdef Node* nodes = self.nodes
        cdef Node* node = nodes
        cdef Node* end_node = node + self.node_count

        cdef float64_t normalizer = 0.

        cdef cnp.float64_t[:] importances = np.zeros(self.n_features)

        with nogil:
            while node != end_node:
                if node.left_child != _TREE_LEAF:
                    # ... and node.right_child != _TREE_LEAF:
                    left = &nodes[node.left_child]
                    right = &nodes[node.right_child]

                    importances[node.feature] += (
                        node.weighted_n_node_samples * node.impurity -
                        left.weighted_n_node_samples * left.impurity -
                        right.weighted_n_node_samples * right.impurity)
                node += 1

        for i in range(self.n_features):
            importances[i] /= nodes[0].weighted_n_node_samples

        if normalize:
            normalizer = np.sum(importances)

            if normalizer > 0.0:
                # Avoid dividing by zero (e.g., when root is pure)
                for i in range(self.n_features):
                    importances[i] /= normalizer

        return np.asarray(importances)

    cdef cnp.ndarray _get_value_ndarray(self):
        # [FUNCTION DOC]
        # Wrap the value buffer as a float64 ndarray (node_count, n_outputs,
        # max_n_classes) without copying; the array holds a reference to self as its
        # base. Raises ValueError if setting the base fails. Increments self refcount.
        """Wraps value as a 3-d NumPy array.

        The array keeps a reference to this Tree, which manages the underlying
        memory.
        """
        cdef cnp.npy_intp shape[3]
        shape[0] = <cnp.npy_intp> self.node_count
        shape[1] = <cnp.npy_intp> self.n_outputs
        shape[2] = <cnp.npy_intp> self.max_n_classes
        cdef cnp.ndarray arr
        arr = cnp.PyArray_SimpleNewFromData(3, shape, cnp.NPY_DOUBLE, self.value)
        Py_INCREF(self)
        if PyArray_SetBaseObject(arr, <PyObject*> self) < 0:
            raise ValueError("Can't initialize array.")
        return arr

    cdef cnp.ndarray _get_node_ndarray(self):
        # [FUNCTION DOC]
        # Wrap the node buffer as a 1-D structured ndarray of length node_count with
        # dtype NODE_DTYPE and stride sizeof(Node), without copying; base set to self.
        # Raises ValueError if setting the base fails. Increments refcounts of
        # NODE_DTYPE (stolen by the constructor) and self (stolen by set-base).
        """Wraps nodes as a NumPy struct array.

        The array keeps a reference to this Tree, which manages the underlying
        memory. Individual fields are publicly accessible as properties of the
        Tree.
        """
        cdef cnp.npy_intp shape[1]
        shape[0] = <cnp.npy_intp> self.node_count
        cdef cnp.npy_intp strides[1]
        strides[0] = sizeof(Node)
        cdef cnp.ndarray arr
        Py_INCREF(NODE_DTYPE)
        arr = PyArray_NewFromDescr(<PyTypeObject *> cnp.ndarray,
                                   <cnp.dtype> NODE_DTYPE, 1, shape,
                                   strides, <void*> self.nodes,
                                   cnp.NPY_ARRAY_DEFAULT, None)
        Py_INCREF(self)
        if PyArray_SetBaseObject(arr, <PyObject*> self) < 0:
            raise ValueError("Can't initialize array.")
        return arr

    def compute_partial_dependence(self, float32_t[:, ::1] X,
                                   const intp_t[::1] target_features,
                                   float64_t[::1] out):
        # [FUNCTION DOC]
        # Weighted tree traversal for partial dependence (the "recursion" method).
        # For each grid row of X: depth-first walk with an explicit stack of node ids and
        # weights starting at the root with weight 1. At a split on one of
        # target_features, follow only the child selected by the row's value for that
        # feature (X columns are aligned with target_features). At other splits push
        # both children with weight multiplied by each child's fraction of the parent's
        # weighted training samples. At a leaf add weight * leaf value to out[row].
        # Inputs: X float32 (n_points, n_target_features), target_features intp
        # (n_target_features), out float64 (n_points), accumulated into (should start at 0).
        # Raises ValueError if the visited leaf weights do not sum to about 1.
        # Runs with the GIL (plain Cython loop).
        # KNOWN ISSUE: the leaf value is read as value[node_idx], i.e. assumes
        # value_stride == 1 (single-output regression trees); for other trees it reads
        # wrong entries. Also NaN grid values are not routed via missing_go_to_left.
        """Partial dependence of the response on the ``target_feature`` set.

        For each sample in ``X`` a tree traversal is performed.
        Each traversal starts from the root with weight 1.0.

        At each non-leaf node that splits on a target feature, either
        the left child or the right child is visited based on the feature
        value of the current sample, and the weight is not modified.
        At each non-leaf node that splits on a complementary feature,
        both children are visited and the weight is multiplied by the fraction
        of training samples which went to each child.

        At each leaf, the value of the node is multiplied by the current
        weight (weights sum to 1 for all visited terminal nodes).

        Parameters
        ----------
        X : view on 2d ndarray, shape (n_samples, n_target_features)
            The grid points on which the partial dependence should be
            evaluated.
        target_features : view on 1d ndarray, shape (n_target_features)
            The set of target features for which the partial dependence
            should be evaluated.
        out : view on 1d ndarray, shape (n_samples)
            The value of the partial dependence function on each grid
            point.
        """
        cdef:
            float64_t[::1] weight_stack = np.zeros(self.node_count,
                                                   dtype=np.float64)
            intp_t[::1] node_idx_stack = np.zeros(self.node_count,
                                                  dtype=np.intp)
            intp_t sample_idx
            intp_t feature_idx
            intp_t stack_size
            float64_t left_sample_frac
            float64_t current_weight
            float64_t total_weight  # used for sanity check only
            Node *current_node  # use a pointer to avoid copying attributes
            intp_t current_node_idx
            bint is_target_feature
            intp_t _TREE_LEAF = TREE_LEAF  # to avoid python interactions

        for sample_idx in range(X.shape[0]):
            # init stacks for current sample
            stack_size = 1
            node_idx_stack[0] = 0  # root node
            weight_stack[0] = 1  # all the samples are in the root node
            total_weight = 0

            while stack_size > 0:
                # pop the stack
                stack_size -= 1
                current_node_idx = node_idx_stack[stack_size]
                current_node = &self.nodes[current_node_idx]

                if current_node.left_child == _TREE_LEAF:
                    # leaf node
                    out[sample_idx] += (weight_stack[stack_size] *
                                        self.value[current_node_idx])
                    total_weight += weight_stack[stack_size]
                else:
                    # non-leaf node

                    # determine if the split feature is a target feature
                    is_target_feature = False
                    for feature_idx in range(target_features.shape[0]):
                        if target_features[feature_idx] == current_node.feature:
                            is_target_feature = True
                            break

                    if is_target_feature:
                        # In this case, we push left or right child on stack
                        if X[sample_idx, feature_idx] <= current_node.threshold:
                            node_idx_stack[stack_size] = current_node.left_child
                        else:
                            node_idx_stack[stack_size] = current_node.right_child
                        stack_size += 1
                    else:
                        # In this case, we push both children onto the stack,
                        # and give a weight proportional to the number of
                        # samples going through each branch.

                        # push left child
                        node_idx_stack[stack_size] = current_node.left_child
                        left_sample_frac = (
                            self.nodes[current_node.left_child].weighted_n_node_samples /
                            current_node.weighted_n_node_samples)
                        current_weight = weight_stack[stack_size]
                        weight_stack[stack_size] = current_weight * left_sample_frac
                        stack_size += 1

                        # push right child
                        node_idx_stack[stack_size] = current_node.right_child
                        weight_stack[stack_size] = (
                            current_weight * (1 - left_sample_frac))
                        stack_size += 1

            # Sanity check. Should never happen.
            if not (0.999 < total_weight < 1.001):
                raise ValueError("Total weight should be 1.0 but was %.9f" %
                                 total_weight)


def _check_n_classes(n_classes, expected_dtype):
    # [FUNCTION DOC]
    # HELPER: validates n_classes for Tree construction/unpickling.
    # Requires 1-D (else ValueError). Returns it unchanged if dtype matches
    # expected_dtype; casts integer arrays of 4 or 8 byte itemsize (other bitness or
    # endianness); otherwise raises ValueError.
    if n_classes.ndim != 1:
        raise ValueError(
            f"Wrong dimensions for n_classes from the pickle: "
            f"expected 1, got {n_classes.ndim}"
        )

    if n_classes.dtype == expected_dtype:
        return n_classes

    # Handles both different endianness and different bitness
    if n_classes.dtype.kind == "i" and n_classes.dtype.itemsize in [4, 8]:
        return n_classes.astype(expected_dtype, casting="same_kind")

    raise ValueError(
        "n_classes from the pickle has an incompatible dtype:\n"
        f"- expected: {expected_dtype}\n"
        f"- got:      {n_classes.dtype}"
    )


def _check_value_ndarray(value_ndarray, expected_dtype, expected_shape):
    # [FUNCTION DOC]
    # HELPER: validates the pickled value array in Tree.__setstate__.
    # Requires exact expected_shape and C-contiguity (ValueError otherwise). Returns
    # it if dtype matches; casts other-endian float64 ('f8'); else ValueError.
    if value_ndarray.shape != expected_shape:
        raise ValueError(
            "Wrong shape for value array from the pickle: "
            f"expected {expected_shape}, got {value_ndarray.shape}"
        )

    if not value_ndarray.flags.c_contiguous:
        raise ValueError(
            "value array from the pickle should be a C-contiguous array"
        )

    if value_ndarray.dtype == expected_dtype:
        return value_ndarray

    # Handles different endianness
    if value_ndarray.dtype.str.endswith('f8'):
        return value_ndarray.astype(expected_dtype, casting='equiv')

    raise ValueError(
        "value array from the pickle has an incompatible dtype:\n"
        f"- expected: {expected_dtype}\n"
        f"- got:      {value_ndarray.dtype}"
    )


def _dtype_to_dict(dtype):
    # [FUNCTION DOC]
    # HELPER: maps each field name of a structured dtype to its dtype string
    # (e.g. '<i8'), used to compare node dtypes field by field.
    return {name: dt.str for name, (dt, *rest) in dtype.fields.items()}


def _dtype_dict_with_modified_bitness(dtype_dict):
    # [FUNCTION DOC]
    # HELPER: returns a copy of a node dtype dict where the intp fields
    # (left_child, right_child, feature, n_node_samples) use the other pointer size
    # (4 <-> 8 bytes) relative to the current platform.
    # field names in Node struct with intp_t types (see sklearn/tree/_tree.pxd)
    indexing_field_names = ["left_child", "right_child", "feature", "n_node_samples"]

    expected_dtype_size = str(struct.calcsize("P"))
    allowed_dtype_size = "8" if expected_dtype_size == "4" else "4"

    allowed_dtype_dict = dtype_dict.copy()
    for name in indexing_field_names:
        allowed_dtype_dict[name] = allowed_dtype_dict[name].replace(
            expected_dtype_size, allowed_dtype_size
        )

    return allowed_dtype_dict


def _all_compatible_dtype_dicts(dtype):
    # [FUNCTION DOC]
    # HELPER: list of the four acceptable node dtype dicts for unpickling: native,
    # other bitness, other endianness, and both changed.
    # The Cython code for decision trees uses platform-specific intp_t
    # typed indexing fields that correspond to either i4 or i8 dtypes for
    # the matching fields in the numpy array depending on the bitness of
    # the platform (32 bit or 64 bit respectively).
    #
    # We need to cast the indexing fields of the NODE_DTYPE-dtyped array at
    # pickle load time to enable cross-bitness deployment scenarios. We
    # typically want to make it possible to run the expensive fit method of
    # a tree estimator on a 64 bit server platform, pickle the estimator
    # for deployment and run the predict method of a low power 32 bit edge
    # platform.
    #
    # A similar thing happens for endianness, the machine where the pickle was
    # saved can have a different endianness than the machine where the pickle
    # is loaded

    dtype_dict = _dtype_to_dict(dtype)
    dtype_dict_with_modified_bitness = _dtype_dict_with_modified_bitness(dtype_dict)
    dtype_dict_with_modified_endianness = _dtype_to_dict(dtype.newbyteorder())
    dtype_dict_with_modified_bitness_and_endianness = _dtype_dict_with_modified_bitness(
        dtype_dict_with_modified_endianness
    )

    return [
        dtype_dict,
        dtype_dict_with_modified_bitness,
        dtype_dict_with_modified_endianness,
        dtype_dict_with_modified_bitness_and_endianness,
    ]


def _check_node_ndarray(node_ndarray, expected_dtype):
    # [FUNCTION DOC]
    # HELPER: validates the pickled node array in Tree.__setstate__.
    # Requires 1-D and C-contiguous (ValueError). Returns as-is if dtype equals
    # expected_dtype; if its field dtypes match one of _all_compatible_dtype_dicts,
    # returns a same_kind cast copy; otherwise ValueError.
    if node_ndarray.ndim != 1:
        raise ValueError(
            "Wrong dimensions for node array from the pickle: "
            f"expected 1, got {node_ndarray.ndim}"
        )

    if not node_ndarray.flags.c_contiguous:
        raise ValueError(
            "node array from the pickle should be a C-contiguous array"
        )

    node_ndarray_dtype = node_ndarray.dtype
    if node_ndarray_dtype == expected_dtype:
        return node_ndarray

    node_ndarray_dtype_dict = _dtype_to_dict(node_ndarray_dtype)
    all_compatible_dtype_dicts = _all_compatible_dtype_dicts(expected_dtype)

    if node_ndarray_dtype_dict not in all_compatible_dtype_dicts:
        raise ValueError(
            "node array from the pickle has an incompatible dtype:\n"
            f"- expected: {expected_dtype}\n"
            f"- got     : {node_ndarray_dtype}"
        )

    return node_ndarray.astype(expected_dtype, casting="same_kind")


# =============================================================================
# Build Pruned Tree
# =============================================================================


cdef class _CCPPruneController:
    # [CLASS DOC]
    # Strategy object for _cost_complexity_prune. Base implementation never stops,
    # records nothing and does nothing after pruning; subclasses override hooks.
    """Base class used by build_pruned_tree_ccp and ccp_pruning_path
    to control pruning.
    """
    cdef bint stop_pruning(self, float64_t effective_alpha) noexcept nogil:
        # [FUNCTION DOC]
        # Hook called with the next weakest-link alpha; return true to stop before
        # pruning that link. Base returns 0 (never stop). nogil, cannot raise.
        """Return 1 to stop pruning and 0 to continue pruning"""
        return 0

    cdef void save_metrics(self, float64_t effective_alpha,
                           float64_t subtree_impurities) noexcept nogil:
        # [FUNCTION DOC]
        # Hook called with each alpha and the total leaf impurity of the subtree after
        # pruning at that alpha (first call: alpha 0, unpruned tree). Base: no-op.
        """Save metrics when pruning"""
        pass

    cdef void after_pruning(self, uint8_t[:] in_subtree) noexcept nogil:
        # [FUNCTION DOC]
        # Hook called once at the end with the mask of nodes kept in the subtree. Base: no-op.
        """Called after pruning"""
        pass


cdef class _AlphaPruner(_CCPPruneController):
    # [CLASS DOC]
    # Controller that stops pruning at the first alpha exceeding ccp_alpha and then
    # counts retained nodes (capacity) for allocating the pruned tree.
    """Use alpha to control when to stop pruning."""
    cdef float64_t ccp_alpha
    cdef intp_t capacity

    def __cinit__(self, float64_t ccp_alpha):
        # [FUNCTION DOC]
        # Store ccp_alpha and set capacity to 0.
        self.ccp_alpha = ccp_alpha
        self.capacity = 0

    cdef bint stop_pruning(self, float64_t effective_alpha) noexcept nogil:
        # [FUNCTION DOC]
        # Stop when ccp_alpha < effective_alpha, so all links with alpha <= ccp_alpha are pruned.
        # The subtree on the previous iteration has the greatest ccp_alpha
        # less than or equal to self.ccp_alpha
        return self.ccp_alpha < effective_alpha

    cdef void after_pruning(self, uint8_t[:] in_subtree) noexcept nogil:
        # [FUNCTION DOC]
        # Add the number of nodes flagged in in_subtree to capacity.
        # KNOWN ISSUE: the docstring says it counts leaves; it counts every retained
        # node (which is the correct capacity for the pruned tree).
        """Updates the number of leaves in subtree"""
        for i in range(in_subtree.shape[0]):
            if in_subtree[i]:
                self.capacity += 1


cdef class _PathFinder(_CCPPruneController):
    # [CLASS DOC]
    # Controller that never stops and records every (alpha, subtree impurity) pair,
    # producing the full cost-complexity pruning path.
    """Record metrics used to return the cost complexity path."""
    cdef float64_t[:] ccp_alphas
    cdef float64_t[:] impurities
    cdef uint32_t count

    def __cinit__(self,  intp_t node_count):
        # [FUNCTION DOC]
        # Allocate ccp_alphas and impurities arrays of length node_count (an upper
        # bound on the number of pruning steps + 1) and set count to 0.
        self.ccp_alphas = np.zeros(shape=(node_count), dtype=np.float64)
        self.impurities = np.zeros(shape=(node_count), dtype=np.float64)
        self.count = 0

    cdef void save_metrics(self,
                           float64_t effective_alpha,
                           float64_t subtree_impurities) noexcept nogil:
        # [FUNCTION DOC]
        # Append (effective_alpha, subtree_impurities) at position count and increment
        # it. No bounds check; relies on node_count being large enough. nogil.
        self.ccp_alphas[self.count] = effective_alpha
        self.impurities[self.count] = subtree_impurities
        self.count += 1


cdef struct CostComplexityPruningRecord:
    # [CLASS DOC]
    # Stack entry used to discover parents: a node id and its parent id.
    intp_t node_idx
    intp_t parent

cdef _cost_complexity_prune(uint8_t[:] leaves_in_subtree,  # OUT
                            Tree orig_tree,
                            _CCPPruneController controller):
    # [FUNCTION DOC]
    # HELPER: minimal cost-complexity (weakest link) pruning used by
    # _build_pruned_tree_ccp and ccp_pruning_path.
    # Algorithm: r_node[i] = weighted_n_node_samples[i]*impurity[i]/root weight.
    # Traverse to get parents and mark original leaves in leaves_in_subtree. For each
    # node compute r_branch (sum of r_node over its subtree's leaves) and n_leaves
    # (leaf count, left 0 for leaves). Report (0, r_branch[root]). Then, while the
    # root is still prunable: over all candidate internal nodes, alpha =
    # (r_node - r_branch)/(n_leaves - 1); pick the minimal one (ties: lowest id).
    # Ask controller.stop_pruning; if not stopping, remove all descendants from the
    # subtree and candidates, make that node a leaf, and propagate the change in
    # leaf count and r_branch up through its ancestors; report (alpha, r_branch[root]).
    # Finally controller.after_pruning(in_subtree).
    # Inputs: leaves_in_subtree uint8 (node_count) OUTPUT, expected zero-initialised;
    # orig_tree (fitted); controller. Outputs: leaves_in_subtree marks leaves of
    # the pruned tree; controller state updated.
    # Exceptions: none in the main loop (runs without the GIL); allocation of the
    # working numpy arrays may raise MemoryError.
    # KNOWN ISSUE: docstring names the parameter ccp_controller; it is controller.
    """Perform cost complexity pruning.

    This function takes an already grown tree, `orig_tree` and outputs a
    boolean mask `leaves_in_subtree` which are the leaves in the pruned tree.
    During the pruning process, the controller is passed the effective alpha and
    the subtree impurities. Furthermore, the controller signals when to stop
    pruning.

    Parameters
    ----------
    leaves_in_subtree : uint8_t[:]
        Output for leaves of subtree
    orig_tree : Tree
        Original tree
    ccp_controller : _CCPPruneController
        Cost complexity controller
    """

    cdef:
        intp_t i
        intp_t n_nodes = orig_tree.node_count
        # prior probability using weighted samples
        float64_t[:] weighted_n_node_samples = orig_tree.weighted_n_node_samples
        float64_t total_sum_weights = weighted_n_node_samples[0]
        float64_t[:] impurity = orig_tree.impurity
        # weighted impurity of each node
        float64_t[:] r_node = np.empty(shape=n_nodes, dtype=np.float64)

        intp_t[:] child_l = orig_tree.children_left
        intp_t[:] child_r = orig_tree.children_right
        intp_t[:] parent = np.zeros(shape=n_nodes, dtype=np.intp)

        stack[CostComplexityPruningRecord] ccp_stack
        CostComplexityPruningRecord stack_record
        intp_t node_idx
        stack[intp_t] node_indices_stack

        intp_t[:] n_leaves = np.zeros(shape=n_nodes, dtype=np.intp)
        float64_t[:] r_branch = np.zeros(shape=n_nodes, dtype=np.float64)
        float64_t current_r
        intp_t leaf_idx
        intp_t parent_idx

        # candidate nodes that can be pruned
        uint8_t[:] candidate_nodes = np.zeros(shape=n_nodes, dtype=np.uint8)
        # nodes in subtree
        uint8_t[:] in_subtree = np.ones(shape=n_nodes, dtype=np.uint8)
        intp_t pruned_branch_node_idx
        float64_t subtree_alpha
        float64_t effective_alpha
        intp_t n_pruned_leaves
        float64_t r_diff
        float64_t max_float64 = np.finfo(np.float64).max

    # find parent node ids and leaves
    with nogil:

        for i in range(r_node.shape[0]):
            r_node[i] = (
                weighted_n_node_samples[i] * impurity[i] / total_sum_weights)

        # Push the root node
        ccp_stack.push({"node_idx": 0, "parent": _TREE_UNDEFINED})

        while not ccp_stack.empty():
            stack_record = ccp_stack.top()
            ccp_stack.pop()

            node_idx = stack_record.node_idx
            parent[node_idx] = stack_record.parent

            if child_l[node_idx] == _TREE_LEAF:
                # ... and child_r[node_idx] == _TREE_LEAF:
                leaves_in_subtree[node_idx] = 1
            else:
                ccp_stack.push({"node_idx": child_l[node_idx], "parent": node_idx})
                ccp_stack.push({"node_idx": child_r[node_idx], "parent": node_idx})

        # computes number of leaves in all branches and the overall impurity of
        # the branch. The overall impurity is the sum of r_node in its leaves.
        for leaf_idx in range(leaves_in_subtree.shape[0]):
            if not leaves_in_subtree[leaf_idx]:
                continue
            r_branch[leaf_idx] = r_node[leaf_idx]

            # bubble up values to ancestor nodes
            current_r = r_node[leaf_idx]
            while leaf_idx != 0:
                parent_idx = parent[leaf_idx]
                r_branch[parent_idx] += current_r
                n_leaves[parent_idx] += 1
                leaf_idx = parent_idx

        for i in range(leaves_in_subtree.shape[0]):
            candidate_nodes[i] = not leaves_in_subtree[i]

        # save metrics before pruning
        controller.save_metrics(0.0, r_branch[0])

        # while root node is not a leaf
        while candidate_nodes[0]:

            # computes ccp_alpha for subtrees and finds the minimal alpha
            effective_alpha = max_float64
            for i in range(n_nodes):
                if not candidate_nodes[i]:
                    continue
                subtree_alpha = (r_node[i] - r_branch[i]) / (n_leaves[i] - 1)
                if subtree_alpha < effective_alpha:
                    effective_alpha = subtree_alpha
                    pruned_branch_node_idx = i

            if controller.stop_pruning(effective_alpha):
                break

            node_indices_stack.push(pruned_branch_node_idx)

            # descendants of branch are not in subtree
            while not node_indices_stack.empty():
                node_idx = node_indices_stack.top()
                node_indices_stack.pop()

                if not in_subtree[node_idx]:
                    continue  # branch has already been marked for pruning
                candidate_nodes[node_idx] = 0
                leaves_in_subtree[node_idx] = 0
                in_subtree[node_idx] = 0

                if child_l[node_idx] != _TREE_LEAF:
                    # ... and child_r[node_idx] != _TREE_LEAF:
                    node_indices_stack.push(child_l[node_idx])
                    node_indices_stack.push(child_r[node_idx])
            leaves_in_subtree[pruned_branch_node_idx] = 1
            in_subtree[pruned_branch_node_idx] = 1

            # updates number of leaves
            n_pruned_leaves = n_leaves[pruned_branch_node_idx] - 1
            n_leaves[pruned_branch_node_idx] = 0

            # computes the increase in r_branch to bubble up
            r_diff = r_node[pruned_branch_node_idx] - r_branch[pruned_branch_node_idx]
            r_branch[pruned_branch_node_idx] = r_node[pruned_branch_node_idx]

            # bubble up values to ancestors
            node_idx = parent[pruned_branch_node_idx]
            while node_idx != _TREE_UNDEFINED:
                n_leaves[node_idx] -= n_pruned_leaves
                r_branch[node_idx] += r_diff
                node_idx = parent[node_idx]

            controller.save_metrics(effective_alpha, r_branch[0])

        controller.after_pruning(in_subtree)


def _build_pruned_tree_ccp(
    Tree tree,  # OUT
    Tree orig_tree,
    float64_t ccp_alpha
):
    # [FUNCTION DOC]
    # Prune orig_tree with an _AlphaPruner at ccp_alpha, then build the pruned copy
    # into tree via _build_pruned_tree using the counted node capacity.
    # Inputs: tree (new empty Tree with same n_features/n_classes/n_outputs, output),
    # orig_tree, ccp_alpha >= 0. Returns None. Raises as _build_pruned_tree.
    """Build a pruned tree from the original tree using cost complexity
    pruning.

    The values and nodes from the original tree are copied into the pruned
    tree.

    Parameters
    ----------
    tree : Tree
        Location to place the pruned tree
    orig_tree : Tree
        Original tree
    ccp_alpha : positive float64_t
        Complexity parameter. The subtree with the largest cost complexity
        that is smaller than ``ccp_alpha`` will be chosen. By default,
        no pruning is performed.
    """

    cdef:
        intp_t n_nodes = orig_tree.node_count
        uint8_t[:] leaves_in_subtree = np.zeros(
            shape=n_nodes, dtype=np.uint8)

    pruning_controller = _AlphaPruner(ccp_alpha=ccp_alpha)

    _cost_complexity_prune(leaves_in_subtree, orig_tree, pruning_controller)

    _build_pruned_tree(tree, orig_tree, leaves_in_subtree,
                       pruning_controller.capacity)


def ccp_pruning_path(Tree orig_tree):
    # [FUNCTION DOC]
    # Run cost-complexity pruning to completion with a _PathFinder and return a dict
    # with 'ccp_alphas' (increasing effective alphas, starting at 0) and
    # 'impurities' (total leaf impurity of the subtree at each alpha) as float64
    # numpy arrays of equal length.
    # KNOWN ISSUE: docstring names the parameter tree; it is orig_tree.
    """Computes the cost complexity pruning path.

    Parameters
    ----------
    tree : Tree
        Original tree.

    Returns
    -------
    path_info : dict
        Information about pruning path with attributes:

        ccp_alphas : ndarray
            Effective alphas of subtree during pruning.

        impurities : ndarray
            Sum of the impurities of the subtree leaves for the
            corresponding alpha value in ``ccp_alphas``.
    """
    cdef:
        uint8_t[:] leaves_in_subtree = np.zeros(
            shape=orig_tree.node_count, dtype=np.uint8)

    path_finder = _PathFinder(orig_tree.node_count)

    _cost_complexity_prune(leaves_in_subtree, orig_tree, path_finder)

    cdef:
        uint32_t total_items = path_finder.count
        float64_t[:] ccp_alphas = np.empty(shape=total_items, dtype=np.float64)
        float64_t[:] impurities = np.empty(shape=total_items, dtype=np.float64)
        uint32_t count = 0

    while count < total_items:
        ccp_alphas[count] = path_finder.ccp_alphas[count]
        impurities[count] = path_finder.impurities[count]
        count += 1

    return {
        'ccp_alphas': np.asarray(ccp_alphas),
        'impurities': np.asarray(impurities),
    }


cdef struct BuildPrunedRecord:
    # [CLASS DOC]
    # Stack entry for _build_pruned_tree: original node id (start), depth, new
    # parent id and side.
    intp_t start
    intp_t depth
    intp_t parent
    bint is_left

cdef void _build_pruned_tree(
    Tree tree,  # OUT
    Tree orig_tree,
    const uint8_t[:] leaves_in_subtree,
    intp_t capacity
):
    # [FUNCTION DOC]
    # Resize tree to capacity, then depth-first (left first) copy nodes of orig_tree
    # starting at the root: each node is re-added with _add_node as a leaf if marked
    # in leaves_in_subtree (its descendants are skipped) or as a split node with its
    # children pushed. The value row is memcpy'd from the original node. Sets
    # tree.max_depth on success.
    # Inputs: tree (output, empty, same value_stride as orig_tree), orig_tree,
    # leaves_in_subtree uint8 mask (node_count), capacity (initial allocation).
    # Exceptions: MemoryError if a node cannot be added; ValueError if an original
    # leaf is reached that is not marked as leaf (invalid mask). Copy loop runs
    # without the GIL; exceptions are raised after re-acquiring it.
    """Build a pruned tree.

    Build a pruned tree from the original tree by transforming the nodes in
    ``leaves_in_subtree`` into leaves.

    Parameters
    ----------
    tree : Tree
        Location to place the pruned tree
    orig_tree : Tree
        Original tree
    leaves_in_subtree : uint8_t memoryview, shape=(node_count, )
        Boolean mask for leaves to include in subtree
    capacity : intp_t
        Number of nodes to initially allocate in pruned tree
    """
    tree._resize(capacity)

    cdef:
        intp_t orig_node_id
        intp_t new_node_id
        intp_t depth
        intp_t parent
        bint is_left
        bint is_leaf

        # value_stride for original tree and new tree are the same
        intp_t value_stride = orig_tree.value_stride
        intp_t max_depth_seen = -1
        int rc = 0
        Node* node
        float64_t* orig_value_ptr
        float64_t* new_value_ptr

        stack[BuildPrunedRecord] prune_stack
        BuildPrunedRecord stack_record

    with nogil:
        # push root node onto stack
        prune_stack.push({"start": 0, "depth": 0, "parent": _TREE_UNDEFINED, "is_left": 0})

        while not prune_stack.empty():
            stack_record = prune_stack.top()
            prune_stack.pop()

            orig_node_id = stack_record.start
            depth = stack_record.depth
            parent = stack_record.parent
            is_left = stack_record.is_left

            is_leaf = leaves_in_subtree[orig_node_id]
            node = &orig_tree.nodes[orig_node_id]

            # protect against an infinite loop as a runtime error, when leaves_in_subtree
            # are improperly set where a node is not marked as a leaf, but is a node
            # in the original tree. Thus, it violates the assumption that the node
            # is a leaf in the pruned tree, or has a descendant that will be pruned.
            if (not is_leaf and node.left_child == _TREE_LEAF
                    and node.right_child == _TREE_LEAF):
                rc = -2
                break

            new_node_id = tree._add_node(
                parent, is_left, is_leaf, node.feature, node.threshold,
                node.impurity, node.n_node_samples,
                node.weighted_n_node_samples, node.missing_go_to_left)

            if new_node_id == INTPTR_MAX:
                rc = -1
                break

            # copy value from original tree to new tree
            orig_value_ptr = orig_tree.value + value_stride * orig_node_id
            new_value_ptr = tree.value + value_stride * new_node_id
            memcpy(new_value_ptr, orig_value_ptr, sizeof(float64_t) * value_stride)

            if not is_leaf:
                # Push right child on stack
                prune_stack.push({"start": node.right_child, "depth": depth + 1,
                                  "parent": new_node_id, "is_left": 0})
                # push left child on stack
                prune_stack.push({"start": node.left_child, "depth": depth + 1,
                                  "parent": new_node_id, "is_left": 1})

            if depth > max_depth_seen:
                max_depth_seen = depth

        if rc >= 0:
            tree.max_depth = max_depth_seen
    if rc == -1:
        raise MemoryError("pruning tree")
    elif rc == -2:
        raise ValueError(
            "Node has reached a leaf in the original tree, but is not "
            "marked as a leaf in the leaves_in_subtree mask."
        )


def _build_pruned_tree_py(Tree tree, Tree orig_tree, const uint8_t[:] leaves_in_subtree):
    # [FUNCTION DOC]
    # Python-accessible wrapper of _build_pruned_tree with capacity orig node_count.
    # Raises ValueError if the mask length differs from orig_tree.node_count.
    """Build a pruned tree.

    Build a pruned tree from the original tree by transforming the nodes in
    ``leaves_in_subtree`` into leaves.

    Parameters
    ----------
    tree : Tree
        Location to place the pruned tree
    orig_tree : Tree
        Original tree
    leaves_in_subtree : uint8_t ndarray, shape=(node_count, )
        Boolean mask for leaves to include in subtree. The array must have
        the same size as the number of nodes in the original tree.
    """
    if leaves_in_subtree.shape[0] != orig_tree.node_count:
        raise ValueError(
            f"The length of leaves_in_subtree {len(leaves_in_subtree)} must be "
            f"equal to the number of nodes in the original tree {orig_tree.node_count}."
        )

    _build_pruned_tree(tree, orig_tree, leaves_in_subtree, orig_tree.node_count)
