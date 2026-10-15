# [MODULE DOC]
# Compiled prediction routines for a single fitted histogram GBDT tree stored as
# an array of node_struct records (TreePredictor.nodes). Provides parallel
# prediction from raw (unbinned float64) data, which handles numerical thresholds,
# categorical bitsets and NaN routing; parallel prediction from binned (uint8)
# data, used for early-stopping/validation on the training bins; and the
# weighted-traversal partial dependence computation (the "recursion" method).
# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

from cython.parallel import prange
from libc.math cimport isnan
import numpy as np

from sklearn.utils._bitset cimport BITSET_INNER_DTYPE_C, in_bitset_2d_memoryview
from sklearn.utils._typedefs cimport intp_t, uint8_t
from sklearn.ensemble._hist_gradient_boosting.common cimport X_DTYPE_C
from sklearn.ensemble._hist_gradient_boosting.common cimport Y_DTYPE_C
from sklearn.ensemble._hist_gradient_boosting.common import Y_DTYPE
from sklearn.ensemble._hist_gradient_boosting.common cimport X_BINNED_DTYPE_C
from sklearn.ensemble._hist_gradient_boosting.common cimport node_struct


def _predict_from_raw_data(  # raw data = non-binned data
        const node_struct [:] nodes,
        const X_DTYPE_C [:, :] numeric_data,
        const BITSET_INNER_DTYPE_C [:, ::1] raw_left_cat_bitsets,
        const BITSET_INNER_DTYPE_C [:, ::1] known_cat_bitsets,
        const unsigned int [::1] f_idx_map,
        int n_threads,
        Y_DTYPE_C [:] out):
    # [FUNCTION DOC]
    # Behavior: for each row i (OpenMP prange, static schedule, GIL released) writes
    # the tree prediction of numeric_data[i] into out[i] using the per-row helper.
    # Inputs: nodes node_struct array (root at index 0); numeric_data float64
    # (n_samples, n_features), NaN means missing; raw_left_cat_bitsets uint32
    # (n_categorical_splits, 8) bitsets of raw categories going left, indexed by
    # node.bitset_idx; known_cat_bitsets uint32 (n_categorical_features, 8) bitsets of
    # categories seen at fit time; f_idx_map uint32 (n_features,) mapping a feature
    # index to its row in known_cat_bitsets; n_threads; out float64 (n_samples,).
    # Outputs: out overwritten in place (not accumulated).
    # Exceptions: none raised from the parallel loop.
    # Side effects: mutates out; OpenMP parallelism.

    cdef:
        int i

    for i in prange(numeric_data.shape[0], schedule='static', nogil=True,
                    num_threads=n_threads):
        out[i] = _predict_one_from_raw_data(
            nodes, numeric_data, raw_left_cat_bitsets,
            known_cat_bitsets,
            f_idx_map, i)


cdef inline Y_DTYPE_C _predict_one_from_raw_data(
        const node_struct [:] nodes,
        const X_DTYPE_C [:, :] numeric_data,
        const BITSET_INNER_DTYPE_C [:, ::1] raw_left_cat_bitsets,
        const BITSET_INNER_DTYPE_C [:, ::1] known_cat_bitsets,
        const unsigned int [::1] f_idx_map,
        const int row) noexcept nogil:
    # [FUNCTION DOC]
    # HELPER: single-row raw-data traversal used by _predict_from_raw_data.
    # Behavior: starting from nodes[0], until a leaf is reached: read the row's value
    # of the node's feature. If NaN, go left iff missing_go_to_left. Else if the node
    # is categorical: a negative value is treated as missing; a category present in
    # the node's raw_left_cat_bitsets row goes left; a category known at fit time
    # (present in known_cat_bitsets row f_idx_map[feature]) but not in the left bitset
    # goes right; an unknown category is treated as missing. Else numerical: go left
    # iff value <= num_threshold. Returns the leaf's value.
    # Inputs: as in _predict_from_raw_data plus row, the row index (the full arrays
    # are passed rather than a row slice to work around a Cython prange issue).
    # Outputs: returns the leaf value (float64).
    # Exceptions: none (noexcept nogil).
    # Side effects: none.
    # KNOWN ISSUE: categorical values are cast to uint8 without an upper-range check,
    # so a raw category >= 256 (or a non-integer value) is converted by C float to
    # unsigned char conversion, which is undefined for out-of-range values and may
    # alias a valid category; this function relies on callers never passing such
    # values.
    # Need to pass the whole array and the row index, else prange won't work.
    # See issue Cython #2798

    cdef:
        node_struct node = nodes[0]
        unsigned int node_idx = 0
        X_DTYPE_C data_val

    while True:
        if node.is_leaf:
            return node.value

        data_val = numeric_data[row, node.feature_idx]

        if isnan(data_val):
            if node.missing_go_to_left:
                node_idx = node.left
            else:
                node_idx = node.right
        elif node.is_categorical:
            if data_val < 0:
                # data_val is not in the accepted range, so it is treated as missing value
                node_idx = node.left if node.missing_go_to_left else node.right
            elif in_bitset_2d_memoryview(
                    raw_left_cat_bitsets,
                    <X_BINNED_DTYPE_C>data_val,
                    node.bitset_idx):
                node_idx = node.left
            elif in_bitset_2d_memoryview(
                    known_cat_bitsets,
                    <X_BINNED_DTYPE_C>data_val,
                    f_idx_map[node.feature_idx]):
                node_idx = node.right
            else:
                # Treat unknown categories as missing.
                node_idx = node.left if node.missing_go_to_left else node.right
        else:
            if data_val <= node.num_threshold:
                node_idx = node.left
            else:
                node_idx = node.right
        node = nodes[node_idx]


def _predict_from_binned_data(
        node_struct [:] nodes,
        const X_BINNED_DTYPE_C [:, :] binned_data,
        BITSET_INNER_DTYPE_C [:, :] binned_left_cat_bitsets,
        const uint8_t missing_values_bin_idx,
        int n_threads,
        Y_DTYPE_C [:] out):
    # [FUNCTION DOC]
    # Behavior: parallel (OpenMP prange, GIL released) prediction from binned data:
    # out[i] is set to the tree prediction of binned_data[i] via the per-row helper.
    # Inputs: nodes node_struct array; binned_data uint8 (n_samples, n_features);
    # binned_left_cat_bitsets uint32 (n_categorical_splits, 8) bitsets of bin codes
    # going left; missing_values_bin_idx the bin code reserved for missing values;
    # n_threads; out float64 (n_samples,).
    # Outputs: out overwritten in place.
    # Exceptions: none from the parallel loop.
    # Side effects: mutates out; OpenMP parallelism.

    cdef:
        int i

    for i in prange(binned_data.shape[0], schedule='static', nogil=True,
                    num_threads=n_threads):
        out[i] = _predict_one_from_binned_data(nodes,
                                               binned_data,
                                               binned_left_cat_bitsets, i,
                                               missing_values_bin_idx)


cdef inline Y_DTYPE_C _predict_one_from_binned_data(
        node_struct [:] nodes,
        const X_BINNED_DTYPE_C [:, :] binned_data,
        const BITSET_INNER_DTYPE_C [:, :] binned_left_cat_bitsets,
        const int row,
        const uint8_t missing_values_bin_idx) noexcept nogil:
    # [FUNCTION DOC]
    # HELPER: single-row binned-data traversal used by _predict_from_binned_data.
    # Behavior: from nodes[0] until a leaf: if the row's bin equals
    # missing_values_bin_idx, go left iff missing_go_to_left; else if categorical go
    # left iff the bin is in the node's binned_left_cat_bitsets row; else go left iff
    # bin <= bin_threshold. Returns the leaf value.
    # Inputs: nodes, binned_data, binned_left_cat_bitsets, row index,
    # missing_values_bin_idx.
    # Outputs: returns the leaf value.
    # Exceptions: none (noexcept nogil).
    # Side effects: none.
    # Need to pass the whole array and the row index, else prange won't work.
    # See issue Cython #2798

    cdef:
        node_struct node = nodes[0]
        unsigned int node_idx = 0
        X_BINNED_DTYPE_C data_val

    while True:
        if node.is_leaf:
            return node.value

        data_val = binned_data[row, node.feature_idx]

        if data_val == missing_values_bin_idx:
            if node.missing_go_to_left:
                node_idx = node.left
            else:
                node_idx = node.right
        elif node.is_categorical:
            if in_bitset_2d_memoryview(
                    binned_left_cat_bitsets,
                    data_val,
                    node.bitset_idx):
                node_idx = node.left
            else:
                node_idx = node.right
        else:
            if data_val <= node.bin_threshold:
                node_idx = node.left
            else:
                node_idx = node.right
        node = nodes[node_idx]


def _compute_partial_dependence(
    node_struct [:] nodes,
    const X_DTYPE_C [:, ::1] X,
    const intp_t [:] target_features,
    Y_DTYPE_C [:] out
):
    # [FUNCTION DOC]
    # Behavior: computes the tree's partial dependence on target_features at each grid
    # point using an explicit stack (depth-first). Each row starts with the root at
    # weight 1. At an internal node splitting on a target feature, only the child
    # selected by comparing the grid value against num_threshold is pushed, with
    # unchanged weight. At a node splitting on another feature both children are
    # pushed, the left weighted by count(left)/count(node) and the right by the
    # remainder. At a leaf, weight times leaf value is added to out[row]. After each
    # row the accumulated leaf weights are checked to sum to about 1.
    # Inputs: nodes node_struct array; X float64 C-contiguous (n_grid_points,
    # n_target_features), column j corresponds to target_features[j];
    # target_features intp (n_target_features,) feature indices; out float64
    # (n_grid_points,) that must be zero-initialized by the caller.
    # Outputs: out accumulated in place; returns None.
    # Exceptions: ValueError if the accumulated weight is not within (0.999, 1.001).
    # Side effects: allocates two stack arrays of size n_nodes; holds the GIL.
    # KNOWN ISSUE: target-feature splits always use the numerical threshold and
    # ignore is_categorical and missing_go_to_left, so categorical splits and NaN grid
    # values are not handled; this relies on the Python callers not requesting the
    # recursion method for such cases.
    """Partial dependence of the response on the ``target_features`` set.

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
    nodes : view on array of PREDICTOR_RECORD_DTYPE, shape (n_nodes)
        The array representing the predictor tree.
    X : view on 2d ndarray, shape (n_samples, n_target_features)
        The grid points on which the partial dependence should be
        evaluated.
    target_features : view on 1d ndarray of intp_t, shape (n_target_features)
        The set of target features for which the partial dependence
        should be evaluated.
    out : view on 1d ndarray, shape (n_samples)
        The value of the partial dependence function on each grid
        point.
    """

    cdef:
        unsigned int current_node_idx
        unsigned int [:] node_idx_stack = np.zeros(shape=nodes.shape[0],
                                                   dtype=np.uint32)
        Y_DTYPE_C [::1] weight_stack = np.zeros(shape=nodes.shape[0],
                                                dtype=Y_DTYPE)
        node_struct * current_node  # pointer to avoid copying attributes

        unsigned int sample_idx
        intp_t feature_idx
        unsigned stack_size
        Y_DTYPE_C left_sample_frac
        Y_DTYPE_C current_weight
        Y_DTYPE_C total_weight  # used for sanity check only
        bint is_target_feature

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
            current_node = &nodes[current_node_idx]

            if current_node.is_leaf:
                out[sample_idx] += (weight_stack[stack_size] *
                                    current_node.value)
                total_weight += weight_stack[stack_size]
            else:
                # determine if the split feature is a target feature
                is_target_feature = False
                for feature_idx in range(target_features.shape[0]):
                    if target_features[feature_idx] == current_node.feature_idx:
                        is_target_feature = True
                        break

                if is_target_feature:
                    # In this case, we push left or right child on stack
                    if X[sample_idx, feature_idx] <= current_node.num_threshold:
                        node_idx_stack[stack_size] = current_node.left
                    else:
                        node_idx_stack[stack_size] = current_node.right
                    stack_size += 1
                else:
                    # In this case, we push both children onto the stack,
                    # and give a weight proportional to the number of
                    # samples going through each branch.

                    # push left child
                    node_idx_stack[stack_size] = current_node.left
                    left_sample_frac = (
                        <Y_DTYPE_C> nodes[current_node.left].count /
                        current_node.count)
                    current_weight = weight_stack[stack_size]
                    weight_stack[stack_size] = current_weight * left_sample_frac
                    stack_size += 1

                    # push right child
                    node_idx_stack[stack_size] = current_node.right
                    weight_stack[stack_size] = (
                        current_weight * (1 - left_sample_frac))
                    stack_size += 1

        # Sanity check. Should never happen.
        if not (0.999 < total_weight < 1.001):
            raise ValueError("Total weight should be 1.0 but was %.9f" %total_weight)
