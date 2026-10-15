# [MODULE DOC]
# Declarations of the two partitioner classes used by the splitters in
# _splitter.pyx: DensePartitioner (2-D float32 X) and SparsePartitioner
# (CSC float32 X). Both expose the same set of nogil methods (sorting a
# node's samples by a feature, min/max lookup, iterating candidate split
# positions, partitioning samples by a threshold) so that _splitter can use
# them through a fused type instead of a virtual base class (the commented
# BasePartitioner documents the shared interface). Also declares
# FEATURE_THRESHOLD (1e-7), the tolerance under which two feature values are
# treated as equal, and the fused type array_data_type (intp or float32)
# used by swap_array_slices.
# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

# See _partitioner.pyx for details.

from sklearn.utils._typedefs cimport (
    float32_t, float64_t, int32_t, intp_t, uint8_t
)
from sklearn.tree._splitter cimport SplitRecord


# Mitigate precision differences between 32 bit and 64 bit
cdef const float32_t FEATURE_THRESHOLD = 1e-7


# We provide here the abstract interface for a Partitioner that would be
# theoretically shared between the Dense and Sparse partitioners. However,
# we leave it commented out for now as it is not used in the current
# implementation due to the performance hit from vtable lookups when using
# inheritance based polymorphism. It is left here for future reference.
#
# Note: Instead, in `_splitter.pyx`, we define a fused type that can be used
# to represent both the dense and sparse partitioners.
#
# cdef class BasePartitioner:
#     cdef intp_t[::1] samples
#     cdef float32_t[::1] feature_values
#     cdef intp_t start
#     cdef intp_t end
#     cdef intp_t n_missing
#     cdef const uint8_t[::1] missing_values_in_feature_mask

#     cdef void sort_samples_and_feature_values(
#         self, intp_t current_feature
#     ) noexcept nogil
#     cdef void init_node_split(
#         self,
#         intp_t start,
#         intp_t end
#     ) noexcept nogil
#     cdef void find_min_max(
#         self,
#         intp_t current_feature,
#         float32_t* min_feature_value_out,
#         float32_t* max_feature_value_out,
#     ) noexcept nogil
#     cdef void next_p(
#         self,
#         intp_t* p_prev,
#         intp_t* p,
#         bint missing_go_to_left
#     ) noexcept nogil
#     cdef intp_t partition_samples(
#         self,
#         float64_t current_threshold
#     ) noexcept nogil
#     cdef void partition_samples_final(
#         self,
#         const SplitRecord* best_split,
#     ) noexcept nogil


cdef class DensePartitioner:
    # [CLASS DOC]
    # Moves sample indices of the current node samples[start:end] for dense X.
    # Fields: X (read-only n_samples x n_features float32), samples (shared with
    # the Splitter, reordered in place), feature_values (shared scratch buffer
    # aligned with samples), start/end of the current node, n_missing (NaN count
    # for the last examined feature), missing_values_in_feature_mask (per-feature
    # flag saying NaNs may occur, or None), swap_buffer (scratch bytes used to
    # move missing values to the left).
    """Partitioner specialized for dense data.

    Note that this partitioner is agnostic to the splitting strategy (best vs. random).
    """
    cdef const float32_t[:, :] X
    cdef intp_t[::1] samples
    cdef float32_t[::1] feature_values
    cdef intp_t start
    cdef intp_t end
    cdef intp_t n_missing
    cdef const uint8_t[::1] missing_values_in_feature_mask
    cdef char[::1] swap_buffer

    cdef void sort_samples_and_feature_values(
        self, intp_t current_feature
    ) noexcept nogil
    # [FUNCTION DOC]
    # Declaration: fill feature_values for one feature, sort samples by it with
    # NaNs grouped at the right, set n_missing.
    # See _partitioner.pyx for the full description.
    cdef void shift_missing_to_the_left(self) noexcept nogil
    # [FUNCTION DOC]
    # Declaration: rotate the node slice so the NaN block moves to the left.
    # See _partitioner.pyx for the full description.
    cdef void init_node_split(
        self,
        intp_t start,
        intp_t end
    ) noexcept nogil
    # [FUNCTION DOC]
    # Declaration: set the current node range and reset n_missing.
    # See _partitioner.pyx for the full description.
    cdef void find_min_max(
        self,
        intp_t current_feature,
        float32_t* min_feature_value_out,
        float32_t* max_feature_value_out,
    ) noexcept nogil
    # [FUNCTION DOC]
    # Declaration: fill feature_values for one feature and return its min/max
    # over non-missing values through the two output pointers.
    # See _partitioner.pyx for the full description.
    cdef void next_p(
        self,
        intp_t* p_prev,
        intp_t* p,
        bint missing_go_to_left
    ) noexcept nogil
    # [FUNCTION DOC]
    # Declaration: advance the candidate split position p (and p_prev) over
    # sorted feature values, skipping near-ties.
    # See _partitioner.pyx for the full description.
    cdef intp_t partition_samples(
        self,
        float64_t current_threshold,
        bint missing_go_to_left
    ) noexcept nogil
    # [FUNCTION DOC]
    # Declaration: partition by a threshold on the current feature_values and
    # return the split position.
    # See _partitioner.pyx for the full description.
    cdef void partition_samples_final(
        self,
        const SplitRecord* best_split,
    ) noexcept nogil
    # [FUNCTION DOC]
    # Declaration: partition samples by the feature/threshold/missing direction
    # of a SplitRecord, reading X directly.
    # See _partitioner.pyx for the full description.


cdef class SparsePartitioner:
    # [CLASS DOC]
    # Same role as DensePartitioner for CSC sparse X. Extra fields: CSC arrays
    # X_data / X_indices / X_indptr, n_total_samples (rows of X),
    # index_to_samples (row index to position in samples, -1 if the row is not
    # used, kept in sync with every swap), sorted_samples (sorted copy of the node
    # samples for the binary-search extraction), end_negative / start_positive
    # (after extraction, samples[start:end_negative] have negative values,
    # samples[start_positive:end] positive values, the rest are implicit zeros),
    # is_samples_sorted (whether sorted_samples is valid for the current node).
    # Missing values are not supported; n_missing is always 0.
    """Partitioner specialized for sparse CSC data.

    Note that this partitioner is agnostic to the splitting strategy (best vs. random).
    """
    cdef const float32_t[::1] X_data
    cdef const int32_t[::1] X_indices
    cdef const int32_t[::1] X_indptr
    cdef intp_t n_total_samples
    cdef intp_t[::1] index_to_samples
    cdef intp_t[::1] sorted_samples
    cdef intp_t start_positive
    cdef intp_t end_negative
    cdef bint is_samples_sorted

    cdef intp_t[::1] samples
    cdef float32_t[::1] feature_values
    cdef intp_t start
    cdef intp_t end
    cdef intp_t n_missing
    cdef const uint8_t[::1] missing_values_in_feature_mask

    cdef void sort_samples_and_feature_values(
        self, intp_t current_feature
    ) noexcept nogil
    # [FUNCTION DOC]
    # Declaration: extract non-zeros of a feature, sort negatives and positives,
    # insert zero markers.
    # See _partitioner.pyx for the full description.
    cdef void shift_missing_to_the_left(self) noexcept nogil
    # [FUNCTION DOC]
    # Declaration: no-op (sparse data has no missing values).
    # See _partitioner.pyx for the full description.
    cdef void init_node_split(
        self,
        intp_t start,
        intp_t end
    ) noexcept nogil
    # [FUNCTION DOC]
    # Declaration: set node range, invalidate sorted_samples, reset n_missing.
    # 
    # See _partitioner.pyx for the full description.
    cdef void find_min_max(
        self,
        intp_t current_feature,
        float32_t* min_feature_value_out,
        float32_t* max_feature_value_out,
    ) noexcept nogil
    # [FUNCTION DOC]
    # Declaration: min/max of a feature over the node including implicit zeros.
    # 
    # See _partitioner.pyx for the full description.
    cdef void next_p(
        self,
        intp_t* p_prev,
        intp_t* p,
        bint missing_go_to_left
    ) noexcept nogil
    # [FUNCTION DOC]
    # Declaration: advance split position, jumping over the implicit zero block.
    # 
    # See _partitioner.pyx for the full description.
    cdef intp_t partition_samples(
        self,
        float64_t current_threshold,
        bint missing_go_to_left,
    ) noexcept nogil
    # [FUNCTION DOC]
    # Declaration: partition by threshold (missing_go_to_left ignored).
    # See _partitioner.pyx for the full description.
    cdef void partition_samples_final(
        self,
        const SplitRecord* best_split,
    ) noexcept nogil
    # [FUNCTION DOC]
    # Declaration: re-extract the best feature and partition by its threshold.
    # 
    # See _partitioner.pyx for the full description.

    cdef void extract_nnz(
        self,
        intp_t feature
    ) noexcept nogil
    # [FUNCTION DOC]
    # Declaration: gather a feature's non-zero values for the node into
    # negative/positive blocks.
    # See _partitioner.pyx for the full description.
    cdef intp_t _partition(
        self,
        float64_t threshold
    ) noexcept nogil
    # [FUNCTION DOC]
    # Declaration: threshold partition of the negative or positive block.
    # 
    # See _partitioner.pyx for the full description.


ctypedef fused array_data_type:
    intp_t
    float32_t
