# [MODULE DOC]
# Declaration file for split criteria (_criterion.pyx). Declares the abstract
# Criterion extension type, whose state tracks a node's samples
# sample_indices[start:end] split at pos into left [start:pos) and right
# [pos:end), and the two partially concrete bases ClassificationCriterion
# (weighted class-count tables) and RegressionCriterion (weighted sums of y).
# Splitters cimport these to evaluate candidate splits without the GIL.
# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

# See _criterion.pyx for implementation details.
from sklearn.utils._typedefs cimport float64_t, int8_t, intp_t


cdef class Criterion:
    # [CLASS DOC]
    # Abstract impurity criterion. Holds references to y, sample_weight and the
    # splitter's sample_indices, the node bounds start/pos/end and the weighted
    # sample counts of node, left and right children. Concrete subclasses supply
    # statistics, impurity and node-value computations.
    # The criterion computes the impurity of a node and the reduction of
    # impurity of a split on that node. It also computes the output statistics
    # such as the mean in regression and class probabilities in classification.

    # Internal structures
    cdef const float64_t[:, ::1] y         # Values of y
    cdef const float64_t[:] sample_weight  # Sample weights

    cdef const intp_t[:] sample_indices    # Sample indices in X, y
    cdef intp_t start                      # samples[start:pos] are the samples in the left node
    cdef intp_t pos                        # samples[pos:end] are the samples in the right node
    cdef intp_t end

    cdef intp_t n_outputs                  # Number of outputs
    cdef intp_t n_samples                  # Number of samples
    cdef intp_t n_node_samples             # Number of samples in the node (end-start)
    cdef float64_t weighted_n_samples         # Weighted number of samples (in total)
    cdef float64_t weighted_n_node_samples    # Weighted number of samples in the node
    cdef float64_t weighted_n_left            # Weighted number of samples in the left node
    cdef float64_t weighted_n_right           # Weighted number of samples in the right node

    # The criterion object is maintained such that left and right collected
    # statistics correspond to samples[start:pos] and samples[pos:end].

    # Methods
    cdef int init(
        self,
        const float64_t[:, ::1] y,
        const float64_t[:] sample_weight,
        float64_t weighted_n_samples,
        const intp_t[:] sample_indices,
        intp_t start,
        intp_t end
    ) except -1 nogil
    # [FUNCTION DOC]
    # Bind the criterion to node sample_indices[start:end], compute node statistics
    # and reset to pos=start. Returns 0, or -1 with an exception set.
    cdef int reset(self) except -1 nogil
    # [FUNCTION DOC]
    # Move split position to start (all samples on the right). Returns 0 or -1.
    cdef int reverse_reset(self) except -1 nogil
    # [FUNCTION DOC]
    # Move split position to end (all samples on the left). Returns 0 or -1.
    cdef int update(self, intp_t new_pos) except -1 nogil
    # [FUNCTION DOC]
    # Advance split position to new_pos, moving samples from right to left stats.
    cdef float64_t node_impurity(self) noexcept nogil
    # [FUNCTION DOC]
    # Return impurity of the whole node [start:end). Cannot raise.
    cdef void children_impurity(
        self,
        float64_t* impurity_left,
        float64_t* impurity_right
    ) noexcept nogil
    # [FUNCTION DOC]
    # Write impurities of left [start:pos) and right [pos:end) into the two pointers.
    cdef void node_value(
        self,
        float64_t* dest
    ) noexcept nogil
    # [FUNCTION DOC]
    # Write the node prediction (class probabilities or regression value per output) to dest.
    cdef void clip_node_value(
        self,
        float64_t* dest,
        float64_t lower_bound,
        float64_t upper_bound
    ) noexcept nogil
    # [FUNCTION DOC]
    # Clamp the prediction in dest to [lower_bound, upper_bound] (monotonic constraints).
    cdef float64_t middle_value(self) noexcept nogil
    # [FUNCTION DOC]
    # Return the midpoint between left and right child values, used as the shared
    # bound passed to children under a monotonic constraint.
    cdef float64_t impurity_improvement(
        self,
        float64_t impurity_parent,
        float64_t impurity_left,
        float64_t impurity_right
    ) noexcept nogil
    # [FUNCTION DOC]
    # Return weighted impurity decrease of the current split relative to all samples.
    cdef float64_t proxy_impurity_improvement(self) noexcept nogil
    # [FUNCTION DOC]
    # Return a cheap quantity with the same argmax as impurity_improvement over splits.
    cdef bint check_monotonicity(
            self,
            int8_t monotonic_cst,
            float64_t lower_bound,
            float64_t upper_bound,
    ) noexcept nogil
    # [FUNCTION DOC]
    # Return whether the current split satisfies the monotonic constraint and bounds.
    cdef inline bint _check_monotonicity(
            self,
            int8_t monotonic_cst,
            float64_t lower_bound,
            float64_t upper_bound,
            float64_t sum_left,
            float64_t sum_right,
    ) noexcept nogil
    # [FUNCTION DOC]
    # HELPER: shared test used by check_monotonicity implementations.
    # True when both child values lie within the bounds and their order agrees
    # with monotonic_cst (+1 increasing, -1 decreasing, 0 none).
    # KNOWN ISSUE: the last two parameters are named sum_left/sum_right here but
    # value_left/value_right in _criterion.pyx; they are child mean values, not sums.

cdef class ClassificationCriterion(Criterion):
    # [CLASS DOC]
    # Classification base: per output k, sum_total/sum_left/sum_right[k, c] hold
    # weighted counts of class c (rows padded to max_n_classes columns).
    """Abstract criterion for classification."""

    cdef intp_t[::1] n_classes
    cdef intp_t max_n_classes

    cdef float64_t[:, ::1] sum_total    # The sum of the weighted count of each label.
    cdef float64_t[:, ::1] sum_left     # Same as above, but for the left side of the split
    cdef float64_t[:, ::1] sum_right    # Same as above, but for the right side of the split

cdef class RegressionCriterion(Criterion):
    # [CLASS DOC]
    # Regression base: per output, weighted sums of y for node/left/right, plus the
    # node's weighted sum of y squared (sq_sum_total).
    """Abstract regression criterion."""

    cdef float64_t sq_sum_total

    cdef float64_t[::1] sum_total    # The sum of w*y.
    cdef float64_t[::1] sum_left     # Same as above, but for the left side of the split
    cdef float64_t[::1] sum_right    # Same as above, but for the right side of the split
