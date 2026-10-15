# [MODULE DOC]
# Declarations for the split-search layer of tree building: the SplitRecord
# struct describing a chosen split, and the Splitter base class with its
# data fields and nogil interface (init, node_reset, node_split, node_value,
# clip_node_value, node_impurity) used by the tree builders in _tree.pyx.
# Concrete splitters (best/random, dense/sparse) are defined in
# _splitter.pyx; impurity computations are delegated to a Criterion from
# _criterion.
# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

# See _splitter.pyx for details.

from sklearn.utils._typedefs cimport (
    float32_t, float64_t, int8_t, int32_t, intp_t, uint8_t, uint32_t
)
from sklearn.tree._criterion cimport Criterion
from sklearn.tree._tree cimport ParentInfo


cdef struct SplitRecord:
    # [CLASS DOC]
    # Result of a node split search. feature and threshold define the test
    # (value <= threshold goes left); pos is the boundary in samples so the left
    # child is samples[start:pos] and the right child samples[pos:end] (pos == end
    # means no valid split, i.e. leaf); improvement is the weighted impurity
    # decrease; impurity_left / impurity_right are child impurities;
    # lower_bound / upper_bound are monotonic-constraint bounds (not set by the
    # splitters in this package); missing_go_to_left says where NaNs are sent.
    # Data to track sample split
    intp_t feature         # Which feature to split on.
    intp_t pos             # Split samples array at the given position,
    #                      # i.e. count of samples below threshold for feature.
    #                      # pos is >= end if the node is a leaf.
    float64_t threshold       # Threshold to split at.
    float64_t improvement     # Impurity improvement given parent node.
    float64_t impurity_left   # Impurity of the left split.
    float64_t impurity_right  # Impurity of the right split.
    float64_t lower_bound     # Lower bound on value of both children for monotonicity
    float64_t upper_bound     # Upper bound on value of both children for monotonicity
    uint8_t missing_go_to_left  # Controls if missing values go to the left node.


cdef class Splitter:
    # [CLASS DOC]
    # Base splitter. Owns the array samples of row indices with positive weight
    # (node samples are the contiguous slice samples[start:end]), the feature
    # permutation features, the constant_features buffer passed from parents to
    # children, the feature_values scratch buffer, y, sample weights, monotonic
    # constraints, split hyper-parameters (max_features, min_samples_leaf,
    # min_weight_leaf), the criterion, and a 32-bit RNG state seeded from
    # random_state at init.
    # The splitter searches in the input space for a feature and a threshold
    # to split the samples samples[start:end].
    #
    # The impurity computations are delegated to a criterion object.

    # Internal structures
    cdef public Criterion criterion      # Impurity criterion
    cdef public intp_t max_features      # Number of features to test
    cdef public intp_t min_samples_leaf  # Min samples in a leaf
    cdef public float64_t min_weight_leaf   # Minimum weight in a leaf

    cdef object random_state             # Random state
    cdef uint32_t rand_r_state           # sklearn_rand_r random number state

    cdef intp_t[::1] samples             # Sample indices in X, y
    cdef intp_t n_samples                # X.shape[0]
    cdef float64_t weighted_n_samples       # Weighted number of samples
    cdef intp_t[::1] features            # Feature indices in X
    cdef intp_t[::1] constant_features   # Constant features indices
    cdef intp_t n_features               # X.shape[1]
    cdef float32_t[::1] feature_values   # temp. array holding feature values

    cdef intp_t start                    # Start position for the current node
    cdef intp_t end                      # End position for the current node

    cdef const float64_t[:, ::1] y
    # Monotonicity constraints for each feature.
    # The encoding is as follows:
    #   -1: monotonic decrease
    #    0: no constraint
    #   +1: monotonic increase
    cdef const int8_t[:] monotonic_cst
    cdef bint with_monotonic_cst
    cdef const float64_t[:] sample_weight

    # The samples vector `samples` is maintained by the Splitter object such
    # that the samples contained in a node are contiguous. With this setting,
    # `node_split` reorganizes the node samples `samples[start:end]` in two
    # subsets `samples[start:pos]` and `samples[pos:end]`.

    # The 1-d  `features` array of size n_features contains the features
    # indices and allows fast sampling without replacement of features.

    # The 1-d `constant_features` array of size n_features holds in
    # `constant_features[:n_constant_features]` the feature ids with
    # constant values for all the samples that reached a specific node.
    # The value `n_constant_features` is given by the parent node to its
    # child nodes.  The content of the range `[n_constant_features:]` is left
    # undefined, but preallocated for performance reasons
    # This allows optimization with depth-based tree building.

    # Methods
    cdef int init(
        self,
        object X,
        const float64_t[:, ::1] y,
        const float64_t[:] sample_weight,
        const uint8_t[::1] missing_values_in_feature_mask,
    ) except -1
    # [FUNCTION DOC]
    # Declaration of init: bind data and allocate buffers (requires the GIL).
    # See _splitter.pyx for details.

    cdef int node_reset(
        self,
        intp_t start,
        intp_t end,
        float64_t* weighted_n_node_samples
    ) except -1 nogil
    # [FUNCTION DOC]
    # Declaration of node_reset: select samples[start:end] as the current node
    # and initialize the criterion. See _splitter.pyx for details.

    cdef int node_split(
        self,
        ParentInfo* parent,
        SplitRecord* split,
    ) except -1 nogil
    # [FUNCTION DOC]
    # Declaration of node_split: search a split of the current node and write it
    # to split. See _splitter.pyx for details.

    cdef void node_value(self, float64_t* dest) noexcept nogil
    # [FUNCTION DOC]
    # Declaration of node_value: write the node's prediction value into dest.

    cdef void clip_node_value(self, float64_t* dest, float64_t lower_bound, float64_t upper_bound) noexcept nogil
    # [FUNCTION DOC]
    # Declaration of clip_node_value: clip dest into [lower_bound, upper_bound].

    cdef float64_t node_impurity(self) noexcept nogil
    # [FUNCTION DOC]
    # Declaration of node_impurity: impurity of the current node.
