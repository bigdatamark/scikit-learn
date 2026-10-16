# [MODULE DOC]
# Histogram construction for histogram gradient boosting. HistogramBuilder
# computes, for every (allowed) feature of a tree node, an array of n_bins
# hist_struct entries holding per-bin sums of gradients, sums of hessians and
# counts. Histograms are built either from scratch over the node's samples
# (parallel over features, with specialized root / constant-hessian kernels whose
# loops are manually unrolled by 4 for vectorization) or by the subtraction trick
# hist(child) = hist(parent) - hist(sibling). The module-level cpdef kernels are
# also exposed to Python for testing.
"""This module contains routines for building histograms."""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

cimport cython
from cython.parallel import prange
from libc.string cimport memset

import numpy as np

from sklearn.ensemble._hist_gradient_boosting.common import HISTOGRAM_DTYPE
from sklearn.ensemble._hist_gradient_boosting.common cimport hist_struct
from sklearn.ensemble._hist_gradient_boosting.common cimport X_BINNED_DTYPE_C
from sklearn.ensemble._hist_gradient_boosting.common cimport G_H_DTYPE_C
from sklearn.utils._typedefs cimport uint8_t


# Notes:
# - IN views are read-only, OUT views are write-only
# - In a lot of functions here, we pass feature_idx and the whole 2d
#   histograms arrays instead of just histograms[feature_idx]. This is because
#   Cython generated C code will have strange Python interactions (likely
#   related to the GIL release and the custom histogram dtype) when using 1d
#   histogram arrays that come from 2d arrays.
# - The for loops are un-wrapped, for example:
#
#   for i in range(n):
#       array[i] = i
#
#   will become
#
#   for i in range(n // 4):
#       array[i] = i
#       array[i + 1] = i + 1
#       array[i + 2] = i + 2
#       array[i + 3] = i + 3
#
#   This is to hint gcc that it can auto-vectorize these 4 operations and
#   perform them all at once.


@cython.final
cdef class HistogramBuilder:
    # [CLASS DOC]
    # Builds per-node histograms for one tree. Holds the Fortran-ordered binned data,
    # the current gradients/hessians (float32), scratch arrays ordered_gradients and
    # ordered_hessians (gradients/hessians gathered in node sample order to improve
    # cache locality), n_bins, a flag for constant hessians, and n_threads. All
    # attributes are public (readable/writable from Python).
    """A Histogram builder... used to build histograms.

    A histogram is an array with n_bins entries of type HISTOGRAM_DTYPE. Each
    feature has its own histogram. A histogram contains the sum of gradients
    and hessians of all the samples belonging to each bin.

    There are different ways to build a histogram:
    - by subtraction: hist(child) = hist(parent) - hist(sibling)
    - from scratch. In this case we have routines that update the hessians
      or not (not useful when hessians are constant for some losses e.g.
      least squares). Also, there's a special case for the root which
      contains all the samples, leading to some possible optimizations.
      Overall all the implementations look the same, and are optimized for
      cache hit.

    Parameters
    ----------
    X_binned : ndarray of int, shape (n_samples, n_features)
        The binned input samples. Must be Fortran-aligned.
    n_bins : int
        The total number of bins, including the bin for missing values. Used
        to define the shape of the histograms.
    gradients : ndarray, shape (n_samples,)
        The gradients of each training sample. Those are the gradients of the
        loss w.r.t the predictions, evaluated at iteration i - 1.
    hessians : ndarray, shape (n_samples,)
        The hessians of each training sample. Those are the hessians of the
        loss w.r.t the predictions, evaluated at iteration i - 1.
    hessians_are_constant : bool
        Whether hessians are constant.
    """
    cdef public:
        const X_BINNED_DTYPE_C [::1, :] X_binned
        unsigned int n_features
        unsigned int n_bins
        G_H_DTYPE_C [::1] gradients
        G_H_DTYPE_C [::1] hessians
        G_H_DTYPE_C [::1] ordered_gradients
        G_H_DTYPE_C [::1] ordered_hessians
        uint8_t hessians_are_constant
        int n_threads

    def __init__(self, const X_BINNED_DTYPE_C [::1, :] X_binned,
                 unsigned int n_bins, G_H_DTYPE_C [::1] gradients,
                 G_H_DTYPE_C [::1] hessians,
                 uint8_t hessians_are_constant,
                 int n_threads):
        # [FUNCTION DOC]
        # Behavior: stores the inputs, sets n_features from X_binned, and initializes
        # ordered_gradients / ordered_hessians as copies of gradients / hessians (already
        # in root order, since the root contains all samples in index order).
        # Inputs: X_binned uint8 Fortran (n_samples, n_features); n_bins total bins
        # including the missing-values bin; gradients and hessians float32 (n_samples,);
        # hessians_are_constant flag; n_threads.
        # Outputs: none.
        # Exceptions: buffer errors on wrong dtype/layout.
        # Side effects: allocates two copies of length n_samples.

        self.X_binned = X_binned
        self.n_features = X_binned.shape[1]
        # Note: all histograms will have <n_bins> bins, but some of the
        # bins may be unused if a feature has a small number of unique values.
        self.n_bins = n_bins
        self.gradients = gradients
        self.hessians = hessians
        # for root node, gradients and hessians are already ordered
        self.ordered_gradients = gradients.copy()
        self.ordered_hessians = hessians.copy()
        self.hessians_are_constant = hessians_are_constant
        self.n_threads = n_threads

    def compute_histograms_brute(
        HistogramBuilder self,
        const unsigned int [::1] sample_indices,       # IN
        const unsigned int [:] allowed_features=None,  # IN
    ):
        # [FUNCTION DOC]
        # Behavior: allocates an uninitialized (n_features, n_bins) histogram array. With
        # the GIL released: if the node is not the root (sample count differs from total),
        # gathers gradients (and hessians unless constant) of sample_indices into the
        # ordered_* scratch arrays in parallel. Then, in parallel over the allowed features
        # (all features, or those listed in allowed_features), zeroes and fills each
        # feature's histogram via the single-feature method. Rows of features not allowed
        # are left uninitialized.
        # Inputs: sample_indices uint32 indices of the node's samples (a view of the
        # splitter partition); allowed_features optional uint32 feature indices.
        # Outputs: returns a numpy array of HISTOGRAM_DTYPE, shape (n_features, n_bins).
        # Exceptions: MemoryError on allocation.
        # Side effects: overwrites the first n_samples entries of ordered_gradients /
        # ordered_hessians; OpenMP parallelism; releases the GIL.
        """Compute the histograms of the node by scanning through all the data.

        For a given feature, the complexity is O(n_samples)

        Parameters
        ----------
        sample_indices : array of int, shape (n_samples_at_node,)
            The indices of the samples at the node to split.

        allowed_features : None or ndarray, dtype=np.uint32
            Indices of the features that are allowed by interaction constraints to be
            split.

        Returns
        -------
        histograms : ndarray of HISTOGRAM_DTYPE, shape (n_features, n_bins)
            The computed histograms of the current node.
        """
        cdef:
            int n_samples
            int feature_idx
            int f_idx
            int i
            # need local views to avoid python interactions
            uint8_t hessians_are_constant = self.hessians_are_constant
            int n_allowed_features = self.n_features
            G_H_DTYPE_C [::1] ordered_gradients = self.ordered_gradients
            G_H_DTYPE_C [::1] gradients = self.gradients
            G_H_DTYPE_C [::1] ordered_hessians = self.ordered_hessians
            G_H_DTYPE_C [::1] hessians = self.hessians
            # Histograms will be initialized to zero later within a prange
            hist_struct [:, ::1] histograms = np.empty(
                shape=(self.n_features, self.n_bins),
                dtype=HISTOGRAM_DTYPE
            )
            bint has_interaction_cst = allowed_features is not None
            int n_threads = self.n_threads

        if has_interaction_cst:
            n_allowed_features = allowed_features.shape[0]

        with nogil:
            n_samples = sample_indices.shape[0]

            # Populate ordered_gradients and ordered_hessians. (Already done
            # for root) Ordering the gradients and hessians helps to improve
            # cache hit.
            if sample_indices.shape[0] != gradients.shape[0]:
                if hessians_are_constant:
                    for i in prange(n_samples, schedule='static',
                                    num_threads=n_threads):
                        ordered_gradients[i] = gradients[sample_indices[i]]
                else:
                    for i in prange(n_samples, schedule='static',
                                    num_threads=n_threads):
                        ordered_gradients[i] = gradients[sample_indices[i]]
                        ordered_hessians[i] = hessians[sample_indices[i]]

            # Compute histogram of each feature
            for f_idx in prange(
                n_allowed_features, schedule='static', num_threads=n_threads
            ):
                if has_interaction_cst:
                    feature_idx = allowed_features[f_idx]
                else:
                    feature_idx = f_idx

                self._compute_histogram_brute_single_feature(
                    feature_idx, sample_indices, histograms
                )

        return histograms

    cdef void _compute_histogram_brute_single_feature(
            HistogramBuilder self,
            const int feature_idx,
            const unsigned int [::1] sample_indices,  # IN
            hist_struct [:, ::1] histograms) noexcept nogil:  # OUT
        # [FUNCTION DOC]
        # HELPER: per-feature worker of compute_histograms_brute.
        # Behavior: zeroes histograms[feature_idx, :], then dispatches to one of four
        # kernels: root vs non-root (root when the node holds all samples, so samples can
        # be read sequentially without indirection), each with or without hessian
        # accumulation (without when hessians are constant; sum_hessians then stays 0).
        # Inputs: feature_idx; sample_indices of the node; histograms output array.
        # Outputs: histograms[feature_idx, :] filled.
        # Exceptions: none (noexcept nogil).
        # Side effects: writes into histograms; called concurrently by different threads
        # on different feature rows.
        """Compute the histogram for a given feature."""

        cdef:
            unsigned int n_samples = sample_indices.shape[0]
            const X_BINNED_DTYPE_C [::1] X_binned = \
                self.X_binned[:, feature_idx]
            unsigned int root_node = X_binned.shape[0] == n_samples
            G_H_DTYPE_C [::1] ordered_gradients = \
                self.ordered_gradients[:n_samples]
            G_H_DTYPE_C [::1] ordered_hessians = \
                self.ordered_hessians[:n_samples]
            uint8_t hessians_are_constant = \
                self.hessians_are_constant

        # Set histograms to zero.
        memset(&histograms[feature_idx, 0], 0, self.n_bins * sizeof(hist_struct))

        if root_node:
            if hessians_are_constant:
                _build_histogram_root_no_hessian(feature_idx, X_binned,
                                                 ordered_gradients,
                                                 histograms)
            else:
                _build_histogram_root(feature_idx, X_binned,
                                      ordered_gradients, ordered_hessians,
                                      histograms)
        else:
            if hessians_are_constant:
                _build_histogram_no_hessian(feature_idx,
                                            sample_indices, X_binned,
                                            ordered_gradients, histograms)
            else:
                _build_histogram(feature_idx, sample_indices,
                                 X_binned, ordered_gradients,
                                 ordered_hessians, histograms)

    def compute_histograms_subtraction(
        HistogramBuilder self,
        hist_struct [:, ::1] parent_histograms,        # IN and OUT
        hist_struct [:, ::1] sibling_histograms,       # IN
        const unsigned int [:] allowed_features=None,  # IN
    ):
        # [FUNCTION DOC]
        # Behavior: in an OpenMP prange over the allowed features (GIL released),
        # subtracts sibling_histograms from parent_histograms row by row, so that the
        # parent array becomes the histogram of the other child. Cost O(n_bins) per feature.
        # Inputs: parent_histograms (n_features, n_bins) parent histograms, overwritten;
        # sibling_histograms same shape, the already-computed sibling; allowed_features
        # optional uint32 feature indices (other rows are left untouched).
        # Outputs: returns parent_histograms (same memory, now the child's histograms).
        # Exceptions: none expected.
        # Side effects: mutates parent_histograms in place; OpenMP parallelism.
        """Compute the histograms of the node using the subtraction trick.

        hist(parent) = hist(left_child) + hist(right_child)

        For a given feature, the complexity is O(n_bins). This is much more
        efficient than compute_histograms_brute, but it's only possible for one
        of the siblings.

        Parameters
        ----------
        parent_histograms : ndarray of HISTOGRAM_DTYPE, \
                shape (n_features, n_bins)
            The histograms of the parent.
        sibling_histograms : ndarray of HISTOGRAM_DTYPE, \
                shape (n_features, n_bins)
            The histograms of the sibling.
        allowed_features : None or ndarray, dtype=np.uint32
            Indices of the features that are allowed by interaction constraints to be
            split.

        Returns
        -------
        histograms : ndarray of HISTOGRAM_DTYPE, shape(n_features, n_bins)
            The computed histograms of the current node.
            We repurpose parent_histograms for this and don't need to allocate new
            memory.
        """

        cdef:
            int feature_idx
            int f_idx
            int n_allowed_features = self.n_features
            bint has_interaction_cst = allowed_features is not None
            int n_threads = self.n_threads

        if has_interaction_cst:
            n_allowed_features = allowed_features.shape[0]

        # Compute histogram of each feature
        for f_idx in prange(n_allowed_features, schedule='static', nogil=True,
                            num_threads=n_threads):
            if has_interaction_cst:
                feature_idx = allowed_features[f_idx]
            else:
                feature_idx = f_idx

            _subtract_histograms(
                feature_idx,
                self.n_bins,
                parent_histograms,
                sibling_histograms,
            )
        return parent_histograms


cpdef void _build_histogram_naive(
        const int feature_idx,
        unsigned int [:] sample_indices,  # IN
        X_BINNED_DTYPE_C [:] binned_feature,  # IN
        G_H_DTYPE_C [:] ordered_gradients,  # IN
        G_H_DTYPE_C [:] ordered_hessians,  # IN
        hist_struct [:, :] out) noexcept nogil:  # OUT
    # [FUNCTION DOC]
    # Behavior: reference implementation; for each i, adds ordered_gradients[i],
    # ordered_hessians[i] and 1 to the bin binned_feature[sample_indices[i]] of row
    # feature_idx of out. No unrolling; used in tests only.
    # Inputs: feature_idx; sample_indices uint32; binned_feature uint8 full column;
    # ordered_gradients / ordered_hessians float32 aligned with sample_indices; out
    # (n_features, n_bins) histograms (caller must zero it).
    # Outputs: out[feature_idx, :] accumulated.
    # Exceptions: none (noexcept nogil).
    # Side effects: mutates out.
    """Build histogram in a naive way, without optimizing for cache hit.

    Used in tests to compare with the optimized version."""
    cdef:
        unsigned int i
        unsigned int n_samples = sample_indices.shape[0]
        unsigned int sample_idx
        unsigned int bin_idx

    for i in range(n_samples):
        sample_idx = sample_indices[i]
        bin_idx = binned_feature[sample_idx]
        out[feature_idx, bin_idx].sum_gradients += ordered_gradients[i]
        out[feature_idx, bin_idx].sum_hessians += ordered_hessians[i]
        out[feature_idx, bin_idx].count += 1


cpdef void _subtract_histograms(
        const int feature_idx,
        unsigned int n_bins,
        hist_struct [:, ::1] hist_a,  # IN and OUT
        hist_struct [:, ::1] hist_b,  # IN
) noexcept nogil:  # OUT
    # [FUNCTION DOC]
    # Behavior: for each bin of row feature_idx, subtracts hist_b's sum_gradients,
    # sum_hessians and count from hist_a's (hist_a = hist_a - hist_b).
    # Inputs: feature_idx; n_bins; hist_a (modified) and hist_b histogram arrays.
    # Outputs: hist_a[feature_idx, :] updated in place.
    # Exceptions: none (noexcept nogil). Counts are unsigned; a larger sibling count
    # would wrap around (cannot happen for consistent parent/child histograms).
    # Side effects: mutates hist_a.
    """compute hist_a = hist_a - hist_b"""
    # Note that subtraction of large sums of floating point numbers, as we have here,
    # can exhibit catastrophic cancallation. This is in particular true for gradients
    # as they can be positive and negative, while hessians are non-negative.
    # Remember that gradients and hessians are originally computed in
    # G_H_DTYPE_C = float32 precision. Therefore, if sum_gradients and sum_hessians are
    # float64, we don't loose precision. But if we also used float32 for summation, we
    # would need to take care of floating point errors.
    #
    # Note that we could protect for negative hessians by setting:
    #     sum_hessians = max(0, sum_hessians)
    # But as we use float64 for summing float32, that's veeeery unlikely.
    cdef:
        unsigned int i = 0
    for i in range(n_bins):
        hist_a[feature_idx, i].sum_gradients -= hist_b[feature_idx, i].sum_gradients
        hist_a[feature_idx, i].sum_hessians -= hist_b[feature_idx, i].sum_hessians
        hist_a[feature_idx, i].count -= hist_b[feature_idx, i].count


cpdef void _build_histogram(
        const int feature_idx,
        const unsigned int [::1] sample_indices,  # IN
        const X_BINNED_DTYPE_C [::1] binned_feature,  # IN
        const G_H_DTYPE_C [::1] ordered_gradients,  # IN
        const G_H_DTYPE_C [::1] ordered_hessians,  # IN
        hist_struct [:, ::1] out) noexcept nogil:  # OUT
    # [FUNCTION DOC]
    # HELPER: histogram kernel used by HistogramBuilder (also exposed for tests).
    # Behavior: for each sample position i, adds the ordered_gradients[i] gradient, hessian and a count of 1
    # to bin binned_feature[sample_indices[i]] of row feature_idx of out. The main loop is unrolled by 4 and a
    # remainder loop handles the last n % 4 samples.
    # Inputs: feature_idx; sample_indices uint32 node sample indices; binned_feature uint8 full column;
    # ordered_gradients / ordered_hessians float32 aligned with the sample positions; out histograms,
    # row feature_idx assumed zeroed by the caller.
    # Outputs: out[feature_idx, :] accumulated in place.
    # Exceptions: none (noexcept nogil, no bounds checks).
    # Side effects: mutates out.
    """Return histogram for a given feature."""
    cdef:
        unsigned int i = 0
        unsigned int n_node_samples = sample_indices.shape[0]
        unsigned int unrolled_upper = (n_node_samples // 4) * 4

        unsigned int bin_0
        unsigned int bin_1
        unsigned int bin_2
        unsigned int bin_3
        unsigned int bin_idx

    for i in range(0, unrolled_upper, 4):
        bin_0 = binned_feature[sample_indices[i]]
        bin_1 = binned_feature[sample_indices[i + 1]]
        bin_2 = binned_feature[sample_indices[i + 2]]
        bin_3 = binned_feature[sample_indices[i + 3]]

        out[feature_idx, bin_0].sum_gradients += ordered_gradients[i]
        out[feature_idx, bin_1].sum_gradients += ordered_gradients[i + 1]
        out[feature_idx, bin_2].sum_gradients += ordered_gradients[i + 2]
        out[feature_idx, bin_3].sum_gradients += ordered_gradients[i + 3]

        out[feature_idx, bin_0].sum_hessians += ordered_hessians[i]
        out[feature_idx, bin_1].sum_hessians += ordered_hessians[i + 1]
        out[feature_idx, bin_2].sum_hessians += ordered_hessians[i + 2]
        out[feature_idx, bin_3].sum_hessians += ordered_hessians[i + 3]

        out[feature_idx, bin_0].count += 1
        out[feature_idx, bin_1].count += 1
        out[feature_idx, bin_2].count += 1
        out[feature_idx, bin_3].count += 1

    for i in range(unrolled_upper, n_node_samples):
        bin_idx = binned_feature[sample_indices[i]]
        out[feature_idx, bin_idx].sum_gradients += ordered_gradients[i]
        out[feature_idx, bin_idx].sum_hessians += ordered_hessians[i]
        out[feature_idx, bin_idx].count += 1


cpdef void _build_histogram_no_hessian(
        const int feature_idx,
        const unsigned int [::1] sample_indices,  # IN
        const X_BINNED_DTYPE_C [::1] binned_feature,  # IN
        const G_H_DTYPE_C [::1] ordered_gradients,  # IN
        hist_struct [:, ::1] out) noexcept nogil:  # OUT
    # [FUNCTION DOC]
    # HELPER: histogram kernel used by HistogramBuilder (also exposed for tests).
    # Behavior: for each sample position i, adds the ordered_gradients[i] gradient and a count of 1 (sum_hessians untouched; used when hessians are constant)
    # to bin binned_feature[sample_indices[i]] of row feature_idx of out. The main loop is unrolled by 4 and a
    # remainder loop handles the last n % 4 samples.
    # Inputs: feature_idx; sample_indices uint32 node sample indices; binned_feature uint8 full column;
    # ordered_gradients float32 aligned with the sample positions; out histograms,
    # row feature_idx assumed zeroed by the caller.
    # Outputs: out[feature_idx, :] accumulated in place.
    # Exceptions: none (noexcept nogil, no bounds checks).
    # Side effects: mutates out.
    """Return histogram for a given feature, not updating hessians.

    Used when the hessians of the loss are constant (typically LS loss).
    """

    cdef:
        unsigned int i = 0
        unsigned int n_node_samples = sample_indices.shape[0]
        unsigned int unrolled_upper = (n_node_samples // 4) * 4

        unsigned int bin_0
        unsigned int bin_1
        unsigned int bin_2
        unsigned int bin_3
        unsigned int bin_idx

    for i in range(0, unrolled_upper, 4):
        bin_0 = binned_feature[sample_indices[i]]
        bin_1 = binned_feature[sample_indices[i + 1]]
        bin_2 = binned_feature[sample_indices[i + 2]]
        bin_3 = binned_feature[sample_indices[i + 3]]

        out[feature_idx, bin_0].sum_gradients += ordered_gradients[i]
        out[feature_idx, bin_1].sum_gradients += ordered_gradients[i + 1]
        out[feature_idx, bin_2].sum_gradients += ordered_gradients[i + 2]
        out[feature_idx, bin_3].sum_gradients += ordered_gradients[i + 3]

        out[feature_idx, bin_0].count += 1
        out[feature_idx, bin_1].count += 1
        out[feature_idx, bin_2].count += 1
        out[feature_idx, bin_3].count += 1

    for i in range(unrolled_upper, n_node_samples):
        bin_idx = binned_feature[sample_indices[i]]
        out[feature_idx, bin_idx].sum_gradients += ordered_gradients[i]
        out[feature_idx, bin_idx].count += 1


cpdef void _build_histogram_root(
        const int feature_idx,
        const X_BINNED_DTYPE_C [::1] binned_feature,  # IN
        const G_H_DTYPE_C [::1] all_gradients,  # IN
        const G_H_DTYPE_C [::1] all_hessians,  # IN
        hist_struct [:, ::1] out) noexcept nogil:  # OUT
    # [FUNCTION DOC]
    # HELPER: histogram kernel used by HistogramBuilder (also exposed for tests).
    # Behavior: for each sample position i, adds the all_gradients[i] gradient, hessian and a count of 1
    # to bin binned_feature[i] (root: samples are all training samples in index order, no indirection) of row feature_idx of out. The main loop is unrolled by 4 and a
    # remainder loop handles the last n % 4 samples.
    # Inputs: feature_idx; binned_feature uint8 full column;
    # all_gradients / all_hessians float32 aligned with the sample positions; out histograms,
    # row feature_idx assumed zeroed by the caller.
    # Outputs: out[feature_idx, :] accumulated in place.
    # Exceptions: none (noexcept nogil, no bounds checks).
    # Side effects: mutates out.
    """Compute histogram of the root node.

    Unlike other nodes, the root node has to find the split among *all* the
    samples from the training set. binned_feature and all_gradients /
    all_hessians already have a consistent ordering.
    """

    cdef:
        unsigned int i = 0
        unsigned int n_samples = binned_feature.shape[0]
        unsigned int unrolled_upper = (n_samples // 4) * 4

        unsigned int bin_0
        unsigned int bin_1
        unsigned int bin_2
        unsigned int bin_3
        unsigned int bin_idx

    for i in range(0, unrolled_upper, 4):

        bin_0 = binned_feature[i]
        bin_1 = binned_feature[i + 1]
        bin_2 = binned_feature[i + 2]
        bin_3 = binned_feature[i + 3]

        out[feature_idx, bin_0].sum_gradients += all_gradients[i]
        out[feature_idx, bin_1].sum_gradients += all_gradients[i + 1]
        out[feature_idx, bin_2].sum_gradients += all_gradients[i + 2]
        out[feature_idx, bin_3].sum_gradients += all_gradients[i + 3]

        out[feature_idx, bin_0].sum_hessians += all_hessians[i]
        out[feature_idx, bin_1].sum_hessians += all_hessians[i + 1]
        out[feature_idx, bin_2].sum_hessians += all_hessians[i + 2]
        out[feature_idx, bin_3].sum_hessians += all_hessians[i + 3]

        out[feature_idx, bin_0].count += 1
        out[feature_idx, bin_1].count += 1
        out[feature_idx, bin_2].count += 1
        out[feature_idx, bin_3].count += 1

    for i in range(unrolled_upper, n_samples):
        bin_idx = binned_feature[i]
        out[feature_idx, bin_idx].sum_gradients += all_gradients[i]
        out[feature_idx, bin_idx].sum_hessians += all_hessians[i]
        out[feature_idx, bin_idx].count += 1


cpdef void _build_histogram_root_no_hessian(
        const int feature_idx,
        const X_BINNED_DTYPE_C [::1] binned_feature,  # IN
        const G_H_DTYPE_C [::1] all_gradients,  # IN
        hist_struct [:, ::1] out) noexcept nogil:  # OUT
    # [FUNCTION DOC]
    # HELPER: histogram kernel used by HistogramBuilder (also exposed for tests).
    # Behavior: for each sample position i, adds the all_gradients[i] gradient and a count of 1 (sum_hessians untouched; used when hessians are constant)
    # to bin binned_feature[i] (root: samples are all training samples in index order, no indirection) of row feature_idx of out. The main loop is unrolled by 4 and a
    # remainder loop handles the last n % 4 samples.
    # Inputs: feature_idx; binned_feature uint8 full column;
    # all_gradients float32 aligned with the sample positions; out histograms,
    # row feature_idx assumed zeroed by the caller.
    # Outputs: out[feature_idx, :] accumulated in place.
    # Exceptions: none (noexcept nogil, no bounds checks).
    # Side effects: mutates out.
    """Compute histogram of the root node, not updating hessians.

    Used when the hessians of the loss are constant (typically LS loss).
    """

    cdef:
        unsigned int i = 0
        unsigned int n_samples = binned_feature.shape[0]
        unsigned int unrolled_upper = (n_samples // 4) * 4

        unsigned int bin_0
        unsigned int bin_1
        unsigned int bin_2
        unsigned int bin_3
        unsigned int bin_idx

    for i in range(0, unrolled_upper, 4):
        bin_0 = binned_feature[i]
        bin_1 = binned_feature[i + 1]
        bin_2 = binned_feature[i + 2]
        bin_3 = binned_feature[i + 3]

        out[feature_idx, bin_0].sum_gradients += all_gradients[i]
        out[feature_idx, bin_1].sum_gradients += all_gradients[i + 1]
        out[feature_idx, bin_2].sum_gradients += all_gradients[i + 2]
        out[feature_idx, bin_3].sum_gradients += all_gradients[i + 3]

        out[feature_idx, bin_0].count += 1
        out[feature_idx, bin_1].count += 1
        out[feature_idx, bin_2].count += 1
        out[feature_idx, bin_3].count += 1

    for i in range(unrolled_upper, n_samples):
        bin_idx = binned_feature[i]
        out[feature_idx, bin_idx].sum_gradients += all_gradients[i]
        out[feature_idx, bin_idx].count += 1
