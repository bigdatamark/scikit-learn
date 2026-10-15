# [MODULE DOC]
# Partitioners: the data-layout-specific half of split finding. A Splitter
# (in _splitter.pyx) owns the array samples of row indices; the samples of a
# tree node are the contiguous slice samples[start:end]. For each candidate
# feature, the partitioner copies that feature's values into the parallel
# buffer feature_values and either sorts the slice by value (best splitter)
# or computes the min/max (random splitter); it then enumerates candidate
# split positions or partitions by a threshold, and finally reorders the
# slice so that left-child samples come first. DensePartitioner reads a 2-D
# float32 array and supports NaN as missing values; SparsePartitioner reads a
# CSC matrix, handles only explicit non-zeros and treats the rest as zeros
# (no missing-value support). Module-level helpers implement sparse non-zero
# extraction (index map or binary search, chosen by an estimated cost), swaps
# that keep the row-to-position map consistent, and an order-preserving
# rotation of two adjacent array slices. All hot methods are nogil and cannot
# raise.
"""Partition samples in the construction of a tree.

This module contains the algorithms for moving sample indices to
the left and right child node given a split determined by the
splitting algorithm in `_splitter.pyx`.

Partitioning is done in a way that is efficient for both dense data,
and sparse data stored in a Compressed Sparse Column (CSC) format.
"""
# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

from cython cimport final
from libc.math cimport INFINITY, isnan, log2
from libc.stdlib cimport qsort
from libc.string cimport memcpy, memmove

import numpy as np
cimport numpy as cnp
cnp.import_array()
from scipy.sparse import issparse

from sklearn.tree._splitter cimport SplitRecord
from sklearn.utils._sorting cimport simultaneous_sort

# Constant to switch between algorithm non zero value extract algorithm
# in SparsePartitioner
cdef float32_t EXTRACT_NNZ_SWITCH = 0.1


@final
cdef class DensePartitioner:
    # [CLASS DOC]
    # Partitioner for dense float32 X with optional NaN missing values. Holds
    # views on the splitter's samples and feature_values buffers (mutated in
    # place) plus a byte swap_buffer large enough for a full copy of either
    # array. State per node: start, end, and n_missing for the last feature
    # examined. Declared final so calls are resolved statically.
    """Partitioner specialized for dense data.

    Note that this partitioner is agnostic to the splitting strategy (best vs. random).
    """
    def __init__(
        self,
        const float32_t[:, :] X,
        intp_t[::1] samples,
        float32_t[::1] feature_values,
        const uint8_t[::1] missing_values_in_feature_mask,
    ):
        # [FUNCTION DOC]
        # Stores X, samples, feature_values and the missing-value mask (may be None)
        # and allocates swap_buffer of samples.size times the larger item size.
        # Inputs: X, 2-D float32 view; samples, intp buffer shared with the splitter;
        # feature_values, float32 buffer of the same length; missing mask, uint8 per
        # feature or None. Outputs: none. Exceptions: MemoryError from numpy.
        # Side effects: allocation.
        self.X = X
        self.samples = samples
        self.feature_values = feature_values
        self.missing_values_in_feature_mask = missing_values_in_feature_mask
        buffer_size = samples.size * max(samples.itemsize, feature_values.itemsize)
        self.swap_buffer = np.empty(buffer_size, dtype=np.uint8)
        # TODO: As optimization we could make `swap_array_slices` always pick the smallest side
        # to get copied in the buffer, which would allow to use a buffer twice smaller.

    cdef inline void init_node_split(self, intp_t start, intp_t end) noexcept nogil:
        # [FUNCTION DOC]
        # Sets start and end for the node about to be split and resets n_missing.
        # Inputs: start, end positions in samples. Outputs: none.
        # Exceptions: none (nogil). Side effects: mutates fields.
        """Initialize splitter at the beginning of node_split."""
        self.start = start
        self.end = end
        self.n_missing = 0

    cdef inline void sort_samples_and_feature_values(
        self, intp_t current_feature
    ) noexcept nogil:
        # [FUNCTION DOC]
        # Behavior: if the feature may contain NaNs, a two-pointer pass over
        # samples[start:end] swaps samples whose value is NaN to the right end
        # (counting them) while copying non-missing values into feature_values at
        # the left; otherwise simply copies X[samples[i], feature] for every i. Then
        # sorts the non-missing prefix of feature_values together with samples
        # (introsort with three-way partitioning) and stores the NaN count in
        # n_missing. Values for the trailing NaN positions in feature_values are not
        # written (they hold stale data); callers rely on n_missing instead.
        # Inputs: current_feature, column index. Outputs: none.
        # Exceptions: none (nogil, noexcept).
        # Side effects: reorders samples[start:end], writes feature_values, sets
        # n_missing.
        """Simultaneously sort based on the feature_values.

        Missing values are stored at the end of feature_values.
        The number of missing values observed in feature_values is stored
        in self.n_missing.
        """
        cdef:
            intp_t i, current_end
            float32_t[::1] feature_values = self.feature_values
            const float32_t[:, :] X = self.X
            intp_t[::1] samples = self.samples
            intp_t n_missing = 0
            const uint8_t[::1] missing_values_in_feature_mask = self.missing_values_in_feature_mask

        # Sort samples along that feature; by copying the values into an array and
        # sorting the array in a manner which utilizes the cache more effectively.
        if missing_values_in_feature_mask is not None and missing_values_in_feature_mask[current_feature]:
            i, current_end = self.start, self.end - 1
            # Missing values are placed at the end and do not participate in the sorting.
            while i <= current_end:
                # Finds the right-most value that is not missing so that
                # it can be swapped with missing values at its left.
                if isnan(X[samples[current_end], current_feature]):
                    n_missing += 1
                    current_end -= 1
                    continue

                # X[samples[current_end], current_feature] is a non-missing value
                if isnan(X[samples[i], current_feature]):
                    samples[i], samples[current_end] = samples[current_end], samples[i]
                    n_missing += 1
                    current_end -= 1

                feature_values[i] = X[samples[i], current_feature]
                i += 1
        else:
            # When there are no missing values, we only need to copy the data into
            # feature_values
            for i in range(self.start, self.end):
                feature_values[i] = X[samples[i], current_feature]

        simultaneous_sort(
            &feature_values[self.start],
            &samples[self.start],
            self.end - self.start - n_missing,
            use_three_way_partition=True,
        )
        self.n_missing = n_missing

    cdef void shift_missing_to_the_left(self) noexcept nogil:
        # [FUNCTION DOC]
        # Behavior: assuming the n_missing NaN entries sit at the right end of the
        # node slice, rotates both samples and feature_values so the NaN block comes
        # first and the non-missing block (order preserved) follows; uses
        # swap_array_slices with swap_buffer. Called before the second search pass
        # of the best splitter (missing values going left).
        # Inputs: none (uses start, end, n_missing). Outputs: none.
        # Exceptions: none (nogil). Side effects: reorders samples and
        # feature_values in place.
        """Moves missing values from the right to the left.

        All missing values are expected to be grouped at the right hand side of the
        [self.start:self.end] slices of the self.samples and self.feature_values arrays
        before calling this method. This will be the case for nominal use as
        the splitter calls sort_samples_and_feature_values() first:
        that method groups missing values on the right and sets self.n_missing.
        shift_missing_to_the_left() is then called only for the second split search
        pass when evaluating missing_go_to_left=True.

        Non-missing values are correspondingly moved from the left to the right while
        preserving their inner ordering.
        """
        cdef intp_t n_non_missing = self.end - self.start - self.n_missing
        swap_array_slices(self.samples, self.start, self.end, n_non_missing, self.swap_buffer)
        swap_array_slices(self.feature_values, self.start, self.end, n_non_missing, self.swap_buffer)

    cdef inline void find_min_max(
        self,
        intp_t current_feature,
        float32_t* min_feature_value_out,
        float32_t* max_feature_value_out,
    ) noexcept nogil:
        # [FUNCTION DOC]
        # Behavior: single pass over the node copying X values of the feature into
        # feature_values in current sample order, counting NaNs, and tracking min and
        # max over non-NaN values. If all values are NaN, min stays +inf and max -inf.
        # Samples are not reordered.
        # Inputs: current_feature; two output pointers.
        # Outputs: min and max written to the pointers; n_missing set.
        # Exceptions: none (nogil). Side effects: writes feature_values.
        # KNOWN ISSUE: the docstring says missing values are stored at the end of
        # feature_values, but this method does not move them; NaNs stay at their
        # original positions (partition_samples handles them with an explicit NaN
        # test, so behavior is still correct).
        """Find the minimum and maximum value for current_feature.

        Missing values are stored at the end of feature_values. The number of missing
        values observed in feature_values is stored in self.n_missing.
        """
        cdef:
            intp_t p
            float32_t current_feature_value
            intp_t[::1] samples = self.samples
            float32_t min_feature_value = INFINITY
            float32_t max_feature_value = -INFINITY
            float32_t[::1] feature_values = self.feature_values
            intp_t n_missing = 0
            bint seen_non_missing = False

        for p in range(self.start, self.end):
            current_feature_value = self.X[samples[p], current_feature]
            feature_values[p] = current_feature_value

            if isnan(current_feature_value):
                n_missing += 1
            elif not seen_non_missing:
                min_feature_value = current_feature_value
                max_feature_value = current_feature_value
                seen_non_missing = True
            elif current_feature_value < min_feature_value:
                min_feature_value = current_feature_value
            elif current_feature_value > max_feature_value:
                max_feature_value = current_feature_value

        min_feature_value_out[0] = min_feature_value
        max_feature_value_out[0] = max_feature_value
        self.n_missing = n_missing

    cdef inline void next_p(
        self,
        intp_t* p_prev,
        intp_t* p,
        bint missing_go_to_left,
    ) noexcept nogil:
        # [FUNCTION DOC]
        # Behavior: iterator step for the best-split scan. With missing values going
        # right, the scan stops at end - n_missing; when p already equals that bound
        # it jumps p and p_prev to end (terminating the caller's loop) after the
        # caller has evaluated the "all missing on the right" split. With missing
        # going left, the first step jumps p past the NaN block at the left. In the
        # general case, increments p and keeps incrementing while the value at p is
        # within FEATURE_THRESHOLD of the value at p - 1 (so splits never separate
        # near-equal values), then sets p_prev = p - 1.
        # Inputs: p_prev, p, in/out positions (p starts at start); missing_go_to_left.
        # Outputs: updated p and p_prev. Exceptions: none (nogil).
        # Side effects: none beyond the two pointers.
        """
        Compute the next p_prev and p for iterating over feature values.

        This method is used inside the best-split search function pass which starts
        by setting p = start at the beginning of each search pass and calls
        this method repeatedly with the same missing_go_to_left as for that pass.
        The expected layout of self.feature_values[start:end] is:
        - first pass (missing_go_to_left=False): after
          sort_samples_and_feature_values(), non-missing values are sorted and
          missing values are grouped at the right;
        - second pass (missing_go_to_left=True): after
          shift_missing_to_the_left(), missing values are grouped at the left.

        Given that layout, this method advances p to the next valid split
        position while skipping ties up to FEATURE_THRESHOLD:
        - if missing_go_to_left: iterate p in [start + n_missing + 1, end)
        - otherwise: iterate p in [start, end - n_missing].
          The special case p == end - n_missing corresponds to "all non-missing
          values on the left and all missing values on the right". The next
          call then sets p to end to terminate the search loop.
        """
        cdef intp_t end_non_missing = (
            self.end if missing_go_to_left
            else self.end - self.n_missing)

        if p[0] == end_non_missing and not missing_go_to_left:
            # skip the missing values up to the end
            # (which will end the for loop in the best split function)
            p[0] = self.end
            p_prev[0] = self.end
        else:
            if missing_go_to_left and p[0] == self.start:
                # skip the missing values up to the first non-missing value:
                p[0] = self.start + self.n_missing
            p[0] += 1
            while (
                p[0] < end_non_missing and
                self.feature_values[p[0]] <= self.feature_values[p[0] - 1] + FEATURE_THRESHOLD
            ):
                p[0] += 1
            p_prev[0] = p[0] - 1

    cdef inline intp_t partition_samples(
        self,
        float64_t threshold,
        bint missing_go_to_left
    ) noexcept nogil:
        # [FUNCTION DOC]
        # Behavior: in-place two-pointer partition of the node slice using the
        # values already in feature_values (from find_min_max): a sample goes left if
        # its value is <= threshold, or if it is NaN and missing_go_to_left; right
        # samples are swapped to the end (both feature_values and samples swapped).
        # Inputs: threshold; missing_go_to_left. Outputs: returns the first position
        # of the right child (number of left samples plus start).
        # Exceptions: none (nogil). Side effects: reorders samples and feature_values.
        """Partition self.samples and self.feature_values
        on current self.feature_values for a given threshold.

        Used while searching splits through random threshold sampling.
        """
        cdef:
            # Local invariance: start <= partition_start <= partition_end <= end
            intp_t partition_start = self.start
            intp_t partition_end = self.end
            intp_t* samples = &self.samples[0]
            float32_t* feature_values = &self.feature_values[0]
            bint go_to_left

        while partition_start < partition_end:
            go_to_left = (
                missing_go_to_left if isnan(feature_values[partition_start])
                else feature_values[partition_start] <= threshold
            )
            if go_to_left:
                partition_start += 1
            else:
                partition_end -= 1
                swap(feature_values, samples, partition_start, partition_end)

        return partition_end

    cdef inline void partition_samples_final(
        self,
        const SplitRecord* best_split,
    ) noexcept nogil:
        # [FUNCTION DOC]
        # Behavior: two-pointer partition of samples[start:end] reading values
        # directly from X for best_split.feature: left if value <= threshold, or if
        # NaN and best_split.missing_go_to_left. feature_values is not updated.
        # Inputs: pointer to the chosen SplitRecord. Outputs: none.
        # Exceptions: none (nogil). Side effects: reorders samples in place.
        """Partition self.samples according to the split described by best_split.

        If missing values are present, this method partitions them accordingly
        to the split strategy.
        """
        cdef:
            # Local invariance: start <= partition_start <= partition_end <= end
            intp_t partition_start = self.start
            intp_t partition_end = self.end
            intp_t* samples = &self.samples[0]
            float64_t best_threshold = best_split[0].threshold
            intp_t best_feature = best_split[0].feature
            bint best_missing_go_to_left = best_split[0].missing_go_to_left
            float32_t current_value
            bint go_to_left

        while partition_start < partition_end:
            current_value = self.X[samples[partition_start], best_feature]
            go_to_left = (
                best_missing_go_to_left if isnan(current_value)
                else current_value <= best_threshold
            )
            if go_to_left:
                partition_start += 1
            else:
                partition_end -= 1
                samples[partition_start], samples[partition_end] = (
                    samples[partition_end], samples[partition_start])


@final
cdef class SparsePartitioner:
    # [CLASS DOC]
    # Partitioner for CSC sparse float32 X. Only explicit non-zeros are read;
    # implicit zeros are represented by the gap between end_negative and
    # start_positive in the node slice. index_to_samples maps each row of X to its
    # current position in samples (or -1 for rows not used, e.g. zero weight) and
    # is kept consistent by sparse_swap. sorted_samples caches a sorted copy of
    # the node's rows for the binary-search extraction. Missing values are not
    # supported. Declared final.
    """Partitioner specialized for sparse CSC data.

    Note that this partitioner is agnostic to the splitting strategy (best vs. random).
    """
    def __init__(
        self,
        object X,
        intp_t[::1] samples,
        intp_t n_samples,
        float32_t[::1] feature_values,
        const uint8_t[::1] missing_values_in_feature_mask,
    ):
        # [FUNCTION DOC]
        # Behavior: validates that X is a scipy CSC matrix, keeps views on its data,
        # indices and indptr, allocates index_to_samples (n_rows, filled with -1) and
        # sorted_samples (n_samples), and records the initial position of each used
        # row.
        # Inputs: X, CSC matrix with float32 data and int32 indices/indptr; samples,
        # shared intp buffer whose first n_samples entries are used rows;
        # n_samples, number of used rows; feature_values buffer; missing mask
        # (stored but unused). Outputs: none.
        # Exceptions: ValueError if X is not CSC; buffer dtype errors if X has other
        # dtypes; MemoryError from numpy. Side effects: allocation.
        if not (issparse(X) and X.format == "csc"):
            raise ValueError("X should be in csc format")

        self.samples = samples
        self.feature_values = feature_values

        # Initialize X
        cdef intp_t n_total_samples = X.shape[0]

        self.X_data = X.data
        self.X_indices = X.indices
        self.X_indptr = X.indptr
        self.n_total_samples = n_total_samples

        # Initialize auxiliary array used to perform split
        self.index_to_samples = np.full(n_total_samples, fill_value=-1, dtype=np.intp)
        self.sorted_samples = np.empty(n_samples, dtype=np.intp)

        cdef intp_t p
        for p in range(n_samples):
            self.index_to_samples[samples[p]] = p

        self.missing_values_in_feature_mask = missing_values_in_feature_mask

    cdef inline void init_node_split(self, intp_t start, intp_t end) noexcept nogil:
        # [FUNCTION DOC]
        # Sets start and end, marks sorted_samples as stale and resets n_missing.
        # Inputs: start, end. Outputs: none. Exceptions: none (nogil).
        # Side effects: mutates fields.
        """Initialize splitter at the beginning of node_split."""
        self.start = start
        self.end = end
        self.is_samples_sorted = 0
        self.n_missing = 0

    cdef inline void sort_samples_and_feature_values(
        self,
        intp_t current_feature
    ) noexcept nogil:
        # [FUNCTION DOC]
        # Behavior: calls extract_nnz to place negative values in
        # [start, end_negative) and positive values in [start_positive, end); sorts
        # each block with its samples; refreshes index_to_samples for both blocks;
        # then, if zeros exist, writes a 0 into feature_values just before
        # start_positive (and, if more than one zero, also at end_negative) and
        # moves the boundaries to include them, so that next_p can step from the last
        # negative to a zero, and from a zero to the first positive, while skipping
        # the interior of the zero block. Sets n_missing to 0.
        # Inputs: current_feature. Outputs: none. Exceptions: none (nogil).
        # Side effects: reorders samples, writes feature_values and
        # index_to_samples, updates end_negative / start_positive.
        """Simultaneously sort based on the feature_values."""
        cdef:
            float32_t[::1] feature_values = self.feature_values
            intp_t[::1] index_to_samples = self.index_to_samples
            intp_t[::1] samples = self.samples

        self.extract_nnz(current_feature)
        # Sort the positive and negative parts of `feature_values`
        simultaneous_sort(
            &feature_values[self.start],
            &samples[self.start],
            self.end_negative - self.start,
            use_three_way_partition=True,
        )
        if self.start_positive < self.end:
            simultaneous_sort(
                &feature_values[self.start_positive],
                &samples[self.start_positive],
                self.end - self.start_positive,
                use_three_way_partition=True,
            )

        # Update index_to_samples to take into account the sort
        for p in range(self.start, self.end_negative):
            index_to_samples[samples[p]] = p
        for p in range(self.start_positive, self.end):
            index_to_samples[samples[p]] = p

        # Add one or two zeros in feature_values, if there is any
        if self.end_negative < self.start_positive:
            self.start_positive -= 1
            feature_values[self.start_positive] = 0.

            if self.end_negative != self.start_positive:
                feature_values[self.end_negative] = 0.
                self.end_negative += 1

        # XXX: When sparse supports missing values, this should be set to the
        # number of missing values for current_feature
        self.n_missing = 0

    cdef void shift_missing_to_the_left(self) noexcept nogil:
        # [FUNCTION DOC]
        # No-op: sparse data cannot contain missing values here. No inputs,
        # outputs, exceptions or side effects.
        pass  # Missing values are not supported for sparse data.

    cdef inline void find_min_max(
        self,
        intp_t current_feature,
        float32_t* min_feature_value_out,
        float32_t* max_feature_value_out,
    ) noexcept nogil:
        # [FUNCTION DOC]
        # Behavior: calls extract_nnz, then starts min/max at 0 if the node has any
        # implicit zero (else at the first extracted value) and scans the negative
        # and positive blocks of feature_values to update them.
        # Inputs: current_feature; two output pointers. Outputs: min and max written.
        # Exceptions: none (nogil). Side effects: those of extract_nnz (reorders
        # samples, writes feature_values and index_to_samples).
        """Find the minimum and maximum value for current_feature."""
        cdef:
            intp_t p
            float32_t current_feature_value, min_feature_value, max_feature_value
            float32_t[::1] feature_values = self.feature_values

        self.extract_nnz(current_feature)

        if self.end_negative != self.start_positive:
            # There is a zero
            min_feature_value = 0
            max_feature_value = 0
        else:
            min_feature_value = feature_values[self.start]
            max_feature_value = min_feature_value

        # Find min, max in feature_values[start:end_negative]
        for p in range(self.start, self.end_negative):
            current_feature_value = feature_values[p]

            if current_feature_value < min_feature_value:
                min_feature_value = current_feature_value
            elif current_feature_value > max_feature_value:
                max_feature_value = current_feature_value

        # Update min, max given feature_values[start_positive:end]
        for p in range(self.start_positive, self.end):
            current_feature_value = feature_values[p]

            if current_feature_value < min_feature_value:
                min_feature_value = current_feature_value
            elif current_feature_value > max_feature_value:
                max_feature_value = current_feature_value

        min_feature_value_out[0] = min_feature_value
        max_feature_value_out[0] = max_feature_value

    cdef inline void next_p(
        self,
        intp_t* p_prev,
        intp_t* p,
        bint missing_go_to_left,
    ) noexcept nogil:
        # [FUNCTION DOC]
        # Behavior: from position p, the next position is p + 1 unless that equals
        # end_negative, in which case it is start_positive (skipping the zero block).
        # Keeps advancing while the next value is within FEATURE_THRESHOLD of the
        # current one and the next position is below end; finally p_prev = last
        # position of the tie run and p = the next position (may equal end).
        # missing_go_to_left is ignored.
        # Inputs: p_prev, p in/out. Outputs: updated pointers.
        # Exceptions: none (nogil). Side effects: none.
        """Compute the next p_prev and p for iterating over feature values.

        The missing_go_to_left argument is ignored for sparse data because
        sparse partitioning does not currently support missing values.
        """
        cdef intp_t p_next

        if p[0] + 1 != self.end_negative:
            p_next = p[0] + 1
        else:
            p_next = self.start_positive

        while (p_next < self.end and
                self.feature_values[p_next] <= self.feature_values[p[0]] + FEATURE_THRESHOLD):
            p[0] = p_next
            if p[0] + 1 != self.end_negative:
                p_next = p[0] + 1
            else:
                p_next = self.start_positive

        p_prev[0] = p[0]
        p[0] = p_next

    cdef inline intp_t partition_samples(
        self,
        float64_t current_threshold,
        bint missing_go_to_left
    ) noexcept nogil:
        # [FUNCTION DOC]
        # Delegates to _partition with the threshold; missing_go_to_left is ignored.
        # Outputs: start of the right child. Exceptions: none (nogil).
        # Side effects: reorders samples, feature_values and index_to_samples.
        """Partition samples for feature_values at the current_threshold."""
        return self._partition(current_threshold)

    cdef inline void partition_samples_final(
        self,
        const SplitRecord* best_split,
    ) noexcept nogil:
        # [FUNCTION DOC]
        # Behavior: re-extracts the non-zeros of best_split.feature (so the slice is
        # in negative / zero / positive layout) and partitions with its threshold.
        # Inputs: pointer to the chosen SplitRecord. Outputs: none.
        # Exceptions: none (nogil). Side effects: reorders samples, feature_values,
        # index_to_samples.
        """Partition samples for X according to the split described by best_split."""
        self.extract_nnz(best_split[0].feature)
        self._partition(best_split[0].threshold)

    cdef inline intp_t _partition(self, float64_t threshold) noexcept nogil:
        # [FUNCTION DOC]
        # HELPER: threshold partition used by partition_samples and
        # partition_samples_final.
        # Behavior: requires the layout produced by extract_nnz. A negative threshold
        # partitions only the negative block (zeros and positives are already right),
        # a positive threshold partitions only the positive block, and a zero
        # threshold returns start_positive directly. Partitioning is the usual
        # two-pointer swap on feature_values and samples (using sparse_swap).
        # Inputs: threshold. Outputs: first position of the right child.
        # Exceptions: none (nogil). Side effects: reorders samples, feature_values,
        # index_to_samples.
        # KNOWN ISSUE: the docstring lists samples[end_negative:start_positive] twice;
        # the positive block is actually samples[start_positive:end].
        """
        Partition samples[start:end] based on threshold.
        Assume extract_nnz was called beforehand, and partitioned samples in:
        - samples[start:end_negative] -> < 0
        - samples[end_negative:start_positive] -> zeros
        - samples[end_negative:start_positive] -> > 0
        """
        cdef:
            intp_t p, partition_end
            intp_t[::1] index_to_samples = self.index_to_samples
            float32_t[::1] feature_values = self.feature_values
            intp_t[::1] samples = self.samples

        if threshold < 0.:
            p = self.start
            partition_end = self.end_negative
        elif threshold > 0.:
            p = self.start_positive
            partition_end = self.end
        else:
            # If threshold is 0, extract_nnz already did the necessary partitioning
            return self.start_positive

        while p < partition_end:
            if feature_values[p] <= threshold:
                p += 1

            else:
                partition_end -= 1

                feature_values[p], feature_values[partition_end] = (
                    feature_values[partition_end], feature_values[p]
                )
                sparse_swap(index_to_samples, samples, p, partition_end)

        return partition_end

    cdef inline void extract_nnz(self, intp_t feature) noexcept nogil:
        # [FUNCTION DOC]
        # Behavior: chooses between the binary-search extraction and the
        # index_to_samples extraction using estimated costs: binary search if
        # (sort cost if sorted_samples is stale) n log2 n + n log2 nnz is below
        # EXTRACT_NNZ_SWITCH (0.1) times nnz, where n is the node size and nnz the
        # number of stored entries of the feature column. Either way, afterwards
        # samples[start:end_negative] hold rows with negative values,
        # samples[start_positive:end] rows with positive values (values copied into
        # feature_values at the same positions) and the rows in between are zeros.
        # Inputs: feature, column index. Outputs: end_negative, start_positive.
        # Exceptions: none (nogil). Side effects: reorders samples, updates
        # index_to_samples, feature_values, and possibly sorted_samples /
        # is_samples_sorted.
        """Extract and partition values for a given feature.

        The extracted values are partitioned between negative values
        feature_values[start:end_negative[0]] and positive values
        feature_values[start_positive[0]:end].
        The samples and index_to_samples are modified according to this
        partition.

        The extraction corresponds to the intersection between the arrays
        X_indices[indptr_start:indptr_end] and samples[start:end].
        This is done efficiently using either an index_to_samples based approach
        or binary search based approach.

        Parameters
        ----------
        feature : intp_t,
            Index of the feature we want to extract non zero value.
        """
        cdef intp_t[::1] samples = self.samples
        cdef float32_t[::1] feature_values = self.feature_values
        cdef intp_t indptr_start = self.X_indptr[feature]
        cdef intp_t indptr_end = self.X_indptr[feature + 1]
        cdef intp_t n_indices = <intp_t>(indptr_end - indptr_start)
        cdef intp_t n_samples = self.end - self.start
        cdef intp_t[::1] index_to_samples = self.index_to_samples
        cdef intp_t[::1] sorted_samples = self.sorted_samples
        cdef const int32_t[::1] X_indices = self.X_indices
        cdef const float32_t[::1] X_data = self.X_data

        # Use binary search if n_samples * log(n_indices) <
        # n_indices and index_to_samples approach otherwise.
        # O(n_samples * log(n_indices)) is the running time of binary
        # search and O(n_indices) is the running time of index_to_samples
        # approach.
        if ((1 - self.is_samples_sorted) * n_samples * log2(n_samples) +
                n_samples * log2(n_indices) < EXTRACT_NNZ_SWITCH * n_indices):
            extract_nnz_binary_search(X_indices, X_data,
                                      indptr_start, indptr_end,
                                      samples, self.start, self.end,
                                      index_to_samples,
                                      feature_values,
                                      &self.end_negative, &self.start_positive,
                                      sorted_samples, &self.is_samples_sorted)

        # Using an index to samples  technique to extract non zero values
        # index_to_samples is a mapping from X_indices to samples
        else:
            extract_nnz_index_to_samples(X_indices, X_data,
                                         indptr_start, indptr_end,
                                         samples, self.start, self.end,
                                         index_to_samples,
                                         feature_values,
                                         &self.end_negative, &self.start_positive)


cdef int compare_SIZE_t(const void* a, const void* b) noexcept nogil:
    # [FUNCTION DOC]
    # HELPER: qsort comparator for intp values used in extract_nnz_binary_search.
    # Returns the difference of the two pointed-to values cast to int.
    # Exceptions: none (nogil). Side effects: none.
    # KNOWN ISSUE: casting the intp difference to int can overflow (wrong sign)
    # when the two row indices differ by more than 2^31 - 1.
    """Comparison function for sort.

    This must return an `int` as it is used by stdlib's qsort, which expects
    an `int` return value.
    """
    return <int>((<intp_t*>a)[0] - (<intp_t*>b)[0])


cdef inline void binary_search(const int32_t[::1] sorted_array,
                               int32_t start, int32_t end,
                               intp_t value, intp_t* index,
                               int32_t* new_start) noexcept nogil:
    # [FUNCTION DOC]
    # HELPER: binary search in a sorted int32 array slice, used for sparse
    # extraction.
    # Behavior: searches value in sorted_array[start:end]; stores the found
    # position (or -1) in index and the next search start in new_start (pivot + 1
    # when found, else the final lower bound), letting the caller shrink the
    # range for successive increasing queries.
    # Inputs: sorted_array, start, end, value. Outputs: index, new_start.
    # Exceptions: none (nogil). Side effects: none beyond the outputs.
    """Return the index of value in the sorted array.

    If not found, return -1. new_start is the last pivot + 1
    """
    cdef int32_t pivot
    index[0] = -1
    while start < end:
        pivot = start + (end - start) / 2

        if sorted_array[pivot] == value:
            index[0] = pivot
            start = pivot + 1
            break

        if sorted_array[pivot] < value:
            start = pivot + 1
        else:
            end = pivot
    new_start[0] = start


cdef inline void extract_nnz_index_to_samples(const int32_t[::1] X_indices,
                                              const float32_t[::1] X_data,
                                              int32_t indptr_start,
                                              int32_t indptr_end,
                                              intp_t[::1] samples,
                                              intp_t start,
                                              intp_t end,
                                              intp_t[::1] index_to_samples,
                                              float32_t[::1] feature_values,
                                              intp_t* end_negative,
                                              intp_t* start_positive) noexcept nogil:
    # [FUNCTION DOC]
    # HELPER: O(nnz) extraction of a feature's non-zeros for the node.
    # Behavior: scans the column's stored entries; for each row whose current
    # position (index_to_samples) is in [start, end), positive values are packed
    # from the right end downward and negative values from the left end upward,
    # copying the value into feature_values and swapping the sample into place
    # with sparse_swap. Explicit zeros are treated like implicit ones.
    # Inputs: CSC column arrays and range indptr_start..indptr_end, samples,
    # start, end, index_to_samples, feature_values.
    # Outputs: end_negative and start_positive pointers.
    # Exceptions: none (nogil). Side effects: reorders samples, updates
    # index_to_samples and feature_values.
    """Extract and partition values for a feature using index_to_samples.

    Complexity is O(indptr_end - indptr_start).
    """
    cdef int32_t k
    cdef intp_t index
    cdef intp_t end_negative_ = start
    cdef intp_t start_positive_ = end

    for k in range(indptr_start, indptr_end):
        if start <= index_to_samples[X_indices[k]] < end:
            if X_data[k] > 0:
                start_positive_ -= 1
                feature_values[start_positive_] = X_data[k]
                index = index_to_samples[X_indices[k]]
                sparse_swap(index_to_samples, samples, index, start_positive_)

            elif X_data[k] < 0:
                feature_values[end_negative_] = X_data[k]
                index = index_to_samples[X_indices[k]]
                sparse_swap(index_to_samples, samples, index, end_negative_)
                end_negative_ += 1

    # Returned values
    end_negative[0] = end_negative_
    start_positive[0] = start_positive_


cdef inline void extract_nnz_binary_search(const int32_t[::1] X_indices,
                                           const float32_t[::1] X_data,
                                           int32_t indptr_start,
                                           int32_t indptr_end,
                                           intp_t[::1] samples,
                                           intp_t start,
                                           intp_t end,
                                           intp_t[::1] index_to_samples,
                                           float32_t[::1] feature_values,
                                           intp_t* end_negative,
                                           intp_t* start_positive,
                                           intp_t[::1] sorted_samples,
                                           bint* is_samples_sorted) noexcept nogil:
    # [FUNCTION DOC]
    # HELPER: O(n log nnz) extraction of a feature's non-zeros for the node.
    # Behavior: if needed, copies samples[start:end] into sorted_samples and
    # sorts it (qsort), marking it valid for the rest of the node. Trims the
    # column's index range to the span of node rows, then for each node row in
    # increasing order binary-searches it in the remaining index range; found
    # non-zeros are placed into the negative/positive blocks exactly as in
    # extract_nnz_index_to_samples.
    # Inputs: as extract_nnz_index_to_samples plus sorted_samples and the
    # is_samples_sorted flag pointer.
    # Outputs: end_negative, start_positive; is_samples_sorted set to true.
    # Exceptions: none (nogil). Side effects: reorders samples, updates
    # index_to_samples, feature_values, sorted_samples.
    """Extract and partition values for a given feature using binary search.

    If n_samples = end - start and n_indices = indptr_end - indptr_start,
    the complexity is

        O((1 - is_samples_sorted[0]) * n_samples * log(n_samples) +
          n_samples * log(n_indices)).
    """
    cdef intp_t n_samples

    if not is_samples_sorted[0]:
        n_samples = end - start
        memcpy(&sorted_samples[start], &samples[start],
               n_samples * sizeof(intp_t))
        qsort(&sorted_samples[start], n_samples, sizeof(intp_t),
              compare_SIZE_t)
        is_samples_sorted[0] = 1

    while (indptr_start < indptr_end and
           sorted_samples[start] > X_indices[indptr_start]):
        indptr_start += 1

    while (indptr_start < indptr_end and
           sorted_samples[end - 1] < X_indices[indptr_end - 1]):
        indptr_end -= 1

    cdef intp_t p = start
    cdef intp_t index
    cdef intp_t k
    cdef intp_t end_negative_ = start
    cdef intp_t start_positive_ = end

    while (p < end and indptr_start < indptr_end):
        # Find index of sorted_samples[p] in X_indices
        binary_search(X_indices, indptr_start, indptr_end,
                      sorted_samples[p], &k, &indptr_start)

        if k != -1:
            # If k != -1, we have found a non zero value

            if X_data[k] > 0:
                start_positive_ -= 1
                feature_values[start_positive_] = X_data[k]
                index = index_to_samples[X_indices[k]]
                sparse_swap(index_to_samples, samples, index, start_positive_)

            elif X_data[k] < 0:
                feature_values[end_negative_] = X_data[k]
                index = index_to_samples[X_indices[k]]
                sparse_swap(index_to_samples, samples, index, end_negative_)
                end_negative_ += 1
        p += 1

    # Returned values
    end_negative[0] = end_negative_
    start_positive[0] = start_positive_


cdef inline void sparse_swap(intp_t[::1] index_to_samples, intp_t[::1] samples,
                             intp_t pos_1, intp_t pos_2) noexcept nogil:
    # [FUNCTION DOC]
    # HELPER: swaps two positions of samples and updates index_to_samples for
    # both moved rows so the row-to-position map stays valid.
    # Inputs: index_to_samples, samples, pos_1, pos_2. Outputs: none.
    # Exceptions: none (nogil). Side effects: in-place mutation.
    """Swap sample pos_1 and pos_2 preserving sparse invariant."""
    samples[pos_1], samples[pos_2] = samples[pos_2], samples[pos_1]
    index_to_samples[samples[pos_1]] = pos_1
    index_to_samples[samples[pos_2]] = pos_2


cdef inline void swap(float32_t* feature_values, intp_t* samples,
                      intp_t i, intp_t j) noexcept nogil:
    # [FUNCTION DOC]
    # HELPER: swaps positions i and j in both feature_values and samples (dense
    # partitioning). Exceptions: none (nogil). Side effects: in-place mutation.
    feature_values[i], feature_values[j] = feature_values[j], feature_values[i]
    samples[i], samples[j] = samples[j], samples[i]


cdef void swap_array_slices(
    array_data_type[::1] array, intp_t start, intp_t end, intp_t n,
    char[::1] buffer
) noexcept nogil:
    # [FUNCTION DOC]
    # HELPER: order-preserving rotation used by shift_missing_to_the_left.
    # Behavior: exchanges the slices array[start:start+n] and array[start+n:end]
    # keeping order inside each: copies the second slice to buffer, moves the
    # first slice right with an overlap-safe move, then copies the buffer to the
    # front. Instantiated for intp and float32 (fused type). No-op if start >= end.
    # Inputs: contiguous array, start, end, n (length of the first slice), byte
    # buffer of at least (end - start - n) items.
    # Outputs: none. Exceptions: none (nogil). Side effects: mutates array and
    # buffer.
    """Swaps the order of the slices array[start:start + n] and array[start + n:end].

    Preserves the order within the slices. Works for any itemsize.
    """
    if start >= end:
        return
    cdef size_t itemsize = sizeof(array[0])
    cdef intp_t n_rev = end - start - n
    cdef char* arr = <char*> &array[0]
    cdef char* buf = &buffer[0]
    # Copy array[start + n : end] to temporary buffer
    memcpy(buf, arr + (start + n) * itemsize, n_rev * itemsize)
    # Move array[start : start + n] to array[start + n_rev : end]
    # `memmove` is needed as the dest & source regions overlap
    memmove(arr + (start + n_rev) * itemsize, arr + start * itemsize, n * itemsize)
    # array[start : start + n_rev] = buffer[:n_rev]
    memcpy(arr + start * itemsize, buf, n_rev * itemsize)


def _py_swap_array_slices(cnp.ndarray array, intp_t start, intp_t end, intp_t n):
    # [FUNCTION DOC]
    # HELPER: Python test wrapper for swap_array_slices.
    # Behavior: allocates a byte buffer the size of the array and dispatches to
    # the intp or float32 specialization by dtype.
    # Inputs: contiguous 1-D ndarray, start, end, n. Outputs: none (array
    # modified in place).
    # Exceptions: ValueError for other dtypes; buffer errors if not contiguous.
    """
    Python wrapper for swap_array_slices for testing.
    `array` must be contiguous.
    """
    buffer = np.empty(array.size * array.dtype.itemsize, dtype=np.uint8)
    # Dispatch to the appropriate specialized version based on dtype
    if array.dtype == np.intp:
        swap_array_slices[intp_t](array, start, end, n, buffer)
    elif array.dtype == np.float32:
        swap_array_slices[float32_t](array, start, end, n, buffer)
    else:
        raise ValueError(f"Unsupported dtype: {array.dtype}. Expected np.intp or np.float32")
